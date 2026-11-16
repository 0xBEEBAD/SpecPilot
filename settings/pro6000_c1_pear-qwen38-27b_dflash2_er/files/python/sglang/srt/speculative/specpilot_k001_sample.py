"""SPECPILOT_K001_SAMPLE: lossless temperature > 0 sampling on the k001 two-level verify ("sample, then match", with
position-keyed Gumbel noise shared by the Q pass and the exact pass).

Rule. Every verify row i of request b (a tree node at depth d_i whose root path is the token prefix it conditions on)
draws ONE token from the target distribution built exactly as the stock DFlash2 chain verify builds it
(temperature, top-k, top-p; ``tree_target_probs``):

    x[b, i] = argmin_v  E(seed_b, pos, v) / p[b, i, v]          pos = prefix_len_b + d_i  (absolute position)

``E`` are i.i.d. Exp(1) variables produced by a counter-based generator keyed by (request seed, absolute position,
vocab id) -- the exponential-race form of the Gumbel-max sampler (argmax log p + G, G = -log E). The unchanged greedy
tree accept then runs with ``pred = x`` (child token == parent's draw -> descend; the last node's draw is the bonus).

Exactness. The noise of position n is a fixed function of (seed, n). A committed token at position n is always
argmin_v E(seed, n, v) / p(v | committed prefix): the row that produces it lies on the committed path, so its root
path IS the committed prefix. By induction the whole output equals the output of the plain autoregressive
exponential-race sampler run with the same noise field E(seed, ., .), token for token, whatever the trees were. The
proposers (drafter trees, the Q pass, the chain it builds) only decide how many of those tokens one step commits;
they may read the noise freely. AR exponential-race sampling with i.i.d. noise is an exact sampler, hence so is this.

Coupling. The Q pass (quantized weights, drafter-vocabulary head) draws with the SAME key, so its draw equals the exact
pass's draw whenever the two distributions agree up to the noise margin: the chain Q builds is then the chain the
exact pass will sample. SPECPILOT_K001_SAMPLE_COUPLE=0 salts the Q pass's key (independent noise; same output
distribution -- in fact the same output tokens for the same seed --, shorter accepted chains).

Knobs (read once at import; unset = parent behaviour: non-greedy requests are rejected in tree mode):
  SPECPILOT_K001_SAMPLE          1 = enable
  SPECPILOT_K001_SAMPLE_COUPLE   1 (default) = Q pass shares the exact pass's noise; 0 = independent Q noise
  SPECPILOT_K001_SAMPLE_DRAFT    1 = also couple the drafter trees: the DDTree edge scores of slot s become
                                 score / T + scale * G(seed, pos, candidate id), the noise the verify row of that
                                 position uses (in-graph kernel on the selector lattice). Default 0: measured no gain
                                 on PEAR (docs/K001_SAMPLE.md section 4), and with 0 the draft graph is k001_sm120's.
  SPECPILOT_K001_SAMPLE_DRAFT_SCALE  multiplier of the drafter's noise term (default 1.0)
  SPECPILOT_K001_SAMPLE_DRAFT_TOPP   1 = also push drafter candidates outside the drafter's top-p nucleus back (default 0)
  SPECPILOT_K001_SAMPLE_IMPL     triton (default on CUDA) | torch (reference implementation, bit-compatible keys)
  SPECPILOT_K001_SAMPLE_STATS    n > 0: log chain / accept statistics every n sampled cycles (host sync; debug only)

Greedy requests never enter this module.
"""
from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, "") or default).strip()


_ENABLED = _env("SPECPILOT_K001_SAMPLE", "0") not in ("", "0")
_COUPLE = _env("SPECPILOT_K001_SAMPLE_COUPLE", "1") not in ("", "0")
_IMPL = _env("SPECPILOT_K001_SAMPLE_IMPL", "triton").lower()
_STATS = int(_env("SPECPILOT_K001_SAMPLE_STATS", "0") or 0)
_DRAFT = _ENABLED and _env("SPECPILOT_K001_SAMPLE_DRAFT", "0") not in ("", "0")
_DRAFT_SCALE = float(_env("SPECPILOT_K001_SAMPLE_DRAFT_SCALE", "1.0"))
# edge-score penalty of drafter candidates outside the drafter's own top-p nucleus (SPECPILOT_K001_SAMPLE_DRAFT_TOPP=1;
# off by default: the drafter is over-confident on PEAR, its nucleus cuts candidates the target can still draw)
_DRAFT_PEN = -30.0 if _env("SPECPILOT_K001_SAMPLE_DRAFT_TOPP", "0") not in ("", "0") else 0.0

SALT_EXACT = 0
SALT_Q_INDEPENDENT = 1

_M64 = (1 << 64) - 1


def _s64(x: int) -> int:
    """Python int (mod 2^64) -> the signed 64-bit integer with the same bits."""
    x &= _M64
    return x - (1 << 64) if x >= (1 << 63) else x


# splitmix64 constants (signed views of the usual unsigned ones)
_GAMMA = _s64(0x9E3779B97F4A7C15)
_C1 = _s64(0xBF58476D1CE4E5B9)
_C2 = _s64(0x94D049BB133111EB)
_KPOS = _s64(0xD6E8FEB86659FD93)
_KSALT = _s64(0xA0761D6478BD642F)
_U53 = 1.0 / float(1 << 53)


def enabled() -> bool:
    return _ENABLED


def coupled() -> bool:
    return _COUPLE


def q_salt() -> int:
    return SALT_EXACT if _COUPLE else SALT_Q_INDEPENDENT


# ----------------------------------------------------------------------------------------------------------------
# request checks, seeds, target distribution
# ----------------------------------------------------------------------------------------------------------------
def check_batch(reqs, sampling_info) -> None:
    """Raise for batches the sampled tree accept does not support (host-side request metadata only, no GPU sync)."""
    if sampling_info is None or sampling_info.is_all_greedy:
        return
    if any(r.sampling_params.top_k <= 1 for r in reqs):  # same rule as SamplingBatchInfo.is_all_greedy
        raise NotImplementedError(
            "SPECPILOT_K001_SAMPLE: a batch mixing greedy and sampled requests is not supported (run batch 1)."
        )
    if getattr(sampling_info, "has_custom_logit_processor", False):
        raise NotImplementedError("SPECPILOT_K001_SAMPLE: custom logit processors are not supported.")
    pen = getattr(sampling_info, "penalizer_orchestrator", None)
    if pen is not None and pen.is_required:
        raise NotImplementedError(
            "SPECPILOT_K001_SAMPLE: frequency/presence/repetition penalties are not supported (verify rows do not "
            "see their in-tree prefix)."
        )
    if getattr(sampling_info, "need_min_p_sampling", False):
        raise NotImplementedError("SPECPILOT_K001_SAMPLE: min_p is not supported.")
    if getattr(sampling_info, "logit_bias", None) is not None:
        raise NotImplementedError("SPECPILOT_K001_SAMPLE: logit_bias is not supported (the Q head is a vocab subset).")


def request_seeds(reqs, device) -> torch.Tensor:
    """int64 [bs]: the request's noise seed. ``sampling_params.sampling_seed`` (OpenAI ``seed``) when given, else 63
    random bits drawn once per request from the OS. Cached on the request (one tiny H2D copy per request)."""
    out = []
    for r in reqs:
        t = getattr(r, "_k001s_seed_t", None)
        if t is None:
            s = getattr(r.sampling_params, "sampling_seed", None)
            s = secrets.randbits(63) if s is None else _s64(int(s))
            t = torch.tensor([s], dtype=torch.int64, device=device)
            r._k001s_seed_t = t
        out.append(t)
    return out[0] if len(out) == 1 else torch.cat(out)


def tree_target_probs(next_token_logits: torch.Tensor, sampling_info, *, bs: int, n_nodes: int, draft_input) -> torch.Tensor:
    """[bs * N, V] fp32 logits -> [bs, N, V] target probabilities, exactly as the DFlash2 chain verify builds them
    (temperature only: ``SoftmaxTemp``; with top-k / top-p: ``build_dflash_verify_target_probs``, top-k first, then
    top-p renormalisation). Rows whose logits are -inf outside a vocab subset (Q head) get probability 0 there."""
    from sglang.kernels.ops.speculative.dspark.dspark_accept import SoftmaxTemp
    from sglang.srt.speculative.dflash_utils import build_dflash_verify_target_probs

    if not sampling_info.need_top_k_sampling and not sampling_info.need_top_p_sampling:
        return SoftmaxTemp.execute(
            logits=next_token_logits, temperatures=sampling_info.temperatures, rows_per_request=n_nodes
        ).view(bs, n_nodes, -1)
    return build_dflash_verify_target_probs(
        next_token_logits=next_token_logits,
        sampling_info=sampling_info,
        draft_token_num=n_nodes,
        bs=bs,
        max_top_k=getattr(draft_input, "max_top_k", None),
        uniform_top_k_value=getattr(draft_input, "uniform_top_k_value", None),
    )


# ----------------------------------------------------------------------------------------------------------------
# keyed exponential race: torch reference
# ----------------------------------------------------------------------------------------------------------------
def _lsr(x: torch.Tensor, s: int) -> torch.Tensor:
    return (x >> s) & ((1 << (64 - s)) - 1)


def _mix(x: torch.Tensor) -> torch.Tensor:
    """splitmix64 finalizer on int64 tensors (wrapping arithmetic, logical shifts)."""
    x = (x ^ _lsr(x, 30)) * _C1
    x = (x ^ _lsr(x, 27)) * _C2
    return x ^ _lsr(x, 31)


def row_keys(seeds: torch.Tensor, positions: torch.Tensor, salt: int = 0) -> torch.Tensor:
    """int64 [bs, N] row key = mix(mix(seed + salt * KSALT) ^ (pos * KPOS)): one stream per (seed, salt, position)."""
    s = _mix(seeds.to(torch.int64).view(-1, 1) + int(salt) * _KSALT)
    return _mix(s ^ (positions.to(torch.int64) * _KPOS))


def exp_noise(keys: torch.Tensor, vocab: int) -> torch.Tensor:
    """float64 [..., vocab] Exp(1) noise of the rows with these keys (reference; the kernel computes the same)."""
    v = torch.arange(1, vocab + 1, dtype=torch.int64, device=keys.device)
    x = _mix(keys.unsqueeze(-1) + v * _GAMMA)
    u = (_lsr(x, 11).to(torch.float64) + 0.5) * _U53
    return -torch.log(u)


def keyed_sample_torch(probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor, salt: int = 0) -> torch.Tensor:
    """[bs, N, V] probs, [bs] seeds, [bs, N] absolute positions -> [bs, N] int64 draws (reference implementation)."""
    bs, n, v = probs.shape
    e = exp_noise(row_keys(seeds, positions, salt), v)  # [bs, n, v]
    p = probs.to(torch.float64)
    r = torch.where(p > 0, e / p, torch.full_like(e, float("inf")))
    return r.argmin(dim=-1).to(torch.int64)


# ----------------------------------------------------------------------------------------------------------------
# keyed exponential race: Triton kernel (one program per (row, vocab chunk); partial minima reduced by torch)
# ----------------------------------------------------------------------------------------------------------------
_KERNEL = None
_CHUNK = 1024


def _kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _mix_tl(x, C1, C2):
        x = (x ^ (x.to(tl.uint64) >> 30).to(tl.int64)) * C1
        x = (x ^ (x.to(tl.uint64) >> 27).to(tl.int64)) * C2
        return x ^ (x.to(tl.uint64) >> 31).to(tl.int64)

    @triton.jit
    def _k001s_race_kernel(
        probs_ptr,  # [R, V] fp32
        seeds_ptr,  # [bs] int64
        pos_ptr,  # [R] int64
        val_out,  # [R, NCH] fp64
        idx_out,  # [R, NCH] int64
        V,
        NCH,
        N_PER_REQ,
        SALT_ADD,  # int64: salt * KSALT
        GAMMA,
        C1,
        C2,
        KPOS,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        ch = tl.program_id(1).to(tl.int64)
        seed = tl.load(seeds_ptr + row // N_PER_REQ)
        pos = tl.load(pos_ptr + row)
        key = _mix_tl(_mix_tl(seed + SALT_ADD, C1, C2) ^ (pos * KPOS), C1, C2)
        offs = ch * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        m = offs < V
        p = tl.load(probs_ptr + row * V + offs, mask=m, other=0.0)
        big = float("inf")
        if tl.max(p, 0) > 0:  # chunks without support (most of the vocabulary after top-p) skip the noise
            x = _mix_tl(key + (offs + 1) * GAMMA, C1, C2)
            u = ((x.to(tl.uint64) >> 11).to(tl.float64) + 0.5) * 1.1102230246251565e-16  # 2^-53
            e = -tl.log(u)
            r = tl.where(p > 0, e / p.to(tl.float64), big)
            best = tl.min(r, 0)
            bi = tl.min(tl.where(r == best, offs, V), 0)
            tl.store(val_out + row * NCH + ch, best)
            tl.store(idx_out + row * NCH + ch, bi)
        else:
            tl.store(val_out + row * NCH + ch, big + 0.0 * seed.to(tl.float64))
            tl.store(idx_out + row * NCH + ch, ch * BLOCK)

    _KERNEL = _k001s_race_kernel
    return _KERNEL


_SCRATCH: dict = {}


def keyed_sample_triton(probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor, salt: int = 0) -> torch.Tensor:
    bs, n, v = probs.shape
    rows = bs * n
    nch = (v + _CHUNK - 1) // _CHUNK
    key = (rows, nch, probs.device)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty((rows, nch), dtype=torch.float64, device=probs.device),
            torch.empty((rows, nch), dtype=torch.int64, device=probs.device),
        )
        _SCRATCH[key] = buf
    val, idx = buf
    p = probs if probs.dtype == torch.float32 else probs.float()
    p = p.contiguous()
    pos = positions.to(torch.int64).reshape(-1).contiguous()
    _kernel()[(rows, nch)](
        p, seeds.to(torch.int64).contiguous(), pos, val, idx, v, nch, n,
        _s64(int(salt) * _KSALT), _GAMMA, _C1, _C2, _KPOS, BLOCK=_CHUNK,
    )
    best = val.argmin(dim=1, keepdim=True)
    return torch.gather(idx, 1, best).view(bs, n)


def keyed_sample(probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor, salt: int = 0) -> torch.Tensor:
    if probs.is_cuda and _IMPL != "torch":
        return keyed_sample_triton(probs, seeds, positions, salt)
    return keyed_sample_torch(probs, seeds, positions, salt)


def sample_nodes(
    next_token_logits: torch.Tensor,
    sampling_info,
    *,
    bs: int,
    n_nodes: int,
    draft_input,
    reqs,
    positions: torch.Tensor,
    q_pass: bool,
) -> torch.Tensor:
    """The per-node draws of one verify pass. positions [bs, N] = prefix_len + node depth."""
    probs = tree_target_probs(next_token_logits, sampling_info, bs=bs, n_nodes=n_nodes, draft_input=draft_input)
    seeds = request_seeds(reqs, probs.device)
    return keyed_sample(probs, seeds, positions, q_salt() if q_pass else SALT_EXACT)


# ----------------------------------------------------------------------------------------------------------------
# drafter coupling (SPECPILOT_K001_SAMPLE_DRAFT): proposals only -- nothing here can change the output distribution
# ----------------------------------------------------------------------------------------------------------------
def draft_enabled() -> bool:
    return _DRAFT


def draft_noise_torch(cand_ids, unary, seeds, pos0, inv_t, scale, top_p):
    """Reference of the kernel's additive term: [bs, S, K] float32 = scale * (G(seed, pos0 + s, cand) + nucleus penalty)."""
    bs, S, K = cand_ids.shape
    pos = pos0.view(bs, 1) + torch.arange(S, device=cand_ids.device).view(1, S)
    keys = row_keys(seeds, pos, SALT_EXACT)  # [bs, S]
    x = _mix(keys.unsqueeze(-1) + (cand_ids.to(torch.int64) + 1) * _GAMMA)
    u = (_lsr(x, 11).to(torch.float64) + 0.5) * _U53
    g = -torch.log(-torch.log(u))
    p = torch.softmax(unary.float() * inv_t.view(bs, 1, 1), dim=-1)
    ahead = (p.unsqueeze(-2) > p.unsqueeze(-1)) | (
        (p.unsqueeze(-2) == p.unsqueeze(-1))
        & (torch.arange(K, device=p.device).view(1, 1, 1, K) < torch.arange(K, device=p.device).view(1, 1, K, 1))
    )
    before = (ahead.to(p.dtype) * p.unsqueeze(-2)).sum(-1)
    pen = torch.where(before >= top_p.view(bs, 1, 1), _DRAFT_PEN, 0.0)
    return (scale.view(bs, 1, 1) * (g.float() + pen)).float()


_DRAFT_KERNEL = None


def _draft_kernel():
    global _DRAFT_KERNEL
    if _DRAFT_KERNEL is not None:
        return _DRAFT_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _mix_tl(x, C1, C2):
        x = (x ^ (x.to(tl.uint64) >> 30).to(tl.int64)) * C1
        x = (x ^ (x.to(tl.uint64) >> 27).to(tl.int64)) * C2
        return x ^ (x.to(tl.uint64) >> 31).to(tl.int64)

    @triton.jit
    def _k001s_draft_kernel(
        scores_ptr,  # [bs, S, K, K] float32, updated in place: score * inv_t + scale * (G + pen)[child]
        cand_ptr,  # [bs, S, K] int64
        unary_ptr,  # [bs, S, K] float32
        seeds_ptr,  # [bs] int64
        pos_ptr,  # [bs] int64: key position of slot 0 (= position key of the anchor's verify row)
        invt_ptr,  # [bs] float32
        scale_ptr,  # [bs] float32 (0 = greedy request: scores unchanged bit for bit)
        topp_ptr,  # [bs] float32
        GAMMA,
        C1,
        C2,
        KPOS,
        PEN,
        S: tl.constexpr,
        K: tl.constexpr,
    ):
        r = tl.program_id(0).to(tl.int64)
        b = r // S
        s = r % S
        kk = tl.arange(0, K)
        seed = tl.load(seeds_ptr + b)
        pos = tl.load(pos_ptr + b) + s
        inv_t = tl.load(invt_ptr + b)
        scale = tl.load(scale_ptr + b)
        top_p = tl.load(topp_ptr + b)
        cand = tl.load(cand_ptr + r * K + kk)
        un = tl.load(unary_ptr + r * K + kk).to(tl.float32) * inv_t
        m = tl.max(un, 0)
        e = tl.exp(un - m)
        p = e / tl.sum(e, 0)
        pj = p[None, :]
        pc = p[:, None]
        ahead = (pj > pc) | ((pj == pc) & (kk[None, :] < kk[:, None]))
        before = tl.sum(tl.where(ahead, pj, 0.0), 1)
        pen = tl.where(before >= top_p, PEN, 0.0)
        key = _mix_tl(_mix_tl(seed, C1, C2) ^ (pos * KPOS), C1, C2)
        x = _mix_tl(key + (cand + 1) * GAMMA, C1, C2)
        u = ((x.to(tl.uint64) >> 11).to(tl.float64) + 0.5) * 1.1102230246251565e-16
        g = (-tl.log(-tl.log(u))).to(tl.float32)
        add = scale * (g + pen)
        offs = kk[:, None] * K + kk[None, :]
        sc = tl.load(scores_ptr + r * (K * K) + offs).to(tl.float32)
        tl.store(scores_ptr + r * (K * K) + offs, sc * inv_t + add[None, :])

    _DRAFT_KERNEL = _k001s_draft_kernel
    return _DRAFT_KERNEL


def alloc_draft_buffers(max_bs: int, device) -> dict:
    return {
        "seed": torch.zeros((max_bs,), dtype=torch.int64, device=device),
        "pos": torch.zeros((max_bs,), dtype=torch.int64, device=device),
        "inv_t": torch.ones((max_bs,), dtype=torch.float32, device=device),
        "scale": torch.zeros((max_bs,), dtype=torch.float32, device=device),
        "top_p": torch.ones((max_bs,), dtype=torch.float32, device=device),
        "live": [False],
    }


def stage_draft(buf: dict, *, bs: int, reqs, sampling_info, draft_prefix: torch.Tensor) -> None:
    """Host-side refresh of the static buffers the in-graph kernel reads (before the draft graph replay; GPU copies
    only, no sync). Greedy batches leave scale = 0 (written only when the previous step was a sampled one)."""
    if sampling_info is None or sampling_info.is_all_greedy:
        if buf["live"][0]:
            buf["scale"].zero_()
            buf["inv_t"].fill_(1.0)
            buf["live"][0] = False
        return
    buf["seed"][:bs].copy_(request_seeds(reqs, buf["seed"].device))
    buf["pos"][:bs].copy_(draft_prefix.view(-1))
    torch.reciprocal(sampling_info.temperatures.view(-1)[:bs].to(torch.float32).clamp_min(1e-5), out=buf["inv_t"][:bs])
    buf["scale"][:bs].fill_(_DRAFT_SCALE)
    buf["top_p"][:bs].copy_(sampling_info.top_ps.view(-1)[:bs])
    buf["live"][0] = True


def couple_draft_scores(scores: torch.Tensor, cand_ids: torch.Tensor, unary: torch.Tensor, buf: dict) -> torch.Tensor:
    """In-graph: [bs, S, K, K] lattice scores -> coupled scores (float32, contiguous; a no-op in value when scale = 0)."""
    bs, S, K = cand_ids.shape
    scores = scores.float().contiguous()
    cand = cand_ids.contiguous()
    if cand.dtype != torch.int64:
        cand = cand.long()
    un = unary.float().contiguous()
    _draft_kernel()[(bs * S,)](
        scores, cand, un, buf["seed"], buf["pos"], buf["inv_t"], buf["scale"], buf["top_p"],
        _GAMMA, _C1, _C2, _KPOS, _DRAFT_PEN, S=S, K=K,
    )
    return scores


# ----------------------------------------------------------------------------------------------------------------
# debug statistics (SPECPILOT_K001_SAMPLE_STATS=n): chain length proposed by Q vs what the exact pass accepted
# ----------------------------------------------------------------------------------------------------------------
def stats_enabled() -> bool:
    return _STATS > 0


class _Stats:
    """Separate accumulators for greedy and sampled cycles (GPU sums; one host sync every _STATS cycles)."""

    def __init__(self):
        self.acc = {}
        self.cycles = {}

    def record(self, chain_len: Optional[torch.Tensor], commit_lens: torch.Tensor, greedy: bool = False) -> None:
        if _STATS <= 0:
            return
        k = "greedy" if greedy else "sampled"
        if k not in self.acc:
            self.acc[k] = torch.zeros(4, dtype=torch.float64, device=commit_lens.device)
            self.cycles[k] = 0
        c = commit_lens.to(torch.float64).sum()
        if chain_len is not None:
            ch = chain_len.to(torch.float64).sum()
            # chain nodes 0..C-1 (node 0 = root) + Q's bonus at depth C: all C - 1 chain edges + the bonus accepted
            # <=> commit_len >= C + 1
            full = (commit_lens.to(torch.int64) >= chain_len.to(torch.int64) + 1).to(torch.float64).sum()
        else:
            ch = torch.zeros((), dtype=torch.float64, device=commit_lens.device)
            full = ch
        self.acc[k] += torch.stack([c, ch, full, torch.ones_like(c)])
        self.cycles[k] += 1
        if self.cycles[k] % _STATS == 0:
            c, ch, full, n = self.acc[k].tolist()
            logger.info(
                "SPECPILOT_K001_SAMPLE stats (%s): cycles=%d commit/cycle=%.3f q_chain_len=%.3f (root included) "
                "exact_passed_whole_chain_and_q_bonus=%.3f",
                k, int(n), c / n, ch / n, full / n,
            )


STATS = _Stats()

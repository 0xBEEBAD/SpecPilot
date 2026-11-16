"""SPECPILOT_FORCE: forced replay for paired speed measurements (measurement instrument; candidate d1-001-force).

The engine decodes normally (draft, Q passes, exact pass, tree accept), except that in the EXACT pass the target
prediction of every tree node is replaced by the token a recorded reference sequence has at that node's output position.
Every arm then emits exactly the reference text, and the per-turn decode times of two engines are paired. The Q passes
are untouched (the Q chain stays Q's own prediction), so cycle statistics are the engine's own wherever its exact
prediction equals the reference (all positions but near-ties).

Knobs (read once at import; both unset = strict no-op):
  SPECPILOT_FORCE_DIR   <dir> holding ``<key>.npy`` (int32 output token ids of the reference turn) and optionally
                        ``<key>.json`` (meta: ``prompt_tokens``, ``prompt_sha``). A request is forced when its rid
                        contains ``pfk.<key>.`` and ``<dir>/<key>.npy`` exists and the meta's prompt length / sha (when
                        present) match the request's prompt; any other request runs unforced.
  SPECPILOT_FORCE_LOG   <file.jsonl>: one line per request (forced or not) with the rid, prompt length and sha1 of the
                        prompt token ids, and for forced requests the divergence audit (below). Written when the next
                        request starts (and at interpreter exit), never inside the decode loop.

Indexing. ``prefix_lens`` (= ``batch.seq_lens``) is the committed KV length; the root node of the verify tree (depth 0)
is the last emitted token (the bonus) at position ``prefix_lens``, so ``prefix_lens = prompt_len + n_out - 1`` and the
node at depth ``d`` predicts output index ``prefix_lens - prompt_len + 1 + d`` (= ``n_out + d``). The root itself is
checked against the reference at output index ``prefix_lens - prompt_len`` (``root_mismatch_ct``: must stay 0). The
first output token comes from the prefill sampler; it is forced to ``reference[0]`` there.

Divergence audit (accumulated on the GPU, read once per request): over the committed path rows of every forced exact
pass (root .. last accepted node: exactly the rows whose prediction was used, all of them on the reference prefix), the
number of rows whose own exact prediction differs from the reference token (``div_ct``), the max / sum / >1 count of the
own-minus-reference gap in the row's score buffer (greedy: bf16 logits; T>0: the buffer after the keyed sampler), and
the first ``LOG_K`` divergences ``[output index, own, reference, gap]``. ``first_mismatch``: the prefill sampler's own
first token differed from ``reference[0]``.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import re
import time

import numpy as np
import torch

logger = logging.getLogger(__name__)

FORCE_DIR = os.environ.get("SPECPILOT_FORCE_DIR", "").strip()
FORCE_LOG = os.environ.get("SPECPILOT_FORCE_LOG", "").strip()
# SPECPILOT_FORCE_GATE=own: a reference token is accepted as a child only if it also passes the arm's own accept rule
# (own argmax; with the tie rule: its logit equals the row max); the token after the accepted path is always the
# reference token. Measures accept-rule changes on a fixed text (plain forcing would accept for both arms).
GATE = os.environ.get("SPECPILOT_FORCE_GATE", "").strip() == "own"
ACTIVE = bool(FORCE_DIR or FORCE_LOG)
LOG_K = 64
_RID_KEY = re.compile(r"pfk\.([A-Za-z0-9_\-]+)\.")


def rid_key(rid) -> str | None:
    m = _RID_KEY.search(str(rid or ""))
    return m.group(1) if m else None


def prompt_sha(ids) -> str:
    return hashlib.sha1(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


class _ForcedReq:
    """Per-request state: the staged reference and the GPU-side audit counters."""

    def __init__(self, rid: str, key: str | None, prompt_len: int, sha: str, device):
        self.rid, self.key, self.prompt_len, self.sha = rid, key, prompt_len, sha
        self.t0 = time.time()
        self.forced = False
        self.reason = "no key" if key is None else None
        self.ref = None
        self.ref_len = 0
        self.n_exact = 0
        self.stash = None
        self.device = device

    def stage(self, ref_ids: np.ndarray) -> None:
        d = self.device
        n = int(ref_ids.shape[0])
        pin = torch.device(d).type == "cuda"
        host = torch.full((n + 1,), -1, dtype=torch.int64, pin_memory=pin)
        host[:n] = torch.from_numpy(ref_ids.astype(np.int64))
        self.ref = host.to(d, non_blocking=True)  # ref[n] = -1: every index >= n falls back to the engine's own prediction
        self._host_ref = host  # keep the pinned source alive until the copy ran
        self.ref_len = n
        self.prompt_len_t = torch.tensor(self.prompt_len, dtype=torch.int64, device=d)
        self.div_ct = torch.zeros((), dtype=torch.int64, device=d)
        self.root_mis = torch.zeros((), dtype=torch.int64, device=d)
        self.first_mis = torch.zeros((), dtype=torch.int64, device=d)
        self.gap_gt1 = torch.zeros((), dtype=torch.int64, device=d)
        self.cut_ct = torch.zeros((), dtype=torch.int64, device=d)
        self.gap_max = torch.zeros((), dtype=torch.float32, device=d)
        self.gap_sum = torch.zeros((), dtype=torch.float32, device=d)
        self.log_i = torch.full((LOG_K + 1, 3), -1, dtype=torch.int64, device=d)
        self.log_g = torch.zeros(LOG_K + 1, dtype=torch.float32, device=d)
        self.forced = True

    def report(self) -> dict:
        out = {"rid": self.rid, "key": self.key, "forced": self.forced, "reason": self.reason,
               "prompt_len": self.prompt_len, "prompt_sha": self.sha, "wall_s": round(time.time() - self.t0, 3), "gate": GATE}
        if self.forced:
            n_log = min(int(self.div_ct), LOG_K)
            li, lg = self.log_i[:n_log].cpu().tolist(), self.log_g[:n_log].cpu().tolist()
            out.update(ref_len=self.ref_len, n_exact=self.n_exact, div_ct=int(self.div_ct),
                       root_mismatch_ct=int(self.root_mis), first_mismatch=int(self.first_mis),
                       gap_max=float(self.gap_max), gap_sum=float(self.gap_sum), gap_gt1=int(self.gap_gt1), cut_ct=int(self.cut_ct),
                       divergences=[[a, b, c, round(g, 4)] for (a, b, c), g in zip(li, lg)])
        return out


_REQS: dict[str, _ForcedReq] = {}
_WARNED = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        logger.warning("SPECPILOT_FORCE: %s", msg)


def _flush(rid: str) -> None:
    st = _REQS.pop(rid, None)
    if st is None:
        return
    rep = st.report()
    if st.forced:
        logger.info("SPECPILOT_FORCE %s key=%s ref_len=%d div=%d root_mis=%d first_mis=%d gap_max=%.3f",
                    rid, st.key, st.ref_len, rep["div_ct"], rep["root_mismatch_ct"], rep["first_mismatch"], rep["gap_max"])
    if FORCE_LOG:
        with open(FORCE_LOG, "a") as f:
            f.write(json.dumps(rep) + "\n")


def flush_all() -> None:
    for rid in list(_REQS):
        try:
            _flush(rid)
        except Exception:  # interpreter shutdown: CUDA may be gone
            pass


atexit.register(flush_all)


def _load_ref(key: str, prompt_len: int, sha: str):
    base = os.path.join(FORCE_DIR, key)
    if not os.path.exists(base + ".npy"):
        return None, "no reference file"
    meta = {}
    if os.path.exists(base + ".json"):
        with open(base + ".json") as f:
            meta = json.load(f)
    if meta.get("prompt_tokens") is not None and int(meta["prompt_tokens"]) != prompt_len:
        return None, f"prompt length {prompt_len} != reference {meta['prompt_tokens']}"
    if meta.get("prompt_sha") and meta["prompt_sha"] != sha:
        return None, f"prompt sha {sha} != reference {meta['prompt_sha']}"
    return np.load(base + ".npy"), None


def begin_extend(reqs, device) -> None:
    """Extend (prefill) step: flush finished requests, stage the reference of a new one (host work, once per request)."""
    live = {str(r.rid) for r in reqs}
    for rid in [x for x in _REQS if x not in live]:
        _flush(rid)
    for r in reqs:
        rid = str(r.rid)
        if rid in _REQS:
            continue
        ids = r.origin_input_ids
        st = _ForcedReq(rid, rid_key(rid), len(ids), prompt_sha(ids), device)
        _REQS[rid] = st
        if FORCE_DIR and st.key is not None:
            ref, why = _load_ref(st.key, st.prompt_len, st.sha)
            if ref is None:
                st.reason = why
                _warn_once(f"{rid}: unforced ({why})")
            else:
                st.stage(ref)


def _batch_state(reqs):
    if not FORCE_DIR:
        return None
    if len(reqs) != 1:
        if any(_REQS.get(str(r.rid)) is not None and _REQS[str(r.rid)].forced for r in reqs):
            _warn_once("batch size > 1: forcing is implemented for batch 1 only; running unforced")
        return None
    st = _REQS.get(str(reqs[0].rid))
    return st if st is not None and st.forced else None


def prefill_token(reqs, next_token_ids: torch.Tensor) -> None:
    """The prefill sampler's first token -> reference[0] (intermediate chunks' tokens are discarded anyway)."""
    st = _batch_state(reqs)
    if st is None or st.ref_len == 0:
        return
    own = next_token_ids[:1].to(torch.int64)
    ref0 = st.ref[:1]
    st.first_mis.copy_((own != ref0).sum())
    next_token_ids[:1].copy_(ref0.to(next_token_ids.dtype))


def exact_pred(pred: torch.Tensor, reqs, prefix_lens: torch.Tensor, depth: torch.Tensor,
               tokens: torch.Tensor, logits: torch.Tensor | None = None, tie: bool = False,
               parents: torch.Tensor | None = None) -> torch.Tensor:
    """Exact pass: [bs, N] per-node predictions -> the reference token at each node's output position.
    GATE: nodes whose reference token fails the own rule get -1 (no child matches; after_accept sets the bonus)."""
    st = _batch_state(reqs)
    if st is None:
        return pred
    lim = st.ref_len
    base = prefix_lens.view(-1, 1).to(torch.int64) - st.prompt_len_t
    idx = (base + 1 + depth.to(torch.int64)).clamp_(0, lim)
    forced = st.ref[idx]
    use = forced >= 0
    out = torch.where(use, forced, pred.to(torch.int64))
    ok = None
    if GATE:
        if tie:
            bs, n = pred.shape
            lg = logits.view(bs * n, -1)
            rows = torch.arange(bs * n, device=pred.device)
            ok = lg[rows, forced.clamp(min=0).view(-1)] == lg[rows, pred.to(torch.int64).view(-1)]
            ok = ok.view(bs, n)
        else:
            ok = forced == pred.to(torch.int64)
        out = torch.where(use & ~ok, torch.full_like(out, -1), out)
    rref = st.ref[base.view(-1).clamp(0, lim)]
    st.root_mis.add_(((rref >= 0) & (rref != tokens[:, 0].to(torch.int64))).sum())
    st.stash = (pred.to(torch.int64), forced, use, idx, ok, tokens, parents)
    st.n_exact += 1
    return out.to(pred.dtype)


def after_accept(reqs, tree_acc: dict, scores: torch.Tensor) -> None:
    """Divergence audit over the committed path rows (root .. last accepted node) of the forced exact pass."""
    st = _batch_state(reqs)
    if st is None or st.stash is None:
        return
    own, forced, use, idx, ok, tokens, parents = st.stash
    st.stash = None
    if parents is not None:  # artifact cut: the path ended at a rejected reference token, the tree had the own token
        bs_, n_ = own.shape
        leaf_ = tree_acc["leaf"][:bs_].to(torch.int64).view(-1, 1)
        own_l, ref_l = own.gather(1, leaf_), forced.gather(1, leaf_)
        rej = (ref_l >= 0) & ((own_l != ref_l) if ok is None else ~ok.gather(1, leaf_))
        kid = (parents.to(torch.int64) == leaf_) & (tokens.to(torch.int64) == own_l)
        st.cut_ct.add_((rej.view(-1) & kid.any(dim=1)).sum())
    if ok is not None:  # GATE: the token after the accepted path is the reference token
        leaf = tree_acc["leaf"][: own.shape[0]].to(torch.int64).view(-1, 1)
        fb = forced.gather(1, leaf).view(-1)
        bonus = torch.where(fb >= 0, fb, tree_acc["bonus"][: own.shape[0]].to(torch.int64))
        tree_acc["bonus"][: own.shape[0]].copy_(bonus.to(tree_acc["bonus"].dtype))
        acc = tree_acc["accept_len"][: own.shape[0]].to(torch.int64).view(-1, 1)
        out_tok = tree_acc["out_tokens"]
        out_tok[: own.shape[0]].scatter_(1, acc, bonus.view(-1, 1).to(out_tok.dtype))
    bs, n = own.shape
    path = tree_acc["path"][:bs].to(torch.int64)
    rows = torch.arange(n, device=own.device).view(1, -1)
    valid = rows < tree_acc["commit_lens"][:bs].view(-1, 1).to(torch.int64)
    p = path.clamp(min=0)
    own_p, ref_p = own.gather(1, p), forced.gather(1, p)
    if ok is None:
        mis = valid & use.gather(1, p) & (own_p != ref_p)
    else:  # GATE: positions where the own rule rejected the reference token (the misses this arm pays)
        mis = valid & use.gather(1, p) & ~ok.gather(1, p)
    flat = scores.view(bs * n, -1)
    rowi = (p + torch.arange(bs, device=own.device).view(-1, 1) * n).view(-1)
    s_own = flat[rowi, own_p.view(-1)].float()
    s_ref = flat[rowi, ref_p.clamp(min=0).view(-1)].float()
    gap = torch.where(mis.view(-1), s_own - s_ref, torch.zeros_like(s_own))
    m = mis.view(-1)
    before = st.div_ct.clone()
    st.div_ct.add_(m.sum())
    st.gap_sum.add_(gap.sum())
    torch.maximum(st.gap_max, gap.max(), out=st.gap_max)
    st.gap_gt1.add_((m & (gap > 1.0)).sum())
    slot = torch.where(m, (before + torch.cumsum(m.to(torch.int64), 0) - 1).clamp(max=LOG_K), LOG_K)
    vals = torch.stack([idx.gather(1, p).view(-1), own_p.view(-1), ref_p.view(-1)], dim=1)
    st.log_i.index_put_((slot,), vals)
    st.log_g.index_put_((slot,), gap)

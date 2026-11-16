"""PEARDR_LOSSY (pear-drafter experiments; NOT lossless): relaxed acceptance and nucleus-restricted steering on the k001
two-level tree verify, temperature > 0 requests only.

Everything is a rewrite of the per-node prediction ``pred`` that the unchanged tree accept matches children against
(same mechanism as specpilot_d1_tieacc / specpilot_k001_sample):

  p~[j, .]   target distribution of verify row j exactly as the lossless sampler builds it (temperature, top-p)
  x*[j]      the lossless keyed draw of row j from p~[j]
  steer      optional: tokens of a suppress list get probability 0 in rows where they hold mass in (0, suppress_max)
             (the rest is renormalised); rows where they hold >= suppress_max are untouched. Applied before the draw.
  relax      pred'[j] = token of the FIRST child c of j in node order (the drafter's best-first order) with
                 thr:      p~[j, tok_c] >= theta
                 rel:      p~[j, tok_c] >= theta * max_v p~[j, v]
                 nucleus:  p~[j, tok_c] > 0            (inside the deployed top-p support)
             else x*[j] (lossless behaviour for that row).  A child whose token is in ``strict`` (optional id list) is
             never relaxed.  The token after the accepted path (bonus) is the lossless draw of the leaf's row.

Every committed token therefore lies inside the top-p support of the exact target under its own prefix ("nucleus"
never emits a token the deployed sampler could not emit); the distribution inside the support is tilted towards the
drafter's first choice. Both verify passes (Q view and exact) use the same rule; the exact pass decides what commits.

Control: env PEARDR_CTL=<json file>, re-read at every request start when its mtime changes:
  {"mode": "off" | "thr" | "rel" | "nucleus", "theta": 0.3, "suppress": [ids], "suppress_max": 0.999,
   "strict": [ids], "stats": "<jsonl path>", "greedy": [ids], "pick": "draft" | "target"}
  greedy: rows whose most likely token is in the list emit it deterministically (steer; e.g. digits).
  pick:   which acceptable child a node takes: the drafter's first (default) or the target's most likely.
  norep:  n > 0 = loop guard (exact pass, batch 1): a child whose token would complete an n-token sequence that was
          already committed earlier in the SAME request is never relaxed -- inside a repetition every token is committed
          only by the target's own lossless draw, so the target keeps its chance to leave the repetition at each token.
          "Earlier in the same request" includes the prompt (earlier turns of the session) unless norep_prompt is 0.
          (Relaxed acceptance of a drafter that proposes "repeat what was written before" removes exactly that chance;
          the observed failure of low thresholds is a verbatim loop that ends in <|im_end|> mid-thought or at the cap.)
mode "off" with an empty suppress list = the parent engine, bit for bit (this module's sampler is then not entered).
Stats (one json line per finished request, GPU counters read once at the next request start): exact / Q passes,
committed tokens, committed drafted tokens that were not the lossless draw of their parent row ("lenient").
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import time

import torch

from sglang.srt.speculative import specpilot_k001_sample as _k001s

logger = logging.getLogger(__name__)

_CTL = (os.environ.get("PEARDR_CTL", "") or "").strip()
_DEFAULT = {"mode": "off", "theta": 0.0, "suppress": [], "suppress_max": 0.999, "strict": [], "stats": "", "greedy": [], "pick": "draft",
            "norep": 0, "norep_prompt": 1}
_REP: dict[str, dict] = {}   # loop guard state per live request: last tokens + bitmap of committed n-gram hashes (GPU)
_REP_BITS = 1 << 24
_REP_C: dict = {}
_CFG = dict(_DEFAULT)
_MT = [None]
_T = {"suppress": None, "strict": None, "greedy": None}  # device tensors, built lazily
_STATS: dict[str, dict] = {}
_LAST = {}


def _load() -> None:
    if not _CTL:
        return
    try:
        mt = os.stat(_CTL).st_mtime_ns
    except OSError:
        return
    if mt == _MT[0]:
        return
    try:
        with open(_CTL) as f:
            d = json.load(f)
    except Exception as e:  # half-written file: keep the old config, retry at the next request
        logger.warning("PEARDR_LOSSY: cannot read %s: %s", _CTL, e)
        return
    _MT[0] = mt
    _CFG.clear()
    _CFG.update(_DEFAULT)
    _CFG.update(d)
    _T["suppress"] = None
    _T["strict"] = None
    _T["greedy"] = None
    logger.info("PEARDR_LOSSY config: %s", {k: (v if not isinstance(v, list) else f"{len(v)} ids") for k, v in _CFG.items()})


def enabled() -> bool:
    """A control file is configured (stats are then recorded even in mode off)."""
    return bool(_CTL)


def active() -> bool:
    return bool(_CTL) and (_CFG["mode"] != "off" or bool(_CFG["suppress"]) or bool(_CFG["greedy"]))


def begin_extend(reqs) -> None:
    """Request start: flush finished requests' counters, poll the control file (host side only)."""
    if not _CTL:
        return
    live = {str(r.rid) for r in reqs}
    for rid in [x for x in _STATS if x not in live]:
        _flush(rid)
    _load()


def _rep_consts(n: int, device):
    key = (n, str(device))
    c = _REP_C.get(key)
    if c is None:
        g = torch.Generator().manual_seed(20261116)
        c = (torch.randint(1, 1 << 62, (n,), generator=g, dtype=torch.int64) * 2 + 1).to(device)   # odd multipliers
        _REP_C[key] = c
    return c


def _gram_idx(win: torch.Tensor) -> torch.Tensor:
    """[..., n] int64 token windows -> bitmap index in [0, _REP_BITS). int64 arithmetic wraps (two's complement)."""
    c = _rep_consts(win.shape[-1], win.device)
    g = ((win + 1) * c).sum(-1)
    g = g * -7046029254386353131          # 0x9E3779B97F4A7C15 as int64
    return (g >> 40) & (_REP_BITS - 1)


def _rep_state(rid: str, n: int, device, req=None):
    st = _REP.get(rid)
    if st is None or st["n"] != n:
        m = n - 1
        tail = torch.full((m,), -1, dtype=torch.int64, device=device)
        bits = torch.zeros(_REP_BITS, dtype=torch.uint8, device=device)
        ids = getattr(req, "origin_input_ids", None) if req is not None else None
        if ids is not None and len(ids) > 0 and len(getattr(req, "output_ids", ())) == 0:
            # request start: the tail is the end of the prompt (its last token is the first root); with norep_prompt
            # the prompt's own n-token sequences count as "already written" (one H2D copy per request)
            pt = torch.tensor(list(ids), dtype=torch.int64, device=device)
            k = min(m, pt.shape[0])
            tail[m - k:] = pt[-k:]
            if int(_CFG.get("norep_prompt", 1) or 0) and pt.shape[0] >= n:
                win = pt.unfold(0, n, 1)
                bits.scatter_reduce_(0, _gram_idx(win), torch.ones(win.shape[0], dtype=torch.uint8, device=device), reduce="amax")
        st = _REP[rid] = {"n": n, "tail": tail, "bits": bits}
    return st


def _rep_children(st, tok: torch.Tensor, par: torch.Tensor) -> torch.Tensor:
    """tok, par: int64 [N] of one request's verify tree (node 0 = root = last committed token, par < 0 for the root).
    Returns bool [N]: node c's token would complete an n-token sequence already committed in this request, where the
    n - 1 preceding tokens are c's ancestors in the tree followed (beyond the root) by the committed tail."""
    n = st["n"]; m = n - 1
    tail = st["tail"]
    N = tok.shape[0]
    a = torch.arange(N, device=tok.device)
    off = torch.zeros(N, dtype=torch.int64, device=tok.device)       # steps taken beyond the root
    cols = []
    for _ in range(m):                                               # token at distance 0, 1, ... from each node
        cols.append(torch.where(off == 0, tok[a], tail[(m - 1 - off).clamp(min=0)]))
        nxt = par[a]
        beyond = (nxt < 0) | (off > 0)
        off = off + beyond.to(torch.int64)
        a = torch.where(beyond, torch.zeros_like(a), nxt.clamp(min=0))
    ctx = torch.stack(cols[::-1], dim=-1)                            # [N, m], oldest first, ends with the node itself
    win = torch.cat([ctx[par.clamp(min=0)], tok.unsqueeze(-1)], dim=-1)   # the parent's m tokens + the child's token
    valid = (win >= 0).all(-1) & (par >= 0)
    return valid & (st["bits"][_gram_idx(win.clamp(min=0))] > 0)


def _rep_commit(st, out: torch.Tensor, commit: torch.Tensor) -> None:
    """Record the n-token sequences ending at each newly committed token; advance the tail. out int64 [L], commit 0-d."""
    n = st["n"]; m = n - 1
    seq = torch.cat([st["tail"], out])                               # [m + L]; tail[-1] is the root (last committed)
    win = seq.unfold(0, n, 1)                                        # [L, n]; window s ends at out[s]
    L = out.shape[0]
    valid = (torch.arange(L, device=out.device) < commit) & (win >= 0).all(-1)
    st["bits"].scatter_reduce_(0, _gram_idx(win.clamp(min=0)), valid.to(torch.uint8), reduce="amax")
    st["tail"] = seq.gather(0, torch.arange(m, device=out.device) + commit.to(torch.int64))


def _flush(rid: str) -> None:
    _REP.pop(rid, None)
    st = _STATS.pop(rid, None)
    if st is None:
        return
    rep = {"rid": rid, "t": time.time(), "cfg": st["cfg"], **{k: int(v) for k, v in st["c"].items()}}
    path = st["stats"]
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(rep) + "\n")
        except Exception as e:
            logger.warning("PEARDR_LOSSY: cannot write stats: %s", e)


def _flush_all() -> None:
    for rid in list(_STATS):
        try:
            _flush(rid)
        except Exception:
            pass


atexit.register(_flush_all)


def _ids(key: str, device):
    if not _CFG[key]:
        return None
    t = _T[key]
    if t is None or t.device != device:
        t = _T[key] = torch.tensor(sorted(set(int(x) for x in _CFG[key])), dtype=torch.int64, device=device)
    return t


def sample_nodes(next_token_logits, sampling_info, *, bs, n_nodes, draft_input, reqs, positions, q_pass, tokens, parents):
    """Drop-in for specpilot_k001_sample.sample_nodes: [bs, N] predictions (steered draw, then relaxed)."""
    probs = _k001s.tree_target_probs(next_token_logits, sampling_info, bs=bs, n_nodes=n_nodes, draft_input=draft_input)
    dev = probs.device
    n_steer = None
    sup = _ids("suppress", dev)
    if sup is not None:
        m = probs.index_select(-1, sup)  # [bs, N, S]
        mass = m.sum(-1)
        do = (mass > 0) & (mass < float(_CFG["suppress_max"]))
        probs = probs.clone()
        probs.index_copy_(-1, sup, torch.where(do.unsqueeze(-1), torch.zeros_like(m), m))
        probs = probs * torch.where(do, 1.0 / (1.0 - mass).clamp_min(1e-6), torch.ones_like(mass)).unsqueeze(-1)
        n_steer = do
    gid = _ids("greedy", dev)
    if gid is not None:
        # "greedy" steer: rows whose most likely token is in the list (e.g. digits) emit that token deterministically
        top = probs.argmax(-1)
        isg = torch.isin(top, gid)
        onehot = torch.zeros_like(probs)
        onehot.scatter_(-1, top.unsqueeze(-1), 1.0)
        probs = torch.where(isg.unsqueeze(-1), onehot, probs)
        n_steer = isg if n_steer is None else (n_steer | isg)
    seeds = _k001s.request_seeds(reqs, dev)
    draw = _k001s.keyed_sample(probs, seeds, positions, _k001s.q_salt() if q_pass else _k001s.SALT_EXACT)
    pred = draw
    mode = _CFG["mode"]
    if mode != "off":
        n = n_nodes
        par = parents.to(torch.int64)
        p = par.clamp(min=0)
        tok = tokens.to(torch.int64)
        rows = (torch.arange(bs, device=dev).view(-1, 1) * n + p).view(-1)
        child_p = probs.view(bs * n, -1)[rows, tok.view(-1)].view(bs, n)
        if mode == "thr":
            ok = child_p >= float(_CFG["theta"])
        elif mode == "rel":
            ok = (child_p > 0) & (child_p >= float(_CFG["theta"]) * probs.amax(-1).gather(1, p))
        elif mode == "nucleus":
            ok = child_p > 0
        else:
            raise ValueError(f"PEARDR_LOSSY: unknown mode {mode!r}")
        ok = ok & (par >= 0)
        strict = _ids("strict", dev)
        if strict is not None:
            ok = ok & ~torch.isin(tok, strict)
        _LAST["rep"] = None
        nrep = int(_CFG.get("norep", 0) or 0)
        if nrep > 1 and not q_pass and bs == 1:
            rep = _rep_children(_rep_state(str(reqs[0].rid), nrep, dev, reqs[0]), tok[0], par[0])
            _LAST["rep"] = (rep & ok[0]).sum()                       # relaxations withheld by the loop guard (all nodes)
            ok = ok & ~rep.unsqueeze(0)
        idx = torch.arange(n, device=dev).view(1, -1).expand(bs, n)
        if _CFG.get("pick", "draft") == "target":
            # among a node's acceptable children take the one the TARGET likes most (ties: drafter order)
            score = torch.where(ok, child_p.float(), torch.full_like(child_p, -1.0, dtype=torch.float32))
            best = torch.full((bs, n), -1.0, dtype=torch.float32, device=dev)
            best.scatter_reduce_(1, p, score, reduce="amax")
            ok = ok & (score >= best.gather(1, p))
        cidx = torch.where(ok, idx, torch.full_like(idx, n))
        first = torch.full((bs, n), n, dtype=torch.int64, device=dev)
        first.scatter_reduce_(1, p, cidx, reduce="amin")
        tok_first = tok.gather(1, first.clamp(max=n - 1))
        pred = torch.where(first < n, tok_first, draw.to(torch.int64)).to(draw.dtype)
    vocab = probs.shape[-1]
    bad = (pred < 0) | (pred >= vocab)          # diagnostic guard: a prediction must be a real token id
    _LAST["bad"] = bad.sum()
    pred = torch.where(bad, draw.to(pred.dtype), pred)
    if not q_pass:
        _LAST["draw"] = draw
        _LAST["steer"] = n_steer
        _LAST["nan_rows"] = torch.isnan(probs).any(-1).sum()
    return pred


def note_lossless_draw(draw, q_pass: bool) -> None:
    """Mode off: remember the lossless draw of the exact pass so that the counters stay comparable."""
    _LAST["bad"] = None
    _LAST["rep"] = None
    if not q_pass:
        _LAST["draw"] = draw
        _LAST["steer"] = None
        _LAST["nan_rows"] = None


def after_accept(reqs, tree_acc, tokens, q_pass: bool) -> None:
    """Per-request GPU counters (batch 1). Exact pass: committed tokens and how many drafted ones were lenient."""
    if len(reqs) != 1:
        return
    rid = str(reqs[0].rid)
    st = _STATS.get(rid)
    dev = tokens.device
    if st is None:
        z = lambda: torch.zeros((), dtype=torch.int64, device=dev)  # noqa: E731
        st = _STATS[rid] = {
            "c": {"q_passes": z(), "x_passes": z(), "commit": z(), "drafted": z(), "lenient": z(), "steer_rows": z(),
                  "bad_pred": z(), "nan_rows": z(), "special_commit": z(), "oov_commit": z(), "last_token": z(),
                  "rep_withheld": z()},
            "cfg": {k: (v if not isinstance(v, list) else len(v)) for k, v in _CFG.items() if k != "stats"},
            "stats": _CFG["stats"],
        }
    c = st["c"]
    if _LAST.get("bad") is not None:
        c["bad_pred"].add_(_LAST["bad"])
    if q_pass:
        c["q_passes"].add_(1)
        return
    c["x_passes"].add_(1)
    if _LAST.get("nan_rows") is not None:
        c["nan_rows"].add_(_LAST["nan_rows"])
    commit = tree_acc["commit_lens"][0].to(torch.int64)
    out = tree_acc["out_tokens"][0].to(torch.int64)
    nrep = int(_CFG.get("norep", 0) or 0)
    if nrep > 1:                                                     # loop guard: record what was just committed
        _rep_commit(_rep_state(rid, nrep, dev, reqs[0]), out, commit)
        if _LAST.get("rep") is not None:
            c["rep_withheld"].add_(_LAST["rep"])
    kk = torch.arange(out.shape[0], device=dev)
    live = kk < commit
    c["special_commit"].add_((live & (out >= 248044) & (out < 248320)).sum())
    c["oov_commit"].add_((live & ((out < 0) | (out >= 248320))).sum())
    c["last_token"].copy_(out[(commit - 1).clamp(min=0)])
    c["commit"].add_(commit)
    c["drafted"].add_(commit - 1)
    draw = _LAST.get("draw")
    if draw is not None:
        n = draw.shape[1]
        path = tree_acc["path"][0].to(torch.int64)
        j = torch.arange(n - 1, device=dev)
        valid = (j + 1) < commit
        par, child = path[:-1].clamp(min=0), path[1:].clamp(min=0)
        tok = tokens[0].to(torch.int64)[child]
        c["lenient"].add_((valid & (tok != draw[0].to(torch.int64)[par])).sum())
        steer = _LAST.get("steer")
        if steer is not None:
            k = torch.arange(n, device=dev)
            onpath = torch.zeros(n, dtype=torch.int32, device=dev)
            onpath.scatter_reduce_(0, path.clamp(min=0), (k < commit).to(torch.int32), reduce="amax")
            c["steer_rows"].add_((steer[0] & (onpath > 0)).sum())

"""SPECPILOT_D1_TIEACC: exact-tie acceptance ("argmax-set greedy") in the exact pass's tree accept (candidate d1-002-tieacc).

Greedy requests only. A child node matches its parent when the parent row's exact bf16 logit of the child's token equals
the row maximum (bitwise), not only when the child is the lowest-id argmax. Implemented by rewriting the per-node
prediction before the unchanged accept kernel: pred'[j] = token of the first child of j (node order: the chain child
comes first) whose logit ties the row max, else argmax[j]. The token after the accepted path (bonus) is the plain argmax
(the leaf has no tied child, else the path would continue). Every committed token therefore has the maximal exact logit
under its own prefix; the output differs from the seed only in which of several exactly tied tokens is emitted.
At temperature > 0 the worker never calls this module (keyed race, no ties): strict no-op.

Knobs (read once at import): SPECPILOT_D1_TIEACC=1 enables; SPECPILOT_D1_TIEACC_LOG=<file.jsonl> receives one line per
request with the self-check counters (also logged): committed tokens checked, tie picks (committed token != the
lowest-id argmax), violations (committed token's logit != row max; must be 0). Counters live on the GPU and are read
once, when the next request starts (or at interpreter exit).
"""
from __future__ import annotations

import atexit
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

ENABLED = os.environ.get("SPECPILOT_D1_TIEACC", "").strip() not in ("", "0")
_LOG = os.environ.get("SPECPILOT_D1_TIEACC_LOG", "").strip()

_STATS: dict[str, dict] = {}


def tie_pred(*, logits: torch.Tensor, pred: torch.Tensor, tokens: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """[bs, N] argmax predictions -> first tied child's token per node (GPU gathers only)."""
    bs, n = pred.shape
    lg = logits.view(bs * n, -1)
    pred64 = pred.to(torch.int64)
    rows = torch.arange(bs, device=pred.device).view(-1, 1) * n
    rowmax = lg[(rows + torch.arange(n, device=pred.device).view(1, -1)).view(-1), pred64.view(-1)].view(bs, n)
    par = parents.to(torch.int64)
    p = par.clamp(min=0)
    child_val = lg[(rows + p).view(-1), tokens.to(torch.int64).view(-1)].view(bs, n)
    tie = (par >= 0) & (child_val == rowmax.gather(1, p))
    cidx = torch.where(tie, torch.arange(n, device=pred.device).view(1, -1).expand(bs, n), n)
    first = torch.full((bs, n), n, dtype=torch.int64, device=pred.device)
    first.scatter_reduce_(1, p, cidx, reduce="amin")
    tok_first = tokens.to(torch.int64).gather(1, first.clamp(max=n - 1))
    return torch.where(first < n, tok_first, pred64).to(pred.dtype)


def begin_extend(reqs) -> None:
    live = {str(r.rid) for r in reqs}
    for rid in [x for x in _STATS if x not in live]:
        _flush(rid)


def _flush(rid: str) -> None:
    st = _STATS.pop(rid, None)
    if st is None:
        return
    rep = {"rid": rid, **{k: int(v) for k, v in st.items()}}
    logger.info("SPECPILOT_D1_TIEACC %s checked=%d tie_picks=%d violations=%d", rid, rep["checked"], rep["tie_picks"],
                rep["violations"])
    if _LOG:
        with open(_LOG, "a") as f:
            f.write(json.dumps(rep) + "\n")


def _flush_all() -> None:
    for rid in list(_STATS):
        try:
            _flush(rid)
        except Exception:  # interpreter shutdown: CUDA may be gone
            pass


atexit.register(_flush_all)


def self_check(*, reqs, logits: torch.Tensor, argmax: torch.Tensor, tokens: torch.Tensor, tree_acc: dict) -> None:
    """Committed drafted tokens (path rows 1..accept_len) must have their parent row's maximal logit (batch 1)."""
    if len(reqs) != 1:
        return
    rid = str(reqs[0].rid)
    st = _STATS.get(rid)
    if st is None:
        z = lambda: torch.zeros((), dtype=torch.int64, device=argmax.device)  # noqa: E731
        st = _STATS[rid] = {"checked": z(), "tie_picks": z(), "violations": z()}
    n = argmax.shape[1]
    lg = logits.view(n, -1)
    path = tree_acc["path"][0].to(torch.int64)
    commit = tree_acc["commit_lens"][0].to(torch.int64)
    j = torch.arange(n - 1, device=argmax.device)
    valid = (j + 1) < commit
    par, child = path[:-1].clamp(min=0), path[1:].clamp(min=0)
    tok = tokens[0].to(torch.int64)[child]
    am = argmax[0].to(torch.int64)[par]
    val, mx = lg[par, tok], lg[par, am]
    st["checked"].add_(valid.sum())
    st["tie_picks"].add_((valid & (tok != am)).sum())
    st["violations"].add_((valid & (val != mx)).sum())

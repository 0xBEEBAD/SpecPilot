"""SPECPILOT_ALGO_TRACE: per-pass anatomy trace of the two-level verify cycle (k001_algo; diagnostic only).

Off unless SPECPILOT_ALGO_TRACE=<file> is set at server start. When on, every verify pass (Q-step or exact step)
appends one pickled dict to <file>:
  stage 'q' | 'x', rid, prefix (committed length before the cycle), n_out (output tokens before the cycle),
  tokens/parents/depth [N] of the verified tree, pred [N] (argmax of this pass per node), topv/topi [N, 4]
  (top-4 logits and ids of this pass per node; Q pass: over the Q head's vocabulary subset), accept_len, path [N],
  bonus, lattice cand [S, K] and edge scores [S, K, K] (raw selector scores and the calibrated best-first edge
  log-scores) of the draft that fed this pass, chain / chain_len / q_bonus (exact step only), and wall-clock
  stamps taken after torch.cuda.synchronize(): t_in, t_draft, t_verify, t_end.
The stamps force host syncs, so a traced server is a few percent slower than an untraced one; outputs are unchanged
(the hook only reads).
"""
from __future__ import annotations

import os
import pickle
import time

import torch

_PATH = (os.environ.get("SPECPILOT_ALGO_TRACE", "") or "").strip()
_LAT = (os.environ.get("SPECPILOT_ALGO_TRACE_LAT", "1") or "1").strip() not in ("", "0")
_FH = [None]


def enabled() -> bool:
    return bool(_PATH)


def lattice_enabled() -> bool:
    return bool(_PATH) and _LAT


def now() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def _np(x):
    return x.detach().cpu().numpy()


def write(rec: dict) -> None:
    if _FH[0] is None:
        os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
        _FH[0] = open(_PATH, "ab")
    out = {}
    for k, v in rec.items():
        out[k] = _np(v) if torch.is_tensor(v) else v
    pickle.dump(out, _FH[0], protocol=4)
    _FH[0].flush()

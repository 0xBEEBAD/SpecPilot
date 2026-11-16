"""SPECPILOT_PA: PEAR algorithm-level knobs on top of the two-level verify (engine k001_pa; proposal side only).

Every mechanism here only changes WHICH tokens are proposed to the exact pass (the chain / tree it verifies). The
exact pass, its accept rule and the commit are untouched, so the emitted tokens are the exact target's greedy choices
(or, at temperature > 0, the keyed draws of specpilot_k001_sample) exactly as without these knobs.

Knobs (read once at import; unset = k001_pear behaviour):
  SPECPILOT_PA_QPASSES   n >= 1 (default 1): number of Q passes per cycle. Pass i > 1 drafts below the chain built so
                         far, verifies [chain + subtree] with the Q view (no commit) and extends the chain; the exact
                         pass then verifies [final chain + subtree] once.
  SPECPILOT_PA_QWIN      "<F>,<W>" (default off): in the Q pass only, the target's verify attention reads the first F and
                         the last W prefix positions instead of the whole prefix (split-KV kernel). The exact pass
                         always attends to everything.
  SPECPILOT_PA_QADAPT    "" (default: always run all Q passes) | "len:<L>": after each Q pass the host reads the chain
                         length (one host sync) and stops adding Q passes once chain_len >= L.
"""
from __future__ import annotations

import os

import torch


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name, "") or default).strip()


QPASSES = max(1, int(_env("SPECPILOT_PA_QPASSES", "1") or 1))
_QADAPT = _env("SPECPILOT_PA_QADAPT", "")
QADAPT_LEN = int(_QADAPT.split(":", 1)[1]) if _QADAPT.startswith("len:") else 0


_QWIN = _env("SPECPILOT_PA_QWIN", "")  # "<first>,<last>": Q-pass prefix attention keeps the first / last tokens only
QWIN = tuple(int(x) for x in _QWIN.split(",")) if _QWIN else (0, 0)
assert len(QWIN) == 2

BATCH_FIELDS = ("forward_mode", "spec_info", "input_ids", "out_cache_loc", "seq_lens_cpu", "seq_lens_sum")


def multi_q() -> bool:
    return QPASSES > 1


def extra_reserve_blocks() -> int:
    """KV / req_to_token headroom (in verify blocks) beyond the stock 2: the drafts of later passes run at
    prefix + chain_len with chain_len up to N - 1."""
    return 2 if QPASSES > 1 else 0


class LenProbe:
    """Host read of the Q chain length (batch 1) through a pinned buffer."""

    def __init__(self):
        self._host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._event = torch.cuda.Event()

    def read(self, chain_len: torch.Tensor) -> int:
        self._host.copy_(chain_len.reshape(-1)[:1].to(torch.int32), non_blocking=True)
        self._event.record()
        self._event.synchronize()
        return int(self._host[0])


_PROBE = [None]


def len_probe() -> LenProbe:
    if _PROBE[0] is None:
        _PROBE[0] = LenProbe()
    return _PROBE[0]

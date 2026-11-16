"""SPECPILOT_DE_VOCAB_SPARE=R: R spare rows next to the static vocabulary subset of the drafter candidate head
(SPECPILOT_DE_VOCAB_SIZE) and of the two-level verify's Q head (SPECPILOT_D4_QHEAD), filled at request start with the
prompt's token ids that the static subset lacks (port of pear-sd's tree_wide_sm120_vocab prompt-token merge).

Proposal-side only: the drafter can propose, and the Q pass can predict, the merged tokens; the exact verify pass keeps
the full bf16/W12 head and decides every emitted token, so outputs are unchanged.

Buffers (allocated once, before CUDA-graph capture; the captured graphs read them by address):
  w    [R, H] bf16   head rows of the merged ids (zeros when unused)
  bias [R]    bf16   0 = active row, -inf = unused row (never wins a top-k / argmax)
  ids  [R]    int64  full-vocabulary id per spare row (0 when unused)
The rows come from a host copy of the target lm_head (2.5 GB of host RAM for Qwen3.8), taken before the W12 encoder
frees the bf16 head. `extend` runs eagerly on the scheduler thread at a request's first prefill; when the spare rows
overflow they are recycled from the start (older requests' tokens are dropped first).

Knob (read once at import): SPECPILOT_DE_VOCAB_SPARE=<R> (0 / unset = off, nothing here runs).
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

SPARE = max(0, int((os.environ.get("SPECPILOT_DE_VOCAB_SPARE", "") or "0").strip() or 0))
_STATE: list = [None]


class SpareRows:
    def __init__(self, weight: torch.Tensor, num_org: int, static_ids: torch.Tensor) -> None:
        dev, h = weight.device, int(weight.shape[1])
        self.num_org = int(num_org)
        self.src = weight[: self.num_org].detach().to("cpu", copy=True)  # host copy of the head rows
        self.static = torch.zeros(self.num_org, dtype=torch.bool)
        self.static[static_ids.cpu()] = True
        self.w = torch.zeros((SPARE, h), dtype=torch.bfloat16, device=dev)
        self.bias = torch.full((SPARE,), float("-inf"), dtype=torch.bfloat16, device=dev)
        self.ids = torch.zeros((SPARE,), dtype=torch.int64, device=dev)
        self.pos = 0  # next row to write (ring)
        self.row_of: dict = {}  # id -> spare row
        self.id_at = [-1] * SPARE
        self.n_calls = self.n_added = 0
        logger.info("SPECPILOT_DE_VOCAB_SPARE: %d spare head rows (static subset %d of %d ids)", SPARE,
                    int(self.static.sum()), self.num_org)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden [M, H] -> [M, R] bf16 logits of the spare rows (-inf for unused rows)."""
        return torch.matmul(hidden.to(torch.bfloat16), self.w.T) + self.bias

    def extend(self, token_ids) -> int:
        self.n_calls += 1
        t = torch.as_tensor(token_ids, dtype=torch.int64).reshape(-1)
        t = t[(t >= 0) & (t < self.num_org)]
        if t.numel() == 0:
            return 0
        uniq, cnt = torch.unique(t, return_counts=True)
        keep = ~self.static[uniq]
        uniq, cnt = uniq[keep], cnt[keep]
        new = [(int(c), int(i)) for i, c in zip(uniq.tolist(), cnt.tolist()) if i not in self.row_of]
        if not new:
            return 0
        new.sort(key=lambda x: (-x[0], x[1]))  # most frequent first if they do not all fit
        new = [i for _, i in new[:SPARE]]
        rows = []
        for i in new:
            r = self.pos
            self.pos = (self.pos + 1) % SPARE
            old = self.id_at[r]
            if old >= 0:
                self.row_of.pop(old, None)
            self.id_at[r] = i
            self.row_of[i] = r
            rows.append(r)
        r_t = torch.tensor(rows, dtype=torch.int64, device=self.w.device)
        i_cpu = torch.tensor(new, dtype=torch.int64)
        self.w.index_copy_(0, r_t, self.src.index_select(0, i_cpu).to(self.w.device))
        self.ids.index_copy_(0, r_t, i_cpu.to(self.w.device))
        self.bias.index_fill_(0, r_t, 0.0)
        self.n_added += len(new)
        return len(new)


def create(weight: torch.Tensor, num_org: int, static_ids: torch.Tensor) -> Optional[SpareRows]:
    """Called where the static subset is built while the bf16 head is still resident (never inside graph capture)."""
    if SPARE <= 0:
        return None
    if _STATE[0] is None:
        _STATE[0] = SpareRows(weight, num_org, static_ids)
    return _STATE[0]


def get() -> Optional[SpareRows]:
    return _STATE[0]


def extend_from_reqs(reqs, seen_rids: set) -> None:
    """Worker hook: at a request's first prefill, merge its prompt ids into the spare rows."""
    sp = _STATE[0]
    if sp is None or not reqs:
        return
    for req in reqs:
        rid = getattr(req, "rid", None)
        if rid is None or rid in seen_rids:
            continue
        seen_rids.add(rid)
        ids = getattr(req, "origin_input_ids_unpadded", None)
        if ids is None:
            ids = getattr(req, "origin_input_ids", None)
        if ids is None or len(ids) == 0:
            continue
        added = sp.extend(ids)
        logger.info("SPECPILOT_DE_VOCAB_SPARE: rid=%s prompt=%d merged=%d active=%d/%d", rid, len(ids), added,
                    len(sp.row_of), SPARE)
    if len(seen_rids) > 100000:
        seen_rids.clear()

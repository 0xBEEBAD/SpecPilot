"""SPECPILOT_PEARCAL: proposal-side calibration knobs for the k001 two-level cycle (PEAR re-calibration).

Everything here only changes which tree the drafter proposes (edge temperatures, calibration table); the exact
verify pass and the accept rule are untouched, so outputs stay lossless.

Knobs (read once at import):
  SPECPILOT_TREE_TEMP_X   edge temperature of the exact pass's subtree (the tree drafted below the Q chain).
                          Unset = SPECPILOT_TREE_TEMP (parent behaviour: both drafts share one temperature).
                          The Q pass's tree keeps SPECPILOT_TREE_TEMP / SPECPILOT_TREE_TEMPS.
  SPECPILOT_PEARCAL_CTL   dev only: path of a JSON file polled (mtime) at every request's prefill; keys
                          {"temp_q": f, "temp_x": f, "blend": f, "log_q": [N_SLOT*N_RANK*N_BIN floats]} overwrite the
                          static tensors the captured draft graph reads (no restart needed for sweeps). Unset = off.

The captured draft graph reads the inverse temperatures from one static tensor (`_SelectorDraftSampler.tree_inv_temps`)
and the calibration table (+ blend, last element) from another, so both can be rewritten in place between replays.
"""
from __future__ import annotations

import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

_RAW_X = (os.environ.get("SPECPILOT_TREE_TEMP_X", "") or "").strip()
_CTL = (os.environ.get("SPECPILOT_PEARCAL_CTL", "") or "").strip()
_ST = {"inv_q": None, "inv_x": None, "mtime": None, "cur": None}
ACTIVE = bool(_RAW_X) or bool(_CTL)


def _init(sampler) -> None:
    if _ST["inv_q"] is None:
        _ST["inv_q"] = sampler.tree_inv_temps.clone()
        _ST["inv_x"] = (
            torch.full_like(sampler.tree_inv_temps, 1.0 / float(_RAW_X)) if _RAW_X else sampler.tree_inv_temps.clone()
        )
        logger.info("SPECPILOT_PEARCAL: temp_x=%s ctl=%s", _RAW_X or "(= Q)", _CTL or "off")


def stage(sampler, in_q: bool) -> None:
    """Before a draft-graph replay: select the pass's inverse temperatures (in-place write, same stream)."""
    if not ACTIVE or sampler is None or getattr(sampler, "tree_inv_temps", None) is None:
        return
    _init(sampler)
    want = "q" if in_q else "x"
    if _ST["cur"] != want:
        sampler.tree_inv_temps.copy_(_ST["inv_q"] if in_q else _ST["inv_x"])
        _ST["cur"] = want


def poll_ctl(sampler) -> None:
    """Dev hook (request prefill): apply the control file if it changed."""
    if not _CTL or sampler is None or getattr(sampler, "tree_inv_temps", None) is None:
        return
    try:
        mt = os.stat(_CTL).st_mtime_ns
    except OSError:
        return
    if mt == _ST["mtime"]:
        return
    _ST["mtime"] = mt
    try:
        with open(_CTL) as f:
            cfg = json.load(f)
    except Exception as e:  # half-written file: retry at the next request
        logger.warning("SPECPILOT_PEARCAL: ctl unreadable (%s)", e)
        _ST["mtime"] = None
        return
    _init(sampler)
    if cfg.get("temp_q"):
        _ST["inv_q"].fill_(1.0 / float(cfg["temp_q"]))
    if cfg.get("temp_x"):
        _ST["inv_x"].fill_(1.0 / float(cfg["temp_x"]))
    _ST["cur"] = None
    from sglang.srt.speculative import specpilot_tree as _dtree

    if _dtree.d3m_calib_enabled():
        tab = _dtree.d3m_calib_table(sampler.tree_inv_temps.device)
        if cfg.get("log_q") is not None:
            lq = torch.tensor(cfg["log_q"], dtype=torch.float32, device=tab.device)
            assert lq.numel() == tab.numel() - _dtree.d3m_n_edges() - 1, lq.numel()
            tab[: lq.numel()].copy_(lq)
        if cfg.get("blend") is not None:
            tab[-1] = float(cfg["blend"])
    logger.info("SPECPILOT_PEARCAL: ctl applied %s", {k: (v if k != "log_q" else "table") for k, v in cfg.items()})

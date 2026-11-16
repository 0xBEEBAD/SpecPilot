#!/usr/bin/env python3
"""Decode-rule arms -> the engine's lossy control file (PEARDR_CTL).

    python3 rule.py <arm> [--out <ctl file>] [--stats <jsonl>]      # print (or atomically write) the control JSON
    python3 rule.py --list

An arm is a "+"-joined list of parts:
  lossless         the engine's lossless T>0 sampler (mode off; the hook is not entered)
  er               alias of thr0.3+nohes+norep+retry (the measured "efficient reasoning" rule)
  thr<theta>       accept a drafted child if the target's (T, top-p) probability of its token is >= theta
  rel<tau>         accept a drafted child if its probability is >= tau * the row's maximum
  nohes            do not sample hesitation tokens (Wait / Hmm / Actually / Hold ...) in rows where the target has
                   an alternative (rows where they hold >= 0.999 of the mass are untouched)
  norep[<n>]       loop guard: a child that would complete an n-token sequence (default 8) already committed earlier
                   in the request is never relaxed
  retry            handled by the closed-loop runner: a turn that ends without a valid action is regenerated once
                   with the lossless rule
Every relaxed arm also gets the structural-token guard: special tokens (ids 248044..248319: <|im_end|>, </think>, ...)
are only ever committed by the target's own lossless draw.

thr / rel below 0.3 are refused unless --allow-below-0.3: lower thresholds produced verbatim loops and turns without an
action in closed loop. The rule was measured at T=1, top-p 0.95 only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Qwen3.8 tokenizer ids of hesitation words: Wait / Hmm / Actually / Hold (with and without leading space, lower case)
HES_IDS = [13784, 13428, 77264, 85152, 50821, 32645, 45847, 22041, 11158, 3655, 84485]
STRUCTURAL_IDS = list(range(248044, 248320))
ALIASES = {"er": "thr0.3+nohes+norep+retry"}
THETA_MIN = 0.3


def arm_cfg(name: str, allow_low: bool = False) -> dict:
    name = ALIASES.get(name, name)
    cfg: dict = {"mode": "off"}
    for part in name.split("+"):
        if part in ("lossless", "retry"):
            pass
        elif part == "nucleus":
            cfg.update(mode="nucleus")
        elif part.startswith("thr") or part.startswith("rel"):
            theta = float(part[3:])
            if theta < THETA_MIN and not allow_low:
                raise SystemExit(f"{part}: threshold below {THETA_MIN} is not validated (loops, turns without action); "
                                 "pass --allow-below-0.3 to use it anyway")
            cfg.update(mode=part[:3], theta=theta)
        elif part == "nohes":
            cfg.update(suppress=HES_IDS, suppress_max=0.999)
        elif part.startswith("nohes"):
            cfg.update(suppress=HES_IDS, suppress_max=float(part[5:]))
        elif part == "norep":
            cfg.update(norep=8)
        elif part.startswith("norep"):
            cfg.update(norep=int(part[5:]))
        else:
            raise SystemExit(f"unknown arm part {part!r} in {name!r}")
    if cfg["mode"] == "nucleus" and not allow_low:
        raise SystemExit("nucleus accepts any token inside top-p (effectively threshold 0): pass --allow-below-0.3")
    if cfg["mode"] != "off":
        cfg["strict"] = sorted(set(cfg.get("strict", [])) | set(STRUCTURAL_IDS))
    return cfg


def wants_retry(name: str) -> bool:
    return "retry" in ALIASES.get(name, name).split("+")


def write_ctl(path: str, cfg: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--out", default="")
    ap.add_argument("--stats", default="", help="engine writes one JSON line per finished request here")
    ap.add_argument("--allow-below-0.3", dest="allow_low", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list or not a.arm:
        print("lossless | er (= thr0.3+nohes+norep+retry) | parts: thr<x> rel<x> nohes norep[<n>] retry, joined by +")
        return
    cfg = arm_cfg(a.arm, a.allow_low)
    if a.stats:
        cfg["stats"] = a.stats
    if a.out:
        write_ctl(a.out, cfg)
        print(f"wrote {a.out}: {a.arm} -> mode {cfg['mode']}", file=sys.stderr)
    else:
        print(json.dumps(cfg))


if __name__ == "__main__":
    main()

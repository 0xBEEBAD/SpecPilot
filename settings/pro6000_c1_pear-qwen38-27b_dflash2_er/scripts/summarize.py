#!/usr/bin/env python3
"""Speed and success of each arm of a closed-loop run against a lossless baseline, paired by episode
(suite, task, trial, seed). (From the pear-drafter experiments' analyze_cl.py; the baseline can now be an arm of the
same run, which is what closed_loop.py produces.)

    python3 summarize.py <run dir> [--base-arm lossless] [--base-run <dir>] [--json out.json]

<run dir> holds <episode>/<arm>/ directories (closed_loop.py's output). The baseline is the arm --base-arm of
--base-run (default: the same run). Definitions (as in the measurement):
  robot clock   per episode: sum of per-turn inference wall time + native steps / 20 Hz
  clock x       sum of baseline clocks / sum of arm clocks over the episodes where BOTH succeeded
  success diff  arm success rate - baseline success rate (percentage points) over all paired episodes
  95% CIs       paired bootstrap over episodes (4000 resamples, seed 0)
An episode without episode.json counts as a failure only if its log shows the server's context-limit rejection
("HTTP Error 400"); otherwise it is unfinished (still running / killed) and left out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_ep(d: Path, logd: Path):
    tf = d / "turns.jsonl"
    ef = d / "episode.json"
    if not ef.exists():
        lg = logd / f"{d.parent.name}_{d.name}.log"
        if not (lg.exists() and "HTTP Error 400" in lg.read_text(errors="ignore")):
            return None
    if not tf.exists():
        return None
    T = [json.loads(l) for l in open(tf) if l.strip()]
    if ef.exists():
        E = json.load(open(ef))
        succ, reason, steps = bool(E["success"]), E.get("end_reason"), E.get("native_steps", 0)
    else:
        succ, reason, steps = False, "context", (T[-1].get("steps_total", 0) if T else 0)
    infer = sum(t["infer_s"] for t in T)
    return dict(success=succ, reason=reason, turns=len(T), tokens=sum(t["completion_tokens"] for t in T), infer_s=infer,
                retries=sum(1 for t in T if t.get("retry")),
                retry_tokens=sum(t["retry"]["first_tokens"] for t in T if t.get("retry")),
                noact=sum(1 for t in T if t.get("action") is None),
                decode_s=sum(t["decode_s"] for t in T), clock=infer + steps / 20.0, steps=steps)


def load_run(root: Path) -> dict:
    arms: dict = {}
    logd = root / "_logs"
    for d in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for ad in sorted(p for p in d.iterdir() if p.is_dir() and ".partial" not in p.name):
            r = load_ep(ad, logd)
            if r:
                arms.setdefault(ad.name, {})[d.name] = r
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--base-arm", default="lossless")
    ap.add_argument("--base-run", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(0)

    def boot(vals, f, n=4000):
        vals = list(vals)
        out = []
        for _ in range(n):
            idx = rng.integers(0, len(vals), len(vals))
            out.append(f([vals[i] for i in idx]))
        return np.percentile(out, [2.5, 97.5])

    arms = load_run(Path(a.run))
    base = (load_run(Path(a.base_run)) if a.base_run else arms).get(a.base_arm, {})
    if not base:
        raise SystemExit(f"no finished episodes of baseline arm {a.base_arm!r}")
    print(f"baseline arm {a.base_arm}: {len(base)} episodes, success {np.mean([b['success'] for b in base.values()]):.3f}")
    hdr = (f'{"arm":<26}{"n":>4}{"succ":>7}{"base":>7}{"diff pts":>9}{"95% CI":>16}{"tok/ep":>8}{"base":>8}'
           f'{"clock s":>8}{"base":>7}{"clock x (both succ)":>20}{"95% CI":>15}{"model-time x":>13}{"tok/s":>7}')
    print(hdr)
    out = {}
    for arm, eps in sorted(arms.items()):
        if arm == a.base_arm and not a.base_run:
            pairs = [(e, e) for e in eps.values()]
        else:
            pairs = [(eps[e], base[e]) for e in eps if e in base]
        if not pairs:
            continue
        s = np.mean([p[0]["success"] for p in pairs]); sb = np.mean([p[1]["success"] for p in pairs])
        f_diff = lambda v: 100 * (np.mean([x[0]["success"] for x in v]) - np.mean([x[1]["success"] for x in v]))
        ci = boot(pairs, f_diff) if len(pairs) > 3 else (float("nan"),) * 2
        both = [p for p in pairs if p[0]["success"] and p[1]["success"]]
        f_clock = lambda v: sum(x[1]["clock"] for x in v) / max(1e-9, sum(x[0]["clock"] for x in v))
        cx = f_clock(both) if both else float("nan")
        cci = boot(both, f_clock) if len(both) > 3 else (float("nan"),) * 2
        mx = sum(p[1]["infer_s"] for p in pairs) / max(1e-9, sum(p[0]["infer_s"] for p in pairs))
        tps = sum(p[0]["tokens"] for p in pairs) / max(1e-9, sum(p[0]["decode_s"] for p in pairs))
        row = dict(n=len(pairs), success=float(s), base_success=float(sb), diff_pts=float(f_diff(pairs)),
                   diff_ci=[float(ci[0]), float(ci[1])],
                   tokens=float(np.mean([p[0]["tokens"] for p in pairs])), base_tokens=float(np.mean([p[1]["tokens"] for p in pairs])),
                   clock=float(np.mean([p[0]["clock"] for p in pairs])), base_clock=float(np.mean([p[1]["clock"] for p in pairs])),
                   n_both=len(both), clock_speedup_both=float(cx), clock_speedup_ci=[float(cci[0]), float(cci[1])],
                   model_time_speedup_all=float(mx), tok_per_s=float(tps),
                   reasons={r: sum(p[0]["reason"] == r for p in pairs) for r in sorted({str(p[0]["reason"]) for p in pairs})},
                   retries=sum(p[0]["retries"] for p in pairs), retry_tokens=sum(p[0]["retry_tokens"] for p in pairs),
                   noaction_turns=sum(p[0]["noact"] for p in pairs), total_turns=sum(p[0]["turns"] for p in pairs),
                   win_loss=dict(arm_only=sum(p[0]["success"] and not p[1]["success"] for p in pairs),
                                 base_only=sum(p[1]["success"] and not p[0]["success"] for p in pairs)))
        out[arm] = row
        print(f'{arm:<26}{len(pairs):>4}{s:>7.3f}{sb:>7.3f}{row["diff_pts"]:>+9.1f}{f"[{ci[0]:+.1f}, {ci[1]:+.1f}]":>16}'
              f'{row["tokens"]:>8.0f}{row["base_tokens"]:>8.0f}{row["clock"]:>8.0f}{row["base_clock"]:>7.0f}{cx:>20.2f}'
              f'{f"[{cci[0]:.2f}, {cci[1]:.2f}]":>15}{mx:>13.2f}{tps:>7.0f}')
        print(f'{"":<26}end reasons {row["reasons"]} | succeeded only here {row["win_loss"]["arm_only"]}, only in baseline '
              f'{row["win_loss"]["base_only"]} | turns without action {row["noaction_turns"]} of {row["total_turns"]}, '
              f'lossless retries {row["retries"]} ({row["retry_tokens"]} tokens discarded)')
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()

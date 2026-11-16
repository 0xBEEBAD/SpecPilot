#!/usr/bin/env python3
"""Paired LIBERO closed-loop evaluation: the efficient-reasoning rule against lossless F on the same servers.
(From the pear-drafter experiments' sched.py; changes: job grid built here, paths from the environment.)

    python3 closed_loop.py --ports 29500,29510 [--arms lossless,er] [--name er_eval] [--limit N] [--dry-run]

Jobs are (arm, suite, task, trial, seed). The default grid is the measured one: libero_spatial / libero_object /
libero_goal x tasks 0-9 x trials 0-1 x seeds 1-2 = 120 episodes per arm, arms interleaved per episode so slow drift
of the machine hits both arms alike. Each server (started by serve.sh, one GPU each) runs one episode at a time;
before each episode the server's control file is set to the job's arm (the engine re-reads it at the next request).
Episodes go to <out>/<suite>_t<task>_r<trial>_s<seed>/<arm>/ (episode.json, turns.jsonl, chunks.jsonl, frames/),
logs to <out>/_logs/. Finished episodes (episode.json) are skipped, so a run can be resumed; several schedulers can
share one output dir (jobs are claimed atomically with mkdir).

Environment: SPECPILOT_WORK [~/specpilot_data/work]; ER_OUT [$SPECPILOT_WORK/pear_er/episodes/<name>];
ER_CTL_DIR [$SPECPILOT_WORK/pear_er/ctl] (must match serve.sh's ER_CTL: port_<port>.json); plus PEAR_REPO,
PEAR_WORK_DIR / LIBERO_* for run_episode.sh. On SIGTERM / SIGINT the scheduler stops its own episodes (by process
group of the children it started) and exits.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rule as R  # noqa: E402

SUITES = ["libero_spatial", "libero_object", "libero_goal"]


def rng(spec: str) -> list[int]:
    out = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ports", required=True, help="comma-separated ports of servers started by serve.sh")
    ap.add_argument("--arms", default="lossless,er")
    ap.add_argument("--name", default="er_eval")
    ap.add_argument("--suites", default=",".join(SUITES))
    ap.add_argument("--tasks", default="0-9")
    ap.add_argument("--trials", default="0-1")
    ap.add_argument("--seeds", default="1-2")
    ap.add_argument("--jobs", default="", help="optional job file instead of the grid: lines 'arm suite task trial seed'")
    ap.add_argument("--limit", type=int, default=0, help="only the first N episodes (x arms), for a smoke run")
    ap.add_argument("--max-turn-tokens", type=int, default=32768)
    ap.add_argument("--max-wall", type=float, default=1800.0, help="inference seconds after which an episode is cut")
    ap.add_argument("--stale-s", type=float, default=600.0)
    ap.add_argument("--dry-run", action="store_true", help="print the jobs and control files, run nothing")
    a = ap.parse_args()

    work = Path(os.environ.get("SPECPILOT_WORK") or Path(os.environ.get("SPECPILOT_DATA") or Path.home() / "specpilot_data") / "work")
    root = Path(os.environ.get("ER_OUT") or work / "pear_er" / "episodes" / a.name)
    ctl_dir = Path(os.environ.get("ER_CTL_DIR") or work / "pear_er" / "ctl")
    stats_dir = work / "pear_er" / "stats"
    arms = a.arms.split(",")
    cfgs = {arm: R.arm_cfg(arm) for arm in arms}           # validates the arms (threshold >= 0.3) before anything runs
    if a.jobs:
        jobs = [l.split() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
    else:
        eps = list(itertools.product(a.suites.split(","), rng(a.tasks), rng(a.trials), rng(a.seeds)))
        eps.sort(key=lambda e: (e[3], e[2], e[1], SUITES.index(e[0]) if e[0] in SUITES else 9))
        if a.limit:
            eps = eps[:a.limit]
        jobs = [[arm, s, str(t), str(r), str(sd)] for (s, t, r, sd) in eps for arm in arms]
    ports = [p.strip() for p in a.ports.split(",") if p.strip()]
    pending = []
    for arm, suite, task, trial, seed in jobs:
        ep = f"{suite}_t{task}_r{trial}_s{seed}"
        out = root / ep / arm
        if (out / "episode.json").exists():
            continue
        pending.append((arm, suite, task, trial, seed, ep, out))
    print(f"{len(jobs)} jobs, {len(pending)} pending, servers on ports {ports}, out {root}", flush=True)
    if a.dry_run:
        for arm in arms:
            print(f"  arm {arm}: control {json.dumps({k: (v if not isinstance(v, list) else f'<{len(v)} ids>') for k, v in cfgs[arm].items()})}"
                  f"{'  + lossless retry of turns without an action' if R.wants_retry(arm) else ''}")
        for j in pending[:8]:
            print("  ", " ".join(j[:5]), "->", j[6])
        if len(pending) > 8:
            print(f"   ... {len(pending) - 8} more")
        return
    logd = root / "_logs"
    logd.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    running: dict = {}
    stop = [False]

    def on_sig(*_):
        stop[0] = True

    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)

    def claim(out: Path) -> bool:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(out)
            return True
        except FileExistsError:
            pass
        try:
            if (out / "episode.json").exists() or (out / "FAILED").exists():
                return False
            tf = out / "turns.jsonl"
            ref = tf if tf.exists() else out
            if time.time() - ref.stat().st_mtime < a.stale_s:
                return False                   # another scheduler is running it
            os.rename(out, str(out) + f".partial{int(time.time())}")   # abandoned attempt: keep it aside, redo
            os.mkdir(out)
            return True
        except OSError:
            return False

    def healthy(port: str) -> bool:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read()
            return True
        except Exception:
            return False

    sh = str(HERE / "run_episode.sh")
    t0 = time.time()
    ndone = 0
    while (pending or running) and not stop[0]:
        for port in ports:
            if port in running:
                proc, job = running[port]
                if proc.poll() is None:
                    continue
                ndone += 1
                if proc.returncode != 0:     # e.g. the context limit (HTTP 400): counted as a failure, not retried
                    try:
                        (job[6] / "FAILED").write_text(str(proc.returncode))
                    except OSError:
                        pass
                print(f"[{time.time() - t0:.0f}s] done {job[5]} {job[0]} rc={proc.returncode} ({ndone} finished, {len(pending)} pending)", flush=True)
                del running[port]
            if pending and not stop[0]:
                if not healthy(port):
                    continue
                job = None
                while pending:
                    cand = pending.pop(0)
                    if claim(cand[6]):
                        job = cand
                        break
                if job is None:
                    continue
                arm, suite, task, trial, seed, ep, out = job
                cfg = dict(cfgs[arm])
                cfg["stats"] = str(stats_dir / f"port_{port}.jsonl")
                ctl = ctl_dir / f"port_{port}.json"
                R.write_ctl(str(ctl), cfg)
                time.sleep(0.05)
                log = open(logd / f"{ep}_{arm}.log", "w")
                cmd = ["bash", sh, "--port", port, "--backend", arm, "--suite", suite, "--task", task, "--trial", trial,
                       "--seed", seed, "--out", str(out), "--max-turn-tokens", str(a.max_turn_tokens),
                       "--max-wall", str(a.max_wall), "--rid-prefix", f"{a.name}|{arm}|{ep}"]
                if R.wants_retry(arm):
                    cmd += ["--retry-ctl-file", str(ctl), "--retry-arm-json", json.dumps(cfg)]
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[port] = (proc, job)
        time.sleep(1.0)
    if stop[0]:
        for port, (proc, job) in running.items():
            try:
                os.killpg(proc.pid, signal.SIGTERM)    # the episode's own session, started above
            except ProcessLookupError:
                pass
        print("stopped by signal; own episodes terminated", flush=True)
    print(f"scheduler exit after {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

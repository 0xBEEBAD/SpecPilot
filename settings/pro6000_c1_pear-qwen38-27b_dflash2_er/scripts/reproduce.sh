#!/usr/bin/env bash
# End to end: one server per GPU (serve.sh), the paired closed-loop evaluation (closed_loop.py: lossless and the
# efficient-reasoning rule on the same servers), the summary (summarize.py), then stop the servers this script started.
#
#   GPUS=0,1,2,3 PEAR_REPO=<pear checkout> PEAR_WORK_DIR=<PEAR install dir> \
#     bash settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/reproduce.sh [closed_loop.py args...]
#
# Environment (defaults in brackets): GPUS [0]; BASE_PORT [29500] (server i on BASE_PORT + 10 * gpu);
# RUN_NAME [er_eval]; PY [the SpecPilot venv's python, for numpy]; DRY_RUN=1 (print what would run: serve.sh dry run on the first GPU, the job list, no GPU use);
# plus everything serve.sh, closed_loop.py and run_episode.sh read (SPECPILOT_DATA, PEAR_MODEL, PEAR_DRAFT, PEAR_CALIB,
# PEAR_REPO, PEAR_WORK_DIR, LIBERO_*). Extra arguments go to closed_loop.py (e.g. --limit 6 for a smoke run).
# The measured run used 120 episodes per arm; at ~90-140 s of robot clock per episode that is roughly 8 GPU-hours
# for both arms.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
GPUS=${GPUS:-0}; BASE_PORT=${BASE_PORT:-29500}; RUN_NAME=${RUN_NAME:-er_eval}
export SPECPILOT_DATA=${SPECPILOT_DATA:-$HOME/specpilot_data}
export SPECPILOT_WORK=${SPECPILOT_WORK:-$SPECPILOT_DATA/work}
OUT=${ER_OUT:-$SPECPILOT_WORK/pear_er/episodes/$RUN_NAME}
# summarize.py needs numpy: the SpecPilot venv (scripts/setup_env.sh) has it
PY=${PY:-$SPECPILOT_DATA/shared/venv/bin/python}; [ -x "$PY" ] || PY=python3
PORTS=()
for g in ${GPUS//,/ }; do PORTS+=($((BASE_PORT + 10 * g))); done
PORT_LIST=$(IFS=,; echo "${PORTS[*]}")

if [ "${DRY_RUN:-0}" = 1 ]; then
  g=${GPUS%%,*}
  DRY_RUN=1 bash "$HERE/serve.sh" "$g" "$((BASE_PORT + 10 * g))"
  ER_OUT=$OUT python3 "$HERE/closed_loop.py" --ports "$PORT_LIST" --name "$RUN_NAME" --dry-run "$@"
  echo "[dry-run] then: $PY $HERE/summarize.py $OUT --base-arm lossless --json $OUT/summary.json"
  exit 0
fi
: "${PEAR_REPO:?set PEAR_REPO (see README)}"

STARTED=()
cleanup() { for p in "${STARTED[@]}"; do bash "$REPO/scripts/stop.sh" "$p" || true; done; }
trap cleanup EXIT
i=0
for g in ${GPUS//,/ }; do
  p=${PORTS[$i]}; i=$((i + 1))
  ER_RULE=lossless bash "$HERE/serve.sh" "$g" "$p"
  STARTED+=("$p")
done
ER_OUT=$OUT python3 "$HERE/closed_loop.py" --ports "$PORT_LIST" --name "$RUN_NAME" "$@"
"$PY" "$HERE/summarize.py" "$OUT" --base-arm lossless --json "$OUT/summary.json"

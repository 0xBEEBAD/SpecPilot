#!/usr/bin/env bash
# Serve PEAR (Qwen3.8-27B-PEAR + DFlash2) on the SpecPilot F engine (d5-009-f) with the efficient-reasoning hook, one
# GPU, and block until the server is ready. Stop it with: bash scripts/stop.sh <port>   (repo root)
#
#   bash settings/pro6000_c1_pear-qwen38-27b_dflash2_er/scripts/serve.sh [gpu] [port]
#
# Environment (defaults in brackets):
#   SPECPILOT_DATA   data root [~/specpilot_data]; venv, models and caches under $SPECPILOT_DATA/shared (XINFER_SHARED),
#                    engines, logs, control files under $SPECPILOT_DATA/work (SPECPILOT_WORK). See research/env.sh.
#   PEAR_MODEL       target checkpoint dir [$XINFER_SHARED/models/Qwen3.8-27B-PEAR]
#   PEAR_DRAFT       DFlash2 drafter dir [$XINFER_SHARED/models/Qwen3.8-27B-DFlash2]
#   PEAR_CALIB       calibration dir of the data-aware Q copy (hash 029035503129f133)
#                    [$XINFER_SHARED/pear_calib/hess_r_029035503129f133]; set PEAR_CALIB= (empty) to serve without it
#                    (plain RTN Q copy: still lossless, a few % slower, not the measured configuration)
#   ER_RULE          rule written to the control file at start: er | lossless | any rule.py arm [er]
#   ER_CTL           control file the engine polls [$SPECPILOT_WORK/pear_er/ctl/port_<port>.json]; the closed-loop
#                    runner rewrites it before every episode
#   MEM              --mem-fraction-static [0.91]
#   RADIX=0          no prefix cache (--disable-radix-cache instead of the mamba radix-cache args); for calibration capture
#   DRY_RUN=1        build the engine tree and print the environment and command, start nothing
# gpu defaults to $GPU or 0, port to $PORT or 29500 + 10 * gpu (torch.distributed uses port + 1).
# Extra sglang.launch_server arguments can be given after the port.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
SETTING=$(basename "$(dirname "$HERE")")
GPU=${1:-${GPU:-0}}; PORT=${2:-${PORT:-$((29500 + 10 * GPU))}}; shift $(( $# >= 2 ? 2 : $# ))
export SPECPILOT_SETTING=$SETTING
export SERVED_MODEL_NAME=qwen3_8_27b                 # the name PEAR's clients use
[ -n "${PEAR_MODEL:-}" ] && export TARGET_MODEL=$PEAR_MODEL
[ -n "${PEAR_DRAFT:-}" ] && export DRAFT_MODEL=$PEAR_DRAFT
source "$REPO/research/env.sh" > /dev/null
PEAR_CALIB=${PEAR_CALIB-$XINFER_SHARED/pear_calib/hess_r_029035503129f133}
ER_RULE=${ER_RULE:-er}
ER_CTL=${ER_CTL:-$SPECPILOT_WORK/pear_er/ctl/port_$PORT.json}
MEM=${MEM:-0.91}
if [ -n "$PEAR_CALIB" ] && [ ! -f "$PEAR_CALIB/CONTENT_HASH" ] && [ "${DRY_RUN:-0}" != 1 ]; then
  echo "PEAR_CALIB=$PEAR_CALIB has no CONTENT_HASH; point it at the calibration dir or set PEAR_CALIB= (see README)"; exit 2
fi

ENG=$SPECPILOT_WORK/engines/$SETTING-final
[ -f "$ENG/TREE_SHA" ] || { rm -rf "$ENG"; python3 "$REPO/scripts/build_engine.py" "$SETTING" final "$ENG"; }
python3 "$HERE/rule.py" "$ER_RULE" --out "$ER_CTL" --stats "$SPECPILOT_WORK/pear_er/stats/port_$PORT.jsonl"
LOG=$SPECPILOT_WORK/logs/server_${SETTING}_$PORT.log
mkdir -p "$(dirname "$LOG")"

EXPORTS=$(python3 - "$REPO" "$SETTING" "$ENG" "$XINFER_SHARED" "$PEAR_CALIB" "$ER_CTL" "$MEM" <<'PY'
import json, shlex, sys
from pathlib import Path
repo, setting, eng, shared, calib, ctl, mem = sys.argv[1:8]
repo, eng, shared = Path(repo), Path(eng), Path(shared)
cfg = json.loads((repo / "settings" / setting / "config.json").read_text())
f = cfg["final"]
sha = (eng / "TREE_SHA").read_text().strip()
own = shared / "cache/engines" / sha[:16]
env = {"XINFER_ENGINE": eng / "python", "XINFER_LAB": eng / "lab", "PYTHONPATH": f"{eng / 'python'}:{eng / 'lab'}",
       "SGLANG_CACHE_DIR": own / "sglang", "SGLANG_JIT_CACHE_DIR": own / "sglang/jit",
       "TORCH_EXTENSIONS_DIR": own / "torch_extensions", "SNAPSHOT_SHA": sha,
       "SPECPILOT_BLOCK_SIZE": f["block_size"], "XINFER_CANDIDATE": f["name"], "PEARDR_CTL": ctl}
env.update(f["env"])
for k, v in f["path_env"].items():
    if k.startswith("_"):
        continue
    if k.startswith("SPECPILOT_D3K_") and not calib:
        continue                       # no calibration: plain RTN Q copy (seed bit widths)
    env[k] = str(v).replace("{ENGINE}", str(eng)).replace("{PEAR_CALIB}", calib)
# the stock flags carry --mem-fraction-static 0.91; MEM overrides it
flags = list(cfg["stock"]["server_flags"])
flags[flags.index("--mem-fraction-static") + 1] = mem
env["SPECPILOT_SERVER_FLAGS"] = " ".join(flags)
for k, v in env.items():
    print(f"export {k}={shlex.quote(str(v))}")
import os
launch = f["launch_args"] if os.environ.get("RADIX", "1") != "0" else ["--disable-radix-cache"]
print(f"LAUNCH=({' '.join(shlex.quote(x) for x in launch)})")
PY
)
PASS=(HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 TERM=dumb
      SPECPILOT_DATA="$SPECPILOT_DATA" XINFER_SHARED="$XINFER_SHARED" SPECPILOT_WORK="$SPECPILOT_WORK"
      SPECPILOT_SETTING="$SETTING" CUDA_VISIBLE_DEVICES="$GPU" SPECPILOT_NCCL_PORT=$((PORT + 1))
      SERVED_MODEL_NAME="$SERVED_MODEL_NAME" TARGET_MODEL="$TARGET_MODEL" DRAFT_MODEL="$DRAFT_MODEL"
      GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo PYTHONDONTWRITEBYTECODE=1)
[ -n "${HF_TOKEN:-}" ] && PASS+=(HF_TOKEN="$HF_TOKEN")
# calibration capture (README, "Calibration"): SPECPILOT_D3K_CAPTURE=<dir>:<module suffixes> is passed through
[ -n "${SPECPILOT_D3K_CAPTURE:-}" ] && PASS+=(SPECPILOT_D3K_CAPTURE="$SPECPILOT_D3K_CAPTURE")

echo "[serve] $SETTING (F + efficient reasoning, rule $ER_RULE) on GPU $GPU port $PORT"
echo "[serve] engine $ENG, control file $ER_CTL, calibration ${PEAR_CALIB:-<none>}, log $LOG"
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "[dry-run] environment:"; echo "$EXPORTS" | sed 's/^/  /'
  echo "[dry-run] command: research/scripts/serve.sh dflash2 $PORT $LOG \${LAUNCH[@]} $*"
  echo "[dry-run]   = python -m sglang.launch_server --model-path $TARGET_MODEL \$SPECPILOT_SERVER_FLAGS --random-seed 0" \
       "--host 127.0.0.1 --port $PORT --nccl-port $((PORT + 1)) --served-model-name $SERVED_MODEL_NAME" \
       "--speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT_MODEL --speculative-num-draft-tokens 32 \${LAUNCH[@]} $*"
  exit 0
fi
env -i "${PASS[@]}" bash -c '
set -euo pipefail
eval "$1"; REPO=$2; PORT=$3; LOG=$4; shift 4
source "$REPO/research/env.sh" > /dev/null
"$REPO/research/scripts/serve.sh" dflash2 "$PORT" "$LOG" "${LAUNCH[@]}" "$@"
' _ "$EXPORTS" "$REPO" "$PORT" "$LOG" "$@"
echo "[serve] ready: http://127.0.0.1:$PORT (stop with: bash scripts/stop.sh $PORT)"

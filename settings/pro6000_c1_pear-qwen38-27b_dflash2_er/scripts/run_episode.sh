#!/usr/bin/env bash
# Run run_episode.py in PEAR's LIBERO simulator env with the variables LiberoEnv.setup_env gives its simulator
# processes (Mesa llvmpipe EGL, single-thread BLAS, no CUDA).   usage: run_episode.sh <run_episode.py args>
# Environment (defaults in brackets):
#   PEAR_REPO          checkout of github.com/mit-han-lab/pear (required; measured at commit f32aa1f)
#   PEAR_WORK_DIR      the WORK_DIR given to PEAR's `scripts/install.sh libero` and `... mesa`
#   LIBERO_PYTHON      [$PEAR_WORK_DIR/envs/libero/bin/python]
#   LIBERO_MESA_DIR    [$PEAR_WORK_DIR/envs/mesa]
#   LIBERO_CONFIG_PATH [$PEAR_WORK_DIR/envs/libero/etc/libero]
set -euo pipefail
: "${PEAR_REPO:?set PEAR_REPO to a checkout of github.com/mit-han-lab/pear}"
if [ -z "${LIBERO_PYTHON:-}" ] || [ -z "${LIBERO_MESA_DIR:-}" ] || [ -z "${LIBERO_CONFIG_PATH:-}" ]; then
  : "${PEAR_WORK_DIR:?set PEAR_WORK_DIR (the WORK_DIR of the PEAR install) or LIBERO_PYTHON, LIBERO_MESA_DIR, LIBERO_CONFIG_PATH}"
fi
LIBERO_PYTHON=${LIBERO_PYTHON:-$PEAR_WORK_DIR/envs/libero/bin/python}
MESA=${LIBERO_MESA_DIR:-$PEAR_WORK_DIR/envs/mesa}
export PEAR_REPO
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-$PEAR_WORK_DIR/envs/libero/etc/libero}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export LD_LIBRARY_PATH=$MESA/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export __EGL_VENDOR_LIBRARY_FILENAMES=$MESA/share/glvnd/egl_vendor.d/50_mesa.json
export LIBGL_DRIVERS_PATH=$MESA/lib/dri GALLIUM_DRIVER=llvmpipe LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=4
export XDG_RUNTIME_DIR=${TMPDIR:-/tmp}/xdg_$USER CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
exec "$LIBERO_PYTHON" -u "$(dirname "${BASH_SOURCE[0]}")/run_episode.py" "$@"

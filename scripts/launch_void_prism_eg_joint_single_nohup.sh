#!/usr/bin/env bash
set -u -o pipefail

# Launch the void-prism joint E_G scoring in a way that survives disconnects.
#
# Run THIS script under a single nohup:
#
#   out_base="outputs/finalization/void_prism_eg_joint_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_void_prism_eg_joint_single_nohup.sh "$out_base" > "$out_base/launcher.log" 2>&1 &
#
# Status:
#   .venv/bin/python scripts/status_void_prism_eg_joint.py "$out_base"
#
# Stop:
#   kill -TERM "$(cat "$out_base/pid.txt")"
#
# Notes:
# - Uses `setsid taskset -c ... < /dev/null &` for robust detachment (same paradigm as info+).

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/finalization/void_prism_eg_joint}"
suite_json="${2:-outputs/void_prism_eg_suite_20260130_091241UTC/tables/suite_joint.json}"
run_base="${3:-outputs/finalization/info_plus_full_256_detached_20260129_0825UTC}"

mkdir -p "$out_base"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)" >&2
  exit 2
fi

if [ ! -f "$suite_json" ]; then
  echo "ERROR: suite_json not found: $suite_json" >&2
  exit 2
fi

common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
)

seeds=(101 202 303 404 505)
run_dirs=()
for s in "${seeds[@]}"; do
  rd="$run_base/M0_start${s}"
  if [ ! -d "$rd" ]; then
    echo "ERROR: run dir missing: $rd" >&2
    exit 2
  fi
  run_dirs+=(--run-dir "$rd")
done

# This scoring run is single-process. Pin to a small core set to keep the machine responsive.
coreset="${VOID_PRISM_CPUSET:-0-7}"

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"
echo "[launcher] suite_json=$suite_json"
echo "[launcher] run_base=$run_base"
echo "[launcher] coreset=$coreset"

cmd=(
  .venv/bin/python scripts/run_void_prism_eg_joint_test.py
  --suite-json "$suite_json"
  --fit-amplitude
  --convention A
  --max-draws "${VOID_PRISM_MAX_DRAWS:-2000}"
  --progress-every-block "${VOID_PRISM_PROGRESS_EVERY_BLOCK:-1}"
  "${run_dirs[@]}"
  --out "$out_base"
)

echo "[launcher] cmd: ${cmd[*]}"

env "${common_env[@]}" \
  setsid taskset -c "$coreset" \
  "${cmd[@]}" \
  > "$out_base/run.log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$out_base/pid.txt"
echo "[launcher] started pid=$pid"
echo "[launcher] run.log=$out_base/run.log"
exit 0


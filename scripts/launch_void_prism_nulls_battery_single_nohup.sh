#!/usr/bin/env bash
set -u -o pipefail

# Launch a "battery" of void-prism null tests in a robust detached way.
#
# IMPORTANT: This script uses `setsid taskset -c ... bash job.sh < /dev/null &` so the job
# survives disconnects and command-runner teardown (same paradigm as info+).
#
# Example:
#   out_base="outputs/finalization/void_prism_nulls_battery_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_void_prism_nulls_battery_single_nohup.sh "$out_base" \
#     outputs/void_prism_eg_suite_20260130_200458UTC/tables/suite_joint.json \
#     50 123 > "$out_base/launcher.log" 2>&1 &
#
# Status:
#   .venv/bin/python scripts/status_void_prism_boss_smica.py "$out_base"
#   tail -n 50 "$out_base/run.log"
#
# Stop:
#   kill -TERM "$(cat "$out_base/pid.txt")"
#
# Output subdirs:
#   $out_base/rotate_voids
#   $out_base/rotate_kappa

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/finalization/void_prism_nulls_battery}"
suite_json="${2:-outputs/void_prism_eg_suite_20260130_200458UTC/tables/suite_joint.json}"
n_null="${3:-50}"
seed="${4:-123}"

mkdir -p "$out_base"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)" >&2
  exit 2
fi

if [ ! -f "$suite_json" ]; then
  echo "ERROR: suite_json not found: $suite_json" >&2
  exit 2
fi

# Defaults: use a big chunk of the machine but keep the OS responsive.
# Override with: VOID_PRISM_CPUSET="0-255"
coreset="${VOID_PRISM_CPUSET:-0-191}"

# Threading knobs:
# - For healpy/libsharp the main knob is typically OMP_NUM_THREADS.
# - We keep BLAS threads at 1 by default to avoid oversubscription.
omp_threads="${VOID_PRISM_OMP_THREADS:-1}"
blas_threads="${VOID_PRISM_BLAS_THREADS:-1}"

common_env=(
  "OMP_NUM_THREADS=$omp_threads"
  "MKL_NUM_THREADS=$blas_threads"
  "OPENBLAS_NUM_THREADS=$blas_threads"
  "NUMEXPR_NUM_THREADS=$blas_threads"
  PYTHONUNBUFFERED=1
)

job="$out_base/job.sh"
cat > "$job" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$repo_root"

echo "[job] repo_root=$repo_root"
echo "[job] out_base=$out_base"
echo "[job] suite_json=$suite_json"
echo "[job] n_null=$n_null seed=$seed"
echo "[job] coreset=$coreset"
echo "[job] started_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

for kind in rotate_voids rotate_kappa; do
  sub="$out_base/\$kind"
  mkdir -p "\$sub"
  echo "[job] BEGIN kind=\$kind out=\$sub"
  env ${common_env[*]} \\
    .venv/bin/python scripts/run_void_prism_nulls.py \\
      --suite-json "$suite_json" \\
      --null-kind "\$kind" \\
      --n-null "$n_null" \\
      --seed "$seed" \\
      --out-base "\$sub"
  echo "[job] END kind=\$kind"
  echo
done

echo "[job] done_utc=\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$job"

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"
echo "[launcher] suite_json=$suite_json"
echo "[launcher] n_null=$n_null seed=$seed"
echo "[launcher] coreset=$coreset"
echo "[launcher] omp_threads=$omp_threads blas_threads=$blas_threads"
echo "[launcher] job=$job"

env "${common_env[@]}" \
  setsid taskset -c "$coreset" \
  bash "$job" \
  > "$out_base/run.log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$out_base/pid.txt"
echo "[launcher] started pid=$pid"
echo "[launcher] run.log=$out_base/run.log"
exit 0

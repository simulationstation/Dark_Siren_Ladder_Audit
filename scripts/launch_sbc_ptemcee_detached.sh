#!/usr/bin/env bash
set -u -o pipefail

# Launch SBC runs in a way that survives session disconnects (same pattern as info+):
# - Run THIS script under a single nohup
# - Each SBC run is started via `setsid ... < /dev/null &` so it's robustly detached
#
# Example:
#   out_base="outputs/validation/sbc_ptemcee_detached_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_sbc_ptemcee_detached.sh "$out_base" > "$out_base/launcher.log" 2>&1 &
#
# Full-system example (256 cores total across bh+prior; only do this when the host is otherwise idle):
#   out_base="outputs/validation/sbc_ptemcee_detached_full_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup env SBC_PROCS=128 SBC_CPU_CORES=0 SBC_MAX_RSS_MB=1024 bash scripts/launch_sbc_ptemcee_detached.sh "$out_base" > "$out_base/launcher.log" 2>&1 &
#
# Monitor:
#   scripts/status_sbc_out_base.sh "$out_base"
#
# Stop:
#   xargs -r kill -TERM < "$out_base/pids.txt"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/validation/sbc_ptemcee_detached_$(date -u +%Y%m%d_%H%M%SUTC)}"
mkdir -p "$out_base"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)" >&2
  exit 2
fi

# Keep BLAS/OpenMP from oversubscribing cores.
common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  MPLBACKEND=Agg
)

# Keep these conservative so we don't slow the main info+ run too much.
# If you want this faster, increase cpu/sbc-procs *after* info+ completes.
sbc_n="${SBC_N:-20}"
sbc_procs="${SBC_PROCS:-10}"
cpu_cores="${SBC_CPU_CORES:-10}"
max_rss_mb="${SBC_MAX_RSS_MB:-1536}"

# PTEMCEE chain settings for SBC (can be overridden via env).
steps="${SBC_STEPS:-400}"
burn="${SBC_BURN:-140}"
draws="${SBC_DRAWS:-200}"
pt_ntemps="${SBC_PT_NTEMPS:-4}"
pt_tmax="${SBC_PT_TMAX:-25}"

common_args=(
  --sampler-kind ptemcee
  --pt-ntemps "$pt_ntemps"
  --pt-tmax "$pt_tmax"
  --steps "$steps"
  --burn "$burn"
  --draws "$draws"
  --sbc-n "$sbc_n"
  --sbc-procs "$sbc_procs"
  --cpu-cores "$cpu_cores"
  --max-rss-mb "$max_rss_mb"
)

pids_path="$out_base/pids.txt"
: > "$pids_path"

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"
echo "[launcher] sbc_n=$sbc_n sbc_procs=$sbc_procs cpu_cores=$cpu_cores max_rss_mb=$max_rss_mb"
echo "[launcher] chain: steps=$steps burn=$burn draws=$draws pt_ntemps=$pt_ntemps pt_tmax=$pt_tmax"

for truth in bh prior; do
  out="$out_base/${truth}"
  mkdir -p "$out"
  echo "[launcher] starting truth=$truth out=$out"
  env "${common_env[@]}" \
    setsid nice -n 10 \
    .venv/bin/python scripts/run_calib_sbc.py \
      --out "$out" \
      --seed 0 \
      --truth-mu "$truth" \
      "${common_args[@]}" \
    > "$out/run.log" 2>&1 < /dev/null &
  pid="$!"
  echo "$pid" >> "$pids_path"
  echo "[launcher] truth=$truth pid=$pid"
done

echo "[launcher] pids written to $pids_path"
echo "[launcher] launched SBC runs (detached)."
exit 0

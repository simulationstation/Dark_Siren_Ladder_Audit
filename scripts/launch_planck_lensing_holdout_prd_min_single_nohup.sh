#!/usr/bin/env bash
set -u -o pipefail

# PRD-minimum Planck lensing holdout test: same structure as the "strong" test,
# but with shorter chains intended to be "publication-usable" while avoiding
# multi-day walltimes. This is still resumable via checkpoints.
#
# Run THIS script under a single nohup:
#
#   out_base="outputs/finalization/planck_lensing_holdout_prd_min_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_planck_lensing_holdout_prd_min_single_nohup.sh "$out_base" > "$out_base/launcher.log" 2>&1 &
#
# Status:
#   .venv/bin/python scripts/status_out_base.py "$out_base/train_mu"
#   .venv/bin/python scripts/status_out_base.py "$out_base/train_bh"
#
# Notes:
# - Uses `setsid taskset -c ...` per seed for robust detachment + core partitioning.
# - Runs 5 seeds in parallel (one per core-set), then repeats for BH baseline, then runs holdout eval.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/finalization/planck_lensing_holdout_prd_min}"
mkdir -p "$out_base"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python (create venv + install deps first)" >&2
  exit 2
fi

common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
  EHR_CAMB_CACHE_DIR="${EHR_CAMB_CACHE_DIR:-data/cache/camb}"
)

# "PRD-min" settings: shorter than the full strong run, but still big enough
# to support stable posterior predictive scoring (LPD) across multiple seeds.
common_args=(
  --mu-sampler ptemcee
  --pt-ntemps 4
  --pt-tmax 10
  --mu-steps 800
  --mu-burn 250
  --mu-draws 400
  --mu-walkers 64
  --mu-procs 0
  --checkpoint-every 50
  --include-rsd
  --rsd-mode dr16_lrg_fs8
  --include-fullshape-pk
  --skip-ablations
  --skip-hz-recon
  --gp-procs 1
)

seeds=(101 202 303 404 505)
coresets=("0-51" "52-102" "103-153" "154-204" "205-255")

active_pids_path="$out_base/pids_active.txt"
: > "$active_pids_path"

_launch_phase() {
  phase_name="$1"       # train_mu | train_bh
  fixed_mu_arg="${2:-}" # empty or "--mu-fixed bh"

  phase_dir="$out_base/$phase_name"
  mkdir -p "$phase_dir"

  pids_path="$phase_dir/pids.txt"
  : > "$pids_path"
  : > "$active_pids_path"

  echo "[launcher] phase=$phase_name out=$phase_dir"
  echo "[launcher] starting $((${#seeds[@]})) seeds..."

  pids=()
  for i in "${!seeds[@]}"; do
    s="${seeds[$i]}"
    c="${coresets[$i]}"
    out="$phase_dir/M0_start${s}"
    mkdir -p "$out"
    echo "[launcher] phase=$phase_name seed=$s cores=$c out=$out"

    env "${common_env[@]}" \
      setsid taskset -c "$c" \
      .venv/bin/python scripts/run_realdata_recon.py \
        --out "$out" \
        --seed "$s" \
        --mu-init-seed "$s" \
        --save-chain "$out/samples/mu_chain.npz" \
        ${fixed_mu_arg} \
        "${common_args[@]}" \
      > "$out/run.log" 2>&1 < /dev/null &

    pid="$!"
    pids+=("$pid")
    echo "$pid" >> "$pids_path"
    echo "$pid" >> "$active_pids_path"
    echo "[launcher] phase=$phase_name seed=$s pid=$pid"
  done

  echo "[launcher] phase=$phase_name launched; waiting..."

  # Wait for all seeds, but don't abort the entire script if one fails.
  failed=0
  for pid in "${pids[@]}"; do
    if wait "$pid"; then
      :
    else
      failed=$((failed+1))
    fi
  done

  echo "[launcher] phase=$phase_name done (failed=$failed)."
  return 0
}

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"

echo "[launcher] === Phase 1/3: train mu-free (no Planck lensing) ==="
_launch_phase "train_mu" ""

echo "[launcher] === Phase 2/3: train BH baseline (mu=1) ==="
_launch_phase "train_bh" "--mu-fixed bh"

echo "[launcher] === Phase 3/3: evaluate Planck lensing holdout (CAMB) ==="
eval_dir="$out_base/holdout_eval"
mkdir -p "$eval_dir"

echo "[launcher] holdout consext8..."
.venv/bin/python scripts/run_planck_lensing_holdout.py \
  --out "$eval_dir/consext8" \
  --dataset consext8 \
  --procs 256 \
  --seed 0 \
  --progress-every 100 \
  $(for s in "${seeds[@]}"; do echo --run-dir "$out_base/train_mu/M0_start${s}"; done) \
  $(for s in "${seeds[@]}"; do echo --run-dir "$out_base/train_bh/M0_start${s}"; done) \
  > "$eval_dir/consext8/run.log" 2>&1

echo "[launcher] holdout agr2..."
.venv/bin/python scripts/run_planck_lensing_holdout.py \
  --out "$eval_dir/agr2" \
  --dataset agr2 \
  --procs 256 \
  --seed 0 \
  --progress-every 100 \
  $(for s in "${seeds[@]}"; do echo --run-dir "$out_base/train_mu/M0_start${s}"; done) \
  $(for s in "${seeds[@]}"; do echo --run-dir "$out_base/train_bh/M0_start${s}"; done) \
  > "$eval_dir/agr2/run.log" 2>&1

echo "[launcher] all done."
exit 0


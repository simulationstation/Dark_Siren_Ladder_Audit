#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <out_base> [extra args...]" >&2
  exit 1
fi

out_base="$1"; shift
coresets=("0-9" "10-19" "20-29" "30-39" "40-49")
seeds=(101 202 303 404 505)
rsd_mode="${RSD_MODE:-dr12+dr16_fsbao}"
lens_arg="--include-lensing"
if [ "${NO_LENSING:-0}" -eq 1 ]; then
  lens_arg=""
fi

for i in "${!seeds[@]}"; do
  s="${seeds[$i]}"
  c="${coresets[$i]}"
  out="$out_base/M0_start${s}"
  mkdir -p "$out"
  setsid taskset -c "$c" \
    python3 scripts/run_realdata_recon.py \
      --out "$out" \
      --seed "$s" \
      --mu-init-seed "$s" \
      --mu-sampler ptemcee \
      --rsd-mode "$rsd_mode" \
      $lens_arg \
      --cpu-cores 0 \
      --mu-procs 0 \
      --gp-procs 0 \
      --save-chain "$out/samples/mu_chain.npz" \
      "$@" \
    > "$out/run.log" 2>&1 < /dev/null &
  echo "Started seed $s on cores $c (pid=$!)"
done

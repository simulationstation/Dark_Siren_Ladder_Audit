#!/usr/bin/env bash
set -euo pipefail

out_dir="${1:-outputs/multistart_info_plus}"
shift || true

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  NO_LENSING=1 \
  scripts/run_finalization_multistart.sh "$out_dir" \
    --mu-procs 10 \
    --gp-procs 1 \
    --include-rsd \
    --rsd-mode dr16_lrg_fs8 \
    --include-planck-lensing-clpp \
    --skip-ablations \
    --skip-hz-recon \
    "$@"

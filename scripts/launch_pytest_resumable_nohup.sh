#!/usr/bin/env bash
set -euo pipefail

# Resumable, step-tracked pytest runner in a single robust nohup job.
# Writes logs under outputs/pytest_resumable_<STAMP>/.

STAMP="$(date -u +%Y%m%d_%H%M%SUTC)"
OUT="outputs/pytest_resumable_${STAMP}"
mkdir -p "${OUT}"

LOG="${OUT}/nohup.log"
PIDFILE="${OUT}/pid.txt"

echo "[launcher] out=${OUT}"
echo "[launcher] log=${LOG}"

nohup bash -lc "
  cd \"$(pwd)\"
  . .venv/bin/activate
  echo \"[job] started \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
  echo \"[job] python=\$(which python)\"
  echo \"[job] pwd=\$(pwd)\"
  echo \"[job] writing pid $$ to ${PIDFILE}\"
  echo $$ > \"${PIDFILE}\"
  python scripts/run_pytest_resumable.py --out \"${OUT}\" --workers 1
  echo \"[job] finished \$(date -u +%Y-%m-%dT%H:%M:%SZ)\"
" > "${LOG}" 2>&1 &

echo "[launcher] background pid $! (wrapper)"
echo "[launcher] tail -f ${LOG}"


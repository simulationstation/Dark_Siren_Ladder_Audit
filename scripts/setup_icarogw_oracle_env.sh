#!/usr/bin/env bash
set -euo pipefail

# Create an isolated venv for ICAROGW.
#
# ICAROGW pins versions (e.g. bilby/matplotlib/ligo.skymap) that conflict with this repo's
# main `.venv`. Keeping a separate environment prevents dependency downgrades.
#
# Usage:
#   bash scripts/setup_icarogw_oracle_env.sh
#   bash scripts/setup_icarogw_oracle_env.sh oracles/icarogw

DEST_DIR="${1:-oracles/icarogw}"
PYTHON="${PYTHON:-python3}"
ICAROGW_REF="${ICAROGW_REF:-v2.0.2}"
ICAROGW_REPO="${ICAROGW_REPO:-https://github.com/simone-mastrogiovanni/icarogw}"
ICAROGW_SCIPY_CONSTRAINT="${ICAROGW_SCIPY_CONSTRAINT:-scipy<1.14}"

mkdir -p "${DEST_DIR}"

if [ ! -x "${PYTHON}" ]; then
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "[error] python not found on PATH: ${PYTHON}" >&2
    exit 2
  fi
fi

if [ ! -d "${DEST_DIR}/.venv" ]; then
  "${PYTHON}" -m venv "${DEST_DIR}/.venv"
fi

"${DEST_DIR}/.venv/bin/python" -m pip install -U pip setuptools wheel

echo "[icarogw] installing from ${ICAROGW_REPO}@${ICAROGW_REF} into ${DEST_DIR}/.venv"
echo "[icarogw] NOTE: pinning ${ICAROGW_SCIPY_CONSTRAINT} (ICAROGW uses scipy.integrate.{trapz,cumtrapz} which were removed in newer SciPy)."
"${DEST_DIR}/.venv/bin/pip" install "${ICAROGW_SCIPY_CONSTRAINT}" "icarogw @ git+${ICAROGW_REPO}@${ICAROGW_REF}"

echo "[icarogw] ok: ${DEST_DIR}/.venv/bin/python -c 'import importlib.metadata as m; print(m.version(\"icarogw\"))'"

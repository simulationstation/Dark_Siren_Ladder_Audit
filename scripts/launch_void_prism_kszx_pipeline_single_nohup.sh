#!/usr/bin/env bash
set -euo pipefail

# Robust detached launcher (single top-level job).
# Writes: <out>/job.sh, <out>/run.log, <out>/pid.txt

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date -u +%Y%m%d_%H%M%SUTC)"

OUT_DIR="${1:-${ROOT}/outputs/void_prism_kszx_pipeline_${STAMP}}"
THETA_DIR="${ROOT}/data/processed/void_prism/theta_tomo_kszx_full_${STAMP}"

CPUSET="${CPUSET:-0-$(($(nproc)-1))}"

mkdir -p "${OUT_DIR}"

cat > "${OUT_DIR}/job.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail

cd "${ROOT}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[job] started \$(date -u)"
echo "[job] out_dir=${OUT_DIR}"
echo "[job] theta_dir=${THETA_DIR}"

echo "[job] step 1/4: build kSZ-weighted tomographic theta maps (kszx-style filter)"
${ROOT}/.venv/bin/python scripts/build_theta_maps_tomo_from_act_dr6_sdss_kszx.py \\
  --act-dr 6 --act-freq 150 \\
  --sdss-surveys CMASSLOWZTOT_North,CMASSLOWZTOT_South --sdss-dr 12 \\
  --zmin 0.20 --zmax 0.70 --z-edges 0.2,0.36,0.48,0.56,0.67 \\
  --ksz-lmin 1500 --ksz-lmax 8000 \\
  --planck-galmask-sky-pct 70 --planck-galmask-apod-deg 0 \\
  --act-rms-threshold 70 \\
  --halo-zeff 0.55 --halo-ngal 1e-4 --halo-profile AGN \\
  --nside 256 --frame galactic \\
  --out-dir "${THETA_DIR}"

THETA_LIST=\$(ls -1 "${THETA_DIR}"/theta_kszx_act_dr6_f150_sdss_z*.fits | paste -sd, -)
TMASK_LIST=\$(ls -1 "${THETA_DIR}"/mask_kszx_act_dr6_f150_sdss_z*.fits | paste -sd, -)
echo "[job] theta_fits=\${THETA_LIST}"
echo "[job] theta_mask_fits=\${TMASK_LIST}"

echo "[job] step 2/4: measure void-prism suite + joint jackknife covariance"
SUITE_OUT="${OUT_DIR}/void_prism_eg_suite"
${ROOT}/.venv/bin/python scripts/measure_void_prism_eg_suite_jackknife.py \\
  --planck \\
  --void-csv data/processed/void_prism/boss_dr12_voids_mao2017.csv \\
  --theta-fits "\${THETA_LIST}" \\
  --theta-mask-fits "\${TMASK_LIST}" \\
  --z-edges 0.2,0.36,0.48,0.56,0.67 \\
  --rv-split median \\
  --bin-edges 0,20,50,100,200,400,700 \\
  --nside 256 --lmax 700 \\
  --jackknife-nside 8 --n-proc 256 \\
  --out-base "\${SUITE_OUT}"

SUITE_JSON="\${SUITE_OUT}/tables/suite_joint.json"

echo "[job] step 3/4: score vs info+ mu(A) posteriors (shape-only with per-draw amplitude fit)"
SCORE_OUT="${OUT_DIR}/void_prism_eg_joint"
${ROOT}/.venv/bin/python scripts/run_void_prism_eg_joint_test.py \\
  --suite-json "\${SUITE_JSON}" \\
  --run-dir outputs/finalization/info_plus_full_256_detached_20260129_0825UTC/M0_start101 \\
  --run-dir outputs/finalization/info_plus_full_256_detached_20260129_0825UTC/M0_start202 \\
  --run-dir outputs/finalization/info_plus_full_256_detached_20260129_0825UTC/M0_start303 \\
  --run-dir outputs/finalization/info_plus_full_256_detached_20260129_0825UTC/M0_start404 \\
  --run-dir outputs/finalization/info_plus_full_256_detached_20260129_0825UTC/M0_start505 \\
  --convention A \\
  --fit-amplitude \\
  --max-draws 5000 \\
  --out "\${SCORE_OUT}"

echo "[job] step 4/4: null sanity checks (rotate_voids + rotate_kappa)"
${ROOT}/.venv/bin/python scripts/run_void_prism_nulls.py \\
  --suite-json "\${SUITE_JSON}" \\
  --null-kind rotate_voids \\
  --n-null 50 \\
  --seed 123 \\
  --out-base "${OUT_DIR}/nulls_rotate_voids"

${ROOT}/.venv/bin/python scripts/run_void_prism_nulls.py \\
  --suite-json "\${SUITE_JSON}" \\
  --null-kind rotate_kappa \\
  --n-null 50 \\
  --seed 123 \\
  --out-base "${OUT_DIR}/nulls_rotate_kappa"

echo "[job] done \$(date -u)"
EOF

chmod +x "${OUT_DIR}/job.sh"

env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  setsid taskset -c "${CPUSET}" bash "${OUT_DIR}/job.sh" > "${OUT_DIR}/run.log" 2>&1 < /dev/null &

echo $! > "${OUT_DIR}/pid.txt"
echo "${OUT_DIR}"


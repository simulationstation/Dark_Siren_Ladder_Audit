#!/usr/bin/env bash
set -u -o pipefail

# End-to-end void-prism run using:
# - BOSS DR12 void catalog (Mao+ 2017)
# - Planck 2018 lensing kappa
# - Planck 2018 SMICA-noSZ temperature (bandpassed) as a kSZ-friendly theta proxy
#
# This is still NOT a full kSZ velocity reconstruction; it is a public-map upgrade over the
# 2MRS-velocity proxy and a step toward a true kappa+kSZ EG test.
#
# Run THIS script under a single nohup:
#
#   out_base="outputs/finalization/void_prism_boss_smica_$(date -u +%Y%m%d_%H%M%SUTC)"
#   mkdir -p "$out_base"
#   nohup bash scripts/launch_void_prism_boss_smica_single_nohup.sh "$out_base" > "$out_base/launcher.log" 2>&1 &
#
# Status:
#   cat "$out_base/pid.txt"
#   ps -p "$(cat "$out_base/pid.txt")" -o pid,pcpu,pmem,etime,cmd
#   tail -n 50 "$out_base/job.log"
#   tail -n 50 "$out_base/mask_build.log"
#   tail -n 50 "$out_base/suite/run.log"
#   tail -n 50 "$out_base/score/run.log"
#
# Tunables (env vars):
#   VP_Z_EDGES="0.25,0.35,0.45,0.55,0.60"
#   VP_BIN_EDGES="0,50,100,200,400,800,1200,1500"
#   VP_JK_NSIDE=2
#   VP_JK_THREADS=64
#   VP_SCORE_MAX_DRAWS=5000
#   VP_CPUSET="0-255"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

out_base="${1:-outputs/finalization/void_prism_boss_smica}"
mkdir -p "$out_base"

if [ ! -x ".venv/bin/python" ]; then
  echo "ERROR: missing .venv/bin/python" >&2
  exit 2
fi

common_env=(
  OMP_NUM_THREADS=1
  MKL_NUM_THREADS=1
  OPENBLAS_NUM_THREADS=1
  NUMEXPR_NUM_THREADS=1
  PYTHONUNBUFFERED=1
)

void_csv="data/processed/void_prism/boss_dr12_voids_mao2017.csv"
theta_fits="data/processed/void_prism/theta_planck_smica_nosz_20260130_072623UTC.fits"

if [ ! -f "$void_csv" ]; then
  echo "ERROR: missing void catalog: $void_csv" >&2
  exit 2
fi
if [ ! -f "$theta_fits" ]; then
  echo "ERROR: missing theta map: $theta_fits" >&2
  exit 2
fi

z_edges="${VP_Z_EDGES:-0.25,0.35,0.45,0.55,0.60}"
bin_edges="${VP_BIN_EDGES:-0,50,100,200,400,800,1200,1500}"
jk_nside="${VP_JK_NSIDE:-2}"
jk_threads="${VP_JK_THREADS:-64}"
score_max_draws="${VP_SCORE_MAX_DRAWS:-5000}"
coreset="${VP_CPUSET:-0-255}"

echo "[launcher] repo_root=$repo_root"
echo "[launcher] out_base=$out_base"
echo "[launcher] coreset=$coreset"
echo "[launcher] void_csv=$void_csv"
echo "[launcher] theta_fits=$theta_fits"
echo "[launcher] z_edges=$z_edges"
echo "[launcher] bin_edges=$bin_edges"
echo "[launcher] jk_nside=$jk_nside jk_threads=$jk_threads"
echo "[launcher] score_max_draws=$score_max_draws"

# IMPORTANT: robust detachment.
# The long-running work happens in a separate session via `setsid` so it survives SSH/tool disconnects.
mask_fits="$out_base/footprint_mask.fits"
suite_dir="$out_base/suite"
score_dir="$out_base/score"
job_log="$out_base/job.log"
job_sh="$out_base/job.sh"

mkdir -p "$suite_dir" "$score_dir"

cat > "$job_sh" <<EOF
#!/usr/bin/env bash
set -u -o pipefail
cd "$repo_root"

echo "[job] start \$(date -u)"
echo "[job] out_base=$out_base"
echo "[job] coreset=$coreset"

echo "[job] === 1/3: footprint mask ==="
env ${common_env[*]} \\
  .venv/bin/python scripts/build_void_footprint_mask.py \\
    --void-csv "$void_csv" \\
    --nside 512 \\
    --frame galactic \\
    --mode from_Rv \\
    --radius-factor 2.0 \\
    --dilate-deg 1.0 \\
    --out "$mask_fits" \\
  > "$out_base/mask_build.log" 2>&1

echo "[job] mask_fits=$mask_fits"

echo "[job] === 2/3: measure suite (jackknife) ==="
env ${common_env[*]} \\
  .venv/bin/python scripts/measure_void_prism_eg_suite_jackknife.py \\
    --void-csv "$void_csv" \\
    --theta-fits "$theta_fits" \\
    --planck \\
    --extra-mask-fits "$mask_fits" \\
    --frame galactic \\
    --nside 512 \\
    --lmax 1500 \\
    --bin-edges "$bin_edges" \\
    --prefactor 1.0 \\
    --eg-sign 1.0 \\
    --auto-eg-sign \\
    --z-edges "$z_edges" \\
    --rv-split median \\
    --min-voids 50 \\
    --jackknife-nside "$jk_nside" \\
    --n-proc "$jk_threads" \\
    --suite-name "void_prism_boss_smica_jk${jk_nside}" \\
    --out-base "$suite_dir" \\
  > "$suite_dir/run.log" 2>&1

suite_json="$suite_dir/tables/suite_joint.json"
if [ ! -f "\$suite_json" ]; then
  echo "ERROR: suite_joint.json missing: \$suite_json" >&2
  exit 2
fi
echo "[job] suite_json=\$suite_json"

echo "[job] === 3/3: score (minimal vs GR) ==="
run_base="outputs/finalization/info_plus_full_256_detached_20260129_0825UTC"
env ${common_env[*]} \\
  .venv/bin/python scripts/run_void_prism_eg_joint_test.py \\
    --suite-json "\$suite_json" \\
    --fit-amplitude \\
    --convention A \\
    --max-draws "$score_max_draws" \\
    --run-dir "\$run_base/M0_start101" \\
    --run-dir "\$run_base/M0_start202" \\
    --run-dir "\$run_base/M0_start303" \\
    --run-dir "\$run_base/M0_start404" \\
    --run-dir "\$run_base/M0_start505" \\
    --out "$score_dir" \\
  > "$score_dir/run.log" 2>&1

echo "[job] results_json=$score_dir/tables/results.json"
echo "[job] done \$(date -u)"
exit 0
EOF

chmod +x "$job_sh"

env "${common_env[@]}" \
  setsid taskset -c "$coreset" \
  bash "$job_sh" \
  > "$job_log" 2>&1 < /dev/null &

pid="$!"
echo "$pid" > "$out_base/pid.txt"
echo "[launcher] started pid=$pid"
echo "[launcher] job_log=$job_log"
echo "[launcher] job_sh=$job_sh"
exit 0

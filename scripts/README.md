# Scripts

- `run_realdata_recon.py`: end-to-end real-data pipeline + report
- `run_synthetic_closure.py`: generate mock data and validate recovery
- `run_ablation_suite.py`: kernel/prior/covariance/domain ablations
- `run_optical_bias_smoke.py`: optical-bias smoke test (mock or minimal)
- `run_optical_bias_realdata.py`: optical-bias real-data pipeline (Track A+B)
- `run_optical_bias_ablation.py`: optical-bias ablation suite
- `bench_lcdm_distance_cache.py`: compare our LCDM distance cache vs astropy
- `compare_1d_json_curves.py`: JSON curve comparator (e.g. H0-grid posteriors, alpha grids)
- `oracle_alpha_h0_wcosmo.py`: compute selection alpha(H0) using wcosmo distances (GWPopulation-style cosmology backend)
- `oracle_distance_cache_icarogw.py`: build an ICAROGW-based f(z) cache (run under `oracles/icarogw/.venv/`)
- `run_oracle_alpha_h0_compare.py`: compute+compare alpha(H0) using ours vs wcosmo vs ICAROGW
- `setup_icarogw_oracle_env.sh`: create an isolated ICAROGW venv under `oracles/icarogw/`

## Resource-safe defaults

All scripts set `OMP_NUM_THREADS=1` (and related BLAS/OpenMP env vars) to avoid nested parallelism.

Useful flags:
- `--cpu-cores N`: best-effort CPU affinity limiter (`0` = use all detected cores; respects pre-set task affinity masks for auto sizing)
- `--mu-procs N`, `--gp-procs N`, `--procs N`: multiprocessing workers (`0` = auto, capped at walkers)
- `--max-rss-mb MB`: per-process memory cap (best-effort; raises `MemoryError` instead of OOM)
- `run_synthetic_closure.py --fast`: minimal-cost settings for debugging (also skips the GP derivative ablation by default)

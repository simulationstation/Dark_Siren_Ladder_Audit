# Gate‑2 SBC A/B — injection `sampling_pdf` mass-frame convention (2026‑02‑05)

This note records a short A/B sweep that isolates a **major** source of Gate‑2 GR \(H_0\) **simulation‑based calibration (SBC)** failure for the “real‑data config” selection proxy.

## Context

We ran the closed‑loop suite:

- runner: `scripts/run_siren_injection_recovery_suite.py`
- injections: `data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3b.hdf5`
- IFAR threshold: `1.0 yr`
- truth policy: `H0_true ~ Uniform[40,200]` (SBC)
- inference grid: `H0 ∈ [40,200]` (161 points)
- `z_max=0.62`
- per replicate: `n_events=25`
- PE synthesis: `pe_obs_mode=noisy`, `pe_obs_condition_on_detection=True`, `pe_n_samples=5000`
- selection proxy: `det_model=snr_mchirp_q_binned` (SNR=200 bins, Mc_det=20 bins, q=10 bins)
- population (from LVK draw used in the corresponding real‑data run `lvk_0000`):
  - `pop_z_mode=comoving_powerlaw`, `pop_z_k=0`
  - `pop_mass_mode=powerlaw_peak_q_smooth`
  - `pop_m1_alpha=2.0454550190`
  - `pop_m_min=4.2961334406`, `pop_m_max=87.3621754365`
  - `pop_q_beta=0.8289290204`
  - `pop_m_taper_delta=4.4776656002`
  - `pop_m_peak=21.0116734620`, `pop_m_peak_sigma=9.2400689115`, `pop_m_peak_frac=0.4552193501`

All tests used `weight_mode=inv_sampling_pdf`, `inj_sampling_pdf_dist=log_dL`, `inj_sampling_pdf_mass_scale=linear`, `inj_mass_pdf_coords=m1m2`, varying only the **mass-frame convention** for the injection `sampling_pdf`.

## Results

### A) Baseline (interpreting `sampling_pdf` mass measure as **detector‑frame**)

Command output dir (quick 16 reps):
- `outputs/gate2_sbc_ab_baseline_20260205/`

Aggregate selection‑ON SBC metrics:
- `u_h0_on_mean ≈ 0.251`
- `coverage_68_on = 0.25`, `coverage_95_on = 0.50`
- `bias_p50_on_mean ≈ +8.28 km/s/Mpc`

This reproduces the full “real‑config” failure seen in:
- `outputs/gate2_closure_sbc_realconfig_lvk0000_20260205_021057UTC/` (128 reps; same convention)

### B) A/B flip (interpreting `sampling_pdf` mass measure as **source‑frame**)

Command output dir (quick 16 reps):
- `outputs/gate2_sbc_ab_massframe_source_20260205/`

Aggregate selection‑ON SBC metrics:
- `u_h0_on_mean ≈ 0.448`
- `coverage_68_on = 0.4375`, `coverage_95_on = 0.8125`
- `bias_p50_on_mean ≈ +1.55 km/s/Mpc`

Higher‑stat run (128 reps; close to the original suite size):
- `outputs/gate2_closure_sbc_realconfig_massframe_source_20260205_025458UTC/`

Aggregate selection‑ON SBC metrics (128 reps):
- `u_h0_on_mean ≈ 0.423`
- `coverage_68_on ≈ 0.391`, `coverage_95_on ≈ 0.719`
- `bias_p50_on_mean ≈ +1.60 km/s/Mpc`

## Takeaway

For this injection file + Gate‑2 selection proxy, interpreting `injections/sampling_pdf` as a density in **detector‑frame linear component masses** (and applying the corresponding \((1+z)\) Jacobians in `inv_sampling_pdf` weighting) produces a large, repeatable SBC failure.

Switching to **source‑frame** mass interpretation is a **first‑order** improvement and should be treated as the default choice for subsequent Gate‑2 calibration/selection work with this injection file (unless an explicit upstream definition proves otherwise).

## Still not “passed”

Even with the corrected mass-frame convention, the selection‑ON SBC is **not yet nominal** over \(H_0\in[40,200]\) (coverage is low and \(u\) is not uniform). This indicates at least one additional mismatch remains (population+selection bookkeeping, proxy limitations, or generator/inference mismatch) that needs further isolation.

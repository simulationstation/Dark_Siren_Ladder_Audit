# Gate‑2 SBC / calibration note (2026‑02‑04)

This short note documents a **simulation-based calibration (SBC)** style diagnostic for the **Gate‑2 GR \(H_0\)** control, and what it implies about where the current “Gate‑2 feels broken” behavior actually comes from.

## What was tested

We ran the injection-recovery suite while sampling the truth \(H_0^\star\) from the same implicit prior used by the grid posterior (uniform on the grid):

- runner: `scripts/run_siren_injection_recovery_suite.py`
- new mode: `--h0-true-mode uniform`

For each replicate we compute:

- a GR selection-on \(p(H_0\mid d, \mathrm{det})\) on an \(H_0\) grid, and
- the SBC-style statistic
  \[
  u \equiv \mathrm{CDF}_{p(H_0\mid d)}(H_0^\star).
  \]

If the full closed-loop generator + inference are self-consistent and \(H_0^\star\sim \mathrm{Uniform}(H_{0,\min},H_{0,\max})\), then **\(u\) should be approximately Uniform\([0,1]\)** across replicates.

## Key results

All runs below used:
- O3b sensitivity injections: `data/cache/gw/zenodo/7890437/endo3_mixture-LIGO-T2100113-v12-1256655642-12905976.hdf5`
- \(H_0\in[40,100]\) with 121 grid points, `z_max=0.62`
- 64 replicates × 25 detected events/replicate
- `pe_synth_mode=likelihood_resample`, `pe_obs_mode=noisy`, `pe_n_samples=12000`

### A) Population weights ON (typical Gate‑2 config)

Config (selection proxy and inference weights both include population assumptions):
- `pop_z_mode=comoving_uniform`
- `pop_mass_mode=powerlaw_peak_q_smooth`
- `det_model=snr_mchirp_binned`

Output:
- `outputs/gate2_suite_sbc_uniform_fixedz062_20260204_030037UTC/tables/suite_aggregate.json`

Observed:
- `u_h0_off_mean ≈ 0.113` and `u_h0_off_ks ≈ 0.645` (very non-uniform; this is the expected “ignore-selection” bias for detected-only samples)
- `u_h0_on_mean ≈ 0.350` and `u_h0_on_ks ≈ 0.243` (KS p-value \(\sim 8\times 10^{-4}\); strongly non-uniform)
- indicates the selection-on posterior tends to place \(H_0^\star\) in the *lower* CDF tail (i.e. the posterior is typically shifted **high** relative to truth under this broad truth prior)

Figures copied into git:
- `2-3-G/figures/gate2_sbc_u_h0_on_cdf_popZmass.png`
- `2-3-G/figures/gate2_sbc_u_h0_on_hist_popZmass.png`

### B) Population weights OFF (diagnostic null)

Config:
- `pop_z_mode=none`
- `pop_mass_mode=none`
- `det_model=snr_binned`

Output:
- `outputs/gate2_suite_sbc_uniform_fixedz062_popNone_det1D_20260204_031452UTC/tables/suite_aggregate.json`

Observed:
- `u_h0_off_mean ≈ 0.061` and `u_h0_off_ks ≈ 0.737` (again: strong ignore-selection bias)
- `u_h0_on_mean ≈ 0.502` and `u_h0_on_ks ≈ 0.105` (much closer to Uniform\([0,1]\))
- BUT many boundary-peaked posteriors (`H0_map_at_edge_on_n` large), i.e. this is *not* a viable science configuration; it’s a **sanity check** showing the SBC machinery itself isn’t broken.

Figures copied into git:
- `2-3-G/figures/gate2_sbc_u_h0_on_cdf_noPop.png`
- `2-3-G/figures/gate2_sbc_u_h0_on_hist_noPop.png`

## Interpretation (why Gate‑2 “takes forever”)

This pattern strongly suggests:

1) The Gate‑2 **GR(H0) control is not “universally broken”**; the core machinery can be close to calibrated in simplified settings.
2) The **population-conditioning layer** (redshift prior and/or mass prior in combination with selection weighting) is where calibration failures show up when we stress the model across a wide \(H_0^\star\) range.

That is consistent with the project’s central thesis: in spectral-only dark-siren inference, **selection/population bookkeeping dominates** and can manufacture apparent preferences unless closed-loop checks are explicit.

## Most likely actionable next step

### Follow‑up: was `sampling_pdf` convention the culprit?

We implemented a “convention adapter” for the injection `sampling_pdf` interpretation and tested whether missing Jacobians / wrong coordinate measures are a major driver of the Gate‑2 SBC failure when population weights are ON.

Code changes (defaults preserve current behavior):
- `inj_sampling_pdf_dist ∈ {z, dL, log_dL}`: treat `sampling_pdf` as a density in z (default), luminosity distance, or log-luminosity-distance (apply the corresponding `dL/dz` + `1/dL` Jacobians).
- `inj_sampling_pdf_mass_frame ∈ {source, detector}`: treat `sampling_pdf` as a density in source-frame masses (default) or detector-frame masses.
- `inj_sampling_pdf_mass_scale ∈ {linear, log}`: treat `sampling_pdf` as a density in linear masses (default) or in log-mass coordinates.

We ran a **12‑way sweep** at fixed settings (`pop_z_mode=comoving_uniform`, `pop_mass_mode=powerlaw_peak_q_smooth`, `det_model=snr_mchirp_binned`, `weight_mode=inv_sampling_pdf`):
- Output dir: `outputs/gate2_injpdf_sweep_20260204_074041UTC`
- Screening: 16 replicates × 25 events/replicate, \(H_0\in[40,100]\) (121 grid points), `z_max=0.62`, `pe_n_samples=2000`
- Confirm: 256 replicates × 25 events/replicate, \(H_0\in[40,100]\) (121 grid points), `z_max=0.62`, `pe_n_samples=4000`

Aggregate SBC metrics (selection‑ON; lower KS statistic is better; Uniform would have mean ~0.5 and small KS):

| config | inj_sampling_pdf_dist | inj_sampling_pdf_mass_frame | inj_sampling_pdf_mass_scale | u_h0_on_mean | u_h0_on_ks |
|---|---|---|---|---:|---:|
| default | z | source | linear | 0.3589 | 0.1919 |
| best (confirm) | log_dL | detector | linear | 0.4356 | 0.1085 |
| (worse) | dL | source | linear | 0.2855 | 0.3258 |

Conclusion:
- `sampling_pdf` convention **does matter** for the calibrated *injection-recovery* Gate‑2 SBC diagnostic; the `log_dL`+`detector`+`linear` adapter improves the SBC uniformity substantially in this configuration.
- However it **does not “solve Gate‑2”** by itself (the suite still shows noticeable deviations from Uniform under mass+z population conditioning).
- On *real data* Gate‑2 (spectral-only control), switching the injection convention changes the inferred `alpha(H0)` curve but does **not** materially change the posterior in the current configuration:
  - new run: `outputs/gate2_realdata_injpdf_logdL_det_20260204_075347UTC/json/gr_h0_selection_on_inv_sampling_pdf.json`
  - baseline (defaults): `outputs/gate2_realdata_fixedzmax_20260203_190502UTC/fixed_wide/json/gr_h0_selection_on_inv_sampling_pdf.json`

### Weight‑mode sanity check

We also verified that turning off injection sampling-pdf correction is not viable:
- Output dir: `outputs/gate2_sbc_weightmode_20260204_044617UTC`
- `weight_mode=none` gives extremely non‑uniform SBC (`u_h0_on_mean≈0.84`, `u_h0_on_ks≈0.58`).

So `weight_mode=inv_sampling_pdf` is required for any serious selection-corrected work.

## Updated next step (what to debug now)

Since the `sampling_pdf` convention alone doesn’t fix the mass+z SBC failures (and doesn’t move the current real‑data posterior), the remaining likely sources of Gate‑2 “feels broken” behavior are:

- **Selection vs population bookkeeping**: whether an \(H_0^{-3}\) “volume scaling” factor is applied consistently (or double-counted / omitted) between the z prior and the selection term.
- **Event-term vs selection-term mismatch**: SBC should be decomposed into “selection‑off” vs “selection‑on” (logL_data vs logL_total) to see which part induces bias.
- **Detectability proxy limitations**: miscalibrated `p_det(SNR, Mchirp_det)` shape can dominate totals even when α(H0) matches external oracles.

Concrete immediate work item:
- extend the suite aggregate + figures to report both `u_h0_off_*` (selection‑off) and `u_h0_on_*` so we can localize the miscalibration to the event term or the selection term (**done**: `scripts/run_siren_injection_recovery_suite.py` now writes `u_h0_off_*` to `tables/suite_aggregate.json` and plots `figures/sbc_u_h0_off_{cdf,hist}.png`).

## Population ablation (z vs mass) — what actually drives the non‑uniformity

We ran a small 32‑replicate ablation sweep holding the rest of the Gate‑2 control fixed (same injection file, same H0 grid, same detection proxy, same PE synthesis settings) while toggling **which population assumptions are enabled**:

- Output bundle: `outputs/gate2_sbc_ablation_popweights_20260204_053915UTC`
- 32 replicates × 25 events/replicate, \(H_0\in[40,100]\) (121 grid points), `det_model=snr_mchirp_binned`

Selection‑ON SBC metrics:

| case | pop_z_mode | pop_mass_mode | u_h0_on_mean | u_h0_on_ks |
|---|---|---|---:|---:|
| popNone | none | none | 0.449 | 0.164 |
| zOnly | comoving_uniform | none | 0.385 | 0.275 |
| mOnly | none | powerlaw_peak_q_smooth | 0.255 | 0.368 |
| popZmass | comoving_uniform | powerlaw_peak_q_smooth | 0.332 | 0.337 |

Conclusion:
- The **mass population conditioning** is the dominant source of the Gate‑2 SBC failure: turning on `pop_mass_mode` is what makes \(u\) collapse toward 0 (posterior pushed **high** relative to truth).
- The redshift prior (`pop_z_mode`) also matters, but its impact is smaller than the mass term in this control.

### Follow‑up checks on the “mass” culprit

1) **Injection mass-coordinate convention** (`inj_mass_pdf_coords`) does not rescue calibration.
   - Output bundle: `outputs/gate2_sbc_ablation_injmasscoords_20260204_054145UTC`
   - Switching `m1m2 → m1q` does not improve `u_h0_on_*` in mass-enabled cases (often slightly worse).

2) **Importance-weight smoothing** (`importance_smoothing=truncate`) does not change the result.
   - Output bundle: `outputs/gate2_sbc_ablation_smoothing_20260204_054359UTC`
   - `truncate` vs `none` gives identical `u_h0_on_*` in the tested cases, suggesting this is not a simple heavy-tail/ESS stabilization issue.

## Volume-scaling bookkeeping (pop z vs selection term)

We ran a 2×2 matrix toggling the optional \(H_0^{-3}\) “volume scaling” factor in:

- the **population z prior** inside the per-event term (`--pop-z-include-h0-volume-scaling`), and
- the **selection normalization** (`--selection-include-h0-volume-scaling`).

Output bundle:
- `outputs/gate2_sbc_volume_scaling_matrix_20260204_062145UTC`

Aggregate selection‑ON SBC metrics:

| case | pop_z_include_h0_volume_scaling | selection_include_h0_volume_scaling | u_h0_on_mean | u_h0_on_ks | note |
|---|---:|---:|---:|---:|---|
| `popz0_sel0` | 0 | 0 | 0.3349 | 0.2765 | baseline |
| `popz1_sel0` | 1 | 0 | 0.9955 | 0.9655 | catastrophic (posterior pins to low edge) |
| `popz0_sel1` | 0 | 1 | 0.0352 | 0.8554 | catastrophic (posterior pins to high edge) |
| `popz1_sel1` | 1 | 1 | 0.3349 | 0.2765 | cancels back to baseline |

Interpretation:
- If the volume scaling is included in **both** places, it cancels (up to constants) and you get the same behavior as baseline.
- If it is included in **only one** place, Gate‑2 becomes violently unstable (edge‑peaked posteriors), as expected for a bookkeeping mismatch.
- This strongly suggests our **remaining SBC failure is not “fixed” by flipping this knob**; rather, **mismatches here are a known failure mode to avoid**.

## PE Monte Carlo size (does more PE sampling fix SBC?)

We tested whether the selection‑ON SBC failure is simply due to Monte Carlo noise from too few PE samples per event.

Output bundle:
- `outputs/gate2_sbc_pe_samples_test_20260204_064152UTC/pe50k`

Setup differences vs baseline:
- `pe_n_samples=50,000` (vs 12,000)
- `n_rep=128` (vs 256)

Result (selection‑ON SBC):
- baseline (`popz0_sel0`): `u_h0_on_mean≈0.3349`, `u_h0_on_ks≈0.2765`
- `pe50k`: `u_h0_on_mean≈0.3268`, `u_h0_on_ks≈0.2578`

Conclusion:
- Increasing PE samples by \(\sim 4\times\) **does not materially restore Uniform\([0,1]\)** behavior for \(u\).
- It can reduce some “edge‑MAP” artifacts, but the core calibration failure persists.

## Mass-measurement strength (ablation)

We also tested whether making mass posteriors “much stronger” or “much weaker” changes SBC under the typical popZ+mass config.

Output bundle:
- `outputs/gate2_sbc_mass_noise_matrix_20260204_062643UTC`

Selection‑ON SBC metrics:

| case | (mc_frac_sigma0, q_sigma0) | u_h0_on_mean | u_h0_on_ks |
|---|---|---:|---:|
| `mass_baseline` | (0.02, 0.08) | 0.3458 | 0.2372 |
| `mass_weak` | (0.05, 0.15) | 0.3271 | 0.2754 |
| `mass_strong` | (0.005, 0.02) | 0.3556 | 0.2377 |

Conclusion:
- The Gate‑2 SBC failure is **not primarily driven by “mass posteriors too strong”** in this synthetic setup.

## New diagnostic: term‑slope decomposition (to avoid endless knob‑turning)

We added a cheap monotonicity diagnostic to the suite runner:

- per‑replicate slopes vs \(\log H_0\) for:
  - event term \(\sum_i \log L_i(H_0)\) (selection‑OFF)
  - selection term \(\log \alpha(H_0)\)
  - total selection‑ON term \(\sum_i \log L_i(H_0) - N \log \alpha(H_0)\)
- new suite artifacts:
  - `tables/suite_summary.csv` columns: `slope_logLsum_off`, `slope_log_alpha_on`, `slope_logL_total_on`
  - `tables/suite_aggregate.json` keys: `slope_*_mean`, `slope_*_frac_pos`
  - figures: `figures/slope_logLsum_off_hist.png`, `figures/bias_vs_slope_logLsum_off.png`

In the baseline `popz0_sel0` run (above), these aggregates show:
- `slope_logLsum_off_mean ≈ +0.76` (event term prefers higher \(H_0\))
- `slope_log_alpha_on_mean ≈ +0.016` (so \(-N\log\alpha\) prefers lower \(H_0\), partially cancelling)
- `slope_logL_total_on_mean ≈ +0.15` (net preference still drifts to higher \(H_0\))

This matches the qualitative SBC symptom: \(u\) collapses toward 0 because the selection-on posterior is still typically shifted high relative to truth.

## Toy closed-loop sanity check (no injection file)

To isolate whether the remaining SBC failure is coming from the **event-term bookkeeping** (hierarchical PE reweighting + mass/z Jacobians) or from the **injection-weighting layer** (interpreting `sampling_pdf`, etc.), we ran a pure toy generator that:

- samples \(z\) and source-frame masses directly from the same population family used in the inference,
- applies a simple, known \(p_\mathrm{det}(\mathrm{SNR})\) (logistic) to generate detected events, and
- computes \(\alpha(H_0)\) via Monte Carlo under the same toy model (no injection file).

Output bundle:
- `outputs/gate2_toy_sbc_isolation_20260204_071440UTC`

Common settings:
- 256 replicates × 25 detected events/replicate
- \(H_0^\star\sim \mathrm{Uniform}[40,100]\) (121 grid points)
- `pe_n_samples=2000` (synthetic PE), `alpha_n_mc=200000` (toy alpha MC)

Toy SBC aggregates:

| case | pop_mass_mode | u_h0_on_mean | u_h0_on_ks |
|---|---|---:|---:|
| `powerlaw_q` | powerlaw_q | 0.5501 | 0.0938 |
| `peak_smooth` | powerlaw_peak_q_smooth | 0.5554 | 0.0879 |

Interpretation:
- The toy suites are **much closer to calibrated** than the injection‑file suites under the same broad prior stress test (which produce \(u\) means \(\sim 0.33\) under popZ+mass).
- This strongly points to the remaining “Gate‑2 SBC feels broken” behavior being dominated by the **injection-weighting / selection-proxy layer** (i.e. how we use `sampling_pdf` + injections), not a fundamental bug in the hierarchical PE event term.

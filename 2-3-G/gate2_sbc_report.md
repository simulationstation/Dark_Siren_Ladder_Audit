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

We implemented a “convention adapter” for the injection `sampling_pdf` interpretation and tested whether a missing Jacobian is responsible for the SBC failure when population weights are ON.

Code changes (defaults preserve current behavior):
- `inj_sampling_pdf_dist ∈ {z, dL}`: treat `sampling_pdf` as a density in z (default) or luminosity distance (apply a `1/(dL/dz)` Jacobian).
- `inj_sampling_pdf_mass_frame ∈ {source, detector}`: treat `sampling_pdf` as a density in source-frame masses (default) or detector-frame masses (apply a `1/(1+z)^k` Jacobian with `k` depending on mass coordinates).

We ran a **4‑way matrix** at fixed settings (`pop_z_mode=comoving_uniform`, `pop_mass_mode=powerlaw_peak_q_smooth`, `det_model=snr_mchirp_binned`, `weight_mode=inv_sampling_pdf`):
- Output dir: `outputs/gate2_sbc_matrix_20260204_043303UTC`
- 32 replicates × 25 events/replicate, \(H_0\in[40,100]\) (121 grid points), `z_max=0.62`

Aggregate SBC metrics (lower KS statistic is better; Uniform would have mean ~0.5 and small KS):

| label | inj_sampling_pdf_dist | inj_sampling_pdf_mass_frame | u_h0_on_mean | u_h0_on_ks |
|---|---|---|---:|---:|
| A (default) | z | source | 0.34696 | 0.25702 |
| B | dL | source | 0.30871 | 0.31860 |
| C | z | detector | 0.21804 | 0.46242 |
| D | dL | detector | 0.20795 | 0.43914 |

Conclusion:
- The “missing Jacobian / wrong sampling_pdf coordinates” hypothesis **does not fix** Gate‑2 SBC under population weighting.
- In this proxy, the injection file is **most consistent with** `sampling_pdf` already being in **(z, source-frame masses)** (because alternative interpretations make SBC worse).

### Weight‑mode sanity check

We also verified that turning off injection sampling-pdf correction is not viable:
- Output dir: `outputs/gate2_sbc_weightmode_20260204_044617UTC`
- `weight_mode=none` gives extremely non‑uniform SBC (`u_h0_on_mean≈0.84`, `u_h0_on_ks≈0.58`).

So `weight_mode=inv_sampling_pdf` is required for any serious selection-corrected work.

## Updated next step (what to debug now)

Since the sampling‑pdf convention isn’t the root cause, the remaining likely sources of Gate‑2 “feels broken” behavior are:

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

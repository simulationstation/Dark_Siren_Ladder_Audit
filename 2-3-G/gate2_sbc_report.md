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

The most plausible technical culprit consistent with the SBC behavior is **injection reweighting conventions** under `weight_mode=inv_sampling_pdf`:

- The O3 sensitivity injection file provides `sampling_pdf` alongside **both** source-frame and detector-frame masses and both `distance` and `redshift`.
- If `sampling_pdf` is defined in **detector-frame masses and luminosity distance**, then importance weights that target a population model written in **source-frame masses and redshift** require additional Jacobians (beyond the already-implemented \((m_1,m_2)\to(m_1,q)\) factor).
- Those Jacobians only matter when `pop_z_mode != none` and/or `pop_mass_mode != none`, matching the observed failure mode.

Concrete work item:
- audit `sampling_pdf` coordinate conventions and implement the missing Jacobian(s) in the injection-weight path used by:
  - `src/entropy_horizon_recon/dark_siren_h0.py::_injection_weights`
  - `src/entropy_horizon_recon/siren_injection_recovery.py::generate_synthetic_detected_events_from_injections`


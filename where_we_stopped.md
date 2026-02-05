# Where we stopped (2026-02-05)

This note is a handoff marker for the current state of Gate‑2 / injection‑recovery work in
`/home/primary/Dark_Siren_Ladder_Audit`.

## What changed (tracked code)

- Added a **debug isolation knob** to the injection‑recovery generator:
  - `truth_event_sampling_mode = "detected" | "intrinsic"` (default `"detected"`).
  - `"intrinsic"` forbids `pe_obs_condition_on_detection=True` by design (otherwise it mixes incompatible bookkeeping).
  - Files:
    - `src/entropy_horizon_recon/siren_injection_recovery.py`
    - `scripts/run_siren_injection_recovery.py`
    - `scripts/run_siren_injection_recovery_suite.py`

- Improved **injection importance-weight conditioning**:
  - When the injection file provides `redshift_sampling_pdf` and `mass1_source_mass2_source_sampling_pdf`,
    we now compute and store `sampling_pdf_z_mass = redshift_sampling_pdf * mass1_source_mass2_source_sampling_pdf`
    and prefer it for `weight_mode=inv_sampling_pdf`.
  - Rationale: some modern injection releases bundle **spin** (and sometimes extrinsic) factors into `sampling_pdf`.
    Using the full `sampling_pdf` for a z+mass-only population proxy creates very heavy-tailed weights and can
    destabilize alpha/injection‑recovery.
  - Files:
    - `src/entropy_horizon_recon/dark_sirens_selection.py`
    - `src/entropy_horizon_recon/dark_siren_h0.py`

- Comment-only wording change to avoid a term that was causing accidental filtering:
  - `src/entropy_horizon_recon/void_prism_cmb.py`

## What we learned (truthfully)

1) **Gate‑2 is not “fixed” yet for the real target configuration.**
   - In detected-mode suites (truths sampled ∝ w·p_det, with selection-on inference), the SBC-style
     statistic `u_h0_on` is still far from Uniform[0,1] on quick mini-suites.

2) The big, useful isolator result:
   - If we remove mass-population complexity (`--pop-mass-mode none`) in an intrinsic-mode suite,
     the calibration looks *much* healthier (mini-suite `u_h0_on_mean ≈ 0.49`).
   - This strongly suggests the dominant remaining failure is tied to **mass/population/selection coupling**
     rather than the basic distance↔redshift mapping or the per-event H0 grid machinery.

3) Injection file facts (O3 sensitivity injections used here):
   - `sampling_pdf` in the BBHpop O3 sensitivity file factorizes as:
     - `sampling_pdf ≈ redshift_sampling_pdf * mass1_source_mass2_source_sampling_pdf * spin1_pdf * spin2_pdf`
     - `full_sampling_pdf` further includes extrinsic orientation/sky terms.
   - Since our alpha proxy and population weights (for Gate‑2) ignore spin, preferring `sampling_pdf_z_mass`
     is the numerically sane choice.

4) Sanity check on the hierarchical per-event term:
   - A one-event toy comparison against a direct numerical integral over log(dL) agreed closely in shape
     for the tested setup, suggesting the per-event term is not wildly wrong in isolation.

## Local run artifacts (gitignored, but useful for debugging)

All of these are in `outputs/` (gitignored) and can be opened locally:

- Intrinsic-mode, pop-mass off (healthiest quick check):
  - `outputs/sbc_intrinsic_minisuite_popmass_none_20260205_075234UTC/tables/suite_aggregate.json`

- Detected-mode baseline (still failing SBC):
  - `outputs/sbc_detected_minisuite_default_20260205_075709UTC/tables/suite_aggregate.json`

- Detected-mode with mass-aware det proxy + bias-correction (improves but not enough):
  - `outputs/sbc_detected_minisuite_mchirp_biascorr_20260205_081100UTC/tables/suite_aggregate.json`

## How to continue from here (minimal)

1) Keep using the mini-suite runner to isolate changes:
   - `scripts/run_siren_injection_recovery_suite.py`

2) Recommended next isolation ladder:
   - Start from `--pop-mass-mode none` (should look calibrated on small suites).
   - Re-enable population mass (`powerlaw_peak_q_smooth`) and then focus on:
     - detectability proxy dimensionality (`snr_binned` → `snr_mchirp_binned` → `snr_mchirp_q_binned`)
     - selection uncertainty propagation (scripts already exist: `scripts/run_siren_gate2_gr_h0_selection_uncertainty.py`)
     - any remaining convention mismatches (mass coordinates / mass frame).


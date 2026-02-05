# Dark_Siren_Ladder_Audit

This repository is a focused working copy of the **Dark Siren “Ladder Audit”** project extracted from the larger `PROJECT/` workspace.

- Start here: `Status.md`
- Run discipline / resumability: `AUDIT.md` + `AGENTS.md`

**Note:** Large local artifacts (GW PE files, injections, outputs, report bundles) are present on disk under `data/`, `outputs/`, and `2-1-*`, but are intentionally gitignored.

## Quick start

```bash
python -m venv .venv
./.venv/bin/pip install -U pip
./.venv/bin/pip install -e '.[sirens,dev,oracles]'
```

## Oracle / benchmark tooling (optional)

This repo includes **oracle-style cross-check runners** so we can sanity-check selection/normalization pieces (e.g. `alpha(H0)`) against best-available community tooling **without changing the audit pipeline’s goals**.

- In the main `.venv`: `gwpopulation` + `wcosmo` (install via the `oracles` extra).
- In an isolated venv (dependency-pinned): `icarogw` under `oracles/icarogw/.venv/`:

```bash
bash scripts/setup_icarogw_oracle_env.sh
```

Then you can compare `alpha(H0)` computed with our distance cache vs wcosmo vs ICAROGW:

```bash
./.venv/bin/python scripts/run_oracle_alpha_h0_compare.py \
  --injection-file data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3b.hdf5 \
  --plot
```

### Injection `sampling_pdf` convention adapters

Some injection releases define `sampling_pdf` in coordinates other than `(z, m_source)` (e.g. `log(dL)` and/or detector-frame masses). Many of the Gate‑2 audit runners support explicit adapters:

- `--inj-sampling-pdf-dist {z,dL,log_dL}`
- `--inj-sampling-pdf-mass-frame {source,detector}`
- `--inj-sampling-pdf-mass-scale {linear,log}`

For the O3 sensitivity mixture injection file above (`o3_sensitivity_mixture_o3b.hdf5`), A/B SBC sweeps show **strong sensitivity** to the injection `sampling_pdf` mass-measure convention. The current best-performing convention in closed loop is:

```bash
--inj-sampling-pdf-dist log_dL --inj-sampling-pdf-mass-frame source --inj-sampling-pdf-mass-scale linear
```

Using `--inj-sampling-pdf-mass-frame detector` for this file can produce large, repeatable Gate‑2 SBC failures on wide \(H_0\) grids; see `FINDINGS/gate2_sbc_injpdf_massframe_ab_20260205.md`.

For the GWTC‑3 population injection file (`data/cache/gw/zenodo/11254021/.../injections/o3a_bbhpop_inj_info.hdf`), closed-loop SBC indicates the `sampling_pdf` mass measure behaves like **log-mass coordinates**, i.e. improved calibration using:

```bash
--inj-sampling-pdf-dist log_dL --inj-sampling-pdf-mass-scale log
```

## Gate‑2 population anchoring (recommended)

The GR \(H_0\) selection-on control is sensitive to population assumptions. For a “best available” baseline, you can marginalize over **LVK GWTC‑3 population hyperposterior draws** (PowerLaw+Peak + powerlaw redshift), using the O3a population data release bilby-result JSON:

```bash
./.venv/bin/python scripts/run_siren_gate2_gr_h0_popmarginal.py \
  --pop-mode lvk_hyper \
  --lvk-pop-result-json data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/o3a_population_data_release/Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json \
  --lvk-n-draws 200
```

## Hubble tension investigator (audit-mode)

Given any Gate‑2 JSON output, you can produce a concise “tension-style” summary (threshold probabilities + an optional Planck-like vs SH0ES-like Bayes factor under Gaussian priors):

```bash
./.venv/bin/python scripts/hubble_tension_report.py outputs/<RUN>/json/gr_h0_selection_on_inv_sampling_pdf.json
```

To propagate a basic finite-injection uncertainty model for the selection factor \(\alpha(H_0)\) (via Beta draws over the binned \(p_{\rm det}\) table) into the GR \(H_0\) posterior:

```bash
./.venv/bin/python scripts/run_siren_gate2_gr_h0_selection_uncertainty.py \
  --gate2-json outputs/<RUN>/json/gr_h0_selection_on_inv_sampling_pdf.json \
  --selection-injections-hdf data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3b.hdf5 \
  --save-alpha-draws-npz
```

## Run log (GR vs mu)

Many siren tests in this repo score **mu (entropy-modified)** vs **GR** using a total ΔLPD metric. To keep comparisons auditable across many detached runs, build a single CSV from finished output folders:

```bash
./.venv/bin/python scripts/build_run_log.py --outputs outputs --out FINDINGS/run_log.csv
```

## Gate‑2 diagnostics jig

To diagnose why Gate‑2 \(H_0\) posteriors slam into grid edges, use the Gate‑2 jig to compute per-term slopes and per-event “effective \(H_0\) exponents” from an existing `gr_h0_selection_*.json`:

```bash
./.venv/bin/python scripts/gate2_jig.py one \
  --json outputs/<RUN>/json/gr_h0_selection_on_inv_sampling_pdf.json \
  --top-n 10
```

Interpretation note (real data): `2-3-G/gate2_realdata_diagnosis.md`

Or scan all Gate‑2 outputs into a summary CSV:

```bash
./.venv/bin/python scripts/gate2_jig.py scan --outputs outputs --out FINDINGS/gate2_jig.csv
```

Or recompute the posterior after dropping/thresholding events (useful to diagnose single-event dominance):

```bash
./.venv/bin/python scripts/gate2_jig.py filter \
  --json outputs/<RUN>/json/gr_h0_selection_on_inv_sampling_pdf.json \
  --out FINDINGS/gate2_filter.csv
```

Or run a leave-one-out influence scan (which events move the posterior the most):

```bash
./.venv/bin/python scripts/gate2_jig.py jackknife \
  --json outputs/<RUN>/json/gr_h0_selection_on_inv_sampling_pdf.json \
  --out FINDINGS/gate2_jackknife.csv
```

## Gate‑2 ladder (toy cancellation + incremental complexity)

To avoid “eternal knob turning”, `scripts/run_siren_gate2_ladder.py` runs:
- a **known-answer toy cancellation check** (prior-only PE + p_det in numerator) and
- a small **injection-recovery ladder** that adds complexity rung-by-rung (z prior, mass model, weight smoothing, detection model).

```bash
./.venv/bin/python scripts/run_siren_gate2_ladder.py --n-events 25 --pe-n-samples 5000 --n-proc 0
```

Outputs include `outputs/siren_gate2_ladder_*/ladder.csv` (quick scan) and `ladder_summary.json` (machine-readable).

To run a larger *resumable* injection-recovery suite and add an SBC-style check on the GR \(H_0\) control, use:

```bash
./.venv/bin/python scripts/run_siren_injection_recovery_suite.py \
  --selection-injections-hdf data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3b.hdf5 \
  --h0-min 40 --h0-max 100 --h0-n 121 \
  --h0-true-mode uniform \
  --n-events 25 --n-rep 64
```

This writes `tables/suite_aggregate.json` with `u_h0_on_*` and produces `figures/sbc_u_h0_on_{cdf,hist}.png`.

Note: the synthetic detected-event generator conditions the noisy distance observation on detection (`pe_obs_condition_on_detection=True`) to avoid a selection-on-θ mismatch; see `2-3-G/gate2_sbc_report.md`.

## Third-party software

This project uses (and optionally benchmarks against) open-source gravitational-wave community software, including:
- `gwpopulation` (GWPopulation contributors)
- `wcosmo` (wcosmo contributors)
- `icarogw` (ICAROGW contributors; installed in an isolated venv here)
- `ligo.skymap`, `bilby`, `healpy`, `astropy`, `h5py` (and other scientific Python dependencies)

## Recent work + findings (2026-02-03)

This section is a “paper trail” of what we ran and what it implies for the audit.

### 1) Sanity checks (oracles)

We cross-checked our `alpha(H0)` implementation (distance cache + injection rescaling proxy) against community tooling:

- Output dir: `outputs/workcycle_oracle_popmarg_20260203_202133UTC/oracle_alpha`
- Summary JSON: `outputs/workcycle_oracle_popmarg_20260203_202133UTC/oracle_alpha/json/alpha_compare_summary.json`
- Result: `alpha(H0)` matches:
  - ours vs `wcosmo`: `max_abs_diff ≈ 7.6e-10`
  - ours vs `icarogw`: `max_abs_diff ≈ 6.6e-05`

Takeaway: the `alpha(H0)` backend is not the source of the Gate‑2 “edge runaway”.

### 2) Gate‑2 population anchoring (LVK hyperposterior marginalization)

Gate‑2 is sensitive to population assumptions. We ran the “best available baseline” option: marginalize Gate‑2 over 200 LVK population hyperposterior draws (PowerLaw+Peak + powerlaw redshift).

- Output dir: `outputs/workcycle_oracle_popmarg_20260203_202133UTC/gate2_popmarg_lvk`
- Summary JSON: `outputs/workcycle_oracle_popmarg_20260203_202133UTC/gate2_popmarg_lvk/json/pop_marginal_summary.json`
- Mixture posterior (200 draws):
  - MAP `H0 ≈ 148` (not at grid edge)
  - median `p50 ≈ 150.24` with `[p16,p84] ≈ [134.81,169.40]`
  - `0/200` edge hits

Takeaway: under LVK-ish population assumptions the GR selection-on posterior is still *high*, but it is not edge-pegged on a wide grid (40–200).

### 3) Gate‑2 real-data ablation grid (what’s actually driving the H0 runaway)

We ran a 6‑case ablation over the two biggest modeling knobs:
- redshift prior (`pop_z_mode`)
- mass prior (`pop_mass_mode`)
plus two “H0 volume scaling bookkeeping” toggles that move an `H0^{-3}` factor between numerator and selection.

- Output dir: `outputs/gate2_realdata_ablationgrid_20260203_212254UTC`
- Summary CSV: `outputs/gate2_realdata_ablationgrid_20260203_212254UTC/ablation_summary.csv`

| case | pop_z_mode | pop_mass_mode | pop_z_include_h0_volume_scaling | selection_include_h0_volume_scaling | H0_map | p50 | edge |
|---|---|---|---:|---:|---:|---:|---|
| znone_massnone | none | none | False | False | 41.0 | 50.39 | False |
| znone_masspeak | none | powerlaw_peak_q_smooth | False | False | 40.5 | 44.13 | False |
| zcomoving_massnone | comoving_uniform | none | False | False | 127.5 | 125.05 | False |
| zcomoving_masspeak | comoving_uniform | powerlaw_peak_q_smooth | False | False | 175.0 | 174.98 | False |
| zcomoving_masspeak_popzvol | comoving_uniform | powerlaw_peak_q_smooth | True | False | 40.0 | 40.00 | True |
| zcomoving_masspeak_popzvol_selvol | comoving_uniform | powerlaw_peak_q_smooth | True | True | 175.0 | 174.98 | False |

Takeaway: the redshift+mass population model and H0-scaling conventions *dominate* the Gate‑2 posterior shape; toggling where the “volume scaling” lives can flip the posterior from high-H0 to low-edge.

### 4) Gate‑2 “single-event dominance” jig (fast heuristic diagnosis)

To quickly see if one or a few events dominate the monotonic H0 push, we added a `filter` subcommand to `scripts/gate2_jig.py` that recomputes the posterior after:
- dropping the worst-ESS events,
- dropping the highest “b exponent” events (events with the strongest positive H0 slope),
- thresholding by `ess_min` / `finite_frac`.

Example (recomputes from an existing Gate‑2 JSON):

```bash
./.venv/bin/python scripts/gate2_jig.py filter \
  --json outputs/gate2_realdata_fixedzmax_20260203_190502UTC/fixed_wide/json/gr_h0_selection_on_inv_sampling_pdf.json \
  --out FINDINGS/gate2_filter_fixed_wide.csv
```

Result file (tracked): `FINDINGS/gate2_filter_fixed_wide.csv`

In that specific fixed-wide run, dropping a single worst-ESS event (`GW191127_050227`) moves the posterior substantially (H0_map ~175 → ~150.5).

Takeaway: event-level diagnostics matter; apparent “global” behavior can be driven by a small subset of events.

### 5) Gate‑2 ladder (toy cancellation + injection-recovery rungs)

We ran the built-in ladder to avoid endless knob-turning and to isolate which rung breaks.

- Output dir: `outputs/siren_gate2_ladder_20260203_221227UTC`
- Results CSV: `outputs/siren_gate2_ladder_20260203_221227UTC/ladder.csv`

Key result: under a mass-peak population model, a 1D SNR-only selection model biases H0 high, while the 2D model (`snr_mchirp_binned`) largely removes that bias in injection-recovery.

| rung | det_model | pop_z_mode | pop_mass_mode | H0_map | p50 | bias_p50 (vs 70) |
|---|---|---|---|---:|---:|---:|
| ladder_2_z_comoving_mass_peak | snr_binned | comoving_uniform | powerlaw_peak_q_smooth | 84.0 | 82.47 | +12.47 |
| ladder_5_det_snr_mchirp_binned | snr_mchirp_binned | comoving_uniform | powerlaw_peak_q_smooth | 72.0 | 70.74 | +0.74 |

Takeaway: selection realism (mass dependence) is a high-leverage fix for Gate‑2 robustness.

### 6) Code fix that fell out of this work

We fixed a real bug in the hierarchical PE implementation:

- Bug: a mass-coordinate Jacobian factor was being applied even when `pop_mass_mode="none"`, making logL spuriously depend on the PE mass samples (q) in “massless” runs.
- Fix: gate the Jacobian under `if pop_mass_mode != "none"`.
- Regression test: `tests/test_dark_sirens_hierarchical_pe_mass_coord_jacobian.py`

This fix is in the `main` branch history (commit `4dba965`).

### 7) Isolator re-run with 2D selection proxy (`snr_mchirp_binned`)

We extended the **selection alpha** calculator to support a simple 2D detectability proxy
\(p_{\rm det}(\mathrm{SNR},\mathcal{M}_{\rm det})\) (`snr_mchirp_binned`) and re-ran the
real-data isolator battery under the same remembered population assumptions as the Gate‑2 control
(comoving_uniform + PowerLaw+Peak).

- Code:
  - `src/entropy_horizon_recon/dark_sirens_selection.py` now supports `det_model=snr_mchirp_binned` for selection alpha.
  - `scripts/run_siren_selection_alpha.py` adds `--det-model snr_mchirp_binned --mchirp-binned-nbins ...`.
- Output dir: `outputs/workcycle_siren_popon_snr_mchirp_20260203_232125UTC`
  - Selection alpha summary: `outputs/workcycle_siren_popon_snr_mchirp_20260203_232125UTC/selection_alpha/tables/selection_alpha_summary.json`
  - Isolator totals: `outputs/workcycle_siren_popon_snr_mchirp_20260203_232125UTC/isolator_totals.json`

Key totals (mode=`none`, 33 events; all scrambles completed with `n_skip=0`):
- \(\Delta \mathrm{LPD}_{\rm data} \approx -2.655\) (data term prefers GR)
- \(\Delta \mathrm{LPD}_{\rm sel} \approx +4.870\) (selection term prefers μ)
- \(\Delta \mathrm{LPD}_{\rm total} \approx +2.214\) (net preference flips to μ)

Event-wise (mode=`none`):
- data term: GR wins in 32/33 events
- total (data+selection): μ wins in 25/33 events

Takeaway: the “ghost” pattern persists under a more selection-realistic proxy — the data term still prefers GR, while selection bookkeeping can dominate and flip the sign of the total.

### 8) “Ghost” robustness checks (injection swaps + LVK population draws + synthetic reproduction) (2026-02-04)

We ran a decisive bundle of robustness checks to see if the μ-vs-GR “ghost flip” could be blamed on a particular
injection file, sensitivity segment, or a brittle choice of population hyperparameters — and then we built a
controlled synthetic example that reproduces the flip under GR truth (when PE distance information is weak).

- Output dir: `outputs/workcycle_decisive_checks_20260203_235923UTC/`
- Aggregated summaries:
  - `outputs/workcycle_decisive_checks_20260203_235923UTC/decisive_checks_summary.csv`
  - `outputs/workcycle_decisive_checks_20260203_235923UTC/decisive_checks_summary.json`

#### A) Injection file / segment swap checks (real data)

All runs below use the same 33-event set and PE cache as `M0_start202` and the same remembered population knobs
(`comoving_uniform + powerlaw_peak_q_smooth`) while swapping the selection-injection source.

Injection sources tested:
- Baseline O3b segment: `data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3b.hdf5`
- GWTC-3 O3a BBHpop injection set: `data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/injections/o3a_bbhpop_inj_info.hdf`
- Full O3 sensitivity file: `data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_full.hdf5`
- O3a sensitivity segment: `data/cache/gw/zenodo/7890437/o3_sensitivity_mixture_o3a.hdf5`

Result (robust across all injection swaps):
- \(\Delta \mathrm{LPD}_{\rm data} \approx -2.45\) to \(-2.66\) (data term prefers GR)
- \(\Delta \mathrm{LPD}_{\rm sel} \approx +4.87\) to \(+5.34\) (selection term prefers μ)
- \(\Delta \mathrm{LPD}_{\rm total} \approx +2.21\) to \(+2.69\) (net preference flips to μ)

Event-wise (typical ranges across these runs, mode=`none`):
- data term: GR wins in ~31–33 / 33 events
- total (data+selection): μ wins in ~25–28 / 33 events

#### B) Population anchoring via LVK hyperposterior draws (real data)

We then repeated the real-data scoring using 5 independent draws from the LVK GWTC‑3 O3a population hyperposterior
(PowerLaw+Peak + powerlaw redshift) to anchor the population hyperparameters to “best available” community inference:

- LVK hyperposterior JSON:
  `data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/o3a_population_data_release/Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json`

Result: the same sign pattern persists under LVK-ish hyperparameters:
- \(\Delta \mathrm{LPD}_{\rm data} < 0\) (GR-favoring)
- \(\Delta \mathrm{LPD}_{\rm sel} > 0\) (μ-favoring)
- \(\Delta \mathrm{LPD}_{\rm total} > 0\) (net μ-favoring)

#### C) Controlled synthetic reproduction of the “ghost flip”

We built two synthetic injection-recovery-style demos to show when the flip is expected:

- `ghost_simulation` (informative PE): generates synthetic PE with “normal-ish” distance widths.
  - Outcome: data term dominates; total stays GR (no flip).
- `ghost_simulation_weakdata` (weak PE distances): generates synthetic PE with very broad distance likelihoods
  (prior-dominated), so selection can dominate.
  - Outcome: reproduces the ghost flip under GR truth:
    - \(\Delta \mathrm{LPD}_{\rm data} \approx -0.714\)
    - \(\Delta \mathrm{LPD}_{\rm sel} \approx +6.220\)
    - \(\Delta \mathrm{LPD}_{\rm total} \approx +5.506\)
    - wins (mode=`none`, 33 events): data term GR wins 27/33, but total μ wins 32/33

Takeaway: the flip is not an “injection file artifact”; it is a predictable regime change — when PE distance information is
weak enough, the selection normalization term can dominate the total score and drive μ-vs-GR “wins” even though the event data term prefers GR.

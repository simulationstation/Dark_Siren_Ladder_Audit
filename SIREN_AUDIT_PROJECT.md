# Siren Audit Project (Spectral-Only) — isolating selection/population systematics in H0 inference

This repository contains a **spectral-only dark-siren** pipeline that is being used as an **instrument** to
isolate *unknown systematics* capable of producing **biased or “tension-like”** behavior in inferred
cosmological parameters (especially \(H_0\)).

This is **not** a “model wins by \(\Delta\mathrm{LPD}\)” project.
Whenever we compute \(\Delta\mathrm{LPD}\) or related log-score differences, it is used **only as an internal
diagnostic** to localize *which term* in the hierarchical likelihood is driving an apparent preference
(data term vs selection normalization), and therefore to identify **miscalibration pathways**.

The current working hypothesis (supported by the control ladder runs to date) is:

- The *apparent* preference for modified propagation in some configurations is driven by **selection/population
  miscalibration**, not by event-specific distance information.

The objective is to **formalize** that conclusion, make it mechanically reproducible, and then iteratively
improve the selection/population modeling until the GR \(H_0\) control is stable and plausible.

---

## What “spectral-only” means here

“Spectral-only” means we do **not** claim host association and we do **not** require a galaxy catalog match.
Instead we use **hierarchical PE sample reweighting**:

- Use released GW posterior samples \(\theta_s\sim p(\theta\mid d)\).
- Divide out the PE prior \(\pi_{\mathrm{PE}}(\theta)\).
- Multiply by an analysis population prior \(\pi_{\mathrm{pop}}(\theta \mid \text{model})\).

In this mode, the main cosmology sensitivity is through the **distance–redshift mapping** and its coupling to
population assumptions (masses, rate evolution, etc.) and to selection \(\alpha\).

---

## Core systematics mechanism (“ghost”) we are hunting

In hierarchical inference, the main failure mode is **selection/population mismatch**:

1. The population prior used inside the event term (numerator) is not consistent with the population that
   effectively enters the selection normalization \(\alpha\).
2. The detectability mapping uses an inconsistent distance (\(d_L^{\mathrm{GW}}\) vs \(d_L^{\mathrm{EM}}\)).
3. Injection weights are incorrect or incomplete (e.g. not accounting for `sampling_pdf`).
4. Mass/redshift Jacobians are missing or inconsistent between event term and selection term.

Any of these can produce an **artificial shift in the GR \(H_0\) posterior when selection is turned on**,
which can masquerade as an “\(H_0\) anomaly” even if the event term itself is benign.

The audit system below is designed to isolate which of these pathways is active.

---

## The audit system: a control ladder (gates)

The siren audit is organized as a sequence of **gates**. The key idea:

- We only treat the pipeline as a meaningful *systematics detector* once the **GR \(H_0\) control** is stable.
- Until then, any “interesting anomaly-like behavior” is treated as **selection/population behavior**.

### Gate 0 — determinism and invariances
Goal: ensure bookkeeping cannot move the result.

Checks (examples):
- fixed seeds lead to fixed outputs
- sample-order permutations do not change results
- event inclusion/QC rules are stable (no silent event drops)

### Gate 1 — mandatory term visibility (not “evidence”)
Goal: make the pipeline transparent about what is driving changes.

Outputs we require for every audit run:
- the selection-free event term behavior (what the event term “wants”)
- the selection term behavior (how \(\alpha\) pushes things)

This is the core reason \(\Delta\mathrm{LPD}\)-like diagnostics remain useful here: not as “evidence”, but as
an *accounting identity* to separate the numerator behavior from \(\alpha\)-driven behavior.

### Gate 2 — GR \(H_0\) selection-on control (the primary control)
Goal: under GR-only distances, selection-on \(H_0\) inference should be:
- **interior-peaked** (not boundary-piled),
- **stable** to modest grid widening and to small nuisance toggles,
- plausibly centered (order-of-magnitude correct).

Implementation:
- `src/entropy_horizon_recon/dark_siren_h0.py` (`compute_gr_h0_posterior_grid_hierarchical_pe`)
- runner: `scripts/run_siren_gate2_gr_h0_control.py`

Diagnostics emitted:
- per-event log-likelihood on the \(H_0\) grid (selection off)
- \(\log\alpha(H_0)\) grid (selection on)
- event QC bookkeeping (skip/fail on zero-support events)

If Gate 2 fails, the highest ROI next step is **selection-model calibration**, not running more events.

### Gate 3 — selection “slope” sensitivity (\(\eta\) scan)
Goal: quantify whether the GR \(H_0\) control is being dominated by a single detectability knob.

Implementation:
- runner: `scripts/run_siren_selection_eta_scan.py`

If small changes in detectability produce large swings, we are seeing a **selection gradient**, not
astrophysical information.

### Gate 4 — mechanism-killing nulls (spectral-only)
Goal: test whether the pipeline is actually using the intended distance information, or if it can produce
“anomaly-like” behavior even when distance structure is destroyed.

Implementation:
- `src/entropy_horizon_recon/siren_isolator.py` + runner `scripts/run_siren_isolator.py`

Key null modes:
- `prior_dL`: replace \(d_L\) samples with draws from the event’s **analytic PE distance prior**
- `shuffle_dL`, `shuffle_mass`, `shuffle_mc`, `shuffle_q`: kill specific joint correlations while preserving marginals

Expected systematics signature:
- “distance-killing” nulls should collapse distance information **in the event term**, but selection-driven
  effects may remain. That pattern identifies a selection ghost.

### Gate 5 — selection implementation audit (alpha variants)
Goal: test whether the GR \(H_0\) control is stable across plausible, internally consistent \(\alpha\)
implementations.

Implementation:
- `src/entropy_horizon_recon/dark_sirens_selection.py` (`compute_selection_alpha_from_injections`)
- runner: `scripts/run_siren_gate5_selection_audit.py`

Audited toggles:
- injection weighting: `weight_mode=none` vs `weight_mode=inv_sampling_pdf`
- detectability distance: `mu_det_distance=gw` (physical) vs `mu_det_distance=em` (diagnostic ablation)
- detectability model family: `det_model=threshold` vs `det_model=snr_binned`
- population knobs (z evolution and mass model) aligned to the event-term assumptions

If the GR \(H_0\) posterior changes qualitatively across these, we have not pinned down selection/population
consistency yet.

---

## What we currently treat as “pass/fail”

**Pass (minimum viable systematics detector):**
- Gate 2: GR \(H_0\) selection-on is interior-peaked and not hypersensitive to small toggles.
- Gate 5: \(\alpha\) variants do not produce sign flips or extreme sensitivity without a clear physical reason.

**Fail (keep debugging):**
- selection-on \(H_0\) piles up at grid edges or moves wildly with small changes,
- mechanism-killing nulls still yield anomaly-like behavior that cannot be attributed to selection/popu\-lation,
- selection audit shows strong dependence on injection-weighting choices or detectability-distance conventions.

---

## How to run (recommended order)

All long runs should follow `AGENTS.md` (detached job + `pid.txt` + `run.log`).
For status, use `python scripts/audit_status.py outputs` (or point it at a specific output dir).

1) **Build isolator cache (per-event logL vectors)**
- runner: `scripts/run_siren_isolator.py`
- output: `outputs/siren_isolator_<stamp>/cache/logL_<mode>_<event>.npz`

2) **Run Gate 5 selection audit**
- runner: `scripts/run_siren_gate5_selection_audit.py`
- output: `.../tables/selection_audit.csv`
- also writes per-draw decomposition arrays in `.../tables/delta_lpd_draws_<config>.npz` (diagnostic only)

3) **Run Gate 2 GR \(H_0\) selection-on control**
- runner: `scripts/run_siren_gate2_gr_h0_control.py`
- output: `.../json/gr_h0_selection_on_*.json` with `log_alpha_grid` and `logL_sum_events_rel`

4) **Run Gate 4 nulls (distance/mass scrambles)**
- runner: `scripts/run_siren_isolator.py` with `--scramble-modes prior_dL,shuffle_*`

---

## Why this is directly relevant to “Hubble tension”

If a spectral-only pipeline exhibits:

- a systematic selection-on shift in the GR \(H_0\) control,
- sensitivity to injection weighting,
- large detectability gradients,
- and persistence of “anomaly-like” preferences under distance-killing nulls,

then the pipeline is demonstrating something important and *general*:

> A plausible-but-wrong selection/population model can manufacture an apparent “\(H_0\)-tension-like” shift
> **without** requiring any event-specific distance information.

That is exactly the kind of failure mode that can propagate into the broader literature if selection
functions are approximated, mismatched, or implicitly double-counted.

This is why **Gate 2** is the primary control: it converts “unknown unknowns” into a specific object to
debug and calibrate — the function \(\alpha(H_0)\) under GR.

---

## What “success” looks like for the audit project

The pipeline becomes a credible **systematics isolator** (for \(H_0\)) when:

1. The GR \(H_0\) selection-on control is interior-peaked and plausibly centered under a clearly specified
   population+selection model.
2. That posterior is stable to reasonable variations in:
   - injection importance weighting,
   - detectability model family,
   - redshift evolution and mass-model knobs.
3. Injection-recovery / closed-loop checks confirm the selection-on \(H_0\) estimator is **not biased**
   under known truth.

At that point, if a real-data application still exhibits a reproducible anomaly-like behavior, it is a
much stronger indication that we are seeing either:
- a genuine data-population inconsistency (astrophysical), or
- a specific, now-localized selection modeling deficiency (instrumental),
rather than a generic “\(\alpha\) ghost”.

---

## Immediate next step (implemented): injection-recovery / closed-loop calibration

The missing piece for full formalization is a closed-loop test that uses the same inference path as Gate 2:

1) Choose a truth \(H_0^\star\) and fixed population hyperparameters (mass + redshift evolution).
2) Generate synthetic “observed” events consistent with the detection model and that truth.
3) Run the **same** GR selection-on \(H_0\) inference and verify we recover \(H_0^\star\) without bias
   (within Monte Carlo error / coverage).

This isolates whether the current selection model is merely *sensitive* (high variance) or actually
*biased* (systematic shift), which is the core “Hubble tension systematics” question.

Implementation:
- `src/entropy_horizon_recon/siren_injection_recovery.py`
- `scripts/run_siren_injection_recovery.py`
- `scripts/run_siren_injection_recovery_suite.py` (multi-replicate, resumable)

Synthetic PE samples (important):
- The injection-recovery harness synthesizes *prior-consistent* PE posteriors so that
  \(p(\theta\mid d)\propto \mathcal{L}(d\mid\theta)\,\pi_{\rm PE}(\theta)\) holds by construction.
- Mode `pe_synth_mode=prior_resample` does this by sampling candidates from the analytic PE priors and
  resampling them with weights proportional to a simple likelihood around the injected truth.
- Mode `pe_synth_mode=naive_gaussian` exists only as a debugging ablation and should not be used for
  “perfect null” claims.
- As of **2026‑02‑02**, synthetic PE generation defaults to **`pe_obs_mode=noisy`** (draw a noisy “observed”
  \((d_L,\mathcal{M}_c,q)\) summary around truth, then build posterior samples around that observation) and
  writes a **distance P–P diagnostic** (`summary.pe_pp`) so calibration failures show up immediately.

Pass/fail (minimum):
- selection-on GR \(H_0\) posterior is interior-peaked and recovers \(H_0^\star\) within expected Monte Carlo error
  for a moderate synthetic sample.

### Current Gate‑2 closed-loop status (2026‑02‑02)

We identified and fixed a concrete selection/population bookkeeping issue that was capable of producing
an artificial \(H_0\) shift even in closed-loop tests:

- **Injection `sampling_pdf` mass-coordinate Jacobian**: when `sampling_pdf` is in \((m_1,m_2)\) but the
  population mass model is in \((m_1,q)\), importance weights must include an additional \(1/m_1\)
  (since \(\mathrm{d}m_2 = m_1\,\mathrm{d}q\)).

Fix implemented in:
- `src/entropy_horizon_recon/dark_siren_h0.py`
- `src/entropy_horizon_recon/siren_injection_recovery.py`
- `src/entropy_horizon_recon/dark_sirens_selection.py`

Closed-loop suite result (O3b mixture injections; `det_model=snr_binned`; `weight_mode=inv_sampling_pdf`;
`pop_z_mode=comoving_uniform`; `pop_mass_mode=powerlaw_peak_q_smooth`; `include_pdet_in_event_term=False`):
- Output: `outputs/gate2_suite_o3b_peak_smooth_jacfix_nopdet_20260202_211955UTC/`
- Aggregate: `tables/suite_aggregate.json` reports `bias_p50_on_mean ≈ +0.93 km/s/Mpc` across 32 replicates
  (60 detected events each), with `bias_p50_on_sd ≈ 2.87` and no boundary-peaked reps.

This is now the minimum “Gate‑2 pass” benchmark for this configuration: selection‑on GR \(H_0\) is
interior‑peaked and approximately unbiased on average.

Additional large closed-loop suite (same knobs; higher replicate count):
- Output: `outputs/gate2_calibration_combo_20260202_225949UTC/inj_suite_big/`
- Aggregate: `bias_p50_on_mean ≈ +0.90 km/s/Mpc`, `bias_p50_on_sd ≈ 4.21` across 256 reps (50 events each),
  with `coverage_68_on ≈ 0.625` and `coverage_95_on ≈ 0.926`.

Synthetic-PE width tuning:
- Output: `outputs/gate2_inj_pe_tune_20260202_230605UTC/`
- Scanning `dl_frac_sigma_floor` shows Gate‑2 calibration degrades rapidly once synthetic PE becomes too prior‑dominated
  (`dl_frac_sigma_floor ≥ 0.15` yields strong bias and severe undercoverage). Keep `dl_frac_sigma_floor` in 0.05–0.10
  unless `pe_n_samples` is increased and/or the log-mean estimator is variance/bias-controlled.

**Caution (wide-grid stress test):** a later wide-grid suite (`outputs/gate2_suite_wide_20260202_221105UTC/`)
showed large positive bias and undercoverage in the selection-on GR \(H_0\) control **when using truth-centered
synthetic PE and strict finite-support QC**. That suite is treated as a *debug trigger*, not as a calibrated result.
The current injection-recovery harness now (i) supports `pe_obs_mode=noisy` and (ii) allows `event_min_finite_frac=0`
to prevent silent event drops, and should be re-run before treating wide-grid behavior as meaningful.

### Gate‑2 real‑data snapshot (hierarchical PE; population‑conditioned)

Using the cached hierarchical PE set from `outputs/dark_siren_hier_distanceonly36_autoanalysis_20260201_222130UTC/cache/`
(25 events), we ran Gate‑2 GR \(H_0\) controls under the same injection‑based selection proxy and the same
mass/redshift population knobs used in the closed‑loop suite.

Key outputs (all use O3b mixture injections `data/cache/gw/zenodo/7890437/endo3_mixture-LIGO-T2100113-v12-1256655642-12905976.hdf5`):

- Baseline population (`pop_z_mode=comoving_uniform`, `pop_mass_mode=powerlaw_peak_q_smooth`; \(H_0\in[40,100]\)):
  - `outputs/siren_gate2_gr_h0_20260202_215226UTC/json/gr_h0_selection_on_inv_sampling_pdf.json`:
    - `H0_map=47.75`, `p50=51.18` (interior‑peaked; Gate‑2 pass by the “not at edge” criterion)
- Redshift‑prior sensitivity (same mass model; \(H_0\in[40,100]\)):
  - `pop_z_mode=comoving_powerlaw, k=2`: `outputs/siren_gate2_gr_h0_20260202_215735UTC/…` → `p50=48.93`
  - `pop_z_mode=comoving_powerlaw, k=4`: `outputs/siren_gate2_gr_h0_20260202_215809UTC/…` → `p50=46.54`
  - `pop_z_mode=none`: `outputs/siren_gate2_gr_h0_20260202_215843UTC/…` → `p50=59.40`
- Mass‑prior ablation (`pop_mass_mode=none`, `pop_z_mode=comoving_uniform`; \(H_0\in[40,100]\)):
  - `outputs/siren_gate2_gr_h0_20260202_220002UTC/json/gr_h0_selection_on_inv_sampling_pdf.json`:
    - `p50=90.62`, `H0_map` hits the upper grid edge (Gate‑2 fail).

Interpretation: in spectral‑only mode, the GR \(H_0\) control is *strongly population‑conditioned*. The fact
that plausible toggles of \(p(z)\) and \(p(m_1,q)\) move the posterior by \(\mathcal{O}(10\)–\(40)\,\mathrm{km/s/Mpc}\)
means “\(H_0\) from BBH dark sirens” is not a stable control unless population hyperparameters are fixed by
external information or explicitly marginalized.

Latest 33-event Gate‑2 snapshot (same PE cache family; `snr_binned`; `z_max=2`):
- `outputs/gate2_calibration_combo_20260202_225949UTC/gate2_real_pop/json/gr_h0_selection_on_inv_sampling_pdf.json`:
  - `H0_map=56.5` (not at edge), `p50≈59.6`, `p84≈82.7` (selection‑on, injection‑weighted)
- `outputs/gate2_calibration_combo_20260202_225949UTC/gate2_real_pop/json/gr_h0_selection_on_none.json`:
  - collapses to the lower grid edge (expected: `weight_mode=none` is not a meaningful physical selection model)

Recommended “perfect null” run (suite):
- Use the suite runner to quantify bias/variance across many replicates:
  - `scripts/run_siren_injection_recovery_suite.py`
- The suite is resumable: it writes `json/rep_*.json` and can be re-summarized with `--rebuild-only`.
- For any non-trivial suite, run detached with `pid.txt` + `run.log` per `AUDIT.md` / `AGENTS.md`.

---

## Where the code lives (entry points)

- GR \(H_0\) control (hierarchical PE + selection):
  - `src/entropy_horizon_recon/dark_siren_h0.py`
  - `scripts/run_siren_gate2_gr_h0_control.py`
- Selection alpha from injections:
  - `src/entropy_horizon_recon/dark_sirens_selection.py`
  - `scripts/run_siren_gate5_selection_audit.py`
- Spectral-only nulls / scrambles:
  - `src/entropy_horizon_recon/siren_isolator.py`
  - `scripts/run_siren_isolator.py`
- Detectability gradient scan:
  - `scripts/run_siren_selection_eta_scan.py`
- Injection-recovery (closed-loop selection + population check):
  - `src/entropy_horizon_recon/siren_injection_recovery.py`
  - `scripts/run_siren_injection_recovery.py`

---

## Where to look for the current audit narrative/results

The living audit log is:

- `FINDINGS/siren_ghost_control_ladder_20260202.md`

It records what gates were run, what outputs were produced, and what the signatures imply about
selection/population miscalibration pathways.

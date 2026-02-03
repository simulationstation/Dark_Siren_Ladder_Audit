# Siren “ghost” control ladder — 2026-02-02

This note records the **control-ladder** diagnostics for the **spectral-only** (hierarchical PE) dark-siren score.
The objective here is *not* to claim evidence for modified propagation; it is to **isolate the mechanism** that produces positive \(\Delta\mathrm{LPD}\) in some configurations.

Primary reference for the ladder: `Other_Plan.tex`.

## Executive summary

Across the current O3 event subset used here (\(N=33\) scored events), the total preference can be **positive** for the \(\mu\)-propagation model **only because** the **selection normalization** term dominates:

\[
\Delta \mathrm{LPD}_{\rm tot} = \Delta \mathrm{LPD}_{\rm data} + \Delta \mathrm{LPD}_{\rm sel}.
\]

Empirically, we see:

- \(\Delta \mathrm{LPD}_{\rm data} < 0\) (the **data term prefers GR** on average).
- \(\Delta \mathrm{LPD}_{\rm sel} > 0\) (the **selection term** boosts \(\mu\) because \(\alpha_\mu < \alpha_{\rm GR}\)).
- Mechanism-killing nulls that remove distance information drive \(\Delta \mathrm{LPD}_{\rm data}\to 0\) but **do not** eliminate \(\Delta \mathrm{LPD}_{\rm tot}\).
- Plausible, “more correct” selection implementations (injection sampling-pdf weighting; detectability distance ablations) can **remove or flip** the sign of \(\Delta \mathrm{LPD}_{\rm tot}\).

This combination is the textbook signature of a **selection/population “ghost”** rather than a propagation signal.

## What was run

### Gate 4 (mechanism-killing null suite)

Runner: `scripts/run_siren_isolator.py` (module `src/entropy_horizon_recon/siren_isolator.py`).

Output:
- `outputs/siren_isolator_gate4_nulls_20260202_035312UTC/json/summary.json`

Null modes included:
- `none` (baseline)
- `prior_dL` (replace \(d_L\) samples by draws from the event’s **analytic PE distance prior**)
- `swap_dL_events` (cross-event swap of \((d_L,\log\pi_{d_L})\) samples)

### Gate 5 (selection audit)

Runner: `scripts/run_siren_gate5_selection_audit.py`.

Output:
- `outputs/siren_gate5_selection_audit_20260202_040534UTC/tables/selection_audit.csv`
- `outputs/siren_gate5_selection_audit_20260202_040534UTC/json/summary.json`
- `outputs/siren_gate5_selection_audit_20260202_040534UTC/figures/delta_lpd_by_selection_config.png`

Key audited toggles:
- injection importance weights: `weight_mode=none` vs `weight_mode=inv_sampling_pdf`
- detectability distance coupling: `mu_det_distance=gw` (physical) vs `mu_det_distance=em` (diagnostic ablation)

### Gate 2 (GR \(H_0\) selection-on control)

Runner: `scripts/run_siren_gate2_gr_h0_control.py`.

Outputs (baseline default pop settings):
- `outputs/siren_gate2_gr_h0_full_20260202_041306UTC/`

This is the “published-control” plumbing: if selection is implemented consistently, the GR-only \(H_0\) posterior (selection ON) should be **interior-peaked** and stable to grid widening. Boundary pile-ups are a Gate-2 fail.

#### Gate 2 update (2026-02-02): closed-loop injection-recovery + sampling\_pdf mass-coordinate Jacobian fix

While chasing Gate‑2 failures/biases, we identified a concrete bookkeeping bug affecting **injection importance weights** when `weight_mode=inv_sampling_pdf` is combined with a **population mass model parameterized in** \((m_1,q)\) (with \(q\equiv m_2/m_1\)).

Many LVK O3 sensitivity-injection releases provide `sampling_pdf` in component-mass coordinates \((m_1,m_2)\), whereas our population mass model is written in \((m_1,q)\). The coordinate transform \(m_2=q\,m_1\) has Jacobian:
\[
\mathrm{d}m_1\,\mathrm{d}m_2 = m_1\,\mathrm{d}m_1\,\mathrm{d}q
\quad\Rightarrow\quad
p_{\rm inj}(m_1,q)=p_{\rm inj}(m_1,m_2)\,m_1.
\]
Therefore, when using weights \(w\propto p_{\rm pop}/p_{\rm inj}\) and modeling \(p_{\rm pop}\) in \((m_1,q)\), the `inv_sampling_pdf` factor must include an additional \(1/m_1\) if `sampling_pdf` is in \((m_1,m_2)\) coordinates. Omitting this factor biases \(\alpha\) (and any closed-loop event sampling from injections) and can produce an apparent selection-driven \(H_0\) shift.

Fix implemented (default `inj_mass_pdf_coords="m1m2"`):
- `src/entropy_horizon_recon/dark_siren_h0.py` (`_injection_weights`, `_alpha_h0_grid_from_injections`)
- `src/entropy_horizon_recon/siren_injection_recovery.py` (closed-loop injection sampling + alpha)
- `src/entropy_horizon_recon/dark_sirens_selection.py` (`compute_selection_alpha_from_injections`)

Closed-loop validation suite (O3b mixture injections; \(H_0^\star=70\); 32 replicates of 60 detected events each):
- Output: `outputs/gate2_suite_o3b_peak_smooth_jacfix_nopdet_20260202_211955UTC/`
- Summary: `tables/suite_aggregate.json` reports `bias_p50_on_mean ≈ +0.93 km/s/Mpc` with `bias_p50_on_sd ≈ 2.87`, and no MAP-at-edge reps.

Interpretation: after the Jacobian fix, Gate‑2 is **interior-peaked** and **approximately unbiased on average** for this closed-loop configuration. Remaining spread is dominated by finite-sample variance (BBH distance posteriors are broad).

#### Gate 2 update (2026-02-02): “moderate” wide-grid injection suite with noisy synthetic PE + P–P diagnostics

To stress the updated synthetic-PE machinery on a wider \(H_0\) grid while avoiding the earlier “truth-centered PE” failure mode, we ran a moderate suite with:

- \(H_0^\star=70\), \(N_{\rm rep}=32\), \(N_{\rm events}=25\)
- \(H_0\in[40,100]\) with 241 grid points
- `pe_obs_mode=noisy` (noisy observation center), `pe_synth_mode=likelihood_resample`, `pe_n_samples=5000`
- `event_min_finite_frac=0` (no “insufficient support” dropping)
- parallel replicates (`--n-proc 0` → capped to 32 workers)

Output:
- `outputs/gate2_suite_moderate_20260202_224746UTC/`

Key aggregate results (`tables/suite_aggregate.json`):
- `bias_p50_on_mean ≈ +1.22 km/s/Mpc`, `bias_p50_on_sd ≈ 6.60`
- `coverage_68_on = 0.6875` (near nominal), `coverage_95_on = 0.90625` (low, but only 32 reps)
- distance P–P across all events: `pp_dL_all_mean ≈ 0.469`, `pp_dL_all_ks ≈ 0.061`
- `skipped_on_mean = 0.0` (no silent event drops)

Interpretation: this run indicates the updated injection-recovery harness is **largely calibrated** (P–P looks close to uniform; 68% coverage is near nominal) while still exhibiting heavy-tailed finite-sample behavior (a few reps drift high/low).

#### Gate 2 update (2026-02-02): event QC for population support

One Gate-2 failure mode is **population mismatch**: e.g. using a BBH-only mass model while including NSBH-like events.
This can cause the hierarchical numerator to have **zero support** (all \(-\infty\)) for an event across the scanned \(H_0\) grid, which collapses the total likelihood.

To make Gate 2 mechanically robust, the GR \(H_0\) control runner now supports:
- `--event-qc-mode skip` (default): skip events that have no finite log-likelihood anywhere on the \(H_0\) grid under the specified population bounds (and record them in the output JSON).
- `--event-qc-mode fail`: strict mode; aborts on the first zero-support event.

Code path:
- `src/entropy_horizon_recon/dark_siren_h0.py` (`compute_gr_h0_posterior_grid_hierarchical_pe`, `event_qc_mode`)
- `scripts/run_siren_gate2_gr_h0_control.py` (`--event-qc-mode`)

Example output demonstrating this for the O3 set used here:
- `outputs/siren_gate2_gr_h0_full_qcskip_20260202_050147UTC/json/gr_h0_selection_on_inv_sampling_pdf.json`
  shows `events_skipped = ["GW191219_163120"]` under a BBH-style mass model, consistent with that event’s extremely small mass ratio (NSBH-like posterior).

#### Gate 2 update (2026-02-02): smooth mass taper mode + explicit decomposition outputs

Hard population bounds (especially a hard \(m_{\min}\) on source-frame masses) can also create **artificial H0 truncation points**
where an event’s inferred \(m_2\) crosses the bound as \(H_0\) changes. This produces a misleading “interior peak” that is actually a
support boundary artifact.

To eliminate this specific failure mode (without pretending the BBH population is known), the Gate‑2 code now supports a smooth-taper
mass prior:

- `pop_mass_mode=powerlaw_q_smooth`: same powerlaw-q form as `powerlaw_q`, but with smooth sigmoid tapers at \(m_{\min}\) and \(m_{\max}\)
  for *both* component masses. Controlled by `pop_m_taper_delta` (Msun).

Gate‑2 outputs now also include **explicit decompositions** intended for debugging:

- `logL_sum_events_rel`: the *selection-off* per-grid log-likelihood sum (up to an additive constant).
- `log_alpha_grid`: \(\log \alpha(H_0)\) for selection-on runs (so you can see whether the edge behavior is coming from \(\alpha\) or the event term).
- `H0_map_at_edge` and `gate2_pass`: a simple pass/fail flag. Current rule:
  - pass iff the MAP is not at a grid edge and no events were skipped by QC.

Smoke validation of the new fields and `powerlaw_q_smooth`:
- `outputs/siren_gate2_gr_h0_smoke_smooth2_20260202/`
  (note: smoke uses only 3 events; it is a plumbing check, not a physics validation).

### Gate 3 (\(\eta\) selection-slope scan)

Runner: `scripts/run_siren_selection_eta_scan.py` (detached job).

Outputs (key comparisons):
- `outputs/eta_scan_gate3_none_gw_20260202_043715UTC/tables/eta_scan.csv` (baseline: `weight_mode=none`, `mu_det_distance=gw`)
- `outputs/eta_scan_gate3_none_em_20260202_044025UTC/tables/eta_scan.csv` (ablation: `weight_mode=none`, `mu_det_distance=em`)
- `outputs/eta_scan_gate3_invpdf_20260202_041613UTC/tables/eta_scan.csv` (audit: `weight_mode=inv_sampling_pdf`, `mu_det_distance=gw`)

This scan perturbs the detectability model with a scalar offset \(\eta\) (implemented as an SNR shift) and recomputes \(\alpha(\eta)\) and the implied \(\Delta\mathrm{LPD}_{\rm sel}(\eta)\).

## Key results to cite

### Gate 4: preference survives when distance information is removed

From `outputs/siren_isolator_gate4_nulls_20260202_035312UTC/json/summary.json`:

- Baseline (`none`): \(\Delta \mathrm{LPD}_{\rm tot}\approx +1.36\), \(\Delta \mathrm{LPD}_{\rm data}\approx -1.73\).
- PE-prior distance null (`prior_dL`): \(\Delta \mathrm{LPD}_{\rm data}\approx 0\) **but** \(\Delta \mathrm{LPD}_{\rm tot}\approx +4.03\).
- Cross-event distance swap (`swap_dL_events`): essentially unchanged from baseline.

Interpretation: the positive total score is **not** tied to event-specific distance structure; it is dominated by selection/population effects.

### Gate 5: the sign depends on selection implementation

From `outputs/siren_gate5_selection_audit_20260202_040534UTC/tables/selection_audit.csv` (33 events, 512 draws):

- `snr_binned + weight_mode=none + mu_det_distance=gw`: \(\Delta \mathrm{LPD}_{\rm tot}\approx +1.34\) with \(\Delta \mathrm{LPD}_{\rm data}\approx -1.73\).
- Same but `mu_det_distance=em`: \(\Delta \mathrm{LPD}_{\rm tot}\approx -1.74\) (selection advantage vanishes; total \(\approx\) data term).
- `snr_binned + weight_mode=inv_sampling_pdf + mu_det_distance=gw`: \(\Delta \mathrm{LPD}_{\rm tot}\approx -0.22\) (sign flip; GR wins).

Interpretation: \(\Delta\mathrm{LPD}\) is currently **pipeline-dependent** through \(\alpha\). That is a Gate-5 fail for any evidence claim.

Gate‑5 update with threshold + inv_sampling_pdf variants:
- `outputs/siren_gate5_selection_audit_baseline2_20260202_0645UTC/tables/selection_audit.csv`
  shows that `threshold + inv_sampling_pdf + mu_det_distance=gw` also yields **GR winning**:
  \(\Delta \mathrm{LPD}_{\rm tot}\approx -0.11\) (with \(\Delta \mathrm{LPD}_{\rm data}\approx -1.73\) and \(\Delta \mathrm{LPD}_{\rm sel}\approx +1.62\)).

### Gate 5 update (2026-02-02): “Gate‑2 consistent” population makes the selection ghost large again

To align the selection audit with Gate‑2’s passing configuration, we re-generated the *data-term* log-likelihood vectors
with a nontrivial population model and then re-ran Gate‑5 under matching pop+selection knobs:

- Isolator (data term) cache under `pop_z_mode=comoving_uniform` and `pop_mass_mode=powerlaw_peak_q_smooth`:
  - `outputs/siren_isolator_masspeak_20260202_0730UTC/cache/`
- Gate‑5 audit using that cache and the same population knobs (plus `weight_mode=inv_sampling_pdf`):
  - `outputs/siren_gate5_selection_audit_masspeak_20260202_0733UTC/tables/selection_audit.csv`

Results (33 events, 512 draws; **spectral-only**, not host-association):

- The data term is even more negative than in the baseline-pop audit:
  - \(\Delta \mathrm{LPD}_{\rm data}\approx -2.64\) (GR wins).
- The selection term becomes very large and positive under the Gate‑2-consistent population model:
  - \(\Delta \mathrm{LPD}_{\rm sel}\approx +5.2\) to \(+5.4\) (model-dependent).
- Therefore the total becomes strongly positive:
  - \(\Delta \mathrm{LPD}_{\rm tot}\approx +2.6\) to \(+2.7\) (mu wins), but **only because selection dominates**.

As usual, the `mu_det_distance=em` ablations collapse \(\Delta \mathrm{LPD}_{\rm sel}\to 0\) and revert to GR winning with \(\Delta \mathrm{LPD}_{\rm tot}\approx\Delta \mathrm{LPD}_{\rm data}\).

Interpretation: even after making Gate‑2 stable, the “mu preference” remains explainable as **selection-model behavior** rather than
an actual improvement in the GW-data fit. This is exactly the ghost mechanism we are isolating.

### Gate 4 update (2026-02-02): correlation‑killing PE scrambles (spectral-only)

To check whether the **data term** preference is coming from a specific *joint* correlation in the PE posterior
(e.g. \(d_L\)–mass coupling), we ran an extended scramble suite under the same “Gate‑2 consistent” population+selection
configuration used above (masspeak + `thr_invpdf_gw` selection vectors).

Runner: `scripts/run_siren_isolator.py` with `--scramble-modes` including mass/distance scrambles.

Output:
- `outputs/siren_isolator_masspeak_extended_20260202_080342UTC/json/summary.json`

Key totals (33 events, 512 draws; selection vectors from `selection_alpha_thr_invpdf_gw_masspeak.npz`):

- `none`: \(\Delta\mathrm{LPD}_{\rm data}=-2.655\), \(\Delta\mathrm{LPD}_{\rm tot}=+2.515\)
- `shuffle_dL`: \(\Delta\mathrm{LPD}_{\rm data}=-2.582\), \(\Delta\mathrm{LPD}_{\rm tot}=+3.031\)
- `shuffle_mass`: \(\Delta\mathrm{LPD}_{\rm data}=-2.569\), \(\Delta\mathrm{LPD}_{\rm tot}=+3.124\)
- `shuffle_dL_mass`: \(\Delta\mathrm{LPD}_{\rm data}=-2.564\), \(\Delta\mathrm{LPD}_{\rm tot}=+3.055\)
- `shuffle_mc`: \(\Delta\mathrm{LPD}_{\rm data}=-2.597\), \(\Delta\mathrm{LPD}_{\rm tot}=+3.361\)
- `shuffle_q`: \(\Delta\mathrm{LPD}_{\rm data}=-2.615\), \(\Delta\mathrm{LPD}_{\rm tot}=+2.533\)
- `prior_dL`: \(\Delta\mathrm{LPD}_{\rm data}=-0.249\) (≈0), \(\Delta\mathrm{LPD}_{\rm tot}=+9.677\)

Interpretation:
- The **data term** preference for GR is **stable** to these correlation-killing scrambles (it stays \(\approx -2.6\)).
- The large positive totals are therefore not coming from a fragile \(d_L\)–mass correlation in PE; they remain
  explainable as **selection behavior**.
- Under `prior_dL`, the data term collapses toward 0 as expected, yet the total becomes *even more* positive. This is a
  decisive “ghost” signature for spectral-only: the preference can be driven without event-specific distance structure.
  (Note: the “selection term” inferred as \(\Delta\mathrm{LPD}_{\rm tot}-\Delta\mathrm{LPD}_{\rm data}\) is not strictly
  constant across modes because \(\log\langle e^{\log L - \log\alpha}\rangle\) is nonlinear in \(\log L\).)

### Gate 3: \(\eta\)-scan shows selection-driven sign flips (as expected for a “ghost”)

All runs below use the same event set (33 events) and the same \(\Delta\mathrm{LPD}_{\rm data}\approx -1.729\).

**Baseline selection proxy (unweighted injections):** `outputs/eta_scan_gate3_none_gw_20260202_043715UTC/tables/eta_scan.csv`

- \(\Delta \mathrm{LPD}_{\rm sel}\approx +2.38\) to \(+3.25\) across \(\eta\in[-2,+2]\).
- Therefore \(\Delta \mathrm{LPD}_{\rm tot}\approx +0.65\) to \(+1.52\): **\(\mu\) wins**, but entirely because selection dominates.

**Detectability-distance ablation:** `outputs/eta_scan_gate3_none_em_20260202_044025UTC/tables/eta_scan.csv`

- Here \(\alpha_\mu=\alpha_{\rm GR}\) by construction (detectability uses \(d_L^{\rm EM}\) even under \(\mu\)), so \(\Delta \mathrm{LPD}_{\rm sel}\approx 0\).
- Therefore \(\Delta \mathrm{LPD}_{\rm tot}\approx \Delta \mathrm{LPD}_{\rm data}\approx -1.73\): **GR wins**.

**Injection-weighted selection audit:** `outputs/eta_scan_gate3_invpdf_20260202_041613UTC/tables/eta_scan.csv`

- \(\Delta \mathrm{LPD}_{\rm sel}\approx +1.44\) to \(+1.54\) as \(\eta\) varies from \(-2\) to \(+2\).
- Therefore \(\Delta \mathrm{LPD}_{\rm tot}\approx -0.29\) to \(-0.19\): **GR wins** for all scanned \(\eta\).

Interpretation: the sign of \(\Delta\mathrm{LPD}_{\rm tot}\) is **controlled by the selection term** and by **implementation choices** for \(\alpha\) (weighting, detectability distance), not by the data term.

### Gate 2: GR \(H_0\) control fails under baseline settings

From `outputs/siren_gate2_gr_h0_full_20260202_041306UTC/`:

- selection OFF: posterior piles up at the **upper** \(H_0\) grid edge (here 90).
- selection ON (default `weight_mode=none`): posterior collapses to the **lower** \(H_0\) grid edge (here 50).

Interpretation: \(\alpha(H_0)\) dominates and is not yet calibrated under the simplistic default population assumptions; treat any selection-driven preference as a ghost until Gate 2 passes.

### Gate 2 update (2026-02-02): selection-on GR \(H_0\) becomes interior-peaked under a threshold det-model

We re-ran Gate 2 on the full \(N=33\) event set with the same PE cache and injections, but using a **threshold** detectability model for \(\alpha(H_0)\)
instead of the `snr_binned` proxy.

Key outputs:
- `outputs/siren_gate2_gr_h0_full_peak_20260202_0638UTC/` (snr_binned; `pop_mass_mode=powerlaw_peak_q_smooth`) — still edge-peaked at the high-\(H_0\) boundary (120).
- `outputs/siren_gate2_gr_h0_full_peak_thresh_unweighted_20260202_0642UTC/` (threshold; same pop knobs) — **passes Gate 2**:
  - `H0_map=52.0` (not at edge), `gate2_pass=True`
  - p50 \(\simeq 56.3\), p84 \(\simeq 66.9\) on the \([40,120]\) grid

Implementation detail (important for “self-consistency”): the threshold is now calibrated **unweighted** from the injection recovery flags
(population weights only enter the subsequent \(\alpha\) expectation). This avoids a pathological “population-dependent detection threshold”.

Interpretation: the earlier Gate‑2 boundary pile-ups were largely a **selection-model calibration artifact** (a too-soft \(\alpha(H_0)\) response under `snr_binned`),
not a statement about the GW data. With a threshold model, \(\alpha(H_0)\) is sufficiently stiff that the selection term cancels the numerator slope and yields an interior peak.

## Immediate next steps (highest ROI)

1. **Compare Gate 3 \(\eta\) scans across \(\alpha\) implementations** (`weight_mode=none` vs `inv_sampling_pdf`, and `mu_det_distance=gw` vs `em`) to quantify how much of the sensitivity is “selection modeling” versus true data preference.
2. **Re-run Gate 2 with a consistent population prior in both numerator and \(\alpha\)** (`pop_z_mode=comoving_uniform` and a minimal mass model) and widen the \(H_0\) grid to avoid boundary artifacts.
3. Only after (1)–(2): consider an injection–recovery calibration (SBC for the siren score) to measure the false-positive rate under GR truth.

---

## Gate‑2 injection‑recovery calibration (2026‑02‑02)

### Large closed‑loop suite (GR truth \(H_0=70\); selection‑on)

Run: `outputs/gate2_calibration_combo_20260202_225949UTC/inj_suite_big/`

Configuration highlights:
- \(z_{\max}=0.62\), \(N_{\rm rep}=256\), \(N_{\rm events}=50\) per rep
- `det_model=snr_binned`, `weight_mode=inv_sampling_pdf`
- `pop_z_mode=comoving_uniform`, `pop_mass_mode=powerlaw_peak_q_smooth`
- synthetic PE: `pe_obs_mode=noisy`, `pe_synth_mode=likelihood_resample`, `pe_n_samples=10000`

Aggregate (from `tables/suite_aggregate.json`):
- `bias_p50_on_mean ≈ +0.90 km/s/Mpc`, `bias_p50_on_sd ≈ 4.21`
- `coverage_68_on ≈ 0.625`, `coverage_95_on ≈ 0.926`
- No event drops (`skipped_on_mean = 0.0`, `used_on_min = used_on_max = 50`)

Interpretation: the Gate‑2 control is *approximately* unbiased and interior‑peaked in the closed loop, but still shows mild undercoverage at this replicate count.

### Synthetic‑PE width tuning (stress test for prior‑dominated posteriors)

Run: `outputs/gate2_inj_pe_tune_20260202_230605UTC/`

We scanned `dl_frac_sigma_floor` holding all other knobs fixed (128 reps each, 50 events each).
Summary:

| `dl_frac_sigma_floor` | bias\_mean (km/s/Mpc) | cov68 | cov95 |
|---:|---:|---:|---:|
| 0.05 | +0.98 | 0.680 | 0.953 |
| 0.10 | −0.28 | 0.719 | 0.945 |
| 0.15 | −3.01 | 0.516 | 0.930 |
| 0.20 | −7.32 | 0.359 | 0.688 |
| 0.25 | −12.88 | 0.141 | 0.430 |

Takeaway: once the synthetic PE posteriors become too prior‑dominated (large distance‑likelihood widths), the finite‑sample hierarchical PE reweighting becomes badly biased and under‑covers. For audit‑grade Gate‑2 calibration, keep `dl_frac_sigma_floor` small (0.05–0.10) *or* raise `pe_n_samples` and/or implement a variance/bias control scheme for the \(\log\langle w\rangle\) estimator.

### Large suite check at `dl_frac_sigma_floor=0.10`

Run: `outputs/gate2_suite_big_floor010_20260202_231312UTC/suite/`

Aggregate:
- `bias_p50_on_mean ≈ −0.57 km/s/Mpc`
- `coverage_68_on ≈ 0.609`, `coverage_95_on ≈ 0.914`

Interpretation: at this replicate count, `dl_frac_sigma_floor=0.10` did not improve coverage relative to 0.05.

---

## Gate‑2 real‑PE control update (2026‑02‑02; 33 events)

Run: `outputs/gate2_calibration_combo_20260202_225949UTC/`

Key observation:
- For **selection‑on**, `weight_mode=none` is not a meaningful physical control (it treats the injection set as population‑drawn) and produces boundary‑peaked posteriors.
- With `weight_mode=inv_sampling_pdf` *and* population weights enabled (`pop_z_mode=comoving_uniform`, `pop_mass_mode=powerlaw_peak_q_smooth`), the GR \(H_0\) posterior becomes **interior‑peaked**:
  - `outputs/gate2_calibration_combo_20260202_225949UTC/gate2_real_pop/json/gr_h0_selection_on_inv_sampling_pdf.json`:
    - `H0_map=56.5` (not at edge), `p50≈59.6`, `p84≈82.7`.

Interpretation: Gate‑2 can be made interior‑peaked on real PE under the hierarchical‑PE + injection‑weighted selection model, but the resulting \(H_0\) control remains strongly population‑conditioned and is not yet a “safe” quantitative null unless population hyperparameters are externally anchored or explicitly marginalized.

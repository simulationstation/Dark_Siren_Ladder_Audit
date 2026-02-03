# Status — Dark Siren Ladder Audit (spectral-only “ghost” isolation)

Date: **2026-02-03**  
Repo: `simulationstation/Dark_Siren_Ladder_Audit` (local path: `/home/primary/Dark_Siren_Ladder_Audit`)

## 0) One-sentence objective

Build a **publication-grade audit system** that isolates and diagnoses **selection/population miscalibration (“ghosts”)** in *spectral-only* dark-siren inference, **before** anyone interprets model preferences as GW evidence.

## 1) What we mean by “ghost”

In hierarchical GW analyses, a model can “win” because of the **selection normalization** term:

\[
\ln p(\{d_i\}\mid M)=\sum_i \ln \left[\int \mathcal{L}(d_i\mid \theta)\,\pi(\theta\mid M)\,d\theta\right]\;-\;N\ln\alpha(M),
\]

where \(\alpha(M)\) is the detectability/selection factor under the *same* population assumptions.

A “ghost” is present when the **total preference** for a model is driven mainly (or entirely) by the \(-\ln\alpha\) term (or by internal inconsistencies between PE priors / population priors / injections), rather than by the event data term.

We are **not** treating any positive “score” as evidence. The goal is to determine *exactly which modeling choice creates a preference* and to make the GR controls pass under a self-consistent model.

## 2) Current state (where we are)

### 2.1 Implemented core capabilities

**Hierarchical PE likelihood from public PE samples**
- Implements a PE-sample reweighting path that divides out a realistic analytic approximation to the PE prior (not just a distance prior), including Jacobians for mass-coordinate changes and \(dz/dd_L\).
- Key module: `src/entropy_horizon_recon/dark_sirens_hierarchical_pe.py`
- Importance-weight stabilization and diagnostics (ESS, truncation, PSIS): `src/entropy_horizon_recon/importance_sampling.py`

**Gate‑2: GR \(H_0\) selection-on control plumbing**
- GR-only \(H_0\) grid posterior computed with the same hierarchical PE machinery and the same selection model \(\alpha(H_0)\).
- Key module: `src/entropy_horizon_recon/dark_siren_h0.py`
- Primary runner: `scripts/run_siren_gate2_gr_h0_control.py`

**Injection-recovery calibration loop**
- Synthetic injection → synthetic PE generation → hierarchical inference → calibration metrics.
- Includes P–P diagnostics for truth percentiles inside synthetic posteriors (distance + mass coordinates).
- Key module: `src/entropy_horizon_recon/siren_injection_recovery.py`
- Runners: `scripts/run_siren_injection_recovery.py`, `scripts/run_siren_injection_recovery_suite.py`

**Run discipline / observability**
- Detached-run rules and artifacts documented in `AUDIT.md` + `AGENTS.md`.
- Status helper: `scripts/audit_status.py` (scans `outputs/**/pid.txt` and tails `run.log`).

### 2.2 What the latest “real-data audit bundle” shows

The most important snapshot is bundled in:
- `2-1-D/` (copied from the run `outputs/dark_siren_bigtests_20260202_013848UTC`)

In that bundle, the hierarchical PE “spectral-only + selection” decomposition shows the classic ghost pattern:
- the **data term** (hierarchical PE integral) tends to prefer GR on average,
- while the **selection term** (\(-\ln \alpha\)) can dominate and make totals look positive.

Also, the GR \(H_0\) control in that run is **not yet acceptable** (selection-on posterior collapses to a grid edge), so we are not “past Gate‑2” on real data yet.

### 2.3 Injection-recovery calibration status (closed-loop)

There are multiple completed suites under `outputs/` (gitignored, but present locally). Two representative aggregates:

- `outputs/gate2_suite_o3b_peak_smooth_jacfix_nopdet_20260202_211955UTC/tables/suite_aggregate.json`  
  bias p50(mean) ≈ **+0.93 km/s/Mpc** on \(H_0=70\) (32 reps; 60 events/rep)

- `outputs/gate2_suite_moderate_20260202_224746UTC/tables/suite_aggregate.json`  
  bias p50(mean) ≈ **+1.22 km/s/Mpc**, coverage(68%) ≈ **0.6875** (32 reps; 25 events/rep)

A *known failure mode* existed for wide \(H_0\) grids when \(z_{\max}\) was too small and QC silently skipped events; that produced a large apparent bias. We added an **auto \(z_{\max}\)** guard to prevent this truncation artifact.

## 3) What’s in this repo (layout)

Tracked (code/docs):
- `src/entropy_horizon_recon/` — package code (includes siren audit machinery)
- `scripts/` — runners + audit utilities
- `tests/` — unit/smoke tests (not all require siren data)
- `AUDIT.md`, `SIREN_AUDIT_PROJECT.md`, `FINDINGS/` — design + narrative

Present locally but gitignored (large artifacts):
- `data/` — GW PE caches and injection files (copied from the parent workspace)
- `outputs/` — all run outputs copied from the parent workspace
- `2-1-D/`, `2-1-d-siren/`, `2-1-c-m/`, `master_siren_x/` — report bundles

## 4) The plan (where we are going)

### Priority A — make Gate‑2 pass *robustly*

Gate‑2 requirement:
- Under GR, with selection **enabled** and a self-consistent population+selection model,
  the \(H_0\) posterior must be **interior-peaked** (not stuck on grid edges) and
  injection-recovery must be **unbiased** (P–P ~ uniform; coverage ~ nominal).

Immediate actions:
1. Re-run the previously “wide-grid” injection-recovery suite using:
   - `z_max_mode=auto` (already implemented),
   - QC **fail-loud** (no silent skipping),
   - confirm bias ~0 and interior peaks across the entire grid.

2. Promote “suite gating” to hard PASS/FAIL:
   - fail reps that skip events,
   - fail reps whose MAP is at grid boundaries,
   - write explicit `pass_fail.json` per rep + aggregate.

### Priority B — selection realism upgrade (biggest remaining robustness gap)

Current selection uses a 1D detectability model (SNR-binned).

Next upgrade:
- Calibrate a **2D detectability model** \(p_{\rm det}(\mathrm{SNR}, \mathcal{M}_{\rm det})\) (or similar),
  from injections, and use it consistently in:
  - \(\alpha(H_0)\) computation (GR control),
  - injection-recovery generator,
  - any μ-vs-GR comparisons (even if we are “spectral-only”).

This is the most likely lever to fix the edge-peaking behavior on real data by removing mass-dependent selection mismatch.

### Priority C — “nulls matched to the claim”

We are **spectral-only**, so nulls should target distance–redshift consistency mechanisms, not host association.

Add nulls that actually destroy spectral information:
- cross-event swaps of \((d_L,\mathcal{M})\) samples,
- prior-resample nulls using the PE prior,
- redshift scrambling under the population prior,
- injection-based “ghost simulations” (generate with one selection threshold, analyze with another) to show the ladder catches the failure.

### Priority D — only then: science-facing runs

Once Gate‑2 passes and selection is self-consistent:
- run the full ladder on real data,
- report the decomposition (\(\ln \mathcal{L}_{\rm data}\) vs \(-\ln\alpha\)) as the primary audit product,
- treat any model preference as “selection behavior” unless the data term also prefers it and nulls pass.

## 5) Quick operational notes

**Status checking (detached runs)**
- `./.venv/bin/python scripts/audit_status.py outputs --tail-lines 40`

**Don’t attach long jobs to a terminal**
- Follow `AGENTS.md`: write `job.sh` into an output dir, start with `setsid ... &`, log to `run.log`, write PID to `pid.txt`.

**Environment**
- This repo mirrors the parent project’s packaging via `pyproject.toml`.
- Typical dev install: `pip install -e .[sirens,dev]`

## 6) What “done” looks like

We are “ready” when:
- Gate‑2 (GR \(H_0\) selection-on) is interior-peaked and injection-recovery is unbiased across realistic grids,
- selection modeling is consistent (same population assumptions across PE de-prior, population prior, and injections),
- matched nulls reliably kill ghost mechanisms,
- only then we interpret anything as an astrophysical/cosmological statement.


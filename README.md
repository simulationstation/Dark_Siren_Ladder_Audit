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
  --injection-file data/cache/gw/zenodo/7890437/endo3_mixture-LIGO-T2100113-v12-1256655642-12905976.hdf5 \
  --plot
```

## Gate‑2 population anchoring (recommended)

The GR \(H_0\) selection-on control is sensitive to population assumptions. For a “best available” baseline, you can marginalize over **LVK GWTC‑3 population hyperposterior draws** (PowerLaw+Peak + powerlaw redshift), using the O3a population data release bilby-result JSON:

```bash
./.venv/bin/python scripts/run_siren_gate2_gr_h0_popmarginal.py \
  --pop-mode lvk_hyper \
  --lvk-pop-result-json data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/o3a_population_data_release/Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json \
  --lvk-n-draws 200
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

## Gate‑2 ladder (toy cancellation + incremental complexity)

To avoid “eternal knob turning”, `scripts/run_siren_gate2_ladder.py` runs:
- a **known-answer toy cancellation check** (prior-only PE + p_det in numerator) and
- a small **injection-recovery ladder** that adds complexity rung-by-rung (z prior, mass model, weight smoothing, detection model).

```bash
./.venv/bin/python scripts/run_siren_gate2_ladder.py --n-events 25 --pe-n-samples 5000 --n-proc 0
```

Outputs include `outputs/siren_gate2_ladder_*/ladder.csv` (quick scan) and `ladder_summary.json` (machine-readable).

## Third-party software

This project uses (and optionally benchmarks against) open-source gravitational-wave community software, including:
- `gwpopulation` (GWPopulation contributors)
- `wcosmo` (wcosmo contributors)
- `icarogw` (ICAROGW contributors; installed in an isolated venv here)
- `ligo.skymap`, `bilby`, `healpy`, `astropy`, `h5py` (and other scientific Python dependencies)

# Dark-siren hierarchical PE: selection-α sensitivity (2026-02-02)

This note documents a **selection-normalization sensitivity sweep** for the *hierarchical PE* (catalog-free) dark-siren scoring.

## Context

The QC-gated hierarchical PE runs consistently show:

- **data-only** preference: \(\Delta\mathrm{LPD}_\mathrm{data}<0\) (GR preferred),
- **selection-corrected** preference: \(\Delta\mathrm{LPD}_\mathrm{total}=\Delta\mathrm{LPD}_\mathrm{data}+\Delta\mathrm{LPD}_\mathrm{sel}\) can become \(>0\),

so the sign hinges on the **selection normalization** \(\alpha(\mathrm{model})\).

To quantify how fragile this is, we re-used cached per-draw event log-likelihood stacks and recomputed \(\alpha\) under several variants.

## Inputs / outputs

- Battery run (hierarchical PE, QC gate `--hier-min-good-frac 0.1 --hier-bad-sample-mode skip`):
  - `outputs/dark_siren_battery_hier_qc_stack_20260202_001103UTC/`
  - Subdir used for sweep: `outputs/dark_siren_battery_hier_qc_stack_20260202_001103UTC/hierarchical`
- Sweep script:
  - `scripts/sweep_dark_siren_hier_selection_sensitivity.py`
- Variant list:
  - `run_cards/dark_siren_hier_alpha_sweep_variants.json`
- Results:
  - `outputs/dark_siren_battery_hier_qc_stack_20260202_001103UTC/hierarchical/alpha_selection_sensitivity_sweep.json`
  - `outputs/dark_siren_battery_hier_qc_stack_20260202_001103UTC/hierarchical/figures/alpha_selection_sensitivity_sweep.png`

## Results (ΔLPD_total across 5 EM seeds)

Aggregate means (±sd across seeds), pulled from `alpha_selection_sensitivity_sweep.json`:

| variant | mean ΔLPD_total | sd |
|---|---:|---:|
| `baseline_saved` | +0.594 | 0.057 |
| `no_selection` | -1.443 | 0.689 |
| `det_threshold_calibrated` | -0.049 | 0.070 |
| `det_threshold_snr12` | +1.411 | 0.109 |
| `weight_none` | +6.915 | 0.703 |
| `popz_none` | -1.434 | 0.331 |
| `popz_powerlaw_k2` | +1.939 | 0.156 |
| `popmass_powerlaw_q` | +2.135 | 0.149 |

Notes:
- The sweep **recomputes only the selection term** \(\alpha(\mathrm{model})\) while reusing the cached hierarchical PE log-likelihood stacks for the data term.
- Variants that change population knobs that also enter the *hierarchical PE data term* (e.g. `popz_*`, and any mass-population choices if the hierarchical likelihood is mass-enabled) should be interpreted as **selection-only sensitivity**. A fully consistent “population” sensitivity test requires re-scoring the hierarchical PE log-likelihood under the same population choice.

Key interpretation:

- The *same fixed* data term (from the hierarchical PE stacks) can be made to “favor μ” or “favor GR” depending on \(\alpha\) assumptions.
- Even within superficially “reasonable” detection models, the sign can change (e.g. `det_threshold_calibrated` vs `baseline_saved`).

## Implication

At present, **a positive \(\Delta\mathrm{LPD}_\mathrm{total}\)** in hierarchical PE mode is **not evidence** by itself, because it is not robust to plausible (and in some cases simply different) selection modeling choices.

The correct next step is **selection validation**, not “more events”:

1) pick a published GR control target (a known dark-siren \(H_0\) baseline) and require our selection machinery reproduces it under GR, and
2) only then interpret \(\Delta\mathrm{LPD}\) comparisons between μ-propagation and GR.

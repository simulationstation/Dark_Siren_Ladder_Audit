# Dark-siren mechanism diagnostic (host-alignment null vs spectral/selection)

Date (UTC): 2026-02-01

This note records a **mechanism** (not “final score”) diagnostic for the dark-siren propagation
test. The goal is to separate:

- **Host-association** information (sky–galaxy alignment), from
- **Spectral / selection** information (global distance–redshift structure and/or selection proxy effects).

We use the newer `ΔLPD` **decomposition** written by `scripts/run_dark_siren_gap_test.py`:

- `ΔLPD_data`: comparison from the *data term only*
- `ΔLPD_sel`: additional contribution from the selection normalization `α(model)`
- `ΔLPD_total = ΔLPD_data + ΔLPD_sel`

All runs here use a small number of EM posterior draws (e.g. `--max-draws 4`) for speed; they are
not intended as final production results.

---

## 1) Catalog mode (PE hist + GLADE+): rotation null does **not** erase the score

Event: `GW200220_061928`

### Real sky (catalog, PE hist)

- Output: `outputs/diagnostics/ds_catalog_real_gw200220/`
- Score JSON: `outputs/diagnostics/ds_catalog_real_gw200220/tables/event_scores_M0_start202.json`
- Result:
  - `ΔLPD_data = +2.0736`
  - `ΔLPD_sel  = +0.0174`
  - `ΔLPD_total= +2.0910`

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_gap_test.py \
  --run-dir outputs/finalization/highpower_multistart_v2/M0_start202 \
  --skymap-dir data/cache/gw/zenodo/5546663/extracted/skymaps \
  --glade-index data/processed/galaxies/gladeplus/index_nside128_wlumB_zmax0.3 \
  --events GW200220_061928 --max-events 1 --max-draws 4 --n-proc 1 \
  --max-area-deg2 5000 --max-gal 3000000 \
  --gw-data-mode pe --pe-like-mode hist --pe-max-samples 50000 \
  --null-mode none \
  --out outputs/diagnostics/ds_catalog_real_gw200220
```

### Rotated-sky null (breaks host alignment)

- Output: `outputs/diagnostics/ds_catalog_rot_gw200220/`
- Score JSON: `outputs/diagnostics/ds_catalog_rot_gw200220/tables/event_scores_M0_start202.json`
- Result:
  - `ΔLPD_data = +2.0399`
  - `ΔLPD_sel  = +0.0159`
  - `ΔLPD_total= +2.0558`

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_gap_test.py \
  --run-dir outputs/finalization/highpower_multistart_v2/M0_start202 \
  --skymap-dir data/cache/gw/zenodo/5546663/extracted/skymaps \
  --glade-index data/processed/galaxies/gladeplus/index_nside128_wlumB_zmax0.3 \
  --events GW200220_061928 --max-events 1 --max-draws 4 --n-proc 1 \
  --max-area-deg2 5000 --max-gal 3000000 \
  --gw-data-mode pe --pe-like-mode hist --pe-max-samples 50000 \
  --null-mode rotate_pe_sky \
  --out outputs/diagnostics/ds_catalog_rot_gw200220
```

**Conclusion:** for this event, the rotation null barely changes the score. Therefore, the large
positive preference in the catalog integral is **not** safely interpretable as host association.

---

## 2) Hierarchical PE mode (no catalog): modest spectral/selection preference

This mode uses PE posterior samples with the event’s analytic priors. It always divides out the
analytic PE prior for `luminosity_distance`, and (optionally) can divide out PE priors for
`chirp_mass` and `mass_ratio` if a simple mass population model is enabled. It does **not** use the
galaxy catalog.

Event: `GW200220_061928`

### No selection

- Output: `outputs/diagnostics/ds_hier_noSel_gw200220/`
- Score JSON: `outputs/diagnostics/ds_hier_noSel_gw200220/tables/event_scores_M0_start202.json`
- Result:
  - `ΔLPD_data = +0.2013` (selection disabled, so total=data)

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_gap_test.py \
  --run-dir outputs/finalization/highpower_multistart_v2/M0_start202 \
  --skymap-dir data/cache/gw/zenodo/5546663/extracted/skymaps \
  --events GW200220_061928 --max-events 1 --max-draws 4 \
  --gw-data-mode pe --pe-like-mode hierarchical --pe-max-samples 20000 \
  --selection-injections-hdf none \
  --out outputs/diagnostics/ds_hier_noSel_gw200220
```

### With selection

- Output: `outputs/diagnostics/ds_hier_sel_gw200220/`
- Score JSON: `outputs/diagnostics/ds_hier_sel_gw200220/tables/event_scores_M0_start202.json`
- Result:
  - `ΔLPD_data = +0.2013`
  - `ΔLPD_sel  = +0.1157`
  - `ΔLPD_total= +0.3170`

Command:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_dark_siren_gap_test.py \
  --run-dir outputs/finalization/highpower_multistart_v2/M0_start202 \
  --skymap-dir data/cache/gw/zenodo/5546663/extracted/skymaps \
  --events GW200220_061928 --max-events 1 --max-draws 4 \
  --gw-data-mode pe --pe-like-mode hierarchical --pe-max-samples 20000 \
  --selection-injections-hdf auto \
  --out outputs/diagnostics/ds_hier_sel_gw200220
```

Event: `GW200308_173609` (hierarchical only)

- Output: `outputs/diagnostics/ds_hier_sel_gw200308/`
- Score JSON: `outputs/diagnostics/ds_hier_sel_gw200308/tables/event_scores_M0_start202.json`
- Result:
  - `ΔLPD_data = +0.1462`
  - `ΔLPD_sel  = +0.1153`
  - `ΔLPD_total= +0.2615`

**Conclusion:** the PE-hierarchical (spectral/selection) preference exists but is modest for these
events. Large catalog-based ΔLPD values for wide-localization events require additional controls
before being interpreted as host association.

### Mass-coupling sensitivity check (hierarchical mode)

We ran a tiny hierarchical null battery for `GW200220_061928` (8 EM draws; 10k PE samples) to test
mass–distance coupling sensitivity.

Distance-only (`--selection-pop-mass-mode none`):
- Output: `outputs/diagnostics/hier_battery_gw200220_nomass_20260201_221157UTC/`
- Result: `ΔLPD_total = +0.3714` and all shuffle modes are identical to real (as expected when masses are not used).

Mass-enabled (`--selection-pop-mass-mode powerlaw_q`):
- Output: `outputs/diagnostics/hier_battery_gw200220_20260201_220902UTC/`
- Real: `ΔLPD_total = +0.7610`
- `shuffle_dl` null: `ΔLPD_total = +2.4474` (larger than real)

**Conclusion:** mass-enabled hierarchical scoring is currently highly sensitive to mass–distance
coupling assumptions and needs additional validation (hyperparameter marginalization + injection-
validated selection) before being treated as evidence.

---

## 3) PE analytic distance-prior division (PE-hist mode)

We added `--gw-distance-prior-mode pe_analytic` to divide out the **event’s analytic PE prior**
`π_PE(dL)` instead of a heuristic prior.

For `GW200220_061928`, the PEDataRelease analytic prior for `luminosity_distance` is:

- `PowerLaw(alpha=2, minimum=100, maximum=25000)` (i.e. \(π(d_L)\propto d_L^2\) on a bounded interval)

So for this event, `pe_analytic` matches the default `dL_powerlaw(k=2)` **in the interior**, and the
score is unchanged (as expected). The `pe_analytic` option still matters for events whose PE prior
differs from \(d_L^2\), and for correct handling of hard-cutoff support.

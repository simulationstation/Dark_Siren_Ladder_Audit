# Gate‑2 real‑data diagnosis — why GR \(H_0\) runs high (and what it means)

Date: **2026-02-05**  
Repo: `simulationstation/Dark_Siren_Ladder_Audit`

## What this note is (and is not)

This note explains a persistent observation in **real-data Gate‑2** runs:

- The GR \(H_0\) posterior can land at **very high \(H_0\)** (often near the top of a wide grid), even when
  the selection factor \(\alpha(H_0)\) is implemented correctly.

This is **not** a claim about cosmology. It is an audit / mechanism diagnosis: *which term is pushing the posterior, and why*.

## The concrete run we analyzed

Representative LVK-hyperposterior draw Gate‑2 JSON:

- `outputs/siren_gate2_popmarg_realdata_lvk50_20260205_005345UTC/json/gr_h0_selection_on_lvk_0000.json`
  - \(N=25\) events
  - \(H_0\) grid: 40–200
  - `pop_z_mode=comoving_powerlaw` (here `k=0` → comoving-uniform)
  - `pop_mass_mode=powerlaw_peak_q_smooth` (PowerLaw+Peak-like)
  - selection: `det_model=snr_mchirp_q_binned`, injection-rescaling proxy

## Decomposition: event term vs selection term

Using the Gate‑2 jig:

```bash
./.venv/bin/python scripts/gate2_jig.py one \
  --json outputs/siren_gate2_popmarg_realdata_lvk50_20260205_005345UTC/json/gr_h0_selection_on_lvk_0000.json \
  --top-n 8
```

we get:

- **event-sum term** increases strongly with \(\log H_0\)
- **selection term** \(-N\log\alpha(H_0)\) decreases with \(\log H_0\) (i.e. it pushes *down* in \(H_0\))
- the **event term wins**, so total prefers high \(H_0\)

For `.../gr_h0_selection_on_lvk_0000.json` specifically:

- effective event scaling is roughly \(H_0^{1.14}\) **per event** (endpoint heuristic from the jig),
- and the posterior is driven by a handful of events with large positive “\(b\) exponents” (nearby/high-information events).

Event dominance sanity check:

```bash
./.venv/bin/python scripts/gate2_jig.py filter \
  --json outputs/siren_gate2_popmarg_realdata_lvk50_20260205_005345UTC/json/gr_h0_selection_on_lvk_0000.json \
  --out FINDINGS/gate2_filter_realdata_lvk0000.csv
```

In this example, **dropping the top few positive‑\(b\)** events moves the posterior substantially (p50 falls from \(\sim\!180\) toward \(\sim\!140\)).

## Selection uncertainty does not fix it

We propagated a basic finite‑injection uncertainty model for \(\alpha(H_0)\) (Beta draws over the binned \(p_{\rm det}\) table) for this same draw:

- output dir: `outputs/siren_gate2_seluncert_lvk0000_20260205_012401UTC/`
- summary file: `outputs/siren_gate2_seluncert_lvk0000_20260205_012401UTC/json/gate2_seluncert_posterior.json`

Result: the marginalized posterior remains essentially unchanged (still high‑\(H_0\)).

Interpretation: the extreme \(H_0\) behavior here is **not** driven by underestimated \(\alpha\) uncertainty.

## Why this happens (structural explanation)

Gate‑2 (as implemented here) is a **spectral-only** control: it attempts to infer \(H_0\) from GW-inferred distances *plus* assumed population structure:

- a **redshift-rate density model** \(p(z)\) (e.g. comoving-uniform / comoving-powerlaw),
- a **mass model** in source frame (so \(m_{\rm src}=m_{\rm det}/(1+z)\)),
- a hard **support window** \(z\le z_{\max}\),
- selection correction via \(\alpha(H_0)\).

For low redshift, comoving-uniform priors imply \(p(z)\propto z^2\), and the mapping \(z=z(d_L;H_0)\) makes the transformed prior in distance carry an \(H_0\) dependence (via \(z\propto H_0\) and \(dz/dd_L\propto H_0\)).

That produces a generic tendency for the **numerator (event) term** to increase with \(H_0\) for nearby events, until the \(z_{\max}\) support cut (and/or selection) pushes back.

So, on real data:

- if a few events have distance posteriors that keep substantial support well inside \(z_{\max}\) even at high \(H_0\),
  they can provide a strong monotonic push upward,
- while very distant events only weakly counteract this (often with low ESS / rare-tail support).

This is an *assumption sensitivity* / *identifiability* issue, not an “alpha implementation” bug.

## What this implies for “Hubble tension investigation”

To estimate a **physical** \(H_0\) robustly (and claim anything about “tension”), we need to move beyond “distance-only + fixed population knobs”:

1. **Use additional redshift information** (galaxy catalogs / cross-correlation), *or*
2. **Marginalize aggressively** over population redshift evolution and selection uncertainty (and show closure/coverage via injection-recovery), *or*
3. Treat Gate‑2 as what it is: a **ghost isolator / consistency control**, not a standalone \(H_0\) estimator on real data.

Next concrete actions (shortest path to clarity):

- Run Gate‑2 with and without mass priors and compare per-event \(b\) exponents (is the push mostly from the comoving prior, or from mass–redshift coupling?).
- Run injection-recovery closure **for the exact real-data configuration** (same \(z_{\max}\), \(p_{\rm det}\) proxy, H0 grid) to quantify bias/coverage under controlled truth.
- Only after that, proceed to catalog-aware dark-siren inference if the goal is a cosmology-facing \(H_0\) posterior.


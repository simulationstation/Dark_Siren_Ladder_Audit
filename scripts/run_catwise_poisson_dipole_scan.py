#!/usr/bin/env python3
"""
CatWISE/Secrest quasar dipole: Poisson-likelihood scan vs faint cut.

This script is the "do what Secrest asked" version of the faint-limit diagnostic:
  - use a Poisson likelihood (not a linear/vector-sum estimator) on HEALPix counts
  - explicitly model a WISE scan proxy via an ecliptic-latitude template
  - scan the faint cut W1_max and record dipole amplitude/direction + alpha_edge

Model (per unmasked pixel p, using a log-link Poisson GLM):
  N_p ~ Poisson(mu_p)
  log mu_p = beta0 + b · n_p + Σ_k c_k T_{k,p}

For small dipoles, the fractional dipole amplitude is D ≈ |b| and the dipole axis is along b.

Outputs:
  - scan.json with per-cut results
  - optional PNG summaries (dipole vs W1_max, alpha_edge vs W1_max, projection vs alpha_edge)

Notes:
  - This is a count-level model; it does not use per-object redshifts.
  - Templates here are *proxies* (e.g. pixel-mean w1cov); for publication, prefer an independent
    depth/exposure map (e.g. unWISE Nexp) as an offset/template.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: Iterable[float], b_deg: Iterable[float]) -> np.ndarray:
    l = np.deg2rad(np.asarray(l_deg, dtype=float) % 360.0)
    b = np.deg2rad(np.asarray(b_deg, dtype=float))
    cosb = np.cos(b)
    return np.column_stack([cosb * np.cos(l), cosb * np.sin(l), np.sin(b)])


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


def compute_alpha_edge_from_cdf(w1: np.ndarray, w1_max: float, dm: float) -> float:
    """alpha_edge ≡ d ln N / d m at the faint cut, estimated from cumulative counts."""
    w1 = np.asarray(w1, dtype=float)
    dm = float(dm)
    if dm <= 0:
        raise ValueError("dm must be > 0")
    n1 = int(np.sum(w1 <= float(w1_max)))
    n0 = int(np.sum(w1 <= float(w1_max - dm)))
    if n0 <= 0 or n1 <= 0:
        return float("nan")
    return float((math.log(n1) - math.log(n0)) / dm)


def fit_poisson_glm(X: np.ndarray, y: np.ndarray, *, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Poisson GLM (log link) via L-BFGS.
    Returns (beta, cov_beta_approx) where cov is Fisher^{-1} if invertible.
    """
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)

    mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = math.log(mu0)

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ beta
        eta = np.clip(eta, -25.0, 25.0)
        mu = np.exp(eta)
        nll = float(np.sum(mu - y * eta))
        grad = X.T @ (mu - y)
        return nll, np.asarray(grad, dtype=float)

    def f(beta: np.ndarray) -> float:
        return fun_and_grad(beta)[0]

    def g(beta: np.ndarray) -> np.ndarray:
        return fun_and_grad(beta)[1]

    res = minimize(
        f,
        beta0,
        jac=g,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12},
    )
    beta = np.asarray(res.x, dtype=float)

    # Fisher / covariance approximation: (X^T diag(mu) X)^{-1}
    try:
        eta = np.clip(X @ beta, -25.0, 25.0)
        mu = np.exp(eta)
        fisher = X.T @ (mu[:, None] * X)
        cov = np.linalg.inv(fisher)
    except Exception:  # noqa: BLE001
        cov = None
    return beta, cov


@dataclass(frozen=True)
class DipoleRow:
    w1_max: float
    N_sources: int
    alpha_edge: float
    dipole_amp: float
    dipole_l_deg: float
    dipole_b_deg: float
    dipole_amp_sigma: float | None
    vec: np.ndarray
    cov_b: list[list[float]] | None
    vector_sum_amp: float
    vector_sum_l_deg: float
    vector_sum_b_deg: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "w1_max": float(self.w1_max),
            "N_sources": int(self.N_sources),
            "alpha_edge": float(self.alpha_edge),
            "vector_sum": {
                "amplitude": float(self.vector_sum_amp),
                "l_deg": float(self.vector_sum_l_deg),
                "b_deg": float(self.vector_sum_b_deg),
            },
            "dipole": {
                "amplitude": float(self.dipole_amp),
                "amplitude_sigma": None if self.dipole_amp_sigma is None else float(self.dipole_amp_sigma),
                "l_deg": float(self.dipole_l_deg),
                "b_deg": float(self.dipole_b_deg),
                "vec": [float(x) for x in np.asarray(self.vec, dtype=float).reshape(3)],
                "cov_b": self.cov_b,
            },
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
        help="CatWISE/Secrest FITS (expects l,b,w1,w1cov,elat,ebv).",
    )
    ap.add_argument("--outdir", default=None)

    ap.add_argument("--nside", type=int, default=32, help="HEALPix NSIDE for the Poisson fit.")
    ap.add_argument("--b-cut", type=float, default=30.0)
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-min", type=float, default=None)

    ap.add_argument("--w1max-grid", default="15.6,16.6,0.05", help="Grid spec 'start,stop,step' (inclusive).")
    ap.add_argument("--alpha-dm", type=float, default=0.05, help="Finite-diff step for alpha_edge.")
    ap.add_argument("--min-N", type=int, default=200_000, help="Skip cuts with fewer than this many sources.")
    ap.add_argument(
        "--footprint-w1-max",
        type=float,
        default=None,
        help=(
            "Define the pixel footprint mask using all sources with W1 <= footprint_w1_max "
            "(after the baseline cuts). Default: max(W1_max grid)."
        ),
    )
    ap.add_argument(
        "--exclude-mask-fits",
        default=None,
        help=(
            "Optional Secrest exclusion regions FITS (ra/dec/radius/use). When provided and found, "
            "regions with use=True are excluded (this matches the intended semantics of the file)."
        ),
    )
    ap.add_argument(
        "--exclude-mask-nside",
        type=int,
        default=512,
        help=(
            "HEALPix NSIDE used internally to apply the exclusion discs as a pixel mask to sources. "
            "Use a high value so small-radius discs do not over-mask (default: 512)."
        ),
    )

    ap.add_argument(
        "--template-set",
        choices=["none", "eclip", "basic"],
        default="eclip",
        help="Systematics templates in the Poisson fit: none | eclip(|sin(elat)|) | basic(eclip+ebv+w1cov).",
    )
    ap.add_argument("--max-iter", type=int, default=200, help="Max optimizer iterations per cut.")

    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/catwise_poisson_dipole_scan_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    # Heavy deps late.
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from astropy.table import Table

    tab = Table.read(args.catalog, memmap=True)
    l = np.asarray(tab["l"], dtype=float)
    b = np.asarray(tab["b"], dtype=float)
    w1 = np.asarray(tab["w1"], dtype=float)
    w1cov = np.asarray(tab["w1cov"], dtype=float) if "w1cov" in tab.colnames else None
    elat = np.asarray(tab["elat"], dtype=float) if "elat" in tab.colnames else None
    ebv = np.asarray(tab["ebv"], dtype=float) if "ebv" in tab.colnames else None

    if w1cov is None:
        raise SystemExit("catalog missing w1cov")
    if elat is None:
        raise SystemExit("catalog missing elat")
    if ebv is None:
        raise SystemExit("catalog missing ebv")

    base = np.isfinite(l) & np.isfinite(b) & np.isfinite(w1) & np.isfinite(w1cov) & np.isfinite(elat) & np.isfinite(ebv)
    base &= np.abs(b) > float(args.b_cut)
    base &= w1cov >= float(args.w1cov_min)
    if args.w1_min is not None:
        base &= w1 >= float(args.w1_min)

    l = l[base]
    b = b[base]
    w1 = w1[base]
    w1cov = w1cov[base]
    elat = elat[base]
    ebv = ebv[base]

    unit = lb_to_unitvec(l, b)

    # Precompute HEALPix pixels once (depends only on position).
    nside = int(args.nside)
    npix = hp.nside2npix(nside)
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(nside, theta, phi, nest=False)

    # Optional: apply Secrest exclusion regions as a pixel mask.
    excluded_pix: np.ndarray | None = None
    n_exclude_regions_used = 0
    ex_path = Path(str(args.exclude_mask_fits)) if args.exclude_mask_fits else None
    if ex_path and ex_path.exists():
        ex = Table.read(ex_path, memmap=True)
        if "use" in ex.colnames:
            ex = ex[np.asarray(ex["use"], dtype=bool)]
        if len(ex) > 0:
            n_exclude_regions_used = int(len(ex))
            # Convert exclusion centers to Galactic, then exclude discs using a high-NSIDE mask.
            nside_mask = int(args.exclude_mask_nside)
            if nside_mask < 1:
                raise SystemExit("--exclude-mask-nside must be >= 1")
            c = SkyCoord(ra=np.asarray(ex["ra"], dtype=float) * u.deg, dec=np.asarray(ex["dec"], dtype=float) * u.deg, frame="icrs")
            g = c.galactic
            l0 = np.asarray(g.l.deg, dtype=float)
            b0 = np.asarray(g.b.deg, dtype=float)
            rdeg = np.asarray(ex["radius"], dtype=float)
            ex_set: set[int] = set()
            for ll, bb, rr in zip(l0, b0, rdeg, strict=True):
                vec = hp.ang2vec(np.deg2rad(90.0 - bb), np.deg2rad(ll))
                disc = hp.query_disc(nside_mask, vec, np.deg2rad(float(rr)), nest=False)
                ex_set.update(int(x) for x in disc)
            if ex_set:
                excluded_pix = np.fromiter(ex_set, dtype=np.int64)
                # Apply to sources using the *mask* NSIDE pixels.
                pix_mask = hp.ang2pix(nside_mask, theta, phi, nest=False)
                keep = ~np.isin(pix_mask, excluded_pix)
                l = l[keep]
                b = b[keep]
                w1 = w1[keep]
                w1cov = w1cov[keep]
                elat = elat[keep]
                ebv = ebv[keep]
                unit = unit[keep]
                theta = theta[keep]
                phi = phi[keep]
                pix = hp.ang2pix(nside, theta, phi, nest=False)

    # Parse W1_max grid.
    start_s, stop_s, step_s = [s.strip() for s in str(args.w1max_grid).split(",")]
    start, stop, step = float(start_s), float(stop_s), float(step_s)
    if step <= 0:
        raise SystemExit("w1max-grid step must be > 0")
    ngrid = int(math.floor((stop - start) / step + 1e-9)) + 1
    w1max_values = start + step * np.arange(ngrid)

    footprint_w1_max = float(args.footprint_w1_max) if args.footprint_w1_max is not None else float(np.max(w1max_values))
    footprint_sel = w1 <= footprint_w1_max
    pix_foot = pix[footprint_sel]
    pix_in_foot = np.zeros(npix, dtype=bool)
    pix_in_foot[pix_foot] = True

    # Pixel centers for the footprint mask.
    foot_pix_idx = np.flatnonzero(pix_in_foot)
    th_c, ph_c = hp.pix2ang(nside, foot_pix_idx, nest=False)
    l_c = np.degrees(ph_c)
    b_c = 90.0 - np.degrees(th_c)
    pix_unit_c = lb_to_unitvec(l_c, b_c)

    # Templates evaluated on the footprint pixels.
    templates_foot: list[np.ndarray] = []
    template_names: list[str] = []

    if args.template_set in ("eclip", "basic"):
        # Independent ecliptic-latitude proxy from pixel centers: |sin(elat)|
        sc = SkyCoord(l=l_c * u.deg, b=b_c * u.deg, frame="galactic")
        elat_c = np.asarray(sc.barycentrictrueecliptic.lat.deg, dtype=float)
        # Use |elat| (deg) to mirror common WISE scan-depth systematics parameterizations.
        t_ecl = np.abs(elat_c)
        t_ecl = zscore(t_ecl, np.isfinite(t_ecl))
        templates_foot.append(t_ecl)
        template_names.append("abs_elat_deg")

    if args.template_set == "basic":
        # EBV + coverage templates (pixel means from the fixed footprint sample).
        # These are not perfectly independent of the catalog, but are held fixed across W1_max scans.
        Np_foot = np.bincount(pix_foot, minlength=npix).astype(float)
        valid_foot = Np_foot > 0

        s_ebv = np.bincount(pix_foot, weights=ebv[footprint_sel], minlength=npix)
        t = np.zeros(npix, dtype=float)
        t[valid_foot] = s_ebv[valid_foot] / Np_foot[valid_foot]
        t = zscore(t, valid_foot)
        templates_foot.append(t[foot_pix_idx])
        template_names.append("ebv_mean_foot")

        s_cov = np.bincount(pix_foot, weights=w1cov[footprint_sel], minlength=npix)
        t = np.zeros(npix, dtype=float)
        t[valid_foot] = s_cov[valid_foot] / Np_foot[valid_foot]
        t = zscore(t, valid_foot)
        templates_foot.append(t[foot_pix_idx])
        template_names.append("w1cov_mean_foot")

    rows: list[DipoleRow] = []
    for w1_max in w1max_values:
        sel = w1 <= float(w1_max)
        N = int(np.sum(sel))
        if N < int(args.min_N):
            continue

        alpha_edge = compute_alpha_edge_from_cdf(w1, float(w1_max), float(args.alpha_dm))

        # Build counts on the *fixed footprint* pixel set (includes zeros).
        pix_sel = pix[sel]
        Np = np.bincount(pix_sel, minlength=npix).astype(float)
        y = Np[foot_pix_idx]

        # Vector-sum (object-level) estimator for comparison.
        dvec_vs = 3.0 * unit[sel].mean(axis=0)
        vs_amp = float(np.linalg.norm(dvec_vs))
        vs_l, vs_b = vec_to_lb(dvec_vs)

        X_cols = [np.ones_like(y), pix_unit_c[:, 0], pix_unit_c[:, 1], pix_unit_c[:, 2]]
        if templates_foot:
            X_cols.extend(templates_foot)
        X = np.column_stack(X_cols)

        beta, cov = fit_poisson_glm(X, y, max_iter=int(args.max_iter))
        bvec = np.asarray(beta[1:4], dtype=float)
        amp = float(np.linalg.norm(bvec))
        l_d, b_d = vec_to_lb(bvec)

        amp_sigma: float | None = None
        cov_b_list: list[list[float]] | None = None
        if cov is not None and np.all(np.isfinite(cov[1:4, 1:4])) and amp > 0:
            cov_b = np.asarray(cov[1:4, 1:4], dtype=float)
            u = bvec / amp
            amp_sigma = float(math.sqrt(max(0.0, float(u @ cov_b @ u))))
            cov_b_list = [[float(x) for x in row] for row in cov_b.tolist()]

        rows.append(
            DipoleRow(
                w1_max=float(w1_max),
                N_sources=N,
                alpha_edge=float(alpha_edge),
                dipole_amp=float(amp),
                dipole_l_deg=float(l_d),
                dipole_b_deg=float(b_d),
                dipole_amp_sigma=amp_sigma,
                vec=bvec,
                cov_b=cov_b_list,
                vector_sum_amp=vs_amp,
                vector_sum_l_deg=vs_l,
                vector_sum_b_deg=vs_b,
            )
        )

    if len(rows) < 3:
        raise SystemExit(f"Too few cuts retained ({len(rows)}). Try lowering --min-N or widening --w1max-grid.")

    out = {
        "inputs": {
            "catalog": str(args.catalog),
            "nside": int(args.nside),
            "b_cut": float(args.b_cut),
            "w1cov_min": float(args.w1cov_min),
            "w1_min": None if args.w1_min is None else float(args.w1_min),
            "w1max_grid": str(args.w1max_grid),
            "alpha_dm": float(args.alpha_dm),
            "min_N": int(args.min_N),
            "footprint_w1_max": float(footprint_w1_max),
            "exclude_mask_fits": None if not ex_path or not ex_path.exists() else str(ex_path),
            "N_exclude_regions_used": int(n_exclude_regions_used),
            "template_set": str(args.template_set),
            "template_names": template_names,
            "max_iter": int(args.max_iter),
            "N_base": int(len(w1)),
            "N_footprint_pix": int(len(foot_pix_idx)),
        },
        "rows": [r.as_dict() for r in rows],
        "notes": [
            "Dipole amplitude reported as |b| from log-link Poisson GLM: log mu = beta0 + b·n + templates.",
            "For D<<1, |b| is the usual fractional dipole amplitude in intensity/number counts.",
            "alpha_edge is computed from cumulative counts N(<=W1_max) via a finite-difference in log-space.",
            "Poisson fits are performed on a fixed pixel footprint mask (defined at footprint_w1_max) and include zero-count pixels.",
            "Ecliptic-latitude template is evaluated at pixel centers (independent of the quasar selection).",
        ],
    }

    (outdir / "scan.json").write_text(json.dumps(out, indent=2, sort_keys=True))

    if args.make_plots:
        try:
            import matplotlib.pyplot as plt

            w1m = np.array([r.w1_max for r in rows], dtype=float)
            D = np.array([r.dipole_amp for r in rows], dtype=float)
            a = np.array([r.alpha_edge for r in rows], dtype=float)

            fig = plt.figure(figsize=(10, 3.5), dpi=200)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            ax1.plot(w1m, D, marker="o", ms=3, lw=1.0)
            ax1.set_xlabel("W1_max")
            ax1.set_ylabel("Dipole amplitude (Poisson GLM)")

            ax2.plot(w1m, a, marker="o", ms=3, lw=1.0, color="tab:orange")
            ax2.set_xlabel("W1_max")
            ax2.set_ylabel("alpha_edge = d ln N / d m")

            ax3.scatter(a, D, s=10)
            ax3.set_xlabel("alpha_edge")
            ax3.set_ylabel("Dipole amplitude")

            fig.tight_layout()
            fig.savefig(outdir / "scan_summary.png")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            (outdir / "plot_error.txt").write_text(f"Plotting failed: {e}\n")

    print(f"Wrote: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

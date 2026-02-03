#!/usr/bin/env python3
"""
Secrest critique verification for CatWISE dipole analyses.

This script creates *direct, testable* evidence for two referee-level points:

  (1) Naive "linear"/vector-sum dipole estimators are biased on partial sky.
  (2) WISE scan-related ecliptic systematics must be modeled; otherwise dipoles
      can shift with flux/magnitude cuts even under a null (no true dipole).

We do this via two small experiments:

  A) MOCKS (fast; no external data):
     Simulate Poisson HEALPix counts with a known injected dipole and an optional
     ecliptic-latitude trend under a Galactic-latitude mask, then compare:
       - vector-sum estimator (linear)
       - WLS count fit (Gaussian approximation)
       - Poisson GLM MLE (log-link), with/without an ecliptic template

  B) REAL DATA (optional; requires CatWISE/Secrest FITS on disk):
     Fit the same methods to the CatWISE catalog after baseline cuts, to show
     estimator/model dependence directly on the data used in the paper.

Outputs go to a gitignored output directory (default under outputs/).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def lb_to_unitvec(l_deg: np.ndarray, b_deg: np.ndarray) -> np.ndarray:
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


def axis_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    """Axis angle in [0,90] deg (sign-invariant)."""
    a = np.asarray(v1, dtype=float).reshape(3)
    b = np.asarray(v2, dtype=float).reshape(3)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    dot = abs(float(np.dot(a, b)) / (na * nb))
    dot = max(-1.0, min(1.0, dot))
    return float(np.degrees(math.acos(dot)))


def zscore(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x[valid])) if np.any(valid) else 0.0
    s = float(np.std(x[valid])) if np.any(valid) else 1.0
    if s == 0.0:
        s = 1.0
    out = (x - m) / s
    out[~valid] = 0.0
    return out


@dataclass(frozen=True)
class DipoleEstimate:
    method: str
    amp: float
    l_deg: float
    b_deg: float
    vec: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "amp": float(self.amp),
            "l_deg": float(self.l_deg),
            "b_deg": float(self.b_deg),
            "vec": [float(x) for x in np.asarray(self.vec, dtype=float).reshape(3)],
        }


def dipole_vector_sum_from_counts(pix_unit: np.ndarray, counts: np.ndarray) -> DipoleEstimate:
    counts = np.asarray(counts, dtype=float)
    S = np.sum(pix_unit * counts[:, None], axis=0)
    N = float(np.sum(counts))
    amp = 3.0 * float(np.linalg.norm(S)) / N if N > 0 else float("nan")
    l, b = vec_to_lb(S)
    return DipoleEstimate(method="vector_sum", amp=amp, l_deg=l, b_deg=b, vec=S)


def solve_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    sw = np.sqrt(np.clip(w, 0.0, np.inf))
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return np.asarray(beta, dtype=float)


def dipole_wls_from_counts(pix_unit: np.ndarray, counts: np.ndarray, *, templates: np.ndarray | None = None) -> DipoleEstimate:
    """
    Count-level WLS model: N = A + B·n + templates.
    Report D = |B|/A.
    """
    y = np.asarray(counts, dtype=float)
    cols = [np.ones_like(y), pix_unit[:, 0], pix_unit[:, 1], pix_unit[:, 2]]
    if templates is not None:
        cols.extend([templates[:, j] for j in range(templates.shape[1])])
    X = np.column_stack(cols)
    # Poisson approx Var(N)~N -> WLS weights ~ 1/max(N,1)
    w = 1.0 / np.clip(y, 1.0, np.inf)
    beta = solve_wls(X, y, w)
    A = float(beta[0])
    B = np.asarray(beta[1:4], dtype=float)
    amp = float(np.linalg.norm(B) / A) if A != 0.0 else float("nan")
    l, b = vec_to_lb(B)
    return DipoleEstimate(method="wls_counts", amp=amp, l_deg=l, b_deg=b, vec=B)


def poisson_glm_fit(beta0: np.ndarray, X: np.ndarray, y: np.ndarray, offset: np.ndarray | None, max_iter: int) -> np.ndarray:
    """Poisson GLM (log link) MLE via L-BFGS."""
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    off = np.zeros_like(y) if offset is None else np.asarray(offset, dtype=float)

    def fun_and_grad(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = off + X @ beta
        eta = np.clip(eta, -25.0, 25.0)  # avoid overflow
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
        np.asarray(beta0, dtype=float),
        jac=g,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-10},
    )
    return np.asarray(res.x, dtype=float)


def dipole_poisson_glm_from_counts(
    pix_unit: np.ndarray,
    counts: np.ndarray,
    *,
    templates: np.ndarray | None = None,
    max_iter: int = 200,
) -> DipoleEstimate:
    """
    Poisson GLM (log link): log mu = beta0 + b·n + templates.
    For small dipoles, D ≈ |b|.
    """
    y = np.asarray(counts, dtype=float)
    cols = [np.ones_like(y), pix_unit[:, 0], pix_unit[:, 1], pix_unit[:, 2]]
    if templates is not None:
        cols.extend([templates[:, j] for j in range(templates.shape[1])])
    X = np.column_stack(cols)

    mu0 = float(np.clip(np.mean(y), 1.0, np.inf))
    beta0 = np.zeros(X.shape[1], dtype=float)
    beta0[0] = math.log(mu0)
    beta = poisson_glm_fit(beta0, X, y, offset=None, max_iter=max_iter)
    bvec = np.asarray(beta[1:4], dtype=float)
    amp = float(np.linalg.norm(bvec))
    l, b = vec_to_lb(bvec)
    return DipoleEstimate(method="poisson_glm", amp=amp, l_deg=l, b_deg=b, vec=bvec)


def make_mock_maps(
    *,
    nside: int,
    b_cut: float,
    dipole_l_deg: float,
    dipole_b_deg: float,
    dipole_amp: float,
    eclip_gamma: float,
    mu0: float,
    seed: int,
    n_sims: int,
    include_eclip_template_in_fit: bool,
    outdir: Path,
) -> dict[str, Any]:
    import healpy as hp
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    rng = np.random.default_rng(seed)

    npix = hp.nside2npix(int(nside))
    theta, phi = hp.pix2ang(int(nside), np.arange(npix), nest=False)
    l_deg = np.degrees(phi)
    b_deg = 90.0 - np.degrees(theta)
    pix_unit = lb_to_unitvec(l_deg, b_deg)

    mask = np.abs(b_deg) > float(b_cut)
    valid = mask.copy()

    # Ecliptic latitude per pixel (precompute).
    sc = SkyCoord(l=l_deg * u.deg, b=b_deg * u.deg, frame="galactic")
    elat = sc.barycentrictrueecliptic.lat.deg.astype(float)
    t_ecl = np.abs(np.sin(np.deg2rad(elat)))
    t_ecl = zscore(t_ecl, valid)

    # True dipole vector b (for log-link model).
    a_true = lb_to_unitvec(np.array([dipole_l_deg]), np.array([dipole_b_deg]))[0]
    b_true = float(dipole_amp) * a_true

    # Intensity model used for mocks.
    # log lambda = log(mu0) + b_true·n + eclip_gamma * t_ecl
    log_mu0 = math.log(float(mu0))
    eta_base = log_mu0 + (pix_unit @ b_true) + float(eclip_gamma) * t_ecl
    eta_base[~valid] = -np.inf
    lam_base = np.exp(np.clip(eta_base, -25.0, 25.0))

    # Containers for repeated simulations.
    rows: list[dict[str, Any]] = []
    angle_rows: dict[str, list[float]] = {"vector_sum": [], "wls_counts": [], "poisson_glm": [], "poisson_glm+eclip": []}
    amp_rows: dict[str, list[float]] = {"vector_sum": [], "wls_counts": [], "poisson_glm": [], "poisson_glm+eclip": []}

    templates_fit = np.column_stack([t_ecl[valid]]) if include_eclip_template_in_fit else None

    for i in range(int(n_sims)):
        counts = rng.poisson(lam_base)
        y = counts[valid].astype(float)
        u = pix_unit[valid]

        est_vec = dipole_vector_sum_from_counts(u, y)
        est_wls = dipole_wls_from_counts(u, y, templates=None)
        est_glm = dipole_poisson_glm_from_counts(u, y, templates=None)
        est_glm_ecl = dipole_poisson_glm_from_counts(u, y, templates=templates_fit) if templates_fit is not None else est_glm

        for est in [est_vec, est_wls, est_glm]:
            amp_rows[est.method].append(est.amp)
            angle_rows[est.method].append(axis_angle_deg(est.vec, a_true))
        amp_rows["poisson_glm+eclip"].append(est_glm_ecl.amp)
        angle_rows["poisson_glm+eclip"].append(axis_angle_deg(est_glm_ecl.vec, a_true))

        rows.append(
            {
                "i": int(i),
                "D_true": float(dipole_amp),
                "vector_sum": est_vec.as_dict(),
                "wls_counts": est_wls.as_dict(),
                "poisson_glm": est_glm.as_dict(),
                "poisson_glm+eclip": est_glm_ecl.as_dict(),
            }
        )

    def summarize(method: str) -> dict[str, Any]:
        amps = np.asarray(amp_rows[method], dtype=float)
        angs = np.asarray(angle_rows[method], dtype=float)
        return {
            "amp_mean": float(np.mean(amps)),
            "amp_std": float(np.std(amps)),
            "amp_bias": float(np.mean(amps) - float(dipole_amp)),
            "axis_angle_deg_mean": float(np.nanmean(angs)),
            "axis_angle_deg_p50": float(np.nanmedian(angs)),
            "axis_angle_deg_p90": float(np.nanpercentile(angs, 90.0)),
        }

    summary = {
        "config": {
            "nside": int(nside),
            "b_cut": float(b_cut),
            "mu0": float(mu0),
            "n_sims": int(n_sims),
            "seed": int(seed),
            "dipole_amp_true": float(dipole_amp),
            "dipole_axis_l_deg": float(dipole_l_deg),
            "dipole_axis_b_deg": float(dipole_b_deg),
            "eclip_gamma_true": float(eclip_gamma),
            "fit_includes_eclip_template": bool(include_eclip_template_in_fit),
        },
        "summary": {m: summarize(m) for m in ["vector_sum", "wls_counts", "poisson_glm", "poisson_glm+eclip"]},
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "mock_draws.json").write_text(json.dumps(rows[: min(50, len(rows))], indent=2, sort_keys=True))
    (outdir / "mock_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    # Quick plot: amplitude distributions.
    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4), dpi=200)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        bins = 40
        for key, label in [
            ("vector_sum", "vector-sum"),
            ("wls_counts", "WLS"),
            ("poisson_glm", "Poisson GLM"),
            ("poisson_glm+eclip", "Poisson GLM + ecl"),
        ]:
            ax1.hist(np.asarray(amp_rows[key], float), bins=bins, histtype="step", linewidth=1.2, label=label, density=True)
            ax2.hist(np.asarray(angle_rows[key], float), bins=bins, histtype="step", linewidth=1.2, label=label, density=True)

        ax1.axvline(float(dipole_amp), color="k", linestyle="--", linewidth=1.0)
        ax1.set_xlabel("Estimated dipole amplitude")
        ax1.set_ylabel("density")
        ax1.set_title("Amplitude")
        ax1.legend(fontsize=7)

        ax2.set_xlabel("Axis angle error [deg]")
        ax2.set_ylabel("density")
        ax2.set_title("Direction (axis) error")
        ax2.legend(fontsize=7)

        fig.suptitle(f"Mock estimator comparison (mu0={mu0}, eclip_gamma={eclip_gamma})")
        fig.tight_layout()
        fig.savefig(outdir / "mock_estimator_bias.png")
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        (outdir / "mock_plot_error.txt").write_text(f"Plotting failed: {e}\n")

    return summary


def run_real_data_methods(
    *,
    catalog_fits: str,
    outdir: Path,
    nside: int,
    b_cut: float,
    w1cov_min: float,
    w1_min: float | None,
    w1_max: float,
) -> dict[str, Any]:
    import healpy as hp
    from astropy.table import Table

    tbl = Table.read(catalog_fits)
    w1 = np.asarray(tbl["w1"], float)
    b = np.asarray(tbl["b"], float)
    l = np.asarray(tbl["l"], float)
    w1cov = np.asarray(tbl["w1cov"], float) if "w1cov" in tbl.colnames else None
    elat = np.asarray(tbl["elat"], float) if "elat" in tbl.colnames else None

    mask = np.ones(len(tbl), dtype=bool)
    if w1cov is not None:
        mask &= w1cov >= float(w1cov_min)
    mask &= np.abs(b) > float(b_cut)
    if w1_min is not None:
        mask &= w1 >= float(w1_min)
    mask &= w1 <= float(w1_max)

    l = l[mask]
    b = b[mask]
    if elat is None:
        raise RuntimeError("catalog missing elat column; expected Secrest/CatWISE extracted FITS.")
    elat = elat[mask]

    # Build counts per pixel.
    npix = hp.nside2npix(int(nside))
    theta = np.deg2rad(90.0 - b)
    phi = np.deg2rad(l % 360.0)
    pix = hp.ang2pix(int(nside), theta, phi, nest=False)
    Np = np.bincount(pix, minlength=npix).astype(float)
    valid = Np > 0

    # Pixel unit vectors: use mean object direction per pixel for best match to vector-sum estimator.
    nvec = lb_to_unitvec(l, b)
    sx = np.bincount(pix, weights=nvec[:, 0], minlength=npix)
    sy = np.bincount(pix, weights=nvec[:, 1], minlength=npix)
    sz = np.bincount(pix, weights=nvec[:, 2], minlength=npix)
    pix_unit = np.column_stack([sx[valid] / Np[valid], sy[valid] / Np[valid], sz[valid] / Np[valid]])

    # Ecliptic template (pixel-mean |sin(elat)|).
    tel = np.abs(np.sin(np.deg2rad(elat)))
    s_tel = np.bincount(pix, weights=tel, minlength=npix)
    t_ecl = np.zeros(npix, dtype=float)
    t_ecl[valid] = s_tel[valid] / Np[valid]
    t_ecl = zscore(t_ecl, valid)
    templates = np.column_stack([t_ecl[valid]])

    y = Np[valid]

    # Estimates
    est_vec = dipole_vector_sum_from_counts(pix_unit, y)
    est_wls = dipole_wls_from_counts(pix_unit, y, templates=None)
    est_wls_ecl = dipole_wls_from_counts(pix_unit, y, templates=templates)
    est_glm = dipole_poisson_glm_from_counts(pix_unit, y, templates=None)
    est_glm_ecl = dipole_poisson_glm_from_counts(pix_unit, y, templates=templates)

    out = {
        "config": {
            "catalog": str(catalog_fits),
            "nside": int(nside),
            "b_cut": float(b_cut),
            "w1cov_min": float(w1cov_min),
            "w1_min": None if w1_min is None else float(w1_min),
            "w1_max": float(w1_max),
            "N_sources": int(len(l)),
            "N_pix_nonzero": int(np.sum(valid)),
        },
        "estimates": {
            "vector_sum": est_vec.as_dict(),
            "wls_counts": est_wls.as_dict(),
            "wls_counts+eclip": est_wls_ecl.as_dict(),
            "poisson_glm": est_glm.as_dict(),
            "poisson_glm+eclip": est_glm_ecl.as_dict(),
        },
        "notes": [
            "poisson_glm amplitude uses D≈|b| from log-link model; for D<<1 this matches fractional dipole amplitude.",
            "Template uses pixel-mean |sin(elat)| standardized on the observed pixels.",
        ],
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "real_methods_comparison.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None, help="Output directory (default: outputs/quasar_secrest_verify_<UTC>).")

    ap.add_argument("--nside", type=int, default=32, help="HEALPix NSIDE for mocks and real-data binning.")
    ap.add_argument("--b-cut", type=float, default=30.0, help="Galactic latitude cut (deg).")

    ap.add_argument("--dipole-l", type=float, default=236.0)
    ap.add_argument("--dipole-b", type=float, default=28.8)
    ap.add_argument("--dipole-amp", type=float, default=0.02)
    ap.add_argument("--eclip-gamma", type=float, default=0.0, help="Mock ecliptic trend strength (log-link coefficient).")
    ap.add_argument("--mu0", type=float, default=80.0, help="Mock mean counts per *unmasked* pixel (approx).")
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--fit-eclip-template", action="store_true", help="Include ecliptic template in GLM/WLS fits in mocks.")

    ap.add_argument("--run-real", action="store_true", help="Also run method comparison on the CatWISE FITS.")
    ap.add_argument(
        "--catalog",
        default="data/external/zenodo_6784602/secrest_extracted/secrest+22_accepted/wise/reference/catwise_agns.fits",
        help="CatWISE/Secrest FITS path (for --run-real).",
    )
    ap.add_argument("--w1cov-min", type=float, default=80.0)
    ap.add_argument("--w1-min", type=float, default=None)
    ap.add_argument("--w1-max", type=float, default=16.4)

    args = ap.parse_args()

    outdir = Path(args.outdir or f"outputs/quasar_secrest_verify_{utc_tag()}")
    outdir.mkdir(parents=True, exist_ok=True)

    mock_dir = outdir / "mock"
    real_dir = outdir / "real"

    mock = make_mock_maps(
        nside=int(args.nside),
        b_cut=float(args.b_cut),
        dipole_l_deg=float(args.dipole_l),
        dipole_b_deg=float(args.dipole_b),
        dipole_amp=float(args.dipole_amp),
        eclip_gamma=float(args.eclip_gamma),
        mu0=float(args.mu0),
        seed=int(args.seed),
        n_sims=int(args.n_sims),
        include_eclip_template_in_fit=bool(args.fit_eclip_template),
        outdir=mock_dir,
    )

    out: dict[str, Any] = {"mock": mock}

    if args.run_real:
        out["real"] = run_real_data_methods(
            catalog_fits=str(args.catalog),
            outdir=real_dir,
            nside=int(args.nside),
            b_cut=float(args.b_cut),
            w1cov_min=float(args.w1cov_min),
            w1_min=args.w1_min,
            w1_max=float(args.w1_max),
        )

    (outdir / "summary.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"Wrote: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


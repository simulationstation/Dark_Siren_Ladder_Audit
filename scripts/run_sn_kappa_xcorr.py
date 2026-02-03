#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty
from entropy_horizon_recon.optical_bias.estimators import (
    cross_cl_pseudo,
    evaluate_kappa_at_sn,
    residual_map_from_samples,
    weighted_linear_fit,
    weighted_corr,
)
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.ingest_sn import load_sn_dataset
from entropy_horizon_recon.optical_bias.injection import inject_mu
from entropy_horizon_recon.optical_bias.null_tests import (
    parallel_nulls,
    permutation_pvalue_two_sided,
    rotate_radec_random,
    shuffle_positions,
)
from entropy_horizon_recon.optical_bias.reporting import write_json, write_report
from entropy_horizon_recon.optical_bias.residuals import detrend_residuals, residuals_mu


def _bin_cl(ell: np.ndarray, cl: np.ndarray, ell_min: int, ell_max: int, ell_bin: int) -> tuple[np.ndarray, np.ndarray]:
    sel = (ell >= ell_min) & (ell <= ell_max)
    ell = ell[sel]
    cl = cl[sel]
    if ell.size == 0:
        return np.array([]), np.array([])
    bins = np.arange(ell_min, ell_max + ell_bin, ell_bin)
    centers = 0.5 * (bins[:-1] + bins[1:])
    cl_b = np.zeros_like(centers)
    for i in range(len(centers)):
        in_bin = (ell >= bins[i]) & (ell < bins[i + 1])
        if not np.any(in_bin):
            cl_b[i] = np.nan
        else:
            vals = cl[in_bin]
            vals = vals[np.isfinite(vals)]
            cl_b[i] = np.nan if vals.size == 0 else float(np.mean(vals))
    return centers, cl_b


def _stack_bins(kappa: np.ndarray, residuals: np.ndarray, weights: np.ndarray, n_bins: int) -> dict:
    q = np.quantile(kappa, np.linspace(0, 1, n_bins + 1))
    means = []
    errs = []
    centers = []
    for i in range(n_bins):
        m = (kappa >= q[i]) & (kappa < q[i + 1])
        if not np.any(m):
            means.append(np.nan)
            errs.append(np.nan)
            centers.append(0.5 * (q[i] + q[i + 1]))
            continue
        w = weights[m]
        r = residuals[m]
        wsum = np.sum(w)
        means.append(float(np.sum(w * r) / wsum))
        # Standard error of the weighted mean using an effective sample size.
        var = float(np.sum(w * (r - means[-1]) ** 2) / wsum)
        n_eff = float((wsum**2) / np.sum(w**2)) if np.sum(w**2) > 0 else float(w.size)
        errs.append(float(np.sqrt(var / max(n_eff, 1.0))))
        centers.append(float(0.5 * (q[i] + q[i + 1])))
    return {"kappa_centers": centers, "mean_residual": means, "err_residual": errs}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--ell-min", type=int, default=8)
    parser.add_argument("--ell-max", type=int, default=512)
    parser.add_argument("--ell-bin", type=int, default=16)
    parser.add_argument("--n-null", type=int, default=200)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--detrend", type=str, default="none", choices=["none", "poly1", "poly2"])
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--inj-amplitude", type=float, default=0.0)
    parser.add_argument("--fid-H0", type=float, default=70.0)
    parser.add_argument("--fid-Om", type=float, default=0.3)
    parser.add_argument("--frame", type=str, default="galactic", choices=["icrs", "galactic"])
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--sn-local-path", type=Path, default=None)
    parser.add_argument("--sn-max", type=int, default=0, help="Limit to first N SNe (0=all).")
    parser.add_argument("--null-jobs", type=int, default=8)
    parser.add_argument("--compute-cl-null", action="store_true", help="Compute null p-values for cl_amp (slow).")
    parser.add_argument("--huber-delta", type=float, default=1.5, help="Huber loss scale for robust regression.")
    args = parser.parse_args()

    out_dir = args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    prov = {
        "git_sha": git_head_sha(repo_root=repo_root),
        "git_dirty": git_is_dirty(repo_root=repo_root),
        "cmd": command_str(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    try:
        from importlib.metadata import version

        prov["versions"] = {
            "numpy": version("numpy"),
            "scipy": version("scipy"),
            "pandas": version("pandas"),
            "healpy": version("healpy"),
            "astropy": version("astropy"),
            "joblib": version("joblib"),
        }
    except Exception:
        pass
    sn = load_sn_dataset(paths=paths, allow_unverified=bool(args.allow_unverified), local_path=args.sn_local_path)
    if args.sn_max and args.sn_max > 0:
        n = int(args.sn_max)
        sn = sn.__class__(
            z=sn.z[:n],
            mu=sn.mu[:n],
            sigma_mu=sn.sigma_mu[:n],
            ra_deg=sn.ra_deg[:n],
            dec_deg=sn.dec_deg[:n],
            meta=dict(sn.meta),
        )
    kappa = load_planck_kappa(paths=paths, nside_out=int(args.nside), allow_unverified=bool(args.allow_unverified))

    ra = sn.ra_deg
    dec = sn.dec_deg
    z = sn.z
    mu = sn.mu
    sigma = sn.sigma_mu

    residuals = residuals_mu(mu, z, H0=float(args.fid_H0), omega_m0=float(args.fid_Om))
    residuals = detrend_residuals(z, residuals, mode=str(args.detrend))

    kappa_sn = evaluate_kappa_at_sn(kappa.kappa_map, ra, dec, nside=kappa.nside, frame=args.frame)
    weights = 1.0 / np.maximum(sigma, 1e-6) ** 2

    mask = np.ones_like(kappa.kappa_map)
    if kappa.mask is not None:
        mask = np.asarray(kappa.mask, dtype=float)

    mask_frac = None
    if kappa.mask is not None:
        from entropy_horizon_recon.optical_bias.maps import radec_to_healpix

        pix = radec_to_healpix(ra, dec, nside=kappa.nside, frame=args.frame)
        good = mask[pix] > 0
        mask_frac = float(np.mean(good)) if good.size > 0 else 0.0
        residuals = residuals[good]
        kappa_sn = kappa_sn[good]
        weights = weights[good]
        z_used = z[good]
        ra = ra[good]
        dec = dec[good]
        mu = mu[good]
    else:
        z_used = z

    if args.inj_amplitude and args.inj_amplitude != 0.0:
        mu_inj = inject_mu(mu, kappa_sn * float(args.inj_amplitude))
        residuals = residuals_mu(mu_inj, z_used, H0=float(args.fid_H0), omega_m0=float(args.fid_Om))
        residuals = detrend_residuals(z_used, residuals, mode=str(args.detrend))

    mean_kappa = float(np.sum(weights * kappa_sn) / np.sum(weights))
    reg = weighted_linear_fit(kappa_sn, residuals, weights)
    corr = weighted_corr(kappa_sn, residuals, weights)

    # Robust regression (Huber loss) as a sensitivity check.
    huber = None
    try:
        from scipy.optimize import least_squares

        x = np.asarray(kappa_sn, dtype=float)
        y = np.asarray(residuals, dtype=float)
        sw = np.sqrt(np.asarray(weights, dtype=float))

        def fun(p: np.ndarray) -> np.ndarray:
            a, b = float(p[0]), float(p[1])
            return sw * (y - (a + b * x))

        res = least_squares(fun, x0=np.array([reg["a"], reg["b"]], dtype=float), loss="huber", f_scale=float(args.huber_delta))
        huber = {"a": float(res.x[0]), "b": float(res.x[1]), "success": bool(res.success)}
    except Exception:
        huber = None

    res_map, hit_map = residual_map_from_samples(ra, dec, residuals, weights, nside=kappa.nside, frame=args.frame)
    # Only include pixels where we have SN residuals *and* are in the Planck mask.
    comb_mask = np.asarray(mask > 0, dtype=float) * np.asarray(hit_map > 0, dtype=float)
    ell, cl_rk = cross_cl_pseudo(res_map, kappa.kappa_map, comb_mask, lmax=int(args.ell_max))
    ell_b, cl_b = _bin_cl(ell, cl_rk, int(args.ell_min), int(args.ell_max), int(args.ell_bin))
    cl_amp = float(np.nanmean(cl_b)) if np.any(np.isfinite(cl_b)) else np.nan

    stack = _stack_bins(kappa_sn, residuals, weights, int(args.n_bins))

    seeds = list(range(int(args.seed), int(args.seed) + int(args.n_null)))

    def _null_rot_b(seed: int) -> float:
        ra_r, dec_r = rotate_radec_random(ra, dec, seed=seed)
        k_sn = evaluate_kappa_at_sn(kappa.kappa_map, ra_r, dec_r, nside=kappa.nside, frame=args.frame)
        if kappa.mask is not None:
            from entropy_horizon_recon.optical_bias.maps import radec_to_healpix

            pix_r = radec_to_healpix(ra_r, dec_r, nside=kappa.nside, frame=args.frame)
            good_r = mask[pix_r] > 0
            if np.sum(good_r) < 50:
                return np.nan
            k_sn = k_sn[good_r]
            r_sn = residuals[good_r]
            w_sn = weights[good_r]
        else:
            r_sn = residuals
            w_sn = weights
        try:
            return float(weighted_linear_fit(k_sn, r_sn, w_sn)["b"])
        except Exception:
            return np.nan

    def _null_shuffle_b(seed: int) -> float:
        ra_s, dec_s = shuffle_positions(ra, dec, seed=seed)
        k_sn = evaluate_kappa_at_sn(kappa.kappa_map, ra_s, dec_s, nside=kappa.nside, frame=args.frame)
        try:
            return float(weighted_linear_fit(k_sn, residuals, weights)["b"])
        except Exception:
            return np.nan

    null_b_rot = parallel_nulls(_null_rot_b, seeds, n_jobs=int(args.null_jobs))
    null_b_shuffle = parallel_nulls(_null_shuffle_b, seeds, n_jobs=int(args.null_jobs))

    p_rot_b = permutation_pvalue_two_sided(null_b_rot, reg["b"])
    p_shuffle_b = permutation_pvalue_two_sided(null_b_shuffle, reg["b"])

    null_cl_rot = None
    null_cl_shuffle = None
    p_rot_cl = None
    p_shuffle_cl = None
    if bool(args.compute_cl_null):
        def _cl_amp_for(ra0: np.ndarray, dec0: np.ndarray, residuals0: np.ndarray) -> float:
            res_map0, hit_map0 = residual_map_from_samples(
                ra0, dec0, residuals0, weights, nside=kappa.nside, frame=args.frame
            )
            comb_mask0 = np.asarray(mask > 0, dtype=float) * np.asarray(hit_map0 > 0, dtype=float)
            ell0, cl0 = cross_cl_pseudo(res_map0, kappa.kappa_map, comb_mask0, lmax=int(args.ell_max))
            _ell_b0, cl_b0 = _bin_cl(ell0, cl0, int(args.ell_min), int(args.ell_max), int(args.ell_bin))
            if not np.any(np.isfinite(cl_b0)):
                return np.nan
            return float(np.nanmean(cl_b0))

        def _null_rot_cl(seed: int) -> float:
            ra_r, dec_r = rotate_radec_random(ra, dec, seed=seed)
            if kappa.mask is not None:
                from entropy_horizon_recon.optical_bias.maps import radec_to_healpix

                pix_r = radec_to_healpix(ra_r, dec_r, nside=kappa.nside, frame=args.frame)
                good_r = mask[pix_r] > 0
                if np.sum(good_r) < 50:
                    return np.nan
                return _cl_amp_for(ra_r[good_r], dec_r[good_r], residuals[good_r])
            return _cl_amp_for(ra_r, dec_r, residuals)

        def _null_shuffle_cl(seed: int) -> float:
            ra_s, dec_s = shuffle_positions(ra, dec, seed=seed)
            return _cl_amp_for(ra_s, dec_s, residuals)

        null_cl_rot = parallel_nulls(_null_rot_cl, seeds, n_jobs=int(args.null_jobs))
        null_cl_shuffle = parallel_nulls(_null_shuffle_cl, seeds, n_jobs=int(args.null_jobs))
        p_rot_cl = permutation_pvalue_two_sided(null_cl_rot, cl_amp)
        p_shuffle_cl = permutation_pvalue_two_sided(null_cl_shuffle, cl_amp)

    summary = {
        "provenance": prov,
        "datasets": {
            "sn": dict(getattr(sn, "meta", {})),
            "planck_lensing": dict(getattr(kappa, "meta", {})),
        },
        "n_sn": int(residuals.size),
        "mask_pass_fraction": mask_frac,
        "kappa_stats": {
            "mean": float(np.mean(kappa_sn)),
            "std": float(np.std(kappa_sn)),
            "p16": float(np.percentile(kappa_sn, 16.0)) if kappa_sn.size > 0 else np.nan,
            "p50": float(np.percentile(kappa_sn, 50.0)) if kappa_sn.size > 0 else np.nan,
            "p84": float(np.percentile(kappa_sn, 84.0)) if kappa_sn.size > 0 else np.nan,
        },
        "mean_kappa": mean_kappa,
        "regression": reg,
        "corr_w": corr,
        "huber_regression": huber,
        "stacking": stack,
        "cl_binned": {"ell": ell_b.tolist(), "cl": cl_b.tolist(), "cl_amp": cl_amp},
        "null_pvalues": {"rotation_b": p_rot_b, "shuffle_b": p_shuffle_b, "rotation_cl": p_rot_cl, "shuffle_cl": p_shuffle_cl},
        "config": {
            "nside": int(args.nside),
            "ell_min": int(args.ell_min),
            "ell_max": int(args.ell_max),
            "ell_bin": int(args.ell_bin),
            "detrend": str(args.detrend),
            "inj_amplitude": float(args.inj_amplitude),
        },
    }
    write_json(out_dir / "tables" / "summary.json", summary)
    write_json(out_dir / "tables" / "null_tests.json", {
        "null_b_rot": null_b_rot.tolist(),
        "null_b_shuffle": null_b_shuffle.tolist(),
        "pvalue_rotation_b": p_rot_b,
        "pvalue_shuffle_b": p_shuffle_b,
        "null_cl_rot": null_cl_rot.tolist() if null_cl_rot is not None else None,
        "null_cl_shuffle": null_cl_shuffle.tolist() if null_cl_shuffle is not None else None,
        "pvalue_rotation_cl": p_rot_cl,
        "pvalue_shuffle_cl": p_shuffle_cl,
    })

    # plots
    try:
        import matplotlib.pyplot as plt

        if ell_b.size > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(ell_b, cl_b, marker="o")
            plt.xlabel(r"$\\ell$")
            plt.ylabel(r"$C_\\ell^{r\\kappa}$")
            plt.tight_layout()
            plt.savefig(out_dir / "figures" / "cl_rk_binned.png")
            plt.close()

        plt.figure(figsize=(6, 4))
        plt.errorbar(stack["kappa_centers"], stack["mean_residual"], yerr=stack["err_residual"], fmt="o")
        plt.xlabel(r"$\\kappa$ bin center")
        plt.ylabel("Mean residual")
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "stacking.png")
        plt.close()
    except Exception:
        pass

    report = [
        "# SN–kappa cross-correlation",
        "",
        f"N_SN used: {summary['n_sn']}",
        f"Mean kappa (weighted): {mean_kappa:.4e}",
        f"Weighted corr(kappa,resid): {corr:.4f}",
        f"Regression slope b: {reg['b']:.4e} ± {reg['b_err']:.4e}  (z={reg.get('z', float('nan')):.2f}, p≈{reg.get('p_two_sided_norm', float('nan')):.3f})",
        f"Huber slope b (diagnostic): {huber['b']:.4e}" if huber is not None else "Huber slope b (diagnostic): n/a",
        f"Null p-values (two-sided) for slope b: rotation={p_rot_b:.3f}, shuffle={p_shuffle_b:.3f}",
        f"cl_amp: {cl_amp:.3e}",
        f"Null p-values (two-sided) for cl_amp: rotation={p_rot_cl:.3f}, shuffle={p_shuffle_cl:.3f}"
        if p_rot_cl is not None
        else "cl_amp nulls: not computed",
    ]
    write_report(out_dir / "report.md", "\n".join(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

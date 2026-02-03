from __future__ import annotations

import os

# Avoid nested parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.optical_bias.estimators import (
    cross_cl_pseudo,
    evaluate_kappa_at_sn,
    residual_map_from_samples,
    weighted_linear_fit,
)
from entropy_horizon_recon.optical_bias.maps import radec_to_healpix
from entropy_horizon_recon.optical_bias.injection import (
    estimate_delta_h0_over_h0,
    inject_mu,
)
from entropy_horizon_recon.optical_bias.ingest_galaxy_shear_kids1000 import load_kids1000
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.ingest_sn import load_sn_dataset
from entropy_horizon_recon.optical_bias.null_tests import parallel_nulls, permutation_pvalue, rotate_map_random
from entropy_horizon_recon.optical_bias.reporting import write_json, write_report
from entropy_horizon_recon.optical_bias.shear import (
    build_shear_maps,
    debias_shape_noise_random,
    effective_gamma2_at_sn,
    gamma2_map,
    zbin_weights_from_sn,
)
from entropy_horizon_recon.optical_bias.weights import h0_estimator_weights
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty


def fiducial_mu(z: np.ndarray, *, H0: float, Om0: float) -> np.ndarray:
    const = PhysicalConstants()
    z = np.asarray(z, dtype=float)
    z_grid = np.linspace(0.0, float(np.max(z)) + 1e-6, 800)
    H = H0 * np.sqrt(Om0 * (1.0 + z_grid) ** 3 + (1.0 - Om0))
    invH = 1.0 / H
    Dc = np.empty_like(z_grid)
    Dc[0] = 0.0
    dz = np.diff(z_grid)
    Dc[1:] = const.c_km_s * np.cumsum(0.5 * dz * (invH[:-1] + invH[1:]))
    Dc_i = np.interp(z, z_grid, Dc)
    Dl = (1.0 + z) * Dc_i
    mu = 5.0 * np.log10(Dl) + 25.0
    return mu


def main() -> int:
    parser = argparse.ArgumentParser(description="Optical-bias real-data pipeline (Track A+B).")
    parser.add_argument("--out", type=Path, default=Path("outputs/optical_bias_main"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--h0-fid", type=float, default=70.0)
    parser.add_argument("--om0-fid", type=float, default=0.3)
    parser.add_argument("--z-min", type=float, default=0.023)
    parser.add_argument("--z-max", type=float, default=0.15)
    parser.add_argument("--nulls", type=int, default=500)
    parser.add_argument("--null-procs", type=int, default=10)
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--sn-path", type=Path, default=None)
    parser.add_argument("--skip-track-b", action="store_true")
    parser.add_argument("--shear-path", type=Path, default=None, help="Local shear catalog (KiDS-1000 CSV).")
    parser.add_argument("--shear-zbins", type=int, default=5)
    parser.add_argument("--shear-rotations", type=int, default=20)
    parser.add_argument("--shear-null-rotations", type=int, default=200)
    parser.add_argument("--shear-use-subset", action="store_true", help="Use the KiDS subset file.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)

    out_dir = args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    sn = load_sn_dataset(paths=paths, allow_unverified=bool(args.allow_unverified), local_path=args.sn_path)
    z = sn.z
    mu_model = fiducial_mu(z, H0=float(args.h0_fid), Om0=float(args.om0_fid))
    residuals = sn.mu - mu_model
    weights = h0_estimator_weights(z, sn.sigma_mu, z_min=float(args.z_min), z_max=float(args.z_max))

    planck = load_planck_kappa(paths=paths, nside_out=int(args.nside), allow_unverified=bool(args.allow_unverified))
    kappa_sn = evaluate_kappa_at_sn(planck.kappa_map, sn.ra_deg, sn.dec_deg, nside=planck.nside)
    mean_kappa = float(np.sum(weights * kappa_sn))

    reg = weighted_linear_fit(kappa_sn, residuals, weights)

    # Cross-correlation
    res_map, hit_map = residual_map_from_samples(
        sn.ra_deg, sn.dec_deg, residuals, weights, nside=planck.nside
    )
    mask = planck.mask if planck.mask is not None else np.where(np.isfinite(res_map), 1.0, 0.0)
    ell, cl_rk = cross_cl_pseudo(res_map, planck.kappa_map, mask)

    # Null tests via rotations of kappa map
    def _null_stat(seed: int) -> float:
        k_rot = rotate_map_random(planck.kappa_map, seed=seed)
        k_sn = evaluate_kappa_at_sn(k_rot, sn.ra_deg, sn.dec_deg, nside=planck.nside)
        return float(np.sum(weights * k_sn))

    seeds = list(rng.integers(1, 2**31 - 1, size=int(args.nulls)))
    nulls = parallel_nulls(_null_stat, seeds, n_jobs=int(args.null_procs))
    pval = permutation_pvalue(nulls, mean_kappa)

    # Injection closure
    mu_inj = inject_mu(sn.mu, kappa_sn)
    dlnH0 = estimate_delta_h0_over_h0(mu_inj, mu_model, weights)

    summary = {
        "mean_kappa": mean_kappa,
        "regression": reg,
        "delta_h0_over_h0_kappa": mean_kappa,
        "null_pvalue_mean_kappa": pval,
        "injection_delta_h0_over_h0": dlnH0,
        "n_sn": int(z.size),
        "settings": {
            "nside": int(args.nside),
            "z_min": float(args.z_min),
            "z_max": float(args.z_max),
            "h0_fid": float(args.h0_fid),
            "om0_fid": float(args.om0_fid),
            "shear_use_subset": bool(args.shear_use_subset),
        },
    }

    shear_bias = None
    if not args.skip_track_b:
        catalog = None
        if args.shear_path is not None:
            catalog = load_kids1000(
                paths=paths, allow_unverified=bool(args.allow_unverified), local_path=args.shear_path
            )
        elif args.allow_unverified:
            catalog = load_kids1000(
                paths=paths,
                allow_unverified=True,
                use_subset=bool(args.shear_use_subset),
            )
        if catalog is not None:
            n_zbin = int(args.shear_zbins)
            g1, g2, wmap = build_shear_maps(
                catalog.ra_deg,
                catalog.dec_deg,
                catalog.e1,
                catalog.e2,
                catalog.weight,
                catalog.zbin,
                nside=planck.nside,
                n_zbin=n_zbin,
            )
            g2map = gamma2_map(g1, g2)
            noise = debias_shape_noise_random(
                catalog.ra_deg,
                catalog.dec_deg,
                catalog.e1,
                catalog.e2,
                catalog.weight,
                catalog.zbin,
                nside=planck.nside,
                n_zbin=n_zbin,
                n_rot=int(args.shear_rotations),
                seed=int(args.seed),
            )
            g2_debiased = g2map - noise
            zbin_edges = np.linspace(np.min(z), np.max(z), n_zbin + 1)
            zbin_w = zbin_weights_from_sn(z, zbin_edges)
            sn_pix = radec_to_healpix(sn.ra_deg, sn.dec_deg, nside=planck.nside)
            g2_sn = effective_gamma2_at_sn(sn_pix, g2_debiased, zbin_w)
            I_eff = float(np.sum(weights * g2_sn))
            shear_bias = 0.5 * I_eff

            def _shear_null(seed: int) -> float:
                rng = np.random.default_rng(seed)
                phi = rng.uniform(0, 2 * np.pi, size=len(catalog.e1))
                c = np.cos(2 * phi)
                s = np.sin(2 * phi)
                e1r = catalog.e1 * c - catalog.e2 * s
                e2r = catalog.e1 * s + catalog.e2 * c
                g1r, g2r, _ = build_shear_maps(
                    catalog.ra_deg,
                    catalog.dec_deg,
                    e1r,
                    e2r,
                    catalog.weight,
                    catalog.zbin,
                    nside=planck.nside,
                    n_zbin=n_zbin,
                )
                g2r_map = gamma2_map(g1r, g2r)
                g2r_sn = effective_gamma2_at_sn(sn_pix, g2r_map, zbin_w)
                return float(np.sum(weights * g2r_sn))

            shear_nulls = parallel_nulls(
                _shear_null,
                list(rng.integers(1, 2**31 - 1, size=int(args.shear_null_rotations))),
                n_jobs=int(args.null_procs),
            )
            shear_pval = permutation_pvalue(shear_nulls, I_eff)
            write_json(
                out_dir / "tables" / "shear_debiasing.json",
                {
                    "I_eff": I_eff,
                    "zbin_weights": zbin_w.tolist(),
                    "null_I_eff": shear_nulls.tolist(),
                    "pvalue": shear_pval,
                },
            )

    if shear_bias is not None:
        summary["delta_h0_over_h0_shear"] = float(shear_bias)
        summary["shear_null_pvalue"] = float(shear_pval)

    write_json(out_dir / "tables" / "summary.json", summary)
    write_json(out_dir / "tables" / "null_tests.json", {"null_mean_kappa": nulls.tolist(), "pvalue": pval})

    report_lines = [
        "# Optical-bias real-data report",
        "",
        f"Mean kappa (weighted): {mean_kappa:.4e}",
        f"deltaH0/H0 (kappa term): {mean_kappa:.4e}",
        f"Null-test p-value: {pval:.3f}",
        f"Injection closure deltaH0/H0: {dlnH0:.4e}",
    ]
    if shear_bias is not None:
        report_lines.append(f"Shear term deltaH0/H0: {shear_bias:.4e}")
    else:
        report_lines.append("Shear term: not run (no shear catalog provided)")
    report = "\n".join(report_lines)
    write_report(out_dir / "report.md", report)

    combined = mean_kappa + (shear_bias if shear_bias is not None else 0.0)
    status = "\n".join(
        [
            f"Measured δH0/H0 (κ): {mean_kappa:.4e}",
            f"Measured δH0/H0 (shear^2): {shear_bias if shear_bias is not None else 'N/A'}",
            f"Combined: {combined:.4e}",
            f"Null-test p-value(s): {pval:.3f}",
            f"Injection closure error: {dlnH0 - mean_kappa:.4e}",
            "Verdict: inconclusive",
            "Next most valuable dataset to add: public shear catalog with sky positions (KiDS-1000 or DES-Y3).",
        ]
    )
    write_report(out_dir / "OPTICAL_BIAS_STATUS.md", status)

    # Provenance
    prov = {
        "git": {"sha": git_head_sha(repo_root=repo_root), "dirty": git_is_dirty(repo_root=repo_root)},
        "command": command_str(),
        "python": {"version": os.sys.version},
    }
    write_json(out_dir / "tables" / "provenance.json", prov)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

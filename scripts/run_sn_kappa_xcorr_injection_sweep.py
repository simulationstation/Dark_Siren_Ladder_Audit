#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.estimators import evaluate_kappa_at_sn, weighted_linear_fit
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.ingest_sn import load_sn_dataset
from entropy_horizon_recon.optical_bias.injection import inject_mu
from entropy_horizon_recon.optical_bias.reporting import write_json, write_report
from entropy_horizon_recon.optical_bias.residuals import detrend_residuals, residuals_mu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--detrend", type=str, default="poly1", choices=["none", "poly1", "poly2"])
    parser.add_argument("--fid-H0", type=float, default=70.0)
    parser.add_argument("--fid-Om", type=float, default=0.3)
    parser.add_argument("--frame", type=str, default="galactic", choices=["icrs", "galactic"])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--allow-unverified", action="store_true")
    args = parser.parse_args()

    out_dir = args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    paths = DataPaths(repo_root=Path(__file__).resolve().parents[1])
    sn = load_sn_dataset(paths=paths, allow_unverified=bool(args.allow_unverified), local_path=None)
    planck = load_planck_kappa(paths=paths, nside_out=int(args.nside), allow_unverified=bool(args.allow_unverified))

    ra = sn.ra_deg
    dec = sn.dec_deg
    z = sn.z
    mu = sn.mu
    sigma = sn.sigma_mu

    kappa_sn = evaluate_kappa_at_sn(planck.kappa_map, ra, dec, nside=planck.nside, frame=args.frame)
    weights = 1.0 / np.maximum(sigma, 1e-6) ** 2

    if planck.mask is not None:
        from entropy_horizon_recon.optical_bias.maps import radec_to_healpix

        pix = radec_to_healpix(ra, dec, nside=planck.nside, frame=args.frame)
        good = np.asarray(planck.mask, dtype=float)[pix] > 0
        ra = ra[good]
        dec = dec[good]
        z = z[good]
        mu = mu[good]
        kappa_sn = kappa_sn[good]
        weights = weights[good]

    r0 = residuals_mu(mu, z, H0=float(args.fid_H0), omega_m0=float(args.fid_Om))
    r0 = detrend_residuals(z, r0, mode=str(args.detrend))
    reg0 = weighted_linear_fit(kappa_sn, r0, weights)
    b0 = float(reg0["b"])

    rows = []
    for A in list(args.amplitudes):
        mu_inj = inject_mu(mu, kappa_sn * float(A))
        rA = residuals_mu(mu_inj, z, H0=float(args.fid_H0), omega_m0=float(args.fid_Om))
        rA = detrend_residuals(z, rA, mode=str(args.detrend))
        regA = weighted_linear_fit(kappa_sn, rA, weights)
        bA = float(regA["b"])
        expected_shift = -(5.0 / math.log(10.0)) * float(A)
        closure_err = (bA - b0) - expected_shift
        rows.append(
            {
                "A": float(A),
                "b": bA,
                "b_err": float(regA.get("b_err", np.nan)),
                "expected_db": expected_shift,
                "measured_db": float(bA - b0),
                "closure_error": float(closure_err),
            }
        )

    write_json(out_dir / "tables" / "injection_sweep.json", {"baseline": reg0, "rows": rows})

    # plot
    try:
        import matplotlib.pyplot as plt

        A = np.array([r["A"] for r in rows], dtype=float)
        b = np.array([r["b"] for r in rows], dtype=float)
        berr = np.array([r["b_err"] for r in rows], dtype=float)
        plt.figure(figsize=(6, 4))
        plt.errorbar(A, b, yerr=berr, fmt="o", label="Recovered")
        Aline = np.linspace(0.0, float(np.max(A)), 100)
        plt.plot(Aline, b0 - (5.0 / math.log(10.0)) * Aline, "--", label="Expected")
        plt.xlabel("Injection amplitude A")
        plt.ylabel("Regression slope b")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "figures" / "injection_sweep.png")
        plt.close()
    except Exception:
        pass

    lines = [
        "# Injection sweep (SN–kappa)",
        "",
        f"N_SN used: {int(len(z))}",
        f"Baseline slope b0: {b0:.4e}",
        "",
        "| A | measured Δb | expected Δb | closure error |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['A']:.2f} | {r['measured_db']:.4e} | {r['expected_db']:.4e} | {r['closure_error']:.4e} |"
        )
    write_report(out_dir / "report.md", "\n".join(lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

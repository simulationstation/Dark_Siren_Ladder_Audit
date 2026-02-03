from __future__ import annotations

import os

# Avoid nested parallelism.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.optical_bias.injection import estimate_delta_h0_over_h0
from entropy_horizon_recon.optical_bias.reporting import write_json, write_report
from entropy_horizon_recon.optical_bias.weights import h0_estimator_weights


def fiducial_mu(z: np.ndarray, *, H0: float, Om0: float) -> np.ndarray:
    const = PhysicalConstants()
    z = np.asarray(z, dtype=float)
    z_grid = np.linspace(0.0, float(np.max(z)) + 1e-6, 400)
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
    parser = argparse.ArgumentParser(description="Optical-bias smoke test (mock).")
    parser.add_argument("--out", type=Path, default=Path("outputs/optical_bias_smoke"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--h0", type=float, default=70.0)
    parser.add_argument("--om0", type=float, default=0.3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    n = int(args.n)
    z = rng.uniform(0.01, 0.2, size=n)
    ra = rng.uniform(0, 360, size=n)
    dec = rng.uniform(-60, 60, size=n)
    mu_model = fiducial_mu(z, H0=float(args.h0), Om0=float(args.om0))
    sigma_mu = rng.uniform(0.08, 0.15, size=n)
    mu_obs = mu_model + rng.normal(scale=sigma_mu)

    kappa = rng.normal(scale=0.01, size=n)
    weights = h0_estimator_weights(z, sigma_mu)
    delta_h0 = estimate_delta_h0_over_h0(mu_obs, mu_model, weights)
    mean_kappa = float(np.sum(weights * kappa))

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "mock": True,
        "n_sn": int(n),
        "delta_h0_over_h0": float(delta_h0),
        "mean_kappa": mean_kappa,
    }
    write_json(out_dir / "tables" / "summary.json", summary)
    report = "\n".join(
        [
            "# Optical-bias smoke test (mock)",
            "",
            f"N_SN: {n}",
            f"Mean kappa (mock): {mean_kappa:.3e}",
            f"Estimated deltaH0/H0 (mock): {delta_h0:.3e}",
            "",
            "Note: This smoke test uses synthetic data and does not require HEALPix.",
        ]
    )
    write_report(out_dir / "report.md", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

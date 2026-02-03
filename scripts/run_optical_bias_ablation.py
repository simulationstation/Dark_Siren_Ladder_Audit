from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.estimators import evaluate_kappa_at_sn
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.ingest_sn import load_sn_dataset
from entropy_horizon_recon.optical_bias.reporting import write_json, write_report
from entropy_horizon_recon.optical_bias.weights import h0_estimator_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Optical-bias ablation suite (Track A).")
    parser.add_argument("--out", type=Path, default=Path("outputs/optical_bias_ablation"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--sn-path", type=Path, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)

    out_dir = args.out
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    sn = load_sn_dataset(paths=paths, allow_unverified=bool(args.allow_unverified), local_path=args.sn_path)

    nsides = [512, 1024]
    zcuts = [(0.023, 0.15), (0.023, 0.2)]

    rows = []
    for nside in nsides:
        planck = load_planck_kappa(paths=paths, nside_out=nside, allow_unverified=bool(args.allow_unverified))
        kappa_sn = evaluate_kappa_at_sn(planck.kappa_map, sn.ra_deg, sn.dec_deg, nside=planck.nside)
        for zmin, zmax in zcuts:
            weights = h0_estimator_weights(sn.z, sn.sigma_mu, z_min=zmin, z_max=zmax)
            mean_kappa = float(np.sum(weights * kappa_sn))
            rows.append({"nside": nside, "zmin": zmin, "zmax": zmax, "mean_kappa": mean_kappa})

    write_json(out_dir / "tables" / "ablations.json", {"rows": rows})
    lines = ["# Optical-bias ablations", "", "| nside | zmin | zmax | mean_kappa |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['nside']} | {r['zmin']:.3f} | {r['zmax']:.3f} | {r['mean_kappa']:.4e} |")
    write_report(out_dir / "report.md", "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

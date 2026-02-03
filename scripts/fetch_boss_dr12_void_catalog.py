from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.void_prism_inputs import load_boss_dr12_void_catalog_arrays


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch + stage the BOSS DR12 void catalog (Mao+ 2017).")
    ap.add_argument("--out-csv", default="data/processed/void_prism/boss_dr12_voids_mao2017.csv")
    ap.add_argument("--z-min", type=float, default=None)
    ap.add_argument("--z-max", type=float, default=None)
    args = ap.parse_args()

    arrays = load_boss_dr12_void_catalog_arrays(paths=DataPaths(Path.cwd()), z_min=args.z_min, z_max=args.z_max)
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Write a simple CSV compatible with load_void_catalog_csv().
    header = "ra,dec,z,Rv_mpc_h,weight_ngal\n"
    lines = [header]
    for ra, dec, z, rv, w in zip(
        arrays["ra_deg"],
        arrays["dec_deg"],
        arrays["z"],
        arrays["Rv_mpc_h"],
        arrays["weight"],
        strict=False,
    ):
        lines.append(f"{ra:.6f},{dec:.6f},{z:.6f},{rv:.6f},{w:.1f}\n")
    out.write_text("".join(lines))

    print(f"Wrote {out} ({arrays['z'].size} voids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


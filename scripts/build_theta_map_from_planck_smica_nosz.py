from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.maps import _require_healpy
from entropy_horizon_recon.void_prism_cmb import bandpass_filter_map, load_planck_smica_nosz_T


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a kSZ-friendly 'theta' proxy map from Planck SMICA-noSZ temperature.")
    ap.add_argument("--nside", type=int, default=512, help="Output nside (default 512).")
    ap.add_argument("--lmin", type=int, default=300, help="Bandpass lmin (default 300).")
    ap.add_argument("--lmax", type=int, default=1500, help="Bandpass lmax (default 1500).")
    ap.add_argument("--no-remove-dipole", action="store_true", help="Do not remove monopole/dipole before filtering.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output FITS path (default: data/processed/void_prism/theta_planck_smica_nosz_<stamp>.fits).",
    )
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow downloading the Planck CMB map without a pinned SHA256 (bootstrap).",
    )
    args = ap.parse_args()

    hp = _require_healpy()

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"theta_planck_smica_nosz_{_utc_stamp()}.fits"
    out.parent.mkdir(parents=True, exist_ok=True)

    cmb = load_planck_smica_nosz_T(paths=DataPaths(Path.cwd()), nside_out=int(args.nside), allow_unverified=bool(args.allow_unverified))
    theta = bandpass_filter_map(
        cmb.T_map,
        nside=int(args.nside),
        lmin=int(args.lmin),
        lmax=int(args.lmax),
        remove_monopole_dipole=not bool(args.no_remove_dipole),
    )

    hp.write_map(str(out), theta, overwrite=True, dtype=np.float64)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


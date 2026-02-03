from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.maps import _require_healpy
from entropy_horizon_recon.void_prism_maps import (
    load_void_catalog_csv,
    measure_void_prism_spectra,
    select_voids_by_z,
    void_overdensity_map,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_edges(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([int(p) for p in parts], dtype=int)
    if edges.size < 2:
        raise ValueError("Need at least two bin edges.")
    return edges


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure void-prism cross spectra from HEALPix maps + a void catalog.")
    ap.add_argument("--void-csv", required=True, help="Void catalog CSV with ra/dec/z columns.")
    ap.add_argument("--z-min", type=float, required=True)
    ap.add_argument("--z-max", type=float, required=True)
    ap.add_argument("--nside", type=int, default=512, help="Target nside (default 512).")
    ap.add_argument(
        "--frame",
        choices=["icrs", "galactic"],
        default="galactic",
        help="Coordinate frame of the input HEALPix maps (default: galactic; Planck maps are galactic).",
    )

    ap.add_argument("--theta-fits", required=True, help="HEALPix map (fits) of a kSZ velocity proxy (e.g. theta).")
    ap.add_argument("--kappa-fits", default=None, help="HEALPix kappa map (fits). If omitted, use --planck.")
    ap.add_argument("--mask-fits", default=None, help="HEALPix mask (fits). If omitted with --planck, uses Planck mask.")
    ap.add_argument("--extra-mask-fits", default=None, help="Optional additional mask to multiply into the analysis mask.")
    ap.add_argument("--planck", action="store_true", help="Use Planck 2018 kappa map from PLA (download/cache).")

    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--bin-edges", default=None, help="Comma-separated ell bin edges, e.g. '0,50,100,200,400'.")
    ap.add_argument("--out", default=None, help="Output .npz path (default: outputs/void_prism_spectra_<UTCSTAMP>.npz).")
    args = ap.parse_args()

    hp = _require_healpy()

    out = Path(args.out) if args.out else Path("outputs") / f"void_prism_spectra_{_utc_stamp()}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    nside = int(args.nside)
    bin_edges = _parse_edges(args.bin_edges) if args.bin_edges else None

    # Load lensing kappa + mask.
    if args.planck:
        planck = load_planck_kappa(paths=DataPaths(Path.cwd()), nside_out=nside)
        kappa = planck.kappa_map
        mask = planck.mask
    else:
        if args.kappa_fits is None:
            raise ValueError("Provide --kappa-fits or set --planck.")
        kappa = hp.read_map(str(args.kappa_fits), verbose=False)
        mask = hp.read_map(str(args.mask_fits), verbose=False) if args.mask_fits else None

    theta = hp.read_map(str(args.theta_fits), verbose=False)

    # Ensure consistent nside.
    def _ud(m: np.ndarray) -> np.ndarray:
        return hp.ud_grade(m, nside) if hp.get_nside(m) != nside else m

    kappa = _ud(kappa)
    theta = _ud(theta)
    if mask is None:
        mask = np.ones_like(kappa)
    else:
        mask = _ud(mask)
    if args.extra_mask_fits:
        extra = hp.read_map(str(args.extra_mask_fits), verbose=False)
        extra = _ud(extra)
        mask = mask * np.asarray(extra, dtype=float)

    # Load + select voids.
    cat = load_void_catalog_csv(args.void_csv)
    cat = select_voids_by_z(cat, z_min=float(args.z_min), z_max=float(args.z_max))
    vmap = void_overdensity_map(cat, nside=nside, mask=mask, frame=str(args.frame), nest=False)

    # Measure spectra.
    res = measure_void_prism_spectra(
        kappa_map=kappa,
        theta_map=theta,
        void_delta_map=vmap,
        mask=mask,
        lmax=int(args.lmax) if args.lmax is not None else None,
        bin_edges=bin_edges,
    )

    np.savez(
        out,
        **res,
        meta=np.array(
            [
                f"void_csv={args.void_csv}",
                f"z=[{float(args.z_min):.6g},{float(args.z_max):.6g}]",
                f"nside={nside}",
                f"planck={bool(args.planck)}",
                f"kappa={args.kappa_fits or 'planck'}",
                f"theta={args.theta_fits}",
            ],
            dtype=object,
        ),
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

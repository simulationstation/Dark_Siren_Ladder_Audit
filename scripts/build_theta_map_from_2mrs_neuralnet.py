from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.maps import _require_healpy
from entropy_horizon_recon.void_prism_velocity_products import load_2mrs_neuralnet_fields


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _coords_1d(n: int, dx: float) -> np.ndarray:
    return (np.arange(int(n), dtype=float) - (0.5 * (int(n) - 1))) * float(dx)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a HEALPix 'theta' map from the 2MRS NeuralNet 3D velocity field.")
    ap.add_argument("--nside", type=int, default=512)
    ap.add_argument(
        "--mode",
        choices=["radial", "divergence"],
        default="radial",
        help=(
            "Which scalar to project onto the sky. "
            "'radial' averages the line-of-sight (radial) velocity v_r [km/s] in a 3D shell. "
            "'divergence' averages -div(v)/H0 (dimensionless) in a 3D shell, where H0=100 km/s/(Mpc/h)."
        ),
    )
    ap.add_argument("--rmin", type=float, default=0.0, help="Min radius (Mpc/h) used for the shell projection.")
    ap.add_argument("--rmax", type=float, default=200.0, help="Max radius (Mpc/h) used for the shell projection.")
    ap.add_argument("--remove-dipole", action="store_true", help="Remove monopole/dipole from the final map.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output theta FITS (default: data/processed/void_prism/theta_2mrs_nn_<stamp>.fits).",
    )
    ap.add_argument("--mask-out", default=None, help="Optional output mask FITS path.")
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow downloading 2MRS NeuralNet zip without a pinned SHA256 (bootstrap).",
    )
    args = ap.parse_args()

    hp = _require_healpy()

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"theta_2mrs_nn_{_utc_stamp()}.fits"
    out.parent.mkdir(parents=True, exist_ok=True)
    mask_out = Path(args.mask_out) if args.mask_out else None
    if mask_out is not None:
        mask_out.parent.mkdir(parents=True, exist_ok=True)

    fields = load_2mrs_neuralnet_fields(paths=DataPaths(Path.cwd()), allow_unverified=bool(args.allow_unverified), include_errors=False)
    vx = np.asarray(fields.vx_kms, dtype=float)
    vy = np.asarray(fields.vy_kms, dtype=float)
    vz = np.asarray(fields.vz_kms, dtype=float)

    n = int(vx.shape[0])
    dx = float(fields.meta["grid_spacing_mpc_h"])
    coords = _coords_1d(n, dx)
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    r = np.sqrt(X * X + Y * Y + Z * Z)

    m = (r >= float(args.rmin)) & (r <= float(args.rmax)) & (r > 0) & np.isfinite(r)
    if args.mode == "radial":
        m = m & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    else:
        # For divergence we still require finite velocities in a neighborhood, but allow NaNs at
        # edges to be handled by finite-difference propagation.
        m = m & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vz)
    if not np.any(m):
        raise RuntimeError("No voxels selected; adjust --rmin/--rmax.")

    if args.mode == "radial":
        # Line-of-sight velocity in km/s.
        scal = (vx * X + vy * Y + vz * Z) / r
        scal = np.asarray(scal[m], dtype=float)
    else:
        # Velocity divergence in km/s/(Mpc/h) -> dimensionless by dividing by H0=100 km/s/(Mpc/h).
        # Use central differences via np.gradient (2nd-order in the interior).
        dvx_dx = np.gradient(vx, dx, axis=0)
        dvy_dy = np.gradient(vy, dx, axis=1)
        dvz_dz = np.gradient(vz, dx, axis=2)
        div_v = dvx_dx + dvy_dy + dvz_dz
        scal = -div_v / 100.0
        scal = np.asarray(scal[m], dtype=float)
    Xs = np.asarray(X[m], dtype=float)
    Ys = np.asarray(Y[m], dtype=float)
    Zs = np.asarray(Z[m], dtype=float)
    rs = np.asarray(r[m], dtype=float)

    # Galactic lon/lat (same assumption as Lilow+ coordinates).
    lon = (np.rad2deg(np.arctan2(Ys, Xs)) + 360.0) % 360.0
    lat = np.rad2deg(np.arcsin(Zs / rs))
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    pix = hp.ang2pix(int(args.nside), theta, phi, nest=False)

    npix = int(hp.nside2npix(int(args.nside)))
    wsum = np.bincount(pix, weights=scal, minlength=npix).astype(float)
    wcnt = np.bincount(pix, weights=np.ones_like(scal), minlength=npix).astype(float)
    theta_map = np.zeros(npix, dtype=float)
    good = wcnt > 0
    theta_map[good] = wsum[good] / wcnt[good]
    mask = np.zeros(npix, dtype=float)
    mask[good] = 1.0

    if args.remove_dipole:
        theta_map = hp.remove_dipole(theta_map, fitval=False, verbose=False)

    hp.write_map(str(out), theta_map, overwrite=True, dtype=np.float64)
    if mask_out is not None:
        hp.write_map(str(mask_out), mask, overwrite=True, dtype=np.float64)

    print(str(out))
    if mask_out is not None:
        print(str(mask_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

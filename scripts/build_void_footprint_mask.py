from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix
from entropy_horizon_recon.void_prism_maps import load_void_catalog_csv


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _ez(z: np.ndarray, *, omega_m0: float) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    om0 = float(omega_m0)
    return np.sqrt(om0 * (1.0 + z) ** 3 + (1.0 - om0))


def _comoving_distance_mpc_h(z: np.ndarray, *, omega_m0: float, h0_km_s_mpc: float) -> np.ndarray:
    """Crude flat LCDM comoving distance in (Mpc/h) for footprint radii.

    This is intentionally lightweight (no SciPy): simple trapezoid integration of c/H(z).
    Accuracy is more than sufficient for building an approximate survey footprint mask.
    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0) or np.any(~np.isfinite(z)):
        raise ValueError("z must be finite and >= 0.")
    om0 = float(omega_m0)
    if not (0.0 < om0 < 1.0):
        raise ValueError("omega_m0 must be in (0,1).")
    h0 = float(h0_km_s_mpc)
    if not np.isfinite(h0) or h0 <= 0:
        raise ValueError("H0 must be positive.")

    # c/H0 in Mpc
    c_km_s = 299792.458
    c_over_h0_mpc = c_km_s / h0
    # Convert to Mpc/h by multiplying by h (h=H0/100).
    h = h0 / 100.0
    c_over_h0_mpc_h = c_over_h0_mpc * h

    # Vectorized integration per z using a shared grid up to z_max.
    z_max = float(np.max(z))
    nz = max(2000, int(2000 * z_max / 1.0))  # keep resolution reasonable for z~<1
    zg = np.linspace(0.0, z_max, nz)
    inv_e = 1.0 / _ez(zg, omega_m0=om0)
    chi = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * np.diff(zg))
    chi = np.r_[0.0, chi] * c_over_h0_mpc_h
    return np.interp(z, zg, chi)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build an approximate HEALPix footprint mask from a void catalog (union of discs).")
    ap.add_argument("--void-csv", required=True, help="Void catalog CSV with ra/dec/z and optional Rv_mpc_h.")
    ap.add_argument("--nside", type=int, default=512, help="Output mask nside (default 512).")
    ap.add_argument("--frame", choices=["icrs", "galactic"], default="galactic", help="Pixel frame for the output maps (default galactic).")
    ap.add_argument("--rv-col", default="Rv_mpc_h", help="Void radius column name (default Rv_mpc_h).")
    ap.add_argument("--mode", choices=["fixed_deg", "from_Rv"], default="from_Rv", help="Disc radius mode.")
    ap.add_argument("--radius-deg", type=float, default=2.0, help="Fixed disc radius in degrees (fixed_deg mode).")
    ap.add_argument("--radius-factor", type=float, default=1.0, help="Multiply computed disc radii by this factor (from_Rv mode).")
    ap.add_argument("--dilate-deg", type=float, default=0.0, help="Add this many degrees to each disc radius (default 0).")
    ap.add_argument("--omega-m0", type=float, default=0.30, help="LCDM Omega_m0 used for chi(z) (default 0.30).")
    ap.add_argument("--H0", type=float, default=70.0, help="LCDM H0 in km/s/Mpc used for chi(z) (default 70).")
    ap.add_argument("--max-voids", type=int, default=None, help="Optional cap on number of voids (for quick tests).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output mask FITS (default: data/processed/void_prism/mask_<stem>_<stamp>.fits).",
    )
    args = ap.parse_args()

    hp = _require_healpy()

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"mask_{Path(args.void_csv).stem}_{_utc_stamp()}.fits"
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load catalog. For from_Rv mode we require the Rv column.
    cat = load_void_catalog_csv(args.void_csv, Rv_col=str(args.rv_col))
    n = int(cat.z.size)
    if args.max_voids is not None:
        n = min(n, int(args.max_voids))
        cat = type(cat)(ra_deg=cat.ra_deg[:n], dec_deg=cat.dec_deg[:n], z=cat.z[:n], Rv=None if cat.Rv is None else cat.Rv[:n], weight=None)

    # Pixel centers in the target frame.
    pix = radec_to_healpix(cat.ra_deg, cat.dec_deg, nside=int(args.nside), frame=str(args.frame), nest=False)
    theta, phi = hp.pix2ang(int(args.nside), pix, nest=False)
    vec = hp.ang2vec(theta, phi)

    if args.mode == "fixed_deg":
        rad = np.full(n, float(args.radius_deg), dtype=float)
    else:
        if cat.Rv is None:
            raise ValueError("from_Rv mode requires an Rv column in the CSV.")
        chi = _comoving_distance_mpc_h(cat.z, omega_m0=float(args.omega_m0), h0_km_s_mpc=float(args.H0))
        # Small-angle approximation: theta ~ R / chi.
        rad = np.rad2deg(np.asarray(cat.Rv, dtype=float) / np.maximum(chi, 1e-6))
        rad *= float(args.radius_factor)

    rad = np.asarray(rad, dtype=float) + float(args.dilate_deg)
    rad = np.clip(rad, 0.0, 30.0)  # avoid pathological huge discs

    npix = int(hp.nside2npix(int(args.nside)))
    m = np.zeros(npix, dtype=float)
    for i in range(n):
        r = float(rad[i])
        if not np.isfinite(r) or r <= 0:
            continue
        disc = hp.query_disc(int(args.nside), vec[i], math.radians(r), inclusive=False, nest=False)
        m[disc] = 1.0

    hp.write_map(str(out), m, overwrite=True, dtype=np.float64)
    fsky = float(np.mean(m > 0))
    print(f"{out}  fsky={fsky:.4f}  n_voids={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


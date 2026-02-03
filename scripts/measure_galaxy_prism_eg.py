from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.estimators import cross_cl_pseudo
from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix
from entropy_horizon_recon.void_prism_maps import bin_cl, eg_void_from_spectra


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str) -> list[str]:
    return [p.strip() for p in (s or "").split(",") if p.strip()]


def _parse_float_edges(s: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("--z-edges must have at least two comma-separated floats")
    edges = np.array([float(p) for p in parts], dtype=float)
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("--z-edges must be finite and strictly increasing")
    return edges


def _parse_int_edges(s: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("--bin-edges must have at least two comma-separated ints")
    edges = np.array([int(p) for p in parts], dtype=int)
    if np.any(edges < 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("--bin-edges must be >=0 and strictly increasing")
    return edges


@dataclass(frozen=True)
class GalaxyPrismBin:
    z_min: float
    z_max: float
    ell: list[float]
    cl_kappa_gal: list[float]
    cl_theta_gal: list[float]
    eg: list[float]


@dataclass(frozen=True)
class GalaxyPrismMeasurement:
    created_utc: str
    input_dir: str
    theta_fits: list[str]
    theta_mask_fits: list[str]
    frame: str
    nside: int
    z_edges: list[float]
    bin_edges: list[int]
    prefactor: float
    bins: list[GalaxyPrismBin]


def _gal_overdensity_map(
    *,
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    w: np.ndarray,
    ra_deg_r: np.ndarray,
    dec_deg_r: np.ndarray,
    w_r: np.ndarray,
    nside: int,
    frame: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (delta_g, mask) on HEALPix nside using a simple rand-subtracted overdensity."""
    hp = _require_healpy()
    npix = int(hp.nside2npix(int(nside)))

    pix_g = radec_to_healpix(ra_deg, dec_deg, nside=int(nside), frame=frame, nest=False)
    pix_r = radec_to_healpix(ra_deg_r, dec_deg_r, nside=int(nside), frame=frame, nest=False)

    G = np.bincount(pix_g, weights=w, minlength=npix).astype(float)
    R = np.bincount(pix_r, weights=w_r, minlength=npix).astype(float)

    sumG = float(np.sum(G))
    sumR = float(np.sum(R))
    if sumG <= 0 or sumR <= 0:
        raise ValueError("Non-positive total galaxy/random weights.")
    alpha = sumG / sumR

    # Unnormalized fluctuation field f = G - alpha R has (approximately) zero mean by construction.
    f = G - alpha * R
    mask = (R > 0).astype(float)

    # Normalize to unit variance-ish scale for numerical stability (optional but convenient).
    good = mask > 0
    denom = float(np.std(f[good])) if np.any(good) else float(np.std(f))
    if np.isfinite(denom) and denom > 0:
        f = f / denom
    f[~good] = 0.0
    return f, mask


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Measure a *galaxy* version of the prism E_G-like statistic, using:\n"
            "  - Planck 2018 kappa (public)\n"
            "  - theta/velocity-proxy maps built from kszx-prepared catalogs (see build_theta_maps_from_kszpipe_input_dir.py)\n"
            "  - galaxy overdensity built from kszx random catalogs\n"
            "\n"
            "This is intended as a baseline plumbing check before void-conditioning.\n"
        )
    )
    ap.add_argument("--input-dir", required=True, help="kszx KszPipe input_dir (contains galaxies.h5 and randoms.h5).")
    ap.add_argument("--theta-fits", required=True, help="Comma-separated theta maps (one per z bin, or single reused).")
    ap.add_argument("--theta-mask-fits", required=True, help="Comma-separated theta masks (one per z bin, or single reused).")
    ap.add_argument("--z-edges", default="0.43,0.50,0.57,0.64,0.70", help="Comma-separated z edges matching theta bins.")

    ap.add_argument("--nside", type=int, default=256, help="HEALPix nside (default 256).")
    ap.add_argument("--frame", choices=["icrs", "galactic"], default="galactic", help="Frame for RA/Dec->HEALPix (default galactic).")
    ap.add_argument("--bin-edges", default="0,50,100,200,400,800,1200,1500", help="Comma-separated ell bin edges.")
    ap.add_argument("--lmax", type=int, default=1500, help="Max ell for anafast (default 1500).")
    ap.add_argument("--prefactor", type=float, default=1.0, help="Prefactor multiplying C_kappa_g/C_theta_g (default 1).")
    ap.add_argument("--out", default=None, help="Output JSON (default: outputs/galaxy_prism_eg_<stamp>.json).")
    args = ap.parse_args()

    hp = _require_healpy()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    gal_h5 = input_dir / "galaxies.h5"
    ran_h5 = input_dir / "randoms.h5"
    if not gal_h5.exists() or not ran_h5.exists():
        raise FileNotFoundError("Missing galaxies.h5/randoms.h5 in input_dir.")

    z_edges = _parse_float_edges(str(args.z_edges))
    bin_edges = _parse_int_edges(str(args.bin_edges))

    theta_paths = [Path(p) for p in _parse_csv_list(str(args.theta_fits))]
    theta_mask_paths = [Path(p) for p in _parse_csv_list(str(args.theta_mask_fits))]
    if not theta_paths or not theta_mask_paths:
        raise ValueError("theta-fits and theta-mask-fits must be non-empty comma-separated lists.")
    if len(theta_paths) == 1:
        theta_paths = theta_paths * int(z_edges.size - 1)
    if len(theta_mask_paths) == 1:
        theta_mask_paths = theta_mask_paths * int(z_edges.size - 1)
    if len(theta_paths) != int(z_edges.size - 1) or len(theta_mask_paths) != int(z_edges.size - 1):
        raise ValueError("theta map count mismatch vs z_edges.")

    # Load Planck kappa.
    paths = DataPaths()
    planck = load_planck_kappa(paths=paths, nside_out=int(args.nside))
    kappa = np.asarray(planck.kappa_map, dtype=float)
    kappa_mask = np.ones_like(kappa, dtype=float) if planck.mask is None else np.asarray(planck.mask, dtype=float)

    # Load catalogs.
    import kszx

    gcat = kszx.Catalog.from_h5(str(gal_h5))
    rcat = kszx.Catalog.from_h5(str(ran_h5))

    ra_g = np.asarray(gcat.ra_deg, dtype=float)
    dec_g = np.asarray(gcat.dec_deg, dtype=float)
    z_g = np.asarray(gcat.z, dtype=float)
    wg = np.asarray(getattr(gcat, "weight_gal", np.ones_like(z_g)), dtype=float)

    ra_r = np.asarray(rcat.ra_deg, dtype=float)
    dec_r = np.asarray(rcat.dec_deg, dtype=float)
    z_r = np.asarray(getattr(rcat, "zobs", rcat.z), dtype=float)
    wr = np.asarray(getattr(rcat, "weight_gal", np.ones_like(z_r)), dtype=float)

    bins_out: list[GalaxyPrismBin] = []
    for i, (z0, z1) in enumerate(zip(z_edges[:-1], z_edges[1:], strict=False)):
        m_g = (z_g >= z0) & (z_g < z1) & np.isfinite(z_g) & np.isfinite(wg) & (wg > 0)
        m_r = (z_r >= z0) & (z_r < z1) & np.isfinite(z_r) & np.isfinite(wr) & (wr > 0)
        if not np.any(m_g) or not np.any(m_r):
            continue

        delta_g, mask_g = _gal_overdensity_map(
            ra_deg=ra_g[m_g],
            dec_deg=dec_g[m_g],
            w=wg[m_g],
            ra_deg_r=ra_r[m_r],
            dec_deg_r=dec_r[m_r],
            w_r=wr[m_r],
            nside=int(args.nside),
            frame=str(args.frame),
        )

        theta = hp.read_map(str(theta_paths[i]), verbose=False)
        tmask = hp.read_map(str(theta_mask_paths[i]), verbose=False)
        theta = hp.ud_grade(theta, int(args.nside)) if int(hp.get_nside(theta)) != int(args.nside) else theta
        tmask = hp.ud_grade(tmask, int(args.nside)) if int(hp.get_nside(tmask)) != int(args.nside) else tmask

        # Global analysis mask.
        mask = (mask_g > 0) & (tmask > 0) & (kappa_mask > 0)
        mask = mask.astype(float)

        ell, cl_kg = cross_cl_pseudo(kappa, delta_g, mask, lmax=int(args.lmax))
        ell2, cl_tg = cross_cl_pseudo(theta, delta_g, mask, lmax=int(args.lmax))
        if ell2.shape != ell.shape:
            raise RuntimeError("Internal ell mismatch")

        ell_b, cl_kg_b = bin_cl(ell, cl_kg, bin_edges=bin_edges)
        ell_b2, cl_tg_b = bin_cl(ell, cl_tg, bin_edges=bin_edges)
        if not np.allclose(ell_b, ell_b2):
            raise RuntimeError("Binned ell mismatch")

        eg = eg_void_from_spectra(cl_kappa_void=cl_kg_b, cl_theta_void=cl_tg_b, prefactor=float(args.prefactor))
        bins_out.append(
            GalaxyPrismBin(
                z_min=float(z0),
                z_max=float(z1),
                ell=[float(x) for x in ell_b.tolist()],
                cl_kappa_gal=[float(x) for x in cl_kg_b.tolist()],
                cl_theta_gal=[float(x) for x in cl_tg_b.tolist()],
                eg=[float(x) for x in np.asarray(eg, dtype=float).tolist()],
            )
        )

    meas = GalaxyPrismMeasurement(
        created_utc=_utc_stamp(),
        input_dir=str(input_dir),
        theta_fits=[str(p) for p in theta_paths],
        theta_mask_fits=[str(p) for p in theta_mask_paths],
        frame=str(args.frame),
        nside=int(args.nside),
        z_edges=[float(x) for x in z_edges.tolist()],
        bin_edges=[int(x) for x in bin_edges.tolist()],
        prefactor=float(args.prefactor),
        bins=bins_out,
    )

    out_path = Path(args.out) if args.out else Path("outputs") / f"galaxy_prism_eg_{_utc_stamp()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(meas), indent=2, sort_keys=True) + "\n")
    print(f"[galaxy_prism_eg] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import multiprocessing as mp
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.maps import _require_healpy
from entropy_horizon_recon.void_prism_maps import (
    eg_void_from_spectra,
    jackknife_covariance,
    jackknife_region_index,
    load_void_catalog_csv,
    measure_void_prism_spectra,
    select_voids_by_Rv,
    select_voids_by_z,
    void_overdensity_map,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


# -----------------------------------------------------------------------------
# Multiprocessing helper: module-level globals + top-level worker for picklability.
# We rely on 'fork' so workers inherit large arrays without pickling overhead.
# -----------------------------------------------------------------------------

_JK_KAPPA: np.ndarray | None = None
_JK_THETA: np.ndarray | None = None
_JK_MASK: np.ndarray | None = None
_JK_REGION_ID: np.ndarray | None = None
_JK_CAT: object | None = None
_JK_NSIDE: int | None = None
_JK_FRAME: str | None = None
_JK_LMAX: int | None = None
_JK_BIN_EDGES: np.ndarray | None = None
_JK_PREFACTOR: float | None = None


def _jk_one_region(rid: int) -> np.ndarray:
    kappa = _JK_KAPPA
    theta = _JK_THETA
    mask = _JK_MASK
    region_id = _JK_REGION_ID
    cat = _JK_CAT
    nside = _JK_NSIDE
    frame = _JK_FRAME
    lmax = _JK_LMAX
    bin_edges = _JK_BIN_EDGES
    pref = _JK_PREFACTOR
    if any(v is None for v in (kappa, theta, mask, region_id, cat, nside, frame, pref)):
        raise RuntimeError("Jackknife globals not initialized.")

    m = np.asarray(mask, dtype=float).copy()
    m[np.asarray(region_id) == int(rid)] = 0.0
    vmap_i = void_overdensity_map(cat, nside=int(nside), mask=m, frame=str(frame), nest=False)
    res_i = measure_void_prism_spectra(
        kappa_map=np.asarray(kappa, dtype=float),
        theta_map=np.asarray(theta, dtype=float),
        void_delta_map=vmap_i,
        mask=m,
        lmax=int(lmax) if lmax is not None else None,
        bin_edges=bin_edges,
    )
    return eg_void_from_spectra(
        cl_kappa_void=res_i["cl_kappa_void"],
        cl_theta_void=res_i["cl_theta_void"],
        prefactor=float(pref),
    )


@dataclass(frozen=True)
class Meta:
    void_csv: str
    z_min: float
    z_max: float
    Rv_min: float | None
    Rv_max: float | None
    nside: int
    theta_fits: str
    lmax: int | None
    bin_edges: list[int] | None
    jackknife_nside: int | None
    prefactor: float
    z_eff: float
    n_voids: int


def _parse_edges(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([int(p) for p in parts], dtype=int)
    if edges.size < 2:
        raise ValueError("Need at least two bin edges.")
    return edges


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure a void-prism E_G(ell) vector with jackknife covariance.")
    ap.add_argument("--void-csv", required=True, help="Void catalog CSV with ra/dec/z columns.")
    ap.add_argument("--z-min", type=float, required=True)
    ap.add_argument("--z-max", type=float, required=True)
    ap.add_argument("--Rv-min", type=float, default=None)
    ap.add_argument("--Rv-max", type=float, default=None)
    ap.add_argument("--nside", type=int, default=512, help="Target nside for maps (default 512).")
    ap.add_argument(
        "--frame",
        choices=["icrs", "galactic"],
        default="galactic",
        help="Coordinate frame of the input HEALPix maps (default: galactic; Planck maps are galactic).",
    )
    ap.add_argument("--theta-fits", required=True, help="HEALPix map (fits) of kSZ/velocity proxy (theta).")
    ap.add_argument("--planck", action="store_true", help="Use Planck 2018 lensing kappa from PLA (download/cache).")
    ap.add_argument("--kappa-fits", default=None, help="If not --planck, provide kappa map fits.")
    ap.add_argument("--mask-fits", default=None, help="Optional mask fits (if not --planck).")
    ap.add_argument("--extra-mask-fits", default=None, help="Optional additional mask to multiply into the analysis mask.")
    ap.add_argument("--lmax", type=int, default=None)
    ap.add_argument("--bin-edges", default=None, help="Comma-separated ell bin edges, e.g. '0,50,100,200,400'.")
    ap.add_argument("--prefactor", type=float, default=1.0, help="Overall normalization constant.")
    ap.add_argument("--jackknife-nside", type=int, default=None, help="Low-res nside used for jackknife regions (e.g. 4 or 8).")
    ap.add_argument("--n-proc", type=int, default=1, help="Parallel processes for jackknife (default 1).")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print jackknife progress every N regions (default 10; always prints for <=200 regions).",
    )
    ap.add_argument("--out-base", default=None, help="Output directory (default: outputs/void_prism_eg_meas_<stamp>/).")
    ap.add_argument("--name", default=None, help="Measurement name.")
    args = ap.parse_args()

    global _JK_KAPPA, _JK_THETA, _JK_MASK, _JK_REGION_ID, _JK_CAT, _JK_NSIDE, _JK_FRAME, _JK_LMAX, _JK_BIN_EDGES, _JK_PREFACTOR

    hp = _require_healpy()

    out_dir = Path(args.out_base) if args.out_base else Path("outputs") / f"void_prism_eg_meas_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    nside = int(args.nside)
    bin_edges = _parse_edges(args.bin_edges)

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

    # Load voids + select bin.
    cat = load_void_catalog_csv(args.void_csv, Rv_col="Rv_mpc_h", weight_col="weight_ngal")
    cat = select_voids_by_z(cat, z_min=float(args.z_min), z_max=float(args.z_max))
    if args.Rv_min is not None and args.Rv_max is not None:
        cat = select_voids_by_Rv(cat, Rv_min=float(args.Rv_min), Rv_max=float(args.Rv_max))
    z_eff = float(np.mean(cat.z))

    vmap = void_overdensity_map(cat, nside=nside, mask=mask, frame=str(args.frame), nest=False)
    res = measure_void_prism_spectra(
        kappa_map=kappa,
        theta_map=theta,
        void_delta_map=vmap,
        mask=mask,
        lmax=int(args.lmax) if args.lmax is not None else None,
        bin_edges=bin_edges,
    )
    eg = eg_void_from_spectra(cl_kappa_void=res["cl_kappa_void"], cl_theta_void=res["cl_theta_void"], prefactor=float(args.prefactor))

    # Jackknife covariance (optional).
    cov = None
    if args.jackknife_nside is not None:
        nside_jk = int(args.jackknife_nside)
        region_id = jackknife_region_index(nside=nside, nside_jk=nside_jk)
        good_regions = np.unique(region_id[np.asarray(mask) > 0])
        print(f"[jk] nside_jk={nside_jk}  regions={good_regions.size}  n_proc={int(args.n_proc)}")
        eg_jk = np.empty((good_regions.size, eg.size), dtype=float)

        n_proc = int(args.n_proc)
        progress_every = int(args.progress_every)
        if good_regions.size <= 200:
            progress_every = 1

        if n_proc <= 1:
            _JK_KAPPA = kappa
            _JK_THETA = theta
            _JK_MASK = mask
            _JK_REGION_ID = region_id
            _JK_CAT = cat
            _JK_NSIDE = nside
            _JK_FRAME = str(args.frame)
            _JK_LMAX = int(args.lmax) if args.lmax is not None else None
            _JK_BIN_EDGES = bin_edges
            _JK_PREFACTOR = float(args.prefactor)

            for i, rid in enumerate(good_regions):
                eg_jk[i] = _jk_one_region(int(rid))
                if (i + 1) % progress_every == 0 or (i + 1) == good_regions.size:
                    pct = 100.0 * float(i + 1) / float(good_regions.size)
                    print(f"[jk] {i+1}/{good_regions.size} ({pct:.1f}%)")
        else:
            # Use fork to avoid pickling large maps on Linux.
            _JK_KAPPA = kappa
            _JK_THETA = theta
            _JK_MASK = mask
            _JK_REGION_ID = region_id
            _JK_CAT = cat
            _JK_NSIDE = nside
            _JK_FRAME = str(args.frame)
            _JK_LMAX = int(args.lmax) if args.lmax is not None else None
            _JK_BIN_EDGES = bin_edges
            _JK_PREFACTOR = float(args.prefactor)

            ctx = mp.get_context("fork")
            with ctx.Pool(processes=n_proc) as pool:
                for i, eg_i in enumerate(pool.imap(_jk_one_region, [int(r) for r in good_regions], chunksize=1)):
                    eg_jk[i] = eg_i
                    if (i + 1) % progress_every == 0 or (i + 1) == good_regions.size:
                        pct = 100.0 * float(i + 1) / float(good_regions.size)
                        print(f"[jk] {i+1}/{good_regions.size} ({pct:.1f}%)")
        cov = jackknife_covariance(eg_jk)

    # Write artifacts.
    ell = res["ell"]
    np.savez(out_dir / "spectra_and_eg.npz", ell=ell, eg=eg, cov=cov if cov is not None else np.array([]))

    if args.name:
        name = str(args.name)
    else:
        stem = Path(args.void_csv).stem
        name = f"{stem}_void_prism_z{float(args.z_min):.2f}_{float(args.z_max):.2f}"
    meas = {
        "measurements": [
            {
                "name": name,
                "z_eff": z_eff,
                "ell": ell.astype(int).tolist(),
                "eg_obs": eg.tolist(),
                "cov": cov.tolist() if cov is not None else np.diag(np.full_like(eg, 1.0)).tolist(),
                "notes": "Derived from maps. If cov is identity, jackknife was not run.",
                "source": {
                    "kappa": "planck" if args.planck else str(args.kappa_fits),
                    "theta": str(args.theta_fits),
                    "void_csv": str(args.void_csv),
                    "prefactor": float(args.prefactor),
                    "jackknife_nside": int(args.jackknife_nside) if args.jackknife_nside is not None else None,
                },
            }
        ]
    }
    (tab_dir / "measurement.json").write_text(json.dumps(meas, indent=2))

    meta = Meta(
        void_csv=str(args.void_csv),
        z_min=float(args.z_min),
        z_max=float(args.z_max),
        Rv_min=float(args.Rv_min) if args.Rv_min is not None else None,
        Rv_max=float(args.Rv_max) if args.Rv_max is not None else None,
        nside=nside,
        theta_fits=str(args.theta_fits),
        lmax=int(args.lmax) if args.lmax is not None else None,
        bin_edges=bin_edges.astype(int).tolist() if bin_edges is not None else None,
        jackknife_nside=int(args.jackknife_nside) if args.jackknife_nside is not None else None,
        prefactor=float(args.prefactor),
        z_eff=z_eff,
        n_voids=int(cat.z.size),
    )
    (tab_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2))

    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

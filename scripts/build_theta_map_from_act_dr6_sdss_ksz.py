from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str) -> list[str]:
    out = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not out:
        raise ValueError("Expected a comma-separated list.")
    return out


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def _subtract_z_binned_means(z: np.ndarray, t: np.ndarray, w: np.ndarray, *, z_edges: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    out = np.array(t, copy=True)
    for z0, z1 in zip(z_edges[:-1], z_edges[1:], strict=False):
        m = (z >= float(z0)) & (z < float(z1)) & np.isfinite(z)
        mu = _weighted_mean(out[m], w[m])
        if np.isfinite(mu):
            out[m] -= mu
    return out


def _run(cmd: list[str]) -> None:
    # Keep this lightweight and shell-free for safety and portability.
    print(f"[build_theta] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _ensure_wget(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Use resumable downloads (unlike python wget) to avoid "restart from 0%" failures.
    _run(["wget", "-c", url, "-O", str(dst)])


def _sdss_dr_str(dr: int) -> str:
    if dr == 11:
        return "DR11v1"
    if dr == 12:
        return "DR12v5"
    raise ValueError(f"Unsupported SDSS dr={dr} (kszx supports DR11/DR12).")


def _ensure_gunzip(src_gz: Path, dst: Path) -> None:
    if dst.exists():
        return
    import gzip

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(dst) + ".tmp")
    try:
        with gzip.open(src_gz, "rb") as f_in, open(tmp, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out, length=1024 * 1024)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def _ensure_sdss_galaxy_catalog(*, kszx_data_dir: Path, survey: str, dr: int) -> None:
    dr_str = _sdss_dr_str(int(dr))
    base = kszx_data_dir / "sdss" / dr_str
    fits = base / f"galaxy_{dr_str}_{survey}.fits"
    gz = Path(str(fits) + ".gz")  # -> .fits.gz
    url = f"https://data.sdss.org/sas/dr{int(dr)}/boss/lss/{fits.name}.gz"
    _ensure_wget(url, gz)
    _ensure_gunzip(gz, fits)


def _ensure_act_file(*, kszx_data_dir: Path, relpath: str, dr: int) -> None:
    if int(dr) == 5:
        dr_dir = "dr5.01"
        url_base = "https://lambda.gsfc.nasa.gov/data/suborbital/ACT/ACT_dr5/maps/"
    elif int(dr) == 6:
        dr_dir = "dr6.02"
        url_base = "https://lambda.gsfc.nasa.gov/data/act/"
    else:
        raise ValueError(f"Unsupported ACT dr={dr} (kszx supports DR5/DR6).")

    dst = kszx_data_dir / "act" / dr_dir / relpath
    _ensure_wget(url_base + relpath, dst)


@dataclass(frozen=True)
class Meta:
    created_utc: str
    act_dr: int
    act_freq_ghz: int
    act_night: bool
    sdss_surveys: list[str]
    sdss_dr: int
    z_min: float
    z_max: float
    z_edges: list[float]
    n_gal_used: int
    frame: str
    nside: int
    lmin: int
    lmax: int
    weights: str
    use_ivar: bool
    remove_dipole: bool
    kszx_data_dir: str


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build a HEALPix 'theta' proxy map from ACT DR6 srcfree temperature evaluated at SDSS/BOSS galaxies.\n"
            "This is a *real kSZ-derived velocity proxy* (still noisy): it uses a high-l bandpass filtered ACT map,\n"
            "samples T at galaxy positions, subtracts mean(T) in redshift bins, and bins to HEALPix."
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--act-dr", type=int, default=6, help="ACT data release (default 6).")
    ap.add_argument("--act-freq", type=int, default=150, help="ACT frequency in GHz (default 150).")
    ap.add_argument("--act-night", action="store_true", help="Use ACT night coadd (default daynight).")
    ap.add_argument("--sdss-surveys", default="CMASSLOWZTOT_North,CMASSLOWZTOT_South", help="Comma-separated SDSS surveys.")
    ap.add_argument("--sdss-dr", type=int, default=12, help="SDSS data release (default 12).")
    ap.add_argument("--zmin", type=float, default=0.20, help="Min redshift (default 0.20).")
    ap.add_argument("--zmax", type=float, default=0.70, help="Max redshift (default 0.70).")
    ap.add_argument("--nzbins", type=int, default=8, help="Number of z bins for mean subtraction (default 8).")
    ap.add_argument(
        "--zbinning",
        choices=["uniform", "quantile"],
        default="quantile",
        help="How to define redshift bins for mean subtraction (default quantile).",
    )
    ap.add_argument("--nside", type=int, default=256, help="Output HEALPix nside (default 256).")
    ap.add_argument("--frame", choices=["galactic", "icrs"], default="galactic", help="Output HEALPix frame (default galactic).")
    ap.add_argument("--lmin", type=int, default=300, help="Flat-sky bandpass lmin (default 300).")
    ap.add_argument("--lmax", type=int, default=3000, help="Flat-sky bandpass lmax (default 3000).")
    ap.add_argument(
        "--weights",
        choices=["wfkp", "fkp_sys"],
        default="fkp_sys",
        help="Galaxy weights used for binning (default fkp_sys).",
    )
    ap.add_argument("--use-ivar", action="store_true", help="Multiply galaxy weights by ACT inverse-variance at each position.")
    ap.add_argument("--max-galaxies", type=int, default=None, help="Optional subsample size for a fast smoke run.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed used when --max-galaxies is set.")
    ap.add_argument("--remove-dipole", action="store_true", help="Remove monopole/dipole from theta map (after binning).")
    ap.add_argument("--mask-out", default=None, help="Optional output mask FITS path (default alongside theta).")
    ap.add_argument("--meta-out", default=None, help="Optional output metadata JSON (default alongside theta).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output theta FITS (default: data/processed/void_prism/theta_act_dr6_sdss_<stamp>.fits).",
    )
    args = ap.parse_args()

    hp = _require_healpy()

    kszx_data_dir = Path(args.kszx_data_dir).resolve()
    kszx_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KSZX_DATA_DIR"] = str(kszx_data_dir)

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"theta_act_dr{int(args.act_dr)}_sdss_{_utc_stamp()}.fits"
    out.parent.mkdir(parents=True, exist_ok=True)
    mask_out = Path(args.mask_out) if args.mask_out else out.with_name(out.name.replace("theta_", "mask_"))
    meta_out = Path(args.meta_out) if args.meta_out else out.with_suffix(".meta.json")

    # Defer heavyweight imports until after we set KSZX_DATA_DIR.
    import pixell.enmap as enmap  # type: ignore
    import kszx  # type: ignore

    from kszx import act as kact  # type: ignore
    from kszx import pixell_utils as pxu  # type: ignore
    from kszx import sdss as ksdss  # type: ignore
    from kszx.Catalog import Catalog  # type: ignore

    # Robust/resumable downloads for ACT and SDSS. (Avoids python-wget "restart from 0%" failures.)
    act_time = "night" if bool(args.act_night) else "daynight"
    if int(args.act_dr) == 5:
        cmb_rel = f"act_planck_dr5.01_s08s18_AA_f{int(args.act_freq):03d}_{act_time}_map_srcfree.fits"
        ivar_rel = f"act_planck_dr5.01_s08s18_AA_f{int(args.act_freq):03d}_{act_time}_ivar.fits"
    elif int(args.act_dr) == 6:
        cmb_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(args.act_freq):03d}_map_srcfree.fits"
        ivar_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(args.act_freq):03d}_ivar.fits"
    else:
        raise ValueError(f"Unsupported ACT dr={args.act_dr} (supported: 5,6).")

    _ensure_act_file(kszx_data_dir=kszx_data_dir, relpath=cmb_rel, dr=int(args.act_dr))
    _ensure_act_file(kszx_data_dir=kszx_data_dir, relpath=ivar_rel, dr=int(args.act_dr))

    surveys = _parse_csv_list(str(args.sdss_surveys))
    for s in surveys:
        _ensure_sdss_galaxy_catalog(kszx_data_dir=kszx_data_dir, survey=s, dr=int(args.sdss_dr))

    # Load ACT map + ivar.
    cmb = kact.read_cmb(int(args.act_freq), int(args.act_dr), night=bool(args.act_night), download=False)
    ivar = kact.read_ivar(int(args.act_freq), int(args.act_dr), night=bool(args.act_night), download=False)
    if cmb.shape != ivar.shape:
        raise RuntimeError(f"ACT cmb/ivar shapes differ: {cmb.shape} vs {ivar.shape}")

    # Flat-sky bandpass filter.
    lmap = enmap.modlmap(cmb.shape, cmb.wcs)
    m = (lmap >= float(args.lmin)) & (lmap <= float(args.lmax))
    cmb_f = enmap.ifft(enmap.fft(cmb) * m).real

    # Load SDSS galaxy catalogs and merge.
    cats: list[Catalog] = []
    for s in surveys:
        cat = ksdss.read_galaxies(s, dr=int(args.sdss_dr), download=False)
        cat.apply_redshift_cut(float(args.zmin), float(args.zmax))
        cats.append(cat)
    gcat = Catalog.concatenate(cats, name=" + ".join(surveys), destructive=True) if len(cats) > 1 else cats[0]

    # Base per-galaxy weights.
    if args.weights == "wfkp":
        w = np.asarray(gcat.wfkp, dtype=float)
    else:
        # SDSS reminder: (wzf + wcp − 1) * wsys * wfkp
        w = np.asarray((gcat.wzf + gcat.wcp - 1.0) * gcat.wsys * gcat.wfkp, dtype=float)

    # Evaluate ACT temperature (filtered) at galaxy positions.
    t, inbounds = pxu.eval_map_on_catalog(cmb_f, gcat, pad=0.0, return_mask=True)
    t = np.asarray(t, dtype=float)
    inbounds = np.asarray(inbounds, dtype=bool)
    z = np.asarray(gcat.z, dtype=float)
    ra = np.asarray(gcat.ra_deg, dtype=float)
    dec = np.asarray(gcat.dec_deg, dtype=float)

    # Optionally include ivar.
    if bool(args.use_ivar):
        iv, inb2 = pxu.eval_map_on_catalog(ivar, gcat, pad=0.0, return_mask=True)
        iv = np.asarray(iv, dtype=float)
        inbounds &= np.asarray(inb2, dtype=bool)
        w = w * np.maximum(iv, 0.0)

    m0 = inbounds & np.isfinite(t) & np.isfinite(z) & np.isfinite(w) & (w > 0.0)
    if not np.any(m0):
        raise RuntimeError("No valid galaxies overlap ACT map footprint after masks/weights.")

    ra = ra[m0]
    dec = dec[m0]
    z = z[m0]
    t = t[m0]
    w = w[m0]

    # Optional subsample for smoke tests.
    if args.max_galaxies is not None and int(args.max_galaxies) > 0 and int(args.max_galaxies) < ra.size:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(ra.size, size=int(args.max_galaxies), replace=False)
        ra, dec, z, t, w = ra[idx], dec[idx], z[idx], t[idx], w[idx]

    # Define z-bin edges and subtract per-bin means.
    nz = int(args.nzbins)
    if nz < 1:
        raise ValueError("--nzbins must be >= 1.")
    if args.zbinning == "uniform":
        z_edges = np.linspace(float(args.zmin), float(args.zmax), nz + 1)
    else:
        qs = np.linspace(0.0, 1.0, nz + 1)
        z_edges = np.quantile(z, qs)
        z_edges[0] = float(args.zmin)
        z_edges[-1] = float(args.zmax)
        # Ensure strict monotonicity.
        for i in range(1, z_edges.size):
            if z_edges[i] <= z_edges[i - 1]:
                z_edges[i] = np.nextafter(z_edges[i - 1], np.inf)

    t = _subtract_z_binned_means(z, t, w, z_edges=np.asarray(z_edges, dtype=float))

    # Bin to HEALPix.
    pix = radec_to_healpix(ra, dec, nside=int(args.nside), frame=str(args.frame), nest=False)
    npix = int(hp.nside2npix(int(args.nside)))
    wsum = np.bincount(pix, weights=w * t, minlength=npix).astype(float)
    wcnt = np.bincount(pix, weights=w, minlength=npix).astype(float)
    theta_map = np.zeros(npix, dtype=float)
    good = wcnt > 0
    theta_map[good] = wsum[good] / wcnt[good]
    mask = np.zeros(npix, dtype=float)
    mask[good] = 1.0

    if bool(args.remove_dipole):
        theta_map = hp.remove_dipole(theta_map, fitval=False, verbose=False)

    hp.write_map(str(out), theta_map, overwrite=True, dtype=np.float64)
    hp.write_map(str(mask_out), mask, overwrite=True, dtype=np.float64)

    meta = Meta(
        created_utc=_utc_stamp(),
        act_dr=int(args.act_dr),
        act_freq_ghz=int(args.act_freq),
        act_night=bool(args.act_night),
        sdss_surveys=surveys,
        sdss_dr=int(args.sdss_dr),
        z_min=float(args.zmin),
        z_max=float(args.zmax),
        z_edges=[float(x) for x in z_edges],
        n_gal_used=int(ra.size),
        frame=str(args.frame),
        nside=int(args.nside),
        lmin=int(args.lmin),
        lmax=int(args.lmax),
        weights=str(args.weights),
        use_ivar=bool(args.use_ivar),
        remove_dipole=bool(args.remove_dipole),
        kszx_data_dir=str(kszx_data_dir),
    )
    meta_out.write_text(json_dumps(asdict(meta)))

    print(str(out))
    print(str(mask_out))
    print(str(meta_out))
    return 0


def json_dumps(d) -> str:
    import json

    return json.dumps(d, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

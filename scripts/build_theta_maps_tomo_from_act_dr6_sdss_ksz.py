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


def _parse_float_edges(s: str | None) -> np.ndarray | None:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    edges = np.array([float(p) for p in parts], dtype=float)
    if edges.size < 2 or np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("Edges must be finite, strictly increasing, length>=2.")
    return edges


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def _run(cmd: list[str]) -> None:
    # Keep this lightweight and shell-free for safety and portability.
    print(f"[build_theta_tomo] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _ensure_wget(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() and dst.stat().st_size > 0:
            return
    except Exception:
        pass
    # Prefer resumable downloads (avoid restart-from-0 failures), but fall back to the Python
    # 'wget' module if system wget has TLS issues in this environment.
    try:
        _run(["wget", "-c", url, "-O", str(dst)])
    except subprocess.CalledProcessError as e:
        # Clean up any empty/partial file before fallback.
        try:
            if dst.exists() and dst.stat().st_size == 0:
                dst.unlink()
        except Exception:
            pass
        print(f"[build_theta_tomo] WARN: system wget failed (code={e.returncode}); trying python wget fallback", flush=True)
        _run(
            [
                os.fspath(Path(".venv/bin/python").resolve()) if Path(".venv/bin/python").exists() else "python3",
                "-c",
                (
                    "import wget,sys; "
                    "url=sys.argv[1]; out=sys.argv[2]; "
                    "wget.download(url, out=out); "
                    "print()"
                ),
                url,
                str(dst),
            ]
        )
        if not dst.exists() or dst.stat().st_size == 0:
            raise RuntimeError(f"Download failed for url={url} -> {dst}")


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
class ThetaBinMeta:
    z_min: float
    z_max: float
    n_gal_used: int
    theta_fits: str
    mask_fits: str


@dataclass(frozen=True)
class Manifest:
    created_utc: str
    act_dr: int
    act_freqs_ghz: list[int]
    act_night: bool
    sdss_surveys: list[str]
    sdss_dr: int
    frame: str
    nside: int
    lmin: int
    lmax: int
    weights: str
    use_ivar: bool
    remove_dipole: bool
    kszx_data_dir: str
    z_edges: list[float]
    bins: list[ThetaBinMeta]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build tomographic HEALPix theta/velocity-proxy maps from ACT srcfree temperature evaluated at SDSS/BOSS galaxies.\n"
            "\n"
            "This is a kSZ-derived *velocity proxy* (still noisy): for each z bin, we bandpass-filter the ACT map(s),\n"
            "sample T at galaxy positions, subtract the weighted mean within the bin, and bin to HEALPix.\n"
            "\n"
            "Differences vs scripts/build_theta_map_from_act_dr6_sdss_ksz.py:\n"
            "  - outputs one theta map per z bin (tomography)\n"
            "  - can combine multiple ACT frequencies (e.g. 90+150) with ivar weights\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--act-dr", type=int, default=6, help="ACT data release (default 6).")
    ap.add_argument("--act-freqs", default="90,150", help="Comma-separated ACT frequencies in GHz (default 90,150).")
    ap.add_argument("--act-night", action="store_true", help="Use ACT night coadd (default daynight).")
    ap.add_argument("--sdss-surveys", default="CMASSLOWZTOT_North,CMASSLOWZTOT_South", help="Comma-separated SDSS surveys.")
    ap.add_argument("--sdss-dr", type=int, default=12, help="SDSS data release (default 12).")
    ap.add_argument("--zmin", type=float, default=0.20, help="Min redshift (default 0.20).")
    ap.add_argument("--zmax", type=float, default=0.70, help="Max redshift (default 0.70).")
    ap.add_argument(
        "--z-edges",
        default=None,
        help="Comma-separated z edges (tomographic bins). If omitted, uses --nzbins with --zbinning.",
    )
    ap.add_argument("--nzbins", type=int, default=4, help="If --z-edges omitted, number of z bins (default 4).")
    ap.add_argument(
        "--zbinning",
        choices=["uniform", "quantile"],
        default="uniform",
        help="If --z-edges omitted, how to define z bins (default uniform).",
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
    ap.add_argument("--use-ivar", action="store_true", help="Use ACT ivar at each galaxy position for per-galaxy inverse-noise weighting.")
    ap.add_argument("--remove-dipole", action="store_true", help="Remove monopole/dipole from each theta map (after binning).")
    ap.add_argument("--max-galaxies", type=int, default=None, help="Optional global subsample size for a fast smoke run.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed used when --max-galaxies is set.")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: data/processed/void_prism/theta_tomo_<stamp>/).")
    args = ap.parse_args()

    hp = _require_healpy()

    kszx_data_dir = Path(args.kszx_data_dir).resolve()
    kszx_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KSZX_DATA_DIR"] = str(kszx_data_dir)

    out_dir = Path(args.out_dir) if args.out_dir else Path("data/processed/void_prism") / f"theta_tomo_act_sdss_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defer heavyweight imports until after we set KSZX_DATA_DIR.
    import pixell.enmap as enmap  # type: ignore
    import kszx  # type: ignore

    from kszx import act as kact  # type: ignore
    from kszx import pixell_utils as pxu  # type: ignore
    from kszx import sdss as ksdss  # type: ignore
    from kszx.Catalog import Catalog  # type: ignore

    act_time = "night" if bool(args.act_night) else "daynight"

    freqs = [int(f) for f in _parse_csv_list(str(args.act_freqs))]
    if not freqs:
        raise ValueError("--act-freqs must be non-empty.")

    # Robust/resumable downloads for ACT and SDSS (avoid python-wget restarts).
    for f in freqs:
        if int(args.act_dr) == 5:
            cmb_rel = f"act_planck_dr5.01_s08s18_AA_f{int(f):03d}_{act_time}_map_srcfree.fits"
            ivar_rel = f"act_planck_dr5.01_s08s18_AA_f{int(f):03d}_{act_time}_ivar.fits"
        elif int(args.act_dr) == 6:
            cmb_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(f):03d}_map_srcfree.fits"
            ivar_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(f):03d}_ivar.fits"
        else:
            raise ValueError(f"Unsupported ACT dr={args.act_dr} (supported: 5,6).")
        _ensure_act_file(kszx_data_dir=kszx_data_dir, relpath=cmb_rel, dr=int(args.act_dr))
        _ensure_act_file(kszx_data_dir=kszx_data_dir, relpath=ivar_rel, dr=int(args.act_dr))

    surveys = _parse_csv_list(str(args.sdss_surveys))
    for s in surveys:
        _ensure_sdss_galaxy_catalog(kszx_data_dir=kszx_data_dir, survey=s, dr=int(args.sdss_dr))

    # Load SDSS galaxy catalogs and merge.
    cats: list[Catalog] = []
    for s in surveys:
        cat = ksdss.read_galaxies(s, dr=int(args.sdss_dr), download=False)
        cat.apply_redshift_cut(float(args.zmin), float(args.zmax))
        cats.append(cat)
    gcat = Catalog.concatenate(cats, name=" + ".join(surveys), destructive=True) if len(cats) > 1 else cats[0]

    # Base per-galaxy weights.
    if args.weights == "wfkp":
        w_base = np.asarray(gcat.wfkp, dtype=float)
    else:
        # SDSS reminder: (wzf + wcp − 1) * wsys * wfkp
        w_base = np.asarray((gcat.wzf + gcat.wcp - 1.0) * gcat.wsys * gcat.wfkp, dtype=float)

    z = np.asarray(gcat.z, dtype=float)
    ra = np.asarray(gcat.ra_deg, dtype=float)
    dec = np.asarray(gcat.dec_deg, dtype=float)

    m0 = np.isfinite(z) & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(w_base) & (w_base > 0)
    if not np.any(m0):
        raise RuntimeError("No valid galaxies after weights/finite cuts.")
    # Subset gcat by mask for eval_map_on_catalog (expects Catalog, but we can carry arrays ourselves).
    ra = ra[m0]
    dec = dec[m0]
    z = z[m0]
    w_base = w_base[m0]

    # Optional global subsample for smoke tests.
    if args.max_galaxies is not None and 0 < int(args.max_galaxies) < ra.size:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(ra.size, size=int(args.max_galaxies), replace=False)
        ra, dec, z, w_base = ra[idx], dec[idx], z[idx], w_base[idx]

    # Define z bin edges.
    z_edges = _parse_float_edges(args.z_edges)
    if z_edges is None:
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
            for i in range(1, z_edges.size):
                if z_edges[i] <= z_edges[i - 1]:
                    z_edges[i] = np.nextafter(z_edges[i - 1], np.inf)
    z_edges = np.asarray(z_edges, dtype=float)

    # Load and bandpass-filter ACT maps once per frequency.
    cmb_f_by_freq: dict[int, enmap.ndmap] = {}
    ivar_by_freq: dict[int, enmap.ndmap] = {}
    for f in freqs:
        cmb = kact.read_cmb(int(f), int(args.act_dr), night=bool(args.act_night), download=False)
        ivar = kact.read_ivar(int(f), int(args.act_dr), night=bool(args.act_night), download=False)
        if cmb.shape != ivar.shape:
            raise RuntimeError(f"ACT cmb/ivar shapes differ for f={f}: {cmb.shape} vs {ivar.shape}")

        lmap = enmap.modlmap(cmb.shape, cmb.wcs)
        m = (lmap >= float(args.lmin)) & (lmap <= float(args.lmax))
        cmb_f = enmap.ifft(enmap.fft(cmb) * m).real
        cmb_f_by_freq[int(f)] = cmb_f
        ivar_by_freq[int(f)] = ivar

    # Build theta maps per z bin.
    bins: list[ThetaBinMeta] = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:], strict=False):
        m_z = (z >= float(z0)) & (z < float(z1)) & np.isfinite(z)
        if not np.any(m_z):
            continue
        ra_sel = ra[m_z]
        dec_sel = dec[m_z]
        w_sel = w_base[m_z]

        # Build a minimal Catalog for eval_map_on_catalog.
        # Catalog expects ra/dec/z arrays; it can be created from columns in a dict.
        # We avoid constructing a full SDSS Catalog slice to keep this script simple.
        from kszx.Catalog import Catalog  # type: ignore

        cat_bin = Catalog({"ra_deg": ra_sel, "dec_deg": dec_sel, "z": z[m_z], "wfkp": np.ones_like(w_sel)})

        # Evaluate bandpassed maps at galaxy positions and combine frequencies.
        t_num = np.zeros_like(w_sel, dtype=float)
        t_den = np.zeros_like(w_sel, dtype=float)
        for f in freqs:
            t, inb = pxu.eval_map_on_catalog(cmb_f_by_freq[int(f)], cat_bin, pad=0.0, return_mask=True)
            t = np.asarray(t, dtype=float)
            inb = np.asarray(inb, dtype=bool)
            if not np.any(inb):
                continue
            if bool(args.use_ivar):
                iv, inb2 = pxu.eval_map_on_catalog(ivar_by_freq[int(f)], cat_bin, pad=0.0, return_mask=True)
                iv = np.asarray(iv, dtype=float)
                inb &= np.asarray(inb2, dtype=bool)
                wf = np.maximum(iv, 0.0)
            else:
                wf = np.ones_like(t, dtype=float)

            ok = inb & np.isfinite(t) & np.isfinite(wf)
            t_num[ok] += t[ok] * wf[ok]
            t_den[ok] += wf[ok]

        ok = t_den > 0
        if not np.any(ok):
            continue

        t_comb = np.zeros_like(w_sel, dtype=float)
        t_comb[ok] = t_num[ok] / t_den[ok]

        # Include ivar weighting in the map binning weights if requested.
        if bool(args.use_ivar):
            w_map = w_sel * t_den
        else:
            w_map = w_sel

        # Subtract weighted mean within the bin.
        mu = _weighted_mean(t_comb[ok], w_map[ok])
        if np.isfinite(mu):
            t_comb[ok] -= mu

        # Bin to HEALPix.
        pix = radec_to_healpix(ra_sel[ok], dec_sel[ok], nside=int(args.nside), frame=str(args.frame), nest=False)
        npix = int(hp.nside2npix(int(args.nside)))
        wsum = np.bincount(pix, weights=w_map[ok] * t_comb[ok], minlength=npix).astype(float)
        wcnt = np.bincount(pix, weights=w_map[ok], minlength=npix).astype(float)
        theta_map = np.zeros(npix, dtype=float)
        good = wcnt > 0
        theta_map[good] = wsum[good] / wcnt[good]
        mask = np.zeros(npix, dtype=float)
        mask[good] = 1.0

        if bool(args.remove_dipole):
            theta_map = hp.remove_dipole(theta_map, fitval=False, verbose=False)

        tag = f"z{float(z0):.3f}-{float(z1):.3f}"
        theta_path = out_dir / f"theta_act_dr{int(args.act_dr)}_{'-'.join(str(f) for f in freqs)}_sdss_{tag}.fits"
        mask_path = out_dir / f"mask_act_dr{int(args.act_dr)}_{'-'.join(str(f) for f in freqs)}_sdss_{tag}.fits"
        hp.write_map(str(theta_path), theta_map, overwrite=True, dtype=np.float64)
        hp.write_map(str(mask_path), mask, overwrite=True, dtype=np.float64)

        bins.append(
            ThetaBinMeta(
                z_min=float(z0),
                z_max=float(z1),
                n_gal_used=int(np.sum(ok)),
                theta_fits=str(theta_path),
                mask_fits=str(mask_path),
            )
        )
        print(f"[build_theta_tomo] wrote {theta_path}  ngal={int(np.sum(ok))}", flush=True)

    if not bins:
        raise RuntimeError("No theta bins produced; check z cuts, ACT footprint, and inputs.")

    manifest = Manifest(
        created_utc=_utc_stamp(),
        act_dr=int(args.act_dr),
        act_freqs_ghz=[int(f) for f in freqs],
        act_night=bool(args.act_night),
        sdss_surveys=surveys,
        sdss_dr=int(args.sdss_dr),
        frame=str(args.frame),
        nside=int(args.nside),
        lmin=int(args.lmin),
        lmax=int(args.lmax),
        weights=str(args.weights),
        use_ivar=bool(args.use_ivar),
        remove_dipole=bool(args.remove_dipole),
        kszx_data_dir=str(kszx_data_dir),
        z_edges=[float(x) for x in z_edges],
        bins=bins,
    )
    manifest_path = out_dir / "theta_tomo_manifest.json"
    manifest_path.write_text(json_dumps(asdict(manifest)))

    print(str(manifest_path))
    return 0


def json_dumps(d) -> str:
    import json

    return json.dumps(d, indent=2, sort_keys=True)


if __name__ == "__main__":
    raise SystemExit(main())

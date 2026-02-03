from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _run(cmd: list[str]) -> None:
    print(f"[kszx_prep] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextlib.contextmanager
def _file_lock(path: Path):
    """POSIX lock to prevent concurrent runs corrupting shared .part files."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _gzip_test(path_gz: Path) -> None:
    """Validate gz CRC before attempting to decompress (fast fail if corrupted)."""
    _run(["gzip", "-t", str(path_gz)])


def _ensure_wget(url: str, dst: Path, *, wait_s: int = 1, limit_rate: str | None = None) -> None:
    """Resumable download helper with safety locking.

    - Uses system wget with `-c` resume by default.
    - Uses aria2c (multi-connection) automatically for SDSS dr*.sdss.org downloads if available.
    - Uses a temporary file + atomic rename to avoid half-written targets.
    - Uses a lockfile so concurrent runs can't write into the same .part file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return

    tmp = Path(str(dst) + ".part")
    lock = Path(str(tmp) + ".lock")
    with _file_lock(lock):
        # Another process might have finished it while we were waiting for the lock.
        if dst.exists() and dst.stat().st_size > 0:
            return

        use_aria2 = ("sdss.org/" in url) and (shutil.which("aria2c") is not None)
        if use_aria2:
            # Keep it modest to avoid hammering endpoints; tune up only if needed.
            conns = 4
            cmd = [
                "aria2c",
                "--continue=true",
                "--auto-file-renaming=false",
                "--allow-overwrite=true",
                "--file-allocation=none",
                "--max-connection-per-server",
                str(conns),
                "--split",
                str(conns),
                "--min-split-size",
                "20M",
                "-d",
                str(tmp.parent),
                "-o",
                tmp.name,
                url,
            ]
            if limit_rate:
                cmd.extend(["--max-overall-download-limit", str(limit_rate)])
            _run(cmd)
        else:
            cmd = ["wget", "-c", "--wait", str(int(wait_s)), "--random-wait", "-O", str(tmp), url]
            if limit_rate:
                cmd.insert(2, f"--limit-rate={limit_rate}")
            _run(cmd)

        if not tmp.exists() or tmp.stat().st_size == 0:
            raise RuntimeError(f"Download failed: {url} -> {tmp}")
        tmp.replace(dst)

    if dst.suffix == ".gz":
        _gzip_test(dst)


def _ensure_gunzip(src_gz: Path, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
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


def _sdss_dr_str(dr: int) -> str:
    if dr == 11:
        return "DR11v1"
    if dr == 12:
        return "DR12v5"
    raise ValueError(f"Unsupported SDSS dr={dr} (supported: 11, 12).")


def _ensure_sdss_galaxy_catalog(*, kszx_data_dir: Path, survey: str, dr: int, limit_rate: str | None = None) -> Path:
    """Download SDSS/BOSS galaxy FITS to the exact path kszx expects."""
    dr_str = _sdss_dr_str(int(dr))
    base = kszx_data_dir / "sdss" / dr_str
    fits = base / f"galaxy_{dr_str}_{survey}.fits"
    gz = Path(str(fits) + ".gz")  # -> .fits.gz
    # Prefer the per-release host (often much faster than data.sdss.org from this machine).
    # Example: https://dr12.sdss.org/sas/dr12/boss/lss/<file>.gz
    url = f"https://dr{int(dr)}.sdss.org/sas/dr{int(dr)}/boss/lss/{fits.name}.gz"
    _ensure_wget(url, gz, limit_rate=limit_rate)
    _ensure_gunzip(gz, fits)
    return fits


def _ensure_sdss_random_catalogs(*, kszx_data_dir: Path, survey: str, dr: int, limit_rate: str | None = None) -> list[Path]:
    """Download SDSS/BOSS random{0,1} FITS to the exact paths kszx expects."""
    dr_str = _sdss_dr_str(int(dr))
    base = kszx_data_dir / "sdss" / dr_str
    out: list[Path] = []
    for n in (0, 1):
        fits = base / f"random{n}_{dr_str}_{survey}.fits"
        gz = Path(str(fits) + ".gz")
        # Prefer the per-release host (often much faster than data.sdss.org from this machine).
        url = f"https://dr{int(dr)}.sdss.org/sas/dr{int(dr)}/boss/lss/{fits.name}.gz"
        _ensure_wget(url, gz, limit_rate=limit_rate)
        _ensure_gunzip(gz, fits)
        out.append(fits)
    return out


def _ensure_planck_hfi_galmask(*, kszx_data_dir: Path, apod_deg: int, limit_rate: str | None = None) -> Path:
    """Download Planck HFI galmask FITS (release 2) to the exact path kszx expects."""
    apod_deg = int(apod_deg)
    if apod_deg not in (0, 2, 5):
        raise ValueError("Planck galmask apodization must be one of {0,2,5} deg.")
    rel = f"release_2/ancillary-data/masks/HFI_Mask_GalPlane-apo{apod_deg}_2048_R2.00.fits"
    dst = kszx_data_dir / "planck" / rel
    url = f"https://irsa.ipac.caltech.edu/data/Planck/{rel}"
    _ensure_wget(url, dst, limit_rate=limit_rate)
    return dst


def _ensure_act_dr6_map_and_ivar(
    *, kszx_data_dir: Path, freq: int, act_night: bool, limit_rate: str | None = None
) -> tuple[Path, Path]:
    """Download ACT DR6.02 srcfree map + ivar to the exact paths kszx expects."""
    act_time = "night" if bool(act_night) else "daynight"
    url_base = "https://lambda.gsfc.nasa.gov/data/act/"
    cmb_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(freq):03d}_map_srcfree.fits"
    ivar_rel = f"maps/published/act-planck_dr6.02_coadd_AA_{act_time}_f{int(freq):03d}_ivar.fits"
    dst_cmb = kszx_data_dir / "act" / "dr6.02" / cmb_rel
    dst_ivar = kszx_data_dir / "act" / "dr6.02" / ivar_rel
    _ensure_wget(url_base + cmb_rel, dst_cmb, limit_rate=limit_rate)
    _ensure_wget(url_base + ivar_rel, dst_ivar, limit_rate=limit_rate)
    return dst_cmb, dst_ivar


@dataclass(frozen=True)
class PrepManifest:
    created_utc: str
    kszx_data_dir: str
    input_dir: str
    act_dr: int
    act_night: bool
    freqs_ghz: list[int]
    ksz_lmin: int
    ksz_lmax: int
    act_rms_threshold_ukarcmin: float
    planck_galmask_sky_pct: int
    planck_galmask_apod_deg: int
    sdss_survey: str
    sdss_dr: int
    zmin: float
    zmax: float
    zeff: float
    pixsize_mpc: float
    rpad_mpc: float
    weights_mode: str
    params_yml: dict[str, object]
    file_sha256: dict[str, str]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare a *real* kszx KszPipe input directory (Pre-KszPipe stage):\n"
            "  - downloads ACT + SDSS + Planck mask inputs (serial, resumable)\n"
            "  - computes halo-model X_l^{ge}, X_l^{gg}\n"
            "  - builds W_cmb(theta) and F_l filters (CmbClFitter)\n"
            "  - writes galaxies.h5, randoms.h5 (with tcmb_90/150, bv_90/150, weight_gal/vr)\n"
            "  - writes bounding_box.pkl and params.yml\n"
            "\n"
            "This is the professional, end-to-end input prep needed before running:\n"
            "  python -m kszx kszpipe_run -p N <input_dir> <output_dir>\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--out", default=None, help="Output input_dir (default: data/processed/kszx/kszpipe_input_<stamp>/).")
    ap.add_argument(
        "--download-limit-rate",
        default=None,
        help="Optional wget --limit-rate value (e.g. 10m) for gentle downloads (default: unlimited).",
    )

    ap.add_argument("--act-dr", type=int, default=6, help="ACT data release (default 6).")
    ap.add_argument("--act-night", action="store_true", help="Use ACT night coadd (default daynight).")
    ap.add_argument("--freqs", default="90,150", help="Comma-separated ACT freqs in GHz (default 90,150).")
    ap.add_argument("--act-rms-threshold", type=float, default=70.0, help="wcmb RMS threshold in uK-arcmin (default 70).")

    ap.add_argument("--planck-galmask-sky-pct", type=int, default=70, help="Planck HFI GAL0xx sky fraction (default 70).")
    ap.add_argument("--planck-galmask-apod-deg", type=int, default=0, help="Planck HFI galmask apodization deg (default 0).")

    ap.add_argument("--sdss-survey", default="CMASS_North", help="SDSS/BOSS survey name (default CMASS_North).")
    ap.add_argument("--sdss-dr", type=int, default=12, help="SDSS data release (default 12).")
    ap.add_argument("--zmin", type=float, default=0.43, help="Redshift min (default 0.43).")
    ap.add_argument("--zmax", type=float, default=0.70, help="Redshift max (default 0.70).")
    ap.add_argument("--zeff", type=float, default=0.57, help="Effective redshift (default 0.57).")

    ap.add_argument("--pixsize-mpc", type=float, default=10.0, help="Bounding box pixel size in Mpc (default 10).")
    ap.add_argument("--rpad-mpc", type=float, default=500.0, help="Bounding box padding in Mpc (default 500).")

    ap.add_argument("--weights", choices=["wfkp", "fkp_sys"], default="fkp_sys", help="Per-object weights mode (default fkp_sys).")

    # kSZ filter params (match kszx notebooks by default).
    ap.add_argument("--ksz-lmin", type=int, default=1500, help="kSZ filter lmin (default 1500).")
    ap.add_argument("--ksz-lmax", type=int, default=8000, help="kSZ filter lmax (default 8000).")

    # Halo-model filter params (used in X_l^{ge}, X_l^{gg} and b_v normalization).
    ap.add_argument("--halo-ngal", type=float, default=1e-4, help="Halo-model galaxy number density (default 1e-4 Mpc^-3).")
    ap.add_argument("--halo-profile", default="AGN", help="Battaglia electron profile family (default AGN).")

    # KszPipe params.yml (controls run-time size of the *next* stage).
    ap.add_argument("--nsurr", type=int, default=4, help="Number of surrogate sims (default 4; increase for real runs).")
    ap.add_argument("--surr-bg", type=float, default=2.1, help="Galaxy bias used in surrogate sims (default 2.1).")
    ap.add_argument("--nzbins-gal", type=int, default=25, help="Mean-subtraction bins for galaxy field (default 25).")
    ap.add_argument("--nzbins-vr", type=int, default=25, help="Mean-subtraction bins for vr field (default 25).")
    ap.add_argument("--nkbins", type=int, default=25, help="Number of k bins (default 25).")
    ap.add_argument("--kmax", type=float, default=0.05, help="Max k in h/Mpc (?) for P(k) bins (default 0.05).")

    args = ap.parse_args()

    print("[kszx_prep] starting", flush=True)

    freqs = [int(x.strip()) for x in str(args.freqs).split(",") if x.strip()]
    if not freqs:
        raise ValueError("No frequencies parsed from --freqs.")
    if set(freqs) != {90, 150}:
        raise ValueError(
            f"This pipeline currently targets the kszx KszPipe defaults (two freqs 90 and 150). "
            f"Got freqs={freqs}."
        )
    if int(args.act_dr) != 6:
        raise ValueError("This script currently supports ACT DR6 only (act-dr=6).")

    kszx_data_dir = Path(args.kszx_data_dir).resolve()
    kszx_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KSZX_DATA_DIR"] = str(kszx_data_dir)

    out_dir = Path(args.out) if args.out else Path("data/processed/kszx") / f"kszpipe_input_{_utc_stamp()}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intermediate").mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # 0) Ensure external data exist (serial, resumable).
    # ---------------------------------------------------------------------
    limit_rate = str(args.download_limit_rate) if args.download_limit_rate else None
    for f in freqs:
        _ensure_act_dr6_map_and_ivar(
            kszx_data_dir=kszx_data_dir,
            freq=f,
            act_night=bool(args.act_night),
            limit_rate=limit_rate,
        )
    _ensure_sdss_galaxy_catalog(
        kszx_data_dir=kszx_data_dir,
        survey=str(args.sdss_survey),
        dr=int(args.sdss_dr),
        limit_rate=limit_rate,
    )
    _ensure_sdss_random_catalogs(
        kszx_data_dir=kszx_data_dir,
        survey=str(args.sdss_survey),
        dr=int(args.sdss_dr),
        limit_rate=limit_rate,
    )
    _ensure_planck_hfi_galmask(
        kszx_data_dir=kszx_data_dir,
        apod_deg=int(args.planck_galmask_apod_deg),
        limit_rate=limit_rate,
    )

    # Defer heavyweight imports until after we set KSZX_DATA_DIR.
    import kszx
    import pixell
    import pixell.reproject
    import scipy.interpolate
    import yaml
    import hmvec as hm

    # ---------------------------------------------------------------------
    # 1) Halo model -> X_l^{ge}, X_l^{gg}
    # ---------------------------------------------------------------------
    lmin = int(args.ksz_lmin)
    lmax = int(args.ksz_lmax)
    zeff = float(args.zeff)
    ngal = float(args.halo_ngal)

    # Match kszx notebook 01 defaults.
    ks = np.geomspace(1e-5, 100.0, 1000)
    ms = np.geomspace(2e10, 1e17, 40)

    print(f"[kszx_prep] halo model: zeff={zeff}  ngal={ngal}  lmax={lmax}  profile={args.halo_profile}", flush=True)
    hcos = hm.HaloModel([zeff], ks, ms=ms)
    hcos.add_battaglia_profile("electron", family=str(args.halo_profile))
    hcos.add_hod(name="g", ngal=np.asarray([ngal]))
    hpge = hcos.get_power_1halo("g", "electron") + hcos.get_power_2halo("g", "electron")
    hpggtot = hcos.get_power_1halo("g", "g") + hcos.get_power_2halo("g", "g") + 1.0 / ngal

    hpge = hpge[0]
    hpggtot = hpggtot[0]
    chieff = float(hcos.comoving_radial_distance(zeff))

    interp_logk_pge = scipy.interpolate.InterpolatedUnivariateSpline(np.log(ks), hpge)
    interp_logk_pgg = scipy.interpolate.InterpolatedUnivariateSpline(np.log(ks), hpggtot)

    ell = np.arange(lmax + 1, dtype=float)
    ell[0] = 0.1
    logk = np.log(ell / chieff)
    xl_ge = interp_logk_pge(logk)
    xl_gg = interp_logk_pgg(logk)

    xl_path = out_dir / "intermediate" / "xl_ge_gg.txt"
    np.savetxt(
        xl_path,
        np.transpose([np.arange(lmax + 1), xl_ge, xl_gg]),
        header=(
            "Col 0: l\n"
            "Col 1: X_l^{ge} (Mpc^3)\n"
            "Col 2: X_l^{gg} (Mpc^3)\n"
            "We define X_l^{ij} = P_{ij}(k)_{k=l/chi}\n"
        ),
    )

    # ---------------------------------------------------------------------
    # 2) Build pixell_galmask.fits in ACT map geometry
    # ---------------------------------------------------------------------
    freq_for_geom = freqs[-1]  # 150 by default (common choice).
    act_cmb = kszx.act.read_cmb(freq=freq_for_geom, dr=int(args.act_dr), night=bool(args.act_night), download=False)
    healpix_mask = kszx.planck.read_hfi_galmask(
        sky_percentage=int(args.planck_galmask_sky_pct),
        apodization=int(args.planck_galmask_apod_deg),
        download=False,
    )
    pixell_mask = pixell.reproject.healpix2map(
        healpix_mask,
        act_cmb.shape,
        act_cmb.wcs,
        rot="gal,equ",
        method="spline",
        order=0,
    )
    pixell_galmask_path = out_dir / "intermediate" / "pixell_galmask.fits"
    kszx.pixell_utils.write_map(str(pixell_galmask_path), pixell_mask)

    # ---------------------------------------------------------------------
    # 3) For each frequency: build wcmb_{freq}.fits and fl_{freq}.txt
    # ---------------------------------------------------------------------
    cosmo = kszx.Cosmology("planck18+bao", lmax=int(lmax + 1000))
    wcmb_paths: dict[int, Path] = {}
    fl_paths: dict[int, Path] = {}
    tcmb_maps: dict[int, "pixell.enmap.ndmap"] = {}
    bv_maps: dict[int, "pixell.enmap.ndmap"] = {}

    for freq in freqs:
        print(f"[kszx_prep] freq={freq}GHz: loading ACT maps", flush=True)
        cmb = kszx.act.read_cmb(freq=freq, dr=int(args.act_dr), night=bool(args.act_night), download=False)
        ivar = kszx.act.read_ivar(freq=freq, dr=int(args.act_dr), night=bool(args.act_night), download=False)
        bl_full = kszx.act.read_beam(freq=freq, dr=int(args.act_dr), night=bool(args.act_night), download=False)

        wcmb = pixell_mask.copy()
        wcmb *= kszx.pixell_utils.uK_arcmin_from_ivar(ivar) < float(args.act_rms_threshold)

        wcmb_path = out_dir / "intermediate" / f"wcmb_{freq}.fits"
        kszx.pixell_utils.write_map(str(wcmb_path), wcmb)
        wcmb_paths[freq] = wcmb_path

        print(f"[kszx_prep] freq={freq}GHz: fitting C_l^tot and computing F_l", flush=True)
        cl_fitter = kszx.CmbClFitter(
            cosmo=cosmo,
            cmb_map=cmb,
            weight_map=wcmb,
            bl=bl_full,
            lmin=lmin,
            lmax=lmax,
            ivar=ivar,
        )
        cl_tot = np.asarray(cl_fitter.cl_tot, dtype=float)
        bl = np.asarray(bl_full[: lmax + 1], dtype=float)
        fl = bl * xl_ge / xl_gg / cl_tot[: lmax + 1]
        fl[:lmin] = 0.0

        fl_path = out_dir / "intermediate" / f"fl_{freq}.txt"
        np.savetxt(
            fl_path,
            np.transpose([np.arange(lmax + 1), fl, cl_tot[: lmax + 1]]),
            header=(
                "Col 0: l\n"
                "Col 1: F_l (CMB l-weighting in kSZ quadratic estimator)\n"
                "Col 2: C_l (fit for CMB power spectrum, beam-convolved, includes noise)\n"
            ),
        )
        fl_paths[freq] = fl_path

        # Filtered temperature map: tcmb(theta) = Alm^{-1}[ F_l * Alm[ Wcmb * T(theta) ] ].
        tcmb = cmb.copy()
        tcmb *= wcmb
        alm = kszx.pixell_utils.map2alm(tcmb, lmax)
        alm = pixell.curvedsky.almxfl(alm, fl)
        tcmb_f = kszx.pixell_utils.alm2map(alm, tcmb.shape, tcmb.wcs)
        tcmb_maps[freq] = tcmb_f

        # Bias map b_v(theta) ≈ B_v(chi_eff) * Wcmb(theta).
        Bv = float(np.sum((2 * np.arange(lmax + 1) + 1) / (4 * np.pi) * bl * fl * xl_ge))
        Bv *= float(cosmo.K(z=zeff) / (cosmo.chi(z=zeff) ** 2))
        bv = Bv * wcmb
        bv_maps[freq] = bv

    # ---------------------------------------------------------------------
    # 4) Read SDSS catalogs, apply z cut, add required KszPipe columns
    # ---------------------------------------------------------------------
    print(f"[kszx_prep] reading SDSS catalogs survey={args.sdss_survey} dr={args.sdss_dr}", flush=True)
    gcat = kszx.sdss.read_galaxies(str(args.sdss_survey), dr=int(args.sdss_dr), download=False)
    rcat = kszx.sdss.read_randoms(str(args.sdss_survey), dr=int(args.sdss_dr), download=False)
    gcat.apply_redshift_cut(float(args.zmin), float(args.zmax))
    rcat.apply_redshift_cut(float(args.zmin), float(args.zmax))

    # Per-object weights.
    if str(args.weights) == "wfkp":
        g_w = np.asarray(gcat.wfkp, dtype=float)
        r_w = np.asarray(rcat.wfkp, dtype=float)
    else:
        g_w = np.asarray((gcat.wzf + gcat.wcp - 1.0) * gcat.wsys * gcat.wfkp, dtype=float)
        r_w = np.asarray(rcat.wfkp, dtype=float)

    gcat.add_column("weight_gal", g_w)
    gcat.add_column("weight_vr", g_w)
    rcat.add_column("weight_gal", r_w)
    rcat.add_column("weight_vr", r_w)

    # Random catalogs: required by KszPipe (spectroscopic -> zobs=ztrue=z).
    rcat.add_column("zobs", np.asarray(rcat.z, dtype=float))
    rcat.add_column("ztrue", np.asarray(rcat.z, dtype=float))

    def _add_maps_to_catalog(cat: "kszx.Catalog") -> None:
        for freq in freqs:
            t = kszx.pixell_utils.eval_map_on_catalog(tcmb_maps[freq], cat, pad=0.0)
            b = kszx.pixell_utils.eval_map_on_catalog(bv_maps[freq], cat, pad=0.0)
            cat.add_column(f"tcmb_{freq}", np.asarray(t, dtype=float))
            cat.add_column(f"bv_{freq}", np.asarray(b, dtype=float))

    print("[kszx_prep] sampling (tcmb,bv) on galaxy catalog", flush=True)
    _add_maps_to_catalog(gcat)
    print("[kszx_prep] sampling (tcmb,bv) on random catalog", flush=True)
    _add_maps_to_catalog(rcat)

    # ---------------------------------------------------------------------
    # 5) Write KszPipe input_dir files
    # ---------------------------------------------------------------------
    params = {
        "version": 1,
        "nsurr": int(args.nsurr),
        "surr_bg": float(args.surr_bg),
        "nzbins_gal": int(args.nzbins_gal),
        "nzbins_vr": int(args.nzbins_vr),
        "nkbins": int(args.nkbins),
        "kmax": float(args.kmax),
    }
    (out_dir / "params.yml").write_text(yaml.safe_dump(params, sort_keys=True))

    gcat_path = out_dir / "galaxies.h5"
    rcat_path = out_dir / "randoms.h5"
    gcat.write_h5(str(gcat_path))
    rcat.write_h5(str(rcat_path))

    # Bounding box uses randoms positions with zobs.
    box = kszx.BoundingBox(rcat.get_xyz(cosmo, "zobs"), float(args.pixsize_mpc), float(args.rpad_mpc))
    bb_path = out_dir / "bounding_box.pkl"
    kszx.io_utils.write_pickle(str(bb_path), box)

    # ---------------------------------------------------------------------
    # 6) Manifest (lightweight reproducibility metadata)
    # ---------------------------------------------------------------------
    file_sha256: dict[str, str] = {}
    for p in (out_dir / "params.yml", gcat_path, rcat_path, bb_path):
        file_sha256[str(p.relative_to(out_dir))] = _sha256(p)

    man = PrepManifest(
        created_utc=_utc_stamp(),
        kszx_data_dir=str(kszx_data_dir),
        input_dir=str(out_dir),
        act_dr=int(args.act_dr),
        act_night=bool(args.act_night),
        freqs_ghz=[int(x) for x in freqs],
        ksz_lmin=int(args.ksz_lmin),
        ksz_lmax=int(args.ksz_lmax),
        act_rms_threshold_ukarcmin=float(args.act_rms_threshold),
        planck_galmask_sky_pct=int(args.planck_galmask_sky_pct),
        planck_galmask_apod_deg=int(args.planck_galmask_apod_deg),
        sdss_survey=str(args.sdss_survey),
        sdss_dr=int(args.sdss_dr),
        zmin=float(args.zmin),
        zmax=float(args.zmax),
        zeff=float(args.zeff),
        pixsize_mpc=float(args.pixsize_mpc),
        rpad_mpc=float(args.rpad_mpc),
        weights_mode=str(args.weights),
        params_yml=params,
        file_sha256=file_sha256,
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(man), indent=2, sort_keys=True) + "\n")

    print(f"[kszx_prep] wrote KszPipe input_dir: {out_dir}", flush=True)
    print(f"[kszx_prep] next: .venv/bin/python -m kszx kszpipe_run -p <NPROC> {out_dir} <OUTPUT_DIR>", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

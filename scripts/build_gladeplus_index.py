from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import healpy as hp
import numpy as np
import pandas as pd


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _iter_gladeplus_chunks(path: Path, *, chunksize: int, usecols: list[int]) -> pd.io.parsers.TextFileReader:
    # GLADE+ is a whitespace-delimited ASCII file with "null" placeholders.
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        usecols=usecols,
        na_values=["null", "NaN", "nan"],
        dtype="float64",
        chunksize=int(chunksize),
        engine="c",
    )


def _compute_hp(ra_deg: np.ndarray, dec_deg: np.ndarray, *, nside: int, nest: bool) -> np.ndarray:
    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg)
    return hp.ang2pix(nside, theta, phi, nest=nest).astype(np.int64, copy=False)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a HEALPix-bucketed index for the GLADE+ galaxy catalog.")
    ap.add_argument("--gladeplus", default="data/cache/galaxies/gladeplus/GLADE+.txt", help="Path to GLADE+.txt")
    ap.add_argument("--out", default=None, help="Output dir (default: data/processed/galaxies/gladeplus/index_nside<N>/)")
    ap.add_argument("--nside", type=int, default=128, help="HEALPix nside for the galaxy index (default: 128).")
    ap.add_argument("--nest", action="store_true", help="Use NESTED ordering (recommended).")
    ap.add_argument("--chunksize", type=int, default=750_000, help="CSV chunksize (lines) (default: 750k).")
    ap.add_argument("--z-max", type=float, default=None, help="Optional max z cut (e.g. 0.3).")
    ap.add_argument(
        "--weight-mode",
        choices=["ones", "luminosity_b"],
        default="ones",
        help=(
            "Per-galaxy prior weight stored in w.npy.\n"
            "  - ones: w=1 for all galaxies\n"
            "  - luminosity_b: w=L_B from absolute B-band magnitude (drops rows with missing/invalid M_B)\n"
        ),
    )
    ap.add_argument("--keep-zerr", action="store_true", help="Also store z_err from column 29 if present.")
    args = ap.parse_args()

    src = Path(args.gladeplus)
    if not src.exists():
        raise FileNotFoundError(f"Missing GLADE+ file: {src}. Run scripts/fetch_gladeplus.py first.")

    nside = int(args.nside)
    nest = bool(args.nest)
    npix = hp.nside2npix(nside)

    # Column map (1-based) from quick inspection:
    #  - RA  : col  9
    #  - DEC : col 10
    #  - M_B : col 14  (absolute B-band magnitude; used for luminosity weights)
    #  - z   : col 28
    #  - zerr: col 29
    # These are 0-based indices:
    col_ra = 8
    col_dec = 9
    col_m_b = 13
    col_z = 27
    col_zerr = 28
    # NOTE: pandas may reorder integer usecols; enforce sorted columns and compute positions.
    want = [col_ra, col_dec, col_z]
    if str(args.weight_mode) == "luminosity_b":
        want.append(col_m_b)
    if args.keep_zerr:
        want.append(col_zerr)
    usecols = sorted(set(want))
    pos_ra = usecols.index(col_ra)
    pos_dec = usecols.index(col_dec)
    pos_z = usecols.index(col_z)
    pos_m_b = usecols.index(col_m_b) if (str(args.weight_mode) == "luminosity_b") else None
    pos_zerr = usecols.index(col_zerr) if args.keep_zerr else None

    out_dir = Path(args.out) if args.out else Path("data") / "processed" / "galaxies" / "gladeplus" / f"index_nside{nside}"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "meta.json"
    marker = out_dir / ".built.ok"
    if marker.exists():
        print(f"[glade_index] already built: {out_dir}")
        return 0

    print(f"[glade_index] PASS 1/2: counting pixels (nside={nside}, nest={nest})", flush=True)
    counts = np.zeros(npix, dtype=np.int64)
    n_keep = 0
    n_total = 0

    for chunk in _iter_gladeplus_chunks(src, chunksize=int(args.chunksize), usecols=usecols):
        n_total += int(chunk.shape[0])
        ra = chunk.iloc[:, pos_ra].to_numpy(dtype=float, copy=False)
        dec = chunk.iloc[:, pos_dec].to_numpy(dtype=float, copy=False)
        z = chunk.iloc[:, pos_z].to_numpy(dtype=float, copy=False)
        m_b = chunk.iloc[:, pos_m_b].to_numpy(dtype=float, copy=False) if pos_m_b is not None else None

        m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0.0)
        if m_b is not None:
            m &= np.isfinite(m_b)
        if args.z_max is not None:
            m &= z <= float(args.z_max)
        if not np.any(m):
            continue

        hpix = _compute_hp(ra[m], dec[m], nside=nside, nest=nest)
        counts += np.bincount(hpix, minlength=npix).astype(np.int64, copy=False)
        n_keep += int(hpix.size)

        if n_total % (5_000_000) < chunk.shape[0]:
            print(f"[glade_index] counted {n_total:,} lines, kept {n_keep:,}", flush=True)

    if n_keep == 0:
        raise RuntimeError("No galaxies kept after filtering; check column mapping / z-max.")

    offsets = np.zeros(npix + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    assert int(offsets[-1]) == int(n_keep)

    # Allocate output arrays as .npy memmaps for efficient random read.
    ra_out = np.lib.format.open_memmap(out_dir / "ra_deg.npy", mode="w+", dtype="float32", shape=(n_keep,))
    dec_out = np.lib.format.open_memmap(out_dir / "dec_deg.npy", mode="w+", dtype="float32", shape=(n_keep,))
    z_out = np.lib.format.open_memmap(out_dir / "z.npy", mode="w+", dtype="float32", shape=(n_keep,))
    w_out = np.lib.format.open_memmap(out_dir / "w.npy", mode="w+", dtype="float32", shape=(n_keep,))
    hp_out = np.lib.format.open_memmap(out_dir / "hpix.npy", mode="w+", dtype="int32", shape=(n_keep,))
    zerr_out = None
    if args.keep_zerr:
        zerr_out = np.lib.format.open_memmap(out_dir / "zerr.npy", mode="w+", dtype="float32", shape=(n_keep,))

    np.save(out_dir / "hpix_offsets.npy", offsets)

    print(f"[glade_index] PASS 2/2: filling arrays (N={n_keep:,})", flush=True)
    write_ptr = offsets[:-1].copy()  # current write head for each pixel
    n_total = 0
    n_written = 0
    for chunk in _iter_gladeplus_chunks(src, chunksize=int(args.chunksize), usecols=usecols):
        n_total += int(chunk.shape[0])
        ra = chunk.iloc[:, pos_ra].to_numpy(dtype=float, copy=False)
        dec = chunk.iloc[:, pos_dec].to_numpy(dtype=float, copy=False)
        z = chunk.iloc[:, pos_z].to_numpy(dtype=float, copy=False)
        m_b = chunk.iloc[:, pos_m_b].to_numpy(dtype=float, copy=False) if pos_m_b is not None else None
        zerr = chunk.iloc[:, pos_zerr].to_numpy(dtype=float, copy=False) if pos_zerr is not None else None

        m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & (z > 0.0)
        if m_b is not None:
            m &= np.isfinite(m_b)
        if args.z_max is not None:
            m &= z <= float(args.z_max)
        if not np.any(m):
            continue

        ra = ra[m]
        dec = dec[m]
        z = z[m]
        if m_b is not None:
            m_b = m_b[m]
        if zerr is not None:
            zerr = zerr[m]

        hpix = _compute_hp(ra, dec, nside=nside, nest=nest).astype(np.int64, copy=False)
        order = np.argsort(hpix, kind="mergesort")
        hpix_s = hpix[order]
        ra_s = ra[order].astype(np.float32, copy=False)
        dec_s = dec[order].astype(np.float32, copy=False)
        z_s = z[order].astype(np.float32, copy=False)
        m_b_s = m_b[order].astype(np.float32, copy=False) if m_b is not None else None
        zerr_s = zerr[order].astype(np.float32, copy=False) if zerr is not None else None

        uniq, idx0, cnt = np.unique(hpix_s, return_index=True, return_counts=True)
        for p, start, n in zip(uniq.tolist(), idx0.tolist(), cnt.tolist(), strict=True):
            p = int(p)
            n = int(n)
            out0 = int(write_ptr[p])
            out1 = out0 + n
            ra_out[out0:out1] = ra_s[start : start + n]
            dec_out[out0:out1] = dec_s[start : start + n]
            z_out[out0:out1] = z_s[start : start + n]
            if m_b_s is None:
                w_out[out0:out1] = 1.0
            else:
                # B-band luminosity weights in solar units (up to an overall constant):
                #   L_B ∝ 10^{-0.4 (M_B - M_sun,B)}.  M_sun,B ~= 5.48.
                M_sun_B = 5.48
                Mb = m_b_s[start : start + n].astype(np.float64, copy=False)
                w_out[out0:out1] = (10.0 ** (-0.4 * (Mb - M_sun_B))).astype(np.float32, copy=False)
            hp_out[out0:out1] = np.int32(p)
            if zerr_out is not None and zerr_s is not None:
                zerr_out[out0:out1] = zerr_s[start : start + n]
            write_ptr[p] = out1
            n_written += n

        if n_total % (5_000_000) < chunk.shape[0]:
            print(f"[glade_index] processed {n_total:,} lines, written {n_written:,} / {n_keep:,}", flush=True)

    # Sanity: ensure all write pointers hit the expected offsets.
    if not np.all(write_ptr == offsets[1:]):
        bad = int(np.sum(write_ptr != offsets[1:]))
        raise RuntimeError(f"Index build incomplete: {bad} pixels have mismatched write counts.")

    meta = {
        "source": str(src),
        "nside": nside,
        "nest": nest,
        "npix": npix,
        "n_total_lines": n_total,
        "n_kept": n_keep,
        "z_max": float(args.z_max) if args.z_max is not None else None,
        "weight_mode": str(args.weight_mode),
        "w_note": "w=1" if str(args.weight_mode) == "ones" else "w=L_B from abs B magnitude (col 14), dropping missing M_B",
        "columns_1based": {
            "ra_deg": 9,
            "dec_deg": 10,
            "M_B_abs": 14 if str(args.weight_mode) == "luminosity_b" else None,
            "z": 28,
            "zerr": 29 if args.keep_zerr else None,
        },
        "timestamp_utc": _utc_stamp(),
        "note": "This is a HEALPix-bucketed index for fast sky-region selection; raw catalog stays in data/cache/.",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    marker.write_text(_utc_stamp() + "\n")
    print(f"[glade_index] done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

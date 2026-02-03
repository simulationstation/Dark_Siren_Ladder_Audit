from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str) -> list[str]:
    out = [p.strip() for p in (s or "").split(",") if p.strip()]
    if not out:
        raise ValueError("Expected a comma-separated list.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Extract a thin CSV (ra,dec,z,weight) from SDSS/BOSS galaxy catalogs.\n"
            "\n"
            "We use kszx.sdss.read_galaxies() so the column parsing matches the rest of our kSZ plumbing.\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--sdss-surveys", default="CMASSLOWZTOT_North,CMASSLOWZTOT_South", help="Comma-separated SDSS surveys.")
    ap.add_argument("--sdss-dr", type=int, default=12, help="SDSS data release (default 12).")
    ap.add_argument("--zmin", type=float, default=0.20, help="Min redshift (default 0.20).")
    ap.add_argument("--zmax", type=float, default=0.70, help="Max redshift (default 0.70).")
    ap.add_argument(
        "--weights",
        choices=["wfkp", "fkp_sys"],
        default="fkp_sys",
        help="Galaxy weight convention (default fkp_sys = (wzf+wcp-1)*wsys*wfkp).",
    )
    ap.add_argument("--max-galaxies", type=int, default=None, help="Optional subsample size for a fast smoke run.")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed used when --max-galaxies is set.")
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV (default: data/processed/void_prism/sdss_galaxies_<stamp>.csv).",
    )
    args = ap.parse_args()

    kszx_data_dir = Path(args.kszx_data_dir).resolve()
    kszx_data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KSZX_DATA_DIR"] = str(kszx_data_dir)

    # Defer heavyweight imports until after we set KSZX_DATA_DIR.
    from kszx import sdss as ksdss  # type: ignore
    from kszx.Catalog import Catalog  # type: ignore

    surveys = _parse_csv_list(str(args.sdss_surveys))
    cats: list[Catalog] = []
    for s in surveys:
        cat = ksdss.read_galaxies(s, dr=int(args.sdss_dr), download=True)
        cat.apply_redshift_cut(float(args.zmin), float(args.zmax))
        cats.append(cat)
    gcat = Catalog.concatenate(cats, name=" + ".join(surveys), destructive=True) if len(cats) > 1 else cats[0]

    ra = np.asarray(gcat.ra_deg, dtype=float)
    dec = np.asarray(gcat.dec_deg, dtype=float)
    z = np.asarray(gcat.z, dtype=float)
    if args.weights == "wfkp":
        w = np.asarray(gcat.wfkp, dtype=float)
    else:
        w = np.asarray((gcat.wzf + gcat.wcp - 1.0) * gcat.wsys * gcat.wfkp, dtype=float)

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & np.isfinite(w) & (w > 0)
    ra, dec, z, w = ra[m], dec[m], z[m], w[m]
    if ra.size == 0:
        raise RuntimeError("No valid galaxies after cuts/weights.")

    if args.max_galaxies is not None and 0 < int(args.max_galaxies) < ra.size:
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(ra.size, size=int(args.max_galaxies), replace=False)
        ra, dec, z, w = ra[idx], dec[idx], z[idx], w[idx]

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"sdss_galaxies_{_utc_stamp()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "ra,dec,z,weight"
    arr = np.column_stack([ra, dec, z, w]).astype(np.float64)
    np.savetxt(out, arr, delimiter=",", header=header, comments="")

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


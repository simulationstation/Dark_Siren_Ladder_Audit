from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np


def _read_column_from_dat_gz(path: Path, col_name: str) -> np.ndarray:
    with gzip.open(path, "rt") as f:
        header = f.readline().strip().split()
        if col_name not in header:
            raise ValueError(f"{path}: missing column '{col_name}'. Columns: {header}")
        idx = header.index(col_name)
        # Stream-load only the required column.
        data = np.loadtxt(f, usecols=[idx], dtype=float)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size < 100:
        raise ValueError(f"{path}: too few finite samples ({data.size}).")
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract GW luminosity distance samples from LIGO .dat.gz posterior files.")
    ap.add_argument("--in", dest="inp", required=True, help="Input .dat.gz posterior file (e.g. P1800061).")
    ap.add_argument(
        "--col",
        default="luminosity_distance_Mpc",
        help="Column name to extract (default: luminosity_distance_Mpc).",
    )
    ap.add_argument("--out", required=True, help="Output .npz path.")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional downsample cap (random, deterministic).")
    args = ap.parse_args()

    inp = Path(args.inp).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    dL = _read_column_from_dat_gz(inp, str(args.col))

    if args.max_samples is not None:
        n = int(args.max_samples)
        if n <= 0:
            raise ValueError("--max-samples must be positive.")
        if dL.size > n:
            rng = np.random.default_rng(0)
            idx = rng.choice(dL.size, size=n, replace=False)
            dL = dL[idx]

    np.savez_compressed(out, dL_samples_Mpc=dL, source=str(inp), col=str(args.col))
    print(f"Wrote {out}  (n={dL.size})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


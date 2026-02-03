#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from entropy_horizon_recon.run_log import collect_run_log_rows, write_run_log_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a run_log.csv summarizing GR vs mu(=entropy) comparisons (ΔLPD).")
    ap.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Outputs directory to scan (default: outputs/).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("FINDINGS") / "run_log.csv",
        help="CSV path to write (default: FINDINGS/run_log.csv).",
    )
    ap.add_argument(
        "--relative-to",
        type=Path,
        default=Path("."),
        help="Base path for writing relative out_dir/summary_path/manifest_path columns (default: repo root).",
    )
    args = ap.parse_args()

    rows = collect_run_log_rows(outputs_dir=args.outputs, relative_to=args.relative_to)
    write_run_log_csv(rows=rows, out_csv=args.out)
    print(f"Wrote {args.out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

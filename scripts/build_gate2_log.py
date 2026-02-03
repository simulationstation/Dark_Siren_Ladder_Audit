#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from entropy_horizon_recon.gate2_log import collect_gate2_log_rows, write_gate2_log_csv


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a gate2_log.csv summarizing GR H0 Gate-2 control runs.")
    ap.add_argument(
        "--outputs",
        type=Path,
        default=Path("outputs"),
        help="Outputs directory to scan (default: outputs/).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("FINDINGS") / "gate2_log.csv",
        help="CSV path to write (default: FINDINGS/gate2_log.csv).",
    )
    ap.add_argument(
        "--relative-to",
        type=Path,
        default=Path("."),
        help="Base path for writing relative out_dir/json_path/manifest_path columns (default: repo root).",
    )
    args = ap.parse_args()

    rows = collect_gate2_log_rows(outputs_dir=args.outputs, relative_to=args.relative_to)
    write_gate2_log_csv(rows=rows, out_csv=args.out)
    print(f"Wrote {args.out} with {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


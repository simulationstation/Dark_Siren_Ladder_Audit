from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(base: Path) -> list[dict]:
    rows = []
    for run in sorted(base.glob("M0_start*/tables/summary.json")):
        seed = run.parent.parent.name.replace("M0_start", "")
        d = json.loads(run.read_text())
        omega = d.get("posterior", {}).get("omega_m0", {})
        m = d.get("departure", {}).get("m", {})
        rows.append(
            {
                "seed": int(seed),
                "omega_p50": omega.get("p50"),
                "m_mean": m.get("mean"),
                "m_std": m.get("std"),
                "p_m_gt0": m.get("p_gt0"),
            }
        )
    return rows


def summarize(rows: list[dict]) -> dict:
    omegas = [r["omega_p50"] for r in rows]
    m_means = [r["m_mean"] for r in rows]
    m_stds = [r["m_std"] for r in rows]
    p_gt0 = [r["p_m_gt0"] for r in rows]
    return {
        "omega_span": float(max(omegas) - min(omegas)) if omegas else None,
        "m_mean_mean": float(np.mean(m_means)) if m_means else None,
        "m_std_mean": float(np.mean(m_stds)) if m_stds else None,
        "p_gt0_mean": float(np.mean(p_gt0)) if p_gt0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--highpower", type=Path, required=True)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--ridge", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    high_rows = load_rows(args.highpower)
    base_sum = summarize(base_rows)
    high_sum = summarize(high_rows)

    # Variant flip detection
    flipped = []
    for var_dir in sorted(args.variants.glob("*")):
        if not var_dir.is_dir():
            continue
        rows = load_rows(var_dir)
        if not rows:
            continue
        # flip if majority of seeds have m_mean > 0
        pos = sum(1 for r in rows if r["m_mean"] is not None and r["m_mean"] > 0)
        if pos >= 3:
            flipped.append(var_dir.name)
    final = {
        "baseline": base_sum,
        "highpower": high_sum,
        "variants_flipped": flipped,
    }

    out_dir = args.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

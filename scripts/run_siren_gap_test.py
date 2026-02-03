from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")  # headless/cluster-safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.siren_gap import load_siren_events, score_siren_events
from entropy_horizon_recon.sirens import load_mu_forward_posterior, predict_r_gw_em


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class BandSummary:
    run_dir: str
    convention: str
    n_draws: int
    z_min: float
    z_max: float
    z_n: int
    q: list[int]


def _quantiles(x: np.ndarray, q: list[int]) -> np.ndarray:
    return np.percentile(x, q, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Publication-style Siren Gap test (held-out scoring vs GR).")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished EM-only run directory (contains samples/).")
    ap.add_argument("--siren-data", required=True, help="Siren JSON file (bright sirens to start).")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/siren_gap_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="R_GW/EM convention (see siren_plan.md).")
    ap.add_argument("--allow-extrapolation", action="store_true", help="Allow extrapolation outside inferred x-domain.")
    ap.add_argument("--max-draws", type=int, default=None, help="Optional cap on posterior draws used for scoring.")

    ap.add_argument("--z-min", type=float, default=0.0, help="Plot grid z_min.")
    ap.add_argument("--z-max", type=float, default=None, help="Plot grid z_max (default: min(run z_max)).")
    ap.add_argument("--z-n", type=int, default=200, help="Plot grid size.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gap_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    events = load_siren_events(args.siren_data)
    # Copy the input siren manifest verbatim for reproducibility (avoid serializing large arrays).
    (tab_dir / "siren_data.json").write_text(Path(args.siren_data).read_text(encoding="utf-8"), encoding="utf-8")

    posts = []
    z_max_candidates = []
    for rd in args.run_dir:
        post = load_mu_forward_posterior(rd)
        posts.append((rd, post))
        z_max_candidates.append(float(post.z_grid[-1]))
    z_max = float(args.z_max) if args.z_max is not None else float(min(z_max_candidates))
    z_eval = np.linspace(float(args.z_min), z_max, int(args.z_n))

    q = [5, 16, 50, 84, 95]

    # 1) R(z) prediction bands (paper figure).
    plt.figure(figsize=(8, 5))
    for rd, post in posts:
        z, R = predict_r_gw_em(
            post,
            z_eval=z_eval,
            convention=args.convention,  # type: ignore[arg-type]
            allow_extrapolation=bool(args.allow_extrapolation),
        )
        qs = _quantiles(R, q=q)
        label = Path(rd).name
        plt.fill_between(z, qs[1], qs[3], alpha=0.25, label=f"{label} (p16-p84)")
        plt.plot(z, qs[2], linewidth=1.5, label=f"{label} (median)")

        summ = BandSummary(
            run_dir=str(rd),
            convention=str(args.convention),
            n_draws=int(R.shape[0]),
            z_min=float(z[0]),
            z_max=float(z[-1]),
            z_n=int(z.size),
            q=q,
        )
        (tab_dir / f"rgwem_band_{label}.json").write_text(json.dumps({**asdict(summ)}, indent=2))
        np.savez_compressed(tab_dir / f"rgwem_band_{label}.npz", z=z, q=np.array(q, dtype=int), R_q=qs)

    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.5)
    plt.xlabel("Redshift z")
    plt.ylabel("R_GW/EM(z) = dL_GW / dL_EM")
    plt.title(f"Siren Gap Prediction Band From mu(A) Posterior (convention {args.convention})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "rgwem_band.png", dpi=160)
    plt.close()

    # 2) Held-out scoring (Delta_LPD vs GR baseline).
    summaries = []
    all_scores = []
    for rd, post in posts:
        label = Path(rd).name
        summ, scores = score_siren_events(
            run_label=label,
            post=post,
            events=events,
            convention=args.convention,  # type: ignore[arg-type]
            allow_extrapolation=bool(args.allow_extrapolation),
            max_draws=args.max_draws,
        )
        summaries.append(asdict(summ))
        all_scores.extend([asdict(s) for s in scores])

    (tab_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    (tab_dir / "event_scores.json").write_text(json.dumps(all_scores, indent=2))

    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

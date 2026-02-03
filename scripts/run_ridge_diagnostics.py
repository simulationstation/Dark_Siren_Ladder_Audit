#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from entropy_horizon_recon.departure import compute_departure_stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    corr_rows = []
    for run in sorted(args.base.glob("M0_start*/samples/mu_forward_posterior.npz")):
        seed = run.parent.parent.name.replace("M0_start", "")
        with np.load(run, allow_pickle=True) as d:
            logmu_x = d["logmu_x_samples"]
            x_grid = d["x_grid"]
            omega = d["omega_m0"]
            H0 = d["H0"]
        # Compute logA grid weights by mapping x->logA with logA0 from H0
        # Use fixed logA grid from x_grid and mean H0
        c = 299792.458
        logA0 = np.log(4.0 * np.pi * (c / np.mean(H0)) ** 2)
        logA_grid = logA0 + x_grid
        dep = compute_departure_stats(logA_grid=logA_grid, logmu_samples=logmu_x)
        m_draw = dep["m"]["mean"]
        # For correlation, compute per-draw m using same weights
        # Recompute weights used in departure
        mean = np.mean(logmu_x, axis=0)
        var = np.var(logmu_x, axis=0, ddof=1)
        w = 1.0 / np.clip(var, 1e-12, np.inf)
        w = w / np.trapezoid(w, x=logA_grid)
        m_draws = np.trapezoid(logmu_x * w[None, :], x=logA_grid, axis=1)
        corr_om = float(np.corrcoef(m_draws, omega)[0, 1])
        corr_h0 = float(np.corrcoef(m_draws, H0)[0, 1])
        corr_rows.append({"seed": int(seed), "corr_m_omega": corr_om, "corr_m_h0": corr_h0})

        # binned m vs omega
        bins = int(args.bins)
        edges = np.quantile(omega, np.linspace(0, 1, bins + 1))
        centers = 0.5 * (edges[:-1] + edges[1:])
        m_bin = []
        for i in range(bins):
            sel = (omega >= edges[i]) & (omega <= edges[i + 1])
            m_bin.append(float(np.mean(m_draws[sel])) if np.any(sel) else np.nan)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(centers, m_bin, marker="o")
        ax.set_xlabel("Omega_m0 bin")
        ax.set_ylabel("E[m | Omega_m0 bin]")
        ax.set_title(f"m vs Omega_m0 (seed {seed})")
        fig.tight_layout()
        fig.savefig(out_dir / f"m_vs_om0_binned_seed{seed}.png", dpi=160)
        plt.close(fig)

    (out_dir / "corr_table.json").write_text(json.dumps(corr_rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

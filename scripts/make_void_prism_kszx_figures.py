#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(p: Path):
    return json.loads(p.read_text())


def _savefig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Generate paper-ready summary figures for the void-prism kszx pipeline run.\n"
            "Outputs are small PDF/PNG files intended for inclusion in reports.\n"
        )
    )
    ap.add_argument(
        "--pipeline-dir",
        default="outputs/void_prism_kszx_pipeline_20260130_225117UTC",
        help="Pipeline output directory (default: latest 2026-01-30 run).",
    )
    ap.add_argument(
        "--out-dir",
        default="1-30-output",
        help="Directory to write figures into (default: 1-30-output).",
    )
    args = ap.parse_args()

    pipeline_dir = Path(args.pipeline_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_path = pipeline_dir / "void_prism_eg_suite" / "tables" / "suite_joint.json"
    results_path = pipeline_dir / "void_prism_eg_joint" / "tables" / "results.json"
    nulls_voids_path = pipeline_dir / "nulls_rotate_voids" / "tables" / "nulls.json"
    nulls_kappa_path = pipeline_dir / "nulls_rotate_kappa" / "tables" / "nulls.json"

    suite = _load_json(suite_path)
    y_obs = np.asarray(suite["y_obs"], dtype=float)
    cov = np.asarray(suite["cov"], dtype=float)
    diag = np.diag(cov).astype(float)

    blocks = suite["blocks"]

    # ---------------------------
    # Figure: E_G^void(ell) per block
    # ---------------------------
    import matplotlib.pyplot as plt

    n_blocks = len(blocks)
    ncols = 2
    nrows = int(np.ceil(n_blocks / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 2.6 * nrows), sharex=False, sharey=False)
    axes = np.atleast_1d(axes).reshape((nrows, ncols))

    for i, b in enumerate(blocks):
        ax = axes[i // ncols][i % ncols]
        lo, hi = b["slice"]
        lo = int(lo)
        hi = int(hi)
        ell = np.asarray(b["ell"], dtype=float)
        y = y_obs[lo:hi]
        yerr = np.sqrt(np.clip(diag[lo:hi], 0.0, np.inf))

        ax.errorbar(ell, y, yerr=yerr, fmt="o-", lw=1.0, ms=3.0, capsize=2.0)
        ax.axhline(0.0, color="k", lw=0.8, alpha=0.6)
        ax.set_title(f"{b['name']} (z_eff={float(b['z_eff']):.3f})")
        ax.set_xlabel(r"$\ell$ bin center")
        ax.set_ylabel(r"$\hat E_G^{\mathrm{void}}(\ell)$")

    # Hide unused axes
    for j in range(n_blocks, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Void-Prism Measurement: $\\hat E_G^{\\mathrm{void}}(\\ell)$ by (z, Rv) block", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir / "void_prism_kszx_eg_obs_by_block.pdf")
    _savefig(fig, out_dir / "void_prism_kszx_eg_obs_by_block.png")
    plt.close(fig)

    # ---------------------------
    # Figure: ΔLPD per seed
    # ---------------------------
    rows = _load_json(results_path)
    runs = [r["run"] for r in rows]
    dlpds = np.asarray([r["delta_lpd_vs_gr"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(np.arange(len(dlpds)), dlpds)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(np.arange(len(dlpds)))
    ax.set_xticklabels(runs, rotation=30, ha="right")
    ax.set_ylabel(r"$\Delta\mathrm{LPD} = \mathrm{LPD}_{\mathrm{minimal}}-\mathrm{LPD}_{\mathrm{GR}}$")
    ax.set_title("Void-Prism Scoring vs info+ posteriors (kszx theta): ΔLPD per seed")
    fig.tight_layout()
    _savefig(fig, out_dir / "void_prism_kszx_delta_lpd_per_seed.pdf")
    _savefig(fig, out_dir / "void_prism_kszx_delta_lpd_per_seed.png")
    plt.close(fig)

    # ---------------------------
    # Figure: null distributions for eg_rms
    # ---------------------------
    def _plot_null(ax, nulls_path: Path, title: str):
        d = _load_json(nulls_path)
        obs = float(d["obs"]["eg_rms"])
        null_rows = d.get("null_rows") or []
        eg_rms = np.asarray([r["eg_rms"] for r in null_rows], dtype=float)
        eg_rms = eg_rms[np.isfinite(eg_rms)]
        ax.hist(eg_rms, bins=20, alpha=0.8)
        ax.axvline(obs, color="r", lw=2.0, label=f"obs eg_rms={obs:.3g}")
        ax.set_title(title)
        ax.set_xlabel("eg_rms")
        ax.set_ylabel("count")
        ax.legend(loc="upper right", fontsize=8)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 3.5))
    _plot_null(axes[0], nulls_voids_path, "Null: rotate_voids")
    _plot_null(axes[1], nulls_kappa_path, "Null: rotate_kappa")
    fig.suptitle("Void-Prism Null Battery (kszx theta): EG RMS distributions", y=1.02)
    fig.tight_layout()
    _savefig(fig, out_dir / "void_prism_kszx_null_eg_rms_hist.pdf")
    _savefig(fig, out_dir / "void_prism_kszx_null_eg_rms_hist.png")
    plt.close(fig)

    # ---------------------------
    # Also write a small JSON with key scalars for convenience.
    # ---------------------------
    out_summary = {
        "pipeline_dir": str(pipeline_dir),
        "suite_joint_json": str(suite_path),
        "results_json": str(results_path),
        "nulls_rotate_voids_json": str(nulls_voids_path),
        "nulls_rotate_kappa_json": str(nulls_kappa_path),
        "delta_lpd": {r["run"]: float(r["delta_lpd_vs_gr"]) for r in rows},
        "delta_lpd_mean": float(np.mean(dlpds)),
        "delta_lpd_min": float(np.min(dlpds)),
        "delta_lpd_max": float(np.max(dlpds)),
        "y_obs_rms": float(np.sqrt(float(np.mean(y_obs * y_obs)))),
    }
    (out_dir / "void_prism_kszx_key_numbers.json").write_text(json.dumps(out_summary, indent=2) + "\n")

    print(f"[ok] wrote figures to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

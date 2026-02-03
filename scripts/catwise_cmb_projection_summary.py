#!/usr/bin/env python3
"""
CatWISE dipole: CMB-axis projection diagnostic.

This is an estimator-agnostic summary intended to separate:
  - a CMB-aligned component of the fitted dipole vector, from
  - a perpendicular component that reflects direction drift / systematics.

It reads one or more JSON outputs produced by:
  - scripts/reproduce_rvmp_fig5_catwise.py            (Secrest-style linear solve)
  - scripts/reproduce_rvmp_fig5_catwise_poisson_glm.py (Poisson GLM MLE)
or compatible JSONs with keys:
  {"meta": {...}, "rows": [{"w1_cut": ..., "dipole": {"D_hat":..., "l_hat_deg":..., "b_hat_deg":...}}, ...]}

Output:
  - cmb_projection_summary.json (machine-readable)
  - cmb_projection_summary.csv  (flat table)
  - cmb_projection_summary.png  (4-panel plot vs W1_max)
  - summary.md                  (human-readable checkpoints)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def utc_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def unitvec_lb(l_deg: float, b_deg: float) -> np.ndarray:
    l = math.radians(float(l_deg) % 360.0)
    b = math.radians(float(b_deg))
    return np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)], dtype=float)


def ang_sep_deg(u: np.ndarray, v: np.ndarray) -> float:
    cuv = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(math.degrees(math.acos(cuv)))


@dataclass(frozen=True)
class Series:
    label: str
    path: Path
    w1_cut: np.ndarray
    D: np.ndarray
    l_deg: np.ndarray
    b_deg: np.ndarray
    cos_cmb: np.ndarray
    theta_deg: np.ndarray
    D_par: np.ndarray
    D_perp: np.ndarray
    meta: dict[str, Any]


def parse_label_path(s: str) -> tuple[str, Path]:
    if "=" not in s:
        raise ValueError("Each --input must be of the form label=/path/to/file.json")
    label, path = s.split("=", 1)
    label = label.strip()
    path = Path(path).expanduser()
    if not label:
        raise ValueError("Empty label in --input")
    return label, path


def load_series(*, label: str, path: Path, cmb_axis: np.ndarray) -> Series:
    obj = json.loads(path.read_text())
    if "rows" not in obj:
        raise ValueError(f"{path} does not look like an rvmp_fig5 JSON (missing 'rows').")
    rows = obj["rows"]
    w1 = np.array([r["w1_cut"] for r in rows], dtype=float)
    D = np.array([r["dipole"]["D_hat"] for r in rows], dtype=float)
    l = np.array([r["dipole"]["l_hat_deg"] for r in rows], dtype=float)
    b = np.array([r["dipole"]["b_hat_deg"] for r in rows], dtype=float)

    cos_cmb = np.empty_like(D)
    theta = np.empty_like(D)
    D_par = np.empty_like(D)
    D_perp = np.empty_like(D)
    for i in range(w1.size):
        n = unitvec_lb(l[i], b[i])
        c = float(np.clip(np.dot(n, cmb_axis), -1.0, 1.0))
        cos_cmb[i] = c
        theta[i] = float(math.degrees(math.acos(c)))
        D_par[i] = D[i] * c
        D_perp[i] = D[i] * float(math.sqrt(max(0.0, 1.0 - c * c)))

    order = np.argsort(w1)
    meta = obj.get("meta", {})
    return Series(
        label=label,
        path=path,
        w1_cut=w1[order],
        D=D[order],
        l_deg=l[order],
        b_deg=b[order],
        cos_cmb=cos_cmb[order],
        theta_deg=theta[order],
        D_par=D_par[order],
        D_perp=D_perp[order],
        meta=meta,
    )


def write_csv(out_csv: Path, series_list: list[Series]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "path",
        "w1_cut",
        "D",
        "l_hat_deg",
        "b_hat_deg",
        "cos_to_cmb",
        "theta_deg",
        "D_par_on_cmb",
        "D_perp_to_cmb",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in series_list:
            for i in range(s.w1_cut.size):
                w.writerow(
                    {
                        "label": s.label,
                        "path": str(s.path),
                        "w1_cut": float(s.w1_cut[i]),
                        "D": float(s.D[i]),
                        "l_hat_deg": float(s.l_deg[i]),
                        "b_hat_deg": float(s.b_deg[i]),
                        "cos_to_cmb": float(s.cos_cmb[i]),
                        "theta_deg": float(s.theta_deg[i]),
                        "D_par_on_cmb": float(s.D_par[i]),
                        "D_perp_to_cmb": float(s.D_perp[i]),
                    }
                )


def write_json(out_json: Path, *, cmb_lb: tuple[float, float], series_list: list[Series]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "meta": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "cmb_axis_lb_deg": [float(cmb_lb[0]), float(cmb_lb[1])],
        },
        "series": [],
    }
    for s in series_list:
        payload["series"].append(
            {
                "label": s.label,
                "path": str(s.path),
                "input_meta": s.meta,
                "rows": [
                    {
                        "w1_cut": float(s.w1_cut[i]),
                        "D": float(s.D[i]),
                        "l_hat_deg": float(s.l_deg[i]),
                        "b_hat_deg": float(s.b_deg[i]),
                        "cos_to_cmb": float(s.cos_cmb[i]),
                        "theta_deg": float(s.theta_deg[i]),
                        "D_par_on_cmb": float(s.D_par[i]),
                        "D_perp_to_cmb": float(s.D_perp[i]),
                    }
                    for i in range(s.w1_cut.size)
                ],
            }
        )
    out_json.write_text(json.dumps(payload, indent=2))


def plot(out_png: Path, series_list: list[Series]) -> None:
    # Import matplotlib lazily to keep import-time errors obvious.
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.5), dpi=160, constrained_layout=True)
    ax = ax.reshape(2, 2)

    for s in series_list:
        ax[0, 0].plot(s.w1_cut, s.D, marker="o", ms=3, lw=1.2, label=s.label)
        ax[0, 1].plot(s.w1_cut, s.D_par, marker="o", ms=3, lw=1.2, label=s.label)
        ax[1, 0].plot(s.w1_cut, s.D_perp, marker="o", ms=3, lw=1.2, label=s.label)
        ax[1, 1].plot(s.w1_cut, s.theta_deg, marker="o", ms=3, lw=1.2, label=s.label)

    ax[0, 0].set_title("Total dipole amplitude $D$")
    ax[0, 0].set_xlabel("$W1_{\\max}$")
    ax[0, 0].set_ylabel("$D$")

    ax[0, 1].set_title("CMB-axis projection $D_{\\parallel}=D\\cos\\theta$")
    ax[0, 1].set_xlabel("$W1_{\\max}$")
    ax[0, 1].set_ylabel("$D_{\\parallel}$")
    ax[0, 1].axhline(0.0, color="k", lw=0.8, alpha=0.35)

    ax[1, 0].set_title("Perpendicular component $D_{\\perp}=D\\sin\\theta$")
    ax[1, 0].set_xlabel("$W1_{\\max}$")
    ax[1, 0].set_ylabel("$D_{\\perp}$")

    ax[1, 1].set_title("Angle to CMB dipole $\\theta$ (deg)")
    ax[1, 1].set_xlabel("$W1_{\\max}$")
    ax[1, 1].set_ylabel("$\\theta$ [deg]")
    ax[1, 1].set_ylim(0.0, 90.0)

    # One shared legend.
    handles, labels = ax[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, max(1, len(labels))), frameon=False)

    fig.suptitle("CatWISE dipole decomposition relative to the CMB dipole axis", y=1.02, fontsize=12)
    fig.savefig(out_png)
    plt.close(fig)


def format_checkpoint(*, s: Series, w: float) -> str:
    # find nearest w1 value
    idx = int(np.argmin(np.abs(s.w1_cut - float(w))))
    return (
        f"- `{s.label}` at `W1_max={s.w1_cut[idx]:.2f}`: "
        f"D={s.D[idx]:.5f}, "
        f"D_par={s.D_par[idx]:.5f}, "
        f"D_perp={s.D_perp[idx]:.5f}, "
        f"theta={s.theta_deg[idx]:.2f} deg"
    )


def write_md(out_md: Path, *, cmb_lb: tuple[float, float], series_list: list[Series]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# CatWISE dipole: CMB-axis projection diagnostic\n")
    lines.append(f"- CMB axis (Galactic): (l,b)=({cmb_lb[0]:.3f}°, {cmb_lb[1]:.3f}°)\n")

    for s in series_list:
        lines.append(f"## {s.label}\n")
        lines.append(f"- Source JSON: `{s.path}`\n")
        if s.meta:
            lines.append("- Input meta keys: " + ", ".join(sorted(s.meta.keys())) + "\n")

        # Stability quickies across all cuts.
        lines.append(
            f"- Across cuts: D mean={float(np.mean(s.D)):.5f}, sd={float(np.std(s.D)):.5f}; "
            f"D_par mean={float(np.mean(s.D_par)):.5f}, sd={float(np.std(s.D_par)):.5f}; "
            f"theta range=[{float(np.min(s.theta_deg)):.2f}, {float(np.max(s.theta_deg)):.2f}] deg\n"
        )
        lines.append("### Checkpoints\n")
        for w in [15.5, 16.0, 16.5, 16.6]:
            if float(np.min(s.w1_cut)) <= w <= float(np.max(s.w1_cut)):
                lines.append(format_checkpoint(s=s, w=w) + "\n")
        lines.append("\n")

    out_md.write_text("".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input JSON as label=/path/to/rvmp_fig5_*.json (repeatable).",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory (default: outputs/catwise_cmb_projection_summary_<UTC>).",
    )
    p.add_argument("--cmb-axis-lb", type=float, nargs=2, default=[264.021, 48.253], help="CMB dipole axis (l b) in deg.")

    args = p.parse_args()

    cmb_l, cmb_b = float(args.cmb_axis_lb[0]), float(args.cmb_axis_lb[1])
    cmb_axis = unitvec_lb(cmb_l, cmb_b)

    outdir = Path(args.outdir) if args.outdir else Path("outputs") / f"catwise_cmb_projection_summary_{utc_tag()}"
    outdir.mkdir(parents=True, exist_ok=True)

    series_list: list[Series] = []
    for s in args.input:
        label, path = parse_label_path(s)
        if not path.exists():
            raise FileNotFoundError(str(path))
        series_list.append(load_series(label=label, path=path, cmb_axis=cmb_axis))

    out_json = outdir / "cmb_projection_summary.json"
    out_csv = outdir / "cmb_projection_summary.csv"
    out_png = outdir / "cmb_projection_summary.png"
    out_md = outdir / "summary.md"

    write_json(out_json, cmb_lb=(cmb_l, cmb_b), series_list=series_list)
    write_csv(out_csv, series_list)
    plot(out_png, series_list)
    write_md(out_md, cmb_lb=(cmb_l, cmb_b), series_list=series_list)

    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] wrote: {out_md}")


if __name__ == "__main__":
    main()


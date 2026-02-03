from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.departure import compute_departure_stats
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty


def _jsonify(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description="Compare real-data mapping variants (M0/M1/M2).")
    p.add_argument("--out", type=Path, default=Path("outputs/variants_compare"))
    p.add_argument(
        "--variants",
        type=Path,
        nargs="+",
        default=[Path("outputs/realdata_variant_M0"), Path("outputs/realdata_variant_M1"), Path("outputs/realdata_variant_M2")],
    )
    p.add_argument("--n-grid", type=int, default=140)
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = git_head_sha(repo_root=repo_root) or "unknown"
    git_dirty = git_is_dirty(repo_root=repo_root)
    cmd = command_str()

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)

    variants = []
    for d in args.variants:
        d = Path(d)
        tag = d.name.replace("realdata_variant_", "")
        npz = np.load(d / "samples" / "logmu_logA_samples.npz")
        logA = np.asarray(npz["logA"], dtype=float)
        samp = np.asarray(npz["logmu_samples"], dtype=float)
        mu_npz = np.load(d / "samples" / "mu_forward_posterior.npz")
        z_grid = np.asarray(mu_npz["z_grid"], dtype=float)
        H0_s = np.asarray(mu_npz["H0"], dtype=float)
        Om_s = np.asarray(mu_npz["omega_m0"], dtype=float)
        dep = _load_json(d / "tables" / "departure_stats.json")
        prox = _load_json(d / "tables" / "proximity.json")["prox"]
        summ = _load_json(d / "tables" / "summary.json")
        variants.append(
            {
                "tag": tag,
                "dir": str(d),
                "logA": logA,
                "logmu": samp,
                "z_grid": z_grid,
                "H0_samples": H0_s,
                "omega_m0_samples": Om_s,
                "departure": dep,
                "prox": prox,
                "summary": summ,
            }
        )

    # Common overlap domain across variants
    lo = max(float(v["logA"].min()) for v in variants)
    hi = min(float(v["logA"].max()) for v in variants)
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        raise RuntimeError("No common logA overlap across variants.")
    logA_grid = np.linspace(lo, hi, int(args.n_grid))

    # Interpolate each draw to the common grid.
    aligned = {}
    for v in variants:
        x0 = v["logA"]
        draws0 = v["logmu"]
        out = np.empty((draws0.shape[0], logA_grid.size))
        for i in range(draws0.shape[0]):
            out[i] = np.interp(logA_grid, x0, draws0[i])
        aligned[v["tag"]] = out

    # Recompute scalar departure stats on the shared overlap domain.
    common_dep = {tag: compute_departure_stats(logA_grid=logA_grid, logmu_samples=samp) for tag, samp in aligned.items()}

    # Overlay plot of logμ(A)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {"M0": "C0", "M1": "C2", "M2": "C3"}
    for tag in sorted(aligned.keys()):
        samp = aligned[tag]
        med = np.median(samp, axis=0)
        lo = np.percentile(samp, 16.0, axis=0)
        hi = np.percentile(samp, 84.0, axis=0)
        c = colors.get(tag, None)
        if tag == "M0":
            ax.fill_between(logA_grid, lo, hi, color=c, alpha=0.22, linewidth=0)
            ax.plot(logA_grid, med, color=c, lw=2.2, label=tag)
        else:
            ax.plot(logA_grid, med, color=c, lw=1.9, alpha=0.95, label=tag)
    ax.axhline(0.0, color="k", lw=1, alpha=0.5)
    ax.set(xlabel="log A", ylabel="log μ(A)", title="Mapping variants (real data)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(report_paths.figures_dir / "logmu_logA_variants_overlay.png", dpi=200)
    plt.close(fig)

    # Overlay plot of K(z) medians (mapping prefactor proxy).
    # K(z) := 3 H0^2 Omega_m0 (1+z)^2  (up to an overall constant factor irrelevant for comparisons).
    z_ref = variants[0]["z_grid"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for v in sorted(variants, key=lambda x: x["tag"]):
        tag = v["tag"]
        H0_med = float(np.median(v["H0_samples"]))
        Om_med = float(np.median(v["omega_m0_samples"]))
        K = 3.0 * (H0_med**2) * Om_med * (1.0 + z_ref) ** 2
        ax.plot(z_ref, K, lw=2.0, label=tag, color=colors.get(tag, None))
    ax.set(xlabel="z", ylabel="K(z) [arb.]", title="Mapping prefactor proxy K(z) at posterior medians")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(report_paths.figures_dir / "Kz_variants_overlay.png", dpi=200)
    plt.close(fig)

    # Comparison table
    rows = []
    for v in sorted(variants, key=lambda x: x["tag"]):
        tag = v["tag"]
        dep = v["departure"]
        dep_c = common_dep[tag]
        prox = v["prox"]
        H0 = v["summary"]["posterior"]["H0"]["p50"]
        Om = v["summary"]["posterior"]["omega_m0"]["p50"]
        rows.append(
            [
                tag,
                f"{H0:.2f}",
                f"{Om:.3f}",
                f"{dep['m']['mean']:.3f} +/- {dep['m']['std']:.3f}",
                f"{dep['m']['p_gt0']:.2f}",
                f"{dep_c['m']['mean']:.3f} +/- {dep_c['m']['std']:.3f}",
                f"{dep_c['m']['p_gt0']:.2f}",
                f"{dep['slope']['mean']:.3f} +/- {dep['slope']['std']:.3f}",
                f"{prox['fit_to_mean']['tsallis']['delta']:.3f}",
                f"{prox['fit_to_mean']['barrow']['Delta']:.3f}",
                f"{prox['fit_to_mean']['kaniadakis']['beta_tilde']:.3g}",
            ]
        )

    table_md = format_table(
        rows,
        headers=[
            "variant",
            "H0_p50",
            "Om0_p50",
            "m(mean±sd)",
            "P(m>0)",
            "m_common(mean±sd)",
            "P(m>0)_common",
            "slope(mean±sd)",
            "Tsallis δ",
            "Barrow Δ",
            "Kaniadakis β̃",
        ],
    )

    out_json = {
        "git": {"sha": git_sha, "dirty": git_dirty},
        "command": cmd,
        "logA_overlap": {"min": float(logA_grid.min()), "max": float(logA_grid.max()), "n": int(logA_grid.size)},
        "variants": [
            {
                "tag": v["tag"],
                "dir": v["dir"],
                "departure": v["departure"],
                "departure_common_domain": common_dep[v["tag"]],
                "prox_fit_to_mean": v["prox"]["fit_to_mean"],
                "prox_D2_mean": v["prox"]["D2_mean"],
                "posterior": v["summary"]["posterior"],
            }
            for v in sorted(variants, key=lambda x: x["tag"])
        ],
    }
    (report_paths.tables_dir / "variants_compare.json").write_text(json.dumps(_jsonify(out_json), indent=2), encoding="utf-8")

    # Dedicated overlap-domain diagnostics (task-oriented file name).
    m_common = {
        "git": {"sha": git_sha, "dirty": git_dirty},
        "command": cmd,
        "logA_common": {"min": float(logA_grid.min()), "max": float(logA_grid.max()), "n": int(logA_grid.size)},
        "variants": {tag: common_dep[tag] for tag in sorted(common_dep.keys())},
    }
    (args.out / "m_common_domain.json").write_text(json.dumps(_jsonify(m_common), indent=2), encoding="utf-8")

    # Human-readable overlap report.
    md_overlap = []
    md_overlap.append("# Mapping overlap diagnostics (common logA domain)\n")
    md_overlap.append(f"_git: {git_sha} (dirty={git_dirty})_\n")
    md_overlap.append(f"_command: `{cmd}`_\n")
    md_overlap.append(
        f"Common logA overlap: [{logA_grid.min():.6g}, {logA_grid.max():.6g}] (n={logA_grid.size})\n"
    )
    md_overlap.append("\n![logmu](figures/logmu_logA_variants_overlay.png)\n")
    md_overlap.append("\n![Kz](figures/Kz_variants_overlay.png)\n")
    md_overlap.append("\n## Scalar departure stats on common domain\n")
    rows2 = []
    for tag in sorted(common_dep.keys()):
        dep_c = common_dep[tag]
        rows2.append(
            [
                tag,
                f"{dep_c['m']['mean']:.3f} +/- {dep_c['m']['std']:.3f}",
                f"{dep_c['m']['p_gt0']:.2f}",
                f"{dep_c['slope']['mean']:.3f} +/- {dep_c['slope']['std']:.3f}",
                f"{dep_c['slope']['p_gt0']:.2f}",
            ]
        )
    md_overlap.append(
        format_table(
            rows2,
            headers=["variant", "m_common(mean±sd)", "P(m>0)", "s_common(mean±sd)", "P(s>0)"],
        )
    )
    (args.out / "mapping_overlap_report.md").write_text("\n".join(md_overlap) + "\n", encoding="utf-8")

    md = []
    md.append("# Real-data mapping variant comparison\n")
    md.append(f"_git: {git_sha} (dirty={git_dirty})_\n")
    md.append(f"_command: `{cmd}`_\n")
    md.append("\n![variants](figures/logmu_logA_variants_overlay.png)\n")
    md.append("\n![Kz](figures/Kz_variants_overlay.png)\n")
    md.append("\n" + table_md + "\n")
    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

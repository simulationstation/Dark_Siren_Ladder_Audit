from __future__ import annotations

import os

# Avoid nested parallelism (BLAS/OpenMP) when using multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike
from entropy_horizon_recon.recon_gp import reconstruct_H_gp
from entropy_horizon_recon.report import ReportPaths, write_markdown_report
from entropy_horizon_recon.repro import command_str, git_head_sha, git_is_dirty
from entropy_horizon_recon.viz import save_band_plot


def _apply_cpu_affinity(n_cores: int) -> None:
    """Best-effort CPU affinity limiter (Linux only)."""
    if n_cores is None or n_cores <= 0:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            total = os.cpu_count() or n_cores
            use = min(int(n_cores), int(total))
            os.sched_setaffinity(0, set(range(use)))
    except Exception:
        return


def _dense_domain_zmax(
    z: np.ndarray,
    *,
    z_min: float,
    z_max_cap: float,
    bin_width: float,
    min_per_bin: int,
) -> float:
    z = np.asarray(z, dtype=float)
    z = z[(z >= z_min) & (z <= z_max_cap)]
    if z.size == 0:
        raise ValueError("No SN redshifts in requested range.")
    edges = np.arange(z_min, z_max_cap + bin_width, bin_width)
    counts, _ = np.histogram(z, bins=edges)
    ok = counts >= min_per_bin
    if not np.any(ok):
        return float(z_min + bin_width)
    last_good = int(np.where(ok)[0].max())
    return float(edges[last_good + 1])


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


def main() -> int:
    p = argparse.ArgumentParser(description="Run H(z) GP cross-check (kernel robustness).")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--z-min", type=float, default=0.02)
    p.add_argument("--z-max-cap", type=float, default=1.2)
    p.add_argument("--sn-bin-width", type=float, default=0.05)
    p.add_argument("--sn-min-per-bin", type=int, default=20)
    p.add_argument("--n-knots", type=int, default=16)
    p.add_argument("--n-grid", type=int, default=200)
    p.add_argument("--gp-kernel", type=str, default="rbf", choices=["rbf", "matern32", "matern52"])
    p.add_argument("--gp-walkers", type=int, default=64)
    p.add_argument("--gp-steps", type=int, default=1200)
    p.add_argument("--gp-burn", type=int, default=400)
    p.add_argument("--gp-procs", type=int, default=16)
    p.add_argument("--cpu-cores", type=int, default=16)
    p.add_argument(
        "--baseline-gp-samples",
        type=Path,
        default=None,
        help="Optional: path to baseline Hz_gp_samples.npz for median/band comparison.",
    )
    args = p.parse_args()
    _apply_cpu_affinity(int(args.cpu_cores))

    repo_root = Path(__file__).resolve().parents[1]
    git_sha = git_head_sha(repo_root=repo_root) or "unknown"
    git_dirty = git_is_dirty(repo_root=repo_root)
    cmd = command_str()

    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()

    # Load data
    sn = load_pantheon_plus(paths=paths, cov_kind="stat+sys", subset="cosmology", z_column="zHD")
    cc = load_chronometers(paths=paths, variant="BC03_all")
    bao12 = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    bao16 = load_bao(paths=paths, dataset="sdss_dr16_lrg_bao_dmdh")
    desi24 = load_bao(paths=paths, dataset="desi_2024_bao_all")

    z_max = _dense_domain_zmax(
        sn.z,
        z_min=float(args.z_min),
        z_max_cap=float(args.z_max_cap),
        bin_width=float(args.sn_bin_width),
        min_per_bin=int(args.sn_min_per_bin),
    )
    z_min = float(args.z_min)

    sn_like = SNLogLike.from_pantheon(sn, z_min=z_min, z_max=z_max)
    cc_like = ChronometerLogLike.from_data(cc, z_min=z_min, z_max=z_max)
    bao_likes = []
    for dataset, bao in [
        ("sdss_dr12_consensus_bao", bao12),
        ("sdss_dr16_lrg_bao_dmdh", bao16),
        ("desi_2024_bao_all", desi24),
    ]:
        try:
            bao_likes.append(BaoLogLike.from_data(bao, dataset=dataset, constants=constants, z_min=z_min, z_max=z_max))
        except ValueError:
            continue

    z_knots = np.linspace(0.0, z_max, int(args.n_knots))
    z_grid = np.linspace(0.0, z_max, int(args.n_grid))

    gp_post = reconstruct_H_gp(
        z_knots=z_knots,
        sn_like=sn_like,
        cc_like=cc_like,
        bao_likes=bao_likes,
        constants=constants,
        z_grid=z_grid,
        z_max_background=z_max,
        kernel=str(args.gp_kernel),
        n_walkers=int(args.gp_walkers),
        n_steps=int(args.gp_steps),
        n_burn=int(args.gp_burn),
        seed=int(args.seed),
        n_processes=int(args.gp_procs),
    )

    # Save samples and bands
    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)
    (report_paths.out_dir / "samples").mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        report_paths.out_dir / "samples" / "Hz_gp_samples.npz",
        z=z_grid,
        H_samples=gp_post.H_samples,
        dH_dz_samples=gp_post.dH_dz_samples,
    )
    (report_paths.out_dir / "samples" / "Hz_gp_meta.json").write_text(
        json.dumps({"meta": gp_post.meta, "hyper_samples": {k: v.tolist() for k, v in gp_post.hyper_samples.items()}}, indent=2),
        encoding="utf-8",
    )

    H_med = np.median(gp_post.H_samples, axis=0)
    H_lo = np.percentile(gp_post.H_samples, 16.0, axis=0)
    H_hi = np.percentile(gp_post.H_samples, 84.0, axis=0)
    save_band_plot(
        z_grid,
        H_med,
        H_lo,
        H_hi,
        xlabel="z",
        ylabel="H(z) [km/s/Mpc]",
        title=f"H(z) GP cross-check ({args.gp_kernel})",
        path=report_paths.figures_dir / "Hz_gp.png",
    )

    compare = None
    if args.baseline_gp_samples is not None and args.baseline_gp_samples.exists():
        base = np.load(args.baseline_gp_samples)
        z0 = np.asarray(base["z"], dtype=float)
        Hs0 = np.asarray(base["H_samples"], dtype=float)
        if z0.shape != z_grid.shape or not np.allclose(z0, z_grid, rtol=0, atol=1e-12):
            raise ValueError("baseline_gp_samples z-grid mismatch.")
        base_med = np.median(Hs0, axis=0)
        base_lo = np.percentile(Hs0, 16.0, axis=0)
        base_hi = np.percentile(Hs0, 84.0, axis=0)
        base_std = 0.5 * (base_hi - base_lo) + 1e-12
        rms_delta_sigma = float(np.sqrt(np.mean(((H_med - base_med) / base_std) ** 2)))
        max_abs = float(np.max(np.abs(H_med - base_med)))
        compare = {"rms_delta_sigma": rms_delta_sigma, "max_abs_delta": max_abs}

    summary = {
        "git": {"sha": git_sha, "dirty": git_dirty},
        "command": cmd,
        "seed": int(args.seed),
        "z_domain": {"z_min": z_min, "z_max": float(z_max)},
        "gp_meta": gp_post.meta,
        "baseline_compare": compare,
    }
    (report_paths.tables_dir / "summary.json").write_text(json.dumps(_jsonify(summary), indent=2), encoding="utf-8")

    md = []
    md.append("# H(z) GP cross-check\n")
    md.append(f"_git: {git_sha} (dirty={git_dirty})_\n")
    md.append(f"_command: `{cmd}`_\n")
    md.append(f"Kernel: `{args.gp_kernel}`; z_max={z_max:.3f}\n")
    md.append(f"- acceptance_fraction_mean: {gp_post.meta.get('acceptance_fraction_mean', float('nan')):.3f}\n")
    if compare is not None:
        md.append(f"- baseline rms_delta_sigma: {compare['rms_delta_sigma']:.3f}\n")
        md.append(f"- baseline max |Δ median|: {compare['max_abs_delta']:.3f} km/s/Mpc\n")
    md.append("\n![Hz_gp](figures/Hz_gp.png)\n")
    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


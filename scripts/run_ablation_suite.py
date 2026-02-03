from __future__ import annotations

import os

# Avoid nested parallelism (BLAS/OpenMP) during ablations.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.ingest import load_bao, load_chronometers, load_pantheon_plus
from entropy_horizon_recon.inversion import reconstruct_logmu_of_logA
from entropy_horizon_recon.likelihoods import BaoLogLike, ChronometerLogLike, SNLogLike
from entropy_horizon_recon.proximity import proximity_summary
from entropy_horizon_recon.recon_gp import reconstruct_H_gp
from entropy_horizon_recon.recon_spline import reconstruct_H_spline
from entropy_horizon_recon.report import ReportPaths, format_table, write_markdown_report


def _apply_cpu_affinity(n_cores: int) -> None:
    """Best-effort CPU affinity limiter (Linux only)."""
    if n_cores is None or n_cores <= 0:
        return
    try:
        if hasattr(os, "sched_setaffinity"):
            if hasattr(os, "sched_getaffinity"):
                allowed = sorted(os.sched_getaffinity(0))
                if not allowed:
                    return
                use = min(int(n_cores), len(allowed))
                os.sched_setaffinity(0, set(allowed[:use]))
                return
            total = os.cpu_count() or n_cores
            use = min(int(n_cores), int(total))
            os.sched_setaffinity(0, set(range(use)))
    except Exception:
        return


def _resolve_cpu_cores(requested: int | None) -> int:
    total = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            allowed = len(os.sched_getaffinity(0))
            if allowed > 0:
                total = min(int(total), int(allowed))
        except Exception:
            pass
    if requested is None or requested <= 0:
        return int(total)
    return int(min(int(requested), int(total)))


@dataclass(frozen=True)
class AblationResult:
    name: str
    kernel: str
    cov_kind: str
    diagonal_cov: bool
    smooth_lambda: float
    D2_bh: float
    D2_tsallis: float
    D2_barrow: float
    D2_kaniadakis: float


def _band(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.median(samples, axis=0)
    lo = np.percentile(samples, 16, axis=0)
    hi = np.percentile(samples, 84, axis=0)
    return med, lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description="Run robustness ablations.")
    parser.add_argument("--out", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z-min", type=float, default=0.02)
    parser.add_argument("--z-max", type=float, default=1.2)
    parser.add_argument("--n-knots", type=int, default=16)
    parser.add_argument("--n-grid", type=int, default=200)
    parser.add_argument("--gp-walkers", type=int, default=64)
    parser.add_argument("--gp-steps", type=int, default=600)
    parser.add_argument("--gp-burn", type=int, default=200)
    parser.add_argument("--omega-m0", type=float, default=0.3)
    parser.add_argument(
        "--cpu-cores",
        type=int,
        default=0,
        help="Limit this run to the first N CPU cores (best-effort, 0=all).",
    )
    args = parser.parse_args()
    cpu_cores = _resolve_cpu_cores(args.cpu_cores)
    _apply_cpu_affinity(cpu_cores)
    args.cpu_cores = cpu_cores

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)
    constants = PhysicalConstants()

    report_paths = ReportPaths(out_dir=args.out)
    report_paths.figures_dir.mkdir(parents=True, exist_ok=True)
    report_paths.tables_dir.mkdir(parents=True, exist_ok=True)

    sn = load_pantheon_plus(paths=paths, cov_kind="stat+sys", subset="cosmology", z_column="zHD")
    cc = load_chronometers(paths=paths, variant="BC03_all")
    bao12 = load_bao(paths=paths, dataset="sdss_dr12_consensus_bao")
    desi24 = load_bao(paths=paths, dataset="desi_2024_bao_all")

    z_min, z_max = float(args.z_min), float(args.z_max)
    sn_like = SNLogLike.from_pantheon(sn, z_min=z_min, z_max=z_max)
    cc_like = ChronometerLogLike.from_data(cc, z_min=z_min, z_max=z_max)
    bao_likes = []
    for dataset, bao in [
        ("sdss_dr12_consensus_bao", bao12),
        ("desi_2024_bao_all", desi24),
    ]:
        try:
            bao_likes.append(BaoLogLike.from_data(bao, dataset=dataset, constants=constants, z_min=z_min, z_max=z_max))
        except ValueError as e:
            print(f"Skipping BAO dataset {dataset}: {e}")

    z_knots = np.linspace(0.0, z_max, args.n_knots)
    z_grid = np.linspace(0.0, z_max, args.n_grid)
    logA_grid = None

    def run_case(name: str, *, kernel: str, cov_kind: str, diagonal_cov: bool, smooth_lambda: float) -> AblationResult:
        sn_use = load_pantheon_plus(paths=paths, cov_kind=cov_kind, subset="cosmology", z_column="zHD")
        sn_like_use = SNLogLike.from_pantheon(sn_use, z_min=z_min, z_max=z_max)
        if diagonal_cov:
            cov = np.diag(np.diag(sn_like_use.cov))
            sn_like_use = SNLogLike.from_arrays(z=sn_like_use.z, m=sn_like_use.m, cov=cov)

        gp_post = reconstruct_H_gp(
            z_knots=z_knots,
            sn_like=sn_like_use,
            cc_like=cc_like,
            bao_likes=bao_likes,
            constants=constants,
            z_grid=z_grid,
            z_max_background=z_max,
            kernel=kernel,
            n_walkers=args.gp_walkers,
            n_steps=args.gp_steps,
            n_burn=args.gp_burn,
            seed=args.seed,
            n_processes=1,
        )

        # spline ablation: smoothing strength
        _ = reconstruct_H_spline(
            z_knots=z_knots,
            sn_like=sn_like_use,
            cc_like=cc_like,
            bao_likes=bao_likes,
            constants=constants,
            z_grid=z_grid,
            z_max_background=z_max,
            smooth_lambda=smooth_lambda,
            n_bootstrap=40,
            seed=args.seed + 1,
        )

        omega_m0 = np.full(gp_post.H_samples.shape[0], float(args.omega_m0))
        A_draws = 4.0 * np.pi * (constants.c_km_s / gp_post.H_samples) ** 2
        logA_min = float(np.max(np.percentile(np.log(A_draws), 2, axis=1)))
        logA_max = float(np.min(np.percentile(np.log(A_draws), 98, axis=1)))
        nonlocal logA_grid
        if logA_grid is None:
            logA_grid = np.linspace(logA_min, logA_max, 120)

        mu_post = reconstruct_logmu_of_logA(
            z=z_grid,
            H_samples=gp_post.H_samples,
            dH_dz_samples=gp_post.dH_dz_samples,
            constants=constants,
            omega_m0_samples=omega_m0,
            logA_grid=logA_grid,
        )
        prox = proximity_summary(logA_grid=logA_grid, logmu_samples=mu_post.logmu_samples)
        return AblationResult(
            name=name,
            kernel=kernel,
            cov_kind=cov_kind,
            diagonal_cov=bool(diagonal_cov),
            smooth_lambda=float(smooth_lambda),
            D2_bh=float(prox["D2_mean"]["bh"]),
            D2_tsallis=float(prox["D2_mean"]["tsallis"]),
            D2_barrow=float(prox["D2_mean"]["barrow"]),
            D2_kaniadakis=float(prox["D2_mean"]["kaniadakis"]),
        )

    cases = [
        ("kernel_rbf", dict(kernel="rbf", cov_kind="stat+sys", diagonal_cov=False, smooth_lambda=8.0)),
        ("kernel_matern32", dict(kernel="matern32", cov_kind="stat+sys", diagonal_cov=False, smooth_lambda=8.0)),
        ("kernel_matern52", dict(kernel="matern52", cov_kind="stat+sys", diagonal_cov=False, smooth_lambda=8.0)),
        ("cov_statonly", dict(kernel="matern32", cov_kind="statonly", diagonal_cov=False, smooth_lambda=8.0)),
        ("cov_diagonal", dict(kernel="matern32", cov_kind="stat+sys", diagonal_cov=True, smooth_lambda=8.0)),
        ("spline_lambda_low", dict(kernel="matern32", cov_kind="stat+sys", diagonal_cov=False, smooth_lambda=3.0)),
        ("spline_lambda_high", dict(kernel="matern32", cov_kind="stat+sys", diagonal_cov=False, smooth_lambda=20.0)),
    ]

    results: list[AblationResult] = []
    for name, cfg in cases:
        print(f"Running ablation {name}...")
        results.append(run_case(name, **cfg))

    rows = []
    for r in results:
        rows.append(
            [
                r.name,
                r.kernel,
                r.cov_kind + (" (diag)" if r.diagonal_cov else ""),
                f"{r.smooth_lambda:.1f}",
                f"{r.D2_bh:.3g}",
                f"{r.D2_tsallis:.3g}",
                f"{r.D2_barrow:.3g}",
                f"{r.D2_kaniadakis:.3g}",
            ]
        )

    (report_paths.tables_dir / "ablations.json").write_text(
        json.dumps([r.__dict__ for r in results], indent=2), encoding="utf-8"
    )

    md = []
    md.append("# Ablation suite\n")
    md.append(
        "This suite probes sensitivity to kernel choice, covariance handling, and spline regularization.\n"
    )
    md.append(format_table(rows, headers=["Case", "GP kernel", "SN cov", "λ", "D² BH", "D² Tsallis", "D² Barrow", "D² Kaniadakis"]))
    write_markdown_report(paths=report_paths, markdown="\n".join(md))
    print(f"Wrote {report_paths.report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

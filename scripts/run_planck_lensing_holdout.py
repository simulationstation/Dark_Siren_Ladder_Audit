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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless/cluster-safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.ingest_planck_lensing_bandpowers import load_planck_lensing_bandpowers
from entropy_horizon_recon.likelihoods_planck_lensing_clpp import PlanckLensingClppLogLike
from entropy_horizon_recon.sirens import load_mu_forward_posterior

_GLOBAL_LIKE: PlanckLensingClppLogLike | None = None
_GLOBAL_LL_CONST: float | None = None


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _detect_mapping_variant(run_dir: Path) -> str:
    """Best-effort detection of mapping variant from the run's tables/summary.json."""
    summary = run_dir / "tables" / "summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text())
            mv = (d.get("settings") or {}).get("mapping_variant")
            if mv:
                return str(mv)
        except Exception:
            pass
    name = run_dir.name
    if "M2" in name:
        return "M2"
    if "M1" in name:
        return "M1"
    return "M0"


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    logw = np.asarray(logw, dtype=float)
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


def _worker_eval(params: tuple[float, float, float, float]) -> tuple[float, float, np.ndarray, str | None]:
    """Return (loglike, chi2, model, error)."""
    like = _GLOBAL_LIKE
    ll_const = _GLOBAL_LL_CONST
    if like is None or ll_const is None:
        raise RuntimeError("Worker globals not initialized.")
    H0, om, ok, s8 = params
    try:
        model = like.predict(H0=float(H0), omega_m0=float(om), omega_k0=float(ok), sigma8_0=float(s8))
        if not np.all(np.isfinite(model)):
            raise ValueError("nonfinite_model")
        r = like.clpp - model
        chi2 = float(r.T @ like.cov_inv @ r)
        ll = float(ll_const - 0.5 * chi2)
        return ll, chi2, np.asarray(model, dtype=float), None
    except Exception as exc:
        model = np.full_like(like.clpp, np.nan, dtype=float)
        return float("-inf"), float("nan"), model, f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class HoldoutRow:
    run_dir: str
    run_id: str
    run: str
    mapping_variant: str
    dataset: str
    n_eval: int
    n_bins: int
    invalid_frac: float
    lpd: float
    chi2_mean: float
    chi2_q16: float
    chi2_q50: float
    chi2_q84: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Planck 2018 CMB lensing C_l^{phiphi} holdout evaluation.")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/planck_lensing_holdout_<UTCSTAMP>).")
    ap.add_argument(
        "--dataset",
        type=str,
        default="consext8",
        choices=["consext8", "agr2"],
        help="Planck lensing bandpower dataset variant.",
    )
    ap.add_argument("--n-eval", type=int, default=0, help="Number of posterior draws to evaluate (0=all).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for draw subsampling.")
    ap.add_argument("--procs", type=int, default=0, help="Multiprocessing workers (0=auto=cpu_count).")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a progress update every N evaluated draws (0 disables).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"planck_lensing_holdout_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    repo_root = Path(__file__).resolve().parents[1]
    paths = DataPaths(repo_root=repo_root)

    bp = load_planck_lensing_bandpowers(paths=paths, dataset=str(args.dataset))
    ell_eff = np.rint(np.asarray(bp.ell_eff, dtype=float)).astype(int)
    like = PlanckLensingClppLogLike.from_data(ell_eff=ell_eff, clpp=bp.clpp, cov=bp.cov, meta=bp.meta)

    sign, logdet = np.linalg.slogdet(like.cov)
    if sign <= 0:
        raise RuntimeError("Planck lensing covariance has non-positive determinant.")
    ll_const = -0.5 * float(logdet + like.clpp.size * np.log(2.0 * np.pi))

    rng = np.random.default_rng(int(args.seed))
    rows: list[dict[str, Any]] = []

    for rd in args.run_dir:
        run_path = Path(rd)
        run_label = run_path.name
        run_id = f"{run_path.parent.name}__{run_label}"
        mapping_variant = _detect_mapping_variant(run_path)

        try:
            post = load_mu_forward_posterior(run_path)
        except Exception as exc:
            rows.append(
                {
                    "run": run_label,
                    "mapping_variant": mapping_variant,
                    "dataset": str(args.dataset),
                    "status": "skipped",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if post.sigma8_0 is None:
            rows.append(
                {
                    "run": run_label,
                    "mapping_variant": mapping_variant,
                    "dataset": str(args.dataset),
                    "status": "skipped",
                    "error": "Missing sigma8_0 in samples/mu_forward_posterior.npz (run must include a sigma8-bearing likelihood).",
                }
            )
            continue

        H0 = np.asarray(post.H0, dtype=float)
        om = np.asarray(post.omega_m0, dtype=float)
        ok = np.asarray(post.omega_k0, dtype=float)
        s8 = np.asarray(post.sigma8_0, dtype=float)
        n_draws = int(H0.size)
        if not (om.size == n_draws and ok.size == n_draws and s8.size == n_draws):
            rows.append(
                {
                    "run": run_label,
                    "mapping_variant": mapping_variant,
                    "dataset": str(args.dataset),
                    "status": "skipped",
                    "error": "Parameter sample arrays have inconsistent lengths.",
                }
            )
            continue

        n_eval = int(args.n_eval) if int(args.n_eval) > 0 else n_draws
        if n_eval > n_draws:
            n_eval = n_draws
        idx = rng.choice(n_draws, size=n_eval, replace=False) if n_eval < n_draws else np.arange(n_draws)

        params = [(float(H0[i]), float(om[i]), float(ok[i]), float(s8[i])) for i in idx]

        # Evaluate CAMB lensing bandpowers for each draw.
        procs = int(args.procs)
        if procs <= 0:
            procs = int(os.cpu_count() or 1)
        procs = max(1, min(int(procs), int(len(params)) if params else 1))

        prog_every = int(args.progress_every)
        t_start = datetime.now(timezone.utc)
        print(
            f"[holdout] run={run_id} mapping={mapping_variant} dataset={args.dataset} "
            f"n_eval={len(params)} procs={procs} start={t_start.isoformat()}",
            flush=True,
        )

        global _GLOBAL_LIKE, _GLOBAL_LL_CONST
        _GLOBAL_LIKE = like
        _GLOBAL_LL_CONST = ll_const
        try:
            if procs == 1:
                out = []
                for i, p in enumerate(params, start=1):
                    out.append(_worker_eval(p))
                    if prog_every > 0 and (i % prog_every == 0 or i == len(params)):
                        dt = (datetime.now(timezone.utc) - t_start).total_seconds()
                        rate = i / dt if dt > 0 else float("nan")
                        eta = (len(params) - i) / rate if rate and rate > 0 else float("nan")
                        print(f"[holdout] {run_id} {i}/{len(params)} ({rate:.3g}/s) ETA~{eta/60.0:.1f} min", flush=True)
            else:
                import multiprocessing as mp

                try:
                    ctx = mp.get_context("fork")
                except ValueError:
                    ctx = mp.get_context()
                out = []
                with ctx.Pool(processes=procs) as pool:
                    it = pool.imap_unordered(_worker_eval, params, chunksize=1)
                    for i, res in enumerate(it, start=1):
                        out.append(res)
                        if prog_every > 0 and (i % prog_every == 0 or i == len(params)):
                            dt = (datetime.now(timezone.utc) - t_start).total_seconds()
                            rate = i / dt if dt > 0 else float("nan")
                            eta = (len(params) - i) / rate if rate and rate > 0 else float("nan")
                            print(
                                f"[holdout] {run_id} {i}/{len(params)} ({rate:.3g}/s) ETA~{eta/60.0:.1f} min",
                                flush=True,
                            )
        finally:
            _GLOBAL_LIKE = None
            _GLOBAL_LL_CONST = None

        ll = np.asarray([x[0] for x in out], dtype=float)
        chi2 = np.asarray([x[1] for x in out], dtype=float)
        models = np.stack([x[2] for x in out], axis=0)
        errors = [x[3] for x in out]
        ll_clean = np.where(np.isfinite(ll), ll, -np.inf)
        ok_mask = np.isfinite(ll)
        invalid_frac = float(np.mean(~ok_mask)) if ll.size else 1.0
        chi2_ok = chi2[np.isfinite(chi2)]
        models_ok = models[ok_mask]

        if not np.any(ok_mask):
            rows.append(
                {
                    "run": run_label,
                    "mapping_variant": mapping_variant,
                    "dataset": str(args.dataset),
                    "status": "failed",
                    "error": "All evaluated draws returned non-finite likelihood.",
                    "invalid_frac": invalid_frac,
                    "first_error": next((e for e in errors if e), None),
                }
            )
            continue

        lpd = float(_logmeanexp(ll_clean))
        qs = np.percentile(chi2_ok, [16, 50, 84]) if chi2_ok.size else [float("nan")] * 3
        row = HoldoutRow(
            run_dir=str(run_path),
            run_id=str(run_id),
            run=run_label,
            mapping_variant=str(mapping_variant),
            dataset=str(args.dataset),
            n_eval=int(ll.size),
            n_bins=int(like.clpp.size),
            invalid_frac=float(invalid_frac),
            lpd=float(lpd),
            chi2_mean=float(np.mean(chi2_ok)) if chi2_ok.size else float("nan"),
            chi2_q16=float(qs[0]),
            chi2_q50=float(qs[1]),
            chi2_q84=float(qs[2]),
        )
        rows.append({**asdict(row), "status": "ok"})

        # Save predictive bands for plotting or later comparisons.
        q_cl = np.percentile(models_ok, [16, 50, 84], axis=0)
        band_npz = tab_dir / f"{run_id}__{args.dataset}__pred_band.npz"
        np.savez_compressed(
            band_npz,
            ell_eff=np.asarray(like.ell_eff, dtype=int),
            clpp_obs=np.asarray(like.clpp, dtype=float),
            clpp_err=np.sqrt(np.diag(like.cov)),
            clpp_pred_q16=np.asarray(q_cl[0], dtype=float),
            clpp_pred_q50=np.asarray(q_cl[1], dtype=float),
            clpp_pred_q84=np.asarray(q_cl[2], dtype=float),
            lpd=float(lpd),
            chi2_mean=float(np.mean(chi2_ok)) if chi2_ok.size else float("nan"),
            chi2_q16=float(qs[0]),
            chi2_q50=float(qs[1]),
            chi2_q84=float(qs[2]),
            invalid_frac=float(invalid_frac),
        )

        # Plot: observed bandpowers + predictive band.
        plt.figure(figsize=(7, 4))
        x = np.asarray(like.ell_eff, dtype=float)
        y = np.asarray(like.clpp, dtype=float)
        yerr = np.sqrt(np.diag(like.cov))
        plt.errorbar(x, y, yerr=yerr, fmt="o", ms=4, capsize=2, label="Planck 2018 lensing (bandpowers)")
        plt.plot(x, q_cl[1], color="C1", lw=1.8, label="Pred median (CAMB)")
        plt.fill_between(x, q_cl[0], q_cl[2], color="C1", alpha=0.25, label="Pred 68% band")
        plt.xlabel(r"$L_{\rm eff}$")
        plt.ylabel(r"$C_L^{\phi\phi}$")
        plt.title(f"{run_label} ({mapping_variant}) holdout vs Planck lensing ({args.dataset})")
        plt.legend(fontsize=8)
        plt.tight_layout()
        out_png = fig_dir / f"clpp_holdout_{run_id}__{args.dataset}.png"
        plt.savefig(out_png, dpi=160)
        plt.close()

        t_done = datetime.now(timezone.utc)
        dt_min = (t_done - t_start).total_seconds() / 60.0
        print(
            f"[holdout] done run={run_id} invalid_frac={invalid_frac:.3g} lpd={lpd:.3f} "
            f"chi2_mean={float(np.mean(chi2_ok)) if chi2_ok.size else float('nan'):.3f} "
            f"elapsed={dt_min:.1f} min",
            flush=True,
        )

    (tab_dir / "results.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

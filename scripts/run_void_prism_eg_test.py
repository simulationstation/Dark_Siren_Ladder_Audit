from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # headless/cluster-safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.void_prism import (
    eg_gr_baseline_from_background,
    load_void_prism_measurements,
    predict_EG_void_from_mu,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


def _logpdf_mvnormal(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """Log N(x | mu, cov) with basic numerical guards (Cholesky)."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if x.shape != mu.shape or cov.shape != (x.size, x.size):
        raise ValueError("Shape mismatch for multivariate normal.")
    # Symmetrize (JSON cov can be slightly asymmetric).
    cov = 0.5 * (cov + cov.T)
    # Jitter for positive definiteness.
    jitter = 1e-12 * np.trace(cov) / max(1, x.size)
    cov_j = cov + np.eye(x.size) * jitter
    L = np.linalg.cholesky(cov_j)
    r = x - mu
    y = np.linalg.solve(L, r)
    maha = float(np.dot(y, y))
    logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
    return -0.5 * (x.size * np.log(2.0 * np.pi) + logdet + maha)


def _bestfit_amplitude(pred: np.ndarray, obs: np.ndarray, cov: np.ndarray) -> float:
    """GLS best-fit scalar amplitude A in obs ~ A * pred (no prior)."""
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    jitter = 1e-12 * np.trace(cov) / max(1, obs.size)
    cov_j = cov + np.eye(obs.size) * jitter
    L = np.linalg.cholesky(cov_j)

    # inv(C) * v via Cholesky.
    def _invC(v: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(L, v)
        return np.linalg.solve(L.T, y)

    iCy = _invC(obs)
    iCp = _invC(pred)
    denom = float(np.dot(pred, iCp))
    if not np.isfinite(denom) or abs(denom) < 1e-30:
        raise ValueError("Degenerate amplitude fit (pred^T C^-1 pred ~ 0).")
    return float(np.dot(pred, iCy) / denom)


def _lpd_from_draws(pred: np.ndarray, obs: np.ndarray, cov: np.ndarray, *, fit_amplitude: bool) -> tuple[float, np.ndarray | None]:
    """Posterior predictive log density for a Gaussian measurement.

    If fit_amplitude=True, fits a per-draw GLS amplitude A_hat and evaluates obs ~ A_hat * pred.
    This is a 'shape-only' scoring mode useful when the overall normalization is uncertain.
    """
    pred = np.asarray(pred, dtype=float)
    obs = np.asarray(obs, dtype=float)
    cov = np.asarray(cov, dtype=float)
    if pred.ndim != 2:
        raise ValueError("pred must be 2D (n_draws, n_dim).")
    if not fit_amplitude:
        logp = np.array([_logpdf_mvnormal(obs, pred[j], cov) for j in range(pred.shape[0])], dtype=float)
        return float(_logmeanexp(logp)), None

    A = np.empty(pred.shape[0], dtype=float)
    logp = np.empty(pred.shape[0], dtype=float)
    for j in range(pred.shape[0]):
        Aj = _bestfit_amplitude(pred[j], obs, cov)
        A[j] = Aj
        logp[j] = _logpdf_mvnormal(obs, Aj * pred[j], cov)
    return float(_logmeanexp(logp)), A


def _detect_mapping_variant(run_dir: Path) -> str:
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


@dataclass(frozen=True)
class PrismRow:
    run: str
    mapping_variant: str
    measurement: str
    embedding: str
    convention: str
    z_eff: float
    n_ell: int
    max_draws: int | None
    params: dict[str, float]
    fit_amplitude: bool
    amp_mean: float | None
    amp_std: float | None
    amp_q16: float | None
    amp_q50: float | None
    amp_q84: float | None
    eg_pred_mean: float
    eg_pred_std: float
    eg_pred_q16: float
    eg_pred_q50: float
    eg_pred_q84: float
    lpd: float
    lpd_gr: float
    delta_lpd_vs_gr: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Void-prism E_G-style holdout test (kappa + kSZ velocity).")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--measurements", required=True, help="JSON file of E_G^void measurements.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/void_prism_eg_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="mu->coupling convention (A: mu(z)/mu0).")
    ap.add_argument(
        "--embedding",
        action="append",
        choices=["minimal", "slip_allowed", "screening_allowed"],
        default=None,
        help="Embedding(s) to score (repeatable). Default: all three.",
    )
    ap.add_argument("--max-draws", type=int, default=5000, help="Subsample posterior draws for speed (default 5000).")
    ap.add_argument(
        "--fit-amplitude",
        action="store_true",
        help="Fit a per-draw scalar amplitude before scoring (shape-only; useful if normalization is unknown).",
    )
    ap.add_argument("--eta0", type=float, default=1.0, help="Slip model eta0 (used for slip_allowed).")
    ap.add_argument("--eta1", type=float, default=0.0, help="Slip model eta1 (used for slip_allowed).")
    ap.add_argument("--env-proxy", type=float, default=0.0, help="Screening env proxy value (used for screening_allowed).")
    ap.add_argument("--env-alpha", type=float, default=0.0, help="Screening alpha (used for screening_allowed).")
    ap.add_argument("--muP-highz", type=float, default=1.0, help="High-z muP value for growth extension (default 1).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_prism_eg_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    meas = load_void_prism_measurements(args.measurements)
    embeddings = args.embedding if args.embedding else ["minimal", "slip_allowed", "screening_allowed"]

    rows: list[dict[str, Any]] = []

    for rd in args.run_dir:
        run_path = Path(rd)
        post = load_mu_forward_posterior(run_path)
        run_label = run_path.name
        mapping_variant = _detect_mapping_variant(run_path)

        for m in meas:
            # Baseline: GR perturbations on the same background draw ensemble.
            eg_gr = eg_gr_baseline_from_background(
                post,
                z_eff=m.z_eff,
                ell=m.ell,
                max_draws=int(args.max_draws) if args.max_draws else None,
            )
            lpd_gr, A_gr = _lpd_from_draws(eg_gr, m.eg_obs, m.cov, fit_amplitude=bool(args.fit_amplitude))

            for emb in embeddings:
                try:
                    eg_pred = predict_EG_void_from_mu(
                        post,
                        z_eff=m.z_eff,
                        ell=m.ell,
                        convention=args.convention,  # type: ignore[arg-type]
                        embedding=emb,  # type: ignore[arg-type]
                        eta0=float(args.eta0),
                        eta1=float(args.eta1),
                        env_proxy=float(args.env_proxy),
                        env_alpha=float(args.env_alpha),
                        muP_highz=float(args.muP_highz),
                        max_draws=int(args.max_draws) if args.max_draws else None,
                    )
                except Exception as e:
                    rows.append(
                        {
                            "run": run_label,
                            "mapping_variant": str(mapping_variant),
                            "measurement": m.name,
                            "embedding": emb,
                            "status": "skipped",
                            "error": str(e),
                        }
                    )
                    continue

                # Summarize a scalar proxy (mean over ell, mean over draws).
                eg_scalar = np.mean(eg_pred, axis=1)
                qs = np.percentile(eg_scalar, [16, 50, 84])
                eg_mean = float(np.mean(eg_scalar))
                eg_std = float(np.std(eg_scalar, ddof=1)) if eg_scalar.size > 1 else float("nan")

                lpd, A = _lpd_from_draws(eg_pred, m.eg_obs, m.cov, fit_amplitude=bool(args.fit_amplitude))
                amp_stats = {"mean": None, "std": None, "q16": None, "q50": None, "q84": None}
                if A is not None and A.size > 0:
                    amp_stats = {
                        "mean": float(np.mean(A)),
                        "std": float(np.std(A, ddof=1)) if A.size > 1 else float("nan"),
                        "q16": float(np.percentile(A, 16)),
                        "q50": float(np.percentile(A, 50)),
                        "q84": float(np.percentile(A, 84)),
                    }

                params: dict[str, float] = {}
                if emb == "slip_allowed":
                    params = {"eta0": float(args.eta0), "eta1": float(args.eta1)}
                elif emb == "screening_allowed":
                    params = {"env_proxy": float(args.env_proxy), "env_alpha": float(args.env_alpha)}

                row = PrismRow(
                    run=run_label,
                    mapping_variant=str(mapping_variant),
                    measurement=m.name,
                    embedding=emb,
                    convention=str(args.convention),
                    z_eff=float(m.z_eff),
                    n_ell=int(m.ell.size),
                    max_draws=int(args.max_draws) if args.max_draws else None,
                    params=params,
                    fit_amplitude=bool(args.fit_amplitude),
                    amp_mean=amp_stats["mean"],
                    amp_std=amp_stats["std"],
                    amp_q16=amp_stats["q16"],
                    amp_q50=amp_stats["q50"],
                    amp_q84=amp_stats["q84"],
                    eg_pred_mean=eg_mean,
                    eg_pred_std=eg_std,
                    eg_pred_q16=float(qs[0]),
                    eg_pred_q50=float(qs[1]),
                    eg_pred_q84=float(qs[2]),
                    lpd=float(lpd),
                    lpd_gr=float(lpd_gr),
                    delta_lpd_vs_gr=float(lpd - lpd_gr),
                )
                rows.append({**asdict(row), "status": "ok", "notes": m.notes, "env": m.env, "source": m.source})

                # Plot: observed EG(ell) with predicted band.
                if args.fit_amplitude and A is not None:
                    eg_plot = eg_pred * A.reshape((-1, 1))
                else:
                    eg_plot = eg_pred
                q_ell = np.percentile(eg_plot, [16, 50, 84], axis=0)
                plt.figure(figsize=(7.2, 4.2))
                x = m.ell
                plt.fill_between(x, q_ell[0], q_ell[2], color="C0", alpha=0.18, label="pred 16-84%")
                plt.plot(x, q_ell[1], color="C0", lw=1.6, label="pred median")
                obs_sig = np.sqrt(np.diag(m.cov))
                plt.errorbar(x, m.eg_obs, yerr=obs_sig, fmt="o", ms=4.0, color="C3", label="obs")
                plt.xlabel(r"$\ell$")
                plt.ylabel(r"$\hat E_G^{\rm void}$ (normalized)")
                title = f"{run_label} ({mapping_variant}) | {m.name} | {emb} z_eff={m.z_eff:.3f}"
                if args.fit_amplitude:
                    title += " | amp-fit"
                plt.title(title)
                plt.legend(fontsize=8)
                plt.tight_layout()
                out_png = fig_dir / f"eg_{run_label}__{m.name}__{emb}__conv{args.convention}.png"
                plt.savefig(out_png, dpi=170)
                plt.close()

    (tab_dir / "results.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

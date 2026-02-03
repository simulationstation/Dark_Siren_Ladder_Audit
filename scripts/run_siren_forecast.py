from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior, predict_dL_gw


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _logpdf_normal(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sig <= 0:
        raise ValueError("sig must be positive.")
    return -np.log(sig) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((x - mu) / sig) ** 2


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


@dataclass(frozen=True)
class ForecastRow:
    n_events: int
    z_min: float
    z_max: float
    sigma_frac: float
    truth: str
    convention: str
    n_sims: int
    delta_lpd_mean: float
    delta_lpd_std: float
    delta_lpd_p50: float
    frac_delta_gt0: float


def _simulate_delta_lpd(
    *,
    rng: np.random.Generator,
    dL_gw_pred: np.ndarray,  # (n_draws, n_events)
    dL_em_pred: np.ndarray,  # (n_draws, n_events)
    dL_true: np.ndarray,  # (n_events,)
    sigma_frac: float,
    n_sims: int,
) -> np.ndarray:
    if sigma_frac <= 0:
        raise ValueError("sigma_frac must be positive.")
    n_draws, n_events = dL_gw_pred.shape
    if dL_em_pred.shape != (n_draws, n_events):
        raise ValueError("dL_em_pred shape mismatch.")
    if dL_true.shape != (n_events,):
        raise ValueError("dL_true shape mismatch.")

    sig = sigma_frac * dL_true

    deltas = np.empty(int(n_sims), dtype=float)
    for i in range(int(n_sims)):
        obs = dL_true + sig * rng.normal(size=n_events)

        # For each posterior draw, compute event log-likelihood sum.
        lp_mu = np.zeros(n_draws, dtype=float)
        lp_gr = np.zeros(n_draws, dtype=float)
        for k in range(n_events):
            lp_mu += _logpdf_normal(dL_gw_pred[:, k], mu=float(obs[k]), sig=float(sig[k]))
            lp_gr += _logpdf_normal(dL_em_pred[:, k], mu=float(obs[k]), sig=float(sig[k]))
        lpd_mu = float(_logmeanexp(lp_mu))
        lpd_gr = float(_logmeanexp(lp_gr))
        deltas[i] = lpd_mu - lpd_gr
    return deltas


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic standard-siren forecast (mu-propagation vs GR baseline).")
    ap.add_argument("--run-dir", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/siren_forecast_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A")
    ap.add_argument("--truth", choices=["mu", "gr"], default="mu", help="Which model generates the mock data.")
    ap.add_argument("--z-min", type=float, default=0.05)
    ap.add_argument("--z-max", type=float, default=None, help="Default: run z_max.")
    ap.add_argument("--sigma-frac", type=float, default=0.1, help="Fractional distance error per event.")
    ap.add_argument("--n-events", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    ap.add_argument("--n-sims", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_forecast_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    post = load_mu_forward_posterior(args.run_dir)
    z_max = float(args.z_max) if args.z_max is not None else float(post.z_grid[-1])
    z_min = float(args.z_min)
    if z_min <= 0 or z_max <= z_min:
        raise ValueError("Invalid z range.")

    rng = np.random.default_rng(int(args.seed))

    rows: list[ForecastRow] = []
    for n_events in sorted(set(int(x) for x in args.n_events)):
        z_events = np.linspace(z_min, z_max, int(n_events))
        dL_gw_pred, R = predict_dL_gw(
            post,
            z_eval=z_events,
            convention=args.convention,  # type: ignore[arg-type]
        )
        dL_em_pred = dL_gw_pred / R

        if args.truth == "mu":
            dL_true = np.median(dL_gw_pred, axis=0)
        else:
            dL_true = np.median(dL_em_pred, axis=0)

        deltas = _simulate_delta_lpd(
            rng=rng,
            dL_gw_pred=dL_gw_pred,
            dL_em_pred=dL_em_pred,
            dL_true=dL_true,
            sigma_frac=float(args.sigma_frac),
            n_sims=int(args.n_sims),
        )

        rows.append(
            ForecastRow(
                n_events=int(n_events),
                z_min=float(z_min),
                z_max=float(z_max),
                sigma_frac=float(args.sigma_frac),
                truth=str(args.truth),
                convention=str(args.convention),
                n_sims=int(args.n_sims),
                delta_lpd_mean=float(np.mean(deltas)),
                delta_lpd_std=float(np.std(deltas, ddof=1)) if deltas.size > 1 else 0.0,
                delta_lpd_p50=float(np.percentile(deltas, 50)),
                frac_delta_gt0=float(np.mean(deltas > 0.0)),
            )
        )

    (tab_dir / "forecast.json").write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")

    # Plot: mean delta_LPD vs N.
    plt.figure(figsize=(7, 4))
    Ns = np.array([r.n_events for r in rows], dtype=float)
    means = np.array([r.delta_lpd_mean for r in rows], dtype=float)
    stds = np.array([r.delta_lpd_std for r in rows], dtype=float)
    plt.errorbar(Ns, means, yerr=stds, fmt="o-", capsize=3)
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
    plt.xscale("log")
    plt.xlabel("Number of bright sirens (N)")
    plt.ylabel("Delta_LPD = LPD(mu-prop) - LPD(GR)")
    plt.title(f"Forecast (truth={args.truth}, sigma_frac={args.sigma_frac:.2f}, convention={args.convention})")
    plt.tight_layout()
    plt.savefig(fig_dir / "delta_lpd_vs_n.png", dpi=170)
    plt.close()

    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


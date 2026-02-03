from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.departure import compute_departure_stats


def _load_chain(chain_path: Path) -> dict:
    with np.load(chain_path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _compute_logA_grid(sample_npz: Path, n_logA: int = 140) -> np.ndarray:
    const = PhysicalConstants()
    with np.load(sample_npz, allow_pickle=True) as d:
        H_samples = d["H_samples"]
    A_draws = 4.0 * np.pi * (const.c_km_s / H_samples) ** 2
    logA_min = float(np.max(np.percentile(np.log(A_draws), 2, axis=1)))
    logA_max = float(np.min(np.percentile(np.log(A_draws), 98, axis=1)))
    return np.linspace(logA_min, logA_max, n_logA)


def _logmu_from_knots(logmu_knots: np.ndarray, x_knots: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    spline = CubicSpline(x_knots, logmu_knots, bc_type="natural", extrapolate=True)
    return spline(x_eval)


def _m_trace(chain: np.ndarray, param_names: list[str], x_knots: np.ndarray, logA_grid: np.ndarray, H0_vals: np.ndarray) -> np.ndarray:
    # Compute weights from posterior draws (as in compute_departure_stats)
    # Use a small subset of chain to estimate weights to limit cost
    n_steps, n_walkers, _ = chain.shape
    idx = np.linspace(0, n_steps - 1, min(n_steps, 200), dtype=int)
    logmu_samples = []
    for t in idx:
        for w in range(n_walkers):
            th = chain[t, w]
            logmu_knots = th[: len(x_knots)]
            logA0 = np.log(4.0 * np.pi * (PhysicalConstants().c_km_s / H0_vals[t, w]) ** 2)
            x_eval = np.clip(logA_grid - logA0, x_knots[0], x_knots[-1])
            logmu_samples.append(_logmu_from_knots(logmu_knots, x_knots, x_eval))
    logmu_samples = np.asarray(logmu_samples)
    # compute weights w(logA)
    mean = np.mean(logmu_samples, axis=0)
    var = np.var(logmu_samples, axis=0, ddof=1)
    w = 1.0 / np.clip(var, 1e-12, np.inf)
    w = w / np.trapezoid(w, x=logA_grid)

    m_trace = np.empty((n_steps, n_walkers))
    for t in range(n_steps):
        for w_i in range(n_walkers):
            th = chain[t, w_i]
            logmu_knots = th[: len(x_knots)]
            logA0 = np.log(4.0 * np.pi * (PhysicalConstants().c_km_s / H0_vals[t, w_i]) ** 2)
            x_eval = np.clip(logA_grid - logA0, x_knots[0], x_knots[-1])
            logmu = _logmu_from_knots(logmu_knots, x_knots, x_eval)
            m_trace[t, w_i] = np.trapezoid(logmu * w, x=logA_grid)
    return m_trace


def _ess_tau(x: np.ndarray) -> tuple[float, float]:
    import emcee

    try:
        tau = emcee.autocorr.integrated_time(x, quiet=True)
    except Exception:
        return float("nan"), float("nan")
    tau = float(np.max(tau)) if np.ndim(tau) else float(tau)
    n = x.size
    ess = n / tau if tau and tau > 0 else float("nan")
    return tau, ess


def main() -> int:
    parser = argparse.ArgumentParser(description="Finalization diagnostics for ptemcee runs.")
    parser.add_argument("--base", type=Path, required=True, help="Base dir containing M0_start* outputs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=101, help="Seed to use for trace plot")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    ess_tau = {}
    swap = {}
    for run_dir in sorted(args.base.glob("M0_start*/")):
        seed = run_dir.name.replace("M0_start", "")
        chain_path = run_dir / "samples" / "mu_chain.npz"
        meta_path = run_dir / "samples" / "mu_forward_meta.json"
        post_path = run_dir / "samples" / "mu_forward_posterior.npz"
        if not chain_path.exists() or not meta_path.exists() or not post_path.exists():
            continue
        chain_data = _load_chain(chain_path)
        chain = chain_data["chain"]
        param_names = [str(x) for x in chain_data["param_names"]]
        x_knots = chain_data["x_knots"]

        # Scalars
        idx_H0 = param_names.index("u_H0")
        idx_om = param_names.index("u_omega_m0") if "u_omega_m0" in param_names else None
        idx_s8 = param_names.index("u_sigma8") if "u_sigma8" in param_names else None
        H0_vals = chain[:, :, idx_H0]
        H0_vals = 40.0 + (100.0 - 40.0) * (1.0 / (1.0 + np.exp(-H0_vals)))
        if idx_om is not None:
            u_om = chain[:, :, idx_om]
            om_vals = 0.2 + (0.4 - 0.2) * (1.0 / (1.0 + np.exp(-u_om)))
        else:
            om_vals = None
        if idx_s8 is not None:
            s8_vals = chain[:, :, idx_s8]
        else:
            s8_vals = None

        # m trace
        logA_grid = _compute_logA_grid(post_path)
        m_trace = _m_trace(chain, param_names, x_knots, logA_grid, H0_vals)

        # flatten for tau
        m_flat = m_trace.reshape(-1)
        H0_flat = H0_vals.reshape(-1)
        om_flat = om_vals.reshape(-1) if om_vals is not None else None
        s8_flat = s8_vals.reshape(-1) if s8_vals is not None else None

        tau_m, ess_m = _ess_tau(m_flat)
        tau_H0, ess_H0 = _ess_tau(H0_flat)
        tau_om, ess_om = _ess_tau(om_flat) if om_flat is not None else (float("nan"), float("nan"))
        tau_s8, ess_s8 = _ess_tau(s8_flat) if s8_flat is not None else (float("nan"), float("nan"))

        ess_tau[seed] = {
            "tau_m": tau_m,
            "ess_m": ess_m,
            "tau_H0": tau_H0,
            "ess_H0": ess_H0,
            "tau_omega_m0": tau_om,
            "ess_omega_m0": ess_om,
            "tau_sigma8": tau_s8,
            "ess_sigma8": ess_s8,
        }

        meta = json.loads(meta_path.read_text())
        swap[seed] = meta.get("meta", {}).get("sampler_extra", {}).get("ptemcee", {})

        # trace plot for one seed
        if str(seed) == str(args.seed):
            fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axes[0].plot(np.mean(m_trace, axis=1), lw=1.0)
            axes[0].set_ylabel("m (mean over walkers)")
            axes[1].plot(np.mean(om_vals, axis=1) if om_vals is not None else np.zeros(chain.shape[0]), lw=1.0)
            axes[1].set_ylabel("omega_m0 (u space)")
            axes[1].set_xlabel("step")
            fig.tight_layout()
            fig.savefig(out_dir / "trace_m_om0.png", dpi=160)
            plt.close(fig)

    (out_dir / "ess_tau.json").write_text(json.dumps(ess_tau, indent=2), encoding="utf-8")
    (out_dir / "ptemcee_swaps.json").write_text(json.dumps(swap, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

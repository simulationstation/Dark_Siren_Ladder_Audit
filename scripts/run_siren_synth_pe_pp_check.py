from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

try:
    import matplotlib

    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAVE_MPL = False

import numpy as np

from entropy_horizon_recon.gwtc_pe_priors import parse_gwtc_analytic_prior
from entropy_horizon_recon.siren_injection_recovery import (
    InjectionRecoveryConfig,
    SyntheticEventTruth,
    generate_synthetic_detected_events_from_injections,
    load_injections_for_recovery,
    synthesize_pe_posterior_samples,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _sample_truncnorm(
    rng: np.random.Generator,
    *,
    mu: float,
    sigma: float,
    lo: float,
    hi: float,
) -> float:
    # Rejection sampling; fine for scalar.
    for _ in range(200):
        x = float(rng.normal(loc=float(mu), scale=float(sigma)))
        if lo <= x <= hi:
            return x
    return float(np.clip(float(mu), float(lo), float(hi)))


def _log_sigma_for_truth(truth: SyntheticEventTruth, cfg: InjectionRecoveryConfig) -> tuple[float, float, float]:
    if cfg.dl_sigma_mode == "constant":
        dl_log_sigma = float(cfg.dl_frac_sigma0)
    else:
        dl_log_sigma = float(cfg.dl_frac_sigma0) / max(float(truth.snr_net_opt_true), 1e-6)
        dl_log_sigma = max(dl_log_sigma, float(cfg.dl_frac_sigma_floor))
    dl_log_sigma = float(np.clip(dl_log_sigma, 1e-4, 2.0))

    mc_log_sigma = float(np.clip(float(cfg.mc_frac_sigma0) / max(float(truth.snr_net_opt_true), 1e-6), 1e-4, 0.5))
    q_sigma = float(np.clip(float(cfg.q_sigma0), 1e-4, 1.0))
    return dl_log_sigma, mc_log_sigma, q_sigma


def _rank(x_true: float, xs: np.ndarray) -> float:
    xs = np.asarray(xs, dtype=float)
    m = np.isfinite(xs)
    if not np.any(m):
        return float("nan")
    return float(np.mean((xs[m] <= float(x_true)).astype(float)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Synthetic PE generator audit via P–P / rank checks.")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/synth_pe_pp_<UTCSTAMP>).")

    ap.add_argument("--truth-mode", choices=["pe_prior", "injections_detected"], default="pe_prior")
    ap.add_argument("--n-trials", type=int, default=100, help="Number of trials (default 100).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--simulate-observation", action="store_true", help="Draw an observation around truth before generating posterior (recommended).")

    # For truth-mode=injections_detected.
    ap.add_argument("--selection-injections-hdf", default=None, help="Path to O3 injection file (required for truth-mode=injections_detected).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")
    ap.add_argument("--h0-true", type=float, default=70.0, help="H0 used for dL(z) mapping when sampling detected injections (default 70).")
    ap.add_argument("--z-max", type=float, default=0.62, help="Max z for truth sampling (default 0.62).")
    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--det-model", choices=["threshold", "snr_binned", "snr_mchirp_binned"], default="snr_binned")
    ap.add_argument("--snr-binned-nbins", type=int, default=200)
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20)
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf")
    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform")
    ap.add_argument("--pop-z-k", type=float, default=0.0)
    ap.add_argument("--pop-mass-mode", choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"], default="powerlaw_peak_q_smooth")
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3)
    ap.add_argument("--pop-m-min", type=float, default=5.0)
    ap.add_argument("--pop-m-max", type=float, default=80.0)
    ap.add_argument("--pop-q-beta", type=float, default=0.0)
    ap.add_argument("--pop-m-taper-delta", type=float, default=3.0)
    ap.add_argument("--pop-m-peak", type=float, default=35.0)
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0)
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1)

    # For truth-mode=pe_prior: set a representative SNR for noise widths.
    ap.add_argument("--snr-for-noise", type=float, default=12.0, help="SNR used for noise widths when truth-mode=pe_prior (default 12).")

    # Synthetic PE config knobs.
    ap.add_argument("--pe-n-samples", type=int, default=4000, help="Posterior samples per trial (default 4000).")
    ap.add_argument("--pe-prior-resample-n-candidates", type=int, default=80_000, help="Candidates for prior_resample (default 80000).")
    ap.add_argument(
        "--pe-synth-mode",
        choices=["naive_gaussian", "prior_resample", "likelihood_resample"],
        default="likelihood_resample",
        help="Synthetic PE synthesis mode (default likelihood_resample).",
    )
    ap.add_argument("--dl-frac-sigma0", type=float, default=0.25)
    ap.add_argument("--dl-frac-sigma-floor", type=float, default=0.05)
    ap.add_argument("--dl-sigma-mode", choices=["constant", "snr"], default="snr")
    ap.add_argument("--mc-frac-sigma0", type=float, default=0.02)
    ap.add_argument("--q-sigma0", type=float, default=0.08)
    ap.add_argument("--pe-prior-dl-expr", default="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)")
    ap.add_argument("--pe-prior-chirp-mass-expr", default="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)")
    ap.add_argument("--pe-prior-mass-ratio-expr", default="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)")

    ap.add_argument("--smoke", action="store_true", help="Tiny smoke (n_trials<=20, fewer samples/candidates).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"synth_pe_pp_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if _HAVE_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    n_trials = int(args.n_trials)
    pe_n = int(args.pe_n_samples)
    pe_cand = int(args.pe_prior_resample_n_candidates)
    if bool(args.smoke):
        n_trials = min(n_trials, 20)
        pe_n = min(pe_n, 2000)
        pe_cand = min(pe_cand, 50_000)

    cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        det_model=str(args.det_model),  # type: ignore[arg-type]
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
        selection_ifar_thresh_yr=float(args.selection_ifar_thresh_yr),
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=float(args.pop_m_taper_delta),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        weight_mode=str(args.weight_mode),  # type: ignore[arg-type]
        pe_n_samples=int(pe_n),
        pe_synth_mode=str(args.pe_synth_mode),  # type: ignore[arg-type]
        pe_prior_resample_n_candidates=int(pe_cand),
        pe_seed=0,
        dl_frac_sigma0=float(args.dl_frac_sigma0),
        dl_frac_sigma_floor=float(args.dl_frac_sigma_floor),
        dl_sigma_mode=str(args.dl_sigma_mode),  # type: ignore[arg-type]
        mc_frac_sigma0=float(args.mc_frac_sigma0),
        q_sigma0=float(args.q_sigma0),
        pe_prior_dL_expr=str(args.pe_prior_dl_expr),
        pe_prior_chirp_mass_expr=str(args.pe_prior_chirp_mass_expr),
        pe_prior_mass_ratio_expr=str(args.pe_prior_mass_ratio_expr),
        event_qc_mode="skip",
        event_min_finite_frac=0.0,
    )

    truth_mode: Literal["pe_prior", "injections_detected"] = str(args.truth_mode)  # type: ignore[assignment]
    rng = np.random.default_rng(int(args.seed))

    truths: list[SyntheticEventTruth] = []
    if truth_mode == "injections_detected":
        if args.selection_injections_hdf is None:
            raise ValueError("--selection-injections-hdf is required for truth-mode=injections_detected")
        injections = load_injections_for_recovery(args.selection_injections_hdf, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
        truths = generate_synthetic_detected_events_from_injections(
            injections=injections,
            cfg=cfg,
            n_events=int(n_trials),
            seed=int(args.seed),
        )
    else:
        # Sample truth from the PE prior for an internal consistency (SBC-like) check.
        _, prior_dL = parse_gwtc_analytic_prior(str(cfg.pe_prior_dL_expr))
        _, prior_mc = parse_gwtc_analytic_prior(str(cfg.pe_prior_chirp_mass_expr))
        _, prior_q = parse_gwtc_analytic_prior(str(cfg.pe_prior_mass_ratio_expr))
        snr = float(args.snr_for_noise)
        if not (np.isfinite(snr) and snr > 0.0):
            raise ValueError("--snr-for-noise must be finite and >0")
        for i in range(int(n_trials)):
            dL_true = float(prior_dL.sample(rng, size=1)[0])
            mc_true = float(prior_mc.sample(rng, size=1)[0])
            q_true = float(prior_q.sample(rng, size=1)[0])
            truths.append(
                SyntheticEventTruth(
                    event=f"PP_{i+1:05d}",
                    z=0.0,
                    dL_mpc_true=dL_true,
                    m1_source=30.0,
                    m2_source=20.0,
                    chirp_mass_det_true=mc_true,
                    mass_ratio_true=q_true,
                    snr_net_opt_fid=float(snr),
                    dL_mpc_fid=float(dL_true),
                    snr_net_opt_true=float(snr),
                    p_det_true=1.0,
                )
            )

    # Main loop: generate posterior samples, compute ranks for the true values.
    rows: list[dict[str, Any]] = []
    for i, truth in enumerate(truths):
        # Derive an observation (optional).
        dl_log_sigma, mc_log_sigma, q_sigma = _log_sigma_for_truth(truth, cfg)
        if bool(args.simulate_observation):
            log_dL_obs = float(rng.normal(loc=np.log(float(truth.dL_mpc_true)), scale=float(dl_log_sigma)))
            dL_obs = float(np.exp(log_dL_obs))
            log_mc_obs = float(rng.normal(loc=np.log(float(truth.chirp_mass_det_true)), scale=float(mc_log_sigma)))
            mc_obs = float(np.exp(log_mc_obs))
            q_obs = _sample_truncnorm(rng, mu=float(truth.mass_ratio_true), sigma=float(q_sigma), lo=0.05, hi=1.0)
        else:
            dL_obs = float(truth.dL_mpc_true)
            mc_obs = float(truth.chirp_mass_det_true)
            q_obs = float(truth.mass_ratio_true)

        like_center = replace(truth, dL_mpc_true=dL_obs, chirp_mass_det_true=mc_obs, mass_ratio_true=q_obs)
        pe = synthesize_pe_posterior_samples(truth=like_center, cfg=cfg, rng=rng)

        rows.append(
            {
                "trial": int(i + 1),
                "event": str(truth.event),
                "truth_dL_mpc": float(truth.dL_mpc_true),
                "truth_mc_det": float(truth.chirp_mass_det_true),
                "truth_q": float(truth.mass_ratio_true),
                "obs_dL_mpc": float(dL_obs),
                "obs_mc_det": float(mc_obs),
                "obs_q": float(q_obs),
                "snr": float(truth.snr_net_opt_true),
                "rank_dL": _rank(float(truth.dL_mpc_true), pe.dL_mpc),
                "rank_mc_det": _rank(float(truth.chirp_mass_det_true), pe.chirp_mass_det),
                "rank_q": _rank(float(truth.mass_ratio_true), pe.mass_ratio),
            }
        )
        if (i + 1) % 10 == 0 or (i + 1) == len(truths):
            print(f"[pp] trial {i+1}/{len(truths)}", flush=True)

    # Summaries.
    def _summ(key: str) -> dict[str, float]:
        x = np.asarray([float(r.get(key, float("nan"))) for r in rows], dtype=float)
        return {"mean": float(np.nanmean(x)), "sd": float(np.nanstd(x)), "p05": float(np.nanquantile(x, 0.05)), "p50": float(np.nanquantile(x, 0.50)), "p95": float(np.nanquantile(x, 0.95))}

    summary = {
        "truth_mode": truth_mode,
        "simulate_observation": bool(args.simulate_observation),
        "n_trials": int(len(rows)),
        "rank_dL": _summ("rank_dL"),
        "rank_mc_det": _summ("rank_mc_det"),
        "rank_q": _summ("rank_q"),
        "note": "Uniform rank histograms are expected only for an SBC-style setup (truth drawn from the same prior used in posterior and observation noise simulated).",
    }

    manifest = {
        "created_utc": _utc_stamp(),
        "seed": int(args.seed),
        "cfg": cfg.__dict__,
        "truth_mode": truth_mode,
        "simulate_observation": bool(args.simulate_observation),
        "n_trials": int(n_trials),
        "pe_n_samples": int(pe_n),
        "pe_synth_mode": str(args.pe_synth_mode),
        "pe_prior_resample_n_candidates": int(pe_cand),
    }
    _write_json(out_dir / "manifest.json", manifest)
    _write_json(out_dir / "summary.json", summary)
    _write_json(out_dir / "ranks.json", rows)

    if _HAVE_MPL:
        assert plt is not None
        for key, title in [("rank_dL", "dL"), ("rank_mc_det", "chirp mass (det)"), ("rank_q", "mass ratio q")]:
            x = np.asarray([float(r[key]) for r in rows], dtype=float)
            plt.figure(figsize=(6.4, 3.6))
            plt.hist(np.clip(x, 0.0, 1.0), bins=20, range=(0.0, 1.0), alpha=0.85)
            plt.axhline(float(len(x)) / 20.0, color="k", lw=1.0, ls="--")
            plt.xlabel("rank / percentile of truth in posterior")
            plt.ylabel("count")
            plt.title(f"P–P / rank check: {title}")
            plt.tight_layout()
            plt.savefig(fig_dir / f"pp_rank_{key}.png", dpi=160)
            plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

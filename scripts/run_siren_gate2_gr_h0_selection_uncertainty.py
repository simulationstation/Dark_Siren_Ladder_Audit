from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_siren_h0 import compute_alpha_h0_grid_pdet_marginalized
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections
from entropy_horizon_recon.hubble_tension import GaussianPrior
from entropy_horizon_recon.hubble_tension import bayes_factor_between_priors_from_uniform_posterior
from entropy_horizon_recon.hubble_tension import integrate_posterior_prob
from entropy_horizon_recon.hubble_tension import normalize_pdf_grid
from entropy_horizon_recon.hubble_tension import posterior_quantiles


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _as_1d(obj: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1D array with >=2 entries.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _maybe_path(s: str | None) -> Path | None:
    if s is None:
        return None
    p = Path(s).expanduser()
    return p.resolve() if p.exists() else None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Gate-2 GR H0: marginalize selection-alpha uncertainty via p_det bin/cell Beta draws."
    )
    ap.add_argument(
        "--gate2-json",
        required=True,
        help="Path to a Gate-2 JSON output (e.g. gr_h0_selection_on_inv_sampling_pdf.json). Must contain H0_grid and logL_sum_events_rel.",
    )
    ap.add_argument(
        "--selection-injections-hdf",
        default=None,
        help="Override selection injection file path. If omitted, tries to use selection_injections_hdf recorded in the Gate-2 JSON (if present).",
    )
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=None, help="Override IFAR threshold (years).")

    ap.add_argument(
        "--inj-mass-pdf-coords",
        choices=["m1m2", "m1q"],
        default=None,
        help="Override injection mass-coordinate convention for sampling_pdf (default: use Gate-2 JSON).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-dist",
        choices=["z", "dL", "log_dL"],
        default=None,
        help="Override injection distance/redshift convention for sampling_pdf (default: use Gate-2 JSON).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-frame",
        choices=["source", "detector"],
        default=None,
        help="Override injection mass-frame convention for sampling_pdf (default: use Gate-2 JSON).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-scale",
        choices=["linear", "log"],
        default=None,
        help="Override injection mass-scale convention for sampling_pdf (default: use Gate-2 JSON).",
    )

    ap.add_argument("--n-pdet-draws", type=int, default=200, help="Number of p_det draws for marginalization (default 200).")
    ap.add_argument("--pdet-pseudocount", type=float, default=1.0, help="Beta prior pseudocount per cell/bin (default 1.0).")
    ap.add_argument("--pdet-draw-seed", type=int, default=0, help="RNG seed for p_det draws (default 0).")
    ap.add_argument(
        "--save-alpha-draws-npz",
        action="store_true",
        help="Save alpha(H0) draws as NPZ (can be large for big grids/draw counts).",
    )

    ap.add_argument("--h0-lo", type=float, default=68.0, help="Lower threshold for P(H0 < h0_lo) (default 68).")
    ap.add_argument("--h0-hi", type=float, default=72.0, help="Upper threshold for P(H0 > h0_hi) (default 72).")
    ap.add_argument("--planck-mean", type=float, default=67.4, help="Planck-like prior mean (default 67.4).")
    ap.add_argument("--planck-sigma", type=float, default=0.5, help="Planck-like prior sigma (default 0.5).")
    ap.add_argument("--shoes-mean", type=float, default=73.0, help="SH0ES-like prior mean (default 73.0).")
    ap.add_argument("--shoes-sigma", type=float, default=1.0, help="SH0ES-like prior sigma (default 1.0).")

    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate2_seluncert_<UTCSTAMP>).")
    args = ap.parse_args()

    gate2_path = Path(args.gate2_json).expanduser().resolve()
    gate2 = json.loads(gate2_path.read_text())
    if not isinstance(gate2, dict):
        raise ValueError("--gate2-json must be an object.")

    H0_grid = _as_1d(gate2.get("H0_grid"), name="H0_grid")
    logL_sum_events_rel = _as_1d(gate2.get("logL_sum_events_rel"), name="logL_sum_events_rel")
    if logL_sum_events_rel.shape != H0_grid.shape:
        raise ValueError("logL_sum_events_rel must match H0_grid.")

    n_events = int(gate2.get("n_events", 0))
    if n_events <= 0:
        raise ValueError("Gate-2 JSON has invalid n_events.")

    omega_m0 = float(gate2.get("omega_m0", 0.31))
    omega_k0 = float(gate2.get("omega_k0", 0.0))
    z_max = float(gate2.get("z_max", 0.0))
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("Gate-2 JSON has invalid z_max.")

    sel_meta = gate2.get("selection_alpha") or {}
    if not isinstance(sel_meta, dict):
        sel_meta = {}

    det_model = str(sel_meta.get("det_model", gate2.get("det_model", "snr_mchirp_binned")))
    if det_model not in ("snr_binned", "snr_mchirp_binned", "snr_mchirp_q_binned"):
        raise ValueError("This marginalizer supports only det_model in {'snr_binned','snr_mchirp_binned','snr_mchirp_q_binned'}.")

    snr_threshold = sel_meta.get("snr_threshold", None)
    snr_threshold_f = float(snr_threshold) if snr_threshold is not None else None
    snr_binned_nbins = int(sel_meta.get("snr_binned_nbins", 200))
    mchirp_binned_nbins = int(sel_meta.get("mchirp_binned_nbins", 20))
    q_binned_raw = sel_meta.get("q_binned_nbins", None)
    q_binned_nbins = int(q_binned_raw) if q_binned_raw is not None else 10
    weight_mode = str(sel_meta.get("weight_mode", "inv_sampling_pdf"))
    pop = gate2.get("population") or {}
    if not isinstance(pop, dict):
        pop = {}

    pop_z_mode = str(pop.get("pop_z_mode", sel_meta.get("pop_z_mode", "none")))
    pop_z_k = float(pop.get("pop_z_k", sel_meta.get("pop_z_k", 0.0)))
    pop_mass_mode = str(pop.get("pop_mass_mode", sel_meta.get("pop_mass_mode", "none")))
    inj_mass_pdf_coords = (
        str(args.inj_mass_pdf_coords) if args.inj_mass_pdf_coords is not None else str(gate2.get("inj_mass_pdf_coords", sel_meta.get("inj_mass_pdf_coords", "m1m2")))
    )
    inj_sampling_pdf_dist = (
        str(args.inj_sampling_pdf_dist) if args.inj_sampling_pdf_dist is not None else str(gate2.get("inj_sampling_pdf_dist", sel_meta.get("inj_sampling_pdf_dist", "z")))
    )
    inj_sampling_pdf_mass_frame = (
        str(args.inj_sampling_pdf_mass_frame)
        if args.inj_sampling_pdf_mass_frame is not None
        else str(gate2.get("inj_sampling_pdf_mass_frame", sel_meta.get("inj_sampling_pdf_mass_frame", "source")))
    )
    inj_sampling_pdf_mass_scale = (
        str(args.inj_sampling_pdf_mass_scale)
        if args.inj_sampling_pdf_mass_scale is not None
        else str(gate2.get("inj_sampling_pdf_mass_scale", sel_meta.get("inj_sampling_pdf_mass_scale", "linear")))
    )

    pop_m1_alpha = float(pop.get("pop_m1_alpha", sel_meta.get("pop_m1_alpha", 2.3)))
    pop_m_min = float(pop.get("pop_m_min", sel_meta.get("pop_m_min", 5.0)))
    pop_m_max = float(pop.get("pop_m_max", sel_meta.get("pop_m_max", 80.0)))
    pop_q_beta = float(pop.get("pop_q_beta", sel_meta.get("pop_q_beta", 0.0)))
    pop_m_taper_delta = float(pop.get("pop_m_taper_delta", sel_meta.get("pop_m_taper_delta", 3.0)))
    pop_m_peak = float(pop.get("pop_m_peak", sel_meta.get("pop_m_peak", 35.0)))
    pop_m_peak_sigma = float(pop.get("pop_m_peak_sigma", sel_meta.get("pop_m_peak_sigma", 5.0)))
    pop_m_peak_frac = float(pop.get("pop_m_peak_frac", sel_meta.get("pop_m_peak_frac", 0.1)))

    selection_include_h0_volume_scaling = bool(gate2.get("selection_include_h0_volume_scaling", False))

    # Injection source path resolution.
    inj_path = _maybe_path(str(args.selection_injections_hdf) if args.selection_injections_hdf is not None else None)
    if inj_path is None:
        inj_path = _maybe_path(str(gate2.get("selection_injections_hdf")) if gate2.get("selection_injections_hdf") is not None else None)
    if inj_path is None:
        raise ValueError("Could not resolve injection file path; pass --selection-injections-hdf explicitly.")

    ifar_thresh = float(args.selection_ifar_thresh_yr) if args.selection_ifar_thresh_yr is not None else float(gate2.get("selection_ifar_threshold_yr", 1.0))
    injections = load_o3_injections(inj_path, ifar_threshold_yr=float(ifar_thresh))

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate2_seluncert_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    json_dir = out_dir / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    alpha_meta, alpha_draws = compute_alpha_h0_grid_pdet_marginalized(
        injections=injections,
        H0_grid=H0_grid,
        omega_m0=omega_m0,
        omega_k0=omega_k0,
        z_max=z_max,
        det_model=det_model,  # type: ignore[arg-type]
        snr_threshold=snr_threshold_f,
        snr_binned_nbins=snr_binned_nbins,
        mchirp_binned_nbins=mchirp_binned_nbins,
        q_binned_nbins=q_binned_nbins,
        weight_mode=weight_mode,  # type: ignore[arg-type]
        pop_z_mode=pop_z_mode,  # type: ignore[arg-type]
        pop_z_powerlaw_k=pop_z_k,
        pop_mass_mode=pop_mass_mode,  # type: ignore[arg-type]
        pop_m1_alpha=pop_m1_alpha,
        pop_m_min=pop_m_min,
        pop_m_max=pop_m_max,
        pop_q_beta=pop_q_beta,
        pop_m_taper_delta=pop_m_taper_delta,
        pop_m_peak=pop_m_peak,
        pop_m_peak_sigma=pop_m_peak_sigma,
        pop_m_peak_frac=pop_m_peak_frac,
        inj_mass_pdf_coords=inj_mass_pdf_coords,  # type: ignore[arg-type]
        inj_sampling_pdf_dist=inj_sampling_pdf_dist,  # type: ignore[arg-type]
        inj_sampling_pdf_mass_frame=inj_sampling_pdf_mass_frame,  # type: ignore[arg-type]
        inj_sampling_pdf_mass_scale=inj_sampling_pdf_mass_scale,  # type: ignore[arg-type]
        pdet_pseudocount=float(args.pdet_pseudocount),
        n_pdet_draws=int(args.n_pdet_draws),
        pdet_draw_seed=int(args.pdet_draw_seed),
    )

    # Convert alpha draws to the "effective" alpha used in the Gate-2 JSON if the xi-style scaling is enabled.
    alpha_draws_eff = np.asarray(alpha_draws, dtype=float)
    alpha_nom = np.asarray(alpha_meta["alpha_grid_nominal"], dtype=float)
    if selection_include_h0_volume_scaling:
        scale = np.clip(H0_grid, 1e-12, np.inf) ** 3
        alpha_draws_eff = alpha_draws_eff / scale.reshape((1, -1))
        alpha_nom = alpha_nom / scale

    # Selection-marginalized posterior: average posteriors across alpha draws.
    logL_data_rel = np.asarray(logL_sum_events_rel, dtype=float)
    p_draws = np.empty_like(alpha_draws_eff)
    for i in range(alpha_draws_eff.shape[0]):
        logL = logL_data_rel - float(n_events) * np.log(np.clip(alpha_draws_eff[i], 1e-300, np.inf))
        logL = logL - float(np.max(logL))
        p = np.exp(logL)
        p_draws[i] = p / float(np.sum(p))
    p_mix = np.mean(p_draws, axis=0)
    p_mix = normalize_pdf_grid(H0_grid, p_mix)

    # Reference: posterior using nominal alpha (no uncertainty).
    logL_nom = logL_data_rel - float(n_events) * np.log(np.clip(alpha_nom, 1e-300, np.inf))
    logL_nom = logL_nom - float(np.max(logL_nom))
    p_nom = normalize_pdf_grid(H0_grid, np.exp(logL_nom))

    # Tension-style summaries.
    prior_planck = GaussianPrior(name="planck_like", mean=float(args.planck_mean), sigma=float(args.planck_sigma))
    prior_shoes = GaussianPrior(name="shoes_like", mean=float(args.shoes_mean), sigma=float(args.shoes_sigma))

    s_nom = posterior_quantiles(H0_grid, p_nom)
    s_mix = posterior_quantiles(H0_grid, p_mix)
    p_hi = integrate_posterior_prob(H0_grid, p_mix, lo=float(args.h0_hi), hi=None)
    p_lo = integrate_posterior_prob(H0_grid, p_mix, lo=None, hi=float(args.h0_lo))
    bf = bayes_factor_between_priors_from_uniform_posterior(H0_grid, p_mix, prior_a=prior_planck, prior_b=prior_shoes)

    # Write outputs.
    (json_dir / "alpha_pdet_uncertainty_summary.json").write_text(json.dumps(alpha_meta, indent=2, sort_keys=True) + "\n")
    if args.save_alpha_draws_npz:
        np.savez(out_dir / "alpha_draws.npz", H0_grid=H0_grid.astype(float), alpha_draws=alpha_draws_eff.astype(float))

    out = {
        "method": "gate2_gr_h0_selection_uncertainty",
        "timestamp_utc": _utc_stamp(),
        "gate2_json": str(gate2_path),
        "selection_injections_hdf": str(inj_path),
        "selection_ifar_thresh_yr": float(ifar_thresh),
        "selection_include_h0_volume_scaling": bool(selection_include_h0_volume_scaling),
        "n_events": int(n_events),
        "inj_mass_pdf_coords": str(inj_mass_pdf_coords),
        "inj_sampling_pdf_dist": str(inj_sampling_pdf_dist),
        "inj_sampling_pdf_mass_frame": str(inj_sampling_pdf_mass_frame),
        "inj_sampling_pdf_mass_scale": str(inj_sampling_pdf_mass_scale),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "posterior_nominal_alpha": [float(x) for x in p_nom.tolist()],
        "posterior_sel_marginalized": [float(x) for x in p_mix.tolist()],
        "summary_nominal_alpha": s_nom,
        "summary_sel_marginalized": s_mix,
        "tension": {
            "h0_lo": float(args.h0_lo),
            "h0_hi": float(args.h0_hi),
            "p_h0_lt_lo": float(p_lo),
            "p_h0_gt_hi": float(p_hi),
            "planck_like": {"mean": float(prior_planck.mean), "sigma": float(prior_planck.sigma)},
            "shoes_like": {"mean": float(prior_shoes.mean), "sigma": float(prior_shoes.sigma)},
            "bf_planck_over_shoes": bf,
        },
    }
    (json_dir / "gate2_seluncert_posterior.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    # Plots
    plt.figure(figsize=(7.5, 4.2))
    plt.plot(H0_grid, p_nom, "-", lw=1.4, label="nominal alpha(H0)")
    plt.plot(H0_grid, p_mix, "-", lw=2.0, label="alpha(H0) marginalized")
    plt.xlabel(r"$H_0$ [km/s/Mpc]")
    plt.ylabel("posterior (grid-normalized)")
    plt.title("Gate-2 GR $H_0$ with selection-alpha uncertainty")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "h0_posterior_seluncert.png", dpi=160)
    plt.close()

    # Alpha quantiles plot (effective alpha used in this run).
    loga = np.log(np.clip(alpha_draws_eff, 1e-300, np.inf))
    loga16 = np.quantile(loga, 0.16, axis=0)
    loga50 = np.quantile(loga, 0.50, axis=0)
    loga84 = np.quantile(loga, 0.84, axis=0)
    plt.figure(figsize=(7.5, 4.2))
    plt.fill_between(H0_grid, np.exp(loga16), np.exp(loga84), alpha=0.25, label="p_det uncertainty band (16-84%)")
    plt.plot(H0_grid, np.exp(loga50), "-", lw=1.6, label="median alpha(H0)")
    plt.yscale("log")
    plt.xlabel(r"$H_0$ [km/s/Mpc]")
    plt.ylabel(r"$\alpha(H_0)$")
    plt.title("Selection alpha uncertainty from p_det bin/cell Beta draws")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "alpha_h0_uncertainty.png", dpi=160)
    plt.close()

    print(f"[gate2_seluncert] wrote {out_dir}", flush=True)
    print(
        f"[gate2_seluncert] marginalized: H0_map={s_mix['H0_map']:.3f} p50={s_mix['p50']:.3f} "
        f"[p16,p84]=[{s_mix['p16']:.3f},{s_mix['p84']:.3f}] "
        f"P(H0<{args.h0_lo:.1f})={p_lo:.3g} P(H0>{args.h0_hi:.1f})={p_hi:.3g} logBF(Planck/SH0ES)={bf['log_bf']:.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

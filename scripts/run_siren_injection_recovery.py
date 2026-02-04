from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # headless safe
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    _HAVE_MPL = False
import numpy as np

from entropy_horizon_recon.siren_injection_recovery import (
    InjectionRecoveryConfig,
    infer_z_max_for_h0_grid_closed_loop,
    load_injections_for_recovery,
    run_injection_recovery_gr_h0,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def main() -> int:
    ap = argparse.ArgumentParser(description="Injection-recovery harness for the siren audit (GR H0 control).")
    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")

    ap.add_argument("--h0-true", type=float, required=True, help="Truth H0 used to generate synthetic detected events.")
    ap.add_argument("--n-events", type=int, default=20, help="Number of synthetic detected events to generate (default 20).")
    ap.add_argument("--seed", type=int, default=0, help="Seed for synthetic event selection (default 0).")
    ap.add_argument(
        "--n-proc",
        type=int,
        default=0,
        help="Worker processes for hierarchical inference across events (default 0 = all cores).",
    )

    ap.add_argument("--h0-min", type=float, default=50.0, help="Min H0 in inference grid (default 50).")
    ap.add_argument("--h0-max", type=float, default=90.0, help="Max H0 in inference grid (default 90).")
    ap.add_argument("--h0-n", type=int, default=81, help="Number of H0 grid points (default 81).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 for GR distances (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 for GR distances (default 0).")
    ap.add_argument("--z-max", type=float, default=0.62, help="Max redshift used in inference + selection proxy (default 0.62).")
    ap.add_argument(
        "--z-max-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="z_max policy: fixed uses --z-max; auto expands z_max to avoid support truncation (debug-only; can silently change the inference regime) (default fixed).",
    )
    ap.add_argument("--z-max-auto-cap", type=float, default=5.0, help="Max z used for the auto z_max inversion cache (default 5).")
    ap.add_argument("--z-max-auto-margin", type=float, default=0.10, help="Additive safety margin on inferred z_max (default 0.10).")

    ap.add_argument(
        "--det-model",
        choices=["threshold", "snr_binned", "snr_mchirp_binned", "snr_mchirp_q_binned"],
        default="snr_binned",
        help="Detectability proxy model (default snr_binned).",
    )
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for det_model=snr_binned (default 200).")
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20, help="Chirp-mass bins for det_model=snr_mchirp_binned (default 20).")
    ap.add_argument("--q-binned-nbins", type=int, default=10, help="Mass-ratio bins for det_model=snr_mchirp_q_binned (default 10).")
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf", help="Injection weight mode (default inv_sampling_pdf).")
    ap.add_argument(
        "--inj-mass-pdf-coords",
        choices=["m1m2", "m1q"],
        default="m1m2",
        help="Mass-coordinate convention for injection sampling_pdf (default m1m2).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-dist",
        choices=["z", "dL", "log_dL"],
        default="z",
        help="Distance/redshift coordinate used by injection sampling_pdf: 'z' => density in z; 'dL' => density in luminosity distance (converted via dL/dz); 'log_dL' => density in log(dL) (converted via (1/dL)*(dL/dz)) (default z).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-frame",
        choices=["source", "detector"],
        default="source",
        help="Mass-frame used by injection sampling_pdf: 'source' means source-frame component masses; 'detector' means detector-frame masses and will be converted using (1+z) Jacobians (default source).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-scale",
        choices=["linear", "log"],
        default="linear",
        help="Mass coordinate scale used by injection sampling_pdf: 'linear' => density in masses; 'log' => density in log-mass coordinates (converted to linear via Jacobians) (default linear).",
    )
    ap.add_argument(
        "--include-pdet-in-event-term",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include p_det(dL) in the hierarchical PE event term (audit knob; default False).",
    )

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform", help="Population z mode (default comoving_uniform).")
    ap.add_argument("--pop-z-k", type=float, default=0.0, help="Powerlaw k for pop_z_mode=comoving_powerlaw (default 0).")
    ap.add_argument(
        "--pop-z-include-h0-volume-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include explicit (c/H0)^3 volume scaling in pop(z) inside the event term (audit-only; default False).",
    )
    ap.add_argument(
        "--pop-mass-mode",
        choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="powerlaw_peak_q_smooth",
        help="Population mass mode (default powerlaw_peak_q_smooth).",
    )
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3, help="Primary-mass powerlaw slope alpha (default 2.3).")
    ap.add_argument("--pop-m-min", type=float, default=5.0, help="Min source-frame mass (default 5).")
    ap.add_argument("--pop-m-max", type=float, default=80.0, help="Max source-frame mass (default 80).")
    ap.add_argument("--pop-q-beta", type=float, default=0.0, help="Mass ratio powerlaw exponent beta (default 0).")
    ap.add_argument("--pop-m-taper-delta", type=float, default=3.0, help="Smooth taper width (Msun) for smooth mass models (default 3).")
    ap.add_argument("--pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for peak mass mode (default 35).")
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for peak mass mode (default 5).")
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for peak mass mode (default 0.1).")

    ap.add_argument(
        "--selection-include-h0-volume-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include an explicit H0^{-3} factor in the selection alpha term (xi-style; audit-only; default False).",
    )

    ap.add_argument("--pe-n-samples", type=int, default=10_000, help="Synthetic PE samples per event (default 10000).")
    ap.add_argument(
        "--pe-obs-mode",
        choices=["truth", "noisy"],
        default="noisy",
        help="Synthetic PE likelihood center: 'truth' or a noisy draw around truth (default noisy).",
    )
    ap.add_argument(
        "--pe-synth-mode",
        choices=["naive_gaussian", "prior_resample", "likelihood_resample"],
        default="likelihood_resample",
        help="How to synthesize PE posteriors (default likelihood_resample).",
    )
    ap.add_argument(
        "--pe-prior-resample-n-candidates",
        type=int,
        default=200_000,
        help="Candidates for prior_resample (default 200000).",
    )
    ap.add_argument("--pe-seed", type=int, default=0, help="Additional seed offset for PE synthesis (default 0).")
    ap.add_argument("--dl-frac-sigma0", type=float, default=0.25, help="Base fractional dL scatter (default 0.25).")
    ap.add_argument("--dl-frac-sigma-floor", type=float, default=0.05, help="Minimum fractional dL scatter under SNR scaling (default 0.05).")
    ap.add_argument("--dl-sigma-mode", choices=["constant", "snr"], default="snr", help="Distance scatter model (default snr).")
    ap.add_argument("--mc-frac-sigma0", type=float, default=0.02, help="Base fractional chirp-mass scatter (default 0.02).")
    ap.add_argument("--q-sigma0", type=float, default=0.08, help="Mass-ratio scatter (default 0.08).")

    ap.add_argument("--pe-prior-dl-expr", default="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)", help="Analytic PE prior expression for dL.")
    ap.add_argument("--pe-prior-chirp-mass-expr", default="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)", help="Analytic PE prior expression for chirp mass.")
    ap.add_argument("--pe-prior-mass-ratio-expr", default="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)", help="Analytic PE prior expression for mass ratio.")

    ap.add_argument("--event-qc-mode", choices=["fail", "skip"], default="skip", help="Event QC mode if an event has insufficient support (default skip).")
    ap.add_argument(
        "--event-min-finite-frac",
        type=float,
        default=0.0,
        help="Minimum finite-support fraction across H0 grid (default 0; disables 'insufficient support' skipping).",
    )
    ap.add_argument(
        "--event-min-ess",
        type=float,
        default=0.0,
        help="Minimum ESS across H0 grid per event (default 0; disables low-ESS skipping).",
    )

    ap.add_argument(
        "--importance-smoothing",
        choices=["none", "truncate", "psis"],
        default="none",
        help="Importance-sampling stabilization for hierarchical PE reweighting (default none).",
    )
    ap.add_argument(
        "--importance-truncate-tau",
        type=float,
        default=None,
        help="Truncation tau for --importance-smoothing=truncate (default sqrt(n)).",
    )

    ap.add_argument("--smoke", action="store_true", help="Tiny smoke run (few events + small H0 grid + small PE samples).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_injection_recovery_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_injection_recovery_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    if _HAVE_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    n_events = int(args.n_events)
    h0_n = int(args.h0_n)
    pe_n = int(args.pe_n_samples)
    if bool(args.smoke):
        n_events = min(n_events, 3)
        h0_n = min(h0_n, 31)
        pe_n = min(pe_n, 2000)
        if str(args.pe_synth_mode) == "prior_resample":
            args.pe_prior_resample_n_candidates = min(int(args.pe_prior_resample_n_candidates), 50_000)

    # Auto-expand z_max if requested: avoid QC-driven biases from partial support at high H0.
    z_max = float(args.z_max)
    if str(args.z_max_mode) == "auto":
        h0_eval = float(max(float(args.h0_min), float(args.h0_max)))
        z_req = infer_z_max_for_h0_grid_closed_loop(
            omega_m0=float(args.omega_m0),
            omega_k0=float(args.omega_k0),
            z_gen_max=float(z_max),
            h0_true=float(args.h0_true),
            h0_eval=float(h0_eval),
            z_cap=float(args.z_max_auto_cap),
        )
        margin = float(args.z_max_auto_margin)
        if not (np.isfinite(margin) and margin >= 0.0):
            raise ValueError("z_max_auto_margin must be finite and >=0.")
        z_max = float(max(z_max, z_req + margin))

    cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(z_max),
        det_model=str(args.det_model),  # type: ignore[arg-type]
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
        q_binned_nbins=int(args.q_binned_nbins),
        selection_ifar_thresh_yr=float(args.selection_ifar_thresh_yr),
        include_pdet_in_event_term=bool(args.include_pdet_in_event_term),
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_z_include_h0_volume_scaling=bool(args.pop_z_include_h0_volume_scaling),
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
        inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
        inj_sampling_pdf_dist=str(args.inj_sampling_pdf_dist),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_frame=str(args.inj_sampling_pdf_mass_frame),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_scale=str(args.inj_sampling_pdf_mass_scale),  # type: ignore[arg-type]
        selection_include_h0_volume_scaling=bool(args.selection_include_h0_volume_scaling),
        pe_obs_mode=str(args.pe_obs_mode),  # type: ignore[arg-type]
        pe_n_samples=int(pe_n),
        pe_synth_mode=str(args.pe_synth_mode),  # type: ignore[arg-type]
        pe_prior_resample_n_candidates=int(args.pe_prior_resample_n_candidates),
        pe_seed=int(args.pe_seed),
        dl_frac_sigma0=float(args.dl_frac_sigma0),
        dl_frac_sigma_floor=float(args.dl_frac_sigma_floor),
        dl_sigma_mode=str(args.dl_sigma_mode),  # type: ignore[arg-type]
        mc_frac_sigma0=float(args.mc_frac_sigma0),
        q_sigma0=float(args.q_sigma0),
        pe_prior_dL_expr=str(args.pe_prior_dl_expr),
        pe_prior_chirp_mass_expr=str(args.pe_prior_chirp_mass_expr),
        pe_prior_mass_ratio_expr=str(args.pe_prior_mass_ratio_expr),
        event_qc_mode=str(args.event_qc_mode),  # type: ignore[arg-type]
        event_min_finite_frac=float(args.event_min_finite_frac),
        event_min_ess=float(args.event_min_ess),
        importance_smoothing=str(args.importance_smoothing),  # type: ignore[arg-type]
        importance_truncate_tau=float(args.importance_truncate_tau) if args.importance_truncate_tau is not None else None,
    )

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(h0_n))

    injections = load_injections_for_recovery(args.selection_injections_hdf, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
    out = run_injection_recovery_gr_h0(
        injections=injections,
        cfg=cfg,
        n_events=int(n_events),
        h0_grid=H0_grid,
        seed=int(args.seed),
        n_processes=int(args.n_proc),
        out_dir=out_dir,
    )

    # Quick figure: selection-off vs selection-on posteriors.
    p_off = np.asarray(out["gr_h0_selection_off"]["posterior"], dtype=float)
    p_on = np.asarray(out["gr_h0_selection_on"]["posterior"], dtype=float)
    H0 = np.asarray(out["gr_h0_selection_on"]["H0_grid"], dtype=float)

    if _HAVE_MPL:
        assert plt is not None
        plt.figure(figsize=(7.4, 4.0))
        plt.plot(H0, p_off, lw=2.0, label="GR H0 (selection OFF)")
        plt.plot(H0, p_on, lw=2.0, label="GR H0 (selection ON)")
        plt.axvline(float(args.h0_true), color="k", ls="--", lw=1.2, label=f"H0_true={float(args.h0_true):.2f}")
        plt.xlabel(r"$H_0$ [km/s/Mpc]")
        plt.ylabel("posterior (normalized)")
        plt.title("Injection-recovery: GR H0 control")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(fig_dir / "gr_h0_injection_recovery.png", dpi=160)
        plt.close()
    else:
        print("[note] matplotlib not available; skipping figures", flush=True)

    (out_dir / "injection_recovery.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

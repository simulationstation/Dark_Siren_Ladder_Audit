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

from entropy_horizon_recon.dark_siren_h0 import compute_gr_h0_posterior_grid_hierarchical_pe
from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache  # noqa: SLF001
from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _infer_required_z_max_for_h0_grid(
    *,
    pe_by_event: dict[str, GWTCPeHierarchicalSamples],
    omega_m0: float,
    omega_k0: float,
    h0_eval: float,
    z_cap: float,
    dL_quantile: float,
) -> tuple[float, dict[str, float]]:
    """Infer a safe z_max so most PE samples map inside z<=z_max at a given H0.

    Gate‑2 (GR H0) can become biased if events are dropped due to partial support across the H0 grid.
    Partial support is typically caused by too-small z_max when mapping PE dL samples to z at high H0.
    """
    q = float(dL_quantile)
    if not (np.isfinite(q) and 0.5 <= q < 1.0):
        raise ValueError("z-max auto dL quantile must be in [0.5, 1).")

    z_cap = float(z_cap)
    if not (np.isfinite(z_cap) and z_cap > 0.0):
        raise ValueError("z-max auto cap must be finite and positive.")

    h0_eval = float(h0_eval)
    if not (np.isfinite(h0_eval) and h0_eval > 0.0):
        raise ValueError("h0_eval must be finite and positive.")

    cache = _build_lcdm_distance_cache(z_max=float(z_cap), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    z_grid = np.asarray(cache.z_grid, dtype=float)
    f_grid = np.asarray(cache.f_grid, dtype=float)
    c_km_s = 299792.458

    per_event_req: dict[str, float] = {}
    req_max = 0.0
    for ev, pe in pe_by_event.items():
        dL = np.asarray(pe.dL_mpc, dtype=float)
        dL = dL[np.isfinite(dL) & (dL > 0.0)]
        if dL.size == 0:
            continue
        dL_q = float(np.quantile(dL, q))
        f = dL_q * h0_eval / c_km_s
        z = float(np.interp(f, f_grid, z_grid, left=np.nan, right=np.nan))
        if not np.isfinite(z):
            z = float(z_cap)
        per_event_req[str(ev)] = float(z)
        req_max = max(req_max, float(z))
    return float(req_max), per_event_req


def _load_pe_hier_cache_npz(path: Path) -> tuple[str, GWTCPeHierarchicalSamples]:
    with np.load(path, allow_pickle=True) as d:
        meta = json.loads(str(d["meta"].tolist())) if "meta" in d.files else {}
        ev = str(meta.get("event", "")) if meta else ""
        pe_obj = GWTCPeHierarchicalSamples(
            file=str(meta.get("pe_file", "<cache>")),
            analysis=str(meta.get("pe_analysis_chosen", meta.get("pe_analysis", "UNKNOWN"))),
            n_total=int(meta.get("n_total", -1)),
            n_used=int(meta.get("n_used", -1)),
            dL_mpc=np.asarray(d["dL_mpc"], dtype=float),
            chirp_mass_det=np.asarray(d["chirp_mass_det"], dtype=float),
            mass_ratio=np.asarray(d["mass_ratio"], dtype=float),
            log_pi_dL=np.asarray(d["log_pi_dL"], dtype=float),
            log_pi_chirp_mass=np.asarray(d["log_pi_chirp_mass"], dtype=float),
            log_pi_mass_ratio=np.asarray(d["log_pi_mass_ratio"], dtype=float),
            prior_spec=json.loads(str(meta.get("prior_spec_json", "{}"))),
        )
        return ev, pe_obj


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 2: GR H0 selection-on control (hierarchical PE).")
    ap.add_argument("--pe-cache-dir", required=True, help="Directory containing cached hierarchical PE .npz files (pe_hier_<EVENT>.npz).")
    ap.add_argument("--from-dark-siren-summary", required=True, help="run_dark_siren_gap_test summary JSON (events_scored, z_max).")
    ap.add_argument(
        "--events",
        default=None,
        help="Optional comma list of events to use (overrides summary events_scored).",
    )
    ap.add_argument(
        "--events-from-pe-cache",
        action="store_true",
        help="Use all events found in --pe-cache-dir (pe_hier_<EVENT>.npz), ignoring summary events_scored.",
    )

    ap.add_argument("--h0-min", type=float, default=50.0, help="Min H0 (default 50).")
    ap.add_argument("--h0-max", type=float, default=90.0, help="Max H0 (default 90).")
    ap.add_argument("--h0-n", type=int, default=81, help="Number of H0 grid points (default 81).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 for GR distances (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 for GR distances (default 0).")
    ap.add_argument("--z-max", type=float, default=None, help="Override z_max (default from summary).")
    ap.add_argument(
        "--z-max-mode",
        choices=["fixed", "auto"],
        default="fixed",
        help="z_max policy: fixed uses summary/--z-max; auto expands z_max to avoid support truncation (debug-only; can silently change the inference regime) (default fixed).",
    )
    ap.add_argument(
        "--z-max-auto-dl-quantile",
        type=float,
        default=0.999,
        help="Distance quantile used to infer required z_max in auto mode (default 0.999).",
    )
    ap.add_argument(
        "--z-max-auto-cap",
        type=float,
        default=5.0,
        help="Max z used in temporary inversion cache for auto mode (default 5).",
    )
    ap.add_argument(
        "--z-max-auto-margin",
        type=float,
        default=0.05,
        help="Fractional safety margin on inferred z_max in auto mode (default 0.05).",
    )

    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")
    ap.add_argument(
        "--det-model",
        choices=["threshold", "snr_binned", "snr_mchirp_binned"],
        default="snr_binned",
        help="Detection model for alpha(H0) (default snr_binned).",
    )
    ap.add_argument("--snr-thresh", type=float, default=None, help="Optional fixed SNR threshold (default: calibrate).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for snr_binned model (default 200).")
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20, help="Chirp-mass bins for det_model=snr_mchirp_binned (default 20).")
    ap.add_argument(
        "--weight-modes",
        default="none,inv_sampling_pdf",
        help="Comma list of injection weight modes to compare for selection-on (default none,inv_sampling_pdf).",
    )

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="none", help="Population z weight mode (default none).")
    ap.add_argument("--pop-z-k", type=float, default=0.0, help="Powerlaw k for --pop-z-mode=comoving_powerlaw.")
    ap.add_argument(
        "--pop-z-include-h0-volume-scaling",
        action="store_true",
        help="Include an explicit (c/H0)^3 factor in the pop_z weight (audit/debug knob).",
    )
    ap.add_argument(
        "--pop-mass-mode",
        choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="none",
        help="Population mass weight mode (default none).",
    )
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3, help="m1 powerlaw slope alpha (default 2.3).")
    ap.add_argument("--pop-m-min", type=float, default=5.0, help="Min source-frame mass (default 5).")
    ap.add_argument("--pop-m-max", type=float, default=200.0, help="Max source-frame mass (default 200).")
    ap.add_argument("--pop-q-beta", type=float, default=0.0, help="q powerlaw exponent beta (default 0).")
    ap.add_argument(
        "--pop-m-taper-delta",
        type=float,
        default=3.0,
        help="Smooth taper width (Msun) for smooth mass models (powerlaw_q_smooth / powerlaw_peak_q_smooth). Must be >0 (default 3).",
    )
    ap.add_argument("--pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for --pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for --pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for --pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument(
        "--inj-mass-pdf-coords",
        choices=["m1m2", "m1q"],
        default="m1m2",
        help="Mass-coordinate convention for injection sampling_pdf (default m1m2).",
    )
    ap.add_argument(
        "--selection-include-h0-volume-scaling",
        action="store_true",
        help="Use an xi-style normalization by including an H0^{-3} factor in the selection term (audit/debug knob).",
    )
    ap.add_argument(
        "--event-qc-mode",
        choices=["fail", "skip"],
        default="skip",
        help="How to handle events with zero support across the H0 grid under the chosen population/z_max (default skip).",
    )
    ap.add_argument(
        "--event-min-finite-frac",
        type=float,
        default=0.0,
        help="Minimum fraction of H0 grid points with finite event logL to keep an event when event_qc_mode=skip (default 0; disables 'insufficient support' skipping).",
    )
    ap.add_argument(
        "--event-min-ess",
        type=float,
        default=0.0,
        help="Minimum ess_min across the H0 grid to keep an event (default 0; disables ESS-based skipping).",
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
    ap.add_argument(
        "--n-proc",
        type=int,
        default=0,
        help="Processes for per-event hierarchical PE term computation (0=auto; default 0).",
    )

    ap.add_argument("--smoke", action="store_true", help="Seconds-scale smoke mode (shrinks event count + grid).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate2_gr_h0_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate2_gr_h0_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    json_dir = out_dir / "json"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    pe_cache_dir = Path(args.pe_cache_dir).expanduser().resolve()

    summary_path = Path(args.from_dark_siren_summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    if bool(args.events_from_pe_cache):
        events = []
        for p in sorted(pe_cache_dir.glob("pe_hier_*.npz")):
            stem = p.stem  # pe_hier_<EVENT>
            if not stem.startswith("pe_hier_"):
                continue
            events.append(stem[len("pe_hier_") :])
    elif args.events is not None:
        events = [e.strip() for e in str(args.events).split(",") if e.strip()]
    else:
        events = [str(x) for x in summary.get("events_scored", [])]
    if not events:
        raise ValueError("No events selected (use --events, --events-from-pe-cache, or ensure summary has events_scored).")

    z_max = float(args.z_max) if args.z_max is not None else float(summary.get("z_max", 0.0))
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("Invalid z_max (provide --z-max or ensure summary has z_max).")

    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = {}
    for ev in sorted(events):
        p = pe_cache_dir / f"pe_hier_{ev}.npz"
        if not p.exists():
            continue
        ev2, pe = _load_pe_hier_cache_npz(p)
        pe_by_event[ev2 or ev] = pe

    if bool(args.smoke):
        events = sorted(pe_by_event.keys())[: min(3, len(pe_by_event))]
        pe_by_event = {e: pe_by_event[e] for e in events}
        args.h0_n = min(int(args.h0_n), 11)
        args.h0_min = float(args.h0_min)
        args.h0_max = float(args.h0_max)

    if not pe_by_event:
        raise ValueError("No PE cache files loaded; check --pe-cache-dir and event list.")

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))
    h0_eval = float(np.nanmax(H0_grid))

    if str(args.z_max_mode) == "auto":
        z_cap = float(max(float(args.z_max_auto_cap), float(z_max) + 1e-6))
        req, req_by_event = _infer_required_z_max_for_h0_grid(
            pe_by_event=pe_by_event,
            omega_m0=float(args.omega_m0),
            omega_k0=float(args.omega_k0),
            h0_eval=float(h0_eval),
            z_cap=float(z_cap),
            dL_quantile=float(args.z_max_auto_dl_quantile),
        )
        margin = float(args.z_max_auto_margin)
        if not (np.isfinite(margin) and margin >= 0.0):
            raise ValueError("z-max auto margin must be finite and >= 0.")
        z_req = float(req) * (1.0 + margin)
        z_base = float(z_max)
        if z_req > z_base:
            z_max = float(z_req)

        (out_dir / "json" / "z_max_auto.json").write_text(
            json.dumps(
                {
                    "z_max_mode": "auto",
                    "z_max_base": float(z_base),
                    "z_max_final": float(z_max),
                    "h0_eval": float(h0_eval),
                    "dL_quantile": float(args.z_max_auto_dl_quantile),
                    "z_cap": float(z_cap),
                    "margin": float(margin),
                    "required_z_max_by_event": req_by_event,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    inj_path = Path(args.selection_injections_hdf).expanduser().resolve()
    injections = load_o3_injections(inj_path, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))

    weight_modes = [m.strip() for m in str(args.weight_modes).split(",") if m.strip()]
    if not weight_modes:
        raise ValueError("No weight modes specified.")

    # Selection OFF (alpha not applied).
    res_off = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event=pe_by_event,
        H0_grid=H0_grid,
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(z_max),
        cache_dir=cache_dir / "selection_off",
        n_processes=int(args.n_proc),
        importance_smoothing=str(args.importance_smoothing),  # type: ignore[arg-type]
        importance_truncate_tau=float(args.importance_truncate_tau) if args.importance_truncate_tau is not None else None,
        injections=None,
        ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
        det_model=str(args.det_model),  # type: ignore[arg-type]
        snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
        weight_mode="none",
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(args.pop_z_k),
        pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=float(args.pop_m_taper_delta),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        pop_z_include_h0_volume_scaling=bool(args.pop_z_include_h0_volume_scaling),
        inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
        selection_include_h0_volume_scaling=bool(args.selection_include_h0_volume_scaling),
        event_qc_mode=str(args.event_qc_mode),  # type: ignore[arg-type]
        event_min_finite_frac=float(args.event_min_finite_frac),
        event_min_ess=float(args.event_min_ess),
        prior="uniform",
    )
    (json_dir / "gr_h0_selection_off.json").write_text(json.dumps(res_off, indent=2, sort_keys=True) + "\n")
    print(
        f"[gate2_gr_h0] selection OFF: n_events={int(res_off.get('n_events', -1))} "
        f"H0_map={float(res_off.get('H0_map', float('nan'))):.3f} "
        f"edge={bool(res_off.get('H0_map_at_edge', False))} "
        f"p50={float(res_off.get('summary', {}).get('p50', float('nan'))):.3f} "
        f"[p16,p84]=[{float(res_off.get('summary', {}).get('p16', float('nan'))):.3f},"
        f"{float(res_off.get('summary', {}).get('p84', float('nan'))):.3f}]",
        flush=True,
    )

    # Selection ON for each weight mode.
    res_on: dict[str, dict[str, Any]] = {}
    for wm in weight_modes:
        res = compute_gr_h0_posterior_grid_hierarchical_pe(
            pe_by_event=pe_by_event,
            H0_grid=H0_grid,
            omega_m0=float(args.omega_m0),
            omega_k0=float(args.omega_k0),
            z_max=float(z_max),
            cache_dir=cache_dir / f"selection_on_{wm}",
            n_processes=int(args.n_proc),
            importance_smoothing=str(args.importance_smoothing),  # type: ignore[arg-type]
            importance_truncate_tau=float(args.importance_truncate_tau) if args.importance_truncate_tau is not None else None,
            injections=injections,
            ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
            det_model=str(args.det_model),  # type: ignore[arg-type]
            snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
            snr_binned_nbins=int(args.snr_binned_nbins),
            mchirp_binned_nbins=int(args.mchirp_binned_nbins),
            weight_mode=str(wm),  # type: ignore[arg-type]
            pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
            pop_z_powerlaw_k=float(args.pop_z_k),
            pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(args.pop_m1_alpha),
            pop_m_min=float(args.pop_m_min),
            pop_m_max=float(args.pop_m_max),
            pop_q_beta=float(args.pop_q_beta),
            pop_m_taper_delta=float(args.pop_m_taper_delta),
            pop_m_peak=float(args.pop_m_peak),
            pop_m_peak_sigma=float(args.pop_m_peak_sigma),
            pop_m_peak_frac=float(args.pop_m_peak_frac),
            pop_z_include_h0_volume_scaling=bool(args.pop_z_include_h0_volume_scaling),
            inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
            selection_include_h0_volume_scaling=bool(args.selection_include_h0_volume_scaling),
            event_qc_mode=str(args.event_qc_mode),  # type: ignore[arg-type]
            event_min_finite_frac=float(args.event_min_finite_frac),
            event_min_ess=float(args.event_min_ess),
            prior="uniform",
        )
        res_on[str(wm)] = res
        (json_dir / f"gr_h0_selection_on_{wm}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        print(
            f"[gate2_gr_h0] selection ON ({wm}): n_events={int(res.get('n_events', -1))} "
            f"H0_map={float(res.get('H0_map', float('nan'))):.3f} "
            f"edge={bool(res.get('H0_map_at_edge', False))} "
            f"p50={float(res.get('summary', {}).get('p50', float('nan'))):.3f} "
            f"[p16,p84]=[{float(res.get('summary', {}).get('p16', float('nan'))):.3f},"
            f"{float(res.get('summary', {}).get('p84', float('nan'))):.3f}]",
            flush=True,
        )

    # Plot overlay.
    plt.figure(figsize=(7.5, 4.2))
    H0 = np.asarray(res_off["H0_grid"], dtype=float)
    p_off = np.asarray(res_off["posterior"], dtype=float)
    plt.plot(H0, p_off, "-", lw=1.6, label="selection OFF")
    for wm, res in res_on.items():
        p = np.asarray(res["posterior"], dtype=float)
        plt.plot(H0, p, "-", lw=1.4, label=f"selection ON ({wm})")
    plt.xlabel(r"$H_0$ [km/s/Mpc]")
    plt.ylabel("posterior (normalized on grid)")
    plt.title("Gate 2 GR $H_0$ control (hierarchical PE)")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "gr_h0_control.png", dpi=160)
    plt.close()

    manifest = {
        "timestamp_utc": _utc_stamp(),
        "from_dark_siren_summary": str(summary_path),
        "pe_cache_dir": str(pe_cache_dir),
        "n_events": int(len(pe_by_event)),
        "events": sorted(pe_by_event.keys()),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "omega_m0": float(args.omega_m0),
        "omega_k0": float(args.omega_k0),
        "z_max": float(z_max),
        "n_processes": int(args.n_proc),
        "selection_injections_hdf": str(inj_path),
        "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
        "det_model": str(args.det_model),
        "snr_threshold": float(args.snr_thresh) if args.snr_thresh is not None else None,
        "snr_binned_nbins": int(args.snr_binned_nbins),
        "weight_modes": list(weight_modes),
        "inj_mass_pdf_coords": str(args.inj_mass_pdf_coords),
        "selection_include_h0_volume_scaling": bool(args.selection_include_h0_volume_scaling),
        "pop": {
            "pop_z_mode": str(args.pop_z_mode),
            "pop_z_k": float(args.pop_z_k),
            "pop_z_include_h0_volume_scaling": bool(args.pop_z_include_h0_volume_scaling),
            "pop_mass_mode": str(args.pop_mass_mode),
            "pop_m1_alpha": float(args.pop_m1_alpha),
            "pop_m_min": float(args.pop_m_min),
            "pop_m_max": float(args.pop_m_max),
            "pop_q_beta": float(args.pop_q_beta),
            "pop_m_taper_delta": float(args.pop_m_taper_delta),
            "pop_m_peak": float(args.pop_m_peak),
            "pop_m_peak_sigma": float(args.pop_m_peak_sigma),
            "pop_m_peak_frac": float(args.pop_m_peak_frac),
        },
        "importance_smoothing": str(args.importance_smoothing),
        "importance_truncate_tau": float(args.importance_truncate_tau) if args.importance_truncate_tau is not None else None,
        "smoke": bool(args.smoke),
    }
    (json_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"[gate2_gr_h0] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

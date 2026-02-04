from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.siren_injection_recovery import (
    InjectionRecoveryConfig,
    compute_selection_alpha_h0_grid_for_cfg,
    load_injections_for_recovery,
    run_injection_recovery_gr_h0,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _posterior_from_logL_rel(H0_grid: np.ndarray, logL_rel: np.ndarray) -> dict[str, Any]:
    H0_grid = np.asarray(H0_grid, dtype=float)
    logL_rel = np.asarray(logL_rel, dtype=float)
    if H0_grid.shape != logL_rel.shape:
        raise ValueError("H0_grid and logL_rel must have the same shape.")

    m = np.isfinite(logL_rel)
    if not np.any(m):
        raise ValueError("logL_rel has no finite entries.")

    log_post = logL_rel - float(np.nanmax(logL_rel[m]))
    p = np.exp(np.clip(log_post, -700.0, 50.0))
    p = np.where(m, p, 0.0)
    s = float(np.sum(p))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Posterior normalization failed.")
    p = p / s

    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))

    mean = float(np.sum(p * H0_grid))
    sd = float(np.sqrt(np.sum(p * (H0_grid - mean) ** 2)))
    i_map = int(np.argmax(p))
    H0_map = float(H0_grid[i_map])
    return {
        "H0_map": float(H0_map),
        "H0_map_index": int(i_map),
        "H0_map_at_edge": bool(i_map == 0 or i_map == (H0_grid.size - 1)),
        "summary": {"mean": mean, "sd": sd, "p50": q50, "p16": q16, "p84": q84},
    }


def _contains_interval(lo: float, x: float, hi: float) -> bool:
    lo = float(lo)
    hi = float(hi)
    x = float(x)
    if not (np.isfinite(lo) and np.isfinite(hi) and np.isfinite(x)):
        return False
    if hi < lo:
        lo, hi = hi, lo
    return bool(lo <= x <= hi)


def main() -> int:
    ap = argparse.ArgumentParser(description="Replicate suite for injection-recovery GR H0 control (selection bookkeeping).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_injection_recovery_suite_<UTCSTAMP>).")

    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")

    ap.add_argument("--h0-true", type=float, required=True, help="Truth H0 used to generate synthetic detected events.")
    ap.add_argument("--n-events", type=int, default=64, help="Detected events per replicate (default 64).")
    ap.add_argument("--n-reps", type=int, default=25, help="Number of replicates (default 25).")
    ap.add_argument("--seed0", type=int, default=0, help="Base seed; replicate r uses seed=seed0+r (default 0).")
    ap.add_argument("--n-proc", type=int, default=0, help="Worker processes for hierarchical inference (default 0 = all cores).")

    ap.add_argument("--h0-min", type=float, default=40.0)
    ap.add_argument("--h0-max", type=float, default=100.0)
    ap.add_argument("--h0-n", type=int, default=121)
    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=0.62)

    ap.add_argument("--det-model", choices=["threshold", "snr_binned", "snr_mchirp_binned", "snr_mchirp_q_binned"], default="snr_mchirp_binned")
    ap.add_argument("--snr-binned-nbins", type=int, default=200)
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20)
    ap.add_argument("--q-binned-nbins", type=int, default=10)
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf")
    ap.add_argument("--inj-mass-pdf-coords", choices=["m1m2", "m1q"], default="m1m2")
    ap.add_argument("--inj-sampling-pdf-dist", choices=["z", "dL", "log_dL"], default="log_dL")
    ap.add_argument("--inj-sampling-pdf-mass-frame", choices=["source", "detector"], default="detector")
    ap.add_argument("--inj-sampling-pdf-mass-scale", choices=["linear", "log"], default="linear")

    ap.add_argument("--include-pdet-in-event-term", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform")
    ap.add_argument("--pop-z-k", type=float, default=0.0)
    ap.add_argument("--pop-z-include-h0-volume-scaling", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument(
        "--pop-mass-mode",
        choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="powerlaw_peak_q_smooth",
    )
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3)
    ap.add_argument("--pop-m-min", type=float, default=5.0)
    ap.add_argument("--pop-m-max", type=float, default=80.0)
    ap.add_argument("--pop-q-beta", type=float, default=0.0)
    ap.add_argument("--pop-m-taper-delta", type=float, default=3.0)
    ap.add_argument("--pop-m-peak", type=float, default=35.0)
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0)
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1)

    ap.add_argument("--pe-n-samples", type=int, default=5000)
    ap.add_argument("--pe-obs-mode", choices=["truth", "noisy"], default="noisy")
    ap.add_argument("--pe-synth-mode", choices=["naive_gaussian", "prior_resample", "likelihood_resample"], default="likelihood_resample")
    ap.add_argument("--pe-prior-resample-n-candidates", type=int, default=200_000)
    ap.add_argument("--pe-seed", type=int, default=0)
    ap.add_argument("--dl-frac-sigma0", type=float, default=0.25)
    ap.add_argument("--dl-frac-sigma-floor", type=float, default=0.05)
    ap.add_argument("--dl-sigma-mode", choices=["constant", "snr"], default="snr")
    ap.add_argument("--mc-frac-sigma0", type=float, default=0.02)
    ap.add_argument("--q-sigma0", type=float, default=0.08)
    ap.add_argument("--pe-prior-dl-expr", default="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)")
    ap.add_argument("--pe-prior-chirp-mass-expr", default="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)")
    ap.add_argument("--pe-prior-mass-ratio-expr", default="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)")

    ap.add_argument("--event-qc-mode", choices=["fail", "skip"], default="skip")
    ap.add_argument("--event-min-finite-frac", type=float, default=0.0)
    ap.add_argument("--event-min-ess", type=float, default=0.0)
    ap.add_argument("--importance-smoothing", choices=["none", "truncate", "psis"], default="none")
    ap.add_argument("--importance-truncate-tau", type=float, default=None)

    ap.add_argument("--smoke", action="store_true", help="Tiny run: n_reps<=3, n_events<=8, smaller grids/samples.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_injection_recovery_suite_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_reps = int(args.n_reps)
    n_events = int(args.n_events)
    pe_n = int(args.pe_n_samples)
    h0_n = int(args.h0_n)
    pe_cand = int(args.pe_prior_resample_n_candidates)
    if bool(args.smoke):
        n_reps = min(n_reps, 3)
        n_events = min(n_events, 8)
        pe_n = min(pe_n, 2000)
        h0_n = min(h0_n, 51)
        pe_cand = min(pe_cand, 80_000)

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(h0_n))
    cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
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
        selection_include_h0_volume_scaling=False,
        pe_obs_mode=str(args.pe_obs_mode),  # type: ignore[arg-type]
        pe_n_samples=int(pe_n),
        pe_synth_mode=str(args.pe_synth_mode),  # type: ignore[arg-type]
        pe_prior_resample_n_candidates=int(pe_cand),
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

    manifest = {
        "created_utc": _utc_stamp(),
        "n_reps": int(n_reps),
        "n_events": int(n_events),
        "n_proc": int(args.n_proc),
        "h0_grid": [float(x) for x in H0_grid.tolist()],
        "cfg": cfg.__dict__,
    }
    _write_json(out_dir / "manifest.json", manifest)

    injections = load_injections_for_recovery(args.selection_injections_hdf, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
    alpha_grid, alpha_meta = compute_selection_alpha_h0_grid_for_cfg(injections=injections, cfg=cfg, h0_grid=H0_grid)
    _write_json(
        out_dir / "alpha.json",
        {"H0_grid": [float(x) for x in H0_grid.tolist()], "alpha_grid": [float(x) for x in np.asarray(alpha_grid, dtype=float).tolist()], "meta": alpha_meta},
    )

    csv_path = out_dir / "run_log.csv"
    want_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if want_header:
            w.writerow(
                [
                    "utc",
                    "rep",
                    "seed",
                    "n_events_used",
                    "bias_p50_off",
                    "bias_p50_on_no_vol",
                    "bias_p50_on_vol",
                    "bias_map_off",
                    "bias_map_on_no_vol",
                    "bias_map_on_vol",
                    "cover68_off",
                    "cover68_on_no_vol",
                    "cover68_on_vol",
                    "edge_off",
                    "edge_on_no_vol",
                    "edge_on_vol",
                ]
            )

        rows: list[dict[str, Any]] = []
        for r in range(int(n_reps)):
            seed = int(args.seed0) + int(r)
            out = run_injection_recovery_gr_h0(
                injections=injections,
                cfg=cfg,
                n_events=int(n_events),
                h0_grid=H0_grid,
                seed=int(seed),
                selection_alpha_h0_grid=np.asarray(alpha_grid, dtype=float),
                selection_alpha_meta=dict(alpha_meta),
                n_processes=int(args.n_proc),
                out_dir=None,
            )

            res_off = out["gr_h0_selection_off"]
            res_on_no_vol = out["gr_h0_selection_on"]
            H0 = np.asarray(res_off["H0_grid"], dtype=float)
            logL_sum_rel = np.asarray(res_off["logL_sum_events_rel"], dtype=float)
            n_used = int(res_off.get("n_events", len(out.get("truths", []))))
            if H0.shape != logL_sum_rel.shape:
                raise ValueError("Selection-off output missing/invalid logL_sum_events_rel.")

            log_alpha = np.log(np.clip(np.asarray(alpha_grid, dtype=float), 1e-300, np.inf))
            if log_alpha.shape != H0.shape:
                raise ValueError("alpha_grid shape mismatch vs H0_grid.")

            # Alternative selection convention: include an explicit H0^{-3} factor in alpha.
            log_alpha_vol = log_alpha - 3.0 * np.log(np.clip(H0, 1e-12, np.inf))
            post_on_vol = _posterior_from_logL_rel(H0, logL_sum_rel - float(n_used) * log_alpha_vol)

            h0_true = float(cfg.h0_true)
            s_off = res_off.get("summary") or {}
            s_on_no_vol = (res_on_no_vol.get("summary") or {}) if isinstance(res_on_no_vol, dict) else {}
            s_on_vol = post_on_vol.get("summary") or {}

            row = {
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rep": int(r),
                "seed": int(seed),
                "n_events_used": int(n_used),
                "bias_p50_off": float(s_off.get("p50", float("nan"))) - h0_true,
                "bias_p50_on_no_vol": float(s_on_no_vol.get("p50", float("nan"))) - h0_true,
                "bias_p50_on_vol": float(s_on_vol.get("p50", float("nan"))) - h0_true,
                "bias_map_off": float(res_off.get("H0_map", float("nan"))) - h0_true,
                "bias_map_on_no_vol": float(res_on_no_vol.get("H0_map", float("nan"))) - h0_true,
                "bias_map_on_vol": float(post_on_vol.get("H0_map", float("nan"))) - h0_true,
                "cover68_off": bool(_contains_interval(float(s_off.get("p16", float("nan"))), h0_true, float(s_off.get("p84", float("nan"))))),
                "cover68_on_no_vol": bool(
                    _contains_interval(float(s_on_no_vol.get("p16", float("nan"))), h0_true, float(s_on_no_vol.get("p84", float("nan"))))
                ),
                "cover68_on_vol": bool(_contains_interval(float(s_on_vol.get("p16", float("nan"))), h0_true, float(s_on_vol.get("p84", float("nan"))))),
                "edge_off": bool(res_off.get("H0_map_at_edge", False)),
                "edge_on_no_vol": bool(res_on_no_vol.get("H0_map_at_edge", False)),
                "edge_on_vol": bool(post_on_vol.get("H0_map_at_edge", False)),
            }
            rows.append(row)
            w.writerow(
                [
                    row["utc"],
                    row["rep"],
                    row["seed"],
                    row["n_events_used"],
                    f"{row['bias_p50_off']:.6g}",
                    f"{row['bias_p50_on_no_vol']:.6g}",
                    f"{row['bias_p50_on_vol']:.6g}",
                    f"{row['bias_map_off']:.6g}",
                    f"{row['bias_map_on_no_vol']:.6g}",
                    f"{row['bias_map_on_vol']:.6g}",
                    int(bool(row["cover68_off"])),
                    int(bool(row["cover68_on_no_vol"])),
                    int(bool(row["cover68_on_vol"])),
                    int(bool(row["edge_off"])),
                    int(bool(row["edge_on_no_vol"])),
                    int(bool(row["edge_on_vol"])),
                ]
            )
            f.flush()

            if (r + 1) % 1 == 0:
                print(f"[injrec-suite] rep {r+1}/{n_reps} done (seed={seed})", flush=True)

        def _agg(key: str) -> dict[str, float]:
            x = np.asarray([float(rr.get(key, float("nan"))) for rr in rows], dtype=float)
            x = x[np.isfinite(x)]
            if x.size == 0:
                return {"n": 0, "mean": float("nan"), "sd": float("nan")}
            return {"n": int(x.size), "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=0))}

        def _rate(key: str) -> dict[str, float]:
            x = np.asarray([bool(rr.get(key, False)) for rr in rows], dtype=bool)
            if x.size == 0:
                return {"n": 0, "rate": float("nan")}
            return {"n": int(x.size), "rate": float(np.mean(x.astype(float)))}

        summary = {
            "h0_true": float(cfg.h0_true),
            "n_reps": int(n_reps),
            "n_events": int(n_events),
            "bias_p50": {"off": _agg("bias_p50_off"), "on_no_vol": _agg("bias_p50_on_no_vol"), "on_vol": _agg("bias_p50_on_vol")},
            "bias_map": {"off": _agg("bias_map_off"), "on_no_vol": _agg("bias_map_on_no_vol"), "on_vol": _agg("bias_map_on_vol")},
            "cover68": {"off": _rate("cover68_off"), "on_no_vol": _rate("cover68_on_no_vol"), "on_vol": _rate("cover68_on_vol")},
            "map_at_edge_rate": {"off": _rate("edge_off"), "on_no_vol": _rate("edge_on_no_vol"), "on_vol": _rate("edge_on_vol")},
        }
        _write_json(out_dir / "summary.json", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

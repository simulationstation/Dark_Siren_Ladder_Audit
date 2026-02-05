from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.gate2_toy import ToyCancellationConfig, ToyPopulationConfig, run_gate2_toy_cancellation
from entropy_horizon_recon.siren_injection_recovery import (
    InjectionRecoveryConfig,
    load_injections_for_recovery,
    run_injection_recovery_gr_h0,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _slope_vs_logH0(H0_grid: np.ndarray, y: np.ndarray) -> float:
    x = np.log(np.asarray(H0_grid, dtype=float))
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if int(np.count_nonzero(m)) < 3:
        return float("nan")
    a, _b = np.polyfit(x[m], y[m], deg=1)
    return float(a)


def _summarize_case(
    *,
    case: str,
    H0_grid: np.ndarray,
    logL_sum_events_rel: np.ndarray | None,
    log_alpha_grid: np.ndarray | None,
    logL_total_rel: np.ndarray,
    H0_map: float,
    H0_map_at_edge: bool,
    p50: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slope_event = _slope_vs_logH0(H0_grid, logL_sum_events_rel) if logL_sum_events_rel is not None else float("nan")
    slope_alpha = _slope_vs_logH0(H0_grid, -log_alpha_grid) if log_alpha_grid is not None else float("nan")
    slope_total = _slope_vs_logH0(H0_grid, logL_total_rel)
    out: dict[str, Any] = {
        "case": str(case),
        "H0_map": float(H0_map),
        "H0_map_at_edge": bool(H0_map_at_edge),
        "p50": float(p50) if p50 is not None else None,
        "slope_event": float(slope_event),
        "slope_minus_log_alpha": float(slope_alpha),
        "slope_total": float(slope_total),
    }
    if extra:
        out.update(extra)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-2 ladder: toy cancellation + incremental injection-recovery checks.")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate2_ladder_<UTCSTAMP>).")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--h0-true", type=float, default=70.0)
    ap.add_argument("--h0-min", type=float, default=40.0)
    ap.add_argument("--h0-max", type=float, default=200.0)
    ap.add_argument("--h0-n", type=int, default=161)

    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=0.62)

    ap.add_argument("--n-events", type=int, default=25, help="Events per ladder rung (injection-recovery steps).")
    ap.add_argument("--pe-n-samples", type=int, default=5000, help="PE samples per event (injection-recovery steps).")
    ap.add_argument("--n-proc", type=int, default=0, help="Processes for hierarchical PE terms (0=all available).")

    ap.add_argument(
        "--inj-file",
        default="data/cache/gw/zenodo/11254021/extracted/GWTC-3-population-data/injections/o3a_bbhpop_inj_info.hdf",
        help="Injection file for ladder (used for selection calibration and event sampling).",
    )
    ap.add_argument(
        "--inj-mass-pdf-coords",
        choices=["m1m2", "m1q"],
        default="m1m2",
        help="Mass-coordinate convention for injection sampling_pdf (default m1m2).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-dist",
        choices=["z", "dL", "log_dL"],
        default="log_dL",
        help="Distance/redshift coordinate used by injection sampling_pdf (default log_dL).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-frame",
        choices=["source", "detector"],
        default="detector",
        help="Mass-frame used by injection sampling_pdf (default detector).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-scale",
        choices=["linear", "log"],
        default="log",
        help="Mass coordinate scale used by injection sampling_pdf (default log).",
    )
    ap.add_argument("--ifar-thresh-yr", type=float, default=1.0)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate2_ladder_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))

    ladder_rows: list[dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Task 1: Known-answer toy cancellation (prior-only PE, p_det in numerator).
    # -------------------------------------------------------------------------
    toy_dir = out_dir / "toy_cancellation"
    toy_cfg = ToyCancellationConfig(
        pop=ToyPopulationConfig(
            omega_m0=float(args.omega_m0),
            omega_k0=float(args.omega_k0),
            z_max=float(args.z_max),
            pop_z_mode="comoving_uniform",
            pop_z_k=0.0,
        ),
        n_events=1,
        pe_n_samples=15_000,
        seed=int(args.seed),
        h0_min=float(args.h0_min),
        h0_max=float(args.h0_max),
        h0_n=int(args.h0_n),
        snr_threshold=8.0,
        snr_norm=25_000.0,
        dL_ref_mpc=1000.0,
        alpha_mc_samples=200_000,
    )
    toy_out = run_gate2_toy_cancellation(toy_cfg, out_dir=toy_dir, n_processes=int(args.n_proc))
    H0_map_toy = float(toy_out["res_event_term"]["H0_map"])
    H0_map_edge_toy = bool(toy_out["res_event_term"]["H0_map_at_edge"])
    logL_total_rel_toy = np.asarray(toy_out["logL_total_rel"], dtype=float)
    flat_range = float(toy_out["diagnostics"]["flat_range_logL"])
    ladder_rows.append(
        _summarize_case(
            case="toy_cancellation_prior_only",
            H0_grid=H0_grid,
            logL_sum_events_rel=np.asarray(toy_out["res_event_term"]["logL_sum_events_rel"], dtype=float),
            log_alpha_grid=np.asarray(toy_out["log_alpha_grid"], dtype=float),
            logL_total_rel=logL_total_rel_toy,
            H0_map=H0_map_toy,
            H0_map_at_edge=H0_map_edge_toy,
            p50=float(toy_out["res_event_term"]["summary"]["p50"]),
            extra={"flat_range_logL": float(flat_range)},
        )
    )

    # -------------------------------------------------------------------------
    # Task 2: Incremental complexity ladder using injection-recovery (self-consistent).
    # -------------------------------------------------------------------------
    inj_path = Path(str(args.inj_file)).expanduser().resolve()
    if not inj_path.exists():
        raise FileNotFoundError(f"Injection file not found: {inj_path}")

    injections = load_injections_for_recovery(inj_path, ifar_threshold_yr=float(args.ifar_thresh_yr))

    base_cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        det_model="snr_binned",
        snr_binned_nbins=200,
        mchirp_binned_nbins=20,
        selection_ifar_thresh_yr=float(args.ifar_thresh_yr),
        pop_z_mode="comoving_uniform",
        pop_z_k=0.0,
        pop_mass_mode="powerlaw_peak_q_smooth",
        pop_m1_alpha=2.3,
        pop_m_min=5.0,
        pop_m_max=80.0,
        pop_q_beta=0.0,
        pop_m_taper_delta=3.0,
        pop_m_peak=35.0,
        pop_m_peak_sigma=5.0,
        pop_m_peak_frac=0.1,
        weight_mode="inv_sampling_pdf",
        inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
        inj_sampling_pdf_dist=str(args.inj_sampling_pdf_dist),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_frame=str(args.inj_sampling_pdf_mass_frame),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_scale=str(args.inj_sampling_pdf_mass_scale),  # type: ignore[arg-type]
        include_pdet_in_event_term=False,
        selection_include_h0_volume_scaling=False,
        pe_obs_mode="noisy",
        pe_n_samples=int(args.pe_n_samples),
        pe_synth_mode="likelihood_resample",
        pe_prior_resample_n_candidates=80_000,
        pe_seed=0,
        dl_frac_sigma0=0.25,
        dl_frac_sigma_floor=0.05,
        dl_sigma_mode="snr",
        mc_frac_sigma0=0.02,
        q_sigma0=0.08,
        pe_prior_dL_expr="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)",
        pe_prior_chirp_mass_expr="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)",
        pe_prior_mass_ratio_expr="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)",
        # Keep partial-support events in the ladder so we can see where/why truncation happens.
        event_qc_mode="skip",
        event_min_finite_frac=0.0,
        importance_smoothing="none",
        importance_truncate_tau=None,
    )

    ladder_cfgs: list[tuple[str, InjectionRecoveryConfig]] = [
        (
            "ladder_0_simple_z_none_mass_none",
            InjectionRecoveryConfig(**{**asdict(base_cfg), "pop_z_mode": "none", "pop_mass_mode": "none"}),
        ),
        (
            "ladder_1_z_comoving_mass_none",
            InjectionRecoveryConfig(**{**asdict(base_cfg), "pop_z_mode": "comoving_uniform", "pop_mass_mode": "none"}),
        ),
        (
            "ladder_2_z_comoving_mass_peak",
            InjectionRecoveryConfig(**{**asdict(base_cfg), "pop_z_mode": "comoving_uniform", "pop_mass_mode": "powerlaw_peak_q_smooth"}),
        ),
        (
            "ladder_3_z_comoving_mass_peak_psis",
            InjectionRecoveryConfig(**{**asdict(base_cfg), "pop_z_mode": "comoving_uniform", "pop_mass_mode": "powerlaw_peak_q_smooth", "importance_smoothing": "psis"}),
        ),
        (
            "ladder_4_z_comoving_mass_peak_trunc",
            InjectionRecoveryConfig(
                **{
                    **asdict(base_cfg),
                    "pop_z_mode": "comoving_uniform",
                    "pop_mass_mode": "powerlaw_peak_q_smooth",
                    "importance_smoothing": "truncate",
                    "importance_truncate_tau": 0.01,
                }
            ),
        ),
        (
            "ladder_5_det_snr_mchirp_binned",
            InjectionRecoveryConfig(**{**asdict(base_cfg), "det_model": "snr_mchirp_binned"}),
        ),
    ]

    for name, cfg in ladder_cfgs:
        rung_dir = out_dir / name
        out = run_injection_recovery_gr_h0(
            injections=injections,
            cfg=cfg,
            n_events=int(args.n_events),
            h0_grid=H0_grid,
            seed=int(args.seed),
            n_processes=int(args.n_proc),
            out_dir=rung_dir,
        )
        res_on = out["gr_h0_selection_on"]
        ladder_rows.append(
            _summarize_case(
                case=name,
                H0_grid=H0_grid,
                logL_sum_events_rel=np.asarray(out["gr_h0_selection_off"]["logL_sum_events_rel"], dtype=float),
                log_alpha_grid=np.asarray(res_on["log_alpha_grid"], dtype=float),
                logL_total_rel=np.asarray(res_on["logL_H0_rel"], dtype=float),
                H0_map=float(res_on["H0_map"]),
                H0_map_at_edge=bool(res_on["H0_map_at_edge"]),
                p50=float(res_on["summary"]["p50"]),
                extra={
                    "h0_true": float(cfg.h0_true),
                    "bias_p50": float(res_on["summary"]["p50"]) - float(cfg.h0_true),
                    "bias_map": float(res_on["H0_map"]) - float(cfg.h0_true),
                    "n_events_used": int(res_on["n_events"]),
                    "pop_z_mode": str(cfg.pop_z_mode),
                    "pop_mass_mode": str(cfg.pop_mass_mode),
                    "importance_smoothing": str(cfg.importance_smoothing),
                    "det_model": str(cfg.det_model),
                },
            )
        )

    # Write machine-readable summary.
    _write_json(out_dir / "ladder_summary.json", {"created_utc": _utc_stamp(), "H0_grid": [float(x) for x in H0_grid.tolist()], "rows": ladder_rows})

    # Write CSV for quick heuristic trend checks.
    csv_path = out_dir / "ladder.csv"
    cols: list[str] = []
    for r in ladder_rows:
        for k in r.keys():
            if k not in cols:
                cols.append(str(k))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ladder_rows:
            w.writerow({k: r.get(k) for k in cols})

    print(f"gate2_ladder: wrote {len(ladder_rows)} rows to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

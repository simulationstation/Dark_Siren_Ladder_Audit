from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    compute_selection_alpha_h0_grid_for_cfg,
    infer_z_max_for_h0_grid_closed_loop,
    load_injections_for_recovery,
    run_injection_recovery_gr_h0,
)

# Global state for multiprocessing workers (avoid pickling large arrays per task).
_G_INJECTIONS = None
_G_CFG = None
_G_N_EVENTS = None
_G_H0_GRID = None
_G_ALPHA_GRID = None
_G_ALPHA_META = None
_G_JSON_DIR = None


def _run_rep_worker(rep: int, seed: int) -> dict[str, Any]:
    global _G_INJECTIONS, _G_CFG, _G_N_EVENTS, _G_H0_GRID, _G_ALPHA_GRID, _G_ALPHA_META, _G_JSON_DIR  # noqa: PLW0603
    if _G_INJECTIONS is None or _G_CFG is None or _G_N_EVENTS is None or _G_H0_GRID is None or _G_ALPHA_GRID is None or _G_JSON_DIR is None:
        raise RuntimeError("Worker globals not initialized.")
    rep_path = Path(_G_JSON_DIR) / f"rep_{int(rep):04d}.json"
    if rep_path.exists():
        return {"rep": int(rep), "seed": int(seed), "skipped": True}

    out = run_injection_recovery_gr_h0(
        injections=_G_INJECTIONS,
        cfg=_G_CFG,
        n_events=int(_G_N_EVENTS),
        h0_grid=_G_H0_GRID,
        seed=int(seed),
        selection_alpha_h0_grid=_G_ALPHA_GRID,
        selection_alpha_meta=_G_ALPHA_META,
        out_dir=None,
    )

    tmp = rep_path.with_suffix(".json.tmp")
    _write_json(tmp, out)
    tmp.replace(rep_path)

    s = out.get("summary", {})
    return {
        "rep": int(rep),
        "seed": int(seed),
        "used_on": int(s.get("n_events_used_selection_on", -1)),
        "p50_on": _safe_float(s.get("selection_on", {}).get("summary", {}).get("p50")),
        "bias_p50_on": _safe_float(s.get("bias_p50_selection_on")),
        "skipped": False,
    }


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _load_summary_from_rep(rep_path: Path) -> dict[str, Any]:
    d = json.loads(rep_path.read_text())
    s = dict(d.get("summary", {}))
    h0_true = _safe_float(s.get("h0_true"))

    def _q_from_grid(grid: list[float], posterior: list[float], q: float) -> float:
        g = np.asarray(grid, dtype=float)
        p = np.asarray(posterior, dtype=float)
        if g.ndim != 1 or p.ndim != 1 or g.shape != p.shape or g.size < 2:
            return float("nan")
        p = np.where(np.isfinite(p), p, 0.0)
        psum = float(np.sum(p))
        if not (np.isfinite(psum) and psum > 0.0):
            return float("nan")
        p = p / psum
        cdf = np.cumsum(p)
        return float(np.interp(float(q), cdf, g))

    on = dict(d.get("gr_h0_selection_on", {}))
    off = dict(d.get("gr_h0_selection_off", {}))
    h0_grid = on.get("H0_grid", off.get("H0_grid", []))

    q025_off = _q_from_grid(h0_grid, off.get("posterior", []), 0.025)
    q16_off = _q_from_grid(h0_grid, off.get("posterior", []), 0.16)
    q975_off = _q_from_grid(h0_grid, off.get("posterior", []), 0.975)
    q025_on = _q_from_grid(h0_grid, on.get("posterior", []), 0.025)
    q16_on = _q_from_grid(h0_grid, on.get("posterior", []), 0.16)
    q975_on = _q_from_grid(h0_grid, on.get("posterior", []), 0.975)
    q84_off = _q_from_grid(h0_grid, off.get("posterior", []), 0.84)
    q84_on = _q_from_grid(h0_grid, on.get("posterior", []), 0.84)

    pp = dict(s.get("pe_pp", {}))
    pp_dL = dict(pp.get("dL", {}))

    # Promote a minimal set of stable keys.
    return {
        "rep": int(rep_path.stem.split("_")[-1]),
        "seed": int(d.get("manifest", {}).get("seed", -1)),
        "h0_true": float(h0_true),
        "n_events_truth": int(s.get("n_events_truth", -1)),
        "n_used_off": int(s.get("n_events_used_selection_off", -1)),
        "n_used_on": int(s.get("n_events_used_selection_on", -1)),
        "n_skipped_off": int(s.get("n_events_skipped_selection_off", -1)),
        "n_skipped_on": int(s.get("n_events_skipped_selection_on", -1)),
        "H0_map_off": _safe_float(s.get("selection_off", {}).get("H0_map")),
        "H0_map_on": _safe_float(s.get("selection_on", {}).get("H0_map")),
        "H0_p025_off": float(q025_off),
        "H0_p16_off": float(q16_off),
        "H0_p50_off": _safe_float(s.get("selection_off", {}).get("summary", {}).get("p50")),
        "H0_p84_off": float(q84_off),
        "H0_p975_off": float(q975_off),
        "H0_p025_on": float(q025_on),
        "H0_p16_on": float(q16_on),
        "H0_p50_on": _safe_float(s.get("selection_on", {}).get("summary", {}).get("p50")),
        "H0_p84_on": float(q84_on),
        "H0_p975_on": float(q975_on),
        "bias_p50_on": _safe_float(s.get("bias_p50_selection_on")),
        "bias_map_on": _safe_float(s.get("bias_map_selection_on")),
        "pp_dL_mean": _safe_float(pp_dL.get("mean")),
        "pp_dL_ks": _safe_float(pp_dL.get("ks_d")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Run a resumable injection-recovery suite (many reps) for the GR H0 control.")
    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")

    ap.add_argument("--h0-true", type=float, required=True, help="Truth H0 used to generate synthetic detected events.")
    ap.add_argument("--n-events", type=int, default=50, help="Synthetic detected events per replicate (default 50).")
    ap.add_argument("--n-rep", type=int, default=32, help="Number of replicates (default 32).")
    ap.add_argument("--seed0", type=int, default=1000, help="Base seed (default 1000).")

    ap.add_argument("--h0-min", type=float, default=40.0, help="Min H0 in inference grid (default 40).")
    ap.add_argument("--h0-max", type=float, default=120.0, help="Max H0 in inference grid (default 120).")
    ap.add_argument("--h0-n", type=int, default=161, help="Number of H0 grid points (default 161).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 for GR distances (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 for GR distances (default 0).")
    ap.add_argument("--z-max", type=float, default=0.62, help="Max redshift used in inference + selection proxy (default 0.62).")
    ap.add_argument(
        "--z-max-mode",
        choices=["fixed", "auto"],
        default="auto",
        help="z_max policy: fixed uses --z-max; auto expands z_max so the H0 grid doesn't induce artificial support truncation (default auto).",
    )
    ap.add_argument("--z-max-auto-cap", type=float, default=5.0, help="Max z used for the auto z_max inversion cache (default 5).")
    ap.add_argument("--z-max-auto-margin", type=float, default=0.10, help="Additive safety margin on inferred z_max (default 0.10).")

    ap.add_argument("--det-model", choices=["threshold", "snr_binned"], default="snr_binned", help="Detectability proxy model (default snr_binned).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for det_model=snr_binned (default 200).")
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf", help="Injection weight mode (default inv_sampling_pdf).")
    ap.add_argument(
        "--inj-mass-pdf-coords",
        choices=["m1m2", "m1q"],
        default="m1m2",
        help="Mass-coordinate convention for injection sampling_pdf (default m1m2).",
    )
    ap.add_argument(
        "--include-pdet-in-event-term",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include p_det(dL) in the hierarchical PE event term (audit knob; default False).",
    )

    ap.add_argument(
        "--selection-include-h0-volume-scaling",
        action="store_true",
        help="Use an xi-style normalization by including an H0^{-3} factor in the selection term (audit/debug knob).",
    )

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform", help="Population z mode (default comoving_uniform).")
    ap.add_argument("--pop-z-k", type=float, default=0.0, help="Powerlaw k for pop_z_mode=comoving_powerlaw (default 0).")
    ap.add_argument(
        "--pop-z-include-h0-volume-scaling",
        action="store_true",
        help="Include an explicit (c/H0)^3 factor in the pop_z weight for the GR(H0) control (audit/debug knob).",
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
        help="Synthetic PE mode (default likelihood_resample).",
    )
    ap.add_argument("--pe-prior-resample-n-candidates", type=int, default=200_000, help="Candidates for prior_resample (default 200000).")
    ap.add_argument("--pe-seed", type=int, default=0, help="Seed offset for PE synthesis (default 0).")
    ap.add_argument("--dl-frac-sigma0", type=float, default=0.25, help="Base fractional dL scatter (default 0.25).")
    ap.add_argument("--dl-frac-sigma-floor", type=float, default=0.05, help="Minimum fractional dL scatter under SNR scaling (default 0.05).")
    ap.add_argument("--dl-sigma-mode", choices=["constant", "snr"], default="snr", help="Distance scatter model (default snr).")
    ap.add_argument("--mc-frac-sigma0", type=float, default=0.02, help="Base fractional chirp-mass scatter (default 0.02).")
    ap.add_argument("--q-sigma0", type=float, default=0.08, help="Mass-ratio scatter (default 0.08).")
    ap.add_argument("--pe-prior-dl-expr", default="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)")
    ap.add_argument("--pe-prior-chirp-mass-expr", default="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)")
    ap.add_argument("--pe-prior-mass-ratio-expr", default="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)")

    ap.add_argument("--event-qc-mode", choices=["fail", "skip"], default="skip", help="Event QC mode (default skip).")
    ap.add_argument(
        "--event-min-finite-frac",
        type=float,
        default=0.0,
        help="Minimum finite-support fraction across H0 grid (default 0; disables 'insufficient support' skipping).",
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
        help="Parallel worker processes for replicates (default 0=auto; capped by n_rep).",
    )
    ap.add_argument("--rebuild-only", action="store_true", help="Do not run new reps; rebuild tables/figures from existing rep_*.json files.")
    ap.add_argument("--smoke", action="store_true", help="Tiny suite smoke (few reps + few events + small grid).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_injection_recovery_suite_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_injection_recovery_suite_{_utc_stamp()}"
    json_dir = out_dir / "json"
    tab_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    json_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    if _HAVE_MPL:
        fig_dir.mkdir(parents=True, exist_ok=True)

    n_rep = int(args.n_rep)
    n_events = int(args.n_events)
    h0_n = int(args.h0_n)
    if bool(args.smoke):
        n_rep = min(n_rep, 3)
        n_events = min(n_events, 10)
        h0_n = min(h0_n, 51)

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(h0_n))

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
        selection_ifar_thresh_yr=float(args.selection_ifar_thresh_yr),
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
        selection_include_h0_volume_scaling=bool(args.selection_include_h0_volume_scaling),
        include_pdet_in_event_term=bool(args.include_pdet_in_event_term),
        pe_obs_mode=str(args.pe_obs_mode),  # type: ignore[arg-type]
        pe_n_samples=int(args.pe_n_samples),
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
        importance_smoothing=str(args.importance_smoothing),  # type: ignore[arg-type]
        importance_truncate_tau=float(args.importance_truncate_tau) if args.importance_truncate_tau is not None else None,
    )

    manifest = {
        "created_utc": _utc_stamp(),
        "config": cfg.__dict__,
        "h0_grid": [float(x) for x in H0_grid.tolist()],
        "n_rep": int(n_rep),
        "n_events": int(n_events),
        "seed0": int(args.seed0),
    }
    _write_json(out_dir / "manifest.json", manifest)

    injections = load_injections_for_recovery(args.selection_injections_hdf, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))
    alpha_grid, alpha_meta = compute_selection_alpha_h0_grid_for_cfg(injections=injections, cfg=cfg, h0_grid=H0_grid)
    _write_json(out_dir / "selection_alpha.json", {"selection_alpha": alpha_meta, "selection_alpha_grid": [float(x) for x in alpha_grid.tolist()]})

    if not bool(args.rebuild_only):
        existing = sorted(json_dir.glob("rep_*.json"))
        (out_dir / "progress.json").write_text(json.dumps({"n_done": len(existing), "n_target": int(n_rep), "updated_utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n")

        todo: list[tuple[int, int]] = []
        for r in range(int(n_rep)):
            rep = r + 1
            seed = int(args.seed0) + r
            rep_path = json_dir / f"rep_{rep:04d}.json"
            if rep_path.exists():
                continue
            todo.append((int(rep), int(seed)))

        n_proc = int(args.n_proc)
        if bool(args.smoke):
            n_proc = min(max(n_proc, 1), 2) if n_proc > 0 else 1
        if n_proc <= 0:
            n_proc = int(os.cpu_count() or 1)
        n_proc = max(1, min(n_proc, int(n_rep)))

        if not todo:
            print(f"[note] all replicates already present ({len(existing)}/{int(n_rep)}); skipping run.", flush=True)
        elif n_proc == 1:
            for rep, seed in todo:
                out = run_injection_recovery_gr_h0(
                    injections=injections,
                    cfg=cfg,
                    n_events=int(n_events),
                    h0_grid=H0_grid,
                    seed=int(seed),
                    selection_alpha_h0_grid=alpha_grid,
                    selection_alpha_meta=alpha_meta,
                    out_dir=None,
                )
                rep_path = json_dir / f"rep_{rep:04d}.json"
                _write_json(rep_path, out)
                s = out.get("summary", {})
                print(
                    f"[rep {rep:04d}/{int(n_rep)}] seed={seed} used_on={s.get('n_events_used_selection_on')} "
                    f"p50_on={s.get('selection_on', {}).get('summary', {}).get('p50'):.3f} "
                    f"bias_p50_on={s.get('bias_p50_selection_on'):.3f}",
                    flush=True,
                )
                done = sorted(json_dir.glob("rep_*.json"))
                (out_dir / "progress.json").write_text(json.dumps({"n_done": len(done), "n_target": int(n_rep), "updated_utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n")
        else:
            # Initialize globals for forked workers.
            global _G_INJECTIONS, _G_CFG, _G_N_EVENTS, _G_H0_GRID, _G_ALPHA_GRID, _G_ALPHA_META, _G_JSON_DIR  # noqa: PLW0603
            _G_INJECTIONS = injections
            _G_CFG = cfg
            _G_N_EVENTS = int(n_events)
            _G_H0_GRID = H0_grid
            _G_ALPHA_GRID = alpha_grid
            _G_ALPHA_META = alpha_meta
            _G_JSON_DIR = str(json_dir)

            done_n = int(len(existing))
            max_workers = min(int(n_proc), int(len(todo)))
            print(f"[suite] launching {len(todo)} reps with n_proc={max_workers} (existing={done_n})", flush=True)
            with ProcessPoolExecutor(max_workers=max_workers) as ex:
                futs = {ex.submit(_run_rep_worker, rep, seed): (rep, seed) for rep, seed in todo}
                for fut in as_completed(futs):
                    rep, seed = futs[fut]
                    res = fut.result()
                    done_n += 1
                    if not bool(res.get("skipped", False)):
                        print(
                            f"[rep {rep:04d}/{int(n_rep)}] seed={seed} used_on={res.get('used_on')} "
                            f"p50_on={float(res.get('p50_on', float('nan'))):.3f} "
                            f"bias_p50_on={float(res.get('bias_p50_on', float('nan'))):.3f}",
                            flush=True,
                        )
                    (out_dir / "progress.json").write_text(json.dumps({"n_done": done_n, "n_target": int(n_rep), "updated_utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n")

    # Build summary table from all rep files.
    rep_paths = sorted(json_dir.glob("rep_*.json"))
    rows = [_load_summary_from_rep(p) for p in rep_paths]
    rows = sorted(rows, key=lambda x: int(x.get("rep", 0)))
    if not rows:
        raise ValueError("No rep_*.json files found; nothing to summarize.")

    # Aggregate P–P values across all events in all reps.
    pp_all_dL: list[float] = []
    for p in rep_paths:
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        by_ev = d.get("pe_pp_by_event", {})
        if not isinstance(by_ev, dict):
            continue
        for vv in by_ev.values():
            if not isinstance(vv, dict):
                continue
            u = vv.get("u_dL")
            try:
                u = float(u)
            except Exception:
                continue
            if np.isfinite(u):
                pp_all_dL.append(float(np.clip(u, 0.0, 1.0)))

    csv_path = tab_dir / "suite_summary.csv"
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    # Aggregate stats.
    bias = np.asarray([_safe_float(r["bias_p50_on"]) for r in rows], dtype=float)
    used_on = np.asarray([int(r["n_used_on"]) for r in rows], dtype=float)
    skipped_on = np.asarray([int(r["n_skipped_on"]) for r in rows], dtype=float)
    h0_true = float(args.h0_true)
    p16_on = np.asarray([_safe_float(r.get("H0_p16_on")) for r in rows], dtype=float)
    p84_on = np.asarray([_safe_float(r.get("H0_p84_on")) for r in rows], dtype=float)
    p025_on = np.asarray([_safe_float(r.get("H0_p025_on")) for r in rows], dtype=float)
    p975_on = np.asarray([_safe_float(r.get("H0_p975_on")) for r in rows], dtype=float)
    cov68 = float(np.nanmean((h0_true >= p16_on) & (h0_true <= p84_on)))
    cov95 = float(np.nanmean((h0_true >= p025_on) & (h0_true <= p975_on)))
    pp_dL_ks = np.asarray([_safe_float(r.get("pp_dL_ks")) for r in rows], dtype=float)

    pp_all_dL_arr = np.asarray(pp_all_dL, dtype=float)
    if pp_all_dL_arr.size:
        pp_all_dL_sorted = np.sort(pp_all_dL_arr)
        i = np.arange(1, pp_all_dL_sorted.size + 1, dtype=float)
        d_plus = float(np.max(i / pp_all_dL_sorted.size - pp_all_dL_sorted))
        d_minus = float(np.max(pp_all_dL_sorted - (i - 1.0) / pp_all_dL_sorted.size))
        pp_all_ks = float(max(d_plus, d_minus))
        pp_all_mean = float(np.mean(pp_all_dL_sorted))
    else:
        pp_all_ks = float("nan")
        pp_all_mean = float("nan")
    agg = {
        "n_rep_done": int(len(rows)),
        "h0_true": h0_true,
        "bias_p50_on_mean": float(np.nanmean(bias)),
        "bias_p50_on_sd": float(np.nanstd(bias)),
        "used_on_mean": float(np.nanmean(used_on)),
        "used_on_min": int(np.nanmin(used_on)),
        "used_on_max": int(np.nanmax(used_on)),
        "skipped_on_mean": float(np.nanmean(skipped_on)),
        "coverage_68_on": cov68,
        "coverage_95_on": cov95,
        "pp_dL_ks_mean": float(np.nanmean(pp_dL_ks)),
        "pp_dL_all_n": int(pp_all_dL_arr.size),
        "pp_dL_all_mean": float(pp_all_mean),
        "pp_dL_all_ks": float(pp_all_ks),
        "note": "Calibration/audit diagnostic. For a well-calibrated synthetic PE generator, per-event truth percentiles (P–P) should be ~Uniform[0,1] and coverage_68_on should be near 0.68.",
    }
    _write_json(tab_dir / "suite_aggregate.json", agg)

    if _HAVE_MPL:
        assert plt is not None
        # Figures.
        plt.figure(figsize=(7.2, 4.0))
        plt.hist(np.clip(bias, -100, 100), bins=30, alpha=0.85)
        plt.axvline(0.0, color="k", lw=1.2)
        plt.xlabel(r"bias in $H_0$ p50 (selection ON)  [$\mathrm{km/s/Mpc}$]")
        plt.ylabel("replicate count")
        plt.title("Injection-recovery bias distribution (selection ON)")
        plt.tight_layout()
        plt.savefig(fig_dir / "bias_p50_on_hist.png", dpi=160)
        plt.close()

        plt.figure(figsize=(7.2, 4.0))
        plt.plot([r["rep"] for r in rows], [r["bias_p50_on"] for r in rows], marker="o", lw=1.2, ms=3)
        plt.axhline(0.0, color="k", lw=1.0)
        plt.xlabel("replicate")
        plt.ylabel(r"bias in $H_0$ p50 (selection ON)")
        plt.title("Injection-recovery per-replicate bias")
        plt.tight_layout()
        plt.savefig(fig_dir / "bias_p50_on_by_rep.png", dpi=160)
        plt.close()

        if pp_all_dL_arr.size:
            # P–P diagnostic: empirical CDF vs uniform line.
            u = np.sort(pp_all_dL_arr)
            n_u = int(u.size)
            plt.figure(figsize=(4.8, 4.8))
            plt.plot(u, (np.arange(1, n_u + 1) / n_u), lw=1.8, label="empirical")
            plt.plot([0, 1], [0, 1], color="k", lw=1.0, ls="--", label="uniform")
            plt.xlabel("truth percentile u (dL)")
            plt.ylabel("empirical CDF")
            plt.title("Synthetic PE distance P–P")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(fig_dir / "pp_dL_cdf.png", dpi=160)
            plt.close()

            plt.figure(figsize=(7.2, 4.0))
            plt.hist(u, bins=20, range=(0, 1), alpha=0.85)
            plt.xlabel("truth percentile u (dL)")
            plt.ylabel("count")
            plt.title("Synthetic PE distance P–P histogram")
            plt.tight_layout()
            plt.savefig(fig_dir / "pp_dL_hist.png", dpi=160)
            plt.close()
    else:
        print("[note] matplotlib not available; skipping figures", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_siren_h0 import compute_gr_h0_posterior_grid_hierarchical_pe
from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


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
            snr_net_opt_ref=float(meta["snr_net_opt_ref"]) if "snr_net_opt_ref" in meta else None,
            dL_mpc_ref=float(meta["dL_mpc_ref"]) if "dL_mpc_ref" in meta else None,
        )
        return ev, pe_obj


@dataclass(frozen=True)
class PopConfig:
    """Small population config record for Gate-2 marginalization/sensitivity averages."""

    name: str
    pop_z_mode: str
    pop_z_k: float
    pop_mass_mode: str


def _default_pop_grid() -> list[PopConfig]:
    # A small, intentionally simple grid: vary z evolution and two common mass modes.
    # This is *not* meant to be an astrophysically complete prior; it’s a stability/sensitivity audit.
    out: list[PopConfig] = []
    for mass_mode in ("powerlaw_peak_q_smooth", "powerlaw_q_smooth"):
        out.append(PopConfig(name=f"z_none__{mass_mode}", pop_z_mode="none", pop_z_k=0.0, pop_mass_mode=mass_mode))
        out.append(PopConfig(name=f"z_comoving_uniform__{mass_mode}", pop_z_mode="comoving_uniform", pop_z_k=0.0, pop_mass_mode=mass_mode))
        for k in (-2.0, 2.0, 4.0):
            out.append(PopConfig(name=f"z_powerlaw_k{k:+.0f}__{mass_mode}", pop_z_mode="comoving_powerlaw", pop_z_k=float(k), pop_mass_mode=mass_mode))
    return out


def _normalize(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Posterior normalization failed (non-positive sum).")
    return p / s


def _summary_from_posterior(H0: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = _normalize(p)
    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0))
    q50 = float(np.interp(0.50, cdf, H0))
    q84 = float(np.interp(0.84, cdf, H0))
    mean = float(np.sum(p * H0))
    sd = float(np.sqrt(np.sum(p * (H0 - mean) ** 2)))
    H0_map = float(H0[int(np.argmax(p))])
    return {"H0_map": H0_map, "mean": mean, "sd": sd, "p16": q16, "p50": q50, "p84": q84}


def _parse_pop_grid_json(path: Path) -> list[PopConfig]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError("--pop-grid-json must be a JSON list of objects.")
    out: list[PopConfig] = []
    for i, rec in enumerate(obj):
        if not isinstance(rec, dict):
            raise ValueError(f"pop-grid[{i}] must be an object.")
        out.append(
            PopConfig(
                name=str(rec.get("name", f"cfg_{i:02d}")),
                pop_z_mode=str(rec.get("pop_z_mode", "none")),
                pop_z_k=float(rec.get("pop_z_k", 0.0)),
                pop_mass_mode=str(rec.get("pop_mass_mode", "none")),
            )
        )
    return out


def _print_one(prefix: str, res: dict[str, Any]) -> None:
    s = res.get("summary", {}) or {}
    print(
        f"{prefix} n_events={int(res.get('n_events', -1))} "
        f"H0_map={float(res.get('H0_map', float('nan'))):.3f} "
        f"edge={bool(res.get('H0_map_at_edge', False))} "
        f"p50={float(s.get('p50', float('nan'))):.3f} "
        f"[p16,p84]=[{float(s.get('p16', float('nan'))):.3f},{float(s.get('p84', float('nan'))):.3f}]",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-2: GR H0 control with population-grid averaging (spectral-only).")
    ap.add_argument("--pe-cache-dir", required=True, help="Directory containing cached hierarchical PE .npz files (pe_hier_<EVENT>.npz).")
    ap.add_argument("--from-dark-siren-summary", required=True, help="A summary JSON containing z_max and optionally events_scored.")
    ap.add_argument("--events-from-pe-cache", action="store_true", help="Use all pe_hier_*.npz files from --pe-cache-dir.")
    ap.add_argument("--events", default=None, help="Optional comma list of events; overrides summary events_scored.")

    ap.add_argument("--h0-min", type=float, default=40.0)
    ap.add_argument("--h0-max", type=float, default=100.0)
    ap.add_argument("--h0-n", type=int, default=241)
    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=None, help="Override z_max (default from summary).")

    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0)
    ap.add_argument("--det-model", choices=["threshold", "snr_binned", "snr_mchirp_binned"], default="snr_binned")
    ap.add_argument("--snr-thresh", type=float, default=None)
    ap.add_argument("--snr-binned-nbins", type=int, default=200)
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20)
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="inv_sampling_pdf")

    # Population baseline knobs (grid items may override modes only; parameters are shared).
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3)
    ap.add_argument("--pop-m-min", type=float, default=5.0)
    ap.add_argument("--pop-m-max", type=float, default=80.0)
    ap.add_argument("--pop-q-beta", type=float, default=0.0)
    ap.add_argument("--pop-m-taper-delta", type=float, default=3.0)
    ap.add_argument("--pop-m-peak", type=float, default=35.0)
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0)
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1)

    ap.add_argument("--event-qc-mode", choices=["fail", "skip"], default="skip")
    ap.add_argument("--event-min-finite-frac", type=float, default=0.0, help="Minimum finite-support fraction across H0 grid (default 0; disables 'insufficient support' skipping).")

    ap.add_argument("--pop-grid-json", default=None, help="JSON list of population configs (name,pop_z_mode,pop_z_k,pop_mass_mode).")
    ap.add_argument("--pop-grid-mode", choices=["default"], default="default", help="Built-in pop grid selector (default: default).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate2_popmarg_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate2_popmarg_{_utc_stamp()}"
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
    z_max = float(args.z_max) if args.z_max is not None else float(summary.get("z_max", 0.0))
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("Invalid z_max (provide --z-max or ensure summary has z_max).")

    if bool(args.events_from_pe_cache):
        events: list[str] = []
        for p in sorted(pe_cache_dir.glob("pe_hier_*.npz")):
            stem = p.stem
            if not stem.startswith("pe_hier_"):
                continue
            events.append(stem[len("pe_hier_") :])
    elif args.events is not None:
        events = [e.strip() for e in str(args.events).split(",") if e.strip()]
    else:
        events = [str(x) for x in summary.get("events_scored", [])]

    if not events:
        raise ValueError("No events selected (use --events, --events-from-pe-cache, or ensure summary has events_scored).")

    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = {}
    for ev in sorted(events):
        p = pe_cache_dir / f"pe_hier_{ev}.npz"
        if not p.exists():
            continue
        ev2, pe = _load_pe_hier_cache_npz(p)
        pe_by_event[ev2 or ev] = pe
    if not pe_by_event:
        raise ValueError("No PE cache files loaded; check --pe-cache-dir and event list.")

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))
    injections = load_o3_injections(Path(args.selection_injections_hdf).expanduser().resolve(), ifar_threshold_yr=float(args.selection_ifar_thresh_yr))

    if args.pop_grid_json is not None:
        pop_grid = _parse_pop_grid_json(Path(args.pop_grid_json).expanduser().resolve())
    else:
        pop_grid = _default_pop_grid()
    if not pop_grid:
        raise ValueError("Empty population grid.")

    # Run each config and collect normalized posteriors.
    per_cfg: list[dict[str, Any]] = []
    p_list: list[np.ndarray] = []

    for cfg in pop_grid:
        res = compute_gr_h0_posterior_grid_hierarchical_pe(
            pe_by_event=pe_by_event,
            H0_grid=H0_grid,
            omega_m0=float(args.omega_m0),
            omega_k0=float(args.omega_k0),
            z_max=float(z_max),
            cache_dir=cache_dir / cfg.name,
            include_pdet_in_event_term=False,
            pop_z_include_h0_volume_scaling=False,
            injections=injections,
            ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
            det_model=str(args.det_model),  # type: ignore[arg-type]
            snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
            snr_binned_nbins=int(args.snr_binned_nbins),
            mchirp_binned_nbins=int(args.mchirp_binned_nbins),
            weight_mode=str(args.weight_mode),  # type: ignore[arg-type]
            pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
            pop_z_powerlaw_k=float(cfg.pop_z_k),
            pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(args.pop_m1_alpha),
            pop_m_min=float(args.pop_m_min),
            pop_m_max=float(args.pop_m_max),
            pop_q_beta=float(args.pop_q_beta),
            pop_m_taper_delta=float(args.pop_m_taper_delta),
            pop_m_peak=float(args.pop_m_peak),
            pop_m_peak_sigma=float(args.pop_m_peak_sigma),
            pop_m_peak_frac=float(args.pop_m_peak_frac),
            event_qc_mode=str(args.event_qc_mode),  # type: ignore[arg-type]
            event_min_finite_frac=float(args.event_min_finite_frac),
            prior="uniform",
        )
        _print_one(f"[pop {cfg.name}]", res)
        (json_dir / f"gr_h0_selection_on_{cfg.name}.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
        p = _normalize(np.asarray(res["posterior"], dtype=float))
        p_list.append(p)
        per_cfg.append({"config": cfg.__dict__, "result_summary": dict(res.get("summary", {})), "H0_map_at_edge": bool(res.get("H0_map_at_edge", False))})

    # Equal-weight mixture of conditional posteriors (stability average; not evidence-weighted).
    p_mix = _normalize(np.mean(np.stack(p_list, axis=0), axis=0))
    mix_summary = _summary_from_posterior(H0_grid, p_mix)

    p50s = [float(x.get("result_summary", {}).get("p50", float("nan"))) for x in per_cfg]
    p50s_f = [x for x in p50s if np.isfinite(x)]
    spread = {"p50_min": float(np.min(p50s_f)) if p50s_f else float("nan"), "p50_max": float(np.max(p50s_f)) if p50s_f else float("nan")}

    out = {
        "timestamp_utc": _utc_stamp(),
        "method": "gate2_gr_h0_pop_grid_mixture",
        "note": "Mixture posterior is an equal-weight average of per-config conditional posteriors; not evidence-weighted across population configs.",
        "inputs": {
            "from_dark_siren_summary": str(summary_path),
            "pe_cache_dir": str(pe_cache_dir),
            "n_events": int(len(pe_by_event)),
            "events": sorted(pe_by_event.keys()),
            "H0_grid": [float(x) for x in H0_grid.tolist()],
            "omega_m0": float(args.omega_m0),
            "omega_k0": float(args.omega_k0),
            "z_max": float(z_max),
            "selection_injections_hdf": str(Path(args.selection_injections_hdf).expanduser().resolve()),
            "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
            "det_model": str(args.det_model),
            "snr_threshold": float(args.snr_thresh) if args.snr_thresh is not None else None,
            "snr_binned_nbins": int(args.snr_binned_nbins),
            "mchirp_binned_nbins": int(args.mchirp_binned_nbins),
            "weight_mode": str(args.weight_mode),
            "pop_params_shared": {
                "pop_m1_alpha": float(args.pop_m1_alpha),
                "pop_m_min": float(args.pop_m_min),
                "pop_m_max": float(args.pop_m_max),
                "pop_q_beta": float(args.pop_q_beta),
                "pop_m_taper_delta": float(args.pop_m_taper_delta),
                "pop_m_peak": float(args.pop_m_peak),
                "pop_m_peak_sigma": float(args.pop_m_peak_sigma),
                "pop_m_peak_frac": float(args.pop_m_peak_frac),
            },
        },
        "pop_grid": [cfg.__dict__ for cfg in pop_grid],
        "per_config": per_cfg,
        "mixture": {"posterior": [float(x) for x in p_mix.tolist()], "summary": mix_summary, "p50_spread": spread},
    }
    (json_dir / "pop_marginal_summary.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    # Plot overlay.
    plt.figure(figsize=(8.5, 4.6))
    for cfg, p in zip(pop_grid, p_list, strict=True):
        plt.plot(H0_grid, p, lw=1.0, alpha=0.7, label=cfg.name)
    plt.plot(H0_grid, p_mix, "k-", lw=2.2, label="mixture (equal-weight)")
    plt.xlabel(r"$H_0$ [km/s/Mpc]")
    plt.ylabel("posterior (normalized on grid)")
    plt.title("Gate 2 GR $H_0$ control: population-grid mixture (spectral-only)")
    plt.legend(frameon=False, fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(fig_dir / "gr_h0_popgrid_mixture.png", dpi=170)
    plt.close()

    print(f"[gate2_popmarg] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

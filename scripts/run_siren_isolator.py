from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples, load_gwtc_pe_hierarchical_samples
from entropy_horizon_recon.gwtc_pe_index import build_gwtc_pe_index
from entropy_horizon_recon.siren_isolator import (
    ScrambleMode,
    apply_hierarchical_pe_scramble,
    score_event_hierarchical_pe,
    stable_int_seed,
)
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_csv_list(s: str | None) -> list[str]:
    if s is None:
        return []
    out: list[str] = []
    for tok in str(s).split(","):
        tok = tok.strip()
        if tok:
            out.append(tok)
    return out


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


def _select_posterior_draws(post: MuForwardPosterior, draw_idx: list[int]) -> MuForwardPosterior:
    idx = np.asarray(draw_idx, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("posterior_draw_idx must be a non-empty 1D list.")
    n = int(post.H_samples.shape[0])
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"posterior_draw_idx out of range for posterior with n_draws={n}.")
    # Preserve ordering from the summary file.
    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=np.asarray(post.logmu_x_samples[idx], dtype=float),
        z_grid=post.z_grid,
        H_samples=np.asarray(post.H_samples[idx], dtype=float),
        H0=np.asarray(post.H0[idx], dtype=float),
        omega_m0=np.asarray(post.omega_m0[idx], dtype=float),
        omega_k0=np.asarray(post.omega_k0[idx], dtype=float),
        sigma8_0=np.asarray(post.sigma8_0[idx], dtype=float) if post.sigma8_0 is not None else None,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                cols.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# Global worker state for fork-based event parallelism.
_WORKER_STATE: dict[str, Any] = {}


def _score_one_event_worker(ev: str) -> dict[str, Any]:
    st = _WORKER_STATE
    mode_label: str = st["mode_label"]
    mode: str = st["mode"]
    post: MuForwardPosterior = st["post"]
    z_max: float = float(st["z_max"])
    convention: str = st["convention"]
    pop_z_mode: str = st["pop_z_mode"]
    pop_z_k: float = float(st["pop_z_k"])
    pop_mass_mode: str = st["pop_mass_mode"]
    pop_m1_alpha: float = float(st["pop_m1_alpha"])
    pop_m_min: float = float(st["pop_m_min"])
    pop_m_max: float = float(st["pop_m_max"])
    pop_q_beta: float = float(st["pop_q_beta"])
    pop_m_taper_delta: float = float(st.get("pop_m_taper_delta", 0.0))
    pop_m_peak: float = float(st.get("pop_m_peak", 35.0))
    pop_m_peak_sigma: float = float(st.get("pop_m_peak_sigma", 5.0))
    pop_m_peak_frac: float = float(st.get("pop_m_peak_frac", 0.1))
    scramble_seed: int = int(st["scramble_seed"])
    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = st["pe_by_event"]
    dL_swap_map: dict[str, str] | None = st.get("dL_swap_map")
    log_alpha_mu: np.ndarray | None = st.get("log_alpha_mu")
    log_alpha_gr: np.ndarray | None = st.get("log_alpha_gr")

    pe0 = pe_by_event.get(str(ev))
    if pe0 is None:
        return {"ok": False, "event": str(ev), "error": "missing PE entry"}

    try:
        if mode == "swap_dL_events":
            if dL_swap_map is None:
                raise ValueError("swap_dL_events requested but no dL_swap_map was provided.")
            donor = str(dL_swap_map.get(str(ev), ""))
            if not donor:
                raise ValueError("swap_dL_events missing donor mapping.")
            pe_donor = pe_by_event.get(donor)
            if pe_donor is None:
                raise ValueError(f"swap_dL_events donor '{donor}' missing PE entry.")

            rng = np.random.default_rng(stable_int_seed(f"siren_isolator:{int(scramble_seed)}:swap_dL_events:{ev}"))
            n = int(pe0.dL_mpc.size)
            m = int(pe_donor.dL_mpc.size)
            if n <= 0 or m <= 0:
                raise ValueError("swap_dL_events requires non-empty PE arrays.")
            idx = rng.choice(m, size=n, replace=True)

            pe = GWTCPeHierarchicalSamples(
                file=str(pe0.file),
                analysis=str(pe0.analysis),
                n_total=int(pe0.n_total),
                n_used=int(pe0.n_used),
                dL_mpc=np.asarray(pe_donor.dL_mpc, dtype=float)[idx],
                chirp_mass_det=np.asarray(pe0.chirp_mass_det, dtype=float),
                mass_ratio=np.asarray(pe0.mass_ratio, dtype=float),
                log_pi_dL=np.asarray(pe_donor.log_pi_dL, dtype=float)[idx],
                log_pi_chirp_mass=np.asarray(pe0.log_pi_chirp_mass, dtype=float),
                log_pi_mass_ratio=np.asarray(pe0.log_pi_mass_ratio, dtype=float),
                prior_spec=dict(pe0.prior_spec),
            )
        else:
            pe = apply_hierarchical_pe_scramble(
                pe0,
                mode=mode,  # type: ignore[arg-type]
                seed=int(scramble_seed),
                tag=str(ev),
            )
        score, logL_mu, logL_gr = score_event_hierarchical_pe(
            event=str(ev),
            mode_label=str(mode_label),
            pe=pe,
            post=post,
            z_max=float(z_max),
            convention=str(convention),  # type: ignore[arg-type]
            pop_z_mode=str(pop_z_mode),  # type: ignore[arg-type]
            pop_z_k=float(pop_z_k),
            pop_mass_mode=str(pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
            pop_m_taper_delta=float(pop_m_taper_delta),
            pop_m_peak=float(pop_m_peak),
            pop_m_peak_sigma=float(pop_m_peak_sigma),
            pop_m_peak_frac=float(pop_m_peak_frac),
            log_alpha_mu=log_alpha_mu,
            log_alpha_gr=log_alpha_gr,
        )
    except Exception as e:
        return {"ok": False, "event": str(ev), "error": str(e)}

    return {"ok": True, "event": str(ev), "row": score.to_jsonable(), "logL_mu": logL_mu, "logL_gr": logL_gr}


def _is_finite_number(x: object) -> bool:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return False
    return bool(np.isfinite(v))


def _logmeanexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_logmeanexp_1d expects a 1D array.")
    if not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _build_derangement_map(items: list[str], *, seed: int, salt: str) -> dict[str, str]:
    items = [str(x) for x in items]
    n = len(items)
    if n <= 1:
        return {items[0]: items[0]} if n == 1 else {}

    rng = np.random.default_rng(stable_int_seed(f"derangement:{int(seed)}:{salt}"))
    donors = np.asarray(rng.permutation(items), dtype=object)
    # Roll through shifts until we find one with no fixed points (deterministic, finite).
    for _ in range(n):
        if all(str(donors[i]) != items[i] for i in range(n)):
            break
        donors = np.roll(donors, 1)
    return {items[i]: str(donors[i]) for i in range(n)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Spectral siren isolator battery (hierarchical PE + mechanism-nulling scrambles).")
    ap.add_argument("--run-dir", required=True, help="Finished EM reconstruction dir (contains samples/mu_forward_posterior.npz).")
    ap.add_argument(
        "--pe-cache-dir",
        default=None,
        help="Directory containing cached hierarchical PE .npz files (pe_hier_<EVENT>.npz). Preferred for speed.",
    )
    ap.add_argument(
        "--pe-base-dir",
        default=None,
        help="Base directory containing downloaded GWTC PEDataRelease files (Zenodo record subdirs). Used if --pe-cache-dir not provided.",
    )
    ap.add_argument("--pe-record-ids", default=None, help="Comma list of Zenodo record IDs to scan under --pe-base-dir (default: auto).")
    ap.add_argument("--pe-prefer-variants", default="combined", help="Comma list of PE variants to prefer (default: combined).")
    ap.add_argument("--pe-analysis", default=None, help="Optional analysis group label to force (else auto-select with priors).")
    ap.add_argument("--pe-max-samples", type=int, default=None, help="Max PE samples per event (default: keep all).")
    ap.add_argument("--pe-seed", type=int, default=0, help="Seed for PE downsampling selection (default 0).")

    ap.add_argument("--events", default=None, help="Comma list of event names. Default: all events available from cache/index.")
    ap.add_argument("--max-events", type=int, default=0, help="If >0, cap number of events (deterministic alphabetical).")
    ap.add_argument(
        "--from-dark-siren-summary",
        default=None,
        help=(
            "Optional run_dark_siren_gap_test summary JSON. If provided and --events/--z-max are not set, "
            "defaults to its events_scored and z_max. If it contains posterior_draw_idx, the mu posterior is "
            "downselected to match precomputed selection-alpha vectors."
        ),
    )
    ap.add_argument("--n-proc", type=int, default=0, help="Worker processes for event-parallel scoring (default: all cores).")

    ap.add_argument(
        "--scramble-modes",
        default="none,shuffle_dL,shuffle_mass,shuffle_dL_mass",
        help="Comma list of scramble modes. See isolator.md for interpretation.",
    )
    ap.add_argument("--scramble-seed", type=int, default=0, help="Seed controlling scrambles (default 0).")

    ap.add_argument("--z-max", type=float, default=None, help="Max redshift used for hierarchical inversion (default: posterior z_grid max).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="GW/EM distance-ratio convention (default A).")

    # Population knobs (must match your selection/pop assumptions if you combine with alpha).
    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="none", help="Population z weight mode (default none).")
    ap.add_argument("--pop-z-k", type=float, default=0.0, help="Powerlaw k for --pop-z-mode=comoving_powerlaw.")
    ap.add_argument(
        "--pop-mass-mode",
        choices=["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="none",
        help="Population mass weight mode (default none).",
    )
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3, help="m1 powerlaw slope alpha (default 2.3).")
    ap.add_argument("--pop-m-min", type=float, default=5.0, help="Min source-frame mass (default 5).")
    ap.add_argument("--pop-m-max", type=float, default=80.0, help="Max source-frame mass (default 80).")
    ap.add_argument("--pop-q-beta", type=float, default=0.0, help="q powerlaw exponent beta (default 0).")
    ap.add_argument("--pop-m-taper-delta", type=float, default=0.0, help="Smooth taper width (Msun) for --pop-mass-mode=powerlaw_q_smooth (default 0).")
    ap.add_argument("--pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for --pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for --pop-mass-mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for --pop-mass-mode=powerlaw_peak_q_smooth.")

    ap.add_argument(
        "--selection-alpha-npz",
        default=None,
        help="Optional NPZ containing log_alpha_mu/log_alpha_gr arrays (per draw). If provided, reports ΔLPD_sel and ΔLPD_total.",
    )

    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_isolator_<UTCSTAMP>).")
    args = ap.parse_args()

    post_full = load_mu_forward_posterior(args.run_dir)

    summary: dict[str, Any] | None = None
    summary_path: Path | None = None
    if args.from_dark_siren_summary is not None:
        summary_path = Path(args.from_dark_siren_summary).expanduser().resolve()
        summary = json.loads(summary_path.read_text())

    z_max = (
        float(args.z_max)
        if args.z_max is not None
        else float(summary.get("z_max"))
        if summary is not None and summary.get("z_max") is not None
        else float(post_full.z_grid[-1])
    )
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("Invalid z-max.")

    # Optional draw downselection (needed to align with selection-alpha vectors from a previous run).
    draw_idx: list[int] | None = None
    post = post_full
    if summary is not None and isinstance(summary.get("posterior_draw_idx"), list) and summary["posterior_draw_idx"]:
        draw_idx = [int(x) for x in summary["posterior_draw_idx"]]
        post = _select_posterior_draws(post_full, draw_idx)
    draw_idx_sha1 = hashlib.sha1(np.asarray(draw_idx if draw_idx is not None else [], dtype=np.int64).tobytes()).hexdigest()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_isolator_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    json_dir = out_dir / "json"
    cache_dir = out_dir / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Optional selection-alpha vectors (per draw).
    log_alpha_mu = None
    log_alpha_gr = None
    selection_alpha_sha1: str | None = None
    if args.selection_alpha_npz is not None:
        selection_alpha_sha1 = hashlib.sha1(Path(args.selection_alpha_npz).expanduser().read_bytes()).hexdigest()
        with np.load(str(args.selection_alpha_npz), allow_pickle=True) as d:
            log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
            log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)
        if log_alpha_mu.shape != (post.H_samples.shape[0],) or log_alpha_gr.shape != (post.H_samples.shape[0],):
            raise ValueError(
                "selection-alpha arrays must have shape (n_draws,) matching the (possibly downselected) mu posterior. "
                "Tip: pass --from-dark-siren-summary to use the same posterior_draw_idx used when alpha was computed."
            )

    # Parse modes.
    scramble_modes: list[str] = _parse_csv_list(args.scramble_modes)
    if not scramble_modes:
        raise ValueError("No scramble modes requested.")
    for m in scramble_modes:
        if m not in ("none", "shuffle_dL", "shuffle_mass", "shuffle_dL_mass", "shuffle_mc", "shuffle_q", "prior_dL", "swap_dL_events"):
            raise ValueError(f"Unknown scramble mode: {m}")

    # Load PE samples per event (prefer cache).
    pe_by_event: dict[str, GWTCPeHierarchicalSamples] = {}
    if args.pe_cache_dir is not None:
        pe_cache_dir = Path(args.pe_cache_dir)
        if not pe_cache_dir.exists():
            raise FileNotFoundError(f"--pe-cache-dir not found: {pe_cache_dir}")
        for p in sorted(pe_cache_dir.glob("pe_hier_*.npz")):
            ev = p.stem.replace("pe_hier_", "")
            try:
                ev2, pe = _load_pe_hier_cache_npz(p)
                if ev2:
                    ev = ev2
                pe_by_event[ev] = pe
            except Exception:
                continue
    else:
        if args.pe_base_dir is None:
            raise ValueError("Provide either --pe-cache-dir or --pe-base-dir.")
        pe_base_dir = Path(args.pe_base_dir).expanduser().resolve()
        pe_record_ids = [int(x) for x in _parse_csv_list(args.pe_record_ids)] if args.pe_record_ids is not None else None
        pe_prefer_variants = _parse_csv_list(args.pe_prefer_variants)
        if not pe_prefer_variants:
            pe_prefer_variants = ["combined"]
        pe_index = build_gwtc_pe_index(base_dir=pe_base_dir, record_ids=pe_record_ids or None)

        # Determine candidate events from the index (may be filtered by --events below).
        for ev, files in pe_index.items():
            # Order candidates by preferred variant, else fall back to index order.
            ordered: list[Any] = []
            seen: set[str] = set()
            for v in pe_prefer_variants:
                for f in files:
                    if getattr(f, "variant", None) == v:
                        p = str(getattr(f, "path"))
                        if p not in seen:
                            ordered.append(f)
                            seen.add(p)
            for f in files:
                p = str(getattr(f, "path"))
                if p not in seen:
                    ordered.append(f)
                    seen.add(p)

            pe_obj = None
            for cand in ordered:
                pe_try = (pe_base_dir / str(getattr(cand, "path"))).resolve()
                try:
                    pe_obj = load_gwtc_pe_hierarchical_samples(
                        path=pe_try,
                        analysis=str(args.pe_analysis) if args.pe_analysis else None,
                        max_samples=int(args.pe_max_samples) if args.pe_max_samples is not None else None,
                        seed=stable_int_seed(f"pe_hier:{int(args.pe_seed)}:{ev}"),
                    )
                    break
                except Exception:
                    pe_obj = None
                    continue
            if pe_obj is not None:
                pe_by_event[ev] = pe_obj

    if not pe_by_event:
        raise ValueError("No hierarchical PE events found (check cache/index inputs).")

    # Apply requested event filter.
    events = sorted(pe_by_event.keys())
    want = _parse_csv_list(args.events)
    if want:
        events = [e for e in events if e in set(want)]
    elif summary is not None and isinstance(summary.get("events_scored"), list) and summary["events_scored"]:
        keep = {str(x) for x in summary["events_scored"]}
        events = [e for e in events if e in keep]
    if int(args.max_events) > 0:
        events = events[: int(args.max_events)]
    if not events:
        raise ValueError("No events remain after filtering.")

    # Persist manifest for reproducibility.
    manifest = {
        "run_dir": str(Path(args.run_dir).resolve()),
        "n_draws_full": int(post_full.H_samples.shape[0]),
        "n_draws": int(post.H_samples.shape[0]),
        "from_dark_siren_summary": str(summary_path) if summary_path is not None else None,
        "posterior_draw_idx_sha1": str(draw_idx_sha1),
        "posterior_draw_idx_n": int(len(draw_idx) if draw_idx is not None else 0),
        "z_max": float(z_max),
        "events": list(events),
        "scramble_modes": list(scramble_modes),
        "scramble_seed": int(args.scramble_seed),
        "convention": str(args.convention),
        "pop_z_mode": str(args.pop_z_mode),
        "pop_z_k": float(args.pop_z_k),
        "pop_mass_mode": str(args.pop_mass_mode),
        "pop_m1_alpha": float(args.pop_m1_alpha),
        "pop_m_min": float(args.pop_m_min),
        "pop_m_max": float(args.pop_m_max),
        "pop_q_beta": float(args.pop_q_beta),
        "selection_alpha_npz": str(Path(args.selection_alpha_npz).resolve()) if args.selection_alpha_npz else None,
        "selection_alpha_sha1": str(selection_alpha_sha1) if selection_alpha_sha1 is not None else None,
    }
    (json_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    totals_by_mode: dict[str, dict[str, float]] = {}
    n_proc = int(args.n_proc) if int(args.n_proc) > 0 else int(os.cpu_count() or 1)

    # Main loop: modes x events (resumable per (mode,event) cache).
    for mode in scramble_modes:
        mode_label = str(mode)
        dL_swap_map: dict[str, str] | None = None
        if mode_label == "swap_dL_events":
            dL_swap_map = _build_derangement_map(events, seed=int(args.scramble_seed), salt="swap_dL_events")
        n_draws = int(post.H_samples.shape[0])
        sum_logL_mu = np.zeros((n_draws,), dtype=float)
        sum_logL_gr = np.zeros((n_draws,), dtype=float)
        totals = {"n_ok": 0.0, "n_skip": 0.0}
        to_run: list[str] = []
        meta_by_event: dict[str, dict[str, Any]] = {}

        # First: reuse cached per-event outputs when possible.
        for ev in events:
            cache_path = cache_dir / f"logL_{mode_label}_{ev}.npz"
            swap_donor = str(dL_swap_map.get(str(ev), "")) if dL_swap_map is not None else None
            meta_key = {
                "event": str(ev),
                "mode": str(mode_label),
                "swap_dL_donor_event": swap_donor,
                "z_max": float(z_max),
                "convention": str(args.convention),
                "pop_z_mode": str(args.pop_z_mode),
                "pop_z_k": float(args.pop_z_k),
                "pop_mass_mode": str(args.pop_mass_mode),
                "pop_m1_alpha": float(args.pop_m1_alpha),
                "pop_m_min": float(args.pop_m_min),
                "pop_m_max": float(args.pop_m_max),
                "pop_q_beta": float(args.pop_q_beta),
                "pop_m_taper_delta": float(args.pop_m_taper_delta),
                "pop_m_peak": float(args.pop_m_peak),
                "pop_m_peak_sigma": float(args.pop_m_peak_sigma),
                "pop_m_peak_frac": float(args.pop_m_peak_frac),
                "scramble_seed": int(args.scramble_seed),
                "selection_alpha": bool(args.selection_alpha_npz is not None),
                "selection_alpha_sha1": str(selection_alpha_sha1) if selection_alpha_sha1 is not None else None,
                "posterior_draw_idx_sha1": str(draw_idx_sha1),
            }
            meta_by_event[str(ev)] = meta_key

            if cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=True) as d:
                        meta_old = json.loads(str(d["meta"].tolist()))
                        if all(meta_old.get(k) == meta_key.get(k) for k in meta_key.keys()):
                            score_row = json.loads(str(d["row_json"].tolist()))
                            logL_mu = np.asarray(d["logL_mu"], dtype=float)
                            logL_gr = np.asarray(d["logL_gr"], dtype=float)
                            if logL_mu.shape == sum_logL_mu.shape and logL_gr.shape == sum_logL_gr.shape:
                                sum_logL_mu += logL_mu
                                sum_logL_gr += logL_gr
                            rows.append(score_row)
                            totals["n_ok"] += 1.0
                            continue
                except Exception:
                    pass

            to_run.append(str(ev))

        # Second: compute missing events in parallel (fork-only; keeps PE arrays unpickled).
        if to_run:
            global _WORKER_STATE
            _WORKER_STATE = {
                "mode_label": str(mode_label),
                "mode": str(mode),
                "post": post,
                "z_max": float(z_max),
                "convention": str(args.convention),
                "pop_z_mode": str(args.pop_z_mode),
                "pop_z_k": float(args.pop_z_k),
                "pop_mass_mode": str(args.pop_mass_mode),
                "pop_m1_alpha": float(args.pop_m1_alpha),
                "pop_m_min": float(args.pop_m_min),
                "pop_m_max": float(args.pop_m_max),
                "pop_q_beta": float(args.pop_q_beta),
                "pop_m_taper_delta": float(args.pop_m_taper_delta),
                "pop_m_peak": float(args.pop_m_peak),
                "pop_m_peak_sigma": float(args.pop_m_peak_sigma),
                "pop_m_peak_frac": float(args.pop_m_peak_frac),
                "scramble_seed": int(args.scramble_seed),
                "pe_by_event": pe_by_event,
                "dL_swap_map": dL_swap_map,
                "log_alpha_mu": log_alpha_mu,
                "log_alpha_gr": log_alpha_gr,
            }

            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=int(n_proc), mp_context=ctx) as ex:
                futs = {ex.submit(_score_one_event_worker, ev): ev for ev in to_run}
                done = 0
                for fut in as_completed(futs):
                    ev = futs[fut]
                    done += 1
                    try:
                        res = fut.result()
                    except Exception as e:
                        skipped.append({"mode": str(mode_label), "event": str(ev), "error": str(e)})
                        totals["n_skip"] += 1.0
                        continue

                    if not bool(res.get("ok")):
                        skipped.append({"mode": str(mode_label), "event": str(ev), "error": str(res.get("error", "unknown"))})
                        totals["n_skip"] += 1.0
                        continue

                    row = dict(res["row"])
                    # Guard against degenerate population support (LPD=-inf -> delta=NaN).
                    if not _is_finite_number(row.get("delta_lpd_data")):
                        skipped.append({"mode": str(mode_label), "event": str(ev), "error": "non-finite delta_lpd_data"})
                        totals["n_skip"] += 1.0
                        continue
                    if not _is_finite_number(row.get("delta_lpd")):
                        skipped.append({"mode": str(mode_label), "event": str(ev), "error": "non-finite delta_lpd"})
                        totals["n_skip"] += 1.0
                        continue
                    logL_mu = np.asarray(res["logL_mu"], dtype=float)
                    logL_gr = np.asarray(res["logL_gr"], dtype=float)
                    if logL_mu.shape == sum_logL_mu.shape and logL_gr.shape == sum_logL_gr.shape:
                        sum_logL_mu += logL_mu
                        sum_logL_gr += logL_gr
                    else:
                        skipped.append({"mode": str(mode_label), "event": str(ev), "error": "logL shape mismatch"})
                        totals["n_skip"] += 1.0
                        continue
                    rows.append(row)
                    totals["n_ok"] += 1.0

                    cache_path = cache_dir / f"logL_{mode_label}_{ev}.npz"
                    meta_key = meta_by_event[str(ev)]
                    np.savez(
                        cache_path,
                        meta=json.dumps(meta_key, sort_keys=True),
                        row_json=json.dumps(row, sort_keys=True),
                        logL_mu=np.asarray(logL_mu, dtype=np.float64),
                        logL_gr=np.asarray(logL_gr, dtype=np.float64),
                    )

                    if done % 5 == 0 or done == len(to_run):
                        print(f"[siren_isolator] mode={mode_label} progress {done}/{len(to_run)}", flush=True)

        # Dataset-level totals: match run_dark_siren_gap_test hierarchical aggregation.
        n_ev = int(totals.get("n_ok", 0.0))
        if n_ev <= 0:
            totals_by_mode[mode_label] = {
                **totals,
                "lpd_mu_total_data": float("-inf"),
                "lpd_gr_total_data": float("-inf"),
                "delta_lpd_total_data": float("nan"),
                "lpd_mu_total": float("-inf"),
                "lpd_gr_total": float("-inf"),
                "delta_lpd_total": float("nan"),
                "delta_lpd_total_sel": float("nan"),
            }
        else:
            lpd_mu_total_data = float(_logmeanexp_1d(sum_logL_mu))
            lpd_gr_total_data = float(_logmeanexp_1d(sum_logL_gr))
            delta_lpd_total_data = float(lpd_mu_total_data - lpd_gr_total_data) if np.isfinite(lpd_mu_total_data) and np.isfinite(lpd_gr_total_data) else float("nan")

            if log_alpha_mu is not None and log_alpha_gr is not None:
                logL_mu_total = sum_logL_mu - float(n_ev) * np.asarray(log_alpha_mu, dtype=float)
                logL_gr_total = sum_logL_gr - float(n_ev) * np.asarray(log_alpha_gr, dtype=float)
                lpd_mu_total = float(_logmeanexp_1d(logL_mu_total))
                lpd_gr_total = float(_logmeanexp_1d(logL_gr_total))
            else:
                lpd_mu_total = float(lpd_mu_total_data)
                lpd_gr_total = float(lpd_gr_total_data)

            delta_lpd_total = float(lpd_mu_total - lpd_gr_total) if np.isfinite(lpd_mu_total) and np.isfinite(lpd_gr_total) else float("nan")
            delta_lpd_total_sel = float(delta_lpd_total - delta_lpd_total_data) if np.isfinite(delta_lpd_total) and np.isfinite(delta_lpd_total_data) else float("nan")

            totals_by_mode[mode_label] = {
                **totals,
                "lpd_mu_total_data": float(lpd_mu_total_data),
                "lpd_gr_total_data": float(lpd_gr_total_data),
                "delta_lpd_total_data": float(delta_lpd_total_data),
                "lpd_mu_total": float(lpd_mu_total),
                "lpd_gr_total": float(lpd_gr_total),
                "delta_lpd_total": float(delta_lpd_total),
                "delta_lpd_total_sel": float(delta_lpd_total_sel),
            }

    _write_csv(tab_dir / "event_scores.csv", rows)
    _write_csv(tab_dir / "skipped.csv", skipped)
    (json_dir / "summary.json").write_text(json.dumps({"totals_by_mode": totals_by_mode}, indent=2, sort_keys=True) + "\n")

    # Figures.
    if rows:
        # Totals.
        modes = list(totals_by_mode.keys())
        data = [totals_by_mode[m].get("delta_lpd_total_data", float("nan")) for m in modes]
        sel = [totals_by_mode[m].get("delta_lpd_total_sel", float("nan")) for m in modes]
        tot = [totals_by_mode[m].get("delta_lpd_total", float("nan")) for m in modes]

        fig, ax = plt.subplots(figsize=(8.5, 4.0))
        x = np.arange(len(modes))
        ax.bar(x, data, label="ΔLPD_data")
        ax.bar(x, sel, bottom=data, label="ΔLPD_sel")
        ax.plot(x, tot, "ko", label="ΔLPD_total")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=20, ha="right")
        ax.set_ylabel("ΔLPD (sum over events)")
        ax.set_title("Spectral siren isolator totals")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / "totals.png", dpi=160)
        plt.close(fig)

        # Per-event deltas (data term).
        rows0 = [r for r in rows if r.get("mode") == "none"]
        if rows0:
            rows0 = sorted(rows0, key=lambda r: float(r.get("delta_lpd_data", 0.0)))
            evs = [str(r.get("event")) for r in rows0]
            vals = [float(r.get("delta_lpd_data", float("nan"))) for r in rows0]
            fig, ax = plt.subplots(figsize=(10.5, 6.0))
            ax.barh(np.arange(len(evs)), vals)
            ax.axvline(0.0, color="k", lw=0.8)
            ax.set_yticks(np.arange(len(evs)))
            ax.set_yticklabels(evs, fontsize=7)
            ax.set_xlabel("ΔLPD_data per event (none)")
            ax.set_title("Per-event spectral preference (data term, unscrambled)")
            fig.tight_layout()
            fig.savefig(fig_dir / "delta_lpd_data_by_event_none.png", dpi=160)
            plt.close(fig)

    print(f"[siren_isolator] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

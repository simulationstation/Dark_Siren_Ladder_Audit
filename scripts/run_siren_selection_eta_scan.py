from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_sirens_selection import compute_selection_alpha_from_injections, load_o3_injections
from entropy_horizon_recon.sirens import MuForwardPosterior, load_mu_forward_posterior


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


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


def _select_posterior_draws(post: MuForwardPosterior, draw_idx: list[int]) -> MuForwardPosterior:
    idx = np.asarray(draw_idx, dtype=int)
    if idx.ndim != 1 or idx.size == 0:
        raise ValueError("posterior_draw_idx must be a non-empty 1D list.")
    n = int(post.H_samples.shape[0])
    if np.any(idx < 0) or np.any(idx >= n):
        raise ValueError(f"posterior_draw_idx out of range for posterior with n_draws={n}.")
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


def _logmeanexp_1d(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_logmeanexp_1d expects a 1D array.")
    if not np.any(np.isfinite(x)):
        return float("-inf")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _load_isolator_sum_logL(
    *,
    isolator_cache_dir: Path,
    mode: str,
    events: list[str],
    n_draws: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    sum_mu = np.zeros((n_draws,), dtype=float)
    sum_gr = np.zeros((n_draws,), dtype=float)
    n_ok = 0
    for ev in events:
        p = isolator_cache_dir / f"logL_{mode}_{ev}.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=True) as d:
            logL_mu = np.asarray(d["logL_mu"], dtype=float)
            logL_gr = np.asarray(d["logL_gr"], dtype=float)
        if logL_mu.shape != (n_draws,) or logL_gr.shape != (n_draws,):
            continue
        sum_mu += logL_mu
        sum_gr += logL_gr
        n_ok += 1
    if n_ok <= 0:
        raise ValueError("No isolator logL files loaded; check cache dir/mode/events.")
    return sum_mu, sum_gr, int(n_ok)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-3 selection-slope scan: vary an SNR-offset nuisance eta and track ΔLPD_sel / ΔLPD_tot.")
    ap.add_argument("--run-dir", required=True, help="Finished EM reconstruction dir (contains samples/mu_forward_posterior.npz).")
    ap.add_argument(
        "--from-dark-siren-summary",
        default=None,
        help="Optional run_dark_siren_gap_test summary JSON; uses its posterior_draw_idx and events_scored if present.",
    )

    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")
    ap.add_argument("--selection-z-max", type=float, default=None, help="Selection z_max (default: posterior z_grid max).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="GW/EM convention for mu draw mapping (default A).")
    ap.add_argument("--det-model", choices=["threshold", "snr_binned"], default="snr_binned", help="Detection model (default snr_binned).")
    ap.add_argument("--snr-thresh", type=float, default=None, help="Optional fixed SNR threshold (default: calibrate).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for snr_binned model (default 200).")
    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="none", help="Injection weighting mode (default none).")
    ap.add_argument(
        "--mu-det-distance",
        choices=["gw", "em"],
        default="gw",
        help="Distance used to evaluate detectability for mu model: gw (default, uses dL_GW) or em (diagnostic ablation).",
    )

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="none", help="Population z weight mode (default none).")
    ap.add_argument("--pop-z-k", type=float, default=0.0, help="Powerlaw k for pop_z_mode=comoving_powerlaw.")
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
    ap.add_argument("--pop-m-taper-delta", type=float, default=0.0, help="Smooth taper width (Msun) for pop_mass_mode=powerlaw_q_smooth (default 0).")
    ap.add_argument("--pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for pop_mass_mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for pop_mass_mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for pop_mass_mode=powerlaw_peak_q_smooth.")

    ap.add_argument("--eta-values", default="-2,-1,-0.5,0,0.5,1,2", help="Comma list of SNR offsets eta to scan (default -2..2).")
    ap.add_argument("--eta-seed", type=int, default=0, help="Seed (reserved; deterministic scan uses fixed grid).")

    ap.add_argument(
        "--isolator-cache-dir",
        default=None,
        help="Optional siren_isolator cache dir to compute exact ΔLPD_tot(eta) using stored per-event logL vectors.",
    )
    ap.add_argument("--isolator-mode", default="none", help="Isolator mode label to load (default none).")
    ap.add_argument(
        "--events",
        default=None,
        help="Comma list of events; overrides summary events_scored. Needed if --isolator-cache-dir is set but no summary is provided.",
    )

    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_selection_eta_scan_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_selection_eta_scan_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    json_dir = out_dir / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    post_full = load_mu_forward_posterior(args.run_dir)
    summary: dict[str, Any] | None = None
    summary_path: Path | None = None
    if args.from_dark_siren_summary is not None:
        summary_path = Path(args.from_dark_siren_summary).expanduser().resolve()
        summary = json.loads(summary_path.read_text())

    draw_idx: list[int] | None = None
    if summary is not None and isinstance(summary.get("posterior_draw_idx"), list) and summary["posterior_draw_idx"]:
        draw_idx = [int(x) for x in summary["posterior_draw_idx"]]
    post = _select_posterior_draws(post_full, draw_idx) if draw_idx is not None else post_full
    draw_idx_sha1 = hashlib.sha1(np.asarray(draw_idx if draw_idx is not None else [], dtype=np.int64).tobytes()).hexdigest()

    # Event list and (optional) exact data-term stacks.
    events: list[str] = []
    if args.events is not None:
        events = [e.strip() for e in str(args.events).split(",") if e.strip()]
    elif summary is not None and isinstance(summary.get("events_scored"), list) and summary["events_scored"]:
        events = [str(x) for x in summary["events_scored"]]

    sum_logL_mu: np.ndarray | None = None
    sum_logL_gr: np.ndarray | None = None
    n_events_data = 0
    if args.isolator_cache_dir is not None:
        if not events:
            raise ValueError("--isolator-cache-dir requires an event list (via --from-dark-siren-summary or --events).")
        cache_dir = Path(args.isolator_cache_dir).expanduser().resolve()
        sum_logL_mu, sum_logL_gr, n_events_data = _load_isolator_sum_logL(
            isolator_cache_dir=cache_dir,
            mode=str(args.isolator_mode),
            events=events,
            n_draws=int(post.H_samples.shape[0]),
        )

    z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(post.z_grid[-1])
    if not (np.isfinite(z_sel) and z_sel > 0.0):
        raise ValueError("Invalid selection z_max.")

    inj_path = Path(args.selection_injections_hdf).expanduser().resolve()
    injections = load_o3_injections(inj_path, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))

    # Parse eta grid.
    eta_vals = [float(x.strip()) for x in str(args.eta_values).split(",") if x.strip()]
    if not eta_vals:
        raise ValueError("No eta values provided.")
    eta = np.asarray(eta_vals, dtype=float)
    if not np.all(np.isfinite(eta)):
        raise ValueError("eta_values must be finite.")

    # Meta for reproducibility.
    meta = {
        "timestamp_utc": _utc_stamp(),
        "run_dir": str(Path(args.run_dir).resolve()),
        "from_dark_siren_summary": str(summary_path) if summary_path is not None else None,
        "posterior_draw_idx_sha1": str(draw_idx_sha1),
        "posterior_draw_idx_n": int(len(draw_idx) if draw_idx is not None else 0),
        "events_n": int(len(events)) if events else None,
        "isolator_cache_dir": str(Path(args.isolator_cache_dir).resolve()) if args.isolator_cache_dir is not None else None,
        "isolator_mode": str(args.isolator_mode),
        "n_events_data_loaded": int(n_events_data) if sum_logL_mu is not None else None,
        "selection_injections_hdf": str(inj_path),
        "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
        "selection_z_max": float(z_sel),
        "convention": str(args.convention),
        "det_model": str(args.det_model),
        "snr_threshold": float(args.snr_thresh) if args.snr_thresh is not None else None,
        "snr_binned_nbins": int(args.snr_binned_nbins),
        "weight_mode": str(args.weight_mode),
        "mu_det_distance": str(args.mu_det_distance),
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
        "eta_values": [float(x) for x in eta.tolist()],
    }
    (json_dir / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    # Loop over eta: compute alpha arrays and derive ΔLPD components.
    rows: list[dict[str, Any]] = []
    alpha_mu_grid: list[np.ndarray] = []
    alpha_gr_grid: list[np.ndarray] = []
    log_alpha_mu_grid: list[np.ndarray] = []
    log_alpha_gr_grid: list[np.ndarray] = []

    for e in eta.tolist():
        alpha = compute_selection_alpha_from_injections(
            post=post,
            injections=injections,
            convention=str(args.convention),  # type: ignore[arg-type]
            z_max=float(z_sel),
            snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
            det_model=str(args.det_model),  # type: ignore[arg-type]
            snr_offset=float(e),
            snr_binned_nbins=int(args.snr_binned_nbins),
            weight_mode=str(args.weight_mode),  # type: ignore[arg-type]
            mu_det_distance=str(args.mu_det_distance),  # type: ignore[arg-type]
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
        )

        a_mu = np.asarray(alpha.alpha_mu, dtype=float)
        a_gr = np.asarray(alpha.alpha_gr, dtype=float)
        la_mu = np.log(np.clip(a_mu, 1e-300, np.inf))
        la_gr = np.log(np.clip(a_gr, 1e-300, np.inf))

        alpha_mu_grid.append(a_mu)
        alpha_gr_grid.append(a_gr)
        log_alpha_mu_grid.append(la_mu)
        log_alpha_gr_grid.append(la_gr)

        # Approximate scalar alpha(model) by mean over draws.
        log_mean_alpha_mu = float(np.log(np.mean(a_mu)))
        log_mean_alpha_gr = float(np.log(np.mean(a_gr)))

        row: dict[str, Any] = {
            "eta": float(e),
            "log_mean_alpha_mu": float(log_mean_alpha_mu),
            "log_mean_alpha_gr": float(log_mean_alpha_gr),
            "log_mean_alpha_mu_minus_gr": float(log_mean_alpha_mu - log_mean_alpha_gr),
        }

        # If we have data-term stacks, compute exact ΔLPD_tot(eta).
        if sum_logL_mu is not None and sum_logL_gr is not None:
            N = int(n_events_data)
            lpd_mu_data = float(_logmeanexp_1d(sum_logL_mu))
            lpd_gr_data = float(_logmeanexp_1d(sum_logL_gr))
            delta_data = float(lpd_mu_data - lpd_gr_data)
            lpd_mu_tot = float(_logmeanexp_1d(sum_logL_mu - float(N) * la_mu))
            lpd_gr_tot = float(_logmeanexp_1d(sum_logL_gr - float(N) * la_gr))
            delta_tot = float(lpd_mu_tot - lpd_gr_tot)
            row.update(
                {
                    "n_events": int(N),
                    "delta_lpd_total_data": float(delta_data),
                    "delta_lpd_total": float(delta_tot),
                    "delta_lpd_total_sel": float(delta_tot - delta_data),
                    "lpd_mu_total": float(lpd_mu_tot),
                    "lpd_gr_total": float(lpd_gr_tot),
                }
            )
        else:
            # Crude selection-only approximation (if no logL stacks given).
            if events:
                N = len(events)
                row["n_events"] = int(N)
                row["delta_lpd_sel_approx"] = float(-float(N) * (log_mean_alpha_mu - log_mean_alpha_gr))
        rows.append(row)

        print(f"[eta_scan] eta={e:+.3f} log(mean α_mu/α_gr)={row['log_mean_alpha_mu_minus_gr']:+.4f}", flush=True)

    # Save arrays.
    np.savez(
        tab_dir / "alpha_eta_scan.npz",
        eta=np.asarray(eta, dtype=float),
        alpha_mu=np.stack(alpha_mu_grid, axis=0).astype(np.float64),
        alpha_gr=np.stack(alpha_gr_grid, axis=0).astype(np.float64),
        log_alpha_mu=np.stack(log_alpha_mu_grid, axis=0).astype(np.float64),
        log_alpha_gr=np.stack(log_alpha_gr_grid, axis=0).astype(np.float64),
        meta=json.dumps(meta, sort_keys=True),
    )

    _write_csv(tab_dir / "eta_scan.csv", rows)
    (json_dir / "summary.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=2, sort_keys=True) + "\n")

    # Figures.
    et = np.asarray([r["eta"] for r in rows], dtype=float)
    if "delta_lpd_total" in rows[0]:
        dtot = np.asarray([float(r.get("delta_lpd_total", float("nan"))) for r in rows], dtype=float)
        dsel = np.asarray([float(r.get("delta_lpd_total_sel", float("nan"))) for r in rows], dtype=float)
        ddata = float(rows[0].get("delta_lpd_total_data", float("nan")))
        plt.figure(figsize=(7.5, 4.2))
        plt.plot(et, dtot, "o-", label=r"$\Delta\mathrm{LPD}_{\rm tot}(\eta)$")
        plt.plot(et, dsel, "o--", label=r"$\Delta\mathrm{LPD}_{\rm sel}(\eta)$")
        plt.axhline(ddata, color="C2", lw=1.2, label=r"$\Delta\mathrm{LPD}_{\rm data}$ (const)")
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel(r"$\eta$ (SNR offset)")
        plt.ylabel(r"$\Delta\mathrm{LPD}$")
        plt.title("Selection-slope scan (Gate 3)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "delta_lpd_vs_eta.png", dpi=160)
        plt.close()
    else:
        ds = np.asarray([float(r.get("delta_lpd_sel_approx", float("nan"))) for r in rows], dtype=float)
        plt.figure(figsize=(7.5, 4.2))
        plt.plot(et, ds, "o-")
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel(r"$\eta$ (SNR offset)")
        plt.ylabel(r"$\Delta\mathrm{LPD}_{\rm sel}$ approx")
        plt.title("Selection-slope scan (approx)")
        plt.tight_layout()
        plt.savefig(fig_dir / "delta_lpd_sel_approx_vs_eta.png", dpi=160)
        plt.close()

    print(f"[eta_scan] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

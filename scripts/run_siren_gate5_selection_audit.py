from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AuditConfig:
    name: str
    det_model: str
    snr_binned_nbins: int
    weight_mode: str
    mu_det_distance: str
    pop_z_mode: str = "none"
    pop_z_k: float = 0.0
    pop_mass_mode: str = "none"
    pop_m1_alpha: float = 2.3
    pop_m_min: float = 5.0
    pop_m_max: float = 80.0
    pop_q_beta: float = 0.0
    pop_m_taper_delta: float = 0.0
    pop_m_peak: float = 35.0
    pop_m_peak_sigma: float = 5.0
    pop_m_peak_frac: float = 0.1


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate 5 audit: recompute alpha(draw) under multiple detectability/weighting variants and quantify ΔLPD sensitivity.")
    ap.add_argument("--run-dir", required=True, help="Finished EM reconstruction dir (contains samples/mu_forward_posterior.npz).")
    ap.add_argument(
        "--from-dark-siren-summary",
        required=True,
        help="run_dark_siren_gap_test summary JSON; used to select events_scored + posterior_draw_idx.",
    )
    ap.add_argument(
        "--isolator-cache-dir",
        required=True,
        help="siren_isolator cache dir containing per-event logL vectors (logL_<mode>_<event>.npz).",
    )
    ap.add_argument("--isolator-mode", default="none", help="Isolator mode label to load (default none).")

    ap.add_argument("--selection-injections-hdf", required=True, help="Path to O3 sensitivity injection file (HDF5).")
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")
    ap.add_argument("--selection-z-max", type=float, default=None, help="Selection z_max (default: posterior z_grid max).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="GW/EM convention for mu draw mapping (default A).")

    ap.add_argument(
        "--config-set",
        choices=["baseline", "extended", "masspeak"],
        default="baseline",
        help="Which built-in selection audit config set to run (default baseline).",
    )
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate5_selection_audit_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate5_selection_audit_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    json_dir = out_dir / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(args.from_dark_siren_summary).expanduser().resolve()
    summary = json.loads(summary_path.read_text())
    events = [str(x) for x in summary.get("events_scored", [])]
    draw_idx = [int(x) for x in summary.get("posterior_draw_idx", [])]
    if not events:
        raise ValueError("summary JSON missing events_scored.")
    if not draw_idx:
        raise ValueError("summary JSON missing posterior_draw_idx.")

    post_full = load_mu_forward_posterior(args.run_dir)
    post = _select_posterior_draws(post_full, draw_idx)
    n_draws = int(post.H_samples.shape[0])
    draw_idx_sha1 = hashlib.sha1(np.asarray(draw_idx, dtype=np.int64).tobytes()).hexdigest()

    # Load sum logL vectors for the requested isolator mode.
    cache_dir = Path(args.isolator_cache_dir).expanduser().resolve()
    sum_logL_mu, sum_logL_gr, n_events = _load_isolator_sum_logL(
        isolator_cache_dir=cache_dir,
        mode=str(args.isolator_mode),
        events=events,
        n_draws=n_draws,
    )

    lpd_mu_data = float(_logmeanexp_1d(sum_logL_mu))
    lpd_gr_data = float(_logmeanexp_1d(sum_logL_gr))
    delta_data = float(lpd_mu_data - lpd_gr_data)

    z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(post.z_grid[-1])
    if not (np.isfinite(z_sel) and z_sel > 0.0):
        raise ValueError("Invalid selection z_max.")

    inj_path = Path(args.selection_injections_hdf).expanduser().resolve()
    injections = load_o3_injections(inj_path, ifar_threshold_yr=float(args.selection_ifar_thresh_yr))

    cfgs: list[AuditConfig]
    if str(args.config_set) == "baseline":
        cfgs = [
            AuditConfig(name="binned_gw", det_model="snr_binned", snr_binned_nbins=200, weight_mode="none", mu_det_distance="gw"),
            AuditConfig(name="binned_em", det_model="snr_binned", snr_binned_nbins=200, weight_mode="none", mu_det_distance="em"),
            AuditConfig(name="thr_gw", det_model="threshold", snr_binned_nbins=200, weight_mode="none", mu_det_distance="gw"),
            AuditConfig(name="thr_em", det_model="threshold", snr_binned_nbins=200, weight_mode="none", mu_det_distance="em"),
            AuditConfig(name="thr_invpdf_gw", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw"),
            AuditConfig(name="thr_invpdf_em", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em"),
            AuditConfig(name="binned_invpdf_gw", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw"),
            AuditConfig(name="binned_invpdf_em", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em"),
        ]
    elif str(args.config_set) == "extended":
        cfgs = [
            AuditConfig(name="binned_gw", det_model="snr_binned", snr_binned_nbins=200, weight_mode="none", mu_det_distance="gw"),
            AuditConfig(name="binned_em", det_model="snr_binned", snr_binned_nbins=200, weight_mode="none", mu_det_distance="em"),
            AuditConfig(name="thr_gw", det_model="threshold", snr_binned_nbins=200, weight_mode="none", mu_det_distance="gw"),
            AuditConfig(name="thr_em", det_model="threshold", snr_binned_nbins=200, weight_mode="none", mu_det_distance="em"),
            AuditConfig(name="thr_invpdf_gw", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw"),
            AuditConfig(name="thr_invpdf_em", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em"),
            AuditConfig(name="binned_invpdf_gw", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw"),
            AuditConfig(name="binned_invpdf_em", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em"),
            AuditConfig(
                name="binned_invpdf_popz_gw",
                det_model="snr_binned",
                snr_binned_nbins=200,
                weight_mode="inv_sampling_pdf",
                mu_det_distance="gw",
                pop_z_mode="comoving_uniform",
            ),
            AuditConfig(
                name="binned_invpdf_popz_em",
                det_model="snr_binned",
                snr_binned_nbins=200,
                weight_mode="inv_sampling_pdf",
                mu_det_distance="em",
                pop_z_mode="comoving_uniform",
            ),
            AuditConfig(
                name="thr_invpdf_popz_gw",
                det_model="threshold",
                snr_binned_nbins=200,
                weight_mode="inv_sampling_pdf",
                mu_det_distance="gw",
                pop_z_mode="comoving_uniform",
            ),
            AuditConfig(
                name="thr_invpdf_popz_em",
                det_model="threshold",
                snr_binned_nbins=200,
                weight_mode="inv_sampling_pdf",
                mu_det_distance="em",
                pop_z_mode="comoving_uniform",
            ),
        ]
    elif str(args.config_set) == "masspeak":
        # A self-consistent population+selection configuration intended to match Gate-2.
        # NOTE: This config set assumes the isolator logL cache was generated with the same pop_z/mass settings.
        mass_cfg = dict(
            pop_z_mode="comoving_uniform",
            pop_mass_mode="powerlaw_peak_q_smooth",
            pop_m1_alpha=2.3,
            pop_m_min=5.0,
            pop_m_max=80.0,
            pop_q_beta=0.0,
            pop_m_taper_delta=3.0,
            pop_m_peak=35.0,
            pop_m_peak_sigma=5.0,
            pop_m_peak_frac=0.1,
        )
        cfgs = [
            AuditConfig(name="thr_invpdf_gw_masspeak", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw", **mass_cfg),
            AuditConfig(name="thr_invpdf_em_masspeak", det_model="threshold", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em", **mass_cfg),
            AuditConfig(name="binned_invpdf_gw_masspeak", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="gw", **mass_cfg),
            AuditConfig(name="binned_invpdf_em_masspeak", det_model="snr_binned", snr_binned_nbins=200, weight_mode="inv_sampling_pdf", mu_det_distance="em", **mass_cfg),
        ]
    else:  # pragma: no cover
        raise ValueError("Unknown config_set.")

    meta = {
        "timestamp_utc": _utc_stamp(),
        "run_dir": str(Path(args.run_dir).resolve()),
        "from_dark_siren_summary": str(summary_path),
        "posterior_draw_idx_sha1": str(draw_idx_sha1),
        "posterior_draw_idx_n": int(len(draw_idx)),
        "isolator_cache_dir": str(cache_dir),
        "isolator_mode": str(args.isolator_mode),
        "selection_injections_hdf": str(inj_path),
        "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
        "selection_z_max": float(z_sel),
        "convention": str(args.convention),
        "n_events_loaded": int(n_events),
        "delta_lpd_total_data": float(delta_data),
        "lpd_mu_total_data": float(lpd_mu_data),
        "lpd_gr_total_data": float(lpd_gr_data),
        "config_set": str(args.config_set),
    }
    (json_dir / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

    rows: list[dict[str, Any]] = []
    for cfg in cfgs:
        row: dict[str, Any] = {
            "name": str(cfg.name),
            "det_model": str(cfg.det_model),
            "snr_binned_nbins": int(cfg.snr_binned_nbins),
            "weight_mode": str(cfg.weight_mode),
            "mu_det_distance": str(cfg.mu_det_distance),
            "pop_z_mode": str(cfg.pop_z_mode),
            "pop_z_k": float(cfg.pop_z_k),
            "pop_mass_mode": str(cfg.pop_mass_mode),
            "pop_m1_alpha": float(cfg.pop_m1_alpha),
            "pop_m_min": float(cfg.pop_m_min),
            "pop_m_max": float(cfg.pop_m_max),
            "pop_q_beta": float(cfg.pop_q_beta),
            "pop_m_taper_delta": float(cfg.pop_m_taper_delta),
            "pop_m_peak": float(cfg.pop_m_peak),
            "pop_m_peak_sigma": float(cfg.pop_m_peak_sigma),
            "pop_m_peak_frac": float(cfg.pop_m_peak_frac),
            "n_events": int(n_events),
            "n_draws": int(n_draws),
        }

        try:
            alpha = compute_selection_alpha_from_injections(
                post=post,
                injections=injections,
                convention=str(args.convention),  # type: ignore[arg-type]
                z_max=float(z_sel),
                det_model=str(cfg.det_model),  # type: ignore[arg-type]
                snr_binned_nbins=int(cfg.snr_binned_nbins),
                weight_mode=str(cfg.weight_mode),  # type: ignore[arg-type]
                mu_det_distance=str(cfg.mu_det_distance),  # type: ignore[arg-type]
                pop_z_mode=str(cfg.pop_z_mode),  # type: ignore[arg-type]
                pop_z_powerlaw_k=float(cfg.pop_z_k),
                pop_mass_mode=str(cfg.pop_mass_mode),  # type: ignore[arg-type]
                pop_m1_alpha=float(cfg.pop_m1_alpha),
                pop_m_min=float(cfg.pop_m_min),
                pop_m_max=float(cfg.pop_m_max),
                pop_q_beta=float(cfg.pop_q_beta),
                pop_m_taper_delta=float(cfg.pop_m_taper_delta),
                pop_m_peak=float(cfg.pop_m_peak),
                pop_m_peak_sigma=float(cfg.pop_m_peak_sigma),
                pop_m_peak_frac=float(cfg.pop_m_peak_frac),
            )

            a_mu = np.asarray(alpha.alpha_mu, dtype=float)
            a_gr = np.asarray(alpha.alpha_gr, dtype=float)
            la_mu = np.log(np.clip(a_mu, 1e-300, np.inf))
            la_gr = np.log(np.clip(a_gr, 1e-300, np.inf))

            logL_mu_tot_draw = sum_logL_mu - float(n_events) * la_mu
            logL_gr_tot_draw = sum_logL_gr - float(n_events) * la_gr
            delta_lpd_data_draw = sum_logL_mu - sum_logL_gr
            delta_lpd_sel_draw = -float(n_events) * (la_mu - la_gr)
            delta_lpd_tot_draw = logL_mu_tot_draw - logL_gr_tot_draw

            lpd_mu_tot = float(_logmeanexp_1d(logL_mu_tot_draw))
            lpd_gr_tot = float(_logmeanexp_1d(logL_gr_tot_draw))
            delta_tot = float(lpd_mu_tot - lpd_gr_tot)
            delta_sel = float(delta_tot - delta_data)

            row.update(
                {
                    "log_mean_alpha_mu_minus_gr": float(np.log(np.mean(a_mu)) - np.log(np.mean(a_gr))),
                    "delta_lpd_sel_draw_p50": float(np.nanmedian(delta_lpd_sel_draw)),
                    "delta_lpd_sel_draw_mean": float(np.nanmean(delta_lpd_sel_draw)),
                    "delta_lpd_sel_draw_sd": float(np.nanstd(delta_lpd_sel_draw)),
                    "delta_lpd_tot_draw_p50": float(np.nanmedian(delta_lpd_tot_draw)),
                    "delta_lpd_tot_draw_mean": float(np.nanmean(delta_lpd_tot_draw)),
                    "delta_lpd_total": float(delta_tot),
                    "delta_lpd_total_sel": float(delta_sel),
                    "delta_lpd_total_data": float(delta_data),
                    "lpd_mu_total": float(lpd_mu_tot),
                    "lpd_gr_total": float(lpd_gr_tot),
                    "alpha_n_injections_used": int(alpha.n_injections_used),
                }
            )

            np.savez(
                tab_dir / f"selection_alpha_{cfg.name}.npz",
                alpha_mu=a_mu.astype(np.float64),
                alpha_gr=a_gr.astype(np.float64),
                log_alpha_mu=la_mu.astype(np.float64),
                log_alpha_gr=la_gr.astype(np.float64),
                meta=json.dumps({**meta, **row}, sort_keys=True),
            )

            np.savez(
                tab_dir / f"delta_lpd_draws_{cfg.name}.npz",
                # Per-draw additive decomposition (exact for a fixed draw j):
                #   ΔLPD_tot(j) = ΔLPD_data(j) + ΔLPD_sel(j),
                # where ΔLPD_sel(j) = -N [log α_mu(j) - log α_gr(j)].
                delta_lpd_tot_draw=np.asarray(delta_lpd_tot_draw, dtype=np.float64),
                delta_lpd_data_draw=np.asarray(delta_lpd_data_draw, dtype=np.float64),
                delta_lpd_sel_draw=np.asarray(delta_lpd_sel_draw, dtype=np.float64),
                logL_mu_tot_draw=np.asarray(logL_mu_tot_draw, dtype=np.float64),
                logL_gr_tot_draw=np.asarray(logL_gr_tot_draw, dtype=np.float64),
                log_alpha_mu=np.asarray(la_mu, dtype=np.float64),
                log_alpha_gr=np.asarray(la_gr, dtype=np.float64),
                meta=json.dumps({**meta, **row}, sort_keys=True),
            )
        except Exception as e:
            row["error"] = str(e)

        print(f"[gate5_audit] {row.get('name')} done", flush=True)
        rows.append(row)

    _write_csv(tab_dir / "selection_audit.csv", rows)
    (json_dir / "summary.json").write_text(json.dumps({"rows": rows, "meta": meta}, indent=2, sort_keys=True) + "\n")

    # Figure: ΔLPD_total by config.
    ok_rows = [r for r in rows if "delta_lpd_total" in r and np.isfinite(float(r["delta_lpd_total"]))]
    if ok_rows:
        names = [str(r["name"]) for r in ok_rows]
        dtot = np.asarray([float(r["delta_lpd_total"]) for r in ok_rows], dtype=float)
        dsel = np.asarray([float(r.get("delta_lpd_total_sel", float("nan"))) for r in ok_rows], dtype=float)
        x = np.arange(len(names))
        plt.figure(figsize=(10.0, 4.2))
        plt.bar(x, dtot, color="C0", alpha=0.85, label=r"$\Delta\mathrm{LPD}_{\rm tot}$")
        plt.plot(x, dsel, "ko", ms=3.5, label=r"$\Delta\mathrm{LPD}_{\rm sel}$ (implied)")
        plt.axhline(float(delta_data), color="C2", lw=1.0, label=r"$\Delta\mathrm{LPD}_{\rm data}$ (fixed)")
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel(r"$\Delta\mathrm{LPD}$")
        plt.title("Gate 5: selection-model audit (alpha variants)")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "delta_lpd_by_selection_config.png", dpi=160)
        plt.close()

    print(f"[gate5_audit] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

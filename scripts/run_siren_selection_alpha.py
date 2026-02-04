from __future__ import annotations

import argparse
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute selection normalization alpha(draw) for mu vs GR from injections (spectral siren selection term).")
    ap.add_argument("--run-dir", required=True, help="Finished EM reconstruction dir (contains samples/mu_forward_posterior.npz).")
    ap.add_argument(
        "--from-dark-siren-summary",
        default=None,
        help="Optional run_dark_siren_gap_test summary JSON; if it contains posterior_draw_idx, downselect to match.",
    )

    ap.add_argument(
        "--selection-injections-hdf",
        required=True,
        help="Path to O3 sensitivity injection file (HDF5).",
    )
    ap.add_argument("--selection-ifar-thresh-yr", type=float, default=1.0, help="IFAR threshold (years) for injections (default 1).")
    ap.add_argument("--selection-z-max", type=float, default=None, help="Selection z_max (default: posterior z_grid max).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="GW/EM convention for mu draw mapping (default A).")
    ap.add_argument(
        "--det-model",
        choices=["threshold", "snr_binned", "snr_mchirp_binned", "snr_mchirp_q_binned"],
        default="snr_binned",
        help="Detection model (default snr_binned).",
    )
    ap.add_argument("--snr-thresh", type=float, default=None, help="Optional fixed SNR threshold (default: calibrate).")
    ap.add_argument("--snr-offset", type=float, default=0.0, help="Shift effective SNR before p_det evaluation (eta; default 0).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for snr_binned model (default 200).")
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20, help="Chirp-mass bins for det_model=snr_mchirp_binned (default 20).")
    ap.add_argument("--q-binned-nbins", type=int, default=10, help="Mass-ratio bins for det_model=snr_mchirp_q_binned (default 10).")
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

    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_selection_alpha_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_selection_alpha_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    post_full = load_mu_forward_posterior(args.run_dir)
    summary_path: Path | None = None
    summary: dict[str, Any] | None = None
    draw_idx: list[int] | None = None
    if args.from_dark_siren_summary is not None:
        summary_path = Path(args.from_dark_siren_summary).expanduser().resolve()
        summary = json.loads(summary_path.read_text())
        if isinstance(summary.get("posterior_draw_idx"), list) and summary["posterior_draw_idx"]:
            draw_idx = [int(x) for x in summary["posterior_draw_idx"]]

    post = _select_posterior_draws(post_full, draw_idx) if draw_idx is not None else post_full
    draw_idx_sha1 = hashlib.sha1(np.asarray(draw_idx if draw_idx is not None else [], dtype=np.int64).tobytes()).hexdigest()

    z_sel = float(args.selection_z_max) if args.selection_z_max is not None else float(post.z_grid[-1])
    if not (np.isfinite(z_sel) and z_sel > 0.0):
        raise ValueError("Invalid selection z_max.")

    inj_path = Path(args.selection_injections_hdf).expanduser().resolve()
    injections = load_o3_injections(
        inj_path,
        ifar_threshold_yr=float(args.selection_ifar_thresh_yr),
    )

    alpha = compute_selection_alpha_from_injections(
        post=post,
        injections=injections,
        convention=str(args.convention),  # type: ignore[arg-type]
        z_max=float(z_sel),
        snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
        det_model=str(args.det_model),  # type: ignore[arg-type]
        snr_offset=float(args.snr_offset),
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
        q_binned_nbins=int(args.q_binned_nbins),
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

    log_alpha_mu = np.log(np.clip(np.asarray(alpha.alpha_mu, dtype=float), 1e-300, np.inf))
    log_alpha_gr = np.log(np.clip(np.asarray(alpha.alpha_gr, dtype=float), 1e-300, np.inf))

    meta = {
        "timestamp_utc": _utc_stamp(),
        "run_dir": str(Path(args.run_dir).resolve()),
        "from_dark_siren_summary": str(summary_path) if summary_path is not None else None,
        "posterior_draw_idx_sha1": str(draw_idx_sha1),
        "posterior_draw_idx_n": int(len(draw_idx) if draw_idx is not None else 0),
        "selection_injections_hdf": str(inj_path),
        "selection_ifar_thresh_yr": float(args.selection_ifar_thresh_yr),
        "selection_z_max": float(z_sel),
        "convention": str(args.convention),
        "det_model": str(args.det_model),
        "snr_threshold": float(args.snr_thresh) if args.snr_thresh is not None else None,
        "snr_offset": float(args.snr_offset),
        "snr_binned_nbins": int(args.snr_binned_nbins),
        "mchirp_binned_nbins": int(args.mchirp_binned_nbins),
        "q_binned_nbins": int(args.q_binned_nbins),
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
        "n_injections_used": int(alpha.n_injections_used),
        "method": str(alpha.method),
    }

    out_npz = tab_dir / "selection_alpha.npz"
    np.savez(
        out_npz,
        alpha_mu=np.asarray(alpha.alpha_mu, dtype=float),
        alpha_gr=np.asarray(alpha.alpha_gr, dtype=float),
        log_alpha_mu=np.asarray(log_alpha_mu, dtype=float),
        log_alpha_gr=np.asarray(log_alpha_gr, dtype=float),
        meta=json.dumps(meta, sort_keys=True),
    )

    # Quick diagnostic plots.
    try:
        dq = log_alpha_mu - log_alpha_gr
        qs = [5, 16, 50, 84, 95]
        qv = {f"p{q}": float(np.nanpercentile(dq, q)) for q in qs}
        (tab_dir / "selection_alpha_summary.json").write_text(json.dumps({"log_alpha_mu_minus_gr": qv, **meta}, indent=2, sort_keys=True) + "\n")

        plt.figure(figsize=(7, 4))
        plt.hist(dq[np.isfinite(dq)], bins=40, density=True, color="C0", alpha=0.8)
        plt.axvline(0.0, color="k", lw=0.8)
        plt.xlabel(r"$\log\\alpha_{\\mu}-\\log\\alpha_{\\rm GR}$")
        plt.ylabel("density")
        plt.title("Selection alpha ratio across posterior draws")
        plt.tight_layout()
        plt.savefig(fig_dir / "log_alpha_mu_minus_gr_hist.png", dpi=160)
        plt.close()
    except Exception:
        pass

    print(f"[siren_selection_alpha] wrote {out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

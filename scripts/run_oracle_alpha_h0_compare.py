from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.dark_siren_h0 import LCDMDistanceCache
from entropy_horizon_recon.dark_siren_h0 import _alpha_h0_grid_from_injections  # noqa: SLF001
from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache  # noqa: SLF001
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _build_wcosmo_distance_cache(*, z_max: float, omega_m0: float, omega_k0: float, tcmb0: float) -> LCDMDistanceCache:
    if float(omega_k0) != 0.0:
        raise ValueError("wcosmo cache only supports omega_k0=0 (FlatLambdaCDM).")

    from wcosmo.astropy import FlatLambdaCDM
    from wcosmo.utils import disable_units

    disable_units()
    z_grid = np.linspace(0.0, float(z_max), 5001)

    h0_ref = 70.0
    c_km_s = 299792.458
    cosmo = FlatLambdaCDM(H0=float(h0_ref), Om0=float(omega_m0), Tcmb0=float(tcmb0))
    dL = np.asarray(cosmo.luminosity_distance(z_grid), dtype=float)
    f_grid = dL * float(h0_ref) / float(c_km_s)
    return LCDMDistanceCache(omega_m0=float(omega_m0), omega_k0=float(omega_k0), z_grid=z_grid, f_grid=f_grid)


def _write_curve(path: Path, *, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _compare_on_same_grid(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Curves must have identical shapes for same-grid compare.")
    diff = a - b
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "l1_diff": float(np.sum(np.abs(diff))),
        "rms_diff": float(np.sqrt(np.mean(diff**2))),
    }


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(description="Compute and compare alpha(H0) curves using our cache, wcosmo, and ICAROGW.")
    ap.add_argument("--injection-file", required=True, help="O3 sensitivity injection file (LIGO-T2100113 style .hdf5).")
    ap.add_argument("--ifar-threshold-yr", type=float, default=1.0, help="Found threshold in iFAR years (default 1.0).")
    ap.add_argument("--h0-min", type=float, default=40.0, help="Min H0 (default 40).")
    ap.add_argument("--h0-max", type=float, default=120.0, help="Max H0 (default 120).")
    ap.add_argument("--h0-n", type=int, default=161, help="Number of H0 grid points (default 161).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 (default 0).")
    ap.add_argument("--tcmb0", type=float, default=0.0, help="CMB temperature for oracle cosmologies (default 0 to match no-radiation LCDM).")
    ap.add_argument("--z-max", type=float, default=2.3, help="Max redshift for injections/selection (default 2.3).")

    ap.add_argument(
        "--det-model",
        choices=["threshold", "snr_binned", "snr_mchirp_binned", "snr_mchirp_q_binned"],
        default="snr_binned",
        help="Detection model for alpha(H0) (default snr_binned).",
    )
    ap.add_argument("--snr-thresh", type=float, default=None, help="Optional fixed SNR threshold (default: calibrate).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for snr_binned model (default 200).")
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20, help="Chirp-mass bins for det_model=snr_mchirp_binned (default 20).")
    ap.add_argument("--q-binned-nbins", type=int, default=10, help="Mass-ratio bins for det_model=snr_mchirp_q_binned (default 10).")

    ap.add_argument("--weight-mode", choices=["none", "inv_sampling_pdf"], default="none", help="Injection weight mode (default none).")
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
    ap.add_argument("--pop-m-max", type=float, default=200.0, help="Max source-frame mass (default 200).")
    ap.add_argument("--pop-q-beta", type=float, default=0.0, help="q powerlaw exponent beta (default 0).")
    ap.add_argument("--pop-m-taper-delta", type=float, default=0.0, help="Smooth taper width (Msun) for pop_mass_mode=powerlaw_q_smooth.")
    ap.add_argument("--pop-m-peak", type=float, default=35.0, help="Gaussian peak location in m1 (Msun) for pop_mass_mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0, help="Gaussian peak sigma in m1 (Msun) for pop_mass_mode=powerlaw_peak_q_smooth.")
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1, help="Gaussian peak mixture fraction for pop_mass_mode=powerlaw_peak_q_smooth.")

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
        help="Distance/redshift coordinate used by injection sampling_pdf (default z).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-frame",
        choices=["source", "detector"],
        default="source",
        help="Mass-frame used by injection sampling_pdf (default source).",
    )
    ap.add_argument(
        "--inj-sampling-pdf-mass-scale",
        choices=["linear", "log"],
        default="linear",
        help="Mass coordinate scale used by injection sampling_pdf (default linear).",
    )

    ap.add_argument("--icarogw-python", default="oracles/icarogw/.venv/bin/python", help="Path to ICAROGW venv python.")
    ap.add_argument("--no-wcosmo", action="store_true", help="Skip the wcosmo oracle.")
    ap.add_argument("--no-icarogw", action="store_true", help="Skip the ICAROGW oracle.")
    ap.add_argument("--plot", action="store_true", help="Write figures/alpha_h0_compare.png.")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/oracle_alpha_h0_compare_<UTCSTAMP>).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"oracle_alpha_h0_compare_{_utc_stamp()}"
    json_dir = out_dir / "json"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.plot):
        fig_dir.mkdir(parents=True, exist_ok=True)

    injections = load_o3_injections(Path(args.injection_file).expanduser().resolve(), ifar_threshold_yr=float(args.ifar_threshold_yr))
    z_max = float(args.z_max)
    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))

    # 1) Our native distance cache.
    dist_ours = _build_lcdm_distance_cache(z_max=z_max, omega_m0=float(args.omega_m0), omega_k0=float(args.omega_k0))
    alpha_ours, meta_ours = _alpha_h0_grid_from_injections(
        injections=injections,
        H0_grid=H0_grid,
        dist_cache=dist_ours,
        z_max=z_max,
        det_model=str(args.det_model),
        snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
        q_binned_nbins=int(args.q_binned_nbins),
        weight_mode=str(args.weight_mode),
        pop_z_mode=str(args.pop_z_mode),
        pop_z_powerlaw_k=float(args.pop_z_k),
        pop_mass_mode=str(args.pop_mass_mode),
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=float(args.pop_m_taper_delta),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
        inj_sampling_pdf_dist=str(args.inj_sampling_pdf_dist),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_frame=str(args.inj_sampling_pdf_mass_frame),  # type: ignore[arg-type]
        inj_sampling_pdf_mass_scale=str(args.inj_sampling_pdf_mass_scale),  # type: ignore[arg-type]
    )
    ours_path = json_dir / "alpha_ours.json"
    _write_curve(
        ours_path,
        payload={
            "oracle": "ours",
            "distance_backend": "entropy_horizon_recon.LCDMDistanceCache",
            "injection_file": str(Path(args.injection_file).expanduser().resolve()),
            "ifar_threshold_yr": float(args.ifar_threshold_yr),
            "omega_m0": float(args.omega_m0),
            "omega_k0": float(args.omega_k0),
            "H0_grid": [float(x) for x in H0_grid.tolist()],
            "selection_alpha_grid": [float(x) for x in np.asarray(alpha_ours, dtype=float).tolist()],
            "selection_alpha": meta_ours,
        },
    )

    out_summary: dict[str, Any] = {
        "out_dir": str(out_dir.resolve()),
        "alpha_ours": str(ours_path.resolve()),
        "inj_sampling_pdf": {
            "inj_mass_pdf_coords": str(args.inj_mass_pdf_coords),
            "inj_sampling_pdf_dist": str(args.inj_sampling_pdf_dist),
            "inj_sampling_pdf_mass_frame": str(args.inj_sampling_pdf_mass_frame),
            "inj_sampling_pdf_mass_scale": str(args.inj_sampling_pdf_mass_scale),
        },
        "comparisons": {},
    }

    # 2) wcosmo oracle.
    alpha_wcosmo = None
    if not bool(args.no_wcosmo):
        try:
            dist_wcosmo = _build_wcosmo_distance_cache(
                z_max=z_max, omega_m0=float(args.omega_m0), omega_k0=float(args.omega_k0), tcmb0=float(args.tcmb0)
            )
        except Exception as e:
            dist_wcosmo = None
            out_summary["wcosmo_error"] = str(e)

        if dist_wcosmo is not None:
            alpha_wcosmo, meta_wcosmo = _alpha_h0_grid_from_injections(
                injections=injections,
                H0_grid=H0_grid,
                dist_cache=dist_wcosmo,
                z_max=z_max,
                det_model=str(args.det_model),
                snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
                snr_binned_nbins=int(args.snr_binned_nbins),
                mchirp_binned_nbins=int(args.mchirp_binned_nbins),
                q_binned_nbins=int(args.q_binned_nbins),
                weight_mode=str(args.weight_mode),
                pop_z_mode=str(args.pop_z_mode),
                pop_z_powerlaw_k=float(args.pop_z_k),
                pop_mass_mode=str(args.pop_mass_mode),
                pop_m1_alpha=float(args.pop_m1_alpha),
                pop_m_min=float(args.pop_m_min),
                pop_m_max=float(args.pop_m_max),
                pop_q_beta=float(args.pop_q_beta),
                pop_m_taper_delta=float(args.pop_m_taper_delta),
                pop_m_peak=float(args.pop_m_peak),
                pop_m_peak_sigma=float(args.pop_m_peak_sigma),
                pop_m_peak_frac=float(args.pop_m_peak_frac),
                inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
                inj_sampling_pdf_dist=str(args.inj_sampling_pdf_dist),  # type: ignore[arg-type]
                inj_sampling_pdf_mass_frame=str(args.inj_sampling_pdf_mass_frame),  # type: ignore[arg-type]
                inj_sampling_pdf_mass_scale=str(args.inj_sampling_pdf_mass_scale),  # type: ignore[arg-type]
            )
            wcosmo_path = json_dir / "alpha_wcosmo.json"
            _write_curve(
                wcosmo_path,
                payload={
                    "oracle": "wcosmo",
                    "distance_backend": f"wcosmo.astropy.FlatLambdaCDM(Tcmb0={float(args.tcmb0)})",
                    "injection_file": str(Path(args.injection_file).expanduser().resolve()),
                    "ifar_threshold_yr": float(args.ifar_threshold_yr),
                    "omega_m0": float(args.omega_m0),
                    "omega_k0": float(args.omega_k0),
                    "H0_grid": [float(x) for x in H0_grid.tolist()],
                    "selection_alpha_grid": [float(x) for x in np.asarray(alpha_wcosmo, dtype=float).tolist()],
                    "selection_alpha": meta_wcosmo,
                },
            )
            out_summary["alpha_wcosmo"] = str(wcosmo_path.resolve())
            out_summary["comparisons"]["ours_vs_wcosmo"] = _compare_on_same_grid(np.asarray(alpha_ours), np.asarray(alpha_wcosmo))

    # 3) ICAROGW oracle (distance cache computed via its isolated venv).
    alpha_icarogw = None
    if not bool(args.no_icarogw):
        # NOTE: do not `.resolve()` here: venv interpreters are often symlinks to the system
        # python, and resolving would bypass the venv's `pyvenv.cfg` and site-packages.
        icarogw_py = Path(str(args.icarogw_python)).expanduser()
        if not icarogw_py.is_absolute():
            icarogw_py = Path.cwd() / icarogw_py
        icarogw_py = icarogw_py.absolute()
        if not icarogw_py.exists():
            out_summary["icarogw_error"] = f"Missing ICAROGW python at {icarogw_py}"
        else:
            cache_path = json_dir / "icarogw_f_cache.json"
            cmd = [
                str(icarogw_py),
                "scripts/oracle_distance_cache_icarogw.py",
                "--z-max",
                str(z_max),
                "--omega-m0",
                str(float(args.omega_m0)),
                "--omega-k0",
                str(float(args.omega_k0)),
                "--tcmb0",
                str(float(args.tcmb0)),
                "--out-json",
                str(cache_path),
            ]
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)  # noqa: S603
            if proc.returncode != 0:
                out_summary["icarogw_error"] = f"ICAROGW cache build failed (rc={proc.returncode}). stderr:\n{proc.stderr}"
            else:
                d = json.loads(cache_path.read_text())
                z_grid = np.asarray(d["z_grid"], dtype=float)
                f_grid = np.asarray(d["f_grid"], dtype=float)
                dist_icarogw = LCDMDistanceCache(
                    omega_m0=float(args.omega_m0),
                    omega_k0=float(args.omega_k0),
                    z_grid=z_grid,
                    f_grid=f_grid,
                )
                alpha_icarogw, meta_icarogw = _alpha_h0_grid_from_injections(
                    injections=injections,
                    H0_grid=H0_grid,
                    dist_cache=dist_icarogw,
                    z_max=z_max,
                    det_model=str(args.det_model),
                    snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
                    snr_binned_nbins=int(args.snr_binned_nbins),
                    mchirp_binned_nbins=int(args.mchirp_binned_nbins),
                    q_binned_nbins=int(args.q_binned_nbins),
                    weight_mode=str(args.weight_mode),
                    pop_z_mode=str(args.pop_z_mode),
                    pop_z_powerlaw_k=float(args.pop_z_k),
                    pop_mass_mode=str(args.pop_mass_mode),
                    pop_m1_alpha=float(args.pop_m1_alpha),
                    pop_m_min=float(args.pop_m_min),
                    pop_m_max=float(args.pop_m_max),
                    pop_q_beta=float(args.pop_q_beta),
                    pop_m_taper_delta=float(args.pop_m_taper_delta),
                    pop_m_peak=float(args.pop_m_peak),
                    pop_m_peak_sigma=float(args.pop_m_peak_sigma),
                    pop_m_peak_frac=float(args.pop_m_peak_frac),
                    inj_mass_pdf_coords=str(args.inj_mass_pdf_coords),  # type: ignore[arg-type]
                    inj_sampling_pdf_dist=str(args.inj_sampling_pdf_dist),  # type: ignore[arg-type]
                    inj_sampling_pdf_mass_frame=str(args.inj_sampling_pdf_mass_frame),  # type: ignore[arg-type]
                    inj_sampling_pdf_mass_scale=str(args.inj_sampling_pdf_mass_scale),  # type: ignore[arg-type]
                )
                icarogw_path = json_dir / "alpha_icarogw.json"
                _write_curve(
                    icarogw_path,
                    payload={
                        "oracle": "icarogw",
                        "distance_backend": d.get("distance_backend", "icarogw"),
                        "injection_file": str(Path(args.injection_file).expanduser().resolve()),
                        "ifar_threshold_yr": float(args.ifar_threshold_yr),
                        "omega_m0": float(args.omega_m0),
                        "omega_k0": float(args.omega_k0),
                        "H0_grid": [float(x) for x in H0_grid.tolist()],
                        "selection_alpha_grid": [float(x) for x in np.asarray(alpha_icarogw, dtype=float).tolist()],
                        "selection_alpha": meta_icarogw,
                        "icarogw_cache": str(cache_path.resolve()),
                        "icarogw_meta": {k: v for k, v in d.items() if k not in {"z_grid", "f_grid"}},
                    },
                )
                out_summary["alpha_icarogw"] = str(icarogw_path.resolve())
                out_summary["comparisons"]["ours_vs_icarogw"] = _compare_on_same_grid(np.asarray(alpha_ours), np.asarray(alpha_icarogw))

    comp_path = json_dir / "alpha_compare_summary.json"
    comp_path.write_text(json.dumps(out_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out_summary, indent=2, sort_keys=True))

    if bool(args.plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"matplotlib required for --plot: {e}")

        plt.figure(figsize=(7.5, 4.2))
        plt.plot(H0_grid, np.asarray(alpha_ours, dtype=float), "-", lw=1.8, label="ours")
        if alpha_wcosmo is not None:
            plt.plot(H0_grid, np.asarray(alpha_wcosmo, dtype=float), "--", lw=1.4, label="wcosmo")
        if alpha_icarogw is not None:
            plt.plot(H0_grid, np.asarray(alpha_icarogw, dtype=float), "-.", lw=1.4, label="icarogw")
        plt.xlabel("H0 [km/s/Mpc]")
        plt.ylabel("alpha(H0)")
        plt.title("Selection alpha(H0) oracle comparison")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(str((fig_dir / "alpha_h0_compare.png").resolve()), dpi=160)
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

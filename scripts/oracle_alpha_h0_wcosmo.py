from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.dark_siren_h0 import LCDMDistanceCache
from entropy_horizon_recon.dark_siren_h0 import _alpha_h0_grid_from_injections  # noqa: SLF001
from entropy_horizon_recon.dark_sirens_selection import load_o3_injections


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _build_wcosmo_distance_cache(*, z_max: float, omega_m0: float, omega_k0: float) -> LCDMDistanceCache:
    if float(omega_k0) != 0.0:
        raise ValueError("wcosmo oracle only supports omega_k0=0 (FlatLambdaCDM).")

    from wcosmo.astropy import FlatLambdaCDM
    from wcosmo.utils import disable_units

    disable_units()
    z_grid = np.linspace(0.0, float(z_max), 5001)

    # Choose an arbitrary reference H0. For flat LCDM with fixed (Om0), dL scales as 1/H0, so
    # f(z) = dL(z;H0_ref)*H0_ref/c is independent of H0_ref.
    h0_ref = 70.0
    c_km_s = 299792.458
    cosmo = FlatLambdaCDM(H0=float(h0_ref), Om0=float(omega_m0), Tcmb0=0.0)
    dL = np.asarray(cosmo.luminosity_distance(z_grid), dtype=float)
    f_grid = dL * float(h0_ref) / float(c_km_s)

    if not np.all(np.isfinite(f_grid)):
        raise ValueError("Non-finite f_grid from wcosmo.")

    return LCDMDistanceCache(
        omega_m0=float(omega_m0),
        omega_k0=float(omega_k0),
        z_grid=np.asarray(z_grid, dtype=float),
        f_grid=np.asarray(f_grid, dtype=float),
    )


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(description="Oracle alpha(H0) from injections using wcosmo distances.")
    ap.add_argument("--injection-file", required=True, help="O3 sensitivity injection file (LIGO-T2100113 style .hdf5).")
    ap.add_argument("--ifar-threshold-yr", type=float, default=1.0, help="Found threshold in iFAR years (default 1.0).")
    ap.add_argument("--h0-min", type=float, default=40.0, help="Min H0 (default 40).")
    ap.add_argument("--h0-max", type=float, default=120.0, help="Max H0 (default 120).")
    ap.add_argument("--h0-n", type=int, default=161, help="Number of H0 grid points (default 161).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 (default 0; required).")
    ap.add_argument("--z-max", type=float, default=2.3, help="Max redshift for injections/selection (default 2.3).")

    ap.add_argument(
        "--det-model",
        choices=["threshold", "snr_binned", "snr_mchirp_binned"],
        default="snr_binned",
        help="Detection model for alpha(H0) (default snr_binned).",
    )
    ap.add_argument("--snr-thresh", type=float, default=None, help="Optional fixed SNR threshold (default: calibrate).")
    ap.add_argument("--snr-binned-nbins", type=int, default=200, help="Bins for snr_binned model (default 200).")
    ap.add_argument("--mchirp-binned-nbins", type=int, default=20, help="Chirp-mass bins for det_model=snr_mchirp_binned (default 20).")

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

    ap.add_argument("--out", default=None, help="Output directory (default outputs/oracle_alpha_h0_wcosmo_<UTCSTAMP>).")
    ap.add_argument("--plot", action="store_true", help="Write figures/alpha_h0.png.")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"oracle_alpha_h0_wcosmo_{_utc_stamp()}"
    json_dir = out_dir / "json"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.plot):
        fig_dir.mkdir(parents=True, exist_ok=True)

    injections = load_o3_injections(Path(args.injection_file).expanduser().resolve(), ifar_threshold_yr=float(args.ifar_threshold_yr))

    z_max = float(args.z_max)
    dist_cache = _build_wcosmo_distance_cache(z_max=z_max, omega_m0=float(args.omega_m0), omega_k0=float(args.omega_k0))
    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))

    alpha, meta = _alpha_h0_grid_from_injections(
        injections=injections,
        H0_grid=H0_grid,
        dist_cache=dist_cache,
        z_max=z_max,
        det_model=str(args.det_model),
        snr_threshold=float(args.snr_thresh) if args.snr_thresh is not None else None,
        snr_binned_nbins=int(args.snr_binned_nbins),
        mchirp_binned_nbins=int(args.mchirp_binned_nbins),
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
    )

    try:
        import gwpopulation
    except Exception:  # pragma: no cover
        gwpopulation = None  # type: ignore[assignment]
    try:
        import wcosmo
    except Exception:  # pragma: no cover
        wcosmo = None  # type: ignore[assignment]

    out: dict[str, Any] = {
        "oracle": "wcosmo",
        "oracle_versions": {
            "gwpopulation": getattr(gwpopulation, "__version__", None),
            "wcosmo": getattr(wcosmo, "__version__", None),
        },
        "distance_backend": "wcosmo.astropy.FlatLambdaCDM(Tcmb0=0)",
        "injection_file": str(Path(args.injection_file).expanduser().resolve()),
        "ifar_threshold_yr": float(args.ifar_threshold_yr),
        "omega_m0": float(args.omega_m0),
        "omega_k0": float(args.omega_k0),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "selection_alpha_grid": [float(x) for x in np.asarray(alpha, dtype=float).tolist()],
        "selection_alpha": meta,
    }

    out_path = json_dir / "oracle_alpha_h0.json"
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_json": str(out_path)}, indent=2, sort_keys=True))

    if bool(args.plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"matplotlib required for --plot: {e}")

        plt.figure(figsize=(7.5, 4.2))
        plt.plot(H0_grid, np.asarray(alpha, dtype=float), "-", lw=1.6)
        plt.xlabel("H0 [km/s/Mpc]")
        plt.ylabel("alpha(H0)")
        plt.title("Oracle alpha(H0) (wcosmo)")
        plt.tight_layout()
        plt.savefig(str((fig_dir / "alpha_h0.png").resolve()), dpi=160)
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

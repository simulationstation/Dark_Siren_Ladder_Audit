from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")  # headless/cluster-safe
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.sirens import load_mu_forward_posterior
from entropy_horizon_recon.voids import load_void_amp_measurements, predict_void_kappa_amp_from_mu


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _detect_mapping_variant(run_dir: Path) -> str:
    """Best-effort detection of mapping variant from the run's tables/summary.json."""
    summary = run_dir / "tables" / "summary.json"
    if summary.exists():
        try:
            d = json.loads(summary.read_text())
            mv = (d.get("settings") or {}).get("mapping_variant")
            if mv:
                return str(mv)
        except Exception:
            pass
    # Fallback heuristic (common naming convention).
    name = run_dir.name
    if "M2" in name:
        return "M2"
    if "M1" in name:
        return "M1"
    return "M0"


def _logpdf_normal(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sig <= 0:
        raise ValueError("sig must be positive.")
    return -np.log(sig) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((x - mu) / sig) ** 2


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


@dataclass(frozen=True)
class AmpResultRow:
    run: str
    mapping_variant: str
    measurement: str
    kind: str
    convention: str
    weight: str
    z_min: float
    z_max: float
    z_n: int
    A_obs: float
    A_sigma: float
    A_pred_mean: float
    A_pred_std: float
    A_pred_q05: float
    A_pred_q16: float
    A_pred_q50: float
    A_pred_q84: float
    A_pred_q95: float
    lpd_mu: float
    lpd_gr: float
    delta_lpd: float


def main() -> int:
    ap = argparse.ArgumentParser(description="Tier-1 void lensing amplitude holdout test.")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--void-data", required=True, help="JSON with amplitude-only void lensing measurements.")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/void_amp_test_<UTCSTAMP>).")
    ap.add_argument(
        "--convention",
        action="append",
        choices=["A", "B"],
        default=None,
        help="mu->R convention (repeatable; default: A).",
    )
    ap.add_argument(
        "--weight",
        choices=["uniform", "z", "z2", "cmb_kappa"],
        default=None,
        help="Override measurement weight mode. If omitted, uses measurement.weight or a kernel default.",
    )
    ap.add_argument(
        "--chi-source-mpc",
        type=float,
        default=None,
        help="Override source-plane comoving distance for kernel weights (e.g. ~14000 for CMB).",
    )
    ap.add_argument("--z-n", type=int, default=200, help="Number of z points used for averaging.")
    ap.add_argument("--allow-extrapolation", action="store_true", help="Allow z-range beyond inferred domain (not recommended).")
    ap.add_argument(
        "--extend-h-to-zmax",
        action="store_true",
        help="If a measurement z_max exceeds posterior z_grid[-1], re-solve H(z) per draw from mu(A) out to z_max (slower, avoids z extrapolation).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"void_amp_test_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    meas = load_void_amp_measurements(args.void_data)

    rows: list[dict[str, Any]] = []
    conventions = args.convention if args.convention else ["A"]

    for rd in args.run_dir:
        run_path = Path(rd)
        post = load_mu_forward_posterior(run_path)
        run_label = run_path.name
        mapping_variant = _detect_mapping_variant(run_path)

        for m in meas:
            for conv in conventions:
                weight = args.weight or m.weight or ("cmb_kappa" if (m.kernel or "").lower() == "cmb_kappa" else "uniform")
                chi_source_mpc = args.chi_source_mpc if args.chi_source_mpc is not None else (m.chi_source_mpc or 14_000.0)

                try:
                    _, A_draws = predict_void_kappa_amp_from_mu(
                        post,
                        z_min=m.z_min,
                        z_max=m.z_max,
                        z_n=int(args.z_n),
                        convention=conv,  # type: ignore[arg-type]
                        weight=weight,  # type: ignore[arg-type]
                        chi_source_mpc=float(chi_source_mpc),
                        nz=m.nz,
                        allow_extrapolation=bool(args.allow_extrapolation),
                        extend_H_to_zmax=bool(args.extend_h_to_zmax),
                        mapping_variant=str(mapping_variant),
                    )
                except Exception as e:
                    rows.append(
                        {
                            "run": run_label,
                            "mapping_variant": str(mapping_variant),
                            "measurement": m.name,
                            "kind": m.kind,
                            "convention": conv,
                            "weight": weight,
                            "status": "skipped",
                            "error": str(e),
                        }
                    )
                    continue

                qs = np.percentile(A_draws, [5, 16, 50, 84, 95])
                A_mean = float(np.mean(A_draws))
                A_std = float(np.std(A_draws, ddof=1)) if A_draws.size > 1 else float("nan")

                # Posterior predictive score against the amplitude observation.
                logp_mu = _logpdf_normal(A_draws, mu=m.A_obs, sig=m.A_sigma)
                lpd_mu = float(_logmeanexp(logp_mu))
                lpd_gr = float(_logpdf_normal(np.array([1.0]), mu=m.A_obs, sig=m.A_sigma)[0])

                row = AmpResultRow(
                    run=run_label,
                    mapping_variant=str(mapping_variant),
                    measurement=m.name,
                    kind=m.kind,
                    convention=str(conv),
                    weight=str(weight),
                    z_min=float(m.z_min),
                    z_max=float(m.z_max),
                    z_n=int(args.z_n),
                    A_obs=float(m.A_obs),
                    A_sigma=float(m.A_sigma),
                    A_pred_mean=A_mean,
                    A_pred_std=A_std,
                    A_pred_q05=float(qs[0]),
                    A_pred_q16=float(qs[1]),
                    A_pred_q50=float(qs[2]),
                    A_pred_q84=float(qs[3]),
                    A_pred_q95=float(qs[4]),
                    lpd_mu=lpd_mu,
                    lpd_gr=lpd_gr,
                    delta_lpd=lpd_mu - lpd_gr,
                )
                rows.append(
                    {
                        **asdict(row),
                        "status": "ok",
                        "notes": m.notes,
                        "kernel": m.kernel,
                        "chi_source_mpc": float(chi_source_mpc),
                        "source": m.source,
                    }
                )

                # Plot predicted amplitude histogram with observed band.
                plt.figure(figsize=(7, 4))
                plt.hist(A_draws, bins=50, density=True, alpha=0.65, label="A_pred draws")
                plt.axvline(1.0, color="k", linewidth=1.0, alpha=0.6, label="GR baseline (A=1)")
                plt.axvline(m.A_obs, color="C3", linewidth=1.5, label="A_obs")
                plt.axvspan(m.A_obs - m.A_sigma, m.A_obs + m.A_sigma, color="C3", alpha=0.15, label="A_obs ± 1σ")
                plt.xlabel("Amplitude A")
                plt.ylabel("Density")
                plt.title(f"{run_label} ({mapping_variant}) vs {m.name} ({conv}, {weight}) z=[{m.z_min:.2f},{m.z_max:.2f}]")
                plt.legend(fontsize=8)
                plt.tight_layout()
                out_png = fig_dir / f"amp_{run_label}__{m.name}__{conv}__{weight}.png"
                plt.savefig(out_png, dpi=160)
                plt.close()

    (tab_dir / "results.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

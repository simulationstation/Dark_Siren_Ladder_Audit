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

from entropy_horizon_recon.sirens import load_mu_forward_posterior, predict_dL_gw, predict_r_gw_em


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class BandSummary:
    run_dir: str
    convention: str
    n_draws: int
    z_min: float
    z_max: float
    z_n: int
    q: list[int]


def _quantiles(x: np.ndarray, q: list[int]) -> np.ndarray:
    # x: (n_draws, n_grid)
    return np.percentile(x, q, axis=0)


def _load_sirens(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "events" in data:
        events = data["events"]
    elif isinstance(data, list):
        events = data
    else:
        raise ValueError("Unsupported siren JSON format. Expected {'events': [...]} or a list.")
    if not isinstance(events, list):
        raise ValueError("Expected 'events' to be a list.")
    return events


def _logpdf_two_piece_normal(x: np.ndarray, mu: float, sig_lo: float, sig_hi: float) -> np.ndarray:
    """Two-piece (split) normal log-PDF evaluated at x.

    Normalization uses: f(x) = 2/(sig_lo+sig_hi) * phi((x-mu)/sig)/sig, sig=lo/hi by side.
    """
    x = np.asarray(x, dtype=float)
    if sig_lo <= 0 or sig_hi <= 0:
        raise ValueError("sig_lo/sig_hi must be positive.")
    sig = np.where(x <= mu, sig_lo, sig_hi)
    return (
        np.log(2.0)
        - np.log(sig_lo + sig_hi)
        - np.log(sig)
        - 0.5 * np.log(2.0 * np.pi)
        - 0.5 * ((x - mu) / sig) ** 2
    )


def _logpdf_normal(x: np.ndarray, mu: float, sig: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if sig <= 0:
        raise ValueError("sig must be positive.")
    return -np.log(sig) - 0.5 * np.log(2.0 * np.pi) - 0.5 * ((x - mu) / sig) ** 2


def _logmeanexp(logw: np.ndarray, axis: int = 0) -> np.ndarray:
    m = np.max(logw, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.mean(np.exp(logw - m), axis=axis, keepdims=True)), axis=axis)


def main() -> int:
    ap = argparse.ArgumentParser(description="Standard-siren GW vs EM propagation test utilities.")
    ap.add_argument("--run-dir", action="append", required=True, help="Finished run directory (contains samples/).")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/siren_test_<UTCSTAMP>).")
    ap.add_argument("--convention", choices=["A", "B"], default="A", help="R_GW/EM convention (see siren_test.md).")
    ap.add_argument("--z-min", type=float, default=0.0, help="Plot grid z_min.")
    ap.add_argument("--z-max", type=float, default=None, help="Plot grid z_max (default: min(run z_max)).")
    ap.add_argument("--z-n", type=int, default=200, help="Plot grid size.")
    ap.add_argument("--allow-extrapolation", action="store_true", help="Allow extrapolation outside inferred x-domain.")
    ap.add_argument("--siren-data", default=None, help="Optional siren JSON (events with z and distance).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_test_{_utc_stamp()}"
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    posts = []
    z_max_candidates = []
    for rd in args.run_dir:
        post = load_mu_forward_posterior(rd)
        posts.append((rd, post))
        z_max_candidates.append(float(post.z_grid[-1]))
    z_max = float(args.z_max) if args.z_max is not None else float(min(z_max_candidates))
    z_eval = np.linspace(float(args.z_min), z_max, int(args.z_n))

    q = [5, 16, 50, 84, 95]

    # Plot R(z) bands.
    plt.figure(figsize=(8, 5))
    for rd, post in posts:
        z, R = predict_r_gw_em(
            post,
            z_eval=z_eval,
            convention=args.convention,
            allow_extrapolation=bool(args.allow_extrapolation),
        )
        qs = _quantiles(R, q=q)  # (len(q), n_z)
        label = Path(rd).name
        plt.fill_between(z, qs[1], qs[3], alpha=0.25, label=f"{label} (p16-p84)")
        plt.plot(z, qs[2], linewidth=1.5, label=f"{label} (median)")

        summ = BandSummary(
            run_dir=str(rd),
            convention=str(args.convention),
            n_draws=int(R.shape[0]),
            z_min=float(z[0]),
            z_max=float(z[-1]),
            z_n=int(z.size),
            q=q,
        )
        (tab_dir / f"rgwem_band_{label}.json").write_text(json.dumps({**asdict(summ)}, indent=2))
        np.savez_compressed(
            tab_dir / f"rgwem_band_{label}.npz",
            z=z,
            q=np.array(q, dtype=int),
            R_q=qs,
        )

    plt.axhline(1.0, color="k", linewidth=1.0, alpha=0.5)
    plt.xlabel("Redshift z")
    plt.ylabel("R_GW/EM(z) = dL_GW / dL_EM")
    plt.title(f"GW Propagation Ratio Predicted From mu(A) Posterior (convention {args.convention})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_png = fig_dir / "rgwem_band.png"
    plt.savefig(out_png, dpi=160)
    plt.close()

    # Optional: evaluate against a siren dataset (simple Gaussian / split-normal distance errors).
    if args.siren_data:
        events = _load_sirens(Path(args.siren_data))
        rows = []
        for rd, post in posts:
            label = Path(rd).name
            for ev in events:
                name = str(ev.get("name", "event"))
                z_ev = float(ev["z"])
                # Predicted distances at this redshift.
                dL_gw, R_ev = predict_dL_gw(
                    post,
                    z_eval=np.array([z_ev], dtype=float),
                    convention=args.convention,  # type: ignore[arg-type]
                    allow_extrapolation=bool(args.allow_extrapolation),
                )
                dL_gw = dL_gw[:, 0]
                dL_em = dL_gw / R_ev[:, 0]

                dist_kind: Literal["normal", "two_piece_normal"] = str(ev.get("dist", "two_piece_normal"))  # type: ignore[assignment]
                if dist_kind == "normal":
                    mu = float(ev["dL_Mpc"])
                    sig = float(ev["dL_sigma_Mpc"])
                    logp_mu = _logpdf_normal(dL_gw, mu=mu, sig=sig)
                    logp_gr = _logpdf_normal(dL_em, mu=mu, sig=sig)
                elif dist_kind == "two_piece_normal":
                    mu = float(ev["dL_Mpc"])
                    sig_lo = float(ev["dL_sigma_lo_Mpc"])
                    sig_hi = float(ev["dL_sigma_hi_Mpc"])
                    logp_mu = _logpdf_two_piece_normal(dL_gw, mu=mu, sig_lo=sig_lo, sig_hi=sig_hi)
                    logp_gr = _logpdf_two_piece_normal(dL_em, mu=mu, sig_lo=sig_lo, sig_hi=sig_hi)
                else:
                    raise ValueError(f"Unsupported dist='{dist_kind}' for event '{name}'.")

                lpd_mu = float(_logmeanexp(logp_mu))
                lpd_gr = float(_logmeanexp(logp_gr))
                rows.append(
                    {
                        "run": label,
                        "event": name,
                        "z": z_ev,
                        "lpd_mu": lpd_mu,
                        "lpd_gr": lpd_gr,
                        "delta_lpd": lpd_mu - lpd_gr,
                    }
                )

        out_csv = tab_dir / "event_logpred.json"
        out_csv.write_text(json.dumps(rows, indent=2))

    print(f"Wrote: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


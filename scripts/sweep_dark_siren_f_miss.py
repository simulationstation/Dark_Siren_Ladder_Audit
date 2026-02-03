from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_sirens_selection import compute_selection_alpha_from_injections, load_o3_injections
from entropy_horizon_recon.sirens import load_mu_forward_posterior, MuForwardPosterior


def _logmeanexp(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_logmeanexp expects a 1D array")
    m = float(np.max(x))
    return float(m + np.log(np.mean(np.exp(x - m))))


def _downsample_posterior(post: MuForwardPosterior, *, draw_idx: list[int]) -> MuForwardPosterior:
    idx = np.asarray(draw_idx, dtype=int)

    def _sel(a):
        a = np.asarray(a)
        return a[idx]

    return MuForwardPosterior(
        x_grid=post.x_grid,
        logmu_x_samples=_sel(post.logmu_x_samples),
        z_grid=post.z_grid,
        H_samples=_sel(post.H_samples),
        H0=_sel(post.H0),
        omega_m0=_sel(post.omega_m0),
        omega_k0=_sel(post.omega_k0),
        sigma8_0=_sel(post.sigma8_0) if post.sigma8_0 is not None else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep dark-siren ΔLPD_total as a function of f_miss using cached per-event draw vectors.")
    ap.add_argument("--out-dir", required=True, help="Output directory from scripts/run_dark_siren_gap_test.py.")
    ap.add_argument("--f-miss-n", type=int, default=21, help="Number of f_miss grid points in [0,1] (default 21).")
    ap.add_argument("--f-miss-min", type=float, default=0.0, help="Min f_miss (default 0).")
    ap.add_argument("--f-miss-max", type=float, default=1.0, help="Max f_miss (default 1).")
    ap.add_argument("--no-selection", action="store_true", help="Skip selection normalization even if injections are present.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = json.loads((out_dir / "manifest.json").read_text())

    run_dirs = [str(x) for x in manifest["run_dir"]]
    if not run_dirs:
        raise ValueError("No run_dir entries in manifest.json")

    # Use summary_*.json to learn which draws were used and which events were scored.
    summaries = sorted(out_dir.glob("summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary_*.json found under {out_dir}")

    tables_dir = out_dir / "tables"
    cache_missing_dir = out_dir / "cache_missing"
    cache_terms_dir = out_dir / "cache_terms"
    if not cache_missing_dir.exists():
        raise FileNotFoundError(f"Missing {cache_missing_dir} (need a mixture-mode run with cache_missing).")
    if not cache_terms_dir.exists():
        raise FileNotFoundError(f"Missing {cache_terms_dir} (need cat_* caches from a rerun with current code).")

    f_grid = np.linspace(float(args.f_miss_min), float(args.f_miss_max), int(args.f_miss_n))
    f_grid = np.clip(f_grid, 0.0, 1.0)

    inj_path = manifest.get("selection_injections_hdf")
    injections = None
    if inj_path and not bool(args.no_selection):
        injections = load_o3_injections(
            inj_path,
            ifar_threshold_yr=float(manifest.get("selection_ifar_thresh_yr", 1.0)),
        )

    per_run = {}
    for p in summaries:
        run_label = p.stem.replace("summary_", "")
        s = json.loads(p.read_text())
        draw_idx = [int(i) for i in s["draw_idx"]]
        n_draws = int(s["n_draws"])
        if len(draw_idx) != n_draws:
            raise ValueError(f"{p}: draw_idx length {len(draw_idx)} != n_draws {n_draws}")

        # Determine event list from the per-event table.
        ev_rows = json.loads((tables_dir / f"event_scores_{run_label}.json").read_text())
        events = [str(r["event"]) for r in ev_rows]
        if not events:
            raise ValueError(f"{run_label}: no events in event_scores")

        # Load and downsample the EM posterior to match the cached draw vectors.
        # (This is needed only to compute selection alpha(draw) consistently.)
        post_full = load_mu_forward_posterior(Path([d for d in run_dirs if Path(d).name == run_label][0]))
        post = _downsample_posterior(post_full, draw_idx=draw_idx)

        log_alpha_mu = None
        log_alpha_gr = None
        alpha_meta = None
        if injections is not None:
            alpha = compute_selection_alpha_from_injections(
                injections=injections,
                post=post,
                convention=str(manifest.get("convention", "A")),  # type: ignore[arg-type]
                z_max=float(manifest.get("selection_z_max") or manifest.get("gal_z_max", 0.3)),
                det_model=str(manifest.get("selection_det_model", "snr_binned")),
                snr_threshold=float(manifest.get("selection_snr_thresh")) if manifest.get("selection_snr_thresh") is not None else None,
                snr_binned_nbins=int(manifest.get("selection_snr_binned_nbins", 200)),
                weight_mode=str(manifest.get("selection_weight_mode", "none")),
                pop_z_mode=str(manifest.get("selection_pop_z_mode", "none")),
                pop_z_powerlaw_k=float(manifest.get("selection_pop_z_k", 0.0)),
                pop_mass_mode=str(manifest.get("selection_pop_mass_mode", "none")),
                pop_m1_alpha=float(manifest.get("selection_pop_m1_alpha", 2.3)),
                pop_m_min=float(manifest.get("selection_pop_m_min", 5.0)),
                pop_m_max=float(manifest.get("selection_pop_m_max", 80.0)),
                pop_q_beta=float(manifest.get("selection_pop_q_beta", 0.0)),
            )
            log_alpha_mu = np.log(np.clip(alpha.alpha_mu, 1e-300, np.inf))
            log_alpha_gr = np.log(np.clip(alpha.alpha_gr, 1e-300, np.inf))
            alpha_meta = json.loads(alpha.to_json())

        # Load per-event draw vectors from caches.
        cat_mu = {}
        cat_gr = {}
        miss_mu = {}
        miss_gr = {}
        for ev in events:
            cat_path = cache_terms_dir / f"cat_{ev}__{run_label}.npz"
            miss_path = cache_missing_dir / f"missing_{ev}__{run_label}.npz"
            if not cat_path.exists():
                raise FileNotFoundError(f"Missing cat cache: {cat_path}")
            if not miss_path.exists():
                raise FileNotFoundError(f"Missing missing cache: {miss_path}")
            with np.load(cat_path, allow_pickle=True) as d:
                cat_mu[ev] = np.asarray(d["logL_cat_mu"], dtype=float)
                cat_gr[ev] = np.asarray(d["logL_cat_gr"], dtype=float)
            with np.load(miss_path, allow_pickle=True) as d:
                miss_mu[ev] = np.asarray(d["logL_missing_mu"], dtype=float)
                miss_gr[ev] = np.asarray(d["logL_missing_gr"], dtype=float)

        # Sweep f_miss.
        rows = []
        for f in f_grid.tolist():
            f = float(f)
            if f == 0.0:
                loga, logb = 0.0, -np.inf
            elif f == 1.0:
                loga, logb = -np.inf, 0.0
            else:
                loga, logb = float(np.log1p(-f)), float(np.log(f))

            logL_mu_total = np.zeros((n_draws,), dtype=float)
            logL_gr_total = np.zeros((n_draws,), dtype=float)
            for ev in events:
                logL_mu_ev = np.logaddexp(loga + cat_mu[ev], logb + miss_mu[ev])
                logL_gr_ev = np.logaddexp(loga + cat_gr[ev], logb + miss_gr[ev])
                logL_mu_total += logL_mu_ev
                logL_gr_total += logL_gr_ev

            if log_alpha_mu is not None and log_alpha_gr is not None:
                logL_mu_total = logL_mu_total - float(len(events)) * log_alpha_mu
                logL_gr_total = logL_gr_total - float(len(events)) * log_alpha_gr

            lpd_mu = _logmeanexp(logL_mu_total)
            lpd_gr = _logmeanexp(logL_gr_total)
            rows.append({"f_miss": f, "lpd_mu_total": lpd_mu, "lpd_gr_total": lpd_gr, "delta_lpd_total": lpd_mu - lpd_gr})

        per_run[run_label] = {
            "run_dir": str([d for d in run_dirs if Path(d).name == run_label][0]),
            "n_events": int(len(events)),
            "events": events,
            "selection_alpha": alpha_meta,
            "rows": rows,
        }

    out_path = out_dir / "f_miss_sweep.json"
    out_path.write_text(json.dumps({"out_dir": str(out_dir), "f_grid": [float(x) for x in f_grid.tolist()], "per_run": per_run}, indent=2) + "\n")
    print(f"Wrote {out_path}")

    # Quick figure: ΔLPD_total(f_miss) per seed + mean band.
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted(per_run.keys())
    y = np.array([[r["delta_lpd_total"] for r in per_run[k]["rows"]] for k in labels], dtype=float)
    y_mean = np.mean(y, axis=0)
    y_sd = np.std(y, axis=0, ddof=1) if y.shape[0] > 1 else np.zeros_like(y_mean)

    plt.figure(figsize=(7, 4))
    for i, lab in enumerate(labels):
        plt.plot(f_grid, y[i], alpha=0.35, linewidth=1.0, label=lab if i < 3 else None)
    plt.plot(f_grid, y_mean, color="k", linewidth=2.0, label="mean across seeds")
    plt.fill_between(f_grid, y_mean - y_sd, y_mean + y_sd, color="k", alpha=0.15, linewidth=0)
    plt.axhline(0.0, color="k", linewidth=1, alpha=0.35)
    plt.xlabel(r"$f_{\mathrm{miss}}$")
    plt.ylabel(r"$\Delta\mathrm{LPD}_{\mathrm{total}}$")
    plt.title("Dark-siren μ vs GR: sensitivity to missing-host fraction")
    plt.tight_layout()
    fig_path = fig_dir / "f_miss_sweep_delta_lpd_total.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"Wrote {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

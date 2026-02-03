from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from entropy_horizon_recon.dark_sirens_selection import (
    compute_selection_alpha_from_injections,
    load_o3_injections,
    resolve_o3_sensitivity_injection_file,
)
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


@dataclass(frozen=True)
class Variant:
    label: str
    no_selection: bool = False
    from_saved: bool = False
    # Selection config overrides (defaults come from manifest.json).
    selection_injections_hdf: str | None = None
    selection_ifar_thresh_yr: float | None = None
    selection_z_max: float | None = None
    selection_det_model: str | None = None
    selection_snr_thresh: float | None = None
    selection_snr_binned_nbins: int | None = None
    selection_weight_mode: str | None = None
    selection_pop_z_mode: str | None = None
    selection_pop_z_k: float | None = None
    selection_pop_mass_mode: str | None = None
    selection_pop_m1_alpha: float | None = None
    selection_pop_m_min: float | None = None
    selection_pop_m_max: float | None = None
    selection_pop_q_beta: float | None = None


def _load_variants(path: Path | None) -> list[Variant]:
    if path is None:
        return [
            Variant(label="baseline_saved", from_saved=True),
            Variant(label="no_selection", no_selection=True),
        ]

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError("alpha_sweep_json must contain a JSON list of variant dicts.")
    out: list[Variant] = []
    for i, d in enumerate(raw):
        if not isinstance(d, dict):
            raise ValueError(f"alpha_sweep_json entry {i} is not an object.")
        if "label" not in d:
            raise ValueError(f"alpha_sweep_json entry {i} missing required field 'label'.")
        out.append(Variant(**d))
    if not out:
        raise ValueError("alpha_sweep_json contains zero variants.")
    return out


def _resolve_run_dir(run_dirs: list[str], *, run_label: str) -> Path:
    matches = [Path(x) for x in run_dirs if Path(x).name == run_label]
    if not matches:
        raise FileNotFoundError(f"Could not resolve run_dir for run_label={run_label!r} from manifest run_dir list.")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous run_dir for run_label={run_label!r}: {matches}")
    return matches[0]


def _get_injections(*, variant: Variant, manifest: dict[str, Any], events: list[str]) -> tuple[Any | None, str | None, float]:
    inj_spec = (variant.selection_injections_hdf or str(manifest.get("selection_injections_hdf") or "none")).strip()
    ifar_thresh = float(
        variant.selection_ifar_thresh_yr
        if variant.selection_ifar_thresh_yr is not None
        else float(manifest.get("selection_ifar_thresh_yr", 1.0))
    )
    if inj_spec.lower() in ("none", "0", "false"):
        return None, None, ifar_thresh
    if inj_spec.lower() == "auto":
        resolved = resolve_o3_sensitivity_injection_file(
            events=[str(x) for x in events],
            base_dir="data/cache/gw/zenodo",
            record_id=7890437,
            population="mixture",
            auto_download=True,
        )
        return load_o3_injections(resolved, ifar_threshold_yr=ifar_thresh), str(resolved), ifar_thresh
    resolved = str(Path(inj_spec).expanduser().resolve())
    return load_o3_injections(resolved, ifar_threshold_yr=ifar_thresh), resolved, ifar_thresh


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Sweep hierarchical ΔLPD_total sensitivity to selection normalization α(model) using cached per-draw logL stacks."
    )
    ap.add_argument("--out-dir", required=True, help="Output directory from scripts/run_dark_siren_gap_test.py (hierarchical mode).")
    ap.add_argument(
        "--alpha-sweep-json",
        default=None,
        help="Optional JSON list of selection-variant dicts (each must include 'label'). If omitted, runs baseline_saved + no_selection.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest = json.loads((out_dir / "manifest.json").read_text())
    run_dirs = [str(x) for x in manifest.get("run_dir", [])]
    if not run_dirs:
        raise ValueError("No run_dir entries in manifest.json")

    tables_dir = out_dir / "tables"
    summaries = sorted(out_dir.glob("summary_*.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary_*.json found under {out_dir}")

    variants = _load_variants(Path(args.alpha_sweep_json).expanduser().resolve() if args.alpha_sweep_json else None)

    # Cache injections by resolved path to avoid repeated loads.
    injections_cache: dict[str, Any] = {}

    per_run: dict[str, Any] = {}
    for p in summaries:
        s = json.loads(p.read_text())
        run_label = str(s["run"])
        mode_label = str(s.get("mode", "real"))
        draw_idx = [int(i) for i in s.get("posterior_draw_idx", [])]
        if not draw_idx:
            raise ValueError(f"{p}: missing posterior_draw_idx (needed to recompute selection alpha).")

        stack_path = tables_dir / f"hier_logL_stack_{run_label}_{mode_label}.npz"
        if not stack_path.exists():
            raise FileNotFoundError(
                f"Missing {stack_path}. Re-run the hierarchical scoring with the current code to generate hier_logL_stack_*."
            )

        with np.load(stack_path, allow_pickle=True) as d:
            logL_mu_events = np.asarray(d["logL_mu_events"], dtype=float)
            logL_gr_events = np.asarray(d["logL_gr_events"], dtype=float)
            meta = json.loads(str(d["meta"]))

        if logL_mu_events.ndim != 2 or logL_gr_events.ndim != 2:
            raise ValueError(f"{stack_path}: expected 2D stacks (n_events, n_draws).")
        if logL_mu_events.shape != logL_gr_events.shape:
            raise ValueError(f"{stack_path}: mu/gr stack shape mismatch: {logL_mu_events.shape} vs {logL_gr_events.shape}.")
        n_ev, n_draws = (int(logL_mu_events.shape[0]), int(logL_mu_events.shape[1]))
        if len(draw_idx) != n_draws:
            raise ValueError(f"{p}: posterior_draw_idx length {len(draw_idx)} != stack n_draws {n_draws}")

        logL_mu_total_data = np.sum(logL_mu_events, axis=0)
        logL_gr_total_data = np.sum(logL_gr_events, axis=0)

        def _eval_variant(log_alpha_mu: np.ndarray | None, log_alpha_gr: np.ndarray | None) -> dict[str, float]:
            lpd_mu_data = _logmeanexp(logL_mu_total_data)
            lpd_gr_data = _logmeanexp(logL_gr_total_data)
            delta_data = lpd_mu_data - lpd_gr_data

            if log_alpha_mu is None or log_alpha_gr is None:
                lpd_mu = lpd_mu_data
                lpd_gr = lpd_gr_data
            else:
                if log_alpha_mu.shape != (n_draws,) or log_alpha_gr.shape != (n_draws,):
                    raise ValueError("log_alpha arrays must be shape (n_draws,).")
                lpd_mu = _logmeanexp(logL_mu_total_data - float(n_ev) * log_alpha_mu)
                lpd_gr = _logmeanexp(logL_gr_total_data - float(n_ev) * log_alpha_gr)

            delta_total = lpd_mu - lpd_gr
            return {
                "lpd_mu_total": float(lpd_mu),
                "lpd_gr_total": float(lpd_gr),
                "delta_lpd_total": float(delta_total),
                "delta_lpd_data": float(delta_data),
                "delta_lpd_sel": float(delta_total - delta_data),
            }

        rows: list[dict[str, Any]] = []
        for v in variants:
            log_alpha_mu = None
            log_alpha_gr = None
            alpha_meta: dict[str, Any] | None = None

            if v.no_selection:
                log_alpha_mu = None
                log_alpha_gr = None
            elif v.from_saved:
                saved_path = tables_dir / f"selection_alpha_{run_label}.npz"
                if not saved_path.exists():
                    raise FileNotFoundError(f"{saved_path} not found (needed for from_saved variant {v.label}).")
                with np.load(saved_path, allow_pickle=True) as d:
                    log_alpha_mu = np.asarray(d["log_alpha_mu"], dtype=float)
                    log_alpha_gr = np.asarray(d["log_alpha_gr"], dtype=float)
                    alpha_meta = json.loads(str(d["meta"])) if "meta" in d else None
            else:
                # Recompute alpha under this variant's selection settings.
                run_dir = _resolve_run_dir(run_dirs, run_label=run_label)
                post_full = load_mu_forward_posterior(run_dir)
                post = _downsample_posterior(post_full, draw_idx=draw_idx)

                injections, inj_path_resolved, ifar_thresh = _get_injections(variant=v, manifest=manifest, events=meta["events_scored"])
                if injections is None:
                    raise ValueError(f"Variant {v.label} requests recompute but selection_injections_hdf is disabled.")
                if inj_path_resolved is not None and inj_path_resolved in injections_cache:
                    injections = injections_cache[inj_path_resolved]
                elif inj_path_resolved is not None:
                    injections_cache[inj_path_resolved] = injections

                z_sel = float(v.selection_z_max) if v.selection_z_max is not None else float(manifest.get("selection_z_max") or post.z_grid[-1])

                alpha = compute_selection_alpha_from_injections(
                    injections=injections,
                    post=post,
                    convention=str(manifest.get("convention", "A")),  # type: ignore[arg-type]
                    z_max=z_sel,
                    det_model=str(v.selection_det_model or manifest.get("selection_det_model", "snr_binned")),  # type: ignore[arg-type]
                    snr_threshold=float(v.selection_snr_thresh)
                    if v.selection_snr_thresh is not None
                    else (float(manifest["selection_snr_thresh"]) if manifest.get("selection_snr_thresh") is not None else None),
                    snr_binned_nbins=int(v.selection_snr_binned_nbins or int(manifest.get("selection_snr_binned_nbins", 200))),
                    weight_mode=str(v.selection_weight_mode or manifest.get("selection_weight_mode", "none")),  # type: ignore[arg-type]
                    pop_z_mode=str(v.selection_pop_z_mode or manifest.get("selection_pop_z_mode", "none")),  # type: ignore[arg-type]
                    pop_z_powerlaw_k=float(v.selection_pop_z_k if v.selection_pop_z_k is not None else float(manifest.get("selection_pop_z_k", 0.0))),
                    pop_mass_mode=str(v.selection_pop_mass_mode or manifest.get("selection_pop_mass_mode", "none")),  # type: ignore[arg-type]
                    pop_m1_alpha=float(
                        v.selection_pop_m1_alpha if v.selection_pop_m1_alpha is not None else float(manifest.get("selection_pop_m1_alpha", 2.3))
                    ),
                    pop_m_min=float(v.selection_pop_m_min if v.selection_pop_m_min is not None else float(manifest.get("selection_pop_m_min", 5.0))),
                    pop_m_max=float(v.selection_pop_m_max if v.selection_pop_m_max is not None else float(manifest.get("selection_pop_m_max", 80.0))),
                    pop_q_beta=float(v.selection_pop_q_beta if v.selection_pop_q_beta is not None else float(manifest.get("selection_pop_q_beta", 0.0))),
                )

                log_alpha_mu = np.log(np.clip(alpha.alpha_mu, 1e-300, np.inf))
                log_alpha_gr = np.log(np.clip(alpha.alpha_gr, 1e-300, np.inf))
                alpha_meta = json.loads(alpha.to_json())
                alpha_meta["injections_path"] = inj_path_resolved
                alpha_meta["ifar_threshold_yr"] = ifar_thresh
                alpha_meta["variant"] = asdict(v)

                # Cache per-run alpha for reuse.
                out_path = tables_dir / f"selection_alpha_sweep_{v.label}_{run_label}.npz"
                np.savez(
                    out_path,
                    log_alpha_mu=log_alpha_mu,
                    log_alpha_gr=log_alpha_gr,
                    meta=json.dumps(alpha_meta, sort_keys=True),
                )

            metrics = _eval_variant(log_alpha_mu, log_alpha_gr)
            rows.append(
                {
                    "variant": v.label,
                    "n_events": n_ev,
                    "n_draws": n_draws,
                    "metrics": metrics,
                    "selection_alpha_meta": alpha_meta,
                }
            )

        per_run[run_label] = {
            "run_dir": str(_resolve_run_dir(run_dirs, run_label=run_label)),
            "mode": mode_label,
            "n_events": n_ev,
            "n_draws": n_draws,
            "events_scored": list(meta.get("events_scored", [])),
            "variants": rows,
        }

    # Aggregate across seeds.
    labels = [v.label for v in variants]
    agg: dict[str, Any] = {}
    for lab in labels:
        vals = np.array([float(next(r for r in per_run[k]["variants"] if r["variant"] == lab)["metrics"]["delta_lpd_total"]) for k in per_run], dtype=float)
        agg[lab] = {
            "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            "per_run": {k: float(next(r for r in per_run[k]["variants"] if r["variant"] == lab)["metrics"]["delta_lpd_total"]) for k in per_run},
        }

    out_path = out_dir / "alpha_selection_sensitivity_sweep.json"
    out_path.write_text(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "variants": [asdict(v) for v in variants],
                "per_run": per_run,
                "aggregate_delta_lpd_total": agg,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {out_path}")

    # Quick figure: ΔLPD_total by variant across seeds.
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(7, 1.2 * len(labels)), 4))
    xs = np.arange(len(labels))
    for i, run_label in enumerate(sorted(per_run.keys())):
        ys = np.array([float(next(r for r in per_run[run_label]["variants"] if r["variant"] == lab)["metrics"]["delta_lpd_total"]) for lab in labels], dtype=float)
        plt.plot(xs, ys, marker="o", alpha=0.35, linewidth=1.0)
    means = np.array([float(agg[lab]["mean"]) for lab in labels], dtype=float)
    sds = np.array([float(agg[lab]["sd"]) for lab in labels], dtype=float)
    plt.plot(xs, means, marker="o", color="k", linewidth=2.0, label="mean across seeds")
    plt.fill_between(xs, means - sds, means + sds, color="k", alpha=0.15, linewidth=0)
    plt.axhline(0.0, color="k", linewidth=1.0, alpha=0.4)
    plt.xticks(xs, labels, rotation=25, ha="right")
    plt.ylabel("ΔLPD_total (model − GR)")
    plt.title("Hierarchical selection α sensitivity sweep")
    plt.tight_layout()
    fig_path = fig_dir / "alpha_selection_sensitivity_sweep.png"
    plt.savefig(fig_path, dpi=160)
    plt.close()
    print(f"Wrote {fig_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


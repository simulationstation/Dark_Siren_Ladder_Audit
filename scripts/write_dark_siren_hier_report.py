from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_float(x: float, *, digits: int = 3) -> str:
    if not np.isfinite(x):
        return "nan"
    return f"{x:+.{digits}f}"


def _try_alpha_ratio_stats(out_dir: Path, *, run: str) -> dict[str, float]:
    """Return small summary stats for log(alpha_GR/alpha_mu), if available."""
    p = out_dir / "tables" / f"selection_alpha_{run}.npz"
    if not p.exists():
        return {}
    try:
        with np.load(p, allow_pickle=False) as d:
            log_mu = np.asarray(d["log_alpha_mu"], dtype=float)
            log_gr = np.asarray(d["log_alpha_gr"], dtype=float)
    except Exception:
        return {}
    if log_mu.shape != log_gr.shape or log_mu.ndim != 1 or log_mu.size == 0:
        return {}
    lr = log_gr - log_mu
    lr = lr[np.isfinite(lr)]
    if lr.size == 0:
        return {}
    return {
        "log_alpha_gr_over_mu_p50": float(np.median(lr)),
        "log_alpha_gr_over_mu_p05": float(np.quantile(lr, 0.05)),
        "log_alpha_gr_over_mu_p95": float(np.quantile(lr, 0.95)),
    }


@dataclass(frozen=True)
class RunBundle:
    run: str
    summary: dict[str, Any]
    event_rows: list[dict[str, Any]]
    jackknife: list[dict[str, Any]]
    alpha_ratio: dict[str, float]


def _load_run_bundle(out_dir: Path, *, run: str) -> RunBundle:
    summary = _read_json(out_dir / f"summary_{run}.json")
    event_rows_path = out_dir / "tables" / f"event_scores_{run}.json"
    jackknife_path = out_dir / "tables" / f"jackknife_{run}.json"
    event_rows = _read_json(event_rows_path) if event_rows_path.exists() else []
    jackknife = _read_json(jackknife_path) if jackknife_path.exists() else []
    alpha_ratio = _try_alpha_ratio_stats(out_dir, run=run)
    return RunBundle(run=str(run), summary=dict(summary), event_rows=list(event_rows), jackknife=list(jackknife), alpha_ratio=dict(alpha_ratio))


def main() -> int:
    ap = argparse.ArgumentParser(description="Write a compact markdown summary for hierarchical PE dark-siren runs.")
    ap.add_argument("--out-dir", required=True, help="Runner output directory containing summary_*.json and tables/.*")
    ap.add_argument("--write-to", default=None, help="Output markdown path (default: <out-dir>/report_hier.md).")
    ap.add_argument("--top-k", type=int, default=10, help="How many top events to list (default 10).")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir)).expanduser().resolve()
    if not out_dir.exists():
        raise SystemExit(f"Missing out-dir: {out_dir}")

    runs: list[str] = []
    for p in sorted(out_dir.glob("summary_*.json")):
        try:
            s = _read_json(p)
        except Exception:
            continue
        runs.append(str(s.get("run", p.stem.replace("summary_", ""))))
    runs = sorted(set(runs))
    if not runs:
        raise SystemExit(f"No summary_*.json found under {out_dir}")

    bundles = [_load_run_bundle(out_dir, run=r) for r in runs]

    deltas = np.array([float(b.summary.get("delta_lpd_total", float("nan"))) for b in bundles], dtype=float)
    deltas_data = np.array([float(b.summary.get("delta_lpd_total_data", float("nan"))) for b in bundles], dtype=float)
    deltas_sel = np.array([float(b.summary.get("delta_lpd_total_sel", float("nan"))) for b in bundles], dtype=float)

    ok = np.isfinite(deltas)
    mean_tot = float(np.mean(deltas[ok])) if np.any(ok) else float("nan")
    sd_tot = float(np.std(deltas[ok], ddof=1)) if int(np.sum(ok)) >= 2 else 0.0

    # Per-event mean ΔLPD across runs.
    by_event: dict[str, list[float]] = {}
    for b in bundles:
        for r in b.event_rows:
            ev = str(r.get("event", ""))
            if not ev:
                continue
            by_event.setdefault(ev, []).append(float(r.get("delta_lpd", float("nan"))))

    event_means: list[tuple[str, float, float, int]] = []
    for ev, vals in by_event.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        event_means.append((ev, float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0, int(arr.size)))
    event_means.sort(key=lambda t: t[1], reverse=True)

    # Per-event mean jackknife influence across runs.
    infl_by_event: dict[str, list[float]] = {}
    for b in bundles:
        for r in b.jackknife:
            ev = str(r.get("event", ""))
            if not ev:
                continue
            infl_by_event.setdefault(ev, []).append(float(r.get("influence", float("nan"))))

    infl_means: list[tuple[str, float, float, int]] = []
    for ev, vals in infl_by_event.items():
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        infl_means.append((ev, float(np.mean(arr)), float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0, int(arr.size)))
    infl_means.sort(key=lambda t: t[1], reverse=True)

    out_path = Path(str(args.write_to)).expanduser().resolve() if args.write_to else (out_dir / "report_hier.md")

    lines: list[str] = []
    lines.append("# Hierarchical PE dark-siren run summary")
    lines.append("")
    lines.append(f"- generated: `{_utc_stamp()}`")
    lines.append(f"- out-dir: `{out_dir}`")
    lines.append(f"- runs: {', '.join(f'`{r}`' for r in runs)}")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append("| run | n_events | ΔLPD_data | ΔLPD_sel | ΔLPD_total | min good(mu/gr) | min ESS(mu/gr) | log(α_GR/α_μ) p50 [p05,p95] |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for b, dd, ds, dt in zip(bundles, deltas_data, deltas_sel, deltas, strict=True):
        s = b.summary
        n_ev = int(s.get("n_events", -1))
        gmu = float(s.get("pe_good_frac_mu_min", float("nan")))
        ggr = float(s.get("pe_good_frac_gr_min", float("nan")))
        emu = float(s.get("pe_weight_ess_mu_min", float("nan")))
        egr = float(s.get("pe_weight_ess_gr_min", float("nan")))
        ar = b.alpha_ratio
        if ar:
            a_txt = f"{ar['log_alpha_gr_over_mu_p50']:+.3f} [{ar['log_alpha_gr_over_mu_p05']:+.3f},{ar['log_alpha_gr_over_mu_p95']:+.3f}]"
        else:
            a_txt = "(none)"
        lines.append(
            f"| `{b.run}` | {n_ev:d} | {_format_float(float(dd))} | {_format_float(float(ds))} | {_format_float(float(dt))} | "
            f"{gmu:.3f}/{ggr:.3f} | {emu:.1f}/{egr:.1f} | {a_txt} |"
        )

    lines.append("")
    lines.append(f"Across-run ΔLPD_total mean±sd: `{mean_tot:+.3f}±{sd_tot:.3f}`")

    lines.append("")
    lines.append("## Top events by mean ΔLPD (per-event totals)")
    lines.append("")
    k = int(max(1, args.top_k))
    for ev, m, sd, n in event_means[:k]:
        lines.append(f"- `{ev}`: mean ΔLPD={m:+.3f} (sd {sd:.3f}, n={n})")

    lines.append("")
    lines.append("## Top events by jackknife influence")
    lines.append("")
    for ev, m, sd, n in infl_means[:k]:
        lines.append(f"- `{ev}`: mean influence={m:+.3f} (sd {sd:.3f}, n={n})")

    lines.append("")
    lines.append("## Skipped events")
    lines.append("")
    any_skipped = False
    for b in bundles:
        skipped = list(b.summary.get("events_skipped", []) or [])
        if not skipped:
            continue
        any_skipped = True
        lines.append(f"- `{b.run}`:")
        for r in skipped:
            ev = str(r.get("event", ""))
            reason = str(r.get("reason", ""))
            details = str(r.get("details", ""))
            if details:
                lines.append(f"  - `{ev}`: {reason} ({details})")
            else:
                lines.append(f"  - `{ev}`: {reason}")
    if not any_skipped:
        lines.append("- (none)")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


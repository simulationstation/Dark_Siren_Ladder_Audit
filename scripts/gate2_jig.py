#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float_array(x: Any, *, name: str) -> np.ndarray:
    if not isinstance(x, list):
        raise ValueError(f"Expected '{name}' to be a JSON list.")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"Expected '{name}' to be a 1D list with >=2 entries.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Non-finite values in '{name}'.")
    return arr


def _slope(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    m = np.isfinite(y) & np.isfinite(x)
    y = y[m]
    x = x[m]
    if y.size < 2:
        return float("nan")
    x0 = x - float(np.mean(x))
    denom = float(np.sum(x0 * x0))
    if not (math.isfinite(denom) and denom > 0.0):
        return float("nan")
    return float(np.sum((y - float(np.mean(y))) * x0) / denom)


def _quantile(x: np.ndarray, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Posterior normalization failed (non-positive sum).")
    return p / s


def _prob_from_logp(logp: np.ndarray) -> np.ndarray:
    logp = np.asarray(logp, dtype=float)
    if logp.ndim != 1 or logp.size < 2 or np.any(~np.isfinite(logp)):
        raise ValueError("logp must be a 1D finite array with >=2 entries.")
    logp = logp - float(np.max(logp))
    return _normalize_prob(np.exp(logp))


def _summary_from_posterior(H0: np.ndarray, p: np.ndarray) -> dict[str, float]:
    H0 = np.asarray(H0, dtype=float)
    p = _normalize_prob(np.asarray(p, dtype=float))
    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0))
    q50 = float(np.interp(0.50, cdf, H0))
    q84 = float(np.interp(0.84, cdf, H0))
    mean = float(np.sum(p * H0))
    sd = float(np.sqrt(np.sum(p * (H0 - mean) ** 2)))
    H0_map = float(H0[int(np.argmax(p))])
    return {"H0_map": H0_map, "mean": mean, "sd": sd, "p16": q16, "p50": q50, "p84": q84}


@dataclass(frozen=True)
class Gate2Summary:
    json_path: Path
    out_dir: Path
    selection_mode: str
    det_model: str | None
    weight_mode: str | None
    pop_z_mode: str | None
    pop_mass_mode: str | None
    n_events: int
    H0_min: float
    H0_max: float
    H0_n: int
    H0_map: float
    H0_edge: str
    slope_event: float
    slope_sel: float | None
    slope_total: float
    delta_event: float
    delta_sel: float | None
    delta_total: float
    exp_event_per_event: float


def _edge_side(H0_grid: np.ndarray, H0_map: float, *, atol: float = 1e-9) -> str:
    h0_min = float(np.min(H0_grid))
    h0_max = float(np.max(H0_grid))
    if math.isfinite(H0_map) and abs(H0_map - h0_min) <= atol:
        return "low"
    if math.isfinite(H0_map) and abs(H0_map - h0_max) <= atol:
        return "high"
    return "interior"


def summarize_gate2_json(path: Path) -> Gate2Summary:
    d = _read_json(path)
    if str(d.get("method", "")) != "gr_h0_grid_hierarchical_pe":
        raise ValueError(f"{path}: not a Gate-2 GR H0 JSON (method mismatch).")

    H0_grid = _as_float_array(d.get("H0_grid"), name="H0_grid")
    logH = np.log(np.clip(H0_grid, 1e-300, np.inf))

    logL_sum_events_rel = _as_float_array(d.get("logL_sum_events_rel"), name="logL_sum_events_rel")
    if logL_sum_events_rel.shape != H0_grid.shape:
        raise ValueError(f"{path}: logL_sum_events_rel shape mismatch.")

    logL_total_rel = _as_float_array(d.get("logL_H0_rel"), name="logL_H0_rel")
    if logL_total_rel.shape != H0_grid.shape:
        raise ValueError(f"{path}: logL_H0_rel shape mismatch.")

    log_alpha_grid = d.get("log_alpha_grid")
    slope_sel = None
    delta_sel = None
    if log_alpha_grid is not None:
        log_alpha = _as_float_array(log_alpha_grid, name="log_alpha_grid")
        if log_alpha.shape != H0_grid.shape:
            raise ValueError(f"{path}: log_alpha_grid shape mismatch.")
        n = int(d.get("n_events", 0))
        sel_term = -float(n) * log_alpha
        slope_sel = _slope(sel_term, logH)
        delta_sel = float(sel_term[-1] - sel_term[0])

    n_events = int(d.get("n_events", 0))
    if n_events <= 0:
        raise ValueError(f"{path}: invalid n_events={n_events}")

    slope_event = _slope(logL_sum_events_rel, logH)
    slope_total = _slope(logL_total_rel, logH)
    delta_event = float(logL_sum_events_rel[-1] - logL_sum_events_rel[0])
    delta_total = float(logL_total_rel[-1] - logL_total_rel[0])
    dlogH = float(logH[-1] - logH[0])
    exp_event_per_event = float("nan")
    if math.isfinite(dlogH) and dlogH != 0.0:
        exp_event_per_event = float(delta_event / dlogH / float(n_events))

    H0_map = float(d.get("H0_map", float("nan")))
    edge = _edge_side(H0_grid, H0_map)

    selection_mode = "on" if (log_alpha_grid is not None) else "off"
    sel_meta = d.get("selection_alpha")
    sel_meta = sel_meta if isinstance(sel_meta, dict) else {}

    pop = d.get("population")
    pop = pop if isinstance(pop, dict) else {}

    out_dir = path.parent.parent if path.name.endswith(".json") and path.parent.name == "json" else path.parent
    return Gate2Summary(
        json_path=path,
        out_dir=out_dir,
        selection_mode=selection_mode,
        det_model=str(sel_meta.get("det_model")) if sel_meta.get("det_model") is not None else None,
        weight_mode=str(sel_meta.get("weight_mode")) if sel_meta.get("weight_mode") is not None else None,
        pop_z_mode=str(pop.get("pop_z_mode")) if pop.get("pop_z_mode") is not None else None,
        pop_mass_mode=str(pop.get("pop_mass_mode")) if pop.get("pop_mass_mode") is not None else None,
        n_events=n_events,
        H0_min=float(H0_grid[0]),
        H0_max=float(H0_grid[-1]),
        H0_n=int(H0_grid.size),
        H0_map=H0_map,
        H0_edge=edge,
        slope_event=slope_event,
        slope_sel=slope_sel,
        slope_total=slope_total,
        delta_event=delta_event,
        delta_sel=delta_sel,
        delta_total=delta_total,
        exp_event_per_event=exp_event_per_event,
    )


def _print_single_report(summary: Gate2Summary, *, top_n: int, json_obj: dict[str, Any]) -> None:
    print(f"[gate2_jig] json={summary.json_path}")
    print(f"[gate2_jig] out_dir={summary.out_dir}")
    print(f"[gate2_jig] selection={summary.selection_mode} det_model={summary.det_model} weight_mode={summary.weight_mode}")
    print(f"[gate2_jig] pop_z={summary.pop_z_mode} pop_mass={summary.pop_mass_mode} n_events={summary.n_events}")
    print(f"[gate2_jig] H0_map={summary.H0_map:.3f} edge={summary.H0_edge} grid=[{summary.H0_min:.1f},{summary.H0_max:.1f}] n={summary.H0_n}")
    print("[gate2_jig] term slopes vs log(H0):")
    print(f"  event_sum: {summary.slope_event:.3f}   (Δ endpoint={summary.delta_event:+.3f})")
    if summary.slope_sel is not None and summary.delta_sel is not None:
        print(f"  -Nlogα   : {summary.slope_sel:.3f}   (Δ endpoint={summary.delta_sel:+.3f})")
    else:
        print("  -Nlogα   : (selection off)")
    print(f"  total    : {summary.slope_total:.3f}   (Δ endpoint={summary.delta_total:+.3f})")
    print(f"[gate2_jig] effective event scaling ≈ H0^({summary.exp_event_per_event:.3f}) per event (from endpoints)")

    # Per-event exponents
    H0_grid = _as_float_array(json_obj.get("H0_grid"), name="H0_grid")
    logH = np.log(np.clip(H0_grid, 1e-300, np.inf))
    evs = json_obj.get("events")
    if not isinstance(evs, list):
        return

    recs = []
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("event", ""))
        ll = ev.get("logL_H0")
        if not isinstance(ll, list):
            continue
        ll_arr = np.asarray(ll, dtype=float)
        if ll_arr.shape != H0_grid.shape:
            continue
        b = _slope(ll_arr, logH)
        delta = float(ll_arr[-1] - ll_arr[0])
        ess_min = ev.get("ess_min")
        try:
            ess_min_f = float(ess_min) if ess_min is not None else float("nan")
        except Exception:
            ess_min_f = float("nan")
        recs.append((name, b, delta, ess_min_f))

    if not recs:
        return
    bvals = np.asarray([r[1] for r in recs], dtype=float)
    print("[gate2_jig] per-event exponent b in logZ ≈ a + b log(H0):")
    print(
        f"  b median={float(np.median(bvals)):.3f} mean={float(np.mean(bvals)):.3f} "
        f"p10={_quantile(bvals,0.1):.3f} p90={_quantile(bvals,0.9):.3f}"
    )

    recs.sort(key=lambda t: t[1], reverse=True)
    print(f"[gate2_jig] top {top_n} events by b (most push to high H0):")
    for name, b, delta, ess_min in recs[: max(0, int(top_n))]:
        print(f"  {name}: b={b:.3f} ΔlogZ={delta:+.3f} ess_min={ess_min:.2f}")
    print(f"[gate2_jig] bottom {top_n} events by b (least push / push low):")
    for name, b, delta, ess_min in recs[-max(0, int(top_n)) :]:
        print(f"  {name}: b={b:.3f} ΔlogZ={delta:+.3f} ess_min={ess_min:.2f}")


def _parse_float_list(spec: str, *, name: str) -> list[float]:
    items = [s.strip() for s in str(spec).split(",") if s.strip()]
    out: list[float] = []
    for s in items:
        try:
            out.append(float(s))
        except Exception as e:
            raise ValueError(f"Invalid {name} entry '{s}': {e}") from e
    return out


def _parse_int_list(spec: str, *, name: str) -> list[int]:
    items = [s.strip() for s in str(spec).split(",") if s.strip()]
    out: list[int] = []
    for s in items:
        try:
            out.append(int(s))
        except Exception as e:
            raise ValueError(f"Invalid {name} entry '{s}': {e}") from e
    return out


def _event_records_from_gate2_json(json_obj: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    H0_grid = _as_float_array(json_obj.get("H0_grid"), name="H0_grid")
    evs = json_obj.get("events")
    if not isinstance(evs, list) or not evs:
        raise ValueError("Gate-2 JSON missing non-empty 'events' list.")
    out: list[dict[str, Any]] = []
    for ev in evs:
        if not isinstance(ev, dict):
            continue
        name = str(ev.get("event", ""))
        ll = ev.get("logL_H0")
        if not isinstance(ll, list):
            continue
        ll_arr = np.asarray(ll, dtype=float)
        if ll_arr.shape != H0_grid.shape or np.any(~np.isfinite(ll_arr)):
            continue
        ess_min = ev.get("ess_min")
        finite_frac = ev.get("finite_frac")
        try:
            ess_min_f = float(ess_min) if ess_min is not None else float("nan")
        except Exception:
            ess_min_f = float("nan")
        try:
            finite_frac_f = float(finite_frac) if finite_frac is not None else float("nan")
        except Exception:
            finite_frac_f = float("nan")

        out.append(
            {
                "event": name,
                "logL_H0": ll_arr,
                "ess_min": ess_min_f,
                "finite_frac": finite_frac_f,
            }
        )
    if not out:
        raise ValueError("No usable per-event logL_H0 arrays found in Gate-2 JSON.")
    return H0_grid, out


def _recompute_posterior_from_events(
    *,
    H0_grid: np.ndarray,
    events: list[dict[str, Any]],
    log_alpha_grid: np.ndarray | None,
) -> dict[str, Any]:
    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("Invalid H0_grid.")
    if not events:
        raise ValueError("No events provided.")

    logL_sum = np.zeros_like(H0_grid, dtype=float)
    for ev in events:
        ll = np.asarray(ev["logL_H0"], dtype=float)
        ll = ll - float(np.max(ll))  # stable; only changes by additive constants
        logL_sum = logL_sum + ll

    sel_term = None
    if log_alpha_grid is not None:
        log_alpha_grid = np.asarray(log_alpha_grid, dtype=float)
        if log_alpha_grid.shape != H0_grid.shape or np.any(~np.isfinite(log_alpha_grid)):
            raise ValueError("log_alpha_grid shape mismatch or non-finite values.")
        sel_term = -float(len(events)) * log_alpha_grid

    logL_total = logL_sum + (sel_term if sel_term is not None else 0.0)

    p = _prob_from_logp(logL_total)
    summary = _summary_from_posterior(H0_grid, p)
    H0_map = float(summary["H0_map"])
    edge = _edge_side(H0_grid, H0_map)

    logH = np.log(np.clip(H0_grid, 1e-300, np.inf))
    slope_event = _slope(logL_sum, logH)
    delta_event = float(logL_sum[-1] - logL_sum[0])
    slope_sel = None
    delta_sel = None
    if sel_term is not None:
        slope_sel = _slope(sel_term, logH)
        delta_sel = float(sel_term[-1] - sel_term[0])
    slope_total = _slope(logL_total, logH)
    delta_total = float(logL_total[-1] - logL_total[0])
    dlogH = float(logH[-1] - logH[0])
    exp_event_per_event = float("nan")
    if math.isfinite(dlogH) and dlogH != 0.0:
        exp_event_per_event = float(delta_event / dlogH / float(len(events)))

    return {
        "n_events": int(len(events)),
        "H0_map": H0_map,
        "H0_edge": edge,
        "summary": summary,
        "slope_event": slope_event,
        "slope_sel": slope_sel,
        "slope_total": slope_total,
        "delta_event": delta_event,
        "delta_sel": delta_sel,
        "delta_total": delta_total,
        "exp_event_per_event": exp_event_per_event,
    }


def _scan_gate2_jsons(outputs_dir: Path) -> list[Path]:
    out: list[Path] = []
    for p in outputs_dir.glob("**/json/gr_h0_selection_*.json"):
        out.append(p)
    return sorted(out)


def _write_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_csv.write_text("no_rows\n", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-2 jig: diagnose monotonic H0 behavior from saved Gate-2 JSON outputs.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_one = sub.add_parser("one", help="Analyze a single Gate-2 JSON (prints per-term slopes and per-event exponents).")
    ap_one.add_argument("--json", type=Path, required=True, help="Path to a gr_h0_selection_*.json file.")
    ap_one.add_argument("--top-n", type=int, default=10, help="How many events to print in top/bottom lists (default 10).")

    ap_filter = sub.add_parser("filter", help="Recompute posterior after filtering/dropping events; writes a CSV of summaries.")
    ap_filter.add_argument("--json", type=Path, required=True, help="Path to a gr_h0_selection_*.json file (must include per-event logL_H0 arrays).")
    ap_filter.add_argument("--out", type=Path, default=Path("FINDINGS") / "gate2_filter.csv", help="CSV to write (default FINDINGS/gate2_filter.csv).")
    ap_filter.add_argument("--relative-to", type=Path, default=Path("."), help="Base path for relative json_path in CSV (default repo root).")
    ap_filter.add_argument("--min-ess-list", default="0,1,10,100,1000", help="Comma list of ESS thresholds to scan (default 0,1,10,100,1000).")
    ap_filter.add_argument("--min-finite-frac-list", default="0.0,0.5,0.9", help="Comma list of finite_frac thresholds to scan (default 0.0,0.5,0.9).")
    ap_filter.add_argument("--drop-worst-ess-list", default="0,1,2,3,5", help="Comma list of N to drop by lowest ess_min (default 0,1,2,3,5).")
    ap_filter.add_argument("--drop-top-b-list", default="0,1,2,3,5", help="Comma list of N to drop by highest b exponent (default 0,1,2,3,5).")

    ap_scan = sub.add_parser("scan", help="Scan outputs/**/json/gr_h0_selection_*.json into a CSV summary.")
    ap_scan.add_argument("--outputs", type=Path, default=Path("outputs"), help="Outputs dir to scan (default outputs/).")
    ap_scan.add_argument("--out", type=Path, default=Path("FINDINGS") / "gate2_jig.csv", help="CSV to write (default FINDINGS/gate2_jig.csv).")
    ap_scan.add_argument("--relative-to", type=Path, default=Path("."), help="Base path for relative paths in CSV (default repo root).")

    args = ap.parse_args()

    if args.cmd == "one":
        path = Path(args.json).expanduser().resolve()
        d = _read_json(path)
        s = summarize_gate2_json(path)
        _print_single_report(s, top_n=int(args.top_n), json_obj=d)
        return 0

    if args.cmd == "filter":
        path = Path(args.json).expanduser().resolve()
        d = _read_json(path)
        H0_grid, recs = _event_records_from_gate2_json(d)
        log_alpha_grid = None
        if d.get("log_alpha_grid") is not None:
            log_alpha_grid = _as_float_array(d.get("log_alpha_grid"), name="log_alpha_grid")

        rel = Path(args.relative_to).expanduser().resolve()

        logH = np.log(np.clip(H0_grid, 1e-300, np.inf))
        for r in recs:
            ll = np.asarray(r["logL_H0"], dtype=float)
            r["b"] = _slope(ll, logH)

        rows: list[dict[str, Any]] = []

        def _relpath(p: Path) -> str:
            try:
                return str(p.resolve().relative_to(rel))
            except Exception:
                return str(p)

        def _add_row(tag: str, *, kept: list[dict[str, Any]], detail: str, dropped: list[str]) -> None:
            res = _recompute_posterior_from_events(H0_grid=H0_grid, events=kept, log_alpha_grid=log_alpha_grid)
            ess = np.asarray([float(x.get("ess_min", float("nan"))) for x in kept], dtype=float)
            ff = np.asarray([float(x.get("finite_frac", float("nan"))) for x in kept], dtype=float)
            b = np.asarray([float(x.get("b", float("nan"))) for x in kept], dtype=float)
            s = res.get("summary", {}) or {}
            rows.append(
                {
                    "json_path": _relpath(path),
                    "tag": tag,
                    "detail": detail,
                    "n_events": int(res["n_events"]),
                    "H0_map": float(res["H0_map"]),
                    "H0_edge": str(res["H0_edge"]),
                    "p50": float(s.get("p50", float("nan"))),
                    "p16": float(s.get("p16", float("nan"))),
                    "p84": float(s.get("p84", float("nan"))),
                    "mean": float(s.get("mean", float("nan"))),
                    "sd": float(s.get("sd", float("nan"))),
                    "slope_event": float(res["slope_event"]),
                    "slope_sel": float(res["slope_sel"]) if res.get("slope_sel") is not None else float("nan"),
                    "slope_total": float(res["slope_total"]),
                    "delta_event": float(res["delta_event"]),
                    "delta_sel": float(res["delta_sel"]) if res.get("delta_sel") is not None else float("nan"),
                    "delta_total": float(res["delta_total"]),
                    "exp_event_per_event": float(res["exp_event_per_event"]),
                    "ess_min_min": float(np.nanmin(ess)) if ess.size else float("nan"),
                    "ess_min_median": float(np.nanmedian(ess)) if ess.size else float("nan"),
                    "finite_frac_min": float(np.nanmin(ff)) if ff.size else float("nan"),
                    "b_median": float(np.nanmedian(b)) if b.size else float("nan"),
                    "dropped_events": ";".join(dropped[:50]) + (";..." if len(dropped) > 50 else ""),
                }
            )

        # Baseline row.
        _add_row("baseline", kept=list(recs), detail="none", dropped=[])

        # Threshold scans.
        for thr in _parse_float_list(str(args.min_ess_list), name="min-ess-list"):
            kept = [r for r in recs if np.isfinite(float(r.get("ess_min", float("nan")))) and float(r["ess_min"]) >= float(thr)]
            dropped = [r["event"] for r in recs if r not in kept]
            if kept:
                _add_row("min_ess", kept=kept, detail=f">={thr:g}", dropped=dropped)

        for thr in _parse_float_list(str(args.min_finite_frac_list), name="min-finite-frac-list"):
            kept = [r for r in recs if np.isfinite(float(r.get("finite_frac", float("nan")))) and float(r["finite_frac"]) >= float(thr)]
            dropped = [r["event"] for r in recs if r not in kept]
            if kept:
                _add_row("min_finite_frac", kept=kept, detail=f">={thr:g}", dropped=dropped)

        # Drop scans.
        for n_drop in _parse_int_list(str(args.drop_worst_ess_list), name="drop-worst-ess-list"):
            n_drop = max(0, int(n_drop))
            ordered = sorted(recs, key=lambda r: float(r.get("ess_min", float("nan"))))
            dropped_recs = ordered[:n_drop]
            kept = [r for r in recs if r not in dropped_recs]
            dropped = [r["event"] for r in dropped_recs]
            if kept:
                _add_row("drop_worst_ess", kept=kept, detail=str(n_drop), dropped=dropped)

        for n_drop in _parse_int_list(str(args.drop_top_b_list), name="drop-top-b-list"):
            n_drop = max(0, int(n_drop))
            ordered = sorted(recs, key=lambda r: float(r.get("b", float("nan"))), reverse=True)
            dropped_recs = ordered[:n_drop]
            kept = [r for r in recs if r not in dropped_recs]
            dropped = [r["event"] for r in dropped_recs]
            if kept:
                _add_row("drop_top_b", kept=kept, detail=str(n_drop), dropped=dropped)

        _write_csv(rows, Path(args.out))
        print(f"[gate2_jig] wrote {args.out} ({len(rows)} rows)")
        return 0

    if args.cmd == "scan":
        outputs = Path(args.outputs).expanduser().resolve()
        rel = Path(args.relative_to).expanduser().resolve()
        paths = _scan_gate2_jsons(outputs)
        rows: list[dict[str, Any]] = []
        for p in paths:
            try:
                s = summarize_gate2_json(p)
            except Exception:
                continue

            def _rel(p2: Path) -> str:
                try:
                    return str(p2.resolve().relative_to(rel))
                except Exception:
                    return str(p2)

            rows.append(
                {
                    "out_dir": _rel(s.out_dir),
                    "json_path": _rel(s.json_path),
                    "selection_mode": s.selection_mode,
                    "det_model": s.det_model,
                    "weight_mode": s.weight_mode,
                    "pop_z_mode": s.pop_z_mode,
                    "pop_mass_mode": s.pop_mass_mode,
                    "n_events": s.n_events,
                    "H0_min": s.H0_min,
                    "H0_max": s.H0_max,
                    "H0_n": s.H0_n,
                    "H0_map": s.H0_map,
                    "H0_edge": s.H0_edge,
                    "slope_event": s.slope_event,
                    "slope_sel": s.slope_sel,
                    "slope_total": s.slope_total,
                    "delta_event": s.delta_event,
                    "delta_sel": s.delta_sel,
                    "delta_total": s.delta_total,
                    "exp_event_per_event": s.exp_event_per_event,
                }
            )
        _write_csv(rows, Path(args.out))
        print(f"[gate2_jig] wrote {args.out} ({len(rows)} rows)")
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

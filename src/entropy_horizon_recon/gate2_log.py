from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from pathlib import Path
from typing import Any

_UTC_STAMP_RE = re.compile(r"(\d{8}_\d{6}UTC)")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _parse_utc_stamp(s: str | None) -> dt.datetime | None:
    if not s:
        return None
    try:
        return dt.datetime.strptime(str(s), "%Y%m%d_%H%M%SUTC").replace(tzinfo=dt.timezone.utc)
    except Exception:
        return None


def _nearest_manifest_json(start_dir: Path, *, stop_dir: Path) -> Path | None:
    start_dir = start_dir.resolve()
    stop_dir = stop_dir.resolve()
    p = start_dir
    while True:
        for cand in (p / "manifest.json", p / "json" / "manifest.json"):
            if cand.exists():
                return cand
        if p == stop_dir or p.parent == p:
            return None
        p = p.parent


def _infer_timestamp_utc(*, out_dir: Path, json_path: Path, manifest: dict[str, Any] | None) -> tuple[str | None, str | None]:
    ts = None if manifest is None else manifest.get("timestamp_utc")
    if ts:
        ts_str = str(ts)
        dt_parsed = _parse_utc_stamp(ts_str)
        return ts_str, dt_parsed.isoformat() if dt_parsed else None

    m = _UTC_STAMP_RE.search(str(out_dir))
    if m:
        ts_str = m.group(1)
        dt_parsed = _parse_utc_stamp(ts_str)
        return ts_str, dt_parsed.isoformat() if dt_parsed else None

    try:
        mtime = json_path.stat().st_mtime
        dt_iso = dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat()
        return None, dt_iso
    except Exception:
        return None, None


def _compact_params(*, d: dict[str, Any], selection_mode: str, weight_mode: str | None, det_model: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in (
        "method",
        "prior",
        "omega_m0",
        "omega_k0",
        "z_max",
        "n_events",
        "n_events_skipped",
        "selection_ifar_threshold_yr",
        "include_pdet_in_event_term",
        "pop_z_include_h0_volume_scaling",
        "selection_include_h0_volume_scaling",
        "inj_mass_pdf_coords",
        "importance_smoothing",
        "importance_truncate_tau",
        "event_qc_mode",
        "event_min_finite_frac",
    ):
        if k in d:
            out[k] = d[k]
    if isinstance(d.get("population"), dict):
        out["population"] = d["population"]
    if isinstance(d.get("selection_alpha"), dict):
        # Avoid the full grid; keep metadata only.
        sel = dict(d["selection_alpha"])
        sel.pop("selection_alpha_grid", None)
        out["selection_alpha"] = sel
    if isinstance(d.get("pdet_model"), dict):
        out["pdet_model"] = d["pdet_model"]

    out["selection_mode"] = str(selection_mode)
    out["weight_mode"] = weight_mode
    out["det_model"] = det_model

    H0_grid = d.get("H0_grid")
    if isinstance(H0_grid, list) and H0_grid:
        try:
            h0 = [float(x) for x in H0_grid]
            h0 = [x for x in h0 if math.isfinite(x)]
            if len(h0) >= 2:
                out["H0_min"] = float(min(h0))
                out["H0_max"] = float(max(h0))
                out["H0_n"] = int(len(h0))
        except Exception:
            pass

    return out


def _row_from_gate2_json(
    *,
    d: dict[str, Any],
    out_dir: Path,
    json_path: Path,
    manifest_path: Path | None,
    manifest: dict[str, Any] | None,
    relative_to: Path | None,
) -> dict[str, Any] | None:
    if "H0_grid" not in d or "summary" not in d:
        return None

    fname = json_path.name
    selection_mode = "off" if fname.startswith("gr_h0_selection_off") else "on"

    selection_alpha = d.get("selection_alpha")
    selection_alpha = selection_alpha if isinstance(selection_alpha, dict) else {}
    weight_mode = selection_alpha.get("weight_mode") if selection_mode == "on" else None
    det_model = selection_alpha.get("det_model")

    pop = d.get("population")
    pop = pop if isinstance(pop, dict) else {}

    summary = d.get("summary")
    summary = summary if isinstance(summary, dict) else {}

    H0_grid = d.get("H0_grid")
    h0_min = h0_max = h0_step = None
    h0_n = None
    if isinstance(H0_grid, list) and H0_grid:
        try:
            h0 = [float(x) for x in H0_grid]
            h0 = [x for x in h0 if math.isfinite(x)]
            if len(h0) >= 2:
                h0_min = float(min(h0))
                h0_max = float(max(h0))
                h0_n = int(len(h0))
                h0_sorted = sorted(h0)
                h0_step = float(h0_sorted[1] - h0_sorted[0])
        except Exception:
            pass

    ts_utc, ts_iso = _infer_timestamp_utc(out_dir=out_dir, json_path=json_path, manifest=manifest)

    def _fmt_path(p: Path | None) -> str | None:
        if p is None:
            return None
        if relative_to is not None:
            try:
                return str(p.resolve().relative_to(relative_to.resolve()))
            except Exception:
                return str(p)
        return str(p)

    params_json = json.dumps(
        _compact_params(d=d, selection_mode=selection_mode, weight_mode=str(weight_mode) if weight_mode is not None else None, det_model=str(det_model) if det_model else None),
        sort_keys=True,
    )

    return {
        "timestamp_utc": ts_utc,
        "timestamp_iso": ts_iso,
        "out_dir": _fmt_path(out_dir),
        "json_path": _fmt_path(json_path),
        "manifest_path": _fmt_path(manifest_path),
        "selection_mode": selection_mode,
        "weight_mode": weight_mode,
        "det_model": det_model,
        "omega_m0": _safe_float(d.get("omega_m0")),
        "omega_k0": _safe_float(d.get("omega_k0")),
        "z_max": _safe_float(d.get("z_max")),
        "H0_min": h0_min,
        "H0_max": h0_max,
        "H0_n": h0_n,
        "H0_step": h0_step,
        "n_events": d.get("n_events"),
        "n_events_skipped": d.get("n_events_skipped"),
        "gate2_pass": bool(d.get("gate2_pass", False)),
        "H0_map": _safe_float(d.get("H0_map")),
        "H0_map_at_edge": bool(d.get("H0_map_at_edge", False)),
        "p16": _safe_float(summary.get("p16")),
        "p50": _safe_float(summary.get("p50")),
        "p84": _safe_float(summary.get("p84")),
        "mean": _safe_float(summary.get("mean")),
        "sd": _safe_float(summary.get("sd")),
        "pop_z_mode": pop.get("pop_z_mode"),
        "pop_z_k": _safe_float(pop.get("pop_z_k")),
        "pop_mass_mode": pop.get("pop_mass_mode"),
        "pop_m1_alpha": _safe_float(pop.get("pop_m1_alpha")),
        "pop_m_min": _safe_float(pop.get("pop_m_min")),
        "pop_m_max": _safe_float(pop.get("pop_m_max")),
        "pop_q_beta": _safe_float(pop.get("pop_q_beta")),
        "pop_m_taper_delta": _safe_float(pop.get("pop_m_taper_delta")),
        "pop_m_peak": _safe_float(pop.get("pop_m_peak")),
        "pop_m_peak_sigma": _safe_float(pop.get("pop_m_peak_sigma")),
        "pop_m_peak_frac": _safe_float(pop.get("pop_m_peak_frac")),
        "selection_include_h0_volume_scaling": bool(d.get("selection_include_h0_volume_scaling", False)),
        "pop_z_include_h0_volume_scaling": bool(d.get("pop_z_include_h0_volume_scaling", False)),
        "include_pdet_in_event_term": bool(d.get("include_pdet_in_event_term", False)),
        "importance_smoothing": d.get("importance_smoothing"),
        "importance_truncate_tau": _safe_float(d.get("importance_truncate_tau")),
        "event_qc_mode": d.get("event_qc_mode"),
        "event_min_finite_frac": _safe_float(d.get("event_min_finite_frac")),
        "params_json": params_json,
    }


def collect_gate2_log_rows(*, outputs_dir: str | Path, relative_to: str | Path | None = None) -> list[dict[str, Any]]:
    outputs_dir = Path(outputs_dir)
    rel_base = None if relative_to is None else Path(relative_to)

    rows: list[dict[str, Any]] = []

    patterns = [
        "**/json/gr_h0_selection_*.json",
        "**/gr_h0_selection_*.json",
    ]
    seen: set[Path] = set()
    for pat in patterns:
        for json_path in sorted(outputs_dir.glob(pat)):
            if json_path in seen:
                continue
            seen.add(json_path)
            try:
                obj = _read_json(json_path)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue

            # Default: outputs/.../<run>/json/<file>.json => out_dir is <run>
            out_dir = json_path.parent.parent if json_path.parent.name == "json" else json_path.parent
            manifest_path = _nearest_manifest_json(json_path.parent, stop_dir=outputs_dir)
            manifest = None if manifest_path is None else _read_json(manifest_path)
            manifest = manifest if isinstance(manifest, dict) else None

            row = _row_from_gate2_json(
                d=obj,
                out_dir=out_dir,
                json_path=json_path,
                manifest_path=manifest_path,
                manifest=manifest,
                relative_to=rel_base,
            )
            if row is not None:
                rows.append(row)

    return rows


def write_gate2_log_csv(*, rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_utc",
        "timestamp_iso",
        "out_dir",
        "json_path",
        "manifest_path",
        "selection_mode",
        "weight_mode",
        "det_model",
        "omega_m0",
        "omega_k0",
        "z_max",
        "H0_min",
        "H0_max",
        "H0_n",
        "H0_step",
        "n_events",
        "n_events_skipped",
        "gate2_pass",
        "H0_map",
        "H0_map_at_edge",
        "p16",
        "p50",
        "p84",
        "mean",
        "sd",
        "pop_z_mode",
        "pop_z_k",
        "pop_mass_mode",
        "pop_m1_alpha",
        "pop_m_min",
        "pop_m_max",
        "pop_q_beta",
        "pop_m_taper_delta",
        "pop_m_peak",
        "pop_m_peak_sigma",
        "pop_m_peak_frac",
        "selection_include_h0_volume_scaling",
        "pop_z_include_h0_volume_scaling",
        "include_pdet_in_event_term",
        "importance_smoothing",
        "importance_truncate_tau",
        "event_qc_mode",
        "event_min_finite_frac",
        "params_json",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


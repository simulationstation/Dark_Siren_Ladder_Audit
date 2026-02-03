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


def _winner_from_delta(delta: float | None) -> str:
    if delta is None:
        return "unknown"
    if delta > 0.0:
        return "mu"
    if delta < 0.0:
        return "gr"
    return "tie"


def _infer_kind(summary: dict[str, Any], *, summary_path: Path) -> str:
    if "pe_like_mode" in summary or "gw_data_mode" in summary:
        return "dark_siren_gap"
    if "convention" in summary and "n_events" in summary:
        return "siren_gap"
    if "delta_lpd_total" in summary:
        return "delta_lpd"
    return f"unknown:{summary_path.name}"


def _nearest_manifest_json(start_dir: Path, *, stop_dir: Path) -> Path | None:
    start_dir = start_dir.resolve()
    stop_dir = stop_dir.resolve()
    p = start_dir
    while True:
        cand = p / "manifest.json"
        if cand.exists():
            return cand
        if p == stop_dir or p.parent == p:
            return None
        p = p.parent


def _infer_timestamp_utc(*, out_dir: Path, summary_path: Path, manifest: dict[str, Any] | None) -> tuple[str | None, str | None]:
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
        mtime = summary_path.stat().st_mtime
        dt_iso = dt.datetime.fromtimestamp(mtime, tz=dt.timezone.utc).isoformat()
        return None, dt_iso
    except Exception:
        return None, None


def _compact_params(*, summary: dict[str, Any], manifest: dict[str, Any] | None) -> dict[str, Any]:
    # Keep this small and stable: select a handful of reproducibility-critical parameters.
    out: dict[str, Any] = {}

    if manifest:
        for k in (
            "run_dir",
            "skymap_dir",
            "gw_data_mode",
            "pe_like_mode",
            "hier_null_mode",
            "hier_null_seed",
            "hier_control_battery",
            "hier_min_good_frac",
            "hier_min_ess",
            "hier_bad_sample_mode",
            "pe_base_dir",
            "pe_record_id",
            "pe_prefer_variant",
            "pe_analysis",
            "pe_max_samples",
            "pe_seed",
            "max_events",
            "events_filter",
            "max_draws",
            "selection_injections_hdf",
            "selection_injections_spec",
            "selection_ifar_thresh_yr",
            "selection_z_max",
            "selection_det_model",
            "selection_snr_thresh",
            "selection_snr_binned_nbins",
            "selection_weight_mode",
            "selection_pop_z_mode",
            "selection_pop_z_k",
            "selection_pop_mass_mode",
            "selection_pop_m1_alpha",
            "selection_pop_m_min",
            "selection_pop_m_max",
            "selection_pop_q_beta",
        ):
            if k in manifest:
                out[k] = manifest[k]

    for k in (
        "convention",
        "gw_data_mode",
        "pe_like_mode",
        "mode",
        "hier_null_mode",
        "hier_null_seed",
        "hier_control_battery",
        "hier_min_good_frac",
        "hier_min_ess",
        "hier_bad_sample_mode",
        "z_max",
        "max_draws",
        "n_events",
        "n_draws",
    ):
        if k in summary:
            out[k] = summary[k]

    if isinstance(summary.get("selection"), dict):
        # Usually tiny; helpful for diagnosing whether ΔLPD is dominated by selection.
        out["selection"] = summary["selection"]

    return out


def _rows_from_summary_dict(
    *,
    summary: dict[str, Any],
    out_dir: Path,
    summary_path: Path,
    manifest_path: Path | None,
    manifest: dict[str, Any] | None,
    relative_to: Path | None,
) -> dict[str, Any] | None:
    if "delta_lpd_total" not in summary:
        return None

    delta_total = _safe_float(summary.get("delta_lpd_total"))
    lpd_mu_total = _safe_float(summary.get("lpd_mu_total"))
    lpd_gr_total = _safe_float(summary.get("lpd_gr_total"))
    delta_data = _safe_float(summary.get("delta_lpd_total_data"))
    delta_sel = _safe_float(summary.get("delta_lpd_total_sel"))

    ts_utc, ts_iso = _infer_timestamp_utc(out_dir=out_dir, summary_path=summary_path, manifest=manifest)

    def _fmt_path(p: Path | None) -> str | None:
        if p is None:
            return None
        if relative_to is not None:
            try:
                return str(p.resolve().relative_to(relative_to.resolve()))
            except Exception:
                return str(p)
        return str(p)

    n_events = summary.get("n_events")
    n_draws = summary.get("n_draws")
    if n_draws is None:
        n_draws = summary.get("max_draws")

    params_json = json.dumps(_compact_params(summary=summary, manifest=manifest), sort_keys=True)

    return {
        "timestamp_utc": ts_utc,
        "timestamp_iso": ts_iso,
        "kind": _infer_kind(summary, summary_path=summary_path),
        "out_dir": _fmt_path(out_dir),
        "summary_path": _fmt_path(summary_path),
        "manifest_path": _fmt_path(manifest_path),
        "run": summary.get("run"),
        "mode": summary.get("mode"),
        "convention": summary.get("convention"),
        "gw_data_mode": summary.get("gw_data_mode"),
        "pe_like_mode": summary.get("pe_like_mode"),
        "n_events": int(n_events) if isinstance(n_events, int) else n_events,
        "n_draws": int(n_draws) if isinstance(n_draws, int) else n_draws,
        "winner": _winner_from_delta(delta_total),
        "metric": "delta_lpd_total",
        "delta_lpd_total": delta_total,
        "delta_lpd_total_data": delta_data,
        "delta_lpd_total_sel": delta_sel,
        "lpd_mu_total": lpd_mu_total,
        "lpd_gr_total": lpd_gr_total,
        "params_json": params_json,
    }


def collect_run_log_rows(*, outputs_dir: str | Path, relative_to: str | Path | None = None) -> list[dict[str, Any]]:
    outputs_dir = Path(outputs_dir)
    rel_base = None if relative_to is None else Path(relative_to)

    rows: list[dict[str, Any]] = []

    # 1) Single-file summaries (common in dark siren gap test runners).
    for summary_path in sorted(outputs_dir.glob("**/summary_*.json")):
        try:
            obj = _read_json(summary_path)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        out_dir = summary_path.parent
        manifest_path = _nearest_manifest_json(out_dir, stop_dir=outputs_dir)
        manifest = None if manifest_path is None else _read_json(manifest_path)
        row = _rows_from_summary_dict(
            summary=obj,
            out_dir=out_dir,
            summary_path=summary_path,
            manifest_path=manifest_path,
            manifest=manifest if isinstance(manifest, dict) else None,
            relative_to=rel_base,
        )
        if row is not None:
            rows.append(row)

    # 2) Table summaries (common in siren gap runners).
    for summary_path in sorted(outputs_dir.glob("**/tables/summary.json")):
        try:
            obj = _read_json(summary_path)
        except Exception:
            continue

        # Here, the run directory is the tables/ parent.
        out_dir = summary_path.parent.parent
        manifest_path = _nearest_manifest_json(out_dir, stop_dir=outputs_dir)
        manifest = None if manifest_path is None else _read_json(manifest_path)
        manifest = manifest if isinstance(manifest, dict) else None

        if isinstance(obj, dict):
            row = _rows_from_summary_dict(
                summary=obj,
                out_dir=out_dir,
                summary_path=summary_path,
                manifest_path=manifest_path,
                manifest=manifest,
                relative_to=rel_base,
            )
            if row is not None:
                rows.append(row)
        elif isinstance(obj, list):
            for entry in obj:
                if not isinstance(entry, dict):
                    continue
                row = _rows_from_summary_dict(
                    summary=entry,
                    out_dir=out_dir,
                    summary_path=summary_path,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    relative_to=rel_base,
                )
                if row is not None:
                    rows.append(row)

    def _sort_key(r: dict[str, Any]) -> tuple[str, str, str]:
        ts = r.get("timestamp_iso") or ""
        outd = r.get("out_dir") or ""
        run = r.get("run") or ""
        return str(ts), str(outd), str(run)

    rows.sort(key=_sort_key)
    return rows


_RUN_LOG_FIELDS = [
    "timestamp_utc",
    "timestamp_iso",
    "kind",
    "out_dir",
    "summary_path",
    "manifest_path",
    "run",
    "mode",
    "convention",
    "gw_data_mode",
    "pe_like_mode",
    "n_events",
    "n_draws",
    "winner",
    "metric",
    "delta_lpd_total",
    "delta_lpd_total_data",
    "delta_lpd_total_sel",
    "lpd_mu_total",
    "lpd_gr_total",
    "params_json",
]


def write_run_log_csv(*, rows: list[dict[str, Any]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_RUN_LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _RUN_LOG_FIELDS})

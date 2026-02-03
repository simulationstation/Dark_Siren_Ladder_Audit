import json
from pathlib import Path

from entropy_horizon_recon.gate2_log import collect_gate2_log_rows


def _write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_collect_gate2_log_rows_parses_gate2_json(tmp_path: Path):
    outputs = tmp_path / "outputs"

    out_dir = outputs / "siren_gate2_gr_h0_20260101_000000UTC"
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    _write_json(json_dir / "manifest.json", {"timestamp_utc": "20260101_000000UTC", "note": "test"})

    base = {
        "method": "gr_h0_grid_hierarchical_pe",
        "prior": "uniform",
        "omega_m0": 0.31,
        "omega_k0": 0.0,
        "z_max": 0.62,
        "H0_grid": [40.0, 41.0, 42.0],
        "n_events": 3,
        "n_events_skipped": 0,
        "H0_map": 41.0,
        "H0_map_at_edge": False,
        "gate2_pass": True,
        "summary": {"p16": 40.5, "p50": 41.0, "p84": 41.5, "mean": 41.0, "sd": 0.2},
        "population": {"pop_z_mode": "none", "pop_mass_mode": "none", "pop_m_min": 5.0, "pop_m_max": 200.0},
        "selection_include_h0_volume_scaling": False,
        "pop_z_include_h0_volume_scaling": False,
        "importance_smoothing": "none",
        "importance_truncate_tau": None,
        "event_qc_mode": "skip",
        "event_min_finite_frac": 0.0,
    }
    on = dict(base)
    on["selection_alpha"] = {"det_model": "snr_mchirp_binned", "weight_mode": "inv_sampling_pdf"}
    _write_json(json_dir / "gr_h0_selection_on_inv_sampling_pdf.json", on)

    off = dict(base)
    off.pop("selection_alpha", None)
    _write_json(json_dir / "gr_h0_selection_off.json", off)

    rows = collect_gate2_log_rows(outputs_dir=outputs, relative_to=tmp_path)
    assert len(rows) == 2

    r_on = next(r for r in rows if r["selection_mode"] == "on")
    assert r_on["timestamp_utc"] == "20260101_000000UTC"
    assert r_on["det_model"] == "snr_mchirp_binned"
    assert r_on["weight_mode"] == "inv_sampling_pdf"
    assert abs(float(r_on["p50"]) - 41.0) < 1e-12

    r_off = next(r for r in rows if r["selection_mode"] == "off")
    assert r_off["timestamp_utc"] == "20260101_000000UTC"
    assert r_off["weight_mode"] is None


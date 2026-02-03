import json
from pathlib import Path

from entropy_horizon_recon.run_log import collect_run_log_rows


def _write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def test_collect_run_log_rows_parses_summary_and_tables(tmp_path: Path):
    outputs = tmp_path / "outputs"

    # A dark-siren style run: summary_*.json + manifest.json in the same output dir.
    ds = outputs / "dark_siren_gap_test_20260101_000000UTC"
    ds.mkdir(parents=True, exist_ok=True)
    _write_json(ds / "manifest.json", {"timestamp_utc": "20260101_000000UTC", "selection_det_model": "snr_mchirp_binned"})
    _write_json(
        ds / "summary_M0_start101.json",
        {
            "run": "M0_start101",
            "gw_data_mode": "pe",
            "pe_like_mode": "hierarchical",
            "n_events": 3,
            "n_draws": 128,
            "lpd_mu_total": 10.0,
            "lpd_gr_total": 8.0,
            "delta_lpd_total": 2.0,
        },
    )

    # A siren-gap style run: tables/summary.json (list-of-dicts).
    sg_tables = outputs / "siren_gap_20260102_000000UTC" / "tables"
    sg_tables.mkdir(parents=True, exist_ok=True)
    _write_json(
        sg_tables / "summary.json",
        [
            {
                "run": "toy",
                "convention": "A",
                "n_events": 1,
                "n_draws": 16,
                "lpd_mu_total": -1.0,
                "lpd_gr_total": 0.0,
                "delta_lpd_total": -1.0,
            }
        ],
    )

    rows = collect_run_log_rows(outputs_dir=outputs, relative_to=tmp_path)
    assert len(rows) == 2

    r_ds = next(r for r in rows if r["run"] == "M0_start101")
    assert r_ds["kind"] == "dark_siren_gap"
    assert r_ds["timestamp_utc"] == "20260101_000000UTC"
    assert r_ds["winner"] == "mu"
    assert abs(float(r_ds["delta_lpd_total"]) - 2.0) < 1e-12
    assert "selection_det_model" in json.loads(r_ds["params_json"])

    r_sg = next(r for r in rows if r["run"] == "toy")
    assert r_sg["kind"] == "siren_gap"
    assert r_sg["timestamp_utc"] == "20260102_000000UTC"
    assert r_sg["winner"] == "gr"
    assert abs(float(r_sg["delta_lpd_total"]) - (-1.0)) < 1e-12


from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from entropy_horizon_recon.lvk_population import LvkBbhPowerlawPeakRedshiftDraw
from entropy_horizon_recon.lvk_population import draw_to_gate2_population_kwargs
from entropy_horizon_recon.lvk_population import load_bilby_result_posterior_columns
from entropy_horizon_recon.lvk_population import sample_lvk_bbh_powerlaw_peak_redshift_draws


def _write_bilby_result_json(path: Path, *, posterior_content: dict[str, list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "label": "test",
                "posterior": {"__dataframe__": True, "content": posterior_content},
            }
        )
        + "\n"
    )


def test_load_bilby_result_posterior_columns(tmp_path: Path) -> None:
    tmp = tmp_path / "bilby_result.json"
    _write_bilby_result_json(tmp, posterior_content={"alpha": [1.0, 2.0, 3.0], "beta": [0.1, 0.2, 0.3]})
    cols = load_bilby_result_posterior_columns(tmp)
    assert set(cols.keys()) == {"alpha", "beta"}
    assert np.allclose(cols["alpha"], np.asarray([1.0, 2.0, 3.0]))
    assert np.allclose(cols["beta"], np.asarray([0.1, 0.2, 0.3]))


def test_sample_lvk_draws_requires_columns(tmp_path: Path) -> None:
    tmp = tmp_path / "bilby_result_missing.json"
    _write_bilby_result_json(tmp, posterior_content={"alpha": [1.0, 2.0], "beta": [0.0, 0.1]})
    with pytest.raises(ValueError, match="missing required posterior columns"):
        sample_lvk_bbh_powerlaw_peak_redshift_draws(tmp, n_draws=1)


def test_sample_lvk_draws_deterministic(tmp_path: Path) -> None:
    tmp = tmp_path / "bilby_result_full.json"
    n = 10
    content = {
        "alpha": [float(i) for i in range(n)],
        "beta": [float(i + 0.1) for i in range(n)],
        "mmin": [5.0] * n,
        "mmax": [80.0] * n,
        "lam": [0.05] * n,
        "mpp": [35.0] * n,
        "sigpp": [4.0] * n,
        "delta_m": [3.0] * n,
        "lamb": [2.0] * n,
    }
    _write_bilby_result_json(tmp, posterior_content=content)
    d1 = sample_lvk_bbh_powerlaw_peak_redshift_draws(tmp, n_draws=5, seed=123)
    d2 = sample_lvk_bbh_powerlaw_peak_redshift_draws(tmp, n_draws=5, seed=123)
    assert d1 == d2
    assert len(d1) == 5


def test_draw_to_gate2_population_kwargs() -> None:
    draw = LvkBbhPowerlawPeakRedshiftDraw(
        alpha=2.3,
        beta=0.0,
        mmin=5.0,
        mmax=90.0,
        lam=0.03,
        mpp=34.0,
        sigpp=4.5,
        delta_m=3.0,
        lamb=2.9,
    )
    kw = draw_to_gate2_population_kwargs(draw)
    assert kw["pop_z_mode"] == "comoving_powerlaw"
    assert kw["pop_mass_mode"] == "powerlaw_peak_q_smooth"
    assert kw["pop_z_powerlaw_k"] == pytest.approx(2.9)
    assert kw["pop_m1_alpha"] == pytest.approx(2.3)

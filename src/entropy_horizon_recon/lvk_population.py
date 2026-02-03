from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def load_bilby_result_posterior_columns(result_json: str | Path) -> dict[str, np.ndarray]:
    """Load posterior samples from a bilby-result JSON.

    The LVK population data release provides bilby result JSONs with:
      obj["posterior"]["__dataframe__"] == True
      obj["posterior"]["content"] == {column_name: [samples...], ...}
    """
    path = Path(result_json).expanduser().resolve()
    obj = json.loads(path.read_text(encoding="utf-8"))

    posterior = obj.get("posterior")
    if not isinstance(posterior, dict):
        raise ValueError(f"{path}: missing 'posterior' dict in bilby result JSON.")
    if not bool(posterior.get("__dataframe__", False)):
        raise ValueError(f"{path}: expected posterior['__dataframe__']=True in bilby result JSON.")

    content = posterior.get("content")
    if not isinstance(content, dict):
        raise ValueError(f"{path}: expected posterior['content'] dict in bilby result JSON.")

    out: dict[str, np.ndarray] = {}
    n: int | None = None
    for k, v in content.items():
        if not isinstance(k, str):
            continue
        if not isinstance(v, list):
            continue
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            continue
        if np.any(~np.isfinite(arr)):
            raise ValueError(f"{path}: posterior column '{k}' contains non-finite values.")
        if n is None:
            n = int(arr.size)
        elif int(arr.size) != n:
            # Bilby should produce consistent-length columns.
            raise ValueError(f"{path}: posterior columns have inconsistent lengths (e.g. '{k}' has n={arr.size}, expected {n}).")
        out[k] = arr

    if not out:
        raise ValueError(f"{path}: no usable posterior columns found in bilby result JSON.")
    return out


@dataclass(frozen=True)
class LvkBbhPowerlawPeakRedshiftDraw:
    """One hyperposterior draw matching our Gate-2 BBH mass+z knobs."""

    alpha: float
    beta: float
    mmin: float
    mmax: float
    lam: float
    mpp: float
    sigpp: float
    delta_m: float
    lamb: float

    @classmethod
    def from_columns(cls, columns: dict[str, np.ndarray], idx: int) -> "LvkBbhPowerlawPeakRedshiftDraw":
        def _get(key: str) -> float:
            if key not in columns:
                raise KeyError(key)
            arr = columns[key]
            if idx < 0 or idx >= int(arr.size):
                raise IndexError(idx)
            return float(arr[int(idx)])

        return cls(
            alpha=_get("alpha"),
            beta=_get("beta"),
            mmin=_get("mmin"),
            mmax=_get("mmax"),
            lam=_get("lam"),
            mpp=_get("mpp"),
            sigpp=_get("sigpp"),
            delta_m=_get("delta_m"),
            lamb=_get("lamb"),
        )


def sample_lvk_bbh_powerlaw_peak_redshift_draws(
    result_json: str | Path,
    *,
    n_draws: int,
    seed: int = 0,
    replace: bool | None = None,
) -> list[LvkBbhPowerlawPeakRedshiftDraw]:
    """Sample LVK BBH population hyperposterior draws from a bilby result JSON.

    Intended for the GWTC-3 O3a population data release JSONs, e.g.
      Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json
    """
    if int(n_draws) <= 0:
        raise ValueError("n_draws must be > 0.")

    columns = load_bilby_result_posterior_columns(result_json)
    required = {"alpha", "beta", "mmin", "mmax", "lam", "mpp", "sigpp", "delta_m", "lamb"}
    missing = sorted(required - set(columns.keys()))
    if missing:
        raise ValueError(f"{Path(result_json)}: missing required posterior columns: {missing}")

    n = int(columns["alpha"].size)
    if n <= 0:
        raise ValueError(f"{Path(result_json)}: empty posterior for 'alpha'.")

    if replace is None:
        replace = bool(int(n_draws) > n)

    rng = np.random.default_rng(int(seed))
    idxs = rng.choice(n, size=int(n_draws), replace=bool(replace))
    return [LvkBbhPowerlawPeakRedshiftDraw.from_columns(columns, int(i)) for i in idxs.tolist()]


def draw_to_gate2_population_kwargs(draw: LvkBbhPowerlawPeakRedshiftDraw) -> dict[str, Any]:
    """Map an LVK draw onto our Gate-2 population parameters (kwargs for compute_gr_h0_posterior_grid_hierarchical_pe)."""
    return {
        "pop_z_mode": "comoving_powerlaw",
        "pop_z_powerlaw_k": float(draw.lamb),
        "pop_mass_mode": "powerlaw_peak_q_smooth",
        "pop_m1_alpha": float(draw.alpha),
        "pop_m_min": float(draw.mmin),
        "pop_m_max": float(draw.mmax),
        "pop_q_beta": float(draw.beta),
        "pop_m_taper_delta": float(draw.delta_m),
        "pop_m_peak": float(draw.mpp),
        "pop_m_peak_sigma": float(draw.sigpp),
        "pop_m_peak_frac": float(draw.lam),
    }

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianPrior:
    """Simple normal prior on H0 for “tension report” style comparisons."""

    name: str
    mean: float
    sigma: float

    def pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        sig = float(self.sigma)
        if not (math.isfinite(sig) and sig > 0.0):
            raise ValueError("sigma must be finite and positive.")
        mu = float(self.mean)
        z = (x - mu) / sig
        return np.exp(-0.5 * z**2) / (sig * math.sqrt(2.0 * math.pi))


def as_1d_finite_array(obj: object, *, name: str) -> np.ndarray:
    arr = np.asarray(obj, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a 1D array with >=2 entries.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def normalize_discrete_pdf(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if np.any(p < 0.0):
        raise ValueError("posterior contains negative values.")
    s = float(np.sum(p))
    if not (math.isfinite(s) and s > 0.0):
        raise ValueError("posterior sum is non-positive.")
    return p / s


def normalize_pdf_grid(H0_grid: np.ndarray, posterior: np.ndarray) -> np.ndarray:
    H0 = as_1d_finite_array(H0_grid, name="H0_grid")
    p = as_1d_finite_array(posterior, name="posterior")
    if H0.shape != p.shape:
        raise ValueError("H0_grid and posterior must have matching shapes.")
    return normalize_discrete_pdf(p)


def posterior_quantiles(H0_grid: np.ndarray, posterior: np.ndarray) -> dict[str, float]:
    H0 = as_1d_finite_array(H0_grid, name="H0_grid")
    p = normalize_pdf_grid(H0, posterior)
    cdf = np.cumsum(p)
    mean = float(np.sum(p * H0))
    sd = float(np.sqrt(np.sum(p * (H0 - mean) ** 2)))
    i_map = int(np.argmax(p))
    return {
        "H0_map": float(H0[i_map]),
        "mean": mean,
        "sd": sd,
        "p16": float(np.interp(0.16, cdf, H0)),
        "p50": float(np.interp(0.50, cdf, H0)),
        "p84": float(np.interp(0.84, cdf, H0)),
    }


def integrate_posterior_prob(
    H0_grid: np.ndarray,
    posterior: np.ndarray,
    *,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    H0 = as_1d_finite_array(H0_grid, name="H0_grid")
    p = normalize_pdf_grid(H0, posterior)
    m = np.ones_like(H0, dtype=bool)
    if lo is not None:
        m &= H0 >= float(lo)
    if hi is not None:
        m &= H0 <= float(hi)
    return float(np.sum(p[m]))


def bayes_factor_between_priors_from_uniform_posterior(
    H0_grid: np.ndarray,
    posterior_under_uniform_prior: np.ndarray,
    *,
    prior_a: GaussianPrior,
    prior_b: GaussianPrior,
) -> dict[str, float]:
    """Compute BF = Z(prior_a)/Z(prior_b) from a posterior produced under a uniform prior.

    If the posterior is produced under a uniform prior on the same grid, then:
      BF(A,B) = ∫ p_unif(H0|d) πA(H0) dH0 / ∫ p_unif(H0|d) πB(H0) dH0,
    since the uniform prior density cancels in the ratio.
    """
    H0 = as_1d_finite_array(H0_grid, name="H0_grid")
    p = normalize_pdf_grid(H0, posterior_under_uniform_prior)
    a = prior_a.pdf(H0)
    b = prior_b.pdf(H0)
    za = float(np.sum(p * a))
    zb = float(np.sum(p * b))
    if not (math.isfinite(za) and math.isfinite(zb) and za > 0.0 and zb > 0.0):
        raise ValueError("Invalid prior-weighted integrals; check prior parameters and grid coverage.")
    bf = za / zb
    return {"bf": bf, "log_bf": float(np.log(bf)), "za": za, "zb": zb}


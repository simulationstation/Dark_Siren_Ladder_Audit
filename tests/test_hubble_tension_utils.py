import numpy as np

from entropy_horizon_recon.hubble_tension import GaussianPrior
from entropy_horizon_recon.hubble_tension import bayes_factor_between_priors_from_uniform_posterior
from entropy_horizon_recon.hubble_tension import integrate_posterior_prob
from entropy_horizon_recon.hubble_tension import posterior_quantiles


def test_posterior_quantiles_smoke() -> None:
    H0 = np.linspace(50.0, 100.0, 501)
    # A narrow-ish posterior peaked at 70.
    p = np.exp(-0.5 * ((H0 - 70.0) / 2.0) ** 2)
    s = posterior_quantiles(H0, p)
    assert 65.0 < s["p50"] < 75.0
    assert s["p16"] < s["p50"] < s["p84"]
    assert 60.0 < s["H0_map"] < 80.0


def test_integrate_prob_bounds() -> None:
    H0 = np.linspace(0.0, 10.0, 11)
    p = np.ones_like(H0)
    p_lo = integrate_posterior_prob(H0, p, hi=5.0)
    p_hi = integrate_posterior_prob(H0, p, lo=6.0)
    assert np.isclose(p_lo + p_hi, 1.0)


def test_bayes_factor_prefers_closer_prior() -> None:
    H0 = np.linspace(60.0, 80.0, 401)
    p = np.exp(-0.5 * ((H0 - 67.4) / 0.8) ** 2)  # posterior near "Planck-like"
    planck = GaussianPrior(name="planck", mean=67.4, sigma=0.5)
    shoes = GaussianPrior(name="shoes", mean=73.0, sigma=1.0)
    bf = bayes_factor_between_priors_from_uniform_posterior(H0, p, prior_a=planck, prior_b=shoes)
    assert bf["bf"] > 1.0


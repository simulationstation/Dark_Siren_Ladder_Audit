import numpy as np
from scipy.special import ndtr

from entropy_horizon_recon.siren_injection_recovery import _sample_log_dL_obs_conditional_on_detection


def test_pe_obs_conditioned_on_detection_enforces_cdf_cut() -> None:
    rng = np.random.default_rng(12345)
    log_dL_true = float(np.log(1000.0))
    sigma = 0.4

    # With p_det<1, samples should be truncated to the lower tail of the Normal noise CDF.
    p_det = 0.2
    log_draws = np.asarray(
        [
            _sample_log_dL_obs_conditional_on_detection(
                rng,
                log_dL_true=log_dL_true,
                sigma_log_dL=sigma,
                p_det_true=p_det,
            )
            for _ in range(2000)
        ],
        dtype=float,
    )
    eps = (log_draws - log_dL_true) / float(sigma)
    u = ndtr(eps)
    assert float(np.max(u)) <= float(p_det) + 1e-12
    assert float(np.mean(eps)) < 0.0

    # With p_det=1, the distribution should be (approximately) untruncated.
    rng = np.random.default_rng(12345)
    log_draws_u = np.asarray(
        [
            _sample_log_dL_obs_conditional_on_detection(
                rng,
                log_dL_true=log_dL_true,
                sigma_log_dL=sigma,
                p_det_true=1.0,
            )
            for _ in range(4000)
        ],
        dtype=float,
    )
    eps_u = (log_draws_u - log_dL_true) / float(sigma)
    assert abs(float(np.mean(eps_u))) < 0.05

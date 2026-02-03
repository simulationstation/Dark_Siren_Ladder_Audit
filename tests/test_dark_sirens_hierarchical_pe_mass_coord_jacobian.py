from __future__ import annotations

import numpy as np

from entropy_horizon_recon.dark_sirens_hierarchical_pe import (
    GWTCPeHierarchicalSamples,
    compute_hierarchical_pe_logL_draws,
)
from entropy_horizon_recon.sirens import MuForwardPosterior, predict_dL_em


def test_hierarchical_pe_does_not_use_mass_coord_jacobian_when_pop_mass_none() -> None:
    # Regression test:
    # When pop_mass_mode="none", the hierarchical PE reweighting must not depend on the PE
    # mass samples (chirp_mass_det, q) or any mass-coordinate Jacobians.
    #
    # Previously, a mass-coordinate Jacobian was applied unconditionally, which made logL
    # depend on q even when no mass population model was requested.

    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((1, x_grid.size), dtype=float)  # 1 draw, mu=1
    z_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    H = np.array([[70.0, 72.0, 75.0, 78.0]], dtype=float)
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.array([70.0], dtype=float),
        omega_m0=np.array([0.3], dtype=float),
        omega_k0=np.zeros((1,), dtype=float),
        sigma8_0=None,
    )

    rng = np.random.default_rng(0)
    z_s = rng.uniform(0.051, 0.149, size=2000)
    dL_s = np.asarray(predict_dL_em(post, z_eval=z_s)[0], dtype=float)

    base_kwargs = dict(
        file="<mem>",
        analysis="TEST",
        n_total=int(dL_s.size),
        n_used=int(dL_s.size),
        dL_mpc=dL_s,
        log_pi_dL=np.zeros((dL_s.size,), dtype=float),
        log_pi_chirp_mass=np.zeros((dL_s.size,), dtype=float),
        log_pi_mass_ratio=np.zeros((dL_s.size,), dtype=float),
        prior_spec={},
    )

    pe_low_q = GWTCPeHierarchicalSamples(
        chirp_mass_det=np.full((dL_s.size,), 30.0, dtype=float),
        mass_ratio=np.full((dL_s.size,), 0.2, dtype=float),
        **base_kwargs,
    )
    pe_high_q = GWTCPeHierarchicalSamples(
        chirp_mass_det=np.full((dL_s.size,), 30.0, dtype=float),
        mass_ratio=np.full((dL_s.size,), 0.95, dtype=float),
        **base_kwargs,
    )

    logL_mu1, logL_gr1 = compute_hierarchical_pe_logL_draws(
        pe=pe_low_q,
        post=post,
        z_max=0.15,
        pop_z_mode="none",
        pop_mass_mode="none",
    )
    logL_mu2, logL_gr2 = compute_hierarchical_pe_logL_draws(
        pe=pe_high_q,
        post=post,
        z_max=0.15,
        pop_z_mode="none",
        pop_mass_mode="none",
    )

    assert np.all(np.isfinite(logL_mu1)) and np.all(np.isfinite(logL_mu2))
    assert np.all(np.isfinite(logL_gr1)) and np.all(np.isfinite(logL_gr2))
    assert np.allclose(logL_gr1, logL_gr2, rtol=0.0, atol=1e-12)
    assert np.allclose(logL_mu1, logL_mu2, rtol=0.0, atol=1e-12)


from __future__ import annotations

import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache, _event_logL_h0_grid_from_hierarchical_pe_samples
from entropy_horizon_recon.dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples


def test_event_logl_tail_completion_avoids_minus_inf_when_no_good_samples() -> None:
    """Regression test for Gate-2 undercoverage failure mode.

    When z_max is used as a hard cutoff, high-H0 grid points can have *zero* mapped PE samples with
    z<=z_max, yielding logL=-inf from naive Monte Carlo. We now tail-complete those grid points so
    logL stays finite (though very small).
    """
    rng = np.random.default_rng(0)
    constants = PhysicalConstants()

    omega_m0 = 0.31
    omega_k0 = 0.0
    z_max = 0.25

    dist_cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=omega_m0, omega_k0=omega_k0)
    f_b = float(dist_cache.f_grid[-1])

    H0_lo = 60.0
    H0_hi = 100.0
    dL_cut_lo = (constants.c_km_s / H0_lo) * f_b
    dL_cut_hi = (constants.c_km_s / H0_hi) * f_b
    assert dL_cut_lo > dL_cut_hi

    # Construct PE samples that are entirely above the high-H0 cutoff but below the low-H0 cutoff,
    # forcing n_good=0 at H0_hi while leaving support at H0_lo.
    dL_min = 1.02 * dL_cut_hi
    dL_max = 0.98 * dL_cut_lo
    assert dL_min < dL_max
    dL_s = rng.uniform(dL_min, dL_max, size=5000).astype(float)

    pe = GWTCPeHierarchicalSamples(
        file="<mem>",
        analysis="TEST",
        n_total=int(dL_s.size),
        n_used=int(dL_s.size),
        dL_mpc=np.asarray(dL_s, dtype=float),
        chirp_mass_det=np.full((dL_s.size,), 30.0, dtype=float),
        mass_ratio=np.full((dL_s.size,), 0.8, dtype=float),
        log_pi_dL=np.zeros((dL_s.size,), dtype=float),
        log_pi_chirp_mass=np.zeros((dL_s.size,), dtype=float),
        log_pi_mass_ratio=np.zeros((dL_s.size,), dtype=float),
        prior_spec={},
    )

    H0_grid = np.array([H0_lo, 70.0, 80.0, 90.0, H0_hi], dtype=float)
    logL, ess, n_good = _event_logL_h0_grid_from_hierarchical_pe_samples(
        pe=pe,
        H0_grid=H0_grid,
        dist_cache=dist_cache,
        constants=constants,
        z_max=z_max,
        det_model=None,
        include_pdet_in_event_term=False,
        pop_z_include_h0_volume_scaling=False,
        pop_z_mode="none",
        pop_z_k=0.0,
        pop_mass_mode="none",
        pop_m1_alpha=2.3,
        pop_m_min=5.0,
        pop_m_max=80.0,
        pop_q_beta=0.0,
        pop_m_taper_delta=3.0,
        pop_m_peak=35.0,
        pop_m_peak_sigma=5.0,
        pop_m_peak_frac=0.1,
        importance_smoothing="none",
        importance_truncate_tau=None,
        mc_logZ_bias_correction=False,
        return_diagnostics=True,
    )

    # No -inf at high-H0 endpoint after tail completion.
    assert np.isfinite(float(logL[-1]))
    assert float(n_good[-1]) >= 0.0
    assert float(ess[-1]) >= 0.0


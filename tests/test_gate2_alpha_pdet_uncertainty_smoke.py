import numpy as np

from entropy_horizon_recon.dark_siren_h0 import compute_alpha_h0_grid_pdet_marginalized
from entropy_horizon_recon.dark_sirens_selection import O3InjectionSet


def test_alpha_h0_grid_pdet_marginalized_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 500
    z = rng.uniform(0.01, 0.6, size=n)
    dL = 1_000.0 * (1.0 + z)
    snr = rng.lognormal(mean=2.0, sigma=0.5, size=n)
    found = snr > 7.5
    m1 = rng.uniform(5.0, 80.0, size=n)
    m2 = rng.uniform(5.0, 80.0, size=n)
    m2 = np.minimum(m1, m2)

    inj = O3InjectionSet(
        path="<synthetic>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=m1,
        m2_source=m2,
        total_generated=n,
        analysis_time_s=1.0,
    )

    H0_grid = np.array([60.0, 70.0, 80.0], dtype=float)
    meta, draws = compute_alpha_h0_grid_pdet_marginalized(
        injections=inj,
        H0_grid=H0_grid,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        det_model="snr_mchirp_binned",
        snr_binned_nbins=40,
        mchirp_binned_nbins=8,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
        n_pdet_draws=25,
        pdet_pseudocount=1.0,
        pdet_draw_seed=0,
    )

    assert meta["method"] == "alpha_h0_grid_pdet_marginalized"
    assert len(meta["H0_grid"]) == H0_grid.size
    assert draws.shape == (25, H0_grid.size)
    assert np.all(np.isfinite(draws))
    assert np.all(draws > 0.0)
    assert np.all(draws <= 1.0)


def test_alpha_h0_grid_pdet_marginalized_q_binned_smoke() -> None:
    rng = np.random.default_rng(1)
    n = 800
    z = rng.uniform(0.01, 0.6, size=n)
    dL = 1_000.0 * (1.0 + z)
    snr = rng.lognormal(mean=2.0, sigma=0.6, size=n)
    found = snr > 7.5
    m1 = rng.uniform(5.0, 80.0, size=n)
    q = rng.uniform(0.05, 1.0, size=n)
    m2 = m1 * q

    inj = O3InjectionSet(
        path="<synthetic>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=m1,
        m2_source=m2,
        total_generated=n,
        analysis_time_s=1.0,
    )

    H0_grid = np.array([60.0, 70.0, 80.0], dtype=float)
    meta, draws = compute_alpha_h0_grid_pdet_marginalized(
        injections=inj,
        H0_grid=H0_grid,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        det_model="snr_mchirp_q_binned",
        snr_binned_nbins=40,
        mchirp_binned_nbins=8,
        q_binned_nbins=6,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
        n_pdet_draws=20,
        pdet_pseudocount=1.0,
        pdet_draw_seed=0,
    )

    assert meta["method"] == "alpha_h0_grid_pdet_marginalized"
    assert meta["det_model"] == "snr_mchirp_q_binned"
    assert draws.shape == (20, H0_grid.size)
    assert np.all(np.isfinite(draws))
    assert np.all(draws > 0.0)
    assert np.all(draws <= 1.0)

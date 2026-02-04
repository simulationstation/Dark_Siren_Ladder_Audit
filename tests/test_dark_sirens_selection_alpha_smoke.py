from __future__ import annotations

import numpy as np

from entropy_horizon_recon.dark_sirens_selection import O3InjectionSet, compute_selection_alpha_from_injections
from entropy_horizon_recon.sirens import MuForwardPosterior


def test_selection_alpha_mu_equals_gr_when_mu_is_one() -> None:
    # Build a tiny MuForwardPosterior with mu=1 so dL_GW == dL_EM.
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((2, x_grid.size), dtype=float)  # 2 draws, mu=1
    z_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    H = np.tile(np.array([70.0, 72.0, 75.0, 78.0], dtype=float), (2, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((2,), 70.0),
        omega_m0=np.full((2,), 0.3),
        omega_k0=np.zeros((2,)),
        sigma8_0=None,
    )

    rng = np.random.default_rng(0)
    n = 200
    z = rng.uniform(0.02, 0.14, size=n)
    dL_fid = np.full((n,), 1000.0, dtype=float)
    snr = rng.uniform(5.0, 50.0, size=n)
    found = snr > 12.0

    inj = O3InjectionSet(
        path="<mem>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL_fid,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=np.full_like(z, 30.0),
        m2_source=np.full_like(z, 20.0),
        total_generated=int(n),
        analysis_time_s=1.0,
    )

    alpha = compute_selection_alpha_from_injections(
        post=post,
        injections=inj,
        convention="A",
        z_max=0.15,
        det_model="threshold",
        snr_threshold=12.0,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
    )

    assert alpha.alpha_mu.shape == alpha.alpha_gr.shape
    assert np.all(np.isfinite(alpha.alpha_mu))
    assert np.all(np.isfinite(alpha.alpha_gr))
    assert np.allclose(alpha.alpha_mu, alpha.alpha_gr, rtol=0.0, atol=0.0)


def test_selection_alpha_snr_mchirp_binned_smoke() -> None:
    # Same invariance check under a 2D detectability proxy: if mu=1 then dL_GW==dL_EM, so
    # the SNR rescaling is identical and alpha_mu must equal alpha_gr.
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((2, x_grid.size), dtype=float)  # 2 draws, mu=1
    z_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    H = np.tile(np.array([70.0, 72.0, 75.0, 78.0], dtype=float), (2, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((2,), 70.0),
        omega_m0=np.full((2,), 0.3),
        omega_k0=np.zeros((2,)),
        sigma8_0=None,
    )

    rng = np.random.default_rng(1)
    n = 500
    z = rng.uniform(0.02, 0.14, size=n)
    dL_fid = np.full((n,), 1000.0, dtype=float)
    snr = rng.uniform(5.0, 50.0, size=n)
    found = snr > 12.0

    m1 = rng.uniform(10.0, 80.0, size=n)
    q = rng.uniform(0.1, 1.0, size=n)
    m2 = m1 * q
    # Enforce m2<=m1 (should already hold), and keep within a sane range.
    m2 = np.minimum(m2, m1)

    inj = O3InjectionSet(
        path="<mem>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL_fid,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=np.asarray(m1, dtype=float),
        m2_source=np.asarray(m2, dtype=float),
        total_generated=int(n),
        analysis_time_s=1.0,
    )

    alpha = compute_selection_alpha_from_injections(
        post=post,
        injections=inj,
        convention="A",
        z_max=0.15,
        det_model="snr_mchirp_binned",
        snr_binned_nbins=30,
        mchirp_binned_nbins=6,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
    )

    assert alpha.alpha_mu.shape == alpha.alpha_gr.shape
    assert np.all(np.isfinite(alpha.alpha_mu))
    assert np.all(np.isfinite(alpha.alpha_gr))
    assert np.allclose(alpha.alpha_mu, alpha.alpha_gr, rtol=0.0, atol=0.0)


def test_selection_alpha_snr_mchirp_q_binned_smoke() -> None:
    # Same invariance check under a 3D detectability proxy: if mu=1 then dL_GW==dL_EM, so
    # the SNR rescaling is identical and alpha_mu must equal alpha_gr.
    x_grid = np.array([-1.0, 0.0], dtype=float)
    logmu = np.zeros((2, x_grid.size), dtype=float)  # 2 draws, mu=1
    z_grid = np.array([0.0, 0.05, 0.10, 0.15], dtype=float)
    H = np.tile(np.array([70.0, 72.0, 75.0, 78.0], dtype=float), (2, 1))
    post = MuForwardPosterior(
        x_grid=x_grid,
        logmu_x_samples=logmu,
        z_grid=z_grid,
        H_samples=H,
        H0=np.full((2,), 70.0),
        omega_m0=np.full((2,), 0.3),
        omega_k0=np.zeros((2,)),
        sigma8_0=None,
    )

    rng = np.random.default_rng(2)
    n = 800
    z = rng.uniform(0.02, 0.14, size=n)
    dL_fid = np.full((n,), 1000.0, dtype=float)
    snr = rng.uniform(5.0, 50.0, size=n)
    found = snr > 12.0

    m1 = rng.uniform(10.0, 80.0, size=n)
    q = rng.uniform(0.1, 1.0, size=n)
    m2 = m1 * q
    m2 = np.minimum(m2, m1)

    inj = O3InjectionSet(
        path="<mem>",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL_fid,
        snr_net_opt=snr,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=np.asarray(m1, dtype=float),
        m2_source=np.asarray(m2, dtype=float),
        total_generated=int(n),
        analysis_time_s=1.0,
    )

    alpha = compute_selection_alpha_from_injections(
        post=post,
        injections=inj,
        convention="A",
        z_max=0.15,
        det_model="snr_mchirp_q_binned",
        snr_binned_nbins=30,
        mchirp_binned_nbins=6,
        q_binned_nbins=6,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
    )

    assert alpha.alpha_mu.shape == alpha.alpha_gr.shape
    assert np.all(np.isfinite(alpha.alpha_mu))
    assert np.all(np.isfinite(alpha.alpha_gr))
    assert np.allclose(alpha.alpha_mu, alpha.alpha_gr, rtol=0.0, atol=0.0)

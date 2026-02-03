import numpy as np

from entropy_horizon_recon.dark_sirens_selection import O3InjectionSet
from entropy_horizon_recon.siren_injection_recovery import (
    InjectionRecoveryConfig,
    generate_synthetic_detected_events_from_injections,
)


def test_generate_synthetic_detected_events_from_injections_smoke() -> None:
    rng = np.random.default_rng(0)
    n = 500

    z = rng.uniform(0.01, 0.5, size=n)
    dL_fid = 6000.0 * z

    snr_fid = rng.lognormal(mean=2.5, sigma=0.35, size=n)
    found = snr_fid > 10.0

    m1 = rng.uniform(10.0, 60.0, size=n)
    q = rng.uniform(0.2, 1.0, size=n)
    m2 = m1 * q

    injections = O3InjectionSet(
        path=":memory:",
        ifar_threshold_yr=1.0,
        z=z,
        dL_mpc_fid=dL_fid,
        snr_net_opt=snr_fid,
        found_ifar=found,
        sampling_pdf=np.ones_like(z),
        mixture_weight=np.ones_like(z),
        m1_source=m1,
        m2_source=m2,
        total_generated=int(n),
        analysis_time_s=1.0,
    )

    cfg = InjectionRecoveryConfig(
        h0_true=70.0,
        omega_m0=0.31,
        omega_k0=0.0,
        z_max=0.62,
        det_model="snr_binned",
        snr_binned_nbins=50,
        weight_mode="none",
        pop_z_mode="none",
        pop_mass_mode="none",
        pe_obs_mode="truth",
    )

    truths = generate_synthetic_detected_events_from_injections(injections=injections, cfg=cfg, n_events=7, seed=123)
    assert len(truths) == 7
    assert all(np.isfinite(t.p_det_true) and 0.0 <= t.p_det_true <= 1.0 for t in truths)

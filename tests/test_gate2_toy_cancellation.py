from __future__ import annotations

import numpy as np

from entropy_horizon_recon.gate2_toy import ToyCancellationConfig, ToyPopulationConfig, run_gate2_toy_cancellation


def test_gate2_toy_cancellation_is_flatish() -> None:
    cfg = ToyCancellationConfig(
        pop=ToyPopulationConfig(
            omega_m0=0.31,
            omega_k0=0.0,
            z_max=0.5,
            pop_z_mode="comoving_uniform",
            pop_z_k=0.0,
        ),
        n_events=1,
        pe_n_samples=4000,
        seed=0,
        h0_min=50.0,
        h0_max=120.0,
        h0_n=71,
        snr_threshold=8.0,
        snr_norm=25_000.0,
        dL_ref_mpc=1000.0,
        alpha_mc_samples=50_000,
    )
    out = run_gate2_toy_cancellation(cfg, out_dir=None, n_processes=1)

    logL_total_rel = np.asarray(out["logL_total_rel"], dtype=float)
    assert logL_total_rel.ndim == 1
    assert np.all(np.isfinite(logL_total_rel))

    # This should be close to flat; allow a small tolerance for MC noise.
    flat_range = float(out["diagnostics"]["flat_range_logL"])
    assert np.isfinite(flat_range)
    assert flat_range < 0.2

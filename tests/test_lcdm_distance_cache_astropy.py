import numpy as np


def test_lcdm_distance_cache_matches_astropy():
    import pytest

    pytest.importorskip("astropy")
    import astropy.units as u
    from astropy.cosmology import LambdaCDM

    from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache  # noqa: SLF001

    H0 = 70.0
    omega_m0 = 0.31
    omega_k0 = 0.0
    omega_lambda0 = 1.0 - omega_m0 - omega_k0

    z_max = 2.0
    cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=omega_m0, omega_k0=omega_k0, n_grid=20_001)
    # Set Tcmb0=0 to avoid radiation-density bookkeeping differences in Ok0.
    cosmo = LambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=omega_m0, Ode0=omega_lambda0, Tcmb0=0.0 * u.K)

    z = np.linspace(0.001, z_max, 256)
    c_km_s = 299792.458
    dL_cache = (c_km_s / H0) * cache.f(z)
    dL_astropy = cosmo.luminosity_distance(z).to_value(u.Mpc)

    rel = (dL_cache - dL_astropy) / dL_astropy
    max_abs_rel = float(np.max(np.abs(rel)))
    assert max_abs_rel < 5e-4

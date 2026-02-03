from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache  # noqa: SLF001


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark LCDM distance cache against astropy.cosmology.")
    ap.add_argument("--H0", type=float, default=70.0, help="H0 in km/s/Mpc (default 70).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 (default 0).")
    ap.add_argument("--z-max", type=float, default=2.0, help="Max z for cache + samples (default 2).")
    ap.add_argument("--n-grid", type=int, default=10_001, help="Cache grid size (default 10001).")
    ap.add_argument("--n-sample", type=int, default=256, help="Random z samples in (0,z_max] (default 256).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0).")
    ap.add_argument("--tol-rel", type=float, default=5e-4, help="Max |ΔdL/dL| tolerance (default 5e-4).")
    ap.add_argument("--out-json", default=None, help="Optional JSON output path.")
    ap.add_argument("--out-plot", default=None, help="Optional PNG path plotting relative error vs z.")
    args = ap.parse_args()

    try:
        import astropy.units as u
        from astropy.cosmology import LambdaCDM
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"astropy is required for this benchmark: {e}")

    H0 = float(args.H0)
    om = float(args.omega_m0)
    ok = float(args.omega_k0)
    z_max = float(args.z_max)

    ol = 1.0 - om - ok
    if not np.isfinite(ol):
        raise ValueError("Non-finite Omega_Lambda.")

    cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=om, omega_k0=ok, n_grid=int(args.n_grid))
    # Set Tcmb0=0 to avoid radiation-density bookkeeping differences in Ok0.
    cosmo = LambdaCDM(H0=H0 * u.km / u.s / u.Mpc, Om0=om, Ode0=ol, Tcmb0=0.0 * u.K)

    rng = np.random.default_rng(int(args.seed))
    z = rng.uniform(low=0.0, high=z_max, size=int(args.n_sample)).astype(float)
    z = z[(z > 0.0) & np.isfinite(z)]
    z = np.sort(z)
    if z.size == 0:
        raise ValueError("No valid z samples.")

    c_km_s = 299792.458
    dL_cache = (c_km_s / H0) * cache.f(z)
    dL_astropy = cosmo.luminosity_distance(z).to_value(u.Mpc)

    rel = (np.asarray(dL_cache, dtype=float) - np.asarray(dL_astropy, dtype=float)) / np.asarray(dL_astropy, dtype=float)
    max_abs_rel = float(np.max(np.abs(rel)))

    out = {
        "H0": H0,
        "omega_m0": om,
        "omega_k0": ok,
        "z_max": z_max,
        "n_grid": int(args.n_grid),
        "n_sample": int(z.size),
        "tol_rel": float(args.tol_rel),
        "max_abs_rel_err": max_abs_rel,
    }
    print(json.dumps(out, indent=2, sort_keys=True))

    if args.out_json is not None:
        Path(args.out_json).expanduser().resolve().write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if args.out_plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"matplotlib required for --out-plot: {e}")

        plt.figure(figsize=(7.2, 3.8))
        plt.plot(z, rel, ".", ms=2.5, alpha=0.8)
        plt.axhline(0.0, color="k", lw=0.8)
        plt.xlabel("z")
        plt.ylabel(r"$(d_L^{\\rm cache}-d_L^{\\rm astropy})/d_L^{\\rm astropy}$")
        plt.title("LCDM distance cache vs astropy")
        plt.tight_layout()
        plt.savefig(str(Path(args.out_plot).expanduser().resolve()), dpi=160)
        plt.close()

    return 0 if max_abs_rel <= float(args.tol_rel) else 2


if __name__ == "__main__":
    raise SystemExit(main())

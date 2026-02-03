from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(description="Build an oracle LCDM f(z) cache using ICAROGW (run under its venv).")
    ap.add_argument("--z-max", type=float, required=True, help="Maximum redshift for cache (matches analysis z_max).")
    ap.add_argument("--omega-m0", type=float, default=0.31, help="Omega_m0 (default 0.31).")
    ap.add_argument("--omega-k0", type=float, default=0.0, help="Omega_k0 (default 0; required).")
    ap.add_argument("--h0-ref", type=float, default=70.0, help="Reference H0 for building f(z) (default 70).")
    ap.add_argument("--tcmb0", type=float, default=0.0, help="CMB temperature for astropy cosmology (default 0 to match no-radiation LCDM).")
    ap.add_argument("--n-z", type=int, default=5001, help="Number of z-grid points (default 5001).")
    ap.add_argument("--out-json", required=True, help="Output JSON path.")
    args = ap.parse_args()

    z_max = float(args.z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("--z-max must be finite and positive.")

    if float(args.omega_k0) != 0.0:
        raise ValueError("ICAROGW oracle cache builder only supports omega_k0=0 (FlatLambdaCDM).")

    out_path = Path(args.out_json).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    from icarogw.wrappers import FlatLambdaCDM_wrap

    z0 = 1e-6
    n_z = int(args.n_z)
    if n_z < 100:
        raise ValueError("--n-z too small (need >= 100).")
    z_grid = np.linspace(z0, z_max, n_z)

    h0_ref = float(args.h0_ref)
    if not (np.isfinite(h0_ref) and h0_ref > 0.0):
        raise ValueError("--h0-ref must be finite and positive.")

    om0 = float(args.omega_m0)
    if not (np.isfinite(om0) and 0.0 < om0 < 1.0):
        raise ValueError("--omega-m0 must be finite and in (0,1).")

    tcmb0 = float(args.tcmb0)
    if not (np.isfinite(tcmb0) and tcmb0 >= 0.0):
        raise ValueError("--tcmb0 must be finite and >=0.")

    c_km_s = 299792.458

    bg = FlatLambdaCDM_wrap(zmax=z_max)
    bg.update(H0=h0_ref, Om0=om0, Tcmb0=tcmb0)
    dL = np.asarray(bg.cosmology.z2dl(z_grid), dtype=float)
    f_grid = dL * h0_ref / c_km_s
    if not np.all(np.isfinite(f_grid)):
        raise ValueError("Non-finite f_grid from ICAROGW cosmology.")

    try:
        import importlib.metadata as m

        icarogw_version = m.version("icarogw")
    except Exception:  # pragma: no cover
        icarogw_version = None

    out: dict[str, Any] = {
        "oracle": "icarogw",
        "oracle_version": icarogw_version,
        "distance_backend": "icarogw.wrappers.FlatLambdaCDM_wrap(astropy FlatLambdaCDM)",
        "tcmb0": float(tcmb0),
        "omega_m0": float(om0),
        "omega_k0": float(args.omega_k0),
        "z_max": float(z_max),
        "h0_ref": float(h0_ref),
        "z_grid": [float(x) for x in z_grid.tolist()],
        "f_grid": [float(x) for x in f_grid.tolist()],
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out_json": str(out_path)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

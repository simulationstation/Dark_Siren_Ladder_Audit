from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _parse_float_edges(s: str) -> np.ndarray:
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("--z-edges must have at least two comma-separated floats")
    edges = np.array([float(p) for p in parts], dtype=float)
    if np.any(~np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise ValueError("--z-edges must be finite and strictly increasing")
    return edges


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


@dataclass(frozen=True)
class ThetaBinMeta:
    z_min: float
    z_max: float
    n_obj_used: int
    theta_fits: str
    mask_fits: str


@dataclass(frozen=True)
class Manifest:
    created_utc: str
    input_dir: str
    estimator: str
    frame: str
    nside: int
    z_edges: list[float]
    bins: list[ThetaBinMeta]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Build HEALPix theta / velocity-proxy maps from a *prepared* kszx KszPipe input_dir.\n"
            "\n"
            "This reads galaxies.h5 (tcmb_90/150, bv_90/150, weight_vr) and produces one theta map per z bin.\n"
            "It is intended for the void-prism / E_G pipeline, *after* we have prepared professional kszx catalogs.\n"
        )
    )
    ap.add_argument("--input-dir", required=True, help="Directory containing galaxies.h5 (from scripts/kszx_prepare_kszpipe_inputs.py).")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: data/processed/void_prism/theta_from_kszpipe_<stamp>/).")

    ap.add_argument("--z-edges", default="0.43,0.50,0.57,0.64,0.70", help="Comma-separated z-bin edges.")
    ap.add_argument("--nside", type=int, default=256, help="HEALPix nside (default 256).")
    ap.add_argument("--frame", choices=["icrs", "galactic"], default="galactic", help="HEALPix frame (default galactic).")

    ap.add_argument(
        "--estimator",
        choices=["tcmb_150", "tcmb_90", "tcmb_sum", "vhat_150", "vhat_90", "vhat_sum"],
        default="vhat_sum",
        help=(
            "Estimator used to build the per-object value before binning.\n"
            "  - tcmb_* : uses mean-subtracted (weight_vr * tcmb_freq)\n"
            "  - vhat_* : uses mean-subtracted (weight_vr * (tcmb_freq / bv_freq))\n"
            "Sum means 90+150 (simple average in the per-object space).\n"
        ),
    )
    ap.add_argument("--remove-dipole", action="store_true", help="Remove monopole/dipole per map (default off).")
    args = ap.parse_args()

    hp = _require_healpy()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    gal_h5 = input_dir / "galaxies.h5"
    if not gal_h5.exists():
        raise FileNotFoundError(gal_h5)

    out_dir = Path(args.out_dir) if args.out_dir else Path("data/processed/void_prism") / f"theta_from_kszpipe_{_utc_stamp()}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Defer kszx import until after input validation.
    import kszx

    cat = kszx.Catalog.from_h5(str(gal_h5))
    z = np.asarray(cat.z, dtype=float)
    ra = np.asarray(cat.ra_deg, dtype=float)
    dec = np.asarray(cat.dec_deg, dtype=float)
    w = np.asarray(getattr(cat, "weight_vr", np.ones_like(z)), dtype=float)

    if not (z.shape == ra.shape == dec.shape == w.shape):
        raise RuntimeError("Catalog columns have inconsistent lengths.")

    def _get_t(freq: int) -> np.ndarray:
        return np.asarray(getattr(cat, f"tcmb_{freq}"), dtype=float)

    def _get_bv(freq: int) -> np.ndarray:
        return np.asarray(getattr(cat, f"bv_{freq}"), dtype=float)

    # Compute per-object value.
    if args.estimator == "tcmb_150":
        val = w * _get_t(150)
    elif args.estimator == "tcmb_90":
        val = w * _get_t(90)
    elif args.estimator == "tcmb_sum":
        val = 0.5 * w * (_get_t(90) + _get_t(150))
    elif args.estimator == "vhat_150":
        val = w * (_get_t(150) / _get_bv(150))
    elif args.estimator == "vhat_90":
        val = w * (_get_t(90) / _get_bv(90))
    elif args.estimator == "vhat_sum":
        val = 0.5 * w * ((_get_t(90) / _get_bv(90)) + (_get_t(150) / _get_bv(150)))
    else:
        raise RuntimeError("unreachable")

    m0 = np.isfinite(z) & np.isfinite(ra) & np.isfinite(dec) & np.isfinite(val) & np.isfinite(w) & (w > 0)
    z = z[m0]
    ra = ra[m0]
    dec = dec[m0]
    w = w[m0]
    val = val[m0]

    z_edges = _parse_float_edges(str(args.z_edges))
    npix = int(hp.nside2npix(int(args.nside)))

    bins: list[ThetaBinMeta] = []
    for z0, z1 in zip(z_edges[:-1], z_edges[1:], strict=False):
        m = (z >= float(z0)) & (z < float(z1))
        if not np.any(m):
            continue

        ra_sel = ra[m]
        dec_sel = dec[m]
        w_sel = w[m]
        v_sel = val[m]

        # Foreground mitigation: subtract weighted mean per z-bin (simple).
        mu = _weighted_mean(v_sel, w_sel)
        if np.isfinite(mu):
            v_sel = v_sel - mu

        pix = radec_to_healpix(ra_sel, dec_sel, nside=int(args.nside), frame=str(args.frame), nest=False)
        wsum = np.bincount(pix, weights=w_sel * v_sel, minlength=npix).astype(float)
        wcnt = np.bincount(pix, weights=w_sel, minlength=npix).astype(float)

        theta_map = np.zeros(npix, dtype=float)
        good = wcnt > 0
        theta_map[good] = wsum[good] / wcnt[good]

        mask = np.zeros(npix, dtype=float)
        mask[good] = 1.0

        if bool(args.remove_dipole):
            theta_map = hp.remove_dipole(theta_map, fitval=False, verbose=False)

        tag = f"z{float(z0):.3f}-{float(z1):.3f}"
        theta_path = out_dir / f"theta_{args.estimator}_{tag}.fits"
        mask_path = out_dir / f"mask_{args.estimator}_{tag}.fits"
        hp.write_map(str(theta_path), theta_map, overwrite=True, dtype=np.float64)
        hp.write_map(str(mask_path), mask, overwrite=True, dtype=np.float64)

        bins.append(
            ThetaBinMeta(
                z_min=float(z0),
                z_max=float(z1),
                n_obj_used=int(np.sum(m)),
                theta_fits=str(theta_path),
                mask_fits=str(mask_path),
            )
        )

    man = Manifest(
        created_utc=_utc_stamp(),
        input_dir=str(input_dir),
        estimator=str(args.estimator),
        frame=str(args.frame),
        nside=int(args.nside),
        z_edges=[float(x) for x in z_edges.tolist()],
        bins=bins,
    )
    (out_dir / "manifest.json").write_text(json.dumps(asdict(man), indent=2, sort_keys=True) + "\n")
    print(f"[theta_from_kszpipe] wrote {len(bins)} bins under {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


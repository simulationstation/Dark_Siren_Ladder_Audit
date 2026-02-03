from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.ingest_planck_lensing import load_planck_kappa
from entropy_horizon_recon.optical_bias.null_tests import rotate_map_random, rotate_radec_random
from entropy_horizon_recon.optical_bias.maps import _require_healpy, radec_to_healpix
from entropy_horizon_recon.void_prism_maps import (
    eg_void_from_spectra,
    load_void_catalog_csv,
    measure_void_prism_spectra,
    void_overdensity_map_from_pix,
)


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


@dataclass(frozen=True)
class BlockIdx:
    name: str
    z_eff: float
    env: dict[str, Any]
    zbin_idx: int
    idx: np.ndarray  # indices into full catalog arrays


def _match_zbin_idx(z_edges: np.ndarray, *, z0: float, z1: float) -> int:
    """Return index i such that z_edges[i]==z0 and z_edges[i+1]==z1 (within tolerance)."""
    z_edges = np.asarray(z_edges, dtype=float)
    if z_edges.ndim != 1 or z_edges.size < 2:
        raise ValueError("Invalid z_edges.")
    # Tolerant match (suite edges are printed with limited decimals).
    eps = 5e-3
    for i in range(z_edges.size - 1):
        if abs(float(z_edges[i]) - float(z0)) <= eps and abs(float(z_edges[i + 1]) - float(z1)) <= eps:
            return int(i)
    # Fallback: match by midpoint.
    zmid = 0.5 * (float(z0) + float(z1))
    i = int(np.searchsorted(z_edges, zmid, side="right") - 1)
    return int(np.clip(i, 0, z_edges.size - 2))


def _select_blocks_from_suite(cat, *, suite_blocks: list[dict[str, Any]], z_edges: np.ndarray) -> list[BlockIdx]:
    z = np.asarray(cat.z, dtype=float)
    Rv = np.asarray(cat.Rv, dtype=float) if cat.Rv is not None else None

    out: list[BlockIdx] = []
    for b in suite_blocks:
        env = dict(b.get("env") or {})
        z0 = float(env["z_min"])
        z1 = float(env["z_max"])
        m = (z >= z0) & (z < z1) & np.isfinite(z)
        rv_min = env.get("rv_min")
        rv_max = env.get("rv_max")
        if rv_min is not None or rv_max is not None:
            if Rv is None:
                raise ValueError("Suite block requests an Rv cut but catalog has no Rv.")
            if rv_min is not None and rv_max is None:
                # Mirror suite behavior: large bin used ">" split, not ">=".
                m &= Rv > float(rv_min)
            elif rv_max is not None and rv_min is None:
                m &= Rv <= float(rv_max)
            elif rv_min is not None and rv_max is not None:
                # Not expected for current suites; use a conservative (rv_min, rv_max] convention.
                m &= (Rv > float(rv_min)) & (Rv <= float(rv_max))

        idx = np.flatnonzero(m).astype(int)
        zbin_idx = _match_zbin_idx(z_edges, z0=z0, z1=z1)
        out.append(BlockIdx(name=str(b.get("name")), z_eff=float(b.get("z_eff")), env=env, zbin_idx=zbin_idx, idx=idx))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run fast null tests for a void-prism suite by breaking correlations via random rotations.\n"
            "This intentionally targets measurement sanity: if the measured signal is real, rotating voids "
            "or maps should strongly reduce cross-spectra amplitudes."
        )
    )
    ap.add_argument("--suite-json", required=True, help="suite_joint.json written by measure_void_prism_eg_suite_jackknife.py")
    ap.add_argument(
        "--null-kind",
        choices=["rotate_voids", "rotate_kappa", "rotate_theta"],
        default="rotate_voids",
        help="How to break correlations (default rotate_voids).",
    )
    ap.add_argument("--n-null", type=int, default=20, help="Number of null realizations (default 20).")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed (default 123).")
    ap.add_argument("--out-base", default=None, help="Output directory (default: outputs/void_prism_nulls_<stamp>/).")
    args = ap.parse_args()

    hp = _require_healpy()

    out_dir = Path(args.out_base) if args.out_base else Path("outputs") / f"void_prism_nulls_{_utc_stamp()}"
    tab_dir = out_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)
    print(f"[void_prism_nulls] out_dir={out_dir}", flush=True)

    suite_path = Path(args.suite_json)
    d = json.loads(suite_path.read_text())
    meta = d.get("meta") or {}
    blocks = d.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        raise ValueError("suite_joint.json missing blocks.")

    nside = int(meta.get("nside"))
    frame = str(meta.get("frame", "galactic"))
    lmax = int(meta["lmax"]) if meta.get("lmax") is not None else None
    bin_edges = np.asarray(meta.get("bin_edges"), dtype=int)
    prefactor = float(meta.get("prefactor", 1.0))
    eg_sign_val = meta.get("eg_sign", 1.0)
    eg_sign = float(eg_sign_val) if eg_sign_val is not None else 1.0
    if eg_sign not in (-1.0, 1.0):
        eg_sign = 1.0

    print(
        f"[void_prism_nulls] suite={suite_path} null_kind={args.null_kind} n_null={args.n_null} seed={args.seed} "
        f"nside={nside} lmax={lmax} blocks={len(blocks)}",
        flush=True,
    )

    # Load maps.
    if str(meta.get("kappa_source")) != "planck":
        raise ValueError("Null runner currently supports only kappa_source='planck'.")
    planck = load_planck_kappa(paths=DataPaths(Path.cwd()), nside_out=nside)
    kappa = np.asarray(planck.kappa_map, dtype=float)
    mask = np.asarray(planck.mask, dtype=float)
    theta_meta = meta.get("theta_fits")
    if isinstance(theta_meta, list):
        theta_paths = [str(p) for p in theta_meta]
    else:
        theta_paths = [str(theta_meta)]
    theta_list = [hp.read_map(p, verbose=False) for p in theta_paths]
    theta_mask_meta = meta.get("theta_mask_fits")
    if isinstance(theta_mask_meta, list):
        theta_mask_paths = [str(p) for p in theta_mask_meta]
    elif theta_mask_meta is None:
        theta_mask_paths = None
    else:
        theta_mask_paths = [str(theta_mask_meta)]
    theta_mask_list = [hp.read_map(p, verbose=False) for p in theta_mask_paths] if theta_mask_paths else None

    def _ud(m: np.ndarray) -> np.ndarray:
        return hp.ud_grade(m, nside) if hp.get_nside(m) != nside else m

    theta_list = [_ud(np.asarray(t, dtype=float)) for t in theta_list]
    if theta_mask_list is None:
        theta_mask_list = [np.ones_like(theta_list[0], dtype=float) for _ in range(len(theta_list))]
    else:
        theta_mask_list = [_ud(np.asarray(t, dtype=float)) for t in theta_mask_list]

    if meta.get("extra_mask_fits"):
        extra = hp.read_map(str(meta["extra_mask_fits"]), verbose=False)
        extra = _ud(extra)
        mask = mask * np.asarray(extra, dtype=float)

    # Load void catalog, build indices for each suite block.
    z_edges = np.asarray(meta.get("z_edges"), dtype=float)
    ra_col = str(meta.get("ra_col", "ra"))
    dec_col = str(meta.get("dec_col", "dec"))
    z_col = str(meta.get("z_col", "z"))
    Rv_col = meta.get("Rv_col", "Rv_mpc_h")
    weight_col = meta.get("weight_col", "weight_ngal")
    cat = load_void_catalog_csv(
        meta["void_csv"],
        ra_col=ra_col,
        dec_col=dec_col,
        z_col=z_col,
        Rv_col=Rv_col,
        weight_col=weight_col,
    )
    if cat.weight is None:
        weights = None
    else:
        weights = np.asarray(cat.weight, dtype=float)
    block_idx = _select_blocks_from_suite(cat, suite_blocks=blocks, z_edges=z_edges)

    # Choose theta map per z bin.
    if len(theta_list) == 1:
        theta_list = [theta_list[0] for _ in range(int(z_edges.size - 1))]
    if len(theta_mask_list) == 1:
        theta_mask_list = [theta_mask_list[0] for _ in range(int(z_edges.size - 1))]
    if len(theta_list) != int(z_edges.size - 1):
        raise ValueError(
            f"theta_fits count mismatch: got {len(theta_list)} but need 1 or {int(z_edges.size - 1)} "
            f"(one per z bin)."
        )
    if len(theta_mask_list) != int(z_edges.size - 1):
        raise ValueError(
            f"theta_mask_fits count mismatch: got {len(theta_mask_list)} but need 1 or {int(z_edges.size - 1)} "
            f"(one per z bin)."
        )

    # Precompute base pixels (ICRS -> map frame conversion happens in radec_to_healpix).
    pix_base = radec_to_healpix(cat.ra_deg, cat.dec_deg, nside=nside, frame=frame, nest=False)

    def _measure_from_pix(
        pix_all: np.ndarray, *, kappa_map: np.ndarray, theta_map_list: list[np.ndarray], theta_mask_list: list[np.ndarray]
    ) -> dict[str, Any]:
        eg_all: list[np.ndarray] = []
        cl_kv_all: list[np.ndarray] = []
        cl_tv_all: list[np.ndarray] = []
        for b in block_idx:
            pix = np.asarray(pix_all[b.idx], dtype=int)
            w = None if weights is None else np.asarray(weights[b.idx], dtype=float)
            m_block = np.asarray(mask, dtype=float) * np.asarray(theta_mask_list[int(b.zbin_idx)], dtype=float)
            vmap = void_overdensity_map_from_pix(pix, nside=nside, mask=m_block, weights=w, allow_empty=True)
            theta_map = np.asarray(theta_map_list[int(b.zbin_idx)], dtype=float)
            res = measure_void_prism_spectra(
                kappa_map=np.asarray(kappa_map, dtype=float),
                theta_map=theta_map,
                void_delta_map=vmap,
                mask=m_block,
                lmax=lmax,
                bin_edges=bin_edges,
            )
            cl_kv = np.asarray(res["cl_kappa_void"], dtype=float)
            cl_tv = np.asarray(res["cl_theta_void"], dtype=float)
            eg = eg_sign * eg_void_from_spectra(cl_kappa_void=cl_kv, cl_theta_void=cl_tv, prefactor=prefactor)
            eg_all.append(eg)
            cl_kv_all.append(cl_kv)
            cl_tv_all.append(cl_tv)

        eg_vec = np.concatenate(eg_all, axis=0)
        cl_kv_vec = np.concatenate(cl_kv_all, axis=0)
        cl_tv_vec = np.concatenate(cl_tv_all, axis=0)
        return {
            "eg_rms": _rms(eg_vec),
            "eg_median": float(np.median(eg_vec)),
            "cl_kappa_void_rms": _rms(cl_kv_vec),
            "cl_theta_void_rms": _rms(cl_tv_vec),
        }

    obs = _measure_from_pix(pix_base, kappa_map=kappa, theta_map_list=theta_list, theta_mask_list=theta_mask_list)

    rng = np.random.default_rng(int(args.seed))
    nulls: list[dict[str, float]] = []
    # Write a partial file periodically so interrupted runs still provide usable debugging info.
    progress_every = max(1, int(args.n_null) // 20)
    for _i in range(int(args.n_null)):
        if (_i + 1) == 1 or (_i + 1) % progress_every == 0 or (_i + 1) == int(args.n_null):
            print(f"[void_prism_nulls] {(_i + 1)}/{int(args.n_null)}", flush=True)

        if args.null_kind == "rotate_voids":
            ra2, dec2 = rotate_radec_random(cat.ra_deg, cat.dec_deg, seed=int(rng.integers(0, 2**31 - 1)))
            pix = radec_to_healpix(ra2, dec2, nside=nside, frame=frame, nest=False)
            k_map = kappa
            t_list = theta_list
        elif args.null_kind == "rotate_kappa":
            pix = pix_base
            k_map = rotate_map_random(kappa, seed=int(rng.integers(0, 2**31 - 1)))
            t_list = theta_list
        else:
            pix = pix_base
            k_map = kappa
            # Rotate each tomographic theta map independently.
            t_list = [
                rotate_map_random(t, seed=int(rng.integers(0, 2**31 - 1))) for t in theta_list
            ]

        nulls.append(_measure_from_pix(pix, kappa_map=k_map, theta_map_list=t_list, theta_mask_list=theta_mask_list))

        # Partial write for resiliency.
        if (_i + 1) == 1 or (_i + 1) % progress_every == 0:
            (tab_dir / "nulls_partial.json").write_text(
                json.dumps(
                    {
                        "suite_json": str(suite_path),
                        "null_kind": str(args.null_kind),
                        "n_null": int(args.n_null),
                        "seed": int(args.seed),
                        "obs": obs,
                        "null_rows": nulls,
                        "note": "PARTIAL (auto-saved during run)",
                    },
                    indent=2,
                )
            )

    eg_rms_null = np.array([n["eg_rms"] for n in nulls], dtype=float)
    kv_rms_null = np.array([n["cl_kappa_void_rms"] for n in nulls], dtype=float)
    tv_rms_null = np.array([n["cl_theta_void_rms"] for n in nulls], dtype=float)

    def _pval_ge(null: np.ndarray, x: float) -> float:
        null = null[np.isfinite(null)]
        if null.size == 0 or not np.isfinite(x):
            return float("nan")
        return float((np.sum(null >= x) + 1) / (null.size + 1))

    summary = {
        "suite_json": str(suite_path),
        "null_kind": str(args.null_kind),
        "n_null": int(args.n_null),
        "seed": int(args.seed),
        "obs": obs,
        "null": {
            "eg_rms": {
                "mean": float(np.mean(eg_rms_null)),
                "std": float(np.std(eg_rms_null, ddof=1)) if eg_rms_null.size > 1 else float("nan"),
                "p_ge_obs": _pval_ge(eg_rms_null, float(obs["eg_rms"])),
            },
            "cl_kappa_void_rms": {
                "mean": float(np.mean(kv_rms_null)),
                "std": float(np.std(kv_rms_null, ddof=1)) if kv_rms_null.size > 1 else float("nan"),
                "p_ge_obs": _pval_ge(kv_rms_null, float(obs["cl_kappa_void_rms"])),
            },
            "cl_theta_void_rms": {
                "mean": float(np.mean(tv_rms_null)),
                "std": float(np.std(tv_rms_null, ddof=1)) if tv_rms_null.size > 1 else float("nan"),
                "p_ge_obs": _pval_ge(tv_rms_null, float(obs["cl_theta_void_rms"])),
            },
        },
        "null_rows": nulls,
    }

    (tab_dir / "nulls.json").write_text(json.dumps(summary, indent=2))
    # Clean up the partial if it exists; final is authoritative.
    (tab_dir / "nulls_partial.json").unlink(missing_ok=True)
    print(str(out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

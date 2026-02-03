from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import ndimage

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.optical_bias.maps import _require_astropy
from entropy_horizon_recon.void_prism_velocity_products import load_2mrs_neuralnet_fields


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _grid_coords(*, n: int, spacing_mpc_h: float) -> np.ndarray:
    """Return 1D coordinate array for index -> comoving coordinate in (Mpc/h)."""
    # Per Lilow+2024 README: (i - 63.5) * 3.125 for n=128.
    return (np.arange(int(n), dtype=float) - (0.5 * (int(n) - 1))) * float(spacing_mpc_h)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a crude matter-defined void catalog from the 2MRS NeuralNet density grid.")
    ap.add_argument(
        "--method",
        choices=["components", "minima"],
        default="minima",
        help="Void definition method. 'components' thresholds underdense regions; 'minima' uses local minima + EDT radius (default minima).",
    )
    ap.add_argument(
        "--delta-thresh",
        type=float,
        default=-0.5,
        help="(components) Threshold on delta = (1+delta)-1 (default -0.5).",
    )
    ap.add_argument(
        "--min-cells",
        type=int,
        default=50,
        help="(components) Minimum number of grid cells in a connected component (default 50).",
    )
    ap.add_argument(
        "--minima-delta-thresh",
        type=float,
        default=-0.4,
        help="(minima) Only keep minima with delta < this (default -0.4).",
    )
    ap.add_argument(
        "--region-delta-thresh",
        type=float,
        default=-0.1,
        help="(minima) Define underdense region as delta < this for distance-transform radii (default -0.1).",
    )
    ap.add_argument(
        "--min-sep-cells",
        type=int,
        default=0,
        help="(minima) Non-maximum suppression: enforce a minimum separation in grid cells (default 0 = none).",
    )
    ap.add_argument(
        "--min-Rv",
        type=float,
        default=0.0,
        help="Optional minimum effective radius (Mpc/h) for minima/voids (default 0).",
    )
    ap.add_argument(
        "--max-voids",
        type=int,
        default=5000,
        help="Keep at most this many voids (components: largest by R_eff; minima: deepest minima first) (default 5000).",
    )
    ap.add_argument("--rmin", type=float, default=0.0, help="Optional min radius (Mpc/h) for component centers (default 0).")
    ap.add_argument("--rmax", type=float, default=200.0, help="Optional max radius (Mpc/h) for component centers (default 200).")
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV (default: data/processed/void_prism/voids_2mrs_nn_<stamp>.csv).",
    )
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow downloading 2MRS NeuralNet zip without a pinned SHA256 (bootstrap).",
    )
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path("data/processed/void_prism") / f"voids_2mrs_nn_{_utc_stamp()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    fields = load_2mrs_neuralnet_fields(paths=DataPaths(Path.cwd()), allow_unverified=bool(args.allow_unverified), include_errors=False)
    dens = np.asarray(fields.density_1pdelta, dtype=float)
    if dens.ndim != 3 or dens.shape[0] != dens.shape[1] or dens.shape[0] != dens.shape[2]:
        raise ValueError("Expected cubic 3D density grid.")

    delta = dens - 1.0
    valid = np.isfinite(delta)
    n = int(dens.shape[0])
    dx = float(fields.meta["grid_spacing_mpc_h"])
    coords = _grid_coords(n=n, spacing_mpc_h=dx)

    if args.method == "components":
        under = valid & (delta < float(args.delta_thresh))

        # Connected components in 3D (26-connectivity).
        structure = np.ones((3, 3, 3), dtype=int)
        labels, num = ndimage.label(under, structure=structure)
        if num <= 0:
            raise RuntimeError("No underdense components found; try raising delta-thresh (less negative).")

        counts = np.bincount(labels.ravel())
        # Drop background label 0.
        lab = np.arange(counts.size, dtype=int)
        good = (lab != 0) & (counts >= int(args.min_cells))
        good_labels = lab[good]
        if good_labels.size == 0:
            raise RuntimeError("No components passed min-cells; lower --min-cells or adjust --delta-thresh.")

        # Component centers in index coordinates.
        # Weight by unity; this is a crude geometry-based catalog (no watershed refinement).
        coms = ndimage.center_of_mass(np.ones_like(labels, dtype=float), labels=labels, index=good_labels)
        coms = np.asarray(coms, dtype=float)  # shape (n_comp, 3) with (i,j,k)

        # Convert COM indices -> physical coordinates (Mpc/h).
        x = np.interp(coms[:, 0], np.arange(n, dtype=float), coords)
        y = np.interp(coms[:, 1], np.arange(n, dtype=float), coords)
        zc = np.interp(coms[:, 2], np.arange(n, dtype=float), coords)
        r = np.sqrt(x * x + y * y + zc * zc)
        m_shell = (r >= float(args.rmin)) & (r <= float(args.rmax)) & np.isfinite(r) & (r > 0)
        if not np.any(m_shell):
            raise RuntimeError("All candidate component centers are outside [rmin,rmax].")

        good_labels = good_labels[m_shell]
        x = x[m_shell]
        y = y[m_shell]
        zc = zc[m_shell]
        r = r[m_shell]
        counts = counts[good_labels]

        # Effective radius from component volume.
        cell_vol = dx**3
        V = counts.astype(float) * cell_vol
        R_eff = (3.0 * V / (4.0 * np.pi)) ** (1.0 / 3.0)

        # Keep largest voids by R_eff.
        max_voids = int(args.max_voids)
        if max_voids > 0 and R_eff.size > max_voids:
            idx = np.argsort(R_eff)[::-1][:max_voids]
            x, y, zc, r, counts, R_eff = x[idx], y[idx], zc[idx], r[idx], counts[idx], R_eff[idx]

        delta_min = np.full_like(R_eff, np.nan, dtype=float)
    else:
        # ------------------------------------------------------------------
        # Minima-based "void centers": local minima + distance-transform radius.
        # This yields many more tracer points than simple connected components
        # and is closer to a watershed-style "basin center" definition.
        # ------------------------------------------------------------------
        work = np.where(valid, delta, np.inf)
        minf = ndimage.minimum_filter(work, size=3, mode="nearest")
        is_min = (work == minf) & np.isfinite(work) & (work < float(args.minima_delta_thresh))

        idxs = np.argwhere(is_min)
        if idxs.size == 0:
            raise RuntimeError("No local minima passed --minima-delta-thresh; adjust threshold.")

        # Sort minima by depth (most negative first).
        depths = work[is_min]
        order = np.argsort(depths)  # ascending => more negative first
        idxs = idxs[order]
        depths = depths[order]

        min_sep = int(args.min_sep_cells)
        if min_sep > 0:
            blocked = np.zeros_like(valid, dtype=bool)
            keep: list[int] = []
            for t, (i, j, k) in enumerate(idxs):
                if blocked[int(i), int(j), int(k)]:
                    continue
                keep.append(t)
                i0 = max(0, int(i) - min_sep)
                i1 = min(n, int(i) + min_sep + 1)
                j0 = max(0, int(j) - min_sep)
                j1 = min(n, int(j) + min_sep + 1)
                k0 = max(0, int(k) - min_sep)
                k1 = min(n, int(k) + min_sep + 1)
                blocked[i0:i1, j0:j1, k0:k1] = True
            idxs = idxs[np.array(keep, dtype=int)]
            depths = depths[np.array(keep, dtype=int)]

        # Region mask for EDT radii.
        region = valid & (delta < float(args.region_delta_thresh))
        dist = ndimage.distance_transform_edt(region) * dx

        # Convert indices -> physical coords.
        ii = idxs[:, 0].astype(float)
        jj = idxs[:, 1].astype(float)
        kk = idxs[:, 2].astype(float)
        x = np.interp(ii, np.arange(n, dtype=float), coords)
        y = np.interp(jj, np.arange(n, dtype=float), coords)
        zc = np.interp(kk, np.arange(n, dtype=float), coords)
        r = np.sqrt(x * x + y * y + zc * zc)
        R_eff = dist[idxs[:, 0], idxs[:, 1], idxs[:, 2]].astype(float)
        counts = np.ones_like(R_eff, dtype=float)
        delta_min = np.asarray(depths, dtype=float)

        m_shell = (
            (r >= float(args.rmin))
            & (r <= float(args.rmax))
            & np.isfinite(r)
            & (r > 0)
            & np.isfinite(R_eff)
            & (R_eff >= float(args.min_Rv))
        )
        if not np.any(m_shell):
            raise RuntimeError("All candidate minima are outside [rmin,rmax] or below --min-Rv.")

        x, y, zc, r, R_eff, counts, delta_min = x[m_shell], y[m_shell], zc[m_shell], r[m_shell], R_eff[m_shell], counts[m_shell], delta_min[m_shell]

        # Keep at most max_voids deepest minima (already sorted by depth).
        max_voids = int(args.max_voids)
        if max_voids > 0 and R_eff.size > max_voids:
            x, y, zc, r, R_eff, counts, delta_min = (
                x[:max_voids],
                y[:max_voids],
                zc[:max_voids],
                r[:max_voids],
                R_eff[:max_voids],
                counts[:max_voids],
                delta_min[:max_voids],
            )

    # Galactic lon/lat from x,y,z (assume standard Cartesian Galactic axes).
    lon = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0
    lat = np.rad2deg(np.arcsin(zc / r))

    # Convert to ICRS RA/Dec for compatibility with existing CSV loader.
    SkyCoord, u = _require_astropy()
    c = SkyCoord(l=lon * u.deg, b=lat * u.deg, frame="galactic")
    icrs = c.icrs
    ra = np.asarray(icrs.ra.deg, dtype=float)
    dec = np.asarray(icrs.dec.deg, dtype=float)

    # Low-z approximation: cz ~ 100 * d_(Mpc/h)  [km/s] => z ~ d_(Mpc/h) / (c/100).
    z_cosmo = r / 2997.92458

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        # Match the column names used by the existing BOSS DR12 void CSV for drop-in reuse
        # in the prism scripts (weight here is *not* galaxy count; we use n_cells).
        w.writerow(["ra", "dec", "z", "Rv_mpc_h", "weight_ngal", "delta_min"])
        for ra_i, dec_i, z_i, rv_i, ncell, dmin in zip(ra, dec, z_cosmo, R_eff, counts, delta_min, strict=False):
            w.writerow([f"{ra_i:.8f}", f"{dec_i:.8f}", f"{z_i:.8f}", f"{rv_i:.6f}", f"{float(ncell):.3f}", f"{float(dmin):.6f}"])

    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

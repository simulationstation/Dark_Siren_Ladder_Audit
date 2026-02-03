#!/usr/bin/env python3
"""
Fit the "intrinsic + selection-scaling" model using dipole vectors from a scan.json.

Given per-cut dipole vectors d_obs(w1_max) (typically the Poisson-GLM b-vectors from
`scripts/run_catwise_poisson_dipole_scan.py`) and the corresponding alpha_edge(w1_max),
fit:

  d_obs_j  ≈  d0  +  alpha_j * dm_vec

where:
  - d0 is an "intrinsic / non-scaling" dipole vector
  - dm_vec is an effective dipolar modulation of the magnitude limit (units: mag)

We perform a block-diagonal generalized least squares (GLS) using each row's 3x3 covariance
matrix cov_b (if provided). If cov_b is missing for a row, we fall back to an isotropic weight.

This is the statistically consistent replacement for the same fit done with the naive vector-sum
estimator: it uses the likelihood-derived dipole vectors and their uncertainties.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def vec_to_lb(vec: np.ndarray) -> tuple[float, float]:
    v = np.asarray(vec, dtype=float).reshape(3)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return float("nan"), float("nan")
    v = v / n
    l = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    b = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))
    return l, b


@dataclass(frozen=True)
class FitResult:
    d_ref: np.ndarray
    dm: np.ndarray
    cov_beta: np.ndarray
    chi2: float
    dof: int

    def as_dict(self) -> dict[str, Any]:
        d0_amp = float(np.linalg.norm(self.d_ref))
        dm_amp = float(np.linalg.norm(self.dm))
        d0_l, d0_b = vec_to_lb(self.d_ref)
        dm_l, dm_b = vec_to_lb(self.dm)

        sig = np.sqrt(np.clip(np.diag(self.cov_beta), 0.0, np.inf))
        d0_sig = sig[:3]
        dm_sig = sig[3:]

        def amp_sigma(v: np.ndarray, cov: np.ndarray) -> float:
            v = np.asarray(v, dtype=float).reshape(3)
            a = float(np.linalg.norm(v))
            if a <= 0:
                return float("nan")
            u = v / a
            return float(math.sqrt(max(0.0, float(u @ cov @ u))))

        d0_amp_sig = amp_sigma(self.d_ref, self.cov_beta[:3, :3])
        dm_amp_sig = amp_sigma(self.dm, self.cov_beta[3:6, 3:6])

        return {
            "d_ref": {
                "vec": [float(x) for x in self.d_ref],
                "sigma_vec": [float(x) for x in d0_sig],
                "amp": d0_amp,
                "amp_sigma": d0_amp_sig,
                "l_deg": d0_l,
                "b_deg": d0_b,
            },
            "delta_m_vec": {
                "vec_mag": [float(x) for x in self.dm],
                "sigma_vec_mag": [float(x) for x in dm_sig],
                "amp_mag": dm_amp,
                "amp_mag_sigma": dm_amp_sig,
                "l_deg": dm_l,
                "b_deg": dm_b,
            },
            "chi2": float(self.chi2),
            "dof": int(self.dof),
            "chi2_dof": float(self.chi2 / self.dof) if self.dof > 0 else float("nan"),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-json", required=True, help="scan.json from run_catwise_poisson_dipole_scan.py")
    ap.add_argument("--outdir", default=None, help="Output directory (default next to scan.json).")
    ap.add_argument("--use", choices=["poisson", "vector_sum"], default="poisson", help="Which dipole vectors to fit.")
    ap.add_argument(
        "--alpha-ref",
        default="mean",
        help="Reference alpha value for the intercept (mean|median|0 or a float literal). Default: mean.",
    )
    ap.add_argument("--make-plots", action="store_true")
    args = ap.parse_args()

    scan_path = Path(args.scan_json)
    d = json.loads(scan_path.read_text())
    rows = d.get("rows") or []
    if len(rows) < 3:
        raise SystemExit(f"Need >=3 rows; got {len(rows)}")

    alpha = np.asarray([float(r["alpha_edge"]) for r in rows], dtype=float)
    if str(args.alpha_ref).lower() in ("mean", "avg"):
        alpha_ref = float(np.mean(alpha))
    elif str(args.alpha_ref).lower() == "median":
        alpha_ref = float(np.median(alpha))
    elif str(args.alpha_ref).lower() in ("0", "zero"):
        alpha_ref = 0.0
    else:
        alpha_ref = float(args.alpha_ref)
    alpha_c = alpha - alpha_ref
    if args.use == "poisson":
        vecs = np.asarray([r["dipole"]["vec"] for r in rows], dtype=float)
        covs = []
        for r in rows:
            cb = r["dipole"].get("cov_b")
            if cb is None:
                covs.append(None)
            else:
                covs.append(np.asarray(cb, dtype=float))
    else:
        # Vector-sum only stores direction+amp, not full 3-vector. Use dipole axis from those.
        # Reconstruct a unit vector from l/b and scale by amplitude.
        def lb_to_unit(l_deg: float, b_deg: float) -> np.ndarray:
            l = math.radians(float(l_deg))
            b = math.radians(float(b_deg))
            return np.array([math.cos(b) * math.cos(l), math.cos(b) * math.sin(l), math.sin(b)], dtype=float)

        vecs = np.asarray(
            [
                float(r["vector_sum"]["amplitude"]) * lb_to_unit(float(r["vector_sum"]["l_deg"]), float(r["vector_sum"]["b_deg"]))
                for r in rows
            ],
            dtype=float,
        )
        covs = [None for _ in rows]

    K = int(len(rows))

    # Build GLS normal equations: sum_j X_j^T W_j X_j and sum_j X_j^T W_j y_j
    A = np.zeros((6, 6), dtype=float)
    bvec = np.zeros(6, dtype=float)
    chi2 = 0.0

    for j in range(K):
        aj = float(alpha_c[j])
        yj = np.asarray(vecs[j], dtype=float).reshape(3)

        # X_j is 3x6: [I3, aj*I3]
        Xj = np.zeros((3, 6), dtype=float)
        Xj[0, 0] = 1.0
        Xj[1, 1] = 1.0
        Xj[2, 2] = 1.0
        Xj[0, 3] = aj
        Xj[1, 4] = aj
        Xj[2, 5] = aj

        Cj = covs[j]
        if Cj is None or not np.all(np.isfinite(Cj)):
            Wj = np.eye(3)
        else:
            Wj = np.linalg.inv(Cj)

        A += Xj.T @ Wj @ Xj
        bvec += Xj.T @ Wj @ yj

    cov_beta = np.linalg.inv(A)
    beta = cov_beta @ bvec

    d_ref = beta[:3]
    dm = beta[3:]

    # Compute chi2 at solution.
    for j in range(K):
        aj = float(alpha_c[j])
        yj = np.asarray(vecs[j], dtype=float).reshape(3)
        pred = d_ref + aj * dm
        rj = yj - pred
        Cj = covs[j]
        if Cj is None or not np.all(np.isfinite(Cj)):
            chi2 += float(rj @ rj)
        else:
            Wj = np.linalg.inv(Cj)
            chi2 += float(rj.T @ Wj @ rj)

    dof = 3 * K - 6
    fit = FitResult(d_ref=np.asarray(d_ref, float), dm=np.asarray(dm, float), cov_beta=cov_beta, chi2=float(chi2), dof=int(dof))

    outdir = Path(args.outdir) if args.outdir else scan_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out = {
        "inputs": {"scan_json": str(scan_path), "use": str(args.use), "K": K, "alpha_ref": float(alpha_ref)},
        "fit": fit.as_dict(),
        "notes": [
            "Fits d_obs = d_ref + (alpha_edge - alpha_ref) * dm_vec using block-diagonal GLS.",
            "For Poisson mode, d_obs vectors are the GLM b-vectors (fractional dipole for small amplitudes).",
        ],
    }
    (outdir / f"alphaedge_scaling_fit_{args.use}.json").write_text(json.dumps(out, indent=2, sort_keys=True))

    if args.make_plots:
        try:
            import matplotlib.pyplot as plt

            # Project dipole onto the fitted dm direction to show scaling.
            dm_hat = dm / float(np.linalg.norm(dm)) if float(np.linalg.norm(dm)) > 0 else np.array([1.0, 0.0, 0.0])
            proj = vecs @ dm_hat
            proj_pred = (d_ref + alpha_c[:, None] * dm) @ dm_hat

            fig = plt.figure(figsize=(10, 3.2), dpi=200)
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

            ax1.scatter(alpha, proj, s=14)
            ax1.plot(alpha, proj_pred, color="k", lw=1.0)
            ax1.set_xlabel("alpha_edge")
            ax1.set_ylabel("proj(d_obs, dm_hat)")

            ax2.plot(alpha, np.linalg.norm(vecs, axis=1), marker="o", ms=3, lw=1.0)
            ax2.set_xlabel("alpha_edge")
            ax2.set_ylabel("|d_obs|")

            ax3.plot(alpha, np.linalg.norm(d_ref + alpha_c[:, None] * dm, axis=1), marker="o", ms=3, lw=1.0, color="tab:orange")
            ax3.set_xlabel("alpha_edge")
            ax3.set_ylabel("|d_fit|")

            fig.tight_layout()
            fig.savefig(outdir / f"alphaedge_scaling_fit_{args.use}.png")
            plt.close(fig)
        except Exception as e:  # noqa: BLE001
            (outdir / "plot_error.txt").write_text(f"Plotting failed: {e}\n")

    print(f"Wrote: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

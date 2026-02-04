from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from entropy_horizon_recon.hubble_tension import GaussianPrior
from entropy_horizon_recon.hubble_tension import as_1d_finite_array
from entropy_horizon_recon.hubble_tension import bayes_factor_between_priors_from_uniform_posterior
from entropy_horizon_recon.hubble_tension import integrate_posterior_prob
from entropy_horizon_recon.hubble_tension import normalize_pdf_grid
from entropy_horizon_recon.hubble_tension import posterior_quantiles


def _load_gate2_json(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError("Input JSON must be an object.")

    # Accept both:
    #  1) Gate-2 JSONs (top-level H0_grid + posterior/logL_H0_rel), and
    #  2) Pop-marginal summaries (inputs.H0_grid + mixture.posterior).
    if "H0_grid" in obj:
        H0 = as_1d_finite_array(obj["H0_grid"], name="H0_grid")
        if "posterior" in obj:
            p = as_1d_finite_array(obj["posterior"], name="posterior")
            return H0, p
        if "logL_H0_rel" in obj:
            logL = as_1d_finite_array(obj["logL_H0_rel"], name="logL_H0_rel")
            if logL.shape != H0.shape:
                raise ValueError("logL_H0_rel must match H0_grid.")
            p = np.exp(logL - float(np.max(logL)))
            return H0, p
        raise ValueError("Gate-2 JSON must contain either posterior or logL_H0_rel.")

    if "inputs" in obj and "mixture" in obj:
        inputs = obj.get("inputs") or {}
        mix = obj.get("mixture") or {}
        if not isinstance(inputs, dict) or not isinstance(mix, dict):
            raise ValueError("Invalid pop-marginal summary structure.")
        if "H0_grid" not in inputs:
            raise ValueError("Pop-marginal summary missing inputs.H0_grid.")
        if "posterior" not in mix:
            raise ValueError("Pop-marginal summary missing mixture.posterior.")
        H0 = as_1d_finite_array(inputs["H0_grid"], name="inputs.H0_grid")
        p = as_1d_finite_array(mix["posterior"], name="mixture.posterior")
        if p.shape != H0.shape:
            raise ValueError("mixture.posterior must match inputs.H0_grid.")
        return H0, p

    raise ValueError("Unrecognized JSON format (expected Gate-2 JSON or pop-marginal summary JSON).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize a Gate-2 GR H0 posterior in “Hubble tension” terms.")
    ap.add_argument("json", nargs="+", help="Gate-2 JSON output(s) with H0_grid and posterior/logL_H0_rel.")
    ap.add_argument("--h0-lo", type=float, default=68.0, help="Lower threshold for P(H0 < h0_lo) (default 68).")
    ap.add_argument("--h0-hi", type=float, default=72.0, help="Upper threshold for P(H0 > h0_hi) (default 72).")

    ap.add_argument("--planck-mean", type=float, default=67.4, help="Planck-like prior mean (default 67.4).")
    ap.add_argument("--planck-sigma", type=float, default=0.5, help="Planck-like prior sigma (default 0.5).")
    ap.add_argument("--shoes-mean", type=float, default=73.0, help="SH0ES-like prior mean (default 73.0).")
    ap.add_argument("--shoes-sigma", type=float, default=1.0, help="SH0ES-like prior sigma (default 1.0).")

    ap.add_argument("--out-json", default=None, help="Optional path to write a machine-readable summary JSON.")
    args = ap.parse_args()

    prior_planck = GaussianPrior(name="planck_like", mean=float(args.planck_mean), sigma=float(args.planck_sigma))
    prior_shoes = GaussianPrior(name="shoes_like", mean=float(args.shoes_mean), sigma=float(args.shoes_sigma))

    out_rows: list[dict[str, Any]] = []
    for p in [Path(x).expanduser().resolve() for x in args.json]:
        H0, post = _load_gate2_json(p)
        post = normalize_pdf_grid(H0, post)
        s = posterior_quantiles(H0, post)
        p_hi = integrate_posterior_prob(H0, post, lo=float(args.h0_hi), hi=None)
        p_lo = integrate_posterior_prob(H0, post, lo=None, hi=float(args.h0_lo))
        bf = bayes_factor_between_priors_from_uniform_posterior(H0, post, prior_a=prior_planck, prior_b=prior_shoes)
        row = {
            "path": str(p),
            "summary": s,
            "p_h0_gt_hi": p_hi,
            "p_h0_lt_lo": p_lo,
            "thresholds": {"h0_lo": float(args.h0_lo), "h0_hi": float(args.h0_hi)},
            "priors": {
                "planck_like": {"mean": float(prior_planck.mean), "sigma": float(prior_planck.sigma)},
                "shoes_like": {"mean": float(prior_shoes.mean), "sigma": float(prior_shoes.sigma)},
            },
            "bf_planck_over_shoes": bf,
        }
        out_rows.append(row)

        print(
            f"{p.name}: H0_map={s['H0_map']:.2f} p50={s['p50']:.2f} [p16,p84]=[{s['p16']:.2f},{s['p84']:.2f}] "
            f"P(H0<{args.h0_lo:.1f})={p_lo:.3g} P(H0>{args.h0_hi:.1f})={p_hi:.3g} "
            f"logBF(Planck/SH0ES)={bf['log_bf']:.3f}",
            flush=True,
        )

    if args.out_json:
        out_path = Path(args.out_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out_rows, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

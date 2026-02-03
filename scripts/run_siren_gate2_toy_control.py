from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.dark_siren_h0 import (
    CalibratedDetectionModel,
    _build_lcdm_distance_cache,
    compute_gr_h0_posterior_grid_hierarchical_pe,
)
from entropy_horizon_recon.siren_injection_recovery import InjectionRecoveryConfig, SyntheticEventTruth, synthesize_pe_posterior_samples


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _sample_powerlaw(rng: np.random.Generator, *, alpha: float, lo: float, hi: float, size: int) -> np.ndarray:
    """Sample x ~ x^{-alpha} on [lo,hi] for alpha != 1; alpha=1 uses log-uniform."""
    lo = float(lo)
    hi = float(hi)
    if not (hi > lo > 0.0):
        raise ValueError("Invalid powerlaw bounds.")
    a = float(alpha)
    u = rng.random(int(size))
    if abs(a - 1.0) < 1e-12:
        return lo * (hi / lo) ** u
    p = 1.0 - a
    return (u * (hi**p - lo**p) + lo**p) ** (1.0 / p)


def _sample_q_powerlaw(rng: np.random.Generator, *, beta: float, qmin: float, qmax: float, size: int) -> np.ndarray:
    qmin = float(qmin)
    qmax = float(qmax)
    if not (0.0 < qmin < qmax <= 1.0):
        raise ValueError("Invalid q bounds.")
    b = float(beta)
    u = rng.random(int(size))
    if abs(b + 1.0) < 1e-12:
        return qmin * (qmax / qmin) ** u
    p = b + 1.0
    return (u * (qmax**p - qmin**p) + qmin**p) ** (1.0 / p)


def _chirp_mass_from_m1_q(m1: np.ndarray, q: np.ndarray) -> np.ndarray:
    m1 = np.asarray(m1, dtype=float)
    q = np.asarray(q, dtype=float)
    m2 = m1 * q
    mt = m1 + m2
    return (m1 * m2) ** (3.0 / 5.0) / (mt ** (1.0 / 5.0))


def _build_z_sampler(
    *,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    n_grid: int = 5001,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (z_grid, cdf) for sampling z from the configured population."""
    z_max = float(z_max)
    z_grid = np.linspace(0.0, z_max, int(n_grid))
    # Use the same dimensionless distance cache helper as the GR(H0) control.
    cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=float(omega_m0), omega_k0=float(omega_k0), n_grid=int(n_grid))
    f = np.asarray(cache.f_grid, dtype=float)
    z = np.asarray(cache.z_grid, dtype=float)

    om = float(omega_m0)
    ok = float(omega_k0)
    ol = 1.0 - om - ok
    Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
    Ez = np.sqrt(np.clip(Ez2, 1e-30, np.inf))

    if pop_z_mode == "none":
        pdf = np.ones_like(z, dtype=float)
    else:
        # p(z) ∝ (dV/dz)/(1+z) shape ∝ f(z)^2 / [(1+z)^3 E(z)]
        pdf = (f**2) / (np.clip(1.0 + z, 1e-12, np.inf) ** 3 * np.clip(Ez, 1e-30, np.inf))
        if pop_z_mode == "comoving_powerlaw":
            pdf = pdf * np.clip(1.0 + z, 1e-12, np.inf) ** float(pop_z_k)
        elif pop_z_mode != "comoving_uniform":
            raise ValueError("Unknown pop_z_mode.")

    pdf = np.clip(pdf, 0.0, np.inf)
    # Drop z=0 exactly to avoid pathological dL=0 cases.
    pdf[0] = 0.0
    area = float(np.trapezoid(pdf, z))
    if not (np.isfinite(area) and area > 0.0):
        raise ValueError("Invalid z pdf normalization.")
    pdf = pdf / area
    cdf = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z))
    cdf = np.concatenate([[0.0], cdf])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return z, cdf


@dataclass(frozen=True)
class ToyDetectConfig:
    model: Literal["threshold", "logistic"] = "threshold"
    snr_threshold: float = 8.0
    snr_width: float = 0.5  # for logistic
    snr0: float = 25_000.0  # sets overall SNR scale (roughly snr ~ snr0/dL)
    mc_ref: float = 30.0
    mc_exp: float = 5.0 / 6.0


def _pdet_from_snr(snr: np.ndarray, cfg: ToyDetectConfig) -> np.ndarray:
    snr = np.asarray(snr, dtype=float)
    thr = float(cfg.snr_threshold)
    if cfg.model == "threshold":
        return (snr > thr).astype(float)
    w = float(cfg.snr_width)
    if not (np.isfinite(w) and w > 0.0):
        raise ValueError("snr_width must be finite and >0 for logistic p_det.")
    x = (snr - thr) / w
    # stable logistic
    out = np.where(x >= 0.0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
    return np.clip(out, 0.0, 1.0)


def _draw_detected_truths(
    *,
    rng: np.random.Generator,
    n_events: int,
    h0_true: float,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    pop_mass_mode: Literal["powerlaw_q", "powerlaw_peak_q"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_q_min: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    det: ToyDetectConfig,
) -> list[SyntheticEventTruth]:
    n_events = int(n_events)
    if n_events <= 0:
        raise ValueError("n_events must be positive.")

    z_grid, z_cdf = _build_z_sampler(
        omega_m0=float(omega_m0),
        omega_k0=float(omega_k0),
        z_max=float(z_max),
        pop_z_mode=str(pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(pop_z_k),
    )
    dist_cache = _build_lcdm_distance_cache(z_max=float(z_max), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    const = PhysicalConstants()

    out: list[SyntheticEventTruth] = []
    tries = 0
    while len(out) < n_events:
        tries += 1
        if tries > 5_000_000:
            raise RuntimeError("Too many rejection-sampling attempts; increase snr0 or lower snr_threshold.")

        # Sample z via inverse CDF.
        u = float(rng.random())
        z = float(np.interp(u, z_cdf, z_grid))
        if not (np.isfinite(z) and z > 0.0):
            continue

        # Sample masses.
        if pop_mass_mode == "powerlaw_q":
            m1 = float(_sample_powerlaw(rng, alpha=float(pop_m1_alpha), lo=float(pop_m_min), hi=float(pop_m_max), size=1)[0])
        elif pop_mass_mode == "powerlaw_peak_q":
            if float(rng.random()) < float(pop_m_peak_frac):
                m1 = float(rng.normal(loc=float(pop_m_peak), scale=float(pop_m_peak_sigma)))
                if not (np.isfinite(m1) and float(pop_m_min) <= m1 <= float(pop_m_max)):
                    continue
            else:
                m1 = float(_sample_powerlaw(rng, alpha=float(pop_m1_alpha), lo=float(pop_m_min), hi=float(pop_m_max), size=1)[0])
        else:
            raise ValueError("Unknown pop_mass_mode.")

        q = float(_sample_q_powerlaw(rng, beta=float(pop_q_beta), qmin=float(pop_q_min), qmax=1.0, size=1)[0])
        m2 = float(m1 * q)
        if not (np.isfinite(m2) and m2 > 0.0 and m2 <= m1):
            continue

        mc_src = float(_chirp_mass_from_m1_q(np.array([m1]), np.array([q]))[0])
        mc_det = float(mc_src * (1.0 + z))

        dL = float((const.c_km_s / float(h0_true)) * float(dist_cache.f(np.array([z]))[0]))
        if not (np.isfinite(dL) and dL > 0.0):
            continue

        snr = float(det.snr0) * (mc_det / float(det.mc_ref)) ** float(det.mc_exp) / dL
        pdet = float(_pdet_from_snr(np.array([snr]), det)[0])
        if float(rng.random()) > pdet:
            continue

        out.append(
            SyntheticEventTruth(
                event=f"TOY_{len(out)+1:05d}",
                z=z,
                dL_mpc_true=dL,
                m1_source=m1,
                m2_source=m2,
                chirp_mass_det_true=mc_det,
                mass_ratio_true=q,
                snr_net_opt_fid=snr,
                dL_mpc_fid=dL,
                snr_net_opt_true=snr,
                p_det_true=pdet,
            )
        )

    return out


def _toy_alpha_grid(
    *,
    rng: np.random.Generator,
    h0_grid: np.ndarray,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    pop_mass_mode: Literal["powerlaw_q", "powerlaw_peak_q"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_q_min: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    det: ToyDetectConfig,
    n_mc: int = 200_000,
) -> np.ndarray:
    """Monte Carlo alpha(H0) for the toy model (detection fraction)."""
    n_mc = int(n_mc)
    if n_mc < 10_000:
        raise ValueError("n_mc too small for alpha MC.")

    # Draw population samples once.
    truths = _draw_detected_truths(
        rng=rng,
        n_events=n_mc,
        h0_true=float(h0_grid[0]),  # placeholder; we only use z/masses, will recompute dL per H0
        omega_m0=float(omega_m0),
        omega_k0=float(omega_k0),
        z_max=float(z_max),
        pop_z_mode=pop_z_mode,
        pop_z_k=float(pop_z_k),
        pop_mass_mode=pop_mass_mode,
        pop_m1_alpha=float(pop_m1_alpha),
        pop_m_min=float(pop_m_min),
        pop_m_max=float(pop_m_max),
        pop_q_beta=float(pop_q_beta),
        pop_q_min=float(pop_q_min),
        pop_m_peak=float(pop_m_peak),
        pop_m_peak_sigma=float(pop_m_peak_sigma),
        pop_m_peak_frac=float(pop_m_peak_frac),
        det=ToyDetectConfig(model="logistic", snr_threshold=-1e9),  # always accept; we only want population draws
    )
    z = np.asarray([t.z for t in truths], dtype=float)
    mc_det = np.asarray([t.chirp_mass_det_true for t in truths], dtype=float)

    dist_cache = _build_lcdm_distance_cache(z_max=float(z_max), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    const = PhysicalConstants()
    fz = np.asarray(dist_cache.f(z), dtype=float)
    alpha = np.zeros((np.asarray(h0_grid, dtype=float).size,), dtype=float)
    for i, H0 in enumerate(np.asarray(h0_grid, dtype=float).tolist()):
        dL = (const.c_km_s / float(H0)) * fz
        snr = float(det.snr0) * (mc_det / float(det.mc_ref)) ** float(det.mc_exp) / np.clip(dL, 1e-12, np.inf)
        alpha[i] = float(np.mean(_pdet_from_snr(snr, det)))
    return alpha


def main() -> int:
    ap = argparse.ArgumentParser(description="Toy Gate-2 GR(H0) selection-on control (self-consistent).")
    ap.add_argument("--out", default=None, help="Output directory (default outputs/siren_gate2_toy_<UTCSTAMP>).")
    ap.add_argument("--h0-true", type=float, default=70.0)
    ap.add_argument("--n-events", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--h0-min", type=float, default=40.0)
    ap.add_argument("--h0-max", type=float, default=120.0)
    ap.add_argument("--h0-n", type=int, default=161)
    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=0.62)

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform")
    ap.add_argument("--pop-z-k", type=float, default=0.0)
    ap.add_argument("--pop-mass-mode", choices=["powerlaw_q", "powerlaw_peak_q"], default="powerlaw_peak_q")
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3)
    ap.add_argument("--pop-m-min", type=float, default=5.0)
    ap.add_argument("--pop-m-max", type=float, default=80.0)
    ap.add_argument("--pop-q-beta", type=float, default=0.0)
    ap.add_argument("--pop-q-min", type=float, default=0.05)
    ap.add_argument("--pop-m-peak", type=float, default=35.0)
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0)
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1)

    ap.add_argument("--det-model", choices=["threshold", "logistic"], default="threshold")
    ap.add_argument("--snr-threshold", type=float, default=8.0)
    ap.add_argument("--snr-width", type=float, default=0.5)
    ap.add_argument("--snr0", type=float, default=25_000.0)
    ap.add_argument("--mc-ref", type=float, default=30.0)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"siren_gate2_toy_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    det = ToyDetectConfig(
        model=str(args.det_model),  # type: ignore[arg-type]
        snr_threshold=float(args.snr_threshold),
        snr_width=float(args.snr_width),
        snr0=float(args.snr0),
        mc_ref=float(args.mc_ref),
    )

    truths = _draw_detected_truths(
        rng=rng,
        n_events=int(args.n_events),
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_q_min=float(args.pop_q_min),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        det=det,
    )

    # Use the same PE synthesis knobs as the injection-recovery suite defaults.
    cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        det_model="threshold",
        snr_binned_nbins=200,
        selection_ifar_thresh_yr=1.0,
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_mass_mode="powerlaw_peak_q_smooth",
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=3.0,
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        weight_mode="none",
        pe_n_samples=10_000,
        pe_synth_mode="likelihood_resample",
        pe_prior_resample_n_candidates=200_000,
        pe_seed=0,
        dl_frac_sigma0=0.25,
        dl_frac_sigma_floor=0.05,
        dl_sigma_mode="snr",
        mc_frac_sigma0=0.02,
        q_sigma0=0.08,
        pe_prior_dL_expr="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)",
        pe_prior_chirp_mass_expr="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)",
        pe_prior_mass_ratio_expr="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)",
    )

    pe_by_event = {t.event: synthesize_pe_posterior_samples(truth=t, cfg=cfg, rng=rng) for t in truths}

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))
    res_off = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event=pe_by_event,
        H0_grid=H0_grid,
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        cache_dir=None,
        include_pdet_in_event_term=True,
        pdet_model=CalibratedDetectionModel(det_model="threshold", snr_threshold=float(det.snr_threshold)),
        injections=None,
        ifar_threshold_yr=1.0,
        det_model="threshold",
        snr_threshold=None,
        snr_binned_nbins=200,
        weight_mode="none",
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(args.pop_z_k),
        pop_mass_mode="powerlaw_peak_q_smooth",
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=3.0,
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        event_qc_mode="skip",
        event_min_finite_frac=0.9,
        prior="uniform",
    )

    alpha_grid = _toy_alpha_grid(
        rng=rng,
        h0_grid=H0_grid,
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_q_min=float(args.pop_q_min),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        det=det,
        n_mc=200_000,
    )
    log_alpha = np.log(np.clip(alpha_grid, 1e-300, np.inf))

    logL_sum = np.asarray(res_off["logL_sum_events_rel"], dtype=float)
    logL_on = logL_sum - float(res_off["n_events"]) * log_alpha
    # Normalize posterior on grid.
    logL_on = logL_on - float(np.max(logL_on))
    p = np.exp(np.clip(logL_on, -700.0, 50.0))
    p = p / float(np.sum(p))
    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))
    H0_map = float(H0_grid[int(np.argmax(p))])

    out = {
        "manifest": {
            "created_utc": _utc_stamp(),
            "h0_true": float(args.h0_true),
            "n_events": int(args.n_events),
            "seed": int(args.seed),
            "toy_detect": asdict(det),
            "pop": {
                "z_max": float(args.z_max),
                "pop_z_mode": str(args.pop_z_mode),
                "pop_z_k": float(args.pop_z_k),
                "pop_mass_mode": str(args.pop_mass_mode),
                "pop_m1_alpha": float(args.pop_m1_alpha),
                "pop_m_min": float(args.pop_m_min),
                "pop_m_max": float(args.pop_m_max),
                "pop_q_beta": float(args.pop_q_beta),
                "pop_q_min": float(args.pop_q_min),
                "pop_m_peak": float(args.pop_m_peak),
                "pop_m_peak_sigma": float(args.pop_m_peak_sigma),
                "pop_m_peak_frac": float(args.pop_m_peak_frac),
            },
            "H0_grid": [float(x) for x in H0_grid.tolist()],
        },
        "summary": {"H0_map": float(H0_map), "p50": float(q50), "p16": float(q16), "p84": float(q84)},
        "alpha_grid": [float(x) for x in alpha_grid.tolist()],
        "posterior_grid": [float(x) for x in p.tolist()],
        "res_off": res_off,
        "truths": [t.to_jsonable() for t in truths],
    }
    _write_json(out_dir / "toy_gate2.json", out)
    print(f"toy_gate2: H0_true={float(args.h0_true):.3f}  p50={q50:.3f}  map={H0_map:.3f}  out={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

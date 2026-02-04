from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from entropy_horizon_recon.constants import PhysicalConstants
from entropy_horizon_recon.dark_siren_h0 import _build_lcdm_distance_cache, compute_gr_h0_posterior_grid_hierarchical_pe  # noqa: SLF001
from entropy_horizon_recon.siren_injection_recovery import InjectionRecoveryConfig, SyntheticEventTruth, synthesize_pe_posterior_samples


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_float_array(x: Any, *, name: str) -> np.ndarray:
    if not isinstance(x, (list, tuple)):
        raise ValueError(f"Expected '{name}' to be a list.")
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"Expected '{name}' to be a 1D list with >=2 entries.")
    if np.any(~np.isfinite(arr)):
        raise ValueError(f"Non-finite values in '{name}'.")
    return arr


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    s = float(np.sum(p))
    if not (np.isfinite(s) and s > 0.0):
        raise ValueError("Posterior normalization failed (non-positive sum).")
    return p / s


def _cdf_at(grid: np.ndarray, posterior: np.ndarray, x: float) -> float:
    g = np.asarray(grid, dtype=float)
    p = _normalize_prob(np.asarray(posterior, dtype=float))
    cdf = np.cumsum(p)
    u = float(np.interp(float(x), g, cdf, left=0.0, right=1.0))
    return float(np.clip(u, 0.0, 1.0))


def _h0_grid_posterior_from_logL_rel(H0_grid: np.ndarray, logL_rel: np.ndarray) -> dict[str, Any]:
    H0_grid = np.asarray(H0_grid, dtype=float)
    logL_rel = np.asarray(logL_rel, dtype=float)
    if H0_grid.shape != logL_rel.shape:
        raise ValueError("H0_grid and logL_rel must have matching shapes.")
    m = np.isfinite(logL_rel)
    if not np.any(m):
        raise ValueError("logL_rel has no finite entries.")
    log_post = logL_rel - float(np.nanmax(logL_rel[m]))
    p = np.exp(np.clip(log_post, -700.0, 50.0))
    p = np.where(m, p, 0.0)
    p = _normalize_prob(p)

    cdf = np.cumsum(p)
    q16 = float(np.interp(0.16, cdf, H0_grid))
    q50 = float(np.interp(0.50, cdf, H0_grid))
    q84 = float(np.interp(0.84, cdf, H0_grid))

    mean = float(np.sum(p * H0_grid))
    sd = float(np.sqrt(np.sum(p * (H0_grid - mean) ** 2)))
    i_map = int(np.argmax(p))
    H0_map = float(H0_grid[i_map])
    return {
        "H0_map": float(H0_map),
        "H0_map_index": int(i_map),
        "H0_map_at_edge": bool(i_map == 0 or i_map == (H0_grid.size - 1)),
        "logL_H0_rel": [float(x) for x in log_post.tolist()],
        "posterior": [float(x) for x in p.tolist()],
        "summary": {"mean": mean, "sd": sd, "p50": q50, "p16": q16, "p84": q84},
    }


def _sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.tanh(0.5 * x))


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
        # Match `dark_siren_h0._event_logL_h0_grid_from_hierarchical_pe_samples` (shape-only):
        #   p(z) ∝ f(z)^2 / [(1+z)^3 E(z)]
        pdf = (f**2) / (np.clip(1.0 + z, 1e-12, np.inf) ** 3 * np.clip(Ez, 1e-30, np.inf))
        if pop_z_mode == "comoving_powerlaw":
            pdf = pdf * np.clip(1.0 + z, 1e-12, np.inf) ** float(pop_z_k)
        elif pop_z_mode != "comoving_uniform":
            raise ValueError("Unknown pop_z_mode.")

    pdf = np.clip(pdf, 0.0, np.inf)
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


class ToyDetectConfig:
    def __init__(self, *, model: Literal["threshold", "logistic"], snr_threshold: float, snr_width: float, snr0: float, mc_ref: float, mc_exp: float) -> None:
        self.model = str(model)  # type: ignore[assignment]
        self.snr_threshold = float(snr_threshold)
        self.snr_width = float(snr_width)
        self.snr0 = float(snr0)
        self.mc_ref = float(mc_ref)
        self.mc_exp = float(mc_exp)


def _pdet_from_snr(snr: np.ndarray, cfg: ToyDetectConfig) -> np.ndarray:
    snr = np.asarray(snr, dtype=float)
    thr = float(cfg.snr_threshold)
    if cfg.model == "threshold":
        return (snr > thr).astype(float)
    w = float(cfg.snr_width)
    if not (np.isfinite(w) and w > 0.0):
        raise ValueError("snr_width must be finite and >0 for logistic p_det.")
    x = (snr - thr) / w
    # stable logistic: 0.5*(1+tanh(x/2))
    return np.clip(_sigmoid_stable(x), 0.0, 1.0)


def _draw_population_sample(
    *,
    rng: np.random.Generator,
    z_grid: np.ndarray,
    z_cdf: np.ndarray,
    pop_mass_mode: Literal["powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_q_min: float,
    pop_m_taper_delta: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
) -> tuple[float, float, float]:
    # z via inverse CDF
    z = float(np.interp(float(rng.random()), z_cdf, z_grid))
    if not (np.isfinite(z) and z > 0.0):
        return float("nan"), float("nan"), float("nan")

    mmin = float(pop_m_min)
    mmax = float(pop_m_max)
    if not (mmax > mmin > 0.0):
        raise ValueError("Invalid pop_m_min/pop_m_max.")
    qmin = float(pop_q_min)
    if not (0.0 < qmin < 1.0):
        raise ValueError("Invalid pop_q_min.")
    beta = float(pop_q_beta)
    q = float(_sample_q_powerlaw(rng, beta=beta, qmin=qmin, qmax=1.0, size=1)[0])
    if not (np.isfinite(q) and 0.0 < q <= 1.0):
        return float("nan"), float("nan"), float("nan")

    # m1 proposal: mixture powerlaw + optional peak; taper handled by rejection
    if pop_mass_mode == "powerlaw_q":
        m1 = float(_sample_powerlaw(rng, alpha=float(pop_m1_alpha), lo=mmin, hi=mmax, size=1)[0])
        m2 = float(m1 * q)
        if not (np.isfinite(m2) and m2 >= mmin and m2 <= m1):
            return float("nan"), float("nan"), float("nan")
        return z, m1, m2

    # Smooth modes: accept-reject using taper(m1,m2)
    delta = float(pop_m_taper_delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError("pop_m_taper_delta must be finite and >0 for smooth mass modes.")

    f_peak = float(pop_m_peak_frac)
    mp = float(pop_m_peak)
    sig = float(pop_m_peak_sigma)

    for _ in range(10_000):
        if pop_mass_mode == "powerlaw_q_smooth":
            m1 = float(_sample_powerlaw(rng, alpha=float(pop_m1_alpha), lo=mmin, hi=mmax, size=1)[0])
        elif pop_mass_mode == "powerlaw_peak_q_smooth":
            if float(rng.random()) < f_peak:
                m1 = float(rng.normal(loc=mp, scale=sig))
                if not (np.isfinite(m1) and mmin <= m1 <= mmax):
                    continue
            else:
                m1 = float(_sample_powerlaw(rng, alpha=float(pop_m1_alpha), lo=mmin, hi=mmax, size=1)[0])
        else:
            raise ValueError("Unknown pop_mass_mode.")

        m2 = float(m1 * q)
        if not (np.isfinite(m2) and m2 > 0.0 and m2 <= m1):
            continue
        t1 = _sigmoid_stable((m1 - mmin) / delta) * _sigmoid_stable((mmax - m1) / delta)
        t2 = _sigmoid_stable((m2 - mmin) / delta) * _sigmoid_stable((mmax - m2) / delta)
        taper = float(np.clip(t1 * t2, 0.0, 1.0))
        if float(rng.random()) <= taper:
            return z, m1, m2

    return float("nan"), float("nan"), float("nan")


def _draw_detected_toy_events(
    *,
    rng: np.random.Generator,
    n_events: int,
    h0_true: float,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    pop_mass_mode: Literal["powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_q_min: float,
    pop_m_taper_delta: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    det: ToyDetectConfig,
) -> list[tuple[float, float, float, float, float]]:
    """Return detected samples as tuples (z, m1_src, m2_src, dL_true, snr_true)."""
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

    out: list[tuple[float, float, float, float, float]] = []
    tries = 0
    while len(out) < n_events:
        tries += 1
        if tries > 10_000_000:
            raise RuntimeError("Too many rejection-sampling attempts; increase snr0 or lower snr_threshold.")

        z, m1, m2 = _draw_population_sample(
            rng=rng,
            z_grid=z_grid,
            z_cdf=z_cdf,
            pop_mass_mode=str(pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
            pop_q_min=float(pop_q_min),
            pop_m_taper_delta=float(pop_m_taper_delta),
            pop_m_peak=float(pop_m_peak),
            pop_m_peak_sigma=float(pop_m_peak_sigma),
            pop_m_peak_frac=float(pop_m_peak_frac),
        )
        if not (np.isfinite(z) and np.isfinite(m1) and np.isfinite(m2)):
            continue
        q = m2 / m1
        mc_src = float(_chirp_mass_from_m1_q(np.array([m1]), np.array([q]))[0])
        mc_det = float(mc_src * (1.0 + z))

        dL = float((const.c_km_s / float(h0_true)) * float(dist_cache.f(np.array([z]))[0]))
        if not (np.isfinite(dL) and dL > 0.0):
            continue

        snr = float(det.snr0) * (mc_det / float(det.mc_ref)) ** float(det.mc_exp) / dL
        pdet = float(_pdet_from_snr(np.array([snr]), det)[0])
        if float(rng.random()) > pdet:
            continue
        out.append((float(z), float(m1), float(m2), float(dL), float(snr)))

    return out


def _toy_alpha_grid_mc(
    *,
    seed: int,
    h0_grid: np.ndarray,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    pop_mass_mode: Literal["powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
    pop_m1_alpha: float,
    pop_m_min: float,
    pop_m_max: float,
    pop_q_beta: float,
    pop_q_min: float,
    pop_m_taper_delta: float,
    pop_m_peak: float,
    pop_m_peak_sigma: float,
    pop_m_peak_frac: float,
    det: ToyDetectConfig,
    n_mc: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    n_mc = int(n_mc)
    if n_mc < 50_000:
        raise ValueError("n_mc too small for alpha MC (need >=50k).")
    h0_grid = np.asarray(h0_grid, dtype=float)

    z_grid, z_cdf = _build_z_sampler(
        omega_m0=float(omega_m0),
        omega_k0=float(omega_k0),
        z_max=float(z_max),
        pop_z_mode=str(pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(pop_z_k),
    )
    dist_cache = _build_lcdm_distance_cache(z_max=float(z_max), omega_m0=float(omega_m0), omega_k0=float(omega_k0))
    const = PhysicalConstants()

    # Draw population samples.
    zs = np.empty((n_mc,), dtype=float)
    m1s = np.empty((n_mc,), dtype=float)
    m2s = np.empty((n_mc,), dtype=float)
    filled = 0
    while filled < n_mc:
        z, m1, m2 = _draw_population_sample(
            rng=rng,
            z_grid=z_grid,
            z_cdf=z_cdf,
            pop_mass_mode=str(pop_mass_mode),  # type: ignore[arg-type]
            pop_m1_alpha=float(pop_m1_alpha),
            pop_m_min=float(pop_m_min),
            pop_m_max=float(pop_m_max),
            pop_q_beta=float(pop_q_beta),
            pop_q_min=float(pop_q_min),
            pop_m_taper_delta=float(pop_m_taper_delta),
            pop_m_peak=float(pop_m_peak),
            pop_m_peak_sigma=float(pop_m_peak_sigma),
            pop_m_peak_frac=float(pop_m_peak_frac),
        )
        if not (np.isfinite(z) and np.isfinite(m1) and np.isfinite(m2)):
            continue
        zs[filled] = float(z)
        m1s[filled] = float(m1)
        m2s[filled] = float(m2)
        filled += 1

    qs = np.clip(m2s / m1s, 1e-6, 1.0)
    mc_src = _chirp_mass_from_m1_q(m1s, qs)
    mc_det = mc_src * (1.0 + zs)

    fz = np.asarray(dist_cache.f(zs), dtype=float)
    if not np.all(np.isfinite(fz)) or np.any(fz <= 0.0):
        raise ValueError("Invalid f(z) in toy alpha MC.")

    # SNR scales linearly with H0 for fixed (z, masses): snr = H0 * A(z, masses).
    A = float(det.snr0) * (mc_det / float(det.mc_ref)) ** float(det.mc_exp) / (float(const.c_km_s) * fz)
    if not np.all(np.isfinite(A)) or np.any(A <= 0.0):
        raise ValueError("Invalid A scaling in toy alpha MC.")

    alpha = np.empty_like(h0_grid, dtype=float)
    for j, H0 in enumerate(h0_grid.tolist()):
        snr = float(H0) * A
        alpha[j] = float(np.mean(_pdet_from_snr(snr, det)))
    return np.clip(alpha, 1e-300, 1.0)


# Globals for ProcessPool workers (avoid pickling large objects repeatedly).
_G = {}


def _init_worker(state: dict[str, Any]) -> None:
    global _G  # noqa: PLW0603
    _G = state


def _run_rep(rep: int, seed: int) -> dict[str, Any]:
    global _G  # noqa: PLW0603
    out_dir = Path(str(_G["json_dir"]))
    rep_path = out_dir / f"rep_{int(rep):04d}.json"
    if rep_path.exists():
        return {"rep": int(rep), "seed": int(seed), "skipped": True}

    cfg: InjectionRecoveryConfig = _G["cfg"]
    H0_grid = np.asarray(_G["H0_grid"], dtype=float)
    log_alpha_grid = np.asarray(_G["log_alpha_grid"], dtype=float)
    det: ToyDetectConfig = _G["det"]
    pop_q_min = float(_G["pop_q_min"])
    n_events = int(_G["n_events"])
    h0_true_mode = str(_G["h0_true_mode"])
    h0_true_fixed = float(_G["h0_true_fixed"])

    rng = np.random.default_rng(int(seed))
    if h0_true_mode == "fixed":
        h0_true = float(h0_true_fixed)
    elif h0_true_mode == "uniform":
        h0_true = float(rng.uniform(float(H0_grid[0]), float(H0_grid[-1])))
    else:
        raise ValueError("Unknown h0_true_mode.")

    cfg_rep = replace(cfg, h0_true=float(h0_true))

    draws = _draw_detected_toy_events(
        rng=rng,
        n_events=int(n_events),
        h0_true=float(h0_true),
        omega_m0=float(cfg_rep.omega_m0),
        omega_k0=float(cfg_rep.omega_k0),
        z_max=float(cfg_rep.z_max),
        pop_z_mode=str(cfg_rep.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(cfg_rep.pop_z_k),
        pop_mass_mode=str(cfg_rep.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(cfg_rep.pop_m1_alpha),
        pop_m_min=float(cfg_rep.pop_m_min),
        pop_m_max=float(cfg_rep.pop_m_max),
        pop_q_beta=float(cfg_rep.pop_q_beta),
        pop_q_min=float(pop_q_min),
        pop_m_taper_delta=float(cfg_rep.pop_m_taper_delta),
        pop_m_peak=float(cfg_rep.pop_m_peak),
        pop_m_peak_sigma=float(cfg_rep.pop_m_peak_sigma),
        pop_m_peak_frac=float(cfg_rep.pop_m_peak_frac),
        det=det,
    )

    truths: list[SyntheticEventTruth] = []
    for i, (z, m1, m2, dL_true, snr_true) in enumerate(draws, start=1):
        q_true = float(m2 / m1)
        mc_src = float(_chirp_mass_from_m1_q(np.array([m1]), np.array([q_true]))[0])
        mc_det_true = float(mc_src * (1.0 + float(z)))
        pdet_true = float(_pdet_from_snr(np.array([float(snr_true)]), det)[0])

        if str(cfg_rep.pe_obs_mode) == "truth":
            dL_obs = float(dL_true)
            mc_det_obs = float(mc_det_true)
            q_obs = float(q_true)
        else:
            snr = float(max(float(snr_true), 1e-6))
            if str(cfg_rep.dl_sigma_mode) == "constant":
                dl_log_sigma = float(cfg_rep.dl_frac_sigma0)
            else:
                dl_log_sigma = float(cfg_rep.dl_frac_sigma0) / snr
                dl_log_sigma = max(dl_log_sigma, float(cfg_rep.dl_frac_sigma_floor))
            dl_log_sigma = float(np.clip(dl_log_sigma, 1e-4, 2.0))
            mc_log_sigma = float(np.clip(float(cfg_rep.mc_frac_sigma0) / snr, 1e-4, 0.5))
            q_sigma = float(np.clip(float(cfg_rep.q_sigma0), 1e-4, 1.0))

            dL_obs = float(np.exp(rng.normal(loc=np.log(float(dL_true)), scale=dl_log_sigma)))
            mc_det_obs = float(np.exp(rng.normal(loc=np.log(float(mc_det_true)), scale=mc_log_sigma)))
            # cheap truncated normal for q
            q_obs = float(np.clip(rng.normal(loc=float(q_true), scale=q_sigma), 0.05, 1.0))

        truths.append(
            SyntheticEventTruth(
                event=f"TOY_{int(rep):04d}_{int(i):03d}",
                z=float(z),
                dL_mpc_true=float(dL_true),
                dL_mpc_obs=float(dL_obs),
                m1_source=float(m1),
                m2_source=float(m2),
                chirp_mass_det_true=float(mc_det_true),
                chirp_mass_det_obs=float(mc_det_obs),
                mass_ratio_true=float(q_true),
                mass_ratio_obs=float(q_obs),
                snr_net_opt_fid=float(snr_true),
                dL_mpc_fid=float(dL_true),
                snr_net_opt_true=float(snr_true),
                p_det_true=float(pdet_true),
            )
        )

    pe_by_event = {t.event: synthesize_pe_posterior_samples(truth=t, cfg=cfg_rep, rng=rng) for t in truths}

    res_off = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event=pe_by_event,
        H0_grid=np.asarray(H0_grid, dtype=float),
        omega_m0=float(cfg_rep.omega_m0),
        omega_k0=float(cfg_rep.omega_k0),
        z_max=float(cfg_rep.z_max),
        cache_dir=None,
        n_processes=1,
        include_pdet_in_event_term=False,
        pdet_model=None,
        pop_z_include_h0_volume_scaling=bool(cfg_rep.pop_z_include_h0_volume_scaling),
        injections=None,
        ifar_threshold_yr=1.0,
        det_model="threshold",
        snr_threshold=None,
        snr_binned_nbins=200,
        mchirp_binned_nbins=20,
        weight_mode="none",
        pop_z_mode=str(cfg_rep.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(cfg_rep.pop_z_k),
        pop_mass_mode=str(cfg_rep.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(cfg_rep.pop_m1_alpha),
        pop_m_min=float(cfg_rep.pop_m_min),
        pop_m_max=float(cfg_rep.pop_m_max),
        pop_q_beta=float(cfg_rep.pop_q_beta),
        pop_m_taper_delta=float(cfg_rep.pop_m_taper_delta),
        pop_m_peak=float(cfg_rep.pop_m_peak),
        pop_m_peak_sigma=float(cfg_rep.pop_m_peak_sigma),
        pop_m_peak_frac=float(cfg_rep.pop_m_peak_frac),
        importance_smoothing=str(cfg_rep.importance_smoothing),  # type: ignore[arg-type]
        importance_truncate_tau=cfg_rep.importance_truncate_tau,
        event_qc_mode=str(cfg_rep.event_qc_mode),  # type: ignore[arg-type]
        event_min_finite_frac=float(cfg_rep.event_min_finite_frac),
        selection_include_h0_volume_scaling=False,
        prior="uniform",
    )

    n_used = int(res_off.get("n_events", 0))
    logL_sum_events_rel = np.asarray(res_off.get("logL_sum_events_rel", []), dtype=float)
    if logL_sum_events_rel.shape != H0_grid.shape:
        raise ValueError("Selection-off result missing/invalid logL_sum_events_rel.")

    post_on = _h0_grid_posterior_from_logL_rel(H0_grid, logL_sum_events_rel - float(n_used) * log_alpha_grid)
    res_on = dict(res_off)
    res_on.update(
        {
            "log_alpha_grid": [float(x) for x in log_alpha_grid.tolist()],
            "logL_H0_rel": list(post_on["logL_H0_rel"]),
            "posterior": list(post_on["posterior"]),
            "H0_map": float(post_on["H0_map"]),
            "H0_map_index": int(post_on["H0_map_index"]),
            "H0_map_at_edge": bool(post_on["H0_map_at_edge"]),
            "summary": dict(post_on["summary"]),
            "selection_alpha": {"note": "toy alpha grid (MC)"},
            "selection_alpha_grid": [float(math.exp(x)) for x in log_alpha_grid.tolist()],
            "selection_ifar_threshold_yr": 1.0,
            "gate2_pass": bool((not post_on["H0_map_at_edge"]) and (int(res_off.get("n_events_skipped", 0)) == 0)),
        }
    )

    out: dict[str, Any] = {
        "manifest": {"seed": int(seed), "config": cfg_rep.__dict__, "h0_grid": [float(x) for x in H0_grid.tolist()]},
        "truths": [t.to_jsonable() for t in truths],
        "summary": {
            "h0_true": float(cfg_rep.h0_true),
            "n_events_truth": int(len(truths)),
            "n_events_used_selection_off": int(res_off.get("n_events", -1)),
            "n_events_skipped_selection_off": int(res_off.get("n_events_skipped", -1)),
            "n_events_used_selection_on": int(res_on.get("n_events", -1)),
            "n_events_skipped_selection_on": int(res_on.get("n_events_skipped", -1)),
            "bias_map_selection_on": float(res_on["H0_map"]) - float(cfg_rep.h0_true),
            "bias_p50_selection_on": float(res_on["summary"]["p50"]) - float(cfg_rep.h0_true),
            "selection_off": {"H0_map": float(res_off["H0_map"]), "summary": dict(res_off["summary"])},
            "selection_on": {"H0_map": float(res_on["H0_map"]), "summary": dict(res_on["summary"])},
        },
        "gr_h0_selection_off": res_off,
        "gr_h0_selection_on": res_on,
    }

    tmp = rep_path.with_suffix(".json.tmp")
    _write_json(tmp, out)
    tmp.replace(rep_path)

    return {"rep": int(rep), "seed": int(seed), "h0_true": float(h0_true), "skipped": False}


def _load_u_from_rep(rep_path: Path) -> dict[str, Any]:
    d = json.loads(rep_path.read_text())
    s = dict(d.get("summary", {}))
    h0_true = _safe_float(s.get("h0_true"))
    on = dict(d.get("gr_h0_selection_on", {}))
    off = dict(d.get("gr_h0_selection_off", {}))
    h0_grid = _as_float_array(on.get("H0_grid", off.get("H0_grid", [])), name="H0_grid")
    u_on = _cdf_at(h0_grid, _as_float_array(on.get("posterior", []), name="posterior"), float(h0_true))
    u_off = _cdf_at(h0_grid, _as_float_array(off.get("posterior", []), name="posterior"), float(h0_true))
    return {"rep": int(rep_path.stem.split("_")[-1]), "h0_true": float(h0_true), "u_h0_off": float(u_off), "u_h0_on": float(u_on)}


def _ks_d_uniform(u: np.ndarray) -> float:
    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    if u.size == 0:
        return float("nan")
    u = np.clip(u, 0.0, 1.0)
    u_sorted = np.sort(u)
    n = int(u_sorted.size)
    i = np.arange(1, n + 1, dtype=float)
    d_plus = float(np.max(i / n - u_sorted))
    d_minus = float(np.max(u_sorted - (i - 1.0) / n))
    return float(max(d_plus, d_minus))


def main() -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    ap = argparse.ArgumentParser(description="Toy closed-loop SBC suite for Gate-2 GR(H0) (no injection file).")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: outputs/gate2_toy_sbc_suite_<UTCSTAMP>).")

    ap.add_argument("--h0-min", type=float, default=40.0)
    ap.add_argument("--h0-max", type=float, default=100.0)
    ap.add_argument("--h0-n", type=int, default=121)
    ap.add_argument("--h0-true-mode", choices=["fixed", "uniform"], default="uniform")
    ap.add_argument("--h0-true", type=float, default=70.0)

    ap.add_argument("--n-rep", type=int, default=64)
    ap.add_argument("--n-events", type=int, default=25)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--n-proc", type=int, default=0, help="Replicate parallelism (0 => use all available).")

    ap.add_argument("--omega-m0", type=float, default=0.31)
    ap.add_argument("--omega-k0", type=float, default=0.0)
    ap.add_argument("--z-max", type=float, default=0.62)

    ap.add_argument("--pop-z-mode", choices=["none", "comoving_uniform", "comoving_powerlaw"], default="comoving_uniform")
    ap.add_argument("--pop-z-k", type=float, default=0.0)
    ap.add_argument("--pop-z-include-h0-volume-scaling", action="store_true", default=False)

    ap.add_argument(
        "--pop-mass-mode",
        choices=["powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"],
        default="powerlaw_peak_q_smooth",
    )
    ap.add_argument("--pop-m1-alpha", type=float, default=2.3)
    ap.add_argument("--pop-m-min", type=float, default=5.0)
    ap.add_argument("--pop-m-max", type=float, default=80.0)
    ap.add_argument("--pop-q-beta", type=float, default=0.0)
    ap.add_argument("--pop-q-min", type=float, default=0.05)
    ap.add_argument("--pop-m-taper-delta", type=float, default=3.0)
    ap.add_argument("--pop-m-peak", type=float, default=35.0)
    ap.add_argument("--pop-m-peak-sigma", type=float, default=5.0)
    ap.add_argument("--pop-m-peak-frac", type=float, default=0.1)

    ap.add_argument("--det-model", choices=["threshold", "logistic"], default="logistic")
    ap.add_argument("--snr-threshold", type=float, default=8.0)
    ap.add_argument("--snr-width", type=float, default=0.5)
    ap.add_argument("--snr0", type=float, default=25_000.0)
    ap.add_argument("--mc-ref", type=float, default=30.0)
    ap.add_argument("--mc-exp", type=float, default=5.0 / 6.0)

    ap.add_argument("--alpha-n-mc", type=int, default=200_000, help="MC samples for alpha(H0) (default 200k).")
    ap.add_argument("--alpha-seed", type=int, default=1234)

    ap.add_argument("--pe-obs-mode", choices=["truth", "noisy"], default="noisy")
    ap.add_argument("--pe-n-samples", type=int, default=12_000)
    ap.add_argument("--pe-synth-mode", choices=["naive_gaussian", "prior_resample", "likelihood_resample"], default="likelihood_resample")
    ap.add_argument("--pe-prior-resample-n-candidates", type=int, default=200_000)
    ap.add_argument("--pe-seed", type=int, default=0)
    ap.add_argument("--dl-frac-sigma0", type=float, default=0.25)
    ap.add_argument("--dl-frac-sigma-floor", type=float, default=0.05)
    ap.add_argument("--dl-sigma-mode", choices=["constant", "snr"], default="snr")
    ap.add_argument("--mc-frac-sigma0", type=float, default=0.02)
    ap.add_argument("--q-sigma0", type=float, default=0.08)
    ap.add_argument("--pe-prior-dl-expr", type=str, default="PowerLaw(alpha=2.0, minimum=1.0, maximum=20000.0)")
    ap.add_argument("--pe-prior-chirp-mass-expr", type=str, default="UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)")
    ap.add_argument("--pe-prior-mass-ratio-expr", type=str, default="UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)")

    args = ap.parse_args()

    H0_grid = np.linspace(float(args.h0_min), float(args.h0_max), int(args.h0_n))
    if H0_grid.size < 2 or not np.all(np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("Invalid H0 grid.")

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("outputs") / f"gate2_toy_sbc_suite_{_utc_stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir = out_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    det = ToyDetectConfig(
        model=str(args.det_model),  # type: ignore[arg-type]
        snr_threshold=float(args.snr_threshold),
        snr_width=float(args.snr_width),
        snr0=float(args.snr0),
        mc_ref=float(args.mc_ref),
        mc_exp=float(args.mc_exp),
    )

    cfg = InjectionRecoveryConfig(
        h0_true=float(args.h0_true),
        omega_m0=float(args.omega_m0),
        omega_k0=float(args.omega_k0),
        z_max=float(args.z_max),
        det_model="threshold",
        snr_binned_nbins=200,
        mchirp_binned_nbins=20,
        selection_ifar_thresh_yr=1.0,
        pop_z_mode=str(args.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(args.pop_z_k),
        pop_z_include_h0_volume_scaling=bool(args.pop_z_include_h0_volume_scaling),
        pop_mass_mode=str(args.pop_mass_mode),  # type: ignore[arg-type]
        pop_m1_alpha=float(args.pop_m1_alpha),
        pop_m_min=float(args.pop_m_min),
        pop_m_max=float(args.pop_m_max),
        pop_q_beta=float(args.pop_q_beta),
        pop_m_taper_delta=float(args.pop_m_taper_delta),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        weight_mode="none",
        pe_obs_mode=str(args.pe_obs_mode),  # type: ignore[arg-type]
        pe_n_samples=int(args.pe_n_samples),
        pe_synth_mode=str(args.pe_synth_mode),  # type: ignore[arg-type]
        pe_prior_resample_n_candidates=int(args.pe_prior_resample_n_candidates),
        pe_seed=int(args.pe_seed),
        dl_frac_sigma0=float(args.dl_frac_sigma0),
        dl_frac_sigma_floor=float(args.dl_frac_sigma_floor),
        dl_sigma_mode=str(args.dl_sigma_mode),  # type: ignore[arg-type]
        mc_frac_sigma0=float(args.mc_frac_sigma0),
        q_sigma0=float(args.q_sigma0),
        pe_prior_dL_expr=str(args.pe_prior_dl_expr),
        pe_prior_chirp_mass_expr=str(args.pe_prior_chirp_mass_expr),
        pe_prior_mass_ratio_expr=str(args.pe_prior_mass_ratio_expr),
        event_qc_mode="skip",
        event_min_finite_frac=0.0,
        importance_smoothing="none",
    )

    manifest = {
        "created_utc": _utc_stamp(),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "n_rep": int(args.n_rep),
        "n_events": int(args.n_events),
        "seed0": int(args.seed0),
        "h0_true_mode": str(args.h0_true_mode),
        "h0_true_fixed": float(args.h0_true),
        "cfg": cfg.__dict__,
        "toy_det": {
            "model": str(args.det_model),
            "snr_threshold": float(args.snr_threshold),
            "snr_width": float(args.snr_width),
            "snr0": float(args.snr0),
            "mc_ref": float(args.mc_ref),
            "mc_exp": float(args.mc_exp),
        },
        "alpha_mc": {"n_mc": int(args.alpha_n_mc), "seed": int(args.alpha_seed)},
    }
    _write_json(out_dir / "manifest.json", manifest)

    alpha = _toy_alpha_grid_mc(
        seed=int(args.alpha_seed),
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
        pop_m_taper_delta=float(args.pop_m_taper_delta),
        pop_m_peak=float(args.pop_m_peak),
        pop_m_peak_sigma=float(args.pop_m_peak_sigma),
        pop_m_peak_frac=float(args.pop_m_peak_frac),
        det=det,
        n_mc=int(args.alpha_n_mc),
    )
    log_alpha_grid = np.log(np.clip(alpha, 1e-300, np.inf))
    _write_json(out_dir / "alpha.json", {"H0_grid": [float(x) for x in H0_grid.tolist()], "alpha_grid": [float(x) for x in alpha.tolist()]})

    existing = sorted(json_dir.glob("rep_*.json"))
    (out_dir / "progress.json").write_text(json.dumps({"n_done": len(existing), "n_target": int(args.n_rep), "updated_utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n")

    todo: list[tuple[int, int]] = []
    for r in range(int(args.n_rep)):
        rep = r + 1
        seed = int(args.seed0) + r
        if (json_dir / f"rep_{rep:04d}.json").exists():
            continue
        todo.append((rep, seed))

    n_proc = int(args.n_proc)
    if n_proc <= 0:
        try:
            n_proc = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        except Exception:
            n_proc = int(os.cpu_count() or 1)
    n_proc = max(1, min(n_proc, int(args.n_rep)))

    state = {
        "cfg": cfg,
        "H0_grid": H0_grid,
        "log_alpha_grid": log_alpha_grid,
        "det": det,
        "pop_q_min": float(args.pop_q_min),
        "n_events": int(args.n_events),
        "h0_true_mode": str(args.h0_true_mode),
        "h0_true_fixed": float(args.h0_true),
        "json_dir": str(json_dir),
    }

    if todo:
        with ProcessPoolExecutor(max_workers=int(n_proc), initializer=_init_worker, initargs=(state,)) as ex:
            futs = {ex.submit(_run_rep, rep, seed): (rep, seed) for rep, seed in todo}
            for fut in as_completed(futs):
                rep, seed = futs[fut]
                _ = fut.result()
                done = sorted(json_dir.glob("rep_*.json"))
                (out_dir / "progress.json").write_text(json.dumps({"n_done": len(done), "n_target": int(args.n_rep), "updated_utc": _utc_stamp()}, indent=2, sort_keys=True) + "\n")
                if len(done) % 10 == 0 or len(done) == int(args.n_rep):
                    print(f"[toy_sbc] done {len(done)}/{int(args.n_rep)}", flush=True)

    # Summarize.
    reps = sorted(json_dir.glob("rep_*.json"))
    rows = [_load_u_from_rep(p) for p in reps]
    out_csv = tables_dir / "suite_summary.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rep", "h0_true", "u_h0_off", "u_h0_on"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    u_on = np.asarray([_safe_float(r.get("u_h0_on")) for r in rows], dtype=float)
    u_off = np.asarray([_safe_float(r.get("u_h0_off")) for r in rows], dtype=float)
    agg = {
        "n_rep": int(len(rows)),
        "u_h0_on_mean": float(np.nanmean(u_on)),
        "u_h0_on_ks": float(_ks_d_uniform(u_on)),
        "u_h0_off_mean": float(np.nanmean(u_off)),
        "u_h0_off_ks": float(_ks_d_uniform(u_off)),
    }
    _write_json(tables_dir / "suite_aggregate.json", agg)
    print(f"[toy_sbc] u_on_mean={agg['u_h0_on_mean']:.4f}  ks={agg['u_h0_on_ks']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

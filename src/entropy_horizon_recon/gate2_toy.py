from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .constants import PhysicalConstants
from .dark_siren_h0 import (
    CalibratedDetectionModel,
    _build_lcdm_distance_cache,
    compute_gr_h0_posterior_grid_hierarchical_pe,
)
from .dark_sirens_hierarchical_pe import GWTCPeHierarchicalSamples
from .gwtc_pe_priors import parse_gwtc_analytic_prior


@dataclass(frozen=True)
class ToyPopulationConfig:
    omega_m0: float = 0.31
    omega_k0: float = 0.0
    z_max: float = 0.6
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "comoving_uniform"
    pop_z_k: float = 0.0


@dataclass(frozen=True)
class ToyPePriorConfig:
    # Use a broad-but-not-pathological distance prior for toy cancellation checks.
    # A dL^2 prior over a huge range makes the cancellation test a heavy-tailed importance
    # sampling problem (posterior=prior), which converges slowly and can look non-flat due
    # to MC variance at high H0.
    pe_prior_dL_expr: str = "Uniform(minimum=1.0, maximum=6000.0)"
    pe_prior_chirp_mass_expr: str = "UniformInComponentsChirpMass(minimum=2.0, maximum=200.0)"
    pe_prior_mass_ratio_expr: str = "UniformInComponentsMassRatio(minimum=0.05, maximum=1.0)"


@dataclass(frozen=True)
class ToyCancellationConfig:
    """Known-answer Gate-2 cancellation test config.

    Constructs PE samples from the PE prior (flat likelihood) and enables the optional
    `include_pdet_in_event_term` diagnostic. In that setup, each event-term integral should
    be proportional to alpha(H0), and the selection-normalized total should be ~flat in H0.
    """

    pop: ToyPopulationConfig = ToyPopulationConfig()
    pe_prior: ToyPePriorConfig = ToyPePriorConfig()

    n_events: int = 8
    pe_n_samples: int = 10_000
    seed: int = 0

    h0_min: float = 40.0
    h0_max: float = 200.0
    h0_n: int = 161

    snr_threshold: float = 8.0
    snr_norm: float = 25_000.0  # event-term assumes snr(dL) = snr_norm / dL
    dL_ref_mpc: float = 1000.0  # used only to store (snr_ref, dL_ref) in PE samples

    alpha_mc_samples: int = 200_000


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def _build_z_sampler(
    *,
    omega_m0: float,
    omega_k0: float,
    z_max: float,
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"],
    pop_z_k: float,
    n_grid: int = 5001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (z_grid, cdf, f_grid) for sampling z from the configured population.

    Note: the returned PDF is shape-only; any overall H0^{-3} scaling cancels in selection-normalized
    likelihoods and is irrelevant for sampling.
    """
    z_max = float(z_max)
    if not (np.isfinite(z_max) and z_max > 0.0):
        raise ValueError("z_max must be finite and positive.")
    n_grid = int(n_grid)
    if n_grid < 100:
        raise ValueError("n_grid too small.")

    dist_cache = _build_lcdm_distance_cache(z_max=z_max, omega_m0=float(omega_m0), omega_k0=float(omega_k0), n_grid=n_grid)
    z = np.asarray(dist_cache.z_grid, dtype=float)
    f = np.asarray(dist_cache.f_grid, dtype=float)

    if pop_z_mode == "none":
        pdf = np.ones_like(z, dtype=float)
    else:
        om = float(omega_m0)
        ok = float(omega_k0)
        ol = 1.0 - om - ok
        Ez2 = om * (1.0 + z) ** 3 + ok * (1.0 + z) ** 2 + ol
        Ez = np.sqrt(np.clip(Ez2, 1e-30, np.inf))

        # p(z) ∝ (dV/dz)/(1+z) shape ∝ f(z)^2 / [(1+z)^3 E(z)]
        pdf = (f**2) / (np.clip(1.0 + z, 1e-12, np.inf) ** 3 * np.clip(Ez, 1e-30, np.inf))
        if pop_z_mode == "comoving_powerlaw":
            pdf = pdf * np.clip(1.0 + z, 1e-12, np.inf) ** float(pop_z_k)
        elif pop_z_mode != "comoving_uniform":  # pragma: no cover
            raise ValueError("Unknown pop_z_mode.")

    pdf = np.clip(pdf, 0.0, np.inf)
    pdf[0] = 0.0  # avoid z=0 exactly
    area = float(np.trapezoid(pdf, z))
    if not (np.isfinite(area) and area > 0.0):
        raise ValueError("Invalid z pdf normalization.")
    pdf = pdf / area

    cdf = np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(z))
    cdf = np.concatenate([[0.0], cdf])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return z, cdf, f


def sample_population_redshifts(
    rng: np.random.Generator,
    *,
    n: int,
    pop: ToyPopulationConfig,
    n_grid: int = 5001,
) -> np.ndarray:
    """Sample redshifts from the configured toy population."""
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive.")
    z_grid, z_cdf, _f = _build_z_sampler(
        omega_m0=float(pop.omega_m0),
        omega_k0=float(pop.omega_k0),
        z_max=float(pop.z_max),
        pop_z_mode=str(pop.pop_z_mode),  # type: ignore[arg-type]
        pop_z_k=float(pop.pop_z_k),
        n_grid=int(n_grid),
    )
    u = rng.random(n).astype(float)
    z = np.interp(u, z_cdf, z_grid).astype(float)
    z = np.clip(z, 0.0, float(pop.z_max))
    z[z <= 0.0] = np.nan
    return z


def build_prior_only_pe_samples(
    rng: np.random.Generator,
    *,
    n_samples: int,
    priors: ToyPePriorConfig,
    snr_norm: float,
    dL_ref_mpc: float,
) -> GWTCPeHierarchicalSamples:
    """Construct synthetic PE samples where posterior == PE prior (flat likelihood)."""
    n_samples = int(n_samples)
    if n_samples < 1000:
        raise ValueError("n_samples must be >= 1000 for stable hierarchical reweighting.")

    spec_dL, prior_dL = parse_gwtc_analytic_prior(str(priors.pe_prior_dL_expr))
    spec_mc, prior_mc = parse_gwtc_analytic_prior(str(priors.pe_prior_chirp_mass_expr))
    spec_q, prior_q = parse_gwtc_analytic_prior(str(priors.pe_prior_mass_ratio_expr))

    dL = np.asarray(prior_dL.sample(rng, size=n_samples), dtype=float)
    mc_det = np.asarray(prior_mc.sample(rng, size=n_samples), dtype=float)
    q = np.asarray(prior_q.sample(rng, size=n_samples), dtype=float)

    log_pi_dL = np.asarray(prior_dL.logpdf(dL), dtype=float)
    log_pi_mc = np.asarray(prior_mc.logpdf(mc_det), dtype=float)
    log_pi_q = np.asarray(prior_q.logpdf(q), dtype=float)
    if not (np.all(np.isfinite(log_pi_dL)) and np.all(np.isfinite(log_pi_mc)) and np.all(np.isfinite(log_pi_q))):
        raise ValueError("Non-finite log π in generated prior-only PE samples.")

    prior_spec = {
        "luminosity_distance": {"expr": spec_dL.expr, "class_name": spec_dL.class_name, "kwargs": spec_dL.kwargs},
        "chirp_mass": {"expr": spec_mc.expr, "class_name": spec_mc.class_name, "kwargs": spec_mc.kwargs},
        "mass_ratio": {"expr": spec_q.expr, "class_name": spec_q.class_name, "kwargs": spec_q.kwargs},
    }

    dL_ref_mpc = float(dL_ref_mpc)
    if not (np.isfinite(dL_ref_mpc) and dL_ref_mpc > 0.0):
        raise ValueError("dL_ref_mpc must be finite and positive.")
    snr_norm = float(snr_norm)
    if not (np.isfinite(snr_norm) and snr_norm > 0.0):
        raise ValueError("snr_norm must be finite and positive.")
    snr_ref = float(snr_norm / dL_ref_mpc)

    return GWTCPeHierarchicalSamples(
        file="<synthetic-prior-only>",
        analysis="PRIOR_ONLY",
        n_total=int(n_samples),
        n_used=int(n_samples),
        dL_mpc=dL,
        chirp_mass_det=mc_det,
        mass_ratio=q,
        log_pi_dL=log_pi_dL,
        log_pi_chirp_mass=log_pi_mc,
        log_pi_mass_ratio=log_pi_q,
        prior_spec=prior_spec,
        snr_net_opt_ref=float(snr_ref),
        dL_mpc_ref=float(dL_ref_mpc),
    )


def alpha_h0_grid_mc(
    *,
    H0_grid: np.ndarray,
    pop: ToyPopulationConfig,
    z_samples: np.ndarray,
    snr_norm: float,
    snr_threshold: float,
) -> np.ndarray:
    """Compute alpha(H0) for toy threshold detectability with snr(dL)=snr_norm/dL."""
    H0_grid = np.asarray(H0_grid, dtype=float)
    if H0_grid.ndim != 1 or H0_grid.size < 2:
        raise ValueError("H0_grid must be 1D with >=2 points.")
    if np.any(~np.isfinite(H0_grid)) or np.any(H0_grid <= 0.0):
        raise ValueError("H0_grid must be finite and > 0.")

    z = np.asarray(z_samples, dtype=float)
    z = z[np.isfinite(z)]
    z = z[(z > 0.0) & (z <= float(pop.z_max))]
    if z.size < 1000:
        raise ValueError("Need >=1000 finite z samples for alpha MC.")

    dist_cache = _build_lcdm_distance_cache(z_max=float(pop.z_max), omega_m0=float(pop.omega_m0), omega_k0=float(pop.omega_k0))
    fz = np.asarray(dist_cache.f(z), dtype=float)
    if not np.all(np.isfinite(fz)) or np.any(fz <= 0.0):
        raise ValueError("Invalid f(z) for alpha MC.")

    constants = PhysicalConstants()
    snr_norm = float(snr_norm)
    snr_threshold = float(snr_threshold)
    if not (np.isfinite(snr_norm) and snr_norm > 0.0):
        raise ValueError("snr_norm must be finite and > 0.")
    if not (np.isfinite(snr_threshold) and snr_threshold > 0.0):
        raise ValueError("snr_threshold must be finite and > 0.")

    alpha = np.zeros((H0_grid.size,), dtype=float)
    for i, H0 in enumerate(H0_grid.tolist()):
        dL = (constants.c_km_s / float(H0)) * fz
        snr = snr_norm / np.clip(dL, 1e-12, np.inf)
        alpha[i] = float(np.mean((snr > snr_threshold).astype(float)))
    return alpha


def run_gate2_toy_cancellation(
    cfg: ToyCancellationConfig,
    *,
    out_dir: str | Path | None = None,
    n_processes: int | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(cfg.seed))

    pop = cfg.pop
    H0_grid = np.linspace(float(cfg.h0_min), float(cfg.h0_max), int(cfg.h0_n))

    # PE samples: posterior == prior (flat likelihood).
    pe = build_prior_only_pe_samples(
        rng=rng,
        n_samples=int(cfg.pe_n_samples),
        priors=cfg.pe_prior,
        snr_norm=float(cfg.snr_norm),
        dL_ref_mpc=float(cfg.dL_ref_mpc),
    )
    pe_by_event = {f"TOY_PRIOR_{i+1:03d}": pe for i in range(int(cfg.n_events))}

    res = compute_gr_h0_posterior_grid_hierarchical_pe(
        pe_by_event=pe_by_event,
        H0_grid=H0_grid,
        omega_m0=float(pop.omega_m0),
        omega_k0=float(pop.omega_k0),
        z_max=float(pop.z_max),
        cache_dir=None,
        n_processes=n_processes,
        include_pdet_in_event_term=True,
        pdet_model=CalibratedDetectionModel(det_model="threshold", snr_threshold=float(cfg.snr_threshold)),
        injections=None,
        ifar_threshold_yr=1.0,
        det_model="threshold",
        snr_threshold=float(cfg.snr_threshold),
        snr_binned_nbins=200,
        mchirp_binned_nbins=20,
        weight_mode="none",
        pop_z_mode=str(pop.pop_z_mode),  # type: ignore[arg-type]
        pop_z_powerlaw_k=float(pop.pop_z_k),
        pop_mass_mode="none",
        pop_m1_alpha=2.3,
        pop_m_min=5.0,
        pop_m_max=80.0,
        pop_q_beta=0.0,
        event_qc_mode="fail",
        event_min_finite_frac=1.0,
        prior="uniform",
    )

    # Selection: alpha(H0) from MC under the same z population and threshold p_det model.
    z_samp = sample_population_redshifts(rng, n=int(cfg.alpha_mc_samples), pop=pop)
    alpha = alpha_h0_grid_mc(
        H0_grid=H0_grid,
        pop=pop,
        z_samples=z_samp,
        snr_norm=float(cfg.snr_norm),
        snr_threshold=float(cfg.snr_threshold),
    )
    log_alpha = np.log(np.clip(alpha, 1e-300, np.inf))

    logL_sum_rel = np.asarray(res["logL_sum_events_rel"], dtype=float)
    n_events = int(res["n_events"])
    logL_total_rel = logL_sum_rel - float(n_events) * log_alpha

    # Flatness diagnostic: range of logL_total_rel after removing a constant offset.
    logL_total_rel = logL_total_rel - float(np.nanmax(logL_total_rel))
    flat_range = float(np.nanmax(logL_total_rel) - np.nanmin(logL_total_rel))

    out = {
        "method": "gate2_toy_cancellation",
        "cfg": asdict(cfg),
        "H0_grid": [float(x) for x in H0_grid.tolist()],
        "alpha_grid": [float(x) for x in alpha.tolist()],
        "log_alpha_grid": [float(x) for x in log_alpha.tolist()],
        "res_event_term": res,
        "logL_total_rel": [float(x) for x in logL_total_rel.tolist()],
        "diagnostics": {"flat_range_logL": float(flat_range)},
    }

    if out_dir is not None:
        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        _write_json(out_path / "toy_cancellation.json", out)
    return out

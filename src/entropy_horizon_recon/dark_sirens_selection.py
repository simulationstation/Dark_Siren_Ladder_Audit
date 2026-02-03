from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
import time
from pathlib import Path
from typing import Literal

import numpy as np

from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class O3InjectionSet:
    """Minimal injection set for computing a distance-sensitive selection proxy alpha(model).

    Supports both the older GWTC-3 population injection release (O3a BBHpop) and the
    GWTC-3-era O3 search-sensitivity injection summaries (LIGO-T2100113).
    """

    path: str
    ifar_threshold_yr: float
    z: np.ndarray  # (N,)
    dL_mpc_fid: np.ndarray  # (N,) fiducial luminosity distance used for injection
    snr_net_opt: np.ndarray  # (N,) optimal network SNR (H+L)
    found_ifar: np.ndarray  # (N,) boolean
    sampling_pdf: np.ndarray  # (N,) injection sampling pdf (not used by default)
    mixture_weight: np.ndarray  # (N,) additional weight for mixture-model injection sets (defaults to 1)
    m1_source: np.ndarray  # (N,) Msun
    m2_source: np.ndarray  # (N,) Msun
    total_generated: int
    analysis_time_s: float


def load_o3_injections(path: str | Path, *, ifar_threshold_yr: float = 1.0) -> O3InjectionSet:
    """Load an O3 injection summary file into a common minimal format.

    We treat an injection as "found" if any available `ifar_*` dataset exceeds the given threshold.
    """
    from h5py import File

    path = Path(path).resolve()
    with File(path, "r") as f:
        inj = f["injections"]
        z = np.asarray(inj["redshift"][()], dtype=float)
        dL = np.asarray(inj["distance"][()], dtype=float)

        # Prefer a precomputed network SNR if available (present in the O3 sensitivity release).
        if "optimal_snr_net" in inj:
            snr_net = np.asarray(inj["optimal_snr_net"][()], dtype=float)
        else:
            # Fallback (GWTC-3 pop injections): compute network SNR from H+L.
            snr_h = np.asarray(inj["optimal_snr_h"][()], dtype=float)
            snr_l = np.asarray(inj["optimal_snr_l"][()], dtype=float)
            snr_net = np.sqrt(np.clip(snr_h, 0.0, np.inf) ** 2 + np.clip(snr_l, 0.0, np.inf) ** 2)

        # Build found mask from all available iFAR datasets.
        ifar_keys = [str(k) for k in inj.keys() if str(k).startswith("ifar_")]
        if not ifar_keys:
            raise KeyError(f"{path}: no injections/ifar_* datasets found.")
        found = np.zeros_like(z, dtype=bool)
        for k in ifar_keys:
            vals = np.asarray(inj[k][()], dtype=float)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            found |= vals > float(ifar_threshold_yr)

        sampling_pdf = np.asarray(inj["sampling_pdf"][()], dtype=float) if "sampling_pdf" in inj else np.ones_like(z, dtype=float)
        mixture_weight = np.asarray(inj["mixture_weight"][()], dtype=float) if "mixture_weight" in inj else np.ones_like(z, dtype=float)

        m1 = np.asarray(inj["mass1_source"][()], dtype=float)
        m2 = np.asarray(inj["mass2_source"][()], dtype=float)

        # Attributes are provided both at root and under injections in the O3 sensitivity release.
        total_generated = f.attrs.get("total_generated", inj.attrs.get("total_generated"))
        if total_generated is None:
            n_acc = int(inj.attrs.get("n_accepted", z.size))
            n_rej = int(inj.attrs.get("n_rejected", 0))
            total_generated = int(n_acc + n_rej)
        total_generated = int(total_generated)

        analysis_time_s = f.attrs.get("analysis_time_s", inj.attrs.get("analysis_time_s"))
        if analysis_time_s is None:
            raise KeyError(f"{path}: missing analysis_time_s attribute.")
        analysis_time_s = float(analysis_time_s)

    return O3InjectionSet(
        path=str(path),
        ifar_threshold_yr=float(ifar_threshold_yr),
        z=z,
        dL_mpc_fid=dL,
        snr_net_opt=snr_net,
        found_ifar=np.asarray(found, dtype=bool),
        sampling_pdf=sampling_pdf,
        mixture_weight=mixture_weight,
        m1_source=m1,
        m2_source=m2,
        total_generated=total_generated,
        analysis_time_s=analysis_time_s,
    )


def load_o3a_bbhpop_injections(path: str | Path, *, ifar_threshold_yr: float = 1.0) -> O3InjectionSet:
    """Backward-compatible alias for `load_o3_injections`."""
    return load_o3_injections(path, ifar_threshold_yr=ifar_threshold_yr)


# Backward-compatible type name (used across older scripts/modules).
O3aBbhInjectionSet = O3InjectionSet


def infer_observing_segment_from_event_name(event: str) -> Literal["o3a", "o3b", "other", "unknown"]:
    """Infer O3 segment from the event name `GWYYMMDD_HHMMSS`."""
    s = str(event)
    if not (s.startswith("GW") and "_" in s and len(s) >= 15):
        return "unknown"
    yymmdd = s[2:8]
    if not yymmdd.isdigit():
        return "unknown"

    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])
    # O3 events are 2019-2020.
    year = 2000 + yy
    try:
        d = date(year, mm, dd)
    except Exception:
        return "unknown"

    # Approximate O3 segment boundaries (UTC dates).
    o3a_start = date(2019, 4, 1)
    o3a_end = date(2019, 10, 1)
    o3b_start = date(2019, 11, 1)
    o3b_end = date(2020, 3, 27)

    if o3a_start <= d < o3a_end:
        return "o3a"
    if o3b_start <= d <= o3b_end:
        return "o3b"
    return "other"


def resolve_o3_sensitivity_injection_file(
    *,
    events: list[str],
    base_dir: str | Path = "data/cache/gw/zenodo",
    record_id: int = 7890437,
    population: Literal["mixture", "bbhpop"] = "mixture",
    auto_download: bool = True,
) -> Path:
    """Resolve (and optionally download) an O3 sensitivity injection summary file.

    Strategy:
      - If all events are O3a -> use the O3a segment file.
      - If all events are O3b -> use the O3b segment file.
      - Otherwise -> use the full O3 file.
    """
    base_dir = Path(base_dir).expanduser().resolve()
    rec_dir = base_dir / str(int(record_id))
    rec_dir.mkdir(parents=True, exist_ok=True)

    segs = {infer_observing_segment_from_event_name(ev) for ev in events}
    if segs == {"o3a"}:
        seg = "o3a"
    elif segs == {"o3b"}:
        seg = "o3b"
    else:
        seg = "o3"

    # GPS window tags used in the Zenodo filenames for segment-specific files.
    gps_tag = {"o3a": "1238166018-15843600", "o3b": "1256655642-12905976"}

    pop = str(population)
    base = f"endo3_{pop}-LIGO-T2100113-v12"
    filename = f"{base}.hdf5" if seg == "o3" else f"{base}-{gps_tag[seg]}.hdf5"
    dest = rec_dir / filename
    if dest.exists():
        return dest

    if not auto_download:
        raise FileNotFoundError(f"Missing injection file {dest} (auto_download=False).")

    # Download directly from Zenodo.
    url = f"https://zenodo.org/records/{int(record_id)}/files/{filename}?download=1"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        from urllib.request import urlretrieve  # noqa: S310

        last_err: Exception | None = None
        for attempt in range(1, 4):
            try:
                urlretrieve(url, tmp)  # type: ignore[misc]
                tmp.replace(dest)
                last_err = None
                break
            except Exception as e:  # pragma: no cover
                last_err = e
                time.sleep(2.0 * attempt)
        if last_err is not None:
            raise last_err
        (rec_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "record_id": int(record_id),
                    "source": "zenodo",
                    "url": url,
                    "file": filename,
                    "population": pop,
                    "segment": seg,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except Exception:
                pass

    if not dest.exists():
        raise FileNotFoundError(f"Download failed: {dest} was not created.")
    return dest


def calibrate_snr_threshold_match_count(*, snr_net_opt: np.ndarray, found_ifar: np.ndarray) -> float:
    """Pick an SNR threshold so that `snr > thresh` matches the IFAR-found count (fiducial cosmology)."""
    snr_net_opt = np.asarray(snr_net_opt, dtype=float)
    found_ifar = np.asarray(found_ifar, dtype=bool)
    if snr_net_opt.shape != found_ifar.shape:
        raise ValueError("snr_net_opt and found_ifar must have matching shapes.")
    n = int(snr_net_opt.size)
    k = int(np.sum(found_ifar))
    if n == 0 or k == 0:
        raise ValueError("No injections / no found injections; cannot calibrate SNR threshold.")

    # Choose a threshold so that exactly k injections are above it (up to ties).
    q = 1.0 - float(k) / float(n)
    # Use 'higher' to be conservative: it gives a threshold such that fraction > thresh is <= (1-q).
    return float(np.quantile(snr_net_opt, q, method="higher"))


def calibrate_snr_threshold_match_found_fraction(*, snr_net_opt: np.ndarray, found_ifar: np.ndarray, weights: np.ndarray) -> float:
    """Pick an SNR threshold so that weighted frac(snr > thresh) matches weighted found_ifar fraction.

    This is the weighted analogue of calibrate_snr_threshold_match_count(). It is useful when
    computing alpha(model) as an importance-weighted expectation over an injection set.
    """
    snr_net_opt = np.asarray(snr_net_opt, dtype=float)
    found_ifar = np.asarray(found_ifar, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if snr_net_opt.shape != found_ifar.shape or snr_net_opt.shape != weights.shape:
        raise ValueError("snr_net_opt, found_ifar, and weights must have matching shapes.")
    if snr_net_opt.size == 0:
        raise ValueError("No injections; cannot calibrate SNR threshold.")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and strictly positive.")

    # Target weighted detection fraction under the fiducial distances.
    wsum = float(np.sum(weights))
    if wsum <= 0.0:
        raise ValueError("Sum of weights is non-positive.")
    target = float(np.sum(weights * found_ifar.astype(float)) / wsum)
    if target <= 0.0:
        raise ValueError("No weighted found injections; cannot calibrate SNR threshold.")

    # Compute weighted quantile q such that P(snr > thresh) ~= target => P(snr <= thresh) ~= 1-target.
    q = 1.0 - target
    order = np.argsort(snr_net_opt)
    snr_sorted = snr_net_opt[order]
    w_sorted = weights[order]
    cdf = np.cumsum(w_sorted) / wsum
    # leftmost index where CDF >= q
    i = int(np.searchsorted(cdf, q, side="left"))
    i = max(0, min(i, snr_sorted.size - 1))
    return float(snr_sorted[i])


@dataclass(frozen=True)
class SelectionAlphaResult:
    """Per-draw selection normalization proxy."""

    method: str
    convention: str
    det_model: str
    weight_mode: str
    mu_det_distance: str
    z_max: float
    snr_threshold: float | None
    snr_offset: float
    n_injections_used: int
    alpha_mu: np.ndarray  # (n_draws,)
    alpha_gr: np.ndarray  # (n_draws,)

    def to_json(self) -> str:
        # Avoid dumping huge arrays in logs by default; callers can serialize explicitly if desired.
        return json.dumps({k: v for k, v in asdict(self).items() if k not in ("alpha_mu", "alpha_gr")}, indent=2)


def compute_selection_alpha_from_injections(
    *,
    post: MuForwardPosterior,
    injections: O3InjectionSet,
    convention: Literal["A", "B"] = "A",
    z_max: float,
    snr_threshold: float | None = None,
    det_model: Literal["threshold", "snr_binned", "snr_mchirp_binned"] = "snr_binned",
    snr_offset: float = 0.0,
    snr_binned_nbins: int = 200,
    mchirp_binned_nbins: int = 20,
    weight_mode: Literal["none", "inv_sampling_pdf"] = "none",
    mu_det_distance: Literal["gw", "em"] = "gw",
    # Optional population importance-weighting (used in addition to inv_sampling_pdf).
    pop_z_mode: Literal["none", "comoving_uniform", "comoving_powerlaw"] = "none",
    pop_z_powerlaw_k: float = 0.0,
    pop_mass_mode: Literal["none", "powerlaw_q", "powerlaw_q_smooth", "powerlaw_peak_q_smooth"] = "none",
    pop_m1_alpha: float = 2.3,
    pop_m_min: float = 5.0,
    pop_m_max: float = 80.0,
    pop_q_beta: float = 0.0,
    pop_m_taper_delta: float = 0.0,
    pop_m_peak: float = 35.0,
    pop_m_peak_sigma: float = 5.0,
    pop_m_peak_frac: float = 0.1,
    inj_mass_pdf_coords: Literal["m1m2", "m1q"] = "m1m2",
) -> SelectionAlphaResult:
    """Compute alpha(model) via distance-rescaled injection SNRs.

    This is an intentionally *simple* and *explicit* selection proxy:
      - Calibrate a network SNR threshold from IFAR-found injections at the fiducial distances.
      - For each posterior draw, rescale SNR by dL_fid / dL_model(z) and recompute the found fraction.

    It captures the leading dependence of detectability on luminosity distance without committing
    to a full population model (masses/spins/rate evolution).

    The `snr_mchirp_binned` option extends the proxy to a simple 2D p_det(SNR, Mchirp_det) table,
    which can reduce selection-normalization bias when detectability depends strongly on mass.
    """
    z_max = float(z_max)
    if z_max <= 0.0:
        raise ValueError("z_max must be positive.")
    snr_offset = float(snr_offset)
    if not np.isfinite(snr_offset):
        raise ValueError("snr_offset must be finite.")
    mu_det_distance = str(mu_det_distance)
    if mu_det_distance not in ("gw", "em"):
        raise ValueError("mu_det_distance must be one of {'gw','em'}.")

    z = np.asarray(injections.z, dtype=float)
    dL_fid = np.asarray(injections.dL_mpc_fid, dtype=float)
    snr = np.asarray(injections.snr_net_opt, dtype=float)
    m1 = np.asarray(injections.m1_source, dtype=float)
    m2 = np.asarray(injections.m2_source, dtype=float)
    found_ifar = np.asarray(injections.found_ifar, dtype=bool)

    # Ensure z stays within the posterior grid (no extrapolation in selection proxy).
    z_hi = float(min(z_max, float(post.z_grid[-1])))
    m = (
        np.isfinite(z)
        & (z > 0.0)
        & (z <= z_hi)
        & np.isfinite(dL_fid)
        & (dL_fid > 0.0)
        & np.isfinite(snr)
        & (snr > 0.0)
        & np.isfinite(m1)
        & np.isfinite(m2)
        & (m1 > 0.0)
        & (m2 > 0.0)
        & (m2 <= m1)
    )
    if not np.any(m):
        raise ValueError(f"No injections remain after z/dL/SNR cuts (z_hi={z_hi}).")
    z = z[m]
    dL_fid = dL_fid[m]
    snr = snr[m]
    m1 = m1[m]
    m2 = m2[m]
    found_ifar = found_ifar[m]

    # Build per-injection weights w_i for estimating alpha(model).
    # These weights implement the usual population-vs-injection importance weighting:
    #   w_i ∝ p_pop(theta_i) / p_inj(theta_i)
    # with optional p_pop factors (redshift evolution, mass model).
    w = np.ones_like(z, dtype=float)
    # Mixture-model weight (close to 1 for O3 mixture injection releases).
    mw = np.asarray(getattr(injections, "mixture_weight", np.ones_like(injections.z)), dtype=float)[m]
    if mw.shape != z.shape:
        raise ValueError("injections.mixture_weight must match injections.z shape.")
    w = w * mw

    if weight_mode == "none":
        pass
    elif weight_mode == "inv_sampling_pdf":
        pdf = np.asarray(injections.sampling_pdf, dtype=float)[m]
        if not np.all(np.isfinite(pdf)) or np.any(pdf <= 0.0):
            raise ValueError("sampling_pdf contains non-finite or non-positive values; cannot use inv_sampling_pdf weighting.")
        w = w / pdf
    else:
        raise ValueError("Unknown weight_mode.")

    if pop_z_mode != "none":
        # Approximate comoving-volume prior using a fixed LCDM background (Planck-ish).
        # This is a population prior factor, not a cosmology-dependent selection effect.
        H0 = 67.7  # km/s/Mpc
        om0 = 0.31
        c = 299792.458  # km/s
        z_grid = np.linspace(0.0, float(np.max(z)), 5001)
        Ez = np.sqrt(om0 * (1.0 + z_grid) ** 3 + (1.0 - om0))
        # Comoving distance D_C(z) = c/H0 * integral dz/E(z).
        dc = (c / H0) * np.cumsum(np.concatenate([[0.0], 0.5 * (1.0 / Ez[1:] + 1.0 / Ez[:-1]) * np.diff(z_grid)]))
        # dV/dz/dOmega = c/H(z) * D_C^2, with H(z)=H0 E(z).
        dVdz = (c / (H0 * np.interp(z, z_grid, Ez))) * (np.interp(z, z_grid, dc) ** 2)
        # Source-frame time dilation factor 1/(1+z).
        base = dVdz / (1.0 + z)
        if pop_z_mode == "comoving_uniform":
            w = w * base
        elif pop_z_mode == "comoving_powerlaw":
            k = float(pop_z_powerlaw_k)
            # R(z) ∝ (1+z)^k
            w = w * base * (1.0 + z) ** k
        else:
            raise ValueError("Unknown pop_z_mode.")

    if pop_mass_mode != "none":
        if weight_mode == "inv_sampling_pdf":
            # Population mass models are parameterized in (m1_source, q=m2/m1). If the injection
            # `sampling_pdf` is provided in (m1_source, m2_source) coordinates, we must include
            # the Jacobian dm2 = m1 dq to convert to the (m1,q) density used by the population.
            if inj_mass_pdf_coords == "m1m2":
                w = w / np.clip(m1, 1e-300, np.inf)
            elif inj_mass_pdf_coords == "m1q":
                pass
            else:
                raise ValueError("Unknown inj_mass_pdf_coords (expected 'm1m2' or 'm1q').")

        alpha = float(pop_m1_alpha)
        mmin = float(pop_m_min)
        mmax = float(pop_m_max)
        beta_q = float(pop_q_beta)
        if not (np.isfinite(alpha) and np.isfinite(mmin) and np.isfinite(mmax) and np.isfinite(beta_q)):
            raise ValueError("Non-finite population-mass parameters.")
        if not (mmin > 0.0 and mmax > mmin):
            raise ValueError("Invalid pop_m_min/pop_m_max.")
        q = np.clip(m2 / m1, 1e-6, 1.0)
        if pop_mass_mode == "powerlaw_q":
            good_m = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= m1)
            w = w * good_m.astype(float) * (m1 ** (-alpha)) * (q ** beta_q)
        elif pop_mass_mode == "powerlaw_q_smooth":
            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for pop_mass_mode=powerlaw_q_smooth.")
            # Stable sigmoid: 0.5*(1+tanh(x/2))
            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 0.0, 1.0)
            w = w * taper * (m1 ** (-alpha)) * (q ** beta_q)
        elif pop_mass_mode == "powerlaw_peak_q_smooth":
            delta = float(pop_m_taper_delta)
            if not (np.isfinite(delta) and delta > 0.0):
                raise ValueError("pop_m_taper_delta must be finite and > 0 for pop_mass_mode=powerlaw_peak_q_smooth.")

            mp = float(pop_m_peak)
            sig = float(pop_m_peak_sigma)
            f_peak = float(pop_m_peak_frac)
            if not (np.isfinite(mp) and mp > 0.0 and np.isfinite(sig) and sig > 0.0):
                raise ValueError("pop_m_peak and pop_m_peak_sigma must be finite and positive for pop_mass_mode=powerlaw_peak_q_smooth.")
            if not (np.isfinite(f_peak) and 0.0 <= f_peak <= 1.0):
                raise ValueError("pop_m_peak_frac must be finite and in [0,1] for pop_mass_mode=powerlaw_peak_q_smooth.")

            def _sig(x: np.ndarray) -> np.ndarray:
                return 0.5 * (1.0 + np.tanh(0.5 * x))

            t1 = _sig((m1 - mmin) / delta) * _sig((mmax - m1) / delta)
            t2 = _sig((m2 - mmin) / delta) * _sig((mmax - m2) / delta)
            taper = np.clip(t1 * t2, 1e-300, 1.0)

            log_q = beta_q * np.log(np.clip(q, 1e-300, np.inf))
            log_taper = np.log(taper)
            log_pl = -alpha * np.log(np.clip(m1, 1e-300, np.inf)) + log_q + log_taper
            log_peak = -0.5 * ((m1 - mp) / sig) ** 2 - np.log(sig) + log_q + log_taper
            if f_peak <= 0.0:
                log_mass = log_pl
            elif f_peak >= 1.0:
                log_mass = log_peak
            else:
                log_mass = np.logaddexp(np.log(1.0 - f_peak) + log_pl, np.log(f_peak) + log_peak)

            m_ok = np.isfinite(log_mass)
            if not np.any(m_ok):
                raise ValueError("All peak-mass weights were non-finite.")
            log_mass = log_mass - float(np.nanmax(log_mass[m_ok]))
            w = w * np.exp(log_mass)
        else:
            raise ValueError("Unknown pop_mass_mode.")

    # Drop injections with zero/invalid weights (outside population support, z~0, etc.).
    good_w = np.isfinite(w) & (w > 0.0)
    if not np.any(good_w):
        raise ValueError("All injections received zero/invalid weights; check population/injection weighting configuration.")
    if not np.all(good_w):
        z = z[good_w]
        dL_fid = dL_fid[good_w]
        snr = snr[good_w]
        m1 = m1[good_w]
        m2 = m2[good_w]
        w = w[good_w]
        found_ifar = found_ifar[good_w]

    if snr_threshold is None:
        # Calibrate to match the IFAR-found count in the *same* z window.
        if det_model == "threshold":
            # For a threshold model we need an explicit threshold.
            # The threshold is a property of the detection pipeline and should not depend on population weights.
            snr_threshold = calibrate_snr_threshold_match_count(snr_net_opt=snr, found_ifar=found_ifar)
        elif det_model in ("snr_binned", "snr_mchirp_binned"):
            snr_threshold = None
        else:
            raise ValueError("Unknown det_model.")

    # Build an empirical detection-probability curve p_det(snr) if requested.
    # We intentionally keep this simple and monotone, since it is used only as a selection proxy.
    pdet_edges: np.ndarray | None = None
    pdet_vals: np.ndarray | None = None
    mchirp_edges: np.ndarray | None = None
    pdet_grid: np.ndarray | None = None
    if det_model == "snr_binned":
        nb = int(snr_binned_nbins)
        if nb < 20:
            raise ValueError("snr_binned_nbins too small (need >= 20).")
        # Quantile bins: ensures roughly equal counts per bin.
        edges = np.quantile(snr, np.linspace(0.0, 1.0, nb + 1))
        edges = np.unique(edges)
        if edges.size < 10:
            raise ValueError("Too few unique SNR edges for snr_binned; injection SNR distribution seems degenerate.")
        # Digitize to bins [edges[i], edges[i+1]).
        bin_idx = np.clip(np.digitize(snr, edges) - 1, 0, edges.size - 2)
        p = np.zeros(edges.size - 1, dtype=float)
        for i in range(p.size):
            m_i = bin_idx == i
            if not np.any(m_i):
                p[i] = p[i - 1] if i > 0 else 0.0
                continue
            # Unweighted conditional detection fraction; this approximates pipeline behavior.
            p[i] = float(np.mean(found_ifar[m_i].astype(float)))
        # Enforce monotone non-decreasing.
        p = np.maximum.accumulate(np.clip(p, 0.0, 1.0))
        pdet_edges = edges
        pdet_vals = p
    elif det_model == "snr_mchirp_binned":
        nb_snr = int(snr_binned_nbins)
        if nb_snr < 20:
            raise ValueError("snr_binned_nbins too small (need >= 20).")
        nb_mc = int(mchirp_binned_nbins)
        if nb_mc < 4:
            raise ValueError("mchirp_binned_nbins too small (need >= 4).")

        mc_src = ((m1 * m2) ** (3.0 / 5.0)) / np.clip(m1 + m2, 1e-300, np.inf) ** (1.0 / 5.0)
        mc_det = np.asarray(mc_src * (1.0 + z), dtype=float)
        if not np.all(np.isfinite(mc_det)) or np.any(mc_det <= 0.0):
            raise ValueError("Non-finite/non-positive injection mchirp_det for snr_mchirp_binned.")

        snr_edges = np.quantile(snr, np.linspace(0.0, 1.0, nb_snr + 1))
        snr_edges = np.unique(snr_edges)
        if snr_edges.size < 10:
            raise ValueError("Too few unique SNR edges for snr_mchirp_binned; injection SNR distribution seems degenerate.")

        mc_edges = np.quantile(mc_det, np.linspace(0.0, 1.0, nb_mc + 1))
        mc_edges = np.unique(mc_edges)
        if mc_edges.size < 4:
            raise ValueError("Too few unique mchirp_det edges for snr_mchirp_binned; injection mass distribution seems degenerate.")

        snr_bin = np.clip(np.digitize(snr, snr_edges) - 1, 0, snr_edges.size - 2)
        mc_bin = np.clip(np.digitize(mc_det, mc_edges) - 1, 0, mc_edges.size - 2)

        n_mc_bins = int(mc_edges.size) - 1
        n_snr_bins = int(snr_edges.size) - 1
        pdet2 = np.zeros((n_mc_bins, n_snr_bins), dtype=float)
        for i_mc in range(n_mc_bins):
            m_i = mc_bin == i_mc
            if not np.any(m_i):
                continue
            for i_snr in range(n_snr_bins):
                m_ij = m_i & (snr_bin == i_snr)
                if not np.any(m_ij):
                    pdet2[i_mc, i_snr] = pdet2[i_mc, i_snr - 1] if i_snr > 0 else 0.0
                    continue
                pdet2[i_mc, i_snr] = float(np.mean(found_ifar[m_ij].astype(float)))
            pdet2[i_mc, :] = np.maximum.accumulate(np.clip(pdet2[i_mc, :], 0.0, 1.0))

        pdet_edges = snr_edges
        mchirp_edges = mc_edges
        pdet_grid = pdet2

    # Precompute model distances on the posterior z_grid for each draw, then interpolate to injection z.
    z_grid = np.asarray(post.z_grid, dtype=float)
    dL_em_grid = predict_dL_em(post, z_eval=z_grid)  # (n_draws, n_z)
    _, R_grid = predict_r_gw_em(post, z_eval=None, convention=convention)  # (n_draws, n_z)
    dL_gw_grid = dL_em_grid * np.asarray(R_grid, dtype=float)

    n_draws = int(dL_em_grid.shape[0])
    alpha_mu = np.empty((n_draws,), dtype=float)
    alpha_gr = np.empty((n_draws,), dtype=float)

    inj_mchirp_det: np.ndarray | None = None
    if det_model == "snr_mchirp_binned":
        mc_src = ((m1 * m2) ** (3.0 / 5.0)) / np.clip(m1 + m2, 1e-300, np.inf) ** (1.0 / 5.0)
        inj_mchirp_det = np.asarray(mc_src * (1.0 + z), dtype=float)
        if not np.all(np.isfinite(inj_mchirp_det)) or np.any(inj_mchirp_det <= 0.0):
            raise ValueError("Non-finite/non-positive injection mchirp_det after cuts.")

    for j in range(n_draws):
        dL_em = np.interp(z, z_grid, dL_em_grid[j])
        dL_gw = np.interp(z, z_grid, dL_gw_grid[j])
        dL_mu_det = dL_gw if mu_det_distance == "gw" else dL_em

        # Distance scaling: SNR ~ 1/dL.
        snr_gr = snr * (dL_fid / np.clip(dL_em, 1e-6, np.inf))
        snr_mu = snr * (dL_fid / np.clip(dL_mu_det, 1e-6, np.inf))
        # Nuisance perturbation: shift effective SNR before applying p_det.
        snr_gr_eff = snr_gr - snr_offset
        snr_mu_eff = snr_mu - snr_offset

        wsum = float(np.sum(w))
        if wsum <= 0.0:
            raise ValueError("Sum of selection weights is non-positive.")

        if det_model == "threshold":
            if snr_threshold is None:
                raise ValueError("snr_threshold must be provided or calibratable for det_model='threshold'.")
            alpha_gr[j] = float(np.sum(w * (snr_gr_eff > float(snr_threshold))) / wsum)
            alpha_mu[j] = float(np.sum(w * (snr_mu_eff > float(snr_threshold))) / wsum)
        elif det_model == "snr_binned":
            assert pdet_edges is not None and pdet_vals is not None
            # Evaluate piecewise-constant p_det at rescaled SNRs.
            idx_gr = np.clip(np.digitize(snr_gr_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
            idx_mu = np.clip(np.digitize(snr_mu_eff, pdet_edges) - 1, 0, pdet_vals.size - 1)
            p_gr = pdet_vals[idx_gr]
            p_mu = pdet_vals[idx_mu]
            alpha_gr[j] = float(np.sum(w * p_gr) / wsum)
            alpha_mu[j] = float(np.sum(w * p_mu) / wsum)
        elif det_model == "snr_mchirp_binned":
            assert pdet_edges is not None and mchirp_edges is not None and pdet_grid is not None and inj_mchirp_det is not None
            i_mc = np.clip(np.digitize(inj_mchirp_det, mchirp_edges) - 1, 0, pdet_grid.shape[0] - 1)
            i_gr = np.clip(np.digitize(snr_gr_eff, pdet_edges) - 1, 0, pdet_grid.shape[1] - 1)
            i_mu = np.clip(np.digitize(snr_mu_eff, pdet_edges) - 1, 0, pdet_grid.shape[1] - 1)
            p_gr = pdet_grid[i_mc, i_gr]
            p_mu = pdet_grid[i_mc, i_mu]
            alpha_gr[j] = float(np.sum(w * p_gr) / wsum)
            alpha_mu[j] = float(np.sum(w * p_mu) / wsum)
        else:
            raise ValueError("Unknown det_model.")

    return SelectionAlphaResult(
        method="o3_injections_snr_rescale",
        convention=str(convention),
        det_model=str(det_model),
        weight_mode=str(weight_mode),
        mu_det_distance=str(mu_det_distance),
        z_max=float(z_hi),
        snr_threshold=float(snr_threshold) if snr_threshold is not None else None,
        snr_offset=float(snr_offset),
        n_injections_used=int(z.size),
        alpha_mu=alpha_mu,
        alpha_gr=alpha_gr,
    )

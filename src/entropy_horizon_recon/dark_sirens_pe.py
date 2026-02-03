from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

import healpy as hp

from .gw_distance_priors import GWDistancePrior
from .sirens import MuForwardPosterior, predict_dL_em, predict_r_gw_em


@dataclass(frozen=True)
class GWTCPeSkySamples:
    """Minimal (sky, distance) posterior samples from a GWTC PEDataRelease file.

    `ra` and `dec` are in radians (as stored in GWTC-2.1/GWTC-3 PEDataRelease files).
    `dL_mpc` is in Mpc.
    """

    file: str
    analysis: str
    n_total: int
    n_used: int

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def _default_pe_analysis_preference(path: Path) -> list[str]:
    # GWTC-2.1 / GWTC-3 convention.
    if path.suffix == ".h5":
        return [
            "C01:Mixed",
            "C01:IMRPhenomXPHM",
            "C01:SEOBNRv4PHM",
        ]
    # GWTC-4.0 convention.
    if path.suffix == ".hdf5":
        return [
            "C00:Mixed",
            "C00:Mixed+XO4a",
            "C00:IMRPhenomXPHM-SpinTaylor",
            "C00:SEOBNRv5PHM",
            "C00:NRSur7dq4",
            "C00:IMRPhenomXO4a",
        ]
    return []


def load_gwtc_pe_sky_samples(
    *,
    path: str | Path,
    analysis: str | None = None,
    max_samples: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, GWTCPeSkySamples]:
    """Load (ra, dec, luminosity_distance) posterior samples from a GWTC PEDataRelease file."""
    path = Path(path).expanduser().resolve()
    try:
        import h5py  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: h5py is required to read PEDataRelease HDF5 files.") from e

    if max_samples is not None and int(max_samples) <= 0:
        raise ValueError("max_samples must be positive when provided.")

    with h5py.File(path, "r") as f:
        if analysis is None:
            # Prefer an analysis group that exists *and* has a large posterior sample set.
            keys = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
            if not keys:
                raise ValueError(f"{path}: no analysis groups found.")

            pref = [k for k in _default_pe_analysis_preference(path) if k in f]
            candidates = pref if pref else keys

            best = None
            best_n = -1
            for k in candidates:
                try:
                    n = int(f[k]["posterior_samples"]["ra"].shape[0])
                except Exception:
                    continue
                if n > best_n:
                    best_n = n
                    best = k
            analysis = best if best is not None else keys[0]
        if analysis not in f:
            keys = [str(k) for k in f.keys() if str(k) not in ("history", "version")]
            raise KeyError(f"{path}: analysis group '{analysis}' not found. Available: {keys}")

        dset = f[analysis]["posterior_samples"]
        ra = np.asarray(dset["ra"], dtype=float)
        dec = np.asarray(dset["dec"], dtype=float)
        dL = np.asarray(dset["luminosity_distance"], dtype=float)

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dL) & (dL > 0.0)
    if not np.any(m):
        raise ValueError(f"{path}: no finite (ra,dec,dL) samples in '{analysis}'.")

    ra = ra[m]
    dec = dec[m]
    dL = dL[m]

    n_total = int(ra.size)
    if n_total < 10_000:
        raise ValueError(f"{path}: too few samples in '{analysis}' ({n_total}).")

    if max_samples is not None and ra.size > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(ra.size, size=int(max_samples), replace=False)
        ra = ra[idx]
        dec = dec[idx]
        dL = dL[idx]

    meta = GWTCPeSkySamples(
        file=str(path),
        analysis=str(analysis),
        n_total=n_total,
        n_used=int(ra.size),
    )
    return ra, dec, dL, meta


@dataclass(frozen=True)
class PePixelDistanceHistogram:
    """A lightweight binned approximation to p(Ω, dL | d) built from PE samples.

    The joint posterior factorizes as:
      p(Ω, dL | d) ≈ prob_pix(Ω) * pdf(dL | pix, d)

    where `prob_pix` stores probability *mass* per HEALPix pixel, and `pdf_bins` stores a
    conditional distance density (per Mpc) in log-spaced distance bins.
    """

    nside: int
    nest: bool
    p_credible: float
    pix_sel: np.ndarray  # (n_pix_sel,)
    prob_pix: np.ndarray  # (n_pix_sel,)
    dL_edges: np.ndarray  # (n_bins+1,)
    pdf_bins: np.ndarray  # (n_pix_sel, n_bins)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "nside": int(self.nside),
            "nest": bool(self.nest),
            "p_credible": float(self.p_credible),
            "n_pix_sel": int(self.pix_sel.size),
            "dl_min_mpc": float(self.dL_edges[0]),
            "dl_max_mpc": float(self.dL_edges[-1]),
            "n_dl_bins": int(self.dL_edges.size - 1),
        }


def credible_region_pixels_from_pe_samples(
    *,
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    nside_out: int,
    p_credible: float,
    nest: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pix_sel, prob_sel, area_deg2) at nside_out covering cumulative prob >= p_credible."""
    if not (0.0 < float(p_credible) <= 1.0):
        raise ValueError("p_credible must be in (0,1].")
    nside_out = int(nside_out)
    if nside_out <= 0:
        raise ValueError("nside_out must be positive.")

    ra = np.asarray(ra_rad, dtype=float)
    dec = np.asarray(dec_rad, dtype=float)
    if ra.ndim != 1 or dec.ndim != 1 or ra.shape != dec.shape:
        raise ValueError("ra_rad/dec_rad must be 1D arrays with matching shapes.")

    theta = 0.5 * np.pi - dec
    phi = np.mod(ra, 2.0 * np.pi)
    ipix = np.asarray(hp.ang2pix(nside_out, theta, phi, nest=bool(nest)), dtype=np.int64)
    if ipix.size == 0:
        raise ValueError("No samples provided.")

    uniq, counts = np.unique(ipix, return_counts=True)
    prob = counts.astype(float) / float(np.sum(counts))
    order = np.argsort(prob)[::-1]
    csum = np.cumsum(prob[order])
    k = int(np.searchsorted(csum, float(p_credible), side="left")) + 1
    pix_sel = np.asarray(uniq[order[:k]], dtype=np.int64)
    prob_sel = np.asarray(prob[order[:k]], dtype=float)

    # Deterministic ordering by pixel index (keep prob aligned).
    srt = np.argsort(pix_sel)
    pix_sel = pix_sel[srt]
    prob_sel = prob_sel[srt]

    area_deg2 = float(pix_sel.size) * float(hp.nside2pixarea(nside_out, degrees=True))
    return pix_sel, prob_sel, area_deg2


def build_pe_pixel_distance_histogram(
    *,
    ra_rad: np.ndarray,
    dec_rad: np.ndarray,
    dL_mpc: np.ndarray,
    nside: int,
    p_credible: float,
    dl_nbins: int,
    dl_min_mpc: float | None = None,
    dl_max_mpc: float | None = None,
    dl_qmin: float = 0.001,
    dl_qmax: float = 0.999,
    dl_pad_factor: float = 1.2,
    dl_pseudocount: float = 0.05,
    dl_smooth_iters: int = 2,
    nest: bool = True,
) -> PePixelDistanceHistogram:
    """Build a per-pixel distance histogram for the PE posterior samples."""
    ra = np.asarray(ra_rad, dtype=float)
    dec = np.asarray(dec_rad, dtype=float)
    dL = np.asarray(dL_mpc, dtype=float)
    if ra.ndim != 1 or dec.ndim != 1 or dL.ndim != 1 or not (ra.shape == dec.shape == dL.shape):
        raise ValueError("ra/dec/dL must be 1D arrays with matching shapes.")
    if dL.size == 0:
        raise ValueError("No PE samples provided.")
    if not (0.0 < float(p_credible) <= 1.0):
        raise ValueError("p_credible must be in (0,1].")
    if int(dl_nbins) < 16:
        raise ValueError("dl_nbins too small (use >=16).")
    if not (0.0 <= float(dl_qmin) < float(dl_qmax) <= 1.0):
        raise ValueError("Invalid dl_qmin/dl_qmax.")
    if not (np.isfinite(float(dl_pad_factor)) and float(dl_pad_factor) > 1.0):
        raise ValueError("dl_pad_factor must be > 1.")
    if not (np.isfinite(float(dl_pseudocount)) and float(dl_pseudocount) >= 0.0):
        raise ValueError("dl_pseudocount must be finite and >= 0.")
    if int(dl_smooth_iters) < 0:
        raise ValueError("dl_smooth_iters must be >= 0.")

    # Select credible pixels first.
    pix_sel, prob_sel, _area_deg2 = credible_region_pixels_from_pe_samples(
        ra_rad=ra,
        dec_rad=dec,
        nside_out=int(nside),
        p_credible=float(p_credible),
        nest=bool(nest),
    )

    # Choose global log-spaced distance bins.
    dL_pos = dL[np.isfinite(dL) & (dL > 0.0)]
    if dL_pos.size < 100:
        raise ValueError("Too few finite positive distance samples.")
    if dl_min_mpc is None:
        lo = float(np.quantile(dL_pos, float(dl_qmin)))
        dl_min_mpc = max(1e-3, lo / float(dl_pad_factor))
    if dl_max_mpc is None:
        hi = float(np.quantile(dL_pos, float(dl_qmax)))
        dl_max_mpc = hi * float(dl_pad_factor)
    dl_min_mpc = float(dl_min_mpc)
    dl_max_mpc = float(dl_max_mpc)
    if not (np.isfinite(dl_min_mpc) and np.isfinite(dl_max_mpc) and dl_min_mpc > 0.0 and dl_max_mpc > dl_min_mpc):
        raise ValueError(f"Invalid dL bin bounds [{dl_min_mpc},{dl_max_mpc}].")
    edges = np.geomspace(dl_min_mpc, dl_max_mpc, int(dl_nbins) + 1)
    widths = np.diff(edges)

    # Pixel index per sample.
    theta = 0.5 * np.pi - dec
    phi = np.mod(ra, 2.0 * np.pi)
    ipix = np.asarray(hp.ang2pix(int(nside), theta, phi, nest=bool(nest)), dtype=np.int64)

    # Map coarse pixels -> row indices.
    npix = int(hp.nside2npix(int(nside)))
    pix_to_row = np.full((npix,), -1, dtype=np.int32)
    pix_to_row[pix_sel.astype(np.int64, copy=False)] = np.arange(pix_sel.size, dtype=np.int32)
    row = pix_to_row[ipix]
    m_pix = row >= 0
    row = row[m_pix].astype(np.int64, copy=False)
    dL_sel = dL[m_pix]

    # Distance bin index per sample.
    bin_idx = np.searchsorted(edges, dL_sel, side="right") - 1
    m_bin = (bin_idx >= 0) & (bin_idx < int(dl_nbins))
    row = row[m_bin]
    bin_idx = bin_idx[m_bin].astype(np.int64, copy=False)

    flat = row * int(dl_nbins) + bin_idx
    counts = np.bincount(flat, minlength=int(pix_sel.size) * int(dl_nbins)).astype(float)
    counts = counts.reshape((int(pix_sel.size), int(dl_nbins)))

    if float(dl_pseudocount) > 0.0:
        counts = counts + float(dl_pseudocount)

    # Simple, fast smoothing along distance bins to reduce zero-bin artifacts.
    # Kernel is [1,2,1]/4 applied `dl_smooth_iters` times.
    for _ in range(int(dl_smooth_iters)):
        c = counts
        out = np.empty_like(c)
        out[:, 1:-1] = 0.25 * c[:, :-2] + 0.5 * c[:, 1:-1] + 0.25 * c[:, 2:]
        out[:, 0] = 0.75 * c[:, 0] + 0.25 * c[:, 1]
        out[:, -1] = 0.75 * c[:, -1] + 0.25 * c[:, -2]
        counts = out

    tot = np.sum(counts, axis=1, keepdims=True)
    tot = np.clip(tot, 1e-30, np.inf)
    pdf = counts / tot / widths.reshape((1, -1))
    pdf = np.asarray(pdf, dtype=np.float32)

    return PePixelDistanceHistogram(
        nside=int(nside),
        nest=bool(nest),
        p_credible=float(p_credible),
        pix_sel=np.asarray(pix_sel, dtype=np.int64),
        prob_pix=np.asarray(prob_sel, dtype=np.float32),
        dL_edges=np.asarray(edges, dtype=np.float64),
        pdf_bins=pdf,
    )


def compute_dark_siren_logL_draws_from_pe_hist(
    *,
    event: str,
    pe: PePixelDistanceHistogram,
    post: MuForwardPosterior,
    z_gal: np.ndarray,
    w_gal: np.ndarray,
    ipix_gal: np.ndarray,
    convention: Literal["A", "B"] = "A",
    gw_distance_prior: GWDistancePrior | None = None,
    distance_mode: Literal["full", "spectral_only"] = "full",
    gal_chunk_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (logL_mu, logL_gr) per posterior draw using PE posterior samples (binned).

    distance_mode:
      - full: use per-pixel conditional distance densities p(dL | pix, data) from the PE histogram.
      - spectral_only: replace p(dL | pix, data) with the sky-marginal p(dL | data) (independent of pix),
        while keeping prob_pix sky weights. This removes sky–distance correlation as a control.
    """
    if not bool(pe.nest):
        raise ValueError("PE pixel histogram must use NESTED ordering.")
    z = np.asarray(z_gal, dtype=float)
    w = np.asarray(w_gal, dtype=float)
    ipix = np.asarray(ipix_gal, dtype=np.int64)
    if z.ndim != 1 or w.ndim != 1 or ipix.ndim != 1 or not (z.shape == w.shape == ipix.shape):
        raise ValueError("z_gal/w_gal/ipix_gal must be 1D arrays with matching shapes.")
    if z.size == 0:
        raise ValueError("No galaxies provided.")

    # Map pixels to row indices in the selected set.
    npix = int(hp.nside2npix(int(pe.nside)))
    pix_to_row = np.full((npix,), -1, dtype=np.int32)
    pix_to_row[np.asarray(pe.pix_sel, dtype=np.int64)] = np.arange(int(pe.pix_sel.size), dtype=np.int32)
    row = pix_to_row[ipix]
    good = (row >= 0) & np.isfinite(z) & (z > 0.0) & np.isfinite(w) & (w > 0.0)
    if not np.any(good):
        raise ValueError("All galaxies map outside the PE credible region (or have invalid z/w).")
    z = z[good]
    w = w[good]
    row = row[good].astype(np.int64, copy=False)

    prob = np.asarray(pe.prob_pix, dtype=float)[row]

    prior = gw_distance_prior or GWDistancePrior()
    edges = np.asarray(pe.dL_edges, dtype=float)
    nb = int(edges.size - 1)

    pdf_flat: np.ndarray | None = None
    pdf_1d: np.ndarray | None = None
    if distance_mode == "full":
        pdf_flat = np.asarray(pe.pdf_bins, dtype=float).reshape(-1)
    elif distance_mode == "spectral_only":
        # Build the sky-marginal distance density on the same bins:
        #   p(dL | data) ∝ Σ_pix prob_pix(pix) * p(dL | pix, data),
        # then renormalize to integrate to 1 over the selected pixels' dL support.
        p_pix = np.asarray(pe.prob_pix, dtype=float)
        pdf_bins = np.asarray(pe.pdf_bins, dtype=float)
        p_sum = float(np.sum(p_pix))
        if not (np.isfinite(p_sum) and p_sum > 0.0):
            raise ValueError("Invalid prob_pix sum while building spectral_only distance density.")

        pdf_1d = np.sum(p_pix.reshape((-1, 1)) * pdf_bins, axis=0) / p_sum  # (n_bins,)
        pdf_1d = np.clip(np.asarray(pdf_1d, dtype=float), 0.0, np.inf)

        widths = np.diff(edges)
        norm = float(np.sum(pdf_1d * widths))
        if not (np.isfinite(norm) and norm > 0.0):
            raise ValueError("Invalid sky-marginal distance density normalization in spectral_only mode.")
        pdf_1d = pdf_1d / norm
    else:
        raise ValueError("distance_mode must be 'full' or 'spectral_only'.")

    chunk = int(gal_chunk_size)
    if chunk <= 0:
        raise ValueError("gal_chunk_size must be positive.")

    n_draws = int(post.H_samples.shape[0])
    logL_mu = np.full((n_draws,), -np.inf, dtype=float)
    logL_gr = np.full((n_draws,), -np.inf, dtype=float)

    for a in range(0, z.size, chunk):
        b = min(z.size, a + chunk)
        z_c = np.asarray(z[a:b], dtype=float)
        w_c = np.asarray(w[a:b], dtype=float)
        row_c = np.asarray(row[a:b], dtype=np.int64)
        prob_c = np.asarray(prob[a:b], dtype=float)

        z_u, inv = np.unique(z_c, return_inverse=True)
        dL_em_u = predict_dL_em(post, z_eval=z_u)  # (n_draws, n_uniq)
        _, R_u = predict_r_gw_em(post, z_eval=z_u, convention=convention, allow_extrapolation=False)
        dL_gw_u = dL_em_u * np.asarray(R_u, dtype=float)

        dL_em = dL_em_u[:, inv]  # (n_draws, n_chunk)
        dL_gw = dL_gw_u[:, inv]

        logw = np.log(np.clip(w_c, 1e-30, np.inf))[None, :]
        logprob = np.log(np.clip(prob_c, 1e-300, np.inf))[None, :]

        def _chunk_logL(dL: np.ndarray) -> np.ndarray:
            dL = np.asarray(dL, dtype=float)
            bin_idx = np.searchsorted(edges, dL, side="right") - 1
            valid = (bin_idx >= 0) & (bin_idx < nb) & np.isfinite(dL) & (dL > 0.0)
            if distance_mode == "full":
                assert pdf_flat is not None
                # Linear indices into flattened [row, bin] table.
                lin = row_c.reshape((1, -1)) * nb + np.clip(bin_idx, 0, nb - 1)
                pdf = pdf_flat[lin]
                pdf = np.where(valid, pdf, 0.0)
            else:
                assert pdf_1d is not None
                pdf = pdf_1d[np.clip(bin_idx, 0, nb - 1)]
                pdf = np.where(valid, pdf, 0.0)

            logpdf = np.log(np.clip(pdf, 1e-300, np.inf))
            logprior = prior.log_pi_dL(np.clip(dL, 1e-6, np.inf))
            logterm = logw + logprob + logpdf - logprior
            # If the PE prior has hard support cutoffs (common), the posterior provides no
            # information outside that support and dividing by π_PE(dL)=0 is undefined.
            # Treat those regions as zero contribution rather than +∞ weight.
            logterm = np.where(np.isfinite(logprior), logterm, -np.inf)
            return logsumexp(logterm, axis=1)

        logL_mu = np.logaddexp(logL_mu, _chunk_logL(dL_gw))
        logL_gr = np.logaddexp(logL_gr, _chunk_logL(dL_em))

    return np.asarray(logL_mu, dtype=float), np.asarray(logL_gr, dtype=float)

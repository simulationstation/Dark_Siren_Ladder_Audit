#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Headless figure generation (safe on servers)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt

from entropy_horizon_recon.sirens import load_mu_forward_posterior, predict_r_gw_em


@dataclass(frozen=True)
class RunBundle:
    name: str
    run_dirs: list[Path]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _copy_tree_images(src_fig_dir: Path, dst_dir: Path, *, prefix: str) -> int:
    if not src_fig_dir.exists():
        return 0
    exts = {".png", ".pdf", ".jpg", ".jpeg", ".svg"}
    n = 0
    for p in sorted(src_fig_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        out = dst_dir / f"{prefix}__{p.name}"
        shutil.copy2(p, out)
        n += 1
    return n


def _load_logmu_logA_samples(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    npz = run_dir / "samples" / "logmu_logA_samples.npz"
    if not npz.exists():
        raise FileNotFoundError(f"Missing {npz}")
    with np.load(npz) as d:
        return np.asarray(d["logA"], dtype=float), np.asarray(d["logmu_samples"], dtype=float)


def _weighted_m_and_s(logA: np.ndarray, logmu_samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-draw m, slope, and weights w(logA).

    Weight definition: w_i ∝ 1/Var[logmu_i] across draws; normalized to sum to 1.
    """
    # Var across draws at each grid point
    v = np.var(logmu_samples, axis=0, ddof=1)
    v = np.where(v > 0, v, np.nan)
    w = 1.0 / v
    w = np.where(np.isfinite(w), w, 0.0)
    w = w / np.sum(w)

    # Weighted mean of g(x)=logmu
    m = logmu_samples @ w

    # Weighted linear slope in logA coordinate (equivalent to x up to a shift).
    xbar = float(np.sum(w * logA))
    x = logA - xbar
    varx = float(np.sum(w * x * x))
    if varx <= 0:
        raise RuntimeError("Degenerate logA grid for slope computation.")
    cov = (logmu_samples * x[None, :]) @ w
    s = cov / varx
    return m, s, w


def _save_mu_band_pdf(
    *,
    out_pdf: Path,
    logA: np.ndarray,
    logmu_samples: np.ndarray,
    title: str,
) -> None:
    q16, q50, q84 = np.percentile(logmu_samples, [16, 50, 84], axis=0)
    plt.figure(figsize=(6.5, 4.0))
    plt.plot(logA, q50, lw=1.8, label="median")
    plt.fill_between(logA, q16, q84, alpha=0.25, label="68% band")
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.6, linestyle="--", label="BH (log mu=0)")
    plt.xlabel(r"$\log A$")
    plt.ylabel(r"$\log \mu$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _save_ms_joint_pdf(*, out_pdf: Path, m: np.ndarray, s: np.ndarray, title: str) -> None:
    plt.figure(figsize=(6.0, 4.2))
    plt.scatter(m, s, s=6, alpha=0.25, edgecolors="none")
    plt.axvline(0.0, color="k", lw=1.0, alpha=0.6)
    plt.axhline(0.0, color="k", lw=1.0, alpha=0.6)
    plt.xlabel(r"$m$ (weighted mean of $\log\mu$)")
    plt.ylabel(r"$s$ (weighted slope of $\log\mu$ vs $\log A$)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _save_horizon_radius_mapping_demo(*, out_pdf: Path) -> None:
    # Simple fiducial LCDM H(z) for a demonstration plot.
    # This plot is intended as a mapping illustration, not a fit.
    c_km_s = 299_792.458
    H0 = 70.0
    om = 0.3
    ol = 1.0 - om
    z = np.linspace(0.0, 0.62, 400)
    H = H0 * np.sqrt(om * (1.0 + z) ** 3 + ol)

    # Three curvature values, chosen to be illustrative and small.
    ok_vals = [-0.05, 0.0, 0.05]

    plt.figure(figsize=(6.8, 4.2))
    for ok in ok_vals:
        denom = H**2 - ok * H0**2 * (1.0 + z) ** 2
        ok_mask = denom > 0
        RA = np.full_like(z, np.nan, dtype=float)
        RA[ok_mask] = c_km_s / np.sqrt(denom[ok_mask])  # units: Mpc if H is km/s/Mpc
        plt.plot(z, RA, lw=2.0, label=fr"$\Omega_{{k0}}={ok:+.2f}$")

    plt.xlabel(r"$z$")
    plt.ylabel(r"$R_A(z)$  [Mpc]")
    plt.title(r"Apparent-horizon radius mapping demo (fiducial $H(z)$)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _save_siren_ratio_band(*, out_pdf: Path, run_dir: Path, title: str) -> None:
    post = load_mu_forward_posterior(run_dir)
    z, R = predict_r_gw_em(post, convention="A")
    q16, q50, q84 = np.percentile(R, [16, 50, 84], axis=0)
    plt.figure(figsize=(6.8, 4.0))
    plt.plot(z, q50, lw=1.8, label="median")
    plt.fill_between(z, q16, q84, alpha=0.25, label="68% band")
    plt.axhline(1.0, color="k", lw=1.0, alpha=0.6, linestyle="--", label="GR (ratio=1)")
    plt.xlabel(r"$z$")
    plt.ylabel(r"$d_L^{GW}(z)/d_L^{EM}(z)$")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def _save_dotG_over_G_hist(*, out_pdf: Path, run_dir: Path, title: str) -> None:
    # Minimal embedding: G_eff ∝ mu(A).
    # dotG/G = d ln mu / dt. We estimate it at z=0 from finite differences on the posterior z_grid.
    post = load_mu_forward_posterior(run_dir)
    z, R = predict_r_gw_em(post, convention="A")
    mu_ratio = R**2  # mu(z)/mu(0)
    lnmu = np.log(mu_ratio)
    dz = float(z[1] - z[0])
    dlnmu_dz0 = (lnmu[:, 1] - lnmu[:, 0]) / dz

    # H(z=0) in 1/yr: H[km/s/Mpc] -> 1/s -> 1/yr
    H0_km_s_Mpc = post.H_samples[:, 0]
    Mpc_km = 3.085677581e19
    sec_per_yr = 31557600.0
    H0_per_yr = (H0_km_s_Mpc / Mpc_km) * sec_per_yr

    dotG_over_G = -(1.0 + 0.0) * H0_per_yr * dlnmu_dz0  # 1/yr

    plt.figure(figsize=(6.8, 4.0))
    plt.hist(dotG_over_G, bins=40, alpha=0.75, color="C0")
    plt.axvline(0.0, color="k", lw=1.0, alpha=0.6)
    plt.xlabel(r"$\dot G_{\rm eff}/G_{\rm eff}$  [1/yr]  (estimated at $z=0$)")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    root = _repo_root()
    out = root / "FINAL_FIGURES"
    _mkdir(out)

    # Subfolders (keep things organized).
    d_proxy = out / "proxy_suite"
    d_infoplus = out / "infoplus"
    d_holdout = out / "planck_lensing_holdout_pilot"
    d_void = out / "void_amp_test"
    d_desy3 = out / "des_y3"
    d_gen = out / "generated"
    for d in (d_proxy, d_infoplus, d_holdout, d_void, d_desy3, d_gen):
        _mkdir(d)

    # --- Copy run-generated figures (PNG/PDF/etc) ---
    bundles: list[RunBundle] = []

    # Proxy master suite used in the paper text (clean commit SHA 32b239...).
    base_proxy = root / "outputs" / "additions_20260129_120658UTC" / "M0+"
    proxy_runs = [
        base_proxy / "fulltest_proxy_clean_20260129",
        base_proxy / "fulltest_proxy_clean_20260129_M0_seed101",
        base_proxy / "fulltest_proxy_clean_20260129_M0_seed202",
        base_proxy / "fulltest_proxy_clean_20260129_M0_seed303",
        base_proxy / "fulltest_proxy_clean_20260129_M1_seed101",
        base_proxy / "fulltest_proxy_clean_20260129_M1_seed123",
        base_proxy / "fulltest_proxy_clean_20260129_M1_seed202",
        base_proxy / "fulltest_proxy_clean_20260129_M1_seed303",
        base_proxy / "fulltest_proxy_clean_20260129_M2_seed101",
        base_proxy / "fulltest_proxy_clean_20260129_M2_seed123",
        base_proxy / "fulltest_proxy_clean_20260129_M2_seed202",
        base_proxy / "fulltest_proxy_clean_20260129_M2_seed303",
        base_proxy / "extended_z" / "fulltest_proxy_clean_extz_20260129_M0_seed123",
    ]
    bundles.append(RunBundle(name="proxy", run_dirs=[p for p in proxy_runs if p.exists()]))

    # Info+ multistart suite (pilot-quality; 5 seeds).
    infoplus_base = root / "outputs" / "finalization" / "info_plus_full_256_detached_20260129_0825UTC"
    infoplus_runs = [infoplus_base / f"M0_start{s}" for s in (101, 202, 303, 404, 505)]
    bundles.append(RunBundle(name="infoplus", run_dirs=[p for p in infoplus_runs if p.exists()]))

    # Planck lensing holdout pilot plots (CLpp predictive bands).
    holdout_runs = [
        root / "outputs" / "planck_lensing_holdout_vl_off_full2_20260130_045353UTC",
        root / "outputs" / "planck_lensing_holdout_vl_off_agr2_20260130_051259UTC",
    ]
    bundles.append(RunBundle(name="holdout", run_dirs=[p for p in holdout_runs if p.exists()]))

    # Void amplitude proxy test figures.
    void_runs = [root / "outputs" / "void_amp_test_infoplus_extz_20260130_032539UTC"]
    bundles.append(RunBundle(name="void", run_dirs=[p for p in void_runs if p.exists()]))

    # DES Y3 contours reproduction.
    des_runs = [root / "outputs" / "des_y3a2_plots_20260129_1050UTC"]
    bundles.append(RunBundle(name="desy3", run_dirs=[p for p in des_runs if p.exists()]))

    copied = 0
    for b in bundles:
        for rd in b.run_dirs:
            fig = rd / "figures"
            if b.name == "proxy":
                dst = d_proxy
            elif b.name == "infoplus":
                dst = d_infoplus
            elif b.name == "holdout":
                dst = d_holdout
            elif b.name == "void":
                dst = d_void
            else:
                dst = d_desy3
            copied += _copy_tree_images(fig, dst, prefix=rd.name)

    # --- Generate missing paper figures from existing posterior artifacts ---
    # Use the clean proxy baseline run (seed 123) as the default source for derived plots.
    baseline_run = base_proxy / "fulltest_proxy_clean_20260129"
    if baseline_run.exists():
        # mu(A) posterior band in logA space + m/s joint posterior.
        logA, logmu = _load_logmu_logA_samples(baseline_run)
        m, s, _w = _weighted_m_and_s(logA, logmu)

        _save_mu_band_pdf(
            out_pdf=d_gen / "proxy_M0_seed123_logmu_band.pdf",
            logA=logA,
            logmu_samples=logmu,
            title="Proxy stack (M0, seed 123): log mu(log A) posterior band",
        )
        _save_ms_joint_pdf(
            out_pdf=d_gen / "proxy_M0_seed123_m_s_joint.pdf",
            m=m,
            s=s,
            title="Proxy stack (M0, seed 123): joint posterior of (m, s)",
        )

        # Standard-siren ratio band and implied dotG/G histogram (minimal alpha_M-only embedding).
        _save_siren_ratio_band(
            out_pdf=d_gen / "proxy_M0_seed123_dLgw_over_dLem_band.pdf",
            run_dir=baseline_run,
            title="Proxy stack (M0, seed 123): predicted dL_GW/dL_EM (alpha_M-only)",
        )
        _save_dotG_over_G_hist(
            out_pdf=d_gen / "proxy_M0_seed123_dotG_over_G_hist.pdf",
            run_dir=baseline_run,
            title="Proxy stack (M0, seed 123): implied dotG/G at z=0 (alpha_M-only; finite diff)",
        )

    # Horizon-radius mapping demo (needed by the paper text).
    _save_horizon_radius_mapping_demo(out_pdf=d_gen / "horizon_radius_mapping_demo.pdf")

    # Small manifest.
    (out / "README.md").write_text(
        "\n".join(
            [
                "# FINAL_FIGURES",
                "",
                "This folder aggregates figure assets (PNG/PDF/etc) used by the paper draft, plus",
                "a few derived figures generated from saved posterior artifacts.",
                "",
                "Subfolders:",
                f"- proxy_suite/: copied from {base_proxy}/<run>/figures/",
                f"- infoplus/: copied from {infoplus_base}/<run>/figures/",
                "- planck_lensing_holdout_pilot/: copied from outputs/planck_lensing_holdout_*/*figures/",
                "- void_amp_test/: copied from outputs/void_amp_test_*/figures/",
                "- des_y3/: copied from outputs/des_y3a2_plots_*/figures/",
                "- generated/: figures generated by scripts/build_final_figures.py",
                "",
                f"Total copied files: {copied}",
                "",
            ]
        )
        + "\n"
    )

    print(f"[final_figures] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


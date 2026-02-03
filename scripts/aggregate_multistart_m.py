#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any


def _read_json(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    # Accept list, or dict with "runs" or "rows".
    if isinstance(data, dict) and "runs" in data:
        data = data["runs"]
    elif isinstance(data, dict) and "rows" in data:
        data = data["rows"]
    if not isinstance(data, list):
        raise ValueError("JSON input must be a list of runs or a dict with key 'runs' or 'rows'.")
    return data


def _read_tsv(path: Path) -> List[Dict[str, Any]]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("TSV input is empty.")
    header = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        parts = ln.split("\t")
        if len(parts) != len(header):
            raise ValueError(f"TSV row has wrong column count: {ln}")
        row = {k: v for k, v in zip(header, parts)}
        rows.append(row)
    return rows


def _extract_runs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        # Allow various keys
        m_mean = None
        m_std = None
        seed = None
        for key in ("m_mean", "m", "m_mean_post"):
            if key in r:
                m_mean = r[key]
                break
        for key in ("m_std", "m_sigma", "m_std_post"):
            if key in r:
                m_std = r[key]
                break
        for key in ("seed", "Seed", "run", "id"):
            if key in r:
                seed = r[key]
                break
        if m_mean is None or m_std is None:
            raise ValueError(f"Missing m_mean or m_std in row: {r}")
        try:
            m_mean_f = float(m_mean)
            m_std_f = float(m_std)
        except Exception as e:
            raise ValueError(f"Non-numeric m_mean/m_std in row: {r}") from e
        if m_std_f <= 0:
            raise ValueError(f"m_std must be > 0. Got {m_std_f} in row: {r}")
        out.append({"seed": seed, "m_mean": m_mean_f, "m_std": m_std_f})
    return out


def _chi2_sf(x: float, k: int) -> float:
    # Survival function for chi-square using regularized gamma.
    # Implement with math and mpmath if available; fallback to scipy if installed.
    try:
        import mpmath as mp

        return float(mp.gammainc(k / 2.0, x / 2.0, mp.inf) / mp.gamma(k / 2.0))
    except Exception:
        try:
            from scipy.stats import chi2

            return float(chi2.sf(x, k))
        except Exception:
            raise RuntimeError("Need mpmath or scipy for chi-square survival function.")


def _phi(x: float) -> float:
    # Standard normal CDF via error function.
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate multi-start m statistics.")
    parser.add_argument("--in", dest="in_path", required=True, help="Path to summary JSON/TSV.")
    parser.add_argument("--outdir", required=True, help="Output directory (e.g. FINDINGS/).")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".json":
        rows = _read_json(in_path)
    elif in_path.suffix.lower() in {".tsv", ".txt"}:
        rows = _read_tsv(in_path)
    else:
        raise ValueError("Input must be .json or .tsv")

    runs = _extract_runs(rows)
    k = len(runs)
    if k < 2:
        raise ValueError("Need at least 2 runs for aggregation.")

    m = [r["m_mean"] for r in runs]
    s = [r["m_std"] for r in runs]
    w = [1.0 / (si * si) for si in s]

    sum_w = sum(w)
    m_hat = sum(wi * mi for wi, mi in zip(w, m)) / sum_w
    se_hat = math.sqrt(1.0 / sum_w)
    z = m_hat / se_hat
    p = 2.0 * (1.0 - _phi(abs(z)))
    ci_lo = m_hat - 1.96 * se_hat
    ci_hi = m_hat + 1.96 * se_hat

    Q = sum(wi * (mi - m_hat) ** 2 for wi, mi in zip(w, m))
    df = k - 1
    p_Q = _chi2_sf(Q, df)
    I2 = 0.0
    if Q > 0:
        I2 = max(0.0, (Q - df) / Q)

    re = None
    if I2 > 0.25:
        denom = sum_w - (sum(wi * wi for wi in w) / sum_w)
        tau2 = max(0.0, (Q - df) / denom) if denom > 0 else 0.0
        w_star = [1.0 / (si * si + tau2) for si in s]
        sum_w_star = sum(w_star)
        m_hat_re = sum(wi * mi for wi, mi in zip(w_star, m)) / sum_w_star
        se_hat_re = math.sqrt(1.0 / sum_w_star)
        z_re = m_hat_re / se_hat_re
        p_re = 2.0 * (1.0 - _phi(abs(z_re)))
        ci_lo_re = m_hat_re - 1.96 * se_hat_re
        ci_hi_re = m_hat_re + 1.96 * se_hat_re
        re = {
            "tau2": tau2,
            "m_hat_re": m_hat_re,
            "se_hat_re": se_hat_re,
            "z_re": z_re,
            "p_re": p_re,
            "ci95_re": [ci_lo_re, ci_hi_re],
        }

    payload = {
        "k": k,
        "runs": runs,
        "fixed": {
            "m_hat": m_hat,
            "se_hat": se_hat,
            "z": z,
            "p": p,
            "ci95": [ci_lo, ci_hi],
        },
        "heterogeneity": {
            "Q": Q,
            "df": df,
            "p_Q": p_Q,
            "I2": I2,
        },
        "random_effects": re,
        "input": str(in_path),
    }

    (outdir / "m_aggregate.json").write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("# m aggregation across PTEMCEE multi-start runs")
    lines.append("")
    lines.append(f"Input: `{in_path}`")
    lines.append("")
    lines.append("## Fixed-effects (inverse-variance)")
    lines.append(f"- m_hat = {m_hat:.6g}")
    lines.append(f"- se_hat = {se_hat:.6g}")
    lines.append(f"- 95% CI = [{ci_lo:.6g}, {ci_hi:.6g}]")
    lines.append(f"- Z = {z:.3f}")
    lines.append(f"- p = {p:.6g}")
    lines.append("")
    lines.append("## Heterogeneity")
    lines.append(f"- Q = {Q:.6g}")
    lines.append(f"- df = {df}")
    lines.append(f"- p_Q = {p_Q:.6g}")
    lines.append(f"- I^2 = {I2:.3f}")
    if re is not None:
        lines.append("")
        lines.append("## Random-effects (DerSimonian–Laird)")
        lines.append(f"- tau^2 = {re['tau2']:.6g}")
        lines.append(f"- m_hat_RE = {re['m_hat_re']:.6g}")
        lines.append(f"- se_hat_RE = {re['se_hat_re']:.6g}")
        lines.append(f"- 95% CI = [{re['ci95_re'][0]:.6g}, {re['ci95_re'][1]:.6g}]")
        lines.append(f"- Z_RE = {re['z_re']:.3f}")
        lines.append(f"- p_RE = {re['p_re']:.6g}")

    (outdir / "m_aggregate.md").write_text("\n".join(lines) + "\n")

    print(
        f"m_hat={m_hat:.6g} se={se_hat:.6g} z={z:.3f} p={p:.6g} I2={I2:.3f}"
        + (f" m_hat_RE={re['m_hat_re']:.6g}" if re is not None else "")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

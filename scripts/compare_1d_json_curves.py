from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_xy(path: Path, *, x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(path.read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: expected a JSON object.")
    if x_key not in d or y_key not in d:
        raise KeyError(f"{path}: missing keys {x_key!r} and/or {y_key!r}.")
    x = np.asarray(d[x_key], dtype=float)
    y = np.asarray(d[y_key], dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError(f"{path}: expected 1D arrays with matching shapes (>=2).")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{path}: x contains non-finite values.")
    return x, y


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = np.where(np.isfinite(y), y, 0.0)
    s = float(np.sum(y))
    if not (np.isfinite(s) and s > 0.0):
        return y
    return y / s


def _kl(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-300) -> float:
    p = np.clip(np.asarray(p, dtype=float), eps, np.inf)
    q = np.clip(np.asarray(q, dtype=float), eps, np.inf)
    p = p / float(np.sum(p))
    q = q / float(np.sum(q))
    return float(np.sum(p * (np.log(p) - np.log(q))))


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two 1D curves stored in JSON.")
    ap.add_argument("a", help="Path to JSON A.")
    ap.add_argument("b", help="Path to JSON B.")
    ap.add_argument("--xkey-a", default="H0_grid", help="JSON key for x in A (default H0_grid).")
    ap.add_argument("--ykey-a", default="posterior", help="JSON key for y in A (default posterior).")
    ap.add_argument("--xkey-b", default="H0_grid", help="JSON key for x in B (default H0_grid).")
    ap.add_argument("--ykey-b", default="posterior", help="JSON key for y in B (default posterior).")
    ap.add_argument("--grid", choices=["a", "b"], default="a", help="Which x-grid to compare on (default a).")
    ap.add_argument("--normalize", action="store_true", help="Normalize y arrays to sum=1 before comparison.")
    ap.add_argument("--out-json", default=None, help="Optional JSON output path.")
    ap.add_argument("--out-plot", default=None, help="Optional PNG plot path.")
    args = ap.parse_args()

    pa = Path(args.a).expanduser().resolve()
    pb = Path(args.b).expanduser().resolve()
    xa, ya = _load_xy(pa, x_key=str(args.xkey_a), y_key=str(args.ykey_a))
    xb, yb = _load_xy(pb, x_key=str(args.xkey_b), y_key=str(args.ykey_b))

    if bool(args.normalize):
        ya = _normalize(ya)
        yb = _normalize(yb)

    if str(args.grid) == "a":
        x = xa
        y1 = ya
        y2 = np.interp(xa, xb, yb, left=np.nan, right=np.nan)
    else:
        x = xb
        y1 = np.interp(xb, xa, ya, left=np.nan, right=np.nan)
        y2 = yb

    finite = np.isfinite(y1) & np.isfinite(y2)
    if not np.any(finite):
        raise ValueError("No overlap between curves after interpolation.")

    diff = (y1 - y2)[finite]
    y1f = y1[finite]
    y2f = y2[finite]
    out: dict[str, Any] = {
        "a": str(pa),
        "b": str(pb),
        "grid": str(args.grid),
        "xkey_a": str(args.xkey_a),
        "ykey_a": str(args.ykey_a),
        "xkey_b": str(args.xkey_b),
        "ykey_b": str(args.ykey_b),
        "normalize": bool(args.normalize),
        "n_overlap": int(np.sum(finite)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "l1_diff": float(np.sum(np.abs(diff))),
    }

    if bool(args.normalize):
        out["kl_a_to_b"] = _kl(y1f, y2f)
        out["kl_b_to_a"] = _kl(y2f, y1f)

    print(json.dumps(out, indent=2, sort_keys=True))
    if args.out_json is not None:
        Path(args.out_json).expanduser().resolve().write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    if args.out_plot is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:  # pragma: no cover
            raise SystemExit(f"matplotlib required for --out-plot: {e}")

        x_f = x[finite]
        plt.figure(figsize=(7.5, 4.2))
        plt.plot(x_f, y1f, "-", lw=1.6, label="A (interp)" if str(args.grid) == "b" else "A")
        plt.plot(x_f, y2f, "-", lw=1.4, label="B (interp)" if str(args.grid) == "a" else "B")
        plt.xlabel(str(args.xkey_a) if str(args.grid) == "a" else str(args.xkey_b))
        plt.ylabel("y")
        plt.title("Curve comparison")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(str(Path(args.out_plot).expanduser().resolve()), dpi=160)
        plt.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


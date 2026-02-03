from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


@dataclass(frozen=True)
class FitResult:
    label: str
    fields: list[list[float]]
    nkbins: int
    kmax: float
    nsurr: int
    snr: float
    bv_mle: list[float]
    fnl_fixed: float


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Analyze a finished kszx KszPipe output directory.\n"
            "\n"
            "This is a baseline validation step for the 'real' kSZ velocity reconstruction:\n"
            "  - loads pk_data.npy + pk_surrogates.npy\n"
            "  - fits b_v (conditional on f_NL=0)\n"
            "  - reports an overall SNR for P_gv(k)\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--outdir", required=True, help="KszPipe output directory (contains pk_data.npy, pk_surrogates.npy).")
    ap.add_argument("--nkbins", type=int, default=10, help="Number of k bins to use in likelihood (default 10).")
    ap.add_argument("--out", default=None, help="Output JSON path (default: <outdir>/analysis_<stamp>.json)")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    if not outdir.exists():
        raise FileNotFoundError(outdir)

    env = dict(os.environ)
    env["KSZX_DATA_DIR"] = str(Path(args.kszx_data_dir).resolve())

    # Defer imports until after KSZX_DATA_DIR is set.
    import kszx

    pout = kszx.KszPipeOutdir(str(outdir))

    nk = int(args.nkbins)
    if nk < 4 or nk > pout.nkbins:
        raise ValueError(f"--nkbins must satisfy 4 <= nkbins <= {pout.nkbins} (got {nk})")

    results: list[FitResult] = []
    # 90, 150, sum-map, joint(90,150).
    for label, fields in [
        ("90", [[1.0, 0.0]]),
        ("150", [[0.0, 1.0]]),
        ("sum", [[1.0, 1.0]]),
        ("joint", [[1.0, 0.0], [0.0, 1.0]]),
    ]:
        lik = kszx.PgvLikelihood.from_pipeline_outdir(pout, fields=fields, nkbins=nk, multi_bias=(label == "joint"))
        snr = float(lik.compute_snr())
        bv = lik.fit_bv(fnl=0.0, bv0=0.3)
        results.append(
            FitResult(
                label=label,
                fields=[[float(a), float(b)] for a, b in np.asarray(fields, dtype=float)],
                nkbins=int(nk),
                kmax=float(pout.kmax),
                nsurr=int(pout.nsurr),
                snr=snr,
                bv_mle=[float(x) for x in np.asarray(bv, dtype=float).tolist()],
                fnl_fixed=0.0,
            )
        )

    payload = {
        "created_utc": _utc_stamp(),
        "kszpipe_outdir": str(outdir),
        "kbin_centers": [float(x) for x in np.asarray(pout.k[:nk], dtype=float).tolist()],
        "results": [asdict(r) for r in results],
    }

    out_path = Path(args.out) if args.out else outdir / f"analysis_{_utc_stamp()}.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"[kszpipe_analyze] wrote {out_path}", flush=True)

    # Print a tight console summary.
    for r in results:
        print(f"[kszpipe_analyze] {r.label:>5s}: SNR={r.snr:.3f}  bv_mle={r.bv_mle}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _as_float(x: Any) -> float:
    try:
        return float(np.asarray(x).reshape(()))
    except Exception:
        return float("nan")


@dataclass(frozen=True)
class FitResult:
    field: str
    nkbins_used: int
    fnl_fixed: float
    bv_mle: list[float]
    snr_pgv: float
    chi2: float
    ndof: int
    pte: float


@dataclass(frozen=True)
class Manifest:
    created_utc: str
    kszpipe_outdir: str
    results: list[FitResult]


def _fit_one(*, pout: "kszx.KszPipeOutdir", field: str, nkbins: int, bv0: float, multi_bias: bool | None) -> FitResult:
    import kszx

    if field == "90":
        fields = [[1, 0]]
        multi_bias = None
    elif field == "150":
        fields = [[0, 1]]
        multi_bias = None
    elif field == "sum":
        fields = [[1, 1]]
        multi_bias = None
    elif field == "null":
        fields = [[1, -1]]
        multi_bias = None
    elif field == "joint_shared":
        fields = [[1, 0], [0, 1]]
        multi_bias = False
    elif field == "joint_independent":
        fields = [[1, 0], [0, 1]]
        multi_bias = True
    else:
        raise ValueError(f"Unknown field mode: {field}")

    lik = kszx.PgvLikelihood.from_pipeline_outdir(
        pout,
        fields=fields,
        nkbins=int(nkbins),
        multi_bias=multi_bias,
        jeffreys_prior=False,
    )

    fnl = 0.0
    bv = lik.fit_bv(fnl=fnl, bv0=float(bv0))
    chi2, ndof, pte = lik.analyze_chi2(fnl, bv)
    snr = lik.compute_snr()

    return FitResult(
        field=str(field),
        nkbins_used=int(nkbins),
        fnl_fixed=float(fnl),
        bv_mle=[float(x) for x in np.asarray(bv, dtype=float).ravel().tolist()],
        snr_pgv=_as_float(snr),
        chi2=_as_float(chi2),
        ndof=int(ndof),
        pte=_as_float(pte),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fit the kSZ velocity-bias parameter(s) b_v from a completed kszx KszPipe output_dir.\n"
            "\n"
            "This is a key 'professional' step for kSZ velocity reconstruction:\n"
            "  - run KszPipe -> pk_data.npy + pk_surrogates.npy\n"
            "  - fit b_v from P_gv(k) using kszx.PgvLikelihood\n"
            "\n"
            "The output JSON is intended to be consumed by theta-map builders and/or EG estimators.\n"
        )
    )
    ap.add_argument("--kszpipe-outdir", required=True, help="Directory containing params.yml + pk_data.npy + pk_surrogates.npy")
    ap.add_argument(
        "--field",
        default="sum",
        help=(
            "Comma-separated modes to fit: 90,150,sum,null,joint_shared,joint_independent.\n"
            "Default: sum"
        ),
    )
    ap.add_argument("--nkbins", type=int, default=None, help="Number of k bins to use from the front (default: all).")
    ap.add_argument("--bv0", type=float, default=0.3, help="Initial b_v for optimizer (default 0.3).")
    ap.add_argument("--out", default=None, help="Output JSON (default: outputs/kszpipe_bv_fit_<stamp>.json)")
    args = ap.parse_args()

    import kszx

    outdir = Path(args.kszpipe_outdir).resolve()
    if not outdir.exists():
        raise FileNotFoundError(outdir)

    pout = kszx.KszPipeOutdir(str(outdir))
    nkbins = int(args.nkbins) if args.nkbins is not None else int(pout.nkbins)
    if nkbins < 4 or nkbins > int(pout.nkbins):
        raise ValueError(f"--nkbins must be in [4, {pout.nkbins}] for this outdir.")

    fields = [p.strip() for p in str(args.field).split(",") if p.strip()]
    if not fields:
        raise ValueError("--field must be non-empty")

    results: list[FitResult] = []
    for f in fields:
        results.append(_fit_one(pout=pout, field=str(f), nkbins=nkbins, bv0=float(args.bv0), multi_bias=None))

    man = Manifest(created_utc=_utc_stamp(), kszpipe_outdir=str(outdir), results=results)
    out = Path(args.out) if args.out else Path("outputs") / f"kszpipe_bv_fit_{_utc_stamp()}.json"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(man), indent=2, sort_keys=True) + "\n")
    print(f"[kszpipe_bv_fit] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


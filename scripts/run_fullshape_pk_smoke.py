#!/usr/bin/env python3
import argparse
import os
import subprocess
from pathlib import Path


def _run(out_dir: Path, steps: int, burn: int, draws: int, *, ntemps: int, tmax: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    })
    cmd = [
        str(Path('.venv/bin/python').resolve()),
        "scripts/run_realdata_recon.py",
        "--out",
        str(out_dir),
        "--seed",
        "202",
        "--mu-init-seed",
        "202",
        "--mu-sampler",
        "ptemcee",
        "--pt-ntemps",
        str(ntemps),
        "--pt-tmax",
        str(tmax),
        "--mu-steps",
        str(steps),
        "--mu-burn",
        str(burn),
        "--mu-draws",
        str(draws),
        "--mu-walkers",
        "32",
        "--cpu-cores",
        "8",
        "--mu-procs",
        "8",
        "--gp-procs",
        "1",
        "--include-fullshape-pk",
        "--skip-ablations",
        "--skip-hz-recon",
    ]
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--medium", action="store_true")
    args = parser.parse_args()

    base = Path("outputs") / "smoke_fullshape_pk"
    if not args.small and not args.medium:
        args.small = True
        args.medium = True
    if args.small:
        _run(base / "small", steps=2, burn=0, draws=1, ntemps=2, tmax=5)
    if args.medium:
        _run(base / "medium", steps=30, burn=8, draws=15, ntemps=3, tmax=8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

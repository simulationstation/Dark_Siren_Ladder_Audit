#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    out_dir = Path("outputs") / "optical_bias" / "sn_kappa_xcorr" / "smoke"
    cmd = [
        str(Path('.venv/bin/python').resolve()),
        "scripts/run_sn_kappa_xcorr.py",
        "--out",
        str(out_dir),
        "--nside",
        "256",
        "--n-null",
        "50",
        "--ell-min",
        "8",
        "--ell-max",
        "256",
        "--ell-bin",
        "32",
        "--detrend",
        "poly1",
        "--sn-max",
        "500",
        "--allow-unverified",
    ]
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

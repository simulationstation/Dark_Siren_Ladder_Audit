from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print(f"[kszpipe_run] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run the kszx KszPipe pipeline on a prepared input_dir.\n"
            "\n"
            "This is the 'real' kSZ velocity reconstruction stage (produces pk_data.npy + pk_surrogates.npy).\n"
            "It is resumable: KszPipe skips outputs already present in the output directory.\n"
        )
    )
    ap.add_argument("--kszx-data-dir", default="data/cache/kszx_data", help="KSZX_DATA_DIR root (default: data/cache/kszx_data).")
    ap.add_argument("--input-dir", required=True, help="Input directory produced by scripts/kszx_prepare_kszpipe_inputs.py")
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/kszpipe_<stamp>/)")
    ap.add_argument(
        "-p",
        "--processes",
        type=int,
        default=8,
        help="Worker processes (default 8). Increase for production runs (ask user before big runs).",
    )
    args = ap.parse_args()

    input_dir = Path(args.input_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    out_dir = Path(args.out) if args.out else Path("outputs") / f"kszpipe_{_utc_stamp()}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["KSZX_DATA_DIR"] = str(Path(args.kszx_data_dir).resolve())

    cmd = [str(Path(".venv/bin/python").resolve()), "-m", "kszx", "kszpipe_run", "-p", str(int(args.processes)), str(input_dir), str(out_dir)]
    _run(cmd, env=env)

    print(f"[kszpipe_run] done: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


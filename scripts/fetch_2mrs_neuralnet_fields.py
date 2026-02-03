from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from entropy_horizon_recon.cache import DataPaths
from entropy_horizon_recon.void_prism_velocity_products import fetch_2mrs_neuralnet_zip


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch the public 2MRS NeuralNet (Lilow+2024) density/velocity zip.")
    ap.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Allow download even if TWO_MRS_NN_ZIP_SHA256 is not pinned yet.",
    )
    args = ap.parse_args()

    zip_path = fetch_2mrs_neuralnet_zip(paths=DataPaths(Path.cwd()), allow_unverified=bool(args.allow_unverified))
    print(f"path: {zip_path}")
    print(f"sha256: {_sha256(zip_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


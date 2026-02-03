from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SUTC")


def _run(cmd: list[str], *, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        return int(p.wait())


@dataclass(frozen=True)
class Step:
    label: str
    node: str  # pytest target (file::testname or file)


def _default_steps() -> list[Step]:
    # File-level steps are coarse but robust: if a long/networky test file fails,
    # the whole suite can be resumed without rerunning earlier files.
    files = sorted(Path("tests").glob("test_*.py")) + sorted(Path("tests/optical_bias").glob("test_*.py"))
    return [Step(label=p.name, node=str(p)) for p in files]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Run pytest in resumable, step-tracked chunks.\n"
            "\n"
            "This is for long smoke/regression suites where downloads or heavy likelihood tests can take time.\n"
            "We run one test file per step, write logs per step, and emit a progress.json that can be resumed.\n"
        )
    )
    ap.add_argument("--out", default=None, help="Output directory (default: outputs/pytest_resumable_<UTCSTAMP>).")
    ap.add_argument("--resume", action="store_true", help="Resume an existing run-dir (skip completed steps).")
    ap.add_argument("--workers", type=int, default=1, help="pytest-xdist workers (default 1). Use >1 only if tests are safe in parallel.")
    ap.add_argument("--k", default=None, help="pytest -k expression (optional).")
    ap.add_argument("--maxfail", type=int, default=1, help="pytest --maxfail (default 1).")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path("outputs") / f"pytest_resumable_{_utc_stamp()}"
    out_dir = out_dir.resolve()
    logs_dir = out_dir / "logs"
    steps_path = out_dir / "steps.json"
    progress_path = out_dir / "progress.json"

    if args.resume:
        if not out_dir.exists():
            raise FileNotFoundError(f"--resume specified but out_dir does not exist: {out_dir}")
        if not steps_path.exists():
            raise FileNotFoundError(f"--resume specified but missing {steps_path}")
        steps = [Step(**x) for x in _load_json(steps_path)["steps"]]
        progress = _load_json(progress_path) if progress_path.exists() else {}
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        steps = _default_steps()
        _write_json(steps_path, {"created_utc": _utc_stamp(), "steps": [asdict(s) for s in steps]})
        progress = {"created_utc": _utc_stamp(), "steps_total": len(steps), "done": [], "failed": None}
        _write_json(progress_path, progress)

    done = set(progress.get("done", []))

    base_cmd = [sys.executable, "-m", "pytest", "-q", f"--maxfail={int(args.maxfail)}"]
    if int(args.workers) > 1:
        base_cmd += ["-n", str(int(args.workers)), "--dist", "loadscope"]
    if args.k:
        base_cmd += ["-k", str(args.k)]

    t0 = time.time()
    for i, step in enumerate(steps, start=1):
        if step.label in done:
            continue

        pct = 100.0 * float(len(done)) / float(len(steps)) if steps else 100.0
        print(f"[pytest_resumable] step {i}/{len(steps)} ({pct:5.1f}% done): {step.node}", flush=True)

        log_path = logs_dir / f"{i:02d}_{step.label}.log"
        ok_path = logs_dir / f"{i:02d}_{step.label}.ok"

        cmd = base_cmd + [step.node]
        rc = _run(cmd, log_path=log_path)
        if rc == 0:
            ok_path.write_text("ok\n")
            done.add(step.label)
            progress["done"] = sorted(done)
            progress["last_ok_step"] = step.label
            progress["elapsed_s"] = float(time.time() - t0)
            _write_json(progress_path, progress)
            continue

        progress["failed"] = {"step": step.label, "node": step.node, "returncode": int(rc)}
        progress["elapsed_s"] = float(time.time() - t0)
        _write_json(progress_path, progress)
        print(f"[pytest_resumable] FAIL: {step.label} (rc={rc}). See {log_path}", flush=True)
        return int(rc)

    progress["elapsed_s"] = float(time.time() - t0)
    progress["failed"] = None
    _write_json(progress_path, progress)
    print(f"[pytest_resumable] DONE: {len(done)}/{len(steps)} steps ok. out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


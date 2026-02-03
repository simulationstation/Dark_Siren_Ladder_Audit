from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunStatus:
    out_dir: Path
    pid: int | None
    ps_line: str | None
    progress: dict[str, Any] | None
    exit_status: dict[str, Any] | None


def _read_int(path: Path) -> int | None:
    try:
        s = path.read_text().strip()
        return int(s)
    except Exception:
        return None


def _tail_text(path: Path, *, n_lines: int = 30, max_bytes: int = 64_000) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            n = int(min(size, max_bytes))
            f.seek(-n, 2)
            data = f.read(n)
        txt = data.decode("utf-8", errors="replace")
        lines = txt.splitlines()
        return "\n".join(lines[-int(n_lines) :])
    except Exception:
        return None


def _ps_status(pid: int) -> str | None:
    try:
        cp = subprocess.run(
            ["ps", "-p", str(int(pid)), "-o", "pid,etime,%cpu,%mem,cmd"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if cp.returncode != 0:
        return None
    lines = [ln.rstrip() for ln in cp.stdout.splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    return lines[-1]


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if root.is_file():
        return []
    out: list[Path] = []
    if (root / "pid.txt").exists():
        return [root]
    for p in root.rglob("pid.txt"):
        if p.is_file():
            out.append(p.parent)
    return sorted(set(out))


def _status_for_run_dir(out_dir: Path) -> RunStatus:
    pid_path = out_dir / "pid.txt"
    log_path = out_dir / "run.log"
    progress_path = out_dir / "progress.json"
    exit_status_path = out_dir / "exit_status.json"

    pid = _read_int(pid_path) if pid_path.exists() else None
    ps_line = _ps_status(pid) if pid is not None else None
    progress = _load_json(progress_path) if progress_path.exists() else None
    exit_status = _load_json(exit_status_path) if exit_status_path.exists() else None
    return RunStatus(out_dir=out_dir, pid=pid, ps_line=ps_line, progress=progress, exit_status=exit_status)


def _fmt_progress(p: dict[str, Any] | None) -> str | None:
    if not p:
        return None
    try:
        n_done = int(p.get("n_done", -1))
        n_target = int(p.get("n_target", -1))
        if n_done >= 0 and n_target > 0:
            frac = 100.0 * float(n_done) / float(n_target)
            extra = []
            if "updated_utc" in p:
                extra.append(f"updated={p['updated_utc']}")
            return f"{n_done}/{n_target} ({frac:.1f}%)" + (f" [{', '.join(extra)}]" if extra else "")
    except Exception:
        return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Summarize status for detached runs (pid.txt + run.log).")
    ap.add_argument("paths", nargs="*", help="Run output dirs or parent dirs to scan (default: outputs/).")
    ap.add_argument("--tail-lines", type=int, default=25, help="Lines of run.log to show per run (default 25).")
    args = ap.parse_args()

    roots = [Path(p).expanduser().resolve() for p in (args.paths if args.paths else ["outputs"])]
    run_dirs: list[Path] = []
    for r in roots:
        run_dirs.extend(_discover_run_dirs(r))
    run_dirs = sorted(set(run_dirs))

    if not run_dirs:
        print("No run directories found (no pid.txt discovered).")
        return 2

    tail_lines = int(args.tail_lines)
    try:
        for out_dir in run_dirs:
            st = _status_for_run_dir(out_dir)
            running = st.ps_line is not None
            prog = _fmt_progress(st.progress)
            print("=" * 88)
            print(f"out: {st.out_dir}")
            print(f"pid: {st.pid if st.pid is not None else '<missing>'}  running: {running}")
            if st.ps_line is not None:
                print(f"ps : {st.ps_line}")
            if st.exit_status is not None:
                code = st.exit_status.get("exit_code", "?")
                end = st.exit_status.get("end_utc", "?")
                print(f"exit: code={code} end_utc={end}")
            if prog is not None:
                print(f"progress: {prog}")
            log_path = st.out_dir / "run.log"
            if log_path.exists():
                print(f"log: {log_path}")
                tail = _tail_text(log_path, n_lines=tail_lines)
                if tail:
                    print("-" * 88)
                    print(tail)
            else:
                print("log: <missing run.log>")
    except BrokenPipeError:  # pragma: no cover
        # Allow piping to `head` without noisy tracebacks.
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


def _tail_text(path: Path, *, max_bytes: int = 150_000) -> str:
    try:
        size = path.stat().st_size
    except Exception:
        return ""
    try:
        with path.open("rb") as f:
            f.seek(max(0, int(size) - int(max_bytes)))
            data = f.read()
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def _ps_snapshot() -> list[tuple[float, float, str]]:
    out = subprocess.check_output(["ps", "-eo", "pcpu,pmem,args"], text=True)
    lines = out.splitlines()
    rows: list[tuple[float, float, str]] = []
    for line in lines[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            cpu = float(parts[0])
            mem = float(parts[1])
        except ValueError:
            continue
        rows.append((cpu, mem, parts[2]))
    return rows


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.1f}h"


def _read_pid(out_base: Path) -> int | None:
    p = out_base / "pid.txt"
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except Exception:
        return None


def _proc_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _parse_progress(run_log_text: str) -> tuple[int | None, int | None, str | None]:
    # We estimate progress by counting the "block i/N" lines. With progress_every_block=1:
    # total_block_lines ~= n_runs * n_blocks * (1 + n_embeddings)
    # (GR + embedding(s)). We don't know n_embeddings from the log robustly, so we report the raw
    # block count and the last-seen block label.
    block_pat = re.compile(r"\\[void_prism_joint\\] .* emb=.* block (\\d+)/(\\d+) z_eff=")
    blocks = list(block_pat.finditer(run_log_text))
    if not blocks:
        return None, None, None
    last = blocks[-1]
    try:
        i = int(last.group(1))
        n = int(last.group(2))
    except Exception:
        return None, None, None
    # Most useful "where are we?" string: last line containing a block marker.
    last_line = None
    for line in reversed(run_log_text.splitlines()):
        if " block " in line and "[void_prism_joint]" in line:
            last_line = line.strip()
            break
    return i, n, last_line


def main() -> int:
    ap = argparse.ArgumentParser(description="Status for a detached void-prism joint E_G scoring run.")
    ap.add_argument("out_base", type=Path, help="Output directory used as --out for run_void_prism_eg_joint_test.py")
    args = ap.parse_args()

    out_base = args.out_base
    run_log = out_base / "run.log"
    results = out_base / "tables" / "results.json"

    load1, load5, load15 = os.getloadavg()
    ps_rows = _ps_snapshot()

    cpu_sum = 0.0
    mem_sum = 0.0
    proc_count = 0
    out_s = str(out_base)
    for cpu, mem, cmd in ps_rows:
        if out_s in cmd and "run_void_prism_eg_joint_test.py" in cmd:
            cpu_sum += cpu
            mem_sum += mem
            proc_count += 1

    pid = _read_pid(out_base)
    pid_running = _proc_running(pid) if pid is not None else False

    now = time.time()
    age_s = None
    if run_log.exists():
        try:
            age_s = now - run_log.stat().st_mtime
        except Exception:
            age_s = None

    log_bytes = run_log.stat().st_size if run_log.exists() else 0
    tail = _tail_text(run_log)
    bi, bn, last_line = _parse_progress(tail)

    cores_used = cpu_sum / 100.0
    print(
        f"out={out_base}  pid={pid if pid is not None else '-'} running={int(pid_running)}  procs={proc_count}  cpu={cpu_sum:.1f}% (~{cores_used:.1f} cores)  mem={mem_sum:.2f}%  loadavg={load1:.1f},{load5:.1f},{load15:.1f}"
    )
    print(f"run_log={run_log} bytes={log_bytes} age={_fmt_age(age_s)} results_json={int(results.exists())}")
    if bi is not None and bn is not None:
        print(f"last_block={bi}/{bn}")
    if last_line:
        print(f"last_line={last_line}")
    if tail:
        print("---- tail(run.log) ----")
        lines = tail.splitlines()[-20:]
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


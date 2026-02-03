#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path


def _tail_text(path: Path, *, max_bytes: int = 200_000) -> str:
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
    rows: list[tuple[float, float, str]] = []
    for line in out.splitlines()[1:]:
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


def _pid_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _parse_jk_progress(text: str) -> tuple[float | None, str | None]:
    # Examples:
    #   [jk] 17/48 (35.4%)
    #   [jk] 190/192 (99.0%)
    pat = re.compile(r"\\[jk\\] (\\d+)/(\\d+) \\((\\d+\\.\\d+)%\\)")
    matches = list(pat.finditer(text))
    if not matches:
        return None, None
    m = matches[-1]
    try:
        pct = float(m.group(3))
    except Exception:
        pct = None
    last_line = None
    for line in reversed(text.splitlines()):
        if "[jk]" in line and "/" in line and "%" in line:
            last_line = line.strip()
            break
    return pct, last_line


def main() -> int:
    ap = argparse.ArgumentParser(description="Status for the void_prism_boss_smica launcher output directory.")
    ap.add_argument("out_base", type=Path)
    args = ap.parse_args()

    out_base = args.out_base
    job_log = out_base / "job.log"
    suite_log = out_base / "suite" / "run.log"
    score_log = out_base / "score" / "run.log"
    suite_json = out_base / "suite" / "tables" / "suite_joint.json"
    results_json = out_base / "score" / "tables" / "results.json"

    pid = _read_pid(out_base)
    running = _pid_running(pid) if pid is not None else False

    load1, load5, load15 = os.getloadavg()
    ps_rows = _ps_snapshot()

    out_s = str(out_base)
    cpu_sum = 0.0
    mem_sum = 0.0
    proc_count = 0
    for cpu, mem, cmd in ps_rows:
        if out_s in cmd and ("measure_void_prism_eg_suite_jackknife.py" in cmd or "run_void_prism_eg_joint_test.py" in cmd or "job.sh" in cmd):
            cpu_sum += cpu
            mem_sum += mem
            proc_count += 1

    now = time.time()
    age_job = None
    if job_log.exists():
        age_job = now - job_log.stat().st_mtime

    stage = "start"
    if results_json.exists():
        stage = "done"
    elif suite_json.exists():
        stage = "score" if score_log.exists() else "measured"
    elif (out_base / "footprint_mask.fits").exists():
        stage = "measure"

    suite_tail = _tail_text(suite_log)
    pct, jk_line = _parse_jk_progress(suite_tail)

    cores_used = cpu_sum / 100.0
    print(
        f"out={out_base}  pid={pid if pid is not None else '-'} running={int(running)}  stage={stage}  procs={proc_count}  cpu={cpu_sum:.1f}% (~{cores_used:.1f} cores)  mem={mem_sum:.2f}%  loadavg={load1:.1f},{load5:.1f},{load15:.1f}"
    )
    print(
        f"job_log_age={_fmt_age(age_job)}  suite_json={int(suite_json.exists())}  results_json={int(results_json.exists())}"
    )
    if pct is not None:
        print(f"jackknife_pct={pct:.1f}%")
    if jk_line:
        print(f"jackknife_line={jk_line}")

    if job_log.exists():
        print("---- tail(job.log) ----")
        for line in _tail_text(job_log).splitlines()[-15:]:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


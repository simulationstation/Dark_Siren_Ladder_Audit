# Agent Instructions (Project)

This repository is used for long-running numerical experiments (MCMC, lensing/void tests, etc.).

## Running Tests / Experiments (REQUIRED)

- **Always use the robust detached launch paradigm** (the only one that has been reliable here):
  - Create an output directory.
  - Write a `job.sh` inside it (or use an existing `scripts/launch_*_single_nohup.sh` launcher that
    already writes `pid.txt` + `run.log`).
  - Start via **one** top-level launcher using:
    - `env ... setsid taskset -c <CPUSET> bash <job.sh> > <out>/run.log 2>&1 < /dev/null &`
    - Write the PID to `<out>/pid.txt`.
  - Do **not** start multiple separate nohups for seeds/components unless explicitly requested.

- **Never leave long-running work attached to a controlling terminal/session (SIGHUP risk).**
  - If a download or job is started “in the foreground” of an ephemeral session and that session
    closes, the OS can deliver `SIGHUP` to the process group (this happened with `wget`: it printed
    `SIGHUP received...` and the wrapper process died before finalizing outputs).
  - Therefore: always use `setsid` (per the detached paradigm above) for any long run, and ensure
    downloads are resumable (`wget -c`/`.part` + atomic rename) so a restart can finalize cleanly.

- **Use max cores by default.**
  - Set `CPUSET` to `0-$(nproc-1)` unless the user explicitly caps resources.
  - Prefer **process-level parallelism** (ptemcee workers / multiprocessing pools / job arrays) to
    consume cores.
  - Keep OpenMP/BLAS threads from oversubscribing by default:
    - `OMP_NUM_THREADS=1`
    - `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`
  - Only raise `OMP_NUM_THREADS` above 1 when running a known single-process, OpenMP-parallel
    kernel that benefits from it (and ensure you are *not* also spawning many processes).
  - Keep BLAS threads from oversubscribing: `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
    `NUMEXPR_NUM_THREADS=1` (unless the code path is known to benefit from BLAS threading).

- **No timeouts.** Do not use `timeout(1)` or any wrapper that can kill long jobs unexpectedly.

- **Progress + resumability.**
  - For any long job, print periodic progress in `run.log` (percent or i/N).
  - Write periodic partial outputs (e.g. `*_partial.json`) so interruptions don’t lose all work.

## Status Reporting (when user asks “status”)

Include, at minimum:
- PID(s) and command line(s)
- CPU usage (and how many cores are effectively being used)
- elapsed time
- a progress number (percent / i/N) from logs or partial outputs
- output directory paths being written

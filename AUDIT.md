# Audit Discipline: runs must be inspectable (no “mystery background work”)

This repo is full of **long-running jobs** (inference, scans, null batteries, downloads). A recurring failure
mode is starting work in a way that is **not inspectable**: the user cannot easily tell what is running,
where it is writing, how to stop it, or whether it finished.

This file defines the **non-negotiable run discipline** for this project so that every run has:
- a single output directory
- a `run.log` with periodic progress
- a `pid.txt` for status/stop
- an `exit_status.json` written on exit (success or failure)
- resumable partial outputs (`progress.json` or `*_partial.*`)

If a run is not inspectable, it is not a valid audit artifact.

---

## Rule 1 — Never run long work “attached”

Any run that can take **longer than ~30 seconds**, or whose duration is uncertain, must be launched
**detached** (to avoid SIGHUP) and must write:
- `pid.txt`
- `run.log`

The required pattern is the one in `AGENTS.md`:

```bash
OUT="outputs/<name>_<UTCSTAMP>"
mkdir -p "$OUT"

# Write the job.
cat > "$OUT/job.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:?OUT_DIR not set}"

# Always write an exit marker (so we can tell “finished” vs “hung”).
_write_exit_status() {
  local code="$1"
  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '{"exit_code":%s,"end_utc":"%s"}\n' "$code" "$end_utc" > "$OUT_DIR/exit_status.json"
}
trap '_write_exit_status "$?"' EXIT

# IMPORTANT: write the *real* running PID from inside the job.
# Do NOT rely on `$!` from the launcher (setsid/taskset wrappers can fork/exit).
echo "$$" > "$OUT_DIR/pid.txt"

# Your command(s) here (must print periodic progress):
echo "[job] start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hello from job"
echo "[job] done_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
chmod +x "$OUT/job.sh"

# Use max cores by default; pin to all CPUs.
CPUSET="0-$(($(nproc)-1))"

# Prevent BLAS/OpenMP oversubscription.
env OUT_DIR="$OUT" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  setsid taskset -c "$CPUSET" bash "$OUT/job.sh" > "$OUT/run.log" 2>&1 < /dev/null &
echo $! > "$OUT/wrapper_pid.txt"
```

---

## Rule 2 — Every run must have an obvious status command

For any run output directory `$OUT`, the status check must work verbatim:

```bash
PID="$(cat "$OUT/pid.txt")"
ps -p "$PID" -o pid,etime,%cpu,%mem,cmd
tail -n 50 "$OUT/run.log"
```

Additionally, use the repo helper to avoid “where is it / what’s running?” confusion:

```bash
python scripts/audit_status.py outputs
python scripts/audit_status.py "$OUT"
```

If the job is finished, `ps` will return nothing; the completion status must be visible in `run.log`
and/or `exit_status.json`.

To stop a job:

```bash
kill "$(cat "$OUT/pid.txt")"
```

---

## Rule 3 — Progress is mandatory (and must be machine-readable when possible)

For long loops (replicates, event batteries, null rotations):
- print periodic progress to `run.log` (e.g. `rep i/N`, percent, timestamps)
- write a tiny `progress.json` (or `*_partial.json`) periodically so an interrupted job can be resumed

Minimal `progress.json` schema:

```json
{"n_done": 17, "n_target": 256}
```

---

## Rule 4 — No “silent” background processes

When initiating a run, record (in `run.log` or `manifest.json`) at minimum:
- the exact command line
- git SHA
- output directory path
- key knobs (e.g. event list, `z_max`, selection model, prior mode)

If this bookkeeping is missing, the run is not suitable for audit conclusions.

---

## Repo-specific notes

- **Siren audit system** documentation lives in `SIREN_AUDIT_PROJECT.md`.
- The control ladder “living narrative” is `FINDINGS/siren_ghost_control_ladder_20260202.md`.

#!/usr/bin/env bash
set -u -o pipefail

out_base="${1:?usage: scripts/status_sbc_out_base.sh OUT_BASE}"

echo "out_base=$out_base"
if [ -f "$out_base/pids.txt" ]; then
  pids="$(tr '\n' ' ' < "$out_base/pids.txt" | sed 's/ $//')"
  echo "pids=$pids"
else
  echo "pids.txt: MISSING"
fi

for truth in bh prior; do
  out="$out_base/$truth"
  echo
  echo "truth=$truth out=$out"
  if [ -f "$out/tables/progress.jsonl" ]; then
    n_done="$(wc -l < "$out/tables/progress.jsonl" | tr -d ' ')"
    echo "progress=$n_done"
    echo "progress_tail:"
    tail -n 1 "$out/tables/progress.jsonl" || true
  fi
  if [ -f "$out/tables/summary.json" ]; then
    echo "status=done"
    summary="$(jq -c '.bh_null // .coverage // .pvalues' "$out/tables/summary.json" 2>/dev/null || true)"
    echo "summary=$summary"
  else
    echo "status=running_or_failed"
  fi
  if [ -f "$out/run.log" ]; then
    echo "log_tail:"
    tail -n 5 "$out/run.log" || true
  else
    echo "run.log: MISSING"
  fi
done

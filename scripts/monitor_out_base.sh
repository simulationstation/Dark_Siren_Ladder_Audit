#!/usr/bin/env bash
set -u

if [ $# -lt 2 ]; then
  echo "Usage: $0 <out_base> <status_jsonl> [interval_sec]" >&2
  exit 2
fi

out_base="$1"
status_jsonl="$2"
interval_sec="${3:-30}"

mkdir -p "$(dirname "$status_jsonl")"

# Append JSONL snapshots until killed.
while true; do
  ts="$(date -Is)"

  # Process snapshot for anything with out_base in the cmdline.
  # Use grep -F and tolerate no matches.
  proc_lines="$(ps -eo pcpu,pmem,args | grep -F "$out_base" | grep -v "grep -F" || true)"
  proc_count="$(printf '%s\n' "$proc_lines" | sed '/^$/d' | wc -l | tr -d ' ')"
  cpu_pct="$(printf '%s\n' "$proc_lines" | awk '{sum+=$1} END{printf "%.1f", sum+0}')"
  mem_pct="$(printf '%s\n' "$proc_lines" | awk '{sum+=$2} END{printf "%.2f", sum+0}')"

  # Seed completion by presence of summary.json.
  seeds_total=0
  seeds_done=0
  for d in "$out_base"/M0_start*; do
    [ -d "$d" ] || continue
    seeds_total=$((seeds_total+1))
    [ -f "$d/tables/summary.json" ] && seeds_done=$((seeds_done+1))
  done

  mem_line="$(free -b | awk '/Mem:/ {print $2" "$3" "$7}')"
  mem_total_b="$(printf '%s' "$mem_line" | awk '{print $1}')"
  mem_used_b="$(printf '%s' "$mem_line" | awk '{print $2}')"
  mem_avail_b="$(printf '%s' "$mem_line" | awk '{print $3}')"

  printf '{"ts":"%s","out_base":"%s","proc_count":%s,"cpu_pct":%s,"mem_pct":%s,"seeds_done":%d,"seeds_total":%d,"mem_total_bytes":%s,"mem_used_bytes":%s,"mem_avail_bytes":%s}\n' \
    "$ts" "$out_base" "$proc_count" "$cpu_pct" "$mem_pct" "$seeds_done" "$seeds_total" "$mem_total_b" "$mem_used_b" "$mem_avail_b" >> "$status_jsonl"

  sleep "$interval_sec"
done

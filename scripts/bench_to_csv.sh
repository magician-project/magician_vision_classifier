#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 file1.txt [file2.txt ...]"
  exit 1
fi

for input in "$@"; do
  if [[ ! -f "$input" ]]; then
    echo "Skipping (not found): $input" >&2
    continue
  fi

  out="${input}.csv"

  {
    echo "Step,Framerate"
    grep -E 'Hz' "$input" | awk -F'[=/@ ]+' '{
      # Expected tokens: ... step=64 ... @ 74.88 Hz
      # With FS above: $3=step, $6=framerate
      if ($3 ~ /^[0-9]+$/ && $6 ~ /^[0-9.]+$/) print $3 "," $6
    }'
  } > "$out"

  echo "Wrote: $out"
done


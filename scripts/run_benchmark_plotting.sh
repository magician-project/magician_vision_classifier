#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 file1.txt [file2.txt ...]"
  exit 1
fi

command -v gnuplot >/dev/null 2>&1 || { echo "Error: gnuplot not found"; exit 1; }

csv_files=()

for input in "$@"; do
  [[ -f "$input" ]] || { echo "Skipping (not found): $input" >&2; continue; }

  out="${input}.csv"

  {
    echo "Step,Framerate"
    grep -E 'Hz' "$input" | awk '
      {
        step=""; hz="";

        # Extract step from "step=64"
        if (match($0, /step=[0-9]+/)) {
          step = substr($0, RSTART+5, RLENGTH-5)
        }

        # Extract framerate from "@ 74.88" (optionally with spaces)
        if (match($0, /@[[:space:]]*[0-9]+(\.[0-9]+)?/)) {
          s = substr($0, RSTART+1, RLENGTH-1)   # remove '@'
          gsub(/[[:space:]]+/, "", s)           # remove spaces
          hz = s
        }

        if (step != "" && hz != "") {
          print step "," hz
        }
      }
    '
  } > "$out"

  # If CSV has only header, skip plotting for it
  if [[ $(wc -l < "$out") -le 1 ]]; then
    echo "Warning: no valid data parsed from $input (CSV has only header). Skipping plots for this file." >&2
    continue
  fi

  csv_files+=("$out")
  echo "Wrote: $out"
done

if [[ ${#csv_files[@]} -eq 0 ]]; then
  echo "No CSVs with data were generated; nothing to plot."
  exit 1
fi

# Build gnuplot list: 'a.csv' 'b.csv'
gnuplot_list=""
for f in "${csv_files[@]}"; do
  gnuplot_list+="'$f' "
done

# --- Per-file plots ---
gnuplot <<GP
set datafile separator ","
set terminal pngcairo size 1200,700 enhanced font "Sans,12"

do for [f in "${gnuplot_list}"] {
    set output sprintf("%s.png", f)
    set title f
    set key off
    set grid
    set xlabel "Step"
    set ylabel "Framerate (Hz)"
    plot f using 1:2 with linespoints linewidth 2 pointsize 0.8
}
unset output
GP

echo "Wrote per-config PNGs: <csv>.png"

# --- Master plot ---
gnuplot <<GP
set datafile separator ","
set terminal pngcairo size 1400,800 enhanced font "Sans,12"
set output "master.png"

set title "Benchmark Master Plot"
set grid
set xlabel "Step"
set ylabel "Framerate (Hz)"
set key outside right
set key box

legend(f) = (strlen(f) > 7 ? f[1:strlen(f)-7] : f)  # strip ".txt.csv" if present

plot for [f in "${gnuplot_list}"] f using 1:2 with linespoints linewidth 2 pointsize 0.7 title legend(f)

unset output
GP

echo "Wrote master plot: master.png"


# Usage:
#   gnuplot -e "FILES='benchmark_*.txt.csv' ; OUT='master.png'" plot_master.gp

set datafile separator ","
set terminal pngcairo size 1400,800 enhanced font "Sans,12"
set output OUT

set title "Benchmark Master Plot"
set grid
set xlabel "Step"
set ylabel "Framerate (Hz)"
set key outside right
set key box

# Helper: nicer legend names (strip ".txt.csv" if present)
legend(f) = (strlen(f) > 7 ? f[1:strlen(f)-7] : f)

plot for [f in FILES] f using 1:2 with linespoints linewidth 2 pointsize 0.7 title legend(f)

unset output


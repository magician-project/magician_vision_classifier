# Usage:
#   gnuplot -e "FILES='benchmark_*.txt.csv'" plot_each.gp
#
# Produces one PNG per CSV: <csvname>.png

set datafile separator ","
set terminal pngcairo size 1200,700 enhanced font "Sans,12"

do for [f in FILES] {
    # Build output name: f + ".png"
    set output sprintf("%s.png", f)

    # Title: base filename (strip path)
    title_str = f
    set title title_str

    set key off
    set grid
    set xlabel "Step"
    set ylabel "Framerate (Hz)"

    # Skip header row using "firstrow" columnhead:
    plot f using 1:2 with linespoints linewidth 2 pointsize 0.8
}
unset output


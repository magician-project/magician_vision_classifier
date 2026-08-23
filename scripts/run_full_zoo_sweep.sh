#!/usr/bin/env bash
# One run of every remaining model the codebase can build: 37 architectures, seed 42,
# 2 epochs, identical protocol to the 11-backbone sweep so the tables merge (49 total).
#
# This is a MAP, not a verdict machine -- see full_zoo_sweep.py. Single seed by request.
# Seed sd is 1.01 factory / 0.43 coverage, so a run locates a model to about +-2 / +-0.9;
# gaps of a few points are not orderings, gaps of ten are. The previous sweep's findings
# were all of the second kind, so n=1 is a reasonable instrument for finding outliers.
#
# ORDER IS FASTEST-FIRST by measured throughput. A breadth scan is most useful when it
# covers the most ground per hour, and the slowest architectures here are also the ones
# that cannot meet 23 Hz on the 5090 -- so stopping early costs only models that could not
# have shipped.
#
# RESUMABLE. Re-running this skips every model that already has a coverage table and picks
# up where it stopped. Three things had to be right for that to actually work, and the
# first two were not:
#
#   1. THE SKIP CHECK MUST HANDLE `timm/` NAMES. A model configured as `timm/tinynet_e`
#      makes the writers emit `fztinye_timm/tinynet_e_coverage.json` -- the slash in the
#      model name becomes a directory. The obvious flat-name check
#      (`fztinye_timm_tinynet_e_coverage.json`) matches nothing, so a restart would have
#      silently RE-TRAINED all 13 completed timm models, ~35 h of duplicated work
#      producing identical numbers. Both spellings are checked below.
#   2. AN INTERRUPTED RUN MUST BE CLEANED, NOT RESUMED. A model killed mid-sweep leaves a
#      checkpoint directory with some epochs in it and no coverage table. Retraining into
#      that directory mixes checkpoints from two different processes, and
#      checkpoint_save_top_k then keeps the best across both -- which is not the protocol
#      any other model in the map ran under. Partial directories are removed first.
#   3. THE CODE MUST NOT CHANGE UNDER THE MAP. See below.
#
# PROTOCOL FINGERPRINT. Every run records a hash of the training-relevant source plus the
# git commit into zoo_sweep_manifest.tsv. The point of a 49-model map is that one protocol
# produced all 49 numbers; if ModelZoo.py or the trainer changes partway through -- which
# is exactly what a mid-sweep `git pull` does -- the later models are not comparable with
# the earlier ones and nothing in the output would otherwise reveal it. On startup this
# script prints the current fingerprint and warns if completed runs used a different one.
set -u
GPU="${GPU:-2}"
LOGDIR=/storage/ammarkov/logs
cd "$(dirname "$0")/.." || exit 1
mkdir -p "$LOGDIR"
MANIFEST=experiments/zoo_sweep_manifest.tsv

# Files whose contents define the training protocol. A change to any of these makes runs
# before and after incomparable. Paths are repo-root-relative; the `cd` above guarantees
# that is the cwd.
PROTOCOL_FILES="mvc/core/model_zoo.py mvc/train.py mvc/core/datasets.py mvc/core/lit_classifier.py analysis/eval/score_checkpoints.py analysis/eval/eval_coverage.py"

# A MISSING protocol file must be fatal, not silent. These paths were the pre-refactor root
# filenames until the layout move, and `md5sum missing 2>/dev/null | md5sum` is perfectly
# happy to hash nothing: the fingerprint silently becomes d41d8cd98f00, the md5 of empty
# input, for every run. The mechanism would still print a fingerprint and still compare it
# against the manifest -- it just would no longer detect anything, which is the one failure
# this whole mechanism exists to prevent.
#
# This check is deliberately NOT inside code_fingerprint(): that is called as `$(...)`, and
# an `exit` inside a command substitution only leaves the subshell -- the sweep would print
# the error and then carry on with an empty fingerprint. It must be its own statement.
require_protocol_files() {
    local missing=""
    for f in $PROTOCOL_FILES; do
        [ -f "$f" ] || missing="$missing $f"
    done
    [ -z "$missing" ] && return 0
    echo "PROTOCOL FILE(S) NOT FOUND:$missing" >&2
    echo "The protocol fingerprint cannot be computed, so this sweep would produce runs" >&2
    echo "that cannot be compared against the existing map. Fix the paths." >&2
    exit 1
}

code_fingerprint() {
    # shellcheck disable=SC2086
    md5sum $PROTOCOL_FILES | md5sum | cut -c1-12
}
git_rev() { git rev-parse --short HEAD 2>/dev/null || echo nogit; }

require_protocol_files
FP="$(code_fingerprint)"
REV="$(git_rev)"
echo "=== $(date -Is) protocol fingerprint $FP (git $REV)"
if [ -f "$MANIFEST" ]; then
    prev=$(awk -F'\t' 'NR>1 {print $3}' "$MANIFEST" | sort -u | grep -v "^$FP$" || true)
    if [ -n "$prev" ]; then
        echo "!!! WARNING: completed runs used a DIFFERENT protocol fingerprint: $prev"
        echo "!!! The map is no longer one comparable table. Re-run the affected models,"
        echo "!!! or accept that early and late entries were produced by different code."
    fi
else
    printf 'run\tmodel\tfingerprint\tgit\tstarted\n' > "$MANIFEST"
fi

. "$(dirname "$0")/gpu_lock.sh"
gpu_wait

# Queue order comes from the generator, which sorts by benched throughput. Derived here
# rather than hardcoded so the two cannot drift apart.
mapfile -t CFGS < <(python3 - <<'PY'
import json
import sys

# Both of these were pre-refactor spellings and both fail SILENTLY here. `mapfile` is
# happy with empty input, so an ImportError or a FileNotFoundError in this heredoc does
# not stop the sweep -- it reports "queue: 0 models", skips the loop, and exits 0 having
# trained nothing. Hence the explicit stderr + non-zero exit below.
try:
    from analysis.sweeps.full_zoo_sweep import TAGS, BENCH
    from mvc.core.artifact_paths import find_artifact
except Exception as exc:                                    # noqa: BLE001
    print(f'queue generator import failed: {exc}', file=sys.stderr)
    raise SystemExit(1)

bench = find_artifact(BENCH)
if not bench:
    print(f'{BENCH} not found: the queue is ordered by benched throughput and cannot '
          f'be built without it', file=sys.stderr)
    raise SystemExit(1)

hz = {r['model']: r.get('hz_step16', 0.0) for r in json.load(open(bench))['rows']}
for m in sorted(TAGS, key=lambda m: -hz.get(m, 0.0)):
    print(f'fz{TAGS[m]}_{m.replace("/", "_")}.json')
PY
)
echo "=== $(date -Is) queue: ${#CFGS[@]} models"
# An empty queue means the generator failed; training nothing and exiting 0 looks like a
# completed sweep in the log.
[ "${#CFGS[@]}" -gt 0 ] || { echo "!!! queue is empty -- generator failed, aborting"; exit 1; }

# A run is complete iff a coverage table exists. `tidy_experiments.py` files finished
# artifacts under experiments/<campaign>/<run>/, so a plain root-only `-f` check goes
# blind on anything already tidied and would retrain it on a resumed queue. find_artifact
# checks the root first, then experiments/, and already handles the `timm/`
# slash-in-model-name spelling (see note 1 above) -- so one call covers both.
coverage_exists() {
    local run="$1"
    python3 -c "
from mvc.core.artifact_paths import exists
import sys
sys.exit(0 if exists(sys.argv[1] + '_coverage.json') else 1)
" "$run"
}

for CFG in "${CFGS[@]}"; do
    run="${CFG%.json}"
    [ -f "$CFG" ] || { echo "!!! missing $CFG, skipping"; continue; }
    if coverage_exists "$run"; then
        echo "=== $(date -Is) $run already complete, skipping"
        continue
    fi

    # Interrupted earlier? Clear the partial checkpoint dir so this run trains under the
    # same conditions as every other entry in the map.
    ckdir=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint_dir'])" "$CFG" 2>/dev/null || true)
    if [ -n "$ckdir" ] && [ -d "$ckdir" ]; then
        echo "=== $(date -Is) [$run] clearing partial checkpoints from an interrupted run: $ckdir"
        rm -rf "${ckdir:?}"
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' "$run" "${CFG%.json}" "$FP" "$REV" "$(date -Is)" >> "$MANIFEST"

    log="$LOGDIR/${run}_$(date +%Y%m%d_%H%M).log"
    echo "=== $(date -Is) [$run] train -> $log"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m mvc.train "$CFG" > "$log" 2>&1
    rc=$?
    echo "=== $(date -Is) [$run] train rc=$rc"
    [ $rc -eq 0 ] || { echo "!!! $run FAILED, continuing to next model"; continue; }

    echo "=== $(date -Is) [$run] score_checkpoints"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.score_checkpoints "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] score rc=$?"

    echo "=== $(date -Is) [$run] eval_coverage"
    CUDA_VISIBLE_DEVICES="$GPU" python -u -m analysis.eval.eval_coverage "$CFG" >> "$log" 2>&1
    echo "=== $(date -Is) [$run] coverage rc=$?"

    echo "--- [$run] (incumbent convnext_pico: 9.24 miss@FA5, TIER_A 73.59) ---"
    grep -aA 8 'epoch   val_loss' "$log" | tail -10
    grep -aA 3 'TIER_A (honest' "$log" | tail -4
done

echo "############ $(date -Is) FULL ZOO SWEEP COMPLETE -- run: python -m analysis.sweeps.full_zoo_report"

# experiments/

Finished run artifacts, filed out of the repo root by `tidy_experiments.py`.

Layout is `experiments/<campaign>/<run-family>/`, where the run family is the leading token
of the run's `name` field. Everything one run produced — threshold curve, confusion matrix
and its four plots, coverage table, exported weights — sits together.

| campaign | what it is |
|---|---|
| `aug26_fulltrain` | The current anchor (`anc`, stride 4) and stride-2 (`ancs2`) full trains on Aug26_78K, plus per-epoch scoring artifacts (`*_ep<N>_*`, `epochcov_*`). **These are live references** — `seed_replicates_report.py` reads them. |
| `aug26_screens` | The 4-arm × 3-seed 30k modifier sweep (`s26*`). Read by `aug26_sweep_report.py`. The screen has sd 1.83 miss@FA5 on Aug26 and resolves nothing on its own — see PLAN.md. |
| `bench_backbones_tile48` | The tile-48 backbone/throughput bench (`tz*`). The input to the deferred model sweep. |
| `legacy_modifier_sweep` | The `mx*` sweep, run on the frame-disjoint split with recording leakage. Its DoLP sign convention is the opposite of the current one — do not compare deltas across the two without reading PLAN.md first. |
| `legacy_phase_sweeps`, `legacy_finetune`, `legacy_forth_altinay`, `legacy_misc` | The FORTH+Altinay era, superseded by the Aug26 campaign. |

## Reading an archived artifact

Don't hardcode these paths. Use `artifact_paths.find_artifact(name)`, which checks the repo
root first and then this tree, so a report works whether a run has been filed away or was
written five minutes ago.

## Adding to it

`tidy_experiments.py` is idempotent and re-runnable. It refuses to move anything git
tracks, anything on its live-files list (the active configs, `val_coverage_frames.json`,
`recommended_configuration.json`), and — importantly — anything younger than
`--min-age-hours` (default 6), so a running job's outputs can never be moved out from under
it. After a queue finishes:

    python tidy_experiments.py --dry-run
    python tidy_experiments.py --apply

## What IS in git

Two subdirectories are tracked, because they are provenance rather than output:

| dir | contents |
|---|---|
| `configs_frozen/` | The inputs the campaign depends on: `anc_convnext_pico.json` (the currently recommended deployment model, including its calibrated gate thresholds), `ancs2_convnext_pico.json`, the two inference benches, `recommended_configuration.json`, and **`val_coverage_frames.json`**. |
| `configs_runs/` | One config per training run — the recipe behind every number in `PLAN.md`. 55 files, 356 K. |

**`val_coverage_frames.json` is the important one.** It defines which frames are carved out
of training to form the coverage validation set. Two boxes using different copies of this
file are not measuring the same thing, and every cross-box coverage comparison silently
becomes meaningless — so it is tracked, and every config references it by the path
`experiments/configs_frozen/val_coverage_frames.json` rather than keeping a loose copy in
the repo root. `eval_coverage.py` resolves it through `artifact_paths.find_artifact`, which
checks the root first, so a local override still works if you need one.

If you change it, every prior coverage number in `PLAN.md` becomes incomparable. Write a new
file instead (`build_coverage_val.py --out ...`) and say so in `PLAN.md`.

## What is NOT in git

The rest of this tree is gitignored. Everything in it is regenerable from
a config plus a checkpoint (`score_checkpoints.py`, `eval_coverage.py`), and the archived
run configs are the same kind of file being removed from the repo root. The results that
must not be lost live in `PLAN.md`, not here. To commit something from in here anyway,
`git add -f` it.

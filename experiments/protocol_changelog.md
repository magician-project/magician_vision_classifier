# Protocol fingerprint changelog

`scripts/run_full_zoo_sweep.sh` hashes the training-relevant source
(`ModelZoo.py`, `trainMagicianVisionClassifierTorch.py`, `Datasets.py`,
`score_checkpoints.py`, `eval_coverage.py`) into a fingerprint recorded per run in
`zoo_sweep_manifest.tsv`, and warns when it changes. The warning is only useful if every
change is explained here — an undocumented transition should be treated as real drift.

| fingerprint | from | change | affects results? |
|---|---|---|---|
| `259fc61ae980` | campaign start | baseline for all 49-model map runs 1–24 | — |
| `61a0a9ff1407` | 2026-08-18 08:25 | **export-only.** The trainer's inline `zip -r … check=False` archive block replaced by a call to `export_models.export_run()`. | **No.** |

## Detail on `61a0a9ff1407`

The trainer's old block had no name sanitisation (a `timm/<x>` model produced an archive
path containing a slash, so `zip` failed) and discarded the exit code, so 13 trained
models silently ended up with no archive. The replacement sanitises names, includes the
`.json` sidecars, and verifies the finished archive.

**The change is strictly post-training packaging.** It runs after training, scoring and
coverage have completed and written their outputs; it reads artifacts and writes a zip. It
cannot alter a metric. Verified by diff: the only lines touched are inside the archive
block.

## Known inaccuracy in the manifest

`run_full_zoo_sweep.sh` computes the fingerprint **once at startup**. The sweep running
across this change started at 2026-08-18 07:55 under `259fc61ae980`, so every row it
writes — including models trained after 08:25 — records the old value. Those rows are
wrong about the fingerprint but right about the science, for the reason above.

Affected: any zoo run started after 2026-08-18 08:25 in the current sweep invocation.
On the next invocation the manifest will correctly record `61a0a9ff1407`, and the drift warning
will fire once against rows 1–24 — expected, and explained by this file.

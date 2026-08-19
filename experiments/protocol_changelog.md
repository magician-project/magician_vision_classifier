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
| `bdf296e227da` | 2026-08-19 05:30 | **refactor, behaviour-preserving.** The eleven hand-written config→kwargs blocks collapsed into `Classifier.from_config()`. | **No — proven, see below.** |

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


## Detail on `bdf296e227da`

Eleven call sites each hand-translated a config dict into ~31 constructor kwargs, and they
disagreed. Four eval tools read the derived-channel flags from the config TOP LEVEL, where
no config has ever carried them (167 have them under `hparams`, zero at the top), and never
passed `monochrome` at all — which is applied at eval, not just training, and preserves
tensor shape, so a mono-trained model scored by those tools loaded perfectly and computed
the wrong thing.

**Why this does not affect the 49-model map**, despite touching two protocol files:

* `test_classifier_from_config.py` asserts `from_config()` produces byte-identical kwargs to
  the trainer's original block **on all 167 real configs**. The conversion is a no-op by
  construction, not by inspection.
* Verified end to end as well: `eval_coverage.py` re-run on `fzr18_resnet18` after the
  conversion produced **byte-identical rows** to the stored result.
* The extra kwargs `from_config` supplies to the scorers are the training-only augmentation
  knobs (`noise_std`, `gain_jitter`, `polar_flip`, `polar_rot`, `channel_jitter`), every one
  of which is gated on `self.training` and therefore inert under `.eval()`.

`save_hyperparameters()` was also added, so new checkpoints carry their own architecture.
Checkpoints written before it have none; `Classifier.load_for_eval()` falls back to a config
and **raises if neither source has them**, rather than silently rebuilding from `__init__`
defaults — which would be the same class of bug in new clothes.

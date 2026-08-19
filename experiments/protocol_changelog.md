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

### Follow-up (2026-08-19) — three defects in the refactor itself, no fingerprint change

An audit of the conversion found three things. Fingerprint stays `bdf296e227da`: none of the
five files touched here is in `PROTOCOL_FILES`, and the sweep is unaffected.

1. **`classifierPnm.py` was still hand-building kwargs.** It is the live/offline inference
   path behind the ROS node and the ensemble — the one place where getting the architecture
   wrong reaches a customer rather than a spreadsheet. It had already been audited and
   corrected once (its comment records being burned by `allclass_customwide`), and it was
   *still* missing `timm_stem_stride` and `custom_wavelet_stem`, both architecture-changing.
   Six real configs set `timm_stem_stride=2`; those models rebuilt at stride 4 here. Now
   converted, and verified against all 167 configs: no key the old block set changes value,
   and the two missing knobs start being honoured.

2. **`torch.load` without `weights_only=False` (5 sites).** Torch 2.6+ defaults it to True.
   The failure mode is nastier than it first looks: existing checkpoints load *fine* under
   the default, because they carry no `hyper_parameters` at all. It is precisely the
   checkpoints written by the new `save_hyperparameters()` that break — Lightning stores an
   `AttributeDict`, an arbitrary class, and `weights_only=True` refuses it outright. So the
   bug would pass every test against the existing zoo and fail on the first run trained
   after this change. Confirmed on torch 2.8.0 against a real checkpoint.

3. **`config_to_kwargs` / `_config_to_kwargs` were the same function.** The split was a
   leftover: it existed because `inspect.signature(cls.__init__)` broke under a patched
   `__init__`, which the `__func__` fix already solved. Two names for one translation, in
   the module whose purpose is to have one. Collapsed.

Also: `load_for_eval()` now forces `pretrained=False`. The state dict overwrites every
parameter immediately afterwards, so fetching ImageNet weights first is a network round-trip
— one that fails outright on an offline deployment box — in exchange for nothing.

`load_for_eval()` was previously untested against a real file. It now round-trips a real
`anc_convnext_pico` checkpoint end to end.

### 2026-08-19 — the validation split and the KPI each become one definition

Fingerprint `bdf296e227da` -> `bf82866e7d44`. Two protocol files changed (`Datasets.py`,
`score_checkpoints.py`, `eval_coverage.py`, `trainMagicianVisionClassifierTorch.py`), and
both changes are asserted-equivalent rather than believed-equivalent.

**The validation split** was rebuilt by hand in the trainer and mirrored in
`score_checkpoints.py` and `eval_ema_tta.py`, coupled by comments naming line numbers in
another file — line numbers that were already stale. `eval_coverage.py` and
`mine_hardneg.py` shared the class-space prefix of the same chain. It now lives in
`Datasets.build_train_val` / `build_val_only` / `load_training_dataset`.

One subtlety that had to be preserved exactly: the trainer seeds **after** any RAM preload
and **before** the split, and the global RNG state that call leaves behind is what the model
is initialised from. Moving it or making it conditional would change the weights a run
starts with, not just which tiles are held out. It is now unconditional inside
`build_train_val`, at the same point in the sequence.

**The KPI** — miss@FA, the number every decision in PLAN.md turns on — had **eight**
implementations, not the five first counted: `phase2_select.py`, `score_checkpoints.py`,
`eval_domain_split.py` (twice), `eval_ema_tta.py`, and inline copies in `eval_coverage.py`,
`evaluateDetection.py` and `detection_ensemble.py`. Seven reporting tools imported it from
`phase2_select`, a one-off selection script that therefore could not be moved or deleted.
All now call `Metrics.py`.

They were not eight copies of one thing. They were **three estimators**:

| form | rule | used by |
|---|---|---|
| curve | interpolate `detected` at target FA from a saved sweep | phase2_select, score_checkpoints |
| quantile | threshold at the (1-fa) quantile of clean scores | eval_domain_split ×2, eval_coverage, evaluateDetection, detection_ensemble |
| constrained max | best detection subject to false_alarm <= fa | eval_ema_tta |

The two curve implementations agree **exactly** (229 real curves, 0.000000000000 pp). The
quantile and constrained-maximum forms **do not**, and the gap is not academic:

```
distribution      fa   quantile     sweep   gap (pp)  realised FA
continuous      0.05     64.750    64.750      0.000       0.0500
2dp ties        0.05     65.325    66.950      1.625       0.0512  <-- over budget
1dp ties        0.10     40.425    59.325     18.900       0.1593  <-- over budget
```

The last column is the finding. Under ties the **quantile threshold overshoots the
false-alarm budget the KPI claims to match** — at 1dp ties it reports a miss rate at a
nominal FA of 10% while actually firing on 15.9% of clean tiles. The constrained-maximum
rule refuses to exceed the budget, which is why it reports a worse (honest) number. Softmax
outputs are float32 and mostly continuous, where the gap is ~0.025 pp, but they saturate
near 0 and 1 — exactly where ties live.

Both estimators are kept, named differently, so the choice is visible at the call site
instead of being an accident of which file the function was copied from. **No published
number changes**: every converted site keeps the estimator it already used. Whether to
unify on the constrained-maximum rule is a decision to make deliberately, with the table
above in hand, and would require re-deriving affected figures.

### 2026-08-19 — artifacts are written where they belong

Fingerprint `bf82866e7d44` -> `a29ffda5d70c`. The writers now emit into
`experiments/<campaign>/<run>/` instead of the repo root, via `artifact_paths.out_path()`.
No computation changed — only where the results land.

The root held **1017 entries**; it now holds 174. 933 files / 2.86 GB were filed into 48
campaign directories by `tidy_experiments.py`, and future runs skip that step entirely.

Three things this had to get right, each of which would have been expensive:

* **Configs are inputs, not artifacts.** 56 run configs sat in the root looking exactly
  like artifacts (`fzv2nano_timm_convnextv2_nano.json` is a config;
  `..._coverage.json` is not). `export_models.discover()` globs them, and the sweep's
  restart skip-check decides what NOT to re-train from them — filing them away would have
  made a resumed sweep re-train every finished model. `is_run_config()` reads the file
  rather than guessing from the name.
* **The `timm/` slash is fixed at the source.** `out_path` sanitises, so a `timm/x` run no
  longer creates a stray `fzx_timm/` directory. The 48 that already existed were renamed
  on the way out — moving them verbatim would have collided in `find_artifact`'s
  basename index, where first-wins would have made one of any two runs sharing a backbone
  unreachable.
* **`plotTool.py` wrote to the cwd.** It took `os.path.basename()` of its input, so the
  plots for a run whose JSON had moved were still written to the root, re-scattering what
  the move existed to fix.

`export_models` checks the run directory first and falls back to the root, so the two
layouts coexist and older runs still export. Verified by rebuilding `fzcnxtiny` — whose
artifacts had already moved — into a complete 10-member archive.

**Residual risk, stated plainly:** the writer change takes effect for the 6 models still
queued, and a path fault would surface only after ~5 h of training. `out_path` was
therefore exercised against all six real run names before this was left in place.

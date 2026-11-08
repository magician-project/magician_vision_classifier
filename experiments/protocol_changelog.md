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
| `0866656c6862` | 2026-10-05 | **opt-in knobs.** `hier_heads` / `hier_type_residual` added to `HierConvNeXt`'s head (kbach). Defaults (1 / false) rebuild the previous head exactly. | **No** for every existing config and checkpoint — see below. |
| `6c7e9a7f164e` | 2026-10-05 | **opt-in loss.** `hier_sev_weight` (hier_* only): training loss `focal(P(type)) + λ·CE(P(sev\|type))` instead of focal on the flat logits. `HierConvNeXt.forward` refactored onto a new `log_parts()`. | **No** unless a config sets `hier_sev_weight` — default `null` keeps the flat focal loss; forward output unchanged (asserted). |
| `d154bf417dc9` | 2026-10-06 | **opt-in head.** `hier_hybrid` / `hier_aux_weight` (hier_* only): a flat Linear 512→C head sets P(clean); the type/severity branches split the defect mass. `_HierHead.forward` split into pooling + `_branches`; `HierConvNeXt` gains `hybrid_parts()`. | **No** unless a config sets `hier_hybrid` — default off rebuilds the previous head and loss exactly (all prior tests unchanged). |
| `bae570245a71` | 2026-10-06 | **resume support.** `mvc.train` accepts `--resume <ckpt>` / `--resume-epoch N` (new `mvc/core/resume.py`); `trainer.fit` gets `ckpt_path=`, and the TensorBoard wipe is skipped on a resume. | **No** for any run started without the flags (`ckpt_path=None` is Lightning's default). A resumed run is not bit-identical to an uninterrupted one — see below. |
| `bd0dedb52d3e` | 2026-10-06 | **opt-in loss term.** `penalize_false_defect` (top-level config key, like `penalize_false_clean`): adds `−log P(clean)` averaged over a batch's clean tiles — the mirror of `penalize_false_clean`. | **No** unless a config sets it — default 0 adds nothing; the `penalize_false_clean` code path is untouched. Guarded in `test_hier_hybrid`. |
| `e6218d2c757b` | 2026-10-06 | **opt-in training knob.** `hier_detach_branches` (needs `hier_hybrid`): the type/severity branches read a detached pooled vector, so only the flat loss trains the body, pooling and flat head — detection learns as in convnext_pico; the hierarchy adds grading only. | **No** unless set. Forward outputs are identical with or without it (asserted), so it changes training only. Guarded by `test_hier_detach_branches` (hier loss → zero body gradient; control without detach reaches the body). |
| `3cd9dd717293` | 2026-10-07 | **output-only.** `eval_coverage.py` also writes `labels` and the full argmax `confusion` matrix (rows = true, columns = predicted) into `<run>_coverage.json`, so type/severity accuracy among defects can be scored. | **No.** Every existing field and number is unchanged; two keys are added. Coverage files written before this lack them — re-run `eval_coverage.py` on a run to add them. |
| `29f4a9cae6d6` | 2026-10-08 | **opt-in architecture knob.** `timm_pad_downsample` (hparams; hier and plain timm models): a forward pre-hook zero-pads odd inputs of every 2×2/s2 downsample conv to even. At 48 px, stage 3 otherwise sees only the top-left 2×2 of stage 2's 3×3 map (5 of 9 positions dropped); with it, all 9, and stage 3 outputs 2×2. | **No** unless set. No parameters and no state-dict changes (pretrained weights load as-is); 114.49 → 143.10 MFLOPs/tile for the hybrid. Guarded by `test_pad_downsample` (9/9 positions reach stage 3, control 4/9). |
| `50fc7be4e30e` | 2026-10-08 | **opt-in loss term.** `rank_loss_weight` / `rank_hard_fraction` (top-level config keys): hard-negative pairwise ranking loss on the detection score s = log P(defect) − log P(clean) — mean softplus(s_clean − s_defect) over every defect × the hardest `rank_hard_fraction` of the batch's clean tiles (a partial-AUC surrogate for the low-false-alarm region miss@FA reads). Shift-invariant in s, so it reorders without recalibrating P(clean). | **No** unless `rank_loss_weight` > 0 (default 0). Guarded by `test_rank_loss` (hand value, shift invariance, gradient only on the hardest cleans + defects). |
| `8abdb92c6cf7` | 2026-10-09 | **opt-in augmentation.** `scale_jitter` (hparams; `polarization.augment_train_batch`, not itself fingerprinted): on half the batch, a random zoom-in crop of side s·48, s ~ U[1 − scale_jitter, 1], resampled bilinearly back to 48×48 with one affine map for all four channels. Motivation: camera distance (tile metadata) varies between recordings, so apparent defect size does too. | **No** unless set (default 0). Training-only; inference untouched. Guarded by `test_scale_jitter` (off/eval identity, uniform tile and DoLP preserved, identical map on all channels). |
| `7752eda92873` | 2026-10-10 | **robustness + recovery.** `mvc.train` gains `--finalize <ckpt>` / `--finalize-epoch N` (restore the last epoch's checkpoint, train nothing, run the post-training steps) and, on Windows, a guard that logs instead of raising when `TerminateProcess` refuses to stop a DataLoader worker at teardown (`PermissionError [WinError 5]`, which killed `kbachpicoscj1eps7` after training, before any results were written; recovered with `--finalize-epoch 0`). | **No.** Training is untouched; the guard only affects teardown after the model is saved; a finalized run's outputs are those of the saved checkpoint (records `finalized_from`). Guarded by `tests/test_resume.py` (0 extra steps, no new checkpoint, weights unchanged, best checkpoint restored). |
| `b5e05d1a92fc` | 2026-10-12 | **opt-in architecture knob.** `hier_sev_per_type` (hparams; `hier_*` models only): the severity branch's output layer emits one grade distribution per defect type (T×S logits, softmax per type) instead of a single S-vector shared by every type. With the shared head, P(grade | NegativeDent) and P(grade | PositiveDent) are identical by construction, so the hierarchy cannot express type-specific grading. Also: `tests/test_classifier_from_config.py` [5] now compares raw head logits (diff > 1e-6) instead of `allclose` on outputs, which sat at the tolerance edge at init. | **No** unless set (default False). +(T−1)(d+1)S parameters (+774 for the 10-class pico hybrid); a default checkpoint refuses to load into a per-type head and vice versa. Existing `kbachhybpenscj1ep*` checkpoints verified to load strictly, unchanged. Guarded by `test_sev_per_type`. |
| `8b7c71b1bcd5` | 2026-10-12 | **opt-in loss term.** `distill_teacher` / `distill_weight` / `distill_temperature` (top-level config keys): knowledge distillation, T²·KL(softmax(teacher/T) ‖ softmax(student/T)) on the student's output (for the hybrid, the combined log-probs whose P(clean) is the flat head's), the teacher seeing the same augmented tile. The teacher is a trained checkpoint loaded lazily on the first training step and kept outside the module tree: not in the state dict, not optimised, never loaded by `load_for_eval`. Motivation: plain convnext_tiny beats the pico hybrid on factory and coverage (3 paired seeds) at 3.3× the FLOPs. | **No** unless `distill_weight` > 0 (default 0). Training slows by the teacher's forward pass; inference and checkpoints unchanged. Guarded by `test_distill`. A teacher must itself have been trained with the coverage frames excluded, or coverage leaks into the student. |
| `ced7873055da` | 2026-10-13 | **opt-in extension.** `distill_teacher` may be a LIST of checkpoints: the target is the mean of the teachers' temperature-softened distributions, log(mean_k softmax(t_k/T)), in log space. Motivation: the 3-seed convnext_tiny average beat every single tiny with no training (factory miss@FA5 5.02 vs 6.53, coverage TIER_A @FA5 81.37 vs 78.54; `scripts/kbach_teacher_ensemble.py`). | **No** unless a list is given: a single path (or a one-element list) gives exactly the previous loss. Guarded by `test_distill` (2-teacher target matches hand, one-element list == single teacher, teachers outside the state dict). |
| `88ab0b99ea3a` | 2026-10-16 | **opt-in training add-on.** `ema_decay` (top-level config key): an exponential moving average of the weights, ema <- d*ema + (1-d)*w after every optimizer step (0.9998 ~ a 5,000-step horizon). Training itself is unchanged; the float32 shadow lives outside the module tree, so state dicts and the normal checkpoints of the raw weights are as before (the shadow is additionally stored under `ema_state` in Lightning checkpoints so a resume keeps it). At each epoch end the averaged weights are written as an ordinary checkpoint to `<checkpoint_dir>_ema/`. Motivation: the campaign trains at a constant learning rate to the last step, so the shipped weights carry end-of-run noise. | **No** unless `ema_decay` > 0 (default 0). Guarded by `test_ema` (exact recurrence, outside the state dict, strict load of the EMA checkpoint, survives resume). |

## Detail on `bae570245a71`

A resume restores weights, AdamW state, global step, loop progress and the ModelCheckpoint
state (so best-epoch selection spans the resume), but not the Python/NumPy RNG that the
BalancedBatchSampler and the augmentations draw from: epochs after the resume see a different
sample order than an uninterrupted run would. `check_resume` refuses weights-only files, any
architecture difference between checkpoint and config (including shape-silent knobs such as
`monochrome` / `hier_heads`), and resumes with no epochs left. The resumed run's JSON records
`resumed_from` / `resumed_after_epoch`. Guarded by `tests/test_resume.py`.

## Detail on `d154bf417dc9`

Replicates the "Hybrid Pico Head" design (kbach). Output, as normalised log-probabilities:
`P(clean) = P_flat(clean)`, `P(defect c) = (1 − P_flat(clean)) · P_hier(c | defect)`, with
`P_hier` the original type×severity distribution renormalised over the defect classes (both
`log(1 − P(clean))` terms are logsumexps over the defects). Training loss:
`focal(flat logits) + hier_aux_weight · focal(hier log-probs)` (+ `penalize_false_clean` on the
combined P(clean), which is the flat head's). For 10 classes: 8,690,611 params (+5,130 over the
original hier head, +150,761 / +1.77% over plain convnext_pico), 114.49 MFLOPs/tile — matches
the design page. Guarded by `test_hier_hybrid` (normalisation, P(clean) identity, defect-split
identity, loss composition, checkpoint round trip, refusals). Unlike `hier_heads`, the extra
`flat_out` layer changes the state dict, so a hybrid checkpoint cannot load silently into a
plain hier model.

## Detail on `6c7e9a7f164e`

`hier_sev_weight` replaces only the TRAINING loss of a `hier_*` model; `val_loss`, the
checkpoint monitor (`val_detect_auroc`), the gate and inference are untouched. The terms are
batch means (severity summed over graded tiles, divided by batch size), so at γ = 0, λ = 1
the split loss equals flat cross-entropy exactly — asserted by `test_hier_sev_weight`
(1.75898 vs 1.75898). At γ = 2 it differs from the flat focal loss by design: the focusing
factor uses P(type) instead of P(type, sev). `penalize_false_clean` still applies on top. A
non-hier model with the knob set raises. Runs trained with it log `train_type_loss` and
`train_sev_loss` separately.

## Detail on `0866656c6862`

Two hparams for `hier_*` models, both defaulting to the head as it was:

* `hier_heads` — `num_heads` of the type→severity cross-attention (was hard-coded 1). Must
  divide `hier_d / hier_tokens`; an indivisible value raises rather than rounding.
* `hier_type_residual` — `h_type = z + type_mlp(z)`.

Neither changes a parameter shape or the FLOP count (8,685,481 params, 114.48 MFLOPs/tile
for hier_convnext_pico in every combination, measured with `torch.utils.flop_counter`). That
is the hazard: a non-default checkpoint loads `strict=True` into a default-built model and
silently computes a different function. Both are persisted by `save_hyperparameters()`,
read by `from_config()`, and guarded by `test_hier_knobs` in
`tests/test_classifier_from_config.py` (knob reaches the module, survives a checkpoint round
trip, and is NOT output-identical to the 1-head model on the same weights).

Fingerprint computed on the Windows workstation; if that checkout's line endings differ from
the Linux box's, the Linux hash of the same source will differ too.
Also in this change, outside the fingerprinted set: `HDF5Dataset` pickling (Windows spawn
workers) and a `/` arcname fix in `mvc/export.py` — packaging/IO only, no metric effect.

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

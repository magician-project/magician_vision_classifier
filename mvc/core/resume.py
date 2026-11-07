"""Resume a training run after a given epoch (2026-10-06, kbach).

Motivation: an unexpected power-off on the Windows workstation killed the second epoch of a
2-epoch convnext_pico run 50 minutes in; the first epoch's checkpoint was intact but the
trainer had no way to continue from it, so the only option was ~6 h of retraining.

The epoch checkpoints ModelCheckpoint writes are FULL Lightning checkpoints: weights, the
AdamW state, global step, fit-loop progress and the ModelCheckpoint callback's own state
(best score / best path). `trainer.fit(..., ckpt_path=...)` restores all of that, so a resumed
run continues the same optimisation, and its best-epoch selection spans the epochs from before
and after the resume. What it does NOT restore is the Python/NumPy RNG state the
BalancedBatchSampler and the augmentations draw from: the epochs after a resume see a different
(equally valid) sample order than an uninterrupted run would. Resumed results are therefore
comparable in kind but not bit-identical -- the same caveat as any rerun with these stochastic
augmentations on.

Usage (see mvc/train.py):
    python -m mvc.train <config.json> --resume-epoch 0        # continue after epoch 0
    python -m mvc.train <config.json> --resume <file.ckpt>    # explicit checkpoint
Raise `hparams.training_epochs` in the config to train beyond the original length.
"""
import glob
import os
import re

import torch

# Architecture-defining constructor args. A checkpoint that differs on any of these is a
# different model; several (hier_heads, monochrome, DoLP...) do NOT change parameter shapes,
# so a mismatched resume would load strict=True and silently train the wrong network.
_ARCH_KEYS = ('model', 'num_classes', 'clean_class', 'class_names')


def find_epoch_checkpoint(ckpt_dir, epoch):
    """The `epoch=<epoch>-...ckpt` file in ckpt_dir. Raises if absent or ambiguous."""
    if not ckpt_dir or not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f'checkpoint_dir {ckpt_dir!r} does not exist')
    hits = [p for p in glob.glob(os.path.join(ckpt_dir, '*.ckpt'))
            if re.match(rf'epoch={int(epoch)}-', os.path.basename(p))]
    if not hits:
        have = sorted(os.path.basename(p) for p in glob.glob(os.path.join(ckpt_dir, '*.ckpt')))
        raise FileNotFoundError(f'no epoch={epoch} checkpoint in {ckpt_dir}; found: {have}')
    if len(hits) > 1:
        raise ValueError(f'{len(hits)} checkpoints for epoch={epoch} in {ckpt_dir}: '
                         f'{[os.path.basename(h) for h in hits]} -- pass --resume <file>')
    return hits[0]


def parse_resume_args(argv):
    """Strip --resume PATH / --resume-epoch N from argv. Returns (argv_rest, path, epoch)."""
    rest, path, epoch = [], None, None
    it = iter(argv)
    for a in it:
        if a in ('--resume', '--resume-epoch'):
            try:
                v = next(it)
            except StopIteration:
                raise SystemExit(f'{a} needs a value')
            if a == '--resume':
                path = v
            else:
                epoch = int(v)
        elif a.startswith('--resume='):
            path = a.split('=', 1)[1]
        elif a.startswith('--resume-epoch='):
            epoch = int(a.split('=', 1)[1])
        else:
            rest.append(a)
    if path is not None and epoch is not None:
        raise SystemExit('give either --resume or --resume-epoch, not both')
    return rest, path, epoch


def parse_finalize_args(argv):
    """Strip --finalize PATH / --finalize-epoch N from argv. Returns (argv_rest, path, epoch).

    Finalize (2026-10-10): the run's last epoch finished and its checkpoint was written, but the
    script died before the post-training steps (on Windows a DataLoader worker refused to be
    terminated at teardown -- PermissionError [WinError 5] in TerminateProcess). Finalizing
    restores that checkpoint, trains nothing, and runs the normal post-training steps (best
    checkpoint restore, .pth, confusion, threshold sweep, gate, run JSON): the outputs the
    original run would have written, from a fixed model, so no retraining is needed."""
    rest, path, epoch = [], None, None
    it = iter(argv)
    for a in it:
        if a in ('--finalize', '--finalize-epoch'):
            try:
                v = next(it)
            except StopIteration:
                raise SystemExit(f'{a} needs a value')
            if a == '--finalize':
                path = v
            else:
                epoch = int(v)
        else:
            rest.append(a)
    if path is not None and epoch is not None:
        raise SystemExit('give either --finalize or --finalize-epoch, not both')
    return rest, path, epoch


def check_resume(ckpt_path, classifier, max_epochs, arch_keys=None, finalize=False):
    """Validate that ckpt_path can resume `classifier`'s run; return (epoch, global_step).

    Refuses (ValueError) a weights-only file, an architecture mismatch (every hparam the
    Classifier treats as architecture-defining, plus model / class list), and a resume that
    has no epochs left to train.
    """
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    missing = [k for k in ('state_dict', 'optimizer_states', 'loops', 'epoch') if k not in ck]
    if missing:
        raise ValueError(f'{ckpt_path} is not a full training checkpoint (missing {missing}); '
                         f'it can be evaluated but not resumed')
    stored = dict(ck.get('hyper_parameters') or {})
    if not stored:
        raise ValueError(f'{ckpt_path} carries no hyper_parameters, so its architecture '
                         f'cannot be checked against the config -- refusing to resume')
    now = dict(classifier.hparams)
    keys = arch_keys or (tuple(type(classifier)._HPARAM_KWARGS) + _ARCH_KEYS)
    # Training-only knobs may legitimately change on resume (e.g. a lower penalty); they
    # are not architecture, and the augmentation ones are inert outside training.
    training_only = {'noise_std', 'noise_clip', 'gain_jitter', 'polar_flip', 'channel_jitter', 'scale_jitter',
                     'polar_rot', 'frozen_body_start_epochs', 'frozen_body_end_epochs',
                     'pretrained', 'seed_pretrained_stem', 'dropout_rate',
                     'hier_sev_weight', 'hier_aux_weight'}
    diff = {k: (stored.get(k), now.get(k)) for k in keys
            if k not in training_only and stored.get(k) != now.get(k)
            and not (k not in stored and _is_default(classifier, k, now.get(k)))}
    if diff:
        raise ValueError(f'{os.path.basename(ckpt_path)} was trained with a different '
                         f'architecture than this config builds -- (checkpoint, config): {diff}')
    epoch, step = int(ck['epoch']), int(ck['global_step'])
    done = epoch + 1 >= int(max_epochs)
    if finalize and not done:
        raise ValueError(f'--finalize needs the checkpoint of the LAST epoch, but this one is after '
                         f'epoch {epoch} of {max_epochs}: training is not finished. Use --resume.')
    if not finalize and done:
        raise ValueError(f'checkpoint is after epoch {epoch} (epoch {epoch + 1} of training) '
                         f'and hparams.training_epochs={max_epochs}: nothing left to train. '
                         f'Raise training_epochs to continue, or --finalize to only run the '
                         f'post-training steps.')
    return epoch, step


def _is_default(classifier, key, value):
    """True if `value` is the constructor default for `key` -- a checkpoint written before a
    knob existed lacks it, and that is only a match when the config also leaves it default."""
    import inspect
    init = getattr(type(classifier).__init__, '__func__', type(classifier).__init__)
    p = inspect.signature(init).parameters.get(key)
    return p is not None and p.default == value

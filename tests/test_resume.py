#!/usr/bin/env python3
"""Resume-after-epoch test (mvc/core/resume.py + mvc.train's --resume / --resume-epoch).

Trains a tiny convnext_pico for one epoch on synthetic tiles (CPU, seconds), then resumes it
for a second epoch from the epoch-0 checkpoint exactly the way mvc.train does, and asserts the
resume CONTINUES the run rather than restarting it:

  1. find_epoch_checkpoint picks the epoch=0 file.
  2. The resumed fit ends at epoch 2 with global_step = 2 x steps/epoch, and AdamW's per-
     parameter step count is 2 x steps/epoch too (a restart would leave it at 1x).
  3. The ModelCheckpoint state carries over: the epoch=1 checkpoint is written beside epoch=0.
  4. check_resume refuses: no epochs left, a weights-only file, and a shape-SILENT
     architecture change (monochrome keeps 4 channels, so strict loading would succeed).
  5. parse_resume_args strips the flags and keeps the positional arguments.

Run:  python -m tests.test_resume
"""
import os
import sys
import tempfile

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvc.core.lit_classifier import Classifier
from mvc.core.resume import check_resume, find_epoch_checkpoint, parse_resume_args

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
    return cond


CFG = {'model': 'convnext_pico',
       'hparams': {'tile_size': 48, 'dropout_rate': 0.0, 'DoLP': True, 'pretrained': False}}
N, BATCH = 32, 8
STEPS = N // BATCH


def build(**hp):
    cfg = {'model': CFG['model'], 'hparams': {**CFG['hparams'], **hp}}
    return Classifier.from_config(cfg, num_classes=10, clean_class=9)


def loaders():
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (N, 4, 48, 48), dtype=torch.uint8, generator=g)
    y = torch.arange(N) % 10                     # every class, clean (9) included
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=BATCH), DataLoader(ds, batch_size=BATCH)


def trainer(ckdir, max_epochs):
    cb = ModelCheckpoint(dirpath=ckdir, monitor='val_detect_auroc', mode='max', save_top_k=-1,
                         filename='{epoch}-{step}-{val_detect_auroc:.4f}')
    return pl.Trainer(max_epochs=max_epochs, accelerator='cpu', devices=1, logger=False,
                      callbacks=[cb], enable_progress_bar=False, enable_model_summary=False,
                      num_sanity_val_steps=0), cb


def adam_steps(opt_state):
    st = next(iter(opt_state['state'].values()))
    return int(st['step'])


def main():
    print('resume-after-epoch test\n')
    pl.seed_everything(0, workers=True)
    tr, va = loaders()
    with tempfile.TemporaryDirectory() as ckdir:
        # --- epoch 0
        t, _ = trainer(ckdir, 1)
        t.fit(build(), tr, va)
        ck0 = find_epoch_checkpoint(ckdir, 0)
        check(os.path.basename(ck0).startswith('epoch=0-'), f'found {ck0}')
        print(f'  [1] epoch 0 trained; found {os.path.basename(ck0)}')

        # --- refusals
        fresh = build()
        try:
            check_resume(ck0, fresh, max_epochs=1)
            failures.append('check_resume accepted a resume with no epochs left')
        except ValueError:
            print('  [4] refusal: no epochs left to train')
        wo = os.path.join(ckdir, 'weights_only.pt')
        torch.save({'state_dict': fresh.state_dict()}, wo)
        try:
            check_resume(wo, fresh, max_epochs=2)
            failures.append('check_resume accepted a weights-only file')
        except ValueError:
            print('  [4] refusal: weights-only file')
        try:
            check_resume(ck0, build(monochrome=True), max_epochs=2)
            failures.append('check_resume accepted a monochrome model for a non-mono checkpoint')
        except ValueError:
            print('  [4] refusal: shape-silent architecture change (monochrome)')

        # --- finalize: restore the finished 1-epoch run, train nothing (mvc.train --finalize)
        from mvc.core.resume import parse_finalize_args
        try:
            check_resume(ck0, fresh, max_epochs=2, finalize=True)
            failures.append('finalize accepted a checkpoint with epochs left to train')
        except ValueError:
            print('  [6] refusal: --finalize on an unfinished run')
        check(check_resume(ck0, fresh, max_epochs=1, finalize=True) == (0, STEPS),
              'finalize check on the last-epoch checkpoint failed')
        fin = build()
        tf, cbf = trainer(ckdir, 1)
        n_ckpt = len(os.listdir(ckdir))
        tf.fit(fin, tr, va, ckpt_path=ck0)
        saved = torch.load(ck0, weights_only=False, map_location='cpu')['state_dict']
        same = all(torch.equal(fin.state_dict()[k], v) for k, v in saved.items())
        check(tf.global_step == STEPS, f'finalize trained: global_step {tf.global_step} != {STEPS}')
        check(len(os.listdir(ckdir)) == n_ckpt, 'finalize wrote a new checkpoint')
        check(same, 'finalize changed the weights')
        check(os.path.basename(cbf.best_model_path) == os.path.basename(ck0),
              f'finalize lost the best-checkpoint state: {cbf.best_model_path}')
        rest, p_, e_ = parse_finalize_args(['prog', 'cfg.json', '--finalize-epoch', '0'])
        check(rest == ['prog', 'cfg.json'] and p_ is None and e_ == 0, 'parse_finalize_args')
        print(f'  [6] finalize: 0 extra steps (global_step {tf.global_step}), no new checkpoint, '
              f'weights unchanged, best checkpoint restored')

        # --- resume for epoch 1, as mvc.train does
        epoch, step = check_resume(ck0, fresh, max_epochs=2)
        check((epoch, step) == (0, STEPS), f'check_resume returned {(epoch, step)}')
        t2, cb2 = trainer(ckdir, 2)
        t2.fit(fresh, tr, va, ckpt_path=ck0)
        check(t2.current_epoch == 2, f'resumed run ended at epoch {t2.current_epoch}, expected 2')
        check(t2.global_step == 2 * STEPS,
              f'global_step {t2.global_step} != {2 * STEPS}: the resume restarted the step count')
        n_adam = adam_steps(t2.optimizers[0].state_dict())
        check(n_adam == 2 * STEPS,
              f'AdamW step {n_adam} != {2 * STEPS}: optimizer state was not restored')
        ck1 = find_epoch_checkpoint(ckdir, 1)
        print(f'  [2] resumed: epoch {t2.current_epoch}, global_step {t2.global_step}, '
              f'AdamW step {n_adam} (= 2 x {STEPS})')
        print(f'  [3] epoch 1 checkpoint written beside epoch 0: {os.path.basename(ck1)}')

    rest, path, ep = parse_resume_args(['prog', 'cfg.json', 'convnext_pico', '--resume-epoch', '0'])
    check(rest == ['prog', 'cfg.json', 'convnext_pico'] and path is None and ep == 0,
          f'parse_resume_args: {rest, path, ep}')
    rest, path, ep = parse_resume_args(['prog', 'cfg.json', '--resume=a.ckpt'])
    check(rest == ['prog', 'cfg.json'] and path == 'a.ckpt' and ep is None,
          f'parse_resume_args: {rest, path, ep}')
    print('  [5] parse_resume_args strips --resume / --resume-epoch, keeps positionals')

    print()
    if failures:
        print(f'FAILED ({len(failures)}):')
        for f in failures:
            print('  -', f)
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())

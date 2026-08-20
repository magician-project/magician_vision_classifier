#!/usr/bin/python3

""" 
Author : "Ammar Qammaz, Nikos Vasilikopoulos"
Copyright : "2025 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH" 
"""

import datetime
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from mvc.core.class_scheme import (
    filter_dataset_classes, strip_severity_classes, align_dataset_to_classes,
    _expand_family_merges, merge_dataset_classes, drop_dataset_classes,
    apply_class_scheme, resolve_auto_drops, print_class_distribution,
)
from mvc.core.config import (
    checkIfPathExists, checkIfPathIsDirectory, checkIfFileExists, load_hyperparameters,
)
from mvc.core.datasets import (
    _human_bytes, _get_available_ram_bytes, _get_path_size_bytes, _estimate_dataset_ram_bytes,
    _dataset_source_frames, frame_disjoint_split, exclude_frames_indices,
    build_train_val,
    load_rgba_image, load_rgba_image_pil,
    load_png_comment_metadata, metadata_collate_fn, RGBAImageFolder, RAMPreloadedDataset,
    CombinedDataset, BalancedBatchSampler,
)
from mvc.core.evaluation import run_confusion_and_threshold_sweep
from mvc.core.artifact_paths import out_path
from mvc.core.model_zoo import (
    build_backbone,
    CustomCNN, WaveletPool, BasicResBlock,
    TIMM_BACKBONES, REPARAM_BACKBONES,
    NONBODY_PREFIXES,
)
from mvc.core.lit_classifier import Classifier, CategoricalFocalLoss
from mvc.paths import repo_root
from torchmetrics import Accuracy, Recall, Precision, AUROC
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger



#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#                                           Main
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------

def main():
    """The training pipeline: config -> datasets -> loaders -> fit -> evaluate.

    Kept as a function (rather than bare module-level code) so the pipeline can be
    imported and driven from elsewhere -- e.g. re-scoring a saved checkpoint --
    instead of only running as a script.
    """
    # Enable TF32 for faster matmul on Ampere+ GPUs without losing meaningful precision.
    torch.set_float32_matmul_precision('high')

    configuration_file = "config.json"
    if len(sys.argv) > 1:
        configuration_file = sys.argv[1]

    print("Using ",configuration_file," configuration for training")
    config_json       = load_hyperparameters(os.path.join(repo_root(), configuration_file))
    print(config_json)

    if len(sys.argv) > 2:
        overwrite_model = sys.argv[2]
        config_json['model'] = overwrite_model
        print("Using model configuration provided by commandline parameter (",overwrite_model,")") 

    #Parse JSON Values
    #-----------------------------------------------------------------
    batch_size        = config_json['hparams']['batch_size']
    seed              = config_json['hparams']['seed']
    dropout_rate      = config_json['hparams']['dropout_rate']
    epochs            = config_json['hparams']['training_epochs']
    tile_size         = config_json['hparams']['tile_size']
    val_split         = config_json['dataloader']['validation_split']
    num_workers       = config_json['dataloader']['num_workers']
    # 0 in the config means "auto": use available cores, capped at 16 to avoid
    # oversubscription.  Note: RAMPreloadedDataset overrides this to 1 later to
    # prevent each worker from holding its own copy of the in-RAM dataset.
    if num_workers == 0:
        num_workers = min(os.cpu_count() or 1, 16)
    balanced_sampling = config_json['dataloader'].get('balanced_sampling', False)
    gradient_clip_val = config_json['hparams']['gradient_clip_value']
    class_weight      = config_json['class_weight']
    lr                = config_json['optimizer']['learning_rate']
    use_wandb         = config_json['wandb']['use_wandb']
    loss              = config_json['loss']
    penalize_false_clean = float(config_json['penalize_false_clean'])
    #-----------------------------------------------------------------
    base_channels     = 32
    if  'base_channels' in config_json['hparams']:
          base_channels = config_json['hparams']['base_channels']
    print("Base channels ", base_channels)
    #-----------------------------------------------------------------
    final_dense_layer=512
    if  'final_dense_layer' in config_json['hparams']:
          final_dense_layer = config_json['hparams']['final_dense_layer']
    print("Final Dense Layer ", final_dense_layer)
    #-----------------------------------------------------------------
    model_name = config_json['model']
    if "name" in config_json:
             model_name = "%s_%s" % (config_json['name'],config_json['model'])
    #-----------------------------------------------------------------
    noise_std = 0.0        
    noise_clip = None
    if  'noise_std' in config_json['hparams']:
          noise_std = config_json['hparams']['noise_std']
    if  'noise_clip' in config_json['hparams']:
          noise_clip = config_json['hparams']['noise_clip']
    print("Simulated Noise STD ",noise_std,"/ CLIP ",noise_clip)
    #-----------------------------------------------------------------
    gain_jitter = float(config_json['hparams'].get('gain_jitter', 0.0))
    polar_flip  = bool(config_json['hparams'].get('polar_flip', False))
    channel_jitter = float(config_json['hparams'].get('channel_jitter', 0.0))
    monochrome = bool(config_json['hparams'].get('monochrome', False))
    polar_rot = bool(config_json['hparams'].get('polar_rot', False))
    frozen_body_start_epochs = int(config_json['hparams'].get('frozen_body_start_epochs', 0))
    frozen_body_end_epochs   = int(config_json['hparams'].get('frozen_body_end_epochs', 0))
    custom_early_convs       = int(config_json['hparams'].get('custom_early_convs', 0))
    custom_channels          = config_json['hparams'].get('custom_channels', None)
    custom_res_blocks        = config_json['hparams'].get('custom_res_blocks', None)
    custom_wavelet_pools     = config_json['hparams'].get('custom_wavelet_pools', None)
    custom_wavelet_stem      = int(config_json['hparams'].get('custom_wavelet_stem', 0) or 0)
    pretrained_backbone      = bool(config_json['hparams'].get('pretrained', True))
    # The torchvision branches used to build a FRESH random stem when widening RGB ->
    # 4+ polarization channels, throwing away the pretrained first layer, while the timm
    # branch adapts it via in_chans=. That made the two families incomparable. Default
    # True seeds both the same way; set false to reproduce a pre-2026-08 torchvision run.
    seed_pretrained_stem     = bool(config_json['hparams'].get('seed_pretrained_stem', True))
    timm_stem_stride         = config_json['hparams'].get('timm_stem_stride', None)
    print("Augmentation: gain_jitter ", gain_jitter, "/ polar_flip ", polar_flip,
          "/ channel_jitter ", channel_jitter, "/ monochrome ", monochrome,
          "/ polar_rot ", polar_rot)
    print("Frozen body epochs: start ", frozen_body_start_epochs, "/ end ", frozen_body_end_epochs)
    #-----------------------------------------------------------------
    # Derived polarization input channels (computed on-GPU from the 4 raw ones)
    use_AoLP  = bool(config_json['hparams'].get('AoLP', False))
    use_DoLP  = bool(config_json['hparams'].get('DoLP', False))
    use_unpol = bool(config_json['hparams'].get('Unpolarized',
                     config_json['hparams'].get('unpolarized', False)))
    use_maxp  = bool(config_json['hparams'].get('MaxPolarization', False))
    use_minp  = bool(config_json['hparams'].get('MinPolarization', False))
    use_rngp  = bool(config_json['hparams'].get('RangePolarization', False))
    print("Derived channels: AoLP", use_AoLP, "/ DoLP", use_DoLP, "/ Unpolarized", use_unpol,
          "/ Max", use_maxp, "/ Min", use_minp, "/ Range", use_rngp)
    #-----------------------------------------------------------------

    #Let's Go..
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    directory =  config_json['training_dataset']


    # Rearrange HWC uint8 numpy array → CHW uint8 tensor.
    # We do NOT divide by 255 here; normalisation happens inside Classifier.forward()
    # on the GPU, so the DataLoader transfers 4× less data over PCIe.
    transform = transforms.Lambda(
        lambda img: torch.from_numpy(img).permute(2, 0, 1).contiguous()
    )

    # THE dataset chain: load each dir -> apply the run's class scheme -> align class
    # ORDER -> split. It lives in Datasets.build_train_val, which score_checkpoints.py,
    # eval_coverage.py and eval_ema_tta.py also call. It used to live here, with those
    # three carrying hand-copied mirrors of it coupled by comments naming line numbers in
    # this file -- line numbers that had already gone stale. A scorer that rebuilds a
    # different validation set than the run trained on reports a number for the wrong
    # split, and nothing in the output looks wrong.
    #
    # The seeding that used to sit here happens inside, at the same point in the sequence:
    # after any RAM preload, before the split. It is not only the split that depends on
    # it -- the global RNG state it leaves behind is what the model is initialised from.
    #
    # test_dataset_split.py asserts this call reproduces the block it replaced, tile for
    # tile, on every distinct dataset chain in the config corpus.
    split = build_train_val(config_json, transform=transform)
    dataset       = split.dataset
    train_dataset = split.train
    val_dataset   = split.val
    if split.ram_preloaded:
        # Each worker would otherwise hold its own copy of the in-RAM dataset.
        num_workers = 1
    H5PYFilename = '%s/dataset.h5' % (directory[0] if isinstance(directory, list)
                                      else directory)   # legacy single-file references

    print_class_distribution(dataset, title="TRAIN (final label space, pre-training)")

    # -----------------------------------------------------------

    print_class_distribution(train_dataset, title="Training Set")
    print_class_distribution(val_dataset, title="Validation Set")





    # Targets of the TRAINING split only (a Subset indexes into its parent).
    if isinstance(train_dataset, torch.utils.data.Subset):
        train_targets = [train_dataset.dataset.targets[i] for i in train_dataset.indices]
    else:
        train_targets = train_dataset.targets

    # Shared DataLoader throughput settings. The pipeline is input-bound (GPU ~63%
    # util, gzip HDF5 chunks), so these are close to free wins:
    #   pin_memory        page-locked staging buffers -> the host->device copy can
    #                     overlap compute instead of blocking on a pageable copy.
    #   persistent_workers keep the worker pool alive between epochs. With
    #                     BalancedBatchSampler an "epoch" is 38k batches so this is
    #                     minor, but under sampling_temperature (8x shorter epochs)
    #                     the respawn cost is paid 8x more often.
    #   prefetch_factor   batches queued per worker; the default is 2, and a little
    #                     more depth smooths out gzip decompression stalls.
    # All three REQUIRE num_workers > 0, hence the guard -- note the
    # cacheAllDataToRAM path forces num_workers to 1, and a plain 0 is legal too.
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs = {
            "pin_memory":         True,
            "persistent_workers": True,
            "prefetch_factor":    int(config_json['dataloader'].get('prefetch_factor', 4)),
        }
    print(f"DataLoader tuning: num_workers={num_workers} {loader_kwargs or '(single-process: no pinning/prefetch)'}")

    # Create DataLoaders
    #
    # `dataloader.sampling_temperature` (tau) is an OPT-IN alternative to
    # BalancedBatchSampler, for experimentation -- it is NOT the default and
    # changes nothing unless the key is present. Sample weight = n_c^(tau-1), so
    # the class is drawn with probability proportional to n_c^tau:
    #     tau = 0.0  fully balanced   (same class PROPORTIONS as BalancedBatchSampler)
    #     tau = 0.5  sqrt-frequency   (the usual middle ground)
    #     tau = 1.0  natural frequency (no correction at all)
    # CAUTION -- tau=0.0 is NOT a drop-in replacement for BalancedBatchSampler.
    # Both give uniform class exposure, but this sampler draws len(train) samples
    # per epoch while BalancedBatchSampler draws max(class_count)*num_classes:
    # on train_nonaltinay_v2 that is 241,784 vs 1,934,350 draws per class, an
    # 8.06x shorter epoch. Scale training_epochs accordingly before comparing.
    # Motivation, measured on train_nonaltinay_v2 (10 classes, batch 504):
    # BalancedBatchSampler gives every class 50 slots x 38,687 batches, i.e. one
    # "epoch" draws 19.5M samples = 8.06x the 2.42M-tile dataset, and the rarest
    # class (MaterialDefectClassA, 3,440 tiles) is replayed 562x per epoch --
    # ~2,250x over a 4-epoch run, which is a lot of memorization pressure on the
    # classes that generalize worst cross-site. At tau=0.5 that replay drops to
    # ~24x and an epoch is one real pass. UNVERIFIED whether it helps the KPI;
    # that is exactly what the knob is for. Leave the key out to reproduce every
    # result obtained so far.
    sampling_temperature = config_json['dataloader'].get('sampling_temperature', None)

    if sampling_temperature is not None:
        tau = float(sampling_temperature)
        counts = np.bincount(np.asarray(train_targets, dtype=np.int64),
                             minlength=len(dataset.classes))
        weights = np.power(np.maximum(counts, 1), tau - 1.0)[np.asarray(train_targets)]
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(train_targets),
            replacement=True,
        )
        print(f"Using tempered WeightedRandomSampler (sampling_temperature={tau}, "
              f"{len(train_targets):,} samples/epoch). Expected replays/epoch per class:")
        for ci in range(len(dataset.classes)):
            n = int(counts[ci])
            if n == 0:
                continue
            # draws for class ci = num_samples * (n * n^(tau-1)) / sum_j(n_j^tau)
            share = (n ** tau) / float(sum(int(c) ** tau for c in counts if c > 0))
            print(f"    {dataset.classes[ci]:28s} {n:9,} tiles  "
                  f"{len(train_targets)*share/n:7.1f}x")
        train_loader = DataLoader(
                                  train_dataset,
                                  batch_size=batch_size,
                                  sampler=sampler,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  **loader_kwargs,
                                 )
    elif balanced_sampling:
        print(f"Using BalancedBatchSampler (batch_size={batch_size}, num_classes={len(dataset.classes)})")
        balanced_sampler = BalancedBatchSampler(
            targets     = train_targets,
            num_classes = len(dataset.classes),
            batch_size  = batch_size,
        )
        train_loader = DataLoader(
                                  train_dataset,
                                  batch_sampler=balanced_sampler,
                                  num_workers=num_workers,
                                  **loader_kwargs,
                                 )
    else:
        train_loader = DataLoader(
                                  train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  **loader_kwargs,
                                 )

    val_loader  = DataLoader(
                            val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            drop_last=False,  # never discard validation samples
                            **loader_kwargs,
                           )

    #Print class names as a sanity check
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    #Get clean class which could be needed for extra penalization. None when the
    #dataset has no clean class (e.g. an 'alldefect' run that dropped class_clean)
    #-- val_detect_auroc / penalize_false_clean / the threshold sweep all skip.
    cleanClassID = None
    for i in range(len(dataset.classes)):
           if (dataset.classes[i] == "class_clean") or (dataset.classes[i] == "Clean"):
              cleanClassID = i
    print(f"Clean class ID is : {cleanClassID}")

    # ---------------------------------------------------------------------
    # DEAD FEATURE: "class_weight" / focal-loss alpha.  KEPT FOR THE RECORD.
    # ---------------------------------------------------------------------
    # This block used to build an inverse-frequency `alpha` vector, normalize it,
    # and move it to the device -- and then NOTHING consumed it. Classifier.__init__
    # hardcodes CategoricalFocalLoss(gamma=2.0, alpha=None), so every run with
    # "class_weight": true has silently been an unweighted run. All configs in
    # configs/ set it to false, so no published result is affected.
    #
    # It is deliberately NOT being revived, because BalancedBatchSampler already
    # performs exactly this correction -- by frequency instead of by loss weight.
    # Wiring alpha back in would double-correct. Measured on train_nonaltinay_v2
    # (10 classes, batch 504), relative to uniform:
    #
    #   class                     tiles    sampler replay  x alpha  = combined
    #   MaterialDefectClassA      3,440         562x         5.2x      2927x
    #   DeformationClassA         7,708         251x         2.3x       583x
    #   WeldingClassA           128,986          15x        0.14x         2x
    #   clean                 1,934,376           1x       0.009x     0.009x
    #
    # i.e. a 3,440-tile class would end up ~3000x over-weighted and the model
    # would simply predict it. Two further defects if anyone reconsiders:
    #   * `alpha /= alpha.sum()` normalizes to a simplex, shrinking the total loss
    #     ~1/num_classes and acting as an uncontrolled learning-rate change.
    #     Focal-loss alpha is meant to be O(1) per class, not a distribution.
    #   * `1 / class_counts[i]` raises ZeroDivisionError for any class with zero
    #     training samples (Counter returns 0 for a missing key).
    #
    # To re-balance, prefer `dataloader.sampling_temperature` above, which is the
    # same idea with a single continuous knob and no double-correction risk.
    # CategoricalFocalLoss still ACCEPTS an alpha tensor -- only the trainer
    # declines to supply one.
    alpha = None
    if class_weight:
        print("[WARNING] config sets \"class_weight\": true, but class weighting is a DEAD "
              "option and is ignored -- see the comment above this warning. "
              "BalancedBatchSampler (or dataloader.sampling_temperature) already handles "
              "class imbalance; applying both would double-correct.")


    # Initialize the classifier
    # Single source of truth: LitClassifier.Classifier.from_config, which was derived
    # from this very block and is asserted equal to it on all 167 real configs by
    # test_classifier_from_config.py. Keeping the translation in one place is what
    # stops the eval tools drifting away from the trainer again.
    classifier = Classifier.from_config(config_json,
                                        num_classes=len(class_names),
                                        clean_class=cleanClassID)
    print(f"Learning rate: {lr}")
    
    model_type = classifier.type

    # Initialize a PyTorch Lightning trainer
    if use_wandb:
        loggers = [TensorBoardLogger("lightning_logs", name="classifier"),  WandbLogger(project=config_json['wandb']['project'], log_model=True,name=config_json['wandb']['name']+model_type+loss)]
    else:
        if checkIfPathExists("tensorboard/"):
            print("Removing previous tensorboard logs..")
            shutil.rmtree("tensorboard/", ignore_errors=True)
        loggers = [TensorBoardLogger("tensorboard", name=model_name)]

    # Keep the epoch with the best validation loss; cross-site val metrics drift
    # non-monotonically, so last-epoch weights are often not the best weights.
    #
    # save_top_k=1 (the default) DELETES every other epoch, which makes the
    # "do more epochs help?" question unanswerable: val_loss is a SOURCE-domain
    # quantity and bottoms out around epoch 0-1, so it selects an early epoch and
    # discards the later ones -- even if cross-site detection (our actual KPI)
    # keeps improving after val_loss stops. Set "checkpoint_save_top_k": -1 in the
    # config to keep every epoch, then rank them by AUROC of 1-P(clean) on the
    # held-out site instead of by val_loss. "checkpoint_dir" redirects the files
    # (they are ~30MB each and /home is nearly full).
    from pytorch_lightning.callbacks import ModelCheckpoint
    ckpt_top_k  = int(config_json.get("checkpoint_save_top_k", 1))
    ckpt_dir    = config_json.get("checkpoint_dir", None)
    # Default to selecting on DETECTION AUROC, not val_loss. Measured over the
    # 24-epoch customwide run, pearson(val_loss, detect AUROC) = -0.09 and
    # pearson(val_loss, macro) = -0.00 => val_loss ranks epochs at random w.r.t.
    # the KPI, because focal+pfc scores 6-way class identity while the KPI asks
    # "defect or clean". Set "checkpoint_monitor": "val_loss" to get the old
    # behaviour back.
    ckpt_monitor = config_json.get("checkpoint_monitor", "val_detect_auroc")
    ckpt_mode    = config_json.get("checkpoint_mode",
                                   "min" if ckpt_monitor.endswith("loss") else "max")
    # Filename references only metrics that are actually logged: val_loss always,
    # plus the monitored metric (val_detect_auroc is absent on alldefect runs).
    ckpt_fname = '{epoch}-{step}-{val_loss:.4f}'
    if ckpt_monitor != 'val_loss':
        ckpt_fname += '-{' + ckpt_monitor + ':.4f}'
    best_ckpt = ModelCheckpoint(monitor=ckpt_monitor, mode=ckpt_mode,
                                save_top_k=ckpt_top_k,
                                dirpath=ckpt_dir,
                                filename=ckpt_fname)
    print(f"Checkpoints: monitor={ckpt_monitor} mode={ckpt_mode} save_top_k={ckpt_top_k} "
          f"({'ALL epochs kept' if ckpt_top_k == -1 else 'best epoch(s) only'}) "
          f"dir={ckpt_dir or '<logger default>'}")

    # Deterministic cuDNN kernels are only worth their speed penalty if the run is
    # ACTUALLY reproducible. The old test looked at noise_std alone, but every
    # augmentation below draws from the RNG too -- with gain_jitter/polar_flip/
    # channel_jitter/polar_rot on (the standard v2rot recipe) the run was never
    # reproducible, so the deterministic kernels bought nothing and only cost
    # throughput. Now every stochastic knob is considered, and the config can force
    # the decision either way with "deterministic": true/false.
    _stochastic = {
        'noise_std':      noise_std > 0.0,
        'gain_jitter':    gain_jitter > 0.0,
        'channel_jitter': channel_jitter > 0.0,
        'polar_flip':     bool(polar_flip),
        'polar_rot':      bool(polar_rot),
    }
    _active = [k for k, v in _stochastic.items() if v]
    if 'deterministic' in config_json:
        deterministic_mode = bool(config_json['deterministic'])
        print(f"Deterministic mode forced by config: {deterministic_mode}")
    else:
        deterministic_mode = not _active
        print(f"Deterministic mode: {deterministic_mode}"
              + (f" (stochastic augmentation active: {', '.join(_active)} -> "
                 f"the run is not reproducible anyway, so deterministic kernels "
                 f"would only cost throughput)" if _active else ""))

    trainer = pl.Trainer(
                         max_epochs=epochs,
                         # Optional fixed cap on training batches/epoch (int) for quick
                         # architecture screens; default 1.0 = full epoch (unchanged).
                         # Validation still runs on the full val set.
                         limit_train_batches=config_json.get('limit_train_batches', 1.0),
                         logger=loggers,
                         callbacks=[best_ckpt],
                         #callbacks=[EarlyStopping(monitor='val_loss')],
                         accelerator=config_json['accelerator'],
                         devices=config_json['devices'],
                         gradient_clip_val=gradient_clip_val,
                         deterministic= deterministic_mode,
                       )

    #Train and log to console
    #------------------------------------------------------------------
    trainer.fit(classifier, train_loader, val_loader)

    # Restore the best-val_loss epoch before saving/validating/confusion matrix
    if best_ckpt.best_model_path:
        print("Restoring best checkpoint:", best_ckpt.best_model_path,
              "(%s = %s)" % (ckpt_monitor, str(best_ckpt.best_model_score)))
        best_state = torch.load(best_ckpt.best_model_path, weights_only=False)
        classifier.load_state_dict(best_state['state_dict'])


    #Save the model
    #------------------------------------------------------------------
    print("Saving the model")
    # Emitted into experiments/<campaign>/<run>/ rather than the cwd, and the model
    # name is sanitised, so a `timm/x` run no longer creates a stray directory.
    trainer.save_checkpoint(out_path(model_name, '.pth'))

    # Evaluate dumped tiles (if directory exists)
    #------------------------------------------------------------------
    #tiles_dir = "tiles_dump_1760036129"  # replace or detect automatically
    #if os.path.isdir(tiles_dir):
    #    metrics = evaluate_dumped_tiles(classifier.model, tiles_dir, class_names, device=device)
    #    if metrics:
    #        with open("tile_evaluation.json", "w") as f:
    #            json.dump(metrics, f, indent=2)


    #Predictions + confusion matrix + detection threshold sweep
    #------------------------------------------------------------------
    # (see Evaluation.py -- runs on the accelerator, not the CPU)
    run_confusion_and_threshold_sweep(classifier, trainer, val_loader, dataset,
                                      config_json, model_name, cleanClassID,
                                      tile_size=tile_size, epochs=epochs)

    pth_path = out_path(model_name, '.pth')
    md5 = hashlib.md5()
    with open(pth_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5.update(chunk)
    config_json['model_md5'] = md5.hexdigest()

    #Save the JSON
    #------------------------------------------------------------------
    print("Saving the JSON")
    with open(out_path(model_name, '.json'), 'w') as f:
       json.dump(config_json, f, indent=2)


    #Update symbolic links 
    #------------------------------------------------------------------
    #os.system("rm last.pth last.json && ln -s %s.pth last.pth && ln -s %s.json last.json" % (model_name,model_name) )
    #No longer needed

    # Package the run -- mvc.export is the single source of truth
    #------------------------------------------------------------------
    # This used to be an inline `subprocess.run(["zip", "-r", ...], check=False)`. It had
    # no name sanitisation, so a model configured as `timm/<x>` produced an archive path
    # containing a slash and zip failed; and because the exit code was discarded, the
    # trainer reported success anyway. Thirteen fully trained models ended up with no
    # archive and nothing said so. export_run() sanitises the names, includes the .json
    # sidecars, and VERIFIES the finished archive (weights load, config parses, CRC and
    # member list check) -- raising instead of failing quietly.
    from mvc.export import export_run
    try:
        zip_path = export_run(config_json)
        print(f"Saved everything as {zip_path}")
        print('To upload ALL models copy/paste:')
        print("  scripts/uploadModels.sh")
        print('To upload just this one copy/paste:')
        print(f"  scp -P 2222 {zip_path} ammar@ammar.gr:"
              "/home/ammar/public_html/magician/models/CameraV2Models/")
    except RuntimeError as _exc:
        # Loud, but not fatal: training and scoring already succeeded and the checkpoints
        # are on disk, so the run is recoverable with `python3 -m mvc.export --apply`.
        print(f"!! MODEL EXPORT FAILED -- the run itself is fine, re-export with "
              f"`python3 -m mvc.export --apply`\n{_exc}")


if __name__ == "__main__":
    main()
  
    

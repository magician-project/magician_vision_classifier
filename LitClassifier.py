"""The LightningModule: loss, model wiring, train/val steps, freeze schedule.

Split out of the trainer so the training script orchestrates rather than defines.
The architecture lives in ModelZoo (build_backbone), the input/polarization maths in
Polarization -- this module is the Lightning glue between them.

Note for deployment: classifierPnm.py / liveClassifierTorch.py import Classifier, so
its constructor signature is a public interface, not an internal detail.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchmetrics import Accuracy, Recall, Precision, AUROC

import Polarization as polarization
from ModelZoo import build_backbone, NONBODY_PREFIXES

class CategoricalFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        """
        Categorical Focal Loss for multi-class classification.

        :param gamma: Focusing parameter (higher values focus more on hard examples)
        :param alpha: Class weighting (TENSOR of shape [num_classes], or None).
                      NOTE: the trainer never supplies this -- it always builds
                      CategoricalFocalLoss(gamma=2.0, alpha=None) and handles class
                      imbalance through the sampler instead. See the "DEAD FEATURE:
                      class_weight" comment in main() for why, and note that a
                      scalar (like the 0.25 default below) would crash the
                      `self.alpha[targets]` lookup -- only a per-class tensor works.
        :param reduction: Reduction mode: 'none' | 'mean' | 'sum'
        """
        super(CategoricalFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Computes the focal loss.
        
        :param logits: Tensor of shape [batch_size, num_classes] (raw model outputs before softmax)
        :param targets: Tensor of shape [batch_size] with class indices (integer labels)
        :return: Scalar loss value
        """
        # Convert class indices to one-hot encoding
        num_classes = logits.shape[1]
        #target_one_hot = F.one_hot(targets, num_classes).float() #<-Original 
        target_one_hot = F.one_hot(targets, num_classes).to(torch.float32)

        # Compute softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Gather the probabilities corresponding to the true class
        pt = (probs * target_one_hot).sum(dim=1)  # Shape: [batch_size]

        # Compute focal loss term
        focal_weight = (1 - pt) ** self.gamma

        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Apply class weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]  # Select alpha value for each target
            ce_loss = alpha_t * ce_loss

        # Compute final loss
        loss = focal_weight * ce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class Classifier(pl.LightningModule):
    """
    A PyTorch Lightning module wrapping a variety of pretrained vision backbones
    (ResNet, ConvNeXt, EfficientNet, Swin, RegNet, MobileNet, ShuffleNet, SqueezeNet,
    DenseNet, MNASNet) or a custom CNN for tile-level image classification.

    Supports polarization-derived input channels (AoLP, DoLP, Unpolarized, Max/Min/Range
    Polarization) computed on-the-fly from the base 4 Stokes channels. Also supports
    input noise augmentation and false-clean penalization for imbalanced datasets.
    """
    def __init__(self,
                      model='resnet18',
                      loss='focal',
                      tile_size=64,
                      num_classes=4,
                      dropout_rate=0.1,
                      lr=1e-4,
                      AoLP=False,
                      DoLP=False,
                      Unpolarized=False,
                      MaxPolarization=False,
                      MinPolarization=False,
                      RangePolarization=False,
                      load_checkpoint=None,
                      penalize_false_clean=0.0,
                      base_channels=32,
                      final_dense_layer=512,
                      clean_class=0,
                      noise_std=0.0,
                      noise_clip=None,
                      gain_jitter=0.0,
                      polar_flip=False,
                      channel_jitter=0.0,
                      monochrome=False,
                      polar_rot=False,
                      frozen_body_start_epochs=0,
                      frozen_body_end_epochs=0,
                      custom_early_convs=0,
                      custom_channels=None,
                      custom_res_blocks=None,
                      custom_wavelet_pools=None,
                      custom_wavelet_stem=0,
                      pretrained=True,
                      seed_pretrained_stem=True,
                      timm_stem_stride=None
                 ):
        super(Classifier, self).__init__()
        # Persist every constructor argument into the checkpoint, so a model can be
        # rebuilt from the checkpoint alone. Before this, eleven call sites each
        # hand-translated a config dict into these ~31 kwargs, and every one of them was
        # free to disagree -- which they did. See from_config() for the whole story.
        # NOTE: checkpoints written before this line have no stored hparams;
        # load_for_eval() below refuses to guess rather than falling back to defaults.
        self.save_hyperparameters()
        #-----------------------------------------
        self.type              = model
        self.lr                = lr
        self.tile_size         = tile_size
        self.num_classes       = num_classes
        self.dropout_rate      = dropout_rate
        self.base_channels     = base_channels
        self.final_dense_layer = final_dense_layer
        #-----------------------------------------
        self.AoLP = AoLP
        self.DoLP = DoLP
        self.Unpolarized = Unpolarized
        self.MaxPolarization   = MaxPolarization
        self.MinPolarization   = MinPolarization
        self.RangePolarization = RangePolarization
        #-----------------------------------------
        self.clean_class  = clean_class
        self.penalize_false_clean = penalize_false_clean
        #-----------------------------------------
        self.noise_std  = noise_std
        self.noise_clip = noise_clip
        self.gain_jitter = gain_jitter   # exposure-emulating multiplicative jitter
        self.polar_flip  = polar_flip    # physics-consistent mirror augmentation
        self.channel_jitter = channel_jitter  # strobe-light-emulating per-channel jitter
        self.polar_rot = polar_rot       # physics-consistent +/-90deg rotations (0<->90, 45<->135 swap)
        # Freeze the pretrained backbone body for the first / last N epochs so a
        # freshly-initialized stem or head can align before the body co-adapts
        # (and for a clean head-only fine-tune at the end). 0 = never freeze.
        self.custom_early_convs = int(custom_early_convs)
        self.custom_channels = custom_channels
        self.custom_res_blocks = custom_res_blocks
        self.custom_wavelet_pools = custom_wavelet_pools
        self.custom_wavelet_stem  = custom_wavelet_stem
        self.pretrained = bool(pretrained)
        # Carry the pretrained RGB stem kernel into the wider polarization stem
        # instead of discarding it (what timm's in_chans= does for free). False
        # reproduces the pre-2026-08 random-stem behaviour of the torchvision branches.
        self.seed_pretrained_stem = bool(seed_pretrained_stem)
        # Re-stride a timm patchify stem so the body sees a larger map (see
        # ModelZoo.retune_timm_stem_stride). None = leave the architecture alone.
        self.timm_stem_stride = timm_stem_stride
        self.frozen_body_start_epochs = int(frozen_body_start_epochs)
        self.frozen_body_end_epochs   = int(frozen_body_end_epochs)
        self._body_frozen_state = None   # cache to avoid redundant requires_grad churn
        # Polarization ablation: replace the 4 channels with their mean (what a
        # regular monochrome camera would record), replicated x4 so the
        # architecture/capacity stay byte-identical. Train AND val see it.
        self.monochrome = monochrome

        # Dynamic input channels (base 4 polarization channels + optional derived channels)
        extra_channels = 0
        if self.DoLP: extra_channels += 1
        if self.AoLP: extra_channels += 1
        if self.Unpolarized: extra_channels += 1
        if self.MaxPolarization: extra_channels += 1
        if self.MinPolarization: extra_channels += 1
        if self.RangePolarization: extra_channels += 1

        self.base_input_channels = 4
        self.in_channels = self.base_input_channels + extra_channels

        # `pretrained=False` must reach EVERY backbone, not just resnet18. At
        # inference time ClassifierPnm rebuilds the architecture and then overwrites
        # it with the checkpoint, so fetching ImageNet weights first is pure waste --
        # it costs a download (and fails outright on an offline deployment machine)
        # for weights that are discarded microseconds later.
        self.model = build_backbone(
                                   self.type,
                                   in_channels=self.in_channels,
                                   num_classes=num_classes,
                                   pretrained=self.pretrained,
                                   seed_pretrained_stem=self.seed_pretrained_stem,
                                   timm_stem_stride=self.timm_stem_stride,
                                   tile_size=tile_size,
                                   dropout_rate=dropout_rate,
                                   base_channels=self.base_channels,
                                   final_dense_layer=self.final_dense_layer,
                                   custom_early_convs=getattr(self, 'custom_early_convs', 0),
                                   custom_channels=getattr(self, 'custom_channels', None),
                                   custom_res_blocks=getattr(self, 'custom_res_blocks', None),
                                   custom_wavelet_pools=getattr(self, 'custom_wavelet_pools', None),
                                   custom_wavelet_stem=getattr(self, 'custom_wavelet_stem', 0),
                                  )

        if load_checkpoint is not None:
            # load_from_checkpoint returns a full Classifier (LightningModule), not a bare
            # backbone. Copy only the backbone weights so self.model keeps the correct type.
            loaded = Classifier.load_from_checkpoint(load_checkpoint)
            self.model.load_state_dict(loaded.model.state_dict())

        if loss == 'focal':
            self.criterion = CategoricalFocalLoss(gamma=2.0, alpha=None)
        elif loss == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss}. Supported types are 'focal' and 'cross_entropy'.")

        self.accuracy  = Accuracy(task='MULTICLASS',  num_classes=num_classes)
        self.recall    = Recall(task='MULTICLASS',    num_classes=num_classes)
        self.precision = Precision(task='MULTICLASS', num_classes=num_classes)
        self.auroc     = AUROC(task='MULTICLASS',     num_classes=num_classes)
        # Binary defect-vs-clean detector AUROC — the metric the KPI cares about,
        # and the one ModelCheckpoint should select on (see validation_step).
        self.detect_auroc = AUROC(task='BINARY')

    def add_input_noise(self, x):
        """Training-only additive input noise."""
        return polarization.add_input_noise(
            x,
            training=self.training,
            noise_std=self.noise_std,
            noise_clip=self.noise_clip,
        )

    def augment_train_batch(self, x):
        """Training-only, physics-consistent augmentation of the raw 4-channel batch."""
        return polarization.augment_train_batch(
            x,
            training=self.training,
            gain_jitter=self.gain_jitter,
            polar_flip=self.polar_flip,
            polar_rot=self.polar_rot,
            channel_jitter=self.channel_jitter,
        )

    def build_input_features(self, x):
        """uint8 tile -> normalized float tensor plus the configured derived channels."""
        return polarization.build_input_features(
            x,
            in_channels=self.in_channels,
            monochrome=self.monochrome,
            AoLP=self.AoLP,
            DoLP=self.DoLP,
            Unpolarized=self.Unpolarized,
            MaxPolarization=self.MaxPolarization,
            MinPolarization=self.MinPolarization,
            RangePolarization=self.RangePolarization,
        )

    def forward(self, x):
        """
        Canonical forward pass: build input features (polarization channels + noise),
        then pass through the backbone model.

        Args:
            x: Input tensor, typically uint8 [B, 4, H, W] or float [B, 4, H, W].

        Returns:
            Logits tensor of shape (B, num_classes).
        """
        x = self.build_input_features(x)
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        """
        Compute training loss with optional false-clean penalization.

        If penalize_false_clean > 0, adds a penalty term that discourages the model
        from assigning high clean-class probability to non-clean samples. The penalty
        is -log(1 - P(clean)) for each non-clean sample.

        Args:
            batch: (x, y) tensor pair from the DataLoader.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch

        x = self.augment_train_batch(x)
        x = self.add_input_noise(x)
        x = self.build_input_features(x)

        y_hat = self.model(x)
        base_loss  = self.criterion(y_hat, y)

        if self.penalize_false_clean > 0.0:
            pred_probs = F.softmax(y_hat, dim=1)
            non_clean_mask = (y != self.clean_class)
            penalty_strength = float(self.penalize_false_clean)

            if non_clean_mask.any():
                p_clean = pred_probs[non_clean_mask, self.clean_class]
                false_clean_loss = -torch.log(1.0 - p_clean + 1e-8).mean()
                loss = base_loss + penalty_strength * false_clean_loss
            else:
                loss = base_loss
        else:
            loss = base_loss

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _batch_idx):
        """
        Execute one validation batch: compute loss and update metric accumulators.
        Noise is intentionally skipped (training-only). Metric .compute() is deferred
        to on_validation_epoch_end to avoid per-batch warnings on small/imbalanced batches.

        Args:
            batch: (x, y) tensor pair from the validation DataLoader.
            _batch_idx: Unused batch index required by Lightning.

        Returns:
            Scalar validation loss tensor.
        """
        x, y = batch

        # Use self(x) — goes through the canonical forward() path.
        # Note: add_input_noise is intentionally skipped here (noise is training-only).
        y_hat = self(x)
        loss  = self.criterion(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)

        # Only accumulate state per batch — do NOT call .compute() here.
        # Computing per-batch fires "no positive samples" warnings whenever a
        # small batch happens to contain only one class (common with imbalanced
        # datasets).  Epoch-level values are logged in on_validation_epoch_end.
        self.accuracy.update(y_hat, y)
        self.recall.update(y_hat, y)
        self.precision.update(y_hat, y)
        self.auroc.update(y_hat, y)

        # Detection AUROC: is this tile ANY defect, vs clean -- the question the
        # KPI (skipped defects) actually asks. val_loss does NOT answer it:
        # focal+penalize_false_clean scores 6-way class identity and calibration,
        # and measured over epochs it is UNCORRELATED with detection
        # (pearson(val_loss, detect AUROC) = -0.09 over the 24-epoch customwide
        # run), so monitoring val_loss selects a checkpoint at random w.r.t. what
        # we care about. Same for val_auroc above: that is macro one-vs-rest over
        # all classes, not defect-vs-clean. Score by 1 - P(clean), matching
        # liveClassifierTorch.gate_tiles' defect_mass gate.
        if self.clean_class is not None:
            probs = F.softmax(y_hat, dim=1)
            self.detect_auroc.update(1.0 - probs[:, self.clean_class],
                                     (y != self.clean_class).long())

        return loss

    # Channel order of the 4 polarization planes as they reach the model. Traced
    # end to end and identical on every path that feeds the network:
    #   readData.readPolarPNMToRGBALive emits [I0, I45, I90, I135], then a
    #   0<->2 swap is applied exactly once --
    #     training H5 : annotator datasetCreator.py `tile[:, :, [2, 1, 0, 3]]`
    #     training PNG: load_rgba_image's cv2.COLOR_RGBA2BGRA
    #     live/ROS    : classifierPnm.runSingle's cv2.COLOR_RGBA2BGRA
    #     ensemble    : EnsembleClassifier.py cv2.COLOR_RGBA2BGRA
    #   -> [I90, I45, I0, I135].
    # Only the Stokes maths below depends on this ordering; the other derived
    # channels (Unpolarized / Max / Min / Range / monochrome) are symmetric over
    # the 4 planes. The polar_flip / polar_rot permutations in
    # augment_train_batch also assume this order.
    CH_I90, CH_I45, CH_I0, CH_I135 = 0, 1, 2, 3

    def calculate_stokes(self, x):
        """Stokes parameters from the 4 linear channels. See Polarization.py."""
        return polarization.calculate_stokes(x)

    def calculate_DoLP(self, x):
        """Degree of linear polarization from Stokes. See Polarization.py."""
        return polarization.calculate_DoLP(x)

    def calculate_AoLP(self, x):
        """Angle of linear polarization from Stokes. See Polarization.py."""
        return polarization.calculate_AoLP(x)

    def on_validation_epoch_end(self):
        # Compute epoch-level metrics from the accumulated state of all validation
        # batches, then reset for the next epoch.  Computing here (not per-batch)
        # avoids "no positive samples" warnings that fire when individual batches
        # happen to contain only one class.
        self.log('val_accuracy',  self.accuracy.compute(),  prog_bar=True, sync_dist=True)
        self.log('val_recall',    self.recall.compute(),    prog_bar=True, sync_dist=True)
        self.log('val_precision', self.precision.compute(), prog_bar=True, sync_dist=True)
        self.log('val_auroc',     self.auroc.compute(),     prog_bar=True, sync_dist=True)
        if self.clean_class is not None:
            # The checkpoint selector should monitor THIS (mode='max'), not val_loss.
            self.log('val_detect_auroc', self.detect_auroc.compute(), prog_bar=True, sync_dist=True)
            self.detect_auroc.reset()
        self.accuracy.reset()
        self.recall.reset()
        self.precision.reset()
        self.auroc.reset()

    # Parameter-name PREFIXES identifying the NON-body (input adapter / stem /
    # classifier head) parameters — the ones we replaced above and that should keep
    # training while the pretrained body is frozen. Everything else is "body".
    #
    # These must be matched as PREFIXES, not substrings. The original code used
    # `key in name`, which meant 'conv1' also matched 'layer1.0.conv1.weight' and
    # 'bn1' also matched 'layer2.1.bn1.bias' — so on resnet18 only 33 tensors were
    # frozen while 29 stayed trainable, including most of the residual body. The
    # freeze silently did roughly half of what its docstring claims.
    #
    # One entry per backbone's replaced stem/head, matching the constructor above:
    #   conv1.         resnet18* (incl. the Sequential stems), resnext50, shufflenet
    #   bn1. / fc.     resnet18*, resnext50, regnet, shufflenet
    #   stem.          regnet
    #   features.0.    convnext, efficientnet, swin, mobilenet, squeezenet
    #   features.conv0. densenet121
    #   layers.0.      mnasnet
    #   classifier. / head.  every classifier head
    # The trailing dot is required: it is what stops 'fc' from also matching a
    # hypothetical 'fc_extra' and, more importantly, anchors the match at the top
    # level of the module tree.
    _NONBODY_PREFIXES = NONBODY_PREFIXES   # see ModelZoo

    def _is_body_param(self, name):
        return not name.startswith(self._NONBODY_PREFIXES)

    def _set_body_frozen(self, frozen):
        if self._body_frozen_state is frozen:
            return
        n = 0
        for name, p in self.model.named_parameters():
            if self._is_body_param(name):
                p.requires_grad = not frozen
                n += 1
        # also stop BN running-stat drift in the frozen body
        for name, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)) and self._is_body_param(name + '.weight'):
                m.eval() if frozen else m.train()
        self._body_frozen_state = frozen
        print(f"[FreezeBody] body {'FROZEN' if frozen else 'UNFROZEN'} ({n} body param tensors)")

    def on_train_epoch_start(self):
        s, e = self.frozen_body_start_epochs, self.frozen_body_end_epochs
        if s == 0 and e == 0:
            return
        ep = self.current_epoch
        total = getattr(self.trainer, 'max_epochs', None) or (ep + 1)
        freeze = (ep < s) or (ep >= total - e)
        self._set_body_frozen(freeze)

    def configure_optimizers(self):
        """
        Configure the AdamW optimizer for all parameters in the Classifier module.
        Using self.parameters() (not self.model.parameters()) ensures that any new
        learnable parameters added directly to Classifier (not the backbone) are included.
        """
        # Optimize self.parameters() (all module params) rather than just self.model.parameters(),
        # so any future additions to Classifier itself are covered automatically.
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # ------------------------------------------------------------------ from_config
    # Constructor kwargs read from config['hparams']. Everything the trainer reads from
    # there is listed, so this is the authoritative nesting: the derived-channel flags,
    # the augmentation knobs and the architecture knobs all live under `hparams`, NOT at
    # the config top level. Four eval tools used to read AoLP/DoLP from the top level,
    # where no config has ever had them -- 167 configs carry them under hparams and zero
    # at the top -- so those tools silently built a 4-channel model for every 5-channel
    # run. That class of bug is what this method exists to make impossible.
    _HPARAM_KWARGS = (
        'tile_size', 'dropout_rate',
        'AoLP', 'DoLP', 'Unpolarized',
        'MaxPolarization', 'MinPolarization', 'RangePolarization',
        'base_channels', 'final_dense_layer',
        'noise_std', 'noise_clip',
        'gain_jitter', 'polar_flip', 'channel_jitter', 'monochrome', 'polar_rot',
        'frozen_body_start_epochs', 'frozen_body_end_epochs',
        'custom_early_convs', 'custom_channels', 'custom_res_blocks',
        'custom_wavelet_pools', 'custom_wavelet_stem',
        'pretrained', 'seed_pretrained_stem', 'timm_stem_stride',
    )
    # Historical spellings that must keep working; first match wins.
    _HPARAM_ALIASES = {'Unpolarized': ('Unpolarized', 'unpolarized')}

    @classmethod
    def config_to_kwargs(cls, cfg, *, num_classes, clean_class=0, lr=None, **overrides):
        """The pure config -> constructor-kwargs translation. THE single definition.

        Separate from from_config() so it can be tested without building a model, and so
        the defaults are read from the real __init__ signature rather than from whatever
        happens to be bound to `cls.__init__` at call time.
        """
        return cls._config_to_kwargs(cfg, num_classes=num_classes,
                                     clean_class=clean_class, lr=lr, **overrides)

    @classmethod
    def _config_to_kwargs(cls, cfg, *, num_classes, clean_class=0, lr=None, **overrides):
        """Build the constructor kwargs from a training config dict.

        `num_classes` is required and `clean_class` is caller-supplied because both come
        from the resolved label space, not from the config -- the config's `classes` list
        is written back by the trainer AFTER a run and is absent on a fresh config.

        Defaults come from this class's own __init__ signature rather than being restated
        here, so there is exactly one definition of every default. Adding a constructor
        argument and listing it in _HPARAM_KWARGS is all it takes to make every call site
        honour it.
        """
        import inspect
        if not isinstance(cfg, dict):
            raise TypeError(f'from_config expects a config dict, got {type(cfg).__name__}')
        hp = cfg.get('hparams')
        if not isinstance(hp, dict):
            raise ValueError(
                'config has no "hparams" block -- refusing to build a model from defaults. '
                'Every architecture-critical knob (DoLP, monochrome, timm_stem_stride, the '
                'CustomCNN ladder) lives there, and guessing them produces a model that '
                'loads cleanly and computes the wrong thing.')

        # __func__ so a decorated/patched cls.__init__ cannot change the defaults.
        init = getattr(cls.__init__, '__func__', cls.__init__)
        sig = inspect.signature(init).parameters
        kwargs = {}
        for key in cls._HPARAM_KWARGS:
            names = cls._HPARAM_ALIASES.get(key, (key,))
            for n in names:
                if n in hp:
                    kwargs[key] = hp[n]
                    break
            else:
                kwargs[key] = sig[key].default

        # Top-level config keys, matching the trainer.
        kwargs['model'] = cfg['model']
        kwargs['penalize_false_clean'] = float(cfg.get('penalize_false_clean', 0.0))
        kwargs['num_classes'] = num_classes
        kwargs['clean_class'] = clean_class
        kwargs['lr'] = (lr if lr is not None
                        else cfg.get('optimizer', {}).get('learning_rate',
                                                          sig['lr'].default))

        # Match the trainer's coercions exactly, so a converted call site is a no-op.
        for k in ('AoLP', 'DoLP', 'Unpolarized', 'MaxPolarization', 'MinPolarization',
                  'RangePolarization', 'polar_flip', 'monochrome', 'polar_rot',
                  'pretrained', 'seed_pretrained_stem'):
            kwargs[k] = bool(kwargs[k])
        for k in ('gain_jitter', 'channel_jitter'):
            kwargs[k] = float(kwargs[k])
        for k in ('frozen_body_start_epochs', 'frozen_body_end_epochs',
                  'custom_early_convs'):
            kwargs[k] = int(kwargs[k])
        kwargs['custom_wavelet_stem'] = int(kwargs['custom_wavelet_stem'] or 0)

        kwargs.update(overrides)
        return kwargs

    @classmethod
    def from_config(cls, cfg, *, num_classes, clean_class=0, lr=None, **overrides):
        """Build a Classifier from a training config dict. THE single translation.

        `num_classes` is required and `clean_class` is caller-supplied because both
        come from the resolved label space, not the config -- the config's `classes`
        list is written back by the trainer AFTER a run and is absent beforehand.
        """
        return cls(**cls._config_to_kwargs(cfg, num_classes=num_classes,
                                           clean_class=clean_class, lr=lr,
                                           **overrides))

    @classmethod
    def load_for_eval(cls, checkpoint_path, cfg=None, *, num_classes=None,
                      clean_class=0, map_location='cpu', **overrides):
        """Rebuild a trained model for evaluation, refusing to guess its architecture.

        Prefers hyperparameters stored in the checkpoint (written by
        save_hyperparameters()). Checkpoints predating that call have none, so it falls
        back to `cfg` -- and if neither source has them it RAISES.

        That refusal is the point. Lightning's load_from_checkpoint silently falls back to
        __init__ defaults when a checkpoint carries no hparams, which rebuilds a
        4-channel, stride-4, polarization-on model regardless of what was trained. The
        state dict often loads anyway, and the wrongness only shows up as a slightly
        disappointing number.
        """
        import torch
        ckpt = torch.load(checkpoint_path, map_location=map_location)
        stored = ckpt.get('hyper_parameters') if isinstance(ckpt, dict) else None
        if stored:
            kwargs = dict(stored)
            kwargs.update(overrides)
            if num_classes is not None:
                kwargs['num_classes'] = num_classes
            model = cls(**kwargs)
        elif cfg is not None:
            if num_classes is None:
                raise ValueError('num_classes is required when rebuilding from a config')
            model = cls.from_config(cfg, num_classes=num_classes,
                                    clean_class=clean_class, **overrides)
        else:
            raise ValueError(
                f'{checkpoint_path} stores no hyper_parameters and no config was given. '
                'Refusing to rebuild from __init__ defaults -- that would silently '
                'produce a model with the wrong input channels and/or stem stride.')
        sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(sd, strict=True)
        model.eval()
        return model

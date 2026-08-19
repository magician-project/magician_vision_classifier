"""Shared model definitions for the Magician vision classifier.

Extracted from trainMagicianVisionClassifierTorch.py so the trainer and the
two-head experiment fork (train2HeadMagicianVisionClassifierTorch.py) build models
from ONE definition. Four separate fixes had to be hand-copied between those two
files before this split existed; anything model-shaped belongs here now.

Contents:
  BasicResBlock / WaveletPool / CustomCNN  - the from-scratch architecture
  TIMM_BACKBONES / REPARAM_BACKBONES       - timm backbone registries
  NONBODY_PREFIXES                         - stem/head param prefixes (freeze logic)
  build_backbone(...)                      - name -> nn.Module, torchvision + timm + custom
"""

import math

import torch
import torch.nn as nn
import torchvision                    # the regnet_y_400mf branch uses the full path
from torch.nn import functional as F
from torchvision.models import (
    resnet18, ResNet18_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    resnext50_32x4d, ResNeXt50_32X4D_Weights,
    efficientnet_v2_s, EfficientNet_V2_S_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    swin_v2_t, Swin_V2_T_Weights,
    regnet_y_800mf, RegNet_Y_800MF_Weights,
    regnet_y_400mf, RegNet_Y_400MF_Weights,
    mobilenet_v3_small, MobileNet_V3_Small_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights,
    shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights,
    squeezenet1_1, SqueezeNet1_1_Weights,
    densenet121, DenseNet121_Weights,
    mnasnet0_5, MNASNet0_5_Weights,
    mnasnet1_0, MNASNet1_0_Weights,
)

# Stem/head parameter prefixes -- everything NOT starting with one of these counts as
# the pretrained "body" for the freeze-body schedule. 'conv_stem.' is timm's
# efficientnet-style stem (ghostnet_100, lcnet_050, mobilenetv4_conv_small); timm's
# other stems/heads are covered by 'stem.', 'head.' and 'classifier.'.
NONBODY_PREFIXES = ('conv1.', 'bn1.', 'stem.', 'conv_stem.', 'fc.', 'classifier.', 'head.',
                    'features.0.', 'features.conv0.', 'layers.0.')


class BasicResBlock(nn.Module):
    """Stride-1 residual block (conv-bn-relu-conv-bn + identity skip, then relu).
    Same channel count in/out so it can be stacked at a fixed stage width;
    lets a deeper FROM-SCRATCH net train inside a short epoch budget."""
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)


class WaveletPool(nn.Module):
    """Lossless 2x downsample: one level of Haar DWT, subbands stacked on channels.

    Drop-in for MaxPool2d(2), except C -> 4C: returns [LL,LH,HL,HH] per input
    channel. The transform is orthonormal and critically sampled, so it is
    invertible -- nothing is discarded, whereas MaxPool2d(2) throws away 3 of
    every 4 samples. The following stage conv then *learns* what to keep instead
    of the pool deciding blindly.

    Scale budget for the 300um (~3px at demosaic) defect on a 48x48 tile:
      stage 1 pool 48->24, level-1 detail resolves ~2px structure  <- the defect
      stage 2 pool 24->12, level-2 detail resolves ~4px            <- borderline
      stage 3 pool 12->6 (~8px) and stage 4 6->3 (~16px)           <- past it
    So this only earns its 4x channel cost at the first pool (and marginally the
    second); leave the later stages on MaxPool. See custom_wavelet_pools.

    Filters are fixed -> zero parameters. Note the decimated DWT is not
    shift-invariant, but it runs on conv1/early_convs feature maps (RF 7), not
    raw pixels, so the 3px response is already spatially smeared before decimation.
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = int(channels)
        # 2D Haar basis from l=[1,1]/sqrt(2), h=[1,-1]/sqrt(2); each has unit norm.
        ll = torch.tensor([[ 1.,  1.], [ 1.,  1.]]) * 0.5
        lh = torch.tensor([[ 1.,  1.], [-1., -1.]]) * 0.5   # horizontal edges
        hl = torch.tensor([[ 1., -1.], [ 1., -1.]]) * 0.5   # vertical edges
        hh = torch.tensor([[ 1., -1.], [-1.,  1.]]) * 0.5   # diagonal
        w = torch.stack([ll, lh, hl, hh]).unsqueeze(1)      # (4,1,2,2)
        self.register_buffer('filters', w.repeat(self.channels, 1, 1, 1))

    def forward(self, x):
        # groups=C: input channel g -> output channels 4g..4g+3 = its LL,LH,HL,HH
        return F.conv2d(x, self.filters, stride=2, groups=self.channels)


class CustomCNN(nn.Module):
    """
    A lightweight 4-layer convolutional network with adaptive global pooling
    and a two-tier fully-connected head. Designed for small tile images.

    Architecture:
        Conv2d → BN → ReLU → MaxPool (×4 stages, channels double each stage)
        → AdaptiveAvgPool2d(1,1) → Flatten
        → Linear → InstanceNorm1d → ReLU → Dropout
        → Linear → InstanceNorm1d → ReLU → Dropout
        → Linear → num_classes logits
    """
    def __init__(self, in_channels=4, intended_tile_size=64, num_classes=4, dropout_rate=0.5, base_channels=32, final_dense_layer=512, early_convs=0, channels=None, res_blocks=None, wavelet_pools=None, wavelet_stem=0):
        super(CustomCNN, self).__init__()
        if res_blocks is None:
            res_blocks = [0, 0, 0, 0]
        res_blocks = [int(x) for x in res_blocks]
        # 1-based stage indices whose MaxPool2d(2) is replaced by a lossless Haar
        # WaveletPool (which also widens that stage's output 4x, for free).
        wavelet_pools = set(int(s) for s in (wavelet_pools or []))
        # wavelet_stem: N levels of lossless Haar DWT applied to the RAW INPUT,
        # before conv1 -- the cheap analogue of convnext's stride-4 patchify stem.
        # Each level does CxHxW -> 4C x H/2 x W/2, so conv1 and every early_conv
        # then run on 1/4 the spatial positions per level. That is the dominant
        # inference cost: full-res early convs are bandwidth-bound, and measured
        # on an A6000 one level cut a batch-504 forward from 59.2 ms to 16.1 ms
        # (3.7x) at +0.02M params, since the DWT itself has none.
        #
        # NOTE this is NOT what custom_wavelet_pools=[1] does. That swaps stage 1's
        # pool, which happens AFTER the full-res conv1/early_convs and so removes
        # none of their cost. The two are independent and can be combined.
        #
        # The docstring on WaveletPool warns the decimated DWT is not shift-invariant
        # on raw pixels. Counter-evidence: convnext_tiny collapses 48->12 with a
        # stride-4 patchify on raw tiles and is the campaign's most accurate model,
        # and a DWT is invertible where patchify is a lossy learned projection.
        # Unproven either way -> screen it before trusting it.
        wavelet_stem = int(wavelet_stem or 0)

        # channels: explicit per-stage width [c1,c2,c3,c4]. Default is the
        # narrow-early/wide-late convention [base, 2base, 4base, 4base]. A
        # wide-early/taper-late schedule (e.g. [128,96,64,64]) puts capacity
        # where pooling has not yet destroyed information (see 3px-feature note).
        if channels is None:
            channels = [base_channels, base_channels*2, base_channels*4, base_channels*4]
        c1, c2, c3, c4 = [int(x) for x in channels]
        print("Custom CNN (",base_channels,",",final_dense_layer,") constructor / early_convs",
              early_convs, "/ channels", [c1,c2,c3,c4], "/ wavelet_pools", sorted(wavelet_pools))
        # Validation in forward() is against the ORIGINAL tile the loader sends;
        # the stem transformation happens inside this module.
        self.channels  = in_channels
        self.tile_size = intended_tile_size
        self.early_convs = int(early_convs)

        # Build the wavelet stem and work out what conv1 actually receives.
        self.wavelet_stem = nn.ModuleList()
        stem_ch, stem_sz = in_channels, int(intended_tile_size)
        for _ in range(wavelet_stem):
            if stem_sz % 2 != 0:
                raise ValueError(f"wavelet_stem: tile size {stem_sz} is not divisible by 2; "
                                 f"{wavelet_stem} levels is too many for a {intended_tile_size}px tile")
            self.wavelet_stem.append(WaveletPool(stem_ch))
            stem_ch *= 4
            stem_sz //= 2
        # Each of the 4 stages halves the map, so the body needs >= 16px to survive
        # pool1-4. Catch it here rather than as an opaque conv error mid-forward.
        if wavelet_stem and stem_sz < 16:
            raise ValueError(f"wavelet_stem={wavelet_stem} leaves a {stem_sz}x{stem_sz} map, "
                             f"too small for 4 pooling stages (needs >=16). Use fewer levels.")
        if wavelet_stem:
            print(f"Custom CNN wavelet_stem: {wavelet_stem} level(s) -> body sees "
                  f"{stem_ch}ch @ {stem_sz}x{stem_sz} (from {in_channels}ch @ {intended_tile_size}x{intended_tile_size})")

        def make_pool(stage, ch):
            """1-based stage -> (pool module, channels it emits)."""
            if stage in wavelet_pools:
                return WaveletPool(ch), ch * 4
            return nn.MaxPool2d(2), ch

        self.conv1 = nn.Conv2d(stem_ch, c1, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(c1)
        # 3px-feature preservation: extra stride-1 3x3 convs at FULL tile
        # resolution before the first pool, so a ~3px (300um) defect is encoded
        # into channels (RF grows 3->5->7...) before any downsampling. From-scratch
        # net -> no pretrained-body confound, unlike the resnet18 stems.
        self.early = nn.ModuleList()
        for _ in range(self.early_convs):
            self.early.append(nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c1), nn.ReLU(inplace=True)))
        self.pool1, p1 = make_pool(1, c1)

        self.conv2 = nn.Conv2d(p1, c2, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(c2)
        self.pool2, p2 = make_pool(2, c2)

        self.conv3 = nn.Conv2d(p2, c3, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(c3)
        self.pool3, p3 = make_pool(3, c3)

        self.conv4 = nn.Conv2d(p3, c4, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm2d(c4)
        self.pool4, p4 = make_pool(4, c4)

        # extra representational depth per stage (residual, at stage width),
        # inserted after the stage conv and BEFORE its pool
        stage_ch = [c1, c2, c3, c4]
        self.res = nn.ModuleList([
            nn.Sequential(*[BasicResBlock(stage_ch[i]) for _ in range(res_blocks[i])])
            for i in range(4)])
        print("Custom CNN res_blocks per stage:", res_blocks)

        prefinalLayerChannels     = int(final_dense_layer * 4.5)
        intermediateLayerChannels = int(final_dense_layer * 1.5)   # new FC layer

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(p4, prefinalLayerChannels)
        self.fc2 = nn.Linear(prefinalLayerChannels, intermediateLayerChannels)  # new layer
        # LayerNorm(..., elementwise_affine=False), NOT InstanceNorm1d.
        # The FC head feeds these a 2-D (B, C) tensor. InstanceNorm1d accepts that
        # but interprets it as an UNBATCHED (channels=B, length=C) signal, warning
        # "input's size at dim=0 does not match num_features" on every call, and
        # normalising each row over the feature dim -- which is LayerNorm, arrived
        # at by accident. PyTorch has deprecated feeding 2-D input to InstanceNorm,
        # so this was a future breakage as well as a misleading name.
        # elementwise_affine=False keeps it parameter-free, so the swap is
        # numerically identical (max abs diff 4.8e-07, i.e. float32 noise) and the
        # state_dict is unchanged -- every existing CustomCNN checkpoint still loads.
        # The bn_dense* attribute names are kept so the diff stays minimal.
        self.bn_dense1 = nn.LayerNorm(prefinalLayerChannels, elementwise_affine=False)
        self.bn_dense2 = nn.LayerNorm(intermediateLayerChannels, elementwise_affine=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(intermediateLayerChannels, num_classes)

    def forward(self, x):
        """
        Forward pass through the CNN backbone + FC head.
        Validates input dimensions then streams through 4 conv stages,
        global average pooling, and a two-tier FC head.

        Args:
            x: Input tensor of shape (B, channels, tile_size, tile_size).

        Returns:
            Logits tensor of shape (B, num_classes).

        Raises:
            ValueError: If input spatial/channel dimensions don't match expected.
        """
        if x.shape[1:] != (self.channels, self.tile_size, self.tile_size):  # Sanity check on desired input size
          raise ValueError(f"Input size must be {self.channels}x{self.tile_size}x{self.tile_size}, got {x.shape[1:]}")
        # Lossless Haar collapse before any full-res convolution (see wavelet_stem
        # in __init__). No-op when the stem is empty, which is the default.
        for lvl in self.wavelet_stem:
            x = lvl(x)
        x = F.relu(self.bn1(self.conv1(x)))
        for blk in self.early:
            x = blk(x)                     # extract 3px feature at full res
        x = self.pool1(self.res[0](x))
        x = self.pool2(self.res[1](F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.res[2](F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.res[3](F.relu(self.bn4(self.conv4(x)))))
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        # First FC
        x = F.relu(self.bn_dense1(self.fc1(x)))
        x = self.dropout(x)

        # Second FC
        x = F.relu(self.bn_dense2(self.fc2(x)))
        x = self.dropout(x)

        x = self.out(x)
        return x


# timm backbones usable as `model` in a config. All VERIFIED to build and run a
# forward pass at 5ch x 48x48 with num_classes=12 on timm 1.0.28 -- most of these
# ship a 224/256 default_cfg, and not all survive a 48px input.
#
# EXCLUDED after testing: `efficientvit_m0` hard-codes a 14x14 expectation
# ("input feature has wrong size, expect (14, 14), got (3, 3)"), the same failure
# mode that rules out maxvit_t. Do not re-add without checking a 48px forward.
#
# Measured on a (contended) A6000, batch 504, fp16, tiles/s -> Hz assumes ~2175
# tiles/frame; 23 Hz is the deployment target:
#     lcnet_050              0.62M   52.4 Hz      convnext_pico     8.54M  19.4 Hz
#     mobilenetv4_conv_small 2.51M   38.5 Hz      repvgg_a0         7.84M  16.0 Hz *
#     edgenext_xx_small      1.16M   27.4 Hz      ghostnet_100      3.92M  15.7 Hz
#     convnext_atto          3.38M   26.8 Hz      convnext_nano    14.96M  14.5 Hz
#     convnext_femto         4.84M   26.0 Hz      fastvit_t8        3.27M  10.4 Hz
#                                                 mobileone_s1      3.56M   6.5 Hz *
#                                                 mobileone_s0      4.28M   4.6 Hz *
TIMM_BACKBONES = {
    # reparameterizable: multi-branch while training, fuse to a plain 3x3 stack for
    # inference. The Hz above are the UNFUSED training graphs and badly understate
    # them -- see REPARAM_BACKBONES.
    'repvgg_a0', 'mobileone_s0', 'mobileone_s1',
    # properly small ConvNeXts (convnext_tiny is 28M for a 48px tile)
    'convnext_atto', 'convnext_femto', 'convnext_pico', 'convnext_nano',
    # modern edge-latency family
    'ghostnet_100', 'lcnet_050', 'mobilenetv4_conv_small', 'fastvit_t8', 'edgenext_xx_small',
    # 48px-verified 2026-08-10 (dev box, build+forward at 4x48x48): ConvNeXt V2 = v1+GRN,
    # drop-in same sizes; plus the tiny tier (small-is-better bet on this data).
    'convnextv2_atto', 'convnextv2_femto', 'convnextv2_pico', 'convnextv2_nano',
    'tinynet_c', 'tinynet_d', 'tinynet_e', 'efficientvit_b0', 'lcnet_100',
}

# These train multi-branch and MUST be reparameterized for deployment, via
# `timm.utils.reparameterize_model(model)` on the eval-mode model. Measured fusion
# speedups at 5ch x 48x48 batch 504 fp16 -- this is not a rounding error:
#     repvgg_a0     14.29 -> 2.35 ms   (6.1x)   16.0 ->  98.7 Hz
#     mobileone_s0  48.71 -> 5.77 ms   (8.4x)    4.6 ->  40.2 Hz
#     mobileone_s1  35.51 -> 8.85 ms   (4.0x)    6.5 ->  26.2 Hz
# Training and eval here run the unfused graph (mathematically equivalent, so the
# KPI is unaffected); the fusion belongs in the deployment path.
REPARAM_BACKBONES = {'repvgg_a0', 'mobileone_s0', 'mobileone_s1'}


def _stem_like(old_conv, in_channels, seed=True, tag=''):
    """Replacement for a pretrained 3-channel first conv that KEEPS the pretrained
    kernel instead of throwing it away.

    Every torchvision branch below has to widen the stem from RGB to our 4(+derived)
    polarization channels, and the obvious way -- assigning a fresh nn.Conv2d --
    silently discards the single most transferable layer in the network. timm's
    `in_chans=` does not: it tiles the pretrained RGB kernel across the new channels
    and rescales by 3/in_chans so the response magnitude is preserved. That is the
    only reason the timm sweep was not directly comparable with the earlier
    torchvision full trains (convnext_tiny 4.00 / efficientnet_b0 5.9 /
    regnet_y_800mf 6.2) -- those ran with random stems. This reproduces timm's
    adapt_input_conv so both families start from the same place.

    Geometry (out_channels, kernel, stride, padding, dilation, groups, bias) is copied
    from `old_conv`, so this only applies to the drop-in stem swaps. The resnet18_*
    variants deliberately build a DIFFERENT stem (deeper, different stride) and cannot
    be seeded -- they stay random, as before.

    seed=False reproduces the historical random-stem behaviour exactly.
    """
    new = nn.Conv2d(in_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                    padding=old_conv.padding, dilation=old_conv.dilation,
                    groups=old_conv.groups, bias=old_conv.bias is not None)
    if not seed:
        return new
    if old_conv.in_channels != 3:
        # Not an ImageNet RGB stem (pretrained=False builds random weights anyway).
        print(f"[stem] {tag}: NOT seeded (source stem has {old_conv.in_channels} in-channels, expected 3)")
        return new
    with torch.no_grad():
        w = old_conv.weight.detach().float()
        reps = int(math.ceil(in_channels / 3))
        w = w.repeat(1, reps, 1, 1)[:, :in_channels] * (3.0 / in_channels)
        new.weight.copy_(w.to(new.weight.dtype))
        if new.bias is not None and old_conv.bias is not None:
            new.bias.copy_(old_conv.bias.detach())
    print(f"[stem] {tag}: seeded {in_channels}ch stem from the pretrained 3ch kernel "
          f"(tiled x{reps}, scaled {3.0/in_channels:.3f})")
    return new


def retune_timm_stem_stride(model, stride, tile_size=48, tag=''):
    """Re-stride a timm patchify stem WITHOUT touching its pretrained weights.

    Why this exists. ConvNeXt patchifies with a 4x4 stride-4 conv, which on a 48px tile
    leaves 12x12 before the body. The four stages then halve: 12 -> 6 -> 3 -> 1. The last
    stage therefore runs on a SINGLE spatial position and the one before it on 3x3, so
    most of the architecture's spatial reasoning is spent on a map too small to hold the
    ~3px (300um) defect this classifier exists to find. That is a plausible reason a 8.54M
    ConvNeXt only ties a 4.02M EfficientNet here.

    The surgery keeps the 4x4 kernel and its pretrained weights EXACTLY as they are and
    changes only the stride (with padding chosen to keep the output size a clean
    tile_size/stride), so the filters are unchanged and only the sampling density moves.
    That is much safer than resizing the kernel, which would need interpolation and would
    genuinely alter what the pretrained filters compute.

    Cost scales with the square of the density change: stride 4 -> 2 quadruples the spatial
    positions in every stage, stride 4 -> 3 raises them ~1.8x. Measure, do not assume.
    """
    stem = getattr(model, 'stem', None)
    if stem is None or not isinstance(stem, nn.Sequential) or not isinstance(stem[0], nn.Conv2d):
        raise ValueError(f"{tag}: no timm-style Sequential(Conv2d, ...) stem to re-stride")
    old = stem[0]
    k = old.kernel_size[0]
    if old.stride[0] == stride:
        return model
    # want out = tile_size // stride  ->  pad = ((out-1)*stride + k - tile_size) / 2
    out = tile_size // stride
    pad2 = (out - 1) * stride + k - tile_size
    if pad2 < 0 or pad2 % 2:
        ok = [s for s in range(1, k + 1)
              if tile_size % s == 0
              and ((tile_size // s - 1) * s + k - tile_size) >= 0
              and ((tile_size // s - 1) * s + k - tile_size) % 2 == 0]
        raise ValueError(
            f"{tag}: stride {stride} with kernel {k} does not tile {tile_size}px under "
            f"symmetric padding (needs {pad2} total, which is negative or odd). "
            f"Strides that do work here: {ok}. Supporting the rest would need asymmetric "
            f"padding around the conv, which is not worth the extra module for an arm "
            f"whose effect the 0.43 noise floor may not resolve anyway.")
    new = nn.Conv2d(old.in_channels, old.out_channels, kernel_size=k, stride=stride,
                    padding=pad2 // 2, bias=old.bias is not None)
    with torch.no_grad():
        new.weight.copy_(old.weight)
        if new.bias is not None and old.bias is not None:
            new.bias.copy_(old.bias)
    stem[0] = new
    print(f"[stem-stride] {tag}: patchify stride {old.stride[0]} -> {stride} "
          f"(kernel {k} kept, pretrained weights copied verbatim, pad {pad2//2}); "
          f"body now sees {out}x{out} instead of {tile_size//old.stride[0]}x{tile_size//old.stride[0]}")
    return model


def build_backbone(model_type,
                   in_channels,
                   num_classes,
                   pretrained=True,
                   seed_pretrained_stem=True,
                   timm_stem_stride=None,
                   tile_size=48,
                   dropout_rate=0.5,
                   base_channels=32,
                   final_dense_layer=512,
                   custom_early_convs=0,
                   custom_channels=None,
                   custom_res_blocks=None,
                   custom_wavelet_pools=None,
                   custom_wavelet_stem=0):
    """Build a backbone by name. Covers the torchvision zoo (stem/head surgery for
    the 4+ channel polarization input), the timm registry, and the from-scratch
    CustomCNN. Raises ValueError on an unknown name.

    seed_pretrained_stem (default True) makes the torchvision stem swaps carry the
    pretrained RGB kernel over to the wider polarization stem, the way timm's
    in_chans= already does -- see _stem_like. Set it False to reproduce a run from
    before this existed (every torchvision result up to 2026-08 used random stems).
    """
    def _w(weights):
        """ImageNet weights when pretrained, else None (random init)."""
        return weights if pretrained else None

    # Only meaningful when there ARE pretrained weights to carry over.
    _seed = bool(pretrained) and bool(seed_pretrained_stem)

    #RESNEXT
    if model_type == 'resnext50':
        model = resnext50_32x4d(weights=_w(ResNeXt50_32X4D_Weights.IMAGENET1K_V2))
        model.conv1 = _stem_like(model.conv1, in_channels, seed=_seed, tag='resnext50.conv1')
        model.fc = nn.Linear(2048, num_classes)
    elif model_type == 'resnet18':
        model = resnet18(weights=_w(ResNet18_Weights.DEFAULT))
        model.conv1 = _stem_like(model.conv1, in_channels, seed=_seed, tag='resnet18.conv1')
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'resnet18_stem':
        # Deeper stem, same downsampling: more early capacity for the small
        # (~3x3) polarization micro-features the stock 7x7/2 conv blurs away.
        model = resnet18(weights=_w(ResNet18_Weights.DEFAULT))
        model.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'resnet18_fullres':
        # Maximum feature preservation for ~3px (300um) defects: the stem does
        # ZERO spatial downsampling. Three 3x3 stride-1 convs (RF 7, ch
        # 4->32->48->64) extract the feature at full 48x48 resolution and the
        # resnet body's own maxpool is removed, so layer1 runs at 48x48 and the
        # first downsample is layer2 (stride 2). Final map 6x6 (vs 1.5x1.5 stock).
        # ~16x the layer1 FLOPs of stock resnet18 — acceptable: inference cost is
        # traded back at deploy time via a larger tile step. Best fidelity for recall.
        model = resnet18(weights=_w(ResNet18_Weights.DEFAULT))
        model.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        model.maxpool = nn.Identity()  # NO downsampling before the residual body
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'resnet18_hires':
        # Feature-preserving stem for ~3px (300um) defects. Nyquist: a 3px
        # feature tolerates <1.5x downsampling BEFORE it is extracted, so the
        # stem runs THREE 3x3 stride-1 convs at full tile resolution
        # (receptive field 7, channels 4->32->48->64) to encode the feature
        # into the channel dimension, THEN a single maxpool downsamples the
        # now-redundant spatial grid. Contrast: stock/stem strides 2 on the
        # first conv (3px -> 0.75px at layer1, unrecoverable).
        model = resnet18(weights=_w(ResNet18_Weights.DEFAULT))
        model.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),   # only downsample: 48->24
        )
        model.maxpool = nn.Identity()  # resnet's own maxpool folded into the stem above
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'resnet18_fine':
        # Deeper stem AND stride 1: layer1 runs at 24x24 instead of 12x12 for
        # 48px tiles (only the maxpool downsamples). ~4x compute of resnet18.
        model = resnet18(weights=_w(ResNet18_Weights.DEFAULT))
        model.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
        )
        model.fc = nn.Linear(512, num_classes)
    elif model_type == 'convnext_tiny':
        model = convnext_tiny(weights=_w(ConvNeXt_Tiny_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='convnext_tiny.features.0.0')
        model.classifier[2]  = nn.Linear(768, num_classes)
    elif model_type == 'efficientnet_v2_s':
        model = efficientnet_v2_s(weights=_w(EfficientNet_V2_S_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='efficientnet_v2_s.features.0.0')
        model.classifier[1]  = nn.Linear(1280, num_classes, bias=True)
    elif model_type == 'swin_v2_t':
        model = swin_v2_t(weights=_w(Swin_V2_T_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='swin_v2_t.features.0.0')
        model.head = nn.Linear(768, num_classes)
    elif model_type == 'regnet_y_800mf':
        model = regnet_y_800mf(weights=_w(RegNet_Y_800MF_Weights.DEFAULT))
        model.stem[0] = _stem_like(model.stem[0], in_channels, seed=_seed, tag='regnet_y_800mf.stem.0')
        model.fc = nn.Linear(784, num_classes)
    elif model_type == 'regnet_y_400mf':
        model = torchvision.models.regnet_y_400mf(weights=_w(RegNet_Y_400MF_Weights.DEFAULT))
        model.stem[0] = _stem_like(model.stem[0], in_channels, seed=_seed, tag='regnet_y_400mf.stem.0')
        model.fc = nn.Linear(440, num_classes)
    elif model_type == 'mobilenet_v3_small':
        model = mobilenet_v3_small(weights=_w(MobileNet_V3_Small_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='mobilenet_v3_small.features.0.0')
        model.classifier[3] = nn.Linear(1024, num_classes)
    elif model_type == 'mobilenet_v3_large':
        model = mobilenet_v3_large(weights=_w(MobileNet_V3_Large_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='mobilenet_v3_large.features.0.0')
        model.classifier[3] = nn.Linear(1280, num_classes)
    elif model_type == 'shufflenet_v2_x0_5':
        model = shufflenet_v2_x0_5(weights=_w(ShuffleNet_V2_X0_5_Weights.DEFAULT))
        model.conv1[0] = _stem_like(model.conv1[0], in_channels, seed=_seed, tag='shufflenet_v2_x0_5.conv1.0')
        model.fc = nn.Linear(1024, num_classes)
    elif model_type == 'shufflenet_v2_x1_0':
        model = shufflenet_v2_x1_0(weights=_w(ShuffleNet_V2_X1_0_Weights.DEFAULT))
        model.conv1[0] = _stem_like(model.conv1[0], in_channels, seed=_seed, tag='shufflenet_v2_x1_0.conv1.0')
        model.fc = nn.Linear(1024, num_classes)
    elif model_type == 'squeezenet1_1':
        model = squeezenet1_1(weights=_w(SqueezeNet1_1_Weights.DEFAULT))
        model.features[0] = _stem_like(model.features[0], in_channels, seed=_seed, tag='squeezenet1_1.features.0')
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes
    elif model_type == 'efficientnet_b0':
        model = efficientnet_b0(weights=_w(EfficientNet_B0_Weights.DEFAULT))
        model.features[0][0] = _stem_like(model.features[0][0], in_channels, seed=_seed, tag='efficientnet_b0.features.0.0')
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == 'densenet121':
        model = densenet121(weights=_w(DenseNet121_Weights.DEFAULT))
        model.features.conv0 = _stem_like(model.features.conv0, in_channels, seed=_seed, tag='densenet121.features.conv0')
        model.classifier = nn.Linear(1024, num_classes)
    elif model_type == 'mnasnet0_5':
        model = mnasnet0_5(weights=_w(MNASNet0_5_Weights.DEFAULT))
        model.layers[0] = _stem_like(model.layers[0], in_channels, seed=_seed, tag='mnasnet0_5.layers.0')
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == 'mnasnet1_0':
        model = mnasnet1_0(weights=_w(MNASNet1_0_Weights.DEFAULT))
        model.layers[0] = _stem_like(model.layers[0], in_channels, seed=_seed, tag='mnasnet1_0.layers.0')
        model.classifier[1] = nn.Linear(1280, num_classes)
    elif ('custom' in model_type) or ('cnn' in model_type):
        model = CustomCNN(
                               in_channels=in_channels,
                               intended_tile_size=tile_size,
                               num_classes=num_classes,
                               dropout_rate=dropout_rate,
                               base_channels=base_channels,
                               final_dense_layer=final_dense_layer,
                               early_convs=custom_early_convs,
                               channels=custom_channels,
                               res_blocks=custom_res_blocks,
                               wavelet_pools=custom_wavelet_pools,
                               wavelet_stem=custom_wavelet_stem
                              )
    elif model_type in TIMM_BACKBONES or model_type.startswith('timm/'):
        # timm backbones. Unlike the torchvision branches above -- which each build
        # a FRESH nn.Conv2d stem and so THROW AWAY the pretrained first layer --
        # timm's in_chans adapts the pretrained RGB kernel to N channels instead of
        # discarding it, and num_classes builds the head. So these get the full
        # benefit of pretraining, stem included.
        import timm
        name = model_type.split('/', 1)[1] if model_type.startswith('timm/') else model_type
        model = timm.create_model(name,
                                       pretrained=pretrained,
                                       in_chans=in_channels,
                                       num_classes=num_classes)
        print(f"[timm] {name} pretrained={pretrained} in_chans={in_channels} "
              f"num_classes={num_classes} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        if timm_stem_stride:
            retune_timm_stem_stride(model, int(timm_stem_stride),
                                    tile_size=tile_size, tag=f'timm/{name}')
        if name in REPARAM_BACKBONES:
            print(f"[timm] NOTE {name} is reparameterizable: training runs the multi-branch "
                  f"graph, but deployment MUST call timm.utils.reparameterize_model() -- worth "
                  f"4-8x inference speed. KPI is unaffected (the fused model is equivalent).")
    else:
        _supported = "', '".join(sorted(TIMM_BACKBONES))
        raise ValueError(f"Unsupported model type: {model_type}. torchvision types include 'resnext50', "
                         f"'resnet18', 'convnext_tiny', 'efficientnet_v2_s', 'swin_v2_t', "
                         f"'regnet_y_800mf'; timm types: '{_supported}'; or 'timm/<any-timm-name>'.")
    return model

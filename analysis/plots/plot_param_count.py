#!/usr/bin/env python3
"""
Plot parameter counts for each trained model using the saved JSON configs.
Usage: python plot_param_count.py [--classifier-dir DIR] [--exclude MODEL ...]
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mvc.paths import repo_root

CLASSIFIER_DIR = repo_root()


def count_params(model_type, hparams, num_classes=9):
    # Import here so torch isn't loaded until needed
    import torch.nn as nn
    import torchvision
    from torchvision.models import (
        resnet18, ResNet18_Weights,
        resnext50_32x4d, ResNeXt50_32X4D_Weights,
        convnext_tiny, ConvNeXt_Tiny_Weights,
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
    from mvc.core.model_zoo import CustomCNN

    in_ch = 4
    tile  = hparams.get("tile_size", 48)
    base  = hparams.get("base_channels", 32)
    dense = hparams.get("final_dense_layer", 512)
    drop  = hparams.get("dropout_rate", 0.25)

    if model_type == "resnet18":
        m = resnet18(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        m.fc = nn.Linear(512, num_classes)
    elif model_type == "resnext50":
        m = resnext50_32x4d(weights=None)
        m.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        m.fc = nn.Linear(2048, num_classes)
    elif model_type == "convnext_tiny":
        m = convnext_tiny(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 96, 4, 4)
        m.classifier[2] = nn.Linear(768, num_classes)
    elif model_type == "efficientnet_v2_s":
        m = efficientnet_v2_s(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False)
        m.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == "efficientnet_b0":
        m = efficientnet_b0(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
        m.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == "swin_v2_t":
        m = swin_v2_t(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 96, 4, 4)
        m.head = nn.Linear(768, num_classes)
    elif model_type == "regnet_y_800mf":
        m = regnet_y_800mf(weights=None)
        m.stem[0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
        m.fc = nn.Linear(784, num_classes)
    elif model_type == "regnet_y_400mf":
        m = torchvision.models.regnet_y_400mf(weights=None)
        m.stem[0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
        m.fc = nn.Linear(440, num_classes)
    elif model_type == "mobilenet_v3_small":
        m = mobilenet_v3_small(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        m.classifier[3] = nn.Linear(1024, num_classes)
    elif model_type == "mobilenet_v3_large":
        m = mobilenet_v3_large(weights=None)
        m.features[0][0] = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        m.classifier[3] = nn.Linear(1280, num_classes)
    elif model_type == "shufflenet_v2_x0_5":
        m = shufflenet_v2_x0_5(weights=None)
        m.conv1[0] = nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False)
        m.fc = nn.Linear(1024, num_classes)
    elif model_type == "shufflenet_v2_x1_0":
        m = shufflenet_v2_x1_0(weights=None)
        m.conv1[0] = nn.Conv2d(in_ch, 24, 3, 2, 1, bias=False)
        m.fc = nn.Linear(1024, num_classes)
    elif model_type == "squeezenet1_1":
        m = squeezenet1_1(weights=None)
        m.features[0] = nn.Conv2d(in_ch, 64, 3, 2)
        m.classifier[1] = nn.Conv2d(512, num_classes, 1)
    elif model_type == "densenet121":
        m = densenet121(weights=None)
        m.features.conv0 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
        m.classifier = nn.Linear(1024, num_classes)
    elif model_type == "mnasnet0_5":
        m = mnasnet0_5(weights=None)
        m.layers[0] = nn.Conv2d(in_ch, 16, 3, 2, 1, bias=False)
        m.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type == "mnasnet1_0":
        m = mnasnet1_0(weights=None)
        m.layers[0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
        m.classifier[1] = nn.Linear(1280, num_classes)
    elif model_type in ("custom", "small_cnn", "verysmall_cnn"):
        m = CustomCNN(in_channels=in_ch, intended_tile_size=tile,
                      num_classes=num_classes, dropout_rate=drop,
                      base_channels=base, final_dense_layer=dense)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return sum(p.numel() for p in m.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier-dir", default=CLASSIFIER_DIR)
    parser.add_argument("--out", default=os.path.join(CLASSIFIER_DIR, "tb_plots", "param_count.png"))
    parser.add_argument("--exclude", nargs="+", default=[], metavar="MODEL",
                        help="Short model names to skip, e.g. --exclude mnasnet1_0 swin_v2_t")
    args = parser.parse_args()

    exclude = set(args.exclude)

    jsons = sorted(glob.glob(os.path.join(args.classifier_dir, "allclass_*.json")))
    entries = []
    for path in jsons:
        # Skip confusion matrix files
        if "_confusion" in path:
            continue
        with open(path) as f:
            cfg = json.load(f)
        short_name = os.path.basename(path).replace("allclass_", "").replace(".json", "")
        model_type = cfg.get("model", short_name)  # fallback: derive from filename
        if short_name in exclude or model_type in exclude:
            print(f"  Skipping {short_name}")
            continue
        print(f"  Counting {short_name} ({model_type})...")
        try:
            n = count_params(model_type, cfg["hparams"])
            entries.append((short_name, n))
            print(f"    {n:,} parameters")
        except Exception as e:
            print(f"    ERROR: {e}")

    if not entries:
        print("No models found.")
        return

    entries.sort(key=lambda x: x[1])
    names  = [e[0] for e in entries]
    params = [e[1] / 1e6 for e in entries]  # millions

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.45)))
    bars = ax.barh(names, params, color="steelblue", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{p:.2f}M" for p in params], padding=4, fontsize=8)
    ax.set_xlabel("Parameters (millions)")
    ax.set_title("Model Parameter Count (4-channel input, 9 classes)")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, max(params) * 1.18)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()

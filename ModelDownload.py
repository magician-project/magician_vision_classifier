#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2026 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Download trained classifier models from the model server and drop them where
ClassifierPnm.model_scan() (classifierPnm.py) will find them.

Model archives are flat zips named {model_name}_{timestamp}.zip containing
{model_name}.pth + {model_name}.json (+ confusion/threshold plots), uploaded
by scripts/uploadToAmmarServer.sh.

CLI:
  python3 ModelDownload.py --list                     # show remote models
  python3 ModelDownload.py crossvalv2aug_resnet18     # newest zip of a model
  python3 ModelDownload.py crossvalv2aug_resnet18_20260711_235959.zip
  python3 ModelDownload.py --all                      # every remote zip
Options: --dest DIR (default: this script's directory), --plots (also extract PNGs)

As a library:
  from ModelDownload import ensure_model
  ensure_model("crossvalv2aug_resnet18")   # no-op if model_scan already sees it
"""

import argparse
import os
import re
import sys
import urllib.request
import zipfile

BASE_URL = "http://ammar.gr/magician/models/CameraV2Models/"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def list_remote_models(base_url=BASE_URL):
    """Return the .zip filenames listed in the server's directory index."""
    with urllib.request.urlopen(base_url, timeout=30) as r:
        html = r.read().decode("utf-8", errors="replace")
    zips = sorted(set(re.findall(r'href="([^"]+\.zip)"', html)))
    return [os.path.basename(z) for z in zips]


def newest_zip_for(model_name, remote_zips):
    """Newest archive of a model ({model_name}_{timestamp}.zip, timestamps sort lexically)."""
    matches = [z for z in remote_zips
               if re.fullmatch(re.escape(model_name) + r"_\d{8}_\d{6}\.zip", z)]
    return sorted(matches)[-1] if matches else None


def download_zip(zip_name, dest_dir, base_url=BASE_URL):
    """Download one archive (with progress) and return its local path."""
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, zip_name)
    url = base_url + zip_name

    last_printed = [-10]

    def report(blocks, block_size, total):
        if total <= 0:
            return
        pct = min(100.0, 100.0 * blocks * block_size / total)
        if pct - last_printed[0] >= 10 or pct >= 100.0 > last_printed[0]:
            last_printed[0] = pct
            end = "\n" if pct >= 100.0 else ""
            sys.stdout.write("\r%s : %5.1f%% of %.1f MB%s" % (zip_name, pct, total / 1e6, end))
            sys.stdout.flush()

    urllib.request.urlretrieve(url, local, reporthook=report)
    print()
    return local


def extract_model(zip_path, directory, include_plots=False):
    """Extract the model files next to the other models; returns extracted names."""
    extracted = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if not name:
                continue
            if name.endswith((".pth", ".json")) or (include_plots and name.endswith(".png")):
                info.filename = name  # flatten any leading paths
                z.extract(info, directory)
                extracted.append(name)
    return extracted


def download_model(model_or_zip, directory=SCRIPT_DIR, include_plots=False, base_url=BASE_URL):
    """
    Download + extract a model by base name (newest archive) or exact zip name.
    Returns the extracted file names.
    """
    if model_or_zip.endswith(".zip"):
        zip_name = model_or_zip
    else:
        zip_name = newest_zip_for(model_or_zip, list_remote_models(base_url))
        if zip_name is None:
            raise FileNotFoundError(f"No archive for model '{model_or_zip}' on {base_url}")
    local = download_zip(zip_name, os.path.join(directory, "models"), base_url)
    extracted = extract_model(local, directory, include_plots)
    print(f"Extracted to {directory}: {', '.join(extracted)}")
    return extracted


def ensure_model(model_name, directory=SCRIPT_DIR, base_url=BASE_URL):
    """
    Glue to ClassifierPnm: if model_scan() already sees a valid {model_name}
    pth/json pair in `directory`, do nothing; otherwise fetch it from the server.
    Returns True if the model is available locally afterwards.
    """
    from classifierPnm import ClassifierPnm  # lazy: keeps --list usable without torch
    if model_name in ClassifierPnm.model_scan(directory):
        print(f"{model_name} already present in {directory}")
        return True
    download_model(model_name, directory, base_url=base_url)
    return model_name in ClassifierPnm.model_scan(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download classifier models from the model server")
    parser.add_argument("models", nargs="*", help="model base name(s) or exact zip name(s)")
    parser.add_argument("--list", action="store_true", help="list remote archives and exit")
    parser.add_argument("--all", action="store_true", help="download every remote archive")
    parser.add_argument("--dest", default=SCRIPT_DIR, help="directory for extracted pth/json")
    parser.add_argument("--plots", action="store_true", help="also extract confusion/curve PNGs")
    args = parser.parse_args()

    if args.list or (not args.models and not args.all):
        for z in list_remote_models():
            print(z)
        sys.exit(0)

    targets = list_remote_models() if args.all else args.models
    for m in targets:
        try:
            download_model(m, args.dest, include_plots=args.plots)
        except Exception as e:
            print(f"FAILED {m}: {e}")

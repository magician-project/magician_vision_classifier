#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2026 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Download trained classifier models from the model server and drop them where
ClassifierPnm.model_scan() (mvc/inference/classifier_pnm.py) will find them.

Model archives are flat zips named {model_name}_{timestamp}.zip containing
{model_name}.pth + {model_name}.json (+ confusion/threshold plots), uploaded
by scripts/uploadToAmmarServer.sh.

CLI (from the repo root):
  python3 -m mvc.inference.model_download --list                     # show remote models
  python3 -m mvc.inference.model_download crossvalv2aug_resnet18     # newest zip of a model
  python3 -m mvc.inference.model_download crossvalv2aug_resnet18_20260711_235959.zip
  python3 -m mvc.inference.model_download --all                      # every remote zip
Options: --dest DIR (default: the repo root), --plots (also extract PNGs)

As a library:
  from mvc.inference.model_download import ensure_model
  ensure_model("crossvalv2aug_resnet18")   # no-op if model_scan already sees it
"""

import argparse
import os
import re
import sys
import urllib.request
import zipfile

from mvc.paths import repo_root

BASE_URL = "http://ammar.gr/magician/models/CameraV2Models/"
SCRIPT_DIR = repo_root()


def list_remote_models(base_url=BASE_URL, timeout=30):
    """Return the .zip filenames listed in the server's directory index."""
    with urllib.request.urlopen(base_url, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="replace")
    zips = sorted(set(re.findall(r'href="([^"]+\.zip)"', html)))
    return [os.path.basename(z) for z in zips]


def remote_model_names(base_url=BASE_URL, timeout=30):
    """Base model names available remotely ({name}_{timestamp}.zip -> name)."""
    names = set()
    for z in list_remote_models(base_url, timeout):
        m = re.match(r"(.+)_\d{8}_\d{6}\.zip$", z)
        if m:
            names.add(m.group(1))
    return sorted(names)


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


def _regenerate_plots(directory, stem):
    """Render {stem}*.png locally from the sidecar JSONs just extracted, for an
    archive that did not carry them -- e.g. one built before mvc.export started
    bundling plots, or a run whose plots failed to render at training time (see
    mvc.export._find_plots / mvc.core.evaluation._write_plots, the same renderer
    used here)."""
    from mvc.core.evaluation import _write_plots
    for suffix in ("_confusion.json", "_threshold_curve.json"):
        p = os.path.join(directory, stem + suffix)
        if os.path.isfile(p):
            _write_plots(p)


def extract_model(zip_path, directory, include_plots=False):
    """Extract the model files next to the other models; returns extracted names.

    When include_plots is set and the archive itself carries no {model}*.png (an
    older export, or a run whose plots never rendered), they are regenerated
    locally from the extracted JSON sidecars instead of shipping a report-less
    model.
    """
    extracted = []
    pth_stem = None
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if not name:
                continue
            if name.endswith((".pth", ".json")) or (include_plots and name.endswith(".png")):
                info.filename = name  # flatten any leading paths
                z.extract(info, directory)
                extracted.append(name)
            if name.endswith(".pth"):
                pth_stem = name[:-len(".pth")]

    if include_plots and pth_stem and not any(n.endswith(".png") for n in extracted):
        _regenerate_plots(os.path.abspath(directory), pth_stem)

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


def ensure_model(model_name, directory=SCRIPT_DIR, base_url=BASE_URL, include_plots=True):
    """
    Glue to ClassifierPnm: if model_scan() already sees a valid {model_name}
    pth/json pair in `directory`, do nothing; otherwise fetch it from the server.
    include_plots defaults to True here (unlike download_model/the CLI) because
    callers of ensure_model want a fully usable local copy -- e.g. the web/GUI
    annotator's report pages, which read the confusion/threshold PNGs.
    Returns True if the model is available locally afterwards.
    """
    from mvc.inference.classifier_pnm import ClassifierPnm  # lazy: keeps --list usable without torch
    if model_name in ClassifierPnm.model_scan(directory):
        print(f"{model_name} already present in {directory}")
        return True
    download_model(model_name, directory, include_plots=include_plots, base_url=base_url)
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

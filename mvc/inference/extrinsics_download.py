#!/usr/bin/python3

"""
Author : "Ammar Qammaz"
Copyright : "2026 Foundation of Research and Technology, Computer Science Department Greece, See license.txt"
License : "FORTH"

Download the pattern-based extrinsics map + camera calibration
(analysis/extrinsics_from_pattern.py) from the same server the model checkpoints live
on, and drop them where mvc/inference/live_torch_ros.py's locate_pattern service looks
for them by default.

Mirrors mvc/inference/model_download.py's shape exactly (list/download/ensure), one
level simpler: there is no allowlist and no plots -- an extrinsics archive is always
exactly two files, {rig_name}_map.npz + {rig_name}_intrinsics.json, uploaded by
scripts/uploadExtrinsics.sh as {rig_name}_extrinsics_{timestamp}.zip. "rig_name" lets
more than one car/camera rig each have their own calibration+map on the same server;
a deployment with only one rig can just use the default name everywhere.

CLI (from the repo root):
  python3 -m mvc.inference.extrinsics_download --list                # show remote rigs
  python3 -m mvc.inference.extrinsics_download default                # newest archive
  python3 -m mvc.inference.extrinsics_download default_20260911_..zip # exact archive

As a library:
  from mvc.inference.extrinsics_download import ensure_extrinsics
  ensure_extrinsics("default")   # no-op if both files already exist locally
"""

import argparse
import os
import re
import sys
import urllib.request
import zipfile

from mvc.paths import repo_root

BASE_URL = "http://ammar.gr/magician/models/CameraV2Extrinsics/"
DEFAULT_RIG = "default"
DEFAULT_DIR = os.path.join(repo_root(), "experiments", "extrinsics")


def list_remote_extrinsics(base_url=BASE_URL, timeout=30):
    """The .zip filenames listed in the server's directory index."""
    with urllib.request.urlopen(base_url, timeout=timeout) as r:
        html = r.read().decode("utf-8", errors="replace")
    zips = sorted(set(re.findall(r'href="([^"]+\.zip)"', html)))
    return [os.path.basename(z) for z in zips]


def newest_zip_for(rig_name, remote_zips):
    """Newest archive of a rig ({rig_name}_extrinsics_{timestamp}.zip, timestamps sort
    lexically), same convention as model_download.newest_zip_for."""
    matches = [z for z in remote_zips
              if re.fullmatch(re.escape(rig_name) + r"_extrinsics_\d{8}_\d{6}\.zip", z)]
    return sorted(matches)[-1] if matches else None


def local_paths(rig_name, directory=DEFAULT_DIR):
    """Where locate_pattern's map/intrinsics for `rig_name` live once extracted."""
    return (os.path.join(directory, f"{rig_name}_map.npz"),
            os.path.join(directory, f"{rig_name}_intrinsics.json"))


def has_extrinsics(rig_name, directory=DEFAULT_DIR):
    map_path, intrinsics_path = local_paths(rig_name, directory)
    return os.path.isfile(map_path) and os.path.isfile(intrinsics_path)


def download_zip(zip_name, dest_dir, base_url=BASE_URL):
    os.makedirs(dest_dir, exist_ok=True)
    local = os.path.join(dest_dir, zip_name)
    urllib.request.urlretrieve(base_url + zip_name, local)
    return local


def extract_extrinsics(zip_path, directory):
    """Extract {rig_name}_map.npz + {rig_name}_intrinsics.json next to each other;
    returns the extracted file names. Refuses a zip missing either member -- half a
    calibration pair is worse than none, since it would fail later with a less obvious
    error (map without intrinsics, or vice versa)."""
    with zipfile.ZipFile(zip_path) as z:
        names = [os.path.basename(n) for n in z.namelist() if os.path.basename(n)]
        has_map = any(n.endswith("_map.npz") for n in names)
        has_intr = any(n.endswith("_intrinsics.json") for n in names)
        if not (has_map and has_intr):
            raise ValueError(f"{zip_path}: expected both *_map.npz and *_intrinsics.json, "
                             f"found {names}")
        os.makedirs(directory, exist_ok=True)
        extracted = []
        for info in z.infolist():
            name = os.path.basename(info.filename)
            if name.endswith(("_map.npz", "_intrinsics.json")):
                info.filename = name  # flatten any leading paths
                z.extract(info, directory)
                extracted.append(name)
    return extracted


def download_extrinsics(rig_or_zip, directory=DEFAULT_DIR, base_url=BASE_URL):
    """Download + extract a rig's map+intrinsics by name (newest archive) or exact zip
    name. Returns the extracted file names."""
    if rig_or_zip.endswith(".zip"):
        zip_name = rig_or_zip
    else:
        zip_name = newest_zip_for(rig_or_zip, list_remote_extrinsics(base_url))
        if zip_name is None:
            raise FileNotFoundError(f"No extrinsics archive for rig '{rig_or_zip}' on {base_url}")
    local = download_zip(zip_name, os.path.join(directory, "_zips"), base_url)
    extracted = extract_extrinsics(local, directory)
    print(f"Extracted to {directory}: {', '.join(extracted)}")
    return extracted


def ensure_extrinsics(rig_name=DEFAULT_RIG, directory=DEFAULT_DIR, base_url=BASE_URL):
    """If `rig_name`'s map+intrinsics already exist locally, do nothing; otherwise fetch
    them from the server. Returns True if both files are available locally afterwards.
    Mirrors model_download.ensure_model's contract exactly (used by
    live_torch_ros.py._load_extrinsics_map/_load_extrinsics_intrinsics the same way
    ClassifierPnm uses ensure_model)."""
    if has_extrinsics(rig_name, directory):
        print(f"{rig_name} extrinsics already present in {directory}")
        return True
    try:
        download_extrinsics(rig_name, directory, base_url=base_url)
    except Exception as e:
        print(f"Failed to fetch '{rig_name}' extrinsics from {base_url}: {e}")
        return False
    return has_extrinsics(rig_name, directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the pattern-based extrinsics map + calibration")
    parser.add_argument("rigs", nargs="*", help="rig name(s) or exact zip name(s)")
    parser.add_argument("--list", action="store_true", help="list remote archives and exit")
    parser.add_argument("--all", action="store_true", help="download every remote archive")
    parser.add_argument("--dest", default=DEFAULT_DIR, help="directory for extracted files")
    args = parser.parse_args()

    if args.list or (not args.rigs and not args.all):
        for z in list_remote_extrinsics():
            print(z)
        sys.exit(0)

    targets = args.rigs
    if args.all:
        targets = sorted({re.match(r"(.+)_extrinsics_\d{8}_\d{6}\.zip", z).group(1)
                          for z in list_remote_extrinsics()
                          if re.match(r"(.+)_extrinsics_\d{8}_\d{6}\.zip", z)})
    for r in targets:
        try:
            download_extrinsics(r, args.dest)
        except Exception as e:
            print(f"FAILED {r}: {e}")

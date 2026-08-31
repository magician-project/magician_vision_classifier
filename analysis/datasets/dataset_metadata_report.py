#!/usr/bin/env python3
"""Scan a tile dataset's HDF5 metadata and plot every characteristic it carries:
class distribution, tiles/frames per source recording, sensor Distance1-3 /
DistanceAverage, Light1-6 / lightDirection / lightNumber / lightConfidence,
per-recording duration & framerate derived from dev_timestamp, and camera
settings (exposure, gain, frameRate, blackLevel) read from each source
recording's info.json.

Metadata is per-tile but most fields (Distance*, Light*, timestamps, ...) are
constant across every tile cropped from the same source frame, so frame-level
fields are parsed once per frame (from the first tile that reaches it), not
once per tile -- the same trick build_coverage_val.py uses for recording/frame
ids, extended to the rest of the metadata. Camera settings are constant per
RECORDING and are not in the H5 at all -- they live in "<recording_dir>/info.json"
on the original capture drive, so they're only plotted when that drive/dir is
reachable (see --recording-lists).

Usage:
    python dataset_metadata_report.py --dataset-dir /home/ammar/Aug26_78K
    python dataset_metadata_report.py --dataset-dir /path/to/Set --splits train val
"""

import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRAME_FIELDS = ["Distance1", "Distance2", "Distance3", "DistanceAverage",
                "Light1", "Light2", "Light3", "Light4", "Light5", "Light6",
                "lightDirection", "lightNumber", "lightConfidence",
                "timestamp", "dev_timestamp", "Button1", "Button2"]

# Camera settings live per-RECORDING in "<recording_dir>/info.json", not in the H5.
RECORDING_INFO_FIELDS = ["exposure", "gain", "frameRate", "blackLevel"]

# train vs val get fixed, easily-told-apart colors; other splits fall back to DEFAULT_COLOR.
SPLIT_COLORS = {"train": "#4C72B0", "val": "#55A868", "coverage": "#DD8452"}
DEFAULT_COLOR = "#8172B2"

# Frame source paths end in "<name>.json(x,y)" (Aug26_78K train/val) or
# "<name>.png(x,y)" (hn_* hard-negative mining dumps) -- match either extension
# by anchoring on the trailing tile-offset parens instead of a fixed suffix.
SOURCE_RE = re.compile(r'^(.*)/([^/]+)\(\d+,\d+\)$')


def extract(h5_path, cache_path, limit=None):
    """Per-tile (rec_id, frame_id) + per-frame metadata fields. Cached to npz."""
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}

    f = h5py.File(h5_path, "r")
    raw = f.attrs["class_names"]
    class_names = json.loads(raw if isinstance(raw, str) else "".join(raw))
    labels = f["labels"][:limit] if limit else f["labels"][:]
    md = f["metadata"]
    n = len(labels)

    rec_ids = np.zeros(n, np.int32)
    frame_ids = np.zeros(n, np.int32)
    recs, frames = {}, {}
    frame_meta = []  # list of dicts, index == frame_id

    CH = 250_000
    for s in range(0, n, CH):
        e = min(s + CH, n)
        for j, m in enumerate(md[s:e]):
            m = m.decode() if isinstance(m, bytes) else m
            src = m.split('"source"', 1)[1].split('"', 2)[1] if '"source"' in m else "?"
            match = SOURCE_RE.match(src)
            if match:
                directory, frame_file = match.group(1), match.group(2)
            else:
                directory, frame_file = "?", src
            rec = directory.rstrip("/").rsplit("/", 1)[-1]
            frame = f"{rec}/{frame_file}"

            fid = frames.get(frame)
            if fid is None:
                fid = frames[frame] = len(frames)
                try:
                    d = json.loads(m)
                except Exception:
                    d = {}
                frame_meta.append({k: d.get(k) for k in FRAME_FIELDS})

            frame_ids[s + j] = fid
            rec_ids[s + j] = recs.setdefault(rec, len(recs))
        print(f"  extracted {e:,}/{n:,}", end="\r")
    f.close()
    print()

    rec_names = [k for k, _ in sorted(recs.items(), key=lambda kv: kv[1])]
    frame_names = [k for k, _ in sorted(frames.items(), key=lambda kv: kv[1])]

    out = dict(labels=labels, rec_id=rec_ids, frame_id=frame_ids,
               rec_names=np.array(rec_names, object),
               frame_names=np.array(frame_names, object),
               class_names=np.array(class_names, object),
               frame_meta=np.array(frame_meta, object))
    np.savez_compressed(cache_path, **out)
    return out


def to_float(values):
    """Best-effort float array from a list of raw (str/num/None) metadata values."""
    out = np.full(len(values), np.nan)
    for i, v in enumerate(values):
        if v is None:
            continue
        try:
            out[i] = float(v)
        except (TypeError, ValueError):
            pass
    return out


def load_recording_info(list_files):
    """{recording_name: {exposure, gain, frameRate, blackLevel}} from info.json
    next to each recording directory listed (one path per line) in list_files."""
    info = {}
    for lf in list_files:
        if not os.path.exists(lf):
            continue
        for line in open(lf):
            d = line.strip()
            if not d:
                continue
            info_path = os.path.join(d, "info.json")
            if not os.path.exists(info_path):
                continue
            try:
                raw = json.load(open(info_path))
            except Exception:
                continue
            rec = os.path.basename(d.rstrip("/"))
            info[rec] = {k: raw.get(k) for k in RECORDING_INFO_FIELDS}
    return info


def value_bar(ax, counts, title, ylabel, rotate=0, color=DEFAULT_COLOR):
    """Bar chart over a {value: count} dict, sorted by value -- for small
    discrete settings (exposure, frameRate, ...) where a histogram would be
    misleading (bins splitting/merging a handful of exact camera settings)."""
    items = sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))
    labels = [str(k) for k, _ in items]
    values = [v for _, v in items]
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)


def per_recording_duration_fps(frame_meta, rec_id_per_frame, rec_names):
    """(rec_name -> duration_sec, fps) from dev_timestamp span per recording."""
    dev_ts = to_float([m.get("dev_timestamp") for m in frame_meta])
    by_rec = defaultdict(list)
    for r, t in zip(rec_id_per_frame, dev_ts):
        if not np.isnan(t):
            by_rec[r].append(t)
    result = {}
    for r, ts in by_rec.items():
        ts = sorted(ts)
        duration = (ts[-1] - ts[0]) / 1000.0  # dev_timestamp ~ milliseconds
        fps = (len(ts) - 1) / duration if duration > 0 else float("nan")
        result[rec_names[r]] = (duration, fps)
    return result


def bar(ax, labels, values, title, ylabel, rotate=90, top=None, color=DEFAULT_COLOR):
    if top is not None and len(labels) > top:
        order = np.argsort(values)[::-1][:top]
        labels = [labels[i] for i in order]
        values = [values[i] for i in order]
        title += f" (top {top})"
    ax.bar(range(len(labels)), values, color=color)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=rotate, ha="right" if rotate else "center", fontsize=7)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.margins(x=0.01)


def report_split(h5_path, out_dir, cache_dir, split_name, recording_info=None, limit=None):
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split_name}_meta_cache.npz")
    d = extract(h5_path, cache_path, limit=limit)
    return analyze_and_plot(d, out_dir, split_name, recording_info=recording_info)


def report_coverage(train_h5_path, train_cache_path, manifest_path, out_dir, recording_info=None):
    """Coverage isn't a separate dataset.h5 -- it's a subset of TRAIN's own frames,
    carved out by build_coverage_val.py and recorded (as full source paths) in its
    manifest json. Reuse train's cached per-tile extraction and just restrict to the
    tiles whose frame is in the manifest, instead of rescanning the H5."""
    d = extract(train_h5_path, train_cache_path)
    manifest = json.load(open(manifest_path))
    frame_names_full = list(d["frame_names"])

    # Manifest frames are full source paths (".../<rec>/<file>.json"); our frame
    # keys are "<rec>/<file>.json" -- normalize the manifest side to match.
    def to_key(raw):
        parts = raw.rstrip("/").split("/")
        return f"{parts[-2]}/{parts[-1]}"

    wanted = {to_key(f) for f in manifest["frames"]}
    keep_frame_ids = {i for i, fn in enumerate(frame_names_full) if fn in wanted}
    if not keep_frame_ids:
        raise SystemExit(f"none of {manifest_path}'s {len(wanted)} frames matched train's "
                          f"cached frame names -- cache/manifest mismatch?")
    tile_mask = np.isin(d["frame_id"], list(keep_frame_ids))
    print(f"coverage: {len(keep_frame_ids):,}/{len(frame_names_full):,} train frames, "
          f"{int(tile_mask.sum()):,} tiles")
    return analyze_and_plot(d, out_dir, "coverage", recording_info=recording_info,
                             tile_indices=np.where(tile_mask)[0])


def analyze_and_plot(d, out_dir, split_name, recording_info=None, tile_indices=None):
    """Plot every characteristic for one split. `tile_indices`, if given, restricts
    to a subset of tiles from `d` (used for the coverage pseudo-split) instead of
    using every tile -- frame-level stats are then recomputed over just the frames
    those tiles touch, not every frame train/val ever saw."""
    os.makedirs(out_dir, exist_ok=True)

    class_names = list(d["class_names"])
    frame_names_full = list(d["frame_names"])
    rec_names = list(d["rec_names"])

    if tile_indices is not None:
        labels = d["labels"][tile_indices]
        rec_id = d["rec_id"][tile_indices]
        frame_id_raw = d["frame_id"][tile_indices]
        uniq_frames, frame_id = np.unique(frame_id_raw, return_inverse=True)
        frame_names = [frame_names_full[i] for i in uniq_frames]
        frame_meta = [d["frame_meta"][i] for i in uniq_frames]
    else:
        labels = d["labels"]
        rec_id = d["rec_id"]
        frame_id = d["frame_id"]
        frame_names = frame_names_full
        frame_meta = list(d["frame_meta"])

    n_tiles, n_frames, n_recs = len(labels), len(frame_names), len(rec_names)
    n_recs_present = int(np.unique(rec_id).size) if n_tiles else 0
    summary = {"split": split_name, "n_tiles": n_tiles, "n_frames": n_frames,
               "n_recordings": n_recs_present}
    color = SPLIT_COLORS.get(split_name, DEFAULT_COLOR)

    # rec_id per unique frame (first tile carrying each frame id also carries its rec)
    rec_of_frame = np.zeros(n_frames, np.int32)
    seen = np.zeros(n_frames, bool)
    for r, fid in zip(rec_id, frame_id):
        if not seen[fid]:
            rec_of_frame[fid] = r
            seen[fid] = True

    # --- class distribution (tile-level: the actual training unit) ---
    cls_counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    bar(ax, [c.replace("class_", "") for c in class_names], cls_counts,
        f"{split_name}: tiles per class", "tiles", color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split_name}_class_distribution.png"), dpi=130)
    plt.close(fig)
    summary["tiles_per_class"] = {class_names[i]: int(cls_counts[i])
                                   for i in range(len(class_names)) if cls_counts[i]}

    # --- tiles & frames per recording ---
    tiles_per_rec = np.bincount(rec_id, minlength=n_recs)
    frames_per_rec = np.bincount(rec_of_frame, minlength=n_recs)
    fig, axes = plt.subplots(2, 1, figsize=(12, 9))
    bar(axes[0], rec_names, tiles_per_rec, f"{split_name}: tiles per recording", "tiles",
        top=40, color=color)
    bar(axes[1], rec_names, frames_per_rec, f"{split_name}: frames per recording", "frames",
        top=40, color=color)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split_name}_recordings.png"), dpi=130)
    plt.close(fig)

    # --- Distance* histograms (frame-level, deduplicated) ---
    dist_fields = ["Distance1", "Distance2", "Distance3", "DistanceAverage"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, field in zip(axes.flat, dist_fields):
        vals = to_float([m.get(field) for m in frame_meta])
        vals = vals[~np.isnan(vals)]
        ax.hist(vals, bins=60, color=color)
        ax.set_title(f"{split_name}: {field} (mean={vals.mean():.1f})" if len(vals) else field)
        ax.set_xlabel("mm")
        summary[f"mean_{field}"] = float(vals.mean()) if len(vals) else None
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split_name}_distances.png"), dpi=130)
    plt.close(fig)

    # --- Light* aggregate + lightDirection + lightNumber + lightConfidence ---
    light_fields = [f"Light{i}" for i in range(1, 7)]
    light_on_counts = {lf: sum(1 for m in frame_meta if str(m.get(lf)) == "1")
                        for lf in light_fields}
    direction_counts = Counter(m.get("lightDirection") for m in frame_meta
                                if m.get("lightDirection") is not None)
    number_counts = Counter(m.get("lightNumber") for m in frame_meta
                             if m.get("lightNumber") is not None)
    confidence = to_float([m.get("lightConfidence") for m in frame_meta])
    confidence = confidence[~np.isnan(confidence)]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    bar(axes[0, 0], light_fields, [light_on_counts[lf] for lf in light_fields],
        f"{split_name}: frames per Light channel ON", "frames", rotate=0, color=color)
    dir_labels = list(direction_counts.keys())
    bar(axes[0, 1], dir_labels, [direction_counts[k] for k in dir_labels],
        f"{split_name}: lightDirection", "frames", color=color)
    num_labels = [str(k) for k in sorted(number_counts, key=lambda x: (x is None, x))]
    bar(axes[1, 0], num_labels,
        [number_counts[k if not str(k).isdigit() else k] for k in
         sorted(number_counts, key=lambda x: (x is None, x))],
        f"{split_name}: lightNumber", "frames", rotate=0, color=color)
    axes[1, 1].hist(confidence, bins=30, color=color)
    axes[1, 1].set_title(f"{split_name}: lightConfidence"
                          f" (mean={confidence.mean():.2f})" if len(confidence) else "lightConfidence")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split_name}_lights.png"), dpi=130)
    plt.close(fig)
    summary["frames_per_light_channel"] = light_on_counts
    summary["light_direction_counts"] = dict(direction_counts)

    # --- duration & framerate per recording (from dev_timestamp) ---
    dur_fps = per_recording_duration_fps(frame_meta, rec_of_frame, rec_names)
    durations = np.array([v[0] for v in dur_fps.values()])
    fps = np.array([v[1] for v in dur_fps.values()])
    fps = fps[~np.isnan(fps)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(durations, bins=30, color=color)
    axes[0].set_title(f"{split_name}: recording duration (s)"
                       f" mean={durations.mean():.1f}s" if len(durations) else "duration")
    axes[1].hist(fps, bins=30, color=color)
    axes[1].set_title(f"{split_name}: framerate (fps)"
                       f" mean={fps.mean():.1f}" if len(fps) else "framerate")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{split_name}_duration_fps.png"), dpi=130)
    plt.close(fig)
    summary["mean_recording_duration_sec"] = float(durations.mean()) if len(durations) else None
    summary["mean_framerate_fps"] = float(fps.mean()) if len(fps) else None

    # --- camera settings (exposure, gain, frameRate, blackLevel) from info.json ---
    if recording_info:
        present_recs = [rec_names[i] for i in np.unique(rec_id)] if n_tiles else []
        matched_recs = [r for r in present_recs if r in recording_info]
        summary["recordings_with_info_json"] = len(matched_recs)
        summary["recordings_missing_info_json"] = [r for r in present_recs if r not in recording_info]
        if matched_recs:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            for ax, field in zip(axes.flat, RECORDING_INFO_FIELDS):
                # weight by frame count -- how many frames were captured under each setting
                counts = Counter()
                for r in matched_recs:
                    v = recording_info[r].get(field)
                    if v is None:
                        continue
                    ridx = rec_names.index(r)
                    counts[v] += int(frames_per_rec[ridx])
                value_bar(ax, counts, f"{split_name}: {field}", "frames", color=color)
                summary[f"{field}_by_frames"] = {str(k): v for k, v in counts.items()}
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{split_name}_camera_settings.png"), dpi=130)
            plt.close(fig)

    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", required=True,
                     help="dataset root containing <split>/dataset.h5 subfolders, e.g. Aug26_78K")
    ap.add_argument("--splits", nargs="*", default=None,
                     help="which split subfolders to scan (default: autodetect */dataset.h5)")
    ap.add_argument("--out-dir", default=None,
                     help="where to write plots/summary (default: <dataset-dir>/metadata_report)")
    ap.add_argument("--cache-dir", default=None,
                     help="where to cache extracted metadata npz (default: --out-dir)")
    ap.add_argument("--limit", type=int, default=None,
                     help="cap tiles scanned per split, for a quick test run")
    ap.add_argument("--recording-lists", nargs="*", default=None,
                     help="text files, one recording directory per line, used to find each "
                          "recording's info.json for camera settings (exposure/gain/frameRate/"
                          "blackLevel). Default: autodetect '<dataset-dir>_*_list.txt'")
    ap.add_argument("--no-camera-settings", action="store_true",
                     help="skip the exposure/gain/frameRate/blackLevel plots entirely")
    ap.add_argument("--coverage-manifest", default=None,
                     help="also report the COVERAGE set: a val_coverage_frames.json manifest "
                          "(see build_coverage_val.py) naming frames carved out of train. "
                          "Reuses train's cached extraction instead of rescanning the H5.")
    args = ap.parse_args()

    if args.splits:
        h5s = [(s, os.path.join(args.dataset_dir, s, "dataset.h5")) for s in args.splits]
    else:
        found = sorted(glob.glob(os.path.join(args.dataset_dir, "*", "dataset.h5")))
        h5s = [(os.path.basename(os.path.dirname(p)), p) for p in found]
    h5s = [(s, p) for s, p in h5s if os.path.exists(p)]
    if not h5s:
        raise SystemExit(f"no dataset.h5 found under {args.dataset_dir}")

    out_dir = args.out_dir or os.path.join(args.dataset_dir, "metadata_report")
    cache_dir = args.cache_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    recording_info = {}
    if not args.no_camera_settings:
        list_files = args.recording_lists
        if list_files is None:
            list_files = sorted(glob.glob(f"{args.dataset_dir.rstrip('/')}_*_list.txt"))
        recording_info = load_recording_info(list_files)
        if recording_info:
            print(f"loaded camera settings for {len(recording_info)} recordings "
                  f"from {list_files}")
        else:
            print(f"no recording info.json found via {list_files or '(no list files found)'} "
                  "-- camera-settings plots skipped")

    all_summaries = {}
    for split_name, h5_path in h5s:
        print(f"=== {split_name}: {h5_path} ===")
        all_summaries[split_name] = report_split(h5_path, out_dir, cache_dir, split_name,
                                                   recording_info=recording_info,
                                                   limit=args.limit)

    if args.coverage_manifest:
        print(f"=== coverage: {args.coverage_manifest} ===")
        train_h5_path = dict(h5s).get("train", os.path.join(args.dataset_dir, "train", "dataset.h5"))
        train_cache_path = os.path.join(cache_dir, "train_meta_cache.npz")
        all_summaries["coverage"] = report_coverage(train_h5_path, train_cache_path,
                                                      args.coverage_manifest, out_dir,
                                                      recording_info=recording_info)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=1)

    print(f"\nwrote plots + {summary_path} to {out_dir}")
    for split_name, s in all_summaries.items():
        print(f"\n{split_name}: {s['n_tiles']:,} tiles, {s['n_frames']:,} frames, "
              f"{s['n_recordings']} recordings")
        if s.get("mean_recording_duration_sec") is not None:
            print(f"  mean recording duration: {s['mean_recording_duration_sec']:.1f}s, "
                  f"mean framerate: {s['mean_framerate_fps']:.1f} fps")
        if s.get("mean_DistanceAverage") is not None:
            print(f"  mean DistanceAverage: {s['mean_DistanceAverage']:.1f} mm")


if __name__ == "__main__":
    main()

#!/usr/bin/python3

"""
Publishes a directory of dataset frames (e.g. colorFrame_0_*.png) into a
SharedMemoryVideoBuffers stream, looping by default, so live_torch.py can be
exercised against a recorded dataset instead of a live camera.

Reference implementation: mga/stream_dataset.py's FolderStreamer +
StreamerFrame.stream_loop in the magician_grabber_annotator repo.

Usage:
    python3 -m mvc.inference.folder_shared_memory_streamer /path/to/dataset
"""

import argparse
import os
import sys
import time

import cv2

from mvc.core.shared_memory import SharedMemoryManager
from mvc.paths import repo_root

IMAGE_EXTENSIONS = (".png", ".pnm", ".jpg", ".jpeg")


def list_image_files(directory, label=None):
    """Sorted image files in directory, optionally filtered to a filename prefix."""
    files = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            continue
        if "foreground.png" in filename:
            continue
        if label and not filename.startswith(label):
            continue
        files.append(os.path.join(directory, filename))
    return files


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", help="directory of dataset frames to stream")
    parser.add_argument("--label", default="colorFrame_0_",
                         help="filename prefix filter (default colorFrame_0_; "
                              "falls back to every image if nothing matches)")
    parser.add_argument("--stream", default="stream1", help="shared memory stream name")
    parser.add_argument("--descriptor", default="video_frames.shm",
                         help="shared memory descriptor file")
    parser.add_argument("--fps", type=float, default=10.0,
                         help="playback rate (0 = as fast as possible)")
    parser.add_argument("--delay", type=float, default=None,
                         help="fixed per-frame delay in milliseconds (overrides --fps)")
    parser.add_argument("--no-loop", dest="loop", action="store_false",
                         help="stop after one pass instead of looping forever")
    parser.set_defaults(loop=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not os.path.isdir(args.dataset):
        sys.exit(f"Not a directory: {args.dataset}")

    files = list_image_files(args.dataset, args.label)
    if not files and args.label:
        print(f"No files matched prefix {args.label!r} in {args.dataset}, "
              f"falling back to every image file")
        files = list_image_files(args.dataset, None)
    if not files:
        sys.exit(f"No image files found in {args.dataset}")

    first = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        sys.exit(f"Could not read {files[0]}")
    height, width = first.shape[:2]
    channels = 1 if first.ndim == 2 else first.shape[2]
    period = args.delay / 1000.0 if args.delay is not None else (1.0 / args.fps if args.fps > 0 else 0.0)
    print(f"Streaming {len(files)} frames from {args.dataset} "
          f"as {width}x{height}x{channels} -> stream {args.stream!r} "
          f"(loop={args.loop}, "
          + (f"delay={args.delay}ms)" if args.delay is not None else f"fps={args.fps})"))

    smm = SharedMemoryManager(
        os.path.join(repo_root(), "libSharedMemoryVideoBuffers.so"),
        descriptor=args.descriptor,
        frameName=args.stream,
        connect=False,
        width=width,
        height=height,
        channels=channels,
    )

    frame_index = 0
    try:
        while True:
            path = files[frame_index]
            frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if frame is not None:
                smm.copy_numpy_to_shared_memory(frame)
            else:
                print(f"Could not read frame {path}", file=sys.stderr)

            frame_index += 1
            if frame_index >= len(files):
                if not args.loop:
                    break
                frame_index = 0

            if period:
                time.sleep(period)
    except KeyboardInterrupt:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

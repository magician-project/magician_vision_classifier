#!/usr/bin/env python3
"""Convert a magician_grabber-style `.calib` file (Stereolabs-compatible "Matlab Load()"
text format, produced by PolarShadowVisionSensorCalibrationFromDatasets.py:
https://github.com/magician-project/magician_grabber/blob/main/PolarShadowVisionSensorCalibrationFromDatasets.py)
into the camera_matrix/dist_coeffs intrinsics.json shape
analysis/extrinsics_from_pattern.py and analysis/extrinsics_from_markers.py's own
load_intrinsics() expect.

FORMAT. '%'-prefixed lines are headers/labels, not data; the four data blocks are
introduced by a bare '%I' / '%D' / '%T' / '%R' marker and are the 9 (3x3, row-major),
5, 3, 3 numbers that follow it. The generating tool prints every number with EXACTLY
8 digits after the decimal point (a fixed `%.8f`) and no separator between fields --
seen directly in a real sample: "...2332.116821430.00000000561..." is
[2332.11682143, 0.0, 561...], not [2332.1168214, 30.0, ...] or any other split. A
generic "find float-shaped substrings" regex genuinely cannot tell those apart; this
instead reads a fixed run of digits, then (if a '.' is present) exactly 8 more digits,
which is unambiguous and was verified character-for-character against a real
flattened `.calib` (chat-pasted, so its original newlines were already gone) before
being trusted. Zero-valued Translation/Rotation fields are seen printed as bare
integers ("000" = 0, 0, 0) rather than "0.00000000" x3 -- handled the same way, since
the rule only consumes a '.'-plus-8-digits tail when a '.' is actually next.

Usage:
  python analysis/calib_to_intrinsics.py last.calib --out intrinsics.json
"""

import argparse
import json
import re

# Order matches the calibration file's own documented header: I[1,1] I[1,2] I[1,3]
# I[2,1] I[2,2] I[2,3] I[3,1] I[3,2] I[3,3] -- row-major, exactly numpy's default reshape.
N_VALUES = {"I": 9, "D": 5, "T": 3, "R": 3}


def _read_fixed_floats(s, count, decimals=8):
    """Read `count` numbers off the front of `s`: an optional '-', a run of digits,
    then EXACTLY `decimals` more digits if (and only if) a '.' immediately follows the
    integer part -- see the module docstring for why this, not a greedy float regex,
    is what correctly splits a run of concatenated fixed-precision numbers. Returns
    (values, unconsumed_remainder_of_s)."""
    vals = []
    i = 0
    for _ in range(count):
        start = i
        if i < len(s) and s[i] == '-':
            i += 1
        digit_start = i
        while i < len(s) and s[i].isdigit():
            i += 1
        if i == digit_start:
            raise ValueError(f"expected a digit at position {i} of {s[start:start+20]!r}...")
        if i < len(s) and s[i] == '.':
            i += 1 + decimals
        vals.append(float(s[start:i]))
    return vals, s[i:]


def parse_calib(text):
    """{'I': [9 floats], 'D': [5 floats], 'T': [3 floats], 'R': [3 floats], 'width':
    int, 'height': int}. Raises ValueError naming which block is missing/short rather
    than silently returning a wrong-shape result."""
    width_m = re.search(r"%Width\s*(\d+)", text)
    height_m = re.search(r"%Height\s*(\d+)", text)
    if not (width_m and height_m):
        raise ValueError("could not find %Width/%Height in the calibration text")

    out = {"width": int(width_m.group(1)), "height": int(height_m.group(1))}
    for key, n in N_VALUES.items():
        # The marker itself is a bare '%<key>' possibly immediately followed (no
        # separator) by its first number, e.g. flattened text can read '...%I2332.11...'.
        # NOT \b here: "%I" then a digit has no letter/digit word-boundary (both are
        # \w), so \b fails to match the real marker; a negative lookahead for a
        # following LETTER is what actually distinguishes '%I2332...' (the data) from
        # '%Intrinsics...' (the header line for the same key) or '%Description='/
        # '%Distortion' (a different key's header containing this one's letter).
        m = re.search(rf"%{key}(?![a-zA-Z])\s*([\-0-9.].*)", text, flags=re.DOTALL)
        if not m:
            if key in ("T", "R"):
                out[key] = None  # not used by to_intrinsics() -- see below
                continue
            raise ValueError(f"could not find a %{key} data block")
        tail = m.group(1)
        stop = tail.find("%")
        chunk = (tail if stop == -1 else tail[:stop]).strip()
        try:
            values, remainder = _read_fixed_floats(chunk, n)
            if remainder.strip(" \n\r\t"):
                raise ValueError(f"{len(remainder)} unconsumed characters: {remainder[:20]!r}")
        except (ValueError, IndexError) as e:
            # I/D (the two blocks intrinsics.json actually needs) always carry a '.'
            # per value in every real file seen, so the fixed-width rule applies
            # cleanly there. T/R are seen printed as bare, undecorated integers with
            # no separator ("000" = 0,0,0) when they are exactly zero -- genuinely
            # ambiguous to split without a decimal point to anchor on, and NOT NEEDED
            # here (to_intrinsics() never reads them), so failure there is a warning,
            # not a hard error; I/D failing IS one, since a wrong camera_matrix or
            # dist_coeffs silently poisons every pose this feeds into.
            if key in ("T", "R"):
                print(f"NOTE: could not parse %{key} unambiguously ({e}) -- "
                      f"unused by to_intrinsics(), continuing.")
                out[key] = None
                continue
            raise ValueError(f"%{key}: failed to read {n} fixed-width numbers from "
                             f"{chunk[:60]!r}...: {e}")
        else:
            out[key] = values
    return out


def to_intrinsics(parsed):
    """(camera_matrix, dist_coeffs) in the shape load_intrinsics() reads -- a 3x3 list
    of lists and a flat list, respectively."""
    i = parsed["I"]
    camera_matrix = [[i[0], i[1], i[2]], [i[3], i[4], i[5]], [i[6], i[7], i[8]]]
    dist_coeffs = parsed["D"]
    return camera_matrix, dist_coeffs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_file", help="path to the .calib file")
    ap.add_argument("--out", default="intrinsics.json")
    args = ap.parse_args()

    with open(args.calib_file) as f:
        text = f.read()
    parsed = parse_calib(text)
    camera_matrix, dist_coeffs = to_intrinsics(parsed)

    print(f"parsed {args.calib_file}: {parsed['width']}x{parsed['height']}, "
          f"fx={camera_matrix[0][0]:.2f} fy={camera_matrix[1][1]:.2f} "
          f"cx={camera_matrix[0][2]:.2f} cy={camera_matrix[1][2]:.2f}")
    zero3 = [0.0, 0.0, 0.0]
    if (parsed["T"] not in (None, zero3)) or (parsed["R"] not in (None, zero3)):
        print(f"NOTE: T={parsed['T']} R={parsed['R']} are non-zero but load_intrinsics() "
              f"only reads camera_matrix/dist_coeffs -- extrinsic T/R from this file are "
              f"not carried into intrinsics.json (extrinsics_from_pattern.py computes its "
              f"own pose against the map instead).")

    with open(args.out, "w") as f:
        json.dump({"camera_matrix": camera_matrix, "dist_coeffs": dist_coeffs}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

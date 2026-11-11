"""Live visualization modes for the classifier output window.

Port of magician_grabber_annotator's mga/core/visualize_data.py:convertPolarCVMATToRGB
(and the processor list in mga/wx_annotator.py / PROCESSOR_WAYS) so the live ROS node can
show the same renders the operator sees when annotating -- DoLP, AoLP, Stokes, ... --
without importing mga (keeps the two repos independent).

Differences from the annotator:
  * Input is the packed half-resolution 4-channel frame the classifier works on
    (to_classifier_layout converts the raw shared-memory mosaic to it), in
    mvc's channel order [I90, I45, I0, I135] (mvc/core/polarization.py CH_*), not the
    full-resolution mosaic, so there is no debayer step.
  * No brightness/contrast knobs and no debug prints.
  * "Classifier" (the default) is what the window always showed: frame[:, :, :3].

Detections are overlaid by `overlay_detections`: every pixel where the classifier's
heatmap differs from the plain frame[:, :, :3] is a cross/legend pixel and is copied on
top of the chosen render.
"""

import cv2
import numpy as np

from mvc.core.polarization import CH_I90, CH_I45, CH_I0, CH_I135
from mvc.core.read_data import readPolarPNMToRGBALive


def to_classifier_layout(frame):
    """The exact (H/2, W/2, 4) [I90, I45, I0, I135] image the classifiers draw their
    heatmap on (same two steps as runSingle / EnsembleClassifierPnm / CascadeClassifierPnm
    .forward), from the raw shared-memory mosaic (or an already-packed frame)."""
    return cv2.cvtColor(readPolarPNMToRGBALive(frame), cv2.COLOR_RGBA2BGRA)

DEFAULT_MODE = "Classifier"

# Same names as the annotator's processor combo box, plus the classifier's own view.
VISUALIZATION_MODES = [
    DEFAULT_MODE,
    "PolarizationRGB1", "PolarizationRGB2", "PolarizationRGB3",
    "Polarization_0_degree", "Polarization_45_degree", "Polarization_90_degree",
    "Polarization_135_degree",
    "AoLP", "DoLP", "Normals", "Intensity", "s0", "s1", "s2", "s3",
    "AoLP (light)", "AoLP (dark)", "DoP", "DoCP", "ToP", "CoP", "RetardationMag",
    "MaxMinAvgRGB", "Sobel", "Visible",
]


def resolve_mode(name):
    """Case/space/underscore-insensitive lookup. Returns the canonical name or None."""
    def key(s):
        return "".join(ch for ch in s.lower() if ch.isalnum())
    wanted = key(name or "")
    if wanted in ("", "default", "heatmap", "none"):
        return DEFAULT_MODE
    for mode in VISUALIZATION_MODES:
        if key(mode) == wanted:
            return mode
    return None


def _gray3(u8):
    return cv2.merge([u8, u8, u8])


def _u8(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def _normalize_to_u8(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) * (255.0 / (hi - lo))).astype(np.uint8)


def _hsv(h, s, v):
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def render(frame, mode):
    """BGR uint8 render of a packed (H, W, 4) [I90, I45, I0, I135] frame
    (see to_classifier_layout) in the given visualization mode."""
    if mode == DEFAULT_MODE:
        return np.ascontiguousarray(frame[:, :, :3])

    I0   = frame[:, :, CH_I0].astype(np.float32)
    I45  = frame[:, :, CH_I45].astype(np.float32)
    I90  = frame[:, :, CH_I90].astype(np.float32)
    I135 = frame[:, :, CH_I135].astype(np.float32)

    if mode == "Sobel":
        # Annotator: Sobel per polarization channel, then PolarizationRGB1
        def sob(c):
            m = cv2.magnitude(cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3),
                              cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3))
            return cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
        I0, I45, I90 = sob(I0), sob(I45), sob(I90)
        mode = "PolarizationRGB1"

    if mode == "PolarizationRGB1":
        return cv2.merge([_u8(I0), _u8(I45), _u8(I90)])
    if mode == "PolarizationRGB2":
        return cv2.merge([_u8(I45), _u8(I90), _u8(I135)])
    if mode == "PolarizationRGB3":
        return cv2.merge([_u8((I0 + I45) / 2), _u8(I45), _u8((I90 + I135) / 2)])
    if mode == "Visible":
        return _gray3(_u8((I0 + I45 + I90 + I135) / 4))
    if mode == "Polarization_0_degree":
        return _gray3(_u8(I0))
    if mode == "Polarization_45_degree":
        return _gray3(_u8(I45))
    if mode == "Polarization_90_degree":
        return _gray3(_u8(I90))
    if mode == "Polarization_135_degree":
        return _gray3(_u8(I135))
    if mode == "MaxMinAvgRGB":
        stack = np.stack([I0, I45, I90, I135], axis=2)
        return cv2.merge([_normalize_to_u8(stack.mean(axis=2)),
                          _normalize_to_u8(stack.min(axis=2)),
                          _normalize_to_u8(stack.max(axis=2))])

    # Stokes-based renders (S3 is not observable with a 4-angle DoFP sensor -> 0)
    eps = 1e-6
    S0 = I0 + I90
    S1 = I0 - I90
    S2 = I45 - I135
    lin = np.sqrt(S1 * S1 + S2 * S2)
    dolp = np.clip(lin / (S0 + eps), 0.0, 1.0)
    aolp = 0.5 * np.arctan2(S2, S1)                     # [-pi/2, pi/2]

    if mode == "Normals":
        az = aolp + (np.pi / 2.0)
        tilt = dolp * (np.pi / 2.0)
        nx = np.sin(tilt) * np.cos(az)
        ny = np.sin(tilt) * np.sin(az)
        nz = np.cos(tilt)
        enc = lambda n: _u8((n * 0.5 + 0.5) * 255.0)
        return cv2.merge([enc(nz), enc(ny), enc(nx)])   # BGR

    dolp_u8 = (dolp * 255.0).astype(np.uint8)
    dop_u8 = dolp_u8                                    # S3 == 0 -> DoP == DoLP
    intensity_u8 = _u8((I0 + I45 + I90 + I135) / 4.0)
    hue = _u8(np.clip(((aolp + np.pi / 2.0) / np.pi) * 179.0, 0, 179))
    turbo = lambda u8: cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    signed = lambda S: (np.clip(S, -255.0, 255.0) * 0.5 + 128.0).astype(np.uint8)

    if mode == "Intensity":
        return _gray3(intensity_u8)
    if mode == "DoLP":
        return turbo(dolp_u8)
    if mode == "AoLP":
        return _hsv(hue, dolp_u8, intensity_u8)
    if mode == "s0":
        return _gray3(_u8(S0 * 0.5))
    if mode == "s1":
        return turbo(signed(S1))
    if mode == "s2":
        return turbo(signed(S2))
    if mode == "s3":
        return turbo(signed(np.zeros_like(S0)))
    if mode == "AoLP (light)":
        return _hsv(hue, dop_u8, np.full_like(hue, 255))
    if mode == "AoLP (dark)":
        return _hsv(hue, dop_u8, intensity_u8)
    if mode == "DoP":
        return turbo(dop_u8)
    if mode in ("DoCP", "ToP"):
        return turbo(np.zeros_like(dolp_u8))            # both need S3 -> 0
    if mode == "RetardationMag":
        S3_mag = np.sqrt(np.maximum(S0 * S0 - S1 * S1 - S2 * S2, 0.0))
        chi_mag = 0.5 * np.arctan2(S3_mag, lin + eps)
        return turbo((np.clip(4.0 * chi_mag / np.pi, 0.0, 1.0) * 255.0).astype(np.uint8))
    if mode == "CoP":
        # chi == 0 with S3 == 0 -> constant mid hue, as in the annotator
        hue_chi = np.full_like(hue, int(0.5 * 179))
        return _hsv(hue_chi, np.full_like(hue, 255), np.full_like(hue, 255))

    return np.ascontiguousarray(frame[:, :, :3])


def overlay_detections(frame, heatmap, mode):
    """Render `frame` in `mode` and paste the classifier's drawn crosses/legend on top."""
    if mode == DEFAULT_MODE:
        return heatmap
    frame = to_classifier_layout(frame)
    base = np.ascontiguousarray(frame[:, :, :3])
    vis = render(frame, mode)
    if heatmap is None or heatmap.shape != base.shape:
        return vis
    drawn = np.any(heatmap != base, axis=2)
    vis[drawn] = heatmap[drawn]
    return vis

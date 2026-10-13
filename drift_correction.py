"""
Drift correction for the press signal.

PROBLEM: the image slowly changes after the reference images were taken
(LED heating, the silicone relaxing after presses, ...). Every frame then
shows a weak, smooth "fake press" over large parts of the sensor.

FIX: a real press is small and sharp; drift is broad and smooth. So, per frame:
  1. estimate a smooth background from the whole sensor,
  2. mark pixels that stand out from it as "press" (plus a margin around them),
  3. re-estimate the smooth background from the remaining (untouched) pixels only,
  4. subtract it.
Untouched areas end up at ~0 even when the image has drifted; the press keeps its
shape and strength (its background is interpolated from the untouched area around it).

Used by: capture (live view), build_gain_map_flat.py, depth_calibration_sphere.py,
reconstruct_with_gain_map.py -- calibration and reconstruction must use the SAME
setting, so it is configured only here.

Limits: without a known contact area (reconstruction), a contact larger than
about 1/3 of the sensor disk's radius can be partly removed as if it were drift.
The calibration scripts pass the drawn circle (exclude=...), which is exact for
any size.
"""

import numpy as np
import cv2

# ============================== CONFIG ==============================
ENABLED = True
SMOOTH_SIGMA_PX = 60.0         # size of "smooth" (full-resolution pixels); presses are smaller than this
EXCLUDE_LOG = 0.03             # |signal - background| above this = press (log signal, ~3%)
EXCLUDE_DIFF = 3.0             # same for the "diff" signal (grey levels)
DILATE_PX = 15                 # also exclude this margin around the press (its weak edges)
MIN_REF = 30.0                 # pixels darker than this in the reference are not used as background
DOWNSCALE = 4                  # background is estimated at 1/4 resolution (fast, it's smooth anyway)
PASS1_SCALE = 4.0              # press detection uses a background this many times smoother
# ======================================================================


def _smooth_masked(sig, weight, sigma, ds):
    h, w = sig.shape
    sw, sh = max(1, w // ds), max(1, h // ds)
    num = cv2.resize((sig * weight).astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    den = cv2.resize(weight.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    s = max(sigma / ds, 0.5)
    num = cv2.GaussianBlur(num, (0, 0), s)
    den = cv2.GaussianBlur(den, (0, 0), s)
    bg = num / np.maximum(den, 1e-6)
    fill = float(np.sum(sig * weight) / max(np.sum(weight), 1e-6))
    bg = np.where(den > 1e-3, bg, fill)
    return cv2.resize(bg.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)


def background(sig, ref, signal_type, sigma=None, ds=None, valid=None, exclude=None):
    """Smooth drift background of a signal image (same shape as sig).
    exclude: optional bool mask of KNOWN contact pixels (e.g. the drawn circle in
    calibration) -- never used as background, whatever their value."""
    sigma = SMOOTH_SIGMA_PX if sigma is None else sigma
    ds = DOWNSCALE if ds is None else ds
    if valid is None:
        valid = ref >= MIN_REF
    thr = EXCLUDE_LOG if signal_type == "log" else EXCLUDE_DIFF
    w = valid.astype(np.float32)
    if w.sum() < 10:
        return np.zeros_like(sig, dtype=np.float32)
    if exclude is not None:
        w = w * (~exclude)                                            # known contact: never background
    # pass 1: a much broader background, only to FIND the press (a local one would
    # partly follow a large contact and hide it)
    bg1 = _smooth_masked(sig, w, PASS1_SCALE * sigma, ds)
    press = (np.abs(sig - bg1) > thr)
    if exclude is not None:
        press |= exclude
    press = press.astype(np.uint8)
    if DILATE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * DILATE_PX + 1, 2 * DILATE_PX + 1))
        press = cv2.dilate(press, k)
    w2 = w * (press == 0)
    if w2.sum() < 0.1 * w.sum():                                      # almost everything "pressed"
        return bg1
    return _smooth_masked(sig, w2, sigma, ds)                         # pass 2: untouched pixels only


def correct(sig, ref, signal_type, exclude=None):
    """sig minus its smooth drift background (no-op if ENABLED is False)."""
    if not ENABLED:
        return sig
    return sig - background(sig, ref, signal_type, exclude=exclude)


def circle_mask(shape, cx, cy, r_px, margin=1.3):
    """Bool mask of a drawn contact circle, enlarged by `margin` (weak edges)."""
    h, w = shape
    yy, xx = np.ogrid[0:h, 0:w]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= (margin * r_px) ** 2

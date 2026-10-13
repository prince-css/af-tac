"""
3D reconstruction with the gain map (fit_gain_model_v2.py) and the depth curve
(depth_calibration_sphere.py):

    signal      = brightness change vs. the unpressed reference
    normalized  = signal / signal_at_depth_map(pixel)      <- local gain from the map
    depth       = depth_curve_inverse(normalized)
    3D point    = undeformed shell point - depth * outward normal

The per-frame conversion is in depth_for_pixels(), so a real-time loop can
import and call it on every camera frame.

Outputs (same formats as before -> make_videos.py / plot_depth_histogram.py work):
    reconstruction_output/<name>_reconstruction.npz   points, depths, pix_u, pix_v
    reconstruction_output/<name>_depth_heatmap.png
    pcd/<name>.pcd

Use references from the SAME session/locked settings as the frames, with the
same lighting the gain map was built under.

Standalone: numpy, opencv-python, geometry.py, depth_calibration_sphere.py.
Usage:
    python reconstruct_with_gain_map.py
"""

import os
import glob
import json
import numpy as np
import cv2

from geometry import pixel_to_surface_point_batch, surface_normal_batch
from depth_calibration_sphere import (
    load_gain_map, load_reference, load_channel, compute_signal, normalize_by_map,
    curve_inverse, natural_key, check_crop_file, OUT_JSON as CURVE_PATH,
)

# ============================== CONFIG ==============================
REF_DIR = "./calibration_images/ref_test_re"
SAMPLE_DIR = "./calibration_images/sample_test_re"     # frames to reconstruct
ROI_CONFIG_PATH = "./roi_config.json"
OUTPUT_DIR = "./reconstruction_output2"
PCD_DIR = "./pcd2"
SAVE_PCD = True

ROI_SHRINK = 0.90           # skip the rim of the visible disk (it shifts when pressed)
MIN_REF_INTENSITY = 15.0
MIN_MAP_SIGNAL_FRACTION = 0.05
DOWNSAMPLE_STRIDE = 1

PRE_BLUR_SIGMA_PX = 2.0     # smooth the signal before converting (0 = off)
POST_BLUR_KERNELS = [9]     # smooth the depth map afterwards ([] = off)
NOISE_FLOOR_MM = 0.02       # depths below this -> 0
HEIGHT_SCALE = 2.0          # exaggerate depth for viewing only
HEATMAP_MAX_MM = None       # None = deepest calibrated depth
# ======================================================================


class Reconstructor:
    """Loads the calibration once; depth_for_pixels() converts one frame."""

    def __init__(self, ref_dir=REF_DIR):
        if not os.path.exists(CURVE_PATH):
            raise FileNotFoundError(f"{CURVE_PATH} not found -- run depth_calibration_sphere.py first.")
        self.curve = json.load(open(CURVE_PATH))
        gm = load_gain_map(self.curve["gain_map"])
        for k in ("signal_type", "signal_channel", "press_brightens"):
            if gm[k] != self.curve[k]:
                raise RuntimeError(f"Gain map and depth curve disagree on {k}: {gm[k]} vs {self.curve[k]}")
        self.sad = gm["signal_at_depth"]
        self.map_extrapolated = gm["extrapolated"]
        self.ch, self.stype, self.bright = self.curve["signal_channel"], self.curve["signal_type"], self.curve["press_brightens"]
        self.ref = load_reference(ref_dir, self.ch)
        self.h, self.w = self.ref.shape
        if self.sad.shape != (self.h, self.w):
            raise RuntimeError(f"Gain map {self.sad.shape} != images {(self.h, self.w)}")
        check_crop_file(self.w, self.h)

        keep = self._roi() & (self.ref >= MIN_REF_INTENSITY) & np.isfinite(self.sad)
        self._precompute(keep)

    def _roi(self):
        if not os.path.exists(ROI_CONFIG_PATH):
            return np.ones((self.h, self.w), bool)
        d = json.load(open(ROI_CONFIG_PATH))
        cx, cy = d["center"]
        yy, xx = np.mgrid[0:self.h, 0:self.w]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= (ROI_SHRINK * d["radius"]) ** 2

    def _precompute(self, keep):
        us, vs = np.meshgrid(np.arange(0, self.w, DOWNSAMPLE_STRIDE), np.arange(0, self.h, DOWNSAMPLE_STRIDE))
        self.grid_shape = us.shape
        sel = keep[vs, us]
        print(f"Precomputing geometry for {sel.sum()} pixels (stride {DOWNSAMPLE_STRIDE})...")
        names, pts, valid = pixel_to_surface_point_batch(us[sel].astype(np.float64), vs[sel].astype(np.float64))
        self.grid_valid = np.zeros(self.grid_shape, bool)
        idx = np.flatnonzero(sel.ravel())
        self.grid_valid.ravel()[idx[valid]] = True
        self.pu, self.pv = us.ravel()[idx[valid]], vs.ravel()[idx[valid]]
        self.points = pts[valid]
        self.normals = surface_normal_batch(self.points, names[valid])
        self.sad_px = self.sad[self.pv, self.pu]

    def _blur_grid(self, values, kernels):
        g = np.zeros(self.grid_shape, np.float32)
        wgt = np.zeros(self.grid_shape, np.float32)
        g[self.grid_valid] = values
        wgt[self.grid_valid] = 1.0
        for k in kernels:
            g = cv2.GaussianBlur(g, (k, k), 0)
            wgt = cv2.GaussianBlur(wgt, (k, k), 0)
        return (g / np.maximum(wgt, 1e-6))[self.grid_valid]

    def depth_for_pixels(self, frame_channel):
        """One frame (the signal channel as a float array) -> depth (mm) per
        precomputed pixel (self.pu, self.pv)."""
        sig = compute_signal(frame_channel, self.ref, self.stype, self.bright)
        if PRE_BLUR_SIGMA_PX > 0:
            sig = cv2.GaussianBlur(sig, (0, 0), PRE_BLUR_SIGMA_PX)
        n = normalize_by_map(np.clip(sig[self.pv, self.pu], 0, None), self.sad_px, MIN_MAP_SIGNAL_FRACTION)
        depth = curve_inverse(np.nan_to_num(n), self.curve)
        if POST_BLUR_KERNELS:
            depth = self._blur_grid(depth, POST_BLUR_KERNELS)
        return np.where(depth > NOISE_FLOOR_MM, depth, 0.0)

    def points_3d(self, depth):
        return self.points - (depth * HEIGHT_SCALE)[:, None] * self.normals


def save_pcd(points, path):
    n = len(points)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\n"
                "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        np.savetxt(f, points, fmt="%.5f")


def heatmap(shape, pu, pv, depth, vmax):
    img = np.zeros(shape)
    img[pv, pu] = depth
    out = cv2.applyColorMap((np.clip(img / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out[img <= 0] = 0
    return out


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if SAVE_PCD:
        os.makedirs(PCD_DIR, exist_ok=True)
    rec = Reconstructor()
    c = rec.curve
    print(f"Depth curve: {c['n_presses']} sphere press(es) at {len(c['spots'])} spot(s), calibrated to "
          f"{c['max_depth_mm']:.3f} mm; gain map: {c['gain_map']}")
    vmax = HEATMAP_MAX_MM or c["max_depth_mm"]

    paths = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*")), key=natural_key)
    print(f"\nReconstructing {len(paths)} frames from {SAMPLE_DIR}\n")
    for i, path in enumerate(paths):
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"[{i + 1}/{len(paths)}] Processing: {os.path.basename(path)}")
        try:
            ch = load_channel(path, rec.ch)
        except ValueError:
            print("  unreadable -- skipped.")
            continue
        if ch.shape != (rec.h, rec.w):
            print(f"  skipped: size {ch.shape} != reference {(rec.h, rec.w)}")
            continue
        depth = rec.depth_for_pixels(ch)
        pts = rec.points_3d(depth)
        np.savez(os.path.join(OUTPUT_DIR, f"{name}_reconstruction.npz"),
                 points=pts, depths=depth * HEIGHT_SCALE, pix_u=rec.pu, pix_v=rec.pv)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_depth_heatmap.png"),
                    heatmap((rec.h, rec.w), rec.pu, rec.pv, depth, vmax))
        if SAVE_PCD:
            save_pcd(pts, os.path.join(PCD_DIR, f"{name}.pcd"))
        beyond = int((depth > c["max_depth_mm"]).sum())
        in_contact = depth > 0
        note = ""
        if rec.map_extrapolated is not None and in_contact.any():
            frac = rec.map_extrapolated[rec.pv[in_contact], rec.pu[in_contact]].mean()
            if frac > 0.5:
                note = "  (contact mostly in the gain map's extrapolated area: less certain)"
        print(f"  max depth {depth.max():.3f} mm, {int(in_contact.sum())} px in contact"
              + (f", {beyond} px beyond the calibrated depth (extrapolated)" if beyond else "") + note)

    print(f"\nDone. Outputs in {os.path.abspath(OUTPUT_DIR)}" + (f" and {os.path.abspath(PCD_DIR)}" if SAVE_PCD else ""))


if __name__ == "__main__":
    main()

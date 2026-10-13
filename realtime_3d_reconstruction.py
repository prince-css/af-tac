"""
REAL-TIME 3D reconstruction of the tactile fingertip, using the saved calibration.

What it does, in order:
  1. Opens the camera with the saved colour gains (camera_settings.json) and the
     saved sensor ROI (roi_config.json) -- no ROI selection.
  2. EXPOSURE TUNING: adjusts the exposure until B_max (the 99.9th percentile of
     the blue channel inside the ROI -- same as the capture script's readout)
     reaches TARGET_BMAX (190). Then the exposure is LOCKED.
     -> keep the sensor UNTOUCHED until the reference is captured.
  3. REFERENCE: averages the first N_REF_FRAMES (50) frames after tuning.
  4. LIVE: every frame -> press signal -> drift correction -> gain map ->
     depth curve -> deformed 3D surface (the same points reconstruct_with_gain_map.py
     saves as .pcd), drawn with lighting in a window you can rotate with the mouse.

Mouse / keys (3D window):
    left-drag   rotate              wheel        zoom
    + / -       exaggerate depth    c            gray shading <-> colour by depth
    v           reset view          f            turn the view to face the deepest point
    b           re-take reference (sensor untouched!)
    p           save current point cloud (true depth) to ./pcd_realtime/
    q / ESC     quit

Needs in the same folder: depth_curve.json, gain_map_model.npz, roi_config.json,
camera_settings.json, geometry.py, depth_calibration_sphere.py,
drift_correction.py, capture_calibration_images_locked.py.
Only numpy + opencv (+ picamera2) -- no 3D library needed.

Usage:
    python realtime_3d_reconstruction.py
"""

import os
import json
import time
import numpy as np
import cv2
from picamera2 import Picamera2

import drift_correction
from geometry import pixel_to_surface_point_batch, surface_normal_batch
from depth_calibration_sphere import load_gain_map, normalize_by_map, curve_inverse, OUT_JSON as CURVE_PATH
from capture_calibration_images_locked import (
    PREVIEW_WIDTH, PREVIEW_HEIGHT, apply_camera_settings, to_bgr, load_roi_config,
)

# ============================== CONFIG ==============================
CAMERA_SETTINGS_PATH = "./camera_settings.json"   # colour gains + starting exposure
ROI_CONFIG_PATH = "./roi_config.json"

# --- exposure tuning ---
TARGET_BMAX = 190.0           # tune exposure until B_max reaches this
BMAX_TOLERANCE = 3.0          # accept B_max within +/- this
BMAX_PERCENTILE = 99.9        # B_max = this percentile of blue inside the ROI
MIN_EXPOSURE_US = 100
MAX_EXPOSURE_US = 33000       # above ~33 ms the frame rate drops below 30 fps
MAX_TUNING_STEPS = 40
SETTLE_FRAMES = 4             # frames to skip after each exposure change

# --- reference ---
N_REF_FRAMES = 50

# --- reconstruction ---
GRID_STRIDE = 4               # reconstruct every 4th pixel (3 = denser/slower, 5 = faster)
ROI_SHRINK = 0.90             # skip the rim of the visible disk (it shifts when pressed)
MIN_REF_INTENSITY = 15.0
MIN_MAP_SIGNAL_FRACTION = 0.05
PRE_BLUR_SIGMA_PX = 2.0       # smoothing of the signal (full-resolution pixels)
POST_BLUR_CELLS = 1.0         # smoothing of the depth map (grid cells, 0 = off)
NOISE_FLOOR_MM = 0.02         # depths below this -> 0

# --- 3D view ---
WINDOW_SIZE = 720             # displayed window size (pixels)
RENDER_SIZE = 360             # internal render resolution (upscaled to WINDOW_SIZE)
HEIGHT_SCALE = 3.0            # depth exaggeration for VIEWING (+/- keys); saved .pcd uses true depth
DEFAULT_YAW_DEG = 0.0
DEFAULT_PITCH_DEG = 145.0     # 180 = looking straight at the fingertip from outside; 90 = side view
MAX_PATCH_PX = 6              # largest screen size of one surface patch (render pixels)
BASE_COLOR = (205, 205, 205)  # BGR, gray "silicone"
AMBIENT = 0.22
DIFFUSE = 0.78
SPECULAR = 0.25
SHININESS = 24.0
LIGHT_DIR_VIEW = (0.4, 0.6, 0.7)     # direction the light TRAVELS (view coords): from upper-left-front
SHOW_DEPTH_INSET = True
MAX_DEPTH_DISPLAY_FACTOR = 2.0  # the "max depth" number (window + terminal) is multiplied by this.
                                # Display only: the 3D surface and saved .pcd files are NOT scaled.
PCD_DIR = "./pcd_realtime"
# ======================================================================


# ------------------------------ camera helpers ------------------------------

def load_start_settings():
    exposure, gain, colour = 10000, 1.0, (1.5, 1.5)
    calib_exposure = None
    if os.path.exists(CAMERA_SETTINGS_PATH):
        req = json.load(open(CAMERA_SETTINGS_PATH))["requested"]
        exposure = int(req["ExposureTime"])
        calib_exposure = exposure
        gain = float(req["AnalogueGain"])
        colour = tuple(req["ColourGains"])
    else:
        print(f"{CAMERA_SETTINGS_PATH} not found -- starting from exposure {exposure} us, colour gains {colour}")
    return exposure, gain, colour, calib_exposure


def roi_masks(h, w, roi):
    cx, cy, r = roi
    yy, xx = np.mgrid[0:h, 0:w]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return d2 <= r ** 2, d2 <= (ROI_SHRINK * r) ** 2


def measure_bmax(frame_bgr, bmax_idx):
    return float(np.percentile(frame_bgr[..., 0].ravel()[bmax_idx], BMAX_PERCENTILE))


def tune_exposure(picam2, exposure, gain, colour, bmax_idx, win):
    """Proportional search: exposure *= TARGET / B_max until B_max is within tolerance."""
    best = (np.inf, exposure, 0.0)
    for step in range(MAX_TUNING_STEPS):
        apply_camera_settings(picam2, exposure, gain, colour)
        for _ in range(SETTLE_FRAMES):
            picam2.capture_array()
        vals = []
        for _ in range(3):
            frame = to_bgr(picam2.capture_array())
            vals.append(measure_bmax(frame, bmax_idx))
        bmax = float(np.median(vals))
        err = abs(bmax - TARGET_BMAX)
        if err < best[0]:
            best = (err, exposure, bmax)
        disp = frame.copy()
        cv2.putText(disp, f"Tuning exposure: {exposure} us   B_max {bmax:.0f} -> target {TARGET_BMAX:.0f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp, "Do NOT touch the sensor", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        cv2.waitKey(1)
        print(f"  step {step + 1}: exposure {exposure:6d} us -> B_max {bmax:6.1f}")
        if err <= BMAX_TOLERANCE:
            return exposure, bmax
        ratio = np.clip(TARGET_BMAX / max(bmax, 1.0), 0.6, 1.6)   # limited step (clipping at 255 hides the truth)
        new_exp = int(np.clip(exposure * ratio, MIN_EXPOSURE_US, MAX_EXPOSURE_US))
        if new_exp == exposure:                                     # at a limit -- can't move further
            break
        exposure = new_exp
    _, exposure, bmax = best
    apply_camera_settings(picam2, exposure, gain, colour)
    for _ in range(SETTLE_FRAMES):
        picam2.capture_array()
    print(f"  using the closest value found: exposure {exposure} us, B_max {bmax:.1f}")
    return exposure, bmax


def capture_reference(picam2, n, win):
    acc = None
    for i in range(n):
        frame = to_bgr(picam2.capture_array())
        b = frame[..., 0].astype(np.float32)
        acc = b if acc is None else acc + b
        disp = frame.copy()
        cv2.putText(disp, f"Capturing reference {i + 1}/{n} -- do NOT touch the sensor",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, disp)
        cv2.waitKey(1)
    return acc / n


# ------------------------------ fast reconstruction ------------------------------

class FastReconstructor:
    """Same pipeline as reconstruct_with_gain_map.py (signal -> drift correction ->
    gain map -> depth curve -> 3D), computed on a GRID_STRIDE-downsampled grid so
    it runs in real time."""

    def __init__(self, curve, gain_map, shape, roi_inner):
        self.curve = curve
        self.bright = curve["press_brightens"]
        self.stype = curve["signal_type"]
        h, w = shape
        s = GRID_STRIDE
        self.gw, self.gh = w // s, h // s
        gv, gu = np.mgrid[0:self.gh, 0:self.gw]                 # full-res pixel at each cell's centre
        self.cu = np.clip(gu * s + s // 2, 0, w - 1)
        self.cv = np.clip(gv * s + s // 2, 0, h - 1)
        self.roi_cell = roi_inner[self.cv, self.cu]
        self.sad_cell = gain_map["signal_at_depth"][self.cv, self.cu]
        self.ref_small = None

    def _small(self, img):
        return cv2.resize(img.astype(np.float32), (self.gw, self.gh), interpolation=cv2.INTER_AREA)

    def set_reference(self, ref_full):
        self.ref_small = self._small(ref_full)
        keep = self.roi_cell & (self.ref_small >= MIN_REF_INTENSITY) & np.isfinite(self.sad_cell)
        flat = np.flatnonzero(keep.ravel())                      # row-major order
        names, pts, valid = pixel_to_surface_point_batch(self.cu.ravel()[flat].astype(np.float64),
                                                         self.cv.ravel()[flat].astype(np.float64))
        flat = flat[valid]
        self.valid = np.zeros((self.gh, self.gw), bool)
        self.valid.ravel()[flat] = True
        self.gi, self.gj = np.unravel_index(flat, (self.gh, self.gw))
        self.points0 = pts[valid]
        self.normals0 = surface_normal_batch(pts[valid], names[valid])
        self.sad_px = self.sad_cell[self.gi, self.gj]
        self.valid_f = self.valid.astype(np.float32)
        print(f"Reconstructing {len(self.gi)} surface points per frame (grid stride {GRID_STRIDE})")

    def depth(self, frame_b):
        small = self._small(frame_b)
        sign = 1.0 if self.bright else -1.0
        if self.stype == "log":
            sig = sign * (np.log(np.maximum(small, 1.0)) - np.log(np.maximum(self.ref_small, 1.0)))
        else:
            sig = sign * (small - self.ref_small)
        if PRE_BLUR_SIGMA_PX > 0:
            sig = cv2.GaussianBlur(sig, (0, 0), max(PRE_BLUR_SIGMA_PX / GRID_STRIDE, 0.3))
        if drift_correction.ENABLED:
            sig = sig - drift_correction.background(
                sig, None, self.stype, sigma=drift_correction.SMOOTH_SIGMA_PX / GRID_STRIDE, ds=1,
                valid=self.valid)
        n = normalize_by_map(np.clip(sig[self.gi, self.gj], 0, None), self.sad_px, MIN_MAP_SIGNAL_FRACTION)
        d = curve_inverse(np.nan_to_num(n), self.curve)
        if POST_BLUR_CELLS > 0:
            g = np.zeros((self.gh, self.gw), np.float32)
            g[self.gi, self.gj] = d
            num = cv2.GaussianBlur(g * self.valid_f, (0, 0), POST_BLUR_CELLS)
            den = cv2.GaussianBlur(self.valid_f, (0, 0), POST_BLUR_CELLS)
            d = (num / np.maximum(den, 1e-6))[self.gi, self.gj]
        return np.where(d > NOISE_FLOOR_MM, d, 0.0)

    def points(self, depth, scale=1.0):
        return self.points0 - (depth * scale)[:, None] * self.normals0


# ------------------------------ shaded 3D renderer ------------------------------

class SurfaceRenderer:
    """Orthographic mesh renderer: the reconstructed points form a grid, and every
    grid cell (4 neighbouring points) is drawn as one filled patch -> a closed
    surface without gaps. Per-point normals from the deformed surface, two-sided
    Blinn-Phong lighting, z-buffer, smooth upscaling."""

    def __init__(self, rec):
        self.rec = rec
        self.center = rec.points0.mean(axis=0)
        self.extent = float(np.ptp(rec.points0, axis=0).max())
        self.reset_view()
        # quads: grid cells whose 4 corners are all reconstructed points
        idx = -np.ones((rec.gh, rec.gw), np.int64)
        idx[rec.gi, rec.gj] = np.arange(len(rec.gi))
        a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[:-1, 1:], idx[1:, 1:]
        ok = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
        self.quads = np.stack([a[ok], b[ok], c[ok], d[ok]], axis=1)
        Ld = np.asarray(LIGHT_DIR_VIEW, float)
        self.L = -Ld / np.linalg.norm(Ld)                   # direction TO the light
        self.V = np.array([0.0, 0.0, -1.0])                 # direction TO the viewer (view looks along +z)
        Hh = self.L + self.V
        self.H = Hh / np.linalg.norm(Hh)
        self.grid = np.full((rec.gh, rec.gw, 3), np.nan, np.float32)
        self.color_by_depth = False
        self.drag = None

    def reset_view(self):
        self.yaw = np.radians(DEFAULT_YAW_DEG)
        self.pitch = np.radians(DEFAULT_PITCH_DEG)
        self.zoom = 0.85

    def rotation(self):
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])   # orbit around the finger axis
        Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])   # tilt
        return Rx @ Rz

    def face_point(self, p):
        """Orbit so that point p (on the shell) faces the viewer."""
        v = p - self.center
        self.yaw = float(np.arctan2(v[0], v[1]))

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (x, y, self.yaw, self.pitch)
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            x0, y0, yaw0, pitch0 = self.drag
            self.yaw = yaw0 + (x - x0) * 0.01
            self.pitch = float(np.clip(pitch0 + (y - y0) * 0.01, -np.pi, np.pi))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self.zoom = float(np.clip(self.zoom * (1.1 if flags > 0 else 1 / 1.1), 0.2, 8.0))

    def normals(self, P):
        rec = self.rec
        g = self.grid
        g[:] = np.nan
        g[rec.gi, rec.gj] = P
        n = np.cross(np.gradient(g, axis=1), np.gradient(g, axis=0))[rec.gi, rec.gj]
        norm = np.linalg.norm(n, axis=1)
        bad = ~np.isfinite(norm) | (norm < 1e-9)
        n[bad] = rec.normals0[bad]
        norm[bad] = 1.0
        n = n / norm[:, None]
        n[np.sum(n * rec.normals0, axis=1) < 0] *= -1          # keep them pointing outward
        return n

    def render(self, P, depth, vmax):
        S = RENDER_SIZE
        R = self.rotation()
        Q = (P - self.center) @ R.T
        N = self.normals(P) @ R.T
        scale = self.zoom * S / max(self.extent, 1e-6)
        u = Q[:, 0] * scale + S / 2
        v = Q[:, 1] * scale + S / 2

        N = np.where((N @ self.V < 0)[:, None], -N, N)         # two-sided lighting
        diff = np.clip(N @ self.L, 0, 1)
        spec = np.clip(N @ self.H, 0, 1) ** SHININESS
        if self.color_by_depth:
            idx = np.clip(depth / max(vmax, 1e-6) * 255, 0, 255).astype(np.uint8)
            base = cv2.applyColorMap(idx.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3).astype(np.float32)
            base = np.where((depth > 0)[:, None], base, np.array(BASE_COLOR, np.float32))
        else:
            base = np.broadcast_to(np.array(BASE_COLOR, np.float32), (len(P), 3))
        col = base * (AMBIENT + DIFFUSE * diff)[:, None] + 255 * SPECULAR * spec[:, None]

        # one filled patch per grid cell: covers the screen box of its 4 corners
        q = self.quads
        qu, qv = u[q], v[q]
        u0 = np.floor(qu.min(1)).astype(np.int32)
        v0 = np.floor(qv.min(1)).astype(np.int32)
        wq = np.minimum(np.ceil(qu.max(1)).astype(np.int32) - u0, MAX_PATCH_PX)
        hq = np.minimum(np.ceil(qv.max(1)).astype(np.int32) - v0, MAX_PATCH_PX)
        cq = np.clip(col[q].mean(1), 0, 255).astype(np.uint8)
        # sort the PATCHES far -> near, then expand each into its pixels in that order:
        # later (nearer) patches overwrite earlier ones = z-buffer without sorting pixels
        order = np.argsort(-Q[q, 2].mean(1))
        u0, v0, wq, hq, cq = u0[order], v0[order], wq[order] + 1, hq[order] + 1, cq[order]
        npx = wq * hq
        start = np.cumsum(npx) - npx
        k = np.arange(int(npx.sum())) - np.repeat(start, npx)
        rep = np.repeat(np.arange(len(npx)), npx)
        uu = u0[rep] + k % wq[rep]
        vv = v0[rep] + k // wq[rep]
        ok = (uu >= 0) & (uu < S) & (vv >= 0) & (vv < S)
        img = np.zeros((S * S, 3), np.uint8)
        img[vv[ok] * S + uu[ok]] = cq[rep[ok]]
        img = img.reshape(S, S, 3)
        img = cv2.resize(img, (WINDOW_SIZE, WINDOW_SIZE), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(img, (3, 3), 0)


def depth_inset(rec, depth, vmax, size=180):
    g = np.zeros((rec.gh, rec.gw), np.float32)
    g[rec.gi, rec.gj] = depth
    img = cv2.applyColorMap((np.clip(g / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~rec.valid] = 0
    ys, xs = np.nonzero(rec.valid)
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh = max(1, int(size * img.shape[0] / img.shape[1]))
    return cv2.resize(img, (size, hh), interpolation=cv2.INTER_NEAREST)


def save_pcd(points, path):
    n = len(points)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\n"
                "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA ascii\n")
        np.savetxt(f, points, fmt="%.5f")


# ------------------------------ main ------------------------------

def main():
    if os.path.exists("./calibration_images/roi_crop_info.json"):
        raise RuntimeError("./calibration_images/roi_crop_info.json exists -- move/rename it (it is for cropped images).")
    roi = load_roi_config(ROI_CONFIG_PATH)
    if roi is None:
        raise FileNotFoundError(f"{ROI_CONFIG_PATH} missing or invalid -- it must contain the saved sensor ROI.")
    curve = json.load(open(CURVE_PATH))
    gm = load_gain_map(curve["gain_map"])
    for k in ("signal_type", "signal_channel", "press_brightens"):
        if gm[k] != curve[k]:
            raise RuntimeError(f"Gain map and depth curve disagree on {k}")
    if curve["signal_channel"] != "B":
        raise RuntimeError("This real-time script expects a calibration on the blue channel.")
    h, w = gm["signal_at_depth"].shape
    if (w, h) != (PREVIEW_WIDTH, PREVIEW_HEIGHT):
        raise RuntimeError(f"Calibration is {w}x{h}, but the camera is set to {PREVIEW_WIDTH}x{PREVIEW_HEIGHT}.")

    exposure, gain, colour, calib_exposure = load_start_settings()
    roi_full, roi_inner = roi_masks(h, w, roi)
    bmax_idx = np.flatnonzero(roi_full.ravel())[::2]           # every 2nd ROI pixel is plenty for B_max

    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}))
    picam2.start()
    cam_win = "camera"
    cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)

    try:
        print(f"ROI (from {ROI_CONFIG_PATH}): center=({roi[0]},{roi[1]}), radius={roi[2]:.0f}px")
        print(f"Tuning exposure for B_max = {TARGET_BMAX:.0f} (sensor untouched)...")
        exposure, bmax = tune_exposure(picam2, exposure, gain, colour, bmax_idx, cam_win)
        print(f"Exposure LOCKED at {exposure} us (B_max {bmax:.1f})")
        if calib_exposure:
            print(f"  calibration exposure: {calib_exposure} us -> ratio {exposure / calib_exposure:.2f} "
                  f"(closer to 1.0 = closer to the calibrated conditions)")

        print(f"Capturing {N_REF_FRAMES} reference frames (sensor untouched)...")
        rec = FastReconstructor(curve, gm, (h, w), roi_inner)
        rec.set_reference(capture_reference(picam2, N_REF_FRAMES, cam_win))
        cv2.destroyWindow(cam_win)

        ren = SurfaceRenderer(rec)
        win = "real-time 3D  (drag rotate | wheel zoom | +/- depth x | c colour | v view | f face press | b ref | p save | q quit)"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, ren.on_mouse)
        vmax = float(curve.get("max_depth_mm", 0.6))
        height_scale = HEIGHT_SCALE
        t_last, fps, n_saved = time.time(), 0.0, 0

        while True:
            frame = to_bgr(picam2.capture_array())
            depth = rec.depth(frame[..., 0])
            img = ren.render(rec.points(depth, height_scale), depth, vmax)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_last, 1e-6)
            t_last = now
            peak = float(depth.max()) if len(depth) else 0.0
            cv2.putText(img, f"max depth {peak * MAX_DEPTH_DISPLAY_FACTOR:.3f} mm   view depth x{height_scale:.1f}   {fps:.1f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(img, f"exposure {exposure} us", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            if SHOW_DEPTH_INSET:
                inset = depth_inset(rec, depth, vmax)
                ih, iw = inset.shape[:2]
                img[WINDOW_SIZE - ih - 10:WINDOW_SIZE - 10, WINDOW_SIZE - iw - 10:WINDOW_SIZE - 10] = inset
            cv2.imshow(win, img)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key in (ord('+'), ord('=')):
                height_scale = min(height_scale * 1.25, 50.0)
            elif key in (ord('-'), ord('_')):
                height_scale = max(height_scale / 1.25, 1.0)
            elif key == ord('c'):
                ren.color_by_depth = not ren.color_by_depth
            elif key == ord('v'):
                ren.reset_view()
            elif key == ord('f') and peak > 0:
                ren.face_point(rec.points0[int(np.argmax(depth))])
            elif key == ord('b'):
                print(f"Re-taking the reference ({N_REF_FRAMES} frames) -- don't touch the sensor...")
                cv2.namedWindow(cam_win, cv2.WINDOW_NORMAL)
                rec.set_reference(capture_reference(picam2, N_REF_FRAMES, cam_win))
                cv2.destroyWindow(cam_win)
            elif key == ord('p'):
                os.makedirs(PCD_DIR, exist_ok=True)
                path = os.path.join(PCD_DIR, f"frame_{n_saved:04d}.pcd")
                save_pcd(rec.points(depth, 1.0), path)              # TRUE depth, not exaggerated
                n_saved += 1
                print(f"Saved {path} ({len(depth)} points, max depth {peak * MAX_DEPTH_DISPLAY_FACTOR:.3f} mm)")
    finally:
        cv2.destroyAllWindows()
        picam2.stop()


if __name__ == "__main__":
    main()

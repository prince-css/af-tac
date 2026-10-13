"""
Two-phase image capture (reference, then sample) with LOCKED camera
settings -- needed for intensity calibration, since auto-exposure/auto
white balance would otherwise change the global brightness between
reference and sample frames and corrupt every diff.

Changes vs capture_calibration_images.py:
  1. Auto-exposure and auto white balance are OFF; exposure, gain, and
     colour gains are fixed (CONFIG below).
  2. Live clipping readout: shows the % of pixels at 255 (in the preview
     ROI) so you can pick an exposure where nothing saturates.
  3. Before the first reference is saved, you can tune exposure live with
     [ and ]. Saving the first reference LOCKS it -- refs and samples are
     then guaranteed to share identical settings.
  4. The settings actually used are saved to camera_settings.json, so
     later scripts/live capture can reuse exactly the same values.
  5. During the sample phase, a second window shows the live diff vs the
     averaged references (press direction only -- see PRESS_BRIGHTENS -- same convention as the rest of
     the pipeline) with its max value, so you can press to a similar
     depth at each location.

PREVIEW ROI: loaded from roi_config.json ({"center": [cx, cy], "radius": r})
and shown immediately at startup. Press ENTER to keep it, or redraw/adjust
it -- if you change it, roi_config.json is updated for next time.
Saved images are still FULL, uncropped frames (the ROI only crops the
preview), so geometry.py's pixel coordinates stay valid.

KEYS:
  [ / ]  decrease / increase exposure (only before the first ref is saved)
  ENTER  save current frame
  N      move from reference phase to sample phase
  S      (sample phase) START recording every frame into SAMPLE_DIR
  E      END recording (frames are named rec01_00000.png, rec01_00001.png, ...;
         timestamps go to rec01_timestamps.csv)
  C      channel-order check: hold a RED object in view, press C;
         R should be the largest printed mean (else flip FRAME_IS_BGR)
  Q      quit

Usage:
    python capture_calibration_images_locked.py
"""

import os
import json
import time
import queue
import threading
import numpy as np
import cv2
from picamera2 import Picamera2
import drift_correction

# ============================== CONFIG ==============================
PREVIEW_WIDTH = 1280
PREVIEW_HEIGHT = 720

REF_DIR = "./calibration_images/ref_test_vid"
SAMPLE_DIR = "./calibration_images/sample_test_vid"
SETTINGS_OUT_PATH = "./calibration_images/camera_settings.json"

# Reuse the settings saved by an earlier run (e.g. the gain-map session) so
# sphere / test captures use EXACTLY the same exposure, gain and colour gains.
#   None  -> tune exposure live with [ ] and save it to SETTINGS_OUT_PATH (first run)
#   path  -> load that file, lock immediately ([ ] disabled), never overwrite it
LOAD_SETTINGS_PATH = None    # e.g. "./calibration_images/camera_settings.json"

# Recording (S/E keys): frames are written by a background thread so the camera
# loop isn't slowed down. PNG level 1 = fast, slightly bigger files (still lossless).
RECORD_PNG_COMPRESSION = 1
RECORD_QUEUE_MAX = 300       # frames waiting to be written; if full, frames are dropped (reported)
RECORD_SHOW_DIFF = True      # keep updating the diff window while recording (False = higher fps)
ROI_CONFIG_PATH = os.path.expanduser("./roi_config.json")  # {"center": [cx, cy], "radius": r}

# Starting camera settings (exposure is tunable live before locking)
EXPOSURE_TIME_US = 11000      # microseconds
ANALOGUE_GAIN = 1.0
COLOUR_GAINS = (1.5, 1.5)    # (red, blue) -- fixed white balance. Raising the BLUE gain
                             # boosts B relative to R/G -- useful since B carries your signal

EXPOSURE_STEP = 1.05         # multiplier per [ or ] keypress
CLIP_WARN_PERCENT = 0.1      # show red warning above this % of clipped pixels

DIFF_DISPLAY_GAIN = 5.0      # display-only gain, used only when LIVE_VIEW = "raw_diff"

# Live press view (sample phase). Display only -- saved images are never changed.
#   "log"      : log(frame / reference), smoothed, noise floor removed, colour-mapped
#                (same signal the calibration uses; recommended)
#   "raw_diff" : the old unsmoothed brightness difference (noisy)
LIVE_VIEW = "log"
LIVE_DOWNSCALE = 2           # compute the view at 1/2 resolution (faster on the Pi, smoother)
LIVE_BLUR_SIGMA_PX = 1.0     # Gaussian smoothing, in downscaled pixels
LIVE_NOISE_FLOOR = 0.02      # log units (~2% brightening); weaker changes are shown as 0
LIVE_DISPLAY_MAX = 0.60      # log units shown at full colour (fixed so presses are comparable); None = auto
LIVE_MIN_REF = 30.0          # ignore pixels this dark in the reference (log ratio is jumpy there)
LIVE_ROI_SHRINK = 0.90       # hide the rim of the disk (it shifts when pressed -> false signal)

# Direction of the press signal. True = pressing makes the image BRIGHTER
# (signal = current - reference); False = pressing makes it DARKER
# (signal = reference - current). Changes in the other direction are
# treated as noise and clipped to 0.
PRESS_BRIGHTENS = True

# Which image channel carries the press signal. Your diagnostic showed the
# signal is almost entirely in BLUE (red layer: G fully absorbed, R ~no
# change), and grayscale weights blue at only 11.4% -- so use "B".
# Options: "B", "G", "R", "mean" (of 3 channels), "gray" (OpenCV weighted).
SIGNAL_CHANNEL = "B"

# picamera2's "RGB888" format is stored in [B, G, R] order despite its
# name, i.e. it is already in OpenCV's BGR order. If so, converting it
# with COLOR_RGB2BGR would SWAP red and blue in every saved image.
# Verify on your hardware with the C key (see KEYS above); set False only
# if the check shows red and blue are swapped.
FRAME_IS_BGR = True
# ======================================================================


_CHANNEL_INDEX = {"B": 0, "G": 1, "R": 2}


def signal_image(img_bgr):
    """The image as a single float channel, per SIGNAL_CHANNEL."""
    if SIGNAL_CHANNEL in _CHANNEL_INDEX:
        return img_bgr[..., _CHANNEL_INDEX[SIGNAL_CHANNEL]].astype(np.float64)
    if SIGNAL_CHANNEL == "mean":
        return img_bgr.astype(np.float64).mean(axis=2)
    if SIGNAL_CHANNEL == "gray":
        return cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    raise ValueError(f"Unknown SIGNAL_CHANNEL: {SIGNAL_CHANNEL}")


def press_signal(reference, current):
    """Press-direction change, clipped at 0 (see PRESS_BRIGHTENS)."""
    d = (current - reference) if PRESS_BRIGHTENS else (reference - current)
    return np.clip(d, 0, None)


def to_bgr(frame):
    """Returns the camera frame in OpenCV BGR order, per FRAME_IS_BGR."""
    return frame if FRAME_IS_BGR else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def report_channel_means(img_bgr, roi):
    """Prints per-channel means of the central region (or preview ROI),
    as this script interprets them. Hold a RED object in view: R should
    be clearly the largest. If B is largest instead, flip FRAME_IS_BGR."""
    region = crop_for_preview(img_bgr, roi)
    h, w = region.shape[:2]
    center = region[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    b, g, r = center.reshape(-1, 3).mean(axis=0)
    print(f"Channel means (center region):  R={r:.1f}  G={g:.1f}  B={b:.1f}")
    if r > max(g, b):
        print("  -> R is largest. If you're holding a RED object, the channel order is correct.")
    elif b > max(r, g):
        print("  -> B is largest. If you're holding a RED object, red/blue are SWAPPED: "
              "flip FRAME_IS_BGR in the config.")
    else:
        print("  -> G is largest; hold a clearly red object to check the order.")


def load_roi_config(path):
    """Returns (cx, cy, radius) from roi_config.json, or None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        cx, cy = data["center"]
        return int(cx), int(cy), float(data["radius"])
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        print(f"Could not read ROI from {path} ({e}) -- you'll need to draw one.")
        return None


def save_roi_config(path, roi):
    """Writes the ROI back, keeping any other keys already in the file."""
    data = {}
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except (ValueError, json.JSONDecodeError):
            data = {}
    cx, cy, radius = roi
    data["center"] = [int(cx), int(cy)]
    data["radius"] = int(round(radius))
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Updated {path}: {data}")


def select_circular_roi(picam2, initial_roi=None,
                        window_name="preview ROI -- ENTER=accept, click=redraw, ESC=full frame"):
    """
    Shows the live preview with the circular ROI. If initial_roi is given
    (loaded from roi_config.json) it is drawn immediately -- press ENTER
    to keep it as-is.

    To change it: click a new CENTER then a point on the EDGE (or press R
    to clear first), and/or fine-tune with w/a/s/d (move), m/n (resize).
    ENTER/q = accept, ESC = use the full frame this session (file untouched).

    The ROI only crops the PREVIEW; saved images stay full-frame.
    Returns (roi, changed), where roi is (cx, cy, radius) or None.
    """
    state = {"center": None, "radius": None, "awaiting_edge": False}
    if initial_roi is not None:
        cx, cy, r = initial_roi
        state["center"] = (int(cx), int(cy))
        state["radius"] = float(r)

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if state["awaiting_edge"] and state["center"] is not None:
            cx, cy = state["center"]
            state["radius"] = float(np.sqrt((x - cx) ** 2 + (y - cy) ** 2))
            state["awaiting_edge"] = False
        else:
            # first click (or a click on a finished circle) starts a new circle
            state["center"] = (x, y)
            state["radius"] = None
            state["awaiting_edge"] = True

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    if initial_roi is not None:
        print(f"Loaded preview ROI from file: center={state['center']}, radius={state['radius']:.0f}px")
        print("  ENTER = keep it.  To change: click new CENTER then EDGE, or w/a/s/d (move),")
        print("  m/n (resize), R (clear).  ESC = full frame this session.\n")
    else:
        print("No saved ROI -- click CENTER, then a point on the EDGE.")
        print("  w/a/s/d=move, m/n=resize, R=clear, ENTER/q=accept, ESC=full frame.\n")

    accepted = None
    while True:
        vis = to_bgr(picam2.capture_array()).copy()
        if state["center"] is not None:
            cv2.drawMarker(vis, state["center"], (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
        if state["center"] is not None and state["radius"] is not None:
            cv2.circle(vis, state["center"], int(state["radius"]), (0, 255, 0), 2)
            current = (state["center"][0], state["center"][1], state["radius"])
            changed = (initial_roi is None or
                       (int(current[0]), int(current[1]), int(round(current[2]))) !=
                       (int(initial_roi[0]), int(initial_roi[1]), int(round(initial_roi[2]))))
            status = "MODIFIED -- ENTER saves to file" if changed else "loaded from file -- ENTER keeps it"
            cv2.putText(vis, f"c=({state['center'][0]},{state['center'][1]}) r={state['radius']:.0f}  {status}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window_name, vis)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            break
        if key in (13, ord('q')):  # ENTER or q
            if state["center"] is not None and state["radius"] is not None:
                accepted = (state["center"][0], state["center"][1], state["radius"])
            break
        if key == ord('r'):
            state["center"] = None
            state["radius"] = None
            state["awaiting_edge"] = False
        if state["center"] is not None and state["radius"] is not None:
            cx, cy = state["center"]
            if key == ord('w'):
                state["center"] = (cx, cy - 1)
            elif key == ord('s'):
                state["center"] = (cx, cy + 1)
            elif key == ord('a'):
                state["center"] = (cx - 1, cy)
            elif key == ord('d'):
                state["center"] = (cx + 1, cy)
            elif key == ord('m'):
                state["radius"] += 1
            elif key == ord('n'):
                state["radius"] = max(1, state["radius"] - 1)

    cv2.destroyWindow(window_name)

    if accepted is None:
        return None, False
    changed = (initial_roi is None or
               (int(accepted[0]), int(accepted[1]), int(round(accepted[2]))) !=
               (int(initial_roi[0]), int(initial_roi[1]), int(round(initial_roi[2]))))
    return accepted, changed


def crop_for_preview(img, roi):
    if roi is None:
        return img
    cx, cy, radius = roi
    h, w = img.shape[:2]
    u_min = max(0, int(cx - radius))
    u_max = min(w, int(cx + radius) + 1)
    v_min = max(0, int(cy - radius))
    v_max = min(h, int(cy + radius) + 1)
    return img[v_min:v_max, u_min:u_max]


def clipped_percent(img_bgr):
    """% of pixels where the SIGNAL channel is saturated (>= 254). Clipping in
    other channels (e.g. red, which carries no signal) is ignored on purpose,
    so it doesn't force exposure lower than the signal channel needs."""
    if SIGNAL_CHANNEL in _CHANNEL_INDEX:
        clipped = img_bgr[..., _CHANNEL_INDEX[SIGNAL_CHANNEL]] >= 254
    else:
        clipped = (img_bgr >= 254).any(axis=2)
    return 100.0 * clipped.mean()


def apply_camera_settings(picam2, exposure_us, gain, colour_gains):
    picam2.set_controls({
        "AeEnable": False,
        "ExposureTime": int(exposure_us),
        "AnalogueGain": float(gain),
        "AwbEnable": False,
        "ColourGains": tuple(colour_gains),
    })


def save_camera_settings(picam2, exposure_us, gain, colour_gains, path):
    """Saves requested settings plus what the camera actually applied
    (from frame metadata), so later scripts can reproduce them."""
    md = picam2.capture_metadata()
    settings = {
        "requested": {
            "ExposureTime": int(exposure_us),
            "AnalogueGain": float(gain),
            "ColourGains": list(colour_gains),
        },
        "actual": {
            "ExposureTime": md.get("ExposureTime"),
            "AnalogueGain": md.get("AnalogueGain"),
            "ColourGains": list(md["ColourGains"]) if md.get("ColourGains") else None,
        },
    }
    with open(path, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Locked and saved camera settings to {path}: {settings['actual']}")


class LivePressView:
    """Smoothed, colour-mapped view of the press signal for the sample phase.
    The reference part is precomputed once; per frame: crop -> downscale -> log
    ratio -> blur -> noise floor -> colour map."""

    def __init__(self, baseline, roi):
        h, w = baseline.shape
        if roi is not None:
            cx, cy, r = roi
            self.box = (max(0, int(cx - r)), min(w, int(cx + r) + 1),
                        max(0, int(cy - r)), min(h, int(cy + r) + 1))
        else:
            self.box = (0, w, 0, h)
        u0, u1, v0, v1 = self.box
        self.size = ((u1 - u0) // LIVE_DOWNSCALE, (v1 - v0) // LIVE_DOWNSCALE)   # (w, h) for cv2.resize
        base = self._prep(baseline)
        self.log_base = np.log(np.maximum(base, 1.0))
        hh, ww = base.shape
        mask = base >= LIVE_MIN_REF
        if roi is not None:
            yy, xx = np.mgrid[0:hh, 0:ww]
            ccx, ccy = (cx - u0) / LIVE_DOWNSCALE, (cy - v0) / LIVE_DOWNSCALE
            mask &= (xx - ccx) ** 2 + (yy - ccy) ** 2 <= (LIVE_ROI_SHRINK * r / LIVE_DOWNSCALE) ** 2
        self.mask = mask

    def _prep(self, img):
        u0, u1, v0, v1 = self.box
        return cv2.resize(img[v0:v1, u0:u1].astype(np.float32), self.size, interpolation=cv2.INTER_AREA)

    def render(self, frame_sig):
        """Returns (BGR display image, peak signal, median signal = drift indicator)."""
        sig = np.log(np.maximum(self._prep(frame_sig), 1.0)) - self.log_base
        if not PRESS_BRIGHTENS:
            sig = -sig
        if LIVE_BLUR_SIGMA_PX > 0:
            sig = cv2.GaussianBlur(sig, (0, 0), LIVE_BLUR_SIGMA_PX)
        drift = float(np.median(sig[self.mask]))           # before correction (how much it has drifted)
        if drift_correction.ENABLED:                       # same correction as calibration/reconstruction
            sig = sig - drift_correction.background(
                sig, None, "log", sigma=drift_correction.SMOOTH_SIGMA_PX / LIVE_DOWNSCALE,
                ds=2, valid=self.mask)
        shown = np.where(self.mask, np.clip(sig - LIVE_NOISE_FLOOR, 0, None), 0)
        peak = float(shown.max()) + (LIVE_NOISE_FLOOR if shown.max() > 0 else 0.0)
        vmax = LIVE_DISPLAY_MAX or max(peak, 0.05)
        img8 = np.clip(255 * shown / max(vmax - LIVE_NOISE_FLOOR, 1e-6), 0, 255).astype(np.uint8)
        disp = cv2.applyColorMap(img8, cv2.COLORMAP_INFERNO)
        disp[~self.mask] = 0
        disp = cv2.resize(disp, (self.size[0] * LIVE_DOWNSCALE, self.size[1] * LIVE_DOWNSCALE),
                          interpolation=cv2.INTER_LINEAR)
        return disp, peak, drift


class FrameWriter:
    """Writes frames to disk in a background thread (keeps the camera loop fast)."""

    def __init__(self, max_queue=RECORD_QUEUE_MAX):
        self.q = queue.Queue(maxsize=max_queue)
        self.written = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            path, img = item
            cv2.imwrite(path, img, [cv2.IMWRITE_PNG_COMPRESSION, RECORD_PNG_COMPRESSION])
            self.written += 1
            self.q.task_done()

    def put(self, path, img):
        try:
            self.q.put_nowait((path, img.copy()))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def close(self):
        """Waits until every queued frame is on disk."""
        self.q.put(None)
        self.thread.join()


def next_recording_index(folder):
    existing = [f for f in os.listdir(folder) if f.startswith("rec") and f.endswith("_timestamps.csv")]
    nums = [int(f[3:5]) for f in existing if f[3:5].isdigit()]
    return max(nums, default=0) + 1


def main():
    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (PREVIEW_WIDTH, PREVIEW_HEIGHT), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    exposure_us, gain, colour_gains = EXPOSURE_TIME_US, ANALOGUE_GAIN, tuple(COLOUR_GAINS)
    settings_locked = False
    if LOAD_SETTINGS_PATH:
        with open(LOAD_SETTINGS_PATH) as f:
            req = json.load(f)["requested"]
        exposure_us = int(req["ExposureTime"])
        gain = float(req["AnalogueGain"])
        colour_gains = tuple(req["ColourGains"])
        settings_locked = True
        print(f"Loaded and LOCKED settings from {LOAD_SETTINGS_PATH}: exposure {exposure_us} us, "
              f"gain {gain}, colour gains {colour_gains}")
    apply_camera_settings(picam2, exposure_us, gain, colour_gains)
    for _ in range(10):  # let the new controls take effect
        picam2.capture_array()
    if LOAD_SETTINGS_PATH:
        # keep a record of what this run actually used, NEXT TO the reference folder
        # (not inside it -- other scripts read every file in that folder as an image)
        save_camera_settings(picam2, exposure_us, gain, colour_gains,
                             os.path.normpath(REF_DIR) + "_camera_settings_used.json")

    saved_roi = load_roi_config(ROI_CONFIG_PATH)
    roi, roi_changed = select_circular_roi(picam2, initial_roi=saved_roi)
    if roi is not None:
        if roi_changed:
            save_roi_config(ROI_CONFIG_PATH, roi)
        print(f"Preview ROI: center=({roi[0]},{roi[1]}), radius={roi[2]:.0f}px "
              f"(display only -- saved images are still full-frame)\n")
    else:
        print("No preview ROI -- showing full frame (roi_config.json unchanged).\n")

    phase = "ref"
    ref_count = 0
    sample_count = 0
    ref_gray_frames = []
    baseline_gray = None
    live_view = None

    # recording state (S = start, E = end)
    writer = None
    rec = None        # dict(idx, n, t0, csv, last_fps_t, last_fps_n, fps) while recording
    rec_total = 0

    window_name = "capture -- ENTER=save, [ ]=exposure, C=color check, N=next phase, Q=quit"
    diff_window_name = ("live press view (smoothed log signal)" if LIVE_VIEW == "log"
                        else "live diff vs reference (press direction only)")
    cv2.namedWindow(window_name)

    print("PHASE 1: capturing REFERENCE images.")
    print("  Tune exposure with [ and ] until the clipping readout stays at 0%")
    print(f"  WHILE PRESSING FIRMLY (only the {SIGNAL_CHANNEL} channel is checked) -- pressing")
    print("  brightens, so the pressed state")
    print("  is the brightest one and is what must not clip. Release, then save refs.")
    print("  Press ENTER to save each reference shot (sensor untouched) --")
    print("  the FIRST save locks the exposure.")
    print("  Press C (holding a RED object) to verify the channel order.")
    print("  Press N when done with references.  Q to quit.\n")

    def start_recording():
        nonlocal writer, rec
        if writer is None:
            writer = FrameWriter()
        idx = next_recording_index(SAMPLE_DIR)
        csv_path = os.path.join(SAMPLE_DIR, f"rec{idx:02d}_timestamps.csv")
        csv_f = open(csv_path, "w")
        csv_f.write("frame,file,time_s\n")
        t0 = time.monotonic()
        rec = dict(idx=idx, n=0, t0=t0, csv=csv_f, csv_path=csv_path, dropped0=writer.dropped,
                   fps_t=t0, fps_n=0, fps=0.0)
        print(f"\n[REC] Recording {idx:02d} STARTED -> {SAMPLE_DIR}/rec{idx:02d}_*.png  (press E to stop)")

    def stop_recording():
        nonlocal rec, rec_total
        if rec is None:
            return
        dur = time.monotonic() - rec["t0"]
        rec["csv"].close()
        dropped = writer.dropped - rec["dropped0"]
        print(f"[REC] Recording {rec['idx']:02d} STOPPED: {rec['n']} frames in {dur:.1f} s "
              f"({rec['n'] / max(dur, 1e-6):.1f} fps)"
              + (f", {dropped} frame(s) DROPPED (disk too slow -- lower resolution or set "
                 f"RECORD_SHOW_DIFF = False)" if dropped else "")
              + f"; timestamps: {rec['csv_path']}")
        rec_total += rec["n"]
        rec = None

    try:
        while True:
            frame_bgr = to_bgr(picam2.capture_array())

            if rec is not None:                      # recording: queue EVERY frame (full, uncropped)
                t = time.monotonic() - rec["t0"]
                fname = f"rec{rec['idx']:02d}_{rec['n']:05d}.png"
                if writer.put(os.path.join(SAMPLE_DIR, fname), frame_bgr):
                    rec["csv"].write(f"{rec['n']},{fname},{t:.4f}\n")
                    rec["n"] += 1
                rec["fps_n"] += 1
                now = time.monotonic()
                if now - rec["fps_t"] >= 1.0:
                    rec["fps"] = rec["fps_n"] / (now - rec["fps_t"])
                    rec["fps_t"], rec["fps_n"] = now, 0

            disp = crop_for_preview(frame_bgr, roi).copy()
            clip_pct = clipped_percent(crop_for_preview(frame_bgr, roi))
            clip_color = (0, 0, 255) if clip_pct > CLIP_WARN_PERCENT else (0, 255, 0)
            lock_txt = "LOCKED" if settings_locked else "tunable [ ]"
            cv2.putText(disp, f"[{phase.upper()}] ref: {ref_count}  sample: {sample_count}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, f"exposure {exposure_us}us ({lock_txt})",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            sig_max = np.percentile(signal_image(crop_for_preview(frame_bgr, roi)), 99.9)
            cv2.putText(disp, f"{SIGNAL_CHANNEL} clipped: {clip_pct:.2f}%   {SIGNAL_CHANNEL} max: {sig_max:.0f}",
                        (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clip_color, 2)
            if rec is not None:
                cv2.circle(disp, (disp.shape[1] - 25, 25), 10, (0, 0, 255), -1)
                cv2.putText(disp, f"REC {rec['idx']:02d}  {rec['n']} fr  {rec['fps']:.1f} fps  (E=stop)",
                            (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            elif phase == "sample":
                cv2.putText(disp, "S = start recording", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow(window_name, disp)

            if phase == "sample" and baseline_gray is not None and (rec is None or RECORD_SHOW_DIFF):
                if live_view is not None:
                    diff_disp, peak, _ = live_view.render(signal_image(frame_bgr))
                    cv2.putText(diff_disp, f"press peak: {peak:.2f}  (~{100 * (np.exp(peak) - 1):.0f}% brighter)",
                                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    diff = press_signal(baseline_gray, signal_image(frame_bgr))
                    diff_crop = crop_for_preview(diff, roi)
                    diff_disp = np.clip(diff_crop * DIFF_DISPLAY_GAIN, 0, 255).astype(np.uint8)
                    diff_disp = cv2.cvtColor(diff_disp, cv2.COLOR_GRAY2BGR)
                    cv2.putText(diff_disp, f"max raw {SIGNAL_CHANNEL} diff: {diff_crop.max():.1f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(diff_window_name, diff_disp)

            key = cv2.waitKey(1 if rec is not None else 20) & 0xFF

            if key == ord('q'):
                break

            if key == ord('s'):
                if phase != "sample":
                    print("Recording is only available in the SAMPLE phase (press N after the references).")
                elif rec is not None:
                    print("Already recording -- press E to stop.")
                else:
                    start_recording()
                continue

            if key == ord('e'):
                if rec is None:
                    print("Not recording (press S to start).")
                else:
                    stop_recording()
                continue

            if key == ord('c'):
                report_channel_means(frame_bgr, roi)
                continue

            if key in (ord('['), ord(']')):
                if settings_locked:
                    print("Exposure is locked (loaded from file, or a reference was already saved).")
                else:
                    if key == ord(']'):
                        exposure_us = int(exposure_us * EXPOSURE_STEP)
                    else:
                        exposure_us = max(100, int(exposure_us / EXPOSURE_STEP))
                    apply_camera_settings(picam2, exposure_us, gain, colour_gains)
                    print(f"Exposure -> {exposure_us} us")
                continue

            if key == ord('n') and phase == "ref":
                if ref_count == 0:
                    print("Save at least one reference image first.")
                    continue
                baseline_gray = np.mean(ref_gray_frames, axis=0)
                live_view = LivePressView(baseline_gray, roi) if LIVE_VIEW == "log" else None
                phase = "sample"
                cv2.namedWindow(diff_window_name)
                print("\nPHASE 2: capturing SAMPLE images.")
                print("  Watch 'press peak' to press to a similar depth at each location.")
                print("  Press ENTER to save each sample shot.")
                print("  Press S to START recording every frame, E to END the recording.  Q to finish.\n")
                continue

            if key in (13, 10):  # ENTER -- save FULL FRAME, never cropped
                if not settings_locked:
                    settings_locked = True
                    save_camera_settings(picam2, exposure_us, gain,
                                         colour_gains, SETTINGS_OUT_PATH)
                if clip_pct > CLIP_WARN_PERCENT:
                    print(f"  WARNING: {clip_pct:.2f}% of preview pixels have {SIGNAL_CHANNEL} clipped in this frame.")
                if phase == "ref":
                    fname = os.path.join(REF_DIR, f"ref_{ref_count:04d}.png")
                    cv2.imwrite(fname, frame_bgr)
                    ref_gray_frames.append(signal_image(frame_bgr))
                    print(f"Saved: {fname}")
                    ref_count += 1
                else:
                    fname = os.path.join(SAMPLE_DIR, f"sample_{sample_count:04d}.png")
                    cv2.imwrite(fname, frame_bgr)
                    print(f"Saved: {fname}")
                    sample_count += 1
    finally:
        stop_recording()                      # Q / Ctrl-C while recording still closes it cleanly
        if writer is not None:
            pending = writer.q.qsize()
            if pending:
                print(f"Writing the last {pending} recorded frame(s) to disk...")
            writer.close()
        cv2.destroyAllWindows()
        picam2.stop()

    print(f"\nDone. Saved {ref_count} reference image(s) to {REF_DIR}")
    print(f"Saved {sample_count} single sample image(s) to {SAMPLE_DIR}")
    if rec_total:
        print(f"Recorded {rec_total} frame(s) to {SAMPLE_DIR} (rec##_#####.png)")


if __name__ == "__main__":
    main()

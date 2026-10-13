"""
STEP 2: depth calibration from sphere presses at 1-3 spots, using the gain map.

For each sphere press you draw the contact circle (click center, click edge).
The sphere geometry gives every contact pixel its true depth. Each pixel's
signal is divided by the gain map's value at that SAME pixel:

    normalized = signal / signal_at_depth_map(pixel)
               = "what fraction of the map's reference press (e.g. 2 mm) this is"

This removes the location-dependent strength, so presses at different spots
should fall on ONE curve:  depth  <->  normalized signal.
With 2-3 spots, each spot's own curve is compared with the shared one -- if
they agree, the gain map transfers correctly between locations.

Settings (signal type, channel, press direction) are read from the gain map,
so they always match it.

Outputs:
    depth_curve.json   the shared curve + settings (read by reconstruct_with_gain_map.py)
    depth_curve.png    per-spot curves vs the shared curve

Images: references + sphere presses from ONE session with LOCKED camera
settings, matching the lighting/settings the gain map was built with.

Standalone: numpy, opencv-python, matplotlib, geometry.py.
Usage:
    python depth_calibration_sphere.py
"""

import os
import re
import glob
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")      # files only (matplotlib + OpenCV GUI clash on the Pi)
import matplotlib.pyplot as plt

import geometry
from geometry import (
    pixel_to_surface_point, point_to_st,
    pixel_to_surface_point_batch, point_to_st_batch, surface_distance_batch,
    surface_distance,
)

# ============================== CONFIG ==============================
REF_DIR = "./calibration_images/ref_new_sphere"
SPHERE_DIR = "./calibration_images/sample_new_sphere"
GAIN_MAP_PATH = "gain_map_model.npz"              # from fit_gain_model_v2.py

BALL_RADIUS_MM = 2.50
MIN_REF_INTENSITY = 15.0
MIN_MAP_SIGNAL_FRACTION = 0.05   # ignore pixels where the map's signal is tiny (unreliable division)
DISPLAY_GAIN = 3.0

CURVE_BIN_MM = 0.02
MIN_PIXELS_PER_BIN = 20
SPOT_MERGE_PX = 40               # presses whose centers are closer than this count as the same spot
ADD_MAP_ANCHOR = False           # also add the map's own point (press depth, normalized 1.0) to the
                                 # curve -- extends it to e.g. 2 mm, but the part between your deepest
                                 # sphere press and that point is a straight-line guess
TRIM_FLAT_END = True             # drop deepest points where the signal stops rising (vertical end)
FLAT_END_FRACTION = 0.01         # "stops rising" = rises less than 1% of the curve's signal range

SELECTIONS_PATH = "sphere_selections.json"
OUT_JSON = "depth_curve.json"
OUT_PNG = "depth_curve.png"
# ======================================================================

_IDX = {"B": 0, "G": 1, "R": 2}


# ------------------------------ shared helpers ------------------------------

def natural_key(path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(path))]


def load_gain_map(path=GAIN_MAP_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- build it with build_gain_map_flat.py + fit_gain_model_v2.py")
    gm = np.load(path, allow_pickle=True)
    return dict(signal_at_depth=gm["signal_at_depth"].astype(np.float64),
                signal_type=str(gm["signal_type"]), signal_channel=str(gm["signal_channel"]),
                press_brightens=bool(gm["press_brightens"]), press_depth_mm=float(gm["press_depth_mm"]),
                extrapolated=gm["extrapolated"] if "extrapolated" in gm.files else None)


def load_channel(path, channel):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read {path}")
    return img[..., _IDX[channel]].astype(np.float64)


def load_reference(ref_dir, channel):
    paths = sorted(glob.glob(os.path.join(ref_dir, "*")))
    if not paths:
        raise ValueError(f"No reference images in {ref_dir}")
    return np.mean([load_channel(p, channel) for p in paths], axis=0)


def compute_signal(sample, ref, signal_type, brightens):
    sign = 1.0 if brightens else -1.0
    if signal_type == "diff":
        return sign * (sample - ref)
    if signal_type == "log":
        return sign * np.log(np.maximum(sample, 1.0) / np.maximum(ref, 1.0))
    raise ValueError(f"Unknown signal type {signal_type}")


def normalize_by_map(sig, sad, min_frac):
    """signal / map value, NaN where the map is missing or too small to divide by."""
    floor = min_frac * np.nanmedian(sad)
    out = np.full(sig.shape, np.nan)
    ok = np.isfinite(sad) & (sad > floor)
    out[ok] = sig[ok] / sad[ok]
    return out


def extrapolation_slope(d, s, top_fraction=0.3):
    """mm per unit signal for extrapolating past the deepest point: a line fitted
    through the top part of the curve (not just the last two points, whose signal
    can be almost equal -> near-infinite slope). None if it can't be estimated."""
    d, s = np.asarray(d, float), np.asarray(s, float)
    sel = s >= s[-1] - top_fraction * (s[-1] - s[0])
    if sel.sum() < 3:
        sel = np.arange(len(s)) >= max(0, len(s) - 3)
    if np.ptp(s[sel]) <= 1e-9:
        return None
    slope = float(np.polyfit(s[sel], d[sel], 1)[0])
    return slope if np.isfinite(slope) and slope > 0 else None


def curve_inverse(n, curve):
    """normalized signal -> depth (mm); linear extrapolation past the deepest point, >= 0."""
    d, s = np.asarray(curve["depth_mm"]), np.asarray(curve["normalized_signal"])
    out = np.interp(n, s, d)
    slope = extrapolation_slope(d, s)
    if slope is not None:
        out = np.where(n > s[-1], d[-1] + slope * (n - s[-1]), out)
    return np.maximum(out, 0.0)


def cap_depth(r, rc, ball_r):
    with np.errstate(invalid="ignore"):
        d = np.sqrt(np.maximum(ball_r ** 2 - r ** 2, 0)) - np.sqrt(max(ball_r ** 2 - rc ** 2, 0))
    return np.where(r < rc, np.maximum(d, 0), 0.0)


def check_crop_file(w, h):
    if os.path.exists(geometry.ROI_CROP_INFO_PATH):
        raise RuntimeError(f"{geometry.ROI_CROP_INFO_PATH} exists, so geometry.py shifted the camera center "
                           f"for CROPPED images, but these are {w}x{h} full frames. Move/rename it and re-run.")


# ------------------------------ circle drawing ------------------------------

def manual_select_circle(diff_disp, raw_disp, title, previous=None):
    """Click CENTER then EDGE. w/a/s/d move, m/n resize, t diff/raw view,
    r restart, ENTER accept, ESC skip. Returns (cx, cy, radius_px, (ux, uy)) or None."""
    win = f"{title} -- click CENTER then EDGE | wasd move, m/n size, t view, r reset, ENTER ok, ESC skip"
    st = {"c": None, "r": None, "dir": (1.0, 0.0), "raw": False}
    if previous:
        st["c"], st["r"], st["dir"] = (int(previous[0]), int(previous[1])), float(previous[2]), tuple(previous[3])

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if st["c"] is None or st["r"] is not None:
            st["c"], st["r"] = (x, y), None
        else:
            dx, dy = x - st["c"][0], y - st["c"][1]
            rad = float(np.hypot(dx, dy))
            if rad >= 2:
                st["r"], st["dir"] = rad, (dx / rad, dy / rad)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    result = None
    while True:
        vis = (raw_disp if st["raw"] else diff_disp).copy()
        if st["c"] is not None:
            cv2.drawMarker(vis, st["c"], (0, 255, 0), cv2.MARKER_CROSS, 14, 1)
        if st["c"] is not None and st["r"] is not None:
            cv2.circle(vis, st["c"], int(round(st["r"])), (0, 255, 0), 1)
            ex = int(round(st["c"][0] + st["r"] * st["dir"][0]))
            ey = int(round(st["c"][1] + st["r"] * st["dir"][1]))
            cv2.circle(vis, (ex, ey), 3, (0, 0, 255), -1)
        cv2.putText(vis, "RAW image" if st["raw"] else "DIFF view", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key in (13, 10):
            if st["c"] is not None and st["r"] is not None:
                result = (float(st["c"][0]), float(st["c"][1]), float(st["r"]), list(st["dir"]))
            break
        if key == ord('t'):
            st["raw"] = not st["raw"]
        elif key == ord('r'):
            st["c"], st["r"] = None, None
        elif st["c"] is not None and st["r"] is not None:
            cx, cy = st["c"]
            if key == ord('w'):
                st["c"] = (cx, cy - 1)
            elif key == ord('s'):
                st["c"] = (cx, cy + 1)
            elif key == ord('a'):
                st["c"] = (cx - 1, cy)
            elif key == ord('d'):
                st["c"] = (cx + 1, cy)
            elif key == ord('m'):
                st["r"] += 1
            elif key == ord('n'):
                st["r"] = max(2.0, st["r"] - 1)
    cv2.destroyWindow(win)
    return result


def drawn_radius_mm(cx, cy, radius_px, direction):
    ex, ey = cx + radius_px * direction[0], cy + radius_px * direction[1]
    sc, pc = pixel_to_surface_point(cx, cy)
    se, pe = pixel_to_surface_point(ex, ey)
    if pc is None or pe is None:
        return None
    s1, t1 = point_to_st(pc, sc)
    s2, t2 = point_to_st(pe, se)
    return float(surface_distance(s1, t1, s2, t2))


def contact_pixels(cx, cy, rc_mm, half_px, h, w):
    x0, x1 = max(0, int(cx - half_px)), min(w, int(cx + half_px) + 1)
    y0, y1 = max(0, int(cy - half_px)), min(h, int(cy + half_px) + 1)
    vs, us = np.mgrid[y0:y1, x0:x1]
    us, vs = us.ravel(), vs.ravel()
    sc, pc = pixel_to_surface_point(cx, cy)
    if pc is None:
        return None
    s_c, t_c = point_to_st(pc, sc)
    names, pts, valid = pixel_to_surface_point_batch(us.astype(np.float64), vs.astype(np.float64))
    us, vs = us[valid], vs[valid]
    s_, t_ = point_to_st_batch(pts[valid], names[valid])
    r = surface_distance_batch(s_, t_, s_c, t_c)
    keep = r < rc_mm
    return us[keep], vs[keep], r[keep]


def binned_curve(D, N):
    edges = np.arange(0, D.max() + CURVE_BIN_MM, CURVE_BIN_MM)
    dep, val = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (D >= lo) & (D < hi)
        if m.sum() >= MIN_PIXELS_PER_BIN:
            dep.append(float(D[m].mean()))
            val.append(float(np.median(N[m])))
    return np.array(dep), np.array(val)


# ------------------------------ main ------------------------------

def main():
    gm = load_gain_map()
    sad = gm["signal_at_depth"]
    ch, stype, bright = gm["signal_channel"], gm["signal_type"], gm["press_brightens"]
    ref = load_reference(REF_DIR, ch)
    h, w = ref.shape
    if sad.shape != (h, w):
        raise RuntimeError(f"Gain map is {sad.shape}, images are {(h, w)}")
    check_crop_file(w, h)
    print(f"Gain map: {GAIN_MAP_PATH} ({stype} signal on {ch}, reference press {gm['press_depth_mm']} mm)")

    paths = sorted(glob.glob(os.path.join(SPHERE_DIR, "*")), key=natural_key)
    if not paths:
        raise ValueError(f"No sphere-press images in {SPHERE_DIR}")
    selections = json.load(open(SELECTIONS_PATH)) if os.path.exists(SELECTIONS_PATH) else {}
    n_saved = sum(1 for p in paths if os.path.normpath(os.path.join(SPHERE_DIR, os.path.basename(p))) in selections)
    print(f"{len(paths)} sphere-press image(s) in {SPHERE_DIR}; {n_saved} already have a saved circle "
          f"in {SELECTIONS_PATH}, {len(paths) - n_saved} to draw\n")

    presses = []
    for i, path in enumerate(paths):
        name = os.path.basename(path)
        key = os.path.normpath(os.path.join(SPHERE_DIR, name))   # folder + name: same file names in
        sample = load_channel(path, ch)                          # different folders never collide
        sig = compute_signal(sample, ref, stype, bright)
        if key in selections:
            circ = selections[key]
            if circ is None:
                print(f"[{i + 1}/{len(paths)}] {name}: excluded (null in {SELECTIONS_PATH})")
                continue
            print(f"[{i + 1}/{len(paths)}] {name}: reusing saved circle")
        else:
            disp_src = np.clip(sig, 0, None) * (DISPLAY_GAIN if stype == "diff" else 255.0)
            disp = cv2.cvtColor(np.clip(disp_src, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            print(f"[{i + 1}/{len(paths)}] {name}: draw the contact circle")
            circ = manual_select_circle(disp, cv2.imread(path), name)
            selections[key] = list(circ) if circ else None
            with open(SELECTIONS_PATH, "w") as f:
                json.dump(selections, f, indent=1)
            if circ is None:
                print("  skipped.")
                continue
        cx, cy, rad_px, direction = circ
        rc = drawn_radius_mm(cx, cy, rad_px, direction)
        if rc is None or rc >= 0.98 * BALL_RADIUS_MM:
            print(f"[{i + 1}/{len(paths)}] {name}: invalid circle (off the shell or larger than the ball) -- skipped")
            continue
        cp = contact_pixels(cx, cy, rc, 1.3 * rad_px + 5, h, w)
        if cp is None:
            print(f"[{i + 1}/{len(paths)}] {name}: center off the shell -- skipped")
            continue
        us, vs, r = cp
        n = normalize_by_map(sig[vs, us], sad[vs, us], MIN_MAP_SIGNAL_FRACTION)
        ok = np.isfinite(n) & (ref[vs, us] >= MIN_REF_INTENSITY)
        d = cap_depth(r[ok], rc, BALL_RADIUS_MM)
        presses.append(dict(name=name, cx=cx, cy=cy, rc=rc, D=d, N=n[ok], S=sig[vs[ok], us[ok]]))
        print(f"[{i + 1}/{len(paths)}] {name}: center ({cx:.0f},{cy:.0f}), contact r={rc:.2f}mm, "
              f"center depth {d.max():.3f}mm, {ok.sum()} px, "
              f"normalized signal at center {np.median(n[ok][d > 0.8 * d.max()]):.3f}")
    cv2.destroyAllWindows()
    if not presses:
        print("No usable presses.")
        return

    # group presses into spots
    spot_of = []
    spots = []
    for p in presses:
        for k, (sx, sy) in enumerate(spots):
            if np.hypot(p["cx"] - sx, p["cy"] - sy) < SPOT_MERGE_PX:
                spot_of.append(k)
                break
        else:
            spots.append((p["cx"], p["cy"]))
            spot_of.append(len(spots) - 1)

    D = np.concatenate([p["D"] for p in presses])
    N = np.concatenate([p["N"] for p in presses])
    dep, val = binned_curve(D, N)
    if len(dep) < 3:
        print("Not enough depth range for a curve -- press deeper or add presses.")
        return
    dep, val = np.concatenate([[0.0], dep]), np.concatenate([[0.0], val])
    if ADD_MAP_ANCHOR and gm["press_depth_mm"] > dep[-1]:
        dep, val = np.append(dep, gm["press_depth_mm"]), np.append(val, 1.0)
        print(f"Added the map's own point ({gm['press_depth_mm']} mm, 1.0); the curve between "
              f"{dep[-2]:.2f} and {dep[-1]:.2f} mm is a straight-line guess.")
    mono = np.maximum.accumulate(val) + 1e-6 * np.arange(len(val))

    # Drop the flat top: deepest bins whose signal no longer rises (saturation / too few
    # pixels at the very center). They show up as a vertical line and make signal ->
    # depth ambiguous there.
    if TRIM_FLAT_END:
        rng = mono[-1] - mono[0]
        n_before = len(mono)
        while len(mono) > 3 and mono[-1] - mono[-2] < FLAT_END_FRACTION * rng:
            dep, val, mono = dep[:-1], val[:-1], mono[:-1]
        if len(mono) < n_before:
            print(f"Trimmed {n_before - len(mono)} flat point(s) at the deep end "
                  f"(signal stopped rising); curve now ends at {dep[-1]:.3f} mm")

    # per-spot consistency (does the gain map transfer between spots?)
    print("\n" + "=" * 64)
    print(f"Shared curve from {len(presses)} press(es) at {len(spots)} spot(s): 0 - {dep[-1]:.3f} mm")
    spot_curves = []
    for k, (sx, sy) in enumerate(spots):
        Dk = np.concatenate([p["D"] for p, s in zip(presses, spot_of) if s == k])
        Nk = np.concatenate([p["N"] for p, s in zip(presses, spot_of) if s == k])
        dk, vk = binned_curve(Dk, Nk)
        spot_curves.append((sx, sy, dk, vk))
        if len(dk) >= 2 and len(spots) > 1:
            shared = np.interp(dk, dep, mono)
            sel = dk > 0.05 * dep[-1]
            dev = 100 * np.median(vk[sel] / shared[sel] - 1) if sel.any() else float("nan")
            print(f"  spot {k + 1} at ({sx:.0f},{sy:.0f}): reads {dev:+.1f}% vs the shared curve")
    if len(spots) == 1:
        print("  Only one spot: the gain map's transfer between locations is not tested.")
        print("  Pressing a 2nd (and 3rd) spot elsewhere checks it: their curves should overlap.")
    print("=" * 64)

    curve = dict(depth_mm=[round(float(v), 5) for v in dep], normalized_signal=[round(float(v), 6) for v in mono],
                 signal_type=stype, signal_channel=ch, press_brightens=bright, gain_map=GAIN_MAP_PATH,
                 map_press_depth_mm=gm["press_depth_mm"], ball_radius_mm=BALL_RADIUS_MM,
                 max_depth_mm=float(dep[-1]), spots=[[float(sx), float(sy)] for sx, sy in spots],
                 n_presses=len(presses), map_anchor_added=bool(ADD_MAP_ANCHOR))
    with open(OUT_JSON, "w") as f:
        json.dump(curve, f, indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    rs = np.random.RandomState(0)
    colors = plt.cm.tab10.colors
    for k, (sx, sy, dk, vk) in enumerate(spot_curves):
        for p, s in zip(presses, spot_of):
            if s == k:
                sub = rs.choice(len(p["D"]), min(len(p["D"]), 3000), replace=False)
                axs[0].scatter(p["D"][sub], p["S"][sub], s=1, alpha=0.15, color=colors[k % 10])
                axs[1].scatter(p["D"][sub], p["N"][sub], s=1, alpha=0.15, color=colors[k % 10])
        axs[1].plot(dk, vk, "-", color=colors[k % 10], lw=1.5, label=f"spot {k + 1} ({sx:.0f},{sy:.0f})")
    axs[1].plot(dep, mono, "k--", lw=2, label="shared curve")
    axs[0].set_title("Raw signal vs depth (differs by spot)")
    axs[0].set_ylabel(f"signal ({stype}, {ch})")
    axs[1].set_title("Normalized by the gain map (should overlap)")
    axs[1].set_ylabel(f"signal / map signal at {gm['press_depth_mm']} mm")
    for ax in axs:
        ax.set_xlabel("depth (mm)")
        ax.grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    print(f"Saved: {os.path.abspath(OUT_JSON)}")
    print(f"Saved: {os.path.abspath(OUT_PNG)}")
    print(f"Saved circles: {os.path.abspath(SELECTIONS_PATH)}")


if __name__ == "__main__":
    main()

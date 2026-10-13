"""
Camera + curved-shell geometry, extracted/extended from your indentation
tracking script. Shared by the calibration script and (later) your 3D
reconstruction pipeline, so both use exactly the same pixel<->surface
correspondence.

Sensor shape: a cylinder of radius R and length L, capped by a hemisphere
of the same radius R, centered at (0, 0, L) -- same convention as your
original script.
"""

import numpy as np
import cv2
import os
import json

# ============================== CONFIG ==============================
# Camera intrinsics -- from fisheye calibration (cv2.fisheye.calibrate),
# RMS reprojection error 0.59 px across 86 images
K = np.array([
    [580.62736263, 0.00000000, 605.68002915],
    [0.00000000, 583.21008639, 340.10588715],
    [0.00000000, 0.00000000, 1.00000000],
], dtype=np.float64)

# Fisheye distortion coefficients (k1, k2, k3, k4) -- NOT the standard
# (k1,k2,p1,p2,k3) pinhole model. Large values here are expected/normal
# for a 160 deg FOV lens.
dist_coeffs = np.array([
    [0.23814962], [2.39255577], [-5.68380066], [3.26449311]
], dtype=np.float64)

# ------------------------- automatic ROI-crop adjustment -------------------------
# If capture_calibration_images_cropped.py was used, it saves the crop
# offset + adjusted principal point to this file. If present, K's cx/cy
# are automatically corrected here so every script that imports geometry
# (calibration, reconstruction, everything) uses the RIGHT principal point
# for the cropped images, without needing to edit K by hand each time.
ROI_CROP_INFO_PATH = "./calibration_images/roi_crop_info.json"

if os.path.exists(ROI_CROP_INFO_PATH):
    with open(ROI_CROP_INFO_PATH, "r") as _f:
        _crop_info = json.load(_f)
    _adj_cx, _adj_cy = _crop_info["adjusted_cx_cy"]
    print(f"[geometry.py] Detected {ROI_CROP_INFO_PATH} -- adjusting principal "
          f"point for cropped images: cx,cy = ({K[0, 2]:.2f},{K[1, 2]:.2f}) "
          f"-> ({_adj_cx:.2f},{_adj_cy:.2f})")
    K[0, 2] = _adj_cx
    K[1, 2] = _adj_cy

# Sensor shell geometry, in mm
R = 10.0
L = 12.0
SPHERE_CENTER = np.array([0.0, 0.0, L])

# ------------------------- camera EXTRINSICS -------------------------
# By default, the camera is assumed to sit exactly at the shell's origin,
# looking straight along its axis (rvec=0, tvec=0 -- i.e. no correction).
# If you've run solve_camera_pose.py against physical target points and it
# found a non-trivial pose, paste the printed EXTRINSIC_RVEC/EXTRINSIC_TVEC
# here to correct for real camera tilt/offset relative to the shell axis.
EXTRINSIC_RVEC = np.array([0.0, 0.0, 0.0])
EXTRINSIC_TVEC = np.array([0.0, 0.0, 0.0])

_R_CAM, _ = cv2.Rodrigues(EXTRINSIC_RVEC)
# Camera's own origin, expressed in shell coordinates (derived from the
# extrinsics above): point_cam = R_CAM @ point_shell + EXTRINSIC_TVEC, so
# the camera origin (point_cam = 0) in shell coordinates is:
_CAMERA_ORIGIN_SHELL = -_R_CAM.T @ EXTRINSIC_TVEC
# ======================================================================


# ------------------------- pixel -> 3D surface point -------------------------

def pixel_to_ray(u, v):
    """Ray direction in CAMERA frame (intrinsics only, no extrinsics)."""
    pts = np.array([[[u, v]]], dtype=np.float64)
    undistorted = cv2.fisheye.undistortPoints(pts, K, dist_coeffs)
    x, y = undistorted[0, 0]
    ray = np.array([x, y, 1.0], dtype=np.float64)
    return ray / np.linalg.norm(ray)


def pixel_to_ray_shell(u, v):
    """Ray origin + direction in SHELL frame, applying the extrinsic
    camera pose correction. Use this (not pixel_to_ray) for anything
    involving the shell geometry."""
    ray_cam = pixel_to_ray(u, v)
    ray_shell = _R_CAM.T @ ray_cam
    return _CAMERA_ORIGIN_SHELL, ray_shell


def intersect_cylinder(origin, ray):
    """origin: ray's starting point in shell frame (usually the camera
    origin expressed in shell coordinates). ray: unit direction, shell
    frame. Returns the nearest valid intersection point, or None."""
    dx, dy, dz = ray
    a = dx ** 2 + dy ** 2
    if a < 1e-12:
        return None
    Ox, Oy, Oz = origin
    b = 2.0 * (Ox * dx + Oy * dy)
    c = Ox ** 2 + Oy ** 2 - R ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    candidates = []
    for t in sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)]):
        if t > 0:
            P = origin + t * ray
            if 0 <= P[2] <= L:
                candidates.append((t, P))
    return candidates[0][1] if candidates else None


def intersect_hemisphere(origin, ray):
    """Same convention as intersect_cylinder."""
    W = origin - SPHERE_CENTER
    a = ray @ ray  # 1.0 for a unit ray, kept general just in case
    b = 2.0 * (W @ ray)
    c = W @ W - R ** 2
    disc = b ** 2 - 4 * a * c
    if disc < 0:
        return None
    sq = np.sqrt(disc)
    candidates = []
    for t in sorted([(-b - sq) / (2 * a), (-b + sq) / (2 * a)]):
        if t > 0:
            P = origin + t * ray
            if P[2] >= L:
                candidates.append((t, P))
    return candidates[0][1] if candidates else None


def pixel_to_surface_point(u, v):
    """Returns (surface_name, 3D point) for pixel (u, v) on the UNDEFORMED
    shell, or (None, None) if the ray misses the shell entirely. Applies
    the extrinsic camera pose correction (see EXTRINSIC_RVEC/TVEC above)."""
    origin, ray = pixel_to_ray_shell(u, v)
    Pcyl = intersect_cylinder(origin, ray)
    Psph = intersect_hemisphere(origin, ray)
    candidates = []
    if Pcyl is not None:
        candidates.append(("cylinder", Pcyl, np.linalg.norm(Pcyl - origin)))
    if Psph is not None:
        candidates.append(("hemisphere", Psph, np.linalg.norm(Psph - origin)))
    if not candidates:
        return None, None
    name, point, _ = min(candidates, key=lambda item: item[2])
    return name, point


# ------------------------- surface normal + parametrization -------------------------

def surface_normal(point, surface_name):
    """Outward unit normal of the UNDEFORMED shell at a given 3D point."""
    if surface_name == "cylinder":
        n = np.array([point[0], point[1], 0.0])
    else:  # hemisphere
        n = point - SPHERE_CENTER
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return n / norm


def surface_normal_batch(points, surf_names):
    """Vectorized version of surface_normal. points: (N,3), surf_names:
    (N,) object array. Returns (N,3) array of outward unit normals."""
    is_cyl = surf_names == "cylinder"
    is_sph = surf_names == "hemisphere"

    n = np.zeros_like(points)
    n[is_cyl, 0] = points[is_cyl, 0]
    n[is_cyl, 1] = points[is_cyl, 1]
    n[is_sph] = points[is_sph] - SPHERE_CENTER

    norms = np.linalg.norm(n, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    unit_n = n / safe_norms
    # fallback direction for degenerate (zero-length) normals
    degenerate = (norms < 1e-12).flatten()
    unit_n[degenerate] = np.array([0.0, 0.0, 1.0])
    return unit_n


def point_to_st(point, surface_name):
    """
    Unroll a 3D point on the shell into (s, theta):
      theta = angle around the central axis
      s     = arc length from the base (z=0), continuous across the
              cylinder/hemisphere boundary and up to the hemisphere tip.
    """
    x, y, z = point
    theta = np.arctan2(y, x)

    if surface_name == "cylinder":
        s = z
    else:
        # phi = polar angle from the +z axis, measured at the sphere center.
        # phi = pi/2 at the cylinder/hemisphere boundary (z = L),
        # phi = 0 at the very tip of the hemisphere.
        dz = z - SPHERE_CENTER[2]
        phi = np.arccos(np.clip(dz / R, -1.0, 1.0))
        s = L + R * (np.pi / 2 - phi)

    return s, theta


def local_radius_from_axis(s):
    """Physical radius from the central axis at arc-length position s
    (R on the cylinder body, shrinking to 0 at the hemisphere tip).
    Needed to convert an angular difference (theta) into a true
    physical (mm) distance at that position."""
    if s <= L:
        return R
    phi = (np.pi / 2) - (s - L) / R
    return R * np.sin(np.clip(phi, 0.0, np.pi / 2))


def angdiff(theta1, theta2):
    """Shortest signed angular difference, wrapped to [-pi, pi]."""
    d = theta1 - theta2
    return (d + np.pi) % (2 * np.pi) - np.pi


def surface_distance(s1, theta1, s2, theta2):
    """
    Approximate true physical (mm) distance between two points on the
    unrolled shell, using the LOCAL radius (evaluated at the midpoint's s)
    to convert the angular difference into a physical arc length. This is
    the flat-tangent-plane approximation -- valid as long as the two points
    are close together relative to the shell's own curvature (true for a
    small ball-indentation contact patch).
    """
    s_mid = (s1 + s2) / 2
    r_mid = local_radius_from_axis(s_mid)
    d_theta = angdiff(theta1, theta2)
    d_s = s1 - s2
    d_arc = r_mid * d_theta
    return np.sqrt(d_s ** 2 + d_arc ** 2)


# ------------------------- vectorized (batch) versions -------------------------
# Same math as above, but operating on arrays of pixels at once via
# cv2.fisheye.undistortPoints' native batch support. Use these for anything beyond a
# handful of pixels -- looping the single-point versions is far too slow for
# a real contact patch (hundreds to thousands of pixels per press).

def pixel_to_ray_batch(us, vs):
    """Ray directions in CAMERA frame (intrinsics only)."""
    pts = np.stack([np.asarray(us, dtype=np.float64), np.asarray(vs, dtype=np.float64)],
                    axis=1).reshape(-1, 1, 2)
    undistorted = cv2.fisheye.undistortPoints(pts, K, dist_coeffs).reshape(-1, 2)
    rays = np.concatenate([undistorted, np.ones((len(undistorted), 1))], axis=1)
    rays = rays / np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def pixel_to_ray_shell_batch(us, vs):
    """Ray directions in SHELL frame (extrinsics applied), plus the
    (constant) camera origin in shell coordinates."""
    rays_cam = pixel_to_ray_batch(us, vs)
    rays_shell = rays_cam @ _R_CAM  # equivalent to (R_CAM.T @ ray) per-row
    return _CAMERA_ORIGIN_SHELL, rays_shell


def intersect_cylinder_batch(origin, rays):
    dx, dy, dz = rays[:, 0], rays[:, 1], rays[:, 2]
    a = dx ** 2 + dy ** 2
    valid_a = a > 1e-12

    Ox, Oy, Oz = origin
    b = 2.0 * (Ox * dx + Oy * dy)
    c = Ox ** 2 + Oy ** 2 - R ** 2
    disc = b ** 2 - 4 * a * c
    valid_disc = valid_a & (disc >= 0)

    sq = np.sqrt(np.maximum(disc, 0))
    t1 = np.full(len(rays), np.nan)
    t2 = np.full(len(rays), np.nan)
    t1[valid_a] = (-b[valid_a] - sq[valid_a]) / (2 * a[valid_a])
    t2[valid_a] = (-b[valid_a] + sq[valid_a]) / (2 * a[valid_a])

    P1 = origin + t1[:, None] * rays
    P2 = origin + t2[:, None] * rays
    cand1 = valid_disc & (t1 > 0) & (P1[:, 2] >= 0) & (P1[:, 2] <= L)
    cand2 = valid_disc & (t2 > 0) & (P2[:, 2] >= 0) & (P2[:, 2] <= L)

    # prefer the smaller positive t (nearest intersection along the ray)
    use1 = cand1 & (~cand2 | (t1 <= t2))
    use2 = cand2 & (~use1)

    P = np.full((len(rays), 3), np.nan)
    P[use1] = P1[use1]
    P[use2] = P2[use2]
    valid = use1 | use2
    t_used = np.where(use1, t1, np.where(use2, t2, np.nan))
    return P, valid, t_used


def intersect_hemisphere_batch(origin, rays):
    W = origin - SPHERE_CENTER
    a = np.sum(rays * rays, axis=1)
    b = 2.0 * (rays @ W)
    c = np.dot(W, W) - R ** 2
    disc = b ** 2 - 4 * a * c
    valid_disc = disc >= 0
    sqrt_disc = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    P1 = origin + t1[:, None] * rays
    P2 = origin + t2[:, None] * rays
    cand1 = valid_disc & (t1 > 0) & (P1[:, 2] >= L)
    cand2 = valid_disc & (t2 > 0) & (P2[:, 2] >= L)

    use1 = cand1 & (~cand2 | (t1 <= t2))
    use2 = cand2 & (~use1)

    P = np.full_like(P1, np.nan)
    P[use1] = P1[use1]
    P[use2] = P2[use2]
    valid = use1 | use2
    t_used = np.where(use1, t1, np.where(use2, t2, np.nan))
    return P, valid, t_used


def pixel_to_surface_point_batch(us, vs):
    """Vectorized version of pixel_to_surface_point. Returns:
      surf_names: object array of 'cylinder' / 'hemisphere' / None
      points:     (N, 3) array, NaN rows where the ray missed the shell
      valid:      boolean mask, True where a valid surface point was found
    """
    origin, rays = pixel_to_ray_shell_batch(us, vs)
    Pcyl, valid_cyl, t_cyl = intersect_cylinder_batch(origin, rays)
    Psph, valid_sph, t_sph = intersect_hemisphere_batch(origin, rays)

    # prefer whichever is valid and has the smaller t (nearer along the ray)
    use_cyl = valid_cyl & (~valid_sph | (t_cyl <= t_sph))
    use_sph = valid_sph & (~use_cyl)

    n = len(us)
    points = np.full((n, 3), np.nan)
    points[use_cyl] = Pcyl[use_cyl]
    points[use_sph] = Psph[use_sph]

    surf_names = np.array([None] * n, dtype=object)
    surf_names[use_cyl] = "cylinder"
    surf_names[use_sph] = "hemisphere"

    valid = use_cyl | use_sph
    return surf_names, points, valid


def point_to_st_batch(points, surf_names):
    """Vectorized version of point_to_st. points: (N,3), surf_names: (N,)
    object array. Returns s, theta arrays (NaN where surf_names is None)."""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arctan2(y, x)

    is_cyl = surf_names == "cylinder"
    is_sph = surf_names == "hemisphere"

    s = np.full(len(points), np.nan)
    s[is_cyl] = z[is_cyl]

    dz = z[is_sph] - SPHERE_CENTER[2]
    phi = np.arccos(np.clip(dz / R, -1.0, 1.0))
    s[is_sph] = L + R * (np.pi / 2 - phi)

    return s, theta


def surface_distance_batch(s1, theta1, s2, theta2):
    """Vectorized version of surface_distance. s1/theta1 are arrays,
    s2/theta2 are scalars (the fixed contact center)."""
    s_mid = (s1 + s2) / 2
    r_mid = np.array([local_radius_from_axis(s) for s in s_mid])
    d_theta = (theta1 - theta2 + np.pi) % (2 * np.pi) - np.pi
    d_s = s1 - s2
    d_arc = r_mid * d_theta
    return np.sqrt(d_s ** 2 + d_arc ** 2)

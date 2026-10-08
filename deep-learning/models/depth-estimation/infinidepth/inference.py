#!/usr/bin/env python3
"""
Minimal depth inference using the exported InfiniDepth ONNX model.

Required Python packages (no PyTorch / CUDA needed):
    pip install onnxruntime numpy pillow

Optional (for colourised depth visualisation):
    pip install matplotlib

Usage — depth map only:
    python infer_depth_onnx.py \\
        --onnx  checkpoints/depth/infinidepth.onnx \\
        --image example_data/image/courtyard.jpg \\
        --out   output/

Usage — depth map + point cloud:
    python infer_depth_onnx.py \\
        --onnx  checkpoints/depth/infinidepth.onnx \\
        --image example_data/image/courtyard.jpg \\
        --out   output/ \\
        --save_pcd \\
        --fx 866 --fy 866 --cx 512 --cy 384
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageOps


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_and_resize(
    image_path: str,
    target_h: int = 768,
    target_w: int = 1024,
) -> tuple[np.ndarray, int, int]:
    """
    Load image, apply EXIF rotation, resize to target_h × target_w.

    Returns:
        img_resized  float32 [1, 3, H, W]  in [0, 1]
        orig_h, orig_w   original dimensions before resize
    """
    pil = Image.open(image_path)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    orig_w, orig_h = pil.size  # PIL uses (W, H)
    pil = pil.resize((target_w, target_h), Image.LANCZOS)
    arr = np.array(pil, dtype=np.float32) / 255.0          # [H, W, 3]
    arr = arr.transpose(2, 0, 1)[None]                      # [1, 3, H, W]
    return arr, orig_h, orig_w


# ---------------------------------------------------------------------------
# Query coordinate helpers
# ---------------------------------------------------------------------------

def make_query_grid(h: int, w: int) -> np.ndarray:
    """
    Dense pixel-aligned query grid, shape [1, h*w, 2].
    Coordinate order: (y_norm, x_norm) both in [-1, 1].
    """
    ys = ((np.arange(h, dtype=np.float32) + 0.5) / h) * 2.0 - 1.0
    xs = ((np.arange(w, dtype=np.float32) + 0.5) / w) * 2.0 - 1.0
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")     # [H, W] each
    coords = np.stack([grid_y, grid_x], axis=-1)             # [H, W, 2]
    return coords.reshape(1, -1, 2)                          # [1, H*W, 2]


# ---------------------------------------------------------------------------
# Depth postprocessing
# ---------------------------------------------------------------------------

def disparity_to_depth(disparity: np.ndarray, min_disp: float = 5e-3) -> np.ndarray:
    """Convert disparity [*, 1] to metric-relative depth [*, 1]."""
    return 1.0 / np.clip(disparity, min_disp, None)


def colorize_depth(
    depth_hw: np.ndarray,
    d_lo: Optional[float] = None,
    d_hi: Optional[float] = None,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Map a [H, W] depth array to an RGB uint8 image.
    Matches the original InfiniDepth Spectral colourmap (near=red/warm, far=blue/cool).

    Args:
        depth_hw : [H, W] depth map
        d_lo     : lower clip value (default: 2nd percentile of the map)
        d_hi     : upper clip value (default: 5× median, capped at 98th percentile).
                   Passing the same cap used for point cloud generation ensures the
                   colour scale covers the scene's main depth range rather than being
                   compressed by extreme sky/background outliers.
    """
    d = depth_hw.ravel()
    if valid_mask is None:
        valid_mask = np.isfinite(depth_hw) & (depth_hw > 0)
    else:
        valid_mask = valid_mask.astype(bool) & np.isfinite(depth_hw) & (depth_hw > 0)

    d_finite = depth_hw[valid_mask]
    if d_finite.size == 0:
        return np.zeros((*depth_hw.shape, 3), dtype=np.uint8)
    if d_lo is None:
        d_lo = float(np.percentile(d_finite, 2))
    if d_hi is None:
        # Default: 5× median so the arch / background is visible in colour;
        # same heuristic as depth_to_pointcloud's max_depth_factor=5 default.
        d_hi = float(min(np.percentile(d_finite, 98), np.median(d_finite) * 5))
    norm = np.clip((depth_hw - d_lo) / max(d_hi - d_lo, 1e-6), 0.0, 1.0)

    try:
        import matplotlib.cm as cm
        # Spectral: 0→red (near), 1→purple/blue (far) — matches InfiniDepth's plot_depth
        rgba = cm.Spectral(norm)            # [H, W, 4]
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    except ImportError:
        gray = ((1.0 - norm) * 255).astype(np.uint8)   # invert: near=bright
        rgb = np.stack([gray, gray, gray], axis=-1)

    rgb[~valid_mask] = 0
    return rgb


def colorize_scalar_map(
    value_hw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    cmap_name: str,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    symmetric: bool = False,
    invalid_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    valid_mask = valid_mask.astype(bool) & np.isfinite(value_hw)
    values = value_hw[valid_mask]
    if values.size == 0:
        return np.full((*value_hw.shape, 3), invalid_rgb, dtype=np.uint8)

    if symmetric:
        hi = float(np.percentile(np.abs(values), upper_percentile))
        lo = -hi
    else:
        lo = float(np.percentile(values, lower_percentile))
        hi = float(np.percentile(values, upper_percentile))

    norm = np.clip((value_hw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    try:
        import matplotlib

        cmap = matplotlib.colormaps[cmap_name]
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    except ImportError:
        gray = (norm * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)

    rgb[~valid_mask] = np.asarray(invalid_rgb, dtype=np.uint8)
    return rgb


def estimate_center_person_mask(
    depth_hw: np.ndarray,
    *,
    seed_y0: float = 0.38,
    seed_y1: float = 0.82,
    seed_x0: float = 0.35,
    seed_x1: float = 0.65,
    anchor_y0: float = 0.45,
    anchor_y1: float = 0.78,
    anchor_x0: float = 0.40,
    anchor_x1: float = 0.60,
    mad_scale: float = 8.0,
    min_margin_ratio: float = 0.05,
    max_margin_ratio: float = 0.35,
    dilate_px: int = 5,
    focus_scale: float = 1.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Estimate a centered upright-person mask from the depth map alone.

    The heuristic is tuned for back / back-left / back-right captures where the
    subject occupies the central torso region and should be the closest large
    connected component in the image. It is intentionally conservative: if it
    cannot find a stable component, it falls back to an all-valid mask.
    """
    valid = np.isfinite(depth_hw) & (depth_hw > 0)
    H, W = depth_hw.shape

    def _box(y0: float, y1: float, x0: float, x1: float) -> tuple[int, int, int, int]:
        iy0 = max(0, min(H - 1, int(round(H * y0))))
        iy1 = max(iy0 + 1, min(H, int(round(H * y1))))
        ix0 = max(0, min(W - 1, int(round(W * x0))))
        ix1 = max(ix0 + 1, min(W, int(round(W * x1))))
        return iy0, iy1, ix0, ix1

    sy0, sy1, sx0, sx1 = _box(seed_y0, seed_y1, seed_x0, seed_x1)
    seed_vals = depth_hw[sy0:sy1, sx0:sx1]
    seed_vals = seed_vals[np.isfinite(seed_vals) & (seed_vals > 0)]
    if seed_vals.size < 256:
        return valid.copy(), {"status": "fallback_seed", "threshold": float("inf")}

    seed_median = float(np.median(seed_vals))
    seed_mad = float(np.median(np.abs(seed_vals - seed_median)))
    min_margin = max(min_margin_ratio * seed_median, 1e-3)
    max_margin = max(max_margin_ratio * seed_median, min_margin)
    base_margin = float(np.clip(max(mad_scale * seed_mad, min_margin), min_margin, max_margin))
    focus_scale = max(float(focus_scale), 1e-3)
    depth_margin = max(1e-3, base_margin * focus_scale)
    depth_threshold = float(seed_median + depth_margin)
    candidate = valid & (depth_hw <= depth_threshold)

    labeled, component_ids = _label_connected_components(candidate)
    if component_ids.size == 0:
        return valid.copy(), {"status": "fallback_empty", "threshold": depth_threshold}

    ay0, ay1, ax0, ax1 = _box(anchor_y0, anchor_y1, anchor_x0, anchor_x1)
    anchor_ids = labeled[ay0:ay1, ax0:ax1]
    anchor_ids = anchor_ids[anchor_ids > 0]
    if anchor_ids.size == 0:
        return valid.copy(), {"status": "fallback_anchor", "threshold": depth_threshold}

    anchor_component_ids, anchor_counts = np.unique(anchor_ids, return_counts=True)
    chosen_id = int(anchor_component_ids[np.argmax(anchor_counts)])
    mask = labeled == chosen_id

    scipy_ndi = _optional_scipy_ndimage()
    if scipy_ndi is not None:
        mask = scipy_ndi.binary_fill_holes(mask)
        if dilate_px > 0:
            mask = scipy_ndi.binary_dilation(mask, iterations=int(dilate_px))
        mask &= valid & (depth_hw <= depth_threshold * 1.08)

    return mask.astype(bool), {
        "status": "ok",
        "threshold": depth_threshold,
        "margin": depth_margin,
        "focus_scale": focus_scale,
        "seed_median": seed_median,
        "seed_mad": seed_mad,
        "pixels": int(mask.sum()),
    }


def _optional_scipy_ndimage():
    try:
        from scipy import ndimage as ndi
        return ndi
    except ImportError:
        return None


def _label_connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scipy_ndi = _optional_scipy_ndimage()
    if scipy_ndi is not None:
        labeled, num = scipy_ndi.label(mask)
        return labeled.astype(np.int32, copy=False), np.arange(1, int(num) + 1, dtype=np.int32)
    return _label_connected_components_fallback(mask)


def _label_connected_components_fallback(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from collections import deque

    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    current = 0
    for y in range(H):
        for x in range(W):
            if not mask[y, x] or labeled[y, x] != 0:
                continue
            current += 1
            q = deque([(y, x)])
            labeled[y, x] = current
            while q:
                cy, cx = q.popleft()
                y0 = max(0, cy - 1)
                y1 = min(H, cy + 2)
                x0 = max(0, cx - 1)
                x1 = min(W, cx + 2)
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if mask[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = current
                            q.append((ny, nx))

    return labeled, np.arange(1, current + 1, dtype=np.int32)


def compute_back_surface_analysis(
    depth_hw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    topograph_sigma: float = 4.5,
    curvature_sigma: float = 3.0,
) -> dict[str, np.ndarray]:
    """
    Build back-surface analysis maps from a cropped person depth map.

    Topograph:
        Remove the broad torso trend with a quadratic fit, then express local
        surface height so protrusions towards the camera are positive.

    Curvature:
        Use the lateral second derivative of the smoothed topograph. This acts
        as a spine cue because a vertical midline groove is a lateral valley.
    """
    valid_mask = valid_mask.astype(bool) & np.isfinite(depth_hw) & (depth_hw > 0)
    if not np.any(valid_mask):
        zeros = np.zeros_like(depth_hw, dtype=np.float32)
        return {
            "trend_depth": zeros,
            "topograph": zeros,
            "curvature": zeros,
        }

    scipy_ndi = _optional_scipy_ndimage()
    if scipy_ndi is None:
        raise RuntimeError("Back-surface analysis requires scipy.ndimage.")

    H, W = depth_hw.shape
    ys, xs = np.nonzero(valid_mask)
    z = depth_hw[valid_mask].astype(np.float32)

    xn = (xs.astype(np.float32) / max(W - 1, 1)) * 2.0 - 1.0
    yn = (ys.astype(np.float32) / max(H - 1, 1)) * 2.0 - 1.0
    design = np.stack(
        [np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn],
        axis=1,
    )
    coeffs, *_ = np.linalg.lstsq(design, z, rcond=None)

    trend_depth = np.zeros_like(depth_hw, dtype=np.float32)
    trend_depth[valid_mask] = (design @ coeffs).astype(np.float32)

    topograph_raw = np.zeros_like(depth_hw, dtype=np.float32)
    topograph_raw[valid_mask] = trend_depth[valid_mask] - depth_hw[valid_mask]

    _, nearest_idx = scipy_ndi.distance_transform_edt(~valid_mask, return_indices=True)
    topograph_filled = topograph_raw[tuple(nearest_idx)]
    topograph = scipy_ndi.gaussian_filter(topograph_filled, sigma=float(topograph_sigma))
    topograph[~valid_mask] = 0.0

    curvature = -scipy_ndi.gaussian_filter(
        topograph,
        sigma=float(curvature_sigma),
        order=[0, 2],
    ).astype(np.float32)
    curvature[~valid_mask] = 0.0

    return {
        "trend_depth": trend_depth,
        "topograph": topograph.astype(np.float32),
        "curvature": curvature,
    }


def estimate_spine_hint(
    topograph_hw: np.ndarray,
    valid_mask: np.ndarray,
    *,
    torso_row_width_ratio: float = 0.55,
    search_half_width_ratio: float = 0.08,
    ridge_window_far_ratio: float = 0.15,
    ridge_window_near_ratio: float = 0.03,
    profile_sigma_px: float = 3.0,
    curvature_sigma_px: float = 2.0,
    transition_weight: float = 4.0,
    balance_weight: float = 0.60,
    center_weight: float = 0.12,
    curvature_weight: float = 0.15,
    smooth_sigma_rows: float = 7.0,
) -> Optional[np.ndarray]:
    """
    Estimate a spine-furrow hint from the detrended back topograph.

    For each usable row, score candidate x locations as a central valley with
    bilateral ridge support instead of just choosing something near the image
    centre. A dynamic-programming pass then selects one continuous path through
    the per-row candidates.
    """
    scipy_ndi = _optional_scipy_ndimage()
    if scipy_ndi is None:
        return None

    valid_mask = valid_mask.astype(bool) & np.isfinite(topograph_hw)
    H, W = topograph_hw.shape
    row_occupancy = valid_mask.sum(axis=1)
    candidate_rows = np.where(row_occupancy > W * 0.12)[0]
    if candidate_rows.size == 0:
        return None

    smoothed_profile = scipy_ndi.gaussian_filter1d(
        topograph_hw.astype(np.float32),
        sigma=float(profile_sigma_px),
        axis=1,
        mode="nearest",
    )
    lateral_second_derivative = scipy_ndi.gaussian_filter1d(
        topograph_hw.astype(np.float32),
        sigma=float(curvature_sigma_px),
        order=2,
        axis=1,
        mode="nearest",
    )

    left_edges = np.full(H, -1, dtype=np.int32)
    right_edges = np.full(H, -1, dtype=np.int32)
    row_centers = np.full(H, np.nan, dtype=np.float32)
    row_widths = np.zeros(H, dtype=np.float32)

    for y in candidate_rows:
        cols = np.where(valid_mask[y])[0]
        left_edges[y] = int(cols[0])
        right_edges[y] = int(cols[-1])
        row_widths[y] = float(cols[-1] - cols[0] + 1)
        row_centers[y] = 0.5 * float(cols[0] + cols[-1])

    torso_width_threshold = float(np.median(row_widths[candidate_rows]) * torso_row_width_ratio)
    track_rows = [int(y) for y in candidate_rows if row_widths[y] >= torso_width_threshold]
    if not track_rows:
        return None

    score_map = np.full((H, W), -np.inf, dtype=np.float32)

    for y in track_rows:
        left = int(left_edges[y])
        right = int(right_edges[y])
        width = float(row_widths[y])
        center_x = float(row_centers[y])

        search_left = max(
            left + int(round(ridge_window_far_ratio * width)),
            int(round(center_x - search_half_width_ratio * width)),
        )
        search_right = min(
            right - int(round(ridge_window_far_ratio * width)),
            int(round(center_x + search_half_width_ratio * width)),
        )
        if search_right <= search_left:
            continue

        for x in range(search_left, search_right + 1):
            left_outer = max(left, x - int(round(ridge_window_far_ratio * width)))
            left_inner = max(left, x - int(round(ridge_window_near_ratio * width)))
            right_inner = min(right, x + int(round(ridge_window_near_ratio * width)))
            right_outer = min(right, x + int(round(ridge_window_far_ratio * width)))
            if left_inner - left_outer < 3 or right_outer - right_inner < 3:
                continue

            valley_value = float(smoothed_profile[y, x])
            left_peak = float(np.max(smoothed_profile[y, left_outer:left_inner + 1]))
            right_peak = float(np.max(smoothed_profile[y, right_inner:right_outer + 1]))

            left_prominence = left_peak - valley_value
            right_prominence = right_peak - valley_value
            shared_prominence = min(left_prominence, right_prominence)
            if shared_prominence <= 0.0:
                continue

            bilateral_balance = abs(left_prominence - right_prominence)
            center_offset = abs(float(x) - center_x) / max(0.5 * width, 1.0)
            valley_curvature = max(float(lateral_second_derivative[y, x]), 0.0)

            score_map[y, x] = (
                shared_prominence
                + curvature_weight * valley_curvature
                - balance_weight * bilateral_balance
                - center_weight * center_offset
            )

    track_rows = [y for y in track_rows if np.isfinite(score_map[y]).any()]
    if not track_rows:
        return None

    previous_cost: Optional[np.ndarray] = None
    previous_xs: Optional[np.ndarray] = None
    backpointers: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for y in track_rows:
        xs = np.where(np.isfinite(score_map[y]))[0]
        data_term = score_map[y, xs]

        if previous_cost is None or previous_xs is None:
            cost = data_term
            prev_idx = np.full(xs.shape, -1, dtype=np.int32)
        else:
            width = max(float(row_widths[y]), 1.0)
            dx = xs[:, None] - previous_xs[None, :]
            transition_penalty = transition_weight * (dx / width) ** 2
            total = data_term[:, None] + previous_cost[None, :] - transition_penalty
            prev_idx = np.argmax(total, axis=1).astype(np.int32)
            cost = total[np.arange(len(xs)), prev_idx]

        backpointers[y] = (xs.astype(np.int32), prev_idx)
        previous_cost = cost.astype(np.float32)
        previous_xs = xs.astype(np.int32)

    if previous_cost is None or previous_xs is None:
        return None

    index = int(np.argmax(previous_cost))
    points: list[tuple[int, int]] = []

    for y in reversed(track_rows):
        xs, prev_idx = backpointers[y]
        points.append((int(xs[index]), int(y)))
        index = int(prev_idx[index])
        if index < 0:
            break

    if len(points) < 2:
        return None

    points.reverse()
    xs_arr = np.asarray([p[0] for p in points], dtype=np.float32)
    ys_arr = np.asarray([p[1] for p in points], dtype=np.float32)
    xs_smooth = scipy_ndi.gaussian_filter1d(xs_arr, sigma=float(smooth_sigma_rows))
    return np.stack([xs_smooth, ys_arr], axis=1)


def draw_polyline_overlay(
    base_rgb: np.ndarray,
    polyline_xy: Optional[np.ndarray],
    *,
    line_rgb: tuple[int, int, int] = (20, 20, 20),
    line_width: int = 4,
) -> np.ndarray:
    if polyline_xy is None or len(polyline_xy) < 2:
        return base_rgb

    image = Image.fromarray(base_rgb)
    draw = ImageDraw.Draw(image)
    draw.line(
        [(float(x), float(y)) for x, y in polyline_xy],
        fill=line_rgb,
        width=int(line_width),
    )
    return np.asarray(image)


def render_oblique_surface_view(
    depth_hw: np.ndarray,
    valid_mask: np.ndarray,
    point_rgb_hw3: np.ndarray,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    yaw_deg: float = 55.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    canvas_h: int = 768,
    canvas_w: int = 512,
    margin_px: int = 24,
    fill_holes: bool = True,
    spine_hint_xy: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Render an oblique orthographic view from the reconstructed back surface.
    """
    valid_mask = valid_mask.astype(bool) & np.isfinite(depth_hw) & (depth_hw > 0)
    if not np.any(valid_mask):
        canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
        return canvas, None

    rows, cols = np.nonzero(valid_mask)
    depth = depth_hw[valid_mask].astype(np.float32)
    colors = point_rgb_hw3[valid_mask].astype(np.uint8)

    X = (cols.astype(np.float32) - float(cx)) * depth / max(float(fx), 1e-6)
    Y = -(rows.astype(np.float32) - float(cy)) * depth / max(float(fy), 1e-6)
    Z = depth
    points = np.stack([X, Y, Z], axis=1)
    points -= np.median(points, axis=0, keepdims=True)

    rotation = _rotation_matrix_xyz(
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
    )
    points_view = points @ rotation.T

    proj_x = points_view[:, 0]
    proj_y = points_view[:, 1]
    view_depth = points_view[:, 2]

    min_x, max_x = float(np.min(proj_x)), float(np.max(proj_x))
    min_y, max_y = float(np.min(proj_y)), float(np.max(proj_y))
    scale = min(
        (canvas_w - 2 * margin_px) / max(max_x - min_x, 1e-6),
        (canvas_h - 2 * margin_px) / max(max_y - min_y, 1e-6),
    )

    px = np.clip(
        np.round((proj_x - min_x) * scale + margin_px).astype(np.int32),
        0,
        canvas_w - 1,
    )
    py = np.clip(
        np.round((max_y - proj_y) * scale + margin_px).astype(np.int32),
        0,
        canvas_h - 1,
    )

    order = np.argsort(view_depth)  # smaller depth = closer in orthographic view
    flat = py[order] * canvas_w + px[order]
    _, first_occurrence = np.unique(flat, return_index=True)
    selected = order[first_occurrence]

    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    canvas[py[selected], px[selected]] = colors[selected]
    occupied[py[selected], px[selected]] = True

    scipy_ndi = _optional_scipy_ndimage()
    if fill_holes and scipy_ndi is not None and np.any(occupied):
        silhouette = scipy_ndi.binary_dilation(occupied, iterations=2)
        _, nearest_idx = scipy_ndi.distance_transform_edt(~occupied, return_indices=True)
        filled = canvas[tuple(nearest_idx)]
        canvas[silhouette & ~occupied] = filled[silhouette & ~occupied]

    spine_projected = None
    if spine_hint_xy is not None and len(spine_hint_xy) >= 2:
        spine_projected = _project_spine_hint_to_oblique(
            spine_hint_xy=spine_hint_xy,
            depth_hw=depth_hw,
            valid_mask=valid_mask,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            rotation=rotation,
            min_x=min_x,
            max_y=max_y,
            scale=scale,
            margin_px=margin_px,
            canvas_h=canvas_h,
            canvas_w=canvas_w,
        )

    return canvas, spine_projected


def _rotation_matrix_xyz(*, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll = np.deg2rad(roll_deg)

    cy, sy = np.cos(yaw), np.sin(yaw)
    cx, sx = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(roll), np.sin(roll)

    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_x @ rot_y


def _project_spine_hint_to_oblique(
    *,
    spine_hint_xy: np.ndarray,
    depth_hw: np.ndarray,
    valid_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rotation: np.ndarray,
    min_x: float,
    max_y: float,
    scale: float,
    margin_px: int,
    canvas_h: int,
    canvas_w: int,
) -> Optional[np.ndarray]:
    sampled_points = []
    H, W = depth_hw.shape

    for x, y in spine_hint_xy:
        ix = int(np.clip(round(float(x)), 0, W - 1))
        iy = int(np.clip(round(float(y)), 0, H - 1))
        if not valid_mask[iy, ix]:
            continue
        depth = float(depth_hw[iy, ix])
        X = (float(ix) - float(cx)) * depth / max(float(fx), 1e-6)
        Y = -(float(iy) - float(cy)) * depth / max(float(fy), 1e-6)
        Z = depth
        sampled_points.append([X, Y, Z])

    if len(sampled_points) < 2:
        return None

    points = np.asarray(sampled_points, dtype=np.float32)
    points -= np.median(points, axis=0, keepdims=True)
    points_view = points @ rotation.T

    px = np.clip(
        np.round((points_view[:, 0] - min_x) * scale + margin_px).astype(np.int32),
        0,
        canvas_w - 1,
    )
    py = np.clip(
        np.round((max_y - points_view[:, 1]) * scale + margin_px).astype(np.int32),
        0,
        canvas_h - 1,
    )
    return np.stack([px.astype(np.float32), py.astype(np.float32)], axis=1)


# ---------------------------------------------------------------------------
# Minimal PLY writer (no open3d)
# ---------------------------------------------------------------------------

def write_ply(path: str, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """
    Write point cloud to a binary little-endian PLY file.

    Args:
        path  : output .ply path
        xyz   : float32 [N, 3]
        rgb   : uint8   [N, 3] or None
    """
    N = len(xyz)
    has_color = rgb is not None

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        if has_color:
            # Build a packed record array: 3×float32 + 3×uint8 = 15 bytes per vertex,
            # with explicit offsets so numpy writes no padding bytes.
            dtype = np.dtype({
                "names":    ["x", "y", "z", "r", "g", "b"],
                "formats":  ["<f4", "<f4", "<f4", "u1", "u1", "u1"],
                "offsets":  [0, 4, 8, 12, 13, 14],
                "itemsize": 15,
            })
            data = np.empty(N, dtype=dtype)
            data["x"] = xyz[:, 0];  data["y"] = xyz[:, 1];  data["z"] = xyz[:, 2]
            data["r"] = rgb[:, 0];  data["g"] = rgb[:, 1];  data["b"] = rgb[:, 2]
            f.write(data.tobytes())
        else:
            f.write(xyz.astype(np.float32).tobytes())


# ---------------------------------------------------------------------------
# Point cloud from depth
# ---------------------------------------------------------------------------

def depth_to_pointcloud(
    depth_hw: np.ndarray,
    image_hw3: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    max_depth: float = 200.0,
    max_depth_factor: float = 5.0,
    filter_outliers: bool = True,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project a depth map into a coloured point cloud.

    Replicates the same filtering as InfiniDepth's build_point_cloud_from_depth_result
    (statistical outlier removal), plus scene-adaptive depth capping to prevent the
    depth range from exploding when the model outputs relative (non-metric) depth.

    Without metric scale alignment (MoGe-2 + RANSAC), relative depth can span 1–200
    while metric depth only spans e.g. 2.8–6.8 m for a typical scene.  Far
    sky/background pixels project to enormous XY values (X≈−100 for Z=200, fx=1024),
    dwarfing the main scene in the point cloud.

    The fix: cap at min(max_depth, max_depth_factor × median_depth).  For the
    default max_depth_factor=5 and a scene median of ≈1.7, the cap is ≈8.5, which
    keeps X inside ±4 — comparable to the metric PLY's ±2 range.

    Args:
        depth_hw         : [H, W] depth values (relative or metric units)
        image_hw3        : [H, W, 3] uint8 RGB (same HW as depth_hw)
        fx, fy, cx, cy   : camera intrinsics in the same pixel space as depth_hw
        max_depth        : hard upper depth cap
        max_depth_factor : adaptive cap = max_depth_factor × median(depth).
                           Set to None or 0 to disable.
        filter_outliers  : run statistical outlier removal on the 3D point cloud
        nb_neighbors     : neighbourhood size for statistical outlier filter
        std_ratio        : points farther than mean + std_ratio*std are removed

    Returns:
        xyz  : float32 [N, 3]
        rgb  : uint8   [N, 3]
    """
    H, W = depth_hw.shape
    rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    d = depth_hw.ravel()
    valid_mask = (d > 0) & np.isfinite(d)

    # Scene-adaptive depth cap: relative depth can span 1–200 without metric scale.
    # Cap at max_depth_factor × median to keep far background from exploding XY coords.
    effective_max = float(max_depth)
    if max_depth_factor and max_depth_factor > 0:
        median_d = float(np.median(d[valid_mask]))
        adaptive_cap = max_depth_factor * median_d
        effective_max = min(effective_max, adaptive_cap)
    valid_mask &= (d < effective_max)

    d = d[valid_mask]
    u = cols.ravel()[valid_mask].astype(np.float32)   # pixel column  (x)
    v = rows.ravel()[valid_mask].astype(np.float32)   # pixel row     (y)

    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d

    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
    rgb = image_hw3.reshape(-1, 3)[valid_mask]

    # Statistical outlier removal — mirrors open3d remove_statistical_outlier
    if filter_outliers and len(xyz) > nb_neighbors:
        keep = _statistical_outlier_mask(xyz, nb_neighbors, std_ratio)
        xyz = xyz[keep]
        rgb = rgb[keep]

    return xyz, rgb


def _statistical_outlier_mask(
    xyz: np.ndarray,
    nb_neighbors: int = 30,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """
    Return a boolean mask keeping inlier points.
    Mirrors open3d's remove_statistical_outlier: for each point compute the mean
    distance to its nb_neighbors nearest neighbours; remove points whose mean
    distance exceeds (global_mean + std_ratio * global_std).

    Uses scipy.spatial.KDTree for fast lookup; falls back to a no-op mask if
    scipy is not available.
    """
    try:
        from scipy.spatial import KDTree
    except ImportError:
        return np.ones(len(xyz), dtype=bool)

    tree = KDTree(xyz)
    # query k+1 because the point itself is always the closest neighbour (dist=0)
    dists, _ = tree.query(xyz, k=nb_neighbors + 1, workers=-1)
    mean_dists = dists[:, 1:].mean(axis=1)   # [N]  — skip self-distance at index 0
    mu, sigma = mean_dists.mean(), mean_dists.std()
    return mean_dists <= (mu + std_ratio * sigma)


# ---------------------------------------------------------------------------
# Default intrinsics
# ---------------------------------------------------------------------------

def default_intrinsics(
    h: int,
    w: int,
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> tuple[float, float, float, float]:
    """Fill missing intrinsics with image-size defaults (FOV ≈ 53°)."""
    default_f = float(max(h, w))
    fx = fx if fx is not None else default_f
    fy = fy if fy is not None else default_f
    cx = cx if cx is not None else w / 2.0
    cy = cy if cy is not None else h / 2.0
    return fx, fy, cx, cy


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

def run_inference(
    onnx_path: str,
    image_path: str,
    output_dir: str,
    input_h: int = 768,
    input_w: int = 1024,
    output_h: Optional[int] = None,
    output_w: Optional[int] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    save_pcd: bool = False,
    scale_factor: float = 1.0,
    max_depth_factor: float = 5.0,
    focus_person: bool = False,
    focus_person_scale: float = 1.0,
    save_back_analysis: bool = False,
    save_spine_hint: bool = False,
    topograph_sigma: float = 4.5,
    curvature_sigma: float = 3.0,
    use_gpu: bool = False,
) -> None:
    import onnxruntime as ort

    save_back_analysis = bool(save_back_analysis or save_spine_hint)

    # --- Session setup ---
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if use_gpu
        else ["CPUExecutionProvider"]
    )
    print(f"Loading ONNX model from {onnx_path} ...")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(onnx_path, providers=providers)
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")

    # --- Image loading ---
    print(f"Loading image {image_path} ...")
    img_batch, orig_h, orig_w = load_and_resize(image_path, input_h, input_w)

    # --- Output resolution ---
    out_h = output_h or input_h
    out_w = output_w or input_w
    query_coord = make_query_grid(out_h, out_w)  # [1, out_h*out_w, 2]

    # --- Intrinsics (scaled to output resolution) ---
    # Scale original-image intrinsics to output resolution
    # If user supplied intrinsics they should be in original-image space.
    scale_x = out_w / orig_w
    scale_y = out_h / orig_h
    fx_scaled, fy_scaled, cx_scaled, cy_scaled = default_intrinsics(orig_h, orig_w, fx, fy, cx, cy)
    fx_out = fx_scaled * scale_x
    fy_out = fy_scaled * scale_y
    cx_out = cx_scaled * scale_x
    cy_out = cy_scaled * scale_y

    # --- Inference ---
    print(f"Running inference (input {input_h}×{input_w}, output {out_h}×{out_w}) ...")
    t1 = time.perf_counter()
    outputs = sess.run(
        ["disparity"],
        {"image": img_batch, "query_coord": query_coord},
    )
    elapsed = time.perf_counter() - t1
    print(f"  Inference done in {elapsed:.1f}s")

    disparity = outputs[0]                          # [1, N, 1]
    depth_flat = disparity_to_depth(disparity)      # [1, N, 1]
    depth_hw = depth_flat[0, :, 0].reshape(out_h, out_w)   # [H, W]

    # Optional uniform depth scaling (use --scale_factor to convert to metric units
    # when you know the ratio; see export docs for how to estimate it via MoGe-2)
    if scale_factor != 1.0:
        depth_hw = depth_hw * float(scale_factor)

    valid_depth_mask = np.isfinite(depth_hw) & (depth_hw > 0)
    if focus_person:
        person_mask, mask_stats = estimate_center_person_mask(
            depth_hw,
            focus_scale=focus_person_scale,
        )
        valid_depth_mask &= person_mask
        depth_hw = np.where(valid_depth_mask, depth_hw, 0.0)
        print(
            "Applied centered-person mask: "
            f"status={mask_stats.get('status')} "
            f"scale={mask_stats.get('focus_scale', float('nan')):.3f} "
            f"threshold={mask_stats.get('threshold', float('nan')):.3f} "
            f"kept_pixels={int(valid_depth_mask.sum()):,}"
        )

    # --- Save outputs ---
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # Raw depth array
    raw_path = os.path.join(output_dir, f"{stem}_depth.npy")
    np.save(raw_path, depth_hw)
    print(f"Saved raw depth: {raw_path}")

    # Colourised depth PNG
    vis_path = os.path.join(output_dir, f"{stem}_depth.png")
    depth_vis = colorize_depth(depth_hw, valid_mask=valid_depth_mask)
    Image.fromarray(depth_vis).save(vis_path)
    print(f"Saved depth visualisation: {vis_path}")

    if focus_person:
        mask_path = os.path.join(output_dir, f"{stem}_person_mask.png")
        Image.fromarray((valid_depth_mask.astype(np.uint8) * 255)).save(mask_path)
        print(f"Saved person mask: {mask_path}")

    if save_back_analysis:
        analysis = compute_back_surface_analysis(
            depth_hw,
            valid_depth_mask,
            topograph_sigma=topograph_sigma,
            curvature_sigma=curvature_sigma,
        )
        topograph = analysis["topograph"]
        curvature = analysis["curvature"]

        topograph_npy_path = os.path.join(output_dir, f"{stem}_topograph.npy")
        curvature_npy_path = os.path.join(output_dir, f"{stem}_curvature.npy")
        np.save(topograph_npy_path, topograph)
        np.save(curvature_npy_path, curvature)
        print(f"Saved topograph array: {topograph_npy_path}")
        print(f"Saved curvature array: {curvature_npy_path}")

        topograph_rgb = colorize_scalar_map(
            topograph,
            valid_depth_mask,
            cmap_name="Spectral_r",
            symmetric=False,
        )
        curvature_rgb = colorize_scalar_map(
            curvature,
            valid_depth_mask,
            cmap_name="seismic",
            symmetric=True,
            upper_percentile=95.0,
        )

        spine_hint_xy = None
        if save_spine_hint:
            spine_hint_xy = estimate_spine_hint(topograph, valid_depth_mask)

        topograph_path = os.path.join(output_dir, f"{stem}_topograph.png")
        curvature_path = os.path.join(output_dir, f"{stem}_curvature.png")
        Image.fromarray(topograph_rgb).save(topograph_path)
        Image.fromarray(curvature_rgb).save(curvature_path)
        print(f"Saved topograph visualisation: {topograph_path}")
        print(f"Saved curvature visualisation: {curvature_path}")

        if spine_hint_xy is not None:
            spine_hint_path = os.path.join(output_dir, f"{stem}_spine_hint.npy")
            np.save(spine_hint_path, spine_hint_xy.astype(np.float32))
            topograph_hint_path = os.path.join(output_dir, f"{stem}_topograph_spine_hint.png")
            curvature_hint_path = os.path.join(output_dir, f"{stem}_curvature_spine_hint.png")
            Image.fromarray(draw_polyline_overlay(topograph_rgb, spine_hint_xy)).save(topograph_hint_path)
            Image.fromarray(draw_polyline_overlay(curvature_rgb, spine_hint_xy)).save(curvature_hint_path)
            print(f"Saved spine hint polyline: {spine_hint_path}")
            print(f"Saved topograph spine-hint visualisation: {topograph_hint_path}")
            print(f"Saved curvature spine-hint visualisation: {curvature_hint_path}")

    # Point cloud (PLY)
    if save_pcd:
        # We need the RGB image at output resolution for colouring
        img_resized_hw3 = (img_batch[0].transpose(1, 2, 0) * 255).astype(np.uint8)
        if img_resized_hw3.shape[:2] != (out_h, out_w):
            pil_rgb = Image.fromarray(img_resized_hw3).resize((out_w, out_h), Image.LANCZOS)
            img_resized_hw3 = np.array(pil_rgb)

        xyz, rgb = depth_to_pointcloud(
            depth_hw, img_resized_hw3,
            fx_out, fy_out, cx_out, cy_out,
            max_depth_factor=max_depth_factor,
        )
        ply_path = os.path.join(output_dir, f"{stem}_pcd.ply")
        print(f"Saving point cloud ({len(xyz):,} points) → {ply_path} ...")
        write_ply(ply_path, xyz, rgb)
        print(f"Saved point cloud: {ply_path}")

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Depth inference with InfiniDepth ONNX model"
    )
    p.add_argument("--onnx",  default="checkpoints/depth/infinidepth.onnx",
                   help="Path to exported ONNX model")
    p.add_argument("--image", required=True,
                   help="Input RGB image path")
    p.add_argument("--out",   default="output",
                   help="Output directory")

    # Resolution
    p.add_argument("--input_h",  type=int, default=768,
                   help="Model input height (must match what was used during export)")
    p.add_argument("--input_w",  type=int, default=1024,
                   help="Model input width (must match what was used during export)")
    p.add_argument("--output_h", type=int, default=None,
                   help="Output depth map height (default: same as input_h)")
    p.add_argument("--output_w", type=int, default=None,
                   help="Output depth map width  (default: same as input_w)")

    # Camera intrinsics (original-image space)
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)

    # Point cloud
    p.add_argument("--save_pcd", action="store_true",
                   help="Save a coloured PLY point cloud alongside the depth map")
    p.add_argument("--scale_factor", type=float, default=1.0,
                   help="Multiply all depth values by this factor before saving. "
                        "Use to convert relative depth to approximate metric (metres). "
                        "The PyTorch pipeline estimates this via MoGe-2 + RANSAC; a rough "
                        "value can be obtained by comparing median depths.")
    p.add_argument("--max_depth_factor", type=float, default=5.0,
                   help="Cap depth at max_depth_factor × median_depth before building the "
                        "point cloud.  Prevents far sky/background pixels from creating "
                        "extreme XY outliers in the PLY.  Set to 0 to disable.")
    p.add_argument("--focus_person", action="store_true",
                   help="Estimate a centered person mask from the depth map and suppress "
                        "background before saving the depth map and point cloud. Intended "
                        "for upright back/back-left/back-right captures.")
    p.add_argument("--focus_person_scale", type=float, default=1.0,
                   help="Scale factor for person-mask aggressiveness. Smaller values tighten "
                        "the focus, larger values loosen it. Default: 1.0.")
    p.add_argument("--save_back_analysis", action="store_true",
                   help="Save a detrended back topograph and a lateral curvature map from "
                        "the depth result. Intended for cropped back-surface analysis.")
    p.add_argument("--save_spine_hint", action="store_true",
                   help="Estimate a spine-furrow hint from the detrended back surface and "
                        "overlay it on the saved topograph/curvature images.")
    p.add_argument("--topograph_sigma", type=float, default=4.5,
                   help="Gaussian smoothing sigma in pixels for the detrended topograph.")
    p.add_argument("--curvature_sigma", type=float, default=3.0,
                   help="Gaussian derivative sigma in pixels for the lateral curvature map.")

    # Runtime
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA execution provider in onnxruntime (requires onnxruntime-gpu)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        onnx_path=args.onnx,
        image_path=args.image,
        output_dir=args.out,
        input_h=args.input_h,
        input_w=args.input_w,
        output_h=args.output_h,
        output_w=args.output_w,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        save_pcd=args.save_pcd,
        scale_factor=args.scale_factor,
        max_depth_factor=args.max_depth_factor,
        focus_person=args.focus_person,
        focus_person_scale=args.focus_person_scale,
        save_back_analysis=args.save_back_analysis,
        save_spine_hint=args.save_spine_hint,
        topograph_sigma=args.topograph_sigma,
        curvature_sigma=args.curvature_sigma,
        use_gpu=args.gpu,
    )

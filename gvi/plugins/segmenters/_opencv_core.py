"""Pure OpenCV/NumPy/Pillow segmentation core for GVI.

Deliberately has ZERO dependency on gvi.core / pydantic so that the actual
computer-vision logic can be unit-tested in isolation, profiled, and reused
by other entry points (CLI, notebook, benchmark script) without booting the
whole plugin framework.

This is the v1.1 rewrite that replaces the v1.0.0 "run every strategy and
keep everything that doesn't overlap too much" approach -- which fixed the
0-element bug on flat synthetic scenes but caused catastrophic
over-segmentation (100-160 spurious boxes) on textured/photographic images.

Pipeline, in order:
  1. classify_scene()         -- flat vs textured/gradient vs alpha-cutout
  2. strategy selection        -- adaptive, not "always everything"
  3. per-candidate plausibility filter -- rejects texture/noise fragments
  4. region merging             -- color+adjacency aware, fights fragmentation
  5. optional GrabCut refinement -- cleans boundaries on flattened photos
  6. optional homography rectification -- de-skews rotated quads
  7. adaptive budget + final ranking
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- data
@dataclass
class Candidate:
    id: str
    element_type: str
    bounds: tuple[int, int, int, int]  # x, y, w, h (axis-aligned, post-rectification)
    confidence: float = 1.0
    asset_path: Path | None = None
    rotated_rect: dict[str, Any] | None = None
    rectified: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- scene analysis
def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        return img[:, :, :3]
    return img


def classify_scene(bgr: np.ndarray, alpha: np.ndarray | None, has_ui_layout: bool = False, has_text: bool = False) -> dict[str, Any]:
    """Returns a richer scene profile than v1.0.0's single label.

    `complexity` in {"flat", "medium", "textured"} drives strategy selection;
    it is estimated from the number of distinct quantized colors in a 64x64
    thumbnail, which separates flat UI/vector art from photographic or
    gradient-heavy content far more reliably than edge density alone
    (verified empirically: wall_of_frames=30 colors, ui_mockup=30, vs.
    map render=151, monster can=109, robot photo=54).
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edge_density = float(np.mean(edges > 0))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat_mean = float(np.mean(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))

    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    quantized = (small // 24).reshape(-1, 3)
    unique_colors = int(len(np.unique(quantized, axis=0)))

    if unique_colors < 45:
        complexity = "flat"
    elif unique_colors < 85:
        complexity = "medium"
    else:
        complexity = "textured"

    if alpha is not None and float(np.mean(alpha < 250)) > 0.05:
        label = "alpha_art"
    elif has_ui_layout or (0.01 < edge_density < 0.16 and val_std < 75 and complexity == "flat"):
        label = "ui_flat"
    elif edge_density > 0.13 or has_text:
        label = "high_contrast"
    elif edge_density < 0.05 and val_std > 40 and sat_mean > 20:
        label = "photo"
    else:
        label = "mixed"

    return {
        "label": label,
        "complexity": complexity,
        "edge_density": edge_density,
        "sat_mean": sat_mean,
        "val_std": val_std,
        "unique_colors_64": unique_colors,
    }


# --------------------------------------------------------------------------- strategies
def _segment_edges_and_rects(bgr, alpha, min_area, close_iters=2) -> list[dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 140)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(cv2.dilate(edges, kernel, iterations=1), cv2.MORPH_CLOSE, kernel, iterations=close_iters)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_raw(bgr, contours, min_area, "edge")


def _segment_threshold(bgr, alpha, min_area) -> list[dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, otsu_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = []
    for mask, label in [(otsu_inv, "threshold_dark"), (otsu, "threshold_light")]:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(_contours_to_raw(bgr, contours, min_area, label))
    return out


def _segment_color_regions(bgr, alpha, min_area, k=8, target_side=640, close_kernel=3) -> list[dict]:
    h, w = bgr.shape[:2]
    scale = min(1.0, target_side / max(w, h))
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA) if scale < 1 else bgr
    pixels = small.reshape(-1, 3).astype(np.float32)
    unique_colors = np.unique(pixels.reshape(-1, 3), axis=0)
    k = min(k, max(2, len(unique_colors)))
    if k < 2:
        return []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.5)
    _, labels, _ = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.reshape(small.shape[:2])
    out = []
    # `close_kernel` > 3 bridges the many small gaps a smooth, continuously
    # shaded surface (glossy plastic, painted metal) creates when its
    # gradient repeatedly crosses a k-means color-bin boundary -- without
    # this, one visual "part" gets quantized into dozens of scattered same-
    # cluster scraps. Scaled relative to the (possibly downsampled) image so
    # the effect is comparable across resolutions.
    ksz = max(3, int(close_kernel * scale)) if scale < 1 else close_kernel
    ksz = ksz if ksz % 2 == 1 else ksz + 1
    close_struct = np.ones((ksz, ksz), np.uint8) if ksz > 3 else None
    for idx in range(k):
        mask_small = (labels == idx).astype(np.uint8) * 255
        if close_struct is not None:
            mask_small = cv2.morphologyEx(mask_small, cv2.MORPH_CLOSE, close_struct, iterations=1)
        mask = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST) if scale < 1 else mask_small
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(_contours_to_raw(bgr, contours, min_area, f"color_{idx}"))
    return out


def _segment_watershed(bgr, alpha, min_area) -> list[dict]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(bgr, markers)
    out = []
    for label in np.unique(markers):
        if label <= 1:
            continue
        mask = (markers == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out.extend(_contours_to_raw(bgr, contours, min_area, "watershed"))
    return out


def _segment_alpha_clusters(bgr, alpha, min_area) -> list[dict]:
    _, alpha_bin = cv2.threshold(alpha, 8, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    alpha_bin = cv2.morphologyEx(alpha_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(alpha_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _contours_to_raw(bgr, contours, min_area, "alpha")


def _contours_to_raw(bgr, contours, min_area, source) -> list[dict]:
    h, w = bgr.shape[:2]
    img_area = h * w
    out = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > img_area * 0.92:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw < 4 or ch < 4 or cw * ch < min_area:
            continue
        if x <= 1 and y <= 1 and cw >= w - 2 and ch >= h - 2:
            continue
        out.append({"contour": contour, "bbox": (x, y, cw, ch), "source": source, "area": area})
    return out


# --------------------------------------------------------------------------- plausibility filter
def _solidity(contour) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    return float(area / hull_area) if hull_area > 0 else 0.0


def _internal_roughness(gray: np.ndarray, mask: np.ndarray, bbox) -> float:
    """0..1 score: how textured/non-homogeneous the pixels under the mask are.

    High roughness on a SMALL region is the signature of a spurious
    fragment carved out of a continuous gradient/texture (a scrap of cloud,
    a paint highlight, a JPEG block) rather than a deliberate flat-colored
    object. This is the single biggest lever against the v1.0.0
    over-segmentation bug.
    """
    x, y, w, h = bbox
    roi = gray[y:y + h, x:x + w]
    roi_mask = mask[y:y + h, x:x + w]
    if roi.size == 0 or np.count_nonzero(roi_mask) < 9:
        return 0.0
    vals = roi[roi_mask > 0]
    return float(np.std(vals) / 80.0)  # ~80 std = "fully noisy", clamp later


def is_plausible(contour, mask, bbox, gray, img_area) -> tuple[bool, float]:
    x, y, w, h = bbox
    area = cv2.contourArea(contour)
    area_ratio = area / max(img_area, 1)
    solidity = _solidity(contour)
    roughness = min(1.0, _internal_roughness(gray, mask, bbox))

    # Large regions are kept even if a bit rough (they're probably a real
    # panel/background slab, just not perfectly flat-colored).
    if area_ratio > 0.06:
        plausible = solidity > 0.35
    elif area_ratio > 0.015:
        plausible = solidity > 0.45 and roughness < 0.55
    else:
        # Small fragments must be both solid AND internally homogeneous,
        # otherwise they are almost certainly texture noise.
        plausible = solidity > 0.55 and roughness < 0.35

    score = (0.5 * solidity + 0.5 * (1.0 - roughness)) * min(1.0, area_ratio * 12)
    return plausible, float(max(0.05, min(0.99, score)))


# --------------------------------------------------------------------------- merging
def _box_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1)


def _box_overlap_fraction(a, b) -> float:
    """Fraction of the SMALLER box covered by the intersection (catches
    containment / heavy overlap that low-IoU would miss for very
    different-sized boxes)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    xi1, yi1 = max(ax, bx), max(ay, by)
    xi2, yi2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    inter = (xi2 - xi1) * (yi2 - yi1)
    return inter / max(1, min(aw * ah, bw * bh))


def _mean_color(bgr, bbox) -> np.ndarray:
    x, y, w, h = bbox
    roi = bgr[y:y + h, x:x + w]
    if roi.size == 0:
        return np.zeros(3)
    return roi.reshape(-1, 3).mean(axis=0)


def _mean_color_masked(bgr, contour, bbox) -> np.ndarray:
    """Mean color over the contour's actual interior, not its bounding box.

    A sparse/elongated shape (a logo mark, thin text) can have a bbox that's
    mostly background -- comparing bbox-average colors in that case
    incorrectly says "these are similar" for any two large overlapping boxes
    that both happen to mostly contain the same backdrop, even when the
    actual shapes inside them are completely different colors. Found by
    actually debugging why a can's body and its distinct logo mark were
    being merged into one element.
    """
    x, y, w, h = bbox
    h_img, w_img = bgr.shape[:2]
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    roi_mask = mask[y:y + h, x:x + w]
    roi_bgr = bgr[y:y + h, x:x + w]
    if roi_bgr.size == 0 or np.count_nonzero(roi_mask) < 4:
        return _mean_color(bgr, bbox)
    return roi_bgr[roi_mask > 0].reshape(-1, 3).mean(axis=0)


def _expand_box(box, margin):
    x, y, w, h = box
    return (x - margin, y - margin, w + 2 * margin, h + 2 * margin)


def merge_fragments(raw: list[dict], bgr: np.ndarray, iou_merge=0.12, overlap_merge=0.55, color_tol=28.0, proximity_px=0) -> list[dict]:
    """Union-find merge of candidates that touch/overlap (optionally within
    `proximity_px` of each other) AND have a similar mean color. This is
    what turns "40 little scraps of the same painted surface" into "1
    region", instead of relying on plain IoU dedup which only removes
    near-duplicates, not genuine fragmentation.

    `proximity_px` lets nearby-but-not-touching fragments merge too -- useful
    on glossy/reflective surfaces (product photos, toys) where specular
    highlights split what is visually one part into several disjoint blobs
    a few pixels apart.
    """
    n = len(raw)
    if n == 0:
        return raw
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    colors = [_mean_color_masked(bgr, r["contour"], r["bbox"]) for r in raw]
    boxes = [_expand_box(r["bbox"], proximity_px) if proximity_px else r["bbox"] for r in raw]
    for i in range(n):
        for j in range(i + 1, n):
            iou = _box_iou(boxes[i], boxes[j])
            overlap = _box_overlap_fraction(boxes[i], boxes[j])
            if iou < iou_merge and overlap < overlap_merge:
                continue
            color_dist = float(np.linalg.norm(colors[i] - colors[j]))
            if color_dist <= color_tol:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for idx_list in groups.values():
        if len(idx_list) == 1:
            merged.append(raw[idx_list[0]])
            continue
        xs1 = min(raw[i]["bbox"][0] for i in idx_list)
        ys1 = min(raw[i]["bbox"][1] for i in idx_list)
        xs2 = max(raw[i]["bbox"][0] + raw[i]["bbox"][2] for i in idx_list)
        ys2 = max(raw[i]["bbox"][1] + raw[i]["bbox"][3] for i in idx_list)
        biggest = max(idx_list, key=lambda i: raw[i]["area"])
        merged.append({
            "contour": raw[biggest]["contour"],
            "bbox": (xs1, ys1, xs2 - xs1, ys2 - ys1),
            "source": raw[biggest]["source"] + "+merged",
            "area": sum(raw[i]["area"] for i in idx_list),
            "merged_from": len(idx_list),
        })
    return merged


# --------------------------------------------------------------------------- specular highlight suppression
def suppress_specular_highlights(bgr: np.ndarray, max_area_ratio: float = 0.01) -> np.ndarray:
    """Inpaint small, very bright/desaturated blobs (specular highlights on
    glossy plastic/metal/paint) before strategy detection runs.

    This does NOT touch the image used for final crop extraction -- only
    the copy handed to the edge/color/watershed strategies. The goal is to
    stop a highlight from being picked up as its own "different color"
    region in the first place, which is cheaper and more reliable than
    trying to merge it back in after the fact (the v1.1.0 color+proximity
    merge already does some of that, but pre-suppression catches cases the
    merge's distance/color thresholds miss).
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v, s = hsv[:, :, 2], hsv[:, :, 1]
    highlight_mask = ((v > 232) & (s < 50)).astype(np.uint8) * 255
    h, w = bgr.shape[:2]
    img_area = h * w
    if img_area == 0:
        return bgr
    # Only inpaint SMALL highlight blobs -- a genuinely huge bright flat
    # region (e.g. a white UI panel) is real content, not a speck.
    contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    keep_mask = np.zeros_like(highlight_mask)
    for c in contours:
        area = cv2.contourArea(c)
        if 0 < area <= img_area * max_area_ratio:
            cv2.drawContours(keep_mask, [c], -1, 255, -1)
    if not np.any(keep_mask):
        return bgr
    keep_mask = cv2.dilate(keep_mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(bgr, keep_mask, 5, cv2.INPAINT_TELEA)


# --------------------------------------------------------------------------- GrabCut refinement
def grabcut_refine(bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Refine a bounding-box-only candidate into a real foreground mask.

    Only worth running on flattened (non-alpha) photographic content where
    the initial mask is a crude rectangle/contour and the true object
    boundary is ambiguous. Returns a 0/255 mask the size of bbox, or None if
    GrabCut fails/diverges (e.g. region too small or too uniform).
    """
    x, y, w, h = bbox
    if w < 12 or h < 12:
        return None
    pad = max(2, int(0.06 * max(w, h)))
    H, W = bgr.shape[:2]
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    roi = bgr[y0:y1, x0:x1]
    if roi.size == 0:
        return None
    mask = np.zeros(roi.shape[:2], np.uint8)
    rect = (x - x0, y - y0, w, h)
    rect = (max(0, rect[0]), max(0, rect[1]), min(rect[2], roi.shape[1] - 1), min(rect[3], roi.shape[0] - 1))
    if rect[2] <= 1 or rect[3] <= 1:
        return None
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(roi, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = np.where((mask == 1) | (mask == 3), 255, 0).astype(np.uint8)
    full = np.zeros((H, W), np.uint8)
    full[y0:y1, x0:x1] = fg
    return full[y:y + h, x:x + w]


# --------------------------------------------------------------------------- homography rectification
def _normalize_angle(angle: float, w: float, h: float) -> float:
    # cv2.minAreaRect angle convention varies by OpenCV version; normalize to
    # "how far from axis-aligned, in [-45, 45]".
    a = angle % 90.0
    if a > 45.0:
        a -= 90.0
    return a


def rectify_if_rotated(bgr: np.ndarray, alpha: np.ndarray | None, contour, min_angle_deg=4.0) -> dict | None:
    """If the contour's minimum-area rectangle is meaningfully rotated AND
    reasonably rectangular (a "tilted frame/panel" rather than an organic
    blob), warp it to an upright rectangle via a 4-point homography instead
    of taking a wasteful/skewed axis-aligned crop.

    Returns {"image": RGBA ndarray, "rect": rrect-dict} or None if rectification
    isn't applicable (not rotated enough, or not rectangular enough).
    """
    rrect = cv2.minAreaRect(contour)
    (cx, cy), (rw, rh), angle = rrect
    if rw < 6 or rh < 6:
        return None
    norm_angle = _normalize_angle(angle, rw, rh)
    if abs(norm_angle) < min_angle_deg:
        return None  # not worth rectifying

    area = cv2.contourArea(contour)
    rect_area = rw * rh
    if rect_area <= 0 or area / rect_area < 0.55:
        return None  # not rectangular enough to assume it's a tilted panel

    box = cv2.boxPoints(rrect)
    # order: top-left, top-right, bottom-right, bottom-left
    s = box.sum(axis=1)
    diff = box[:, 0] - box[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = box[np.argmin(s)]
    ordered[2] = box[np.argmax(s)]
    ordered[1] = box[np.argmax(diff)]
    ordered[3] = box[np.argmin(diff)]

    out_w, out_h = max(2, int(round(rw))), max(2, int(round(rh)))
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped_bgr = cv2.warpPerspective(bgr, matrix, (out_w, out_h))

    if alpha is not None:
        warped_alpha = cv2.warpPerspective(alpha, matrix, (out_w, out_h))
    else:
        mask = np.zeros(bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        warped_alpha = cv2.warpPerspective(mask, matrix, (out_w, out_h))

    rgba = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = warped_alpha
    return {
        "image": rgba,
        "rect": {"center": [float(cx), float(cy)], "size": [float(rw), float(rh)], "angle_deg": float(norm_angle)},
    }


# --------------------------------------------------------------------------- element extraction
def classify_element(aspect: float, extent: float, area_ratio: float, source: str) -> str:
    if area_ratio > 0.25:
        return "panel"
    if aspect > 4.0 and extent > 0.20:
        return "panel"
    if aspect > 1.8 and extent > 0.35:
        return "frame"
    if 0.6 <= aspect <= 1.8 and extent > 0.65 and "color" in source:
        return "button"
    if extent < 0.25:
        return "decoration"
    return "sprite"


def extract_candidate(
    bgr: np.ndarray,
    alpha: np.ndarray | None,
    raw: dict,
    gray: np.ndarray,
    output_dir: Path,
    use_grabcut: bool,
    use_rectify: bool,
) -> Candidate | None:
    h, w = bgr.shape[:2]
    x, y, cw, ch = raw["bbox"]
    contour = raw["contour"]

    rectified_payload = None
    if use_rectify and alpha is None:
        # Rectification only makes sense before we've already decided a
        # plain axis-aligned crop; alpha-art assets are virtually always
        # authored upright, so we restrict this to flattened photo content.
        rectified_payload = rectify_if_rotated(bgr, alpha, contour)

    if rectified_payload is not None:
        rgba = rectified_payload["image"]
        pil_img = Image.fromarray(rgba)
        bbox_trim = pil_img.getbbox()
        if not bbox_trim:
            return None
        trimmed = pil_img.crop(bbox_trim)
        new_w, new_h = trimmed.size
        new_x, new_y = x, y  # rectified crops keep the source bbox as anchor
        area = int(np.count_nonzero(np.array(trimmed)[:, :, 3] > 0))
        source = raw["source"] + "+rectified"
        rotated_rect = rectified_payload["rect"]
    else:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        if use_grabcut and alpha is None:
            refined = grabcut_refine(bgr, (x, y, cw, ch))
            if refined is not None and np.count_nonzero(refined) > 0.15 * cw * ch:
                mask[y:y + ch, x:x + cw] = np.maximum(mask[y:y + ch, x:x + cw], refined)

        x2, y2 = min(w, x + cw), min(h, y + ch)
        if x2 <= x or y2 <= y:
            return None
        roi_bgr = bgr[y:y2, x:x2]
        roi_mask = mask[y:y2, x:x2]
        if alpha is not None:
            roi_mask = np.minimum(roi_mask, alpha[y:y2, x:x2])
        rgba = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = roi_mask
        pil_img = Image.fromarray(rgba)
        bbox_trim = pil_img.getbbox()
        if not bbox_trim:
            return None
        trimmed = pil_img.crop(bbox_trim)
        tx, ty, tx2, ty2 = bbox_trim
        new_x, new_y = x + tx, y + ty
        new_w, new_h = tx2 - tx, ty2 - ty
        area = int(np.sum(mask > 0))
        source = raw["source"]
        rrect = cv2.minAreaRect(contour)
        rotated_rect = {"center": [float(rrect[0][0]), float(rrect[0][1])], "size": [float(rrect[1][0]), float(rrect[1][1])], "angle_deg": float(rrect[2])}

    if new_w <= 2 or new_h <= 2:
        return None

    element_id = f"elem_{uuid.uuid4().hex[:8]}"
    asset_path = output_dir / f"{element_id}.png"
    trimmed.save(asset_path, "PNG")

    extent = area / max(new_w * new_h, 1)
    aspect = max(new_w, new_h) / max(min(new_w, new_h), 1)
    elem_type = classify_element(aspect, extent, area / max(h * w, 1), source)

    return Candidate(
        id=element_id,
        element_type=elem_type,
        bounds=(int(new_x), int(new_y), int(new_w), int(new_h)),
        confidence=float(max(0.25, min(0.98, extent))),
        asset_path=asset_path,
        rotated_rect=rotated_rect,
        rectified=rectified_payload is not None,
        metadata={"area": area, "extent": float(extent), "aspect_ratio": float(aspect), "source": source},
    )


def _final_dedupe(elements: list[Candidate], iou_threshold: float, containment_threshold: float = 0.85) -> list[Candidate]:
    def area(e: Candidate) -> int:
        return e.bounds[2] * e.bounds[3]

    kept: list[Candidate] = []
    for elem in sorted(elements, key=lambda e: (e.confidence, area(e)), reverse=True):
        ok = True
        for other in kept:
            ax, ay, aw, ah = elem.bounds
            bx, by, bw, bh = other.bounds
            xi1, yi1 = max(ax, bx), max(ay, by)
            xi2, yi2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            if xi2 <= xi1 or yi2 <= yi1:
                continue
            inter = (xi2 - xi1) * (yi2 - yi1)
            union = aw * ah + bw * bh - inter
            iou = inter / max(union, 1)
            # Same-type containment catches "the same big region detected
            # twice at slightly different extents" -- a plain IoU check
            # misses this when the two boxes differ a lot in aspect ratio.
            # Gated on comparable size so a genuinely small nested detail
            # (a logo mark inside a can body, both classified "panel") is
            # NOT swallowed by its much bigger parent -- only true near-
            # duplicates of similar size get collapsed this way.
            contained = inter / max(1, min(aw * ah, bw * bh))
            size_ratio = min(aw * ah, bw * bh) / max(1, max(aw * ah, bw * bh))
            same_type = elem.element_type == other.element_type
            if iou >= iou_threshold or (same_type and contained >= containment_threshold and size_ratio >= 0.5):
                ok = False
                break
        if ok:
            kept.append(elem)
    return kept


# --------------------------------------------------------------------------- orchestration
def run_segmentation(
    img: np.ndarray,
    output_dir: Path,
    preset: str = "balanced",
    min_area_ratio: float = 0.0001,
    max_elements: int | None = None,
    merge_overlaps: bool = True,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    h, w = img.shape[:2]
    has_alpha = img.ndim == 3 and img.shape[2] == 4
    bgr = to_bgr(img)
    alpha = img[:, :, 3] if has_alpha else None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    img_area = h * w
    min_area = max(24, int(img_area * min_area_ratio))

    scene = classify_scene(bgr, alpha)
    label, complexity = scene["label"], scene["complexity"]

    # Detection-only despecularized copy: glossy/reflective surfaces (toy
    # photos, painted products) scatter specular highlights into their own
    # spurious "color" regions. Suppressing them before strategy detection
    # (NOT before final crop extraction, which still reads `bgr`) measurably
    # reduces fragment count on exactly this kind of image -- see
    # docs/CHANGELOG_v1.1.md for the before/after numbers.
    detect_bgr = bgr
    if complexity in {"medium", "textured"}:
        detect_bgr = suppress_specular_highlights(bgr)

    # ---- 1) adaptive strategy selection (the core fix) ---------------------
    raw: list[dict] = []
    if complexity == "flat":
        # Flat UI / vector-style art: the cheap, precise strategies are
        # enough and additional ones would only add noise.
        raw += _segment_edges_and_rects(detect_bgr, alpha, min_area)
        raw += _segment_threshold(detect_bgr, alpha, min_area)
        if label == "ui_flat":
            raw += _segment_color_regions(detect_bgr, alpha, min_area * 2, k=8)
    elif complexity == "medium":
        raw += _segment_edges_and_rects(detect_bgr, alpha, min_area, close_iters=3)
        raw += _segment_color_regions(detect_bgr, alpha, min_area * 4, k=4, close_kernel=21)
        if label != "alpha_art":
            raw += _segment_threshold(detect_bgr, alpha, min_area * 2)
    else:  # textured / photographic / gradient-heavy
        # Coarse-only: a handful of large coherent regions, not a fragment
        # storm. Watershed and per-pixel threshold are intentionally
        # skipped here -- they are the two biggest sources of micro-noise
        # on continuous-tone content.
        raw += _segment_edges_and_rects(detect_bgr, alpha, min_area * 6, close_iters=4)
        raw += _segment_color_regions(detect_bgr, alpha, min_area * 10, k=5, target_side=384, close_kernel=3)

    if has_alpha and label != "ui_flat":
        raw += _segment_alpha_clusters(bgr, alpha, min_area)

    # Watershed only for genuinely flat/medium scenes -- on textured photos
    # it is the single largest contributor to spurious fragments.
    if complexity in {"flat", "medium"} and label not in {"alpha_art"}:
        raw += _segment_watershed(detect_bgr, alpha, min_area * (2 if complexity == "flat" else 4))

    # ---- 2) plausibility filter --------------------------------------------
    plausible_raw = []
    for r in raw:
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [r["contour"]], -1, 255, -1)
        ok, score = is_plausible(r["contour"], mask, r["bbox"], gray, img_area)
        if ok:
            r["plausibility"] = score
            plausible_raw.append(r)

    # ---- 3) merge fragments (color + adjacency aware) ----------------------
    merge_iou = 0.12 if merge_overlaps else 0.95
    # Glossy/reflective surfaces (product photos, painted toys) scatter one
    # visual part into several disjoint, slightly-different-colored blobs
    # (specular highlights). Be more permissive on color and allow merging
    # across a small pixel gap for anything that isn't flat UI art.
    if complexity == "flat":
        color_tol, proximity_px = 28.0, 0
    elif complexity == "medium":
        color_tol, proximity_px = 38.0, 4
    else:
        color_tol, proximity_px = 28.0, 4
    merged_raw = merge_fragments(plausible_raw, bgr, iou_merge=merge_iou, color_tol=color_tol, proximity_px=proximity_px)

    # ---- 4) extract crops (+ optional GrabCut / homography) ----------------
    use_grabcut = complexity != "flat" and not has_alpha
    use_rectify = True
    candidates: list[Candidate] = []
    for r in merged_raw:
        elem = extract_candidate(bgr, alpha, r, gray, output_dir, use_grabcut=use_grabcut, use_rectify=use_rectify)
        if elem:
            candidates.append(elem)

    dedupe_iou = 0.55 if merge_overlaps else 0.95
    if complexity != "flat":
        dedupe_iou = 0.35  # textured/medium scenes tend to leave near-duplicate
        # giant panels behind after merge; be stricter about collapsing them.
    candidates = _final_dedupe(candidates, iou_threshold=dedupe_iou)
    candidates.sort(key=lambda e: (e.confidence, e.bounds[2] * e.bounds[3]), reverse=True)

    # ---- 5) adaptive budget --------------------------------------------------
    if max_elements is None:
        budget_by_label = {"ui_flat": 60, "high_contrast": 45, "alpha_art": 30, "photo": 20, "mixed": 25}
        max_elements = budget_by_label.get(label, 25)
    candidates = candidates[:max_elements]
    candidates.sort(key=lambda e: (e.bounds[1], e.bounds[0], -e.bounds[2] * e.bounds[3]))

    mask_union = np.zeros((h, w), dtype=np.uint8)
    for elem in candidates:
        x, y, ew, eh = elem.bounds
        mask_union[y:y + eh, x:x + ew] = 255
    coverage_ratio = float(np.mean(mask_union > 0)) if candidates else 0.0

    return {
        "elements": candidates,
        "scene": scene,
        "num_elements": len(candidates),
        "num_raw_candidates": len(raw),
        "num_after_plausibility": len(plausible_raw),
        "num_after_merge": len(merged_raw),
        "coverage_ratio": coverage_ratio,
        "max_elements_budget": max_elements,
    }

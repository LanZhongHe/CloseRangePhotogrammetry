"""Coarse detection of ring-shaped control points.

Finds candidate ring contours in a binarized image using geometric
filters (area, circularity, aspect ratio, convexity) and ring-structure
verification (parent-child contour hierarchy).  Also excludes number-ID
regions that sit below each ring.

Supports both thin rings (with a visible inner hole) and thick rings
(where the inner hole may be absent or very small after binarization).
"""

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class CandidateRing:
    """A detected ring candidate before subpixel refinement."""
    outer_contour: np.ndarray
    inner_contour: Optional[np.ndarray]
    bounding_rect: tuple          # (x, y, w, h)
    center_approx: tuple          # (cx, cy) approximate center
    area: float
    circularity: float
    aspect_ratio: float
    is_thick: bool = False        # True if detected as thick/solid ring


@dataclass
class DetectionParams:
    """Parameters for coarse detection, auto-derived from target size."""
    target_size_px: int = 100

    area_tolerance: float = 0.5     # ±50% area tolerance
    circularity_min: float = 0.65
    aspect_min: float = 0.5
    aspect_max: float = 1.8
    convexity_min: float = 0.85
    min_contour_points: int = 40
    inner_hole_ratio_min: float = 0.02  # min inner/outer area ratio for thin rings
    inner_hole_ratio_max: float = 0.9   # max inner/outer area ratio for thin rings

    @property
    def area_expected(self) -> float:
        """Expected contour area based on target bounding circle."""
        r = self.target_size_px / 2.0
        return np.pi * r ** 2

    @property
    def area_min(self) -> float:
        return self.area_expected * (1.0 - self.area_tolerance)

    @property
    def area_max(self) -> float:
        return self.area_expected * (1.0 + self.area_tolerance)

    @property
    def min_dist_between_targets(self) -> float:
        return self.target_size_px * 0.8


def _circularity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    return 4.0 * np.pi * area / (peri * peri)


def _convexity(contour: np.ndarray) -> float:
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return area / hull_area


def _aspect_ratio(contour: np.ndarray) -> float:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w < 1 or h < 1:
        return 0.0
    return min(w, h) / max(w, h)


def _make_candidate(contour, child_contour, is_thick=False) -> Optional[CandidateRing]:
    """Build a CandidateRing from a contour and optional child."""
    x, y, w, h = cv2.boundingRect(contour)
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return CandidateRing(
        outer_contour=contour,
        inner_contour=child_contour,
        bounding_rect=(x, y, w, h),
        center_approx=(cx, cy),
        area=cv2.contourArea(contour),
        circularity=_circularity(contour),
        aspect_ratio=_aspect_ratio(contour),
        is_thick=is_thick,
    )


def detect_candidates(
    binary_inv: np.ndarray,
    params: DetectionParams | None = None,
) -> List[CandidateRing]:
    """Detect ring candidates from an inverted binary image.

    Supports two detection strategies:
      1. Thin rings: contour has a child (inner hole) in the hierarchy.
      2. Thick rings: solid-looking contour with no child, but passes
         geometric filters (high circularity, correct size).

    Args:
        binary_inv: Binary image where ring regions are white (foreground).
        params: Detection parameters.

    Returns:
        List of CandidateRing passing all geometric filters.
    """
    if params is None:
        params = DetectionParams()

    contours, hierarchy = cv2.findContours(
        binary_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]  # shape: (N, 1, 4) -> (N, 4)

    # --- build child map: parent_idx -> list of child indices ---
    children_map: dict[int, list[int]] = {}
    for i, h in enumerate(hierarchy):
        parent_idx = h[3]
        if parent_idx >= 0:
            children_map.setdefault(parent_idx, []).append(i)

    candidates: List[CandidateRing] = []

    for i, contour in enumerate(contours):
        if len(contour) < params.min_contour_points:
            continue

        area = cv2.contourArea(contour)
        if area < params.area_min or area > params.area_max:
            continue

        circ = _circularity(contour)
        if circ < params.circularity_min:
            continue

        ar = _aspect_ratio(contour)
        if ar < params.aspect_min or ar > params.aspect_max:
            continue

        conv = _convexity(contour)
        if conv < params.convexity_min:
            continue

        # --- check for child contours (inner hole = thin ring) ---
        child_indices = children_map.get(i, [])
        has_valid_hole = False
        best_child = None
        best_child_area = 0

        for ci in child_indices:
            ca = cv2.contourArea(contours[ci])
            ratio = ca / area if area > 0 else 0
            if params.inner_hole_ratio_min < ratio < params.inner_hole_ratio_max:
                has_valid_hole = True
                if ca > best_child_area:
                    best_child_area = ca
                    best_child = ci

        if has_valid_hole and best_child is not None:
            # Thin ring with visible inner hole
            cand = _make_candidate(contour, contours[best_child], is_thick=False)
            if cand:
                candidates.append(cand)
        elif not child_indices:
            # No children at all — could be a thick/solid ring.
            # Accept if circularity is high enough (very round).
            if circ >= 0.80:
                cand = _make_candidate(contour, None, is_thick=True)
                if cand:
                    candidates.append(cand)
        # else: has children but none are valid holes → skip (noise)

    # --- exclude number-ID regions ---
    candidates = _exclude_number_regions(candidates, params)

    # --- remove overlapping detections ---
    candidates = _remove_overlaps(candidates, params)

    return candidates


def _exclude_number_regions(
    candidates: List[CandidateRing],
    params: DetectionParams,
) -> List[CandidateRing]:
    """Remove contours that look like number-ID regions sitting below a ring."""
    ring_centers = [c.center_approx for c in candidates]

    filtered = []
    for c in candidates:
        cx, cy = c.center_approx

        is_number = False
        for rcx, rcy in ring_centers:
            vertical_offset = cy - rcy
            if 0 < vertical_offset < params.target_size_px * 0.9:
                horizontal_dist = abs(cx - rcx)
                if horizontal_dist < params.target_size_px * 0.4:
                    if c.area < params.area_expected * 0.3:
                        is_number = True
                        break
        if not is_number:
            filtered.append(c)

    return filtered


def _remove_overlaps(
    candidates: List[CandidateRing],
    params: DetectionParams,
) -> List[CandidateRing]:
    """Remove duplicate detections of the same ring (keep higher circularity)."""
    if len(candidates) <= 1:
        return candidates

    candidates = sorted(candidates, key=lambda c: c.circularity, reverse=True)
    keep: list[bool] = [True] * len(candidates)
    min_dist = params.min_dist_between_targets

    for i in range(len(candidates)):
        if not keep[i]:
            continue
        cxi, cyi = candidates[i].center_approx
        for j in range(i + 1, len(candidates)):
            if not keep[j]:
                continue
            cxj, cyj = candidates[j].center_approx
            dist = np.hypot(cxi - cxj, cyi - cyj)
            if dist < min_dist:
                keep[j] = False

    return [c for c, k in zip(candidates, keep) if k]

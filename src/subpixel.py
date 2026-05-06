"""Subpixel center localization for ring targets.

Provides two localization methods:
  1. Ellipse fitting on ring contours (~0.1 px)
  2. Grayscale-weighted centroid refinement (~0.02 px)

The main entry point `localize_target` chains these methods from
coarse to fine.
"""

import cv2
import numpy as np

from .data_model import EllipseInfo, TargetPoint
from .detection import CandidateRing


def ellipse_fit(contour: np.ndarray) -> tuple[float, float, EllipseInfo]:
    """Fit an ellipse to a contour.

    Returns:
        (center_x, center_y, EllipseInfo)
    """
    ellipse = cv2.fitEllipse(contour)
    (cx, cy), (d1, d2), angle = ellipse
    semi_major = max(d1, d2) / 2.0
    semi_minor = min(d1, d2) / 2.0
    return cx, cy, EllipseInfo(semi_major, semi_minor, angle)


def eccentricity(ellipse: EllipseInfo) -> float:
    """Compute eccentricity from ellipse axes."""
    if ellipse.semi_major <= 0:
        return 0.0
    return np.sqrt(1.0 - (ellipse.semi_minor / ellipse.semi_major) ** 2)


def centroid_refine(
    gray: np.ndarray,
    center_x: float,
    center_y: float,
    radius: int = 20,
) -> tuple[float, float]:
    """Refine center using grayscale-weighted centroid in a small ROI.

    Args:
        gray: Full grayscale image.
        center_x: Approximate center x.
        center_y: Approximate center y.
        radius: Half-size of the ROI.

    Returns:
        (refined_x, refined_y)
    """
    cx_int = int(round(center_x))
    cy_int = int(round(center_y))
    h, w = gray.shape[:2]

    x1 = max(0, cx_int - radius)
    y1 = max(0, cy_int - radius)
    x2 = min(w, cx_int + radius)
    y2 = min(h, cy_int + radius)

    roi = gray[y1:y2, x1:x2].astype(np.float64)
    if roi.size == 0:
        return center_x, center_y

    # Invert: dark target becomes bright for weighting
    weights = 255.0 - roi

    yy, xx = np.mgrid[0:roi.shape[0], 0:roi.shape[1]].astype(np.float64)
    total = weights.sum()
    if total < 1e-10:
        return center_x, center_y

    fine_cx = np.sum(xx * weights) / total + x1
    fine_cy = np.sum(yy * weights) / total + y1

    return float(fine_cx), float(fine_cy)


def localize_target(
    gray: np.ndarray,
    candidate: CandidateRing,
) -> TargetPoint:
    """Run the localization pipeline on one candidate.

    Pipeline:
        1. Ellipse fit (outer + inner contours) -> weighted center
        2. Centroid refinement

    Args:
        gray: Grayscale image (after preprocessing CLAHE/bilateral).
        candidate: Detected ring candidate.

    Returns:
        TargetPoint with coordinates and metadata (id left empty).
    """
    # --- Level 1: Ellipse fit ---
    cx1, cy1, ellipse_info = ellipse_fit(candidate.outer_contour)
    ecc = eccentricity(ellipse_info)

    # If inner contour exists, also fit it and average
    if candidate.inner_contour is not None and len(candidate.inner_contour) >= 5:
        cx2, cy2, inner_ellipse = ellipse_fit(candidate.inner_contour)
        # Weight outer more heavily (larger contour = more stable)
        w_out = ellipse_info.semi_major
        w_in = inner_ellipse.semi_major
        w_total = w_out + w_in
        cx_ell = (cx1 * w_out + cx2 * w_in) / w_total
        cy_ell = (cy1 * w_out + cy2 * w_in) / w_total
    else:
        cx_ell, cy_ell = cx1, cy1

    method = "ellipse"
    cx_final, cy_final = cx_ell, cy_ell
    confidence = 0.7 + 0.3 * candidate.circularity

    # --- Level 2: Centroid refinement ---
    refine_radius = max(10, int(ellipse_info.semi_minor * 0.3))
    cx_ref, cy_ref = centroid_refine(gray, cx_final, cy_final, refine_radius)

    dist_ref = np.hypot(cx_ref - cx_final, cy_ref - cy_final)
    if dist_ref < ellipse_info.semi_minor * 0.2:
        cx_final, cy_final = cx_ref, cy_ref
        method = "centroid"

    return TargetPoint(
        id="",
        pixel_x=cx_final,
        pixel_y=cy_final,
        confidence=round(confidence, 4),
        source="auto",
        subpixel_method=method,
        ellipse=ellipse_info,
        eccentricity=round(ecc, 4),
    )

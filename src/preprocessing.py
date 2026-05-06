"""Image preprocessing module for control point detection.

Handles grayscale conversion, CLAHE contrast enhancement, bilateral
filtering, and adaptive binarization. All parameters auto-derive from
the known target pixel size.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessingParams:
    """Parameters derived from the known target pixel size."""
    target_size_px: int = 200           # approximate target diameter in pixels
    clahe_clip_limit: float = 2.0
    bilateral_d: int = 0                # 0 = auto from target size
    adaptive_block: int = 0             # 0 = auto from target size
    adaptive_c: int = 5                 # constant subtracted from mean

    def __post_init__(self):
        if self.bilateral_d <= 0:
            self.bilateral_d = max(5, self.target_size_px // 20)
        if self.adaptive_block <= 0:
            self.adaptive_block = int(self.target_size_px * 1.5)
            if self.adaptive_block % 2 == 0:
                self.adaptive_block += 1
        self.adaptive_block = max(3, self.adaptive_block)


def preprocess(image: np.ndarray, params: PreprocessingParams | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
    """Run the full preprocessing pipeline.

    Args:
        image: Input BGR or grayscale image.
        params: Preprocessing parameters (auto-generated if None).

    Returns:
        (gray, binary_inv) where binary_inv has the dark ring as white
        foreground (THRESH_BINARY_INV).
    """
    if params is None:
        params = PreprocessingParams()

    # --- grayscale ---
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # --- CLAHE contrast enhancement ---
    tile = max(8, params.target_size_px // 25)
    clahe = cv2.createCLAHE(
        clipLimit=params.clahe_clip_limit,
        tileGridSize=(tile, tile),
    )
    gray = clahe.apply(gray)

    # --- bilateral filter (edge-preserving denoise) ---
    gray = cv2.bilateralFilter(gray, params.bilateral_d, 75, 75)

    # --- adaptive threshold (INV: dark ring -> white foreground) ---
    binary_inv = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.adaptive_block,
        params.adaptive_c,
    )

    return gray, binary_inv

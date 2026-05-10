"""Data structures for image-to-object point matching."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class MatchedPoint:
    """A matched pair of image point and object (control field) point."""
    control_id: str
    pixel_x: float
    pixel_y: float
    image_x_mm: float
    image_y_mm: float
    obj_x: float
    obj_y: float
    obj_z: float
    is_manual: bool = False

"""Data structures for image-to-object point matching."""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MatchedPoint:
    """A matched pair of image point and object (control field) point."""
    detected_id: str = ""      # detected point ID (e.g. "001"), for re-association on load
    control_id: str = ""
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    image_x_mm: float = 0.0
    image_y_mm: float = 0.0
    obj_x: float = 0.0
    obj_y: float = 0.0
    obj_z: float = 0.0
    is_manual: bool = False

    def to_dict(self) -> dict:
        return {
            "detected_id": self.detected_id,
            "control_id": self.control_id,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "image_x_mm": self.image_x_mm,
            "image_y_mm": self.image_y_mm,
            "obj_x": self.obj_x,
            "obj_y": self.obj_y,
            "obj_z": self.obj_z,
            "is_manual": self.is_manual,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MatchedPoint":
        # Accept old format without detected_id
        d = dict(d)
        d.setdefault("detected_id", "")
        return cls(**d)


def save_matched_points(
    points: list[MatchedPoint], path: str, image_path: str = ""
) -> None:
    """Save matched point pairs to a JSON file."""
    data = {
        "image_path": image_path,
        "matched_points": [p.to_dict() for p in points],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_matched_points(path: str) -> tuple[list[MatchedPoint], str]:
    """Load matched point pairs from a JSON file.

    Returns:
        (matched_points, image_path)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    points = [MatchedPoint.from_dict(d) for d in data["matched_points"]]
    image_path = data.get("image_path", "")
    return points, image_path

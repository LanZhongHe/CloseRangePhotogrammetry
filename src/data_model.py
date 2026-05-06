"""Data structures for control point detection system."""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class EllipseInfo:
    """Ellipse fitting parameters for a ring target."""
    semi_major: float       # long semi-axis (px)
    semi_minor: float       # short semi-axis (px)
    angle_deg: float        # rotation angle (degrees)


@dataclass
class TargetPoint:
    """Single control point data."""
    id: str                                     # target ID string, e.g. "001"
    pixel_x: float                              # image x coordinate (subpixel)
    pixel_y: float                              # image y coordinate (subpixel)
    obj_x: Optional[float] = None               # object-space X (to be bound later)
    obj_y: Optional[float] = None               # object-space Y (to be bound later)
    obj_z: Optional[float] = None               # object-space Z (to be bound later)
    confidence: float = 1.0                     # detection confidence [0, 1]
    source: str = "auto"                        # "auto" or "manual"
    subpixel_method: str = ""                   # "ellipse" / "centroid" / "manual"
    ellipse: Optional[EllipseInfo] = None       # outer ring ellipse params
    eccentricity: Optional[float] = None        # eccentricity (deformation measure)


@dataclass
class DetectionResult:
    """Detection results for one image."""
    image_path: str
    image_width: int
    image_height: int
    detection_time: str = ""
    targets: List[TargetPoint] = field(default_factory=list)

    def __post_init__(self):
        if not self.detection_time:
            self.detection_time = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        d = {
            "image": self.image_path,
            "image_size": [self.image_width, self.image_height],
            "detection_time": self.detection_time,
            "targets": [],
        }
        for t in self.targets:
            td = {
                "id": t.id,
                "pixel_x": round(t.pixel_x, 4),
                "pixel_y": round(t.pixel_y, 4),
                "confidence": round(t.confidence, 4),
                "source": t.source,
                "subpixel_method": t.subpixel_method,
            }
            if t.ellipse is not None:
                td["ellipse"] = {
                    "semi_major": round(t.ellipse.semi_major, 2),
                    "semi_minor": round(t.ellipse.semi_minor, 2),
                    "angle_deg": round(t.ellipse.angle_deg, 2),
                }
            if t.eccentricity is not None:
                td["eccentricity"] = round(t.eccentricity, 4)
            d["targets"].append(td)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DetectionResult":
        """Deserialize from a dict."""
        w, h = d["image_size"]
        result = cls(
            image_path=d["image"],
            image_width=w,
            image_height=h,
            detection_time=d.get("detection_time", ""),
        )
        for td in d.get("targets", []):
            ellipse = None
            if "ellipse" in td:
                e = td["ellipse"]
                ellipse = EllipseInfo(
                    semi_major=e["semi_major"],
                    semi_minor=e["semi_minor"],
                    angle_deg=e["angle_deg"],
                )
            tp = TargetPoint(
                id=td["id"],
                pixel_x=td["pixel_x"],
                pixel_y=td["pixel_y"],
                confidence=td.get("confidence", 1.0),
                source=td.get("source", "auto"),
                subpixel_method=td.get("subpixel_method", ""),
                ellipse=ellipse,
                eccentricity=td.get("eccentricity"),
            )
            result.targets.append(tp)
        return result

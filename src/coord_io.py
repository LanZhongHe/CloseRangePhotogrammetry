"""Load control field coordinates from text file."""

from pathlib import Path


def load_control_field(path: str | Path) -> dict[str, tuple[float, float, float]]:
    """Parse control field coordinate file.

    File format:
        Line 1: number of points
        Lines 2+: point_id X Y Z flag  (space/tab separated)

    Args:
        path: path to the coordinate file (e.g. docs/控制场坐标.txt)

    Returns:
        dict mapping point_id (str) -> (X, Y, Z) as floats
    """
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip first line (point count)
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        point_id = parts[0]
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        result[point_id] = (x, y, z)

    return result

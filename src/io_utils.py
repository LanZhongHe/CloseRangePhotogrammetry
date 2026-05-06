"""JSON I/O utilities for detection results."""

import json
from pathlib import Path
from typing import List

from .data_model import DetectionResult


def save_results(results: List[DetectionResult], output_path: str | Path) -> None:
    """Save a list of DetectionResult to a JSON file.

    Args:
        results: Detection results (one per image).
        output_path: Output JSON file path.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [r.to_dict() for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_results(input_path: str | Path) -> List[DetectionResult]:
    """Load detection results from a JSON file.

    Args:
        input_path: Path to JSON file.

    Returns:
        List of DetectionResult.
    """
    input_path = Path(input_path)
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        data = [data]

    return [DetectionResult.from_dict(d) for d in data]

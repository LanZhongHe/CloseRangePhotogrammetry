"""Control point ID assignment by sequential numbering (001-999).

IDs are assigned in order of detection (sorted top-to-bottom,
left-to-right by position).
"""


def assign_sequential_ids(targets: list, num_digits: int = 3) -> None:
    """Assign sequential IDs to targets sorted by position (top-bottom, left-right).

    Sorts targets by (pixel_y, pixel_x) and assigns "001", "002", ... in order.

    Args:
        targets: List of TargetPoint objects (modified in place).
        num_digits: Number of digits for zero-padding (default 3 -> 001-999).
    """
    targets.sort(key=lambda t: (t.pixel_y, t.pixel_x))
    for i, tp in enumerate(targets):
        tp.id = str(i + 1).zfill(num_digits)

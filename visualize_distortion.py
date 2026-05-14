"""Visualize lens distortion curves and vector fields."""

import json
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def radial_distortion(r, K1, K2, K3=0):
    """Compute radial distortion delta_r = K1*r^3 + K2*r^5 + K3*r^7."""
    return K1 * r**3 + K2 * r**5 + K3 * r**7


def compute_distortion_field(x, y, x0, y0, K1, K2, K3, P1, P2):
    """Compute distortion (dx, dy) at each (x, y) point."""
    dx = x - x0
    dy = y - y0
    r2 = dx**2 + dy**2
    r4 = r2**2
    r6 = r4 * r2
    radial = K1 * r2 + K2 * r4 + K3 * r6
    dd_x = radial * dx + P1 * (r2 + 2 * dx**2) + 2 * P2 * dx * dy
    dd_y = radial * dy + P2 * (r2 + 2 * dy**2) + 2 * P1 * dx * dy
    return dd_x, dd_y


def extract_distortion_params(data: dict, source: str = "auto") -> dict:
    """Extract distortion parameters from DLT or resection JSON."""
    if "results" in data:
        key = "dlt_K1" if "dlt_K1" in data["results"] else list(data["results"].keys())[0]
        r = data["results"][key]
        return {
            "K1": r["distortion"]["K1"],
            "K2": r["distortion"]["K2"],
            "K3": r["distortion"].get("K3", 0),
            "P1": r["distortion"]["P1"],
            "P2": r["distortion"]["P2"],
            "x0": r["intrinsics"]["x0"],
            "y0": r["intrinsics"]["y0"],
            "label": r.get("label", key),
        }
    else:
        return {
            "K1": data["distortion"]["K1"],
            "K2": data["distortion"]["K2"],
            "K3": data["distortion"].get("K3", 0),
            "P1": data["distortion"]["P1"],
            "P2": data["distortion"]["P2"],
            "x0": data["intrinsics"]["x0"],
            "y0": data["intrinsics"]["y0"],
            "label": "Resection",
        }


def main():
    parser = argparse.ArgumentParser(description="Visualize lens distortion")
    parser.add_argument("files", nargs="+", help="DLT or resection result JSON files")
    parser.add_argument("--output", "-o", default="", help="Output PDF path")
    parser.add_argument("--no-show", action="store_true", help="Save without showing")
    args = parser.parse_args()

    params_list = []
    for fpath in args.files:
        if not Path(fpath).exists():
            print(f"File not found: {fpath}", file=sys.stderr)
            continue
        data = load_json(fpath)
        params_list.append(extract_distortion_params(data))

    if not params_list:
        print("No valid files.", file=sys.stderr)
        sys.exit(1)

    n = len(params_list)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Radial distortion curve ---
    ax1 = axes[0]
    r_max = 21.0  # mm (half of sensor diagonal ~21mm for full-frame)
    r = np.linspace(0, r_max, 200)

    colors = ["steelblue", "coral", "seagreen", "orange"]
    for i, p in enumerate(params_list):
        c = colors[i % len(colors)]
        dr = radial_distortion(r, p["K1"], p["K2"], p["K3"])
        ax1.plot(r, dr * 1000, color=c, lw=2, label=p["label"])
        # Mark max distortion
        max_dr = np.max(np.abs(dr)) * 1000
        r_at_max = r[np.argmax(np.abs(dr))]
        ax1.annotate(f"{max_dr:.1f} μm", xy=(r_at_max, dr[np.argmax(np.abs(dr))] * 1000),
                     fontsize=8, color=c, ha="left", va="bottom")

    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.set_xlabel("像场半径 r (mm)")
    ax1.set_ylabel("径向畸变 δr (μm)")
    ax1.set_title("径向畸变曲线")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Distortion vector field ---
    ax2 = axes[1]
    # Create grid of points across the image sensor
    sensor_w, sensor_h = 18.0, 12.0  # half-sensor in mm
    nx, ny = 15, 10
    x_grid = np.linspace(-sensor_w, sensor_w, nx)
    y_grid = np.linspace(-sensor_h, sensor_h, ny)
    X, Y = np.meshgrid(x_grid, y_grid)

    for i, p in enumerate(params_list):
        c = colors[i % len(colors)]
        DX, DY = compute_distortion_field(X, Y, p["x0"], p["y0"],
                                          p["K1"], p["K2"], p["K3"], p["P1"], p["P2"])
        # Scale for visibility
        scale = 5000
        ax2.quiver(X, Y, DX * scale, DY * scale, color=c, alpha=0.6,
                   scale=1, scale_units="xy", label=p["label"], width=0.004)

    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("畸变矢量场 (放大5000倍)")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    # Draw sensor boundary
    from matplotlib.patches import Rectangle
    ax2.add_patch(Rectangle((-sensor_w, -sensor_h), 2*sensor_w, 2*sensor_h,
                             fill=False, edgecolor="gray", ls="--", lw=0.8))

    fig.tight_layout()

    out = args.output or "output/distortion.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()

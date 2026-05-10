"""Visualize space resection results with 3D plot — supports multiple cameras."""

import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

from src.camera_model import rotation_matrix


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_resection_result(filepath: str) -> dict:
    """Load resection result from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_matched_points(filepath: str) -> dict:
    """Load matched points JSON (for image_path and control_id list)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_control_field(filepath: str) -> dict[str, tuple[float, float, float]]:
    """Load control field coordinates."""
    points = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip first line (count)
            parts = line.strip().split()
            if len(parts) >= 4:
                pid = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points[pid] = (x, y, z)
    return points


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

CAMERA_COLORS = [
    '#e74c3c',  # red
    '#2980b9',  # blue
    '#27ae60',  # green
    '#8e44ad',  # purple
    '#e67e22',  # orange
    '#1abc9c',  # teal
    '#c0392b',  # dark red
    '#2c3e50',  # dark blue-gray
]


def plot_camera(ax, ext: dict, f: float, color: str, label: str,
                scale: float = 200.0):
    """Plot camera position, axes, and scaled view frustum."""
    Xs, Ys, Zs = ext['Xs'], ext['Ys'], ext['Zs']
    omega, phi, kappa = ext['omega_rad'], ext['phi_rad'], ext['kappa_rad']

    # Camera position
    ax.scatter([Xs], [Ys], [Zs], c=color, s=150, marker='^', zorder=5, edgecolors='black', linewidth=0.5)

    # Rotation matrix
    R = rotation_matrix(omega, phi, kappa)

    # Camera axes in object space (R^T: camera -> object)
    axes_obj = R.T  # columns are camera X,Y,Z in object space

    # Plot camera axes
    for i, (c, lbl) in enumerate(zip(['red','green','blue'], ['X','Y','Z'])):
        ax.quiver(Xs, Ys, Zs,
                  axes_obj[0, i], axes_obj[1, i], axes_obj[2, i],
                  color=c, length=scale * 0.4, alpha=0.6,
                  arrow_length_ratio=0.15, linewidth=1.2)

    # Scaled frustum — use scene-relative size so it's visible
    fw = scale * 0.25   # frustum half-width
    fh = scale * 0.18   # frustum half-height
    depth = scale * 0.35  # frustum depth (along camera Z)

    corners_cam = np.array([
        [-fw, -fh, depth],
        [ fw, -fh, depth],
        [ fw,  fh, depth],
        [-fw,  fh, depth],
    ])

    corners_obj = (R.T @ corners_cam.T)
    corners_obj[0, :] += Xs
    corners_obj[1, :] += Ys
    corners_obj[2, :] += Zs

    # Camera center as point 4
    pts = np.column_stack([corners_obj, [[Xs], [Ys], [Zs]]])

    # Image plane edges
    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot([pts[0,i], pts[0,j]], [pts[1,i], pts[1,j]], [pts[2,i], pts[2,j]],
                color=color, alpha=0.5, linewidth=1.2)
    # Lines from corners to camera center
    for i in range(4):
        ax.plot([pts[0,i], pts[0,4]], [pts[1,i], pts[1,4]], [pts[2,i], pts[2,4]],
                color=color, alpha=0.35, linewidth=1.0, linestyle='--')

    # Fill image plane
    verts = [list(zip(corners_obj[0, :], corners_obj[1, :], corners_obj[2, :]))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor=color, edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)

    # Label
    ax.text(Xs, Ys, Zs - scale * 0.25, label, fontsize=9, color=color, fontweight='bold',
            ha='center', va='top')


def plot_control_points(ax, control_field: dict, all_matched_ids: set[str]):
    """Plot control points, highlighting those matched by any camera."""
    all_x = [p[0] for p in control_field.values()]
    all_y = [p[1] for p in control_field.values()]
    all_z = [p[2] for p in control_field.values()]

    # Unmatched (light gray)
    unmatched_ids = set(control_field.keys()) - all_matched_ids
    if unmatched_ids:
        ux = [control_field[pid][0] for pid in unmatched_ids]
        uy = [control_field[pid][1] for pid in unmatched_ids]
        uz = [control_field[pid][2] for pid in unmatched_ids]
        ax.scatter(ux, uy, uz, c='lightgray', s=15, alpha=0.4)

    # Matched (blue)
    if all_matched_ids:
        mx = [control_field[pid][0] for pid in all_matched_ids if pid in control_field]
        my = [control_field[pid][1] for pid in all_matched_ids if pid in control_field]
        mz = [control_field[pid][2] for pid in all_matched_ids if pid in control_field]
        ax.scatter(mx, my, mz, c='blue', s=40, alpha=0.8)

        for pid in all_matched_ids:
            if pid in control_field:
                x, y, z = control_field[pid]
                ax.text(x, y, z, f'  {pid}', fontsize=7, color='navy')


def plot_origin(ax, scale: float = 200.0):
    """Plot coordinate origin with axes."""
    ax.scatter([0], [0], [0], c='black', s=100, marker='o', zorder=5)
    ax.text(0, 0, 0, '  O', fontsize=9, fontweight='bold')

    ax.quiver(0, 0, 0, scale, 0, 0, color='red', arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, scale, 0, color='green', arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, scale, color='blue', arrow_length_ratio=0.08, linewidth=1.5)

    ax.text(scale, 0, 0, ' X', fontsize=9, color='red')
    ax.text(0, scale, 0, ' Y', fontsize=9, color='green')
    ax.text(0, 0, scale, ' Z', fontsize=9, color='blue')


def plot_rays(ax, ext: dict, control_field: dict, matched_ids: list[str], color: str):
    """Plot photography rays from camera to matched control points."""
    Xs, Ys, Zs = ext['Xs'], ext['Ys'], ext['Zs']

    for pid in matched_ids:
        if pid in control_field:
            x, y, z = control_field[pid]
            ax.plot([Xs, x], [Ys, y], [Zs, z],
                    color=color, linewidth=0.6, alpha=0.35)


# ---------------------------------------------------------------------------
# Multi-camera entry point
# ---------------------------------------------------------------------------

def visualize_multi(
    result_paths: list[str],
    control_field_path: str,
    output_path: str = 'output/resection_visualization.png',
    show: bool = True,
):
    """Visualize multiple resection results in a single 3D scene.

    Args:
        result_paths: list of resection result JSON file paths
        control_field_path: path to control field coordinate file
        output_path: where to save the PNG
        show: whether to call plt.show()
    """
    control_field = load_control_field(control_field_path)

    # Load all results
    results = []
    for rp in result_paths:
        r = load_resection_result(rp)
        r['_source'] = os.path.basename(rp)
        results.append(r)

    # Collect all matched control IDs
    all_matched_ids: set[str] = set()
    for r in results:
        for res in r.get('residuals', []):
            all_matched_ids.add(res['control_id'])

    # --- Figure ---
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111, projection='3d')

    # Aspect ratio from control field extent
    all_pts = list(control_field.values())
    ranges = [
        max(p[0] for p in all_pts) - min(p[0] for p in all_pts),
        max(p[1] for p in all_pts) - min(p[1] for p in all_pts),
        max(p[2] for p in all_pts) - min(p[2] for p in all_pts),
    ]
    max_range = max(ranges) / 2.0
    mid = [np.mean([p[i] for p in all_pts]) for i in range(3)]

    # Plot elements
    plot_origin(ax, scale=max_range * 0.2)
    plot_control_points(ax, control_field, all_matched_ids)

    # Plot each camera
    info_lines = []
    for i, r in enumerate(results):
        color = CAMERA_COLORS[i % len(CAMERA_COLORS)]
        label = os.path.splitext(r['_source'])[0]

        ext = r['exterior_orientation']
        f = r['intrinsics']['f']
        matched_ids = [res['control_id'] for res in r.get('residuals', [])]

        plot_camera(ax, ext, f, color, label, scale=max_range * 0.15)
        plot_rays(ax, ext, control_field, matched_ids, color)

        sigma_px = r.get('sigma0_px', 0)
        info_lines.append(
            f"[{label}]  "
            f"Xs={ext['Xs']:.1f} Ys={ext['Ys']:.1f} Zs={ext['Zs']:.1f}  "
            f"sigma0={sigma_px:.2f}px  pts={len(matched_ids)}"
        )

    # Axes
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'Space Resection — {len(results)} Camera(s)', fontsize=13)

    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Legend
    legend_handles = []
    for i, r in enumerate(results):
        color = CAMERA_COLORS[i % len(CAMERA_COLORS)]
        label = os.path.splitext(r['_source'])[0]
        legend_handles.append(Line2D([0], [0], marker='^', color='w',
                              markerfacecolor=color, markersize=10, label=label))
    legend_handles.append(Line2D([0], [0], color='blue', marker='o', linestyle='None',
                          markersize=5, label='Matched Points'))
    legend_handles.append(Line2D([0], [0], color='lightgray', marker='o', linestyle='None',
                          markersize=5, label='Control Points'))
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    # Info text box
    fig.text(0.02, 0.02, '\n'.join(info_lines), fontsize=8, fontfamily='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")

    if show:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Visualize resection results (single or multiple cameras).'
    )
    parser.add_argument(
        'results', nargs='*',
        help='Resection result JSON files. If omitted, auto-detect from output/*.json'
    )
    parser.add_argument(
        '--control', default='docs/控制场坐标.txt',
        help='Control field coordinate file (default: docs/控制场坐标.txt)'
    )
    parser.add_argument(
        '--output', default='output/resection_visualization.png',
        help='Output PNG path'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Skip plt.show() (save only)'
    )
    args = parser.parse_args()

    # Auto-detect result files if none specified
    result_paths = args.results
    if not result_paths:
        result_paths = sorted(glob.glob('output/resection_*.json'))
        if not result_paths:
            print("No resection result files found in output/. "
                  "Specify paths as arguments or run resection first.")
            return
        print(f"Auto-detected {len(result_paths)} result file(s):")
        for p in result_paths:
            print(f"  {p}")

    visualize_multi(
        result_paths=result_paths,
        control_field_path=args.control,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == '__main__':
    main()

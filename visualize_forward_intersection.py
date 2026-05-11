"""Visualize forward intersection results in 3D with photography rays.

Usage:
    python visualize_forward_intersection.py output/forward_intersection.json
    python visualize_forward_intersection.py output/fi.json --cam1 output/dlt_DSC35.json --cam2 output/dlt_DSC37.json
    python visualize_forward_intersection.py output/fi.json --cam1 cam1.json --cam2 cam2.json --no-show
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

from src.camera_model import rotation_matrix
from src.dlt import derive_orientation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_fi_result(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_camera_json(filepath: str) -> dict:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_camera_info(data: dict) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Extract camera center, focal length, and rotation matrix from DLT or resection JSON.

    Returns:
        (center, f, R) where R is the 3x3 rotation matrix (world-to-camera),
        or None if data format is unrecognized.
    """
    if 'L_params' in data:
        L = np.array(data['L_params'])
        _, ext = derive_orientation(L)
        f = data.get('intrinsics', {}).get('f', 50.0)
        R = rotation_matrix(ext.omega, ext.phi, ext.kappa)
        return np.array([ext.Xs, ext.Ys, ext.Zs]), f, R
    if 'exterior_orientation' in data:
        ext = data['exterior_orientation']
        f = data.get('intrinsics', {}).get('f', 50.0)
        R = rotation_matrix(ext['omega'], ext['phi'], ext['kappa'])
        return np.array([ext['Xs'], ext['Ys'], ext['Zs']]), f, R
    return None


# ---------------------------------------------------------------------------
# Ray geometry helpers
# ---------------------------------------------------------------------------

def compute_ray_params(
    cam_center: np.ndarray, point_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute ray direction and closest-approach geometry.

    Returns:
        direction: unit vector from camera to point
        midpoint: halfway point between camera and target
        length: distance from camera to point
    """
    d = point_xyz - cam_center
    length = np.linalg.norm(d)
    direction = d / length if length > 0 else d
    midpoint = cam_center + direction * length * 0.4
    return direction, midpoint, length


def project_to_frustum(
    cam_center: np.ndarray, direction: np.ndarray,
    R: np.ndarray, f: float,
    image_w: int, image_h: int, sensor_w: float,
) -> np.ndarray:
    """Project ray direction to image plane corners (for frustum visualization)."""
    pixel_size = sensor_w / image_w
    fw = (image_w / 2) * pixel_size
    fh = (image_h / 2) * pixel_size

    corners_cam = np.array([
        [-fw, -fh, f],
        [ fw, -fh, f],
        [ fw,  fh, f],
        [-fw,  fh, f],
    ])
    return (R.T @ corners_cam.T).T + cam_center


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CAMERA_COLORS = ['#e74c3c', '#2980b9']  # red, blue for cam1, cam2


def plot_fi_points(ax, pts: list[dict], color_map: dict | None = None):
    """Plot 3D points with labels."""
    xs = [p['X'] for p in pts]
    ys = [p['Y'] for p in pts]
    zs = [p['Z'] for p in pts]

    # Color by residual magnitude if available
    residuals = [np.sqrt(
        p.get('residuals', {}).get('vx1', 0)**2 +
        p.get('residuals', {}).get('vy1', 0)**2
    ) for p in pts]

    scatter = ax.scatter(xs, ys, zs, c=residuals if any(r > 0 for r in residuals) else '#2196F3',
                         cmap='RdYlGn_r', s=80, marker='o',
                         edgecolors='black', linewidth=0.8, zorder=5,
                         vmin=0, vmax=max(residuals) * 1.5 if residuals else 1)

    for i, p in enumerate(pts):
        ax.text(p['X'], p['Y'], p['Z'] + 20,
                f"  {p['point_id']}", fontsize=8, color='#1565C0', fontweight='bold')


def plot_camera(ax, center: np.ndarray, f: float, color: str, label: str,
                R: np.ndarray | None = None, scale: float = 200.0):
    """Plot camera position with axes and image plane."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    Xs, Ys, Zs = center
    ax.scatter([Xs], [Ys], [Zs], c=color, s=150, marker='^',
               zorder=5, edgecolors='black', linewidth=0.5)

    if R is None:
        R = np.eye(3)

    # Image plane: perpendicular to camera Z-axis, in front of camera
    fw = scale * 0.25
    fh = scale * 0.18
    depth = scale * 0.35
    corners_cam = np.array([
        [ fw, -fh, -depth], [-fw, -fh, -depth],
        [-fw,  fh, -depth], [ fw,  fh, -depth],
    ])
    corners_obj = (R.T @ corners_cam.T).T + center

    verts = [list(zip(corners_obj[:, 0], corners_obj[:, 1], corners_obj[:, 2]))]
    poly = Poly3DCollection(verts, alpha=0.15, facecolor=color, edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)

    for i, j in [(0,1),(1,2),(2,3),(3,0)]:
        ax.plot([corners_obj[i, 0], corners_obj[j, 0]],
                [corners_obj[i, 1], corners_obj[j, 1]],
                [corners_obj[i, 2], corners_obj[j, 2]],
                color=color, alpha=0.5, linewidth=1.2)

    # Camera axes (X=red, Y=green, Z=blue)
    for i, (c, lbl) in enumerate(zip(['red','green','blue'], ['X','Y','Z'])):
        ax.quiver(Xs, Ys, Zs,
                  R[0, i], R[1, i], R[2, i],
                  color=c, length=scale * 0.15, alpha=0.5,
                  arrow_length_ratio=0.15, linewidth=1.0)

    ax.text(Xs, Ys, Zs - scale * 0.25, label, fontsize=9, color=color,
            fontweight='bold', ha='center', va='top')


def plot_rays(ax, cam_center: np.ndarray, points: list[dict], color: str,
              show_markers: bool = True):
    """Plot rays from camera center through each point.

    Each ray is drawn as:
      - solid line from camera center to the 3D point
      - dashed extension beyond the point (indicating ray direction)
    If show_markers, small circles mark the camera and the point intersection.
    """
    Xs, Ys, Zs = cam_center
    for p in points:
        d = np.array([p['X'] - Xs, p['Y'] - Ys, p['Z'] - Zs])
        length = np.linalg.norm(d)
        if length < 1e-6:
            continue
        d_norm = d / length

        # Main ray: camera → point (solid)
        ax.plot([Xs, p['X']], [Ys, p['Y']], [Zs, p['Z']],
                color=color, linewidth=0.8, alpha=0.4)

        # Extension beyond point (dashed, 30% of ray length)
        ext = length * 0.3
        ext_end = np.array([p['X'], p['Y'], p['Z']]) + d_norm * ext
        ax.plot([p['X'], ext_end[0]], [p['Y'], ext_end[1]], [p['Z'], ext_end[2]],
                color=color, linewidth=0.6, alpha=0.25, linestyle=':')

    # Camera center marker
    if show_markers and points:
        ax.scatter([Xs], [Ys], [Zs], c=color, s=80, marker='^', alpha=0.6, zorder=4)


def plot_origin(ax, scale: float = 200.0):
    """Plot coordinate origin with labeled axes."""
    ax.scatter([0], [0], [0], c='black', s=80, marker='o', zorder=5)
    ax.text(0, 0, 0, '  O', fontsize=9, fontweight='bold')
    ax.quiver(0, 0, 0, scale, 0, 0, color='red', arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, scale, 0, color='green', arrow_length_ratio=0.08, linewidth=1.5)
    ax.quiver(0, 0, 0, 0, 0, scale, color='blue', arrow_length_ratio=0.08, linewidth=1.5)
    ax.text(scale, 0, 0, ' X', fontsize=9, color='red')
    ax.text(0, scale, 0, ' Y', fontsize=9, color='green')
    ax.text(0, 0, scale, ' Z', fontsize=9, color='blue')


# ---------------------------------------------------------------------------
# Main visualization
# ---------------------------------------------------------------------------

def visualize(
    fi_path: str,
    cam1_path: str | None = None,
    cam2_path: str | None = None,
    output_path: str = 'output/forward_intersection_visualization.png',
    show: bool = True,
):
    result = load_fi_result(fi_path)
    pts = result['points']

    # Load camera info
    cam1_center, cam1_f, cam1_R = None, 50.0, None
    cam2_center, cam2_f, cam2_R = None, 50.0, None
    if cam1_path:
        info = extract_camera_info(load_camera_json(cam1_path))
        if info:
            cam1_center, cam1_f, cam1_R = info
    if cam2_path:
        info = extract_camera_info(load_camera_json(cam2_path))
        if info:
            cam2_center, cam2_f, cam2_R = info

    # --- Layout: 3D plot ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scene scale
    all_xyz = np.array([[p['X'], p['Y'], p['Z']] for p in pts])
    centers = [c for c in [cam1_center, cam2_center] if c is not None]
    if centers:
        all_ref = np.vstack([all_xyz] + [c.reshape(1, 3) for c in centers])
    else:
        all_ref = all_xyz

    mid = all_ref.mean(axis=0)
    max_range = max(all_ref.max(axis=0) - all_ref.min(axis=0)) / 2.0
    if max_range < 100:
        max_range = 500

    # Plot
    plot_origin(ax, scale=max_range * 0.3)
    plot_fi_points(ax, pts)

    if cam1_center is not None:
        plot_camera(ax, cam1_center, cam1_f, CAMERA_COLORS[0], 'Camera 1', R=cam1_R, scale=max_range * 0.3)
        plot_rays(ax, cam1_center, pts, CAMERA_COLORS[0])
    if cam2_center is not None:
        plot_camera(ax, cam2_center, cam2_f, CAMERA_COLORS[1], 'Camera 2', R=cam2_R, scale=max_range * 0.3)
        plot_rays(ax, cam2_center, pts, CAMERA_COLORS[1])

    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
    ax.set_title(f'Forward Intersection — {len(pts)} pts, σ₀ = {result["sigma0_mm"]:.4f} mm', fontsize=13)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    # Legend
    legend_handles = [
        Line2D([0], [0], color='#2196F3', marker='o', linestyle='None', markersize=6, label='FI Points'),
        Line2D([0], [0], color=CAMERA_COLORS[0], linewidth=1.5, label='Camera 1 rays'),
        Line2D([0], [0], color=CAMERA_COLORS[1], linewidth=1.5, label='Camera 2 rays'),
    ]
    if cam1_center is not None:
        legend_handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor=CAMERA_COLORS[0], markersize=10, label='Camera 1'))
    if cam2_center is not None:
        legend_handles.append(Line2D([0], [0], marker='^', color='w', markerfacecolor=CAMERA_COLORS[1], markersize=10, label='Camera 2'))
    ax.legend(handles=legend_handles, loc='upper left', fontsize=8)

    # Summary
    sigma = result['sigma0_mm']
    angles = [p.get('intersection_angle_deg', 0) for p in pts]
    summary = (
        f"σ₀ = {sigma:.4f} mm   "
        f"Angle: min={min(angles):.1f}°  max={max(angles):.1f}°  mean={np.mean(angles):.1f}°   "
        f"Pts: {len(pts)}"
    )
    fig.text(0.5, 0.02, summary, fontsize=10, fontfamily='monospace',
             ha='center', va='bottom',
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
    parser = argparse.ArgumentParser(description='Visualize forward intersection results in 3D.')
    parser.add_argument('fi_result', help='Forward intersection result JSON file')
    parser.add_argument('--cam1', default=None, help='DLT/resection JSON for camera 1')
    parser.add_argument('--cam2', default=None, help='DLT/resection JSON for camera 2')
    parser.add_argument('--output', default='output/forward_intersection_visualization.png', help='Output PNG path')
    parser.add_argument('--no-show', action='store_true', help='Skip plt.show() (save only)')
    args = parser.parse_args()
    visualize(fi_path=args.fi_result, cam1_path=args.cam1, cam2_path=args.cam2,
              output_path=args.output, show=not args.no_show)


if __name__ == '__main__':
    main()

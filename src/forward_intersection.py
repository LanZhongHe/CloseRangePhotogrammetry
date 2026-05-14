"""Forward intersection to compute 3D object coordinates from image pairs."""

from dataclasses import dataclass
import numpy as np

from .camera_model import (
    CameraIntrinsics, ExteriorOrientation, DistortionCoefficients,
    rotation_matrix, compute_distortion,
)


@dataclass
class TiePoint:
    """A pair of corresponding image coordinates in two images."""
    point_id: str
    image1_x: float  # mm
    image1_y: float  # mm
    image2_x: float  # mm
    image2_y: float  # mm


@dataclass
class ForwardIntersectionResult:
    """Results from forward intersection."""
    point_ids: list[str]
    coordinates: np.ndarray          # (n, 3) — X, Y, Z
    residuals: list[tuple[float, float, float, float]]  # (vx1, vy1, vx2, vy2) per point
    sigma0: float                    # unit weight std dev (mm)
    intersection_angles: list[float] # degrees, per point


# ---- DLT-based forward intersection ----

def forward_intersection_dlt(
    tie_points: list[TiePoint],
    L1: np.ndarray,
    L2: np.ndarray,
) -> ForwardIntersectionResult:
    """Compute 3D coordinates from two-image DLT parameters.

    For each tie point, the DLT equations from two images give 4 equations
    for 3 unknowns (X, Y, Z), solved by least squares.

    Args:
        tie_points: corresponding image points in two images
        L1: 11 DLT parameters for image 1
        L2: 11 DLT parameters for image 2

    Returns:
        ForwardIntersectionResult with coordinates and accuracy
    """
    n = len(tie_points)
    if n < 1:
        raise ValueError("Need at least 1 tie point")

    point_ids = []
    coordinates = np.zeros((n, 3))
    residuals = []
    angles = []

    for i, tp in enumerate(tie_points):
        point_ids.append(tp.point_id)

        # Build 4x3 system: A @ [X,Y,Z]^T = b
        # Image 1: (L1-x*L9)*X + (L2-x*L10)*Y + (L3-x*L11)*Z = x - L4
        x1, y1 = tp.image1_x, tp.image1_y
        x2, y2 = tp.image2_x, tp.image2_y

        A = np.array([
            [L1[0] - x1*L1[8],  L1[1] - x1*L1[9],  L1[2] - x1*L1[10]],
            [L1[4] - y1*L1[8],  L1[5] - y1*L1[9],  L1[6] - y1*L1[10]],
            [L2[0] - x2*L2[8],  L2[1] - x2*L2[9],  L2[2] - x2*L2[10]],
            [L2[4] - y2*L2[8],  L2[5] - y2*L2[9],  L2[6] - y2*L2[10]],
        ])
        b = np.array([x1 - L1[3], y1 - L1[7], x2 - L2[3], y2 - L2[7]])

        # Least squares: (A^T A) X = A^T b
        AtA = A.T @ A
        Atb = A.T @ b
        try:
            xyz = np.linalg.solve(AtA, Atb)
        except np.linalg.LinAlgError:
            xyz = np.array([0.0, 0.0, 0.0])

        coordinates[i] = xyz

        # Residuals
        v = A @ xyz - b
        residuals.append((v[0], v[1], v[2], v[3]))

        # Intersection angle: angle between the two rays
        # Ray 1 direction: from camera 1 center through image point 1
        # Ray 2 direction: from camera 2 center through image point 2
        # Approximate using the DLT geometry
        angle = _compute_intersection_angle_dlt(L1, L2, x1, y1, x2, y2)
        angles.append(angle)

    # Overall accuracy
    all_v = np.array(residuals).flatten()
    dof = 4 * n - 3 * n  # 4 equations per point, 3 unknowns per point
    if dof > 0:
        sigma0 = np.sqrt(np.sum(all_v**2) / dof)
    else:
        sigma0 = 0.0

    return ForwardIntersectionResult(
        point_ids=point_ids,
        coordinates=coordinates,
        residuals=residuals,
        sigma0=sigma0,
        intersection_angles=angles,
    )


def _compute_intersection_angle_dlt(
    L1: np.ndarray, L2: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
) -> float:
    """Compute intersection angle between two rays using DLT parameters.

    Derives camera centers and ray directions from L parameters.
    """
    # Derive orientation from each set of L params
    from .dlt import derive_orientation
    intr1, ext1 = derive_orientation(L1)
    intr2, ext2 = derive_orientation(L2)

    # Camera centers
    C1 = np.array([ext1.Xs, ext1.Ys, ext1.Zs])
    C2 = np.array([ext2.Xs, ext2.Ys, ext2.Zs])

    # Rotation matrices
    R1 = rotation_matrix(ext1.omega, ext1.phi, ext1.kappa)
    R2 = rotation_matrix(ext2.omega, ext2.phi, ext2.kappa)

    # Ray directions in object space: d = R^T @ [x, y, -f]
    d1 = R1.T @ np.array([x1, y1, -intr1.f])
    d2 = R2.T @ np.array([x2, y2, -intr2.f])
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)

    cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
    return np.degrees(np.arccos(abs(cos_angle)))


# ---- Resection-based forward intersection ----

def forward_intersection_resection(
    tie_points: list[TiePoint],
    ext1: ExteriorOrientation,
    ext2: ExteriorOrientation,
    intrinsics: CameraIntrinsics,
) -> ForwardIntersectionResult:
    """Compute 3D coordinates by ray intersection from two resection results.

    For each tie point, projects rays from each camera center and finds
    the closest point between them.

    Args:
        tie_points: corresponding image points
        ext1, ext2: exterior orientation of each image
        intrinsics: camera intrinsics (shared)

    Returns:
        ForwardIntersectionResult
    """
    n = len(tie_points)
    f = intrinsics.f

    R1 = rotation_matrix(ext1.omega, ext1.phi, ext1.kappa)
    R2 = rotation_matrix(ext2.omega, ext2.phi, ext2.kappa)

    C1 = np.array([ext1.Xs, ext1.Ys, ext1.Zs])
    C2 = np.array([ext2.Xs, ext2.Ys, ext2.Zs])

    point_ids = []
    coordinates = np.zeros((n, 3))
    residuals = []
    angles = []

    for i, tp in enumerate(tie_points):
        point_ids.append(tp.point_id)

        # Ray directions in object space
        d1 = R1.T @ np.array([tp.image1_x, tp.image1_y, -f])
        d2 = R2.T @ np.array([tp.image2_x, tp.image2_y, -f])
        d1 = d1 / np.linalg.norm(d1)
        d2 = d2 / np.linalg.norm(d2)

        # Closest point between two rays
        # P = C1 + t1*d1 = C2 + t2*d2 (overdetermined)
        # Solve: [d1, -d2] @ [t1, t2]^T = C2 - C1
        w = C2 - C1
        A = np.column_stack([d1, -d2])
        t, _, _, _ = np.linalg.lstsq(A, w, rcond=None)

        # Midpoint of the two closest points
        P1 = C1 + t[0] * d1
        P2 = C2 + t[1] * d2
        xyz = (P1 + P2) / 2.0
        coordinates[i] = xyz

        # Residual: distance between the two ray points at closest approach
        gap = P2 - P1
        residuals.append((gap[0], gap[1], gap[2], np.linalg.norm(gap)))

        # Intersection angle
        cos_angle = np.clip(np.dot(d1, d2), -1.0, 1.0)
        angles.append(np.degrees(np.arccos(abs(cos_angle))))

    # Overall accuracy: RMS of gaps
    gaps = [r[3] for r in residuals]
    sigma0 = np.sqrt(np.mean(np.array(gaps)**2)) if gaps else 0.0

    return ForwardIntersectionResult(
        point_ids=point_ids,
        coordinates=coordinates,
        residuals=residuals,
        sigma0=sigma0,
        intersection_angles=angles,
    )


# ---- Distance computation ----

def compute_point_distance(
    coords: np.ndarray,
    idx1: int,
    idx2: int,
) -> float:
    """Compute Euclidean distance between two points by index."""
    return float(np.linalg.norm(coords[idx1] - coords[idx2]))


def compute_all_distances(
    coords: np.ndarray,
    point_ids: list[str],
) -> list[dict]:
    """Compute pairwise distances between all points.

    Returns:
        list of dicts with 'point1', 'point2', 'distance'
    """
    n = len(point_ids)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            distances.append({
                "point1": point_ids[i],
                "point2": point_ids[j],
                "distance": d,
            })
    return distances


# ---- DLT-based forward intersection with distortion correction ----

def _undistort_point(
    x: float, y: float,
    distortion: DistortionCoefficients,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Remove lens distortion from image coordinates.

    x_undistorted = x - delta_x(x, y)

    Args:
        x, y: distorted image coordinates in mm
        distortion: distortion coefficients from DLT
        intrinsics: camera intrinsics (provides x0, y0)

    Returns:
        (x_undist, y_undist) in mm
    """
    dx, dy = compute_distortion(
        x, y, intrinsics.x0, intrinsics.y0,
        distortion.K1, distortion.K2, distortion.K3,
        distortion.P1, distortion.P2,
    )
    return x - dx, y - dy


def forward_intersection_dlt_distorted(
    tie_points: list[TiePoint],
    L1: np.ndarray,
    L2: np.ndarray,
    distortion1: DistortionCoefficients,
    distortion2: DistortionCoefficients,
    intrinsics1: CameraIntrinsics,
    intrinsics2: CameraIntrinsics,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> ForwardIntersectionResult:
    """Forward intersection with DLT distortion correction and iterative refinement.

    Algorithm:
        1. Undistort image coordinates using DLT distortion coefficients
        2. Compute initial object coordinates via DLT (least squares)
        3. Iteratively refine: DLT reproject → compute residual → correct object coords

    Args:
        tie_points: corresponding image points in two images
        L1, L2: 11 DLT parameters for each image
        distortion1, distortion2: distortion coefficients from DLT for each image
        intrinsics1, intrinsics2: camera intrinsics (x0, y0) for each image
        max_iter: iteration limit
        tol: convergence threshold on coordinate corrections (mm)

    Returns:
        ForwardIntersectionResult with coordinates and accuracy
    """
    n = len(tie_points)
    if n < 1:
        raise ValueError("Need at least 1 tie point")

    point_ids = []
    coordinates = np.zeros((n, 3))
    residuals = []
    angles = []

    for i, tp in enumerate(tie_points):
        point_ids.append(tp.point_id)

        # Step 1: Undistort image coordinates
        x1, y1 = _undistort_point(tp.image1_x, tp.image1_y, distortion1, intrinsics1)
        x2, y2 = _undistort_point(tp.image2_x, tp.image2_y, distortion2, intrinsics2)

        # Step 2: Initial DLT forward intersection with undistorted coordinates
        A = np.array([
            [L1[0] - x1*L1[8],  L1[1] - x1*L1[9],  L1[2] - x1*L1[10]],
            [L1[4] - y1*L1[8],  L1[5] - y1*L1[9],  L1[6] - y1*L1[10]],
            [L2[0] - x2*L2[8],  L2[1] - x2*L2[9],  L2[2] - x2*L2[10]],
            [L2[4] - y2*L2[8],  L2[5] - y2*L2[9],  L2[6] - y2*L2[10]],
        ])
        b = np.array([x1 - L1[3], y1 - L1[7], x2 - L2[3], y2 - L2[7]])

        try:
            xyz = np.linalg.solve(A.T @ A, A.T @ b)
        except np.linalg.LinAlgError:
            xyz = np.array([0.0, 0.0, 0.0])

        # Step 3: Iterative refinement within DLT framework
        for iteration in range(max_iter):
            X, Y, Z = xyz

            # DLT reproject: compute expected image coordinates
            denom1 = L1[8]*X + L1[9]*Y + L1[10]*Z + 1.0
            denom2 = L2[8]*X + L2[9]*Y + L2[10]*Z + 1.0

            if abs(denom1) < 1e-20 or abs(denom2) < 1e-20:
                break

            x1_calc = (L1[0]*X + L1[1]*Y + L1[2]*Z + L1[3]) / denom1
            y1_calc = (L1[4]*X + L1[5]*Y + L1[6]*Z + L1[7]) / denom1
            x2_calc = (L2[0]*X + L2[1]*Y + L2[2]*Z + L2[3]) / denom2
            y2_calc = (L2[4]*X + L2[5]*Y + L2[6]*Z + L2[7]) / denom2

            # Residuals: undistorted - calculated
            dx1 = x1 - x1_calc
            dy1 = y1 - y1_calc
            dx2 = x2 - x2_calc
            dy2 = y2 - y2_calc

            # Build Jacobian for correction equations
            # d(x_calc)/dX = (L1 - x_calc*L9) / denom, etc.
            A_corr = np.array([
                [(L1[0] - x1_calc*L1[8]) / denom1,
                 (L1[1] - x1_calc*L1[9]) / denom1,
                 (L1[2] - x1_calc*L1[10]) / denom1],
                [(L1[4] - y1_calc*L1[8]) / denom1,
                 (L1[5] - y1_calc*L1[9]) / denom1,
                 (L1[6] - y1_calc*L1[10]) / denom1],
                [(L2[0] - x2_calc*L2[8]) / denom2,
                 (L2[1] - x2_calc*L2[9]) / denom2,
                 (L2[2] - x2_calc*L2[10]) / denom2],
                [(L2[4] - y2_calc*L2[8]) / denom2,
                 (L2[5] - y2_calc*L2[9]) / denom2,
                 (L2[6] - y2_calc*L2[10]) / denom2],
            ])
            l_corr = np.array([dx1, dy1, dx2, dy2])

            # Solve for correction
            try:
                dxyz = np.linalg.solve(A_corr.T @ A_corr, A_corr.T @ l_corr)
            except np.linalg.LinAlgError:
                break

            xyz += dxyz

            # Check convergence
            if np.max(np.abs(dxyz)) < tol:
                break

        coordinates[i] = xyz

        # Final residuals (using undistorted coordinates)
        X, Y, Z = xyz
        denom1 = L1[8]*X + L1[9]*Y + L1[10]*Z + 1.0
        denom2 = L2[8]*X + L2[9]*Y + L2[10]*Z + 1.0
        if abs(denom1) > 1e-20 and abs(denom2) > 1e-20:
            x1_calc = (L1[0]*X + L1[1]*Y + L1[2]*Z + L1[3]) / denom1
            y1_calc = (L1[4]*X + L1[5]*Y + L1[6]*Z + L1[7]) / denom1
            x2_calc = (L2[0]*X + L2[1]*Y + L2[2]*Z + L2[3]) / denom2
            y2_calc = (L2[4]*X + L2[5]*Y + L2[6]*Z + L2[7]) / denom2
            residuals.append((
                x1 - x1_calc, y1 - y1_calc,
                x2 - x2_calc, y2 - y2_calc,
            ))
        else:
            residuals.append((0.0, 0.0, 0.0, 0.0))

        # Intersection angle
        angle = _compute_intersection_angle_dlt(L1, L2, x1, y1, x2, y2)
        angles.append(angle)

    # Overall accuracy
    all_v = np.array(residuals).flatten()
    dof = 4 * n - 3 * n  # 4 equations per point, 3 unknowns per point
    if dof > 0:
        sigma0 = np.sqrt(np.sum(all_v**2) / dof)
    else:
        sigma0 = 0.0

    return ForwardIntersectionResult(
        point_ids=point_ids,
        coordinates=coordinates,
        residuals=residuals,
        sigma0=sigma0,
        intersection_angles=angles,
    )

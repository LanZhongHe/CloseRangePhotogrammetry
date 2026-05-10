"""Direct Linear Transform (DLT) algorithm."""

from dataclasses import dataclass
import numpy as np

from .camera_model import (
    CameraIntrinsics, ExteriorOrientation, DistortionCoefficients,
    SolveConfig, compute_distortion,
)
from .matching import MatchedPoint


@dataclass
class DLTResult:
    """Results from DLT computation."""
    L_params: np.ndarray                    # 11 DLT parameters
    intrinsics: CameraIntrinsics            # derived interior orientation
    exterior: ExteriorOrientation           # derived exterior orientation
    distortion: DistortionCoefficients      # estimated distortion coefficients
    sigma0: float                           # a posteriori standard deviation
    residuals: list[tuple[float, float]]    # (vx, vy) per point
    param_std: dict[str, float]             # std dev of each L parameter
    num_iterations: int


def _build_dlt_system(
    points: list[MatchedPoint],
) -> tuple[np.ndarray, np.ndarray]:
    """Build the coefficient matrix A and observation vector b for DLT.

    Each point contributes 2 equations:
        L1*X + L2*Y + L3*Z + L4 - x*L9*X - x*L10*Y - x*L11*Z = x
        L5*X + L6*Y + L7*Z + L8 - y*L9*X - y*L10*Y - y*L11*Z = y

    Returns:
        A: (2n, 11) coefficient matrix
        b: (2n,) observation vector
    """
    n = len(points)
    A = np.zeros((2 * n, 11))
    b = np.zeros(2 * n)

    for i, pt in enumerate(points):
        X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
        x, y = pt.image_x_mm, pt.image_y_mm

        # x-equation (row 2i)
        A[2*i, 0] = X
        A[2*i, 1] = Y
        A[2*i, 2] = Z
        A[2*i, 3] = 1.0
        # cols 4-7 are 0
        A[2*i, 8] = -x * X
        A[2*i, 9] = -x * Y
        A[2*i, 10] = -x * Z
        b[2*i] = x

        # y-equation (row 2i+1)
        # cols 0-3 are 0
        A[2*i+1, 4] = X
        A[2*i+1, 5] = Y
        A[2*i+1, 6] = Z
        A[2*i+1, 7] = 1.0
        A[2*i+1, 8] = -y * X
        A[2*i+1, 9] = -y * Y
        A[2*i+1, 10] = -y * Z
        b[2*i+1] = y

    return A, b


def dlt_solve(matched_points: list[MatchedPoint]) -> DLTResult:
    """Compute 11 DLT parameters from matched points (no distortion correction).

    Minimum 6 points required (11 unknowns, 2n equations).

    Returns:
        DLTResult with L parameters, derived orientation, and accuracy assessment
    """
    n = len(matched_points)
    if n < 6:
        raise ValueError(f"DLT requires at least 6 points, got {n}")

    A, b = _build_dlt_system(matched_points)

    # Solve: L = (A^T A)^{-1} A^T b
    AtA = A.T @ A
    Atb = A.T @ b
    L_params = np.linalg.solve(AtA, Atb)

    # Residuals
    v = A @ L_params - b
    residuals = [(v[2*i], v[2*i+1]) for i in range(n)]

    # Accuracy
    dof = 2 * n - 11
    if dof > 0:
        sigma0 = np.sqrt(v @ v / dof)
        try:
            Qll = np.linalg.inv(AtA)
            param_std = {}
            for i in range(11):
                param_std[f"L{i+1}"] = sigma0 * np.sqrt(abs(Qll[i, i]))
        except np.linalg.LinAlgError:
            param_std = {}
            sigma0 = 0.0
    else:
        sigma0 = 0.0
        param_std = {}

    # Derive orientation
    intrinsics, exterior = derive_orientation(L_params)

    return DLTResult(
        L_params=L_params,
        intrinsics=intrinsics,
        exterior=exterior,
        distortion=DistortionCoefficients(),
        sigma0=sigma0,
        residuals=residuals,
        param_std=param_std,
        num_iterations=1,
    )


def derive_orientation(L: np.ndarray) -> tuple[CameraIntrinsics, ExteriorOrientation]:
    """Derive interior and exterior orientation from 11 DLT parameters.

    The relationship between L parameters and the projection matrix:
        [L1 L2 L3 L4]       [R11 R12 R13 Tx]   [f  0 x0]
        [L5 L6 L7 L8] = -1  [R21 R22 R23 Ty] @ [0  f y0] / tz
        [L9 L10 L11 1]   tz [R31 R32 R33 Tz]   [0  0  1]

    Returns:
        (CameraIntrinsics, ExteriorOrientation)
    """
    L1, L2, L3, L4 = L[0], L[1], L[2], L[3]
    L5, L6, L7, L8 = L[4], L[5], L[6], L[7]
    L9, L10, L11 = L[8], L[9], L[10]

    # Principal point
    denom = L9**2 + L10**2 + L11**2
    if denom < 1e-20:
        raise ValueError("Degenerate DLT solution: L9^2 + L10^2 + L11^2 ≈ 0")

    x0 = -(L1 * L9 + L2 * L10 + L3 * L11) / denom
    y0 = -(L5 * L9 + L6 * L10 + L7 * L11) / denom

    # Focal length
    f_sq = 1.0 / denom - x0**2 - y0**2
    if f_sq < 0:
        f = abs(np.sqrt(abs(f_sq)))
    else:
        f = np.sqrt(f_sq)

    intrinsics = CameraIntrinsics(f=f, x0=x0, y0=y0)

    # Extract rotation matrix and translation
    # The 3x3 sub-matrix M = [L1 L2 L3; L5 L6 L7; L9 L10 L11]
    # M = -1/tz * R @ K  where K is the intrinsic matrix
    # R = -tz * M @ K^{-1}

    K = np.array([
        [f, 0, x0],
        [0, f, y0],
        [0, 0, 1],
    ])
    K_inv = np.linalg.inv(K)

    M = np.array([
        [L1, L2, L3],
        [L5, L6, L7],
        [L9, L10, L11],
    ])

    # tz is negative of the scale factor
    # From the third row: tz = -1 / norm(R3) where R3 = M[2,:] @ K_inv
    # But simpler: tz = -1 / sqrt(denom) (from the third row of R being unit vector)
    # Actually: the scale factor lambda = -1/tz, and lambda^2 * (f^2 + x0^2 + y0^2) = L9^2 + L10^2 + L11^2
    # Wait, that's not quite right. Let me use a more direct approach.

    # From M = -1/tz * R @ K:
    # R = -tz * M @ K_inv
    # We need tz. From R's third row being a unit vector:
    # norm(R[2,:]) = 1
    # R[2,:] = -tz * [L9, L10, L11] @ K_inv
    # |tz| * norm([L9, L10, L11] @ K_inv) = 1

    r3_raw = np.array([L9, L10, L11]) @ K_inv
    tz_mag = 1.0 / np.linalg.norm(r3_raw)

    # Determine sign of tz: tz should be negative (camera looks at scene)
    # The third row of R should point in the -Z direction (toward scene)
    # R[2,2] = cw*cp should be positive for small rotations
    R_est = -tz_mag * M @ K_inv

    # Ensure R is a proper rotation (project to SO(3) via SVD)
    U, _, Vt = np.linalg.svd(R_est)
    R = U @ Vt

    # If det(R) < 0, flip sign
    if np.linalg.det(R) < 0:
        R = -R

    # Extract omega, phi, kappa from R
    # R = R_kappa @ R_phi @ R_omega
    # R[2,0] = -sin(phi)
    # R[2,1] = sin(omega)*cos(phi)
    # R[2,2] = cos(omega)*cos(phi)
    phi = -np.arcsin(np.clip(R[2, 0], -1, 1))
    cp = np.cos(phi)
    if abs(cp) > 1e-10:
        omega = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        kappa = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        omega = 0.0
        kappa = np.arctan2(-R[0, 1], R[1, 1])

    # Camera center: C = -R^T @ t, where t = [L4, L8, 1]^T * tz
    # From M = -1/tz * R @ K:
    # [L4; L8; 1] = -1/tz * (R @ K @ [Xs; Ys; Zs] + t)
    # Actually, the translation vector t = -R @ C
    # [L4, L8, 1] = -1/tz * t = 1/tz * R @ C
    # So C = tz * R^T @ [L4, L8, 1]

    # More precisely: the full projection is [L1..L4; L5..L8; L9..L11,1] = -1/tz * [R|t] @ K_aug
    # where K_aug = [K 0; 0 1]
    # The last column: [L4; L8; 1] = -1/tz * t
    # t = -tz * [L4; L8; 1]
    # C = -R^T @ t = tz * R^T @ [L4; L8; 1]

    # But wait - the third element should be 1, but L[10] is L11, not 1.
    # The correct relationship: the bottom-right element of the augmented matrix is 1.
    # So the translation part is: t = -tz * [L4; L8; 1]
    # We use tz (not tz_mag) with the correct sign

    # tz sign: R = -tz * M @ K_inv, and we found R with positive det
    # So tz = -tz_mag (negative, as camera looks at scene)
    tz = -tz_mag

    t = -tz * np.array([L4, L8, 1.0])
    C = -R.T @ t  # camera center in object coords

    exterior = ExteriorOrientation(
        Xs=C[0], Ys=C[1], Zs=C[2],
        omega=omega, phi=phi, kappa=kappa,
    )

    return intrinsics, exterior


def dlt_with_distortion(
    matched_points: list[MatchedPoint],
    solve_config: SolveConfig,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
) -> DLTResult:
    """Iterative DLT with lens distortion correction.

    Steps:
    1. Initial DLT without distortion
    2. Derive f, x0, y0 from L parameters
    3. Compute distortion corrections using current distortion coefficients
    4. Apply corrections to image coordinates
    5. Re-run DLT with corrected coordinates
    6. Repeat until L parameters converge

    Args:
        matched_points: matched point pairs
        solve_config: which distortion parameters to solve
        max_iterations: iteration limit
        convergence_threshold: L-parameter change threshold

    Returns:
        DLTResult with distortion-corrected results
    """
    n = len(matched_points)
    if n < 6:
        raise ValueError(f"DLT requires at least 6 points, got {n}")

    # Work on copies of image coordinates
    corrected_points = [
        MatchedPoint(
            control_id=pt.control_id,
            pixel_x=pt.pixel_x, pixel_y=pt.pixel_y,
            image_x_mm=pt.image_x_mm, image_y_mm=pt.image_y_mm,
            obj_x=pt.obj_x, obj_y=pt.obj_y, obj_z=pt.obj_z,
            is_manual=pt.is_manual,
        )
        for pt in matched_points
    ]

    dist = DistortionCoefficients()
    num_iter = 0
    L_prev = None

    for iteration in range(max_iterations):
        num_iter = iteration + 1

        # Run basic DLT on corrected coordinates
        A, b = _build_dlt_system(corrected_points)
        AtA = A.T @ A
        Atb = A.T @ b
        L_params = np.linalg.solve(AtA, Atb)

        # Check convergence
        if L_prev is not None:
            if np.max(np.abs(L_params - L_prev)) < convergence_threshold:
                break
        L_prev = L_params.copy()

        # Derive interior orientation from L
        intrinsics, _ = derive_orientation(L_params)
        f = intrinsics.f
        x0 = intrinsics.x0
        y0 = intrinsics.y0

        # Compute distortion corrections and update corrected coordinates
        # For DLT distortion, we use K1 as default distortion parameter
        K1 = dist.K1 if dist.K1 != 0 else 1e-8  # small default

        for i, pt in enumerate(matched_points):
            dx, dy = compute_distortion(
                pt.image_x_mm, pt.image_y_mm, x0, y0,
                dist.K1, dist.K2, dist.K3,
                dist.P1, dist.P2,
                dist.A1, dist.A2, dist.B1, dist.B2,
            )
            corrected_points[i].image_x_mm = pt.image_x_mm - dx
            corrected_points[i].image_y_mm = pt.image_y_mm - dy

        # Estimate K1 from residuals (simple approach)
        # Use the difference between original and corrected coordinates
        # This is a rough estimate; proper distortion estimation would use
        # the resection approach
        if solve_config.solve_k1 and iteration > 0:
            total_r2 = 0.0
            total_dx_r2 = 0.0
            for i, pt in enumerate(matched_points):
                dx_orig = pt.image_x_mm - x0
                dy_orig = pt.image_y_mm - y0
                r2 = dx_orig**2 + dy_orig**2
                # Residual before correction
                A_i = A[2*i:2*i+2, :]
                b_i = b[2*i:2*i+2]
                residual = A_i @ L_params - b_i
                total_r2 += r2 * r2
                total_dx_r2 += residual[0] * dx_orig * r2
            if total_r2 > 1e-20:
                dist.K1 = -total_dx_r2 / total_r2

    # Final results
    v = A @ L_params - b
    residuals = [(v[2*i], v[2*i+1]) for i in range(n)]
    dof = 2 * n - 11
    if dof > 0:
        sigma0 = np.sqrt(v @ v / dof)
        try:
            Qll = np.linalg.inv(AtA)
            param_std = {}
            for i in range(11):
                param_std[f"L{i+1}"] = sigma0 * np.sqrt(abs(Qll[i, i]))
        except np.linalg.LinAlgError:
            param_std = {}
            sigma0 = 0.0
    else:
        sigma0 = 0.0
        param_std = {}

    intrinsics_out, exterior = derive_orientation(L_params)

    return DLTResult(
        L_params=L_params,
        intrinsics=intrinsics_out,
        exterior=exterior,
        distortion=dist,
        sigma0=sigma0,
        residuals=residuals,
        param_std=param_std,
        num_iterations=num_iter,
    )

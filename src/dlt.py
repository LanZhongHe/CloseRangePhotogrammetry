"""Direct Linear Transform (DLT) algorithm."""

from dataclasses import dataclass, field
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
    check_point_ids: list[str] = field(default_factory=list)
    check_point_residuals: list[tuple[float, float]] = field(default_factory=list)
    check_point_sigma0: float = 0.0


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


def _compute_check_point_residuals(
    check_points: list[MatchedPoint],
    L_params: np.ndarray,
    distortion: DistortionCoefficients,
    intrinsics: CameraIntrinsics,
) -> tuple[list[str], list[tuple[float, float]], float]:
    """Compute residuals for check points using solved DLT parameters."""
    check_ids = []
    check_residuals = []
    check_v = []
    for pt in check_points:
        x, y = pt.image_x_mm, pt.image_y_mm
        X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
        # Apply distortion correction if any
        if any([distortion.K1, distortion.K2, distortion.P1, distortion.P2]):
            from .camera_model import compute_distortion as cd
            dx, dy = cd(x, y, intrinsics.x0, intrinsics.y0,
                        distortion.K1, distortion.K2, distortion.K3,
                        distortion.P1, distortion.P2)
            x, y = x - dx, y - dy
        # DLT projection: compute expected image coords
        denom = L_params[8]*X + L_params[9]*Y + L_params[10]*Z + 1.0
        if abs(denom) < 1e-20:
            check_ids.append(pt.control_id)
            check_residuals.append((0.0, 0.0))
            continue
        x_calc = (L_params[0]*X + L_params[1]*Y + L_params[2]*Z + L_params[3]) / denom
        y_calc = (L_params[4]*X + L_params[5]*Y + L_params[6]*Z + L_params[7]) / denom
        vx = x - x_calc
        vy = y - y_calc
        check_ids.append(pt.control_id)
        check_residuals.append((vx, vy))
        check_v.extend([vx, vy])

    check_sigma0 = 0.0
    if check_v:
        check_sigma0 = np.sqrt(np.mean(np.array(check_v)**2))
    return check_ids, check_residuals, check_sigma0


def dlt_solve(matched_points: list[MatchedPoint]) -> DLTResult:
    """Compute 11 DLT parameters from matched points (no distortion correction).

    Minimum 6 control points required (11 unknowns, 2n equations).

    Returns:
        DLTResult with L parameters, derived orientation, and accuracy assessment
    """
    control_points = [p for p in matched_points if not p.is_check]
    check_points = [p for p in matched_points if p.is_check]
    n = len(control_points)
    if n < 6:
        raise ValueError(f"DLT requires at least 6 control points, got {n}")

    A, b = _build_dlt_system(control_points)

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

    # Check point residuals
    check_ids, check_residuals, check_sigma0 = _compute_check_point_residuals(
        check_points, L_params, DistortionCoefficients(), intrinsics,
    )

    return DLTResult(
        L_params=L_params,
        intrinsics=intrinsics,
        exterior=exterior,
        distortion=DistortionCoefficients(),
        sigma0=sigma0,
        residuals=residuals,
        param_std=param_std,
        num_iterations=1,
        check_point_ids=check_ids,
        check_point_residuals=check_residuals,
        check_point_sigma0=check_sigma0,
    )


def derive_orientation(L: np.ndarray) -> tuple[CameraIntrinsics, ExteriorOrientation]:
    """Derive interior and exterior orientation from 11 DLT parameters.

    The DLT projection: x = (L1*X+L2*Y+L3*Z+L4) / (L9*X+L10*Y+L11*Z+1)

    Decomposition with K_eff = [[-f,0,x0],[0,-f,y0],[0,0,1]]:
        M = lambda * K_eff @ R,  where M = [L1 L2 L3; L5 L6 L7; L9 L10 L11]
    This matches the photogrammetric collinearity equation: x - x0 = -f * Xbar/Zbar

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

    x0 = (L1 * L9 + L2 * L10 + L3 * L11) / denom
    y0 = (L5 * L9 + L6 * L10 + L7 * L11) / denom

    # Focal length: from ||row1||^2 = lambda^2 * (f^2 + x0^2)
    f_sq = (L1**2 + L2**2 + L3**2) / denom - x0**2
    if f_sq < 0:
        f = abs(np.sqrt(abs(f_sq)))
    else:
        f = np.sqrt(f_sq)

    intrinsics = CameraIntrinsics(f=f, x0=x0, y0=y0)

    # Extract rotation matrix
    # M = lambda * K_eff @ R  =>  R = K_eff_inv @ M / lambda
    # |lambda| = sqrt(denom), tz_mag = 1/|lambda| = 1/sqrt(denom)
    K_eff = np.array([
        [-f, 0, x0],
        [0, -f, y0],
        [0,  0,  1],
    ])
    K_eff_inv = np.linalg.inv(K_eff)

    M = np.array([
        [L1, L2, L3],
        [L5, L6, L7],
        [L9, L10, L11],
    ])

    tz_mag = 1.0 / np.sqrt(denom)

    # R = K_eff_inv @ M / lambda = K_eff_inv @ M * sign(lambda) * tz_mag
    # For lambda > 0 (normal case): R_est = K_eff_inv @ M * tz_mag
    # SVD projection to SO(3) handles the sign ambiguity
    R_est = K_eff_inv @ M * tz_mag

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

    # Camera center: at C the projection denominator is zero
    # L9*Cx + L10*Cy + L11*Cz + 1 = 0  and similarly for numerator rows
    # Solve M @ C = (-L4, -L8, -1)
    b = np.array([-L4, -L8, -1.0])
    C = np.linalg.solve(M, b)

    exterior = ExteriorOrientation(
        Xs=C[0], Ys=C[1], Zs=C[2],
        omega=omega, phi=phi, kappa=kappa,
    )

    return intrinsics, exterior


def _estimate_distortion_from_residuals(
    matched_points: list[MatchedPoint],
    neg_residuals: np.ndarray,
    x0: float, y0: float,
    solve_config: SolveConfig,
    current_dist: DistortionCoefficients,
) -> DistortionCoefficients:
    """Estimate distortion coefficients from (x_measured - x_computed) residuals.

    The caller must pass NEGATED DLT residuals, i.e., -(A @ L - b) = x_measured - x_computed,
    since the DLT residual v = x_computed - x_measured ≈ -distortion.

    The model solved is:
        rhs_x = K1*dx*r2 + K2*dx*r4 + K3*dx*r6 + P1*(r2+2*dx^2) + 2*P2*dx*dy
        rhs_y = K1*dy*r2 + K2*dy*r4 + K3*dy*r6 + 2*P1*dx*dy + P2*(r2+2*dy^2)
    where rhs = x_measured - x_computed.

    Args:
        matched_points: original (uncorrected) matched points
        neg_residuals: negated 2n residual vector (= x_measured - x_computed)
        x0, y0: principal point in mm
        solve_config: which parameters to estimate
        current_dist: current distortion coefficients (carries over unsolved params)

    Returns:
        Updated DistortionCoefficients
    """
    n = len(matched_points)
    # Determine which parameters to solve
    param_names = []
    if solve_config.solve_k1: param_names.append("K1")
    if solve_config.solve_k2: param_names.append("K2")
    if solve_config.solve_k3: param_names.append("K3")
    if solve_config.solve_p1: param_names.append("P1")
    if solve_config.solve_p2: param_names.append("P2")

    if not param_names:
        return current_dist

    n_params = len(param_names)
    A_dist = np.zeros((2 * n, n_params))
    b_dist = neg_residuals.copy()

    for i, pt in enumerate(matched_points):
        dx = pt.image_x_mm - x0
        dy = pt.image_y_mm - y0
        r2 = dx * dx + dy * dy
        r4 = r2 * r2

        for j, pname in enumerate(param_names):
            if pname == "K1":
                A_dist[2*i, j] = dx * r2
                A_dist[2*i+1, j] = dy * r2
            elif pname == "K2":
                A_dist[2*i, j] = dx * r4
                A_dist[2*i+1, j] = dy * r4
            elif pname == "K3":
                r6 = r4 * r2
                A_dist[2*i, j] = dx * r6
                A_dist[2*i+1, j] = dy * r6
            elif pname == "P1":
                A_dist[2*i, j] = r2 + 2 * dx * dx
                A_dist[2*i+1, j] = 2 * dx * dy
            elif pname == "P2":
                A_dist[2*i, j] = 2 * dx * dy
                A_dist[2*i+1, j] = r2 + 2 * dy * dy

    # Least squares: A_dist @ coeffs = b_dist
    try:
        AtA = A_dist.T @ A_dist
        Atb = A_dist.T @ b_dist
        coeffs = np.linalg.solve(AtA, Atb)
    except np.linalg.LinAlgError:
        return current_dist

    # Update distortion coefficients
    dist = DistortionCoefficients(
        K1=current_dist.K1, K2=current_dist.K2, K3=current_dist.K3,
        P1=current_dist.P1, P2=current_dist.P2,
    )
    for j, pname in enumerate(param_names):
        setattr(dist, pname, coeffs[j])

    return dist


def _build_augmented_dlt_system(
    points: list[MatchedPoint],
    x0: float, y0: float,
    solve_config: SolveConfig,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build augmented DLT system with distortion columns.

    Each point contributes 2 equations:
        L1*X + L2*Y + L3*Z + L4 - x*(L9*X + L10*Y + L11*Z) + dist_x = x
        L5*X + L6*Y + L7*Z + L8 - y*(L9*X + L10*Y + L11*Z) + dist_y = y

    where dist_x/disty are the distortion correction terms:
        dist_x = K1*dx*r2 + K2*dx*r4 + K3*dx*r6 + P1*(r2+2*dx^2) + 2*P2*dx*dy
        dist_y = K1*dy*r2 + K2*dy*r4 + K3*dy*r6 + 2*P1*dx*dy + P2*(r2+2*dy^2)

    with dx = x - x0, dy = y - y0, r2 = dx^2 + dy^2.

    This prevents L parameters from absorbing distortion effects.

    Returns:
        A: (2n, 11+k) coefficient matrix
        b: (2n,) observation vector
        param_names: list of parameter names (11 L-params + distortion params)
    """
    n = len(points)

    # Determine distortion parameters to include
    dist_params = []
    if solve_config.solve_k1: dist_params.append("K1")
    if solve_config.solve_k2: dist_params.append("K2")
    if solve_config.solve_k3: dist_params.append("K3")
    if solve_config.solve_p1: dist_params.append("P1")
    if solve_config.solve_p2: dist_params.append("P2")

    n_cols = 11 + len(dist_params)
    A = np.zeros((2 * n, n_cols))
    b = np.zeros(2 * n)

    for i, pt in enumerate(points):
        X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
        x, y = pt.image_x_mm, pt.image_y_mm

        # Standard DLT columns (L1..L11)
        A[2*i, 0] = X
        A[2*i, 1] = Y
        A[2*i, 2] = Z
        A[2*i, 3] = 1.0
        A[2*i, 8] = -x * X
        A[2*i, 9] = -x * Y
        A[2*i, 10] = -x * Z
        b[2*i] = x

        A[2*i+1, 4] = X
        A[2*i+1, 5] = Y
        A[2*i+1, 6] = Z
        A[2*i+1, 7] = 1.0
        A[2*i+1, 8] = -y * X
        A[2*i+1, 9] = -y * Y
        A[2*i+1, 10] = -y * Z
        b[2*i+1] = y

        # Distortion columns
        dx = x - x0
        dy = y - y0
        r2 = dx * dx + dy * dy
        r4 = r2 * r2

        for j, pname in enumerate(dist_params):
            col = 11 + j
            if pname == "K1":
                A[2*i, col] = dx * r2
                A[2*i+1, col] = dy * r2
            elif pname == "K2":
                A[2*i, col] = dx * r4
                A[2*i+1, col] = dy * r4
            elif pname == "K3":
                r6 = r4 * r2
                A[2*i, col] = dx * r6
                A[2*i+1, col] = dy * r6
            elif pname == "P1":
                A[2*i, col] = r2 + 2 * dx * dx
                A[2*i+1, col] = 2 * dx * dy
            elif pname == "P2":
                A[2*i, col] = 2 * dx * dy
                A[2*i+1, col] = r2 + 2 * dy * dy

    param_names = [f"L{i+1}" for i in range(11)] + dist_params
    return A, b, param_names


def dlt_with_distortion(
    matched_points: list[MatchedPoint],
    solve_config: SolveConfig,
    max_iterations: int = 10,
    convergence_threshold: float = 1e-6,
) -> DLTResult:
    """DLT with joint distortion estimation using augmented system.

    Solves for L parameters and distortion coefficients simultaneously in a
    single least-squares system. This prevents L parameters from absorbing
    distortion effects. The distortion center (x0, y0) is updated iteratively
    from the L parameters.

    Args:
        matched_points: matched point pairs
        solve_config: which distortion parameters to solve (K1, K2, K3, P1, P2)
        max_iterations: iteration limit for updating distortion center
        convergence_threshold: parameter change threshold

    Returns:
        DLTResult with distortion-corrected results
    """
    control_points = [p for p in matched_points if not p.is_check]
    check_points = [p for p in matched_points if p.is_check]
    n = len(control_points)
    if n < 6:
        raise ValueError(f"DLT requires at least 6 control points, got {n}")

    # Count distortion parameters
    dist_params = []
    if solve_config.solve_k1: dist_params.append("K1")
    if solve_config.solve_k2: dist_params.append("K2")
    if solve_config.solve_k3: dist_params.append("K3")
    if solve_config.solve_p1: dist_params.append("P1")
    if solve_config.solve_p2: dist_params.append("P2")
    n_dist = len(dist_params)
    n_total = 11 + n_dist
    min_pts = (n_total + 1) // 2
    if n < min_pts:
        raise ValueError(
            f"DLT with {n_dist} distortion params requires at least {min_pts} "
            f"control points, got {n}"
        )

    # Phase 1: Initial DLT without distortion to get x0, y0
    A_dlt, b_dlt = _build_dlt_system(control_points)
    L_init = np.linalg.solve(A_dlt.T @ A_dlt, A_dlt.T @ b_dlt)
    intrinsics, _ = derive_orientation(L_init)
    x0, y0 = intrinsics.x0, intrinsics.y0

    # Phase 2: Iterative joint solve (update distortion center each iteration)
    num_iter = 0
    params = np.zeros(n_total)

    for iteration in range(max_iterations):
        num_iter = iteration + 1
        params_prev = params.copy()

        # Build and solve augmented system with current x0, y0
        A, b, param_names = _build_augmented_dlt_system(
            control_points, x0, y0, solve_config,
        )
        AtA = A.T @ A
        Atb = A.T @ b
        params = np.linalg.solve(AtA, Atb)

        # Extract L parameters and update intrinsics
        L_params = params[:11]
        intrinsics_new, _ = derive_orientation(L_params)

        # Update distortion center for next iteration
        x0, y0 = intrinsics_new.x0, intrinsics_new.y0

        # Convergence check
        if np.max(np.abs(params - params_prev)) < convergence_threshold:
            break

    # Extract final results
    L_params = params[:11]
    intrinsics_out, exterior = derive_orientation(L_params)

    # Extract distortion coefficients
    dist = DistortionCoefficients()
    for j, pname in enumerate(dist_params):
        setattr(dist, pname, params[11 + j])

    # Residuals from the augmented system using full parameter vector
    v_full = A @ params - b
    residuals = [(v_full[2*i], v_full[2*i+1]) for i in range(n)]

    # Accuracy assessment
    dof = 2 * n - n_total
    if dof > 0:
        sigma0 = np.sqrt(v_full @ v_full / dof)
        try:
            Qll = np.linalg.inv(AtA)
            param_std = {}
            for i in range(11):
                param_std[f"L{i+1}"] = sigma0 * np.sqrt(abs(Qll[i, i]))
            for j, pname in enumerate(dist_params):
                param_std[pname] = sigma0 * np.sqrt(abs(Qll[11 + j, 11 + j]))
        except np.linalg.LinAlgError:
            param_std = {}
            sigma0 = 0.0
    else:
        sigma0 = 0.0
        param_std = {}

    # Check point residuals
    check_ids, check_residuals, check_sigma0 = _compute_check_point_residuals(
        check_points, L_params, dist, intrinsics_out,
    )

    return DLTResult(
        L_params=L_params,
        intrinsics=intrinsics_out,
        exterior=exterior,
        distortion=dist,
        sigma0=sigma0,
        residuals=residuals,
        param_std=param_std,
        num_iterations=num_iter,
        check_point_ids=check_ids,
        check_point_residuals=check_residuals,
        check_point_sigma0=check_sigma0,
    )

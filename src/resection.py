"""Single-image space resection using collinearity equations."""

from dataclasses import dataclass, field
import numpy as np

from .camera_model import (
    CameraIntrinsics, ExteriorOrientation, DistortionCoefficients,
    SolveConfig, rotation_matrix, rotation_matrix_derivatives,
    compute_distortion, distortion_derivatives,
)
from .matching import MatchedPoint


@dataclass
class ResectionResult:
    """Results from space resection."""
    exterior: ExteriorOrientation
    intrinsics: CameraIntrinsics
    distortion: DistortionCoefficients
    sigma0: float
    residuals: list[tuple[float, float]]
    param_covariance: np.ndarray
    param_std: dict[str, float]
    num_iterations: int
    converged: bool


def _build_param_vector(
    ext: ExteriorOrientation,
    intr: CameraIntrinsics,
    dist: DistortionCoefficients,
    config: SolveConfig,
) -> np.ndarray:
    """Assemble the unknown parameter vector from current values."""
    params = [ext.Xs, ext.Ys, ext.Zs, ext.omega, ext.phi, ext.kappa]
    if config.solve_f:
        params.append(intr.f)
    if config.solve_x0:
        params.append(intr.x0)
    if config.solve_y0:
        params.append(intr.y0)
    if config.solve_k1:
        params.append(dist.K1)
    if config.solve_k2:
        params.append(dist.K2)
    if config.solve_k3:
        params.append(dist.K3)
    if config.solve_p1:
        params.append(dist.P1)
    if config.solve_p2:
        params.append(dist.P2)
    if config.solve_a1:
        params.append(dist.A1)
    if config.solve_a2:
        params.append(dist.A2)
    if config.solve_b1:
        params.append(dist.B1)
    if config.solve_b2:
        params.append(dist.B2)
    return np.array(params)


def _unpack_param_vector(
    params: np.ndarray,
    intr: CameraIntrinsics,
    dist: DistortionCoefficients,
    config: SolveConfig,
) -> tuple[ExteriorOrientation, CameraIntrinsics, DistortionCoefficients]:
    """Extract parameter values from the solution vector."""
    ext = ExteriorOrientation(
        Xs=params[0], Ys=params[1], Zs=params[2],
        omega=params[3], phi=params[4], kappa=params[5],
    )
    idx = 6
    intr_out = CameraIntrinsics(
        f=intr.f, x0=intr.x0, y0=intr.y0,
        sensor_width=intr.sensor_width, sensor_height=intr.sensor_height,
        img_width=intr.img_width, img_height=intr.img_height,
    )
    dist_out = DistortionCoefficients(
        K1=dist.K1, K2=dist.K2, K3=dist.K3,
        P1=dist.P1, P2=dist.P2,
        A1=dist.A1, A2=dist.A2, B1=dist.B1, B2=dist.B2,
    )
    if config.solve_f:
        intr_out.f = params[idx]; idx += 1
    if config.solve_x0:
        intr_out.x0 = params[idx]; idx += 1
    if config.solve_y0:
        intr_out.y0 = params[idx]; idx += 1
    if config.solve_k1:
        dist_out.K1 = params[idx]; idx += 1
    if config.solve_k2:
        dist_out.K2 = params[idx]; idx += 1
    if config.solve_k3:
        dist_out.K3 = params[idx]; idx += 1
    if config.solve_p1:
        dist_out.P1 = params[idx]; idx += 1
    if config.solve_p2:
        dist_out.P2 = params[idx]; idx += 1
    if config.solve_a1:
        dist_out.A1 = params[idx]; idx += 1
    if config.solve_a2:
        dist_out.A2 = params[idx]; idx += 1
    if config.solve_b1:
        dist_out.B1 = params[idx]; idx += 1
    if config.solve_b2:
        dist_out.B2 = params[idx]; idx += 1
    return ext, intr_out, dist_out


def _estimate_initial_exterior(points: list[MatchedPoint]) -> ExteriorOrientation:
    """Estimate initial exterior orientation from matched point distribution.

    Places camera above the centroid and estimates initial rotation angles
    from the camera-to-centroid direction.
    """
    X_mean = np.mean([p.obj_x for p in points])
    Y_mean = np.mean([p.obj_y for p in points])
    Z_mean = np.mean([p.obj_z for p in points])
    Z_range = np.ptp([p.obj_z for p in points])
    offset = max(Z_range * 1.5, 3000.0)

    # Use small non-zero initial angles to avoid rank-deficient Jacobian.
    # When all angles are zero, the kappa derivative column vanishes entirely.
    omega_init = 0.01
    phi_init = 0.01
    kappa_init = 0.01

    return ExteriorOrientation(
        Xs=X_mean, Ys=Y_mean, Zs=Z_mean + offset,
        omega=omega_init, phi=phi_init, kappa=kappa_init,
    )


def space_resection(
    matched_points: list[MatchedPoint],
    intrinsics: CameraIntrinsics,
    solve_config: SolveConfig,
    initial_exterior: ExteriorOrientation | None = None,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> ResectionResult:
    """Run single-image space resection.

    Args:
        matched_points: list of matched (image, object) point pairs
        intrinsics: camera interior orientation
        solve_config: which parameters to solve
        initial_exterior: starting values for exterior orientation (auto-estimated if None)
        max_iter: iteration limit
        tol: convergence threshold on parameter corrections

    Returns:
        ResectionResult with all outputs and accuracy assessment
    """
    n = len(matched_points)
    u = solve_config.num_unknowns
    if n < solve_config.min_points:
        raise ValueError(
            f"Need at least {solve_config.min_points} points, got {n}"
        )

    # Initial values
    if initial_exterior is None:
        ext = _estimate_initial_exterior(matched_points)
    else:
        ext = ExteriorOrientation(
            Xs=initial_exterior.Xs, Ys=initial_exterior.Ys, Zs=initial_exterior.Zs,
            omega=initial_exterior.omega, phi=initial_exterior.phi, kappa=initial_exterior.kappa,
        )
    dist = DistortionCoefficients()
    f = intrinsics.f
    x0 = intrinsics.x0
    y0 = intrinsics.y0

    converged = False
    num_iter = 0
    damping = 1.0  # Levenberg-Marquardt damping factor (start high for robustness)

    for iteration in range(max_iter):
        num_iter = iteration + 1

        # Current rotation matrix and its derivatives
        R = rotation_matrix(ext.omega, ext.phi, ext.kappa)
        dR_do, dR_dp, dR_dk = rotation_matrix_derivatives(ext.omega, ext.phi, ext.kappa)
        a = R.flatten()  # a1..a9

        # Build Jacobian B (2n x u) and residual vector L (2n)
        B = np.zeros((2 * n, u))
        L = np.zeros(2 * n)

        for i, pt in enumerate(matched_points):
            X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
            xi, yi = pt.image_x_mm, pt.image_y_mm

            dX = X - ext.Xs
            dY = Y - ext.Ys
            dZ = Z - ext.Zs

            Xbar = a[0] * dX + a[1] * dY + a[2] * dZ
            Ybar = a[3] * dX + a[4] * dY + a[5] * dZ
            Zbar = a[6] * dX + a[7] * dY + a[8] * dZ

            # Distortion
            dx, dy = compute_distortion(
                xi, yi, x0, y0,
                dist.K1, dist.K2, dist.K3,
                dist.P1, dist.P2,
                dist.A1, dist.A2, dist.B1, dist.B2,
            )

            # Residuals: Fx = xi - x0 + dx + f * Xbar / Zbar = 0 when correct
            Fx = xi - x0 + dx + f * Xbar / Zbar
            Fy = yi - y0 + dy + f * Ybar / Zbar
            L[2 * i] = Fx
            L[2 * i + 1] = Fy

            # Partial derivatives w.r.t. exterior orientation
            # Since L = -F and we solve B @ delta = L, B = -dF/d(params)
            Zbar2 = Zbar * Zbar

            # dF/dXs = -f/Zbar^2 * (a1*Zbar - a7*Xbar), so B = -dF/dXs = f/Zbar^2 * (...)
            # dF/dXs = -f*(a1*Zbar - a9*Xbar)/Zbar^2, B = -dF/dXs
            B[2*i, 0] = f / Zbar2 * (a[0] * Zbar - a[8] * Xbar)
            B[2*i, 1] = f / Zbar2 * (a[1] * Zbar - a[8] * Xbar)
            B[2*i, 2] = f / Zbar2 * (a[2] * Zbar - a[8] * Xbar)
            B[2*i+1, 0] = f / Zbar2 * (a[3] * Zbar - a[8] * Ybar)
            B[2*i+1, 1] = f / Zbar2 * (a[4] * Zbar - a[8] * Ybar)
            B[2*i+1, 2] = f / Zbar2 * (a[5] * Zbar - a[8] * Ybar)

            # dF/domega = -f/Zbar^2 * (dXbar_do * Zbar - Xbar * dZbar_do)
            dXbar_do = dR_do[0, 0] * dX + dR_do[0, 1] * dY + dR_do[0, 2] * dZ
            dYbar_do = dR_do[1, 0] * dX + dR_do[1, 1] * dY + dR_do[1, 2] * dZ
            dZbar_do = dR_do[2, 0] * dX + dR_do[2, 1] * dY + dR_do[2, 2] * dZ
            # Rotation angle derivatives: dF/dangle = f*(dXbar*Zbar - Xbar*dZbar)/Zbar^2
            # B = -dF/dangle
            dXbar_do = dR_do[0, 0] * dX + dR_do[0, 1] * dY + dR_do[0, 2] * dZ
            dYbar_do = dR_do[1, 0] * dX + dR_do[1, 1] * dY + dR_do[1, 2] * dZ
            dZbar_do = dR_do[2, 0] * dX + dR_do[2, 1] * dY + dR_do[2, 2] * dZ
            B[2*i, 3] = -f / Zbar2 * (dXbar_do * Zbar - Xbar * dZbar_do)
            B[2*i+1, 3] = -f / Zbar2 * (dYbar_do * Zbar - Ybar * dZbar_do)

            dXbar_dp = dR_dp[0, 0] * dX + dR_dp[0, 1] * dY + dR_dp[0, 2] * dZ
            dYbar_dp = dR_dp[1, 0] * dX + dR_dp[1, 1] * dY + dR_dp[1, 2] * dZ
            dZbar_dp = dR_dp[2, 0] * dX + dR_dp[2, 1] * dY + dR_dp[2, 2] * dZ
            B[2*i, 4] = -f / Zbar2 * (dXbar_dp * Zbar - Xbar * dZbar_dp)
            B[2*i+1, 4] = -f / Zbar2 * (dYbar_dp * Zbar - Ybar * dZbar_dp)

            dXbar_dk = dR_dk[0, 0] * dX + dR_dk[0, 1] * dY + dR_dk[0, 2] * dZ
            dYbar_dk = dR_dk[1, 0] * dX + dR_dk[1, 1] * dY + dR_dk[1, 2] * dZ
            dZbar_dk = dR_dk[2, 0] * dX + dR_dk[2, 1] * dY + dR_dk[2, 2] * dZ
            B[2*i, 5] = -f / Zbar2 * (dXbar_dk * Zbar - Xbar * dZbar_dk)
            B[2*i+1, 5] = -f / Zbar2 * (dYbar_dk * Zbar - Ybar * dZbar_dk)

            # Intrinsic parameter derivatives
            col = 6
            if solve_config.solve_f:
                B[2*i, col] = -Xbar / Zbar
                B[2*i+1, col] = -Ybar / Zbar
                col += 1
            if solve_config.solve_x0:
                B[2*i, col] = 1.0
                B[2*i+1, col] = 0.0
                col += 1
            if solve_config.solve_y0:
                B[2*i, col] = 0.0
                B[2*i+1, col] = 1.0
                col += 1

            # Distortion parameter derivatives
            dist_derivs = distortion_derivatives(xi, yi, x0, y0, dist.K1, dist.K2, dist.K3, solve_config)
            for pname in solve_config.distortion_param_names:
                ddx, ddy = dist_derivs[pname]
                B[2*i, col] = -f * ddx / Zbar
                B[2*i+1, col] = -f * ddy / Zbar
                col += 1

        # Current cost
        cost = L @ L  # sum of squared residuals (L = -F)

        # Levenberg-Marquardt: (B^T B + lambda * diag(B^T B)) delta = B^T L
        BtB = B.T @ B
        BtL = B.T @ L
        diag_BtB = np.diag(BtB)
        # Ensure positive diagonal for damping
        diag_damped = np.maximum(diag_BtB, 1e-12)

        for _lm_try in range(20):
            A_lm = BtB + damping * np.diag(diag_damped)
            try:
                delta = np.linalg.solve(A_lm, BtL)
            except np.linalg.LinAlgError:
                damping *= 10
                continue

            # Trial update
            params_trial = _build_param_vector(ext, CameraIntrinsics(
                f=f, x0=x0, y0=y0,
                sensor_width=intrinsics.sensor_width, sensor_height=intrinsics.sensor_height,
                img_width=intrinsics.img_width, img_height=intrinsics.img_height,
            ), dist, solve_config)
            params_trial += delta
            ext_trial, intr_trial, dist_trial = _unpack_param_vector(
                params_trial, intrinsics, dist, solve_config
            )

            # Compute trial cost
            R_t = rotation_matrix(ext_trial.omega, ext_trial.phi, ext_trial.kappa)
            a_t = R_t.flatten()
            cost_trial = 0.0
            for i_pt, pt in enumerate(matched_points):
                X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
                xi, yi = pt.image_x_mm, pt.image_y_mm
                dX = X - ext_trial.Xs; dY = Y - ext_trial.Ys; dZ = Z - ext_trial.Zs
                Xb = a_t[0]*dX + a_t[1]*dY + a_t[2]*dZ
                Yb = a_t[3]*dX + a_t[4]*dY + a_t[5]*dZ
                Zb = a_t[6]*dX + a_t[7]*dY + a_t[8]*dZ
                dx_t, dy_t = compute_distortion(
                    xi, yi, intr_trial.x0, intr_trial.y0,
                    dist_trial.K1, dist_trial.K2, dist_trial.K3,
                    dist_trial.P1, dist_trial.P2,
                    dist_trial.A1, dist_trial.A2, dist_trial.B1, dist_trial.B2,
                )
                Fx = xi - intr_trial.x0 + dx_t + intr_trial.f * Xb / Zb
                Fy = yi - intr_trial.y0 + dy_t + intr_trial.f * Yb / Zb
                cost_trial += Fx**2 + Fy**2

            if cost_trial < cost:
                # Accept step
                damping *= 0.3
                ext = ext_trial
                dist = dist_trial
                f = intr_trial.f
                x0 = intr_trial.x0
                y0 = intr_trial.y0
                break
            else:
                # Reject step, increase damping
                damping *= 3.0
        else:
            # Could not find a reducing step
            break

        # Check convergence
        if np.max(np.abs(delta)) < tol:
            converged = True
            break

    # Final accuracy assessment
    R = rotation_matrix(ext.omega, ext.phi, ext.kappa)
    a = R.flatten()
    residuals = []
    v = np.zeros(2 * n)

    for i, pt in enumerate(matched_points):
        X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
        xi, yi = pt.image_x_mm, pt.image_y_mm

        dX = X - ext.Xs
        dY = Y - ext.Ys
        dZ = Z - ext.Zs

        Xbar = a[0] * dX + a[1] * dY + a[2] * dZ
        Ybar = a[3] * dX + a[4] * dY + a[5] * dZ
        Zbar = a[6] * dX + a[7] * dY + a[8] * dZ

        dx, dy = compute_distortion(
            xi, yi, x0, y0,
            dist.K1, dist.K2, dist.K3,
            dist.P1, dist.P2,
            dist.A1, dist.A2, dist.B1, dist.B2,
        )

        vx = xi - x0 + dx + f * Xbar / Zbar
        vy = yi - y0 + dy + f * Ybar / Zbar
        v[2 * i] = vx
        v[2 * i + 1] = vy
        residuals.append((vx, vy))

    # Rebuild B for covariance
    B = np.zeros((2 * n, u))
    dR_do, dR_dp, dR_dk = rotation_matrix_derivatives(ext.omega, ext.phi, ext.kappa)
    for i, pt in enumerate(matched_points):
        X, Y, Z = pt.obj_x, pt.obj_y, pt.obj_z
        xi, yi = pt.image_x_mm, pt.image_y_mm
        dX = X - ext.Xs; dY = Y - ext.Ys; dZ = Z - ext.Zs
        Xbar = a[0]*dX + a[1]*dY + a[2]*dZ
        Ybar = a[3]*dX + a[4]*dY + a[5]*dZ
        Zbar = a[6]*dX + a[7]*dY + a[8]*dZ
        Zbar2 = Zbar * Zbar

        B[2*i, 0] = f/Zbar2*(a[0]*Zbar - a[8]*Xbar)
        B[2*i, 1] = f/Zbar2*(a[1]*Zbar - a[8]*Xbar)
        B[2*i, 2] = f/Zbar2*(a[2]*Zbar - a[8]*Xbar)
        B[2*i+1, 0] = f/Zbar2*(a[3]*Zbar - a[8]*Ybar)
        B[2*i+1, 1] = f/Zbar2*(a[4]*Zbar - a[8]*Ybar)
        B[2*i+1, 2] = f/Zbar2*(a[5]*Zbar - a[8]*Ybar)

        dXb=dR_do[0,0]*dX+dR_do[0,1]*dY+dR_do[0,2]*dZ
        dYb=dR_do[1,0]*dX+dR_do[1,1]*dY+dR_do[1,2]*dZ
        dZb=dR_do[2,0]*dX+dR_do[2,1]*dY+dR_do[2,2]*dZ
        B[2*i,3]=-f/Zbar2*(dXb*Zbar-Xbar*dZb)
        B[2*i+1,3]=-f/Zbar2*(dYb*Zbar-Ybar*dZb)
        dXb=dR_dp[0,0]*dX+dR_dp[0,1]*dY+dR_dp[0,2]*dZ
        dYb=dR_dp[1,0]*dX+dR_dp[1,1]*dY+dR_dp[1,2]*dZ
        dZb=dR_dp[2,0]*dX+dR_dp[2,1]*dY+dR_dp[2,2]*dZ
        B[2*i,4]=-f/Zbar2*(dXb*Zbar-Xbar*dZb)
        B[2*i+1,4]=-f/Zbar2*(dYb*Zbar-Ybar*dZb)
        dXb=dR_dk[0,0]*dX+dR_dk[0,1]*dY+dR_dk[0,2]*dZ
        dYb=dR_dk[1,0]*dX+dR_dk[1,1]*dY+dR_dk[1,2]*dZ
        dZb=dR_dk[2,0]*dX+dR_dk[2,1]*dY+dR_dk[2,2]*dZ
        B[2*i,5]=-f/Zbar2*(dXb*Zbar-Xbar*dZb)
        B[2*i+1,5]=-f/Zbar2*(dYb*Zbar-Ybar*dZb)

        col = 6
        if solve_config.solve_f:
            B[2*i,col]=-Xbar/Zbar; B[2*i+1,col]=-Ybar/Zbar; col+=1
        if solve_config.solve_x0:
            B[2*i,col]=1.0; B[2*i+1,col]=0.0; col+=1
        if solve_config.solve_y0:
            B[2*i,col]=0.0; B[2*i+1,col]=1.0; col+=1
        dist_derivs = distortion_derivatives(xi, yi, x0, y0, dist.K1, dist.K2, dist.K3, solve_config)
        for pname in solve_config.distortion_param_names:
            ddx, ddy = dist_derivs[pname]
            B[2*i,col]=-f*ddx/Zbar; B[2*i+1,col]=-f*ddy/Zbar; col+=1

    BtB = B.T @ B
    dof = 2 * n - u
    if dof > 0:
        sigma0 = np.sqrt(v @ v / dof)
        try:
            Qxx = np.linalg.inv(BtB)
            param_std = {}
            param_names = (
                ["Xs", "Ys", "Zs", "omega", "phi", "kappa"]
                + solve_config.intrinsics_param_names
                + solve_config.distortion_param_names
            )
            for i, name in enumerate(param_names):
                param_std[name] = sigma0 * np.sqrt(abs(Qxx[i, i]))
            param_covariance = sigma0**2 * Qxx
        except np.linalg.LinAlgError:
            param_std = {}
            param_covariance = np.zeros((u, u))
    else:
        sigma0 = 0.0
        param_std = {}
        param_covariance = np.zeros((u, u))

    intrinsics_out = CameraIntrinsics(
        f=f, x0=x0, y0=y0,
        sensor_width=intrinsics.sensor_width, sensor_height=intrinsics.sensor_height,
        img_width=intrinsics.img_width, img_height=intrinsics.img_height,
    )

    return ResectionResult(
        exterior=ext,
        intrinsics=intrinsics_out,
        distortion=dist,
        sigma0=sigma0,
        residuals=residuals,
        param_covariance=param_covariance,
        param_std=param_std,
        num_iterations=num_iter,
        converged=converged,
    )

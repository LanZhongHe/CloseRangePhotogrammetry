"""Camera parameter dataclasses, rotation matrix, distortion model, and coordinate conversion."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CameraIntrinsics:
    """Interior orientation parameters."""
    f: float = 50.0             # focal length (mm)
    x0: float = 0.0             # principal point x (mm), default = image center
    y0: float = 0.0             # principal point y (mm), default = image center
    sensor_width: float = 35.9  # sensor width (mm)
    sensor_height: float = 23.9 # sensor height (mm)
    img_width: int = 8256       # image width (pixels)
    img_height: int = 5504      # image height (pixels)

    @property
    def pixel_size(self) -> float:
        """Pixel pitch computed from sensor size and pixel count (mm/px)."""
        return self.sensor_width / self.img_width

    def update_x0_y0_to_center(self):
        """Set x0, y0 to image center in mm."""
        self.x0 = 0.0
        self.y0 = 0.0


@dataclass
class ExteriorOrientation:
    """Exterior orientation elements."""
    Xs: float = 0.0
    Ys: float = 0.0
    Zs: float = 5000.0
    omega: float = 0.0   # rotation around X-axis (rad)
    phi: float = 0.0     # rotation around Y-axis (rad)
    kappa: float = 0.0   # rotation around Z-axis (rad)


@dataclass
class DistortionCoefficients:
    """Lens distortion coefficients (Brown model)."""
    K1: float = 0.0
    K2: float = 0.0
    K3: float = 0.0
    P1: float = 0.0
    P2: float = 0.0
    A1: float = 0.0
    A2: float = 0.0
    B1: float = 0.0
    B2: float = 0.0


@dataclass
class SolveConfig:
    """Controls which parameters participate in the least-squares solve."""
    solve_f: bool = False
    solve_x0: bool = False
    solve_y0: bool = False
    solve_k1: bool = True
    solve_k2: bool = False
    solve_k3: bool = False
    solve_p1: bool = False
    solve_p2: bool = False
    solve_a1: bool = False
    solve_a2: bool = False
    solve_b1: bool = False
    solve_b2: bool = False

    @property
    def num_unknowns(self) -> int:
        """Total number of unknown parameters (6 exterior + selected others)."""
        return 6 + sum([
            self.solve_f, self.solve_x0, self.solve_y0,
            self.solve_k1, self.solve_k2, self.solve_k3,
            self.solve_p1, self.solve_p2,
            self.solve_a1, self.solve_a2,
            self.solve_b1, self.solve_b2,
        ])

    @property
    def min_points(self) -> int:
        """Minimum number of matched points required."""
        return (self.num_unknowns + 1) // 2

    @property
    def distortion_param_names(self) -> list[str]:
        """List of distortion parameter names being solved."""
        names = []
        if self.solve_k1: names.append("K1")
        if self.solve_k2: names.append("K2")
        if self.solve_k3: names.append("K3")
        if self.solve_p1: names.append("P1")
        if self.solve_p2: names.append("P2")
        if self.solve_a1: names.append("A1")
        if self.solve_a2: names.append("A2")
        if self.solve_b1: names.append("B1")
        if self.solve_b2: names.append("B2")
        return names

    @property
    def intrinsics_param_names(self) -> list[str]:
        """List of intrinsic parameter names being solved."""
        names = []
        if self.solve_f: names.append("f")
        if self.solve_x0: names.append("x0")
        if self.solve_y0: names.append("y0")
        return names


# --- Rotation Matrix ---

def rotation_matrix(omega: float, phi: float, kappa: float) -> np.ndarray:
    """Build 3x3 rotation matrix from omega, phi, kappa (radians).

    Convention: R = R_kappa @ R_phi @ R_omega
    (rotate by omega around X, then phi around Y, then kappa around Z)
    """
    cw, sw = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)

    R = np.array([
        [cp * ck,  sw * sp * ck - cw * sk,  cw * sp * ck + sw * sk],
        [cp * sk,  sw * sp * sk + cw * ck,  cw * sp * sk - sw * ck],
        [-sp,      sw * cp,                 cw * cp               ],
    ])
    return R


def rotation_matrix_derivatives(omega: float, phi: float, kappa: float):
    """Compute partial derivatives of R w.r.t. omega, phi, kappa.

    Returns:
        (dR_domega, dR_dphi, dR_dkappa) — each a 3x3 numpy array
    """
    cw, sw = np.cos(omega), np.sin(omega)
    cp, sp = np.cos(phi), np.sin(phi)
    ck, sk = np.cos(kappa), np.sin(kappa)

    # dR/domega: differentiate R w.r.t. omega (only sw, cw terms change)
    dR_do = np.array([
        [0, cw * sp * ck + sw * sk,  -sw * sp * ck + cw * sk],
        [0, cw * sp * sk - sw * ck,  -sw * sp * sk - cw * ck],
        [0, cw * cp,                 -sw * cp               ],
    ])

    # dR/dphi: differentiate R w.r.t. phi (only cp, sp terms change)
    dR_dp = np.array([
        [-sp * ck, sw * cp * ck, cw * cp * ck],
        [-sp * sk, sw * cp * sk, cw * cp * sk],
        [-cp,      -sw * sp,     -cw * sp    ],
    ])

    # dR/dkappa: differentiate R w.r.t. kappa (only ck, sk terms change)
    dR_dk = np.array([
        [-cp * sk, -sw * sp * sk - cw * ck, -cw * sp * sk + sw * ck],
        [cp * ck,  sw * sp * ck - cw * sk,  cw * sp * ck + sw * sk ],
        [0,        0,                       0                      ],
    ])

    return dR_do, dR_dp, dR_dk


# --- Distortion ---

def compute_distortion(
    x: float, y: float,
    x0: float, y0: float,
    K1: float, K2: float, K3: float,
    P1: float, P2: float,
    A1: float, A2: float, B1: float, B2: float,
) -> tuple[float, float]:
    """Compute total distortion corrections (delta_x, delta_y) in mm.

    Uses the Brown distortion model (radial + decentering + thin prism).
    """
    dx = x - x0
    dy = y - y0
    r2 = dx * dx + dy * dy
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial symmetric distortion
    radial = K1 * r2 + K2 * r4 + K3 * r6
    dr_x = dx * radial
    dr_y = dy * radial

    # Decentering distortion
    dd_x = P1 * (r2 + 2 * dx * dx) + 2 * P2 * dx * dy
    dd_y = P2 * (r2 + 2 * dy * dy) + 2 * P1 * dx * dy

    # Thin prism distortion
    dp_x = A1 * r2 + A2 * r4
    dp_y = B1 * r2 + B2 * r4

    return dr_x + dd_x + dp_x, dr_y + dd_y + dp_y


def distortion_derivatives(
    x: float, y: float,
    x0: float, y0: float,
    K1: float, K2: float, K3: float,
    solve_config: SolveConfig,
) -> dict[str, tuple[float, float]]:
    """Compute partial derivatives of (delta_x, delta_y) w.r.t. each distortion parameter.

    Returns:
        dict mapping param_name -> (d(delta_x)/d_param, d(delta_y)/d_param)
    """
    dx = x - x0
    dy = y - y0
    r2 = dx * dx + dy * dy
    r4 = r2 * r2
    r6 = r4 * r2

    result = {}

    # Radial
    if solve_config.solve_k1:
        result["K1"] = (dx * r2, dy * r2)
    if solve_config.solve_k2:
        result["K2"] = (dx * r4, dy * r4)
    if solve_config.solve_k3:
        result["K3"] = (dx * r6, dy * r6)

    # Decentering
    if solve_config.solve_p1:
        result["P1"] = (r2 + 2 * dx * dx, 2 * dx * dy)
    if solve_config.solve_p2:
        result["P2"] = (2 * dx * dy, r2 + 2 * dy * dy)

    # Thin prism
    if solve_config.solve_a1:
        result["A1"] = (r2, 0.0)
    if solve_config.solve_a2:
        result["A2"] = (r4, 0.0)
    if solve_config.solve_b1:
        result["B1"] = (0.0, r2)
    if solve_config.solve_b2:
        result["B2"] = (0.0, r4)

    return result


# --- Coordinate Conversion ---

def pixel_to_image_coords(
    pixel_x: float, pixel_y: float,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Convert pixel coordinates to image-plane coordinates in mm.

    Convention:
      - Origin at principal point (default: image center)
      - x-axis points right
      - y-axis points UP (opposite to pixel y)

    Args:
        pixel_x, pixel_y: measured pixel coordinates
        intrinsics: camera intrinsics (provides pixel_size, img dimensions)

    Returns:
        (x_img, y_img) in mm
    """
    ps = intrinsics.pixel_size
    # Principal point: image center + x0/y0 offset (in mm, converted to pixels)
    # x0/y0 are offsets from image center in mm; positive x0 moves principal point right,
    # positive y0 moves principal point up (opposite to pixel y direction)
    cx = intrinsics.img_width / 2.0 + intrinsics.x0 / ps
    cy = intrinsics.img_height / 2.0 - intrinsics.y0 / ps

    x_img = (pixel_x - cx) * ps
    y_img = -(pixel_y - cy) * ps
    return x_img, y_img

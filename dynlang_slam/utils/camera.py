"""Camera utilities for DynLang-SLAM."""

import torch
import torch.nn.functional as F


def get_projection_matrix(
    fx: float, fy: float, cx: float, cy: float,
    height: int, width: int,
    near: float = 0.01, far: float = 100.0,
) -> torch.Tensor:
    """Create an OpenGL-style projection matrix from intrinsics.

    Returns:
        (4, 4) projection matrix
    """
    proj = torch.zeros(4, 4)
    proj[0, 0] = 2.0 * fx / width
    proj[1, 1] = 2.0 * fy / height
    proj[0, 2] = 1.0 - 2.0 * cx / width
    proj[1, 2] = 2.0 * cy / height - 1.0
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    return proj


def pose_to_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """Convert rotation (quaternion) + translation to 4x4 matrix.

    Args:
        rotation: (4,) quaternion [w, x, y, z]
        translation: (3,) translation vector

    Returns:
        (4, 4) transformation matrix
    """
    q = F.normalize(rotation, dim=-1)
    w, x, y, z = q[0], q[1], q[2], q[3]

    R = torch.stack([
        1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
        2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
        2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y,
    ]).reshape(3, 3)

    # Build T without in-place ops to preserve autograd graph
    bottom = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rotation.device, dtype=rotation.dtype)
    top = torch.cat([R, translation.unsqueeze(1)], dim=1)  # (3, 4)
    T = torch.cat([top, bottom], dim=0)  # (4, 4)
    return T


def matrix_to_pose(T: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert 4x4 matrix to quaternion + translation.

    Args:
        T: (4, 4) transformation matrix

    Returns:
        rotation: (4,) quaternion [w, x, y, z]
        translation: (3,) translation vector
    """
    R = T[:3, :3]
    translation = T[:3, 3]

    # Matrix to quaternion
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / torch.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    rotation = torch.stack([w, x, y, z])
    return rotation, translation


def fast_se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Fast inverse of a rigid body transformation (SE(3)).

    Much faster and more numerically stable than torch.inverse() for 4x4
    rigid transforms. Exploits the structure: T^{-1} = [R^T | -R^T @ t]
    """
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -(R_inv @ t)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


def depth_to_points(
    depth: torch.Tensor,
    fx: float, fy: float, cx: float, cy: float,
    pose: torch.Tensor | None = None,
) -> torch.Tensor:
    """Unproject depth map to 3D points.

    Args:
        depth: (H, W) or (1, H, W) depth map in meters
        fx, fy, cx, cy: camera intrinsics
        pose: optional (4, 4) camera-to-world transform

    Returns:
        (N, 3) world-space 3D points (only valid depth pixels)
    """
    if depth.dim() == 3:
        depth = depth.squeeze(0)

    H, W = depth.shape
    device = depth.device

    # Create pixel grid
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Unproject to camera space
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Stack into (H, W, 3)
    points_cam = torch.stack([x, y, z], dim=-1)

    # Filter invalid depth
    valid = depth > 0
    points_cam = points_cam[valid]  # (N, 3)

    # Transform to world space
    if pose is not None:
        R = pose[:3, :3]  # (3, 3)
        t = pose[:3, 3]  # (3,)
        points_world = points_cam @ R.T + t.unsqueeze(0)
        return points_world

    return points_cam

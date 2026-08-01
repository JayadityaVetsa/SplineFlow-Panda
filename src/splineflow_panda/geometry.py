from __future__ import annotations

import numpy as np


def intrinsic_from_fovy(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_degrees) / 2)
    fx = fy
    return np.array([[fx, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1]])


def project_points(
    world_points: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project to OpenCV-style pixels: +x right, +y down, +z forward."""
    points = np.asarray(world_points, dtype=float)
    homogeneous = np.c_[points, np.ones(len(points))]
    camera = (np.asarray(world_to_camera) @ homogeneous.T).T[:, :3]
    depth = camera[:, 2]
    safe_depth = np.where(depth > 0, depth, 1.0)
    pixels_h = (np.asarray(intrinsic) @ camera.T).T
    pixels = pixels_h[:, :2] / safe_depth[:, None]
    width, height = image_size
    valid = (
        (depth > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    pixels[depth <= 0] = np.nan
    return pixels, depth, valid


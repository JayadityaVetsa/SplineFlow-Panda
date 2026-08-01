from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def write_rgb_video(path: Path, frames: np.ndarray, fps: float) -> Path:
    """Encode RGB uint8 frames as a browser-compatible H.264 MP4."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
        path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        pixelformat="yuv420p",
        macro_block_size=16,
        ffmpeg_log_level="error",
    ) as writer:
        for frame in np.asarray(frames):
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    return path


def depth_to_grayscale(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Map metric depth to display intensity while excluding the far-plane background."""
    values = np.asarray(depth, dtype=np.float32)
    finite = values[np.isfinite(values) & (values > 0)]
    if not len(finite):
        return np.zeros(values.shape, dtype=np.uint8), 0.0, 0.0
    raw_max = float(finite.max())
    foreground = finite[finite < raw_max * 0.95]
    if len(foreground) < 16:
        foreground = finite
    near, far = np.percentile(foreground, [1, 99])
    if far <= near:
        far = near + 1e-6
    normalized = (far - np.clip(values, near, far)) / (far - near)
    normalized[(~np.isfinite(values)) | (values >= raw_max * 0.95)] = 0.0
    return np.round(255 * normalized).astype(np.uint8), float(near), float(far)

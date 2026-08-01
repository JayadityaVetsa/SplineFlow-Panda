from __future__ import annotations

import numpy as np


def trajectory_metrics(
    time: np.ndarray,
    desired: np.ndarray,
    actual: np.ndarray,
    waypoint_times: np.ndarray | None = None,
    stop_threshold: float = 0.01,
) -> dict[str, float]:
    error = np.linalg.norm(actual - desired, axis=1)
    velocity = np.gradient(actual, time, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, time, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, time, axis=0, edge_order=2)
    speed = np.linalg.norm(velocity, axis=1)
    dt = np.diff(time, prepend=time[0])
    result = {
        "tracking_rmse_m": float(np.sqrt(np.mean(error**2))),
        "tracking_max_m": float(error.max()),
        "path_length_m": float(np.linalg.norm(np.diff(actual, axis=0), axis=1).sum()),
        "execution_time_s": float(time[-1] - time[0]),
        "speed_max_m_s": float(speed.max()),
        "acceleration_rms_m_s2": float(np.sqrt(np.mean(np.sum(acceleration[2:-2] ** 2, axis=1)))),
        "jerk_rms_m_s3": float(np.sqrt(np.mean(np.sum(jerk[3:-3] ** 2, axis=1)))),
        "stopped_duration_s": float(dt[speed < stop_threshold].sum()),
    }
    if waypoint_times is not None:
        indices = np.searchsorted(time, waypoint_times).clip(0, len(time) - 1)
        result["waypoint_error_max_m"] = float(error[indices].max())
    return result

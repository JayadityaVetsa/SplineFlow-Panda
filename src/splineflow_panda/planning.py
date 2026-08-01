from __future__ import annotations

import numpy as np
from scipy.interpolate import make_interp_spline

from .models import ExperimentConfig, PlannerKind, Trajectory


def _sample_times(duration: float, rate: float) -> np.ndarray:
    return np.linspace(0.0, duration, max(2, int(round(duration * rate)) + 1))


def plan_bspline(config: ExperimentConfig) -> Trajectory:
    points = config.sampled_waypoints()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    waypoint_times = np.r_[0.0, np.cumsum(segment_lengths)]
    duration = config.planner.duration / config.planner.speedup
    waypoint_times *= duration / waypoint_times[-1]
    degree = min(3, len(points) - 1)
    spline = make_interp_spline(waypoint_times, points, k=degree)
    time = _sample_times(duration, config.planner.sample_rate)
    return Trajectory(
        time, spline(time), spline.derivative(1)(time), spline.derivative(2)(time),
        waypoint_times, "bspline", {"degree": degree, "parameterization": "chord-length"},
    )


def _minimum_jerk(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    position = 10 * s**3 - 15 * s**4 + 6 * s**5
    velocity = 30 * s**2 - 60 * s**3 + 30 * s**4
    acceleration = 60 * s - 180 * s**2 + 120 * s**3
    return position, velocity, acceleration


def plan_sequential(config: ExperimentConfig) -> Trajectory:
    points = config.sampled_waypoints()
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    move_duration = config.planner.duration / config.planner.speedup
    segment_durations = move_duration * lengths / lengths.sum()
    dwell = config.planner.dwell
    time_parts, pos_parts, vel_parts, acc_parts = [], [], [], []
    waypoint_times = [0.0]
    cursor = 0.0
    rate = config.planner.sample_rate
    for i, duration in enumerate(segment_durations):
        count = max(2, int(round(duration * rate)) + 1)
        local = np.linspace(0.0, duration, count)
        s = local / duration
        q, qd, qdd = _minimum_jerk(s)
        delta = points[i + 1] - points[i]
        sl = slice(None) if i == 0 else slice(1, None)
        time_parts.append((cursor + local)[sl])
        pos_parts.append((points[i] + q[:, None] * delta)[sl])
        vel_parts.append((qd[:, None] * delta / duration)[sl])
        acc_parts.append((qdd[:, None] * delta / duration**2)[sl])
        cursor += duration
        waypoint_times.append(cursor)
        if i < len(segment_durations) - 1 and dwell > 0:
            dwell_local = np.arange(1 / rate, dwell + 0.5 / rate, 1 / rate)
            time_parts.append(cursor + dwell_local)
            pos_parts.append(np.repeat(points[i + 1][None], len(dwell_local), axis=0))
            vel_parts.append(np.zeros((len(dwell_local), 3)))
            acc_parts.append(np.zeros((len(dwell_local), 3)))
            cursor += len(dwell_local) / rate
    return Trajectory(
        np.concatenate(time_parts), np.concatenate(pos_parts), np.concatenate(vel_parts),
        np.concatenate(acc_parts), np.asarray(waypoint_times), "sequential",
        {"dwell_seconds": dwell, "movement_duration": move_duration},
    )


def plan(config: ExperimentConfig) -> Trajectory:
    if config.planner.kind in {
        PlannerKind.BSPLINE,
        PlannerKind.BSPLINE_ACTION,
        PlannerKind.ACTION_CHUNK,
    }:
        return plan_bspline(config)
    return plan_sequential(config)

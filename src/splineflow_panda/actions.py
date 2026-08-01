from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import BSpline, make_lsq_spline


@dataclass(frozen=True)
class ActionSequence:
    """Commands at explicit controller timestamps."""

    time: np.ndarray
    command: np.ndarray
    representation: str
    policy_time: np.ndarray
    segment_index: np.ndarray
    diagnostics: dict[str, float]

    def __post_init__(self) -> None:
        if self.time.ndim != 1 or np.any(np.diff(self.time) <= 0):
            raise ValueError("Action time must be strictly increasing")
        if self.command.shape[0] != len(self.time):
            raise ValueError("Action command and time lengths differ")
        if not np.isfinite(self.command).all():
            raise ValueError("Action commands must be finite")


@dataclass(frozen=True)
class SplineParameters:
    degree: int
    knots: np.ndarray
    control_points: np.ndarray
    reconstruction_error: float
    valid_mask: np.ndarray

    def evaluate(self, time: np.ndarray, derivative: int = 0) -> np.ndarray:
        spline = BSpline(self.knots, self.control_points, self.degree)
        return spline.derivative(derivative)(time)


def retime(time: np.ndarray, speedup: float) -> np.ndarray:
    """Traverse the same normalized trajectory speedup-times faster."""
    if speedup <= 0:
        raise ValueError("Speedup must be positive")
    return (np.asarray(time, dtype=float) - time[0]) / speedup


def _open_knot_vector(start: float, end: float, interior: list[float], degree: int) -> np.ndarray:
    return np.r_[
        np.repeat(start, degree + 1),
        np.asarray(sorted(interior), dtype=float),
        np.repeat(end, degree + 1),
    ]


def fit_adaptive_bspline(
    time: np.ndarray,
    actions: np.ndarray,
    *,
    tolerance: float = 0.002,
    max_control_points: int = 16,
    degree: int = 3,
) -> SplineParameters:
    """FITPACK-style greedy knot insertion with a bounded sample reconstruction error."""
    time = np.asarray(time, dtype=float)
    actions = np.asarray(actions, dtype=float)
    if len(time) < degree + 1 or actions.shape[0] != len(time):
        raise ValueError("Insufficient or mismatched demonstration samples")
    if np.any(np.diff(time) <= 0):
        raise ValueError("Demonstration time must be strictly increasing")
    interior: list[float] = []
    while True:
        knots = _open_knot_vector(time[0], time[-1], interior, degree)
        spline = make_lsq_spline(time, actions, knots, k=degree)
        residual = np.linalg.norm(spline(time) - actions, axis=1)
        error = float(residual.max())
        control_count = len(knots) - degree - 1
        if error <= tolerance or control_count >= max_control_points:
            mask = np.zeros(max_control_points, dtype=bool)
            mask[:control_count] = True
            return SplineParameters(
                degree=degree,
                knots=knots,
                control_points=np.asarray(spline.c),
                reconstruction_error=error,
                valid_mask=mask,
            )
        candidates = np.argsort(residual)[::-1]
        minimum_spacing = (time[-1] - time[0]) / (4 * max_control_points)
        inserted = False
        for index in candidates:
            candidate = float(time[index])
            existing = [float(time[0]), *interior, float(time[-1])]
            if all(abs(candidate - value) >= minimum_spacing for value in existing):
                interior.append(candidate)
                inserted = True
                break
        if not inserted:
            raise RuntimeError("Adaptive knot insertion could not reduce fitting error")


def sample_bspline_actions(
    parameters: SplineParameters,
    *,
    control_rate: float,
    speedup: float = 1.0,
) -> ActionSequence:
    source_start = float(parameters.knots[parameters.degree])
    source_end = float(parameters.knots[-parameters.degree - 1])
    duration = (source_end - source_start) / speedup
    time = np.linspace(0.0, duration, max(2, round(duration * control_rate) + 1))
    source_time = source_start + time * speedup
    commands = parameters.evaluate(source_time)
    return ActionSequence(
        time=time,
        command=commands,
        representation="bspline_action",
        policy_time=np.array([0.0]),
        segment_index=np.zeros(len(time), dtype=int),
        diagnostics={
            "speedup": speedup,
            "fitting_error": parameters.reconstruction_error,
            "control_points": float(len(parameters.control_points)),
        },
    )


def sample_action_chunks(
    time: np.ndarray,
    actions: np.ndarray,
    *,
    policy_rate: float,
    control_rate: float,
    speedup: float = 1.0,
) -> ActionSequence:
    """Zero-order-hold discrete actions; intentionally does not smooth chunk boundaries."""
    source_time = np.asarray(time, dtype=float)
    actions = np.asarray(actions, dtype=float)
    duration = (source_time[-1] - source_time[0]) / speedup
    controller_time = np.linspace(0.0, duration, max(2, round(duration * control_rate) + 1))
    policy_period = 1.0 / (policy_rate * speedup)
    policy_time = np.arange(0.0, duration + 0.5 * policy_period, policy_period)
    source_policy_time = np.minimum(source_time[-1], source_time[0] + policy_time * speedup)
    policy_commands = np.stack(
        [actions[np.argmin(np.abs(source_time - value))] for value in source_policy_time]
    )
    segment = np.searchsorted(policy_time, controller_time, side="right") - 1
    segment = np.clip(segment, 0, len(policy_commands) - 1)
    commands = policy_commands[segment]
    discontinuity = (
        float(np.linalg.norm(np.diff(policy_commands, axis=0), axis=1).max())
        if len(policy_commands) > 1
        else 0.0
    )
    return ActionSequence(
        time=controller_time,
        command=commands,
        representation="action_chunk",
        policy_time=policy_time,
        segment_index=segment,
        diagnostics={"speedup": speedup, "boundary_discontinuity": discontinuity},
    )


def align_segment(segment: np.ndarray, last_action: np.ndarray, search_samples: int) -> int:
    """Return the locally closest start sample, as used for inference-time phase alignment."""
    if search_samples < 1:
        raise ValueError("search_samples must be positive")
    window = np.asarray(segment)[:search_samples]
    if not len(window):
        raise ValueError("Cannot align an empty segment")
    return int(np.argmin(np.sum((window - np.asarray(last_action)) ** 2, axis=1)))

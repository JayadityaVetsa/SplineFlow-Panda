from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PathSuccessCriteria:
    final_error_m: float = 0.02
    waypoint_error_m: float = 0.02
    orientation_error_rad: float = np.deg2rad(5.0)
    tracking_max_m: float = 0.05


DEFAULT_PATH_SUCCESS_CRITERIA = PathSuccessCriteria()


def classify_contact(
    first: str,
    second: str,
    *,
    robot_geoms: set[str],
    obstacle_geoms: set[str],
    table_geoms: set[str],
    puck_geoms: set[str],
) -> str:
    pair = {first, second}
    if pair & robot_geoms and pair & obstacle_geoms:
        return "forbidden"
    if pair & robot_geoms and pair & table_geoms:
        return "forbidden"
    if pair & robot_geoms and pair & puck_geoms:
        return "allowed"
    if pair & puck_geoms and pair & table_geoms:
        return "allowed"
    if pair & puck_geoms and pair & obstacle_geoms:
        return "forbidden"
    return "ignored"


def point_box_signed_clearance(
    point: np.ndarray, center: np.ndarray, half_size: np.ndarray
) -> float:
    """Signed Euclidean distance to an axis-aligned box; negative means inside."""
    delta = np.abs(np.asarray(point) - np.asarray(center)) - np.asarray(half_size)
    outside = np.maximum(delta, 0)
    outside_distance = np.linalg.norm(outside)
    if np.any(delta > 0):
        return float(outside_distance)
    return float(np.max(delta))


def path_success(
    *,
    final_error_m: float,
    waypoint_error_m: float,
    orientation_error_rad: float,
    tracking_max_m: float,
    forbidden_contacts: int,
    criteria: PathSuccessCriteria = DEFAULT_PATH_SUCCESS_CRITERIA,
) -> bool:
    return bool(
        final_error_m <= criteria.final_error_m
        and waypoint_error_m <= criteria.waypoint_error_m
        and orientation_error_rad <= criteria.orientation_error_rad
        and tracking_max_m <= criteria.tracking_max_m
        and forbidden_contacts == 0
    )


def sustained_completion_time(
    time: np.ndarray,
    distance: np.ndarray,
    *,
    tolerance: float,
    hold_time: float,
) -> float | None:
    inside = np.asarray(distance) <= tolerance
    required = max(1, int(np.ceil(hold_time / np.median(np.diff(time)))))
    if required == 1:
        indices = np.flatnonzero(inside)
    else:
        indices = np.flatnonzero(
            np.convolve(inside.astype(int), np.ones(required, dtype=int), mode="valid")
            == required
        )
    if not len(indices):
        return None
    return float(time[indices[0] + required - 1])


def bootstrap_mean_interval(
    values: Iterable[float], *, seed: int = 0, samples: int = 2000
) -> tuple[float, float, float]:
    data = np.asarray(list(values), dtype=float)
    if not len(data):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(data.mean()), float(low), float(high)


def maximum_reliable_speedup(
    rows: list[dict[str, float | bool]], *, minimum_success_rate: float = 0.8
) -> float:
    by_speed: dict[float, list[bool]] = {}
    for row in rows:
        by_speed.setdefault(float(row["speedup"]), []).append(bool(row["success"]))
    reliable = [
        speed for speed, successes in by_speed.items() if np.mean(successes) >= minimum_success_rate
    ]
    return max(reliable, default=float("nan"))

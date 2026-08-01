import numpy as np
import pytest

from splineflow_panda.models import ExperimentConfig, PlannerConfig, PlannerKind
from splineflow_panda.planning import plan_bspline, plan_sequential


@pytest.fixture
def config() -> ExperimentConfig:
    return ExperimentConfig(
        name="synthetic",
        waypoints=[(0, 0, 0), (1, 0.5, 0.2), (2, -0.2, 0.4), (3, 0, 0.5)],
        planner=PlannerConfig(duration=3, sample_rate=100, dwell=0.1),
    )


def test_bspline_interpolates_waypoints(config: ExperimentConfig) -> None:
    trajectory = plan_bspline(config)
    for target, target_time in zip(config.waypoints, trajectory.waypoint_times, strict=True):
        index = np.argmin(np.abs(trajectory.time - target_time))
        assert np.linalg.norm(trajectory.position[index] - target) < 0.03
    assert np.max(np.abs(np.diff(trajectory.velocity, axis=0))) < 0.2


def test_sequential_stops_and_preserves_order(config: ExperimentConfig) -> None:
    config.planner.kind = PlannerKind.SEQUENTIAL
    trajectory = plan_sequential(config)
    for target_time in trajectory.waypoint_times:
        index = np.argmin(np.abs(trajectory.time - target_time))
        assert np.linalg.norm(trajectory.velocity[index]) < 1e-8
    assert np.all(np.diff(trajectory.time) > 0)


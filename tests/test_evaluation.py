import numpy as np

from splineflow_panda.evaluation import (
    bootstrap_mean_interval,
    classify_contact,
    maximum_reliable_speedup,
    path_success,
    point_box_signed_clearance,
    sustained_completion_time,
)


def test_contact_roles() -> None:
    roles = {
        "robot_geoms": {"hand"},
        "obstacle_geoms": {"wall"},
        "table_geoms": {"table"},
        "puck_geoms": {"puck"},
    }
    assert classify_contact("hand", "puck", **roles) == "allowed"
    assert classify_contact("hand", "wall", **roles) == "forbidden"
    assert classify_contact("puck", "table", **roles) == "allowed"


def test_signed_box_clearance() -> None:
    center = np.zeros(3)
    half_size = np.ones(3)
    assert point_box_signed_clearance(np.array([2.0, 0, 0]), center, half_size) == 1
    assert point_box_signed_clearance(np.zeros(3), center, half_size) == -1


def test_success_frontier_and_interval_are_deterministic() -> None:
    assert path_success(
        final_error_m=0.01,
        waypoint_error_m=0.01,
        orientation_error_rad=0.01,
        tracking_max_m=0.03,
        forbidden_contacts=0,
    )
    rows = [
        {"speedup": speed, "success": success}
        for speed, successes in [(1, [1, 1]), (2, [1, 1]), (3, [1, 0])]
        for success in successes
    ]
    assert maximum_reliable_speedup(rows, minimum_success_rate=0.8) == 2
    assert bootstrap_mean_interval([1, 2, 3], seed=4) == bootstrap_mean_interval(
        [1, 2, 3], seed=4
    )


def test_sustained_completion_time_requires_hold() -> None:
    time = np.arange(6) * 0.1
    distance = np.array([1.0, 0.1, 0.01, 0.01, 0.01, 0.2])
    assert np.isclose(
        sustained_completion_time(time, distance, tolerance=0.02, hold_time=0.2),
        0.3,
    )

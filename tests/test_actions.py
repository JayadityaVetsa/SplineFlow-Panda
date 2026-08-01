import numpy as np

from splineflow_panda.actions import (
    align_segment,
    fit_adaptive_bspline,
    sample_action_chunks,
    sample_bspline_actions,
)


def demonstration() -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0, 2, 201)
    actions = np.c_[np.sin(time), np.cos(time), 0.2 * time]
    return time, actions


def test_adaptive_spline_respects_error_and_is_temporally_scalable() -> None:
    time, actions = demonstration()
    fitted = fit_adaptive_bspline(time, actions, tolerance=0.003, max_control_points=16)
    assert fitted.reconstruction_error <= 0.003
    assert np.all(np.diff(fitted.knots) >= 0)
    normal = sample_bspline_actions(fitted, control_rate=100, speedup=1)
    fast = sample_bspline_actions(fitted, control_rate=100, speedup=2)
    assert np.isclose(fast.time[-1], normal.time[-1] / 2)
    assert np.allclose(fast.command[[0, -1]], normal.command[[0, -1]])


def test_chunks_are_zero_order_hold_and_expose_discontinuity() -> None:
    time, actions = demonstration()
    sampled = sample_action_chunks(
        time, actions, policy_rate=10, control_rate=100, speedup=2
    )
    assert sampled.diagnostics["boundary_discontinuity"] > 0
    assert len(np.unique(sampled.segment_index)) < len(sampled.time)
    assert np.isclose(sampled.time[-1], 1.0)


def test_segment_alignment_selects_nearest_local_action() -> None:
    segment = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    assert align_segment(segment, np.array([0.45, 0.0]), 3) == 1

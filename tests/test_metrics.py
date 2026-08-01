import numpy as np

from splineflow_panda.metrics import trajectory_metrics


def test_metrics_on_exact_linear_track() -> None:
    time = np.linspace(0, 1, 21)
    desired = np.c_[time, np.zeros((len(time), 2))]
    metrics = trajectory_metrics(time, desired, desired)
    assert metrics["tracking_rmse_m"] == 0
    assert np.isclose(metrics["path_length_m"], 1)
    assert np.isclose(metrics["speed_max_m_s"], 1)

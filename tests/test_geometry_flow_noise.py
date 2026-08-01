import numpy as np

from splineflow_panda.geometry import project_points


def test_projection_known_points_and_visibility() -> None:
    intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
    points = np.array([[0, 0, 1], [0.1, 0.2, 1], [0, 0, -1]])
    pixels, depth, valid = project_points(points, intrinsic, np.eye(4), (100, 80))
    np.testing.assert_allclose(pixels[:2], [[50, 40], [60, 60]])
    np.testing.assert_allclose(depth, [1, 1, -1])
    assert valid.tolist() == [True, True, False]

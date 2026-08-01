import numpy as np

from splineflow_panda.visualization import depth_to_grayscale


def test_depth_display_excludes_far_plane_and_preserves_order() -> None:
    depth = np.array([[1.0, 2.0, 3.0, 100.0]], dtype=np.float32)
    display, near, far = depth_to_grayscale(depth)
    assert near < far < 100.0
    assert display[0, 0] > display[0, 1] > display[0, 2]
    assert display[0, 3] == 0


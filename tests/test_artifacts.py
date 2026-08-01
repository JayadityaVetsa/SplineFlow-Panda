from pathlib import Path

from splineflow_panda.artifacts import ExperimentBundle
from splineflow_panda.models import ExperimentConfig
from splineflow_panda.pipeline import plan_experiment


def test_unique_complete_planning_bundle(tmp_path: Path) -> None:
    config = ExperimentConfig(
        name="bundle",
        waypoints=[(0, 0, 1), (0.1, 0, 1), (0.2, 0.1, 1), (0.3, 0, 1)],
    )
    first = plan_experiment(config, tmp_path)
    second = plan_experiment(config, tmp_path)
    assert first.root != second.root
    assert ExperimentBundle(first.root).validate() == []


from pathlib import Path

import pytest
from pydantic import ValidationError

from splineflow_panda.config import load_config
from splineflow_panda.models import ExperimentConfig


def test_load_scenario() -> None:
    config = load_config(Path("configs/scenarios/open_space.yaml"), Path("configs/defaults.yaml"))
    assert len(config.waypoints) == 4
    assert config.camera.width == 320
    assert config.schema_version == "0.3.0"


def test_repeated_waypoints_rejected() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        ExperimentConfig(
            name="bad",
            waypoints=[(0, 0, 1), (0, 0, 1), (0, 1, 1)],
        )


def test_legacy_seed_migrates_to_scientific_seed_fields() -> None:
    config = ExperimentConfig(
        name="legacy",
        seed=12,
        waypoints=[(0, 0, 0), (1, 0, 0), (2, 0, 0)],
    )
    assert config.task_seed == 12

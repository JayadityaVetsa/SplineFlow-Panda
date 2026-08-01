from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import yaml

from .models import SCHEMA_VERSION, ExperimentConfig, RunStage, RunStatus, Trajectory
from .provenance import environment_provenance


class ExperimentBundle:
    def __init__(self, root: Path):
        self.root = Path(root)

    @classmethod
    def create(cls, output: Path, config: ExperimentConfig) -> ExperimentBundle:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = output / f"{stamp}-{config.name}-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("states", "metrics", "frames", "media"):
            (root / folder).mkdir()
        bundle = cls(root)
        bundle.write_yaml("config.yaml", config.model_dump(mode="json"))
        bundle.write_json(
            "metadata.json",
            {
                "schema_version": SCHEMA_VERSION,
                "experiment_id": root.name,
                "name": config.name,
                "created_at": datetime.now(UTC).isoformat(),
                "task_seed": config.task_seed,
                "training_seed": config.training_seed,
                "units": {"length": "metre", "time": "second"},
                "coordinates": {
                    "world": "MuJoCo world",
                    "image": "origin top-left; u right; v down",
                },
                "evaluation_mode": "measured_scripted_mujoco",
                "environment": environment_provenance(),
                "sampled_waypoints": config.sampled_waypoints().tolist(),
            },
        )
        bundle.set_status(RunStage.PENDING, "Experiment bundle created")
        return bundle

    def write_json(self, relative: str, value: object) -> None:
        (self.root / relative).write_text(
            json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
        )

    def write_yaml(self, relative: str, value: object) -> None:
        (self.root / relative).write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    def set_status(self, stage: RunStage, message: str) -> None:
        status = RunStatus(stage=stage, message=message).model_dump(mode="json")
        self.write_json("status.json", status)

    def save_trajectory(self, trajectory: Trajectory, name: str = "desired_trajectory.npz") -> None:
        trajectory.save(self.root / "states" / name)

    def validate(self) -> list[str]:
        required = ["config.yaml", "metadata.json", "status.json", "states/desired_trajectory.npz"]
        return [name for name in required if not (self.root / name).exists()]

    def load_config(self) -> ExperimentConfig:
        return ExperimentConfig.model_validate(
            yaml.safe_load((self.root / "config.yaml").read_text(encoding="utf-8"))
        )


def load_state(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

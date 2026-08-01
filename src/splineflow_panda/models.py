from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "0.3.0"


class PlannerKind(StrEnum):
    SEQUENTIAL = "sequential"
    BSPLINE = "bspline"
    ACTION_CHUNK = "action_chunk"
    BSPLINE_ACTION = "bspline_action"


class RunStage(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    SIMULATING = "simulating"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PlannerConfig(BaseModel):
    kind: PlannerKind = PlannerKind.BSPLINE
    duration: float = Field(4.0, gt=0)
    sample_rate: float = Field(60.0, gt=1)
    dwell: float = Field(0.2, ge=0)
    policy_rate: float = Field(10.0, gt=0)
    control_rate: float = Field(100.0, gt=1)
    chunk_horizon: int = Field(16, ge=4)
    speedup: float = Field(1.0, gt=0)
    spline_tolerance: float = Field(0.002, gt=0)

    @model_validator(mode="after")
    def validate_control_rates(self):
        if self.policy_rate > self.control_rate:
            raise ValueError("Policy rate cannot exceed control rate")
        return self


class CameraConfig(BaseModel):
    width: int = Field(320, ge=32)
    height: int = Field(240, ge=32)
    fps: float = Field(30.0, gt=0)
    fovy_degrees: float = Field(45.0, gt=1, lt=179)


class ControllerConfig(BaseModel):
    control_rate: float = Field(100.0, gt=1)
    gain_scale: float = Field(2.0, gt=0)
    damping_scale: float = Field(1.4, gt=0)
    torque_limit_scale: float = Field(1.0, gt=0, le=1)
    settling_time: float = Field(1.0, ge=0)
    terminal_hold: float = Field(0.5, ge=0)
    saturation_limit: float = Field(0.05, ge=0, le=1)
    command_step_limit: float = Field(0.03, gt=0)


class SimulationConfig(BaseModel):
    timestep: float = Field(0.002, gt=0)
    scene: str = "builtin_panda"
    camera: str = "overview"
    damping: float = Field(1e-3, gt=0)
    ik_position_tolerance: float = Field(1e-4, gt=0)
    ik_orientation_tolerance: float = Field(np.deg2rad(1.0), gt=0)
    ik_iterations: int = Field(100, ge=1)
    ik_step_limit: float = Field(0.15, gt=0)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_v01(cls, value: Any):
        if isinstance(value, dict):
            value = dict(value)
            value.pop("kp", None)
            if "ik_tolerance" in value and "ik_position_tolerance" not in value:
                value["ik_position_tolerance"] = value.pop("ik_tolerance")
        return value


class BoxObjectConfig(BaseModel):
    name: str
    position: tuple[float, float, float]
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.75, 0.2, 0.15, 1.0)
    collision_role: Literal["obstacle", "table", "goal", "decorative"] = "obstacle"


class PuckConfig(BaseModel):
    name: str = "puck"
    position: tuple[float, float, float] = (0.5, 0.0, 0.24)
    radius: float = Field(0.035, gt=0)
    height: float = Field(0.025, gt=0)


class GoalConfig(BaseModel):
    center: tuple[float, float] = (0.6, 0.15)
    radius: float = Field(0.06, gt=0)
    hold_time: float = Field(0.25, ge=0)


class TaskConfig(BaseModel):
    kind: Literal["path", "pushing"] = "path"
    boxes: list[BoxObjectConfig] = Field(default_factory=list)
    puck: PuckConfig | None = None
    goal: GoalConfig | None = None
    waypoint_jitter: float = Field(0.0, ge=0)

    @model_validator(mode="after")
    def validate_pushing(self):
        if self.kind == "pushing" and (self.puck is None or self.goal is None):
            raise ValueError("Pushing tasks require both puck and goal configuration")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = SCHEMA_VERSION
    name: str
    task_seed: int = 0
    training_seed: int = 0
    waypoints: list[tuple[float, float, float]]
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    task: TaskConfig = Field(default_factory=TaskConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_seed_and_schema(cls, value: Any):
        if isinstance(value, dict):
            value = dict(value)
            legacy_seed = value.pop("seed", None)
            if legacy_seed is not None:
                value.setdefault("task_seed", legacy_seed)
            value.pop("noise_seed", None)
            value.pop("flow", None)
            if value.get("schema_version") in {"0.1.0", "0.2.0"}:
                value["schema_version"] = SCHEMA_VERSION
        return value

    @field_validator("waypoints")
    @classmethod
    def validate_waypoints(cls, value: list[tuple[float, float, float]]):
        if len(value) < 3:
            raise ValueError("At least three ordered 3D waypoints are required")
        points = np.asarray(value, dtype=float)
        if not np.isfinite(points).all():
            raise ValueError("Waypoints must contain finite coordinates")
        if np.any(np.linalg.norm(np.diff(points, axis=0), axis=1) < 1e-9):
            raise ValueError("Consecutive waypoints must be distinct")
        return value

    @model_validator(mode="after")
    def validate_rates(self):
        if self.camera.fps > self.planner.sample_rate:
            raise ValueError("Camera FPS cannot exceed the trajectory sample rate")
        return self

    def sampled_waypoints(self) -> np.ndarray:
        points = np.asarray(self.waypoints, dtype=float)
        if self.task.waypoint_jitter == 0:
            return points
        rng = np.random.default_rng(self.task_seed)
        jitter = rng.uniform(
            -self.task.waypoint_jitter,
            self.task.waypoint_jitter,
            size=points.shape,
        )
        jitter[[0, -1]] = 0
        return points + jitter


class Diagnostic(BaseModel):
    stage: RunStage
    code: str
    message: str
    detail: str | None = None


class RunStatus(BaseModel):
    schema_version: str = SCHEMA_VERSION
    stage: RunStage = RunStage.PENDING
    message: str = ""
    diagnostic: Diagnostic | None = None


class Trajectory:
    """Planner-neutral, array-oriented trajectory contract."""

    def __init__(
        self,
        time: np.ndarray,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: np.ndarray,
        waypoint_times: np.ndarray,
        planner: str,
        metadata: dict[str, Any] | None = None,
    ):
        self.time = np.asarray(time, dtype=float)
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.acceleration = np.asarray(acceleration, dtype=float)
        self.waypoint_times = np.asarray(waypoint_times, dtype=float)
        self.planner = planner
        self.metadata = metadata or {}
        self._validate()

    def _validate(self) -> None:
        n = len(self.time)
        if self.time.shape != (n,) or np.any(np.diff(self.time) <= 0):
            raise ValueError("Trajectory time must be a strictly increasing vector")
        for name in ("position", "velocity", "acceleration"):
            if getattr(self, name).shape != (n, 3):
                raise ValueError(f"{name} must have shape (T, 3)")
        if not all(np.isfinite(x).all() for x in (self.time, self.position, self.velocity)):
            raise ValueError("Trajectory arrays must be finite")

    def save(self, path: Path) -> None:
        np.savez_compressed(
            path,
            time=self.time,
            position=self.position,
            velocity=self.velocity,
            acceleration=self.acceleration,
            waypoint_times=self.waypoint_times,
            planner=np.asarray(self.planner),
        )

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        with np.load(path, allow_pickle=False) as data:
            return cls(
                data["time"], data["position"], data["velocity"], data["acceleration"],
                data["waypoint_times"], str(data["planner"]),
            )


class CameraCalibration(BaseModel):
    width: int
    height: int
    intrinsic: list[list[float]]
    world_to_camera: list[list[float]]
    convention: Literal["opencv"] = "opencv"

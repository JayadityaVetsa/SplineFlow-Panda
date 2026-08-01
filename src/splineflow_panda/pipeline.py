from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .artifacts import ExperimentBundle
from .config import load_config
from .evaluation import sustained_completion_time
from .metrics import trajectory_metrics
from .models import ExperimentConfig, PlannerKind, RunStage, Trajectory
from .planning import plan


def plan_experiment(config: ExperimentConfig, output: Path) -> ExperimentBundle:
    bundle = ExperimentBundle.create(output, config)
    try:
        bundle.set_status(RunStage.PLANNING, "Planning desired trajectory")
        started = time.perf_counter()
        trajectory = plan(config)
        planning_time = time.perf_counter() - started
        bundle.save_trajectory(trajectory)
        metrics = trajectory_metrics(
            trajectory.time, trajectory.position, trajectory.position, trajectory.waypoint_times
        )
        metrics["planning_time_s"] = planning_time
        bundle.write_json("metrics/metrics.json", metrics)
        bundle.set_status(RunStage.COMPLETED, "Planning completed")
        return bundle
    except Exception:
        bundle.set_status(RunStage.FAILED, "Planning failed; inspect the exception")
        raise


def evaluate_bundle(bundle: ExperimentBundle) -> dict[str, float]:
    desired = Trajectory.load(bundle.root / "states" / "desired_trajectory.npz")
    commanded_path = bundle.root / "states" / "commanded_trajectory.npz"
    if commanded_path.exists():
        desired = Trajectory.load(commanded_path)
    executed_path = bundle.root / "states" / "executed_trajectory.npz"
    actual = Trajectory.load(executed_path) if executed_path.exists() else desired
    metrics = trajectory_metrics(
        actual.time, desired.position, actual.position, desired.waypoint_times
    )
    robot_state_path = bundle.root / "states" / "robot_state.npz"
    if robot_state_path.exists():
        with np.load(robot_state_path, allow_pickle=False) as state:
            if "forbidden_contact_count" in state:
                metrics["forbidden_contacts"] = float(
                    np.sum(state["forbidden_contact_count"])
                )
            if "robot_obstacle_mesh_clearance" in state:
                finite = state["robot_obstacle_mesh_clearance"][
                    np.isfinite(state["robot_obstacle_mesh_clearance"])
                ]
                if len(finite):
                    metrics["minimum_clearance_m"] = float(finite.min())
                    metrics["clearance_method"] = "mujoco_collision_geom_distance"
            if "actuator_saturated" in state:
                metrics["actuator_saturation_rate"] = float(
                    np.mean(state["actuator_saturated"])
                )
            if "ik_orientation_residual" in state:
                metrics["ik_orientation_error_max_deg"] = float(
                    np.rad2deg(state["ik_orientation_residual"]).max()
                )
            config = bundle.load_config()
            if config.task.kind == "pushing" and "puck_position" in state:
                goal = np.asarray(config.task.goal.center)
                distance = np.linalg.norm(state["puck_position"][:, :2] - goal, axis=1)
                completion = sustained_completion_time(
                    state["time"],
                    distance,
                    tolerance=config.task.goal.radius,
                    hold_time=config.task.goal.hold_time,
                )
                metrics["puck_goal_error_m"] = float(distance[-1])
                success = completion is not None and distance[-1] <= config.task.goal.radius
                metrics["task_success"] = float(success)
                if success:
                    metrics["completion_time_s"] = completion
            else:
                distance = np.linalg.norm(
                    state["actual_position"] - desired.position[-1], axis=1
                )
                completion = sustained_completion_time(
                    state["time"], distance, tolerance=0.02, hold_time=0.0
                )
                metrics["final_error_m"] = float(distance[-1])
                if completion is not None:
                    metrics["completion_time_s"] = completion
    bundle.write_json("metrics/metrics.json", metrics)
    return metrics


def compare_configs(path: Path, output: Path) -> tuple[ExperimentBundle, ExperimentBundle]:
    config = load_config(path)
    sequential = config.model_copy(deep=True)
    sequential.planner.kind = PlannerKind.SEQUENTIAL
    spline = config.model_copy(deep=True)
    spline.planner.kind = PlannerKind.BSPLINE
    return plan_experiment(sequential, output), plan_experiment(spline, output)

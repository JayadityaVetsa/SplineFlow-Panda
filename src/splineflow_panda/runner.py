from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import ExperimentBundle
from .models import ExperimentConfig, RunStage, Trajectory
from .pipeline import evaluate_bundle, plan_experiment
from .provenance import sha256_path
from .scene import resolved_scene
from .simulation import MujocoSimulator
from .visualization import write_rgb_video


def run_experiment(
    config: ExperimentConfig,
    model: Path,
    output: Path,
    *,
    policy_checkpoint: Path | None = None,
) -> ExperimentBundle:
    bundle = plan_experiment(config, output)
    try:
        metadata_path = bundle.root / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["model_path"] = str(model)
        metadata["model_sha256"] = sha256_path(model)
        bundle.write_json("metadata.json", metadata)
        bundle.set_status(RunStage.SIMULATING, "Executing trajectory in MuJoCo")
        desired = Trajectory.load(bundle.root / "states" / "desired_trajectory.npz")
        with resolved_scene(model, config.task) as scene_path:
            simulator = MujocoSimulator(scene_path)
        recorded = simulator.execute(
            desired, config, policy_checkpoint=policy_checkpoint
        )
        frames = {key: recorded.pop(key) for key in ("rgb", "depth", "segmentation")}
        np.savez_compressed(bundle.root / "states" / "robot_state.npz", **recorded)
        for modality, values in frames.items():
            np.save(bundle.root / "frames" / f"{modality}.npy", values)
        write_rgb_video(
            bundle.root / "media" / "execution.mp4", frames["rgb"], config.camera.fps
        )
        commanded = _trajectory_from_recording(
            recorded["time"],
            recorded["desired_position"],
            desired,
            motion_end_time=float(recorded["motion_end_time"]),
        )
        executed = _trajectory_from_recording(
            recorded["time"],
            recorded["actual_position"],
            commanded,
            motion_end_time=float(recorded["motion_end_time"]),
        )
        bundle.save_trajectory(commanded, "commanded_trajectory.npz")
        bundle.save_trajectory(executed, "executed_trajectory.npz")
        evaluate_bundle(bundle)
        if policy_checkpoint is not None:
            from .learning import file_sha256

            metadata_path = bundle.root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.update(
                {
                    "evaluation_mode": "measured_learned_closed_loop",
                    "checkpoint": str(policy_checkpoint),
                    "checkpoint_sha256": file_sha256(policy_checkpoint),
                }
            )
            bundle.write_json("metadata.json", metadata)
        bundle.set_status(RunStage.COMPLETED, "Simulation completed")
        return bundle
    except Exception as error:
        bundle.set_status(RunStage.FAILED, f"Simulation failed: {error}")
        raise


def _trajectory_from_recording(
    time: np.ndarray,
    position: np.ndarray,
    reference: Trajectory,
    *,
    motion_end_time: float,
) -> Trajectory:
    velocity = np.gradient(position, time, axis=0)
    acceleration = np.gradient(velocity, time, axis=0)
    waypoint_times = np.interp(
        reference.waypoint_times,
        [reference.time[0], reference.time[-1]],
        [time[0], motion_end_time],
    )
    return Trajectory(
        time,
        position,
        velocity,
        acceleration,
        waypoint_times,
        reference.planner,
    )

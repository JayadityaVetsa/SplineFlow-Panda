from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .actions import fit_adaptive_bspline, sample_action_chunks, sample_bspline_actions
from .evaluation import classify_contact
from .geometry import intrinsic_from_fovy
from .models import ExperimentConfig, PlannerKind, Trajectory


@dataclass
class IKResult:
    qpos: np.ndarray
    position_residual: float
    orientation_residual: float
    iterations: int
    success: bool
    condition_number: float
    joint_limit_event: bool


class MujocoSimulator:
    """Thin MuJoCo adapter; scientific processing does not depend on this class."""

    def __init__(self, model_path: Path, end_effector_site: str = "attachment_site"):
        try:
            import mujoco
        except ImportError as error:
            raise RuntimeError("Install the project dependencies to use MuJoCo") from error
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, end_effector_site
        )
        self.body_id = -1
        if self.site_id < 0:
            self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        if self.site_id < 0 and self.body_id < 0:
            raise ValueError(
                f"Neither end-effector site {end_effector_site!r} nor body 'hand' was found"
            )
        self.joint_ids = np.asarray(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
                for i in range(1, 8)
            ],
            dtype=int,
        )
        self.actuator_ids = np.asarray(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"actuator{i}"
                )
                for i in range(1, 8)
            ],
            dtype=int,
        )
        if np.any(self.joint_ids < 0) or np.any(self.actuator_ids < 0):
            raise ValueError("Panda arm joints or actuators could not be resolved by name")
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.dof_indices = self.model.jnt_dofadr[self.joint_ids]
        self.robot_dofs = len(self.joint_ids)
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)
        mujoco.mj_forward(self.model, self.data)

    def robot_obstacle_mesh_clearance(self) -> float:
        """Minimum MuJoCo geom distance between Panda collision meshes and obstacles."""
        robot_bodies = {
            body_id
            for body_id in range(1, self.model.nbody)
            if (self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or "").startswith(("link", "hand", "left_finger", "right_finger"))
        }
        robot_geoms = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in robot_bodies
            and int(self.model.geom_group[geom_id]) == 3
        ]
        obstacle_geoms = [
            geom_id
            for geom_id in range(self.model.ngeom)
            if (self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_GEOM, geom_id
            ) or "").startswith("obstacle__")
        ]
        if not obstacle_geoms:
            return float("inf")
        distances = []
        for robot in robot_geoms:
            for obstacle in obstacle_geoms:
                from_to = np.zeros(6)
                distance = float(
                    self.mujoco.mj_geomDistance(
                        self.model, self.data, robot, obstacle, 10.0, from_to
                    )
                )
                # Some convex mesh/box pairs return zero while still providing distinct
                # closest points. Preserve true penetration (coincident points), otherwise
                # use the returned closest-point segment length.
                segment_distance = float(np.linalg.norm(from_to[3:] - from_to[:3]))
                if distance == 0.0 and segment_distance > 1e-9:
                    distance = segment_distance
                distances.append(distance)
        return min(distances, default=float("inf"))

    def end_effector_position(self) -> np.ndarray:
        if self.site_id >= 0:
            return self.data.site_xpos[self.site_id]
        return self.data.xpos[self.body_id]

    def end_effector_rotation(self) -> np.ndarray:
        if self.site_id >= 0:
            return self.data.site_xmat[self.site_id].reshape(3, 3)
        return self.data.xmat[self.body_id].reshape(3, 3)

    @staticmethod
    def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
        relative = np.asarray(target) @ np.asarray(current).T
        cosine = np.clip((np.trace(relative) - 1) / 2, -1.0, 1.0)
        angle = float(np.arccos(cosine))
        if angle < 1e-9:
            return np.zeros(3)
        vector = np.array(
            [
                relative[2, 1] - relative[1, 2],
                relative[0, 2] - relative[2, 0],
                relative[1, 0] - relative[0, 1],
            ]
        )
        sine = np.sin(angle)
        if abs(sine) < 1e-7:
            eigenvalues, eigenvectors = np.linalg.eigh(relative)
            axis = eigenvectors[:, np.argmin(np.abs(eigenvalues - 1))]
        else:
            axis = vector / (2 * sine)
        return axis * angle

    def configure_controller(self, config: ExperimentConfig) -> None:
        controller = config.simulation.controller
        ids = self.actuator_ids
        self.model.actuator_gainprm[ids, 0] *= controller.gain_scale
        self.model.actuator_biasprm[ids, 1] *= controller.gain_scale
        self.model.actuator_biasprm[ids, 2] *= controller.damping_scale
        self.model.actuator_forcerange[ids] *= controller.torque_limit_scale

    def solve_ik(
        self,
        target: np.ndarray,
        *,
        target_rotation: np.ndarray | None,
        damping: float,
        position_tolerance: float,
        orientation_tolerance: float,
        iterations: int,
        step_limit: float,
    ) -> IKResult:
        mujoco = self.mujoco
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        target_rotation = (
            self.end_effector_rotation().copy()
            if target_rotation is None
            else np.asarray(target_rotation)
        )
        joint_limit_event = False
        condition = float("inf")
        for iteration in range(1, iterations + 1):
            mujoco.mj_forward(self.model, self.data)
            position_error = np.asarray(target) - self.end_effector_position()
            rotation_error = self.rotation_error(
                target_rotation, self.end_effector_rotation()
            )
            position_residual = float(np.linalg.norm(position_error))
            orientation_residual = float(np.linalg.norm(rotation_error))
            if (
                position_residual <= position_tolerance
                and orientation_residual <= orientation_tolerance
            ):
                return IKResult(
                    self.data.qpos.copy(),
                    position_residual,
                    orientation_residual,
                    iteration,
                    True,
                    condition,
                    joint_limit_event,
                )
            jacobian_position.fill(0)
            jacobian_rotation.fill(0)
            if self.site_id >= 0:
                mujoco.mj_jacSite(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    self.site_id,
                )
            else:
                mujoco.mj_jacBody(
                    self.model,
                    self.data,
                    jacobian_position,
                    jacobian_rotation,
                    self.body_id,
                )
            jacobian = np.vstack(
                [
                    jacobian_position[:, self.dof_indices],
                    jacobian_rotation[:, self.dof_indices],
                ]
            )
            error = np.r_[position_error, rotation_error]
            singular_values = np.linalg.svd(jacobian, compute_uv=False)
            condition = float(
                singular_values[0] / max(singular_values[-1], np.finfo(float).eps)
            )
            step = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping**2 * np.eye(6), error
            )
            norm = np.linalg.norm(step)
            if norm > step_limit:
                step *= step_limit / norm
            self.data.qpos[self.qpos_indices] += step
            limits = self.model.jnt_range[self.joint_ids]
            limited = self.model.jnt_limited[self.joint_ids].astype(bool)
            before = self.data.qpos[self.qpos_indices].copy()
            self.data.qpos[self.qpos_indices[limited]] = np.clip(
                self.data.qpos[self.qpos_indices[limited]],
                limits[limited, 0],
                limits[limited, 1],
            )
            joint_limit_event |= not np.allclose(
                before, self.data.qpos[self.qpos_indices]
            )
        return IKResult(
            self.data.qpos.copy(),
            position_residual,
            orientation_residual,
            iterations,
            False,
            condition,
            joint_limit_event,
        )

    def execute(
        self,
        trajectory: Trajectory,
        config: ExperimentConfig,
        *,
        policy_checkpoint: Path | None = None,
    ) -> dict[str, np.ndarray]:
        if config.task.puck:
            joint_id = self.mujoco.mj_name2id(
                self.model,
                self.mujoco.mjtObj.mjOBJ_JOINT,
                f"{config.task.puck.name}_free",
            )
            if joint_id < 0:
                raise ValueError("Configured puck free joint was not found")
            address = self.model.jnt_qposadr[joint_id]
            self.data.qpos[address : address + 3] = config.task.puck.position
            self.data.qpos[address + 3 : address + 7] = [1.0, 0.0, 0.0, 0.0]
            self.mujoco.mj_forward(self.model, self.data)
        initial_qpos = self.data.qpos.copy()
        initial_ctrl = self.data.ctrl.copy()
        self.configure_controller(config)
        target_rotation = self.end_effector_rotation().copy()
        q_targets, position_residuals, orientation_residuals = [], [], []
        conditions, joint_limit_events = [], []
        for target in trajectory.position:
            result = self.solve_ik(
                target,
                target_rotation=target_rotation,
                damping=config.simulation.damping,
                position_tolerance=config.simulation.ik_position_tolerance,
                orientation_tolerance=config.simulation.ik_orientation_tolerance,
                iterations=config.simulation.ik_iterations,
                step_limit=config.simulation.ik_step_limit,
            )
            if not result.success:
                raise RuntimeError(
                    "IK failed: "
                    f"position={result.position_residual:.6f} m, "
                    f"orientation={np.rad2deg(result.orientation_residual):.3f} deg"
                )
            q_targets.append(result.qpos[self.qpos_indices])
            position_residuals.append(result.position_residual)
            orientation_residuals.append(result.orientation_residual)
            conditions.append(result.condition_number)
            joint_limit_events.append(result.joint_limit_event)
        q_targets = np.asarray(q_targets)
        execution_time = trajectory.time
        desired_position = trajectory.position
        representation = trajectory.planner
        policy_time = trajectory.time
        segment_index = np.arange(len(trajectory.time))
        action_diagnostics: dict[str, float] = {}
        if config.planner.kind == PlannerKind.ACTION_CHUNK:
            sequence = sample_action_chunks(
                trajectory.time,
                q_targets,
                policy_rate=config.planner.policy_rate,
                control_rate=config.planner.control_rate,
            )
            execution_time = sequence.time
            q_targets = sequence.command
            representation = sequence.representation
            policy_time = sequence.policy_time
            segment_index = sequence.segment_index
            action_diagnostics = sequence.diagnostics
            desired_position = np.column_stack(
                [
                    np.interp(execution_time, trajectory.time, trajectory.position[:, axis])
                    for axis in range(3)
                ]
            )
        elif config.planner.kind == PlannerKind.BSPLINE_ACTION:
            parameters = fit_adaptive_bspline(
                trajectory.time,
                q_targets,
                tolerance=config.planner.spline_tolerance,
                max_control_points=config.planner.chunk_horizon,
            )
            sequence = sample_bspline_actions(
                parameters,
                control_rate=config.planner.control_rate,
            )
            execution_time = sequence.time
            q_targets = sequence.command
            representation = sequence.representation
            policy_time = sequence.policy_time
            segment_index = sequence.segment_index
            action_diagnostics = sequence.diagnostics
            desired_position = np.column_stack(
                [
                    np.interp(execution_time, trajectory.time, trajectory.position[:, axis])
                    for axis in range(3)
                ]
            )
        loaded_policy = None
        policy_update_times: list[float] = []
        policy_alignment: list[float] = []
        policy_safety_clips = 0
        if policy_checkpoint is not None:
            from .learning import load_policy

            loaded_policy = load_policy(policy_checkpoint)
            duration = float(trajectory.time[-1]) / config.planner.speedup
            execution_time = np.arange(
                0.0,
                duration + 0.5 / config.planner.control_rate,
                1.0 / config.planner.control_rate,
            )
            desired_position = np.column_stack(
                [
                    np.interp(
                        execution_time * config.planner.speedup,
                        trajectory.time,
                        trajectory.position[:, axis],
                    )
                    for axis in range(3)
                ]
            )
            q_targets = np.repeat(q_targets[0][None], len(execution_time), axis=0)
            representation = f"learned_{loaded_policy.checkpoint['representation']}"
            policy_time = np.empty(0, dtype=float)
            segment_index = np.zeros(len(execution_time), dtype=int)
        motion_end_time = float(execution_time[-1])
        terminal_hold = config.simulation.controller.terminal_hold
        if terminal_hold > 0:
            hold_count = max(
                1, round(terminal_hold * config.simulation.controller.control_rate)
            )
            hold_time = motion_end_time + np.arange(1, hold_count + 1) / (
                config.simulation.controller.control_rate
            )
            execution_time = np.r_[execution_time, hold_time]
            q_targets = np.vstack(
                [q_targets, np.repeat(q_targets[-1][None], hold_count, axis=0)]
            )
            desired_position = np.vstack(
                [
                    desired_position,
                    np.repeat(desired_position[-1][None], hold_count, axis=0),
                ]
            )
            segment_index = np.r_[
                segment_index,
                np.repeat(segment_index[-1], hold_count),
            ]
        # Reset to the named initial state and physically settle to the first command.
        self.data.qpos[:] = initial_qpos
        self.data.ctrl[:] = initial_ctrl
        self.data.ctrl[self.actuator_ids] = q_targets[0]
        self.data.time = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        settling_time = config.simulation.controller.settling_time
        while self.data.time + 1e-12 < settling_time:
            self.mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        rgb_renderer = self.mujoco.Renderer(
            self.model, height=config.camera.height, width=config.camera.width
        )
        depth_renderer = self.mujoco.Renderer(
            self.model, height=config.camera.height, width=config.camera.width
        )
        depth_renderer.enable_depth_rendering()
        segmentation_renderer = self.mujoco.Renderer(
            self.model, height=config.camera.height, width=config.camera.width
        )
        segmentation_renderer.enable_segmentation_rendering()
        render_stride = max(
            1, round(config.simulation.controller.control_rate / config.camera.fps)
        )
        actual_position, actual_rotation, qpos, qvel, contacts = [], [], [], [], []
        actuator_force, saturation, forbidden_contacts, clearance = [], [], [], []
        rgb, depth, segmentation = [], [], []
        puck_position = []
        puck_body_id = -1
        if config.task.puck:
            puck_body_id = self.mujoco.mj_name2id(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, config.task.puck.name
            )
        next_sample = 0
        policy_buffer = np.asarray([q_targets[0]])
        policy_buffer_start = 0
        policy_stride = max(1, round(config.planner.control_rate / config.planner.policy_rate))
        policy_segment = 0
        while next_sample < len(execution_time):
            target_time = execution_time[next_sample]
            if (
                loaded_policy is not None
                and target_time <= motion_end_time + 1e-12
                and next_sample % policy_stride == 0
            ):
                from .learning import decode_policy_output

                observation = np.r_[
                    self.data.qpos[self.qpos_indices],
                    self.data.qvel[self.dof_indices],
                    desired_position[next_sample],
                ]
                prediction = loaded_policy.predict(observation)
                policy_buffer, diagnostic = decode_policy_output(
                    prediction,
                    loaded_policy.checkpoint,
                    policy_rate=config.planner.policy_rate,
                    control_rate=config.planner.control_rate,
                    speedup=config.planner.speedup,
                    last_action=q_targets[max(0, next_sample - 1)],
                )
                policy_buffer_start = next_sample
                policy_update_times.append(float(target_time))
                policy_alignment.append(diagnostic["segment_alignment_index"])
                policy_segment += 1
            if loaded_policy is not None:
                buffer_index = min(next_sample - policy_buffer_start, len(policy_buffer) - 1)
                candidate = policy_buffer[buffer_index]
                limits = self.model.jnt_range[self.joint_ids]
                limited = np.clip(candidate, limits[:, 0], limits[:, 1])
                previous = q_targets[max(0, next_sample - 1)]
                step = config.simulation.controller.command_step_limit
                safe = np.clip(limited, previous - step, previous + step)
                policy_safety_clips += int(not np.allclose(candidate, safe))
                q_targets[next_sample] = safe
                segment_index[next_sample] = policy_segment
            while self.data.time + 1e-12 < target_time:
                self.data.ctrl[self.actuator_ids] = q_targets[next_sample]
                self.mujoco.mj_step(self.model, self.data)
            self.mujoco.mj_forward(self.model, self.data)
            actual_position.append(self.end_effector_position().copy())
            actual_rotation.append(self.end_effector_rotation().copy())
            qpos.append(self.data.qpos[self.qpos_indices].copy())
            qvel.append(self.data.qvel[self.dof_indices].copy())
            contacts.append(self.data.ncon)
            forces = self.data.actuator_force[self.actuator_ids].copy()
            actuator_force.append(forces)
            force_limits = np.abs(self.model.actuator_forcerange[self.actuator_ids])
            saturation.append(
                np.any(np.abs(forces) >= 0.99 * np.maximum(force_limits[:, 1], 1e-9))
            )
            forbidden = 0
            for contact in self.data.contact:
                first = self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                ) or ""
                second = self.mujoco.mj_id2name(
                    self.model, self.mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                ) or ""
                category = classify_contact(
                    first,
                    second,
                    robot_geoms={first, second}
                    - {
                        name
                        for name in (first, second)
                        if name.startswith(("obstacle__", "puck__", "goal__"))
                        or name in {"table", "floor"}
                    },
                    obstacle_geoms={
                        name for name in (first, second) if name.startswith("obstacle__")
                    },
                    table_geoms={name for name in (first, second) if name == "table"},
                    puck_geoms={
                        name for name in (first, second) if name.startswith("puck__")
                    },
                )
                forbidden += int(category == "forbidden")
            forbidden_contacts.append(forbidden)
            clearance.append(self.robot_obstacle_mesh_clearance())
            if puck_body_id >= 0:
                puck_position.append(self.data.xpos[puck_body_id].copy())
            if next_sample % render_stride == 0:
                rgb_renderer.update_scene(self.data, camera=config.simulation.camera)
                depth_renderer.update_scene(self.data, camera=config.simulation.camera)
                segmentation_renderer.update_scene(self.data, camera=config.simulation.camera)
                rgb.append(rgb_renderer.render().copy())
                depth.append(depth_renderer.render().copy())
                segmentation.append(segmentation_renderer.render().copy())
            next_sample += 1
        if loaded_policy is not None:
            policy_time = np.asarray(policy_update_times)
            action_diagnostics = {
                "policy_updates": float(len(policy_update_times)),
                "mean_segment_alignment_index": float(np.mean(policy_alignment)),
                "safety_clipped_commands": float(policy_safety_clips),
            }
        result = {
            "time": execution_time,
            "motion_end_time": np.asarray(motion_end_time),
            "policy_time": policy_time,
            "segment_index": segment_index,
            "action_representation": np.asarray(representation),
            "action_diagnostic_names": np.asarray(list(action_diagnostics)),
            "action_diagnostic_values": np.asarray(list(action_diagnostics.values())),
            "desired_position": desired_position,
            "actual_position": np.asarray(actual_position),
            "actual_rotation": np.asarray(actual_rotation),
            "joint_position": np.asarray(qpos),
            "joint_velocity": np.asarray(qvel),
            "joint_command": np.asarray(q_targets),
            "actuator_force": np.asarray(actuator_force),
            "actuator_saturated": np.asarray(saturation),
            "ik_position_residual": np.asarray(position_residuals),
            "ik_orientation_residual": np.asarray(orientation_residuals),
            "ik_condition_number": np.asarray(conditions),
            "ik_joint_limit_event": np.asarray(joint_limit_events),
            "contact_count": np.asarray(contacts),
            "forbidden_contact_count": np.asarray(forbidden_contacts),
            "robot_obstacle_mesh_clearance": np.asarray(clearance),
            "rgb": np.asarray(rgb),
            "depth": np.asarray(depth),
            "segmentation": np.asarray(segmentation),
        }
        if puck_position:
            result["puck_position"] = np.asarray(puck_position)
        return result

    def camera_calibration(
        self, camera_name: str, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        camera_id = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if camera_id < 0:
            raise ValueError(f"Camera {camera_name!r} was not found")
        self.mujoco.mj_forward(self.model, self.data)
        camera_to_world_rotation = self.data.cam_xmat[camera_id].reshape(3, 3)
        # MuJoCo cameras look along -Z with +Y up; convert to OpenCV +Z forward/+Y down.
        world_to_camera_rotation = np.vstack(
            [
                camera_to_world_rotation[:, 0],
                -camera_to_world_rotation[:, 1],
                -camera_to_world_rotation[:, 2],
            ]
        )
        world_to_camera = np.eye(4)
        world_to_camera[:3, :3] = world_to_camera_rotation
        world_to_camera[:3, 3] = (
            -world_to_camera_rotation @ self.data.cam_xpos[camera_id]
        )
        intrinsic = intrinsic_from_fovy(
            width, height, float(self.model.cam_fovy[camera_id])
        )
        return intrinsic, world_to_camera

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import make_interp_spline

from .actions import fit_adaptive_bspline


@dataclass(frozen=True)
class LearningDataset:
    observation: np.ndarray
    chunk_target: np.ndarray
    spline_control_target: np.ndarray
    spline_control_mask: np.ndarray
    spline_interval_target: np.ndarray
    task_seed: np.ndarray

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> LearningDataset:
        with np.load(path, allow_pickle=False) as data:
            value = cls(**{name: data[name] for name in cls.__dataclass_fields__})
        value.validate()
        return value

    def validate(self) -> None:
        count = len(self.observation)
        arrays = (
            self.observation,
            self.chunk_target,
            self.spline_control_target,
            self.spline_control_mask,
            self.spline_interval_target,
            self.task_seed,
        )
        if not count or any(len(value) != count for value in arrays):
            raise ValueError("Dataset arrays must have one shared, non-zero sample dimension")
        if self.chunk_target.ndim != 3 or self.chunk_target.shape[-1] != 7:
            raise ValueError("Chunk targets must have shape (samples, horizon, 7)")
        if self.spline_control_target.shape != self.chunk_target.shape:
            raise ValueError("Spline controls and chunk targets must share a shape")
        if self.spline_control_mask.shape != self.chunk_target.shape[:2]:
            raise ValueError("Spline masks must have shape (samples, horizon)")
        if self.spline_interval_target.shape != self.chunk_target.shape[:2]:
            raise ValueError("Spline intervals must have shape (samples, horizon)")
        numeric = (self.observation, self.chunk_target, self.spline_control_target)
        if any(not np.isfinite(value).all() for value in numeric):
            raise ValueError("Dataset contains non-finite values")
        interval_sum = self.spline_interval_target.sum(axis=1)
        if np.any((interval_sum < 0.999) | (interval_sum > 1.001)):
            raise ValueError("Spline intervals must sum to one")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0)
    return mean, np.where(scale < 1e-8, 1.0, scale)


def decode_policy_output(
    prediction: np.ndarray,
    checkpoint: dict[str, object],
    *,
    policy_rate: float = 10.0,
    control_rate: float = 100.0,
    speedup: float = 1.0,
    last_action: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Decode one network prediction into controller-rate absolute joint targets."""
    horizon = int(checkpoint["horizon"])
    action_dim = int(checkpoint["action_dim"])
    control_size = horizon * action_dim
    target = (
        np.asarray(prediction[:control_size]) * np.asarray(checkpoint["target_scale"])
        + np.asarray(checkpoint["target_mean"])
    ).reshape(horizon, action_dim)
    duration = horizon / policy_rate / speedup
    samples = max(2, round(duration * control_rate))
    query = np.linspace(0.0, 1.0, samples)
    diagnostics = {"segment_alignment_index": 0.0}
    if checkpoint["representation"] == "action_chunk":
        indices = np.minimum((query * horizon).astype(int), horizon - 1)
        return target[indices], diagnostics

    logits = np.asarray(prediction[control_size : control_size + horizon], dtype=float)
    intervals = np.exp(logits - logits.max())
    intervals /= intervals.sum()
    parameter = np.r_[0.0, np.cumsum(intervals[:-1])]
    parameter[-1] = min(parameter[-1], 1.0)
    parameter = np.maximum.accumulate(parameter + np.arange(horizon) * 1e-8)
    parameter /= parameter[-1]
    curve = make_interp_spline(parameter, target, k=min(3, horizon - 1), axis=0)
    command = np.asarray(curve(query))
    if last_action is not None:
        distances = np.linalg.norm(command - np.asarray(last_action), axis=1)
        alignment = int(np.argmin(distances[: max(1, len(command) // 2)]))
        diagnostics["segment_alignment_index"] = float(alignment)
        command = command[alignment:]
    return command, diagnostics


def predict_policy(checkpoint_path: Path, observation: np.ndarray) -> tuple[np.ndarray, dict]:
    """Load a matched MLP checkpoint and return one normalized network prediction."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Install the optional learning dependencies: uv sync --extra learn"
        ) from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = nn.Sequential(
        nn.Linear(checkpoint["observation_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["output_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized = (np.asarray(observation) - checkpoint["observation_mean"]) / checkpoint[
        "observation_scale"
    ]
    with torch.no_grad():
        prediction = model(torch.as_tensor(normalized, dtype=torch.float32)).numpy()
    return prediction, checkpoint


@dataclass
class LoadedPolicy:
    model: Any
    checkpoint: dict[str, Any]
    torch: Any

    def predict(self, observation: np.ndarray) -> np.ndarray:
        mean = np.asarray(self.checkpoint["observation_mean"])
        scale = np.asarray(self.checkpoint["observation_scale"])
        normalized = (
            np.asarray(observation) - mean
        ) / scale
        with self.torch.no_grad():
            return self.model(
                self.torch.as_tensor(normalized, dtype=self.torch.float32)
            ).numpy()


def load_policy(checkpoint_path: Path) -> LoadedPolicy:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Install the optional learning dependencies: uv sync --extra learn"
        ) from error
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = nn.Sequential(
        nn.Linear(checkpoint["observation_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["output_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LoadedPolicy(model, checkpoint, torch)


def build_learning_dataset(
    experiment_root: Path,
    *,
    horizon: int = 16,
    stride: int = 4,
    spline_tolerance: float = 0.002,
) -> LearningDataset:
    observations, chunks, controls, masks, intervals, seeds = [], [], [], [], [], []
    state_paths = sorted(experiment_root.glob("**/states/robot_state.npz"))
    if not state_paths:
        raise ValueError(f"No robot_state.npz files found under {experiment_root}")
    for state_path in state_paths:
        metadata_path = state_path.parents[1] / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with np.load(state_path, allow_pickle=False) as state:
            time = state["time"]
            qpos = state["joint_position"]
            qvel = state["joint_velocity"]
            command = state["joint_command"]
            desired = state["desired_position"]
            for start in range(0, len(time), stride):
                indices = np.arange(start, min(start + horizon, len(time)))
                local_time = time[indices] - time[start]
                if len(local_time) < 4:
                    continue
                local_command = command[indices]
                fitted = fit_adaptive_bspline(
                    local_time,
                    local_command,
                    tolerance=spline_tolerance,
                    max_control_points=horizon,
                )
                control = np.empty((horizon, command.shape[1]), dtype=float)
                count = len(fitted.control_points)
                control[:count] = fitted.control_points
                control[count:] = fitted.control_points[-1]
                mask = np.zeros(horizon, dtype=bool)
                mask[:count] = True
                unique_knots = np.unique(fitted.knots)
                knot_intervals = np.diff(unique_knots)
                knot_intervals /= knot_intervals.sum()
                interval_target = np.zeros(horizon, dtype=float)
                interval_target[: len(knot_intervals)] = knot_intervals
                observations.append(np.r_[qpos[start], qvel[start], desired[start]])
                padded_chunk = np.empty((horizon, command.shape[1]), dtype=float)
                padded_chunk[: len(local_command)] = local_command
                padded_chunk[len(local_command) :] = local_command[-1]
                chunks.append(padded_chunk)
                controls.append(control)
                masks.append(mask)
                intervals.append(interval_target)
                seeds.append(metadata.get("task_seed", 0))
    return LearningDataset(
        observation=np.asarray(observations),
        chunk_target=np.asarray(chunks),
        spline_control_target=np.asarray(controls),
        spline_control_mask=np.asarray(masks),
        spline_interval_target=np.asarray(intervals),
        task_seed=np.asarray(seeds),
    )


def split_by_seed(
    dataset: LearningDataset, validation_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(dataset.task_seed)
    if len(unique) < 2:
        split = max(1, round(len(dataset.task_seed) * (1 - validation_fraction)))
        return np.arange(split), np.arange(split, len(dataset.task_seed))
    count = max(1, round(len(unique) * validation_fraction))
    validation_seeds = unique[-count:]
    validation = np.isin(dataset.task_seed, validation_seeds)
    return np.flatnonzero(~validation), np.flatnonzero(validation)


def dataset_summary(dataset: LearningDataset) -> dict[str, object]:
    dataset.validate()
    train, validation = split_by_seed(dataset)
    train_seeds = sorted(set(dataset.task_seed[train].tolist()))
    validation_seeds = sorted(set(dataset.task_seed[validation].tolist()))
    if set(train_seeds) & set(validation_seeds):
        raise ValueError("Task seeds leak across training and validation splits")
    return {
        "samples": len(dataset.observation),
        "task_seeds": sorted(set(dataset.task_seed.tolist())),
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "observation_size": int(dataset.observation.shape[1]),
        "horizon": int(dataset.chunk_target.shape[1]),
        "action_dimension": int(dataset.chunk_target.shape[2]),
        "spline_control_utilization": float(np.mean(dataset.spline_control_mask)),
    }


def train_matched_policy(
    dataset_path: Path,
    output: Path,
    *,
    representation: str,
    training_seed: int = 0,
    epochs: int = 100,
    hidden_size: int = 128,
) -> Path:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Install the optional learning dependencies: uv sync --extra learn"
        ) from error

    torch.manual_seed(training_seed)
    dataset = LearningDataset.load(dataset_path)
    train_indices, validation_indices = split_by_seed(dataset)
    observation_mean, observation_scale = normalization(dataset.observation[train_indices])
    normalized_observation = (dataset.observation - observation_mean) / observation_scale
    observation = torch.as_tensor(normalized_observation, dtype=torch.float32)
    horizon, action_dim = dataset.chunk_target.shape[1:]
    control_size = horizon * action_dim
    if representation == "action_chunk":
        raw_target = dataset.chunk_target.reshape(len(dataset.observation), -1)
        output_size = horizon * action_dim
    elif representation == "bspline_action":
        controls = dataset.spline_control_target.reshape(len(dataset.observation), -1)
        raw_target = controls
        interval_target = torch.as_tensor(
            dataset.spline_interval_target, dtype=torch.float32
        )
        control_mask = torch.as_tensor(
            np.repeat(dataset.spline_control_mask, action_dim, axis=1),
            dtype=torch.float32,
        )
        output_size = horizon * action_dim + horizon
    else:
        raise ValueError("representation must be action_chunk or bspline_action")

    target_mean, target_scale = normalization(raw_target[train_indices])
    target = torch.as_tensor((raw_target - target_mean) / target_scale, dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(observation.shape[1], hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_x, train_y = observation[train_indices], target[train_indices]
    loss_history: list[float] = []
    for _ in range(epochs):
        prediction = model(train_x)
        if representation == "action_chunk":
            loss = nn.functional.mse_loss(prediction, train_y)
        else:
            predicted_controls = prediction[:, :control_size]
            predicted_intervals = torch.softmax(prediction[:, control_size:], dim=1)
            mask = control_mask[train_indices]
            control_loss = torch.sum(((predicted_controls - train_y) * mask) ** 2) / (
                torch.sum(mask) + 1e-9
            )
            interval_loss = nn.functional.mse_loss(
                predicted_intervals, interval_target[train_indices]
            )
            loss = control_loss + interval_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        validation_prediction = model(observation[validation_indices])
        if representation == "action_chunk":
            validation_loss = float(
                nn.functional.mse_loss(
                    validation_prediction, target[validation_indices]
                )
            )
        else:
            validation_controls = validation_prediction[:, :control_size]
            validation_intervals = torch.softmax(
                validation_prediction[:, control_size:], dim=1
            )
            mask = control_mask[validation_indices]
            validation_loss = float(
                torch.sum(
                    ((validation_controls - target[validation_indices]) * mask) ** 2
                )
                / (torch.sum(mask) + 1e-9)
                + nn.functional.mse_loss(
                    validation_intervals, interval_target[validation_indices]
                )
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "representation": representation,
            "observation_size": observation.shape[1],
            "hidden_size": hidden_size,
            "output_size": output_size,
            "horizon": horizon,
            "action_dim": action_dim,
            "training_seed": training_seed,
            "validation_loss": validation_loss,
            "loss_history": loss_history,
            "observation_mean": torch.as_tensor(observation_mean, dtype=torch.float32),
            "observation_scale": torch.as_tensor(observation_scale, dtype=torch.float32),
            "target_mean": torch.as_tensor(target_mean, dtype=torch.float32),
            "target_scale": torch.as_tensor(target_scale, dtype=torch.float32),
            "dataset_sha256": file_sha256(dataset_path),
            "train_seeds": sorted(set(dataset.task_seed[train_indices].tolist())),
            "validation_seeds": sorted(set(dataset.task_seed[validation_indices].tolist())),
        },
        output,
    )
    return output


def evaluate_policy_checkpoint(checkpoint_path: Path, dataset_path: Path) -> dict[str, float | str]:
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise RuntimeError(
            "Install the optional learning dependencies: uv sync --extra learn"
        ) from error

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dataset = LearningDataset.load(dataset_path)
    _, validation_indices = split_by_seed(dataset)
    model = nn.Sequential(
        nn.Linear(checkpoint["observation_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["hidden_size"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_size"], checkpoint["output_size"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    observation_mean = np.asarray(checkpoint["observation_mean"])
    observation_scale = np.asarray(checkpoint["observation_scale"])
    normalized = (
        dataset.observation[validation_indices] - observation_mean
    ) / observation_scale
    observation = torch.as_tensor(normalized, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(observation)
    horizon = checkpoint["horizon"]
    action_dim = checkpoint["action_dim"]
    if checkpoint["representation"] == "action_chunk":
        target = dataset.chunk_target[validation_indices].reshape(len(observation), -1)
        reconstruction = prediction.numpy() * np.asarray(
            checkpoint["target_scale"]
        ) + np.asarray(checkpoint["target_mean"])
    else:
        control_size = horizon * action_dim
        target = dataset.spline_control_target[validation_indices].reshape(
            len(observation), -1
        )
        reconstruction = prediction[:, :control_size].numpy() * np.asarray(
            checkpoint["target_scale"]
        ) + np.asarray(checkpoint["target_mean"])
    return {
        "evaluation_mode": "open_loop_validation",
        "representation": checkpoint["representation"],
        "samples": float(len(validation_indices)),
        "reconstruction_mse": float(np.mean((reconstruction - target) ** 2)),
        "training_validation_loss": float(checkpoint["validation_loss"]),
        "dataset_sha256": checkpoint["dataset_sha256"],
    }

from pathlib import Path

import numpy as np

from splineflow_panda.learning import (
    LearningDataset,
    dataset_summary,
    decode_policy_output,
    evaluate_policy_checkpoint,
    normalization,
    split_by_seed,
    train_matched_policy,
)


def test_learning_dataset_round_trip_and_seed_split(tmp_path: Path) -> None:
    dataset = LearningDataset(
        observation=np.zeros((4, 17)),
        chunk_target=np.zeros((4, 16, 7)),
        spline_control_target=np.zeros((4, 16, 7)),
        spline_control_mask=np.ones((4, 16), dtype=bool),
        spline_interval_target=np.full((4, 16), 1 / 16),
        task_seed=np.array([1, 1, 2, 2]),
    )
    path = tmp_path / "dataset.npz"
    dataset.save(path)
    loaded = LearningDataset.load(path)
    train, validation = split_by_seed(loaded, validation_fraction=0.5)
    assert set(loaded.task_seed[train]) == {1}
    assert set(loaded.task_seed[validation]) == {2}


def test_single_seed_smoke_dataset_still_has_train_and_validation() -> None:
    dataset = LearningDataset(
        observation=np.zeros((10, 17)),
        chunk_target=np.zeros((10, 16, 7)),
        spline_control_target=np.zeros((10, 16, 7)),
        spline_control_mask=np.ones((10, 16), dtype=bool),
        spline_interval_target=np.zeros((10, 16)),
        task_seed=np.ones(10, dtype=int),
    )
    train, validation = split_by_seed(dataset)
    assert len(train) == 8
    assert len(validation) == 2


def test_normalization_and_policy_decoding_are_finite() -> None:
    values = np.array([[1.0, 2.0], [1.0, 4.0]])
    mean, scale = normalization(values)
    assert np.allclose(mean, [1.0, 3.0])
    assert np.allclose(scale, [1.0, 1.0])
    checkpoint = {
        "horizon": 4,
        "action_dim": 7,
        "representation": "bspline_action",
        "target_mean": np.zeros(28),
        "target_scale": np.ones(28),
    }
    prediction = np.r_[np.arange(28) / 100, np.zeros(4)]
    command, diagnostic = decode_policy_output(prediction, checkpoint)
    assert command.shape[1] == 7
    assert np.isfinite(command).all()
    assert diagnostic["segment_alignment_index"] == 0


def test_dataset_summary_has_disjoint_seed_split() -> None:
    dataset = LearningDataset(
        observation=np.zeros((4, 17)),
        chunk_target=np.zeros((4, 16, 7)),
        spline_control_target=np.zeros((4, 16, 7)),
        spline_control_mask=np.ones((4, 16), dtype=bool),
        spline_interval_target=np.full((4, 16), 1 / 16),
        task_seed=np.array([1, 1, 2, 2]),
    )
    summary = dataset_summary(dataset)
    assert set(summary["train_seeds"]).isdisjoint(summary["validation_seeds"])


def test_checkpoint_is_safe_loadable_and_contains_provenance(tmp_path: Path) -> None:
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    dataset = LearningDataset(
        observation=rng.normal(size=(8, 17)),
        chunk_target=rng.normal(scale=0.1, size=(8, 4, 7)),
        spline_control_target=rng.normal(scale=0.1, size=(8, 4, 7)),
        spline_control_mask=np.ones((8, 4), dtype=bool),
        spline_interval_target=np.full((8, 4), 0.25),
        task_seed=np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    dataset_path = tmp_path / "dataset.npz"
    checkpoint_path = tmp_path / "policy.pt"
    dataset.save(dataset_path)
    train_matched_policy(
        dataset_path,
        checkpoint_path,
        representation="action_chunk",
        epochs=1,
        hidden_size=8,
    )
    result = evaluate_policy_checkpoint(checkpoint_path, dataset_path)
    assert result["dataset_sha256"]
    assert np.isfinite(result["reconstruction_mse"])

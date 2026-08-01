# Experiment bundle schema 0.3.0

Every run has a unique directory containing its resolved `config.yaml`,
`metadata.json`, and `status.json`. Numeric robot states and desired, commanded, and
executed trajectories live in `states/`; scalar measurements live in `metrics/`;
recorded RGB, metric depth, and segmentation arrays live in `frames/`; and replay
video lives in `media/`.

Required minimal files:

- `config.yaml`
- `metadata.json`
- `status.json`
- `states/desired_trajectory.npz`

A complete simulation additionally contains `states/robot_state.npz`, commanded and
executed trajectories, `metrics/metrics.json`, and its recorded modalities. Flow and
noise tensors are deliberately excluded from schema 0.3 because they are outside the
B-spline action milestone.

Benchmark directories contain `rollouts.json` as the source ledger and `report.json`
as its aggregate. A report without a rollout ledger is not milestone evidence.

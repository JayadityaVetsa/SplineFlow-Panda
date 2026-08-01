# SplineFlow-Panda

SplineFlow-Panda is an educational MuJoCo benchmark for studying how an action
representation changes robot execution. It compares discrete absolute joint-position
chunks with continuous cubic B-spline actions on the same Franka Panda tasks,
controller, layouts, and success criteria.

The research question is:

> How much faster can the robot execute the same task with B-spline actions before
> tracking error, collisions, or task failure become unacceptable?

This is a reproduction-inspired student project, not an official reproduction and not
a claim of a novel algorithm.

## Measured development result

The committed compact result summarizes eight real MuJoCo rollouts: two paired
direction-change layouts, two representations, and target speedups of 1x and 2x.

| Representation | 1x success | 2x success | Reliable frontier |
|---|---:|---:|---:|
| Action chunk | 2/2 | 0/2 | 1x |
| B-spline action | 2/2 | 2/2 | 2x |

At 2x, action chunks violated the 2 cm waypoint criterion. B-spline actions completed
both layouts in about 5 seconds, compared with the successful 1x action-chunk baseline
of about 10 seconds. The ledger, paired layouts, aggregate report, and interactive plot
are in [`results/v0.3-development`](results/v0.3-development).

Two seeds are enough to demonstrate the pipeline, not to support a general research
claim. Use the milestone command to produce the planned ten-seed result.

## What is being compared?

An action chunk updates an absolute joint command at 10 Hz and holds it until the next
update. A B-spline predicts a continuous joint-space curve and samples it at the 100 Hz
controller rate. Both are executed by the same MuJoCo position controller.

For degree `p`, control points `c_i`, and basis functions `N_i,p`, the curve is

```text
q(t) = sum_i N_i,p(t) c_i
```

Temporal scaling evaluates the same normalized curve at `s*t`; ideal derivatives scale
as velocity `s`, acceleration `s^2`, and jerk `s^3`. That is why success and completion
time must be reported together: requesting a larger speedup is not itself a result.

## Quick start on Windows

```powershell
uv sync --all-extras
.\scripts\fetch_menagerie.ps1
.\.venv\Scripts\Activate.ps1
splineflow validate configs/scenarios/direction_change.yaml
splineflow run configs/scenarios/direction_change.yaml
splineflow dashboard
```

The fetch script sparsely downloads only the Franka Panda model from MuJoCo Menagerie.

## Reproduce and audit results

Run a small smoke benchmark:

```powershell
splineflow benchmark configs/scenarios/direction_change.yaml --seeds 2 --speedups 1,2
splineflow validate-report benchmarks/<generated-benchmark>
```

Inspect the complete planned milestone without running it:

```powershell
splineflow reproduce-milestone --dry-run
```

Run the full scripted milestone (240 CPU rollouts):

```powershell
splineflow reproduce-milestone --seeds 10 --speedups 1,1.5,2,3
```

The command benchmarks direction change, obstacle corridor, and planar pushing. It
stores paired layouts before execution and validates every aggregate against the
rollout ledger.

## Experimental learned policies

The optional state-policy pipeline trains matched MLPs on the same demonstrations:

```powershell
splineflow generate-dataset configs/scenarios/direction_change.yaml --seeds 10
splineflow train datasets/demonstrations.npz --representation action_chunk `
  --output checkpoints/action-chunk.pt
splineflow train datasets/demonstrations.npz --representation bspline_action `
  --output checkpoints/bspline-action.pt
splineflow rollout checkpoints/bspline-action.pt datasets/demonstrations.npz `
  --mode open-loop
splineflow rollout checkpoints/bspline-action.pt `
  configs/scenarios/direction_change.yaml --mode closed-loop
```

Closed-loop inference observes measured joint position and velocity, replans at 10 Hz,
and issues controller commands at 100 Hz. Checkpoints contain training-only
normalization statistics, seed splits, dataset hashes, loss history, and architecture
metadata. Learned results remain explicitly experimental unless both representations
reach at least 80% success at 1x on ten unseen layouts.

## Metrics

- Success rate: successful paired rollouts divided by all paired rollouts.
- Achieved speedup: mean successful 1x chunk time divided by the method's mean
  successful completion time.
- Tracking RMSE: root mean squared Cartesian commanded-to-executed error.
- Smoothness: RMS acceleration and jerk from timestamp-aware derivatives.
- Safety diagnostics: forbidden contacts, actuator saturation, and collision-geometry
  mesh clearance.

Failures count against success rate and are excluded from mean completion time. Every
dashboard result is labeled as measured scripted MuJoCo, measured learned closed-loop,
or open-loop model evaluation.

## Architecture

```text
YAML scenario -> paired layout manifest -> task-space reference -> IK demonstration
     -> action chunk or B-spline joint commands -> 100 Hz controller -> MuJoCo
     -> synchronized state/contact recording -> metrics ledger -> validated report

Demonstrations -> paired targets -> matched MLP -> receding-horizon closed-loop rollout
```

Core robotics and reporting logic lives in the Python package. Streamlit only reads
configuration and saved experiment artifacts.

## Testing

```powershell
ruff check src tests
pytest -m "not slow"
uv build
```

The manually triggered `MuJoCo smoke test` GitHub Action downloads the model and runs a
headless rollout. Numerical tests use tolerances rather than brittle bitwise physics
equality.

## Limitations

- Results are simulation-only and do not establish real-robot performance or safety.
- The committed frontier contains two seeds and one scenario; it is a development
  artifact, not a publication-scale benchmark.
- The current learned policies are compact state-based regressors, not Diffusion Policy,
  ACT, or a reproduction of the paper's large-scale experiments.
- Obstacle clearance is measured after planning; the planner does not guarantee
  collision avoidance.
- RGB, metric depth, and segmentation are diagnostics, not policy inputs.
- Fixed end-effector orientation and position control simplify the manipulation problem.

## Citation and attribution

Project citation metadata is provided in [`CITATION.cff`](CITATION.cff). The mechanism is
inspired by:

> Xiaoshen Han, Haoyu Xiong, Haonan Chen, Chaoqi Liu, Antonio Torralba, Yuke Zhu,
> and Yilun Du. "B-spline Policy: Accelerating Manipulation Policies via B-spline
> Action Representations." arXiv:2607.09648, 2026.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for MuJoCo and model attribution.
Original project code is MIT licensed.

Additional technical documentation is available in [`docs`](docs), including the
learning path, coordinate conventions, metrics, architecture, and research context.

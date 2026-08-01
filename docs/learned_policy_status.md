# Learned-policy status

The state-based learned-policy path is implemented but remains experimental. It uses
matched two-layer MLPs, training-only normalization, held-out task-seed validation,
10 Hz receding-horizon inference, and 100 Hz joint-position control.

## Development smoke test

Both representations trained and completed a closed-loop MuJoCo execution using the
existing two-seed development demonstrations. Neither passed the publication gate.

| Representation | Open-loop reconstruction MSE | Closed-loop Cartesian RMSE | Result |
|---|---:|---:|---|
| Action chunk | 0.01796 | 0.120 m | Failed 1x task criteria |
| B-spline action | 0.02017 | 0.725 m | Failed 1x task and safety criteria |

The B-spline smoke checkpoint extrapolated between predicted controls, causing command
overshoot, actuator saturation, and contacts. The runtime now clips predictions to
joint limits and limits per-sample command changes, but this safety envelope is not a
substitute for a successful policy. These smoke numbers are diagnostic and are not used
in the headline scripted action-representation result.

Promotion requires both representations to achieve at least 80% success at 1x over ten
unseen layouts. Until then, `rollout --mode closed-loop` is an experimental research
tool and the validated scripted benchmark is the completed v0.3 study.

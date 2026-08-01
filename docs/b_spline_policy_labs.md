# B-spline policy labs

These labs use the production pipeline rather than duplicating it in notebooks.

## 1. Servo lag

Run `splineflow calibrate-controller configs/scenarios/open_space.yaml`.
Compare commanded and measured joints. Stiffness can reduce lag, but force limits
and contact can prevent tracking regardless of gain.

Checkpoint: why must controller error be repaired before comparing representations?

## 2. Pose inverse kinematics

The solver minimizes a six-dimensional residual: position plus axis-angle
orientation. Change damping and inspect residual, condition number, and limit events.

Checkpoint: why does low position error not guarantee a valid orientation?

## 3. Temporal scaling

For `q(u)`, speedup `n` executes `q(n t)`. Geometry is unchanged, but velocity,
acceleration, and jerk scale approximately as `n`, `n²`, and `n³`.

Checkpoint: predict what happens when a four-second motion executes in one second.

## 4. Chunks versus continuous actions

`action_chunk` holds discrete policy output until the next inference and is not
smoothed. `bspline_action` fits a cubic curve and samples it at 100 Hz even when
policy inference occurs at 10 Hz.

Checkpoint: why can a higher controller rate not remove an existing chunk boundary?

## 5. Adaptive knots

Greedy insertion adds knots where fitting error is high, allocating representation
capacity to curved portions of a demonstration.

Checkpoint: distinguish required waypoints from spline control points.

## 6. Speed-success frontier

Run `splineflow benchmark configs/scenarios/direction_change.yaml --seeds 10`.
Both representations receive the same task seed, state, controller, and speedup.

The result is the highest speedup maintaining the declared success threshold—not
the shortest configured duration.

Checkpoint: why must completion time always be reported with success rate?

# Learning path

Every phase follows **concept → experiment → implementation → test → interpretation**.

## 1. Curves and time

Start with `splineflow plan`. Inspect position, velocity, and acceleration. A waypoint
is a required point; a control point shapes a curve and need not lie on it. The
B-spline implementation interpolates required waypoints using chord-length timing.
The sequential planner uses a minimum-jerk polynomial per segment and reaches zero
speed at each boundary.

Explain-it-back checkpoint: why can a spatially smooth curve still execute badly?

## 2. Robot state and kinematics

MuJoCo stores generalized position `qpos` and velocity `qvel`. Forward kinematics
maps them to an end-effector position. The translational Jacobian locally maps joint
velocity to Cartesian velocity.

Explain-it-back checkpoint: why is inverse kinematics not a unique inverse function?

## 3. Control and measurement

IK creates joint targets; the controller applies them; physics produces measured
motion. Always compare desired against measured motion.

Explain-it-back checkpoint: name three reasons executed jerk can exceed planned jerk.

## 4. Reproducible experiments

Configs, seeds, schema versions, statuses, and immutable directories make results
auditable. Failed experiments are data and remain inspectable.

## 5. Camera geometry

Projection applies world-to-camera transformation, intrinsic calibration, then
perspective division. Image pixels use `(u,v)`: right and down.

## 6. Flow and noise

Guidance flow encodes intended displacement `(du,dv)` per frame. It is not observed
RGB optical flow. Backward warping asks where each destination pixel should sample
from in the source.


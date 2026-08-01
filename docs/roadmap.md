# Research roadmap

SplineFlow-Panda is currently a trajectory-representation and simulation benchmark.
It is not yet a learned robot policy. The next stages deliberately separate reliable
robotics evaluation from policy learning.

## 1. Establish a trustworthy trajectory benchmark

- Tune and validate the controller until tracking error is small relative to each path.
- Make obstacles scenario-specific and verify collision and clearance measurements.
- Apply every exposed controller parameter or remove it from the configuration schema.
- Compare direct waypoint, minimum-jerk segment, and interpolating B-spline
  representations under matched movement duration, controller, initial state, and rate.
- Report results over seeded task variations with confidence intervals.
- Add end-effector orientation and explicit SE(3) conventions.

## 2. Learn a spline policy

- Construct demonstrations containing observations and successful action trajectories.
- Train a policy to predict spline parameters or residual spline corrections.
- Use the same observation encoder and training budget for a direct action-chunk baseline.
- Evaluate temporal resampling, trajectory compression, perturbation recovery,
  constraint projection, and execution smoothness.
- Keep the planner interface as the decoder between predicted parameters and executable
  robot targets.

## 3. Use perception as policy input

- Convert metric depth into point clouds and obstacle-relative geometric features.
- Use segmentation for controlled object-centric experiments, not as a required
  real-world input.
- Add image observations only after state-based policy experiments are reproducible.

## 4. Extend to visual guidance

- Reintroduce projected guidance fields and temporally warped noise as a separate
  experiment family.
- Compare guidance fields with measured image motion and clearly distinguish them from
  estimated optical flow.

## Release gates

Before the first public research release:

- document model and code licensing;
- add a citation file and contribution guide;
- publish a small reproducible benchmark and sample recording;
- trace every reported metric to a test or documented validation;
- state limitations prominently, including position-only IK where applicable.

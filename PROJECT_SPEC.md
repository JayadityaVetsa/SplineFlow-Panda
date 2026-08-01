# SplineFlow-Panda

## Project Specification and Product Charter

**Working title:** SplineFlow-Panda  
**Subtitle:** Interactive Continuous Robot Planning and Visual-Flow Dataset Generation for the Franka Panda  
**Document status:** Initial specification  
**Target release:** v0.1 — Local Research and Portfolio MVP  
**Primary platform:** Local desktop development environment  
**Primary language:** Python  
**Simulation engine:** MuJoCo  
**Robot:** Franka Emika Panda  

---

## 1. Executive Summary

SplineFlow-Panda is a local robotics research and visualization project that plans continuous multi-waypoint trajectories for a simulated Franka Panda, executes those trajectories in MuJoCo, and converts the resulting three-dimensional robot motion into visual guidance representations suitable for future video world-model research.

The first release will compare two trajectory strategies:

1. Sequential point-to-point waypoint motion.
2. Continuous cubic B-spline motion through the same task.

For each experiment, the system will record synchronized robot states and visual observations, including RGB, depth, segmentation, projected image-space trajectories, guidance-flow fields, and warped-noise tensors. An interactive dashboard will allow users to configure experiments, run simulations, replay results, compare planners, inspect metrics, and export datasets.

The project is intentionally designed to be useful without requiring a large diffusion model, a cloud GPU, or a physical robot. It establishes the mathematical, robotic, visual, and data-generation foundations needed for later experiments with motion-controllable video diffusion models.

---

## 2. Project Vision

The long-term research question is:

> Can continuous robot trajectories be converted into visual motion guidance that helps a video world model predict and evaluate the physical consequences of candidate robot actions?

The v0.1 project does not attempt to answer that entire question. It builds and validates the infrastructure required to study it:

- Continuous robot trajectory construction.
- Physics-based trajectory execution.
- Camera-aware projection from 3D motion to 2D motion.
- Visual guidance-field generation.
- Structured experiment recording.
- Quantitative comparison of planned and observed motion.
- Accessible interactive exploration.

The project should feel like a small research instrument, not merely a simulation demo.

---

## 3. Purpose

### 3.1 Educational purpose

The project should help its developer gain practical understanding of:

- Forward and inverse robot kinematics.
- Joint-space and task-space representations.
- Trajectory interpolation and time parameterization.
- Position, velocity, acceleration, and jerk.
- B-spline control points, knots, degree, and continuity.
- Robot control in a physics simulator.
- Collision and joint-limit constraints.
- Pinhole-camera projection and coordinate transformations.
- Optical flow and motion-guidance representations.
- Reproducible robotics experimentation.
- Dataset design for embodied and generative AI.

### 3.2 Research purpose

The project should provide a controlled environment for testing how well trajectory-derived guidance fields describe the visual motion produced by a robot.

It should enable questions such as:

- Does a B-spline trajectory produce smoother execution than chained waypoint commands?
- How closely does actual end-effector motion follow the desired curve?
- How does camera viewpoint affect the projected motion path?
- How well does sparse or local guidance flow approximate observed visual motion?
- How should a robot action be represented for use by a future video world model?

### 3.3 Portfolio purpose

The public repository should demonstrate competence across:

- Robotics simulation.
- Motion planning.
- Robot control.
- Numerical methods.
- Computer vision.
- Experimental evaluation.
- Research software engineering.
- Interactive technical visualization.
- Clear documentation and reproducibility.

---

## 4. End Goal

The v0.1 end goal is a polished local application in which a user can:

1. Open a predefined Franka Panda scene.
2. Select or configure multiple 3D end-effector waypoints.
3. Choose sequential or B-spline trajectory planning.
4. Run the trajectory in MuJoCo.
5. View the robot execution and camera output.
6. Inspect RGB, depth, segmentation, desired motion, observed motion, guidance flow, and warped noise.
7. Compare trajectory smoothness, accuracy, clearance, timing, and visual-flow agreement.
8. Export a complete, reproducible experiment bundle.

A successful release must include multiple reproducible example scenarios, automated metrics, documentation, tests, and a short visual demonstration.

---

## 5. Intended Users

### 5.1 Primary user: robotics or embodied-AI student

This user wants to understand how a continuous trajectory becomes robot motion and how that motion appears in camera observations.

The user should not need expert knowledge of MuJoCo or video diffusion to run the included examples.

### 5.2 Secondary user: research mentor or reviewer

This user wants to inspect:

- The technical assumptions.
- The trajectory comparison methodology.
- The quality of the generated data.
- The connection to video world-model research.
- The reproducibility of the results.

### 5.3 Secondary user: recruiter or GitHub visitor

This user may spend only a few minutes on the repository. The landing page must quickly communicate:

- What problem is being studied.
- What the system does.
- What was implemented by the author.
- What the key quantitative results are.
- How to run one representative demo.

---

## 6. Core Use Cases

### UC-1: Compare sequential and continuous motion

The user selects the same ordered waypoints for both planners and compares:

- Execution time.
- Number and duration of intermediate stops.
- Path length.
- Tracking error.
- Velocity continuity.
- Acceleration and jerk.

### UC-2: Generate a visual robot-trajectory dataset

The user runs an experiment and exports synchronized:

- Robot state.
- Planned trajectory.
- Executed trajectory.
- RGB frames.
- Depth frames.
- Segmentation masks.
- Projected trajectories.
- Guidance-flow fields.
- Warped-noise tensors.
- Experiment metadata and metrics.

### UC-3: Inspect 3D-to-2D trajectory projection

The user views a world-space end-effector path overlaid onto the rendered camera frames and verifies that the projection follows the visible robot motion.

### UC-4: Compare guidance flow with observed motion

The user compares a flow field constructed from the desired trajectory with motion estimated or derived from the rendered execution.

### UC-5: Reproduce a benchmark

The user runs a committed scenario with a fixed seed and obtains results within documented numerical tolerances.

### UC-6: Explore an experiment interactively

The user uses the dashboard to change waypoints, planner parameters, speed, or camera settings and then views the resulting effects.

---

## 7. Scope

## 7.1 Required for v0.1

### Simulation

- Franka Panda model running in MuJoCo.
- Fixed-base robot.
- At least one fixed RGB-D camera.
- Static tabletop or workspace.
- Simple static geometric obstacles.
- Deterministic simulation configuration where practical.

### Trajectory planning

- Ordered 3D end-effector waypoints.
- Sequential point-to-point baseline.
- Continuous cubic B-spline trajectory.
- Explicit time sampling.
- Desired position and velocity calculation.
- Configurable motion duration or nominal speed.

### Robot execution

- Inverse-kinematics conversion from task-space targets to joint targets.
- Joint-space position control or another documented controller.
- Joint-limit validation.
- Collision/contact reporting.
- Desired and actual end-effector tracking.

### Visual data generation

- RGB frames.
- Depth frames.
- Robot or object segmentation masks.
- Camera calibration metadata.
- 3D-to-2D projection of desired and actual end-effector trajectories.
- Sparse trajectory guidance.
- Gaussian-local dense guidance flow.
- Temporally warped Gaussian-noise sequence.

### Evaluation

- Goal and waypoint error.
- Path length.
- Execution time.
- Planning time.
- Velocity, acceleration, and jerk metrics.
- Number or duration of intermediate stops.
- Minimum obstacle clearance where available.
- Collision/contact count.
- Projected image-space tracking error.
- Guidance-versus-observed motion comparison.

### Dashboard

- Experiment configuration.
- Planner selection.
- Simulation launch.
- Progress and error reporting.
- Result replay.
- Visualization-layer selection.
- Planner comparison.
- Metric display.
- Export controls.

### Engineering

- Command-line experiment runner.
- Configuration files.
- Structured output schema.
- Automated tests for critical mathematics.
- Reproducible example scenarios.
- Installation and usage documentation.

## 7.2 Optional for v0.1

- Basic obstacle-aware candidate B-spline sampling.
- Dense observed optical flow estimated from RGB frames.
- Multiple fixed camera presets.
- Batch experiment execution.
- A simple report generator.
- A lightweight hosted gallery of precomputed results.

## 7.3 Explicitly out of scope for v0.1

- Wan2.2 inference or fine-tuning.
- Training a large video diffusion model.
- Claims of learned world-model planning.
- Physical Franka deployment.
- Safety certification.
- Real-time browser-to-simulator teleoperation.
- Dynamic obstacles.
- Grasping and dexterous manipulation.
- Photorealistic rendering.
- Full-body, pixel-perfect future robot rendering.
- General-purpose motion planning.
- Replacement of cuRobo or other production planners.

These exclusions protect the first release from uncontrolled scope expansion.

---

## 8. Conceptual System Model

```text
User / Experiment Configuration
              |
              v
      Scene and Waypoints
              |
              v
   +-----------------------+
   | Trajectory Planner    |
   | - Sequential baseline |
   | - Cubic B-spline      |
   +-----------------------+
              |
              v
 Desired task-space trajectory
              |
              v
   Inverse Kinematics + Controller
              |
              v
       MuJoCo Simulation
          /          \
         v            v
 Robot state       Camera observations
         \            /
          v          v
     Synchronized Recorder
              |
              v
  Projection + Flow + Noise Pipeline
              |
              v
 Metrics, Artifacts, Dataset, Dashboard
```

---

## 9. Terminology and Important Distinctions

### Desired trajectory

The time-parameterized path produced by the planner.

### Executed trajectory

The measured end-effector path obtained from the simulator during controller execution.

### Guidance flow

A flow field constructed from the desired projected trajectory. It describes intended image-space motion and is suitable as a future conditioning signal.

### Observed motion or observed flow

Motion derived from actual rendered frames, simulator geometry, or an optical-flow estimator.

### Warped noise

A temporally correlated Gaussian-noise sequence created by transporting noise according to a guidance-flow field.

### Planner

The component that produces a desired trajectory. The controller is separate and attempts to execute that trajectory.

### Waypoint interpolation

A spline may pass through required waypoints or approximate control points depending on its construction. The implementation and UI must clearly distinguish interpolation points from B-spline control points.

---

## 10. Functional Requirements

Each requirement is labeled for future issue tracking.

### 10.1 Scene and configuration

- **FR-SCENE-001:** The system shall load a Franka Panda simulation scene.
- **FR-SCENE-002:** The system shall support at least three ordered task-space waypoints.
- **FR-SCENE-003:** The system shall support predefined benchmark scenarios.
- **FR-SCENE-004:** The system shall allow configuration of motion duration, sampling rate, and random seed.
- **FR-SCENE-005:** The system shall support simple static obstacles.
- **FR-SCENE-006:** The system shall store camera intrinsic and extrinsic parameters with every experiment.

### 10.2 Trajectory generation

- **FR-TRAJ-001:** The system shall generate a sequential point-to-point baseline.
- **FR-TRAJ-002:** The system shall generate a cubic B-spline trajectory.
- **FR-TRAJ-003:** The system shall sample desired position over time.
- **FR-TRAJ-004:** The system shall compute or sample velocity and acceleration.
- **FR-TRAJ-005:** The system shall validate required waypoint passage within a configurable tolerance.
- **FR-TRAJ-006:** The system shall expose spline degree, duration, and sampling parameters in configuration.
- **FR-TRAJ-007:** The system shall reject invalid waypoint sets with an actionable error.
- **FR-TRAJ-008:** The system shall preserve waypoint order.

### 10.3 Kinematics and control

- **FR-CTRL-001:** The system shall compute joint targets for task-space trajectory samples.
- **FR-CTRL-002:** The system shall detect IK failures.
- **FR-CTRL-003:** The system shall validate joint position limits.
- **FR-CTRL-004:** The system shall record desired and measured joint states.
- **FR-CTRL-005:** The system shall record desired and measured end-effector poses.
- **FR-CTRL-006:** The system shall report contacts or collisions.
- **FR-CTRL-007:** The controller and its gains shall be explicitly documented.

### 10.4 Recording

- **FR-REC-001:** The system shall record state and visual data using synchronized timestamps or frame indices.
- **FR-REC-002:** The system shall record RGB frames.
- **FR-REC-003:** The system shall record metric depth or document its exact representation.
- **FR-REC-004:** The system shall record segmentation masks.
- **FR-REC-005:** The system shall preserve the configuration used for each run.
- **FR-REC-006:** The system shall store software and schema versions in experiment metadata.
- **FR-REC-007:** The system shall avoid silently overwriting an existing experiment.

### 10.5 Projection and flow

- **FR-FLOW-001:** The system shall project world-space end-effector points into image coordinates.
- **FR-FLOW-002:** The system shall identify points that are behind the camera or outside the image.
- **FR-FLOW-003:** The system shall render desired and actual trajectory overlays.
- **FR-FLOW-004:** The system shall generate sparse frame-to-frame displacement vectors.
- **FR-FLOW-005:** The system shall generate a Gaussian-local dense flow field.
- **FR-FLOW-006:** The system shall store flow direction, magnitude, units, and coordinate convention.
- **FR-FLOW-007:** The system shall generate a deterministic warped-noise sequence from a specified seed.
- **FR-FLOW-008:** The system shall visualize flow direction and magnitude.

### 10.6 Metrics

- **FR-METRIC-001:** The system shall calculate trajectory tracking error.
- **FR-METRIC-002:** The system shall calculate waypoint error.
- **FR-METRIC-003:** The system shall calculate path length.
- **FR-METRIC-004:** The system shall calculate velocity, acceleration, and jerk summaries.
- **FR-METRIC-005:** The system shall estimate intermediate stopping behavior.
- **FR-METRIC-006:** The system shall report planning and execution time.
- **FR-METRIC-007:** The system shall report collisions and minimum clearance when supported.
- **FR-METRIC-008:** The system shall calculate projected desired-versus-actual path error in pixels.
- **FR-METRIC-009:** The system shall compare guidance motion with observed motion using a documented metric.

### 10.7 Dashboard

- **FR-UI-001:** The dashboard shall expose predefined scenarios.
- **FR-UI-002:** The dashboard shall allow planner selection.
- **FR-UI-003:** The dashboard shall allow waypoint inspection and editing.
- **FR-UI-004:** The dashboard shall launch a local experiment.
- **FR-UI-005:** The dashboard shall clearly display running, completed, and failed states.
- **FR-UI-006:** The dashboard shall replay synchronized results.
- **FR-UI-007:** The dashboard shall switch among RGB, depth, segmentation, flow, noise, and overlays.
- **FR-UI-008:** The dashboard shall compare two compatible experiments.
- **FR-UI-009:** The dashboard shall show metric definitions or tooltips.
- **FR-UI-010:** The dashboard shall export selected or complete artifacts.

### 10.8 CLI

- **FR-CLI-001:** A user shall be able to run an experiment without the dashboard.
- **FR-CLI-002:** A user shall be able to evaluate an existing experiment.
- **FR-CLI-003:** A user shall be able to generate flow and warped noise from an existing valid recording.
- **FR-CLI-004:** CLI commands shall return nonzero status on failure.
- **FR-CLI-005:** CLI errors shall identify invalid configurations or missing artifacts.

---

## 11. Non-Functional Requirements

- **NFR-001 — Reproducibility:** Fixed configurations and seeds should produce equivalent results within documented tolerances.
- **NFR-002 — Local operation:** Core features must run locally without paid services.
- **NFR-003 — CPU accessibility:** Core planning, simulation, recording, and analysis must not require a GPU.
- **NFR-004 — Modularity:** Planning, simulation, recording, projection, flow, evaluation, and UI must remain separable modules.
- **NFR-005 — Inspectability:** Experiment artifacts must use documented, common formats where practical.
- **NFR-006 — Testability:** Mathematical transformations and metrics must be testable without launching the full dashboard.
- **NFR-007 — Usability:** A new user should be able to run one predefined demo by following the README.
- **NFR-008 — Failure transparency:** IK failures, invalid projections, collisions, and missing artifacts must not be hidden.
- **NFR-009 — Portability:** The project should support major desktop operating systems where dependencies allow.
- **NFR-010 — Performance:** The dashboard must remain responsive while a simulation runs in a separate process or worker.
- **NFR-011 — Documentation:** Coordinate frames, tensor shapes, units, and conventions must be documented.
- **NFR-012 — Honest framing:** Results must distinguish implemented evidence from proposed future research.

---

## 12. Technical Design

## 12.1 Recommended component boundaries

### Configuration layer

Responsibilities:

- Load and validate experiment configurations.
- Assign experiment identifiers.
- Store seeds and version information.

### Simulation layer

Responsibilities:

- Load MuJoCo models.
- Step physics.
- Expose robot and contact state.
- Render synchronized camera modalities.

### Planning layer

Responsibilities:

- Implement the sequential baseline.
- Implement the B-spline trajectory.
- Return a common trajectory representation.

### Kinematics and control layer

Responsibilities:

- Solve IK.
- Validate joint limits.
- Convert desired states to actuator commands.
- Report tracking and feasibility failures.

### Recording layer

Responsibilities:

- Synchronize numeric state with rendered frames.
- Write artifact files.
- Validate output completeness.

### Vision geometry layer

Responsibilities:

- Manage coordinate frames.
- Project 3D points.
- Generate overlays.
- Validate camera matrices.

### Flow and noise layer

Responsibilities:

- Convert projected motion into flow fields.
- Warp noise through time.
- Preserve documented tensor conventions.

### Evaluation layer

Responsibilities:

- Calculate metrics.
- Compare planners.
- Produce machine-readable and human-readable summaries.

### Dashboard layer

Responsibilities:

- Configure and launch experiments.
- Inspect artifacts and metrics.
- Avoid containing core planning or evaluation logic.

## 12.2 Common trajectory representation

All planners should return the same conceptual data:

- Time vector.
- Desired end-effector position.
- Optional desired orientation.
- Desired linear velocity.
- Optional desired angular velocity.
- Planner metadata.
- Required waypoint indices or timestamps.
- Validity and diagnostic information.

This allows the simulator and evaluator to remain planner-independent.

## 12.3 B-spline behavior

The initial implementation should use a cubic spline or cubic B-spline because it provides smooth position and velocity under appropriate knot construction.

The specification must document:

- Whether waypoints are interpolation targets or control points.
- Knot-vector construction.
- Boundary conditions.
- Endpoint behavior.
- Time parameterization.
- Derivative calculation.
- Handling of insufficient or repeated points.

Maintaining nonzero speed at intermediate waypoints should be an evaluated outcome, not an unconditional promise. Curvature, constraints, controller behavior, and time parameterization can all affect actual velocity.

## 12.4 Sequential baseline behavior

The sequential baseline should intentionally represent stop-and-go execution:

- Plan or interpolate one segment at a time.
- Reach each intermediate waypoint within tolerance.
- Apply a zero-velocity boundary condition or a short dwell.
- Continue to the next segment.

Its behavior must be clearly documented so the comparison is fair and reproducible.

## 12.5 Inverse kinematics

The IK design should define:

- End-effector body/site.
- Position and optional orientation target.
- Solver method.
- Initial joint-state strategy.
- Iteration and error limits.
- Joint-limit behavior.
- Failure behavior.

The initial release may keep end-effector orientation fixed to reduce complexity.

## 12.6 Camera projection

The projection pipeline should implement:

\[
\mathbf{p}_{world}
\rightarrow
\mathbf{p}_{camera}
\rightarrow
\mathbf{p}_{image}
\]

For a pinhole camera:

\[
\tilde{\mathbf{p}} =
\mathbf{K}
\begin{bmatrix}
\mathbf{R} & \mathbf{t}
\end{bmatrix}
\tilde{\mathbf{P}}
\]

followed by perspective division:

\[
u = \frac{x'}{z'}, \qquad v = \frac{y'}{z'}
\]

The implementation must define:

- World-frame convention.
- Camera-frame convention.
- Image origin.
- Pixel-axis directions.
- Depth sign and units.
- Intrinsic matrix.
- Extrinsic transform direction.
- Image width and height.

A projection unit test should use known synthetic points.

## 12.7 Guidance-flow generation

### Sparse guidance

For projected positions \(\mathbf{x}_t = (u_t, v_t)\), define:

\[
\Delta \mathbf{x}_t = \mathbf{x}_{t+1} - \mathbf{x}_t
\]

Store the track and displacement for every valid frame pair.

### Gaussian-local dense guidance

For pixel \(\mathbf{p}\), a simple initial field is:

\[
\mathbf{F}_t(\mathbf{p}) =
\exp\left(
-\frac{\|\mathbf{p}-\mathbf{x}_t\|^2}{2\sigma^2}
\right)
\Delta\mathbf{x}_t
\]

The configuration should expose \(\sigma\), truncation radius, and normalization behavior.

### Future mask-based guidance

A later implementation may apply motion to a rendered robot or link segmentation mask. This should not be required for the first complete release.

## 12.8 Warped-noise generation

The system should:

1. Generate seeded Gaussian noise.
2. Use the guidance field to transport or resample noise across frames.
3. Save both independent-noise and warped-noise baselines.
4. Preserve tensor shape, dtype, seed, coordinate convention, padding mode, and interpolation mode.
5. Provide visual diagnostics showing temporal correspondence.

The project must not claim that noise warping controls a video diffusion model until that claim is tested with an actual compatible model.

---

## 13. Data and Artifact Specification

## 13.1 Experiment bundle

Each run should produce an immutable or uniquely named directory similar to:

```text
experiments/
  <experiment_id>/
    config.yaml
    metadata.json
    status.json
    states/
      robot_state.npz
      desired_trajectory.npz
      executed_trajectory.npz
    frames/
      rgb/
      depth/
      segmentation/
    projection/
      desired_track.npy
      executed_track.npy
      overlays/
    flow/
      sparse_guidance.npz
      dense_guidance.npy
      observed_flow.npy
      visualizations/
    noise/
      independent_noise.npy
      warped_noise.npy
      visualizations/
    metrics/
      metrics.json
      time_series.csv
      summary.md
    media/
      execution.mp4
      comparison.mp4
```

Exact filenames may change during implementation, but the schema must remain versioned and documented.

## 13.2 Required metadata

- Experiment ID.
- Human-readable experiment name.
- Creation timestamp.
- Schema version.
- Software version or Git commit.
- Random seed.
- Simulation timestep.
- Recording frame rate.
- Planner name and parameters.
- Controller name and parameters.
- Robot model identifier.
- Scene identifier.
- Camera intrinsics and extrinsics.
- Image dimensions.
- Units and coordinate conventions.
- Completion or failure status.

## 13.3 State fields

At minimum:

- Timestamp or frame index.
- Joint position.
- Joint velocity.
- Joint command.
- Desired end-effector position.
- Actual end-effector position.
- Optional desired and actual orientation.
- Contact or collision state.
- Active trajectory segment.

## 13.4 Recommended file formats

- YAML for human-edited configuration.
- JSON for metadata and summary metrics.
- CSV for easily inspected scalar time series.
- NPZ or NPY for numeric arrays.
- PNG for lossless diagnostic images and masks.
- MP4 for demonstrations.

Large generated artifacts should normally be excluded from Git and represented by small committed samples or release assets.

---

## 14. Metrics and Evaluation Protocol

## 14.1 Robotics metrics

### Waypoint error

Euclidean distance between each required waypoint and the executed trajectory at the associated time or closest valid point.

### Tracking error

Root-mean-square and maximum distance between desired and executed end-effector positions.

### Path length

Sum of distances between consecutive executed positions.

### Execution time

Elapsed simulated time from motion start to completion.

### Planning time

Wall-clock duration required to produce the trajectory, reported separately from execution.

### Intermediate-stop metric

Count or total duration for which end-effector speed remains below a defined threshold near nonterminal waypoints.

### Smoothness

Report velocity, acceleration, and jerk using both time-series plots and scalar summaries. The exact jerk metric and numerical differentiation method must be documented.

### Feasibility

- IK failure count.
- Joint-limit violations.
- Collision count.
- Minimum obstacle distance where reliably available.

## 14.2 Visual metrics

### Projected trajectory error

Pixel distance between desired and executed end-effector projections.

### Guidance endpoint error

Endpoint error between the trajectory-derived guidance displacement and an observed or geometry-derived displacement in a defined region.

### Directional agreement

Cosine similarity between desired and observed motion vectors for valid pixels or tracked points.

### Visibility rate

Percentage of trajectory frames for which the target point is in front of the camera and inside the image.

## 14.3 Benchmark scenarios

At least three committed scenarios should be included:

1. **Open-space curve:** Multiple reachable waypoints with no nearby obstacle.
2. **Direction-change path:** Waypoints requiring a pronounced change in direction.
3. **Obstacle-adjacent path:** A valid trajectory passing near a simple obstacle.

An optional fourth scenario may intentionally fail due to unreachable waypoints or collision, demonstrating diagnostic behavior.

## 14.4 Comparison fairness

Both planners should use:

- The same initial robot state.
- The same ordered task targets.
- The same simulator settings.
- The same controller where possible.
- Comparable speed or duration constraints.
- The same metric definitions.

Any unavoidable asymmetry must be disclosed.

---

## 15. Dashboard Product Specification

## 15.1 Product principle

The dashboard is an interface to a reproducible experiment system. It must not become the only way to use the project or the location of core scientific logic.

## 15.2 Recommended information architecture

### Experiment setup

- Scenario selector.
- Planner selector.
- Waypoint editor.
- Duration and sampling controls.
- Camera preset.
- Seed.
- Advanced parameters in a collapsed section.

### Run status

- Current stage.
- Progress.
- Logs or concise diagnostics.
- Cancel option if safely supported.
- Clear failure message.

### Simulation and replay

- Execution video.
- Timeline scrubber.
- Playback speed.
- Desired and actual trajectory overlays.
- Current waypoint and state.

### Visual representations

- RGB.
- Depth.
- Segmentation.
- Desired projected path.
- Actual projected path.
- Guidance flow.
- Observed flow.
- Independent noise.
- Warped noise.

### Metrics

- Summary cards.
- Position, speed, acceleration, and jerk plots.
- Planner comparison table.
- Metric definitions.

### Export

- Configuration.
- Metrics.
- Video.
- Complete experiment bundle.

## 15.3 Usability requirements

- Provide safe default values.
- Make predefined demos available immediately.
- Separate basic and advanced controls.
- Label all units.
- Explain invalid waypoints or failed IK in plain language.
- Do not use color as the only indicator.
- Use consistent colors for desired and actual trajectories.
- Display the active coordinate frame where ambiguity is possible.
- Keep experiment results accessible after completion.

## 15.4 Recommended interaction model

For v0.1:

1. Configure an experiment.
2. Submit it as a background process.
3. Wait for completion while viewing status.
4. Load and explore the saved artifacts.

This is preferable to tightly coupling real-time MuJoCo control to browser interaction.

---

## 16. Proposed Repository Structure

```text
splineflow-panda/
  README.md
  PROJECT_SPEC.md
  LICENSE
  CITATION.cff
  CONTRIBUTING.md
  pyproject.toml
  configs/
    scenarios/
    planners/
    cameras/
  src/
    splineflow_panda/
      config/
      simulation/
      planning/
      kinematics/
      control/
      recording/
      geometry/
      flow/
      noise/
      evaluation/
      dashboard/
      cli/
  tests/
    unit/
    integration/
    regression/
  scripts/
  notebooks/
    analysis/
  examples/
  docs/
    architecture.md
    coordinate_frames.md
    dataset_schema.md
    metrics.md
    research_context.md
  assets/
    images/
    videos/
  sample_data/
```

The final layout may follow framework conventions, but scientific logic should remain independent of dashboard code.

---

## 17. GitHub Repository Framing

## 17.1 Repository tagline

> Continuous B-spline motion planning, visual-flow generation, and interactive analysis for a simulated Franka Panda.

## 17.2 README opening

The README should answer these questions above the fold:

1. What does the project do?
2. Why does continuous multi-waypoint motion matter?
3. What is trajectory-to-flow generation?
4. What can a visitor run locally?
5. What results are demonstrated?

## 17.3 Recommended README structure

1. Title and one-sentence description.
2. Demo GIF or short video.
3. Key result comparison.
4. Project motivation.
5. System diagram.
6. Features.
7. Quick start.
8. Included scenarios.
9. Metrics and representative results.
10. Dataset format.
11. Technical design.
12. Testing.
13. Limitations.
14. Research context.
15. Roadmap.
16. Citation and license.

## 17.4 Public claims

Use claims such as:

- “Compares stop-and-go waypoint execution with continuous B-spline trajectories.”
- “Projects simulated 3D robot motion into camera-space guidance fields.”
- “Exports synchronized robot state and visual observations.”
- “Provides infrastructure for future video world-model experiments.”

Avoid claims such as:

- “Solves long-horizon robotic manipulation.”
- “Replaces cuRobo.”
- “Uses a world model” before one is implemented.
- “Guarantees collision-free motion.”
- “Controls Wan2.2” before the integration is evaluated.

## 17.5 Visual assets for the repository

The repository should include:

- A 15–30 second hero demo.
- Side-by-side sequential and B-spline execution.
- A trajectory and jerk comparison plot.
- An RGB/segmentation/flow/noise visual grid.
- A system architecture diagram.
- One concise dataset example.

## 17.6 Suggested GitHub topics

`robotics`, `mujoco`, `franka-panda`, `motion-planning`, `trajectory-optimization`, `b-splines`, `computer-vision`, `optical-flow`, `world-models`, `embodied-ai`

## 17.7 Licensing

Use a permissive code license if compatible with all dependencies and model assets. Verify and document the license and attribution requirements for:

- MuJoCo.
- The Franka model.
- Any copied scene assets.
- Any optical-flow model.
- Any example datasets.

Do not assume all robot assets share the project’s code license.

---

## 18. Development Plan

## Milestone 0 — Project foundation

### Deliverables

- Repository scaffolding.
- Dependency and environment definition.
- Configuration schema.
- Basic CI.
- Documented coordinate and unit conventions.

### Exit criteria

- Clean installation succeeds on the primary development machine.
- Test command and lint command run.
- A minimal configuration can be parsed and validated.

## Milestone 1 — Franka simulation

### Deliverables

- MuJoCo scene with Franka Panda.
- Programmatic stepping.
- Fixed camera rendering.
- State and contact access.

### Exit criteria

- Robot loads in a known initial pose.
- A deterministic short simulation can be rendered.
- Joint states and end-effector pose are recorded.

## Milestone 2 — Trajectory planning and control

### Deliverables

- Sequential baseline.
- Cubic B-spline planner.
- IK and controller integration.
- Desired-versus-executed trajectory recording.

### Exit criteria

- Both planners complete the open-space benchmark.
- Required waypoints are reached within the documented tolerance.
- B-spline position and velocity continuity tests pass.
- Failures produce useful diagnostics.

## Milestone 3 — Metrics and comparison

### Deliverables

- Metric library.
- Standard plots.
- Three benchmark scenarios.
- Planner-comparison output.

### Exit criteria

- Results can be regenerated from committed configurations.
- Sequential and B-spline experiments can be compared automatically.
- Metric units and formulas are documented.

## Milestone 4 — Visual recording and projection

### Deliverables

- RGB, depth, and segmentation recording.
- Camera metadata export.
- Desired and actual 3D-to-2D projection.
- Overlay videos.

### Exit criteria

- Projection tests pass.
- Projected actual end-effector position visually aligns with the rendered robot.
- Invalid or occluded projections are handled explicitly.

## Milestone 5 — Flow and warped noise

### Deliverables

- Sparse guidance tracks.
- Gaussian-local flow fields.
- Seeded independent and warped noise.
- Flow and noise visualizations.
- Guidance-versus-observed motion metrics.

### Exit criteria

- Outputs are deterministic for a fixed seed.
- Array conventions and schema are documented.
- Flow direction is validated using synthetic tests.
- At least one benchmark produces a complete dataset bundle.

## Milestone 6 — Dashboard

### Deliverables

- Experiment configuration interface.
- Background run launch.
- Replay and layer viewer.
- Metric comparison.
- Export.

### Exit criteria

- A new user can run and inspect a predefined demo.
- Simulation failure is communicated without crashing the dashboard.
- All core capabilities remain available through the CLI.

## Milestone 7 — Public release

### Deliverables

- Polished README.
- Demo video and images.
- Installation guide.
- Research context and limitations.
- License and citation information.
- Tagged v0.1 release.

### Exit criteria

- Fresh-environment setup is tested.
- Tests pass.
- Sample artifacts are available.
- Public claims match implemented evidence.

---

## 19. Testing Strategy

## 19.1 Unit tests

- B-spline endpoint behavior.
- Waypoint ordering.
- Position and velocity continuity.
- Numerical derivative correctness on known functions.
- Camera projection of known points.
- Coordinate conversion round trips.
- Flow-vector sign and magnitude.
- Gaussian-field construction.
- Deterministic noise generation.
- Metric calculations on synthetic trajectories.
- Configuration validation.

## 19.2 Integration tests

- Planner-to-controller execution.
- Simulation-to-recorder synchronization.
- Camera metadata-to-projection pipeline.
- Projection-to-flow pipeline.
- Complete experiment bundle creation.
- CLI command success and failure.

## 19.3 Regression tests

- Fixed-seed benchmark metric ranges.
- Known projected pixel tracks.
- Dataset schema validity.
- Required artifact presence.

## 19.4 Visual verification

Automated tests cannot fully establish visual correctness. Release checks should include:

- End-effector overlay alignment.
- Depth orientation and scaling.
- Segmentation identity.
- Flow direction and color legend.
- Warped-noise temporal transport.
- Dashboard layout at common viewport sizes.

---

## 20. Acceptance Criteria for v0.1

The release is complete only when all of the following are true:

- A user can install and run the project locally from documented instructions.
- The Franka Panda completes at least three predefined multi-waypoint scenarios.
- Sequential and B-spline planners operate through a common interface.
- The system records desired and executed robot motion.
- RGB, depth, and segmentation outputs are synchronized with robot state.
- Desired and actual 3D trajectories are projected into the camera image.
- Sparse and Gaussian-local guidance flow are generated.
- Seeded warped-noise tensors are generated and documented.
- Core trajectory and visual metrics are computed automatically.
- The dashboard can configure, launch, replay, compare, and export experiments.
- The CLI can run the core pipeline independently.
- Tests cover spline continuity, projection, flow conventions, and metrics.
- A public demo and sample experiment are included.
- Known limitations and non-claims are documented.

---

## 21. Risks and Mitigations

### Risk: Project scope grows into full video diffusion

**Mitigation:** Treat diffusion inference as a post-v0.1 extension. The first release ends at validated flow and warped-noise artifacts.

### Risk: IK instability obscures planner comparison

**Mitigation:** Begin with reachable waypoints, fixed orientation, warm-started IK, and documented solver tolerances.

### Risk: B-spline does not pass through intended waypoints

**Mitigation:** Explicitly choose and test an interpolation method or clearly label control points separately from required waypoints.

### Risk: A smoother desired curve still produces jerky execution

**Mitigation:** Measure executed motion, tune control and sampling, and avoid evaluating only the mathematical reference curve.

### Risk: Camera projection is subtly incorrect

**Mitigation:** Document conventions, implement synthetic projection tests, and verify overlays visually.

### Risk: Guidance flow is confused with true optical flow

**Mitigation:** Use distinct names, schemas, colors, and documentation throughout the UI and code.

### Risk: Simulation recordings become too large for Git

**Mitigation:** Commit only small samples; use release assets or external dataset storage for larger bundles.

### Risk: Dashboard work delays robotics functionality

**Mitigation:** Build and stabilize the CLI pipeline first. Add the dashboard after artifacts and APIs are reliable.

### Risk: Metrics favor one planner unfairly

**Mitigation:** Define comparison conditions before collecting headline results and disclose duration or speed normalization.

### Risk: Repository appears to overclaim research results

**Mitigation:** Separate current implementation, experimental findings, hypotheses, and future work.

---

## 22. Future Roadmap

Future releases may add:

### v0.2 — Candidate spline planning

- Sample B-spline control-point variations.
- Score goal progress, smoothness, collision, and clearance.
- Execute in a receding-horizon loop.

### v0.3 — Rich visual correspondence

- Link-level segmentation flow.
- Dense observed optical flow.
- Occlusion-aware motion fields.
- Multiple camera views.

### v0.4 — Learned outcome prediction

- Train a lightweight model to predict success, collision, clearance, or final state from short simulated videos.
- Use predicted outcomes to rank trajectory candidates.

### v0.5 — Video diffusion experiment

- Test whether trajectory-derived warped noise measurably improves generated motion adherence in a manageable video model.
- Compare independent noise, generic temporal noise, and trajectory-warped noise.

### Later research

- Wan2.2 adaptation.
- Video-world-model scoring inside MPC.
- Robot-factorized representations.
- Sim-to-real calibration.
- Physical Franka execution with an independent safety layer.

---

## 23. Research Connection

SplineFlow-Panda is motivated by three complementary ideas:

1. Continuous action representations can avoid artificial segmentation of multi-stage robot motion.
2. Motion-controllable video generation can use temporally structured or warped noise as a guidance mechanism.
3. Robot-aware visual world models can separate predictable robot motion from uncertain environmental response.

The v0.1 system connects these ideas without requiring that all of them be solved simultaneously:

```text
3D waypoints
    |
    v
Continuous B-spline action
    |
    v
Simulated Franka execution
    |
    v
Camera-space trajectory and motion guidance
    |
    v
Flow and warped-noise research artifacts
    |
    v
Future video world-model conditioning and evaluation
```

This framing makes the project scientifically relevant while keeping its implemented claims narrow and verifiable.

---

## 24. Definition of Success

### Technical success

The system reliably executes, records, projects, analyzes, and exports continuous Franka trajectories.

### Educational success

The developer can clearly explain every transformation from:

```text
waypoints
→ spline
→ task-space samples
→ joint commands
→ simulated motion
→ camera pixels
→ flow field
→ warped noise
```

### Research success

The project produces a validated dataset and experimental framework that can support a precise next question about video-based robot prediction or control.

### Portfolio success

A GitHub visitor can understand the project in under two minutes, run a representative demo with reasonable effort, inspect quantitative results, and recognize meaningful robotics and computer-vision engineering.

---

## 25. Suggested Résumé Framing

> Developed SplineFlow-Panda, a MuJoCo-based research platform for continuous multi-waypoint control of a 7-DoF Franka Panda. Implemented B-spline trajectory generation, inverse-kinematics tracking, synchronized RGB-D and robot-state recording, camera-space motion projection, visual-flow guidance, warped-noise generation, quantitative planner benchmarks, and an interactive experiment dashboard.

Quantitative results should be added only after the benchmark suite is complete. Example:

> Reduced intermediate stopping time by **X%** and measured **Y%** lower jerk than a sequential waypoint baseline across **N** reproducible simulated tasks.

Never insert placeholder values into the public résumé or README.

---

## 26. Guiding Principles

1. **Build the robotics foundation before adding generative AI.**
2. **Measure executed behavior, not only planned behavior.**
3. **Keep desired guidance distinct from observed motion.**
4. **Prefer reproducible experiments over one-off demonstrations.**
5. **Make coordinate frames, units, and tensor conventions explicit.**
6. **Keep scientific logic independent of the dashboard.**
7. **Treat failures as useful experiment outputs.**
8. **Make the first release small enough to finish and strong enough to extend.**
9. **Use precise public claims supported by visible evidence.**
10. **Optimize for understanding, inspectability, and credible engineering.**


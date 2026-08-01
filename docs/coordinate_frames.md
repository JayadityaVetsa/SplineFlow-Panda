# Coordinate and tensor conventions

- Physical quantities use SI units.
- MuJoCo world coordinates are preserved in recorded states.
- Camera projection accepts a 4×4 world-to-camera transform using an OpenCV-style
  camera frame: `+x` right, `+y` down, `+z` forward.
- Images and fields are indexed `[row, column]`, while points are `(u, v)`.
- Flow fields have shape `(T-1, H, W, 2)` and store `(du, dv)` in pixels/frame.
- Trajectories have positions, velocities, and accelerations shaped `(T, 3)`.
- Noise has shape `(T, H, W, C)` and `float32` dtype.


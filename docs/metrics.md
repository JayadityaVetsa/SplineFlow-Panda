# Metric definitions

- Tracking RMSE/max: Euclidean desired-to-executed position error in metres.
- Path length: sum of executed point-to-point distances.
- Speed max: maximum norm of the timestamp-aware numerical velocity.
- Acceleration/jerk RMS: RMS vector norm after excluding derivative boundary samples.
- Stopped duration: recorded time with speed below the configured threshold.
- Projected tracking RMSE: pixel distance between valid desired and executed tracks.
- Guidance EPE: endpoint error between desired and geometry-derived executed displacement.
- Directional agreement: cosine similarity of valid displacement vectors.
- Visibility rate: fraction of points in front of the camera and inside the image.


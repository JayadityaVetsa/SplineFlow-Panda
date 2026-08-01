from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from splineflow_panda.artifacts import ExperimentBundle
from splineflow_panda.config import load_config
from splineflow_panda.models import PlannerKind, Trajectory
from splineflow_panda.planning import plan
from splineflow_panda.visualization import depth_to_grayscale, write_rgb_video

ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ROOT / "configs" / "scenarios"
EXPERIMENTS = ROOT / "experiments"
BENCHMARKS = ROOT / "benchmarks"
PAPER_URL = "https://arxiv.org/pdf/2607.09648"

st.set_page_config(
    page_title="SplineFlow-Panda research lab",
    page_icon=":material/precision_manufacturing:",
    layout="wide",
)


@st.cache_data(max_entries=12)
def load_trajectory(config_path: str, kind: str, duration: float, rate: float) -> Trajectory:
    config = load_config(Path(config_path), ROOT / "configs" / "defaults.yaml")
    config.planner.kind = PlannerKind(kind)
    config.planner.duration = duration
    config.planner.sample_rate = rate
    return plan(config)


@st.cache_data(max_entries=24)
def load_npz_trajectory(path: str) -> dict[str, np.ndarray]:
    trajectory = Trajectory.load(Path(path))
    return {
        "time": trajectory.time,
        "position": trajectory.position,
        "velocity": trajectory.velocity,
        "acceleration": trajectory.acceleration,
        "waypoint_times": trajectory.waypoint_times,
    }


@st.cache_data(max_entries=8)
def load_array(path: str) -> np.ndarray:
    return np.load(path, mmap_mode="r")


@st.cache_data(max_entries=16)
def ensure_video(experiment_path: str, fps: float) -> str:
    experiment = Path(experiment_path)
    video_path = experiment / "media" / "execution.mp4"
    if not video_path.exists():
        frames = np.load(experiment / "frames" / "rgb.npy", mmap_mode="r")
        write_rgb_video(video_path, frames, fps)
    return str(video_path)


def vector_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def numeric_or_nan(value: float | int | None) -> float:
    return np.nan if value is None else float(value)


def kinematic_frame(trajectory: Trajectory, label: str) -> pd.DataFrame:
    velocity = vector_norm(trajectory.velocity)
    acceleration = vector_norm(trajectory.acceleration)
    jerk_vector = np.gradient(trajectory.acceleration, trajectory.time, axis=0)
    return pd.DataFrame(
        {
            "time (s)": trajectory.time,
            "speed (m/s)": velocity,
            "acceleration (m/s²)": acceleration,
            "jerk (m/s³)": vector_norm(jerk_vector),
            "planner": label,
        }
    )


def comparison_summary(sequential: Trajectory, spline: Trajectory) -> pd.DataFrame:
    rows = []
    for label, trajectory in (("Sequential", sequential), ("B-spline", spline)):
        speed = vector_norm(trajectory.velocity)
        jerk = vector_norm(np.gradient(trajectory.acceleration, trajectory.time, axis=0))
        rows.append(
            {
                "Planner": label,
                "Total time (s)": trajectory.time[-1],
                "Path length (m)": np.linalg.norm(
                    np.diff(trajectory.position, axis=0), axis=1
                ).sum(),
                "Peak speed (m/s)": speed.max(),
                "RMS jerk (m/s³)": np.sqrt(np.mean(jerk**2)),
                "Near-stop samples": int(np.sum(speed < 0.01)),
            }
        )
    return pd.DataFrame(rows)


def run_simulation_pair(config_path: Path) -> list[Path]:
    outputs: list[Path] = []
    for kind in ("action_chunk", "bspline_action"):
        config = load_config(config_path, ROOT / "configs" / "defaults.yaml")
        config.planner.kind = PlannerKind(kind)
        temporary = ROOT / "configs" / f".dashboard-{kind}.yaml"
        temporary.write_text(
            __import__("yaml").safe_dump(config.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "splineflow_panda.cli",
                    "run",
                    str(temporary),
                    "--output",
                    str(EXPERIMENTS),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=240,
            )
        finally:
            temporary.unlink(missing_ok=True)
        if process.returncode:
            raise RuntimeError(process.stderr or process.stdout)
        outputs.append(Path(process.stdout.strip().splitlines()[-1]))
    return outputs


def concise_simulation_error(error: Exception) -> str:
    text = str(error)
    match = re.search(
        r"IK failed: position=([0-9.]+) m, orientation=([0-9.]+) deg",
        text,
    )
    if match:
        return (
            f"IK rejected the target pose: {float(match.group(1)) * 1000:.1f} mm "
            f"position residual and {float(match.group(2)):.3f}° orientation residual."
        )
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else type(error).__name__


def metric_card(label: str, value: float, unit: str, help_text: str) -> None:
    st.metric(label, f"{value:.3g} {unit}".strip(), border=True, help=help_text)


def render_header() -> None:
    st.title("SplineFlow-Panda research lab")
    st.markdown(
        "Evaluate **B-spline trajectories as structured robot actions**: plan continuous "
        "motion, execute it on a simulated Franka Panda, and measure controller-compatible "
        "behavior under reproducible conditions."
    )
    st.caption("Simulation study · measured MuJoCo rollouts · no physical-robot claims")


def render_overview() -> None:
    st.header("What is this project?")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown(
            """
            A Franka Panda receives the same task under two action representations:

            1. **Action chunks** — discrete commands updated at the policy rate.
            2. **B-spline actions** — a continuous cubic curve sampled by the controller.

            Both execute progressively faster. A shorter configured duration only counts
            when tracking, collision, and task-success criteria still pass.
            """
        )
        st.info(
            "**Primary research question:** How much faster can continuous B-spline "
            "actions execute before controller tracking or task success breaks down?",
            icon=":material/science:",
        )
    with right:
        st.mermaid_chart(
            """
            flowchart TD
              A["Same task and seed"] --> B["Chunk / B-spline action"]
              B --> C["Temporal scaling"]
              C --> D["100 Hz joint controller"]
              D --> E["MuJoCo execution"]
              E --> F["Success + completion time"]
              F --> G["Speed-success frontier"]
            """
        )
    st.subheader("Project summary")
    st.markdown(
        """
        SplineFlow-Panda is a reproduction-inspired study of B-spline action
        representations. Its takeaway is the measured maximum reliable speedup under a
        fixed success threshold—not an automatically smoother plot or artificial dwell.
        """
    )


def render_results_page() -> None:
    st.header("Results")
    st.subheader("Speed-success frontier")
    st.markdown(
        "The research result is the fastest execution that still satisfies the same "
        "tracking, collision, and task-success criteria."
    )
    reports = sorted(BENCHMARKS.glob("*/report.json")) if BENCHMARKS.exists() else []
    if not reports:
        st.info(
            "No report exists yet. Run `splineflow benchmark "
            "configs/scenarios/direction_change.yaml --seeds 10`.",
            icon=":material/speed:",
        )
        return
    report_path = st.selectbox(
        "Benchmark report", reports, format_func=lambda path: path.parent.name
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rollout_path = report_path.with_name("rollouts.json")
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8")) if rollout_path.exists() else []
    source_paths = [Path(str(row["bundle"])) for row in rollouts if row.get("bundle")]
    try:
        from splineflow_panda.reporting import validate_benchmark_report

        validation = validate_benchmark_report(report_path.parent)
    except ValueError as error:
        validation = None
        st.error(f"Report validation failed: {error}", icon=":material/error:")
    if rollouts:
        verified = sum(path.exists() for path in source_paths)
        if validation:
            st.success(
                f"Measured scripted MuJoCo benchmark: {len(rollouts)} rollouts; "
                f"{verified}/{len(source_paths)} source bundles present; aggregates "
                "recomputed from the ledger.",
                icon=":material/verified:",
            )
    else:
        st.warning(
            "This report has no rollout ledger, so it is not accepted as a reproducible "
            "milestone benchmark.",
            icon=":material/warning:",
        )
    rows = []
    for representation in ("action_chunk", "bspline_action"):
        values = report.get(representation, {}).get("by_speedup", {})
        for speed, result in values.items():
            rows.append(
                {
                    "Representation": representation.replace("_", " "),
                    "Target speedup": float(speed),
                    "Success rate": result["success_rate"],
                    "Mean completion time (s)": result["completion_time_mean_s"],
                    "Achieved speedup": result.get("achieved_speedup"),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame):
        enough_for_frontier = frame.groupby("Representation").size().min() >= 2
        if enough_for_frontier:
            st.line_chart(frame, x="Target speedup", y="Success rate", color="Representation")
        else:
            st.info(
                "A frontier chart requires at least two measured speeds per representation. "
                "This report is shown as tables only.",
                icon=":material/table_chart:",
            )
        st.dataframe(frame, hide_index=True, width="stretch")
        if rollouts:
            detail_rows = []
            for row in rollouts:
                bundle_path = Path(str(row.get("bundle", "")))
                metrics_path = bundle_path / "metrics" / "metrics.json"
                bundle_metrics = (
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                    if metrics_path.exists()
                    else {}
                )
                failure_reasons = []
                if not row.get("success"):
                    if bundle_metrics.get("waypoint_error_max_m", 0) > 0.02:
                        failure_reasons.append("waypoint error exceeded 2 cm")
                    if row.get("tracking_max_m", 0) > 0.05:
                        failure_reasons.append("tracking error exceeded 5 cm")
                    if bundle_metrics.get("forbidden_contacts", 0):
                        failure_reasons.append("forbidden contact")
                    if not failure_reasons:
                        failure_reasons.append("task success criteria were not sustained")
                detail_rows.append(
                    {
                        "Representation": str(row["representation"]).replace("_", " "),
                        "Seed": row.get("task_seed"),
                        "Speedup": row.get("speedup"),
                        "Success": row.get("success"),
                        "Completion (s)": row.get("completion_time_s"),
                        "Tracking RMSE (cm)": 100 * row.get("tracking_rmse_m", np.nan),
                        "Max error (cm)": 100 * row.get("tracking_max_m", np.nan),
                        "Waypoint error (cm)": 100
                        * bundle_metrics.get("waypoint_error_max_m", np.nan),
                        "Peak speed (m/s)": row.get("speed_max_m_s"),
                        "RMS acceleration (m/s²)": row.get("acceleration_rms_m_s2"),
                        "RMS jerk (m/s³)": row.get("jerk_rms_m_s3"),
                        "Path length (m)": row.get("path_length_m"),
                        "Saturation": row.get("actuator_saturation_rate"),
                        "Min clearance (cm)": 100
                        * numeric_or_nan(row.get("minimum_clearance_m")),
                        "Forbidden contacts": row.get("forbidden_contacts"),
                        "Outcome": "success" if row.get("success") else "; ".join(failure_reasons),
                    }
                )
            st.subheader("Measured rollout details")
            st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")
            chunk_rows = [
                row
                for row in rollouts
                if row["representation"] == "action_chunk" and row["speedup"] == 1.0
            ]
            spline_rows = [
                row
                for row in rollouts
                if row["representation"] == "bspline_action" and row["speedup"] == 1.0
            ]
            if chunk_rows and spline_rows:

                def percent_change(metric: str) -> float:
                    baseline_value = float(np.mean([row[metric] for row in chunk_rows]))
                    spline_value = float(np.mean([row[metric] for row in spline_rows]))
                    return 100 * (spline_value - baseline_value) / baseline_value

                st.subheader("Paired-seed 1x comparison")
                with st.container(horizontal=True):
                    st.metric(
                        "Tracking RMSE change",
                        f"{percent_change('tracking_rmse_m'):+.1f}%",
                        help="Negative means lower executed Cartesian tracking error.",
                        border=True,
                    )
                    st.metric(
                        "Acceleration change",
                        f"{percent_change('acceleration_rms_m_s2'):+.1f}%",
                        help="Negative means lower RMS executed acceleration.",
                        border=True,
                    )
                    st.metric(
                        "Jerk change",
                        f"{percent_change('jerk_rms_m_s3'):+.1f}%",
                        help="Negative means smoother executed motion.",
                        border=True,
                    )
                    st.metric(
                        "Completion-time change",
                        f"{percent_change('completion_time_s'):+.1f}%",
                        help="Negative means the successful rollout completed sooner.",
                        border=True,
                    )
                st.write(
                    "At 1x across the paired task seeds, the B-spline action had lower tracking "
                    "error, acceleration, jerk, and actuator saturation. This supports a "
                    "smoothness/control-load observation, but it does not yet prove a "
                    "higher reliable speed frontier."
                )
        baseline = report.get("baseline_completion_time_s")
        chunk_frontier = report.get("action_chunk", {}).get("maximum_reliable_speedup")
        spline_frontier = report.get("bspline_action", {}).get("maximum_reliable_speedup")
        cols = st.columns(3)
        cols[0].metric("1x chunk baseline", f"{baseline:.2f} s" if baseline else "Not measured")
        cols[1].metric(
            "Chunk reliable frontier",
            f"{chunk_frontier:g}x" if chunk_frontier else "Not established",
        )
        cols[2].metric(
            "B-spline reliable frontier",
            f"{spline_frontier:g}x" if spline_frontier else "Not established",
        )
        st.subheader("Plain-language interpretation")
        if baseline and spline_frontier and chunk_frontier:
            relation = "higher" if spline_frontier > chunk_frontier else "not higher"
            st.write(
                f"At the report's success threshold, the measured B-spline frontier is "
                f"**{relation}** than the action-chunk frontier ({spline_frontier:g}x "
                f"versus {chunk_frontier:g}x). These values summarize saved simulator "
                "rollouts; configured target speeds are never presented as achieved results."
            )
        else:
            st.write(
                "This pilot is insufficient to establish a frontier. A defensible result "
                "needs paired task seeds at multiple speedups."
            )
        st.subheader("Definitions and formulas")
        st.latex(
            r"\mathrm{success\ rate}="
            r"\frac{\#\ \mathrm{successful\ rollouts}}{\#\ \mathrm{all\ paired\ rollouts}}"
        )
        st.latex(
            r"\mathrm{achieved\ speedup}="
            r"\frac{\overline{T}_{\mathrm{chunk},1\times}}{\overline{T}_{\mathrm{method},s}}"
        )
        st.latex(
            r"\mathrm{RMSE}=\sqrt{\frac{1}{N}\sum_{i=1}^{N}"
            r"\lVert x_i^{\mathrm{executed}}-x_i^{\mathrm{commanded}}\rVert_2^2}"
        )
        st.latex(
            r"\mathrm{RMS\ jerk}=\sqrt{\frac{1}{M}\sum_{i=1}^{M}"
            r"\lVert d^3x_i/dt^3\rVert_2^2}"
        )
        st.caption(
            "Timing starts after settling and ends at sustained success. Failures count "
            "against success rate and are excluded from mean completion time."
        )
    if report.get("warning"):
        st.warning(report["warning"], icon=":material/warning:")


def render_trajectory_lab(config_path: Path, duration: float, rate: float) -> None:
    st.header("Trajectory lab")
    st.caption("Change time and sampling in the sidebar. The curves update without running MuJoCo.")
    sequential = load_trajectory(str(config_path), "sequential", duration, rate)
    spline = load_trajectory(str(config_path), "bspline", duration, rate)
    config = load_config(config_path, ROOT / "configs" / "defaults.yaml")
    points = np.asarray(config.waypoints)

    st.subheader("Same task, different motion assumptions")
    left, right = st.columns(2, gap="large")
    with left.container(border=True):
        st.markdown("**Sequential: stop at every intermediate waypoint**")
        st.line_chart(
            pd.DataFrame(
                {
                    "x": sequential.position[:, 0],
                    "y": sequential.position[:, 1],
                    "z": sequential.position[:, 2],
                },
                index=sequential.time,
            ),
            x_label="Time (s)",
            y_label="Position (m)",
        )
        st.caption("Each segment uses a minimum-jerk polynomial; dwell samples hold position.")
    with right.container(border=True):
        st.markdown("**B-spline: one interpolating curve through all waypoints**")
        st.line_chart(
            pd.DataFrame(
                {
                    "x": spline.position[:, 0],
                    "y": spline.position[:, 1],
                    "z": spline.position[:, 2],
                },
                index=spline.time,
            ),
            x_label="Time (s)",
            y_label="Position (m)",
        )
        st.caption("Chord length assigns waypoint times; a cubic basis blends nearby points.")

    st.subheader("The smoothness comparison")
    metric = st.segmented_control(
        "Quantity",
        ["Speed", "Acceleration", "Jerk"],
        default="Speed",
        key="kinematic_quantity",
    )
    frames = pd.concat(
        [kinematic_frame(sequential, "Sequential"), kinematic_frame(spline, "B-spline")]
    )
    column = {
        "Speed": "speed (m/s)",
        "Acceleration": "acceleration (m/s²)",
        "Jerk": "jerk (m/s³)",
    }[metric]
    st.line_chart(
        frames,
        x="time (s)",
        y=column,
        color="planner",
        x_label="Time (s)",
        y_label=column,
    )
    st.caption(
        "Sharp jerk spikes mean acceleration changes abruptly. The comparison must ultimately "
        "be repeated on executed motion because the controller and contacts can add jerk."
    )
    st.dataframe(
        comparison_summary(sequential, spline),
        hide_index=True,
        width="stretch",
        column_config={
            "Total time (s)": st.column_config.NumberColumn(format="%.3f"),
            "Path length (m)": st.column_config.NumberColumn(format="%.4f"),
            "Peak speed (m/s)": st.column_config.NumberColumn(format="%.4f"),
            "RMS jerk (m/s³)": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    with st.expander("Learn the B-spline mathematics", icon=":material/function:"):
        st.latex(r"\mathbf{p}(u)=\sum_{i=0}^{n}N_{i,3}(u)\mathbf{c}_i")
        st.markdown(
            """
            `u` is a curve parameter, `cᵢ` are control coefficients, and `Nᵢ,₃` are
            degree-three basis functions. Only a few basis functions are nonzero at any
            one `u`, so changing one coefficient has local—not global—effect.

            This project uses an **interpolating representation**: SciPy solves for control
            coefficients so the resulting curve passes through the user waypoints. The
            visible waypoints are therefore not naively treated as the B-spline control
            polygon.
            """
        )
        st.markdown(
            f"Waypoint count: **{len(points)}** · spline degree: **{spline.metadata['degree']}** "
            f"· trajectory samples: **{len(spline.time)}**"
        )

    with st.expander("Learn time, duration, and sampling", icon=":material/schedule:"):
        st.markdown(
            f"""
            - **Duration ({duration:.1f} s)** changes how quickly the same spatial path is
              traversed.
            - **Sample rate ({rate:.0f} Hz)** creates approximately
              `{len(spline.time)}` target samples.
            - MuJoCo's physics step is smaller than the target interval; the controller holds each
              target while physics advances.
            - Camera FPS can be lower still. State and frames remain linked by
              timestamps/frame indices.
            """
        )


def experiment_directories() -> list[Path]:
    if not EXPERIMENTS.exists():
        return []
    paths = [path for path in EXPERIMENTS.iterdir() if (path / "status.json").exists()]

    def display_order(path: Path) -> tuple[bool, float]:
        status = json.loads((path / "status.json").read_text(encoding="utf-8"))
        return status.get("stage") == "completed", path.stat().st_mtime

    return sorted(paths, key=display_order, reverse=True)


def latest_comparison_pair(scenario_name: str) -> list[Path]:
    found: dict[PlannerKind, Path] = {}
    for path in experiment_directories():
        try:
            config = ExperimentBundle(path).load_config()
        except (OSError, ValueError):
            continue
        if config.name != scenario_name or config.planner.kind in found:
            continue
        executed = path / "states" / "executed_trajectory.npz"
        if executed.exists():
            found[config.planner.kind] = path
        if len(found) == 2:
            break
    if PlannerKind.ACTION_CHUNK in found and PlannerKind.BSPLINE_ACTION in found:
        return [found[PlannerKind.ACTION_CHUNK], found[PlannerKind.BSPLINE_ACTION]]
    return []


def render_recorded_experiment(experiment: Path) -> None:
    status = json.loads((experiment / "status.json").read_text(encoding="utf-8"))
    metadata = json.loads((experiment / "metadata.json").read_text(encoding="utf-8"))
    config = ExperimentBundle(experiment).load_config()
    with st.container(horizontal=True):
        st.badge(status["stage"], color="green" if status["stage"] == "completed" else "orange")
        st.caption(
            f"{config.name} · {config.planner.kind.value} · task seed {config.task_seed}"
        )
    if metadata.get("schema_version") != "0.3.0":
        st.warning(
            "Legacy experiment schema. It is not comparable with schema 0.3 results.",
            icon=":material/history:",
        )

    executed_state = experiment / "states" / "executed_trajectory.npz"
    if status["stage"] == "failed":
        if config.name == "intentionally-unreachable":
            st.success(
                "Expected-failure test passed: IK rejected the unreachable waypoint.",
                icon=":material/check_circle:",
            )
            st.caption(concise_simulation_error(RuntimeError(status["message"])))
        else:
            st.error(
                "Unexpected simulation failure. "
                + concise_simulation_error(RuntimeError(status["message"])),
                icon=":material/error:",
            )
        st.caption("No execution metrics are reported because no rollout was executed.")
        with st.expander("Failure record and reproducibility metadata"):
            st.json(status)
            st.json(metadata)
        return

    metrics_path = experiment / "metrics" / "metrics.json"
    if metrics_path.exists() and executed_state.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        cols = st.columns(4)
        with cols[0]:
            metric_card(
                "Tracking RMSE",
                metrics.get("tracking_rmse_m", np.nan) * 100,
                "cm",
                "Typical 3D distance between desired and executed end-effector positions.",
            )
        with cols[1]:
            metric_card(
                "Maximum error",
                metrics.get("tracking_max_m", np.nan) * 100,
                "cm",
                "Worst desired-versus-executed position error.",
            )
        with cols[2]:
            metric_card(
                "Executed path",
                metrics.get("path_length_m", np.nan),
                "m",
                "Distance traveled by the measured end effector.",
            )
        with cols[3]:
            metric_card(
                "Minimum clearance",
                metrics.get("minimum_clearance_m", np.nan) * 100,
                "cm",
                "Smallest distance between Panda collision geometry and an obstacle.",
            )

    desired_path = experiment / "states" / "commanded_trajectory.npz"
    if not desired_path.exists():
        desired_path = experiment / "states" / "desired_trajectory.npz"
    actual_path = experiment / "states" / "executed_trajectory.npz"
    if desired_path.exists() and actual_path.exists():
        desired = load_npz_trajectory(str(desired_path))
        actual = load_npz_trajectory(str(actual_path))
        frame = pd.DataFrame(
            {
                "desired x": desired["position"][:, 0],
                "actual x": actual["position"][:, 0],
                "desired y": desired["position"][:, 1],
                "actual y": actual["position"][:, 1],
                "desired z": desired["position"][:, 2],
                "actual z": actual["position"][:, 2],
            },
            index=desired["time"],
        )
        st.line_chart(frame, x_label="Time (s)", y_label="End-effector position (m)")
        st.caption(
            "A visible gap is controller tracking error—not a plotting error. The desired "
            "curve and simulated robot are intentionally recorded separately."
        )

    frame_files = {
        name: experiment / "frames" / f"{name}.npy"
        for name in ("rgb", "depth", "segmentation")
    }
    if all(path.exists() for path in frame_files.values()):
        rgb = load_array(str(frame_files["rgb"]))
        depth = load_array(str(frame_files["depth"]))
        segmentation = load_array(str(frame_files["segmentation"]))
        video_path = ensure_video(str(experiment), config.camera.fps)
        st.subheader("Execution recording")
        st.video(video_path, loop=True, muted=True)
        st.caption(
            f"Fixed overview camera · {config.camera.fps:.0f} FPS · "
            f"{len(rgb) / config.camera.fps:.1f} seconds"
        )
        st.subheader("Frame-level diagnostics")
        index = st.slider("Recorded frame", 0, len(rgb) - 1, 0, key=f"frame-{experiment.name}")
        cols = st.columns(3)
        cols[0].image(rgb[index], caption="RGB", width="stretch")
        depth_image, near_depth, far_depth = depth_to_grayscale(depth[index])
        cols[1].image(
            depth_image,
            caption=f"Depth: {near_depth:.2f}–{far_depth:.2f} m",
            width="stretch",
        )
        segment_image = segmentation[index]
        if segment_image.ndim == 3:
            segment_image = segment_image[..., 0]
        cols[2].image(segment_image, caption="Segmentation ID", clamp=True, width="stretch")
        st.caption(
            "Depth stores camera-to-surface distance in metres. Brighter pixels are nearer; "
            "the display excludes the far-plane background so scene geometry remains visible. "
            "Segmentation stores simulator object IDs, not RGB colors."
        )

    with st.expander("Raw status and reproducibility metadata", icon=":material/data_object:"):
        st.json(status)
        st.json(metadata)


def render_executed_comparison(run_paths: list[Path]) -> None:
    if len(run_paths) != 2 or not all(path.exists() for path in run_paths):
        return
    records = []
    trajectories: dict[str, dict[str, np.ndarray]] = {}
    for path in run_paths:
        config = ExperimentBundle(path).load_config()
        label = (
            "Action chunk"
            if config.planner.kind == PlannerKind.ACTION_CHUNK
            else "B-spline action"
        )
        metrics = json.loads((path / "metrics" / "metrics.json").read_text(encoding="utf-8"))
        records.append(
            {
                "Planner": label,
                "Tracking RMSE (cm)": 100 * metrics["tracking_rmse_m"],
                "Path length (m)": metrics["path_length_m"],
                "RMS jerk (m/s³)": metrics["jerk_rms_m_s3"],
                "Saturation rate": metrics.get("actuator_saturation_rate", np.nan),
                "Minimum clearance (cm)": 100 * metrics.get("minimum_clearance_m", np.nan),
                "Completion time (s)": metrics.get(
                    "completion_time_s", metrics["execution_time_s"]
                ),
            }
        )
        trajectories[label] = load_npz_trajectory(
            str(path / "states" / "executed_trajectory.npz")
        )
    frame = pd.DataFrame(records).set_index("Planner")
    st.subheader("Executed comparison")
    st.dataframe(
        frame.reset_index(),
        hide_index=True,
        width="stretch",
        column_config={
            name: st.column_config.NumberColumn(format="%.3f")
            for name in frame.columns
        },
    )
    sequential = frame.loc["Action chunk"]
    spline = frame.loc["B-spline action"]
    cols = st.columns(3)
    jerk_change = 100 * (
        spline["RMS jerk (m/s³)"] - sequential["RMS jerk (m/s³)"]
    ) / max(abs(sequential["RMS jerk (m/s³)"]), 1e-12)
    saturation_change = spline["Saturation rate"] - sequential["Saturation rate"]
    tracking_change = spline["Tracking RMSE (cm)"] - sequential["Tracking RMSE (cm)"]
    cols[0].metric(
        "B-spline jerk change",
        f"{jerk_change:+.1f}%",
        border=True,
        help="Negative means the executed B-spline was smoother.",
    )
    cols[1].metric(
        "B-spline saturation change",
        f"{saturation_change:+.1%}",
        border=True,
        help="Negative means fewer controller samples reached an actuator force limit.",
    )
    cols[2].metric(
        "B-spline tracking-error change",
        f"{tracking_change:+.2f} cm",
        border=True,
        help="Negative means more accurate execution.",
    )
    motion_frames = []
    for label, values in trajectories.items():
        velocity = np.gradient(values["position"], values["time"], axis=0)
        acceleration = np.gradient(velocity, values["time"], axis=0)
        motion_frames.append(
            pd.DataFrame(
                {
                    "time (s)": values["time"],
                    "speed (m/s)": vector_norm(velocity),
                    "acceleration (m/s²)": vector_norm(acceleration),
                    "planner": label,
                }
            )
        )
    motion = pd.concat(motion_frames)
    quantity = st.segmented_control(
        "Executed quantity",
        ["Speed", "Acceleration"],
        default="Speed",
        key=f"executed_quantity-{run_paths[0].name}-{run_paths[1].name}",
    )
    value_column = "speed (m/s)" if quantity == "Speed" else "acceleration (m/s²)"
    st.line_chart(
        motion,
        x="time (s)",
        y=value_column,
        color="planner",
        x_label="Simulated time (s)",
        y_label=value_column,
    )
    st.caption(
        "These curves come from measured MuJoCo end-effector positions. They are the "
        "scientifically relevant comparison—not the ideal planner curves above."
    )
    rgb_paths = {
        "Action chunk": run_paths[0] / "frames" / "rgb.npy",
        "B-spline action": run_paths[1] / "frames" / "rgb.npy",
    }
    if all(path.exists() for path in rgb_paths.values()):
        rgb_frames = {label: load_array(str(path)) for label, path in rgb_paths.items()}
        frame_count = min(len(values) for values in rgb_frames.values())
        frame_index = st.slider(
            "Side-by-side camera frame",
            0,
            frame_count - 1,
            0,
            key=f"comparison_camera_frame-{run_paths[0].name}-{run_paths[1].name}",
        )
        left, right = st.columns(2, gap="large")
        left.image(
            rgb_frames["Action chunk"][frame_index],
            caption="Action-chunk execution",
            width="stretch",
        )
        right.image(
            rgb_frames["B-spline action"][frame_index],
            caption="B-spline-action execution",
            width="stretch",
        )
        st.caption(
            "Both panels use the same fixed camera and scenario. Scrub the frame to compare "
            "discrete policy updates against continuous execution."
        )


def render_simulation_lab(config_path: Path) -> None:
    config = load_config(config_path, ROOT / "configs" / "defaults.yaml")
    scenario_name = config.name
    expected_failure = scenario_name == "intentionally-unreachable"
    st.header("MuJoCo experiment lab")
    st.markdown(
        "MuJoCo execution adds IK, actuator dynamics, contacts, and tracking error. "
        "Compare discrete chunks and continuous B-spline actions under the same conditions."
    )
    if expected_failure:
        st.info(
            "This scenario is an expected-failure test. Its middle waypoint is outside the "
            "Panda workspace; passing means IK rejects it cleanly without moving the robot.",
            icon=":material/science:",
        )
    button_label = (
        "Run expected IK failure test"
        if expected_failure
        else "Run action-chunk and B-spline simulations"
    )
    if st.button(
        button_label,
        type="primary",
        icon=":material/play_arrow:",
    ):
        if expected_failure:
            try:
                run_simulation_pair(config_path)
                st.warning("Unexpected result: the unreachable target was accepted.")
            except Exception as error:
                st.success(
                    "Expected failure detected. " + concise_simulation_error(error),
                    icon=":material/check_circle:",
                )
        else:
            try:
                with st.status("Running fair two-planner comparison…", expanded=True) as status:
                    st.write("Action-chunk experiment: 10 Hz updates with zero-order hold")
                    st.write("B-spline experiment: continuous commands sampled at 100 Hz")
                    outputs = run_simulation_pair(config_path)
                    status.update(label="Both experiments completed", state="complete")
                runs_by_scenario = st.session_state.setdefault(
                    "comparison_runs_by_scenario", {}
                )
                runs_by_scenario[scenario_name] = [str(path) for path in outputs]
                st.toast("Comparison saved", icon=":material/check_circle:")
            except Exception as error:
                st.error(concise_simulation_error(error), icon=":material/error:")

    if not expected_failure:
        runs_by_scenario = st.session_state.get("comparison_runs_by_scenario", {})
        comparison_runs = [Path(path) for path in runs_by_scenario.get(scenario_name, [])]
        if not comparison_runs:
            comparison_runs = latest_comparison_pair(scenario_name)
        if comparison_runs:
            render_executed_comparison(comparison_runs)

    runs = [
        path
        for path in experiment_directories()
        if ExperimentBundle(path).load_config().name == scenario_name
    ]
    if not runs:
        st.info("No recorded experiments exist for this scenario yet.")
        return
    selected = st.selectbox(
        "Inspect a recorded experiment",
        runs,
        format_func=lambda path: path.name,
        key=f"recorded-experiment-{scenario_name}",
    )
    render_recorded_experiment(selected)


def render_methods_page() -> None:
    st.header("Methods, scope, and related work")
    st.subheader("Research hypothesis")
    st.markdown(
        """
        Discrete action chunks can become sparse or discontinuous when a manipulation policy
        is accelerated. A cubic B-spline represents the same action as a continuous curve,
        allowing 10 Hz policy predictions to be decoded into 100 Hz controller commands.

        We test whether this increases the maximum speed that preserves task success—not
        whether a spline merely looks smoother at a fixed duration.
        """
    )
    st.subheader("Controlled variables")
    st.markdown(
        """
        Both representations use the same task layout, task seed, initial robot state,
        six-dimensional IK, joint actuators, controller gains, and temporal speedup. The
        action representation is the intended independent variable.
        """
    )
    st.subheader("What has been implemented")
    st.markdown(
        """
        - Discrete action chunks and adaptive cubic B-spline actions.
        - Temporal scaling with separate policy and controller rates.
        - Six-dimensional damped least-squares IK with fixed orientation.
        - Named Panda joints and actuators, physical settling, and calibrated gains.
        - Scenario-specific obstacles, pushing geometry, contact roles, and clearance.
        - Path and pushing success predicates with sustained completion timing.
        - Paired task seeds, bootstrap intervals, and speed-success reports.
        - Matched compact policy encoders with chunk and spline prediction heads.
        """
    )
    st.subheader("Current limitations")
    st.markdown(
        """
        - The learned policy pipeline is state-first and small-scale.
        - This is not a reproduction of the paper's Diffusion Policy, ACT, or real robot.
        - Clearance uses MuJoCo collision-geometry distance, which approximates the visual meshes.
        - Simulation success does not establish physical-robot safety.
        - RGB-D policy learning is outside this milestone; depth and segmentation are diagnostics.
        """
    )
    st.info(
        "A defensible conclusion requires the completed multi-seed frontier. Single videos "
        "and configured durations are diagnostic evidence, not benchmark results.",
        icon=":material/fact_check:",
    )
    st.subheader("Technical questions and answers")
    questions = {
        "Where does the time improvement come from?": (
            "The same curve can be traversed faster and sampled densely without retraining "
            "or introducing zero-order-hold chunk boundaries."
        ),
        "Why is success measured with time?": (
            "A controller can always be commanded to finish sooner. It is only an "
            "improvement if the task, tracking, contact, and saturation criteria still pass."
        ),
        "What do task seeds change?": (
            "They produce paired task layouts and waypoint perturbations. Training uses a "
            "separate seed so task variation and model initialization remain attributable."
        ),
        "Why did the original tracking error look so large?": (
            "The table overlapped the Panda base. Joint 1 saturated against a persistent "
            "collision; correcting the scene reduced open-space RMSE from 6.05 cm to 0.89 cm."
        ),
        "Is this the full B-spline Policy paper?": (
            "No. It is an educational, reproduction-inspired test of the action "
            "representation and acceleration mechanism using a smaller matched policy."
        ),
    }
    for question, answer in questions.items():
        with st.expander(question):
            st.write(answer)
    st.link_button("B-spline Policy paper", PAPER_URL, icon=":material/article:")


render_header()

scenario_paths = sorted(SCENARIOS.glob("*.yaml"))
if not scenario_paths:
    st.error("No scenarios found.", icon=":material/error:")
    st.stop()

with st.sidebar:
    st.header("Experiment controls")
    scenario_path = st.selectbox(
        "Scenario",
        scenario_paths,
        format_func=lambda path: path.stem.replace("_", " "),
    )
    base_config = load_config(scenario_path, ROOT / "configs" / "defaults.yaml")
    duration = st.slider(
        "Movement duration (seconds)", 1.0, 10.0, base_config.planner.duration, 0.5
    )
    sample_rate = st.slider(
        "Trajectory targets per second",
        20.0,
        120.0,
        base_config.planner.sample_rate,
        10.0,
    )
    st.caption("SplineFlow-Panda v0.3 · CPU-first research benchmark")

page = st.segmented_control(
    "Workspace",
    ["Overview", "B-spline lab", "Experiments", "Results", "Methods & scope"],
    default="Overview",
    label_visibility="collapsed",
)

if page == "Overview":
    render_overview()
elif page == "B-spline lab":
    render_trajectory_lab(scenario_path, duration, sample_rate)
elif page == "Experiments":
    render_simulation_lab(scenario_path)
elif page == "Results":
    render_results_page()
else:
    render_methods_page()

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .evaluation import bootstrap_mean_interval, maximum_reliable_speedup, path_success
from .models import ExperimentConfig, PlannerKind
from .runner import run_experiment


@dataclass(frozen=True)
class BenchmarkDefinition:
    speedups: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 4.0)
    seeds: tuple[int, ...] = tuple(range(10))
    representations: tuple[PlannerKind, ...] = (
        PlannerKind.ACTION_CHUNK,
        PlannerKind.BSPLINE_ACTION,
    )
    minimum_success_rate: float = 0.8


DEFAULT_BENCHMARK = BenchmarkDefinition()


def run_benchmark(
    base: ExperimentConfig,
    model: Path,
    output: Path,
    definition: BenchmarkDefinition = DEFAULT_BENCHMARK,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    root = output / f"benchmark-{base.name}-{stamp}"
    root.mkdir(parents=True, exist_ok=False)
    layouts = [
        {
            "task_seed": seed,
            "sampled_waypoints": base.model_copy(
                update={"task_seed": seed}
            ).sampled_waypoints().tolist(),
        }
        for seed in definition.seeds
    ]
    (root / "layout_manifest.json").write_text(
        json.dumps({"scenario": base.name, "layouts": layouts}, indent=2),
        encoding="utf-8",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for seed in definition.seeds:
        for representation in definition.representations:
            for speedup in definition.speedups:
                config = base.model_copy(deep=True)
                config.name = f"{base.name}-{representation.value}-{speedup:g}x-s{seed}"
                config.task_seed = seed
                config.planner.kind = representation
                config.planner.speedup = speedup
                try:
                    bundle = run_experiment(config, model, root / "runs")
                    metrics = json.loads(
                        (bundle.root / "metrics" / "metrics.json").read_text()
                    )
                    if config.task.kind == "pushing":
                        success = bool(metrics.get("task_success", 0))
                    else:
                        success = path_success(
                            final_error_m=metrics.get("final_error_m", np.inf),
                            waypoint_error_m=metrics.get("waypoint_error_max_m", np.inf),
                            orientation_error_rad=np.deg2rad(
                                metrics.get("ik_orientation_error_max_deg", np.inf)
                            ),
                            tracking_max_m=metrics["tracking_max_m"],
                            forbidden_contacts=int(metrics.get("forbidden_contacts", 0)),
                        )
                    rows.append(
                        {
                            "task_seed": seed,
                            "representation": representation.value,
                            "speedup": speedup,
                            "success": success,
                            "completion_time_s": metrics.get(
                                "completion_time_s", metrics["execution_time_s"]
                            ),
                            "tracking_rmse_m": metrics["tracking_rmse_m"],
                            "tracking_max_m": metrics["tracking_max_m"],
                            "speed_max_m_s": metrics["speed_max_m_s"],
                            "acceleration_rms_m_s2": metrics["acceleration_rms_m_s2"],
                            "jerk_rms_m_s3": metrics["jerk_rms_m_s3"],
                            "path_length_m": metrics["path_length_m"],
                            "actuator_saturation_rate": metrics.get(
                                "actuator_saturation_rate", 0.0
                            ),
                            "forbidden_contacts": metrics.get("forbidden_contacts", 0.0),
                            "minimum_clearance_m": metrics.get("minimum_clearance_m"),
                            "bundle": str(bundle.root),
                        }
                    )
                except Exception as error:
                    rows.append(
                        {
                            "task_seed": seed,
                            "representation": representation.value,
                            "speedup": speedup,
                            "success": False,
                            "error": str(error),
                        }
                    )
    report = summarize_benchmark(rows, definition.minimum_success_rate)
    (root / "rollouts.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    columns = sorted({key for row in rows for key in row})
    with (root / "rollouts.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return root
def summarize_benchmark(
    rows: list[dict[str, float | int | str | bool]], minimum_success_rate: float = 0.8
) -> dict[str, object]:
    representations = sorted({str(row["representation"]) for row in rows})
    summary: dict[str, object] = {
        "report_kind": "measured_mujoco_benchmark",
        "minimum_success_rate": minimum_success_rate,
        "rollout_count": len(rows),
        "data_policy": "All values are aggregated from the adjacent rollout ledger.",
    }
    baseline_times = [
        float(row["completion_time_s"])
        for row in rows
        if row["representation"] == "action_chunk"
        and float(row["speedup"]) == 1.0
        and row["success"]
        and "completion_time_s" in row
    ]
    baseline_time = float(np.mean(baseline_times)) if baseline_times else None
    summary["baseline_completion_time_s"] = baseline_time
    for representation in representations:
        selected = [row for row in rows if row["representation"] == representation]
        frontier = maximum_reliable_speedup(selected, minimum_success_rate=minimum_success_rate)
        speed_rows: dict[str, object] = {}
        for speed in sorted({float(row["speedup"]) for row in selected}):
            group = [row for row in selected if float(row["speedup"]) == speed]
            successful_times = [
                float(row["completion_time_s"])
                for row in group
                if row["success"] and "completion_time_s" in row
            ]
            mean, low, high = bootstrap_mean_interval(successful_times, seed=0)
            mean_value = mean if np.isfinite(mean) else None
            speed_rows[str(speed)] = {
                "success_rate": float(np.mean([bool(row["success"]) for row in group])),
                "completion_time_mean_s": mean_value,
                "completion_time_ci95_s": [
                    low if np.isfinite(low) else None,
                    high if np.isfinite(high) else None,
                ],
                "achieved_speedup": (
                    baseline_time / mean_value
                    if baseline_time is not None and mean_value
                    else None
                ),
                "rollouts": len(group),
            }
        summary[representation] = {
            "maximum_reliable_speedup": frontier if np.isfinite(frontier) else None,
            "by_speedup": speed_rows,
        }
    return summary

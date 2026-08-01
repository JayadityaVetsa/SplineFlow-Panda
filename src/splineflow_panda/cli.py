from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer

from .artifacts import ExperimentBundle
from .config import load_config
from .pipeline import evaluate_bundle, plan_experiment
from .runner import run_experiment

app = typer.Typer(help="SplineFlow-Panda experiment tools", no_args_is_help=True)


@app.command()
def validate(config: Path) -> None:
    value = load_config(config)
    typer.echo(f"Valid: {value.name} ({len(value.waypoints)} waypoints)")


@app.command()
def plan(
    config: Path,
    output: Annotated[Path, typer.Option()] = Path("experiments"),
) -> None:
    bundle = plan_experiment(load_config(config), output)
    typer.echo(str(bundle.root))


@app.command()
def evaluate(experiment: Path) -> None:
    typer.echo(json.dumps(evaluate_bundle(ExperimentBundle(experiment)), indent=2))


@app.command()
def inspect(experiment: Path) -> None:
    bundle = ExperimentBundle(experiment)
    missing = bundle.validate()
    if missing:
        typer.echo("Missing: " + ", ".join(missing), err=True)
        raise typer.Exit(1)
    typer.echo((experiment / "metadata.json").read_text(encoding="utf-8"))


@app.command()
def compare(first: Path, second: Path) -> None:
    a = json.loads((first / "metrics" / "metrics.json").read_text())
    b = json.loads((second / "metrics" / "metrics.json").read_text())
    for key in sorted(a.keys() & b.keys()):
        typer.echo(f"{key}: {a[key]:.6g} | {b[key]:.6g}")


@app.command()
def run(
    config: Path,
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
    output: Annotated[Path, typer.Option()] = Path("experiments"),
) -> None:
    value = load_config(config)
    if model is None:
        model = Path(
            "assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml"
        )
    if not model.exists():
        typer.echo(
            f"Missing Franka model: {model}. Run scripts/fetch_menagerie.ps1 first.", err=True
        )
        raise typer.Exit(2)
    bundle = run_experiment(value, model, output)
    typer.echo(str(bundle.root))


@app.command()
def benchmark(
    config: Path,
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
    output: Annotated[Path, typer.Option()] = Path("benchmarks"),
    seeds: Annotated[int, typer.Option(min=1, max=30)] = 10,
    speedups: Annotated[
        str, typer.Option(help="Comma-separated target speedups, for example 1,2,3")
    ] = "1,1.5,2,3,4",
) -> None:
    from .benchmark import BenchmarkDefinition, run_benchmark

    if model is None:
        model = Path(
            "assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml"
        )
    parsed_speedups = tuple(float(value.strip()) for value in speedups.split(","))
    if not parsed_speedups or any(value <= 0 for value in parsed_speedups):
        raise typer.BadParameter("Speedups must be positive comma-separated numbers")
    definition = BenchmarkDefinition(
        seeds=tuple(range(seeds)),
        speedups=parsed_speedups,
    )
    root = run_benchmark(load_config(config), model, output, definition)
    typer.echo(str(root))


@app.command("calibrate-controller")
def calibrate_controller(
    config: Path,
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
    output: Annotated[Path, typer.Option()] = Path("calibration"),
) -> None:
    if model is None:
        model = Path(
            "assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml"
        )
    base = load_config(config)
    profiles = [
        ("conservative", 1.0, 1.0),
        ("balanced", 2.0, 1.4),
        ("stiff", 2.5, 1.7),
    ]
    rows = []
    output.mkdir(parents=True, exist_ok=True)
    for name, gain, damping in profiles:
        candidate = base.model_copy(deep=True)
        candidate.name = f"{base.name}-controller-{name}"
        candidate.simulation.controller.gain_scale = gain
        candidate.simulation.controller.damping_scale = damping
        bundle = run_experiment(candidate, model, output / "runs")
        metrics = json.loads((bundle.root / "metrics" / "metrics.json").read_text())
        rows.append(
            {
                "profile": name,
                "gain_scale": gain,
                "damping_scale": damping,
                "tracking_rmse_m": metrics["tracking_rmse_m"],
                "tracking_max_m": metrics["tracking_max_m"],
                "actuator_saturation_rate": metrics["actuator_saturation_rate"],
                "bundle": str(bundle.root),
            }
        )
    selected = min(
        (row for row in rows if row["actuator_saturation_rate"] <= 0.05),
        key=lambda row: row["tracking_rmse_m"],
    )
    report_path = output / "controller-calibration.json"
    report_path.write_text(
        json.dumps({"profiles": rows, "selected": selected}, indent=2),
        encoding="utf-8",
    )
    typer.echo(str(report_path))


@app.command("generate-dataset")
def generate_dataset(
    source: Path,
    output: Annotated[Path, typer.Option()] = Path("datasets/demonstrations.npz"),
    seeds: Annotated[int, typer.Option(min=2, max=30)] = 10,
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
) -> None:
    from .learning import build_learning_dataset
    from .models import PlannerKind

    experiments = source
    if source.suffix.lower() in {".yaml", ".yml"}:
        if model is None:
            model = Path(
                "assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml"
            )
        experiments = output.parent / f"{output.stem}-source-runs"
        manifest = []
        for task_seed in range(seeds):
            config = load_config(source)
            config.name = f"{config.name}-demonstration-s{task_seed}"
            config.task_seed = task_seed
            config.planner.kind = PlannerKind.BSPLINE_ACTION
            try:
                bundle = run_experiment(config, model, experiments)
                manifest.append(
                    {
                        "task_seed": task_seed,
                        "bundle": str(bundle.root),
                        "source": "scripted_bspline_demonstration",
                        "accepted": True,
                    }
                )
            except Exception as error:
                manifest.append(
                    {"task_seed": task_seed, "accepted": False, "error": str(error)}
                )
        manifest_path = output.with_suffix(".manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    from .learning import dataset_summary

    dataset = build_learning_dataset(experiments)
    dataset.save(output)
    output.with_suffix(".summary.json").write_text(
        json.dumps(dataset_summary(dataset), indent=2), encoding="utf-8"
    )
    unique_seeds = len(set(dataset.task_seed.tolist()))
    typer.echo(
        f"{output} ({len(dataset.observation)} samples across {unique_seeds} task seeds)"
    )


@app.command("fit-splines")
def fit_splines(
    experiments: Path,
    output: Annotated[Path, typer.Option()] = Path("datasets/demonstrations.npz"),
) -> None:
    """Build chunk and adaptively fitted spline targets from the same demonstrations."""
    from .learning import build_learning_dataset

    dataset = build_learning_dataset(experiments)
    dataset.save(output)
    typer.echo(f"{output} ({len(dataset.observation)} paired chunk/spline targets)")


@app.command()
def train(
    dataset: Path,
    representation: Annotated[
        str, typer.Option(help="action_chunk or bspline_action")
    ] = "bspline_action",
    output: Annotated[Path, typer.Option()] = Path("checkpoints/policy.pt"),
    epochs: Annotated[int, typer.Option(min=1)] = 100,
) -> None:
    from .learning import train_matched_policy

    train_matched_policy(
        dataset,
        output,
        representation=representation,
        epochs=epochs,
    )
    typer.echo(str(output))


@app.command()
def rollout(
    checkpoint: Path,
    source: Path,
    mode: Annotated[Literal["open-loop", "closed-loop"], typer.Option()] = "open-loop",
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
    output: Annotated[Path, typer.Option()] = Path("learned-rollouts"),
) -> None:
    """Evaluate reconstruction or execute a learned policy in closed-loop MuJoCo."""
    if mode == "open-loop":
        from .learning import evaluate_policy_checkpoint

        result = evaluate_policy_checkpoint(checkpoint, source)
        typer.echo(json.dumps(result, indent=2))
        return
    if model is None:
        model = Path("assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml")
    if not model.exists():
        raise typer.BadParameter(f"MuJoCo model does not exist: {model}")
    config = load_config(source)
    bundle = run_experiment(
        config, model, output, policy_checkpoint=checkpoint
    )
    typer.echo(str(bundle.root))


@app.command("validate-report")
def validate_report(report: Path) -> None:
    from .reporting import validate_benchmark_report

    typer.echo(json.dumps(validate_benchmark_report(report), indent=2))


@app.command("export-release")
def export_release(
    benchmark: Path,
    output: Annotated[Path, typer.Option()] = Path("results/v0.3-development"),
) -> None:
    from .release import export_release_artifacts

    typer.echo(str(export_release_artifacts(benchmark, output)))


@app.command("export-example")
def export_example(
    experiment: Path,
    output: Annotated[Path, typer.Option()] = Path("examples/measured-bundle"),
) -> None:
    from .release import export_example_bundle

    typer.echo(str(export_example_bundle(experiment, output)))


@app.command("reproduce-milestone")
def reproduce_milestone(
    scenario_dir: Path = Path("configs/scenarios"),
    model: Annotated[Path | None, typer.Option(help="MuJoCo Panda scene XML")] = None,
    output: Annotated[Path, typer.Option()] = Path("milestone-runs"),
    seeds: Annotated[int, typer.Option(min=1, max=30)] = 10,
    speedups: Annotated[str, typer.Option()] = "1,1.5,2,3",
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Reproduce and validate the three-scenario scripted v0.3 milestone."""
    from .benchmark import BenchmarkDefinition, run_benchmark
    from .reporting import validate_benchmark_report

    if model is None:
        model = Path("assets/mujoco_menagerie/franka_emika_panda/splineflow_scene.xml")
    scenarios = [
        scenario_dir / "direction_change.yaml",
        scenario_dir / "obstacle_adjacent.yaml",
        scenario_dir / "pushing.yaml",
    ]
    missing = [str(path) for path in [model, *scenarios] if not path.exists()]
    if missing:
        raise typer.BadParameter("Missing milestone inputs: " + ", ".join(missing))
    values = tuple(float(value.strip()) for value in speedups.split(","))
    definition = BenchmarkDefinition(seeds=tuple(range(seeds)), speedups=values)
    if dry_run:
        typer.echo(
            json.dumps(
                {
                    "status": "ready",
                    "scenarios": [str(path) for path in scenarios],
                    "seeds": seeds,
                    "speedups": values,
                    "planned_rollouts": len(scenarios) * seeds * len(values) * 2,
                },
                indent=2,
            )
        )
        return
    reports = []
    for scenario in scenarios:
        root = run_benchmark(load_config(scenario), model, output, definition)
        reports.append({"root": str(root), **validate_benchmark_report(root)})
    output.mkdir(parents=True, exist_ok=True)
    (output / "milestone-summary.json").write_text(
        json.dumps({"reports": reports}, indent=2), encoding="utf-8"
    )
    typer.echo(str(output / "milestone-summary.json"))


@app.command()
def dashboard() -> None:
    module = Path(__file__).with_name("dashboard.py")
    raise typer.Exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(module)]))


if __name__ == "__main__":
    app()

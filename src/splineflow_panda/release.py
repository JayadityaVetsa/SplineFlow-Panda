from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import plotly.express as px


def export_release_artifacts(benchmark: Path, output: Path) -> Path:
    """Create a Git-friendly, measured result package without raw frame tensors."""
    from .reporting import validate_benchmark_report

    validation = validate_benchmark_report(benchmark)
    output.mkdir(parents=True, exist_ok=False)
    report = json.loads((benchmark / "report.json").read_text(encoding="utf-8"))
    rows = json.loads((benchmark / "rollouts.json").read_text(encoding="utf-8"))
    compact_rows = []
    for row in rows:
        compact = dict(row)
        bundle = Path(str(compact.pop("bundle", "")))
        compact["bundle_id"] = bundle.name
        compact["source_bundle_in_release"] = False
        compact_rows.append(compact)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "rollouts.json").write_text(
        json.dumps(compact_rows, indent=2), encoding="utf-8"
    )
    shutil.copy2(benchmark / "layout_manifest.json", output / "layout_manifest.json")
    frame = pd.DataFrame(compact_rows)
    frame.to_csv(output / "rollouts.csv", index=False)
    aggregate = (
        frame.groupby(["representation", "speedup"], as_index=False)
        .agg(success_rate=("success", "mean"))
    )
    figure = px.line(
        aggregate,
        x="speedup",
        y="success_rate",
        color="representation",
        markers=True,
        labels={"speedup": "Target speedup", "success_rate": "Success rate"},
        title="Measured speed-success frontier",
    )
    figure.update_yaxes(range=[-0.02, 1.02])
    figure.write_html(output / "frontier.html", include_plotlyjs="cdn")
    (output / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return output


def export_example_bundle(source: Path, output: Path) -> Path:
    """Copy one inspectable bundle while excluding videos and dense frame tensors."""
    if output.exists():
        raise FileExistsError(output)
    for relative in ("states", "metrics"):
        (output / relative).mkdir(parents=True, exist_ok=True)
    for name in ("config.yaml", "metadata.json", "status.json"):
        shutil.copy2(source / name, output / name)
    for name in (
        "desired_trajectory.npz",
        "commanded_trajectory.npz",
        "executed_trajectory.npz",
    ):
        path = source / "states" / name
        if path.exists():
            shutil.copy2(path, output / "states" / name)
    shutil.copy2(source / "metrics" / "metrics.json", output / "metrics" / "metrics.json")
    return output

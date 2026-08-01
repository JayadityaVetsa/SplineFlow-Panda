import json
from pathlib import Path

import pytest

from splineflow_panda.benchmark import summarize_benchmark
from splineflow_panda.reporting import validate_benchmark_report


def test_report_validator_recomputes_measured_pairs(tmp_path: Path) -> None:
    rows = []
    for seed in (0, 1):
        for representation in ("action_chunk", "bspline_action"):
            rows.append(
                {
                    "task_seed": seed,
                    "representation": representation,
                    "speedup": 1.0,
                    "success": True,
                    "completion_time_s": 4.0,
                }
            )
    report = summarize_benchmark(rows)
    (tmp_path / "rollouts.json").write_text(json.dumps(rows), encoding="utf-8")
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (tmp_path / "layout_manifest.json").write_text(
        json.dumps({"layouts": [{"task_seed": 0}, {"task_seed": 1}]}),
        encoding="utf-8",
    )
    result = validate_benchmark_report(tmp_path)
    assert result["rollouts"] == 4


def test_report_validator_rejects_duplicate_conditions(tmp_path: Path) -> None:
    row = {
        "task_seed": 0,
        "representation": "action_chunk",
        "speedup": 1.0,
        "success": True,
        "completion_time_s": 4.0,
    }
    (tmp_path / "rollouts.json").write_text(json.dumps([row, row]), encoding="utf-8")
    (tmp_path / "report.json").write_text(
        json.dumps(summarize_benchmark([row, row])), encoding="utf-8"
    )
    (tmp_path / "layout_manifest.json").write_text(
        json.dumps({"layouts": [{"task_seed": 0}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        validate_benchmark_report(tmp_path)

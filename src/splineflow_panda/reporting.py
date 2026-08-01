from __future__ import annotations

import json
from pathlib import Path

from .benchmark import summarize_benchmark


def validate_benchmark_report(root: Path) -> dict[str, int | str]:
    ledger_path = root / "rollouts.json"
    report_path = root / "report.json"
    manifest_path = root / "layout_manifest.json"
    missing = [
        path.name for path in (ledger_path, report_path, manifest_path) if not path.exists()
    ]
    if missing:
        raise ValueError("Missing benchmark files: " + ", ".join(missing))
    rows = json.loads(ledger_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if report.get("report_kind") != "measured_mujoco_benchmark":
        raise ValueError("Only measured MuJoCo reports are accepted")
    keys = [
        (int(row["task_seed"]), str(row["representation"]), float(row["speedup"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed/representation/speedup rows")
    seeds = sorted({key[0] for key in keys})
    representations = sorted({key[1] for key in keys})
    speedups = sorted({key[2] for key in keys})
    expected = {
        (seed, representation, speedup)
        for seed in seeds
        for representation in representations
        for speedup in speedups
    }
    if set(keys) != expected:
        raise ValueError("Benchmark does not contain complete paired conditions")
    manifest_seeds = sorted(int(item["task_seed"]) for item in manifest["layouts"])
    if manifest_seeds != seeds:
        raise ValueError("Layout manifest seeds do not match the rollout ledger")
    for row in rows:
        if "bundle" in row and not Path(str(row["bundle"])).exists():
            raise ValueError(f"Referenced experiment bundle is missing: {row['bundle']}")
    recomputed = summarize_benchmark(rows, float(report["minimum_success_rate"]))
    for key in ("action_chunk", "bspline_action", "baseline_completion_time_s"):
        if report.get(key) != recomputed.get(key):
            raise ValueError(f"Report aggregate differs from ledger for {key}")
    return {
        "status": "valid",
        "rollouts": len(rows),
        "seeds": len(seeds),
        "conditions": len(representations) * len(speedups),
    }

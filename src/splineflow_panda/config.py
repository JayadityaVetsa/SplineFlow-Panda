from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .models import ExperimentConfig


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    path: Path, defaults: Path | None = None, overrides: dict[str, Any] | None = None
) -> ExperimentConfig:
    base = {}
    if defaults:
        base = yaml.safe_load(defaults.read_text(encoding="utf-8")) or {}
    scenario = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(_merge(_merge(base, scenario), overrides or {}))


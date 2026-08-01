from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path | None = None) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def environment_provenance() -> dict[str, str | None]:
    try:
        import mujoco

        mujoco_version = mujoco.__version__
    except ImportError:
        mujoco_version = None
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mujoco": mujoco_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch_version,
        "git_commit": git_commit(Path.cwd()),
    }

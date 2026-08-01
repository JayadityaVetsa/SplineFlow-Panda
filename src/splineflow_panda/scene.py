from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from .models import TaskConfig


def _task_xml(task: TaskConfig) -> str:
    entries: list[str] = []
    for box in task.boxes:
        position = " ".join(map(str, box.position))
        size = " ".join(map(str, box.half_size))
        rgba = " ".join(map(str, box.rgba))
        collision = 'contype="0" conaffinity="0"' if box.collision_role in {
            "goal",
            "decorative",
        } else ""
        entries.append(
            f'<geom name="{box.collision_role}__{box.name}" type="box" '
            f'pos="{position}" size="{size}" rgba="{rgba}" {collision}/>'
        )
    if task.puck:
        puck = task.puck
        position = " ".join(map(str, puck.position))
        entries.append(
            f'<body name="{puck.name}" pos="{position}">'
            f'<freejoint name="{puck.name}_free"/>'
            f'<geom name="puck__{puck.name}" type="cylinder" '
            f'size="{puck.radius} {puck.height / 2}" '
            'friction="1.0 0.01 0.001" rgba="0.12 0.55 0.95 1"/>'
            "</body>"
        )
    if task.goal:
        goal = task.goal
        entries.append(
            f'<geom name="goal__target" type="cylinder" '
            f'pos="{goal.center[0]} {goal.center[1]} 0.215" '
            f'size="{goal.radius} 0.002" rgba="0.1 0.8 0.3 0.45" '
            'contype="0" conaffinity="0"/>'
        )
    return "\n".join(entries)


@contextmanager
def resolved_scene(base_scene: Path, task: TaskConfig) -> Iterator[Path]:
    """Create a scenario-specific scene beside Panda assets so relative includes remain valid."""
    source = base_scene.read_text(encoding="utf-8")
    if "</worldbody>" not in source:
        raise ValueError(f"Scene {base_scene} has no worldbody")
    source = source.replace("</worldbody>", _task_xml(task) + "\n</worldbody>", 1)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix="splineflow-resolved-",
            dir=base_scene.parent,
            encoding="utf-8",
            delete=False,
        ) as handle:
            handle.write(source)
            temporary = Path(handle.name)
        yield temporary
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

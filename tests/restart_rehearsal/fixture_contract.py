"""Validated access to the candidate-owned restart rehearsal fixture."""

from __future__ import annotations

import json
from pathlib import Path


FIXTURE_PATH = Path(
    "tests/restart_rehearsal/fixtures/production_shaped_v2.json"
)


def load_rehearsal_metagraph_hotkeys(
    source_root: Path,
) -> tuple[str, ...]:
    """Return the ordered metagraph identities declared by the candidate."""

    fixture_path = Path(source_root) / FIXTURE_PATH
    try:
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            "candidate rehearsal fixture is unavailable or invalid"
        ) from exc
    if not isinstance(fixture, dict) or fixture.get("schema_version") != (
        "leadpoet.restart_rehearsal_fixture.v1"
    ):
        raise ValueError("candidate rehearsal fixture schema differs")
    metagraph = fixture.get("metagraph")
    hotkeys = metagraph.get("hotkeys") if isinstance(metagraph, dict) else None
    if not isinstance(hotkeys, list):
        raise ValueError("candidate rehearsal metagraph hotkeys differ")
    normalized = tuple(
        value.strip() if isinstance(value, str) else ""
        for value in hotkeys
    )
    if (
        len(normalized) != 4
        or any(not value for value in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("candidate rehearsal metagraph hotkeys differ")
    return normalized

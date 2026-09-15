"""Atomic, complete Arena output checkpoints available to submitted harnesses.

The model chooses when to call ``write``. The Arena host only observes complete
output bytes that it can validate before the signed execution deadline.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


OUTPUT_PATH = Path("/output/companies.json")
MAX_OUTPUT_BYTES = 512 * 1024


def write(companies: list[dict[str, Any]], *, output_path: Path = OUTPUT_PATH) -> None:
    """Publish one complete ``{"companies": [...]}`` document atomically."""

    if not isinstance(companies, list) or any(not isinstance(row, dict) for row in companies):
        raise ValueError("companies must be a list of company objects")
    payload = json.dumps(
        {"companies": companies}, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValueError("checkpoint exceeds Arena output limit")
    output_path = Path(output_path)
    descriptor, name = tempfile.mkstemp(prefix=".arena-checkpoint-", dir=output_path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
        os.replace(name, output_path)
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass

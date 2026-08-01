"""Test-only helpers for production-shaped private model source fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from research_lab.sourcing_model_contract_check import (
    CONTRACT_PATH,
    reviewed_consumer_snapshots,
)


DEFAULT_CONSUMER_CONTRACT_ID = str(
    json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["contract_id"]
)


def install_reviewed_consumer_snapshot(
    source_root: Path,
    *,
    contract_id: str = DEFAULT_CONSUMER_CONTRACT_ID,
) -> None:
    """Install one exact reviewed contract/parity pair into a test source."""

    snapshot = reviewed_consumer_snapshots()[contract_id]
    contract = snapshot["contract"]
    contract_target = source_root / str(contract["canonical_path"])
    parity_target = source_root / str(contract["parity_fixture_path"])
    contract_target.parent.mkdir(parents=True, exist_ok=True)
    parity_target.parent.mkdir(parents=True, exist_ok=True)
    contract_target.write_bytes(Path(snapshot["contract_path"]).read_bytes())
    parity_target.write_bytes(Path(snapshot["parity_path"]).read_bytes())

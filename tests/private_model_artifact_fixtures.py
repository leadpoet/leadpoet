"""Test-only helpers for production-shaped private model source fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from research_lab.sourcing_model_contract_check import (
    CONTRACT_PATH,
    _resolve_reviewed_consumer_contract_pair,
    compute_compatibility_source_tree_hash_v1,
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


def build_private_artifact_with_adapted_source_admission(
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a structural artifact for tests outside source admission.

    The real builder is still exercised, but its semantic-admission call is an
    explicit test boundary. The boundary accepts only an exact reviewed
    contract/parity pair and never represents the synthetic tree as a reviewed
    release or a production compatibility receipt.
    """

    from research_lab.eval import private_runtime

    source_root = Path(kwargs["source_path"])
    if _resolve_reviewed_consumer_contract_pair(source_root) is None:
        raise AssertionError("test artifact requires an exact reviewed contract pair")

    def adapted_source_admission(
        root: Path,
        *,
        source_tree_hash: str = "",
        **_ignored: Any,
    ) -> dict[str, Any]:
        observed = compute_compatibility_source_tree_hash_v1(Path(root))
        if source_tree_hash != observed:
            raise AssertionError("test source admission received a stale tree hash")
        return {"source_admission_exercised": False}

    with patch.object(
        private_runtime,
        "source_tree_compatibility_admission_v1",
        adapted_source_admission,
    ):
        return private_runtime.build_local_private_artifact_manifest(**kwargs)

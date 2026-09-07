"""SOURCE_ADD public-intake credential boundary tests."""

from __future__ import annotations

from typing import Any

from research_lab.source_add_execution import (
    SourceAddRejectionReason,
    intake_source_add_submission,
)


def _manifest_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "adapter_id": "adapter:glue-test-1",
        "miner_ref": "miner:hotkey-g",
        "source_name": "Glue Test Source",
        "source_kind": "news",
        "declared_base_domains": ["gluefeed.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": [
            "evidence_refs",
            "snapshot_refs",
            "content_hashes",
            "normalized_text_hashes",
        ],
        "submitted_artifact_ref": "artifact:glue",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 500,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:glue"],
    }
    doc.update(overrides)
    return doc


def test_legacy_miner_credential_manifest_is_rejected_at_intake():
    record, errors = intake_source_add_submission(
        _manifest_doc(
            adapter_id="adapter:glue-cred-1",
            credential_policy="credential_ref_only",
            credential_ref="encrypted_ref:source_add:abc",
        ),
        miner_hotkey="hk-glue",
    )

    assert record is None
    assert any(
        SourceAddRejectionReason.CREDENTIAL_INVALID.value in item
        for item in errors
    )

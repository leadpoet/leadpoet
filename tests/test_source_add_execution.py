"""SOURCE_ADD intake and anti-spam admission tests."""

from __future__ import annotations

import json
from typing import Any

from research_lab.source_add_execution import (
    SourceAddFunnelStage,
    SourceAddRejectionReason,
    intake_source_add_submission,
    normalize_base_domain,
)


def _manifest_doc(**overrides: Any) -> dict[str, Any]:
    doc = {
        "adapter_id": "adapter:intent-feed-1",
        "miner_ref": "miner:hotkey-a",
        "source_name": "Intent Feed One",
        "source_kind": "news",
        "declared_base_domains": ["intentfeed.example"],
        "output_schema_ref": "schema:source-add-output:v1",
        "allowed_output_fields": ["evidence_refs", "snapshot_refs", "content_hashes", "normalized_text_hashes"],
        "submitted_artifact_ref": "artifact:bundle-1",
        "code_bundle_hash": "sha256:" + "a" * 64,
        "sandbox_policy_ref": "policy:sandbox-v1",
        "max_trial_cost_cents": 500,
        "max_request_cost_cents": 5,
        "max_latency_ms": 30_000,
        "fixture_refs": ["fixture:sample-1"],
    }
    doc.update(overrides)
    return doc


def _fake_kms(raw_key: str, miner_hotkey: str, adapter_ref: str) -> dict[str, str]:
    return {
        "ciphertext_b64": "ZW5jcnlwdGVk",
        "kms_key_id": "alias/test",
        "encryption_context_hash": "sha256:ctx",
    }


def _intake(**overrides: Any):
    kwargs = dict(
        miner_hotkey="hotkey-a",
        existing_catalog_domains=(),
        open_submission_count_for_hotkey=0,
        submissions_last_day_for_hotkey=0,
        submissions_last_30d_for_hotkey=0,
        kms_encrypt=_fake_kms,
    )
    manifest = overrides.pop("manifest_doc", _manifest_doc())
    kwargs.update(overrides)
    return intake_source_add_submission(manifest, **kwargs)


class TestIntake:
    def test_valid_submission_advances_to_manifest_validated(self):
        record, errors = _intake()
        assert errors == []
        assert record.stage == SourceAddFunnelStage.MANIFEST_VALIDATED.value
        assert record.submission_id.startswith("source_add_submission:")

    def test_invalid_manifest_rejected_with_reasons(self):
        record, errors = _intake(manifest_doc=_manifest_doc(declared_base_domains=[]))
        assert record is None
        assert any(SourceAddRejectionReason.MANIFEST_INVALID.value in error for error in errors)

    def test_thin_wrapper_of_existing_catalog_source_rejected(self):
        record, errors = _intake(existing_catalog_domains=("www.IntentFeed.example",))
        assert record is None
        assert SourceAddRejectionReason.DUPLICATE_SOURCE.value in errors

    def test_hotkey_concurrent_cap(self):
        record, errors = _intake(open_submission_count_for_hotkey=3)
        assert record is None
        assert SourceAddRejectionReason.HOTKEY_CONCURRENT_CAP.value in errors

    def test_hotkey_day_cap(self):
        record, errors = _intake(submissions_last_day_for_hotkey=5)
        assert record is None
        assert SourceAddRejectionReason.HOTKEY_DAY_CAP.value in errors

    def test_hotkey_30d_cap(self):
        record, errors = _intake(submissions_last_30d_for_hotkey=10)
        assert record is None
        assert SourceAddRejectionReason.HOTKEY_30D_CAP.value in errors

    def test_miner_credential_policy_and_raw_secret_are_rejected(self):
        record, errors = _intake(
            manifest_doc=_manifest_doc(credential_policy="credential_ref_only"),
            raw_credential="raw-secret-key-123456",
        )
        assert record is None
        assert any(SourceAddRejectionReason.CREDENTIAL_INVALID.value in error for error in errors)
        assert "raw-secret-key-123456" not in json.dumps(errors)

    def test_raw_credential_with_no_credentials_policy_rejected(self):
        record, errors = _intake(raw_credential="oops-key-123456789012")
        assert record is None
        assert any(SourceAddRejectionReason.CREDENTIAL_INVALID.value in error for error in errors)

    def test_credential_ref_only_without_key_or_ref_rejected(self):
        record, errors = _intake(manifest_doc=_manifest_doc(credential_policy="credential_ref_only"))
        assert record is None
        assert any("miner credentials are not accepted" in error for error in errors)


class TestDomainNormalization:
    def test_normalize_strips_scheme_www_port_path(self):
        assert normalize_base_domain("https://www.Example.COM:443/path") == "example.com"

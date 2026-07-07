"""W5 SOURCE_ADD execution: intake, anti-spam, funnel, category-scoped trial."""

from __future__ import annotations

import json
from typing import Any

import pytest

from research_lab.source_add_execution import (
    SourceAddFunnelStage,
    SourceAddRejectionReason,
    SourceAddSuggestionDoc,
    apply_trial_result,
    evaluate_source_add_acceptance,
    intake_source_add_submission,
    normalize_base_domain,
    run_llm_review_stage,
    run_sandboxed_trial,
    run_static_scan_stage,
    static_scan_adapter_bundle,
    validate_source_add_suggestion,
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

    def test_hotkey_30d_cap(self):
        record, errors = _intake(submissions_last_30d_for_hotkey=10)
        assert record is None
        assert SourceAddRejectionReason.HOTKEY_30D_CAP.value in errors

    def test_credential_encrypts_via_kms_and_never_persists_raw(self):
        record, errors = _intake(
            manifest_doc=_manifest_doc(credential_policy="credential_ref_only"),
            raw_credential="raw-secret-key-123456",
        )
        assert errors == []
        assert record.credential_envelope["ciphertext_b64"] == "ZW5jcnlwdGVk"
        assert record.credential_envelope["credential_ref"].startswith("encrypted_ref:source_add:")
        assert "raw-secret-key-123456" not in json.dumps(record.to_dict())

    def test_raw_credential_with_no_credentials_policy_rejected(self):
        record, errors = _intake(raw_credential="oops-key-123456789012")
        assert record is None
        assert any(SourceAddRejectionReason.CREDENTIAL_INVALID.value in error for error in errors)

    def test_credential_ref_only_without_key_or_ref_rejected(self):
        record, errors = _intake(manifest_doc=_manifest_doc(credential_policy="credential_ref_only"))
        assert record is None
        assert any("requires a key at intake" in error for error in errors)


class TestStaticScan:
    def test_clean_bundle_passes(self):
        assert static_scan_adapter_bundle({"adapter.py": "import json\n\ndef fetch(icp):\n    return []\n"}) == []

    @pytest.mark.parametrize(
        "snippet,label",
        [
            ("import subprocess\n", "subprocess"),
            ("os.system('ls')", "os_system"),
            ("eval(user_input)", "eval"),
            ("import socket\n", "socket"),
            ("API_KEY = 'abcdef123456789012'", "raw_credential"),
            ("os.environ['KEY']", "env_read"),
            ("__import__('os')", "dunder_import"),
        ],
    )
    def test_forbidden_patterns_fail(self, snippet, label):
        errors = static_scan_adapter_bundle({"adapter.py": snippet})
        assert any(error.startswith(label + ":") for error in errors)

    def test_scan_stage_rejects_record(self):
        record, _ = _intake()
        rejected = run_static_scan_stage(record, {"adapter.py": "import subprocess"})
        assert rejected.stage == SourceAddFunnelStage.REJECTED.value
        assert rejected.rejection_stage == SourceAddFunnelStage.STATIC_SCAN_PASSED.value
        assert SourceAddRejectionReason.STATIC_SCAN_FAILED.value in rejected.rejection_reasons


def _output_doc(adapter_id: str, icp_ref: str, count: int) -> dict[str, Any]:
    return {
        "output_ref": f"output:{icp_ref}",
        "adapter_id": adapter_id,
        "icp_ref": icp_ref,
        "evidence_refs": [f"evidence:{icp_ref}:{index}" for index in range(count)],
        "snapshot_refs": [f"snapshot:{icp_ref}"],
        "content_hashes": ["sha256:" + "b" * 64],
        "normalized_text_hashes": ["sha256:" + "c" * 64],
    }


class TestSandboxedTrial:
    def _record(self):
        record, _ = _intake()
        record = run_static_scan_stage(record, {"adapter.py": "import json"})
        return run_llm_review_stage(record, llm_reviewer=lambda _r: {"verdict": "pass"})

    def test_category_scoped_yield_counts_only_declared_category(self):
        record = self._record()  # declared source_kind = "news"

        def _runner(rec, icp_ref):
            return {"output": _output_doc(rec.adapter_id, icp_ref, 4), "cost_cents": 10}

        def _classifier(evidence_ref):
            # Half the evidence classifies as news, half as firmographics.
            return "news" if evidence_ref.endswith((":0", ":1")) else "firmographic"

        trial = run_sandboxed_trial(
            record, trial_icp_refs=("icp:1", "icp:2"), sandbox_runner=_runner, evidence_classifier=_classifier
        )
        assert trial.failure_reason == ""
        assert trial.total_evidence_count == 8
        assert trial.category_matching_evidence_count == 4
        # 4 matching over 2 ICPs, capped at 1.0.
        assert trial.measured_trial_yield == 1.0

    def test_category_mismatch_scores_zero_with_diagnostics(self):
        record = self._record()

        def _runner(rec, icp_ref):
            return {"output": _output_doc(rec.adapter_id, icp_ref, 3), "cost_cents": 5}

        trial = run_sandboxed_trial(
            record,
            trial_icp_refs=("icp:1",),
            sandbox_runner=_runner,
            evidence_classifier=lambda _ref: "firmographic",
        )
        assert trial.failure_reason == SourceAddRejectionReason.CATEGORY_MISMATCH.value
        assert trial.measured_trial_yield == 0.0
        assert "category_mismatch" in trial.diagnostics

    def test_zero_yield_rejection(self):
        record = self._record()
        trial = run_sandboxed_trial(
            record,
            trial_icp_refs=("icp:1",),
            sandbox_runner=lambda rec, icp: {"output": None, "cost_cents": 0},
            evidence_classifier=lambda _ref: "news",
        )
        assert trial.failure_reason == SourceAddRejectionReason.ZERO_YIELD.value

    def test_trial_stops_at_miner_declared_cost_cap(self):
        record = self._record()  # max_trial_cost_cents = 500
        calls: list[str] = []

        def _runner(rec, icp_ref):
            calls.append(icp_ref)
            return {"output": _output_doc(rec.adapter_id, icp_ref, 1), "cost_cents": 300}

        trial = run_sandboxed_trial(
            record,
            trial_icp_refs=("icp:1", "icp:2", "icp:3"),
            sandbox_runner=_runner,
            evidence_classifier=lambda _ref: "news",
        )
        assert trial.failure_reason == SourceAddRejectionReason.QUOTA_EXCEEDED.value
        assert calls == ["icp:1", "icp:2"]  # third never runs

    def test_auth_failure_terminates_with_structured_diagnostics(self):
        record = self._record()
        trial = run_sandboxed_trial(
            record,
            trial_icp_refs=("icp:1", "icp:2"),
            sandbox_runner=lambda rec, icp: {"error": "auth_failure", "cost_cents": 0},
            evidence_classifier=lambda _ref: "news",
        )
        assert trial.failure_reason == SourceAddRejectionReason.AUTH_FAILURE.value

    def test_schema_violation_rejects(self):
        record = self._record()
        bad_output = _output_doc(record.adapter_id, "icp:1", 2)
        bad_output["content_hashes"] = []  # schema violation

        trial = run_sandboxed_trial(
            record,
            trial_icp_refs=("icp:1",),
            sandbox_runner=lambda rec, icp: {"output": bad_output, "cost_cents": 1},
            evidence_classifier=lambda _ref: "news",
        )
        assert trial.failure_reason == SourceAddRejectionReason.SCHEMA_VIOLATION.value


class TestAcceptance:
    def _completed_record(self, measured_yield: float):
        record, _ = _intake()
        record = run_static_scan_stage(record, {"adapter.py": "import json"})
        record = run_llm_review_stage(record, llm_reviewer=lambda _r: {"verdict": "pass"})

        def _runner(rec, icp_ref):
            return {"output": _output_doc(rec.adapter_id, icp_ref, 1), "cost_cents": 1}

        # One evidence item per ICP; the first `measured_yield * 10` ICPs
        # classify into the declared category → yield == measured_yield.
        matching_icps = {f"icp:{i}" for i in range(int(measured_yield * 10))}

        def _classifier(evidence_ref):
            icp = evidence_ref.split(":", 1)[1].rsplit(":", 1)[0]
            return "news" if icp in matching_icps else "other"

        trial = run_sandboxed_trial(
            record, trial_icp_refs=tuple(f"icp:{i}" for i in range(10)), sandbox_runner=_runner, evidence_classifier=_classifier
        )
        return apply_trial_result(record, trial)

    def test_yield_above_floor_plus_human_gate_accepts_and_builds_catalog_entry(self):
        record = self._completed_record(0.5)
        accepted, entry = evaluate_source_add_acceptance(
            record, human_gate_passed=True, accepted_at="2026-07-06T00:00:00Z", registry_provider_id="intentfeed"
        )
        assert accepted.stage == SourceAddFunnelStage.ACCEPTED.value
        assert accepted.acceptance_human_gate_passed is True
        assert entry is not None
        assert entry.adapter_id == record.adapter_id
        assert entry.registry_provider_id == "intentfeed"

    def test_yield_below_floor_rejects(self):
        record = self._completed_record(0.5)
        rejected, entry = evaluate_source_add_acceptance(
            record, human_gate_passed=True, acceptance_floor_yield=0.9
        )
        assert entry is None
        assert SourceAddRejectionReason.BELOW_ACCEPTANCE_FLOOR.value in rejected.rejection_reasons

    def test_human_gate_required_even_at_high_yield(self):
        record = self._completed_record(0.9)
        rejected, entry = evaluate_source_add_acceptance(record, human_gate_passed=False)
        assert entry is None
        assert SourceAddRejectionReason.HUMAN_GATE_NOT_PASSED.value in rejected.rejection_reasons

    def test_acceptance_requires_completed_trial(self):
        record, _ = _intake()
        rejected, entry = evaluate_source_add_acceptance(record, human_gate_passed=True)
        assert entry is None
        assert "acceptance_requires_completed_trial" in rejected.rejection_reasons


class TestSuggestionDoc:
    def test_build_and_validate(self):
        doc = SourceAddSuggestionDoc.build(
            run_id="run-1",
            provider_hint="procurement registry",
            endpoint_class="/tenders/search",
            evidence_gap="no coverage of EU public tenders in intent evidence",
            probe_receipt_hashes=["sha256:abc"],
        )
        assert validate_source_add_suggestion(doc) == []
        assert doc.suggestion_id.startswith("source_add_suggestion:")

    def test_suggestion_doc_rejects_urls_and_credentials(self):
        doc = SourceAddSuggestionDoc.build(
            run_id="run-1",
            provider_hint="fetch https://secret.example directly",
            endpoint_class="/x",
            evidence_gap="gap",
        )
        assert any("must not carry URLs" in error for error in validate_source_add_suggestion(doc))


class TestDomainNormalization:
    def test_normalize_strips_scheme_www_port_path(self):
        assert normalize_base_domain("https://www.Example.COM:443/path") == "example.com"

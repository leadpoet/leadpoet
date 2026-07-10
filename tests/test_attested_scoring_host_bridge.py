import base64
import hashlib
import json

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.research_lab import attested_scoring
from leadpoet_canonical.attested_receipts import build_receipt_body, create_signed_receipt


CONFIG_HASH = "sha256:" + "1" * 64
BUILD_HASH = "sha256:" + "2" * 64
COMMIT_SHA = "3" * 40


def _signed_parent_receipt(
    *,
    purpose,
    job_id,
    epoch_id,
    parents=(),
    input_root="sha256:" + "7" * 64,
    output_root="sha256:" + "8" * 64,
    evidence_roots=None,
):
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    body = build_receipt_body(
        role="gateway_scoring",
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        commit_sha=COMMIT_SHA,
        build_manifest_hash=BUILD_HASH,
        config_hash=CONFIG_HASH,
        input_root=input_root,
        output_root=output_root,
        evidence_roots=dict(evidence_roots or {}),
        parent_receipt_hashes=list(parents),
        status="succeeded",
        issued_at="2026-07-10T00:00:00Z",
    )
    return create_signed_receipt(
        body=body,
        enclave_pubkey=public_key,
        attestation_document_b64=base64.b64encode(b"parent-attestation").decode(),
        sign_digest=private_key.sign,
    )


class _FakeTEEClient:
    def __init__(self, result, *, enclave_mode="shadow"):
        self.result = result
        self.enclave_mode = enclave_mode
        self.calls = []
        self.manifest = None
        self.payload = bytearray()
        self.private_key = Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        ).hex()
        self.result_bytes = attested_scoring.canonical_json_bytes(result)
        self.runtime_hash = None

    async def scoring_configure_runtime(self, *, environment, configuration_hash):
        self.calls.append("configure")
        assert "QUALIFICATION_OPENROUTER_API_KEY" in environment
        self.runtime_hash = configuration_hash
        return {"status": "configured", "configuration_hash": configuration_hash}

    async def scoring_health(self):
        self.calls.append("health")
        health = {
            "mode": self.enclave_mode,
            "config_hash": self.runtime_hash or CONFIG_HASH,
            "commit_sha": COMMIT_SHA,
        }
        if self.runtime_hash:
            from gateway.tee.egress_policy import destination_policy_hash

            health["egress_proxy"] = {
                "status": "running",
                "policy_hash": destination_policy_hash(),
            }
        return health

    async def scoring_submit_job(self, manifest):
        self.calls.append("submit")
        self.manifest = dict(manifest)
        return {"state": "uploading", "uploaded_bytes": len(self.payload)}

    async def scoring_put_chunk(self, *, job_id, offset, data):
        self.calls.append("put")
        assert job_id == self.manifest["job_id"]
        assert offset == len(self.payload)
        self.payload.extend(data)
        return {"state": "uploading", "uploaded_bytes": len(self.payload)}

    async def scoring_seal_job(self, job_id):
        self.calls.append("seal")
        assert job_id == self.manifest["job_id"]
        assert attested_scoring.sha256_bytes(bytes(self.payload)) == self.manifest["payload_sha256"]
        return {"state": "queued"}

    async def scoring_get_status(self, job_id):
        self.calls.append("status")
        return {
            "state": "succeeded",
            "result_sha256": attested_scoring.sha256_bytes(self.result_bytes),
        }

    async def scoring_get_result(self, job_id, *, offset=0):
        self.calls.append("result")
        chunk = self.result_bytes[offset:]
        return {
            "data_b64": base64.b64encode(chunk).decode(),
            "chunk_sha256": attested_scoring.sha256_bytes(chunk),
            "eof": True,
        }

    async def scoring_get_receipt(self, job_id):
        self.calls.append("receipt")
        evidence_roots = dict(self.manifest["evidence_roots"])
        operation = self.manifest["operation"]
        payload = json.loads(bytes(self.payload))
        if operation == "build_score_bundle":
            bundle = self.result["score_bundle"]
            evidence_roots["score_bundle"] = bundle["score_bundle_hash"]
            gate = bundle.get("private_holdout_gate")
            if isinstance(gate, dict) and gate.get("baseline_benchmark_hash"):
                evidence_roots["baseline_score_summary"] = gate["baseline_benchmark_hash"]
        elif operation == "build_baseline_score_summary":
            evidence_roots["baseline_score_summary"] = attested_scoring.sha256_bytes(
                attested_scoring.canonical_json_bytes(self.result["score_summary_doc"])
            )
        elif operation == "promotion_improvement":
            score_bundle_hash = payload["score_bundle"].get("score_bundle_hash")
            if score_bundle_hash:
                evidence_roots["score_bundle"] = score_bundle_hash
        elif operation == "promotion_gate_decision":
            score_bundle_hash = payload["score_bundle"].get("score_bundle_hash")
            if score_bundle_hash:
                evidence_roots["score_bundle"] = score_bundle_hash
            evidence_roots["promotion_decision_status"] = attested_scoring.sha256_bytes(
                attested_scoring.canonical_json_bytes(
                    {"status": self.result["decision"]["status"]}
                )
            )
        elif operation == "research_lab_allocation":
            evidence_roots["allocation"] = self.result["allocation"]["allocation_hash"]
        body = build_receipt_body(
            role="gateway_scoring",
            purpose=self.manifest["purpose"],
            job_id=self.manifest["job_id"],
            epoch_id=self.manifest["epoch_id"],
            commit_sha=self.manifest["commit_sha"],
            build_manifest_hash=BUILD_HASH,
            config_hash=self.manifest["config_hash"],
            input_root=self.manifest["payload_sha256"],
            output_root=attested_scoring.sha256_bytes(self.result_bytes),
            evidence_roots=evidence_roots,
            parent_receipt_hashes=self.manifest["parent_receipt_hashes"],
            status="succeeded",
            issued_at="2026-07-10T00:00:00Z",
        )
        return create_signed_receipt(
            body=body,
            enclave_pubkey=self.public_key,
            attestation_document_b64=base64.b64encode(b"attestation").decode(),
            sign_digest=self.private_key.sign,
        )

    async def scoring_cancel_job(self, job_id):
        self.calls.append("cancel")
        return {"state": "cancelled"}


@pytest.fixture(autouse=True)
def _fixed_commit(monkeypatch):
    monkeypatch.setattr(attested_scoring, "_commit_sha", lambda: COMMIT_SHA)
    monkeypatch.setattr(
        attested_scoring,
        "verify_gateway_receipt_attestation",
        lambda *, receipt, expected_purpose, expected_epoch_id: (
            True,
            {
                "pcr0": "a" * 96,
                "purpose": expected_purpose,
                "epoch_id": expected_epoch_id,
                "enclave_pubkey": receipt["enclave_pubkey"],
            },
        ),
    )


@pytest.mark.asyncio
async def test_off_mode_has_no_enclave_or_business_side_effect(monkeypatch):
    fake = _FakeTEEClient({"ok": True})
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "off")
    result = await attested_scoring.execute_attested_scoring_operation(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v1",
        epoch_id=1,
        payload={"scores": [1.0]},
    )
    assert result == {"status": "off"}
    assert fake.calls == []


@pytest.mark.asyncio
async def test_shadow_bridge_binds_payload_result_and_signed_receipt(monkeypatch):
    fake = _FakeTEEClient({"score": 1.5})
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    result = await attested_scoring.execute_attested_scoring_operation(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v1",
        epoch_id=7,
        payload={"scores": [1.0, 2.0]},
        evidence_roots={"provider_trace": "sha256:" + "4" * 64},
    )
    assert result["status"] == "succeeded"
    assert result["pcr0"] == "a" * 96
    assert result["result"] == {"score": 1.5}
    assert result["receipt"]["input_root"] == fake.manifest["payload_sha256"]
    assert json.loads(bytes(fake.payload)) == {"scores": [1.0, 2.0]}
    assert fake.calls == ["health", "submit", "put", "seal", "status", "result", "receipt"]


@pytest.mark.asyncio
async def test_qualification_shadow_provisions_runtime_and_egress_before_health(monkeypatch):
    from gateway.tee.egress_policy import destination_policy_hash
    from gateway.utils import tee_egress_forwarder
    from research_lab.eval.http_tape import HttpTapeRecorder

    fake = _FakeTEEClient({"breakdowns": [], "scores": []})
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(
        tee_egress_forwarder,
        "ensure_tee_egress_forwarder",
        lambda: {
            "status": "running",
            "port": 5001,
            "policy_hash": destination_policy_hash(),
        },
    )
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")

    result = await attested_scoring.compare_qualification_company_scores(
        purpose="research_lab.candidate_score.v1",
        epoch_id=7,
        companies=[],
        icp={},
        is_reference_model=False,
        provider_tape=HttpTapeRecorder().document(),
        expected_breakdowns=[],
    )

    assert result["status"] == "matched"
    assert fake.manifest["evidence_roots"]["provider_http_tape"].startswith("sha256:")
    assert fake.calls == [
        "configure",
        "health",
        "submit",
        "put",
        "seal",
        "status",
        "result",
        "receipt",
    ]


@pytest.mark.asyncio
async def test_shadow_allocation_mismatch_is_observational(monkeypatch, caplog):
    from gateway.research_lab import attested_receipt_store

    fake = _FakeTEEClient({"allocation": {"allocation_hash": "sha256:" + "d" * 64}})
    persisted_links = []

    async def _persist(**kwargs):
        persisted_links.append(list(kwargs.get("artifact_links") or []))
        return {}

    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")
    outcome = await attested_scoring.compare_allocation(
        epoch_id=8,
        payload={
            "epoch": 8,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [],
        },
        expected_allocation={"allocation_hash": "sha256:" + "e" * 64},
    )
    assert outcome["status"] == "shadow_mismatch"
    assert "research_lab_attested_scoring_shadow_mismatch" in caplog.text
    assert persisted_links == [[]]


@pytest.mark.asyncio
async def test_shadow_promotion_metric_match(monkeypatch):
    event_doc = {
        "improvement_basis": "legacy_paired_mean_delta_no_holdout_gate",
        "daily_baseline_available": False,
    }
    fake = _FakeTEEClient({"improvement_points": 1.5, "event_doc": event_doc})
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")

    outcome = await attested_scoring.compare_promotion_metric(
        epoch_id=10,
        score_bundle={"evaluation_epoch": 10, "aggregates": {"mean_delta": 1.5}},
        expected_improvement_points=1.5,
        expected_event_doc=event_doc,
    )

    assert outcome["status"] == "matched"
    assert fake.manifest["operation"] == "promotion_improvement"
    assert fake.manifest["purpose"] == "research_lab.promotion_metric.v1"


@pytest.mark.asyncio
async def test_shadow_promotion_gate_decision_has_metric_ancestry(monkeypatch):
    score_bundle = {
        "evaluation_epoch": 10,
        "score_bundle_hash": "sha256:" + "a" * 64,
        "aggregates": {"mean_delta": 1.5},
    }
    event_doc = {
        "improvement_basis": "legacy_paired_mean_delta_no_holdout_gate",
        "daily_baseline_available": False,
    }
    metric_fake = _FakeTEEClient({"improvement_points": 1.5, "event_doc": event_doc})
    monkeypatch.setattr(attested_scoring, "tee_client", metric_fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    metric_outcome = await attested_scoring.compare_promotion_metric(
        epoch_id=10,
        score_bundle=score_bundle,
        expected_improvement_points=1.5,
        expected_event_doc=event_doc,
    )

    expected_decision = {
        "status": "promotion_passed",
        "improvement_points": 1.5,
        "threshold_points": 1.0,
        "candidate_kind": "image_build",
        "auto_promotion_enabled": True,
        "active_parent_matches": True,
        "metric_rejection_status": None,
    }
    decision_fake = _FakeTEEClient({"decision": expected_decision})
    monkeypatch.setattr(attested_scoring, "tee_client", decision_fake)
    outcome = await attested_scoring.compare_promotion_gate_decision(
        epoch_id=10,
        score_bundle=score_bundle,
        decision_payload={
            "candidate_kind": "image_build",
            "candidate_parent": "sha256:parent",
            "active_parent": "sha256:parent",
            "threshold_points": 1.0,
            "auto_promotion_enabled": True,
        },
        expected_decision=expected_decision,
        metric_outcome=metric_outcome,
    )

    assert outcome["status"] == "matched"
    assert decision_fake.manifest["operation"] == "promotion_gate_decision"
    assert decision_fake.manifest["purpose"] == "research_lab.promotion_decision.v1"
    assert decision_fake.manifest["parent_receipt_hashes"] == [
        metric_outcome["receipt"]["receipt_hash"]
    ]


@pytest.mark.asyncio
async def test_required_promotion_metric_mismatch_fails_closed(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    async def _persist(**_kwargs):
        return {}

    fake = _FakeTEEClient(
        {"improvement_points": 999.0, "event_doc": {}},
        enclave_mode="required",
    )
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "required")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)

    with pytest.raises(attested_scoring.AttestedScoringError, match="promotion metric differs"):
        await attested_scoring.compare_promotion_metric(
            epoch_id=11,
            score_bundle={"evaluation_epoch": 11, "aggregates": {"mean_delta": 1.5}},
            expected_improvement_points=1.5,
            expected_event_doc={"improvement_basis": "legacy"},
        )


@pytest.mark.asyncio
async def test_required_host_mode_rejects_nonrequired_enclave(monkeypatch):
    fake = _FakeTEEClient({"ok": True}, enclave_mode="shadow")
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "required")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")
    with pytest.raises(attested_scoring.AttestedScoringError, match="not in required"):
        await attested_scoring.execute_attested_scoring_operation(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v1",
            epoch_id=9,
            payload={"scores": [1.0]},
        )
    assert fake.calls == ["health"]


@pytest.mark.asyncio
async def test_shadow_rejects_receipt_without_valid_nitro_attestation(monkeypatch):
    fake = _FakeTEEClient({"score": 1.5})
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setattr(
        attested_scoring,
        "verify_gateway_receipt_attestation",
        lambda **_kwargs: (False, {"error": "bad COSE signature"}),
    )

    result = await attested_scoring.execute_attested_scoring_operation(
        operation="benchmark_icp_score",
        purpose="research_lab.benchmark.v1",
        epoch_id=12,
        payload={"scores": [1.0, 2.0]},
    )

    assert result["status"] == "shadow_failed"
    assert result["error_type"] == "AttestedScoringError"


@pytest.mark.asyncio
async def test_required_rejects_zero_pcr0(monkeypatch):
    fake = _FakeTEEClient({"score": 1.5}, enclave_mode="required")
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "required")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")
    monkeypatch.setattr(
        attested_scoring,
        "verify_gateway_receipt_attestation",
        lambda *, receipt, expected_purpose, expected_epoch_id: (
            True,
            {
                "pcr0": "0" * 96,
                "purpose": expected_purpose,
                "epoch_id": expected_epoch_id,
                "enclave_pubkey": receipt["enclave_pubkey"],
            },
        ),
    )

    with pytest.raises(attested_scoring.AttestedScoringError, match="invalid PCR0"):
        await attested_scoring.execute_attested_scoring_operation(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v1",
            epoch_id=13,
            payload={"scores": [1.0, 2.0]},
        )


@pytest.mark.asyncio
async def test_required_mode_fails_before_enclave_without_receipt_persistence(monkeypatch):
    fake = _FakeTEEClient({"score": 1.5}, enclave_mode="required")
    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "required")
    monkeypatch.delenv(attested_scoring.PERSIST_RECEIPTS_ENV, raising=False)

    with pytest.raises(attested_scoring.AttestedScoringError, match="receipt persistence"):
        await attested_scoring.execute_attested_scoring_operation(
            operation="benchmark_icp_score",
            purpose="research_lab.benchmark.v1",
            epoch_id=14,
            payload={"scores": [1.0, 2.0]},
        )
    assert fake.calls == []


@pytest.mark.asyncio
async def test_score_bundle_receipt_is_linked_without_changing_bundle(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    score_bundle_hash = "sha256:" + "9" * 64
    score_bundle = {"score_bundle_hash": score_bundle_hash, "aggregate": 12.5}
    fake = _FakeTEEClient({"score_bundle": score_bundle})
    company_receipt = _signed_parent_receipt(
        purpose="research_lab.candidate_score.v1",
        job_id="company-score:1",
        epoch_id=21,
    )
    persisted = {}

    async def _persist(**kwargs):
        persisted.update(kwargs)
        return {}

    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")

    result = await attested_scoring.compare_score_bundle(
        epoch_id=21,
        purpose="research_lab.candidate_score.v1",
        build_payload={"input": "unchanged"},
        expected_score_bundle=score_bundle,
        parent_receipts=[company_receipt],
    )

    assert result["status"] == "matched"
    assert result["result"]["score_bundle"] == score_bundle
    assert fake.manifest["parent_receipt_hashes"] == [company_receipt["receipt_hash"]]
    assert persisted["artifact_links"] == [
        {
            "artifact_kind": "score_bundle",
            "artifact_ref": "score_bundle:" + "9" * 64,
            "artifact_hash": score_bundle_hash,
        }
    ]


@pytest.mark.asyncio
async def test_promotion_receipt_inherits_score_bundle_parent(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    score_bundle_hash = "sha256:" + "a" * 64
    score_bundle = {"score_bundle_hash": score_bundle_hash}
    score_receipt = _signed_parent_receipt(
        purpose="research_lab.candidate_score.v1",
        job_id="score:parent",
        epoch_id=22,
        evidence_roots={"score_bundle": score_bundle_hash},
        output_root=attested_scoring.sha256_bytes(
            attested_scoring.canonical_json_bytes({"score_bundle": score_bundle})
        ),
    )
    event_doc = {"improvement_basis": "paired"}
    fake = _FakeTEEClient({"improvement_points": 2.0, "event_doc": event_doc})
    persisted = {}

    async def _load_receipt(**_kwargs):
        return score_receipt

    async def _load_lineage(_receipt):
        return []

    async def _persist(**kwargs):
        persisted.update(kwargs)
        return {}

    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(attested_receipt_store, "load_receipt_for_artifact", _load_receipt)
    monkeypatch.setattr(attested_receipt_store, "load_attested_receipt_lineage", _load_lineage)
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")

    result = await attested_scoring.compare_promotion_metric(
        epoch_id=22,
        score_bundle=score_bundle,
        expected_improvement_points=2.0,
        expected_event_doc=event_doc,
    )

    assert result["status"] == "matched"
    assert fake.manifest["parent_receipt_hashes"] == [score_receipt["receipt_hash"]]
    assert persisted["artifact_links"][0]["artifact_kind"] == "promotion_metric"


@pytest.mark.asyncio
async def test_allocation_receipt_binds_complete_promotion_lineage(monkeypatch):
    from gateway.research_lab import attested_receipt_store
    from gateway.research_lab import store
    score_bundle_hash = "sha256:" + "b" * 64
    score_bundle_id = "score_bundle:" + "b" * 64
    score_bundle = {
        "score_bundle_hash": score_bundle_hash,
        "aggregates": {"mean_delta": 2.0},
        "private_holdout_gate": {
            "baseline_benchmark_hash": "sha256:" + "d" * 64,
        },
    }
    baseline_receipt = _signed_parent_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline:allocation-parent",
        epoch_id=22,
        evidence_roots={"baseline_score_summary": "sha256:" + "d" * 64},
    )
    score_receipt = _signed_parent_receipt(
        purpose="research_lab.candidate_score.v1",
        job_id="score:allocation-parent",
        epoch_id=22,
        parents=[baseline_receipt["receipt_hash"]],
        evidence_roots={
            "score_bundle": score_bundle_hash,
            "baseline_score_summary": "sha256:" + "d" * 64,
        },
        output_root=attested_scoring.sha256_bytes(
            attested_scoring.canonical_json_bytes({"score_bundle": score_bundle})
        ),
    )
    promotion_receipt = _signed_parent_receipt(
        purpose="research_lab.promotion_decision.v1",
        job_id="promotion:allocation-parent",
        epoch_id=22,
        parents=[score_receipt["receipt_hash"]],
        evidence_roots={
            "score_bundle": score_bundle_hash,
            "promotion_decision_status": attested_scoring.sha256_bytes(
                attested_scoring.canonical_json_bytes({"status": "promotion_passed"})
            ),
        },
    )
    allocation = {"allocation_hash": "sha256:" + "c" * 64}
    fake = _FakeTEEClient({"allocation": allocation})

    async def _load_receipt(**kwargs):
        if kwargs["artifact_kind"] == "promotion_decision":
            return promotion_receipt
        return None

    async def _load_lineage(_receipt):
        return [baseline_receipt, score_receipt]

    async def _persist(**_kwargs):
        return {}

    async def _select_one(_table, *, filters):
        assert filters == (("score_bundle_id", score_bundle_id),)
        return {"score_bundle_doc": score_bundle}

    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(attested_receipt_store, "load_receipt_for_artifact", _load_receipt)
    monkeypatch.setattr(attested_receipt_store, "load_attested_receipt_lineage", _load_lineage)
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setattr(store, "select_one", _select_one)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")

    result = await attested_scoring.compare_allocation(
        epoch_id=23,
        payload={
            "epoch": 23,
            "policy": {},
            "active_reimbursement_obligations": [],
            "active_champion_obligations": [{"score_bundle_id": score_bundle_id}],
        },
        expected_allocation=allocation,
    )

    submitted_payload = json.loads(bytes(fake.payload))
    assert result["status"] == "matched"
    assert result["lineage_complete"] is True
    assert fake.manifest["parent_receipt_hashes"] == [promotion_receipt["receipt_hash"]]
    assert submitted_payload["receipt_lineage_bindings"] == [
        {
            "score_bundle_id": score_bundle_id,
            "score_bundle_hash": score_bundle_hash,
            "receipt_hash": promotion_receipt["receipt_hash"],
            "receipt_purpose": "research_lab.promotion_decision.v1",
        }
    ]


@pytest.mark.asyncio
async def test_baseline_summary_receipt_inherits_company_scoring_receipt(monkeypatch):
    from gateway.research_lab import attested_receipt_store
    from research_lab.eval.baseline_summary import build_baseline_score_summary

    payload = {
        "artifact_manifest": {
            "model_artifact_hash": "sha256:" + "1" * 64,
            "manifest_hash": "sha256:" + "2" * 64,
            "manifest_uri": "s3://private/model.json",
            "git_commit_sha": "3" * 40,
            "image_digest": "repo@sha256:" + "4" * 64,
            "config_hash": "sha256:" + "5" * 64,
            "component_registry_version": "v1",
            "scoring_adapter_version": "v1",
            "build_id": "build-1",
        },
        "benchmark_date": "2026-07-10",
        "benchmark_attempt": 2,
        "rolling_window_hash": "sha256:" + "6" * 64,
        "evaluation_epoch": 24,
        "benchmark_items": [
            {
                "icp_ref": "icp:%s" % index,
                "icp_hash": "sha256:" + str(index + 1) * 64,
                "set_id": 1,
                "day_index": 0,
                "day_rank": index + 1,
            }
            for index in range(3)
        ],
        "per_icp_summaries": [
            {
                "icp_ref": "icp:%s" % index,
                "icp_hash": "sha256:" + str(index + 1) * 64,
                "score": float(index + 1),
            }
            for index in range(3)
        ],
        "public_icps_per_day": 1,
        "public_weak_per_day": 0,
        "public_total_icps": 1,
        "public_weak_total": 0,
        "retried": 0,
        "recovered": 0,
        "max_unresolved_icps": 0,
        "day_jump_points": None,
        "elapsed_seconds": 1.234,
    }
    expected = build_baseline_score_summary(**payload)
    company_receipt = _signed_parent_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline-company-score:1",
        epoch_id=24,
    )
    fake = _FakeTEEClient(expected)
    persisted = {}

    async def _persist(**kwargs):
        persisted.update(kwargs)
        return {}

    monkeypatch.setattr(attested_scoring, "tee_client", fake)
    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")

    outcome = await attested_scoring.compare_baseline_score_summary(
        epoch_id=24,
        build_payload=payload,
        expected_result=expected,
        parent_receipts=[company_receipt],
    )

    assert outcome["status"] == "matched"
    assert fake.manifest["parent_receipt_hashes"] == [company_receipt["receipt_hash"]]
    assert persisted["artifact_links"] == [
        {
            "artifact_kind": "benchmark_score_summary",
            "artifact_ref": "private_baseline:2026-07-10:2:" + "6" * 24,
            "artifact_hash": attested_scoring.sha256_bytes(
                attested_scoring.canonical_json_bytes(expected["score_summary_doc"])
            ),
        }
    ]


@pytest.mark.asyncio
async def test_final_artifact_links_reuse_the_verified_persisted_receipt(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    receipt = _signed_parent_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline-summary:1",
        epoch_id=25,
    )
    calls = []

    async def _persist(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr(attested_receipt_store, "persist_attested_receipt", _persist)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")
    links = [
        {
            "artifact_kind": "benchmark_bundle",
            "artifact_ref": "private_benchmark:" + "a" * 64,
            "artifact_hash": "sha256:" + "b" * 64,
        }
    ]

    status = await attested_scoring.persist_attested_outcome_artifact_links(
        {
            "status": "matched",
            "persistence_status": "persisted",
            "receipt": receipt,
            "pcr0": "c" * 96,
        },
        artifact_links=links,
    )

    assert status == "persisted"
    assert calls == [{"receipt": receipt, "pcr0": "c" * 96, "artifact_links": links}]


@pytest.mark.asyncio
async def test_artifact_lineage_resolver_returns_root_and_complete_graph(monkeypatch):
    from gateway.research_lab import attested_receipt_store

    ancestor = _signed_parent_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline-company:ancestor",
        epoch_id=26,
    )
    root = _signed_parent_receipt(
        purpose="research_lab.rebenchmark.v1",
        job_id="baseline-summary:root",
        epoch_id=26,
        parents=[ancestor["receipt_hash"]],
    )

    async def _load_receipt(**kwargs):
        assert kwargs == {
            "artifact_kind": "benchmark_score_summary",
            "artifact_ref": "private_benchmark:" + "d" * 64,
            "artifact_hash": "sha256:" + "e" * 64,
        }
        return root

    async def _load_lineage(receipt):
        assert receipt == root
        return [ancestor]

    monkeypatch.setattr(attested_receipt_store, "load_receipt_for_artifact", _load_receipt)
    monkeypatch.setattr(attested_receipt_store, "load_attested_receipt_lineage", _load_lineage)
    monkeypatch.setenv(attested_scoring.MODE_ENV, "shadow")
    monkeypatch.setenv(attested_scoring.PERSIST_RECEIPTS_ENV, "true")

    resolved, graph = await attested_scoring.resolve_attested_artifact_lineage(
        artifact_kind="benchmark_score_summary",
        artifact_ref="private_benchmark:" + "d" * 64,
        artifact_hash="sha256:" + "e" * 64,
    )

    assert resolved == root
    assert [item["receipt_hash"] for item in graph] == [
        ancestor["receipt_hash"],
        root["receipt_hash"],
    ]

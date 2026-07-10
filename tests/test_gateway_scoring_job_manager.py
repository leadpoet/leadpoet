import asyncio
import base64
import hashlib
import json
import threading
import time

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from gateway.tee.scoring_executor import (
    OP_BENCHMARK_ICP_SCORE,
    OP_BUILD_BASELINE_SCORE_SUMMARY,
    OP_RESEARCH_LAB_ALLOCATION,
    ScoringExecutionResult,
    execute_scoring_operation,
    configuration_hash,
)
from gateway.tee.scoring_job_manager import (
    JOB_SCHEMA_VERSION,
    ScoringJobError,
    ScoringJobManager,
)
from leadpoet_canonical.attested_receipts import validate_signed_receipt
from leadpoet_verifier.economics import allocate_research_lab_epoch
from research_lab.eval.evaluator import benchmark_icp_score_from_company_scores
from research_lab.eval.baseline_summary import build_baseline_score_summary


BUILD_HASH = "sha256:" + "a" * 64
CONFIG_HASH = "sha256:" + "b" * 64
COMMIT_SHA = "c" * 40


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _hash(value):
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _signing_material():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    ).hex()
    return private_key, public_key


def _manager(*, executor=None, mode="shadow", worker_count=1):
    private_key, public_key = _signing_material()
    kwargs = {}
    if executor is not None:
        kwargs["executor"] = executor
    return ScoringJobManager(
        build_manifest_hash=BUILD_HASH,
        commit_sha=COMMIT_SHA,
        signer=private_key.sign,
        public_key_supplier=lambda: public_key,
        attestation_supplier=lambda manifest: base64.b64encode(
            ("nitro-attestation:" + manifest["purpose"]).encode()
        ).decode(),
        config_hash_supplier=lambda: CONFIG_HASH,
        mode=mode,
        worker_count=worker_count,
        **kwargs,
    )


def _manifest(payload, *, job_id="score-job-1", operation=OP_BENCHMARK_ICP_SCORE, purpose="research_lab.benchmark.v1"):
    encoded = _canonical(payload)
    return {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": job_id,
        "operation": operation,
        "purpose": purpose,
        "epoch_id": 42,
        "commit_sha": COMMIT_SHA,
        "config_hash": CONFIG_HASH,
        "payload_sha256": _hash(encoded),
        "payload_size_bytes": len(encoded),
        "evidence_roots": {"provider_trace": "sha256:" + "d" * 64},
        "parent_receipt_hashes": [],
    }


def _upload_and_seal(manager, manifest, payload):
    encoded = _canonical(payload)
    manager.submit(manifest)
    midpoint = max(1, len(encoded) // 2)
    offset = 0
    for chunk in (encoded[:midpoint], encoded[midpoint:]):
        if not chunk:
            continue
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=offset,
            data_b64=base64.b64encode(chunk).decode(),
            chunk_sha256=_hash(chunk),
        )
        offset += len(chunk)
    manager.seal(manifest["job_id"])


def _wait_terminal(manager, job_id, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.status(job_id)
        if status["state"] in {"succeeded", "failed", "cancelled"}:
            return status
        time.sleep(0.01)
    raise AssertionError("scoring job did not reach a terminal state")


def _read_result(manager, job_id):
    output = bytearray()
    offset = 0
    while True:
        part = manager.result_chunk(job_id=job_id, offset=offset, max_bytes=7)
        chunk = base64.b64decode(part["data_b64"])
        assert part["chunk_sha256"] == _hash(chunk)
        output.extend(chunk)
        offset += len(chunk)
        if part["eof"]:
            assert _hash(bytes(output)) == part["result_sha256"]
            return json.loads(bytes(output))


def test_default_off_rejects_jobs_without_starting_worker():
    manager = _manager(mode="off")
    health = manager.health()
    assert health["mode"] == "off"
    assert health["enabled"] is False
    assert health["worker_alive"] is False
    with pytest.raises(ScoringJobError, match="disabled"):
        manager.submit(_manifest({"scores": [1.0]}))


def test_successful_job_is_chunked_nonblocking_and_signed():
    async def executor(operation, payload):
        await asyncio.sleep(0.02)
        return {"operation": operation, "scores": payload["scores"]}

    manager = _manager(executor=executor)
    payload = {"scores": [1.25, 2.5]}
    manifest = _manifest(payload)
    _upload_and_seal(manager, manifest, payload)
    assert manager.health()["mode"] == "shadow"
    status = _wait_terminal(manager, manifest["job_id"])
    assert status["state"] == "succeeded"
    assert "input" not in status
    assert "result" not in status
    assert _read_result(manager, manifest["job_id"]) == {
        "operation": OP_BENCHMARK_ICP_SCORE,
        "scores": [1.25, 2.5],
    }
    receipt = manager.receipt(manifest["job_id"])
    validate_signed_receipt(receipt)
    assert receipt["input_root"] == manifest["payload_sha256"]
    assert receipt["build_manifest_hash"] == BUILD_HASH
    assert receipt["config_hash"] == CONFIG_HASH
    assert base64.b64decode(receipt["attestation_document_b64"]).decode() == (
        "nitro-attestation:" + manifest["purpose"]
    )


def test_enclave_derived_evidence_is_signed_but_not_returned_as_business_output():
    provider_root = "sha256:" + "e" * 64

    def executor(_operation, payload):
        return ScoringExecutionResult(
            {"scores": payload["scores"]},
            {"provider_http_tape": provider_root},
        )

    manager = _manager(executor=executor)
    payload = {"scores": [3.5]}
    manifest = _manifest(payload, job_id="derived-evidence-job")
    _upload_and_seal(manager, manifest, payload)
    assert _wait_terminal(manager, manifest["job_id"])["state"] == "succeeded"
    assert _read_result(manager, manifest["job_id"]) == {"scores": [3.5]}
    receipt = manager.receipt(manifest["job_id"])
    validate_signed_receipt(receipt)
    assert receipt["evidence_roots"] == {
        "provider_http_tape": provider_root,
        "provider_trace": "sha256:" + "d" * 64,
    }


def test_blocked_scoring_worker_does_not_block_status_or_health():
    started = threading.Event()
    release = threading.Event()

    def executor(_operation, _payload):
        started.set()
        assert release.wait(2)
        return {"ok": True}

    manager = _manager(executor=executor)
    payload = {"scores": [1.0]}
    manifest = _manifest(payload, job_id="blocking-job")
    _upload_and_seal(manager, manifest, payload)
    assert started.wait(1)
    before = time.monotonic()
    assert manager.status("blocking-job")["state"] == "running"
    assert manager.health()["active_job_id"] == "blocking-job"
    assert time.monotonic() - before < 0.1
    release.set()
    assert _wait_terminal(manager, "blocking-job")["state"] == "succeeded"


def test_bounded_worker_pool_runs_jobs_concurrently_without_unbounded_threads():
    started = {"job-a": threading.Event(), "job-b": threading.Event()}
    release = threading.Event()

    def executor(_operation, payload):
        started[payload["job"]].set()
        assert release.wait(2)
        return {"job": payload["job"]}

    manager = _manager(executor=executor, worker_count=2)
    for job_id in ("job-a", "job-b"):
        payload = {"job": job_id}
        manifest = _manifest(payload, job_id=job_id)
        _upload_and_seal(manager, manifest, payload)
    assert started["job-a"].wait(1)
    assert started["job-b"].wait(1)
    health = manager.health()
    assert health["worker_count"] == 2
    assert health["active_job_ids"] == ["job-a", "job-b"]
    release.set()
    assert _wait_terminal(manager, "job-a")["state"] == "succeeded"
    assert _wait_terminal(manager, "job-b")["state"] == "succeeded"


def test_manifest_idempotency_config_and_chunk_integrity_fail_closed():
    manager = _manager()
    payload = {"scores": [1.0]}
    manifest = _manifest(payload)
    manager.submit(manifest)
    assert manager.submit(manifest)["manifest_hash"]

    conflicting = dict(manifest, payload_sha256="sha256:" + "e" * 64)
    with pytest.raises(ScoringJobError, match="different manifest"):
        manager.submit(conflicting)

    wrong_config = _manifest(payload, job_id="wrong-config")
    wrong_config["config_hash"] = "sha256:" + "f" * 64
    with pytest.raises(ScoringJobError, match="config hash"):
        manager.submit(wrong_config)

    wrong_commit = _manifest(payload, job_id="wrong-commit")
    wrong_commit["commit_sha"] = "d" * 40
    with pytest.raises(ScoringJobError, match="build identity"):
        manager.submit(wrong_commit)

    encoded = _canonical(payload)
    with pytest.raises(ScoringJobError, match="offset"):
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=1,
            data_b64=base64.b64encode(encoded).decode(),
            chunk_sha256=_hash(encoded),
        )
    with pytest.raises(ScoringJobError, match="hash mismatch"):
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=0,
            data_b64=base64.b64encode(encoded).decode(),
            chunk_sha256="sha256:" + "0" * 64,
        )


def test_failure_and_cancellation_do_not_return_business_results():
    def failing(_operation, _payload):
        raise RuntimeError("raw provider response must not escape")

    manager = _manager(executor=failing)
    payload = {"scores": [1.0]}
    manifest = _manifest(payload, job_id="failed-job")
    _upload_and_seal(manager, manifest, payload)
    status = _wait_terminal(manager, "failed-job")
    assert status["state"] == "failed"
    assert status["error_code"] == "execution_error"
    assert "provider response" not in json.dumps(status)
    receipt = manager.receipt("failed-job")
    validate_signed_receipt(receipt)
    assert receipt["status"] == "failed"
    with pytest.raises(ScoringJobError, match="result is not available"):
        manager.result_chunk(job_id="failed-job")

    cancelled = _manifest(payload, job_id="cancelled-job")
    manager.submit(cancelled)
    assert manager.cancel("cancelled-job")["state"] == "cancelled"


@pytest.mark.asyncio
async def test_executor_delegates_to_exact_existing_benchmark_function(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    scores = [9.0, 5.0, 2.5, 1.0, 0.5, 0.25]
    expected = benchmark_icp_score_from_company_scores(scores)
    result = await execute_scoring_operation(OP_BENCHMARK_ICP_SCORE, {"scores": scores})
    assert result == {"score": expected}


@pytest.mark.asyncio
async def test_executor_delegates_to_exact_existing_allocation_function():
    payload = {
        "epoch": 71,
        "policy": {"research_lab_emission_percent": 10, "reward_epochs": 20},
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
    }
    expected = allocate_research_lab_epoch(
        payload["epoch"],
        payload["policy"],
        payload["active_reimbursement_obligations"],
        payload["active_champion_obligations"],
    )
    result = await execute_scoring_operation(OP_RESEARCH_LAB_ALLOCATION, payload)
    assert isinstance(result, ScoringExecutionResult)
    assert result.result == {"allocation": expected}
    assert result.evidence_roots == {"allocation": expected["allocation_hash"]}


@pytest.mark.asyncio
async def test_executor_delegates_to_shared_baseline_summary_builder():
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
        "benchmark_attempt": 1,
        "rolling_window_hash": "sha256:" + "6" * 64,
        "evaluation_epoch": 42,
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
        "elapsed_seconds": 12.345,
    }

    result = await execute_scoring_operation(OP_BUILD_BASELINE_SCORE_SUMMARY, payload)

    expected = build_baseline_score_summary(**payload)
    assert isinstance(result, ScoringExecutionResult)
    assert result.result == expected
    assert result.evidence_roots["baseline_score_summary"] == _hash(
        _canonical(expected["score_summary_doc"])
    )


def test_config_hash_commits_behavior_and_secret_identity_without_exposing_secret(monkeypatch):
    from gateway.tee import scoring_executor

    monkeypatch.setattr(
        scoring_executor,
        "_manifest_configuration_env_names",
        lambda: ("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "OPENROUTER_API_KEY"),
    )
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "false")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-a")
    first = configuration_hash()
    snapshot = scoring_executor.configuration_snapshot()
    assert "secret-a" not in json.dumps(snapshot)
    assert snapshot["environment"]["OPENROUTER_API_KEY"]["value_sha256"].startswith("sha256:")
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-b")
    assert configuration_hash() != first
    second = configuration_hash()
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE", "true")
    assert configuration_hash() != second

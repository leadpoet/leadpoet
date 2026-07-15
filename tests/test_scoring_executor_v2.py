from __future__ import annotations

import base64
import httpx
import pytest

from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.scoring_executor import (
    OP_BENCHMARK_ICP_SCORE,
    execute_scoring_operation,
)
from gateway.tee.scoring_executor_v2 import (
    DEV_HYBRID_REQUEST_SCHEMA_VERSION,
    DEV_REPLAY_REQUEST_SCHEMA_VERSION,
    OP_DEV_HYBRID_V2,
    OP_DEV_REPLAY_V2,
    OP_PROVIDER_PREFLIGHT_V2,
    OP_RUN_MODEL_SANDBOX_V2,
    OP_SOURCE_ADD_LEG2_JUDGE_V2,
    PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
    SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
    ScoringExecutorV2,
)
from gateway.tee.model_sandbox_v2 import provider_evidence_tape_input_root
from gateway.tee.research_lab_runtime_config_v2 import (
    build_research_lab_execution_config,
)
from gateway.tee.source_bundle_v2 import build_source_bundle_v2
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import (
    build_transport_attempt,
    canonical_json,
    sha256_bytes,
    sha256_json,
)
from research_lab.eval import build_local_private_artifact_manifest
from research_lab.eval.dev_eval import compute_dev_set_hash, evaluate_dev
from research_lab.eval.private_runtime import canonicalize_private_model_icp
from research_lab.eval.provider_evidence_cache import (
    EVIDENCE_CACHE_SCHEMA_VERSION,
    icp_evidence_cache_key,
)
from research_lab.eval.snapshot_store import MODE_RECORD, MODE_REPLAY, ProviderSnapshotStore


HASH = "sha256:" + "a" * 64


def _model_catalog_evidence():
    runtime_catalog = build_source_add_runtime_catalog_v2([])
    result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": [],
        "provisioned_sources_hash": sha256_json([]),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": runtime_catalog,
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    root_hash = "sha256:" + "c" * 64
    evidence = {"result": result, "root_receipt_hash": root_hash}
    graph = {
        "root_receipt_hash": root_hash,
        "receipts": [
            {
                "receipt_hash": root_hash,
                "role": "gateway_coordinator",
                "purpose": "research_lab.source_add_catalog_snapshot.v2",
                "status": "succeeded",
                "output_root": sha256_json(result),
            }
        ],
    }
    return runtime_catalog, evidence, graph


def _seal_artifact(*, plaintext, job_id, purpose, artifact_kind):
    identity = sha256_json(
        {
            "job_id": job_id,
            "purpose": purpose,
            "artifact_kind": artifact_kind,
            "plaintext_hash": sha256_bytes(plaintext),
        }
    )
    return {
        "status": "sealed",
        "artifact_id": identity,
        "plaintext_hash": sha256_bytes(plaintext),
        "ciphertext_hash": sha256_json({"ciphertext": identity}),
        "artifact_kind": artifact_kind,
        "job_id": job_id,
        "purpose": purpose,
        "object_lock_mode": "COMPLIANCE",
        "retain_until": "2027-07-10T00:00:00Z",
        "encryption_context_hash": sha256_json({"aad": identity}),
        "persisted": False,
    }


@pytest.mark.asyncio
async def test_v2_preflight_reuses_existing_cache_and_failure_streak_logic(
    monkeypatch,
):
    from gateway.research_lab import provider_preflight

    calls = {"exa": 0, "scrapingdog": 0}

    def _healthy(provider):
        def probe(_timeout=None):
            calls[provider] += 1
            return provider_preflight.ProviderVerdict(
                provider=provider,
                healthy=True,
                status="healthy",
            )

        return probe

    monkeypatch.setitem(provider_preflight._PROBES, "exa", _healthy("exa"))
    monkeypatch.setitem(
        provider_preflight._PROBES,
        "scrapingdog",
        _healthy("scrapingdog"),
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail("healthy stub must not call network"),
        retry_policy_hashes={"exa": HASH, "scrapingdog": HASH},
    )
    payload = {
        "_v2_provider_credential_profile": "benchmark_model",
        "_v2_provider_credential_ref_hashes": {"exa": HASH},
        "schema_version": PROVIDER_PREFLIGHT_REQUEST_SCHEMA_VERSION,
        "scope_key": "scoring:worker-1",
        "force": False,
        "settings": {
            "enabled": True,
            "ttl_seconds": 600.0,
            "timeout_seconds": 12.0,
            "failure_streak_threshold": 3,
        },
    }
    try:
        for job_id in ("preflight-job-1", "preflight-job-2"):
            result = await executor(
                OP_PROVIDER_PREFLIGHT_V2,
                payload,
                ExecutionContextV2(
                    job_id=job_id,
                    purpose="research_lab.provider_preflight.v2",
                    epoch_id=0,
                    provider_credential_profile="benchmark_model",
                    provider_credential_ref_hashes={"exa": HASH},
                ),
            )
            assert result.output["healthy"] is True
    finally:
        executor.close()
    assert calls == {"exa": 1, "scrapingdog": 1}


@pytest.mark.asyncio
async def test_v2_adapter_calls_exact_existing_pure_scoring_function():
    payload = {"scores": [100.0, 80.0, 60.0, 40.0, 20.0, 1.0]}
    expected = await execute_scoring_operation(OP_BENCHMARK_ICP_SCORE, payload)
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"public_web": HASH},
    )
    try:
        result = await executor(
            OP_BENCHMARK_ICP_SCORE,
            payload,
            ExecutionContextV2(
                job_id="score-job-1",
                purpose="research_lab.benchmark.v2",
                epoch_id=24000,
            ),
        )
    finally:
        executor.close()
    assert dict(result.output) == dict(expected)


@pytest.mark.asyncio
async def test_v2_adapter_routes_provider_call_and_collects_terminal(monkeypatch):
    observed = []

    def _provider_execute(request):
        observed.append(request)
        body = b'{"ok":true}'
        attempt = build_transport_attempt(
            request_id="1" * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id=request["provider_id"],
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
        }

    async def _existing_scoring(_operation, _payload):
        async with httpx.AsyncClient(trust_env=False) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "model-1"},
            )
        return {"provider_result": response.json()}

    monkeypatch.setattr(
        "gateway.tee.scoring_executor_v2.execute_scoring_operation",
        _existing_scoring,
    )
    context = ExecutionContextV2(
        job_id="score-job-1",
        purpose="research_lab.candidate_score.v2",
        epoch_id=24000,
    )
    executor = ScoringExecutorV2(
        provider_execute=_provider_execute,
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        result = await executor(
            "qualification_company_scores",
            {},
            context,
        )
    finally:
        executor.close()
    assert result.output == {"provider_result": {"ok": True}}
    assert len(observed) == 1
    assert len(context.transport_attempts) == 1


@pytest.mark.asyncio
async def test_source_add_judge_runs_existing_logic_inside_measured_transport(monkeypatch):
    from gateway.research_lab.source_add_llm_judge import SourceAddJudgeVerdict

    observed = []

    def _provider_execute(request):
        observed.append(dict(request))
        body = b'{"choices":[{"message":{"content":"{}"}}]}'
        attempt = build_transport_attempt(
            request_id="2" * 32,
            logical_operation_id=request["logical_operation_id"],
            job_id=request["job_id"],
            purpose=request["purpose"],
            provider_id="openrouter",
            attempt_number=request["attempt_number"],
            method=request["method"],
            destination_host="openrouter.ai",
            destination_port=443,
            path_hash=HASH,
            nonsecret_headers_hash=HASH,
            body_hash=HASH,
            credential_ref_hash=HASH,
            retry_policy_hash=HASH,
            timeout_ms=request["timeout_ms"],
            started_at="2026-07-10T20:00:00Z",
            terminal_status="authenticated_response",
            http_status=200,
            response_hash=HASH,
            request_artifact_hash=HASH,
            response_artifact_hash=HASH,
            tls_peer_chain_hash=HASH,
            tls_protocol="TLSv1.3",
            failure_code=None,
            completed_at="2026-07-10T20:00:00Z",
        )
        return {
            "terminal_status": "authenticated_response",
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body_b64": base64.b64encode(body).decode("ascii"),
            "encrypted_request_artifact_id": HASH,
            "encrypted_artifact_id": HASH,
            "transport_attempt": attempt,
        }

    async def _judge(**kwargs):
        assert kwargs["api_key"] == "leadpoet-v2-brokered-credential"
        async with httpx.AsyncClient(trust_env=False) as client:
            await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json={"model": "openai/gpt-5.6-sol"},
            )
        return SourceAddJudgeVerdict(
            verdict="helped",
            confidence=0.9,
            source_used=True,
            adapter_id="adapter:test",
            registry_provider_id="test",
            evidence_summary="matched",
            reason_codes=("matched_api_usage",),
            model_id="openai/gpt-5.6-sol",
            provider_usage={"cost_usd": 0.01},
            raw_doc={"verdict": "helped"},
        )

    monkeypatch.setattr(
        "gateway.research_lab.source_add_llm_judge.judge_source_add_implementation",
        _judge,
    )
    provisioned_sources = [{"adapter_id": "adapter:test"}]
    catalog_output = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": provisioned_sources,
        "provisioned_sources_hash": sha256_json(provisioned_sources),
    }
    catalog_receipt_hash = "sha256:" + "b" * 64
    context = ExecutionContextV2(
        job_id="source-add-judge-1",
        purpose="research_lab.source_add_judge.v2",
        epoch_id=24000,
        parent_receipt_hashes=(catalog_receipt_hash,),
        provider_credential_profile="source_add_judge",
        provider_credential_ref_hashes={"openrouter": HASH},
        external_receipt_graphs=[
            {
                "root_receipt_hash": catalog_receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": catalog_receipt_hash,
                        "purpose": "research_lab.source_add_catalog_snapshot.v2",
                        "output_root": sha256_json(catalog_output),
                    }
                ],
            }
        ],
    )
    executor = ScoringExecutorV2(
        provider_execute=_provider_execute,
        retry_policy_hashes={"openrouter": HASH},
    )
    try:
        result = await executor(
            OP_SOURCE_ADD_LEG2_JUDGE_V2,
            {
                "_v2_provider_credential_profile": "source_add_judge",
                "_v2_provider_credential_ref_hashes": {"openrouter": HASH},
                "schema_version": SOURCE_ADD_JUDGE_REQUEST_SCHEMA_VERSION,
                "candidate": {"candidate_id": "candidate:1"},
                "score_bundle": {"score_bundle_hash": HASH},
                "provisioned_sources": provisioned_sources,
                "timeout_seconds": 180,
            },
            context,
        )
    finally:
        executor.close()

    assert result.output["verdict"]["verdict"] == "helped"
    assert result.output["verdict"]["judge_doc_hash"].startswith("sha256:")
    assert len(observed) == 1
    assert observed[0]["purpose"] == "research_lab.source_add_judge.v2"
    assert len(context.transport_attempts) == 1


@pytest.mark.asyncio
async def test_v2_adapter_routes_model_jobs_through_measured_sandbox():
    observed = {}
    runtime_catalog, catalog_evidence, catalog_graph = _model_catalog_evidence()
    commitments = {
        "model_artifact_hash": "sha256:" + "1" * 64,
        "model_manifest_hash": "sha256:" + "2" * 64,
        "source_bundle_hash": "sha256:" + "3" * 64,
        "runtime_config_hash": "sha256:" + "4" * 64,
        "input_hash": "sha256:" + "5" * 64,
        "provider_evidence_cache_hash": "sha256:" + "6" * 64,
        "provider_snapshot_archive_hash": sha256_json({}),
        "provider_snapshot_tree_hash": sha256_json({}),
        "provider_snapshot_manifest_hash": sha256_json({}),
        "provider_runtime_catalog_hash": runtime_catalog["catalog_hash"],
        "generated_provider_evidence_cache_hash": "sha256:" + "7" * 64,
        "trace_entries_hash": "sha256:" + "8" * 64,
        "output_hash": "sha256:" + "9" * 64,
    }

    class _Sandbox:
        def execute(self, payload, **kwargs):
            observed["payload"] = dict(payload)
            observed["kwargs"] = dict(kwargs)
            return {"output": {"companies": []}, **commitments}

    context = ExecutionContextV2(
        job_id="model-job-1",
        purpose="research_lab.private_model_run.v2",
        epoch_id=24000,
        external_receipt_graphs=[catalog_graph],
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=_Sandbox(),
        artifact_seal=_seal_artifact,
    )
    try:
        result = await executor(
            OP_RUN_MODEL_SANDBOX_V2,
            {
                "model_kind": "private",
                "environment": {},
                "provider_evidence_cache": {},
                "provider_evidence_cache_ref": "",
                "provider_evidence_mode": "live",
                "provider_snapshot_bundle": {},
                "provider_snapshot_tree_hash": "",
                "provider_snapshot_manifest_hash": "",
                "provider_cost_scope": HASH,
                "provider_cost_cap_microusd": 0,
                "provider_call_cap": 0,
                "provider_runtime_catalog": runtime_catalog,
                "provider_catalog_evidence": catalog_evidence,
            },
            context,
        )
    finally:
        executor.close()

    assert observed["payload"] == {
        "model_kind": "private",
        "environment": {},
        "provider_evidence_cache": {},
        "provider_evidence_cache_ref": "",
        "provider_evidence_mode": "live",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": HASH,
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": runtime_catalog,
        "provider_catalog_evidence": catalog_evidence,
    }
    assert observed["kwargs"]["job_id"] == "model-job-1"
    assert observed["kwargs"]["purpose"] == "research_lab.private_model_run.v2"
    assert observed["kwargs"]["retry_policy_hashes"] == {"openrouter": HASH}
    assert observed["kwargs"]["terminal_sink"] == context.record_transport
    assert result.output["output"] == {"companies": []}
    assert result.artifact_hashes[: len(commitments)] == tuple(commitments.values())
    assert [item["artifact_kind"] for item in result.output["sealed_artifacts"]] == [
        "model_output",
        "model_trace",
    ]


@pytest.mark.asyncio
async def test_v2_model_cache_requires_exact_tape_ancestry():
    runtime_catalog, catalog_evidence, catalog_graph = _model_catalog_evidence()
    cache_ref = "a" * 64
    cache_doc = {
        "schema_version": "1.1",
        "rolling_window_hash": "",
        "icp_ref": cache_ref,
        "utc_day": "2026-07-10",
        "entries": {},
    }
    cache_hash = sha256_json(cache_doc)
    calls = []

    class _Sandbox:
        def execute(self, payload, **kwargs):
            calls.append((dict(payload), dict(kwargs)))
            return {
                "model_artifact_hash": HASH,
                "model_manifest_hash": HASH,
                "source_bundle_hash": HASH,
                "runtime_config_hash": HASH,
                "input_hash": HASH,
                "provider_evidence_cache_hash": cache_hash,
                "provider_snapshot_archive_hash": sha256_json({}),
                "provider_snapshot_tree_hash": sha256_json({}),
                "provider_snapshot_manifest_hash": sha256_json({}),
                "provider_runtime_catalog_hash": runtime_catalog["catalog_hash"],
                "generated_provider_evidence_cache_hash": sha256_json({}),
                "trace_entries_hash": HASH,
                "output_hash": HASH,
                "output": [],
                "generated_provider_evidence_cache": {},
            }

    context = ExecutionContextV2(
        job_id="model-job-cache",
        purpose="research_lab.candidate_model_run.v2",
        epoch_id=24000,
        external_receipt_graphs=[catalog_graph],
    )
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: {},
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=_Sandbox(),
        artifact_seal=_seal_artifact,
    )
    payload = {
        "model_kind": "candidate",
        "environment": {},
        "provider_evidence_cache": cache_doc,
        "provider_evidence_cache_ref": cache_ref,
        "provider_evidence_mode": "cache_live",
        "provider_snapshot_bundle": {},
        "provider_snapshot_tree_hash": "",
        "provider_snapshot_manifest_hash": "",
        "provider_cost_scope": HASH,
        "provider_cost_cap_microusd": 0,
        "provider_call_cap": 0,
        "provider_runtime_catalog": runtime_catalog,
        "provider_catalog_evidence": catalog_evidence,
    }
    try:
        with pytest.raises(ValueError, match="measured tape ancestry"):
            await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
        context.external_receipt_graphs = [
            catalog_graph,
            {
                "receipts": [
                    {
                        "role": "gateway_scoring",
                        "purpose": "research_lab.provider_evidence_tape.v2",
                        "status": "succeeded",
                        "input_root": provider_evidence_tape_input_root(
                            cache_ref,
                            cache_hash,
                        ),
                        "output_root": cache_hash,
                    }
                ]
            }
        ]
        result = await executor(OP_RUN_MODEL_SANDBOX_V2, payload, context)
        assert result.output["output"] == []
        changed = {**cache_doc, "entries": {"f" * 64: {"status": 500}}}
        with pytest.raises(ValueError, match="measured tape ancestry"):
            await executor(
                OP_RUN_MODEL_SANDBOX_V2,
                {**payload, "provider_evidence_cache": changed},
                context,
            )
    finally:
        executor.close()
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_v2_dev_replay_preserves_score_and_adds_tree_commitments(tmp_path):
    source = tmp_path / "candidate-source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n",
        encoding="utf-8",
    )
    artifact = build_local_private_artifact_manifest(
        source_path=source,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/candidate@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/candidates/manifest.json",
        signature_ref="kms:signature",
        component_registry_version="1",
        scoring_adapter_version="1",
    )
    dev_items = [
        {
            "icp": {
                "icp_id": f"dev-{index}",
                "industry": "Software Development",
                "sub_industry": f"DevOps Tooling {index}",
                "product_service": "CI/CD platform",
                "geography": "United States",
                "country": "United States",
                "employee_count": "51-200",
                "intent_signals": [f"Hiring platform engineer {index}"],
            },
            "icp_ref": f"dev_set:{index}",
            "icp_hash": "sha256:" + str(index) * 64,
        }
        for index in range(1, 9)
    ]
    snapshot_root = tmp_path / "snapshot-set"
    record_store = ProviderSnapshotStore(str(snapshot_root), mode=MODE_RECORD)
    record_store.write_dev_icp_items(dev_items)
    manifest = record_store.build_manifest(
        icp_set_hash=compute_dev_set_hash(dev_items),
        dev_set_manifest={"manifest_type": "research_lab_dev_icp_set"},
        recorded_at="2026-07-10T00:00:00Z",
    )
    record_store.write_manifest(manifest)
    snapshot_bundle = build_source_bundle_v2(snapshot_root)
    source_bundle = build_source_bundle_v2(source)

    def companies(icp):
        return [
            {
                "company_name": "Acme " + str(icp["icp_id"]),
                "company_website": "https://acme.test",
                "industry": "Software Development",
                "sub_industry": "DevOps Tooling",
                "employee_count": "51-200",
                "country": "United States",
                "description": "CI/CD platform for DevOps teams",
                "intent_signals": [
                    {
                        "source": "job_board",
                        "description": "Hiring a DevOps engineer",
                        "url": "https://acme.test/jobs/1",
                        "date": "2026-05-01",
                    }
                ],
            }
        ]

    class _Sandbox:
        def __init__(self):
            self.calls = []
            self.hybrid_calls = []

        def execute_dev_replay(self, **kwargs):
            self.calls.append(dict(kwargs))
            return companies(kwargs["icp"])

        def execute_dev_provider_replay(self, **kwargs):
            self.hybrid_calls.append(dict(kwargs))
            return companies(kwargs["icp"])

    replay_store = ProviderSnapshotStore(str(snapshot_root), mode=MODE_REPLAY)

    async def direct_runner(icp, _context):
        return companies(icp)

    expected = await evaluate_dev(
        candidate_runner=direct_runner,
        dev_items=dev_items,
        snapshot_store=replay_store,
        run_label="candidate-node-1",
        install_replay_seams=False,
        require_manifest=True,
    )
    sandbox = _Sandbox()
    executor = ScoringExecutorV2(
        provider_execute=lambda _request: pytest.fail(
            "frozen dev replay must not call a provider"
        ),
        retry_policy_hashes={"openrouter": HASH},
        model_sandbox=sandbox,
        execution_config=build_research_lab_execution_config(
            environment={
                "RESEARCH_LAB_LOOP_DEV_EVAL_ICP_TIMEOUT_SECONDS": "30",
                "RESEARCH_LAB_LOOP_DEV_EVAL_TIMEOUT_SECONDS": "60",
            }
        ),
    )
    cohort_hash = "sha256:" + "d" * 64
    payload = {
        "schema_version": DEV_REPLAY_REQUEST_SCHEMA_VERSION,
        "artifact": artifact,
        "source_bundle": source_bundle,
        "snapshot_bundle": snapshot_bundle,
        "snapshot_tree_hash": snapshot_bundle["source_tree_hash"],
        "snapshot_manifest_hash": manifest["manifest_hash"],
        "module_name": "research_lab_adapter",
        "callable_name": "run_icp",
        "environment": {},
        "credential_env_names": [],
        "run_label": "candidate-node-1",
        "cohort_hash": cohort_hash,
        "miss_policy": "strict",
        "per_icp_timeout_seconds": 30,
        "total_timeout_seconds": 60,
    }
    caches = {}
    cache_graphs = []
    for item in dev_items:
        canonical_icp = canonicalize_private_model_icp(item["icp"])
        cache_ref = icp_evidence_cache_key(canonical_icp)
        cache = {
            "schema_version": EVIDENCE_CACHE_SCHEMA_VERSION,
            "icp_ref": cache_ref,
            "entries": {},
        }
        cache_hash = sha256_json(cache)
        receipt_hash = sha256_json({"cache_ref": cache_ref})
        caches[cache_ref] = cache
        cache_graphs.append(
            {
                "root_receipt_hash": receipt_hash,
                "receipts": [
                    {
                        "receipt_hash": receipt_hash,
                        "role": "gateway_scoring",
                        "purpose": "research_lab.provider_evidence_tape.v2",
                        "status": "succeeded",
                        "input_root": provider_evidence_tape_input_root(
                            cache_ref, cache_hash
                        ),
                        "output_root": cache_hash,
                    }
                ],
            }
        )
    overlay_hash = sha256_json(caches)
    hybrid_cohort_hash = "sha256:" + "e" * 64
    hybrid_payload = {
        **payload,
        "schema_version": DEV_HYBRID_REQUEST_SCHEMA_VERSION,
        "cohort_hash": hybrid_cohort_hash,
        "provider_evidence_caches": caches,
        "overlay_hash": overlay_hash,
    }
    try:
        measured = await executor(
            OP_DEV_REPLAY_V2,
            payload,
            ExecutionContextV2(
                job_id="dev-replay-job-1",
                purpose="research_lab.candidate_test.v2",
                epoch_id=24000,
            ),
        )
        hybrid_measured = await executor(
            OP_DEV_HYBRID_V2,
            hybrid_payload,
            ExecutionContextV2(
                job_id="dev-hybrid-job-1",
                purpose="research_lab.candidate_hybrid_test.v2",
                epoch_id=24000,
                external_receipt_graphs=cache_graphs,
            ),
        )
    finally:
        executor.close()

    expected_output = {
        **expected.to_dict(),
        "evaluation_mode": "replay",
        "overlay_hash": sha256_json({}),
        "cohort_hash": cohort_hash,
    }
    expected_output["score_commitment"] = sha256_json(
        {
            "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
            "dev_score_version": expected.dev_score_version,
            "dev_set_hash": expected.dev_set_hash,
            "snapshot_manifest_hash": expected.snapshot_manifest_hash,
            "miss_policy": expected.miss_policy,
            "evaluation_mode": "replay",
            "overlay_hash": sha256_json({}),
            "cohort_hash": cohort_hash,
        }
    )
    assert canonical_json(measured.output) == canonical_json(expected_output)
    assert measured.output["aggregate_dev_score"] == expected.aggregate_dev_score
    assert cohort_hash in measured.artifact_hashes
    assert len(sandbox.calls) == len(dev_items)
    assert all(call["timeout_seconds"] == 30 for call in sandbox.calls)
    assert all(call["miss_policy"] == "strict" for call in sandbox.calls)
    assert hybrid_measured.output["aggregate_dev_score"] == (
        expected.aggregate_dev_score
    )
    assert hybrid_measured.output["evaluation_mode"] == "hybrid"
    assert hybrid_measured.output["overlay_hash"] == overlay_hash
    assert hybrid_measured.output["cohort_hash"] == hybrid_cohort_hash
    assert hybrid_measured.output["score_commitment"] == sha256_json(
        {
            "schema_version": "research_lab.git_tree_dev_score_commitment.v1",
            "dev_score_version": expected.dev_score_version,
            "dev_set_hash": expected.dev_set_hash,
            "snapshot_manifest_hash": expected.snapshot_manifest_hash,
            "miss_policy": expected.miss_policy,
            "evaluation_mode": "hybrid",
            "overlay_hash": overlay_hash,
            "cohort_hash": hybrid_cohort_hash,
        }
    )
    assert len(sandbox.hybrid_calls) == len(dev_items)
    assert hybrid_cohort_hash in hybrid_measured.artifact_hashes
    assert overlay_hash in hybrid_measured.artifact_hashes

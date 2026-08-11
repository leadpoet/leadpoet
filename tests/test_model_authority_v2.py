from __future__ import annotations

import asyncio
import json
import subprocess
from types import SimpleNamespace

import pytest

from gateway.research_lab import model_authority_v2
from gateway.research_lab.attested_scoring_v2 import AttestedScoringV2Error
from gateway.research_lab.model_authority_v2 import (
    AttestedPrivateModelRunnerV2Error,
    AttestedPrivateModelRunnerV2,
    V2_PROVIDER_PROFILE_ENV,
)
from gateway.utils.tee_artifact_store_v2 import TEEArtifactStoreV2Error
from gateway.research_lab.tee_protocol import ResearchLabTeeProtocolError
from gateway.tee.model_sandbox_v2 import provider_evidence_tape_input_root
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.eval import DockerPrivateModelSpec, build_local_private_artifact_manifest
from research_lab.eval.private_runtime import (
    PROVIDER_COST_EVALUATION_SCOPE_ENV,
    begin_attested_receipt_hash_collection,
    begin_incontainer_trace_collection,
    compute_private_source_tree_hash,
    end_attested_receipt_hash_collection,
    end_incontainer_trace_collection,
)
from tests.private_model_artifact_fixtures import install_reviewed_consumer_snapshot


def _artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n",
        encoding="utf-8",
    )
    install_reviewed_consumer_snapshot(source)
    return build_local_private_artifact_manifest(
        source_path=source,
        git_commit_sha="a" * 40,
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "b" * 64
        ),
        manifest_uri="s3://private/manifests/current.json",
        signature_ref="kms:signature",
        component_registry_version="1",
        scoring_adapter_version="1",
    )


def test_measured_environment_excludes_legacy_host_evidence_proxy():
    spec = DockerPrivateModelSpec(
        image_digest=(
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
            + "b" * 64
        ),
        extra_env={
            "RESEARCH_LAB_EVIDENCE_PROXY_URL": "http://127.0.0.1:8765",
            "RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD": "1",
        },
    )

    measured = model_authority_v2._measured_environment(spec)

    assert "RESEARCH_LAB_EVIDENCE_PROXY_URL" not in measured
    assert measured["RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD"] == "1"


def test_source_bundle_uses_exact_signed_repo_when_image_is_runtime_subset(
    tmp_path, monkeypatch
):
    source = tmp_path / "private-repo"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n",
        encoding="utf-8",
    )
    install_reviewed_consumer_snapshot(source)
    (source / "repo-only.txt").write_text("signed full source\n", encoding="utf-8")
    subprocess.run(["git", "init", "--quiet"], cwd=source, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Research Lab Test"],
        cwd=source,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@leadpoet.local"],
        cwd=source,
        check=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=source, check=True)
    subprocess.run(
        ["git", "commit", "--quiet", "-m", "signed source"],
        cwd=source,
        check=True,
    )
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    artifact = model_authority_v2.PrivateModelArtifactManifest.from_mapping(
        build_local_private_artifact_manifest(
            source_path=source,
            git_commit_sha=commit_sha,
            image_digest=(
                "123456789012.dkr.ecr.us-east-1.amazonaws.com/private@sha256:"
                + "b" * 64
            ),
            manifest_uri="s3://private/manifests/current.json",
            signature_ref="kms:signature",
            component_registry_version="1",
            scoring_adapter_version="1",
        )
    )

    def extract_runtime_subset(*, image_digest, source_dir, timeout_seconds):
        assert image_digest == artifact.image_digest
        assert timeout_seconds >= 120
        source_dir.mkdir(parents=True)
        (source_dir / "research_lab_adapter.py").write_text(
            "def run_icp(icp, context):\n    return []\n",
            encoding="utf-8",
        )
        return compute_private_source_tree_hash(source_dir), [
            "research_lab_adapter.py"
        ]

    monkeypatch.setattr(
        model_authority_v2,
        "_extract_parent_image_source",
        extract_runtime_subset,
    )
    monkeypatch.setenv("RESEARCH_LAB_PRIVATE_REPO_URL", str(source))
    model_authority_v2._SOURCE_BUNDLE_CACHE.clear()

    bundle = model_authority_v2._source_bundle_for_artifact(
        artifact,
        timeout_seconds=120,
    )

    assert bundle["source_tree_hash"] == artifact.model_artifact_hash
    restored = tmp_path / "restored"
    extract_source_bundle_v2(
        bundle,
        destination=restored,
        expected_source_tree_hash=artifact.model_artifact_hash,
    )
    assert (restored / "repo-only.txt").read_text(encoding="utf-8") == (
        "signed full source\n"
    )


def _catalog_outcome(rows=()):
    provisioned_sources = [dict(item) for item in rows]
    runtime_catalog = build_source_add_runtime_catalog_v2(provisioned_sources)
    result = {
        "schema_version": "leadpoet.source_add_catalog_snapshot.v2",
        "provisioned_sources": provisioned_sources,
        "provisioned_sources_hash": sha256_json(provisioned_sources),
        "private_registry_rows": [],
        "private_registry_rows_hash": sha256_json([]),
        "runtime_catalog": runtime_catalog,
        "runtime_catalog_hash": runtime_catalog["catalog_hash"],
    }
    execution_receipt = {
        "receipt_hash": "sha256:" + "c" * 64,
        "role": "gateway_coordinator",
        "purpose": "research_lab.source_add_catalog_snapshot.v2",
        "status": "succeeded",
        "output_root": sha256_json(result),
    }
    artifact_receipt = {
        "receipt_hash": "sha256:" + "d" * 64,
        "role": "gateway_coordinator",
        "purpose": "leadpoet.artifact_persistence.v2",
        "status": "succeeded",
        "output_root": "sha256:" + "d" * 64,
    }
    return {
        "result": result,
        "receipt": artifact_receipt,
        "execution_receipt": execution_receipt,
        "execution_receipt_graph": {
            "root_receipt_hash": execution_receipt["receipt_hash"],
            "receipts": [execution_receipt],
        },
        "receipt_graph": {
            "root_receipt_hash": artifact_receipt["receipt_hash"],
            "receipts": [execution_receipt, artifact_receipt],
        },
    }


async def _load_empty_catalog(*, epoch_id):
    assert epoch_id >= 0
    return _catalog_outcome()


@pytest.mark.asyncio
async def test_catalog_snapshot_load_is_singleflight_across_shared_runner_clones(
    tmp_path,
):
    artifact = _artifact(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def load_catalog(*, epoch_id):
        calls.append(epoch_id)
        started.set()
        await release.wait()
        return _catalog_outcome()

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        epoch_id=24001,
        catalog_snapshot_loader=load_catalog,
    )
    clone = runner.with_spec(runner.spec)

    loads = [
        asyncio.create_task(candidate._load_catalog_snapshot(epoch_id=24001))
        for candidate in (runner, clone, runner, clone)
    ]
    await started.wait()
    await asyncio.sleep(0)
    assert calls == [24001]

    release.set()
    outcomes = await asyncio.gather(*loads)
    assert all(outcome is outcomes[0] for outcome in outcomes)
    assert calls == [24001]


@pytest.mark.asyncio
async def test_catalog_snapshot_failed_load_is_not_cached(tmp_path):
    artifact = _artifact(tmp_path)
    calls = []

    async def load_catalog(*, epoch_id):
        calls.append(epoch_id)
        if len(calls) == 1:
            raise RuntimeError("catalog unavailable")
        return _catalog_outcome()

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        epoch_id=24001,
        catalog_snapshot_loader=load_catalog,
    )

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        await runner._load_catalog_snapshot(epoch_id=24001)
    outcome = await runner._load_catalog_snapshot(epoch_id=24001)

    assert outcome == _catalog_outcome()
    assert calls == [24001, 24001]


@pytest.mark.asyncio
async def test_catalog_snapshot_failed_load_reaches_followers_and_allows_retry(
    tmp_path,
):
    artifact = _artifact(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def load_catalog(*, epoch_id):
        calls.append(epoch_id)
        if len(calls) == 1:
            started.set()
            await release.wait()
            raise RuntimeError("catalog unavailable")
        return _catalog_outcome()

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        epoch_id=24001,
        catalog_snapshot_loader=load_catalog,
    )
    clone = runner.with_spec(runner.spec)
    loads = [
        asyncio.create_task(candidate._load_catalog_snapshot(epoch_id=24001))
        for candidate in (runner, clone, runner, clone)
    ]
    await started.wait()
    await asyncio.sleep(0)
    release.set()

    outcomes = await asyncio.gather(*loads, return_exceptions=True)
    assert all(
        isinstance(outcome, RuntimeError)
        and str(outcome) == "catalog unavailable"
        for outcome in outcomes
    )
    assert calls == [24001]

    assert await clone._load_catalog_snapshot(epoch_id=24001) == _catalog_outcome()
    assert calls == [24001, 24001]


@pytest.mark.asyncio
async def test_measured_execution_failure_preserves_private_runner_contract(
    tmp_path, monkeypatch
):
    artifact = _artifact(tmp_path)
    source_bundle = {
        "schema_version": "leadpoet.private_source_bundle.v2",
        "archive_sha256": "sha256:" + "2" * 64,
        "source_tree_hash": artifact["model_artifact_hash"],
        "archive_size_bytes": 1,
        "archive_b64": "AA==",
    }
    monkeypatch.setattr(
        model_authority_v2,
        "_source_bundle_for_artifact",
        lambda *_args, **_kwargs: dict(source_bundle),
    )

    async def execute(**_kwargs):
        raise AttestedScoringV2Error(
            "V2 scoring failed closed: execution_modelsandboxv2error"
        )

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        epoch_id=24001,
        execute=execute,
        catalog_snapshot_loader=_load_empty_catalog,
    )

    with pytest.raises(
        AttestedPrivateModelRunnerV2Error,
        match="execution_modelsandboxv2error",
    ):
        await runner(
            {"industry": "Software", "intent_signal": "Hiring"},
            {"mode": "private_baseline"},
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attempts,marked",
    [
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "terminal_status": "transport_failure",
                }
            ],
            True,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "terminal_status": "transport_failure",
                },
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 1,
                    "terminal_status": "authenticated_response",
                },
            ],
            False,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "provider_id": "scrapingdog",
                    "terminal_status": "authenticated_response",
                    "http_status": 400,
                }
            ],
            True,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "provider_id": "public_web",
                    "terminal_status": "authenticated_response",
                    "http_status": 403,
                }
            ],
            False,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "provider_id": "exa",
                    "terminal_status": "authenticated_response",
                    "http_status": 429,
                }
            ],
            True,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "provider_id": "public_web",
                    "terminal_status": "authenticated_response",
                    "http_status": 503,
                }
            ],
            True,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 0,
                    "provider_id": "public_web",
                    "terminal_status": "authenticated_response",
                    "http_status": 503,
                },
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": 1,
                    "provider_id": "public_web",
                    "terminal_status": "authenticated_response",
                    "http_status": 200,
                },
            ],
            False,
        ),
        (
            [
                {
                    "logical_operation_id": "provider-op-a",
                    "attempt_number": True,
                    "provider_id": "public_web",
                    "terminal_status": "authenticated_response",
                    "http_status": 503,
                }
            ],
            False,
        ),
        ([], False),
    ],
)
async def test_provider_client_failure_marks_only_latest_attested_transport_failure(
    attempts, marked
):
    async def fail_measured_operation(**_kwargs):
        raise AttestedScoringV2Error(
            "V2 scoring failed closed: execution_providerclientv2error",
            authority={"transport_attempts": attempts},
        )

    runner = object.__new__(AttestedPrivateModelRunnerV2)
    runner.spec = SimpleNamespace(timeout_seconds=1800)
    runner._execute_operation = fail_measured_operation
    with pytest.raises(AttestedPrivateModelRunnerV2Error) as captured:
        await runner._invoke_operation(operation="run_icp")

    marker = model_authority_v2.RETRYABLE_ATTESTED_PROVIDER_TRANSPORT_MARKER
    assert (marker in str(captured.value)) is marked
    assert isinstance(captured.value.__cause__, AttestedScoringV2Error)


async def test_measured_model_invocation_timeout_cancels_complete_operation(monkeypatch):
    cancelled = asyncio.Event()

    async def never_complete(**_kwargs):
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    runner = object.__new__(AttestedPrivateModelRunnerV2)
    runner.spec = SimpleNamespace(timeout_seconds=1800)
    runner._execute_operation = never_complete
    monkeypatch.setattr(
        model_authority_v2,
        "_model_invocation_timeout_seconds",
        lambda _timeout: 0.01,
    )

    with pytest.raises(
        AttestedPrivateModelRunnerV2Error,
        match="measured model invocation timed out",
    ) as captured:
        await runner._invoke_operation(operation="run_icp")

    assert cancelled.is_set()
    assert isinstance(captured.value.__cause__, asyncio.TimeoutError)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "retryable"),
    (
        ("enclave rejected artifact persistence: unexpected_eof", True),
        ("encrypted artifact document hash mismatch", False),
    ),
)
async def test_measured_model_artifact_failure_stays_in_runner_contract(
    message,
    retryable,
):
    async def fail_artifact_persistence(**_kwargs):
        raise TEEArtifactStoreV2Error(message)

    runner = object.__new__(AttestedPrivateModelRunnerV2)
    runner.spec = SimpleNamespace(timeout_seconds=1800)
    runner._execute_operation = fail_artifact_persistence

    with pytest.raises(AttestedPrivateModelRunnerV2Error) as captured:
        await runner._invoke_operation(operation="run_icp")

    marker = (
        model_authority_v2.RETRYABLE_ATTESTED_ARTIFACT_PERSISTENCE_MARKER
    )
    assert (marker in str(captured.value)) is retryable
    assert isinstance(captured.value.__cause__, TEEArtifactStoreV2Error)


def test_measured_model_invocation_budget_covers_attested_persistence():
    model_timeout = 1800.0
    assert model_authority_v2._MODEL_INVOCATION_ATTESTED_PHASES == 2.0
    attested_phase_timeout = (
        model_timeout
        + model_authority_v2._MODEL_INVOCATION_TIMEOUT_OVERHEAD_SECONDS
    )

    assert model_authority_v2._model_invocation_timeout_seconds(model_timeout) == (
        attested_phase_timeout * 2.0
        + model_authority_v2._MODEL_INVOCATION_PERSISTENCE_RESERVE_SECONDS
    )


def _ready_adapter_metadata() -> dict:
    routing_catalog = {"schema_version": 1}
    routing_policy = {"schema_version": 1}
    runtime_catalog = {
        "schema_version": 1,
        "tools": [
            {"tool_id": tool_id}
            for tool_id in (
                "candidate.backlog",
                "candidate.registry_feed",
                "candidate.jobs_feed",
                "candidate.deepline_firmographic",
                "candidate.model_semantic",
                "intent.existing_evidence",
                "intent.jobs_feed",
                "intent.company_search",
                "intent.first_party",
                "intent.newsroom",
            )
        ],
    }
    runtime_policy = {"schema_version": 1}
    return {
        "adapter_version": "sourcing-model-research-lab-adapter:v3",
        "component_registry_version": "sourcing-model-components:v2",
        "capability_contract_version": "sourcing-model-runtime-capabilities:v2",
        "runtime_capabilities": [
            "deadline",
            "emit",
            "http_fetch",
            "probe_origin",
            "resolve_host",
        ],
        "resilience_policy_version": "sourcing-model-resilience:v1",
        "firmographic_discovery": {
            "firmographic_policy_version": "sourcing-model-firmographic-discovery:v1"
        },
        "industry_taxonomy": {
            "taxonomy_content_hash": "sha256:" + "d" * 64
        },
        "routing": {
            "compiler_version": "routing-compiler-v2",
            "catalog": routing_catalog,
            "catalog_sha256": sha256_json(routing_catalog).removeprefix("sha256:"),
            "policy": routing_policy,
            "policy_sha256": sha256_json(routing_policy).removeprefix("sha256:"),
            "intent_sources": ["company_site", "job_listing", "news"],
            "source_add_requires_manifest_sha256": True,
            "private_bindings_exposed": False,
        },
        "runtime_routing": {
            "compiler_version": "routing-compiler-v2",
            "catalog": runtime_catalog,
            "catalog_sha256": sha256_json(runtime_catalog).removeprefix("sha256:"),
            "policy": runtime_policy,
            "policy_sha256": sha256_json(runtime_policy).removeprefix("sha256:"),
            "candidate_tool_lanes": {
                "candidate.backlog": "backlog",
                "candidate.registry_feed": "registry_signal",
                "candidate.jobs_feed": "jobs_signal",
                "candidate.deepline_firmographic": "deepline_firmographic",
                "candidate.model_semantic": "model_semantic",
            },
            "intent_tool_tiers": {
                "intent.existing_evidence": "fused",
                "intent.jobs_feed": "jobs_feed",
                "intent.company_search": "company_search",
                "intent.first_party": "first_party",
                "intent.newsroom": "newsroom",
            },
            "private_bindings_exposed": False,
        },
        "component_registry": {
            "source_router": {
                "strategy_options": ["company_site", "job_listing", "news"],
            }
        },
    }


def _runtime_receipt(runtime_cap_seconds: float) -> dict:
    return {
        "kind": "sourcing_branch_receipt",
        "runtime_cap_seconds": runtime_cap_seconds,
        "capability_contract": {
            "host_registered": [
                "deadline",
                "emit",
                "probe_origin",
                "resolve_host",
            ],
        },
        "industry_taxonomy": {
            "taxonomy_content_hash": "sha256:" + "d" * 64,
        },
        "firmographic_discovery": {"plan": {"target": 5}},
        "branches": [
            {
                "source": "news",
                "compiled_source": "news",
                "source_override": False,
                "route_tool_ids": ["intent.news", "intent.company_site"],
                "route_sources": ["news", "company_site"],
                "route_plan_sha256": "5" * 64,
                "route_policy_sha256": "6" * 64,
                "route_catalog_sha256": "7" * 64,
                "route_context_sha256": "8" * 64,
            }
        ],
    }


@pytest.mark.asyncio
async def test_legacy_protocol_cannot_select_host_model_runner(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    calls = []

    class HostRunner:
        def __init__(self, spec):
            self.spec = spec

        def __call__(self, icp, context):
            calls.append((dict(icp), dict(context)))
            return [{"company_name": "Legacy Host Result"}]

        def metadata(self):
            return {"runtime": "host", "image_digest": self.spec.image_digest}

    monkeypatch.setenv("RESEARCH_LAB_TEE_PROTOCOL", "legacy_v1")
    monkeypatch.setattr(model_authority_v2, "DockerPrivateModelRunner", HostRunner)
    with pytest.raises(ResearchLabTeeProtocolError, match="V1 authority is retired"):
        AttestedPrivateModelRunnerV2(
            artifact=artifact,
            spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
            model_kind="candidate",
            worker_index=4,
            epoch_id=24001,
        )
    assert calls == []


@pytest.mark.asyncio
async def test_attested_model_runner_preserves_inputs_but_never_sends_parent_credentials(
    tmp_path, monkeypatch
):
    artifact = _artifact(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    icp = {"industry": "Software", "intent_signal": "Hiring"}
    from research_lab.eval.private_runtime import canonicalize_private_model_icp
    from research_lab.eval.provider_evidence_cache import icp_evidence_cache_key

    canonical_icp = canonicalize_private_model_icp(icp)
    cache_doc = {
        "schema_version": "1.1",
        "rolling_window_hash": "sha256:" + "1" * 64,
        "icp_ref": "icp-1",
        "utc_day": "2026-07-10",
        "entries": {},
    }
    (cache_dir / (icp_evidence_cache_key(canonical_icp) + ".json")).write_text(
        json.dumps(cache_doc),
        encoding="utf-8",
    )
    source_bundle = {
        "schema_version": "leadpoet.private_source_bundle.v2",
        "archive_sha256": "sha256:" + "2" * 64,
        "source_tree_hash": artifact["model_artifact_hash"],
        "archive_size_bytes": 1,
        "archive_b64": "AA==",
    }
    monkeypatch.setattr(
        model_authority_v2,
        "_source_bundle_for_artifact",
        lambda *_args, **_kwargs: dict(source_bundle),
    )
    cache_ref = icp_evidence_cache_key(canonical_icp)
    cache_hash = sha256_json(cache_doc)
    tape_graph = {
        "root_receipt_hash": "sha256:" + "5" * 64,
        "receipts": [
            {
                "receipt_hash": "sha256:" + "5" * 64,
                "role": "gateway_scoring",
                "purpose": "research_lab.provider_evidence_tape.v2",
                "status": "succeeded",
                "input_root": provider_evidence_tape_input_root(
                    cache_ref,
                    cache_hash,
                ),
                "output_root": cache_hash,
            }
        ],
    }
    tape_lineage_graph = {
        "root_receipt_hash": "sha256:" + "b" * 64,
        "receipts": [
            {
                "receipt_hash": "sha256:" + "b" * 64,
                "role": "gateway_coordinator",
                "purpose": "leadpoet.artifact_persistence.v2",
                "status": "succeeded",
                "parent_receipt_hashes": [tape_graph["root_receipt_hash"]],
            }
        ],
    }

    async def load_tape_graphs(**kwargs):
        assert kwargs == {"cache_ref": cache_ref, "cache_hash": cache_hash}
        return (dict(tape_graph), dict(tape_lineage_graph))

    monkeypatch.setattr(
        model_authority_v2,
        "_load_provider_evidence_tape_graphs",
        load_tape_graphs,
    )
    observed = []

    async def execute(**kwargs):
        observed.append(kwargs)
        payload = kwargs["payload"]
        output = [{"company_name": "Measured Co"}]
        return {
            "result": {
                "schema_version": "leadpoet.model_sandbox_result.v2",
                "model_kind": "candidate",
                "operation": "run_icp",
                "model_artifact_hash": artifact["model_artifact_hash"],
                "model_manifest_hash": artifact["manifest_hash"],
                "compatibility_image_digest": artifact["image_digest"],
                "source_bundle_hash": source_bundle["archive_sha256"],
                "runtime_config_hash": "sha256:" + "3" * 64,
                "input_hash": sha256_json(payload["input"]),
                "provider_evidence_cache_hash": sha256_json(cache_doc),
                "provider_evidence_cache_ref": cache_ref,
                "provider_evidence_mode": payload["provider_evidence_mode"],
                "provider_snapshot_archive_hash": sha256_json({}),
                "provider_snapshot_tree_hash": sha256_json({}),
                "provider_snapshot_manifest_hash": sha256_json({}),
                "provider_cost_cap_microusd": 0,
                "provider_call_cap": 0,
                "provider_runtime_catalog_hash": observed[0]["payload"][
                    "provider_runtime_catalog"
                ]["catalog_hash"],
                "generated_provider_evidence_cache_hash": sha256_json({}),
                "trace_entries_hash": sha256_json(
                    [_runtime_receipt(1500.0), {"provider": "exa"}]
                ),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [_runtime_receipt(1500.0), {"provider": "exa"}],
                "generated_provider_evidence_cache": {},
            },
            "receipt": {"receipt_hash": "sha256:" + "4" * 64},
        }

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(
            image_digest=artifact["image_digest"],
            timeout_seconds=1800,
            env_passthrough=("EXA_API_KEY",),
            extra_env={
                "EXA_API_KEY": "parent-secret-value",
                PROVIDER_COST_EVALUATION_SCOPE_ENV: "sha256:" + "6" * 64,
                "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR": str(cache_dir),
                "RESEARCH_LAB_PROVIDER_EVIDENCE_RECORD": "1",
                V2_PROVIDER_PROFILE_ENV: "benchmark_model",
            },
        ),
        model_kind="candidate",
        worker_index=4,
        epoch_id=24001,
        execute=execute,
        catalog_snapshot_loader=_load_empty_catalog,
    )
    entries, token = begin_incontainer_trace_collection()
    receipt_hashes, receipt_token = begin_attested_receipt_hash_collection()
    try:
        result = await runner(
            icp,
            {"evaluation_epoch": 24000, "run_id": "run-1"},
        )
    finally:
        end_attested_receipt_hash_collection(receipt_token)
        end_incontainer_trace_collection(token)

    assert result == [{"company_name": "Measured Co"}]
    payload = observed[0]["payload"]
    assert payload["operation"] == "run_icp"
    assert payload["callable_name"] == "run_icp"
    assert "EXA_API_KEY" not in payload["environment"]
    assert V2_PROVIDER_PROFILE_ENV not in payload["environment"]
    assert (
        payload["environment"][PROVIDER_COST_EVALUATION_SCOPE_ENV]
        == payload["provider_cost_scope"]
    )
    assert payload["provider_cost_scope"] != "sha256:" + "6" * 64
    assert payload["provider_evidence_cache"] == cache_doc
    assert payload["provider_evidence_cache_ref"] == cache_ref
    assert observed[0]["parent_graphs"] == (
        tape_graph,
        tape_lineage_graph,
        _catalog_outcome()["execution_receipt_graph"],
        _catalog_outcome()["receipt_graph"],
    )
    assert payload["provider_catalog_evidence"]["root_receipt_hash"] == (
        _catalog_outcome()["execution_receipt"]["receipt_hash"]
    )
    assert observed[0]["purpose"] == "research_lab.candidate_model_run.v2"
    assert observed[0]["provider_credential_profile"] == "benchmark_model"
    assert observed[0]["epoch_id"] == 24001
    assert observed[0]["timeout_seconds"] == 1920.0
    assert payload["input"]["context"] == {
        "evaluation_epoch": 24000,
        "run_id": "run-1",
        "runtime_options": {
            "runtime_cap_seconds": 1500.0,
            "finalization_reserve_seconds": 60.0,
            "agent_timeout_seconds": 900,
        },
    }
    assert entries == [_runtime_receipt(1500.0), {"provider": "exa"}]
    assert runner.attested_receipts() == [
        {"receipt_hash": "sha256:" + "4" * 64}
    ]
    assert receipt_hashes == {"sha256:" + "4" * 64}


def test_attested_model_metadata_uses_same_measured_authority(tmp_path, monkeypatch):
    artifact = _artifact(tmp_path)
    source_bundle = {
        "schema_version": "leadpoet.private_source_bundle.v2",
        "archive_sha256": "sha256:" + "2" * 64,
        "source_tree_hash": artifact["model_artifact_hash"],
        "archive_size_bytes": 1,
        "archive_b64": "AA==",
    }
    monkeypatch.setattr(
        model_authority_v2,
        "_source_bundle_for_artifact",
        lambda *_args, **_kwargs: dict(source_bundle),
    )

    async def execute(**kwargs):
        payload = kwargs["payload"]
        assert payload["operation"] == "metadata"
        assert payload["callable_name"] == "adapter_metadata"
        output = _ready_adapter_metadata()
        return {
            "result": {
                "schema_version": "leadpoet.model_sandbox_result.v2",
                "model_kind": "private",
                "operation": "metadata",
                "model_artifact_hash": artifact["model_artifact_hash"],
                "model_manifest_hash": artifact["manifest_hash"],
                "compatibility_image_digest": artifact["image_digest"],
                "source_bundle_hash": source_bundle["archive_sha256"],
                "runtime_config_hash": "sha256:" + "3" * 64,
                "input_hash": sha256_json(payload["input"]),
                "provider_evidence_cache_hash": sha256_json({}),
                "provider_evidence_cache_ref": "",
                "provider_evidence_mode": payload["provider_evidence_mode"],
                "provider_snapshot_archive_hash": sha256_json({}),
                "provider_snapshot_tree_hash": sha256_json({}),
                "provider_snapshot_manifest_hash": sha256_json({}),
                "provider_cost_cap_microusd": 0,
                "provider_call_cap": 0,
                "provider_runtime_catalog_hash": payload[
                    "provider_runtime_catalog"
                ]["catalog_hash"],
                "generated_provider_evidence_cache_hash": sha256_json({}),
                "trace_entries_hash": sha256_json([]),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [],
                "generated_provider_evidence_cache": {},
            },
            "receipt": {"receipt_hash": "sha256:" + "4" * 64},
        }

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        execute=execute,
        catalog_snapshot_loader=_load_empty_catalog,
    )
    assert runner.metadata() == _ready_adapter_metadata()


@pytest.mark.asyncio
async def test_private_baseline_persists_signed_tape_before_atomic_cache_publish(
    tmp_path, monkeypatch
):
    artifact = _artifact(tmp_path)
    source_bundle = {
        "schema_version": "leadpoet.private_source_bundle.v2",
        "archive_sha256": "sha256:" + "2" * 64,
        "source_tree_hash": artifact["model_artifact_hash"],
        "archive_size_bytes": 1,
        "archive_b64": "AA==",
    }
    monkeypatch.setattr(
        model_authority_v2,
        "_source_bundle_for_artifact",
        lambda *_args, **_kwargs: dict(source_bundle),
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv("RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR", str(cache_dir))
    icp = {"industry": "Software", "intent_signal": "Hiring"}
    from research_lab.eval.private_runtime import canonicalize_private_model_icp
    from research_lab.eval.provider_evidence_cache import icp_evidence_cache_key

    canonical_icp = canonicalize_private_model_icp(icp)
    cache_ref = icp_evidence_cache_key(canonical_icp)
    cache_doc = {
        "schema_version": "1.1",
        "rolling_window_hash": "",
        "icp_ref": cache_ref,
        "utc_day": "2026-07-10",
        "entries": {},
    }
    cache_hash = sha256_json(cache_doc)
    tape_receipt = {
        "receipt_hash": "sha256:" + "5" * 64,
        "role": "gateway_scoring",
        "purpose": "research_lab.provider_evidence_tape.v2",
        "status": "succeeded",
        "input_root": provider_evidence_tape_input_root(cache_ref, cache_hash),
        "output_root": cache_hash,
    }
    source_receipt = {"receipt_hash": "sha256:" + "6" * 64}
    lineage_receipt = {"receipt_hash": "sha256:" + "7" * 64}
    events = []

    async def persist_link(**kwargs):
        assert not (cache_dir / (cache_ref + ".json")).exists()
        events.append(("persist", kwargs))
        return {"business_artifact_link_count": 1}

    monkeypatch.setattr(
        model_authority_v2,
        "_persist_provider_evidence_tape_link",
        persist_link,
    )

    async def execute(**kwargs):
        payload = kwargs["payload"]
        output = []
        return {
            "result": {
                "schema_version": "leadpoet.model_sandbox_result.v2",
                "model_kind": "private",
                "operation": "run_icp",
                "model_artifact_hash": artifact["model_artifact_hash"],
                "model_manifest_hash": artifact["manifest_hash"],
                "compatibility_image_digest": artifact["image_digest"],
                "source_bundle_hash": source_bundle["archive_sha256"],
                "runtime_config_hash": "sha256:" + "3" * 64,
                "input_hash": sha256_json(payload["input"]),
                "provider_evidence_cache_hash": sha256_json({}),
                "provider_evidence_cache_ref": cache_ref,
                "provider_evidence_mode": payload["provider_evidence_mode"],
                "provider_snapshot_archive_hash": sha256_json({}),
                "provider_snapshot_tree_hash": sha256_json({}),
                "provider_snapshot_manifest_hash": sha256_json({}),
                "provider_cost_cap_microusd": 0,
                "provider_call_cap": 0,
                "provider_runtime_catalog_hash": payload[
                    "provider_runtime_catalog"
                ]["catalog_hash"],
                "generated_provider_evidence_cache_hash": cache_hash,
                "trace_entries_hash": sha256_json([_runtime_receipt(897.0)]),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [_runtime_receipt(897.0)],
                "generated_provider_evidence_cache": cache_doc,
            },
            "receipt": lineage_receipt,
            "receipt_graph": {
                "root_receipt_hash": lineage_receipt["receipt_hash"],
                "receipts": [lineage_receipt],
            },
            "execution_receipt": source_receipt,
            "execution_receipt_graph": {
                "root_receipt_hash": source_receipt["receipt_hash"],
                "receipts": [tape_receipt, source_receipt],
            },
        }

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(image_digest=artifact["image_digest"]),
        model_kind="private",
        worker_index=0,
        epoch_id=24001,
        execute=execute,
        catalog_snapshot_loader=_load_empty_catalog,
    )
    assert await runner(icp, {"mode": "private_baseline"}) == []
    assert events == [
        (
            "persist",
            {
                "receipt_hash": lineage_receipt["receipt_hash"],
                "cache_ref": cache_ref,
                "cache_hash": cache_hash,
            },
        )
    ]
    published = cache_dir / (cache_ref + ".json")
    assert published.read_text(encoding="utf-8") == json.dumps(
        cache_doc,
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.mark.asyncio
async def test_provider_tape_loader_preserves_source_and_persistence_ancestry(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    cache_ref = "1" * 64
    cache_hash = "sha256:" + "2" * 64
    source_hash = "sha256:" + "3" * 64
    lineage_hash = "sha256:" + "4" * 64
    tape_receipt = {
        "receipt_hash": "sha256:" + "5" * 64,
        "role": "gateway_scoring",
        "purpose": "research_lab.provider_evidence_tape.v2",
        "status": "succeeded",
        "input_root": provider_evidence_tape_input_root(cache_ref, cache_hash),
        "output_root": cache_hash,
    }
    source_graph = {
        "root_receipt_hash": source_hash,
        "receipts": [tape_receipt, {"receipt_hash": source_hash}],
    }
    lineage_graph = {
        "root_receipt_hash": lineage_hash,
        "receipts": [
            {
                "receipt_hash": lineage_hash,
                "role": "gateway_coordinator",
                "purpose": "leadpoet.artifact_persistence.v2",
                "status": "succeeded",
                "parent_receipt_hashes": [source_hash],
            }
        ],
    }

    async def load_business(**kwargs):
        assert kwargs == {
            "artifact_kind": "provider_evidence_tape_v2",
            "artifact_ref": cache_ref,
            "artifact_hash": cache_hash,
        }
        return lineage_graph

    async def load_receipt(receipt_hash):
        assert receipt_hash == source_hash
        return source_graph

    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        load_business,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_receipt_graph_v2",
        load_receipt,
    )

    assert await model_authority_v2._load_provider_evidence_tape_graphs(
        cache_ref=cache_ref,
        cache_hash=cache_hash,
    ) == (source_graph, lineage_graph)


@pytest.mark.asyncio
async def test_provider_tape_link_reuses_exact_existing_receipt_owner(monkeypatch):
    from gateway.research_lab import attested_v2_store

    cache_ref = "1" * 64
    cache_hash = "sha256:" + "2" * 64
    owner_hash = "sha256:" + "3" * 64
    tape_receipt = {
        "receipt_hash": "sha256:" + "4" * 64,
        "role": "gateway_scoring",
        "purpose": "research_lab.provider_evidence_tape.v2",
        "status": "succeeded",
        "input_root": provider_evidence_tape_input_root(cache_ref, cache_hash),
        "output_root": cache_hash,
    }
    owner_graph = {
        "root_receipt_hash": owner_hash,
        "receipts": [tape_receipt, {"receipt_hash": owner_hash}],
    }

    async def persist_business_artifact_links_v2(**_kwargs):
        raise attested_v2_store.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        )

    async def load_business_artifact_graph_v2(**kwargs):
        assert kwargs == {
            "artifact_kind": "provider_evidence_tape_v2",
            "artifact_ref": cache_ref,
            "artifact_hash": cache_hash,
        }
        return owner_graph

    monkeypatch.setattr(
        attested_v2_store,
        "persist_business_artifact_links_v2",
        persist_business_artifact_links_v2,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        load_business_artifact_graph_v2,
    )

    result = await model_authority_v2._persist_provider_evidence_tape_link(
        receipt_hash="sha256:" + "5" * 64,
        cache_ref=cache_ref,
        cache_hash=cache_hash,
    )

    assert result == {
        "business_artifact_link_count": 1,
        "business_artifact_link_set_hash": sha256_json(
            [
                {
                    "receipt_hash": owner_hash,
                    "artifact_kind": "provider_evidence_tape_v2",
                    "artifact_ref": cache_ref,
                    "artifact_hash": cache_hash,
                }
            ]
        ),
    }


@pytest.mark.asyncio
async def test_provider_tape_link_replay_rejects_mismatched_existing_graph(monkeypatch):
    from gateway.research_lab import attested_v2_store

    cache_ref = "1" * 64
    cache_hash = "sha256:" + "2" * 64
    mismatched_graph = {
        "root_receipt_hash": "sha256:" + "3" * 64,
        "receipts": [
            {
                "receipt_hash": "sha256:" + "4" * 64,
                "role": "gateway_scoring",
                "purpose": "research_lab.provider_evidence_tape.v2",
                "status": "succeeded",
                "input_root": provider_evidence_tape_input_root(
                    cache_ref,
                    "sha256:" + "9" * 64,
                ),
                "output_root": "sha256:" + "9" * 64,
            },
            {"receipt_hash": "sha256:" + "3" * 64},
        ],
    }

    async def persist_business_artifact_links_v2(**_kwargs):
        raise attested_v2_store.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at receipt_hash"
        )

    async def load_business_artifact_graph_v2(**_kwargs):
        return mismatched_graph

    monkeypatch.setattr(
        attested_v2_store,
        "persist_business_artifact_links_v2",
        persist_business_artifact_links_v2,
    )
    monkeypatch.setattr(
        attested_v2_store,
        "load_business_artifact_graph_v2",
        load_business_artifact_graph_v2,
    )

    with pytest.raises(
        AttestedPrivateModelRunnerV2Error,
        match="provider evidence cache has no unique measured tape receipt",
    ):
        await model_authority_v2._persist_provider_evidence_tape_link(
            receipt_hash="sha256:" + "5" * 64,
            cache_ref=cache_ref,
            cache_hash=cache_hash,
        )


@pytest.mark.asyncio
async def test_provider_tape_link_replay_does_not_mask_other_store_conflicts(
    monkeypatch,
):
    from gateway.research_lab import attested_v2_store

    async def persist_business_artifact_links_v2(**_kwargs):
        raise attested_v2_store.AttestedV2StoreError(
            "research_lab_attested_business_artifact_links_v2 "
            "stored row conflicts at artifact_hash"
        )

    monkeypatch.setattr(
        attested_v2_store,
        "persist_business_artifact_links_v2",
        persist_business_artifact_links_v2,
    )

    with pytest.raises(
        attested_v2_store.AttestedV2StoreError,
        match="stored row conflicts at artifact_hash",
    ):
        await model_authority_v2._persist_provider_evidence_tape_link(
            receipt_hash="sha256:" + "5" * 64,
            cache_ref="1" * 64,
            cache_hash="sha256:" + "2" * 64,
        )


@pytest.mark.asyncio
async def test_candidate_cache_without_exact_tape_graph_fails_before_execution(
    tmp_path, monkeypatch
):
    artifact = _artifact(tmp_path)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    icp = {"industry": "Software", "intent_signal": "Hiring"}
    from research_lab.eval.private_runtime import canonicalize_private_model_icp
    from research_lab.eval.provider_evidence_cache import icp_evidence_cache_key

    cache_ref = icp_evidence_cache_key(canonicalize_private_model_icp(icp))
    cache_doc = {
        "schema_version": "1.1",
        "rolling_window_hash": "",
        "icp_ref": cache_ref,
        "utc_day": "2026-07-10",
        "entries": {},
    }
    (cache_dir / (cache_ref + ".json")).write_text(
        json.dumps(cache_doc),
        encoding="utf-8",
    )

    async def missing_graph(**_kwargs):
        raise RuntimeError("measured tape missing")

    monkeypatch.setattr(
        model_authority_v2,
        "_load_provider_evidence_tape_graphs",
        missing_graph,
    )
    calls = []

    async def execute(**kwargs):
        calls.append(kwargs)
        return {}

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(
            image_digest=artifact["image_digest"],
            extra_env={"RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_DIR": str(cache_dir)},
        ),
        model_kind="candidate",
        worker_index=0,
        execute=execute,
    )
    with pytest.raises(RuntimeError, match="measured tape missing"):
        await runner(icp, {"mode": "candidate"})
    assert calls == []

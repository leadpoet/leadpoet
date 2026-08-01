from __future__ import annotations

import json
import subprocess

import pytest

from gateway.research_lab import model_authority_v2
from gateway.research_lab.model_authority_v2 import (
    AttestedPrivateModelRunnerV2,
    V2_PROVIDER_PROFILE_ENV,
)
from gateway.research_lab.tee_protocol import ResearchLabTeeProtocolError
from gateway.tee.model_sandbox_v2 import provider_evidence_tape_input_root
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from gateway.tee.source_bundle_v2 import extract_source_bundle_v2
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.eval import DockerPrivateModelSpec, build_local_private_artifact_manifest
from research_lab.eval.private_runtime import (
    begin_incontainer_trace_collection,
    compute_private_source_tree_hash,
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

    async def load_tape_graph(**kwargs):
        assert kwargs == {"cache_ref": cache_ref, "cache_hash": cache_hash}
        return dict(tape_graph)

    monkeypatch.setattr(
        model_authority_v2,
        "_load_provider_evidence_tape_graph",
        load_tape_graph,
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
                    [_runtime_receipt(897.0), {"provider": "exa"}]
                ),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [_runtime_receipt(897.0), {"provider": "exa"}],
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
    try:
        result = await runner(
            icp,
            {"evaluation_epoch": 24000, "run_id": "run-1"},
        )
    finally:
        end_incontainer_trace_collection(token)

    assert result == [{"company_name": "Measured Co"}]
    payload = observed[0]["payload"]
    assert "EXA_API_KEY" not in payload["environment"]
    assert V2_PROVIDER_PROFILE_ENV not in payload["environment"]
    assert payload["provider_evidence_cache"] == cache_doc
    assert payload["provider_evidence_cache_ref"] == cache_ref
    assert observed[0]["parent_graphs"] == (
        tape_graph,
        _catalog_outcome()["receipt_graph"],
    )
    assert observed[0]["purpose"] == "research_lab.candidate_model_run.v2"
    assert observed[0]["provider_credential_profile"] == "benchmark_model"
    assert observed[0]["epoch_id"] == 24001
    assert payload["input"]["context"] == {
        "evaluation_epoch": 24000,
        "run_id": "run-1",
        "runtime_options": {
            "runtime_cap_seconds": 897.0,
            "finalization_reserve_seconds": 5.0,
            "agent_timeout_seconds": 892,
        },
    }
    assert entries == [_runtime_receipt(897.0), {"provider": "exa"}]
    assert runner.attested_receipts() == [
        {"receipt_hash": "sha256:" + "4" * 64}
    ]


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
    root_receipt = {"receipt_hash": "sha256:" + "6" * 64}
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
            "receipt": root_receipt,
            "receipt_graph": {
                "root_receipt_hash": root_receipt["receipt_hash"],
                "receipts": [tape_receipt, root_receipt],
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
                "receipt_hash": root_receipt["receipt_hash"],
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
        "_load_provider_evidence_tape_graph",
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

from __future__ import annotations

import json

import pytest

from gateway.research_lab import model_authority_v2
from gateway.research_lab.model_authority_v2 import (
    AttestedPrivateModelRunnerV2,
    V2_PROVIDER_PROFILE_ENV,
)
from gateway.research_lab.tee_protocol import ResearchLabTeeProtocolError
from gateway.tee.model_sandbox_v2 import provider_evidence_tape_input_root
from gateway.tee.source_add_runtime_v2 import build_source_add_runtime_catalog_v2
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.eval import DockerPrivateModelSpec, build_local_private_artifact_manifest
from research_lab.eval.private_runtime import (
    begin_incontainer_trace_collection,
    end_incontainer_trace_collection,
)


def _artifact(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "research_lab_adapter.py").write_text(
        "def run_icp(icp, context):\n    return []\n",
        encoding="utf-8",
    )
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
    receipt = {
        "receipt_hash": "sha256:" + "c" * 64,
        "role": "gateway_coordinator",
        "purpose": "research_lab.source_add_catalog_snapshot.v2",
        "status": "succeeded",
        "output_root": sha256_json(result),
    }
    return {
        "result": result,
        "receipt": receipt,
        "receipt_graph": {
            "root_receipt_hash": receipt["receipt_hash"],
            "receipts": [receipt],
        },
    }


async def _load_empty_catalog(*, epoch_id):
    assert epoch_id >= 0
    return _catalog_outcome()


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
                "trace_entries_hash": sha256_json([{"provider": "exa"}]),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [{"provider": "exa"}],
                "generated_provider_evidence_cache": {},
            },
            "receipt": {"receipt_hash": "sha256:" + "4" * 64},
        }

    runner = AttestedPrivateModelRunnerV2(
        artifact=artifact,
        spec=DockerPrivateModelSpec(
            image_digest=artifact["image_digest"],
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
    }
    assert entries == [{"provider": "exa"}]
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
        output = {"adapter_version": "v1"}
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
    assert runner.metadata() == {"adapter_version": "v1"}


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
                "trace_entries_hash": sha256_json([]),
                "output_hash": sha256_json(output),
                "output": output,
                "trace_entries": [],
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

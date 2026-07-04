from __future__ import annotations

from typing import Any

import pytest

from gateway.research_lab import store


def _artifact_manifest(**overrides: Any) -> dict[str, Any]:
    doc = {
        "model_artifact_hash": "sha256:" + "a" * 64,
        "manifest_hash": "sha256:" + "b" * 64,
        "manifest_uri": "s3://bucket/research-lab/sourcing-model/main.json",
        "git_commit_sha": "1" * 40,
        "config_hash": "sha256:" + "c" * 64,
        "component_registry_version": "component-registry-v1",
        "scoring_adapter_version": "adapter-v1",
        "signature_ref": "kms://signature/ref",
        "build_id": "build-1",
    }
    doc.update(overrides)
    return doc


@pytest.mark.asyncio
async def test_create_private_model_version_reuses_existing_matching_artifact(monkeypatch):
    artifact = _artifact_manifest()
    existing_row = {
        "private_model_version_id": "private_model_version:existing",
        "model_artifact_hash": artifact["model_artifact_hash"],
        "private_model_manifest_hash": artifact["manifest_hash"],
        "private_model_manifest_uri": artifact["manifest_uri"],
        "git_commit_sha": artifact["git_commit_sha"],
        "config_hash": artifact["config_hash"],
        "component_registry_version": artifact["component_registry_version"],
        "scoring_adapter_version": artifact["scoring_adapter_version"],
        "signature_ref": artifact["signature_ref"],
        "build_id": artifact["build_id"],
    }
    select_filters: list[tuple[tuple[Any, ...], ...]] = []
    event_kwargs: list[dict[str, Any]] = []

    async def fake_select_one(table: str, *, filters: Any, columns: str = "*") -> dict[str, Any] | None:
        assert table == "research_lab_private_model_versions"
        select_filters.append(tuple(filters))
        if tuple(filters) == (("model_artifact_hash", artifact["model_artifact_hash"]),):
            return dict(existing_row)
        return None

    async def fake_create_private_model_version_event(**kwargs: Any) -> dict[str, Any]:
        event_kwargs.append(dict(kwargs))
        return {"event_id": "event-1", **kwargs}

    async def fail_insert_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("duplicate artifact reuse must not insert a new private model version")

    monkeypatch.setattr(store, "select_one", fake_select_one)
    monkeypatch.setattr(store, "create_private_model_version_event", fake_create_private_model_version_event)
    monkeypatch.setattr(store, "insert_row", fail_insert_row)

    row, event = await store.create_private_model_version(
        artifact_manifest=artifact,
        manifest_uri=artifact["manifest_uri"],
        source_candidate_id="candidate:new",
        source_score_bundle_id="score_bundle:new",
        redacted_version_doc={"source": "promotion"},
        version_status="active",
        reason="research_lab_image_build_candidate_repo_head_manifest_promoted",
    )

    assert row == existing_row
    assert event["private_model_version_id"] == "private_model_version:existing"
    assert event["event_type"] == "active"
    assert event["event_doc"]["reused_existing_model_artifact_hash"] is True
    assert event["event_doc"]["requested_private_model_version_id"].startswith("private_model_version:sha256:")
    assert (("model_artifact_hash", artifact["model_artifact_hash"]),) in select_filters


@pytest.mark.asyncio
async def test_create_private_model_version_rejects_conflicting_existing_artifact(monkeypatch):
    artifact = _artifact_manifest()
    existing_row = {
        "private_model_version_id": "private_model_version:existing",
        "model_artifact_hash": artifact["model_artifact_hash"],
        "private_model_manifest_hash": artifact["manifest_hash"],
        "private_model_manifest_uri": artifact["manifest_uri"],
        "git_commit_sha": "2" * 40,
        "config_hash": artifact["config_hash"],
        "component_registry_version": artifact["component_registry_version"],
        "scoring_adapter_version": artifact["scoring_adapter_version"],
        "signature_ref": artifact["signature_ref"],
        "build_id": artifact["build_id"],
    }

    async def fake_select_one(table: str, *, filters: Any, columns: str = "*") -> dict[str, Any] | None:
        if tuple(filters) == (("model_artifact_hash", artifact["model_artifact_hash"]),):
            return dict(existing_row)
        return None

    async def fail_create_private_model_version_event(**kwargs: Any) -> dict[str, Any]:
        raise AssertionError("conflicting private model version must not emit an event")

    async def fail_insert_row(table: str, row: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("conflicting private model version must not insert")

    monkeypatch.setattr(store, "select_one", fake_select_one)
    monkeypatch.setattr(store, "create_private_model_version_event", fail_create_private_model_version_event)
    monkeypatch.setattr(store, "insert_row", fail_insert_row)

    with pytest.raises(RuntimeError, match="conflicts on git_commit_sha"):
        await store.create_private_model_version(
            artifact_manifest=artifact,
            manifest_uri=artifact["manifest_uri"],
            version_status="active",
            reason="repo_head_sync",
        )

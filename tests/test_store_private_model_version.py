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
async def test_create_private_model_version_is_retired_before_any_io(monkeypatch):
    async def fail_io(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("retired private model writer performed I/O")

    monkeypatch.setattr(store, "select_one", fail_io)
    monkeypatch.setattr(store, "insert_row", fail_io)
    monkeypatch.setattr(store, "create_private_model_version_event", fail_io)

    with pytest.raises(RuntimeError, match="central private model activation coordinator"):
        await store.create_private_model_version(
            artifact_manifest=_artifact_manifest(),
        )


@pytest.mark.asyncio
async def test_create_private_model_version_event_is_retired_before_any_io(
    monkeypatch,
) -> None:
    async def fail_io(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("retired private model event writer performed I/O")

    monkeypatch.setattr(store, "append_event_with_seq", fail_io)

    with pytest.raises(RuntimeError, match="central private model activation coordinator"):
        await store.create_private_model_version_event(
            private_model_version_id="private_model_version:retired",
            event_type="active",
            version_status="active",
        )


@pytest.mark.asyncio
async def test_ensure_version_row_recovers_lost_insert_and_ignores_transient_doc(
    monkeypatch,
) -> None:
    artifact = _artifact_manifest()
    persisted: dict[str, Any] = {}
    insert_calls = 0

    async def fake_select_one(
        table: str, *, filters: Any, columns: str = "*"
    ) -> dict[str, Any] | None:
        del columns
        assert table == "research_lab_private_model_versions"
        assert tuple(filters) == (
            ("model_artifact_hash", artifact["model_artifact_hash"]),
        )
        return dict(persisted) if persisted else None

    async def committed_but_response_lost(
        table: str, row: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal insert_calls
        assert table == "research_lab_private_model_versions"
        insert_calls += 1
        persisted.update(row)
        raise ConnectionError("response lost after commit")

    monkeypatch.setattr(store, "select_one", fake_select_one)
    monkeypatch.setattr(store, "insert_row", committed_but_response_lost)

    first, created = await store.ensure_private_model_version_row_exact(
        artifact_manifest=artifact,
        source_candidate_id="candidate:exact",
        source_score_bundle_id="score:exact",
        source_benchmark_bundle_id="benchmark:exact",
        redacted_version_doc={"activation_generation": 7, "status": "pending"},
    )
    second, reused = await store.ensure_private_model_version_row_exact(
        artifact_manifest=artifact,
        source_candidate_id="candidate:exact",
        source_score_bundle_id="score:exact",
        source_benchmark_bundle_id="benchmark:exact",
        redacted_version_doc={"activation_generation": 9, "status": "active"},
    )

    assert created is False
    assert reused is False
    assert first == second == persisted
    assert first["redacted_version_doc"]["activation_generation"] == 7
    assert insert_calls == 1


@pytest.mark.asyncio
async def test_ensure_version_row_rejects_conflicting_source_provenance(
    monkeypatch,
) -> None:
    artifact = _artifact_manifest()
    existing = {
        "model_artifact_hash": artifact["model_artifact_hash"],
        "private_model_manifest_hash": artifact["manifest_hash"],
        "private_model_manifest_uri": artifact["manifest_uri"],
        "git_commit_sha": artifact["git_commit_sha"],
        "config_hash": artifact["config_hash"],
        "component_registry_version": artifact["component_registry_version"],
        "scoring_adapter_version": artifact["scoring_adapter_version"],
        "source_candidate_id": "candidate:other",
        "source_score_bundle_id": "score:exact",
        "source_benchmark_bundle_id": "benchmark:exact",
        "signature_ref": artifact["signature_ref"],
        "build_id": artifact["build_id"],
    }

    async def fake_select_one(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(existing)

    monkeypatch.setattr(store, "select_one", fake_select_one)
    with pytest.raises(RuntimeError, match="source_candidate_id"):
        await store.ensure_private_model_version_row_exact(
            artifact_manifest=artifact,
            source_candidate_id="candidate:exact",
            source_score_bundle_id="score:exact",
            source_benchmark_bundle_id="benchmark:exact",
        )


@pytest.mark.asyncio
async def test_private_model_event_cas_recovers_identical_lost_response_once(
    monkeypatch,
) -> None:
    inserted: dict[str, Any] = {}
    insert_calls = 0

    async def fake_select_many(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return [
            {
                "seq": 3,
                "anchored_hash": "sha256:" + "a" * 64,
                "version_status": "superseded",
            }
        ]

    async def committed_but_response_lost(
        table: str, row: dict[str, Any]
    ) -> dict[str, Any]:
        nonlocal insert_calls
        assert table == "research_lab_private_model_version_events"
        insert_calls += 1
        inserted.update(row)
        raise ConnectionError("response lost after commit")

    async def fake_select_one(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return dict(inserted)

    monkeypatch.setattr(store, "select_many", fake_select_many)
    monkeypatch.setattr(store, "insert_row", committed_but_response_lost)
    monkeypatch.setattr(store, "select_one", fake_select_one)

    event = await store.create_private_model_version_event_cas(
        private_model_version_id="private_model_version:exact",
        expected_current_event_seq=3,
        expected_current_event_hash="sha256:" + "a" * 64,
        expected_current_version_status="superseded",
        event_type="active",
        version_status="active",
        reason="exact_resume",
        event_doc={"expected_global_lineage_generation": 11},
    )

    assert event == inserted
    assert event["seq"] == 4
    assert insert_calls == 1


@pytest.mark.asyncio
async def test_private_model_event_cas_never_retries_stale_seq(monkeypatch) -> None:
    insert_calls = 0

    async def fake_select_many(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    async def stale_insert(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        nonlocal insert_calls
        insert_calls += 1
        raise RuntimeError("23505 duplicate key unique constraint seq")

    async def missing_event(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(store, "select_many", fake_select_many)
    monkeypatch.setattr(store, "insert_row", stale_insert)
    monkeypatch.setattr(store, "select_one", missing_event)

    with pytest.raises(RuntimeError, match="expected-state CAS conflict"):
        await store.create_private_model_version_event_cas(
            private_model_version_id="private_model_version:stale",
            expected_current_event_seq=None,
            event_type="active",
            version_status="active",
        )
    assert insert_calls == 1

from datetime import datetime, timedelta, timezone

import boto3
import pytest
from fastapi import HTTPException

from gateway.api import weights as weights_api
from gateway.tee import release_channel_v2


CURRENT_COMMIT = "a" * 40
HISTORICAL_COMMIT = "b" * 40
VERSION_ID = "immutable-version-id"


class _LockedReleaseStore:
    def __init__(self, *, head=None):
        self.head = head or {
            "ObjectLockMode": "COMPLIANCE",
            "ObjectLockRetainUntilDate": datetime.now(timezone.utc)
            + timedelta(days=1),
            "VersionId": VERSION_ID,
        }
        self.head_calls = []
        self.presign_calls = []

    def head_object(self, **kwargs):
        self.head_calls.append(kwargs)
        return self.head

    def generate_presigned_url(self, operation, **kwargs):
        self.presign_calls.append((operation, kwargs))
        return f"https://release.invalid/{operation}"


def _install_locked_store(monkeypatch, store):
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: store)


def test_immutable_release_evidence_get_route_is_registered():
    matching = [
        route
        for route in weights_api.router.routes
        if route.path == "/weights/v2/immutable-release-evidence/{commit_sha}"
    ]

    assert len(matching) == 1
    assert matching[0].methods == {"GET"}


@pytest.mark.asyncio
async def test_current_immutable_v2_preserves_current_local_v1(monkeypatch):
    store = _LockedReleaseStore()
    _install_locked_store(monkeypatch, store)
    gateway_release = {"commit_sha": CURRENT_COMMIT}
    validator_release = {"commit_sha": CURRENT_COMMIT}
    local_channel = {
        "schema_version": "leadpoet.attested_release_channel.v2",
        "commit_sha": CURRENT_COMMIT,
    }
    monkeypatch.setattr(
        weights_api, "_gateway_v2_release_manifest", lambda: gateway_release
    )
    monkeypatch.setattr(
        release_channel_v2, "_load_json", lambda path, label: validator_release
    )
    monkeypatch.setattr(
        release_channel_v2,
        "build_release_channel_v2",
        lambda **kwargs: local_channel,
    )

    immutable = await weights_api.get_immutable_auditor_release_evidence_v2(
        CURRENT_COMMIT
    )
    local = await weights_api.get_auditor_release_evidence_v2(CURRENT_COMMIT)

    assert immutable == {
        "schema_version": "leadpoet.auditor_release_evidence.v2",
        "commit_sha": CURRENT_COMMIT,
        "release_channel_version_id": VERSION_ID,
        "release_channel_get_url": "https://release.invalid/get_object",
        "release_channel_head_url": "https://release.invalid/head_object",
    }
    assert local == {
        "schema_version": "leadpoet.auditor_local_release_evidence.v1",
        "commit_sha": CURRENT_COMMIT,
        "release_channel": local_channel,
    }
    assert len(store.head_calls) == 1
    assert store.head_calls[0] == {
        "Bucket": release_channel_v2.DEFAULT_BUCKET,
        "Key": release_channel_v2.release_channel_key(
            CURRENT_COMMIT, prefix=release_channel_v2.DEFAULT_PREFIX
        ),
    }
    assert all(
        call[1]["Params"]["VersionId"] == VERSION_ID
        for call in store.presign_calls
    )


@pytest.mark.asyncio
async def test_historical_release_evidence_delegates_to_immutable_v2(monkeypatch):
    store = _LockedReleaseStore()
    _install_locked_store(monkeypatch, store)
    monkeypatch.setattr(
        weights_api,
        "_gateway_v2_release_manifest",
        lambda: {"commit_sha": CURRENT_COMMIT},
    )

    evidence = await weights_api.get_auditor_release_evidence_v2(HISTORICAL_COMMIT)

    assert evidence["schema_version"] == "leadpoet.auditor_release_evidence.v2"
    assert evidence["commit_sha"] == HISTORICAL_COMMIT
    assert evidence["release_channel_version_id"] == VERSION_ID
    assert "VersionId" not in store.head_calls[0]


@pytest.mark.asyncio
async def test_immutable_release_evidence_rejects_invalid_commit():
    with pytest.raises(HTTPException) as exc:
        await weights_api.get_immutable_auditor_release_evidence_v2("not-a-commit")

    assert exc.value.status_code == 422
    assert exc.value.detail == "release commit is invalid"


@pytest.mark.asyncio
async def test_immutable_release_evidence_requires_compliance_lock(monkeypatch):
    store = _LockedReleaseStore(head={"VersionId": VERSION_ID})
    _install_locked_store(monkeypatch, store)

    with pytest.raises(HTTPException) as exc:
        await weights_api.get_immutable_auditor_release_evidence_v2(CURRENT_COMMIT)

    assert exc.value.status_code == 404
    assert exc.value.detail == "immutable V2 release evidence is unavailable"
    assert store.presign_calls == []

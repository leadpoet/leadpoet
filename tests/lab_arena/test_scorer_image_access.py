from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import urlencode

import pytest

from lab_arena import images, leased_images
from lab_arena.scorer_image_access import (
    EcrScorerImageAccess,
    MAX_SIGNED_URL_TTL_SECONDS,
    SCHEMA_VERSION,
    ScorerImageAccessError,
    ecr_provider_from_repository,
    validate_ecr_download_url,
)
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import hash_lease_token


NOW = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
REGISTRY = "493765492819.dkr.ecr.us-east-1.amazonaws.com"
REPOSITORY = "leadpoet/sourcing-model"


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _signed_url(
    *,
    issued: datetime = NOW,
    expires: int = MAX_SIGNED_URL_TTL_SECONDS,
    host: str = "prod-us-east-1-starport-layer-bucket.s3.us-east-1.amazonaws.com",
) -> str:
    query = urlencode(
        {
            "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
            "X-Amz-Credential": "temporary/credential",
            "X-Amz-Date": issued.strftime("%Y%m%dT%H%M%SZ"),
            "X-Amz-Expires": str(expires),
            "X-Amz-SignedHeaders": "host",
            "X-Amz-Signature": "a" * 64,
        }
    )
    return "https://%s/layer?%s" % (host, query)


def _manifest(layer_count: int = 19, *, layer_size: int = 11):
    config_bytes = b'{"architecture":"amd64","os":"linux"}'
    config = {
        "mediaType": images.CONFIG_MEDIA_TYPES[0],
        "digest": _digest(config_bytes),
        "size": len(config_bytes),
    }
    layers = []
    for index in range(layer_count):
        payload = ("layer-%02d" % index).encode("ascii")
        layers.append(
            {
                "mediaType": images.LAYER_GZIP_MEDIA_TYPES[0],
                "digest": _digest(payload),
                "size": layer_size,
            }
        )
    body = json.dumps(
        {
            "schemaVersion": 2,
            "mediaType": images.MANIFEST_DOCKER,
            "config": config,
            "layers": layers,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return body, config, layers


class _Ecr:
    def __init__(self, manifest: bytes):
        self.manifest = manifest
        self.manifest_calls = []
        self.blob_calls = []
        self.returned_blob_digest = None
        self.url = _signed_url()

    def batch_get_image(self, **kwargs):
        self.manifest_calls.append(kwargs)
        digest = _digest(self.manifest)
        return {
            "images": [
                {
                    "registryId": "493765492819",
                    "repositoryName": REPOSITORY,
                    "imageId": {"imageDigest": digest},
                    "imageManifest": self.manifest.decode("utf-8"),
                    "imageManifestMediaType": images.MANIFEST_DOCKER,
                }
            ],
            "failures": [],
        }

    def get_download_url_for_layer(self, **kwargs):
        self.blob_calls.append(kwargs)
        return {
            "layerDigest": self.returned_blob_digest or kwargs["layerDigest"],
            "downloadUrl": self.url,
        }


def _provider(client: _Ecr, *, max_bytes: int = images.DEFAULT_MAX_IMAGE_BYTES):
    return EcrScorerImageAccess(
        client=client,
        registry=REGISTRY,
        repository=REPOSITORY,
        registry_id="493765492819",
        rules=images.ImageRules(max_image_bytes=max_bytes),
        clock=lambda: NOW,
    )


def test_ecr_provider_returns_only_the_exact_manifest_descriptor_set():
    manifest, config, layers = _manifest()
    client = _Ecr(manifest)
    digest = _digest(manifest)
    reference = "%s/%s@%s" % (REGISTRY, REPOSITORY, digest)

    document = _provider(client)(reference, digest)

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["image_reference"] == reference
    assert document["image_digest"] == digest
    assert document["manifest_media_type"] == images.MANIFEST_DOCKER
    assert len(document["blobs"]) == 20
    assert [item["digest"] for item in document["blobs"]] == [
        config["digest"],
        *(layer["digest"] for layer in layers),
    ]
    assert client.manifest_calls == [
        {
            "registryId": "493765492819",
            "repositoryName": REPOSITORY,
            "imageIds": [{"imageDigest": digest}],
            "acceptedMediaTypes": list(images.MANIFEST_MEDIA_TYPES),
        }
    ]
    assert {call["layerDigest"] for call in client.blob_calls} == {
        item["digest"] for item in document["blobs"]
    }


def _duplicate_response(client, monkeypatch, *, copies=2, mutate=None):
    original = client.batch_get_image

    def repeated(**kwargs):
        response = original(**kwargs)
        response["images"] = [deepcopy(response["images"][0]) for _ in range(copies)]
        for index, image in enumerate(response["images"]):
            image["imageId"]["imageTag"] = "reviewer-tag-%d" % index
        if mutate is not None:
            mutate(response)
        return response

    monkeypatch.setattr(client, "batch_get_image", repeated)


@pytest.mark.parametrize("copies", [2, 3])
def test_ecr_accepts_identical_digest_entries_with_different_tags(monkeypatch, copies):
    # Retained production metadata: repeated Docker V2 entries, each 4,935 bytes.
    # Use a synthetic manifest; no production manifest or capability is retained.
    manifest, config, layers = _manifest(layer_count=30)
    manifest += b" " * (4935 - len(manifest))
    assert len(manifest) == 4935
    client = _Ecr(manifest)
    _duplicate_response(client, monkeypatch, copies=copies)
    digest = _digest(manifest)

    document = _provider(client)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)

    assert document["image_digest"] == digest
    assert document["manifest_media_type"] == images.MANIFEST_DOCKER
    assert len(client.manifest_calls) == 1
    assert len(client.blob_calls) == len(layers) + 1
    assert [blob["digest"] for blob in document["blobs"]] == [
        config["digest"], *(layer["digest"] for layer in layers)
    ]
    assert len({call["layerDigest"] for call in client.blob_calls}) == len(client.blob_calls)


@pytest.mark.parametrize("entry_index", [0, 1])
@pytest.mark.parametrize(
    "mutation",
    [
        lambda image: None,
        lambda image: [],
        lambda image: {**image, "registryId": "000000000000"},
        lambda image: {**image, "repositoryName": "other/repository"},
        lambda image: {**image, "imageId": None},
        lambda image: {**image, "imageId": {}},
        lambda image: {**image, "imageId": {"imageDigest": "sha256:" + "f" * 64}},
        lambda image: {**image, "imageManifestMediaType": images.MANIFEST_OCI},
        lambda image: {**image, "imageManifestMediaType": "invalid"},
        lambda image: {**image, "imageManifestMediaType": None},
        lambda image: {**image, "imageManifest": None},
        lambda image: {**image, "imageManifest": 4935},
        lambda image: {**image, "imageManifest": "\ud800"},
        lambda image: {**image, "imageManifest": image["imageManifest"] + " "},
    ],
)
def test_duplicate_entries_fail_closed_on_any_conflicting_identity_or_manifest(
    monkeypatch, entry_index, mutation
):
    manifest = _manifest()[0]
    client = _Ecr(manifest)

    def mutate(response):
        response["images"][entry_index] = mutation(response["images"][entry_index])

    _duplicate_response(client, monkeypatch, mutate=mutate)
    digest = _digest(manifest)
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)
    assert caught.value.code in {"ecr_manifest_invalid", "ecr_manifest_digest_mismatch"}
    assert len(client.manifest_calls) == 1
    assert client.blob_calls == []
    assert "https://" not in str(caught.value)
    assert caught.value.__context__ is None


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {"images": []},
        {"images": None},
        {"images": {}},
        {"images": "invalid"},
    ],
)
def test_empty_or_malformed_manifest_response_still_fails_closed(monkeypatch, response):
    manifest = _manifest()[0]
    client = _Ecr(manifest)
    monkeypatch.setattr(client, "batch_get_image", lambda **_kwargs: response)
    digest = _digest(manifest)
    with pytest.raises(ScorerImageAccessError):
        _provider(client)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)
    assert client.blob_calls == []


def test_duplicate_response_with_ecr_failure_is_not_partially_accepted(monkeypatch):
    manifest = _manifest()[0]
    client = _Ecr(manifest)
    _duplicate_response(
        client, monkeypatch,
        mutate=lambda response: response.update(failures=[{"failureCode": "ImageNotFound"}]),
    )
    digest = _digest(manifest)
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)
    assert caught.value.code == "ecr_manifest_unavailable"
    assert client.blob_calls == []


def test_identical_duplicates_still_enforce_image_rules_before_url_calls(monkeypatch):
    manifest = _manifest(layer_count=1, layer_size=100)[0]
    client = _Ecr(manifest)
    _duplicate_response(client, monkeypatch)
    digest = _digest(manifest)
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client, max_bytes=99)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)
    assert caught.value.code == "ecr_manifest_invalid"
    assert client.blob_calls == []


@pytest.mark.parametrize(
    ("reference", "digest", "code"),
    [
        (
            "493765492819.dkr.ecr.us-east-1.amazonaws.com/other/repo@sha256:" + "a" * 64,
            "sha256:" + "a" * 64,
            "image_repository_mismatch",
        ),
        (
            REGISTRY + "/" + REPOSITORY + "@sha256:" + "a" * 64,
            "sha256:" + "b" * 64,
            "image_repository_mismatch",
        ),
    ],
)
def test_ecr_provider_rejects_an_image_outside_the_trusted_pin(reference, digest, code):
    client = _Ecr(_manifest()[0])
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)(reference, digest)
    assert caught.value.code == code
    assert client.manifest_calls == []


def test_ecr_provider_rejects_wrong_manifest_and_blob_digests_without_url_detail():
    manifest = _manifest()[0]
    client = _Ecr(manifest)
    digest = _digest(manifest)
    reference = "%s/%s@%s" % (REGISTRY, REPOSITORY, digest)
    response = client.batch_get_image

    def wrong_manifest(**kwargs):
        value = response(**kwargs)
        value["images"][0]["imageManifest"] += " "
        return value

    client.batch_get_image = wrong_manifest
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)(reference, digest)
    assert caught.value.code == "ecr_manifest_digest_mismatch"

    client = _Ecr(manifest)
    client.returned_blob_digest = "sha256:" + "f" * 64
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)(reference, digest)
    assert caught.value.code == "ecr_blob_invalid"
    assert "https://" not in str(caught.value)


def test_ecr_provider_enforces_existing_image_size_limit_before_url_calls():
    manifest = _manifest(layer_count=1, layer_size=100)[0]
    client = _Ecr(manifest)
    digest = _digest(manifest)
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client, max_bytes=99)(
            "%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest
        )
    assert caught.value.code == "ecr_manifest_invalid"
    assert client.blob_calls == []


@pytest.mark.parametrize(
    ("url", "code"),
    [
        (_signed_url(issued=NOW - timedelta(hours=2)), "signed_url_expired"),
        (_signed_url(expires=3601), "signed_url_lifetime_invalid"),
        (_signed_url(host="127.0.0.1"), "signed_url_invalid"),
        (_signed_url(host="example.com"), "signed_url_invalid"),
        (_signed_url().replace("https://", "http://", 1), "signed_url_invalid"),
        (_signed_url() + "&bad field", "signed_url_invalid"),
        (_signed_url() + "&malformed", "signed_url_invalid"),
    ],
)
def test_signed_url_must_be_current_bounded_and_public_aws(url, code):
    with pytest.raises(ScorerImageAccessError) as caught:
        validate_ecr_download_url(url, now=NOW)
    assert caught.value.code == code
    assert "X-Amz-" not in str(caught.value)


def test_repository_factory_preserves_generic_registry_operation():
    rules = images.ImageRules()
    assert (
        ecr_provider_from_repository(
            "registry.example/team/scorer", rules=rules, client=object()
        )
        is None
    )
    provider = ecr_provider_from_repository(
        REGISTRY + "/" + REPOSITORY, rules=rules, client=object()
    )
    assert provider is not None
    assert provider.registry_id == "493765492819"


def test_manifest_encoding_failure_is_sanitized():
    manifest = _manifest()[0]
    client = _Ecr(manifest)
    digest = _digest(manifest)

    def malformed(**_kwargs):
        return {
            "images": [
                {
                    "registryId": "493765492819",
                    "repositoryName": REPOSITORY,
                    "imageId": {"imageDigest": digest},
                    "imageManifest": "\ud800",
                    "imageManifestMediaType": images.MANIFEST_DOCKER,
                }
            ],
            "failures": [],
        }

    client.batch_get_image = malformed
    with pytest.raises(ScorerImageAccessError) as caught:
        _provider(client)("%s/%s@%s" % (REGISTRY, REPOSITORY, digest), digest)
    assert caught.value.code == "ecr_manifest_invalid"
    assert "ud800" not in str(caught.value)


def _service(*, token: str = "a" * 64, status: str = "leased", expiry=None, role="validator"):
    digest = "sha256:" + "c" * 64
    reference = "%s/%s@%s" % (REGISTRY, REPOSITORY, digest)
    run = {
        "run_id": "run-1",
        "round_id": "arena-2026-09-10",
        "runner_hotkey": "5" * 48,
        "status": status,
        "lease_token_hash": hash_lease_token(token),
        "lease_expires_at": expiry or NOW + timedelta(minutes=7),
    }
    round_row = {
        "round_id": "arena-2026-09-10",
        "status": "stage1",
        "configuration_doc": {
            "mode": "live",
            "network_name": "finney",
            "netuid": 71,
            "scorer_image_reference": reference,
            "scorer_image_digest": digest,
        },
    }
    calls = []
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        get_run=lambda requested: run if requested == "run-1" else None,
        get_round=lambda requested: round_row if requested == round_row["round_id"] else None,
    )
    service._clock = lambda: NOW
    service._config = SimpleNamespace(
        mode="live",
        network_name="finney",
        netuid=71,
        pinned_round_id=None,
        validator_authorizer=lambda _hotkey: (
            role is not None,
            role,
        ),
        scorer_image_access=lambda ref, pin: (
            calls.append((ref, pin))
            or {
                "schema_version": SCHEMA_VERSION,
                "image_reference": ref,
                "image_digest": pin,
                "manifest_b64": "e30=",
                "manifest_media_type": images.MANIFEST_DOCKER,
                "blobs": [],
            }
        ),
    )
    service._test_round = round_row
    return service, run, calls


def test_service_derives_image_only_from_the_frozen_round_after_live_lease_auth():
    service, _run, calls = _service()
    result = service.handle_scorer_image_access("run-1", "a" * 64)
    assert calls == [(result["image_reference"], result["image_digest"])]


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_ecr_response_through_service_and_shared_lease_consumer(monkeypatch, conflicting):
    manifest = _manifest()[0]
    client = _Ecr(manifest)
    current = datetime.now(timezone.utc)
    client.url = _signed_url(issued=current)

    def mutate(response):
        if conflicting:
            response["images"][1]["imageManifest"] += " "

    _duplicate_response(client, monkeypatch, mutate=mutate)
    digest = _digest(manifest)
    reference = "%s/%s@%s" % (REGISTRY, REPOSITORY, digest)
    service, _run, _calls = _service()
    service._test_round["configuration_doc"].update(
        scorer_image_digest=digest, scorer_image_reference=reference
    )
    rules = images.ImageRules()
    service._config.scorer_image_access = EcrScorerImageAccess(
        client, REGISTRY, REPOSITORY, "493765492819", rules, clock=lambda: current
    )

    if conflicting:
        with pytest.raises(ServiceError) as caught:
            service.handle_scorer_image_access("run-1", "a" * 64)
        assert caught.value.code == "scorer_image_access_unavailable"
        assert caught.value.__context__ is None
        assert client.blob_calls == []
    else:
        document = service.handle_scorer_image_access("run-1", "a" * 64)
        access = leased_images.validate_image_access(
            document, image_reference=reference, image_digest=digest, rules=rules
        )
        assert access.manifest == manifest
        assert access.image_digest == digest
        assert len(access.blobs) == len(client.blob_calls) == 20
    assert len(client.manifest_calls) == 1


@pytest.mark.parametrize(
    ("mutation", "token", "code"),
    [
        ({}, "b" * 64, "lease_invalid"),
        ({"status": "accepted"}, "a" * 64, "lease_inactive"),
        ({"lease_expires_at": NOW}, "a" * 64, "lease_expired"),
        ({"round_id": "another-round"}, "a" * 64, "round_missing"),
    ],
)
def test_service_rejects_invalid_inactive_expired_or_wrong_round_lease(
    mutation, token, code
):
    service, run, calls = _service()
    run.update(mutation)
    with pytest.raises(ServiceError) as caught:
        service.handle_scorer_image_access("run-1", token)
    assert caught.value.code == code
    assert calls == []


@pytest.mark.parametrize(
    ("role", "code"),
    [(None, "runner_hotkey_unregistered"), ("miner", "runner_validator_required")],
)
def test_service_reauthorizes_the_current_lease_holder(role, code):
    service, _run, calls = _service(role=role)
    with pytest.raises(ServiceError) as caught:
        service.handle_scorer_image_access("run-1", "a" * 64)
    assert caught.value.code == code
    assert calls == []


@pytest.mark.parametrize("status", ["published", "cancelled"])
def test_service_never_issues_fresh_urls_after_the_round_ends(status):
    service, _run, calls = _service()
    service._test_round["status"] = status
    with pytest.raises(ServiceError) as caught:
        service.handle_scorer_image_access("run-1", "a" * 64)
    assert caught.value.code == "round_ended"
    assert calls == []


def test_service_redacts_provider_failures():
    service, _run, _calls = _service()
    secret_url = _signed_url()

    def fail(_reference, _digest):
        raise RuntimeError(secret_url)

    service._config.scorer_image_access = fail
    with pytest.raises(ServiceError) as caught:
        service.handle_scorer_image_access("run-1", "a" * 64)
    assert caught.value.code == "scorer_image_access_unavailable"
    assert secret_url not in str(caught.value)
    assert caught.value.__context__ is None

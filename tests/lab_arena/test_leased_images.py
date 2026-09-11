"""Lease-bound private ECR image transfer for ordinary Arena runners."""

from __future__ import annotations

import base64
import copy
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict
from urllib.parse import quote

import httpx
import pytest

from lab_arena import images, leased_images, runner
from tests.lab_arena.test_lab_arena_images import (
    digest_of,
    public_test_resolver,
    simple_image,
)
from tests.lab_arena.test_lab_arena_runner import (
    BridgingRuntime,
    FakeApi,
    IMAGE,
    lease,
    make_config,
    valid_company,
)


ECR_REGISTRY = "123456789012.dkr.ecr.us-east-1.amazonaws.com"
ECR_REPOSITORY = "leadpoet/arena-scorer"
S3_HOST = "prod-us-east-1-starport-layer-bucket.s3.us-east-1.amazonaws.com"


def _signed_url(digest: str, *, host: str = S3_HOST) -> str:
    issued = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    query = (
        "X-Amz-Algorithm=AWS4-HMAC-SHA256"
        "&X-Amz-Credential=" + quote("AKIAEXAMPLE/20260910/us-east-1/s3/aws4_request")
        + "&X-Amz-Date=" + issued
        + "&X-Amz-Expires=3600"
        "&X-Amz-SignedHeaders=host"
        "&X-Amz-Signature=do-not-disclose"
    )
    return "https://%s/blobs/%s?%s" % (host, digest, query)


def _access_document(image: Dict[str, Any]) -> tuple[str, Dict[str, Any], Dict[str, bytes]]:
    reference = "%s/%s@%s" % (ECR_REGISTRY, ECR_REPOSITORY, image["digest"])
    manifest = json.loads(image["manifest"])
    descriptors = [manifest["config"], *manifest["layers"]]
    payloads = {
        digest_of(image["config"]): image["config"],
        **{digest_of(layer): layer for layer in image["layers"]},
    }
    document = {
        "schema_version": leased_images.IMAGE_ACCESS_SCHEMA_VERSION,
        "image_reference": reference,
        "image_digest": image["digest"],
        "manifest_b64": base64.b64encode(image["manifest"]).decode("ascii"),
        "manifest_media_type": image["media"],
        "blobs": [
            {
                "digest": descriptor["digest"],
                "size": descriptor["size"],
                "media_type": descriptor["mediaType"],
                "url": _signed_url(descriptor["digest"]),
            }
            for descriptor in descriptors
        ],
    }
    return reference, document, payloads


class AccessApi:
    def __init__(self, documents):
        self.documents = list(documents)
        self.requests = []

    def image_access(self, run_id, lease_token):
        self.requests.append((run_id, lease_token))
        return copy.deepcopy(self.documents.pop(0))


class BlobTransport:
    def __init__(self, payloads: Dict[str, bytes]):
        self.payloads = payloads
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        digest = request.url.path.rsplit("/", 1)[-1]
        payload = self.payloads.get(digest)
        return httpx.Response(200 if payload is not None else 404, content=payload or b"")


def _client_factory(transport: BlobTransport):
    def build() -> images.RegistryClient:
        return images.RegistryClient(
            http=httpx.Client(transport=httpx.MockTransport(transport)),
            address_resolver=public_test_resolver,
        )

    return build


def test_cache_miss_materializes_lease_image_and_hit_never_refetches_access(tmp_path):
    image = simple_image()
    reference, document, payloads = _access_document(image)
    api = AccessApi([document])
    transport = BlobTransport(payloads)
    exporter = leased_images.leased_image_exporter(
        api,
        "run-1",
        "lease-secret",
        client_factory=_client_factory(transport),
    )
    fallback_calls = []
    cache = runner.ImageCache(
        tmp_path / "cache",
        lambda *_args: fallback_calls.append(_args),
    )

    with cache.acquire(image["digest"], reference, exporter=exporter) as rootfs:
        assert (rootfs / "app/main.py").read_text() == "print('hi')\n"
    with cache.acquire(
        image["digest"],
        reference,
        exporter=lambda *_args: pytest.fail("cache hit invoked exporter"),
    ) as cached:
        assert cached == rootfs

    assert api.requests == [("run-1", "lease-secret")]
    assert fallback_calls == []
    # Config is checked once; each layer is downloaded once for extraction.
    assert len(transport.requests) == 2


@pytest.mark.parametrize(
    "mutation",
    ["lease_hash", "manifest_hash", "blob_size", "blob_size_bool", "extra_blob"],
)
def test_access_document_is_strictly_bound_to_lease_and_manifest(mutation):
    image = simple_image()
    reference, document, _payloads = _access_document(image)
    if mutation == "lease_hash":
        document["image_digest"] = "sha256:" + "f" * 64
    elif mutation == "manifest_hash":
        document["manifest_b64"] = base64.b64encode(b"{}").decode("ascii")
    elif mutation == "blob_size":
        document["blobs"][0]["size"] += 1
    elif mutation == "blob_size_bool":
        document["blobs"][0]["size"] = True
    else:
        document["blobs"].append(copy.deepcopy(document["blobs"][-1]))

    with pytest.raises(leased_images.LeasedImageError):
        leased_images.validate_image_access(
            document,
            image_reference=reference,
            image_digest=image["digest"],
            rules=images.ImageRules(),
        )


def test_private_or_expired_url_is_rejected_without_secret_in_error_or_traceback(tmp_path):
    image = simple_image()
    reference, document, payloads = _access_document(image)
    secret = "must-not-appear"
    document["blobs"][0]["url"] = (
        "https://169.254.169.254/blob?X-Amz-Signature=" + secret
    )
    api = AccessApi([document])
    transport = BlobTransport(payloads)
    exporter = leased_images.leased_image_exporter(
        api,
        "run-1",
        "lease-secret",
        client_factory=_client_factory(transport),
    )

    with pytest.raises(leased_images.LeasedImageError) as caught:
        exporter(reference, image["digest"], tmp_path / "target")

    rendered = "".join(traceback.format_exception(caught.value))
    assert secret not in str(caught.value)
    assert secret not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert transport.requests == []


def test_signed_url_query_is_not_emitted_by_httpx_or_httpcore_logs(tmp_path, caplog):
    image = simple_image()
    reference, document, payloads = _access_document(image)
    canary = "signed-query-canary"
    for blob in document["blobs"]:
        blob["url"] = blob["url"].replace("do-not-disclose", canary)
    api = AccessApi([document])
    transport = BlobTransport(payloads)
    exporter = leased_images.leased_image_exporter(
        api,
        "run-1",
        "lease-secret",
        client_factory=_client_factory(transport),
    )

    caplog.set_level("DEBUG")
    exporter(reference, image["digest"], tmp_path / "target")

    assert canary not in caplog.text
    assert "X-Amz-Signature" not in caplog.text


def test_transport_exception_does_not_retain_signed_url(tmp_path):
    image = simple_image()
    reference, document, _payloads = _access_document(image)
    canary = "transport-exception-canary"
    for blob in document["blobs"]:
        blob["url"] = blob["url"].replace("do-not-disclose", canary)
    api = AccessApi([document])

    def fail(request):
        raise httpx.ConnectError(
            "failed signed request %s" % request.url,
            request=request,
        )

    def client_factory():
        return images.RegistryClient(
            http=httpx.Client(transport=httpx.MockTransport(fail)),
            address_resolver=public_test_resolver,
        )

    exporter = leased_images.leased_image_exporter(
        api,
        "run-1",
        "lease-secret",
        client_factory=client_factory,
    )
    with pytest.raises(leased_images.LeasedImageError) as caught:
        exporter(reference, image["digest"], tmp_path / "target")

    rendered = "".join(traceback.format_exception(caught.value))
    assert canary not in str(caught.value)
    assert canary not in rendered
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_failed_transfer_is_not_cached_and_retry_refetches_access(tmp_path):
    image = simple_image()
    reference, first, payloads = _access_document(image)
    second = copy.deepcopy(first)
    api = AccessApi([first, second])
    layer_digest = first["blobs"][1]["digest"]

    class FailFirstLayer(BlobTransport):
        failed = False

        def __call__(self, request):
            digest = request.url.path.rsplit("/", 1)[-1]
            if digest == layer_digest and not self.failed:
                self.failed = True
                self.requests.append(request)
                return httpx.Response(200, content=self.payloads[digest] + b"corrupt")
            return super().__call__(request)

    transport = FailFirstLayer(payloads)
    cache = runner.ImageCache(
        tmp_path / "cache",
        lambda *_args: pytest.fail("generic exporter used for ECR"),
    )

    with pytest.raises(leased_images.LeasedImageError):
        with cache.acquire(
            image["digest"],
            reference,
            exporter=leased_images.leased_image_exporter(
                api, "run-1", "lease-1", client_factory=_client_factory(transport)
            ),
        ):
            pass
    assert not (tmp_path / "cache" / image["digest"].replace(":", "-")).exists()

    with cache.acquire(
        image["digest"],
        reference,
        exporter=leased_images.leased_image_exporter(
            api, "run-1", "lease-1", client_factory=_client_factory(transport)
        ),
    ) as rootfs:
        assert (rootfs / "app/main.py").is_file()
    assert len(api.requests) == 2


def test_public_registry_cache_miss_keeps_generic_exporter(tmp_path):
    digest = "sha256:" + "a" * 64
    reference = "ghcr.io/leadpoet/arena-scorer@" + digest
    calls = []

    def fallback(got_reference, got_digest, target):
        calls.append((got_reference, got_digest))
        (target / "rootfs").mkdir()

    cache = runner.ImageCache(tmp_path / "cache", fallback)
    with cache.acquire(digest, reference):
        pass
    assert leased_images.is_ecr_reference(reference) is False
    assert calls == [(reference, digest)]


def test_normal_assignment_executor_automatically_uses_lease_exporter(
    tmp_path, monkeypatch
):
    api = FakeApi([])
    leased = lease()
    leased["image_reference"] = "%s/%s@%s" % (
        ECR_REGISTRY,
        ECR_REPOSITORY,
        IMAGE,
    )
    sandbox = BridgingRuntime(output={"companies": [valid_company(1)]}, calls=0)
    (tmp_path / "work").mkdir()
    config = make_config(tmp_path, api, sandbox)
    calls = []

    def factory(got_api, run_id, lease_token):
        calls.append((got_api, run_id, lease_token))

        def export(reference, digest, target):
            calls.append((reference, digest))
            (target / "rootfs").mkdir()

        return export

    monkeypatch.setattr(leased_images, "leased_image_exporter", factory)
    envelope = runner.AssignmentExecutor(config).execute(
        leased,
        leased["lease_token"],
        leased["icp"],
    )

    assert envelope["body"]["result"]["terminal_status"] == "accepted"
    assert calls == [
        (api, "r1", "tok-r1"),
        (leased["image_reference"], IMAGE),
    ]


def test_http_api_image_access_sends_lease_header_and_enforces_bounds():
    requests = []
    payload = {
        "schema_version": leased_images.IMAGE_ACCESS_SCHEMA_VERSION,
        "image_reference": "ref",
        "image_digest": "digest",
        "manifest_b64": "e30=",
        "manifest_media_type": images.MANIFEST_OCI,
        "blobs": [],
    }

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    api = runner.HttpArenaApiClient(
        "https://arena.example",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert api.image_access("run-1", "lease-secret") == payload
    assert requests[0].url.path == "/arena/v1/runs/run-1/image-access"
    assert requests[0].headers["x-lab-arena-lease"] == "lease-secret"

    oversized = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                headers={
                    "content-length": str(
                        leased_images.MAX_IMAGE_ACCESS_DOCUMENT_BYTES + 1
                    )
                },
            )
        )
    )
    bounded = runner.HttpArenaApiClient("https://arena.example", client=oversized)
    with pytest.raises(runner.RunnerError, match="document limit"):
        bounded.image_access("run-1", "lease-secret")


@pytest.mark.parametrize(
    "body",
    [
        b"[]",
        b"not-json",
        b'{"schema_version":"leadpoet.arena.scorer_image_access.v1"}',
    ],
)
def test_http_api_image_access_rejects_invalid_documents(body):
    api = runner.HttpArenaApiClient(
        "https://arena.example",
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=body)
            )
        ),
    )
    with pytest.raises(runner.RunnerError):
        api.image_access("run-1", "lease-secret")

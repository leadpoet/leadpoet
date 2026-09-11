from fastapi.testclient import TestClient

from lab_arena import images, leased_images, runner
from lab_arena.api import create_app
from lab_arena.scorer_image_access import EcrScorerImageAccess
from tests.lab_arena.test_lab_arena_images import digest_of, simple_image
from tests.lab_arena.test_leased_images import (
    BlobTransport,
    _client_factory,
    _signed_url,
)
from tests.lab_arena.test_scorer_image_access import (
    REGISTRY,
    REPOSITORY,
    _service,
)


class FakeEcr:
    def __init__(self, image):
        self.image = image
        self.manifest_calls = 0
        self.layer_calls = []

    def batch_get_image(self, **kwargs):
        self.manifest_calls += 1
        return {
            "images": [{
                "registryId": "493765492819",
                "repositoryName": REPOSITORY,
                "imageId": {"imageDigest": self.image["digest"]},
                "imageManifest": self.image["manifest"].decode("utf-8"),
                "imageManifestMediaType": self.image["media"],
            }],
            "failures": [],
        }

    def get_download_url_for_layer(self, **kwargs):
        self.layer_calls.append(kwargs["layerDigest"])
        return {
            "layerDigest": kwargs["layerDigest"],
            "downloadUrl": _signed_url(kwargs["layerDigest"]),
        }


def test_gateway_runner_ecr_lease_roundtrip_materializes_and_caches_without_secrets(tmp_path):
    image = simple_image()
    reference = f"{REGISTRY}/{REPOSITORY}@{image['digest']}"
    service, _, _ = _service()
    service._test_round["configuration_doc"].update(
        scorer_image_reference=reference,
        scorer_image_digest=image["digest"],
    )

    ecr = FakeEcr(image)
    provider = EcrScorerImageAccess(
        client=ecr,
        registry=REGISTRY,
        repository=REPOSITORY,
        registry_id="493765492819",
        rules=images.ImageRules(),
    )
    provider_calls = []
    service._config.scorer_image_access = lambda ref, digest: (
        provider_calls.append((ref, digest)) or provider(ref, digest)
    )
    gateway = TestClient(create_app(service))
    api = runner.HttpArenaApiClient(
        "http://127.0.0.1",
        client=gateway,
    )
    payloads = {
        digest_of(image["config"]): image["config"],
        **{digest_of(layer): layer for layer in image["layers"]},
    }
    transport = BlobTransport(payloads)
    exporter = leased_images.leased_image_exporter(
        api,
        "run-1",
        "a" * 64,
        client_factory=_client_factory(transport),
    )
    cache_root = tmp_path / "cache"
    cache = runner.ImageCache(
        cache_root,
        lambda *_args: (_ for _ in ()).throw(AssertionError("generic exporter used")),
    )

    try:
        with cache.acquire(image["digest"], reference, exporter=exporter) as rootfs:
            assert (rootfs / "app/main.py").read_text() == "print('hi')\n"
        first_api_calls = list(provider_calls)
        first_ecr_calls = (ecr.manifest_calls, list(ecr.layer_calls))
        first_blob_calls = len(transport.requests)

        with cache.acquire(image["digest"], reference, exporter=exporter) as cached:
            assert cached == rootfs
        assert provider_calls == first_api_calls
        assert (ecr.manifest_calls, ecr.layer_calls) == first_ecr_calls
        assert len(transport.requests) == first_blob_calls

        bad_token = gateway.get(
            "/arena/v1/runs/run-1/image-access",
            headers={"x-lab-arena-lease": "b" * 64},
        )
        assert bad_token.status_code == 401
        assert bad_token.json()["code"] == "lease_invalid"

        service._config.validator_authorizer = lambda _hotkey: (True, "miner")
        non_validator = gateway.get(
            "/arena/v1/runs/run-1/image-access",
            headers={"x-lab-arena-lease": "a" * 64},
        )
        assert non_validator.status_code == 403
        assert non_validator.json()["code"] == "runner_validator_required"
        assert provider_calls == first_api_calls

        for path in cache_root.rglob("*"):
            if path.is_file():
                assert b"X-Amz-" not in path.read_bytes()
                assert b"amazonaws.com" not in path.read_bytes()
        assert all(request.headers.get("authorization") is None for request in transport.requests)
    finally:
        api.close()

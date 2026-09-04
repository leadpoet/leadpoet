"""Common runtime image resolution and safe root filesystem materialization.

A fake OCI registry served through ``httpx.MockTransport`` exercises
anonymous and credentialed reads plus blob redirects. The extractor tests
apply real layer tarballs, including whiteouts and hostile members.
"""

from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import stat
import tarfile
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

import httpx
import pytest

from lab_arena import images
from lab_arena.images import ImageError, ImageReference, ImageRules, RegistryClient

SOURCE = "registry.example"
RUNTIME_REGISTRY = "runtime.example"
CDN = "cdn.example"
AUTH = "https://auth.example/token"
READ_CREDENTIAL = ("organizer", "read-secret")
PUBLIC_TEST_ADDRESS = "93.184.216.34"


def public_test_resolver(_host: str, _port: int) -> Tuple[str, ...]:
    return (PUBLIC_TEST_ADDRESS,)


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Image construction
# ---------------------------------------------------------------------------


def layer_tar(entries: List[Tuple[Any, ...]], *, compressed: bool = True) -> bytes:
    """Entries: ``(name, "file", bytes[, mode])``, ``(name, "dir")``, ``(name, "symlink", target)``,
    ``(name, "hardlink", target)``, ``(name, "chr")``."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for entry in entries:
            name, kind = entry[0], entry[1]
            info = tarfile.TarInfo(name)
            if kind == "file":
                data = entry[2]
                info.size = len(data)
                info.mode = entry[3] if len(entry) > 3 else 0o644
                archive.addfile(info, io.BytesIO(data))
            elif kind == "dir":
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                archive.addfile(info)
            elif kind == "symlink":
                info.type = tarfile.SYMTYPE
                info.linkname = entry[2]
                archive.addfile(info)
            elif kind == "hardlink":
                info.type = tarfile.LNKTYPE
                info.linkname = entry[2]
                archive.addfile(info)
            elif kind == "chr":
                info.type = tarfile.CHRTYPE
                info.devmajor, info.devminor = 1, 3
                archive.addfile(info)
            else:
                raise AssertionError(kind)
    raw = buffer.getvalue()
    return gzip.compress(raw, mtime=0) if compressed else raw


def build_image(
    *,
    layers: List[bytes],
    layer_media: str = images.LAYER_GZIP_MEDIA_TYPES[0],
    os_name: str = "linux",
    arch: str = "amd64",
    entrypoint: Optional[List[str]] = None,
    cmd: Optional[List[str]] = None,
    env: Optional[List[str]] = None,
    workdir: str = "/app",
) -> Dict[str, Any]:
    config = {
        "architecture": arch,
        "os": os_name,
        "config": {"Entrypoint": entrypoint, "Cmd": cmd, "Env": env, "WorkingDir": workdir, "User": ""},
        "rootfs": {"type": "layers", "diff_ids": [digest_of(gzip.decompress(layer)) if layer[:2] == b"\x1f\x8b" else digest_of(layer) for layer in layers]},
    }
    config_bytes = json.dumps(config, sort_keys=True).encode("utf-8")
    manifest = {
        "schemaVersion": 2,
        "mediaType": images.MANIFEST_OCI,
        "config": {"mediaType": images.CONFIG_MEDIA_TYPES[0], "digest": digest_of(config_bytes), "size": len(config_bytes)},
        "layers": [{"mediaType": layer_media, "digest": digest_of(layer), "size": len(layer)} for layer in layers],
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return {"manifest": manifest_bytes, "media": images.MANIFEST_OCI, "config": config_bytes, "layers": layers, "digest": digest_of(manifest_bytes)}


def build_index(children: List[Tuple[Dict[str, Any], str, str, bool]]) -> Dict[str, Any]:
    entries = []
    for image, os_name, arch, attestation in children:
        entry: Dict[str, Any] = {"mediaType": image["media"], "digest": image["digest"], "size": len(image["manifest"]), "platform": {"os": os_name, "architecture": arch}}
        if attestation:
            entry["platform"] = {"os": "unknown", "architecture": "unknown"}
            entry["annotations"] = {"vnd.docker.reference.type": "attestation-manifest"}
        entries.append(entry)
    document = {"schemaVersion": 2, "mediaType": images.INDEX_OCI, "manifests": entries}
    index_bytes = json.dumps(document, sort_keys=True).encode("utf-8")
    return {"manifest": index_bytes, "media": images.INDEX_OCI, "digest": digest_of(index_bytes), "children": [child[0] for child in children]}


def simple_image(**overrides: Any) -> Dict[str, Any]:
    layers = overrides.pop("layers", None) or [layer_tar([("app", "dir"), ("app/main.py", "file", b"print('hi')\n")])]
    options = dict(entrypoint=["python3", "/app/main.py"], cmd=["--fast"], env=["PATH=/usr/local/bin:/usr/bin", "APP_MODE=fast"], workdir="/app")
    options.update(overrides)
    return build_image(layers=layers, **options)


# ---------------------------------------------------------------------------
# Fake registry
# ---------------------------------------------------------------------------

_MANIFEST_RE = re.compile(r"^/v2/(.+)/manifests/(.+)$")
_BLOB_RE = re.compile(r"^/v2/(.+)/blobs/(sha256:[0-9a-f]{64})$")


class FakeRegistry:
    """In-memory read-only registries with optional bearer auth and redirects."""

    def __init__(self, *, bearer_hosts: Tuple[str, ...] = (), redirect_blobs: bool = False) -> None:
        self.repos: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.bearer_hosts = bearer_hosts
        self.redirect_blobs = redirect_blobs
        self.requests: List[Tuple[str, str, str, Optional[str]]] = []
        self.token_requests: List[Tuple[str, Optional[str]]] = []

    def repo(self, host: str, name: str) -> Dict[str, Any]:
        return self.repos.setdefault((host, name), {"manifests": {}, "blobs": {}})

    def put_image(self, host: str, name: str, image: Dict[str, Any]) -> str:
        repo = self.repo(host, name)
        repo["manifests"][image["digest"]] = (image["manifest"], image["media"])
        for child in image.get("children", []):
            self.put_image(host, name, child)
        if "config" in image:
            repo["blobs"][digest_of(image["config"])] = image["config"]
            for layer in image["layers"]:
                repo["blobs"][digest_of(layer)] = layer
        return "%s/%s@%s" % (host, name, image["digest"])

    # -- HTTP -----------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        path = request.url.path
        authorization = request.headers.get("authorization")
        self.requests.append((request.method, host, path, authorization))
        if host == "auth.example" or path == "/token":
            return self._token(request)
        if host == CDN:
            assert authorization is None, "redirected blob fetch must not carry the registry credential"
            _, _, source_host, digest = path.split("/", 3)
            name = (parse_qs(request.url.query.decode("utf-8")).get("repo") or [""])[0]
            blob = self.repo(source_host, name)["blobs"].get(digest)
            return httpx.Response(200 if blob is not None else 404, content=blob or b"")
        if host in self.bearer_hosts:
            scope_match = _MANIFEST_RE.match(path) or _BLOB_RE.match(path)
            name = scope_match.group(1) if scope_match else "unknown"
            granted = authorization[len("Bearer tok:"):] if authorization and authorization.startswith("Bearer tok:") else ""
            parts = granted.split(":", 2)
            allowed = len(parts) == 3 and parts[0] == "repository" and parts[1] == name and "pull" in parts[2].split(",")
            if not allowed:
                challenge = 'Bearer realm="%s",service="registry",scope="repository:%s:pull"' % (AUTH, name)
                return httpx.Response(401, headers={"WWW-Authenticate": challenge})
        manifest = _MANIFEST_RE.match(path)
        if manifest:
            return self._manifest(request, host, manifest.group(1), manifest.group(2))
        blob = _BLOB_RE.match(path)
        if blob:
            return self._blob(request, host, blob.group(1), blob.group(2))
        return httpx.Response(404)

    def _token(self, request: httpx.Request) -> httpx.Response:
        query = parse_qs(request.url.query.decode("utf-8"))
        scope = (query.get("scope") or [None])[0]
        authorization = request.headers.get("authorization")
        self.token_requests.append((scope or "", authorization))
        return httpx.Response(200, json={"token": "tok:" + (scope or "")})

    def _manifest(self, request: httpx.Request, host: str, name: str, reference: str) -> httpx.Response:
        repo = self.repo(host, name)
        if request.method == "GET":
            stored = repo["manifests"].get(reference)
            if stored is None:
                return httpx.Response(404)
            body, media = stored
            return httpx.Response(200, headers={"Content-Type": media, "Docker-Content-Digest": digest_of(body)}, content=body)
        return httpx.Response(405)

    def _blob(self, request: httpx.Request, host: str, name: str, digest: str) -> httpx.Response:
        blob = self.repo(host, name)["blobs"].get(digest)
        if blob is None:
            return httpx.Response(404)
        if self.redirect_blobs:
            return httpx.Response(307, headers={"Location": "https://%s/blob/%s/%s?repo=%s" % (CDN, host, digest, name)})
        return httpx.Response(200, content=blob)


def client_for(registry: FakeRegistry) -> RegistryClient:
    credentials = lambda host: READ_CREDENTIAL if host == RUNTIME_REGISTRY else None  # noqa: E731
    return RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(registry.handler)),
        credentials=credentials,
        address_resolver=public_test_resolver,
    )


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_reference_parsing_requires_a_registry_host_and_a_tag_or_digest():
    digest = "sha256:" + "a" * 64
    parsed = images.parse_reference("ghcr.io/acme/agent:v3@" + digest)
    assert (parsed.registry, parsed.repository, parsed.tag, parsed.digest) == ("ghcr.io", "acme/agent", "v3", digest)
    assert str(parsed) == "ghcr.io/acme/agent:v3@" + digest and parsed.name == "ghcr.io/acme/agent"
    tagged = images.parse_reference("ghcr.io/acme/agent:v3")
    assert (tagged.tag, tagged.digest, tagged.selector, str(tagged)) == ("v3", None, "v3", "ghcr.io/acme/agent:v3")
    hub = images.parse_reference("docker.io/python@" + digest)
    assert hub.repository == "library/python" and hub.api_registry == images.DOCKER_HUB_REGISTRY
    local = images.parse_reference("localhost:5000/team/model@" + digest)
    assert local.registry == "localhost:5000" and local.tag is None
    assert images.parse_repository("Runtime.Example/organizer/runtime") == (
        "runtime.example",
        "organizer/runtime",
    )
    for bad in ("acme/agent@" + digest, "ghcr.io/acme/agent", "ghcr.io/acme/agent@sha256:short", "ghcr.io/Acme/Agent@" + digest, "ghcr.io/acme/agent@" + digest + " ", "", None, "ghcr.io/acme/agent:bad tag@" + digest):
        with pytest.raises(ImageError) as excinfo:
            images.parse_reference(bad)
        assert excinfo.value.rule_id == images.RULE_REFERENCE_INVALID
    with pytest.raises(ImageError):
        images.parse_repository("ghcr.io/acme/agent@" + digest)


@pytest.mark.parametrize(
    "host",
    ("127.0.0.1", "10.0.0.8", "100.64.0.1", "169.254.169.254", "192.168.1.4", "224.0.0.1", "240.0.0.1", "0.0.0.0"),
)
def test_registry_client_refuses_non_public_literal_hosts_before_a_request(host):
    requests = []
    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500))),
        address_resolver=public_test_resolver,
    )
    reference = images.parse_reference("%s/acme/agent@sha256:%s" % (host, "a" * 64))
    with pytest.raises(ImageError, match="not public") as excinfo:
        client.get_manifest(reference)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert requests == []


def test_registry_client_refuses_a_name_that_resolves_to_a_private_address():
    requests = []

    def resolver(host, _port):
        return (PUBLIC_TEST_ADDRESS, "10.0.0.8") if host == SOURCE else (PUBLIC_TEST_ADDRESS,)

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(lambda request: requests.append(request) or httpx.Response(500))),
        address_resolver=resolver,
    )
    reference = images.parse_reference("%s/acme/agent@sha256:%s" % (SOURCE, "a" * 64))
    with pytest.raises(ImageError, match="not public") as excinfo:
        client.get_manifest(reference)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert requests == []


def test_registry_client_refuses_a_private_connected_peer_after_public_dns():
    class PrivatePeer:
        @staticmethod
        def get_extra_info(name):
            return ("10.0.0.8", 443) if name == "server_addr" else None

    def handler(_request):
        return httpx.Response(500, extensions={"network_stream": PrivatePeer()})

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        address_resolver=public_test_resolver,
    )
    reference = images.parse_reference("%s/acme/agent@sha256:%s" % (SOURCE, "a" * 64))
    with pytest.raises(ImageError, match="peer is not public") as excinfo:
        client.get_manifest(reference)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE


def test_registry_client_refuses_a_private_token_realm_without_contacting_it():
    requests = []

    def handler(request):
        requests.append((request.url.host, request.headers.get("authorization")))
        return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer realm="https://169.254.169.254/token",scope="repository:acme/agent:pull"'})

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        credentials=lambda host: ("operator", "registry-secret") if host == SOURCE else None,
        address_resolver=public_test_resolver,
    )
    reference = images.parse_reference("%s/acme/agent@sha256:%s" % (SOURCE, "a" * 64))
    with pytest.raises(ImageError, match="not public") as excinfo:
        client.get_manifest(reference)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert requests == [(SOURCE, "Basic " + base64.b64encode(b"operator:registry-secret").decode("ascii"))]


def test_cross_origin_token_realm_never_receives_the_registry_credential():
    registry = FakeRegistry(bearer_hosts=(SOURCE,))
    image = simple_image()
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(registry.handler)),
        credentials=lambda host: ("operator", "registry-secret") if host == SOURCE else None,
        address_resolver=public_test_resolver,
    )
    body, _media = client.get_manifest(reference)
    assert body == image["manifest"]
    assert registry.token_requests == [("repository:acme/agent:pull", None)]
    assert any(host == SOURCE and authorization and authorization.startswith("Basic ") for _method, host, _path, authorization in registry.requests)


def test_registry_token_response_is_size_bounded():
    def handler(request):
        if request.url.host == "auth.example":
            return httpx.Response(200, content=b'{"token":"' + b"x" * images.MAX_TOKEN_RESPONSE_BYTES + b'"}')
        return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer realm="%s",scope="repository:acme/agent:pull"' % AUTH})

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        address_resolver=public_test_resolver,
    )
    reference = images.parse_reference("%s/acme/agent@sha256:%s" % (SOURCE, "a" * 64))
    with pytest.raises(ImageError, match="exceeds") as excinfo:
        client.get_manifest(reference)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE


def test_blob_stream_stops_at_the_shared_transfer_deadline():
    now = [0.0]
    payload = b"ab"
    seen_timeouts = []

    class SlowStream(httpx.SyncByteStream):
        def __iter__(self):
            yield payload[:1]
            now[0] = 11.0
            yield payload[1:]

    def handler(request):
        seen_timeouts.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, stream=SlowStream())

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(handler)),
        address_resolver=public_test_resolver,
        clock=lambda: now[0],
    )
    received = []
    with pytest.raises(ImageError, match="deadline exceeded") as excinfo:
        client.stream_blob(
            SOURCE,
            "acme/agent",
            digest_of(payload),
            expected_size=len(payload),
            sink=received.append,
            deadline=10.0,
        )
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert received == [payload[:1]] and seen_timeouts == [10.0]


def test_blob_redirect_to_a_private_host_is_refused_before_a_request():
    registry = FakeRegistry()
    image = simple_image()
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    requests = []

    def handler(request):
        requests.append(request.url.host)
        if request.url.host == SOURCE and request.method == "GET" and "/blobs/" in request.url.path:
            return httpx.Response(307, headers={"Location": "https://169.254.169.254/latest/meta-data"})
        return registry.handler(request)

    client = RegistryClient(
        http=httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True),
        address_resolver=public_test_resolver,
    )
    with pytest.raises(ImageError, match="not public") as excinfo:
        images.resolve_image(client, reference, ImageRules())
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert "169.254.169.254" not in requests


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_pins_the_amd64_child_of_an_index():
    registry = FakeRegistry(bearer_hosts=(SOURCE,), redirect_blobs=True)
    amd64 = simple_image()
    arm64 = simple_image(arch="arm64")
    attestation = build_image(layers=[layer_tar([("x", "file", b"attest")])], entrypoint=["/bin/true"], os_name="unknown", arch="unknown")
    index = build_index([(arm64, "linux", "arm64", False), (attestation, "unknown", "unknown", True), (amd64, "linux", "amd64", False)])
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", index))
    descriptor = images.resolve_image(client_for(registry), reference, ImageRules())
    assert descriptor.image_digest == amd64["digest"]
    assert descriptor.reference == ImageReference(SOURCE, "acme/agent", amd64["digest"])
    # Anonymous pull: the bearer came from the challenge, the blob fetch followed the redirect without it.
    assert registry.token_requests and all(auth is None for _scope, auth in registry.token_requests)
    assert any(host == CDN for _method, host, _path, _auth in registry.requests)


def test_a_single_platform_manifest_resolves_directly_and_a_docker_manifest_is_accepted():
    registry = FakeRegistry()
    image = simple_image(entrypoint=None, cmd=["node", "agent.js"], env=None, workdir="")
    image["media"] = images.MANIFEST_DOCKER
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    descriptor = images.resolve_image(client_for(registry), reference, ImageRules())
    assert descriptor.image_digest == image["digest"]


def test_a_tag_is_resolved_once_to_the_manifest_digest():
    registry = FakeRegistry()
    image = simple_image(entrypoint=None, cmd=None, env=["LAB_ARENA_OUTPUT_PATH=/ignored"], workdir="relative")
    registry.put_image(SOURCE, "acme/agent", image)
    registry.repo(SOURCE, "acme/agent")["manifests"]["v3"] = (image["manifest"], image["media"])
    submitted = images.parse_reference("%s/acme/agent:v3" % SOURCE)
    descriptor = images.resolve_image(client_for(registry), submitted, ImageRules())
    assert descriptor.image_digest == image["digest"]
    assert descriptor.reference == ImageReference(SOURCE, "acme/agent", image["digest"])
    assert str(descriptor.reference) == "%s/acme/agent@%s" % (SOURCE, image["digest"])


@pytest.mark.parametrize(
    "case, rule",
    [
        ("zstd_layer", images.RULE_LAYER_UNSUPPORTED),
        ("too_many_layers", images.RULE_TOO_MANY_LAYERS),
        ("too_large", images.RULE_TOO_LARGE),
        ("arm_config", images.RULE_PLATFORM_UNSUPPORTED),
        ("index_without_amd64", images.RULE_PLATFORM_UNSUPPORTED),
        ("missing", images.RULE_UNAVAILABLE),
        ("tampered", images.RULE_DIGEST_MISMATCH),
    ],
)
def test_resolve_refuses_images_outside_the_rules(case, rule):
    registry = FakeRegistry()
    rules = ImageRules()
    if case == "zstd_layer":
        image = simple_image(layer_media="application/vnd.oci.image.layer.v1.tar+zstd")
    elif case == "too_many_layers":
        image = simple_image(layers=[layer_tar([("a", "file", b"1")]), layer_tar([("b", "file", b"2")])])
        rules = ImageRules(max_layers=1)
    elif case == "too_large":
        image = simple_image()
        rules = ImageRules(max_image_bytes=8)
    elif case == "arm_config":
        image = simple_image(arch="arm64")
    elif case == "index_without_amd64":
        image = build_index([(simple_image(arch="arm64"), "linux", "arm64", False)])
    elif case == "missing":
        image = simple_image()
        reference = images.parse_reference("%s/acme/agent@%s" % (SOURCE, image["digest"]))
        with pytest.raises(ImageError) as excinfo:
            images.resolve_image(client_for(registry), reference, rules)
        assert excinfo.value.rule_id == rule
        return
    else:  # tampered: the registry serves other bytes under the digest
        image = simple_image()
        registry.put_image(SOURCE, "acme/agent", image)
        registry.repo(SOURCE, "acme/agent")["manifests"][image["digest"]] = (image["manifest"] + b" ", image["media"])
        with pytest.raises(ImageError) as excinfo:
            images.resolve_image(client_for(registry), images.parse_reference("%s/acme/agent@%s" % (SOURCE, image["digest"])), rules)
        assert excinfo.value.rule_id == rule
        return
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    with pytest.raises(ImageError) as excinfo:
        images.resolve_image(client_for(registry), reference, rules)
    assert excinfo.value.rule_id == rule


def test_image_rules_round_trip_through_their_document():
    rules = ImageRules(max_image_bytes=123, max_layers=4)
    document = rules.to_document()
    assert document["schema_version"] == images.IMAGE_RULES_SCHEMA_VERSION and document["platform"] == {"os": "linux", "architecture": "amd64"}
    assert ImageRules.from_document(document) == rules
    with pytest.raises(ImageError):
        ImageRules(max_layers=0)


# ---------------------------------------------------------------------------
# Root filesystem
# ---------------------------------------------------------------------------


def test_materialize_rootfs_applies_layers_whiteouts_and_hardening(tmp_path):
    registry = FakeRegistry(redirect_blobs=True)
    first = layer_tar([
        ("bin", "dir"),
        ("bin/app", "file", b"#!/bin/sh\n", 0o4755),
        ("bin/sh", "symlink", "/bin/busybox"),
        ("etc", "dir"),
        ("etc/old.conf", "file", b"old"),
        ("data", "dir"),
        ("data/a", "file", b"a"),
        ("data/b", "file", b"b"),
        ("dev", "dir"),
        ("dev/null", "chr"),
    ])
    second = layer_tar([
        ("etc/.wh.old.conf", "file", b""),
        ("data/.wh..wh..opq", "file", b""),
        ("data/c", "file", b"c"),
        ("app", "dir"),
        ("app/main.py", "file", b"print('hi')\n"),
        ("app/main2.py", "hardlink", "app/main.py"),
    ], compressed=False)
    image = build_image(layers=[first, second], layer_media=images.LAYER_GZIP_MEDIA_TYPES[0], entrypoint=["/bin/app"])
    manifest = json.loads(image["manifest"])
    manifest["layers"][1]["mediaType"] = images.LAYER_TAR_MEDIA_TYPES[0]
    image["manifest"] = json.dumps(manifest, sort_keys=True).encode("utf-8")
    image["digest"] = digest_of(image["manifest"])
    reference = images.parse_reference(registry.put_image(RUNTIME_REGISTRY, "organizer/runtime", image))
    result = images.materialize_rootfs(client_for(registry), reference, tmp_path / "image", rules=ImageRules())
    rootfs = tmp_path / "image" / "rootfs"
    assert result["layers"] == 2 and result["image_digest"] == image["digest"]
    assert stat.S_IMODE(rootfs.stat().st_mode) == 0o755
    assert not (rootfs / "etc" / "old.conf").exists() and (rootfs / "etc").is_dir()
    assert sorted(os.listdir(rootfs / "data")) == ["c"]
    assert (rootfs / "app" / "main.py").read_bytes() == b"print('hi')\n" and (rootfs / "app" / "main2.py").read_bytes() == b"print('hi')\n"
    mode = stat.S_IMODE((rootfs / "bin" / "app").stat().st_mode)
    assert mode & stat.S_ISUID == 0 and mode & stat.S_IXUSR
    assert os.readlink(rootfs / "bin" / "sh") == "/bin/busybox"
    assert not (rootfs / "dev" / "null").exists() and (rootfs / "dev").is_dir()
    assert not list((tmp_path / "image").glob("lab-arena-layers-*"))  # the spool is gone
    expected_auth = "Basic " + base64.b64encode(
        ("%s:%s" % READ_CREDENTIAL).encode("utf-8")
    ).decode("ascii")
    runtime_requests = [
        authorization
        for _method, host, _path, authorization in registry.requests
        if host == RUNTIME_REGISTRY
    ]
    assert runtime_requests and set(runtime_requests) == {expected_auth}


def test_materialize_rootfs_bounds_total_tar_members_across_layers(tmp_path, monkeypatch):
    registry = FakeRegistry()
    layers = [
        layer_tar([("one", "dir"), ("two", "dir")]),
        layer_tar([("three", "dir")]),
    ]
    image = build_image(layers=layers, entrypoint=["/bin/app"])
    reference = images.parse_reference(registry.put_image(RUNTIME_REGISTRY, "organizer/runtime", image))
    monkeypatch.setattr(images, "MAX_IMAGE_TAR_MEMBERS", 2)
    with pytest.raises(ImageError, match="tar-member budget") as excinfo:
        images.materialize_rootfs(client_for(registry), reference, tmp_path / "image", rules=ImageRules())
    assert excinfo.value.rule_id == images.RULE_TOO_LARGE


@pytest.mark.parametrize(
    "case, rule",
    [
        ("parent_escape", images.RULE_LAYER_INVALID),
        ("hardlink_escape", images.RULE_LAYER_INVALID),
        ("symlink_escape", images.RULE_LAYER_INVALID),
        ("budget", images.RULE_TOO_LARGE),
        ("corrupt_blob", images.RULE_DIGEST_MISMATCH),
        ("not_a_tar", images.RULE_LAYER_INVALID),
    ],
)
def test_materialize_rootfs_refuses_hostile_layers(tmp_path, case, rule):
    registry = FakeRegistry()
    rules = ImageRules()
    if case == "parent_escape":
        layers = [layer_tar([("../escape", "file", b"x")])]
    elif case == "hardlink_escape":
        layers = [layer_tar([("passwd", "hardlink", "../../etc/passwd")])]
    elif case == "symlink_escape":
        layers = [layer_tar([("x", "symlink", str(tmp_path / "outside")), ("x/inner", "file", b"x")])]
    elif case == "budget":
        layers = [layer_tar([("big", "file", b"z" * 100)])]
        rules = ImageRules(max_rootfs_bytes=10)
    elif case == "corrupt_blob":
        layers = [layer_tar([("a", "file", b"a")])]
    else:
        layers = [gzip.compress(b"this is not a tar archive", mtime=0)]
    image = build_image(layers=layers, entrypoint=["/bin/app"])
    reference = images.parse_reference(registry.put_image(RUNTIME_REGISTRY, "organizer/runtime", image))
    if case == "corrupt_blob":
        registry.repo(RUNTIME_REGISTRY, "organizer/runtime")["blobs"][digest_of(layers[0])] = layers[0][:-1] + b"?"
    with pytest.raises(ImageError) as excinfo:
        images.materialize_rootfs(client_for(registry), reference, tmp_path / "image", rules=rules)
    assert excinfo.value.rule_id == rule
    (tmp_path / "outside").mkdir(exist_ok=True)
    assert not list((tmp_path / "outside").iterdir())

"""Images by digest: references, resolution, mirroring, and root filesystem materialization.

A fake OCI registry (one per host, served through ``httpx.MockTransport``)
exercises the anonymous bearer flow for pulls, the credentialed flow for the
Arena's pushes, blob redirects to a content store, cross-repository mounts,
and byte-identical manifest mirroring. The extractor tests apply real layer
tarballs, including whiteouts and hostile members.
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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs

import httpx
import pytest

from lab_arena import images
from lab_arena.images import ImageError, ImageReference, ImageRules, RegistryClient

SOURCE = "registry.example"
ARENA = "arena.example"
CDN = "cdn.example"
AUTH = "https://auth.example/token"
PUSH_CREDENTIAL = ("arena", "push-secret")


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

_UPLOAD_RE = re.compile(r"^/v2/(.+)/blobs/uploads/(.*)$")
_MANIFEST_RE = re.compile(r"^/v2/(.+)/manifests/(.+)$")
_BLOB_RE = re.compile(r"^/v2/(.+)/blobs/(sha256:[0-9a-f]{64})$")


class FakeRegistry:
    """In-memory registries keyed by host; optional bearer auth and blob redirects."""

    def __init__(self, *, bearer_hosts: Tuple[str, ...] = (), redirect_blobs: bool = False, refuse_mounts: bool = False) -> None:
        self.repos: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.uploads: Dict[str, Tuple[str, str]] = {}
        self.bearer_hosts = bearer_hosts
        self.redirect_blobs = redirect_blobs
        self.refuse_mounts = refuse_mounts
        self.requests: List[Tuple[str, str, str, Optional[str]]] = []
        self.token_requests: List[Tuple[str, Optional[str]]] = []
        self._upload_counter = 0

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
        if host == "auth.example":
            return self._token(request)
        if host == CDN:
            assert authorization is None, "redirected blob fetch must not carry the registry credential"
            _, _, source_host, digest = path.split("/", 3)
            name = (parse_qs(request.url.query.decode("utf-8")).get("repo") or [""])[0]
            blob = self.repo(source_host, name)["blobs"].get(digest)
            return httpx.Response(200 if blob is not None else 404, content=blob or b"")
        if host in self.bearer_hosts:
            # Tokens are scoped like a real registry's: a pull token never authorizes a push.
            scope_match = _UPLOAD_RE.match(path) or _MANIFEST_RE.match(path) or _BLOB_RE.match(path)
            name = scope_match.group(1) if scope_match else "unknown"
            action = "push" if request.method in ("POST", "PUT", "PATCH", "DELETE") else "pull"
            granted = authorization[len("Bearer tok:"):] if authorization and authorization.startswith("Bearer tok:") else ""
            parts = granted.split(":", 2)
            allowed = len(parts) == 3 and parts[0] == "repository" and parts[1] == name and action in parts[2].split(",")
            if not allowed:
                actions = "pull,push" if action == "push" else "pull"
                challenge = 'Bearer realm="%s",service="registry",scope="repository:%s:%s"' % (AUTH, name, actions)
                return httpx.Response(401, headers={"WWW-Authenticate": challenge})
        upload = _UPLOAD_RE.match(path)
        if upload:
            return self._upload(request, host, upload.group(1), upload.group(2))
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
        if scope and "push" in scope:
            expected = "Basic " + base64.b64encode(("%s:%s" % PUSH_CREDENTIAL).encode()).decode()
            if authorization != expected:
                return httpx.Response(401)
        return httpx.Response(200, json={"token": "tok:" + (scope or "")})

    def _manifest(self, request: httpx.Request, host: str, name: str, reference: str) -> httpx.Response:
        repo = self.repo(host, name)
        if request.method in ("GET", "HEAD"):
            stored = repo["manifests"].get(reference)
            if stored is None:
                return httpx.Response(404)
            body, media = stored
            return httpx.Response(200, headers={"Content-Type": media, "Docker-Content-Digest": digest_of(body)}, content=body if request.method == "GET" else b"")
        if request.method == "PUT":
            body = request.content
            digest = digest_of(body)
            if digest != reference:
                return httpx.Response(400)
            repo["manifests"][reference] = (body, request.headers.get("content-type", ""))
            return httpx.Response(201, headers={"Docker-Content-Digest": digest})
        return httpx.Response(405)

    def _blob(self, request: httpx.Request, host: str, name: str, digest: str) -> httpx.Response:
        blob = self.repo(host, name)["blobs"].get(digest)
        if blob is None:
            return httpx.Response(404)
        if request.method == "HEAD":
            return httpx.Response(200, headers={"Content-Length": str(len(blob))})
        if self.redirect_blobs:
            return httpx.Response(307, headers={"Location": "https://%s/blob/%s/%s?repo=%s" % (CDN, host, digest, name)})
        return httpx.Response(200, content=blob)

    def _upload(self, request: httpx.Request, host: str, name: str, upload_id: str) -> httpx.Response:
        query = parse_qs(request.url.query.decode("utf-8"))
        if request.method == "POST" and not upload_id:
            mount = (query.get("mount") or [None])[0]
            source = (query.get("from") or [None])[0]
            if mount and source:
                if self.refuse_mounts:
                    return httpx.Response(405)
                blob = self.repo(host, source)["blobs"].get(mount)
                if blob is not None:
                    self.repo(host, name)["blobs"][mount] = blob
                    return httpx.Response(201)
            self._upload_counter += 1
            identifier = "upload-%d" % self._upload_counter
            self.uploads[identifier] = (host, name)
            return httpx.Response(202, headers={"Location": "/v2/%s/blobs/uploads/%s" % (name, identifier)})
        if request.method == "PUT" and upload_id in self.uploads:
            expected = (query.get("digest") or [""])[0]
            body = request.read()
            if digest_of(body) != expected or request.headers.get("content-length") != str(len(body)):
                return httpx.Response(400)
            self.repo(host, name)["blobs"][expected] = body
            del self.uploads[upload_id]
            return httpx.Response(201)
        return httpx.Response(404)


def client_for(registry: FakeRegistry) -> RegistryClient:
    credentials = lambda host: PUSH_CREDENTIAL if host == ARENA else None  # noqa: E731
    return RegistryClient(http=httpx.Client(transport=httpx.MockTransport(registry.handler)), credentials=credentials)


# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------


def test_reference_parsing_requires_a_registry_host_and_a_digest():
    digest = "sha256:" + "a" * 64
    parsed = images.parse_reference("ghcr.io/acme/agent:v3@" + digest)
    assert (parsed.registry, parsed.repository, parsed.tag, parsed.digest) == ("ghcr.io", "acme/agent", "v3", digest)
    assert str(parsed) == "ghcr.io/acme/agent:v3@" + digest and parsed.name == "ghcr.io/acme/agent"
    hub = images.parse_reference("docker.io/python@" + digest)
    assert hub.repository == "library/python" and hub.api_registry == images.DOCKER_HUB_REGISTRY
    local = images.parse_reference("localhost:5000/team/model@" + digest)
    assert local.registry == "localhost:5000" and local.tag is None
    assert images.parse_repository("Arena.Example/lab-arena/models") == ("arena.example", "lab-arena/models")
    for bad in ("acme/agent@" + digest, "ghcr.io/acme/agent", "ghcr.io/acme/agent@sha256:short", "ghcr.io/Acme/Agent@" + digest, "ghcr.io/acme/agent@" + digest + " ", "", None, "ghcr.io/acme/agent:bad tag@" + digest):
        with pytest.raises(ImageError) as excinfo:
            images.parse_reference(bad)
        assert excinfo.value.rule_id == images.RULE_REFERENCE_INVALID
    with pytest.raises(ImageError):
        images.parse_repository("ghcr.io/acme/agent@" + digest)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def test_resolve_pins_the_amd64_child_of_an_index_and_reads_the_process():
    registry = FakeRegistry(bearer_hosts=(SOURCE,), redirect_blobs=True)
    amd64 = simple_image()
    arm64 = simple_image(arch="arm64")
    attestation = build_image(layers=[layer_tar([("x", "file", b"attest")])], entrypoint=["/bin/true"], os_name="unknown", arch="unknown")
    index = build_index([(arm64, "linux", "arm64", False), (attestation, "unknown", "unknown", True), (amd64, "linux", "amd64", False)])
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", index))
    descriptor = images.resolve_image(client_for(registry), reference, ImageRules())
    assert descriptor.image_digest == amd64["digest"] and descriptor.submitted_digest == index["digest"]
    assert descriptor.reference == ImageReference(SOURCE, "acme/agent", amd64["digest"])
    assert descriptor.entry_command == ("python3", "/app/main.py", "--fast")
    assert dict(descriptor.environment) == {"PATH": "/usr/local/bin:/usr/bin", "APP_MODE": "fast"} and descriptor.working_dir == "/app"
    assert descriptor.image_size_bytes == len(amd64["config"]) + sum(len(layer) for layer in amd64["layers"])
    document = descriptor.to_document()
    assert document["entry_command"] == ["python3", "/app/main.py", "--fast"] and document["layer_count"] == 1 and "manifest_bytes" not in document
    # Anonymous pull: the bearer came from the challenge, the blob fetch followed the redirect without it.
    assert registry.token_requests and all(auth is None for _scope, auth in registry.token_requests)
    assert any(host == CDN for _method, host, _path, _auth in registry.requests)


def test_a_single_platform_manifest_resolves_directly_and_a_docker_manifest_is_accepted():
    registry = FakeRegistry()
    image = simple_image(entrypoint=None, cmd=["node", "agent.js"], env=None, workdir="")
    image["media"] = images.MANIFEST_DOCKER
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    descriptor = images.resolve_image(client_for(registry), reference, ImageRules())
    assert descriptor.submitted_digest == descriptor.image_digest == image["digest"]
    assert descriptor.entry_command == ("node", "agent.js") and dict(descriptor.environment) == {} and descriptor.working_dir == ""


@pytest.mark.parametrize(
    "case, rule",
    [
        ("no_entry", images.RULE_NO_ENTRY_COMMAND),
        ("zstd_layer", images.RULE_LAYER_UNSUPPORTED),
        ("too_many_layers", images.RULE_TOO_MANY_LAYERS),
        ("too_large", images.RULE_TOO_LARGE),
        ("arm_config", images.RULE_PLATFORM_UNSUPPORTED),
        ("index_without_amd64", images.RULE_PLATFORM_UNSUPPORTED),
        ("reserved_env", images.RULE_CONFIG_INVALID),
        ("missing", images.RULE_UNAVAILABLE),
        ("tampered", images.RULE_DIGEST_MISMATCH),
    ],
)
def test_resolve_refuses_images_outside_the_public_rules(case, rule):
    registry = FakeRegistry()
    rules = ImageRules()
    if case == "no_entry":
        image = simple_image(entrypoint=None, cmd=None)
    elif case == "zstd_layer":
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
    elif case == "reserved_env":
        image = simple_image(env=["LAB_ARENA_OUTPUT_PATH=/etc/passwd"])
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


def test_image_rules_round_trip_through_their_public_document():
    rules = ImageRules(max_image_bytes=123, max_layers=4)
    document = rules.to_document()
    assert document["schema_version"] == images.IMAGE_RULES_SCHEMA_VERSION and document["platform"] == {"os": "linux", "architecture": "amd64"}
    assert ImageRules.from_document(document) == rules
    with pytest.raises(ImageError):
        ImageRules(max_layers=0)


# ---------------------------------------------------------------------------
# Mirroring
# ---------------------------------------------------------------------------


def test_mirror_copies_blobs_into_the_arena_repository_and_preserves_the_manifest_digest():
    registry = FakeRegistry(bearer_hosts=(SOURCE, ARENA), redirect_blobs=True)
    image = simple_image(layers=[layer_tar([("a", "file", b"first")]), layer_tar([("b", "file", b"second")])])
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    client = client_for(registry)
    descriptor = images.resolve_image(client, reference, ImageRules())
    mirrored = images.mirror_image(client, descriptor, "%s/lab-arena/models" % ARENA)
    assert mirrored == ImageReference(ARENA, "lab-arena/models", image["digest"])
    arena_repo = registry.repo(ARENA, "lab-arena/models")
    assert arena_repo["manifests"][image["digest"]] == (image["manifest"], images.MANIFEST_OCI)
    assert set(arena_repo["blobs"]) == {digest_of(image["config"])} | {digest_of(layer) for layer in image["layers"]}
    # The push credential reached only the Arena registry's token endpoint, for a push scope.
    push_tokens = [(scope, auth) for scope, auth in registry.token_requests if "push" in scope]
    assert push_tokens and all(auth and auth.startswith("Basic ") for _scope, auth in push_tokens)
    assert all("lab-arena/models" in scope for scope, _auth in push_tokens)
    assert all(auth is None for scope, auth in registry.token_requests if "acme/agent" in scope)  # the source is pulled anonymously
    assert not any(host == SOURCE and auth and auth.startswith("Basic") for _m, host, _p, auth in registry.requests)
    # A second mirror finds every blob and the manifest already present.
    uploads_before = len([r for r in registry.requests if r[0] == "PUT"])
    again = images.mirror_image(client, descriptor, "%s/lab-arena/models" % ARENA)
    assert again == mirrored
    puts_after = [r for r in registry.requests if r[0] == "PUT"]
    assert len(puts_after) == uploads_before + 1  # only the (idempotent) manifest write repeats
    # The mirrored image resolves from the Arena repository with the same process.
    resolved = images.resolve_image(client, mirrored, ImageRules())
    assert resolved.image_digest == image["digest"] and resolved.entry_command == descriptor.entry_command


def test_mirror_mounts_blobs_within_the_same_registry_and_falls_back_to_upload():
    for refuse in (False, True):
        registry = FakeRegistry(refuse_mounts=refuse)
        image = simple_image()
        reference = images.parse_reference(registry.put_image(ARENA, "miners/alice", image))
        client = client_for(registry)
        descriptor = images.resolve_image(client, reference, ImageRules())
        mirrored = images.mirror_image(client, descriptor, "%s/lab-arena/models" % ARENA)
        assert mirrored.repository == "lab-arena/models" and registry.repo(ARENA, "lab-arena/models")["manifests"][image["digest"]][0] == image["manifest"]
        mounts = [r for r in registry.requests if r[0] == "POST" and "/blobs/uploads/" in r[2]]
        blob_puts = [r for r in registry.requests if r[0] == "PUT" and "/blobs/uploads/" in r[2]]
        assert mounts, "a same-registry mirror asks for a mount"
        assert bool(blob_puts) == refuse, "uploads happen only when the mount is refused"


def test_mirror_fails_closed_when_the_destination_refuses_the_credential():
    registry = FakeRegistry(bearer_hosts=(ARENA,))
    image = simple_image()
    reference = images.parse_reference(registry.put_image(SOURCE, "acme/agent", image))
    client = RegistryClient(http=httpx.Client(transport=httpx.MockTransport(registry.handler)), credentials=lambda host: ("arena", "wrong") if host == ARENA else None)
    descriptor = images.resolve_image(client, reference, ImageRules())
    with pytest.raises(ImageError) as excinfo:
        images.mirror_image(client, descriptor, "%s/lab-arena/models" % ARENA)
    assert excinfo.value.rule_id == images.RULE_UNAVAILABLE
    assert image["digest"] not in registry.repo(ARENA, "lab-arena/models")["manifests"]


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
    reference = images.parse_reference(registry.put_image(ARENA, "lab-arena/models", image))
    result = images.materialize_rootfs(client_for(registry), reference, tmp_path / "image", rules=ImageRules())
    rootfs = tmp_path / "image" / "rootfs"
    assert result["layers"] == 2 and result["image_digest"] == image["digest"]
    assert not (rootfs / "etc" / "old.conf").exists() and (rootfs / "etc").is_dir()
    assert sorted(os.listdir(rootfs / "data")) == ["c"]
    assert (rootfs / "app" / "main.py").read_bytes() == b"print('hi')\n" and (rootfs / "app" / "main2.py").read_bytes() == b"print('hi')\n"
    mode = stat.S_IMODE((rootfs / "bin" / "app").stat().st_mode)
    assert mode & stat.S_ISUID == 0 and mode & stat.S_IXUSR
    assert os.readlink(rootfs / "bin" / "sh") == "/bin/busybox"
    assert not (rootfs / "dev" / "null").exists() and (rootfs / "dev").is_dir()
    assert not list((tmp_path / "image").glob("lab-arena-layers-*"))  # the spool is gone


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
    reference = images.parse_reference(registry.put_image(ARENA, "lab-arena/models", image))
    if case == "corrupt_blob":
        registry.repo(ARENA, "lab-arena/models")["blobs"][digest_of(layers[0])] = layers[0][:-1] + b"?"
    with pytest.raises(ImageError) as excinfo:
        images.materialize_rootfs(client_for(registry), reference, tmp_path / "image", rules=rules)
    assert excinfo.value.rule_id == rule
    (tmp_path / "outside").mkdir(exist_ok=True)
    assert not list((tmp_path / "outside").iterdir())

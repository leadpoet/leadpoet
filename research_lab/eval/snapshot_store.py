"""Provider snapshot recording/replay for the L1 dev-eval rung (§6.3-1).

Every hypothesis currently costs a full live-provider benchmark. This module
freezes real provider traffic (Exa search, Scrapingdog fetches, OpenRouter
calls) for a frozen dev ICP set once, then replays it deterministically so the
inner loop can score candidates for ~$0 per iteration.

Two modes:

* ``record`` — wrap outbound provider calls at the same seams the bug-35
  diagnostics hooks use (urllib / httpx / requests) and persist
  request-key -> response JSON.
* ``replay`` — serve recorded responses deterministically. Unknown requests
  raise a typed :class:`SnapshotMiss` under the default ``strict`` miss
  policy, or return a synthesized empty provider result under ``empty``
  (``RESEARCH_LAB_DEV_SNAPSHOT_MISS_POLICY``). Replay NEVER falls through to
  the live network.

Request key scheme (stable contract shared with the in-container bootstraps):

    provider | METHOD | endpoint | sha256_json(significant_params)

where ``provider`` is inferred from the host (``exa`` / ``scrapingdog`` /
``openrouter`` / the bare host otherwise), ``endpoint`` is the lowercased
``host/path`` with the query string removed, and ``significant_params`` is
``{"query": {...}, "body": ...}`` with auth-shaped parameters
(``api_key`` and friends) stripped so replay matches regardless of which key
recorded the traffic. Snapshot files are named by the sha256 of the key.

Storage is a local directory, an immutable S3 prefix, or a signed S3
``current.json`` pointer (``RESEARCH_LAB_DEV_SNAPSHOT_URI``) with the layout::

    <root>/manifest.json
    <root>/snapshots/<sha256-hex>.json

The snapshot-set manifest carries a ``content_hash`` over every stored record
plus the dev ICP set hash so replay integrity is verifiable before scores are
trusted (see :meth:`ProviderSnapshotStore.verify_manifest`).

Container seams: :func:`dev_replay_bootstrap` / :func:`dev_record_bootstrap`
return self-contained Python preambles analogous to
``private_runtime._PROVIDER_DIAGNOSTICS_BOOTSTRAP`` — prepend them to the
adapter bootstrap and point ``RESEARCH_LAB_DEV_SNAPSHOT_DIR`` at a mounted
snapshot directory. In-process runners (tests, future engine fast path) use
:meth:`ProviderSnapshotStore.replay_installed` or call
:meth:`ProviderSnapshotStore.replay` directly.

The live inner-loop runner imports this module. Snapshot records and manifests
remain deterministic by construction: no wall clocks or unseeded randomness;
the optional manifest ``recorded_at`` is caller-supplied by the recording CLI.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

from research_lab.canonical import sha256_json, sha256_text

from .private_runtime import SECRET_MARKERS

SNAPSHOT_SCHEMA_VERSION = "1.0"
SNAPSHOT_RECORD_TYPE = "research_lab_dev_provider_snapshot"
SNAPSHOT_MANIFEST_TYPE = "research_lab_dev_snapshot_set"
SNAPSHOT_READY_TYPE = "research_lab_dev_snapshot_ready"
SNAPSHOT_POINTER_TYPE = "research_lab_dev_snapshot_pointer"
SNAPSHOT_URI_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_URI"
SNAPSHOT_MISS_POLICY_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_MISS_POLICY"
SNAPSHOT_DIR_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_DIR"
MISS_POLICY_STRICT = "strict"
MISS_POLICY_EMPTY = "empty"
MISS_POLICIES = (MISS_POLICY_STRICT, MISS_POLICY_EMPTY)
MODE_RECORD = "record"
MODE_REPLAY = "replay"
SNAPSHOT_SUBDIR = "snapshots"
MANIFEST_NAME = "manifest.json"
DEV_ICPS_NAME = "dev_icps.json"
READY_NAME = "READY.json"
POINTER_NAME = "current.json"
RECORD_FAILURES_NAME = "record_failures.jsonl"


def build_snapshot_pointer_document(
    *,
    snapshot_uri: str,
    manifest_hash: str,
    ready_hash: str,
    recorded_at: str,
) -> dict[str, Any]:
    """Build an unsigned pointer to one already-published immutable snapshot."""
    payload = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "pointer_type": SNAPSHOT_POINTER_TYPE,
        "snapshot_uri": str(snapshot_uri).rstrip("/"),
        "manifest_hash": str(manifest_hash),
        "ready_hash": str(ready_hash),
        "recorded_at": str(recorded_at),
    }
    return {**payload, "pointer_hash": sha256_json(payload)}


def load_snapshot_pointer_document(pointer_uri: str) -> dict[str, Any]:
    """Load a local or S3 pointer object without resolving its target."""
    uri = str(pointer_uri or "").strip()
    if uri.startswith("s3://"):
        bucket, key = _parse_s3_object_uri(uri)
        try:
            response = _s3_client().get_object(Bucket=bucket, Key=key)
        except Exception as exc:
            raise DevSnapshotStoreError(
                f"snapshot pointer could not be loaded: {type(exc).__name__}"
            ) from exc
        raw = response["Body"].read().decode("utf-8")
    else:
        path = Path(uri).expanduser()
        if not path.is_file():
            raise DevSnapshotStoreError(f"snapshot pointer does not exist: {path}")
        raw = path.read_text(encoding="utf-8")
    decoded = json.loads(raw)
    if not isinstance(decoded, Mapping):
        raise DevSnapshotStoreError("snapshot pointer must be a JSON object")
    return dict(decoded)


def verify_snapshot_pointer_document(
    pointer_uri: str,
    pointer: Mapping[str, Any] | None = None,
    *,
    require_signature: bool | None = None,
) -> dict[str, Any]:
    """Verify pointer integrity, signature, and same-base target containment."""
    errors: list[str] = []
    try:
        doc = dict(pointer) if pointer is not None else load_snapshot_pointer_document(pointer_uri)
    except Exception as exc:
        return {
            "passed": False,
            "errors": [f"pointer_load_error:{type(exc).__name__}"],
            "snapshot_uri": "",
        }
    signature_fields = {"signature_b64", "kms_key_id", "signing_algorithm"}
    payload = {
        key: value
        for key, value in doc.items()
        if key not in signature_fields and key != "pointer_hash"
    }
    pointer_hash = sha256_json(payload)
    if str(doc.get("pointer_hash") or "") != pointer_hash:
        errors.append("pointer_hash_mismatch")
    if str(doc.get("pointer_type") or "") != SNAPSHOT_POINTER_TYPE:
        errors.append("pointer_type_mismatch")
    target = str(doc.get("snapshot_uri") or "").rstrip("/")
    manifest_hash = str(doc.get("manifest_hash") or "")
    ready_hash = str(doc.get("ready_hash") or "")
    if not manifest_hash.startswith("sha256:"):
        errors.append("pointer_manifest_hash_invalid")
    if not ready_hash.startswith("sha256:"):
        errors.append("pointer_ready_hash_invalid")
    try:
        _validate_pointer_target(str(pointer_uri), target)
    except DevSnapshotStoreError as exc:
        errors.append(str(exc))

    require_sig = str(pointer_uri).startswith("s3://") if require_signature is None else bool(require_signature)
    signature = str(doc.get("signature_b64") or "")
    key_id = str(doc.get("kms_key_id") or "")
    algorithm = str(doc.get("signing_algorithm") or "")
    if require_sig and not (signature and key_id and algorithm):
        errors.append("pointer_signature_missing")
    elif signature and key_id and algorithm:
        try:
            response = _s3_client_for("kms").verify(
                KeyId=key_id,
                Message=pointer_hash.encode("utf-8"),
                MessageType="RAW",
                Signature=base64.b64decode(signature, validate=True),
                SigningAlgorithm=algorithm,
            )
            if not bool(response.get("SignatureValid")):
                errors.append("pointer_signature_invalid")
        except Exception as exc:
            errors.append(f"pointer_signature_error:{type(exc).__name__}")
    return {
        "passed": not errors,
        "errors": list(dict.fromkeys(errors)),
        "snapshot_uri": target,
        "manifest_hash": manifest_hash,
        "ready_hash": ready_hash,
        "recorded_at": str(doc.get("recorded_at") or ""),
        "pointer_hash": str(doc.get("pointer_hash") or ""),
    }


def resolve_snapshot_uri(root_uri: str) -> dict[str, Any]:
    """Resolve a signed ``current.json`` pointer or return an immutable root."""
    uri = str(root_uri or "").strip()
    if not uri:
        raise DevSnapshotStoreError("snapshot root URI is required")
    is_pointer = uri.rstrip("/").endswith(f"/{POINTER_NAME}") or (
        not uri.startswith("s3://") and Path(uri).name == POINTER_NAME
    )
    if not is_pointer:
        return {
            "snapshot_uri": uri.rstrip("/"),
            "manifest_hash": "",
            "ready_hash": "",
            "pointer_hash": "",
        }
    verification = verify_snapshot_pointer_document(uri)
    if not verification.get("passed"):
        raise DevSnapshotStoreError(
            "snapshot pointer verification failed: "
            + "; ".join(verification.get("errors") or ())
        )
    return verification

# Auth-shaped query/body parameter names stripped from significant params so
# the request key is independent of which credential recorded the traffic.
# Keep in sync with the copies embedded in the bootstraps below.
_AUTH_PARAM_NAMES = (
    "api_key",
    "apikey",
    "x-api-key",
    "authorization",
    "token",
    "access_token",
    "bearer",
)

# Synthesized bodies for the ``empty`` miss policy: the candidate sees "no
# results" instead of crashing. Keyed by inferred provider; default "{}".
_EMPTY_MISS_BODIES = {
    "exa": '{"results": []}',
    "scrapingdog": "{}",
    "openrouter": "{}",
}
SYNTHESIZED_EMPTY_MARKER = "research_lab_dev_snapshot_synthesized_empty"
SNAPSHOT_MISS_SENTINEL = "RESEARCH_LAB_DEV_SNAPSHOT_MISS:"


class DevSnapshotStoreError(RuntimeError):
    """Raised when the snapshot store cannot operate safely."""


class SnapshotMiss(DevSnapshotStoreError):
    """Raised in strict replay mode for a request with no recorded snapshot."""

    def __init__(self, request_key: str):
        super().__init__(f"no recorded provider snapshot for request: {request_key}")
        self.request_key = request_key


@dataclass(frozen=True)
class SnapshotRequest:
    """Normalized identity of one provider request."""

    provider: str
    method: str
    endpoint: str
    params_hash: str

    @property
    def request_key(self) -> str:
        return f"{self.provider}|{self.method}|{self.endpoint}|{self.params_hash}"

    @property
    def storage_name(self) -> str:
        return sha256_text(self.request_key).split(":", 1)[1]


def provider_for_host(host: str) -> str:
    lowered = str(host or "").strip().lower()
    if "exa.ai" in lowered:
        return "exa"
    if "scrapingdog" in lowered:
        return "scrapingdog"
    if "openrouter" in lowered:
        return "openrouter"
    return lowered or "unknown"


def build_snapshot_request(
    method: str,
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    body: Any = None,
) -> SnapshotRequest:
    """Normalize a provider call into its stable snapshot identity.

    ``url`` may already carry a query string (it always does at the HTTP
    seams); ``params`` merges extra query parameters for callers that keep
    them separate. ``body`` may be bytes/str (JSON is parsed when possible)
    or an already-parsed object.
    """
    split = urlsplit(str(url or ""))
    host = split.netloc.lower().rsplit("@", 1)[-1]
    for default_port in (":80", ":443"):
        if host.endswith(default_port):
            host = host.rsplit(":", 1)[0]
    path = split.path.rstrip("/")
    endpoint = f"{host}{path}" if path else host
    query: dict[str, list[str]] = {}
    for name, value in parse_qsl(split.query, keep_blank_values=True):
        query.setdefault(str(name), []).append(str(value))
    for name, value in dict(params or {}).items():
        query.setdefault(str(name), []).append(str(value))
    significant = {
        "query": _strip_auth_params({name: sorted(values) for name, values in query.items()}),
        "body": _normalized_body(body),
    }
    return SnapshotRequest(
        provider=provider_for_host(host),
        method=str(method or "GET").strip().upper() or "GET",
        endpoint=endpoint,
        params_hash=sha256_json(significant),
    )


def _strip_auth_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in sorted(params.items())
        if str(name).strip().lower() not in _AUTH_PARAM_NAMES
    }


def _normalized_body(body: Any) -> Any:
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", "replace")
    if isinstance(body, str):
        text = body.strip()
        if not text:
            return None
        try:
            body = json.loads(text)
        except json.JSONDecodeError:
            return text
    if isinstance(body, Mapping):
        return _strip_auth_params({str(k): v for k, v in body.items()})
    return body


def default_miss_policy() -> str:
    """Miss policy from ``RESEARCH_LAB_DEV_SNAPSHOT_MISS_POLICY`` (strict default)."""
    raw = str(os.getenv(SNAPSHOT_MISS_POLICY_ENV) or "").strip().lower()
    return raw if raw in MISS_POLICIES else MISS_POLICY_STRICT


def synthesized_empty_response(provider: str) -> dict[str, Any]:
    """The deterministic ``empty``-policy response for one provider."""
    return {
        "status": 200,
        "headers": {"content-type": "application/json"},
        "body_text": _EMPTY_MISS_BODIES.get(provider, "{}"),
        "synthesized": SYNTHESIZED_EMPTY_MARKER,
    }


class ProviderSnapshotStore:
    """Record/replay store for provider responses on a frozen dev ICP set.

    ``root_uri`` is a local directory path or an ``s3://bucket/prefix`` URI;
    it defaults to ``RESEARCH_LAB_DEV_SNAPSHOT_URI``. ``mode`` is ``replay``
    (default) or ``record``. Replay integrity is verifiable through the
    snapshot-set manifest (:meth:`build_manifest` / :meth:`verify_manifest`).
    """

    def __init__(
        self,
        root_uri: str | None = None,
        *,
        mode: str = MODE_REPLAY,
        miss_policy: str | None = None,
    ):
        resolved = str(root_uri if root_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or "").strip()
        if not resolved:
            raise DevSnapshotStoreError(
                f"snapshot root URI is required (arg or {SNAPSHOT_URI_ENV})"
            )
        if mode not in (MODE_RECORD, MODE_REPLAY):
            raise DevSnapshotStoreError(f"unknown snapshot store mode: {mode}")
        policy = str(miss_policy if miss_policy is not None else default_miss_policy()).strip().lower()
        if policy not in MISS_POLICIES:
            raise DevSnapshotStoreError(f"unknown snapshot miss policy: {policy}")
        self.root_uri = resolved
        self.mode = mode
        self.miss_policy = policy
        self._is_s3 = resolved.startswith("s3://")
        if not self._is_s3:
            self._root_path = Path(resolved).expanduser()
            if mode == MODE_RECORD:
                (self._root_path / SNAPSHOT_SUBDIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, *, mode: str = MODE_REPLAY) -> "ProviderSnapshotStore":
        return cls(None, mode=mode)

    # -- recording -----------------------------------------------------------

    def record_response(
        self,
        request: SnapshotRequest,
        *,
        status: int,
        body_text: str,
        content_type: str = "application/json",
    ) -> dict[str, Any]:
        """Persist one provider response under its request key.

        Records are timestamp-free so the snapshot-set content hash is a pure
        function of the recorded traffic. Re-recording the same request
        overwrites in place (last write wins, one file per key).
        """
        if self.mode != MODE_RECORD:
            raise DevSnapshotStoreError("record_response requires a record-mode store")
        record = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "record_type": SNAPSHOT_RECORD_TYPE,
            "request_key": request.request_key,
            "provider": request.provider,
            "method": request.method,
            "endpoint": request.endpoint,
            "params_hash": request.params_hash,
            "response": {
                "status": int(status),
                "headers": {"content-type": str(content_type or "application/json")},
                "body_text": str(body_text),
            },
        }
        if _contains_secret_material(record):
            raise DevSnapshotStoreError(
                f"refusing to record snapshot containing raw secret material: {request.request_key}"
            )
        self._write_text(
            f"{SNAPSHOT_SUBDIR}/{request.storage_name}.json",
            json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )
        return record

    # -- replay --------------------------------------------------------------

    def lookup(self, request: SnapshotRequest) -> dict[str, Any]:
        """Return the recorded response doc for ``request``.

        Strict policy raises :class:`SnapshotMiss` on unknown requests; the
        ``empty`` policy returns :func:`synthesized_empty_response` instead.
        Never touches the network.
        """
        raw = self._read_text(f"{SNAPSHOT_SUBDIR}/{request.storage_name}.json")
        if raw is None:
            if self.miss_policy == MISS_POLICY_EMPTY:
                return synthesized_empty_response(request.provider)
            raise SnapshotMiss(request.request_key)
        record = json.loads(raw)
        response = record.get("response")
        if not isinstance(response, Mapping):
            raise DevSnapshotStoreError(f"corrupt snapshot record for {request.request_key}")
        return dict(response)

    def replay(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
    ) -> dict[str, Any]:
        """Build the request key and look up the recorded response."""
        return self.lookup(build_snapshot_request(method, url, params=params, body=body))

    def fetch(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
        live_fetch: Callable[[], tuple[int, str]] | None = None,
    ) -> dict[str, Any]:
        """Mode-dispatching seam for in-process provider clients.

        Replay mode serves the recorded response. Record mode invokes
        ``live_fetch`` (returning ``(status, body_text)``), persists the
        response, and returns it — the symmetric API a snapshot-aware client
        wrapper uses in both modes.
        """
        request = build_snapshot_request(method, url, params=params, body=body)
        if self.mode == MODE_REPLAY:
            return self.lookup(request)
        if live_fetch is None:
            raise DevSnapshotStoreError("record-mode fetch requires a live_fetch callable")
        status, body_text = live_fetch()
        return dict(self.record_response(request, status=status, body_text=body_text)["response"])

    @contextmanager
    def replay_installed(self) -> Iterator["ProviderSnapshotStore"]:
        """Serve replay responses through in-process HTTP seams.

        Monkeypatches the same seams the bug-35 diagnostics hooks cover —
        ``urllib.request.urlopen``, ``requests.Session.send``,
        ``httpx.Client.send`` and ``httpx.AsyncClient.send`` (the HTTP
        libraries best-effort, only when importable) — so an in-process
        candidate runner needs no snapshot awareness. Patches are
        process-global while active; run ICPs serially.
        """
        if self.mode != MODE_REPLAY:
            raise DevSnapshotStoreError("replay_installed requires a replay-mode store")
        import urllib.request as _urllib_request

        original_urlopen = _urllib_request.urlopen

        def _replay_urlopen(req, *args, **kwargs):  # noqa: ANN001 - urllib contract
            url = getattr(req, "full_url", req if isinstance(req, str) else "")
            method = getattr(req, "get_method", lambda: "GET")()
            data = getattr(req, "data", None)
            response = self.replay(method, str(url), body=data)
            return _FakeUrllibResponse(str(url), response)

        _urllib_request.urlopen = _replay_urlopen
        restore: list[Callable[[], None]] = [
            lambda: setattr(_urllib_request, "urlopen", original_urlopen)
        ]
        try:
            self._install_requests_replay(restore)
            self._install_httpx_replay(restore)
            self._install_aiohttp_replay(restore)
            yield self
        finally:
            for undo in reversed(restore):
                undo()

    def _install_requests_replay(self, restore: list[Callable[[], None]]) -> None:
        try:
            import requests
        except (ImportError, ModuleNotFoundError):
            return
        original_send = requests.Session.send
        store = self

        def _replay_send(session, prepared, *args, **kwargs):  # noqa: ANN001
            doc = store.replay(
                str(prepared.method or "GET"),
                str(prepared.url or ""),
                body=prepared.body,
            )
            response = requests.models.Response()
            response.status_code = int(doc.get("status") or 0)
            response._content = str(doc.get("body_text") or "").encode("utf-8")  # noqa: SLF001
            response.encoding = "utf-8"
            response.headers.update(dict(doc.get("headers") or {}))
            response.url = str(prepared.url or "")
            response.request = prepared
            return response

        requests.Session.send = _replay_send
        restore.append(lambda: setattr(requests.Session, "send", original_send))

    def _install_httpx_replay(self, restore: list[Callable[[], None]]) -> None:
        try:
            import httpx
        except (ImportError, ModuleNotFoundError):
            return
        original_send = httpx.Client.send
        store = self

        def _replay_send(client, request, *args, **kwargs):  # noqa: ANN001
            doc = store.replay(
                str(request.method or "GET"),
                str(request.url or ""),
                body=bytes(getattr(request, "content", b"") or b""),
            )
            return httpx.Response(
                status_code=int(doc.get("status") or 0),
                headers=dict(doc.get("headers") or {}),
                content=str(doc.get("body_text") or "").encode("utf-8"),
                request=request,
            )

        httpx.Client.send = _replay_send
        restore.append(lambda: setattr(httpx.Client, "send", original_send))

        original_async_send = httpx.AsyncClient.send

        async def _replay_async_send(client, request, *args, **kwargs):  # noqa: ANN001
            return _replay_send(client, request, *args, **kwargs)

        httpx.AsyncClient.send = _replay_async_send
        restore.append(lambda: setattr(httpx.AsyncClient, "send", original_async_send))

    def _install_aiohttp_replay(self, restore: list[Callable[[], None]]) -> None:
        try:
            import aiohttp
        except (ImportError, ModuleNotFoundError):
            return
        original_request = aiohttp.ClientSession._request  # noqa: SLF001

        store = self

        async def _replay_request(session, method, str_or_url, *args, **kwargs):  # noqa: ANN001
            body = kwargs.get("json")
            if body is None:
                body = kwargs.get("data")
            doc = store.replay(
                str(method or "GET"),
                str(str_or_url),
                params=kwargs.get("params"),
                body=body,
            )
            response = _FakeAiohttpResponse(str(str_or_url), doc)
            if kwargs.get("raise_for_status") is True:
                response.raise_for_status()
            return response

        aiohttp.ClientSession._request = _replay_request  # noqa: SLF001
        restore.append(
            lambda: setattr(aiohttp.ClientSession, "_request", original_request)
        )

    # -- manifest ------------------------------------------------------------

    def request_keys(self) -> list[str]:
        keys: list[str] = []
        for name in self._list_snapshot_names():
            raw = self._read_text(f"{SNAPSHOT_SUBDIR}/{name}")
            if raw is None:
                continue
            record = json.loads(raw)
            key = str(record.get("request_key") or "")
            if key:
                keys.append(key)
        return sorted(keys)

    def snapshot_count(self) -> int:
        return len(self._list_snapshot_names())

    def content_hash(self) -> str:
        """Hash over every stored record, keyed and sorted by request key."""
        entries: list[list[str]] = []
        for name in sorted(self._list_snapshot_names()):
            raw = self._read_text(f"{SNAPSHOT_SUBDIR}/{name}")
            if raw is None:
                continue
            record = json.loads(raw)
            entries.append([str(record.get("request_key") or name), sha256_json(record)])
        entries.sort()
        return sha256_json(entries)

    def build_manifest(
        self,
        *,
        icp_set_hash: str = "",
        dev_set_manifest: Mapping[str, Any] | None = None,
        recorded_at: str = "",
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the snapshot-set manifest binding traffic to the dev ICP set.

        ``recorded_at`` is caller-supplied (the recording CLI passes its own
        clock) so this library stays wall-clock-free and deterministic.
        """
        dev_manifest = dict(dev_set_manifest or {})
        dev_icps_hash = self.dev_icp_items_hash()
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "manifest_type": SNAPSHOT_MANIFEST_TYPE,
            "snapshot_count": self.snapshot_count(),
            "request_keys": self.request_keys(),
            "content_hash": self.content_hash(),
            "icp_set_hash": str(icp_set_hash or dev_manifest.get("dev_set_hash") or ""),
            "dev_set_manifest": dev_manifest,
            "dev_set_manifest_hash": sha256_json(dev_manifest) if dev_manifest else "",
            "dev_icps_hash": dev_icps_hash,
            "dev_set_size": len(self.load_dev_icp_items() or ()),
            "provenance": dict(provenance or {}),
            "recorded_at": str(recorded_at or ""),
        }
        return {**payload, "manifest_hash": sha256_json(payload)}

    def write_manifest(self, manifest: Mapping[str, Any]) -> None:
        self._write_text(
            MANIFEST_NAME,
            json.dumps(dict(manifest), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )

    def load_manifest(self) -> dict[str, Any] | None:
        raw = self._read_text(MANIFEST_NAME)
        if raw is None:
            return None
        decoded = json.loads(raw)
        if not isinstance(decoded, Mapping):
            raise DevSnapshotStoreError("snapshot manifest must be a JSON object")
        return dict(decoded)

    def verify_manifest(
        self,
        manifest: Mapping[str, Any] | None = None,
        *,
        expected_icp_set_hash: str = "",
    ) -> dict[str, Any]:
        """Verify manifest self-hash and stored-content hash; never raises."""
        errors: list[str] = []
        doc = dict(manifest) if manifest is not None else self.load_manifest()
        if doc is None:
            return {"passed": False, "errors": ["manifest_missing"], "manifest_hash": ""}
        payload = {key: value for key, value in doc.items() if key != "manifest_hash"}
        if str(doc.get("manifest_hash") or "") != sha256_json(payload):
            errors.append("manifest_hash_mismatch")
        if str(doc.get("manifest_type") or "") != SNAPSHOT_MANIFEST_TYPE:
            errors.append("manifest_type_mismatch")
        try:
            stored_content_hash = self.content_hash()
            request_keys = self.request_keys()
        except Exception as exc:
            stored_content_hash = ""
            request_keys = []
            errors.append(f"snapshot_record_invalid:{type(exc).__name__}")
        if str(doc.get("content_hash") or "") != stored_content_hash:
            errors.append("content_hash_mismatch")
        if int(doc.get("snapshot_count") or 0) != self.snapshot_count():
            errors.append("snapshot_count_mismatch")
        if expected_icp_set_hash and str(doc.get("icp_set_hash") or "") != expected_icp_set_hash:
            errors.append("icp_set_hash_mismatch")
        if list(doc.get("request_keys") or ()) != request_keys:
            errors.append("request_keys_mismatch")
        dev_manifest = doc.get("dev_set_manifest")
        if isinstance(dev_manifest, Mapping):
            if dev_manifest and str(doc.get("dev_set_manifest_hash") or "") != sha256_json(dict(dev_manifest)):
                errors.append("dev_set_manifest_hash_mismatch")
        elif doc.get("dev_set_manifest_hash"):
            errors.append("dev_set_manifest_invalid")
        items = self.load_dev_icp_items()
        if not items:
            errors.append("dev_icps_missing")
        else:
            if str(doc.get("dev_icps_hash") or "") != self.dev_icp_items_hash(items):
                errors.append("dev_icps_hash_mismatch")
            if int(doc.get("dev_set_size") or 0) != len(items):
                errors.append("dev_set_size_mismatch")
        return {
            "passed": not errors,
            "errors": errors,
            "manifest_hash": str(doc.get("manifest_hash") or ""),
            "content_hash": stored_content_hash,
            "icp_set_hash": str(doc.get("icp_set_hash") or ""),
            "snapshot_count": self.snapshot_count(),
        }

    def build_ready_document(self, manifest: Mapping[str, Any]) -> dict[str, Any]:
        """Build the unsigned publication marker written only after validation."""
        items = self.load_dev_icp_items() or []
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "ready_type": SNAPSHOT_READY_TYPE,
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "content_hash": str(manifest.get("content_hash") or ""),
            "icp_set_hash": str(manifest.get("icp_set_hash") or ""),
            "dev_icps_hash": self.dev_icp_items_hash(items),
            "dev_set_size": len(items),
            "recorded_at": str(manifest.get("recorded_at") or ""),
            "replay_validation": "passed",
        }
        return {**payload, "ready_hash": sha256_json(payload)}

    def write_ready_document(self, ready: Mapping[str, Any]) -> None:
        self._write_text(
            READY_NAME,
            json.dumps(dict(ready), sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )

    def load_ready_document(self) -> dict[str, Any] | None:
        raw = self._read_text(READY_NAME)
        if raw is None:
            return None
        decoded = json.loads(raw)
        if not isinstance(decoded, Mapping):
            raise DevSnapshotStoreError("snapshot READY document must be a JSON object")
        return dict(decoded)

    def verify_ready_document(
        self,
        ready: Mapping[str, Any] | None = None,
        *,
        expected_dev_icp_count: int,
        require_signature: bool | None = None,
    ) -> dict[str, Any]:
        """Verify publication completeness, configured cohort binding, and signature."""
        if (
            isinstance(expected_dev_icp_count, bool)
            or not isinstance(expected_dev_icp_count, int)
            or expected_dev_icp_count < 1
        ):
            raise DevSnapshotStoreError(
                "expected development ICP count must be a positive integer"
            )
        errors: list[str] = []
        doc = dict(ready) if ready is not None else self.load_ready_document()
        if doc is None:
            return {"passed": False, "errors": ["ready_missing"], "ready_hash": ""}
        signature_fields = {"signature_b64", "kms_key_id", "signing_algorithm"}
        payload = {
            key: value
            for key, value in doc.items()
            if key not in signature_fields and key != "ready_hash"
        }
        ready_hash = sha256_json(payload)
        if str(doc.get("ready_hash") or "") != ready_hash:
            errors.append("ready_hash_mismatch")
        if str(doc.get("ready_type") or "") != SNAPSHOT_READY_TYPE:
            errors.append("ready_type_mismatch")
        manifest = self.load_manifest()
        verification = self.verify_manifest(manifest)
        errors.extend(str(item) for item in verification.get("errors") or ())
        if not isinstance(manifest, Mapping):
            errors.append("manifest_missing")
        else:
            for key in ("manifest_hash", "content_hash", "icp_set_hash", "dev_icps_hash"):
                if str(doc.get(key) or "") != str(manifest.get(key) or ""):
                    errors.append(f"ready_{key}_mismatch")
        items = self.load_dev_icp_items() or []
        if len(items) != expected_dev_icp_count:
            errors.append("dev_set_size_does_not_match_config")
        if int(doc.get("dev_set_size") or 0) != len(items):
            errors.append("ready_dev_set_size_mismatch")
        provenance = manifest.get("provenance") if isinstance(manifest, Mapping) else {}
        provenance = provenance if isinstance(provenance, Mapping) else {}
        for key in (
            "champion_image_digest",
            "source_commit",
            "model_config_hash",
            "provider_model_ids",
            "replay_output_hashes",
        ):
            value = provenance.get(key)
            if value in (None, "", [], {}):
                errors.append(f"snapshot_provenance_missing:{key}")
        require_sig = self._is_s3 if require_signature is None else bool(require_signature)
        signature = str(doc.get("signature_b64") or "")
        key_id = str(doc.get("kms_key_id") or "")
        algorithm = str(doc.get("signing_algorithm") or "")
        if require_sig and not (signature and key_id and algorithm):
            errors.append("ready_signature_missing")
        elif signature and key_id and algorithm:
            try:
                response = _s3_client_for("kms").verify(
                    KeyId=key_id,
                    Message=ready_hash.encode("utf-8"),
                    MessageType="RAW",
                    Signature=base64.b64decode(signature, validate=True),
                    SigningAlgorithm=algorithm,
                )
                if not bool(response.get("SignatureValid")):
                    errors.append("ready_signature_invalid")
            except Exception as exc:
                errors.append(f"ready_signature_error:{type(exc).__name__}")
        return {
            "passed": not errors,
            "errors": list(dict.fromkeys(errors)),
            "ready_hash": str(doc.get("ready_hash") or ""),
            "manifest_hash": str(doc.get("manifest_hash") or ""),
            "dev_set_size": len(items),
            "expected_dev_set_size": expected_dev_icp_count,
            "recorded_at": str(doc.get("recorded_at") or ""),
        }

    def write_dev_icp_items(self, items: Sequence[Mapping[str, Any]]) -> None:
        """Persist the full dev ICP payloads next to the manifest.

        The snapshot-set manifest records only ``{icp_ref, icp_hash}`` per
        selected item; the replay runner needs the actual ICP payloads to
        drive a candidate container, so the recording CLI stores them here.
        Callers verify the items against the manifest's ``icp_set_hash``
        (``dev_eval.compute_dev_set_hash``) at load time.
        """
        rows = [dict(item) for item in items]
        if not rows:
            raise DevSnapshotStoreError("dev ICP items are required")
        if _contains_secret_material(rows):
            raise DevSnapshotStoreError("dev ICP items contain raw secret material")
        self._write_text(
            DEV_ICPS_NAME,
            json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        )

    def load_dev_icp_items(self) -> list[dict[str, Any]] | None:
        """Load the persisted dev ICP payloads, or None when absent."""
        raw = self._read_text(DEV_ICPS_NAME)
        if raw is None:
            return None
        decoded = json.loads(raw)
        if not isinstance(decoded, list) or not all(
            isinstance(item, Mapping) for item in decoded
        ):
            raise DevSnapshotStoreError("dev ICP items must be a JSON array of objects")
        return [dict(item) for item in decoded]

    def dev_icp_items_hash(
        self, items: Sequence[Mapping[str, Any]] | None = None
    ) -> str:
        rows = [dict(item) for item in (items if items is not None else self.load_dev_icp_items() or ())]
        return sha256_json(rows) if rows else ""

    # -- storage backends ----------------------------------------------------

    def _list_snapshot_names(self) -> list[str]:
        if self._is_s3:
            bucket, prefix = _parse_s3_root(self.root_uri)
            client = _s3_client()
            names: list[str] = []
            paginator = client.get_paginator("list_objects_v2")
            full_prefix = f"{prefix}{SNAPSHOT_SUBDIR}/"
            for page in paginator.paginate(Bucket=bucket, Prefix=full_prefix):
                for item in page.get("Contents", []) or []:
                    key = str(item.get("Key") or "")
                    name = key.rsplit("/", 1)[-1]
                    if name.endswith(".json"):
                        names.append(name)
            return sorted(names)
        directory = self._root_path / SNAPSHOT_SUBDIR
        if not directory.exists():
            return []
        return sorted(path.name for path in directory.glob("*.json"))

    def _read_text(self, relative: str) -> str | None:
        if self._is_s3:
            bucket, prefix = _parse_s3_root(self.root_uri)
            client = _s3_client()
            try:
                response = client.get_object(Bucket=bucket, Key=f"{prefix}{relative}")
            except Exception as exc:  # boto3 raises client-specific NoSuchKey
                if type(exc).__name__ in ("NoSuchKey", "NoSuchBucket") or "NoSuchKey" in str(exc):
                    return None
                raise
            return response["Body"].read().decode("utf-8")
        path = self._root_path / relative
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def _write_text(self, relative: str, content: str) -> None:
        if self._is_s3:
            bucket, prefix = _parse_s3_root(self.root_uri)
            _s3_client().put_object(
                Bucket=bucket,
                Key=f"{prefix}{relative}",
                Body=content.encode("utf-8"),
                ContentType="application/json",
            )
            return
        path = self._root_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class _FakeUrllibResponse:
    """Minimal urlopen-compatible response over a replayed snapshot doc."""

    def __init__(self, url: str, doc: Mapping[str, Any]):
        self._body = str(doc.get("body_text") or "").encode("utf-8")
        self._offset = 0
        self.status = int(doc.get("status") or 0)
        self.code = self.status
        self.url = url
        self.headers = dict(doc.get("headers") or {})

    def read(self, amt: int | None = None) -> bytes:
        if amt is None:
            data = self._body[self._offset:]
            self._offset = len(self._body)
            return data
        data = self._body[self._offset : self._offset + max(0, int(amt))]
        self._offset += len(data)
        return data

    def getcode(self) -> int:
        return self.status

    def getheader(self, name: str, default: Any = None) -> Any:
        for key, value in self.headers.items():
            if str(key).lower() == str(name).lower():
                return value
        return default

    def close(self) -> None:  # pragma: no cover - trivial
        return None

    def __enter__(self) -> "_FakeUrllibResponse":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()


class _FakeAiohttpContent:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        start = self._offset
        self._offset = min(len(self._body), self._offset + max(0, int(size)))
        return self._body[start : self._offset]


class _FakeAiohttpResponse:
    """Minimal aiohttp response over one replayed provider snapshot."""

    def __init__(self, url: str, doc: Mapping[str, Any]) -> None:
        self.status = int(doc.get("status") or 0)
        self.headers = dict(doc.get("headers") or {})
        self.url = url
        self.reason = "replayed provider response"
        self.history = ()
        self.request_info = SimpleNamespace(real_url=url)
        self._body = str(doc.get("body_text") or "").encode("utf-8")
        self.content = _FakeAiohttpContent(self._body)

    async def __aenter__(self) -> "_FakeAiohttpResponse":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self.release()
        await self.wait_for_close()

    async def read(self) -> bytes:
        return self._body

    async def text(self, encoding: str | None = None, errors: str = "strict") -> str:
        return self._body.decode(encoding or "utf-8", errors=errors)

    async def json(self, *args: Any, **kwargs: Any) -> Any:
        return json.loads(self._body.decode(kwargs.get("encoding") or "utf-8"))

    def raise_for_status(self) -> None:
        if self.status < 400:
            return
        import aiohttp

        raise aiohttp.ClientResponseError(
            request_info=self.request_info,
            history=(),
            status=self.status,
            message=self.reason,
            headers=self.headers,
        )

    def release(self) -> None:
        return None

    def close(self) -> None:
        return None

    async def wait_for_close(self) -> None:
        return None


def container_replay_env(
    snapshot_dir: str,
    *,
    miss_policy: str = MISS_POLICY_STRICT,
) -> dict[str, str]:
    """Env vars a candidate container needs for bootstrap-driven replay.

    The future engine wiring mounts the materialized snapshot directory into
    the container and passes this mapping as ``extra_env`` alongside
    :func:`dev_replay_bootstrap` prepended to the adapter bootstrap.
    """
    if miss_policy not in MISS_POLICIES:
        raise DevSnapshotStoreError(f"unknown snapshot miss policy: {miss_policy}")
    return {
        SNAPSHOT_DIR_ENV: str(snapshot_dir),
        SNAPSHOT_MISS_POLICY_ENV: miss_policy,
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_secret_material(key) or _contains_secret_material(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in SECRET_MARKERS)
    return False


def _s3_client() -> Any:
    return _s3_client_for("s3")


def _s3_client_for(service_name: str) -> Any:
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - depends on env
        raise DevSnapshotStoreError("boto3 is required for S3 snapshot stores") from exc
    return boto3.client(service_name)


def _parse_s3_root(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise DevSnapshotStoreError(f"expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise DevSnapshotStoreError(f"invalid s3 URI: {uri}")
    prefix = prefix.strip("/")
    return bucket, f"{prefix}/" if prefix else ""


def _parse_s3_object_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise DevSnapshotStoreError(f"expected s3:// object URI, got {uri}")
    without_scheme = uri[5:]
    bucket, separator, key = without_scheme.partition("/")
    if not bucket or not separator or not key.strip("/"):
        raise DevSnapshotStoreError(f"invalid S3 object URI: {uri}")
    return bucket, key.strip("/")


def _validate_pointer_target(pointer_uri: str, target_uri: str) -> None:
    if not target_uri:
        raise DevSnapshotStoreError("pointer_target_missing")
    if pointer_uri.startswith("s3://"):
        pointer_bucket, pointer_key = _parse_s3_object_uri(pointer_uri)
        target_bucket, target_prefix = _parse_s3_root(target_uri)
        pointer_parent = pointer_key.rsplit("/", 1)[0] + "/" if "/" in pointer_key else ""
        if target_bucket != pointer_bucket or not target_prefix.startswith(pointer_parent):
            raise DevSnapshotStoreError("pointer_target_outside_base")
        if target_prefix.rstrip("/") == pointer_key:
            raise DevSnapshotStoreError("pointer_target_recursive")
        return
    pointer_path = Path(pointer_uri).expanduser().resolve()
    target_path = Path(target_uri).expanduser().resolve()
    try:
        target_path.relative_to(pointer_path.parent)
    except ValueError as exc:
        raise DevSnapshotStoreError("pointer_target_outside_base") from exc
    if target_path == pointer_path:
        raise DevSnapshotStoreError("pointer_target_recursive")


# ---------------------------------------------------------------------------
# In-container bootstraps (analogous to _PROVIDER_DIAGNOSTICS_BOOTSTRAP).
# The key-derivation helpers below MUST mirror build_snapshot_request exactly;
# tests/test_dev_eval.py asserts host/bootstrap key parity via a subprocess
# round-trip. The bootstraps are inert unless RESEARCH_LAB_DEV_SNAPSHOT_DIR is
# set, so accidentally shipping them in a live path cannot intercept traffic.
# ---------------------------------------------------------------------------

_BOOTSTRAP_KEY_HELPERS = r"""
import hashlib
import json
import os
from urllib.parse import parse_qsl, urlsplit

_RL_DEV_SNAPSHOT_DIR = os.environ.get("RESEARCH_LAB_DEV_SNAPSHOT_DIR", "").strip()
_RL_DEV_MISS_POLICY = (os.environ.get("RESEARCH_LAB_DEV_SNAPSHOT_MISS_POLICY", "").strip().lower() or "strict")
_RL_DEV_RECORD_ICP_REF = os.environ.get("RESEARCH_LAB_DEV_RECORD_ICP_REF", "").strip()
_RL_DEV_AUTH_PARAMS = ("api_key", "apikey", "x-api-key", "authorization", "token", "access_token", "bearer")
_RL_DEV_EMPTY_BODIES = {"exa": '{"results": []}', "scrapingdog": "{}", "openrouter": "{}"}
# Keep in sync with research_lab.eval.private_runtime.SECRET_MARKERS.
_RL_DEV_SECRET_MARKERS = ("sk-or-", "sb_secret_", "aws_secret_access_key", "openrouter_api_key", "scrapingdog_api_key", "exa_api_key", "raw_secret", "service_role")
_RL_DEV_RECORD_FAILURES_PATH = os.path.join(_RL_DEV_SNAPSHOT_DIR, "record_failures.jsonl")


def _rl_dev_canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _rl_dev_sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rl_dev_provider_for_host(host):
    lowered = str(host or "").strip().lower()
    if "exa.ai" in lowered:
        return "exa"
    if "scrapingdog" in lowered:
        return "scrapingdog"
    if "openrouter" in lowered:
        return "openrouter"
    return lowered or "unknown"


def _rl_dev_strip_auth(params):
    return {
        name: value
        for name, value in sorted(params.items())
        if str(name).strip().lower() not in _RL_DEV_AUTH_PARAMS
    }


def _rl_dev_normalized_body(body):
    if body is None:
        return None
    if isinstance(body, (bytes, bytearray)):
        body = bytes(body).decode("utf-8", "replace")
    if isinstance(body, str):
        text = body.strip()
        if not text:
            return None
        try:
            body = json.loads(text)
        except ValueError:
            return text
    if isinstance(body, dict):
        return _rl_dev_strip_auth({str(k): v for k, v in body.items()})
    return body


def _rl_dev_request_identity(method, url, body, params=None):
    split = urlsplit(str(url or ""))
    host = split.netloc.lower().rsplit("@", 1)[-1]
    for default_port in (":80", ":443"):
        if host.endswith(default_port):
            host = host.rsplit(":", 1)[0]
    path = split.path.rstrip("/")
    endpoint = host + path if path else host
    query = {}
    for name, value in parse_qsl(split.query, keep_blank_values=True):
        query.setdefault(str(name), []).append(str(value))
    for name, value in dict(params or {}).items():
        query.setdefault(str(name), []).append(str(value))
    significant = {
        "query": _rl_dev_strip_auth({name: sorted(values) for name, values in query.items()}),
        "body": _rl_dev_normalized_body(body),
    }
    params_hash = "sha256:" + _rl_dev_sha256_text(_rl_dev_canonical_json(significant))
    provider = _rl_dev_provider_for_host(host)
    method_norm = str(method or "GET").strip().upper() or "GET"
    request_key = provider + "|" + method_norm + "|" + endpoint + "|" + params_hash
    return provider, request_key, _rl_dev_sha256_text(request_key)


def _rl_dev_snapshot_path(storage_name):
    return os.path.join(_RL_DEV_SNAPSHOT_DIR, "snapshots", storage_name + ".json")


class _RlDevSnapshotMiss(RuntimeError):
    pass


def _rl_dev_lookup(method, url, body, params=None):
    provider, request_key, storage_name = _rl_dev_request_identity(method, url, body, params)
    path = _rl_dev_snapshot_path(storage_name)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            record = json.load(handle)
        return dict(record.get("response") or {})
    if _RL_DEV_MISS_POLICY == "empty":
        return {
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body_text": _RL_DEV_EMPTY_BODIES.get(provider, "{}"),
            "synthesized": "research_lab_dev_snapshot_synthesized_empty",
        }
    raise _RlDevSnapshotMiss("RESEARCH_LAB_DEV_SNAPSHOT_MISS:" + request_key)


class _RlDevFakeResponse(object):
    def __init__(self, url, doc):
        self._body = str(doc.get("body_text") or "").encode("utf-8")
        self._offset = 0
        self.status = int(doc.get("status") or 0)
        self.code = self.status
        self.url = url
        self.headers = dict(doc.get("headers") or {})

    def read(self, amt=None):
        if amt is None:
            data = self._body[self._offset:]
            self._offset = len(self._body)
            return data
        data = self._body[self._offset:self._offset + max(0, int(amt))]
        self._offset += len(data)
        return data

    def getcode(self):
        return self.status

    def getheader(self, name, default=None):
        for key, value in self.headers.items():
            if str(key).lower() == str(name).lower():
                return value
        return default

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


class _RlDevAiohttpContent(object):
    def __init__(self, body):
        self._body = body
        self._offset = 0

    async def read(self, size=-1):
        if size is None or size < 0:
            size = len(self._body) - self._offset
        start = self._offset
        self._offset = min(len(self._body), self._offset + max(0, int(size)))
        return self._body[start:self._offset]


class _RlDevRequestInfo(object):
    def __init__(self, url):
        self.real_url = url


class _RlDevAiohttpResponse(object):
    def __init__(self, url, doc):
        self.status = int(doc.get("status") or 0)
        self.headers = dict(doc.get("headers") or {})
        self.url = url
        self.reason = "replayed provider response"
        self.history = ()
        self.request_info = _RlDevRequestInfo(url)
        self._body = str(doc.get("body_text") or "").encode("utf-8")
        self.content = _RlDevAiohttpContent(self._body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        self.release()
        await self.wait_for_close()

    async def read(self):
        return self._body

    async def text(self, encoding=None, errors="strict"):
        return self._body.decode(encoding or "utf-8", errors=errors)

    async def json(self, *args, **kwargs):
        return json.loads(self._body.decode(kwargs.get("encoding") or "utf-8"))

    def raise_for_status(self):
        if self.status < 400:
            return
        import aiohttp
        raise aiohttp.ClientResponseError(
            request_info=self.request_info,
            history=(),
            status=self.status,
            message=self.reason,
            headers=self.headers,
        )

    def release(self):
        return None

    def close(self):
        return None

    async def wait_for_close(self):
        return None
"""


_REPLAY_BOOTSTRAP_BODY = r"""
def _rl_dev_install_replay():
    if not _RL_DEV_SNAPSHOT_DIR:
        return

    import urllib.request as _rl_urllib_request

    def _rl_dev_replay_urlopen(req, *args, **kwargs):
        url = getattr(req, "full_url", req if isinstance(req, str) else "")
        method = getattr(req, "get_method", lambda: "GET")()
        data = getattr(req, "data", None)
        return _RlDevFakeResponse(str(url), _rl_dev_lookup(method, str(url), data))

    _rl_urllib_request.urlopen = _rl_dev_replay_urlopen

    try:
        import requests as _rl_requests

        def _rl_dev_replay_requests_send(session, prepared, *args, **kwargs):
            doc = _rl_dev_lookup(str(prepared.method or "GET"), str(prepared.url or ""), prepared.body)
            response = _rl_requests.models.Response()
            response.status_code = int(doc.get("status") or 0)
            response._content = str(doc.get("body_text") or "").encode("utf-8")
            response.encoding = "utf-8"
            response.headers.update(dict(doc.get("headers") or {}))
            response.url = str(prepared.url or "")
            response.request = prepared
            return response

        _rl_requests.Session.send = _rl_dev_replay_requests_send
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as exc:
        raise RuntimeError(
            "requests replay hook installation failed: " + type(exc).__name__
        ) from exc

    try:
        import httpx as _rl_httpx

        def _rl_dev_replay_httpx_send(client, request, *args, **kwargs):
            doc = _rl_dev_lookup(
                str(request.method or "GET"),
                str(request.url or ""),
                bytes(getattr(request, "content", b"") or b""),
            )
            return _rl_httpx.Response(
                status_code=int(doc.get("status") or 0),
                headers=dict(doc.get("headers") or {}),
                content=str(doc.get("body_text") or "").encode("utf-8"),
                request=request,
            )

        _rl_httpx.Client.send = _rl_dev_replay_httpx_send

        async def _rl_dev_replay_httpx_async_send(client, request, *args, **kwargs):
            return _rl_dev_replay_httpx_send(client, request, *args, **kwargs)

        _rl_httpx.AsyncClient.send = _rl_dev_replay_httpx_async_send
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as exc:
        raise RuntimeError(
            "httpx replay hook installation failed: " + type(exc).__name__
        ) from exc

    try:
        import aiohttp as _rl_aiohttp

        async def _rl_dev_replay_aiohttp_request(session, method, str_or_url, *args, **kwargs):
            body = kwargs.get("json")
            if body is None:
                body = kwargs.get("data")
            doc = _rl_dev_lookup(
                str(method or "GET"),
                str(str_or_url),
                body,
                kwargs.get("params"),
            )
            response = _RlDevAiohttpResponse(str(str_or_url), doc)
            if kwargs.get("raise_for_status") is True:
                response.raise_for_status()
            return response

        _rl_aiohttp.ClientSession._request = _rl_dev_replay_aiohttp_request
    except (ImportError, ModuleNotFoundError):
        pass
    except Exception as exc:
        raise RuntimeError(
            "aiohttp replay guard installation failed: " + type(exc).__name__
        ) from exc


_rl_dev_install_replay()
"""


_RECORD_BOOTSTRAP_BODY = r"""
def _rl_dev_record_failure(reason, request_key=""):
    try:
        os.makedirs(_RL_DEV_SNAPSHOT_DIR, exist_ok=True)
        row = {
            "reason": str(reason or "record_failure")[:240],
            "request_key": str(request_key or "")[:1000],
            "icp_ref": _RL_DEV_RECORD_ICP_REF[:500],
        }
        with open(_RL_DEV_RECORD_FAILURES_PATH, "a", encoding="utf-8") as handle:
            handle.write(_rl_dev_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as failure_exc:
        import sys
        sys.stderr.write(
            "research_lab_dev_snapshot_failure_telemetry_error="
            + type(failure_exc).__name__
            + "\n"
        )


def _rl_dev_record(method, url, body, status, headers, body_text, params=None):
    request_key = ""
    try:
        provider, request_key, storage_name = _rl_dev_request_identity(
            method, url, body, params
        )
        content_type = ""
        for key, value in dict(headers or {}).items():
            if str(key).lower() == "content-type":
                content_type = str(value)
                break
        record = {
            "schema_version": "1.0",
            "record_type": "research_lab_dev_provider_snapshot",
            "request_key": request_key,
            "provider": provider,
            "method": str(method or "GET").strip().upper() or "GET",
            "endpoint": request_key.split("|")[2],
            "params_hash": request_key.split("|")[3],
            "response": {
                "status": int(status or 0),
                "headers": {"content-type": content_type or "application/json"},
                "body_text": str(body_text or ""),
            },
        }
        payload_text = _rl_dev_canonical_json(record).lower()
        if any(marker in payload_text for marker in _RL_DEV_SECRET_MARKERS):
            _rl_dev_record_failure("secret_material_rejected", request_key)
            return False
        path = _rl_dev_snapshot_path(storage_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + "." + str(os.getpid()) + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            handle.write(_rl_dev_canonical_json(record))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    except Exception as exc:
        _rl_dev_record_failure("record_write_error:" + type(exc).__name__, request_key)
        return False


def _rl_dev_install_record():
    if not _RL_DEV_SNAPSHOT_DIR:
        return

    import urllib.request as _rl_urllib_request

    _rl_dev_original_urlopen = _rl_urllib_request.urlopen

    def _rl_dev_record_urlopen(req, *args, **kwargs):
        url = str(getattr(req, "full_url", req if isinstance(req, str) else ""))
        method = getattr(req, "get_method", lambda: "GET")()
        data = getattr(req, "data", None)
        response = _rl_dev_original_urlopen(req, *args, **kwargs)
        body = response.read()
        status = getattr(response, "status", None) or getattr(response, "code", 0)
        headers = dict(getattr(response, "headers", {}) or {})
        body_text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
        _rl_dev_record(method, url, data, status, headers, body_text)
        return _RlDevFakeResponse(url, {
            "status": int(status or 0),
            "headers": headers,
            "body_text": body_text,
        })

    _rl_urllib_request.urlopen = _rl_dev_record_urlopen

    try:
        import requests as _rl_requests

        _rl_dev_original_requests_send = _rl_requests.Session.send

        def _rl_dev_record_requests_send(session, prepared, *args, **kwargs):
            response = _rl_dev_original_requests_send(session, prepared, *args, **kwargs)
            try:
                _rl_dev_record(
                    str(prepared.method or "GET"),
                    str(prepared.url or ""),
                    prepared.body,
                    response.status_code,
                    dict(response.headers or {}),
                    response.text,
                )
            except Exception as exc:
                _rl_dev_record_failure(
                    "requests_record_hook_error:" + type(exc).__name__
                )
            return response

        _rl_requests.Session.send = _rl_dev_record_requests_send
    except Exception as exc:
        if type(exc).__name__ not in ("ImportError", "ModuleNotFoundError"):
            _rl_dev_record_failure(
                "requests_record_hook_install_error:" + type(exc).__name__
            )

    try:
        import httpx as _rl_httpx

        _rl_dev_original_httpx_send = _rl_httpx.Client.send

        def _rl_dev_record_httpx_send(client, request, *args, **kwargs):
            response = _rl_dev_original_httpx_send(client, request, *args, **kwargs)
            try:
                _rl_dev_record(
                    str(request.method or "GET"),
                    str(request.url or ""),
                    bytes(getattr(request, "content", b"") or b""),
                    response.status_code,
                    dict(response.headers or {}),
                    response.text,
                )
            except Exception as exc:
                _rl_dev_record_failure(
                    "httpx_record_hook_error:" + type(exc).__name__
                )
            return response

        _rl_httpx.Client.send = _rl_dev_record_httpx_send

        _rl_dev_original_httpx_async_send = _rl_httpx.AsyncClient.send

        async def _rl_dev_record_httpx_async_send(client, request, *args, **kwargs):
            response = await _rl_dev_original_httpx_async_send(client, request, *args, **kwargs)
            _rl_dev_record(
                str(request.method or "GET"),
                str(request.url or ""),
                bytes(getattr(request, "content", b"") or b""),
                response.status_code,
                dict(response.headers or {}),
                response.text,
            )
            return response

        _rl_httpx.AsyncClient.send = _rl_dev_record_httpx_async_send
    except Exception as exc:
        if type(exc).__name__ not in ("ImportError", "ModuleNotFoundError"):
            _rl_dev_record_failure("httpx_record_hook_error:" + type(exc).__name__)

    try:
        import aiohttp as _rl_aiohttp

        _rl_dev_original_aiohttp_request = _rl_aiohttp.ClientSession._request

        async def _rl_dev_record_aiohttp_request(session, method, str_or_url, *args, **kwargs):
            response = await _rl_dev_original_aiohttp_request(
                session, method, str_or_url, *args, **kwargs
            )
            body = await response.read()
            charset = getattr(response, "charset", None) or "utf-8"
            _rl_dev_record(
                str(method or "GET"),
                str(str_or_url or ""),
                kwargs.get("json") if kwargs.get("json") is not None else kwargs.get("data"),
                response.status,
                dict(response.headers or {}),
                body.decode(charset, "replace"),
                kwargs.get("params"),
            )
            return response

        _rl_aiohttp.ClientSession._request = _rl_dev_record_aiohttp_request
    except Exception as exc:
        if type(exc).__name__ not in ("ImportError", "ModuleNotFoundError"):
            _rl_dev_record_failure("aiohttp_record_hook_error:" + type(exc).__name__)


_rl_dev_install_record()
"""


def dev_replay_bootstrap() -> str:
    """Self-contained replay preamble for candidate containers/subprocesses.

    Prepend to the adapter bootstrap; activates only when
    ``RESEARCH_LAB_DEV_SNAPSHOT_DIR`` is set (see :func:`container_replay_env`).
    Serves urllib/requests/httpx traffic from the mounted snapshot directory
    and never opens a live connection, including aiohttp.
    """
    return _BOOTSTRAP_KEY_HELPERS + _REPLAY_BOOTSTRAP_BODY


def dev_record_bootstrap() -> str:
    """Self-contained record preamble for the one-time champion recording run.

    Prepend to the adapter bootstrap with ``RESEARCH_LAB_DEV_SNAPSHOT_DIR``
    pointing at a writable directory: live urllib/requests/httpx responses
    pass through unchanged while a snapshot record is persisted per request
    key (recording is best-effort; a persistence failure never breaks the
    live call), including aiohttp traffic.
    """
    return _BOOTSTRAP_KEY_HELPERS + _RECORD_BOOTSTRAP_BODY

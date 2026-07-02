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

Storage is a local directory or an S3 prefix (``RESEARCH_LAB_DEV_SNAPSHOT_URI``)
with the layout::

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

This module is dormant: nothing in the live pipeline imports it yet. It is
deterministic by construction — no wall clocks, no unseeded randomness; the
optional manifest ``recorded_at`` is caller-supplied by the recording CLI.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping
from urllib.parse import parse_qsl, urlsplit

from research_lab.canonical import sha256_json, sha256_text

from .private_runtime import SECRET_MARKERS

SNAPSHOT_SCHEMA_VERSION = "1.0"
SNAPSHOT_RECORD_TYPE = "research_lab_dev_provider_snapshot"
SNAPSHOT_MANIFEST_TYPE = "research_lab_dev_snapshot_set"
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
        ``urllib.request.urlopen``, ``requests.Session.send`` and
        ``httpx.Client.send`` (the latter two best-effort, only when the
        library is importable) — so an in-process candidate runner needs no
        snapshot awareness. ``aiohttp`` replay is not supported: its session
        request is patched to raise loudly rather than leak live traffic.
        Patches are process-global while active; run ICPs serially.
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
            self._install_aiohttp_guard(restore)
            yield self
        finally:
            for undo in reversed(restore):
                undo()

    def _install_requests_replay(self, restore: list[Callable[[], None]]) -> None:
        try:
            import requests
        except Exception:
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
        except Exception:
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

    def _install_aiohttp_guard(self, restore: list[Callable[[], None]]) -> None:
        try:
            import aiohttp
        except Exception:
            return
        original_request = aiohttp.ClientSession._request  # noqa: SLF001

        async def _replay_unsupported(session, method, str_or_url, *args, **kwargs):  # noqa: ANN001
            raise DevSnapshotStoreError(
                "aiohttp replay is not supported in-process; "
                f"refusing live request to {str_or_url}"
            )

        aiohttp.ClientSession._request = _replay_unsupported  # noqa: SLF001
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
    ) -> dict[str, Any]:
        """Build the snapshot-set manifest binding traffic to the dev ICP set.

        ``recorded_at`` is caller-supplied (the recording CLI passes its own
        clock) so this library stays wall-clock-free and deterministic.
        """
        dev_manifest = dict(dev_set_manifest or {})
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "manifest_type": SNAPSHOT_MANIFEST_TYPE,
            "snapshot_count": self.snapshot_count(),
            "request_keys": self.request_keys(),
            "content_hash": self.content_hash(),
            "icp_set_hash": str(icp_set_hash or dev_manifest.get("dev_set_hash") or ""),
            "dev_set_manifest": dev_manifest,
            "dev_set_manifest_hash": sha256_json(dev_manifest) if dev_manifest else "",
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
        stored_content_hash = self.content_hash()
        if str(doc.get("content_hash") or "") != stored_content_hash:
            errors.append("content_hash_mismatch")
        if int(doc.get("snapshot_count") or 0) != self.snapshot_count():
            errors.append("snapshot_count_mismatch")
        if expected_icp_set_hash and str(doc.get("icp_set_hash") or "") != expected_icp_set_hash:
            errors.append("icp_set_hash_mismatch")
        return {
            "passed": not errors,
            "errors": errors,
            "manifest_hash": str(doc.get("manifest_hash") or ""),
            "content_hash": stored_content_hash,
            "icp_set_hash": str(doc.get("icp_set_hash") or ""),
            "snapshot_count": self.snapshot_count(),
        }

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
        path.write_text(content, encoding="utf-8")


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
    try:
        import boto3
    except Exception as exc:  # pragma: no cover - depends on env
        raise DevSnapshotStoreError("boto3 is required for S3 snapshot stores") from exc
    return boto3.client("s3")


def _parse_s3_root(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise DevSnapshotStoreError(f"expected s3:// URI, got {uri}")
    without_scheme = uri[5:]
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise DevSnapshotStoreError(f"invalid s3 URI: {uri}")
    prefix = prefix.strip("/")
    return bucket, f"{prefix}/" if prefix else ""


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
_RL_DEV_AUTH_PARAMS = ("api_key", "apikey", "x-api-key", "authorization", "token", "access_token", "bearer")
_RL_DEV_EMPTY_BODIES = {"exa": '{"results": []}', "scrapingdog": "{}", "openrouter": "{}"}


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


def _rl_dev_request_identity(method, url, body):
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


def _rl_dev_lookup(method, url, body):
    provider, request_key, storage_name = _rl_dev_request_identity(method, url, body)
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
    raise _RlDevSnapshotMiss("no recorded provider snapshot for request: " + request_key)


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
    except Exception:
        pass

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
    except Exception:
        pass

    try:
        import aiohttp as _rl_aiohttp

        async def _rl_dev_replay_aiohttp_unsupported(session, method, str_or_url, *args, **kwargs):
            raise RuntimeError(
                "aiohttp replay is not supported; refusing live request to " + str(str_or_url)
            )

        _rl_aiohttp.ClientSession._request = _rl_dev_replay_aiohttp_unsupported
    except Exception:
        pass


_rl_dev_install_replay()
"""


_RECORD_BOOTSTRAP_BODY = r"""
def _rl_dev_record(method, url, body, status, headers, body_text):
    try:
        provider, request_key, storage_name = _rl_dev_request_identity(method, url, body)
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
        path = _rl_dev_snapshot_path(storage_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(_rl_dev_canonical_json(record))
    except Exception:
        pass


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
            except Exception:
                pass
            return response

        _rl_requests.Session.send = _rl_dev_record_requests_send
    except Exception:
        pass

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
            except Exception:
                pass
            return response

        _rl_httpx.Client.send = _rl_dev_record_httpx_send
    except Exception:
        pass


_rl_dev_install_record()
"""


def dev_replay_bootstrap() -> str:
    """Self-contained replay preamble for candidate containers/subprocesses.

    Prepend to the adapter bootstrap; activates only when
    ``RESEARCH_LAB_DEV_SNAPSHOT_DIR`` is set (see :func:`container_replay_env`).
    Serves urllib/requests/httpx traffic from the mounted snapshot directory
    and never opens a live connection; aiohttp raises loudly (unsupported).
    """
    return _BOOTSTRAP_KEY_HELPERS + _REPLAY_BOOTSTRAP_BODY


def dev_record_bootstrap() -> str:
    """Self-contained record preamble for the one-time champion recording run.

    Prepend to the adapter bootstrap with ``RESEARCH_LAB_DEV_SNAPSHOT_DIR``
    pointing at a writable directory: live urllib/requests/httpx responses
    pass through unchanged while a snapshot record is persisted per request
    key (recording is best-effort; a persistence failure never breaks the
    live call). aiohttp traffic is not recorded (documented follow-up).
    """
    return _BOOTSTRAP_KEY_HELPERS + _RECORD_BOOTSTRAP_BODY

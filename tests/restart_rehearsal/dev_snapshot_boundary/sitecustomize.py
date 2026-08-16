"""Strict external boundaries for the exact dev-snapshot rehearsal.

This module is loaded only by subprocesses that receive
``REHEARSAL_DEV_SNAPSHOT_BOUNDARY_STATE``.  It leaves the production export,
record, publish, pointer, and readiness code untouched while replacing the
four privileged edges they require: Supabase, provider HTTP, Docker's
container boundary, and S3/KMS.  Unknown calls fail closed.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, urlsplit


STATE_ENV = "REHEARSAL_DEV_SNAPSHOT_BOUNDARY_STATE"
NEGATIVE_PROBE_ENV = "REHEARSAL_DEV_SNAPSHOT_NEGATIVE_PROBE"
_state_path_raw = str(os.getenv(STATE_ENV) or "").strip()


if _state_path_raw:
    _state_path = Path(_state_path_raw).expanduser().resolve()
    _state = json.loads(_state_path.read_text(encoding="utf-8"))
    if (
        not isinstance(_state, Mapping)
        or _state.get("schema_version")
        != "leadpoet.rehearsal_dev_snapshot_boundary.v1"
    ):
        raise RuntimeError("dev-snapshot boundary state is invalid")

    _root = Path(str(_state["root"])).resolve()
    _object_root = _root / "s3"
    _event_path = _root / "events.jsonl"
    _source_root = Path(str(_state["source_root"])).resolve()
    _champion_root = Path(str(_state["champion_root"])).resolve()
    _bucket = str(_state["bucket"])
    _base_prefix = str(_state["base_prefix"]).strip("/") + "/"
    _kms_key_id = str(_state["kms_key_id"])
    _active_artifact = dict(_state["active_artifact"])
    _image_digest = str(_active_artifact["image_digest"])
    _selection_seed = str(_state["selection_seed"])
    _provider_model_ids = [
        str(value) for value in (_state.get("provider_model_ids") or ())
    ]
    _expected_cli_argv_contract_hashes = dict(
        _state.get("expected_cli_argv_contract_hashes") or {}
    )
    _expected_docker_bootstrap_hashes = dict(
        _state.get("expected_docker_bootstrap_hashes") or {}
    )
    _adapter_root = Path(__file__).resolve().parent
    _real_subprocess_run = subprocess.run
    _observed_refresh_work_dir: Path | None = None
    _observed_production_command_phases: list[str] = []

    for _required_path in (_root, _source_root, _champion_root):
        if not _required_path.exists():
            raise RuntimeError("dev-snapshot boundary path is unavailable")
    if (
        not _bucket
        or not _base_prefix
        or not _kms_key_id
        or "@sha256:" not in _image_digest
        or not _selection_seed
        or _provider_model_ids != ["openai/rehearsal-model"]
        or set(_expected_cli_argv_contract_hashes)
        != {"export", "record", "publish_immutable", "publish_pointer"}
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)) is None
            for value in _expected_cli_argv_contract_hashes.values()
        )
        or set(_expected_docker_bootstrap_hashes) != {"record", "replay"}
        or any(
            re.fullmatch(r"sha256:[0-9a-f]{64}", str(value)) is None
            for value in _expected_docker_bootstrap_hashes.values()
        )
    ):
        raise RuntimeError("dev-snapshot boundary identity is incomplete")
    _boundary_contract_path = (
        _source_root / "tests/restart_rehearsal/boundary_contract.json"
    )
    _boundary_contract = json.loads(
        _boundary_contract_path.read_text(encoding="utf-8")
    )
    if (
        not isinstance(_boundary_contract, Mapping)
        or _boundary_contract.get("schema_version")
        != "leadpoet.local_restart_boundary_contract.v1"
        or not isinstance(_boundary_contract.get("boundaries"), Mapping)
    ):
        raise RuntimeError("dev-snapshot boundary contract is invalid")

    def _require_declared_boundary(boundary: str, operation: str) -> None:
        definition = _boundary_contract["boundaries"].get(boundary)
        if (
            not isinstance(definition, Mapping)
            or definition.get("reject_unknown") is not True
            or operation not in tuple(definition.get("allowed_operations") or ())
        ):
            raise RuntimeError("dev-snapshot external boundary is undeclared")

    def _negative_probe_fields(expected: str) -> dict[str, Any]:
        configured = str(os.getenv(NEGATIVE_PROBE_ENV) or "").strip()
        matched = configured == str(expected)
        return {
            "negative_probe": 1 if matched else 0,
            "probe_id": str(expected) if matched else "",
        }

    def _event(kind: str, operation: str, **fields: Any) -> None:
        row = {
            "kind": str(kind),
            "operation": str(operation),
            **{
                str(key): value
                for key, value in fields.items()
                if value is not None
            },
        }
        _event_path.parent.mkdir(parents=True, exist_ok=True)
        with _event_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    def _load_rows(table: str) -> list[dict[str, Any]]:
        tables = _state.get("supabase_tables")
        rows = tables.get(table) if isinstance(tables, Mapping) else None
        if table not in {
            "research_lab_private_model_benchmark_current",
            "research_lab_rolling_icp_windows",
            "qualification_private_icp_sets",
        } or not isinstance(rows, list):
            raise RuntimeError("dev-snapshot Supabase table is not allowlisted")
        if not all(isinstance(row, Mapping) for row in rows):
            raise RuntimeError("dev-snapshot Supabase fixture row is invalid")
        return [copy.deepcopy(dict(row)) for row in rows]

    _SUPABASE_SELECT_CONTRACTS = {
        "research_lab_private_model_benchmark_current": (
            "benchmark_bundle_id,benchmark_bundle_hash,benchmark_date,"
            "private_model_manifest_hash,rolling_window_hash,evaluation_epoch,"
            "benchmark_quality,score_summary_doc,current_benchmark_status,created_at",
            "created_at",
            True,
        ),
        "research_lab_rolling_icp_windows": (
            "rolling_window_hash,window_doc,created_at",
            "created_at",
            True,
        ),
        "qualification_private_icp_sets": (
            "set_id,icps,icp_set_hash,active_from,active_until,is_active",
            "set_id",
            True,
        ),
    }

    class _SupabaseQuery:
        def __init__(self, table: str) -> None:
            self._table = table
            self._columns = "*"
            self._order = ""
            self._desc = False
            self._start = 0
            self._end = 999

        def select(self, columns: str) -> "_SupabaseQuery":
            normalized = str(columns or "").strip()
            if not normalized:
                raise RuntimeError("dev-snapshot Supabase projection is empty")
            self._columns = normalized
            return self

        def order(self, column: str, *, desc: bool = False) -> "_SupabaseQuery":
            normalized = str(column or "").strip()
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,127}", normalized):
                raise RuntimeError("dev-snapshot Supabase order is invalid")
            self._order = normalized
            self._desc = bool(desc)
            return self

        def range(self, start: int, end: int) -> "_SupabaseQuery":
            self._start = int(start)
            self._end = int(end)
            if self._start < 0 or self._end < self._start:
                raise RuntimeError("dev-snapshot Supabase range is invalid")
            return self

        def execute(self) -> Any:
            _require_declared_boundary("supabase_postgrest", "select")
            expected_columns, expected_order, expected_desc = (
                _SUPABASE_SELECT_CONTRACTS[self._table]
            )
            if (
                self._columns != expected_columns
                or self._order != expected_order
                or self._desc is not expected_desc
                or self._start != 0
                or self._end != 999
            ):
                raise RuntimeError("dev-snapshot Supabase select contract differs")
            rows = _load_rows(self._table)
            if self._order:
                rows.sort(
                    key=lambda row: str(row.get(self._order) or ""),
                    reverse=self._desc,
                )
            rows = rows[self._start : self._end + 1]
            if self._columns != "*":
                columns = [
                    value.strip()
                    for value in self._columns.split(",")
                    if value.strip()
                ]
                if not columns or any(
                    re.fullmatch(r"[a-z][a-z0-9_]{0,127}", value) is None
                    for value in columns
                ):
                    raise RuntimeError(
                        "dev-snapshot Supabase projection is invalid"
                    )
                rows = [
                    {column: row.get(column) for column in columns}
                    for row in rows
                ]
            _event(
                "supabase",
                "select",
                table=self._table,
                row_count=len(rows),
                offset=self._start,
                limit=self._end - self._start + 1,
            )
            return SimpleNamespace(data=rows)

    class _SupabaseClient:
        def table(self, table: str) -> _SupabaseQuery:
            return _SupabaseQuery(str(table))

    def _create_client(url: str, key: str, *_args: Any, **_kwargs: Any) -> Any:
        if (
            str(url) != "https://rehearsal.supabase.invalid"
            or str(key) != "rehearsal-service-role"
        ):
            raise RuntimeError("dev-snapshot Supabase identity differs")
        return _SupabaseClient()

    _supabase_module = ModuleType("supabase")
    _supabase_module.__path__ = []  # type: ignore[attr-defined]
    _supabase_module.create_client = _create_client  # type: ignore[attr-defined]
    _supabase_module.Client = _SupabaseClient  # type: ignore[attr-defined]
    _supabase_module.AsyncClient = _SupabaseClient  # type: ignore[attr-defined]
    _supabase_module.create_async_client = _create_client  # type: ignore[attr-defined]
    sys.modules["supabase"] = _supabase_module
    _supabase_lib_module = ModuleType("supabase.lib")
    _supabase_lib_module.__path__ = []  # type: ignore[attr-defined]
    _supabase_options_module = ModuleType("supabase.lib.client_options")

    class _SyncClientOptions:
        def __init__(self, **kwargs: Any) -> None:
            self.httpx_client = kwargs.get("httpx_client")
            self.postgrest_client_timeout = 120.0

    _supabase_options_module.SyncClientOptions = _SyncClientOptions  # type: ignore[attr-defined]
    sys.modules["supabase.lib"] = _supabase_lib_module
    sys.modules["supabase.lib.client_options"] = _supabase_options_module

    def _object_path(bucket: str, key: str) -> Path:
        normalized_bucket = str(bucket or "")
        normalized_key = str(key or "").lstrip("/")
        key_path = PurePosixPath(normalized_key)
        if (
            normalized_bucket != _bucket
            or not normalized_key.startswith(_base_prefix)
            or not normalized_key
            or key_path.is_absolute()
            or ".." in key_path.parts
        ):
            raise RuntimeError("dev-snapshot S3 object identity differs")
        return _object_root / normalized_bucket / Path(*key_path.parts)

    class NoSuchKey(KeyError):
        pass

    class _Paginator:
        def __init__(self, client: "_S3") -> None:
            self._client = client

        def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
            return [
                self._client.list_objects_v2(
                    Bucket=Bucket,
                    Prefix=Prefix,
                    MaxKeys=1000,
                )
            ]

    class _S3:
        def put_object(self, **kwargs: Any) -> dict[str, Any]:
            _require_declared_boundary("aws_s3_object_lock", "put_object")
            allowed = {
                "Bucket",
                "Key",
                "Body",
                "ContentType",
                "ServerSideEncryption",
                "SSEKMSKeyId",
            }
            if set(kwargs) != allowed:
                raise RuntimeError("dev-snapshot S3 put contract differs")
            bucket = str(kwargs["Bucket"])
            key = str(kwargs["Key"])
            body = bytes(kwargs["Body"])
            content_type = str(kwargs["ContentType"])
            if (
                not body
                or content_type
                not in {"application/json", "application/octet-stream"}
                or kwargs["ServerSideEncryption"] != "aws:kms"
                or str(kwargs["SSEKMSKeyId"]) != _kms_key_id
            ):
                raise RuntimeError("dev-snapshot S3 encryption contract differs")
            path = _object_path(bucket, key)
            pointer_key = _base_prefix + "current.json"
            if key == pointer_key:
                try:
                    pointer = json.loads(body)
                    target_uri = str(pointer["snapshot_uri"])
                    target_prefix = target_uri.removeprefix(
                        f"s3://{_bucket}/"
                    ).rstrip("/") + "/"
                    if (
                        not target_prefix.startswith(_base_prefix)
                        or target_prefix == _base_prefix
                        or not _object_path(
                            _bucket, target_prefix + "READY.json"
                        ).is_file()
                    ):
                        raise ValueError
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    raise RuntimeError(
                        "dev-snapshot pointer advanced before immutable READY"
                    ) from None
            elif path.exists() and path.read_bytes() != body:
                raise RuntimeError("dev-snapshot immutable S3 object differs")
            path.parent.mkdir(parents=True, exist_ok=True)
            staging = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
            staging.write_bytes(body)
            os.replace(staging, path)
            _event(
                "s3",
                "put_object",
                key=key,
                body_sha256="sha256:" + hashlib.sha256(body).hexdigest(),
                current_pointer=key == pointer_key,
            )
            return {
                "ETag": '"' + hashlib.md5(body).hexdigest() + '"',  # noqa: S324
                "VersionId": "rehearsal-"
                + hashlib.sha256(body).hexdigest()[:24],
            }

        def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
            _require_declared_boundary("aws_s3_object_lock", "get_object")
            path = _object_path(Bucket, Key)
            if not path.is_file():
                raise NoSuchKey(str(Key))
            body = path.read_bytes()
            _event("s3", "get_object", key=str(Key), body_size=len(body))
            return {
                "Body": io.BytesIO(body),
                "ContentLength": len(body),
                "VersionId": "rehearsal-"
                + hashlib.sha256(body).hexdigest()[:24],
            }

        def list_objects_v2(
            self,
            *,
            Bucket: str,
            Prefix: str,
            MaxKeys: int,
        ) -> dict[str, Any]:
            _require_declared_boundary(
                "aws_s3_object_lock", "list_objects_v2"
            )
            prefix_path = _object_path(Bucket, Prefix)
            root = _object_root / str(Bucket)
            contents: list[dict[str, str]] = []
            if prefix_path.is_file():
                contents.append({"Key": str(Prefix)})
            elif prefix_path.exists():
                for path in sorted(prefix_path.rglob("*")):
                    if path.is_file():
                        contents.append(
                            {"Key": path.relative_to(root).as_posix()}
                        )
            else:
                parent = prefix_path.parent
                stem = prefix_path.name
                if parent.exists():
                    for path in sorted(parent.glob(stem + "*")):
                        if path.is_file():
                            contents.append(
                                {"Key": path.relative_to(root).as_posix()}
                            )
                        elif path.is_dir():
                            for nested in sorted(path.rglob("*")):
                                if nested.is_file():
                                    contents.append(
                                        {
                                            "Key": nested.relative_to(root).as_posix()
                                        }
                                    )
            if int(MaxKeys) < 1:
                raise RuntimeError("dev-snapshot S3 pagination is invalid")
            selected = contents[: int(MaxKeys)]
            _event(
                "s3",
                "list_objects_v2",
                prefix=str(Prefix),
                result_count=len(selected),
            )
            return {
                "Contents": selected,
                "IsTruncated": len(contents) > len(selected),
            }

        def get_paginator(self, operation: str) -> _Paginator:
            if str(operation) != "list_objects_v2":
                raise RuntimeError("dev-snapshot S3 paginator is not allowlisted")
            return _Paginator(self)

    def _signature(key_id: str, message: bytes) -> bytes:
        return hashlib.sha256(
            b"leadpoet-rehearsal-dev-snapshot\0"
            + key_id.encode("utf-8")
            + b"\0"
            + bytes(message)
        ).digest()

    class _KMS:
        @staticmethod
        def _validate(
            *,
            KeyId: str,
            Message: bytes,
            MessageType: str,
            SigningAlgorithm: str,
        ) -> None:
            if (
                str(KeyId) != _kms_key_id
                or str(MessageType) != "RAW"
                or str(SigningAlgorithm) != "ECDSA_SHA_256"
                or re.fullmatch(rb"sha256:[0-9a-f]{64}", bytes(Message))
                is None
            ):
                raise RuntimeError("dev-snapshot KMS contract differs")

        def sign(self, **kwargs: Any) -> dict[str, Any]:
            _require_declared_boundary("aws_kms", "sign")
            self._validate(**kwargs)
            signature = _signature(str(kwargs["KeyId"]), bytes(kwargs["Message"]))
            _event(
                "kms",
                "sign",
                message_hash="sha256:"
                + hashlib.sha256(bytes(kwargs["Message"])).hexdigest(),
            )
            return {
                "KeyId": str(kwargs["KeyId"]),
                "SigningAlgorithm": "ECDSA_SHA_256",
                "Signature": signature,
            }

        def verify(self, **kwargs: Any) -> dict[str, Any]:
            _require_declared_boundary("aws_kms", "verify")
            signature = bytes(kwargs.pop("Signature"))
            self._validate(**kwargs)
            valid = signature == _signature(
                str(kwargs["KeyId"]), bytes(kwargs["Message"])
            )
            _event("kms", "verify", signature_valid=valid)
            return {
                "KeyId": str(kwargs["KeyId"]),
                "SigningAlgorithm": "ECDSA_SHA_256",
                "SignatureValid": valid,
            }

    try:
        import boto3 as _boto3
    except ImportError:
        _boto3 = ModuleType("boto3")
        sys.modules["boto3"] = _boto3

    def _boto3_client(service_name: str, *_args: Any, **_kwargs: Any) -> Any:
        if service_name == "s3":
            return _S3()
        if service_name == "kms":
            return _KMS()
        _event(
            "aws",
            "rejected",
            service_class="unknown",
            **_negative_probe_fields("aws_service"),
        )
        raise RuntimeError("dev-snapshot AWS service is not allowlisted")

    _boto3.client = _boto3_client  # type: ignore[attr-defined]

    def _provider_response(prepared: Any) -> Any:
        import requests

        _require_declared_boundary("http_service", "provider_request")
        method = str(prepared.method or "").upper()
        parsed = urlsplit(str(prepared.url or ""))
        headers = {
            str(name).lower(): str(value)
            for name, value in dict(prepared.headers or {}).items()
        }
        try:
            request_body = json.loads(bytes(prepared.body or b"{}").decode("utf-8"))
        except (AttributeError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
            request_body = None
        query = parse_qs(parsed.query)
        if (
            parsed.hostname == "api.exa.ai"
            and parsed.path == "/search"
            and method == "POST"
            and headers.get("x-api-key") == "rehearsal-exa-key"
            and isinstance(request_body, Mapping)
            and set(request_body) == {"query", "numResults"}
            and re.fullmatch(
                r"dev-snapshot-[0-9]{3}", str(request_body.get("query") or "")
            )
            and request_body.get("numResults") == 1
        ):
            provider = "exa"
            body = {"results": [{"title": "rehearsal evidence"}]}
        elif (
            parsed.hostname == "api.scrapingdog.com"
            and parsed.path == "/scrape"
            and method == "GET"
            and set(query) == {"api_key", "url"}
            and query.get("api_key") == ["rehearsal-scrapingdog-key"]
            and len(query.get("url") or ()) == 1
            and re.fullmatch(
                r"https://dev-snapshot-[0-9]{3}\.example",
                str((query.get("url") or [""])[0]),
            )
        ):
            provider = "scrapingdog"
            body = {"html": "<html>rehearsal evidence</html>"}
        elif (
            parsed.hostname == "openrouter.ai"
            and parsed.path == "/api/v1/chat/completions"
            and method == "POST"
            and headers.get("authorization")
            == "Bearer rehearsal-openrouter-key"
            and isinstance(request_body, Mapping)
            and set(request_body) == {"model", "messages"}
            and request_body.get("model") == "openai/rehearsal-model"
            and isinstance(request_body.get("messages"), list)
            and len(request_body["messages"]) == 1
            and isinstance(request_body["messages"][0], Mapping)
            and request_body["messages"][0].get("role") == "user"
            and re.fullmatch(
                r"dev-snapshot-[0-9]{3}",
                str(request_body["messages"][0].get("content") or ""),
            )
        ):
            provider = "openrouter"
            body = {
                "choices": [
                    {"message": {"content": "rehearsal qualified"}}
                ]
            }
        else:
            return _reject_http_seam("requests")
        response = requests.models.Response()
        response.status_code = 200
        response._content = json.dumps(  # noqa: SLF001
            body, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        response.encoding = "utf-8"
        response.headers.update({"content-type": "application/json"})
        response.url = str(prepared.url)
        response.request = prepared
        _event(
            "provider",
            "request",
            provider=provider,
            method=method,
            client="requests",
        )
        return response

    def _reject_http_seam(client: str) -> Any:
        _require_declared_boundary("http_service", "provider_request")
        _event(
            "http_seam",
            "rejected",
            client=str(client),
            **_negative_probe_fields(str(client)),
        )
        raise RuntimeError(
            f"dev-snapshot HTTP seam {client} is not allowlisted"
        )

    try:
        import requests as _requests
    except ImportError:
        _requests = None
    if _requests is not None:
        _requests.sessions.Session.send = (  # type: ignore[method-assign]
            lambda _session, prepared, *_args, **_kwargs: _provider_response(
                prepared
            )
        )

    import urllib.request as _urllib_request

    _urllib_request.urlopen = (  # type: ignore[assignment]
        lambda *_args, **_kwargs: _reject_http_seam("urllib")
    )

    try:
        import httpx as _httpx
    except ImportError:
        _httpx = None
    if _httpx is not None:
        _httpx.Client.send = (  # type: ignore[method-assign]
            lambda _client, _request, *_args, **_kwargs: _reject_http_seam(
                "httpx_sync"
            )
        )

        async def _reject_httpx_async(
            _client: Any, _request: Any, *_args: Any, **_kwargs: Any
        ) -> Any:
            return _reject_http_seam("httpx_async")

        _httpx.AsyncClient.send = _reject_httpx_async  # type: ignore[method-assign]

    try:
        import aiohttp as _aiohttp
    except ImportError:
        _aiohttp = None
    if _aiohttp is not None:

        async def _reject_aiohttp(
            _session: Any,
            _method: Any,
            _url: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> Any:
            return _reject_http_seam("aiohttp")

        _aiohttp.ClientSession._request = (  # type: ignore[method-assign]  # noqa: SLF001
            _reject_aiohttp
        )

    _PRODUCTION_SCRIPT_NAMES = {
        "export_research_lab_dev_icp_inputs.py",
        "record_research_lab_dev_snapshots.py",
        "publish_research_lab_dev_snapshot.py",
    }
    _PRODUCTION_PHASE_ORDER = (
        "export",
        "record",
        "publish_immutable",
        "publish_pointer",
    )

    def _canonical_contract_hash(value: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def _redacted_argv_for_phase(phase: str) -> list[str]:
        provider_args: list[str] = []
        for index, _model_id in enumerate(_provider_model_ids):
            provider_args.extend(
                ("--provider-model-id", f"<provider-model-id:{index}>")
            )
        shapes = {
            "export": [
                "<python>",
                "<candidate>/scripts/export_research_lab_dev_icp_inputs.py",
                "--out-dir",
                "<refresh>/inputs",
                "--seed",
                "<selection-seed>",
                "--expected-private-model-manifest-hash",
                "<active-manifest-hash>",
            ],
            "record": [
                "<python>",
                "<candidate>/scripts/record_research_lab_dev_snapshots.py",
                "--source-icps",
                "<refresh>/inputs/source_icps.json",
                "--snapshot-dir",
                "<refresh>/snapshot",
                "--champion-image",
                "<active-image-digest>",
                "--source-commit",
                "<active-source-commit>",
                "--model-config-hash",
                "<active-config-hash>",
                "--private-model-manifest-hash",
                "<active-manifest-hash>",
                "--cancel-file",
                "<refresh>/cancel-recording",
                "--record",
                *provider_args,
            ],
            "publish_immutable": [
                "<python>",
                "<candidate>/scripts/publish_research_lab_dev_snapshot.py",
                "--source-dir",
                "<refresh>/snapshot",
                "--s3-base-uri",
                "<snapshot-base-uri>",
                "--kms-key-id",
                "<snapshot-kms-key-id>",
                "--skip-current-pointer",
            ],
            "publish_pointer": [
                "<python>",
                "<candidate>/scripts/publish_research_lab_dev_snapshot.py",
                "--source-dir",
                "<refresh>/snapshot",
                "--s3-base-uri",
                "<snapshot-base-uri>",
                "--kms-key-id",
                "<snapshot-kms-key-id>",
            ],
        }
        return shapes[phase]

    def _validate_refresh_work_dir(path: Path) -> Path:
        resolved = path.resolve()
        if (
            resolved.parent != (_root / "work").resolve()
            or re.fullmatch(r"refresh-[0-9]+-[A-Za-z0-9_-]+", resolved.name)
            is None
        ):
            raise RuntimeError("dev-snapshot refresh work directory differs")
        return resolved

    def _bind_refresh_work_dir(path: Path) -> Path:
        global _observed_refresh_work_dir
        resolved = _validate_refresh_work_dir(path)
        if _observed_refresh_work_dir is None:
            _observed_refresh_work_dir = resolved
        elif _observed_refresh_work_dir != resolved:
            raise RuntimeError("dev-snapshot production commands mixed work roots")
        return resolved

    def _validate_production_command(argv: Sequence[str]) -> tuple[str, str]:
        if len(argv) < 2:
            raise RuntimeError("dev-snapshot production command is incomplete")
        script_name = Path(str(argv[1])).name
        script_path = (_source_root / "scripts" / script_name).resolve()
        if (
            str(argv[0]) != sys.executable
            or script_name not in _PRODUCTION_SCRIPT_NAMES
            or Path(str(argv[1])).resolve() != script_path
            or not script_path.is_file()
        ):
            raise RuntimeError("dev-snapshot production command identity differs")

        if script_name == "export_research_lab_dev_icp_inputs.py":
            phase = "export"
            if len(argv) != 8:
                raise RuntimeError("dev-snapshot export argv shape differs")
            inputs_dir = Path(str(argv[3])).resolve()
            work_dir = _bind_refresh_work_dir(inputs_dir.parent)
            expected = [
                sys.executable,
                str(script_path),
                "--out-dir",
                str(work_dir / "inputs"),
                "--seed",
                _selection_seed,
                "--expected-private-model-manifest-hash",
                str(_active_artifact["manifest_hash"]),
            ]
        elif script_name == "record_research_lab_dev_snapshots.py":
            phase = "record"
            if len(argv) < 15:
                raise RuntimeError("dev-snapshot recorder argv shape differs")
            source_icps = Path(str(argv[3])).resolve()
            work_dir = _bind_refresh_work_dir(source_icps.parents[1])
            provider_args: list[str] = []
            for model_id in _provider_model_ids:
                provider_args.extend(("--provider-model-id", model_id))
            expected = [
                sys.executable,
                str(script_path),
                "--source-icps",
                str(work_dir / "inputs" / "source_icps.json"),
                "--snapshot-dir",
                str(work_dir / "snapshot"),
                "--champion-image",
                str(_active_artifact["image_digest"]),
                "--source-commit",
                str(_active_artifact["git_commit_sha"]),
                "--model-config-hash",
                str(_active_artifact["config_hash"]),
                "--private-model-manifest-hash",
                str(_active_artifact["manifest_hash"]),
                "--cancel-file",
                str(work_dir / "cancel-recording"),
                "--record",
                *provider_args,
            ]
        else:
            phase = (
                "publish_immutable"
                if list(argv[-1:]) == ["--skip-current-pointer"]
                else "publish_pointer"
            )
            source_dir = Path(str(argv[3])).resolve() if len(argv) >= 4 else Path()
            work_dir = _bind_refresh_work_dir(source_dir.parent)
            expected = [
                sys.executable,
                str(script_path),
                "--source-dir",
                str(work_dir / "snapshot"),
                "--s3-base-uri",
                f"s3://{_bucket}/{_base_prefix.rstrip('/')}",
                "--kms-key-id",
                _kms_key_id,
            ]
            if phase == "publish_immutable":
                expected.append("--skip-current-pointer")

        if list(argv) != expected:
            raise RuntimeError(
                f"dev-snapshot production {phase} argv contract differs"
            )
        expected_phase = _PRODUCTION_PHASE_ORDER[
            len(_observed_production_command_phases)
        ] if len(_observed_production_command_phases) < len(_PRODUCTION_PHASE_ORDER) else ""
        if phase != expected_phase:
            raise RuntimeError("dev-snapshot production command order differs")
        contract_hash = _canonical_contract_hash(
            {
                "schema_version": "leadpoet.rehearsal_dev_snapshot_argv.v1",
                "phase": phase,
                "redacted_argv": _redacted_argv_for_phase(phase),
            }
        )
        if contract_hash != str(_expected_cli_argv_contract_hashes.get(phase) or ""):
            raise RuntimeError("dev-snapshot redacted argv commitment differs")
        _observed_production_command_phases.append(phase)
        return phase, contract_hash

    def _docker_run(
        argv: Sequence[str],
        *,
        input_text: str,
        environment: Mapping[str, str],
        timeout: Any,
    ) -> subprocess.CompletedProcess[str]:
        _require_declared_boundary("docker_daemon", "run")
        from research_lab.eval import private_runtime
        from research_lab.eval.snapshot_store import (
            MISS_POLICY_STRICT,
            SNAPSHOT_DIR_ENV,
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
            container_replay_env,
            dev_record_bootstrap,
            dev_replay_bootstrap,
        )
        from scripts.record_research_lab_dev_snapshots import PROVIDER_KEY_GROUPS

        values = [str(value) for value in argv]
        if (
            len(values) < 12
            or values[:4] != ["docker", "run", "--rm", "--name"]
            or values[5] != "-i"
            or not 0.1 <= float(timeout or 0.0) <= 900.0
        ):
            raise RuntimeError("dev-snapshot Docker run contract differs")
        container_name = values[4]
        read_only = values[6:8] == ["--network", "none"]
        mount_index = 9 if read_only else 7
        if (
            values[mount_index - 1] != "-v"
            or re.fullmatch(
                r"leadpoet-dev-snapshot-(?:replay|record)-[0-9a-f]{32}",
                container_name,
            )
            is None
        ):
            raise RuntimeError("dev-snapshot Docker isolation contract differs")
        mount_suffix = (
            ":/research_lab_dev_snapshots:ro"
            if read_only
            else ":/research_lab_dev_snapshots"
        )
        mount = values[mount_index]
        if not mount.endswith(mount_suffix):
            raise RuntimeError("dev-snapshot Docker snapshot mount differs")
        snapshot_root_path = Path(mount[: -len(mount_suffix)]).resolve()
        work_dir = _bind_refresh_work_dir(snapshot_root_path.parent)
        if (
            snapshot_root_path.parent != work_dir
            or re.fullmatch(
                r"\.snapshot\.recording\.[0-9]+\.[0-9a-f]{32}",
                snapshot_root_path.name,
            )
            is None
            or not snapshot_root_path.is_dir()
        ):
            raise RuntimeError("dev-snapshot Docker staging mount differs")

        docker_bootstrap = getattr(
            private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", ""
        )
        expected_bootstrap = (
            dev_replay_bootstrap() if read_only else dev_record_bootstrap()
        ) + docker_bootstrap
        bootstrap_kind = "replay" if read_only else "record"
        bootstrap_hash = "sha256:" + hashlib.sha256(
            expected_bootstrap.encode("utf-8")
        ).hexdigest()
        if (
            not docker_bootstrap
            or bootstrap_hash
            != str(_expected_docker_bootstrap_hashes.get(bootstrap_kind) or "")
        ):
            raise RuntimeError("dev-snapshot Docker bootstrap commitment differs")

        if read_only:
            if not container_name.startswith("leadpoet-dev-snapshot-replay-"):
                raise RuntimeError("dev-snapshot replay container name differs")
            env_args: list[str] = []
            for name, value in container_replay_env(
                "/research_lab_dev_snapshots",
                miss_policy=MISS_POLICY_STRICT,
            ).items():
                env_args.extend(("-e", f"{name}={value}"))
            for group in PROVIDER_KEY_GROUPS:
                for name in group:
                    env_args.extend(("-e", f"{name}=research-lab-offline-replay"))
            expected = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "-i",
                "--network",
                "none",
                "-v",
                f"{snapshot_root_path}:/research_lab_dev_snapshots:ro",
                *env_args,
                _image_digest,
                "python",
                "-c",
                expected_bootstrap,
                "research_lab_adapter",
                "run_icp",
            ]
            reuse_existing = False
        else:
            if not container_name.startswith("leadpoet-dev-snapshot-record-"):
                raise RuntimeError("dev-snapshot record container name differs")
            icp_ref = str(environment.get("RESEARCH_LAB_DEV_RECORD_ICP_REF") or "")
            if not icp_ref.startswith("qualification_private_icp_sets:"):
                raise RuntimeError("dev-snapshot Docker ICP binding differs")
            env_args = [
                "-e",
                f"{SNAPSHOT_DIR_ENV}=/research_lab_dev_snapshots",
                "-e",
                f"RESEARCH_LAB_DEV_RECORD_ICP_REF={icp_ref}",
            ]
            for name in private_runtime.private_model_env_passthrough():
                if name in environment:
                    env_args.extend(("-e", name))
            base = [
                "docker",
                "run",
                "--rm",
                "--name",
                container_name,
                "-i",
                "-v",
                f"{snapshot_root_path}:/research_lab_dev_snapshots",
                *env_args,
            ]
            tail = [
                _image_digest,
                "python",
                "-c",
                expected_bootstrap,
                "research_lab_adapter",
                "run_icp",
            ]
            expected_without_reuse = [*base, *tail]
            expected_with_reuse = [
                *base,
                "-e",
                f"{SNAPSHOT_RECORD_REUSE_EXISTING_ENV}=true",
                *tail,
            ]
            if values == expected_without_reuse:
                reuse_existing = False
                expected = expected_without_reuse
            elif values == expected_with_reuse:
                reuse_existing = True
                expected = expected_with_reuse
            else:
                expected = []
                reuse_existing = False

        if values != expected:
            raise RuntimeError("dev-snapshot Docker argv or environment differs")
        try:
            payload = json.loads(str(input_text))
        except json.JSONDecodeError as exc:
            raise RuntimeError("dev-snapshot Docker stdin is invalid") from exc
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"icp", "context"}
            or not isinstance(payload.get("icp"), Mapping)
            or not isinstance(payload.get("context"), Mapping)
            or payload["context"].get("dev_snapshot_recording") is not True
            or not isinstance(payload["context"].get("runtime_options"), Mapping)
        ):
            raise RuntimeError("dev-snapshot Docker stdin contract differs")

        container_env: dict[str, str] = {}
        for index, item in enumerate(values):
            if item != "-e" or index + 1 >= len(values):
                continue
            name, separator, configured = values[index + 1].partition("=")
            if name in container_env:
                raise RuntimeError("dev-snapshot Docker environment is duplicated")
            if separator:
                container_env[name] = configured
            elif name in environment:
                container_env[name] = str(environment[name])
            else:
                raise RuntimeError("dev-snapshot Docker inherited environment is absent")
        snapshot_root = str(snapshot_root_path)
        for key, value in list(container_env.items()):
            if value == "/research_lab_dev_snapshots":
                container_env[key] = snapshot_root
        child_env = dict(environment)
        child_env.update(container_env)
        child_env["PYTHONPATH"] = os.pathsep.join(
            (str(_adapter_root), str(_source_root))
        )
        child_env[STATE_ENV] = str(_state_path)
        completed = _real_subprocess_run(
            [
                sys.executable,
                "-c",
                expected_bootstrap,
                "research_lab_adapter",
                "run_icp",
            ],
            input=input_text,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=child_env,
            cwd=str(_champion_root),
            check=False,
        )
        _event(
            "provider_container",
            bootstrap_kind,
            returncode=int(completed.returncode),
            network_disabled=read_only,
            reuse_existing=reuse_existing,
            bootstrap_hash=bootstrap_hash,
            argv_exact=True,
            timeout_bounded=True,
        )
        return completed

    def _docker_info(
        argv: Sequence[str],
        *,
        options: Mapping[str, Any],
    ) -> subprocess.CompletedProcess[str]:
        """Model the shared-lock daemon readiness probe exactly."""

        _require_declared_boundary("docker_daemon", "state")
        timeout = options.get("timeout")
        try:
            timeout_seconds = float(timeout)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "dev-snapshot Docker info timeout differs"
            ) from exc
        if (
            list(argv) != ["docker", "info"]
            or set(options)
            != {"text", "capture_output", "timeout", "env", "check"}
            or options.get("text") is not True
            or options.get("capture_output") is not True
            or options.get("check") is not False
            or not isinstance(options.get("env"), Mapping)
            or not 0.1 <= timeout_seconds <= 10.0
        ):
            raise RuntimeError("dev-snapshot Docker info contract differs")
        _event(
            "docker_daemon",
            "state",
            returncode=0,
            ready=True,
            argv_exact=True,
            timeout_bounded=True,
        )
        return subprocess.CompletedProcess(
            list(argv),
            0,
            stdout="rehearsal Docker daemon ready\n",
            stderr="",
        )

    def _patched_run(*popenargs: Any, **kwargs: Any) -> Any:
        command = popenargs[0] if popenargs else kwargs.get("args")
        if isinstance(command, (list, tuple)) and command:
            argv = [str(value) for value in command]
            executable = Path(argv[0]).name
            if executable == "docker":
                if argv[1:2] == ["run"]:
                    try:
                        completed = _docker_run(
                            argv,
                            input_text=str(kwargs.get("input") or ""),
                            environment=dict(kwargs.get("env") or os.environ),
                            timeout=kwargs.get("timeout"),
                        )
                    except Exception:
                        _event(
                            "subprocess",
                            "rejected",
                            command_class="docker",
                            command_name="docker",
                            **_negative_probe_fields("docker_argv"),
                        )
                        raise
                elif argv[1:2] == ["info"]:
                    try:
                        completed = _docker_info(argv, options=kwargs)
                    except Exception:
                        _event(
                            "subprocess",
                            "rejected",
                            command_class="docker",
                            command_name="docker",
                            **_negative_probe_fields("docker_argv"),
                        )
                        raise
                elif argv[1:3] == ["rm", "-f"] and len(argv) == 4:
                    _require_declared_boundary("docker_daemon", "remove")
                    if re.fullmatch(
                        r"leadpoet-dev-snapshot-(?:record|replay)-[0-9a-f]{32}",
                        argv[3],
                    ) is None:
                        _event(
                            "subprocess",
                            "rejected",
                            command_class="docker",
                            command_name="docker",
                            **_negative_probe_fields("docker_argv"),
                        )
                        raise RuntimeError(
                            "dev-snapshot Docker removal target differs"
                        )
                    _event("provider_container", "remove")
                    completed = subprocess.CompletedProcess(
                        argv, 0, stdout="", stderr=""
                    )
                else:
                    _event(
                        "subprocess",
                        "rejected",
                        command_class="docker",
                        command_name="docker",
                        **_negative_probe_fields("docker_argv"),
                    )
                    raise RuntimeError(
                        "dev-snapshot subprocess operation is not allowlisted"
                    )
                if kwargs.get("check") and completed.returncode:
                    raise subprocess.CalledProcessError(
                        completed.returncode,
                        argv,
                        output=completed.stdout,
                        stderr=completed.stderr,
                    )
                return completed
            if len(argv) >= 2 and Path(argv[1]).name in _PRODUCTION_SCRIPT_NAMES:
                phase, argv_contract_hash = _validate_production_command(argv)
                completed = _real_subprocess_run(*popenargs, **kwargs)
                _event(
                    "production_command",
                    Path(argv[1]).name,
                    phase=phase,
                    argv_contract_hash=argv_contract_hash,
                    argv_redacted=True,
                    argv_argument_count=len(argv),
                    returncode=int(completed.returncode),
                )
                return completed
            _event(
                "subprocess",
                "rejected",
                command_class=(
                    "python" if executable.startswith("python") else "other"
                ),
                command_name=executable,
                **_negative_probe_fields("subprocess"),
            )
            raise RuntimeError(
                "dev-snapshot subprocess operation is not allowlisted"
            )
        _event(
            "subprocess",
            "rejected",
            command_class="non_argv",
            **_negative_probe_fields("subprocess"),
        )
        raise RuntimeError("dev-snapshot subprocess operation is not allowlisted")

    subprocess.run = _patched_run

"""Measured SOURCE_ADD provenance authority with authenticated provider I/O."""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Callable, Dict, Mapping
from urllib.parse import urlencode

from gateway.research_lab.source_add_provenance import (
    SourceAddProvenanceResult,
    evaluate_source_add_provenance,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from gateway.tee.source_add_runtime_v2 import (
    build_source_add_probe_route_v2,
    source_add_dynamic_retry_policy_hash,
)
from gateway.tee.supabase_source_v2 import SupabaseSourceReaderV2
from leadpoet_canonical.attested_v2 import canonical_json, sha256_bytes, sha256_json


SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION = (
    "leadpoet.source_add_provenance_request.v2"
)
SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION = (
    "leadpoet.source_add_provenance_result.v2"
)
OP_SOURCE_ADD_PROVENANCE_V2 = "source_add_provenance_v2"
OP_SOURCE_ADD_FUNCTIONAL_PROBE_V2 = "source_add_functional_probe_v2"
SOURCE_ADD_FUNCTIONAL_PROBE_REQUEST_SCHEMA_VERSION = (
    "leadpoet.source_add_functional_probe_request.v2"
)
SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION = (
    "leadpoet.source_add_functional_probe_result.v2"
)
SOURCE_ADD_FUNCTIONAL_PROBE_EVALUATOR_VERSION = (
    "leadpoet.source_add_functional_probe_evaluator.v2.1"
)
SCRAPINGDOG_ORIGIN = "https://api.scrapingdog.com"
WAYBACK_ORIGIN = "https://archive.org"
_MAX_PROVIDER_BODY = 240_000
_MAX_PROBE_BODY = 1024 * 1024
_MAX_JSON_DEPTH = 12
_MAX_JSON_NODES = 20_000
_MAX_JSON_KEYS = 10_000
_MAX_JSON_STRING = 100_000
_NON_DATA_PROBE_TERMINALS = frozenset(
    {
        "api-docs",
        "docs",
        "documentation",
        "health",
        "healthcheck",
        "healthz",
        "openapi.json",
        "ping",
        "readiness",
        "ready",
        "status",
        "swagger",
        "swagger.json",
        "version",
    }
)


class CoordinatorSourceAddV2Error(RuntimeError):
    """A SOURCE_ADD provenance request or authenticated response is invalid."""


class CoordinatorSourceAddProvenanceV2:
    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hash: str,
        wayback_retry_policy_hash: str = "",
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hash = str(retry_policy_hash or "")
        if not self._retry_policy_hash:
            raise CoordinatorSourceAddV2Error(
                "ScrapingDog retry policy is unavailable"
            )
        self._wayback_retry_policy_hash = str(wayback_retry_policy_hash or "")

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "submission_id",
            "source_name",
            "source_kind",
            "declared_base_domains",
            "source_metadata",
            "timeout_seconds",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provenance fields are invalid"
            )
        if payload.get("schema_version") != SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provenance schema is invalid"
            )
        domains = payload.get("declared_base_domains")
        metadata = payload.get("source_metadata")
        timeout_seconds = payload.get("timeout_seconds")
        if (
            not isinstance(domains, list)
            or any(not isinstance(item, str) for item in domains)
            or not isinstance(metadata, Mapping)
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 120
        ):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provenance inputs are invalid"
            )

        def provider_fetch(
            path: str,
            params: Mapping[str, str],
            timeout: int,
        ) -> Mapping[str, Any]:
            return self._fetch(
                path=path,
                params=params,
                timeout_seconds=timeout,
                context=context,
            )

        result = evaluate_source_add_provenance(
            source_name=str(payload.get("source_name") or ""),
            source_kind=str(payload.get("source_kind") or ""),
            declared_base_domains=domains,
            source_metadata=dict(metadata),
            timeout_seconds=timeout_seconds,
            provider_fetch=provider_fetch,
        )
        if not isinstance(result, SourceAddProvenanceResult):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provenance result is invalid"
            )
        document = {
            "schema_version": SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION,
            "submission_id": str(payload.get("submission_id") or ""),
            "precheck_status": result.precheck_status,
            "reasons": list(result.reasons),
            "precheck_doc": result.to_record_doc(),
        }
        context.record_artifact(sha256_json(document["precheck_doc"]))
        return document

    def _fetch(
        self,
        *,
        path: str,
        params: Mapping[str, str],
        timeout_seconds: int,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if path not in {"/scrape", "/google/ai_mode", "/archive/available"}:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provider path is not measured"
            )
        if path == "/archive/available":
            provider_id = "wayback"
            origin = WAYBACK_ORIGIN
            provider_path = "/wayback/available"
            retry_policy_hash = self._wayback_retry_policy_hash
            logical_suffix = "archive-history"
            if not retry_policy_hash:
                raise CoordinatorSourceAddV2Error(
                    "Wayback retry policy is unavailable"
                )
        else:
            provider_id = "scrapingdog"
            origin = SCRAPINGDOG_ORIGIN
            provider_path = path
            retry_policy_hash = self._retry_policy_hash
            logical_suffix = "documentation" if path == "/scrape" else "ai-mode"
        result = dict(
            self._execute_provider(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": "%s:source-add:%s"
                    % (context.job_id, logical_suffix),
                    "job_id": context.job_id,
                    "purpose": context.purpose,
                    "provider_id": provider_id,
                    "attempt_number": 0,
                    "method": "GET",
                    "url": origin + provider_path + "?" + urlencode(dict(params)),
                    "headers": {
                        "accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                        "user-agent": "leadpoet-source-add-precheck/1.0",
                    },
                    "body_b64": "",
                    "timeout_ms": int(timeout_seconds) * 1000,
                    "retry_policy_hash": retry_policy_hash,
                    "max_response_bytes": _MAX_PROVIDER_BODY,
                    "artifact_mode": "hash_only",
                }
            )
        )
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provider terminal attempt is missing"
            )
        context.record_transport(attempt)
        context.record_artifact(str(attempt["request_artifact_hash"]))
        if attempt.get("terminal_status") != "authenticated_response":
            return {
                "provider_status": "error",
                "error_type": str(
                    result.get("failure_code")
                    or attempt.get("failure_code")
                    or "transport_failure"
                )[:80],
                "error": "authenticated provider transport failed",
            }
        context.record_artifact(str(attempt["response_artifact_hash"]))
        try:
            body = base64.b64decode(str(result.get("body_b64") or ""), validate=True)
        except Exception as exc:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provider response encoding is invalid"
            ) from exc
        if sha256_bytes(body) != attempt.get("response_hash"):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provider response differs from terminal record"
            )
        body = body[:_MAX_PROVIDER_BODY]
        text = body.decode("utf-8", "replace")
        parsed = None
        try:
            parsed = json.loads(text)
        except Exception:
            parsed = None
        headers = result.get("headers")
        content_type = ""
        if isinstance(headers, Mapping):
            content_type = next(
                (
                    str(value)
                    for name, value in headers.items()
                    if str(name).lower() == "content-type"
                ),
                "",
            )
        return {
            "provider_status": "ok",
            "status": int(result.get("http_status") or 0),
            "content_type": content_type[:120],
            "body_text": text,
            "json": parsed if isinstance(parsed, Mapping) else None,
        }


class CoordinatorSourceAddFunctionalProbeV2:
    """Execute a provisional SOURCE_ADD route without catalog eligibility."""

    def __init__(
        self,
        *,
        reader: SupabaseSourceReaderV2,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    ) -> None:
        self._reader = reader
        self._execute_provider = execute_provider

    def resolve(
        self,
        *,
        payload: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        required = {
            "schema_version",
            "submission_id",
            "config_ref",
            "evaluation_mode",
            "timeout_seconds",
        }
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe fields are invalid"
            )
        if (
            payload.get("schema_version")
            != SOURCE_ADD_FUNCTIONAL_PROBE_REQUEST_SCHEMA_VERSION
        ):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe schema is invalid"
            )
        submission_id = str(payload.get("submission_id") or "")
        config_ref = str(payload.get("config_ref") or "")
        evaluation_mode = str(payload.get("evaluation_mode") or "")
        timeout_seconds = payload.get("timeout_seconds")
        if (
            evaluation_mode not in {"functional_probe", "provisioning_smoke"}
            or isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 1 <= timeout_seconds <= 120
        ):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe policy is invalid"
            )

        submission_rows = self._read(
            "source_add_submission_by_id",
            {"submission_id": submission_id},
            context,
        )
        config_rows = self._read(
            "source_add_probe_config_by_submission",
            {"submission_id": submission_id},
            context,
        )
        if len(submission_rows) != 1 or len(config_rows) != 1:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe source is missing or ambiguous"
            )
        submission = submission_rows[0]
        config = config_rows[0]
        if (
            str(config.get("config_ref") or "") != config_ref
            or str(config.get("config_status") or "") != "active"
            or str(config.get("adapter_id") or "")
            != str(submission.get("adapter_id") or "")
        ):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe config differs from current state"
            )
        measured_row = {
            **dict(config),
            "miner_hotkey": str(submission.get("miner_hotkey") or ""),
        }
        route = build_source_add_probe_route_v2(measured_row)
        probe_doc = config.get("probe_doc")
        if not isinstance(probe_doc, Mapping):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe document is invalid"
            )
        probes = probe_doc.get("probes")
        if not isinstance(probes, list) or not probes:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe examples are missing"
            )

        summaries = []
        selected = None
        for index, probe in enumerate(probes[:3]):
            summary = self._execute_one(
                index=index,
                probe=probe,
                probe_doc=probe_doc,
                route=route,
                timeout_seconds=timeout_seconds,
                context=context,
            )
            summaries.append(summary)
            if summary["result_status"] == "passed":
                selected = summary
                break
        if selected is None:
            order = ("retryable", "awaiting_operator", "manual_review", "failed")
            selected = next(
                (
                    item
                    for status in order
                    for item in summaries
                    if item["result_status"] == status
                ),
                summaries[-1],
            )

        result = {
            "schema_version": SOURCE_ADD_FUNCTIONAL_PROBE_RESULT_SCHEMA_VERSION,
            "evaluator_version": SOURCE_ADD_FUNCTIONAL_PROBE_EVALUATOR_VERSION,
            "submission_id": submission_id,
            "adapter_id": str(submission.get("adapter_id") or ""),
            "config_ref": config_ref,
            "evaluation_mode": evaluation_mode,
            "result_status": selected["result_status"],
            "route_hash": route["route_hash"],
            "selected_probe_index": selected["probe_index"],
            "response_hash": selected["response_hash"],
            "status_class": selected["status_class"],
            "content_type": selected["content_type"],
            "byte_count": selected["byte_count"],
            "duration_ms": selected["duration_ms"],
            "retry_after_seconds": selected["retry_after_seconds"],
            "reason_codes": selected["reason_codes"],
            "probe_summaries": summaries,
        }
        context.record_artifact(sha256_json(result))
        return result

    def _execute_one(
        self,
        *,
        index: int,
        probe: Any,
        probe_doc: Mapping[str, Any],
        route: Mapping[str, Any],
        timeout_seconds: int,
        context: ExecutionContextV2,
    ) -> Dict[str, Any]:
        if not isinstance(probe, Mapping):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe example is invalid"
            )
        method = str(probe.get("method") or "").upper()
        path = str(probe.get("path") or "")
        query = probe.get("query")
        body_json = probe.get("body_json")
        if not isinstance(query, Mapping):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional probe query is invalid"
            )
        headers = {
            str(name): str(value)
            for name, value in dict(probe_doc.get("request_headers") or {}).items()
        }
        if not any(name.lower() == "accept" for name in headers):
            headers["Accept"] = "application/json"
        if not any(name.lower() == "user-agent" for name in headers):
            headers["User-Agent"] = "leadpoet-source-add-probe/2.0"
        query_string = urlencode(
            [(str(name), str(value)) for name, value in sorted(query.items())]
        )
        url = str(route["base_url"]) + path
        if query_string:
            url += "?" + query_string
        body = b""
        if method == "POST":
            body = canonical_json(body_json).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")

        started = time.monotonic()
        result = dict(
            self._execute_provider(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": "%s:source-add-probe:%d"
                    % (context.job_id, index),
                    "job_id": context.job_id,
                    "purpose": context.purpose,
                    "provider_id": str(route["provider_id"]),
                    "attempt_number": index,
                    "method": method,
                    "url": url,
                    "headers": headers,
                    "body_b64": base64.b64encode(body).decode("ascii"),
                    "timeout_ms": timeout_seconds * 1000,
                    "retry_policy_hash": source_add_dynamic_retry_policy_hash(route),
                    "dynamic_route": dict(route),
                    "max_response_bytes": _MAX_PROBE_BODY,
                    "artifact_mode": "hash_only",
                }
            )
        )
        duration_ms = min(180_000, int((time.monotonic() - started) * 1000))
        attempt = result.get("transport_attempt")
        if not isinstance(attempt, Mapping):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD probe terminal attempt is missing"
            )
        context.record_transport(attempt)
        context.record_artifact(str(attempt["request_artifact_hash"]))
        response_artifact_hash = str(attempt.get("response_artifact_hash") or "")
        if response_artifact_hash:
            context.record_artifact(response_artifact_hash)
        if attempt.get("terminal_status") != "authenticated_response":
            failure_code = str(
                result.get("failure_code")
                or attempt.get("failure_code")
                or "transport_failure"
            )[:80]
            result_status = (
                "failed"
                if failure_code in {"tls_failure", "response_too_large"}
                else "retryable"
            )
            return self._summary(
                index=index,
                result_status=result_status,
                response_hash="",
                status_class="transport_failure",
                content_type="",
                byte_count=0,
                duration_ms=duration_ms,
                reason_codes=[failure_code],
            )

        try:
            response_body = base64.b64decode(
                str(result.get("body_b64") or ""), validate=True
            )
        except Exception as exc:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional response encoding is invalid"
            ) from exc
        response_hash = sha256_bytes(response_body)
        if response_hash != str(attempt.get("response_hash") or ""):
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD functional response differs from terminal record"
            )
        headers_out = result.get("headers")
        normalized_headers = {
            str(name).lower(): str(value)
            for name, value in dict(headers_out or {}).items()
        }
        content_type = normalized_headers.get("content-type", "")[:160]
        retry_after_seconds = self._bounded_retry_after(
            normalized_headers.get("retry-after", "")
        )
        status = int(result.get("http_status") or 0)
        status_class = "%dxx" % (status // 100) if status else "unknown"
        reason_codes = []
        result_status = "failed"
        if status in {408, 429} or status >= 500:
            result_status = "retryable"
            reason_codes.append("http_%d" % status)
        elif status in {401, 403}:
            result_status = "awaiting_operator"
            reason_codes.append("operator_credential_or_headers_required")
        elif status in {404, 410}:
            reason_codes.append("endpoint_not_found")
        elif 300 <= status < 400:
            reason_codes.append("redirect_forbidden")
        elif not 200 <= status < 300:
            result_status = "manual_review"
            reason_codes.append("unexpected_http_status")
        else:
            result_status, reason_codes = self._validate_json_response(
                response_body=response_body,
                content_type=content_type,
            )
            if result_status == "passed" and self._is_non_data_probe_path(path):
                result_status = "manual_review"
                reason_codes = ["non_data_probe_endpoint"]
        return self._summary(
            index=index,
            result_status=result_status,
            response_hash=response_hash,
            status_class=status_class,
            content_type=content_type,
            byte_count=len(response_body),
            duration_ms=duration_ms,
            retry_after_seconds=retry_after_seconds,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _bounded_retry_after(value: str) -> int:
        text = str(value or "").strip()
        if not text.isdigit():
            return 0
        return min(21_600, max(0, int(text)))

    @staticmethod
    def _validate_json_response(
        *, response_body: bytes, content_type: str
    ) -> tuple[str, list[str]]:
        media_type = str(content_type or "").split(";", 1)[0].strip().lower()
        if media_type != "application/json" and not media_type.endswith("+json"):
            return "failed", ["non_json_content_type"]
        try:
            text = response_body.decode("utf-8", "strict")
        except UnicodeDecodeError:
            return "failed", ["invalid_utf8"]
        try:
            value = json.loads(text)
        except Exception:
            return "failed", ["malformed_json"]
        if not isinstance(value, (dict, list)):
            return "failed", ["json_root_not_object_or_list"]
        if not value:
            return "failed", ["empty_json_payload"]
        counters = {"nodes": 0, "keys": 0}
        try:
            CoordinatorSourceAddFunctionalProbeV2._validate_json_shape(
                value, depth=0, counters=counters
            )
        except CoordinatorSourceAddV2Error as exc:
            return "failed", [str(exc)]
        if isinstance(value, dict):
            keys = {str(item).lower() for item in value}
            data_keys = {"data", "items", "records", "results", "response"}
            if keys.intersection({"error", "errors", "exception", "fault"}) and not keys.intersection(data_keys):
                return "failed", ["json_error_envelope"]
            if value.get("success") is False or value.get("ok") is False:
                return "failed", ["json_unsuccessful_envelope"]
            root_status = str(value.get("status") or "").strip().lower()
            if root_status in {
                "denied",
                "error",
                "failed",
                "failure",
                "forbidden",
                "unauthenticated",
                "unauthorized",
            }:
                return "failed", ["json_error_status"]
            root_message = " ".join(
                str(value.get(name) or "") for name in ("message", "detail")
            ).strip().lower()
            if not keys.intersection(data_keys) and any(
                marker in root_message
                for marker in (
                    "access denied",
                    "authentication required",
                    "forbidden",
                    "invalid api key",
                    "invalid token",
                    "login required",
                    "not authenticated",
                    "unauthorized",
                )
            ):
                return "failed", ["json_auth_or_login_envelope"]
            metadata_keys = {
                "authenticated",
                "code",
                "detail",
                "health",
                "healthy",
                "message",
                "ok",
                "service",
                "status",
                "success",
                "timestamp",
                "uptime",
                "version",
            }
            if keys and keys.issubset(metadata_keys):
                return "failed", ["json_non_data_envelope"]
            present_data = [
                item for name, item in value.items() if str(name).lower() in data_keys
            ]
            if present_data and all(item in (None, "", [], {}) for item in present_data):
                payload_keys = keys - data_keys - {
                    "count",
                    "limit",
                    "offset",
                    "page",
                    "page_size",
                    "total",
                }
                if not payload_keys or payload_keys.issubset(metadata_keys):
                    return "failed", ["empty_json_data"]
        else:
            if all(item in (None, "", [], {}) for item in value):
                return "failed", ["empty_json_data"]
            list_messages = " ".join(
                str(item).lower()
                for item in value[:20]
                if isinstance(item, str)
            )
            list_objects = [item for item in value[:20] if isinstance(item, dict)]
            if list_messages and any(
                marker in list_messages
                for marker in (
                    "access denied",
                    "authentication required",
                    "forbidden",
                    "invalid api key",
                    "invalid token",
                    "login required",
                    "unauthorized",
                )
            ):
                return "failed", ["json_auth_or_login_envelope"]
            if list_objects and all(isinstance(item, dict) for item in value):
                error_keys = {"error", "errors", "exception", "fault"}
                if all(
                    {str(key).lower() for key in item}.intersection(error_keys)
                    or str(item.get("status") or "").lower()
                    in {"denied", "error", "failed", "failure", "forbidden"}
                    or any(
                        marker
                        in " ".join(
                            str(item.get(name) or "").lower()
                            for name in ("message", "detail")
                        )
                        for marker in (
                            "access denied",
                            "authentication required",
                            "forbidden",
                            "invalid api key",
                            "invalid token",
                            "login required",
                            "unauthorized",
                        )
                    )
                    for item in list_objects
                ):
                    return "failed", ["json_error_envelope"]
        return "passed", ["bounded_json_data_response"]

    @staticmethod
    def _is_non_data_probe_path(path: str) -> bool:
        normalized = str(path or "").split("?", 1)[0].strip().lower().rstrip("/")
        if not normalized:
            return True
        terminal = normalized.rsplit("/", 1)[-1]
        return terminal in _NON_DATA_PROBE_TERMINALS

    @staticmethod
    def _validate_json_shape(
        value: Any, *, depth: int, counters: Dict[str, int]
    ) -> None:
        if depth > _MAX_JSON_DEPTH:
            raise CoordinatorSourceAddV2Error("json_depth_exceeded")
        counters["nodes"] += 1
        if counters["nodes"] > _MAX_JSON_NODES:
            raise CoordinatorSourceAddV2Error("json_node_limit_exceeded")
        if isinstance(value, dict):
            counters["keys"] += len(value)
            if counters["keys"] > _MAX_JSON_KEYS:
                raise CoordinatorSourceAddV2Error("json_key_limit_exceeded")
            for key, item in value.items():
                if not isinstance(key, str) or len(key) > 500:
                    raise CoordinatorSourceAddV2Error("json_key_invalid")
                CoordinatorSourceAddFunctionalProbeV2._validate_json_shape(
                    item, depth=depth + 1, counters=counters
                )
        elif isinstance(value, list):
            for item in value:
                CoordinatorSourceAddFunctionalProbeV2._validate_json_shape(
                    item, depth=depth + 1, counters=counters
                )
        elif isinstance(value, str) and len(value) > _MAX_JSON_STRING:
            raise CoordinatorSourceAddV2Error("json_string_limit_exceeded")
        elif not isinstance(value, (str, int, float, bool, type(None))):
            raise CoordinatorSourceAddV2Error("json_value_invalid")

    @staticmethod
    def _summary(
        *,
        index: int,
        result_status: str,
        response_hash: str,
        status_class: str,
        content_type: str,
        byte_count: int,
        duration_ms: int,
        reason_codes: list[str],
        retry_after_seconds: int = 0,
    ) -> Dict[str, Any]:
        return {
            "probe_index": index,
            "result_status": result_status,
            "response_hash": response_hash,
            "status_class": status_class[:40],
            "content_type": content_type[:160],
            "byte_count": int(byte_count),
            "duration_ms": int(duration_ms),
            "retry_after_seconds": min(21_600, max(0, int(retry_after_seconds))),
            "reason_codes": sorted({str(item)[:80] for item in reason_codes}),
        }

    def _read(
        self,
        policy_id: str,
        parameters: Mapping[str, Any],
        context: ExecutionContextV2,
    ) -> list[Dict[str, Any]]:
        return self._reader.read(
            policy_id=policy_id,
            parameters=parameters,
            job_id=context.job_id,
            purpose=context.purpose,
            record_transport=context.record_transport,
            record_artifact=context.record_artifact,
        )

"""Measured SOURCE_ADD provenance authority with authenticated provider I/O."""

from __future__ import annotations

import base64
import json
from typing import Any, Callable, Dict, Mapping
from urllib.parse import urlencode

from gateway.research_lab.source_add_provenance import (
    SourceAddProvenanceResult,
    evaluate_source_add_provenance,
)
from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


SOURCE_ADD_PROVENANCE_REQUEST_SCHEMA_VERSION = (
    "leadpoet.source_add_provenance_request.v2"
)
SOURCE_ADD_PROVENANCE_RESULT_SCHEMA_VERSION = (
    "leadpoet.source_add_provenance_result.v2"
)
OP_SOURCE_ADD_PROVENANCE_V2 = "source_add_provenance_v2"
SCRAPINGDOG_ORIGIN = "https://api.scrapingdog.com"
_MAX_PROVIDER_BODY = 240_000


class CoordinatorSourceAddV2Error(RuntimeError):
    """A SOURCE_ADD provenance request or authenticated response is invalid."""


class CoordinatorSourceAddProvenanceV2:
    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hash: str,
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hash = str(retry_policy_hash or "")
        if not self._retry_policy_hash:
            raise CoordinatorSourceAddV2Error(
                "ScrapingDog retry policy is unavailable"
            )

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
        if path not in {"/scrape", "/google/ai_mode"}:
            raise CoordinatorSourceAddV2Error(
                "SOURCE_ADD provider path is not measured"
            )
        logical_suffix = "documentation" if path == "/scrape" else "ai-mode"
        result = dict(
            self._execute_provider(
                {
                    "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                    "logical_operation_id": "%s:source-add:%s"
                    % (context.job_id, logical_suffix),
                    "job_id": context.job_id,
                    "purpose": context.purpose,
                    "provider_id": "scrapingdog",
                    "attempt_number": 0,
                    "method": "GET",
                    "url": SCRAPINGDOG_ORIGIN + path + "?" + urlencode(dict(params)),
                    "headers": {
                        "accept": "application/json,text/html;q=0.9,*/*;q=0.8",
                        "user-agent": "leadpoet-source-add-precheck/1.0",
                    },
                    "body_b64": "",
                    "timeout_ms": int(timeout_seconds) * 1000,
                    "retry_policy_hash": self._retry_policy_hash,
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

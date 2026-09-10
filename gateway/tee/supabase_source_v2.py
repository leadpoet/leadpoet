"""Measured PostgREST reads for qualification admission.

The coordinator owns every query shape. Callers may supply only the small,
typed values named by a policy; they cannot select another table, project,
column set, ordering, page size, retry policy, or timeout.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import time
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlencode
from uuid import UUID

from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_SUPABASE_ORIGIN,
    configured_supabase_origin_v2,
)
from gateway.tee.provider_broker_v2 import PROVIDER_BROKER_SCHEMA_VERSION
from leadpoet_canonical.attested_v2 import sha256_bytes, sha256_json


SUPABASE_SOURCE_ORIGIN = PRODUCTION_SUPABASE_ORIGIN
SUPABASE_SOURCE_SCHEMA_VERSION = "leadpoet.supabase_source.v2"
SUPABASE_READ_TIMEOUT_MS = 45_000
SUPABASE_PAGE_SIZE = 1_000
SUPABASE_RETRY_BACKOFF_SECONDS = (1.0, 3.0)


class SupabaseSourceV2Error(RuntimeError):
    """A measured database read did not end in an authenticated valid page."""


@dataclass(frozen=True)
class SupabaseQueryV2:
    policy_id: str
    table: str
    select: str
    parameter_names: Tuple[str, ...]
    max_pages: int
    page_size: int = SUPABASE_PAGE_SIZE
    order: str = ""
    limit: int = 0


QUERY_POLICIES = {
    'qualification_epoch_assignment': SupabaseQueryV2(
        policy_id="qualification_epoch_assignment",
        table="transparency_log",
        select="payload",
        parameter_names=("epoch_id",),
        max_pages=1,
        order="ts.desc",
        limit=1,
    ),
    'qualification_leads_by_ids': SupabaseQueryV2(
        policy_id="qualification_leads_by_ids",
        table="leads_private",
        select=(
            "lead_id,lead_blob,lead_blob_hash,miner_hotkey,"
            "first_name,last_name,email,role,company_name,linkedin,website,"
            "company_linkedin,industry,sub_industry,city,state,country,"
            "hq_city,hq_state,hq_country,employee_count,description"
        ),
        parameter_names=("lead_ids",),
        max_pages=1,
    ),
    'banned_hotkeys': SupabaseQueryV2(
        policy_id="banned_hotkeys",
        table="banned_hotkeys",
        select="hotkey",
        parameter_names=(),
        max_pages=20,
        order="hotkey.asc",
    )
}


def _non_negative_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SupabaseSourceV2Error("%s must be an integer" % field)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise SupabaseSourceV2Error("%s must be an integer" % field) from exc
    if result < 0:
        raise SupabaseSourceV2Error("%s must be non-negative" % field)
    return result


def _filters(policy: SupabaseQueryV2, parameters: Mapping[str, Any]) -> Sequence[Tuple[str, str]]:
    if not isinstance(parameters, Mapping) or set(parameters) != set(
        policy.parameter_names
    ):
        raise SupabaseSourceV2Error("Supabase policy parameters are invalid")
    if policy.policy_id == "qualification_epoch_assignment":
        epoch_id = _non_negative_int(parameters["epoch_id"], "epoch_id")
        return (
            ("event_type", "eq.EPOCH_INITIALIZATION"),
            ("payload->>epoch_id", "eq.%d" % epoch_id),
        )
    if policy.policy_id == "qualification_leads_by_ids":
        values = parameters["lead_ids"]
        if (
            not isinstance(values, (list, tuple))
            or not values
            or len(values) > 200
        ):
            raise SupabaseSourceV2Error("lead_ids must contain 1-200 UUIDs")
        normalized = []
        for value in values:
            try:
                normalized.append(str(UUID(str(value))))
            except (TypeError, ValueError, AttributeError) as exc:
                raise SupabaseSourceV2Error("lead_id is not a UUID") from exc
        if len(normalized) != len(set(normalized)):
            raise SupabaseSourceV2Error("lead_ids are duplicated")
        return (("lead_id", "in.(%s)" % ",".join(normalized)),)
    if policy.policy_id == "banned_hotkeys":
        return ()
    raise SupabaseSourceV2Error("Supabase query policy is unsupported")


class SupabaseSourceReaderV2:
    def __init__(
        self,
        *,
        execute_provider: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        retry_policy_hash: str,
        origin: Optional[str] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._execute_provider = execute_provider
        self._retry_policy_hash = str(retry_policy_hash or "")
        self._origin = str(origin or configured_supabase_origin_v2())
        self._sleep = sleep

    def read(
        self,
        *,
        policy_id: str,
        parameters: Mapping[str, Any],
        job_id: str,
        purpose: str,
        record_transport: Callable[[Mapping[str, Any]], None],
        record_artifact: Callable[[str], None],
    ) -> list[Dict[str, Any]]:
        policy = QUERY_POLICIES.get(str(policy_id or ""))
        if policy is None:
            raise SupabaseSourceV2Error("Supabase query policy is not measured")
        if not 1 <= policy.page_size <= SUPABASE_PAGE_SIZE:
            raise SupabaseSourceV2Error("Supabase policy page size is invalid")
        filters = _filters(policy, parameters)
        rows = []
        for page_index in range(policy.max_pages):
            page = self._read_page(
                policy=policy,
                filters=filters,
                page_index=page_index,
                job_id=job_id,
                purpose=purpose,
                record_transport=record_transport,
                record_artifact=record_artifact,
            )
            rows.extend(page)
            if policy.limit or len(page) < policy.page_size:
                break
        else:
            raise SupabaseSourceV2Error("Supabase query exceeded its measured page limit")
        return rows

    def _read_page(
        self,
        *,
        policy: SupabaseQueryV2,
        filters: Sequence[Tuple[str, str]],
        page_index: int,
        job_id: str,
        purpose: str,
        record_transport: Callable[[Mapping[str, Any]], None],
        record_artifact: Callable[[str], None],
    ) -> list[Dict[str, Any]]:
        query = [("select", policy.select), *filters]
        if policy.order:
            query.append(("order", policy.order))
        if policy.limit:
            query.append(("limit", str(policy.limit)))
        url = "%s/rest/v1/%s?%s" % (
            self._origin,
            policy.table,
            urlencode(query),
        )
        start = page_index * policy.page_size
        end = start + policy.page_size - 1
        query_scope_hash = sha256_json(
            {
                "schema_version": "leadpoet.supabase_query_scope.v2",
                "policy_id": policy.policy_id,
                "filters": [list(item) for item in filters],
                "page_size": policy.page_size,
            }
        )
        logical_operation_id = "%s:%s:%s:page-%d" % (
            job_id,
            policy.policy_id,
            query_scope_hash,
            page_index,
        )
        last_error = "unavailable"
        for attempt_number in range(len(SUPABASE_RETRY_BACKOFF_SECONDS) + 1):
            result = dict(
                self._execute_provider(
                    {
                        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
                        "logical_operation_id": logical_operation_id,
                        "job_id": job_id,
                        "purpose": purpose,
                        "provider_id": "supabase",
                        "attempt_number": attempt_number,
                        "method": "GET",
                        "url": url,
                        "headers": {
                            "accept": "application/json",
                            "range": "%d-%d" % (start, end),
                            "range-unit": "items",
                        },
                        "body_b64": base64.b64encode(b"").decode("ascii"),
                        "timeout_ms": SUPABASE_READ_TIMEOUT_MS,
                        "retry_policy_hash": self._retry_policy_hash,
                    }
                )
            )
            attempt = result.get("transport_attempt")
            if not isinstance(attempt, Mapping):
                raise SupabaseSourceV2Error("Supabase terminal attempt is missing")
            record_transport(attempt)
            record_artifact(str(attempt["request_artifact_hash"]))
            if attempt.get("terminal_status") == "authenticated_response":
                record_artifact(str(attempt["response_artifact_hash"]))
            if (
                result.get("terminal_status") == "authenticated_response"
                and 200 <= int(result.get("http_status") or 0) < 300
            ):
                try:
                    body = base64.b64decode(
                        str(result.get("body_b64") or ""), validate=True
                    )
                    parsed = json.loads(body.decode("utf-8"))
                except Exception as exc:
                    last_error = "malformed_json"
                else:
                    if not isinstance(parsed, list) or any(
                        not isinstance(item, Mapping) for item in parsed
                    ):
                        last_error = "response_not_row_array"
                    else:
                        if sha256_bytes(body) != attempt.get("response_hash"):
                            raise SupabaseSourceV2Error(
                                "Supabase response body hash differs from terminal record"
                            )
                        return [dict(item) for item in parsed]
            else:
                last_error = str(
                    result.get("failure_code")
                    or "http_%s" % result.get("http_status")
                )
            if attempt_number < len(SUPABASE_RETRY_BACKOFF_SECONDS):
                self._sleep(SUPABASE_RETRY_BACKOFF_SECONDS[attempt_number])
        raise SupabaseSourceV2Error(
            "Supabase %s page %d failed: %s"
            % (policy.policy_id, page_index, last_error)
        )

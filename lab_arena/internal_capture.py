"""Private observations of trusted Arena host calls, using existing object storage.

A request event is a plan, not proof of dispatch. HTTP response and committed
business outcomes remain separate. These records confer no miner eligibility,
accounting, admission, or training authority.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from lab_arena import contracts, operations, reasoning_capture, training_capture, training_export, training_steps, training_thinking
from lab_arena.runtime_version import SOURCE_METADATA

SCHEMA_VERSION = "leadpoet.lab_arena.internal_capture_event.v1"
EXAMPLE_SCHEMA_VERSION = "leadpoet.lab_arena.internal_training_example.v1"
SFT_SCHEMA_VERSION = "leadpoet.lab_arena.internal_sft_candidate.v1"
OBJECT_PREFIX = "private/arena-training/internal/"
MAX_EVENT_BYTES = 64 * 1024
MAX_OUTCOME_BYTES = 512 * 1024
MAX_EXPORT_EVENTS = 30_000
STAGES = ("request", "response", "outcome")
_REF = re.compile(re.escape(OBJECT_PREFIX) + r"(\d{4}-\d{2}-\d{2})/([0-9a-f]{32})/(request|response|outcome)\.json\Z")
_PURPOSE = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
_LOGGER = logging.getLogger(__name__)
_LOG_PURPOSES = frozenset({"admission_review", "icp_generation", "icp_exclusion", "judge"})


def _json(value: Any) -> bytes:
    return contracts.canonical_json(value).encode("utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_hash() -> str | None:
    try:
        return contracts.hash_bytes(Path(__file__).read_bytes())
    except OSError:
        return None


CAPTURE_SOURCE_SHA256 = _source_hash()


def from_environment() -> "InternalCapture":
    """Return an optional bounded sink, without constructing a service or worker."""
    try:
        import boto3
        from botocore.config import Config
        from lab_arena.service import S3ObjectStore

        bucket = os.environ["LAB_ARENA_BUCKET"].strip()
        if not bucket:
            raise ValueError("bucket_unavailable")
        client = boto3.client("s3", region_name=os.environ.get("AWS_REGION"), config=Config(
            signature_version="s3v4", connect_timeout=0.5, read_timeout=0.5,
            retries={"total_max_attempts": 1, "mode": "standard"}))
        objects = S3ObjectStore(bucket, client=client, prefix=os.environ.get("LAB_ARENA_OBJECT_PREFIX", ""))
    except Exception:
        objects = None
    return InternalCapture(objects)


class InternalCapture:
    """Best-effort capture. All persistence uses the existing write-once stores."""

    def __init__(self, objects: Any) -> None:
        self.objects = objects

    @classmethod
    def from_environment(cls) -> "InternalCapture":
        return from_environment()

    def start(self, *, purpose: str, context: Mapping[str, Any], request: Mapping[str, Any],
              api: str = "openrouter.chat", secrets: Iterable[str] = ()) -> "InternalCall":
        call = InternalCall(self.objects)
        try:
            call._secrets = tuple(value for value in secrets if isinstance(value, str) and value)
            if not isinstance(purpose, str) or not _PURPOSE.fullmatch(purpose):
                raise ValueError("purpose_invalid")
            if api not in reasoning_capture.OPERATIONS or not isinstance(request, Mapping):
                raise ValueError("request_invalid")
            call._common.update(purpose=purpose, api=api,
                                model=request.get("model") if isinstance(request.get("model"), str) else None)
            # Store the body once. Exporters can pass this same verified value
            # to shared parsers without persisting redundant prompt copies.
            document = {"operation_id": api, "outbound": {"body": dict(request)}}
            cleaned, redactions = training_capture.sanitize_training_value(dict(context), call._secrets)
            if len(_json(cleaned)) > MAX_EVENT_BYTES // 2:
                raise ValueError("context_size_limit")
            call._common.update(context=cleaned,
                                context_privacy={"redacted": bool(redactions), "redactions": redactions})
            call.request_pointer = training_capture.RequestCapture(self.objects)(document, secrets=call._secrets)
            call._event("request", {"capture": call.request_pointer, "dispatch": "not_established"})
        except Exception:
            # Do not log exception text, endpoints, caller data, or secrets.
            call.request_pointer = training_capture.unavailable("internal_request_capture_failed")
            call._event("request", {"capture": call.request_pointer, "dispatch": "not_established"})
        return call


class InternalCall:
    def __init__(self, objects: Any) -> None:
        self.call_id = uuid.uuid4().hex
        self.created_at = _now()
        self.prefix = OBJECT_PREFIX + self.created_at[:10] + "/" + self.call_id + "/"
        self.request_pointer: dict[str, Any] = training_capture.unavailable()
        self.response_pointer: dict[str, Any] = training_capture.unavailable("response_not_observed")
        self._objects = objects
        self._secrets: tuple[str, ...] = ()
        self._lock = threading.RLock()
        self._stages: dict[str, dict[str, Any]] = {}
        self._common = {"schema_version": SCHEMA_VERSION, "actor_type": "arena_internal",
                        "call_id": self.call_id, "created_at": self.created_at,
                        "host_source": {key.removeprefix("validator_"): value
                                        for key, value in SOURCE_METADATA.items()},
                        "capture_source_sha256": CAPTURE_SOURCE_SHA256,
                        "purpose": "unknown", "api": None, "context": {},
                        "model": None, "context_privacy": {"redacted": False, "redactions": []}}

    def _event(self, stage: str, fields: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if stage in self._stages:
                return self._stages[stage]
            result = {"status": "unavailable", "reason": "internal_index_write_failed"}
            try:
                document = {**self._common, "stage": stage, "recorded_at": _now(), **fields}
                document, redactions = training_capture.sanitize_training_value(document, self._secrets)
                document["event_privacy"] = {"redacted": bool(redactions), "redactions": redactions}
                document["event_sha256"] = contracts.document_hash(document)
                payload = _json(document)
                if len(payload) > MAX_EVENT_BYTES:
                    raise ValueError("event_size_limit")
                ref = self.prefix + stage + ".json"
                self._objects.put(ref, payload)
                result = {"status": "stored", "ref": ref, "sha256": document["event_sha256"]}
            except Exception:
                # A total object-store outage has no discoverable index. Keep
                # one bounded diagnostic in existing app logs, without source
                # data, object refs, provider errors, or exception text.
                try:
                    purpose = self._common.get("purpose")
                    call_id = self.call_id if re.fullmatch(r"[0-9a-f]{32}", self.call_id) else "unknown"
                    _LOGGER.warning(
                        "arena_internal_capture_write_failed stage=%s purpose=%s call_id=%s",
                        stage if stage in STAGES else "unknown",
                        purpose if purpose in _LOG_PURPOSES else "unknown", call_id)
                except Exception:
                    pass  # Diagnostics must also remain observational.
            self._stages[stage] = result
            return result

    def response(self, *, body: bytes, status: int,
                 elapsed_seconds: float | None = None) -> dict[str, Any]:
        with self._lock:
            if "response" in self._stages or "outcome" in self._stages:
                return self._stages.get("response", {"status": "unavailable", "reason": "call_finished"})
            fields: dict[str, Any] = {"http_status": status if type(status) is int and 100 <= status <= 599 else None,
                                      "request_event": self._stages.get("request"),
                                      "response_observed": True}
            try:
                if not isinstance(body, bytes) or len(body) > training_capture.MAX_OBJECT_BYTES:
                    raise ValueError("response_size_or_type")
                fields.update(wire_body_sha256=contracts.hash_bytes(body), wire_body_bytes=len(body))
                if (type(elapsed_seconds) in (int, float) and math.isfinite(elapsed_seconds)
                        and elapsed_seconds >= 0):
                    fields["elapsed_seconds"] = elapsed_seconds
                # Every supported internal API is a JSON model endpoint.
                # Do not turn malformed structured data into untyped text:
                # that would bypass opaque/credential field redaction.
                provider_response = training_capture.model_response_object(body.decode("utf-8"))
                document = {"provider_calls": [{"response": provider_response}]}
                fields["body_format"] = "json"
                self.response_pointer = training_capture.RequestCapture(self._objects)(document, secrets=self._secrets)
                fields["capture"] = self.response_pointer
            except Exception:
                self.response_pointer = training_capture.unavailable("internal_response_capture_failed")
                fields["capture"] = self.response_pointer
            return self._event("response", fields)

    def finish(self, *, outcome: Mapping[str, Any], cost_microusd: int | None = None) -> dict[str, Any]:
        with self._lock:
            if "outcome" in self._stages:
                return self._stages["outcome"]
            pointer = training_capture.unavailable("internal_outcome_capture_failed")
            try:
                if not isinstance(outcome, Mapping) or len(_json(outcome)) > MAX_OUTCOME_BYTES:
                    raise ValueError("outcome_size_or_type")
                pointer = training_capture.RequestCapture(self._objects)({"outcome": dict(outcome)}, secrets=self._secrets)
            except Exception:
                pass
            known_cost = type(cost_microusd) is int and cost_microusd >= 0
            result = self._event("outcome", {
                "capture": pointer, "request_event": self._stages.get("request"),
                "response_event": self._stages.get("response"),
                "cost_microusd": cost_microusd if known_cost else None,
                "cost_status": "known" if known_cost else "unknown",
                "outcome_authority": "observed_host_result_not_verified_ground_truth"})
            self._secrets = ()
            return result


def call_prefix(ref: str) -> str:
    """Restrict read access to explicit internal event references."""
    match = _REF.fullmatch(ref) if isinstance(ref, str) else None
    if match is None:
        raise ValueError("internal_event_ref_invalid")
    date.fromisoformat(match[1])
    return ref.rsplit("/", 1)[0] + "/"


def _read_event(objects: Any, ref: str) -> dict[str, Any] | None:
    prefix = call_prefix(ref)
    reader = getattr(objects, "get_optional_bounded", None)
    raw = reader(ref, MAX_EVENT_BYTES) if callable(reader) else objects.get_bounded(ref, MAX_EVENT_BYTES)
    if raw is None:
        return None
    event = json.loads(raw)
    if not isinstance(event, dict):
        raise ValueError("event_invalid")
    digest = event.get("event_sha256")
    if (event.get("schema_version") != SCHEMA_VERSION or event.get("actor_type") != "arena_internal"
            or event.get("call_id") != prefix.split("/")[-2]
            or event.get("created_at", "")[:10] != prefix.split("/")[-3]
            or event.get("stage") + ".json" != ref.rsplit("/", 1)[1]
            or digest != contracts.document_hash({key: value for key, value in event.items() if key != "event_sha256"})):
        raise ValueError("event_identity_or_hash_invalid")
    # Index metadata has no provider reasoning path. Its sanitizer must be a
    # fixed point before private output, including manually supplied local refs.
    if training_capture.sanitize_training_value(event)[0] != event:
        raise ValueError("event_privacy_invalid")
    return event


def _reasoning(request: Mapping[str, Any], response: Any, api: str, redactions: list[str], registry: Any) -> dict:
    call = {"provider": "openrouter", "operation_id": api, "response": response,
            "request": {"outbound": {"body": request}, "admitted_parameters": request,
                        "hashed_parameters": request}, "response_privacy_redactions": redactions}
    try:
        checked, annotation = reasoning_capture.apply(api, request, policy=reasoning_capture.POLICY, registry=registry)
        # The exporter cannot add controls that the saved request did not send.
        if checked == request:
            call["reasoning_capture"] = annotation
            call["route_observation"] = reasoning_capture.route_observation(response, annotation)
    except (TypeError, ValueError, KeyError, AttributeError):
        pass
    segments, summary, opaque = training_thinking._segments(response, api)
    full, status = training_thinking.full_text_segments(call, "arena_internal", registry)
    return {"available_segments": segments, "contains_summary": summary,
            "contains_opaque": opaque or "hidden_reasoning" in redactions,
            "full_generated_thinking": full, "status": status,
            "registry_sha256": contracts.document_hash(registry) if isinstance(registry, Mapping) else None}


def _internal_responses_input(value, *, tools=(), max_tool_calls=None,
                              allow_reasoning_history=False, allow_tool_error_history=False) -> None:
    """Check trusted host initial messages without miner dispatch restrictions.

    This validates saved evidence only. It does not alter a request, reserve
    money, dispatch tools, or change the shared miner input validator.
    """
    require = training_steps._require
    if isinstance(value, str):
        require(bool(value.strip()), "internal_input_empty")
    else:
        require(isinstance(value, list) and 1 <= len(value) <= 256, "internal_input_shape_unsupported")
        for message in value:
            require(isinstance(message, Mapping) and set(message) <= {"id", "type", "role", "content"}
                    and message.get("type", "message") == "message"
                    and message.get("role") in {"system", "developer", "user"}, "internal_initial_message_unsupported")
            if "id" in message:
                require(isinstance(message["id"], str) and 1 <= len(message["id"]) <= 256,
                        "internal_message_id_invalid")
            content = message.get("content")
            require(isinstance(content, str) or isinstance(content, list) and bool(content),
                    "internal_initial_content_unsupported")
            if isinstance(content, list):
                require(all(isinstance(part, Mapping) and set(part) == {"type", "text"}
                            and part["type"] == "input_text" and isinstance(part["text"], str)
                            for part in content), "internal_initial_content_unsupported")
    _internal_search_tools(tools, max_tool_calls)


def _internal_search_tools(tools, max_tool_calls, *, allow_chat_fast_mode=False):
    """Check retained internal controls; this grants no dispatch or budget authority."""
    require = training_steps._require
    require(isinstance(tools, (list, tuple)) and len(tools) <= 1, "internal_tools_unsupported")
    if tools:
        tool = tools[0]
        require(isinstance(tool, Mapping) and set(tool) <= {"type", "parameters"}
                and tool.get("type") == "openrouter:web_search", "internal_tools_unsupported")
        parameters = tool.get("parameters", {})
        fields = {"engine", "max_uses", "max_results", "max_total_results", "max_characters"}
        if allow_chat_fast_mode:
            fields.add("mode")
        require(isinstance(parameters, Mapping) and set(parameters) <= fields,
                "internal_search_controls_unsupported")
        if "mode" in parameters:
            require(parameters["mode"] == "fast", "internal_search_mode_unsupported")
        if "engine" in parameters:
            require(parameters["engine"] in {"exa", "native"}, "internal_search_engine_invalid")
        for name, number in parameters.items():
            if name not in {"engine", "mode"}:
                require(type(number) is int and 1 <= number <= 1_000_000,
                        "internal_search_limit_invalid")
        require(type(max_tool_calls) is int and 1 <= max_tool_calls <= 1000,
                "internal_tool_limit_invalid")
    else:
        require(max_tool_calls is None, "internal_tool_limit_without_tools")


def _internal_chat_annotations(body, message):
    """Validate exact nested Chat citations without rewriting the saved target."""
    require = training_steps._require
    tools = body.get("tools", [])
    _internal_search_tools(tools, body.get("max_tool_calls"), allow_chat_fast_mode=True)
    require(bool(tools) and not message.get("tool_calls"), "internal_chat_search_undeclared")
    training_steps._chat_citation_annotations(body, message)


def _has_retained_search_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.replace("[REDACTED]", "").strip())


def _search_evidence(response: Any) -> dict:
    """Distinguish retained provider events, URL references and result text.

    Text counters describe retained content. A privacy placeholder does not
    establish that the provider never exposed the original text.
    """
    output = response.get("output") if isinstance(response, Mapping) else None
    events = []
    for index, item in enumerate(output if isinstance(output, list) else []):
        if not isinstance(item, Mapping) or item.get("type") not in training_steps._HOSTED_SEARCH_TYPES:
            continue
        action = item.get("action")
        action = action if isinstance(action, Mapping) else {}
        sources = action.get("sources")
        sources = sources if isinstance(sources, list) else []
        # Native Responses normally exposes URL-only sources. Do not promote
        # a citation or the model's reasoning into retrieved source content.
        results = action.get("results")
        results = results if isinstance(results, list) else []
        text_fields = ("text", "content", "snippet")
        text_results = [row for row in sources + results if isinstance(row, Mapping)
                        and any(_has_retained_search_text(row.get(key)) for key in text_fields)]
        text_status = "exposed" if text_results else "not_exposed"
        if not text_results and any(isinstance(row, Mapping) and any(
                isinstance(row.get(key), str) and "[REDACTED]" in row[key]
                for key in text_fields) for row in sources + results):
            text_status = "redacted"
        events.append({"output_index": index, "status": item.get("status"),
                       "source_references": len(sources), "results_with_text": len(text_results),
                       "result_text_status": text_status})
    citations = []
    choices = response.get("choices") if isinstance(response, Mapping) else None
    for choice_index, choice in enumerate(choices if isinstance(choices, list) else []):
        message = choice.get("message") if isinstance(choice, Mapping) else None
        annotations = message.get("annotations") if isinstance(message, Mapping) else None
        for index, annotation in enumerate(annotations if isinstance(annotations, list) else []):
            citation = annotation.get("url_citation") if isinstance(annotation, Mapping) and annotation.get("type") == "url_citation" else None
            if not isinstance(citation, Mapping) or not isinstance(citation.get("url"), str) or not citation["url"].strip():
                continue
            path = f"/choices/{choice_index}/message/annotations/{index}"
            content = citation.get("content")
            exposed = _has_retained_search_text(content)
            citations.append({"annotation_path": path, "url_path": path + "/url_citation/url",
                              "content_path": path + "/url_citation/content" if exposed else None,
                              "content_characters": len(content) if exposed else 0})
    usage = response.get("usage") if isinstance(response, Mapping) else None
    details = usage.get("server_tool_use_details") if isinstance(usage, Mapping) else None
    requests = details.get("web_search_requests") if isinstance(details, Mapping) else None
    requests = requests if type(requests) is int and requests >= 0 else None
    return {"scope": "provider_exposed_search_events_only", "search_events": len(events),
            "events_with_source_references": sum(row["source_references"] > 0 for row in events),
            "source_references": sum(row["source_references"] for row in events),
            "events_with_result_text": sum(row["results_with_text"] > 0 for row in events),
            "all_search_events_have_result_text": bool(events) and all(row["results_with_text"] for row in events),
            "underlying_page_contents_verified": False, "all_internal_provider_turns_verified": False,
            "events": events,
            "chat_url_citations": {"scope": "provider_exposed_chat_annotations_only",
                "references": len(citations),
                "annotations_with_content": sum(row["content_path"] is not None for row in citations),
                "content_characters": sum(row["content_characters"] for row in citations), "items": citations},
            "provider_reported_web_search_requests": {"scope": "provider_reported_aggregate_only",
                "count": requests, "path": "/usage/server_tool_use_details/web_search_requests" if requests is not None else None}}


def _opaque_response_projection_allowed(events, documents) -> bool:
    """Keep opaque omission separate from every other nonexact privacy edit."""
    if set(events) != set(STAGES) or set(documents) != set(STAGES):
        return False
    for stage, event in events.items():
        if any(event.get(field) != {"redacted": False, "redactions": []}
               for field in ("context_privacy", "event_privacy")):
            return False
        pointer = event.get("capture") or {}
        if (pointer.get("redacted") is not (stage == "response")
                or pointer.get("redactions") != (["hidden_reasoning"] if stage == "response" else [])):
            return False
    return True


def read_call(objects: Any, ref: str, *, registry: Any = None) -> tuple[dict, dict | None]:
    """Read one call; corrupt, missing, redacted and unavailable evidence is explicit."""
    prefix = call_prefix(ref)
    events, documents, gaps = {}, {}, []
    for stage in STAGES:
        try:
            event = _read_event(objects, prefix + stage + ".json")
        except Exception:
            event = None
            gaps.append(stage + "_event_unavailable_or_invalid")
        if event is None:
            gaps.append(stage + "_event_missing")
            continue
        events[stage] = event
        try:
            documents[stage] = training_capture.read_request(objects, event.get("capture") or {})
        except Exception:
            gaps.append(stage + "_body_unavailable_or_invalid")
    first = next(iter(events.values()), {})
    for event in events.values():
        if any(event.get(key) != first.get(key) for key in
               ("purpose", "context", "api", "created_at", "model", "capture_source_sha256", "host_source")):
            gaps.append("event_context_mismatch")
    request = ((documents.get("request") or {}).get("outbound") or {}).get("body")
    response_document = documents.get("response") or {}
    calls = response_document.get("provider_calls") or []
    response = calls[0].get("response") if len(calls) == 1 and isinstance(calls[0], Mapping) else None
    redactions = sorted({reason for event in events.values()
                         for field in (event.get("capture") or {}, event.get("context_privacy") or {},
                                       event.get("event_privacy") or {})
                         for reason in field.get("redactions") or []})
    if redactions:
        gaps.append("privacy_redactions")
    outcome_event = events.get("outcome") or {}
    if outcome_event.get("cost_status") != "known":
        gaps.append("cost_unknown")
    dimensions = {stage: stage in documents for stage in STAGES}
    dimensions.update(privacy=not redactions, costs=outcome_event.get("cost_status") == "known")
    reasoning = {"available_segments": [], "full_generated_thinking": [], "status": {"returned": "unavailable"}}
    if isinstance(request, Mapping):
        try:
            reasoning = _reasoning(request, response, first.get("api"), redactions, registry)
        except Exception:
            gaps.append("reasoning_parse_unavailable")
    raw = {"schema_version": EXAMPLE_SCHEMA_VERSION, "actor_type": "arena_internal",
           "call_id": prefix.split("/")[-2], "purpose": first.get("purpose"),
           "api": first.get("api"), "context": first.get("context", {}),
           "host_source": first.get("host_source", {}),
           "created_at": first.get("created_at"), "event_refs": [prefix + stage + ".json" for stage in STAGES],
           "events": events, "request": request, "response": response,
           **({"response_text": response_document["body_text"]} if "body_text" in response_document else {}),
           "outcome": (documents.get("outcome") or {}).get("outcome"),
           "cost_microusd": outcome_event.get("cost_microusd"),
           "cost_status": outcome_event.get("cost_status", "unknown"), "reasoning": reasoning,
           "research_evidence": _search_evidence(response),
           "completeness": {"complete": all(dimensions.values()) and not gaps,
                            "dimensions": dimensions, "gaps": sorted(set(gaps))}}
    opaque_projection = redactions == ["hidden_reasoning"] and _opaque_response_projection_allowed(events, documents)
    candidate, exclusions = None, []
    if not dimensions["request"] or not dimensions["response"] or "event_context_mismatch" in gaps:
        exclusions.append("request_response_unavailable")
    elif redactions and not opaque_projection:
        exclusions.append("privacy_nonexact")
    elif events.get("response", {}).get("http_status") != 200:
        exclusions.append("provider_http_error")
    else:
        try:
            if isinstance(response, Mapping) and any(response.get(key) for key in ("error", "errors", "refusal")):
                raise ValueError("internal_provider_error_or_refusal")
            target_response = training_export.without_hidden_reasoning(response)[0] if opaque_projection else response
            target, _ = training_steps._target(request, target_response, first["api"], allow_reasoning_history=True,
                                               allow_tool_error_history=True,
                                               responses_input_validator=_internal_responses_input,
                                               chat_annotations_validator=_internal_chat_annotations)
            if first["api"] == "openrouter.responses":
                training_steps._validate_target_tools(request, target, first["api"])
            elif target.get("tool_calls"):
                # Hosted search executes at the provider; it cannot declare
                # a model-authored function call. Preserve mixed requests and
                # validate only their callable function definitions here.
                function_tools = [tool for tool in request.get("tools", [])
                                  if tool.get("type") == "function"]
                training_steps._validate_target_tools(
                    {"tools": function_tools}, target, first["api"])
            candidate = {"schema_version": SFT_SCHEMA_VERSION, "actor_type": "arena_internal",
                         "purpose": first.get("purpose"), "call_id": raw["call_id"],
                         "format": first["api"], "input": request, "target": target,
                         "available_reasoning": reasoning["available_segments"],
                         # A projected response grants no new full-thinking target.
                         "generated_thinking": [] if opaque_projection else reasoning["full_generated_thinking"],
                         "quality": {"status": "unreviewed", "correct_action": "not_established"},
                         "training_permission": {"status": "unconfirmed"},
                         "source": {"event_refs": raw["event_refs"], "context": raw["context"],
                                    "request_sha256": (events["request"].get("capture") or {}).get("sha256"),
                                    "response_sha256": (events["response"].get("capture") or {}).get("sha256"),
                                    **({"response_projection": {"opaque_reasoning_omitted": True,
                                                                 "exact_provider_replay": False}}
                                       if opaque_projection else {}),
                                    "outcome": raw["outcome"], "cost_status": raw["cost_status"],
                                    "research_evidence": raw["research_evidence"],
                                    "capture_completeness": raw["completeness"]}}
        except Exception:
            exclusions.append("target_shape_or_completion_unsupported")
    raw["sft_exclusions"] = exclusions
    raw["example_sha256"] = contracts.document_hash(raw)
    return raw, candidate


def iter_exports(objects: Any, refs: Iterable[str], *, registry: Any = None):
    prefixes = set()
    for index, ref in enumerate(refs):
        if index >= MAX_EXPORT_EVENTS:
            raise ValueError("event_limit_exceeded")
        prefixes.add(call_prefix(ref))
    for prefix in sorted(prefixes):
        yield read_call(objects, prefix + "request.json", registry=registry)



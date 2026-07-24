"""Opt-in, infra-only OpenTelemetry for the gateway HOST process.

Scope is deliberately narrow and privacy-preserving:

- It emits ONE server span per HTTP request capturing only operational
  metadata: request method, the matched route *template* (never the concrete
  path, so ids/hotkeys/epochs do not leak), response status class, and
  duration. It never records request/response bodies, query strings, headers,
  database statements, or any LLM prompt/completion content.
- It does NOT auto-instrument the database, outbound HTTP clients, or the
  Research Lab / scoring / model / Langfuse code paths. Training data and model
  I/O are never touched.
- It runs only in the gateway host process (`python -m gateway.main`), never in
  the attested enclaves, and adds no new dependencies (it uses the OpenTelemetry
  packages already pinned in requirements.txt), so it cannot change any enclave
  measurement / PCR0.

The boundary is CODE-ENFORCED, not discipline-enforced:

1. The exporter destination is passed EXPLICITLY from private, namespaced
   environment variables (``GATEWAY_OTEL_ENDPOINT``, ``GATEWAY_OTEL_TOKEN``,
   ``GATEWAY_OTEL_SERVICE_NAME``). The standard ambient exporter/service
   variables are never read and never set, so auto-instrumentation or a
   stray bare exporter anywhere in the process has NO destination — it
   no-ops or errors instead of silently shipping data. The telemetry
   address lives only inside this module's exporter object. The provider
   resource is constructed as a FIXED attribute set (never via the SDK's
   env-merging factory), so ambient resource variables cannot leak either.
2. Every span passes through a FAIL-CLOSED schema validator before export.
   A span is exported only if it has the ``gateway.http`` instrumentation
   scope, EXACTLY the four approved attributes with the approved types,
   no events, no links, no status description, the fixed resource, and a
   route label that is either a registered route template or the literal
   ``/_unmatched``. Anything else is DROPPED (never mutated, never
   exported) and counted in a warning that does not include the rejected
   values.
3. ``tests/test_otel_boundary_guard.py`` fails CI if the boundary is crossed
   anywhere in the repo (auto-instrumentation packages, the
   process-wrapping launcher, a global tracer provider, ambient exporter
   or resource env vars, the env-merging resource factory in this module,
   or OTLP exporter imports outside this module).

It is a complete no-op unless BOTH ``GATEWAY_OTEL_ENABLED`` is truthy and
``GATEWAY_OTEL_ENDPOINT`` is set. Any failure while wiring it up is swallowed
so it can never delay or break gateway startup. No endpoint or token is ever
hard-coded or committed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, Optional

_TRUTHY = {"1", "true", "yes", "on"}

# Route templates whose spans are suppressed entirely — health/liveness noise.
_SUPPRESSED_ROUTES = {"/health", "/health/live", "/health/ready"}

# Route label used whenever the request did not resolve to a registered route
# template. A client-controlled path segment must NEVER be exported.
UNMATCHED_ROUTE_LABEL = "/_unmatched"

# The instrumentation scope every exported span must carry.
INSTRUMENTATION_SCOPE = "gateway.http"

# The ONLY attributes a span may carry out of this process, with their
# required types (bool is explicitly rejected for the numeric fields).
SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "duration_ms": (int, float),
}

SPAN_ATTRIBUTE_ALLOWLIST = frozenset(SPAN_ATTRIBUTE_TYPES)


def _enabled() -> bool:
    return (
        os.getenv("GATEWAY_OTEL_ENABLED", "").strip().lower() in _TRUTHY
        and bool(os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip())
    )


def _safe_route_label(request: Any) -> str:
    """Return the low-cardinality route template, never the concrete path.

    Using the matched route template (e.g. ``/research-lab/allocations/attested/
    {epoch}``) keeps ids, hotkeys, and epoch numbers out of telemetry. When the
    template cannot be resolved the label is the fixed ``/_unmatched`` literal —
    a client-controlled path segment is never exported.
    """
    try:
        route = request.scope.get("route")
        template = getattr(route, "path", None)
        if isinstance(template, str) and template:
            return template
    except Exception:
        pass
    return UNMATCHED_ROUTE_LABEL


def _span_scope_name(span: Any) -> str:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None:  # older SDK naming
        scope = getattr(span, "instrumentation_info", None)
    return str(getattr(scope, "name", "") or "")


def _attributes_conform(attributes: Dict[str, Any]) -> bool:
    if set(attributes) != SPAN_ATTRIBUTE_ALLOWLIST:
        return False
    for key, allowed_types in SPAN_ATTRIBUTE_TYPES.items():
        value = attributes[key]
        if isinstance(value, bool) or not isinstance(value, allowed_types):
            return False
    return True


def _validating_exporter(
    delegate: Any,
    *,
    allowed_routes: Callable[[], set],
    expected_resource: Dict[str, Any],
) -> Any:
    """Wrap a span exporter in a FAIL-CLOSED schema validator.

    A span is exported only when every check passes; a non-conforming span is
    dropped entirely — never mutated, never partially exported. Drops are
    logged as a per-reason count WITHOUT the rejected values.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    def _violation(span: Any) -> Optional[str]:
        try:
            if _span_scope_name(span) != INSTRUMENTATION_SCOPE:
                return "scope"
            if not _attributes_conform(dict(span.attributes or {})):
                return "attributes"
            if getattr(span, "events", None):
                return "events"
            if getattr(span, "links", None):
                return "links"
            status = getattr(span, "status", None)
            if status is not None and getattr(status, "description", None):
                return "status_description"
            resource = getattr(span, "resource", None)
            if dict(getattr(resource, "attributes", {}) or {}) != expected_resource:
                return "resource"
            route = dict(span.attributes or {}).get("http.route")
            if route != UNMATCHED_ROUTE_LABEL and route not in allowed_routes():
                return "route"
            return None
        except Exception:
            # Fail closed: a span we cannot fully validate is never exported.
            return "validation_error"

    class _FailClosedSpanExporter(SpanExporter):
        def export(self, spans):  # type: ignore[override]
            accepted = []
            dropped: Dict[str, int] = {}
            for span in spans:
                reason = _violation(span)
                if reason is None:
                    accepted.append(span)
                else:
                    dropped[reason] = dropped.get(reason, 0) + 1
            if dropped:
                try:
                    # Reasons and counts only — never the rejected values.
                    print(
                        "gateway_otel_span_dropped %s"
                        % " ".join("%s=%d" % kv for kv in sorted(dropped.items())),
                        flush=True,
                    )
                except Exception:
                    pass
            if not accepted:
                return SpanExportResult.SUCCESS
            return delegate.export(accepted)

        def shutdown(self) -> None:
            try:
                delegate.shutdown()
            except Exception:
                pass

        def force_flush(self, timeout_millis: int = 30_000) -> bool:
            try:
                return bool(delegate.force_flush(timeout_millis))
            except Exception:
                return True

    return _FailClosedSpanExporter()


def configure_gateway_otel(app: Any, *, span_exporter: Optional[Any] = None) -> bool:
    """Wire infra-only request spans onto the gateway app. No-op unless enabled.

    Returns True if instrumentation was installed, False otherwise.
    ``span_exporter`` is a test seam; production always builds the OTLP
    exporter with an EXPLICIT endpoint + headers from the private
    ``GATEWAY_OTEL_*`` variables (measure A) — ambient exporter variables
    are never read.
    """
    if span_exporter is None and not _enabled():
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.trace import SpanKind, StatusCode, Status

        service_name = (
            os.getenv("GATEWAY_OTEL_SERVICE_NAME", "").strip() or "leadpoet-gateway"
        )
        # FIXED resource: the plain constructor takes exactly these attributes.
        # The SDK's env-merging factory is deliberately avoided so ambient
        # resource variables can never leak into exported metadata.
        expected_resource = {"service.name": service_name}
        resource = Resource(expected_resource)

        def _registered_routes() -> set:
            try:
                return {
                    template
                    for template in (
                        getattr(route, "path", None)
                        for route in getattr(app, "routes", [])
                    )
                    if isinstance(template, str) and template
                }
            except Exception:
                return set()

        if span_exporter is not None:
            delegate = span_exporter
            make_processor = SimpleSpanProcessor
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip()
            token = os.getenv("GATEWAY_OTEL_TOKEN", "").strip()
            headers = {"Authorization": "Bearer " + token} if token else {}
            # Explicit arguments only: the destination exists solely inside
            # this exporter object, never in ambient environment variables.
            delegate = OTLPSpanExporter(endpoint=endpoint, headers=headers)
            make_processor = BatchSpanProcessor

        processor = make_processor(
            _validating_exporter(
                delegate,
                allowed_routes=_registered_routes,
                expected_resource=expected_resource,
            )
        )

        # A dedicated provider, NOT the global one, so nothing else (e.g. Langfuse)
        # is affected and no ambient instrumentation is picked up.
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(INSTRUMENTATION_SCOPE)

        def _emit(request, status_code, start_ns, start_mono):
            # Resolve the route template only after routing has run, so ids stay
            # out of the label. Suppression already handled static health paths.
            route = _safe_route_label(request)
            span = tracer.start_span(
                "%s %s" % (request.method, route),
                kind=SpanKind.SERVER,
                start_time=start_ns,
            )
            # Only method / route template / status / duration — nothing else.
            span.set_attribute("http.request.method", request.method)
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", int(status_code))
            span.set_attribute("duration_ms", (time.monotonic() - start_mono) * 1000.0)
            if int(status_code) >= 500:
                span.set_status(Status(StatusCode.ERROR))
            span.end()

        @app.middleware("http")
        async def _otel_request_span(request, call_next):
            # Suppress health/liveness noise up front (static paths, no params).
            if request.url.path in _SUPPRESSED_ROUTES:
                return await call_next(request)
            start_ns = time.time_ns()
            start_mono = time.monotonic()
            try:
                response = await call_next(request)
            except Exception:
                _emit(request, 500, start_ns, start_mono)
                raise
            _emit(request, response.status_code, start_ns, start_mono)
            return response

        return True
    except Exception as exc:  # never let telemetry break the gateway
        try:
            print("gateway_otel_bootstrap_skipped error=%s: %s"
                  % (type(exc).__name__, str(exc)[:200]), flush=True)
        except Exception:
            pass
        return False

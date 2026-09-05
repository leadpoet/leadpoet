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
   environment variables (``GATEWAY_OTEL_ENDPOINT``, ``GATEWAY_OTEL_TOKEN``).
   A non-empty token is REQUIRED (the pinned exporter falls back to ambient
   header variables when the explicit headers dict is empty), and
   initialization is REFUSED outright if any ambient standard exporter
   variable is present in the runtime environment — the exporter also
   consults those for TLS certificates, client keys, timeout, and
   compression, and a CI grep cannot see variables injected by a restart
   script or the live process environment. The service name is a CONSTANT,
   not an environment value. The provider resource is a fixed attribute set
   (never the SDK's env-merging factory), so ambient resource variables
   cannot leak either.
2. Every span passes a FAIL-CLOSED schema validator that checks the COMPLETE
   span envelope before export: the ``gateway.http`` instrumentation scope
   with no version/schema metadata, ``SERVER`` kind, a root span (no parent,
   empty trace state), EXACTLY the four approved attributes with the
   approved types — plus, on a >=500 response only, an OPTIONAL fifth
   ``gateway.stage`` label whose value must come from a fixed in-repo
   vocabulary so a generic error names the stage that refused it — a standard HTTP method, a span name equal to exactly
   ``<method> <route>``, no events, no links, no status description, the
   fixed resource, and a route label that is either a registered route
   template or the literal ``/_unmatched``. Anything else is DROPPED (never
   mutated, never exported) and counted in a warning that does not include
   the rejected values.
3. ``tests/test_otel_boundary_guard.py`` fails CI if the boundary is crossed
   anywhere in the repo (auto-instrumentation packages, the
   process-wrapping launcher, a global tracer provider, ambient exporter
   or resource env vars, the env-merging resource factory in this module,
   or OTLP exporter imports outside this module).

It is a complete no-op unless ``GATEWAY_OTEL_ENABLED`` is truthy AND
``GATEWAY_OTEL_ENDPOINT`` AND ``GATEWAY_OTEL_TOKEN`` are all set. Any failure
while wiring it up is swallowed so it can never delay or break gateway
startup. No endpoint or token is ever hard-coded or committed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional

from gateway.observability import stage_context

_TRUTHY = {"1", "true", "yes", "on"}

# Route templates whose spans are suppressed entirely — health/liveness noise.
_SUPPRESSED_ROUTES = {"/health", "/health/live", "/health/ready"}

# Route label used whenever the request did not resolve to a registered route
# template. A client-controlled path segment must NEVER be exported.
UNMATCHED_ROUTE_LABEL = "/_unmatched"

# The instrumentation scope every exported span must carry.
INSTRUMENTATION_SCOPE = "gateway.http"

# Fixed service identity — a constant, never an environment value.
SERVICE_NAME = "leadpoet-gateway"

# Fixed batching limits. Passing every value explicitly prevents ambient
# ``OTEL_BSP_*`` variables from changing gateway memory or shutdown behavior.
_BATCH_MAX_QUEUE_SIZE = 2048
_BATCH_SCHEDULE_DELAY_MILLIS = 5000
_BATCH_MAX_EXPORT_BATCH_SIZE = 512
_BATCH_EXPORT_TIMEOUT_MILLIS = 30000

# The standard exporter env-var prefix that must NOT exist at runtime. The
# pinned exporter consults these for headers, TLS certificates, client keys,
# timeout, and compression even when endpoint/headers are passed explicitly.
# (Split literal so the repo-wide CI grep for the ambient prefix stays
# meaningful everywhere else.)
_AMBIENT_EXPORTER_PREFIX = "OTEL_" + "EXPORTER_"

# Standard HTTP request methods; anything else is a client-controlled string.
_ALLOWED_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "DELETE", "CONNECT", "OPTIONS", "TRACE", "PATCH"}
)

# The ONLY attributes a span may carry out of this process, with their
# required types (bool is explicitly rejected for the numeric fields).
SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "http.request.method": (str,),
    "http.route": (str,),
    "http.response.status_code": (int,),
    "duration_ms": (int, float),
}

SPAN_ATTRIBUTE_ALLOWLIST = frozenset(SPAN_ATTRIBUTE_TYPES)

# One OPTIONAL attribute, attached only to a failed (>=500) request: the name of
# the internal stage the request last entered. Its value must come from the fixed
# vocabulary below, which is owned by this codebase and contains no identifiers,
# hotkeys, epochs, paths, or caller-supplied text. A stage name outside the
# vocabulary is a boundary violation and the span is dropped like any other.
STAGE_ATTRIBUTE = "gateway.stage"

STAGE_ATTRIBUTE_VALUES = frozenset(
    {
        "canonical_bundle_durable_lookup",
        "canonical_bundle_durable_persistence",
        "canonical_bundle_epoch_authority",
        "canonical_bundle_identity_authorization",
        "canonical_bundle_measured_publication",
        "canonical_bundle_publication_persistence",
        "canonical_bundle_shape_verification",
        "canonical_bundle_transparency_event",
        "compact_bundle_authority_persistence",
        "compact_bundle_authority_readback",
        "compact_bundle_authority_verification",
        "compact_bundle_cryptographic_verification",
        "compact_bundle_cutover_authority",
        "compact_bundle_durable_persistence",
        "compact_bundle_epoch_authority",
        "compact_bundle_epoch_evidence",
        "compact_bundle_measured_publication",
        "compact_bundle_publication_intent_lookup",
        "compact_bundle_publication_intent_persistence",
        "compact_bundle_shape_verification",
        "compact_bundle_transparency_event",
        "compact_bundle_validator_ancestry_persistence",
        "compact_finalization_ancestry_persistence",
        "compact_finalization_authority_persistence",
        "compact_finalization_authority_verification",
        "compact_finalization_cutover_authority",
        "compact_finalization_publication_readback",
        "compact_finalization_publication_verification",
        "compact_finalization_recovery_verification",
        "compact_finalization_shape_verification",
        "primary_finalization_persistence",
        "primary_finalization_receipt_graph_verification",
        "primary_finalization_shape_verification",
    }
)


def _enabled() -> bool:
    return (
        os.getenv("GATEWAY_OTEL_ENABLED", "").strip().lower() in _TRUTHY
        and bool(os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip())
    )


def _ambient_exporter_env_names() -> List[str]:
    return sorted(k for k in os.environ if k.startswith(_AMBIENT_EXPORTER_PREFIX))


def _safe_route_label(scope_or_request: Any) -> str:
    """Return the low-cardinality route template, never the concrete path.

    Using the matched route template (e.g. ``/research-lab/allocations/attested/
    {epoch}``) keeps ids, hotkeys, and epoch numbers out of telemetry. When the
    template cannot be resolved the label is the fixed ``/_unmatched`` literal —
    a client-controlled path segment is never exported.
    """
    try:
        scope = getattr(scope_or_request, "scope", scope_or_request)
        route = scope.get("route")
        template = getattr(route, "path", None)
        if isinstance(template, str) and template:
            return template
    except Exception:
        pass
    return UNMATCHED_ROUTE_LABEL


def _span_scope(span: Any) -> Any:
    scope = getattr(span, "instrumentation_scope", None)
    if scope is None:  # older SDK naming
        scope = getattr(span, "instrumentation_info", None)
    return scope


def _attributes_conform(attributes: Dict[str, Any]) -> bool:
    present = set(attributes)
    stage = attributes.get(STAGE_ATTRIBUTE)
    if present - {STAGE_ATTRIBUTE} != SPAN_ATTRIBUTE_ALLOWLIST:
        return False
    for key, allowed_types in SPAN_ATTRIBUTE_TYPES.items():
        value = attributes[key]
        if isinstance(value, bool) or not isinstance(value, allowed_types):
            return False
    if STAGE_ATTRIBUTE in present:
        # A stage label is only meaningful on a failure, and only from the
        # fixed vocabulary. Anything else fails closed.
        if stage not in STAGE_ATTRIBUTE_VALUES:
            return False
        status = attributes["http.response.status_code"]
        if not isinstance(status, int) or status < 500:
            return False
    return True


class _GatewayOtelMiddleware:
    """Raw-ASGI request telemetry that cannot affect request behavior."""

    def __init__(self, app: Any, *, emit: Callable[..., None]) -> None:
        self.app = app
        self._emit = emit

    def _emit_safely(
        self,
        scope: Dict[str, Any],
        status_code: int,
        start_ns: int,
        start_mono: float,
    ) -> None:
        try:
            self._emit(scope, status_code, start_ns, start_mono)
        except BaseException as exc:
            # Telemetry is observational only. Log the exception type without
            # values that could contain request or exporter data.
            try:
                print(
                    "gateway_otel_emit_failed error=%s"
                    % type(exc).__name__,
                    flush=True,
                )
            except BaseException:
                pass

    async def __call__(self, scope: Dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        if scope.get("path") in _SUPPRESSED_ROUTES:
            await self.app(scope, receive, send)
            return

        stage_context.reset_stage()
        start_ns = time.time_ns()
        start_mono = time.monotonic()
        status_code = 500
        response_started = False

        async def _send_with_status(message: Dict[str, Any]) -> None:
            nonlocal response_started, status_code
            if message.get("type") == "http.response.start":
                observed = message.get("status")
                if isinstance(observed, int) and not isinstance(observed, bool):
                    status_code = observed
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive, _send_with_status)
        except BaseException:
            self._emit_safely(
                scope,
                status_code if response_started else 500,
                start_ns,
                start_mono,
            )
            raise
        self._emit_safely(scope, status_code, start_ns, start_mono)


def _validating_exporter(
    delegate: Any,
    *,
    allowed_routes: Callable[[], set],
    expected_resource: Dict[str, Any],
) -> Any:
    """Wrap a span exporter in a FAIL-CLOSED complete-envelope validator.

    A span is exported only when every check passes; a non-conforming span is
    dropped entirely — never mutated, never partially exported. Drops are
    logged as a per-reason count WITHOUT the rejected values.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import SpanKind

    def _violation(span: Any) -> Optional[str]:
        try:
            scope = _span_scope(span)
            if str(getattr(scope, "name", "") or "") != INSTRUMENTATION_SCOPE:
                return "scope"
            if getattr(scope, "version", None) or getattr(scope, "schema_url", None):
                return "scope_metadata"
            if getattr(span, "kind", None) is not SpanKind.SERVER:
                return "kind"
            if getattr(span, "parent", None) is not None:
                return "parent"
            context = getattr(span, "context", None) or span.get_span_context()
            trace_state = getattr(context, "trace_state", None)
            if trace_state is not None and len(trace_state) > 0:
                return "trace_state"
            attributes = dict(span.attributes or {})
            if not _attributes_conform(attributes):
                return "attributes"
            method = attributes["http.request.method"]
            if method not in _ALLOWED_HTTP_METHODS:
                return "method"
            route = attributes["http.route"]
            if span.name != "%s %s" % (method, route):
                return "name"
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
    are never read, and their presence at runtime refuses initialization.
    """
    if span_exporter is None and not _enabled():
        return False
    ambient = _ambient_exporter_env_names()
    if ambient:
        # A CI grep cannot see variables injected by a restart script or the
        # live process environment; refuse at runtime instead. Names only —
        # never the values.
        try:
            print(
                "gateway_otel_bootstrap_refused ambient_exporter_env=%s"
                % ",".join(ambient),
                flush=True,
            )
        except Exception:
            pass
        return False
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            SimpleSpanProcessor,
        )
        from opentelemetry.trace import SpanKind, StatusCode, Status
        from opentelemetry.context import Context

        # FIXED resource: the plain constructor takes exactly these attributes.
        # The SDK's env-merging factory is deliberately avoided so ambient
        # resource variables can never leak into exported metadata.
        expected_resource = {"service.name": SERVICE_NAME}
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
        else:
            token = os.getenv("GATEWAY_OTEL_TOKEN", "").strip()
            if not token:
                # An empty explicit headers dict would let the pinned exporter
                # fall back to ambient header variables — require the token.
                try:
                    print("gateway_otel_bootstrap_refused token_missing", flush=True)
                except Exception:
                    pass
                return False

            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            endpoint = os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip()
            headers = {"Authorization": "Bearer " + token}
            # Explicit arguments only: the destination exists solely inside
            # this exporter object, never in ambient environment variables.
            delegate = OTLPSpanExporter(endpoint=endpoint, headers=headers)
        validating_exporter = _validating_exporter(
            delegate,
            allowed_routes=_registered_routes,
            expected_resource=expected_resource,
        )
        if span_exporter is not None:
            processor = SimpleSpanProcessor(validating_exporter)
        else:
            processor = BatchSpanProcessor(
                validating_exporter,
                max_queue_size=_BATCH_MAX_QUEUE_SIZE,
                schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MILLIS,
                max_export_batch_size=_BATCH_MAX_EXPORT_BATCH_SIZE,
                export_timeout_millis=_BATCH_EXPORT_TIMEOUT_MILLIS,
            )

        # A dedicated provider, NOT the global one, so nothing else (e.g. Langfuse)
        # is affected and no ambient instrumentation is picked up.
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(INSTRUMENTATION_SCOPE)

        def _emit(scope, status_code, start_ns, start_mono):
            # Resolve the route template only after routing has run, so ids stay
            # out of the label. Suppression already handled static health paths.
            route = _safe_route_label(scope)
            method = str(scope.get("method") or "")
            span = tracer.start_span(
                "%s %s" % (method, route),
                kind=SpanKind.SERVER,
                # A fresh root context: never adopt a caller-supplied parent
                # or trace state.
                context=Context(),
                start_time=start_ns,
            )
            # Only method / route template / status / duration — nothing else.
            span.set_attribute("http.request.method", method)
            span.set_attribute("http.route", route)
            span.set_attribute("http.response.status_code", int(status_code))
            span.set_attribute("duration_ms", (time.monotonic() - start_mono) * 1000.0)
            if int(status_code) >= 500:
                # Name the internal stage this request last entered, so a generic
                # 5xx says where it failed. Vocabulary-checked at the boundary.
                stage = stage_context.current_stage()
                if stage in STAGE_ATTRIBUTE_VALUES:
                    span.set_attribute(STAGE_ATTRIBUTE, stage)
                span.set_status(Status(StatusCode.ERROR))
            span.end()

        # Register a raw ASGI middleware. FastAPI's ``@app.middleware("http")``
        # silently wraps the function in BaseHTTPMiddleware and changes
        # exception, cancellation, streaming, and context behavior.
        app.add_middleware(_GatewayOtelMiddleware, emit=_emit)

        return True
    except Exception as exc:  # never let telemetry break the gateway
        try:
            # The exception message may contain a configured endpoint or
            # exporter detail. Log only its class.
            print(
                "gateway_otel_bootstrap_skipped error=%s"
                % type(exc).__name__,
                flush=True,
            )
        except Exception:
            pass
        return False

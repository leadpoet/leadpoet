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
   approved types, a standard HTTP method, a span name equal to exactly
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

The same module also wires the **Lab Arena sidecar** (``leadpoet-arena``),
which runs the Arena competition pipeline in its own process. It reuses the
identical fail-closed machinery under two additional scopes:

- ``arena.http`` — one SERVER span per sidecar request, carrying the SAME four
  attributes. At the gateway every Arena call collapses into the catch-all
  ``/arena/{arena_path:path}`` template, so the operation that was invoked is
  not recoverable from gateway telemetry at all; the sidecar's own route
  templates restore it without exporting a single client-controlled string.
- ``arena.task`` — one INTERNAL span per driver / worker stage, so a stalled
  pipeline stops being indistinguishable from an idle one. Its attribute set
  is a separate, equally fixed allowlist: a stage name from a frozen
  vocabulary, an outcome from a frozen vocabulary, an exception CLASS name
  (code-controlled, shape-validated) and a bounded integer count. No round id,
  submission id, hotkey, source, prompt, score, or model output can pass it.

Both Arena scopes share the gateway's destination (the same private
``GATEWAY_OTEL_*`` values, read from the same protected env file) so enabling
them requires no new secret and no new host configuration.

It is a complete no-op unless ``GATEWAY_OTEL_ENABLED`` is truthy AND
``GATEWAY_OTEL_ENDPOINT`` AND ``GATEWAY_OTEL_TOKEN`` are all set. Any failure
while wiring it up is swallowed so it can never delay or break gateway
startup. No endpoint or token is ever hard-coded or committed.
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

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

# The Arena sidecar is a DIFFERENT process with a DIFFERENT fixed identity.
# Never let an Arena reading inherit the gateway's service name.
ARENA_SERVICE_NAME = "leadpoet-arena"

# The Arena sidecar's two scopes: request envelopes and pipeline stages.
ARENA_HTTP_SCOPE = "arena.http"
ARENA_TASK_SCOPE = "arena.task"

# Every pipeline stage that may be reported, as a FROZEN vocabulary. A stage
# name outside this set is dropped whole, so a future caller cannot smuggle a
# computed or client-derived string out through the stage label.
ARENA_TASK_STAGES = frozenset(
    {
        "driver_tick",
        "promote_baselines",
        "active_rounds",
        "advance_round",
        "ensure_daily_round",
        "activate_rewards",
        "reconcile_provider_costs",
        "review_submissions",
    }
)

# The only outcomes a stage may report.
ARENA_TASK_OUTCOMES = frozenset({"ok", "idle", "failed"})

# An exception CLASS name is code-controlled, not client-controlled, but it is
# still shape-checked so no message, path, or value can ever ride along.
_ERROR_TYPE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
ARENA_NO_ERROR = "-"

# Upper bound on a reported count: an operational magnitude, never a payload.
_ARENA_MAX_COUNT = 1_000_000

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

# The ONLY attributes an Arena pipeline-stage span may carry. A separate,
# equally exact allowlist — the HTTP four do not apply to a task span.
TASK_SPAN_ATTRIBUTE_TYPES: Dict[str, tuple] = {
    "arena.stage": (str,),
    "arena.outcome": (str,),
    "arena.error_type": (str,),
    "arena.count": (int,),
    "duration_ms": (int, float),
}

TASK_SPAN_ATTRIBUTE_ALLOWLIST = frozenset(TASK_SPAN_ATTRIBUTE_TYPES)

# Log tokens, written out in full so the CI boundary guard can grep them.
_GATEWAY_LOG = {
    "refused": "gateway_otel_bootstrap_refused",
    "skipped": "gateway_otel_bootstrap_skipped",
    "dropped": "gateway_otel_span_dropped",
    "emit_failed": "gateway_otel_emit_failed",
}
_ARENA_LOG = {
    "refused": "arena_otel_bootstrap_refused",
    "skipped": "arena_otel_bootstrap_skipped",
    "dropped": "arena_otel_span_dropped",
    "emit_failed": "arena_otel_emit_failed",
}


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


def _attributes_conform(
    attributes: Dict[str, Any],
    types: Optional[Dict[str, tuple]] = None,
) -> bool:
    required = SPAN_ATTRIBUTE_TYPES if types is None else types
    if set(attributes) != set(required):
        return False
    for key, allowed_types in required.items():
        value = attributes[key]
        if isinstance(value, bool) or not isinstance(value, allowed_types):
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
    scopes: Optional[Dict[str, str]] = None,
    log: Optional[Dict[str, str]] = None,
) -> Any:
    """Wrap a span exporter in a FAIL-CLOSED complete-envelope validator.

    A span is exported only when every check passes; a non-conforming span is
    dropped entirely — never mutated, never partially exported. Drops are
    logged as a per-reason count WITHOUT the rejected values.

    ``scopes`` maps each ACCEPTED instrumentation scope to its envelope kind —
    ``"http"`` (a SERVER request span carrying exactly the four approved HTTP
    attributes) or ``"task"`` (an INTERNAL pipeline-stage span carrying exactly
    the five approved Arena stage attributes). A span whose scope is not listed
    is dropped, so one process can never emit through another's envelope.
    """
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
    from opentelemetry.trace import SpanKind

    accepted_scopes = (
        {INSTRUMENTATION_SCOPE: "http"} if scopes is None else dict(scopes)
    )
    tokens = _GATEWAY_LOG if log is None else log

    def _http_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes):
            return "attributes"
        method = attributes["http.request.method"]
        if method not in _ALLOWED_HTTP_METHODS:
            return "method"
        route = attributes["http.route"]
        if span.name != "%s %s" % (method, route):
            return "name"
        if route != UNMATCHED_ROUTE_LABEL and route not in allowed_routes():
            return "route"
        return None

    def _task_violation(span: Any, attributes: Dict[str, Any]) -> Optional[str]:
        if not _attributes_conform(attributes, TASK_SPAN_ATTRIBUTE_TYPES):
            return "attributes"
        stage = attributes["arena.stage"]
        if stage not in ARENA_TASK_STAGES:
            return "stage"
        if attributes["arena.outcome"] not in ARENA_TASK_OUTCOMES:
            return "outcome"
        error_type = attributes["arena.error_type"]
        if error_type != ARENA_NO_ERROR and not _ERROR_TYPE_RE.fullmatch(error_type):
            return "error_type"
        count = attributes["arena.count"]
        if count < 0 or count > _ARENA_MAX_COUNT:
            return "count"
        if span.name != "arena.%s" % stage:
            return "name"
        return None

    def _violation(span: Any) -> Optional[str]:
        try:
            scope = _span_scope(span)
            envelope = accepted_scopes.get(str(getattr(scope, "name", "") or ""))
            if envelope is None:
                return "scope"
            if getattr(scope, "version", None) or getattr(scope, "schema_url", None):
                return "scope_metadata"
            expected_kind = SpanKind.SERVER if envelope == "http" else SpanKind.INTERNAL
            if getattr(span, "kind", None) is not expected_kind:
                return "kind"
            if getattr(span, "parent", None) is not None:
                return "parent"
            context = getattr(span, "context", None) or span.get_span_context()
            trace_state = getattr(context, "trace_state", None)
            if trace_state is not None and len(trace_state) > 0:
                return "trace_state"
            attributes = dict(span.attributes or {})
            shape = (
                _http_violation(span, attributes)
                if envelope == "http"
                else _task_violation(span, attributes)
            )
            if shape is not None:
                return shape
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
                        "%s %s"
                        % (
                            tokens["dropped"],
                            " ".join("%s=%d" % kv for kv in sorted(dropped.items())),
                        ),
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


def _build_otlp_exporter(log: Dict[str, str]) -> Optional[Any]:
    """Build the single OTLP exporter, or None when it must stay off.

    The destination is passed EXPLICITLY from the private ``GATEWAY_OTEL_*``
    variables; ambient exporter variables are never consulted. Both the
    gateway and the Arena sidecar share this one construction site, which is
    what keeps ``tests/test_otel_boundary_guard.py``'s "exporters live in one
    module" guarantee true.
    """
    token = os.getenv("GATEWAY_OTEL_TOKEN", "").strip()
    if not token:
        # An empty explicit headers dict would let the pinned exporter
        # fall back to ambient header variables — require the token.
        try:
            print(log["refused"] + " token_missing", flush=True)
        except Exception:
            pass
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    endpoint = os.getenv("GATEWAY_OTEL_ENDPOINT", "").strip()
    headers = {"Authorization": "Bearer " + token}
    # Explicit arguments only: the destination exists solely inside
    # this exporter object, never in ambient environment variables.
    return OTLPSpanExporter(endpoint=endpoint, headers=headers)


def _build_span_processor(validating_exporter: Any, *, simple: bool) -> Any:
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    if simple:
        return SimpleSpanProcessor(validating_exporter)
    return BatchSpanProcessor(
        validating_exporter,
        max_queue_size=_BATCH_MAX_QUEUE_SIZE,
        schedule_delay_millis=_BATCH_SCHEDULE_DELAY_MILLIS,
        max_export_batch_size=_BATCH_MAX_EXPORT_BATCH_SIZE,
        export_timeout_millis=_BATCH_EXPORT_TIMEOUT_MILLIS,
    )


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

        delegate = span_exporter
        if delegate is None:
            delegate = _build_otlp_exporter(_GATEWAY_LOG)
            if delegate is None:
                return False
        validating_exporter = _validating_exporter(
            delegate,
            allowed_routes=_registered_routes,
            expected_resource=expected_resource,
            scopes={INSTRUMENTATION_SCOPE: "http"},
            log=_GATEWAY_LOG,
        )
        processor = _build_span_processor(
            validating_exporter,
            simple=span_exporter is not None,
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


class ArenaTelemetry:
    """Records Arena pipeline-stage spans. Never raises, never blocks.

    One INTERNAL root span per stage, carrying only the five approved
    attributes. The caller supplies a stage name and an outcome from the
    frozen vocabularies; anything else is dropped by the same fail-closed
    validator that guards the HTTP envelope.
    """

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    def record(
        self,
        stage: str,
        outcome: str,
        *,
        error_type: str = ARENA_NO_ERROR,
        count: int = 0,
        duration_ms: float = 0.0,
        start_ns: Optional[int] = None,
    ) -> None:
        try:
            self._record(
                stage,
                outcome,
                error_type=error_type,
                count=count,
                duration_ms=duration_ms,
                start_ns=start_ns,
            )
        except BaseException as exc:
            # Telemetry is observational only — the pipeline must not notice.
            try:
                print(
                    "%s error=%s" % (_ARENA_LOG["emit_failed"], type(exc).__name__),
                    flush=True,
                )
            except BaseException:
                pass

    def _record(
        self,
        stage: str,
        outcome: str,
        *,
        error_type: str,
        count: int,
        duration_ms: float,
        start_ns: Optional[int],
    ) -> None:
        from opentelemetry.context import Context
        from opentelemetry.trace import SpanKind, Status, StatusCode

        span = self._tracer.start_span(
            "arena.%s" % stage,
            kind=SpanKind.INTERNAL,
            # A fresh root context: stage spans never adopt a caller's parent.
            context=Context(),
            start_time=start_ns if start_ns is not None else time.time_ns(),
        )
        span.set_attribute("arena.stage", str(stage))
        span.set_attribute("arena.outcome", str(outcome))
        span.set_attribute("arena.error_type", str(error_type))
        span.set_attribute("arena.count", int(count))
        span.set_attribute("duration_ms", float(duration_ms))
        if outcome == "failed":
            span.set_status(Status(StatusCode.ERROR))
        span.end()


def configure_arena_otel(
    app: Any = None, *, span_exporter: Optional[Any] = None
) -> Optional[ArenaTelemetry]:
    """Wire Arena sidecar telemetry. No-op (returns None) unless enabled.

    Installs two things on the SAME fail-closed pipeline the gateway uses:

    - request spans under ``arena.http`` for the sidecar's own route
      templates, which the gateway's ``/arena/{arena_path:path}`` catch-all
      cannot express;
    - a recorder for pipeline-stage spans under ``arena.task``.

    The returned recorder is what ``lab_arena.telemetry`` installs; when this
    returns None the pipeline keeps calling a no-op and nothing changes.
    """
    if span_exporter is None and not _enabled():
        return None
    ambient = _ambient_exporter_env_names()
    if ambient:
        try:
            print(
                "%s ambient_exporter_env=%s"
                % (_ARENA_LOG["refused"], ",".join(ambient)),
                flush=True,
            )
        except Exception:
            pass
        return None
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import SpanKind, StatusCode, Status
        from opentelemetry.context import Context

        # FIXED resource, distinct service identity: an Arena reading must
        # never be attributed to the gateway (or inherit its sensors).
        expected_resource = {"service.name": ARENA_SERVICE_NAME}
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

        delegate = span_exporter
        if delegate is None:
            delegate = _build_otlp_exporter(_ARENA_LOG)
            if delegate is None:
                return None
        validating_exporter = _validating_exporter(
            delegate,
            allowed_routes=_registered_routes,
            expected_resource=expected_resource,
            scopes={ARENA_HTTP_SCOPE: "http", ARENA_TASK_SCOPE: "task"},
            log=_ARENA_LOG,
        )
        processor = _build_span_processor(
            validating_exporter,
            simple=span_exporter is not None,
        )

        # A dedicated provider, never the global one.
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)

        if app is not None:

            http_tracer = provider.get_tracer(ARENA_HTTP_SCOPE)

            def _emit(scope, status_code, start_ns, start_mono):
                route = _safe_route_label(scope)
                method = str(scope.get("method") or "")
                span = http_tracer.start_span(
                    "%s %s" % (method, route),
                    kind=SpanKind.SERVER,
                    context=Context(),
                    start_time=start_ns,
                )
                span.set_attribute("http.request.method", method)
                span.set_attribute("http.route", route)
                span.set_attribute("http.response.status_code", int(status_code))
                span.set_attribute(
                    "duration_ms", (time.monotonic() - start_mono) * 1000.0
                )
                if int(status_code) >= 500:
                    span.set_status(Status(StatusCode.ERROR))
                span.end()

            app.add_middleware(_GatewayOtelMiddleware, emit=_emit)

        return ArenaTelemetry(provider.get_tracer(ARENA_TASK_SCOPE))
    except Exception as exc:  # never let telemetry break the Arena service
        try:
            print(
                "%s error=%s" % (_ARENA_LOG["skipped"], type(exc).__name__),
                flush=True,
            )
        except Exception:
            pass
        return None

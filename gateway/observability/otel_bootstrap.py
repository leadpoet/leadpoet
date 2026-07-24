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
- It is a complete no-op unless BOTH `GATEWAY_OTEL_ENABLED` is truthy and
  `OTEL_EXPORTER_OTLP_ENDPOINT` is set. Any failure while wiring it up is
  swallowed so it can never delay or break gateway startup.

The exporter reads the standard OTel environment variables
(`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_EXPORTER_OTLP_HEADERS`,
`OTEL_SERVICE_NAME`); no endpoint or token is ever hard-coded or committed.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

_TRUTHY = {"1", "true", "yes", "on"}

# Route templates whose spans are suppressed entirely — health/liveness noise.
_SUPPRESSED_ROUTES = {"/health", "/health/live", "/health/ready"}


def _enabled() -> bool:
    return (
        os.getenv("GATEWAY_OTEL_ENABLED", "").strip().lower() in _TRUTHY
        and bool(os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip())
    )


def _safe_route_label(request: Any) -> str:
    """Return the low-cardinality route template, never the concrete path.

    Using the matched route template (e.g. ``/research-lab/allocations/attested/
    {epoch}``) keeps ids, hotkeys, and epoch numbers out of telemetry. When the
    template cannot be resolved, fall back to the first path segment only.
    """
    try:
        route = request.scope.get("route")
        template = getattr(route, "path", None)
        if isinstance(template, str) and template:
            return template
    except Exception:
        pass
    try:
        segments = [s for s in request.url.path.split("/") if s]
        return "/" + segments[0] + "/*" if segments else "/"
    except Exception:
        return "/"


def configure_gateway_otel(app: Any, *, span_exporter: Optional[Any] = None) -> bool:
    """Wire infra-only request spans onto the gateway app. No-op unless enabled.

    Returns True if instrumentation was installed, False otherwise. ``span_exporter``
    is a test seam; production always builds the OTLP exporter from env.
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

        if span_exporter is not None:
            exporter = span_exporter
            processor = SimpleSpanProcessor(exporter)
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter = OTLPSpanExporter()  # reads OTEL_EXPORTER_OTLP_* env
            processor = BatchSpanProcessor(exporter)

        service_name = os.getenv("OTEL_SERVICE_NAME", "").strip() or "leadpoet-gateway"
        # A dedicated provider, NOT the global one, so nothing else (e.g. Langfuse)
        # is affected and no ambient instrumentation is picked up.
        provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("gateway.http")

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

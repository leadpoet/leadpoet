"""OpenTelemetry bootstrap for the gateway — traces, metrics, logs over OTLP/HTTP.

Activates only when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set (gateway/config.py
loads .env before this module is initialized). Degrades to a no-op when the
endpoint is unset or the OTel packages are missing, so the gateway never gains
a hard runtime dependency on telemetry.

Config (standard OTel env vars — see env.example):
  OTEL_EXPORTER_OTLP_ENDPOINT   OTLP HTTP root URL (SDK appends /v1/<signal>)
  OTEL_EXPORTER_OTLP_HEADERS    e.g. Authorization=Bearer%20<token>
  OTEL_SERVICE_NAME             defaults to leadpoet-gateway
"""
from __future__ import annotations

import logging
import os

_initialized = False


def init_telemetry() -> None:
    """Idempotent SDK init + auto-instrumentation of outbound HTTP clients."""
    global _initialized
    if _initialized or not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        return

    os.environ.setdefault("OTEL_SERVICE_NAME", "leadpoet-gateway")
    resource = Resource.create()  # reads OTEL_SERVICE_NAME + OTEL_RESOURCE_ATTRIBUTES

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    metrics.set_meter_provider(MeterProvider(
        resource=resource,
        metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())],
    ))

    logger_provider = LoggerProvider(resource=resource)
    logger_provider.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logger_provider)
    logging.getLogger().addHandler(LoggingHandler(level=logging.INFO, logger_provider=logger_provider))

    for path, cls in (
        ("opentelemetry.instrumentation.aiohttp_client", "AioHttpClientInstrumentor"),
        ("opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("opentelemetry.instrumentation.requests", "RequestsInstrumentor"),
        ("opentelemetry.instrumentation.urllib", "URLLibInstrumentor"),
        ("opentelemetry.instrumentation.urllib3", "URLLib3Instrumentor"),
    ):
        try:
            module = __import__(path, fromlist=[cls])
            getattr(module, cls)().instrument()
        except Exception:
            pass  # a missing/incompatible instrumentation must never block the gateway

    _initialized = True


def instrument_app(app) -> None:
    """Attach SERVER-span instrumentation to the FastAPI app (no-op when off)."""
    if not _initialized:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app, excluded_urls="health")
    except Exception:
        pass

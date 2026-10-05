"""Exercise the real batched OTLP HTTP exporter against a local collector."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from gateway.observability import otel_bootstrap as otel


def test_real_http_export_keeps_only_approved_operational_metadata(monkeypatch):
    received = []
    arrived = threading.Event()

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append((
                self.path,
                self.headers.get("Authorization"),
                self.rfile.read(int(self.headers["Content-Length"])),
            ))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-protobuf")
            self.send_header("Content-Length", "0")
            self.end_headers()
            arrived.set()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Collector)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    processors = []
    build_processor = otel._build_span_processor

    def capture_processor(exporter, *, simple):
        assert simple is False
        processor = build_processor(exporter, simple=simple)
        processors.append(processor)
        return processor

    monkeypatch.setenv("GATEWAY_OTEL_ENABLED", "1")
    monkeypatch.setenv("GATEWAY_OTEL_ENDPOINT", "http://127.0.0.1:%d/v1/traces" % server.server_port)
    monkeypatch.setenv("GATEWAY_OTEL_TOKEN", "fixture-ingest-token")
    monkeypatch.setattr(otel, "_build_span_processor", capture_processor)
    try:
        recorder = otel.configure_arena_otel()
        assert recorder is not None
        recorder.record_provider("openrouter", "openrouter.responses", "ok", cost_microusd=60000)
        recorder.record("driver_tick", "ok", count=1)
        recorder.record("fixture-private-prompt", "ok")
        assert processors[0].force_flush(timeout_millis=3000)
        assert arrived.wait(1)
        assert len(received) == 1
        path, authorization, body = received[0]
        assert path == "/v1/traces"
        assert authorization == "Bearer fixture-ingest-token"
        request = ExportTraceServiceRequest.FromString(body)
        resources = request.resource_spans
        assert len(resources) == 1
        assert resources[0].resource.attributes[0].value.string_value == "leadpoet-arena"
        spans = [span for scope in resources[0].scope_spans for span in scope.spans]
        assert {span.name for span in spans} == {"arena.provider.openrouter", "arena.driver_tick"}
        assert b"fixture-private-prompt" not in body
        assert b"fixture-ingest-token" not in body
    finally:
        for processor in processors:
            processor.shutdown()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

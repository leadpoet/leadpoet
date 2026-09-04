from __future__ import annotations

import base64
from dataclasses import replace
import errno
import gzip
import socket
import ssl
import sys
import threading
import time
from types import SimpleNamespace

import httpcore
import httpx
import pytest

from gateway.tee.egress_framing import (
    TUNNEL_FRAMING_HEADER,
    TUNNEL_FRAMING_MODE,
)
from gateway.tee.provider_broker_v2 import (
    BUILTIN_PROVIDER_ROUTES,
    DIRECT_PROVIDER_KEEPALIVE_EXPIRY_SECONDS,
    EGRESS_POLICY_JOB_PROXY_ALLOWED,
    HTTPXProviderTransport,
    MAX_RESPONSE_BODY_BYTES,
    MAX_TRANSPORT_RESPONSE_BODY_BYTES,
    MEASURED_TRANSPORT_REQUEST_HEADERS,
    PROVIDER_RPC_RESPONSE_RESERVE_BYTES,
    PROVIDER_BROKER_SCHEMA_VERSION,
    PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION,
    ProviderBrokerV2,
    ProviderBrokerV2Error,
    ProviderTransportCleanupError,
    _authenticated_body_is_complete_after_stream_error,
    _extract_tls_metadata,
    _failure_code,
    _close_client_transports,
    _force_close_response_network_stream,
    _local_resource_failure,
    _provider_transport_failure_diagnostic,
    _provider_rpc_response_body_limit,
    credential_reference_hash,
    credential_value_hash,
    expected_job_credential_slot_ref_hashes,
    expected_provider_credential_slots,
    measured_retry_policy_hashes,
    provider_registry_document,
    provider_registry_hash,
)
from gateway.tee.inter_enclave_tls import MAX_FRAME_BYTES, REPLAY_WAIT_SECONDS
from leadpoet_canonical.attested_v2 import (
    DIRECT_EGRESS_REF_HASH,
    validate_transport_attempt,
)
from leadpoet_canonical.attested_v2 import sha256_bytes


HASH = "sha256:" + "a" * 64
NOW = "2026-07-10T20:00:00Z"


def test_default_response_limit_fits_authenticated_rpc_frame_budget():
    assert MAX_RESPONSE_BODY_BYTES == _provider_rpc_response_body_limit(
        frame_bytes=MAX_FRAME_BYTES,
        reserve_bytes=PROVIDER_RPC_RESPONSE_RESERVE_BYTES,
    )
    encoded_body_bytes = 4 * ((MAX_RESPONSE_BODY_BYTES + 2) // 3)

    assert encoded_body_bytes + PROVIDER_RPC_RESPONSE_RESERVE_BYTES <= (
        MAX_FRAME_BYTES
    )
    assert (
        4 * ((MAX_RESPONSE_BODY_BYTES + 3) // 3)
        + PROVIDER_RPC_RESPONSE_RESERVE_BYTES
        > MAX_FRAME_BYTES
    )


def test_tls_metadata_supports_python39_positional_only_peer_certificate():
    class PositionalOnlyTLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return PositionalOnlyTLS()

    class Response:
        extensions = {"network_stream": Stream()}

    assert _extract_tls_metadata(Response()) == (
        sha256_bytes(b"peer-certificate"),
        "TLSv1.3",
    )


def test_response_stream_false_close_falls_back_to_confirmed_raw_socket():
    class RawSocket:
        def shutdown(self, _direction):
            return None

        def close(self):
            return None

    class NetworkStream:
        def close(self):
            return False

        def get_extra_info(self, name):
            assert name == "socket"
            return RawSocket()

    response = SimpleNamespace(extensions={"network_stream": NetworkStream()})
    _force_close_response_network_stream(response, required=True)


def test_response_stream_and_raw_socket_false_close_fail_closed():
    class RawSocket:
        def shutdown(self, _direction):
            return None

        def close(self):
            return False

    class NetworkStream:
        def close(self):
            return False

        def get_extra_info(self, name):
            assert name == "socket"
            return RawSocket()

    response = SimpleNamespace(extensions={"network_stream": NetworkStream()})
    with pytest.raises(ProviderBrokerV2Error, match="cleanup failed"):
        _force_close_response_network_stream(response, required=True)


@pytest.mark.parametrize(
    ("client_result", "transport_result", "expected_closed"),
    (
        (False, None, (False, True)),
        (None, False, (True, False)),
    ),
)
def test_client_or_explicit_transport_false_close_is_unconfirmed(
    client_result,
    transport_result,
    expected_closed,
):
    transport = SimpleNamespace(close=lambda: transport_result)
    client = SimpleNamespace(close=lambda: client_result)
    setattr(client, "_leadpoet_explicit_http_transport", transport)

    client_closed, transport_closed, cleanup_error = _close_client_transports(
        client
    )

    assert (client_closed, transport_closed) == expected_closed
    assert cleanup_error is not None


def test_httpx_transport_captures_tls_before_peer_closes_after_body(monkeypatch):
    client_options = []
    transport_options = []

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def __init__(self):
            self.ssl_object = TLS()
            self.closed = False

        def get_extra_info(self, name):
            assert name == "ssl_object"
            return self.ssl_object

        def close(self):
            self.closed = True

    class Response:
        def __init__(self):
            self.status_code = 200
            self.headers = {
                "content-length": "11",
                "content-type": "application/json",
                "transfer-encoding": "chunked",
            }
            self.stream = Stream()
            self.extensions = {"network_stream": self.stream}

        def iter_bytes(self):
            yield b'{"ok":true}'
            # S3 can close immediately after delivering the complete body.
            self.stream.ssl_object = None

    response = Response()

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self, **kwargs):
            client_options.append(dict(kwargs))

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def close(self):
            pass

        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    class HTTPTransport:
        def __init__(self, **kwargs):
            self.options = dict(kwargs)
            self.closed = False
            transport_options.append(self.options)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            Client=Client,
            HTTPTransport=HTTPTransport,
            Limits=lambda **kwargs: kwargs,
            Proxy=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    result = HTTPXProviderTransport()(
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_ms=1000,
    )

    assert result == {
        "http_status": 200,
        "headers": {"content-type": "application/json"},
        "body": b'{"ok":true}',
        "tls_peer_chain_hash": sha256_bytes(b"peer-certificate"),
        "tls_protocol": "TLSv1.3",
    }
    assert set(client_options[-1]) == {
        "transport",
        "trust_env",
        "cookies",
        "follow_redirects",
    }
    assert transport_options[-1]["http2"] is True
    assert transport_options[-1]["http1"] is True
    assert transport_options[-1]["limits"] == {
        "max_connections": 64,
        "max_keepalive_connections": 0,
        "keepalive_expiry": DIRECT_PROVIDER_KEEPALIVE_EXPIRY_SECONDS,
    }

    artifact_transport = HTTPXProviderTransport(
        response_body_ceiling_bytes=MAX_TRANSPORT_RESPONSE_BODY_BYTES
    )
    response.stream.ssl_object = TLS()
    artifact_result = artifact_transport(
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_ms=1000,
        max_response_bytes=MAX_TRANSPORT_RESPONSE_BODY_BYTES,
    )
    assert artifact_result["body"] == b'{"ok":true}'

    response.stream.ssl_object = TLS()
    http1_result = artifact_transport(
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_ms=1000,
        allow_http2=False,
    )
    assert http1_result["body"] == b'{"ok":true}'
    assert transport_options[-1]["http1"] is True
    assert transport_options[-1]["http2"] is False

    with pytest.raises(ProviderBrokerV2Error, match="response limit"):
        HTTPXProviderTransport()(
            method="GET",
            url="https://example.com/artifact",
            headers={},
            body=b"",
            timeout_ms=1000,
            max_response_bytes=MAX_RESPONSE_BODY_BYTES + 1,
        )
    with pytest.raises(ProviderBrokerV2Error, match="transport response ceiling"):
        HTTPXProviderTransport(
            response_body_ceiling_bytes=MAX_TRANSPORT_RESPONSE_BODY_BYTES + 1
        )


def test_httpx_transport_preserves_complete_response_when_stream_cleanup_fails():
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def __init__(self):
            self.closed = False

        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            self.closed = True

    class CloseFailsAfterCompleteBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"ok":true}'

        def close(self):
            raise EOFError("TLS close_notify was not available")

    network_stream = NetworkStream()
    response = httpx.Response(
        200,
        headers={"content-type": "application/json"},
        stream=CloseFailsAfterCompleteBody(),
        extensions={"network_stream": network_stream},
        request=httpx.Request("GET", "https://example.com/artifact"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    result = HTTPXProviderTransport._execute_with_client(
        Client(),
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=1024,
    )

    assert result["http_status"] == 200
    assert result["body"] == b'{"ok":true}'
    assert result["tls_peer_chain_hash"] == sha256_bytes(b"peer-certificate")
    assert network_stream.closed is True
    assert result["tls_protocol"] == "TLSv1.3"


def test_response_cleanup_failure_force_closes_underlying_socket():
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    for _ in range(32):
        local_socket, peer_socket = socket.socketpair()

        class NetworkStream:
            def get_extra_info(self, name):
                if name == "ssl_object":
                    return TLS()
                if name == "socket":
                    return local_socket
                return None

            def close(self):
                raise EOFError("relay close acknowledgement was lost")

        class CloseFailsAfterCompleteBody(httpx.SyncByteStream):
            def __iter__(self):
                yield b'{"ok":true}'

            def close(self):
                raise EOFError("TLS close_notify was not available")

        response = httpx.Response(
            200,
            headers={"content-length": "11"},
            stream=CloseFailsAfterCompleteBody(),
            extensions={"network_stream": NetworkStream()},
            request=httpx.Request("GET", "https://example.com/artifact"),
        )

        class ResponseContext:
            def __enter__(self):
                return response

            def __exit__(self, *_args):
                response.close()

        class Client:
            def stream(self, *_args, **_kwargs):
                return ResponseContext()

        try:
            result = HTTPXProviderTransport._execute_with_client(
                Client(),
                method="GET",
                url="https://example.com/artifact",
                headers={},
                body=b"",
                timeout_seconds=1.0,
                max_response_bytes=1024,
            )
            assert result["body"] == b'{"ok":true}'
            peer_socket.settimeout(0.2)
            assert peer_socket.recv(1) == b""
            assert local_socket.fileno() == -1
        finally:
            local_socket.close()
            peer_socket.close()


def test_request_scoped_direct_repeated_cleanup_failures_release_network(
    monkeypatch,
):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    transport = HTTPXProviderTransport()
    client_options = []
    client_close_checks = []
    local_sockets = []
    peer_sockets = []

    class Client:
        def __init__(self, **options):
            client_options.append(dict(options))
            self.local_socket, peer_socket = socket.socketpair()
            local_sockets.append(self.local_socket)
            peer_sockets.append(peer_socket)

            class NetworkStream:
                def get_extra_info(inner_self, name):
                    if name == "ssl_object":
                        return TLS()
                    if name == "socket":
                        return self.local_socket
                    return None

                def close(inner_self):
                    raise EOFError("relay close acknowledgement was lost")

            class CloseFailsAfterCompleteBody(httpx.SyncByteStream):
                def __iter__(inner_self):
                    yield b'{"ok":true}'

                def close(inner_self):
                    raise EOFError("TLS close_notify was not available")

            self.response = httpx.Response(
                200,
                headers={"content-length": "11"},
                stream=CloseFailsAfterCompleteBody(),
                extensions={"network_stream": NetworkStream()},
                request=httpx.Request("GET", "https://example.com/data"),
            )

        def stream(self, *_args, **_kwargs):
            response = self.response

            class ResponseContext:
                def __enter__(inner_self):
                    return response

                def __exit__(inner_self, *_args):
                    response.close()

            return ResponseContext()

        def close(self):
            # The response wrapper must force-close the only request-scoped
            # network stream before the pool/client cleanup boundary runs.
            assert self.local_socket.fileno() == -1
            client_close_checks.append(True)
            raise EOFError("client cleanup also lost close_notify")

    monkeypatch.setattr(httpx, "Client", Client)
    try:
        for _ in range(8):
            result = transport(
                method="GET",
                url="https://example.com/data",
                headers={},
                body=b"",
                timeout_ms=1000,
            )
            assert result["body"] == b'{"ok":true}'
            peer_sockets[-1].settimeout(0.2)
            assert peer_sockets[-1].recv(1) == b""
    finally:
        for local_socket in local_sockets:
            local_socket.close()
        for peer_socket in peer_sockets:
            peer_socket.close()

    assert len(client_options) == 8
    assert all(
        options["transport"]._pool._max_keepalive_connections == 0
        for options in client_options
    )
    assert transport.health()["request_counters"]["direct"] == {
        "started": 8,
        "succeeded": 8,
        "failed": 0,
    }
    assert client_close_checks == [True] * 8
    assert transport.health()["cleanup_counters"]["direct"] == {
        "attempted": 8,
        "succeeded": 8,
        "client_close_failed": 8,
        "transport_close_failed": 0,
    }


def test_request_scoped_cleanup_failure_preserves_primary_body_error(monkeypatch):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def __init__(self):
            self.closed = False

        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            self.closed = True

    network_stream = NetworkStream()

    class Response:
        status_code = 200
        headers = {"content-length": "12"}
        extensions = {"network_stream": network_stream}

        def iter_bytes(self):
            yield b"partial"
            raise RuntimeError("sensitive primary body failure")

    class ResponseContext:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class ExplicitTransport:
        def close(self):
            raise OSError(errno.ENOBUFS, "sensitive explicit-pool failure")

    class Client:
        def __init__(self):
            self._leadpoet_explicit_http_transport = ExplicitTransport()

        def stream(self, *_args, **_kwargs):
            return ResponseContext()

        def close(self):
            raise EOFError("sensitive client-close failure")

    transport = HTTPXProviderTransport()
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())

    with pytest.raises(ProviderTransportCleanupError) as captured:
        transport(
            method="GET",
            url="https://example.com/data",
            headers={},
            body=b"",
            timeout_ms=1000,
        )

    error = captured.value
    assert error.failure_stage == "client_transport_cleanup"
    assert error.primary_error_type == "RuntimeError"
    assert error.cleanup_error_type == "OSError"
    assert isinstance(error.__cause__, RuntimeError)
    assert network_stream.closed is True
    health = transport.health()
    assert health["direct_active_scope_count"] == 0
    assert health["direct_retired_scope_count"] == 0
    assert health["direct_active_lease_count"] == 0
    assert health["direct_retired_lease_count"] == 0
    assert health["cleanup_counters"]["direct"] == {
        "attempted": 1,
        "succeeded": 0,
        "client_close_failed": 1,
        "transport_close_failed": 1,
    }
    assert health["last_failure"] == {
        "route": "direct",
        "stage": "client_transport_cleanup",
        "failure_code": "proxy_failure",
        "error_type": "ProviderTransportCleanupError",
        "errno": errno.ENOBUFS,
        "local_resource_kind": "socket_buffer_exhausted",
        "primary_error_type": "RuntimeError",
        "cleanup_error_type": "OSError",
    }
    assert "sensitive" not in str(health)


def test_httpx_transport_does_not_hide_incomplete_response_body():
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def __init__(self):
            self.closed = False

        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            self.closed = True

    class BodyReadFails(httpx.SyncByteStream):
        def __iter__(self):
            yield b"partial"
            raise EOFError("response body is incomplete")

        def close(self):
            raise EOFError("cleanup also failed")

    response = httpx.Response(
        200,
        headers={"content-length": "12", "content-type": "application/json"},
        stream=BodyReadFails(),
        extensions={"network_stream": NetworkStream()},
        request=httpx.Request("GET", "https://example.com/artifact"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    with pytest.raises(EOFError, match="response body is incomplete"):
        HTTPXProviderTransport._execute_with_client(
            Client(),
            method="GET",
            url="https://example.com/artifact",
            headers={},
            body=b"",
            timeout_seconds=1.0,
            max_response_bytes=1024,
            allow_authenticated_complete_body_eof=True,
        )


@pytest.mark.parametrize("method", ("GET", "POST", "PATCH", "HEAD"))
def test_httpx_transport_accepts_authenticated_complete_body_before_eof(method):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    body = b'{"ok":true}' if method != "HEAD" else b""

    class CompleteBodyThenEOF(httpx.SyncByteStream):
        def __iter__(self):
            if body:
                yield body
            raise httpx.RemoteProtocolError(
                "peer closed after the complete authenticated response"
            )

        def close(self):
            return None

    response = httpx.Response(
        200,
        headers={"content-length": str(len(body) if method != "HEAD" else 321)},
        stream=CompleteBodyThenEOF(),
        extensions={"network_stream": NetworkStream()},
        request=httpx.Request(method, "https://example.com/artifact"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    result = HTTPXProviderTransport._execute_with_client(
        Client(),
        method=method,
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=1024,
        allow_authenticated_complete_body_eof=True,
    )

    assert result["http_status"] == 200
    assert result["body"] == body


@pytest.mark.parametrize("method", ("PATCH", "DELETE"))
def test_complete_body_classifier_rejects_unmeasured_mutating_response(method):
    body = b'{"ok":true}'
    response = SimpleNamespace(
        status_code=200,
        headers=httpx.Headers({"content-type": "application/json"}),
        http_version="HTTP/2",
    )

    assert not _authenticated_body_is_complete_after_stream_error(
        method=method,
        response=response,
        byte_count=len(body),
        body=body,
        error=EOFError("terminal stream signal missing"),
    )


@pytest.mark.parametrize("method", ("GET", "POST"))
def test_complete_body_classifier_rejects_encoded_exact_length_response(method):
    body = b'{"ok":true}'
    response = SimpleNamespace(
        status_code=200,
        headers=httpx.Headers(
            {
                "content-encoding": "gzip",
                "content-length": str(len(body)),
            }
        ),
    )

    assert not _authenticated_body_is_complete_after_stream_error(
        method=method,
        response=response,
        byte_count=len(body),
        body=body,
        error=EOFError("compressed stream ended ambiguously"),
    )


@pytest.mark.parametrize(
    "nested_error",
    (
        httpcore.ReadError("relay read ended after the complete body"),
        httpcore.RemoteProtocolError("relay closed after the complete body"),
        ssl.SSLEOFError("TLS peer closed after the complete body"),
    ),
)
def test_complete_body_classifier_accepts_wrapped_transport_eof(nested_error):
    body = b'{"ok":true}'
    try:
        raise nested_error
    except BaseException as exc:
        wrapped = RuntimeError("transport stream terminated")
        wrapped.__cause__ = exc
    response = SimpleNamespace(
        headers=httpx.Headers({"content-length": str(len(body))})
    )

    assert _authenticated_body_is_complete_after_stream_error(
        method="GET",
        response=response,
        byte_count=len(body),
        error=wrapped,
    )
    assert not _authenticated_body_is_complete_after_stream_error(
        method="GET",
        response=response,
        byte_count=len(body) - 1,
        error=wrapped,
    )


def test_complete_body_classifier_accepts_opaque_error_after_exact_body():
    body = b'{"ok":true}'
    wrapped = RuntimeError("stream processing failed")
    wrapped.__cause__ = ValueError("response decoder invariant failed")
    response = SimpleNamespace(
        headers=httpx.Headers({"content-length": str(len(body))})
    )

    assert _authenticated_body_is_complete_after_stream_error(
        method="GET",
        response=response,
        byte_count=len(body),
        error=wrapped,
    )
    assert not _authenticated_body_is_complete_after_stream_error(
        method="GET",
        response=response,
        byte_count=len(body) - 1,
        error=wrapped,
    )


def test_httpx_transport_accepts_opaque_error_after_exact_artifact_body():
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    body = b'{"ok":true}'

    class CompleteBodyThenOpaqueError(httpx.SyncByteStream):
        def __iter__(self):
            yield body
            raise RuntimeError("relay stream wrapper terminated")

        def close(self):
            return None

    response = httpx.Response(
        200,
        headers={"content-length": str(len(body))},
        stream=CompleteBodyThenOpaqueError(),
        extensions={"network_stream": NetworkStream()},
        request=httpx.Request("GET", "https://example.com/artifact"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    result = HTTPXProviderTransport._execute_with_client(
        Client(),
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=1024,
        allow_authenticated_complete_body_eof=True,
    )

    assert result["http_status"] == 200
    assert result["body"] == body


def test_httpx_transport_requires_declared_length_for_eof_recovery():
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    class UndeclaredBodyThenEOF(httpx.SyncByteStream):
        def __iter__(self):
            yield b'{"ok":true}'
            raise EOFError("ambiguous response framing")

        def close(self):
            return None

    response = httpx.Response(
        200,
        stream=UndeclaredBodyThenEOF(),
        extensions={"network_stream": NetworkStream()},
        request=httpx.Request("GET", "https://example.com/artifact"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    with pytest.raises(EOFError, match="ambiguous response framing"):
        HTTPXProviderTransport._execute_with_client(
            Client(),
            method="GET",
            url="https://example.com/artifact",
            headers={},
            body=b"",
            timeout_seconds=1.0,
            max_response_bytes=1024,
            allow_authenticated_complete_body_eof=True,
        )


@pytest.mark.parametrize("method", ("GET", "POST"))
def test_httpx_transport_accepts_complete_chunked_json_before_eof(method):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    body = b'[{"provider":"exa"}]'

    class CompleteJSONThenEOF(httpx.SyncByteStream):
        def __iter__(self):
            yield body[:7]
            yield body[7:]
            raise httpx.RemoteProtocolError(
                "peer closed before the terminal chunk"
            )

        def close(self):
            return None

    response = httpx.Response(
        200,
        headers={
            "content-type": "application/json; charset=utf-8",
            "transfer-encoding": "chunked",
        },
        stream=CompleteJSONThenEOF(),
        extensions={"network_stream": NetworkStream()},
        request=httpx.Request(method, "https://example.com/rest/v1/providers"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    result = HTTPXProviderTransport._execute_with_client(
        Client(),
        method=method,
        url="https://example.com/rest/v1/providers",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=1024,
        allow_authenticated_complete_body_eof=True,
    )

    assert result["http_status"] == 200
    assert result["body"] == body


@pytest.mark.parametrize("method", ("GET", "POST"))
def test_httpx_transport_accepts_complete_http2_json_before_terminal_eof(method):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    body = b'[{"receipt_hash":"sha256:' + (b"a" * 64) + b'"}]'

    class CompleteJSONThenEOF(httpx.SyncByteStream):
        def __iter__(self):
            yield body[:11]
            yield body[11:]
            raise httpx.RemoteProtocolError(
                "peer closed after DATA without relaying END_STREAM"
            )

        def close(self):
            return None

    response = httpx.Response(
        200,
        headers={"content-type": "application/json; charset=utf-8"},
        stream=CompleteJSONThenEOF(),
        extensions={
            "http_version": b"HTTP/2",
            "network_stream": NetworkStream(),
        },
        request=httpx.Request(method, "https://example.com/rest/v1/receipts"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    result = HTTPXProviderTransport._execute_with_client(
        Client(),
        method=method,
        url="https://example.com/rest/v1/receipts",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=1024,
        allow_authenticated_complete_body_eof=True,
    )

    assert result["http_status"] == 200
    assert result["body"] == body


def _execute_gzip_response(
    *,
    compressed_body,
    method="GET",
    status=200,
    content_type="application/json",
    terminal_error=True,
    max_response_bytes=1024,
    http_version="HTTP/2",
):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class NetworkStream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    class GzipStream(httpx.SyncByteStream):
        def __iter__(self):
            midpoint = max(1, len(compressed_body) // 2)
            yield compressed_body[:midpoint]
            yield compressed_body[midpoint:]
            if terminal_error:
                raise httpx.RemoteProtocolError(
                    "peer closed after gzip DATA without relaying END_STREAM"
                )

        def close(self):
            return None

    response = httpx.Response(
        status,
        headers={
            "content-encoding": "gzip",
            "content-type": content_type,
        },
        stream=GzipStream(),
        extensions={
            "http_version": http_version.encode("ascii"),
            "network_stream": NetworkStream(),
        },
        request=httpx.Request(method, "https://example.com/rest/v1/receipts"),
    )

    class ResponseContext:
        def __enter__(self):
            return response

        def __exit__(self, *_args):
            response.close()

    class Client:
        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    return HTTPXProviderTransport._execute_with_client(
        Client(),
        method=method,
        url="https://example.com/rest/v1/receipts",
        headers={},
        body=b"",
        timeout_seconds=1.0,
        max_response_bytes=max_response_bytes,
        allow_authenticated_complete_body_eof=True,
    )


@pytest.mark.parametrize("terminal_error", (False, True))
@pytest.mark.parametrize(
    ("method", "http_version"),
    (
        ("GET", "HTTP/1.1"),
        ("GET", "HTTP/2"),
        ("POST", "HTTP/1.1"),
        ("POST", "HTTP/2"),
    ),
)
def test_httpx_transport_requires_complete_gzip_member_before_recovery(
    terminal_error,
    method,
    http_version,
):
    body = b'[{"receipt_hash":"sha256:' + (b"a" * 64) + b'"}]'

    result = _execute_gzip_response(
        compressed_body=gzip.compress(body),
        terminal_error=terminal_error,
        method=method,
        http_version=http_version,
    )

    assert result["http_status"] == 200
    assert result["body"] == body
    assert result["headers"] == {"content-type": "application/json"}
    assert result["tls_protocol"] == "TLSv1.3"


def test_httpx_transport_rejects_gzip_body_without_authenticated_footer():
    compressed = gzip.compress(b'[{"ok":true}]')

    with pytest.raises(httpx.RemoteProtocolError, match="without relaying"):
        _execute_gzip_response(compressed_body=compressed[:-8])


def test_httpx_transport_rejects_gzip_body_with_corrupt_checksum():
    compressed = bytearray(gzip.compress(b'[{"ok":true}]'))
    compressed[-8] ^= 0x01

    with pytest.raises(Exception, match="incorrect data check"):
        _execute_gzip_response(compressed_body=bytes(compressed))


def test_httpx_transport_rejects_complete_gzip_for_mutating_method():
    with pytest.raises(httpx.RemoteProtocolError, match="without relaying"):
        _execute_gzip_response(
            compressed_body=gzip.compress(b'{"ok":true}'),
            method="PATCH",
            http_version="HTTP/1.1",
        )


def test_httpx_transport_rejects_gzip_member_with_trailing_data():
    with pytest.raises(ProviderBrokerV2Error, match="trailing data"):
        _execute_gzip_response(
            compressed_body=gzip.compress(b'{"ok":true}') + b"trailing",
        )


def test_httpx_transport_rejects_complete_gzip_with_invalid_json_after_eof():
    with pytest.raises(httpx.RemoteProtocolError, match="without relaying"):
        _execute_gzip_response(compressed_body=gzip.compress(b'{"ok":'))


def test_httpx_transport_enforces_decoded_gzip_response_ceiling():
    with pytest.raises(ProviderBrokerV2Error, match="response exceeds size limit"):
        _execute_gzip_response(
            compressed_body=gzip.compress(b"x" * 1025),
            terminal_error=False,
            max_response_bytes=1024,
        )


@pytest.mark.parametrize(
    ("http_version", "status", "headers", "body"),
    (
        ("HTTP/1.1", 200, {"content-type": "application/json"}, b"[]"),
        ("HTTP/2", 500, {"content-type": "application/json"}, b"[]"),
        ("HTTP/2", 200, {"content-type": "text/plain"}, b"[]"),
        (
            "HTTP/2",
            200,
            {"content-type": "application/json", "content-encoding": "gzip"},
            b"[]",
        ),
        ("HTTP/2", 200, {"content-type": "application/json"}, b'[{"x":1}'),
        ("HTTP/2", 200, {"content-type": "application/json"}, b"true"),
    ),
)
def test_complete_body_classifier_rejects_ambiguous_unframed_response(
    http_version,
    status,
    headers,
    body,
):
    response = SimpleNamespace(
        status_code=status,
        headers=httpx.Headers(headers),
        http_version=http_version,
    )

    assert not _authenticated_body_is_complete_after_stream_error(
        method="GET",
        response=response,
        byte_count=len(body),
        body=body,
        error=EOFError("terminal stream signal missing"),
    )


@pytest.mark.parametrize(
    ("method", "status", "headers", "body"),
    (
        (
            "GET",
            200,
            {"content-type": "application/json", "transfer-encoding": "chunked"},
            b'[{"provider":',
        ),
        (
            "GET",
            200,
            {"content-type": "text/plain", "transfer-encoding": "chunked"},
            b"[]",
        ),
        (
            "GET",
            500,
            {"content-type": "application/json", "transfer-encoding": "chunked"},
            b'{"error":"failed"}',
        ),
        (
            "GET",
            200,
            {
                "content-type": "application/json",
                "content-encoding": "gzip",
                "transfer-encoding": "chunked",
            },
            b"[]",
        ),
        (
            "GET",
            200,
            {
                "content-type": "application/json",
                "content-length": "2",
                "transfer-encoding": "chunked",
            },
            b"[]",
        ),
    ),
)
def test_complete_body_classifier_rejects_ambiguous_chunked_response(
    method,
    status,
    headers,
    body,
):
    response = SimpleNamespace(status_code=status, headers=httpx.Headers(headers))

    assert not _authenticated_body_is_complete_after_stream_error(
        method=method,
        response=response,
        byte_count=len(body),
        body=body,
        error=EOFError("terminal chunk missing"),
    )


def test_httpx_transport_enforces_artifact_specific_large_response_ceiling(
    monkeypatch,
):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            pass

    chunk = b"x" * (1024 * 1024)
    chunk_count = (MAX_RESPONSE_BODY_BYTES // len(chunk)) + 1

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            for _ in range(chunk_count):
                yield chunk

    class ResponseContext:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def close(self):
            pass

        def stream(self, *_args, **_kwargs):
            return ResponseContext()

    class HTTPTransport:
        def __init__(self, **_kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            Client=Client,
            HTTPTransport=HTTPTransport,
            Limits=lambda **kwargs: kwargs,
            Proxy=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    with pytest.raises(ProviderBrokerV2Error, match="response exceeds size limit"):
        HTTPXProviderTransport()(
            method="GET",
            url="https://example.com/artifact",
            headers={},
            body=b"",
            timeout_ms=1000,
        )

    result = HTTPXProviderTransport(
        response_body_ceiling_bytes=MAX_TRANSPORT_RESPONSE_BODY_BYTES
    )(
        method="GET",
        url="https://example.com/artifact",
        headers={},
        body=b"",
        timeout_ms=1000,
        max_response_bytes=MAX_TRANSPORT_RESPONSE_BODY_BYTES,
    )

    assert len(result["body"]) == chunk_count * len(chunk)
    assert len(result["body"]) > MAX_RESPONSE_BODY_BYTES
    assert result["tls_peer_chain_hash"] == sha256_bytes(b"peer-certificate")
    assert result["tls_protocol"] == "TLSv1.3"


def test_httpx_transport_reuses_only_credential_free_direct_client(monkeypatch):
    clients = []
    proxies = []

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            pass

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self, **options):
            self.options = dict(options)
            self.closed = False
            self.stream_calls = []
            clients.append(self)

        def stream(self, *args, **kwargs):
            self.stream_calls.append((args, kwargs))
            return ResponseContext()

        def close(self):
            self.closed = True

    class HTTPTransport:
        def __init__(self, **options):
            self.options = dict(options)
            self.closed = False

        def close(self):
            self.closed = True

    def proxy(url, **options):
        value = {"url": url, **options}
        proxies.append(value)
        return value

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            Client=Client,
            HTTPTransport=HTTPTransport,
            Limits=lambda **kwargs: kwargs,
            Proxy=proxy,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    transport = HTTPXProviderTransport(
        parent_tunnel_framing=TUNNEL_FRAMING_MODE,
        reuse_direct_connections=True,
    )
    request = {
        "method": "GET",
        "url": "https://example.com/artifact",
        "headers": {},
        "body": b"",
        "timeout_ms": 1200,
    }
    errors = []

    def request_direct():
        try:
            transport(**request)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=request_direct) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    assert len(clients) == 1
    assert len(clients[0].stream_calls) == 8
    assert clients[0].stream_calls[0][1]["timeout"] == pytest.approx(
        1.2,
        abs=0.01,
    )
    assert clients[0].closed is False
    assert proxies[0] == {
        "url": "http://127.0.0.1:18080",
        "headers": {TUNNEL_FRAMING_HEADER: TUNNEL_FRAMING_MODE},
    }

    upstream_proxy = "https://worker:test-secret@proxy.example.com:443"
    transport(**request, upstream_proxy_url=upstream_proxy)
    transport(**request, upstream_proxy_url=upstream_proxy)

    assert len(clients) == 3
    assert clients[1].closed is True
    assert clients[2].closed is True
    assert (
        clients[0].options["transport"].options["limits"][
            "max_keepalive_connections"
        ]
        == 32
    )
    assert (
        clients[1].options["transport"].options["limits"][
            "max_keepalive_connections"
        ]
        == 0
    )
    assert (
        clients[2].options["transport"].options["limits"][
            "max_keepalive_connections"
        ]
        == 0
    )
    expected_proxy_header = base64.b64encode(upstream_proxy.encode("utf-8")).decode(
        "ascii"
    )
    assert proxies[1]["headers"] == {
        "X-Leadpoet-Upstream-Proxy-B64": expected_proxy_header,
        TUNNEL_FRAMING_HEADER: TUNNEL_FRAMING_MODE,
    }
    assert proxies[2]["headers"] == proxies[1]["headers"]

    transport.close()
    assert clients[0].closed is True


def test_httpx_transport_reuses_assigned_proxy_only_within_exact_scope(monkeypatch):
    clients = []

    class Client:
        def __init__(self):
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    transport = HTTPXProviderTransport(
        reuse_upstream_proxy_connections=True,
    )
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())
    monkeypatch.setattr(
        transport,
        "_execute_with_client",
        lambda _client, **_kwargs: {
            "http_status": 200,
            "headers": {"content-type": "application/json"},
            "body": b'{"ok":true}',
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        },
    )
    request = {
        "method": "GET",
        "url": "https://api.exa.ai/search",
        "headers": {"Accept-Encoding": "identity"},
        "body": b"",
        "timeout_ms": 1000,
        "upstream_proxy_url": "https://worker:secret@proxy.example.com:443",
    }
    first_scope = sha256_bytes(b"job-1:proxy-1")
    second_scope = sha256_bytes(b"job-2:proxy-1")

    with pytest.raises(ProviderBrokerV2Error, match="scope is required"):
        transport(**request)

    assert transport(**request, connection_scope=first_scope)["body"] == b'{"ok":true}'
    assert transport(**request, connection_scope=first_scope)["body"] == b'{"ok":true}'
    assert transport(**request, connection_scope=second_scope)["body"] == b'{"ok":true}'

    assert len(clients) == 2
    assert clients[0].closed is False
    assert clients[1].closed is False
    transport.release_connection_scope(first_scope)
    assert clients[0].closed is True
    assert clients[1].closed is False

    transport(**request, connection_scope=first_scope)
    assert len(clients) == 3
    assert clients[2].closed is False
    transport.close()
    assert clients[1].closed is True
    assert clients[2].closed is True


def test_httpx_transport_retires_failed_assigned_proxy_generation(monkeypatch):
    clients = []

    class Client:
        def __init__(self):
            self.closed = False
            clients.append(self)

        def close(self):
            self.closed = True

    transport = HTTPXProviderTransport(
        reuse_upstream_proxy_connections=True,
    )
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())

    def execute(client, **_kwargs):
        if client is clients[0]:
            raise RuntimeError("expired assigned-proxy tunnel")
        return {
            "http_status": 200,
            "headers": {},
            "body": b"ok",
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }

    monkeypatch.setattr(transport, "_execute_with_client", execute)
    request = {
        "method": "GET",
        "url": "https://api.exa.ai/search",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
        "upstream_proxy_url": "https://worker:secret@proxy.example.com:443",
        "connection_scope": sha256_bytes(b"job-1:proxy-1"),
    }

    with pytest.raises(RuntimeError, match="expired assigned-proxy tunnel"):
        transport(**request)

    assert len(clients) == 1
    assert clients[0].closed is True
    assert transport(**request)["body"] == b"ok"
    assert len(clients) == 2
    assert clients[1].closed is False
    transport.close()
    assert clients[1].closed is True


def test_httpx_transport_can_frame_upstream_without_framing_direct(monkeypatch):
    proxies = []

    class Client:
        def __init__(self, **options):
            self.options = dict(options)

        def __getitem__(self, name):
            return self.options[name]

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            Client=Client,
            HTTPTransport=lambda **options: options,
            Limits=lambda **options: options,
            Proxy=lambda url, **options: proxies.append(
                {"url": url, **options}
            )
            or proxies[-1],
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    transport = HTTPXProviderTransport(
        upstream_parent_tunnel_framing=TUNNEL_FRAMING_MODE,
    )

    direct_client = transport._new_client()
    retained_direct_client = transport._new_client(
        retain_direct_connections=True,
    )
    request_scoped_proxy_client = transport._new_client(
        proxy_headers={"X-Leadpoet-Upstream-Proxy-B64": "opaque"}
    )
    retained_proxy_client = transport._new_client(
        proxy_headers={"X-Leadpoet-Upstream-Proxy-B64": "opaque"},
        retain_assigned_proxy_connections=True,
    )

    assert proxies == [
        {"url": "http://127.0.0.1:18080", "headers": None},
        {"url": "http://127.0.0.1:18080", "headers": None},
        {
            "url": "http://127.0.0.1:18080",
            "headers": {
                "X-Leadpoet-Upstream-Proxy-B64": "opaque",
                TUNNEL_FRAMING_HEADER: TUNNEL_FRAMING_MODE,
            },
        },
        {
            "url": "http://127.0.0.1:18080",
            "headers": {
                "X-Leadpoet-Upstream-Proxy-B64": "opaque",
                TUNNEL_FRAMING_HEADER: TUNNEL_FRAMING_MODE,
            },
        },
    ]
    assert (
        direct_client["transport"]["limits"]["max_keepalive_connections"]
        == 0
    )
    assert (
        retained_direct_client["transport"]["limits"][
            "max_keepalive_connections"
        ]
        == 32
    )
    assert (
        request_scoped_proxy_client["transport"]["limits"][
            "max_keepalive_connections"
        ]
        == 0
    )
    assert (
        retained_proxy_client["transport"]["limits"][
            "max_keepalive_connections"
        ]
        == 32
    )


@pytest.mark.parametrize(
    ("error_number", "message", "expected_kind"),
    (
        (errno.EADDRNOTAVAIL, "redacted", "ephemeral_port_exhausted"),
        (errno.EMFILE, "redacted", "process_file_descriptor_exhausted"),
        (errno.ENOBUFS, "redacted", "socket_buffer_exhausted"),
        (errno.ENOMEM, "redacted", "memory_exhausted"),
    ),
)
def test_httpx_transport_classifies_chained_local_resource_failures(
    error_number,
    message,
    expected_kind,
):
    underlying = OSError(error_number, message)
    try:
        raise underlying
    except OSError as exc:
        wrapped = httpx.ConnectError("provider request failed")
        wrapped.__cause__ = exc

    assert _failure_code(wrapped) == "proxy_failure"
    assert _local_resource_failure(wrapped) == (error_number, expected_kind)


def test_httpx_transport_health_is_bounded_and_route_specific(monkeypatch):
    class ExplicitTransport:
        def close(self):
            return None

    class Client:
        def __init__(self):
            self._leadpoet_explicit_http_transport = ExplicitTransport()

        def close(self):
            return None

    transport = HTTPXProviderTransport()
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())
    calls = []

    def execute(_client, **_kwargs):
        calls.append(True)
        if len(calls) == 2:
            underlying = OSError(
                errno.EADDRNOTAVAIL,
                "sensitive upstream text",
            )
            raise httpx.ConnectError("sensitive proxy text") from underlying
        return {
            "http_status": 200,
            "headers": {},
            "body": b"ok",
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }

    monkeypatch.setattr(transport, "_execute_with_client", execute)
    request = {
        "method": "GET",
        "url": "https://example.com/data",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
    }

    assert transport(**request)["body"] == b"ok"
    with pytest.raises(httpx.ConnectError):
        transport(
            **request,
            upstream_proxy_url="https://worker:secret@proxy.example.com:443",
        )

    health = transport.health()
    assert health["schema_version"] == PROVIDER_TRANSPORT_HEALTH_SCHEMA_VERSION
    assert health["reuse_direct_connections"] is False
    assert health["reuse_assigned_proxy_connections"] is False
    assert health["direct_active_scope_count"] == 0
    assert health["direct_retired_scope_count"] == 0
    assert health["direct_active_lease_count"] == 0
    assert health["direct_retired_lease_count"] == 0
    assert health["assigned_active_scope_count"] == 0
    assert health["assigned_retired_scope_count"] == 0
    assert health["assigned_active_lease_count"] == 0
    assert health["assigned_retired_lease_count"] == 0
    assert health["request_counters"] == {
        "direct": {"started": 1, "succeeded": 1, "failed": 0},
        "assigned_proxy": {"started": 1, "succeeded": 0, "failed": 1},
    }
    assert health["cleanup_counters"] == {
        "direct": {
            "attempted": 1,
            "succeeded": 1,
            "client_close_failed": 0,
            "transport_close_failed": 0,
        },
        "assigned_proxy": {
            "attempted": 1,
            "succeeded": 1,
            "client_close_failed": 0,
            "transport_close_failed": 0,
        },
    }
    assert health["last_failure"] == {
        "route": "assigned_proxy",
        "stage": "provider_request",
        "failure_code": "proxy_failure",
        "error_type": "ConnectError",
        "errno": errno.EADDRNOTAVAIL,
        "local_resource_kind": "ephemeral_port_exhausted",
    }
    assert "sensitive" not in str(health)
    assert "secret" not in str(health)


def test_httpx_transport_serializes_request_scoped_direct_failure_isolation(
    monkeypatch,
):
    clients = []
    results = []
    errors = []
    activity = {"current": 0, "maximum": 0}
    activity_lock = threading.Lock()

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            pass

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __init__(self, error):
            self.error = error

        def __enter__(self):
            with activity_lock:
                activity["current"] += 1
                activity["maximum"] = max(
                    activity["maximum"],
                    activity["current"],
                )
            time.sleep(0.01)
            if self.error is not None:
                with activity_lock:
                    activity["current"] -= 1
                raise self.error
            return Response()

        def __exit__(self, *_args):
            with activity_lock:
                activity["current"] -= 1
            return False

    class Client:
        def __init__(self, **_options):
            self.index = len(clients)
            self.closed = False
            clients.append(self)

        def stream(self, *_args, **_kwargs):
            error = (
                RuntimeError("relay generation expired")
                if self.index == 0
                else None
            )
            return ResponseContext(error)

        def close(self):
            self.closed = True

    class HTTPTransport:
        def __init__(self, **_options):
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "httpx",
            SimpleNamespace(
                Client=Client,
                HTTPTransport=HTTPTransport,
                Limits=lambda **kwargs: kwargs,
            Proxy=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    transport = HTTPXProviderTransport()
    request = {
        "method": "GET",
        "url": "https://example.com/data",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
    }

    def run_request():
        try:
            results.append(transport(**request))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 1
    assert str(errors[0]) == "relay generation expired"
    assert len(results) == 7
    assert len(clients) == 8
    assert all(client.closed for client in clients)
    assert activity == {"current": 0, "maximum": 1}


def test_httpx_transport_concurrent_assigned_requests_are_isolated(
    monkeypatch,
):
    clients = []
    transports = []
    results = []
    errors = []
    activity = {"current": 0, "maximum": 0}
    activity_lock = threading.Lock()
    entered = threading.Barrier(8)

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            return None

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __init__(self, error):
            self.error = error

        def __enter__(self):
            with activity_lock:
                activity["current"] += 1
                activity["maximum"] = max(
                    activity["maximum"],
                    activity["current"],
                )
            entered.wait(timeout=2.0)
            if self.error is not None:
                with activity_lock:
                    activity["current"] -= 1
                raise self.error
            return Response()

        def __exit__(self, *_args):
            with activity_lock:
                activity["current"] -= 1
            return False

    class Client:
        def __init__(self, **_options):
            self.index = len(clients)
            self.closed = False
            clients.append(self)

        def stream(self, *_args, **_kwargs):
            return ResponseContext(
                RuntimeError("one assigned relay expired")
                if self.index == 0
                else None
            )

        def close(self):
            self.closed = True

    class HTTPTransport:
        def __init__(self, **_options):
            self.closed = False
            transports.append(self)

        def close(self):
            self.closed = True

    monkeypatch.setitem(
        sys.modules,
        "httpx",
        SimpleNamespace(
            Client=Client,
            HTTPTransport=HTTPTransport,
            Limits=lambda **kwargs: kwargs,
            Proxy=lambda *_args, **_kwargs: object(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "certifi",
        SimpleNamespace(where=lambda: "/tmp/test-ca.pem"),
    )

    transport = HTTPXProviderTransport()
    request = {
        "method": "GET",
        "url": "https://api.exa.ai/search",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
        "upstream_proxy_url": "https://worker:secret@proxy.example.com:443",
        "connection_scope": sha256_bytes(b"job-1:proxy-1"),
    }

    def run_request():
        try:
            results.append(transport(**request))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_request) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3.0)

    assert all(not thread.is_alive() for thread in threads)
    assert [str(error) for error in errors] == ["one assigned relay expired"]
    assert len(results) == 7
    assert len(clients) == len(transports) == 8
    assert all(client.closed for client in clients)
    assert all(item.closed for item in transports)
    assert activity["current"] == 0
    assert activity["maximum"] == 8


def test_httpx_request_scoped_direct_slot_wait_is_timeout_bounded():
    transport = HTTPXProviderTransport()
    assert transport._direct_request_slot.acquire(timeout=0.1)
    try:
        with pytest.raises(
            TimeoutError,
            match="direct provider concurrency slot timed out",
        ):
            transport(
                method="GET",
                url="https://example.com/data",
                headers={},
                body=b"",
                timeout_ms=10,
            )
    finally:
        transport._direct_request_slot.release()


def test_job_scoped_client_cleanup_cannot_replace_request_result(monkeypatch):
    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            pass

    class ExplicitTransport:
        def close(self):
            pass

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __enter__(self):
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self):
            self._leadpoet_explicit_http_transport = ExplicitTransport()

        def stream(self, *_args, **_kwargs):
            return ResponseContext()

        def close(self):
            raise EOFError("proxy cleanup lost close_notify")

    transport = HTTPXProviderTransport()
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())

    result = transport(
        method="POST",
        url="https://api.exa.ai/search",
        headers={"Accept-Encoding": "identity"},
        body=b"{}",
        timeout_ms=1000,
        upstream_proxy_url="https://worker:secret@proxy.example.com:443",
    )

    assert result["http_status"] == 200
    assert result["body"] == b'{"ok":true}'
    assert transport.health()["cleanup_counters"]["assigned_proxy"] == {
        "attempted": 1,
        "succeeded": 1,
        "client_close_failed": 1,
        "transport_close_failed": 0,
    }


def test_job_scoped_client_cleanup_preserves_request_failure(monkeypatch):
    class ExplicitTransport:
        def close(self):
            pass

    class Client:
        def __init__(self):
            self._leadpoet_explicit_http_transport = ExplicitTransport()

        def stream(self, *_args, **_kwargs):
            raise RuntimeError("provider request failed")

        def close(self):
            raise EOFError("proxy cleanup also failed")

    transport = HTTPXProviderTransport()
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())

    with pytest.raises(RuntimeError, match="provider request failed"):
        transport(
            method="POST",
            url="https://api.exa.ai/search",
            headers={"Accept-Encoding": "identity"},
            body=b"{}",
            timeout_ms=1000,
            upstream_proxy_url="https://worker:secret@proxy.example.com:443",
        )


def test_httpx_transport_retry_uses_fresh_client_after_direct_failure(monkeypatch):
    clients = []

    class ExplicitTransport:
        def close(self):
            pass

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

        def close(self):
            pass

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __init__(self, error=None):
            self.error = error

        def __enter__(self):
            if self.error is not None:
                raise self.error
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self, error=None):
            self.error = error
            self.closed = False
            self._leadpoet_explicit_http_transport = ExplicitTransport()
            clients.append(self)

        def stream(self, *_args, **_kwargs):
            return ResponseContext(self.error)

        def close(self):
            self.closed = True

    planned = [Client(RuntimeError("stale enclave tunnel")), Client()]
    transport = HTTPXProviderTransport()
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: planned.pop(0))
    request = {
        "method": "GET",
        "url": "https://example.com/data",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
    }

    with pytest.raises(RuntimeError, match="stale enclave tunnel"):
        transport(**request)

    assert clients[0].closed is True
    assert transport(**request)["body"] == b'{"ok":true}'
    assert len(clients) == 2
    transport.close()
    assert clients[1].closed is True


def test_httpx_transport_serializes_and_retires_failed_direct_generation(
    monkeypatch,
):
    first_started = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()

    class TLS:
        def getpeercert(self, binary_form=False, /):
            assert binary_form is True
            return b"peer-certificate"

        def version(self):
            return "TLSv1.3"

    class Stream:
        def get_extra_info(self, name):
            assert name == "ssl_object"
            return TLS()

    class Response:
        status_code = 200
        headers = {"content-type": "application/json"}
        extensions = {"network_stream": Stream()}

        def iter_bytes(self):
            yield b'{"ok":true}'

    class ResponseContext:
        def __init__(self, client_index):
            self.client_index = client_index

        def __enter__(self):
            if self.client_index == 0:
                first_started.set()
                assert release_first.wait(timeout=2)
                raise RuntimeError("stale enclave tunnel")
            second_started.set()
            return Response()

        def __exit__(self, *_args):
            return False

    class Client:
        def __init__(self):
            self.index = len(clients)
            self.closed = False
            clients.append(self)

        def stream(self, *_args, **_kwargs):
            return ResponseContext(self.index)

        def close(self):
            self.closed = True

    clients = []
    transport = HTTPXProviderTransport(reuse_direct_connections=True)
    monkeypatch.setattr(transport, "_new_client", lambda **_kwargs: Client())
    request = {
        "method": "GET",
        "url": "https://example.com/data",
        "headers": {},
        "body": b"",
        "timeout_ms": 1000,
    }
    errors = []

    def run_request():
        try:
            transport(**request)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run_request) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert first_started.wait(timeout=2)
    assert second_started.is_set() is False
    assert len(clients) == 1
    assert clients[0].closed is False
    release_first.set()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert len(errors) == 1
    assert str(errors[0]) == "stale enclave tunnel"
    assert second_started.is_set()
    assert len(clients) == 2
    assert clients[0].closed is True
    assert clients[1].closed is False
    assert transport._direct_client is clients[1]


def test_provider_registry_hash_binds_measured_https_routes():
    document = provider_registry_document()
    assert document["transport"] == {
        "scheme": "https",
        "port": 443,
        "tls_termination": "gateway_coordinator_enclave",
        "plaintext_external_http": False,
        "request_headers": [
            {"name": "Accept-Encoding", "value": "identity"}
        ],
    }
    assert set(document["routes"]) == set(BUILTIN_PROVIDER_ROUTES)
    assert document["routes"]["openrouter"]["hosts"] == ["openrouter.ai"]
    assert document["routes"]["dns"]["hosts"] == ["cloudflare-dns.com"]
    assert document["routes"]["rdap"]["hosts"] == ["rdap.org"]
    assert document["routes"]["wayback"]["hosts"] == [
        "archive.org",
        "web.archive.org",
        "arquivo.pt",
    ]
    assert document["routes"]["wayback"]["path_prefixes"] == [
        "/wayback/available",
        "/cdx/search/cdx",
        "/wayback/cdx",
    ]
    assert document["routes"]["bittensor_chain"]["hosts"] == [
        "entrypoint-finney.opentensor.ai"
    ]
    assert document["routes"]["bittensor_archive"] == {
        "hosts": ["archive.chain.opentensor.ai"],
        "path_prefixes": ["/"],
        "credential_slot": "",
        "credential_location": "none",
        "credential_name": "",
        "credential_prefix": "",
        "credential_header_aliases": [],
        "egress_policy": "job_proxy_allowed",
        "allowed_methods": ["POST"],
    }
    assert document["routes"]["supabase"]["egress_policy"] == "direct_only"
    assert all(
        route["egress_policy"] == "job_proxy_allowed"
        for provider_id, route in document["routes"].items()
        if provider_id != "supabase"
    )
    assert document["routes"]["arweave"] == {
        "hosts": ["arweave.net"],
        "path_prefixes": ["/"],
        "credential_slot": "",
        "credential_location": "none",
        "credential_name": "",
        "credential_prefix": "",
        "credential_header_aliases": [],
        "egress_policy": "job_proxy_allowed",
        "allowed_methods": ["GET"],
    }
    assert document["routes"]["exa"]["request_headers"] == [
        {"name": "Accept-Encoding", "value": "gzip"}
    ]
    assert document["routes"]["supabase"] == {
        "hosts": ["qplwoislplkcegvdmbim.supabase.co"],
        "path_prefixes": ["/rest/v1/"],
        "credential_slot": "supabase_service_role",
        "credential_location": "header",
        "credential_name": "Authorization",
        "credential_prefix": "Bearer ",
        "credential_header_aliases": [{"name": "apikey", "prefix": ""}],
        "egress_policy": "direct_only",
    }
    assert MEASURED_TRANSPORT_REQUEST_HEADERS == (("Accept-Encoding", "identity"),)
    assert provider_registry_hash().startswith("sha256:")
    assert expected_provider_credential_slots() == (
        "deepline",
        "exa",
        "openrouter",
        "scrapingdog",
        "supabase_service_role",
        "truelist",
    )
    assert set(measured_retry_policy_hashes(HASH)) == set(BUILTIN_PROVIDER_ROUTES)
    assert set(expected_job_credential_slot_ref_hashes()) == {"egress_proxy"}
    assert "openrouter_management" not in BUILTIN_PROVIDER_ROUTES


def test_provider_registry_hash_rejects_direct_only_policy_tampering():
    tampered = dict(BUILTIN_PROVIDER_ROUTES)
    tampered["supabase"] = replace(
        tampered["supabase"],
        egress_policy=EGRESS_POLICY_JOB_PROXY_ALLOWED,
    )

    assert provider_registry_hash(tampered) != provider_registry_hash()


class FakeTransport:
    def __init__(self, *, error=None, delay=0.0):
        self.calls = []
        self.error = error
        self.delay = delay

    def __call__(self, **request):
        self.calls.append(request)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return {
            "http_status": 503,
            "headers": {"content-type": "application/json", "set-cookie": "ignored"},
            "body": b'{"error":"provider unavailable"}',
            "tls_peer_chain_hash": "sha256:" + "b" * 64,
            "tls_protocol": "TLSv1.3",
        }


def _broker(transport, **broker_kwargs):
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: NOW,
        **broker_kwargs,
    )
    broker.provision_credentials(credentials)
    return broker


def test_implicit_provider_transports_are_broker_scoped():
    first = _broker(None)
    second = _broker(None)

    assert first._transport is not second._transport


def _request(**overrides):
    request = {
        "schema_version": PROVIDER_BROKER_SCHEMA_VERSION,
        "logical_operation_id": "operation-1",
        "job_id": "job-1",
        "purpose": "research_lab.provider_evidence.v2",
        "provider_id": "openrouter",
        "attempt_number": 0,
        "method": "POST",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "headers": {"content-type": "application/json", "x-title": "Leadpoet"},
        "body_b64": base64.b64encode(b'{"model":"model-1"}').decode("ascii"),
        "timeout_ms": 30000,
        "retry_policy_hash": HASH,
    }
    request.update(overrides)
    return request


def test_authenticated_provider_error_is_recorded_only_with_tls_evidence():
    transport = FakeTransport()
    result = _broker(transport).execute(_request())
    assert result["terminal_status"] == "authenticated_response"
    assert result["http_status"] == 503
    attempt = result["transport_attempt"]
    validate_transport_attempt(attempt)
    assert attempt["terminal_status"] == "authenticated_response"
    assert attempt["http_status"] == 503
    assert attempt["tls_protocol"] == "TLSv1.3"
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer openrouter-secret"
    assert transport.calls[0]["headers"]["Accept-Encoding"] == "identity"


@pytest.mark.parametrize(
    ("provider_id", "url"),
    (
        ("exa", "https://api.exa.ai/search"),
        ("openrouter", "https://openrouter.ai/api/v1/chat/completions"),
        ("scrapingdog", "https://api.scrapingdog.com/scrape?url=test"),
    ),
)
def test_measured_transport_binds_provider_response_encoding(
    provider_id,
    url,
):
    transport = FakeTransport()
    broker = _broker(transport)

    broker.execute(
        _request(
            provider_id=provider_id,
            url=url,
            headers={
                "content-type": "application/json",
                "accept-encoding": "gzip, deflate, br",
            },
        )
    )

    outbound_headers = transport.calls[0]["headers"]
    expected_encoding = "gzip" if provider_id == "exa" else "identity"
    assert outbound_headers["Accept-Encoding"] == expected_encoding
    assert "accept-encoding" not in outbound_headers


def test_parent_or_network_error_cannot_masquerade_as_provider_status():
    transport = FakeTransport(error=RuntimeError("proxy generated 502"))
    result = _broker(transport).execute(_request())
    assert result == {
        "terminal_status": "transport_failure",
        "call_count": 1,
        "failure_code": "proxy_failure",
        "failure_stage": "provider_transport",
        "failure_error_type": "RuntimeError",
        "encrypted_request_artifact_id": result[
            "encrypted_request_artifact_id"
        ],
        "transport_attempt": result["transport_attempt"],
        "evidence_artifact_hashes": result["evidence_artifact_hashes"],
    }
    attempt = result["transport_attempt"]
    validate_transport_attempt(attempt)
    assert attempt["http_status"] is None
    assert attempt["response_hash"] is None
    assert attempt["request_artifact_hash"].startswith("sha256:")
    assert attempt["failure_code"] == "proxy_failure"


def test_oversized_response_is_a_canonical_terminal_and_releases_inflight():
    transport = FakeTransport(
        error=ProviderBrokerV2Error("provider response exceeds size limit")
    )
    broker = _broker(transport)

    result = broker.execute(_request())

    assert result["terminal_status"] == "transport_failure"
    assert result["failure_code"] == "response_too_large"
    assert result["transport_attempt"]["failure_code"] == "response_too_large"
    validate_transport_attempt(result["transport_attempt"])
    assert broker.health()["inflight_count"] == 0


def test_transport_failure_uses_only_safe_error_type_projection():
    class UnsafeTypeNameError(RuntimeError):
        pass

    UnsafeTypeNameError.__name__ = "unsafe-type-name"
    broker = _broker(FakeTransport(error=UnsafeTypeNameError("secret")))

    result = broker.execute(_request())

    assert result["failure_error_type"] == "Exception"


def test_transport_cleanup_diagnostic_rejects_invalid_internal_stage():
    cleanup_error = ProviderTransportCleanupError(
        stage="untrusted_cleanup_stage",
        primary_error=ValueError("secret primary"),
        cleanup_error=OSError(errno.EIO, "secret cleanup"),
    )

    with pytest.raises(ProviderBrokerV2Error, match="stage is invalid"):
        _provider_transport_failure_diagnostic(
            provider="openrouter",
            request_hash=HASH,
            attempt_number=0,
            exc=cleanup_error,
        )


def test_diagnostic_artifact_failure_releases_owner_and_allows_retry():
    transport = FakeTransport(error=RuntimeError("provider transport failed"))
    broker = _broker(transport)
    artifact_sink = broker._artifact_sink

    def fail_diagnostic(body, **kwargs):
        if kwargs.get("artifact_kind") == (
            "provider_transport_failure_diagnostic"
        ):
            raise OSError("diagnostic vault unavailable")
        return artifact_sink(body, **kwargs)

    broker._artifact_sink = fail_diagnostic
    with pytest.raises(OSError, match="diagnostic vault unavailable"):
        broker.execute(_request())

    assert broker.health()["inflight_count"] == 0
    assert broker.health()["terminal_count"] == 0
    assert broker._inflight == {}

    broker._artifact_sink = artifact_sink
    result = broker.execute(_request())
    assert result["terminal_status"] == "transport_failure"
    assert broker.health()["inflight_count"] == 0
    assert broker.health()["terminal_count"] == 1
    assert len(transport.calls) == 2


def test_request_artifact_failure_does_not_poison_logical_attempt_retry():
    transport = FakeTransport()
    broker = _broker(transport)
    artifact_sink = broker._artifact_sink
    calls = 0

    def fail_once(body, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("artifact vault capacity is full")
        return artifact_sink(body, **kwargs)

    broker._artifact_sink = fail_once

    with pytest.raises(RuntimeError, match="capacity is full"):
        broker.execute(_request())
    assert broker.health()["inflight_count"] == 0

    result = broker.execute(_request())
    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 1


def test_owner_failure_releases_waiters_and_allows_safe_pretransport_retry():
    transport = FakeTransport()
    broker = _broker(transport)
    owner_entered = threading.Event()
    release_owner = threading.Event()

    def blocking_clock():
        owner_entered.set()
        assert release_owner.wait(2)
        raise RuntimeError("measured clock unavailable")

    broker._clock = blocking_clock
    errors = []

    def _call():
        try:
            broker.execute(_request())
        except Exception as exc:
            errors.append(exc)

    owner = threading.Thread(target=_call)
    owner.start()
    assert owner_entered.wait(2)

    waiter_entered = threading.Event()

    class ObservedWaitEvent:
        def __init__(self):
            self._event = threading.Event()

        def wait(self, timeout):
            waiter_entered.set()
            return self._event.wait(timeout)

        def set(self):
            self._event.set()

    with broker._lock:
        inflight = broker._inflight[("operation-1", 0)]
        broker._inflight[("operation-1", 0)] = (
            inflight[0],
            ObservedWaitEvent(),
            inflight[2],
        )

    waiter = threading.Thread(target=_call)
    waiter.start()
    assert waiter_entered.wait(2)
    release_owner.set()
    owner.join(2)
    waiter.join(2)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert sorted(str(exc) for exc in errors) == [
        "duplicate provider attempt did not terminate",
        "measured clock unavailable",
    ]
    assert transport.calls == []
    assert broker.health()["inflight_count"] == 0

    broker._clock = lambda: NOW
    result = broker.execute(_request())
    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 1


def test_one_logical_attempt_is_executed_and_charged_once_under_concurrency():
    transport = FakeTransport(delay=0.05)
    broker = _broker(transport)
    results = []
    errors = []

    def _call():
        try:
            results.append(broker.execute(_request()))
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=_call) for _ in range(10)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(transport.calls) == 1
    assert len(results) == 10
    assert {item["transport_attempt"]["attempt_hash"] for item in results} == {
        results[0]["transport_attempt"]["attempt_hash"]
    }


def test_attempt_id_cannot_be_reused_for_different_request():
    broker = _broker(FakeTransport())
    broker.execute(_request())
    changed_body = base64.b64encode(b'{"model":"different"}').decode("ascii")
    with pytest.raises(ProviderBrokerV2Error, match="different request"):
        broker.execute(_request(body_b64=changed_body))


def test_plaintext_unmeasured_destination_and_runner_credentials_fail_closed():
    broker = _broker(FakeTransport())
    with pytest.raises(ProviderBrokerV2Error, match="HTTPS"):
        broker.execute(_request(url="http://openrouter.ai/api/v1/chat/completions"))
    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(_request(url="https://example.com/api/v1/chat/completions"))
    with pytest.raises(ProviderBrokerV2Error, match="credential header"):
        broker.execute(_request(headers={"Authorization": "Bearer host-value"}))


def test_historical_settlement_routes_are_host_and_method_bound():
    broker = _broker(FakeTransport())
    broker.execute(
        _request(
            provider_id="bittensor_archive",
            method="POST",
            url="https://archive.chain.opentensor.ai/",
        )
    )
    broker.execute(
        _request(
            logical_operation_id="operation-arweave",
            provider_id="arweave",
            method="GET",
            url="https://arweave.net/" + "A" * 43,
            body_b64=base64.b64encode(b"").decode("ascii"),
        )
    )
    with pytest.raises(ProviderBrokerV2Error, match="method"):
        broker.execute(
            _request(
                logical_operation_id="operation-archive-get",
                provider_id="bittensor_archive",
                method="GET",
                url="https://archive.chain.opentensor.ai/",
            )
        )
    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(
            _request(
                logical_operation_id="operation-arweave-host",
                provider_id="arweave",
                method="GET",
                url="https://example.com/" + "A" * 43,
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )


def test_scrapingdog_key_is_injected_only_inside_coordinator_query():
    transport = FakeTransport()
    broker = _broker(transport)
    broker.execute(
        _request(
            provider_id="scrapingdog",
            url="https://api.scrapingdog.com/scrape?dynamic=true",
        )
    )
    assert "api_key=scrapingdog-secret" in transport.calls[0]["url"]
    attempt = broker.execute(
        _request(
            logical_operation_id="operation-2",
            provider_id="scrapingdog",
            url="https://api.scrapingdog.com/scrape?dynamic=true",
        )
    )["transport_attempt"]
    assert "scrapingdog-secret" not in str(attempt)


def test_source_add_provenance_static_route_uses_bounded_hash_only_artifacts():
    transport = FakeTransport()
    artifact_bodies = []
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }

    def sink(body, **_kwargs):
        artifact_bodies.append(bytes(body))
        return {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        }

    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=transport,
        artifact_sink=sink,
        clock=lambda: NOW,
    )
    broker.provision_credentials(credentials)
    result = broker.execute(
        _request(
            purpose="research_lab.source_add_provenance.v2",
            provider_id="scrapingdog",
            method="GET",
            url=(
                "https://api.scrapingdog.com/scrape?"
                "url=https%3A%2F%2Fdocs.example.com&dynamic=false"
            ),
            body_b64="",
            max_response_bytes=240_000,
            artifact_mode="hash_only",
        )
    )

    assert transport.calls[0]["max_response_bytes"] == 240_000
    assert b'"error":"provider unavailable"' not in artifact_bodies
    assert all(b"docs.example.com" not in body for body in artifact_bodies)
    assert result["transport_attempt"]["response_hash"].startswith("sha256:")


def test_source_add_archive_fallback_is_an_exact_measured_route():
    transport = FakeTransport()
    broker = _broker(transport)
    request = _request(
        purpose="research_lab.source_add_provenance.v2",
        provider_id="wayback",
        method="GET",
        url=(
            "https://arquivo.pt/wayback/cdx?"
            "url=api.example.com%2F%2A&output=json&"
            "filter=statuscode%3A200&limit=1"
        ),
        body_b64="",
        max_response_bytes=240_000,
        artifact_mode="hash_only",
    )

    result = broker.execute(request)

    assert transport.calls[0]["url"] == request["url"]
    assert result["transport_attempt"]["provider_id"] == "wayback"
    cdx_request = {
        **request,
        "logical_operation_id": "operation-cdx",
        "url": (
            "https://web.archive.org/cdx/search/cdx?"
            "url=api.example.com%2F%2A&output=json&"
            "filter=statuscode%3A200&limit=1&fl=timestamp"
        ),
    }
    cdx_result = broker.execute(cdx_request)
    assert transport.calls[1]["url"] == cdx_request["url"]
    assert cdx_result["transport_attempt"]["provider_id"] == "wayback"
    with pytest.raises(ProviderBrokerV2Error, match="path differs"):
        broker.execute(
            {
                **request,
                "logical_operation_id": "operation-2",
                "url": "https://arquivo.pt/textsearch?maxItems=1",
            }
        )


def test_unrelated_static_route_cannot_request_hash_only_artifacts():
    broker = _broker(FakeTransport())
    with pytest.raises(ProviderBrokerV2Error, match="measured SOURCE_ADD route"):
        broker.execute(
            _request(
                max_response_bytes=240_000,
                artifact_mode="hash_only",
            )
        )


def test_supabase_service_role_is_injected_only_for_measured_project():
    transport = FakeTransport()
    broker = _broker(transport)
    result = broker.execute(
        _request(
            provider_id="supabase",
            method="GET",
            url=(
                "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                "banned_hotkeys?select=hotkey"
            ),
            body_b64=base64.b64encode(b"").decode("ascii"),
        )
    )
    outbound = transport.calls[0]
    assert outbound["headers"]["Authorization"] == (
        "Bearer supabase-service-role-secret"
    )
    assert outbound["headers"]["apikey"] == "supabase-service-role-secret"
    assert outbound["headers"]["Accept-Encoding"] == "identity"
    assert outbound.get("allow_http2", True) is True
    assert "supabase-service-role-secret" not in str(result)

    broker.execute(
        _request(
            logical_operation_id="operation-supabase-encoding",
            provider_id="supabase",
            method="GET",
            url=(
                "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                "research_lab_provider_registry?select=registry_hash"
            ),
            headers={"accept-encoding": "br"},
            body_b64=base64.b64encode(b"").decode("ascii"),
        )
    )
    encoded_headers = transport.calls[1]["headers"]
    assert encoded_headers["Accept-Encoding"] == "identity"
    assert "accept-encoding" not in encoded_headers

    with pytest.raises(ProviderBrokerV2Error, match="destination"):
        broker.execute(
            _request(
                logical_operation_id="operation-wrong-project",
                provider_id="supabase",
                method="GET",
                url="https://attacker.supabase.co/rest/v1/banned_hotkeys?select=hotkey",
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )
    with pytest.raises(ProviderBrokerV2Error, match="credential header"):
        broker.execute(
            _request(
                logical_operation_id="operation-host-key",
                provider_id="supabase",
                method="GET",
                url=(
                    "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
                    "banned_hotkeys?select=hotkey"
                ),
                headers={"apikey": "host-supplied"},
                body_b64=base64.b64encode(b"").decode("ascii"),
            )
        )


def test_kms_unwrapped_slots_are_provisioned_individually_and_immutably():
    credentials = {
        "openrouter": "openrouter-secret",
        "exa": "exa-secret",
        "scrapingdog": "scrapingdog-secret",
        "deepline": "deepline-secret",
        "supabase_service_role": "supabase-service-role-secret",
        "truelist": "truelist-secret",
    }
    broker = ProviderBrokerV2(
        credential_ref_hashes={
            name: credential_reference_hash(value)
            for name, value in credentials.items()
        },
        retry_policy_hashes={name: HASH for name in BUILTIN_PROVIDER_ROUTES},
        transport=FakeTransport(),
        artifact_sink=lambda body, **_: {
            "artifact_id": sha256_bytes(b"artifact:" + body),
            "plaintext_hash": sha256_bytes(body),
        },
        clock=lambda: NOW,
    )
    status = broker.provision_credential(
        slot="openrouter",
        credential=credentials["openrouter"],
    )
    assert status["status"] == "provisioning"
    assert status["credential_slots"] == ["openrouter"]
    assert broker.provision_credential(
        slot="openrouter",
        credential=credentials["openrouter"],
    )["status"] == "provisioning"
    with pytest.raises(ProviderBrokerV2Error, match="hash mismatch"):
        broker.provision_credential(slot="openrouter", credential="wrong-secret")


def test_job_credential_lease_overrides_boot_key_for_only_that_job():
    transport = FakeTransport()
    broker = _broker(transport)
    miner_key = "miner-owned-openrouter-key"
    lease = broker.provision_job_credential(
        job_id="job-1",
        slot="openrouter",
        credential=miner_key,
        credential_value_hash_expected=credential_value_hash(miner_key),
    )
    assert lease["status"] == "ready"
    result = broker.execute(_request())
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer " + miner_key
    assert result["transport_attempt"]["credential_ref_hash"] == credential_value_hash(
        miner_key
    )

    released = broker.release_job_credentials("job-1")
    assert released["released_slot_count"] == 1
    broker.execute(
        _request(
            job_id="job-2",
            logical_operation_id="operation-2",
        )
    )
    assert transport.calls[1]["headers"]["Authorization"] == "Bearer openrouter-secret"


def test_completed_job_release_removes_only_its_terminal_records():
    transport = FakeTransport()
    broker = _broker(transport)
    first_request = _request()
    broker.execute(first_request)
    broker.execute(first_request)
    broker.execute(
        _request(
            job_id="job-2",
            logical_operation_id="operation-2",
        )
    )

    assert len(transport.calls) == 2
    assert broker.health()["terminal_count"] == 2

    released = broker.release_job_credentials("job-1")

    assert released["released_terminal_count"] == 1
    health = broker.health()
    assert health["terminal_count"] == 1
    assert health["released_terminal_count"] == 1


def test_failed_artifact_transaction_removes_terminal_record_before_retry():
    transport = FakeTransport()
    broker = _broker(transport)
    request = _request()

    with pytest.raises(RuntimeError, match="checkpoint failed"):
        with broker.transient_terminal_transaction():
            broker.execute(request)
            raise RuntimeError("checkpoint failed")

    rolled_back_health = broker.health()
    assert rolled_back_health["terminal_count"] == 0
    assert rolled_back_health["provider_terminal_counts"]["openrouter"][
        "direct"
    ] == {"authenticated_response": 0, "transport_failure": 0}
    assert rolled_back_health["provider_2xx_success_counts"]["openrouter"] == {
        "direct": 0,
        "assigned_proxy": 0,
    }

    result = broker.execute(request)

    assert result["terminal_status"] == "authenticated_response"
    assert len(transport.calls) == 2
    committed_health = broker.health()
    assert committed_health["terminal_count"] == 1
    assert committed_health["provider_terminal_counts"]["openrouter"][
        "direct"
    ] == {"authenticated_response": 1, "transport_failure": 0}
    # FakeTransport returns an authenticated 503, which must never count as a
    # healthy physical post-idle provider success.
    assert committed_health["provider_2xx_success_counts"]["openrouter"] == {
        "direct": 0,
        "assigned_proxy": 0,
    }


def test_chain_weight_health_counts_only_committed_direct_2xx_supabase():
    class HealthyTransport(FakeTransport):
        def __call__(self, **request):
            self.calls.append(request)
            return {
                "http_status": 200,
                "headers": {"content-type": "application/json"},
                "body": b'{"rows":[]}',
                "tls_peer_chain_hash": "sha256:" + "b" * 64,
                "tls_protocol": "TLSv1.3",
            }

    transport = HealthyTransport()
    broker = _broker(transport)
    proxy_url = "https://worker:test-secret@proxy.example.com:443"
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=credential_value_hash(proxy_url),
    )
    request = _request(
        provider_id="supabase",
        method="GET",
        url=(
            "https://qplwoislplkcegvdmbim.supabase.co/"
            "rest/v1/research_lab_chain_weight_observations"
        ),
        purpose="research_lab.chain_weight_observation.v1",
        body_b64=base64.b64encode(b"").decode("ascii"),
    )

    with broker.transient_terminal_transaction():
        result = broker.execute(request)

    assert result["terminal_status"] == "authenticated_response"
    assert "upstream_proxy_url" not in transport.calls[0]
    health = broker.health()
    assert health["provider_2xx_success_counts"]["supabase"] == {
        "direct": 1,
        "assigned_proxy": 0,
    }
    assert health["chain_weight_observation_success_count"] == 1


@pytest.mark.parametrize("commit", [False, True])
def test_concurrent_duplicate_cannot_read_uncommitted_terminal_record(commit):
    transport = FakeTransport()
    broker = _broker(transport)
    request = _request()
    owner_ready = threading.Event()
    release_owner = threading.Event()
    waiter_entered = threading.Event()
    owner_results = []
    waiter_results = []
    waiter_errors = []
    observed_wait_timeouts = []

    class ObservedWaitEvent:
        def __init__(self):
            self._event = threading.Event()

        def wait(self, timeout):
            waiter_entered.set()
            observed_wait_timeouts.append(timeout)
            return self._event.wait(timeout)

        def set(self):
            self._event.set()

    def _owner():
        try:
            with broker.transient_terminal_transaction():
                owner_results.append(broker.execute(request))
                owner_ready.set()
                assert release_owner.wait(2)
                if not commit:
                    raise RuntimeError("checkpoint failed")
        except RuntimeError:
            if commit:
                raise

    def _waiter():
        try:
            waiter_results.append(broker.execute(request))
        except Exception as exc:
            waiter_errors.append(exc)

    owner = threading.Thread(target=_owner)
    owner.start()
    assert owner_ready.wait(2)
    with broker._lock:
        inflight = broker._inflight[("operation-1", 0)]
        broker._inflight[("operation-1", 0)] = (
            inflight[0],
            ObservedWaitEvent(),
            inflight[2],
        )

    waiter = threading.Thread(target=_waiter)
    waiter.start()
    assert waiter_entered.wait(2)
    assert not waiter_results
    release_owner.set()
    owner.join(2)
    waiter.join(2)

    assert not owner.is_alive()
    assert not waiter.is_alive()
    assert observed_wait_timeouts == [REPLAY_WAIT_SECONDS]
    assert len(owner_results) == 1
    if commit:
        assert len(waiter_results) == 1
        assert waiter_errors == []
        assert len(transport.calls) == 1
        assert broker.health()["terminal_count"] == 1
    else:
        assert waiter_results == []
        assert [str(exc) for exc in waiter_errors] == [
            "duplicate provider attempt did not terminate"
        ]
        assert broker.health()["terminal_count"] == 0
        assert broker.execute(request)["terminal_status"] == "authenticated_response"
        assert len(transport.calls) == 2


def test_terminal_record_cleanup_prevents_completed_jobs_exhausting_capacity(
    monkeypatch,
):
    monkeypatch.setattr(
        "gateway.tee.provider_broker_v2.MAX_DEDUPLICATION_RECORDS",
        2,
    )
    transport = FakeTransport()
    broker = _broker(transport)

    for index in range(5):
        job_id = "job-%d" % index
        broker.execute(
            _request(
                job_id=job_id,
                logical_operation_id="operation-%d" % index,
            )
        )
        released = broker.release_job_credentials(job_id)
        assert released["released_terminal_count"] == 1

    assert len(transport.calls) == 5
    assert broker.health()["terminal_count"] == 0


def test_abandoned_terminal_records_expire_before_capacity_is_reused(monkeypatch):
    monkeypatch.setattr(
        "gateway.tee.provider_broker_v2.MAX_DEDUPLICATION_RECORDS",
        1,
    )
    now = [100.0]
    transport = FakeTransport()
    broker = _broker(transport, monotonic_clock=lambda: now[0])
    broker.execute(_request())

    now[0] += 3601.0
    broker.execute(
        _request(
            job_id="job-2",
            logical_operation_id="operation-2",
        )
    )

    health = broker.health()
    assert health["terminal_count"] == 1
    assert health["expired_terminal_count"] == 1


@pytest.mark.parametrize(
    "proxy_url",
    (
        "https://worker-7:password@proxy.example.com:443",
        "https://worker-7:password@proxy.example.com:8443",
        "http://worker-7:password@proxy.example.com:6162",
    ),
)
def test_job_scoped_proxy_is_bound_to_transport_receipt(proxy_url):
    class ScopedTransport(FakeTransport):
        def __init__(self):
            super().__init__()
            self.released_connection_scopes = []

        def release_connection_scope(self, connection_scope):
            self.released_connection_scopes.append(connection_scope)

    transport = ScopedTransport()
    broker = _broker(transport)
    proxy_hash = credential_value_hash(proxy_url)
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=proxy_hash,
    )

    result = broker.execute(_request())
    references = broker.transport_reference_hashes(_request())
    expected_scope = broker._proxy_connection_scope("job-1", proxy_hash)

    assert transport.calls[0]["upstream_proxy_url"] == proxy_url
    assert transport.calls[0]["connection_scope"] == expected_scope
    assert result["transport_attempt"]["egress_proxy_ref_hash"] == proxy_hash
    assert references == {
        "credential_ref_hash": result["transport_attempt"][
            "credential_ref_hash"
        ],
        "egress_proxy_ref_hash": proxy_hash,
    }
    validate_transport_attempt(result["transport_attempt"])
    assert proxy_url not in str(result)
    assert broker.release_job_credentials("job-1")["released_slot_count"] == 1
    assert transport.released_connection_scopes == [expected_scope]


def test_supabase_direct_only_ignores_job_proxy_in_selection_and_references():
    transport = FakeTransport()
    broker = _broker(transport)
    proxy_url = "https://worker:secret@proxy.example.com:443"
    proxy_hash = credential_value_hash(proxy_url)
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=proxy_hash,
    )
    request = _request(
        provider_id="supabase",
        method="GET",
        url=(
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
            "research_lab_rebenchmark_controls?select=sequence"
        ),
        body_b64="",
        logical_operation_id="supabase-direct-only",
    )

    references = broker.transport_reference_hashes(request)
    result = broker.execute(request)

    assert references["egress_proxy_ref_hash"] == DIRECT_EGRESS_REF_HASH
    assert (
        result["transport_attempt"]["egress_proxy_ref_hash"]
        == DIRECT_EGRESS_REF_HASH
    )
    assert "upstream_proxy_url" not in transport.calls[0]
    assert "connection_scope" not in transport.calls[0]
    assert proxy_hash not in str(result)


def test_transport_reference_hashes_cannot_be_supplied_or_tampered_by_caller():
    transport = FakeTransport()
    broker = _broker(transport)
    request = _request(
        provider_id="supabase",
        method="GET",
        url=(
            "https://qplwoislplkcegvdmbim.supabase.co/rest/v1/"
            "research_lab_rebenchmark_controls?select=sequence"
        ),
        body_b64="",
        logical_operation_id="supabase-ref-tamper",
    )

    with pytest.raises(
        ProviderBrokerV2Error,
        match="request fields are invalid",
    ):
        broker.execute(
            {
                **request,
                "egress_proxy_ref_hash": "sha256:" + "f" * 64,
            }
        )

    assert transport.calls == []


@pytest.mark.parametrize(
    "proxy_url",
    (
        "https://user@proxy.example.com:443",
        "socks5://worker:secret@proxy.example.com:6162",
        "http://worker:secret@proxy.example.com:0",
    ),
)
def test_job_scoped_proxy_rejects_invalid_or_incomplete_routes(proxy_url):
    broker = _broker(FakeTransport())
    broker.provision_job_credential(
        job_id="job-1",
        slot="egress_proxy",
        credential=proxy_url,
        credential_value_hash_expected=credential_value_hash(proxy_url),
    )

    with pytest.raises(ProviderBrokerV2Error, match="proxy"):
        broker.execute(_request())

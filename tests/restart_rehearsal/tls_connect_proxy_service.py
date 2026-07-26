#!/usr/bin/env python3
"""Local authenticated TLS CONNECT boundary for the exact restart rehearsal."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import signal
import socket
import ssl
import threading

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


STATE_ROOT = Path(os.environ.get("REHEARSAL_STATE_ROOT", "/rehearsal-state"))
CA_PATH = STATE_ROOT / "tls-connect-proxy-ca.pem"
CERT_PATH = STATE_ROOT / "tls-connect-proxy-cert.pem"
KEY_PATH = STATE_ROOT / "tls-connect-proxy-key.pem"
READY_PATH = STATE_ROOT / "tls-connect-proxy.ready"
EVENT_PATH = STATE_ROOT / "events.jsonl"
HOST = "127.0.0.1"
PORT = int(os.environ.get("REHEARSAL_TLS_CONNECT_PROXY_PORT", "18443"))
PROXY_HOSTS = (
    "autoresearch-proxy.example.com",
    "scoring-proxy.example.com",
)
PROXY_IP = "93.184.216.34"
EXPECTED_CREDENTIALS = {
    "rehearsal-auto": "rehearsal-auto-password",
    "rehearsal-scoring": "rehearsal-scoring-password",
}
ALLOWED_TARGETS = {
    "openrouter.ai:443",
    "api.exa.ai:443",
    "api.scrapingdog.com:443",
    "code.deepline.com:443",
}
_STOP = threading.Event()
_EVENT_LOCK = threading.Lock()


def _write_event(**details: object) -> None:
    row = {
        "kind": "tls-connect-proxy",
        "status": "ok",
        "boundary": "http_service",
        "operation": "proxy_connect",
        "implementation": "external_boundary",
        "fixture_authenticity": "production_shaped_sanitized",
        "reject_unknown": True,
        **details,
    }
    with _EVENT_LOCK:
        with EVENT_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_certificates() -> None:
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "Leadpoet Restart Rehearsal CA")]
    )
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )

    server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    server_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, PROXY_HOSTS[0])]
    )
    server_cert = (
        x509.CertificateBuilder()
        .subject_name(server_name)
        .issuer_name(ca_name)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName(host) for host in PROXY_HOSTS]
                + [x509.IPAddress(ipaddress.ip_address(PROXY_IP))]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )

    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    CA_PATH.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    CERT_PATH.write_bytes(server_cert.public_bytes(serialization.Encoding.PEM))
    KEY_PATH.write_bytes(
        server_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    for path in (CA_PATH, CERT_PATH):
        path.chmod(0o644)
    KEY_PATH.chmod(0o600)


def _read_headers(connection: ssl.SSLSocket) -> bytes:
    value = bytearray()
    while b"\r\n\r\n" not in value:
        chunk = connection.recv(4096)
        if not chunk:
            raise ValueError("CONNECT request ended before headers")
        value.extend(chunk)
        if len(value) > 16384:
            raise ValueError("CONNECT request headers exceed limit")
    headers, remainder = bytes(value).split(b"\r\n\r\n", 1)
    if remainder:
        raise ValueError("CONNECT request included an unexpected payload")
    return headers


def _authorized(headers: list[str]) -> bool:
    expected_tokens = {
        "Basic "
        + base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
        for user, password in EXPECTED_CREDENTIALS.items()
    }
    authorization = [
        line.partition(":")[2].strip()
        for line in headers[1:]
        if line.lower().startswith("proxy-authorization:")
    ]
    return len(authorization) == 1 and authorization[0] in expected_tokens


def _handle(raw_connection: socket.socket, context: ssl.SSLContext) -> None:
    with raw_connection:
        try:
            with context.wrap_socket(raw_connection, server_side=True) as connection:
                connection.settimeout(10)
                lines = _read_headers(connection).decode("iso-8859-1").split("\r\n")
                parts = lines[0].split(" ")
                if len(parts) != 3 or parts[0] != "CONNECT":
                    raise ValueError("unexpected proxy request")
                target = parts[1]
                if target not in ALLOWED_TARGETS:
                    raise ValueError("unexpected CONNECT destination")
                if not _authorized(lines):
                    connection.sendall(
                        b"HTTP/1.1 407 Proxy Authentication Required\r\n\r\n"
                    )
                    return
                connection.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                _write_event(target=target, authenticated=True)
        except (OSError, ssl.SSLError, ValueError):
            return


def main() -> int:
    _write_certificates()
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(str(CERT_PATH), str(KEY_PATH))
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, PORT))
    listener.listen(16)
    listener.settimeout(0.2)
    READY_PATH.write_text("ready\n", encoding="ascii")
    READY_PATH.chmod(0o600)

    def stop(_signum: int, _frame: object) -> None:
        _STOP.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while not _STOP.is_set():
            try:
                connection, _address = listener.accept()
            except socket.timeout:
                continue
            threading.Thread(
                target=_handle,
                args=(connection, context),
                daemon=True,
            ).start()
    finally:
        listener.close()
        READY_PATH.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

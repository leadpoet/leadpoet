"""Real pinned nginx to a private HTTPS wire stub; no source role inference.

Actual managed-source key/role parity is a separate cold production read gate.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time
from uuid import uuid4

import httpx
import pytest

from scripts import aws_subnet_api as api
from tests.test_aws_subnet_api import secrets
from tests.lab_arena.test_lab_arena_postgrest_route import mint_jwt


def _docker(*args):
    result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, "local nginx container operation failed"
    return result.stdout.strip()


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _certificate(directory, name, hostname):
    key, certificate = directory / (name + ".key"), directory / (name + ".pem")
    result = subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-days", "1", "-subj", "/CN=" + hostname, "-addext", "subjectAltName=DNS:" + hostname,
        "-keyout", str(key), "-out", str(certificate)], capture_output=True, timeout=20)
    assert result.returncode == 0, "local test certificate creation failed"
    return certificate, key


@contextmanager
def _proxy(directory, *, failure=None, echo_location=False):
    assert shutil.which("docker") and shutil.which("openssl"), "nginx integration tools required"
    directory.mkdir(mode=0o700)
    trusted, trusted_key = _certificate(directory, "trusted", api.SOURCE_REST_HOST)
    other, other_key = _certificate(directory, "other", "wrong-source.invalid")
    certificate, key = (other, other_key) if failure == "wrong_host" else (trusted, trusted_key)
    ca = certificate if failure != "untrusted" else other
    records, server_names = [], []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            records.append((self.command, self.path, dict(self.headers), self.rfile.read(size)))
            payload = b'{"code":"42501","message":"wire contract"}'
            self.send_response(403)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Content-Range", "0-1/2")
            self.send_header("X-Source-Receipt", "unchanged")
            if echo_location:
                self.send_header("Content-Location", self.path)
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

        def log_message(self, *_args):
            pass

    # Bind the Docker host's private interface on Linux. Desktop forwards its
    # host.docker.internal address to the host's loopback listener.
    bind = "127.0.0.1" if sys.platform == "darwin" else _docker(
        "network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}")
    server = ThreadingHTTPServer((bind, 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, key)
    context.set_servername_callback(lambda _socket, name, _context: server_names.append(name))
    server.socket = context.wrap_socket(server.socket, server_side=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    port = _free_port()
    name = "subnet-source-nginx-" + uuid4().hex[:12]
    hosts = _docker("run", "--rm", "--add-host", "host.docker.internal:host-gateway",
                    "--entrypoint", "cat", api.NGINX_IMAGE, "/etc/hosts")
    gateway = next(line.split()[0] for line in hosts.splitlines()
                   if "host.docker.internal" in line and ":" not in line.split()[0])
    configuration = api.render_source_nginx(secrets(), listen="0.0.0.0:8081")
    configuration = configuration.replace("https://" + api.SOURCE_REST_HOST + ";",
        "https://%s:%d;" % (gateway, server.server_port)).replace(
        "/etc/ssl/certs/ca-certificates.crt", "/run/test-ca.pem")
    config = directory / "nginx.conf"
    config.write_text(configuration)
    config.chmod(0o600)
    try:
        _docker("run", "--rm", "--detach", "--name", name, "--read-only",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
            "--user", str(os.getuid()), "--log-driver", "none",
            "--add-host", "host.docker.internal:host-gateway",
            "--publish", "127.0.0.1:%d:8081" % port,
            "--tmpfs", "/tmp:uid=%d,mode=0700" % os.getuid(),
            "--volume", str(config) + ":/run/nginx.conf:ro",
            "--volume", str(ca) + ":/run/test-ca.pem:ro", "--entrypoint", "nginx",
            api.NGINX_IMAGE, "-c", "/run/nginx.conf", "-g", "daemon off;")
        with httpx.Client(base_url="http://127.0.0.1:%d" % port, timeout=15, trust_env=False) as client:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                try:
                    if client.get("/").status_code == 404:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                pytest.fail("local nginx did not start")
            yield client, records, server_names
    finally:
        _docker("rm", "--force", name)
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.fixture(scope="module")
def source_proxy(tmp_path_factory):
    with _proxy(tmp_path_factory.mktemp("source-wire") / "private") as proxy:
        yield proxy


def test_exact_keys_forward_original_uri_body_and_error_response(source_proxy):
    client, records, server_names = source_proxy
    values = secrets()
    path = "/rest/v1/rpc/echo%2Fname?select=a%2Cb&filter=eq.x%2Fy&duplicate=1&duplicate=2"
    payload = b'{"original":"bytes","padding":"' + b"x" * 1100000 + b'"}'
    for field in api.KEY_ROLES:
        key = values[field]
        for authorization in (None, "bEaReR " + key):
            headers = {"apikey": key, "Content-Type": "application/json", "Cookie": "session=spoof",
                       "X-JWT-Claims": '{"role":"authenticator"}',
                       "X-Forwarded-Authorization": "Bearer forged", "Prefer": "return=representation",
                       "Range": "0-1", "Accept-Profile": "public", "X-Arbitrary-Input": "preserved"}
            if authorization:
                headers["Authorization"] = authorization
            response = client.post(path, headers=headers, content=payload)
            assert response.status_code == 403
            assert response.content == b'{"code":"42501","message":"wire contract"}'
            assert response.headers["Content-Range"] == "0-1/2"
            assert response.headers["X-Source-Receipt"] == "unchanged"
            method, upstream_path, upstream_headers, body = records[-1]
            assert method == "POST" and upstream_path == path and body == payload
            lowered = {name.lower(): value for name, value in upstream_headers.items()}
            assert lowered["host"] == api.SOURCE_REST_HOST and lowered["apikey"] == key
            assert all(name not in lowered for name in (
                "authorization", "cookie", "x-jwt-claims", "x-forwarded-authorization"))
            assert lowered["prefer"] == "return=representation" and lowered["range"] == "0-1"
            assert lowered["accept-profile"] == "public" and lowered["x-arbitrary-input"] == "preserved"
    for method in ("GET", "PATCH", "DELETE"):
        response = client.request(method, "/rest/v1/cache?select=id&limit=0",
                                  headers={"apikey": values["gateway_key"]})
        assert response.status_code == 403
        assert records[-1][0:2] == (method, "/rest/v1/cache?select=id&limit=0")
        assert records[-1][3] == b""
    assert len(records) == 9
    assert server_names and set(server_names) == {api.SOURCE_REST_HOST}


def test_invalid_identities_never_reach_source(source_proxy):
    client, records, server_names = source_proxy
    values = secrets()
    before, handshakes = len(records), len(server_names)
    bad_headers = [
        {}, {"apikey": values["arena_key"].swapcase()}, {"apikey": "sb_secret_unknown"},
        {"Authorization": "Bearer " + values["arena_key"]},
        {"apikey": values["arena_key"], "Authorization": "Bearer " + values["gateway_key"]},
        {"apikey": values["public_key"], "Cookie": "session=admin", "Authorization": "Bearer authenticated"},
        [("apikey", values["public_key"]), ("apikey", values["arena_key"])],
    ]
    for role in ("authenticator", "authenticated", "lab_arena_service", "service_role", "anon", "postgres"):
        bad_headers.append({"apikey": values["public_key"], "Authorization": "Bearer " + mint_jwt(role, secret=values["jwt_secret"])})
        bad_headers.append({"Authorization": "Bearer " + mint_jwt(role, secret=values["jwt_secret"]), "X-JWT-Claims": '{"role":"service_role"}'})
    for headers in bad_headers:
        assert client.post("/rest/v1/rpc/echo", headers=headers, content=b"{}").status_code == 401
    assert client.post("/rest/v1", headers={"apikey": values["arena_key"]}).status_code == 301
    for path in ("/auth/v1/token", "/storage/v1/object", "/realtime/v1"):
        assert client.post(path, headers={"apikey": values["arena_key"]}).status_code == 404
    assert len(records) == before and len(server_names) == handshakes


@pytest.mark.parametrize("failure", ["untrusted", "wrong_host"])
def test_tls_trust_and_hostname_are_verified_before_any_http_request(tmp_path, failure):
    with _proxy(tmp_path / "private", failure=failure) as (client, records, server_names):
        assert client.post("/rest/v1/rpc/echo", headers={"apikey": secrets()["arena_key"]}, content=b"{}").status_code == 502
        assert records == []
        assert server_names == [api.SOURCE_REST_HOST]


def test_long_read_uri_and_content_location_survive_source_https(tmp_path):
    # Current evaluation reads can filter 215 runs. PostgREST reflects that URI
    # in Content-Location, so both request and upstream-header limits matter.
    run_ids = ["run-" + "x" * 36 + "-%03d" % number for number in range(215)]
    params = [("select", "run_id"), ("run_id", "in.(" + ",".join(run_ids) + ")"), ("limit", "1")]
    with _proxy(tmp_path / "private", echo_location=True) as (client, records, server_names):
        request = client.build_request("GET", "/rest/v1/lab_arena_runs", params=params,
                                       headers={"apikey": secrets()["arena_key"]})
        assert 8192 < len(request.url.raw_path) < 16384
        response = client.send(request)
        assert response.status_code == 403
        assert response.content == b'{"code":"42501","message":"wire contract"}'
        assert response.headers["Content-Location"] == records[-1][1]
        assert len(response.headers["Content-Location"]) > 8192
        assert records[-1][0] == "GET" and records[-1][3] == b""
        assert server_names and set(server_names) == {api.SOURCE_REST_HOST}

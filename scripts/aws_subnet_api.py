"""Render private configuration for the subnet's PostgREST compatibility pair.

This does not connect to a database, provision resources or change a service.
Opaque client keys are checked at nginx. The target receives private role
tokens; managed-source REST receives the original API key alone.
Configuration and input errors never print their values.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Mapping
from urllib.parse import parse_qs, unquote, urlsplit


POSTGREST_IMAGE = "postgrest/postgrest:v14.5@sha256:b574528fe109c8343c1247155734d03df8c34b462f342dca0ccc20244fc36ef9"
NGINX_IMAGE = "nginx:1.28.3-alpine@sha256:a8b39bd9cf0f83869a2162827a0caf6137ddf759d50a171451b335cecc87d236"
JWT_AUDIENCE = "leadpoet-subnet-api"
KEY_ROLES = {"arena_key": "lab_arena_service", "gateway_key": "service_role", "public_key": "anon"}
_KEY = re.compile(r"sb_(?:secret|publishable)_[A-Za-z0-9_-]{16,256}\Z")
_UPSTREAM = re.compile(r"(?:[a-zA-Z0-9][a-zA-Z0-9.-]*|\[[0-9a-fA-F:]+\]):[0-9]{1,5}\Z")


class ConfigurationError(ValueError):
    """A fixed diagnostic with no secret-bearing exception text."""


def validate_secrets(values: Mapping[str, str]) -> dict[str, str]:
    expected = {*KEY_ROLES, "jwt_secret", "db_uri"}
    if (not isinstance(values, Mapping) or set(values) != expected
            or any(not isinstance(v, str) for v in values.values())):
        raise ConfigurationError("invalid configuration fields")
    result = dict(values)
    for name in KEY_ROLES:
        key = result[name]
        prefix = "sb_publishable_" if name == "public_key" else "sb_secret_"
        if not _KEY.fullmatch(key) or not key.startswith(prefix):
            raise ConfigurationError("invalid opaque key")
    if len({result[name] for name in KEY_ROLES}) != 3:
        raise ConfigurationError("client identities must differ")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", result["jwt_secret"]):
        raise ConfigurationError("invalid signing secret")
    try:
        uri = urlsplit(result["db_uri"])
        port = uri.port
    except ValueError as exc:
        raise ConfigurationError("invalid database URI") from None
    if (uri.scheme not in {"postgres", "postgresql"} or not uri.hostname
            or not uri.username or not uri.password or uri.fragment
            or unquote(uri.username) in {"postgres", "arena_admin", "lab_arena_owner", "service_role", "supabase_admin", "supabase_auth_admin"}
            or uri.path != "/postgres" or port not in (None, 5432)
            or any(ord(c) < 32 for c in result["db_uri"])):
        raise ConfigurationError("invalid database URI")
    if set(parse_qs(uri.query)) & {"user", "password", "dbname", "host", "hostaddr", "port", "service", "passfile", "options"}:
        raise ConfigurationError("database identity overrides are forbidden")
    return result


def role_jwt(role: str, secret: str) -> str:
    if role not in KEY_ROLES.values():
        raise ConfigurationError("invalid request role")
    def encode(value: object) -> str:
        data = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(data).decode().rstrip("=")
    message = encode({"alg": "HS256", "typ": "JWT"}) + "." + encode({
        "role": role, "iss": JWT_AUDIENCE, "aud": JWT_AUDIENCE,
    })
    signature = hmac.new(secret.encode(), message.encode(), hashlib.sha256).digest()
    return message + "." + base64.urlsafe_b64encode(signature).decode().rstrip("=")


def render_nginx(values: Mapping[str, str], *, upstream: str = "127.0.0.1:3000",
                 listen: str = "127.0.0.1:8081") -> str:
    values = validate_secrets(values)
    if not _UPSTREAM.fullmatch(upstream) or not _UPSTREAM.fullmatch(listen):
        raise ConfigurationError("invalid HTTP address")
    keys = "\n".join('        ~^%s$ %s;' % (values[name], role) for name, role in KEY_ROLES.items())
    bearers = "\n".join('        "~^[Bb][Ee][Aa][Rr][Ee][Rr] +%s$" %s;' % (values[name], role)
                        for name, role in KEY_ROLES.items())
    tokens = "\n".join('        %s %s;' % (role, role_jwt(role, values["jwt_secret"]))
                       for role in KEY_ROLES.values())
    allowed = "\n".join('        "%s:%s" 1;\n        "%s:absent" 1;' % (role, role, role)
                        for role in KEY_ROLES.values())
    return f'''# Private generated file. Never print nginx -T output.
worker_processes 1;
pid /tmp/nginx.pid;
error_log /dev/null crit;
events {{ worker_connections 1024; }}
http {{
    access_log off;
    server_tokens off;
    client_body_temp_path /tmp/client-body;
    proxy_temp_path /tmp/proxy;
    fastcgi_temp_path /tmp/fastcgi;
    uwsgi_temp_path /tmp/uwsgi;
    scgi_temp_path /tmp/scgi;
    # Literal nginx map keys ignore case. These anchored regexes must not.
    map $http_apikey $request_role {{
        default invalid;
{keys}
    }}
    map $http_authorization $bearer_role {{
        default invalid;
        "" absent;
        "~^[ \\t]+$" absent;
{bearers}
    }}
    map "$request_role:$bearer_role" $authorized {{
        default 0;
{allowed}
    }}
    map $request_role $backend_jwt {{
        default "";
{tokens}
    }}
    server {{
        listen {listen};
        # RPC/input contracts own their bounds; do not add nginx's 1 MB cap.
        client_max_body_size 0;
        location ^~ /rest/v1/ {{
            if ($authorized = 0) {{ return 401; }}
            proxy_pass http://{upstream}/;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Authorization "Bearer $backend_jwt";
            proxy_set_header apikey "";
            proxy_set_header Cookie "";
            proxy_set_header X-Forwarded-Authorization "";
            proxy_set_header X-JWT-Claims "";
            proxy_connect_timeout 8s;
            proxy_read_timeout 610s;
            proxy_send_timeout 610s;
            proxy_next_upstream off;
            proxy_redirect off;
            proxy_intercept_errors off;
            proxy_buffering off;
            proxy_request_buffering off;
            proxy_max_temp_file_size 0;
        }}
        location / {{ return 404; }}
    }}
}}
'''


SOURCE_REST_HOST = "qplwoislplkcegvdmbim.supabase.co"


def render_source_nginx(values: Mapping[str, str], *, listen: str = "127.0.0.1:8081") -> str:
    """Retain the exact client-key gate while forwarding to managed REST.

    The source API resolves the original opaque key. Private target JWTs and
    database credentials never reach it. The URI has no replacement component,
    so nginx forwards the original /rest/v1/ path and query unchanged.
    """
    body = render_nginx(values, listen=listen)
    token_map_start = body.index("    map $request_role $backend_jwt {")
    token_map_end = body.index("    server {", token_map_start)
    body = body[:token_map_start] + body[token_map_end:]
    if any(body.count(directive) != 1 for directive in (
            "proxy_pass http://127.0.0.1:3000/;",
            'proxy_set_header Authorization "Bearer $backend_jwt";',
            'proxy_set_header apikey "";')):
        raise ConfigurationError("source renderer contract changed")
    body = body.replace("proxy_pass http://127.0.0.1:3000/;", f"""proxy_pass https://{SOURCE_REST_HOST};
            proxy_set_header Host {SOURCE_REST_HOST};
            proxy_ssl_server_name on;
            proxy_ssl_name {SOURCE_REST_HOST};
            proxy_ssl_trusted_certificate /etc/ssl/certs/ca-certificates.crt;
            proxy_ssl_verify on;
            proxy_ssl_verify_depth 3;""")
    return body.replace('proxy_set_header Authorization "Bearer $backend_jwt";',
                        'proxy_set_header Authorization "";').replace(
        'proxy_set_header apikey "";', 'proxy_set_header apikey $http_apikey;')


def render_postgrest(values: Mapping[str, str], *, host: str = "127.0.0.1") -> str:
    values = validate_secrets(values)
    if not re.fullmatch(r"[a-zA-Z0-9.:-]+", host):
        raise ConfigurationError("invalid bind host")
    # JSON string quoting also escapes the config language's quotes/backslashes.
    return "\n".join((
        "# Private generated file; no anonymous backend fallback.",
        "db-uri = " + json.dumps(values["db_uri"]),
        'db-schemas = "public"', 'db-extra-search-path = "public, extensions"',
        "db-max-rows = 1000",
        "db-pool = 10", "db-pool-acquisition-timeout = 8",
        "jwt-secret = " + json.dumps(values["jwt_secret"]),
        "jwt-aud = " + json.dumps(JWT_AUDIENCE),
        'jwt-role-claim-key = ".role"', 'log-level = "crit"',
        'openapi-mode = "disabled"',
        "server-host = " + json.dumps(host), "server-port = 3000", "",
    ))


def _read_private(path: Path) -> dict[str, str]:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd) as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ConfigurationError("input must be a private regular file")
        data = stream.read(16385)
        if len(data) > 16384:
            raise ConfigurationError("configuration too large")
        return validate_secrets(json.loads(data))


def write_configuration(values: Mapping[str, str], directory: Path, *, owner_uid: int | None = None,
                        owner_gid: int | None = None, rest_backend: str = "target") -> None:
    if rest_backend not in {"target", "source"}:
        raise ConfigurationError("invalid REST backend")
    directory.mkdir(mode=0o700, parents=False, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_mode & 0o077:
        raise ConfigurationError("output must be a private directory")
    # Render both before replacing either; the operator owns pair stop/start
    # during explicit key rotation. Ordinary app deploys retain this pair.
    bodies = {"nginx.conf": (render_source_nginx(values) if rest_backend == "source"
                              else render_nginx(values)),
              "postgrest.conf": render_postgrest(values)}
    for name, body in bodies.items():
        fd, temporary = tempfile.mkstemp(prefix=".config-", dir=directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
                if owner_uid is not None:
                    os.fchown(stream.fileno(), owner_uid, owner_gid if owner_gid is not None else -1)
            os.replace(temporary, directory / name)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    if owner_uid is not None:
        os.chown(directory, owner_uid, owner_gid if owner_gid is not None else -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rest-backend", choices=("target", "source"), default="target")
    parser.add_argument("--owner-uid", type=int)
    parser.add_argument("--owner-gid", type=int)
    args = parser.parse_args()
    try:
        if args.owner_uid is not None and args.owner_uid <= 0:
            raise ConfigurationError("runtime owner must be unprivileged")
        values = _read_private(args.secrets_file)
        tls = parse_qs(urlsplit(values["db_uri"]).query)
        if (tls.get("sslmode") != ["verify-full"]
                or tls.get("sslrootcert") != ["/run/subnet-api/rds-ca.pem"]):
            raise ConfigurationError("production database TLS is required")
        write_configuration(values, args.output_dir,
                            owner_uid=args.owner_uid, owner_gid=args.owner_gid,
                            rest_backend=args.rest_backend)
    except (ConfigurationError, ValueError, TypeError, OSError, OverflowError):
        parser.exit(1, "subnet_api_configuration_invalid\n")
    print("subnet_api_configuration_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

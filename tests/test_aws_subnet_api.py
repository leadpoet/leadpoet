"""Private API configuration rejects credential confusion before rendering."""

import base64
import hashlib
import hmac
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from scripts import aws_subnet_api as api


def secrets():
    return {
        "arena_key": "sb_secret_Arena_" + "Aa1_" * 10,
        "gateway_key": "sb_secret_Gateway_" + "Bb2_" * 10,
        "public_key": "sb_publishable_Public_" + "Cc3_" * 10,
        "jwt_secret": "Dedicated_private_test_signing_secret_" + "z" * 32,
        "db_uri": "postgres://authenticator:test-only@database:5432/postgres",
    }


def test_role_tokens_are_private_signed_and_scoped():
    values = secrets()
    for role in api.KEY_ROLES.values():
        token = api.role_jwt(role, values["jwt_secret"])
        header, claims, signature = token.split(".")
        assert json.loads(base64.urlsafe_b64decode(claims + "=" * (-len(claims) % 4))) == {
            "role": role, "iss": api.JWT_AUDIENCE, "aud": api.JWT_AUDIENCE,
        }
        expected = hmac.new(values["jwt_secret"].encode(), (header + "." + claims).encode(), hashlib.sha256).digest()
        assert base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)) == expected


@pytest.mark.parametrize("field,value", [
    ("arena_key", 'sb_secret_";include /etc/passwd;'),
    ("arena_key", "sb_secret_" + "a" * 24 + "\n"),
    ("arena_key", "sb_publishable_" + "a" * 24),
    ("public_key", "sb_secret_" + "a" * 24),
    ("jwt_secret", "short"),
    ("db_uri", "postgres://user:private@database:5432/postgres\nlog-level=debug"),
    ("db_uri", "postgres://user:private@database:5432/postgres#fragment"),
    ("db_uri", "postgres://arena_admin:private@database:5432/postgres"),
    ("db_uri", "postgres://arena%5Fadmin:private@database:5432/postgres"),
    ("db_uri", "postgres://authenticator:private@database:5432/postgres?user=postgres"),
])
def test_invalid_configuration_cannot_inject_config(field, value):
    values = secrets()
    values[field] = value
    with pytest.raises(api.ConfigurationError):
        api.render_nginx(values)


def test_client_identities_cannot_share_key():
    values = secrets()
    values["gateway_key"] = values["arena_key"]
    with pytest.raises(api.ConfigurationError):
        api.render_nginx(values)


def test_configuration_permissions_and_no_secret_stdout(tmp_path):
    source = tmp_path / "private-input.json"
    values = secrets()
    values["db_uri"] += "?sslmode=verify-full&sslrootcert=/run/subnet-api/rds-ca.pem"
    source.write_text(json.dumps(values))
    source.chmod(0o600)
    output = tmp_path / "private-output"
    command = [sys.executable, str(Path(api.__file__)), "--secrets-file", str(source), "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout == "subnet_api_configuration_ready\n"
    assert not result.stderr
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in ("nginx.conf", "postgrest.conf"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    source.chmod(0o644)
    failure = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert failure.returncode == 1
    assert failure.stderr == "subnet_api_configuration_invalid\n"
    assert not failure.stdout
    assert all(value not in failure.stderr + failure.stdout for value in values.values())


def test_symlink_input_and_output_directory_are_rejected(tmp_path):
    source = tmp_path / "private-input.json"
    source.write_text(json.dumps(secrets()))
    source.chmod(0o600)
    linked = tmp_path / "linked.json"
    linked.symlink_to(source)
    with pytest.raises(OSError):
        api._read_private(linked)
    output = tmp_path / "output"
    output.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(api.ConfigurationError):
        api.write_configuration(secrets(), output)


def test_existing_rpc_timeouts_and_body_limits_are_not_reduced():
    body = api.render_nginx(secrets())
    assert "client_max_body_size 0;" in body
    assert "proxy_read_timeout 610s;" in body
    assert "proxy_next_upstream off;" in body
    assert "proxy_http_version 1.1;" in body
    assert "access_log off;" in body and "error_log /dev/null crit;" in body


def test_source_fallback_retains_target_contract_and_uses_only_original_client_keys():
    values = secrets()
    assert hashlib.sha256(api.render_nginx(values).encode()).hexdigest() == (
        "508bed53a5f391c61a2995330b2e0f4c520099c78f21a72aa550898c6e003bd4"
    )
    source = api.render_source_nginx(values)
    assert "proxy_pass https://qplwoislplkcegvdmbim.supabase.co;" in source
    assert 'proxy_set_header Authorization "";' in source
    assert 'proxy_set_header apikey $http_apikey;' in source
    assert "proxy_ssl_server_name on;" in source and "proxy_ssl_verify on;" in source
    assert values["db_uri"] not in source and values["jwt_secret"] not in source
    assert "$backend_jwt" not in source
    for role in api.KEY_ROLES.values():
        assert api.role_jwt(role, values["jwt_secret"]) not in source
    for directive in ("client_max_body_size 0;", "proxy_connect_timeout 8s;",
                      "proxy_read_timeout 610s;", "proxy_send_timeout 610s;",
                      "proxy_next_upstream off;", "proxy_buffering off;",
                      "proxy_request_buffering off;", "proxy_max_temp_file_size 0;"):
        assert directive in source


def test_source_cli_keeps_private_pair_and_retained_target_database(tmp_path):
    values = secrets()
    values["db_uri"] += "?sslmode=verify-full&sslrootcert=/run/subnet-api/rds-ca.pem"
    source = tmp_path / "input.json"
    source.write_text(json.dumps(values))
    source.chmod(0o600)
    output = tmp_path / "configuration"
    result = subprocess.run([sys.executable, str(Path(api.__file__)), "--secrets-file", str(source),
                             "--output-dir", str(output), "--rest-backend", "source"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0 and result.stdout == "subnet_api_configuration_ready\n"
    assert not result.stderr
    assert (output / "nginx.conf").read_text() == api.render_source_nginx(values)
    assert (output / "postgrest.conf").read_text() == api.render_postgrest(values)
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for path in output.iterdir():
        assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_source_selection_fails_closed_if_target_render_contract_changes(monkeypatch):
    changed = api.render_nginx(secrets()).replace("proxy_pass http://127.0.0.1:3000/;",
                                                "proxy_pass http://different-target/;")
    monkeypatch.setattr(api, "render_nginx", lambda *_args, **_kwargs: changed)
    with pytest.raises(api.ConfigurationError, match="source renderer contract changed"):
        api.render_source_nginx(secrets())


def test_header_buffer_change_is_only_three_directives_for_each_backend():
    directives = {"large_client_header_buffers 4 64k;", "proxy_buffer_size 64k;", "proxy_buffers 3 64k;"}
    for renderer, previous in (
        (api.render_nginx, "759a1b25eca6b75bf8b2b47be1c616c144b7af1903dde168bc6d8b951ccece35"),
        (api.render_source_nginx, "2e933330afbf57ce3d55507aee03ba5c2727ff22a44b378f0a61ea5554a40a0b"),
    ):
        body = renderer(secrets())
        assert all(sum(line.strip() == directive for line in body.splitlines()) == 1
                   for directive in directives)
        previous_bytes = "".join(line for line in body.splitlines(keepends=True)
                                 if line.strip() not in directives)
        assert hashlib.sha256(previous_bytes.encode()).hexdigest() == previous

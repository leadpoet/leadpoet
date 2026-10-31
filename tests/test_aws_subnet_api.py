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


def test_optional_capture_reader_key_maps_only_its_own_role():
    values = secrets()
    before = api.render_nginx(values), api.render_postgrest(values)
    values["capture_reader_key"] = "sb_secret_Capture_" + "Ee5_" * 10
    body = api.render_nginx(values)
    reader = "leadpoet_capture_reader"
    assert "    map_hash_bucket_size 128;\n" in body
    assert "map_hash_bucket_size" not in before[0]
    assert "        ~^%s$ %s;" % (values["capture_reader_key"], reader) in body
    assert '        "~^[Bb][Ee][Aa][Rr][Ee][Rr] +%s$" %s;' % (values["capture_reader_key"], reader) in body
    assert "        %s %s;" % (reader, api.role_jwt(reader, values["jwt_secret"])) in body
    assert '        "%s:%s" 1;\n        "%s:absent" 1;' % (reader, reader, reader) in body
    assert 'map "$request_role:$request_method:$uri:$request_uri:' in body
    assert "~^leadpoet_capture_reader:POST:/rest/v1/rpc/lab_arena_capture_summary_v1:" in body
    assert "        ~^leadpoet_capture_reader: 0;" in body
    assert "if ($capture_reader_route_allowed = 0) { return 403; }" in body
    assert "capture_reader_route_allowed" not in before[0]
    assert api.render_postgrest(values) == before[1]
    # Removing the key is the rollback: it restores the exact previous files.
    del values["capture_reader_key"]
    assert (api.render_nginx(values), api.render_postgrest(values)) == before


@pytest.mark.parametrize("name,value", [
    ("capture_reader_key", "sb_publishable_Capture_" + "Ee5_" * 10),
    ("capture_reader_key", 'sb_secret_";include /etc/passwd;'),
    ("capture_reader_key", None),
    ("reader_key", "sb_secret_Capture_" + "Ee5_" * 10),
])
def test_capture_reader_key_must_be_a_valid_named_secret_key(name, value):
    values = secrets()
    values[name] = value if value is not None else values["arena_key"]
    with pytest.raises(api.ConfigurationError):
        api.render_nginx(values)


@pytest.mark.parametrize("backend_args", [[], ["--rest-backend", "target"]])
def test_configuration_permissions_and_no_secret_stdout(tmp_path, backend_args):
    source = tmp_path / "private-input.json"
    values = secrets()
    values["db_uri"] += "?sslmode=verify-full&sslrootcert=/run/subnet-api/rds-ca.pem"
    source.write_text(json.dumps(values))
    source.chmod(0o600)
    output = tmp_path / "private-output"
    command = [sys.executable, str(Path(api.__file__)), "--secrets-file", str(source), "--output-dir", str(output), *backend_args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert result.stdout == "subnet_api_configuration_ready\n"
    assert not result.stderr
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    for name in ("nginx.conf", "postgrest.conf"):
        assert stat.S_IMODE((output / name).stat().st_mode) == 0o600
    # Exact target output from main 71d35786f229ca34848dc7e6d2dfef18a1b2b40a.
    assert hashlib.sha256((output / "nginx.conf").read_bytes()).hexdigest() == (
        "508bed53a5f391c61a2995330b2e0f4c520099c78f21a72aa550898c6e003bd4"
    )
    assert hashlib.sha256((output / "postgrest.conf").read_bytes()).hexdigest() == (
        "af4961a9909f4b3d0d524cdb9f7d38f3cd7b8867c72007adbc60294f17b4a96d"
    )
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


def test_retired_source_cli_is_rejected_before_reading_input_or_writing(tmp_path):
    source = tmp_path / "absent-input.json"
    output = tmp_path / "configuration"
    result = subprocess.run([sys.executable, str(Path(api.__file__)), "--secrets-file", str(source),
                             "--output-dir", str(output), "--rest-backend", "source"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert "invalid choice: 'source'" in result.stderr
    assert not result.stdout and not output.exists()


def test_retired_source_writer_cannot_replace_existing_target_configuration(tmp_path):
    output = tmp_path / "configuration"
    api.write_configuration(secrets(), output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(api.ConfigurationError, match="invalid REST backend"):
        api.write_configuration(secrets(), output, rest_backend="source")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_target_header_buffer_change_is_only_three_directives():
    directives = {"large_client_header_buffers 4 64k;", "proxy_buffer_size 64k;", "proxy_buffers 3 64k;"}
    body = api.render_nginx(secrets())
    assert all(sum(line.strip() == directive for line in body.splitlines()) == 1
               for directive in directives)
    previous_bytes = "".join(line for line in body.splitlines(keepends=True)
                             if line.strip() not in directives)
    assert hashlib.sha256(previous_bytes.encode()).hexdigest() == (
        "759a1b25eca6b75bf8b2b47be1c616c144b7af1903dde168bc6d8b951ccece35"
    )

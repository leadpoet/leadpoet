#!/usr/bin/env python3
"""Commission production parity through exact-code, in-memory secret bridges."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shlex
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, unquote
from urllib.request import Request, urlopen

if __package__:
    from scripts import setup_production_parity_staging as setup
else:
    import setup_production_parity_staging as setup


ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = "scripts/156-production-parity-readonly-role.sql"
SETUP_PATH = "scripts/setup_production_parity_staging.py"
INSTALLER_PATH = "scripts/install_production_parity_static_secrets.py"
ORCHESTRATOR_PATH = "scripts/bootstrap_production_parity_staging.py"
PROJECT_REF = "qplwoislplkcegvdmbim"
POOLER_HOST = "aws-0-us-east-1.pooler.supabase.com"
READER_USER = f"leadpoet_parity_reader.{PROJECT_REF}"
GATEWAY_HOST = "ec2-user@52.91.135.79"
VALIDATOR_HOST = "ec2-user@100.59.201.156"
SSH_KEY = Path("/Users/pranav/Downloads/leadpoet-2026-07-28.pem")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^[0-9a-f]{64}$")

ROLE_STATE_QUERY = """
SELECT
  current_database() = 'postgres' AS database_ready,
  current_user = 'postgres' AS migration_role_ready,
  EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
  ) AS reader_exists,
  COALESCE((
    SELECT rolcanlogin FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
  ), false) AS reader_can_login,
  to_regprocedure(
    'public.leadpoet_set_production_parity_reader_password_v1(text)'
  ) IS NOT NULL AS binder_ready,
  to_regprocedure(
    'public.leadpoet_production_parity_reader_contract_v1()'
  ) IS NOT NULL AS contract_ready,
  public.leadpoet_production_parity_reader_contract_v1() AS reader_contract
""".strip()
BIND_QUERY = (
    "SELECT public.leadpoet_set_production_parity_reader_password_v1($1) "
    "AS result"
)


class BootstrapError(RuntimeError):
    """A commissioning authority, transport, or verification gate failed."""


GATEWAY_LOADER = r'''
import base64, contextlib, hashlib, io, json, sys
try:
    payload = json.loads(sys.stdin.buffer.read(2 * 1024 * 1024))
    source = base64.b64decode(payload["source"], validate=True)
    if hashlib.sha256(source).hexdigest() != payload["source_sha256"]:
        raise RuntimeError("source hash")
    namespace = {"__name__": "__leadpoet_remote__", "__file__": payload["path"]}
    exec(compile(source, payload["path"], "exec"), namespace)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = namespace["main"](payload["argv"])
    if status != 0:
        raise RuntimeError("remote command")
    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("remote receipt")
    receipt = json.loads(lines[0])
    sys.stdout.write(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
except Exception:
    sys.stderr.write("REMOTE_PARITY_IAM_ERROR\n")
    raise SystemExit(1)
'''.strip()


VALIDATOR_LOADER = r'''
import base64, contextlib, hashlib, hmac, io, json, os, sys
def read_all(fd):
    chunks = []
    while True:
        value = os.read(fd, 65536)
        if not value:
            return b"".join(chunks)
        chunks.append(value)
def write_all(fd, value):
    view = memoryview(value)
    while view:
        size = os.write(fd, view)
        if size <= 0:
            raise RuntimeError("pipe write")
        view = view[size:]
def crypt(value, key, nonce):
    output = bytearray(len(value))
    offset = 0
    counter = 0
    while offset < len(value):
        block = hmac.new(
            key, b"leadpoet-parity-transport-v1" + nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        take = min(len(block), len(value) - offset)
        for index in range(take):
            output[offset + index] = value[offset + index] ^ block[index]
        offset += take
        counter += 1
    return bytes(output)
try:
    payload = json.loads(sys.stdin.buffer.read(2 * 1024 * 1024))
    source = base64.b64decode(payload["source"], validate=True)
    key = base64.b64decode(payload["transport_key"], validate=True)
    if len(key) != 32 or hashlib.sha256(source).hexdigest() != payload["source_sha256"]:
        raise RuntimeError("source identity")
    request_r, request_w = os.pipe()
    receipt_r, receipt_w = os.pipe()
    secret_r, secret_w = os.pipe()
    if min(request_r, request_w, receipt_r, receipt_w, secret_r, secret_w) < 3:
        raise RuntimeError("descriptor identity")
    write_all(request_w, json.dumps(payload["request"], separators=(",", ":")).encode())
    os.close(request_w)
    namespace = {"__name__": "__leadpoet_remote__", "__file__": payload["path"]}
    exec(compile(source, payload["path"], "exec"), namespace)
    argv = payload["argv"] + [
        "--request-fd", str(request_r), "--receipt-fd", str(receipt_w),
        "--secret-response-fd", str(secret_w),
    ]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = namespace["main"](argv)
    os.close(request_r); os.close(receipt_w); os.close(secret_w)
    if status != 0 or stdout.getvalue():
        raise RuntimeError("installer")
    receipt = json.loads(read_all(receipt_r))
    secret = read_all(secret_r)
    os.close(receipt_r); os.close(secret_r)
    nonce = os.urandom(16)
    ciphertext = crypt(secret, key, nonce)
    mac = hmac.new(key, b"leadpoet-parity-mac-v1" + nonce + ciphertext, hashlib.sha256).digest()
    response = {
        "receipt": receipt,
        "nonce": base64.b64encode(nonce).decode(),
        "ciphertext": base64.b64encode(ciphertext).decode(),
        "mac": base64.b64encode(mac).decode(),
    }
    sys.stdout.write(json.dumps(response, sort_keys=True, separators=(",", ":")))
except Exception:
    sys.stderr.write("REMOTE_PARITY_STATIC_ERROR\n")
    raise SystemExit(1)
'''.strip()


def _run(*args: str, input_value: bytes | None = None, timeout: int = 120) -> bytes:
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LANG", "LC_ALL")
        if os.environ.get(name)
    }
    try:
        result = subprocess.run(
            list(args),
            cwd=ROOT,
            env=env,
            input=input_value,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
            start_new_session=True,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise BootstrapError(f"command failed: {args[0]}") from exc
    if result.returncode != 0:
        raise BootstrapError(f"command failed: {args[0]}")
    return result.stdout


def _committed_blob(commit: str, path: str) -> bytes:
    if not SHA_RE.fullmatch(commit):
        raise BootstrapError("candidate commit is invalid")
    origin = _run("git", "rev-parse", "origin/main").decode().strip()
    if origin != commit:
        raise BootstrapError("candidate commit is not current origin/main")
    blob = _run("git", "show", f"{commit}:{path}")
    local = (ROOT / path).read_bytes()
    if not hmac.compare_digest(blob, local):
        raise BootstrapError(f"local candidate file differs: {path}")
    return blob


def _validate_ssh_key() -> None:
    before = SSH_KEY.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or SSH_KEY.is_symlink()
        or before.st_uid != os.geteuid()
        or before.st_nlink != 1
        or before.st_mode & 0o077
    ):
        raise BootstrapError("SSH key metadata is unsafe")


def _ssh(host: str, loader: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if host not in {GATEWAY_HOST, VALIDATOR_HOST}:
        raise BootstrapError("remote host identity differs")
    _validate_ssh_key()
    command = "python3 -c " + shlex.quote(loader)
    value = _run(
        "ssh",
        "-i",
        str(SSH_KEY),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        host,
        command,
        input_value=json.dumps(payload, separators=(",", ":")).encode(),
        timeout=180,
    )
    try:
        result = json.loads(value)
    except (UnicodeDecodeError, ValueError) as exc:
        raise BootstrapError("remote receipt is invalid") from exc
    if not isinstance(result, dict):
        raise BootstrapError("remote receipt is invalid")
    return result


def _gateway_command(
    source: bytes, *, path: str, argv: list[str]
) -> dict[str, Any]:
    return _ssh(
        GATEWAY_HOST,
        GATEWAY_LOADER,
        {
            "source": base64.b64encode(source).decode(),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "path": path,
            "argv": argv,
        },
    )


def _crypt(value: bytes, key: bytes, nonce: bytes) -> bytes:
    output = bytearray(len(value))
    offset = 0
    counter = 0
    while offset < len(value):
        block = hmac.new(
            key,
            b"leadpoet-parity-transport-v1"
            + nonce
            + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        take = min(len(block), len(value) - offset)
        for index in range(take):
            output[offset + index] = value[offset + index] ^ block[index]
        offset += take
        counter += 1
    return bytes(output)


def _validator_command(
    source: bytes,
    *,
    commit: str,
    migration_sha256: str,
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    key = secrets.token_bytes(32)
    outer = _ssh(
        VALIDATOR_HOST,
        VALIDATOR_LOADER,
        {
            "source": base64.b64encode(source).decode(),
            "source_sha256": hashlib.sha256(source).hexdigest(),
            "path": INSTALLER_PATH,
            "transport_key": base64.b64encode(key).decode(),
            "argv": [
                "--commit",
                commit,
                "--migration-sha256",
                migration_sha256,
            ],
            "request": request,
        },
    )
    try:
        nonce = base64.b64decode(outer["nonce"], validate=True)
        ciphertext = base64.b64decode(outer["ciphertext"], validate=True)
        mac = base64.b64decode(outer["mac"], validate=True)
        expected = hmac.new(
            key,
            b"leadpoet-parity-mac-v1" + nonce + ciphertext,
            hashlib.sha256,
        ).digest()
        if len(nonce) != 16 or not hmac.compare_digest(mac, expected):
            raise ValueError("transport MAC")
        secret_response = json.loads(_crypt(ciphertext, key, nonce))
        receipt = outer["receipt"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BootstrapError("validator secret response is invalid") from exc
    key = b""
    if not isinstance(receipt, dict) or not isinstance(secret_response, dict):
        raise BootstrapError("validator secret response is invalid")
    return receipt, secret_response


def _access_token() -> str:
    if platform.system() != "Darwin":
        raise BootstrapError("Supabase Keychain access requires macOS")
    if os.environ.get("SUPABASE_ACCESS_TOKEN"):
        raise BootstrapError("ambient Supabase credentials are forbidden")
    stored = _run(
        "security", "find-generic-password", "-s", "Supabase CLI", "-w"
    ).decode().strip()
    prefix = "go-keyring-base64:"
    if stored.startswith(prefix):
        try:
            stored = base64.b64decode(
                stored[len(prefix):], validate=True
            ).decode()
        except (ValueError, UnicodeDecodeError) as exc:
            raise BootstrapError("Supabase Keychain credential is invalid") from exc
    if not 20 <= len(stored) <= 1024 or any(c.isspace() for c in stored):
        raise BootstrapError("Supabase Keychain credential is invalid")
    return stored


def _management_query(
    token: str, query: str, *, parameters: list[str] | None = None
) -> list[dict[str, Any]]:
    body: dict[str, Any] = {"query": query}
    if parameters is not None:
        body["parameters"] = parameters
    request = Request(
        f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query",
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": "leadpoet-production-parity-bootstrap/1",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            value = json.loads(response.read().decode())
    except HTTPError as exc:
        raise BootstrapError(
            f"Supabase Management API failed with HTTP {exc.code}"
        ) from exc
    except (URLError, TimeoutError, UnicodeDecodeError, ValueError) as exc:
        raise BootstrapError("Supabase Management API failed") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise BootstrapError("Supabase Management API response is invalid")
    return value


def _role_state(token: str) -> dict[str, Any]:
    rows = _management_query(token, ROLE_STATE_QUERY)
    if len(rows) != 1:
        raise BootstrapError("production reader state is invalid")
    state = rows[0]
    contract = state.get("reader_contract")
    if isinstance(contract, str):
        try:
            contract = json.loads(contract)
        except ValueError as exc:
            raise BootstrapError("production reader contract is invalid") from exc
    if (
        state.get("database_ready") is not True
        or state.get("migration_role_ready") is not True
        or state.get("reader_exists") is not True
        or state.get("binder_ready") is not True
        or state.get("contract_ready") is not True
        or not isinstance(state.get("reader_can_login"), bool)
        or not isinstance(contract, Mapping)
        or contract.get("schema_version")
        != "leadpoet.production-parity-reader-contract.v1"
        or contract.get("database_name") != "postgres"
        or contract.get("reader_role") != "leadpoet_parity_reader"
        or contract.get("login_enabled") is not state.get("reader_can_login")
        or contract.get("superuser") is not False
        or contract.get("bypass_rls") is not True
        or contract.get("createdb") is not False
        or contract.get("createrole") is not False
        or contract.get("inherit") is not False
        or contract.get("replication") is not False
        or contract.get("connection_limit") != 2
        or contract.get("default_read_only") is not True
        or contract.get("membership_count") != 0
        or contract.get("schema_create_capable") is not False
        or contract.get("table_write_capable") is not False
        or contract.get("sequence_write_capable") is not False
    ):
        raise BootstrapError("production reader state is invalid")
    return state


def _dsn(password: str) -> str:
    if HASH_RE.fullmatch(password) is None:
        raise BootstrapError("production reader password is invalid")
    return (
        f"postgresql://{quote(READER_USER, safe='.')}:"
        f"{password}@{POOLER_HOST}:5432/postgres?sslmode=require"
    )


def _password_from_dsn(dsn: str) -> str:
    parsed = urlparse(dsn)
    password = unquote(str(parsed.password or ""))
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or str(parsed.hostname or "").lower() != POOLER_HOST
        or (parsed.port or 5432) != 5432
        or unquote(str(parsed.username or "")) != READER_USER
        or unquote(parsed.path.lstrip("/")) != "postgres"
        or parsed.query != "sslmode=require"
        or HASH_RE.fullmatch(password) is None
    ):
        raise BootstrapError("production reader DSN identity differs")
    return password


def _disable_repository() -> None:
    setup._gh_variable(setup.DEFAULT_REPOSITORY, "LEADPOET_PARITY_ENABLED", "false")
    if (
        setup._gh_variable_value(
            setup.DEFAULT_REPOSITORY, "LEADPOET_PARITY_ENABLED"
        )
        != "false"
    ):
        raise BootstrapError("GitHub parity disable readback differs")


def _configuration_args(
    *, commit: str, migration_sha256: str, receipt_fd: int
) -> argparse.Namespace:
    return argparse.Namespace(
        repository=setup.DEFAULT_REPOSITORY,
        region=setup.EXPECTED_REGION,
        production_gateway_ip=setup.PRODUCTION_GATEWAY_IP,
        production_gateway_url=setup.PRODUCTION_GATEWAY_URL,
        production_gateway_secret_id=setup.PRODUCTION_GATEWAY_SECRET_ID,
        readonly_dsn_secret_id=setup.READONLY_DSN_SECRET_ID,
        miner_intake_secret_id=setup.DEFAULT_MINER_INTAKE_SECRET_ID,
        volume_gib=setup.DEFAULT_VOLUME_GIB,
        enabled="true",
        commit=commit,
        migration_sha256=migration_sha256,
        receipt_fd=receipt_fd,
    )


def bootstrap(*, commit: str, migration_sha256: str) -> dict[str, Any]:
    _run(
        "git",
        "fetch",
        "--no-tags",
        "origin",
        "refs/heads/main:refs/remotes/origin/main",
    )
    if _run("git", "rev-parse", "HEAD").decode().strip() != commit:
        raise BootstrapError("candidate is not the current checkout HEAD")
    _run("git", "diff", "--exit-code", commit, "--")
    _committed_blob(commit, ORCHESTRATOR_PATH)
    migration = _committed_blob(commit, MIGRATION_PATH)
    setup_source = _committed_blob(commit, SETUP_PATH)
    installer_source = _committed_blob(commit, INSTALLER_PATH)
    if hashlib.sha256(migration).hexdigest() != migration_sha256:
        raise BootstrapError("migration SHA-256 differs")
    try:
        migration.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BootstrapError("migration is not UTF-8") from exc

    _disable_repository()
    bootstrap_role_created = False
    configured = False
    cleanup_error: Exception | None = None
    try:
        # The remote IAM transaction may commit before SSH or receipt parsing
        # fails. Cleanup is exact-owned and idempotent when the role is absent,
        # so make it mandatory before crossing that ambiguous boundary.
        bootstrap_role_created = True
        iam_receipt = _gateway_command(
            setup_source, path=SETUP_PATH, argv=["iam-only"]
        )
        if (
            iam_receipt.get("status") != "iam_ready"
            or iam_receipt.get("account_id") != setup.EXPECTED_ACCOUNT_ID
            or iam_receipt.get("github_variables_mutated") is not False
        ):
            raise BootstrapError("IAM-only receipt differs")
        try:
            trust_expiry = datetime.fromisoformat(
                str(iam_receipt.get("static_bootstrap_trust_expires_at") or "")
                .replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise BootstrapError("IAM-only trust expiry differs") from exc
        now = datetime.now(timezone.utc)
        if not now < trust_expiry <= now + timedelta(minutes=16):
            raise BootstrapError("IAM-only trust expiry differs")
        token = _access_token()
        state = _role_state(token)
        probe_receipt, probe_secret = _validator_command(
            installer_source,
            commit=commit,
            migration_sha256=migration_sha256,
            request={"mode": "probe", "migration_sha256": migration_sha256},
        )
        existing_dsn = str(probe_secret.get("readonly_dsn") or "")
        if bool(existing_dsn) != bool(probe_secret.get("readonly_dsn_available")):
            raise BootstrapError("static secret probe differs")
        if state.get("reader_can_login") is True and not existing_dsn:
            raise BootstrapError(
                "production reader LOGIN has no recoverable static DSN"
            )
        dsn = existing_dsn or _dsn(secrets.token_hex(32))
        password = _password_from_dsn(dsn)

        ensure_receipt, ensure_secret = _validator_command(
            installer_source,
            commit=commit,
            migration_sha256=migration_sha256,
            request={
                "mode": "ensure",
                "migration_sha256": migration_sha256,
                "readonly_dsn": dsn,
            },
        )
        if (
            ensure_receipt.get("status") != "installed"
            or not hmac.compare_digest(
                str(ensure_secret.get("readonly_dsn") or ""), dsn
            )
        ):
            raise BootstrapError("static secret installation readback differs")

        verified = False
        if existing_dsn and state.get("reader_can_login") is True:
            try:
                setup._verify_readonly_dsn(dsn)
                verified = True
            except setup.SetupError:
                verified = False
        if not verified:
            rows = _management_query(token, BIND_QUERY, parameters=[password])
            password = ""
            if len(rows) != 1:
                raise BootstrapError("production reader binder response differs")
            result = rows[0].get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except ValueError as exc:
                    raise BootstrapError(
                        "production reader binder response differs"
                    ) from exc
            if (
                not isinstance(result, Mapping)
                or result.get("status") != "bound"
                or result.get("reader_role") != "leadpoet_parity_reader"
                or result.get("login_enabled") is not True
            ):
                raise BootstrapError("production reader binder failed closed")
            setup._verify_readonly_dsn(dsn)
        token = ""
        dsn = ""

        final_receipt = dict(ensure_receipt)
        final_receipt.update(
            {
                "reader_default_read_only_verified": True,
                "installer_sha256": hashlib.sha256(installer_source).hexdigest(),
                "exact_source_streamed": True,
            }
        )

        # Remove and read back temporary authority before the final enable.
        cleanup = _gateway_command(
            setup_source, path=SETUP_PATH, argv=["cleanup-bootstrap"]
        )
        if cleanup.get("status") != "static_bootstrap_authority_removed":
            raise BootstrapError("static bootstrap cleanup receipt differs")
        bootstrap_role_created = False

        receipt_r, receipt_w = os.pipe()
        try:
            if receipt_r < 3 or receipt_w < 3:
                raise BootstrapError("receipt pipe allocation failed")
            os.write(
                receipt_w,
                json.dumps(final_receipt, separators=(",", ":")).encode(),
            )
            os.close(receipt_w)
            receipt_w = -1
            configuration = setup.configure_repository(
                _configuration_args(
                    commit=commit,
                    migration_sha256=migration_sha256,
                    receipt_fd=receipt_r,
                )
            )
        finally:
            if receipt_w >= 0:
                os.close(receipt_w)
            os.close(receipt_r)
        if configuration.get("enabled") is not True:
            raise BootstrapError("GitHub parity enable receipt differs")
        configured = True
        return {
            "status": "commissioned",
            "commit": commit,
            "migration": MIGRATION_PATH,
            "migration_sha256": migration_sha256,
            "iam_only": True,
            "exact_source_streamed": True,
            "reader_default_read_only_verified": True,
            "static_bootstrap_authority_removed": True,
            "github_enabled_last": True,
            "secret_values_printed": False,
        }
    finally:
        if bootstrap_role_created:
            try:
                cleanup = _gateway_command(
                    setup_source, path=SETUP_PATH, argv=["cleanup-bootstrap"]
                )
                if cleanup.get("status") != "static_bootstrap_authority_removed":
                    raise BootstrapError("static bootstrap cleanup receipt differs")
            except Exception as exc:  # noqa: BLE001 - failure remains fail-closed
                cleanup_error = exc
        if not configured or cleanup_error is not None:
            _disable_repository()
        if cleanup_error is not None:
            raise BootstrapError("static bootstrap cleanup failed") from cleanup_error


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--migration-sha256", required=True)
    args = parser.parse_args(argv)
    if (
        not SHA_RE.fullmatch(args.commit)
        or not HASH_RE.fullmatch(args.migration_sha256)
    ):
        print("ERROR: bootstrap identity is invalid", file=sys.stderr)
        return 1
    try:
        result = bootstrap(
            commit=args.commit, migration_sha256=args.migration_sha256
        )
    except (BootstrapError, OSError, setup.SetupError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

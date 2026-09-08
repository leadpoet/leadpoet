"""Production-parity SOURCE_ADD restart boundary tests."""

from __future__ import annotations

import http.client
import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import venv
from unittest.mock import patch
from urllib.request import urlopen
from uuid import uuid4

import pytest

from gateway.tee import gateway_miner_maintenance_restart_v1 as maintenance
from gateway.tee.release_manifest_v2 import build_local_release_identity
from gateway.tee.topology import ROLE_SPECS, topology_hash
from leadpoet_canonical.production_parity_boundary_v2 import (
    PRODUCTION_SUPABASE_ORIGIN,
)
from scripts import run_production_parity_full_host as full_host
from tests.test_source_add_end_to_end_postgres import (
    _database_with_migrations,
)
from tests.test_source_add_restart_state_restore_postgres import MIGRATIONS


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_COMMIT = "a" * 40
CANDIDATE_TREE = "b" * 40
RUN_ID = "parity-source-add-001"
CLONE_ORIGIN = "https://d111111abcdef8.cloudfront.net"
RESTART_INVOCATION_ID = "gateway-parity-source-add-001"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _unverified_service_role_token(*, issuer: str) -> str:
    def encoded(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    now = int(time.time())
    header = encoded(b'{"alg":"HS256","typ":"JWT"}')
    payload = encoded(
        json.dumps(
            {
                "aud": "authenticated",
                "exp": now + 172_800,
                "iat": now - 5,
                "iss": issuer,
                "role": "service_role",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    return f"{header}.{payload}.{encoded(b'x' * 32)}"


def _clone_service_role_token(secret: str) -> str:
    token = _unverified_service_role_token(
        issuer="leadpoet-production-parity"
    )
    header, payload, _signature = token.split(".")
    signature = hmac.new(
        secret.encode("ascii"),
        f"{header}.{payload}".encode("ascii"),
        hashlib.sha256,
    ).digest()
    encoded_signature = base64.urlsafe_b64encode(signature).decode("ascii")
    return f"{header}.{payload}.{encoded_signature.rstrip('=')}"


def _postgrest_run_command(
    *,
    container: str,
    database_port: int,
    postgrest_port: int,
    jwt_secret: str,
    native_linux: bool,
) -> list[str]:
    network = (
        [
            "--network",
            "host",
            "--env",
            f"PGRST_SERVER_PORT={postgrest_port}",
            "--env",
            "PGRST_SERVER_HOST=127.0.0.1",
        ]
        if native_linux
        else [
            "--add-host",
            "host.docker.internal:host-gateway",
            "--publish",
            f"127.0.0.1:{postgrest_port}:3000",
        ]
    )
    database_host = "127.0.0.1" if native_linux else "host.docker.internal"
    return [
        "docker",
        "run",
        "--rm",
        "--detach",
        "--name",
        container,
        *network,
        "--env",
        (
            "PGRST_DB_URI=postgres://postgres:postgres@"
            f"{database_host}:{database_port}/postgres"
        ),
        "--env",
        "PGRST_DB_SCHEMAS=public",
        "--env",
        "PGRST_DB_ANON_ROLE=anon",
        "--env",
        f"PGRST_JWT_SECRET={jwt_secret}",
        "postgrest/postgrest:v12.2.8",
    ]


@pytest.fixture
def clone_database():
    database = _database_with_migrations(MIGRATIONS)
    psycopg2, dsn = next(database)
    postgrest = f"source-add-parity-{uuid4().hex[:12]}"
    port = _free_port()
    jwt_secret = "production-parity-source-add-jwt-secret-0123456789"
    started = False
    try:
        result = subprocess.run(
            _postgrest_run_command(
                container=postgrest,
                database_port=int(dsn["port"]),
                postgrest_port=port,
                jwt_secret=jwt_secret,
                native_linux=sys.platform.startswith("linux"),
            ),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail("PostgREST container could not start")
        started = True
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:  # noqa: BLE001 - bounded local readiness check
                time.sleep(0.25)
        else:
            pytest.fail("PostgREST container did not become ready")
        yield {
            "psycopg2": psycopg2,
            "dsn": dsn,
            "port": port,
            "service_role_key": _clone_service_role_token(jwt_secret),
        }
    finally:
        if started:
            subprocess.run(
                ["docker", "rm", "--force", postgrest],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
        try:
            next(database)
        except StopIteration:
            pass


def _clone_connection_factory(port: int, observed_hosts: list[str]):
    class CloneConnection:
        def __init__(self, host: str, requested_port: int, *, timeout: float):
            observed_hosts.append(f"{host}:{requested_port}")
            self._connection = http.client.HTTPConnection(
                "127.0.0.1", port, timeout=timeout
            )

        def request(self, *args, **kwargs):
            method, path, *remaining = args
            assert path.startswith("/rest/v1/")
            return self._connection.request(
                method, path.removeprefix("/rest/v1"), *remaining, **kwargs
            )

        def getresponse(self):
            return self._connection.getresponse()

        def close(self) -> None:
            self._connection.close()

    return CloneConnection


def _runtime_status(paused: bool) -> dict[str, object]:
    return {
        "source_add": {
            "control": {"paused": paused, "unavailable": False},
            "dispatcher_enabled": False,
            "effective_dispatcher_enabled": False,
            "intake_enabled": False,
        }
    }


def _control_paused(clone_database) -> bool:
    connection = clone_database["psycopg2"].connect(**clone_database["dsn"])
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT paused FROM public.research_lab_source_add_control "
                "WHERE singleton"
            )
            return bool(cursor.fetchone()[0])
    finally:
        connection.close()


def _set_active(clone_database) -> None:
    connection = clone_database["psycopg2"].connect(**clone_database["dsn"])
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT public.research_lab_source_add_set_paused"
                "(false, 'parity test setup', 'operator:parity-test')"
            )
    finally:
        connection.close()


def _parity_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    service_role_key: str,
    candidate_commit: str = CANDIDATE_COMMIT,
) -> dict[str, str]:
    work_root = tmp_path / "opt" / "leadpoet-production-parity"
    marker = tmp_path / "run" / "early-boot-isolated"
    marker.parent.mkdir(mode=0o700)
    marker.write_text("isolated\n", encoding="ascii")
    marker.chmod(0o644)
    env_path = work_root / RUN_ID / "runtime" / "gateway.env"
    env_path.parent.mkdir(parents=True)
    env_path.parent.chmod(0o700)
    hydrated = {
        "BITTENSOR_NETWORK": "finney",
        "BITTENSOR_NETUID": "71",
        "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
        "LEADPOET_PARITY_CANDIDATE_SHA": candidate_commit,
        "LEADPOET_PRODUCTION_PARITY_BENCHMARK_DATE": "2026-09-07",
        "LEADPOET_PRODUCTION_PARITY_MODE": "enabled",
        "LEADPOET_PRODUCTION_PARITY_RUN_ID": RUN_ID,
        "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN": CLONE_ORIGIN,
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
        "SUPABASE_SERVICE_ROLE_KEY": service_role_key,
        "SUPABASE_URL": CLONE_ORIGIN,
    }
    env_path.write_text(
        json.dumps(hydrated, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    env_path.chmod(0o600)
    monkeypatch.setattr(maintenance, "PRODUCTION_PARITY_WORK_ROOT", work_root)
    monkeypatch.setattr(
        maintenance, "PRODUCTION_PARITY_ISOLATION_MARKER", marker
    )
    return full_host._full_restart_environment(
        region="us-east-1",
        home=env_path.parent,
        updates={
            **hydrated,
            "GATEWAY_ACTIVE_RELEASE_FALLBACK_CONTEXT": "full-parity",
            "GATEWAY_ENV_FILE": str(env_path),
            "GATEWAY_RESTART_INVOCATION_ID": RESTART_INVOCATION_ID,
            "LEADPOET_GATEWAY_ENV_SECRET_ID": (
                "leadpoet/staging/production-parity/runs/"
                f"{RUN_ID}/gateway"
            ),
        },
    )


def test_postgrest_command_keeps_postgres_on_platform_loopback():
    linux = _postgrest_run_command(
        container="linux-clone",
        database_port=54321,
        postgrest_port=30001,
        jwt_secret="test-jwt-secret",
        native_linux=True,
    )
    assert linux[linux.index("--network") : linux.index("--network") + 2] == [
        "--network",
        "host",
    ]
    assert "--publish" not in linux
    assert "PGRST_SERVER_PORT=30001" in linux
    assert "PGRST_SERVER_HOST=127.0.0.1" in linux
    assert (
        "PGRST_DB_URI=postgres://postgres:postgres@127.0.0.1:54321/postgres"
        in linux
    )

    desktop = _postgrest_run_command(
        container="desktop-clone",
        database_port=54321,
        postgrest_port=30001,
        jwt_secret="test-jwt-secret",
        native_linux=False,
    )
    assert "--network" not in desktop
    assert "host.docker.internal:host-gateway" in desktop
    assert "127.0.0.1:30001:3000" in desktop
    assert not any(item.startswith("PGRST_SERVER_PORT=") for item in desktop)
    assert (
        "PGRST_DB_URI=postgres://postgres:postgres@"
        "host.docker.internal:54321/postgres"
        in desktop
    )


def test_full_restart_environment_scrubs_before_first_aws_command(
    tmp_path: Path,
):
    restart_home = tmp_path / "restart-home"
    restart_home.mkdir(mode=0o700)
    gateway_environment = restart_home / "gateway.env"
    gateway_environment.write_text("CLONE_ONLY=true\n", encoding="ascii")
    gateway_environment.chmod(0o600)
    environment = full_host._full_restart_environment(
        region="us-east-1",
        home=restart_home,
        updates={"GATEWAY_ENV_FILE": str(gateway_environment)},
    )
    assert maintenance._FORBIDDEN_AWS_ENV_NAMES.isdisjoint(environment)
    assert environment["LEADPOET_AWS_INSTANCE_ROLE_ONLY"] == "true"
    assert environment["HOME"] == str(restart_home)
    child = subprocess.run(
        [
            "/bin/bash",
            "-c",
            (
                'test "$HOME" = "$EXPECTED_HOME" '
                '&& test -r "$GATEWAY_ENV_FILE" '
                "&& command -v python3 >/dev/null "
                "&& command -v git >/dev/null "
                "&& command -v aws >/dev/null"
            ),
        ],
        env={**environment, "EXPECTED_HOME": str(restart_home)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert child.returncode == 0
    with pytest.raises(full_host.FullParityError, match="home override differs"):
        full_host._full_restart_environment(
            region="us-east-1",
            home=restart_home,
            updates={"HOME": str(tmp_path / "poison-home")},
        )
    source = (ROOT / "gw_restart.sh").read_text(encoding="utf-8")
    validation_definition = source.index("validate_gateway_aws_authority()")
    scrub = source.index(
        "  scrub_gateway_bootstrap_aws_environment\n", validation_definition
    )
    validation = source.index(
        "  validate_gateway_aws_authority\n", scrub
    )
    first_aws_command = min(
        source.index("aws secretsmanager get-secret-value"),
        source.index("aws sts get-caller-identity"),
    )
    assert validation_definition < scrub < validation < first_aws_command
    assert (
        "export LEADPOET_AWS_INSTANCE_ROLE_ONLY=true"
        in source[scrub:first_aws_command]
    )
    boundary_functions = source[
        source.index("scrub_gateway_bootstrap_aws_environment()") : source.index(
            "on_gateway_restart_exit()", validation_definition
        )
    ]
    accepted = subprocess.run(
        [
            "/bin/bash",
            "-c",
            boundary_functions
            + "\nvalidate_gateway_aws_authority"
            + "\ntest \"$LEADPOET_AWS_INSTANCE_ROLE_ONLY\" = true"
            + "\ntest -z \"${AWS_CONFIG_FILE:-}\"",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert accepted.returncode == 0
    rejected = subprocess.run(
        [
            "/bin/bash",
            "-c",
            boundary_functions
            + "\nif validate_gateway_aws_authority; then exit 9; fi",
        ],
        env={**environment, "AWS_CONFIG_FILE": "/dev/null"},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert rejected.returncode == 0
    clone_child = full_host._clone_child_environment(region="us-east-1")
    assert {
        name: clone_child[name]
        for name in (
            "AWS_CONFIG_FILE",
            "AWS_SHARED_CREDENTIALS_FILE",
            "BOTO_CONFIG",
        )
    } == {
        "AWS_CONFIG_FILE": "/dev/null",
        "AWS_SHARED_CREDENTIALS_FILE": "/dev/null",
        "BOTO_CONFIG": "/dev/null",
    }
    (restart_home / ".aws").symlink_to(restart_home / "missing-credentials")
    with pytest.raises(full_host.FullParityError, match="home is not isolated"):
        full_host._full_restart_environment(
            region="us-east-1", home=restart_home, updates={}
        )


def test_full_restart_environment_selects_controller_python(
    tmp_path: Path,
):
    controller_venv = tmp_path / "controller-venv"
    venv.EnvBuilder(with_pip=True).create(controller_venv)
    controller_python = controller_venv / "bin" / "python3"
    restart_home = tmp_path / "restart-home"
    restart_home.mkdir(mode=0o700)

    environment = full_host._full_restart_environment(
        region="us-east-1",
        home=restart_home,
        updates={
            "GATEWAY_PYTHON_BIN": str(controller_python),
            "PATH": "/untrusted/bin",
            "PYTHONNOUSERSITE": "0",
        },
    )
    unqualified = subprocess.run(
        ["python3", "-c", "import sys;print(sys.prefix)"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    pip = subprocess.run(
        ["python3", "-m", "pip", "--version"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert unqualified.returncode == 0
    assert unqualified.stdout.strip() == str(controller_venv)
    assert pip.returncode == 0
    assert environment["GATEWAY_PYTHON_BIN"] == str(controller_python)
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PATH"].split(":") == [
        str(controller_python.parent),
        "/usr/local/sbin",
        "/usr/local/bin",
        "/usr/sbin",
        "/usr/bin",
        "/sbin",
        "/bin",
    ]

    with pytest.raises(
        full_host.FullParityError,
        match="Python runtime is unavailable",
    ):
        full_host._full_restart_environment(
            region="us-east-1",
            home=restart_home,
            updates={
                "GATEWAY_PYTHON_BIN": str(tmp_path / "missing-python")
            },
        )


def test_production_and_incomplete_boundaries_cannot_select_clone_authority(
    tmp_path: Path,
):
    assert (
        maintenance._production_parity_clone_authority(
            {}, deploy_commit=CANDIDATE_COMMIT
        )
        is None
    )
    calls: list[str] = []
    incomplete = {
        "LEADPOET_PRODUCTION_PARITY_MODE": "enabled",
        "LEADPOET_PRODUCTION_PARITY_RUN_ID": RUN_ID,
        "LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN": CLONE_ORIGIN,
        "BITTENSOR_NETWORK": "finney",
        "BITTENSOR_NETUID": "71",
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
    }
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="boundary is invalid",
    ):
        maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
            deploy_commit=CANDIDATE_COMMIT,
            parent_environment=incomplete,
            connection_factory=lambda *_args, **_kwargs: calls.append("HTTP"),
        )
    assert calls == []


def test_parity_boundary_rejects_production_origin_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    environment = _parity_environment(
        tmp_path, monkeypatch, service_role_key="clone-key"
    )
    environment["LEADPOET_PRODUCTION_PARITY_SUPABASE_ORIGIN"] = (
        PRODUCTION_SUPABASE_ORIGIN
    )
    environment["SUPABASE_URL"] = PRODUCTION_SUPABASE_ORIGIN
    calls: list[str] = []
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="boundary is invalid",
    ):
        maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
            deploy_commit=CANDIDATE_COMMIT,
            parent_environment=environment,
            connection_factory=lambda *_args, **_kwargs: calls.append("HTTP"),
        )
    assert calls == []


def test_parity_boundary_rejects_nonclone_service_key_before_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    environment = _parity_environment(
        tmp_path,
        monkeypatch,
        service_role_key=_unverified_service_role_token(issuer="supabase"),
    )
    calls: list[str] = []
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="clone credential identity differs",
    ):
        maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
            deploy_commit=CANDIDATE_COMMIT,
            parent_environment=environment,
            connection_factory=lambda *_args, **_kwargs: calls.append("HTTP"),
        )
    assert calls == []


def test_full_cli_quiesces_real_clone_and_runtime_restores_control(
    clone_database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    _set_active(clone_database)
    candidate_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    environment = _parity_environment(
        tmp_path,
        monkeypatch,
        service_role_key=clone_database["service_role_key"],
        candidate_commit=candidate_commit,
    )
    observed_hosts: list[str] = []
    connection_factory = _clone_connection_factory(
        clone_database["port"], observed_hosts
    )
    monkeypatch.setattr(
        maintenance.http.client, "HTTPSConnection", connection_factory
    )
    with patch.dict(os.environ, environment, clear=True):
        assert (
            maintenance.main(
                [
                    "--verify-shutdown-quiescence",
                    "--expected-commit",
                    candidate_commit,
                ]
            )
            == 0
        )
    result = json.loads(capsys.readouterr().out)
    assert result["authority"] == "production_parity_clone"
    assert result["status"] == "shutdown_quiescence_verified"
    assert _control_paused(clone_database) is True
    assert set(observed_hosts) == {"d111111abcdef8.cloudfront.net:443"}

    release_manifest = build_local_release_identity(
        [
            {
                "build_identity_hash": "sha256:" + "1" * 64,
                "commit_sha": candidate_commit,
                "dependency_lock_hash": "sha256:" + "2" * 64,
                "dockerfile_hash": "sha256:" + "3" * 64,
                "execution_manifest_hash": "sha256:" + "4" * 64,
                "image_id": "sha256:" + "5" * 64,
                "pcr0": "6" * 96,
                "role": role,
                "source_manifest_hash": "sha256:" + "7" * 64,
                "topology_hash": topology_hash(),
            }
            for role in ROLE_SPECS
        ]
    )
    release_path = tmp_path / "release.json"
    release_path.write_text(
        json.dumps(release_manifest, sort_keys=True), encoding="utf-8"
    )
    status_reads = iter(
        [
            _runtime_status(True),
            _runtime_status(False),
        ]
    )
    monkeypatch.setattr(
        maintenance, "_fetch_runtime_status", lambda: next(status_reads)
    )
    with patch.dict(os.environ, environment, clear=True):
        assert (
            maintenance.main(
                [
                    "--verify-runtime",
                    "--expected-commit",
                    candidate_commit,
                    "--repo-root",
                    str(ROOT),
                    "--release-manifest",
                    str(release_path),
                ]
            )
            == 0
        )
    runtime = json.loads(capsys.readouterr().out)
    assert runtime["authority"] == "production_parity_clone"
    assert runtime["source_add_restart_guard_status"] == (
        "released_restored_active"
    )
    assert _control_paused(clone_database) is False


def test_real_clone_active_lease_fails_closed_and_stays_paused(
    clone_database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_active(clone_database)
    connection = clone_database["psycopg2"].connect(**clone_database["dsn"])
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO public.research_lab_source_add_work_items (
                    work_id, submission_id, adapter_id, work_kind, work_status,
                    lease_token, leased_by, lease_expires_at, job_doc
                ) VALUES (
                    'source_add_work:9999999999999999',
                    'source_add_submission:9999999999999999',
                    'adapter:parity-active-lease', 'provenance', 'leased',
                    gen_random_uuid(), 'worker:parity',
                    NOW() + INTERVAL '5 minutes', '{}'::JSONB
                )
                """
            )
    finally:
        connection.close()
    environment = _parity_environment(
        tmp_path,
        monkeypatch,
        service_role_key=clone_database["service_role_key"],
    )
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="clone is not quiescent",
    ):
        maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
            deploy_commit=CANDIDATE_COMMIT,
            parent_environment=environment,
            connection_factory=_clone_connection_factory(
                clone_database["port"], []
            ),
        )
    assert _control_paused(clone_database) is True


@pytest.mark.parametrize("mutation", ["missing", "expired", "changed"])
def test_runtime_rejects_changed_clone_guard_and_forces_paused(
    mutation: str,
    clone_database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _set_active(clone_database)
    environment = _parity_environment(
        tmp_path,
        monkeypatch,
        service_role_key=clone_database["service_role_key"],
    )
    connection_factory = _clone_connection_factory(
        clone_database["port"], []
    )
    maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
        deploy_commit=CANDIDATE_COMMIT,
        parent_environment=environment,
        connection_factory=connection_factory,
    )
    connection = clone_database["psycopg2"].connect(**clone_database["dsn"])
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            if mutation == "missing":
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_control
                    SET paused = false,
                        reason = 'external clone state change',
                        actor_ref = 'operator:parity-test',
                        restart_guard_commitment = '',
                        restart_guard_owner_commitment = '',
                        restart_guard_expires_at = NULL,
                        restart_guard_acquired_at = NULL,
                        restart_guard_actor_ref = '',
                        restart_guard_restore_paused = NULL
                    WHERE singleton
                    """
                )
            elif mutation == "expired":
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_control
                    SET restart_guard_expires_at = NOW() - INTERVAL '1 minute'
                    WHERE singleton
                    """
                )
            else:
                cursor.execute(
                    """
                    UPDATE public.research_lab_source_add_control
                    SET restart_guard_owner_commitment = %s
                    WHERE singleton
                    """,
                    ("sha256:" + "9" * 64,),
                )
    finally:
        connection.close()

    with pytest.raises(maintenance.GatewayMinerMaintenanceRestartError):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=CANDIDATE_TREE,
            runtime_environment=environment,
            runtime_status=_runtime_status(True),
            connection_factory=connection_factory,
            runtime_status_provider=lambda: _runtime_status(
                _control_paused(clone_database)
            ),
        )
    assert _control_paused(clone_database) is True

import hashlib
import json
from pathlib import Path

import pytest

from scripts.complete_gateway_migration_203_barrier import (
    _load_environment,
    complete,
)


COMMIT = "a" * 40


class _Response:
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def getcode(self): return 200
    def read(self, _size=-1):
        return b'{"schema_version":"leadpoet.lab_arena.incentive_retirement_schema.v1","version":203}'


def _open(_request, timeout):
    assert timeout == 10
    return _Response()


ENV = {"SUPABASE_URL": "https://db.example", "SUPABASE_SERVICE_ROLE_KEY": "secret"}


def test_environment_loader_ignores_unrelated_raw_dump_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gateway.env"
    path.write_text(
        "GIT_SSH_COMMAND=ssh -i /tmp/key -o IdentitiesOnly=yes\n"
        "LESSOPEN=| /usr/bin/lesspipe %s\n"
        "SSH_CLIENT=192.0.2.1 12345 22\n"
        "SSH_CONNECTION=192.0.2.1 12345 192.0.2.2 22\n"
        "which_declare=declare -f\n"
        "SUPABASE_URL=https://db.example\n"
        "export SUPABASE_SERVICE_ROLE_KEY='service role test'\n"
    )
    assert _load_environment(path) == {
        "SUPABASE_URL": "https://db.example",
        "SUPABASE_SERVICE_ROLE_KEY": "service role test",
    }


def test_environment_loader_rejects_duplicate_or_malformed_required_values(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gateway.env"
    path.write_text(
        "SUPABASE_URL=https://one.example\n"
        "SUPABASE_URL=https://two.example\n"
    )
    with pytest.raises(ValueError, match="duplicate required"):
        _load_environment(path)
    path.write_text("SUPABASE_URL=https://db.example unexpected\n")
    with pytest.raises(ValueError, match="required gateway environment is malformed"):
        _load_environment(path)


def test_environment_loader_preserves_required_json_input(tmp_path: Path) -> None:
    path = tmp_path / "gateway.json"
    path.write_text(json.dumps({**ENV, "UNRELATED": "ignored"}))
    assert _load_environment(path) == ENV
    path.write_text(
        '{"SUPABASE_URL":"https://one.example",'
        '"SUPABASE_URL":"https://two.example",'
        '"SUPABASE_SERVICE_ROLE_KEY":"secret"}'
    )
    with pytest.raises(ValueError, match="duplicate required"):
        _load_environment(path)


def test_completion_is_bound_to_exact_sql_candidate_and_invocation(tmp_path: Path) -> None:
    sql = tmp_path / "203.sql"
    sql.write_text("select 203;\n")
    digest = hashlib.sha256(sql.read_bytes()).hexdigest()
    barrier = tmp_path / "barrier.json"
    barrier.write_text(json.dumps({
        "schema_version": "leadpoet.gateway.migration_203_barrier.v1",
        "candidate_commit": COMMIT,
        "sql_sha256": digest,
        "restart_invocation_id": "gateway-1-2",
        "old_producers_stopped": True,
    }))
    barrier.chmod(0o600)
    completion = tmp_path / "complete.json"
    result = complete(barrier=barrier, completion=completion, sql=sql, commit=COMMIT,
                      environment=ENV, opener=_open)
    assert json.loads(completion.read_text()) == result
    assert result["migration_203_verified"] is True
    assert completion.stat().st_mode & 0o777 == 0o600


def test_changed_sql_cannot_release_barrier(tmp_path: Path) -> None:
    sql = tmp_path / "203.sql"
    sql.write_text("select 203;\n")
    barrier = tmp_path / "barrier.json"
    barrier.write_text(json.dumps({
        "schema_version": "leadpoet.gateway.migration_203_barrier.v1",
        "candidate_commit": COMMIT,
        "sql_sha256": "0" * 64,
        "restart_invocation_id": "gateway-1-2",
        "old_producers_stopped": True,
    }))
    barrier.chmod(0o600)
    with pytest.raises(ValueError, match="differs"):
        complete(barrier=barrier, completion=tmp_path / "complete.json", sql=sql,
                 commit=COMMIT, environment=ENV, opener=_open)


def test_missing_live_capability_cannot_release_barrier(tmp_path: Path) -> None:
    sql = tmp_path / "203.sql"
    sql.write_text("select 203;\n")
    barrier = tmp_path / "barrier.json"
    barrier.write_text(json.dumps({
        "schema_version": "leadpoet.gateway.migration_203_barrier.v1",
        "candidate_commit": COMMIT,
        "sql_sha256": hashlib.sha256(sql.read_bytes()).hexdigest(),
        "restart_invocation_id": "gateway-1-2",
        "old_producers_stopped": True,
    }))
    barrier.chmod(0o600)
    def wrong(_request, timeout):
        response = _Response()
        response.read = lambda _size=-1: b'{"version":0}'
        return response
    completion = tmp_path / "complete.json"
    with pytest.raises(ValueError, match="capability differs"):
        complete(barrier=barrier, completion=completion, sql=sql, commit=COMMIT,
                 environment=ENV, opener=wrong)
    assert not completion.exists()

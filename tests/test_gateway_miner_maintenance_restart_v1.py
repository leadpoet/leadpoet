from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from gateway.tee import disable_gateway_miner_submissions_secret as disable_operation
from gateway.tee import gateway_miner_maintenance_restart_v1 as maintenance


INITIAL_VERSION = "11111111-1111-4111-8111-111111111111"
CONCURRENT_VERSION = "22222222-2222-4222-8222-222222222222"
CANDIDATE_COMMIT = "a" * 40
TREE_HASH = "b" * 40
BLOB_HASH = "e" * 64
CONTROLLER_COMMIT = next(iter(maintenance.SUPPORTED_N_MINUS_ONE_CONTROLLER_COMMITS))
REAL_REPOSITORY = Path(__file__).resolve().parents[1]
SEQUENTIAL_N_MINUS_ONE_COMMIT = "d649562b8c1e0077f670431cd9b22714eb686cd5"
SEQUENTIAL_CANDIDATE_COMMIT = "d72e475381e127aa209be33deed763d44a8289e6"
RECOVERY_VERSION = "33333333-3333-4333-8333-333333333333"
PREVIOUS_VERSION = "44444444-4444-4444-8444-444444444444"
PENDING_VERSION = "55555555-5555-4555-8555-555555555555"
REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT = (
    maintenance._require_hydrated_environment_commitment
)
REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT = (
    maintenance._live_gateway_restart_authority_commitment
)
REAL_PAUSE_SOURCE_ADD_FOR_RESTART = maintenance._pause_source_add_for_restart
REAL_REQUIRE_SOURCE_ADD_PAUSED = maintenance._require_source_add_paused
REAL_WAIT_FOR_SOURCE_ADD_QUIESCENCE = (
    maintenance._wait_for_source_add_quiescence
)
REAL_REQUIRE_SOURCE_ADD_QUIESCENT = maintenance._require_source_add_quiescent
REAL_RENEW_SOURCE_ADD_RESTART_GUARD = (
    maintenance._renew_source_add_restart_guard
)
REAL_RELEASE_SOURCE_ADD_RESTART_GUARD = (
    maintenance._release_source_add_restart_guard
)
REAL_FORCE_SOURCE_ADD_PAUSED_AFTER_RESTART_FAILURE = (
    maintenance._force_source_add_paused_after_restart_failure
)
SOURCE_ADD_CONTROL_COMMITMENT = "sha256:" + "6" * 64
SOURCE_ADD_QUIESCENCE_COMMITMENT = "sha256:" + "7" * 64
DEFAULT_RESTART_INVOCATION_ID = "gateway-test-invocation"
DEFAULT_SOURCE_ADD_GUARD_GENERATION = 1


def _restart_guard_commitment(
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
) -> str:
    return maintenance._source_add_restart_guard_identity(invocation_id)[
        "guard_commitment"
    ]


def _restart_owner_commitment(
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
) -> str:
    return maintenance._source_add_restart_guard_identity(invocation_id)[
        "owner_commitment"
    ]


def _restart_owner_generation_commitment(
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    generation: int = DEFAULT_SOURCE_ADD_GUARD_GENERATION,
) -> str:
    return maintenance._source_add_owner_generation_commitment(
        _restart_owner_commitment(invocation_id), generation
    )


def _source_add_guard_result_fields(
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    generation: int = DEFAULT_SOURCE_ADD_GUARD_GENERATION,
) -> dict[str, str]:
    return {
        "source_add_restart_guard_commitment": (
            _restart_guard_commitment(invocation_id)
        ),
        "source_add_restart_guard_generation": str(generation),
        "source_add_restart_guard_owner_generation_commitment": (
            _restart_owner_generation_commitment(invocation_id, generation)
        ),
        "source_add_restart_guard_restore_paused": "false",
    }


def _closed_source_add_runtime_status() -> dict[str, object]:
    return {
        "source_add": {
            "control": {"paused": True, "unavailable": False},
            "effective_dispatcher_enabled": False,
            "intake_enabled": False,
        }
    }


def _active_source_add_runtime_status() -> dict[str, object]:
    return {
        "source_add": {
            "control": {"paused": False, "unavailable": False},
            "effective_dispatcher_enabled": True,
            "intake_enabled": True,
        }
    }


@pytest.fixture(autouse=True)
def _stable_live_gateway_identity(monkeypatch: pytest.MonkeyPatch):
    for name in disable_operation._FORBIDDEN_AWS_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **_kwargs: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        lambda **kwargs: str(kwargs["expected_commitment"]),
    )
    monkeypatch.setattr(
        maintenance,
        "_pause_source_add_for_restart",
        lambda **_kwargs: {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
            **_source_add_guard_result_fields(),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_paused",
        lambda **_kwargs: {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_state",
        lambda **kwargs: {
            "status": "paused" if kwargs["expected_paused"] else "active",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_wait_for_source_add_quiescence",
        lambda **_kwargs: {
            "status": "quiescent",
            **_source_add_guard_result_fields(),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_quiescent",
        lambda **_kwargs: {
            "status": "quiescent",
            **_source_add_guard_result_fields(),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_renew_source_add_restart_guard",
        lambda **_kwargs: {
            "status": "renewed",
            **_source_add_guard_result_fields(),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_release_source_add_restart_guard",
        lambda **_kwargs: {
            "status": "released_restored_active",
            "source_add_restart_guard_restore_paused": "false",
        },
    )


def test_ci_environment_fixture_scrubs_static_aws_authority() -> None:
    assert not (
        disable_operation._FORBIDDEN_AWS_ENV_NAMES & set(os.environ)
    )


def test_prepare_still_rejects_explicit_static_aws_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSecretsClient(
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "explicit-rejection-sentinel")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="authority differs from production",
    ):
        _prepare(tmp_path, monkeypatch, client)


def _installed_controller_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    controller_commit: str = CONTROLLER_COMMIT,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    controller_parent = tmp_path / "restart-controller"
    controller_root = controller_parent / "gateway"
    releases_root = controller_root / "releases"
    release = releases_root / controller_commit
    for directory in (controller_parent, controller_root, releases_root):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o775)
    (release / "scripts").mkdir(parents=True)
    (release / "Leadpoet/utils").mkdir(parents=True)
    (release / "gateway/tee").mkdir(parents=True)
    release.chmod(0o700)
    files = {
        "gw_restart.sh": b"#!/bin/bash\nexit 0\n",
        "scripts/gateway_git_deploy.py": b"HELPER = True\n",
        "Leadpoet/utils/exact_commit_restart_v2.py": b"EXACT = True\n",
        "gateway/tee/host_memory_guard_v2.py": b"GUARD = True\n",
    }
    for relative, payload in files.items():
        destination = release / relative
        destination.write_bytes(payload)
        destination.chmod(0o700 if relative == "gw_restart.sh" else 0o600)
    current = controller_root / "current"
    current.symlink_to(f"releases/{controller_commit}")
    host_restart = tmp_path / "gw_restart.sh"
    host_restart.write_bytes(files["gw_restart.sh"])
    host_restart.chmod(0o700)
    monkeypatch.setattr(
        maintenance,
        "_run_git_bytes",
        lambda _repo, _show, object_name: files[object_name.split(":", 1)[1]],
    )
    monkeypatch.setattr(
        maintenance.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )
    return controller_parent, controller_root, releases_root, release, current, host_restart


def _real_installed_controller_fixture(
    tmp_path: Path,
    *,
    controller_commit: str,
) -> tuple[Path, Path]:
    controller_parent = tmp_path / "restart-controller"
    controller_root = controller_parent / "gateway"
    releases_root = controller_root / "releases"
    release = releases_root / controller_commit
    for directory in (controller_parent, controller_root, releases_root):
        directory.mkdir(exist_ok=True)
        directory.chmod(0o700)
    controller_files = {
        "gw_restart.sh": 0o700,
        "scripts/gateway_git_deploy.py": 0o600,
        "Leadpoet/utils/exact_commit_restart_v2.py": 0o600,
        "gateway/tee/host_memory_guard_v2.py": 0o600,
    }
    for relative_path, installed_mode in controller_files.items():
        destination = release / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            subprocess.check_output(
                ["git", "show", f"{controller_commit}:{relative_path}"],
                cwd=REAL_REPOSITORY,
            )
        )
        destination.chmod(installed_mode)
    release.chmod(0o700)
    current = controller_root / "current"
    current.symlink_to(f"releases/{controller_commit}")
    host_restart = tmp_path / "gw_restart.sh"
    host_restart.write_bytes((release / "gw_restart.sh").read_bytes())
    host_restart.chmod(0o700)
    return current, host_restart


class FakeSecretsClient:
    def __init__(self, secret: str):
        self.versions = {INITIAL_VERSION: secret}
        self.stages = {INITIAL_VERSION: {"AWSCURRENT"}}
        self.read_count = 0

    @property
    def current(self) -> str:
        current = [
            version for version, labels in self.stages.items() if "AWSCURRENT" in labels
        ]
        if len(current) != 1:
            raise RuntimeError("current stage is ambiguous")
        return current[0]

    def get_secret_value(
        self,
        *,
        SecretId,
        VersionId=None,
        VersionStage=None,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        self.read_count += 1
        if VersionId is not None:
            version = VersionId
        elif VersionStage == "AWSCURRENT":
            version = self.current
        else:
            raise AssertionError("test reads must be version-bound")
        return {
            "Name": SecretId,
            "VersionId": version,
            "SecretString": self.versions[version],
        }

    def describe_secret(self, *, SecretId):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        return {
            "Name": SecretId,
            "VersionIdsToStages": {
                version: sorted(labels)
                for version, labels in self.stages.items()
                if labels
            },
        }

    def put_secret_value(
        self,
        *,
        SecretId,
        SecretString,
        ClientRequestToken,
        VersionStages,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        if ClientRequestToken in self.versions:
            raise RuntimeError("version token already exists")
        self.versions[ClientRequestToken] = SecretString
        self.stages[ClientRequestToken] = set(VersionStages)
        return {"VersionId": ClientRequestToken}

    def update_secret_version_stage(
        self,
        *,
        SecretId,
        VersionStage,
        MoveToVersionId=None,
        RemoveFromVersionId=None,
    ):
        assert SecretId == disable_operation.GATEWAY_SECRET_ID
        if RemoveFromVersionId is not None:
            if VersionStage not in self.stages.get(RemoveFromVersionId, set()):
                raise RuntimeError("version-stage fence failed")
        if MoveToVersionId is not None:
            if VersionStage == "AWSCURRENT" and RemoveFromVersionId is not None:
                for labels in self.stages.values():
                    labels.discard("AWSPREVIOUS")
                self.stages[RemoveFromVersionId].add("AWSPREVIOUS")
            if RemoveFromVersionId is not None:
                self.stages[RemoveFromVersionId].discard(VersionStage)
            self.stages[MoveToVersionId].add(VersionStage)
        elif RemoveFromVersionId is not None:
            self.stages[RemoveFromVersionId].remove(VersionStage)
        return {}

    def install_concurrent_current(self) -> None:
        prior = self.current
        self.versions[CONCURRENT_VERSION] = self.versions[prior] + "DRIFT=value\n"
        for labels in self.stages.values():
            labels.discard("AWSPREVIOUS")
        self.stages[prior].discard("AWSCURRENT")
        self.stages[prior].add("AWSPREVIOUS")
        self.stages[CONCURRENT_VERSION] = {"AWSCURRENT"}


def _source_add_secret(service_role_key: str = "unit-test-service-role") -> str:
    return (
        f"SUPABASE_URL='{maintenance.PRODUCTION_SUPABASE_ORIGIN}'\n"
        f"SUPABASE_SERVICE_ROLE_KEY='{service_role_key}'\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )


def _source_add_connection_factory(
    responses: list[tuple[int, object]],
    requests: list[dict[str, object]],
):
    class FakeResponse:
        def __init__(self, status: int, payload: object):
            self.status = status
            self.payload = json.dumps(
                payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")

        def getheader(self, name: str):
            if name == "Content-Length":
                return str(len(self.payload))
            return None

        def read(self, limit: int):
            return self.payload[:limit]

    class FakeConnection:
        def __init__(self, host: str, port: int, timeout: float):
            if not responses:
                raise AssertionError("unexpected SOURCE_ADD control connection")
            self.response = FakeResponse(*responses.pop(0))
            self.record: dict[str, object] = {
                "host": host,
                "port": port,
                "timeout": timeout,
            }
            requests.append(self.record)

        def request(self, method: str, path: str, body, headers):
            self.record.update(
                {
                    "method": method,
                    "path": path,
                    "body": body,
                    "headers": dict(headers),
                }
            )

        def getresponse(self):
            return self.response

        def close(self):
            self.record["closed"] = True

    return FakeConnection


def _paused_source_add_control(*, actor_ref: str) -> dict[str, object]:
    return {
        "singleton": True,
        "paused": True,
        "reason": maintenance.SOURCE_ADD_PAUSE_REASON,
        "actor_ref": actor_ref,
        "updated_at": "2026-08-31T18:20:00Z",
    }


def _source_add_admission_contract() -> dict[str, object]:
    return {
        "schema_version": "leadpoet.source_add_admission_control_contract.v1",
        "control_row_present": True,
        "trigger_enabled": True,
        "pause_rpc": maintenance.SOURCE_ADD_PAUSE_RPC,
        "admission_trigger": "trg_source_add_work_admission_control",
    }


def _source_add_claim_control_contract(
    **overrides: object,
) -> dict[str, object]:
    contract: dict[str, object] = {
        "schema_version": "leadpoet.source_add_claim_control_contract.v2",
        "control_lock": "source-add-control",
        "pause_rpc": "research_lab_source_add_set_paused",
        "pause_signature": "boolean,text,text",
        "claim_rpc": "research_lab_source_add_claim_work",
        "claim_signature": "text,integer",
        "acquire_guard_rpc": (
            "research_lab_source_add_acquire_restart_guard_v2"
        ),
        "acquire_guard_signature": "text,text,bigint,integer,text",
        "guard_state_rpc": (
            "research_lab_source_add_restart_guard_state_v2"
        ),
        "guard_state_signature": "",
        "release_guard_rpc": (
            "research_lab_source_add_release_restart_guard_v2"
        ),
        "release_guard_signature": "text,text,bigint,text",
        "guard_state_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "acquire_guard_result_fields": [
            "schema_version",
            "paused",
            "guard_active",
            "guard_commitment",
            "owner_commitment",
            "guard_generation",
            "owner_generation_commitment",
            "guard_expires_at",
            "restore_paused",
        ],
        "release_guard_result_fields": [
            "schema_version",
            "released",
            "paused",
            "guard_active",
            "guard_generation",
            "owner_generation_commitment",
            "restored_pre_restart_state",
        ],
        "restart_quiescence_rpc": (
            "research_lab_source_add_restart_quiescence_v1"
        ),
        "restart_quiescence_signature": "text,text,bigint",
        "restore_state_column": "restart_guard_restore_paused",
        "acquire_captures_pre_restart_paused": True,
        "renewal_preserves_restore_state": True,
        "expired_takeover_preserves_restore_state": True,
        "operator_pause_wins": True,
        "release_restores_pre_restart_state": True,
        "failed_restart_keeps_paused": True,
        "rollback_v1_contract_schema_version": (
            "leadpoet.source_add_claim_control_contract.v1"
        ),
        "rollback_v1_contract_sha256": (
            maintenance.SOURCE_ADD_CLAIM_CONTROL_ROLLBACK_V1_CONTRACT_SHA256
        ),
        "migration_requires_paused": True,
        "migration_requires_zero_leased": True,
        "migration_requires_guard_clear": True,
        "function_authority_sha256": (
            maintenance.SOURCE_ADD_CLAIM_CONTROL_V2_FUNCTION_AUTHORITY_SHA256
        ),
        "functions": {
            "admission_guard": True,
            "acquire_restart_guard_v1": True,
            "acquire_restart_guard_v2": True,
            "claim_work": True,
            "pause": True,
            "release_restart_guard_v1": True,
            "release_restart_guard_v2": True,
            "restart_guard_state_v1": True,
            "restart_guard_state_v2": True,
            "restart_quiescence_v1": True,
            "restore_trigger_v2": True,
        },
        "permissions": {
            "service_role_exists": True,
            "service_role_callable": True,
            "anon_callable": False,
            "authenticated_callable": False,
        },
    }
    contract.update(overrides)
    return contract


def _source_add_restart_guard(
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    generation: int = DEFAULT_SOURCE_ADD_GUARD_GENERATION,
) -> dict[str, object]:
    return {
        "schema_version": "leadpoet.source_add_restart_guard.v2",
        "paused": True,
        "guard_active": True,
        "guard_commitment": _restart_guard_commitment(invocation_id),
        "owner_commitment": _restart_owner_commitment(invocation_id),
        "guard_generation": generation,
        "owner_generation_commitment": (
            _restart_owner_generation_commitment(invocation_id, generation)
        ),
        "guard_expires_at": "2099-01-01T00:00:00+00:00",
        "restore_paused": False,
    }


def _source_add_guard_state(
    *,
    paused: bool = True,
    guard_active: bool = True,
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    generation: int = DEFAULT_SOURCE_ADD_GUARD_GENERATION,
    guard_expires_at: str | None = "2099-01-01T00:00:00+00:00",
    restore_paused: bool | None = False,
) -> dict[str, object]:
    if guard_active or guard_expires_at is not None:
        guard_commitment = _restart_guard_commitment(invocation_id)
        owner_commitment = _restart_owner_commitment(invocation_id)
        owner_generation_commitment = (
            _restart_owner_generation_commitment(invocation_id, generation)
        )
    else:
        guard_commitment = ""
        owner_commitment = ""
        owner_generation_commitment = ""
        restore_paused = None
    return {
        "schema_version": "leadpoet.source_add_restart_guard_state.v2",
        "paused": paused,
        "guard_active": guard_active,
        "guard_commitment": guard_commitment,
        "owner_commitment": owner_commitment,
        "guard_generation": generation,
        "owner_generation_commitment": owner_generation_commitment,
        "guard_expires_at": guard_expires_at,
        "restore_paused": restore_paused,
    }


def _empty_source_add_guard_state(
    generation: int = 0,
) -> dict[str, object]:
    return _source_add_guard_state(
        paused=False,
        guard_active=False,
        generation=generation,
        guard_expires_at=None,
    )


def _source_add_quiescence(
    *,
    paused: bool = True,
    guard_active: bool = True,
    guard_matches: bool = True,
    owner_matches: bool = True,
    generation_matches: bool = True,
    leased_work_count: int = 0,
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    generation: int = DEFAULT_SOURCE_ADD_GUARD_GENERATION,
) -> dict[str, object]:
    return {
        "schema_version": "leadpoet.source_add_restart_quiescence.v1",
        "paused": paused,
        "guard_active": guard_active,
        "guard_matches": guard_matches,
        "owner_matches": owner_matches,
        "generation_matches": generation_matches,
        "guard_commitment": _restart_guard_commitment(invocation_id),
        "owner_commitment": _restart_owner_commitment(invocation_id),
        "guard_generation": generation,
        "owner_generation_commitment": (
            _restart_owner_generation_commitment(invocation_id, generation)
        ),
        "guard_expires_at": "2099-01-01T00:00:00+00:00",
        "leased_work_count": leased_work_count,
        "quiescent": (
            paused
            and guard_active
            and guard_matches
            and owner_matches
            and generation_matches
            and leased_work_count == 0
        ),
    }


def _controller_bundle(
    controller_commit: str = CONTROLLER_COMMIT,
) -> dict[str, object]:
    payloads = {
        "wrapper": b"#!/bin/bash\nexit 0\n",
        "git_helper": b"HELPER = True\n",
        "exact_commit_helper": b"EXACT = True\n",
        "memory_guard": b"GUARD = True\n",
    }
    return {
        "controller_commit": controller_commit,
        "payloads": payloads,
        "commitments": {
            name: "sha256:" + hashlib.sha256(payload).hexdigest()
            for name, payload in payloads.items()
        },
    }


def _prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    client: FakeSecretsClient,
    *,
    candidate_commit: str = CANDIDATE_COMMIT,
    controller_commit: str = CONTROLLER_COMMIT,
    invocation_id: str = DEFAULT_RESTART_INVOCATION_ID,
    source_add_pause_hook=None,
    source_add_readback_hook=None,
    source_add_wait_hook=None,
    source_add_quiescence_readback_hook=None,
    runtime_status_hook=None,
    live_process_commitment: str = "sha256:" + "1" * 64,
) -> dict[str, object]:
    monkeypatch.setattr(
        maintenance,
        "_require_canonical_restart_lock_fd",
        lambda: None,
    )
    monkeypatch.setattr(
        maintenance,
        "_validate_candidate_identity",
        lambda **_kwargs: {
            "tree_hash": TREE_HASH,
            "blob_manifest_sha256": BLOB_HASH,
            "previous_sha": controller_commit,
            "n_minus_one_controller_commit": controller_commit,
            "controller_bundle": _controller_bundle(controller_commit),
        },
    )
    monkeypatch.setattr(maintenance, "_verify_protected_source", lambda: None)
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **_kwargs: live_process_commitment,
    )
    monkeypatch.setattr(
        maintenance,
        "_pause_source_add_for_restart",
        source_add_pause_hook
        or (
            lambda **kwargs: {
                "status": "paused",
                "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
                **_source_add_guard_result_fields(
                    kwargs["restart_invocation_id"]
                ),
            }
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_paused",
        source_add_readback_hook
        or (
            lambda **_kwargs: {
                "status": "paused",
                "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
            }
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_wait_for_source_add_quiescence",
        source_add_wait_hook
        or (
            lambda **kwargs: {
                "status": "quiescent",
                **_source_add_guard_result_fields(
                    kwargs["restart_invocation_id"]
                ),
                "source_add_quiescence_commitment": (
                    SOURCE_ADD_QUIESCENCE_COMMITMENT
                ),
            }
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_quiescent",
        source_add_quiescence_readback_hook
        or (
            lambda **kwargs: {
                "status": "quiescent",
                **_source_add_guard_result_fields(
                    kwargs["restart_invocation_id"]
                ),
                "source_add_quiescence_commitment": (
                    SOURCE_ADD_QUIESCENCE_COMMITMENT
                ),
            }
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_runtime_status",
        runtime_status_hook or _closed_source_add_runtime_status,
    )
    return maintenance.prepare_gateway_miner_maintenance_restart(
        repo_root=tmp_path / "repo",
        candidate_root=tmp_path / "candidate",
        plan_file=tmp_path / "plan.json",
        expected_commit=candidate_commit,
        controller_current=tmp_path / "controller/current",
        host_restart_path=tmp_path / "gw_restart.sh",
        restart_invocation_id=invocation_id,
        recovery_journal_path=tmp_path / "private" / "transaction.json",
        secrets_client=client,
    )


def test_prepare_changes_only_target_and_returns_redacted_invocation_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    secret_marker = "unrelated-secret-must-not-escape"
    client = FakeSecretsClient(
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
        f"UNRELATED_SECRET={secret_marker}\n"
        "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n"
    )

    result = _prepare(tmp_path, monkeypatch, client)
    proof = result["proof"]

    assert result["status"] == "prepared"
    assert proof["candidate_commit"] == CANDIDATE_COMMIT
    assert proof["candidate_tree_hash"] == TREE_HASH
    assert "gateway_release_hash" not in proof
    assert proof["current_secret_version_id"] == client.current
    assert proof["current_document_commitment"].startswith("sha256:")
    assert proof["current_stage_topology_commitment"].startswith("sha256:")
    assert proof["source_add_restart_guard_commitment"] == (
        _restart_guard_commitment()
    )
    assert proof["source_add_restart_guard_generation"] == "1"
    assert proof[
        "source_add_restart_guard_owner_generation_commitment"
    ] == _restart_owner_generation_commitment()
    assert secret_marker not in json.dumps(proof)
    assert secret_marker not in json.dumps(
        {name: value for name, value in result.items() if name != "tree_evidence"}
    )
    current_secret = client.versions[client.current]
    assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n" in current_secret
    assert f"UNRELATED_SECRET={secret_marker}\n" in current_secret
    assert "RESEARCH_LAB_AUTORESEARCH_WORKER_COUNT=0\n" in current_secret
    assert not (tmp_path / "private" / "transaction.json").exists()
    assert not any(path.name.endswith("receipt.json") for path in tmp_path.rglob("*"))


def test_source_add_pause_uses_fixed_rpc_then_exact_durable_readback(
    capsys: pytest.CaptureFixture[str],
):
    service_role_key = "unit-test-service-role-private"
    client = FakeSecretsClient(_source_add_secret(service_role_key))
    invocation_id = "gateway-source-add-pause-test"
    actor_ref = "gateway-restart:" + hashlib.sha256(
        invocation_id.encode("utf-8")
    ).hexdigest()
    row = _paused_source_add_control(actor_ref=actor_ref)
    requests: list[dict[str, object]] = []
    responses = [
        (200, _source_add_admission_contract()),
        (200, _source_add_claim_control_contract()),
        (200, _empty_source_add_guard_state()),
        (200, _source_add_restart_guard(invocation_id)),
        (200, [row]),
    ]

    result = REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
        secrets_client=client,
        restart_invocation_id=invocation_id,
        connection_factory=_source_add_connection_factory(responses, requests),
    )

    assert result == {
        "status": "paused",
        "source_add_control_commitment": maintenance.sha256_json(row),
        **_source_add_guard_result_fields(invocation_id),
    }
    assert responses == []
    assert [request["method"] for request in requests] == [
        "POST",
        "POST",
        "POST",
        "POST",
        "GET",
    ]
    assert [request["path"] for request in requests] == [
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_ADMISSION_CONTRACT_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_GUARD_STATE_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC}",
        (
            f"/rest/v1/{maintenance.SOURCE_ADD_CONTROL_TABLE}"
            "?select=singleton,paused,reason,actor_ref,updated_at"
            "&singleton=eq.true&limit=2"
        ),
    ]
    assert all(
        request["host"]
        == maintenance.PRODUCTION_SUPABASE_ORIGIN.removeprefix("https://")
        and request["port"] == 443
        and request["closed"] is True
        for request in requests
    )
    assert json.loads(requests[0]["body"]) == {}
    assert json.loads(requests[1]["body"]) == {}
    assert json.loads(requests[2]["body"]) == {}
    identity = maintenance._source_add_restart_guard_identity(invocation_id)
    assert json.loads(requests[3]["body"]) == {
        "p_actor_ref": actor_ref,
        "p_expected_generation": 0,
        "p_guard_id": identity["guard_id"],
        "p_lease_seconds": maintenance.SOURCE_ADD_RESTART_GUARD_LEASE_SECONDS,
        "p_owner_id": identity["owner_id"],
    }
    assert requests[4]["body"] is None
    assert all(service_role_key not in str(request["path"]) for request in requests)
    assert service_role_key not in json.dumps(result)
    assert capsys.readouterr() == ("", "")


def test_source_add_pause_fails_closed_when_migration_rpc_is_unavailable():
    client = FakeSecretsClient(_source_add_secret())
    before = dict(client.versions)
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="pause authority request was rejected",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id="gateway-source-add-missing-rpc",
            connection_factory=_source_add_connection_factory(
                [(404, {"code": "PGRST202"})], requests
            ),
        )

    assert client.versions == before
    assert len(requests) == 1
    assert requests[0]["path"] == (
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_ADMISSION_CONTRACT_RPC}"
    )


def test_source_add_guard_acquisition_fails_closed_when_rpc_is_unavailable():
    client = FakeSecretsClient(_source_add_secret())
    before = dict(client.versions)
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="pause authority request was rejected",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_admission_contract()),
                    (200, _source_add_claim_control_contract()),
                    (200, _empty_source_add_guard_state()),
                    (404, {"code": "PGRST202"}),
                ],
                requests,
            ),
        )

    assert client.versions == before
    assert requests[-1]["path"] == (
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC}"
    )


def test_source_add_pause_fails_before_acquire_when_claim_control_rpc_is_missing():
    client = FakeSecretsClient(_source_add_secret())
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="claim-control contract is unavailable or invalid",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_admission_contract()),
                    (404, {"code": "PGRST202"}),
                ],
                requests,
            ),
        )

    assert [request["path"] for request in requests] == [
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_ADMISSION_CONTRACT_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_CLAIM_CONTROL_CONTRACT_RPC}",
    ]


@pytest.mark.parametrize("invalid_field", ["schema", "hash", "signature", "acl"])
def test_source_add_pause_rejects_wrong_claim_control_contract_before_acquire(
    invalid_field: str,
):
    client = FakeSecretsClient(_source_add_secret())
    contract = _source_add_claim_control_contract()
    if invalid_field == "schema":
        contract["schema_version"] = "leadpoet.invalid.v1"
    elif invalid_field == "hash":
        contract["function_authority_sha256"] = "sha256:" + "9" * 64
    elif invalid_field == "signature":
        contract["acquire_guard_signature"] = "text,integer"
    else:
        permissions = dict(contract["permissions"])
        permissions["anon_callable"] = True
        contract["permissions"] = permissions
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="claim-control contract is unavailable or invalid",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_admission_contract()),
                    (200, contract),
                ],
                requests,
            ),
        )

    assert len(requests) == 2
    assert all(
        maintenance.SOURCE_ADD_ACQUIRE_RESTART_GUARD_RPC
        not in str(request["path"])
        for request in requests
    )


def test_source_add_guard_acquisition_rejects_expired_or_wrong_guard():
    client = FakeSecretsClient(_source_add_secret())
    wrong_guard = {
        **_source_add_restart_guard(),
        "guard_commitment": "sha256:" + "9" * 64,
        "guard_expires_at": "2020-01-01T00:00:00+00:00",
    }

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="unexpected state",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_admission_contract()),
                    (200, _source_add_claim_control_contract()),
                    (200, _empty_source_add_guard_state()),
                    (200, wrong_guard),
                ],
                [],
            ),
        )


def test_source_add_restart_guard_identity_is_stable_across_exact_command_retry():
    first = maintenance._source_add_restart_guard_identity(
        "gateway-first-command-100"
    )
    retry = maintenance._source_add_restart_guard_identity(
        "gateway-retry-command-200"
    )

    assert first["guard_id"] == retry["guard_id"]
    assert first["guard_commitment"] == retry["guard_commitment"]
    assert first["owner_id"] != retry["owner_id"]
    assert first["owner_commitment"] != retry["owner_commitment"]
    assert first["actor_ref"] != retry["actor_ref"]


def test_fresh_retry_takeover_fences_stale_invocation_renewal_and_release():
    client = FakeSecretsClient(_source_add_secret())
    stale_invocation = "gateway-stale-command-100"
    fresh_invocation = "gateway-fresh-command-200"
    fresh_identity = maintenance._source_add_restart_guard_identity(
        fresh_invocation
    )
    fresh_control = _paused_source_add_control(
        actor_ref=fresh_identity["actor_ref"]
    )
    takeover_requests: list[dict[str, object]] = []

    fresh = REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
        secrets_client=client,
        restart_invocation_id=fresh_invocation,
        connection_factory=_source_add_connection_factory(
            [
                (200, _source_add_admission_contract()),
                (200, _source_add_claim_control_contract()),
                (
                    200,
                    _source_add_guard_state(
                        invocation_id=stale_invocation,
                        generation=1,
                    ),
                ),
                (200, _source_add_restart_guard(fresh_invocation, 2)),
                (200, [fresh_control]),
            ],
            takeover_requests,
        ),
    )

    assert fresh == {
        "status": "paused",
        "source_add_control_commitment": maintenance.sha256_json(
            fresh_control
        ),
        **_source_add_guard_result_fields(fresh_invocation, 2),
    }
    takeover_payload = json.loads(takeover_requests[3]["body"])
    assert takeover_payload["p_expected_generation"] == 1
    assert takeover_payload["p_owner_id"] == fresh_identity["owner_id"]

    for operation in (
        REAL_RENEW_SOURCE_ADD_RESTART_GUARD,
        REAL_RELEASE_SOURCE_ADD_RESTART_GUARD,
    ):
        stale_requests: list[dict[str, object]] = []
        with pytest.raises(
            maintenance.GatewayMinerMaintenanceRestartError,
            match="not owned|ownership changed",
        ):
            operation(
                secrets_client=client,
                restart_invocation_id=stale_invocation,
                connection_factory=_source_add_connection_factory(
                    [
                        (
                            200,
                            _source_add_guard_state(
                                invocation_id=fresh_invocation,
                                generation=2,
                            ),
                        )
                    ],
                    stale_requests,
                ),
            )
        assert [request["path"] for request in stale_requests] == [
            f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_GUARD_STATE_RPC}"
        ]


def test_shutdown_renewal_uses_full_canonical_deadline_lease_without_generation_change():
    client = FakeSecretsClient(_source_add_secret())
    requests: list[dict[str, object]] = []
    before = {
        **_source_add_guard_state(),
        "guard_expires_at": "2098-01-01T00:00:00+00:00",
    }
    renewed = {
        **_source_add_restart_guard(),
        "guard_expires_at": "2099-01-01T00:00:00+00:00",
    }

    result = REAL_RENEW_SOURCE_ADD_RESTART_GUARD(
        secrets_client=client,
        restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
        expected_guard_generation="1",
        expected_owner_generation_commitment=(
            _restart_owner_generation_commitment()
        ),
        connection_factory=_source_add_connection_factory(
            [(200, before), (200, renewed)], requests
        ),
    )

    assert result == {
        "status": "renewed",
        **_source_add_guard_result_fields(),
    }
    payload = json.loads(requests[1]["body"])
    assert payload["p_expected_generation"] == 1
    assert payload["p_lease_seconds"] == 14_400
    assert payload["p_lease_seconds"] == (
        maintenance.SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS
        + maintenance.SOURCE_ADD_RESTART_GUARD_SAFETY_MARGIN_SECONDS
    )
    assert payload["p_lease_seconds"] > 9_300


def test_source_add_pause_rejects_pre_145_or_disabled_admission_contract():
    client = FakeSecretsClient(_source_add_secret())
    contract = {**_source_add_admission_contract(), "trigger_enabled": False}
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="admission-control contract is unavailable",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id="gateway-source-add-old-contract",
            connection_factory=_source_add_connection_factory(
                [(200, contract)], requests
            ),
        )

    assert len(requests) == 1


def test_source_add_pause_rejects_invalid_or_changed_readback():
    client = FakeSecretsClient(_source_add_secret())
    invocation_id = "gateway-source-add-invalid-readback"
    actor_ref = "gateway-restart:" + hashlib.sha256(
        invocation_id.encode("utf-8")
    ).hexdigest()
    row = _paused_source_add_control(actor_ref=actor_ref)
    resumed = {**row, "paused": False, "reason": "operator_resume"}
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="SOURCE_ADD pause changed during guarded readback",
    ):
        REAL_PAUSE_SOURCE_ADD_FOR_RESTART(
            secrets_client=client,
            restart_invocation_id=invocation_id,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_admission_contract()),
                    (200, _source_add_claim_control_contract()),
                    (200, _empty_source_add_guard_state()),
                    (200, _source_add_restart_guard(invocation_id)),
                    (200, [resumed]),
                ],
                requests,
            ),
        )

    assert len(requests) == 5


def test_source_add_control_normalizer_accepts_exact_active_readback():
    active = {
        **_paused_source_add_control(actor_ref="operator:source-add-active"),
        "paused": False,
        "reason": "operator_source_add_active",
    }

    assert maintenance._normalized_source_add_control(active) == {
        key: active[key] for key in sorted(active)
    }


def test_restart_completion_failure_forces_and_verifies_durable_pause():
    client = FakeSecretsClient(_source_add_secret())
    requests: list[dict[str, object]] = []
    actor_ref = "gateway-restart:" + hashlib.sha256(
        DEFAULT_RESTART_INVOCATION_ID.encode("utf-8")
    ).hexdigest()

    REAL_FORCE_SOURCE_ADD_PAUSED_AFTER_RESTART_FAILURE(
        secrets_client=client,
        restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
        connection_factory=_source_add_connection_factory(
            [
                (200, {"paused": True}),
                (200, [_paused_source_add_control(actor_ref=actor_ref)]),
            ],
            requests,
        ),
    )

    assert requests[0]["path"] == (
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_PAUSE_RPC}"
    )
    assert json.loads(requests[0]["body"]) == {
        "p_actor_ref": actor_ref,
        "p_paused": True,
        "p_reason": "canonical_restart_completion_failed",
    }
    assert requests[1]["method"] == "GET"


def test_source_add_quiescence_wait_polls_every_lease_until_exact_zero(
    capsys: pytest.CaptureFixture[str],
):
    service_role_key = "unit-test-quiescence-service-role-private"
    client = FakeSecretsClient(_source_add_secret(service_role_key))
    requests: list[dict[str, object]] = []
    responses = [
        (200, _source_add_guard_state()),
        (200, _source_add_quiescence(leased_work_count=2)),
        (200, _source_add_quiescence(leased_work_count=1)),
        (200, _source_add_quiescence()),
    ]
    clock = iter([0.0, 0.0, 0.5])
    sleeps: list[float] = []

    result = REAL_WAIT_FOR_SOURCE_ADD_QUIESCENCE(
        secrets_client=client,
        restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
        connection_factory=_source_add_connection_factory(responses, requests),
        timeout_seconds=10.0,
        poll_seconds=1.0,
        monotonic=lambda: next(clock),
        sleep=sleeps.append,
    )

    expected_state = _source_add_quiescence()
    assert result == {
        "status": "quiescent",
        **_source_add_guard_result_fields(),
        "source_add_quiescence_commitment": (
            maintenance._source_add_quiescence_commitment(expected_state)
        ),
    }
    assert responses == []
    assert sleeps == [1.0, 1.0]
    assert [request["method"] for request in requests] == ["POST"] * 4
    assert [request["path"] for request in requests] == [
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_GUARD_STATE_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_QUIESCENCE_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_QUIESCENCE_RPC}",
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_QUIESCENCE_RPC}"
    ]
    identity = maintenance._source_add_restart_guard_identity(
        DEFAULT_RESTART_INVOCATION_ID
    )
    assert json.loads(requests[0]["body"]) == {}
    assert all(
        json.loads(request["body"])
        == {
            "p_guard_generation": DEFAULT_SOURCE_ADD_GUARD_GENERATION,
            "p_guard_id": identity["guard_id"],
            "p_owner_id": identity["owner_id"],
        }
        for request in requests[1:]
    )
    assert service_role_key not in json.dumps(result)
    assert capsys.readouterr() == ("", "")


def test_source_add_quiescence_wait_times_out_fail_closed_without_secret_leak(
    capsys: pytest.CaptureFixture[str],
):
    service_role_key = "unit-test-timeout-service-role-private"
    client = FakeSecretsClient(_source_add_secret(service_role_key))
    before = dict(client.versions)
    requests: list[dict[str, object]] = []
    clock = iter([0.0, 1.0])

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="did not quiesce before the restart deadline",
    ):
        REAL_WAIT_FOR_SOURCE_ADD_QUIESCENCE(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_guard_state()),
                    (200, _source_add_quiescence(leased_work_count=1)),
                ],
                requests,
            ),
            timeout_seconds=1.0,
            poll_seconds=0.1,
            monotonic=lambda: next(clock),
            sleep=lambda _seconds: pytest.fail("timeout must not sleep"),
        )

    assert client.versions == before
    assert len(requests) == 2
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "response",
    [
        _source_add_quiescence(paused=False),
        _source_add_quiescence(guard_active=False),
        _source_add_quiescence(guard_matches=False),
        _source_add_quiescence(leased_work_count=1),
        {
            **_source_add_quiescence(),
            "leased_work_count": True,
        },
        {
            **_source_add_quiescence(),
            "quiescent": False,
        },
    ],
)
def test_source_add_quiescence_readback_rejects_nonquiescent_or_invalid_state(
    response: dict[str, object],
):
    client = FakeSecretsClient(_source_add_secret())
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="restart-quiescence readback is invalid|work is not quiescent",
    ):
        REAL_REQUIRE_SOURCE_ADD_QUIESCENT(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [(200, _source_add_guard_state()), (200, response)], requests
            ),
        )

    assert len(requests) == 2


def test_source_add_quiescence_fails_closed_when_migration_rpc_is_unavailable():
    client = FakeSecretsClient(_source_add_secret())
    before = dict(client.versions)
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="pause authority request was rejected",
    ):
        REAL_REQUIRE_SOURCE_ADD_QUIESCENT(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_guard_state()),
                    (404, {"code": "PGRST202"}),
                ],
                requests,
            ),
        )

    assert client.versions == before
    assert requests[-1]["path"] == (
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RESTART_QUIESCENCE_RPC}"
    )


def test_source_add_quiescence_readback_binds_exact_proof_commitment():
    client = FakeSecretsClient(_source_add_secret())
    requests: list[dict[str, object]] = []

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from the invocation proof",
    ):
        REAL_REQUIRE_SOURCE_ADD_QUIESCENT(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            expected_quiescence_commitment="sha256:" + "9" * 64,
            connection_factory=_source_add_connection_factory(
                [
                    (200, _source_add_guard_state()),
                    (200, _source_add_quiescence()),
                ],
                requests,
            ),
        )

    assert len(requests) == 2


def test_source_add_quiescence_rejects_expired_guard_even_when_payload_claims_active():
    client = FakeSecretsClient(_source_add_secret())
    expired = {
        **_source_add_quiescence(),
        "guard_expires_at": "2020-01-01T00:00:00+00:00",
    }

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="work is not quiescent",
    ):
        REAL_REQUIRE_SOURCE_ADD_QUIESCENT(
            secrets_client=client,
            restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
            connection_factory=_source_add_connection_factory(
                [(200, _source_add_guard_state()), (200, expired)], []
            ),
        )


def test_source_add_restart_guard_release_is_exact_and_restores_active_state(
    capsys: pytest.CaptureFixture[str],
):
    service_role_key = "unit-test-release-service-role-private"
    client = FakeSecretsClient(_source_add_secret(service_role_key))
    requests: list[dict[str, object]] = []
    release = {
        "schema_version": "leadpoet.source_add_restart_guard_release.v2",
        "released": True,
        "paused": False,
        "guard_active": False,
        "guard_generation": DEFAULT_SOURCE_ADD_GUARD_GENERATION,
        "owner_generation_commitment": (
            _restart_owner_generation_commitment()
        ),
        "restored_pre_restart_state": True,
    }

    result = REAL_RELEASE_SOURCE_ADD_RESTART_GUARD(
        secrets_client=client,
        restart_invocation_id=DEFAULT_RESTART_INVOCATION_ID,
        connection_factory=_source_add_connection_factory(
            [(200, _source_add_guard_state()), (200, release)], requests
        ),
    )

    identity = maintenance._source_add_restart_guard_identity(
        DEFAULT_RESTART_INVOCATION_ID
    )
    assert result == {
        "status": "released_restored_active",
        "source_add_restart_guard_generation": "1",
        "source_add_restart_guard_owner_generation_commitment": (
            _restart_owner_generation_commitment()
        ),
        "source_add_restart_guard_restore_paused": "false",
    }
    assert requests[1]["path"] == (
        f"/rest/v1/rpc/{maintenance.SOURCE_ADD_RELEASE_RESTART_GUARD_RPC}"
    )
    assert json.loads(requests[1]["body"]) == {
        "p_actor_ref": identity["actor_ref"],
        "p_guard_generation": DEFAULT_SOURCE_ADD_GUARD_GENERATION,
        "p_guard_id": identity["guard_id"],
        "p_owner_id": identity["owner_id"],
    }
    assert "p_paused" not in json.loads(requests[1]["body"])
    assert service_role_key not in json.dumps(result)
    assert capsys.readouterr() == ("", "")


def test_prepare_pauses_source_before_global_secret_and_rechecks_without_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    calls: list[str] = []

    def pause(**_kwargs):
        calls.append("source_pause")
        assert len(client.versions) == 1
        assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n" in client.versions[
            client.current
        ]
        return {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
            **_source_add_guard_result_fields(),
        }

    def readback(**kwargs):
        calls.append("source_readback")
        assert kwargs["expected_control_commitment"] == (
            SOURCE_ADD_CONTROL_COMMITMENT
        )
        assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n" in client.versions[
            client.current
        ]
        return {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
        }

    def wait_for_quiescence(**_kwargs):
        calls.append("source_drain")
        assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n" in client.versions[
            client.current
        ]
        return {
            "status": "quiescent",
            **_source_add_guard_result_fields(),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        }

    def quiescence_readback(**kwargs):
        calls.append("source_quiescence_readback")
        assert kwargs["expected_quiescence_commitment"] == (
            SOURCE_ADD_QUIESCENCE_COMMITMENT
        )
        assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n" in client.versions[
            client.current
        ]
        return {
            "status": "quiescent",
            **_source_add_guard_result_fields(),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        }

    def runtime_status():
        calls.append("runtime_closed")
        return _closed_source_add_runtime_status()

    result = _prepare(
        tmp_path,
        monkeypatch,
        client,
        source_add_pause_hook=pause,
        source_add_readback_hook=readback,
        source_add_wait_hook=wait_for_quiescence,
        source_add_quiescence_readback_hook=quiescence_readback,
        runtime_status_hook=runtime_status,
    )

    assert calls == [
        "source_pause",
        "runtime_closed",
        "source_drain",
        "source_readback",
        "source_quiescence_readback",
        "runtime_closed",
    ]
    assert result["proof"]["source_add_control_commitment"] == (
        SOURCE_ADD_CONTROL_COMMITMENT
    )
    assert result["proof"]["source_add_quiescence_commitment"] == (
        SOURCE_ADD_QUIESCENCE_COMMITMENT
    )
    assert all("resume" not in call for call in calls)


def test_prepare_retry_with_proved_absent_gateway_skips_only_loopback_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    absent_commitment = maintenance.sha256_json({"status": "absent"})

    def unexpected_runtime_status():
        pytest.fail("proved-absent pre-hydration gateway has no loopback status")

    result = _prepare(
        tmp_path,
        monkeypatch,
        client,
        live_process_commitment=absent_commitment,
        runtime_status_hook=unexpected_runtime_status,
    )

    assert result["status"] == "prepared"
    assert result["proof"]["pre_hydration_live_process_commitment"] == (
        absent_commitment
    )
    assert result["proof"]["source_add_quiescence_commitment"] == (
        SOURCE_ADD_QUIESCENCE_COMMITMENT
    )
    assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n" in client.versions[
        client.current
    ]


def test_prepare_present_gateway_fails_closed_when_loopback_status_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    wait_called = False

    def unavailable_runtime_status():
        raise maintenance.GatewayMinerMaintenanceRestartError(
            "running Research Lab status is unavailable"
        )

    def wait_for_quiescence(**_kwargs):
        nonlocal wait_called
        wait_called = True
        return {
            "status": "quiescent",
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        }

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="status is unavailable",
    ):
        _prepare(
            tmp_path,
            monkeypatch,
            client,
            runtime_status_hook=unavailable_runtime_status,
            source_add_wait_hook=wait_for_quiescence,
        )

    assert wait_called is False
    assert client.current == INITIAL_VERSION
    assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n" in client.versions[
        client.current
    ]


def test_prepare_quiescence_timeout_aborts_before_global_secret_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")

    def timeout(**_kwargs):
        raise maintenance.GatewayMinerMaintenanceRestartError(
            "SOURCE_ADD work did not quiesce before the restart deadline"
        )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="did not quiesce before the restart deadline",
    ):
        _prepare(
            tmp_path,
            monkeypatch,
            client,
            source_add_wait_hook=timeout,
        )

    assert client.current == INITIAL_VERSION
    assert set(client.versions) == {INITIAL_VERSION}
    assert "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n" in client.versions[
        client.current
    ]


@pytest.mark.parametrize(
    "crash_point",
    ["after_stage", "after_promotion", "during_rollback"],
)
def test_prepare_recovers_crashed_secret_transaction_and_remains_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crash_point: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    client.versions[PREVIOUS_VERSION] = "OLDER=value\n"
    client.stages[PREVIOUS_VERSION] = {"AWSPREVIOUS"}
    client.versions[PENDING_VERSION] = "PENDING=value\n"
    client.stages[PENDING_VERSION] = {"AWSPENDING"}
    initial_topology = disable_operation._version_stages(client)
    prior_secret = client.versions[INITIAL_VERSION]
    candidate_secret, _document_format, status = (
        disable_operation._validated_candidate(prior_secret)
    )
    assert status == "verified"
    custom_label = disable_operation._custom_stage_label(RECOVERY_VERSION)
    journal_path = tmp_path / "private" / "transaction.json"
    disable_operation._write_recovery_journal(
        journal_path,
        disable_operation._recovery_journal_body(
            prior_version_id=INITIAL_VERSION,
            candidate_version_id=RECOVERY_VERSION,
            custom_stage_label=custom_label,
            initial_topology=initial_topology,
            prior_document_commitment=disable_operation._document_commitment(
                prior_secret
            ),
            candidate_document_commitment=disable_operation._document_commitment(
                candidate_secret
            ),
        ),
    )
    client.put_secret_value(
        SecretId=disable_operation.GATEWAY_SECRET_ID,
        SecretString=candidate_secret,
        ClientRequestToken=RECOVERY_VERSION,
        VersionStages=[custom_label],
    )
    if crash_point in {"after_promotion", "during_rollback"}:
        client.update_secret_version_stage(
            SecretId=disable_operation.GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=RECOVERY_VERSION,
            RemoveFromVersionId=INITIAL_VERSION,
        )
    if crash_point == "during_rollback":
        client.update_secret_version_stage(
            SecretId=disable_operation.GATEWAY_SECRET_ID,
            VersionStage="AWSCURRENT",
            MoveToVersionId=INITIAL_VERSION,
            RemoveFromVersionId=RECOVERY_VERSION,
        )

    first = _prepare(tmp_path, monkeypatch, client)
    second = _prepare(tmp_path, monkeypatch, client)

    assert first["status"] == "prepared"
    assert second["status"] == "prepared"
    assert not journal_path.exists()
    assert disable_operation._validated_candidate(client.versions[client.current])[2] == (
        "already_disabled"
    )
    topology = disable_operation._version_stages(client)
    assert topology[PENDING_VERSION] == frozenset({"AWSPENDING"})
    assert all(
        not any(
            stage.startswith(disable_operation._CUSTOM_STAGE_PREFIX)
            for stage in stages
        )
        for stages in topology.values()
    )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LEADPOET_GATEWAY_ENV_SECRET_ID", "another/secret"),
        ("AWS_REGION", "us-west-2"),
        ("AWS_DEFAULT_REGION", "eu-west-1"),
    ],
)
def test_prepare_rejects_nonproduction_authority_before_aws_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    before_versions = dict(client.versions)
    before_stages = {key: set(labels) for key, labels in client.stages.items()}
    monkeypatch.setenv(name, value)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="authority differs from production",
    ):
        _prepare(tmp_path, monkeypatch, client)

    assert client.versions == before_versions
    assert client.stages == before_stages


def test_fresh_same_and_different_candidate_retries_accept_already_false_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    first = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-first",
    )
    after_first_versions = dict(client.versions)
    after_first_stages = {key: set(labels) for key, labels in client.stages.items()}

    second = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-second",
    )
    different_commit = "f" * 40
    third = _prepare(
        tmp_path,
        monkeypatch,
        client,
        candidate_commit=different_commit,
        invocation_id="gateway-third",
    )

    assert first["proof"]["restart_invocation_id"] == "gateway-first"
    assert second["proof"]["restart_invocation_id"] == "gateway-second"
    assert third["proof"]["candidate_commit"] == different_commit
    assert len({
        first["proof"]["proof_hash"],
        second["proof"]["proof_hash"],
        third["proof"]["proof_hash"],
    }) == 3
    assert client.versions == after_first_versions
    assert client.stages == after_first_stages


def test_direct_restart_requires_durable_false_state(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        },
        secrets_client=client,
    )

    assert result["status"] == "durable_false_verified"
    assert result["current_secret_version_id"] == INITIAL_VERSION


def test_direct_restart_acquires_guard_for_exact_restart_invocation(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    invocation_id = "gateway-direct-guard-test"
    calls: list[tuple[str, object]] = []

    def acquire(**kwargs):
        calls.append(("acquire", kwargs["restart_invocation_id"]))
        return {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
            **_source_add_guard_result_fields(invocation_id),
        }

    def quiescent(**kwargs):
        calls.append(("quiescent", kwargs["expected_guard_commitment"]))
        assert kwargs["restart_invocation_id"] == invocation_id
        return {
            "status": "quiescent",
            **_source_add_guard_result_fields(invocation_id),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        }

    monkeypatch.setattr(maintenance, "_pause_source_add_for_restart", acquire)
    monkeypatch.setattr(maintenance, "_require_source_add_quiescent", quiescent)
    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "GATEWAY_RESTART_INVOCATION_ID": invocation_id,
        },
        secrets_client=client,
    )

    assert result["status"] == "durable_false_verified"
    assert calls == [
        ("acquire", invocation_id),
        ("quiescent", _restart_guard_commitment(invocation_id)),
    ]


def test_shutdown_boundary_requires_same_active_guard_and_zero_leases(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    calls: list[str] = []

    monkeypatch.setattr(
        maintenance,
        "_renew_source_add_restart_guard",
        lambda **_kwargs: calls.append("renew")
        or {
            "status": "renewed",
            **_source_add_guard_result_fields(),
        },
    )

    monkeypatch.setattr(
        maintenance,
        "_require_source_add_paused",
        lambda **_kwargs: calls.append("pause")
        or {
            "status": "paused",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
        },
    )

    def quiescent(**kwargs):
        calls.append("quiescence")
        assert kwargs["restart_invocation_id"] == DEFAULT_RESTART_INVOCATION_ID
        return {
            "status": "quiescent",
            **_source_add_guard_result_fields(),
            "source_add_quiescence_commitment": (
                SOURCE_ADD_QUIESCENCE_COMMITMENT
            ),
        }

    monkeypatch.setattr(maintenance, "_require_source_add_quiescent", quiescent)

    result = maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
        deploy_commit=CANDIDATE_COMMIT,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "GATEWAY_RESTART_INVOCATION_ID": DEFAULT_RESTART_INVOCATION_ID,
        },
        secrets_client=client,
    )

    assert result["status"] == "shutdown_quiescence_verified"
    assert calls == ["renew", "pause", "quiescence"]


def test_resume_after_preflight_blocks_shutdown_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_quiescent",
        lambda **_kwargs: (_ for _ in ()).throw(
            maintenance.GatewayMinerMaintenanceRestartError(
                "SOURCE_ADD work is not quiescent for restart"
            )
        ),
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="work is not quiescent for restart",
    ):
        maintenance.verify_gateway_miner_maintenance_shutdown_quiescence(
            deploy_commit=CANDIDATE_COMMIT,
            parent_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
                "GATEWAY_RESTART_INVOCATION_ID": (
                    DEFAULT_RESTART_INVOCATION_ID
                ),
            },
            secrets_client=client,
        )


def test_receiptless_second_restart_binds_actual_hydrated_cache_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    raw_secret = (
        "UNRELATED='preserved value'\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
    )
    client = FakeSecretsClient(raw_secret)
    hydrated = tmp_path / "private" / "gateway.env"
    hydrated.parent.mkdir(mode=0o700)
    hydrated.write_text(
        disable_operation._n_minus_one_hydrated_environment(raw_secret),
        encoding="utf-8",
    )
    hydrated.chmod(0o600)
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )
    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
            "AWS_REGION": "us-east-1",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
        secrets_client=client,
        hydrated_environment_path=hydrated,
    )

    assert result["status"] == "durable_false_verified"
    assert client.read_count >= 2

    hydrated.write_text(
        "UNRELATED=tampered\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n",
        encoding="utf-8",
    )
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="hydrated gateway environment differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
                "LEADPOET_AWS_INSTANCE_ROLE_ONLY": "true",
                "AWS_REGION": "us-east-1",
                "AWS_DEFAULT_REGION": "us-east-1",
            },
            secrets_client=client,
            hydrated_environment_path=hydrated,
        )


@pytest.mark.parametrize(
    ("parent_value", "secret_value", "message"),
    [
        ("true", "false", "did not hydrate"),
        ("false", "true", "durable gateway secret"),
    ],
)
def test_direct_restart_never_bypasses_false_state(
    monkeypatch: pytest.MonkeyPatch,
    parent_value: str,
    secret_value: str,
    message: str,
):
    client = FakeSecretsClient(
        f"RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED={secret_value}\n"
    )
    with pytest.raises(maintenance.GatewayMinerMaintenanceRestartError, match=message):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": parent_value,
            },
            secrets_client=client,
        )


def _verification_environment() -> dict[str, str]:
    return {
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        maintenance.PROOF_FD_ENV_NAME: str(maintenance.PROOF_FD_NUMBER),
        "GATEWAY_RESTART_INVOCATION_ID": "gateway-proof-test",
    }


def _fake_running_gateway_proc(
    tmp_path: Path,
    *,
    runtime_commit: str = CONTROLLER_COMMIT,
    controller_helper: str = maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
    identity_overrides: dict[str, str | None] | None = None,
) -> Path:
    proc_root = tmp_path / "proc"
    process = proc_root / "4242"
    process.mkdir(parents=True)
    (process / "cmdline").write_bytes(b"python3\0-m\0gateway.main\0")
    stat_fields = ["S", *("1" for _index in range(18)), "987654"]
    (process / "stat").write_text(
        f"4242 (python3) {' '.join(stat_fields)}\n",
        encoding="ascii",
    )
    overrides = identity_overrides or {}
    environment = []
    for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES:
        value = overrides.get(name, runtime_commit)
        if value is not None:
            environment.append(f"{name}={value}".encode("ascii"))
    environment.append(f"GATEWAY_GIT_HELPER={controller_helper}".encode("ascii"))
    (process / "environ").write_bytes(b"\0".join(environment) + b"\0")
    return proc_root


def _proof_with_current_hash(
    proof: dict[str, object],
    **updates: str,
) -> dict[str, object]:
    updated = {**proof, **updates}
    body = {
        name: str(updated[name])
        for name in maintenance._PROOF_FIELDS
        if name != "proof_hash"
    }
    updated["proof_hash"] = maintenance.sha256_json(body)
    return updated


@pytest.mark.parametrize("pointer", ["", "191", "not-a-fd"])
def test_open_fixed_proof_fd_cannot_be_downgraded_by_pointer(
    pointer: str,
):
    source = os.open("/dev/null", os.O_RDONLY)
    try:
        os.dup2(source, maintenance.PROOF_FD_NUMBER)
        with pytest.raises(
            maintenance.GatewayMinerMaintenanceRestartError,
            match="pointer was downgraded",
        ):
            maintenance._proof_fd_from_environment(
                {maintenance.PROOF_FD_ENV_NAME: pointer}
            )
    finally:
        os.close(source)
        os.close(maintenance.PROOF_FD_NUMBER)


def test_pointer_without_fixed_proof_fd_fails_closed():
    try:
        os.close(maintenance.PROOF_FD_NUMBER)
    except OSError:
        pass
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="descriptor was lost",
    ):
        maintenance._proof_fd_from_environment(
            {
                maintenance.PROOF_FD_ENV_NAME: str(
                    maintenance.PROOF_FD_NUMBER
                )
            }
        )


@pytest.mark.parametrize(
    "name",
    sorted(maintenance._RESTART_AUTHORITY_NAMES),
)
@pytest.mark.parametrize("value", ["", "/tmp/arbitrary-helper"])
def test_live_gateway_restart_authority_collision_fails_closed(
    name: str,
    value: str,
):
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="restart-only authority",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            f"SAFE=value\0{name}={value}\0".encode("ascii")
        )


def test_exact_frozen_n_minus_one_legacy_git_helper_is_bound_and_accepted():
    commit = maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
    payload = (
        "".join(
            f"{name}={commit}\0"
            for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
        )
        + "GATEWAY_GIT_HELPER="
        f"{maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER}\0"
    ).encode("ascii")

    authority = maintenance._require_restart_authority_absent_from_environment_payload(
        payload,
        expected_runtime_commit=commit,
        verified_controller_commit=commit,
        allow_legacy_n_minus_one_git_helper=True,
    )

    assert authority["restart_authority_names"] == ("GATEWAY_GIT_HELPER",)
    assert authority["runtime_build_identities"] == {
        name: commit for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    }


@pytest.mark.parametrize(
    ("runtime_commit", "controller_commit", "helper_path"),
    [
        (
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            "/tmp/gateway_git_deploy.py",
        ),
        (
            CANDIDATE_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
        ),
        (
            maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT,
            CANDIDATE_COMMIT,
            maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER,
        ),
    ],
)
def test_legacy_git_helper_near_misses_fail_closed(
    runtime_commit: str,
    controller_commit: str,
    helper_path: str,
):
    payload = (
        "".join(
            f"{name}={runtime_commit}\0"
            for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
        )
        + f"GATEWAY_GIT_HELPER={helper_path}\0"
    ).encode("ascii")

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="restart-only authority|build identity differs",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            payload,
            expected_runtime_commit=runtime_commit,
            verified_controller_commit=controller_commit,
            allow_legacy_n_minus_one_git_helper=True,
        )


@pytest.mark.parametrize("identity_name", maintenance.RUNTIME_BUILD_IDENTITY_NAMES)
@pytest.mark.parametrize("failure_kind", ["missing", "mismatch"])
def test_legacy_git_helper_requires_every_exact_build_identity(
    identity_name: str,
    failure_kind: str,
):
    commit = maintenance.LEGACY_N_MINUS_ONE_CONTROLLER_COMMIT
    records = []
    for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES:
        if name == identity_name and failure_kind == "missing":
            continue
        value = "b" * 40 if name == identity_name else commit
        records.append(f"{name}={value}\0")
    records.append(
        "GATEWAY_GIT_HELPER="
        f"{maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER}\0"
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="build identity differs",
    ):
        maintenance._require_restart_authority_absent_from_environment_payload(
            "".join(records).encode("ascii"),
            expected_runtime_commit=commit,
            verified_controller_commit=commit,
            allow_legacy_n_minus_one_git_helper=True,
        )


def test_candidate_runtime_requires_restart_authority_absent():
    payload = "".join(
        f"{name}={CANDIDATE_COMMIT}\0"
        for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    ).encode("ascii")

    authority = maintenance._require_restart_authority_absent_from_environment_payload(
        payload,
        expected_runtime_commit=CANDIDATE_COMMIT,
        verified_controller_commit=CANDIDATE_COMMIT,
    )
    assert authority["restart_authority_names"] == ()
    assert authority["runtime_build_identities"] == {
        name: CANDIDATE_COMMIT
        for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES
    }


def test_candidate_preflight_accepts_only_proof_bound_exact_n_minus_one_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    proc_root = _fake_running_gateway_proc(tmp_path)
    live_commitment = REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
        expected_runtime_commit=CONTROLLER_COMMIT,
        verified_controller_commit=CONTROLLER_COMMIT,
        allow_legacy_n_minus_one_git_helper=True,
        proc_root=proc_root,
    )
    proof = _proof_with_current_hash(
        proof,
        pre_hydration_live_process_commitment=live_commitment,
    )
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **kwargs: REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
            **kwargs,
            proc_root=proc_root,
        ),
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment=_verification_environment(),
        secrets_client=client,
    )

    assert result["status"] == "invocation_verified"
    assert result["proof_hash"] == proof["proof_hash"]


@pytest.mark.parametrize(
    "failure_case",
    [
        *(f"identity:{name}" for name in maintenance.RUNTIME_BUILD_IDENTITY_NAMES),
        "proof-runtime-candidate",
        "proof-controller-candidate",
        "helper-path-near-miss",
    ],
)
def test_candidate_preflight_rejects_n_minus_one_runtime_near_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
) -> None:
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    identity_overrides: dict[str, str | None] = {}
    helper_path = maintenance.LEGACY_N_MINUS_ONE_GATEWAY_GIT_HELPER
    if failure_case.startswith("identity:"):
        identity_overrides[failure_case.split(":", 1)[1]] = CANDIDATE_COMMIT
    elif failure_case == "proof-runtime-candidate":
        proof = _proof_with_current_hash(
            proof,
            pre_hydration_runtime_commit=CANDIDATE_COMMIT,
        )
    elif failure_case == "proof-controller-candidate":
        proof = _proof_with_current_hash(
            proof,
            n_minus_one_controller_commit=CANDIDATE_COMMIT,
        )
    elif failure_case == "helper-path-near-miss":
        helper_path = "/tmp/gateway_git_deploy.py"
    proc_root = _fake_running_gateway_proc(
        tmp_path,
        identity_overrides=identity_overrides,
        controller_helper=helper_path,
    )
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_live_gateway_restart_authority_commitment",
        lambda **kwargs: REAL_LIVE_GATEWAY_RESTART_AUTHORITY_COMMITMENT(
            **kwargs,
            proc_root=proc_root,
        ),
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="build identity differs|restart-only authority",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
        )


def test_candidate_preflight_binds_sealed_proof_to_exact_secret_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    prepared = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )
    proof = prepared["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )

    verified = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment=_verification_environment(),
        secrets_client=client,
    )
    assert verified["status"] == "invocation_verified"
    assert verified["proof_hash"] == proof["proof_hash"]

    client.install_concurrent_current()
    with pytest.raises(
        disable_operation.GatewayMinerSubmissionsDisableError,
        match="differs from the expected current version",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
        )

@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deploy_commit", "f" * 40),
        ("candidate_tree_hash", "f" * 40),
    ],
)
def test_candidate_preflight_rejects_proof_candidate_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    arguments = {
        "deploy_commit": CANDIDATE_COMMIT,
        "candidate_tree_hash": TREE_HASH,
        "parent_environment": _verification_environment(),
        "secrets_client": client,
    }
    arguments[field] = value

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from the candidate",
    ):
        maintenance.verify_gateway_miner_maintenance_state(**arguments)


def test_invocation_proof_rejects_durable_secret_topology_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    client.stages[client.current].add("UNEXPECTED_LABEL")

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="durable miner-maintenance state differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment=_verification_environment(),
            secrets_client=client,
        )

def test_invocation_proof_rejects_false_document_hydration_aba(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient(
        "CONFIG_GENERATION=v1\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-hydration-aba",
    )["proof"]
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )
    hydrated_path = tmp_path / "hydrated-cache" / "gateway.env"
    hydrated_path.parent.mkdir(mode=0o700)
    hydrated_path.write_text(
        disable_operation._n_minus_one_hydrated_environment(
            "CONFIG_GENERATION=v2\n"
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n"
        ),
        encoding="utf-8",
    )
    hydrated_path.chmod(0o600)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="hydrated gateway environment differs",
    ):
        maintenance.verify_gateway_miner_maintenance_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            parent_environment={
                **_verification_environment(),
                "GATEWAY_RESTART_INVOCATION_ID": "gateway-proof-hydration-aba",
            },
            secrets_client=client,
            hydrated_environment_path=hydrated_path,
        )


def test_identical_document_alternate_version_hydration_equivalence_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient(
        "CONFIG_GENERATION=v1\n"
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n"
    )
    invocation_id = "gateway-proof-identical-version-aba"
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id=invocation_id,
    )["proof"]
    proof_current = client.current
    proof_secret = client.versions[proof_current]
    proof_stages = {
        version: set(labels) for version, labels in client.stages.items()
    }
    client.versions[CONCURRENT_VERSION] = proof_secret
    client.stages[CONCURRENT_VERSION] = {"AWSCURRENT"}
    client.stages[proof_current].discard("AWSCURRENT")
    hydrated_path = tmp_path / "identical-cache" / "gateway.env"
    hydrated_path.parent.mkdir(mode=0o700)
    hydrated_path.write_text(
        disable_operation._n_minus_one_hydrated_environment(
            client.versions[CONCURRENT_VERSION]
        ),
        encoding="utf-8",
    )
    hydrated_path.chmod(0o600)
    client.stages = proof_stages
    client.stages[CONCURRENT_VERSION] = set()
    monkeypatch.setattr(maintenance, "_proof_from_fd", lambda _fd: proof)
    monkeypatch.setattr(
        maintenance,
        "_proof_fd_from_environment",
        lambda _environment: maintenance.PROOF_FD_NUMBER,
    )
    monkeypatch.setattr(
        maintenance,
        "_require_hydrated_environment_commitment",
        REAL_REQUIRE_HYDRATED_ENVIRONMENT_COMMITMENT,
    )

    result = maintenance.verify_gateway_miner_maintenance_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        parent_environment={
            **_verification_environment(),
            "GATEWAY_RESTART_INVOCATION_ID": invocation_id,
        },
        secrets_client=client,
        hydrated_environment_path=hydrated_path,
    )

    assert result["status"] == "invocation_verified"
    assert result["current_secret_version_id"] == proof_current


@pytest.mark.parametrize("runtime_value", [True, None, "false", 0])
def test_runtime_state_requires_live_boolean_exact_false(
    monkeypatch: pytest.MonkeyPatch,
    runtime_value,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="running gateway has miner submissions enabled",
    ):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            runtime_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            },
            runtime_status={
                "miner_submissions_enabled": runtime_value,
                **_closed_source_add_runtime_status(),
            },
            secrets_client=client,
        )


def test_runtime_state_rechecks_live_and_durable_false_state(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    monkeypatch.setattr(
        maintenance, "_fetch_runtime_status", _active_source_add_runtime_status
    )
    result = maintenance.verify_gateway_miner_maintenance_runtime_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        runtime_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
        },
        runtime_status={
            "miner_submissions_enabled": False,
            **_closed_source_add_runtime_status(),
        },
        secrets_client=client,
    )
    assert result["runtime_status"] == "disabled"
    assert result["status"] == "durable_false_verified"
    assert result["source_add_restart_guard_status"] == (
        "released_restored_active"
    )


def test_runtime_releases_guard_only_after_candidate_state_verifies(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    calls: list[str] = []

    def verify_state(**kwargs):
        calls.append("candidate_state")
        assert kwargs["acquire_source_add_restart_guard"] is False
        return {
            "status": "durable_false_verified",
            "current_secret_version_id": INITIAL_VERSION,
            **_source_add_guard_result_fields(),
        }

    def release(**kwargs):
        calls.append("release")
        assert kwargs["expected_current_version_id"] == INITIAL_VERSION
        assert kwargs["expected_guard_generation"] == "1"
        assert kwargs[
            "expected_owner_generation_commitment"
        ] == _restart_owner_generation_commitment()
        return {
            "status": "released_restored_active",
            "source_add_restart_guard_restore_paused": "false",
        }

    def restored(**kwargs):
        calls.append("restored_readback")
        assert kwargs["expected_paused"] is False
        return {
            "status": "active",
            "source_add_control_commitment": SOURCE_ADD_CONTROL_COMMITMENT,
        }

    monkeypatch.setattr(
        maintenance, "verify_gateway_miner_maintenance_state", verify_state
    )
    monkeypatch.setattr(maintenance, "_release_source_add_restart_guard", release)
    monkeypatch.setattr(maintenance, "_require_source_add_state", restored)
    monkeypatch.setattr(
        maintenance, "_fetch_runtime_status", _active_source_add_runtime_status
    )

    result = maintenance.verify_gateway_miner_maintenance_runtime_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        runtime_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "GATEWAY_RESTART_INVOCATION_ID": DEFAULT_RESTART_INVOCATION_ID,
        },
        runtime_status={
            "miner_submissions_enabled": False,
            **_closed_source_add_runtime_status(),
        },
        secrets_client=client,
    )

    assert calls == ["candidate_state", "release", "restored_readback"]
    assert result["source_add_restart_guard_status"] == (
        "released_restored_active"
    )


def test_runtime_restores_a_previously_paused_source_add_state(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")

    monkeypatch.setattr(
        maintenance,
        "verify_gateway_miner_maintenance_state",
        lambda **_kwargs: {
            "status": "durable_false_verified",
            "current_secret_version_id": INITIAL_VERSION,
            **{
                **_source_add_guard_result_fields(),
                "source_add_restart_guard_restore_paused": "true",
            },
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_release_source_add_restart_guard",
        lambda **_kwargs: {
            "status": "released_restored_paused",
            "source_add_restart_guard_restore_paused": "true",
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_runtime_status",
        _closed_source_add_runtime_status,
    )

    result = maintenance.verify_gateway_miner_maintenance_runtime_state(
        deploy_commit=CANDIDATE_COMMIT,
        candidate_tree_hash=TREE_HASH,
        runtime_environment={
            "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            "GATEWAY_RESTART_INVOCATION_ID": DEFAULT_RESTART_INVOCATION_ID,
        },
        runtime_status={
            "miner_submissions_enabled": False,
            **_closed_source_add_runtime_status(),
        },
        secrets_client=client,
    )

    assert result["source_add_restart_guard_status"] == (
        "released_restored_paused"
    )


def test_runtime_restoration_failure_forces_source_add_back_to_paused(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    forced: list[dict[str, object]] = []

    monkeypatch.setattr(
        maintenance,
        "verify_gateway_miner_maintenance_state",
        lambda **_kwargs: {
            "status": "durable_false_verified",
            "current_secret_version_id": INITIAL_VERSION,
            **_source_add_guard_result_fields(),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_fetch_runtime_status",
        _closed_source_add_runtime_status,
    )
    monkeypatch.setattr(
        maintenance,
        "_force_source_add_paused_after_restart_failure",
        lambda **kwargs: forced.append(dict(kwargs)),
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="did not restore active",
    ):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            runtime_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
                "GATEWAY_RESTART_INVOCATION_ID": DEFAULT_RESTART_INVOCATION_ID,
            },
            runtime_status={
                "miner_submissions_enabled": False,
                **_closed_source_add_runtime_status(),
            },
            secrets_client=client,
        )

    assert len(forced) == 1
    assert forced[0]["restart_invocation_id"] == DEFAULT_RESTART_INVOCATION_ID


def test_runtime_secret_read_failure_after_release_forces_source_add_paused(
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=false\n")
    forced: list[dict[str, object]] = []

    monkeypatch.setattr(
        maintenance,
        "verify_gateway_miner_maintenance_state",
        lambda **_kwargs: {
            "status": "durable_false_verified",
            "current_secret_version_id": INITIAL_VERSION,
            **_source_add_guard_result_fields(),
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_require_source_add_state",
        lambda **_kwargs: (_ for _ in ()).throw(
            disable_operation.GatewayMinerSubmissionsDisableError(
                "transient secret read failure"
            )
        ),
    )
    monkeypatch.setattr(
        maintenance,
        "_force_source_add_paused_after_restart_failure",
        lambda **kwargs: forced.append(dict(kwargs)),
    )

    with pytest.raises(
        disable_operation.GatewayMinerSubmissionsDisableError,
        match="transient secret read failure",
    ):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            runtime_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
                "GATEWAY_RESTART_INVOCATION_ID": DEFAULT_RESTART_INVOCATION_ID,
            },
            runtime_status={
                "miner_submissions_enabled": False,
                **_closed_source_add_runtime_status(),
            },
            secrets_client=client,
        )

    assert len(forced) == 1
    assert forced[0]["restart_invocation_id"] == DEFAULT_RESTART_INVOCATION_ID


@pytest.mark.parametrize(
    "source_add_override",
    [
        {"intake_enabled": True},
        {"effective_dispatcher_enabled": True},
        {"control": {"paused": False, "unavailable": False}},
        {"control": {"paused": True, "unavailable": True}},
    ],
)
def test_runtime_state_rejects_source_add_that_is_not_exactly_closed(
    source_add_override: dict[str, object],
):
    status = _closed_source_add_runtime_status()
    source_add = dict(status["source_add"])
    source_add.update(source_add_override)
    status["source_add"] = source_add

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="SOURCE_ADD intake is not durably paused",
    ):
        maintenance._require_runtime_source_add_closed(status)


def test_runtime_source_add_check_accepts_missing_intake_only_for_explicit_legacy_path():
    status = _closed_source_add_runtime_status()
    del status["source_add"]["intake_enabled"]

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="SOURCE_ADD intake is not durably paused",
    ):
        maintenance._require_runtime_source_add_closed(status)

    maintenance._require_runtime_source_add_closed(
        status,
        allow_legacy_missing_intake=True,
    )


def test_bootstrap_accepts_legacy_intake_projection_only_before_activation(
    monkeypatch: pytest.MonkeyPatch,
):
    status = _closed_source_add_runtime_status()
    del status["source_add"]["intake_enabled"]
    monkeypatch.setattr(maintenance, "_fetch_runtime_status", lambda: status)

    maintenance._require_pre_activation_runtime_source_add_closed()

    bootstrap_names = (
        maintenance.bootstrap_gateway_miner_maintenance_restart.__code__.co_names
    )
    assert "_require_pre_activation_runtime_source_add_closed" in bootstrap_names
    assert "_require_runtime_source_add_closed" not in bootstrap_names
    assert "_controller_exec_environment" in bootstrap_names


def test_bootstrap_handoff_accepts_the_canonical_paired_coordination_window():
    marker = Path(
        "/tmp/leadpoet-gateway-miner-maintenance-handoff.%s-test" % os.getpid()
    )
    nonce = "0" * 64
    try:
        marker.write_text("%s %s\n" % (CANDIDATE_COMMIT, nonce), encoding="ascii")
        marker.chmod(0o600)

        maintenance._wait_for_handoff_marker(
            path=marker,
            expected_commit=CANDIDATE_COMMIT,
            nonce=nonce,
            timeout_seconds=(
                maintenance.SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS
            ),
        )

        assert not marker.exists()
    finally:
        marker.unlink(missing_ok=True)


def test_bootstrap_handoff_accepts_marker_after_the_old_five_minute_limit(
    monkeypatch: pytest.MonkeyPatch,
):
    marker = Path(
        "/tmp/leadpoet-gateway-miner-maintenance-handoff.%s-late" % os.getpid()
    )
    nonce = "1" * 64
    clock = {"seconds": 0.0}

    def monotonic() -> float:
        return clock["seconds"]

    def sleep(seconds: float) -> None:
        clock["seconds"] += seconds
        if clock["seconds"] >= 301.0 and not marker.exists():
            marker.write_text(
                "%s %s\n" % (CANDIDATE_COMMIT, nonce), encoding="ascii"
            )
            marker.chmod(0o600)

    monkeypatch.setattr(maintenance.time, "monotonic", monotonic)
    monkeypatch.setattr(maintenance.time, "sleep", sleep)
    try:
        maintenance._wait_for_handoff_marker(
            path=marker,
            expected_commit=CANDIDATE_COMMIT,
            nonce=nonce,
            timeout_seconds=(
                maintenance.SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS
            ),
        )

        assert clock["seconds"] >= 301.0
        assert not marker.exists()
    finally:
        marker.unlink(missing_ok=True)


def test_bootstrap_handoff_still_fails_closed_at_its_deadline(
    monkeypatch: pytest.MonkeyPatch,
):
    marker = Path(
        "/tmp/leadpoet-gateway-miner-maintenance-handoff.%s-missing"
        % os.getpid()
    )
    clock = {"seconds": 0.0}
    monkeypatch.setattr(
        maintenance.time, "monotonic", lambda: clock["seconds"]
    )
    monkeypatch.setattr(
        maintenance.time,
        "sleep",
        lambda seconds: clock.__setitem__("seconds", clock["seconds"] + seconds),
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="did not provide a bounded miner-maintenance handoff",
    ):
        maintenance._wait_for_handoff_marker(
            path=marker,
            expected_commit=CANDIDATE_COMMIT,
            nonce="2" * 64,
            timeout_seconds=2,
        )


def test_bootstrap_handoff_cancel_and_bad_nonce_fail_closed():
    marker = Path(
        "/tmp/leadpoet-gateway-miner-maintenance-handoff.%s-cancel" % os.getpid()
    )
    nonce = "3" * 64
    try:
        marker.write_text(
            "failed:%s %s\n" % (CANDIDATE_COMMIT, nonce), encoding="ascii"
        )
        marker.chmod(0o600)
        with pytest.raises(
            maintenance.GatewayMinerMaintenanceRestartError,
            match="paired operator cancelled",
        ):
            maintenance._wait_for_handoff_marker(
                path=marker,
                expected_commit=CANDIDATE_COMMIT,
                nonce=nonce,
                timeout_seconds=1,
            )
    finally:
        marker.unlink(missing_ok=True)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="handoff request is invalid",
    ):
        maintenance._wait_for_handoff_marker(
            path=marker,
            expected_commit=CANDIDATE_COMMIT,
            nonce="bad-nonce",
            timeout_seconds=1,
        )


def test_bootstrap_uses_the_canonical_paired_coordination_deadline():
    source = Path(maintenance.__file__).read_text(encoding="utf-8")
    bootstrap = source.split(
        "def bootstrap_gateway_miner_maintenance_restart(", 1
    )[1].split("\ndef ", 1)[0]

    assert (
        "timeout_seconds=SOURCE_ADD_CANONICAL_COORDINATION_DEADLINE_SECONDS"
        in bootstrap
    )


def test_candidate_bootstrap_protected_source_binding_matches_current_source():
    # Bootstrap calls this same fixed-purpose verifier before it mutates or
    # waits.  Use the real candidate manifest so a changed protected helper
    # cannot pass tests with only placeholder commitments.
    maintenance._verify_protected_source()


def test_candidate_runtime_rejects_missing_source_add_intake_field(
    monkeypatch: pytest.MonkeyPatch,
):
    status = {
        "miner_submissions_enabled": False,
        **_closed_source_add_runtime_status(),
    }
    del status["source_add"]["intake_enabled"]
    state_verification_called = False

    def verify_state(**_kwargs):
        nonlocal state_verification_called
        state_verification_called = True
        return {"status": "durable_false_verified"}

    monkeypatch.setattr(
        maintenance,
        "verify_gateway_miner_maintenance_state",
        verify_state,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="SOURCE_ADD intake is not durably paused",
    ):
        maintenance.verify_gateway_miner_maintenance_runtime_state(
            deploy_commit=CANDIDATE_COMMIT,
            candidate_tree_hash=TREE_HASH,
            runtime_environment={
                "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "false",
            },
            runtime_status=status,
            secrets_client=object(),
        )

    assert state_verification_called is False


def test_runtime_status_fetch_uses_exact_loopback_path_and_never_follows_redirects(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[tuple[object, ...]] = []

    class RedirectResponse:
        status = 302

        def getheader(self, _name):
            return None

    class FakeConnection:
        def __init__(self, host, port, timeout):
            requests.append(("connect", host, port, timeout))

        def request(self, method, path, body, headers):
            requests.append(("request", method, path, body, dict(headers)))

        def getresponse(self):
            return RedirectResponse()

        def close(self):
            requests.append(("close",))

    monkeypatch.setattr(maintenance.http.client, "HTTPConnection", FakeConnection)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="response is not successful",
    ):
        maintenance._fetch_runtime_status()

    assert requests == [
        ("connect", "127.0.0.1", 8000, 15.0),
        (
            "request",
            "GET",
            "/research-lab/status",
            None,
            {"Host": "127.0.0.1:8000", "Connection": "close"},
        ),
        ("close",),
    ]
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="URL is not canonical",
    ):
        maintenance._fetch_runtime_status(
            url="http://169.254.169.254/latest/meta-data/",
        )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create")
    or not hasattr(maintenance.fcntl, "F_ADD_SEALS"),
    reason="Linux sealed memfd behavior",
)
def test_sealed_invocation_proof_survives_exec_and_rejects_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client = FakeSecretsClient("RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED=true\n")
    proof = _prepare(
        tmp_path,
        monkeypatch,
        client,
        invocation_id="gateway-proof-test",
    )["proof"]
    try:
        maintenance._seal_payload_at_fd_number(
            payload=maintenance._serialized_proof(proof),
            fd_number=maintenance.PROOF_FD_NUMBER,
            name="test-proof",
            max_bytes=maintenance.MAX_PROOF_BYTES,
        )
        assert maintenance._proof_from_fd(maintenance.PROOF_FD_NUMBER) == proof
        with pytest.raises(OSError):
            os.write(maintenance.PROOF_FD_NUMBER, b"tamper")
        environment = dict(os.environ)
        environment[maintenance.PROOF_FD_ENV_NAME] = str(
            maintenance.PROOF_FD_NUMBER
        )
        child = subprocess.run(
            [
                "/bin/bash",
                "-c",
                'test "$GATEWAY_MINER_MAINTENANCE_PROOF_FD" = 190 '
                '&& test -r /proc/$$/fd/190 '
                '&& exec python3 -c "import os; os.fstat(190)"',
            ],
            check=False,
            env=environment,
            pass_fds=(maintenance.PROOF_FD_NUMBER,),
        )
        assert child.returncode == 0
    finally:
        try:
            os.close(maintenance.PROOF_FD_NUMBER)
        except OSError:
            pass


@pytest.mark.parametrize("parent_value", [None, "true"])
def test_controller_exec_carries_proved_disabled_miner_submissions(parent_value):
    parent = {"UNRELATED": "preserved"}
    if parent_value is not None:
        parent[disable_operation.TARGET_ENV_NAME] = parent_value

    environment = maintenance._controller_exec_environment(parent)

    assert environment == {
        "UNRELATED": "preserved",
        disable_operation.TARGET_ENV_NAME: disable_operation.TARGET_ENV_VALUE,
    }
    assert parent == (
        {"UNRELATED": "preserved"}
        if parent_value is None
        else {
            "UNRELATED": "preserved",
            disable_operation.TARGET_ENV_NAME: parent_value,
        }
    )


@pytest.mark.skipif(
    not hasattr(os, "memfd_create")
    or not hasattr(maintenance.fcntl, "F_ADD_SEALS"),
    reason="Linux sealed memfd behavior",
)
def test_closed_or_tampered_proof_fd_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    try:
        os.close(maintenance.PROOF_FD_NUMBER)
    except OSError:
        pass
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="unavailable",
    ):
        maintenance._proof_from_fd(maintenance.PROOF_FD_NUMBER)

    proof = {
        name: "invalid"
        for name in maintenance._PROOF_FIELDS
    }
    proof["schema_version"] = maintenance.SCHEMA_VERSION
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="commitments are invalid",
    ):
        maintenance._validate_proof_document(proof)


def test_bootstrap_cleanup_leaves_a_valid_exec_working_directory() -> None:
    bootstrap_root = Path(
        f"/tmp/gateway-miner-maintenance-bootstrap.{os.getpid()}cwd"
    )
    candidate_root = bootstrap_root / "candidate"
    nested_root = candidate_root / "nested"
    original_cwd = os.open(".", os.O_RDONLY)
    try:
        bootstrap_root.mkdir(mode=0o700)
        candidate_root.mkdir()
        nested_root.mkdir()
        payload = nested_root / "payload"
        payload.write_bytes(b"verified archive payload")
        payload.chmod(0o400)
        nested_root.chmod(0o500)
        candidate_root.chmod(0o500)
        os.chdir(candidate_root)

        maintenance._leave_and_close_bootstrap_tree(bootstrap_root)

        assert Path.cwd() == Path("/")
        assert not bootstrap_root.exists()
    finally:
        os.fchdir(original_cwd)
        os.close(original_cwd)
        if bootstrap_root.exists():
            for directory, _names, _files in os.walk(
                bootstrap_root,
                topdown=False,
                followlinks=False,
            ):
                Path(directory).chmod(0o700)
            maintenance.shutil.rmtree(bootstrap_root)


def test_unexpected_cli_failure_never_renders_exception_detail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    secret_marker = "raw-secret-must-not-render"
    monkeypatch.setattr(
        maintenance,
        "bootstrap_gateway_miner_maintenance_restart",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError(secret_marker)),
    )

    status = maintenance.main(
        [
            "--bootstrap-exec",
            "--expected-commit",
            CANDIDATE_COMMIT,
            "--plan-file",
            "/tmp/nonexistent-plan",
            "--bootstrap-root",
            "/tmp/gateway-miner-maintenance-bootstrap.test",
            "--handoff-file",
            "/tmp/leadpoet-gateway-miner-maintenance-handoff.test",
            "--handoff-nonce",
            "0" * 64,
        ]
    )

    captured = capsys.readouterr()
    assert status == 2
    assert secret_marker not in captured.err
    assert "unexpected miner-maintenance restart failure" in captured.err


def test_candidate_identity_binds_isolated_n_minus_one_plan_and_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo = tmp_path / "repo"
    candidate = tmp_path / "candidate"
    repo.mkdir()
    candidate.mkdir()
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": maintenance.GIT_DEPLOYMENT_SCHEMA_VERSION,
                "source": "github",
                "status": "prepared",
                "stage": "git_prepare",
                "mode": "pinned",
                "branch": maintenance.DEFAULT_BRANCH,
                "target_sha": CANDIDATE_COMMIT,
                "branch_head_sha": CANDIDATE_COMMIT,
                "repo_root": str(repo.resolve()),
                "remote_url": maintenance.DEFAULT_REPO_URL,
                "previous_sha": "9" * 40,
                "tree_hash": TREE_HASH,
            }
        ),
        encoding="utf-8",
    )
    responses = {
        ("rev-parse", "HEAD"): "9" * 40,
        ("rev-parse", "origin/main^{commit}"): CANDIDATE_COMMIT,
        ("remote", "get-url", "origin"): maintenance.DEFAULT_REPO_URL,
        ("status", "--porcelain=v1", "--untracked-files=all"): "",
    }
    monkeypatch.setattr(
        maintenance,
        "_run_git",
        lambda _repo, *arguments: responses[arguments],
    )
    monkeypatch.setattr(
        maintenance,
        "_require_unmodified_git_object_authority",
        lambda _repo: None,
    )
    monkeypatch.setattr(
        maintenance,
        "verify_materialized_tree",
        lambda **_kwargs: {
            "tree_hash": TREE_HASH,
            "blob_manifest_sha256": BLOB_HASH,
        },
    )
    monkeypatch.setattr(
        maintenance,
        "_verified_installed_controller_bundle",
        lambda **_kwargs: _controller_bundle(),
    )
    evidence = maintenance._validate_candidate_identity(
        repo_root=repo,
        candidate_root=candidate,
        plan_file=plan,
        expected_commit=CANDIDATE_COMMIT,
        controller_current=tmp_path / "controller/current",
        host_restart_path=tmp_path / "gw_restart.sh",
    )

    assert evidence["tree_hash"] == TREE_HASH
    tampered = json.loads(plan.read_text(encoding="utf-8"))
    tampered["branch_head_sha"] = "f" * 40
    plan.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from the exact candidate",
    ):
        maintenance._validate_candidate_identity(
            repo_root=repo,
            candidate_root=candidate,
            plan_file=plan,
            expected_commit=CANDIDATE_COMMIT,
            controller_current=tmp_path / "controller/current",
            host_restart_path=tmp_path / "gw_restart.sh",
        )


def test_git_replacement_and_graft_authority_fail_before_candidate_resolution(
    tmp_path: Path,
):
    repository = tmp_path / "git-authority"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("official\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "official",
        ],
        check=True,
    )
    official = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked.write_text("replacement\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "replacement",
        ],
        check=True,
    )
    replacement = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ["git", "-C", str(repository), "replace", official, replacement],
        check=True,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="replacement refs",
    ):
        maintenance._require_unmodified_git_object_authority(repository)

    subprocess.run(
        ["git", "-C", str(repository), "replace", "-d", official],
        check=True,
    )
    graft = Path(
        subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "--git-path", "info/grafts"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not graft.is_absolute():
        graft = repository / graft
    graft.parent.mkdir(parents=True, exist_ok=True)
    graft.write_text(f"{replacement} {official}\n", encoding="ascii")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="graft or alternate",
    ):
        maintenance._require_unmodified_git_object_authority(repository)


def test_git_object_override_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("GIT_OBJECT_DIRECTORY", "/tmp/unsafe-objects")
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="object-resolution overrides",
    ):
        maintenance._safe_git_environment()


def test_live_0775_controller_ancestry_is_hardened_and_all_four_files_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        controller_parent,
        controller_root,
        releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)

    observed = maintenance._verify_installed_controller(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    )

    assert observed == CONTROLLER_COMMIT
    assert [
        path.stat().st_mode & 0o777
        for path in (controller_parent, controller_root, releases_root)
    ] == [0o700, 0o700, 0o700]


def test_controller_hardening_rejects_wrong_owner_and_symlink_ancestry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)
    actual_euid = os.geteuid()
    monkeypatch.setattr(maintenance.os, "geteuid", lambda: actual_euid + 1)
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="ancestry is unsafe",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )
    monkeypatch.undo()

    (
        _controller_parent,
        second_root,
        _releases_root,
        _release,
        second_current,
        second_host,
    ) = _installed_controller_fixture(
        tmp_path / "second",
        monkeypatch,
    )
    real_root = second_root.with_name("gateway-real")
    second_root.rename(real_root)
    second_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="ancestry",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=second_current,
            host_restart_path=second_host,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_controller_verifier_rejects_tampered_memory_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)
    memory_guard = release / "gateway/tee/host_memory_guard_v2.py"
    memory_guard.write_bytes(b"TAMPERED = True\n")
    memory_guard.chmod(0o600)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="bytes differ",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_exact_candidate_controller_is_allowed_for_post_install_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=CANDIDATE_COMMIT,
    )

    assert maintenance._verify_installed_controller(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
    ) == CANDIDATE_COMMIT


def test_real_sequential_release_accepts_d649_controller_for_d72(
    tmp_path: Path,
) -> None:
    current, host_restart = _real_installed_controller_fixture(
        tmp_path,
        controller_commit=SEQUENTIAL_N_MINUS_ONE_COMMIT,
    )

    assert maintenance._verify_installed_controller(
        repo_root=REAL_REPOSITORY,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=SEQUENTIAL_CANDIDATE_COMMIT,
    ) == SEQUENTIAL_N_MINUS_ONE_COMMIT


def test_real_controller_lineage_rejects_pre_floor_commit(
    tmp_path: Path,
) -> None:
    controller_commit = subprocess.check_output(
        ["git", "rev-parse", f"{CONTROLLER_COMMIT}^"],
        cwd=REAL_REPOSITORY,
        text=True,
    ).strip()
    current, host_restart = _real_installed_controller_fixture(
        tmp_path,
        controller_commit=controller_commit,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="not compatible",
    ):
        maintenance._verify_installed_controller(
            repo_root=REAL_REPOSITORY,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=SEQUENTIAL_CANDIDATE_COMMIT,
        )


@pytest.mark.parametrize("floor_is_ancestor", (False, True))
def test_unrelated_and_non_ancestor_controllers_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    floor_is_ancestor: bool,
) -> None:
    lineage_controller = "e" * 40
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=lineage_controller,
    )

    def lineage(_repository: Path, ancestor: str, descendant: str) -> bool:
        if ancestor == CONTROLLER_COMMIT and descendant == lineage_controller:
            return floor_is_ancestor
        if ancestor == lineage_controller and descendant == CANDIDATE_COMMIT:
            return False
        raise AssertionError((ancestor, descendant))

    monkeypatch.setattr(maintenance, "_git_is_ancestor", lineage)
    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="not compatible",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_missing_installed_controller_object_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_commit = "f" * 40
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=missing_commit,
    )
    monkeypatch.setattr(
        maintenance,
        "_git_commit_exists",
        lambda _repository, commit: commit != missing_commit,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="Git object is unavailable",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_missing_candidate_object_fails_closed_after_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        _release,
        current,
        host_restart,
    ) = _installed_controller_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        maintenance,
        "_git_commit_exists",
        lambda _repository, commit: commit != CANDIDATE_COMMIT,
    )

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="Git object is unavailable",
    ):
        maintenance._verify_installed_controller(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )


def test_partial_controller_cutover_reconciles_exact_old_host_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (
        _controller_parent,
        _controller_root,
        _releases_root,
        release,
        current,
        host_restart,
    ) = _installed_controller_fixture(
        tmp_path,
        monkeypatch,
        controller_commit=CANDIDATE_COMMIT,
    )
    candidate_payloads = {
        "gw_restart.sh": (release / "gw_restart.sh").read_bytes(),
        "scripts/gateway_git_deploy.py": (
            release / "scripts/gateway_git_deploy.py"
        ).read_bytes(),
        "Leadpoet/utils/exact_commit_restart_v2.py": (
            release / "Leadpoet/utils/exact_commit_restart_v2.py"
        ).read_bytes(),
        "gateway/tee/host_memory_guard_v2.py": (
            release / "gateway/tee/host_memory_guard_v2.py"
        ).read_bytes(),
    }
    old_wrapper = b"#!/bin/bash\n# exact supported N-1 wrapper\nexit 0\n"
    host_restart.write_bytes(old_wrapper)
    host_restart.chmod(0o700)

    def git_bytes(_repo, _show, object_name):
        commit, relative_path = object_name.split(":", 1)
        if commit == CANDIDATE_COMMIT:
            return candidate_payloads[relative_path]
        assert commit == CONTROLLER_COMMIT
        assert relative_path == "gw_restart.sh"
        return old_wrapper

    monkeypatch.setattr(maintenance, "_run_git_bytes", git_bytes)

    with pytest.raises(
        maintenance.GatewayMinerMaintenanceRestartError,
        match="differs from current controller",
    ):
        maintenance._verified_installed_controller_bundle(
            repo_root=tmp_path,
            controller_current=current,
            host_restart_path=host_restart,
            expected_commit=CANDIDATE_COMMIT,
        )

    bundle = maintenance._verified_installed_controller_bundle(
        repo_root=tmp_path,
        controller_current=current,
        host_restart_path=host_restart,
        expected_commit=CANDIDATE_COMMIT,
        reconcile_host_wrapper=True,
    )

    assert bundle["controller_commit"] == CANDIDATE_COMMIT
    assert host_restart.read_bytes() == candidate_payloads["gw_restart.sh"]
    assert host_restart.stat().st_mode & 0o777 == 0o700


def test_exact_deployed_n_minus_one_preserves_proof_until_candidate_gates():
    root = Path(__file__).resolve().parents[1]
    deployed_n_minus_one = subprocess.run(
        [
            "git",
            "show",
            "0dd3a385a23a3af0fa17210bfe02a39cc4023952:gw_restart.sh",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout
    hydration = deployed_n_minus_one.index(
        "Hydrating gateway env from Secrets Manager before stopping processes"
    )
    git_prepare = deployed_n_minus_one.index(
        'GATEWAY_DEPLOY_STAGE="git_prepare"', hydration
    )
    archive = deployed_n_minus_one.index(
        'git -C "$LEADPOET_REPO_ROOT" archive "$PREPARED_GATEWAY_SHA"',
        git_prepare,
    )
    candidate_preflight = deployed_n_minus_one.index(
        "gateway.tee.restart_preflight_v2", archive
    )
    shutdown = deployed_n_minus_one.index(
        "Stopping existing gateway and Research Lab worker processes",
        candidate_preflight,
    )
    assert hydration < git_prepare < archive < candidate_preflight < shutdown
    post_activate = deployed_n_minus_one.index("exec env", shutdown)
    assert "env -i" not in deployed_n_minus_one[post_activate:post_activate + 200]
    assert "190>&-" not in deployed_n_minus_one

    preflight_source = (root / "gateway/tee/restart_preflight_v2.py").read_text(
        encoding="utf-8"
    )
    tree_verification = preflight_source.index("write_tree_verification_evidence")
    state_gate = preflight_source.index(
        "verify_gateway_miner_maintenance_state(",
        tree_verification,
    )
    shared_aws_authority = preflight_source.index(
        "aws_clients = _instance_role_aws_clients(",
        tree_verification,
    )
    output = preflight_source.index(
        "print(json.dumps(result, sort_keys=True, indent=2))",
        state_gate,
    )
    assert tree_verification < shared_aws_authority < state_gate < output
    assert 'boto3.client("s3")' not in preflight_source
    assert 'artifact_s3_client=aws_clients["s3"]' in preflight_source
    assert 'secrets_client=aws_clients["secretsmanager"]' in preflight_source

    candidate_restart = (root / "gw_restart.sh").read_text(encoding="utf-8")
    install = candidate_restart.index(
        'GATEWAY_DEPLOY_STAGE="host_restart_script_install"'
    )
    runtime_verify = candidate_restart.index("--verify-runtime", install)
    finalize = candidate_restart.index(
        "finalize_deployment_record succeeded", runtime_verify
    )
    close_parent = candidate_restart.index(
        "exec 190>&- 191>&- 192>&- 193>&- 194>&-",
        finalize,
    )
    completed = candidate_restart.index("GATEWAY_DEPLOY_COMPLETED=1", close_parent)
    assert install < runtime_verify < finalize < close_parent < completed


def test_long_lived_runtime_children_receive_no_proof_or_controller_fds():
    restart = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    close_set = "190>&- 191>&- 192>&- 193>&- 194>&-"
    for module_name in (
        "gateway.utils.tee_egress_forwarder",
        "gateway.utils.tee_inter_enclave_relay",
        "gateway.main",
    ):
        position = restart.rindex(f"-m {module_name}")
        position = restart.rfind("env -u", 0, position)
        command_end = restart.index("\n\n", position)
        command = restart[position:command_end]
        assert close_set in command
        assert "-u GATEWAY_MINER_MAINTENANCE_PROOF_FD" in command
        assert "-u GATEWAY_GIT_HELPER" in command
        assert "-u GATEWAY_EXACT_COMMIT_HELPER" in command
        assert "-u GATEWAY_HOST_MEMORY_GUARD_PATH" in command
    for function_name, marker in (
        ("start_gateway_offline_artifact_prepare", '"${prepare_command[@]}"'),
        ("start_gateway_ancestry_checkpoint_bootstrap", '"${checkpoint_command[@]}"'),
    ):
        function_start = restart.index(f"{function_name}() {{")
        command_start = restart.index("env -u", function_start)
        position = restart.index(marker, command_start)
        command = restart[command_start:position + 300]
        assert close_set in command
        assert "-u GATEWAY_MINER_MAINTENANCE_PROOF_FD" in command
        assert "-u GATEWAY_GIT_HELPER" in command
        assert "-u GATEWAY_EXACT_COMMIT_HELPER" in command
        assert "-u GATEWAY_HOST_MEMORY_GUARD_PATH" in command


def test_hydrated_and_live_env_clones_reserve_invocation_only_keys():
    restart = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    assert restart.count('"GATEWAY_MINER_MAINTENANCE_PROOF_FD",') >= 3
    assert restart.count('"GATEWAY_EXACT_COMMIT_HELPER",') >= 3
    assert restart.count('"GATEWAY_HOST_MEMORY_GUARD_PATH",') >= 3


@pytest.mark.skipif(
    sys.platform != "linux",
    reason="production installer uses GNU stat and Linux mv -T",
)
@pytest.mark.parametrize("crash_point", ["release", "current", "host"])
def test_controller_install_recovers_every_publication_crash_point(
    tmp_path: Path,
    crash_point: str,
):
    repository = tmp_path / "candidate"
    controller_root = tmp_path / "restart-controller" / "gateway"
    releases_root = controller_root / "releases"
    candidate_release = releases_root / CANDIDATE_COMMIT
    previous_release = releases_root / CONTROLLER_COMMIT
    host_restart = tmp_path / "gw_restart.sh"
    candidate_payloads = {
        "gw_restart.sh": b"#!/bin/bash\necho candidate\n",
        "scripts/gateway_git_deploy.py": b"CANDIDATE_HELPER = True\n",
        "Leadpoet/utils/exact_commit_restart_v2.py": b"CANDIDATE_EXACT = True\n",
        "gateway/tee/host_memory_guard_v2.py": b"CANDIDATE_GUARD = True\n",
    }
    for relative_path, payload in candidate_payloads.items():
        source = repository / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(payload)
        source.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
        installed = candidate_release / relative_path
        installed.parent.mkdir(parents=True, exist_ok=True)
        installed.write_bytes(payload)
        installed.chmod(0o700 if relative_path == "gw_restart.sh" else 0o600)
    candidate_release.chmod(0o700)
    previous_release.mkdir(parents=True)
    previous_wrapper = previous_release / "gw_restart.sh"
    previous_wrapper.write_bytes(b"#!/bin/bash\necho previous\n")
    previous_wrapper.chmod(0o700)
    previous_release.chmod(0o700)
    current = controller_root / "current"
    current.symlink_to(
        f"releases/{CANDIDATE_COMMIT if crash_point != 'release' else CONTROLLER_COMMIT}"
    )
    host_restart.write_bytes(
        candidate_payloads["gw_restart.sh"]
        if crash_point == "host"
        else previous_wrapper.read_bytes()
    )
    host_restart.chmod(0o700)

    restart_source = (
        Path(__file__).resolve().parents[1] / "gw_restart.sh"
    ).read_text(encoding="utf-8")
    body = restart_source.split(
        "install_successful_restart_script() {\n",
        1,
    )[1].split("\n}\n\ninstall_research_lab_admin_wrapper()", 1)[0]
    script = (
        "set -euo pipefail\n"
        "install_successful_restart_script() {\n"
        + body
        + "\n}\ninstall_successful_restart_script\n"
    )
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GATEWAY_DEPLOY_SHA": CANDIDATE_COMMIT,
            "GATEWAY_RESTART_CONTROLLER_ROOT": str(controller_root),
            "GATEWAY_RESTART_CONTROLLER_CURRENT": str(current),
            "GATEWAY_HOST_RESTART_SCRIPT": str(host_restart),
            "LEADPOET_REPO_ROOT": str(repository),
        },
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert os.readlink(current) == f"releases/{CANDIDATE_COMMIT}"
    assert host_restart.read_bytes() == candidate_payloads["gw_restart.sh"]
    assert stat.S_IMODE(host_restart.stat().st_mode) == 0o700
    for relative_path, payload in candidate_payloads.items():
        installed = candidate_release / relative_path
        assert installed.read_bytes() == payload
        assert stat.S_IMODE(installed.stat().st_mode) == (
            0o700 if relative_path == "gw_restart.sh" else 0o600
        )


def test_candidate_restart_rechecks_guard_immediately_before_first_shutdown_action():
    restart = (Path(__file__).resolve().parents[1] / "gw_restart.sh").read_text(
        encoding="utf-8"
    )
    long_wait = restart.index("validator_tee.host.docker_operation_guard_v2")
    handoff = restart.index("wait_for_paired_gateway_destructive_handoff", long_wait)
    lineage = restart.index("prepare_gateway_active_release_lineage", handoff)
    boundary = restart.index(
        'GATEWAY_DEPLOY_STAGE="source_add_shutdown_quiescence"', lineage
    )
    boundary_rpc = restart.index("--verify-shutdown-quiescence", boundary)
    shutdown = restart.index(
        'echo "Stopping existing gateway and Research Lab worker processes"',
        boundary_rpc,
    )
    first_action = restart.index(
        "sudo systemctl stop leadpoet-tee-egress-forwarder.service",
        shutdown,
    )
    preflight_cleanup = restart.index(
        'rm -rf "$GATEWAY_PREFLIGHT_TREE"', first_action
    )

    assert long_wait < handoff < lineage < boundary < boundary_rpc < shutdown
    boundary_environment = restart[boundary:boundary_rpc]
    assert 'set -a\n    . "$ENV_CLONE"\n    set +a' in boundary_environment
    assert shutdown < first_action < preflight_cleanup
    between_boundary_and_shutdown = restart[boundary_rpc:shutdown]
    assert "wait_for_" not in between_boundary_and_shutdown
    assert "prepare_" not in between_boundary_and_shutdown
    assert 'rm -rf "$GATEWAY_PREFLIGHT_TREE"' not in between_boundary_and_shutdown

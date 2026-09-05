from __future__ import annotations

import ast
import hashlib
import io
import os
from pathlib import Path
import re
import subprocess
from typing import Optional

import pytest

from Leadpoet.utils.subnet_epoch import read_subnet_epoch_snapshot
from tests.restart_rehearsal.verify_evidence import (
    _verify_production_identity,
)
from tests.restart_rehearsal import sitecustomize as rehearsal_boundary


SOURCE_PATH = Path("scripts/gateway_git_deploy.py")
ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit(repo: Path, content: str, message: str) -> str:
    source = repo / SOURCE_PATH
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(content, encoding="utf-8")
    _git("add", SOURCE_PATH.as_posix(), cwd=repo)
    _git("commit", "-qm", message, cwd=repo)
    return _git("rev-parse", "HEAD", cwd=repo)


@pytest.fixture
def transition_repo(tmp_path: Path) -> tuple[Path, str, str, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git("init", "-q", cwd=repo)
    _git("config", "user.email", "rehearsal@example.invalid", cwd=repo)
    _git("config", "user.name", "Restart Rehearsal", cwd=repo)
    installed_content = "VERSION = 'installed'\n"
    installed_sha = _commit(repo, installed_content, "installed")
    candidate_sha = _commit(repo, "VERSION = 'candidate'\n", "candidate")
    return repo, installed_sha, candidate_sha, installed_content


def test_installed_source_identity_survives_expected_checkout_activation(
    transition_repo: tuple[Path, str, str, str],
) -> None:
    repo, installed_sha, candidate_sha, installed_content = transition_repo
    source = repo / SOURCE_PATH
    row = {
        "candidate_sha": candidate_sha,
        "source_commit": installed_sha,
        "source_git_path": SOURCE_PATH.as_posix(),
        "source_kind": "installed_checkout",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(
            installed_content.encode("utf-8")
        ).hexdigest(),
    }

    _verify_production_identity(
        row,
        installed_sha,
        candidate_sha,
        (repo,),
    )


def test_candidate_checkout_tampering_is_still_rejected(
    transition_repo: tuple[Path, str, str, str],
) -> None:
    repo, installed_sha, candidate_sha, _ = transition_repo
    source = repo / SOURCE_PATH
    candidate_blob = subprocess.run(
        ["git", "show", f"{candidate_sha}:{SOURCE_PATH.as_posix()}"],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout
    row = {
        "candidate_sha": candidate_sha,
        "source_commit": candidate_sha,
        "source_git_path": SOURCE_PATH.as_posix(),
        "source_kind": "candidate_checkout",
        "source_path": str(source),
        "source_sha256": hashlib.sha256(candidate_blob).hexdigest(),
    }
    source.write_text("TAMPERED = True\n", encoding="utf-8")

    with pytest.raises(
        SystemExit,
        match="candidate production source bytes changed",
    ):
        _verify_production_identity(
            row,
            installed_sha,
            candidate_sha,
            (repo,),
        )


def test_gateway_enclave_uses_the_exact_transition_target_tree() -> None:
    launcher = (
        ROOT / "tests/restart_rehearsal/run_inside.sh"
    ).read_text(encoding="utf-8")
    service = (
        ROOT / "tests/restart_rehearsal/gateway_enclave_service.py"
    ).read_text(encoding="utf-8")

    assert (
        'git --git-dir=/srv/origin.git archive "$CANDIDATE_SHA" gateway'
        in launcher
    )
    assert (
        'REHEARSAL_GATEWAY_CANDIDATE_ROOT='
        '"$SELECTED_GATEWAY_SOURCE_ROOT/gateway"'
        in launcher
    )
    assert 'REHEARSAL_GATEWAY_CANDIDATE_ROOT="/source/gateway"' not in launcher
    assert 'controller_source = Path("/source").resolve()' in service
    assert '"/source/gateway"' not in service


def test_gateway_provider_adapter_tracks_production_transport_interface() -> None:
    production = ast.parse(
        (ROOT / "gateway/tee/provider_broker_v2.py").read_text(encoding="utf-8")
    )
    rehearsal = ast.parse(
        (ROOT / "tests/restart_rehearsal/sitecustomize.py").read_text(
            encoding="utf-8"
        )
    )

    production_call = next(
        item
        for node in production.body
        if isinstance(node, ast.ClassDef) and node.name == "HTTPXProviderTransport"
        for item in node.body
        if isinstance(item, ast.FunctionDef) and item.name == "__call__"
    )
    rehearsal_call = next(
        node
        for node in rehearsal.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_local_provider_transport"
    )

    production_keywords = {arg.arg for arg in production_call.args.kwonlyargs}
    rehearsal_keywords = {arg.arg for arg in rehearsal_call.args.kwonlyargs}
    assert production_keywords <= rehearsal_keywords


def test_local_chain_adapter_supports_exact_historical_epoch_search(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(rehearsal_boundary, "STATE_ROOT", tmp_path)
    monkeypatch.setattr(
        rehearsal_boundary,
        "EVENT_PATH",
        tmp_path / "events.jsonl",
    )
    substrate = rehearsal_boundary._LocalSubstrate()
    target_epoch = rehearsal_boundary.CUTOVER_EPOCH_INDEX + 73
    boundary_block = rehearsal_boundary._subnet_epoch_transition_block(
        target_epoch
    )

    for block, expected_epoch in (
        (
            rehearsal_boundary.CUTOVER_BLOCK - 1,
            rehearsal_boundary.CUTOVER_EPOCH_INDEX - 1,
        ),
        (boundary_block - 1, target_epoch - 1),
        (boundary_block, target_epoch),
        (boundary_block + 1, target_epoch),
        (
            rehearsal_boundary.CURRENT_BLOCK,
            rehearsal_boundary.SUBNET_EPOCH_INDEX,
        ),
    ):
        block_hash = rehearsal_boundary._block_hash(block)
        assert substrate.get_block_number(block_hash) == block
        observed = {
            name: substrate.query(
                module="SubtensorModule",
                storage_function=name,
                params=[71],
                block_hash=block_hash,
            ).value
            for name in (
                "Tempo",
                "LastEpochBlock",
                "PendingEpochAt",
                "SubnetEpochIndex",
                "BlocksSinceLastStep",
            )
        }
        assert observed == rehearsal_boundary._subnet_epoch_state_at(block)
        assert observed["SubnetEpochIndex"] == expected_epoch
        assert observed["LastEpochBlock"] <= block
        assert observed["PendingEpochAt"] > block
        snapshot = read_subnet_epoch_snapshot(
            rehearsal_boundary._LocalSubtensor(network="finney"),
            netuid=71,
            block_number=block,
        )
        assert snapshot.current_block == block
        assert snapshot.subnet_epoch_index == expected_epoch
        assert snapshot.tempo > 0

    captured_current_hash = rehearsal_boundary._block_hash(
        rehearsal_boundary.CURRENT_BLOCK
    )
    rehearsal_boundary._BLOCK_NUMBERS_BY_HASH.clear()
    assert substrate.get_block_number(captured_current_hash) == (
        rehearsal_boundary.CURRENT_BLOCK
    )


def test_release_reuses_candidate_migrated_durable_boundary_state() -> None:
    controller = (
        ROOT / "scripts/run_local_restart_rehearsal.py"
    ).read_text(encoding="utf-8")
    launcher = (
        ROOT / "tests/restart_rehearsal/run_inside.sh"
    ).read_text(encoding="utf-8")

    assert 'dst=/rehearsal-durable-state"' in controller
    assert 'dst=/rehearsal-from-fixture-seed,readonly"' in controller
    assert "durable_state_root=durable_state_root" in controller
    assert re.search(
        r"from_fixture_seed_root=fixture_seeds\[\s*run_from\s*\]",
        controller,
    )
    assert re.search(
        r"durable_fixture_seed_root=fixture_seeds\[\s*candidate_sha\s*\]",
        controller,
    )
    assert 'REHEARSAL_DURABLE_SCHEMA_SHA:-' in launcher
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" in launcher
    assert (
        '"$REHEARSAL_DURABLE_STATE_ROOT/postgrest-state.json"'
        in launcher
    )
    assert (
        '"$DURABLE_SCHEMA_SEED_ROOT/release-build-input.json"'
        in launcher
    )


def test_exact_launcher_evaluates_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "invalid",
            "REHEARSAL_DURABLE_SCHEMA_SHA": "b" * 40,
        },
    )

    assert result.returncode == 2
    assert (
        "REHEARSAL_COMPONENT must be gateway, validator, or workflow"
        in result.stderr
    )


def test_workflow_does_not_require_launcher_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "workflow",
            "REHEARSAL_PROFILE": "invalid",
        },
    )

    assert result.returncode == 2
    assert "REHEARSAL_PROFILE must be prepush or release" in result.stderr
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" not in result.stderr


def test_exact_launcher_requires_durable_schema_identity() -> None:
    launcher = ROOT / "tests/restart_rehearsal/run_inside.sh"
    result = subprocess.run(
        ["bash", str(launcher)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "REHEARSAL_FROM_SHA": "a" * 40,
            "REHEARSAL_CANDIDATE_SHA": "b" * 40,
            "REHEARSAL_TRANSITION": "forward",
            "REHEARSAL_COMPONENT": "gateway",
        },
    )

    assert result.returncode == 2
    assert "REHEARSAL_DURABLE_SCHEMA_SHA is required" in result.stderr

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess

import pytest

from scripts import run_local_restart_rehearsal as rehearsal
from tests.restart_rehearsal import contract_adapter
from tests.restart_rehearsal.verify_evidence import (
    EXPECTED_GATEWAY_PRIVATE_MODEL_ENV,
    verify_gateway_private_model_environment,
    verify_rehearsal_integrity,
)


COMMIT = "1" * 40


def _exact_capacity_rows() -> list[dict]:
    return [
        {
            "kind": "host-command",
            "substitution": "host.cpu_capacity",
            "implementation": "internal_substitution",
            "advertised_vcpus": 16,
            "outer_limit": "4",
        },
        {
            "kind": "host-command",
            "substitution": "host.memory_capacity",
            "implementation": "internal_substitution",
            "advertised_memory_mib": 131072,
            "outer_limit": "6g",
        },
    ]


def test_gateway_rehearsal_requires_canonical_private_model_environment() -> None:
    row = {
        "kind": "process",
        "process": "gateway.main",
        "status": "started",
        "environment_contract": dict(EXPECTED_GATEWAY_PRIVATE_MODEL_ENV),
    }
    verify_gateway_private_model_environment([row])

    row["environment_contract"][
        "RESEARCH_LAB_PRIVATE_REPO_BRANCH"
    ] = "main"
    with pytest.raises(SystemExit, match="private-model source environment"):
        verify_gateway_private_model_environment([row])

    with pytest.raises(SystemExit, match="exactly one gateway.main"):
        verify_gateway_private_model_environment([])


def test_rehearsal_driver_must_match_frozen_harness_commit(
    monkeypatch,
    tmp_path,
) -> None:
    driver = tmp_path / "run_local_restart_rehearsal.py"
    driver.write_bytes(b"candidate driver\n")
    monkeypatch.setattr(rehearsal, "__file__", str(driver))
    monkeypatch.setattr(
        rehearsal,
        "_git_file",
        lambda _sha, _path: b"candidate driver\n",
    )

    rehearsal._verify_driver_identity(COMMIT)

    driver.write_bytes(b"dirty driver\n")
    with pytest.raises(SystemExit, match="differs from the frozen harness"):
        rehearsal._verify_driver_identity(COMMIT)


def test_rehearsal_source_snapshot_is_independent_and_complete(
    monkeypatch,
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "Restart Test"],
        check=True,
    )
    source_file = repo / "source.txt"
    source_file.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "frozen source"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(rehearsal, "REPO_ROOT", repo)

    with rehearsal._isolated_source_snapshot(
        harness_sha=commit,
        required_shas=(commit,),
    ) as snapshot:
        assert snapshot != repo
        assert (snapshot / "source.txt").read_text(encoding="utf-8") == "frozen\n"
        assert subprocess.run(
            ["git", "-C", str(snapshot), "cat-file", "-e", f"{commit}^{{commit}}"],
            check=False,
        ).returncode == 0
        assert not (snapshot / ".git" / "objects" / "info" / "alternates").exists()

    assert not snapshot.exists()


def test_rehearsal_resolves_forward_and_rollback_transitions(monkeypatch) -> None:
    relationships = {
        ("from", "target"): True,
        ("target", "from"): False,
        ("newer", "older"): False,
        ("older", "newer"): True,
    }
    monkeypatch.setattr(
        rehearsal,
        "_is_ancestor",
        lambda ancestor, descendant: relationships.get(
            (ancestor, descendant),
            False,
        ),
    )

    assert rehearsal._resolve_transition("from", "target", "auto") == "forward"
    assert rehearsal._resolve_transition("newer", "older", "auto") == "rollback"
    assert (
        rehearsal._resolve_transition("newer", "older", "rollback")
        == "rollback"
    )
    with pytest.raises(SystemExit, match="does not descend"):
        rehearsal._resolve_transition("newer", "older", "forward")


def test_rehearsal_rejects_unrelated_transition(monkeypatch) -> None:
    monkeypatch.setattr(rehearsal, "_is_ancestor", lambda *_args: False)

    with pytest.raises(SystemExit, match="unrelated"):
        rehearsal._resolve_transition("from", "target", "auto")


def test_rollback_rehearsal_keeps_newer_commit_on_origin_main() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")
    rollback_refs = script.index(
        'if [ "$TRANSITION" = "rollback" ]; then'
    )
    component_setup = script.index(
        'if [ "$COMPONENT" = "gateway" ]; then'
    )
    section = script[rollback_refs:component_setup]
    rollback_section, forward_section = section.split("\nelse\n", 1)

    assert '"$FROM_SHA:refs/heads/main"' in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/rehearsal-target"' in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/main"' not in rollback_section
    assert '"$CANDIDATE_SHA:refs/heads/main"' in forward_section


def test_forward_rehearsal_uses_the_normal_unpinned_operator_paths() -> None:
    script = (
        Path(__file__).resolve().parent / "run_inside.sh"
    ).read_text(encoding="utf-8")

    assert 'GATEWAY_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert 'VALIDATOR_DEPLOY_COMMIT="$CANDIDATE_SHA"' not in script
    assert script.count('--commit "$CANDIDATE_SHA"') == 2


def test_exact_rehearsal_rejects_repository_module_substitution() -> None:
    rows = [
        {
            "kind": "python-module",
            "module": "gateway.tee.restart_preflight_v2",
            "implementation": "internal_substitution",
        }
    ]

    with pytest.raises(SystemExit, match="repository-code substitutions"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_exact_rehearsal_accepts_strict_capacity_substitutions() -> None:
    verify_rehearsal_integrity(
        _exact_capacity_rows(),
        candidate_sha=COMMIT,
        scope="exact",
    )


@pytest.mark.parametrize(
    "missing_identity",
    ("host.cpu_capacity", "host.memory_capacity"),
)
def test_exact_rehearsal_requires_both_capacity_substitutions(
    missing_identity,
) -> None:
    rows = _exact_capacity_rows()
    rows = [
        row for row in rows if row["substitution"] != missing_identity
    ]

    with pytest.raises(
        SystemExit,
        match=f"missing required capacity contracts: {missing_identity}",
    ):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_exact_rehearsal_rejects_incomplete_capacity_evidence() -> None:
    rows = [
        {
            "kind": "host-command",
            "substitution": "host.cpu_capacity",
            "implementation": "internal_substitution",
            "advertised_vcpus": 16,
            "outer_limit": "4",
        },
        {
            "kind": "host-command",
            "substitution": "host.memory_capacity",
            "implementation": "internal_substitution",
            "advertised_memory_mib": 131072,
            "outer_limit": "",
        }
    ]

    with pytest.raises(SystemExit, match="capacity contract is invalid"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_rehearsal_component_uses_constrained_outer_profile(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        rehearsal,
        "_run",
        lambda args, **_kwargs: calls.append(args),
    )

    rehearsal._run_component(
        "rehearsal:test",
        component="gateway",
        from_sha="1" * 40,
        candidate_sha="2" * 40,
        scope="exact",
        outer_cpus="3.5",
        outer_memory="7g",
    )

    command = calls[0]
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert command[command.index("--cpus") + 1] == "3.5"
    assert command[command.index("--memory") + 1] == "7g"
    assert "REHEARSAL_OUTER_CPUS=3.5" in command
    assert "REHEARSAL_OUTER_MEMORY=7g" in command


def test_exact_adapter_allows_only_capacity_substitutions(monkeypatch) -> None:
    monkeypatch.setenv("REHEARSAL_SCOPE", "exact")
    monkeypatch.setattr(
        contract_adapter,
        "_event",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        contract_adapter,
        "_fail",
        lambda *_args, **_kwargs: 97,
    )

    assert (
        contract_adapter._record_internal_substitution(
            kind="host-command",
            argv=["getconf", "_NPROCESSORS_CONF"],
            substitution="host.cpu_capacity",
            advertised_vcpus=16,
            outer_limit="4",
        )
        == 0
    )
    assert (
        contract_adapter._record_internal_substitution(
            kind="python-module",
            argv=["-m", "gateway.tee.restart_preflight_v2"],
            module="gateway.tee.restart_preflight_v2",
        )
        == 97
    )


def test_targeted_regression_is_distinct_and_rejects_unknown_substitution() -> None:
    known = [
        {
            "kind": "python-module",
            "module": "gateway.tee.restart_preflight_v2",
            "implementation": "internal_substitution",
        }
    ]
    verify_rehearsal_integrity(
        known,
        candidate_sha=COMMIT,
        scope="weight_readiness_regression",
    )

    unknown = [
        {
            "kind": "python-module",
            "module": "gateway.future_unexercised_stage",
            "implementation": "internal_substitution",
        }
    ]
    with pytest.raises(SystemExit, match="unclassified"):
        verify_rehearsal_integrity(
            unknown,
            candidate_sha=COMMIT,
            scope="weight_readiness_regression",
        )


def test_targeted_regression_classifies_dependency_bootstrap() -> None:
    rows = [
        {
            "kind": "python-script",
            "script": "get-pip.py",
            "substitution": "python_dependencies.bootstrap",
            "implementation": "internal_substitution",
        }
    ]

    verify_rehearsal_integrity(
        rows,
        candidate_sha=COMMIT,
        scope="weight_readiness_regression",
    )

    with pytest.raises(SystemExit, match="repository-code substitutions"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_exact_rehearsal_rejects_synthetic_external_fixture() -> None:
    rows = [
        *_exact_capacity_rows(),
        {
            "kind": "aws",
            "operation": "secretsmanager",
            "fixture_authenticity": "synthetic",
        }
    ]

    with pytest.raises(SystemExit, match="synthetic external fixtures"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_production_stage_requires_exact_candidate_source_identity(tmp_path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "module.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "test source"],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row = {
        "kind": "python-module",
        "module": "gateway.real_stage",
        "implementation": "production_module",
        "candidate_sha": commit,
        "source_commit": commit,
        "source_path": str(source),
        "source_git_path": "module.py",
        "source_kind": "candidate_checkout",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    rows = [*_exact_capacity_rows(), row]
    verify_rehearsal_integrity(
        rows,
        candidate_sha=commit,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    row["source_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="Git identity"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )

    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after execution"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )


def test_rollback_accepts_installed_launcher_source_bound_to_from_sha(
    tmp_path,
) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Restart Test"],
        check=True,
    )
    source = tmp_path / "compatibility.py"
    source.write_text("VALUE = 'installed'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "compatibility.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "rollback target"],
        check=True,
    )
    candidate = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source.write_text("VALUE = 'installed launcher'\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "compatibility.py"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-qm", "installed launcher"],
        check=True,
    )
    installed = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    row = {
        "kind": "python-script",
        "script": "compatibility.py",
        "implementation": "production_script",
        "candidate_sha": candidate,
        "source_commit": installed,
        "source_path": str(source),
        "source_git_path": "compatibility.py",
        "source_kind": "installed_checkout",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }

    verify_rehearsal_integrity(
        [row],
        from_sha=installed,
        candidate_sha=candidate,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    with pytest.raises(SystemExit, match="source identity is invalid"):
        verify_rehearsal_integrity(
            [row],
            from_sha="2" * 40,
            candidate_sha=candidate,
            scope="exact",
            candidate_roots=(tmp_path,),
        )

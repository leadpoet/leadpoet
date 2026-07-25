from __future__ import annotations

import hashlib
import subprocess

import pytest

from scripts import run_local_restart_rehearsal as rehearsal
from tests.restart_rehearsal.verify_evidence import (
    EXPECTED_GATEWAY_PRIVATE_MODEL_ENV,
    verify_gateway_private_model_environment,
    verify_rehearsal_integrity,
)


COMMIT = "1" * 40


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
        "source_path": str(source),
        "source_git_path": "module.py",
        "source_kind": "candidate_checkout",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    verify_rehearsal_integrity(
        [row],
        candidate_sha=commit,
        scope="exact",
        candidate_roots=(tmp_path,),
    )

    row["source_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="Git identity"):
        verify_rehearsal_integrity(
            [row],
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )

    row["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="changed after execution"):
        verify_rehearsal_integrity(
            [row],
            candidate_sha=commit,
            scope="exact",
            candidate_roots=(tmp_path,),
        )

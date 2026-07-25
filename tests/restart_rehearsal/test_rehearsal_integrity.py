from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import run_local_restart_rehearsal as rehearsal
from tests.restart_rehearsal import contract_adapter
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


def test_contract_adapter_imports_with_python_safe_path(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(Path(contract_adapter.__file__).resolve()),
            "future-command",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONSAFEPATH": "1",
            "PYTHONPATH": str(Path.cwd()),
            "REHEARSAL_STATE_ROOT": str(tmp_path / "state"),
        },
    )

    assert result.returncode == 97
    assert "unknown adapter command" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_rehearsal_driver_must_match_frozen_candidate(
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
    with pytest.raises(SystemExit, match="differs from the frozen candidate"):
        rehearsal._verify_driver_identity(COMMIT)


def test_exact_rehearsal_allows_only_classified_contract_adapters() -> None:
    rows = [
        {
            "kind": "python-module",
            "module": "gateway.tee.restart_preflight_v2",
            "implementation": "internal_substitution",
        }
    ]

    verify_rehearsal_integrity(
        rows,
        candidate_sha=COMMIT,
        scope="exact",
    )

    rows[0]["module"] = "gateway.future_unexercised_stage"
    with pytest.raises(SystemExit, match="repository-code substitutions"):
        verify_rehearsal_integrity(
            rows,
            candidate_sha=COMMIT,
            scope="exact",
        )


@pytest.mark.parametrize(
    ("identity", "evidence"),
    (
        (
            "host.cpu_capacity",
            {"advertised_vcpus": 16, "outer_limit": "4"},
        ),
        (
            "host.memory_capacity",
            {"advertised_memory_mib": 131072, "outer_limit": "6g"},
        ),
    ),
)
def test_exact_rehearsal_accepts_strict_capacity_substitutions(
    identity,
    evidence,
) -> None:
    verify_rehearsal_integrity(
        [
            {
                "kind": "host-command",
                "substitution": identity,
                "implementation": "internal_substitution",
                **evidence,
            }
        ],
        candidate_sha=COMMIT,
        scope="exact",
    )


def test_exact_rehearsal_rejects_incomplete_capacity_evidence() -> None:
    rows = [
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


def test_exact_adapter_allows_only_classified_substitutions(monkeypatch) -> None:
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
        == 0
    )
    assert (
        contract_adapter._record_internal_substitution(
            kind="python-module",
            argv=["-m", "gateway.future_unexercised_stage"],
            module="gateway.future_unexercised_stage",
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


def test_exact_rehearsal_accepts_classified_external_contract_fixture() -> None:
    verify_rehearsal_integrity(
        [
            {
                "kind": "docker",
                "argv": ["run", "--cpus", "16", "--memory", "128g"],
                "fixture_authenticity": "contract_enforced",
            }
        ],
        candidate_sha=COMMIT,
        scope="exact",
    )

    with pytest.raises(SystemExit, match="invalid external contract adapter"):
        verify_rehearsal_integrity(
            [
                {
                    "kind": "future-service",
                    "argv": ["run"],
                    "fixture_authenticity": "contract_enforced",
                }
            ],
            candidate_sha=COMMIT,
            scope="exact",
        )


def test_exact_rehearsal_accepts_classified_weight_contract_fixture() -> None:
    verify_rehearsal_integrity(
        [
            {
                "kind": "weight-readiness-boundary",
                "boundary": "direct_allocation",
                "fixture_authenticity": "contract_enforced",
            },
            {
                "kind": "weight-readiness-persistence",
                "attempts": [
                    {
                        "method": "GET",
                        "attempt_number": 1,
                    }
                ],
                "fixture_authenticity": "contract_enforced",
            },
        ],
        candidate_sha=COMMIT,
        scope="exact",
    )

    with pytest.raises(SystemExit, match="invalid external contract adapter"):
        verify_rehearsal_integrity(
            [
                {
                    "kind": "weight-readiness-boundary",
                    "boundary": "future_boundary",
                    "fixture_authenticity": "contract_enforced",
                }
            ],
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

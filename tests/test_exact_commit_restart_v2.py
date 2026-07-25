from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from Leadpoet.utils.exact_commit_restart_v2 import (
    AUDITOR_PROTOCOL_CONTRACT,
    ExactCommitRestartCompatibilityError,
    verify_exact_commit_restart_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]
KNOWN_ATTESTED_PROTOCOL_COMPATIBLE_RELEASE = (
    "d0ee33f6e9c2ec22f6c2cbe82eedc25a4bae11b4"
)


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_protocol_contract(
    repo: Path,
    *,
    omitted_selected_capability: str = "",
    omitted_auditor_capability: str = "",
) -> None:
    files: dict[str, list[str]] = {}
    for name, _value, selected_markers, auditor_markers in (
        AUDITOR_PROTOCOL_CONTRACT
    ):
        if name != omitted_selected_capability:
            for path, marker in selected_markers:
                files.setdefault(path, []).append(marker)
        if (
            name != omitted_selected_capability
            and name != omitted_auditor_capability
        ):
            for path, marker in auditor_markers:
                files.setdefault(path, []).append(marker)
    files.setdefault(
        "leadpoet_canonical/weight_computation.py",
        [],
    ).append(
        "def compute_final_weights(snapshot):\n"
        "    return snapshot\n"
    )
    files.setdefault("leadpoet_canonical/weights.py", []).append(
        "def bundle_weights_hash(netuid, epoch_id, block, uid_weights):\n"
        "    return str((netuid, epoch_id, block, uid_weights))\n"
    )
    for relative, markers in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(set(markers))) + "\n", encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", message)
    return _run(repo, "git", "rev-parse", "HEAD")


def _repo(
    tmp_path: Path,
    *,
    omitted_selected_capability: str = "",
) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-q")
    _run(repo, "git", "config", "user.email", "test@leadpoet.invalid")
    _run(repo, "git", "config", "user.name", "Leadpoet Test")
    _write_protocol_contract(
        repo,
        omitted_selected_capability=omitted_selected_capability,
    )
    selected = _commit(repo, "selected release")
    return repo, selected


def test_known_attested_historical_release_matches_current_auditor_protocol() -> None:
    report = verify_exact_commit_restart_compatibility(
        repo_root=ROOT,
        selected_commit=KNOWN_ATTESTED_PROTOCOL_COMPATIBLE_RELEASE,
        branch_ref="HEAD",
    )

    assert report["status"] == "compatible"
    assert report["compatibility_scope"] == "auditor_weight_protocol"
    assert report["selected_commit"] == KNOWN_ATTESTED_PROTOCOL_COMPATIBLE_RELEASE
    assert report["implementation_history_compared"] is False
    assert report["auditor_protocol_entry_count"] == len(
        AUDITOR_PROTOCOL_CONTRACT
    )


def test_rollback_accepts_later_reliability_and_implementation_changes(
    tmp_path: Path,
) -> None:
    repo, selected = _repo(tmp_path)
    protected = repo / "gateway/research_lab/attested_v2_store.py"
    protected.write_text(
        protected.read_text(encoding="utf-8")
        + "\n# Later pagination and retry reliability implementation.\n",
        encoding="utf-8",
    )
    computation = repo / "leadpoet_canonical/weight_computation.py"
    computation.write_text(
        computation.read_text(encoding="utf-8")
        + "\ndef unrelated_retry_helper():\n"
        "    return 'newer reliability behavior'\n",
        encoding="utf-8",
    )
    manifest = repo / "gateway/tee/protected_workflows.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"newer_reliability_metadata":true}\n', encoding="utf-8")
    branch = _commit(repo, "newer reliability fixes")

    report = verify_exact_commit_restart_compatibility(
        repo_root=repo,
        selected_commit=selected,
        branch_ref=branch,
    )

    assert report["status"] == "compatible"
    assert report["selected_commit"] == selected
    assert report["branch_commit"] == branch


def test_rollback_rejects_a_different_canonical_weight_algorithm(
    tmp_path: Path,
) -> None:
    repo, selected = _repo(tmp_path)
    computation = repo / "leadpoet_canonical/weight_computation.py"
    computation.write_text(
        computation.read_text(encoding="utf-8").replace(
            "    return snapshot\n",
            "    return dict(snapshot)\n",
            1,
        ),
        encoding="utf-8",
    )
    branch = _commit(repo, "change canonical weight algorithm")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="canonical weight algorithm differs from current auditors",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
        )


@pytest.mark.parametrize(
    "capability",
    [
        "published_authority_endpoint",
        "release_evidence_schema",
        "weight_bundle_schema",
        "weight_publication_schema",
        "finalized_authority_schema",
    ],
)
def test_rollback_rejects_missing_selected_auditor_protocol_capability(
    tmp_path: Path,
    capability: str,
) -> None:
    repo, selected = _repo(
        tmp_path,
        omitted_selected_capability=capability,
    )
    _write_protocol_contract(repo)
    branch = _commit(repo, "current compatible auditor protocol")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="selected release lacks auditor protocol capability %s"
        % capability,
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
        )


def test_rollback_fails_if_current_auditor_contract_metadata_is_stale(
    tmp_path: Path,
) -> None:
    repo, selected = _repo(tmp_path)
    _write_protocol_contract(
        repo,
        omitted_auditor_capability="finalized_authority_endpoint",
    )
    branch = _commit(repo, "break current auditor declaration")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match=(
            "current auditor protocol declaration is inconsistent "
            "for finalized_authority_endpoint"
        ),
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
        )


def test_protocol_checker_does_not_impose_history_or_floor_policy(
    tmp_path: Path,
) -> None:
    repo, branch = _repo(tmp_path)
    _run(repo, "git", "checkout", "--orphan", "attested-release")
    _run(repo, "git", "rm", "-rf", ".")
    _write_protocol_contract(repo)
    selected = _commit(repo, "independently reachable attested release")

    report = verify_exact_commit_restart_compatibility(
        repo_root=repo,
        selected_commit=selected,
        branch_ref=branch,
    )

    assert report["status"] == "compatible"
    assert report["selected_commit"] == selected


def test_rollback_requires_a_full_selected_commit_sha(tmp_path: Path) -> None:
    repo, selected = _repo(tmp_path)

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="lowercase full Git SHA",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected[:12],
            branch_ref=selected,
        )

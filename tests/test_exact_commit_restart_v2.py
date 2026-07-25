from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from Leadpoet.utils.exact_commit_restart_v2 import (
    ExactCommitRestartCompatibilityError,
    verify_exact_commit_restart_compatibility,
)


ROOT = Path(__file__).resolve().parents[1]
FLOOR = "94f1c923d092d12cbab95ef8d86317420eede621"


def _run(repo: Path, *args: str) -> str:
    return subprocess.run(
        list(args),
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _manifest(
    *,
    schema_version: str,
    path: str,
    symbol: str,
    source: str,
) -> dict:
    node = ast.parse(source).body[0]
    digest = "sha256:" + hashlib.sha256(
        ast.dump(
            node,
            annotate_fields=True,
            include_attributes=False,
        ).encode("utf-8")
    ).hexdigest()
    body = {
        "schema_version": schema_version,
        "baseline_commit": "0" * 40,
        "protected_source_commit": "0" * 40,
        "entries": [
            {
                "path": path,
                "symbol": symbol,
                "ast_sha256": digest,
            }
        ],
    }
    manifest_hash = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    return {**body, "manifest_hash": manifest_hash}


def _write_release_contract(
    repo: Path,
    *,
    validator_value: str = "a" * 64,
    gateway_value: str = "a" * 64,
) -> None:
    validator_source = (
        "def compute_final_weights():\n    return %r\n" % validator_value
    )
    gateway_source = (
        "def build_gateway_weight_inputs_v2():\n    return %r\n"
        % gateway_value
    )
    files = {
        "gateway/api/weights.py": (
            '@router.get("/v2/release-evidence/{commit_sha}")\n'
            '"leadpoet.auditor_release_evidence.v2"\n'
        ),
        "gateway/main.py": (
            '@app.get("/health/v2-authority")\n'
            '"leadpoet.gateway_v2_authority_health.v2"\n'
        ),
        "gw_restart.sh": "GATEWAY_RESTART_PHASE\ngateway.tee.release_channel_v2\n",
        "validator_restart.sh": (
            "validator_tee.host.restart_preflight_v2\n"
            "validator_tee.host.verify_release_gate_v2\n"
        ),
        "validator_models/containerizing/deploy_dynamic.sh": (
            "VALIDATOR_V2_DEPLOY_COMMIT\nauthoritative_v2\n"
        ),
        "leadpoet_canonical/weight_computation.py": validator_source,
        "gateway/research_lab/weight_inputs_v2.py": gateway_source,
        "gateway/tee/protected_workflows.json": json.dumps(
            _manifest(
                schema_version="leadpoet.protected_workflows.v2",
                path="gateway/research_lab/weight_inputs_v2.py",
                symbol="build_gateway_weight_inputs_v2",
                source=gateway_source,
            ),
            sort_keys=True,
        ),
        "validator_tee/enclave/protected_workflows_v2.json": json.dumps(
            _manifest(
                schema_version="leadpoet.validator_protected_workflows.v2",
                path="leadpoet_canonical/weight_computation.py",
                symbol="compute_final_weights",
                source=validator_source,
            ),
            sort_keys=True,
        ),
    }
    for relative, contents in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", message)
    return _run(repo, "git", "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-q")
    _run(repo, "git", "config", "user.email", "test@leadpoet.invalid")
    _run(repo, "git", "config", "user.name", "Leadpoet Test")
    _write_release_contract(repo)
    floor = _commit(repo, "floor")
    return repo, floor


def test_declared_floor_contains_the_required_public_release_contract() -> None:
    report = verify_exact_commit_restart_compatibility(
        repo_root=ROOT,
        selected_commit=FLOOR,
        branch_ref=FLOOR,
        compatibility_floor=FLOOR,
    )

    assert report["status"] == "compatible"
    assert report["selected_commit"] == FLOOR


def test_compatible_rollback_preserves_protected_workflow_contract(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    selected = floor
    (repo / "unrelated.txt").write_text("newer orchestration\n", encoding="utf-8")
    branch = _commit(repo, "newer orchestration")

    report = verify_exact_commit_restart_compatibility(
        repo_root=repo,
        selected_commit=selected,
        branch_ref=branch,
        compatibility_floor=floor,
    )

    assert report["status"] == "compatible"
    assert report["selected_commit"] == selected
    assert report["branch_commit"] == branch


def test_rollback_rejects_changed_protected_workflow_contract(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    selected = floor
    _write_release_contract(repo, validator_value="b" * 64)
    branch = _commit(repo, "change protected weight computation")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="changes the validator protected V2 workflow contract",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )


def test_rollback_rejects_changed_gateway_protected_workflow_contract(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    selected = floor
    _write_release_contract(repo, gateway_value="b" * 64)
    branch = _commit(repo, "change protected gateway weight input")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="gateway protected V2 workflow contract",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )


def test_rollback_rejects_unmanifested_change_in_protected_source_file(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    selected = floor
    source = repo / "leadpoet_canonical/weight_computation.py"
    source.write_text(
        source.read_text(encoding="utf-8")
        + "\nHELPER_CONSTANT = 'changed outside protected symbol'\n",
        encoding="utf-8",
    )
    branch = _commit(repo, "change protected source outside declared symbol")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="validator protected V2 source file",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )


def test_rollback_rejects_manifest_that_differs_from_selected_source(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    manifest_path = (
        repo / "validator_tee/enclave/protected_workflows_v2.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["ast_sha256"] = "sha256:" + "b" * 64
    body = {
        key: manifest[key]
        for key in (
            "schema_version",
            "baseline_commit",
            "protected_source_commit",
            "entries",
        )
    }
    manifest["manifest_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    selected = _commit(repo, "forge protected manifest")
    (repo / "later.txt").write_text("branch head\n", encoding="utf-8")
    branch = _commit(repo, "later branch head")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="manifest differs from source",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )


def test_rollback_rejects_release_without_public_evidence_endpoint(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    weights = repo / "gateway/api/weights.py"
    weights.write_text("endpoint removed\n", encoding="utf-8")
    selected = _commit(repo, "remove public release evidence")
    (repo / "later.txt").write_text("branch head\n", encoding="utf-8")
    branch = _commit(repo, "later branch head")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="lacks required V2 contract marker",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )


def test_rollback_rejects_commit_outside_configured_branch(
    tmp_path: Path,
) -> None:
    repo, floor = _repo(tmp_path)
    branch = floor
    _run(repo, "git", "checkout", "--orphan", "unrelated")
    _run(repo, "git", "rm", "-rf", ".")
    _write_release_contract(repo)
    selected = _commit(repo, "unrelated release")

    with pytest.raises(
        ExactCommitRestartCompatibilityError,
        match="not reachable from the configured branch",
    ):
        verify_exact_commit_restart_compatibility(
            repo_root=repo,
            selected_commit=selected,
            branch_ref=branch,
            compatibility_floor=floor,
        )

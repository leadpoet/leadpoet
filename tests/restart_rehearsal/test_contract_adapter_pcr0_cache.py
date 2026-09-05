from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

from tests.restart_rehearsal.artifact_identity import (
    VALIDATOR_ROLE,
    eif_bytes,
    normalized_config,
    normalized_image_id,
    pcr0,
)
from validator_tee.enclave.runtime_v2 import compute_app_manifest_hash
from validator_tee.host.docker_image_normalizer_v2 import normalize_saved_image


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = Path(__file__).with_name("contract_adapter.py")
PCR0_BUILDER_PATH = ROOT / "gateway" / "utils" / "pcr0_builder.py"


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["/usr/bin/git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _two_commit_repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Rehearsal")
    _git(root, "config", "user.email", "rehearsal@example.invalid")
    dockerfile = root / "validator_tee" / "Dockerfile.enclave"
    dockerfile.parent.mkdir()
    dockerfile.write_text("FROM scratch\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "parent")
    parent = _git(root, "rev-parse", "HEAD")
    dockerfile.write_text("FROM scratch\n# child\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "child")
    child = _git(root, "rev-parse", "HEAD")
    _git(root, "checkout", "-q", parent)
    return parent, child


def _production_cache_tag(timestamp: int) -> str:
    """Derive the tag from the candidate's exact timestamp-tag AST."""

    tree = ast.parse(PCR0_BUILDER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef)
        and node.name == "build_enclave_and_extract_pcr0"
    )
    assignment = next(
        node
        for node in function.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "docker_image"
            for target in node.targets
        )
    )
    assert isinstance(assignment.value, ast.JoinedStr)
    assert len(assignment.value.values) == 2
    prefix, timestamp_expression = assignment.value.values
    assert isinstance(prefix, ast.Constant)
    assert isinstance(prefix.value, str)
    assert isinstance(timestamp_expression, ast.FormattedValue)
    converted = timestamp_expression.value
    assert isinstance(converted, ast.Call)
    assert isinstance(converted.func, ast.Name)
    assert converted.func.id == "int"
    assert len(converted.args) == 1
    clock = converted.args[0]
    assert isinstance(clock, ast.Call)
    assert isinstance(clock.func, ast.Attribute)
    assert isinstance(clock.func.value, ast.Name)
    assert (clock.func.value.id, clock.func.attr) == ("time", "time")
    assert not clock.args
    assert not clock.keywords
    return prefix.value + str(timestamp)


def _load_adapter(
    *,
    monkeypatch: pytest.MonkeyPatch,
    state_root: Path,
    build_root: Path,
    candidate: str,
):
    monkeypatch.setenv("REHEARSAL_STATE_ROOT", str(state_root))
    monkeypatch.setenv("REHEARSAL_CANDIDATE_SHA", candidate)
    monkeypatch.setenv("PCR0_BUILD_DIR", str(build_root))
    specification = importlib.util.spec_from_file_location(
        "rehearsal_contract_adapter_pcr0_cache_test",
        ADAPTER_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    adapter = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(adapter)
    # The focused unit calls the boundary handlers directly. The full N-1
    # rehearsal separately validates the staged /harness allowlist.
    adapter._record_external_boundary = lambda **_kwargs: None
    return adapter


def _build_argv(tag: str) -> list[str]:
    return [
        "build",
        "--no-cache",
        "-f",
        "validator_tee/Dockerfile.enclave",
        "-t",
        tag,
        ".",
    ]


def test_pcr0_cache_git_identity_is_one_atomic_bounded_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "pcr0-builder"
    parent, candidate = _two_commit_repository(build_root)
    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=tmp_path / "state",
        build_root=build_root,
        candidate=candidate,
    )
    calls: list[tuple[list[str], dict]] = []
    real_run = adapter.subprocess.run

    def counted_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        return real_run(argv, **kwargs)

    monkeypatch.setattr(adapter.subprocess, "run", counted_run)
    assert adapter._pcr0_cache_git_identity(build_root.resolve()) == parent
    assert calls == [
        (
            [
                "/usr/bin/git",
                "-C",
                str(build_root.resolve()),
                "rev-parse",
                "--show-toplevel",
                "--verify",
                "HEAD^{commit}",
            ],
            {
                "check": True,
                "capture_output": True,
                "text": True,
                "timeout": 30,
            },
        )
    ]


def test_pcr0_cache_adapter_preserves_checked_out_n_minus_one_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    build_root = tmp_path / "pcr0-builder"
    parent, candidate = _two_commit_repository(build_root)
    state_root = tmp_path / "state"
    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=state_root,
        build_root=build_root,
        candidate=candidate,
    )
    monkeypatch.chdir(build_root)
    tag = _production_cache_tag(1_786_872_221)
    normalized_tag = tag + "-normalized:latest"
    raw_archive = tmp_path / "raw.tar"
    normalized_archive = tmp_path / "normalized.tar"
    eif_path = tmp_path / "validator.eif"

    assert adapter.command_docker(_build_argv(tag)) == 0
    assert adapter.command_docker(["save", tag, "-o", str(raw_archive)]) == 0
    assert normalize_saved_image(
        archive_path=raw_archive,
        output_path=normalized_archive,
        normalized_image=normalized_tag,
        temporary_parent=tmp_path,
    ) == normalized_image_id(parent, VALIDATOR_ROLE)
    assert adapter.command_docker(["load", "-i", str(normalized_archive)]) == 0
    assert adapter.command_docker(
        ["tag", normalized_image_id(parent, VALIDATOR_ROLE), normalized_tag]
    ) == 0

    capsys.readouterr()
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            normalized_tag,
            "--output-file",
            str(eif_path),
        ]
    ) == 0
    measurement = json.loads(capsys.readouterr().out)["Measurements"]["PCR0"]
    assert measurement == pcr0(parent)
    assert measurement != pcr0(candidate)
    assert eif_path.read_bytes() == eif_bytes(parent, VALIDATOR_ROLE)
    assert not (state_root / "validator-app").exists()

    state = json.loads((state_root / "state.json").read_text(encoding="utf-8"))
    record = state["images"][normalized_tag]
    normalized_layer, _ = normalized_config(parent, VALIDATOR_ROLE)
    assert record == {
        "build_root": str(build_root),
        "commit": parent,
        "id": normalized_image_id(parent, VALIDATOR_ROLE),
        "provenance": adapter.PCR0_CACHE_PROVENANCE,
        "role": VALIDATOR_ROLE,
        "rootfs_layers": ["sha256:" + hashlib.sha256(normalized_layer).hexdigest()],
        "source_tag": tag,
    }

    assert adapter.command_docker(["tag", normalized_tag, "copied:latest"]) == 0
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            "copied:latest",
            "--output-file",
            str(tmp_path / "copied.eif"),
        ]
    ) == 97


def test_candidate_commit_cache_cannot_fall_back_to_release_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "pcr0-builder"
    _parent, candidate = _two_commit_repository(build_root)
    _git(build_root, "checkout", "-q", candidate)
    state_root = tmp_path / "state"
    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=state_root,
        build_root=build_root,
        candidate=candidate,
    )
    monkeypatch.chdir(build_root)
    tag = _production_cache_tag(1_786_872_223)
    normalized_tag = tag + "-normalized:latest"
    raw_archive = tmp_path / "raw.tar"
    normalized_archive = tmp_path / "normalized.tar"

    assert adapter.command_docker(_build_argv(tag)) == 0
    assert adapter.command_docker(["save", tag, "-o", str(raw_archive)]) == 0
    normalize_saved_image(
        archive_path=raw_archive,
        output_path=normalized_archive,
        normalized_image=normalized_tag,
        temporary_parent=tmp_path,
    )
    assert adapter.command_docker(["load", "-i", str(normalized_archive)]) == 0

    handle, state = adapter._locked_state()
    cache_record = state["images"][normalized_tag]
    for field in ("provenance", "source_tag", "build_root"):
        cache_record.pop(field)
    adapter._save_state(handle, state)
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            normalized_tag,
            "--output-file",
            str(tmp_path / "stripped-cache.eif"),
        ]
    ) == 97

    alternate_archive = tmp_path / "alternate.tar"
    alternate_tag = "candidate-cache-alias:latest"
    normalize_saved_image(
        archive_path=raw_archive,
        output_path=alternate_archive,
        normalized_image=alternate_tag,
        temporary_parent=tmp_path,
    )
    assert adapter.command_docker(["load", "-i", str(alternate_archive)]) == 0
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            alternate_tag,
            "--output-file",
            str(tmp_path / "alternate-cache.eif"),
        ]
    ) == 97


def test_redundant_normalized_tag_tolerates_transient_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "pcr0-builder"
    parent, candidate = _two_commit_repository(build_root)
    state_root = tmp_path / "state"
    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=state_root,
        build_root=build_root,
        candidate=candidate,
    )
    monkeypatch.chdir(build_root)
    tag = _production_cache_tag(1_786_872_224)
    normalized_tag = tag + "-normalized:latest"
    raw_archive = tmp_path / "raw.tar"
    normalized_archive = tmp_path / "normalized.tar"

    assert adapter.command_docker(_build_argv(tag)) == 0
    assert adapter.command_docker(["save", tag, "-o", str(raw_archive)]) == 0
    normalize_saved_image(
        archive_path=raw_archive,
        output_path=normalized_archive,
        normalized_image=normalized_tag,
        temporary_parent=tmp_path,
    )
    assert adapter.command_docker(["load", "-i", str(normalized_archive)]) == 0

    # ``docker load`` already established the immutable normalized-image
    # provenance. The production normalizer then repeats the same tag while
    # the shared historical checkout can be moving; only the later Nitro use
    # must require the checkout to have returned to the recorded commit.
    _git(build_root, "checkout", "-q", candidate)
    assert adapter.command_docker(
        ["tag", normalized_image_id(parent, VALIDATOR_ROLE), normalized_tag]
    ) == 0
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            normalized_tag,
            "--output-file",
            str(tmp_path / "wrong-head.eif"),
        ]
    ) == 97

    _git(build_root, "checkout", "-q", parent)
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            normalized_tag,
            "--output-file",
            str(tmp_path / "restored-head.eif"),
        ]
    ) == 0


def test_candidate_validator_app_restores_measured_copy_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = "a" * 40
    state_root = tmp_path / "state"
    validator_app = state_root / "validator-app"
    package = validator_app / "validator_tee" / "enclave"
    package.mkdir(parents=True)
    (package / "runtime_v2.py").write_text("VALUE = 1\n", encoding="utf-8")
    for path in validator_app.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    expected = compute_app_manifest_hash(validator_app)

    # Evidence ownership handoff deliberately grants write access. The exact
    # Nitro boundary must restore the production Docker-context modes before
    # the enclave computes its signed application manifest.
    for path in validator_app.rglob("*"):
        path.chmod(0o777 if path.is_dir() else 0o666)

    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=state_root,
        build_root=tmp_path / "build",
        candidate=candidate,
    )
    runtime_app = tmp_path / "runtime-app"
    monkeypatch.setattr(
        adapter,
        "Path",
        lambda value: runtime_app if value == "/app" else Path(value),
    )
    handle, state = adapter._locked_state()
    state["images"]["validator-tee-enclave:latest"] = {
        "commit": candidate,
        "id": normalized_image_id(candidate, VALIDATOR_ROLE),
        "role": VALIDATOR_ROLE,
    }
    adapter._save_state(handle, state)
    eif_path = tmp_path / "validator.eif"

    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            "validator-tee-enclave:latest",
            "--output-file",
            str(eif_path),
        ]
    ) == 0
    assert compute_app_manifest_hash(runtime_app) == expected
    assert all(
        (path.stat().st_mode & 0o777) == (0o755 if path.is_dir() else 0o644)
        for path in runtime_app.rglob("*")
    )


def test_pcr0_cache_adapter_rejects_near_miss_builds_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_root = tmp_path / "pcr0-builder"
    parent, candidate = _two_commit_repository(build_root)
    state_root = tmp_path / "state"
    adapter = _load_adapter(
        monkeypatch=monkeypatch,
        state_root=state_root,
        build_root=build_root,
        candidate=candidate,
    )
    monkeypatch.chdir(build_root)
    tag = _production_cache_tag(1_786_872_222)

    wrong_dockerfile = _build_argv(tag)
    wrong_dockerfile[3] = "validator_tee/Dockerfile.base"
    assert adapter.command_docker(wrong_dockerfile) == 97
    wrong_context = _build_argv(tag)
    wrong_context[-1] = "validator_tee"
    assert adapter.command_docker(wrong_context) == 97
    monkeypatch.chdir(tmp_path)
    assert adapter.command_docker(_build_argv(tag)) == 97
    monkeypatch.chdir(build_root)

    near_miss_tag = "validator-enclave-build-0"
    assert adapter.command_docker(_build_argv(near_miss_tag)) == 0
    assert adapter.command_docker(
        ["save", near_miss_tag, "-o", str(tmp_path / "near-miss.tar")]
    ) == 97

    assert adapter.command_docker(_build_argv(tag)) == 0
    _git(build_root, "checkout", "-q", candidate)
    changed_head_archive = tmp_path / "changed-head.tar"
    assert adapter.command_docker(
        ["save", tag, "-o", str(changed_head_archive)]
    ) == 97
    assert not changed_head_archive.exists()
    _git(build_root, "checkout", "-q", parent)
    assert adapter.command_docker(_build_argv(tag)) == 0
    raw_archive = tmp_path / "raw.tar"
    normalized_archive = tmp_path / "normalized.tar"
    normalized_tag = tag + "-normalized:latest"
    assert adapter.command_docker(["save", tag, "-o", str(raw_archive)]) == 0
    normalize_saved_image(
        archive_path=raw_archive,
        output_path=normalized_archive,
        normalized_image=normalized_tag,
        temporary_parent=tmp_path,
    )

    handle, state = adapter._locked_state()
    state["pcr0_cache_normalizations"][normalized_tag]["commit"] = candidate
    adapter._save_state(handle, state)
    assert adapter.command_docker(["load", "-i", str(normalized_archive)]) == 97

    handle, state = adapter._locked_state()
    state["pcr0_cache_normalizations"][normalized_tag]["commit"] = parent
    adapter._save_state(handle, state)
    assert adapter.command_docker(["load", "-i", str(normalized_archive)]) == 0
    handle, state = adapter._locked_state()
    state["images"][normalized_tag]["provenance"] = "copied-marker"
    adapter._save_state(handle, state)
    assert adapter.command_nitro(
        [
            "build-enclave",
            "--docker-uri",
            normalized_tag,
            "--output-file",
            str(tmp_path / "tampered.eif"),
        ]
    ) == 97

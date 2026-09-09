import io
from pathlib import Path
import subprocess
import tarfile

import pytest

from lab_arena.promotion import GitPromoter, PromotionError


def git(cwd: Path, *args: str, input_bytes: bytes | None = None) -> str:
    return subprocess.run(
        ("git", *args), cwd=cwd, input=input_bytes, check=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.decode().strip()


def archive(files: dict[str, tuple[bytes, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as bundle:
        for name, (data, mode) in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = mode
            bundle.addfile(info, io.BytesIO(data))
    return output.getvalue()


def repository(tmp_path: Path) -> tuple[Path, str, str]:
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "init", str(seed))
    git(seed, "config", "user.name", "Test")
    git(seed, "config", "user.email", "test@example.test")
    (seed / "old.txt").write_text("main")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "main")
    main = git(seed, "rev-parse", "HEAD")
    git(seed, "switch", "-c", "lab")
    (seed / "old.txt").write_text("lab")
    git(seed, "commit", "-am", "lab")
    lab = git(seed, "rev-parse", "HEAD")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "origin", f"{main}:refs/heads/main", f"{lab}:refs/heads/lab")
    return remote, main, lab


def source(prefix: str = "winner") -> bytes:
    return archive({
        f"{prefix}/harness.py": (b"def run_icp(icp):\n    return []\n", 0o644),
        f"{prefix}/bin/run": (b"#!/bin/sh\nexit 0\n", 0o755),
    })


def test_prepare_and_publish_exact_tree_with_both_history_parents(tmp_path):
    remote, main, lab = repository(tmp_path)
    promoter = GitPromoter(str(remote), tmp_path / "objects")
    plan = promoter.prepare(
        source(), round_id="round-1", submission_id="submission-1",
        timestamp="2026-09-07T12:34:56Z",
    )
    assert plan["main_before"] == main and plan["lab_before"] == lab
    assert promoter.publish(source(), plan=plan, round_id="round-1", submission_id="submission-1") == plan["commit"]
    assert git(remote, "rev-parse", "main") == git(remote, "rev-parse", "lab") == plan["commit"]
    assert git(remote, "show", "-s", "--format=%P", plan["commit"]).split() == [main, lab]
    tree = git(remote, "ls-tree", "-r", plan["commit"])
    assert "100755 blob" in tree and "\tbin/run" in tree
    assert "100644 blob" in tree and "\tharness.py" in tree
    assert "old.txt" not in tree


def test_retry_reconstructs_commit_in_empty_cache_and_already_published_is_success(tmp_path):
    remote, _, _ = repository(tmp_path)
    payload = source()
    first = GitPromoter(str(remote), tmp_path / "objects-one")
    plan = first.prepare(payload, round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z")
    second = GitPromoter(str(remote), tmp_path / "objects-two")
    assert second.publish(payload, plan=plan, round_id="r", submission_id="s") == plan["commit"]
    third = GitPromoter(str(remote), tmp_path / "objects-three")
    changed = archive({"harness.py": (b"def run_icp(icp):\n    return ['changed']\n", 0o644)})
    with pytest.raises(PromotionError, match="promotion_plan_mismatch"):
        third.publish(changed, plan=plan, round_id="r", submission_id="s")
    assert third.publish(payload, plan=plan, round_id="r", submission_id="s") == plan["commit"]


def test_lost_successful_push_response_is_confirmed_from_remote_heads(tmp_path):
    remote, _, _ = repository(tmp_path)

    class LostResponsePromoter(GitPromoter):
        def _git(self, *arguments, **kwargs):
            output = super()._git(*arguments, **kwargs)
            if arguments and arguments[0] == "push":
                raise PromotionError("promotion_git_failed")
            return output

    promoter = LostResponsePromoter(str(remote), tmp_path / "objects")
    plan = promoter.prepare(source(), round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z")
    assert promoter.publish(source(), plan=plan, round_id="r", submission_id="s") == plan["commit"]


def test_publish_fails_closed_after_concurrent_head_change(tmp_path):
    remote, _, _ = repository(tmp_path)
    promoter = GitPromoter(str(remote), tmp_path / "objects")
    plan = promoter.prepare(source(), round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z")
    seed = tmp_path / "concurrent"
    git(tmp_path, "clone", str(remote), str(seed))
    git(seed, "config", "user.name", "Test")
    git(seed, "config", "user.email", "test@example.test")
    git(seed, "switch", "main")
    (seed / "other").write_text("change")
    git(seed, "add", "other")
    git(seed, "commit", "-m", "concurrent")
    git(seed, "push", "origin", "main")
    with pytest.raises(PromotionError, match="promotion_remote_changed"):
        promoter.publish(source(), plan=plan, round_id="r", submission_id="s")
    assert git(remote, "rev-parse", "lab") == plan["lab_before"]


def test_atomic_capability_rejection_updates_neither_branch(tmp_path):
    remote, main, lab = repository(tmp_path)
    git(remote, "config", "receive.advertiseAtomic", "false")
    promoter = GitPromoter(str(remote), tmp_path / "objects")
    plan = promoter.prepare(source(), round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z")
    with pytest.raises(PromotionError, match="promotion_git_failed"):
        promoter.publish(source(), plan=plan, round_id="r", submission_id="s")
    assert git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", "lab") == lab


@pytest.mark.parametrize("name", [
    "repo/.git/config", "repo/.GIT/config", "repo/.github/workflows/pwn.yml",
    "repo/nested/.gitattributes",
])
def test_forbidden_git_and_workflow_paths_are_rejected(tmp_path, name):
    remote, _, _ = repository(tmp_path)
    payload = archive({
        "repo/harness.py": (b"def run_icp(icp):\n    return []\n", 0o644),
        name: (b"unsafe", 0o644),
    })
    # Admission now rejects Git automation before the publisher's redundant
    # path check, and the publisher maps all invalid source archives uniformly.
    with pytest.raises(PromotionError, match="promotion_archive_invalid"):
        GitPromoter(str(remote), tmp_path / "objects").prepare(
            payload, round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z"
        )


def test_source_or_metadata_change_cannot_publish_prepared_commit(tmp_path):
    remote, main, lab = repository(tmp_path)
    promoter = GitPromoter(str(remote), tmp_path / "objects")
    plan = promoter.prepare(source(), round_id="r", submission_id="s", timestamp="2026-09-07T00:00:00Z")
    with pytest.raises(PromotionError, match="promotion_plan_mismatch"):
        promoter.publish(source(), plan=plan, round_id="r", submission_id="different")
    assert git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", "lab") == lab

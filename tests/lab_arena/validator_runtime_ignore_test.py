"""Validator runtime paths must not alter Git source identity."""

from pathlib import Path
import subprocess

from lab_arena import runtime_version


def test_root_runtime_paths_are_ignored_without_hiding_other_source_changes(tmp_path):
    root = tmp_path / "checkout"
    package = root / "lab_arena"
    package.mkdir(parents=True)
    module = package / "runner.py"
    module.write_text("# committed runner\n")
    ignore = Path(__file__).resolve().parents[2] / ".gitignore"
    (root / ".gitignore").write_bytes(ignore.read_bytes())

    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args],
                              capture_output=True, text=True, check=True).stdout.strip()

    git("init", "-q")
    git("add", ".gitignore", "lab_arena/runner.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "initial")
    commit = git("rev-parse", "HEAD")
    for directory in ("arena-runner", "validator-state"):
        path = root / directory / "runtime.bin"
        path.parent.mkdir()
        path.write_bytes(b"private runtime state")
        git("check-ignore", "-q", str(path))
    assert git("status", "--porcelain=v1", "--untracked-files=normal") == ""
    assert runtime_version.source_metadata(str(module)) == {
        "validator_source_commit": commit,
        "validator_source_origin": "git_checkout",
        "validator_source_dirty": "clean",
    }

    nested = root / "nested" / "arena-runner" / "unrelated.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("unrelated")
    assert runtime_version.source_metadata(str(module))["validator_source_dirty"] == "dirty"
    nested.unlink()
    unrelated = root / "unrelated.txt"
    unrelated.write_text("unrelated")
    assert runtime_version.source_metadata(str(module))["validator_source_dirty"] == "dirty"
    unrelated.unlink()
    module.write_text("# tracked edit\n")
    assert runtime_version.source_metadata(str(module))["validator_source_dirty"] == "dirty"

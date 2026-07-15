from pathlib import Path


def test_restart_preserves_all_tracked_diffs_before_pull():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    preserve = script.index("preserving tracked local validator checkout changes")
    stash = script.index('git stash push -m "$restart_stash_message" -- .')
    pull = script.index("git pull --ff-only origin main")

    assert preserve < stash < pull
    assert "--include-untracked" not in script[preserve:pull]

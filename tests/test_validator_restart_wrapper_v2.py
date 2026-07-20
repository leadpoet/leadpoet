from pathlib import Path


def test_restart_preserves_all_tracked_diffs_before_pull():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")
    preserve = script.index("preserving tracked local validator checkout changes")
    stash = script.index('git stash push -m "$restart_stash_message" -- .')
    fetch = script.index("git fetch origin")

    assert preserve < stash < fetch
    assert "--include-untracked" not in script[preserve:fetch]


def test_restart_allows_only_one_invocation_pinned_ancestor_commit():
    script = Path("validator_restart.sh").read_text(encoding="utf-8")

    assert 'REQUESTED_VALIDATOR_DEPLOY_COMMIT="${VALIDATOR_DEPLOY_COMMIT:-}"' in script
    assert "unset VALIDATOR_DEPLOY_COMMIT" in script
    assert '[[ "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" =~ ^[0-9a-f]{40}$ ]]' in script
    assert (
        'git merge-base --is-ancestor "$REQUESTED_VALIDATOR_DEPLOY_COMMIT" origin/main'
        in script
    )
    assert 'git checkout --detach "$REQUESTED_VALIDATOR_DEPLOY_COMMIT"' in script
    assert '"VALIDATOR_DEPLOY_COMMIT",' in script

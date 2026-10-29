import os
import sys

# Import-time gateway clients require structurally valid values. Tests replace
# these clients before I/O and explicitly remove the variables when exercising
# missing-credential behavior, so no production credential is needed in CI.
os.environ.setdefault("SUPABASE_URL", "https://test.invalid")
os.environ.setdefault(
    "SUPABASE_SERVICE_ROLE_KEY",
    "test-only-service-role-placeholder",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest

# `lab_arena.validator.main` replaces its own process with `sudo ... validator`
# when it runs on a non-root Linux host that already grants non-interactive
# sudo. CI runners are exactly that host, so any check that calls `main`
# without `--check-only` execve's the pytest session away: the run ends with
# exit code 1, no traceback, no failing test name and no coverage report, and
# every test ordered after it silently never runs.
#
# Default every test to the in-process path, which is what they all assume,
# and leave `os.execve` armed to fail loudly so a future call site cannot go
# back to killing the run silently. The re-exec keeps its own coverage in
# tests/test_arena_validator_privilege_startup.py, which stubs both itself.
_REEXEC_OWNER = "test_arena_validator_privilege_startup"


@pytest.fixture(autouse=True)
def _never_reexec_the_test_session(request, monkeypatch):
    if _REEXEC_OWNER in request.node.nodeid:
        return
    from lab_arena import validator_startup

    monkeypatch.setattr(validator_startup, "maybe_reexec_rootful", lambda *_: None)

    def refuse(path, command, environment):
        raise RuntimeError(
            "a test tried to replace the pytest process with %r; stub the "
            "re-exec in the test instead" % (path,)
        )

    monkeypatch.setattr(validator_startup.os, "execve", refuse)


@pytest.fixture(autouse=True)
def _isolate_scorer_process_environment(monkeypatch):
    # score_input is a process entrypoint. Its frozen-policy and placeholder
    # credential bindings must not leak into another in-process test. CI also
    # supplies import-only fake keys that must not become a scorer credential.
    # Individual tests still set conflicting values to exercise fail-closed
    # policy enforcement; no production validation is changed.
    from lab_arena import contracts, scoring, shim

    names = (*scoring.CREDENTIAL_ENV_NAMES, 'OPENROUTER_KEY',
             'ARENA_INTENT_EVIDENCE_MODEL', 'ARENA_INTENT_DETAILS_REVIEW_MODEL',
             'ARENA_ROLE_BATCH_MODEL', contracts.SCORE_NORMALIZATION_BINDING,
             contracts.PROVIDER_OBSERVATION_HANDOFF_BINDING,
             contracts.FIT_FP_CLASSIFICATION_BINDING, shim.TRUSTED_SCORER_ENV)
    for name in names:
        # Register every name, including absent ones, so direct environment
        # writes from the entrypoint are removed when this test ends.
        monkeypatch.setenv(name, '')
        monkeypatch.delenv(name)

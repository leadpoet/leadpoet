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


@pytest.fixture(autouse=True)
def _isolate_optional_industry_policy(monkeypatch):
    # score_input binds its process policy directly. Each production judge is
    # a fresh process; a prior test must not opt later legacy fixtures in.
    # Tests of the new reviewer or policy conflicts set the value themselves.
    monkeypatch.delenv("ARENA_INDUSTRY_EVIDENCE_MODEL", raising=False)

@pytest.fixture
def scorer_process_environment():
    """Run the scorer entrypoint with its proxy credentials, then restore CI env.

    The gateway CI keys belong to a different process. Entrypoint tests use
    the scorer container's placeholders and retain the real conflict checks.
    This is opt-in; credential-conflict tests keep their supplied environment.
    """
    from unittest.mock import patch
    from lab_arena.scorer_entrypoint import PLACEHOLDER_CREDENTIALS

    with patch.dict(os.environ, PLACEHOLDER_CREDENTIALS):
        yield

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

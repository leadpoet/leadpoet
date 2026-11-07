"""Give in-process scorer entrypoint tests their own sandbox environment."""

import os

import pytest

from lab_arena import scoring


@pytest.fixture(autouse=True)
def isolated_scorer_environment(monkeypatch):
    # The real entrypoint runs in a fresh judge process. Tests call it in the
    # pytest process, which inherits unrelated CI credentials and survives
    # across different frozen policies. Keep the real policy checks enabled.
    environment = dict(os.environ)
    bound = {}
    scoring.apply_policy_to_environment(scoring.build_scorer_policy(),
        environ=bound, credentials={name: "test-only-placeholder" for name in scoring.CREDENTIAL_ENV_NAMES})
    for name in bound:
        environment.pop(name, None)
    monkeypatch.setattr(os, "environ", environment)

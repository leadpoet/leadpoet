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
    for name in (
        *scoring.CREDENTIAL_ENV_NAMES,
        *scoring.build_scorer_policy()["env_bindings"],
        "ARENA_INTENT_EVIDENCE_MODEL",
        "ARENA_INTENT_DETAILS_REVIEW_MODEL",
        "ARENA_ROLE_BATCH_MODEL",
    ):
        environment.pop(name, None)
    monkeypatch.setattr(os, "environ", environment)

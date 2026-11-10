"""Explicit economics for tests whose PostgreSQL schema predates migration 681.

Import this fixture only in historical-policy modules. Keep the real selector,
all earlier policy transitions and original reward assertions. Current-policy
and cutover tests must not import it.
"""

from datetime import date

import pytest

from lab_arena import rewards


@pytest.fixture(autouse=True)
def historical_slot_policy(monkeypatch):
    monkeypatch.setattr(rewards, "REWARD_SLOT_POLICY_NOV11_START", date.max)
    monkeypatch.setattr(rewards, "REWARD_SLOT_POLICY_NOV11_EFFECTIVE_EPOCH", 10**12)

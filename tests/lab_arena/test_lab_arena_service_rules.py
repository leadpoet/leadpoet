"""Pure service rules that guard what a validator may assert."""

from __future__ import annotations

from lab_arena.service import refusal_evidenced


def test_a_refusal_needs_arena_recorded_evidence():
    """A judge_key_refused receipt stands only on a ledger refusal or a 401/403 the broker itself saw."""

    assert not refusal_evidenced([], [])
    assert not refusal_evidenced([{"outcome": "settled"}], [{"event_type": "provider_call", "payload": {"status": 200, "provider_status": 200}}])
    assert refusal_evidenced([{"outcome": "refused"}], [])
    assert refusal_evidenced([], [{"event_type": "provider_call", "payload": {"status": 502, "provider_status": 401}}])
    assert refusal_evidenced([], [{"event_type": "provider_call", "payload": {"status": 502, "provider_status": 403}}])
    # Events the runner writes (stdout, process events) carry no evidence, whatever they claim.
    assert not refusal_evidenced([], [{"event_type": "stdout", "payload": {"provider_status": 401, "outcome": "refused"}}])

"""A reward activation that refuses without raising must not read as idle."""
from types import SimpleNamespace

import pytest

from lab_arena.driver import drive_once


def _service(rewards):
    return SimpleNamespace(
        promote_pending_baselines=lambda: {"promoted": 0},
        active_rounds=lambda: [],
        ensure_daily_round=lambda: {"status": "existing"},
        activate_pending_rewards=lambda: rewards,
        reconcile_closed_provider_costs=lambda: {"status": "pending"},
    )


@pytest.mark.parametrize("status", ["stale", "reward_history_invalid", "blocked"])
def test_a_refused_activation_is_reported_as_a_failed_stage(status):
    result = drive_once(_service({"status": status, "activated": 0}))
    assert "failed activate_rewards: RewardActivationRefused" in result


def test_a_walked_pending_list_that_paid_nothing_stays_idle():
    assert drive_once(_service({"status": "ok", "activated": 0})) == "idle"


def test_a_disabled_activator_stays_idle():
    assert drive_once(_service({"status": "disabled", "activated": 0})) == "idle"


def test_an_absent_status_is_not_read_as_a_refusal():
    assert drive_once(_service({"activated": 0})) == "idle"


def test_a_successful_activation_still_reports_its_count():
    assert "activated rewards 1" in drive_once(_service({"status": "ok", "activated": 1}))

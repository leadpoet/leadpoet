"""An expired restart-guard lease must not withhold paid review work."""
import pytest

from scripts.run_lab_arena_service import restart_guard_withholds_review

SCHEMA = "leadpoet.lab_arena.review_guard_state.v1"


def state(**fields):
    return {"schema_version": SCHEMA, **fields}


def test_no_guard_admits_review():
    assert restart_guard_withholds_review(
        state(guard_present=False, guard_active=False)
    ) is False


def test_live_lease_withholds_review():
    assert restart_guard_withholds_review(
        state(guard_present=True, guard_active=True)
    ) is True


def test_retained_expired_lease_admits_review():
    assert restart_guard_withholds_review(
        state(guard_present=True, guard_active=False)
    ) is False


def test_presence_still_governs_before_the_migration_lands():
    # The lease-aware read ships as a hand-applied migration, so the service
    # must keep refusing on presence alone until it is in place.
    assert restart_guard_withholds_review(state(guard_present=True)) is True
    assert restart_guard_withholds_review(state(guard_present=False)) is False


@pytest.mark.parametrize("guard", [
    None,
    "guard",
    {},
    {"guard_present": True, "guard_active": False},
    state(),
    state(guard_active=False),
    state(guard_present="false", guard_active=False),
    state(guard_present=True, guard_active="false"),
    state(guard_present=True, guard_active=None),
    {"schema_version": "leadpoet.lab_arena.restart_guard_state.v1",
     "guard_present": True, "guard_active": False},
])
def test_unreadable_state_is_rejected_rather_than_admitting_review(guard):
    with pytest.raises(ValueError):
        restart_guard_withholds_review(guard)

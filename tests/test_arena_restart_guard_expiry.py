"""An abandoned restart guard must not withhold review work forever."""
import pytest

from scripts.run_lab_arena_service import restart_guard_blocks_review


def state(**overrides):
    guard = {
        "schema_version": "leadpoet.lab_arena.review_guard_state.v1",
        "guard_present": True,
        "guard_active": True,
    }
    guard.update(overrides)
    return guard


def test_live_guard_withholds_review_work() -> None:
    assert restart_guard_blocks_review(state()) is True


def test_expired_guard_releases_review_work() -> None:
    # The owner never reached its release step, so the commitment is still
    # recorded; its lease has lapsed, and paid review work must resume.
    assert restart_guard_blocks_review(state(guard_active=False)) is False


def test_absent_guard_allows_review_work() -> None:
    assert restart_guard_blocks_review(
        state(guard_present=False, guard_active=False)
    ) is False


@pytest.mark.parametrize("guard", [
    None,
    {},
    {"schema_version": "leadpoet.lab_arena.review_guard_state.v2",
     "guard_present": False, "guard_active": False},
    {"schema_version": "leadpoet.lab_arena.review_guard_state.v1",
     "guard_present": False},
    {"schema_version": "leadpoet.lab_arena.review_guard_state.v1",
     "guard_present": "no", "guard_active": False},
])
def test_unreadable_guard_state_is_rejected(guard) -> None:
    with pytest.raises(ValueError):
        restart_guard_blocks_review(guard)

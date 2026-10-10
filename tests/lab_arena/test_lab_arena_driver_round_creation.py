"""The daily driver opens the next submission round in the cutoff tick."""

from lab_arena.driver import drive_once


class DailyRoundService:
    def __init__(self, *, close_on_advance=True, fail_advance=False, fail_after_commit=False):
        self.close_on_advance = close_on_advance
        self.fail_advance = fail_advance
        self.fail_after_commit = fail_after_commit
        self.old_status = "open"
        self.next_open = False
        self.created = 0
        self.calls = []

    def promote_pending_baselines(self):
        return {"promoted": 0}

    def active_rounds(self):
        rows = [{"round_id": "old", "status": self.old_status}]
        if self.next_open:
            rows.append({"round_id": "next", "status": "open"})
        return rows

    def advance_round(self, round_id):
        self.calls.append("advance:" + round_id)
        if round_id == "old" and self.fail_advance:
            raise RuntimeError("synthetic advance failure")
        if round_id == "old" and self.close_on_advance:
            self.old_status = "committed"
            if self.fail_after_commit:
                raise RuntimeError("synthetic post-commit failure")
            return {"status": "ok"}
        return {"status": "waiting"}

    def ensure_daily_round(self):
        self.calls.append("ensure")
        if self.old_status == "open" or self.next_open:
            return {"status": "existing"}
        self.next_open = True
        self.created += 1
        return {"status": "created", "round_id": "next"}

    def activate_pending_rewards(self):
        self.calls.append("rewards")
        return {"activated": 0}

    def reconcile_closed_provider_costs(self):
        self.calls.append("billing")
        return {"status": "none"}


def test_cutoff_commit_creates_next_round_in_the_same_tick():
    service = DailyRoundService()

    result = drive_once(service)

    assert result == "advanced old; created next"
    assert service.calls == ["advance:old", "ensure", "rewards", "billing"]
    assert service.old_status == "committed"
    assert service.next_open
    assert service.created == 1

    drive_once(service)
    assert service.created == 1  # The service operation remains idempotent.


def test_existing_open_round_is_not_duplicated():
    service = DailyRoundService(close_on_advance=False)

    assert drive_once(service) == "advanced old"
    assert service.calls == ["advance:old", "ensure", "rewards", "billing"]
    assert service.old_status == "open"
    assert not service.next_open
    assert service.created == 0


def test_failed_advance_does_not_create_a_second_open_round():
    service = DailyRoundService(fail_advance=True)

    result = drive_once(service)

    assert result == "failed advance_round old: RuntimeError"
    assert service.calls == ["advance:old", "ensure", "rewards", "billing"]
    assert service.old_status == "open"
    assert not service.next_open
    assert service.created == 0


def test_ambiguous_post_commit_failure_still_opens_the_next_round():
    service = DailyRoundService(fail_after_commit=True)

    result = drive_once(service)

    assert result == "failed advance_round old: RuntimeError; created next"
    assert service.calls == ["advance:old", "ensure", "rewards", "billing"]
    assert service.old_status == "committed"
    assert service.next_open
    assert service.created == 1

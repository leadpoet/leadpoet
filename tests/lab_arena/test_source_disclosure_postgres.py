"""Acceptance time survives freeze, retries, and process restarts."""

from datetime import datetime, timedelta, timezone

from tests.lab_arena.test_lab_arena_service_round import Harness, connect, database


def test_database_stamps_acceptance_once_and_preserves_it(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    config = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12), round_id="arena-2026-09-09-disclosure"
    )
    harness.round_id = config["round_id"]
    harness.clock.now = datetime.now(timezone.utc)
    before = datetime.now(timezone.utc)
    submission_id = harness.submit("Disclosure", harness.round_id)
    after = datetime.now(timezone.utc)
    row = harness.service.store.get_submission(submission_id)
    accepted = datetime.fromisoformat(str(row["accepted_at"]).replace("Z", "+00:00"))
    assert before <= accepted <= after
    connection = connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE public.lab_arena_submissions SET accepted_at=accepted_at-interval '1 day' WHERE submission_id=%s",
                (submission_id,),
            )
        connection.commit()
    finally:
        connection.close()
    assert harness.service.store.get_submission(submission_id)["accepted_at"] == row["accepted_at"]
    assert harness.service.store.update_submission(
        harness.round_id, submission_id, "accepted", "frozen"
    )["status"] == "ok"
    assert harness.build_service().store.get_submission(submission_id)["accepted_at"] == row["accepted_at"]

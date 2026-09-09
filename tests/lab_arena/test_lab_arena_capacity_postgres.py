"""The capacity migration repairs open configuration without evicting miners."""

from datetime import datetime, timedelta, timezone

import pytest

from lab_arena.capacity import daily_challenger_capacity
from lab_arena.service import ServiceError
from tests.lab_arena.test_lab_arena_service_round import Harness, connect, database
from tests.postgres_migration_harness import SCRIPTS


def old_round(connect, tmp_path, suffix, count, *, cap=16, changed_schedule=None):
    harness = Harness(connect, tmp_path, challengers=[], runners=["capacity"])
    harness.clock.now = datetime.now(timezone.utc)
    cutoff = (harness.clock.now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    harness.service.config.defaults.stage_minutes = {
        "benchmark": 30, "stage_1": 240, "stage_1_scoring": 360,
        "stage_2": 180, "final_scoring": 240,
    }
    # Build a historical configuration, then register it through the actual
    # store. New create_round() correctly no longer advertises this capacity.
    round_id = "arena-2099-01-01-" + suffix
    harness.service.config.mode = "shadow"
    config = harness.service.create_round(cutoff, round_id=round_id)
    # Use a separate live ID so no immutable row needs rewriting in setup.
    config.update(round_id=round_id + "live", mode="live", max_challengers=cap)
    if changed_schedule:
        field = changed_schedule
        value = datetime.fromisoformat(config["schedule"][field].replace("Z", "+00:00"))
        config["schedule"][field] = (value + timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    harness.service.config.mode = "live"
    harness.service.store.create_round(config["round_id"], config)
    for index in range(count):
        harness.submit("capacity-%s-%d" % (suffix, index), config["round_id"])
    return harness, config


def apply_migration(connect):
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute((SCRIPTS / "201-lab-arena-daily-capacity.sql").read_text())
    finally:
        connection.close()


def test_open_round_upgrade_preserves_submissions_and_reenables_immutability(connect, tmp_path):
    harness, before = old_round(connect, tmp_path, "upgrade", 7)
    round_id = before["round_id"]
    submissions = harness.service.store.list_submissions(round_id)
    apply_migration(connect)
    after = harness.service.store.get_round(round_id)["configuration_doc"]
    assert after["max_challengers"] == daily_challenger_capacity(after) == 8
    assert after["schedule"]["submission_open"] == before["schedule"]["submission_open"]
    assert after["schedule"]["submission_cutoff"] == before["schedule"]["submission_cutoff"]
    assert {k: v for k, v in after.items() if k not in ("schedule", "max_challengers")} == {
        k: v for k, v in before.items() if k not in ("schedule", "max_challengers")
    }
    assert harness.service.store.list_submissions(round_id) == submissions
    apply_migration(connect)
    assert harness.service.store.get_round(round_id)["configuration_doc"] == after
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT tgenabled FROM pg_trigger WHERE tgname='lab_arena_rounds_write_once'")
            assert cursor.fetchone() == ("O",)
            with pytest.raises(Exception, match="write-once"):
                cursor.execute("UPDATE public.lab_arena_rounds SET configuration_doc=configuration_doc || '{\"max_challengers\":99}'::jsonb WHERE round_id=%s", (round_id,))


def test_upgrade_refuses_to_evict_previously_accepted_miners(connect, tmp_path):
    harness, before = old_round(connect, tmp_path, "preserve", 9)
    try:
        with pytest.raises(Exception, match="preserving_all_submissions"):
            apply_migration(connect)
        assert harness.service.store.get_round(before["round_id"])["configuration_doc"] == before
        assert len(harness.service.store.list_submissions(before["round_id"], status="accepted")) == 9
        with connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("SELECT tgenabled FROM pg_trigger WHERE tgname='lab_arena_rounds_write_once'")
                assert cursor.fetchone() == ("O",)
    finally:
        harness.service.cancel(before["round_id"], "operator")


def test_parallel_finalizations_cannot_overbook_or_store_rejected_credentials(connect, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    harness = Harness(connect, tmp_path, challengers=[], runners=["capacity"])
    harness.max_challengers = 1
    harness.service = harness.build_service()
    harness.clock.now = datetime.now(timezone.utc)
    config = harness.service.create_round(
        harness.clock.now + timedelta(hours=12), round_id="arena-2099-01-01-parallel"
    )
    barrier = Barrier(3)
    manager = harness.service.config.credential_manager
    original = manager.validate_and_encrypt

    def validate(*args, **kwargs):
        result = original(*args, **kwargs)
        barrier.wait(timeout=10)
        return result

    manager.validate_and_encrypt = validate

    def submit(index):
        try:
            harness.submit("parallel-capacity-%d" % index, config["round_id"])
            return "accepted"
        except ServiceError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(submit, range(3)))
    assert results.count("accepted") == 1
    assert results.count("submission_rejected:capacity.round_full") == 2
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM public.lab_arena_submission_credentials c "
                "JOIN public.lab_arena_submissions s USING(submission_id) WHERE s.round_id=%s",
                (config["round_id"],),
            )
            assert cursor.fetchone() == (2,)


@pytest.mark.parametrize("index,field", list(enumerate([
    "submission_open", "benchmark_deadline", "stage_1_start", "stage_1_close",
    "stage_1_scoring_close", "stage_2_start", "stage_2_close",
    "final_scoring_close", "publication_deadline",
])))
def test_upgrade_leaves_custom_schedules_unchanged(connect, tmp_path, index, field):
    harness, before = old_round(connect, tmp_path, "custom" + str(index), 0, changed_schedule=field)
    apply_migration(connect)
    assert harness.service.store.get_round(before["round_id"])["configuration_doc"] == before


def test_upgrade_leaves_operator_limited_round_unchanged(connect, tmp_path):
    harness, before = old_round(connect, tmp_path, "lowercap", 0, cap=5)
    # Uploading rows can predate a lowered cap; a repair must not silently turn
    # that operator configuration into a daily round with a different ceiling.
    from lab_arena import contracts
    from tests.lab_arena.test_lab_arena_service_round import keypair
    for index in range(7):
        miner = keypair("uploading-lower-%d" % index)
        request = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_PRESIGN,
            round_id=before["round_id"],
            hotkey=miner.ss58_address,
            body={"source_size_bytes": 100, "consent": {"public_rerun": True}},
            timestamp=int(harness.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex(),
        )
        # Registration does not finalize or store credentials.
        harness.service.handle_submission_presign(request)
    submissions = harness.service.store.list_submissions(before["round_id"])
    assert len(submissions) == 7
    apply_migration(connect)
    assert harness.service.store.get_round(before["round_id"])["configuration_doc"] == before
    assert harness.service.store.list_submissions(before["round_id"]) == submissions

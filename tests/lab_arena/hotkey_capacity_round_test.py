"""Current-schema admission and a full twenty-challenger competition."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import capacity, contracts, service as svc
from tests.lab_arena import test_lab_arena_service_round as fixtures
from tests.lab_arena.contact_round_test import ContactHarness, _install_contact_sandbox
from tests.lab_arena.lab_arena_pg_harness import POSTGREST_MIGRATIONS, database_with_lab_arena_migration


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


connect = fixtures.connect


def test_default_admission_is_twenty_with_one_planned_runner(connect, tmp_path, monkeypatch):
    harness = fixtures.Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    harness.service.config.defaults = replace(harness.service.config.defaults, max_challengers=contracts.DEFAULT_MAX_CHALLENGERS)
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    configuration = harness.service.create_round(cutoff, round_id="arena-2098-01-01-capdefault")
    assert configuration["max_challengers"] == 20
    assert capacity.daily_challenger_capacity(configuration) == 8
    assert configuration["runner_slot_ceiling"] == 8
    assert configuration["max_attempts_per_assignment"] == 2
    monkeypatch.setattr(capacity, "daily_challenger_capacity", lambda _: 0)
    with pytest.raises(svc.ServiceError, match="daily_runner_capacity_insufficient"):
        harness.service.create_round(cutoff, round_id="arena-2098-01-02-capzero")


def test_twenty_shared_owner_hotkeys_publish_with_intact_source_and_credentials(connect, tmp_path):
    """Signed intake, real PostgreSQL/broker/runner, nonempty contact results, publication."""
    flavors = [f"SharedOwner{index:02d}" for index in range(20)]
    harness = ContactHarness(connect, tmp_path, challengers=flavors, runners=["alpha"])
    _install_contact_sandbox(harness)
    harness.service.config.defaults = replace(harness.service.config.defaults, max_challengers=contracts.DEFAULT_MAX_CHALLENGERS)
    original_metagraph = harness.chain.metagraph
    shared_owner = fixtures.keypair("twenty-shared-coldkey").ss58_address

    def metagraph(*, finalized=True):
        snapshot = original_metagraph(finalized=finalized)
        return replace(snapshot, coldkeys=tuple(
            hotkey if hotkey in harness.runner_keys else shared_owner
            for hotkey in snapshot.hotkeys
        ))

    harness.chain.metagraph = metagraph
    harness.chain.epoch = 49_000
    harness.clock.now = datetime.now(timezone.utc)
    harness.round_id = "arena-2098-01-03-capfull"
    configuration = harness.service.create_round(harness.clock.now + timedelta(hours=12), round_id=harness.round_id)
    assert configuration["max_challengers"] == 20
    assert configuration["integrity_policy"] == "arena_integrity_v1"
    assert configuration["contact_policy"] == "contacts_v1"
    submitted = [harness.submit(flavor, harness.round_id) for flavor in flavors]
    original_rows = {submission: harness.service.store.get_submission(submission) for submission in submitted}
    original_source = {submission: harness.objects.get(row["source_ref"]) for submission, row in original_rows.items()}
    assert {row["owner_coldkey"] for row in original_rows.values()} == {shared_owner}

    # Accepted source is immutable, including when its owner has other hotkeys.
    with pytest.raises(svc.ServiceError, match="submission_conflict"):
        harness.submit("ChangedSourceWithADifferentSize", harness.round_id, miner_label=flavors[0])
    with pytest.raises(svc.ServiceError, match="submission_rejected:capacity.round_full"):
        harness.submit("TwentyFirst", harness.round_id)
    assert all(harness.service.store.get_submission(submission) == row for submission, row in original_rows.items())

    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    published = harness.advance_until("published", runners=1, max_steps=100)
    assert len(published["participants"]) == 21
    assert sum(participant["is_king"] for participant in published["participants"]) == 1
    assert len(published["publication_doc"]["final_ranking"]) == 21
    for stage in (1, 2):
        for kind in ("execute", "score"):
            runs = harness.service.store.list_runs(harness.round_id, stage=stage, kind=kind)
            assert len(runs) == 21 * 10
            assert {run["status"] for run in runs} == {"accepted"}
    ranking = {row["submission_id"]: row for row in published["publication_doc"]["final_ranking"]}
    assert published["king_outcome"] == "crowned"
    for submission, original in original_rows.items():
        assert ranking[submission]["main_score"] > 0
        saved = harness.service.store.get_submission(submission)
        assert saved["owner_coldkey"] == shared_owner
        assert saved["source_ref"] == original["source_ref"]
        assert harness.objects.get(saved["source_ref"]) == original_source[submission]
        public = harness.service.public_results(harness.round_id, submission)
        # Only selected finalists have a confirmation score; all twenty keep
        # their main score and the twenty evaluated ICPs.
        assert (public["submission_scores"]["final"] is not None) == ranking[submission]["confirmation_selected"]
        assert len(public["outputs"]) >= 20
        assert all(output["companies"] for output in public["outputs"].values())
    fixtures.assert_canary_absent(harness, connect)

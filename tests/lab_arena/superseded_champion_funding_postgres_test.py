"""Funding ignores only crowns superseded by a later published live day."""

from __future__ import annotations

from contextlib import closing
from datetime import datetime, timedelta, timezone
import json

import pytest

from lab_arena import contracts
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import (
    claim,
    commit_round,
    frozen_participants,
    hotkey,
    round_config,
    source_submission_doc,
)
from tests.lab_arena.test_lab_arena_service_round import Harness
from tests.postgres_migration_harness import SCRIPTS


MIGRATION = "393-lab-arena-superseded-champion-funding.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
    )


def _store(database):
    psycopg2, dsn = database
    return ArenaStore(PsycopgTransport(lambda: psycopg2.connect(**dsn)))


def _connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


def _round(store, label, day, *, network="finney", netuid=71,
           mode="live", rewards=True):
    round_id = f"arena-{day}-{label.replace('-', '')[:16]}"
    config = round_config(
        round_id, [hotkey(label + "-runner")], mode=mode,
        rewards_enabled=rewards if mode == "live" else False,
    )
    config.update(network_name=network, netuid=netuid)
    assert store.create_round(round_id, config)["status"] == "created"
    return round_id


def _publish(database, store, label, day, *, network="finney", netuid=71,
             mode="live", rewards=True, crowned=False, promoted=False):
    round_id = _round(
        store, label, day, network=network, netuid=netuid,
        mode=mode, rewards=rewards,
    )
    winner = (
        frozen_participants(store, round_id, 1, prefix=label + "-winner")[0]
        if crowned else None
    )
    decision = {"outcome": "crowned" if crowned else "defended"}
    if winner:
        decision.update(
            winner_submission_id=winner["submission_id"],
            king_hotkey=winner["miner_hotkey"],
        )
    with closing(_connect(database)()) as connection, connection:
        with connection.cursor() as cursor:
            cursor.execute("ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER")
            cursor.execute(
                "UPDATE public.lab_arena_rounds SET status='published', "
                "evaluation_date=%s, published_at=clock_timestamp(), "
                "promotion_required=%s, baseline_promoted_at=%s, "
                "publication_doc=%s::jsonb, king_outcome=%s, king_hotkey=%s "
                "WHERE round_id=%s",
                (
                    day, crowned,
                    datetime.now(timezone.utc) if promoted else None,
                    json.dumps({"king_decision": decision}),
                    decision["outcome"], decision.get("king_hotkey"), round_id,
                ),
            )
            cursor.execute("ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER")
    return round_id, winner


def test_freeze_blocks_latest_pending_but_not_superseded_history(database):
    connect = _connect(database)
    store = _store(database)
    promoted_id, promoted_winner = _publish(
        database, store, "prior-promoted", "2026-09-30", netuid=791,
        crowned=True, promoted=True,
    )
    pending_id, _ = _publish(
        database, store, "prior-pending", "2026-10-01", netuid=791,
        crowned=True,
    )
    current_id = _round(store, "current", "2026-10-03", netuid=791)
    assert store.freeze_champion_funding(current_id) == {
        "status": "promotion_pending"
    }
    assert store.get_round(current_id)["champion_funding_frozen"] is False

    # Neither same-day publication nor a later publication outside the exact
    # live chain scope supersedes the outstanding crown.
    _publish(database, store, "same-day", "2026-10-01", netuid=791,
             crowned=True)
    _publish(database, store, "other-mode", "2026-10-02", netuid=791,
             mode="shadow", crowned=True)
    _publish(database, store, "other-network", "2026-10-02",
             network="testnet", netuid=791, crowned=True)
    _publish(database, store, "other-netuid", "2026-10-02", netuid=792,
             crowned=True)
    assert store.freeze_champion_funding(current_id) == {
        "status": "promotion_pending"
    }

    # The service's published-day authority does not require the newer day
    # itself to be reward-enabled. The older crown remains unpromoted history.
    _publish(database, store, "newer-live", "2026-10-02", netuid=791,
             rewards=False)
    assert store.freeze_champion_funding(current_id) == {"status": "frozen"}
    assert store.freeze_champion_funding(current_id) == {"status": "existing"}
    current = store.get_round(current_id)
    assert current["champion_submission_id"] == promoted_winner["submission_id"]
    assert current["champion_hotkey"] == promoted_winner["miner_hotkey"]
    assert store.get_round(pending_id)["baseline_promoted_at"] is None
    assert store.get_round(promoted_id)["baseline_promoted_at"] is not None

    baseline_id = "baseline-" + current_id.removeprefix("arena-")
    baseline_hotkey = current["configuration_doc"]["baseline_hotkey"]
    assert store.register_submission(
        current_id, baseline_id, baseline_hotkey,
        source_submission_doc(current_id, baseline_id, is_king=True),
    )["status"] == "registered"
    assert store.update_submission(
        current_id, baseline_id, "uploading", "accepted",
    )["status"] == "ok"
    assert store.update_submission(
        current_id, baseline_id, "accepted", "frozen", {"is_king": True},
    )["status"] == "ok"
    participant = {
        "submission_id": baseline_id, "miner_hotkey": baseline_hotkey,
        "is_king": True,
    }
    commit_round(store, current_id, [participant])
    assert store.open_stage(
        current_id, 1, [participant], list(contracts.stage_positions(1)),
    )["status"] == "ok"
    run, _token, _request_id, _request_hash = claim(
        store, current_id, hotkey("current-runner"),
    )
    for provider in contracts.PROVIDERS:
        funding = store.provider_funding(run["run_id"], provider)
        assert funding["funding_source"] == "miner_key"
        assert funding["credential_submission_id"] == promoted_winner[
            "submission_id"
        ]
        assert funding["credential_miner_hotkey"] == promoted_winner[
            "miner_hotkey"
        ]

    shadow_id = _round(store, "shadowcurrent", "2026-10-04", netuid=791,
                       mode="shadow")
    assert store.freeze_champion_funding(shadow_id) == {
        "status": "promotion_pending"
    }

    # A newly published but still unpromoted crown is not superseded by the
    # older published day. A new open round must still wait for its promotion.
    _publish(database, store, "latest-pending", "2026-10-03", netuid=791,
             crowned=True)
    next_id = _round(store, "next", "2026-10-04", netuid=791)
    assert store.freeze_champion_funding(next_id) == {
        "status": "promotion_pending"
    }
    assert store.get_round(next_id)["champion_funding_frozen"] is False

    # Reapply the numbered migration without changing ownership or behavior.
    with closing(connect()) as connection, connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_freeze_champion_funding(text)'::regprocedure)"
            )
            before = cursor.fetchone()[0]
            cursor.execute((SCRIPTS / MIGRATION).read_text(encoding="utf-8"))
            cursor.execute(
                "SELECT pg_get_functiondef("
                "'public.lab_arena_freeze_champion_funding(text)'::regprocedure)"
            )
            assert cursor.fetchone()[0] == before
    assert store.freeze_champion_funding(next_id) == {
        "status": "promotion_pending"
    }


def test_service_commit_benchmark_crosses_superseded_crown(database, tmp_path):
    connect = _connect(database)
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(minutes=30),
        round_id="arena-2026-10-03-superseded",
    )
    harness.round_id = configuration["round_id"]
    pending_id, _ = _publish(
        database, harness.service.store, "service-old-pending", "2026-10-01",
        crowned=True,
    )
    _publish(database, harness.service.store, "service-newer", "2026-10-02",
             rewards=False)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    result = harness.service.commit_benchmark(harness.round_id)
    assert result["status"] == "ok"
    current = harness.service.store.get_round(harness.round_id)
    assert current["status"] == "committed"
    assert current["champion_funding_frozen"] is True
    assert harness.service.store.get_round(pending_id)["baseline_promoted_at"] is None

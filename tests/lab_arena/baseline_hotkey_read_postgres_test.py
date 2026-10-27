"""The per-call baseline payer check reads one configured key, not the round row.

``SubmissionProviderKeys`` decides the payer on every provider call. Reading the
whole round there moved its multi-megabyte ``configuration_doc`` on each call.
"""

from __future__ import annotations

import pytest

from lab_arena.broker import RunContext
from lab_arena.store import ArenaStore, PsycopgTransport
from lab_arena.submission_runtime import SubmissionProviderKeys
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_migration_postgres import hotkey, round_config


ROUND_ID = "arena-2026-10-27-baselineread"
BASELINE_ID = "baseline-2026-10-27-baselineread"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


class RecordingTransport(PsycopgTransport):
    def __init__(self, connect):
        super().__init__(connect)
        self.round_reads = []

    def select(self, table, **kwargs):
        rows = super().select(table, **kwargs)
        if table == "lab_arena_rounds":
            self.round_reads.append((kwargs.get("columns", "*"), rows))
        return rows


class SubmissionStore(ArenaStore):
    """Real round reads; the submission row is the only synthetic input."""

    def __init__(self, transport, submission):
        super().__init__(transport)
        self.submission = submission

    def get_submission(self, submission_id):
        return dict(self.submission) if submission_id == self.submission["submission_id"] else None


@pytest.fixture(scope="module")
def transport(database):
    psycopg2, dsn = database
    transport = RecordingTransport(lambda: psycopg2.connect(**dsn))
    store = ArenaStore(transport)
    config = round_config(ROUND_ID, [hotkey("baseline-read-runner")])
    assert store.create_round(ROUND_ID, config)["status"] == "created"
    return transport


def _payer(transport, **submission):
    row = {
        "submission_id": BASELINE_ID,
        "round_id": ROUND_ID,
        "miner_hotkey": hotkey("baseline"),
        "status": "frozen",
        "is_king": True,
        **submission,
    }
    keys = SubmissionProviderKeys(
        store=SubmissionStore(transport, row), credentials=None, organizer_keys={}
    )
    context = RunContext(
        run_id="run-baseline-read",
        assignment_id=ROUND_ID + ":baseline-read",
        icp_position=0,
        lease_token_hash="0" * 64,
        miner_hotkey=row["miner_hotkey"],
        submission_id=row["submission_id"],
        stage=1,
        kind="execute",
        round_id=ROUND_ID,
    )
    return keys.funding_source_for(context)


def test_store_reads_only_the_configured_baseline_hotkey(transport):
    store = ArenaStore(transport)
    transport.round_reads.clear()

    assert store.get_round_baseline_hotkey(ROUND_ID) == hotkey("baseline")
    assert store.get_round_baseline_hotkey("arena-2026-10-27-missing") is None

    for columns, rows in transport.round_reads:
        assert columns == "cfg_baseline_hotkey:configuration_doc->baseline_hotkey::text"
        assert all(set(row) == {"cfg_baseline_hotkey"} for row in rows)


def test_baseline_payer_decision_is_unchanged_without_round_row_reads(transport):
    transport.round_reads.clear()

    assert _payer(transport) == "host"
    assert _payer(transport, is_king=False) == "miner_key"
    assert _payer(transport, miner_hotkey=hotkey("not-baseline")) == "miner_key"
    assert _payer(transport, submission_id="sub-not-baseline") == "miner_key"

    assert transport.round_reads, "the baseline payer check must read its round"
    assert all(columns != "*" for columns, _rows in transport.round_reads)
    assert all(
        "configuration_doc" not in row for _columns, rows in transport.round_reads for row in rows
    )

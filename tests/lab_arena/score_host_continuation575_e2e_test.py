"""Ten-slot execution, judging and publication with current recovery SQL."""

from pathlib import Path

import pytest
from scripts.verify_arena_full_thinking_e2e import full_thinking_database

from tests.lab_arena.idle_worker_claims416_e2e_test import (
    test_ten_slot_baseline_first_round_publishes_promotes_and_derives_weights
    as _complete_round,
)
from tests.lab_arena.parallel_twenty_runtime_e2e_test import (
    bounded_socket_shutdown_poll,
)


@pytest.fixture
def database():
    # Prove the exact575 upgrade from its own preimage, independently of the
    # current-runtime fixture used by the imported full-round scenario.
    yield from full_thinking_database()


def test_ten_slot_round_preserves_results_and_weights_after_recovery_fix(
    database, tmp_path, monkeypatch,
):
    migration = Path(__file__).parents[2] / "scripts/575-lab-arena-score-host-continuation.sql"
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute(migration.read_text())
    _complete_round(database, tmp_path, monkeypatch)

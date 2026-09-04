"""Capacity: one round at the challenger cap on disposable PostgreSQL.

Every registered miner may enter one agent per day, up to ``MAX_CHALLENGERS``
(256). This test admits that many, runs both stages with fake sandboxes and
in-process validators, and reports the wall time of each service step so the
floor and the scoring workers can be sized (RUNBOOK section 8). It is slow,
so it runs only when ``LAB_ARENA_SCALE_TEST=1``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

from lab_arena import contracts

from tests.lab_arena.test_lab_arena_service_round import Harness, connect, database  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.environ.get("LAB_ARENA_SCALE_TEST") != "1", reason="set LAB_ARENA_SCALE_TEST=1 to run the challenger-cap round")

CHALLENGERS = int(os.environ.get("LAB_ARENA_SCALE_CHALLENGERS", str(contracts.MAX_CHALLENGERS)))
RUNNERS = ["alpha", "beta", "gamma"]


def test_round_at_the_challenger_cap_scores_every_participant_and_publishes(connect, tmp_path):
    timings = {}

    def timed(label, call):
        started = time.monotonic()
        result = call()
        timings[label] = round(time.monotonic() - started, 2)
        return result

    flavors = ["Scale-%03d" % index for index in range(CHALLENGERS)]
    harness = Harness(connect, tmp_path, challengers=flavors, runners=RUNNERS)
    harness.max_challengers = contracts.MAX_CHALLENGERS
    harness.service = harness.build_service()
    service = harness.service
    harness.chain.epoch = 30500
    # PostgreSQL enforces the submission window with its real clock. Keep the
    # disposable round open while the fake service clock drives later stages.
    configuration = service.create_round(datetime.now(timezone.utc) + timedelta(hours=1))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    timed("submissions", lambda: [harness.submit(flavor, round_id) for flavor in flavors])
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    freeze = timed("freeze_and_benchmark", lambda: service.advance_round(round_id))
    assert freeze["status"] == "ok", freeze
    participants = service.store.get_round(round_id)["participants"]
    assert len(participants) == CHALLENGERS + 1
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = timed("stage1_open", lambda: service.advance_round(round_id))
    assert opened["assignments"] == contracts.STAGE_1_ICP_COUNT * (CHALLENGERS + 1)
    timed("stage1_executions", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_close", lambda: service.advance_round(round_id))["status"] == "ok"
    assert timed("stage1_scoring_open", lambda: service.advance_round(round_id))["assignments"] == contracts.STAGE_1_ICP_COUNT * (CHALLENGERS + 1)
    timed("stage1_scorings", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_scoring_close", lambda: service.advance_round(round_id))["status"] == "closed"
    assert timed("stage1_score_stage", lambda: service.advance_round(round_id))["status"] == "ok"
    assert harness.status() == "stage1_scored"
    timed("stage2_to_published", lambda: harness.advance_until("published", runners=len(RUNNERS), max_steps=200))
    round_row = service.store.get_round(round_id)
    assert round_row["status"] == "published" and round_row["king_outcome"] in ("crowned", "no_king")
    publication = round_row["publication_doc"]
    finalist_count = min(contracts.FINALIST_COUNT, CHALLENGERS)
    assert len(publication["stage1_ranking"]) == CHALLENGERS
    assert len(round_row["finalists"]) == finalist_count
    assert len(publication["final_ranking"]) == finalist_count + 1
    assert len(service.store.list_runs(round_id, stage=1, kind="execute")) == contracts.STAGE_1_ICP_COUNT * (CHALLENGERS + 1)
    assert len(service.store.list_runs(round_id, stage=2, kind="execute")) == contracts.STAGE_2_ICP_COUNT * (finalist_count + 1)
    report = {
        "participants": CHALLENGERS + 1,
        "stage1_runs": contracts.STAGE_1_ICP_COUNT * (CHALLENGERS + 1),
        "stage2_runs": contracts.STAGE_2_ICP_COUNT * (finalist_count + 1),
        "timings_seconds": timings,
    }
    (tmp_path / "scale_report.json").write_text(json.dumps(report, indent=2))
    print("\nSCALE REPORT " + json.dumps(report))

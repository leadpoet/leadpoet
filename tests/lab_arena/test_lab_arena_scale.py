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
from datetime import datetime, timezone

import pytest

from lab_arena import contracts

from tests.lab_arena.test_lab_arena_service_round import Harness, connect, database  # noqa: F401  (fixtures)

pytestmark = pytest.mark.skipif(os.environ.get("LAB_ARENA_SCALE_TEST") != "1", reason="set LAB_ARENA_SCALE_TEST=1 to run the challenger-cap round")

CHALLENGERS = int(os.environ.get("LAB_ARENA_SCALE_CHALLENGERS", str(contracts.MAX_CHALLENGERS)))
RUNNERS = ["alpha", "beta", "gamma"]
REPLAY = os.environ.get("LAB_ARENA_SCALE_REPLAY") == "1"  # replay every scoring in the stage assembly, as production does


def test_round_at_the_challenger_cap_scores_every_participant_and_publishes(connect, tmp_path):
    timings = {}

    def timed(label, call):
        started = time.monotonic()
        result = call()
        timings[label] = round(time.monotonic() - started, 2)
        return result

    import sys

    flavors = ["Scale-%03d" % index for index in range(CHALLENGERS)]
    harness = Harness(connect, tmp_path, challengers=flavors, runners=RUNNERS)
    harness.max_challengers = contracts.MAX_CHALLENGERS
    if REPLAY:
        from tests.lab_arena.test_lab_arena_service_round import REPLAY_SCRIPT

        script = tmp_path / "replay_entry.py"
        script.write_text(REPLAY_SCRIPT)
        harness.replay_command = [sys.executable, str(script)]
    harness.service = harness.build_service()
    service = harness.service
    harness.chain.epoch = 30500
    configuration = service.create_round(datetime(2026, 10, 20, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    timed("submissions", lambda: [harness.submit(flavor, round_id) for flavor in flavors])
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    freeze = timed("freeze_and_benchmark", lambda: service.advance_round(round_id))
    assert freeze["status"] == "ok", freeze
    participants = service.store.get_round(round_id)["participants"]
    assert len(participants) == CHALLENGERS
    for participant in participants:
        harness.flavors.setdefault(participant["image_digest"], "King")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = timed("stage1_open", lambda: service.advance_round(round_id))
    assert opened["assignments"] == 30 * CHALLENGERS
    timed("stage1_executions", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_close", lambda: service.advance_round(round_id))["status"] == "ok"
    assert timed("stage1_scoring_open", lambda: service.advance_round(round_id))["assignments"] == 30 * CHALLENGERS
    timed("stage1_scorings", lambda: harness.run_stage_with_runners(len(RUNNERS)))
    assert timed("stage1_scoring_close", lambda: service.advance_round(round_id))["status"] == "closed"
    assert timed("stage1_score_stage", lambda: service.advance_round(round_id))["status"] == "ok"
    assert harness.status() == "scored"
    timed("scored_to_published", lambda: harness.advance_until("published", runners=len(RUNNERS), max_steps=200))
    round_row = service.store.get_round(round_id)
    assert round_row["status"] == "published" and round_row["king_outcome"] in ("crowned", "defended")
    final = json.loads(harness.objects.get(round_row["final_scores_ref"]).decode())
    assert len(final["submission_scores"]) == CHALLENGERS and len(final["rows"]) == 30 * CHALLENGERS
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert timing["judge_executions"] == 30 * CHALLENGERS and timing["work_items"] == 30 * CHALLENGERS
    if REPLAY:
        # The replay is a post-publication report: drive it to completion and check every scoring reproduced.
        for _ in range(10_000):
            if timed("replay_chunk", lambda: service.replay_pending())["status"] == "reported":
                break
        report = json.loads(harness.objects.get("arena/%s/public/replay_report.json" % round_id).decode())
        assert report["replayed"] == 30 * CHALLENGERS and not report["flagged"], report["per_validator"]
    report = {"challengers": CHALLENGERS, "runs_stage1": 30 * CHALLENGERS, "replay": REPLAY, "timings_seconds": timings}
    (tmp_path / "scale_report.json").write_text(json.dumps(report, indent=2))
    print("\nSCALE REPORT " + json.dumps(report))

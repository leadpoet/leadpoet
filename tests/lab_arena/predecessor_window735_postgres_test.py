"""Exercise first dispatch and claim through the real database transition."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json

import pytest

from tests.lab_arena import prior_reward_baseline640_postgres_test as previous

database = previous.database
prepared = previous.prepared
prior = previous.prior
SQL = prior.prior.SCRIPTS / "735-lab-arena-predecessor-dispatch-window.sql"


@pytest.mark.parametrize(
    "enabled,late,expect_shift",
    [
        (True, True, True),
        (True, False, False),
        (False, True, False),
    ],
)
def test_dispatch_after_predecessor(prepared, enabled, late, expect_shift):
    p = prepared
    previous.apply(p)
    previous.pending_predecessor(p)
    config = deepcopy(p.store.get_round(prior.ROUND)["configuration_doc"])
    now = datetime.now(timezone.utc).replace(microsecond=0)
    start = now - timedelta(hours=20)
    keys = (
        "stage_1_start",
        "stage_1_close",
        "stage_1_scoring_close",
        "stage_2_start",
        "stage_2_close",
        "final_scoring_close",
        "publication_deadline",
    )
    schedule = config["schedule"]
    for key, hours in zip(
        keys,
        (0, 4, 10.5, 10.5 + 1 / 3600, 13.5 + 1 / 3600, 20 + 1 / 3600, 20 + 2 / 3600),
    ):
        schedule[key] = (start + timedelta(hours=hours)).isoformat()
    schedule["submission_open"] = (start - timedelta(days=1, minutes=30)).isoformat()
    schedule["submission_cutoff"] = (start - timedelta(minutes=30)).isoformat()
    schedule["benchmark_deadline"] = (start - timedelta(seconds=1)).isoformat()
    config["execution_sequence_policy"] = "baseline_scored_first_v1"
    config["stage_1_icp_count"] = config["stage_2_icp_count"] = 5
    if enabled:
        config["dispatch_window_policy"] = "late_predecessor_v1"
    prior.change(
        p,
        "UPDATE lab_arena_rounds SET status='committed',configuration_doc=%s::jsonb "
        "WHERE round_id=%s",
        (json.dumps(config), prior.ROUND),
    )
    assert not prior.current(p)
    published = now if late else start - timedelta(hours=1)
    prior.change(
        p,
        "UPDATE lab_arena_rounds SET status='published',published_at=%s,"
        "publication_doc=%s::jsonb WHERE round_id=%s",
        (
            published,
            json.dumps({"king_decision": {"outcome": "defended"}}),
            prior.prior.ROUND,
        ),
    )
    assert prior.current(p)
    with p.conn.cursor() as cur:
        cur.execute(SQL.read_text())
        cur.execute(SQL.read_text())
        cur.execute(
            "SELECT has_function_privilege('lab_arena_service', "
            "'public.lab_arena__predecessor_dispatch_window_v1()', 'EXECUTE')"
        )
        assert not cur.fetchone()[0]
    p.conn.commit()
    participants = [p.target_baseline]
    first = p.store.open_stage(prior.ROUND, 1, participants, list(range(10)))
    assert first["status"] == "ok"
    updated = p.store.get_round(prior.ROUND)["configuration_doc"]
    if expect_shift:

        def instant(s):
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        offsets = {
            instant(updated["schedule"][key]) - instant(schedule[key]) for key in keys
        }
        assert len(offsets) == 1 and offsets.pop() >= timedelta(hours=20)
        assert {**updated, "schedule": schedule} == config
        for key in set(schedule) - set(keys):
            assert updated["schedule"][key] == schedule[key]
        assert prior.claim(p)["status"] == "leased"
    else:
        assert updated == config
    assert (
        p.store.open_stage(prior.ROUND, 1, participants, list(range(10)))["status"]
        == "stale"
    )
    assert p.store.get_round(prior.ROUND)["configuration_doc"] == updated
    with p.conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM lab_arena_scoring_supersessions WHERE round_id=%s "
            "AND request_id='late-predecessor-dispatch-v1'",
            (prior.ROUND,),
        )
        assert cur.fetchone()[0] == int(expect_shift)

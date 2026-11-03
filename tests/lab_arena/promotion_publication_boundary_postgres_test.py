"""Frozen promotion margins through the real publication RPC."""

import json
import pytest

from tests.lab_arena import incomplete_participant_publication540_postgres_test as publication


@pytest.fixture(scope="module")
def database():
    fixture = publication.database.__wrapped__()
    try:
        yield next(fixture)
    finally:
        fixture.close()


@pytest.fixture(scope="module")
def completed(database, tmp_path_factory):
    fixture = publication.completed.__wrapped__(database, tmp_path_factory)
    try:
        yield next(fixture)
    finally:
        fixture.close()


@pytest.mark.parametrize(
    "margin,challenger_score,expected",
    [(8, 47.999, "no_king"), (8, 48, "crowned"), (0.5, 40.5, "crowned")],
)
def test_publication_uses_frozen_margin_at_exact_boundary(
    database, completed, margin, challenger_score, expected
):
    harness = publication.reset(database, completed)
    baseline = publication.selected(completed, "PublicBaseline")
    challenger = publication.selected(completed, "Complete")
    before = harness.service.store.get_round(harness.round_id)
    assert before["reward_basis_doc"] is None
    assert before["reward_basis_hash"] is None
    assert before["reward_activated_at"] is None
    with database[0].connect(**database[1]) as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{promotion_margin}',%s::jsonb) WHERE round_id=%s",
            (json.dumps(margin), harness.round_id),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=40 WHERE round_id=%s AND submission_id=%s AND kind='execute'",
            (harness.round_id, baseline["submission_id"]),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET per_icp_score=%s WHERE round_id=%s AND submission_id=%s AND kind='execute'",
            (challenger_score, harness.round_id, challenger["submission_id"]),
        )
        cursor.execute("SET LOCAL session_replication_role=origin")
    assert harness.service.publish(harness.round_id)["status"] == "ok"
    row = harness.service.store.get_round(harness.round_id)
    ranking = {item["submission_id"]: item for item in row["publication_doc"]["final_ranking"]}
    assert ranking[baseline["submission_id"]]["final_score"] == 40
    assert ranking[challenger["submission_id"]]["final_score"] == challenger_score
    assert ranking[challenger["submission_id"]]["eligible"] is True
    assert row["publication_doc"]["king_decision"]["outcome"] == expected
    assert row["configuration_doc"]["promotion_margin"] == margin
    assert row["reward_basis_doc"] is None
    assert row["reward_basis_hash"] is None
    assert row["reward_activated_at"] is None

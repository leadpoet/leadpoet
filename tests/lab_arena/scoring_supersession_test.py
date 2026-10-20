"""A model-policy supersession cannot reuse prior grading authority."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.completed_model_scores_test import completed


def replacement_judges(judges, *, status="pending", generation=7):
    return [{**judge, "run_id": judge["run_id"] + "-replacement", "attempt": 2,
             "stage_generation": generation, "scoring_generation_floor": generation, "status": status,
             "terminal_cause": "accepted" if status == "accepted" else None}
            for judge in judges]


def test_generation_floor_excludes_old_accepted_before_retry_ranking(completed):
    _, row, _, judges, _ = completed
    row["scoring_generation_floor"] = 7
    replacements = replacement_judges(judges)
    chosen = ArenaService._select_scoring_outputs(judges + replacements, round_row=row)
    assert set(chosen) == {judge["scored_run_id"] for judge in replacements}
    assert all(judge["status"] == "pending" for judge in chosen.values())
    # Ordinary rounds retain the original accepted-first retry policy.
    legacy = ArenaService._select_scoring_outputs(judges + replacements, round_row={})
    assert all(judge["status"] == "accepted" for judge in legacy.values())


def test_later_close_generation_does_not_discard_current_policy_acceptance(completed):
    _, row, _, judges, _ = completed
    row.update(scoring_generation_floor=7, stage_generation=9)
    accepted = replacement_judges(judges, status="accepted")
    failed = [{**judge, "attempt": 3, "stage_generation": 8, "status": "failed"}
              for judge in accepted]
    chosen = ArenaService._select_scoring_outputs(judges + accepted + failed, round_row=row)
    assert all(judge["stage_generation"] == 7 and judge["status"] == "accepted"
               for judge in chosen.values())


@pytest.mark.parametrize("round_status", ["stage2_scoring", "stage2_judged", "scored"])
@pytest.mark.parametrize("replacement", [None, "pending", "leased", "failed"])
def test_stored_old_projection_cannot_make_superseded_judgment_public(completed, round_status, replacement):
    service, row, _, judges, _ = completed
    row.update(scoring_generation_floor=7, status=round_status)
    if replacement is not None:
        judges.extend(replacement_judges(judges, status=replacement))
    assert service._read_completed_submission_scores(row) == {}
    service._submission_cost_eligibility.assert_not_called()


def test_current_accepted_judgments_still_require_cost_eligibility(completed):
    service, row, runs, judges, eligibility = completed
    row["scoring_generation_floor"] = 7
    judges.extend(replacement_judges(judges, status="accepted"))
    before = deepcopy(runs)
    assert service._read_completed_submission_scores(row)["miner"]["final_score"] == 60
    eligibility["eligible"] = False
    assert service._read_completed_submission_scores(row) == {}
    assert runs == before
    assert service._submission_cost_eligibility.call_count == 2


def test_direct_authority_check_rejects_old_image_identical_judgment(completed):
    service, row, runs, judges, _ = completed
    row["scoring_generation_floor"] = 7
    judges[0]["output_ref"] = "unchanged-old-image-output"
    assert not service._judgment_matches_execution(row, judges[0], runs[0])


@pytest.mark.parametrize("floor", [-1, True, "7", None])
def test_bad_generation_floor_fails_closed(floor):
    with pytest.raises(ServiceError, match="scoring_generation_floor_invalid"):
        ArenaService._score_generation_floor({"scoring_generation_floor": floor})


def test_public_evaluation_counts_only_current_judge_authority():
    from datetime import datetime,timezone
    from types import SimpleNamespace
    from lab_arena.public_dashboard import _attach_evaluations
    old={'run_id':'old','assignment_id':'a','submission_id':'miner','kind':'score',
         'status':'accepted','attempt':1,'stage_generation':2,'scoring_generation_floor':0}
    new={**old,'run_id':'new','status':'pending','attempt':2,'stage_generation':7,'scoring_generation_floor':7}
    service=SimpleNamespace(now=lambda:datetime.now(timezone.utc),
        _store=SimpleNamespace(list_runs=lambda *_a,**_k:[old,new],list_runtime_starts=lambda *_a,**_k:[]))
    entries=[{'submission_id':'miner','status':'scoring'}]
    _attach_evaluations(service,'round',entries,scoring_generation_floor=7)
    assert entries[0]['evaluation']['counts']['completed']==0


def test_model_slot_revision_rekeys_every_company_without_changing_saved_rows():
    from lab_arena import company_judgments,scoring
    from tests.lab_arena.company_judgments_test import _input
    document=_input()
    document['scorer_policy'].pop('company_quality_policy')
    document['scorer_policy']['company_judgment_policy']='company_judgment_v1'
    document['scorer_policy']['judge_models']['role_batch_check']='google/gemini-2.5-flash'
    revised=deepcopy(document)
    revised['scorer_policy']['judge_models']['role_batch_check']='google/gemini-3.5-flash-lite'
    revised['scorer_policy']['env_bindings']['ARENA_ROLE_BATCH_MODEL']='google/gemini-3.5-flash-lite'
    def refs(value):
        return company_judgments.build_company_scopes(scoring_input=value,round_id='round',
            network_name='finney',netuid=71,scorer_image_digest='sha256:'+'a'*64,
            scorer_image_reference='registry/scorer@sha256:'+'a'*64,
            integrity_policy='arena_integrity_v1',company_judgment_policy='company_judgment_v1')
    before,after=refs(document),refs(revised)
    assert len(before)==len(after)==len(document['companies'])
    assert not {r['cache_key'] for r in before}&{r['cache_key'] for r in after}
    assert document['companies']==revised['companies']
    assert document['icp']==revised['icp']
    assert document['evaluation_date']==revised['evaluation_date']


def test_stale_round_cannot_consume_future_policy_authority(completed):
    service,row,runs,judges,_=completed
    future=replacement_judges(judges,status='accepted',generation=8)
    for judge in judges:judge['status']='failed'
    selected=ArenaService._select_scoring_outputs(judges+future,round_row=row)
    assert all(judge['status']=='failed' for judge in selected.values())
    assert not service._judgment_matches_execution(row,future[0],runs[0])

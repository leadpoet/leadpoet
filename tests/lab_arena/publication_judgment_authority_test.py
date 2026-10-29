"""Publication recognizes each frozen judgment format without replaying work."""
import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, scoring, service
from tests.lab_arena.complete_baseline_final_authority_test import fixture
from tests.lab_arena.company_judgments_test import _company


def authority_fixture(format):
    instance, row, sources, judges, *_ = fixture()
    row["configuration_doc"].update(
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        scorer_policy=scoring.build_scorer_policy(),
    )
    scope = {"round_id": row["round_id"], **{
        key: row["configuration_doc"][key]
        for key in ("scorer_image_digest", "scorer_image_reference")}}
    if format != "legacy":
        row["configuration_doc"]["integrity_policy"] = "arena_integrity_v1"
    if format == "company":
        row["configuration_doc"]["company_quality_policy"] = "company_quality_v1"
        scope.update(company_input_hash="sha256:" + "b" * 64)
        key = contracts.document_hash(scope)
        for judge in judges:
            judge["company_judgment_refs"] = [{
                "schema_version": "leadpoet.lab_arena.company_judgment_ref.v1",
                "company_index": 0, "cache_key": key,
                "company_input_hash": scope["company_input_hash"],
                "scope_doc": {**scope, "cache_key": key},
            }]
    elif format == "integrity":
        for judge in judges:
            judge["judgment_scope_doc"] = copy.deepcopy(scope)
    instance._store.list_runs = lambda _rid, kind=None, **kw: [
        run for run in (sources if kind == "execute" else judges)
        if kw.get("stage", run["stage"]) == run["stage"]]
    return instance, row, sources, judges


@pytest.mark.parametrize("format", ["legacy", "integrity", "company"])
def test_completed_judgments_use_the_frozen_scorer_format(format):
    instance, row, sources, judges = authority_fixture(format)
    before = copy.deepcopy((sources, judges))
    assert not instance._unfinished_judgments(row)
    assert (sources, judges) == before


@pytest.mark.parametrize("format", ["legacy", "integrity", "company"])
@pytest.mark.parametrize("mutation", [
    "missing_score", "missing_judge", "failed_judge", "wrong_execution",
    "wrong_miner", "wrong_round", "wrong_stage", "missing_output",
])
def test_unfinished_or_unbound_judgments_still_block(format, mutation):
    instance, row, sources, judges = authority_fixture(format)
    if mutation == "missing_score":
        sources[0]["per_icp_score"] = None
    elif mutation == "missing_judge":
        judges.pop(0)
    elif mutation == "failed_judge":
        judges[0].update(status="failed", terminal_cause="judge_error")
    elif mutation == "wrong_execution":
        judges[0]["scored_run_id"] = "different-execution"
    elif mutation == "wrong_miner":
        judges[0]["miner_hotkey"] = "different-miner"
    elif mutation == "wrong_round":
        judges[0]["round_id"] = "different-round"
    elif mutation == "wrong_stage":
        judges[0]["stage"] = 2
    elif mutation == "missing_output":
        judges[0]["output_ref"] = None
    if mutation in {"wrong_miner", "wrong_round", "wrong_stage", "missing_output"}:
        with pytest.raises(service.ServiceError, match="^accepted_judgment_binding_invalid$"):
            instance._unfinished_judgments(row)
    else:
        assert instance._unfinished_judgments(row)


@pytest.mark.parametrize("format", ["integrity", "company"])
@pytest.mark.parametrize("mutation", ["missing_scope", "wrong_image", "wrong_round"])
def test_cache_formats_cannot_fall_back_to_legacy_authority(format, mutation):
    instance, row, _, judges = authority_fixture(format)
    if format == "integrity":
        scope = judges[0]["judgment_scope_doc"]
        if mutation == "missing_scope":
            judges[0].pop("judgment_scope_doc")
        elif mutation == "wrong_image":
            scope["scorer_image_digest"] = "sha256:" + "c" * 64
        else:
            scope["round_id"] = "different-round"
    else:
        if mutation == "missing_scope":
            judges[0].pop("company_judgment_refs")
        else:
            ref = judges[0]["company_judgment_refs"][0]
            scope = ref["scope_doc"]
            scope.pop("cache_key")
            scope["scorer_image_digest" if mutation == "wrong_image" else "round_id"] = "wrong"
            ref["cache_key"] = scope["cache_key"] = contracts.document_hash(scope)
    with pytest.raises(service.ServiceError, match="^accepted_judgment_binding_invalid$"):
        instance._unfinished_judgments(row)


@pytest.mark.parametrize("empty", [True, False])
def test_empty_company_refs_require_a_valid_empty_execution(empty):
    instance, row, sources, judges = authority_fixture("company")
    judges[0]["company_judgment_refs"] = []
    instance._objects = SimpleNamespace(get_bounded=lambda *_: json.dumps({
        "schema_version": service.contact_policy.output_schema(row["configuration_doc"]),
        "companies": [] if empty else [_company("A")],
    }).encode())
    if empty:
        assert not instance._unfinished_judgments(row)
    else:
        with pytest.raises(service.ServiceError, match="^accepted_judgment_binding_invalid$"):
            instance._unfinished_judgments(row)


@pytest.mark.parametrize("format", ["legacy", "integrity", "company"])
def test_recovered_score_writer_uses_the_same_authority(format, monkeypatch):
    instance, row, sources, judges = authority_fixture(format)
    sources[0]["per_icp_score"] = None
    writes = []
    verified = []
    instance.evaluation_icps = lambda *_: [{}, {}]
    instance._outputs_by_run = lambda *_: {source["run_id"]: [] for source in sources}
    instance._verified_breakdowns = lambda judge, **_kwargs: verified.append(judge["run_id"]) or []
    instance._store.record_run_scores = lambda *args, **kwargs: writes.append((args, kwargs)) or {"status": "ok"}
    monkeypatch.setattr(service.verify, "scored_row", lambda *args: {"run_id": args[2]})
    monkeypatch.setattr(scoring, "run_scores_for_store", lambda result, _: result["rows"])
    instance._record_recovered_judgment_scores(row)
    assert len(writes) == 1
    assert writes[0][0][2] == [{"run_id": sources[0]["run_id"]}]
    assert writes[0][1] == {"scoring_generation_floor": 0}
    assert verified == [judges[0]["run_id"]]
    assert all(source["per_icp_score"] == 0.0 for source in sources[1:])


def test_recovered_score_writer_rejects_missing_integrity_scope():
    instance, row, sources, judges = authority_fixture("integrity")
    sources[0]["per_icp_score"] = None
    judges[0].pop("judgment_scope_doc")
    instance.evaluation_icps = lambda *_: [{}, {}]
    instance._outputs_by_run = lambda *_: {}
    instance._store.record_run_scores = lambda *_, **_kwargs: pytest.fail("unbound score persisted")
    with pytest.raises(service.ServiceError, match="recovered_judgment_binding_invalid"):
        instance._record_recovered_judgment_scores(row)


@pytest.mark.parametrize("format", ["legacy", "integrity", "company"])
def test_recovered_score_writer_skips_ready_excluded_sources(format):
    instance, row, sources, judges = authority_fixture(format)
    sources[0]["per_icp_score"] = None
    excluded = sources[0]["submission_id"]
    row["participants"] = [p for p in row["participants"]
                           if p["submission_id"] != excluded]
    before = copy.deepcopy((sources, judges))
    instance.evaluation_icps = lambda *_: [{}, {}]
    instance._outputs_by_run = lambda *_: pytest.fail("excluded output read for materialization")
    instance._verified_breakdowns = lambda *_args, **_kwargs: pytest.fail("excluded judgment revalidated")
    instance._store.record_run_scores = lambda *_args, **_kwargs: pytest.fail("excluded score persisted")
    instance._record_recovered_judgment_scores(row)
    assert (sources, judges) == before


@pytest.mark.parametrize("format", ["legacy", "integrity", "company"])
def test_recovered_score_writer_keeps_admitted_recovery_with_excluded_ready_peer(format, monkeypatch):
    instance, row, sources, judges = authority_fixture(format)
    sources[0]["per_icp_score"] = None
    excluded = {**copy.deepcopy(sources[0]), "run_id": "excluded-source", "submission_id": "excluded"}
    sources.append(excluded)
    # This source is otherwise ready. Its invalid binding must not block the
    # admitted recovery or reach the write-once score RPC.
    judges.append({**copy.deepcopy(judges[0]), "run_id": "excluded-judge", "scored_run_id": excluded["run_id"],
                   "submission_id": excluded["submission_id"], "miner_hotkey": "wrong-excluded-binding"})
    writes = []
    verified = []
    instance.evaluation_icps = lambda *_: [{}, {}]
    instance._outputs_by_run = lambda *_: {sources[0]["run_id"]: []}
    instance._verified_breakdowns = lambda judge, **_kwargs: verified.append(judge["run_id"]) or []
    instance._store.record_run_scores = lambda *args, **kwargs: writes.append((args, kwargs)) or {"status": "ok"}
    monkeypatch.setattr(service.verify, "scored_row", lambda *args: {"run_id": args[2]})
    monkeypatch.setattr(scoring, "run_scores_for_store", lambda result, _: result["rows"])
    instance._record_recovered_judgment_scores(row)
    assert len(writes) == 1 and writes[0][0][2] == [{"run_id": sources[0]["run_id"]}]
    assert writes[0][1] == {"scoring_generation_floor": 0}
    assert verified == [judges[0]["run_id"]]
    assert excluded["per_icp_score"] is None

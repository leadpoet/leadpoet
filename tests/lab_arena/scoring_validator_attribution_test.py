"""Public scoring-validator attribution without private run identity disclosure."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lab_arena import company_judgments, contact_policy, contracts, integrity, judgment_cache, scoring
from lab_arena.api import create_app
from lab_arena.service import ArenaService, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError
from tests.lab_arena.judgment_cache_test import _icp


ROUND_ID = "arena-2026-09-20"
SUBMISSION_ID = "submission-public"
MINER = "5" + "M" * 47
VALIDATOR_A = "5" + "A" * 47
VALIDATOR_B = "5" + "B" * 47
VALIDATOR_C = "5" + "C" * 47


def _execution(run_id: str, position: int, *, submission_id=SUBMISSION_ID, status="accepted"):
    return {
        "run_id": run_id,
        "round_id": ROUND_ID,
        "submission_id": submission_id,
        "kind": "execute",
        "status": status,
        "stage": 1,
        "icp_position": position,
        "per_icp_score": 0.0 if status != "accepted" else 1.0,
        "output_ref": "",
        "result_doc": None,
    }


def _score(
    run_id: str,
    execution: dict,
    runner_hotkey: str | None,
    *,
    status="accepted",
    attempt=1,
    **extra,
):
    return {
        "run_id": run_id,
        "round_id": ROUND_ID,
        "submission_id": execution["submission_id"],
        "miner_hotkey": MINER,
        "kind": "score",
        "status": status,
        "stage": execution["stage"],
        "icp_position": execution["icp_position"],
        "scored_run_id": execution["run_id"],
        "runner_hotkey": runner_hotkey,
        "stage_generation": 1,
        "attempt": attempt,
        **extra,
    }


class Store:
    def __init__(self, rows, *, cache_rows=None):
        self.rows = {row["run_id"]: row for row in rows}
        self.cache_rows = cache_rows or {}

    def list_runs(self, round_id, **filters):
        assert round_id == ROUND_ID
        return [
            row
            for row in self.rows.values()
            if row["round_id"] == round_id
            and all(row.get(key) == value for key, value in filters.items())
        ]

    def get_run(self, run_id):
        return self.rows.get(run_id)

    def get_judgment_cache(self, cache_key):
        return self.cache_rows.get(cache_key)

    @staticmethod
    def get_submission(_submission_id):
        return {"is_king": False}


def _install_completed_scores_cache(service):
    """Give a bare ``ArenaService`` the completed-scores cache its reads need.

    These stubs are built with ``object.__new__``, so nothing ``__init__``
    installs is present. The published-results path reads the cache, and
    without it the test fails on a missing attribute rather than on the
    disclosure behaviour it is actually asserting.
    """
    service._completed_scores_lock = threading.Lock()
    service._completed_scores_cache = {}
    service._completed_scores_refresh_slots = threading.BoundedSemaphore(2)


def _service(rows, *, public_positions, cache_rows=None, status="published"):
    service = object.__new__(ArenaService)
    _install_completed_scores_cache(service)
    service._store = Store(rows, cache_rows=cache_rows)
    service._objects = SimpleNamespace(
        get_bounded=lambda *_args, **_kwargs: pytest.fail(
            "attribution must not read an execution output"
        )
    )
    service._round = lambda _round_id: {
        "round_id": ROUND_ID,
        "status": status,
        "configuration_doc": {},
        "publication_doc": {
            "participants": [{
                "submission_id": SUBMISSION_ID,
                "miner_hotkey": MINER,
                "is_baseline": False,
            }],
            "stage1_ranking": [],
            "final_ranking": [],
        },
    }
    service._public_icp_disclosure = lambda _row: (
        {"public_positions": list(public_positions)}
        if public_positions is not None
        else None
    )
    return service


def _cache(source_score, source_execution, source_runner):
    scope = {
        "schema_version": judgment_cache.CACHE_SCOPE_SCHEMA_VERSION,
        "scoring_input_hash": "sha256:" + "1" * 64,
    }
    cache_key = contracts.document_hash(scope)
    evidence = judgment_cache.build_evidence_snapshot(
        output={
            "schema_version": "leadpoet.lab_arena.scoring_output.v1",
            "scored_run_id": source_execution["run_id"],
            "breakdowns": [],
        },
        cache_scope={**scope, "cache_key": cache_key},
        source_score_run_id=source_score["run_id"],
        source_scored_run_id=source_execution["run_id"],
        source_output_ref="arena/private-score-output.json",
        source_runner_hotkey=source_runner,
        runner_authority_exclusions=[source_runner],
    )
    source_score.update({
        "judgment_cache_key": cache_key,
        "judgment_cache_source_run_id": source_score["run_id"],
    })
    return cache_key, {
        "cache_key": cache_key,
        "scoring_input_hash": evidence["scoring_input_hash"],
        "evidence_hash": contracts.document_hash(evidence),
        "evidence_doc": evidence,
        "source_score_run_id": source_score["run_id"],
        "source_scored_run_id": source_execution["run_id"],
        "source_runner_hotkey": source_runner,
    }


def test_public_results_attribute_fresh_judges_and_skip_execution_zeros():
    first = _execution("execute-0", 0)
    second = _execution("execute-1", 1)
    failed_execution = _execution("execute-2", 2, status="failed")
    service = _service(
        [
            first,
            second,
            failed_execution,
            _score("score-0", first, VALIDATOR_A),
            _score("score-1", second, VALIDATOR_B),
        ],
        public_positions={0, 1, 2},
    )

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]

    assert attribution == {
        "validators": [
            {"hotkey": VALIDATOR_A, "icp_count": 1, "reused_icp_count": 0},
            {"hotkey": VALIDATOR_B, "icp_count": 1, "reused_icp_count": 0},
        ],
        "icps": [
            {
                "icp_position": 0,
                "validator_hotkeys": [VALIDATOR_A],
                "reused_judgment": False,
            },
            {
                "icp_position": 1,
                "validator_hotkeys": [VALIDATOR_B],
                "reused_judgment": False,
            },
        ],
        "unattributed_icp_count": 0,
        "code_versions": [],
    }


def test_cached_follower_attributes_original_judge_with_blank_runner():
    original_execution = _execution(
        "execute-original", 0, submission_id="submission-original"
    )
    original_score = _score(
        "score-original", original_execution, VALIDATOR_A
    )
    follower_execution = _execution("execute-follower", 0)
    cache_key, cache_row = _cache(
        original_score, original_execution, VALIDATOR_A
    )
    follower_score = _score(
        "score-follower",
        follower_execution,
        None,
        judgment_cache_key=cache_key,
        judgment_cache_source_run_id=original_score["run_id"],
        result_doc={
            "schema_version": "leadpoet.lab_arena.cached_run_result.v1",
            "terminal_status": "accepted",
            "cache_key": cache_key,
            "source_score_run_id": original_score["run_id"],
        },
    )
    service = _service(
        [original_execution, original_score, follower_execution, follower_score],
        public_positions={0},
        cache_rows={cache_key: cache_row},
    )
    list_calls = []
    get_calls = []
    list_runs = service._store.list_runs
    get_run = service._store.get_run

    def counted_list_runs(round_id, **filters):
        list_calls.append(filters)
        return list_runs(round_id, **filters)

    def counted_get_run(run_id):
        get_calls.append(run_id)
        return get_run(run_id)

    service._store.list_runs = counted_list_runs
    service._store.get_run = counted_get_run

    assert service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ] == {
        "validators": [{
            "hotkey": VALIDATOR_A,
            "icp_count": 1,
            "reused_icp_count": 1,
        }],
        "icps": [{
            "icp_position": 0,
            "validator_hotkeys": [VALIDATOR_A],
            "reused_judgment": True,
        }],
        "unattributed_icp_count": 0,
        "code_versions": [],
    }
    assert list_calls == [
        {"kind": "execute", "submission_id": SUBMISSION_ID},
        {"kind": "score", "submission_id": SUBMISSION_ID},
    ]
    assert get_calls == [original_score["run_id"], original_execution["run_id"]]


def test_public_results_bulk_authorities_match_single_reads_and_hide_private_icp(monkeypatch):
    original_execution = _execution("execute-original", 0, submission_id="original")
    original_score = _score("score-original", original_execution, VALIDATOR_A)
    visible = _execution("execute-visible", 0)
    visible["output_ref"] = "visible-output"
    hidden = _execution("execute-hidden", 1)
    hidden["output_ref"] = "hidden-output"
    key, cache_row = _cache(original_score, original_execution, VALIDATOR_A)
    follower = _score(
        "score-follower", visible, None,
        judgment_cache_key=key,
        judgment_cache_source_run_id=original_score["run_id"],
        result_doc={
            "schema_version": "leadpoet.lab_arena.cached_run_result.v1",
            "terminal_status": "accepted", "cache_key": key,
            "source_score_run_id": original_score["run_id"],
        },
    )
    hidden_score = _score("score-hidden", hidden, VALIDATOR_B,
                          judgment_cache_key="private-cache-key",
                          judgment_cache_source_run_id="score-hidden")
    rows = [original_execution, original_score, visible, follower, hidden, hidden_score]

    def setup():
        service = _service(rows, public_positions={0}, cache_rows={key: cache_row})
        round_row = service._round(ROUND_ID)
        round_row["configuration_doc"] = {
            "contact_policy": contact_policy.POLICY,
            "scorer_policy": {"max_scored_companies": 5},
        }
        service._round = lambda _round_id: round_row
        service._benchmark_icps_from_row = lambda _round_id, _row: [_icp()]
        service._objects = SimpleNamespace(get_bounded=lambda ref, _limit: (
            pytest.fail("private output read") if ref != "visible-output"
            else json.dumps({"companies": []}).encode()
        ))
        return service

    monkeypatch.setattr("lab_arena.service.validate_output_document", lambda doc: doc)
    single = setup().public_results(ROUND_ID, SUBMISSION_ID)
    bulk_service = setup()
    calls = {"caches": [], "runs": []}
    store = bulk_service._store
    store.get_judgment_caches = lambda keys: (
        calls["caches"].append(keys) or
        {key: store.cache_rows[key] for key in keys if key in store.cache_rows}
    )
    store.get_runs = lambda ids: (
        calls["runs"].append(ids) or
        {run_id: store.rows[run_id] for run_id in ids if run_id in store.rows}
    )
    store.get_judgment_cache = lambda _key: pytest.fail("cache N+1 read")
    store.get_run = lambda _id: pytest.fail("run N+1 read")

    bulk = bulk_service.public_results(ROUND_ID, SUBMISSION_ID)
    assert bulk == single
    assert calls["caches"] == [[key]]
    assert calls["runs"] == [["score-original"], ["execute-original"]]
    assert "execute-hidden" not in json.dumps(bulk)
    assert "private-cache-key" not in json.dumps(bulk)

    for failure in ("cache", "run_first", "run_second"):
        broken = setup()
        broken._store.get_run = lambda _id: pytest.fail("bulk failure used single run read")
        broken._store.get_judgment_cache = lambda _key: pytest.fail("bulk failure used single cache read")
        if failure == "cache":
            broken._store.get_judgment_caches = lambda _keys: (
                _ for _ in ()
            ).throw(ArenaStoreError("bulk cache authority invalid"))
        else:
            broken._store.get_judgment_caches = lambda _keys: {key: cache_row}
            requests = []
            def fail_run_batch(ids):
                requests.append(ids)
                if failure == "run_first" or len(requests) == 2:
                    raise ArenaStoreError("bulk run authority invalid")
                return {run_id: broken._store.rows[run_id] for run_id in ids}
            broken._store.get_runs = fail_run_batch
        with TestClient(create_app(broken)) as http:
            response = http.get(f"/arena/v1/rounds/{ROUND_ID}/results/{SUBMISSION_ID}")
        assert response.status_code == 503
        assert response.json()["code"] == "public_result_unavailable"


def test_active_completed_public_results_bulk_response_matches_single_reads():
    source_execution = _execution("execute-completed-source", 0, submission_id="original")
    source_score = _score("score-completed-source", source_execution, VALIDATOR_A)
    execution = _execution("execute-completed", 0)
    key, cache_row = _cache(source_score, source_execution, VALIDATOR_A)
    follower = _score(
        "score-completed", execution, None,
        judgment_cache_key=key,
        judgment_cache_source_run_id=source_score["run_id"],
        result_doc={
            "schema_version": "leadpoet.lab_arena.cached_run_result.v1",
            "terminal_status": "accepted", "cache_key": key,
            "source_score_run_id": source_score["run_id"],
        },
    )
    rows = [source_execution, source_score, execution, follower]

    def setup():
        service = _service(rows, public_positions={0}, cache_rows={key: cache_row},
                           status="stage1_scored")
        round_row = service._round(ROUND_ID)
        round_row["participants"] = round_row["publication_doc"]["participants"]
        service._round = lambda _round_id: round_row
        service.completed_submission_scores = lambda _row: {
            SUBMISSION_ID: {
                "execution_runs": [{**execution, "per_icp_score": 4.25}],
                "final_score": 4.25,
            }
        }
        return service

    expected = setup().public_results(ROUND_ID, SUBMISSION_ID)
    bulk = setup()
    bulk._store.get_runs = lambda ids: {
        run_id: bulk._store.rows[run_id] for run_id in ids
    }
    bulk._store.get_run = lambda _id: pytest.fail("active bulk response used single read")
    result = bulk.public_results(ROUND_ID, SUBMISSION_ID)
    assert result == expected
    assert result["score_status"] == "complete"
    assert result["submission_scores"]["final"] == 4.25
    assert result["scores"]["stage_1"][0]["per_icp_score"] == 4.25


@pytest.mark.parametrize("damage", ["missing", "invalid_hash", "wrong_source"])
def test_bulk_cache_authority_still_fails_closed(damage):
    execution = _execution("execute-cache", 0)
    score = _score("score-cache", execution, VALIDATOR_A)
    key, cache_row = _cache(score, execution, VALIDATOR_A)
    service = _service([execution, score], public_positions={0}, cache_rows={key: cache_row})
    cached = dict(cache_row)
    if damage == "missing":
        cached = None
    elif damage == "invalid_hash":
        cached["evidence_hash"] = "sha256:" + "0" * 64
    else:
        cached["source_score_run_id"] = "foreign-score"
    with pytest.raises(scoring.ScoringError):
        service._verified_breakdowns(
            score, icp=_icp(), companies=[], policy={"max_scored_companies": 5},
            cached_rows={key: cached}, source_rows={score["run_id"]: score},
        )


def test_exact_bulk_store_reads_are_bounded_and_reject_duplicate_authority():
    class Transport:
        def __init__(self):
            self.calls = []
            self.duplicate = False

        def select(self, table, **kwargs):
            self.calls.append((table, kwargs))
            ids = kwargs.get("run_ids") or kwargs.get("cache_keys")
            column = "run_id" if table == "lab_arena_runs" else "cache_key"
            rows = [{column: value} for value in ids]
            return rows + rows[:1] if self.duplicate else rows

    transport = Transport()
    store = ArenaStore(transport)
    ids = ["source-%02d" % index for index in range(26)]
    assert set(store.get_runs(ids)) == set(ids)
    assert set(store.get_judgment_caches(ids)) == set(ids)
    assert [len(call[1].get("run_ids") or call[1].get("cache_keys"))
            for call in transport.calls] == [25, 1, 25, 1]
    assert all(call[1]["limit"] == 50 for call in transport.calls)

    transport.duplicate = True
    with pytest.raises(ArenaStoreError, match="bulk read is invalid"):
        store.get_runs(["source-0"])
    with pytest.raises(ArenaStoreError, match="bulk read is invalid"):
        store.get_judgment_caches(["cache-0"])
    transport.duplicate = False
    original_select = transport.select
    transport.select = lambda table, **kwargs: [{
        "run_id" if table == "lab_arena_runs" else "cache_key": "foreign"
    }]
    with pytest.raises(ArenaStoreError, match="bulk read is invalid"):
        store.get_runs(["source-0"])
    with pytest.raises(ArenaStoreError, match="bulk read is invalid"):
        store.get_judgment_caches(["cache-0"])
    transport.select = original_select


def test_self_referenced_cache_is_not_reported_as_reused():
    execution = _execution("execute-self", 0)
    score = _score("score-self", execution, VALIDATOR_A)
    cache_key, cache_row = _cache(score, execution, VALIDATOR_A)
    score.update({
        "judgment_cache_key": cache_key,
        "judgment_cache_source_run_id": score["run_id"],
    })
    service = _service(
        [execution, score],
        public_positions={0},
        cache_rows={cache_key: cache_row},
    )

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]
    assert attribution["validators"][0]["reused_icp_count"] == 0
    assert attribution["icps"][0]["reused_judgment"] is False


def test_accepted_judge_wins_over_later_failed_retry():
    execution = _execution("execute-retry", 0)
    accepted = _score("score-accepted", execution, VALIDATOR_A, attempt=1)
    failed = _score(
        "score-failed", execution, VALIDATOR_B, status="failed", attempt=2
    )
    service = _service(
        [execution, accepted, failed], public_positions={0}
    )

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]
    assert attribution["validators"] == [{
        "hotkey": VALIDATOR_A,
        "icp_count": 1,
        "reused_icp_count": 0,
    }]


def test_aggregate_is_public_after_publication_before_icp_disclosure():
    execution = _execution("execute-private-icp", 0)
    service = _service(
        [execution, _score("score-private-icp", execution, VALIDATOR_A)],
        public_positions=None,
    )

    result = service.public_results(ROUND_ID, SUBMISSION_ID)

    assert result["public_icp_status"] == "pending"
    assert result["scoring_attribution"]["validators"] == [{
        "hotkey": VALIDATOR_A,
        "icp_count": 1,
        "reused_icp_count": 0,
    }]
    assert result["scoring_attribution"]["icps"] == []


def test_attribution_is_private_before_round_publication():
    execution = _execution("execute-unpublished", 0)
    service = _service(
        [execution, _score("score-unpublished", execution, VALIDATOR_A)],
        public_positions={0},
        status="stage1_scored",
    )
    service._store.list_runs = lambda *_args, **_kwargs: pytest.fail(
        "unpublished attribution must not query runs"
    )

    with pytest.raises(ServiceError, match="results_not_public"):
        service.public_results(ROUND_ID, SUBMISSION_ID)


def test_published_results_scope_both_run_reads_to_one_of_143_submissions():
    rows = []
    for participant in range(143):
        submission_id = SUBMISSION_ID if participant == 0 else "submission-%d" % participant
        for position in range(10):
            execution = _execution(
                "execute-%d-%d" % (participant, position),
                position,
                submission_id=submission_id,
            )
            rows.extend([
                execution,
                _score("score-%d-%d" % (participant, position), execution, VALIDATOR_A),
            ])
    service = _service(rows, public_positions=None)
    calls = []
    list_runs = service._store.list_runs

    def counted_list_runs(round_id, **filters):
        calls.append(filters)
        return list_runs(round_id, **filters)

    service._store.list_runs = counted_list_runs
    service._store.get_run = lambda run_id: pytest.fail("local judgments need no cross-submission lookup")

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]

    assert calls == [
        {"kind": "execute", "submission_id": SUBMISSION_ID},
        {"kind": "score", "submission_id": SUBMISSION_ID},
    ]
    assert attribution["validators"] == [{
        "hotkey": VALIDATOR_A,
        "icp_count": 10,
        "reused_icp_count": 0,
    }]
    assert attribution["icps"] == []


def _policy_response(monkeypatch, contacts_required):
    visible = _execution("execute-visible", 0)
    visible["output_ref"] = "visible-output"
    private = _execution("execute-private", 1)
    private["output_ref"] = "private-output"
    other = _execution("execute-other", 0, submission_id="submission-other")
    other["output_ref"] = "other-output"
    service = _service(
        [visible, private, other,
         _score("score-visible", visible, VALIDATOR_A),
         _score("score-private", private, VALIDATOR_B),
         _score("score-other", other, VALIDATOR_C)],
        public_positions={0},
    )
    round_row = service._round(ROUND_ID)
    round_row["configuration_doc"] = {
        "integrity_policy": integrity.POLICY,
        "scorer_policy": {"max_scored_companies": 5},
        **({"contact_policy": contact_policy.POLICY} if contacts_required else {}),
    }
    service._round = lambda _round_id: round_row
    output_reads = []

    def read_output(ref, _limit):
        output_reads.append(ref)
        return json.dumps({"schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                           "companies": [{"company_name": "Public Company"}]}).encode()

    service._objects = SimpleNamespace(get_bounded=read_output)
    service._benchmark_icps_from_row = lambda _round_id, _row: [{}]
    service._verified_breakdowns = lambda *_args, **_kwargs: [{
        "company_index": 0,
        "company_qualified": True,
        "contact_qualified": True,
        "contact_identity_key": "public-identity",
        "email_status": "verified",
        "contact_verification": {"decision": "verified"},
    }]
    monkeypatch.setattr("lab_arena.service.validate_output_document", lambda document: document)
    list_calls = []
    list_runs = service._store.list_runs

    def counted_list_runs(round_id, **filters):
        list_calls.append(filters)
        return list_runs(round_id, **filters)

    service._store.list_runs = counted_list_runs
    with TestClient(create_app(service)) as http:
        response = http.get(f"/arena/v1/rounds/{ROUND_ID}/results/{SUBMISSION_ID}")

    assert response.status_code == 200
    return response.json(), list_calls, output_reads


@pytest.mark.parametrize("contacts_required", [False, True])
def test_policy_results_read_only_one_submission_and_keep_public_diagnostics(
    monkeypatch, contacts_required,
):
    result, list_calls, output_reads = _policy_response(monkeypatch, contacts_required)
    assert list_calls == [
        {"kind": "execute", "submission_id": SUBMISSION_ID},
        {"kind": "score", "submission_id": SUBMISSION_ID},
    ]
    assert output_reads == ["visible-output"]
    assert set(result["outputs"]) == {"execute-visible"}
    assert result["company_diagnostics"][0]["company_name"] == "Public Company"
    assert result["company_diagnostics"][0]["qualified"] is True
    assert ("contact_verifications" in result) is contacts_required
    if contacts_required:
        assert result["contact_verifications"]["execute-visible"][0]["contact_qualified"] is True


def test_broken_cache_source_is_unattributed_without_breaking_results():
    source_execution = _execution(
        "execute-missing-source", 0, submission_id="submission-original"
    )
    source_score = _score("score-missing-source", source_execution, VALIDATOR_A)
    execution = _execution("execute-broken-cache", 0)
    cache_key, cache_row = _cache(
        source_score, source_execution, VALIDATOR_A
    )
    follower = _score(
        "score-broken-cache",
        execution,
        None,
        judgment_cache_key=cache_key,
        judgment_cache_source_run_id=source_score["run_id"],
        result_doc={
            "schema_version": "leadpoet.lab_arena.cached_run_result.v1",
            "terminal_status": "accepted",
            "cache_key": cache_key,
            "source_score_run_id": source_score["run_id"],
        },
    )
    service = _service(
        [execution, follower],
        public_positions={0},
        cache_rows={cache_key: cache_row},
    )

    result = service.public_results(ROUND_ID, SUBMISSION_ID)

    assert result["scoring_attribution"] == {
        "validators": [],
        "icps": [{
            "icp_position": 0,
            "validator_hotkeys": [],
            "reused_judgment": False,
        }],
        "unattributed_icp_count": 1,
        "code_versions": [],
    }
    service._store.get_runs = lambda _ids: {}
    service._store.get_run = lambda _id: pytest.fail("missing source must stay missing")
    assert service.public_results(ROUND_ID, SUBMISSION_ID)["scoring_attribution"] == result["scoring_attribution"]


def test_cross_round_cache_source_cannot_supply_public_attribution():
    source_execution = _execution("execute-other-round", 0, submission_id="other")
    source_score = _score("score-other-round", source_execution, VALIDATOR_A)
    source_execution["round_id"] = "arena-other-round"
    source_score["round_id"] = "arena-other-round"
    follower_execution = _execution("execute-follower", 0)
    cache_key, cache_row = _cache(source_score, source_execution, VALIDATOR_A)
    follower_score = _score(
        "score-follower", follower_execution, None,
        judgment_cache_key=cache_key,
        judgment_cache_source_run_id=source_score["run_id"],
        result_doc={
            "schema_version": "leadpoet.lab_arena.cached_run_result.v1",
            "terminal_status": "accepted",
            "cache_key": cache_key,
            "source_score_run_id": source_score["run_id"],
        },
    )
    service = _service(
        [source_execution, source_score, follower_execution, follower_score],
        public_positions={0}, cache_rows={cache_key: cache_row},
    )

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)["scoring_attribution"]

    assert attribution["validators"] == []
    assert attribution["unattributed_icp_count"] == 1
    service._store.get_runs = lambda ids: {
        run_id: service._store.rows[run_id]
        for run_id in ids if run_id in service._store.rows
    }
    service._store.get_run = lambda _id: pytest.fail("foreign source N+1 read")
    assert service.public_results(ROUND_ID, SUBMISSION_ID)["scoring_attribution"] == attribution


def _company_ref(index):
    company_input_hash = contracts.document_hash({"company": index})
    scope = {"company_input_hash": company_input_hash, "company": index}
    cache_key = contracts.document_hash(scope)
    return {
        "schema_version": company_judgments.COMPANY_REF_SCHEMA_VERSION,
        "company_index": index,
        "cache_key": cache_key,
        "company_input_hash": company_input_hash,
        "scope_doc": {**scope, "cache_key": cache_key},
    }


def test_company_cache_reports_each_authority_once_per_icp(monkeypatch):
    execution = _execution("execute-company", 0)
    current_score = _score("score-company", execution, VALIDATOR_A)
    source_execution = _execution(
        "execute-company-source", 0, submission_id="submission-original"
    )
    source_score = _score(
        "score-company-source", source_execution, VALIDATOR_B
    )
    refs = [_company_ref(0), _company_ref(1), _company_ref(2)]
    current_score.update({
        "company_judgment_refs": refs,
        "claim_response": {"company_judgment_cache": {}},
    })
    service = _service(
        [execution, current_score, source_execution, source_score],
        public_positions={0},
    )
    lease = {
        "hits": [],
        "misses": [
            {
                "company_index": index,
                "cache_key": ref["cache_key"],
                "company_input_hash": ref["company_input_hash"],
                "authority_slot": 0,
            }
            for index, ref in enumerate(refs)
        ],
    }
    evidence_by_key = {
        refs[0]["cache_key"]: {
            "source_score_run_id": current_score["run_id"],
            "source_scored_run_id": execution["run_id"],
            "source_runner_hotkey": VALIDATOR_A,
        },
        refs[1]["cache_key"]: {
            "source_score_run_id": source_score["run_id"],
            "source_scored_run_id": source_execution["run_id"],
            "source_runner_hotkey": VALIDATOR_B,
        },
        refs[2]["cache_key"]: {
            "source_score_run_id": source_score["run_id"],
            "source_scored_run_id": source_execution["run_id"],
            "source_runner_hotkey": VALIDATOR_B,
        },
    }
    lease["hits"] = [
        {**lease["misses"][1], "evidence_doc": evidence_by_key[refs[1]["cache_key"]]},
        {**lease["misses"][2], "evidence_doc": evidence_by_key[refs[2]["cache_key"]]},
    ]
    lease["misses"] = lease["misses"][:1]
    monkeypatch.setattr(
        company_judgments,
        "validate_lease_context",
        lambda _document: lease,
    )

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]

    assert attribution["validators"] == [
        {"hotkey": VALIDATOR_A, "icp_count": 1, "reused_icp_count": 0},
        {"hotkey": VALIDATOR_B, "icp_count": 1, "reused_icp_count": 1},
    ]
    assert attribution["icps"] == [{
        "icp_position": 0,
        "validator_hotkeys": [VALIDATOR_A, VALIDATOR_B],
        "reused_judgment": True,
    }]
    execution["output_ref"] = "company-output"
    round_row = service._round(ROUND_ID)
    round_row["configuration_doc"] = {
        "integrity_policy": integrity.POLICY,
        "scorer_policy": {"max_scored_companies": 5},
    }
    service._round = lambda _round_id: round_row
    service._benchmark_icps_from_row = lambda _round_id, _row: [_icp()]
    service._objects = SimpleNamespace(
        get_bounded=lambda ref, _limit: (
            json.dumps({"companies": []}).encode()
            if ref == "company-output" else pytest.fail("unexpected output read")
        )
    )
    monkeypatch.setattr("lab_arena.service.validate_output_document", lambda doc: doc)
    service._verified_breakdowns = lambda *_args, **_kwargs: []
    full_single = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert full_single["company_diagnostics"] == []
    requests = []
    service._store.get_runs = lambda ids: (
        requests.append(ids) or {
            run_id: service._store.rows[run_id] for run_id in ids
            if run_id in service._store.rows
        }
    )
    service._store.get_run = lambda _id: pytest.fail("company source N+1 read")
    full_bulk = service.public_results(ROUND_ID, SUBMISSION_ID)
    assert full_bulk == full_single
    assert full_bulk["scoring_attribution"] == attribution
    assert requests == [["score-company-source"], ["execute-company-source"]]


def test_public_code_versions_bind_to_actual_judge_and_disclosed_icps():
    first, private = _execution("execute-visible", 0), _execution("execute-private", 1)
    service = _service([first, private,
        _score("score-visible", first, VALIDATOR_A, result_doc={"resource_summary": {
            "validator_source_commit": "a" * 40, "validator_source_dirty": "dirty"}}),
        _score("score-private", private, VALIDATOR_B, result_doc={"resource_summary": {
            "validator_source_commit": "b" * 40, "validator_source_dirty": "clean"}})],
        public_positions={0})
    assert service.public_results(ROUND_ID, SUBMISSION_ID)["scoring_attribution"]["code_versions"] == [{
        "validator_hotkey": VALIDATOR_A, "commit": "a" * 40,
        "working_tree": "dirty", "icp_positions": [0]}]


def test_cached_code_version_is_original_judge_not_follower_or_current_checkout():
    original = _execution("execute-original", 0, submission_id="original")
    original_score = _score("score-original", original, VALIDATOR_A,
        result_doc={"resource_summary": {"validator_source_commit": "a" * 40}})
    follower = _execution("execute-follower", 0)
    cache_key, cache_row = _cache(original_score, original, VALIDATOR_A)
    cached = _score("score-follower", follower, VALIDATOR_B,
        judgment_cache_key=cache_key, judgment_cache_source_run_id=original_score["run_id"],
        result_doc={"schema_version": "leadpoet.lab_arena.cached_run_result.v1", "terminal_status": "accepted",
                    "cache_key": cache_key, "source_score_run_id": original_score["run_id"],
                    "resource_summary": {"validator_source_commit": "b" * 40}})
    service = _service([original, original_score, follower, cached], public_positions={0}, cache_rows={cache_key: cache_row})
    assert service.public_results(ROUND_ID, SUBMISSION_ID)["scoring_attribution"]["code_versions"] == [{
        "validator_hotkey": VALIDATOR_A, "commit": "a" * 40,
        "working_tree": "unknown", "icp_positions": [0]}]


@pytest.mark.parametrize("resource", [None, "bad", {}, {"validator_source_commit": "unknown"}, {"validator_source_commit": "https://bad"}])
def test_missing_or_invalid_code_metadata_does_not_invent_a_revision(resource):
    execution = _execution("execute-one", 0)
    score = _score("score-one", execution, VALIDATOR_A, result_doc={"resource_summary": resource})
    result = _service([execution, score], public_positions={0}).public_results(ROUND_ID, SUBMISSION_ID)
    assert result["scoring_attribution"]["validators"][0]["hotkey"] == VALIDATOR_A
    assert result["scoring_attribution"]["code_versions"] == []

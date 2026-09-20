"""Public scoring-validator attribution without private run identity disclosure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import company_judgments, contracts, judgment_cache
from lab_arena.service import ArenaService, ServiceError


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


def _service(rows, *, public_positions, cache_rows=None, status="published"):
    service = object.__new__(ArenaService)
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
    }


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


def test_live_shape_uses_two_bulk_run_reads_for_98_judgments():
    rows = []
    for index in range(98):
        submission_id = SUBMISSION_ID if index < 20 else "submission-%d" % index
        execution = _execution(
            "execute-bulk-%d" % index,
            index if index < 20 else index % 20,
            submission_id=submission_id,
        )
        rows.extend([
            execution,
            _score("score-bulk-%d" % index, execution, VALIDATOR_A),
        ])
    service = _service(rows, public_positions=None)
    calls = []
    list_runs = service._store.list_runs

    def counted_list_runs(round_id, **filters):
        calls.append(filters)
        return list_runs(round_id, **filters)

    service._store.list_runs = counted_list_runs

    attribution = service.public_results(ROUND_ID, SUBMISSION_ID)[
        "scoring_attribution"
    ]

    assert calls == [{"kind": "execute"}, {"kind": "score"}]
    assert attribution["validators"] == [{
        "hotkey": VALIDATOR_A,
        "icp_count": 20,
        "reused_icp_count": 0,
    }]
    assert attribution["icps"] == []


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
    }


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

from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from lab_arena import contracts, judgment_cache, scoring
from lab_arena.service import ArenaService


def _icp() -> dict:
    return {"icp_id": "i-1", "prompt": "Find buyers", "industry": "Software",
            "employee_count": ["51-200"], "intent_signals": ["Product launch"]}


def _company(name: str) -> dict:
    return {
        "company_name": name, "company_website": "https://acme.com",
        "industry": "Software", "employee_count": "51-200", "country": "United States",
        "fit_summary": "Software company", "fit_evidence_urls": ["https://acme.com/about"],
        "intent_signals": [
            {"matched_icp_signal": 0, "description": "Product launch",
             "url": "https://acme.com/" + path, "date": None,
             "snippet": "A new product", "why_now": "Recent launch"}
            for path in ("first", "second")
        ],
    }


def _input(scored_run_id: str = "run-a") -> dict:
    return scoring.build_scoring_input(
        scored_run_id=scored_run_id,
        icp=_icp(),
        companies=[_company("A"), _company("B")],
        policy=scoring.build_scorer_policy(
            scoring_adapter_version="qualification_integrity_v2"
        ),
        evaluation_date="2026-09-11",
    )


def _scope(document: dict) -> dict:
    return judgment_cache.build_cache_scope(
        scoring_input=document,
        round_id="arena-2026-09-11",
        network_name="finney",
        netuid=401,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
    )


def test_cache_identity_normalizes_inputs_and_preserves_meaningful_order():
    first = _scope(_input("run-a"))
    assert _scope(_input("run-b"))["cache_key"] == first["cache_key"]

    reordered = _input("run-c")
    reordered["companies"] = list(reversed(reordered["companies"]))
    assert _scope(reordered)["cache_key"] != first["cache_key"]

    reordered_claims = _input("run-d")
    reordered_claims["companies"][0]["intent_signals"].reverse()
    assert _scope(reordered_claims)["cache_key"] != first["cache_key"]

    changed_policy = _input("run-e")
    changed_policy["scorer_policy"]["max_scored_companies"] -= 1
    assert _scope(changed_policy)["cache_key"] != first["cache_key"]


def test_cache_identity_refuses_silent_scoring_input_contract_drift():
    document = _input()
    document["submission_id"] = "untrusted-metadata"
    with pytest.raises(judgment_cache.JudgmentCacheError, match="fields"):
        _scope(document)

    reordered_fields = {key: document for key, document in reversed(list(_input().items()))}
    with pytest.raises(judgment_cache.JudgmentCacheError, match="field order"):
        _scope(reordered_fields)


def test_unused_prose_duplicates_and_padding_cannot_buy_new_judgments():
    document = _input()
    document["companies"][0]["fit_summary"] = "x" * 500
    original = _scope(document)["cache_key"]
    for index in range(100):
        candidate = copy.deepcopy(document)
        row = candidate["companies"][0]
        row["fit_summary"] += "ignored suffix %d" % index
        row["intent_signals"][0]["why_now"] = "unused prose %d" % index
        row["intent_signals"].append(copy.deepcopy(row["intent_signals"][0]))
        assert _scope(candidate)["cache_key"] == original

    document["companies"] = [_company(str(index)) for index in range(5)]
    original = _scope(document)["cache_key"]
    document["companies"].append(_company("padding"))
    assert _scope(document)["cache_key"] == original


@pytest.mark.parametrize("field,value", [
    ("url", "https://acme.com/different"), ("description", "A different event"),
    ("date", "2026-09-01"), ("snippet", "Different evidence"),
    ("matched_icp_signal", 1),
])
def test_meaningful_claim_changes_still_require_a_new_judgment(field, value):
    document = _input()
    original = _scope(document)["cache_key"]
    document["companies"][0]["intent_signals"][0][field] = value
    assert _scope(document)["cache_key"] != original


def test_evidence_beyond_criterion_cap_is_ignored_by_cache_and_adapter():
    from qualification.scoring.competition import _normalized_company

    document = _input()
    signals = document["companies"][0]["intent_signals"]
    signals.append({**signals[0], "url": "https://acme.com/third"})
    original = _scope(document)["cache_key"]
    effective = _normalized_company(document["companies"][0], integrity_policy=True)
    signals.extend({**signals[0], "url": "https://acme.com/noise-%d" % i} for i in range(50))
    assert _scope(document)["cache_key"] == original
    assert _normalized_company(document["companies"][0], integrity_policy=True) == effective


def test_authority_partition_is_stable_and_never_reuses_the_blocked_key():
    scope = _scope(_input())
    first = judgment_cache.partition_cache_scope(
        scope, incompatible_hotkeys=["validator-a", "miner-a", "miner-a"]
    )
    reordered = judgment_cache.partition_cache_scope(
        scope, incompatible_hotkeys=["miner-a", "validator-a"]
    )
    assert first == reordered
    assert first["cache_key"] != scope["cache_key"]
    with pytest.raises(judgment_cache.JudgmentCacheError, match="loop"):
        judgment_cache.partition_cache_scope(
            first, incompatible_hotkeys=["validator-a", "miner-a"]
        )


def test_open_scoring_partitions_a_cache_that_is_not_valid_for_recipient():
    round_id = "arena-2026-09-11"
    submission_id = "sub-a"
    miner_hotkey = "miner-a"
    policy = scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2"
    )
    icp = _icp()
    output = {
        "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "companies": [],
    }
    scoring_input = scoring.build_scoring_input(
        scored_run_id="execute-a",
        icp=icp,
        companies=[],
        policy=policy,
        evaluation_date="2026-09-11",
    )
    scope = judgment_cache.build_cache_scope(
        scoring_input=scoring_input,
        round_id=round_id,
        network_name="finney",
        netuid=401,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry/scorer@sha256:" + "a" * 64,
        integrity_policy="arena_integrity_v1",
    )
    cached_output = scoring.build_scoring_output("execute-source", [])
    evidence = judgment_cache.build_evidence_snapshot(
        output=cached_output,
        cache_scope=scope,
        source_score_run_id="score-source",
        source_scored_run_id="execute-source",
        source_output_ref="arena/source.json",
        source_runner_hotkey="validator-a",
        runner_authority_exclusions=["miner-a", "validator-a"],
    )
    cached = {
        "cache_key": scope["cache_key"],
        "evidence_doc": evidence,
        "evidence_hash": contracts.document_hash(evidence),
        "source_runner_hotkey": "validator-a",
    }
    captured = {}

    class Store:
        @staticmethod
        def list_runs(*args, **kwargs):
            return [{
                "run_id": "execute-a", "submission_id": submission_id,
                "icp_position": 0, "output_ref": "arena/output.json",
                "status": "accepted",
            }]

        @staticmethod
        def get_judgment_cache(cache_key):
            return cached if cache_key == scope["cache_key"] else None

        @staticmethod
        def open_scoring(_round_id, _stage, items, *, integrity_cache):
            captured["items"] = copy.deepcopy(items)
            return {"status": "ok", "round_status": "stage1_scoring", "assignments": 1}

    row = {
        "round_id": round_id,
        "status": "stage1_closed",
        "evaluation_date": "2026-09-11",
        "arena_network_name": "finney",
        "arena_netuid": 401,
        "participants": [{
            "submission_id": submission_id,
            "miner_hotkey": miner_hotkey,
            "is_king": False,
        }],
        "configuration_doc": {
            "integrity_policy": "arena_integrity_v1",
            "scorer_policy": policy,
            "scorer_image_digest": "sha256:" + "a" * 64,
            "scorer_image_reference": "registry/scorer@sha256:" + "a" * 64,
        },
    }
    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = SimpleNamespace(
        get_bounded=lambda _ref, _limit: json.dumps(output).encode("utf-8")
    )
    service._config = SimpleNamespace(chain=SimpleNamespace(
        hotkeys_owned_by_same_coldkey=lambda _hotkey: pytest.fail(
            "cache reuse consulted mutable chain ownership"
        )
    ))
    service._round = lambda _round_id: row
    service._load_scoring_plan = lambda _row, _stage: {
        "work_items": [{
            "scored_run_id": "execute-a", "submission_id": submission_id,
            "icp_position": 0, "output_ref": "arena/output.json",
        }]
    }
    service._require_code_review = lambda *_args: None
    service.evaluation_icps = lambda _round_id: [icp]

    service.open_scoring(round_id, 1)
    item = captured["items"][0]
    assert item["judgment_cache_key"] != scope["cache_key"]
    assert "reuse_cache_key" not in item
    assert item["judgment_group_leader"] is True


def test_frozen_evidence_is_hash_bound_and_cannot_name_another_execution():
    scope = _scope(_input())
    output = scoring.build_scoring_output(
        "run-a", [{"company_name": "A", "final_score": 80.0}]
    )
    snapshot = judgment_cache.build_evidence_snapshot(
        output=output,
        cache_scope=scope,
        source_score_run_id="score-a",
        source_scored_run_id="run-a",
        source_output_ref="arena/round/scores/items/score-a.json",
        source_runner_hotkey="validator-a",
        runner_authority_exclusions=["validator-a"],
    )
    evidence_hash = contracts.document_hash(snapshot)
    assert judgment_cache.validate_evidence_snapshot(
        snapshot, cache_key=scope["cache_key"], evidence_hash=evidence_hash
    ) == snapshot

    legacy = copy.deepcopy(snapshot)
    del legacy["runner_authority_exclusions"]
    with pytest.raises(judgment_cache.JudgmentCacheError, match="fields"):
        judgment_cache.validate_evidence_snapshot(
            legacy,
            cache_key=scope["cache_key"],
            evidence_hash=contracts.document_hash(legacy),
        )

    tampered = copy.deepcopy(snapshot)
    tampered["breakdowns"][0]["final_score"] = 1.0
    with pytest.raises(judgment_cache.JudgmentCacheError, match="hash"):
        judgment_cache.validate_evidence_snapshot(
            tampered, cache_key=scope["cache_key"], evidence_hash=evidence_hash
        )

    with pytest.raises(judgment_cache.JudgmentCacheError, match="another execution"):
        judgment_cache.build_evidence_snapshot(
            output=output,
            cache_scope=scope,
            source_score_run_id="score-a",
            source_scored_run_id="run-b",
            source_output_ref="arena/round/scores/items/score-a.json",
            source_runner_hotkey="validator-a",
            runner_authority_exclusions=["validator-a"],
        )


def test_service_uses_only_hash_validated_authoritative_cached_evidence(monkeypatch):
    scope = _scope(_input())
    output = scoring.build_scoring_output(
        "run-a", [{"company_name": "A", "final_score": 80.0}]
    )
    evidence = judgment_cache.build_evidence_snapshot(
        output=output,
        cache_scope=scope,
        source_score_run_id="score-a",
        source_scored_run_id="run-a",
        source_output_ref="arena/round/scores/items/score-a.json",
        source_runner_hotkey="validator-a",
        runner_authority_exclusions=["validator-a"],
    )
    cache_row = {
        "cache_key": scope["cache_key"],
        "scoring_input_hash": scope["scoring_input_hash"],
        "evidence_hash": contracts.document_hash(evidence),
        "evidence_doc": evidence,
        "source_score_run_id": "score-a",
    }
    source = {
        "run_id": "score-a", "kind": "score", "status": "accepted",
        "runner_hotkey": "validator-a", "scored_run_id": "run-a",
    }
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        get_judgment_cache=lambda _key: cache_row,
        get_run=lambda _run_id: source,
    )
    service._objects = SimpleNamespace(
        get=lambda _ref: pytest.fail("cached evidence read the object store")
    )
    captured = {}

    def validate(rows, **kwargs):
        captured.update(kwargs)
        return list(rows)

    monkeypatch.setattr(scoring, "validate_breakdowns_for_item", validate)
    run = {
        "judgment_cache_key": scope["cache_key"],
        "judgment_cache_source_run_id": "score-a",
    }
    assert service._verified_breakdowns(
        run,
        icp={"icp_id": "i-1"},
        companies=[{"company_name": "A"}],
        policy={
            "max_scored_companies": 5,
            "scoring_adapter_version": "qualification_integrity_v2",
        },
    ) == evidence["breakdowns"]
    assert captured["integrity_policy"] is True

    cache_row["evidence_hash"] = "sha256:" + "0" * 64
    with pytest.raises(scoring.ScoringError, match="cache evidence"):
        service._verified_breakdowns(
            run,
            icp={"icp_id": "i-1"},
            companies=[{"company_name": "A"}],
            policy={
                "max_scored_companies": 5,
                "scoring_adapter_version": "qualification_integrity_v2",
            },
        )


@pytest.mark.parametrize("field,value", [
    ("fit_summary", "Different validated prose"),
    ("state", "California"),
])
def test_validated_but_unused_company_fields_share_a_judgment(field, value):
    document = _input()
    original = _scope(document)["cache_key"]
    document["companies"][0][field] = value
    assert _scope(document)["cache_key"] == original


def test_fit_url_hints_beyond_actual_lookup_limit_share_a_judgment():
    from qualification.scoring.competition import _normalized_company
    from qualification.scoring.lead_scorer import _fit_evidence_url_hints
    from gateway.qualification.models import CompanyOutput

    document = _input()
    row = document["companies"][0]
    row["fit_evidence_urls"] = ["https://acme.com/fit-%d" % i for i in range(3)]
    original = _scope(document)["cache_key"]
    expected = row["fit_evidence_urls"][:]
    for index in range(20):
        row["fit_evidence_urls"] = expected + ["https://acme.com/unused-%d" % index]
        assert _scope(document)["cache_key"] == original
        model = CompanyOutput(**_normalized_company(row, integrity_policy=True))
        assert _fit_evidence_url_hints(model) == expected


@pytest.mark.parametrize("url", ["https://example.github.io/post", "https://8.8.8.8/post"])
def test_evidence_normalization_does_not_throw_for_valid_public_hosts(url):
    document = _input()
    document["companies"][0]["intent_signals"][0]["url"] = url
    assert _scope(document)["cache_key"]

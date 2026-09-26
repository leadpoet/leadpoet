"""Authenticated listing observations stay scoped to one accepted execution."""

from __future__ import annotations

import base64
from copy import deepcopy
import json

import pytest

from lab_arena import contracts, provider_observations, scorer_entrypoint, scoring


RUN_ID = "arena-2026-09-25:submission:2:8:2"
SOURCE_URL = "https://abnormal.ai/careers/jobs/7896108003?gh_jid=7896108003"


def _rows(
    *, call_identity: str = "sha256:" + "c" * 64,
    tool: str = provider_observations.TOOL,
    source_url: str = SOURCE_URL, first_seen_at: str = "2026-08-22T12:47:06Z",
    succeeded: bool = True, request_hash: str = "sha256:" + "a" * 64,
) -> list[dict]:
    response = {
        "status": "completed",
        "result": {"data": [{"attributes": {
            "url": source_url,
            "first_seen_at": first_seen_at,
            "title": "Senior Cloud Security Engineer",
            "provider_private_key": "must-not-leave-ledger",
        }}]},
    }
    scope = {
        "run_id": RUN_ID,
        "call_identity": call_identity,
        "provider": "deepline",
        "operation_id": "deepline.execute",
    }
    return [
        {
            **scope,
            "entry_kind": "reservation",
            "entry_doc": {"tool": tool, "request_hash": request_hash},
        },
        {
            **scope,
            "entry_kind": "settlement",
            "terminal_response": {
                "status": 200,
                "call_succeeded": succeeded,
                "body_b64": base64.b64encode(
                    json.dumps(response).encode()
                ).decode(),
            },
        },
    ]


class Store:
    def __init__(self, rows):
        self.rows = rows

    def list_ledger(self, **kwargs):
        selected = self.rows
        for key in ("run_id", "provider", "entry_kind", "call_identity"):
            if kwargs.get(key):
                selected = [row for row in selected if row.get(key) == kwargs[key]]
        return selected[: kwargs.get("limit")]


def _company(*, website="https://abnormal.ai/", source_url=SOURCE_URL):
    return {
        "company_name": "Abnormal AI",
        "company_website": website,
        "intent_signals": [{"url": source_url}],
    }


def _resolve(rows, *, company=None, status="accepted", evaluated="2026-09-25"):
    return provider_observations.resolve_observations(
        Store(rows), {"run_id": RUN_ID, "status": status},
        [company or _company()], evaluated,
    )


def test_projects_only_bounded_first_observation_fields():
    result = _resolve(_rows())

    assert result == [{
        "company_index": 0,
        "company_domain": "abnormal.ai",
        "source_url": SOURCE_URL,
        "first_observed_date": "2026-08-22",
    }]
    assert "provider_private_key" not in json.dumps(result)
    assert "body_b64" not in json.dumps(result)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_url", "wrong_company", "wrong_tool", "failed", "future",
        "wrong_request_hash", "duplicate_reservation", "ambiguous_call",
        "unaccepted_run",
    ],
)
def test_rejects_unbound_or_ambiguous_receipts(mutation):
    rows = _rows()
    company = _company()
    status = "accepted"
    if mutation == "wrong_url":
        company = _company(source_url=SOURCE_URL + "&other=1")
    elif mutation == "wrong_company":
        company = _company(website="https://other.example/")
    elif mutation == "wrong_tool":
        rows[0]["entry_doc"]["tool"] = "predictleads_company_news_events"
    elif mutation == "failed":
        rows[1]["terminal_response"]["call_succeeded"] = False
    elif mutation == "future":
        rows = _rows(first_seen_at="2026-09-26T00:00:00Z")
    elif mutation == "wrong_request_hash":
        rows[0]["entry_doc"]["request_hash"] = "sha256:invalid"
    elif mutation == "duplicate_reservation":
        rows.append(deepcopy(rows[0]))
    elif mutation == "ambiguous_call":
        rows += _rows(
            call_identity="sha256:" + "d" * 64,
            first_seen_at="2026-08-23T00:00:00Z",
        )
    else:
        status = "failed"

    assert _resolve(rows, company=company, status=status) == []


def test_company_selector_refuses_malformed_or_multiple_rows():
    valid = _resolve(_rows())[0]
    assert provider_observations.company_observation([valid], 0) == valid
    assert provider_observations.company_observation(
        [valid, {**valid, "first_observed_date": "2026-08-23"}], 0
    ) is None
    assert provider_observations.company_observation(
        [{**valid, "raw_response": {}}], 0
    ) is None


def test_one_ambiguous_company_does_not_erase_an_unrelated_observation():
    other_url = "https://other.example/careers/job-1"
    rows = (
        _rows()
        + _rows(
            call_identity="sha256:" + "d" * 64,
            first_seen_at="2026-08-24T00:00:00Z",
        )
        + _rows(
            call_identity="sha256:" + "e" * 64,
            source_url=other_url,
            first_seen_at="2026-08-23T00:00:00Z",
        )
    )
    result = provider_observations.resolve_observations(
        Store(rows), {"run_id": RUN_ID, "status": "accepted"},
        [_company(), _company(
            website="https://other.example/", source_url=other_url
        )],
        "2026-09-25",
    )

    assert result == [{
        "company_index": 1,
        "company_domain": "other.example",
        "source_url": other_url,
        "first_observed_date": "2026-08-23",
    }]


def test_repeated_identical_authenticated_observation_is_deduplicated():
    rows = _rows() + _rows(call_identity="sha256:" + "d" * 64)

    assert _resolve(rows) == [{
        "company_index": 0,
        "company_domain": "abnormal.ai",
        "source_url": SOURCE_URL,
        "first_observed_date": "2026-08-22",
    }]


def test_conflicting_authenticated_dates_are_ambiguous():
    rows = _rows() + _rows(
        call_identity="sha256:" + "d" * 64,
        first_seen_at="2026-08-23T00:00:00Z",
    )

    assert _resolve(rows) == []


def test_invalid_company_row_does_not_poison_an_unrelated_selector():
    valid = {
        "company_index": 1,
        "company_domain": "other.example",
        "source_url": "https://other.example/jobs/1",
        "first_observed_date": "2026-08-23",
    }
    malformed = {
        "company_index": 0,
        "company_domain": "abnormal.ai",
        "source_url": SOURCE_URL,
        "first_observed_date": "not-a-date",
    }

    assert provider_observations.company_observation(
        [malformed, valid], 0
    ) is None
    assert provider_observations.company_observation(
        [malformed, valid], 1
    ) == valid


def _policy(*, handoff: bool) -> dict:
    return scoring.build_scorer_policy(
        scoring_adapter_version="qualification_integrity_v2",
        intent_details=True,
        provider_observation_handoff=handoff,
    )


def _scoring_input(policy: dict, observations=None) -> dict:
    return scoring.build_scoring_input(
        scored_run_id=RUN_ID,
        icp={"intent_signals": ["Hiring"]},
        companies=[_company()],
        policy=policy,
        evaluation_date="2026-09-25",
        provider_observations=observations,
    )


def test_new_gateway_preserves_old_frozen_scorer_input_and_hash():
    policy = _policy(handoff=False)
    legacy = _scoring_input(policy)
    attempted = _scoring_input(policy, _resolve(_rows()))

    assert "provider_observations" not in attempted
    assert provider_observations.SCORING_ICP_KEY not in attempted["icp"]
    assert attempted == legacy
    assert contracts.document_hash(attempted) == contracts.document_hash(legacy)


def test_new_scorer_accepts_old_gateway_input_without_observation_field(
    monkeypatch,
):
    document = _scoring_input(_policy(handoff=True))
    calls = []
    monkeypatch.setattr(
        scorer_entrypoint.scoring, "apply_policy_to_environment",
        lambda *args, **kwargs: None,
    )

    def scorer(*args, **kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(scorer_entrypoint.scoring, "lab_scorer", scorer)
    monkeypatch.setattr(
        scorer_entrypoint.scoring, "score_work_item",
        lambda *args, **kwargs: [],
    )

    result = scorer_entrypoint.score_input(document)

    assert calls == [{}]
    assert result["breakdowns"] == []


def test_scorer_rejects_observations_without_frozen_capability():
    document = _scoring_input(_policy(handoff=False))
    document["icp"][provider_observations.SCORING_ICP_KEY] = _resolve(_rows())

    with pytest.raises(
        scoring.ScoringError, match="not enabled by the frozen scorer policy"
    ):
        scorer_entrypoint.score_input(document)


def test_legacy_runner_input_shape_transports_observation_to_new_scorer(
    monkeypatch,
):
    policy = _policy(handoff=True)
    observations = _resolve(_rows())
    lease_icp = provider_observations.scoring_icp(
        {"intent_signals": ["Hiring"]}, policy, observations
    )
    # Preserve the field projection used by legacy build_scoring_input.
    document = {
        "schema_version": scoring.SCORING_INPUT_SCHEMA_VERSION,
        "scored_run_id": RUN_ID,
        "icp": dict(lease_icp),
        "companies": [_company()],
        "scorer_policy": contracts.validate_scorer_policy(policy),
        "evaluation_date": "2026-09-25",
    }
    calls = []
    monkeypatch.setattr(
        scorer_entrypoint.scoring, "apply_policy_to_environment",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scorer_entrypoint.scoring, "lab_scorer",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )
    monkeypatch.setattr(
        scorer_entrypoint.scoring, "score_work_item",
        lambda *args, **kwargs: [],
    )

    result = scorer_entrypoint.score_input(document)

    assert calls == [{"provider_observations": observations}]
    assert result["breakdowns"] == []
    assert provider_observations.SCORING_ICP_KEY not in document["companies"][0]


def test_scoring_transport_does_not_mutate_original_icp_or_observations():
    policy = _policy(handoff=True)
    icp = {"intent_signals": ["Hiring"]}
    observations = _resolve(_rows())
    original_icp = deepcopy(icp)
    original_observations = deepcopy(observations)

    transported = provider_observations.scoring_icp(
        icp, policy, observations
    )
    transported["intent_signals"].append("Changed")
    transported[provider_observations.SCORING_ICP_KEY][0][
        "first_observed_date"
    ] = "2026-08-23"

    assert icp == original_icp
    assert observations == original_observations


def test_scorer_rejects_legacy_top_level_observation_transport():
    document = _scoring_input(_policy(handoff=True))
    document["provider_observations"] = _resolve(_rows())

    with pytest.raises(scoring.ScoringError, match="unsupported transport"):
        scorer_entrypoint.score_input(document)


def test_handoff_capability_requires_intent_details_policy():
    with pytest.raises(
        contracts.ArenaContractError,
        match="requires intent details",
    ):
        scoring.build_scorer_policy(provider_observation_handoff=True)

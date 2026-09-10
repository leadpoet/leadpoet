from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Optional, Set

import pytest

from lab_arena import contracts, public_dashboard
from lab_arena.service import ArenaService, RoundDefaults, ServiceError
from lab_arena.store import ArenaStore, ArenaStoreError


ROUND_ID = "arena-2026-09-12"
BASELINE_HOTKEY = "5" + "A" * 47
HIGH_HOTKEY = "5" + "B" * 47
LOW_HOTKEY = "5" + "C" * 47


def _company(domain: str, name: str) -> dict:
    return {
        "company_name": name,
        "company_website": "https://%s/" % domain,
        "company_linkedin": "",
        "industry": "Software",
        "employee_count": "10",
        "company_stage": "Seed",
        "country": "US",
        "state": "CA",
        "fit_summary": "Matches the ICP.",
        "fit_evidence_urls": ["https://%s/evidence" % domain],
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Hiring",
                "date": "2026-09-01",
                "why_now": "The team is growing.",
                "url": "https://%s/news" % domain,
                "snippet": "The company announced hiring.",
            }
        ],
        "required_attribute": None,
    }


def _output(companies: list[dict]) -> bytes:
    return contracts.canonical_json(
        {
            "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
            "companies": companies,
        }
    ).encode("utf-8")


def _runs(submission_id: str, score: float = 60.0) -> list[dict]:
    return [
        {
            "run_id": "%s-%d" % (submission_id, position),
            "round_id": ROUND_ID,
            "submission_id": submission_id,
            "icp_position": position,
            "stage": 1 if position < contracts.STAGE_1_ICP_COUNT else 2,
            "attempt": 1,
            "status": "accepted",
            "terminal_cause": "accepted",
            "per_icp_score": score,
            "output_ref": "%s-%d.json" % (submission_id, position),
        }
        for position in range(contracts.BENCHMARK_ICP_COUNT)
    ]


def _provider_row(
    amount: int,
    *,
    kind: str = "execute",
    inflight_calls: int = 0,
    uncertain_calls: int = 0,
    call_count: int = 1,
) -> dict:
    return {
        "kind": kind,
        "provider": "openrouter",
        "settled_microusd": amount,
        "reserved_or_uncertain_microusd": 0,
        "inflight_calls": inflight_calls,
        "uncertain_calls": uncertain_calls,
        "refused_calls": 0,
        "call_count": call_count,
    }


def _costs(submission_id: str, *rows: dict) -> dict:
    return contracts.validate_submission_costs(
        {
            "schema_version": contracts.SUBMISSION_COSTS_SCHEMA_VERSION,
            "submission_id": submission_id,
            "providers": list(rows),
        }
    )


def _cost_service(
    *,
    amount: int,
    companies_per_icp: int = 1,
    duplicate_domains: bool = False,
    populated_positions: Optional[Set[int]] = None,
):
    submission_id = "sub-cost"
    runs = _runs(submission_id)
    objects = {}
    for position in range(contracts.BENCHMARK_ICP_COUNT):
        companies = [
            _company(
                (
                    "same.co.uk"
                    if index == 0
                    else "www.same.co.uk"
                    if index == 1
                    else "department-%d.same.co.uk" % index
                )
                if duplicate_domains
                else "company-%d-%d.com" % (position, index),
                "Company %d %d" % (position, index),
            )
            for index in range(companies_per_icp)
        ] if populated_positions is None or position in populated_positions else []
        objects[runs[position]["output_ref"]] = _output(companies)

    class Objects:
        @staticmethod
        def get_bounded(ref, max_bytes):
            assert len(objects[ref]) <= max_bytes
            return objects[ref]

    class Store:
        @staticmethod
        def submission_costs(requested):
            assert requested == submission_id
            return _costs(requested, _provider_row(amount))

    service = object.__new__(ArenaService)
    service._objects = Objects()
    service._store = Store()
    row = {
        "round_id": ROUND_ID,
        "configuration_doc": {
            "execution_cap_microusd": 50_000_000,
            "cost_per_company_microusd": 500_000,
        },
    }
    return service, row, submission_id, runs


@pytest.mark.parametrize(
    ("amount", "eligible", "reason"),
    [
        (500_000, True, "eligible"),
        (500_001, False, "cost_per_company_exceeded"),
    ],
)
def test_half_dollar_per_company_boundary_is_exact(amount, eligible, reason):
    service, row, submission_id, runs = _cost_service(
        amount=amount, populated_positions={0}
    )

    result = service._submission_cost_eligibility(row, submission_id, runs)

    assert result["eligible"] is eligible
    assert result["eligibility_reason"] == reason
    assert result["cost_summary"]["returned_company_count"] == 1


@pytest.mark.parametrize(
    ("amount", "eligible", "reason"),
    [
        (50_000_000, True, "eligible"),
        (50_000_001, False, "execution_cap_exceeded"),
    ],
)
def test_fifty_dollar_total_boundary_is_exact(amount, eligible, reason):
    service, row, submission_id, runs = _cost_service(
        amount=amount, companies_per_icp=5
    )

    result = service._submission_cost_eligibility(row, submission_id, runs)

    assert result["eligible"] is eligible
    assert result["eligibility_reason"] == reason
    assert result["cost_summary"]["returned_company_count"] == 100


def test_duplicate_domains_do_not_pad_the_denominator_but_repeat_icps_count():
    service, row, submission_id, runs = _cost_service(
        amount=10_000_001, companies_per_icp=5, duplicate_domains=True
    )

    result = service._submission_cost_eligibility(row, submission_id, runs)

    assert result["cost_summary"]["returned_company_count"] == 20
    assert result["cost_summary"]["eligibility_cap_microusd"] == 10_000_000
    assert result["eligible"] is False


def test_submission_rpc_total_is_not_reset_by_attempt_selection():
    service, row, submission_id, runs = _cost_service(
        amount=600_000, populated_positions={0}
    )
    retry = dict(runs[0], run_id="sub-cost-0-retry", attempt=2)
    runs.append(retry)

    result = service._submission_cost_eligibility(row, submission_id, runs)

    assert result["cost_summary"]["returned_company_count"] == 1
    assert result["cost_summary"]["execution"]["settled_microusd"] == 600_000
    assert result["eligible"] is False


def test_invalid_stored_output_fails_only_cost_eligibility():
    service, row, submission_id, runs = _cost_service(amount=0)
    service._objects.get_bounded = lambda _ref, _limit: b'{"bad":true}'
    service._store.submission_costs = lambda _submission_id: pytest.fail(
        "invalid output must fail before cost data can be treated as zero"
    )

    result = service._submission_cost_eligibility(row, submission_id, runs)

    assert result == {
        "cost_summary": None,
        "eligible": False,
        "eligibility_reason": "stored_output_invalid",
    }


def test_inflight_calls_fail_closed_while_uncertain_cost_is_conservative():
    service, row, submission_id, runs = _cost_service(amount=0)
    service._store.submission_costs = lambda requested: _costs(
        requested,
        _provider_row(0, inflight_calls=1),
    )
    inflight = service._submission_cost_eligibility(row, submission_id, runs)
    assert inflight["eligible"] is False
    assert inflight["eligibility_reason"] == "provider_calls_inflight"

    uncertain_row = _provider_row(0, uncertain_calls=1)
    uncertain_row["reserved_or_uncertain_microusd"] = 10_000_001
    service._store.submission_costs = lambda requested: _costs(
        requested, uncertain_row
    )
    uncertain = service._submission_cost_eligibility(row, submission_id, runs)
    assert uncertain["cost_summary"]["execution"]["settled_microusd"] == 0
    assert uncertain["cost_summary"]["execution"]["conservative_microusd"] == 10_000_001
    assert uncertain["eligible"] is False


def test_publish_excludes_only_cost_ineligible_challenger_and_keeps_raw_scores():
    participants = [
        {"submission_id": "baseline", "miner_hotkey": BASELINE_HOTKEY, "is_king": True},
        {"submission_id": "high", "miner_hotkey": HIGH_HOTKEY, "is_king": False},
        {"submission_id": "low", "miner_hotkey": LOW_HOTKEY, "is_king": False},
    ]
    runs = _runs("baseline", 50.0) + _runs("high", 90.0) + _runs("low", 60.0)
    objects = {
        run["output_ref"]: _output(
            [_company("%s-%d.example.com" % (run["submission_id"], run["icp_position"]), "Company")]
        )
        for run in runs
    }
    amounts = {"baseline": 50_000_000, "high": 10_000_001, "low": 10_000_000}
    writes = []

    class Store:
        @staticmethod
        def list_runs(_round_id, **_filters):
            return runs

        @staticmethod
        def submission_costs(submission_id):
            return _costs(submission_id, _provider_row(amounts[submission_id]))

        @staticmethod
        def transition_round(_round_id, old, new, patch):
            writes.append((old, new, patch))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = type(
        "Objects",
        (),
        {"get_bounded": staticmethod(lambda ref, _limit: objects[ref])},
    )()
    service._clock = lambda: datetime(2026, 9, 12, tzinfo=timezone.utc)
    service._round = lambda _round_id: {
        "round_id": ROUND_ID,
        "status": "scored",
        "participants": participants,
        "finalists": ["high", "low"],
        "configuration_doc": {
            "execution_cap_microusd": 50_000_000,
            "cost_per_company_microusd": 500_000,
        },
    }

    result = service.publish(ROUND_ID)

    assert result["king_outcome"] == "crowned"
    assert result["king_hotkey"] == LOW_HOTKEY
    publication = writes[0][2]["publication_doc"]
    assert len(publication) == 8
    ranking = {row["submission_id"]: row for row in publication["final_ranking"]}
    assert ranking["high"]["final_score"] == 90.0
    assert ranking["high"]["eligible"] is False
    assert ranking["low"]["eligible"] is True
    assert ranking["baseline"]["final_score"] == 50.0
    assert ranking["baseline"]["eligible"] is False


def test_historical_round_stays_eligible_without_cost_or_output_reads():
    service = object.__new__(ArenaService)
    service._store = type(
        "Store",
        (),
        {"submission_costs": staticmethod(lambda _submission_id: pytest.fail("unexpected RPC"))},
    )()
    service._objects = type(
        "Objects",
        (),
        {"get_bounded": staticmethod(lambda *_args: pytest.fail("unexpected object read"))},
    )()

    result = service._submission_cost_eligibility(
        {"configuration_doc": {"execution_cap_microusd": 5_000_000}},
        "historical",
        [],
    )

    assert result == {
        "cost_summary": None,
        "eligible": True,
        "eligibility_reason": "historical_round",
    }


def test_cost_rpc_shape_is_strict_and_submission_bound():
    calls = []

    class Transport:
        @staticmethod
        def rpc(function, params):
            calls.append((function, params))
            return _costs("sub-cost", _provider_row(123))

    assert ArenaStore(Transport()).submission_costs("sub-cost")["providers"][0][
        "settled_microusd"
    ] == 123
    assert calls == [
        ("lab_arena_submission_costs", {"p_submission_id": "sub-cost"})
    ]

    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_submission_costs(
            {
                "schema_version": contracts.SUBMISSION_COSTS_SCHEMA_VERSION,
                "submission_id": "sub-cost",
                "providers": [{**_provider_row(1), "settled_microusd": -1}],
            }
        )

    class WrongSubmissionTransport:
        @staticmethod
        def rpc(_function, _params):
            return _costs("other", _provider_row(1))

    with pytest.raises(ArenaStoreError, match="wrong submission"):
        ArenaStore(WrongSubmissionTransport()).submission_costs("sub-cost")


def test_dashboard_projects_costs_only_from_published_final_ranking():
    cost = {
        "returned_company_count": 1,
        "execution_cap_microusd": 50_000_000,
        "cost_per_company_cap_microusd": 500_000,
        "eligibility_cap_microusd": 500_000,
        "execution": ArenaService._cost_kind_summary(
            _costs("sub", _provider_row(500_000)), "execute"
        ),
        "judge": ArenaService._cost_kind_summary(
            _costs("sub", _provider_row(50, kind="score")), "score"
        ),
    }
    final = {
        "rank": 1,
        "submission_id": "sub",
        "final_score": 60.0,
        "is_baseline": False,
        "cost_summary": cost,
        "eligible": True,
        "eligibility_reason": "eligible",
        "private_cost_detail": "must-not-leak",
    }
    base = {
        "round_id": ROUND_ID,
        "participants": [
            {"submission_id": "sub", "miner_hotkey": LOW_HOTKEY, "is_king": False}
        ],
        "publication_doc": {
            "participants": [
                {"submission_id": "sub", "miner_hotkey": LOW_HOTKEY, "is_baseline": False}
            ],
            "final_ranking": [final],
            "king_decision": {
                "outcome": "crowned",
                "winner_submission_id": "sub",
                "king_submission_id": "sub",
                "king_hotkey": LOW_HOTKEY,
            },
        },
        "configuration_doc": {"mode": "live"},
    }
    unpublished = public_dashboard.round_summary({**base, "status": "scored"})
    assert "eligible" not in json.dumps(unpublished)

    published = public_dashboard.round_summary({**base, "status": "published"})
    serialized = json.dumps(published, sort_keys=True)
    assert '"eligible": true' in serialized
    assert "private_cost_detail" not in serialized


def test_new_round_defaults_freeze_fifty_dollars_and_half_dollar_per_company():
    defaults = RoundDefaults()
    assert defaults.execution_cap_microusd == 50_000_000
    assert defaults.cost_per_company_microusd == 500_000


def test_commit_preflight_adopts_only_legacy_live_round_budget():
    legacy_live = ArenaService._configuration_for_commit(
        {"mode": "live", "execution_cap_microusd": 5_000_000}
    )
    assert legacy_live["execution_cap_microusd"] == 50_000_000
    assert legacy_live["cost_per_company_microusd"] == 500_000

    typed_live = ArenaService._configuration_for_commit(
        {
            "mode": "live",
            "execution_cap_microusd": 8_000_000,
            "cost_per_company_microusd": 250_000,
        }
    )
    assert typed_live["execution_cap_microusd"] == 8_000_000
    assert typed_live["cost_per_company_microusd"] == 250_000

    shadow = ArenaService._configuration_for_commit(
        {"mode": "shadow", "execution_cap_microusd": 123_000}
    )
    assert shadow == {"mode": "shadow", "execution_cap_microusd": 123_000}


def _startup_service(cost_rpc_result):
    class Transport:
        def __init__(self):
            self.cost_probe_seen = False

        def rpc(self, function, params):
            if function == "lab_arena_schema_version_v1":
                return {
                    "schema_version": "leadpoet.lab_arena.schema_version.v1",
                    "version": 197,
                }
            if function == "lab_arena_submission_costs":
                self.cost_probe_seen = True
                assert params == {"p_submission_id": "__arena_budget_probe__"}
                if isinstance(cost_rpc_result, Exception):
                    raise cost_rpc_result
                return cost_rpc_result
            raise ArenaStoreError("lab_arena_round_missing")

        @staticmethod
        def select(_table, **_kwargs):
            return []

    class Objects:
        value = b""

        def put(self, _ref, value):
            self.value = value

        def get(self, _ref):
            return self.value

    transport = Transport()
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        require_service_role=lambda: {"current_user": "lab_arena_service"},
        _transport=transport,
    )
    service._objects = Objects()
    service._config = SimpleNamespace(
        daily_icp_source=lambda **_kwargs: {"status": "unavailable"}
    )
    service._scorer_policy = {"scoring_adapter_version": "test"}
    service._clock = lambda: datetime(2026, 9, 10, tzinfo=timezone.utc)
    service.current_round = lambda: None
    return service, transport


def test_startup_probes_cost_rpc_grant_and_missing_submission_path():
    service, transport = _startup_service(
        ArenaStoreError("lab_arena_submission_missing")
    )

    assert service.startup_checks()["current_round"] is None
    assert transport.cost_probe_seen is True


@pytest.mark.parametrize(
    "result",
    [
        {},
        ArenaStoreError("permission denied for function lab_arena_submission_costs"),
    ],
)
def test_startup_rejects_missing_or_misgranted_cost_rpc(result):
    service, _transport = _startup_service(result)

    with pytest.raises(ServiceError) as caught:
        service.startup_checks()

    assert "lab_arena_submission_costs" in str(caught.value)

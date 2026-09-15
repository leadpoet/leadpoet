from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import contracts
from lab_arena.service import DEFAULT_BASELINE_SOURCE_URL
from scripts import verify_arena_parallel_round as verification


ROUND_ID = "arena-2026-09-14-proxye2e"
SUBMISSION_ID = "baseline-submission"


def _configuration():
    return {
        "round_id": ROUND_ID,
        "mode": "shadow",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": False,
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "runner_slot_ceiling": 20,
        "parallel_twenty_icp_execution": True,
        "baseline_hotkey": "baseline-hotkey",
        "baseline_source_url": DEFAULT_BASELINE_SOURCE_URL,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/scorer@sha256:" + "a" * 64,
        "runner_hotkeys": ["runner-hotkey"],
    }


def _provider_cost(kind, amount):
    return {
        "kind": kind,
        "provider": "openrouter",
        "settled_microusd": amount,
        "reserved_or_uncertain_microusd": 0,
        "inflight_calls": 0,
        "uncertain_calls": 0,
        "refused_calls": 0,
        "call_count": 20,
        "successful_microusd": amount,
        "successful_calls": 20,
        "success_unresolved_microusd": 0,
        "success_unresolved_calls": 0,
    }


class Store:
    def __init__(self):
        execution_cost = _provider_cost("execute", 123)
        score_cost = _provider_cost("score", 456)
        self.row = {
            "round_id": ROUND_ID,
            "status": "published",
            "status_generation": 9,
            "stage_generation": 6,
            "champion_funding_frozen": True,
            "champion_submission_id": None,
            "configuration_doc": _configuration(),
            "participants": [
                {
                    "submission_id": SUBMISSION_ID,
                    "miner_hotkey": "baseline-hotkey",
                    "is_king": True,
                }
            ],
            "king_outcome": "no_king",
            "cancel_reason": None,
            "publication_doc": {
                "final_ranking": [
                    {
                        "submission_id": SUBMISSION_ID,
                        "miner_hotkey": "baseline-hotkey",
                        "is_baseline": True,
                        "rank": 1,
                        "final_score": 77.5,
                        "eligible": True,
                        "eligibility_reason": "eligible",
                        "cost_summary": {
                            "execution": {"settled_microusd": 123},
                            "judge": {"settled_microusd": 456},
                        },
                    }
                ]
            },
        }
        self.runs = []
        for kind in ("execute", "score"):
            for position in range(20):
                if kind == "execute":
                    wave = 0 if position < 10 else 10
                    resource_summary = {
                        "web_egress": {
                            "worker_slot": position % 10,
                            "exit_fingerprint": "exit%012d" % (position % 10),
                        }
                    }
                else:
                    wave = 30 + position
                    resource_summary = {}
                self.runs.append(
                    {
                        "kind": kind,
                        "run_id": "%s-%d-run" % (kind, position),
                        "assignment_id": "%s-%d" % (kind, position),
                        "submission_id": SUBMISSION_ID,
                        "icp_position": position,
                        "status": "accepted",
                        "runner_hotkey": "runner-hotkey",
                        "output_ref": "arena/output/%s/%d.json" % (kind, position),
                        "per_icp_score": 77.5 if kind == "execute" else None,
                        "result_doc": {
                            "terminal_status": "accepted",
                            "started_at": "2026-09-14T00:00:%02dZ" % wave,
                            "finished_at": "2026-09-14T00:00:%02dZ" % (wave + 10),
                            "resource_summary": resource_summary,
                        },
                    }
                )
        self.costs = {
            "submission_id": SUBMISSION_ID,
            "providers": [execution_cost, score_cost],
        }

    def get_round(self, round_id):
        return self.row if round_id == ROUND_ID else None

    def list_runs(self, round_id):
        assert round_id == ROUND_ID
        return list(self.runs)

    def submission_costs(self, submission_id):
        assert submission_id == SUBMISSION_ID
        return self.costs

    def list_submissions(self, round_id):
        assert round_id == ROUND_ID
        return [{"submission_id": SUBMISSION_ID, "is_king": True}]

    def provider_funding(self, run_id, provider):
        assert run_id and provider in contracts.PROVIDERS
        return {
            "status": "available",
            "funding_source": "host",
            "champion_funding": False,
            "credential_submission_id": None,
            "credential_miner_hotkey": None,
            "restart_required": False,
        }

    def list_ledger(self, *, run_id):
        assert run_id
        return [{"funding_source": "host"}]


class Service:
    def __init__(self):
        self.store = Store()
        self.config = SimpleNamespace(
            mode="shadow",
            pinned_round_id=ROUND_ID,
            network_name="finney",
            netuid=71,
            defaults=SimpleNamespace(
                baseline_hotkey="baseline-hotkey",
                baseline_source_url=DEFAULT_BASELINE_SOURCE_URL,
                scorer_image_digest="sha256:" + "a" * 64,
                scorer_image_reference=(
                    "registry.example/scorer@sha256:" + "a" * 64
                ),
                runner_hotkeys=("runner-hotkey",),
            ),
        )
        self.calls = []

    def public_results(self, round_id, submission_id):
        assert (round_id, submission_id) == (ROUND_ID, SUBMISSION_ID)
        return {
            "outputs": {"run-%d" % position: {"companies": []} for position in range(20)},
            "scores": {
                "stage_1": [
                    {"icp_position": position} for position in range(10)
                ],
                "stage_2": [
                    {"icp_position": position} for position in range(10, 20)
                ],
            },
            "submission_scores": {"stage_1": 76.0, "final": 77.5},
        }

    def advance_round(self, round_id):
        self.calls.append(("advance", round_id))
        return {"status": "terminal", "round_status": "published"}

    def reconcile_closed_provider_costs(self):
        self.calls.append(("reconcile", ROUND_ID))
        return {"status": "none"}


def test_published_evidence_requires_all_twenty_unique_outputs_scores_and_costs():
    service = Service()

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["execution"]["assignment_count"] == 20
    assert document["execution"]["accepted_output_ref_count"] == 20
    assert document["execution"]["scored_assignment_count"] == 20
    assert document["scoring"]["assignment_count"] == 20
    assert document["scoring"]["accepted_output_ref_count"] == 20
    assert document["execution_timing"] == {
        "accepted_interval_count": 20,
        "invalid_result_count": 0,
        "runner_hotkeys": ["runner-hotkey"],
        "worker_slots": list(range(10)),
        "exit_fingerprint_count": 10,
        "concurrency_high_water_lower_bound": 10,
        "first_batch_latest_finish": "2026-09-14T00:00:10Z",
        "second_batch_earliest_start": "2026-09-14T00:00:10Z",
        "second_batch_started_after_first_finished": True,
    }
    assert document["public_result"]["output_count"] == 20
    assert document["ledger_funding"] == {
        "entry_count": 40,
        "sources": ["host"],
        "all_host": True,
    }
    assert document["ledger"][0]["totals"] == {
        "execute": {
            **{field: 0 for field in verification.LEDGER_FIELDS},
            "settled_microusd": 123,
            "call_count": 20,
            "successful_microusd": 123,
            "successful_calls": 20,
        },
        "score": {
            **{field: 0 for field in verification.LEDGER_FIELDS},
            "settled_microusd": 456,
            "call_count": 20,
            "successful_microusd": 456,
            "successful_calls": 20,
        },
    }

    service.store.runs.append(dict(service.store.runs[0]))
    incomplete = verification._evidence(service, ROUND_ID)
    assert incomplete["proof"]["complete"] is False
    assert "execute_accepted_uniqueness" in incomplete["proof"]["errors"]


def test_published_evidence_rejects_early_second_batch_and_wrong_high_water():
    service = Service()
    execution = [run for run in service.store.runs if run["kind"] == "execute"]
    execution[9]["result_doc"]["started_at"] = "2026-09-14T00:00:09Z"
    execution[10]["result_doc"]["started_at"] = "2026-09-14T00:00:09Z"

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"]["complete"] is False
    assert "execution_high_water" in document["proof"]["errors"]
    assert "execution_batch_barrier" in document["proof"]["errors"]


def test_round_scope_and_driver_are_strictly_pinned():
    service = Service()
    verification._validate_frozen_round(service, service.store.row)
    advance, reconciliation = verification._tick(service, ROUND_ID)
    assert advance["round_status"] == "published"
    assert reconciliation == {"status": "none"}
    assert service.calls == [
        ("advance", ROUND_ID),
        ("reconcile", ROUND_ID),
    ]
    verified = set()
    verification._verify_host_funding(service, ROUND_ID, verified)
    assert len(verified) == 40

    service.store.row["configuration_doc"]["mode"] = "live"
    with pytest.raises(verification.VerificationError, match="mode"):
        verification._validate_frozen_round(service, service.store.row)

    service.store.row["configuration_doc"]["mode"] = "shadow"
    service.store.row["champion_submission_id"] = "prior-shadow-champion"
    with pytest.raises(verification.VerificationError, match="champion"):
        verification._verify_host_funding(service, ROUND_ID, set())


@pytest.mark.parametrize(
    "round_id",
    (
        "arena-2026-09-14",
        "arena-2026-09-14-PROXY",
        "arena-2026-09-14-proxy-e2e-too-long",
    ),
)
def test_round_id_requires_a_bounded_explicit_suffix(round_id):
    with pytest.raises(verification.VerificationError):
        verification._validate_round_id(round_id)


def test_round_id_accepts_the_operator_suffix():
    assert verification._validate_round_id(ROUND_ID) == ROUND_ID

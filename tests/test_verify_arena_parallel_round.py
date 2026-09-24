from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, icp_disclosure, intent_details_policy, scoring
from lab_arena.service import DEFAULT_BASELINE_SOURCE_URL
from scripts import verify_arena_parallel_round as verification


ROUND_ID = "arena-2026-09-14-proxye2e"
SUBMISSION_ID = "baseline-submission"
REPLAY_SOURCE_ROUND_ID = "arena-2026-09-24"
REPLAY_TARGET_ROUND_ID = "arena-2026-09-24-replay"


def _replay_company(name: str, index: int) -> dict:
    slug = name.casefold().replace(" ", "-")
    return {
        "company_name": name,
        "company_website": f"https://{slug}.example.com/",
        "company_linkedin": f"https://linkedin.com/company/{slug}",
        "industry": "Software",
        "employee_count": "51-200",
        "company_stage": "Series A",
        "country": "United States",
        "state": "CA",
        "intent_details": f"{name} announced a product launch.",
        "intent_signals": [
            {
                "matched_icp_signal": 0,
                "description": "Product launch",
                "date": "2026-09-%02d" % (index + 1),
                "url": f"https://{slug}.example.com/launch",
            }
        ],
        "company_stage_evidence": [],
        "required_attribute": None,
    }


def _replay_source_fixture(monkeypatch):
    source_configuration = {
        "network_name": "finney",
        "netuid": 71,
        "stage_1_icp_count": 1,
        "stage_2_icp_count": 1,
        "integrity_policy": "arena_integrity_v1",
        "intent_details_policy": intent_details_policy.POLICY,
    }
    participant_ids = ("source-a", "source-b", "source-c")
    source = {
        "round_id": REPLAY_SOURCE_ROUND_ID,
        "status": "published",
        "evaluation_date": "2026-09-24",
        "benchmark_ref": "arena/source/benchmark.json",
        "configuration_doc": source_configuration,
        "participants": [
            {"submission_id": submission_id}
            for submission_id in participant_ids
        ],
    }
    benchmark = {
        "round_id": REPLAY_SOURCE_ROUND_ID,
        "icps": [
            {"icp_id": "source-0", "prompt": "First ICP"},
            {"icp_id": "source-1", "prompt": "Second ICP"},
        ],
    }
    documents = {}
    runs = []
    groups = {
        ("source-a", 0): [_replay_company("Acme", 0)],
        ("source-a", 1): [],
        ("source-b", 0): [_replay_company("Beta", 1)],
        ("source-b", 1): [_replay_company("Cedar", 2)],
    }
    for submission_id in participant_ids[:2]:
        for position in range(2):
            output_ref = f"arena/source/{submission_id}/{position}.json"
            document = {
                "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
                "companies": groups[(submission_id, position)],
            }
            documents[output_ref] = json.dumps(document).encode()
            runs.append(
                {
                    "kind": "execute",
                    "status": "accepted",
                    "submission_id": submission_id,
                    "icp_position": position,
                    "run_id": f"{submission_id}:{position}",
                    "output_ref": output_ref,
                    "output_hash": "",
                }
            )
    for position in range(2):
        runs.append(
            {
                "kind": "execute",
                "status": "failed",
                "submission_id": "source-c",
                "icp_position": position,
                "run_id": f"source-c:{position}",
            }
        )
    documents[source["benchmark_ref"]] = json.dumps(benchmark).encode()

    class SourceStore:
        def get_round(self, round_id):
            return source if round_id == REPLAY_SOURCE_ROUND_ID else None

        def list_runs(self, round_id):
            assert round_id == REPLAY_SOURCE_ROUND_ID
            return list(runs)

    def get(ref):
        return documents[ref]

    def get_bounded(ref, maximum):
        value = documents[ref]
        assert len(value) <= maximum
        return value

    monkeypatch.setattr(
        verification,
        "_utc_now",
        lambda: datetime(2026, 9, 24, tzinfo=timezone.utc),
    )
    built = SimpleNamespace(
        store=SourceStore(),
        _objects=SimpleNamespace(get=get, get_bounded=get_bounded),
        config=SimpleNamespace(network_name="finney", netuid=71),
    )
    return built, runs, documents


def test_saved_output_replay_binds_all_published_nonempty_groups_and_archive(monkeypatch):
    built, _runs, _documents = _replay_source_fixture(monkeypatch)

    first = verification._saved_output_replay(
        built, REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID
    )
    second = verification._saved_output_replay(
        built, REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID
    )

    assert first["archive"] == second["archive"]
    assert first["source_url"].endswith(
        first["evidence"]["archive_hash"].removeprefix("sha256:") + ".tar.gz"
    )
    assert first["evidence"] | {
        "origins": [],
    } == {
        "source_round": REPLAY_SOURCE_ROUND_ID,
        "input_hash": first["evidence"]["input_hash"],
        "archive_hash": contracts.hash_bytes(first["archive"]),
        "archive_size_bytes": len(first["archive"]),
        "evaluation_date": "2026-09-24",
        "output_schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
        "source_benchmark_icp_count": 2,
        "source_participant_count": 3,
        "source_accepted_output_count": 4,
        "omitted_failed_assignment_count": 2,
        "source_failed_attempt_count": 2,
        "omitted_empty_output_count": 1,
        "source_contributor_count": 2,
        "replayed_group_count": 3,
        "replayed_company_count": 3,
        "selection_scope": "all_published_participant_accepted_nonempty_execute_outputs",
        "origins": [],
    }
    assert [origin["source_submission_id"] for origin in first["evidence"]["origins"]] == [
        "source-a",
        "source-b",
        "source-b",
    ]
    assert all(
        origin["source_output_hash"] == origin["target_output_hash"]
        and origin["source_stored_output_hash_present"] is False
        for origin in first["evidence"]["origins"]
    )


def test_saved_output_replay_rejects_noncanonical_or_mismatched_source_hash(monkeypatch):
    built, runs, documents = _replay_source_fixture(monkeypatch)
    first = runs[0]
    document = json.loads(documents[first["output_ref"]])
    document["companies"][0]["contact"] = "removed by V6 normalization"
    documents[first["output_ref"]] = json.dumps(document).encode()

    with pytest.raises(verification.VerificationError, match="output hash"):
        verification._saved_output_replay(
            built, REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID
        )

    documents[first["output_ref"]] = json.dumps(
        {
            "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
            "companies": [_replay_company("Acme", 0)],
        }
    ).encode()
    first["output_hash"] = "sha256:" + "0" * 64
    with pytest.raises(verification.VerificationError, match="output hash"):
        verification._saved_output_replay(
            built, REPLAY_SOURCE_ROUND_ID, REPLAY_TARGET_ROUND_ID
        )


def test_specialized_shadow_verifier_preserves_current_count_and_margin(monkeypatch):
    from lab_arena import api, service, wiring

    @dataclass
    class Defaults:
        benchmark_icp_count: int = 10
        promotion_margin: float = 0.5
        runner_slot_ceiling: int = 10
        rewards_enabled: bool = True
        daily_cutoff_hour_utc: int = 0
        baseline_source_url: str = "example"

    @dataclass
    class Config:
        defaults: Defaults
        mode: str = "live"
        pinned_round_id: str | None = None
        reward_signer_factory: object = object()
        baseline_promoter_factory: object = object()
        code_reviewer: object = object()

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(wiring, "build_service_from_environment", lambda _mode: (SimpleNamespace(config=Config(Defaults())), None))
    monkeypatch.setattr(service, "ArenaService", lambda config: SimpleNamespace(config=config))
    monkeypatch.setattr(api, "create_app", lambda _service: "test-app")

    built, app = verification._build_pinned_service(ROUND_ID)
    assert app == "test-app"
    assert built.config.defaults.benchmark_icp_count == 10
    assert built.config.defaults.promotion_margin == 0.5
    assert built.config.mode == "shadow"
    assert built.config.pinned_round_id == ROUND_ID


def _configuration(*, count=10, parallel=False):
    configuration = {
        "round_id": ROUND_ID,
        "mode": "shadow",
        "network_name": "finney",
        "netuid": 71,
        "rewards_enabled": False,
        "stage_1_icp_count": (count + 1) // 2,
        "stage_2_icp_count": count // 2,
        "promotion_margin": 0.5,
        "runner_slot_ceiling": 10,
        "baseline_hotkey": "baseline-hotkey",
        "baseline_source_url": DEFAULT_BASELINE_SOURCE_URL,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/scorer@sha256:" + "a" * 64,
        "runner_hotkeys": ["runner-hotkey"],
        "benchmark_disclosure_policy": icp_disclosure.DELAYED_DISCLOSURE_POLICY,
        "integrity_policy": "arena_integrity_v1",
        "intent_details_policy": intent_details_policy.POLICY,
        "schedule": {
            "submission_open": "2026-09-14T00:00:00Z",
            "submission_cutoff": "2026-09-15T00:00:00Z",
        },
    }
    if parallel:
        configuration["parallel_twenty_icp_execution"] = True
    else:
        configuration["execution_sequence_policy"] = (
            contracts.BASELINE_SCORED_FIRST_POLICY
        )
    return configuration


def _provider_cost(kind, amount, count):
    return {
        "kind": kind,
        "provider": "openrouter",
        "settled_microusd": amount,
        "reserved_or_uncertain_microusd": 0,
        "inflight_calls": 0,
        "uncertain_calls": 0,
        "refused_calls": 0,
        "call_count": count,
        "successful_microusd": amount,
        "successful_calls": count,
        "success_unresolved_microusd": 0,
        "success_unresolved_calls": 0,
    }


class Store:
    def __init__(self, *, count=10, parallel=False):
        execution_cost = _provider_cost("execute", 123, count)
        score_cost = _provider_cost("score", 456, count)
        self.row = {
            "round_id": ROUND_ID,
            "status": "published",
            "status_generation": 9,
            "stage_generation": 6,
            "champion_funding_frozen": True,
            "champion_submission_id": None,
            "icp_set_date": "2026-09-14",
            "evaluation_date": "2026-09-15",
            "benchmark_ref": "arena/benchmarks/2026-09-14.json",
            "configuration_doc": _configuration(count=count, parallel=parallel),
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
            for position in range(count):
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
                        "scored_run_id": (
                            "execute-%d-run" % position if kind == "score" else None
                        ),
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
    def __init__(self, *, count=10, parallel=False):
        self.store = Store(count=count, parallel=parallel)
        self.current_time = datetime(2026, 9, 16, tzinfo=timezone.utc)
        self.object_get_calls = 0
        self._objects = SimpleNamespace(get_bounded=self._get_bounded)
        self.object_documents = {}
        for run in self.store.runs:
            if run["kind"] == "execute":
                document = {
                    "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
                    "companies": [],
                }
            else:
                document = {
                    "schema_version": scoring.SCORING_OUTPUT_SCHEMA_VERSION,
                    "scored_run_id": run["scored_run_id"],
                    "breakdowns": [],
                }
                run["output_hash"] = contracts.document_hash(document)
            self.object_documents[run["output_ref"]] = json.dumps(document).encode()
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
                benchmark_icp_count=count,
                promotion_margin=0.5,
                runner_slot_ceiling=10,
            ),
        )
        self.calls = []

    def now(self):
        return self.current_time

    def _get_bounded(self, ref, max_bytes):
        self.object_get_calls += 1
        value = self.object_documents[ref]
        if len(value) > max_bytes:
            raise ValueError("object is too large")
        return value

    def public_results(self, round_id, submission_id):
        assert (round_id, submission_id) == (ROUND_ID, SUBMISSION_ID)
        disclosed = icp_disclosure.baseline_disclosure(
            self.store.row, self.store.runs, now=self.now()
        )
        if disclosed is None:
            return {
                "outputs": {},
                "run_results": [],
                "scores": {"stage_1": [], "stage_2": []},
                "public_icp_status": "pending",
                "public_icp_count": 0,
                "submission_scores": {"stage_1": 76.0, "final": 77.5},
            }
        stage_one_count = int(
            self.store.row["configuration_doc"]["stage_1_icp_count"]
        )
        return {
            "outputs": {
                "run-%d" % position: {"companies": []}
                for position in range(len(self.store.runs) // 2)
            },
            "run_results": [{} for _position in range(len(self.store.runs) // 2)],
            "scores": {
                "stage_1": [
                    {"icp_position": position}
                    for position in range(stage_one_count)
                ],
                "stage_2": [
                    {"icp_position": position}
                    for position in range(
                        stage_one_count, len(self.store.runs) // 2
                    )
                ],
            },
            "public_icp_status": "ready",
            "public_icp_count": len(self.store.runs) // 2,
            "submission_scores": {"stage_1": 76.0, "final": 77.5},
        }

    def advance_round(self, round_id):
        self.calls.append(("advance", round_id))
        return {"status": "terminal", "round_status": "published"}

    def reconcile_closed_provider_costs(self):
        self.calls.append(("reconcile", ROUND_ID))
        return {"status": "none"}


def _replay_target_service() -> Service:
    service = Service(count=2)
    archive = b"deterministic replay archive"
    archive_hash = contracts.hash_bytes(archive)
    source_url = (
        "https://arena.invalid/saved-output-replay/"
        f"{archive_hash.removeprefix('sha256:')}.tar.gz"
    )
    service.store.row["configuration_doc"]["baseline_source_url"] = source_url
    service.config.defaults.baseline_source_url = source_url
    service.store.row["publication_doc"]["final_ranking"][0]["cost_summary"][
        "execution"
    ]["settled_microusd"] = 0
    execute_cost = _provider_cost_row(service, "execute")
    execute_cost.update({field: 0 for field in verification.LEDGER_FIELDS})
    benchmark = {
        "round_id": ROUND_ID,
        "icps": [
            {"icp_id": f"{ROUND_ID}:replay:{position}", "prompt": f"Replay {position}"}
            for position in range(2)
        ],
    }
    service.object_documents[service.store.row["benchmark_ref"]] = json.dumps(
        benchmark
    ).encode()
    source_ref = (
        f"arena/{ROUND_ID}/sources/"
        f"baseline-{ROUND_ID.removeprefix('arena-')}.tar.gz"
    )
    service.object_documents[source_ref] = archive
    origins = []
    for position, run in enumerate(
        row for row in service.store.runs if row["kind"] == "execute"
    ):
        companies = [_replay_company("Target %d" % position, position)]
        document = {
            "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
            "companies": companies,
        }
        service.object_documents[run["output_ref"]] = json.dumps(document).encode()
        document_hash = contracts.document_hash(document)
        run["output_hash"] = ""
        origins.append(
            {
                "position": position,
                "source_round": REPLAY_SOURCE_ROUND_ID,
                "source_run_id": f"source:{position}",
                "source_submission_id": "source-a",
                "source_icp_position": position,
                "source_output_hash": document_hash,
                "source_stored_output_hash_present": False,
                "target_output_hash": document_hash,
                "original_icp_hash": contracts.document_hash(
                    {"prompt": f"Replay {position}"}
                ),
                "companies_hash": contracts.document_hash(companies),
                "company_count": 1,
            }
        )
    service._objects = SimpleNamespace(
        get=lambda ref: service.object_documents[ref],
        get_bounded=service._get_bounded,
    )
    service._saved_output_replay = {
        "source_round": REPLAY_SOURCE_ROUND_ID,
        "input_hash": "sha256:" + "1" * 64,
        "archive_hash": archive_hash,
        "archive_size_bytes": len(archive),
        "evaluation_date": service.store.row["evaluation_date"],
        "output_schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
        "source_benchmark_icp_count": 2,
        "source_participant_count": 1,
        "source_accepted_output_count": 2,
        "omitted_failed_assignment_count": 0,
        "source_failed_attempt_count": 0,
        "omitted_empty_output_count": 0,
        "source_contributor_count": 1,
        "replayed_group_count": 2,
        "replayed_company_count": 2,
        "selection_scope": "all_published_participant_accepted_nonempty_execute_outputs",
        "origins": origins,
    }
    return service


def test_replay_evidence_requires_exact_outputs_and_zero_execute_provider_use():
    service = _replay_target_service()

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["saved_output_replay"]["replayed_group_count"] == 2
    assert document["durable_outputs"]["execute"]["stored_hash_count"] == 0

    charged = _replay_target_service()
    _provider_cost_row(charged, "execute")["call_count"] = 1
    charged_document = verification._evidence(charged, ROUND_ID)
    assert "replay_execute_provider_use" in charged_document["proof"]["errors"]

    changed = _replay_target_service()
    changed._saved_output_replay["origins"][0]["target_output_hash"] = (
        "sha256:" + "0" * 64
    )
    changed_document = verification._evidence(changed, ROUND_ID)
    assert "replay_output_changed" in changed_document["proof"]["errors"]


def test_replay_resume_rejects_archive_or_output_schema_mismatch():
    service = _replay_target_service()
    source_ref = (
        f"arena/{ROUND_ID}/sources/"
        f"baseline-{ROUND_ID.removeprefix('arena-')}.tar.gz"
    )
    service.object_documents[source_ref] = b"different archive"

    with pytest.raises(verification.VerificationError, match="archive differs"):
        verification._validate_frozen_round(service, service.store.row)

    schema_changed = _replay_target_service()
    schema_changed._saved_output_replay["output_schema_version"] = (
        intent_details_policy.OUTPUT_SCHEMA
    )
    with pytest.raises(verification.VerificationError, match="replay_output_schema"):
        verification._validate_frozen_round(schema_changed, schema_changed.store.row)


def test_published_evidence_requires_all_configured_outputs_scores_and_costs():
    service = Service()

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["execution"]["assignment_count"] == 10
    assert document["execution"]["accepted_output_ref_count"] == 10
    assert document["execution"]["scored_assignment_count"] == 10
    assert document["scoring"]["assignment_count"] == 10
    assert document["scoring"]["accepted_output_ref_count"] == 10
    assert document["execution_timing"] == {
        "accepted_interval_count": 10,
        "invalid_result_count": 0,
        "failed_attempt_interval_count": 0,
        "invalid_failed_attempt_interval_count": 0,
        "runner_hotkeys": ["runner-hotkey"],
        "worker_slots": list(range(10)),
        "exit_fingerprint_count": 10,
        "concurrency_evidence": "overlapping_valid_attempt_intervals",
        "physical_concurrency_evidence": "operator_sandbox_pid_observations_required",
        "concurrency_high_water_lower_bound": 10,
        "first_batch_concurrency_high_water_lower_bound": 10,
        "second_batch_concurrency_high_water_lower_bound": 0,
        "first_batch_latest_finish": "2026-09-14T00:00:10Z",
        "second_batch_earliest_start": None,
        "second_batch_started_after_first_finished": False,
    }
    assert document["public_result"]["output_count"] == 10
    assert document["durable_outputs"] == {
        "execute": {
            "accepted_count": 10,
            "verified_count": 10,
            "stored_hash_count": 0,
            "invalid_count": 0,
            "contact_field_count": 0,
        },
        "score": {
            "accepted_count": 10,
            "verified_count": 10,
            "stored_hash_count": 10,
            "invalid_count": 0,
        },
    }
    assert document["disclosure"] == {
        "status": "ready",
        "public_at": "2026-09-16T00:00:00Z",
        "policy": icp_disclosure.DELAYED_DISCLOSURE_POLICY,
    }
    assert document["ledger_funding"] == {
        "entry_count": 20,
        "sources": ["host"],
        "all_host": True,
    }
    assert document["ledger"][0]["totals"] == {
        "execute": {
            **{field: 0 for field in verification.LEDGER_FIELDS},
            "settled_microusd": 123,
            "call_count": 10,
            "successful_microusd": 123,
            "successful_calls": 10,
        },
        "score": {
            **{field: 0 for field in verification.LEDGER_FIELDS},
            "settled_microusd": 456,
            "call_count": 10,
            "successful_microusd": 456,
            "successful_calls": 10,
        },
    }

    service.store.runs.append(dict(service.store.runs[0]))
    incomplete = verification._evidence(service, ROUND_ID)
    assert incomplete["proof"]["complete"] is False
    assert "execute_accepted_uniqueness" in incomplete["proof"]["errors"]


def _provider_cost_row(service, kind):
    return next(
        row for row in service.store.costs["providers"] if row["kind"] == kind
    )


@pytest.mark.parametrize("kind", ("execute", "score"))
def test_current_cost_proof_keeps_terminal_failed_uncertainty_as_audit_evidence(
    kind,
):
    service = Service()
    service.store.row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    provider = _provider_cost_row(service, kind)
    provider["call_count"] += 1
    provider["uncertain_calls"] = 1

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["ledger"][0]["totals"][kind]["uncertain_calls"] == 1
    assert document["ledger"][0]["totals"][kind][
        "reserved_or_uncertain_microusd"
    ] == 0
    assert document["ledger"][0]["totals"][kind][
        "success_unresolved_calls"
    ] == 0


@pytest.mark.parametrize("kind", ("execute", "score"))
def test_legacy_cost_proof_still_rejects_terminal_uncertainty(kind):
    service = Service()
    provider = _provider_cost_row(service, kind)
    provider["call_count"] += 1
    provider["uncertain_calls"] = 1

    document = verification._evidence(service, ROUND_ID)

    assert "open_costs" in document["proof"]["errors"]


@pytest.mark.parametrize("kind", ("execute", "score"))
@pytest.mark.parametrize(
    "blocked_state",
    (
        {"inflight_calls": 1},
        {"uncertain_calls": 1, "success_unresolved_calls": 1},
        {"uncertain_calls": 1, "success_unresolved_microusd": 1},
        {"uncertain_calls": 1, "reserved_or_uncertain_microusd": 1},
    ),
    ids=(
        "active-call",
        "possible-success-call",
        "possible-success-amount",
        "nonzero-hold",
    ),
)
def test_current_cost_proof_rejects_open_or_potentially_billable_state(
    kind, blocked_state,
):
    service = Service()
    service.store.row["configuration_doc"]["sourcing_cost_eligibility_policy"] = (
        contracts.PER_ICP_SUCCESSFUL_CALLS_COST_POLICY
    )
    _provider_cost_row(service, kind).update(blocked_state)

    document = verification._evidence(service, ROUND_ID)

    assert "open_costs" in document["proof"]["errors"]


def test_pre_disclosure_evidence_requires_pending_public_projection_and_private_objects():
    service = Service()
    service.current_time = datetime(2026, 9, 15, 12, tzinfo=timezone.utc)

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["disclosure"] == {
        "status": "pending",
        "public_at": "2026-09-16T00:00:00Z",
        "policy": icp_disclosure.DELAYED_DISCLOSURE_POLICY,
    }
    assert document["public_result"] == {
        "output_count": 0,
        "run_result_count": 0,
        "stage_1_score_count": 0,
        "stage_2_score_count": 0,
        "scored_positions": [],
        "public_icp_status": "pending",
        "public_icp_count": 0,
        "contact_verifications_present": False,
        "contact_verification_count": 0,
        "submission_scores": {"stage_1": 76.0, "final": 77.5},
    }
    assert document["durable_outputs"]["execute"]["verified_count"] == 10
    assert document["durable_outputs"]["score"]["verified_count"] == 10

    original = service.public_results

    def leaking_public_results(round_id, submission_id):
        result = original(round_id, submission_id)
        result["outputs"] = {"private-run": {"companies": []}}
        return result

    service.public_results = leaking_public_results
    leaked = verification._evidence(service, ROUND_ID)
    assert "public_outputs_pending_contract" in leaked["proof"]["errors"]

    def leaking_score_results(round_id, submission_id):
        result = original(round_id, submission_id)
        result["scores"]["stage_1"] = [{"icp_position": 0}]
        return result

    service.public_results = leaking_score_results
    leaked = verification._evidence(service, ROUND_ID)
    assert "public_outputs_pending_contract" in leaked["proof"]["errors"]

    def leaking_contact_results(round_id, submission_id):
        result = original(round_id, submission_id)
        result["contact_verifications"] = {"private-run": [{}]}
        return result

    service.public_results = leaking_contact_results
    leaked = verification._evidence(service, ROUND_ID)
    assert "public_outputs_pending_contract" in leaked["proof"]["errors"]


def test_durable_objects_are_read_only_at_terminal_state_and_cached_by_identity():
    service = Service()
    service.store.row["status"] = "scoring_2"

    active = verification._evidence(service, ROUND_ID)

    assert active["durable_outputs"] is None
    assert service.object_get_calls == 0

    service.store.row["status"] = "published"
    verification._evidence(service, ROUND_ID)
    assert service.object_get_calls == 20
    verification._evidence(service, ROUND_ID)
    assert service.object_get_calls == 20


def test_published_evidence_rejects_an_invalid_durable_score_object():
    service = Service()
    score = next(run for run in service.store.runs if run["kind"] == "score")
    score["output_hash"] = "sha256:" + "0" * 64

    document = verification._evidence(service, ROUND_ID)

    assert document["durable_outputs"]["score"] == {
        "accepted_count": 10,
        "verified_count": 9,
        "stored_hash_count": 10,
        "invalid_count": 1,
    }
    assert "durable_score_outputs" in document["proof"]["errors"]


def test_company_only_evidence_checks_raw_durable_contact_fields():
    service = Service()
    execution = next(
        run for run in service.store.runs if run["kind"] == "execute"
    )
    service.object_documents[execution["output_ref"]] = json.dumps(
        {
            "schema_version": intent_details_policy.COMPANY_ONLY_OUTPUT_SCHEMA,
            "companies": [
                {
                    "company_name": "Acme",
                    "company_website": "https://acme.example.com/",
                    "company_linkedin": "https://linkedin.com/company/acme",
                    "industry": "Software",
                    "employee_count": "51-200",
                    "company_stage": "Series A",
                    "country": "United States",
                    "state": "CA",
                    "intent_details": "Acme announced a product launch.",
                    "intent_signals": [
                        {
                            "matched_icp_signal": 0,
                            "description": "Product launch",
                            "date": "2026-09-01",
                            "url": "https://acme.example.com/launch",
                        }
                    ],
                    "company_stage_evidence": [],
                    "required_attribute": None,
                    "contact": "discarded by V6 normalization",
                }
            ],
        }
    ).encode()

    document = verification._evidence(service, ROUND_ID)

    assert document["durable_outputs"]["execute"]["verified_count"] == 10
    assert document["durable_outputs"]["execute"]["contact_field_count"] == 1
    assert "company_only_output_contains_contacts" in document["proof"]["errors"]


def test_published_evidence_rejects_early_second_batch_and_wrong_high_water():
    service = Service(count=20, parallel=True)
    execution = [run for run in service.store.runs if run["kind"] == "execute"]
    execution[9]["result_doc"]["started_at"] = "2026-09-14T00:00:09Z"
    execution[10]["result_doc"]["started_at"] = "2026-09-14T00:00:09Z"

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"]["complete"] is False
    assert "execution_high_water" in document["proof"]["errors"]
    assert "execution_batch_barrier" in document["proof"]["errors"]


def test_published_evidence_rejects_sequential_first_batch_before_parallel_second():
    service = Service(count=20, parallel=True)
    execution = [run for run in service.store.runs if run["kind"] == "execute"]
    for position in range(10):
        execution[position]["result_doc"]["started_at"] = (
            "2026-09-14T00:00:%02dZ" % position
        )
        execution[position]["result_doc"]["finished_at"] = (
            "2026-09-14T00:00:%02dZ" % (position + 1)
        )

    document = verification._evidence(service, ROUND_ID)

    assert document["execution_timing"][
        "first_batch_concurrency_high_water_lower_bound"
    ] == 1
    assert document["execution_timing"][
        "second_batch_concurrency_high_water_lower_bound"
    ] == 10
    assert "execution_first_batch_high_water" in document["proof"]["errors"]
    assert "execution_second_batch_high_water" not in document["proof"]["errors"]


def test_failed_initial_attempt_keeps_real_first_wave_overlap_in_proof():
    service = Service(count=20, parallel=True)
    execution = [run for run in service.store.runs if run["kind"] == "execute"]
    accepted_retry = execution[0]
    accepted_retry["result_doc"]["started_at"] = "2026-09-14T00:00:10Z"
    accepted_retry["result_doc"]["finished_at"] = "2026-09-14T00:00:20Z"
    for run in execution[10:]:
        run["result_doc"]["started_at"] = "2026-09-14T00:00:20Z"
        run["result_doc"]["finished_at"] = "2026-09-14T00:00:30Z"
    failed_initial = {
        **accepted_retry,
        "run_id": "execute-0-initial-failed",
        "status": "failed",
        "output_ref": "",
        "per_icp_score": None,
        "result_doc": {
            **accepted_retry["result_doc"],
            "terminal_status": "model_error",
            "started_at": "2026-09-14T00:00:00Z",
            "finished_at": "2026-09-14T00:00:10Z",
        },
    }
    service.store.runs.append(failed_initial)

    document = verification._evidence(service, ROUND_ID)

    assert document["proof"] == {"complete": True, "errors": []}
    assert document["execution"]["failed_attempts"] == 1
    assert document["execution_timing"]["failed_attempt_interval_count"] == 1
    assert document["execution_timing"][
        "first_batch_concurrency_high_water_lower_bound"
    ] == 10
    assert document["execution_timing"][
        "second_batch_concurrency_high_water_lower_bound"
    ] == 10


def test_failed_second_batch_attempt_cannot_hide_an_early_batch_start():
    service = Service(count=20, parallel=True)
    accepted_second = next(
        run
        for run in service.store.runs
        if run["kind"] == "execute" and run["icp_position"] == 10
    )
    failed_early = {
        **accepted_second,
        "run_id": "execute-10-initial-failed",
        "status": "failed",
        "output_ref": "",
        "per_icp_score": None,
        "result_doc": {
            **accepted_second["result_doc"],
            "terminal_status": "model_error",
            "started_at": "2026-09-14T00:00:09Z",
            "finished_at": "2026-09-14T00:00:10Z",
        },
    }
    service.store.runs.append(failed_early)

    document = verification._evidence(service, ROUND_ID)

    assert document["execution_timing"]["failed_attempt_interval_count"] == 1
    assert (
        document["execution_timing"]["second_batch_earliest_start"]
        == "2026-09-14T00:00:09Z"
    )
    assert document["execution_timing"][
        "second_batch_started_after_first_finished"
    ] is False
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
    assert len(verified) == 20

    service.store.row["configuration_doc"]["mode"] = "live"
    with pytest.raises(verification.VerificationError, match="mode"):
        verification._validate_frozen_round(service, service.store.row)

    service.store.row["configuration_doc"]["mode"] = "shadow"
    service.store.row["champion_submission_id"] = "prior-shadow-champion"
    with pytest.raises(verification.VerificationError, match="champion"):
        verification._verify_host_funding(service, ROUND_ID, set())


def test_current_baseline_first_policy_accepts_cutoff_disclosure():
    service = Service()
    configuration = service.store.row["configuration_doc"]
    configuration.pop("parallel_twenty_icp_execution", None)
    configuration["execution_sequence_policy"] = contracts.BASELINE_SCORED_FIRST_POLICY
    configuration["benchmark_disclosure_policy"] = icp_disclosure.CUTOFF_PUBLIC_POLICY
    verification._validate_frozen_round(service, service.store.row)

    for run in service.store.runs:
        if run["kind"] == "execute" and run["icp_position"] == 0:
            run["result_doc"]["finished_at"] = "2026-09-14T00:00:05Z"
        elif run["kind"] == "execute" and run["icp_position"] == 10:
            run["result_doc"]["started_at"] = "2026-09-14T00:00:05Z"
    document = verification._evidence(service, ROUND_ID)
    assert "execution_batch_barrier" not in document["proof"]["errors"]
    assert document["proof"]["complete"] is True, document["proof"]
    assert document["round"]["execution_sequence_policy"] == contracts.BASELINE_SCORED_FIRST_POLICY


@pytest.mark.parametrize("field,value", [
    ("execution_sequence_policy", "unknown_policy"),
    ("benchmark_disclosure_policy", "unknown_policy"),
    ("rewards_enabled", True),
])
def test_current_policy_still_rejects_unknown_or_rewarding_configuration(field, value):
    service = Service()
    configuration = service.store.row["configuration_doc"]
    configuration.pop("parallel_twenty_icp_execution", None)
    configuration["execution_sequence_policy"] = contracts.BASELINE_SCORED_FIRST_POLICY
    configuration["benchmark_disclosure_policy"] = icp_disclosure.CUTOFF_PUBLIC_POLICY
    configuration[field] = value
    with pytest.raises(verification.VerificationError, match=field):
        verification._validate_frozen_round(service, service.store.row)


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

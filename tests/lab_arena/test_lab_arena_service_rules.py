"""Focused service rules for the simple two-stage competition."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, rewards, scoring, signing, source_bundle
from lab_arena.output import validate_output_document
from lab_arena.service import ArenaService, S3ObjectStore, ServiceError
from lab_arena.store import ArenaStoreError, hash_lease_token


def _schedule():
    return {
        "submission_open": "2026-09-02T00:00:00Z",
        "submission_cutoff": "2026-09-02T01:00:00Z",
    }


def test_startup_requires_the_next_day_bank_date_column():
    class Transport:
        @staticmethod
        def rpc(function, _params):
            assert function == "lab_arena_schema_version_v1"
            return {
                "schema_version": "leadpoet.lab_arena.schema_version.v1",
                "version": 197,
            }

        @staticmethod
        def select(table, *, limit, columns):
            assert table == "lab_arena_rounds"
            assert limit == 1 and columns == "icp_set_date"
            raise ArenaStoreError("column icp_set_date does not exist")

    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        require_service_role=lambda: {"current_user": "lab_arena_service"},
        _transport=Transport(),
    )
    with pytest.raises(ServiceError) as caught:
        service.startup_checks()
    assert caught.value.code == "table_unavailable:lab_arena_rounds"


@pytest.mark.parametrize(
    "status",
    ["open", "benchmark_committed", "stage1", "stage1_closed", "stage1_scoring",
     "stage1_judged", "stage1_scored", "stage2"],
)
def test_public_benchmark_stays_private_while_execution_can_continue(status):
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {"status": status}
    service.benchmark_icps = lambda _round_id: pytest.fail("private benchmark was read")

    with pytest.raises(ServiceError, match="benchmark_not_public"):
        service.public_benchmark("arena-2026-09-09")


def test_cancelled_round_without_a_benchmark_does_not_invent_one():
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {"status": "cancelled", "benchmark_ref": None}
    service.benchmark_icps = lambda _round_id: pytest.fail("undisclosed benchmark was read")

    with pytest.raises(ServiceError, match="benchmark_not_public"):
        service.public_benchmark("arena-2026-09-09")


def test_cancelled_benchmark_remains_bound_to_the_service_network():
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71)
    service._store = SimpleNamespace(get_round=lambda _round_id: {
        "status": "cancelled",
        "configuration_doc": {"mode": "live", "network_name": "test", "netuid": 401},
    })

    with pytest.raises(ServiceError, match="round_network_mismatch"):
        service.public_benchmark("arena-2026-09-09")


def _scoring_driver_service(stage, runs, *, work_item_ids=("execute-planned",)):
    round_id = "arena-2026-09-02"
    positions = list(contracts.stage_positions(stage))
    work_items = [
        {
            "scored_run_id": scored_run_id,
            "submission_id": "submission-%d" % index,
            "icp_position": positions[index],
            "output_ref": "arena/output-%d.json" % index,
        }
        for index, scored_run_id in enumerate(work_item_ids)
    ]
    plan = {
        "schema_version": contracts.SCORING_PLAN_SCHEMA_VERSION,
        "round_id": round_id,
        "stage": stage,
        "work_items": work_items,
        "zero_rows": [],
    }
    row = {
        "round_id": round_id,
        "status": "stage%d_scoring" % stage,
        "configuration_doc": {
            "schedule": {
                "submission_open": "2026-09-01T00:00:00Z",
                "submission_cutoff": "2026-09-02T00:00:00Z",
                "benchmark_deadline": "2026-09-02T00:30:00Z",
                "stage_1_start": "2026-09-02T00:30:01Z",
                "stage_1_close": "2026-09-02T04:30:01Z",
                "stage_1_scoring_close": "2026-09-02T11:00:01Z",
                "stage_2_start": "2026-09-02T11:00:02Z",
                "stage_2_close": "2026-09-02T14:00:02Z",
                "final_scoring_close": "2026-09-02T19:00:02Z",
                "publication_deadline": "2026-09-02T19:00:03Z",
            }
        },
        "stage%d_scoring_plan_doc" % stage: plan,
    }

    class Store:
        def __init__(self):
            self.cancelled = []
            self.expired = 0

        def expire_leases(self, requested_round_id):
            assert requested_round_id == round_id
            self.expired += 1
            return {"status": "ok"}

        def list_runs(self, requested_round_id, **filters):
            assert requested_round_id == round_id
            assert filters == {"stage": stage, "kind": "score"}
            return list(runs)

        def cancel_round(self, requested_round_id, reason):
            self.cancelled.append((requested_round_id, reason))
            return {"status": "cancelled", "round_status": "cancelled"}

        def close_scoring(self, *_args):
            return {"status": "closed", "round_status": "stage%d_judged" % stage}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._clock = lambda: datetime(2026, 9, 2, tzinfo=timezone.utc)
    service._lock = threading.RLock()
    service._round = lambda requested_round_id: row
    return service


@pytest.mark.parametrize("status", ["committed", "stage1_scored"])
def test_advance_never_opens_execution_before_submission_cutoff(status):
    cutoff = datetime(2026, 9, 2, tzinfo=timezone.utc)
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        defaults=SimpleNamespace(stage_minutes={
            "benchmark": 30,
            "stage_1": 240,
            "stage_1_scoring": 390,
            "stage_2": 180,
            "final_scoring": 300,
        })
    )
    schedule = service.build_schedule(cutoff)
    service._clock = lambda: cutoff - timedelta(microseconds=1)
    service._lock = threading.RLock()
    service._round = lambda _round_id: {
        "status": status,
        "configuration_doc": {"schedule": schedule},
    }
    service.open_stage = lambda *_args: pytest.fail("execution opened before cutoff")

    assert service._advance_round_locked("arena-2026-09-02") == {
        "status": "waiting",
        "round_status": status,
    }


@pytest.mark.parametrize(
    ("stage", "cause"),
    [(1, "judge_error"), (2, "judge_error"), (1, "lease_expired")],
)
def test_advance_cancels_scoring_when_a_planned_judge_failure_exhausts_retries(
    stage, cause
):
    work_item_ids = ("execute-failed", "execute-pending")
    runs = [
        {
            "scored_run_id": "execute-failed",
            "attempt": 1,
            "status": "failed",
            "terminal_cause": "judge_error",
        },
        {
            "scored_run_id": "execute-failed",
            "attempt": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
            "status": "failed",
            "terminal_cause": cause,
        },
        {
            "scored_run_id": "execute-pending",
            "attempt": 1,
            "status": "pending",
            "terminal_cause": None,
        },
    ]
    service = _scoring_driver_service(stage, runs, work_item_ids=work_item_ids)

    result = service._advance_round_locked("arena-2026-09-02")

    assert result == {"status": "cancelled", "round_status": "cancelled"}
    assert service._store.cancelled == [
        ("arena-2026-09-02", "scoring_incomplete")
    ]


@pytest.mark.parametrize(
    "runs",
    [
        [
            {
                "scored_run_id": "execute-planned",
                "attempt": 1,
                "status": "failed",
                "terminal_cause": "judge_error",
            },
            {
                "scored_run_id": "execute-planned",
                "attempt": 2,
                "status": "pending",
                "terminal_cause": None,
            },
        ],
        [
            {
                "scored_run_id": "execute-planned",
                "attempt": 1,
                "status": "accepted",
                "terminal_cause": "accepted",
            },
            {
                "scored_run_id": "execute-planned",
                "attempt": 2,
                "status": "failed",
                "terminal_cause": "judge_error",
            },
        ],
        [
            {
                "scored_run_id": "execute-planned",
                "attempt": 1,
                "status": "pending",
                "terminal_cause": None,
            },
            {
                "scored_run_id": "execute-outside-plan",
                "attempt": 2,
                "status": "failed",
                "terminal_cause": "judge_error",
            },
        ],
    ],
    ids=("retry-active", "accepted-overrides-failure", "nonplan-failure"),
)
def test_advance_keeps_scoring_open_when_a_planned_item_can_still_succeed(runs):
    work_item_ids = ("execute-planned", "execute-pending")
    runs = list(runs) + [
        {
            "scored_run_id": "execute-pending",
            "attempt": 1,
            "status": "pending",
            "terminal_cause": None,
        }
    ]
    service = _scoring_driver_service(1, runs, work_item_ids=work_item_ids)

    result = service._advance_round_locked("arena-2026-09-02")

    assert result == {"status": "waiting", "round_status": "stage1_scoring"}
    assert service._store.cancelled == []


@pytest.mark.parametrize("cause", ["credential_error", "budget_exhausted"])
def test_challenger_account_failure_does_not_trigger_early_scoring_cancel(cause):
    work_item_ids = ("execute-ineligible", "execute-pending")
    runs = [
        {
            "scored_run_id": "execute-ineligible",
            "attempt": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
            "status": "failed",
            "terminal_cause": cause,
        },
        {
            "scored_run_id": "execute-pending",
            "attempt": 1,
            "status": "pending",
            "terminal_cause": None,
        },
    ]
    service = _scoring_driver_service(1, runs, work_item_ids=work_item_ids)

    result = service._advance_round_locked("arena-2026-09-02")

    assert result == {"status": "waiting", "round_status": "stage1_scoring"}
    assert service._store.cancelled == []


def test_run_context_keeps_the_durable_round_identity():
    run = {
        "round_id": "arena-2026-09-04",
        "assignment_id": "assignment-1",
        "attempt": 2,
        "icp_position": 3,
        "miner_hotkey": "5" * 48,
        "submission_id": "submission-1",
        "stage": 2,
        "kind": "score",
    }
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(get_run=lambda _run_id: run)
    returned, context = service._run_context("run-1", "lease-token")
    assert returned is run
    assert context.round_id == "arena-2026-09-04"
    assert context.lease_token_hash == hash_lease_token("lease-token")


@pytest.mark.parametrize("moment", ["2026-09-01T23:59:59+00:00", "2026-09-02T01:00:00+00:00"])
def test_submission_requires_the_full_half_open_time_window(moment):
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime.fromisoformat(moment)
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": "5" + "A" * 47, "body": {}},
        {"round_id": "arena-2026-09-02", "status": "open", "configuration_doc": {"schedule": _schedule()}},
    )
    with pytest.raises(ServiceError, match="submission_window_closed"):
        service.handle_submission_presign({})


def test_submission_inside_the_window_registers_normally():
    documents = []

    class Store:
        @staticmethod
        def register_submission(*_args):
            documents.append(_args[3])
            return {
                "status": "registered",
                "submission_id": _args[1],
                "source_ref": _args[3]["source_ref"],
            }

    class Objects:
        @staticmethod
        def presign_put(ref, **_kwargs):
            return {
                "upload_url": "https://uploads.example/" + ref,
                "upload_headers": {
                    "content-type": source_bundle.SOURCE_CONTENT_TYPE
                },
                "expires_in_seconds": 900,
            }

    hotkey = "5" + "A" * 47
    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = Objects()
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._config = SimpleNamespace(chain=SimpleNamespace(uid_for_hotkey=lambda value: 1 if value == hotkey else None))
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": hotkey, "body": {"source_size_bytes": 10, "consent": {"public_rerun": True}}},
        {"round_id": "arena-2026-09-02", "status": "open", "configuration_doc": {"schedule": _schedule(), "baseline_hotkey": "5" + "Z" * 47}},
    )
    result = service.handle_submission_presign({})
    assert result["status"] == "upload_ready"
    assert result["submission_id"].startswith("sub-")
    assert result["source_ref"].endswith(result["submission_id"] + ".tar.gz")
    assert "source_sha256" not in documents[0]
    assert "source_cache_key" not in documents[0]


def test_source_upload_target_is_write_once_and_size_bound():
    calls = []
    client = SimpleNamespace(
        generate_presigned_url=lambda operation, **kwargs: calls.append(
            (operation, kwargs)
        )
        or "https://uploads.example/source"
    )
    objects = S3ObjectStore("arena-bucket", client=client)
    target = objects.presign_put(
        "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        size_bytes=123,
        content_type=source_bundle.SOURCE_CONTENT_TYPE,
        expires_seconds=900,
    )
    assert target["upload_headers"] == {
        "content-type": source_bundle.SOURCE_CONTENT_TYPE,
        "content-length": "123",
        "if-none-match": "*",
    }
    params = calls[0][1]["Params"]
    assert params["ContentLength"] == 123 and params["IfNoneMatch"] == "*"


def test_bounded_s3_read_always_closes_the_streaming_body():
    class Body:
        closed = False

        @staticmethod
        def read(_limit):
            return b"source"

        def close(self):
            self.closed = True

    body = Body()
    client = SimpleNamespace(
        head_object=lambda **_kwargs: {"ContentLength": 6},
        get_object=lambda **_kwargs: {"Body": body},
    )
    assert S3ObjectStore("arena-bucket", client=client).get_bounded("ref", 10) == b"source"
    assert body.closed is True


def test_completion_cannot_cross_the_signed_round_boundary():
    result = {
        "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {
            "wall_seconds": 1.0, "cpu_seconds": 1.0, "max_rss_bytes": 1,
            "stdout_bytes": 1, "stderr_bytes": 0, "provider_call_count": 0,
        },
        "started_at": "2026-09-02T00:00:00Z",
        "finished_at": "2026-09-02T00:00:01Z",
        "terminal_status": "model_error",
    }
    service = object.__new__(ArenaService)
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": "runner", "body": {"run_id": "run-1", "result": result}},
        {"round_id": "arena-a"},
    )
    service._store = SimpleNamespace(get_run=lambda _run_id: {"round_id": "arena-b", "runner_hotkey": "runner"})
    with pytest.raises(ServiceError, match="run_round_mismatch"):
        service.handle_complete({})


def test_final_admission_must_finish_before_the_round_freezes():
    service = object.__new__(ArenaService)
    service._hot_round_lock = __import__("threading").Lock()
    service._hot_rounds = {}
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-02",
        "status": "open",
        "configuration_doc": {"schedule": {"submission_cutoff": "2026-09-02T01:00:00Z"}},
    }
    service._clock = lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)
    service.admit_uploaded_submissions = lambda *_args, **_kwargs: {"status": "ok", "remaining": 2}
    service.commit_benchmark = lambda _round_id: pytest.fail("round froze before admission finished")
    assert service.advance_round("arena-2026-09-02") == {
        "status": "retry", "round_status": "open", "remaining_admissions": 2,
    }


def test_compact_publication_combines_persisted_ten_plus_twenty_scores_without_objects():
    round_id = "arena-2026-09-02"
    king = "5" + "A" * 47
    challenger = "5" + "B" * 47
    participants = [
        {"submission_id": "king", "miner_hotkey": king, "submitted_reference": "registry.example/king:latest", "is_king": True},
        {"submission_id": "challenger", "miner_hotkey": challenger, "submitted_reference": "registry.example/challenger:latest", "is_king": False},
    ]
    runs = []
    for participant, score in ((participants[0], 50.0), (participants[1], 60.0)):
        for position in range(contracts.BENCHMARK_ICP_COUNT):
            runs.append({
                "run_id": "%s-%d" % (participant["submission_id"], position),
                "round_id": round_id,
                "submission_id": participant["submission_id"],
                "icp_position": position,
                "stage": 1 if position < contracts.STAGE_1_ICP_COUNT else 2,
                "attempt": 1,
                "terminal_cause": "accepted",
                "per_icp_score": score,
            })
    writes = []

    class Store:
        @staticmethod
        def list_runs(_round_id, **_filters):
            return runs

        @staticmethod
        def transition_round(_round_id, old, new, patch):
            writes.append((old, new, patch))
            return {"status": "ok"}

    class NoObjects:
        def get(self, _ref):
            pytest.fail("publication read an object")

        def put(self, _ref, _data):
            pytest.fail("publication wrote an object")

    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = NoObjects()
    service._clock = lambda: datetime(2026, 9, 2, 2, 0, tzinfo=timezone.utc)
    service._round = lambda _round_id: {
        "round_id": round_id, "status": "scored", "participants": participants, "finalists": ["challenger"],
    }
    result = service.publish(round_id)
    assert result == {"status": "ok", "king_outcome": "crowned", "king_hotkey": challenger}
    publication = writes[0][2]["publication_doc"]
    assert set(writes[0][2]) == {"publication_doc", "published_at"}
    assert set(publication) == {
        "schema_version", "round_id", "participants", "stage1_ranking", "finalists",
        "final_ranking", "king_decision", "published_at",
    }
    assert publication["stage1_ranking"][0]["stage1_score"] == 60.0
    assert publication["final_ranking"][0]["final_score"] == 60.0
    assert publication["participants"] == [
        {"submission_id": "king", "miner_hotkey": king, "is_baseline": True},
        {"submission_id": "challenger", "miner_hotkey": challenger, "is_baseline": False},
    ]
    assert all("is_king" not in row for row in publication["final_ranking"])


def test_reward_activation_processes_only_enabled_live_rounds_oldest_first():
    rows = [
        {"round_id": "new", "configuration_doc": {"mode": "live", "rewards_enabled": True}},
        {"round_id": "shadow", "configuration_doc": {"mode": "shadow", "rewards_enabled": False}},
        {"round_id": "old", "configuration_doc": {"mode": "live", "rewards_enabled": True}},
    ]
    calls = []
    service = object.__new__(ArenaService)
    queries = []
    service._config = SimpleNamespace(mode="live", network_name="test", netuid=401)
    service._store = SimpleNamespace(
        list_rounds=lambda **kwargs: queries.append(kwargs) or rows
    )
    service.activate_reward = lambda round_id: calls.append(round_id) or {"status": "activated"}
    assert service.activate_pending_rewards() == {"status": "ok", "activated": 2}
    assert calls == ["old", "new"]
    assert queries == [{
        "status": "published", "mode": "live", "network_name": "test",
        "netuid": 401, "limit": 200,
    }]

    service._config = SimpleNamespace(mode="shadow", network_name="test", netuid=401)
    calls.clear()
    assert service.activate_pending_rewards() == {"status": "disabled", "activated": 0}
    assert calls == []


@pytest.mark.parametrize("status", ["published", "open"])
def test_pinned_reward_activation_never_scans_foreign_rounds(status):
    round_id = "arena-2026-09-07-promotion-proof"
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(
        mode="live", network_name="test", netuid=401, pinned_round_id=round_id
    )
    row = {
        "round_id": round_id, "status": status,
        "configuration_doc": {
            "mode": "live", "network_name": "test", "netuid": 401,
            "rewards_enabled": True,
        },
    }
    service._store = SimpleNamespace(get_round=lambda selected: row if selected == round_id else None)
    activated = []
    service.activate_reward = lambda selected: activated.append(selected) or {"status": "activated"}
    count = int(status == "published")
    assert service.activate_pending_rewards() == {"status": "ok", "activated": count}
    assert activated == ([round_id] if count else [])


def test_round_scope_treats_legacy_rows_as_finney_71_and_rejects_cross_chain():
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live", network_name="finney", netuid=71)
    legacy = {"configuration_doc": {"mode": "live"}}
    assert service._require_round_mode(legacy) is legacy

    service._config = SimpleNamespace(mode="live", network_name="test", netuid=401)
    with pytest.raises(ServiceError, match="round_network_mismatch"):
        service._require_round_mode(legacy)
    testnet = {
        "configuration_doc": {
            "mode": "live", "network_name": "test", "netuid": 401,
        }
    }
    assert service._require_round_mode(testnet) is testnet


def test_reward_activation_carries_only_the_latest_miner_winner():
    baseline = "5" + "A" * 47
    miner_a = "5" + "B" * 47
    miner_b = "5" + "C" * 47
    constants = rewards.reward_constants_document()

    def activate(daily_hotkey="", previous_hotkey="", previous_start=80):
        signer = signing.LocalSigner.generate()
        prior = []
        if previous_hotkey:
            prior_basis = signing.sign_document(
                signer,
                rewards.reward_basis_document(
                    round_id="arena-2026-09-01",
                    published_at="2026-09-01T00:00:00Z",
                    finalized_epoch=99,
                    king_hotkey=previous_hotkey,
                    king_outcome="defended",
                    previous_king_start_epoch=previous_start,
                    reward_constants=constants,
                ),
                hash_field="reward_basis_hash",
            )
            prior = [{
                "effective_reward_epoch": 100,
                "reward_basis_doc": prior_basis,
                "reward_activated_at": "2026-09-01T00:00:01Z",
                "configuration_doc": {"mode": "live", "baseline_hotkey": baseline},
            }]

        captured = {}

        reward_queries = []

        class Store:
            @staticmethod
            def pending_promotions(**kwargs):
                reward_queries.append(("promotions", kwargs))
                return []

            @staticmethod
            def published_reward_bases(**kwargs):
                reward_queries.append(("bases", kwargs))
                return prior

            @staticmethod
            def activate_reward(_round_id, basis, key):
                captured.update(basis=basis, key=key)
                return {"status": "activated"}

        service = object.__new__(ArenaService)
        service._store = Store()
        service._signer = signer
        service._signer_lock = threading.Lock()
        service._config = SimpleNamespace(
            mode="live",
            network_name="test",
            netuid=401,
            chain=SimpleNamespace(current_settlement_epoch=lambda: 100),
            reward_signer_factory=None,
        )
        service._round = lambda _round_id: {
            "round_id": "arena-2026-09-02",
            "status": "published",
            "reward_activated_at": None,
            "configuration_doc": {
                "mode": "live",
                "rewards_enabled": True,
                "baseline_hotkey": baseline,
                "reward_constants": constants,
            },
            "publication_doc": {
                "published_at": "2026-09-02T00:00:00Z",
                "king_decision": {
                    "outcome": "crowned" if daily_hotkey else "no_king",
                    "king_hotkey": daily_hotkey,
                },
            },
        }
        assert service.activate_reward("arena-2026-09-02")["status"] == "activated"
        assert reward_queries == [
            (
                "promotions",
                {"network_name": "test", "netuid": 401, "limit": 1},
            ),
            (
                "bases",
                {
                    "mode": "live",
                    "network_name": "test",
                    "netuid": 401,
                    "limit": 200,
                },
            ),
        ]
        return captured["basis"]

    assert activate()["king_outcome"] == "no_king"
    defended = activate(previous_hotkey=miner_a)
    assert (defended["king_outcome"], defended["king_hotkey"], defended["king_start_epoch"]) == (
        "defended", miner_a, 80,
    )
    same = activate(daily_hotkey=miner_a, previous_hotkey=miner_a)
    assert (same["king_outcome"], same["king_start_epoch"]) == ("defended", 80)
    changed = activate(daily_hotkey=miner_b, previous_hotkey=miner_a)
    assert (changed["king_outcome"], changed["king_hotkey"], changed["king_start_epoch"]) == (
        "crowned", miner_b, 101,
    )
    # A historical organizer-baseline basis is never carried or paid.
    assert activate(previous_hotkey=baseline)["king_outcome"] == "no_king"


def test_round_selection_and_direct_access_are_scoped_to_service_mode():
    live = {
        "round_id": "live-round", "status": "open", "created_at": "2026-09-02T00:00:00Z",
        "configuration_doc": {"mode": "live"},
    }
    shadow = {
        "round_id": "shadow-round", "status": "open", "created_at": "2026-09-03T00:00:00Z",
        "configuration_doc": {"mode": "shadow"},
    }

    class Store:
        @staticmethod
        def list_rounds(**_kwargs):
            return [shadow, live]

        @staticmethod
        def get_round(round_id):
            return {"live-round": live, "shadow-round": shadow}.get(round_id)

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(mode="live")

    assert service.current_round()["round_id"] == "live-round"
    assert service.open_round()["round_id"] == "live-round"
    assert service.active_rounds() == [
        {"round_id": "live-round", "status": "open", "schedule": {}}
    ]
    with pytest.raises(ServiceError, match="round_mode_mismatch"):
        service._round("shadow-round")


def test_runtime_round_pin_rejects_alien_shadow_rounds_and_scopes_discovery():
    pinned = {
        "round_id": "arena-2026-09-07-pinned",
        "status": "open",
        "created_at": "2026-09-07T00:00:00Z",
        "configuration_doc": {
            "mode": "shadow",
            "schedule": {"submission_cutoff": "2026-09-08T00:00:00Z"},
        },
    }
    alien = {
        "round_id": "arena-2026-09-07-alien",
        "status": "published",
        "created_at": "2026-09-07T01:00:00Z",
        "configuration_doc": {"mode": "shadow", "schedule": {}},
    }
    rows = {row["round_id"]: row for row in (pinned, alien)}

    class Store:
        get_round_calls = []

        @classmethod
        def get_round(cls, round_id):
            cls.get_round_calls.append(round_id)
            return rows.get(round_id)

        @staticmethod
        def list_rounds(**_kwargs):
            raise AssertionError("a pinned service must not scan shared round history")

        @staticmethod
        def get_run(_run_id):
            return {
                "run_id": "alien-run",
                "round_id": alien["round_id"],
                "assignment_id": "assignment-1",
                "attempt": 1,
                "icp_position": 0,
                "miner_hotkey": "5" * 48,
                "submission_id": "alien-submission",
                "stage": 1,
                "kind": "execute",
            }

        @staticmethod
        def get_submission(_submission_id):
            return {
                "submission_id": "alien-submission",
                "round_id": alien["round_id"],
                "status": "frozen",
            }

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(mode="shadow", pinned_round_id=pinned["round_id"])

    assert service.current_round() == pinned
    assert service.open_round() == pinned
    assert service.active_rounds() == [
        {
            "round_id": pinned["round_id"],
            "status": "open",
            "schedule": {"submission_cutoff": "2026-09-08T00:00:00Z"},
        }
    ]
    assert service.latest_published_round() is None
    assert set(Store.get_round_calls) == {pinned["round_id"]}

    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service._round(alien["round_id"])
    service.validate_request = lambda *_args, **_kwargs: {
        "round_id": alien["round_id"],
        "hotkey": "5" * 48,
    }
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.handle_claim({})
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service._run_context("alien-run", "lease-token")
    service._broker_for = lambda _round_id: (_ for _ in ()).throw(
        AssertionError("alien run must not reach provider credentials")
    )
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.handle_provider(
            "alien-run",
            "lease-token",
            {
                "operation_id": "openrouter.chat",
                "parameters": {},
                "timeout_ms": 1_000,
                "action_sequence": 0,
            },
        )
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.handle_source("alien-run", "lease-token")
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.public_results(alien["round_id"], "alien-submission")
    with pytest.raises(ServiceError, match="round_scope_mismatch"):
        service.submission_status("alien-submission")


def test_round_discovery_filters_before_limits_and_pages_every_active_round():
    def row(round_id, status, mode, created_at, **extra):
        return {
            "round_id": round_id,
            "status": status,
            "created_at": created_at,
            "configuration_doc": {
                "mode": mode,
                "schedule": {"submission_cutoff": created_at},
                **extra.pop("configuration", {}),
            },
            **extra,
        }

    live_active = [
        row(
            "live-active-%02d" % index,
            "open" if index == 44 else "committed",
            "live",
            "2026-09-%02dT00:00:00Z" % (index + 1),
        )
        for index in range(45)
    ]
    live_published = [
        row(
            "live-published-new",
            "published",
            "live",
            "2026-12-31T00:00:00Z",
            configuration={"rewards_enabled": True},
            reward_activated_at=None,
        ),
        row(
            "live-published-old",
            "published",
            "live",
            "2026-12-30T00:00:00Z",
            configuration={"rewards_enabled": True},
            reward_activated_at=None,
        ),
    ]
    rows = (
        [
            row(
                "shadow-published-%03d" % index,
                "published",
                "shadow",
                "2027-01-%03dT00:00:00Z" % index,
            )
            for index in range(201)
        ]
        + [
            row(
                "live-terminal-%02d" % index,
                "cancelled",
                "live",
                "2026-12-%02dT00:00:00Z" % (index + 1),
            )
            for index in range(25)
        ]
        + live_published
        + list(reversed(live_active))
    )

    class Store:
        calls = []

        @classmethod
        def list_rounds(
            cls,
            *,
            status=None,
            statuses=None,
            mode=None,
            limit=100,
            offset=None,
            **_kwargs,
        ):
            cls.calls.append(
                {
                    "status": status,
                    "statuses": statuses,
                    "mode": mode,
                    "limit": limit,
                    "offset": offset,
                }
            )
            selected = list(rows)
            if status is not None:
                selected = [item for item in selected if item["status"] == status]
            if statuses is not None:
                selected = [item for item in selected if item["status"] in statuses]
            if mode is not None:
                selected = [
                    item
                    for item in selected
                    if item["configuration_doc"]["mode"] == mode
                ]
            start = int(offset or 0)
            return selected[start : start + int(limit)]

        @staticmethod
        def get_round(round_id):
            return next((item for item in rows if item["round_id"] == round_id), None)

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(mode="live")
    service._round = Store.get_round

    assert service.current_round()["round_id"] == "live-active-44"
    assert service.open_round()["round_id"] == "live-active-44"
    assert service.latest_published_round()["round_id"] == "live-published-new"
    Store.calls.clear()

    active = service.active_rounds()
    assert [item["round_id"] for item in active] == [
        "live-active-%02d" % index for index in range(45)
    ]
    assert [call["offset"] for call in Store.calls] == [0, 20, 40]
    assert all(call["mode"] == "live" for call in Store.calls)
    assert all(
        "published" not in call["statuses"] and "cancelled" not in call["statuses"]
        for call in Store.calls
    )

    activated = []
    service.activate_reward = (
        lambda round_id: activated.append(round_id) or {"status": "activated"}
    )
    assert service.activate_pending_rewards() == {"status": "ok", "activated": 2}
    assert activated == ["live-published-old", "live-published-new"]
    assert Store.calls[-1] == {
        "status": "published",
        "statuses": None,
        "mode": "live",
        "limit": 200,
        "offset": None,
    }


def test_public_round_does_not_expose_source_transport_fields():
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-02",
        "status": "committed",
        "configuration_doc": {"mode": "shadow"},
        "participants": [
            {
                "submission_id": "sub-random",
                "miner_hotkey": "5" + "A" * 47,
                "is_king": False,
                "source_ref": "arena/private/source.tar.gz",
                "source_size_bytes": 100,
            }
        ],
    }
    view = service.public_round("arena-2026-09-02")
    assert view["participants"] == [
        {
            "submission_id": "sub-random",
            "miner_hotkey": "5" + "A" * 47,
            "is_baseline": False,
        }
    ]
    assert "configuration" not in view


def test_public_current_discovers_latest_published_round_without_rewards():
    published = {
        "round_id": "arena-2026-09-02",
        "status": "published",
        "published_at": "2026-09-03T08:00:00Z",
        "configuration_doc": {
            "mode": "live",
            "rewards_enabled": False,
            "runner_hotkeys": ["private-runner"],
        },
        "publication_doc": {"private": "must-not-leak"},
        "reward_basis_doc": None,
    }
    newer_open = {
        "round_id": "arena-2026-09-03",
        "status": "open",
        "created_at": "2026-09-03T09:00:00Z",
        "configuration_doc": {
            "mode": "live",
            "schedule": {"submission_cutoff": "2026-09-04T00:00:00Z"},
        },
    }

    class Store:
        @staticmethod
        def list_rounds(*, status=None, **_kwargs):
            return [published] if status == "published" else [newer_open, published]

        @staticmethod
        def published_reward_bases(**_kwargs):
            return []

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        mode="live",
        chain=SimpleNamespace(
            current_settlement_epoch=lambda: (_ for _ in ()).throw(RuntimeError())
        ),
    )
    service.public_reward_basis = lambda _epoch: None

    current = service.public_current()

    assert current["published_round"] == {
        "round_id": "arena-2026-09-02",
        "status": "published",
        "published_at": "2026-09-03T08:00:00Z",
    }
    assert current["open_round"]["round_id"] == "arena-2026-09-03"
    assert current["king"] is None
    assert "private" not in json.dumps(current["published_round"])
    assert "runner_hotkeys" not in json.dumps(current["published_round"])


def test_public_current_has_no_published_round_before_publication():
    class Store:
        @staticmethod
        def list_rounds(*, status=None, **_kwargs):
            if status == "published":
                return []
            return []

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        mode="shadow",
        chain=SimpleNamespace(
            current_settlement_epoch=lambda: (_ for _ in ()).throw(RuntimeError())
        ),
    )

    assert service.public_current()["published_round"] is None


def test_public_views_never_serialize_source_or_private_runtime_fields():
    source_ref = "arena/private/source-secret.tar.gz"
    hotkey = "5" + "A" * 47
    schedule = {"submission_cutoff": "2026-09-02T01:00:00Z"}
    raw_configuration = {
        "mode": "shadow",
        "schedule": schedule,
        "runner_hotkeys": ["private-runner"],
        "scorer_image_reference": "private.example/scorer@sha256:" + "a" * 64,
        "source_ref": source_ref,
    }
    published = {
        "round_id": "arena-2026-09-02",
        "status": "published",
        "configuration_doc": raw_configuration,
        "participants": [
            {
                "submission_id": "sub-random",
                "miner_hotkey": hotkey,
                "is_king": False,
                "source_ref": source_ref,
                "source_size_bytes": 321,
            }
        ],
        "publication_doc": {
            "participants": [
                {"submission_id": "sub-random", "miner_hotkey": hotkey, "is_king": False}
            ],
            "stage1_ranking": [],
            "final_ranking": [],
        },
    }
    active = dict(published, round_id="arena-2026-09-03", status="open")

    class Store:
        @staticmethod
        def list_rounds(**_kwargs):
            return [active]

        @staticmethod
        def list_runs(*_args, **_kwargs):
            return []

        @staticmethod
        def get_submission(_submission_id):
            return {
                "miner_hotkey": hotkey,
                "is_king": False,
                "source_ref": source_ref,
                "source_size_bytes": 321,
            }

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        mode="shadow",
        chain=SimpleNamespace(
            current_settlement_epoch=lambda: (_ for _ in ()).throw(RuntimeError())
        ),
    )
    service._round = lambda _round_id: published
    service.latest_published_round = lambda: None
    service._objects = SimpleNamespace(get=lambda _ref: b"{}")

    current = service.public_current()
    round_view = service.public_round("arena-2026-09-02")
    results = service.public_results("arena-2026-09-02", "sub-random")
    serialized = json.dumps([current, round_view, results], sort_keys=True)
    for private_value in (
        source_ref,
        "source_size_bytes",
        "private-runner",
        "private.example/scorer",
    ):
        assert private_value not in serialized
    assert current["open_round"] == {
        "round_id": "arena-2026-09-03",
        "status": "open",
        "schedule": schedule,
    }
    assert current["published_round"] is None
    assert round_view["participants"] == [
        {
            "submission_id": "sub-random",
            "miner_hotkey": hotkey,
            "is_baseline": False,
        }
    ]


@pytest.mark.parametrize("submission_id", ["sub-unknown", "sub-current-open-round"])
def test_public_results_refuse_ids_outside_the_published_round(submission_id):
    published = {
        "round_id": "arena-2026-09-02",
        "status": "published",
        "publication_doc": {
            "participants": [
                {
                    "submission_id": "sub-published",
                    "miner_hotkey": "5" + "A" * 47,
                    "is_baseline": False,
                }
            ],
            "stage1_ranking": [],
            "final_ranking": [],
        },
    }

    class Store:
        @staticmethod
        def list_runs(*_args, **_kwargs):
            pytest.fail("a nonparticipant must be rejected before run lookup")

        @staticmethod
        def get_submission(_submission_id):
            # This simulates a known submission in the current open round. The
            # global submission row must not make it part of this publication.
            return {
                "round_id": "arena-2026-09-03",
                "submission_id": "sub-current-open-round",
                "miner_hotkey": "5" + "B" * 47,
                "is_king": False,
            }

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: published

    with pytest.raises(ServiceError) as caught:
        service.public_results(published["round_id"], submission_id)
    assert caught.value.status == 404 and caught.value.code == "submission_missing"


def test_public_results_take_valid_identity_from_the_round_publication():
    hotkey = "5" + "C" * 47
    published = {
        "round_id": "arena-2026-09-02",
        "status": "published",
        "publication_doc": {
            "participants": [
                {
                    "submission_id": "sub-published",
                    "miner_hotkey": hotkey,
                    "is_baseline": True,
                }
            ],
            "stage1_ranking": [],
            "final_ranking": [],
        },
    }

    class Store:
        @staticmethod
        def list_runs(_round_id, **filters):
            assert filters == {"submission_id": "sub-published", "kind": "execute"}
            return []

        @staticmethod
        def get_submission(_submission_id):
            pytest.fail("public identity must not come from the global submission row")

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: published

    result = service.public_results(published["round_id"], "sub-published")
    assert result == {
        "round_id": published["round_id"],
        "submission_id": "sub-published",
        "submission": {"miner_hotkey": hotkey, "is_baseline": True},
        "outputs": {},
        "run_results": [],
        "scores": {"stage_1": [], "stage_2": []},
        "public_icp_status": "pending",
        "public_icp_count": 0,
        "submission_scores": {"stage_1": None, "final": None},
    }


def _with_completed_public_baseline(service):
    """Exercise evidence handling after the real 7/3 split is available."""
    original_round = service._round
    original_list_runs = service._store.list_runs
    baseline_id = "baseline-public-evidence"

    def round_with_baseline(round_id):
        row = original_round(round_id)
        return {
            **row,
            "participants": [
                *(row.get("participants") or []),
                {"submission_id": baseline_id, "is_king": True},
            ],
        }

    def list_runs(round_id, **filters):
        if filters == {"submission_id": baseline_id, "kind": "execute"}:
            return [
                {
                    "submission_id": baseline_id, "kind": "execute",
                    "icp_position": position, "attempt": 1,
                    "per_icp_score": float(position), "terminal_cause": "accepted",
                }
                for position in range(contracts.BENCHMARK_ICP_COUNT)
            ]
        return original_list_runs(round_id, **filters)

    service._round = round_with_baseline
    service._store.list_runs = list_runs


def _stored_public_output(name):
    return validate_output_document({
        "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
        "companies": [
            {
                "company_name": name,
                "company_website": "https://visible.example.com",
                "company_linkedin": "",
                "industry": "Software",
                "employee_count": "51-200",
                "company_stage": "Series A",
                "country": "United States",
                "state": "",
                "fit_summary": "The company matches the ICP.",
                "fit_evidence_urls": ["https://visible.example.com/about"],
                "intent_signals": [
                    {
                        "matched_icp_signal": 0,
                        "description": "Raised a round",
                        "date": "2026-08-01",
                        "why_now": "The funding makes outreach timely.",
                        "url": "https://visible.example.com/news",
                        "snippet": "Funding announced",
                    }
                ],
            }
        ],
    })


def test_cancelled_results_return_scoped_outputs_scores_and_redacted_judge_evidence():
    hotkey = "5" + "D" * 47
    round_id = "arena-2026-09-07"
    execute = {
        "run_id": "execute-target-0",
        "submission_id": "sub-target",
        "kind": "execute",
        "stage": 1,
        "icp_position": 0,
        "status": "accepted",
        "terminal_cause": "accepted",
        "output_ref": "arena/target-output.json",
        "per_icp_score": 72.5,
        "result_doc": {"terminal_status": "accepted"},
    }
    score = {
        "run_id": "score-target-0",
        "scored_run_id": execute["run_id"],
        "submission_id": "sub-target",
        "kind": "score",
        "stage": 1,
        "icp_position": 0,
        "attempt": 1,
        "status": "accepted",
        "terminal_cause": "accepted",
        "output_ref": "arena/target-score.json",
    }
    round_row = {
        "round_id": round_id,
        "status": "cancelled",
        "cancel_reason": "scoring_incomplete",
        "participants": [
            {
                "submission_id": "sub-target",
                "miner_hotkey": hotkey,
                "is_king": False,
                "source_ref": "private-target-source",
            },
            {
                "submission_id": "sub-other",
                "miner_hotkey": "5" + "E" * 47,
                "is_king": False,
                "source_ref": "private-other-source",
            },
        ],
        "publication_doc": None,
    }

    class Store:
        @staticmethod
        def list_runs(requested_round, **filters):
            assert requested_round == round_id
            assert filters["submission_id"] == "sub-target"
            return [execute] if filters["kind"] == "execute" else [score]

    judge_document = scoring.build_scoring_output(
        execute["run_id"],
        [
            {
                "final_score": 72.5,
                "failure_reason": "",
                "proof_quote": "private judge payload",
                "intent_signals_detail": [],
                "verifier_gate_receipts": [],
            }
        ],
    )
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: round_row
    service._store = Store()
    output_document = _stored_public_output("Visible Company")
    service._objects = SimpleNamespace(
        get=lambda ref: json.dumps(output_document).encode(),
        get_bounded=lambda ref, limit: json.dumps(
            output_document if ref == execute["output_ref"] else judge_document
        ).encode(),
    )

    _with_completed_public_baseline(service)

    with pytest.raises(ServiceError) as caught:
        service.public_results(round_id, "sub-target")
    assert caught.value.status == 403 and caught.value.code == "results_not_public"
    return

    assert result["round_status"] == "cancelled"
    assert result["cancel_reason"] == "scoring_incomplete"
    assert result["incomplete"] is True
    assert result["outputs"] == {
        execute["run_id"]: output_document
    }
    assert result["scores"]["stage_1"] == [
        {
            "run_id": execute["run_id"],
            "icp_position": 0,
            "per_icp_score": 72.5,
        }
    ]
    assert result["submission_scores"] == {"stage_1": None, "final": None}
    assert result["execution_jobs"] == [
        {
            "run_id": execute["run_id"],
            "stage": 1,
            "icp_position": 0,
            "status": "accepted",
            "terminal_cause": "accepted",
            "output_status": "available",
        }
    ]
    assert result["judge_jobs"] == [
        {
            "run_id": score["run_id"],
            "scored_run_id": execute["run_id"],
            "stage": 1,
            "icp_position": 0,
            "status": "accepted",
            "terminal_cause": "accepted",
            "evidence_status": "available",
        }
    ]
    assert result["judge_evidence"][0]["per_icp_score"] == 72.5
    assert result["judge_evidence"][0]["breakdowns"] == [
        {
            "final_score": 72.5,
            "failure_reason": "",
            "intent_signals_detail": [],
            "verifier_gate_receipts": [],
        }
    ]
    serialized = json.dumps(result, sort_keys=True)
    for private_value in (
        "private-target-source",
        "private-other-source",
        "private judge payload",
        "sub-other",
    ):
        assert private_value not in serialized
    for unpublished_field in ("final_ranking", "king_decision", "reward_basis"):
        assert unpublished_field not in result


@pytest.mark.parametrize("cause", ["judge_timeout", "lease_expired", "stage_closed"])
def test_cancelled_results_keep_partial_jobs_incomplete_and_report_safe_judge_failure(cause):
    round_id = "arena-2026-09-07-partial"
    execute_runs = [
        {
            "run_id": "execute-complete",
            "submission_id": "sub-partial",
            "kind": "execute",
            "stage": 1,
            "icp_position": 0,
            "status": "accepted",
            "output_ref": "arena/complete.json",
            "per_icp_score": None,
            "result_doc": {"terminal_status": "accepted"},
        },
        {
            "run_id": "execute-pending",
            "submission_id": "sub-partial",
            "kind": "execute",
            "stage": 1,
            "icp_position": 1,
            "status": "pending",
            "output_ref": None,
            "per_icp_score": None,
            "result_doc": None,
        },
    ]
    failed_score = {
        "run_id": "score-failed",
        "scored_run_id": "execute-complete",
        "submission_id": "sub-partial",
        "kind": "score",
        "stage": 1,
        "icp_position": 0,
        "attempt": 2,
        "status": "failed",
        "terminal_cause": cause,
        "output_ref": None,
        "result_doc": {"unsafe_detail": "must stay private"},
    }

    class Store:
        @staticmethod
        def list_runs(_round_id, **filters):
            return execute_runs if filters["kind"] == "execute" else [failed_score]

    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {
        "round_id": round_id,
        "status": "cancelled",
        "cancel_reason": "scoring_incomplete",
        "publication_doc": {
            "stage1_ranking": [{"submission_id": "sub-partial", "stage1_score": 80}],
            "final_ranking": [{"submission_id": "sub-partial", "final_score": 90}],
        },
        "participants": [
            {
                "submission_id": "sub-partial",
                "miner_hotkey": "5" + "F" * 47,
                "is_king": False,
            }
        ],
    }
    service._store = Store()
    output_document = _stored_public_output("Partial")
    service._objects = SimpleNamespace(
        get=lambda _ref: json.dumps(output_document).encode(),
        get_bounded=lambda ref, _limit: json.dumps(output_document).encode(),
    )

    _with_completed_public_baseline(service)

    with pytest.raises(ServiceError) as caught:
        service.public_results(round_id, "sub-partial")
    assert caught.value.status == 403 and caught.value.code == "results_not_public"
    return

    assert result["incomplete"] is True
    assert list(result["outputs"]) == ["execute-complete"]
    assert [job["output_status"] for job in result["execution_jobs"]] == [
        "available",
        "unavailable",
    ]
    assert result["scores"] == {"stage_1": [], "stage_2": []}
    assert result["submission_scores"] == {"stage_1": None, "final": None}
    assert result["judge_evidence"] == []
    assert result["judge_jobs"] == [
        {
            "run_id": "score-failed",
            "scored_run_id": "execute-complete",
            "stage": 1,
            "icp_position": 0,
            "status": "failed",
            "terminal_cause": cause,
            "evidence_status": "unavailable",
        }
    ]
    assert "unsafe_detail" not in json.dumps(result)


def test_cancelled_results_preserve_good_evidence_when_other_artifacts_fail():
    round_id = "arena-2026-09-07-mixed-evidence"
    execute_runs = [
        {
            "run_id": "execute-%s" % label,
            "submission_id": "sub-mixed",
            "kind": "execute",
            "stage": 1,
            "icp_position": position,
            "status": "accepted",
            "output_ref": "arena/output-%s.json" % label,
            "per_icp_score": None,
            "result_doc": {"terminal_status": "accepted"},
        }
        for position, label in enumerate(("good", "missing", "invalid"))
    ]
    score_runs = [
        {
            "run_id": "score-%s" % label,
            "scored_run_id": "execute-%s" % label,
            "submission_id": "sub-mixed",
            "kind": "score",
            "stage": 1,
            "icp_position": position,
            "attempt": 1,
            "status": "accepted",
            "terminal_cause": "accepted",
            "output_ref": "arena/score-%s.json" % label,
        }
        for position, label in enumerate(("good", "missing", "invalid"))
    ]

    class Store:
        @staticmethod
        def list_runs(_round_id, **filters):
            return execute_runs if filters["kind"] == "execute" else score_runs

    good = scoring.build_scoring_output(
        "execute-good", [{"final_score": 50.0, "proof_quote": "private"}]
    )

    def get_bounded(ref, _limit):
        if "/output-" in ref:
            if ref.endswith("output-invalid.json"):
                return b'{"secret": "raw execute secret"}'
            return json.dumps(
                {
                    "schema_version": contracts.OUTPUT_DOCUMENT_SCHEMA_VERSION,
                    "companies": [],
                }
            ).encode()
        if ref.endswith("score-good.json"):
            return json.dumps(good).encode()
        if ref.endswith("score-missing.json"):
            raise OSError("private bucket diagnostic")
        return b"not-json"

    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {
        "round_id": round_id,
        "status": "cancelled",
        "cancel_reason": "scoring_incomplete",
        "participants": [
            {
                "submission_id": "sub-mixed",
                "miner_hotkey": "5" + "I" * 47,
                "is_king": False,
            }
        ],
    }
    service._store = Store()
    service._objects = SimpleNamespace(
        get=lambda _ref: b'{"companies": []}',
        get_bounded=get_bounded,
    )

    _with_completed_public_baseline(service)

    with pytest.raises(ServiceError) as caught:
        service.public_results(round_id, "sub-mixed")
    assert caught.value.status == 403 and caught.value.code == "results_not_public"
    return

    assert [job["evidence_status"] for job in result["judge_jobs"]] == [
        "available",
        "unavailable",
        "invalid",
    ]
    assert [item["scored_run_id"] for item in result["judge_evidence"]] == [
        "execute-good"
    ]
    assert [job["output_status"] for job in result["execution_jobs"]] == [
        "available",
        "available",
        "invalid",
    ]
    assert set(result["outputs"]) == {"execute-good", "execute-missing"}
    serialized = json.dumps(result, sort_keys=True)
    assert "private bucket diagnostic" not in serialized
    assert "proof_quote" not in serialized
    assert "raw execute secret" not in serialized


def test_cancelled_results_refuse_an_unrelated_submission_before_run_lookup():
    class Store:
        @staticmethod
        def list_runs(*_args, **_kwargs):
            pytest.fail("an unrelated submission must not reach run lookup")

        @staticmethod
        def get_submission(_submission_id):
            pytest.fail("global submission identity must not authorize results")

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-07",
        "status": "cancelled",
        "cancel_reason": "operator",
        "participants": [
            {
                "submission_id": "sub-frozen",
                "miner_hotkey": "5" + "G" * 47,
                "is_king": False,
            }
        ],
    }

    with pytest.raises(ServiceError) as caught:
        service.public_results("arena-2026-09-07", "sub-other-round")
    assert caught.value.status == 403 and caught.value.code == "results_not_public"


@pytest.mark.parametrize("status", ["open", "stage1", "stage2_judged", "scored", "cancelled"])
def test_public_results_keep_nonterminal_rounds_private(status):
    service = object.__new__(ArenaService)
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-08",
        "status": status,
        "participants": [
            {
                "submission_id": "sub-live",
                "miner_hotkey": "5" + "H" * 47,
                "is_king": False,
            }
        ],
    }
    service._store = SimpleNamespace(
        list_runs=lambda *_args, **_kwargs: pytest.fail(
            "live round must be rejected before run lookup"
        )
    )

    with pytest.raises(ServiceError) as caught:
        service.public_results("arena-2026-09-08", "sub-live")
    assert caught.value.status == 403 and caught.value.code == "results_not_public"


def test_source_download_requires_the_active_execute_lease():
    token = "a" * 64
    payload = b"private source bytes"
    submission = {
        "status": "frozen",
        "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        "source_size_bytes": len(payload),
    }
    run = {
        "run_id": "run-1",
        "submission_id": "sub-1",
        "kind": "execute",
        "status": "leased",
        "lease_token_hash": hash_lease_token(token),
        "lease_expires_at": "2026-09-02T01:10:00+00:00",
    }
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        get_run=lambda _run_id: run,
        get_submission=lambda _submission_id: submission,
    )
    service._objects = SimpleNamespace(get_bounded=lambda *_args: payload)
    service._clock = lambda: datetime(2026, 9, 2, 1, 0, tzinfo=timezone.utc)

    assert service.handle_source("run-1", token) == payload
    # PostgreSQL removes trailing fractional zeros, so PostgREST can return a
    # five-digit fraction that Python 3.9's datetime.fromisoformat rejects.
    run["lease_expires_at"] = "2026-09-02T01:10:00.12345+00:00"
    assert service.handle_source("run-1", token) == payload
    with pytest.raises(ServiceError, match="lease_invalid"):
        service.handle_source("run-1", "b" * 64)
    run["status"] = "accepted"
    with pytest.raises(ServiceError, match="lease_inactive"):
        service.handle_source("run-1", token)


def test_open_stage_preserves_legacy_finalists_but_new_policy_runs_everyone_twice():
    calls = []

    class Store:
        def open_stage(self, round_id, stage, participants, positions):
            calls.append((round_id, stage, participants, positions))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    row = {
        "participants": [
            {"submission_id": "king", "miner_hotkey": "hk", "is_king": True},
            {"submission_id": "c1", "miner_hotkey": "h1", "is_king": False},
            {"submission_id": "c2", "miner_hotkey": "h2", "is_king": False},
        ],
        "finalists": ["c2"],
    }
    service._round = lambda _round_id: row
    service.benchmark_icps = lambda _round_id: [{} for _ in range(contracts.BENCHMARK_ICP_COUNT)]

    service.open_stage("arena-2026-09-02", 1)
    service.open_stage("arena-2026-09-02", 2)
    assert [row["submission_id"] for row in calls[0][2]] == ["king", "c1", "c2"]
    assert calls[0][3] == list(range(10))
    assert [row["submission_id"] for row in calls[1][2]] == ["king", "c2"]
    assert calls[1][3] == list(range(10, 20))
    row["icp_set_date"] = "2026-09-01"
    service.open_stage("arena-2026-09-02", 2)
    assert [item["submission_id"] for item in calls[2][2]] == ["king", "c1", "c2"]
    assert calls[2][3] == list(range(10, 20))


def test_first_round_baseline_is_read_from_the_frozen_round_configuration():
    configured = "5" + "A" * 47
    baseline = {
        "submission_id": "baseline-2026-09-02",
        "round_id": "arena-2026-09-02",
        "miner_hotkey": configured,
        "source_ref": "arena/arena-2026-09-02/sources/baseline-2026-09-02.tar.gz",
        "source_size_bytes": 100,
        "status": "accepted",
        "is_king": True,
    }
    updates = []

    class Store:
        @staticmethod
        def list_submissions(_round_id, status):
            return [baseline] if status == "accepted" else []

        @staticmethod
        def update_submission(round_id, submission_id, old, new, patch):
            updates.append((round_id, submission_id, old, new, patch))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-02",
        "configuration_doc": {
            "baseline_hotkey": configured,
            "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz",
            "max_challengers": 10,
        },
    }
    service._initial_baseline = lambda round_row: (
        baseline
        if round_row["configuration_doc"]["baseline_hotkey"] == configured
        else pytest.fail("baseline did not use the frozen round configuration")
    )

    participants = service.freeze_participants("arena-2026-09-02")
    assert len(participants) == 1 and participants[0]["submission_id"] == baseline["submission_id"]
    assert participants[0]["is_king"] is True
    assert updates == [("arena-2026-09-02", baseline["submission_id"], "accepted", "frozen", {"is_king": True})]


def test_each_new_day_uses_the_public_baseline_even_after_a_miner_won_yesterday():
    baseline_hotkey = "5" + "A" * 47
    winner_hotkey = "5" + "B" * 47
    baseline = {
        "submission_id": "baseline-2026-09-03",
        "round_id": "arena-2026-09-03",
        "miner_hotkey": baseline_hotkey,
        "source_ref": "arena/arena-2026-09-03/sources/baseline-2026-09-03.tar.gz",
        "source_size_bytes": 100,
        "status": "accepted",
        "is_king": True,
    }

    class Store:
        @staticmethod
        def list_submissions(_round_id, status):
            return [baseline] if status == "accepted" else []

        @staticmethod
        def update_submission(*_args):
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: {
        "round_id": "arena-2026-09-03",
        "configuration_doc": {
            "baseline_hotkey": baseline_hotkey,
            "max_challengers": 10,
        },
    }
    service._initial_baseline = lambda _round_row: baseline
    # Yesterday's result remains available for reward history. It must not
    # select today's threshold participant.
    service.latest_published_round = lambda: {
        "round_id": "arena-2026-09-02",
        "king_hotkey": winner_hotkey,
        "king_outcome": "crowned",
    }

    participants = service.freeze_participants("arena-2026-09-03")
    assert [(row["miner_hotkey"], row["is_king"]) for row in participants] == [
        (baseline_hotkey, True)
    ]


def test_score_lease_uses_the_round_pinned_scorer_after_restart():
    runner = "5" * 48
    pinned_digest = "sha256:" + "a" * 64
    pinned_reference = "registry.example/lab/scorer@" + pinned_digest

    class Store:
        def claim_assignment(self, **_kwargs):
            return {"status": "leased", "kind": "score", "scored_run_id": "execute-1", "icp_position": 0}

        def get_run(self, _run_id):
            return {"output_ref": "out.json"}

    class Objects:
        def get(self, _ref):
            return json.dumps({"companies": []}).encode("utf-8")

    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = Objects()
    service._config = SimpleNamespace(
        chain=SimpleNamespace(hotkeys_owned_by_same_coldkey=lambda _hotkey: []),
        defaults=SimpleNamespace(
            scorer_image_digest="sha256:" + "b" * 64,
            scorer_image_reference="registry.example/wrong@sha256:" + "b" * 64,
        ),
    )
    configuration = {
        "runner_hotkeys": [runner],
        "runner_slot_ceiling": 8,
        "lease_ttl_seconds": 420,
        "scorer_image_digest": pinned_digest,
        "scorer_image_reference": pinned_reference,
        "scorer_policy": {"policy": "pinned"},
    }
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": runner, "body": {"declared_parallelism": 1}, "request_id": "0" * 32, "signature": "sig"},
        {"round_id": "arena-2026-09-02", "status": "stage1_scoring", "configuration_doc": configuration},
    )
    service._lease_token = lambda _validated: "token"
    service.benchmark_icps = lambda _round_id: [{}]

    lease = service.handle_claim({})
    assert (lease["image_digest"], lease["image_reference"]) == (pinned_digest, pinned_reference)


def _runner_claim_service(*, network_name, hotkeys, validator_permit, active=None):
    runner = "5" * 48
    snapshot_values = {"hotkeys": hotkeys}
    if validator_permit is not None:
        snapshot_values["validator_permit"] = validator_permit
    if active is not None:
        snapshot_values["active"] = active
    snapshot = SimpleNamespace(**snapshot_values)
    chain = SimpleNamespace(
        metagraph=lambda: snapshot,
        hotkeys_owned_by_same_coldkey=lambda _hotkey: [],
    )
    service = object.__new__(ArenaService)
    service._store = SimpleNamespace(
        claim_assignment=lambda **_kwargs: {"status": "empty"}
    )
    service._config = SimpleNamespace(network_name=network_name, chain=chain)
    service._request_round = lambda *_args, **_kwargs: (
        {
            "hotkey": runner,
            "body": {"declared_parallelism": 1},
            "request_id": "0" * 32,
            "signature": "sig",
        },
        {
            "round_id": "arena-2026-09-05-testnet",
            "status": "stage1",
            "configuration_doc": {
                "runner_hotkeys": [runner],
                "runner_slot_ceiling": 8,
                "lease_ttl_seconds": 420,
            },
        },
    )
    service._lease_token = lambda _validated: "token"
    return service


@pytest.mark.parametrize(
    ("hotkeys", "validator_permit", "expected_code"),
    [
        (["5" + "A" * 47], [True], "runner_hotkey_unregistered"),
        (["5" * 48], [False], "runner_validator_permit_required"),
        (["5" * 48], None, "runner_validator_permit_unavailable"),
        (["5" * 48], [], "runner_validator_permit_unavailable"),
        (["5" * 48], ["false"], "runner_validator_permit_unavailable"),
    ],
)
def test_test_network_claim_requires_registered_validator_permit(
    hotkeys, validator_permit, expected_code
):
    service = _runner_claim_service(
        network_name="test",
        hotkeys=hotkeys,
        validator_permit=validator_permit,
    )

    with pytest.raises(ServiceError) as rejected:
        service.handle_claim({})

    assert rejected.value.code == expected_code


def test_test_network_claim_accepts_permitted_inactive_validator():
    service = _runner_claim_service(
        network_name="test",
        hotkeys=["5" * 48],
        validator_permit=[True],
        active=[False],
    )

    assert service.handle_claim({}) == {"status": "empty"}


def test_finney_claim_does_not_add_the_testnet_validator_permit_gate():
    service = _runner_claim_service(
        network_name="finney",
        hotkeys=["5" * 48],
        validator_permit=[False],
    )

    assert service.handle_claim({}) == {"status": "empty"}


def test_execute_lease_uses_private_source_and_the_common_trusted_python_image():
    runner = "5" * 48
    digest = "sha256:" + "a" * 64
    reference = "registry.example/lab/scorer@" + digest
    participant = {
        "submission_id": "sub-1",
        "miner_hotkey": "5" + "A" * 47,
        "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        "source_size_bytes": 123,
        "is_king": False,
    }

    class Store:
        @staticmethod
        def claim_assignment(**_kwargs):
            return {
                "status": "leased",
                "kind": "execute",
                "submission_id": "sub-1",
                "icp_position": 0,
            }

    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        chain=SimpleNamespace(hotkeys_owned_by_same_coldkey=lambda _hotkey: [])
    )
    service._request_round = lambda *_args, **_kwargs: (
        {
            "hotkey": runner,
            "body": {"declared_parallelism": 1},
            "request_id": "0" * 32,
            "signature": "sig",
        },
        {
            "round_id": "arena-2026-09-02",
            "status": "stage1",
            "participants": [participant],
            "evaluation_date": "2026-09-02",
            "configuration_doc": {
                "runner_hotkeys": [runner],
                "runner_slot_ceiling": 8,
                "lease_ttl_seconds": 420,
                "scorer_image_digest": digest,
                "scorer_image_reference": reference,
            },
        },
    )
    service._lease_token = lambda _validated: "token"
    service.benchmark_icps = lambda _round_id: [{}]

    lease = service.handle_claim({})
    assert (lease["image_digest"], lease["image_reference"]) == (digest, reference)
    assert {
        key: lease[key]
        for key in ("source_ref", "source_size_bytes")
    } == {
        key: participant[key]
        for key in ("source_ref", "source_size_bytes")
    }


def test_finalize_rejects_an_invalid_source_archive():
    row = {
        "submission_id": "sub-1",
        "round_id": "arena-2026-09-02",
        "miner_hotkey": "5" + "A" * 47,
        "source_ref": "arena/arena-2026-09-02/sources/sub-1.tar.gz",
        "source_size_bytes": 4,
        "status": "uploading",
    }
    updates = []
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._store = SimpleNamespace(
        get_submission=lambda _submission_id: row,
        update_submission=lambda *args: updates.append(args) or {"status": "ok"},
    )
    service._objects = SimpleNamespace(get_bounded=lambda *_args: b"nope")
    service._request_round = lambda *_args, **_kwargs: (
        {
            "hotkey": row["miner_hotkey"],
            "body": {
                "submission_id": row["submission_id"],
                "source_ref": row["source_ref"],
                "source_size_bytes": row["source_size_bytes"],
                "credentials": {
                    "openrouter_api_key": "sk-or-v1-" + "a" * 32,
                    "openrouter_management_key": "sk-or-v1-" + "b" * 32,
                    "deepline_api_key": "deepline-" + "c" * 32,
                },
            },
        },
        {
            "round_id": row["round_id"],
            "status": "open",
            "configuration_doc": {"schedule": _schedule()},
        },
    )

    with pytest.raises(ServiceError, match="source_archive_invalid"):
        service.handle_submission_finalize(row["submission_id"], {})
    assert updates == [
        (
            row["round_id"],
            row["submission_id"],
            "uploading",
            "rejected",
            {"rejection_rule": "source_archive_invalid"},
        )
    ]


def _daily_source_icps():
    return [
        {
            "icp_id": "icp_20260903_%03d" % index,
            "prompt": "ICP %d" % index,
        }
        for index in range(1, contracts.BENCHMARK_ICP_COUNT + 1)
    ]


def _daily_source_service(source):
    from tests.lab_arena.test_lab_arena_contracts import base_round_configuration

    class Objects:
        def __init__(self):
            self.values = {}

        def put(self, ref, data):
            self.values[ref] = bytes(data)

    class Store:
        def __init__(self):
            self.cancelled = []
            self.transitions = []

        def cancel_round(self, round_id, reason):
            self.cancelled.append((round_id, reason))

        def commit_round_v2(self, round_id, **patch):
            self.transitions.append((round_id, "open", "committed", patch))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = Objects()
    configuration = base_round_configuration()
    configuration["schedule"] = {
        key: (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            + timedelta(days=1)
        ).isoformat().replace("+00:00", "Z")
        for key, value in configuration["schedule"].items()
    }
    service._config = SimpleNamespace(
        daily_icp_source=source,
        defaults=SimpleNamespace(
            scorer_image_digest=configuration["scorer_image_digest"],
            scorer_image_reference=configuration["scorer_image_reference"],
        ),
    )
    service._clock = lambda: datetime(2026, 9, 3, tzinfo=timezone.utc)
    service._round = lambda _round_id: {
        "status": "open", "configuration_doc": configuration
    }
    service.freeze_participants = lambda _round_id: [{"submission_id": "baseline"}]
    return service


def test_benchmark_commit_uses_the_exact_daily_icps_in_source_order():
    expected = _daily_source_icps()
    calls = []

    def source(**kwargs):
        calls.append(kwargs)
        return {"status": "ready", "set_id": kwargs["set_id"], "icps": expected}

    service = _daily_source_service(source)
    result = service.commit_benchmark("arena-2026-09-03")

    assert result == {"status": "ok", "participants": 1}
    assert calls == [
        {
            "set_id": 20260902,
            "active_at": datetime(2026, 9, 3, tzinfo=timezone.utc),
        }
    ]
    stored = json.loads(
        service._objects.values["arena/arena-2026-09-03/benchmark.json"]
    )
    assert stored["icps"] == expected


def test_benchmark_commit_before_cutoff_does_not_read_or_freeze():
    service = _daily_source_service(
        lambda **_kwargs: pytest.fail("daily bank read before cutoff")
    )
    service._clock = lambda: datetime(
        2026, 9, 2, 23, 59, 59, tzinfo=timezone.utc
    )
    service.freeze_participants = lambda _round_id: pytest.fail(
        "participants froze before cutoff"
    )
    assert service.commit_benchmark("arena-2026-09-03") == {
        "status": "waiting",
        "round_status": "open",
    }


def test_unavailable_daily_set_retries_before_participants_freeze():
    service = _daily_source_service(
        lambda **_kwargs: {"status": "unavailable"}
    )
    service.freeze_participants = lambda _round_id: (_ for _ in ()).throw(
        AssertionError("participants froze before the daily set was ready")
    )

    assert service.commit_benchmark("arena-2026-09-03") == {
        "status": "retry",
        "reason": "daily_icp_set_not_ready",
        "set_id": 20260902,
    }
    assert service._store.cancelled == []


def test_unavailable_historical_bank_cancels_at_benchmark_deadline():
    service = _daily_source_service(
        lambda **_kwargs: {"status": "unavailable"}
    )
    service._clock = lambda: datetime(
        2026, 9, 3, 0, 30, tzinfo=timezone.utc
    )
    assert service.commit_benchmark("arena-2026-09-03") == {
        "status": "cancelled",
        "reason": "benchmark_data_invalid",
    }
    assert service._store.cancelled == [
        ("arena-2026-09-03", "benchmark_data_invalid")
    ]


def test_duplicate_daily_icp_ids_cancel_the_round_as_invalid():
    invalid = _daily_source_icps()
    invalid[-1] = dict(invalid[-1], icp_id=invalid[0]["icp_id"])
    service = _daily_source_service(
        lambda **kwargs: {
            "status": "ready",
            "set_id": kwargs["set_id"],
            "icps": invalid,
        }
    )

    assert service.commit_benchmark("arena-2026-09-03") == {
        "status": "cancelled",
        "reason": "benchmark_data_invalid",
    }
    assert service._store.cancelled == [
        ("arena-2026-09-03", "benchmark_data_invalid")
    ]

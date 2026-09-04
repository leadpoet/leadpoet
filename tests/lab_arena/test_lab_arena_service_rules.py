"""Focused service rules for the simple two-stage competition."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import contracts, images
from lab_arena import service as service_module
from lab_arena.service import ArenaService, ServiceError


def _schedule():
    return {
        "submission_open": "2026-09-02T00:00:00Z",
        "submission_cutoff": "2026-09-02T01:00:00Z",
    }


@pytest.mark.parametrize("moment", ["2026-09-01T23:59:59+00:00", "2026-09-02T01:00:00+00:00"])
def test_submission_requires_the_full_half_open_time_window(moment):
    service = object.__new__(ArenaService)
    service._clock = lambda: datetime.fromisoformat(moment)
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": "5" + "A" * 47, "body": {}},
        {"round_id": "arena-2026-09-02", "status": "open", "configuration_doc": {"schedule": _schedule()}},
    )
    with pytest.raises(ServiceError, match="submission_window_closed"):
        service.handle_submission({})


def test_submission_inside_the_window_registers_normally():
    class Store:
        @staticmethod
        def register_submission(*_args):
            return {"status": "registered"}

    hotkey = "5" + "A" * 47
    digest = "sha256:" + "a" * 64
    service = object.__new__(ArenaService)
    service._store = Store()
    service._clock = lambda: datetime(2026, 9, 2, 0, 30, tzinfo=timezone.utc)
    service._config = SimpleNamespace(chain=SimpleNamespace(uid_for_hotkey=lambda value: 1 if value == hotkey else None))
    service._request_round = lambda *_args, **_kwargs: (
        {"hotkey": hotkey, "body": {"image_reference": "registry.example/agent@" + digest, "consent": {"public_rerun": True}}},
        {"round_id": "arena-2026-09-02", "status": "open", "configuration_doc": {"schedule": _schedule()}},
    )
    assert service.handle_submission({})["status"] == "uploaded"


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
    service._config = SimpleNamespace(registry=object())
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


def test_reward_activation_processes_only_enabled_live_rounds_oldest_first():
    rows = [
        {"round_id": "new", "configuration_doc": {"mode": "live", "rewards_enabled": True}},
        {"round_id": "shadow", "configuration_doc": {"mode": "shadow", "rewards_enabled": False}},
        {"round_id": "old", "configuration_doc": {"mode": "live", "rewards_enabled": True}},
    ]
    calls = []
    service = object.__new__(ArenaService)
    service._config = SimpleNamespace(mode="live")
    service._store = SimpleNamespace(list_rounds=lambda **_kwargs: rows)
    service.activate_reward = lambda round_id: calls.append(round_id) or {"status": "activated"}
    assert service.activate_pending_rewards() == {"status": "ok", "activated": 2}
    assert calls == ["old", "new"]

    service._config = SimpleNamespace(mode="shadow")
    calls.clear()
    assert service.activate_pending_rewards() == {"status": "disabled", "activated": 0}
    assert calls == []


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
    assert service.active_rounds() == [{"round_id": "live-round", "status": "open"}]
    with pytest.raises(ServiceError, match="round_mode_mismatch"):
        service._round("shadow-round")


def test_open_stage_runs_everyone_on_ten_then_only_finalists_and_king_on_ten():
    calls = []

    class Store:
        def open_stage(self, round_id, stage, participants, positions):
            calls.append((round_id, stage, participants, positions))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._round = lambda _round_id: {
        "participants": [
            {"submission_id": "king", "miner_hotkey": "hk", "is_king": True},
            {"submission_id": "c1", "miner_hotkey": "h1", "is_king": False},
            {"submission_id": "c2", "miner_hotkey": "h2", "is_king": False},
        ],
        "finalists": ["c2"],
    }
    service.benchmark_icps = lambda _round_id: [{} for _ in range(contracts.BENCHMARK_ICP_COUNT)]

    service.open_stage("arena-2026-09-02", 1)
    service.open_stage("arena-2026-09-02", 2)
    assert [row["submission_id"] for row in calls[0][2]] == ["king", "c1", "c2"]
    assert calls[0][3] == list(range(10))
    assert [row["submission_id"] for row in calls[1][2]] == ["king", "c2"]
    assert calls[1][3] == list(range(10, 20))


def test_first_round_baseline_is_read_from_the_frozen_round_configuration():
    configured = "5" + "A" * 47
    changed_environment = "5" + "B" * 47
    baseline = {
        "submission_id": "baseline",
        "miner_hotkey": configured,
        "image_digest": "sha256:" + "a" * 64,
        "image_reference": "arena.example/lab/baseline@sha256:" + "a" * 64,
        "submitted_reference": "source.example/lab/baseline:latest",
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
    service._config = SimpleNamespace(defaults=SimpleNamespace(baseline_hotkey=changed_environment))
    service._round = lambda _round_id: {
        "configuration_doc": {"baseline_hotkey": configured, "max_challengers": 10},
    }
    service._reigning_king_hotkey = lambda: None

    participants = service.freeze_participants("arena-2026-09-02")
    assert len(participants) == 1 and participants[0]["submission_id"] == "baseline"
    assert participants[0]["is_king"] is True
    assert updates == [("arena-2026-09-02", "baseline", "accepted", "frozen", {"is_king": True})]


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


def test_image_admission_uses_one_absolute_tick_deadline(monkeypatch):
    source_registry = object()
    destination_registry = object()
    digest = "sha256:" + "c" * 64
    source_reference = images.parse_reference("source.example/miner/bundle@" + digest)
    destination_reference = images.parse_reference("arena.example/lab/bundle@" + digest)
    calls = []

    class Descriptor:
        reference = source_reference

        @staticmethod
        def to_document():
            return {"image_digest": digest, "image_size_bytes": 1024, "layer_count": 1}

    class Store:
        @staticmethod
        def list_submissions(_round_id, status):
            assert status == "uploaded"
            return [{"submission_id": "s1", "submitted_reference": str(source_reference)}]

        @staticmethod
        def update_submission(_round_id, _submission_id, _old, _new, _patch):
            return {"status": "ok"}

    def resolve(client, reference, rules, *, deadline=None):
        calls.append(("resolve", client, reference, rules, deadline))
        return Descriptor()

    def mirror(client, descriptor, repository, *, destination_client=None, deadline=None):
        calls.append(("mirror", client, descriptor, repository, destination_client, deadline))
        return destination_reference

    ticks = iter((100.0, 101.0))
    monkeypatch.setattr(service_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(images, "resolve_image", resolve)
    monkeypatch.setattr(images, "mirror_image", mirror)
    service = object.__new__(ArenaService)
    service._store = Store()
    service._config = SimpleNamespace(
        registry=destination_registry,
        source_registry=source_registry,
        admission_tick_seconds=10,
    )
    service._round = lambda _round_id: {
        "status": "open",
        "configuration_doc": {
            "image_rules": images.ImageRules().to_document(),
            "registry_repository": "arena.example/lab/bundle",
        },
    }

    assert service.admit_uploaded_submissions("arena-2026-09-02") == {
        "status": "ok",
        "accepted": 1,
        "rejected": 0,
        "deferred": 0,
        "remaining": 0,
    }
    assert calls[0][0:3] == ("resolve", source_registry, source_reference)
    assert calls[0][-1] == 110.0
    assert calls[1][0] == "mirror" and calls[1][1] is source_registry
    assert calls[1][4] is destination_registry and calls[1][-1] == 110.0


def _daily_source_icps():
    return [
        {
            "icp_id": "icp_20260903_%03d" % index,
            "prompt": "ICP %d" % index,
        }
        for index in range(1, contracts.BENCHMARK_ICP_COUNT + 1)
    ]


def _daily_source_service(source):
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

        def transition_round(self, round_id, expected, next_status, patch):
            self.transitions.append((round_id, expected, next_status, patch))
            return {"status": "ok"}

    service = object.__new__(ArenaService)
    service._store = Store()
    service._objects = Objects()
    service._config = SimpleNamespace(daily_icp_source=source)
    service._clock = lambda: datetime(2026, 9, 3, 12, tzinfo=timezone.utc)
    service._round = lambda _round_id: {"status": "open"}
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
            "set_id": 20260903,
            "active_at": datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        }
    ]
    stored = json.loads(
        service._objects.values["arena/arena-2026-09-03/benchmark.json"]
    )
    assert stored["icps"] == expected


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
        "set_id": 20260903,
    }
    assert service._store.cancelled == []


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

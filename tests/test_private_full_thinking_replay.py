"""Private replay keeps full capture without claiming the production baseline."""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
import json

import pytest

from lab_arena import api, contracts, reasoning_capture, service, wiring
from scripts import verify_arena_parallel_round as verification
from tests.test_verify_arena_parallel_round import (
    REPLAY_SOURCE_ROUND_ID, _replay_source_fixture,
)

CUTOFF = datetime(2026, 9, 27, 1, tzinfo=timezone.utc)
TARGET = "arena-2026-09-27-replay"
PINNED_BASELINE = (
    "https://github.com/leadpoet/leadpoet-sales-agent/archive/" + "b" * 40 + ".tar.gz"
)


def private_service(monkeypatch, tmp_path, *, replay=True, miner_route=True, fresh=False):
    source, _runs, _documents = _replay_source_fixture(monkeypatch)
    make_replay = verification._saved_output_replay
    replay_input = make_replay(
        source, REPLAY_SOURCE_ROUND_ID, TARGET,
        assignments=["source-a:0", "source-b:1"], historical=True,
        miner_score_route=miner_route,
    )
    rows = {}
    store = SimpleNamespace(
        get_round=lambda rid: rows.get(rid),
        list_submissions=lambda rid: [],
    )
    objects = service.LocalObjectStore(tmp_path / "objects")
    defaults = service.RoundDefaults(
        benchmark_icp_count=2, runner_hotkeys=("5" * 48,), baseline_hotkey="5" * 48,
        scorer_image_digest="sha256:" + "a" * 64,
        scorer_image_reference="registry.example/scorer@sha256:" + "a" * 64,
        integrity_from="2000-01-01T00:00:00Z", intent_details_from="2000-01-01T00:00:00Z",
        execution_sequence_from="2000-01-01T00:00:00Z",
        benchmark_disclosure_from="2000-01-01T00:00:00Z",
        flat_per_icp_cost_policy=True, reasoning_capture_from="2000-01-01T00:00:00Z",
        reasoning_baseline_source_url=PINNED_BASELINE,
        reasoning_baseline_source_sha256="sha256:" + "b" * 64,
        training_minimum_icp_coverage_percent=80,
        training_maximum_missing_reasoning_percent=2,
    )
    config = service.ServiceConfig(
        mode="live", store=store, object_store=objects, signer=None, chain=None,
        verify_signature=lambda *_: True, daily_icp_source=lambda **_: {},
        banned_hotkeys_source=lambda: [], broker_factory=lambda *_: None,
        defaults=defaults, reasoning_registry_source=reasoning_capture.load_registry,
    )
    built = service.ArenaService(config)
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(wiring, "build_service_from_environment", lambda mode: (built, None))
    monkeypatch.setattr(verification, "_saved_output_replay", lambda *_a, **_k: replay_input)
    if fresh:
        monkeypatch.setattr(verification, "_fresh_source_icps", lambda *_a, **_k: {
            "icps": replay_input["icps"],
            "evidence": {"mode": "fresh_sourcing", "saved_outputs_reused": False},
        })
    monkeypatch.setattr(verification, "_utc_now", lambda: CUTOFF)
    monkeypatch.setattr(api, "create_app", lambda obj: None)
    monkeypatch.setattr(service.ArenaService, "runner_settings", lambda obj: (["5" * 48], []))
    monkeypatch.setattr(service.ArenaService, "_require_integrity_schema", lambda obj: None)
    pinned, _ = verification._build_pinned_service(
        TARGET, **({"replay_published_round": REPLAY_SOURCE_ROUND_ID,
                    "replay_assignments": ["source-a:0", "source-b:1"],
                    "replay_historical": True, "replay_miner_score_route": miner_route}
                   if replay else {"published_icp_round": REPLAY_SOURCE_ROUND_ID,
                                   "icp_positions": [0, 1]} if fresh else {}),
    )
    writes = []
    def create_round(rid, configuration):
        # Inspect the actual native pre-persistence boundary. Wrong production
        # URL/hash must fail this regression before any row is stored.
        expected_url = replay_input["source_url"] if replay else PINNED_BASELINE
        expected_hash = replay_input["evidence"]["archive_hash"] if replay else defaults.reasoning_baseline_source_sha256
        assert configuration["baseline_source_url"] == expected_url
        assert configuration["reasoning_baseline_source_sha256"] == expected_hash
        writes.append(deepcopy(configuration))
        rows[rid] = {"round_id": rid, "status": "open", "configuration_doc": deepcopy(configuration),
                     "champion_submission_id": None, "participants": []}
        return {"status": "created"}
    store.create_round = create_round
    return pinned, replay_input, rows, writes


@pytest.mark.parametrize("miner_route", [False, True])
def test_native_private_freeze_and_resume_keep_replay_url_hash_and_capture(monkeypatch, tmp_path, miner_route):
    obj, replay, rows, writes = private_service(monkeypatch, tmp_path, miner_route=miner_route)
    assert verification._create_or_resume(obj, TARGET, now=CUTOFF) == "created"
    configuration = rows[TARGET]["configuration_doc"]
    assert configuration["reasoning_capture_policy"] == reasoning_capture.POLICY
    assert configuration["training_minimum_icp_coverage_percent"] == 80
    assert configuration["training_maximum_missing_reasoning_percent"] == 2
    assert configuration["judge_reasoning_route_registry"]["approved_routes"]
    assert contracts.validate_round_configuration(json.loads(json.dumps(configuration))) == configuration
    assert obj.config.baseline_source_fetcher(replay["source_url"], len(replay["archive"])) == replay["archive"]
    assert verification._create_or_resume(obj, TARGET, now=CUTOFF) == "resumed"
    assert len(writes) == 1


@pytest.mark.parametrize("fresh", [False, True])
def test_nonreplay_private_freeze_accepts_effective_pinned_full_thinking_baseline(monkeypatch, tmp_path, fresh):
    obj, _, rows, writes = private_service(monkeypatch, tmp_path, replay=False, fresh=fresh)
    assert verification._create_or_resume(obj, TARGET, now=CUTOFF) == "created"
    assert rows[TARGET]["configuration_doc"]["baseline_source_url"] == PINNED_BASELINE
    assert verification._create_or_resume(obj, TARGET, now=CUTOFF) == "resumed"


@pytest.mark.parametrize("change", [
    {"mode": "live"}, {"rewards_enabled": True},
    {"round_id": "arena-2026-09-27"},
    {"reasoning_baseline_source_sha256": "sha256:" + "0" * 64},
    {"baseline_source_url": "https://example.com/replay.tar.gz"},
    {"baseline_source_url": "https://arena.invalid/saved-output-replay/not-a-hash.tar.gz"},
    {"baseline_source_url": "https://arena.invalid/saved-output-replay/" + "a" * 64 + ".tar.gz?override=1"},
])
def test_shared_full_thinking_contract_rejects_unbound_private_sources(monkeypatch, tmp_path, change):
    obj, _, rows, _ = private_service(monkeypatch, tmp_path)
    verification._create_or_resume(obj, TARGET, now=CUTOFF)
    configuration = dict(rows[TARGET]["configuration_doc"], **change)
    assert not contracts.shadow_replay_baseline_source_matches(configuration)
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(configuration)


@pytest.mark.parametrize("pin", [None, "arena-2026-09-27-other"])
def test_private_source_requires_process_ownership_on_create_and_resume(monkeypatch, tmp_path, pin):
    obj, _, rows, _ = private_service(monkeypatch, tmp_path)
    verification._create_or_resume(obj, TARGET, now=CUTOFF)
    configuration = rows[TARGET]["configuration_doc"]
    obj._config.pinned_round_id = pin
    with pytest.raises(service.ServiceError, match="round_scope_mismatch"):
        obj._require_round_mode(rows[TARGET])
    # The existing readiness behavior stays intact, but an unowned replay may
    # never acquire the full-thinking source exception on a new round.
    obj._store.get_round = lambda _: None
    obj._store.create_round = lambda _rid, frozen: {"status": "created"}
    if pin is None:
        with pytest.raises(service.ServiceError, match="reasoning_capture_activation_not_ready:baseline_unready"):
            obj.create_round(CUTOFF, round_id=TARGET)
    else:
        with pytest.raises(service.ServiceError, match="round_scope_mismatch"):
            obj.create_round(CUTOFF, round_id=TARGET)
    assert configuration["reasoning_capture_policy"] == reasoning_capture.POLICY


@pytest.mark.parametrize("mode,rewards,digest,url", [
    ("live", False, None, None),
    ("shadow", True, None, None),
    ("shadow", False, "sha256:" + "0" * 64, None),
    ("shadow", False, None, "https://example.com/replay.tar.gz"),
])
def test_native_private_exception_fails_before_persistence(monkeypatch, tmp_path, mode, rewards, digest, url):
    obj, _, _, writes = private_service(monkeypatch, tmp_path)
    obj._config.mode = mode
    defaults = obj.config.defaults
    obj._config.defaults = replace(defaults, rewards_enabled=rewards,
        reasoning_baseline_source_sha256=digest or defaults.reasoning_baseline_source_sha256,
        reasoning_baseline_source_url=url or defaults.reasoning_baseline_source_url)
    with pytest.raises(service.ServiceError, match="reasoning_capture_activation_not_ready:baseline_unready"):
        obj.create_round(CUTOFF, round_id=TARGET)
    assert writes == []


@pytest.mark.parametrize("stored", [False, True])
def test_native_baseline_freeze_rejects_changed_archive_bytes(monkeypatch, tmp_path, stored):
    obj, replay, rows, _ = private_service(monkeypatch, tmp_path)
    verification._create_or_resume(obj, TARGET, now=CUTOFF)
    obj._store.get_submission = lambda _: None
    from tests.lab_arena.test_lab_arena_service_round import flavor_source_archive
    changed = flavor_source_archive("Changed")
    if stored:
        obj._objects.put("arena/%s/sources/baseline-%s.tar.gz" % (TARGET, TARGET[6:]), changed)
    else:
        obj._config.baseline_source_fetcher = lambda *_: changed
    with pytest.raises(service.ServiceError, match="reasoning_baseline_source_mismatch"):
        obj._initial_baseline(rows[TARGET])
    assert contracts.hash_bytes(replay["archive"]) == rows[TARGET]["configuration_doc"]["reasoning_baseline_source_sha256"]


def test_private_replay_fetch_and_frozen_resume_are_exactly_bound(monkeypatch, tmp_path):
    obj, replay, rows, _ = private_service(monkeypatch, tmp_path)
    verification._create_or_resume(obj, TARGET, now=CUTOFF)
    with pytest.raises(verification.VerificationError, match="scope"):
        obj.config.baseline_source_fetcher(replay["source_url"] + "?other", len(replay["archive"]))
    with pytest.raises(verification.VerificationError, match="scope"):
        obj.config.baseline_source_fetcher(replay["source_url"], len(replay["archive"]) - 1)
    rows[TARGET]["configuration_doc"]["reasoning_baseline_source_sha256"] = "sha256:" + "0" * 64
    with pytest.raises(verification.VerificationError, match="replay_reasoning_baseline_source"):
        verification._create_or_resume(obj, TARGET, now=CUTOFF)


def test_saved_output_without_calls_does_not_become_training_eligible():
    from tests.lab_arena.training_eligibility_test import captured_example, capture
    from tests.lab_arena.training_steps_test import seal
    example = captured_example()
    example["provider_calls"] = []
    result = capture(seal(example))
    assert result["eligible"] is False
    assert "execution_model_call_missing" in result["reasons"]


def test_native_replay_commits_and_accepts_execute_and_score_without_source_calls(monkeypatch, tmp_path):
    """Use disposable PostgreSQL, real source delivery/runner/completion/RPCs.

    Only the sandbox boundary and judge result are controlled. The actual
    generated replay harness runs and cannot manufacture a model trajectory.
    """
    import runpy
    from datetime import timedelta
    from lab_arena import runtime, scoring
    from tests.lab_arena.lab_arena_pg_harness import current_service_database
    from tests.lab_arena.test_integrity_round import IntegrityHarness
    from tests.lab_arena import test_lab_arena_service_round as fixtures

    database = current_service_database()
    try:
        psycopg2, dsn = next(database)
        harness = IntegrityHarness(lambda: psycopg2.connect(**dsn), tmp_path,
                                   challengers=[], runners=["alpha"])
        cutoff = datetime.now(timezone.utc) + timedelta(minutes=30)
        target = "arena-%s-replay" % cutoff.date().isoformat()
        source, _, documents = _replay_source_fixture(monkeypatch)
        benchmark_ref = source.store.get_round(REPLAY_SOURCE_ROUND_ID)["benchmark_ref"]
        documents[benchmark_ref] = json.dumps({
            "round_id": REPLAY_SOURCE_ROUND_ID,
            "icps": fixtures.daily_icps()[:2],
        }).encode()
        replay = verification._saved_output_replay(
            source, REPLAY_SOURCE_ROUND_ID, target,
            assignments=["source-a:0", "source-b:1"], historical=True,
            miner_score_route=True,
        )
        defaults = replace(harness.service.config.defaults,
            flat_per_icp_cost_policy=True,
            integrity_from="2000-01-01T00:00:00Z",
            reasoning_capture_from="2000-01-01T00:00:00Z",
            reasoning_baseline_source_url=PINNED_BASELINE,
            reasoning_baseline_source_sha256="sha256:" + "b" * 64,
            intent_details_from="2000-01-01T00:00:00Z",
            execution_sequence_from="2000-01-01T00:00:00Z",
            benchmark_disclosure_from="2000-01-01T00:00:00Z",
            training_minimum_icp_coverage_percent=80,
            training_maximum_missing_reasoning_percent=2,
        )
        harness.service._config.defaults = defaults
        monkeypatch.setenv("LAB_ARENA_MODE", "live")
        monkeypatch.setattr(wiring, "build_service_from_environment", lambda mode: (harness.service, None))
        monkeypatch.setattr(verification, "_saved_output_replay", lambda *_a, **_k: replay)
        monkeypatch.setattr(verification, "_utc_now", lambda: cutoff)
        native, _ = verification._build_pinned_service(
            target, replay_published_round=REPLAY_SOURCE_ROUND_ID,
            replay_assignments=["source-a:0", "source-b:1"],
            replay_historical=True, replay_miner_score_route=True,
        )
        harness.service = native
        harness.round_id = target
        assert verification._create_or_resume(native, target, now=cutoff) == "created"
        configuration = native.store.get_round(target)["configuration_doc"]
        harness.clock.advance_to(configuration["schedule"]["submission_cutoff"])
        assert native.commit_benchmark(target) == {"status": "ok", "participants": 1}
        frozen = native.store.get_round(target)
        verification._validate_frozen_round(native, frozen)
        assert native.benchmark_icps(target) == replay["icps"]
        assert configuration["reasoning_capture_policy"] == reasoning_capture.POLICY
        assert configuration["training_minimum_icp_coverage_percent"] == 80
        assert configuration["training_maximum_missing_reasoning_percent"] == 2

        class ReplaySandbox:
            executions = 0
            scores = 0

            def run_icp(self, spec, **_):
                document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
                if document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION:
                    self.scores += 1
                    output = scoring.build_scoring_output(document["scored_run_id"],
                        fixtures.deterministic_scorer(document["companies"], document["icp"], False))
                else:
                    self.executions += 1
                    assert spec.source_dir is not None
                    run_icp = runpy.run_path(str(spec.source_dir / "harness.py"))["run_icp"]
                    output = {"companies": run_icp(document["icp"])}
                return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())

        harness.sandbox = ReplaySandbox()
        from tests.lab_arena.parallel_twenty_runtime_e2e_test import _proxy_pool
        original_runner = harness.runner
        def runner(index):
            obj = original_runner(index)
            obj._config.proxy_worker_pool = _proxy_pool(10)
            return obj
        harness.runner = runner
        harness.advance_until("published")
        runs = native.store.list_runs(target)
        executions = [row for row in runs if row["kind"] == "execute"]
        scores = [row for row in runs if row["kind"] == "score"]
        assert executions and scores
        assert all(row["status"] == "accepted" for row in executions + scores)
        for row in executions:
            output = json.loads(native._objects.get(row["output_ref"]))
            # Compare against the actual generated source's recorded companies.
            assert contracts.document_hash(output["companies"]) == replay["evidence"]["origins"][row["icp_position"]]["companies_hash"]
        assert harness.sandbox.executions == len(executions)
        assert harness.sandbox.scores == len(scores)
        assert all(native.store.list_ledger(run_id=row["run_id"]) == [] for row in runs)
        assert frozen["configuration_doc"]["rewards_enabled"] is False
        published = native.store.get_round(target)
        result = verification._evidence(native, target)
        # The controlled judge above has no paid transport. Only the unchanged
        # paid-score proof gates may remain; missing mining reasoning is not a
        # publication or private replay completeness requirement.
        assert set(result["proof"]["errors"]) == {
            "paid_ledger_empty",
            "replay_score_provider_route", "score_paid_calls", "score_paid_cost",
        }
        assert published["status"] == "published"
        assert published["publication_doc"]["final_ranking"][0]["training_data_eligibility"] == {
            "status": "not_assessed", "eligible": None, "reasons": [],
        }
        assert published["configuration_doc"]["reasoning_capture_policy"] == reasoning_capture.POLICY
        assert verification._create_or_resume(native, target, now=cutoff) == "resumed"
    finally:
        database.close()

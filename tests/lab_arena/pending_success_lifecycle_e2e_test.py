"""Pending provider billing survives restart through the signed round workflow.

Only provider responses and the sandbox are controlled. HTTP, signatures,
outbox files, billing selection, PostgreSQL, scoring and publication are real.
"""
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import pytest

from lab_arena import broker as br, contracts, runner as rn, runtime, scoring, shim
from tests.lab_arena.champion_score_funding576_e2e_test import _bind_http
from tests.lab_arena.lab_arena_pg_harness import current_service_database
from tests.lab_arena.successful_call_cost_round_test import ChargedTransport, CostHarness
from tests.lab_arena import test_lab_arena_service_round as fixtures


ROOT = Path(__file__).parents[2]


@pytest.fixture
def database():
    database = current_service_database()
    try:
        pg, dsn = next(database)
        with pg.connect(**dsn) as connection, connection.cursor() as cursor:
            for name in (
                "593-lab-arena-capture-summary-reader.sql",
                "595-lab-arena-capture-called-icps.sql",
                "598-lab-arena-capture-summary-v2.sql",
                "619-lab-arena-release-undispatched-call.sql",
                "620-lab-arena-retained-success-capture.sql",
            ):
                cursor.execute((ROOT / "scripts" / name).read_text())
        yield pg, dsn
    finally:
        database.close()


class PendingTransport(ChargedTransport):
    def __init__(self):
        super().__init__()
        self.billed = False
        self.pending_posts = []

    def send(self, **kwargs):
        if kwargs["method"] == "GET" and kwargs["url"].startswith(br.OPENROUTER_GENERATION_URL):
            generation = kwargs["url"][len(br.OPENROUTER_GENERATION_URL):]
            document = ({"data": {"id": generation, "total_cost": "0.000019"}}
                        if self.billed else {"error": {"code": 404}})
            return br.ProviderResponse(200 if self.billed else 404,
                {"content-type": "application/json"}, json.dumps(document).encode())
        request = json.loads(kwargs.get("body") or b"{}")
        prompt = (request.get("messages") or [{}])[0].get("content", "")
        if isinstance(prompt, str) and prompt.startswith("pending-success:"):
            self.pending_posts.append(prompt)
            document = {"id": "gen-" + contracts.document_hash(prompt)[7:23],
                "model": request["model"],
                "choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "completed sourcing answer"}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 4}}
            return br.ProviderResponse(200, {"content-type": "application/json"},
                                       json.dumps(document).encode())
        return super().send(**kwargs)


class PendingHarness(CostHarness):
    def build_service(self):
        service = super().build_service()
        if not hasattr(self, "pending_transport"):
            self.pending_transport = PendingTransport()
        original = service.config.broker_factory

        def factory(owner, row):
            broker = original(owner, row)
            broker._transport = self.pending_transport
            return broker

        service.config.broker_factory = factory
        return service


def test_pending_success_restarts_replays_settles_and_publishes_baseline_and_miner(
    database, tmp_path, monkeypatch,
):
    pg, dsn = database
    connect = lambda: pg.connect(**dsn)
    harness = PendingHarness(connect, tmp_path, challengers=["Bravo"], runners=["alpha"])
    # Fake missing-price reads need no wall-clock polling to prove persistence.
    monkeypatch.setattr(br, "_OPENROUTER_BILLING_MAX_ATTEMPTS", 1)
    execute = harness.sandbox.run_icp
    replay_inputs = {}
    injected = set()

    def capture_usage(submission):
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET ROLE leadpoet_capture_reader")
            cursor.execute("SELECT model_calls,input_tokens,output_tokens FROM "
                           "public.lab_arena_capture_summary_v2(%s,%s)",
                           (harness.round_id, submission))
            return cursor.fetchone()

    def run_icp(spec, **kwargs):
        document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        if document.get("schema_version") != scoring.SCORING_INPUT_SCHEMA_VERSION:
            submission = spec.source_dir.parent.name.removeprefix("submission-")
        else:
            submission = None
        if submission is not None and submission not in injected:
            injected.add(submission)
            with harness.sandbox.lock:
                os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
                try:
                    status, _, _ = shim.dispatch("openrouter.chat", {
                        "model": "openai/gpt-4o-mini", "max_tokens": 100,
                        "messages": [{"role": "user", "content": "pending-success:" + submission}],
                    }, 5000)
                    assert status == 200
                finally:
                    os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        return execute(spec, **kwargs)

    harness.sandbox.run_icp = run_icp
    with ExitStack() as stack:
        stack.callback(harness.service.store.close)
        _bind_http(harness, stack)
        api_factory = harness.api_factory

        def recording_api():
            api = api_factory()
            provider, complete = api.provider, api.complete

            def send(run_id, token, frame, **kwargs):
                result = provider(run_id, token, frame, **kwargs)
                if frame["operation_id"] == "openrouter.chat":
                    replay_inputs[run_id] = (token, deepcopy(frame), deepcopy(result))
                return result

            api.provider = send

            def unavailable_completion(envelope):
                if envelope["body"]["run_id"] in replay_inputs:
                    # A controlled connection outage before completion HTTP
                    # delivery leaves the signed outbox and current lease live.
                    raise rn.RunnerError("controlled completion connection outage", http_status=503)
                return complete(envelope)

            api.complete = unavailable_completion
            return api

        harness.api_factory = recording_api
        participants = fixtures._start_round(harness, day=19, epoch=44919)
        frozen = harness.service.store.get_round(harness.round_id)
        configuration = frozen["configuration_doc"]
        benchmark = harness.objects.get(frozen["benchmark_ref"])
        source_hashes = {row["submission_id"]: contracts.hash_bytes(harness.objects.get(row["source_ref"]))
                         for row in harness.service.store.list_submissions(harness.round_id)}
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        assert harness.service.advance_round(harness.round_id)["assignments"] == participants * contracts.STAGE_1_ICP_COUNT
        scheduled = harness.clock.now
        harness.clock.now = datetime.now(timezone.utc)
        outputs = {}
        # The native baseline-first allocator can stop after the outstanding
        # baseline lease. Recover each owner in sequence; do not bypass it.
        while len(outputs) < participants:
            harness.pending_transport.billed = False
            worker = harness.runner(0, parallel=1)
            worker._config.completion_retry_seconds = ()
            try:
                assert worker.run_once(max_claims=participants * contracts.STAGE_1_ICP_COUNT) >= 1
                assert worker.abandoned == 1
            finally:
                worker.close()
            directory = tmp_path / "work-0" / "pending-completions"
            pending = list(directory.glob("[0-9a-f]" * 32 + ".json"))
            assert len(pending) == 1
            document = json.loads(pending[0].read_text())
            run_id = document["body"]["run_id"]
            assert run_id in replay_inputs and run_id not in outputs
            assert document["body"]["result"]["terminal_status"] == "accepted"
            outputs[run_id] = contracts.document_hash(document["body"]["output"])
            saved = harness.service.store.get_run(run_id)
            assert saved["status"] == "leased"
            assert capture_usage(saved["submission_id"]) == (1, 3, 4)
            posts = list(harness.pending_transport.pending_posts)

            # Fresh service and worker have no process-local response or outbox state.
            harness.service.store.close()
            harness.service = harness.build_service()
            stack.callback(harness.service.store.close)
            _bind_http(harness, stack)
            api_factory = harness.api_factory
            fresh = harness.runner(0, parallel=1)
            try:
                token, frame, original = replay_inputs[run_id]
                replay = fresh._config.api.provider(run_id, token, frame)
                assert {key: replay[key] for key in ("status", "headers", "body_b64")} == {
                    key: original[key] for key in ("status", "headers", "body_b64")}
                assert replay["call"]["outcome"] == "uncertain"
                assert replay["call"].get("actual_microusd") is None
                assert harness.pending_transport.pending_posts == posts
                harness.pending_transport.billed = True
                assert harness.service._reconcile_openrouter_cost(harness.round_id, run_id=run_id)["status"] == "settled"
                assert fresh.run_once(max_claims=0) == 0
                assert not list(directory.iterdir())
            finally:
                fresh.close()
            saved = harness.service.store.get_run(run_id)
            assert saved["status"] == "accepted"
            assert contracts.document_hash(json.loads(harness.objects.get(saved["output_ref"]))) == outputs[run_id]
            rows = harness.service.store.list_ledger(call_identity=original["call"]["call_identity"])
            assert [row["entry_kind"] for row in rows] == ["reservation", "dispatch", "uncertain", "settlement"]
            assert rows[-1]["amount_microusd"] == 19
            assert capture_usage(saved["submission_id"]) == (1, 3, 4)
            harness.api_factory = recording_api
        assert {harness.flavors[harness.service.store.get_run(run_id)["submission_id"]]
                for run_id in outputs} == {"PublicBaseline", "Bravo"}
        harness.api_factory = api_factory
        harness.clock.now = scheduled
        harness.advance_until("published", runners=1)
        published = harness.service.store.get_round(harness.round_id)
        assert published["configuration_doc"] == configuration
        assert harness.objects.get(published["benchmark_ref"]) == benchmark
        assert len(published["publication_doc"]["final_ranking"]) == participants
        assert all(row["eligible"] for row in published["publication_doc"]["final_ranking"])
        for submission, source_hash in source_hashes.items():
            assert contracts.hash_bytes(harness.objects.get(harness.service.store.get_submission(submission)["source_ref"])) == source_hash
            public = harness.service.public_results(harness.round_id, submission)
            assert len(public["scores"]["stage_1"] + public["scores"]["stage_2"]) == contracts.BENCHMARK_ICP_COUNT
        for run_id, expected_hash in outputs.items():
            saved = harness.service.store.get_run(run_id)
            assert saved["status"] == "accepted"
            assert contracts.document_hash(json.loads(harness.objects.get(saved["output_ref"]))) == expected_hash
        assert len(harness.pending_transport.pending_posts) == 2
        fixtures.assert_canary_absent(harness, connect)

"""Real worker, gateway and ledger recovery for paid and routed Deepline calls."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import httpx
import pytest

from lab_arena import broker as br, runner
from tests.lab_arena.deepline_completed_response_recovery_postgres_test import setup
from tests.lab_arena.deepline_completed_response_recovery_test import NATIVE, catalog
from tests.lab_arena.deepline_late_response_recovery_postgres_test import database
from tests.lab_arena.deepline_late_response_recovery_test import LateTransport


class GatewayApi:
    def __init__(self, service):
        self.service = service
        self.frames = []

    def provider(self, run_id, lease_token, frame):
        self.frames.append(dict(frame))
        return self.service.handle_provider(run_id, lease_token, frame)


def _socket_worker(h, lease, token, api, *, deepline_catalog=None):
    path = Path("/tmp") / ("arena-deepline-recovery-" + uuid.uuid4().hex + ".sock")
    worker_lease = dict(lease)
    if deepline_catalog is not None:
        worker_lease["deepline_catalog"] = deepline_catalog
    worker = runner.WorkerSocketServer(
        path, api, runner.RunState(lease=worker_lease, lease_token=token)
    )
    worker.start()
    return worker, path


def _ledger_state(connect, identity):
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT entry_kind, amount_microusd, terminal_response "
            "FROM public.lab_arena_ledger WHERE call_identity=%s ORDER BY entry_id",
            (identity,),
        )
        entries = cursor.fetchall()
        cursor.execute(
            "SELECT terminal_response FROM public.lab_arena_deepline_call_responses "
            "WHERE call_identity=%s", (identity,),
        )
        overlays = cursor.fetchall()
    return entries, overlays


def test_billed_timeout_recovers_through_socket_with_one_charge(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = LateTransport(bill=True)
    h, lease, token, connect, _broker = setup(database, tmp_path, "23", transport)
    api = GatewayApi(h.service)
    worker, path = _socket_worker(h, lease, token, api, deepline_catalog=catalog())
    try:
        with httpx.HTTPTransport(uds=str(path)) as http_transport:
            response = http_transport.handle_request(httpx.Request(
                "POST", "http://code.deepline.com/api/v2/integrations/parallel_search/execute",
                json={"payload": {"objective": "Find the official example.com website"}},
                extensions={"timeout": {"connect": 10, "read": 10, "write": 10, "pool": 10}},
            ))
            response.read()
        assert response.status_code == 200
        assert response.json()["result"]["data"]["results"][0]["url"] == "https://example.com"
        assert response.headers[runner.SETTLED_MICROUSD_HEADER] == "2000"
        assert [frame["action_sequence"] for frame in api.frames] == [0, 0]
        assert api.frames[0] == api.frames[1]
        assert worker._state.action_sequence == 1
        assert len(worker._state.calls) == 1
        call = worker._state.calls[0]
        assert call["outcome"] == "settled" and call["actual_microusd"] == 2000
        posts = [request for request in transport.requests if request["method"] == "POST"]
        assert len(posts) == 2 and transport.paid == 1
        assert posts[0]["headers"]["idempotency-key"] == posts[1]["headers"]["idempotency-key"]
        assert (posts[0]["url"], posts[0]["body"]) == (posts[1]["url"], posts[1]["body"])
        entries, overlays = _ledger_state(connect, call["call_identity"])
        assert [entry[0] for entry in entries].count("dispatch") == 1
        settlements = [entry for entry in entries if entry[0] == "settlement"]
        assert len(settlements) == 1 and settlements[0][1] == 2000
        assert settlements[0][2]["deepline_response_missing"] is True
        assert len(overlays) == 1 and overlays[0][0]["call_succeeded"] is True
        before = len(transport.requests)
        assert h.service.handle_provider(lease["run_id"], token, api.frames[0])["status"] == 200
        assert len(transport.requests) == before
    finally:
        worker.stop()


class RoutedFirecrawlTransport(LateTransport):
    def send(self, **request):
        response = super().send(**request)
        if request["method"] == "POST":
            return br.ProviderResponse(200, {}, json.dumps({
                "job_id": NATIVE, "status": "completed",
                "billing": {"credits_charged": 0.02},
                "result": {"data": {
                    "rawHtml": "<html>Recovered scoring page</html>",
                    "metadata": {"url": "https://example.com/",
                                 "sourceURL": "https://example.com/", "statusCode": 200},
                }},
            }).encode())
        payload = json.loads(response.body)
        if "/executions/by-key/" in request["url"]:
            payload["toolId"] = "firecrawl_scrape"
        else:
            payload["recent"]["entries"][0].update(
                provider="firecrawl", operation="firecrawl_scrape"
            )
        return br.ProviderResponse(response.status, response.headers,
                                   json.dumps(payload).encode())


def test_routed_score_resumes_same_key_and_adapts_html(database, tmp_path, monkeypatch):
    monkeypatch.setattr(br, "_DEEPLINE_BILLING_MAX_ATTEMPTS", 1)
    transport = RoutedFirecrawlTransport()
    h, lease, token, connect, _broker = setup(database, tmp_path, "24", transport)
    # A disposable scoring lease with a miner payer. The broker and SQL still
    # enforce the current lease, key, charge and response recovery contracts.
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute("SET LOCAL session_replication_role=replica")
        cursor.execute(
            "UPDATE public.lab_arena_rounds SET status='stage1_scoring', "
            "configuration_doc=jsonb_set(jsonb_set(configuration_doc - 'deepline_catalog', "
            "'{call_quotas,deepline}','200'::jsonb), "
            "'{scoring_call_quotas,deepline}','200'::jsonb) WHERE round_id=%s",
            (h.round_id,),
        )
        cursor.execute(
            "UPDATE public.lab_arena_runs SET kind='score' WHERE run_id=%s",
            (lease["run_id"],),
        )
        cursor.execute(
            "UPDATE public.lab_arena_submissions SET is_king=FALSE WHERE submission_id=%s",
            (lease["submission_id"],),
        )
    h.service._hot_rounds.clear()
    assert h.service.store.provider_funding(lease["run_id"], "deepline")["funding_source"] == "miner_key"
    api = GatewayApi(h.service)
    worker, path = _socket_worker(h, dict(lease, kind="score"), token, api)
    try:
        with httpx.HTTPTransport(uds=str(path)) as http_transport:
            response = http_transport.handle_request(httpx.Request(
                "GET", "http://api.scrapingdog.com/scrape?url=https%3A%2F%2Fexample.com%2F",
                extensions={"timeout": {"connect": 10, "read": 10, "write": 10, "pool": 10}},
            ))
            response.read()
        assert response.status_code == 200
        assert response.content == b"<html>Recovered scoring page</html>"
        assert [frame["action_sequence"] for frame in api.frames] == [0, 0]
        assert api.frames[0] == api.frames[1]
        assert len(worker._state.calls) == 1
        call = worker._state.calls[0]
        assert call["operation_id"] == "scrapingdog.scrape"
        assert call["effective_operation_id"] == "deepline.execute"
        assert call["outcome"] == "settled" and call["actual_microusd"] == 2000
        posts = [request for request in transport.requests if request["method"] == "POST"]
        assert len(posts) == 2 and transport.paid == 1
        assert posts[0]["headers"]["idempotency-key"] == posts[1]["headers"]["idempotency-key"]
        assert posts[0]["headers"]["authorization"] == posts[1]["headers"]["authorization"]
        assert (posts[0]["url"], posts[0]["body"]) == (posts[1]["url"], posts[1]["body"])
        entries, overlays = _ledger_state(connect, call["call_identity"])
        assert [entry[0] for entry in entries].count("dispatch") == 1
        settlements = [entry for entry in entries if entry[0] == "settlement"]
        assert len(settlements) == 1 and settlements[0][1] == 2000
        assert len(overlays) == 1 and overlays[0][0]["call_succeeded"] is True
        before = len(transport.requests)
        assert h.service.handle_provider(lease["run_id"], token, api.frames[0])["status"] == 200
        assert len(transport.requests) == before
    finally:
        worker.stop()

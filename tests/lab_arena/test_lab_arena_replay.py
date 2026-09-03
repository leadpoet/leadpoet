"""Replay recomputation: the Arena re-derives a scoring result from recorded judge responses."""

from __future__ import annotations

import base64
import json
import os
import socket
import sys
import textwrap
from pathlib import Path

import pytest

from lab_arena import broker as br
from lab_arena import contracts, replay, scoring, shim

WORK_ITEM = contracts.document_hash("item")
JUDGE_REQUEST = {"tool": "exa_search", "payload": {"query": "acme funding"}}
JUDGE_REPLY = {"job_id": "j", "status": "completed", "result": {"data": {"requestId": "r", "results": [{"url": "https://acme.example/news", "title": "Acme raises"}]}}, "billing": {"cost_usd": 0.01}}


def ledger_for(request_hash: str, *, refused_hash: str = None):
    entries = [
        {"entry_kind": "reservation", "call_identity": contracts.document_hash("c1"), "operation_id": "deepline.execute", "entry_doc": {"request_hash": request_hash, "action_sequence": 0}},
        {"entry_kind": "dispatch", "call_identity": contracts.document_hash("c1"), "entry_doc": {}},
        {"entry_kind": "settlement", "call_identity": contracts.document_hash("c1"), "entry_doc": {}, "terminal_response": {"status": 200, "headers": {"content-type": "application/json"}, "body_b64": base64.b64encode(json.dumps(JUDGE_REPLY).encode()).decode()}},
    ]
    if refused_hash:
        entries.append({"entry_kind": "refusal", "call_identity": contracts.document_hash("c2"), "operation_id": "openrouter.chat", "entry_doc": {"reason": "per_icp_quota", "call": {"request_hash": refused_hash}}})
    return entries


def test_recorded_responses_key_settled_replies_and_refusals_by_request_hash():
    request_hash = contracts.document_hash(br.normalized_request("deepline.execute", JUDGE_REQUEST))
    refused_hash = contracts.document_hash("refused")
    responses = replay.recorded_responses(ledger_for(request_hash, refused_hash=refused_hash))
    assert responses[request_hash]["kind"] == "response" and responses[request_hash]["terminal"]["status"] == 200
    assert responses[refused_hash] == {"kind": "error", "code": "budget_refused"}
    # The normalizer is the broker's: OpenRouter bodies hash with the explicit output cap the broker sends.
    chat = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "judge"}]}
    assert br.normalized_request("openrouter.chat", chat)["max_tokens"] == br.operations.OPENROUTER_MAX_OUTPUT_TOKENS


def test_replay_server_answers_recorded_frames_and_refuses_unknown_ones(tmp_path):
    request_hash = contracts.document_hash(br.normalized_request("deepline.execute", JUDGE_REQUEST))
    server = replay.ReplayServer(Path("/tmp") / ("lr-%s" % os.getpid()) / "worker.sock", replay.recorded_responses(ledger_for(request_hash)))
    server.start()
    try:
        os.environ[shim.WORKER_SOCKET_ENV] = str(server._path)
        status, headers, body = shim.dispatch("deepline.execute", JUDGE_REQUEST, 5000)
        assert status == 200 and json.loads(body) == JUDGE_REPLY and headers["content-type"] == "application/json"
        with pytest.raises(shim.ShimProviderError) as refused:
            shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": "never recorded"}}, 5000)
        assert refused.value.code == replay.REPLAY_ERROR_CODE
        assert server.served == [request_hash] and len(server.misses) == 1
    finally:
        os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        server.stop()


FAKE_ENTRYPOINT = textwrap.dedent('''
    import json, os, sys
    import httpx  # patched by the shim installed through sitecustomize
    from lab_arena import scoring
    document = json.loads(open(os.environ["LAB_ARENA_INPUT_PATH"]).read())
    assert os.environ.get("LAB_ARENA_SHIM_TRUSTED_SCORER") == "1"
    reply = httpx.post("https://code.deepline.com/api/v2/integrations/exa_search/execute", json={"payload": {"query": "acme funding"}}, headers={"Authorization": "Bearer placeholder"}, timeout=5)
    titles = [item["title"] for item in reply.json()["result"]["data"]["results"]]
    breakdowns = [{"final_score": 50.0 + len(titles) * 10.0, "failure_reason": "", "proof_quote": titles[0]} for _ in document["companies"]]
    out = scoring.build_scoring_output(document["work_item_id"], breakdowns)
    open(os.environ["LAB_ARENA_OUTPUT_PATH"], "w").write(json.dumps(out))
''')


def test_replay_runs_the_entrypoint_against_recorded_responses_in_a_subprocess(tmp_path):
    script = tmp_path / "fake_entrypoint.py"
    script.write_text(FAKE_ENTRYPOINT)
    request_hash = contracts.document_hash(br.normalized_request("deepline.execute", JUDGE_REQUEST))
    icp = {"icp_id": "arena:x", "prompt": "p", "max_companies": 2, "employee_count": ["11-50"], "company_stage": "Seed"}
    companies = [{"company_name": "Acme", "website": "https://acme.example", "employee_count": "11-50"}, {"company_name": "Beta", "website": "https://beta.example", "employee_count": "11-50"}]
    input_document = scoring.build_scoring_input(work_item_id=WORK_ITEM, icp=icp, companies=companies, policy=scoring.build_scorer_policy(), evaluation_date="2026-09-02")
    output, report = replay.replay_work_item(input_document=input_document, ledger_entries=ledger_for(request_hash), work_dir=tmp_path, entry_command=[sys.executable, str(script)], timeout_seconds=120)
    assert output["work_item_id"] == WORK_ITEM and [b["final_score"] for b in output["breakdowns"]] == [60.0, 60.0]
    assert output["breakdowns"][0]["proof_quote"] == "Acme raises"  # the recorded reply reached the judge through the shim
    assert report == {"served": 1, "misses": [], "recorded": 1}
    # A judge request that was never recorded is refused, and the fake reports it as a judge error.
    failing = tmp_path / "fail_entrypoint.py"
    failing.write_text(FAKE_ENTRYPOINT.replace('"acme funding"', '"unrecorded"'))
    with pytest.raises(replay.ReplayError):
        replay.replay_work_item(input_document=input_document, ledger_entries=ledger_for(request_hash), work_dir=tmp_path, entry_command=[sys.executable, str(failing)], timeout_seconds=120)
    # Nothing of the replay survives on disk.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("replay-")]


def test_replay_timeout_and_failure_are_replay_errors(tmp_path):
    hang = tmp_path / "hang.py"
    hang.write_text("import time\ntime.sleep(30)\n")
    input_document = scoring.build_scoring_input(work_item_id=WORK_ITEM, icp={"icp_id": "x", "prompt": "p", "max_companies": 1, "employee_count": ["11-50"], "company_stage": "Seed"}, companies=[{"company_name": "A", "website": "https://a.example", "employee_count": "11-50"}], policy=scoring.build_scorer_policy(), evaluation_date="2026-09-02")
    with pytest.raises(replay.ReplayError, match="timed out"):
        replay.replay_work_item(input_document=input_document, ledger_entries=[], work_dir=tmp_path, entry_command=[sys.executable, str(hang)], timeout_seconds=2)
    crash = tmp_path / "crash.py"
    crash.write_text("import sys\nsys.exit(3)\n")
    with pytest.raises(replay.ReplayError, match="exit code 3"):
        replay.replay_work_item(input_document=input_document, ledger_entries=[], work_dir=tmp_path, entry_command=[sys.executable, str(crash)], timeout_seconds=30)

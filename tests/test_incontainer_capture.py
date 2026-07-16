"""Tests for in-container trace capture (sourcing-model training data).

Covers the RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE pipeline:
- the diagnostics-bootstrap tee emits single-line trace markers on success and
  failure for urllib/httpx/requests (plus the aiohttp read hook),
- redaction (no Authorization / sk-or- / api_key material in emitted JSON),
- per-call and per-run body caps (drop body, keep sha256/byte_len/truncated),
- the flag-off kill switch,
- host-side collection: runners publish parsed entries, the evaluator hands
  them to the injectable trace_sink, and per-ICP rows carry POINTERS ONLY,
- the default sink (S3 upload when configured, count-and-drop otherwise),
- capture-disabled / no-capture rows stay byte-identical to the legacy shape.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import subprocess
import sys
import types

import pytest

from research_lab.canonical import sha256_json
from research_lab.eval import evaluator
from research_lab.eval import private_runtime
from research_lab.eval.private_runtime import (
    INCONTAINER_TRACE_MARKER,
    PrivateModelRuntimeError,
    SubprocessPrivateModelRunner,
)

TRACE_ENV_NAMES = (
    "RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE",
    "RESEARCH_LAB_INCONTAINER_TRACE_MAX_CALL_BYTES",
    "RESEARCH_LAB_INCONTAINER_TRACE_MAX_TOTAL_BYTES",
    "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX",
    "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID",
    "RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY",
)

# The exact per-ICP row shape produced before trace capture existed. Rows from
# capture-disabled runs (or runners that emit no markers) must match this
# key-for-key so bundle hashing and downstream consumers see no change.
PRE_CAPTURE_ROW_KEYS = {
    "icp_ref",
    "icp_hash",
    # Requested company count for the ICP; per-ICP scores normalize against
    # it, so the evaluator stamps it on every row (capture on or off).
    "icp_company_goal",
    "status",
    "hard_failure",
    "base_company_scores",
    "candidate_company_scores",
    "failure_reason",
    "provider_excluded",
    "sourced_zero_no_error",
    "reference_sourced_zero_no_error",
}

POINTER_KEYS = {
    "incontainer_trace_ref",
    "incontainer_trace_sha256",
    "incontainer_trace_call_count",
}

# P5: dropped captures carry an explicit dropped marker instead of
# masquerading as populated rows (call_count is zeroed, dropped count kept).
DROPPED_POINTER_KEYS = POINTER_KEYS | {
    "incontainer_trace_dropped",
    "incontainer_trace_dropped_call_count",
}


@pytest.fixture(autouse=True)
def _clear_trace_env(monkeypatch):
    for name in TRACE_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Bootstrap probes (subprocess, mirroring tests/test_evaluator_fixes.py)
# ---------------------------------------------------------------------------


def _run_bootstrap_probe(probe: str, env_overrides: dict[str, str] | None = None):
    env = {**os.environ, **(env_overrides or {})}
    completed = subprocess.run(
        [sys.executable, "-c", private_runtime._PROVIDER_DIAGNOSTICS_BOOTSTRAP + probe],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return completed


def _entries(stderr: str) -> list[dict]:
    return private_runtime.parse_incontainer_trace_lines(stderr)


def _entry_for(entries: list[dict], url_fragment: str, phase: str = "call") -> dict:
    matches = [
        entry
        for entry in entries
        if url_fragment in entry.get("url_redacted", "") and entry.get("phase") == phase
    ]
    assert matches, (
        f"no phase={phase!r} trace entry for {url_fragment!r}: "
        f"{[(e.get('phase'), e.get('url_redacted')) for e in entries]}"
    )
    return matches[0]


def _decoded_body(entry: dict, field: str) -> str:
    return base64.b64decode(entry[field] or b"").decode("utf-8")


SUCCESS_AND_FAILURE_PROBE = r"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

BODY = b'{"result": "urllib payload ok"}'

class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass

server = HTTPServer(("127.0.0.1", 0), _Handler)
port = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()

# urllib success: capture must not consume the body the model reads.
import urllib.request
with urllib.request.urlopen(f"http://127.0.0.1:{port}/urllib-ok") as resp:
    data = resp.read()
assert data == BODY, data

# urllib failure.
try:
    urllib.request.urlopen("http://127.0.0.1:9/urllib-fail")
except Exception:
    pass

# httpx success (an OpenRouter-shaped LLM call).
import httpx

def _ok(request):
    return httpx.Response(200, request=request, json={"choices": [{"text": "llm says hi"}]})

client = httpx.Client(transport=httpx.MockTransport(_ok))
r = client.post("https://openrouter.ai/api/v1/chat/completions", json={"model": "m", "messages": []})
assert r.status_code == 200
assert r.json()["choices"][0]["text"] == "llm says hi"

# httpx failure (an Exa-shaped search call).
def _boom(request):
    raise httpx.ConnectError("httpx boom for trace test")

client_boom = httpx.Client(transport=httpx.MockTransport(_boom))
try:
    client_boom.get("https://api.exa.ai/search")
except Exception:
    pass

# requests success (a Scrapingdog-shaped fetch call).
import requests
from requests.adapters import BaseAdapter

class _OkAdapter(BaseAdapter):
    def send(self, request, **kwargs):
        response = requests.models.Response()
        response.status_code = 200
        response._content = b'{"scrape": "page content here"}'
        response.url = request.url
        response.request = request
        return response

    def close(self):
        pass

session = requests.Session()
session.mount("https://", _OkAdapter())
resp = session.post("https://api.scrapingdog.com/scrape", data=b'{"url": "https://x.test"}')
# Capture must not consume or alter the body the model reads.
assert resp.content == b'{"scrape": "page content here"}'

# requests failure.
class _BoomAdapter(BaseAdapter):
    def send(self, request, **kwargs):
        raise requests.ConnectionError("requests boom for trace test")

    def close(self):
        pass

session_boom = requests.Session()
session_boom.mount("https://", _BoomAdapter())
try:
    session_boom.get("https://provider.test/requests-fail")
except Exception:
    pass

# aiohttp success: the _request seam plus the read() body hook.
import aiohttp
import asyncio

async def _aio():
    async with aiohttp.ClientSession() as aio_session:
        async with aio_session.get(f"http://127.0.0.1:{port}/aiohttp-ok") as aio_resp:
            body = await aio_resp.read()
            assert body == BODY
            # Second read returns the cache and must not re-emit.
            body_again = await aio_resp.read()
            assert body_again == BODY

asyncio.run(_aio())

server.shutdown()
print("TRACE_PROBE_DONE")
"""


def test_bootstrap_emits_single_line_markers_on_success_and_failure():
    completed = _run_bootstrap_probe(SUCCESS_AND_FAILURE_PROBE)
    assert "TRACE_PROBE_DONE" in completed.stdout

    # Every marker line is a single, fully parseable JSON record.
    marker_lines = [line for line in completed.stderr.splitlines() if INCONTAINER_TRACE_MARKER in line]
    assert marker_lines
    for line in marker_lines:
        payload = line.split(INCONTAINER_TRACE_MARKER, 1)[1].strip()
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)

    entries = _entries(completed.stderr)
    assert len(entries) == len(marker_lines)
    schema_keys = {
        "seq",
        "phase",
        "provider_class",
        "method",
        "url_redacted",
        "request_body_b64",
        "response_status",
        "response_body_b64",
        "request_sha256",
        "response_sha256",
        "request_byte_len",
        "response_byte_len",
        "truncated",
        "outcome",
        "error",
    }
    for entry in entries:
        assert schema_keys <= set(entry), entry
    seqs = [entry["seq"] for entry in entries]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)

    # urllib success: body captured non-destructively, status + hash recorded.
    urllib_ok = _entry_for(entries, "/urllib-ok")
    assert urllib_ok["outcome"] == "success"
    assert urllib_ok["method"] == "GET"
    assert urllib_ok["response_status"] == 200
    assert urllib_ok["provider_class"] == "other"
    body = _decoded_body(urllib_ok, "response_body_b64")
    assert body == '{"result": "urllib payload ok"}'
    assert urllib_ok["truncated"] is False
    assert urllib_ok["response_sha256"] == hashlib.sha256(body.encode()).hexdigest()

    # urllib failure.
    urllib_fail = _entry_for(entries, "/urllib-fail")
    assert urllib_fail["outcome"] == "error"
    assert urllib_fail["error"]

    # httpx success classified as an LLM call, request+response bodies kept.
    openrouter = _entry_for(entries, "openrouter.ai")
    assert openrouter["provider_class"] == "llm"
    assert openrouter["method"] == "POST"
    assert openrouter["response_status"] == 200
    assert "llm says hi" in _decoded_body(openrouter, "response_body_b64")
    assert '"model"' in _decoded_body(openrouter, "request_body_b64")

    # httpx failure classified as a search call.
    exa = _entry_for(entries, "api.exa.ai")
    assert exa["provider_class"] == "search"
    assert exa["outcome"] == "error"
    assert "httpx boom for trace test" in exa["error"]
    assert exa["response_status"] is None

    # requests success classified as a fetch call.
    scrapingdog = _entry_for(entries, "scrapingdog.com")
    assert scrapingdog["provider_class"] == "fetch"
    assert scrapingdog["response_status"] == 200
    assert "page content here" in _decoded_body(scrapingdog, "response_body_b64")
    assert '"url"' in _decoded_body(scrapingdog, "request_body_b64")

    # requests failure.
    requests_fail = _entry_for(entries, "/requests-fail")
    assert requests_fail["outcome"] == "error"
    assert "requests boom for trace test" in requests_fail["error"]

    # aiohttp: call entry (body deliberately not consumed) plus exactly one
    # response_body entry emitted by the read() hook.
    aio_call = _entry_for(entries, "/aiohttp-ok", phase="call")
    assert aio_call["response_status"] == 200
    assert aio_call["response_body_b64"] == ""
    assert aio_call["truncated"] is True
    aio_bodies = [
        entry
        for entry in entries
        if "/aiohttp-ok" in entry.get("url_redacted", "") and entry.get("phase") == "response_body"
    ]
    assert len(aio_bodies) == 1
    assert _decoded_body(aio_bodies[0], "response_body_b64") == '{"result": "urllib payload ok"}'

    # The provider-error diagnostics marker still fires for failures
    # (regression guard for the bug #35 seams the tee rides on).
    assert private_runtime.PROVIDER_ERROR_MARKER in completed.stderr


REDACTION_PROBE = r"""
import httpx

SECRET = "sk-or-redaction-secret-XYZ123"

def _ok(request):
    return httpx.Response(200, request=request, json={"echo": request.content.decode("utf-8")})

client = httpx.Client(transport=httpx.MockTransport(_ok))
client.post(
    "https://openrouter.ai/api/v1/chat?api_key=" + SECRET,
    headers={"Authorization": "Bearer " + SECRET},
    json={"api_key": SECRET, "prompt_text": "hello world"},
)
print("REDACTION_PROBE_DONE")
"""


def test_bootstrap_redacts_secrets_before_encoding():
    secret = "sk-or-redaction-secret-XYZ123"
    completed = _run_bootstrap_probe(
        REDACTION_PROBE,
        env_overrides={"OPENROUTER_API_KEY": secret},
    )
    assert "REDACTION_PROBE_DONE" in completed.stdout
    # The secret must not appear anywhere in stderr: not in URLs, not in
    # headers (never emitted), not inside base64 bodies.
    assert secret not in completed.stderr
    entries = _entries(completed.stderr)
    entry = _entry_for(entries, "openrouter.ai")
    assert "api_key=[REDACTED]" in entry["url_redacted"]
    request_body = _decoded_body(entry, "request_body_b64")
    assert secret not in request_body
    assert "[REDACTED]" in request_body
    assert "hello world" in request_body  # non-secret content is verbatim
    response_body = _decoded_body(entry, "response_body_b64")
    assert secret not in response_body


PER_CALL_CAP_PROBE = r"""
import httpx

def _big(request):
    return httpx.Response(200, request=request, content=b"A" * 200)

client = httpx.Client(transport=httpx.MockTransport(_big))
client.get("https://provider.test/big-body")
print("CAP_PROBE_DONE")
"""


def test_per_call_cap_truncates_body_but_keeps_full_hash_and_length():
    completed = _run_bootstrap_probe(
        PER_CALL_CAP_PROBE,
        env_overrides={"RESEARCH_LAB_INCONTAINER_TRACE_MAX_CALL_BYTES": "32"},
    )
    assert "CAP_PROBE_DONE" in completed.stdout
    entry = _entry_for(_entries(completed.stderr), "/big-body")
    assert entry["truncated"] is True
    assert _decoded_body(entry, "response_body_b64") == "A" * 32
    assert entry["response_byte_len"] == 200
    assert entry["response_sha256"] == hashlib.sha256(b"A" * 200).hexdigest()


TOTAL_CAP_PROBE = r"""
import httpx

def _fifty(request):
    return httpx.Response(200, request=request, content=b"B" * 50)

client = httpx.Client(transport=httpx.MockTransport(_fifty))
client.get("https://provider.test/first-call")
client.get("https://provider.test/second-call")
print("TOTAL_CAP_PROBE_DONE")
"""


def test_total_cap_drops_bodies_but_keeps_hash_length_truncated():
    completed = _run_bootstrap_probe(
        TOTAL_CAP_PROBE,
        env_overrides={"RESEARCH_LAB_INCONTAINER_TRACE_MAX_TOTAL_BYTES": "64"},
    )
    assert "TOTAL_CAP_PROBE_DONE" in completed.stdout
    entries = _entries(completed.stderr)
    first = _entry_for(entries, "/first-call")
    second = _entry_for(entries, "/second-call")
    # First call fits the 64-byte run budget.
    assert _decoded_body(first, "response_body_b64") == "B" * 50
    assert first["truncated"] is False
    # Second call would exceed it: body dropped, hash/length/truncated kept.
    assert second["response_body_b64"] == ""
    assert second["response_sha256"] == hashlib.sha256(b"B" * 50).hexdigest()
    assert second["response_byte_len"] == 50
    assert second["truncated"] is True


FLAG_OFF_PROBE = r"""
import httpx

def _ok(request):
    return httpx.Response(200, request=request, content=b"ok")

client = httpx.Client(transport=httpx.MockTransport(_ok))
client.get("https://provider.test/flag-off-ok")

def _boom(request):
    raise httpx.ConnectError("boom while capture disabled")

client_boom = httpx.Client(transport=httpx.MockTransport(_boom))
try:
    client_boom.get("https://provider.test/flag-off-fail")
except Exception:
    pass
print("FLAG_OFF_PROBE_DONE")
"""


def test_flag_off_emits_no_trace_markers_but_keeps_provider_errors():
    completed = _run_bootstrap_probe(
        FLAG_OFF_PROBE,
        env_overrides={"RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE": "false"},
    )
    assert "FLAG_OFF_PROBE_DONE" in completed.stdout
    assert INCONTAINER_TRACE_MARKER not in completed.stderr
    # The provider-error diagnostics channel is independent of the tee.
    assert private_runtime.PROVIDER_ERROR_MARKER in completed.stderr


# ---------------------------------------------------------------------------
# Host-side helpers: parse / strip / flag / env passthrough
# ---------------------------------------------------------------------------


def test_parse_and_strip_trace_lines():
    stderr = "\n".join(
        [
            "noise before",
            'research_lab_private_runtime_trace {"seq": 1, "provider_class": "llm"}',
            "research_lab_private_runtime_provider_error HTTPError: boom; status=500",
            'prefix research_lab_private_runtime_trace {"seq": 2, "provider_class": "search"}',
            "research_lab_private_runtime_trace not-json",
            "noise after",
        ]
    )
    entries = private_runtime.parse_incontainer_trace_lines(stderr)
    assert [entry["seq"] for entry in entries] == [1, 2]
    stripped = private_runtime.strip_incontainer_trace_lines(stderr)
    assert INCONTAINER_TRACE_MARKER not in stripped
    assert private_runtime.PROVIDER_ERROR_MARKER in stripped
    assert "noise before" in stripped
    assert "noise after" in stripped


def test_trace_flag_defaults_and_env_passthrough():
    assert private_runtime.incontainer_trace_capture_enabled() is True
    for name in private_runtime.INCONTAINER_TRACE_ENV_PASSTHROUGH:
        assert name in private_runtime.DEFAULT_ENV_PASSTHROUGH
        assert name in private_runtime.private_model_env_passthrough()
        assert name in private_runtime.private_model_env_passthrough(include_proxy=True)


def test_trace_flag_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "false")
    assert private_runtime.incontainer_trace_capture_enabled() is False


def test_publish_without_collector_is_noop():
    # Direct gateway runner calls have no collector installed; publishing must
    # neither raise nor accumulate anywhere.
    private_runtime.publish_incontainer_trace_entries([{"seq": 1}])


# ---------------------------------------------------------------------------
# Runner integration: subprocess runner publishes entries, strips diagnostics
# ---------------------------------------------------------------------------


PUBLISHING_ADAPTER = """
import json
import sys

FAKE_ENTRY = {
    "seq": 1,
    "phase": "call",
    "provider_class": "llm",
    "method": "POST",
    "url_redacted": "https://openrouter.ai/api/v1/chat/completions",
    "request_body_b64": "eyJtb2RlbCI6ICJtIn0=",
    "response_status": 200,
    "response_body_b64": "",
    "request_sha256": "abc",
    "response_sha256": "",
    "request_byte_len": 13,
    "response_byte_len": 0,
    "truncated": False,
    "outcome": "success",
    "error": "",
}


def run_icp(icp, context):
    sys.stderr.write("research_lab_private_runtime_trace " + json.dumps(FAKE_ENTRY) + "\\n")
    second = dict(FAKE_ENTRY, seq=2, provider_class="search")
    sys.stderr.write("research_lab_private_runtime_trace " + json.dumps(second) + "\\n")
    sys.stderr.write("adapter diagnostic noise line\\n")
    return [{"company_name": "TraceCo", "company_website": "https://traceco.test", "score": 42.0}]
"""

FAILING_ADAPTER = """
import sys


def run_icp(icp, context):
    sys.stderr.write('research_lab_private_runtime_trace {"seq": 1, "outcome": "success"}\\n')
    raise RuntimeError("adapter exploded for test")
"""

REAL_ICP = {
    "industry": "Software",
    "intent_signal": "hiring sales reps",
    "employee_count": "51-200",
}


def _subprocess_runner(tmp_path, source: str) -> SubprocessPrivateModelRunner:
    (tmp_path / "research_lab_adapter.py").write_text(source, encoding="utf-8")
    return SubprocessPrivateModelRunner({"source_path": str(tmp_path)})


def test_subprocess_runner_publishes_entries_and_strips_error_diagnostics(tmp_path):
    runner = _subprocess_runner(tmp_path, FAILING_ADAPTER)
    entries, token = private_runtime.begin_incontainer_trace_collection()
    try:
        with pytest.raises(PrivateModelRuntimeError) as excinfo:
            runner(dict(REAL_ICP), {})
    finally:
        private_runtime.end_incontainer_trace_collection(token)
    # Entries emitted before the failure are still published (training data
    # from failed attempts matters), and the marker line never reaches the
    # exception text bound for diagnostics/audit docs.
    assert [entry["seq"] for entry in entries] == [1]
    assert INCONTAINER_TRACE_MARKER not in str(excinfo.value)
    assert "adapter exploded for test" in str(excinfo.value)


def test_subprocess_runner_skips_publish_when_capture_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "false")
    runner = _subprocess_runner(tmp_path, PUBLISHING_ADAPTER)
    entries, token = private_runtime.begin_incontainer_trace_collection()
    try:
        outputs = runner(dict(REAL_ICP), {})
    finally:
        private_runtime.end_incontainer_trace_collection(token)
    assert entries == []
    assert outputs and outputs[0]["company_name"] == "TraceCo"


# ---------------------------------------------------------------------------
# Evaluator integration: sink dispatch + pointer-only rows
# ---------------------------------------------------------------------------


def _benchmark_items(count: int) -> list[dict]:
    return [
        {
            "icp_ref": f"icp-{index}",
            "icp_hash": f"hash-{index}",
            "icp": {"name": f"icp-{index}"},
        }
        for index in range(count)
    ]


async def _fake_scorer(companies, icp, is_reference_model):
    return [float(company.get("score", 0.0) or 0.0) for company in companies]


async def _score_items(runner, items, **kwargs):
    return await evaluator.score_private_model_pair_items(
        benchmark_items=items,
        base_runner=None,
        candidate_runner=runner,
        company_scorer=_fake_scorer,
        run_context={"run_id": "run-trace-1"},
        image_candidate=True,
        **kwargs,
    )


class PublishingRunner:
    """Async fake runner that publishes one trace entry per call."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, icp, context):
        self.calls += 1
        private_runtime.publish_incontainer_trace_entries(
            [
                {
                    "seq": 1,
                    "phase": "call",
                    "provider_class": "llm",
                    "method": "POST",
                    "url_redacted": "https://openrouter.ai/x",
                    "request_body_b64": "",
                    "response_status": 200,
                    "response_body_b64": "",
                    "request_sha256": "",
                    "response_sha256": "",
                    "request_byte_len": 0,
                    "response_byte_len": 0,
                    "truncated": False,
                    "outcome": "success",
                    "error": "",
                    "icp_marker": str(icp.get("name") or ""),
                }
            ]
        )
        return [{"score": 42.0}]


class PlainRunner:
    """Old-runner stand-in: never publishes trace entries."""

    async def __call__(self, icp, context):
        return [{"score": 42.0}]


async def test_evaluator_hands_entries_to_custom_sink_and_rows_carry_pointers_only(tmp_path):
    runner = _subprocess_runner(tmp_path, PUBLISHING_ADAPTER)
    received: dict[str, list[dict]] = {}

    async def sink(icp_ref, entries):
        received[icp_ref] = entries
        return "s3://custom/prefix/icp-real.json"

    items = [{"icp_ref": "icp-real", "icp_hash": "hash-real", "icp": dict(REAL_ICP)}]
    rows = await _score_items(runner, items, trace_sink=sink)

    entries = received["icp-real"]
    assert [entry["seq"] for entry in entries] == [1, 2]
    assert all(entry["runner_role"] == "candidate" for entry in entries)

    row = rows[0]
    assert row["candidate_company_scores"] == [42.0]
    assert row["incontainer_trace_ref"] == "s3://custom/prefix/icp-real.json"
    assert row["incontainer_trace_call_count"] == 2
    assert row["incontainer_trace_sha256"] == sha256_json(entries)
    assert set(row) == PRE_CAPTURE_ROW_KEYS | POINTER_KEYS

    # Scanner-poison guard: no decoded/encoded body material in the row.
    serialized = json.dumps(row)
    assert "request_body_b64" not in serialized
    assert "response_body_b64" not in serialized
    assert "eyJtb2RlbCI6" not in serialized
    assert "url_redacted" not in serialized


async def test_default_sink_counts_and_drops_without_s3_env(caplog):
    runner = PublishingRunner()
    with caplog.at_level(logging.INFO, logger="research_lab.eval.evaluator"):
        rows = await _score_items(runner, _benchmark_items(3))
    for row in rows:
        assert row["incontainer_trace_ref"] == ""
        assert row["incontainer_trace_call_count"] == 0
        assert row["incontainer_trace_dropped"] is True
        assert row["incontainer_trace_dropped_call_count"] == 1
        assert row["incontainer_trace_sha256"].startswith("sha256:")
        assert set(row) == PRE_CAPTURE_ROW_KEYS | DROPPED_POINTER_KEYS
    drop_records = [
        record
        for record in caplog.records
        if "research_lab_incontainer_trace_dropped" in record.getMessage()
    ]
    assert len(drop_records) == 1  # logged once per eval, not per ICP


async def test_default_sink_uploads_to_s3_with_sse_kms(monkeypatch):
    puts: list[dict] = []

    class _FakeS3Client:
        def put_object(self, **kwargs):
            puts.append(kwargs)
            return {}

    fake_boto3 = types.SimpleNamespace(client=lambda service: _FakeS3Client())
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    monkeypatch.setenv(
        "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://trace-bucket/lab/traces/"
    )
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID", "kms-key-42")

    rows = await _score_items(PublishingRunner(), _benchmark_items(1))
    row = rows[0]
    assert row["incontainer_trace_ref"] == "s3://trace-bucket/lab/traces/run-trace-1/icp-0.json"
    assert row["incontainer_trace_call_count"] == 1

    assert len(puts) == 1
    put = puts[0]
    assert put["Bucket"] == "trace-bucket"
    assert put["Key"] == "lab/traces/run-trace-1/icp-0.json"
    assert put["ServerSideEncryption"] == "aws:kms"
    assert put["SSEKMSKeyId"] == "kms-key-42"
    payload = json.loads(put["Body"].decode("utf-8"))
    assert payload["icp_ref"] == "icp-0"
    assert payload["run_ref"] == "run-trace-1"
    assert payload["call_count"] == 1
    assert payload["entries"][0]["runner_role"] == "candidate"
    # The content lives only in S3; the row carries the pointer.
    assert row["incontainer_trace_sha256"] == sha256_json(payload["entries"])


async def test_sink_failure_never_fails_the_run(caplog):
    async def exploding_sink(icp_ref, entries):
        raise RuntimeError("sink exploded for test")

    with caplog.at_level(logging.WARNING, logger="research_lab.eval.evaluator"):
        rows = await _score_items(PublishingRunner(), _benchmark_items(2), trace_sink=exploding_sink)
    for row in rows:
        assert row["status"] == "completed"
        assert row["candidate_company_scores"] == [42.0]
        # Capture attempt still recorded via hash + dropped marker; ref empty.
        assert row["incontainer_trace_ref"] == ""
        assert row["incontainer_trace_call_count"] == 0
        assert row["incontainer_trace_dropped"] is True
        assert row["incontainer_trace_dropped_call_count"] == 1
    assert any(
        "research_lab_incontainer_trace_sink_failed" in record.getMessage()
        for record in caplog.records
    )


async def test_capture_disabled_rows_byte_identical_to_legacy_shape(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_INCONTAINER_TRACE_CAPTURE", "false")
    sink_calls: list[str] = []

    async def sink(icp_ref, entries):
        sink_calls.append(icp_ref)

    rows = await _score_items(PublishingRunner(), _benchmark_items(2), trace_sink=sink)
    assert sink_calls == []  # the kill switch overrides injected sinks too
    assert rows[0] == {
        "icp_ref": "icp-0",
        "icp_hash": "hash-0",
        "icp_company_goal": 5,
        "status": "completed",
        "hard_failure": False,
        "base_company_scores": [],
        "candidate_company_scores": [42.0],
        "failure_reason": "",
        "provider_excluded": False,
        "sourced_zero_no_error": False,
        "reference_sourced_zero_no_error": False,
    }
    for row in rows:
        assert set(row) == PRE_CAPTURE_ROW_KEYS


async def test_capture_on_with_old_runner_rows_unchanged():
    # Runners that emit no markers (old images) must produce rows identical to
    # the pre-capture shape even with the flag on (its default).
    rows = await _score_items(PlainRunner(), _benchmark_items(2))
    for row in rows:
        assert set(row) == PRE_CAPTURE_ROW_KEYS


async def test_concurrent_capture_isolated_per_icp(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY", "3")
    received: dict[str, list[dict]] = {}

    async def sink(icp_ref, entries):
        received[icp_ref] = entries

    rows = await _score_items(PublishingRunner(), _benchmark_items(3), trace_sink=sink)
    assert [row["icp_ref"] for row in rows] == ["icp-0", "icp-1", "icp-2"]
    for index in range(3):
        entries = received[f"icp-{index}"]
        assert len(entries) == 1
        # Each ICP's collector saw only its own runner call.
        assert entries[0]["icp_marker"] == f"icp-{index}"


# ---------------------------------------------------------------------------
# Bundle surface: pointer fields never move the score-bundle hash
# ---------------------------------------------------------------------------


def _artifact_manifest(seed: str) -> dict:
    payload = {
        "model_artifact_hash": sha256_json({"artifact": seed}),
        "git_commit_sha": "a" * 12,
        "image_digest": (
            f"123456789012.dkr.ecr.us-east-1.amazonaws.com/private/{seed}@sha256:{'0' * 64}"
        ),
        "config_hash": sha256_json({"config": seed}),
        "component_registry_version": "1.0",
        "scoring_adapter_version": "1.0",
        "manifest_uri": f"s3://research-lab-test/{seed}.json",
        "signature_ref": f"kms-signature:test-key:{seed}",
        "build_id": "",
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def _build_bundle(rows: list[dict]) -> dict:
    parent = _artifact_manifest("parent")
    candidate = _artifact_manifest("candidate")
    return evaluator.build_score_bundle_from_scored_icps(
        artifact_manifest=parent,
        benchmark={
            "benchmark_id": "bench-1",
            "icp_set_hash": sha256_json({"icps": 1}),
            "split_ref": "split-1",
            "item_refs": ["icp-0", "icp-1"],
            "scoring_version": "v1",
        },
        patch_manifest={"parent_artifact_hash": parent["model_artifact_hash"]},
        candidate_artifact_manifest=candidate,
        per_icp_results=rows,
        run_context={
            "run_id": "run-1",
            "ticket_id": "ticket-1",
            "miner_hotkey": "hotkey-1",
            "island": "island-1",
            "evaluation_epoch": 1,
            "evaluator_version": "eval-1",
            "execution_trace_ref": "trace-1",
            "cost_ledger_ref": "cost-1",
            "signature_ref": "pending",
        },
    )


def test_pointer_fields_ride_bundle_rows_and_never_leak_content():
    """P12/P13: trace POINTERS survive the aggregation rebuild into the bundle
    (previously stripped — the 'orphaned in S3' gap), the hash covers them
    deterministically, and no trace CONTENT ever reaches a bundle row."""
    base_rows = [
        {
            "icp_ref": f"icp-{index}",
            "icp_hash": f"hash-{index}",
            "status": "completed",
            "hard_failure": False,
            "base_company_scores": [],
            "candidate_company_scores": [42.0],
            "failure_reason": "",
            "provider_excluded": False,
            "sourced_zero_no_error": False,
            "reference_sourced_zero_no_error": False,
        }
        for index in range(2)
    ]
    pointered_rows = [
        {
            **row,
            "incontainer_trace_ref": "s3://trace-bucket/lab/run-1/icp.json",
            "incontainer_trace_sha256": sha256_json([{"seq": 1}]),
            "incontainer_trace_call_count": 3,
            "scorer_trace_ref": "s3://trace-bucket/lab/scorer-traces/run-1/icp.json",
            "scorer_trace_sha256": sha256_json({"doc": 1}),
        }
        for row in base_rows
    ]
    bundle_with = _build_bundle(pointered_rows)
    for row in bundle_with["aggregates"]["per_icp_results"]:
        assert row["incontainer_trace_ref"] == "s3://trace-bucket/lab/run-1/icp.json"
        assert row["incontainer_trace_call_count"] == 3
        assert row["scorer_trace_ref"].startswith("s3://trace-bucket/lab/scorer-traces/")
        # pointers only — never decoded trace content
        serialized = json.dumps(row)
        assert "request_body_b64" not in serialized
        assert "response_body_b64" not in serialized
    # Deterministic: same pointer inputs → same hash; different pointer
    # inputs → different hash (the pointers are now covered by the signature).
    assert (
        _build_bundle(pointered_rows)["score_bundle_hash"]
        == bundle_with["score_bundle_hash"]
    )
    assert (
        _build_bundle(base_rows)["score_bundle_hash"]
        != bundle_with["score_bundle_hash"]
    )


# ---------------------------------------------------------------------------
# P13 (trajectoryimprovements.md): corpus-mode budgets, provider classes,
# truncation surfacing, bootstrap patch assertion
# ---------------------------------------------------------------------------


PROVIDER_CLASS_PROBE = r"""
cases = {
    "https://openrouter.ai/api/v1/chat/completions": "llm",
    "https://api.peopledatalabs.com/v5/person/enrich": "enrichment_pdl",
    "https://api.hunter.io/v2/email-finder": "enrichment_hunter",
    "https://api.apollo.io/v1/people/match": "enrichment_apollo",
    "https://api.exa.ai/search": "search",
    "https://google.serper.dev/search": "search",
    "https://api.scrapingdog.com/scrape": "fetch",
    "https://www.linkedin.com/company/acme": "linkedin",
    "https://api.zerobounce.net/v2/validate": "verification_zerobounce",
    "https://random.example.com/x": "other",
}
for url, expected in cases.items():
    actual = _research_lab_trace_provider_class(url)
    assert actual == expected, f"{url}: {actual} != {expected}"
print("PROVIDER_CLASS_PROBE_DONE")
"""


def test_provider_class_names_enrichment_providers():
    """P13: the in-container tee names PDL/hunter/apollo/etc instead of
    collapsing everything non-core to "other"."""
    completed = _run_bootstrap_probe(PROVIDER_CLASS_PROBE)
    assert "PROVIDER_CLASS_PROBE_DONE" in completed.stdout


def test_bootstrap_emits_patch_assertion_marker():
    """P13/P19: the bootstrap asserts which client libs the tee patched."""
    completed = _run_bootstrap_probe('print("BOOT_ONLY_DONE")')
    assert "BOOT_ONLY_DONE" in completed.stdout
    lines = [
        line
        for line in completed.stderr.splitlines()
        if line.startswith("research_lab_private_runtime_capture_bootstrap ")
    ]
    assert len(lines) == 1
    doc = json.loads(lines[0].split(" ", 1)[1])
    assert doc["patched_clients"]["urllib"] is True
    # Installed clients must be verifiably patched (None = not installed).
    for lib in ("httpx", "requests", "aiohttp"):
        assert doc["patched_clients"][lib] in (True, None)
    assert doc["max_call_bytes"] > 0
    assert doc["streaming_posture"] == "request_only_for_streams"


def test_corpus_env_defaults_larger_caps_only_with_prefix(monkeypatch):
    from research_lab.eval import private_runtime as pr

    for name in (
        "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX",
        pr.INCONTAINER_TRACE_MAX_CALL_BYTES_ENV,
        pr.INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV,
    ):
        monkeypatch.delenv(name, raising=False)
    # Diagnostics mode: no prefix → no overrides (small caps stay).
    assert pr.incontainer_trace_corpus_env() == {}

    monkeypatch.setenv(
        "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX", "s3://bucket/traces"
    )
    env = pr.incontainer_trace_corpus_env()
    assert env == {
        pr.INCONTAINER_TRACE_MAX_CALL_BYTES_ENV: str(
            pr.INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES
        ),
        pr.INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV: str(
            pr.INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES
        ),
    }
    assert pr.INCONTAINER_TRACE_CORPUS_MAX_CALL_BYTES > pr.INCONTAINER_TRACE_DEFAULT_MAX_CALL_BYTES
    assert pr.INCONTAINER_TRACE_CORPUS_MAX_TOTAL_BYTES > pr.INCONTAINER_TRACE_DEFAULT_MAX_TOTAL_BYTES

    # Explicit operator caps always win over the corpus defaults.
    monkeypatch.setenv(pr.INCONTAINER_TRACE_MAX_CALL_BYTES_ENV, "9999")
    env = pr.incontainer_trace_corpus_env()
    assert pr.INCONTAINER_TRACE_MAX_CALL_BYTES_ENV not in env
    assert pr.INCONTAINER_TRACE_MAX_TOTAL_BYTES_ENV in env


async def test_truncated_entries_surface_in_pointer_fields():
    from research_lab.eval.evaluator import _finalize_incontainer_trace

    async def sink(icp_ref, entries):
        return "s3://bucket/traces/run/icp.json"

    entries = [
        {"seq": 1, "truncated": False},
        {"seq": 2, "truncated": True},
        {"seq": 3, "truncated": True},
    ]
    fields = await _finalize_incontainer_trace(
        trace_sink=sink, icp_ref="icp-1", entries=entries
    )
    assert fields["incontainer_trace_call_count"] == 3
    assert fields["incontainer_trace_truncated_count"] == 2
    fields_clean = await _finalize_incontainer_trace(
        trace_sink=sink, icp_ref="icp-1", entries=[{"seq": 1, "truncated": False}]
    )
    assert "incontainer_trace_truncated_count" not in fields_clean

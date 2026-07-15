import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from gateway.research_lab.git_tree_evaluator import (
    classify_tree_evaluation,
    evaluation_plan_from_mapping,
)


def _diff(line: str, path: str = "model/ranker.py") -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        f"+{line}\n"
    )


def test_non_network_change_uses_frozen_replay():
    plan = classify_tree_evaluation(
        unified_diff=_diff("score = quality * 2"),
        target_files=("model/ranker.py",),
    )
    assert plan.mode == "replay"
    assert plan.reason_codes == ("no_outbound_contract_change_detected",)


def test_query_or_provider_change_requires_hybrid():
    plan = classify_tree_evaluation(
        unified_diff=_diff("params['query'] = company_name"),
        target_files=("model/ranker.py",),
    )
    assert plan.mode == "hybrid"
    assert "outbound_request_contract_changed" in plan.reason_codes


@pytest.mark.parametrize(
    "changed_line",
    (
        'method = "POST"',
        'body = {"company": company_name}',
        'timeout = 15',
        'search_term = company_name',
        'headers["authorization"] = token',
    ),
)
def test_method_payload_transport_and_search_changes_require_hybrid(changed_line):
    plan = classify_tree_evaluation(
        unified_diff=_diff(changed_line),
        target_files=("model/ranker.py",),
    )

    assert plan.mode == "hybrid"
    assert "outbound_request_contract_changed" in plan.reason_codes


def test_network_path_and_ambiguous_patch_fail_closed_to_hybrid():
    network = classify_tree_evaluation(
        unified_diff=_diff("timeout = 5", "model/provider_client.py"),
        target_files=("model/provider_client.py",),
    )
    ambiguous = classify_tree_evaluation(
        unified_diff="+score = 1\n",
        target_files=("model/ranker.py",),
    )
    assert network.mode == "hybrid"
    assert ambiguous.mode == "hybrid"
    assert "ambiguous_patch_contract" in ambiguous.reason_codes


def test_plan_round_trip_detects_tampering():
    plan = classify_tree_evaluation(
        unified_diff=_diff("score = quality * 2"),
        target_files=("model/ranker.py",),
    )
    assert evaluation_plan_from_mapping(plan.to_dict()) == plan


def test_measured_http_shim_routes_all_supported_clients_with_identical_body():
    pytest.importorskip("requests")
    pytest.importorskip("httpx")
    pytest.importorskip("aiohttp")
    root = Path(__file__).resolve().parents[1]
    script = r'''
import asyncio
import base64
import json
import urllib.request

import aiohttp
import httpx
import requests

import gateway.tee.sandbox_http_shim_v2 as shim

observed = []
def fake_execute(**kwargs):
    observed.append({
        "method": kwargs["method"],
        "url": kwargs["url"],
        "body": bytes(kwargs["body"]).decode("utf-8"),
    })
    return {
        "terminal_status": "attested_local_response",
        "http_status": 200,
        "headers": {"content-type": "application/json"},
        "body_b64": base64.b64encode(b'{"ok":true}').decode("ascii"),
        "failure_code": None,
    }

shim.execute = fake_execute
shim.install()
url = "https://api.example.invalid/search"
body = b"same-payload"
with urllib.request.urlopen(url, data=body, timeout=1) as response:
    assert response.status == 200
assert requests.post(url, data=body, timeout=1).status_code == 200
with httpx.Client() as client:
    assert client.post(url, content=body).status_code == 200

async def run_async_clients():
    async with httpx.AsyncClient() as client:
        assert (await client.post(url, content=body)).status_code == 200
    async with aiohttp.ClientSession() as client:
        async with client.post(url, data=body) as response:
            assert response.status == 200

asyncio.run(run_async_clients())
print(json.dumps(observed, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert len(observed) == 5
    assert {item["method"] for item in observed} == {"POST"}
    assert {item["url"] for item in observed} == {
        "https://api.example.invalid/search"
    }
    assert {item["body"] for item in observed} == {"same-payload"}


def test_frozen_http_evidence_miss_never_opens_a_network_socket(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cache_path = tmp_path / "empty-cache.json"
    cache_path.write_text(
        json.dumps({"schema_version": "1.1", "entries": {}}), encoding="utf-8"
    )
    script = r'''
import urllib.request
import gateway.tee.sandbox_http_shim_v2 as shim

# install() first: its setup lazily imports httpx/httpcore (which subclass
# socket.socket at module load) and legitimately uses AF_UNIX sockets to reach
# the sandbox. The leak detector is armed AFTER setup so it guards only the
# frozen urlopen under test, not the interceptor's own installation.
shim.install()
shim.socket.socket = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError("network socket opened")
)
urllib.request.urlopen("https://api.example.invalid/missing")
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(root),
            "RESEARCH_LAB_PROVIDER_EVIDENCE_MODE": "frozen",
            "RESEARCH_LAB_PROVIDER_EVIDENCE_CACHE_PATH": str(cache_path),
        },
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode != 0
    assert "RESEARCH_LAB_PROVIDER_EVIDENCE_MISS:" in completed.stderr
    assert "network socket opened" not in completed.stderr

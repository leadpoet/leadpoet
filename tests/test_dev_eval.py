"""Tests for the dormant L1 dev-eval harness (§6.3-1 / follow-up item 4.1).

Covers:
- snapshot request-key scheme (auth-param insensitivity, body normalization),
- record -> replay round trip through a fresh store instance,
- strict-miss policy (typed SnapshotMiss) and the ``empty`` miss policy
  (explicit and env-driven),
- in-container replay bootstrap parity with the host key scheme via a
  urllib subprocess round trip (and bootstrap inertness without the env),
- in-process replay seams for requests/httpx and the aiohttp live-traffic
  guard,
- deterministic mechanical scorer stability + monotonicity (better-matching
  companies score higher) + duplicate zeroing + live top-5 scale arithmetic,
- evaluate_dev determinism, failure/miss bookkeeping, and replay-mode guard,
- dev-set leak exclusion proof (ref/hash/intent-signature matches) and
  seeded selection determinism,
- snapshot-set manifest hash integrity (tamper detection).
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from leadpoet_verifier.aggregation import per_icp_normalized_score
from research_lab.canonical import sha256_json
from research_lab.eval import dev_eval
from research_lab.eval.dev_eval import (
    DEV_LEADS_PER_ICP,
    DevEvalError,
    MechanicalDevScorer,
    build_current_day_dev_bank,
    build_dev_icp_set,
    compute_dev_set_hash,
    evaluate_dev,
    mechanical_company_score,
    select_current_day_dev_icps,
)
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    MODE_REPLAY,
    SNAPSHOT_DIR_ENV,
    SNAPSHOT_MISS_POLICY_ENV,
    SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
    SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV,
    SNAPSHOT_RUNTIME_SECRET_REDACTION,
    SNAPSHOT_URI_ENV,
    SYNTHESIZED_EMPTY_MARKER,
    DevSnapshotStoreError,
    ProviderSnapshotStore,
    SnapshotMiss,
    build_snapshot_request,
    container_replay_env,
    dev_record_bootstrap,
    dev_replay_bootstrap,
)

DEV_ENV_VARS = (
    SNAPSHOT_URI_ENV,
    SNAPSHOT_MISS_POLICY_ENV,
    SNAPSHOT_DIR_ENV,
    SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
)

SCRAPINGDOG_URL = (
    "https://api.scrapingdog.com/linkedin?type=company&linkId={link_id}&api_key={key}"
)


@pytest.fixture(autouse=True)
def _clear_dev_env(monkeypatch):
    for name in DEV_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _record_store(tmp_path) -> ProviderSnapshotStore:
    return ProviderSnapshotStore(str(tmp_path / "snapshot_set"), mode=MODE_RECORD)


def _replay_store(tmp_path, **kwargs) -> ProviderSnapshotStore:
    return ProviderSnapshotStore(str(tmp_path / "snapshot_set"), mode=MODE_REPLAY, **kwargs)


def _dev_icp(index: int) -> dict:
    return {
        "icp_id": f"dev-{index}",
        "industry": "Software Development",
        "sub_industry": "DevOps Tooling",
        "product_service": "CI/CD platform",
        "geography": "United States",
        "country": "United States",
        "employee_count": "51-200",
        "intent_signals": [f"Hiring a DevOps engineer wave {index}"],
        "intent_signal": f"Hiring a DevOps engineer wave {index}",
    }


def _dev_items(count: int) -> list[dict]:
    items = []
    for index in range(count):
        icp = _dev_icp(index)
        items.append(
            {
                "icp": icp,
                "icp_ref": f"dev_set:{index}",
                "icp_hash": sha256_json({"icp": icp}),
            }
        )
    return items


def _rich_company(index: int = 0) -> dict:
    return {
        "company_name": f"Acme {index}",
        "company_website": f"https://acme-{index}.test",
        "industry": "Software Development",
        "sub_industry": "DevOps Tooling",
        "employee_count": "51-200",
        "country": "United States",
        "description": "CI/CD platform for DevOps teams",
        "intent_signals": [
            {
                "source": "job_board",
                "description": "Hiring a DevOps engineer to build pipelines",
                "url": f"https://acme-{index}.test/jobs/1",
                "date": "2026-05-01",
            }
        ],
    }


def _medium_company() -> dict:
    return {
        "company_name": "Middling Inc",
        "company_website": "https://middling.test",
        "industry": "Software Development",
        "employee_count": "51-200",
    }


def _mismatched_bucket_company() -> dict:
    return {**_rich_company(9), "employee_count": "10,001+"}


def _record_companies_response(store, link_id: str, companies: list[dict]) -> str:
    body = json.dumps({"companies": companies})
    request = build_snapshot_request(
        "GET", SCRAPINGDOG_URL.format(link_id=link_id, key="RECORDKEY")
    )
    store.record_response(request, status=200, body_text=body)
    return body


def _urllib_runner(icp, context):
    """In-process candidate runner that sources through urllib (seam-patched)."""
    import urllib.request

    url = SCRAPINGDOG_URL.format(link_id=icp["icp_id"], key="RUNTIMEKEY")
    with urllib.request.urlopen(url) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    return decoded.get("companies", [])


# ---------------------------------------------------------------------------
# Snapshot request-key scheme
# ---------------------------------------------------------------------------


def test_request_key_strips_auth_params_and_orders_query():
    with_key = build_snapshot_request(
        "get", "https://api.scrapingdog.com/linkedin?api_key=SECRET1&type=company&linkId=acme"
    )
    reordered_other_key = build_snapshot_request(
        "GET", "https://api.scrapingdog.com/linkedin?linkId=acme&type=company&api_key=SECRET2"
    )
    without_key = build_snapshot_request(
        "GET", "https://api.scrapingdog.com/linkedin?type=company&linkId=acme"
    )
    assert with_key == reordered_other_key == without_key
    assert with_key.provider == "scrapingdog"
    assert with_key.method == "GET"
    assert with_key.endpoint == "api.scrapingdog.com/linkedin"
    assert with_key.request_key.startswith("scrapingdog|GET|api.scrapingdog.com/linkedin|sha256:")


def test_request_key_normalizes_json_bodies_and_separates_params():
    as_text = build_snapshot_request(
        "POST", "https://api.exa.ai/search", body='{"numResults": 5, "query": "devops"}'
    )
    as_bytes = build_snapshot_request(
        "POST", "https://api.exa.ai/search", body=b'{"query": "devops", "numResults": 5}'
    )
    as_mapping = build_snapshot_request(
        "POST", "https://api.exa.ai/search", body={"query": "devops", "numResults": 5}
    )
    assert as_text == as_bytes == as_mapping
    assert as_text.provider == "exa"
    different = build_snapshot_request(
        "POST", "https://api.exa.ai/search", body={"query": "fintech", "numResults": 5}
    )
    assert different.params_hash != as_text.params_hash
    assert different.storage_name != as_text.storage_name


# ---------------------------------------------------------------------------
# Record -> replay round trip + miss policies
# ---------------------------------------------------------------------------


def test_record_replay_round_trip_through_fresh_store(tmp_path):
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    assert recorder.snapshot_count() == 1
    # Re-recording the same request overwrites in place (one file per key).
    _record_companies_response(recorder, "acme", [_rich_company()])
    assert recorder.snapshot_count() == 1

    replayer = _replay_store(tmp_path)
    doc = replayer.replay("GET", SCRAPINGDOG_URL.format(link_id="acme", key="DIFFERENT"))
    assert doc["status"] == 200
    assert doc["body_text"] == body


def test_strict_miss_raises_typed_snapshot_miss(tmp_path):
    _record_store(tmp_path)  # empty set
    replayer = _replay_store(tmp_path)
    with pytest.raises(SnapshotMiss) as exc_info:
        replayer.replay("GET", SCRAPINGDOG_URL.format(link_id="unknown", key="K"))
    assert exc_info.value.request_key.startswith("scrapingdog|GET|")
    assert isinstance(exc_info.value, DevSnapshotStoreError)


def test_empty_miss_policy_returns_synthesized_empty(tmp_path, monkeypatch):
    _record_store(tmp_path)
    replayer = _replay_store(tmp_path, miss_policy="empty")
    exa = replayer.replay("POST", "https://api.exa.ai/search", body={"query": "x"})
    assert exa["body_text"] == '{"results": []}'
    assert exa["synthesized"] == SYNTHESIZED_EMPTY_MARKER
    dog = replayer.replay("GET", SCRAPINGDOG_URL.format(link_id="none", key="K"))
    assert dog["body_text"] == "{}"

    # Env-driven default policy.
    monkeypatch.setenv(SNAPSHOT_MISS_POLICY_ENV, "empty")
    env_replayer = _replay_store(tmp_path)
    assert env_replayer.miss_policy == "empty"
    assert env_replayer.replay("GET", "https://api.exa.ai/contents")["synthesized"] == (
        SYNTHESIZED_EMPTY_MARKER
    )


def test_store_from_env_and_missing_uri(monkeypatch, tmp_path):
    with pytest.raises(DevSnapshotStoreError):
        ProviderSnapshotStore(None)
    monkeypatch.setenv(SNAPSHOT_URI_ENV, str(tmp_path / "snapshot_set"))
    store = ProviderSnapshotStore.from_env()
    assert store.mode == MODE_REPLAY
    assert store.miss_policy == "strict"


def test_record_refuses_secret_material(tmp_path):
    recorder = _record_store(tmp_path)
    request = build_snapshot_request("GET", "https://api.exa.ai/contents?id=1")
    with pytest.raises(DevSnapshotStoreError):
        recorder.record_response(
            request,
            status=200,
            body_text='{"leak": "sk-or-v1-abcdefghijklmnopqrstuvwxyz012345"}',
        )


@pytest.mark.parametrize(
    "secret_value",
    (
        "sk-or-" + "abcdefghijklm123456",
        "sb_secret_" + "abcdefgh",
    ),
)
def test_record_refuses_short_secret_values(tmp_path, secret_value):
    recorder = _record_store(tmp_path)
    request = build_snapshot_request("GET", "https://api.exa.ai/contents?id=short")

    with pytest.raises(DevSnapshotStoreError):
        recorder.record_response(
            request,
            status=200,
            body_text=json.dumps({"leak": secret_value}),
        )


def test_record_allows_secret_names_and_incomplete_prefixes(tmp_path):
    recorder = _record_store(tmp_path)
    request = build_snapshot_request("GET", "https://api.exa.ai/contents?id=1")
    body = json.dumps(
        {
            "configuration": "OPENROUTER_API_KEY",
            "format": "sk-or-",
            "role": "service_role",
        }
    )

    record = recorder.record_response(request, status=200, body_text=body)

    assert record["response"]["body_text"] == body


def test_record_redacts_exact_runtime_secret_without_known_prefix(
    tmp_path, monkeypatch
):
    runtime_secret = "opaque-runtime-credential-value-for-testing"
    monkeypatch.setenv("EXA_API_KEY", runtime_secret)
    recorder = _record_store(tmp_path)
    request = build_snapshot_request("GET", "https://api.exa.ai/contents?id=1")

    record = recorder.record_response(
        request,
        status=200,
        body_text=json.dumps({"echo": runtime_secret}),
    )

    response = record["response"]
    assert response["body_text"] == json.dumps(
        {"echo": SNAPSHOT_RUNTIME_SECRET_REDACTION}
    )
    assert response["runtime_secret_redaction_count"] == 1
    assert _replay_store(tmp_path).lookup(request) == response
    assert runtime_secret not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "snapshot_set").rglob("*")
        if path.is_file()
    )


def test_container_replay_env_shape():
    env = container_replay_env("/mnt/snapshots", miss_policy="empty")
    assert env == {
        SNAPSHOT_DIR_ENV: "/mnt/snapshots",
        SNAPSHOT_MISS_POLICY_ENV: "empty",
    }
    with pytest.raises(DevSnapshotStoreError):
        container_replay_env("/mnt/snapshots", miss_policy="loose")


# ---------------------------------------------------------------------------
# In-container replay bootstrap (subprocess parity with the host key scheme)
# ---------------------------------------------------------------------------


def test_replay_bootstrap_serves_urllib_from_snapshot_dir(tmp_path):
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    probe = (
        "\nimport json, urllib.request\n"
        "with urllib.request.urlopen("
        f"{SCRAPINGDOG_URL.format(link_id='acme', key='CONTAINERKEY')!r}"
        ") as response:\n"
        "    payload = {'status': response.status, 'body': response.read().decode('utf-8')}\n"
        "print(json.dumps(payload))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"), "PATH": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    assert decoded["status"] == 200
    assert decoded["body"] == body


def test_urllib_record_and_replay_preserve_standard_header_contract(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    record_probe = r'''
import http.server
import json
import threading
import urllib.request

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("content-type", "application/json; charset=iso-8859-1")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/headers" % server.server_port
try:
    with urllib.request.urlopen(url) as response:
        outcome = {
            "body": response.read().decode(response.headers.get_content_charset()),
            "charset": response.info().get_content_charset(),
            "header": response.getheader("CONTENT-TYPE"),
            "headers": dict(response.getheaders()),
        }
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"outcome": outcome, "url": url}, sort_keys=True))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert recorded.returncode == 0, recorded.stderr
    recorded_doc = json.loads(recorded.stdout)

    replay_probe = (
        "\nimport json, urllib.request\n"
        f"with urllib.request.urlopen({recorded_doc['url']!r}) as response:\n"
        "    outcome = {\n"
        "        'body': response.read().decode(response.headers.get_content_charset()),\n"
        "        'charset': response.info().get_content_charset(),\n"
        "        'header': response.getheader('CONTENT-TYPE'),\n"
        "        'headers': dict(response.getheaders()),\n"
        "    }\n"
        "print(json.dumps(outcome, sort_keys=True))\n"
    )
    replayed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + replay_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout) == recorded_doc["outcome"]
    assert recorded_doc["outcome"]["body"] == '{"ok": true}'
    assert recorded_doc["outcome"]["charset"] == "iso-8859-1"
    assert recorded_doc["outcome"]["header"] == (
        "application/json; charset=iso-8859-1"
    )


def test_replay_bootstrap_serves_httpx_async_client_from_snapshot_dir(tmp_path):
    pytest.importorskip("httpx")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="CONTAINERASYNCKEY")
    probe = (
        "\nimport asyncio, json, httpx\n"
        "async def _probe():\n"
        f"    async with httpx.AsyncClient() as client:\n"
        f"        response = await client.get({url!r})\n"
        "    print(json.dumps({'status': response.status_code, 'body': response.text}))\n"
        "asyncio.run(_probe())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"), "PATH": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    assert decoded["status"] == 200
    assert decoded["body"] == body


def test_replay_bootstrap_serves_aiohttp_from_snapshot_dir(tmp_path):
    pytest.importorskip("aiohttp")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="CONTAINERAIOHTTPKEY")
    probe = (
        "\nimport asyncio, json, aiohttp\n"
        "async def _probe():\n"
        "    async with aiohttp.ClientSession() as client:\n"
        f"        async with client.get({url!r}) as response:\n"
        "            payload = {'status': response.status, 'body': await response.text()}\n"
        "    print(json.dumps(payload))\n"
        "asyncio.run(_probe())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"), "PATH": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    assert decoded == {"status": 200, "body": body}


def test_replay_bootstrap_strict_miss_fails_loudly(tmp_path):
    _record_store(tmp_path)
    probe = (
        "\nimport urllib.request\n"
        "urllib.request.urlopen('https://api.exa.ai/contents?id=missing')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"), "PATH": ""},
        check=False,
    )
    assert completed.returncode != 0
    assert "RESEARCH_LAB_DEV_SNAPSHOT_MISS:" in completed.stderr


def test_replay_bootstrap_reports_strict_miss_even_when_model_catches_it(tmp_path):
    _record_store(tmp_path)
    probe = (
        "\nimport urllib.request\n"
        "try:\n"
        "    urllib.request.urlopen('https://api.exa.ai/contents?id=missing')\n"
        "except Exception:\n"
        "    pass\n"
        "print('caught')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"), "PATH": ""},
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "caught"
    assert "RESEARCH_LAB_DEV_SNAPSHOT_MISS:" in completed.stderr


def test_record_bootstrap_persists_response_and_skips_secret_material(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = (
        "\nimport json, os\n"
        "_rl_dev_record('GET', 'https://api.exa.ai/search?q=clean', None, 200,"
        " {'content-type': 'application/json'}, '{\"results\": [1]}')\n"
        "_rl_dev_record('GET', 'https://api.exa.ai/search?q=leaky', None, 200,"
        " {'content-type': 'application/json'},"
        " '{\"echo\": \"sk-or-v1-abcdefghijklmnopqrstuvwxyz012345\"}')\n"
        "snapshots = os.path.join(os.environ['RESEARCH_LAB_DEV_SNAPSHOT_DIR'], 'snapshots')\n"
        "names = sorted(os.listdir(snapshots)) if os.path.isdir(snapshots) else []\n"
        "bodies = []\n"
        "for name in names:\n"
        "    with open(os.path.join(snapshots, name), 'r', encoding='utf-8') as handle:\n"
        "        bodies.append(json.load(handle)['response']['body_text'])\n"
        "print(json.dumps({'count': len(names), 'bodies': bodies}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    # The clean response persisted; the one carrying secret material did not.
    assert decoded["count"] == 1
    assert decoded["bodies"] == ['{"results": [1]}']
    failures = [
        json.loads(line)
        for line in (snapshot_dir / "record_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(failures) == 1
    assert failures[0]["reason"] == "secret_value_shape_rejected"
    assert failures[0]["request_key"].startswith("exa|GET|api.exa.ai/search|")


def test_record_bootstrap_rejects_short_secret_values(tmp_path):
    snapshot_dir = tmp_path / "short_secret_record_set"
    probe = (
        "\nimport json, os\n"
        "_rl_dev_record('GET', 'https://api.exa.ai/search?q=short-secret', None, 200,"
        " {'content-type': 'application/json'},"
        " json.dumps({'echo': 'sk-or-' + 'abcdefghijklm123456'}))\n"
        "snapshots = os.path.join(os.environ['RESEARCH_LAB_DEV_SNAPSHOT_DIR'], 'snapshots')\n"
        "names = sorted(os.listdir(snapshots)) if os.path.isdir(snapshots) else []\n"
        "print(json.dumps({'count': len(names)}))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"count": 0}
    failures = [
        json.loads(line)
        for line in (snapshot_dir / "record_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(failures) == 1
    assert failures[0]["reason"] == "secret_value_shape_rejected"


def test_record_bootstrap_redacts_runtime_values_and_allows_secret_names(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    runtime_secret = "opaque-runtime-credential-value-for-testing"
    probe = (
        "\nimport json, os\n"
        "_rl_dev_record('GET', 'https://api.scrapingdog.com/profile?id=clean', None, 200,"
        " {'content-type': 'application/json'},"
        " json.dumps({'role': 'service_role', 'configuration': 'SCRAPINGDOG_API_KEY',"
        " 'documented_prefix': 'sk-or-'}))\n"
        "_rl_dev_record('GET', 'https://api.scrapingdog.com/profile?id=leaky', None, 200,"
        " {'content-type': 'application/json'},"
        " json.dumps({'echo': os.environ['SCRAPINGDOG_API_KEY']}))\n"
        "snapshots = os.path.join(os.environ['RESEARCH_LAB_DEV_SNAPSHOT_DIR'], 'snapshots')\n"
        "names = sorted(os.listdir(snapshots)) if os.path.isdir(snapshots) else []\n"
        "responses = []\n"
        "for name in names:\n"
        "    with open(os.path.join(snapshots, name), 'r', encoding='utf-8') as handle:\n"
        "        responses.append(json.load(handle)['response'])\n"
        "print(json.dumps({'count': len(names), 'responses': responses}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            "SCRAPINGDOG_API_KEY": runtime_secret,
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    decoded = json.loads(completed.stdout)
    assert decoded["count"] == 2
    responses = decoded["responses"]
    assert any(
        response["body_text"]
        == '{"role": "service_role", "configuration": "SCRAPINGDOG_API_KEY", '
        '"documented_prefix": "sk-or-"}'
        for response in responses
    )
    assert any(
        response["body_text"]
        == json.dumps({"echo": SNAPSHOT_RUNTIME_SECRET_REDACTION})
        and response["runtime_secret_redaction_count"] == 1
        for response in responses
    )
    assert not (snapshot_dir / "record_failures.jsonl").exists()
    assert runtime_secret not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    )


def test_record_bootstrap_still_rejects_unknown_secret_shape_after_redaction(
    tmp_path,
):
    snapshot_dir = tmp_path / "record_set"
    runtime_secret = "opaque-runtime-credential-value-for-testing"
    probe = (
        "\nimport json, os\n"
        "_rl_dev_record('GET', 'https://api.scrapingdog.com/profile?id=mixed', None, 200,"
        " {'content-type': 'application/json'},"
        " json.dumps({'echo': os.environ['SCRAPINGDOG_API_KEY'],"
        " 'unknown': 'sk-or-' + 'abcdefghijklm123456'}))\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            "SCRAPINGDOG_API_KEY": runtime_secret,
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    failures = [
        json.loads(line)
        for line in (snapshot_dir / "record_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(failures) == 1
    assert failures[0]["reason"] == "secret_value_shape_rejected"
    assert not (snapshot_dir / "snapshots").exists()
    assert runtime_secret not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    )


def test_urllib_recording_returns_and_replays_the_same_redacted_response(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    runtime_secret = "opaque-runtime-credential-value-for-testing"
    record_probe = r'''
import http.server
import json
import os
import threading
import urllib.request

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "echo": os.environ["SCRAPINGDOG_API_KEY"],
            "ok": True,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/echo" % server.server_port
try:
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"body": body, "url": url}))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            "SCRAPINGDOG_API_KEY": runtime_secret,
            "PATH": "",
        },
        check=False,
    )

    assert recorded.returncode == 0, recorded.stderr
    recorded_doc = json.loads(recorded.stdout)
    expected_body = json.dumps(
        {"echo": SNAPSHOT_RUNTIME_SECRET_REDACTION, "ok": True}
    )
    assert recorded_doc["body"] == expected_body
    snapshot_paths = list((snapshot_dir / "snapshots").glob("*.json"))
    assert len(snapshot_paths) == 1
    persisted = json.loads(snapshot_paths[0].read_text(encoding="utf-8"))
    assert persisted["response"]["body_text"] == expected_body
    assert persisted["response"]["runtime_secret_redaction_count"] == 1
    assert not (snapshot_dir / "record_failures.jsonl").exists()
    assert runtime_secret not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    )

    replay_probe = (
        "\nimport json, urllib.request\n"
        f"with urllib.request.urlopen({recorded_doc['url']!r}) as response:\n"
        "    print(json.dumps({'body': response.read().decode('utf-8')}))\n"
    )
    replayed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + replay_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout) == {"body": expected_body}


def test_urllib_recording_does_not_deliver_unknown_secret_shape(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import os
import threading
import urllib.request

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"unknown": "sk-or-abcdefghijklm123456"}).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/reject" % server.server_port
try:
    try:
        urllib.request.urlopen(url)
    except RuntimeError as exc:
        outcome = type(exc).__name__
    else:
        raise AssertionError("secret-shaped response reached the caller")
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"outcome": outcome}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"outcome": "RuntimeError"}
    failures = [
        json.loads(line)
        for line in (snapshot_dir / "record_failures.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(failures) == 1
    assert failures[0]["reason"] == "secret_value_shape_rejected"
    assert not (snapshot_dir / "snapshots").exists()


def test_recording_http_clients_return_only_redacted_runtime_values(tmp_path):
    pytest.importorskip("aiohttp")
    pytest.importorskip("httpx")
    pytest.importorskip("requests")
    snapshot_dir = tmp_path / "record_set"
    runtime_secret = "opaque-runtime-credential-value-for-testing"
    probe = r'''
import asyncio
import http.server
import json
import os
import threading

import aiohttp
import httpx
import requests

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({
            "echo": os.environ["SCRAPINGDOG_API_KEY"],
            "ok": True,
        }).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/echo" % server.server_port

async def aiohttp_body():
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def httpx_async_body():
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text

try:
    bodies = {
        "aiohttp": asyncio.run(aiohttp_body()),
        "httpx_async": asyncio.run(httpx_async_body()),
        "httpx_sync": httpx.get(url).text,
        "requests": requests.get(url).text,
    }
finally:
    server.shutdown()
    server.server_close()
print(json.dumps(bodies, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            "SCRAPINGDOG_API_KEY": runtime_secret,
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected_body = json.dumps(
        {"echo": SNAPSHOT_RUNTIME_SECRET_REDACTION, "ok": True}
    )
    assert json.loads(completed.stdout) == {
        "aiohttp": expected_body,
        "httpx_async": expected_body,
        "httpx_sync": expected_body,
        "requests": expected_body,
    }
    snapshot_paths = list((snapshot_dir / "snapshots").glob("*.json"))
    assert len(snapshot_paths) == 1
    persisted = json.loads(snapshot_paths[0].read_text(encoding="utf-8"))
    assert persisted["response"]["runtime_secret_redaction_count"] == 1
    assert runtime_secret not in "\n".join(
        path.read_text(encoding="utf-8")
        for path in snapshot_dir.rglob("*")
        if path.is_file()
    )


def test_record_bootstrap_reuses_existing_urllib_response_without_network(tmp_path):
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="REUSEKEY")
    probe = (
        "\nimport json, urllib.request\n"
        f"with urllib.request.urlopen({url!r}) as response:\n"
        "    payload = {'status': response.status, 'body': response.read().decode('utf-8')}\n"
        "print(json.dumps(payload))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"status": 200, "body": body}


def test_record_bootstrap_retry_replaces_existing_urllib_transport_error(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import os
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = b'{"results":["recovered"]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/retry" % server.server_port
assert _rl_dev_record(
    "GET",
    url,
    None,
    0,
    {},
    "",
    response_override={
        "outcome": "urllib_transport_error",
        "error_type": "TimeoutError",
        "reason_type": "timeout",
    },
)
try:
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
finally:
    server.shutdown()
    server.server_close()
snapshot_path = os.path.join(
    os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"],
    "snapshots",
    os.listdir(os.path.join(
        os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots"
    ))[0],
)
with open(snapshot_path, "r", encoding="utf-8") as handle:
    stored = json.load(handle)["response"]
print(json.dumps({"body": body, "hits": len(hits), "stored": stored}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "body": '{"results":["recovered"]}',
        "hits": 1,
        "stored": {
            "body_text": '{"results":["recovered"]}',
            "headers": {"content-type": "application/json"},
            "status": 200,
        },
    }


@pytest.mark.parametrize("status", [408, 425, 429, 500, 503])
def test_record_bootstrap_retry_replaces_existing_transient_http_response(
    tmp_path,
    status,
):
    snapshot_dir = tmp_path / "record_set"
    probe = f'''
import http.server
import json
import os
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = b'{{"results":["recovered"]}}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/retry" % server.server_port
assert _rl_dev_record(
    "GET", url, None, {status}, {{"content-type": "application/json"}},
    '{{"error":"temporary"}}', reason="temporary",
)
try:
    with urllib.request.urlopen(url) as response:
        result = {{
            "status": response.status,
            "body": response.read().decode("utf-8"),
        }}
finally:
    server.shutdown()
    server.server_close()
snapshot_path = os.path.join(
    os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"],
    "snapshots",
    os.listdir(os.path.join(
        os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots"
    ))[0],
)
with open(snapshot_path, "r", encoding="utf-8") as handle:
    stored = json.load(handle)["response"]
print(json.dumps({{"result": result, "hits": len(hits), "stored": stored}}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "result": {"status": 200, "body": '{"results":["recovered"]}'},
        "hits": 1,
        "stored": {
            "body_text": '{"results":["recovered"]}',
            "headers": {"content-type": "application/json"},
            "status": 200,
        },
    }


@pytest.mark.parametrize("status", [400, 403, 404])
def test_record_bootstrap_retry_reuses_existing_nontransient_http_response(
    tmp_path,
    status,
):
    snapshot_dir = tmp_path / "record_set"
    probe = f'''
import http.server
import json
import threading
import urllib.error
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/no-retry" % server.server_port
assert _rl_dev_record(
    "GET", url, None, {status}, {{"content-type": "application/json"}},
    '{{"error":"terminal"}}', reason="terminal",
)
try:
    urllib.request.urlopen(url)
except urllib.error.HTTPError as exc:
    result = {{
        "status": exc.code,
        "body": exc.read().decode("utf-8"),
    }}
else:
    raise AssertionError("expected the recorded HTTP error")
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({{"result": result, "hits": len(hits)}}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            SNAPSHOT_RECORD_RETRY_TRANSIENT_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "result": {"status": status, "body": '{"error":"terminal"}'},
        "hits": 0,
    }


def test_record_bootstrap_reuses_existing_transient_http_without_retry(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import threading
import urllib.error
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        self.send_response(200)
        self.end_headers()
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/no-retry" % server.server_port
assert _rl_dev_record(
    "GET", url, None, 503, {"content-type": "application/json"},
    '{"error":"temporary"}', reason="temporary",
)
try:
    urllib.request.urlopen(url)
except urllib.error.HTTPError as exc:
    result = {"status": exc.code, "body": exc.read().decode("utf-8")}
else:
    raise AssertionError("expected the recorded HTTP error")
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"result": result, "hits": len(hits)}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "result": {"status": 503, "body": '{"error":"temporary"}'},
        "hits": 0,
    }


def test_record_bootstrap_exa_poll_records_only_terminal_response(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import os
import threading
import urllib.request

hits = []
responses = [
    b'{"id":"agent_run_1","status":"running"}',
    b'{"id":"agent_run_1","status":"completed","results":[]}',
]
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = responses.pop(0)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/agent/runs/agent_run_1" % server.server_port
_rl_dev_provider_for_host = lambda _host: "exa"
try:
    with urllib.request.urlopen(url) as response:
        first = json.loads(response.read().decode("utf-8"))
    with urllib.request.urlopen(url) as response:
        second = json.loads(response.read().decode("utf-8"))
finally:
    server.shutdown()
    server.server_close()
snapshot_paths = os.listdir(os.path.join(
    os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots"
))
with open(
    os.path.join(
        os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"],
        "snapshots",
        snapshot_paths[0],
    ),
    "r",
    encoding="utf-8",
) as handle:
    stored = json.loads(json.load(handle)["response"]["body_text"])
print(json.dumps({
    "first": first,
    "hits": len(hits),
    "snapshot_count": len(snapshot_paths),
    "second": second,
    "stored": stored,
}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "first": {"id": "agent_run_1", "status": "running"},
        "hits": 2,
        "snapshot_count": 1,
        "second": {
            "id": "agent_run_1",
            "status": "completed",
            "results": [],
        },
        "stored": {
            "id": "agent_run_1",
            "status": "completed",
            "results": [],
        },
    }


def test_record_bootstrap_exa_poll_replaces_seeded_nonterminal_response(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import os
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = b'{"id":"agent_run_1","status":"completed","results":[]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/agent/runs/agent_run_1" % server.server_port
_rl_dev_provider_for_host = lambda _host: "exa"
provider, request_key, storage_name = _rl_dev_request_identity("GET", url, None)
snapshot_root = os.path.join(
    os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots"
)
os.makedirs(snapshot_root, exist_ok=True)
snapshot_path = os.path.join(snapshot_root, storage_name + ".json")
with open(snapshot_path, "w", encoding="utf-8") as handle:
    json.dump({
        "schema_version": "1.0",
        "record_type": "research_lab_dev_provider_snapshot",
        "request_key": request_key,
        "provider": provider,
        "method": "GET",
        "endpoint": request_key.split("|")[2],
        "params_hash": request_key.split("|")[3],
        "response": {
            "status": 200,
            "headers": {"content-type": "application/json"},
            "body_text": '{"id":"agent_run_1","status":"running"}',
        },
    }, handle)
try:
    with urllib.request.urlopen(url) as response:
        observed = json.loads(response.read().decode("utf-8"))
finally:
    server.shutdown()
    server.server_close()
with open(snapshot_path, "r", encoding="utf-8") as handle:
    stored = json.loads(json.load(handle)["response"]["body_text"])
print(json.dumps({"hits": len(hits), "observed": observed, "stored": stored}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    terminal = {
        "id": "agent_run_1",
        "status": "completed",
        "results": [],
    }
    assert json.loads(completed.stdout) == {
        "hits": 1,
        "observed": terminal,
        "stored": terminal,
    }


def test_record_bootstrap_exa_start_response_remains_replayable(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        hits.append(self.path)
        body = b'{"id":"agent_run_1","status":"running"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/agent/runs" % server.server_port
_rl_dev_provider_for_host = lambda _host: "exa"
request = urllib.request.Request(url, data=b'{"query":"bounded"}', method="POST")
try:
    with urllib.request.urlopen(request) as response:
        first = json.loads(response.read().decode("utf-8"))
    with urllib.request.urlopen(request) as response:
        second = json.loads(response.read().decode("utf-8"))
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"first": first, "hits": len(hits), "second": second}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    expected = {"id": "agent_run_1", "status": "running"}
    assert json.loads(completed.stdout) == {
        "first": expected,
        "hits": 1,
        "second": expected,
    }


def test_record_bootstrap_reuses_existing_httpx_async_response_without_network(
    tmp_path,
):
    pytest.importorskip("httpx")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="ASYNCREUSEKEY")
    probe = (
        "\nimport asyncio, json, httpx\n"
        "async def _probe():\n"
        "    async with httpx.AsyncClient() as client:\n"
        f"        response = await client.get({url!r})\n"
        "    print(json.dumps({'status': response.status_code, 'body': response.text}))\n"
        "asyncio.run(_probe())\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(tmp_path / "snapshot_set"),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {"status": 200, "body": body}


def test_record_bootstrap_calls_live_boundary_for_missing_identity_in_reuse_mode(
    tmp_path,
):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import os
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = b'{"results": ["live"]}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/new-request" % server.server_port
try:
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
finally:
    server.shutdown()
    server.server_close()
snapshots = os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots")
print(json.dumps({"body": body, "hits": len(hits), "snapshots": len(os.listdir(snapshots))}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={
            SNAPSHOT_DIR_ENV: str(snapshot_dir),
            SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true",
            "PATH": "",
        },
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "body": '{"results": ["live"]}',
        "hits": 1,
        "snapshots": 1,
    }


def test_record_bootstrap_without_reuse_flag_keeps_live_recording_semantics(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import http.server
import json
import threading
import urllib.request

hits = []
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        hits.append(self.path)
        body = b'{"source": "live"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/existing-request" % server.server_port
assert _rl_dev_record("GET", url, None, 200, {"content-type": "application/json"}, '{"source": "cached"}')
try:
    with urllib.request.urlopen(url) as response:
        body = response.read().decode("utf-8")
finally:
    server.shutdown()
    server.server_close()
print(json.dumps({"body": body, "hits": len(hits)}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "body": '{"source": "live"}',
        "hits": 1,
    }


def test_urllib_http_error_is_recorded_reused_and_replayed(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    record_probe = r'''
import http.server
import json
import os
import threading
import urllib.error
import urllib.request

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"detail": "not found"}'
        self.send_response(404, "Missing")
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        return

server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
url = "http://127.0.0.1:%d/missing" % server.server_port
try:
    try:
        urllib.request.urlopen(url)
    except urllib.error.HTTPError as exc:
        outcome = {
            "body": exc.read().decode("utf-8"),
            "code": exc.code,
            "content_type": exc.headers.get("content-type"),
            "reason": exc.reason,
        }
    else:
        raise AssertionError("live HTTP error was not preserved")
finally:
    server.shutdown()
    server.server_close()
snapshots = os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots")
failures = os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "record_failures.jsonl")
print(json.dumps({
    "failure_file_exists": os.path.exists(failures),
    "outcome": outcome,
    "snapshot_count": len(os.listdir(snapshots)),
    "url": url,
}))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert recorded.returncode == 0, recorded.stderr
    recorded_doc = json.loads(recorded.stdout)
    expected = {
        "body": '{"detail": "not found"}',
        "code": 404,
        "content_type": "application/json",
        "reason": "Missing",
    }
    assert recorded_doc["outcome"] == expected
    assert recorded_doc["snapshot_count"] == 1
    assert recorded_doc["failure_file_exists"] is False

    replay_probe = (
        "\nimport json, urllib.error, urllib.request\n"
        "try:\n"
        f"    urllib.request.urlopen({recorded_doc['url']!r})\n"
        "except urllib.error.HTTPError as exc:\n"
        "    print(json.dumps({"
        "'body': exc.read().decode('utf-8'), 'code': exc.code, "
        "'content_type': exc.headers.get('content-type'), 'reason': exc.reason}))\n"
        "else:\n"
        "    raise AssertionError('replayed HTTP error was not preserved')\n"
    )
    for bootstrap, extra_env in (
        (
            dev_record_bootstrap(),
            {SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true"},
        ),
        (dev_replay_bootstrap(), {}),
    ):
        replayed = subprocess.run(
            [sys.executable, "-c", bootstrap + replay_probe],
            text=True,
            capture_output=True,
            timeout=60,
            env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": "", **extra_env},
            check=False,
        )
        assert replayed.returncode == 0, replayed.stderr
        assert json.loads(replayed.stdout) == expected
    assert not (snapshot_dir / "record_failures.jsonl").exists()


def test_urllib_url_error_is_recorded_reused_and_replayed(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    record_probe = r'''
import json
import os
import socket
import urllib.error
import urllib.request

sock = socket.socket()
sock.bind(("127.0.0.1", 0))
port = sock.getsockname()[1]
sock.close()
try:
    urllib.request.urlopen("http://127.0.0.1:%d/search?q=bounded" % port, timeout=1)
except urllib.error.URLError as exc:
    outcome = {
        "error_type": type(exc).__name__,
        "reason_type": type(exc.reason).__name__,
    }
else:
    raise AssertionError("connection refusal did not reach urllib")
snapshots = os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "snapshots")
failures = os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "record_failures.jsonl")
print(json.dumps({
    "failure_file_exists": os.path.exists(failures),
    "outcome": outcome,
    "snapshot_count": len(os.listdir(snapshots)),
    "url": "http://127.0.0.1:%d/search?q=bounded" % port,
}))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert recorded.returncode == 0, recorded.stderr
    recorded_doc = json.loads(recorded.stdout)
    expected = {
        "error_type": "URLError",
        "reason_type": "ConnectionRefusedError",
    }
    assert recorded_doc["outcome"] == expected
    assert recorded_doc["snapshot_count"] == 1
    assert recorded_doc["failure_file_exists"] is False

    replay_probe = (
        "\nimport json, urllib.error, urllib.request\n"
        "try:\n"
        f"    urllib.request.urlopen({recorded_doc['url']!r}, timeout=1)\n"
        "except urllib.error.URLError as exc:\n"
        "    print(json.dumps({'error_type': type(exc).__name__, "
        "'reason_type': type(exc.reason).__name__}))\n"
        "else:\n"
        "    raise AssertionError('replayed URL error was not preserved')\n"
    )
    for bootstrap, extra_env in (
        (
            dev_record_bootstrap(),
            {SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true"},
        ),
        (dev_replay_bootstrap(), {}),
    ):
        replayed = subprocess.run(
            [sys.executable, "-c", bootstrap + replay_probe],
            text=True,
            capture_output=True,
            timeout=60,
            env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": "", **extra_env},
            check=False,
        )
        assert replayed.returncode == 0, replayed.stderr
        assert json.loads(replayed.stdout) == expected
    assert not (snapshot_dir / "record_failures.jsonl").exists()


def test_direct_urllib_timeout_is_recorded_and_replayed(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    record_probe = r'''
import json
import os
import socket
import threading
import time
import urllib.request

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(1)

def accept_without_responding():
    connection, _ = listener.accept()
    try:
        time.sleep(1)
    finally:
        connection.close()

threading.Thread(target=accept_without_responding, daemon=True).start()
url = "http://127.0.0.1:%d/probe" % listener.getsockname()[1]
try:
    urllib.request.urlopen(url, timeout=0.05)
except (TimeoutError, socket.timeout):
    outcome = {"is_timeout": True}
else:
    raise AssertionError("timeout did not reach urllib")
finally:
    listener.close()
print(json.dumps({
    "failure_file_exists": os.path.exists(os.path.join(
        os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "record_failures.jsonl"
    )),
    "outcome": outcome,
}))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert recorded.returncode == 0, recorded.stderr
    assert json.loads(recorded.stdout) == {
        "failure_file_exists": False,
        "outcome": {"is_timeout": True},
    }

    replay_probe = r'''
import json
import socket
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:1/probe", timeout=1)
except (TimeoutError, socket.timeout):
    print(json.dumps({"is_timeout": True}))
else:
    raise AssertionError("replayed timeout was not preserved")
'''
    snapshot_files = list((snapshot_dir / "snapshots").glob("*.json"))
    assert len(snapshot_files) == 1
    snapshot_text = snapshot_files[0].read_text(encoding="utf-8")
    assert "live timeout detail" not in snapshot_text
    snapshot = json.loads(snapshot_text)
    assert snapshot["response"] == {
        "outcome": "urllib_transport_error",
        "error_type": "TimeoutError",
        "reason_type": "timeout",
    }
    replay_url = snapshot["request_key"].split("|")[2]
    replay_probe = replay_probe.replace(
        "http://127.0.0.1:1/probe", "http://" + replay_url
    )
    replayed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + replay_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )
    assert replayed.returncode == 0, replayed.stderr
    assert json.loads(replayed.stdout) == {"is_timeout": True}


def test_httpx_async_timeout_is_recorded_reused_and_replayed(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    record_probe = r'''
import asyncio
import httpx
import json
import os
import socket
import threading
import time

listener = socket.socket()
listener.bind(("127.0.0.1", 0))
listener.listen(1)

def accept_without_responding():
    connection, _ = listener.accept()
    try:
        time.sleep(1)
    finally:
        connection.close()

threading.Thread(target=accept_without_responding, daemon=True).start()
url = "http://127.0.0.1:%d/probe" % listener.getsockname()[1]

async def run():
    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            await client.get(url, timeout=0.05)
        except httpx.ReadTimeout as exc:
            return {"error_type": type(exc).__name__}
        raise AssertionError("read timeout did not reach httpx")

try:
    outcome = asyncio.run(run())
finally:
    listener.close()
print(json.dumps({
    "failure_file_exists": os.path.exists(os.path.join(
        os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "record_failures.jsonl"
    )),
    "outcome": outcome,
    "url": url,
}))
'''
    recorded = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + record_probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert recorded.returncode == 0, recorded.stderr
    recorded_doc = json.loads(recorded.stdout)
    assert recorded_doc["outcome"] == {"error_type": "ReadTimeout"}
    assert recorded_doc["failure_file_exists"] is False
    snapshots = list((snapshot_dir / "snapshots").glob("*.json"))
    assert len(snapshots) == 1
    snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
    assert snapshot["response"] == {
        "error_type": "ReadTimeout",
        "outcome": "httpx_transport_error",
    }

    replay_probe = (
        "\nimport asyncio, httpx, json\n"
        "async def run():\n"
        "    async with httpx.AsyncClient(trust_env=False) as client:\n"
        "        try:\n"
        f"            await client.get({recorded_doc['url']!r}, timeout=1)\n"
        "        except httpx.ReadTimeout as exc:\n"
        "            return {'error_type': type(exc).__name__}\n"
        "        raise AssertionError('replayed timeout was not raised')\n"
        "print(json.dumps(asyncio.run(run())))\n"
    )
    for bootstrap, extra_env in (
        (
            dev_record_bootstrap(),
            {SNAPSHOT_RECORD_REUSE_EXISTING_ENV: "true"},
        ),
        (dev_replay_bootstrap(), {}),
    ):
        replayed = subprocess.run(
            [sys.executable, "-c", bootstrap + replay_probe],
            text=True,
            capture_output=True,
            timeout=60,
            env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": "", **extra_env},
            check=False,
        )
        assert replayed.returncode == 0, replayed.stderr
        assert json.loads(replayed.stdout) == {"error_type": "ReadTimeout"}


def test_unsupported_urllib_failure_remains_terminal_recording_evidence(tmp_path):
    snapshot_dir = tmp_path / "record_set"
    probe = r'''
import json
import os
import urllib.request

try:
    urllib.request.urlopen(None)
except Exception as exc:
    caught_type = type(exc).__name__
with open(os.path.join(os.environ["RESEARCH_LAB_DEV_SNAPSHOT_DIR"], "record_failures.jsonl"), "r", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]
print(json.dumps({"caught_type": caught_type, "rows": rows}))
'''
    completed = subprocess.run(
        [sys.executable, "-c", dev_record_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={SNAPSHOT_DIR_ENV: str(snapshot_dir), "PATH": ""},
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    failures = result["rows"]
    assert len(failures) == 1
    assert failures[0]["reason"] == (
        "urllib_live_request_error:" + result["caught_type"]
    )
    assert failures[0]["request_key"].startswith("unknown|GET|")


def test_replay_bootstrap_inert_without_snapshot_dir_env():
    probe = "\nimport urllib.request\nprint(urllib.request.urlopen.__name__)\n"
    completed = subprocess.run(
        [sys.executable, "-c", dev_replay_bootstrap() + probe],
        text=True,
        capture_output=True,
        timeout=60,
        env={"PATH": ""},
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "urlopen"


# ---------------------------------------------------------------------------
# In-process replay seams
# ---------------------------------------------------------------------------


def test_replay_seams_serve_requests_and_httpx(tmp_path):
    requests_lib = pytest.importorskip("requests")
    httpx_lib = pytest.importorskip("httpx")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="SEAMKEY")

    replayer = _replay_store(tmp_path)
    with replayer.replay_installed():
        via_requests = requests_lib.get(url)
        assert via_requests.status_code == 200
        assert via_requests.text == body
        with httpx_lib.Client() as client:
            via_httpx = client.get(url)
        assert via_httpx.status_code == 200
        assert via_httpx.text == body


def test_inprocess_urllib_replay_preserves_http_error(tmp_path):
    import urllib.error
    import urllib.request

    recorder = _record_store(tmp_path)
    url = "https://api.scrapingdog.com/missing?api_key=REDACTED"
    recorder.record_response(
        build_snapshot_request("GET", url),
        status=404,
        body_text='{"detail": "not found"}',
        content_type="application/json",
        reason="Missing",
    )

    with _replay_store(tmp_path).replay_installed():
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(url)

    assert raised.value.code == 404
    assert raised.value.reason == "Missing"
    assert raised.value.headers["content-type"] == "application/json"
    assert raised.value.headers.get_content_charset() is None
    assert raised.value.read() == b'{"detail": "not found"}'


def test_inprocess_urllib_replay_preserves_standard_header_contract(tmp_path):
    import urllib.request

    recorder = _record_store(tmp_path)
    url = "https://api.scrapingdog.com/headers?api_key=REDACTED"
    recorder.record_response(
        build_snapshot_request("GET", url),
        status=200,
        body_text='{"ok": true}',
        content_type="application/json; charset=iso-8859-1",
    )

    with _replay_store(tmp_path).replay_installed():
        response = urllib.request.urlopen(url)

    assert response.headers.get_content_charset() == "iso-8859-1"
    assert response.info() is response.headers
    assert response.getheader("CONTENT-TYPE") == (
        "application/json; charset=iso-8859-1"
    )
    assert dict(response.getheaders()) == {
        "content-type": "application/json; charset=iso-8859-1"
    }


def test_inprocess_urllib_replay_preserves_dns_failure(tmp_path):
    import socket
    import urllib.error
    import urllib.request

    recorder = _record_store(tmp_path)
    url = "https://unresolvable.invalid/probe"
    recorder.record_urllib_transport_error(
        build_snapshot_request("GET", url),
        error=urllib.error.URLError(
            socket.gaierror(socket.EAI_NONAME, "live detail is not persisted")
        ),
    )
    snapshot_text = next((tmp_path / "snapshot_set" / "snapshots").iterdir()).read_text(
        encoding="utf-8"
    )
    assert "live detail" not in snapshot_text

    with _replay_store(tmp_path).replay_installed():
        with pytest.raises(urllib.error.URLError) as raised:
            urllib.request.urlopen(url)

    assert isinstance(raised.value.reason, socket.gaierror)
    assert "live detail" not in str(raised.value)


def test_inprocess_urllib_transport_recording_rejects_http_error(tmp_path):
    import io
    import urllib.error

    recorder = _record_store(tmp_path)
    request = build_snapshot_request("GET", "https://example.invalid/missing")
    with pytest.raises(DevSnapshotStoreError, match="unsupported urllib transport"):
        recorder.record_urllib_transport_error(
            request,
            error=urllib.error.HTTPError(
                request.endpoint,
                404,
                "Missing",
                {},
                io.BytesIO(b"missing"),
            ),
        )


def test_inprocess_httpx_replay_preserves_read_timeout(tmp_path):
    httpx_lib = pytest.importorskip("httpx")
    recorder = _record_store(tmp_path)
    url = "https://api.exa.ai/search?q=timeout"
    snapshot_request = build_snapshot_request("GET", url)
    live_request = httpx_lib.Request("GET", url)
    recorder.record_httpx_transport_error(
        snapshot_request,
        error=httpx_lib.ReadTimeout(
            "live provider detail must not persist",
            request=live_request,
        ),
    )
    snapshot_text = next(
        (tmp_path / "snapshot_set" / "snapshots").iterdir()
    ).read_text(encoding="utf-8")
    assert "live provider detail" not in snapshot_text

    with _replay_store(tmp_path).replay_installed():
        with httpx_lib.Client() as client:
            with pytest.raises(httpx_lib.ReadTimeout) as raised:
                client.get(url)

    assert raised.value.request.url == live_request.url
    assert "live provider detail" not in str(raised.value)


def test_inprocess_httpx_transport_recording_rejects_unsupported_error(tmp_path):
    recorder = _record_store(tmp_path)
    request = build_snapshot_request(
        "GET", "https://api.exa.ai/search?q=bad"
    )

    with pytest.raises(
        DevSnapshotStoreError, match="unsupported httpx transport"
    ):
        recorder.record_httpx_transport_error(
            request,
            error=RuntimeError("not an httpx transport error"),
        )


async def test_replay_seam_serves_httpx_async_client(tmp_path):
    httpx_lib = pytest.importorskip("httpx")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="ASYNCSEAMKEY")

    replayer = _replay_store(tmp_path)
    with replayer.replay_installed():
        async with httpx_lib.AsyncClient() as client:
            via_async = await client.get(url)
        assert via_async.status_code == 200
        assert via_async.text == body


async def test_replay_seam_async_client_strict_miss(tmp_path):
    httpx_lib = pytest.importorskip("httpx")
    _record_store(tmp_path)
    replayer = _replay_store(tmp_path)
    with replayer.replay_installed():
        async with httpx_lib.AsyncClient() as client:
            with pytest.raises(SnapshotMiss):
                await client.get("https://api.exa.ai/search")


async def test_replay_seam_serves_aiohttp_without_live_traffic(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    recorder = _record_store(tmp_path)
    body = _record_companies_response(recorder, "acme", [_rich_company()])
    url = SCRAPINGDOG_URL.format(link_id="acme", key="AIOHTTPKEY")
    replayer = _replay_store(tmp_path)
    with replayer.replay_installed():
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                assert response.status == 200
                assert await response.text() == body


async def test_replay_seam_aiohttp_strict_miss_is_typed(tmp_path):
    aiohttp = pytest.importorskip("aiohttp")
    _record_store(tmp_path)
    replayer = _replay_store(tmp_path)
    with replayer.replay_installed():
        async with aiohttp.ClientSession() as session:
            with pytest.raises(SnapshotMiss):
                await session.get("https://api.exa.ai/search")


def test_replay_installed_requires_replay_mode(tmp_path):
    recorder = _record_store(tmp_path)
    with pytest.raises(DevSnapshotStoreError):
        with recorder.replay_installed():
            pass


# ---------------------------------------------------------------------------
# Deterministic mechanical scorer
# ---------------------------------------------------------------------------


def test_mechanical_scorer_is_stable_across_instances_and_calls():
    icp = _dev_icp(0)
    companies = [_rich_company(), _medium_company(), _mismatched_bucket_company()]
    first = MechanicalDevScorer()(companies, icp, False)
    second = MechanicalDevScorer()(companies, icp, False)
    third = MechanicalDevScorer()(companies, icp, False)
    assert first == second == third
    assert all(0.0 <= score <= 100.0 for score in first)


def test_mechanical_scorer_monotonicity_and_bucket_prefilter():
    icp = _dev_icp(0)
    rich = mechanical_company_score(_rich_company(), icp)
    medium = mechanical_company_score(_medium_company(), icp)
    mismatched = mechanical_company_score(_mismatched_bucket_company(), icp)
    assert rich > medium > mismatched
    # The bucket pre-filter mirrors the live scorer: out-of-bucket scores 0.
    assert mismatched == 0.0
    # Losing intent evidence must strictly lower the score.
    no_evidence = dict(_rich_company())
    no_evidence.pop("intent_signals")
    assert mechanical_company_score(no_evidence, icp) < rich


def test_mechanical_scorer_zeroes_duplicates():
    icp = _dev_icp(0)
    duplicate = dict(_rich_company())
    scores = MechanicalDevScorer()([_rich_company(), duplicate], icp, False)
    assert scores[0] > 0.0
    assert scores[1] == 0.0


def test_dev_icp_score_uses_verifier_capped_top5_arithmetic():
    icp = _dev_icp(0)
    companies = [_rich_company(index) for index in range(7)]
    scores = MechanicalDevScorer()(companies, icp, False)
    expected = per_icp_normalized_score(
        sorted(scores, reverse=True)[:DEV_LEADS_PER_ICP],
        max_leads=DEV_LEADS_PER_ICP,
    )
    top5 = sorted(scores, reverse=True)[:5]
    assert expected == pytest.approx(sum(top5) / 5.0)


# ---------------------------------------------------------------------------
# evaluate_dev
# ---------------------------------------------------------------------------


async def test_evaluate_dev_round_trip_is_deterministic(tmp_path):
    items = _dev_items(2)
    recorder = _record_store(tmp_path)
    _record_companies_response(recorder, "dev-0", [_rich_company(), _medium_company()])
    _record_companies_response(recorder, "dev-1", [_medium_company()])

    replayer = _replay_store(tmp_path)
    first = await evaluate_dev(
        candidate_runner=_urllib_runner,
        dev_items=items,
        snapshot_store=replayer,
        run_label="iteration-1",
        expected_icp_count=len(items),
    )
    second = await evaluate_dev(
        candidate_runner=_urllib_runner,
        dev_items=items,
        snapshot_store=_replay_store(tmp_path),
        run_label="iteration-1",
        expected_icp_count=len(items),
    )
    assert first.to_dict() == second.to_dict()
    assert first.dev_score_version == dev_eval.DEV_SCORE_VERSION
    assert first.ranking_only is True
    assert first.icp_count == 2
    assert first.scored_icp_count == 2
    assert first.failure_count == 0
    assert first.snapshot_miss_count == 0
    assert first.dev_set_hash == compute_dev_set_hash(items)

    # Per-ICP scores ride the verifier's capped top-5 arithmetic.
    scorer = MechanicalDevScorer()
    for row, item in zip(first.per_icp, items):
        replay_body = replayer.replay(
            "GET", SCRAPINGDOG_URL.format(link_id=item["icp"]["icp_id"], key="X")
        )
        companies = json.loads(replay_body["body_text"])["companies"]
        expected_scores = scorer(companies, item["icp"], False)
        expected = per_icp_normalized_score(
            sorted(expected_scores, reverse=True)[:DEV_LEADS_PER_ICP],
            max_leads=DEV_LEADS_PER_ICP,
        )
        assert row["dev_score"] == pytest.approx(expected)
    expected_aggregate = sum(row["dev_score"] for row in first.per_icp) / 2
    assert first.aggregate_dev_score == pytest.approx(expected_aggregate)


async def test_evaluate_dev_strict_miss_books_zero_and_flags(tmp_path):
    items = _dev_items(2)
    recorder = _record_store(tmp_path)
    _record_companies_response(recorder, "dev-0", [_rich_company()])
    # dev-1 has no snapshot: strict replay must book 0 and flag the miss.
    result = await evaluate_dev(
        candidate_runner=_urllib_runner,
        dev_items=items,
        snapshot_store=_replay_store(tmp_path),
        expected_icp_count=len(items),
    )
    assert result.per_icp[0]["dev_score"] > 0.0
    missed = result.per_icp[1]
    assert missed["dev_score"] == 0.0
    assert missed["snapshot_miss"] is True
    assert missed["failure_reason"].startswith("dev_snapshot_miss:scrapingdog|GET|")
    assert result.snapshot_miss_count == 1
    assert result.failure_count == 1
    assert result.scored_icp_count == 1


async def test_evaluate_dev_empty_policy_yields_zero_companies_not_miss(tmp_path):
    items = _dev_items(1)
    _record_store(tmp_path)
    result = await evaluate_dev(
        candidate_runner=_urllib_runner,
        dev_items=items,
        snapshot_store=_replay_store(tmp_path, miss_policy="empty"),
        expected_icp_count=len(items),
    )
    row = result.per_icp[0]
    assert row["snapshot_miss"] is False
    assert row["failure_reason"] == ""
    assert row["zero_output"] is True
    assert row["dev_score"] == 0.0


async def test_evaluate_dev_requires_replay_mode_and_items(tmp_path):
    with pytest.raises(DevEvalError):
        await evaluate_dev(
            candidate_runner=_urllib_runner,
            dev_items=_dev_items(1),
            snapshot_store=_record_store(tmp_path),
            expected_icp_count=1,
        )
    with pytest.raises(DevEvalError):
        await evaluate_dev(
            candidate_runner=_urllib_runner,
            dev_items=[],
            snapshot_store=_replay_store(tmp_path),
            expected_icp_count=1,
        )


async def test_evaluate_dev_crashing_candidate_ranks_zero_without_aborting(tmp_path):
    _record_store(tmp_path)

    def _broken_runner(icp, context):
        raise RuntimeError("candidate exploded")

    result = await evaluate_dev(
        candidate_runner=_broken_runner,
        dev_items=_dev_items(2),
        snapshot_store=_replay_store(tmp_path),
        install_replay_seams=False,
        expected_icp_count=2,
    )
    assert result.aggregate_dev_score == 0.0
    assert result.failure_count == 2
    assert all(
        row["failure_reason"].startswith("dev_runner_error:RuntimeError")
        for row in result.per_icp
    )


async def test_evaluate_dev_manifest_verification(tmp_path):
    items = _dev_items(1)
    recorder = _record_store(tmp_path)
    _record_companies_response(recorder, "dev-0", [_rich_company()])

    # require_manifest with no manifest written yet -> hard error.
    with pytest.raises(DevEvalError):
        await evaluate_dev(
            candidate_runner=_urllib_runner,
            dev_items=items,
            snapshot_store=_replay_store(tmp_path),
            require_manifest=True,
            expected_icp_count=len(items),
        )

    recorder.write_dev_icp_items(items)
    manifest = recorder.build_manifest(icp_set_hash=compute_dev_set_hash(items))
    recorder.write_manifest(manifest)
    result = await evaluate_dev(
        candidate_runner=_urllib_runner,
        dev_items=items,
        snapshot_store=_replay_store(tmp_path),
        require_manifest=True,
        expected_icp_count=len(items),
    )
    assert result.snapshot_manifest_hash == manifest["manifest_hash"]

    # Tampering with a stored snapshot must fail verification before scoring.
    snapshot_file = next((tmp_path / "snapshot_set" / "snapshots").glob("*.json"))
    record = json.loads(snapshot_file.read_text(encoding="utf-8"))
    record["response"]["body_text"] = '{"companies": []}'
    snapshot_file.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(DevEvalError, match="content_hash_mismatch"):
        await evaluate_dev(
            candidate_runner=_urllib_runner,
            dev_items=items,
            snapshot_store=_replay_store(tmp_path),
            expected_icp_count=len(items),
        )


# ---------------------------------------------------------------------------
# Dev-set discipline (leak-cluster guard)
# ---------------------------------------------------------------------------


def _current_day_bank_items(count: int = 20) -> list[dict]:
    industries = [
        "Legal Services",
        "Biotechnology Research",
        "Financial Services",
        "Manufacturing",
        "Hospitals and Health Care",
        "Education",
        "Advertising Services",
        "Cybersecurity",
        "Retail",
        "Telecommunications",
    ]
    items: list[dict] = []
    for index in range(count):
        industry = industries[index % len(industries)]
        icp = {
            **_dev_icp(index),
            "industry": industry,
            "sub_industry": f"{industry} segment {index}",
            "product_service": f"{industry} intent platform {index}",
            "intent_signals": [f"{industry} buying intent signal {index}"],
            "intent_signal": f"{industry} buying intent signal {index}",
        }
        items.append(
            {
                "icp": icp,
                "icp_ref": f"current-day:{index}",
                "icp_hash": sha256_json({"icp": icp}),
                "baseline_score": float(index),
                "benchmark_category": (
                    "public" if index < count // 2 else "private"
                ),
                "set_id": 100,
                "day_index": 1,
                "day_rank": index + 1,
            }
        )
    return items


def _current_day_bank(items: list[dict]):
    return build_current_day_dev_bank(
        items,
        benchmark_date="2026-07-16",
        benchmark_bundle_id="private_benchmark:" + "1" * 64,
        benchmark_bundle_hash="sha256:" + "2" * 64,
        rolling_window_hash="sha256:" + "3" * 64,
        private_model_manifest_hash="sha256:" + "4" * 64,
        evaluation_epoch=24000,
    )


def test_current_day_selector_uses_three_weak_and_two_strong_across_visibility():
    bank = _current_day_bank(_current_day_bank_items())
    selected = select_current_day_dev_icps(
        bank.items,
        size=5,
        seed="tree-run-1",
        miner_direction="improve legal industry intent finding",
        bank_manifest=bank.manifest,
    )

    assert selected.manifest["weak_count"] == 3
    assert selected.manifest["strong_count"] == 2
    assert "current-day:0" in {row["icp_ref"] for row in selected.items}
    assert sum(float(row["baseline_score"]) < 10 for row in selected.items) == 3
    assert sum(float(row["baseline_score"]) >= 10 for row in selected.items) == 2
    assert {row["benchmark_category"] for row in selected.items} == {
        "public",
        "private",
    }

    repeated = select_current_day_dev_icps(
        bank.items,
        size=5,
        seed="tree-run-1",
        miner_direction="improve legal industry intent finding",
        bank_manifest=bank.manifest,
    )
    assert repeated == selected


def test_current_day_selector_prioritizes_direction_relevant_weak_case():
    bank = _current_day_bank(_current_day_bank_items())
    legal = select_current_day_dev_icps(
        bank.items,
        size=1,
        seed="same-tree",
        miner_direction="legal services intent",
        bank_manifest=bank.manifest,
    )
    biotech = select_current_day_dev_icps(
        bank.items,
        size=1,
        seed="same-tree",
        miner_direction="biotechnology research intent",
        bank_manifest=bank.manifest,
    )
    assert legal.items[0]["icp"]["industry"] == "Legal Services"
    assert biotech.items[0]["icp"]["industry"] == "Biotechnology Research"
    assert legal.dev_set_hash != biotech.dev_set_hash


def test_current_day_selector_keeps_strong_regression_guards_diverse():
    items = _current_day_bank_items()
    for index in range(10, 20):
        items[index]["icp"]["industry"] = "Legal Services"
        items[index]["icp"]["sub_industry"] = "Legal software"
        items[index]["icp"]["country"] = "United States"
        items[index]["icp"]["employee_count"] = "51-200"
        items[index]["icp_hash"] = sha256_json({"icp": items[index]["icp"]})
    items[10]["icp"]["industry"] = "Financial Services"
    items[10]["icp"]["sub_industry"] = "Payments"
    items[10]["icp"]["country"] = "Canada"
    items[10]["icp"]["employee_count"] = "201-500"
    items[10]["icp_hash"] = sha256_json({"icp": items[10]["icp"]})
    bank = _current_day_bank(items)

    selected = select_current_day_dev_icps(
        bank.items,
        size=5,
        seed="tree-run-diversity",
        miner_direction="improve legal industry intent finding",
        bank_manifest=bank.manifest,
    )

    strong = [
        row for row in selected.items if float(row["baseline_score"]) >= 10
    ]
    assert len(strong) == 2
    assert {row["icp"]["industry"] for row in strong} == {
        "Financial Services",
        "Legal Services",
    }


@pytest.mark.parametrize("failure", ["duplicate_ref", "duplicate_hash", "duplicate_intent", "nan_score"])
def test_current_day_bank_rejects_ambiguous_or_invalid_inputs(failure):
    items = _current_day_bank_items(6)
    if failure == "duplicate_ref":
        items[1]["icp_ref"] = items[0]["icp_ref"]
    elif failure == "duplicate_hash":
        items[1]["icp_hash"] = items[0]["icp_hash"]
    elif failure == "duplicate_intent":
        items[1]["intent_signal_signature"] = dev_eval.intent_signal_signature(
            items[0]["icp"]
        )
        items[0]["intent_signal_signature"] = items[1][
            "intent_signal_signature"
        ]
    else:
        items[1]["baseline_score"] = float("nan")
    with pytest.raises(DevEvalError):
        _current_day_bank(items)


def test_build_dev_icp_set_hard_excludes_holdout_matches():
    source = _dev_items(8)
    excluded_by_hash = source[0]
    excluded_by_bare_hash = source[1]
    excluded_by_ref = source[2]
    excluded_by_signature = source[3]
    exclusions = [
        excluded_by_hash["icp_hash"],
        excluded_by_bare_hash["icp_hash"].split(":", 1)[1],
        excluded_by_ref["icp_ref"],
        dev_eval.intent_signal_signature(excluded_by_signature["icp"]),
    ]
    dev_set = build_dev_icp_set(
        source, exclude_window_hashes=exclusions, size=3, seed="dev-v1"
    )
    selected_hashes = {item["icp_hash"] for item in dev_set.items}
    for excluded in (excluded_by_hash, excluded_by_bare_hash, excluded_by_ref, excluded_by_signature):
        assert excluded["icp_hash"] not in selected_hashes

    proof = dev_set.manifest["exclusion_proof"]
    assert proof["excluded_item_count"] == 4
    assert proof["selected_overlap_with_exclusions"] == []
    matched_on = {
        entry["icp_ref"]: entry["matched_on"] for entry in proof["excluded_items"]
    }
    assert matched_on[excluded_by_ref["icp_ref"]] == ["icp_ref"]
    assert matched_on[excluded_by_hash["icp_ref"]] == ["icp_hash"]
    assert matched_on[excluded_by_bare_hash["icp_ref"]] == ["icp_hash"]
    assert matched_on[excluded_by_signature["icp_ref"]] == ["intent_signal_signature"]


def test_build_dev_icp_set_is_deterministic_per_seed():
    source = _dev_items(8)
    first = build_dev_icp_set(source, exclude_window_hashes=[], size=4, seed="dev-v1")
    second = build_dev_icp_set(source, exclude_window_hashes=[], size=4, seed="dev-v1")
    assert first.manifest == second.manifest
    assert first.dev_set_hash == second.dev_set_hash
    assert first.items == second.items
    other_seed = build_dev_icp_set(source, exclude_window_hashes=[], size=4, seed="dev-v2")
    assert other_seed.manifest["selection_seed"] == "dev-v2"

    # Manifest self-hash integrity.
    payload = {
        key: value for key, value in first.manifest.items() if key != "manifest_hash"
    }
    assert first.manifest["manifest_hash"] == sha256_json(payload)


def test_build_dev_icp_set_deterministically_maximizes_available_diversity():
    dimensions = [
        ("Software Development", "DevOps", "United States", "51-200"),
        ("Financial Services", "Payments", "United Kingdom", "201-500"),
        ("Hospitals and Health Care", "Telehealth", "Canada", "11-50"),
        ("Manufacturing", "Industrial Automation", "Germany", "501-1,000"),
    ]
    rows = []
    for index in range(12):
        industry, sub_industry, country, employee_count = dimensions[
            0 if index < 9 else index - 8
        ]
        icp = {
            **_dev_icp(index),
            "industry": industry,
            "sub_industry": sub_industry,
            "country": country,
            "geography": country,
            "employee_count": employee_count,
        }
        rows.append(
            {
                "icp": icp,
                "icp_ref": f"diverse:{index}",
                "icp_hash": sha256_json({"icp": icp}),
            }
        )

    selected = build_dev_icp_set(
        rows,
        exclude_window_hashes=[],
        size=4,
        seed="diversity-proof",
    )
    assert len({row["icp"]["industry"] for row in selected.items}) == 4
    assert len({row["icp"]["country"] for row in selected.items}) == 4
    assert len({row["icp"]["employee_count"] for row in selected.items}) == 4
    assert selected.manifest["selection_policy"] == "seeded_greedy_diversity_v1"
    assert selected.manifest["diversity_proof"]["selected_unique_counts"] == {
        "industry": 4,
        "sub_industry": 4,
        "country_or_geography": 4,
        "employee_count": 4,
    }


def test_build_dev_icp_set_fails_when_exclusions_starve_the_pool():
    source = _dev_items(4)
    exclusions = [item["icp_hash"] for item in source[:3]]
    with pytest.raises(DevEvalError, match="dev_icp_set_requires_2_eligible_icps_found_1"):
        build_dev_icp_set(source, exclude_window_hashes=exclusions, size=2, seed="dev-v1")


# ---------------------------------------------------------------------------
# Snapshot-set manifest integrity
# ---------------------------------------------------------------------------


def test_manifest_round_trip_and_tamper_detection(tmp_path):
    recorder = _record_store(tmp_path)
    _record_companies_response(recorder, "acme", [_rich_company()])
    _record_companies_response(recorder, "globex", [_medium_company()])
    dev_set = build_dev_icp_set(_dev_items(4), exclude_window_hashes=[], size=2, seed="dev-v1")
    recorder.write_dev_icp_items(dev_set.items)
    manifest = recorder.build_manifest(
        icp_set_hash=dev_set.dev_set_hash,
        dev_set_manifest=dev_set.manifest,
        recorded_at="2026-07-02T00:00:00Z",
    )
    recorder.write_manifest(manifest)

    replayer = _replay_store(tmp_path)
    verification = replayer.verify_manifest(expected_icp_set_hash=dev_set.dev_set_hash)
    assert verification["passed"], verification["errors"]
    assert verification["manifest_hash"] == manifest["manifest_hash"]
    assert manifest["snapshot_count"] == 2
    assert len(manifest["request_keys"]) == 2

    # Wrong expected ICP set hash is caught.
    mismatch = replayer.verify_manifest(expected_icp_set_hash="sha256:" + "0" * 64)
    assert "icp_set_hash_mismatch" in mismatch["errors"]

    # Tampered manifest self-hash is caught.
    tampered = {**manifest, "manifest_hash": "sha256:" + "0" * 64}
    assert "manifest_hash_mismatch" in replayer.verify_manifest(tampered)["errors"]

    # Tampered snapshot content is caught.
    snapshot_file = next((tmp_path / "snapshot_set" / "snapshots").glob("*.json"))
    record = json.loads(snapshot_file.read_text(encoding="utf-8"))
    record["response"]["status"] = 500
    snapshot_file.write_text(json.dumps(record), encoding="utf-8")
    broken = replayer.verify_manifest()
    assert not broken["passed"]
    assert "content_hash_mismatch" in broken["errors"]


def test_dormant_modules_are_not_imported_by_eval_package():
    # The harness must stay dormant: importing the eval package must not pull
    # dev_eval/snapshot_store in (flag consumers land in a later wave).
    code = (
        "import sys\n"
        "import research_lab.eval\n"
        "assert 'research_lab.eval.dev_eval' not in sys.modules\n"
        "assert 'research_lab.eval.snapshot_store' not in sys.modules\n"
        "import research_lab.eval.dev_eval, research_lab.eval.snapshot_store\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"

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
    build_dev_icp_set,
    compute_dev_set_hash,
    evaluate_dev,
    mechanical_company_score,
)
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    MODE_REPLAY,
    SNAPSHOT_DIR_ENV,
    SNAPSHOT_MISS_POLICY_ENV,
    SNAPSHOT_URI_ENV,
    SYNTHESIZED_EMPTY_MARKER,
    DevSnapshotStoreError,
    ProviderSnapshotStore,
    SnapshotMiss,
    build_snapshot_request,
    container_replay_env,
    dev_replay_bootstrap,
)

DEV_ENV_VARS = (SNAPSHOT_URI_ENV, SNAPSHOT_MISS_POLICY_ENV, SNAPSHOT_DIR_ENV)

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
        recorder.record_response(request, status=200, body_text='{"leak": "sk-or-abc123"}')


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


def test_record_bootstrap_persists_response_and_skips_secret_material(tmp_path):
    from research_lab.eval.snapshot_store import dev_record_bootstrap

    snapshot_dir = tmp_path / "record_set"
    probe = (
        "\nimport json, os\n"
        "_rl_dev_record('GET', 'https://api.exa.ai/search?q=clean', None, 200,"
        " {'content-type': 'application/json'}, '{\"results\": [1]}')\n"
        "_rl_dev_record('GET', 'https://api.exa.ai/search?q=leaky', None, 200,"
        " {'content-type': 'application/json'}, '{\"echo\": \"sk-or-abc123\"}')\n"
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
    assert failures[0]["reason"] == "secret_material_rejected"
    assert failures[0]["request_key"].startswith("exa|GET|api.exa.ai/search|")


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

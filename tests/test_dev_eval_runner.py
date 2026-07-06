"""Tests for the worker-side docker replay dev evaluator (§6.3 activation).

Covers:
- factory gating (flag off / URI missing / configured),
- local snapshot-set resolution and the missing-manifest guard,
- dev ICP payload loading bound to the manifest icp_set_hash (tamper guard),
- end-to-end candidate evaluation through the runner seam (fake docker
  invocation, real snapshot store + evaluate_dev arithmetic) and its
  determinism,
- immutable-digest requirement and best-effort failure shape.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from gateway.research_lab.dev_eval_runner import (
    DEV_EVAL_ENABLED_ENV,
    DevEvalRunnerError,
    DockerReplayDevEvaluator,
    build_code_edit_dev_evaluator,
    ensure_local_snapshot_set,
    load_verified_dev_items,
)
from research_lab.canonical import sha256_json
from research_lab.eval.dev_eval import DEV_SCORE_VERSION, compute_dev_set_hash
from research_lab.eval.snapshot_store import (
    MODE_RECORD,
    MODE_REPLAY,
    SNAPSHOT_URI_ENV,
    ProviderSnapshotStore,
    build_snapshot_request,
)

IMAGE_DIGEST = "123456789.dkr.ecr.test/model@sha256:" + "a" * 64

SCRAPINGDOG_URL = (
    "https://api.scrapingdog.com/linkedin?type=company&linkId={link_id}&api_key={key}"
)


@pytest.fixture(autouse=True)
def _clear_dev_env(monkeypatch):
    monkeypatch.delenv(DEV_EVAL_ENABLED_ENV, raising=False)
    monkeypatch.delenv(SNAPSHOT_URI_ENV, raising=False)


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


def _write_snapshot_set(tmp_path, items: list[dict]) -> str:
    """Record a complete snapshot set (snapshots + manifest + dev ICPs)."""
    root = str(tmp_path / "snapshot_set")
    store = ProviderSnapshotStore(root, mode=MODE_RECORD)
    for item in items:
        body = json.dumps({"companies": [_rich_company(0), _rich_company(1)]})
        request = build_snapshot_request(
            "GET", SCRAPINGDOG_URL.format(link_id=item["icp"]["icp_id"], key="RECORDKEY")
        )
        store.record_response(request, status=200, body_text=body)
    manifest = store.build_manifest(
        icp_set_hash=compute_dev_set_hash(items),
        dev_set_manifest={"manifest_type": "research_lab_dev_icp_set"},
        recorded_at="2026-07-06T00:00:00Z",
    )
    store.write_manifest(manifest)
    store.write_dev_icp_items(items)
    return root


def _candidate(node_id: str = "node-1", image_digest: str = IMAGE_DIGEST):
    return SimpleNamespace(
        node_id=node_id,
        build=SimpleNamespace(
            candidate_model_manifest=SimpleNamespace(image_digest=image_digest)
        ),
    )


def _fake_docker_runner(companies_by_icp_id: dict[str, list[dict]]):
    calls: list[dict] = []

    def _run(*, image_digest, icp, context, snapshot_dir, timeout_seconds):
        calls.append(
            {
                "image_digest": image_digest,
                "icp_id": icp.get("icp_id"),
                "snapshot_dir": str(snapshot_dir),
                "timeout_seconds": timeout_seconds,
                "context_dev_eval": context.get("dev_eval"),
            }
        )
        return companies_by_icp_id.get(str(icp.get("icp_id")), [])

    _run.calls = calls
    return _run


# ---------------------------------------------------------------------------
# Factory gating
# ---------------------------------------------------------------------------


def test_factory_returns_none_when_flag_off(monkeypatch, tmp_path):
    monkeypatch.setenv(SNAPSHOT_URI_ENV, str(tmp_path))
    assert build_code_edit_dev_evaluator() is None


def test_factory_returns_none_without_snapshot_uri(monkeypatch):
    monkeypatch.setenv(DEV_EVAL_ENABLED_ENV, "true")
    assert build_code_edit_dev_evaluator() is None


def test_factory_returns_evaluator_when_configured(monkeypatch, tmp_path):
    monkeypatch.setenv(DEV_EVAL_ENABLED_ENV, "true")
    monkeypatch.setenv(SNAPSHOT_URI_ENV, str(tmp_path / "snapshot_set"))
    evaluator = build_code_edit_dev_evaluator()
    assert isinstance(evaluator, DockerReplayDevEvaluator)


# ---------------------------------------------------------------------------
# Snapshot-set resolution + dev ICP payload guard
# ---------------------------------------------------------------------------


def test_ensure_local_snapshot_set_uses_local_dir_in_place(tmp_path):
    items = _dev_items(2)
    root = _write_snapshot_set(tmp_path, items)
    assert ensure_local_snapshot_set(root) == __import__("pathlib").Path(root)


def test_ensure_local_snapshot_set_requires_manifest(tmp_path):
    empty = tmp_path / "empty_set"
    empty.mkdir()
    with pytest.raises(DevEvalRunnerError, match="manifest missing"):
        ensure_local_snapshot_set(str(empty))


def test_load_verified_dev_items_round_trip_and_tamper_guard(tmp_path):
    items = _dev_items(2)
    root = _write_snapshot_set(tmp_path, items)
    replay = ProviderSnapshotStore(root, mode=MODE_REPLAY)
    assert load_verified_dev_items(replay) == [dict(item) for item in items]

    # Swap in a different ICP set without re-recording the manifest.
    tampered = ProviderSnapshotStore(root, mode=MODE_RECORD)
    tampered.write_dev_icp_items(_dev_items(3))
    with pytest.raises(DevEvalRunnerError, match="icp_set_hash"):
        load_verified_dev_items(replay)


def test_load_verified_dev_items_requires_payloads(tmp_path):
    items = _dev_items(2)
    root = _write_snapshot_set(tmp_path, items)
    (__import__("pathlib").Path(root) / "dev_icps.json").unlink()
    replay = ProviderSnapshotStore(root, mode=MODE_REPLAY)
    with pytest.raises(DevEvalRunnerError, match="dev ICP payloads"):
        load_verified_dev_items(replay)


# ---------------------------------------------------------------------------
# End-to-end evaluation through the runner seam
# ---------------------------------------------------------------------------


async def test_evaluator_scores_candidate_and_is_deterministic(tmp_path):
    items = _dev_items(3)
    root = _write_snapshot_set(tmp_path, items)
    companies = {
        "dev-0": [_rich_company(0), _rich_company(1)],
        "dev-1": [_rich_company(2)],
        "dev-2": [],
    }
    fake_docker = _fake_docker_runner(companies)
    evaluator = DockerReplayDevEvaluator(
        snapshot_uri=root, run_icp_in_docker=fake_docker
    )

    first = await evaluator(_candidate())
    second = await evaluator(_candidate())

    assert first == second
    assert first["dev_score_version"] == DEV_SCORE_VERSION
    assert first["ranking_only"] is True
    assert first["icp_count"] == 3
    assert len(first["per_icp"]) == 3
    assert first["aggregate_dev_score"] > 0.0
    # dev-2 returned zero companies: booked as a failure, not an abort.
    assert first["failure_count"] == 1
    # The fake docker seam received the candidate image and the local dir.
    assert all(call["image_digest"] == IMAGE_DIGEST for call in fake_docker.calls)
    assert all(call["snapshot_dir"] == root for call in fake_docker.calls)
    assert all(call["context_dev_eval"] is True for call in fake_docker.calls)


async def test_evaluator_rejects_mutable_image_reference(tmp_path):
    items = _dev_items(2)
    root = _write_snapshot_set(tmp_path, items)
    evaluator = DockerReplayDevEvaluator(
        snapshot_uri=root, run_icp_in_docker=_fake_docker_runner({})
    )
    with pytest.raises(DevEvalRunnerError, match="immutable"):
        await evaluator(_candidate(image_digest="registry/model:latest"))


async def test_evaluator_books_docker_failures_per_icp_without_aborting(tmp_path):
    items = _dev_items(2)
    root = _write_snapshot_set(tmp_path, items)

    def _crashing(*, image_digest, icp, context, snapshot_dir, timeout_seconds):
        raise DevEvalRunnerError("adapter exploded")

    evaluator = DockerReplayDevEvaluator(snapshot_uri=root, run_icp_in_docker=_crashing)
    result = await evaluator(_candidate())
    assert result["aggregate_dev_score"] == 0.0
    assert result["failure_count"] == 2
    assert result["icp_count"] == 2

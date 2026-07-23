from __future__ import annotations

import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest
import requests

from Leadpoet.utils import cloud_db
from gateway.fulfillment.models import FulfillmentScoreResult
from gateway.research_lab.worker_autostart import (
    ResearchLabWorkerStartupError,
    build_research_lab_worker_environment,
)
from neurons.validator import (
    Validator,
    _ValidatorEpochState,
    detect_fulfillment_worker_ids,
)
from qualification.scoring.fulfillment_scorer import format_scores_for_gateway

ROOT = Path(__file__).resolve().parents[1]


def _clear_fulfillment_proxies(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(os.environ):
        if name.startswith("FULFILLMENT_WEBSHARE_PROXY_"):
            monkeypatch.delenv(name, raising=False)


def _epoch_state(epoch: int = 24_105) -> _ValidatorEpochState:
    return _ValidatorEpochState(
        current_block=8_682_450,
        workflow_epoch_id=epoch,
        epoch_block=245,
        blocks_remaining=115,
        epoch_start_block=8_682_205,
        next_epoch_block=8_682_565,
        tempo=360,
        subnet_epoch_index=epoch,
        epoch_ref="sha256:" + "1" * 64,
    )


def _validator_stub(*, last_processed_epoch: int) -> SimpleNamespace:
    return SimpleNamespace(
        wallet=SimpleNamespace(),
        _last_processed_epoch=last_processed_epoch,
    )


def _work_paths(
    tmp_path: Path,
    *,
    worker_id: int = 1,
    epoch: int = 24_105,
    request_id: str = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358",
) -> tuple[Path, Path]:
    weights = tmp_path / "validator_weights"
    weights.mkdir(exist_ok=True)
    work = weights / (f"fulfillment_worker_{worker_id}_work_{epoch}_{request_id}.json")
    results = weights / work.name.replace("_work_", "_results_", 1)
    return work, results


def test_fulfillment_worker_detection_uses_all_configured_numeric_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_fulfillment_proxies(monkeypatch)
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_10", "http://proxy-10")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_3", "http://proxy-3")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_1", "http://proxy-1")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_0", "ignored")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_7", "")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_BAD", "ignored")
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_11", "ignored")

    assert detect_fulfillment_worker_ids() == [1, 3, 10]


@pytest.mark.asyncio
async def test_sparse_worker_ids_receive_work_without_renumbering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    for worker_id in (1, 3, 10):
        monkeypatch.setenv(
            f"FULFILLMENT_WEBSHARE_PROXY_{worker_id}",
            f"http://proxy-{worker_id}",
        )
    request_id = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358"
    submissions = [
        {
            "submission_id": f"submission-{index}",
            "miner_hotkey": f"miner-{index}",
            "lead_ids": [f"lead-{index}"],
            "leads": [{"lead": index}],
        }
        for index in range(4)
    ]
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {
            "requests": [{
                "request_id": request_id,
                "status": "scoring",
                "icp": {},
                "submissions": submissions,
            }]
        },
    )

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=-1),
        _epoch_state(),
    )

    files = list((tmp_path / "validator_weights").glob("*_work_*.json"))
    assert {int(path.name.split("_")[2]) for path in files} == {1, 3, 10}
    assigned = [
        row["submission_id"]
        for path in files
        for row in json.loads(path.read_text(encoding="utf-8"))["submissions"]
    ]
    assert sorted(assigned) == sorted(row["submission_id"] for row in submissions)


@pytest.mark.asyncio
async def test_single_request_fans_out_across_all_ten_fulfillment_workers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    for worker_id in range(1, 11):
        monkeypatch.setenv(
            f"FULFILLMENT_WEBSHARE_PROXY_{worker_id}",
            f"http://proxy-{worker_id}",
        )

    request_id = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358"
    submissions = [
        {
            "submission_id": f"submission-{index}",
            "miner_hotkey": f"miner-{index}",
            "lead_ids": [f"lead-{index}"],
            "leads": [{"lead": index}],
        }
        for index in range(15)
    ]
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {
            "requests": [
                {
                    "request_id": request_id,
                    "status": "scoring",
                    "icp": {"prompt": "test"},
                    "submissions": submissions,
                }
            ]
        },
    )

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=-1),
        _epoch_state(),
    )

    files = sorted((tmp_path / "validator_weights").glob("*_work_*.json"))
    assert len(files) == 10
    observed_workers = {int(path.name.split("_")[2]) for path in files}
    assert observed_workers == set(range(1, 11))
    assigned_submissions = []
    for path in files:
        assigned_submissions.extend(
            json.loads(path.read_text(encoding="utf-8"))["submissions"]
        )
    assert {row["submission_id"] for row in assigned_submissions} == {
        row["submission_id"] for row in submissions
    }
    assert len(assigned_submissions) == len(submissions)


@pytest.mark.asyncio
async def test_false_gateway_ack_keeps_fulfillment_files_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_1", "http://proxy-1")
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {"requests": []},
    )
    monkeypatch.setattr(
        cloud_db,
        "gateway_submit_fulfillment_scores",
        lambda *_args, **_kwargs: False,
    )

    epoch = 24_105
    request_id = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358"
    work, results = _work_paths(
        tmp_path,
        epoch=epoch,
        request_id=request_id,
    )
    work.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "request_id": request_id,
                "icp": {},
                "submissions": [],
            }
        ),
        encoding="utf-8",
    )
    results.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "fulfillment_worker_id": 1,
                "request_id": request_id,
                "submission_results": [
                    {
                        "miner_hotkey": "miner-1",
                        "submission_id": "submission-1",
                        "lead_ids": ["lead-1"],
                        "results": [
                            FulfillmentScoreResult(
                                lead_id="lead-1",
                                failure_reason="test",
                            ).model_dump()
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=epoch),
        _epoch_state(epoch),
    )

    assert work.exists()
    assert results.exists()


@pytest.mark.asyncio
async def test_worker_infrastructure_error_retries_without_scoring_miners_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_1", "http://proxy-1")
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {"requests": []},
    )

    epoch = 24_105
    request_id = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358"
    work, results = _work_paths(
        tmp_path,
        epoch=epoch,
        request_id=request_id,
    )
    work.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "request_id": request_id,
                "icp": {},
                "submissions": [{"submission_id": "submission-1"}],
            }
        ),
        encoding="utf-8",
    )
    original_mtime = work.stat().st_mtime
    results.write_text(
        json.dumps(
            {
                "epoch": epoch,
                "fulfillment_worker_id": 1,
                "request_id": request_id,
                "error_type": "RuntimeError",
                "error": "provider infrastructure unavailable",
                "submission_results": [],
            }
        ),
        encoding="utf-8",
    )

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=epoch),
        _epoch_state(epoch),
    )

    assert work.exists()
    assert work.stat().st_mtime >= original_mtime
    assert not results.exists()


@pytest.mark.asyncio
async def test_empty_worker_error_message_is_still_retryable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_1", "http://proxy-1")
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {"requests": []},
    )
    epoch = 24_105
    work, results = _work_paths(tmp_path, epoch=epoch)
    work.write_text(json.dumps({"epoch": epoch, "submissions": [{}]}))
    results.write_text(json.dumps({
        "epoch": epoch,
        "request_id": "43c6bc55-80f9-49e3-af0c-7a6ae6e39358",
        "error_type": "TimeoutError",
        "error": "",
        "submission_results": [],
    }))

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=epoch),
        _epoch_state(epoch),
    )

    assert work.exists()
    assert not results.exists()


@pytest.mark.asyncio
async def test_pending_delivery_blocks_redispatch_after_scoring_lease_expires(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ENABLE_FULFILLMENT", "true")
    _clear_fulfillment_proxies(monkeypatch)
    monkeypatch.setenv("FULFILLMENT_WEBSHARE_PROXY_1", "http://proxy-1")
    request_id = "43c6bc55-80f9-49e3-af0c-7a6ae6e39358"
    submission = {
        "submission_id": "submission-1",
        "miner_hotkey": "miner-1",
        "lead_ids": ["lead-1"],
        "leads": [{"lead": 1}],
    }
    monkeypatch.setattr(
        cloud_db,
        "gateway_get_fulfillment_reveals",
        lambda _wallet: {"requests": [{
            "request_id": request_id,
            "status": "scoring",
            "icp": {},
            "submissions": [submission],
        }]},
    )
    monkeypatch.setattr(
        cloud_db,
        "gateway_submit_fulfillment_scores",
        lambda *_args, **_kwargs: False,
    )
    old_epoch = 24_104
    work, results = _work_paths(
        tmp_path,
        epoch=old_epoch,
        request_id=request_id,
    )
    work.write_text(json.dumps({
        "epoch": old_epoch,
        "request_id": request_id,
        "submissions": [submission],
    }))
    results.write_text(json.dumps({
        "epoch": old_epoch,
        "request_id": request_id,
        "submission_results": [{
            "miner_hotkey": "miner-1",
            "submission_id": "submission-1",
            "lead_ids": ["lead-1"],
            "results": [FulfillmentScoreResult(
                lead_id="lead-1",
                failure_reason="test",
            ).model_dump()],
        }],
    }))
    stale = time.time() - 81 * 60
    os.utime(work, (stale, stale))

    await Validator.process_fulfillment_workflow(
        _validator_stub(last_processed_epoch=24_105),
        _epoch_state(24_105),
    )

    matching = list(
        (tmp_path / "validator_weights").glob(f"*_work_*_{request_id}.json")
    )
    assert matching == [work]
    assert results.exists()
    assert work.stat().st_mtime > stale


def test_score_formatter_rejects_cardinality_mismatch() -> None:
    with pytest.raises(ValueError, match="cardinality mismatch"):
        format_scores_for_gateway(
            "miner-1",
            ["lead-1", "lead-2"],
            [FulfillmentScoreResult()],
            request_id="request-1",
            submission_id="submission-1",
        )


def test_gateway_score_submit_raises_after_exactly_three_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = []
    sleeps = []

    def fail_post(*_args, **_kwargs):
        attempts.append(1)
        raise requests.ConnectionError("gateway unavailable")

    monkeypatch.setattr(cloud_db.requests, "post", fail_post)
    monkeypatch.setattr(cloud_db.time, "sleep", sleeps.append)
    wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="validator-hotkey",
            sign=lambda _message: b"signature",
        )
    )

    with pytest.raises(RuntimeError, match="after 3 attempts"):
        cloud_db.gateway_submit_fulfillment_scores(
            wallet,
            "43c6bc55-80f9-49e3-af0c-7a6ae6e39358",
            [{"lead_id": "lead-1"}],
        )

    assert len(attempts) == 3
    assert sleeps == [2, 2]


def test_research_lab_child_environment_requires_matching_epoch_authority(
    tmp_path: Path,
) -> None:
    manifest = (ROOT / "config" / "stateful-epoch-cutover-sn71.json").read_text(
        encoding="utf-8"
    )
    env = {
        "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON": manifest,
        "BITTENSOR_NETUID": "71",
        "SENTINEL": "preserved",
    }

    child = build_research_lab_worker_environment(env)

    assert child == env
    with pytest.raises(ResearchLabWorkerStartupError, match="missing or invalid"):
        build_research_lab_worker_environment({"BITTENSOR_NETUID": "71"})
    with pytest.raises(ResearchLabWorkerStartupError, match="different netuid"):
        build_research_lab_worker_environment(
            {
                "LEADPOET_SUBNET_EPOCH_CUTOVER_JSON": manifest,
                "BITTENSOR_NETUID": "72",
            }
        )

    path = tmp_path / "cutover.json"
    path.write_text(manifest, encoding="utf-8")
    path_child = build_research_lab_worker_environment(
        {
            "LEADPOET_SUBNET_EPOCH_CUTOVER_PATH": str(path),
            "NETUID": "71",
        }
    )
    assert path_child["LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"] == str(path)


def test_every_validator_and_research_worker_launch_has_epoch_guard() -> None:
    deploy = (
        ROOT / "validator_models" / "containerizing" / "deploy_dynamic.sh"
    ).read_text(encoding="utf-8")
    env_arg = (
        "-e LEADPOET_SUBNET_EPOCH_CUTOVER_JSON="
        '"${LEADPOET_SUBNET_EPOCH_CUTOVER_JSON:-}"'
    )

    assert deploy.count(env_arg) == 3
    assert deploy.count(
        '-e VALIDATOR_RUNTIME_GENERATION="$VALIDATOR_RUNTIME_GENERATION"'
    ) == 3
    fulfillment_section = deploy[
        deploy.index("# Auto-detect FULFILLMENT proxies"):
        deploy.index("# Wait for containers to start")
    ]
    assert 'FF_WORKER_IDS+=("$i")' in fulfillment_section
    assert 'for i in "${FF_WORKER_IDS[@]}"' in fulfillment_section
    assert "break" not in fulfillment_section
    assert "validate_worker_epoch_authority" in deploy
    assert "validate_validator_shared_epoch_file(" in deploy
    assert "from neurons.validator" not in deploy
    for container in (
        "leadpoet-validator-worker-$i",
        "leadpoet-qual-worker-$i",
        "leadpoet-ff-worker-$i",
    ):
        assert f'validate_worker_epoch_authority "{container}"' in deploy

    for script_name in (
        "run_research_lab_hosted_worker.py",
        "run_research_lab_hosted_worker_fleet.py",
        "run_research_lab_scoring_worker.py",
        "run_research_lab_scoring_worker_fleet.py",
    ):
        source = (ROOT / "scripts" / script_name).read_text(encoding="utf-8")
        assert "build_research_lab_worker_environment()" in source
        if script_name.endswith("_fleet.py"):
            assert "env = worker_environment.copy()" in source

    worker_process = (
        ROOT / "gateway" / "research_lab" / "worker_process.py"
    ).read_text(encoding="utf-8")
    assert "build_research_lab_worker_environment()" in worker_process

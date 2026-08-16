"""Joined exact N-1 to candidate rebenchmark recovery implementation."""

from __future__ import annotations

import asyncio
import base64
import copy
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

from dynamic_rebenchmark_workflow import (
    git_blob_identity,
    patched_rebenchmark_launch_environment,
    rebenchmark_launch_environment,
    transition_source_paths_by_commit,
)


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_CONFIG_FIELDS = (
    "private_baseline_concurrency",
    "private_baseline_retry_concurrency",
    "private_baseline_provider_retry_rounds",
    "scoring_worker_total_workers",
    "scoring_worker_model_timeout_seconds",
    "scoring_worker_min_available_memory_mb",
    "scoring_worker_max_load_per_cpu",
)


class _Body:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def read(self) -> bytes:
        return self._value


class _CheckpointS3:
    def __init__(self, location: tuple[str, str], document: Mapping[str, Any]) -> None:
        self.location = location
        self.objects = {
            location: json.dumps(
                dict(document), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        }
        self.put_count = 0

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        location = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        if location != self.location:
            raise RuntimeError("candidate checkpoint escaped its exact object")
        self.objects[location] = bytes(kwargs["Body"])
        self.put_count += 1
        return {"ETag": '"exact-candidate"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        location = (str(kwargs["Bucket"]), str(kwargs["Key"]))
        return {"Body": _Body(self.objects[location])}


class _Runner:
    def __init__(self, worker_index: int = 0) -> None:
        self.worker_index = worker_index

    def with_worker_index(self, worker_index: int) -> "_Runner":
        return _Runner(worker_index)


def _candidate_contract(config: Any) -> dict[str, Any]:
    return {
        **{name: getattr(config, name) for name in _CONFIG_FIELDS},
        "policy": config.conditional_validation_policy().to_dict(),
    }


def _extract_exact_commit(
    source_root: Path, commit_sha: str, destination: Path
) -> None:
    archive = subprocess.run(
        ["git", "archive", "--format=tar", commit_sha],
        cwd=source_root,
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=BytesIO(archive), mode="r:") as bundle:
        bundle.extractall(destination)


def _run_exact_n_minus_one(
    *,
    source_root: Path,
    exact_root: Path,
    launch_environment: Mapping[str, str],
    full_launch_environment: Mapping[str, str],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    child_environment = dict(os.environ)
    child_environment.update(
        {str(name): str(value) for name, value in full_launch_environment.items()}
    )
    child_environment.update(
        {str(name): str(value) for name, value in launch_environment.items()}
    )
    child_environment["LEADPOET_SUBNET_EPOCH_CUTOVER_PATH"] = str(
        exact_root / "config/stateful-epoch-cutover-sn71.json"
    )
    child_environment["PYTHONPATH"] = str(exact_root)
    harness_path = (
        source_root / "tests/restart_rehearsal/dynamic_rebenchmark_n_minus_one.py"
    )
    completed = subprocess.run(
        [sys.executable, str(harness_path), str(exact_root)],
        cwd=exact_root,
        env=child_environment,
        input=json.dumps(dict(context), sort_keys=True, separators=(",", ":")),
        text=True,
        capture_output=True,
        check=False,
        timeout=max(
            30.0,
            float(context["model_invocation_timeout_seconds"]),
        ),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "exact N-1 rebenchmark producer failed "
            f"with exit code {completed.returncode}: "
            + str(completed.stderr or completed.stdout)[-1000:]
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("exact N-1 producer returned no evidence")
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("exact N-1 producer evidence is invalid") from exc
    if not isinstance(result, Mapping):
        raise RuntimeError("exact N-1 producer evidence is not an object")
    return dict(result)


def _make_receipt_factory(
    *,
    commit_sha: str,
    issued_at: str,
    evaluation_epoch: int,
) -> tuple[Any, dict[str, dict[str, Any]]]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from leadpoet_canonical.attested_v2 import (
        EMPTY_HOST_OPERATION_ROOT,
        EMPTY_TRANSPORT_ROOT,
        build_boot_identity_body,
        build_execution_receipt_body,
        build_receipt_graph,
        create_boot_identity,
        create_signed_execution_receipt,
        merkle_root,
        sha256_json,
        validate_receipt_graph,
    )

    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = (
        signing_key.public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )
    boot = create_boot_identity(
        body=build_boot_identity_body(
            role="gateway_scoring",
            physical_role="gateway_scoring",
            commit_sha=commit_sha,
            pcr0=hashlib.sha384((commit_sha + ":pcr0").encode("ascii")).hexdigest(),
            build_manifest_hash=sha256_json({"commit": commit_sha, "kind": "build"}),
            dependency_lock_hash=sha256_json({"commit": commit_sha, "kind": "lock"}),
            config_hash=sha256_json({"commit": commit_sha, "kind": "config"}),
            boot_nonce=hashlib.sha256(
                (commit_sha + ":boot").encode("ascii")
            ).hexdigest()[:32],
            signing_pubkey=signing_pubkey,
            transport_pubkey=signing_pubkey,
            transport_certificate_hash=sha256_json(
                {"commit": commit_sha, "kind": "transport"}
            ),
            attestation_user_data_hash=sha256_json(
                {"commit": commit_sha, "kind": "user-data"}
            ),
            issued_at=issued_at,
        ),
        attestation_document_b64=base64.b64encode(
            b"strict-exact-candidate-attestation"
        ).decode("ascii"),
    )
    graphs: dict[str, dict[str, Any]] = {}

    def build(
        *, item_ref: str, item_index: int, retry_round: int, terminal: bool
    ) -> list[str]:
        purposes = ["research_lab.private_model_run.v2"]
        if terminal:
            purposes.append("research_lab.company_score.v2")
        roots: list[str] = []
        for sequence, purpose in enumerate(purposes):
            payload = {
                "icp_ref": item_ref,
                "retry_round": retry_round,
                "purpose": purpose,
            }
            result = {"terminal": terminal, "item_index": item_index}
            receipt = create_signed_execution_receipt(
                body=build_execution_receipt_body(
                    role="gateway_scoring",
                    purpose=purpose,
                    job_id=(
                        "candidate-"
                        + hashlib.sha256(
                            f"{commit_sha}:{item_index}:{retry_round}:{purpose}".encode(
                                "ascii"
                            )
                        ).hexdigest()[:24]
                    ),
                    epoch_id=evaluation_epoch,
                    sequence=sequence,
                    commit_sha=commit_sha,
                    pcr0=str(boot["pcr0"]),
                    build_manifest_hash=str(boot["build_manifest_hash"]),
                    dependency_lock_hash=str(boot["dependency_lock_hash"]),
                    config_hash=str(boot["config_hash"]),
                    boot_identity_hash=str(boot["boot_identity_hash"]),
                    input_root=sha256_json(payload),
                    output_root=sha256_json(result),
                    transport_root_hash=EMPTY_TRANSPORT_ROOT,
                    host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                    artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
                    parent_receipt_hashes=(),
                    status="succeeded",
                    failure_code=None,
                    issued_at=issued_at,
                ),
                enclave_pubkey=signing_pubkey,
                sign_digest=signing_key.sign,
            )
            graph = build_receipt_graph(
                root_receipt_hash=str(receipt["receipt_hash"]),
                boot_identities=(boot,),
                receipts=(receipt,),
                transport_attempts=(),
                host_operations=(),
            )
            validate_receipt_graph(graph)
            root = str(graph["root_receipt_hash"])
            graphs[root] = graph
            roots.append(root)
        return roots

    return build, graphs


def exercise_transition(
    *,
    source_root: Path,
    from_sha: str,
    candidate_sha: str,
    publication_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_root = source_root.resolve()
    normalized_from = str(from_sha or "").strip().lower()
    normalized_candidate = str(candidate_sha or "").strip().lower()
    if (
        not _SHA_RE.fullmatch(normalized_from)
        or not _SHA_RE.fullmatch(normalized_candidate)
        or normalized_from == normalized_candidate
    ):
        raise RuntimeError("dynamic rebenchmark requires distinct exact releases")

    import boto3

    from contract_adapter import _gateway_secret
    from gateway.research_lab import scoring_worker as scoring_worker_module
    from gateway.research_lab.config import ResearchLabGatewayConfig
    from gateway.research_lab.scoring_worker import (
        BaselineCheckpointRecycle,
        ResearchLabGatewayScoringWorker,
        _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD,
        _baseline_attempt_ledger_entry,
        _baseline_scoring_contract_hash,
        _baseline_summary_checkpointable,
        _baseline_wave_stall_timeout_seconds,
        _load_baseline_scoring_progress,
        _model_invocation_timeout_seconds,
        _store_baseline_scoring_progress,
        _worker_recycle_rss_mb,
    )
    from gateway.research_lab.provider_preflight import provider_preflight_settings
    from gateway.research_lab.store import canonical_hash
    from gateway.tee.scoring_executor import (
        configuration_hash,
        runtime_environment_values,
    )
    from gateway.tee.topology import validate_manifest, validate_production_capacity
    from leadpoet_canonical.attested_v2 import sha256_json

    with patch.dict(
        os.environ,
        {
            "REHEARSAL_FROM_SHA": normalized_from,
            "REHEARSAL_CANDIDATE_SHA": normalized_candidate,
        },
    ):
        launch_environment = rebenchmark_launch_environment()
        full_launch_environment = {
            str(name): str(value) for name, value in _gateway_secret().items()
        }
    with patched_rebenchmark_launch_environment(launch_environment):
        candidate_config = ResearchLabGatewayConfig.from_env()
        candidate_contract = _candidate_contract(candidate_config)
        scoring_runtime_environment = runtime_environment_values()
        scoring_configuration_hash = configuration_hash(scoring_runtime_environment)
        scoring_contract_hash = _baseline_scoring_contract_hash()
        preflight_ttl_seconds = float(provider_preflight_settings()["ttl_seconds"])
        recycle_rss_mb = int(_worker_recycle_rss_mb())
    policy = candidate_config.conditional_validation_policy()
    total_icps = int(policy.total_icps)
    retry_width = max(1, int(candidate_config.private_baseline_retry_concurrency))
    retry_rounds = int(candidate_config.private_baseline_provider_retry_rounds)
    completed_receipts_per_attempt = int(
        scoring_worker_module._V2_BASELINE_RECEIPTS_PER_COMPLETED_ICP
    )
    retryable_receipts_per_attempt = completed_receipts_per_attempt - 1
    if retry_rounds < 1 or total_icps <= retry_width:
        raise RuntimeError("launch contract cannot exercise partial retry recovery")
    if not 0 < retryable_receipts_per_attempt < completed_receipts_per_attempt:
        raise RuntimeError("candidate receipt frontier contract is invalid")
    first_pass_success_count = min(
        total_icps - 1,
        max(1, int(policy.public_strong_total)),
    )
    retryable_count = total_icps - first_pass_success_count
    if retryable_count <= retry_width:
        raise RuntimeError(
            "candidate-derived incident skew cannot exercise retry waves"
        )
    retryable_indexes = set(range(first_pass_success_count + 1, total_icps + 1))

    if publication_context is None:
        items = [
            {
                "icp_ref": f"dynamic-rebenchmark-icp-{index}",
                "icp_hash": sha256_json({"dynamic_icp": index}),
            }
            for index in range(1, total_icps + 1)
        ]
        benchmark_date = "2026-07-25"
        window_hash = sha256_json({"items": items})
        model_artifact_hash = sha256_json({"model": normalized_candidate})
        manifest_hash = sha256_json({"artifact": model_artifact_hash})
        repo_git_sha = normalized_candidate
        evaluation_epoch = int(os.environ.get("REHEARSAL_EVALUATION_EPOCH", "24600"))
        issued_at = benchmark_date + "T00:30:00Z"
        checkpoint_bucket = "dynamic-rebenchmark-checkpoints"
        checkpoint_key = "exact-transition/progress.json"
    else:
        items = [
            copy.deepcopy(dict(item)) for item in publication_context["benchmark_items"]
        ]
        benchmark_date = str(publication_context["benchmark_date"])
        window_hash = str(publication_context["window_hash"])
        model_artifact_hash = str(publication_context["model_artifact_hash"])
        manifest_hash = str(publication_context["manifest_hash"])
        repo_git_sha = str(publication_context["repo_git_sha"])
        evaluation_epoch = int(publication_context["evaluation_epoch"])
        issued_at = str(publication_context["issued_at"])
        checkpoint_bucket = str(publication_context["checkpoint_bucket"])
        checkpoint_key = str(publication_context["checkpoint_key"])
    if len(items) != total_icps:
        raise RuntimeError("publication and launch ICP banks differ")
    provider_cost_scope_hash = sha256_json(
        {
            "benchmark_date": benchmark_date,
            "window_hash": window_hash,
            "model_artifact_hash": model_artifact_hash,
            "scoring_configuration_hash": scoring_configuration_hash,
        }
    )
    model_invocation_timeout_seconds = _model_invocation_timeout_seconds(
        float(candidate_config.scoring_worker_model_timeout_seconds)
    )
    watchdog_timeout_seconds = _baseline_wave_stall_timeout_seconds(candidate_config)
    if watchdog_timeout_seconds <= model_invocation_timeout_seconds:
        raise RuntimeError("watchdog bound does not contain model invocation")

    n_minus_context = {
        "from_sha": normalized_from,
        "benchmark_items": items,
        "first_pass_success_count": first_pass_success_count,
        "benchmark_date": benchmark_date,
        "window_hash": window_hash,
        "model_artifact_hash": model_artifact_hash,
        "manifest_hash": manifest_hash,
        "repo_git_sha": repo_git_sha,
        "evaluation_epoch": evaluation_epoch,
        "issued_at": issued_at,
        "checkpoint_bucket": checkpoint_bucket,
        "checkpoint_key": checkpoint_key,
        "provider_cost_scope_hash": provider_cost_scope_hash,
        "scoring_configuration_hash": scoring_configuration_hash,
        "scoring_runtime_environment": scoring_runtime_environment,
        "scoring_contract_hash": scoring_contract_hash,
        "provider_preflight_ttl_seconds": preflight_ttl_seconds,
        "model_invocation_timeout_seconds": model_invocation_timeout_seconds,
        "watchdog_timeout_seconds": watchdog_timeout_seconds,
    }
    with tempfile.TemporaryDirectory(prefix="exact-rebenchmark-n-minus-one-") as raw:
        exact_root = Path(raw)
        _extract_exact_commit(normalized_root, normalized_from, exact_root)
        n_minus = _run_exact_n_minus_one(
            source_root=normalized_root,
            exact_root=exact_root,
            launch_environment=launch_environment,
            full_launch_environment=full_launch_environment,
            context=n_minus_context,
        )
        from dynamic_docker_collision_workflow import (
            exercise_dynamic_docker_collision,
        )

        docker_collision = exercise_dynamic_docker_collision(
            source_root=normalized_root,
            exact_root=exact_root,
            from_sha=normalized_from,
            candidate_sha=normalized_candidate,
            launch_environment=launch_environment,
            scoring_worker_count=int(candidate_config.scoring_worker_total_workers),
            scoring_memory_floor_mib=int(
                candidate_config.scoring_worker_min_available_memory_mb
            ),
            model_timeout_seconds=max(1, int(model_invocation_timeout_seconds)),
        )
    n_minus_config = dict(n_minus["config"])
    if (
        any(
            n_minus_config.get(field) != candidate_contract[field]
            for field in _CONFIG_FIELDS
        )
        or n_minus_config.get("policy") != candidate_contract["policy"]
        or n_minus_config.get("scoring_configuration_hash")
        != scoring_configuration_hash
        or float(n_minus_config["provider_preflight_ttl_seconds"])
        != preflight_ttl_seconds
        or float(n_minus_config["model_invocation_timeout_seconds"])
        != model_invocation_timeout_seconds
        or float(n_minus_config["watchdog_timeout_seconds"]) != watchdog_timeout_seconds
        or int(n_minus_config["worker_recycle_rss_mb"]) != recycle_rss_mb
    ):
        raise RuntimeError("N-1/candidate scoring contract changed")
    n_minus_calls = [
        tuple(int(value) for value in row) for row in n_minus["attempt_calls"]
    ]
    n_minus_started_calls = [
        tuple(int(value) for value in row)
        for row in n_minus["started_attempt_calls"]
    ]
    n_minus_completed_calls = [
        tuple(int(value) for value in row)
        for row in n_minus["completed_attempt_calls"]
    ]
    checkpointed_peer_call = tuple(
        int(value) for value in n_minus["checkpointed_peer_call"]
    )
    interrupted_peer_call = tuple(
        int(value) for value in n_minus["interrupted_peer_call"]
    )
    expected_retry_wave_calls = {
        (item_index, 1)
        for item_index in sorted(retryable_indexes)[:retry_width]
    }
    uncheckpointed_n_minus_calls = set(n_minus_started_calls) - set(n_minus_calls)
    expected_n_minus_calls = total_icps + 1
    expected_n_minus_started_calls = total_icps + retry_width
    expected_n_minus_completed = first_pass_success_count + (
        1 if retry_rounds == 1 else 0
    )
    expected_n_minus_receipt_frontier = (
        first_pass_success_count * completed_receipts_per_attempt
        + retryable_count * retryable_receipts_per_attempt
        + 1
        * (
            completed_receipts_per_attempt
            if retry_rounds == 1
            else retryable_receipts_per_attempt
        )
    )
    if (
        len(n_minus_calls) != expected_n_minus_calls
        or len(set(n_minus_calls)) != len(n_minus_calls)
        or len(n_minus_started_calls) != expected_n_minus_started_calls
        or len(set(n_minus_started_calls)) != len(n_minus_started_calls)
        or len(n_minus_completed_calls) != total_icps + 2
        or set(n_minus_started_calls[-retry_width:]) != expected_retry_wave_calls
        or set(n_minus_completed_calls) - set(n_minus_calls)
        != {interrupted_peer_call}
        or uncheckpointed_n_minus_calls
        != expected_retry_wave_calls - {checkpointed_peer_call}
        or checkpointed_peer_call not in n_minus_calls
        or interrupted_peer_call in n_minus_calls
        or interrupted_peer_call not in n_minus_completed_calls
        or n_minus["peer_interruption"].get("reason")
        != "peer_interrupted_after_checkpoint"
        or int(n_minus["checkpoint_attempt_count"]) != expected_n_minus_calls
        or int(n_minus["checkpoint_completed_icp_count"]) != expected_n_minus_completed
        or int(n_minus["checkpoint_receipt_frontier_count"])
        != expected_n_minus_receipt_frontier
        or n_minus["exact_per_icp_scoring_path_executed"] is not True
        or n_minus["exact_scoring_worker_module"]
        != "gateway/research_lab/scoring_worker.py"
        or n_minus["exact_fleet_launcher_module"]
        != "scripts/run_research_lab_scoring_worker_fleet.py"
        or n_minus["exact_envelope_preparer_module"]
        != "gateway/tee/prepare_gateway_envelopes_v2.py"
        or n_minus["exact_scoring_worker_entrypoint"]
        != "scripts/run_research_lab_scoring_worker.py"
        or n_minus["exact_supervisor_module"]
        != "gateway/research_lab/worker_autostart.py"
        or docker_collision.get("dynamic_docker_collision_exact") is not True
        or docker_collision.get(
            "candidate_gateway_emergency_uses_guarded_reclaim"
        )
        is not True
        or docker_collision.get(
            "first_activation_requires_preexisting_disk_reserve"
        )
        is not True
    ):
        raise RuntimeError("N-1 did not execute exact production paths")

    checkpoint_location = (checkpoint_bucket, checkpoint_key)
    checkpoint_s3 = _CheckpointS3(
        checkpoint_location,
        n_minus["checkpoint_document"],
    )
    restored_attempts: list[dict[str, Any]] = []
    restored_scope: list[str] = []
    restored_producers: set[str] = set()
    restored_roots: set[str] = set()
    with patch.object(boto3, "client", return_value=checkpoint_s3):
        restored_rows = _load_baseline_scoring_progress(
            checkpoint_bucket,
            checkpoint_key,
            benchmark_date=benchmark_date,
            window_hash=window_hash,
            private_model_artifact_hash=model_artifact_hash,
            gateway_runtime_commit_sha=normalized_candidate,
            scoring_configuration_hash_value=scoring_configuration_hash,
            repo_git_sha=repo_git_sha,
            manifest_hash=manifest_hash,
            parent_receipt_hashes_out=restored_roots,
            attempt_ledger_out=restored_attempts,
            producer_runtime_commits_out=restored_producers,
            provider_cost_base_scope_out=restored_scope,
            scoring_contract_hash_value=scoring_contract_hash,
            benchmark_items=items,
        )
    if (
        restored_scope != [provider_cost_scope_hash]
        or restored_producers != {normalized_from}
        or len(restored_attempts) != expected_n_minus_calls
    ):
        raise RuntimeError("candidate rejected exact N-1 checkpoint lineage")

    clock = {"value": preflight_ttl_seconds / 2.0}
    candidate_receipt, candidate_graphs = _make_receipt_factory(
        commit_sha=normalized_candidate,
        issued_at=issued_at,
        evaluation_epoch=evaluation_epoch,
    )
    active_by_round: dict[int, int] = {}
    maximum_active_by_round: dict[int, int] = {}
    candidate_calls: list[tuple[int, int]] = []

    class CandidateWorker(ResearchLabGatewayScoringWorker):
        async def _run_baseline_icp(
            self, *, item_index: int, retry_round: int, **_kwargs: Any
        ) -> dict[str, Any]:
            active_by_round[retry_round] = active_by_round.get(retry_round, 0) + 1
            maximum_active_by_round[retry_round] = max(
                maximum_active_by_round.get(retry_round, 0),
                active_by_round[retry_round],
            )
            try:
                await asyncio.sleep(0)
                retryable = (
                    item_index in retryable_indexes and retry_round < retry_rounds
                )
                item = items[item_index - 1]
                roots = candidate_receipt(
                    item_ref=str(item["icp_ref"]),
                    item_index=item_index,
                    retry_round=retry_round,
                    terminal=not retryable,
                )
                candidate_calls.append((item_index, retry_round))
                return {
                    "icp_ref": str(item["icp_ref"]),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "score": 0.0 if retryable else float(item_index),
                    "company_count": 0 if retryable else 1,
                    "sourced_count": 0 if retryable else 1,
                    "diagnostics": (
                        {
                            "sourcing_failed": True,
                            "runtime_error": {"category": "provider"},
                        }
                        if retryable
                        else {}
                    ),
                    "_item_index": item_index,
                    "_retryable": retryable,
                    "_nonempty": not retryable,
                    "_runtime_error": "provider unavailable" if retryable else "",
                    "_retry_backoff_seconds": 0.0,
                    "_retry_round": retry_round,
                    _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: roots,
                }
            finally:
                active_by_round[retry_round] -= 1

    candidate_worker = object.__new__(CandidateWorker)
    candidate_worker.worker_ref = "exact-candidate-rebenchmark"
    candidate_worker.config = candidate_config
    candidate_worker._lease_holder_ref = "exact-candidate-preflight-owner"
    candidate_worker._baseline_profile_preflight_monotonic_at = {}

    # Actual owned aggregation/control is exercised for uncached, cached,
    # unhealthy, partial, and lease-loss cases. Only the provider gate and
    # durable maintenance-control boundaries are strict adapters.
    preflight_calls: list[dict[str, Any]] = []
    control_events: list[dict[str, Any]] = []
    control_state: dict[str, Any] = {"paused": False, "reason": ""}
    preflight_mode = {"name": "healthy_uncached"}

    async def strict_preflight_gate(**kwargs: Any) -> dict[str, Any]:
        worker_index = int(kwargs["worker_index"])
        mode = str(preflight_mode["name"])
        preflight_calls.append(
            {
                "worker_index": worker_index,
                "force_measurement": bool(kwargs.get("force_measurement")),
                "mode": mode,
            }
        )
        if mode == "partial" and worker_index == 0:
            raise RuntimeError("strict profile measurement failed")
        unhealthy = mode == "unhealthy"
        return {
            "proceed": not unhealthy,
            "healthy": not unhealthy,
            "pause_worthy": unhealthy,
            "disabled": False,
            "measurement_cached": mode == "healthy_cached",
            "verdicts": [
                {
                    "provider": f"profile_{worker_index}",
                    "healthy": not unhealthy,
                    "status": "healthy" if not unhealthy else "credit_or_auth",
                    "http_status": 200 if not unhealthy else 402,
                }
            ],
        }

    async def read_control() -> dict[str, Any]:
        return dict(control_state)

    async def write_control(**kwargs: Any) -> dict[str, Any]:
        control_events.append(dict(kwargs))
        control_state.update(
            {"paused": bool(kwargs["paused"]), "reason": str(kwargs["reason"])}
        )
        return dict(control_state)

    class HeldHeartbeat:
        def __init__(self, fail_on_check: int | None = None) -> None:
            self.check_count = 0
            self.fail_on_check = fail_on_check

        def ensure_held(self) -> None:
            self.check_count += 1
            if self.fail_on_check == self.check_count:
                raise RuntimeError("strict maintenance lease lost")

    async def run_owned(mode: str, heartbeat: HeldHeartbeat) -> Mapping[str, Any]:
        preflight_mode["name"] = mode
        return await candidate_worker._run_owned_provider_preflight(
            maintenance_state=dict(control_state),
            heartbeat=heartbeat,
            force_measurement=True,
        )

    with (
        patch.object(scoring_worker_module, "preflight_gate", strict_preflight_gate),
        patch.object(
            scoring_worker_module, "get_scoring_maintenance_state", read_control
        ),
        patch.object(
            scoring_worker_module, "set_scoring_maintenance_paused", write_control
        ),
        patch.object(
            scoring_worker_module,
            "_baseline_preflight_monotonic",
            lambda: clock["value"],
        ),
    ):
        healthy_uncached = asyncio.run(run_owned("healthy_uncached", HeldHeartbeat()))
        uncached_freshness = dict(
            candidate_worker._baseline_profile_preflight_monotonic_at
        )
        healthy_cached = asyncio.run(run_owned("healthy_cached", HeldHeartbeat()))
        cached_freshness = dict(
            candidate_worker._baseline_profile_preflight_monotonic_at
        )
        unhealthy = asyncio.run(run_owned("unhealthy", HeldHeartbeat()))
        unhealthy_pause_seen = bool(control_state["paused"])
        control_state.update({"paused": False, "reason": ""})
        partial_start = len(preflight_calls)
        partial = asyncio.run(run_owned("partial", HeldHeartbeat()))
        partial_measurements = sum(
            1 for call in preflight_calls[partial_start:] if call["worker_index"] != 0
        )
        control_state.update({"paused": False, "reason": ""})
        try:
            asyncio.run(run_owned("healthy_uncached", HeldHeartbeat(fail_on_check=2)))
        except RuntimeError as exc:
            lease_loss_seen = "lease lost" in str(exc)
        else:
            lease_loss_seen = False
    worker_count = max(1, int(candidate_config.scoring_worker_total_workers))
    expected_profiles = set(range(worker_count))
    if (
        healthy_uncached.get("proceed") is not True
        or set(uncached_freshness) != expected_profiles
        or healthy_cached.get("proceed") is not True
        or cached_freshness
        or unhealthy.get("proceed") is not False
        or not unhealthy_pause_seen
        or partial.get("proceed") is not False
        or partial_measurements >= worker_count
        or not lease_loss_seen
    ):
        raise RuntimeError("actual owned provider preflight matrix differs")

    # Fresh admission succeeds, stale admission recycles, and only the actual
    # maintenance-lease path below can refresh the full fleet afterward.
    candidate_worker._baseline_profile_preflight_monotonic_at = uncached_freshness
    with (
        patch.object(
            scoring_worker_module,
            "provider_preflight_settings",
            lambda: {"ttl_seconds": preflight_ttl_seconds},
        ),
        patch.object(
            scoring_worker_module,
            "_baseline_preflight_monotonic",
            lambda: clock["value"],
        ),
    ):
        asyncio.run(
            candidate_worker._enforce_baseline_wave_preflight_freshness(
                run_start=time.time(),
                item_indexes=tuple(sorted(retryable_indexes)[:retry_width]),
                retry_round=1,
                completed_icps=len(restored_rows),
                total_icps=total_icps,
            )
        )
        clock["value"] = (
            max(uncached_freshness.values())
            + preflight_ttl_seconds
            + max(1.0, preflight_ttl_seconds / max(1, total_icps))
        )
        try:
            asyncio.run(
                candidate_worker._enforce_baseline_wave_preflight_freshness(
                    run_start=time.time(),
                    item_indexes=tuple(sorted(retryable_indexes)[:retry_width]),
                    retry_round=1,
                    completed_icps=len(restored_rows),
                    total_icps=total_icps,
                )
            )
        except BaselineCheckpointRecycle as exc:
            stale_recycle = dict(exc.pressure)
        else:
            raise RuntimeError("stale candidate proof admitted another attempt")

    lease_available = {"value": False}
    lease_force_calls: list[bool] = []

    async def acquire_lease(**_kwargs: Any) -> bool:
        return bool(lease_available["value"])

    class LeaseHeartbeat:
        def __init__(self, **_kwargs: Any) -> None:
            self.held = False

        async def start(self) -> None:
            self.held = True

        def ensure_held(self) -> None:
            if not self.held:
                raise RuntimeError("candidate maintenance lease lost")

        async def stop(self) -> None:
            self.held = False

    async def maintenance_noop() -> None:
        return None

    async def baseline_not_ready(**_kwargs: Any) -> dict[str, Any]:
        return {"available": False}

    async def lease_preflight_gate(**kwargs: Any) -> dict[str, Any]:
        lease_force_calls.append(bool(kwargs.get("force_measurement")))
        return {
            "proceed": True,
            "healthy": True,
            "pause_worthy": False,
            "disabled": False,
            "measurement_cached": False,
            "verdicts": [
                {
                    "provider": f"profile_{int(kwargs['worker_index'])}",
                    "healthy": True,
                    "status": "healthy",
                    "http_status": 200,
                }
            ],
        }

    candidate_worker._recover_stale_candidate_claims = maintenance_noop
    candidate_worker._alert_stuck_candidates = maintenance_noop
    candidate_worker._requeue_quarantined_candidates = maintenance_noop
    candidate_worker._candidate_scoring_start_gate = baseline_not_ready
    candidate_worker._baseline_profile_preflight_monotonic_at = {}
    control_state.update({"paused": False, "reason": ""})
    with (
        patch.object(
            scoring_worker_module, "try_acquire_maintenance_lease", acquire_lease
        ),
        patch.object(
            scoring_worker_module, "MaintenanceLeaseHeartbeat", LeaseHeartbeat
        ),
        patch.object(scoring_worker_module, "preflight_gate", lease_preflight_gate),
        patch.object(
            scoring_worker_module, "get_scoring_maintenance_state", read_control
        ),
        patch.object(
            scoring_worker_module, "set_scoring_maintenance_paused", write_control
        ),
        patch.object(
            scoring_worker_module,
            "_baseline_preflight_monotonic",
            lambda: clock["value"],
        ),
    ):
        pending_owner = asyncio.run(
            candidate_worker._run_lease_held_recovery_and_preflight({"paused": False})
        )
        pre_lease_measurement_count = len(lease_force_calls)
        lease_available["value"] = True
        refreshed = asyncio.run(
            candidate_worker._run_lease_held_recovery_and_preflight({"paused": False})
        )
    lease_owner_measurement_count = len(lease_force_calls) - pre_lease_measurement_count
    if (
        pending_owner.get("reason") != "provider_preflight_full_fleet_owner_pending"
        or refreshed.get("proceed") is not True
        or pre_lease_measurement_count != 0
        or lease_owner_measurement_count != worker_count
        or len(lease_force_calls) != worker_count
        or not all(lease_force_calls)
        or set(candidate_worker._baseline_profile_preflight_monotonic_at)
        != expected_profiles
    ):
        raise RuntimeError("stale proof bypassed lease-owned forced measurement")

    attempt_ledger = list(restored_attempts)
    terminal_rows = {str(row["icp_ref"]): dict(row) for row in restored_rows}
    parent_roots = set(restored_roots)

    async def checkpoint_attempt(row: Mapping[str, Any], *, retry_round: int) -> bool:
        entry = _baseline_attempt_ledger_entry(
            row,
            retry_round=retry_round,
            gateway_runtime_commit_sha=normalized_candidate,
        )
        key = (str(entry["icp_ref"]), int(entry["retry_round"]))
        if key in {
            (str(value["icp_ref"]), int(value["retry_round"]))
            for value in attempt_ledger
        }:
            raise RuntimeError("candidate repeated a settled N-1 attempt")
        attempt_ledger.append(entry)
        parent_roots.update(
            str(value) for value in row.get(_BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD, ())
        )
        if _baseline_summary_checkpointable(row):
            terminal_rows[str(row["icp_ref"])] = dict(row)
        with patch.object(boto3, "client", return_value=checkpoint_s3):
            _store_baseline_scoring_progress(
                checkpoint_bucket,
                checkpoint_key,
                benchmark_date=benchmark_date,
                window_hash=window_hash,
                private_model_artifact_hash=model_artifact_hash,
                gateway_runtime_commit_sha=normalized_candidate,
                scoring_configuration_hash_value=scoring_configuration_hash,
                rows=list(terminal_rows.values()),
                attested_parent_receipt_hashes=sorted(parent_roots),
                repo_git_sha=repo_git_sha,
                manifest_hash=manifest_hash,
                attempt_ledger=attempt_ledger,
                provider_cost_base_scope_hash=provider_cost_scope_hash,
                scoring_contract_hash_value=scoring_contract_hash,
            )
        return True

    async def unpaused() -> dict[str, Any]:
        return {"paused": False}

    with (
        patch.object(scoring_worker_module, "get_scoring_maintenance_state", unpaused),
        patch.object(
            scoring_worker_module,
            "provider_preflight_settings",
            lambda: {"ttl_seconds": preflight_ttl_seconds},
        ),
        patch.object(
            scoring_worker_module,
            "_baseline_preflight_monotonic",
            lambda: clock["value"],
        ),
        patch.object(
            scoring_worker_module,
            "_retry_runner_with_provider_cost_scope",
            lambda runner, **_kwargs: runner,
        ),
    ):
        final_rows, retry_stats = asyncio.run(
            candidate_worker._run_baseline_batch_inner(
                runner=_Runner(),
                retry_runner=_Runner(),
                scorer=object(),
                window=SimpleNamespace(benchmark_items=items),
                run_start=time.time(),
                resume_results=restored_rows,
                resume_attempt_ledger=restored_attempts,
                attempt_checkpoint=checkpoint_attempt,
                provider_cost_base_scope=provider_cost_scope_hash,
                provider_preflight_boundary=(
                    candidate_worker._enforce_baseline_wave_preflight_freshness
                ),
            )
        )
    all_calls = [*n_minus_calls, *candidate_calls]
    candidate_replayed_uncheckpointed = set(candidate_calls) & set(
        n_minus_started_calls
    )
    expected_attempt_count = total_icps + retryable_count * retry_rounds
    retry_attempt_counts_by_round = [
        sum(
            1 for _item_index, observed_round in all_calls if observed_round == round_no
        )
        for round_no in range(1, retry_rounds + 1)
    ]
    retry_wave_counts_by_round = [
        math.ceil(count / retry_width) for count in retry_attempt_counts_by_round
    ]
    if (
        len(all_calls) != expected_attempt_count
        or len(set(all_calls)) != len(all_calls)
        or candidate_replayed_uncheckpointed != uncheckpointed_n_minus_calls
        or len(final_rows) != total_icps
        or retry_stats
        != {
            "retried": retryable_count * retry_rounds,
            "recovered": retryable_count,
            "unresolved": 0,
        }
        or set(round_no for _index, round_no in all_calls)
        != set(range(retry_rounds + 1))
        or any(count != retryable_count for count in retry_attempt_counts_by_round)
        or any(wave_count <= 1 for wave_count in retry_wave_counts_by_round)
        or any(
            maximum_active_by_round.get(round_no, 0) > retry_width
            for round_no in range(1, retry_rounds + 1)
        )
    ):
        raise RuntimeError("candidate retry continuation duplicated or skipped work")

    # Every configured retry round also reaches exhaustion when the provider
    # never recovers. This is independent of the successful transition above.
    exhausted_calls: list[tuple[int, int]] = []
    exhaustion_items = items[: min(total_icps, retry_width)]

    class ExhaustionWorker(ResearchLabGatewayScoringWorker):
        async def _run_baseline_icp(
            self, *, item_index: int, retry_round: int, **_kwargs: Any
        ) -> dict[str, Any]:
            exhausted_calls.append((item_index, retry_round))
            item = exhaustion_items[item_index - 1]
            return {
                "icp_ref": str(item["icp_ref"]),
                "icp_hash": str(item.get("icp_hash") or ""),
                "score": 0.0,
                "company_count": 0,
                "sourced_count": 0,
                "diagnostics": {
                    "sourcing_failed": True,
                    "runtime_error": {"category": "provider"},
                },
                "_item_index": item_index,
                "_retryable": True,
                "_nonempty": False,
                "_runtime_error": "provider unavailable",
                "_retry_backoff_seconds": 0.0,
                "_retry_round": retry_round,
                _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [],
            }

    exhaustion_worker = object.__new__(ExhaustionWorker)
    exhaustion_worker.worker_ref = "candidate-retry-exhaustion"
    exhaustion_worker.config = candidate_config
    with (
        patch.object(scoring_worker_module, "get_scoring_maintenance_state", unpaused),
        patch.object(
            scoring_worker_module,
            "_retry_runner_with_provider_cost_scope",
            lambda runner, **_kwargs: runner,
        ),
    ):
        _exhausted_rows, exhausted_stats = asyncio.run(
            exhaustion_worker._run_baseline_batch_inner(
                runner=_Runner(),
                retry_runner=_Runner(),
                scorer=object(),
                window=SimpleNamespace(benchmark_items=exhaustion_items),
                run_start=time.time(),
            )
        )
    if (
        set(round_no for _index, round_no in exhausted_calls)
        != set(range(retry_rounds + 1))
        or exhausted_stats["unresolved"] != len(exhaustion_items)
        or exhausted_stats["retried"] != len(exhaustion_items) * retry_rounds
    ):
        raise RuntimeError("configured retry exhaustion coverage differs")

    topology = validate_manifest(
        json.loads(
            (normalized_root / "gateway/tee/topology.json").read_text(encoding="utf-8")
        )
    )
    parent_vcpus = int(topology["production_parent_vcpus"])
    parent_memory_mib = int(topology["production_parent_memory_mib"])
    reserved_vcpus = int(topology["host_reserved_vcpus"])
    reserved_memory_mib = int(topology["host_reserved_memory_mib"])
    scoring_memory_mib = int(topology["roles"]["gateway_scoring"]["memory_mib"])
    capacity = validate_production_capacity(
        parent_vcpus=parent_vcpus,
        parent_memory_mib=parent_memory_mib,
    )
    memory_floor_mib = int(candidate_config.scoring_worker_min_available_memory_mb)
    if (
        capacity["host_vcpus"] != reserved_vcpus
        or capacity["host_memory_mib"] != reserved_memory_mib
        or not 0 < recycle_rss_mb <= scoring_memory_mib
        or not 0 < memory_floor_mib <= scoring_memory_mib
    ):
        raise RuntimeError("launch pressure contract exceeds topology")
    pressure_count = retryable_count
    pressure_items = items[first_pass_success_count:]
    if len(pressure_items) != pressure_count or pressure_count <= retry_width:
        raise RuntimeError("pressure boundary did not receive the unresolved frontier")

    def pressure_run(
        *, rss_mb: int, available_memory_mb: int, expect_recycle: bool
    ) -> dict[str, Any]:
        calls: list[int] = []
        checkpoints: list[int] = []

        class PressureWorker(ResearchLabGatewayScoringWorker):
            async def _run_baseline_icp(
                self, *, item_index: int, **_kwargs: Any
            ) -> dict[str, Any]:
                calls.append(item_index)
                item = pressure_items[item_index - 1]
                return {
                    "icp_ref": str(item["icp_ref"]),
                    "icp_hash": str(item.get("icp_hash") or ""),
                    "score": float(item_index),
                    "company_count": 1,
                    "sourced_count": 1,
                    "diagnostics": {},
                    "_item_index": item_index,
                    "_retryable": False,
                    "_nonempty": True,
                    "_runtime_error": "",
                    "_retry_backoff_seconds": 0.0,
                    "_retry_round": 0,
                    _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD: [],
                }

        async def checkpoint(row: Mapping[str, Any], *, retry_round: int) -> bool:
            if retry_round != 0:
                raise RuntimeError("pressure probe entered a retry round")
            checkpoints.append(int(row["_item_index"]))
            return True

        worker = object.__new__(PressureWorker)
        worker.worker_ref = "candidate-pressure-boundary"
        worker.config = candidate_config
        with (
            patch.dict(os.environ, full_launch_environment),
            patched_rebenchmark_launch_environment(launch_environment),
            patch.object(
                scoring_worker_module, "get_scoring_maintenance_state", unpaused
            ),
            patch.object(
                scoring_worker_module, "_read_own_rss_mb", return_value=rss_mb
            ),
            patch.object(
                scoring_worker_module,
                "_read_mem_available_mb",
                return_value=available_memory_mb,
            ),
        ):
            try:
                rows, _stats = asyncio.run(
                    worker._run_baseline_batch_inner(
                        runner=_Runner(),
                        retry_runner=_Runner(),
                        scorer=object(),
                        window=SimpleNamespace(benchmark_items=pressure_items),
                        run_start=time.time(),
                        attempt_checkpoint=checkpoint,
                        checkpoint_recycle_enabled=True,
                    )
                )
            except BaselineCheckpointRecycle as exc:
                if not expect_recycle:
                    raise RuntimeError("below-threshold pressure recycled") from exc
                return {
                    "recycled": True,
                    "pressure": dict(exc.pressure),
                    "calls": calls,
                    "checkpoints": checkpoints,
                }
        if expect_recycle:
            raise RuntimeError("threshold pressure did not recycle")
        if len(rows) != len(pressure_items) or calls != checkpoints:
            raise RuntimeError("below-threshold pressure lost checkpoint work")
        return {
            "recycled": False,
            "pressure": {},
            "calls": calls,
            "checkpoints": checkpoints,
        }

    rss_positive = pressure_run(
        rss_mb=recycle_rss_mb,
        available_memory_mb=max(memory_floor_mib, reserved_memory_mib),
        expect_recycle=True,
    )
    host_positive = pressure_run(
        rss_mb=max(0, recycle_rss_mb - 1),
        available_memory_mb=memory_floor_mib - 1,
        expect_recycle=True,
    )
    rss_negative = pressure_run(
        rss_mb=max(0, recycle_rss_mb - 1),
        available_memory_mb=max(memory_floor_mib, reserved_memory_mib),
        expect_recycle=False,
    )
    floor_negative = pressure_run(
        rss_mb=max(0, recycle_rss_mb - 1),
        available_memory_mb=memory_floor_mib,
        expect_recycle=False,
    )
    expected_pressure_settle_count = min(
        pressure_count,
        int(candidate_config.private_baseline_concurrency),
    )
    if (
        rss_positive["pressure"].get("reason") != "worker_rss_limit"
        or host_positive["pressure"].get("reason") != "host_memory_pressure"
        or len(rss_positive["checkpoints"]) != expected_pressure_settle_count
        or rss_positive["calls"] != rss_positive["checkpoints"]
        or len(host_positive["checkpoints"]) != expected_pressure_settle_count
        or host_positive["calls"] != host_positive["checkpoints"]
        or rss_negative["recycled"]
        or floor_negative["recycled"]
        or len(rss_negative["calls"]) != pressure_count
        or rss_negative["calls"] != rss_negative["checkpoints"]
        or len(floor_negative["calls"]) != pressure_count
        or floor_negative["calls"] != floor_negative["checkpoints"]
    ):
        raise RuntimeError("derived RSS/host pressure boundaries differ")

    final_attempts: list[dict[str, Any]] = []
    final_scope: list[str] = []
    final_producers: set[str] = set()
    final_roots: set[str] = set()
    with patch.object(boto3, "client", return_value=checkpoint_s3):
        checkpoint_rows = _load_baseline_scoring_progress(
            checkpoint_bucket,
            checkpoint_key,
            benchmark_date=benchmark_date,
            window_hash=window_hash,
            private_model_artifact_hash=model_artifact_hash,
            gateway_runtime_commit_sha=normalized_candidate,
            scoring_configuration_hash_value=scoring_configuration_hash,
            repo_git_sha=repo_git_sha,
            manifest_hash=manifest_hash,
            parent_receipt_hashes_out=final_roots,
            attempt_ledger_out=final_attempts,
            producer_runtime_commits_out=final_producers,
            provider_cost_base_scope_out=final_scope,
            scoring_contract_hash_value=scoring_contract_hash,
            benchmark_items=items,
        )
    if (
        len(checkpoint_rows) != total_icps
        or len(final_attempts) != expected_attempt_count
        or final_producers != {normalized_from, normalized_candidate}
        or final_scope != [provider_cost_scope_hash]
    ):
        raise RuntimeError("terminal candidate checkpoint lineage differs")
    checkpoint_document = json.loads(
        checkpoint_s3.objects[checkpoint_location].decode("utf-8")
    )
    terminal_rows_hash = canonical_hash(
        sorted(checkpoint_rows, key=lambda row: str(row.get("icp_ref") or ""))
    )
    attempt_ledger_hash = canonical_hash(checkpoint_document["attempt_ledger"])
    receipt_graphs = [
        *[dict(graph) for graph in n_minus["receipt_graphs"]],
        *[dict(graph) for graph in candidate_graphs.values()],
    ]
    graph_roots = {
        str(graph.get("root_receipt_hash") or "") for graph in receipt_graphs
    }
    expected_receipt_frontier_count = (
        first_pass_success_count * completed_receipts_per_attempt
        + retryable_count
        * (
            retry_rounds * retryable_receipts_per_attempt
            + completed_receipts_per_attempt
        )
    )
    receipt_frontier_capacity = int(scoring_worker_module.MAX_EXTERNAL_RECEIPT_GRAPHS)
    if (
        graph_roots != final_roots
        or len(final_roots) != expected_receipt_frontier_count
        or expected_receipt_frontier_count > receipt_frontier_capacity
        or int(checkpoint_document["completed_icp_count"]) != total_icps
        or len(checkpoint_document["attempt_ledger"]["entries"])
        != expected_attempt_count
    ):
        raise RuntimeError("terminal checkpoint receipt graph frontier differs")

    source_inventory = transition_source_paths_by_commit(
        from_sha=normalized_from,
        candidate_sha=normalized_candidate,
    )
    source_identities = [
        git_blob_identity(normalized_root, commit, path)
        for commit, paths in source_inventory.items()
        for path in paths
    ]
    docker_collision["source_inventory"] = {
        commit: list(paths) for commit, paths in source_inventory.items()
    }
    docker_collision["source_identities"] = [
        dict(identity) for identity in source_identities
    ]
    if any(
        identity["sha256"]
        != hashlib.sha256((normalized_root / identity["path"]).read_bytes()).hexdigest()
        for identity in source_identities
        if identity["commit_sha"] == normalized_candidate
    ):
        raise RuntimeError("candidate production source differs from its exact commit")
    n_minus_maximum = {
        int(round_no): int(value)
        for round_no, value in n_minus["maximum_active_by_round"].items()
    }
    maximum_retry_active = max(
        [
            maximum_active_by_round.get(round_no, 0)
            for round_no in range(1, retry_rounds + 1)
        ]
        or [0]
    )
    return {
        "candidate_sha": normalized_candidate,
        "from_sha": normalized_from,
        "configured_icp_count": total_icps,
        "first_pass_success_count": first_pass_success_count,
        "first_pass_retryable_count": retryable_count,
        "completed_icp_count": len(checkpoint_rows),
        "baseline_concurrency": int(candidate_config.private_baseline_concurrency),
        "retry_concurrency": retry_width,
        "n_minus_one_baseline_concurrency": int(
            n_minus_config["private_baseline_concurrency"]
        ),
        "n_minus_one_retry_concurrency": int(
            n_minus_config["private_baseline_retry_concurrency"]
        ),
        "retry_rounds": retry_rounds,
        "scoring_worker_count": worker_count,
        "provider_preflight_ttl_seconds": preflight_ttl_seconds,
        "model_invocation_timeout_seconds": model_invocation_timeout_seconds,
        "watchdog_timeout_seconds": watchdog_timeout_seconds,
        "worker_recycle_rss_mb": recycle_rss_mb,
        "host_memory_floor_mib": memory_floor_mib,
        "topology_parent_vcpus": parent_vcpus,
        "topology_parent_memory_mib": parent_memory_mib,
        "topology_reserved_vcpus": reserved_vcpus,
        "topology_reserved_memory_mib": reserved_memory_mib,
        "topology_scoring_memory_mib": scoring_memory_mib,
        "checkpoint_write_count": int(n_minus["checkpoint_write_count"])
        + checkpoint_s3.put_count,
        "expected_attempt_count": expected_attempt_count,
        "maximum_first_pass_active": int(n_minus_maximum.get(0, 0)),
        "maximum_n_minus_retry_active": int(n_minus_maximum.get(1, 0)),
        "maximum_retry_active": maximum_retry_active,
        "retry_rounds_exercised": sorted(
            set(round_no for _index, round_no in all_calls)
        ),
        "retry_attempt_counts_by_round": retry_attempt_counts_by_round,
        "retry_wave_counts_by_round": retry_wave_counts_by_round,
        "retry_exhaustion_rounds_exercised": sorted(
            set(round_no for _index, round_no in exhausted_calls)
        ),
        "launch_environment_names": sorted(launch_environment),
        "n_minus_one_config": n_minus_config,
        "candidate_config": candidate_contract,
        "stale_recycle": stale_recycle,
        "rss_pressure_recycle": rss_positive["pressure"],
        "host_pressure_recycle": host_positive["pressure"],
        "rss_pressure_checkpoint_count": len(rss_positive["checkpoints"]),
        "host_pressure_checkpoint_count": len(host_positive["checkpoints"]),
        "pressure_exercised_icp_count": pressure_count,
        "rss_below_threshold_checkpoint_count": len(rss_negative["checkpoints"]),
        "memory_at_floor_checkpoint_count": len(floor_negative["checkpoints"]),
        "rss_below_threshold_no_recycle": not rss_negative["recycled"],
        "memory_at_floor_no_recycle": not floor_negative["recycled"],
        "source_identities": source_identities,
        "n_minus_one_envelope_scoring_worker_count": int(
            n_minus["envelope_scoring_worker_count"]
        ),
        "n_minus_one_sealed_scoring_profile_count": int(
            n_minus["sealed_scoring_profile_count"]
        ),
        "n_minus_one_fleet_launcher_worker_count": int(
            n_minus["fleet_launcher_worker_count"]
        ),
        "n_minus_one_supervisor_scoring_running": int(
            n_minus["supervisor_scoring_running"]
        ),
        "n_minus_one_supervisor_respawn_count": int(
            n_minus["supervisor_respawn_count"]
        ),
        "preflight_control_event_count": len(control_events),
        "preflight_partial_measurement_count": partial_measurements,
        "lease_non_owner_measurement_count": pre_lease_measurement_count,
        "lease_owner_forced_measurement_count": lease_owner_measurement_count,
        "n_minus_one_checkpoint_completed_icp_count": int(
            n_minus["checkpoint_completed_icp_count"]
        ),
        "n_minus_one_checkpoint_attempt_count": int(
            n_minus["checkpoint_attempt_count"]
        ),
        "n_minus_one_checkpoint_receipt_frontier_count": int(
            n_minus["checkpoint_receipt_frontier_count"]
        ),
        "n_minus_one_started_attempt_count": len(n_minus_started_calls),
        "n_minus_one_completed_attempt_count": len(n_minus_completed_calls),
        "n_minus_one_checkpointed_peer_call": list(checkpointed_peer_call),
        "n_minus_one_interrupted_peer_call": list(interrupted_peer_call),
        "n_minus_one_uncheckpointed_peer_count": len(
            uncheckpointed_n_minus_calls
        ),
        "candidate_replayed_uncheckpointed_peer_count": len(
            candidate_replayed_uncheckpointed
        ),
        "receipt_frontier_count": len(final_roots),
        "expected_receipt_frontier_count": expected_receipt_frontier_count,
        "receipt_frontier_capacity": receipt_frontier_capacity,
        "terminal_rows_hash": terminal_rows_hash,
        "attempt_ledger_hash": attempt_ledger_hash,
        "checkpoint_document_hash": canonical_hash(checkpoint_document),
        "receipt_frontier_hash": canonical_hash(sorted(final_roots)),
        "scoring_configuration_hash": scoring_configuration_hash,
        "scoring_contract_hash": scoring_contract_hash,
        "rolling_window_hash": window_hash,
        "model_artifact_hash": model_artifact_hash,
        "manifest_hash": manifest_hash,
        "repo_git_sha": repo_git_sha,
        "checkpoint_document": checkpoint_document,
        "receipt_graphs": receipt_graphs,
        "docker_collision": docker_collision,
        "dynamic_launch_config_candidate_n_minus_one_bound": True,
        "dynamic_exact_n_minus_one_worker_executed": True,
        "dynamic_n_minus_one_envelope_launcher_supervisor_exact": True,
        "dynamic_skewed_retry_round_checkpointed": True,
        "dynamic_all_retry_rounds_and_exhaustion_exact": True,
        "dynamic_retry_concurrency_bounded": True,
        "dynamic_preflight_actual_owned_matrix": True,
        "dynamic_preflight_fresh_and_stale_admission": True,
        "dynamic_full_fleet_lease_refresh_bound": True,
        "dynamic_pressure_checkpoint_recycle_bound": True,
        "dynamic_pressure_negative_boundaries_exact": True,
        "dynamic_watchdog_supervised_resume_bound": True,
        "dynamic_n_minus_one_candidate_resume_exact": True,
        "dynamic_peer_interruption_resume_exact": True,
        "dynamic_docker_collision_exact": True,
        "candidate_gateway_emergency_uses_guarded_reclaim": True,
        "first_activation_requires_preexisting_disk_reserve": True,
    }

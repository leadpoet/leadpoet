#!/usr/bin/env python3
"""Run the dynamic rebenchmark's producer half from an exact N-1 archive.

This file is a harness entrypoint.  The caller extracts the complete frozen
N-1 Git tree, and this process removes the candidate checkout from ``sys.path``
before importing any production module.  Only S3/KMS, provider transport, and
child-process boundaries are adapted.
"""

from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch


def _install_exact_root(root: Path) -> None:
    exact = str(root.resolve())
    candidate_root = Path(__file__).resolve().parents[2]
    retained: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved = Path(entry).resolve()
        except OSError:
            retained.append(entry)
            continue
        if resolved == candidate_root or candidate_root in resolved.parents:
            continue
        retained.append(entry)
    sys.path[:] = retained
    sys.path.insert(0, exact)


class _Body:
    def __init__(self, value: bytes) -> None:
        self._value = value

    def read(self) -> bytes:
        return self._value


class _MemoryS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}
        self.put_count = 0

    def put_object(self, **kwargs: Any) -> dict[str, str]:
        self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))] = bytes(
            kwargs["Body"]
        )
        self.put_count += 1
        return {"ETag": '"exact-n-minus-one"'}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "Body": _Body(self.objects[(str(kwargs["Bucket"]), str(kwargs["Key"]))])
        }


class _Runner:
    def __init__(self, worker_index: int = 0) -> None:
        self.worker_index = worker_index

    def with_worker_index(self, worker_index: int) -> "_Runner":
        return _Runner(worker_index)


def _append_collision_event(path: Path, event: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(event + "\n")
        handle.flush()


def _run_exact_docker_source_extraction(exact_root: Path) -> int:
    """Exercise archived model-authority source extraction at Docker only."""

    _install_exact_root(exact_root)
    context = json.loads(sys.stdin.read())
    if not isinstance(context, Mapping):
        raise RuntimeError("exact N-1 Docker extraction context is invalid")

    from gateway.research_lab import code_build as code_build_module
    from gateway.research_lab import model_authority_v2 as authority_module
    from gateway.research_lab.code_build import CodeEditBuildError
    from gateway.tee.source_bundle_v2 import (
        build_source_bundle_v2,
        compute_private_source_tree_hash,
    )

    image_digest = str(context["image_digest"])
    event_path = Path(str(context["event_path"]))
    host_detect_path = Path(str(context["host_detect_path"]))
    collision_timeout_seconds = float(context["collision_timeout_seconds"])
    strict_commands: list[list[str]] = []
    container_id = "strict-exact-n-minus-one-container"
    reader_started = False

    with tempfile.TemporaryDirectory(prefix="exact-n-minus-one-docker-source-") as raw:
        fixture_root = Path(raw) / "fixture"
        fixture_root.mkdir()
        required_dirs = tuple(code_build_module._REQUIRED_PARENT_APP_DIRS)
        required_files = tuple(code_build_module._REQUIRED_PARENT_APP_FILES)
        for relative in required_dirs:
            directory = fixture_root / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "rehearsal_source.py").write_text(
                f"SOURCE_PATH = {relative!r}\n",
                encoding="utf-8",
            )
        for relative in required_files:
            target = fixture_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                f"exact N-1 source fixture: {relative}\n",
                encoding="utf-8",
            )
        expected_tree_hash = compute_private_source_tree_hash(fixture_root)
        extracted_root = Path(raw) / "extracted"

        def strict_run(command: list[str], *, cwd: Path, timeout_seconds: int) -> str:
            nonlocal reader_started
            del cwd, timeout_seconds
            normalized = [str(value) for value in command]
            if not normalized or normalized[0] != "docker":
                raise RuntimeError(
                    "exact N-1 source extraction crossed a non-Docker boundary"
                )
            strict_commands.append(normalized)
            if normalized[1:3] == ["image", "inspect"]:
                if normalized[3:] != [image_digest]:
                    raise RuntimeError("exact N-1 inspected another image")
                if not reader_started:
                    reader_started = True
                    _append_collision_event(event_path, "n_minus_reader_started")
                    deadline = time.monotonic() + collision_timeout_seconds
                    while not host_detect_path.exists():
                        if time.monotonic() >= deadline:
                            raise RuntimeError(
                                "candidate host-live detection did not overlap "
                                "the exact N-1 reader"
                            )
                        time.sleep(
                            min(
                                0.02,
                                max(0.001, collision_timeout_seconds / 100.0),
                            )
                        )
                raise CodeEditBuildError("strict cold image cache")
            if normalized[1] == "pull":
                if normalized[-1] != image_digest:
                    raise RuntimeError("exact N-1 pulled another image")
                return "strict pull complete"
            if normalized[1] == "create":
                if image_digest not in normalized:
                    raise RuntimeError("exact N-1 created from another image")
                return container_id
            if normalized[1] == "cp":
                if normalized[2] != f"{container_id}:/app/.":
                    raise RuntimeError("exact N-1 copied another container path")
                destination = Path(normalized[3])
                shutil.copytree(fixture_root, destination, dirs_exist_ok=True)
                return "strict copy complete"
            if normalized[1:3] == ["rm", "-f"]:
                if normalized[3:] != [container_id]:
                    raise RuntimeError("exact N-1 cleaned another container")
                return "strict cleanup complete"
            raise RuntimeError(
                "exact N-1 source extraction issued an unknown Docker command"
            )

        with patch.object(code_build_module, "_run", strict_run):
            observed_tree_hash, _top_level_paths = (
                authority_module._extract_parent_image_source(
                    image_digest=image_digest,
                    source_dir=extracted_root,
                    timeout_seconds=int(context["timeout_seconds"]),
                )
            )
            bundle = build_source_bundle_v2(extracted_root)
        if observed_tree_hash != expected_tree_hash:
            raise RuntimeError("exact N-1 Docker extraction changed its tree identity")
        if bundle.get("source_tree_hash") != expected_tree_hash:
            raise RuntimeError("exact N-1 source bundle changed its tree identity")
        observed_operations = [" ".join(command[1:3]) for command in strict_commands]
        expected_operations = [
            "image inspect",
            "pull --platform",
            "create --platform",
            "cp " + f"{container_id}:/app/.",
            "rm -f",
        ]
        if observed_operations != expected_operations:
            raise RuntimeError("exact N-1 Docker extraction command order differs")
        _append_collision_event(event_path, "n_minus_source_extracted")

    result = {
        "source_tree_hash": expected_tree_hash,
        "source_bundle_hash": str(bundle["archive_sha256"]),
        "strict_docker_commands": strict_commands,
        "exact_model_authority_module": str(
            Path(authority_module.__file__).resolve().relative_to(exact_root)
        ),
        "exact_code_build_module": str(
            Path(code_build_module.__file__).resolve().relative_to(exact_root)
        ),
        "strict_docker_boundary_executed": True,
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--docker-source-extraction":
        return _run_exact_docker_source_extraction(Path(sys.argv[2]).resolve())
    if len(sys.argv) != 2:
        raise RuntimeError("exact N-1 root argument is required")
    exact_root = Path(sys.argv[1]).resolve()
    _install_exact_root(exact_root)
    context = json.loads(sys.stdin.read())
    if not isinstance(context, Mapping):
        raise RuntimeError("exact N-1 transition context is invalid")

    import boto3
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from gateway.research_lab import scoring_worker as scoring_worker_module
    from gateway.research_lab.scoring_worker import (
        ResearchLabGatewayScoringWorker,
        _BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD,
        _BASELINE_WAVE_STALL_EXIT_CODE,
        _baseline_attempt_ledger_entry,
        _baseline_summary_checkpointable,
        _baseline_wave_stall_timeout_seconds,
        _model_invocation_timeout_seconds,
        _store_baseline_scoring_progress,
        _worker_recycle_rss_mb,
    )
    from gateway.research_lab.provider_preflight import provider_preflight_settings
    from gateway.tee.scoring_executor import (
        SCORING_RUNTIME_ENV_NAMES,
        configuration_hash,
        runtime_environment_values,
    )
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

    commit_sha = str(context["from_sha"])
    envelope_environment = dict(os.environ)
    expected_runtime_environment = context["scoring_runtime_environment"]
    if not isinstance(expected_runtime_environment, Mapping) or set(
        expected_runtime_environment
    ) != set(SCORING_RUNTIME_ENV_NAMES):
        raise RuntimeError("exact N-1 scoring environment contract is invalid")
    for name in SCORING_RUNTIME_ENV_NAMES:
        os.environ.pop(name, None)
        value = expected_runtime_environment[name]
        if value is not None:
            os.environ[name] = str(value)
    config = ResearchLabGatewayConfig.from_env()
    scoring_configuration_hash = configuration_hash(runtime_environment_values())
    if scoring_configuration_hash != str(context["scoring_configuration_hash"]):
        raise RuntimeError("exact N-1 scoring configuration differs")
    items = [copy.deepcopy(dict(item)) for item in context["benchmark_items"]]
    total_icps = len(items)
    configured_total = int(config.conditional_validation_policy().total_icps)
    if total_icps != configured_total:
        raise RuntimeError("exact N-1 configured ICP bank differs")
    retry_rounds = int(config.private_baseline_provider_retry_rounds)
    retry_width = max(1, int(config.private_baseline_retry_concurrency))
    preflight_ttl_seconds = float(provider_preflight_settings()["ttl_seconds"])
    model_invocation_timeout_seconds = _model_invocation_timeout_seconds(
        float(config.scoring_worker_model_timeout_seconds)
    )
    watchdog_timeout_seconds = _baseline_wave_stall_timeout_seconds(config)
    recycle_rss_mb = int(_worker_recycle_rss_mb())
    if (
        preflight_ttl_seconds != float(context["provider_preflight_ttl_seconds"])
        or model_invocation_timeout_seconds
        != float(context["model_invocation_timeout_seconds"])
        or watchdog_timeout_seconds != float(context["watchdog_timeout_seconds"])
    ):
        raise RuntimeError("exact N-1 timing contract differs")
    first_pass_success_count = int(context["first_pass_success_count"])
    if not 0 < first_pass_success_count < total_icps:
        raise RuntimeError("exact N-1 skew is invalid")
    retryable_indexes = set(range(first_pass_success_count + 1, total_icps + 1))
    if retry_rounds < 1 or retry_width < 2 or len(retryable_indexes) <= retry_width:
        raise RuntimeError("exact N-1 cannot settle a partial retry wave")

    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = (
        signing_key.public_key()
        .public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        .hex()
    )
    issued_at = str(context["issued_at"])
    boot_body = build_boot_identity_body(
        role="gateway_scoring",
        physical_role="gateway_scoring",
        commit_sha=commit_sha,
        pcr0=hashlib.sha384((commit_sha + ":pcr0").encode("ascii")).hexdigest(),
        build_manifest_hash=sha256_json({"commit": commit_sha, "kind": "build"}),
        dependency_lock_hash=sha256_json({"commit": commit_sha, "kind": "lock"}),
        config_hash=sha256_json({"commit": commit_sha, "kind": "config"}),
        boot_nonce=hashlib.sha256((commit_sha + ":boot").encode("ascii")).hexdigest()[
            :32
        ],
        signing_pubkey=signing_pubkey,
        transport_pubkey=signing_pubkey,
        transport_certificate_hash=sha256_json(
            {"commit": commit_sha, "kind": "transport"}
        ),
        attestation_user_data_hash=sha256_json(
            {"commit": commit_sha, "kind": "user-data"}
        ),
        issued_at=issued_at,
    )
    boot_identity = create_boot_identity(
        body=boot_body,
        attestation_document_b64=base64.b64encode(
            b"strict-exact-n-minus-one-attestation"
        ).decode("ascii"),
    )
    receipt_graphs: dict[str, dict[str, Any]] = {}

    def attempt_receipts(
        *, item_index: int, retry_round: int, terminal: bool
    ) -> list[str]:
        roots: list[str] = []
        purposes = ["research_lab.private_model_run.v2"]
        if terminal:
            purposes.append("research_lab.company_score.v2")
        for sequence, purpose in enumerate(purposes):
            payload = {
                "icp_ref": str(items[item_index - 1]["icp_ref"]),
                "retry_round": retry_round,
                "purpose": purpose,
            }
            result = {"terminal": terminal, "item_index": item_index}
            body = build_execution_receipt_body(
                role="gateway_scoring",
                purpose=purpose,
                job_id=(
                    "n-minus-one-"
                    + hashlib.sha256(
                        f"{commit_sha}:{item_index}:{retry_round}:{purpose}".encode(
                            "ascii"
                        )
                    ).hexdigest()[:24]
                ),
                epoch_id=int(context["evaluation_epoch"]),
                sequence=sequence,
                commit_sha=commit_sha,
                pcr0=str(boot_identity["pcr0"]),
                build_manifest_hash=str(boot_identity["build_manifest_hash"]),
                dependency_lock_hash=str(boot_identity["dependency_lock_hash"]),
                config_hash=str(boot_identity["config_hash"]),
                boot_identity_hash=str(boot_identity["boot_identity_hash"]),
                input_root=sha256_json(payload),
                output_root=sha256_json(result),
                transport_root_hash=EMPTY_TRANSPORT_ROOT,
                host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
                artifact_root=merkle_root((), domain="leadpoet-artifact-v2"),
                parent_receipt_hashes=(),
                status="succeeded",
                failure_code=None,
                issued_at=issued_at,
            )
            receipt = create_signed_execution_receipt(
                body=body,
                enclave_pubkey=signing_pubkey,
                sign_digest=signing_key.sign,
            )
            graph = build_receipt_graph(
                root_receipt_hash=str(receipt["receipt_hash"]),
                boot_identities=(boot_identity,),
                receipts=(receipt,),
                transport_attempts=(),
                host_operations=(),
            )
            validate_receipt_graph(graph)
            root = str(graph["root_receipt_hash"])
            receipt_graphs[root] = graph
            roots.append(root)
        return roots

    # Invoke the exact N-1 per-ICP implementation once through strict model
    # and scorer interfaces. The larger skew/restart matrix below then drives
    # the same exact scheduler with deterministic checkpoint-shaped results.
    class StrictModelBoundary:
        async def __call__(
            self, _icp: Mapping[str, Any], _context: Mapping[str, Any]
        ) -> list[dict[str, str]]:
            return [
                {
                    "name": "Exact N-1 strict model boundary",
                    "domain": "example.invalid",
                }
            ]

    class StrictScorerBoundary:
        async def score_with_breakdowns(
            self,
            _outputs: Any,
            _icp: Mapping[str, Any],
            _include_details: bool,
        ) -> list[dict[str, Any]]:
            return [
                {
                    "final_score": 1.0,
                    "failure_reason": None,
                    "icp_fit": 1.0,
                    "intent_signal_final": 1.0,
                    "company_fit_decision": "match",
                    "verifier_gate_receipts": [],
                    "intent_signals_detail": [],
                }
            ]

    exact_icp_worker = object.__new__(ResearchLabGatewayScoringWorker)
    exact_icp_worker.worker_ref = "exact-n-minus-one-icp-path"
    exact_icp_worker.config = config
    exact_probe_item = {
        **items[0],
        "icp": dict(items[0].get("icp") or {}),
    }

    async def run_exact_icp_path() -> dict[str, Any]:
        with ThreadPoolExecutor(max_workers=1) as executor:
            return await ResearchLabGatewayScoringWorker._run_baseline_icp(
                exact_icp_worker,
                runner=StrictModelBoundary(),
                scorer=StrictScorerBoundary(),
                item=exact_probe_item,
                item_index=1,
                total_icps=total_icps,
                run_start=time.time(),
                executor=executor,
            )

    exact_icp_result = asyncio.run(run_exact_icp_path())
    if (
        exact_icp_result.get("_retryable") is not False
        or int(exact_icp_result.get("company_count") or 0) != 1
        or float(exact_icp_result.get("score") or 0.0) <= 0.0
    ):
        raise RuntimeError("exact N-1 per-ICP scoring path differs")

    clock = {"value": preflight_ttl_seconds / 2.0}
    active_by_round: dict[int, int] = {}
    maximum_active_by_round: dict[int, int] = {}
    started_attempt_calls: list[tuple[int, int]] = []
    completed_attempt_calls: list[tuple[int, int]] = []
    settled_attempt_calls: list[tuple[int, int]] = []
    retry_peers = sorted(retryable_indexes)[:2]
    checkpointed_peer_call = (retry_peers[0], 1)
    interrupted_peer_call = (retry_peers[1], 1)
    peer_checkpointed = None
    peer_interrupted = None

    class ExactPeerInterrupted(RuntimeError):
        pass

    class ExactPeerWaveAborted(RuntimeError):
        pass

    class ExactNMinusOneWorker(ResearchLabGatewayScoringWorker):
        async def _run_baseline_icp(
            self, *, item_index: int, retry_round: int, **_kwargs: Any
        ) -> dict[str, Any]:
            nonlocal peer_checkpointed, peer_interrupted
            active_by_round[retry_round] = active_by_round.get(retry_round, 0) + 1
            maximum_active_by_round[retry_round] = max(
                maximum_active_by_round.get(retry_round, 0),
                active_by_round[retry_round],
            )
            call = (item_index, retry_round)
            started_attempt_calls.append(call)
            try:
                await asyncio.sleep(0)
                if retry_round == 1:
                    if peer_checkpointed is None:
                        peer_checkpointed = asyncio.Event()
                        peer_interrupted = asyncio.Event()
                    if call == interrupted_peer_call:
                        await peer_checkpointed.wait()
                    elif call != checkpointed_peer_call:
                        assert peer_interrupted is not None
                        await peer_interrupted.wait()
                        raise ExactPeerWaveAborted(
                            "exact N-1 retry peer cancelled after fleet interruption"
                        )
                retryable = (
                    item_index in retryable_indexes and retry_round < retry_rounds
                )
                roots = attempt_receipts(
                    item_index=item_index,
                    retry_round=retry_round,
                    terminal=not retryable,
                )
                completed_attempt_calls.append(call)
                row = {
                    "icp_ref": str(items[item_index - 1]["icp_ref"]),
                    "icp_hash": str(items[item_index - 1].get("icp_hash") or ""),
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
                return row
            finally:
                active_by_round[retry_round] -= 1

    checkpoint_s3 = _MemoryS3()
    attempt_ledger: list[dict[str, Any]] = []
    terminal_rows: dict[str, dict[str, Any]] = {}
    parent_roots: set[str] = set()

    async def checkpoint_attempt(row: Mapping[str, Any], *, retry_round: int) -> bool:
        call = (int(row["_item_index"]), retry_round)
        if call == interrupted_peer_call:
            for root in row.get(_BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD, ()):
                receipt_graphs.pop(str(root), None)
            if peer_interrupted is None:
                raise RuntimeError("exact N-1 peer interrupt event is unavailable")
            peer_interrupted.set()
            raise ExactPeerInterrupted(
                "exact N-1 stopped after one peer checkpoint and one late reader"
            )
        attempt_ledger.append(
            _baseline_attempt_ledger_entry(
                row,
                retry_round=retry_round,
                gateway_runtime_commit_sha=commit_sha,
            )
        )
        parent_roots.update(
            str(value) for value in row.get(_BASELINE_ATTEMPT_RECEIPT_HASHES_FIELD, ())
        )
        if _baseline_summary_checkpointable(row):
            terminal_rows[str(row["icp_ref"])] = dict(row)
        with patch.object(boto3, "client", return_value=checkpoint_s3):
            _store_baseline_scoring_progress(
                str(context["checkpoint_bucket"]),
                str(context["checkpoint_key"]),
                benchmark_date=str(context["benchmark_date"]),
                window_hash=str(context["window_hash"]),
                private_model_artifact_hash=str(context["model_artifact_hash"]),
                gateway_runtime_commit_sha=commit_sha,
                scoring_configuration_hash_value=scoring_configuration_hash,
                rows=list(terminal_rows.values()),
                attested_parent_receipt_hashes=sorted(parent_roots),
                repo_git_sha=str(context["repo_git_sha"]),
                manifest_hash=str(context["manifest_hash"]),
                attempt_ledger=attempt_ledger,
                provider_cost_base_scope_hash=str(context["provider_cost_scope_hash"]),
                scoring_contract_hash_value=str(context["scoring_contract_hash"]),
            )
        settled_attempt_calls.append(call)
        if call == checkpointed_peer_call:
            if peer_checkpointed is None:
                raise RuntimeError("exact N-1 peer checkpoint event is unavailable")
            peer_checkpointed.set()
        return True

    async def unpaused() -> dict[str, Any]:
        return {"paused": False}

    worker = object.__new__(ExactNMinusOneWorker)
    worker.worker_ref = "exact-n-minus-one-rebenchmark"
    worker.config = config
    worker._baseline_profile_preflight_monotonic_at = {
        index: 0.0 for index in range(max(1, int(config.scoring_worker_total_workers)))
    }
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
        try:
            asyncio.run(
                worker._run_baseline_batch_inner(
                    runner=_Runner(),
                    retry_runner=_Runner(),
                    scorer=object(),
                    window=SimpleNamespace(benchmark_items=items),
                    run_start=time.time(),
                    attempt_checkpoint=checkpoint_attempt,
                    provider_cost_base_scope=str(context["provider_cost_scope_hash"]),
                    provider_preflight_boundary=(
                        worker._enforce_baseline_wave_preflight_freshness
                    ),
                )
            )
        except ExactPeerInterrupted as exc:
            peer_interruption = {
                "reason": "peer_interrupted_after_checkpoint",
                "error": str(exc),
                "checkpointed_peer_call": list(checkpointed_peer_call),
                "interrupted_peer_call": list(interrupted_peer_call),
            }
        else:
            raise RuntimeError("exact N-1 did not stop at its peer interruption")
    if (
        peer_interruption.get("reason") != "peer_interrupted_after_checkpoint"
        or checkpointed_peer_call not in settled_attempt_calls
        or interrupted_peer_call in settled_attempt_calls
        or interrupted_peer_call not in completed_attempt_calls
    ):
        raise RuntimeError("exact N-1 peer interruption boundary differs")

    checkpoint_location = (
        str(context["checkpoint_bucket"]),
        str(context["checkpoint_key"]),
    )
    checkpoint_document = json.loads(
        checkpoint_s3.objects[checkpoint_location].decode("utf-8")
    )

    # Execute the exact N-1 envelope preparer against the launch environment.
    from gateway.tee import prepare_gateway_envelopes_v2 as envelope_module

    class StrictKms:
        def encrypt(self, **kwargs: Any) -> dict[str, Any]:
            if not kwargs.get("KeyId") or not isinstance(
                kwargs.get("Plaintext"), (bytes, bytearray)
            ):
                raise RuntimeError("exact N-1 KMS envelope request differs")
            digest = hashlib.sha256(bytes(kwargs["Plaintext"])).digest()
            return {"CiphertextBlob": b"strict-kms:" + digest, "KeyId": kwargs["KeyId"]}

    probe_counts: dict[str, int] = {}

    def proxy_probe(fleets: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
        selected = {
            str(role): tuple(str(value) for value in values)
            for role, values in fleets.items()
        }
        probe_counts.update({role: len(values) for role, values in selected.items()})
        return selected

    with tempfile.TemporaryDirectory(prefix="exact-n-minus-one-envelopes-") as raw:
        envelope_report = envelope_module.prepare_gateway_envelopes_v2(
            environment=envelope_environment,
            kms_key_id="alias/strict-rehearsal-envelope",
            deploy_commit=commit_sha,
            output_dir=Path(raw) / "sealed",
            kms_client=StrictKms(),
            proxy_fleet_probe=proxy_probe,
        )

    # Execute the exact N-1 fleet launcher.  Only Popen is adapted; all worker
    # commands, partition environment, and cardinality come from production.
    from scripts import run_research_lab_scoring_worker_fleet as fleet_launcher

    fleet_commands: list[dict[str, Any]] = []

    class FinishedChild:
        pid = os.getpid()

        def __init__(self, command: Any, **kwargs: Any) -> None:
            fleet_commands.append(
                {
                    "command": [str(value) for value in command],
                    "env": dict(kwargs["env"]),
                }
            )

        def poll(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    launcher_argv = [str(fleet_launcher.WORKER_SCRIPT.parent / "fleet"), "--once"]
    with (
        patch.object(fleet_launcher.subprocess, "Popen", FinishedChild),
        patch.object(fleet_launcher.signal, "signal", lambda *_args: None),
        patch.object(fleet_launcher.time, "sleep", lambda *_args: None),
        patch.object(sys, "argv", launcher_argv),
    ):
        if fleet_launcher.main() != 0:
            raise RuntimeError("exact N-1 fleet launcher failed")

    expected_scoring_workers = int(envelope_report["scoring_worker_count"])
    if (
        len(fleet_commands) != expected_scoring_workers
        or any(
            int(entry["env"].get("RESEARCH_LAB_SCORING_WORKER_TOTAL_WORKERS") or 0)
            != expected_scoring_workers
            for entry in fleet_commands
        )
        or any(
            Path(entry["command"][1]).resolve()
            != (exact_root / "scripts/run_research_lab_scoring_worker.py").resolve()
            for entry in fleet_commands
        )
    ):
        raise RuntimeError("exact N-1 fleet launcher cardinality differs")

    # Produce the real derived watchdog exit, then let the exact production
    # supervisor observe that exit and respawn the affected scoring slot.
    watchdog_script = """
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.research_lab import scoring_worker as module
class ImmediateTimer:
    def __init__(self, _seconds, callback):
        self.callback = callback
        self.daemon = False
    def start(self):
        self.callback()
    def cancel(self):
        pass
module.threading.Timer = ImmediateTimer
config = ResearchLabGatewayConfig.from_env()
with module._baseline_wave_watchdog(
    worker_ref="exact-n-minus-one-supervised-watchdog",
    phase="retry",
    item_indexes=(max(1, config.private_baseline_retry_concurrency),),
    timeout_seconds=module._baseline_wave_stall_timeout_seconds(config),
):
    pass
"""
    watchdog_environment = dict(os.environ)
    watchdog_environment["PYTHONPATH"] = str(exact_root)
    watchdog = subprocess.run(
        [sys.executable, "-c", watchdog_script],
        cwd=exact_root,
        env=watchdog_environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=max(
            1.0,
            min(
                watchdog_timeout_seconds,
                model_invocation_timeout_seconds,
            ),
        ),
    )
    if (
        watchdog.returncode != _BASELINE_WAVE_STALL_EXIT_CODE
        or "research_lab_baseline_wave_stalled" not in watchdog.stderr
    ):
        raise RuntimeError("exact N-1 watchdog exit differs")

    from gateway.research_lab import worker_autostart as worker_autostart_module

    supervised_children: list[Any] = []

    class SupervisedChild:
        _next_pid = os.getpid() + 1

        def __init__(self, command: Any, **kwargs: Any) -> None:
            self.command = [str(value) for value in command]
            self.env = dict(kwargs["env"])
            self.returncode: int | None = None
            self.pid = SupervisedChild._next_pid
            SupervisedChild._next_pid += 1
            write_fd = int(tuple(kwargs.get("pass_fds") or ())[0])
            os.write(write_fd, b"ready\n")
            supervised_children.append(self)

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = 0

        def kill(self) -> None:
            self.returncode = 0

    supervisor_environment = dict(os.environ)
    poll_seconds = max(
        0.001,
        min(
            0.05,
            model_invocation_timeout_seconds
            / max(1.0, watchdog_timeout_seconds)
            / max(1, expected_scoring_workers),
        ),
    )
    supervisor_environment["RESEARCH_LAB_WORKER_SUPERVISOR_POLL_SECONDS"] = str(
        poll_seconds
    )
    with (
        patch.dict(os.environ, supervisor_environment, clear=True),
        patch.object(worker_autostart_module.subprocess, "Popen", SupervisedChild),
        patch.object(worker_autostart_module, "_child_rss_mb", return_value=None),
    ):
        plan = worker_autostart_module.build_research_lab_worker_autostart_plan(
            supervisor_environment
        )
        supervisor = worker_autostart_module.ResearchLabWorkerSupervisor(
            plan,
            environment=supervisor_environment,
        )
        supervisor.start()
        initial_child_count = len(supervised_children)
        initial_health = supervisor.health()
        failing = supervisor.children["scoring:0"]
        failing.returncode = watchdog.returncode
        deadline = time.monotonic() + max(
            1.0, poll_seconds * expected_scoring_workers * 4
        )
        resumed_health = None
        while time.monotonic() < deadline:
            if len(supervised_children) > initial_child_count:
                try:
                    observed_health = supervisor.health()
                except worker_autostart_module.ResearchLabWorkerStartupError:
                    pass
                else:
                    if (
                        int(observed_health["scoring_running"])
                        == expected_scoring_workers
                    ):
                        resumed_health = observed_health
                        break
            time.sleep(poll_seconds)
        supervisor.stop()
    if (
        len(supervised_children) <= initial_child_count
        or int(initial_health["scoring_running"]) != expected_scoring_workers
        or resumed_health is None
        or int(resumed_health["scoring_running"]) != expected_scoring_workers
        or not any(
            Path(child.command[1]).resolve()
            == (exact_root / "gateway/research_lab/worker_process.py").resolve()
            for child in supervised_children
        )
    ):
        raise RuntimeError("exact N-1 supervisor did not respawn watchdog exit")

    result = {
        "config": {
            "private_baseline_concurrency": int(config.private_baseline_concurrency),
            "private_baseline_retry_concurrency": retry_width,
            "private_baseline_provider_retry_rounds": retry_rounds,
            "scoring_worker_total_workers": int(config.scoring_worker_total_workers),
            "scoring_worker_model_timeout_seconds": int(
                config.scoring_worker_model_timeout_seconds
            ),
            "scoring_worker_min_available_memory_mb": int(
                config.scoring_worker_min_available_memory_mb
            ),
            "scoring_worker_max_load_per_cpu": float(
                config.scoring_worker_max_load_per_cpu
            ),
            "policy": config.conditional_validation_policy().to_dict(),
            "scoring_configuration_hash": scoring_configuration_hash,
            "provider_preflight_ttl_seconds": preflight_ttl_seconds,
            "model_invocation_timeout_seconds": (model_invocation_timeout_seconds),
            "watchdog_timeout_seconds": watchdog_timeout_seconds,
            "worker_recycle_rss_mb": recycle_rss_mb,
        },
        "checkpoint_document": checkpoint_document,
        "checkpoint_completed_icp_count": int(
            checkpoint_document["completed_icp_count"]
        ),
        "checkpoint_attempt_count": len(
            checkpoint_document["attempt_ledger"]["entries"]
        ),
        "checkpoint_receipt_frontier_count": len(
            checkpoint_document["attested_parent_receipt_hashes"]
        ),
        "receipt_graphs": list(receipt_graphs.values()),
        "attempt_calls": settled_attempt_calls,
        "started_attempt_calls": started_attempt_calls,
        "completed_attempt_calls": completed_attempt_calls,
        "checkpointed_peer_call": list(checkpointed_peer_call),
        "interrupted_peer_call": list(interrupted_peer_call),
        "peer_interruption": peer_interruption,
        "maximum_active_by_round": maximum_active_by_round,
        "checkpoint_write_count": checkpoint_s3.put_count,
        "envelope_scoring_worker_count": expected_scoring_workers,
        "envelope_hosted_worker_count": int(envelope_report["hosted_worker_count"]),
        "sealed_scoring_profile_count": int(
            envelope_report["worker_proxy_profile_counts"]["gateway_scoring"][
                "sealed_worker_slots"
            ]
        ),
        "proxy_probe_counts": probe_counts,
        "fleet_launcher_worker_count": len(fleet_commands),
        "supervisor_initial_child_count": initial_child_count,
        "supervisor_respawn_count": len(supervised_children) - initial_child_count,
        "supervisor_scoring_running": int(resumed_health["scoring_running"]),
        "watchdog_exit_code": watchdog.returncode,
        "watchdog_timeout_seconds": watchdog_timeout_seconds,
        "exact_per_icp_scoring_path_executed": True,
        "exact_scoring_worker_module": str(
            Path(scoring_worker_module.__file__).resolve().relative_to(exact_root)
        ),
        "exact_fleet_launcher_module": str(
            Path(fleet_launcher.__file__).resolve().relative_to(exact_root)
        ),
        "exact_envelope_preparer_module": str(
            Path(envelope_module.__file__).resolve().relative_to(exact_root)
        ),
        "exact_scoring_worker_entrypoint": str(
            (exact_root / "scripts/run_research_lab_scoring_worker.py")
            .resolve()
            .relative_to(exact_root)
        ),
        "exact_supervisor_module": str(
            Path(worker_autostart_module.__file__).resolve().relative_to(exact_root)
        ),
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

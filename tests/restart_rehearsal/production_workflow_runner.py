#!/usr/bin/env python3.11
"""Execute the real V2 canonical, signing, SDK, receipt, and auditor path.

Input generation is test-only.  Every security-sensitive output is produced or
validated by candidate production modules.  The irreversible chain broadcast
and production database are replaced by :mod:`local_services`.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


SOURCE_ROOT = Path(os.environ.get("REHEARSAL_SOURCE_ROOT", "/source")).resolve()
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from leadpoet_canonical.attested_v2 import (  # noqa: E402
    EMPTY_HOST_OPERATION_ROOT,
    build_execution_receipt_body,
    build_receipt_graph,
    create_signed_execution_receipt,
    merkle_root,
    sha256_json,
)
from leadpoet_canonical.auditor_v2 import (  # noqa: E402
    verify_attested_weight_authority_v2,
    verify_attested_weight_bundle_v2,
)
from leadpoet_canonical.hotkey_authority_v2 import (  # noqa: E402
    build_weight_extrinsic_authorization_v2,
    chain_signing_profiles,
    encode_signed_extrinsic_v2,
    signed_extrinsic_hash_v2,
)
from leadpoet_canonical.weight_authority_v2 import (  # noqa: E402
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)
from local_services import (  # noqa: E402
    LocalBoundaryServices,
    LocalEnclaveSigningBoundary,
    LocalSDKSubstrateBoundary,
    local_enclave_backed_wallet,
)
from sanitized_weight_fixture import (  # noqa: E402
    EMPTY_ARTIFACT_ROOT,
    EMPTY_TRANSPORT_ROOT,
    SanitizedWeightFixture,
    VALIDATOR_HOTKEY,
)
from validator_tee.enclave.hotkey_authority_v2 import (  # noqa: E402
    _Sr25519Backend,
)
from validator_tee.host.weight_authority_v2 import (  # noqa: E402
    build_authoritative_weight_bundle_v2,
)
from gateway.tee.rehearsal_behavior_contract_v2 import (  # noqa: E402
    build_rehearsal_behavior_contract_v2,
    validate_rehearsal_behavior_contract_v2,
)
from validator_tee.host.enclave_hotkey_v2 import (  # noqa: E402
    AuthoritativeSetWeightsContextV2,
    _weight_extrinsic_module,
)


NOW = "2026-07-25T00:00:00Z"
GENESIS_HASH = (
    "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
)
HOST_RESTART_SUMMARY_SOURCE_PATHS = (
    "leadpoet_observability/sentry_operations.py",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")



def _run_workflow_stage(
    *,
    stage: str,
    action: Callable[[], Any],
    stages: list[dict[str, Any]],
    duration_seconds: float | None = None,
) -> tuple[bool, Any]:
    """Fail one stage while allowing independent downstream probes to run."""

    try:
        value = action()
    except KeyboardInterrupt:
        raise
    except BaseException as exc:
        result = {
            "error": str(exc)[:2000],
            "error_type": type(exc).__name__,
            "stage": stage,
            "status": "failed",
            "traceback": traceback.format_exc(limit=20)[-12000:],
        }
        if duration_seconds is not None:
            result["duration_seconds"] = duration_seconds
        stages.append(result)
        print(
            "PRODUCTION_WORKFLOW_STAGE_FAILED_CONTINUING "
            f"stage={stage} error_type={result['error_type']} "
            f"error={result['error']!r}",
            file=sys.stderr,
            flush=True,
        )
        return False, None
    result = {"stage": stage, "status": "passed"}
    if duration_seconds is not None:
        result["duration_seconds"] = duration_seconds
    stages.append(result)
    print(f"PRODUCTION_WORKFLOW_STAGE_PASSED stage={stage}", flush=True)
    return True, value


_BEHAVIOR_WORKER_SCHEMA = "leadpoet.workflow_behavior_worker.v1"
_BEHAVIOR_WORKER_LIMIT = 3
_BEHAVIOR_WORKER_POLL_SECONDS = 0.005
_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS = 2.0
_BEHAVIOR_WORKER_RESULT_MAX_BYTES = 16 * 1024 * 1024
_BEHAVIOR_WORKER_DURATION_MAX_SECONDS = 600.0
_BEHAVIOR_WORKER_ERROR_MAX_CHARS = 2000
_BEHAVIOR_WORKER_ERROR_TYPE_MAX_CHARS = 200
_BEHAVIOR_WORKER_TRACEBACK_MAX_CHARS = 12000


class _BehaviorWorkerSignal(SystemExit):
    """Preserve the invoking signal exit status after child cleanup."""

    def __init__(self, signum: int) -> None:
        self.signum = int(signum)
        super().__init__(128 + self.signum)


class _BehaviorWorkerProtocolError(RuntimeError):
    """A behavior subprocess did not return one exact bound result."""


def _behavior_worker_command(
    *,
    scenario: str,
    result_path: Path,
    token: str,
    ordinal: int,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--behavior-worker",
        "--scenario",
        scenario,
        "--result",
        str(result_path),
        "--token",
        token,
        "--ordinal",
        str(ordinal),
    ]


def _write_behavior_worker_result(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = _canonical(dict(payload)) + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(encoded)
            handle.flush()
    finally:
        os.close(descriptor)


def _behavior_worker_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--ordinal", type=int, required=True)
    args = parser.parse_args(list(argv))
    state_root = os.environ.get("REHEARSAL_STATE_ROOT")
    source_root = os.environ.get("REHEARSAL_SOURCE_ROOT")
    common = {
        "ordinal": args.ordinal,
        "scenario": args.scenario,
        "schema_version": _BEHAVIOR_WORKER_SCHEMA,
        "source_root": source_root,
        "state_root": state_root,
        "token": args.token,
    }
    action_started_at = time.monotonic()
    try:
        if not state_root:
            raise RuntimeError("behavior worker state root is unavailable")
        if not source_root:
            raise RuntimeError("behavior worker source root is unavailable")
        action = BEHAVIOR_ACTIONS.get(args.scenario)
        if action is None:
            raise RuntimeError(
                f"candidate behavior scenario has no runner: {args.scenario}"
            )
        value = action()
        duration_seconds = time.monotonic() - action_started_at
        _write_behavior_worker_result(
            args.result,
            {
                **common,
                "duration_seconds": duration_seconds,
                "status": "passed",
                "value": value,
            },
        )
    except Exception as exc:
        duration_seconds = time.monotonic() - action_started_at
        _write_behavior_worker_result(
            args.result,
            {
                **common,
                "duration_seconds": duration_seconds,
                "error": str(exc)[:_BEHAVIOR_WORKER_ERROR_MAX_CHARS],
                "error_type": type(exc).__name__[
                    :_BEHAVIOR_WORKER_ERROR_TYPE_MAX_CHARS
                ],
                "status": "failed",
                "traceback": traceback.format_exc(limit=20)[
                    -_BEHAVIOR_WORKER_TRACEBACK_MAX_CHARS:
                ],
            },
        )
    return 0


def _terminate_behavior_worker(process: subprocess.Popen[Any]) -> None:
    """TERM, bounded KILL, and reap one action-owned process group."""

    if process.poll() is not None:
        process.wait(timeout=0)
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=_BEHAVIOR_WORKER_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"behavior worker pid={process.pid} could not be reaped"
        ) from exc


@contextmanager
def _behavior_worker_signal_handlers():
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("behavior worker scheduler requires the main thread")
    previous = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt(signum: int, _frame: Any) -> None:
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise _BehaviorWorkerSignal(signum)

    for signum in previous:
        signal.signal(signum, interrupt)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _defer_behavior_worker_signals():
    previous = signal.pthread_sigmask(
        signal.SIG_BLOCK,
        {signal.SIGINT, signal.SIGTERM},
    )
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _read_behavior_worker_result(
    *,
    result_path: Path,
    scenario: str,
    token: str,
    ordinal: int,
    state_root: Path,
    source_root: str,
) -> dict[str, Any]:
    try:
        size = result_path.stat().st_size
    except FileNotFoundError as exc:
        raise _BehaviorWorkerProtocolError(
            "behavior worker result is missing"
        ) from exc
    if size <= 0 or size > _BEHAVIOR_WORKER_RESULT_MAX_BYTES:
        raise _BehaviorWorkerProtocolError(
            f"behavior worker result size is invalid: {size}"
        )
    lines = result_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise _BehaviorWorkerProtocolError(
            f"behavior worker emitted {len(lines)} results instead of one"
        )
    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise _BehaviorWorkerProtocolError(
            "behavior worker result is malformed"
        ) from exc
    if not isinstance(payload, dict):
        raise _BehaviorWorkerProtocolError(
            "behavior worker result must be an object"
        )
    common = {
        "ordinal": ordinal,
        "scenario": scenario,
        "schema_version": _BEHAVIOR_WORKER_SCHEMA,
        "source_root": source_root,
        "state_root": str(state_root),
        "token": token,
    }
    for field, expected in common.items():
        if payload.get(field) != expected:
            raise _BehaviorWorkerProtocolError(
                f"behavior worker result has wrong {field}"
            )
    duration_seconds = payload.get("duration_seconds")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or not 0.0 < float(duration_seconds) <= _BEHAVIOR_WORKER_DURATION_MAX_SECONDS
    ):
        raise _BehaviorWorkerProtocolError(
            "behavior worker result has invalid duration_seconds"
        )
    status = payload.get("status")
    if status == "passed":
        expected_fields = {*common, "duration_seconds", "status", "value"}
    elif status == "failed":
        expected_fields = {
            *common,
            "duration_seconds",
            "error",
            "error_type",
            "status",
            "traceback",
        }
        if not all(
            isinstance(payload.get(field), str)
            for field in ("error", "error_type", "traceback")
        ):
            raise _BehaviorWorkerProtocolError(
                "behavior worker failure result is malformed"
            )
        if (
            len(payload["error"]) > _BEHAVIOR_WORKER_ERROR_MAX_CHARS
            or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_.]*",
                payload["error_type"],
            )
            or len(payload["error_type"])
            > _BEHAVIOR_WORKER_ERROR_TYPE_MAX_CHARS
            or len(payload["traceback"])
            > _BEHAVIOR_WORKER_TRACEBACK_MAX_CHARS
        ):
            raise _BehaviorWorkerProtocolError(
                "behavior worker failure result exceeds its bounds"
            )
    else:
        raise _BehaviorWorkerProtocolError(
            "behavior worker result has invalid status"
        )
    if set(payload) != expected_fields:
        raise _BehaviorWorkerProtocolError(
            "behavior worker result has an unexpected field inventory"
        )
    return payload


def _observed_behavior_duration(started_at: float) -> float:
    return min(
        _BEHAVIOR_WORKER_DURATION_MAX_SECONDS,
        max(sys.float_info.epsilon, time.monotonic() - started_at),
    )


def _replay_behavior_worker_payload(
    *,
    stage: str,
    payload: Mapping[str, Any],
    stages: list[dict[str, Any]],
) -> tuple[bool, Any]:
    duration_seconds = float(payload["duration_seconds"])
    if payload["status"] == "passed":
        stages.append(
            {
                "duration_seconds": duration_seconds,
                "stage": stage,
                "status": "passed",
            }
        )
        print(f"PRODUCTION_WORKFLOW_STAGE_PASSED stage={stage}", flush=True)
        return True, payload["value"]
    result = {
        "duration_seconds": duration_seconds,
        "error": payload["error"],
        "error_type": payload["error_type"],
        "stage": stage,
        "status": "failed",
        "traceback": payload["traceback"],
    }
    stages.append(result)
    print(
        "PRODUCTION_WORKFLOW_STAGE_FAILED_CONTINUING "
        f"stage={stage} error_type={result['error_type']} "
        f"error={result['error']!r}",
        file=sys.stderr,
        flush=True,
    )
    return False, None


def _cleanup_behavior_workers(
    active: Mapping[int, Mapping[str, Any]],
    original: BaseException,
) -> None:
    errors = []
    with _defer_behavior_worker_signals():
        for ordinal in sorted(active):
            process = active[ordinal]["process"]
            try:
                _terminate_behavior_worker(process)
            except BaseException as exc:
                errors.append(
                    f"pid={process.pid} error_type={type(exc).__name__} "
                    f"error={exc!s}"
                )
    if not errors:
        return
    message = "behavior worker cleanup failed: " + "; ".join(errors)
    print(message, file=sys.stderr, flush=True)
    add_note = getattr(original, "add_note", None)
    if callable(add_note):
        add_note(message)


def _run_behavior_actions(
    *,
    scenarios: Sequence[str],
    stages: list[dict[str, Any]],
    deadline_monotonic: float | None = None,
    worker_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Run independent behavior probes in isolated execs and merge canonically."""

    if worker_timeout_seconds is not None and worker_timeout_seconds <= 0:
        raise ValueError("behavior worker timeout must be positive")
    source_root = os.environ.get("REHEARSAL_SOURCE_ROOT") or str(SOURCE_ROOT)
    pending = list(enumerate(str(item) for item in scenarios))
    active: dict[int, dict[str, Any]] = {}
    completed: dict[int, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="leadpoet-workflow-behavior-") as raw:
        worker_root = Path(raw)
        with _behavior_worker_signal_handlers():
            try:
                while pending or active:
                    if (
                        deadline_monotonic is not None
                        and time.monotonic() >= deadline_monotonic
                    ):
                        raise TimeoutError(
                            "behavior worker scheduler deadline exceeded"
                        )
                    while pending and len(active) < _BEHAVIOR_WORKER_LIMIT:
                        ordinal, scenario = pending.pop(0)
                        token = hashlib.sha256(
                            f"{os.getpid()}:{time.monotonic_ns()}:{ordinal}:"
                            f"{scenario}".encode("utf-8")
                        ).hexdigest()
                        state_root = worker_root / f"state-{ordinal:02d}-{token[:12]}"
                        result_path = worker_root / f"result-{ordinal:02d}.jsonl"
                        stdout_path = worker_root / f"stdout-{ordinal:02d}.log"
                        stderr_path = worker_root / f"stderr-{ordinal:02d}.log"
                        env = os.environ.copy()
                        env["REHEARSAL_SOURCE_ROOT"] = source_root
                        env["REHEARSAL_STATE_ROOT"] = str(state_root)
                        entry = {
                            "ordinal": ordinal,
                            "result_path": result_path,
                            "scenario": scenario,
                            "source_root": source_root,
                            "state_root": state_root,
                            "stderr_path": stderr_path,
                            "stdout_path": stdout_path,
                            "token": token,
                        }
                        entry["started_at"] = time.monotonic()
                        if scenario not in BEHAVIOR_ACTIONS:
                            completed[ordinal] = {
                                **entry,
                                "observed_duration_seconds": (
                                    _observed_behavior_duration(
                                        entry["started_at"]
                                    )
                                ),
                                "error": (
                                    "candidate behavior scenario has no runner: "
                                    f"{scenario}"
                                ),
                            }
                            continue
                        with (
                            stdout_path.open("xb") as stdout_handle,
                            stderr_path.open("xb") as stderr_handle,
                        ):
                            with _defer_behavior_worker_signals():
                                process = subprocess.Popen(
                                    _behavior_worker_command(
                                        scenario=scenario,
                                        result_path=result_path,
                                        token=token,
                                        ordinal=ordinal,
                                    ),
                                    env=env,
                                    stdin=subprocess.DEVNULL,
                                    stdout=stdout_handle,
                                    stderr=stderr_handle,
                                    start_new_session=True,
                                )
                                entry["process"] = process
                                active[ordinal] = entry

                    now = time.monotonic()
                    if (
                        deadline_monotonic is not None
                        and now >= deadline_monotonic
                    ):
                        raise TimeoutError(
                            "behavior worker scheduler deadline exceeded"
                        )
                    ready = [
                        ordinal
                        for ordinal, entry in active.items()
                        if entry["process"].poll() is not None
                    ]
                    if worker_timeout_seconds is not None:
                        timed_out = [
                            ordinal
                            for ordinal, entry in active.items()
                            if ordinal not in ready
                            and now - entry["started_at"]
                            >= worker_timeout_seconds
                        ]
                    else:
                        timed_out = []
                    for ordinal in timed_out:
                        entry = active[ordinal]
                        _terminate_behavior_worker(entry["process"])
                        active.pop(ordinal)
                        entry["observed_duration_seconds"] = (
                            _observed_behavior_duration(entry["started_at"])
                        )
                        entry["error"] = "behavior worker timed out"
                        completed[ordinal] = entry
                    for ordinal in ready:
                        if ordinal not in active:
                            continue
                        entry = active.pop(ordinal)
                        process = entry["process"]
                        process.wait(timeout=0)
                        if process.returncode != 0:
                            entry["error"] = (
                                "behavior worker crashed "
                                f"returncode={process.returncode}"
                            )
                        else:
                            try:
                                entry["payload"] = _read_behavior_worker_result(
                                    result_path=entry["result_path"],
                                    scenario=entry["scenario"],
                                    token=entry["token"],
                                    ordinal=entry["ordinal"],
                                    state_root=entry["state_root"],
                                    source_root=entry["source_root"],
                                )
                            except Exception as exc:
                                entry["error"] = (
                                    f"{type(exc).__name__}: {exc!s}"
                                )
                        entry["observed_duration_seconds"] = (
                            _observed_behavior_duration(entry["started_at"])
                        )
                        completed[ordinal] = entry
                    if not ready and not timed_out and active:
                        sleep_seconds = _BEHAVIOR_WORKER_POLL_SECONDS
                        if deadline_monotonic is not None:
                            sleep_seconds = min(
                                sleep_seconds,
                                max(0.0, deadline_monotonic - time.monotonic()),
                            )
                        if worker_timeout_seconds is not None:
                            sleep_seconds = min(
                                sleep_seconds,
                                max(
                                    0.0,
                                    min(
                                        entry["started_at"]
                                        + worker_timeout_seconds
                                        - time.monotonic()
                                        for entry in active.values()
                                    ),
                                ),
                            )
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
            except BaseException as original:
                _cleanup_behavior_workers(active, original)
                raise

        evidence: dict[str, Any] = {}
        for ordinal, scenario in enumerate(scenarios):
            entry = completed[ordinal]
            stdout = entry["stdout_path"].read_text(
                encoding="utf-8", errors="replace"
            ) if entry["stdout_path"].exists() else ""
            stderr = entry["stderr_path"].read_text(
                encoding="utf-8", errors="replace"
            ) if entry["stderr_path"].exists() else ""
            if stdout:
                sys.stdout.write(stdout)
                sys.stdout.flush()
            if stderr:
                sys.stderr.write(stderr)
                sys.stderr.flush()
            if "error" in entry:
                action = lambda entry=entry: (_ for _ in ()).throw(
                    _BehaviorWorkerProtocolError(str(entry["error"]))
                )
                duration_seconds = entry["observed_duration_seconds"]
                passed, value = _run_workflow_stage(
                    stage=f"behavior:{scenario}",
                    action=action,
                    stages=stages,
                    duration_seconds=duration_seconds,
                )
            else:
                passed, value = _replay_behavior_worker_payload(
                    stage=f"behavior:{scenario}",
                    payload=entry["payload"],
                    stages=stages,
                )
            if passed:
                evidence[str(scenario)] = value
        return evidence


def _mark_workflow_stage_unexercised(
    *,
    stage: str,
    blocked_by: list[str],
    stages: list[dict[str, Any]],
) -> None:
    stages.append(
        {
            "blocked_by": list(blocked_by),
            "stage": stage,
            "status": "unexercised",
        }
    )
    print(
        "PRODUCTION_WORKFLOW_STAGE_UNEXERCISED "
        f"stage={stage} blocked_by={','.join(blocked_by)}",
        file=sys.stderr,
        flush=True,
    )


def _require_equal(left: Any, right: Any, message: str) -> Any:
    if left != right:
        raise RuntimeError(message)
    return left


class _AuditorScaleValue:
    def __init__(self, value: Any):
        self.value = value


class _AuditorLocalSubstrate:
    """Exact-hash chain boundary consumed by the production auditor."""

    def __init__(self, *, epoch_id: int, block: int):
        self.epoch_id = int(epoch_id)
        self.block = int(block)
        self.last_epoch_block = self.epoch_id * 360
        self.last_update = 0
        self.weights: list[tuple[int, int]] = []

    @staticmethod
    def _hash(block: int) -> str:
        return "0x" + hashlib.sha256(
            f"leadpoet-auditor-local-block:{block}".encode("ascii")
        ).hexdigest()

    def get_block_hash(self, block: int) -> str:
        return GENESIS_HASH if int(block) == 0 else self._hash(int(block))

    def get_chain_finalised_head(self) -> str:
        return self._hash(self.block)

    def get_chain_head(self) -> str:
        return self._hash(self.block)

    def get_block_number(self, block_hash: str) -> int:
        if block_hash == GENESIS_HASH:
            return 0
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local chain received an unknown hash")
        return self.block

    def query(
        self,
        *,
        module: str,
        storage_function: str,
        params: list[Any],
        block_hash: str,
    ) -> _AuditorScaleValue:
        if block_hash != self._hash(self.block):
            raise RuntimeError("auditor local query is not exact-hash pinned")
        if module == "Timestamp" and storage_function == "Now" and params == []:
            return _AuditorScaleValue(
                int(datetime(2026, 7, 25, tzinfo=timezone.utc).timestamp())
                * 1000
            )
        if module != "SubtensorModule":
            raise RuntimeError("auditor local query module differs")
        if params == [71]:
            scheduler = {
                "Tempo": 360,
                "LastEpochBlock": self.last_epoch_block,
                "PendingEpochAt": self.last_epoch_block + 360,
                "SubnetEpochIndex": self.epoch_id,
                "BlocksSinceLastStep": self.block - self.last_epoch_block,
                "RevealPeriodEpochs": 1,
                "LastUpdate": [self.last_update],
            }
            if storage_function not in scheduler:
                raise RuntimeError("auditor local scheduler field differs")
            return _AuditorScaleValue(scheduler[storage_function])
        if params == [71, 0] and storage_function == "Weights":
            return _AuditorScaleValue(list(self.weights))
        raise RuntimeError("auditor local query shape differs")


class _AuditorLocalSubtensor:
    def __init__(self, substrate: _AuditorLocalSubstrate):
        self.substrate = substrate

    def get_subnet_hyperparameters(
        self, netuid: int, block: int | None = None
    ) -> Any:
        if int(netuid) != 71 or block is not None:
            raise RuntimeError("auditor local hyperparameter request differs")
        return SimpleNamespace(tempo=360, commit_reveal_period=1)

    def set_weights(
        self,
        *,
        netuid: int,
        wallet: Any,
        uids: list[int],
        weights: list[float],
        wait_for_finalization: bool,
        mechid: int,
    ) -> tuple[bool, str]:
        del wallet
        if (
            int(netuid) != 71
            or wait_for_finalization is not True
            or int(mechid) != 0
            or len(uids) != len(weights)
        ):
            raise RuntimeError("auditor local set_weights contract differs")
        from leadpoet_canonical.weights import normalize_to_u16

        self.substrate.weights = list(
            zip(
                [int(uid) for uid in uids],
                normalize_to_u16(
                    [int(uid) for uid in uids],
                    [float(weight) for weight in weights],
                ),
            )
        )
        self.substrate.last_update = self.substrate.block
        return True, "local finalized chain boundary accepted"


def _run_production_auditor(
    *,
    authority: Mapping[str, Any],
    identity_cache: Mapping[str, Any],
    epoch_id: int,
    block: int,
) -> dict[str, Any]:
    """Run the real auditor verifier, exact-block gate, and submit loop."""

    import neurons.auditor_validator as auditor_module
    from Leadpoet.utils.subnet_epoch import SubnetEpochCutover

    substrate = _AuditorLocalSubstrate(epoch_id=epoch_id, block=block)
    auditor = auditor_module.AuditorValidator.__new__(
        auditor_module.AuditorValidator
    )
    auditor.config = SimpleNamespace(
        netuid=71,
        subtensor=SimpleNamespace(network="local"),
    )
    auditor.epoch_cutover = SubnetEpochCutover(
        network_genesis_hash=GENESIS_HASH,
        netuid=71,
        cutover_block=30_000 * 360,
        cutover_block_hash=_AuditorLocalSubstrate._hash(30_000 * 360),
        first_subnet_epoch_index=30_000,
        first_settlement_epoch_id=30_000,
        last_legacy_epoch_id=29_999,
    )
    auditor.epoch_archive_endpoint = "local://archive-boundary"
    auditor.epoch_archive_subtensor = _AuditorLocalSubtensor(substrate)
    auditor.subtensor = _AuditorLocalSubtensor(substrate)
    auditor.uid = 0
    auditor.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(
            ss58_address="5FqLp5QmNRiHGyj3xbLVnDHfCx25qxJX5CUhpndF9GFfZZiK"
        )
    )
    auditor.last_submitted_epoch = None
    auditor.last_authority_epoch = None

    original = auditor_module.verify_attested_weight_authority_v2

    def verify_with_local_nitro(
        value: Mapping[str, Any],
        *,
        identity_cache: Mapping[str, Any],
        chain_signing_profile: Mapping[str, Any],
    ) -> dict[str, Any]:
        return original(
            value,
            identity_cache=identity_cache,
            chain_signing_profile=chain_signing_profile,
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        )

    auditor_module.verify_attested_weight_authority_v2 = (
        verify_with_local_nitro
    )
    try:
        verified = auditor.verify_attested_weights_v2(
            dict(authority),
            identity_cache=dict(identity_cache),
        )
    finally:
        auditor_module.verify_attested_weight_authority_v2 = original
    if verified is None:
        raise RuntimeError("production auditor rejected local authority")
    submitted = auditor.submit_weights_to_chain(
        epoch_id,
        verified,
        submission_epoch_id=epoch_id,
    )
    if not submitted:
        raise RuntimeError("production auditor did not finalize local weights")
    return verified


def _file_identity(path: str, candidate_sha: str) -> dict[str, str]:
    source = SOURCE_ROOT / path
    if not source.is_file():
        raise RuntimeError(f"candidate production source is absent: {path}")
    import subprocess

    expected = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "show", f"{candidate_sha}:{path}"],
        check=True,
        capture_output=True,
    ).stdout
    observed = source.read_bytes()
    if observed != expected:
        raise RuntimeError(f"candidate production source differs: {path}")
    return {
        "path": path,
        "sha256": hashlib.sha256(observed).hexdigest(),
        "commit_sha": candidate_sha,
    }


def _receipt(
    *,
    epoch_id: int,
    candidate_sha: str,
    role: str,
    purpose: str,
    job_id: str,
    private_key: Ed25519PrivateKey,
    boot: Mapping[str, Any],
    config_hash: str,
    input_root: str,
    output_root: str,
    parents: list[str],
    sequence: int,
    transport_attempts: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    attempts = list(transport_attempts or [])
    public_key = private_key.public_key().public_bytes_raw().hex()
    artifact_hashes = [
        item[key]
        for item in attempts
        for key in ("request_artifact_hash", "response_artifact_hash")
    ]
    body = build_execution_receipt_body(
        role=role,
        purpose=purpose,
        job_id=job_id,
        epoch_id=epoch_id,
        sequence=sequence,
        commit_sha=candidate_sha,
        pcr0=str(boot["pcr0"]),
        build_manifest_hash=str(boot["build_manifest_hash"]),
        dependency_lock_hash=str(boot["dependency_lock_hash"]),
        config_hash=config_hash,
        boot_identity_hash=str(boot["boot_identity_hash"]),
        input_root=input_root,
        output_root=output_root,
        transport_root_hash=(
            merkle_root(
                [str(item["attempt_hash"]) for item in attempts],
                domain="leadpoet-transport-v2",
            )
            if attempts
            else EMPTY_TRANSPORT_ROOT
        ),
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=(
            merkle_root(artifact_hashes, domain="leadpoet-artifact-v2")
            if artifact_hashes
            else EMPTY_ARTIFACT_ROOT
        ),
        parent_receipt_hashes=parents,
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    return create_signed_execution_receipt(
        body=body,
        enclave_pubkey=public_key,
        sign_digest=private_key.sign,
    )


def _exercise_sdk_bridge(
    *,
    epoch_id: int,
    uids: list[int],
    weights_u16: list[int],
    submission_event_hash: str,
) -> dict[str, Any]:
    """Run the production Bittensor SDK interception with strict boundaries."""

    client = LocalEnclaveSigningBoundary()
    substrate = LocalSDKSubstrateBoundary()
    wallet = local_enclave_backed_wallet(client)
    mechanism = _weight_extrinsic_module()
    with AuthoritativeSetWeightsContextV2(
        substrate=substrate,
        wallet=wallet,
        weight_authorization_id=sha256_json(
            {"epoch_id": epoch_id, "kind": "sdk-weight-authorization"}
        ),
        weight_submission_event_hash=submission_event_hash,
        expected_era_period=8,
    ) as context:
        mechanism.get_encrypted_commit_v2(
            uids=uids,
            weights=weights_u16,
            version_key=10005000,
            last_epoch_block=epoch_id * 360,
            pending_epoch_at=0,
            subnet_epoch_index=epoch_id,
            tempo=360,
            blocks_since_last_step=22,
            current_block=epoch_id * 360 + 22,
            subnet_reveal_period_epochs=1,
            block_time=12.0,
            hotkey=wallet.hotkey.public_key,
        )
        signed = substrate.create_signed_extrinsic(
            call=object(),
            keypair=wallet.hotkey,
            era={"period": 8},
            nonce=None,
        )
    commit_requests = [
        request for kind, request in client.requests if kind == "commit"
    ]
    extrinsic_requests = [
        request for kind, request in client.requests if kind == "extrinsic"
    ]
    if (
        len(commit_requests) != 1
        or len(extrinsic_requests) != 1
        or commit_requests[0]["uids"] != uids
        or commit_requests[0]["weights_u16"] != weights_u16
        or len(context.extrinsic_signature_results) != 1
    ):
        raise RuntimeError("production SDK signing bridge evidence differs")
    return {
        "verified": True,
        "commit_request_hash": sha256_json(commit_requests[0]),
        "extrinsic_request_hash": sha256_json(extrinsic_requests[0]),
        "signature_hex": bytes(signed.signature).hex(),
    }


def _recompose_candidate_bundle(
    *,
    epoch_fixture: SanitizedWeightFixture,
    bundle: Mapping[str, Any],
    epoch_id: int,
) -> dict[str, Any]:
    binding_receipt = next(
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt["purpose"] == "validator.hotkey_signature.v2"
    )
    weight_boot_for_handoff = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    enclave_graph = build_receipt_graph(
        root_receipt_hash=binding_receipt["parent_receipt_hashes"][0],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            receipt
            for receipt in bundle["receipt_graph"]["receipts"]
            if receipt["receipt_hash"] != binding_receipt["receipt_hash"]
        ],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
        host_operations=bundle["receipt_graph"]["host_operations"],
    )
    return build_authoritative_weight_bundle_v2(
        enclave_response={
            "weight_snapshot": bundle["weight_snapshot"],
            "weight_result": bundle["weight_result"],
            "weights_signature": bundle["weights_signature"],
            "receipt_graph": enclave_graph,
            "boot_identity": weight_boot_for_handoff,
            "weight_authorization_id": sha256_json(
                {"epoch_id": epoch_id, "kind": "local-authorization"}
            ),
            "source_artifacts": [],
        },
        validator_hotkey=bundle["validator_hotkey"],
        binding_message=bundle["binding_message"],
        binding_signature_result={
            "purpose": "validator.gateway_binding.v2",
            "validator_hotkey": bundle["validator_hotkey"],
            "signature": bundle["validator_hotkey_signature"],
            "receipt": binding_receipt,
        },
    )


def _load_production_allocation(
    path: Path | None,
    *,
    candidate_sha: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("production allocation override is unreadable") from exc
    expected_fields = {
        "schema_version",
        "candidate_sha",
        "source_epoch",
        "root_receipt_hash",
        "handoff_hash",
        "allocation_hash",
        "allocation_doc",
    }
    if (
        not isinstance(document, Mapping)
        or set(document) != expected_fields
        or document.get("schema_version")
        != "leadpoet.rehearsal_production_allocation.v1"
        or document.get("candidate_sha") != candidate_sha
    ):
        raise RuntimeError("production allocation override identity differs")
    try:
        source_epoch = int(document["source_epoch"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError("production allocation source epoch is invalid") from exc
    allocation_doc = document.get("allocation_doc")
    allocation_hash = str(document.get("allocation_hash") or "")
    hash_payload = (
        {
            key: value
            for key, value in dict(allocation_doc).items()
            if key != "allocation_hash"
        }
        if isinstance(allocation_doc, Mapping)
        else {}
    )
    hashes = (
        allocation_hash,
        str(document.get("root_receipt_hash") or ""),
        str(document.get("handoff_hash") or ""),
    )
    if (
        source_epoch <= 0
        or not isinstance(allocation_doc, Mapping)
        or any(
            len(value) != 71
            or not value.startswith("sha256:")
            or any(character not in "0123456789abcdef" for character in value[7:])
            for value in hashes
        )
        or allocation_doc.get("allocation_hash") != allocation_hash
        or sha256_json(hash_payload) != allocation_hash
    ):
        raise RuntimeError("production allocation override hash differs")
    return dict(document)


def _run_independent_epoch_diagnostics(
    *,
    candidate_sha: str,
    epoch_id: int,
    stages: list[dict[str, Any]],
    production_allocation: Mapping[str, Any] | None = None,
) -> None:
    """Exercise independent downstream contracts before the joined epoch."""

    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
        production_allocation_doc=(
            production_allocation.get("allocation_doc")
            if production_allocation is not None
            else None
        ),
    )
    bundle_passed, bundle = _run_workflow_stage(
        stage="diagnostic:candidate-bundle-generation",
        action=epoch_fixture.bundle,
        stages=stages,
    )
    dependent_stages = (
        "diagnostic:host-bundle-composition",
        "diagnostic:primary-bundle-verification",
        "diagnostic:auditor-bundle-verification",
        "diagnostic:primary-auditor-vector-equality",
        "diagnostic:sdk-signing-bridge",
    )
    if not bundle_passed:
        for stage in dependent_stages:
            _mark_workflow_stage_unexercised(
                stage=stage,
                blocked_by=["diagnostic:candidate-bundle-generation"],
                stages=stages,
            )
        return

    _run_workflow_stage(
        stage="diagnostic:host-bundle-composition",
        action=lambda: _require_equal(
            _recompose_candidate_bundle(
                epoch_fixture=epoch_fixture,
                bundle=bundle,
                epoch_id=epoch_id,
            ),
            bundle,
            "production host bundle composition differs from canonical fixture",
        ),
        stages=stages,
    )
    primary_passed, primary = _run_workflow_stage(
        stage="diagnostic:primary-bundle-verification",
        action=lambda: validate_published_weight_bundle_v2(bundle),
        stages=stages,
    )
    auditor_passed, auditor = _run_workflow_stage(
        stage="diagnostic:auditor-bundle-verification",
        action=lambda: verify_attested_weight_bundle_v2(
            bundle,
            identity_cache=epoch_fixture.identity_cache(bundle),
            boot_verifier=lambda _boot, expected_pcr0=None: {
                "verified": True,
                "pcr0": expected_pcr0,
                "boundary": "local_nitro_attestation",
            },
        ),
        stages=stages,
    )
    if primary_passed and auditor_passed:
        _run_workflow_stage(
            stage="diagnostic:primary-auditor-vector-equality",
            action=lambda: _require_equal(
                {
                    "uids": list(primary["uids"]),
                    "weights_u16": list(primary["weights_u16"]),
                },
                {
                    "uids": list(auditor["uids"]),
                    "weights_u16": list(auditor["weights_u16"]),
                },
                "primary and auditor canonical vectors differ",
            ),
            stages=stages,
        )
    else:
        blocked_by = []
        if not primary_passed:
            blocked_by.append("diagnostic:primary-bundle-verification")
        if not auditor_passed:
            blocked_by.append("diagnostic:auditor-bundle-verification")
        _mark_workflow_stage_unexercised(
            stage="diagnostic:primary-auditor-vector-equality",
            blocked_by=blocked_by,
            stages=stages,
        )
    _run_workflow_stage(
        stage="diagnostic:sdk-signing-bridge",
        action=lambda: _exercise_sdk_bridge(
            epoch_id=epoch_id,
            uids=[
                int(value)
                for value in bundle["weight_result"]["sparse_uids"]
            ],
            weights_u16=[
                int(value)
                for value in bundle["weight_result"]["sparse_weights_u16"]
            ],
            submission_event_hash=sha256_json(
                {"epoch_id": epoch_id, "kind": "diagnostic-publication"}
            ),
        ),
        stages=stages,
    )


def _run_epoch(
    *,
    services: LocalBoundaryServices,
    fixture: Mapping[str, Any],
    candidate_sha: str,
    epoch_id: int,
    production_allocation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    epoch_fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
        production_allocation_doc=(
            production_allocation.get("allocation_doc")
            if production_allocation is not None
            else None
        ),
    )
    coordinator_key = epoch_fixture.coordinator_key
    weight_key = epoch_fixture.weight_key
    bundle = epoch_fixture.bundle()
    assembled_bundle = _recompose_candidate_bundle(
        epoch_fixture=epoch_fixture,
        bundle=bundle,
        epoch_id=epoch_id,
    )
    if assembled_bundle != bundle:
        raise RuntimeError(
            "production host bundle composition differs from canonical fixture"
        )
    verified_bundle = validate_published_weight_bundle_v2(bundle)
    identity_cache = epoch_fixture.identity_cache(bundle)
    auditor_bundle = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=identity_cache,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    primary_vector = {
        "uids": list(verified_bundle["uids"]),
        "weights_u16": list(verified_bundle["weights_u16"]),
    }
    auditor_vector = {
        "uids": list(auditor_bundle["uids"]),
        "weights_u16": list(auditor_bundle["weights_u16"]),
    }
    if primary_vector != auditor_vector:
        raise RuntimeError("primary and auditor canonical vectors differ")

    persisted_bundle = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "published_weight_bundle_v2",
            "epoch_id": epoch_id,
            "body": bundle,
        },
    )
    coordinator_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "gateway_coordinator"
    )
    weight_boot = next(
        identity
        for identity in bundle["receipt_graph"]["boot_identities"]
        if identity["physical_role"] == "validator_weights"
    )
    publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "durable_readback_hash": persisted_bundle["evidence_hash"],
        "transparency_event_hash": sha256_json(
            {"epoch_id": epoch_id, "kind": "transparency"}
        ),
    }
    publication_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="gateway_coordinator",
        purpose="gateway.weights.publication.v2",
        job_id=f"weight-publication-{epoch_id}",
        private_key=coordinator_key,
        boot=coordinator_boot,
        config_hash=str(coordinator_boot["config_hash"]),
        input_root=sha256_json({"publication": "input", "epoch_id": epoch_id}),
        output_root=sha256_json(publication_doc),
        parents=[verified_bundle["root_receipt_hash"]],
        sequence=200,
    )
    publication_graph = build_receipt_graph(
        root_receipt_hash=publication_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=bundle["receipt_graph"]["receipts"] + [publication_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"],
    )
    submission_event_hash = sha256_json(
        {
            "bundle_hash": verified_bundle["bundle_hash"],
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "transparency_event_hash": publication_doc["transparency_event_hash"],
            "durable_readback_hash": publication_doc["durable_readback_hash"],
        }
    )
    sdk_bridge = _exercise_sdk_bridge(
        epoch_id=epoch_id,
        uids=primary_vector["uids"],
        weights_u16=primary_vector["weights_u16"],
        submission_event_hash=submission_event_hash,
    )

    profile_manifest = json.loads(
        (
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ).read_text(encoding="utf-8")
    )
    profile = next(
        item
        for item in chain_signing_profiles(profile_manifest)
        if int(item["spec_version"])
        == int(profile_manifest["spec_version"])
    )
    seed = hashlib.sha256(
        b"hotkey-seed:" + candidate_sha.encode("ascii")
    ).digest()
    sr25519 = _Sr25519Backend()
    public_key, secret_key = sr25519.pair_from_seed(seed)
    commitment = hashlib.sha512(
        b"timelocked:" + epoch_id.to_bytes(8, "big") + _canonical(primary_vector)
    ).digest()
    block = int(verified_bundle["block"])
    authorization = build_weight_extrinsic_authorization_v2(
        profile=profile,
        validator_hotkey=verified_bundle["validator_hotkey"],
        hotkey_public_key_hex=public_key.hex(),
        epoch_id=epoch_id,
        netuid=int(verified_bundle["netuid"]),
        subnet_epoch_index=epoch_id,
        weight_receipt_hash=verified_bundle["weight_receipt_hash"],
        weight_submission_event_hash=submission_event_hash,
        weights_hash=verified_bundle["weights_hash"],
        sparse_uids=primary_vector["uids"],
        sparse_weights_u16=primary_vector["weights_u16"],
        commitment=commitment,
        reveal_round=epoch_id + 1,
        era_current=block,
        nonce=epoch_id,
        block_hash=hashlib.sha256(f"block:{block}".encode("ascii")).hexdigest(),
    )
    signature = sr25519.sign(
        (public_key, secret_key),
        bytes.fromhex(authorization["signed_message_hex"]),
    )
    signed_extrinsic = encode_signed_extrinsic_v2(
        hotkey_public_key_hex=public_key.hex(),
        signature_hex=signature.hex(),
        era_period=int(authorization["era_period"]),
        era_current=int(authorization["era_current"]),
        nonce=int(authorization["nonce"]),
        call_data_hex=str(authorization["call_data_hex"]),
    )
    extrinsic_hash = signed_extrinsic_hash_v2(signed_extrinsic)
    services.request(
        "POST",
        "/chain/submit_extrinsic",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "extrinsic_hex": signed_extrinsic.hex(),
            "bundle_hash": verified_bundle["bundle_hash"],
            "weights_hash": verified_bundle["weights_hash"],
            **primary_vector,
        },
    )
    finalized = services.request(
        "POST",
        "/chain/finalize",
        {
            "epoch_id": epoch_id,
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": block + 1,
        },
    )

    extrinsic_output = {
        "schema_version": "leadpoet.weight_extrinsic_signature.v2",
        "authorization_hash": authorization["authorization_hash"],
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "signature": signature.hex(),
        "extrinsic_hash": extrinsic_hash,
    }
    extrinsic_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.set_weights_extrinsic.v2",
        job_id=f"set-weights-{epoch_id}",
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=authorization["authorization_hash"],
        output_root=sha256_json(extrinsic_output),
        parents=[verified_bundle["weight_receipt_hash"]],
        sequence=201,
    )
    finalization_job = f"weight-finalization-{epoch_id}"
    attempts = [
        epoch_fixture.source_attempt(
            category="weight-finalization",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=300,
            provider_id="bittensor_chain",
            host="entrypoint-finney.opentensor.ai",
            method="POST",
        ),
        epoch_fixture.source_attempt(
            category="weight-finalization-archive",
            job_id=finalization_job,
            purpose="validator.weights.finalized.v2",
            sequence=301,
            provider_id="bittensor_archive",
            host="archive.chain.opentensor.ai",
            method="POST",
        ),
    ]
    finalization_doc = {
        "schema_version": "leadpoet.weight_finalization.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "netuid": int(verified_bundle["netuid"]),
        "epoch_id": epoch_id,
        "weights_hash": verified_bundle["weights_hash"],
        "weight_receipt_hash": verified_bundle["weight_receipt_hash"],
        "weight_submission_event_hash": submission_event_hash,
        "extrinsic_authorization": authorization,
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "extrinsic_signature": signature.hex(),
        "extrinsic_receipt_hash": extrinsic_receipt["receipt_hash"],
        "extrinsic_hash": extrinsic_hash,
        "finalized_block": int(finalized["finalized_block"]),
        "finalized_block_hash": str(finalized["finalized_block_hash"]),
        "state_transition_hash": str(finalized["state_transition_hash"]),
    }
    final_receipt = _receipt(
        epoch_id=epoch_id,
        candidate_sha=candidate_sha,
        role="validator_weights",
        purpose="validator.weights.finalized.v2",
        job_id=finalization_job,
        private_key=weight_key,
        boot=weight_boot,
        config_hash=str(weight_boot["config_hash"]),
        input_root=sha256_json(
            {
                "weight_submission_event_hash": submission_event_hash,
                "extrinsic_receipt_hashes": [extrinsic_receipt["receipt_hash"]],
            }
        ),
        output_root=sha256_json(finalization_doc),
        parents=[extrinsic_receipt["receipt_hash"]],
        sequence=202,
        transport_attempts=attempts,
    )
    final_graph = build_receipt_graph(
        root_receipt_hash=final_receipt["receipt_hash"],
        boot_identities=bundle["receipt_graph"]["boot_identities"],
        receipts=[
            item
            for item in bundle["receipt_graph"]["receipts"]
            if item["purpose"] != "validator.hotkey_signature.v2"
        ]
        + [extrinsic_receipt, final_receipt],
        transport_attempts=bundle["receipt_graph"]["transport_attempts"]
        + attempts,
    )
    finalization_submission = {
        "schema_version": "leadpoet.weight_finalization_submission.v2",
        "validator_hotkey": verified_bundle["validator_hotkey"],
        "weight_submission_event_hash": submission_event_hash,
        "finalization": finalization_doc,
        "receipt_graph": final_graph,
    }
    verified_finalization = validate_weight_finalization_submission_v2(
        finalization_submission,
        chain_signing_profile=profile_manifest,
    )
    finalization_event_hash = sha256_json(
        {
            "weight_submission_event_hash": submission_event_hash,
            "bundle_hash": verified_bundle["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": authorization["authorization_hash"],
            "extrinsic_hash": extrinsic_hash,
            "finalized_block": finalization_doc["finalized_block"],
            "finalized_block_hash": finalization_doc["finalized_block_hash"],
            "state_transition_hash": finalization_doc["state_transition_hash"],
        }
    )
    authority = {
        "schema_version": "leadpoet.published_weight_authority.v2",
        "bundle": bundle,
        "publication": {
            "weight_submission_event_hash": submission_event_hash,
            "publication_receipt_hash": publication_receipt["receipt_hash"],
            "publication_doc": publication_doc,
            "receipt_graph": publication_graph,
        },
        "finalization": {
            "weight_finalization_event_hash": finalization_event_hash,
            "submission": finalization_submission,
        },
    }
    auditor_authority = verify_attested_weight_authority_v2(
        authority,
        identity_cache=identity_cache,
        chain_signing_profile=profile_manifest,
        boot_verifier=lambda _boot, expected_pcr0=None: {
            "verified": True,
            "pcr0": expected_pcr0,
            "boundary": "local_nitro_attestation",
        },
    )
    if auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError("auditor finalization differs from local chain")
    production_auditor_authority = _run_production_auditor(
        authority=authority,
        identity_cache=identity_cache,
        epoch_id=epoch_id,
        block=int(verified_bundle["block"]),
    )
    if production_auditor_authority["extrinsic_hash"] != extrinsic_hash:
        raise RuntimeError(
            "production auditor finalization differs from local chain"
        )

    reveal = services.request(
        "POST",
        "/chain/reveal",
        {"epoch_id": epoch_id, **primary_vector},
    )
    last_update = services.request(
        "GET", f"/chain/epoch/{epoch_id}/last_update"
    )
    revealed = services.request("GET", f"/chain/epoch/{epoch_id}/reveal")
    if revealed["reveal"]["vector_hash"] != reveal["vector_hash"]:
        raise RuntimeError("revealed vector readback differs")
    result = {
        "epoch_id": epoch_id,
        "pcr0": weight_boot["pcr0"],
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "publication_receipt_hash": publication_receipt["receipt_hash"],
        "finalization_receipt_hash": verified_finalization[
            "finalization_receipt_hash"
        ],
        "receipt_ancestry_verified": True,
        "canonical_vector_hash": sha256_json(primary_vector),
        "canonical_vector_equal": True,
        "weights_hash": verified_bundle["weights_hash"],
        "extrinsic_authorization_hash": authorization["authorization_hash"],
        "signed_extrinsic_hash": extrinsic_hash,
        "sdk_bridge_verified": sdk_bridge["verified"],
        "sdk_commit_request_hash": sdk_bridge["commit_request_hash"],
        "sdk_extrinsic_request_hash": sdk_bridge["extrinsic_request_hash"],
        "finalized_block": finalized["finalized_block"],
        "last_update": last_update["last_update"],
        "reveal_vector_hash": reveal["vector_hash"],
        "auditor_verified": True,
        "auditor_runtime_verified": True,
    }
    if production_allocation is not None:
        result["production_allocation_hash"] = production_allocation[
            "allocation_hash"
        ]
        result["production_allocation_handoff_hash"] = production_allocation[
            "handoff_hash"
        ]
        result["production_allocation_source_epoch"] = int(
            production_allocation["source_epoch"]
        )
    return result


def _exercise_fault(
    services: LocalBoundaryServices,
    *,
    fault: str,
    ordinal: int,
) -> dict[str, Any]:
    services.inject(fault)
    status = {
        "http_400": 400,
        "http_403": 403,
        "http_429": 429,
        "http_500": 500,
        "duplicate_response": 409,
        "malformed_json": 502,
        "partial_body": 502,
        "unexpected_eof": 502,
        "timeout": 504,
    }.get(fault, 503)
    response = services.request(
        "POST",
        "/database/insert",
        {
            "kind": "fault_probe",
            "epoch_id": -1,
            "body": {"fault": fault, "ordinal": ordinal},
        },
        expected_status=status,
    )
    if response.get("fault") != fault:
        raise RuntimeError(f"fault response differs for {fault}")
    return {"fault": fault, "status": "fail_closed"}


def _exercise_concurrency(services: LocalBoundaryServices) -> int:
    def insert(ordinal: int) -> str:
        response = services.request(
            "POST",
            "/database/insert",
            {
                "kind": "concurrency_probe",
                "epoch_id": -2,
                "body": {"caller": ordinal},
            },
        )
        return str(response["evidence_hash"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        hashes = list(pool.map(insert, range(32)))
    if len(set(hashes)) != 32:
        raise RuntimeError("concurrent durable writes were not isolated")
    return len(hashes)


async def _exercise_chain_settlement_state_space_async() -> dict[str, Any]:
    """Exercise every prefix topology through the production bootstrap gate."""

    from gateway.research_lab import champion_settlement_v2 as settlement
    from gateway.research_lab import store
    from gateway.research_lab import v2_authority

    netuid = 71
    activation_epoch = 40_000
    target_epoch = activation_epoch + 4
    source_bundle_hash = sha256_json(
        {"kind": "rehearsal-settlement-source", "epoch": activation_epoch}
    )
    activation = {
        "netuid": netuid,
        "schema_version": (
            "leadpoet.research_lab_chain_realized_settlement_activation.v1"
        ),
        "first_epoch_id": activation_epoch,
        "source_bundle_hash": source_bundle_hash,
        "source_bundle_epoch_id": activation_epoch,
        "source_finalized_block": 8_700_039,
    }
    state: dict[str, Any] = {"rows": []}
    validated_ranges: list[tuple[int, int]] = []

    async def select_many(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
            return [dict(activation)]
        if table == settlement.FINALIZED_ALLOCATION_VIEW_V2:
            return [
                {
                    "bundle_hash": source_bundle_hash,
                    "netuid": netuid,
                    "epoch_id": activation_epoch,
                    "finalized_block": activation["source_finalized_block"],
                    "finalization_receipt_hash": sha256_json(
                        {"kind": "finalization", "epoch": activation_epoch}
                    ),
                }
            ]
        raise AssertionError(f"unexpected settlement select_many table: {table}")

    async def select_all(table: str, **_kwargs: Any) -> list[dict[str, Any]]:
        if table != settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
            raise AssertionError(
                f"unexpected settlement select_all table: {table}"
            )
        return [dict(row) for row in state["rows"]]

    async def load_chain_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if netuid != 71 or start_epoch != activation_epoch:
            raise AssertionError("settlement prefix validation range differs")
        validated_ranges.append((start_epoch, end_epoch))
        return [
            {"epoch": epoch}
            for epoch in range(start_epoch, end_epoch + 1)
        ]

    async def load_finalized_history(
        *,
        netuid: int,
        start_epoch: int,
        end_epoch: int,
    ) -> list[dict[str, Any]]:
        if (
            netuid != 71
            or start_epoch != activation_epoch
            or end_epoch != target_epoch
        ):
            raise AssertionError("finalized source validation range differs")
        return [
            {
                "epoch": activation_epoch,
                "finalized_bundle_hashes": [source_bundle_hash],
            }
        ]

    originals = (
        store.select_many,
        store.select_all,
        settlement.load_chain_realized_allocation_history_v1,
        settlement.load_finalized_allocation_history_v2,
    )
    store.select_many = select_many
    store.select_all = select_all
    settlement.load_chain_realized_allocation_history_v1 = load_chain_history
    settlement.load_finalized_allocation_history_v2 = load_finalized_history
    try:
        accepted: list[dict[str, Any]] = []
        total_epochs = target_epoch - activation_epoch + 1
        for prefix_length in range(total_epochs + 1):
            validated_ranges.clear()
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "settlement", "epoch": epoch}
                    ),
                }
                for epoch in range(
                    activation_epoch,
                    activation_epoch + prefix_length,
                )
            ]
            result = (
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            )
            expected_status = (
                "pristine_bootstrap_pending"
                if prefix_length == 0
                else "resumable_bootstrap_pending"
            )
            if (
                result["status"] != expected_status
                or result["backlog_epoch_count"]
                != total_epochs - prefix_length
                or result["validated_chain_realized_epochs"]
                != [
                    activation_epoch + offset
                    for offset in range(prefix_length)
                ]
                or (
                    prefix_length > 0
                    and validated_ranges
                    != [
                        (
                            activation_epoch,
                            activation_epoch + prefix_length - 1,
                        )
                    ]
                )
                or (prefix_length == 0 and validated_ranges)
            ):
                raise RuntimeError(
                    "chain settlement prefix behavior differs from contract"
                )
            accepted.append(
                {
                    "prefix_length": prefix_length,
                    "status": result["status"],
                    "backlog_epoch_count": result["backlog_epoch_count"],
                }
            )

        invalid_states = {
            "duplicate": [activation_epoch, activation_epoch],
            "gap": [activation_epoch, activation_epoch + 2],
            "missing-first": [activation_epoch + 1],
            "ahead": list(range(activation_epoch, target_epoch + 2)),
        }
        rejected: list[str] = []
        for name, epochs in invalid_states.items():
            state["rows"] = [
                {
                    "netuid": netuid,
                    "epoch_id": epoch,
                    "settlement_hash": sha256_json(
                        {"kind": "invalid-settlement", "name": name, "epoch": epoch}
                    ),
                }
                for epoch in epochs
            ]
            try:
                await settlement.validate_chain_realized_settlement_bootstrap_v1(
                    netuid=netuid,
                    target_epoch=target_epoch,
                    maximum_backlog=total_epochs,
                )
            except settlement.ChampionSettlementV2Error:
                rejected.append(name)
            else:
                raise RuntimeError(
                    f"invalid settlement topology was accepted: {name}"
                )

        state["rows"] = []
        try:
            await settlement.validate_chain_realized_settlement_bootstrap_v1(
                netuid=netuid,
                target_epoch=target_epoch,
                maximum_backlog=total_epochs - 1,
            )
        except settlement.ChampionSettlementV2Error:
            rejected.append("backlog-exceeds-policy")
        else:
            raise RuntimeError("excessive settlement backlog was accepted")

        observation_calls: list[dict[str, Any]] = []

        class ObservationReached(RuntimeError):
            pass

        async def capture_observation(**kwargs: Any) -> dict[str, Any]:
            observation_calls.append(kwargs)
            raise ObservationReached

        try:
            await v2_authority.settle_chain_realized_epoch_v1(
                epoch_id=target_epoch,
                netuid=netuid,
                settlement_attempt=4,
                execute=capture_observation,
            )
        except ObservationReached:
            pass
        if (
            len(observation_calls) != 1
            or observation_calls[0].get("operation")
            != v2_authority.OP_OBSERVE_CHAIN_REALIZED_WEIGHTS_V1
            or observation_calls[0].get("sequence") != 8
        ):
            raise RuntimeError(
                "chain settlement retry replayed the prior observation identity"
            )

        async def load_durable_attempt_history(
            table: str,
            **kwargs: Any,
        ) -> list[dict[str, Any]]:
            if (
                table != "research_lab_attested_execution_receipts_v2"
                or kwargs.get("columns")
                != "purpose,sequence,receipt_status,issued_at"
                or kwargs.get("order_by") != (("sequence", True),)
                or kwargs.get("limit") != 1
            ):
                raise RuntimeError(
                    "chain settlement durable retry history query differs"
                )
            return [
                {
                    "sequence": 7,
                    "receipt_status": "succeeded",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                }
            ]

        durable_retry_attempt = (
            await v2_authority._resolve_chain_settlement_attempt_v1(
                epoch_id=target_epoch,
                requested_attempt=0,
                load_attempt_history=load_durable_attempt_history,
            )
        )
        if durable_retry_attempt != 4:
            raise RuntimeError(
                "chain settlement retry identity did not survive restart"
            )

        async def load_recent_failure(*_args: Any, **_kwargs: Any):
            return [
                {
                    "sequence": 7,
                    "receipt_status": "failed",
                    "issued_at": datetime.now(timezone.utc).isoformat(),
                }
            ]

        try:
            await v2_authority._resolve_chain_settlement_attempt_v1(
                epoch_id=target_epoch,
                requested_attempt=0,
                load_attempt_history=load_recent_failure,
            )
        except v2_authority.ResearchLabV2AuthorityError as exc:
            if "retry is cooling down" not in str(exc):
                raise
        else:
            raise RuntimeError("recent chain settlement failure was retried")

        propagated_attempts: list[tuple[int, int]] = []

        async def load_retry_frontier(
            table: str,
            **_kwargs: Any,
        ) -> list[dict[str, Any]]:
            if table == settlement.CHAIN_REALIZED_SETTLEMENT_ACTIVATION_TABLE_V1:
                return [dict(activation)]
            if table == settlement.CHAIN_REALIZED_EPOCH_SETTLEMENT_TABLE_V1:
                return []
            raise AssertionError(f"unexpected retry-frontier table: {table}")

        async def capture_settlement_attempt(
            *,
            epoch_id: int,
            settlement_attempt: int,
            **_kwargs: Any,
        ) -> dict[str, Any]:
            propagated_attempts.append((epoch_id, settlement_attempt))
            return {"epoch_id": epoch_id}

        await v2_authority.ensure_chain_realized_settlements_v1(
            epoch_id=activation_epoch + 2,
            netuid=netuid,
            settlement_attempt=4,
            load_latest=load_retry_frontier,
            settle=capture_settlement_attempt,
        )
        if propagated_attempts != [
            (activation_epoch, 4),
            (activation_epoch + 1, 4),
        ]:
            raise RuntimeError(
                "allocation retry generation did not reach settlement backlog"
            )
        return {
            "accepted_prefixes": accepted,
            "accepted_count": len(accepted),
            "rejected_state_classes": sorted(rejected),
            "retry_observation_sequence": 8,
            "retry_attempts_propagated": len(propagated_attempts),
            "durable_retry_attempt": durable_retry_attempt,
            "durable_retry_cooldown": True,
        }
    finally:
        (
            store.select_many,
            store.select_all,
            settlement.load_chain_realized_allocation_history_v1,
            settlement.load_finalized_allocation_history_v2,
        ) = originals


def _exercise_chain_settlement_state_space() -> dict[str, Any]:
    return asyncio.run(_exercise_chain_settlement_state_space_async())


def _exercise_historical_metagraph_layouts() -> dict[str, Any]:
    """Exercise every candidate-declared archive layout through production."""

    from fixture_contract import (
        load_rehearsal_metagraph_account_ids,
        load_rehearsal_metagraph_hotkeys,
    )
    from gateway.tee.coordinator_chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_URL,
        CoordinatorChainSourceV2,
    )
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from leadpoet_canonical.attested_v2 import (
        build_transport_attempt,
        sha256_bytes,
    )
    from leadpoet_canonical.chain_source_v2 import (
        CHAIN_ARCHIVE_ENDPOINT_HOST,
        CHAIN_RPC_METHOD,
        ChainSourceV2Error,
        chain_source_policy_document,
        chain_source_policy_hash,
        encode_selective_metagraph_params,
        weights_storage_key,
    )

    policy = chain_source_policy_document()
    layouts = tuple(
        int(value) for value in policy["selective_result_last_fields"]
    )
    if (
        not layouts
        or tuple(sorted(set(layouts))) != layouts
        or any(value <= 52 for value in layouts)
    ):
        raise RuntimeError(
            "candidate chain-source result layouts are invalid"
        )
    account_ids = load_rehearsal_metagraph_account_ids(SOURCE_ROOT)
    hotkeys = load_rehearsal_metagraph_hotkeys(SOURCE_ROOT)
    validator_hotkey = hotkeys[0]
    cutover = json.loads(
        (
            SOURCE_ROOT
            / "config"
            / "stateful-epoch-cutover-sn71.json"
        ).read_text(encoding="utf-8")
    )
    netuid = int(cutover["netuid"])
    epoch_id = int(cutover["last_legacy_epoch_id"])
    target_block = (epoch_id + 1) * 360 - 1
    retry_hashes = {
        "bittensor_chain": sha256_json({"retry": "chain"}),
        "bittensor_archive": sha256_json({"retry": "archive"}),
        "coingecko": sha256_json({"retry": "coingecko"}),
    }
    def selective_fixture(last_field: int) -> str:
        if netuid < 1 << 6:
            compact_netuid = bytes((netuid << 2,))
        elif netuid < 1 << 14:
            compact_netuid = ((netuid << 2) | 1).to_bytes(2, "little")
        else:
            compact_netuid = ((netuid << 2) | 2).to_bytes(4, "little")
        encoded = bytearray(b"\x01" + compact_netuid)
        encoded.extend(b"\x00" * 4)
        encoded.extend(b"\x01" + account_ids[0])
        encoded.extend(b"\x00")
        encoded.extend(
            b"\x01"
            + ((target_block << 2) | 2).to_bytes(4, "little")
        )
        encoded.extend(b"\x00" * 44)
        encoded.extend(
            b"\x01"
            + bytes((len(account_ids) << 2,))
            + b"".join(account_ids)
        )
        encoded.extend(b"\x00" * (int(last_field) - 52))
        return "0x" + bytes(encoded).hex()

    class StrictArchiveBoundary:
        def __init__(self, *, last_field: int) -> None:
            self.last_field = int(last_field)
            self.calls: list[dict[str, Any]] = []

        def execute(
            self,
            request: Mapping[str, Any],
        ) -> dict[str, Any]:
            if (
                request.get("provider_id") != "bittensor_archive"
                or request.get("method") != "POST"
                or request.get("url") != CHAIN_ARCHIVE_ENDPOINT_URL
                or request.get("retry_policy_hash")
                != retry_hashes["bittensor_archive"]
            ):
                raise RuntimeError(
                    "historical layout probe crossed an undeclared boundary"
                )
            request_body = base64.b64decode(
                str(request["body_b64"]),
                validate=True,
            )
            rpc = json.loads(request_body)
            if set(rpc) != {"jsonrpc", "id", "method", "params"} or (
                rpc.get("jsonrpc") != "2.0"
            ):
                raise RuntimeError(
                    "historical layout probe received malformed JSON-RPC"
                )
            method = rpc.get("method")
            call_number = len(self.calls) + 1
            self.calls.append(
                {
                    "method": method,
                    "params": rpc.get("params"),
                }
            )
            if method == "chain_getFinalizedHead":
                if rpc.get("params") != []:
                    raise RuntimeError(
                        "historical finalized-head request differs"
                    )
                value: Any = "0x" + "a" * 64
            elif method == "chain_getBlockHash":
                if rpc.get("params") != [target_block]:
                    raise RuntimeError(
                        "historical layout probe requested another block"
                    )
                value = "0x" + "b" * 64
            elif method == "chain_getHeader":
                at_hash = str((rpc.get("params") or [""])[0])
                is_target = at_hash == "0x" + "b" * 64
                if at_hash not in {
                    "0x" + "a" * 64,
                    "0x" + "b" * 64,
                }:
                    raise RuntimeError(
                        "historical layout probe requested another hash"
                    )
                value = {
                    "number": hex(
                        target_block if is_target else target_block + 20
                    ),
                    "stateRoot": "0x" + "c" * 64,
                    "parentHash": "0x" + "d" * 64,
                    "extrinsicsRoot": "0x" + "e" * 64,
                    "digest": {"logs": []},
                }
            elif method == "state_call":
                if rpc.get("params") != [
                    CHAIN_RPC_METHOD,
                    encode_selective_metagraph_params(netuid=netuid),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical selective metagraph request differs"
                    )
                value = selective_fixture(self.last_field)
            elif method == "state_getStorage":
                if rpc.get("params") != [
                    weights_storage_key(
                        netuid=netuid,
                        validator_uid=0,
                    ),
                    "0x" + "b" * 64,
                ]:
                    raise RuntimeError(
                        "historical weight-storage request differs"
                    )
                value = "0x" + (
                    b"\x08"
                    + (1).to_bytes(2, "little")
                    + (1000).to_bytes(2, "little")
                    + (4).to_bytes(2, "little")
                    + (2000).to_bytes(2, "little")
                ).hex()
            else:
                raise RuntimeError(
                    "historical layout probe received an unknown RPC"
                )
            response_body = json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": rpc.get("id"),
                    "result": value,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            artifact_hash = sha256_json(
                {
                    "call": call_number,
                    "layout": self.last_field,
                    "method": method,
                }
            )
            attempt = build_transport_attempt(
                request_id=("%032x" % call_number),
                logical_operation_id=str(
                    request["logical_operation_id"]
                ),
                job_id=str(request["job_id"]),
                purpose=str(request["purpose"]),
                provider_id="bittensor_archive",
                attempt_number=int(request["attempt_number"]),
                method="POST",
                destination_host=CHAIN_ARCHIVE_ENDPOINT_HOST,
                destination_port=443,
                path_hash=sha256_json({"path": "/"}),
                nonsecret_headers_hash=sha256_json(
                    {"headers": "application/json"}
                ),
                body_hash=sha256_bytes(request_body),
                credential_ref_hash=sha256_json(
                    {"credential": "public-archive"}
                ),
                retry_policy_hash=str(request["retry_policy_hash"]),
                timeout_ms=int(request["timeout_ms"]),
                started_at=NOW,
                terminal_status="authenticated_response",
                http_status=200,
                response_hash=sha256_bytes(response_body),
                request_artifact_hash=artifact_hash,
                response_artifact_hash=sha256_bytes(response_body),
                tls_peer_chain_hash=sha256_json(
                    {"tls": "archive-rehearsal"}
                ),
                tls_protocol="TLSv1.3",
                failure_code=None,
                completed_at=NOW,
            )
            return {
                "terminal_status": "authenticated_response",
                "http_status": 200,
                "body_b64": base64.b64encode(response_body).decode(
                    "ascii"
                ),
                "transport_attempt": attempt,
            }

    def execute_layout(last_field: int) -> tuple[dict[str, Any], int]:
        boundary = StrictArchiveBoundary(last_field=last_field)
        source = CoordinatorChainSourceV2(
            execute_provider=boundary.execute,
            retry_policy_hashes=retry_hashes,
            epoch_authority={
                "mode": "stateful_v1",
                "cutover": cutover,
            },
            sleep=lambda _seconds: None,
        )
        context = ExecutionContextV2(
            job_id=f"rehearsal:historical-layout:{last_field}",
            purpose="research_lab.legacy_finalized_allocation.v2",
            epoch_id=epoch_id,
        )
        result = source.read_historical_finalized_weights(
            netuid=netuid,
            epoch_id=epoch_id,
            validator_hotkey=validator_hotkey,
            context=context,
        )
        return result, len(boundary.calls)

    accepted: list[int] = []
    call_counts: dict[str, int] = {}
    for last_field in layouts:
        result, call_count = execute_layout(last_field)
        if (
            result["target_block"] != target_block
            or result["validator_uid"] != 0
            or result["weights"] != [[1, 1000], [4, 2000]]
            or call_count != 6
        ):
            raise RuntimeError(
                "historical archive layout produced different authority"
            )
        accepted.append(last_field)
        call_counts[str(last_field)] = call_count

    rejected_layout = next(
        (
            value
            for value in range(53, max(layouts) + 1)
            if value not in layouts
        ),
        max(layouts) + 1,
    )
    try:
        execute_layout(rejected_layout)
    except ChainSourceV2Error:
        pass
    else:
        raise RuntimeError(
            "undeclared historical archive layout did not fail closed"
        )
    return {
        "policy_hash": chain_source_policy_hash(),
        "accepted_layouts": accepted,
        "rejected_layout": rejected_layout,
        "rpc_call_counts": call_counts,
    }


def _exercise_research_lab_allocation_conservation() -> dict[str, Any]:
    """Exercise the configured conservative and no-burn allocation modes."""

    from gateway.research_lab.config import ResearchLabGatewayConfig
    from leadpoet_verifier.economics import allocate_research_lab_epoch

    policy = ResearchLabGatewayConfig.from_env().reimbursement_policy_doc(
        enabled=True
    )
    policy_hash = sha256_json(policy)
    epoch = 30_000
    cap = Decimal(str(policy["research_lab_emission_percent"]))
    if (
        cap <= 0
        or policy.get("enable_conservative") is not True
        or policy.get("enable_champ_cap") is not True
        or Decimal(
            str(
                policy[
                    "reimbursement_max_cost_multiplier_with_champions"
                ]
            )
        )
        != Decimal("2")
    ):
        raise RuntimeError(
            "Research Lab default allocation policy differs from conservative V2"
        )
    no_burn_policy = dict(policy)
    no_burn_policy["enable_conservative"] = False
    no_burn_policy["enable_champ_cap"] = False

    def reimbursement(
        uid: int,
        compute_microusd: int,
    ) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "reimbursement-%d" % uid,
            "source_id": "reimbursement_schedule:rehearsal-%d" % uid,
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reimbursement_epochs"]),
            "target_reimbursement_microusd": compute_microusd,
            "eligible_compute_microusd": compute_microusd,
        }

    current = allocate_research_lab_epoch(
        epoch,
        no_burn_policy,
        [reimbursement(1, 1_000_000), reimbursement(2, 3_000_000)],
        [],
    )
    current_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in current["reimbursement_allocations"]
    }
    if (
        sum(current_paid.values()) != cap
        or current_paid[2] != current_paid[1] * Decimal("3")
        or Decimal(str(current["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "current reimbursements did not conserve the Lab cap by compute"
        )

    source_hash = sha256_json({"fixture": "historical-compute"})

    def fallback(uid: int, compute_microusd: int) -> dict[str, Any]:
        return {
            "uid": uid,
            "miner_hotkey": "fallback-%d" % uid,
            "source_id": "historical_compute_fallback:%064d" % uid,
            "island": "historical_compute",
            "status": "active",
            "target_reimbursement_microusd": compute_microusd,
            "fallback_window_start_epoch": epoch - 20,
            "fallback_window_end_epoch": epoch - 1,
            "source_allocation_epoch": epoch - 1,
            "source_allocation_hash": source_hash,
            "contribution_count": 1,
            "contribution_hash": sha256_json(
                {"uid": uid, "compute_microusd": compute_microusd}
            ),
        }

    historical = allocate_research_lab_epoch(
        epoch,
        no_burn_policy,
        [],
        [],
        fallback_reimbursement_obligations=[
            fallback(3, 1_000_000),
            fallback(4, 3_000_000),
        ],
    )
    historical_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in historical["reimbursement_allocations"]
    }
    if (
        sum(historical_paid.values()) != cap
        or historical_paid[4] != historical_paid[3] * Decimal("3")
        or historical.get("historical_compute_fallback_source_epoch")
        != epoch - 1
        or Decimal(str(historical["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "historical compute fallback did not conserve the Lab cap"
        )

    champions = [
        {
            "uid": 5,
            "miner_hotkey": "champion-5",
            "source_id": "champion_reward:rehearsal-5",
            "champion_reward_id": "champion_reward:rehearsal-5",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 1.0,
            "desired_alpha_percent": 7.0,
        },
        {
            "uid": 6,
            "miner_hotkey": "champion-6",
            "source_id": "champion_reward:rehearsal-6",
            "champion_reward_id": "champion_reward:rehearsal-6",
            "island": "generalist",
            "status": "active",
            "start_epoch": epoch,
            "epoch_count": int(policy["reward_epochs"]),
            "improvement_points": 2.0,
            "desired_alpha_percent": 14.0,
        },
    ]
    champion_allocation = allocate_research_lab_epoch(
        epoch,
        no_burn_policy,
        [],
        champions,
    )
    champion_paid = {
        int(row["uid"]): Decimal(str(row["paid_alpha_percent"]))
        for row in [
            *champion_allocation["champion_allocations"],
            *champion_allocation["queued_champion_allocations"],
        ]
    }
    if (
        sum(champion_paid.values()) != cap
        or champion_paid[6] != champion_paid[5] * Decimal("2")
        or Decimal(str(champion_allocation["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "champions did not split the remaining Lab cap by configured reward"
        )

    valuation_microusd = int(
        (
            Decimal(str(policy["usd_per_0_1_percent_epoch"]))
            * Decimal(1_000_000)
        ).to_integral_value()
    )
    capped = allocate_research_lab_epoch(
        epoch,
        no_burn_policy,
        [
            reimbursement(
                7,
                valuation_microusd * int(policy["reimbursement_epochs"]),
            )
        ],
        [champions[0]],
    )
    capped_reimbursement = Decimal(
        str(capped["reimbursement_allocations"][0]["paid_alpha_percent"])
    )
    if (
        capped_reimbursement != Decimal("0.2")
        or Decimal(str(capped["champion_alpha_percent"]))
        != cap - capped_reimbursement
        or Decimal(str(capped["unallocated_percent"])) != 0
    ):
        raise RuntimeError(
            "active-champion reimbursement cap or remainder differs"
        )

    conservative = allocate_research_lab_epoch(
        epoch,
        policy,
        [],
        [],
    )
    if (
        Decimal(str(conservative["unallocated_percent"])) != cap
        or conservative["reimbursement_allocations"]
        or conservative["champion_allocations"]
    ):
        raise RuntimeError(
            "conservative compatibility mode no longer preserves burn"
        )
    return {
        "policy_hash": policy_hash,
        "lab_cap_percent": float(cap),
        "current_reimbursement_alpha_percent": float(
            current["reimbursement_alpha_percent"]
        ),
        "historical_reimbursement_alpha_percent": float(
            historical["reimbursement_alpha_percent"]
        ),
        "champion_alpha_percent": float(
            champion_allocation["champion_alpha_percent"]
        ),
        "active_champion_reimbursement_alpha_percent": float(
            capped["reimbursement_alpha_percent"]
        ),
        "conservative_unallocated_percent": float(
            conservative["unallocated_percent"]
        ),
        "conserved": True,
    }


def _exercise_settlement_frontier_terminal_retirement() -> dict[str, Any]:
    """Reproduce and close the terminal-obligation frontier transition."""

    from gateway.tee.coordinator_allocation_source_v2 import (
        CoordinatorAllocationSourceV2,
        CoordinatorAllocationSourceV2Error,
    )
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from gateway.tee.reward_executor_v2 import (
        champion_reward_row_projection_v2,
        source_add_reward_row_projection_v2,
    )
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
        build_reward_settlement_checkpoint_v2,
        frontier_artifact_hashes_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_boot_identity_body,
        create_boot_identity,
    )

    champion = {
        "champion_reward_id": "champion_reward:sha256:" + "a" * 64,
        "score_bundle_id": "score-bundle-rehearsal",
        "candidate_id": "candidate-rehearsal",
        "run_id": "run-rehearsal",
        "miner_hotkey": "5ChampionRehearsal",
        "miner_uid": 10,
        "island": "generalist",
        "evaluation_epoch": 119,
        "start_epoch": 120,
        "epoch_count": 20,
        "improvement_points": 1.0,
        "threshold_points": 0.0,
        "desired_alpha_percent": 7.3,
        "input_hash": "sha256:" + "b" * 64,
        "anchored_hash": "sha256:" + "c" * 64,
        "current_reward_status": "paid",
    }
    source_add = {
        "reward_ref": "source_add_reward:" + "d" * 16,
        "adapter_id": "adapter-rehearsal",
        "miner_hotkey": "5SourceAddRehearsal",
        "leg": 1,
        "reward_kind": "source_acceptance",
        "alpha_percent": 1.0,
        "reward_epochs": 20,
        "start_epoch": 120,
        "current_reward_status": "stopped_forward",
        "trigger_evidence_doc": {
            "submission_id": "source_add_submission:abcd1234abcd1234"
        },
        "public_label": "Source acceptance",
        "desired_alpha_percent": 1.0,
        "epoch_count": 20,
    }
    champion_checkpoint = build_reward_settlement_checkpoint_v2(
        reward_kind="champion",
        source_id=champion["champion_reward_id"],
        obligation_hash=sha256_json(
            champion_reward_row_projection_v2(champion)
        ),
        start_epoch=120,
        epoch_count=20,
        desired_alpha_percent=7.3,
        applied_alpha_percent=30,
        realized_alpha_percent=30,
        excess_alpha_percent=0,
    )
    source_add_checkpoint = build_reward_settlement_checkpoint_v2(
        reward_kind="source_add",
        source_id=source_add["reward_ref"],
        obligation_hash=sha256_json(
            source_add_reward_row_projection_v2(
                "source_add_leg1",
                {**source_add, "initial_reward_status": "active"},
            )
        ),
        start_epoch=120,
        epoch_count=20,
        desired_alpha_percent=1,
        applied_alpha_percent=10,
        realized_alpha_percent=10,
        excess_alpha_percent=0,
    )
    predecessor = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=120,
        predecessor_frontier_hash=None,
        reward_checkpoints=(champion_checkpoint, source_add_checkpoint),
    )
    resolver = object.__new__(CoordinatorAllocationSourceV2)
    try:
        resolver._build_settlement_frontier(
            epoch=121,
            netuid=71,
            champion_rows=[],
            source_add_rows=[],
            history=[],
            predecessor=predecessor,
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "unsettled reward disappeared" not in str(exc):
            raise
    else:
        raise RuntimeError("terminal frontier failure was not reproduced")

    rows = {
        "champion_reward_by_id": [champion],
        "source_add_reward_by_ref": [source_add],
    }
    observed_queries: list[str] = []

    def read(policy_id, parameters, _context):
        observed_queries.append(str(policy_id))
        return [dict(row) for row in rows.get(policy_id, [])]

    resolver._read = read
    context = ExecutionContextV2(
        job_id="allocation-v2:terminal-retirement-rehearsal",
        purpose="research_lab.allocation.v2",
        epoch_id=121,
        parent_receipt_hashes=(),
    )
    retirements = resolver._resolve_settlement_frontier_retirements(
        predecessor=predecessor,
        champion_rows=[],
        source_add_rows=[],
        context=context,
    )
    successor = resolver._build_settlement_frontier(
        epoch=121,
        netuid=71,
        champion_rows=[],
        source_add_rows=[],
        history=[],
        predecessor=predecessor,
        terminal_retirements=retirements,
    )
    if (
        successor["reward_checkpoint_count"] != 0
        or observed_queries
        != ["champion_reward_by_id", "source_add_reward_by_ref"]
        or {item["terminal_status"] for item in retirements}
        != {"paid", "stopped_forward"}
    ):
        raise RuntimeError("terminal frontier retirement was not exact")

    rows["champion_reward_by_id"] = [
        {**champion, "input_hash": "sha256:" + "e" * 64}
    ]
    try:
        resolver._resolve_settlement_frontier_retirements(
            predecessor=predecessor,
            champion_rows=[],
            source_add_rows=[],
            context=context,
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "terminal reward identity changed" not in str(exc):
            raise
    else:
        raise RuntimeError("mutated terminal reward did not fail closed")

    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes_raw().hex()
    boot_body = build_boot_identity_body(
        role="gateway_coordinator",
        physical_role="gateway_coordinator",
        commit_sha="a" * 40,
        pcr0="b" * 96,
        build_manifest_hash="sha256:" + "c" * 64,
        dependency_lock_hash="sha256:" + "d" * 64,
        config_hash="sha256:" + "e" * 64,
        boot_nonce="f" * 32,
        signing_pubkey=signing_pubkey,
        transport_pubkey="1" * 64,
        transport_certificate_hash="sha256:" + "2" * 64,
        attestation_user_data_hash="sha256:" + "3" * 64,
        issued_at=NOW,
    )
    boot_identity = create_boot_identity(
        body=boot_body,
        attestation_document_b64=base64.b64encode(
            b"frontier-release-rehearsal"
        ).decode("ascii"),
    )
    source_state = {"settlement_frontier": predecessor}
    source_state_hash = sha256_json(source_state)
    allocation = {"allocation_hash": "sha256:" + "4" * 64}
    result = {
        "allocation": allocation,
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        set(frontier_artifact_hashes_v2(predecessor))
        | {source_state_hash}
    )
    artifact_root = merkle_root(
        artifact_hashes,
        domain="leadpoet-artifact-v2",
    )
    output_root = sha256_json({"allocation": allocation})
    receipt_body = build_execution_receipt_body(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="allocation-v2:frontier-release-rehearsal:120",
        epoch_id=120,
        sequence=0,
        commit_sha=boot_identity["commit_sha"],
        pcr0=boot_identity["pcr0"],
        build_manifest_hash=boot_identity["build_manifest_hash"],
        dependency_lock_hash=boot_identity["dependency_lock_hash"],
        config_hash=boot_identity["config_hash"],
        boot_identity_hash=boot_identity["boot_identity_hash"],
        input_root="sha256:" + "5" * 64,
        output_root=output_root,
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=artifact_root,
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    receipt = create_signed_execution_receipt(
        body=receipt_body,
        enclave_pubkey=signing_pubkey,
        sign_digest=signing_key.sign,
    )
    execution = {
        "schema_version": "leadpoet.attested_execution_result.v2",
        "receipt_hash": receipt["receipt_hash"],
        "role": "gateway_coordinator",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "job_id": receipt["job_id"],
        "epoch_id": 120,
        "sequence": 0,
        "release_hash": "sha256:" + "6" * 64,
        "input_root": receipt["input_root"],
        "output_root": output_root,
        "artifact_root": artifact_root,
        "result_hash": sha256_json(result),
        "artifact_hashes": artifact_hashes,
        "result_doc": result,
    }
    frontier_row = {
        "schema_version": predecessor["schema_version"],
        "netuid": 71,
        "allocation_epoch": 120,
        "settled_through_epoch": 119,
        "frontier_hash": predecessor["frontier_hash"],
        "predecessor_frontier_hash": None,
        "source_receipt_hash": receipt["receipt_hash"],
        "source_state_hash": source_state_hash,
        "frontier_doc": predecessor,
    }

    def authority_read(policy_id, parameters, _context):
        if policy_id == "allocation_settlement_frontier_activation":
            return [
                {
                    "schema_version": (
                        "leadpoet.research_lab_allocation_"
                        "settlement_frontier_activation.v2"
                    ),
                    "netuid": 71,
                    "first_allocation_epoch": 120,
                    "first_frontier_hash": predecessor["frontier_hash"],
                    "source_receipt_hash": receipt["receipt_hash"],
                }
            ]
        if policy_id in {
            "allocation_settlement_frontiers",
            "allocation_settlement_frontier_by_epoch",
        }:
            return [dict(frontier_row)]
        if policy_id == "attested_execution_result_by_receipt":
            return [dict(execution)]
        if policy_id == "attested_receipt_by_hash":
            return [
                {
                    "receipt_hash": receipt["receipt_hash"],
                    "receipt_doc": dict(receipt),
                }
            ]
        return []

    resolver._read = authority_read
    authority_context = ExecutionContextV2(
        job_id="allocation-v2:frontier-release-successor:121",
        purpose="research_lab.allocation.v2",
        epoch_id=121,
        parent_receipt_hashes=(receipt["receipt_hash"],),
    )
    authority_context.external_receipt_graphs = [
        build_receipt_graph(
            root_receipt_hash=receipt["receipt_hash"],
            boot_identities=(boot_identity,),
            receipts=(receipt,),
            transport_attempts=(),
        )
    ]
    required_parents: set[str] = set()
    authority = resolver._load_prior_settlement_frontier(
        epoch=121,
        netuid=71,
        context=authority_context,
        required_parents=required_parents,
    )
    if (
        authority
        != {"frontier": predecessor, "receipt_hash": receipt["receipt_hash"]}
        or required_parents != {receipt["receipt_hash"]}
        or "release_hash" in receipt
    ):
        raise RuntimeError(
            "canonical receipt and execution release authority differed"
        )
    execution["release_hash"] = "invalid"
    try:
        resolver._load_prior_settlement_frontier(
            epoch=121,
            netuid=71,
            context=authority_context,
            required_parents=set(),
        )
    except CoordinatorAllocationSourceV2Error as exc:
        if "execution authority differs" not in str(exc):
            raise
    else:
        raise RuntimeError("malformed execution release hash did not fail closed")
    return {
        "original_failure_reproduced": True,
        "champion_terminal_retired": True,
        "source_add_terminal_retired": True,
        "tampered_identity_rejected": True,
        "successor_reward_checkpoint_count": 0,
        "canonical_receipt_without_release_hash_accepted": True,
        "execution_release_hash_validated": True,
    }


def _exercise_current_frontier_release_recovery() -> dict[str, Any]:
    """Prove a release transition reuses one immutable epoch authority."""

    from gateway.research_lab import attested_v2_store, v2_authority
    from leadpoet_canonical.allocation_settlement_frontier_v2 import (
        build_allocation_settlement_frontier_v2,
        frontier_artifact_hashes_v2,
    )
    from leadpoet_canonical.allocation_handoff_v2 import (
        build_allocation_handoff_v2,
        validate_allocation_handoff_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_boot_identity_body,
        create_boot_identity,
    )

    epoch = 24321
    frontier = build_allocation_settlement_frontier_v2(
        mode="legacy_full_history_bootstrap",
        netuid=71,
        allocation_epoch=epoch,
        predecessor_frontier_hash=None,
        reward_checkpoints=(),
    )
    policy: dict[str, Any] = {}
    source_state = {
        "epoch": epoch,
        "netuid": 71,
        "policy": policy,
        "reimbursement_obligations": [],
        "champion_obligations": [],
        "settlement_frontier": frontier,
    }
    allocation_inputs = {
        "epoch": epoch,
        "policy": policy,
        "active_reimbursement_obligations": [],
        "active_champion_obligations": [],
    }
    allocation = {
        "epoch": epoch,
        "netuid": 71,
        "allocation_hash": sha256_json({"epoch": epoch, "netuid": 71}),
    }
    source_state_hash = sha256_json(source_state)
    result = {
        "allocation": allocation,
        "allocation_inputs": allocation_inputs,
        "source_state": source_state,
        "source_state_hash": source_state_hash,
    }
    artifact_hashes = sorted(
        set(frontier_artifact_hashes_v2(frontier)) | {source_state_hash}
    )
    signing_key = Ed25519PrivateKey.generate()
    signing_pubkey = signing_key.public_key().public_bytes_raw().hex()
    source_commit = "1" * 40
    boot_body = build_boot_identity_body(
        role="gateway_coordinator",
        physical_role="gateway_coordinator",
        commit_sha=source_commit,
        pcr0="2" * 96,
        build_manifest_hash="sha256:" + "3" * 64,
        dependency_lock_hash="sha256:" + "4" * 64,
        config_hash="sha256:" + "5" * 64,
        boot_nonce="6" * 32,
        signing_pubkey=signing_pubkey,
        transport_pubkey="7" * 64,
        transport_certificate_hash="sha256:" + "8" * 64,
        attestation_user_data_hash="sha256:" + "9" * 64,
        issued_at=NOW,
    )
    boot = create_boot_identity(
        body=boot_body,
        attestation_document_b64=base64.b64encode(
            b"current-frontier-release-recovery"
        ).decode("ascii"),
    )
    receipt_body = build_execution_receipt_body(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="allocation-v2:prior-release:24321",
        epoch_id=epoch,
        sequence=0,
        commit_sha=source_commit,
        pcr0=boot["pcr0"],
        build_manifest_hash=boot["build_manifest_hash"],
        dependency_lock_hash=boot["dependency_lock_hash"],
        config_hash=boot["config_hash"],
        boot_identity_hash=boot["boot_identity_hash"],
        input_root=sha256_json({"epoch": epoch, "netuid": 71}),
        output_root=sha256_json({"allocation": allocation}),
        transport_root_hash=EMPTY_TRANSPORT_ROOT,
        host_operation_root_hash=EMPTY_HOST_OPERATION_ROOT,
        artifact_root=merkle_root(
            artifact_hashes,
            domain="leadpoet-artifact-v2",
        ),
        parent_receipt_hashes=(),
        status="succeeded",
        failure_code=None,
        issued_at=NOW,
    )
    receipt = create_signed_execution_receipt(
        body=receipt_body,
        enclave_pubkey=signing_pubkey,
        sign_digest=signing_key.sign,
    )
    graph = build_receipt_graph(
        root_receipt_hash=receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(receipt,),
        transport_attempts=(),
    )
    context = {
        "frontier": frontier,
        "row": {
            "source_receipt_hash": receipt["receipt_hash"],
        },
        "source": {
            "row": {
                "receipt_hash": receipt["receipt_hash"],
                "operation": v2_authority.OP_RESEARCH_LAB_ALLOCATION,
                "purpose": "research_lab.allocation.v2",
                "role": "gateway_coordinator",
                "epoch_id": epoch,
                "release_hash": "sha256:" + "a" * 64,
            },
            "result": result,
            "receipt": receipt,
            "receipt_graph": graph,
            "artifact_hashes": artifact_hashes,
        },
    }
    execute_calls = 0

    async def load_context(**kwargs):
        if kwargs != {"netuid": 71, "before_epoch": epoch + 1}:
            raise RuntimeError("current frontier lookup scope changed")
        return context

    async def execute(**_kwargs):
        nonlocal execute_calls
        execute_calls += 1
        raise RuntimeError("current frontier was re-executed under a new release")

    async def persist_links(**kwargs):
        if kwargs.get("receipt_hash") != receipt["receipt_hash"]:
            raise RuntimeError("current frontier business link changed authority")
        return {"business_artifact_link_count": 1}

    original_loader = (
        attested_v2_store.load_allocation_settlement_frontier_context_v2
    )
    attested_v2_store.load_allocation_settlement_frontier_context_v2 = (
        load_context
    )
    try:
        recovered = asyncio.run(
            v2_authority.build_allocation_v2(
                epoch_id=epoch,
                netuid=71,
                policy=policy,
                execute=execute,
                persist_links=persist_links,
            )
        )
        if (
            execute_calls != 0
            or recovered.get("result") != result
            or recovered.get("receipt") != receipt
            or recovered.get("receipt_graph") != graph
            or recovered.get("replay_status")
            != "durable_current_frontier"
        ):
            raise RuntimeError("current frontier release recovery differed")
        handoff = build_allocation_handoff_v2(
            bundle={
                "epoch": epoch,
                "netuid": 71,
                "allocation_doc": allocation,
            },
            receipt_graph=recovered["receipt_graph"],
            lineage_bindings=recovered["lineage_bindings"],
            lineage_complete=recovered["lineage_complete"],
            persistence=recovered["persistence"],
        )
        validate_allocation_handoff_v2(
            handoff,
            expected_epoch_id=epoch,
            expected_netuid=71,
        )

        context["source"]["row"]["release_hash"] = "invalid"
        try:
            asyncio.run(
                v2_authority.build_allocation_v2(
                    epoch_id=epoch,
                    netuid=71,
                    policy=policy,
                    execute=execute,
                    persist_links=persist_links,
                )
            )
        except v2_authority.ResearchLabV2AuthorityError as exc:
            if "source authority differs" not in str(exc):
                raise
        else:
            raise RuntimeError("malformed current frontier release was accepted")
    finally:
        attested_v2_store.load_allocation_settlement_frontier_context_v2 = (
            original_loader
        )

    return {
        "cross_release_execution_skipped": True,
        "exact_signed_authority_reused": True,
        "immutable_frontier_preserved": True,
        "canonical_handoff_verified": True,
        "malformed_release_rejected": True,
    }


def _exercise_validator_publication_release_recovery() -> dict[str, Any]:
    """Prove an approved N-1 validator journal survives N activation."""

    import subprocess
    import neurons.validator as validator_module

    from leadpoet_canonical.attested_v2 import sha256_json
    from validator_tee.enclave.hotkey_authority_v2 import (
        ValidatorHotkeyAuthorityV2,
        ValidatorHotkeyAuthorityV2Error,
        load_chain_signing_profile,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    from_sha = str(os.environ.get("REHEARSAL_FROM_SHA") or "").lower()
    if (
        len(from_sha) != 40
        or any(value not in "0123456789abcdef" for value in from_sha)
        or from_sha == candidate_sha
    ):
        raise RuntimeError(
            "validator publication recovery requires a distinct N-1 release"
        )
    current = SanitizedWeightFixture(candidate_sha=candidate_sha, epoch_id=30_000)
    previous = SanitizedWeightFixture(candidate_sha=from_sha, epoch_id=30_000)
    current_boot = current._boot(
        role="validator_weights",
        key=current.weight_key,
        config_hash=sha256_json({"release": candidate_sha}),
    )
    old_boot = previous._boot(
        role="validator_weights",
        key=previous.weight_key,
        config_hash=sha256_json({"release": from_sha}),
    )
    expectation_fields = (
        "commit_sha",
        "pcr0",
        "build_manifest_hash",
        "dependency_lock_hash",
    )
    lineage = {
        from_sha: {
            "roles": {
                "validator_weights": {
                    field: old_boot[field] for field in expectation_fields
                }
            }
        }
    }
    verified = []

    def verify_boot(identity, **kwargs):
        if identity.get("pcr0") != kwargs.get("expected_pcr0"):
            raise RuntimeError("recovery boot PCR0 differs")
        if kwargs.get("certificate_validity_at_attestation_time") is not True:
            raise RuntimeError("historical attestation time was not enforced")
        verified.append(str(identity["boot_identity_hash"]))
        return {"verified": True}

    class _UnusedSr25519:
        pass

    authority = ValidatorHotkeyAuthorityV2(
        boot_identity_supplier=lambda: current_boot,
        gateway_release_lineage_supplier=lambda: lineage,
        validator_hotkey=VALIDATOR_HOTKEY,
        hotkey_public_key_hex="1" * 64,
        chain_profile=load_chain_signing_profile(
            SOURCE_ROOT
            / "validator_tee/enclave/chain_signing_profile_v2.json"
        ),
        sign_receipt_digest=current.weight_key.sign,
        attestation_supplier=lambda **_kwargs: b"unused",
        drand_backend=object(),
        sr25519_backend=_UnusedSr25519(),
        boot_verifier=verify_boot,
    )
    authority._verify_recovery_validator_boot(old_boot)
    if verified != [old_boot["boot_identity_hash"]]:
        raise RuntimeError("approved N-1 validator boot was not attested")

    rejected = 0
    for field, value in (
        ("pcr0", "0" * 96),
        ("build_manifest_hash", "sha256:" + "0" * 64),
        ("dependency_lock_hash", "sha256:" + "0" * 64),
    ):
        try:
            authority._verify_recovery_validator_boot(
                {**old_boot, field: value}
            )
        except ValidatorHotkeyAuthorityV2Error:
            rejected += 1
    try:
        authority._verify_recovery_validator_boot(
            {**old_boot, "commit_sha": "0" * 40}
        )
    except ValidatorHotkeyAuthorityV2Error:
        rejected += 1
    try:
        authority._verify_recovery_validator_boot(
            {
                **current_boot,
                "config_hash": "sha256:" + "0" * 64,
            }
        )
    except ValidatorHotkeyAuthorityV2Error:
        rejected += 1
    if rejected != 5:
        raise RuntimeError("validator recovery release tampering was accepted")
    if authority._recovery_finalization_only_mode(
        old_boot=current_boot,
        extrinsic_signature_results=[],
        allow_cross_release_finalization_only=False,
    ):
        raise RuntimeError("same-release recovery became finalization-only")
    finalization_only = authority._recovery_finalization_only_mode(
        old_boot=old_boot,
        extrinsic_signature_results=[{"durable_signed_extrinsic": True}],
        allow_cross_release_finalization_only=True,
    )
    finalization_mode_rejections = 0
    for signatures, allowed in (
        ([{"durable_signed_extrinsic": True}], False),
        ([], True),
    ):
        try:
            authority._recovery_finalization_only_mode(
                old_boot=old_boot,
                extrinsic_signature_results=signatures,
                allow_cross_release_finalization_only=allowed,
            )
        except ValidatorHotkeyAuthorityV2Error:
            finalization_mode_rejections += 1
    if not finalization_only or finalization_mode_rejections != 2:
        raise RuntimeError(
            "cross-release recovery was not constrained to signed finalization"
        )

    event_hash = "sha256:" + "7" * 64
    authorization_id = "sha256:" + "8" * 64

    class _RecoveryJournal:
        def __init__(self) -> None:
            self.record = {
                "weight_authorization_id": authorization_id,
                "published_bundle": {
                    "weight_result": {"epoch_id": current.epoch_id}
                },
                "publication": {
                    "weight_submission_event_hash": event_hash
                },
                "extrinsic_signature_results": [
                    {"durable_signed_extrinsic": True}
                ],
            }
            self.scan = 0
            self.cleared = False

        def load(self):
            return self.record

        def replace_authorization(self, value):
            self.record = {**self.record, "weight_authorization_id": value}
            return self.record

        def reserve_finalization_scan(self):
            self.scan += 1
            return "sha256:" + format(self.scan, "064x")

        def clear(self, *, expected_event_hash):
            if expected_event_hash != event_hash:
                raise RuntimeError("rehearsal cleared another publication")
            self.record = None
            self.cleared = True

    class _RecoveryClient:
        def recover_weight_publication_v2(self, **_kwargs):
            return {
                "weight_authorization_id": authorization_id,
                "signed_extrinsics": [
                    {
                        "authorization_hash": "sha256:" + "9" * 64,
                        "extrinsic_hash": "0x" + "a" * 64,
                        "extrinsic_hex": "00",
                    }
                ],
                "finalization_only": True,
            }

        def confirm_weight_publication_v2(
            self, _authorization_id, *, finalization_scan_id
        ):
            if not str(finalization_scan_id).startswith("sha256:"):
                raise RuntimeError("finalization scan identity is invalid")
            return {"finalized": True}

    journal = _RecoveryJournal()
    validator = validator_module.Validator.__new__(validator_module.Validator)
    validator._weight_publication_journal_v2 = journal
    validator._validator_v2_client = _RecoveryClient()
    validator.wallet = SimpleNamespace(
        hotkey=SimpleNamespace(ss58_address=VALIDATOR_HOTKEY)
    )
    active_epoch = current.epoch_id

    async def epoch_state():
        return SimpleNamespace(workflow_epoch_id=active_epoch)

    validator._get_epoch_state_async = epoch_state
    validator._get_best_epoch_state_async = epoch_state
    original_finalize = (
        validator_module.finalize_authoritative_weight_publication_v2
    )

    async def finalize(**_kwargs):
        return {
            "acknowledgment": {
                "weight_finalization_event_hash": "sha256:" + "b" * 64
            }
        }

    validator_module.finalize_authoritative_weight_publication_v2 = finalize
    try:
        same_epoch = asyncio.run(
            validator._recover_weight_publication_before_new_authority_v2(
                epoch_id=current.epoch_id,
                gateway_url="https://gateway.rehearsal.invalid",
            )
        )
        if not same_epoch or journal.record is None or journal.cleared:
            raise RuntimeError(
                "same-epoch finalized publication did not survive restart"
            )
        active_epoch = current.epoch_id + 1
        next_epoch = asyncio.run(
            validator._recover_weight_publication_before_new_authority_v2(
                epoch_id=active_epoch,
                gateway_url="https://gateway.rehearsal.invalid",
            )
        )
        if next_epoch or journal.record is not None or not journal.cleared:
            raise RuntimeError(
                "revalidated prior publication blocked the next epoch"
            )
    finally:
        validator_module.finalize_authoritative_weight_publication_v2 = (
            original_finalize
        )
    return {
        "approved_n_minus_one_recovered": True,
        "nitro_attestation_rechecked": True,
        "release_tampering_rejected": True,
        "same_release_config_mismatch_rejected": True,
        "cross_release_finalization_only": True,
        "unsigned_cross_release_rejected": True,
        "implicit_cross_release_rejected": True,
        "same_epoch_finalized_journal_retained": True,
        "next_epoch_finalized_journal_retired": True,
    }


def _exercise_receipt_graph_aggregate_pagination() -> dict[str, Any]:
    """Exercise aggregate evidence paging through the candidate store helper."""

    from gateway.research_lab import attested_v2_store

    row_limit = int(attested_v2_store._MAX_GRAPH_ROWS)
    query_chunk = int(attested_v2_store._GRAPH_QUERY_CHUNK)
    if row_limit < 1 or query_chunk < 1 or query_chunk > row_limit:
        raise RuntimeError("candidate V2 receipt graph limits are invalid")

    row_count = row_limit + 1
    width = len(str(row_count))
    expected_rows = [
        {
            "attempt_hash": (
                f"rehearsal-aggregate-attempt-{index:0{width}d}"
            )
        }
        for index in range(row_count)
    ]
    expected_by_key = {
        str(row["attempt_hash"]): dict(row) for row in expected_rows
    }
    expected_keys = set(expected_by_key)
    observed_queries: list[dict[str, Any]] = []
    original_select_all = attested_v2_store.select_all

    async def strict_select_all(
        table: str,
        *,
        filters: tuple[tuple[str, str, Any], ...],
        order_by: tuple[tuple[str, bool], ...],
        max_rows: int,
        **_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if (
            table != attested_v2_store.TRANSPORT_TABLE
            or len(filters) != 1
            or filters[0][0] != "attempt_hash"
            or filters[0][1] != "in"
            or order_by != (("attempt_hash", False),)
            or int(max_rows) != row_limit
        ):
            raise RuntimeError(
                "receipt graph rehearsal received an unknown store operation"
            )
        values = [str(value) for value in filters[0][2]]
        if not values or len(values) > query_chunk:
            raise RuntimeError(
                "receipt graph rehearsal query exceeded candidate chunk limit"
            )
        unknown = sorted(set(values) - expected_keys)
        if unknown:
            raise RuntimeError(
                "receipt graph rehearsal queried undeclared evidence"
            )
        observed_queries.append(
            {
                "count": len(values),
                "first": values[0],
                "last": values[-1],
            }
        )
        return [dict(expected_by_key[value]) for value in values]

    async def exercise() -> tuple[set[str], bool]:
        attested_v2_store.select_all = strict_select_all
        try:
            existing = await attested_v2_store._existing_exact_rows(
                attested_v2_store.TRANSPORT_TABLE,
                key_field="attempt_hash",
                expected_rows=expected_rows,
            )
            try:
                await attested_v2_store._select_by_values(
                    attested_v2_store.RECEIPT_TABLE,
                    field="receipt_hash",
                    values=(
                        f"rehearsal-receipt-{index:0{width}d}"
                        for index in range(row_count)
                    ),
                    key_fields=("receipt_hash",),
                )
            except attested_v2_store.AttestedV2StoreError as exc:
                if str(exc) != "V2 receipt graph exceeds row limit":
                    raise
                structural_limit_enforced = True
            else:
                structural_limit_enforced = False
            return existing, structural_limit_enforced
        finally:
            attested_v2_store.select_all = original_select_all

    existing, structural_limit_enforced = asyncio.run(exercise())
    if existing != expected_keys:
        raise RuntimeError("aggregate V2 receipt evidence was not exact")
    if len(observed_queries) < 2:
        raise RuntimeError("aggregate V2 receipt evidence was not paged")
    if (
        max(int(query["count"]) for query in observed_queries) > query_chunk
        or not structural_limit_enforced
    ):
        raise RuntimeError("V2 receipt graph safety bounds were weakened")
    parent_hash = "sha256:" + "1" * 64
    child_hash = "sha256:" + "2" * 64
    checkpoint_delta = {
        "receipts": [
            {
                "receipt_hash": child_hash,
                "parent_receipt_hashes": [parent_hash],
            },
            {
                "receipt_hash": parent_hash,
                "parent_receipt_hashes": [],
            },
        ]
    }
    parent_first = attested_v2_store._parent_first_receipt_hashes_v2(
        checkpoint_delta,
        validated_receipts=(child_hash, parent_hash),
    )
    if parent_first != (parent_hash, child_hash):
        raise RuntimeError(
            "checkpoint receipt membership was used as insertion order"
        )
    return {
        "aggregate_rows": row_count,
        "aggregate_evidence_paged": True,
        "checkpoint_parent_first_persistence": True,
        "per_query_row_limit": row_limit,
        "query_chunk": query_chunk,
        "query_count": len(observed_queries),
        "structural_limit_enforced": True,
    }


def _exercise_receipt_graph_transport_deduplication() -> dict[str, Any]:
    """Run shared ancestry through the exact job admission and decode path."""

    import subprocess

    from gateway.tee.execution_job_manager_v2 import (
        JOB_SCHEMA_VERSION,
        MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
        MAX_ALLOCATION_ANCESTRY_INPUT_BYTES,
        MAX_EXTERNAL_RECEIPT_GRAPHS,
        MAX_INPUT_BYTES,
        PARENT_RECEIPT_GRAPH_SET_FIELD,
        ExecutionJobManagerV2,
        ExecutionJobV2Error,
        pack_parent_receipt_graph_set_v2,
        unpack_parent_receipt_graph_set_v2,
    )
    from gateway.tee.release_lineage_v2 import _required_commits
    from gateway.research_lab.attested_scoring_v2 import (
        _build_transport_payload_document,
    )
    from gateway.tee.coordinator_allocation_source_v2 import (
        _receipt_graphs_by_declared_root,
    )
    from leadpoet_canonical.attested_v2 import (
        CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION,
        RECEIPT_GRAPH_SCHEMA_VERSION,
        build_checkpointed_receipt_graph,
        sha256_bytes,
    )
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    from_sha = str(os.environ.get("REHEARSAL_FROM_SHA") or "").lower()
    if (
        len(from_sha) != 40
        or any(value not in "0123456789abcdef" for value in from_sha)
        or from_sha == candidate_sha
    ):
        raise RuntimeError(
            "receipt ancestry rehearsal requires a distinct N-1 release"
        )
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=30_000,
    )
    historical_fixture = SanitizedWeightFixture(
        candidate_sha=from_sha,
        epoch_id=30_000,
    )
    config_hash = sha256_json({"rehearsal": "shared-receipt-ancestry"})
    boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=config_hash,
    )
    historical_boot = historical_fixture._boot(
        role="gateway_coordinator",
        key=historical_fixture.coordinator_key,
        config_hash=config_hash,
    )
    sample_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="rehearsal-shared-ancestry-sample",
        key=fixture.coordinator_key,
        boot=boot,
        config_hash=config_hash,
        sequence=0,
    )
    sample_receipt_bytes = len(_canonical(sample_receipt))
    graph_count = MAX_ALLOCATION_ANCESTRY_AUTHORITIES - 1
    legacy_reproduction_target = MAX_INPUT_BYTES * 2 + MAX_INPUT_BYTES // 32
    shared_receipt_count = (
        legacy_reproduction_target
        + graph_count * sample_receipt_bytes
        - 1
    ) // (graph_count * sample_receipt_bytes)
    shared_receipt_count += 8

    shared_receipts: list[dict[str, Any]] = []
    parents: list[str] = []
    for index in range(shared_receipt_count):
        receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-shared-ancestry-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=parents,
            sequence=index,
        )
        shared_receipts.append(receipt)
        parents = [str(receipt["receipt_hash"])]

    checkpoint_graph_count = 2
    graphs: list[dict[str, Any]] = []
    for index in range(graph_count - checkpoint_graph_count):
        child = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-independent-root-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=parents,
            sequence=100 + index,
        )
        graph = {
            "schema_version": RECEIPT_GRAPH_SCHEMA_VERSION,
            "root_receipt_hash": str(child["receipt_hash"]),
            "boot_identities": [boot],
            "receipts": [*shared_receipts, child],
            "transport_attempts": [],
            "host_operations": [],
        }
        graphs.append(graph)

    lineage_id = sha256_json({"rehearsal": "mixed-allocation-frontier"})

    def verify_boot(identity):
        return identity

    for index in range(checkpoint_graph_count):
        receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="research_lab.allocation.v2",
            job_id=f"rehearsal-checkpointed-root-{index}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            parents=(),
            sequence=graph_count + index,
        )
        delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        issuer_boot = historical_boot if index == 0 else boot
        issuer_key = (
            historical_fixture.coordinator_key
            if index == 0
            else fixture.coordinator_key
        )
        certificate = issue_ancestry_certificate_v2(
            local_delta=delta,
            lineage_id=lineage_id,
            certificate_sequence=0,
            issuer_boot_identity=issuer_boot,
            issued_at="2026-07-10T20:00:00Z",
            sign_digest=issuer_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            required_purposes=("research_lab.allocation.v2",),
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        graphs.append(
            build_checkpointed_receipt_graph(
                root_receipt_hash=receipt["receipt_hash"],
                boot_identities=(boot,),
                receipts=(receipt,),
                transport_attempts=(),
                host_operations=(),
                ancestry_lineage_id=lineage_id,
                ancestry_proof=proof,
                boot_attestation_verifier=verify_boot,
                require_boot_attestation_verification=True,
            )
        )
    if sum(
        graph.get("schema_version")
        == CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSION
        for graph in graphs
    ) != checkpoint_graph_count:
        raise RuntimeError("mixed checkpoint ancestry fixture is incomplete")
    required_release_commits = _required_commits(tuple(graphs))
    if required_release_commits != {candidate_sha, from_sha}:
        raise RuntimeError(
            "checkpoint issuer N-1 release was omitted from ancestry lineage"
        )

    try:
        pack_parent_receipt_graph_set_v2(graphs)
    except ExecutionJobV2Error as exc:
        if "external receipt graph count exceeds limit" not in str(exc):
            raise
    else:
        raise RuntimeError("ordinary V2 ancestry accepted allocation frontier")

    transport_document, transport_metadata = _build_transport_payload_document(
        payload={"epoch": 30_000},
        parent_graphs=graphs,
        max_parent_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    )
    if transport_metadata.get("encoding") != "receipt_graph_set":
        raise RuntimeError("oversized shared ancestry was not deduplicated")
    legacy_size_bytes = int(transport_metadata["legacy_size_bytes"])
    if legacy_size_bytes <= MAX_INPUT_BYTES * 2:
        raise RuntimeError("legacy payload did not reproduce the old boundary")
    graph_set = transport_document[PARENT_RECEIPT_GRAPH_SET_FIELD]
    reconstructed = unpack_parent_receipt_graph_set_v2(
        graph_set,
        max_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
    )
    if reconstructed != graphs:
        raise RuntimeError("deduplicated receipt graph membership differs")
    del reconstructed
    transport_payload = _canonical(transport_document)
    if len(transport_payload) >= legacy_size_bytes:
        raise RuntimeError("shared receipt ancestry was not deduplicated")
    projected_transport_bytes = (
        MAX_ALLOCATION_ANCESTRY_INPUT_BYTES * len(transport_payload)
        + legacy_size_bytes
        - 1
    ) // legacy_size_bytes
    if projected_transport_bytes > MAX_INPUT_BYTES:
        raise RuntimeError(
            "candidate graph-set ratio lacks ordinary-input headroom"
        )

    observed: dict[str, Any] = {}

    def executor(_operation, payload, context):
        observed["payload"] = dict(payload)
        observed["graphs"] = list(context.external_receipt_graphs)
        observed["derived_graphs"] = _receipt_graphs_by_declared_root(
            context.external_receipt_graphs,
            context.parent_receipt_hashes,
        )
        return {"status": "verified"}

    manager = ExecutionJobManagerV2(
        boot_identity_supplier=lambda: boot,
        sign_digest=fixture.coordinator_key.sign,
        operations={
            "research_lab_allocation": {"research_lab.allocation.v2"}
        },
        executor=executor,
        worker_count=1,
        configured_worker_count=0,
        ancestry_lineage_id=lineage_id,
        ancestry_boot_attestation_verifier=verify_boot,
        ancestry_allowed_issuer_roles=("gateway_coordinator",),
    )
    manifest = {
        "schema_version": JOB_SCHEMA_VERSION,
        "job_id": "rehearsal-shared-ancestry-job",
        "operation": "research_lab_allocation",
        "purpose": "research_lab.allocation.v2",
        "epoch_id": 30_000,
        "sequence": 0,
        "payload_sha256": sha256_bytes(transport_payload),
        "payload_size_bytes": len(transport_payload),
        "parent_receipt_hashes": [
            str(graph["root_receipt_hash"]) for graph in graphs
        ],
        "input_artifact_hashes": [],
        "provider_credential_profile": "default",
        "provider_credential_ref_hashes": {},
    }
    manager.submit(manifest)
    for offset in range(0, len(transport_payload), 512 * 1024):
        chunk = transport_payload[offset : offset + 512 * 1024]
        manager.put_chunk(
            job_id=manifest["job_id"],
            offset=offset,
            data_b64=base64.b64encode(chunk).decode("ascii"),
            chunk_sha256=sha256_bytes(chunk),
        )
    manager.seal(manifest["job_id"])
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = manager.status(manifest["job_id"])
        if status["state"] in {"succeeded", "failed", "cancelled"}:
            break
        time.sleep(0.01)
    else:
        raise RuntimeError("deduplicated receipt graph job did not terminate")
    expected_graphs_by_root = {
        str(graph["root_receipt_hash"]): graph for graph in graphs
    }
    if (
        status["state"] != "succeeded"
        or observed.get("payload") != {"epoch": 30_000}
        or observed.get("graphs") != graphs
        or observed.get("derived_graphs") != expected_graphs_by_root
    ):
        raise RuntimeError("deduplicated receipt graph job was not exact")
    observed_graphs = observed["graphs"]
    derived_graphs = observed["derived_graphs"]
    first_shared_root = str(graphs[0]["root_receipt_hash"])
    second_shared_root = str(graphs[1]["root_receipt_hash"])
    if (
        observed_graphs[0]["boot_identities"][0]
        is not observed_graphs[1]["boot_identities"][0]
        or observed_graphs[0]["receipts"][0]
        is not observed_graphs[1]["receipts"][0]
        or derived_graphs[first_shared_root]["receipts"][0]
        is not derived_graphs[second_shared_root]["receipts"][0]
    ):
        raise RuntimeError("shared receipt graph evidence was rematerialized")

    malformed = json.loads(json.dumps(graph_set))
    malformed["receipts"].append(
        {
            **dict(malformed["receipts"][0]),
            "receipt_hash": sha256_json({"unreferenced": True}),
        }
    )
    try:
        unpack_parent_receipt_graph_set_v2(
            malformed,
            max_graph_count=MAX_ALLOCATION_ANCESTRY_AUTHORITIES,
        )
    except Exception as exc:
        if "unreferenced evidence" not in str(exc):
            raise
    else:
        raise RuntimeError("unreferenced graph-set evidence did not fail closed")

    return {
        "graph_count": len(graphs),
        "shared_receipt_count": len(shared_receipts),
        "legacy_size_bytes": legacy_size_bytes,
        "transport_size_bytes": len(transport_payload),
        "projected_transport_bytes_at_scoped_limit": (
            projected_transport_bytes
        ),
        "unique_receipt_count": len(graph_set["receipts"]),
        "exact_job_path_verified": True,
        "allocation_source_path_verified": True,
        "shared_object_identity_preserved": True,
        "malformed_evidence_rejected": True,
        "ordinary_graph_bound_preserved": True,
        "checkpointed_graph_count": checkpoint_graph_count,
        "checkpoint_authority_preserved": True,
        "checkpoint_release_commits": sorted(required_release_commits),
        "historical_checkpoint_issuer_included": True,
    }


def _exercise_fresh_weight_input_lineage() -> dict[str, Any]:
    """Exercise fresh checkpoint lineage, replay, and fail-closed mismatch."""

    import subprocess

    from gateway.research_lab.attested_weight_inputs_v2 import (
        AttestedWeightInputsV2Error,
        build_gateway_weight_inputs_v2,
    )
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        build_checkpointed_receipt_graph,
        validate_receipt_graph,
    )
    from leadpoet_canonical.weight_authority_v2 import (
        GATEWAY_WEIGHT_INPUT_CATEGORIES,
        WEIGHT_INPUT_PURPOSES,
        gateway_weight_input_value_documents_v2,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=30_000,
    )
    config_hash = sha256_json({"rehearsal": "fresh-weight-input-lineage"})
    boot = fixture._boot(
        role="gateway_coordinator",
        key=fixture.coordinator_key,
        config_hash=config_hash,
    )
    allocation_receipt = fixture.receipt(
        role="gateway_coordinator",
        purpose="research_lab.allocation.v2",
        job_id="rehearsal-weight-input-allocation",
        key=fixture.coordinator_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json({"allocation": 30_000}),
        sequence=0,
    )
    allocation_graph = build_receipt_graph(
        root_receipt_hash=allocation_receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(allocation_receipt,),
        transport_attempts=(),
    )
    snapshot = fixture.calculation_snapshot(
        [allocation_receipt["receipt_hash"]],
        allocation_receipt["receipt_hash"],
    )
    expected_documents = gateway_weight_input_value_documents_v2(
        calculation_snapshot=snapshot,
        gateway_authority_event_hash=allocation_receipt["receipt_hash"],
    )
    lineage_id = sha256_json({"lineage": "fresh-weight-input"})

    def verify_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        return identity

    def outcome(
        *,
        category: str,
        sequence: int,
        fresh: bool,
        mismatched_execution: bool = False,
    ) -> dict[str, Any]:
        role, purpose = WEIGHT_INPUT_PURPOSES[category]
        document = expected_documents[category]
        execution_receipt = fixture.receipt(
            role=role,
            purpose=purpose,
            job_id=f"rehearsal-weight-input-{category}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            output_root=sha256_json(document),
            sequence=100 + sequence,
        )
        execution_graph = build_receipt_graph(
            root_receipt_hash=execution_receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(execution_receipt,),
            transport_attempts=(),
        )
        execution_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": execution_receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [execution_receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        execution_certificate = issue_ancestry_certificate_v2(
            local_delta=execution_delta,
            lineage_id=lineage_id,
            certificate_sequence=0,
            issuer_boot_identity=boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            required_purposes=(purpose,),
        )
        execution_proof = build_compact_ancestry_proof_from_delta_v2(
            execution_delta,
            execution_certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        if not fresh:
            return {
                "status": "succeeded",
                "result": document,
                "receipt": execution_receipt,
                "execution_receipt": execution_receipt,
                "execution_receipt_graph": execution_graph,
                "receipt_graph": execution_graph,
                "execution_ancestry_compact_proof": execution_proof,
                "ancestry_compact_proof": execution_proof,
            }

        persistence_receipt = fixture.receipt(
            role="gateway_coordinator",
            purpose="leadpoet.artifact_persistence.v2",
            job_id=f"rehearsal-weight-input-persistence-{category}",
            key=fixture.coordinator_key,
            boot=boot,
            config_hash=config_hash,
            output_root=sha256_json(
                {"source_receipt_hash": execution_receipt["receipt_hash"]}
            ),
            parents=(execution_receipt["receipt_hash"],),
            sequence=1_000 + sequence,
        )
        local_delta = {
            "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
            "root_receipt_hash": persistence_receipt["receipt_hash"],
            "boot_identities": [boot],
            "receipts": [persistence_receipt],
            "transport_attempts": [],
            "host_operations": [],
        }
        certificate = issue_ancestry_certificate_v2(
            local_delta=local_delta,
            lineage_id=lineage_id,
            certificate_sequence=1,
            issuer_boot_identity=boot,
            issued_at=NOW,
            sign_digest=fixture.coordinator_key.sign,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
            parent_proof_disclosures=(
                (execution_proof, execution_receipt["receipt_hash"]),
            ),
            required_purposes=("leadpoet.artifact_persistence.v2",),
        )
        proof = build_compact_ancestry_proof_from_delta_v2(
            local_delta,
            certificate,
            expected_lineage_id=lineage_id,
            boot_attestation_verifier=verify_boot,
            allowed_issuer_roles=("gateway_coordinator",),
        )
        lineage_graph = build_checkpointed_receipt_graph(
            root_receipt_hash=persistence_receipt["receipt_hash"],
            boot_identities=(boot,),
            receipts=(persistence_receipt,),
            transport_attempts=(),
            host_operations=(),
            ancestry_lineage_id=lineage_id,
            ancestry_proof=proof,
            boot_attestation_verifier=verify_boot,
            require_boot_attestation_verification=True,
        )
        validate_receipt_graph(
            lineage_graph,
            required_purposes=(purpose, "leadpoet.artifact_persistence.v2"),
        )
        exposed_execution_receipt = execution_receipt
        if mismatched_execution:
            exposed_execution_receipt = fixture.receipt(
                role=role,
                purpose=purpose,
                job_id=f"rehearsal-weight-input-mismatch-{category}",
                key=fixture.coordinator_key,
                boot=boot,
                config_hash=config_hash,
                output_root=sha256_json(document),
                sequence=2_000 + sequence,
            )
        return {
            "status": "succeeded",
            "result": document,
            "receipt": persistence_receipt,
            "execution_receipt": exposed_execution_receipt,
            "execution_receipt_graph": execution_graph,
            "execution_ancestry_compact_proof": execution_proof,
            "ancestry_compact_proof": proof,
            "receipt_graph": lineage_graph,
        }

    async def run(*, fresh: bool, mismatch_category: str | None = None):
        async def execute(**kwargs):
            category = str(kwargs["payload"]["category"])
            return outcome(
                category=category,
                sequence=int(kwargs["sequence"]),
                fresh=fresh,
                mismatched_execution=category == mismatch_category,
            )

        return await build_gateway_weight_inputs_v2(
            calculation_snapshot=snapshot,
            allocation_graph=allocation_graph,
            leaderboard_window_start="2026-07-24T00:00:00Z",
            leaderboard_window_end="2026-07-25T00:00:00Z",
            execute=execute,
            load_sourcing_graphs=lambda **_kwargs: _async_value([]),
            coordinator_client_factory=object,
        )

    async def _async_value(value):
        return value

    fresh = asyncio.run(run(fresh=True))
    replay = asyncio.run(run(fresh=False))
    if fresh["input_receipt_hashes"] != replay["input_receipt_hashes"]:
        raise RuntimeError("fresh and replay input identities differ")
    compact = fresh.get("compact_ancestry")
    if not isinstance(compact, Mapping):
        raise RuntimeError("fresh execution compact ancestry is absent")
    proof_roots = {
        category: str(
            proof["certificate"]["claim"]["output_root_receipt_hash"]
        )
        for category, proof in compact["upstream_ancestry_proofs"].items()
    }
    if proof_roots != fresh["input_receipt_hashes"]:
        raise RuntimeError("fresh compact ancestry does not bind direct inputs")
    direct_hashes = set(fresh["input_receipt_hashes"].values())
    receipt_hashes = {
        str(item["receipt_hash"])
        for item in fresh["upstream_receipt_set"]["receipts"]
    }
    if (
        set(fresh["input_receipt_hashes"])
        != set(GATEWAY_WEIGHT_INPUT_CATEGORIES)
        or not direct_hashes.issubset(receipt_hashes)
        or len(receipt_hashes) != 2 * len(GATEWAY_WEIGHT_INPUT_CATEGORIES)
    ):
        raise RuntimeError("fresh weight input receipt persistence is incomplete")
    try:
        asyncio.run(run(fresh=True, mismatch_category="fulfillment_rewards"))
    except AttestedWeightInputsV2Error as exc:
        if "measured input receipt is invalid" not in str(exc):
            raise
    else:
        raise RuntimeError("mismatched fresh execution receipt did not fail closed")
    return {
        "fresh_checkpoint_lineage_accepted": True,
        "direct_execution_proof_selected": True,
        "replay_identity_equal": True,
        "direct_receipts_persisted": True,
        "mismatched_execution_rejected": True,
    }


def _exercise_stateful_compact_graph_readback() -> dict[str, Any]:
    """Exercise V3 persistence followed by its canonical V4 readback."""

    import copy
    import subprocess

    from Leadpoet.utils.subnet_epoch import (
        SubnetEpochCutover,
        SubnetEpochSnapshot,
    )
    from gateway.research_lab.stateful_epoch_authority_v1 import (
        BOUNDARY_TABLE,
        SNAPSHOT_TABLE,
        StatefulEpochAuthorityStoreError,
        persist_post_cutover_evidence_v1,
    )
    from gateway.tee.coordinator_epoch_cutover_v2 import SNAPSHOT_PURPOSE
    from leadpoet_canonical.ancestry_checkpoint_v2 import (
        ANCESTRY_DELTA_SCHEMA_VERSION,
        build_compact_ancestry_proof_from_delta_v2,
        issue_ancestry_certificate_v2,
    )
    from leadpoet_canonical.attested_v2 import (
        WEIGHT_ROLE,
        build_checkpointed_receipt_graph,
        compact_checkpointed_receipt_graph,
    )

    candidate_sha = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    cutover = SubnetEpochCutover(
        network_genesis_hash="0x" + "1" * 64,
        netuid=71,
        cutover_block=1_000,
        cutover_block_hash="0x" + "2" * 64,
        first_subnet_epoch_index=10,
        first_settlement_epoch_id=101,
        last_legacy_epoch_id=100,
    )
    boundary_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash="0x" + "3" * 64,
        current_block=1_360,
        last_epoch_block=1_360,
        pending_epoch_at=0,
        subnet_epoch_index=11,
        tempo=360,
        blocks_since_last_step=0,
        observed_at=NOW,
    )
    current_snapshot = SubnetEpochSnapshot(
        network_genesis_hash=cutover.network_genesis_hash,
        netuid=cutover.netuid,
        head_kind="finalized",
        block_hash="0x" + "4" * 64,
        current_block=1_700,
        last_epoch_block=1_360,
        pending_epoch_at=0,
        subnet_epoch_index=11,
        tempo=360,
        blocks_since_last_step=340,
        observed_at=NOW,
    )
    boundary_doc = boundary_snapshot.to_dict(cutover=cutover)
    current_doc = current_snapshot.to_dict(cutover=cutover)
    epoch_id = int(current_doc["settlement_epoch_id"])
    fixture = SanitizedWeightFixture(
        candidate_sha=candidate_sha,
        epoch_id=epoch_id,
    )
    config_hash = sha256_json({"rehearsal": "stateful-compact-readback"})
    boot = fixture._boot(
        role=WEIGHT_ROLE,
        key=fixture.weight_key,
        config_hash=config_hash,
    )
    boundary_receipt = fixture.receipt(
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id=f"subnet-epoch-boundary:{epoch_id}",
        key=fixture.weight_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json(boundary_doc),
        sequence=0,
    )
    current_receipt = fixture.receipt(
        role=WEIGHT_ROLE,
        purpose=SNAPSHOT_PURPOSE,
        job_id=f"subnet-epoch-current:{epoch_id}",
        key=fixture.weight_key,
        boot=boot,
        config_hash=config_hash,
        output_root=sha256_json(current_doc),
        parents=(boundary_receipt["receipt_hash"],),
        sequence=1,
    )
    delta = {
        "schema_version": ANCESTRY_DELTA_SCHEMA_VERSION,
        "root_receipt_hash": current_receipt["receipt_hash"],
        "boot_identities": [boot],
        "receipts": [boundary_receipt, current_receipt],
        "transport_attempts": [],
        "host_operations": [],
    }
    lineage_id = sha256_json(
        {
            "cutover_mapping_hash": cutover.mapping_hash,
            "candidate_sha": candidate_sha,
        }
    )

    def verify_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        if identity.get("commit_sha") != candidate_sha:
            raise RuntimeError("checkpoint boot commit differs")
        return identity

    certificate = issue_ancestry_certificate_v2(
        local_delta=delta,
        lineage_id=lineage_id,
        certificate_sequence=0,
        issuer_boot_identity=boot,
        issued_at=NOW,
        sign_digest=fixture.weight_key.sign,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(WEIGHT_ROLE,),
        required_purposes=(SNAPSHOT_PURPOSE,),
    )
    proof = build_compact_ancestry_proof_from_delta_v2(
        delta,
        certificate,
        expected_lineage_id=lineage_id,
        boot_attestation_verifier=verify_boot,
        allowed_issuer_roles=(WEIGHT_ROLE,),
    )
    graph = build_checkpointed_receipt_graph(
        root_receipt_hash=current_receipt["receipt_hash"],
        boot_identities=(boot,),
        receipts=(boundary_receipt, current_receipt),
        transport_attempts=(),
        host_operations=(),
        ancestry_lineage_id=lineage_id,
        ancestry_proof=proof,
        boot_attestation_verifier=verify_boot,
        require_boot_attestation_verification=True,
    )
    compact_graph = compact_checkpointed_receipt_graph(
        graph,
        boot_attestation_verifier=verify_boot,
        require_boot_attestation_verification=True,
    )
    evidence = {
        "schema_version": "leadpoet.validator_subnet_epoch_evidence.v1",
        "validator_hotkey": VALIDATOR_HOTKEY,
        "bundle_hash": sha256_json({"bundle": epoch_id}),
        "cutover_mapping_hash": cutover.mapping_hash,
        "epoch_authority": current_doc,
        "epoch_authority_hash": sha256_json(current_doc),
        "epoch_authority_receipt_hash": current_receipt["receipt_hash"],
        "epoch_boundary": boundary_doc,
        "epoch_boundary_hash": sha256_json(boundary_doc),
        "epoch_boundary_receipt_hash": boundary_receipt["receipt_hash"],
        "receipt_graph": graph,
    }
    tables: dict[str, dict[str, dict[str, Any]]] = {
        BOUNDARY_TABLE: {},
        SNAPSHOT_TABLE: {},
    }

    async def persist_graph(value):
        return {
            "root_receipt_hash": value["root_receipt_hash"],
            "graph_hash": sha256_json(dict(value)),
        }

    async def load_graph(_root):
        return copy.deepcopy(compact_graph)

    async def insert(table, row):
        key_field = {
            BOUNDARY_TABLE: "boundary_hash",
            SNAPSHOT_TABLE: "snapshot_hash",
        }[table]
        key = str(row[key_field])
        if key in tables[table]:
            raise RuntimeError("23505 duplicate key unique constraint")
        tables[table][key] = copy.deepcopy(dict(row))
        return copy.deepcopy(dict(row))

    async def select(table, *, filters):
        field, value = filters[0]
        for row in tables[table].values():
            if row.get(field) == value:
                return copy.deepcopy(row)
        return None

    durable = asyncio.run(
        persist_post_cutover_evidence_v1(
            evidence,
            cutover=cutover.to_dict(),
            persist_graph=persist_graph,
            load_graph=load_graph,
            insert=insert,
            select=select,
        )
    )
    if (
        durable["receipt_graph_hash"] != sha256_json(graph)
        or durable["boundary"]["boundary_hash"]
        != evidence["epoch_boundary_hash"]
        or durable["snapshot"]["snapshot_hash"]
        != evidence["epoch_authority_hash"]
    ):
        raise RuntimeError("canonical compact readback changed stateful evidence")

    tampered = copy.deepcopy(compact_graph)
    tampered["receipts"] = []
    attempted_insert = False

    async def load_tampered(_root):
        return copy.deepcopy(tampered)

    async def reject_insert(_table, _row):
        nonlocal attempted_insert
        attempted_insert = True
        raise RuntimeError("tampered graph reached stateful persistence")

    try:
        asyncio.run(
            persist_post_cutover_evidence_v1(
                evidence,
                cutover=cutover.to_dict(),
                persist_graph=persist_graph,
                load_graph=load_tampered,
                insert=reject_insert,
                select=select,
            )
        )
    except StatefulEpochAuthorityStoreError as exc:
        if "receipt graph readback differs" not in str(exc):
            raise
    else:
        raise RuntimeError("tampered compact graph readback was accepted")
    if attempted_insert:
        raise RuntimeError("tampered compact graph mutated stateful evidence")
    return {
        "checkpoint_v3_persisted": True,
        "canonical_v4_readback_accepted": True,
        "boundary_persisted": True,
        "snapshot_persisted": True,
        "tampered_v4_rejected_before_write": True,
    }


_BROKER_OWNED_HTTPX_FAIL_CLOSED_CASES = (
    "async_client_marker",
    "copied_marker",
    "genuine_client_without_grant",
    "injected_mount",
    "redirect_enabled",
    "transport_swap",
    "wrong_role",
)


def _coordinator_broker_httpx_evidence_is_complete(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    blocked = value.get("fail_closed_cases")
    return (
        set(value)
        == {
            "coordinator_role_authority_bound",
            "direct_supabase_sidecar_receipt_bound",
            "fail_closed_cases",
            "real_broker_external_send_bound",
        }
        and value.get("coordinator_role_authority_bound") is True
        and value.get("real_broker_external_send_bound") is True
        and value.get("direct_supabase_sidecar_receipt_bound") is True
        and isinstance(blocked, Mapping)
        and set(blocked) == set(_BROKER_OWNED_HTTPX_FAIL_CLOSED_CASES)
        and all(
            blocked.get(case) is True
            for case in _BROKER_OWNED_HTTPX_FAIL_CLOSED_CASES
        )
    )


def _exercise_coordinator_broker_owned_httpx_grant() -> dict[str, Any]:
    """Exercise the exact coordinator HTTPX bypass and its fail-closed shape."""

    import httpx

    from gateway.tee.artifact_vault_v2 import EncryptedArtifactVaultV2
    from gateway.tee.execution_job_manager_v2 import ExecutionContextV2
    from gateway.tee.provider_broker_v2 import (
        BUILTIN_PROVIDER_ROUTES,
        HTTPXProviderTransport,
        ProviderBrokerV2,
        _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
        _broker_owned_httpx_send_scope,
        _close_client_transports,
        credential_reference_hash,
        expected_provider_credential_slots,
        is_broker_owned_httpx_client,
    )
    from gateway.tee.provider_client_v2 import (
        BrokeredProviderTransportV2,
        ProviderClientV2Error,
    )
    from gateway.tee.provider_outcome_store_v2 import ProviderOutcomeStoreV2
    from gateway.tee.rpc_authority import COORDINATOR_ROLE as RPC_COORDINATOR_ROLE
    from gateway.tee.topology import (
        COORDINATOR_ROLE as TOPOLOGY_COORDINATOR_ROLE,
        ROLE_SPECS,
        topology_document,
    )
    from leadpoet_canonical.attested_v2 import (
        DIRECT_EGRESS_REF_HASH,
        sha256_json,
    )

    class _TLS:
        def getpeercert(self, binary_form=False, /):
            if binary_form is not True:
                raise RuntimeError("broker HTTPX TLS evidence was not requested in DER")
            return b"candidate-derived-broker-httpx-peer"

        def version(self):
            return "TLSv1.3"

    class _NetworkStream:
        def get_extra_info(self, name):
            if name != "ssl_object":
                raise RuntimeError("unexpected broker HTTPX stream evidence lookup")
            return _TLS()

        def close(self):
            return None

    original_sync_send = httpx.Client.send
    previous_role = os.environ.get("LEADPOET_ENCLAVE_ROLE")
    physical_transport = HTTPXProviderTransport()
    router = None
    sync_clients: list[Any] = []
    async_client = None
    external_requests: list[httpx.Request] = []
    untrusted_requests: list[httpx.Request] = []
    blocked: dict[str, bool] = {}
    try:
        topology = topology_document()
        role_document = topology.get("roles", {}).get(TOPOLOGY_COORDINATOR_ROLE)
        if (
            RPC_COORDINATOR_ROLE != TOPOLOGY_COORDINATOR_ROLE
            or RPC_COORDINATOR_ROLE != "gateway_coordinator"
            or not isinstance(role_document, Mapping)
            or role_document.get("service_role") != RPC_COORDINATOR_ROLE
            or ROLE_SPECS.get(RPC_COORDINATOR_ROLE) != role_document
        ):
            raise RuntimeError("broker HTTPX coordinator role authority differs")
        os.environ["LEADPOET_ENCLAVE_ROLE"] = RPC_COORDINATOR_ROLE
        credential_values = {
            slot: "rehearsal-%s" % slot
            for slot in expected_provider_credential_slots()
        }
        retry_policy_hashes = {
            provider_id: sha256_json(
                {
                    "schema_version": "leadpoet.rehearsal_retry_policy.v1",
                    "provider_id": provider_id,
                }
            )
            for provider_id in BUILTIN_PROVIDER_ROUTES
        }
        vault = EncryptedArtifactVaultV2(
            master_key=bytes(reversed(range(32))),
            boot_identity_hash=sha256_json(
                {"boot": "coordinator-broker-httpx-grant"}
            ),
            retention_days=30,
        )

        def strict_external_send(client, request, *args, **kwargs):
            if args or kwargs.get("stream") is not True:
                raise RuntimeError("broker HTTPX external send options differ")
            if not is_broker_owned_httpx_client(client):
                raise RuntimeError("unowned HTTPX client reached the external adapter")
            route = BUILTIN_PROVIDER_ROUTES["supabase"]
            if (
                request.method != "GET"
                or request.url.host not in route.hosts
                or not any(
                    request.url.path.startswith(path)
                    for path in route.path_prefixes
                )
            ):
                raise RuntimeError(
                    "broker HTTPX external request differs from Supabase route"
                )
            required_headers = {
                name.lower()
                for name in (
                    route.credential_name,
                    *(alias for alias, _prefix in route.credential_header_aliases),
                )
                if name
            }
            if not required_headers <= set(request.headers):
                raise RuntimeError("broker HTTPX credential headers are incomplete")
            external_requests.append(request)
            body = b"[]"
            return httpx.Response(
                200,
                headers={
                    "content-length": str(len(body)),
                    "content-type": "application/json",
                },
                content=body,
                extensions={"network_stream": _NetworkStream()},
                request=request,
            )

        httpx.Client.send = strict_external_send
        broker = ProviderBrokerV2(
            credential_ref_hashes={
                slot: credential_reference_hash(value)
                for slot, value in credential_values.items()
            },
            retry_policy_hashes=retry_policy_hashes,
            transport=physical_transport,
            artifact_sink=vault.seal,
            clock=lambda: NOW,
        )
        broker.provision_credentials(credential_values)
        router = BrokeredProviderTransportV2(
            lambda _request: (_ for _ in ()).throw(
                RuntimeError("raw broker HTTPX request was recursively intercepted")
            )
        )
        router.install()

        job_id = "rehearsal:coordinator-broker-httpx"
        purpose = "research_lab.provider_preflight.v2"
        outcome_store = ProviderOutcomeStoreV2(
            broker=broker,
            vault=vault,
            sleeper=lambda _seconds: None,
        )
        sidecar = outcome_store.load_latest(
            utc_day=NOW[:10],
            job_id=job_id,
            purpose=purpose,
            operation_suffix="httpx-grant",
        )
        assigned_proxy_hash = sha256_json(
            {"credential": "coordinator-broker-httpx-assigned-proxy"}
        )
        execution_context = ExecutionContextV2(
            job_id=job_id,
            purpose=purpose,
            epoch_id=1,
            provider_credential_ref_hashes={
                "egress_proxy": assigned_proxy_hash,
            },
        )
        for attempt in sidecar.get("transport_attempts") or ():
            execution_context.record_transport(attempt)
        if (
            sidecar.get("found") is not False
            or not external_requests
            or len(execution_context.transport_attempts)
            != len(external_requests)
            or any(
                attempt["provider_id"] != "supabase"
                or attempt["egress_proxy_ref_hash"] != DIRECT_EGRESS_REF_HASH
                or not str(attempt["logical_operation_id"]).startswith(
                    job_id + ":provider-outcome:"
                )
                for attempt in execution_context.transport_attempts
            )
        ):
            raise RuntimeError("broker HTTPX Supabase sidecar evidence differs")

        probe_url = str(external_requests[0].url)

        def expect_sync_blocked(name: str, client: Any, *, grant: bool) -> None:
            before = len(external_requests)
            try:
                if grant:
                    with _broker_owned_httpx_send_scope(
                        client,
                        "GET",
                        probe_url,
                    ):
                        raise RuntimeError("blocked broker HTTPX client returned a response")
                else:
                    client.get(probe_url)
            except ProviderClientV2Error as exc:
                if "outside an attested job" not in str(exc):
                    raise
            else:
                raise RuntimeError(
                    "broker HTTPX bypass mutation was accepted: " + name
                )
            if len(external_requests) != before:
                raise RuntimeError(
                    "broker HTTPX bypass mutation reached external send: " + name
                )
            blocked[name] = True

        genuine_without_grant = physical_transport._new_client()
        sync_clients.append(genuine_without_grant)
        expect_sync_blocked(
            "genuine_client_without_grant",
            genuine_without_grant,
            grant=False,
        )

        def untrusted_send(request):
            untrusted_requests.append(request)
            return httpx.Response(200, content=b"{}", request=request)

        copied_marker = httpx.Client(
            transport=httpx.MockTransport(untrusted_send),
            trust_env=False,
            follow_redirects=False,
        )
        sync_clients.append(copied_marker)
        setattr(
            copied_marker,
            _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
            copied_marker._transport,
        )
        expect_sync_blocked("copied_marker", copied_marker, grant=True)

        transport_swap = physical_transport._new_client()
        sync_clients.append(transport_swap)
        transport_swap._transport = httpx.MockTransport(untrusted_send)
        expect_sync_blocked("transport_swap", transport_swap, grant=True)

        injected_mount = physical_transport._new_client()
        sync_clients.append(injected_mount)
        injected_mount._mounts = {object(): httpx.MockTransport(untrusted_send)}
        expect_sync_blocked("injected_mount", injected_mount, grant=True)

        redirect_enabled = physical_transport._new_client()
        sync_clients.append(redirect_enabled)
        redirect_enabled.follow_redirects = True
        expect_sync_blocked("redirect_enabled", redirect_enabled, grant=True)

        wrong_role = physical_transport._new_client()
        sync_clients.append(wrong_role)
        os.environ["LEADPOET_ENCLAVE_ROLE"] = next(
            role for role in ROLE_SPECS if role != RPC_COORDINATOR_ROLE
        )
        try:
            expect_sync_blocked("wrong_role", wrong_role, grant=True)
        finally:
            os.environ["LEADPOET_ENCLAVE_ROLE"] = RPC_COORDINATOR_ROLE

        async def async_untrusted_send(request):
            untrusted_requests.append(request)
            return httpx.Response(200, content=b"{}", request=request)

        async_client = httpx.AsyncClient(
            transport=httpx.MockTransport(async_untrusted_send),
            trust_env=False,
            follow_redirects=False,
        )
        setattr(
            async_client,
            _EXPLICIT_HTTP_TRANSPORT_ATTRIBUTE,
            async_client._transport,
        )

        async def expect_async_blocked() -> None:
            try:
                await async_client.get(probe_url)
            except ProviderClientV2Error as exc:
                if "outside an attested job" not in str(exc):
                    raise
            else:
                raise RuntimeError("async HTTPX marker bypass was accepted")

        asyncio.run(expect_async_blocked())
        blocked["async_client_marker"] = True
        if untrusted_requests:
            raise RuntimeError("broker HTTPX mutation reached an untrusted transport")
    finally:
        for client in reversed(sync_clients):
            _close_client_transports(client)
        if async_client is not None:
            asyncio.run(async_client.aclose())
        if router is not None:
            router.restore()
        httpx.Client.send = original_sync_send
        physical_transport.close()
        if previous_role is None:
            os.environ.pop("LEADPOET_ENCLAVE_ROLE", None)
        else:
            os.environ["LEADPOET_ENCLAVE_ROLE"] = previous_role

    evidence = {
        "coordinator_role_authority_bound": True,
        "real_broker_external_send_bound": True,
        "direct_supabase_sidecar_receipt_bound": True,
        "fail_closed_cases": blocked,
    }
    if not _coordinator_broker_httpx_evidence_is_complete(evidence):
        raise RuntimeError("broker HTTPX grant evidence is incomplete")
    return evidence


def _exercise_artifact_egress_sustained_readback() -> dict[str, Any]:
    """Exercise the exact artifact transport across both production relays."""

    from datetime import timedelta
    import errno
    import socket
    import ssl

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from gateway.tee.artifact_persistence_v2 import (
        ARTIFACT_VERIFICATION_TRANSPORT_MAX_IDLE_SECONDS,
        MAX_ARTIFACT_VERIFICATION_TRANSPORTS,
        _ArtifactVerificationTransportPool,
    )
    from gateway.tee.egress_framing import (
        EgressTunnelFramingError,
        TUNNEL_FRAME_BYTES,
        TUNNEL_FRAMING_MODE,
        receive_tunnel_frame,
        relay_raw_and_framed,
    )
    from gateway.tee.egress_proxy import EnclaveEgressProxy
    from gateway.tee.egress_proxy import (
        DEFAULT_IDLE_TIMEOUT_SECONDS as EGRESS_TUNNEL_IDLE_TIMEOUT_SECONDS,
    )
    from gateway.tee.provider_broker_v2 import HTTPXProviderTransport
    from gateway.utils.tee_client import AF_VSOCK, _recv_exact
    from gateway.utils.tee_egress_forwarder import _handle_connection

    # Keep the last request a GET so the exact relay path also exercises a
    # complete authenticated PostgREST-shaped JSON body whose terminal chunk
    # is lost when the provider closes first.
    request_count = max(65, int(MAX_ARTIFACT_VERIFICATION_TRANSPORTS) * 4 + 1)
    requests_seen: list[str] = []
    concurrent_requests_seen: set[str] = set()
    origin_errors: list[str] = []
    origin_threads: list[threading.Thread] = []
    parent_threads: list[threading.Thread] = []
    parent_tunnel_count = 0
    sequence_lock = threading.Lock()
    provider_first_close = threading.Event()
    concurrent_workers = min(8, int(MAX_ARTIFACT_VERIFICATION_TRANSPORTS))
    concurrent_requests_per_worker = 8

    payloads = []
    for ordinal in range(request_count):
        padding_bytes = TUNNEL_FRAME_BYTES + 1024 if ordinal % 17 == 0 else 128
        payloads.append(
            b'{"ordinal":'
            + str(ordinal).encode("ascii")
            + b',"padding":"'
            + (b"x" * padding_bytes)
            + b'"}'
        )

    with tempfile.TemporaryDirectory(prefix="leadpoet-artifact-egress-") as root:
        root_path = Path(root)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "example.com")]
        )
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now.replace(microsecond=0) - timedelta(minutes=1))
            .not_valid_after(now.replace(microsecond=0) + timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("example.com")]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )
        certificate_path = root_path / "origin-cert.pem"
        private_key_path = root_path / "origin-key.pem"
        certificate_path.write_bytes(
            certificate.public_bytes(serialization.Encoding.PEM)
        )
        private_key_path.write_bytes(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls_context.load_cert_chain(
            certfile=str(certificate_path),
            keyfile=str(private_key_path),
        )

        def serve_origin(raw_connection: socket.socket) -> None:
            try:
                with raw_connection:
                    with tls_context.wrap_socket(
                        raw_connection,
                        server_side=True,
                    ) as protected:
                        pending = bytearray()
                        while True:
                            while b"\r\n\r\n" not in pending:
                                chunk = protected.recv(16 * 1024)
                                if not chunk:
                                    return
                                pending.extend(chunk)
                            encoded_headers, remainder = bytes(pending).split(
                                b"\r\n\r\n",
                                1,
                            )
                            pending = bytearray(remainder)
                            request_line = encoded_headers.split(b"\r\n", 1)[0]
                            parts = request_line.split(b" ")
                            if len(parts) != 3:
                                raise RuntimeError("artifact request line differs")
                            method = parts[0].decode("ascii")
                            target = parts[1].decode("ascii")
                            if target.startswith("/concurrent/"):
                                target_parts = target.split("/")
                                if (
                                    method != "GET"
                                    or len(target_parts) != 4
                                    or not target_parts[2].isdigit()
                                    or not target_parts[3].isdigit()
                                ):
                                    raise RuntimeError(
                                        "concurrent artifact request differs"
                                    )
                                worker = int(target_parts[2])
                                worker_ordinal = int(target_parts[3])
                                if (
                                    worker >= concurrent_workers
                                    or worker_ordinal >= concurrent_requests_per_worker
                                ):
                                    raise RuntimeError(
                                        "concurrent artifact request exceeds contract"
                                    )
                                identity = f"{worker}:{worker_ordinal}"
                                with sequence_lock:
                                    if identity in concurrent_requests_seen:
                                        raise RuntimeError(
                                            "concurrent artifact request duplicated"
                                        )
                                    concurrent_requests_seen.add(identity)
                                payload = (
                                    b'{"concurrent":"'
                                    + identity.encode("ascii")
                                    + b'"}'
                                )
                                final_response = (
                                    worker_ordinal + 1
                                    == concurrent_requests_per_worker
                                )
                                protected.sendall(
                                    b"HTTP/1.1 200 OK\r\n"
                                    b"Content-Type: application/json\r\n"
                                    b"Content-Length: "
                                    + str(len(payload)).encode("ascii")
                                    + b"\r\nConnection: "
                                    + (
                                        b"close"
                                        if final_response
                                        else b"keep-alive"
                                    )
                                    + b"\r\n\r\n"
                                    + payload
                                )
                                if final_response:
                                    return
                                continue
                            with sequence_lock:
                                ordinal = len(requests_seen)
                                if ordinal >= request_count:
                                    raise RuntimeError(
                                        "artifact transport exceeded request contract"
                                    )
                                expected_method = "GET" if ordinal % 2 == 0 else "HEAD"
                                if method != expected_method:
                                    raise RuntimeError("artifact method sequence differs")
                                requests_seen.append(method)
                            payload = payloads[ordinal]
                            final_response = ordinal + 1 == request_count
                            response_body = payload if method == "GET" else b""
                            if final_response:
                                if method != "GET":
                                    raise RuntimeError(
                                        "final artifact request must be GET"
                                    )
                                protected.sendall(
                                    b"HTTP/1.1 200 OK\r\n"
                                    b"Content-Type: application/json\r\n"
                                    b"Transfer-Encoding: chunked\r\n"
                                    b"Connection: close\r\n\r\n"
                                    + format(len(response_body), "x").encode("ascii")
                                    + b"\r\n"
                                    + response_body
                                    + b"\r\n"
                                )
                            else:
                                protected.sendall(
                                    b"HTTP/1.1 200 OK\r\n"
                                    b"Content-Type: application/json\r\n"
                                    b"Content-Length: "
                                    + str(len(payload)).encode("ascii")
                                    + b"\r\nConnection: keep-alive\r\n\r\n"
                                    + response_body
                                )
                            if final_response:
                                provider_first_close.set()
                                return
            except Exception as exc:
                origin_errors.append(type(exc).__name__ + ":" + str(exc))

        def connect_origin(_host: str, _port: int) -> socket.socket:
            provider_side, origin_side = socket.socketpair()
            thread = threading.Thread(
                target=serve_origin,
                args=(origin_side,),
                name="rehearsal-artifact-origin",
                daemon=True,
            )
            origin_threads.append(thread)
            thread.start()
            return provider_side

        class ConnectedVsock:
            def __init__(self, connection: socket.socket) -> None:
                self._connection = connection

            def connect(self, _address: object) -> None:
                return None

            def fileno(self) -> int:
                return self._connection.fileno()

            def recv(self, size: int) -> bytes:
                return self._connection.recv(size)

            def sendall(self, payload: bytes) -> None:
                self._connection.sendall(payload)

            def shutdown(self, how: int) -> None:
                self._connection.shutdown(how)

            def close(self) -> None:
                self._connection.close()

        def socket_factory(
            family: int,
            socket_type: int,
            protocol: int = 0,
        ) -> Any:
            nonlocal parent_tunnel_count
            if family != AF_VSOCK:
                return socket.socket(family, socket_type, protocol)
            enclave_side, parent_side = socket.socketpair()
            with sequence_lock:
                parent_tunnel_count += 1
            thread = threading.Thread(
                target=_handle_connection,
                kwargs={
                    "connection": parent_side,
                    "connector": connect_origin,
                    "idle_timeout_seconds": 5.0,
                },
                name="rehearsal-artifact-parent-forwarder",
                daemon=True,
            )
            parent_threads.append(thread)
            thread.start()
            return ConnectedVsock(enclave_side)

        port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            port_probe.bind(("127.0.0.1", 0))
            proxy_port = int(port_probe.getsockname()[1])
        finally:
            port_probe.close()

        proxy = EnclaveEgressProxy(
            recv_exact=_recv_exact,
            local_port=proxy_port,
            socket_factory=socket_factory,
            loopback_initializer=lambda: None,
            idle_timeout_seconds=5.0,
        )
        proxy._configure_environment = lambda: None  # type: ignore[method-assign]
        transport = HTTPXProviderTransport(
            proxy_url=f"http://127.0.0.1:{proxy_port}",
            ca_bundle=str(certificate_path),
            response_body_ceiling_bytes=2 * TUNNEL_FRAME_BYTES,
            allow_authenticated_complete_body_eof=True,
            parent_tunnel_framing=TUNNEL_FRAMING_MODE,
            reuse_direct_connections=True,
        )
        results: list[dict[str, Any]] = []
        try:
            proxy.start()
            for ordinal in range(request_count):
                method = "GET" if ordinal % 2 == 0 else "HEAD"
                result = transport(
                    method=method,
                    url=f"https://example.com/artifacts/{ordinal}",
                    headers={"accept": "application/json"},
                    body=b"",
                    timeout_ms=5_000,
                    max_response_bytes=2 * TUNNEL_FRAME_BYTES,
                )
                expected_body = payloads[ordinal] if method == "GET" else b""
                if result.get("http_status") != 200 or result.get("body") != expected_body:
                    raise RuntimeError("artifact transport readback differs")
                results.append(result)
            if not provider_first_close.wait(timeout=2):
                raise RuntimeError("provider-first close was not observed")

            def exercise_concurrent_transport(worker: int) -> int:
                concurrent_transport = HTTPXProviderTransport(
                    proxy_url=f"http://127.0.0.1:{proxy_port}",
                    ca_bundle=str(certificate_path),
                    response_body_ceiling_bytes=2 * TUNNEL_FRAME_BYTES,
                    allow_authenticated_complete_body_eof=True,
                    parent_tunnel_framing=TUNNEL_FRAMING_MODE,
                    reuse_direct_connections=True,
                )
                try:
                    for ordinal in range(concurrent_requests_per_worker):
                        expected = (
                            b'{"concurrent":"'
                            + f"{worker}:{ordinal}".encode("ascii")
                            + b'"}'
                        )
                        response = concurrent_transport(
                            method="GET",
                            url=(
                                "https://example.com/concurrent/"
                                f"{worker}/{ordinal}"
                            ),
                            headers={"accept": "application/json"},
                            body=b"",
                            timeout_ms=5_000,
                            max_response_bytes=2 * TUNNEL_FRAME_BYTES,
                        )
                        if (
                            response.get("http_status") != 200
                            or response.get("body") != expected
                        ):
                            raise RuntimeError(
                                "concurrent artifact readback differs"
                            )
                    return concurrent_requests_per_worker
                finally:
                    concurrent_transport.close()

            with ThreadPoolExecutor(max_workers=concurrent_workers) as executor:
                concurrent_counts = list(
                    executor.map(
                        exercise_concurrent_transport,
                        range(concurrent_workers),
                    )
                )
        finally:
            transport.close()
            proxy_status = proxy.status()
            proxy.stop()
            for thread in parent_threads + origin_threads:
                thread.join(timeout=3)

    framed, truncated_peer = socket.socketpair()
    truncated_rejected = False
    try:
        truncated_peer.sendall((8).to_bytes(4, "big") + b"short")
        truncated_peer.shutdown(socket.SHUT_WR)
        try:
            receive_tunnel_frame(
                framed,
                deadline=time.monotonic() + 1,
            )
        except EgressTunnelFramingError:
            truncated_rejected = True
    finally:
        framed.close()
        truncated_peer.close()
    if not truncated_rejected:
        raise RuntimeError("truncated artifact tunnel frame was accepted")

    def exercise_provider_first_idle_client(index: int) -> bool:
        client, enclave_raw = socket.socketpair()
        enclave_framed, parent_framed = socket.socketpair()
        parent_raw, provider = socket.socketpair()
        relay_errors: list[str] = []

        def run_relay(*args: Any, **kwargs: Any) -> None:
            try:
                relay_raw_and_framed(*args, **kwargs)
            except Exception as exc:
                relay_errors.append(type(exc).__name__ + ":" + str(exc))

        enclave_thread = threading.Thread(
            target=run_relay,
            args=(enclave_raw, enclave_framed),
            kwargs={
                "idle_timeout_seconds": 1.0,
                "max_bytes_per_direction": 1024,
                "raw_label": "client",
                "framed_label": "parent",
                "terminal_initiator": True,
            },
            daemon=True,
        )
        parent_thread = threading.Thread(
            target=run_relay,
            args=(parent_raw, parent_framed),
            kwargs={
                "idle_timeout_seconds": 1.0,
                "max_bytes_per_direction": 1024,
                "raw_label": "provider",
                "framed_label": "enclave",
                "terminal_initiator": False,
            },
            daemon=True,
        )
        enclave_thread.start()
        parent_thread.start()
        request = ("request-%d" % index).encode("ascii")
        response = ("response-%d" % index).encode("ascii")
        try:
            client.sendall(request)
            if _recv_exact(provider, len(request)) != request:
                raise RuntimeError("provider-first idle request differs")
            provider.sendall(response)
            provider.close()
            if _recv_exact(client, len(response)) != response or client.recv(1):
                raise RuntimeError("provider-first idle response differs")

            # Deliberately keep the pooled client write side open. Production
            # S3 can close an otherwise idle keep-alive connection after the
            # complete authenticated response, and tunnel termination must not
            # depend on the HTTP pool making another request.
            enclave_thread.join(timeout=2)
            parent_thread.join(timeout=2)
        finally:
            for connection in (
                client,
                enclave_raw,
                enclave_framed,
                parent_framed,
                parent_raw,
            ):
                connection.close()
        if relay_errors or enclave_thread.is_alive() or parent_thread.is_alive():
            raise RuntimeError(
                "provider-first idle framed terminal handshake failed: "
                + json.dumps(relay_errors, sort_keys=True)
            )
        return True

    with ThreadPoolExecutor(max_workers=8) as executor:
        provider_first_idle_results = list(
            executor.map(exercise_provider_first_idle_client, range(32))
        )
    if provider_first_idle_results != [True] * 32:
        raise RuntimeError("provider-first idle framed tunnel result differs")

    class LifecycleTransport:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    idle_now = [0.0]
    lifecycle_transports: list[LifecycleTransport] = []
    lifecycle_pool = _ArtifactVerificationTransportPool(
        maximum_transports=1,
        wait_seconds=1,
        idle_clock=lambda: idle_now[0],
    )

    def new_lifecycle_transport() -> LifecycleTransport:
        instance = LifecycleTransport()
        lifecycle_transports.append(instance)
        return instance

    lifecycle_pool._new_transport = new_lifecycle_transport  # type: ignore[method-assign]
    stale_transport = lifecycle_pool.acquire()
    lifecycle_pool.release(stale_transport)
    idle_now[0] = ARTIFACT_VERIFICATION_TRANSPORT_MAX_IDLE_SECONDS
    replacement_transport = lifecycle_pool.acquire()
    stale_transport_evicted = (
        replacement_transport is not stale_transport
        and stale_transport.closed is True
        and replacement_transport.closed is False
        and ARTIFACT_VERIFICATION_TRANSPORT_MAX_IDLE_SECONDS
        < EGRESS_TUNNEL_IDLE_TIMEOUT_SECONDS
    )
    lifecycle_pool.release(replacement_transport, failed=True)

    failed_generation_transports: list[LifecycleTransport] = []
    failed_generation_pool = _ArtifactVerificationTransportPool(
        maximum_transports=4,
        wait_seconds=1,
    )

    def new_failed_generation_transport() -> LifecycleTransport:
        instance = LifecycleTransport()
        failed_generation_transports.append(instance)
        return instance

    failed_generation_pool._new_transport = (  # type: ignore[method-assign]
        new_failed_generation_transport
    )
    peer_closed_generation = [
        failed_generation_pool.acquire() for _index in range(4)
    ]
    for pooled_transport in peer_closed_generation:
        failed_generation_pool.release(pooled_transport)
    failed_lease = failed_generation_pool.acquire()
    failed_generation_pool.release(failed_lease, failed=True)
    fresh_after_failure = failed_generation_pool.acquire()
    failed_generation_evicted = (
        all(item.closed is True for item in peer_closed_generation)
        and fresh_after_failure not in peer_closed_generation
        and fresh_after_failure.closed is False
    )
    failed_generation_pool.release(fresh_after_failure, failed=True)

    ordinary_clients: list[Any] = []

    class OrdinaryTLS:
        def getpeercert(self, binary_form: bool = False) -> bytes:
            if not binary_form:
                raise RuntimeError("binary peer certificate was not requested")
            return b"ordinary-provider-peer"

        def version(self) -> str:
            return "TLSv1.3"

    class OrdinaryNetworkStream:
        def __init__(self) -> None:
            self.closed = False

        def get_extra_info(self, name: str) -> Any:
            if name != "ssl_object":
                raise RuntimeError("unexpected network metadata request")
            return OrdinaryTLS()

        def close(self) -> None:
            self.closed = True

    class OrdinaryExplicitTransport:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class OrdinaryResponseContext:
        def __init__(self, *, fail: bool) -> None:
            self.fail = fail

        def __enter__(self) -> Any:
            if self.fail:
                raise RuntimeError("expired relay generation")
            return SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                extensions={"network_stream": OrdinaryNetworkStream()},
                iter_bytes=lambda: iter((b'{"ok":true}',)),
            )

        def __exit__(self, *_args: Any) -> bool:
            return False

    class OrdinaryClient:
        def __init__(self) -> None:
            self.index = len(ordinary_clients)
            self.closed = False
            self._leadpoet_explicit_http_transport = (
                OrdinaryExplicitTransport()
            )
            ordinary_clients.append(self)

        def stream(self, *_args: Any, **_kwargs: Any) -> OrdinaryResponseContext:
            return OrdinaryResponseContext(fail=self.index == 0)

        def close(self) -> None:
            self.closed = True

    ordinary_transport = HTTPXProviderTransport(
        allow_authenticated_complete_body_eof=True,
        reuse_direct_connections=False,
    )
    ordinary_transport._new_client = (  # type: ignore[method-assign]
        lambda **_kwargs: OrdinaryClient()
    )
    ordinary_request = {
        "method": "GET",
        "url": "https://example.com/weight-input",
        "headers": {"accept": "application/json"},
        "body": b"",
        "timeout_ms": 1_000,
    }
    try:
        first_direct_failed = False
        try:
            ordinary_transport(**ordinary_request)
        except RuntimeError as exc:
            first_direct_failed = str(exc) == "expired relay generation"
        second_direct = ordinary_transport(**ordinary_request)
        ordinary_transport_generation_safe = (
            ordinary_transport.allow_authenticated_complete_body_eof
            and ordinary_transport.parent_tunnel_framing == ""
            and ordinary_transport.upstream_parent_tunnel_framing == ""
            and not ordinary_transport.reuse_direct_connections
            and ordinary_transport._direct_request_slot is not None
        )
        ordinary_direct_generation_recovered = (
            first_direct_failed
            and second_direct.get("body") == b'{"ok":true}'
            and len(ordinary_clients) == 2
            and ordinary_clients[0].closed is True
            and ordinary_clients[1].closed is True
            and ordinary_transport._direct_client is None
        )
    finally:
        ordinary_transport.close()
    ordinary_direct_cleanup = all(client.closed for client in ordinary_clients)
    if (
        origin_errors
        or len(results) != request_count
        or len(requests_seen) != request_count
        or len(concurrent_requests_seen)
        != concurrent_workers * concurrent_requests_per_worker
        or concurrent_counts
        != [concurrent_requests_per_worker] * concurrent_workers
        or parent_tunnel_count != 1 + concurrent_workers
        or proxy_status.get("last_failure")
        or not ordinary_transport_generation_safe
        or not ordinary_direct_generation_recovered
        or not ordinary_direct_cleanup
        or not stale_transport_evicted
        or not failed_generation_evicted
    ):
        raise RuntimeError(
            "sustained artifact egress contract failed: "
            + json.dumps(
                {
                    "origin_errors": origin_errors,
                    "result_count": len(results),
                    "request_count": len(requests_seen),
                    "concurrent_request_count": len(concurrent_requests_seen),
                    "parent_tunnel_count": parent_tunnel_count,
                    "proxy_failure": proxy_status.get("last_failure"),
                    "ordinary_transport_generation_safe": (
                        ordinary_transport_generation_safe
                    ),
                    "ordinary_direct_generation_recovered": (
                        ordinary_direct_generation_recovered
                    ),
                    "ordinary_direct_cleanup": ordinary_direct_cleanup,
                    "stale_transport_evicted": stale_transport_evicted,
                    "failed_generation_evicted": failed_generation_evicted,
                },
                sort_keys=True,
            )
        )
    return {
        "exact_transport_proxy_forwarder_path": True,
        "sustained_single_tunnel_reused": True,
        "bounded_concurrent_tunnels_verified": True,
        "multi_frame_response_verified": True,
        "provider_first_close_verified": True,
        "provider_first_idle_terminal_handshake_verified": True,
        "complete_chunked_json_eof_recovered": True,
        "truncated_frame_rejected": True,
        "ordinary_provider_transport_request_scoped": True,
        "ordinary_direct_serialized_generation_recovery_verified": True,
        "stale_pooled_transport_evicted_before_relay_timeout": True,
        "failed_pooled_generation_evicted_before_retry": True,
        "request_count": request_count,
        "concurrent_request_count": len(concurrent_requests_seen),
    }


def _exercise_company_fit_numeric_observation_projection() -> dict[str, Any]:
    from gateway.qualification.models import CompanyOutput, ICPPrompt
    from qualification.scoring.company_fit_decision import (
        COMPANY_FIT_MATCH,
        COMPANY_FIT_MISMATCH,
        COMPANY_FIT_UNAVAILABLE,
    )
    from qualification.scoring.lead_scorer import (
        _decision_from_observed_employee_size,
        _reverify_decision,
    )

    company = CompanyOutput(
        company_name="Example Inc",
        company_website="https://example.com",
        company_linkedin="https://www.linkedin.com/company/example-inc",
        industry="Software",
        employee_count="11-50",
        country="United States",
        intent_signals=[
            {
                "description": "Example announced an active evaluation.",
                "source": "news",
                "url": "https://example.com/news/evaluation",
                "date": "2026-08-15",
                "snippet": "Example is evaluating the relevant product.",
            }
        ],
    )
    icp = ICPPrompt(
        icp_id="rehearsal-numeric-employee-observation",
        prompt="test",
        industry="Software",
        sub_industry="SaaS",
        employee_count="11-50",
        company_stage="",
        geography="United States",
        product_service="test",
    )
    verdict = {
        "observed_company_name": "Example Inc",
        "observed_company_website": "https://example.com",
        "observed_company_linkedin": (
            "https://www.linkedin.com/company/example-inc"
        ),
        "observed_employee_count": "50",
        "employee_size_matches": True,
        "employee_size_evidence_url": "https://example.com/about",
        "employee_size_evidence_quote": "Example has fifty employees.",
        "observed_industry": "Software",
        "observed_subindustry": "SaaS",
        "industry_matches": True,
        "industry_evidence_url": "https://example.com/about",
        "industry_evidence_quote": "Example builds software.",
        "observed_hq_country": "United States",
        "observed_hq_state": "",
        "geography_matches": True,
        "geography_evidence_url": "https://example.com/about",
        "geography_evidence_quote": "Example is based in the United States.",
        "reason": "verified production-shaped company fit",
    }
    matched = _reverify_decision(
        verdict,
        "",
        "",
        icp=icp,
        company=company,
    )
    provider_observations = dict(
        (matched.details or {}).get("provider_observations") or {}
    )
    inconsistent = dict(verdict)
    inconsistent["employee_size_matches"] = False
    contradicted = dict(inconsistent)
    contradicted["observed_employee_count"] = "51"
    malformed = ("50.0", "1-10", -1)
    if (
        matched.decision != COMPANY_FIT_MATCH
        or provider_observations.get("observed_employee_count") != "50"
        or _decision_from_observed_employee_size(contradicted, icp)
        != COMPANY_FIT_MISMATCH
        or _decision_from_observed_employee_size(inconsistent, icp)
        != COMPANY_FIT_UNAVAILABLE
        or any(
            _decision_from_observed_employee_size(
                {
                    "observed_employee_count": value,
                    "employee_size_matches": True,
                },
                icp,
            )
            != COMPANY_FIT_UNAVAILABLE
            for value in malformed
        )
    ):
        raise RuntimeError("numeric employee observation contract failed")
    return {
        "numeric_observation_with_web_evidence_matched": True,
        "raw_observation_committed": True,
        "contradictory_boolean_failed_closed": True,
        "malformed_range_decimal_negative_failed_closed": True,
    }


def _company_fit_numeric_observation_projection_evidence_is_complete(
    evidence: Mapping[str, Any],
) -> bool:
    return all(
        evidence.get(field) is True
        for field in (
            "numeric_observation_with_web_evidence_matched",
            "raw_observation_committed",
            "contradictory_boolean_failed_closed",
            "malformed_range_decimal_negative_failed_closed",
        )
    )


def _exercise_measured_raw_provider_transport(
    *,
    assigned_proxy: bool,
) -> dict[str, Any]:
    """Exercise fresh direct or assigned raw tunnels through production seams."""

    from datetime import timedelta
    import socket
    import ssl

    import certifi
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    from gateway.tee.egress_framing import TUNNEL_FRAMING_MODE
    from gateway.tee.egress_proxy import EnclaveEgressProxy, _relay_bidirectional
    from gateway.tee.provider_broker_v2 import HTTPXProviderTransport
    from gateway.utils.tee_client import AF_VSOCK, _recv_exact
    from gateway.utils.tee_egress_forwarder import _handle_connection

    observed: dict[str, Any] = {}
    errors: list[str] = []
    parent_destinations: list[tuple[str, int]] = []
    parent_threads: list[threading.Thread] = []
    request_count = 8
    failed_attempt_ordinal = 4
    attempt_count = request_count + 1
    upstream_proxy_port = 18080

    def process_fd_count() -> int | None:
        for path in ("/proc/self/fd", "/dev/fd"):
            try:
                return sum(name.isdigit() for name in os.listdir(path))
            except OSError:
                continue
        return None

    fd_count_before = process_fd_count()

    with tempfile.TemporaryDirectory(prefix="leadpoet-upstream-proxy-") as root:
        root_path = Path(root)
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "example.com")]
        )
        now = datetime.now(timezone.utc)
        certificate = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(private_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now.replace(microsecond=0) - timedelta(minutes=1))
            .not_valid_after(now.replace(microsecond=0) + timedelta(hours=1))
            .add_extension(
                x509.SubjectAlternativeName([x509.DNSName("example.com")]),
                critical=False,
            )
            .sign(private_key, hashes.SHA256())
        )
        certificate_path = root_path / "transport-cert.pem"
        private_key_path = root_path / "transport-key.pem"
        certificate_path.write_bytes(
            certificate.public_bytes(serialization.Encoding.PEM)
        )
        private_key_path.write_bytes(
            private_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        provider_tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        provider_tls.load_cert_chain(
            certfile=str(certificate_path),
            keyfile=str(private_key_path),
        )
        provider_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        provider_listener.bind(("127.0.0.1", 0))
        provider_listener.listen(1)
        provider_address = provider_listener.getsockname()
        upstream_listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        upstream_listener.bind(("127.0.0.1", 0))
        upstream_listener.listen(1)
        upstream_address = upstream_listener.getsockname()

        def serve_provider() -> None:
            try:
                requests = observed.setdefault("provider_requests", [])
                for _ordinal in range(request_count):
                    connection, _address = provider_listener.accept()
                    protected = provider_tls.wrap_socket(
                        connection, server_side=True
                    )
                    try:
                        request = bytearray()
                        while b"\r\n\r\n" not in request:
                            chunk = protected.recv(4096)
                            if not chunk:
                                raise RuntimeError("provider request ended early")
                            request.extend(chunk)
                        requests.append(bytes(request))
                        protected.sendall(
                            b"HTTP/1.1 200 OK\r\n"
                            b"Content-Type: application/json\r\n"
                            b"Content-Length: 11\r\n"
                            b"Connection: close\r\n\r\n"
                            b'{"ok":true}'
                        )
                    finally:
                        protected.close()
            except Exception as exc:
                errors.append(type(exc).__name__ + ":" + str(exc))
            finally:
                provider_listener.close()

        def serve_upstream_proxy() -> None:
            if not assigned_proxy:
                upstream_listener.close()
                return
            try:
                proxy_requests = observed.setdefault("proxy_headers", [])
                for _ordinal in range(request_count):
                    connection = None
                    provider = None
                    try:
                        connection, _address = upstream_listener.accept()
                        headers = bytearray()
                        while b"\r\n\r\n" not in headers:
                            chunk = connection.recv(4096)
                            if not chunk:
                                raise RuntimeError("proxy CONNECT ended early")
                            headers.extend(chunk)
                        proxy_requests.append(bytes(headers))
                        connection.sendall(
                            b"HTTP/1.1 200 Connection Established\r\n\r\n"
                        )
                        provider = socket.create_connection(
                            provider_address, timeout=2
                        )
                        _relay_bidirectional(
                            connection,
                            provider,
                            idle_timeout_seconds=5,
                        )
                    finally:
                        for candidate in (connection, provider):
                            if candidate is not None:
                                try:
                                    candidate.close()
                                except Exception:
                                    pass
            except Exception as exc:
                errors.append(type(exc).__name__ + ":" + str(exc))
            finally:
                upstream_listener.close()

        provider_thread = threading.Thread(
            target=serve_provider,
            name="rehearsal-measured-provider",
            daemon=True,
        )
        upstream_thread = threading.Thread(
            target=serve_upstream_proxy,
            name="rehearsal-measured-http-connect-proxy",
            daemon=True,
        )
        provider_thread.start()
        upstream_thread.start()

        class ConnectedVsock:
            def __init__(self, connection: socket.socket) -> None:
                self._connection = connection

            def connect(self, _address: Any) -> None:
                return None

            def fileno(self) -> int:
                return self._connection.fileno()

            def recv(self, size: int) -> bytes:
                return self._connection.recv(size)

            def sendall(self, payload: bytes) -> None:
                self._connection.sendall(payload)

            def shutdown(self, how: int) -> None:
                self._connection.shutdown(how)

            def close(self) -> None:
                self._connection.close()

            def __getattr__(self, name: str) -> Any:
                return getattr(self._connection, name)

        def connect_upstream(host: str, port: int) -> socket.socket:
            parent_destinations.append((host, port))
            if len(parent_destinations) - 1 == failed_attempt_ordinal:
                raise OSError(errno.EADDRNOTAVAIL, "local endpoint exhausted")
            destination = upstream_address if assigned_proxy else provider_address
            return socket.create_connection(destination, timeout=2)

        def socket_factory(
            family: int,
            socket_type: int,
            protocol: int = 0,
        ) -> Any:
            if family != AF_VSOCK:
                return socket.socket(family, socket_type, protocol)
            enclave_side, parent_side = socket.socketpair()
            thread = threading.Thread(
                target=_handle_connection,
                kwargs={
                    "connection": parent_side,
                    "connector": connect_upstream,
                    "idle_timeout_seconds": 5,
                },
                name="rehearsal-measured-parent-forwarder",
                daemon=True,
            )
            parent_threads.append(thread)
            thread.start()
            return ConnectedVsock(enclave_side)

        port_probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            port_probe.bind(("127.0.0.1", 0))
            local_proxy_port = int(port_probe.getsockname()[1])
        finally:
            port_probe.close()
        enclave_proxy = EnclaveEgressProxy(
            recv_exact=_recv_exact,
            local_port=local_proxy_port,
            socket_factory=socket_factory,
            loopback_initializer=lambda: None,
            idle_timeout_seconds=5,
        )
        enclave_proxy._configure_environment = lambda: None  # type: ignore[method-assign]
        transport = HTTPXProviderTransport(
            proxy_url=f"http://127.0.0.1:{local_proxy_port}",
            ca_bundle=str(certificate_path),
            allow_authenticated_complete_body_eof=True,
            parent_tunnel_framing="",
            upstream_parent_tunnel_framing="",
            reuse_direct_connections=False,
            reuse_upstream_proxy_connections=False,
        )
        connection_scope = sha256_json(
            {
                "schema_version": "leadpoet.provider_connection_scope.v2",
                "job_id": "rehearsal-scoring-job",
                "egress_proxy_ref_hash": sha256_json(
                    {
                        "schema_version": "leadpoet.rehearsal_proxy_ref.v1",
                        "profile": "assigned-scoring-proxy",
                    }
                ),
            }
        )
        original_certifi_where = certifi.where
        try:
            certifi.where = lambda: str(certificate_path)  # type: ignore[assignment]
            enclave_proxy.start()
            results = []
            successful_ordinals = []
            expected_failures = []
            midflight_health = None
            for ordinal in range(attempt_count):
                request_kwargs = {
                    "method": "GET",
                    "url": f"https://example.com/search/{ordinal}",
                    "headers": {"accept": "application/json"},
                    "body": b"",
                    "timeout_ms": 5_000,
                }
                if assigned_proxy:
                    request_kwargs.update(
                        {
                            "upstream_proxy_url": (
                                "http://rehearsal-worker:rehearsal-secret@"
                                f"example.com:{upstream_proxy_port}"
                            ),
                            "connection_scope": connection_scope,
                        }
                    )
                try:
                    results.append(transport(**request_kwargs))
                    successful_ordinals.append(ordinal)
                except Exception as exc:
                    if ordinal != failed_attempt_ordinal:
                        raise
                    expected_failures.append(type(exc).__name__)
                if ordinal == failed_attempt_ordinal:
                    midflight_health = transport.health()
        finally:
            certifi.where = original_certifi_where  # type: ignore[assignment]
            transport.close()
            for _index in range(100):
                proxy_status = enclave_proxy.status()
                if proxy_status.get("active_tunnel_count") == 0:
                    break
                time.sleep(0.01)
            enclave_proxy.stop()
            provider_thread.join(timeout=5)
            upstream_thread.join(timeout=5)
            for thread in parent_threads:
                thread.join(timeout=5)

    fd_count_after = process_fd_count()

    proxy_headers = [
        bytes(value) for value in (observed.get("proxy_headers") or [])
    ]
    provider_requests = [
        bytes(value) for value in (observed.get("provider_requests") or [])
    ]
    if (
        errors
        or len(results) != request_count
        or len(expected_failures) != 1
        or any(result.get("http_status") != 200 for result in results)
        or any(result.get("body") != b'{"ok":true}' for result in results)
        or any(
            not str(result.get("tls_protocol") or "").startswith("TLSv1.")
            for result in results
        )
        or parent_destinations
        != [
            (
                "example.com",
                upstream_proxy_port if assigned_proxy else 443,
            )
        ]
        * attempt_count
        or len(proxy_headers) != (request_count if assigned_proxy else 0)
        or any(
            not headers.startswith(b"CONNECT example.com:443 HTTP/1.1")
            for headers in proxy_headers
        )
        or any(
            b"Proxy-Authorization: Basic " not in headers
            for headers in proxy_headers
        )
        or len(provider_requests) != request_count
        or any(
            not request.startswith(
                f"GET /search/{ordinal} HTTP/".encode("ascii")
            )
            for ordinal, request in zip(successful_ordinals, provider_requests)
        )
        or transport.reuse_direct_connections
        or transport.reuse_upstream_proxy_connections
        or not isinstance(midflight_health, dict)
        or midflight_health.get("direct_active_scope_count") != 0
        or midflight_health.get("assigned_active_scope_count") != 0
        or midflight_health.get("last_failure", {}).get("route")
        != ("assigned_proxy" if assigned_proxy else "direct")
        or midflight_health.get("last_failure", {}).get("failure_code")
        != "proxy_failure"
        or proxy_status.get("accepted_tunnel_count") != attempt_count
        or proxy_status.get("active_tunnel_count") != 0
        or proxy_status.get("completed_tunnel_count") != request_count
        or proxy_status.get("failed_tunnel_count") != 1
        or (
            fd_count_before is not None
            and fd_count_after is not None
            and fd_count_after > fd_count_before + 1
        )
        or provider_thread.is_alive()
        or upstream_thread.is_alive()
        or any(thread.is_alive() for thread in parent_threads)
    ):
        raise RuntimeError(
            "measured raw provider transport contract failed: "
            + json.dumps(
                {
                    "errors": errors,
                    "http_statuses": [result.get("http_status") for result in results],
                    "parent_destinations": parent_destinations,
                    "proxy_connect_count": sum(
                        headers.startswith(b"CONNECT example.com:443 HTTP/1.1")
                        for headers in proxy_headers
                    ),
                    "proxy_auth_count": sum(
                        b"Proxy-Authorization: Basic " in headers
                        for headers in proxy_headers
                    ),
                    "provider_request_count": len(provider_requests),
                    "proxy_failure": proxy_status.get("last_failure"),
                },
                sort_keys=True,
            )
        )
    if any(b"rehearsal-secret" in headers for headers in proxy_headers):
        raise RuntimeError("upstream proxy credential escaped Basic auth encoding")
    return {
        "exact_httpx_enclave_parent_proxy_provider_path": True,
        "parent_tunnel_framing": transport.parent_tunnel_framing,
        "upstream_parent_tunnel_framing": (
            transport.upstream_parent_tunnel_framing
        ),
        "nested_tls_verified": True,
        "proxy_auth_remained_in_enclave": True,
        "provider_first_close_verified": True,
        "bounded_cleanup_verified": True,
        "production_http_connect_proxy_verified": True,
        "request_scoped_connection_cleanup_verified": True,
        "one_connect_per_request_verified": True,
        "failure_recovery_on_fresh_tunnel_verified": True,
        "classified_failure_health_verified": True,
        "stable_process_resource_count_verified": True,
        "repeated_request_count": request_count,
        "attempt_count": attempt_count,
    }


def _exercise_measured_assigned_proxy_raw_transport() -> dict[str, Any]:
    evidence = _exercise_measured_raw_provider_transport(
        assigned_proxy=True,
    )
    if evidence.pop("parent_tunnel_framing"):
        raise RuntimeError("direct coordinator transport was not raw")
    if evidence.pop("upstream_parent_tunnel_framing"):
        raise RuntimeError("assigned proxy parent transport was not raw")
    evidence["assigned_proxy_raw_parent_tunnel_verified"] = True
    return evidence


def _exercise_measured_coordinator_raw_transport() -> dict[str, Any]:
    evidence = _exercise_measured_raw_provider_transport(
        assigned_proxy=False,
    )
    if evidence.pop("parent_tunnel_framing"):
        raise RuntimeError("measured coordinator raw transport was not exercised")
    if evidence.pop("upstream_parent_tunnel_framing"):
        raise RuntimeError("measured coordinator upstream transport was not raw")
    evidence["raw_parent_tunnel_verified"] = True
    return evidence


_RESTART_SUMMARY_DEADLINE_EVIDENCE_FIELDS = (
    "active_timeout_terminal",
    "host_module_source_exact",
    "later_failure_overrides_stale_completion",
    "passive_wait_does_not_relabel_failure",
    "restart_invocation_identity_exact",
    "successful_slow_wait_nonterminal",
    "successful_duration_metrics_retained",
)


def _restart_summary_deadline_evidence_is_complete(
    value: Any,
    *,
    candidate_sha: str | None = None,
) -> bool:
    normalized_sha = str(
        candidate_sha
        or os.environ.get("REHEARSAL_CANDIDATE_SHA")
        or ""
    ).strip().lower()
    identities = (
        value.get("host_source_identities")
        if isinstance(value, Mapping)
        else None
    )
    try:
        expected_identities = [
            _file_identity(path, normalized_sha)
            for path in HOST_RESTART_SUMMARY_SOURCE_PATHS
        ]
    except Exception:
        expected_identities = None
    return (
        isinstance(value, Mapping)
        and set(value)
        == {
            *_RESTART_SUMMARY_DEADLINE_EVIDENCE_FIELDS,
            "host_source_identities",
        }
        and all(
            value.get(field) is True
            for field in _RESTART_SUMMARY_DEADLINE_EVIDENCE_FIELDS
        )
        and re.fullmatch(r"[0-9a-f]{40}", normalized_sha) is not None
        and isinstance(identities, list)
        and len(identities) == len(HOST_RESTART_SUMMARY_SOURCE_PATHS)
        and all(isinstance(item, Mapping) for item in identities)
        and identities == expected_identities
        and sorted(str(item.get("path") or "") for item in identities)
        == sorted(HOST_RESTART_SUMMARY_SOURCE_PATHS)
        and all(
            isinstance(item, Mapping)
            and item.get("commit_sha") == normalized_sha
            and re.fullmatch(
                r"[0-9a-f]{64}",
                str(item.get("sha256") or ""),
            )
            is not None
            for item in identities
        )
    )


def _exercise_restart_summary_deadline_classification() -> dict[str, Any]:
    """Replay exact candidate restart summaries without sleeping or Sentry I/O."""

    candidate_sha = str(
        os.environ.get("REHEARSAL_CANDIDATE_SHA") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        raise RuntimeError("restart summary rehearsal candidate SHA is invalid")
    host_source_identities = [
        _file_identity(path, candidate_sha)
        for path in HOST_RESTART_SUMMARY_SOURCE_PATHS
    ]

    from leadpoet_observability import sentry_operations

    expected_module_path = (
        SOURCE_ROOT / HOST_RESTART_SUMMARY_SOURCE_PATHS[0]
    ).resolve()
    if Path(sentry_operations.__file__).resolve() != expected_module_path:
        raise RuntimeError("restart summary host module source differs")

    breadcrumbs: list[dict[str, Any]] = []
    distributions: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    terminal_events: list[dict[str, Any]] = []
    configured_contexts: list[dict[str, Any]] = []
    recorded_stages: list[dict[str, Any]] = []
    previous_deadline = os.environ.get(
        "LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS"
    )
    deadline_seconds = 5

    original_configure = sentry_operations.configure_sentry_context
    original_record_stage = sentry_operations.record_stage
    original_breadcrumb = sentry_operations.sentry_bootstrap.add_sentry_breadcrumb
    original_distribution = (
        sentry_operations.sentry_bootstrap.record_sentry_distribution
    )
    original_capture = sentry_operations.sentry_bootstrap.capture_sentry_failure

    def write_ledger(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        path.write_text(
            "".join(
                json.dumps(dict(record), sort_keys=True, separators=(",", ":"))
                + "\n"
                for record in records
            ),
            encoding="utf-8",
        )

    def emit(
        *,
        component: str,
        status: str,
        stage: str,
        ledger_path: Path,
    ) -> None:
        sentry_operations.emit_restart_summary(
            component=component,
            status=status,
            stage=stage,
            ledger_path=ledger_path,
            restart_invocation_id=ledger_path.stem,
            candidate_sha=candidate_sha,
        )

    try:
        os.environ["LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS"] = str(
            deadline_seconds
        )
        sentry_operations.configure_sentry_context = (
            lambda **fields: configured_contexts.append(dict(fields))
        )
        sentry_operations.record_stage = (
            lambda **fields: recorded_stages.append(dict(fields))
        )
        sentry_operations.sentry_bootstrap.add_sentry_breadcrumb = (
            lambda **fields: breadcrumbs.append(dict(fields))
        )
        sentry_operations.sentry_bootstrap.record_sentry_distribution = (
            lambda *args, **kwargs: distributions.append((args, dict(kwargs)))
        )
        sentry_operations.sentry_bootstrap.capture_sentry_failure = (
            lambda **fields: terminal_events.append(dict(fields)) or True
        )

        with tempfile.TemporaryDirectory(
            prefix="restart-summary-rehearsal-"
        ) as raw:
            root = Path(raw)
            token = root.name
            successful_invocations: set[str] = set()
            for component, milestones in sorted(
                sentry_operations._RESTART_MILESTONES.items()
            ):
                if not milestones or milestones[-1] != "completed":
                    raise RuntimeError("candidate restart milestone contract differs")
                elapsed = float(deadline_seconds + 1)
                records = []
                for ordinal, milestone in enumerate(milestones):
                    if ordinal:
                        elapsed += 1.0
                    records.append(
                        {
                            "stage": milestone,
                            "status": "passed",
                            "elapsed_seconds": elapsed,
                        }
                    )
                ledger = root / f"{component}-success-{token}.jsonl"
                write_ledger(ledger, records)
                emit(
                    component=component,
                    status="failed",
                    stage=milestones[-1],
                    ledger_path=ledger,
                )
                successful_invocations.add(ledger.stem)

            active_timeout = root / f"gateway-active-timeout-{token}.jsonl"
            write_ledger(
                active_timeout,
                (
                    {
                        "stage": "active_restart_work",
                        "status": "failed",
                        "elapsed_seconds": float(deadline_seconds + 1),
                    },
                ),
            )
            emit(
                component="gateway",
                status="failed",
                stage="active_restart_work",
                ledger_path=active_timeout,
            )

            passive_then_failure = root / f"validator-passive-failure-{token}.jsonl"
            validator_milestones = sentry_operations._RESTART_MILESTONES[
                "validator"
            ]
            write_ledger(
                passive_then_failure,
                (
                    {
                        "stage": validator_milestones[0],
                        "status": "passed",
                        "elapsed_seconds": float(deadline_seconds + 1),
                    },
                    {
                        "stage": "active_restart_work",
                        "status": "failed",
                        "elapsed_seconds": float(deadline_seconds + 2),
                    },
                ),
            )
            emit(
                component="validator",
                status="failed",
                stage="active_restart_work",
                ledger_path=passive_then_failure,
            )

            stale_completion = root / f"gateway-stale-completion-{token}.jsonl"
            gateway_milestones = sentry_operations._RESTART_MILESTONES["gateway"]
            write_ledger(
                stale_completion,
                (
                    {
                        "stage": gateway_milestones[-1],
                        "status": "passed",
                        "elapsed_seconds": 1.0,
                    },
                    {
                        "stage": "active_restart_work",
                        "status": "failed",
                        "elapsed_seconds": 2.0,
                    },
                ),
            )
            emit(
                component="gateway",
                status="failed",
                stage="active_restart_work",
                ledger_path=stale_completion,
            )

            terminal_codes = [
                str(event.get("failure_code") or "")
                for event in terminal_events
            ]
            terminal_ids = {
                str(event.get("context", {}).get("restart_invocation_id") or "")
                for event in terminal_events
            }
            context_ids = {
                str(context.get("restart_invocation_id") or "")
                for context in configured_contexts
            }
            warning_ids = {
                str(item.get("data", {}).get("restart_invocation_id") or "")
                for item in breadcrumbs
                if item.get("message") == "restart.stage_deadline_exceeded"
            }

            if len(distributions) != len(successful_invocations):
                raise RuntimeError("successful restart duration metrics differ")
            if warning_ids != successful_invocations:
                raise RuntimeError("successful restart warning identity differs")
            if terminal_codes.count("restart.stage_deadline_exceeded") != 1:
                raise RuntimeError("active restart timeout classification differs")
            if terminal_codes.count("restart.terminal_failure") != 2:
                raise RuntimeError("non-timeout restart failure classification differs")
            if successful_invocations & terminal_ids:
                raise RuntimeError("successful restart emitted a terminal event")
            expected_invocations = successful_invocations | {
                active_timeout.stem,
                passive_then_failure.stem,
                stale_completion.stem,
            }
            if context_ids != expected_invocations or terminal_ids != {
                active_timeout.stem,
                passive_then_failure.stem,
                stale_completion.stem,
            }:
                raise RuntimeError("restart invocation identity differs from ledger")
            if not recorded_stages:
                raise RuntimeError("candidate restart stage diagnostics were skipped")
    finally:
        sentry_operations.configure_sentry_context = original_configure
        sentry_operations.record_stage = original_record_stage
        sentry_operations.sentry_bootstrap.add_sentry_breadcrumb = original_breadcrumb
        sentry_operations.sentry_bootstrap.record_sentry_distribution = (
            original_distribution
        )
        sentry_operations.sentry_bootstrap.capture_sentry_failure = original_capture
        if previous_deadline is None:
            os.environ.pop(
                "LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS",
                None,
            )
        else:
            os.environ[
                "LEADPOET_SENTRY_RESTART_STAGE_DEADLINE_SECONDS"
            ] = previous_deadline

    evidence = {
        "active_timeout_terminal": True,
        "host_module_source_exact": True,
        "host_source_identities": host_source_identities,
        "later_failure_overrides_stale_completion": True,
        "passive_wait_does_not_relabel_failure": True,
        "restart_invocation_identity_exact": True,
        "successful_slow_wait_nonterminal": True,
        "successful_duration_metrics_retained": True,
    }
    if not _restart_summary_deadline_evidence_is_complete(
        evidence,
        candidate_sha=candidate_sha,
    ):
        raise RuntimeError("restart summary deadline evidence is incomplete")
    return evidence


def _exercise_compact_weight_joined_path() -> dict[str, Any]:
    from compact_weight_joined_runner import exercise_compact_weight_joined_path

    return exercise_compact_weight_joined_path()


BEHAVIOR_ACTIONS: dict[str, Callable[[], dict[str, Any]]] = {
    "chain-settlement-state-space": _exercise_chain_settlement_state_space,
    "restart-summary-deadline-classification": (
        _exercise_restart_summary_deadline_classification
    ),
    "compact-weight-joined-path": _exercise_compact_weight_joined_path,
    "company-fit-numeric-observation-projection": (
        _exercise_company_fit_numeric_observation_projection
    ),
    "measured-assigned-proxy-raw-transport": (
        _exercise_measured_assigned_proxy_raw_transport
    ),
    "measured-coordinator-raw-transport": (
        _exercise_measured_coordinator_raw_transport
    ),
    "artifact-egress-sustained-readback": (
        _exercise_artifact_egress_sustained_readback
    ),
    "historical-metagraph-layouts": _exercise_historical_metagraph_layouts,
    "receipt-graph-aggregate-pagination": (
        _exercise_receipt_graph_aggregate_pagination
    ),
    "receipt-graph-transport-deduplication": (
        _exercise_receipt_graph_transport_deduplication
    ),
    "fresh-weight-input-lineage": _exercise_fresh_weight_input_lineage,
    "stateful-compact-graph-readback": (
        _exercise_stateful_compact_graph_readback
    ),
    "research-lab-allocation-conservation": (
        _exercise_research_lab_allocation_conservation
    ),
    "settlement-frontier-terminal-retirement": (
        _exercise_settlement_frontier_terminal_retirement
    ),
    "current-frontier-release-recovery": (
        _exercise_current_frontier_release_recovery
    ),
    "validator-publication-release-recovery": (
        _exercise_validator_publication_release_recovery
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["--behavior-worker"]:
        return _behavior_worker_main(raw_argv[1:])
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("prepush", "release"), required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--boundary-contract", type=Path, required=True)
    parser.add_argument("--production-allocation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(raw_argv)
    if len(args.candidate_sha) != 40 or any(
        value not in "0123456789abcdef" for value in args.candidate_sha
    ):
        parser.error("--candidate-sha must be a full lowercase Git SHA")
    expected_epochs = 1 if args.profile == "prepush" else 100
    if args.epochs != expected_epochs:
        parser.error(f"{args.profile} requires exactly {expected_epochs} epochs")

    stages: list[dict[str, Any]] = []
    fixture: dict[str, Any] | None = None
    boundary_contract: dict[str, Any] | None = None
    behavior_contract: dict[str, Any] | None = None

    def load_inputs() -> tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ]:
        loaded_fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
        loaded_boundary_contract = json.loads(
            args.boundary_contract.read_text(encoding="utf-8")
        )
        if loaded_fixture["sanitization"]["contains_production_credentials"]:
            raise RuntimeError("rehearsal fixture contains production credentials")
        if set(loaded_boundary_contract["forbidden_substitutions"]) != {
            "gateway",
            "validator",
            "auditor",
            "canonical_bundle",
            "receipt_graph",
            "signature",
            "sdk_extrinsic",
            "verification",
        }:
            raise RuntimeError("rehearsal substitution policy is incomplete")
        loaded_behavior_contract = validate_rehearsal_behavior_contract_v2(
            build_rehearsal_behavior_contract_v2(
                source_root=SOURCE_ROOT,
                candidate_sha=args.candidate_sha,
                profile=args.profile,
                epoch_count=args.epochs,
            )
        )
        if args.profile == "release" and list(
            loaded_fixture.get("fault_matrix") or []
        ) != loaded_behavior_contract["fault_ids"]:
            raise RuntimeError(
                "mounted fault matrix differs from candidate contract"
            )
        return (
            loaded_fixture,
            loaded_boundary_contract,
            loaded_behavior_contract,
        )

    inputs_passed, inputs = _run_workflow_stage(
        stage="input-contract",
        action=load_inputs,
        stages=stages,
    )
    if inputs_passed:
        fixture, boundary_contract, behavior_contract = inputs

    allocation_passed, production_allocation = _run_workflow_stage(
        stage="production-allocation-input",
        action=lambda: _load_production_allocation(
            args.production_allocation,
            candidate_sha=args.candidate_sha,
        ),
        stages=stages,
    )
    if not allocation_passed:
        production_allocation = None

    identities: list[dict[str, str]] = []
    source_paths = (
        list(behavior_contract["production_source_paths"])
        if behavior_contract is not None
        else []
    )
    for path in source_paths:
        passed, identity = _run_workflow_stage(
            stage=f"source-identity:{path}",
            action=lambda path=path: _file_identity(path, args.candidate_sha),
            stages=stages,
        )
        if passed:
            identities.append(identity)

    behavior_scenarios = (
        list(behavior_contract["behavior_scenarios"])
        if behavior_contract is not None
        else []
    )
    behavior_evidence = _run_behavior_actions(
        scenarios=behavior_scenarios,
        stages=stages,
    )

    _run_independent_epoch_diagnostics(
        candidate_sha=args.candidate_sha,
        epoch_id=30_000,
        stages=stages,
        production_allocation=production_allocation,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    service_root = args.output.parent / "local-services"
    faults: list[dict[str, Any]] = []
    concurrent_writes = 0
    epochs: list[dict[str, Any]] = []
    boundary_events: list[dict[str, Any]] = []
    cleanup = {
        "pending_faults": 0,
        "boundary_thread_alive_before_close": False,
        "boundary_thread_alive_after_close": False,
        "local_chain_epochs": 0,
    }

    if fixture is None:
        if args.profile == "release":
            _mark_workflow_stage_unexercised(
                stage="fault-matrix",
                blocked_by=["input-contract"],
                stages=stages,
            )
            _mark_workflow_stage_unexercised(
                stage="concurrency",
                blocked_by=["input-contract"],
                stages=stages,
            )
        for ordinal in range(args.epochs):
            _mark_workflow_stage_unexercised(
                stage=f"epoch-{30_000 + ordinal}",
                blocked_by=["input-contract"],
                stages=stages,
            )
        _mark_workflow_stage_unexercised(
            stage="boundary-cleanup",
            blocked_by=["input-contract"],
            stages=stages,
        )
    else:
        if args.profile == "release":
            for ordinal, fault in enumerate(fixture["fault_matrix"]):
                def run_fault(
                    *,
                    ordinal: int = ordinal,
                    fault: str = str(fault),
                ) -> dict[str, Any]:
                    with LocalBoundaryServices(
                        root=service_root / f"fault-{ordinal:02d}",
                        fixture=fixture,
                    ) as fault_services:
                        return _exercise_fault(
                            fault_services,
                            fault=fault,
                            ordinal=ordinal,
                        )

                passed, result = _run_workflow_stage(
                    stage=f"fault:{ordinal}:{fault}",
                    action=run_fault,
                    stages=stages,
                )
                if passed:
                    faults.append(result)

            def run_concurrency() -> int:
                with LocalBoundaryServices(
                    root=service_root / "concurrency",
                    fixture=fixture,
                ) as concurrency_services:
                    return _exercise_concurrency(concurrency_services)

            passed, result = _run_workflow_stage(
                stage="concurrency",
                action=run_concurrency,
                stages=stages,
            )
            if passed:
                concurrent_writes = result

        services = LocalBoundaryServices(
            root=service_root / "epochs",
            fixture=fixture,
        )
        services_started, _ = _run_workflow_stage(
            stage="boundary-start",
            action=services.__enter__,
            stages=stages,
        )
        if services_started:
            try:
                first_epoch = 30_000
                for ordinal in range(args.epochs):
                    epoch_id = first_epoch + ordinal
                    passed, epoch = _run_workflow_stage(
                        stage=f"epoch-{epoch_id}",
                        action=lambda epoch_id=epoch_id: _run_epoch(
                            services=services,
                            fixture=fixture,
                            candidate_sha=args.candidate_sha,
                            epoch_id=epoch_id,
                            production_allocation=production_allocation,
                        ),
                        stages=stages,
                    )
                    if passed:
                        epochs.append(epoch)
                boundary_events = list(services.state.events)
                cleanup = {
                    "pending_faults": len(services.state.faults),
                    "boundary_thread_alive_before_close": (
                        services.thread.is_alive()
                    ),
                    "boundary_thread_alive_after_close": True,
                    "local_chain_epochs": len(services.state.chain),
                }
            finally:
                cleanup_passed, _ = _run_workflow_stage(
                    stage="boundary-cleanup",
                    action=lambda: services.__exit__(None, None, None),
                    stages=stages,
                )
                cleanup["boundary_thread_alive_after_close"] = (
                    services.thread.is_alive()
                )
                cleanup["local_chain_epochs"] = len(services.state.chain)
                if not cleanup_passed:
                    cleanup["cleanup_failed"] = True
        else:
            for ordinal in range(args.epochs):
                _mark_workflow_stage_unexercised(
                    stage=f"epoch-{30_000 + ordinal}",
                    blocked_by=["boundary-start"],
                    stages=stages,
                )
            _mark_workflow_stage_unexercised(
                stage="boundary-cleanup",
                blocked_by=["boundary-start"],
                stages=stages,
            )

    validation_dependencies = [
        item["stage"] for item in stages if item.get("status") != "passed"
    ]
    stage_status = {
        str(item.get("stage")): str(item.get("status"))
        for item in stages
        if isinstance(item, Mapping)
    }
    duplicate_stage_ids = len(stage_status) != len(stages)
    expected_before_validation = (
        set(behavior_contract["required_stage_ids"])
        - {"workflow-evidence-validation"}
        if behavior_contract is not None
        else set()
    )
    observed_before_validation = set(stage_status)

    epoch_authority_complete = (
        len(epochs) == expected_epochs
        and all(
            epoch.get("canonical_vector_equal") is True
            and epoch.get("receipt_ancestry_verified") is True
            and epoch.get("auditor_verified") is True
            and epoch.get("auditor_runtime_verified") is True
            and epoch.get("sdk_bridge_verified") is True
            and bool(epoch.get("signed_extrinsic_hash"))
            and epoch.get("last_update") == epoch.get("finalized_block")
            for epoch in epochs
        )
    )
    identity_paths = [str(item.get("path")) for item in identities]
    identity_commits = {
        str(item.get("commit_sha")) for item in identities
    }
    boundary_definitions = (
        boundary_contract.get("boundaries")
        if isinstance(boundary_contract, Mapping)
        else None
    )
    unknown_boundaries_rejected = (
        isinstance(boundary_definitions, Mapping)
        and bool(boundary_definitions)
        and all(
            isinstance(definition, Mapping)
            and definition.get("reject_unknown") is True
            for definition in boundary_definitions.values()
        )
    )
    behavioral_invariants = {
        "candidate_identity_exact": (
            behavior_contract is not None
            and behavior_contract.get("candidate_sha") == args.candidate_sha
        ),
        "protected_source_identity_exact": (
            behavior_contract is not None
            and sorted(identity_paths)
            == sorted(behavior_contract["production_source_paths"])
            and identity_commits == {args.candidate_sha}
        ),
        "restart_summary_deadline_classification_exact": (
            _restart_summary_deadline_evidence_is_complete(
                behavior_evidence.get(
                    "restart-summary-deadline-classification",
                    {},
                ),
                candidate_sha=args.candidate_sha,
            )
        ),
        "compact_weight_joined_path_verified": (
            behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("production_allocation_guard")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("production_primary_compact_lifecycle")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("gateway_compact_submit_persist_get_finalize")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("compact_ancestry_checkpoint_persistence")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("real_epoch_evidence_endpoint")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("stateful_epoch_evidence_persisted")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("stateful_epoch_evidence_readback_exact")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("cutover_authority_db_boundary_exact")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("release_lineage_file_archive_boundary_exact")
            is True
            and bool(
                behavior_evidence.get(
                    "compact-weight-joined-path",
                    {},
                ).get("extrinsic_hash")
            )
            and int(
                behavior_evidence.get(
                    "compact-weight-joined-path",
                    {},
                ).get("finalized_block")
                or 0
            )
            > 0
        ),
        "compact_ancestry_unknown_commit_recovery_verified": (
            behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("ancestry_unknown_commit_recovered_read_only")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("ancestry_unknown_commit_rpc_write_count")
            == 1
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("ancestry_unknown_commit_readback_count")
            == 3
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("single_canonical_publish_finalize_after_unknown_commit")
            is True
        ),
        "compact_primary_auditor_byte_identity_verified": (
            behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("primary_auditor_byte_identity")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("independent_auditor_count")
            == 2
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("independent_auditor_submission_count")
            == 2
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("auditor_submission_success")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("auditor_last_update_advanced")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("auditor_finalized_vector_readback_equal")
            is True
            and re.fullmatch(
                r"sha256:[0-9a-f]{64}",
                str(
                    behavior_evidence.get(
                        "compact-weight-joined-path",
                        {},
                    ).get("primary_auditor_vector_hash")
                    or ""
                ),
            )
            is not None
            and len(
                behavior_evidence.get(
                    "compact-weight-joined-path",
                    {},
                ).get("auditor_submission_states")
                or ()
            )
            == 2
            and len(
                {
                    int(state.get("uid") or -1)
                    for state in (
                        behavior_evidence.get(
                            "compact-weight-joined-path",
                            {},
                        ).get("auditor_submission_states")
                        or ()
                    )
                }
            )
            == 2
            and {
                str(state.get("vector_hash") or "")
                for state in (
                    behavior_evidence.get(
                        "compact-weight-joined-path",
                        {},
                    ).get("auditor_submission_states")
                    or ()
                )
            }
            == {
                str(
                    behavior_evidence.get(
                        "compact-weight-joined-path",
                        {},
                    ).get("primary_auditor_vector_hash")
                    or ""
                )
            }
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("auditor_verified_cache_replay")
            is True
        ),
        "compact_publication_journal_recovery_verified": (
            behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("same_epoch_compact_journal_recovered")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("same_epoch_compact_fresh_scan_recovered")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("compact_finalization_job_ids_scan_derived")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("compact_fresh_scan_recovery_writes")
            == 0
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("compact_mismatched_recovery_conflict")
            is True
            and behavior_evidence.get(
                "compact-weight-joined-path",
                {},
            ).get("next_epoch_compact_journal_retired")
            is True
        ),
        "company_fit_numeric_observation_projection_verified": (
            _company_fit_numeric_observation_projection_evidence_is_complete(
                behavior_evidence.get(
                    "company-fit-numeric-observation-projection",
                    {},
                )
            )
        ),
        "measured_assigned_proxy_raw_transport_verified": (
            behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("exact_httpx_enclave_parent_proxy_provider_path")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("assigned_proxy_raw_parent_tunnel_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("nested_tls_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("proxy_auth_remained_in_enclave")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("provider_first_close_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("bounded_cleanup_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("production_http_connect_proxy_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("request_scoped_connection_cleanup_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("one_connect_per_request_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("failure_recovery_on_fresh_tunnel_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("classified_failure_health_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("stable_process_resource_count_verified")
            is True
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("repeated_request_count")
            == 8
            and behavior_evidence.get(
                "measured-assigned-proxy-raw-transport",
                {},
            ).get("attempt_count")
            == 9
        ),
        "measured_coordinator_raw_transport_verified": (
            behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("exact_httpx_enclave_parent_proxy_provider_path")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("raw_parent_tunnel_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("nested_tls_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("proxy_auth_remained_in_enclave")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("provider_first_close_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("bounded_cleanup_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("request_scoped_connection_cleanup_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("one_connect_per_request_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("failure_recovery_on_fresh_tunnel_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("classified_failure_health_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("stable_process_resource_count_verified")
            is True
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("repeated_request_count")
            == 8
            and behavior_evidence.get(
                "measured-coordinator-raw-transport",
                {},
            ).get("attempt_count")
            == 9
        ),
        "artifact_egress_sustained_readback_verified": (
            behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("exact_transport_proxy_forwarder_path")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("sustained_single_tunnel_reused")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("bounded_concurrent_tunnels_verified")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("multi_frame_response_verified")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("provider_first_close_verified")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("provider_first_idle_terminal_handshake_verified")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("complete_chunked_json_eof_recovered")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("truncated_frame_rejected")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("ordinary_provider_transport_request_scoped")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("ordinary_direct_serialized_generation_recovery_verified")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("stale_pooled_transport_evicted_before_relay_timeout")
            is True
            and behavior_evidence.get(
                "artifact-egress-sustained-readback",
                {},
            ).get("failed_pooled_generation_evicted_before_retry")
            is True
        ),
        "chain_settlement_state_space_complete": (
            "chain-settlement-state-space" in behavior_evidence
            and behavior_evidence["chain-settlement-state-space"].get(
                "retry_observation_sequence"
            )
            == 8
            and behavior_evidence["chain-settlement-state-space"].get(
                "retry_attempts_propagated"
            )
            == 2
            and behavior_evidence["chain-settlement-state-space"].get(
                "durable_retry_attempt"
            )
            == 4
            and behavior_evidence["chain-settlement-state-space"].get(
                "durable_retry_cooldown"
            )
            is True
        ),
        "historical_metagraph_layouts_policy_bound": (
            "historical-metagraph-layouts" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence["historical-metagraph-layouts"].get(
                "policy_hash"
            )
            == behavior_contract["policy_commitments"]["chain_source"].get(
                "policy_hash"
            )
            and behavior_evidence["historical-metagraph-layouts"].get(
                "accepted_layouts"
            )
            == behavior_contract["policy_commitments"]["chain_source"][
                "policy"
            ].get("selective_result_last_fields")
        ),
        "receipt_graph_aggregate_evidence_paged": (
            behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("aggregate_evidence_paged")
            is True
            and behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("structural_limit_enforced")
            is True
            and behavior_evidence.get(
                "receipt-graph-aggregate-pagination",
                {},
            ).get("checkpoint_parent_first_persistence")
            is True
        ),
        "receipt_graph_transport_deduplicated_and_verified": (
            behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("exact_job_path_verified")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("malformed_evidence_rejected")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("ordinary_graph_bound_preserved")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("historical_checkpoint_issuer_included")
            is True
            and behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("transport_size_bytes", 1)
            < behavior_evidence.get(
                "receipt-graph-transport-deduplication",
                {},
            ).get("legacy_size_bytes", 0)
        ),
        "fresh_weight_input_lineage_verified": (
            behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("fresh_checkpoint_lineage_accepted")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("direct_execution_proof_selected")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("replay_identity_equal")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("direct_receipts_persisted")
            is True
            and behavior_evidence.get(
                "fresh-weight-input-lineage", {}
            ).get("mismatched_execution_rejected")
            is True
        ),
        "stateful_compact_graph_readback_verified": (
            behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("checkpoint_v3_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("canonical_v4_readback_accepted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("boundary_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("snapshot_persisted")
            is True
            and behavior_evidence.get(
                "stateful-compact-graph-readback", {}
            ).get("tampered_v4_rejected_before_write")
            is True
        ),
        "research_lab_allocation_policy_config_bound": (
            "research-lab-allocation-conservation" in behavior_evidence
            and behavior_contract is not None
            and behavior_evidence[
                "research-lab-allocation-conservation"
            ].get("policy_hash")
            == behavior_contract["policy_commitments"][
                "research_lab_allocation"
            ].get("policy_hash")
        ),
        "research_lab_allocation_conserved": (
            behavior_evidence.get(
                "research-lab-allocation-conservation",
                {},
            ).get("conserved")
            is True
        ),
        "settlement_frontier_terminal_retirement_verified": (
            behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("original_failure_reproduced")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("champion_terminal_retired")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("source_add_terminal_retired")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("tampered_identity_rejected")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("canonical_receipt_without_release_hash_accepted")
            is True
            and behavior_evidence.get(
                "settlement-frontier-terminal-retirement",
                {},
            ).get("execution_release_hash_validated")
            is True
        ),
        "current_frontier_release_recovery_verified": (
            behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("cross_release_execution_skipped")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("exact_signed_authority_reused")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("immutable_frontier_preserved")
            is True
            and behavior_evidence.get(
                "current-frontier-release-recovery",
                {},
            ).get("malformed_release_rejected")
            is True
        ),
        "validator_publication_release_recovery_verified": (
            behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("approved_n_minus_one_recovered")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("nitro_attestation_rechecked")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("release_tampering_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("same_release_config_mismatch_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("cross_release_finalization_only")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("unsigned_cross_release_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("implicit_cross_release_rejected")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("same_epoch_finalized_journal_retained")
            is True
            and behavior_evidence.get(
                "validator-publication-release-recovery",
                {},
            ).get("next_epoch_finalized_journal_retired")
            is True
        ),
        "canonical_vector_primary_auditor_equal": (
            epoch_authority_complete
            and all(
                epoch.get("canonical_vector_equal") is True
                for epoch in epochs
            )
        ),
        "receipt_ancestry_verified": (
            epoch_authority_complete
            and all(
                epoch.get("receipt_ancestry_verified") is True
                for epoch in epochs
            )
        ),
        "sdk_signing_bridge_verified": (
            epoch_authority_complete
            and all(
                epoch.get("sdk_bridge_verified") is True
                for epoch in epochs
            )
        ),
        "submission_finalized": (
            epoch_authority_complete
            and all(bool(epoch.get("signed_extrinsic_hash")) for epoch in epochs)
        ),
        "last_update_readback_equal": (
            epoch_authority_complete
            and all(
                epoch.get("last_update") == epoch.get("finalized_block")
                for epoch in epochs
            )
        ),
        "boundary_cleanup_complete": (
            cleanup["pending_faults"] == 0
            and cleanup["boundary_thread_alive_after_close"] is False
            and cleanup["local_chain_epochs"] == expected_epochs
        ),
        "unknown_boundaries_rejected": unknown_boundaries_rejected,
    }

    def validate_workflow_evidence() -> None:
        if behavior_contract is None:
            raise RuntimeError("candidate behavior contract is unavailable")
        if duplicate_stage_ids:
            raise RuntimeError("workflow emitted duplicate stage evidence")
        if observed_before_validation != expected_before_validation:
            missing = sorted(
                expected_before_validation - observed_before_validation
            )
            unexpected = sorted(
                observed_before_validation - expected_before_validation
            )
            raise RuntimeError(
                "workflow stage contract differs "
                f"missing={missing} unexpected={unexpected}"
            )
        required_invariants = set(
            behavior_contract["required_invariant_ids"]
        )
        if set(behavioral_invariants) != required_invariants:
            raise RuntimeError("workflow invariant contract differs")
        failed_invariants = sorted(
            name
            for name, passed in behavioral_invariants.items()
            if passed is not True
        )
        if failed_invariants:
            raise RuntimeError(
                "joined V2 workflow invariants failed: "
                + ",".join(failed_invariants)
            )
        if args.profile == "release" and (
            len(faults) != len(behavior_contract["fault_ids"])
            or concurrent_writes != 32
        ):
            raise RuntimeError("release fault or concurrency evidence is incomplete")

    if validation_dependencies:
        _mark_workflow_stage_unexercised(
            stage="workflow-evidence-validation",
            blocked_by=validation_dependencies,
            stages=stages,
        )
    else:
        _run_workflow_stage(
            stage="workflow-evidence-validation",
            action=validate_workflow_evidence,
            stages=stages,
        )

    status = (
        "passed"
        if all(item.get("status") == "passed" for item in stages)
        else "failed"
    )
    manifest = {
        "schema_version": "leadpoet.local_v2_workflow_evidence.v1",
        "status": status,
        "profile": args.profile,
        "release_sha": args.candidate_sha,
        "fixture_hash": sha256_json(fixture) if fixture is not None else None,
        "boundary_contract_hash": (
            sha256_json(boundary_contract)
            if boundary_contract is not None
            else None
        ),
        "behavior_contract": behavior_contract,
        "behavior_contract_hash": (
            behavior_contract.get("contract_hash")
            if behavior_contract is not None
            else None
        ),
        "behavior_evidence": behavior_evidence,
        "behavioral_invariants": behavioral_invariants,
        "production_source_identities": identities,
        "epoch_count": len(epochs),
        "epochs": epochs,
        "fault_matrix": faults,
        "concurrent_write_count": concurrent_writes,
        "boundary_event_count": len(boundary_events),
        "cleanup": cleanup,
        "stages": stages,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    args.output.write_bytes(_canonical(manifest) + b"\n")
    if status != "passed":
        failed = sum(item.get("status") == "failed" for item in stages)
        unexercised = sum(
            item.get("status") == "unexercised" for item in stages
        )
        print(
            "PRODUCTION_WORKFLOW_REHEARSAL_FAILED "
            f"profile={args.profile} failed={failed} "
            f"unexercised={unexercised} evidence={args.output}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(
        f"PRODUCTION_WORKFLOW_REHEARSAL_SUCCESS profile={args.profile} "
        f"epochs={len(epochs)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

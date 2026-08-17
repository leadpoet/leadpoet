#!/usr/bin/env python3
"""Dry-run-first recorder for current-day Research Lab dev snapshots.

Runs the CURRENT champion once over the complete scored daily benchmark bank,
capturing every provider response (Exa, Scrapingdog, OpenRouter, ...) into a
frozen snapshot set that `research_lab.eval.dev_eval.evaluate_dev` replays
deterministically (§6.3-1). Recording spends real provider budget, so it is
double-gated: the default invocation only prints the plan, and a live run
requires BOTH

  --record

and the environment gate

  RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED=true

The per-tree weak/strong cohort is selected later from this immutable bank.
This recorder never chooses ICPs from retired sets and never prints hidden ICP
refs or payloads.

Champion runners:
  --adapter-path   private champion checkout, run in a subprocess (mirrors
                   SubprocessPrivateModelRunner with the record bootstrap
                   prepended) — the path used on a gateway box.
  --champion-image immutable ECR digest, run through docker with the snapshot
                   directory volume-mounted (mirrors DockerPrivateModelRunner).

Recording writes to a LOCAL directory (the in-process/in-container record
bootstrap persists files); sync the directory to the S3 prefix behind
RESEARCH_LAB_DEV_SNAPSHOT_URI afterwards if the fleet replays from S3.

Example (gateway box):

  RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED=true \
  python3 scripts/record_research_lab_dev_snapshots.py \
      --source-icps /tmp/source_icps.json \
      --snapshot-dir /var/lib/research_lab/dev_snapshots/dev-v1 \
      --champion-image <immutable-ecr-digest> \
      --record
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import uuid
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.config import (  # noqa: E402
    MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
    RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD,
    ResearchLabGitTreeConfig,
    ResearchLabGitTreeConfigError,
)
from research_lab.eval.private_runtime import SECRET_MARKERS  # noqa: E402
from research_lab.docker_operation_lock_v2 import (  # noqa: E402
    DockerOperationLockError,
    shared_docker_operation_lock,
)
from research_lab.eval.snapshot_store import (  # noqa: E402
    RECORDED_PROVIDER_MODELS_NAME,
    SNAPSHOT_MISS_SENTINEL,
    SNAPSHOT_RECORD_REUSE_EXISTING_ENV,
)

RECORD_ENABLED_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
MAX_SNAPSHOT_CLOSURE_ROUNDS = 8
SNAPSHOT_EXECUTION_CONTEXT_MARKER = "dev_snapshot_recording"
MAX_SNAPSHOT_RECORD_ATTEMPTS = 3
SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS = (5.0, 15.0)
SNAPSHOT_RECORD_CANCELLED_EXIT_CODE = 75
DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS = 900
SNAPSHOT_DOCKER_CLEANUP_TIMEOUT_SECONDS = 30
SNAPSHOT_RECORD_FINALIZATION_RESERVE_SECONDS = 300

PROVIDER_KEY_GROUPS = (
    ("EXA_API_KEY",),
    ("SCRAPINGDOG_API_KEY", "QUALIFICATION_SCRAPINGDOG_API_KEY"),
    ("OPENROUTER_API_KEY", "QUALIFICATION_OPENROUTER_API_KEY", "OPENROUTER_KEY"),
)


class SnapshotRecordingCancelled(RuntimeError):
    """Raised only at an ICP boundary when the controller supersedes a run."""


def _raise_if_snapshot_record_cancelled(cancel_file: Path | None) -> None:
    if cancel_file is not None and cancel_file.is_file():
        raise SnapshotRecordingCancelled("active_private_model_changed")


def snapshot_record_workflow_timeout_seconds(
    *,
    item_count: int,
    item_timeout_seconds: int = DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS,
) -> int:
    """Return a finite outer bound for the sequential full-bank recorder.

    Every bank item can run once during initial capture, once in each closure
    round, and once during offline replay. Capture and closure runs use the
    bounded retry policy; every Docker attempt also reserves time for named
    container cleanup. The deliberately conservative bound lets the recorder
    finish or fail closed under its own limits instead of being killed by a
    shorter controller timeout.
    """

    if isinstance(item_count, bool) or not isinstance(item_count, int):
        raise ValueError("snapshot bank item count must be an integer")
    if item_count < 1:
        raise ValueError("snapshot bank item count must be positive")
    if (
        isinstance(item_timeout_seconds, bool)
        or not isinstance(item_timeout_seconds, int)
        or item_timeout_seconds < 1
    ):
        raise ValueError("snapshot ICP timeout must be a positive integer")
    if not SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS:
        raise ValueError("snapshot retry delays must not be empty")

    retry_delay_seconds = math.ceil(
        sum(
            SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS[
                min(
                    attempt_index,
                    len(SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS) - 1,
                )
            ]
            for attempt_index in range(MAX_SNAPSHOT_RECORD_ATTEMPTS - 1)
        )
    )
    docker_attempt_seconds = (
        item_timeout_seconds + SNAPSHOT_DOCKER_CLEANUP_TIMEOUT_SECONDS
    )
    capture_or_closure_seconds = (
        MAX_SNAPSHOT_RECORD_ATTEMPTS * docker_attempt_seconds
        + retry_delay_seconds
    )
    per_item_seconds = (
        (1 + MAX_SNAPSHOT_CLOSURE_ROUNDS) * capture_or_closure_seconds
        + docker_attempt_seconds
    )
    return (
        item_count * per_item_seconds
        + SNAPSHOT_RECORD_FINALIZATION_RESERVE_SECONDS
    )


def _load_json_file(path: str) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def _load_source_items(path: str) -> list[dict[str, Any]]:
    decoded = _load_json_file(path)
    if isinstance(decoded, Mapping):
        for key in ("items", "benchmark_items", "icps"):
            if isinstance(decoded.get(key), list):
                decoded = decoded[key]
                break
    if not isinstance(decoded, list):
        raise ValueError(f"source ICP file must be a JSON list (or hold one): {path}")
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


def _load_source_export(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    decoded = _load_json_file(path)
    if (
        not isinstance(decoded, Mapping)
        or decoded.get("schema_version") != "research_lab.dev_icp_export.v2"
        or not isinstance(decoded.get("items"), list)
        or not isinstance(decoded.get("daily_bank_manifest"), Mapping)
    ):
        raise ValueError(
            "source ICP file must be a current-day dev ICP export v2 document"
        )
    items = [
        dict(item) for item in decoded["items"] if isinstance(item, Mapping)
    ]
    if len(items) != len(decoded["items"]):
        raise ValueError("source ICP export contains an invalid item")
    return items, dict(decoded["daily_bank_manifest"])


def snapshot_export_bank_size(path: str) -> int:
    """Return the full exported bank size only when its envelope is coherent."""

    items, manifest = _load_source_export(path)
    bank_size = manifest.get("bank_size")
    if (
        isinstance(bank_size, bool)
        or not isinstance(bank_size, int)
        or bank_size < 1
        or bank_size != len(items)
    ):
        raise ValueError("source ICP export bank size differs from its items")
    return bank_size


def _terminate_snapshot_recorder_on_signal(
    signal_number: int,
    _frame: Any,
) -> None:
    """Turn host SIGTERM into stack unwinding and named Docker cleanup."""

    raise SystemExit(128 + int(signal_number))


def _provider_key_presence() -> dict[str, bool]:
    return {
        "/".join(group): any(os.getenv(name) for name in group)
        for group in PROVIDER_KEY_GROUPS
    }


def _subprocess_env(snapshot_dir: str, *, icp_ref: str = "") -> dict[str, str]:
    from research_lab.eval.private_runtime import private_model_env_passthrough
    from research_lab.eval.snapshot_store import SNAPSHOT_DIR_ENV

    env = {"PATH": os.environ.get("PATH", ""), "PYTHONUNBUFFERED": "1"}
    for name in private_model_env_passthrough():
        if name in os.environ:
            env[name] = os.environ[name]
    env[SNAPSHOT_DIR_ENV] = snapshot_dir
    env["RESEARCH_LAB_DEV_RECORD_ICP_REF"] = str(icp_ref)
    return env


def _snapshot_runtime_context(
    marker: str,
    *,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Finish model work before the host-owned Docker deadline."""

    from research_lab.eval.private_runtime import context_with_runtime_options

    host_reserve_seconds = min(
        60.0,
        max(10.0, float(timeout_seconds) * 0.1),
    )
    return context_with_runtime_options(
        {str(marker): True},
        outer_timeout_seconds=max(
            10.0,
            float(timeout_seconds) - host_reserve_seconds,
        ),
    )


def _run_named_docker(
    command: Sequence[str],
    *,
    container_name: str,
    input_text: str,
    timeout_seconds: int,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    """Run one uniquely named container and remove it on interruption."""

    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    try:
        with shared_docker_operation_lock(
            timeout_seconds=float(timeout_seconds),
            docker_executable=str(command[0]),
            environment=environment,
            deadline_monotonic=deadline,
        ):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise DockerOperationLockError(
                    "dev snapshot Docker lifecycle deadline was exhausted"
                )
            try:
                return subprocess.run(
                    command,
                    input=input_text,
                    text=True,
                    capture_output=True,
                    timeout=max(0.1, remaining),
                    env=dict(environment),
                    check=False,
                )
            except BaseException:
                try:
                    cleanup = subprocess.run(
                        [str(command[0]), "rm", "-f", container_name],
                        text=True,
                        capture_output=True,
                        timeout=SNAPSHOT_DOCKER_CLEANUP_TIMEOUT_SECONDS,
                        env={"PATH": os.environ.get("PATH", "")},
                        check=False,
                    )
                except (OSError, subprocess.SubprocessError) as cleanup_exc:
                    print(
                        "WARNING: interrupted dev snapshot Docker cleanup "
                        f"failed for {container_name}: "
                        f"{type(cleanup_exc).__name__}",
                        file=sys.stderr,
                    )
                else:
                    if cleanup.returncode != 0:
                        print(
                            "WARNING: interrupted dev snapshot Docker cleanup "
                            f"failed for {container_name}: "
                            f"exit={cleanup.returncode}",
                            file=sys.stderr,
                        )
                raise
    except DockerOperationLockError as exc:
        raise RuntimeError(
            f"dev snapshot Docker lifecycle unavailable: {exc}"
        ) from exc


def _record_icp_with_subprocess(
    *,
    adapter_path: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    icp_ref: str = "",
    snapshot_dir: str,
    timeout_seconds: int,
) -> list[Mapping[str, Any]]:
    """Run one champion ICP in a subprocess with the record bootstrap installed.

    Mirrors SubprocessPrivateModelRunner but prepends the snapshot record
    bootstrap so live provider responses are persisted per request key while
    passing through unchanged.
    """
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import dev_record_bootstrap

    adapter_bootstrap = getattr(private_runtime, "_ADAPTER_BOOTSTRAP", None)
    if not adapter_bootstrap:
        raise RuntimeError("private_runtime adapter bootstrap is unavailable")
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": _snapshot_runtime_context(
            SNAPSHOT_EXECUTION_CONTEXT_MARKER,
            timeout_seconds=timeout_seconds,
        ),
    }
    command = [
        sys.executable,
        "-c",
        dev_record_bootstrap() + adapter_bootstrap,
        str(Path(adapter_path).expanduser().resolve()),
        module_name,
        callable_name,
    ]
    completed = subprocess.run(
        command,
        input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        env=_subprocess_env(snapshot_dir, icp_ref=icp_ref),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"champion adapter failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    private_runtime.validate_sourcing_runtime_receipt(
        completed.stderr,
        expected_runtime_options=payload["context"]["runtime_options"],
    )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("champion adapter must return a JSON array")
    return decoded


def _record_icp_with_docker(
    *,
    image_digest: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    icp_ref: str = "",
    snapshot_dir: str,
    timeout_seconds: int,
    docker_executable: str = "docker",
    reuse_existing: bool = False,
) -> list[Mapping[str, Any]]:
    """Run one champion ICP through docker with the snapshot dir mounted.

    Mirrors DockerPrivateModelRunner but volume-mounts the snapshot directory
    and prepends the record bootstrap to the in-container adapter bootstrap.
    """
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import (
        SNAPSHOT_DIR_ENV,
        dev_record_bootstrap,
    )

    docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", None)
    if not docker_bootstrap:
        raise RuntimeError("private_runtime docker adapter bootstrap is unavailable")
    if "@sha256:" not in image_digest:
        raise RuntimeError("champion image must be an immutable digest")
    container_dir = "/research_lab_dev_snapshots"
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": _snapshot_runtime_context(
            SNAPSHOT_EXECUTION_CONTEXT_MARKER,
            timeout_seconds=timeout_seconds,
        ),
    }
    container_name = "leadpoet-dev-snapshot-record-" + uuid.uuid4().hex
    env_args: list[str] = []
    for name in private_runtime.private_model_env_passthrough():
        if name in os.environ:
            env_args.extend(["-e", name])
    if reuse_existing:
        env_args.extend(["-e", f"{SNAPSHOT_RECORD_REUSE_EXISTING_ENV}=true"])
    command = [
        docker_executable,
        "run",
        "--rm",
        "--name",
        container_name,
        "-i",
        "-v",
        f"{Path(snapshot_dir).expanduser().resolve()}:{container_dir}",
        "-e",
        f"{SNAPSHOT_DIR_ENV}={container_dir}",
        "-e",
        f"RESEARCH_LAB_DEV_RECORD_ICP_REF={icp_ref}",
        *env_args,
        image_digest,
        "python",
        "-c",
        dev_record_bootstrap() + docker_bootstrap,
        module_name,
        callable_name,
    ]
    completed = _run_named_docker(
        command,
        container_name=container_name,
        input_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        timeout_seconds=timeout_seconds,
        environment={
            **_subprocess_env(str(snapshot_dir), icp_ref=icp_ref),
            "PATH": os.environ.get("PATH", ""),
        },
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"docker champion adapter failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    private_runtime.validate_sourcing_runtime_receipt(
        completed.stderr,
        expected_runtime_options=payload["context"]["runtime_options"],
    )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("docker champion adapter must return a JSON array")
    return decoded


def _record_icp_with_retries(
    *,
    image_digest: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    icp_ref: str,
    snapshot_dir: str,
    timeout_seconds: int,
    reuse_existing: bool,
    item_index: int,
    item_count: int,
    max_attempts: int = MAX_SNAPSHOT_RECORD_ATTEMPTS,
    cancel_file: Path | None = None,
) -> list[Mapping[str, Any]]:
    """Retry one model run without repeating already recorded requests."""

    if max_attempts < 1:
        raise ValueError("snapshot record attempts must be positive")
    for attempt in range(1, max_attempts + 1):
        _raise_if_snapshot_record_cancelled(cancel_file)
        try:
            return _record_icp_with_docker(
                image_digest=image_digest,
                module_name=module_name,
                callable_name=callable_name,
                icp=icp,
                icp_ref=icp_ref,
                snapshot_dir=snapshot_dir,
                timeout_seconds=timeout_seconds,
                reuse_existing=reuse_existing or attempt > 1,
            )
        except Exception:  # noqa: BLE001 - bounded retry remains fail closed
            if attempt >= max_attempts:
                raise
            _raise_if_snapshot_record_cancelled(cancel_file)
            delay = SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS[
                min(attempt - 1, len(SNAPSHOT_RECORD_RETRY_DELAYS_SECONDS) - 1)
            ]
            print(
                "WARNING: snapshot recording attempt "
                f"{attempt}/{max_attempts} failed for daily ICP "
                f"{item_index}/{item_count}; retrying after {delay:g}s"
            )
            time.sleep(delay)
    raise AssertionError("unreachable snapshot record retry state")


def _close_snapshot_request_set(
    *,
    items: Sequence[Mapping[str, Any]],
    store: Any,
    image_digest: str,
    module_name: str,
    callable_name: str,
    snapshot_dir: str,
    timeout_seconds: int,
    max_rounds: int = MAX_SNAPSHOT_CLOSURE_ROUNDS,
    cancel_file: Path | None = None,
) -> dict[str, Any]:
    """Record newly exposed request identities until every ICP is stable."""

    if max_rounds < 1:
        raise ValueError("snapshot closure requires at least one round")
    pending = list(enumerate(items, start=1))
    runner_failure_refs: list[str] = []
    completed_rounds = 0
    for round_index in range(1, max_rounds + 1):
        completed_rounds = round_index
        next_pending: list[tuple[int, Mapping[str, Any]]] = []
        for item_index, item in pending:
            ref = str(item["icp_ref"])
            before_count = int(store.snapshot_count())
            try:
                companies = _record_icp_with_retries(
                    image_digest=image_digest,
                    module_name=module_name,
                    callable_name=callable_name,
                    icp=item["icp"],
                    icp_ref=ref,
                    snapshot_dir=snapshot_dir,
                    timeout_seconds=timeout_seconds,
                    reuse_existing=True,
                    item_index=item_index,
                    item_count=len(items),
                    cancel_file=cancel_file,
                )
            except SnapshotRecordingCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - collect every failed ICP
                runner_failure_refs.append(ref)
                print(
                    "WARNING: snapshot closure failed for daily ICP "
                    f"{item_index} in round {round_index}: {type(exc).__name__}"
                )
                continue
            after_count = int(store.snapshot_count())
            if after_count < before_count:
                runner_failure_refs.append(ref)
                print(
                    "WARNING: snapshot count regressed for daily ICP "
                    f"{item_index} in round {round_index}"
                )
                continue
            added_count = after_count - before_count
            print(
                "snapshot closure daily ICP "
                f"{item_index}/{len(items)} round={round_index}: "
                f"{len(companies)} companies, added={added_count}, "
                f"snapshots={after_count}"
            )
            if added_count:
                next_pending.append((item_index, item))

        if runner_failure_refs:
            return {
                "stable": False,
                "rounds": completed_rounds,
                "pending_icp_count": len(next_pending),
                "runner_failure_refs": runner_failure_refs,
            }
        if not next_pending:
            return {
                "stable": True,
                "rounds": completed_rounds,
                "pending_icp_count": 0,
                "runner_failure_refs": [],
            }
        pending = next_pending

    return {
        "stable": False,
        "rounds": completed_rounds,
        "pending_icp_count": len(pending),
        "runner_failure_refs": [],
    }


def _replay_icp_with_docker(
    *,
    image_digest: str,
    module_name: str,
    callable_name: str,
    icp: Mapping[str, Any],
    snapshot_dir: str,
    timeout_seconds: int,
    docker_executable: str = "docker",
) -> list[Mapping[str, Any]]:
    """Replay one ICP with networking disabled to prove the set is complete."""
    from research_lab.eval import private_runtime
    from research_lab.eval.snapshot_store import (
        MISS_POLICY_STRICT,
        container_replay_env,
        dev_replay_bootstrap,
    )

    docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", None)
    if not docker_bootstrap:
        raise RuntimeError("private_runtime docker adapter bootstrap is unavailable")
    container_dir = "/research_lab_dev_snapshots"
    payload = {
        "icp": private_runtime.canonicalize_private_model_icp(icp),
        "context": _snapshot_runtime_context(
            # Strict replay must present the exact same model input as capture.
            # Network isolation and the replay bootstrap enforce replay mode.
            SNAPSHOT_EXECUTION_CONTEXT_MARKER,
            timeout_seconds=timeout_seconds,
        ),
    }
    container_name = "leadpoet-dev-snapshot-replay-" + uuid.uuid4().hex
    env_args: list[str] = []
    for name, value in container_replay_env(
        container_dir, miss_policy=MISS_POLICY_STRICT
    ).items():
        env_args.extend(["-e", f"{name}={value}"])
    # The measured sourcing adapter validates provider-key presence before it
    # issues its first request. Replay never receives real credentials and has
    # no network; these non-secret sentinels only let startup reach the strict
    # request-keyed cache, where every miss still fails closed.
    for group in PROVIDER_KEY_GROUPS:
        for name in group:
            env_args.extend(["-e", f"{name}=research-lab-offline-replay"])
    command = [
        docker_executable,
        "run",
        "--rm",
        "--name",
        container_name,
        "-i",
        "--network",
        "none",
        "-v",
        f"{Path(snapshot_dir).expanduser().resolve()}:{container_dir}:ro",
        *env_args,
        image_digest,
        "python",
        "-c",
        dev_replay_bootstrap() + docker_bootstrap,
        module_name,
        callable_name,
    ]
    completed = _run_named_docker(
        command,
        container_name=container_name,
        input_text=json.dumps(payload, separators=(",", ":"), sort_keys=True),
        timeout_seconds=timeout_seconds,
        environment={**os.environ},
    )
    if SNAPSHOT_MISS_SENTINEL in completed.stderr:
        raise RuntimeError("offline replay observed a strict snapshot miss")
    if completed.returncode != 0:
        raise RuntimeError(
            f"offline replay failed with code {completed.returncode}: "
            f"{completed.stderr[-1200:]}"
        )
    private_runtime.validate_sourcing_runtime_receipt(
        completed.stderr,
        expected_runtime_options=payload["context"]["runtime_options"],
    )
    decoded = json.loads(completed.stdout)
    if not isinstance(decoded, list):
        raise RuntimeError("offline replay adapter must return a JSON array")
    return [dict(item) for item in decoded if isinstance(item, Mapping)]


def _print_plan(
    *,
    dev_set: Any,
    snapshot_dir: str,
    runner_label: str,
    recording: bool,
) -> None:
    print("Research Lab dev-snapshot recorder")
    print(f"  mode:                {'RECORD (live providers)' if recording else 'DRY RUN'}")
    print(f"  daily_bank_hash:     {dev_set.manifest['daily_bank_hash']}")
    print(f"  benchmark_date:      {dev_set.manifest['benchmark_date']}")
    print(f"  bank_icps:           {len(dev_set.items)}")
    print(f"  snapshot_dir:        {snapshot_dir}")
    print(f"  champion_runner:     {runner_label}")
    for group, present in _provider_key_presence().items():
        print(f"  provider_key[{group}]: {'present' if present else 'MISSING'}")


def _recording_failure_summary(
    *,
    runner_failure_refs: Sequence[str],
    failure_file: Path,
) -> dict[str, Any]:
    """Return deduplicated event and affected-ICP counts for one recording run."""
    provider_events: set[tuple[str, str, str]] = set()
    invalid_rows = 0
    if failure_file.exists():
        for line in failure_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_rows += 1
                continue
            if not isinstance(row, Mapping):
                invalid_rows += 1
                continue
            provider_events.add(
                (
                    str(row.get("icp_ref") or ""),
                    str(row.get("request_key") or ""),
                    str(row.get("reason") or "record_failure"),
                )
            )

    runner_refs = {str(ref) for ref in runner_failure_refs if str(ref)}
    provider_refs = {event[0] for event in provider_events if event[0]}
    return {
        "runner_failure_count": len(runner_refs),
        "provider_failure_event_count": len(provider_events) + invalid_rows,
        "failed_icp_count": len(runner_refs | provider_refs),
        "unattributed_provider_failure_count": (
            sum(1 for event in provider_events if not event[0]) + invalid_rows
        ),
        "has_failures": bool(runner_refs or provider_events or invalid_rows),
    }


def _recording_is_complete(
    *,
    closure_result: Mapping[str, Any],
    failure_summary: Mapping[str, Any],
) -> bool:
    return bool(closure_result.get("stable")) and not bool(
        failure_summary.get("has_failures")
    )


def _recorded_provider_model_ids(snapshot_dir: Path) -> list[str]:
    """Load exact OpenRouter model IDs observed by the recording bootstrap."""

    path = snapshot_dir / RECORDED_PROVIDER_MODELS_NAME
    if not path.is_file():
        return []
    model_ids: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, Mapping) or row.get("provider") != "openrouter":
                raise ValueError("recorded provider-model row is invalid")
            model_id = str(row.get("model_id") or "").strip()
            lowered = model_id.lower()
            if (
                not model_id
                or len(model_id) > 500
                or any(marker in lowered for marker in SECRET_MARKERS)
            ):
                raise ValueError("recorded provider model ID is invalid")
            model_ids.add(model_id)
    finally:
        path.unlink(missing_ok=True)
    return sorted(model_ids)


def _resolve_provider_model_ids(
    observed: Sequence[str], declared: Sequence[str]
) -> list[str]:
    actual = sorted({str(item).strip() for item in observed if str(item).strip()})
    allowed = {str(item).strip() for item in declared if str(item).strip()}
    if not actual:
        raise ValueError("champion emitted no attributable OpenRouter model request")
    unexpected = sorted(set(actual) - allowed) if allowed else []
    if unexpected:
        raise ValueError(
            "champion used an OpenRouter model outside the declared allowlist"
        )
    return actual


def _resolve_snapshot_provider_model_ids(
    *,
    store: ProviderSnapshotStore,
    observed: Sequence[str],
    declared: Sequence[str],
) -> list[str]:
    """Bind model provenance only when the snapshots contain OpenRouter traffic."""

    openrouter_model_request_count = store.provider_model_request_counts().get(
        "openrouter", 0
    )
    if openrouter_model_request_count:
        return _resolve_provider_model_ids(observed, declared)
    if any(str(item).strip() for item in observed):
        raise ValueError(
            "recorded OpenRouter model provenance has no model-bearing snapshot request"
        )
    return []


def main() -> int:
    signal.signal(signal.SIGTERM, _terminate_snapshot_recorder_on_signal)
    parser = argparse.ArgumentParser(
        description="Record a frozen provider snapshot set for the L1 dev-eval rung"
    )
    parser.add_argument("--source-icps", required=True, help="Current-day dev ICP export v2 JSON")
    parser.add_argument(
        "--size",
        type=int,
        default=None,
        help=(
            "Compatibility check only; when supplied it must match "
            + RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]
        ),
    )
    parser.add_argument("--snapshot-dir", default="", help="Local snapshot directory (default: RESEARCH_LAB_DEV_SNAPSHOT_URI when it is a local path)")
    parser.add_argument("--adapter-path", default="", help="Private champion checkout for subprocess execution")
    parser.add_argument("--champion-image", default="", help="Immutable champion ECR digest for docker execution")
    parser.add_argument("--source-commit", default=os.getenv("RESEARCH_LAB_PRIVATE_COMMIT_SHA", ""))
    parser.add_argument("--model-config-hash", default=os.getenv("RESEARCH_LAB_PRIVATE_MODEL_CONFIG_HASH", ""))
    parser.add_argument("--private-model-manifest-hash", required=True)
    parser.add_argument(
        "--provider-model-id",
        action="append",
        default=[],
        help=(
            "Optional expected OpenRouter model allowlist. Signed provenance "
            "is always derived from requests observed during recording."
        ),
    )
    parser.add_argument("--module-name", default="research_lab_adapter")
    parser.add_argument("--callable-name", default="run_icp")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=DEFAULT_SNAPSHOT_ICP_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--cancel-file",
        default="",
        help=(
            "Controller-owned cancellation marker checked only between ICP "
            "operations; an in-flight provider call is never interrupted"
        ),
    )
    parser.add_argument("--record", action="store_true", help=f"Actually run the champion with live providers (also requires {RECORD_ENABLED_ENV}=true)")
    args = parser.parse_args()

    try:
        configured_icp_count = (
            ResearchLabGitTreeConfig.from_env().live_max_icps_per_node
        )
    except ResearchLabGitTreeConfigError as exc:
        print(f"ERROR: invalid Git-tree configuration: {exc}")
        return 1
    if not 1 <= configured_icp_count <= MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT:
        print("ERROR: configured Git-tree development ICP count is invalid")
        return 1
    if args.size is not None and args.size != configured_icp_count:
        print(
            "ERROR: --size differs from the configured Git-tree development "
            f"ICP count ({configured_icp_count})"
        )
        return 1

    from research_lab.canonical import utc_now_iso
    from research_lab.eval.dev_eval import build_current_day_dev_bank
    from research_lab.eval.snapshot_store import (
        DevSnapshotStoreError,
        MODE_RECORD,
        SNAPSHOT_URI_ENV,
        ProviderSnapshotStore,
    )

    try:
        source_items, source_bank_manifest = _load_source_export(
            args.source_icps
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}")
        return 1

    try:
        dev_set = build_current_day_dev_bank(
            source_items,
            benchmark_date=str(source_bank_manifest.get("benchmark_date") or ""),
            benchmark_bundle_id=str(
                source_bank_manifest.get("benchmark_bundle_id") or ""
            ),
            benchmark_bundle_hash=str(
                source_bank_manifest.get("benchmark_bundle_hash") or ""
            ),
            rolling_window_hash=str(
                source_bank_manifest.get("rolling_window_hash") or ""
            ),
            private_model_manifest_hash=str(
                source_bank_manifest.get("private_model_manifest_hash") or ""
            ),
            evaluation_epoch=int(
                source_bank_manifest.get("evaluation_epoch") or 0
            ),
        )
        if dev_set.manifest != source_bank_manifest:
            raise ValueError("daily bank manifest differs from exported ICPs")
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: could not validate current-day dev ICP bank: {exc}")
        return 1
    if len(dev_set.items) < configured_icp_count:
        print("ERROR: daily bank is smaller than the configured per-node ICP count")
        return 1

    snapshot_dir = str(args.snapshot_dir or os.getenv(SNAPSHOT_URI_ENV) or "").strip()
    if not snapshot_dir:
        print(f"ERROR: set --snapshot-dir or {SNAPSHOT_URI_ENV}")
        return 1
    if snapshot_dir.startswith("s3://"):
        print(
            "ERROR: recording requires a LOCAL snapshot directory (the record "
            "bootstrap writes files); record locally, then sync to S3."
        )
        return 1

    if args.adapter_path:
        print("ERROR: production snapshot recording requires --champion-image; --adapter-path is not isolated")
        return 1
    if args.champion_image:
        runner_label = f"docker:{args.champion_image}"
    else:
        runner_label = "NOT CONFIGURED (pass --adapter-path or --champion-image)"

    recording = bool(args.record)
    _print_plan(
        dev_set=dev_set,
        snapshot_dir=snapshot_dir,
        runner_label=runner_label,
        recording=recording,
    )

    if not recording:
        print("DRY RUN: no provider calls were made and nothing was written.")
        print(f"Re-run with --record and {RECORD_ENABLED_ENV}=true to record.")
        return 0

    if str(os.getenv(RECORD_ENABLED_ENV) or "").strip().lower() not in TRUTHY_VALUES:
        print(f"ERROR: --record requires {RECORD_ENABLED_ENV}=true")
        return 1
    if not args.champion_image:
        print("ERROR: production recording requires --champion-image for offline replay proof")
        return 1
    if "@sha256:" not in args.champion_image:
        print("ERROR: --champion-image must be an immutable image digest")
        return 1
    if len(str(args.source_commit)) != 40:
        print("ERROR: --source-commit must be the exact 40-character champion commit")
        return 1
    if not str(args.model_config_hash).startswith("sha256:"):
        print("ERROR: --model-config-hash must be an exact sha256 commitment")
        return 1
    if str(args.private_model_manifest_hash) != str(
        dev_set.manifest.get("private_model_manifest_hash") or ""
    ):
        print("ERROR: daily baseline model manifest differs from the active champion")
        return 1
    declared_provider_model_ids = sorted(
        {str(item).strip() for item in args.provider_model_id if str(item).strip()}
    )
    missing = [group for group, present in _provider_key_presence().items() if not present]
    if missing:
        print(f"ERROR: missing provider keys: {', '.join(missing)}")
        return 1

    target = Path(snapshot_dir).expanduser().resolve()
    cancel_file = (
        Path(args.cancel_file).expanduser().resolve() if args.cancel_file else None
    )
    staging = target.with_name(f".{target.name}.recording.{os.getpid()}.{uuid.uuid4().hex}")
    if target.exists():
        print(f"ERROR: immutable snapshot destination already exists: {target}")
        return 1
    staging.mkdir(parents=True, exist_ok=False)
    store = ProviderSnapshotStore(str(staging), mode=MODE_RECORD)
    runner_failure_refs: list[str] = []
    replay_output_hashes: list[dict[str, str]] = []
    try:
        for item_index, item in enumerate(dev_set.items, start=1):
            ref = item["icp_ref"]
            try:
                companies = _record_icp_with_retries(
                    image_digest=args.champion_image,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    icp_ref=ref,
                    snapshot_dir=str(staging),
                    timeout_seconds=args.timeout_seconds,
                    reuse_existing=False,
                    item_index=item_index,
                    item_count=len(dev_set.items),
                    cancel_file=cancel_file,
                )
                print(
                    f"recorded daily ICP {item_index}/{len(dev_set.items)}: "
                    f"{len(companies)} companies, snapshots={store.snapshot_count()}"
                )
            except SnapshotRecordingCancelled:
                raise
            except Exception as exc:  # noqa: BLE001 - collect every failed ICP
                runner_failure_refs.append(str(ref))
                print(
                    f"WARNING: recording failed for daily ICP {item_index}: "
                    f"{type(exc).__name__}"
                )

        failure_file = staging / "record_failures.jsonl"
        initial_failure_summary = _recording_failure_summary(
            runner_failure_refs=runner_failure_refs,
            failure_file=failure_file,
        )
        closure_result: dict[str, Any] = {
            "stable": False,
            "rounds": 0,
            "pending_icp_count": len(dev_set.items),
            "runner_failure_refs": [],
        }
        if not initial_failure_summary["has_failures"]:
            closure_result = _close_snapshot_request_set(
                items=dev_set.items,
                store=store,
                image_digest=args.champion_image,
                module_name=args.module_name,
                callable_name=args.callable_name,
                snapshot_dir=str(staging),
                timeout_seconds=args.timeout_seconds,
                cancel_file=cancel_file,
            )
            runner_failure_refs.extend(closure_result["runner_failure_refs"])

        failure_summary = _recording_failure_summary(
            runner_failure_refs=runner_failure_refs,
            failure_file=failure_file,
        )
        recording_complete = _recording_is_complete(
            closure_result=closure_result,
            failure_summary=failure_summary,
        )
        if failure_summary["provider_failure_event_count"]:
            print(
                "WARNING: "
                f"{failure_summary['provider_failure_event_count']} distinct provider "
                "snapshot failure event(s) recorded"
            )
        if not closure_result["stable"]:
            print(
                "WARNING: snapshot request set did not converge: "
                f"rounds={closure_result['rounds']} "
                f"pending_icps={closure_result['pending_icp_count']}"
            )

        try:
            provider_model_ids = _resolve_snapshot_provider_model_ids(
                store=store,
                observed=_recorded_provider_model_ids(staging),
                declared=declared_provider_model_ids,
            )
        except (
            DevSnapshotStoreError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            print(f"ERROR: could not bind provider-model provenance: {exc}")
            return 1

        store.write_dev_icp_items(dev_set.items)
        if recording_complete:
            for item in dev_set.items:
                _raise_if_snapshot_record_cancelled(cancel_file)
                outputs = _replay_icp_with_docker(
                    image_digest=args.champion_image,
                    module_name=args.module_name,
                    callable_name=args.callable_name,
                    icp=item["icp"],
                    snapshot_dir=str(staging),
                    timeout_seconds=args.timeout_seconds,
                )
                encoded = json.dumps(outputs, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                replay_output_hashes.append(
                    {
                        "icp_hash": str(item["icp_hash"]),
                        "output_hash": "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                    }
                )

        manifest = store.build_manifest(
            icp_set_hash=dev_set.dev_set_hash,
            dev_set_manifest=dev_set.manifest,
            recorded_at=utc_now_iso(),
            provenance={
                "champion_image_digest": args.champion_image,
                "source_commit": str(args.source_commit),
                "model_config_hash": str(args.model_config_hash),
                "private_model_manifest_hash": str(
                    args.private_model_manifest_hash
                ),
                "provider_model_ids": provider_model_ids,
                "replay_output_hashes": replay_output_hashes,
            },
        )
        store.write_manifest(manifest)
        verification = store.verify_manifest(expected_icp_set_hash=dev_set.dev_set_hash)
        ready = store.build_ready_document(manifest)
        if verification["passed"] and recording_complete:
            store.write_ready_document(ready)
        ready_verification = store.verify_ready_document(
            expected_dev_icp_count=configured_icp_count,
            require_signature=False,
        )
        print(f"snapshot_count={manifest['snapshot_count']}")
        print(f"content_hash={manifest['content_hash']}")
        print(f"manifest_hash={manifest['manifest_hash']}")
        print(f"manifest_verified={verification['passed']} errors={verification['errors']}")
        print(f"ready_verified={ready_verification['passed']} errors={ready_verification['errors']}")
        if failure_summary["has_failures"]:
            print(
                "WARNING: snapshot set was not published: "
                f"failed_icps={failure_summary['failed_icp_count']} "
                f"runner_failures={failure_summary['runner_failure_count']} "
                "provider_failure_events="
                f"{failure_summary['provider_failure_event_count']} "
                "unattributed_provider_failures="
                f"{failure_summary['unattributed_provider_failure_count']}"
            )
        if (
            not verification["passed"]
            or not ready_verification["passed"]
            or not recording_complete
        ):
            return 1
        _raise_if_snapshot_record_cancelled(cancel_file)
        os.replace(staging, target)
        print(f"snapshot_ready={target}")
        return 0
    except SnapshotRecordingCancelled as exc:
        print(f"snapshot_record_cancelled={exc}")
        return SNAPSHOT_RECORD_CANCELLED_EXIT_CODE
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

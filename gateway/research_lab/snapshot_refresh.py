"""Single-worker refresh controller for immutable inner-loop dev snapshots."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Awaitable, Callable, Iterator, Mapping, Sequence

from gateway.research_lab.config import (
    DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
    RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD,
)
from gateway.research_lab.dev_eval_runner import snapshot_readiness
from gateway.research_lab.git_tree_models import TreePolicy
from gateway.research_lab.promotion import load_active_private_model
from research_lab.eval.snapshot_store import POINTER_NAME, SNAPSHOT_URI_ENV

logger = logging.getLogger(__name__)

AUTO_REFRESH_ENABLED_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_AUTO_REFRESH_ENABLED"
RECORD_ENABLED_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_RECORD_ENABLED"
KMS_KEY_ID_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_KMS_KEY_ID"
PROVIDER_MODEL_IDS_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_PROVIDER_MODEL_IDS"
RUNTIME_SOURCE_ROOT_ENV = "RESEARCH_LAB_RUNTIME_SOURCE_ROOT"
REFRESH_STATE_PATH_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_STATE_PATH"
REFRESH_WORK_ROOT_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_WORK_ROOT"
CHECK_INTERVAL_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_CHECK_INTERVAL_SECONDS"
REFRESH_INTERVAL_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_INTERVAL_SECONDS"
RETRY_INTERVAL_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_RETRY_SECONDS"
COMMAND_TIMEOUT_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_REFRESH_COMMAND_TIMEOUT_SECONDS"
SELECTION_SEED_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_SELECTION_SEED"

DEFAULT_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
DEFAULT_REFRESH_INTERVAL_SECONDS = 7 * 24 * 60 * 60
DEFAULT_RETRY_INTERVAL_SECONDS = 60 * 60
DEFAULT_COMMAND_TIMEOUT_SECONDS = 3 * 60 * 60
_TRUTHY = frozenset({"1", "true", "yes", "on"})

CommandRunner = Callable[[Sequence[str], Mapping[str, str], int], str]
ReadinessLoader = Callable[..., Mapping[str, Any]]
ActiveLoader = Callable[..., Awaitable[Any]]


def snapshot_auto_refresh_enabled() -> bool:
    return str(os.getenv(AUTO_REFRESH_ENABLED_ENV) or "").strip().lower() in _TRUTHY


def _positive_env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(str(os.getenv(name) or default).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _state_path() -> Path:
    raw = str(os.getenv(REFRESH_STATE_PATH_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".config" / "leadpoet" / "research-lab-dev-snapshot-refresh.json"


def _work_root() -> Path:
    raw = str(os.getenv(REFRESH_WORK_ROOT_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "research_lab_dev_snapshot_refresh"


def _load_state(path: Path) -> dict[str, Any]:
    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return dict(decoded) if isinstance(decoded, Mapping) else {}


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    staging.write_text(
        json.dumps(dict(state), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.chmod(staging, 0o600)
    os.replace(staging, path)


@contextmanager
def _exclusive_refresh_lock(path: Path) -> Iterator[bool]:
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _provider_model_ids() -> list[str]:
    raw = str(os.getenv(PROVIDER_MODEL_IDS_ENV) or "").strip()
    if not raw:
        return []
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        decoded = [item.strip() for item in raw.split(",")]
    if not isinstance(decoded, list):
        return []
    return sorted({str(item).strip() for item in decoded if str(item).strip()})


def _s3_base_uri(pointer_uri: str) -> str:
    normalized = str(pointer_uri or "").strip().rstrip("/")
    if not normalized.startswith("s3://") or not normalized.endswith(f"/{POINTER_NAME}"):
        raise RuntimeError(
            f"{SNAPSHOT_URI_ENV} must be an s3://.../{POINTER_NAME} pointer"
        )
    return normalized.rsplit("/", 1)[0]


def _runtime_script(source_root: Path, name: str) -> str:
    script = (source_root / "scripts" / name).resolve()
    if not script.is_file():
        raise RuntimeError(f"snapshot pipeline script is missing: {script}")
    return str(script)


def _run_command(command: Sequence[str], env: Mapping[str, str], timeout_seconds: int) -> str:
    completed = subprocess.run(
        list(command),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
        env=dict(env),
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "")[-1600:]
        raise RuntimeError(
            f"snapshot pipeline command failed ({Path(command[1]).name}, "
            f"exit={completed.returncode}): {detail}"
        )
    return str(completed.stdout or "")


def _artifact_identity(active: Any) -> tuple[str, str, str]:
    artifact = active.artifact
    return (
        str(artifact.image_digest or ""),
        str(artifact.git_commit_sha or ""),
        str(artifact.config_hash or ""),
    )


def _readiness_matches_artifact(
    readiness: Mapping[str, Any], identity: tuple[str, str, str]
) -> bool:
    return bool(
        readiness.get("ready")
        and str(readiness.get("champion_image_digest") or "") == identity[0]
        and str(readiness.get("source_commit") or "") == identity[1]
        and str(readiness.get("model_config_hash") or "") == identity[2]
    )


def _refresh_reason(
    readiness: Mapping[str, Any], identity: tuple[str, str, str]
) -> str:
    if not readiness.get("ready"):
        return str(readiness.get("reason") or "snapshot_not_ready")
    if not _readiness_matches_artifact(readiness, identity):
        return "snapshot_active_model_mismatch"
    age = readiness.get("snapshot_age_seconds")
    try:
        age_seconds = float(age)
    except (TypeError, ValueError):
        return "snapshot_age_invalid"
    if age_seconds >= _positive_env_int(
        REFRESH_INTERVAL_ENV, DEFAULT_REFRESH_INTERVAL_SECONDS, minimum=300
    ):
        return "weekly_refresh_due"
    return ""


def _safe_error(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    for marker in ("authorization", "api_key", "service_role", "sb_secret", "sk-or-"):
        if marker in text.lower():
            return f"{type(exc).__name__}:redacted_sensitive_error"
    return f"{type(exc).__name__}:{text[:300]}"


async def maybe_refresh_dev_snapshot(
    config: Any,
    *,
    worker_index: int,
    tree_policy: TreePolicy,
    now: float | None = None,
    command_runner: CommandRunner = _run_command,
    readiness_loader: ReadinessLoader = snapshot_readiness,
    active_loader: ActiveLoader = load_active_private_model,
) -> dict[str, Any]:
    """Check daily and refresh weekly; a failed run never mutates current.json."""

    if int(worker_index) != 0:
        return {"status": "skipped", "reason": "not_refresh_worker"}
    if tree_policy.mode != "active":
        return {"status": "skipped", "reason": "tree_mode_not_active"}
    if not snapshot_auto_refresh_enabled():
        return {"status": "skipped", "reason": "auto_refresh_disabled"}

    timestamp = float(now if now is not None else time.time())
    state_path = _state_path()
    state = _load_state(state_path)
    check_interval = _positive_env_int(
        CHECK_INTERVAL_ENV, DEFAULT_CHECK_INTERVAL_SECONDS, minimum=300
    )
    retry_interval = _positive_env_int(
        RETRY_INTERVAL_ENV, DEFAULT_RETRY_INTERVAL_SECONDS, minimum=60
    )
    last_check = float(state.get("last_check_unix") or 0.0)
    retry_after = retry_interval if state.get("last_error") else check_interval
    if last_check and timestamp - last_check < retry_after:
        return {"status": "skipped", "reason": "check_not_due"}

    with _exclusive_refresh_lock(state_path) as acquired:
        if not acquired:
            return {"status": "skipped", "reason": "refresh_lock_held"}
        state = _load_state(state_path)
        last_check = float(state.get("last_check_unix") or 0.0)
        retry_after = retry_interval if state.get("last_error") else check_interval
        if last_check and timestamp - last_check < retry_after:
            return {"status": "skipped", "reason": "check_not_due_after_lock"}

        pointer_uri = str(
            os.getenv(SNAPSHOT_URI_ENV) or DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI
        ).strip()
        try:
            active_before = await active_loader(config, register_bootstrap=False)
            identity_before = _artifact_identity(active_before)
            readiness = await asyncio.to_thread(
                readiness_loader,
                pointer_uri,
                expected_dev_icp_count=tree_policy.live_max_icps_per_node,
            )
            reason = _refresh_reason(readiness, identity_before)
            base_state = {
                "schema_version": "research_lab.dev_snapshot_refresh_state.v1",
                "last_check_unix": timestamp,
                "last_check_at": datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat(),
                "active_image_digest": identity_before[0],
                "snapshot_manifest_hash": str(readiness.get("manifest_hash") or ""),
                "snapshot_ready": bool(readiness.get("ready")),
            }
            if not reason:
                healthy = {**base_state, "status": "healthy", "last_error": ""}
                _write_state(state_path, healthy)
                return healthy

            if str(os.getenv(RECORD_ENABLED_ENV) or "").strip().lower() not in _TRUTHY:
                raise RuntimeError(f"{RECORD_ENABLED_ENV} must be true for automatic refresh")
            kms_key_id = str(os.getenv(KMS_KEY_ID_ENV) or "").strip()
            if not kms_key_id:
                raise RuntimeError(f"{KMS_KEY_ID_ENV} is required")
            provider_model_ids = _provider_model_ids()
            if not provider_model_ids:
                raise RuntimeError(f"{PROVIDER_MODEL_IDS_ENV} is required")
            raw_source_root = str(os.getenv(RUNTIME_SOURCE_ROOT_ENV) or "").strip()
            if not raw_source_root:
                raise RuntimeError(f"{RUNTIME_SOURCE_ROOT_ENV} is required")
            source_root = Path(raw_source_root).expanduser().resolve()
            if not source_root.is_dir():
                raise RuntimeError(f"{RUNTIME_SOURCE_ROOT_ENV} is not a directory")
            base_uri = _s3_base_uri(pointer_uri)
            seed = str(os.getenv(SELECTION_SEED_ENV) or "research-lab-dev-v1").strip()
            timeout_seconds = _positive_env_int(
                COMMAND_TIMEOUT_ENV, DEFAULT_COMMAND_TIMEOUT_SECONDS, minimum=300
            )
            work_root = _work_root()
            work_root.mkdir(parents=True, exist_ok=True)
            work_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"refresh-{os.getpid()}-", dir=str(work_root.resolve())
                )
            )
            os.chmod(work_dir, 0o700)
            inputs_dir = work_dir / "inputs"
            snapshot_dir = work_dir / "snapshot"
            env = dict(os.environ)
            env[
                RESEARCH_LAB_GIT_TREE_ENV_BY_FIELD["live_max_icps_per_node"]
            ] = str(tree_policy.live_max_icps_per_node)
            try:
                await asyncio.to_thread(
                    command_runner,
                    (
                        sys.executable,
                        _runtime_script(source_root, "export_research_lab_dev_icp_inputs.py"),
                        "--out-dir",
                        str(inputs_dir),
                        "--seed",
                        seed,
                    ),
                    env,
                    timeout_seconds,
                )
                record_command: list[str] = [
                    sys.executable,
                    _runtime_script(source_root, "record_research_lab_dev_snapshots.py"),
                    "--source-icps",
                    str(inputs_dir / "source_icps.json"),
                    "--exclude-hashes",
                    str(inputs_dir / "holdout_window_hashes.json"),
                    "--seed",
                    seed,
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--champion-image",
                    identity_before[0],
                    "--source-commit",
                    identity_before[1],
                    "--model-config-hash",
                    identity_before[2],
                    "--record",
                ]
                for model_id in provider_model_ids:
                    record_command.extend(("--provider-model-id", model_id))
                await asyncio.to_thread(
                    command_runner, tuple(record_command), env, timeout_seconds
                )

                publish_base = [
                    sys.executable,
                    _runtime_script(source_root, "publish_research_lab_dev_snapshot.py"),
                    "--source-dir",
                    str(snapshot_dir),
                    "--s3-base-uri",
                    base_uri,
                    "--kms-key-id",
                    kms_key_id,
                ]
                await asyncio.to_thread(
                    command_runner,
                    tuple([*publish_base, "--skip-current-pointer"]),
                    env,
                    timeout_seconds,
                )

                active_after_record = await active_loader(
                    config, register_bootstrap=False
                )
                if _artifact_identity(active_after_record) != identity_before:
                    raise RuntimeError(
                        "active private model changed before snapshot pointer promotion"
                    )
                await asyncio.to_thread(
                    command_runner, tuple(publish_base), env, timeout_seconds
                )
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)

            final_readiness = await asyncio.to_thread(
                readiness_loader,
                pointer_uri,
                expected_dev_icp_count=tree_policy.live_max_icps_per_node,
            )
            if not _readiness_matches_artifact(final_readiness, identity_before):
                raise RuntimeError("published snapshot does not match the active private model")
            completed = {
                **base_state,
                "status": "refreshed",
                "refresh_reason": reason,
                "last_refresh_unix": timestamp,
                "last_refresh_at": datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat(),
                "snapshot_manifest_hash": str(
                    final_readiness.get("manifest_hash") or ""
                ),
                "snapshot_ready": True,
                "last_error": "",
            }
            _write_state(state_path, completed)
            logger.info(
                "research_lab_dev_snapshot_refresh_complete reason=%s manifest_hash=%s",
                reason,
                completed["snapshot_manifest_hash"][:24],
            )
            return completed
        except Exception as exc:
            failure = {
                "schema_version": "research_lab.dev_snapshot_refresh_state.v1",
                "status": "failed",
                "last_check_unix": timestamp,
                "last_check_at": datetime.fromtimestamp(
                    timestamp, tz=timezone.utc
                ).isoformat(),
                "last_error": _safe_error(exc),
            }
            _write_state(state_path, failure)
            logger.warning(
                "research_lab_dev_snapshot_refresh_failed error=%s",
                failure["last_error"],
            )
            return failure

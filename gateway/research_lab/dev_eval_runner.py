"""§6.3 docker replay dev-eval runner: the worker-side ``dev_evaluator`` seam.

``CodeEditLoopEngine`` exposes a deliberately-unwired ``dev_evaluator``
callable (one built candidate in, a ``DevEvalResult.to_dict()``-shaped mapping
out). This module supplies the production implementation:

1. Resolve the frozen snapshot set from ``RESEARCH_LAB_DEV_SNAPSHOT_URI``.
   S3 prefixes are synced once into a local content-addressed cache directory
   because the candidate container needs a bind-mountable path.
2. Load the dev ICP payloads (``dev_icps.json``, written by
   ``scripts/record_research_lab_dev_snapshots.py``) and verify them against
   the manifest's ``icp_set_hash`` (leak/tamper guard).
3. For each dev ICP, ``docker run`` the candidate's just-built image with the
   snapshot directory mounted read-only, networking disabled
   (``--network none``), ``dev_replay_bootstrap()`` prepended to the adapter
   bootstrap, and ``container_replay_env()`` exported — the container serves
   all provider traffic from the frozen snapshots and cannot open a live
   connection.
4. Score the outputs with ``research_lab.eval.dev_eval.evaluate_dev``
   (mechanical scorer, capped-top-5 per-ICP arithmetic) and return
   ``DevEvalResult.to_dict()``.

Guardrails (plan §3, non-negotiable):
- Dev scores are RANKING-ONLY. They order candidates within a run and inform
  later drafts in the same run; they are never promotion evidence.
- Any evaluation failure produces an explicitly ineligible candidate. The
  selector then ignores every development score for that run and preserves
  ordinary build order; dev-eval never fails a run that built an image.
  The wall-clock cap below also keeps a wedged replay from consuming the
  loop's ``RESEARCH_LAB_AUTO_RESEARCH_MAX_SECONDS`` envelope.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import fcntl
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research_lab.eval.dev_eval import compute_dev_set_hash, evaluate_dev
from research_lab.eval.snapshot_store import (
    DEV_ICPS_NAME,
    MANIFEST_NAME,
    READY_NAME,
    EXPECTED_DEV_ICP_COUNT,
    MISS_POLICY_STRICT,
    MODE_REPLAY,
    SNAPSHOT_SUBDIR,
    SNAPSHOT_URI_ENV,
    SNAPSHOT_MISS_SENTINEL,
    SnapshotMiss,
    ProviderSnapshotStore,
    container_replay_env,
    default_miss_policy,
    dev_replay_bootstrap,
    resolve_snapshot_uri,
)

logger = logging.getLogger(__name__)

DEV_EVAL_ENABLED_ENV = "RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED"
DEV_EVAL_TOTAL_TIMEOUT_ENV = "RESEARCH_LAB_LOOP_DEV_EVAL_TIMEOUT_SECONDS"
DEV_EVAL_ICP_TIMEOUT_ENV = "RESEARCH_LAB_LOOP_DEV_EVAL_ICP_TIMEOUT_SECONDS"
DEV_EVAL_CACHE_DIR_ENV = "RESEARCH_LAB_DEV_SNAPSHOT_CACHE_DIR"
DEFAULT_TOTAL_TIMEOUT_SECONDS = 300
CONTAINER_SNAPSHOT_DIR = "/research_lab_dev_snapshots"
_TRUTHY = ("1", "true", "yes", "on")
_PROVIDER_CREDENTIAL_ENV_NAMES = frozenset(
    {
        "DEEPLINE_API_KEY",
        "EXA_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "QUALIFICATION_OPENROUTER_API_KEY",
        "QUALIFICATION_SCRAPINGDOG_API_KEY",
        "SCRAPINGDOG_API_KEY",
    }
)


class DevEvalRunnerError(RuntimeError):
    """Raised when the replay runner cannot evaluate a candidate."""


def dev_eval_runner_enabled() -> bool:
    """Mirror of the engine's ``_dev_eval_enabled`` gate (env-driven)."""
    raw = str(os.getenv(DEV_EVAL_ENABLED_ENV) or "").strip().lower()
    return raw in _TRUTHY


def dev_eval_total_timeout_seconds() -> int:
    raw = str(os.getenv(DEV_EVAL_TOTAL_TIMEOUT_ENV) or "").strip()
    try:
        value = int(raw) if raw else DEFAULT_TOTAL_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_TOTAL_TIMEOUT_SECONDS
    return max(30, value)


def _per_icp_timeout_seconds(item_count: int) -> int:
    raw = str(os.getenv(DEV_EVAL_ICP_TIMEOUT_ENV) or "").strip()
    if raw:
        try:
            return max(10, int(raw))
        except ValueError:
            pass
    total = dev_eval_total_timeout_seconds()
    return max(30, total // max(1, item_count))


def _default_cache_root() -> Path:
    raw = str(os.getenv(DEV_EVAL_CACHE_DIR_ENV) or "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(tempfile.gettempdir()) / "research_lab_dev_snapshot_cache"


def ensure_local_snapshot_set(root_uri: str, *, cache_root: Path | None = None) -> Path:
    """Return a local directory holding the snapshot set.

    Local URIs are used in place. S3 prefixes are synced once into a
    content-addressed cache directory (keyed by the manifest hash) so
    repeated candidates in one run — and repeated runs on one worker —
    reuse the same download. The ``.complete`` marker makes a partially
    synced directory invisible to readers.
    """
    configured_uri = str(root_uri or "").strip()
    if not configured_uri:
        raise DevEvalRunnerError(f"snapshot root URI is required ({SNAPSHOT_URI_ENV})")
    try:
        resolution = resolve_snapshot_uri(configured_uri)
    except Exception as exc:
        raise DevEvalRunnerError(str(exc)) from exc
    uri = str(resolution.get("snapshot_uri") or "").strip()
    expected_manifest_hash = str(resolution.get("manifest_hash") or "")
    expected_ready_hash = str(resolution.get("ready_hash") or "")
    if not uri.startswith("s3://"):
        local = Path(uri).expanduser()
        verification = ProviderSnapshotStore(str(local), mode=MODE_REPLAY).verify_ready_document(
            require_signature=False
        )
        if not verification.get("passed"):
            raise DevEvalRunnerError(
                "snapshot set is not READY: " + "; ".join(verification.get("errors") or ())
            )
        _verify_pointer_bindings(
            verification,
            expected_manifest_hash=expected_manifest_hash,
            expected_ready_hash=expected_ready_hash,
        )
        return local

    remote = ProviderSnapshotStore(uri, mode=MODE_REPLAY)
    ready_verification = remote.verify_ready_document(require_signature=True)
    if not ready_verification.get("passed"):
        raise DevEvalRunnerError(
            "remote snapshot set is not READY: "
            + "; ".join(ready_verification.get("errors") or ())
        )
    _verify_pointer_bindings(
        ready_verification,
        expected_manifest_hash=expected_manifest_hash,
        expected_ready_hash=expected_ready_hash,
    )
    manifest = remote.load_manifest()
    if manifest is None:
        raise DevEvalRunnerError(f"snapshot manifest missing under {uri}")
    manifest_hash = str(manifest.get("manifest_hash") or "")
    if not manifest_hash:
        raise DevEvalRunnerError("snapshot manifest carries no manifest_hash")

    root = cache_root if cache_root is not None else _default_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    target = root / manifest_hash.replace("sha256:", "")
    lock_path = root / f".{target.name}.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        marker = target / ".complete"
        if marker.is_file():
            cached = ProviderSnapshotStore(str(target), mode=MODE_REPLAY)
            verification = cached.verify_ready_document(require_signature=True)
            if verification.get("passed"):
                _verify_pointer_bindings(
                    verification,
                    expected_manifest_hash=expected_manifest_hash,
                    expected_ready_hash=expected_ready_hash,
                )
                return target
            shutil.rmtree(target, ignore_errors=True)

        staging = root / f".{target.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
        (staging / SNAPSHOT_SUBDIR).mkdir(parents=True, exist_ok=False)
        try:
            for relative in (MANIFEST_NAME, DEV_ICPS_NAME):
                raw = remote._read_text(relative)  # noqa: SLF001 - storage seam
                if raw is None:
                    raise DevEvalRunnerError(f"snapshot set is missing {relative}")
                (staging / relative).write_text(raw, encoding="utf-8")
            for name in remote._list_snapshot_names():  # noqa: SLF001
                raw = remote._read_text(f"{SNAPSHOT_SUBDIR}/{name}")  # noqa: SLF001
                if raw is None:
                    raise DevEvalRunnerError(f"snapshot object vanished during sync: {name}")
                (staging / SNAPSHOT_SUBDIR / name).write_text(raw, encoding="utf-8")
            ready_raw = remote._read_text(READY_NAME)  # noqa: SLF001
            if ready_raw is None:
                raise DevEvalRunnerError("snapshot READY document vanished during sync")
            (staging / READY_NAME).write_text(ready_raw, encoding="utf-8")
            staged = ProviderSnapshotStore(str(staging), mode=MODE_REPLAY)
            verification = staged.verify_ready_document(require_signature=True)
            if not verification.get("passed"):
                raise DevEvalRunnerError(
                    "downloaded snapshot verification failed: "
                    + "; ".join(verification.get("errors") or ())
                )
            _verify_pointer_bindings(
                verification,
                expected_manifest_hash=expected_manifest_hash,
                expected_ready_hash=expected_ready_hash,
            )
            (staging / ".complete").write_text(manifest_hash + "\n", encoding="utf-8")
            os.replace(staging, target)
            return target
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)


def _verify_pointer_bindings(
    verification: Mapping[str, Any],
    *,
    expected_manifest_hash: str,
    expected_ready_hash: str,
) -> None:
    if expected_manifest_hash and str(verification.get("manifest_hash") or "") != expected_manifest_hash:
        raise DevEvalRunnerError("snapshot pointer manifest hash does not match immutable target")
    if expected_ready_hash and str(verification.get("ready_hash") or "") != expected_ready_hash:
        raise DevEvalRunnerError("snapshot pointer READY hash does not match immutable target")


def snapshot_readiness(
    root_uri: str,
    *,
    cache_root: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Return fail-closed snapshot preflight evidence for activation policy."""
    if default_miss_policy() != MISS_POLICY_STRICT:
        return {"ready": False, "reason": "snapshot_miss_policy_must_be_strict"}
    try:
        resolution = resolve_snapshot_uri(root_uri)
        local = ensure_local_snapshot_set(root_uri, cache_root=cache_root)
        store = ProviderSnapshotStore(str(local), mode=MODE_REPLAY, miss_policy=MISS_POLICY_STRICT)
        verification = store.verify_ready_document(
            require_signature=str(root_uri or "").startswith("s3://")
        )
        if not verification.get("passed"):
            return {
                "ready": False,
                "reason": "snapshot_ready_verification_failed",
                "errors": list(verification.get("errors") or ()),
            }
        manifest = store.load_manifest() or {}
        provenance = (
            dict(manifest.get("provenance") or {})
            if isinstance(manifest.get("provenance"), Mapping)
            else {}
        )
        items = load_verified_dev_items(store)
        recorded_at = str(manifest.get("recorded_at") or "")
        recorded = datetime.fromisoformat(recorded_at.replace("Z", "+00:00"))
        if recorded.tzinfo is None:
            recorded = recorded.replace(tzinfo=timezone.utc)
        age = max(
            0.0,
            ((now or datetime.now(timezone.utc)) - recorded.astimezone(timezone.utc)).total_seconds(),
        )
        raw_provider_ids = provenance.get("provider_model_ids") or ()
        provider_ids = (
            [str(item) for item in raw_provider_ids if str(item).strip()]
            if isinstance(raw_provider_ids, (list, tuple, set))
            else []
        )
        return {
            "ready": len(items) == EXPECTED_DEV_ICP_COUNT,
            "reason": "ready" if len(items) == EXPECTED_DEV_ICP_COUNT else "dev_set_size_must_equal_eight",
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "dev_set_hash": str(manifest.get("icp_set_hash") or ""),
            "recorded_at": recorded_at,
            "snapshot_age_seconds": age,
            "dev_set_size": len(items),
            "ready_hash": str(verification.get("ready_hash") or ""),
            "configured_snapshot_uri": str(root_uri or ""),
            "resolved_snapshot_uri": str(resolution.get("snapshot_uri") or ""),
            "pointer_hash": str(resolution.get("pointer_hash") or ""),
            "champion_image_digest": str(
                provenance.get("champion_image_digest") or ""
            ),
            "source_commit": str(provenance.get("source_commit") or ""),
            "model_config_hash": str(
                provenance.get("model_config_hash") or ""
            ),
            "provider_model_ids": provider_ids,
        }
    except Exception as exc:
        return {
            "ready": False,
            "reason": f"snapshot_preflight_error:{type(exc).__name__}",
            "error": str(exc)[:240],
        }


def load_verified_dev_items(store: ProviderSnapshotStore) -> list[dict[str, Any]]:
    """Load dev ICP payloads and bind them to the manifest's icp_set_hash."""
    items = store.load_dev_icp_items()
    if not items:
        raise DevEvalRunnerError("snapshot set carries no dev ICP payloads")
    manifest = store.load_manifest()
    expected = str((manifest or {}).get("icp_set_hash") or "")
    if expected and compute_dev_set_hash(items) != expected:
        raise DevEvalRunnerError(
            "dev_icps.json does not match the manifest icp_set_hash "
            "(tampered or mixed snapshot vintages)"
        )
    return items


class DockerReplayDevEvaluator:
    """Callable satisfying the engine's ``dev_evaluator`` seam contract."""

    def __init__(
        self,
        *,
        snapshot_uri: str | None = None,
        cache_root: Path | None = None,
        docker_executable: str = "docker",
        run_icp_in_docker: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
    ) -> None:
        self._snapshot_uri = str(
            snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
        ).strip()
        self._cache_root = cache_root
        self._docker_executable = docker_executable
        # Test seam: replaces the docker invocation, everything else is real.
        self._run_icp_in_docker = run_icp_in_docker or self._run_icp_in_docker_default
        self._prepare_lock: asyncio.Lock | None = None
        self._local_dir: Path | None = None
        self._replay_store: ProviderSnapshotStore | None = None
        self._dev_items: list[dict[str, Any]] | None = None

    async def _ensure_prepared(self) -> tuple[Path, ProviderSnapshotStore, list[dict[str, Any]]]:
        if self._prepare_lock is None:
            self._prepare_lock = asyncio.Lock()
        async with self._prepare_lock:
            if self._local_dir is None or self._replay_store is None or self._dev_items is None:
                local_dir = await asyncio.to_thread(
                    ensure_local_snapshot_set, self._snapshot_uri, cache_root=self._cache_root
                )
                replay_store = ProviderSnapshotStore(
                    str(local_dir), mode=MODE_REPLAY, miss_policy=MISS_POLICY_STRICT
                )
                dev_items = await asyncio.to_thread(load_verified_dev_items, replay_store)
                self._local_dir = local_dir
                self._replay_store = replay_store
                self._dev_items = dev_items
            return self._local_dir, self._replay_store, self._dev_items

    async def __call__(self, candidate: Any) -> Mapping[str, Any]:
        """Evaluate one built candidate; raises on failure (engine catches)."""
        image_digest = str(
            candidate.build.candidate_model_manifest.image_digest or ""
        ).strip()
        if "@sha256:" not in image_digest:
            raise DevEvalRunnerError("candidate image digest is not immutable")
        local_dir, replay_store, dev_items = await self._ensure_prepared()
        per_icp_timeout = _per_icp_timeout_seconds(len(dev_items))

        async def _runner(icp: Mapping[str, Any], context: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
            return await asyncio.to_thread(
                self._run_icp_in_docker,
                image_digest=image_digest,
                icp=icp,
                context=context,
                snapshot_dir=local_dir,
                timeout_seconds=per_icp_timeout,
            )

        result = await asyncio.wait_for(
            evaluate_dev(
                candidate_runner=_runner,
                dev_items=dev_items,
                snapshot_store=replay_store,
                run_label=str(candidate.node_id or ""),
                install_replay_seams=False,
                require_manifest=True,
            ),
            timeout=dev_eval_total_timeout_seconds(),
        )
        # Coverage/miss-rate monitoring line (plan "Monitoring" section): one
        # structured record per evaluated candidate, greppable by lane once
        # joined with the candidate's lane in the loop events.
        logger.info(
            "research_lab_loop_dev_eval_result node_id=%s lane=%s aggregate_dev_score=%s "
            "icp_count=%s scored_icp_count=%s snapshot_miss_count=%s failure_count=%s "
            "miss_policy=%s dev_score_version=%s",
            str(candidate.node_id or "")[:80],
            str(getattr(getattr(candidate, "draft", None), "lane", "") or "")[:80],
            round(result.aggregate_dev_score, 6),
            result.icp_count,
            result.scored_icp_count,
            result.snapshot_miss_count,
            result.failure_count,
            replay_store.miss_policy,
            result.dev_score_version,
        )
        return result.to_dict()

    def _run_icp_in_docker_default(
        self,
        *,
        image_digest: str,
        icp: Mapping[str, Any],
        context: Mapping[str, Any],
        snapshot_dir: Path,
        timeout_seconds: int,
    ) -> list[Mapping[str, Any]]:
        """Run one dev ICP inside the candidate image against the replay seams.

        Mirrors ``DockerPrivateModelRunner._run_json`` / the recording CLI's
        ``_record_icp_with_docker``, but read-only-mounts the snapshot
        directory and prepends the replay bootstrap so all provider traffic
        is served from the frozen set.
        """
        from research_lab.eval import private_runtime

        docker_bootstrap = getattr(private_runtime, "_DOCKER_ADAPTER_BOOTSTRAP", None)
        if not docker_bootstrap:
            raise DevEvalRunnerError("private_runtime docker adapter bootstrap is unavailable")
        payload = {
            "icp": private_runtime.canonicalize_private_model_icp(icp),
            "context": dict(context),
        }
        env_args: list[str] = []
        # Provider key names pass through so adapter import-time checks hold;
        # the replay bootstrap guarantees no live connection is opened.
        for name in private_runtime.private_model_env_passthrough():
            if name in os.environ:
                env_args.extend(["-e", name])
        for name, value in container_replay_env(
            CONTAINER_SNAPSHOT_DIR, miss_policy=MISS_POLICY_STRICT
        ).items():
            env_args.extend(["-e", f"{name}={value}"])
        platform = str(getattr(private_runtime, "_default_docker_platform")() or "").strip()
        platform_args = ["--platform", platform] if platform else []
        command = [
            self._docker_executable,
            "run",
            "--rm",
            "-i",
            # Replay serves every provider call from the mounted snapshot
            # directory, so the container needs no network at all; with it
            # removed, any HTTP path the replay seams don't cover fails
            # loudly (per-ICP failure) instead of reaching live providers.
            "--network",
            "none",
            *platform_args,
            "-v",
            f"{Path(snapshot_dir).resolve()}:{CONTAINER_SNAPSHOT_DIR}:ro",
            *env_args,
            image_digest,
            "python",
            "-c",
            dev_replay_bootstrap() + docker_bootstrap,
            "research_lab_adapter",
            "run_icp",
        ]
        completed = subprocess.run(
            command,
            input=json.dumps(payload, separators=(",", ":"), sort_keys=True),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env={**os.environ},
            check=False,
        )
        if completed.returncode != 0:
            stderr = str(completed.stderr or "")
            if SNAPSHOT_MISS_SENTINEL in stderr:
                request_key = stderr.rsplit(SNAPSHOT_MISS_SENTINEL, 1)[-1].splitlines()[0]
                raise SnapshotMiss(request_key.strip())
            raise DevEvalRunnerError(
                f"dev replay adapter failed with code {completed.returncode}: "
                f"{stderr[-1200:]}"
            )
        decoded = json.loads(completed.stdout)
        if not isinstance(decoded, list):
            raise DevEvalRunnerError("dev replay adapter must return a JSON array")
        return [item for item in decoded if isinstance(item, Mapping)]


class AttestedReplayDevEvaluatorV2:
    """The existing dev-evaluator seam backed by the measured scoring EIF."""

    def __init__(
        self,
        *,
        epoch_id: int,
        worker_index: int,
        snapshot_uri: str | None = None,
        cache_root: Path | None = None,
        execute: Any = None,
    ) -> None:
        self._epoch_id = max(0, int(epoch_id))
        self._worker_index = int(worker_index)
        self._snapshot_uri = str(
            snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
        ).strip()
        self._cache_root = cache_root
        self._execute = execute
        self._prepare_lock: asyncio.Lock | None = None
        self._local_dir: Path | None = None
        self._replay_store: ProviderSnapshotStore | None = None
        self._snapshot_bundle: dict[str, Any] | None = None

    async def _ensure_prepared(
        self,
    ) -> tuple[Path, ProviderSnapshotStore, dict[str, Any]]:
        if self._prepare_lock is None:
            self._prepare_lock = asyncio.Lock()
        async with self._prepare_lock:
            if (
                self._local_dir is None
                or self._replay_store is None
                or self._snapshot_bundle is None
            ):
                from gateway.tee.source_bundle_v2 import build_source_bundle_v2

                local_dir = await asyncio.to_thread(
                    ensure_local_snapshot_set,
                    self._snapshot_uri,
                    cache_root=self._cache_root,
                )
                replay_store = ProviderSnapshotStore(
                    str(local_dir),
                    mode=MODE_REPLAY,
                    miss_policy=MISS_POLICY_STRICT,
                )
                await asyncio.to_thread(load_verified_dev_items, replay_store)
                verification = await asyncio.to_thread(replay_store.verify_manifest)
                if not verification.get("passed"):
                    raise DevEvalRunnerError(
                        "snapshot-set manifest failed verification: "
                        + "; ".join(verification.get("errors") or ())
                    )
                snapshot_bundle = await asyncio.to_thread(
                    build_source_bundle_v2,
                    local_dir,
                )
                self._local_dir = local_dir
                self._replay_store = replay_store
                self._snapshot_bundle = snapshot_bundle
            return (
                self._local_dir,
                self._replay_store,
                dict(self._snapshot_bundle),
            )

    async def __call__(self, candidate: Any) -> Mapping[str, Any]:
        from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
        from gateway.research_lab.model_authority_v2 import (
            source_bundle_for_artifact_v2,
        )
        from gateway.tee.scoring_executor_v2 import (
            DEV_REPLAY_REQUEST_SCHEMA_VERSION,
            OP_DEV_REPLAY_V2,
        )
        from leadpoet_canonical.attested_v2 import sha256_json
        from research_lab.eval.private_runtime import private_model_env_passthrough

        artifact = candidate.build.candidate_model_manifest
        image_digest = str(artifact.image_digest or "").strip()
        if "@sha256:" not in image_digest:
            raise DevEvalRunnerError("candidate image digest is not immutable")
        _local_dir, replay_store, snapshot_bundle = await self._ensure_prepared()
        manifest = replay_store.load_manifest()
        if not isinstance(manifest, Mapping):
            raise DevEvalRunnerError("snapshot-set manifest is required and missing")
        source_bundle = await source_bundle_for_artifact_v2(
            artifact,
            timeout_seconds=dev_eval_total_timeout_seconds(),
        )
        environment: dict[str, str] = {}
        credential_env_names: list[str] = []
        for name in private_model_env_passthrough():
            if name not in os.environ:
                continue
            if name in _PROVIDER_CREDENTIAL_ENV_NAMES:
                credential_env_names.append(name)
            else:
                environment[name] = str(os.environ[name])
        per_icp_timeout = _per_icp_timeout_seconds(
            len(load_verified_dev_items(replay_store))
        )
        execute = self._execute or execute_scoring_v2
        outcome = await execute(
            operation=OP_DEV_REPLAY_V2,
            purpose="research_lab.candidate_test.v2",
            epoch_id=self._epoch_id,
            sequence=max(0, int(candidate.iteration)),
            payload={
                "schema_version": DEV_REPLAY_REQUEST_SCHEMA_VERSION,
                "artifact": artifact.to_dict(),
                "source_bundle": source_bundle,
                "snapshot_bundle": snapshot_bundle,
                "snapshot_tree_hash": snapshot_bundle["source_tree_hash"],
                "snapshot_manifest_hash": str(manifest.get("manifest_hash") or ""),
                "module_name": "research_lab_adapter",
                "callable_name": "run_icp",
                "environment": environment,
                "credential_env_names": sorted(credential_env_names),
                "run_label": str(candidate.node_id or ""),
                "miss_policy": replay_store.miss_policy,
                "per_icp_timeout_seconds": per_icp_timeout,
                "total_timeout_seconds": dev_eval_total_timeout_seconds(),
            },
            worker_index=self._worker_index,
            input_artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_bundle["archive_sha256"]),
                str(snapshot_bundle["archive_sha256"]),
                str(snapshot_bundle["source_tree_hash"]),
                str(manifest.get("manifest_hash") or ""),
            ),
            timeout_seconds=float(dev_eval_total_timeout_seconds() + 120),
        )
        result = outcome.get("result")
        graph = outcome.get("receipt_graph")
        if not isinstance(result, Mapping) or not isinstance(graph, Mapping):
            raise DevEvalRunnerError("measured dev replay result is incomplete")
        if str(result.get("dev_set_hash") or "") != str(
            manifest.get("icp_set_hash") or ""
        ):
            raise DevEvalRunnerError("measured dev replay ICP commitment differs")
        if str(result.get("snapshot_manifest_hash") or "") != str(
            manifest.get("manifest_hash") or ""
        ):
            raise DevEvalRunnerError("measured dev replay snapshot commitment differs")
        root = next(
            (
                item
                for item in graph.get("receipts", ())
                if item.get("receipt_hash") == graph.get("root_receipt_hash")
            ),
            None,
        )
        if not isinstance(root, Mapping) or root.get("output_root") != sha256_json(
            dict(result)
        ):
            raise DevEvalRunnerError("measured dev replay output commitment differs")
        logger.info(
            "research_lab_loop_dev_eval_result node_id=%s lane=%s aggregate_dev_score=%s "
            "icp_count=%s scored_icp_count=%s snapshot_miss_count=%s failure_count=%s "
            "miss_policy=%s dev_score_version=%s",
            str(candidate.node_id or "")[:80],
            str(getattr(getattr(candidate, "draft", None), "lane", "") or "")[:80],
            round(float(result.get("aggregate_dev_score") or 0.0), 6),
            int(result.get("icp_count") or 0),
            int(result.get("scored_icp_count") or 0),
            int(result.get("snapshot_miss_count") or 0),
            int(result.get("failure_count") or 0),
            replay_store.miss_policy,
            str(result.get("dev_score_version") or ""),
        )
        return {"result": dict(result), "receipt_graph": dict(graph)}


def build_code_edit_dev_evaluator(
    *,
    snapshot_uri: str | None = None,
    cache_root: Path | None = None,
) -> DockerReplayDevEvaluator | None:
    """Worker factory for the engine's ``dev_evaluator`` seam.

    Returns None (leaving the seam unwired — the engine's existing safe
    default) unless ``RESEARCH_LAB_LOOP_DEV_EVAL_ENABLED`` is on AND
    ``RESEARCH_LAB_DEV_SNAPSHOT_URI`` is set. Construction never touches the
    network: the snapshot sync happens lazily on the first evaluation so a
    misconfigured URI degrades to unscored candidates, never a failed run.
    """
    if not dev_eval_runner_enabled():
        return None
    uri = str(
        snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
    ).strip()
    if not uri:
        logger.info(
            "research_lab_loop_dev_eval_unwired (%s is on but %s is unset)",
            DEV_EVAL_ENABLED_ENV,
            SNAPSHOT_URI_ENV,
        )
        return None
    return DockerReplayDevEvaluator(snapshot_uri=uri, cache_root=cache_root)


def build_attested_code_edit_dev_evaluator_v2(
    *,
    epoch_id: int,
    worker_index: int,
    snapshot_uri: str | None = None,
    cache_root: Path | None = None,
    execute: Any = None,
) -> AttestedReplayDevEvaluatorV2 | None:
    """V2 worker factory preserving the existing optional dev-eval gate."""
    if not dev_eval_runner_enabled():
        return None
    uri = str(
        snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
    ).strip()
    if not uri:
        logger.info(
            "research_lab_loop_dev_eval_unwired (%s is on but %s is unset)",
            DEV_EVAL_ENABLED_ENV,
            SNAPSHOT_URI_ENV,
        )
        return None
    return AttestedReplayDevEvaluatorV2(
        epoch_id=epoch_id,
        worker_index=worker_index,
        snapshot_uri=uri,
        cache_root=cache_root,
        execute=execute,
    )

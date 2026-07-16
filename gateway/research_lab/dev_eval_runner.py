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
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from gateway.research_lab.config import (
    DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI,
    DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG,
    MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT,
)
from leadpoet_canonical.attested_v2 import sha256_json
from research_lab.eval.dev_eval import (
    CURRENT_DAY_DEV_BANK_SCHEMA_VERSION,
    DevIcpSet,
    compute_dev_set_hash,
    evaluate_dev,
    select_snapshot_dev_icps,
)
from research_lab.eval.snapshot_store import (
    DEV_ICPS_NAME,
    MANIFEST_NAME,
    READY_NAME,
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
MAX_DEV_SNAPSHOT_BANK_ICP_COUNT = 100
CONTAINER_SNAPSHOT_DIR = "/research_lab_dev_snapshots"
_TRUTHY = ("1", "true", "yes", "on")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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
    """Mirror the engine's enabled-by-default development evaluation gate."""
    raw = str(os.getenv(DEV_EVAL_ENABLED_ENV) or "true").strip().lower()
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


def ensure_local_snapshot_set(
    root_uri: str,
    *,
    cache_root: Path | None = None,
    expected_dev_icp_count: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    ),
) -> Path:
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
        verification = ProviderSnapshotStore(
            str(local), mode=MODE_REPLAY
        ).verify_ready_document(
            expected_dev_icp_count=expected_dev_icp_count,
            require_signature=False,
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
    ready_verification = remote.verify_ready_document(
        expected_dev_icp_count=expected_dev_icp_count,
        require_signature=True,
    )
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
            verification = cached.verify_ready_document(
                expected_dev_icp_count=expected_dev_icp_count,
                require_signature=True,
            )
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
            verification = staged.verify_ready_document(
                expected_dev_icp_count=expected_dev_icp_count,
                require_signature=True,
            )
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
    expected_dev_icp_count: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    ),
    selection_seed: str = "snapshot-readiness",
    miner_direction: str = "",
    require_current_day: bool = False,
) -> dict[str, Any]:
    """Return fail-closed snapshot preflight evidence for activation policy."""
    if default_miss_policy() != MISS_POLICY_STRICT:
        return {"ready": False, "reason": "snapshot_miss_policy_must_be_strict"}
    try:
        resolution = resolve_snapshot_uri(root_uri)
        local = ensure_local_snapshot_set(
            root_uri,
            cache_root=cache_root,
            expected_dev_icp_count=expected_dev_icp_count,
        )
        store = ProviderSnapshotStore(str(local), mode=MODE_REPLAY, miss_policy=MISS_POLICY_STRICT)
        verification = store.verify_ready_document(
            expected_dev_icp_count=expected_dev_icp_count,
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
        items = load_verified_dev_items(
            store,
            expected_dev_icp_count=expected_dev_icp_count,
        )
        selection = select_verified_dev_items(
            items,
            manifest=manifest,
            expected_dev_icp_count=expected_dev_icp_count,
            selection_seed=selection_seed,
            miner_direction=miner_direction,
        )
        bank_manifest = (
            dict(manifest.get("dev_set_manifest") or {})
            if isinstance(manifest.get("dev_set_manifest"), Mapping)
            else {}
        )
        current_day_bank = (
            bank_manifest.get("schema_version")
            == CURRENT_DAY_DEV_BANK_SCHEMA_VERSION
        )
        benchmark_date = str(bank_manifest.get("benchmark_date") or "")
        current_utc_date = (now or datetime.now(timezone.utc)).astimezone(
            timezone.utc
        ).date().isoformat()
        if require_current_day and (
            not current_day_bank or benchmark_date != current_utc_date
        ):
            return {
                "ready": False,
                "reason": "snapshot_is_not_current_day_rebenchmark_bank",
                "benchmark_date": benchmark_date,
                "required_benchmark_date": current_utc_date,
            }
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
            "ready": len(selection.items) == expected_dev_icp_count,
            "reason": (
                "ready"
                if len(selection.items) == expected_dev_icp_count
                else "dev_selection_size_does_not_match_config"
            ),
            "manifest_hash": str(manifest.get("manifest_hash") or ""),
            "dev_set_hash": selection.dev_set_hash,
            "recorded_at": recorded_at,
            "snapshot_age_seconds": age,
            "dev_set_size": len(selection.items),
            "expected_dev_set_size": expected_dev_icp_count,
            "snapshot_bank_hash": str(manifest.get("icp_set_hash") or ""),
            "snapshot_bank_size": len(items),
            "daily_bank_hash": str(bank_manifest.get("daily_bank_hash") or ""),
            "selection_manifest_hash": str(
                selection.manifest.get("selection_manifest_hash") or ""
            ),
            "selection_seed_hash": str(
                selection.manifest.get("selection_seed_hash") or ""
            ),
            "miner_direction_hash": str(
                selection.manifest.get("miner_direction_hash") or ""
            ),
            "benchmark_date": benchmark_date,
            "benchmark_bundle_id": str(
                bank_manifest.get("benchmark_bundle_id") or ""
            ),
            "benchmark_bundle_hash": str(
                bank_manifest.get("benchmark_bundle_hash") or ""
            ),
            "rolling_window_hash": str(
                bank_manifest.get("rolling_window_hash") or ""
            ),
            "private_model_manifest_hash": str(
                bank_manifest.get("private_model_manifest_hash") or ""
            ),
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


def load_verified_dev_items(
    store: ProviderSnapshotStore,
    *,
    expected_dev_icp_count: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    ),
) -> list[dict[str, Any]]:
    """Load dev ICP payloads and bind them to the manifest's icp_set_hash."""
    items = store.load_dev_icp_items()
    if not items:
        raise DevEvalRunnerError("snapshot set carries no dev ICP payloads")
    if len(items) < expected_dev_icp_count:
        raise DevEvalRunnerError(
            "development snapshot bank is smaller than configured ICP count: "
            f"minimum={expected_dev_icp_count} actual={len(items)}"
        )
    if len(items) > MAX_DEV_SNAPSHOT_BANK_ICP_COUNT:
        raise DevEvalRunnerError("development snapshot bank exceeds the safety cap")
    manifest = store.load_manifest()
    expected = str((manifest or {}).get("icp_set_hash") or "")
    if expected and compute_dev_set_hash(items) != expected:
        raise DevEvalRunnerError(
            "dev_icps.json does not match the manifest icp_set_hash "
            "(tampered or mixed snapshot vintages)"
        )
    return items


def select_verified_dev_items(
    items: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    expected_dev_icp_count: int,
    selection_seed: str,
    miner_direction: str,
) -> DevIcpSet:
    """Resolve one exact run cohort from a verified immutable snapshot bank."""

    try:
        return select_snapshot_dev_icps(
            items,
            snapshot_manifest=manifest,
            size=int(expected_dev_icp_count),
            seed=str(selection_seed),
            miner_direction=str(miner_direction),
        )
    except Exception as exc:
        raise DevEvalRunnerError(str(exc)) from exc


class DockerReplayDevEvaluator:
    """Callable satisfying the engine's ``dev_evaluator`` seam contract."""

    def __init__(
        self,
        *,
        snapshot_uri: str | None = None,
        cache_root: Path | None = None,
        docker_executable: str = "docker",
        run_icp_in_docker: Callable[..., Sequence[Mapping[str, Any]]] | None = None,
        expected_dev_icp_count: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
        ),
        selection_seed: str = "",
        miner_direction: str = "",
    ) -> None:
        self._snapshot_uri = str(
            snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
        ).strip()
        self._cache_root = cache_root
        self._docker_executable = docker_executable
        # Test seam: replaces the docker invocation, everything else is real.
        self._run_icp_in_docker = run_icp_in_docker or self._run_icp_in_docker_default
        self._expected_dev_icp_count = int(expected_dev_icp_count)
        self._selection_seed = str(selection_seed)
        self._miner_direction = str(miner_direction)
        if not 1 <= self._expected_dev_icp_count <= MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT:
            raise DevEvalRunnerError("configured development ICP count is invalid")
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
                    ensure_local_snapshot_set,
                    self._snapshot_uri,
                    cache_root=self._cache_root,
                    expected_dev_icp_count=self._expected_dev_icp_count,
                )
                replay_store = ProviderSnapshotStore(
                    str(local_dir), mode=MODE_REPLAY, miss_policy=MISS_POLICY_STRICT
                )
                dev_items = await asyncio.to_thread(
                    load_verified_dev_items,
                    replay_store,
                    expected_dev_icp_count=self._expected_dev_icp_count,
                )
                manifest = replay_store.load_manifest()
                if not isinstance(manifest, Mapping):
                    raise DevEvalRunnerError("snapshot-set manifest is required")
                selected = await asyncio.to_thread(
                    select_verified_dev_items,
                    dev_items,
                    manifest=manifest,
                    expected_dev_icp_count=self._expected_dev_icp_count,
                    selection_seed=self._selection_seed,
                    miner_direction=self._miner_direction,
                )
                self._local_dir = local_dir
                self._replay_store = replay_store
                self._dev_items = list(selected.items)
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
                expected_icp_count=self._expected_dev_icp_count,
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
        provider_environment: Mapping[str, str] | None = None,
        model_env_passthrough: Sequence[str] | None = None,
        parent_graphs: Sequence[Mapping[str, Any]] = (),
        live_provider_call_cap: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_provider_calls
        ),
        live_cost_cap_microusd: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_cap_microusd
        ),
        live_max_icps_per_node: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
        ),
        live_timeout_seconds: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_timeout_seconds
        ),
        evaluation_concurrency: int = (
            DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.evaluation_concurrency
        ),
        prior_provider_call_count: int = 0,
        prior_settled_cost_microusd: int = 0,
        model_runner_factory: Any = None,
        selection_seed: str = "",
        miner_direction: str = "",
    ) -> None:
        self._epoch_id = max(0, int(epoch_id))
        self._worker_index = int(worker_index)
        self._snapshot_uri = str(
            snapshot_uri if snapshot_uri is not None else os.getenv(SNAPSHOT_URI_ENV) or ""
        ).strip()
        self._cache_root = cache_root
        self._execute = execute
        self._provider_environment = {
            str(name): str(value)
            for name, value in dict(provider_environment or {}).items()
        }
        self._model_env_passthrough = (
            tuple(str(item) for item in model_env_passthrough)
            if model_env_passthrough is not None
            else None
        )
        self._parent_graphs = tuple(dict(item) for item in parent_graphs)
        self._live_provider_call_cap = int(live_provider_call_cap)
        self._live_cost_cap_microusd = int(live_cost_cap_microusd)
        self._live_max_icps_per_node = int(live_max_icps_per_node)
        self._live_timeout_seconds = int(live_timeout_seconds)
        self._evaluation_concurrency = int(evaluation_concurrency)
        self._selection_seed = str(selection_seed)
        self._miner_direction = str(miner_direction)
        if not 1 <= self._live_provider_call_cap <= 32:
            raise DevEvalRunnerError("tree live provider-call cap is invalid")
        if not 1 <= self._live_cost_cap_microusd <= 500_000:
            raise DevEvalRunnerError("tree live provider-cost cap is invalid")
        if not 1 <= self._live_max_icps_per_node <= MAX_RESEARCH_LAB_GIT_TREE_ICP_COUNT:
            raise DevEvalRunnerError("tree live ICP cap is invalid")
        if self._live_timeout_seconds < 30:
            raise DevEvalRunnerError("tree live evaluation timeout is invalid")
        if not 1 <= self._evaluation_concurrency <= 16:
            raise DevEvalRunnerError("tree evaluation concurrency is invalid")
        self._model_runner_factory = model_runner_factory
        self._prepare_lock: asyncio.Lock | None = None
        self._discovery_lock: asyncio.Lock | None = None
        self._local_dir: Path | None = None
        self._replay_store: ProviderSnapshotStore | None = None
        self._snapshot_bundle: dict[str, Any] | None = None
        self._dev_items: list[dict[str, Any]] | None = None
        self._selection_manifest: dict[str, Any] | None = None
        self._tree_paid_call_count = max(0, int(prior_provider_call_count))
        self._tree_cost_microusd = max(
            0, int(prior_settled_cost_microusd)
        )
        if self._tree_paid_call_count > self._live_provider_call_cap:
            raise DevEvalRunnerError(
                "prior tree provider-call usage exceeds the configured cap"
            )
        if self._tree_cost_microusd > self._live_cost_cap_microusd:
            raise DevEvalRunnerError(
                "prior tree provider-cost usage exceeds the configured cap"
            )

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
                or self._dev_items is None
            ):
                from gateway.tee.source_bundle_v2 import build_source_bundle_v2

                local_dir = await asyncio.to_thread(
                    ensure_local_snapshot_set,
                    self._snapshot_uri,
                    cache_root=self._cache_root,
                    expected_dev_icp_count=self._live_max_icps_per_node,
                )
                replay_store = ProviderSnapshotStore(
                    str(local_dir),
                    mode=MODE_REPLAY,
                    miss_policy=MISS_POLICY_STRICT,
                )
                dev_items = await asyncio.to_thread(
                    load_verified_dev_items,
                    replay_store,
                    expected_dev_icp_count=self._live_max_icps_per_node,
                )
                verification = await asyncio.to_thread(replay_store.verify_manifest)
                if not verification.get("passed"):
                    raise DevEvalRunnerError(
                        "snapshot-set manifest failed verification: "
                        + "; ".join(verification.get("errors") or ())
                    )
                manifest = replay_store.load_manifest()
                if not isinstance(manifest, Mapping):
                    raise DevEvalRunnerError("snapshot-set manifest is required")
                selection = await asyncio.to_thread(
                    select_verified_dev_items,
                    dev_items,
                    manifest=manifest,
                    expected_dev_icp_count=self._live_max_icps_per_node,
                    selection_seed=self._selection_seed,
                    miner_direction=self._miner_direction,
                )
                snapshot_bundle = await asyncio.to_thread(
                    build_source_bundle_v2,
                    local_dir,
                )
                self._local_dir = local_dir
                self._replay_store = replay_store
                self._snapshot_bundle = snapshot_bundle
                self._dev_items = list(selection.items)
                self._selection_manifest = dict(selection.manifest)
            return (
                self._local_dir,
                self._replay_store,
                dict(self._snapshot_bundle),
            )

    def _candidate_identity(self, candidate: Any) -> dict[str, str]:
        from leadpoet_canonical.attested_v2 import sha256_json

        artifact = candidate.build.candidate_model_manifest
        image_digest = str(artifact.image_digest or "").strip()
        if "@sha256:" not in image_digest:
            raise DevEvalRunnerError("candidate image digest is not immutable")
        node_id = str(candidate.node_id or "").strip()
        if not node_id:
            raise DevEvalRunnerError("candidate node identity is missing")
        document = {
            "node_id": node_id,
            "iteration": str(max(0, int(candidate.iteration))),
            "model_artifact_hash": str(artifact.model_artifact_hash),
            "manifest_hash": str(artifact.manifest_hash),
            "image_digest": image_digest,
            "source_diff_hash": str(
                getattr(candidate.build, "source_diff_hash", "") or ""
            ),
        }
        return {**document, "candidate_hash": sha256_json(document)}

    def _measured_environment(self) -> tuple[dict[str, str], list[str]]:
        from research_lab.eval.private_runtime import private_model_env_passthrough

        environment: dict[str, str] = {}
        credential_env_names: list[str] = []
        passthrough = self._model_env_passthrough or tuple(
            private_model_env_passthrough()
        )
        for name in passthrough:
            if name not in os.environ:
                continue
            if name in _PROVIDER_CREDENTIAL_ENV_NAMES:
                credential_env_names.append(name)
            else:
                environment[name] = str(os.environ[name])
        return environment, sorted(set(credential_env_names))

    @staticmethod
    def _validate_outcome(
        *,
        outcome: Mapping[str, Any],
        manifest: Mapping[str, Any],
        evaluation_mode: str,
        overlay_hash: str,
        cohort_hash: str,
        expected_dev_set_hash: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from leadpoet_canonical.attested_v2 import sha256_json

        result = outcome.get("result")
        graph = outcome.get("receipt_graph")
        if not isinstance(result, Mapping) or not isinstance(graph, Mapping):
            raise DevEvalRunnerError("measured dev evaluation result is incomplete")
        if str(result.get("dev_set_hash") or "") != str(
            expected_dev_set_hash
        ):
            raise DevEvalRunnerError("measured dev evaluation ICP commitment differs")
        if str(result.get("snapshot_manifest_hash") or "") != str(
            manifest.get("manifest_hash") or ""
        ):
            raise DevEvalRunnerError(
                "measured dev evaluation snapshot commitment differs"
            )
        if str(result.get("evaluation_mode") or "") != evaluation_mode:
            raise DevEvalRunnerError(
                "measured dev evaluation mode commitment differs"
            )
        if str(result.get("overlay_hash") or "") != overlay_hash:
            raise DevEvalRunnerError(
                "measured dev evaluation overlay commitment differs"
            )
        if str(result.get("cohort_hash") or "") != cohort_hash:
            raise DevEvalRunnerError(
                "measured dev evaluation cohort commitment differs"
            )
        root = next(
            (
                item
                for item in graph.get("receipts", ())
                if isinstance(item, Mapping)
                and item.get("receipt_hash") == graph.get("root_receipt_hash")
            ),
            None,
        )
        if not isinstance(root, Mapping) or root.get("output_root") != sha256_json(
            dict(result)
        ):
            raise DevEvalRunnerError(
                "measured dev evaluation output commitment differs"
            )
        return dict(result), dict(graph)

    async def _evaluate_candidate(
        self,
        candidate: Any,
        *,
        evaluation_mode: str,
        cohort_hash: str,
        provider_evidence_caches: Mapping[str, Any] | None = None,
        overlay_hash: str = "",
        parent_graphs: Sequence[Mapping[str, Any]] = (),
    ) -> Mapping[str, Any]:
        from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
        from gateway.research_lab.model_authority_v2 import (
            source_bundle_for_artifact_v2,
        )
        from gateway.tee.scoring_executor_v2 import (
            DEV_HYBRID_REQUEST_SCHEMA_VERSION,
            DEV_REPLAY_REQUEST_SCHEMA_VERSION,
            OP_DEV_HYBRID_V2,
            OP_DEV_REPLAY_V2,
        )
        from leadpoet_canonical.attested_v2 import sha256_json

        artifact = candidate.build.candidate_model_manifest
        self._candidate_identity(candidate)
        if not _HASH_RE.fullmatch(cohort_hash):
            raise DevEvalRunnerError(
                "candidate evaluation cohort commitment is invalid"
            )
        _local_dir, replay_store, snapshot_bundle = await self._ensure_prepared()
        manifest = replay_store.load_manifest()
        if not isinstance(manifest, Mapping):
            raise DevEvalRunnerError("snapshot-set manifest is required and missing")
        source_bundle = await source_bundle_for_artifact_v2(
            artifact,
            timeout_seconds=dev_eval_total_timeout_seconds(),
        )
        environment, credential_env_names = self._measured_environment()
        dev_items = list(self._dev_items or ())
        selection_manifest = dict(self._selection_manifest or {})
        if not selection_manifest:
            raise DevEvalRunnerError("candidate evaluation selection is missing")
        per_icp_timeout = _per_icp_timeout_seconds(len(dev_items))
        execute = self._execute or execute_scoring_v2
        if evaluation_mode == "hybrid":
            caches = dict(provider_evidence_caches or {})
            if not caches or sha256_json(caches) != overlay_hash:
                raise DevEvalRunnerError("hybrid evidence overlay is incomplete")
            operation = OP_DEV_HYBRID_V2
            purpose = "research_lab.candidate_hybrid_test.v2"
            schema_version = DEV_HYBRID_REQUEST_SCHEMA_VERSION
        elif evaluation_mode == "replay":
            caches = {}
            operation = OP_DEV_REPLAY_V2
            purpose = "research_lab.candidate_test.v2"
            schema_version = DEV_REPLAY_REQUEST_SCHEMA_VERSION
        else:
            raise DevEvalRunnerError("candidate evaluation mode is invalid")
        payload = {
            "schema_version": schema_version,
            "artifact": artifact.to_dict(),
            "source_bundle": source_bundle,
            "snapshot_bundle": snapshot_bundle,
            "snapshot_tree_hash": snapshot_bundle["source_tree_hash"],
            "snapshot_manifest_hash": str(manifest.get("manifest_hash") or ""),
            "dev_selection_request": {
                "selection_seed": self._selection_seed,
                "miner_direction": self._miner_direction,
                "selection_manifest_hash": str(
                    selection_manifest.get("selection_manifest_hash") or ""
                ),
            },
            "module_name": "research_lab_adapter",
            "callable_name": "run_icp",
            "environment": environment,
            "credential_env_names": credential_env_names,
            "run_label": str(candidate.node_id or ""),
            "cohort_hash": cohort_hash,
            "miss_policy": replay_store.miss_policy,
            "per_icp_timeout_seconds": per_icp_timeout,
            "total_timeout_seconds": dev_eval_total_timeout_seconds(),
        }
        if evaluation_mode == "hybrid":
            payload.update(
                {
                    "provider_evidence_caches": caches,
                    "overlay_hash": overlay_hash,
                }
            )
        outcome = await execute(
            operation=operation,
            purpose=purpose,
            epoch_id=self._epoch_id,
            sequence=max(0, int(candidate.iteration)),
            payload=payload,
            worker_index=self._worker_index,
            parent_graphs=(
                tuple(dict(item) for item in parent_graphs)
            ),
            input_artifact_hashes=(
                artifact.model_artifact_hash,
                artifact.manifest_hash,
                str(source_bundle["archive_sha256"]),
                str(snapshot_bundle["archive_sha256"]),
                str(snapshot_bundle["source_tree_hash"]),
                str(manifest.get("manifest_hash") or ""),
                cohort_hash,
                *((overlay_hash,) if evaluation_mode == "hybrid" else ()),
                *(
                    tuple(sha256_json(dict(caches[key])) for key in sorted(caches))
                    if evaluation_mode == "hybrid"
                    else ()
                ),
            ),
            timeout_seconds=float(dev_eval_total_timeout_seconds() + 120),
        )
        result, graph = self._validate_outcome(
            outcome=outcome,
            manifest=manifest,
            evaluation_mode=evaluation_mode,
            overlay_hash=(overlay_hash if evaluation_mode == "hybrid" else sha256_json({})),
            cohort_hash=cohort_hash,
            expected_dev_set_hash=str(
                selection_manifest.get("dev_set_hash") or ""
            ),
        )
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
        return {"result": result, "receipt_graph": graph}

    async def _discover_provider_overlay(
        self,
        *,
        candidates: Sequence[Any],
        cohort_hash: str,
        remaining_tree_budget_microusd: int | None,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], int, int]:
        if self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()
        async with self._discovery_lock:
            return await self._discover_provider_overlay_locked(
                candidates=candidates,
                cohort_hash=cohort_hash,
                remaining_tree_budget_microusd=(
                    remaining_tree_budget_microusd
                ),
            )

    async def _discover_provider_overlay_locked(
        self,
        *,
        candidates: Sequence[Any],
        cohort_hash: str,
        remaining_tree_budget_microusd: int | None,
    ) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], int, int]:
        from gateway.research_lab.attested_scoring_v2 import execute_scoring_v2
        from gateway.research_lab.model_authority_v2 import (
            AttestedPrivateModelRunnerV2,
        )
        from leadpoet_canonical.attested_v2 import sha256_json
        from research_lab.eval import DockerPrivateModelSpec
        from research_lab.eval.private_runtime import (
            canonicalize_private_model_icp,
            private_model_env_passthrough,
        )
        from research_lab.eval.provider_evidence_cache import (
            EVIDENCE_CACHE_SCHEMA_VERSION,
            icp_evidence_cache_key,
        )

        _local_dir, replay_store, snapshot_bundle = await self._ensure_prepared()
        manifest = replay_store.load_manifest()
        if not isinstance(manifest, Mapping):
            raise DevEvalRunnerError("snapshot-set manifest is required and missing")
        dev_items = list(self._dev_items or ())
        if len(dev_items) != self._live_max_icps_per_node:
            raise DevEvalRunnerError(
                "hybrid discovery snapshot size differs from configured ICP count"
            )
        discovery_items = dev_items[: self._live_max_icps_per_node]
        runner_factory = self._model_runner_factory or AttestedPrivateModelRunnerV2
        execute = self._execute or execute_scoring_v2
        env_passthrough = self._model_env_passthrough or tuple(
            private_model_env_passthrough()
        )
        tree_ids = {
            str(getattr(candidate, "tree_id", "") or "")
            for candidate in candidates
        }
        if len(tree_ids) != 1 or not next(iter(tree_ids)):
            raise DevEvalRunnerError(
                "hybrid discovery candidates do not share one tree"
            )
        tree_id = next(iter(tree_ids))
        if (
            self._tree_paid_call_count >= self._live_provider_call_cap
            or self._tree_cost_microusd >= self._live_cost_cap_microusd
        ):
            raise DevEvalRunnerError("hybrid discovery tree budget is exhausted")
        scope = sha256_json(
            {
                "schema_version": "research_lab.git_tree_hybrid_scope.v1",
                "epoch_id": self._epoch_id,
                "tree_id": tree_id,
                "cohort_hash": cohort_hash,
                "snapshot_manifest_hash": str(
                    manifest.get("manifest_hash") or ""
                ),
            }
        )
        caches: dict[str, dict[str, Any]] = {}
        cache_graphs: dict[str, dict[str, Any]] = {}
        total_paid_calls = 0
        total_cost_microusd = 0
        remaining_provider_calls = (
            self._live_provider_call_cap - self._tree_paid_call_count
        )
        remaining_cost_microusd = (
            self._live_cost_cap_microusd - self._tree_cost_microusd
        )
        if remaining_tree_budget_microusd is not None:
            remaining_cost_microusd = min(
                remaining_cost_microusd,
                max(0, int(remaining_tree_budget_microusd)),
            )
        if remaining_provider_calls <= 0 or remaining_cost_microusd <= 0:
            raise DevEvalRunnerError("hybrid discovery tree budget is exhausted")
        for item_index, item in enumerate(discovery_items):
            raw_icp = item.get("icp") if isinstance(item, Mapping) else None
            canonical_icp = canonicalize_private_model_icp(
                dict(raw_icp or item)
            )
            cache_ref = icp_evidence_cache_key(canonical_icp)
            current_cache: dict[str, Any] = {}
            current_graph: dict[str, Any] | None = None
            current_hash = ""
            for candidate in candidates:
                artifact = candidate.build.candidate_model_manifest
                runner = runner_factory(
                    artifact=artifact,
                    spec=DockerPrivateModelSpec(
                        image_digest=artifact.image_digest,
                        env_passthrough=tuple(env_passthrough),
                        extra_env=self._provider_environment,
                        timeout_seconds=self._live_timeout_seconds,
                    ),
                    model_kind="candidate",
                    worker_index=self._worker_index,
                    epoch_id=self._epoch_id,
                    parent_graphs=self._parent_graphs,
                    execute=execute,
                )
                await runner.run_with_provider_evidence(
                    canonical_icp,
                    {
                        "mode": "candidate_hybrid_discovery",
                        "evaluation_epoch": self._epoch_id,
                        "dev_eval": True,
                        "run_label": str(candidate.node_id or ""),
                        "dev_item_number": item_index + 1,
                        "cohort_hash": cohort_hash,
                        "snapshot_manifest_hash": str(
                            manifest.get("manifest_hash") or ""
                        ),
                        "miss_policy": MISS_POLICY_STRICT,
                    },
                    provider_evidence_cache=current_cache,
                    provider_evidence_mode="record",
                    cache_parent_graphs=((current_graph,) if current_graph else ()),
                    provider_snapshot_bundle=snapshot_bundle,
                    provider_snapshot_tree_hash=str(
                        snapshot_bundle["source_tree_hash"]
                    ),
                    provider_snapshot_manifest_hash=str(
                        manifest.get("manifest_hash") or ""
                    ),
                    provider_cost_scope=scope,
                    provider_cost_cap_microusd=remaining_cost_microusd,
                    provider_call_cap=remaining_provider_calls,
                )
                generated = runner.generated_provider_evidence_cache(cache_ref)
                authorities = runner.attested_authorities()
                summary = runner.provider_evidence_summary(cache_ref)
                cost_summary = summary.get("cost_summary")
                if (
                    not isinstance(generated, Mapping)
                    or generated.get("schema_version")
                    != EVIDENCE_CACHE_SCHEMA_VERSION
                    or generated.get("icp_ref") != cache_ref
                    or not isinstance(generated.get("entries"), Mapping)
                    or not authorities
                    or not isinstance(cost_summary, Mapping)
                ):
                    raise DevEvalRunnerError(
                        "hybrid discovery returned incomplete measured evidence"
                    )
                generated_doc = dict(generated)
                generated_hash = sha256_json(generated_doc)
                if generated_hash != current_hash:
                    authority_graph = authorities[-1].get("receipt_graph")
                    if not isinstance(authority_graph, Mapping):
                        raise DevEvalRunnerError(
                            "hybrid discovery tape graph is missing"
                        )
                    current_cache = generated_doc
                    current_graph = dict(authority_graph)
                    current_hash = generated_hash
                total_paid_calls += max(
                    0, int(cost_summary.get("paid_call_count") or 0)
                )
                total_cost_microusd += max(
                    0,
                    int(
                        round(
                            float(cost_summary.get("total_cost_usd") or 0.0)
                            * 1_000_000
                        )
                    ),
                )
                if (
                    int(cost_summary.get("tracking_failed_count") or 0) > 0
                    or bool(cost_summary.get("cap_blocked"))
                    or bool(cost_summary.get("cap_exceeded_after_success"))
                ):
                    raise DevEvalRunnerError(
                        "hybrid discovery provider accounting did not settle"
                    )
            if not current_cache or current_graph is None:
                raise DevEvalRunnerError(
                    "hybrid discovery did not produce a frozen ICP overlay"
                )
            caches[cache_ref] = current_cache
            cache_graphs[cache_ref] = current_graph
        if (
            self._tree_paid_call_count + total_paid_calls
            > self._live_provider_call_cap
        ):
            raise DevEvalRunnerError("hybrid discovery provider-call cap exceeded")
        if (
            self._tree_cost_microusd + total_cost_microusd
            > self._live_cost_cap_microusd
        ):
            raise DevEvalRunnerError("hybrid discovery provider-cost cap exceeded")
        if total_cost_microusd > remaining_cost_microusd:
            raise DevEvalRunnerError(
                "hybrid discovery exceeded the remaining funded budget"
            )
        self._tree_paid_call_count += total_paid_calls
        self._tree_cost_microusd += total_cost_microusd
        return (
            caches,
            tuple(cache_graphs[key] for key in sorted(cache_graphs)),
            total_paid_calls,
            total_cost_microusd,
        )

    async def evaluate_cohort(
        self,
        candidates: Sequence[Any],
        *,
        remaining_tree_budget_microusd: int | None = None,
    ) -> Mapping[str, Any]:
        from gateway.research_lab.git_tree_evaluator import (
            TreeEvaluationPlan,
            classify_candidate_tree_evaluation,
        )
        from leadpoet_canonical.attested_v2 import sha256_json

        ordered = sorted(tuple(candidates), key=lambda item: str(item.node_id))
        if not ordered:
            raise DevEvalRunnerError("development-evaluation cohort is empty")
        identities = [self._candidate_identity(item) for item in ordered]
        if len({item["node_id"] for item in identities}) != len(identities):
            raise DevEvalRunnerError("development-evaluation cohort has duplicate nodes")
        plans = {
            str(candidate.node_id): classify_candidate_tree_evaluation(candidate)
            for candidate in ordered
        }
        identities_by_node = {
            item["node_id"]: item for item in identities
        }

        def cohort_commitment() -> str:
            return sha256_json(
                {
                    "schema_version": "research_lab.git_tree_eval_cohort.v1",
                    "epoch_id": self._epoch_id,
                    "candidates": identities,
                    "evaluation_plans": [
                        plans[item["node_id"]].to_dict()
                        for item in identities
                    ],
                }
            )

        semaphore = asyncio.Semaphore(self._evaluation_concurrency)

        async def evaluate_round(
            *,
            evaluation_mode: str,
            cohort_hash: str,
            caches: Mapping[str, Any],
            overlay_hash: str,
            parent_graphs: Sequence[Mapping[str, Any]],
        ) -> dict[str, dict[str, Any]]:
            async def evaluate_one(candidate: Any) -> tuple[str, dict[str, Any]]:
                async with semaphore:
                    envelope = await self._evaluate_candidate(
                        candidate,
                        evaluation_mode=evaluation_mode,
                        cohort_hash=cohort_hash,
                        provider_evidence_caches=caches,
                        overlay_hash=overlay_hash,
                        parent_graphs=parent_graphs,
                    )
                return str(candidate.node_id), dict(envelope)

            return dict(
                await asyncio.gather(
                    *(evaluate_one(item) for item in ordered)
                )
            )

        probe_cohort_hash = cohort_commitment()
        hybrid_candidates = [
            item for item in ordered if plans[str(item.node_id)].mode == "hybrid"
        ]
        if hybrid_candidates:
            (
                caches,
                cache_graphs,
                provider_call_count,
                settled_cost_microusd,
            ) = await self._discover_provider_overlay(
                candidates=hybrid_candidates,
                cohort_hash=probe_cohort_hash,
                remaining_tree_budget_microusd=(
                    remaining_tree_budget_microusd
                ),
            )
            overlay_hash = sha256_json(caches)
            mode = "hybrid"
            cohort_hash = probe_cohort_hash
            envelopes = await evaluate_round(
                evaluation_mode=mode,
                cohort_hash=cohort_hash,
                caches=caches,
                overlay_hash=overlay_hash,
                parent_graphs=cache_graphs,
            )
        else:
            caches = {}
            cache_graphs = ()
            provider_call_count = 0
            settled_cost_microusd = 0
            overlay_hash = sha256_json({})
            mode = "replay"
            envelopes = await evaluate_round(
                evaluation_mode=mode,
                cohort_hash=probe_cohort_hash,
                caches=caches,
                overlay_hash=overlay_hash,
                parent_graphs=(),
            )
            runtime_miss_candidates = [
                candidate
                for candidate in ordered
                if int(
                    dict(envelopes[str(candidate.node_id)]["result"]).get(
                        "snapshot_miss_count"
                    )
                    or 0
                )
                > 0
                or int(
                    dict(envelopes[str(candidate.node_id)]["result"]).get(
                        "true_miss_count"
                    )
                    or 0
                )
                > 0
            ]
            if runtime_miss_candidates:
                for candidate in runtime_miss_candidates:
                    node_id = str(candidate.node_id)
                    plan = plans[node_id]
                    plans[node_id] = TreeEvaluationPlan(
                        mode="hybrid",
                        reason_codes=tuple(
                            sorted(
                                set(plan.reason_codes)
                                | {"strict_replay_miss_requires_live_overlay"}
                            )
                        ),
                        changed_line_count=plan.changed_line_count,
                        target_file_count=plan.target_file_count,
                        patch_hash=plan.patch_hash,
                    )
                cohort_hash = cohort_commitment()
                (
                    caches,
                    cache_graphs,
                    provider_call_count,
                    settled_cost_microusd,
                ) = await self._discover_provider_overlay(
                    candidates=runtime_miss_candidates,
                    cohort_hash=cohort_hash,
                    remaining_tree_budget_microusd=(
                        remaining_tree_budget_microusd
                    ),
                )
                overlay_hash = sha256_json(caches)
                initial_graphs = tuple(
                    dict(envelopes[str(candidate.node_id)]["receipt_graph"])
                    for candidate in ordered
                )
                parents_by_root = {
                    str(graph.get("root_receipt_hash") or ""): dict(graph)
                    for graph in (*initial_graphs, *cache_graphs)
                }
                if "" in parents_by_root:
                    raise DevEvalRunnerError(
                        "runtime hybrid ancestry has an invalid receipt root"
                    )
                mode = "hybrid"
                envelopes = await evaluate_round(
                    evaluation_mode=mode,
                    cohort_hash=cohort_hash,
                    caches=caches,
                    overlay_hash=overlay_hash,
                    parent_graphs=tuple(
                        parents_by_root[key]
                        for key in sorted(parents_by_root)
                    ),
                )
            else:
                cohort_hash = probe_cohort_hash

        results = [
            {
                "node_id": str(candidate.node_id),
                "candidate_hash": identities_by_node[str(candidate.node_id)][
                    "candidate_hash"
                ],
                "result": dict(envelopes[str(candidate.node_id)]["result"]),
                "receipt_graph": dict(
                    envelopes[str(candidate.node_id)]["receipt_graph"]
                ),
                "evaluation_metadata": {
                    "evaluation_mode": mode,
                    "overlay_hash": overlay_hash,
                    "cohort_hash": cohort_hash,
                    "provider_call_count": provider_call_count,
                    "settled_cost_microusd": settled_cost_microusd,
                    "evaluation_plan": plans[str(candidate.node_id)].to_dict(),
                },
            }
            for candidate in ordered
        ]
        return {
            "schema_version": "research_lab.git_tree_eval_cohort_result.v1",
            "cohort_hash": cohort_hash,
            "evaluation_mode": mode,
            "overlay_hash": overlay_hash,
            "provider_call_count": provider_call_count,
            "settled_cost_microusd": settled_cost_microusd,
            "results": results,
        }

    async def __call__(self, candidate: Any) -> Mapping[str, Any]:
        # Kept only for the signed host-operation compatibility surface while
        # the tree engine uses evaluate_cohort(). A scalar call never performs
        # live discovery because it cannot establish round-wide fairness.
        from leadpoet_canonical.attested_v2 import sha256_json

        identity = self._candidate_identity(candidate)
        cohort_hash = sha256_json(
            {
                "schema_version": "research_lab.git_tree_scalar_eval.v1",
                "epoch_id": self._epoch_id,
                "candidate": identity,
                "evaluation_mode": "replay",
            }
        )
        return await self._evaluate_candidate(
            candidate,
            evaluation_mode="replay",
            cohort_hash=cohort_hash,
        )


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
        snapshot_uri
        if snapshot_uri is not None
        else os.getenv(SNAPSHOT_URI_ENV) or DEFAULT_RESEARCH_LAB_DEV_SNAPSHOT_URI
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
    provider_environment: Mapping[str, str] | None = None,
    model_env_passthrough: Sequence[str] | None = None,
    parent_graphs: Sequence[Mapping[str, Any]] = (),
    live_provider_call_cap: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_provider_calls
    ),
    live_cost_cap_microusd: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_cap_microusd
    ),
    live_max_icps_per_node: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_max_icps_per_node
    ),
    live_timeout_seconds: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.live_timeout_seconds
    ),
    evaluation_concurrency: int = (
        DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.evaluation_concurrency
    ),
    prior_provider_call_count: int = 0,
    prior_settled_cost_microusd: int = 0,
    model_runner_factory: Any = None,
    selection_seed: str = "",
    miner_direction: str = "",
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
        provider_environment=provider_environment,
        model_env_passthrough=model_env_passthrough,
        parent_graphs=parent_graphs,
        live_provider_call_cap=live_provider_call_cap,
        live_cost_cap_microusd=live_cost_cap_microusd,
        live_max_icps_per_node=live_max_icps_per_node,
        live_timeout_seconds=live_timeout_seconds,
        evaluation_concurrency=evaluation_concurrency,
        prior_provider_call_count=prior_provider_call_count,
        prior_settled_cost_microusd=prior_settled_cost_microusd,
        model_runner_factory=model_runner_factory,
        selection_seed=selection_seed,
        miner_direction=miner_direction,
    )

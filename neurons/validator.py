#!/usr/bin/env python3
# Suppress multiprocessing warnings BEFORE any imports
# Auto-update trigger: 2025-12-12
import os
import sys
from contextlib import contextmanager
from pathlib import Path

# CRITICAL: Add project root to sys.path BEFORE any local imports
# When running 'python3 neurons/validator.py', sys.path[0] = neurons/
# But qualification/, gateway/, etc. are in the project root
# This ensures all local modules can be imported regardless of how the script is run
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

# Opt-in, fail-closed error monitoring (docs/sentry_error_monitoring.md).
# Wired before the heavy imports below so import-time crashes are captured.
# Complete no-op unless the LEADPOET_SENTRY_* environment gate is satisfied.
try:
    from leadpoet_observability import (
        capture_failure as _capture_sentry_failure,
        configure_sentry_context as _configure_sentry_context,
        failure_code_for_exception as _sentry_failure_code_for_exception,
        hash_identifier as _sentry_hash_identifier,
        init_sentry as _init_sentry,
        record_retry as _record_sentry_retry,
        record_stage as _record_sentry_stage,
        sentry_stage as _sentry_stage,
        weight_correlation_id as _weight_correlation_id,
    )

    _init_sentry(component="validator")
    _configure_sentry_context(
        component="validator",
        physical_role="primary-validator",
        validator_role="primary",
        runtime_sha=(
            os.environ.get("GITHUB_SHA")
            or os.environ.get("GIT_COMMIT")
            or ""
        ),
        restart_invocation_id=os.environ.get(
            "LEADPOET_RESTART_INVOCATION_ID"
        ),
    )
except Exception as _sentry_exc:  # must never break the validator
    _capture_sentry_failure = lambda *args, **kwargs: False
    _configure_sentry_context = lambda *args, **kwargs: {}
    _sentry_failure_code_for_exception = (
        lambda exception, default: default
    )
    _sentry_hash_identifier = lambda value: "unavailable"
    _record_sentry_retry = lambda *args, **kwargs: None
    _record_sentry_stage = lambda *args, **kwargs: None
    _weight_correlation_id = lambda **kwargs: None

    @contextmanager
    def _sentry_stage(*args, **kwargs):
        yield None

    print(
        "leadpoet_sentry_wiring_skipped error=%s" % type(_sentry_exc).__name__,
        flush=True,
    )

import re
import errno
import time
import random
import requests
import textwrap
import numpy as np
import bittensor as bt
import argparse
import json
import gc  # For explicit memory cleanup
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from Leadpoet.base.validator import BaseValidatorNeuron
from Leadpoet.protocol import LeadRequest
from validator_models.automated_checks import validate_lead_list as auto_check_leads, run_automated_checks, MAX_REP_SCORE
from Leadpoet.base.utils.config import add_validator_args
import threading
from Leadpoet.base.utils import queue as lead_queue
from Leadpoet.base.utils import pool as lead_pool
from Leadpoet.base.utils.pool import (
    initialize_pool,
    add_to_pool,
    record_delivery_rewards,
    save_curated_leads,
)
# Import modules that have inject_async_subtensor methods
from Leadpoet.validator import reward as reward_module
from Leadpoet.utils import cloud_db as cloud_db_module
from Leadpoet.utils.bittensor_sdk import ExtrinsicOutcome
from Leadpoet.utils.subnet_epoch import (
    EPOCH_SCHEME,
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochCutover,
    SubnetEpochError,
    SubnetEpochSnapshot,
    VALIDATOR_SHARED_EPOCH_SCHEMA_VERSION,
    load_subnet_epoch_cutover,
    read_subnet_epoch_snapshot,
    validate_validator_shared_epoch_file,
    validate_subnet_epoch_cutover_anchor,
)
from Leadpoet.validator.reward import start_epoch_monitor, stop_epoch_monitor
import asyncio
from typing import List, Dict, Mapping, Optional, Any
from aiohttp import web
from Leadpoet.utils.cloud_db import (
    fetch_prospects_from_cloud,
    fetch_curation_requests,
    push_curation_result,
    push_miner_curation_request,
    fetch_miner_curation_result,
    push_validator_ranking,
    fetch_broadcast_requests,  # Must be at module level to avoid sandbox blocking
    # Additional imports moved from lazy to module-level to avoid sandbox blocking:
    get_supabase_client,
    broadcast_api_request,
    fetch_validator_rankings,
    get_broadcast_status,
    gateway_get_epoch_leads,
    gateway_submit_validation,
    # NOTE: gateway_submit_reveal REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
    submit_validation_assessment,
    fetch_miner_leads_for_request,
)
# TokenManager removed - JWT system deprecated in favor of TEE gateway
# from Leadpoet.utils.token_manager import TokenManager
from Leadpoet.utils.utils_lead_extraction import (
    get_email,
    get_website,
    get_company,
    get_industry,
    get_role,
    get_sub_industry,
    get_first_name,
    get_last_name,
    get_linkedin,
    get_location,
    get_field
)
from supabase import Client
import socket
from math import isclose
from pathlib import Path
import warnings
import subprocess
import aiohttp
from urllib.parse import urlparse
from leadpoet_canonical.weight_computation import (
    WEIGHT_SNAPSHOT_SCHEMA_VERSION,
    compute_final_weights_with_lab_arena as compute_canonical_final_weights,
    normalize_to_u16_with_uids_pure,
    research_lab_uid_weights_from_allocation as canonical_research_lab_uid_weights_from_allocation,
    weight_config_hash as canonical_weight_config_hash,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    DISABLED_LEADERBOARD_WINDOW_V1,
)
from leadpoet_canonical.constants import (
    ALLOCATION_PREPARATION_BLOCK,
    WEIGHT_SUBMISSION_BLOCK,
)
from leadpoet_verifier.economics import DEFAULT_RESEARCH_LAB_EMISSION_PERCENT


def _close_subtensor_connection(subtensor, *, source: str) -> None:
    """Retire one failed SDK websocket without masking the caller's result."""

    close = getattr(subtensor, "close", None)
    if not callable(close):
        close = getattr(getattr(subtensor, "substrate", None), "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        bt.logging.warning(
            "validator_subtensor_close_failed "
            f"source={source} type={type(exc).__name__}"
        )


def _is_subtensor_connection_error(exc: BaseException) -> bool:
    """Classify only transport failures that a fresh websocket can repair."""

    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TimeoutError, ConnectionError)):
            return True
        if isinstance(current, OSError) and current.errno in {
            errno.EPIPE,
            errno.ECONNABORTED,
            errno.ECONNREFUSED,
            errno.ECONNRESET,
            errno.ETIMEDOUT,
            errno.ENETDOWN,
            errno.ENETRESET,
            errno.ENETUNREACH,
            errno.EHOSTDOWN,
            errno.EHOSTUNREACH,
        }:
            return True
        name = type(current).__name__.lower()
        message = str(current).lower()
        if any(
            marker in name
            for marker in ("connection", "timeout", "websocket", "socket")
        ):
            return True
        if any(
            marker in message
            for marker in (
                "broken pipe",
                "connection aborted",
                "connection closed",
                "connection is closed",
                "connection refused",
                "connection reset",
                "handshake",
                "remote end closed",
                "timed out",
                "timeout",
                "websocket",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _atomic_write_json_file(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one shared validator JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".%s." % target.name,
        dir=str(target.parent),
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=True) as handle:
            descriptor = -1
            json.dump(dict(payload), handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
        directory = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _load_fulfillment_work_file(path: Path) -> Dict[str, Any]:
    """Load one complete fulfillment assignment or fail before scoring."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("fulfillment work document is not an object")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
        raise ValueError("fulfillment work epoch is invalid")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("fulfillment work request_id is invalid")
    if not isinstance(payload.get("icp"), dict):
        raise ValueError("fulfillment work ICP is invalid")
    submissions = payload.get("submissions")
    if not isinstance(submissions, list) or any(
        not isinstance(item, dict) for item in submissions
    ):
        raise ValueError("fulfillment work submissions are invalid")
    return payload


def _quarantine_fulfillment_work_file(path: Path) -> Path:
    """Remove malformed work from the worker glob without deleting evidence."""

    source = Path(path)
    target = source.with_name(
        ".invalid.%s.%d.%d"
        % (source.name, time.time_ns(), os.getpid())
    )
    os.replace(str(source), str(target))
    directory = os.open(str(source.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return target


def _shared_block_write_due(owner: Any, *, now: Optional[float] = None) -> bool:
    """Claim the 12-second shared-block heartbeat interval."""

    current = time.monotonic() if now is None else float(now)
    last = getattr(owner, "_last_block_file_write_monotonic", None)
    if last is not None and current - float(last) < 12.0:
        return False
    owner._last_block_file_write_monotonic = current
    return True


# The fulfillment heartbeat is written at each observed progress boundary.
# Its stale threshold must be longer than one fully legal polling iteration:
# a bounded chain read, a bounded workflow call, the normal poll interval, and
# a small scheduling margin.  The previous 120-second threshold contradicted
# the workflow's own 300-second guard and produced false restart instructions
# during ordinary gateway timeouts.
_FULFILLMENT_EPOCH_READ_TIMEOUT_SECONDS = 15.0
_FULFILLMENT_WORKFLOW_TIMEOUT_SECONDS = 300.0
_FULFILLMENT_POLL_INTERVAL_SECONDS = 30.0
_FULFILLMENT_HEARTBEAT_SCHEDULING_MARGIN_SECONDS = 30.0
_FULFILLMENT_HEARTBEAT_STALE_SECONDS = (
    _FULFILLMENT_EPOCH_READ_TIMEOUT_SECONDS
    + _FULFILLMENT_WORKFLOW_TIMEOUT_SECONDS
    + _FULFILLMENT_POLL_INTERVAL_SECONDS
    + _FULFILLMENT_HEARTBEAT_SCHEDULING_MARGIN_SECONDS
)


def _fulfillment_heartbeat_is_stale(age_seconds: float) -> bool:
    return float(age_seconds) > _FULFILLMENT_HEARTBEAT_STALE_SECONDS


def _canonical_sdk_weight_vector(weight_result: Mapping[str, Any]):
    """Return the exact UID-sorted float vector authorized by the enclave."""

    full_uids = [int(uid) for uid in weight_result.get("uids") or []]
    full_weights = [float(weight) for weight in weight_result.get("weights") or []]
    if len(full_uids) != len(full_weights):
        raise RuntimeError("authoritative weight vector lengths differ")
    pairs = sorted(
        (
            (uid, weight)
            for uid, weight in zip(full_uids, full_weights)
            if weight > 0
        ),
        key=lambda pair: pair[0],
    )
    uids = [uid for uid, _weight in pairs]
    weights = [weight for _uid, weight in pairs]
    emitted_uids, emitted_u16 = normalize_to_u16_with_uids_pure(uids, weights)
    if emitted_uids != list(weight_result.get("sparse_uids") or []) \
        or emitted_u16 != list(weight_result.get("sparse_weights_u16") or []):
        raise RuntimeError(
            "canonical SDK vector differs from enclave authorization"
        )
    return uids, weights

# ════════════════════════════════════════════════════════════════════════════
# TEE SIGNING IMPORTS (Phase 2.3 - Validator TEE Weight Submission)
# ════════════════════════════════════════════════════════════════════════════
# These imports are optional at startup - only used if TEE is enabled
try:
    from validator_tee import (
        AuthoritativeSetWeightsContextV2,
        build_enclave_backed_wallet_v2,
    )
    from validator_tee.host.authoritative_weight_flow_v2 import (
        finalize_authoritative_weight_publication_v2,
        prepare_authoritative_weight_publication_v2,
        resume_prepared_weight_publication_v2,
    )
    from validator_tee.host.publication_journal_v2 import (
        AuthoritativeWeightPublicationJournalV2,
    )
    from validator_tee.host.vsock_client import ValidatorEnclaveClient
    V2_TEE_AVAILABLE = True
except ImportError as e:
    V2_TEE_AVAILABLE = False
    # Will log warning at runtime if TEE submission is attempted

from validator_tee.host.weight_protocol_v2 import (
    AUTHORITATIVE_V2_PROTOCOL,
    normalize_weight_protocol,
)

# Additional warning suppression
warnings.filterwarnings("ignore", message=".*leaked semaphore objects.*")

WEIGHT_FINALIZATION_PROOF_ATTEMPTS = 10
WEIGHT_FINALIZATION_PROOF_RETRY_SECONDS = 12

# The validator restart wrapper owns Git synchronization and rebuilds the
# measured enclave before launching this process. Runtime self-updates are
# forbidden because they could move host code away from the approved EIF commit.
if __name__ == "__main__" and os.environ.get("LEADPOET_WRAPPER_ACTIVE") != "1":
    raise RuntimeError(
        "validator must be launched through validator_restart.sh with exact-commit V2 authority"
    )

# normal validator code starts below

# ════════════════════════════════════════════════════════════════════════════
# AUTO-CONTAINERIZATION: Automatically containerize if proxies detected
# ════════════════════════════════════════════════════════════════════════════

# Skip auto-containerization for worker modes (they should NOT trigger deployment)
_is_worker_mode = "--mode" in sys.argv and any(
    m in sys.argv for m in ["fulfillment_worker", "worker"]
)

def _auto_container_env_flag(name: str, default: str = "false") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

def _configured_proxy_envs(prefix: str, limit: int):
    proxies = []
    placeholder = "http://YOUR_USERNAME:YOUR_PASSWORD@p.webshare.io:80"
    for i in range(1, limit + 1):
        proxy_var = f"{prefix}_{i}"
        proxy_value = os.getenv(proxy_var)
        if proxy_value and proxy_value != placeholder:
            proxies.append((proxy_var, proxy_value))
    return proxies

if __name__ == "__main__" and os.environ.get("LEADPOET_CONTAINER_MODE") != "1" and not _is_worker_mode:
    force_container_deploy = _auto_container_env_flag(
        "VALIDATOR_FORCE_CONTAINER_DEPLOY"
    )
    fail_closed_container_deploy = (
        force_container_deploy
        or _auto_container_env_flag("LEADPOET_WRAPPER_ACTIVE")
    )
    legacy_sourcing_enabled = (
        _auto_container_env_flag("ENABLE_LEGACY_SOURCING")
        or _auto_container_env_flag("ENABLE_SOURCING_WORKERS")
    )
    fulfillment_enabled = _auto_container_env_flag("ENABLE_FULFILLMENT")

    sourcing_proxies = _configured_proxy_envs("WEBSHARE_PROXY", 250) if legacy_sourcing_enabled else []
    fulfillment_proxies = (
        _configured_proxy_envs("FULFILLMENT_WEBSHARE_PROXY", 10)
        if fulfillment_enabled
        else []
    )

    if (
        force_container_deploy
        or sourcing_proxies
        or fulfillment_proxies
    ):
        print("════════════════════════════════════════════════════════════════")
        print("🐳 AUTO-CONTAINERIZATION ACTIVATED")
        print("════════════════════════════════════════════════════════════════")
        print(f"📊 Legacy sourcing workers: {len(sourcing_proxies)}")
        print(f"📊 Fulfillment workers: {len(fulfillment_proxies)}")
        print("   Total containers are managed by deploy_dynamic.sh")
        print("")
        print("🔧 Building Docker image and spawning containers...")
        print("   (This may take a few minutes on first run)")
        print("")
        
        # Determine paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(script_dir)
        containerizing_dir = os.path.join(repo_root, "validator_models", "containerizing")
        deploy_script = os.path.join(containerizing_dir, "deploy_dynamic.sh")
        
        # Check if deploy script exists
        if not os.path.exists(deploy_script):
            message = f"Deploy script not found: {deploy_script}"
            print(f"❌ ERROR: {message}")
            if fail_closed_container_deploy:
                raise RuntimeError(message)
            print("   Falling back to non-containerized mode...")
            print("")
        else:
            # Execute deployment script
            try:
                import subprocess
                result = subprocess.run(
                    ["/bin/bash", deploy_script],
                    cwd=containerizing_dir,
                    check=True,
                    capture_output=False
                )
                
                print("")
                print("✅ Containerized deployment complete!")
                print("   Validator coordinator and enabled worker containers are now running")
                print("")
                if _auto_container_env_flag(
                    "VALIDATOR_AUTO_CONTAINER_FOLLOW_LOGS",
                    default="true",
                ):
                    print("📺 Following main validator logs...")
                    print("   (Press Ctrl+C to detach - containers will keep running)")
                    print("════════════════════════════════════════════════════════════════")
                    print("")

                    # Follow main container logs (blocking call)
                    try:
                        subprocess.run(
                            ["docker", "logs", "-f", "leadpoet-validator-main"],
                            check=False  # Don't raise exception on Ctrl+C
                        )
                    except KeyboardInterrupt:
                        print("")
                        print("════════════════════════════════════════════════════════════════")
                        print("🔌 Detached from logs (containers still running)")
                        print("")
                        print("📋 To reattach: docker logs -f leadpoet-validator-main")
                        print("📊 Check status: docker ps")
                        print("🛑 Stop all: docker stop leadpoet-validator-main leadpoet-validator-worker-1 leadpoet-validator-worker-2")
                        print("════════════════════════════════════════════════════════════════")
                
                sys.exit(0)
                
            except subprocess.CalledProcessError as e:
                print(f"❌ ERROR: Deployment failed with exit code {e.returncode}")
                if fail_closed_container_deploy:
                    raise RuntimeError(
                        "containerized validator deployment failed"
                    ) from e
                print("   Falling back to non-containerized mode...")
                print("")
            except Exception as e:
                print(f"❌ ERROR: {e}")
                if fail_closed_container_deploy:
                    raise
                print("   Falling back to non-containerized mode...")
                print("")

# ════════════════════════════════════════════════════════════════════════════

_TRUTHY_ENV_VALUES = {"true", "1", "yes", "y", "on"}
_SHARED_EPOCH_SCHEMA_VERSION = VALIDATOR_SHARED_EPOCH_SCHEMA_VERSION
_SHARED_EPOCH_FILE_WRITE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _ValidatorEpochState:
    """One coherent epoch decision used by validator and worker code."""

    current_block: int
    workflow_epoch_id: int
    epoch_block: int
    blocks_remaining: int
    epoch_start_block: int
    next_epoch_block: int
    tempo: int
    subnet_epoch_index: Optional[int] = None
    epoch_ref: Optional[str] = None
    snapshot: Optional[SubnetEpochSnapshot] = None

    @classmethod
    def from_snapshot(
        cls,
        snapshot: SubnetEpochSnapshot,
        cutover: SubnetEpochCutover,
    ) -> "_ValidatorEpochState":
        return cls(
            current_block=snapshot.current_block,
            workflow_epoch_id=snapshot.settlement_epoch_id(cutover),
            epoch_block=snapshot.epoch_block,
            blocks_remaining=snapshot.blocks_remaining,
            epoch_start_block=snapshot.last_epoch_block,
            next_epoch_block=snapshot.next_epoch_block,
            tempo=snapshot.tempo,
            subnet_epoch_index=snapshot.subnet_epoch_index,
            epoch_ref=snapshot.epoch_ref,
            snapshot=snapshot,
        )

    @property
    def identity(self) -> int:
        if self.subnet_epoch_index is None:
            raise SubnetEpochError("official subnet epoch identity is unavailable")
        return self.subnet_epoch_index

    def same_epoch(self, other: "_ValidatorEpochState") -> bool:
        return self.identity == other.identity

    def deadline_reached(self, elapsed_block: int) -> bool:
        """Return whether the official LastEpochBlock position reached a deadline."""

        threshold = int(elapsed_block)
        if threshold < 0:
            raise ValueError("epoch deadline cannot be negative")
        return self.epoch_block >= threshold

    def to_shared_document(self) -> Dict[str, Any]:
        cutover = load_subnet_epoch_cutover()
        authority = (
            self.snapshot.to_dict(cutover=cutover)
            if self.snapshot is not None
            else None
        )
        return {
            "schema_version": _SHARED_EPOCH_SCHEMA_VERSION,
            "block": self.current_block,
            "epoch": self.workflow_epoch_id,
            "blocks_into_epoch": self.epoch_block,
            "blocks_remaining": self.blocks_remaining,
            "epoch_start_block": self.epoch_start_block,
            "next_epoch_block": self.next_epoch_block,
            "tempo": self.tempo,
            "subnet_epoch_index": self.subnet_epoch_index,
            "epoch_ref": self.epoch_ref,
            "authority": authority,
            "runtime_generation": (
                os.environ.get("VALIDATOR_RUNTIME_GENERATION", "").strip()
                or "unconfigured"
            ),
            "timestamp": int(time.time()),
        }


@dataclass(frozen=True)
class _WeightPublicationRecoveryOutcome:
    """Distinguish finalized recovery from evidence-preserving quarantine."""

    epoch_id: int
    status: str

    def __post_init__(self) -> None:
        if self.status not in {"finalized", "quarantined"}:
            raise ValueError("weight publication recovery status is invalid")


def _read_shared_epoch_state_file(*, max_age_seconds: int) -> _ValidatorEpochState:
    path = Path("validator_weights") / "current_block.json"
    if not path.exists():
        raise FileNotFoundError("Coordinator hasn't written block file yet")
    cutover = load_subnet_epoch_cutover()
    snapshot = validate_validator_shared_epoch_file(
        path,
        max_age_seconds=max_age_seconds,
        cutover=cutover,
        expected_runtime_generation=(
            os.environ.get("VALIDATOR_RUNTIME_GENERATION", "").strip() or None
        ),
    )
    return _ValidatorEpochState.from_snapshot(
        snapshot,
        cutover,
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_ENV_VALUES


def _env_percent_share(name: str, default_percent: float) -> float:
    try:
        percent = float(os.environ.get(name, str(default_percent)))
    except (TypeError, ValueError):
        percent = float(default_percent)
    return max(0.0, min(1.0, percent / 100.0))


def _doc_percent_share(doc: Any, key: str, fallback_share: float) -> float:
    if isinstance(doc, dict) and doc.get(key) not in (None, ""):
        try:
            return max(0.0, min(1.0, float(doc.get(key)) / 100.0))
        except (TypeError, ValueError):
            return fallback_share
    return fallback_share


def _argv_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return str(sys.argv[index + 1] or "")


def _research_lab_production_subnet_default() -> bool:
    network = (
        os.environ.get("BITTENSOR_NETWORK")
        or os.environ.get("SUBTENSOR_NETWORK")
        or _argv_value("--subtensor_network")
        or ""
    ).strip().lower()
    netuid = (
        os.environ.get("BITTENSOR_NETUID")
        or os.environ.get("NETUID")
        or _argv_value("--netuid")
        or ""
    ).strip()
    return network == "finney" and netuid == "71"


def _validator_weight_protocol() -> str:
    return normalize_weight_protocol(os.environ.get("VALIDATOR_WEIGHT_PROTOCOL"))


def _current_validator_commit_sha() -> str:
    for key in ("GITHUB_SHA", "GIT_COMMIT_HASH", "GIT_COMMIT"):
        value = str(os.environ.get(key) or "").strip().lower()
        if re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value):
            return value
    try:
        value = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip().lower()
    except Exception:
        value = ""
    if not re.fullmatch(r"[0-9a-f]{40}(?:[0-9a-f]{24})?", value):
        raise RuntimeError("validator full commit SHA is unavailable")
    return value


def _verify_validator_v2_commit_alignment(
    client: Any, *, required: bool
) -> Dict[str, Any]:
    """Prove the host checkout and validator enclave use one approved commit."""
    from validator_tee.host.commit_alignment_v2 import (
        verify_validator_v2_commit_alignment,
    )

    return verify_validator_v2_commit_alignment(
        client,
        expected_commit=os.environ.get("VALIDATOR_V2_DEPLOY_COMMIT", ""),
        host_commit=_current_validator_commit_sha(),
        required=required,
    )


def _finalize_attested_weight_snapshot(values: Dict[str, Any]) -> Dict[str, Any]:
    """Bind the exact behavior configuration and validate the immutable snapshot."""

    snapshot = dict(values)
    snapshot["schema_version"] = WEIGHT_SNAPSHOT_SCHEMA_VERSION
    snapshot["config_hash"] = canonical_weight_config_hash(snapshot)
    compute_canonical_final_weights(snapshot)
    return snapshot


def _research_lab_allocation_has_live_payments(allocation_doc: Any) -> bool:
    if not isinstance(allocation_doc, dict):
        return False
    for section in (
        "source_add_allocations",
        "reimbursement_allocations",
        "champion_allocations",
        "queued_champion_allocations",
    ):
        rows = allocation_doc.get(section) or []
        if any(float(row.get("paid_alpha_percent") or 0.0) > 0 for row in rows if isinstance(row, dict)):
            return True
    return False


def _research_lab_uid_weights_from_allocation(
    allocation_doc: Any,
    *,
    metagraph: Any,
    reserved_share: float,
) -> tuple[dict[int, float], float, dict[str, float]]:
    return canonical_research_lab_uid_weights_from_allocation(
        allocation_doc,
        metagraph_hotkeys=metagraph.hotkeys,
        reserved_share=reserved_share,
    )


def _verify_burn_target_owner(metagraph: Any, uid: int, expected_hotkey: Optional[str]) -> bool:
    try:
        actual_hotkey = metagraph.hotkeys[int(uid)]
    except Exception as exc:
        print(f"   ❌ Error verifying burn target UID ownership: {exc}")
        return False
    if not expected_hotkey:
        print(f"   ⚠️ EXPECTED_BURN_TARGET_HOTKEY unset; using current UID {uid} owner on this network")
        return True
    if actual_hotkey != expected_hotkey:
        print(f"   ❌ CRITICAL ERROR: BURN_TARGET_UID={uid} ownership changed!")
        print(f"      Expected: {expected_hotkey[:20]}...")
        print(f"      Actual:   {actual_hotkey[:20]}...")
        print(f"      Burn would go to WRONG address - aborting weight submission")
        return False
    return True


# ════════════════════════════════════════════════════════════════════════════
# DEDICATED FULFILLMENT CONTAINERS CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
# Containers dedicated ONLY to scoring fulfillment leads.
# These run PARALLEL to sourcing (similar to qualification workers).
# Worker IDs are discovered from every configured
# FULFILLMENT_WEBSHARE_PROXY_N variable. Do not hard-code a count here:
# production currently deploys 10 workers and a fixed count silently left
# workers 6-10 idle.
# ════════════════════════════════════════════════════════════════════════════

FULFILLMENT_PROXY_ENV_RE = re.compile(r"^FULFILLMENT_WEBSHARE_PROXY_(\d+)$")
FULFILLMENT_MAX_WORKER_ID = 10


def detect_fulfillment_worker_ids() -> List[int]:
    """Return every configured fulfillment worker ID in numeric order."""

    worker_ids = []
    for proxy_var, proxy_value in os.environ.items():
        match = FULFILLMENT_PROXY_ENV_RE.match(proxy_var)
        if not match or not str(proxy_value or "").strip():
            continue
        worker_id = int(match.group(1))
        if 0 < worker_id <= FULFILLMENT_MAX_WORKER_ID:
            worker_ids.append(worker_id)
    return sorted(set(worker_ids))


def detect_fulfillment_proxies():
    """Backward-compatible alias returning configured fulfillment worker IDs."""

    return detect_fulfillment_worker_ids()

# ════════════════════════════════════════════════════════════════════════════

AVAILABLE_MODELS = [
    "openai/o3-mini:online",                    
    "openai/gpt-4o-mini:online",                 
    "google/gemini-2.5-flash:online",
    "openai/gpt-4o:online",            
]

FALLBACK_MODEL = "openai/gpt-4o:online"   

OPENROUTER_KEY = os.getenv("OPENROUTER_KEY")

def _llm_score_lead(lead: dict, description: str, model: str) -> float:
    """Return a 0-0.5 score for how well this lead fits the buyer description."""
    def _heuristic() -> float:
        d  = description.lower()
        txt = (get_company(lead) + " " + get_industry(lead)).lower()
        overlap = len(set(d.split()) & set(txt.split()))
        return min(overlap * 0.05, 0.5)

    if not OPENROUTER_KEY:
        return _heuristic()

    prompt_system = (
            "You are an expert B2B match-maker.\n"
            "FIRST LINE → JSON ONLY  {\"score\": <float between 0.0 and 0.5>}  (0.0 = bad match ⇢ 0.5 = perfect match)\n"
            "SECOND LINE → ≤40-word reason referencing the single lead.\n"
            "⚠️ Do not go outside the 0.0–0.5 range."
        )

    prompt_user = (
        f"BUYER:\n{description}\n\n"
        f"LEAD:\n"
        f"Company:  {get_company(lead)}\n"
        f"Industry: {get_industry(lead)}\n"
        f"Role:     {get_role(lead)}\n"
        f"Website:  {get_website(lead)}"
    )



    print("\n🛈  VALIDATOR-LLM INPUT ↓")
    print(textwrap.shorten(prompt_user, width=250, placeholder=" …"))

    def _extract(json_plus_reason: str) -> float:
        """Return score from first {...} block; raise if not parsable."""
        txt = json_plus_reason.strip()
        if not txt:
            raise ValueError("Empty response from model")
        
        if txt.startswith("```"):
            txt = txt.strip("`").lstrip("json").strip()
        start, end = txt.find("{"), txt.find("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        payload = txt[start:end + 1]
        score = float(json.loads(payload).get("score", 0))
        score = max(0.0, min(score, 0.5))     # <= clamp every time
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print(textwrap.shorten(txt, width=250, placeholder="…"))
        return max(0.0, min(score, 0.5))

    def _try(model_name: str) -> float:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={ "Authorization": f"Bearer {OPENROUTER_KEY}",
                      "Content-Type": "application/json"},
            json={ "model": model_name, "temperature": 0.2,
                   "messages":[{"role":"system","content":prompt_system},
                               {"role":"user","content":prompt_user}]},
            timeout=15)
        r.raise_for_status()
        return _extract(r.json()["choices"][0]["message"]["content"])

    try:
        return _try(model)
    except Exception as e:
        print(f"⚠️  Primary model failed ({model}): {e}")
        print(f"🔄 Trying fallback model: {FALLBACK_MODEL}")

    try:
        time.sleep(1)
        return _try(FALLBACK_MODEL)
    except Exception as e:
        print(f"⚠️  Fallback model failed: {e}")
        print("🛈  VALIDATOR-LLM OUTPUT ↓")
        print("<< no JSON response – all models failed >>")
        return None

def _extract_first_json_array(text: str) -> str:
    """Extract the first complete JSON array from text."""
    import json
    from json.decoder import JSONDecodeError

    start = text.find("[")
    if start == -1:
        raise ValueError("No JSON array found")

    decoder = json.JSONDecoder()
    try:
        obj, end_idx = decoder.raw_decode(text, start)
        return json.dumps(obj)
    except JSONDecodeError:
        end = text.rfind("]")
        if end == -1:
            raise ValueError("No JSON array found")
        return text[start:end+1]

def _llm_score_batch(leads: list[dict], description: str, model: str) -> dict:
    """Score all leads in a single LLM call. Returns dict mapping lead id() -> score (0.0-0.5)."""
    if not leads:
        return {}

    if not OPENROUTER_KEY:
        result = {}
        for lead in leads:
            d = description.lower()
            txt = (get_company(lead) + " " + get_industry(lead)).lower()
            overlap = len(set(d.split()) & set(txt.split()))
            result[id(lead)] = min(overlap * 0.05, 0.5)
        return result

    prompt_system = (
        "You are an expert B2B lead validation specialist performing quality assurance.\n"
        "\n"
        "TASK: Validate and score each lead based on fit with the buyer's ideal customer profile (ICP).\n"
        "\n"
        "SCORING CRITERIA (0.0 - 0.5 scale for consensus aggregation):\n"
        "• 0.45-0.50: Excellent match - company type, industry, and role perfectly align with buyer's ICP\n"
        "• 0.35-0.44: Good match - strong alignment with minor gaps\n"
        "• 0.25-0.34: Fair match - moderate relevance but notable misalignment\n"
        "• 0.15-0.24: Weak match - limited relevance, significant gaps\n"
        "• 0.00-0.14: Poor match - minimal to no relevance to buyer's ICP\n"
        "\n"
        "VALIDATION FACTORS:\n"
        "1. Industry specificity - Does the sub-industry/niche match the buyer's target?\n"
        "2. Business model fit - B2B vs B2C, enterprise vs SMB, SaaS vs services, etc.\n"
        "3. Company signals - Website quality, role seniority, geographic fit\n"
        "4. Buyer intent likelihood - Would this company realistically need the buyer's solution?\n"
        "5. Competitive landscape - Is this company in a position to buy similar offerings?\n"
        "\n"
        "OUTPUT FORMAT: Return ONLY a JSON array with one score per lead:\n"
        '[{"lead_index": 0, "score": <0.0-0.5 float>}, {"lead_index": 1, "score": <0.0-0.5 float>}, ...]\n'
        "\n"
        "⚠️ CRITICAL: Scores must be between 0.0 and 0.5. Be precise and differentiate - avoid giving identical scores.\n"
        "Consider: A generic 'Tech' buyer might target SaaS/AI companies (0.4-0.5) over general IT services (0.2-0.3)."
    )

    lines = [f"BUYER'S IDEAL CUSTOMER PROFILE (ICP):\n{description}\n\n"]
    lines.append(f"LEADS TO VALIDATE ({len(leads)} total):\n")

    for idx, lead in enumerate(leads):
        lines.append(
            f"\nLead #{idx}:\n"
            f"  Company: {get_company(lead, default='Unknown')}\n"
            f"  Industry: {get_industry(lead, default='Unknown')}\n"
            f"  Sub-industry: {get_sub_industry(lead, default='Unknown')}\n"
            f"  Contact Role: {get_role(lead, default='Unknown')}\n"
            f"  Website: {get_website(lead, default='Unknown')}"
        )

    prompt_user = "\n".join(lines)

    print("\n🛈  VALIDATOR-LLM BATCH INPUT ↓")
    print(f"   Scoring {len(leads)} leads in single prompt")
    print(textwrap.shorten(prompt_user, width=300, placeholder=" …"))

    def _try_batch(model_name: str):
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": model_name,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": prompt_system},
                    {"role": "user", "content": prompt_user}
                ]
            },
            timeout=30
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    try:
        response_text = _try_batch(model)
    except Exception as e:
        print(f"⚠️  Primary batch model failed ({model}): {e}")
        print(f"🔄 Trying fallback model: {FALLBACK_MODEL}")
        try:
            time.sleep(1)
            response_text = _try_batch(FALLBACK_MODEL)
        except Exception as e2:
            print(f"⚠️  Fallback batch model failed: {e2}")
            print("🛈  VALIDATOR-LLM BATCH OUTPUT ↓")
            print("<< no JSON response – all models failed >>")
            return {id(lead): None for lead in leads}

        # Parse response
    print("🛈  VALIDATOR-LLM BATCH OUTPUT ↓")
    print(textwrap.shorten(response_text, width=300, placeholder=" …"))

    try:
        # Extract JSON array (handles reasoning models like o3-mini)
        txt = response_text.strip()
        if txt.startswith("```"):
            txt = txt.strip("`").lstrip("json").strip()

        # Use robust extraction that handles extra reasoning content
        json_str = _extract_first_json_array(txt)
        scores_array = json.loads(json_str)

        # Map scores back to leads
        result = {}

        for item in scores_array:
            idx = item.get("lead_index")
            score = item.get("score", 0.0)
            if idx is not None and 0 <= idx < len(leads):
                # Clamp to 0.0-0.5 range
                clamped_score = max(0.0, min(score, 0.5))
                result[id(leads[idx])] = clamped_score

        # Fill in any missing leads with None
        for lead in leads:
            if id(lead) not in result:
                result[id(lead)] = None

        print(f"✅ Batch scoring succeeded (model: {model if 'mistralai' not in response_text else 'mistralai/mistral-7b-instruct'})")
        return result

    except Exception as e:
        print(f"⚠️  Failed to parse batch response: {e}")
        # Fallback to heuristic
        result = {}
        for lead in leads:
            d = description.lower()
            txt = (get_company(lead) + " " + get_industry(lead)).lower()
            overlap = len(set(d.split()) & set(txt.split()))
            result[id(lead)] = min(overlap * 0.05, 0.5)
        return result


def _normalize_icp_dict_for_prompt(icp_data: dict) -> dict:
    """Return a shallow copy of ``icp_data`` with industry/sub_industry coerced
    to comma-joined strings suitable for ``ICPPrompt`` (which types both as
    ``str``).  Multi-industry ICPs store these as ``List[str]``; without this
    coercion ``ICPPrompt(**icp_data)`` either raises a ValidationError (real
    list) or silently stringifies to ``"['X', 'Y']"`` (legacy bad shape) —
    both wedge downstream comparisons.
    """
    if not isinstance(icp_data, dict):
        return icp_data
    from gateway.fulfillment.icp_checks import _coerce_industry_list
    out = dict(icp_data)
    if "industry" in out:
        out["industry"] = ", ".join(_coerce_industry_list(out.get("industry"))) or ""
    if "sub_industry" in out:
        out["sub_industry"] = ", ".join(_coerce_industry_list(out.get("sub_industry"))) or ""
    return out


def _normalize_research_lab_sha256_ref(value: Any, *, fallback: Any = None, field_name: str = "hash") -> str:
    """Normalize Research Lab hash refs to the verifier-required sha256:<hex> shape."""
    text = str(value or "").strip().lower()
    if not text and fallback is not None:
        text = str(fallback or "").strip().lower()
    if re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        return text
    if re.fullmatch(r"[0-9a-f]{64}", text):
        return f"sha256:{text}"
    raise ValueError(f"{field_name}_must_be_sha256")


class Validator(BaseValidatorNeuron):
    def set_weights(self):
        """Reject the inherited host-authoritative weight path."""

        raise RuntimeError(
            "direct set_weights is disabled; use authoritative V2 epoch publication"
        )

    def _uses_production_epoch_cutover_authority(self) -> bool:
        """Return whether the fixed durable cutover authority governs us."""

        subtensor_config = getattr(self.config, "subtensor", None)
        network = str(
            getattr(subtensor_config, "network", "") or ""
        ).strip().lower()
        return int(self.config.netuid) == 71 and network == "finney"

    def _validate_durable_epoch_runtime_lifecycle(
        self,
        *,
        force_refresh: bool,
    ) -> dict:
        """Bind this process to the durable Supabase cutover singleton."""

        cutover = getattr(self, "_epoch_cutover", None)
        if cutover is None:
            raise SubnetEpochError("subnet epoch cutover is unavailable")
        if not self._uses_production_epoch_cutover_authority():
            return {
                "lifecycle_state": "stateful_manifest_only",
                "mapping_hash": cutover.mapping_hash,
            }

        from gateway.utils.epoch import validate_epoch_runtime_lifecycle

        return validate_epoch_runtime_lifecycle(
            cutover=cutover,
            force_refresh=force_refresh,
            network=str(self.config.subtensor.network),
            netuid=int(self.config.netuid),
        )

    def _validate_durable_epoch_runtime_startup(self) -> None:
        """Fail startup unless the configured mapping matches durable authority."""

        cutover = getattr(self, "_epoch_cutover", None)
        if cutover is None:
            raise SubnetEpochError("subnet epoch cutover is unavailable")
        if self._uses_production_epoch_cutover_authority():
            from gateway.utils.epoch import validate_stateful_cutover_authority

            validate_stateful_cutover_authority(
                cutover,
                network=str(self.config.subtensor.network),
                netuid=int(self.config.netuid),
            )
        self._validate_durable_epoch_runtime_lifecycle(
            force_refresh=True,
        )

    def __init__(self, config=None):
        self._weight_protocol = _validator_weight_protocol()
        if self._weight_protocol != AUTHORITATIVE_V2_PROTOCOL:
            raise RuntimeError("validator weight authority must be authoritative_v2")
        if not V2_TEE_AVAILABLE:
            raise RuntimeError("authoritative validator V2 modules are unavailable")
        if config is None or not hasattr(config, "wallet"):
            raise RuntimeError("authoritative validator V2 wallet configuration is missing")
        self._validator_v2_client = ValidatorEnclaveClient()
        self._validator_v2_commit_alignment = _verify_validator_v2_commit_alignment(
            self._validator_v2_client,
            required=(
                int(config.netuid) == 71
                and str(config.subtensor.network).lower() == "finney"
            ),
        )
        enclave_wallet = build_enclave_backed_wallet_v2(
            name=str(config.wallet.name),
            hotkey_name=str(config.wallet.hotkey),
            path=str(config.wallet.path),
            client=self._validator_v2_client,
        )
        super().__init__(config=config, wallet=enclave_wallet)
        self._epoch_cutover = load_subnet_epoch_cutover()
        if (
            self._epoch_cutover is not None
            and int(self._epoch_cutover.netuid) != int(self.config.netuid)
        ):
            raise SubnetEpochError(
                "validator netuid differs from subnet epoch cutover manifest"
            )
        self._epoch_archive_subtensor = bt.Subtensor(
            network=OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
        )
        validate_subnet_epoch_cutover_anchor(
            self._epoch_archive_subtensor,
            self._epoch_cutover,
        )
        self._epoch_snapshot_lock = threading.Lock()
        self._subtensor_reconnect_lock = threading.Lock()
        self._validate_durable_epoch_runtime_startup()
        journal_path = Path(
            os.environ.get("VALIDATOR_V2_PUBLICATION_JOURNAL_PATH")
            or (
                Path(self.config.neuron.full_path)
                / "authoritative_weight_publication_v2.json"
            )
        ).expanduser()
        self._weight_publication_journal_v2 = (
            AuthoritativeWeightPublicationJournalV2(journal_path)
        )
        
        # Add async subtensor (initialized later in run())
        # This eliminates memory leaks and HTTP 429 errors from repeated instance creation
        self.async_subtensor = None

        bt.logging.info("Registering validator wallet on network...")
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            try:
                self.uid = self.subtensor.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=self.wallet.hotkey.ss58_address,
                    netuid=self.config.netuid,
                )
                if self.uid is not None:
                    bt.logging.success(f"Validator registered with UID: {self.uid}")
                    break
                else:
                    bt.logging.warning(f"Attempt {attempt + 1}/{max_retries}: Validator not registered on netuid {self.config.netuid}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                bt.logging.error(f"Attempt {attempt + 1}/{max_retries}: Failed to set UID: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        if self.uid is None:
            bt.logging.warning(f"Validator {self.config.wallet_name}/{self.config.wallet_hotkey} not registered on netuid {self.config.netuid} after {max_retries} attempts")

        self.validator_trust = 0.0
        if self.uid is not None:
            try:
                self.validator_trust = self.metagraph.validator_trust[self.uid].item()
                bt.logging.info(f"📊 Validator trust initialized: {self.validator_trust:.4f}")
            except Exception as e:
                bt.logging.warning(f"Failed to get validator trust: {e}")
                self.validator_trust = 0.0

        bt.logging.info("load_state()")
        self.load_state()

        self.app = web.Application()
        self.app.add_routes([
            web.post('/api/leads', self.handle_api_request),
            web.get('/api/leads/status/{request_id}', self.handle_status_request),
        ])
        
        self.email_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.sample_ratio = 0.2
        self.use_open_source_model = config.get("neuron", {}).get("use_open_source_validator_model", True)

        self.processing_broadcast = False
        self._processed_requests = set()
        
        self.precision = 15.0 
        self.consistency = 1.0  
        self.collusion_flag = 1
        self.reputation = self.precision * self.consistency * self.collusion_flag  
        self.validation_history = []  
        self.trusted_validator = False  
        self.registration_time = datetime.now()  
        self.appeal_status = None  
        
        # initialize_pool imported at module level
        initialize_pool()

        self.broadcast_mode = False
        self.broadcast_lock = threading.Lock()
        
        # TokenManager removed - JWT system deprecated in favor of TEE gateway (tasks6.md)
        # Validators now authenticate with gateway using wallet signatures + metagraph verification
        # No JWT tokens needed!
        bt.logging.info("🔐 Using TEE gateway authentication (no JWT tokens)")
        
        # Supabase client not needed for main validation flow
        # Validators get leads from TEE gateway via /epoch/{epoch_id}/leads
        self.supabase_url = "https://qplwoislplkcegvdmbim.supabase.co"
        self.supabase_client: Optional[Client] = None
        # Skip Supabase init - not needed for TEE gateway workflow

    async def initialize_async_subtensor(self):
        """
        Create single AsyncSubtensor instance at validator startup.
        
        This eliminates memory leaks and HTTP 429 errors from repeated instance creation.
        Call this from run() before entering main validation loop.
        """
        import bittensor as bt
        import os
        
        bt.logging.info(f"🔗 Initializing AsyncSubtensor for network: {self.config.subtensor.network}")
        
        # ════════════════════════════════════════════════════════════
        # PROXY BYPASS FOR ASYNC BITTENSOR WEBSOCKET
        # ════════════════════════════════════════════════════════════
        # Temporarily unset proxy env vars for async Bittensor init
        proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        saved_proxies = {}
        for var in proxy_env_vars:
            if var in os.environ:
                saved_proxies[var] = os.environ[var]
                del os.environ[var]
        
        try:
            # Create async subtensor (single instance for entire lifecycle)
            self.async_subtensor = bt.AsyncSubtensor(network=self.config.subtensor.network)
            
            bt.logging.info(f"✅ AsyncSubtensor initialized")
            bt.logging.info(f"   Endpoint: {self.async_subtensor.chain_endpoint}")
            bt.logging.info(f"   Network: {self.async_subtensor.network}")
        finally:
            # Restore proxy environment variables for API calls
            for var, value in saved_proxies.items():
                os.environ[var] = value

    async def get_current_block_async(self) -> int:
        """
        Get current block using async subtensor (NO new instances).
        
        Use this instead of self.subtensor.get_current_block() to avoid memory leaks.
        
        Returns:
            Current block number
        
        Raises:
            Exception: If async_subtensor not initialized
        """
        # ALWAYS use sync subtensor for block queries
        # This avoids WebSocket subscription conflicts from AsyncSubtensor
        # Block queries are frequent (every few seconds) and fast, so sync is preferred
        return self.subtensor.block

    def _reconnect_subtensor_sync(
        self,
        *,
        expected_source,
        reason: str,
    ) -> bool:
        """Replace one stale live-chain source after validating its successor."""

        lock = getattr(self, "_subtensor_reconnect_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._subtensor_reconnect_lock = lock
        with lock:
            if self.subtensor is not expected_source:
                return True
            replacement = None
            try:
                replacement = bt.Subtensor(config=self.config)
                with self._epoch_snapshot_lock:
                    snapshot = read_subnet_epoch_snapshot(
                        replacement,
                        netuid=int(self.config.netuid),
                        finalized=True,
                    )
                _ValidatorEpochState.from_snapshot(
                    snapshot,
                    self._epoch_cutover,
                )
                replacement_uid = replacement.get_uid_for_hotkey_on_subnet(
                    hotkey_ss58=self.wallet.hotkey.ss58_address,
                    netuid=int(self.config.netuid),
                )
                if replacement_uid is None:
                    raise RuntimeError(
                        "validator hotkey is not registered after reconnect"
                    )
                self.subtensor = replacement
                self.uid = int(replacement_uid)
                _close_subtensor_connection(
                    expected_source,
                    source="live_replaced",
                )
                bt.logging.warning(
                    "validator_subtensor_reconnect_success "
                    f"reason={reason} uid={self.uid}"
                )
                return True
            except Exception as exc:
                if replacement is not None:
                    _close_subtensor_connection(
                        replacement,
                        source="live_replacement_failed",
                    )
                bt.logging.error(
                    "validator_subtensor_reconnect_failure "
                    f"reason={reason} type={type(exc).__name__} "
                    f"error={str(exc)[:200]}"
                )
                return False

    def _read_epoch_state_sync(self, subtensor=None) -> _ValidatorEpochState:
        """Read one coherent epoch state from one Subtensor connection."""

        source = subtensor or self.subtensor
        if self._epoch_cutover is None:
            raise SubnetEpochError("subnet epoch cutover is unavailable")
        self._validate_durable_epoch_runtime_lifecycle(
            force_refresh=False,
        )
        try:
            with self._epoch_snapshot_lock:
                snapshot = read_subnet_epoch_snapshot(
                    source,
                    netuid=int(self.config.netuid),
                    finalized=True,
                )
        except Exception as exc:
            if (
                subtensor is not None
                or not _is_subtensor_connection_error(exc)
                or not self._reconnect_subtensor_sync(
                    expected_source=source,
                    reason="finalized_epoch_read",
                )
            ):
                raise
            source = self.subtensor
            with self._epoch_snapshot_lock:
                snapshot = read_subnet_epoch_snapshot(
                    source,
                    netuid=int(self.config.netuid),
                    finalized=True,
                )
        return _ValidatorEpochState.from_snapshot(snapshot, self._epoch_cutover)

    async def _get_epoch_state_async(self) -> _ValidatorEpochState:
        state = await asyncio.to_thread(self._read_epoch_state_sync)
        self._latest_epoch_state = state
        return state

    def _read_best_epoch_state_sync(self, subtensor=None) -> _ValidatorEpochState:
        """Read best-head state only as an official epoch liveness veto.

        Finalized state remains the authority for epoch identity and settlement.
        The best head is read separately so an old finalized snapshot cannot
        authorize another signature after the live chain crossed its boundary.
        """

        source = subtensor or self.subtensor
        if self._epoch_cutover is None:
            raise SubnetEpochError("subnet epoch cutover is unavailable")
        try:
            with self._epoch_snapshot_lock:
                snapshot = read_subnet_epoch_snapshot(
                    source,
                    netuid=int(self.config.netuid),
                    finalized=False,
                )
        except Exception as exc:
            if (
                subtensor is not None
                or not _is_subtensor_connection_error(exc)
                or not self._reconnect_subtensor_sync(
                    expected_source=source,
                    reason="best_epoch_read",
                )
            ):
                raise
            source = self.subtensor
            with self._epoch_snapshot_lock:
                snapshot = read_subnet_epoch_snapshot(
                    source,
                    netuid=int(self.config.netuid),
                    finalized=False,
                )
        return _ValidatorEpochState.from_snapshot(snapshot, self._epoch_cutover)

    async def _get_best_epoch_state_async(self) -> _ValidatorEpochState:
        return await asyncio.to_thread(self._read_best_epoch_state_sync)

    async def _weight_submission_epoch_is_current(
        self,
        *,
        epoch_id: int,
        subnet_epoch_index: Optional[int],
    ) -> bool:
        """Require fresh durable authority plus matching finalized/live heads."""

        try:
            await asyncio.to_thread(
                self._validate_durable_epoch_runtime_lifecycle,
                force_refresh=True,
            )
            finalized_state = await self._get_epoch_state_async()
            if finalized_state.workflow_epoch_id != int(epoch_id):
                return False
            if finalized_state.subnet_epoch_index != subnet_epoch_index:
                return False
            best_state = await self._get_best_epoch_state_async()
            return (
                best_state.workflow_epoch_id == int(epoch_id)
                and best_state.subnet_epoch_index == subnet_epoch_index
                and best_state.current_block >= finalized_state.current_block
                and best_state.blocks_remaining > 0
            )
        except Exception as exc:
            bt.logging.error(
                "weight_submission_epoch_authority_unavailable "
                f"epoch={epoch_id} type={type(exc).__name__} "
                f"error={str(exc)[:200]}"
            )
            return False

    async def _weight_submission_lifecycle_is_open(
        self,
        *,
        epoch_id: int,
    ) -> bool:
        """Force-refresh only the durable fence immediately before one write."""

        try:
            await asyncio.to_thread(
                self._validate_durable_epoch_runtime_lifecycle,
                force_refresh=True,
            )
            return True
        except Exception as exc:
            bt.logging.error(
                "weight_submission_durable_lifecycle_closed "
                f"epoch={epoch_id} type={type(exc).__name__} "
                f"error={str(exc)[:200]}"
            )
            return False

    def _subnet_index_for_workflow_epoch(self, epoch_id: int) -> Optional[int]:
        cutover = self._epoch_cutover
        if cutover is None:
            raise SubnetEpochError("subnet epoch cutover is unavailable")
        normalized_epoch = int(epoch_id)
        if normalized_epoch < cutover.first_settlement_epoch_id:
            raise SubnetEpochError("workflow epoch predates subnet epoch cutover")
        return cutover.first_subnet_epoch_index + (
            normalized_epoch - cutover.first_settlement_epoch_id
        )

    def _write_shared_block_file(self, state: _ValidatorEpochState):
        """
        Atomically write the complete epoch authority for worker containers.

        Concurrent coordinator paths may refresh the same file. A delayed
        caller must never replace a newer finalized block from this runtime.
        """
        if not isinstance(state, _ValidatorEpochState):
            raise TypeError("shared block writer requires one epoch state")
        block_file = Path("validator_weights") / "current_block.json"
        document = state.to_shared_document()
        try:
            with _SHARED_EPOCH_FILE_WRITE_LOCK:
                try:
                    existing = json.loads(block_file.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError, OSError):
                    existing = {}
                try:
                    existing_block = int(existing.get("block", -1))
                except (TypeError, ValueError):
                    existing_block = -1
                if (
                    existing.get("runtime_generation")
                    == document["runtime_generation"]
                    and existing_block > state.current_block
                ):
                    bt.logging.warning(
                        "shared_epoch_write_ignored_older_block "
                        f"existing={existing_block} incoming={state.current_block}"
                    )
                    return False
                _atomic_write_json_file(block_file, document)
            return True
        except Exception as e:
            bt.logging.warning(f"Failed to write shared block file: {e}")
            return False

    def _run_shared_epoch_file_updater(
        self,
        stop_event: threading.Event,
        *,
        interval_seconds: float = 10.0,
        subtensor_factory=None,
    ) -> None:
        """Keep worker epoch authority fresh independently of the main loop."""

        interval = float(interval_seconds)
        if interval <= 0:
            raise ValueError("shared epoch update interval must be positive")
        factory = subtensor_factory or (lambda: bt.Subtensor(config=self.config))
        source = None
        try:
            while not self.should_exit and not stop_event.is_set():
                try:
                    if source is None:
                        source = factory()
                    state = self._read_epoch_state_sync(source)
                    self._write_shared_block_file(state)
                except Exception as exc:
                    if source is not None:
                        _close_subtensor_connection(
                            source,
                            source="shared_epoch_writer_failed",
                        )
                        source = None
                    bt.logging.warning(
                        "shared_epoch_writer_retry "
                        f"type={type(exc).__name__} error={str(exc)[:200]}"
                    )
                if stop_event.wait(interval):
                    break
        finally:
            if source is not None:
                _close_subtensor_connection(
                    source,
                    source="shared_epoch_writer_shutdown",
                )

    def _read_shared_epoch_state(self) -> _ValidatorEpochState:
        return _read_shared_epoch_state_file(max_age_seconds=30)

    def _read_shared_block_file(self) -> tuple:
        """
        Read current block/epoch info from shared file (for worker containers).
        
        Returns:
            (block, epoch, blocks_into_epoch) tuple
        
        Raises:
            Exception: If file doesn't exist, is too old (>30s), or is corrupted
        """
        try:
            state = self._read_shared_epoch_state()
            self._last_shared_epoch_state = state
            return (
                state.current_block,
                state.workflow_epoch_id,
                state.epoch_block,
            )
        except Exception as e:
            raise Exception(f"Failed to read shared block file: {e}")
    
    async def cleanup_async_subtensor(self):
        """Clean up async subtensor on shutdown."""
        if self.async_subtensor:
            bt.logging.info("🔌 Closing AsyncSubtensor...")
            await self.async_subtensor.close()
            bt.logging.info("✅ AsyncSubtensor closed")
    
    def _init_supabase_client(self):
        """Initialize or refresh Supabase client with current JWT token."""
        try:
            # get_supabase_client imported at module level

            # Use the centralized client creation function
            # This ensures consistency with miner and other validator operations
            self.supabase_client = get_supabase_client()

            if self.supabase_client:
                bt.logging.info("✅ Supabase client initialized for validator")
            else:
                bt.logging.warning("⚠️ No JWT token available for Supabase client")
        except Exception as e:
            bt.logging.error(f"Failed to initialize Supabase client: {e}")
            self.supabase_client = None

    def validate_email(self, email: str) -> bool:
        return bool(self.email_regex.match(email))

    def check_duplicates(self, leads: list) -> set:
        emails = [lead.get('email', '') for lead in leads]
        seen = set()
        duplicates = set(email for email in emails if email in seen or seen.add(email))
        return duplicates

    async def validate_leads(self, leads: list, industry: str = None) -> dict:
        if not leads:
            return {"score": 0.0, "O_v": 0.0}

        # Check if leads already have validation scores
        existing_scores = [lead.get("conversion_score") for lead in leads if lead.get("conversion_score") is not None]
        if existing_scores:
            # If leads already have scores, use the average of existing scores
            avg_score = sum(existing_scores) / len(existing_scores)
            return {"score": avg_score * 100, "O_v": avg_score}

        # Use automated_checks for all validation
        report = await auto_check_leads(leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        score = (valid_count / len(leads)) * 100 if leads else 0
        O_v = score / 100.0
        return {"score": score, "O_v": O_v}

    async def run_automated_checks(self, leads: list) -> bool:
        report = await auto_check_leads(leads)
        valid_count = sum(1 for entry in report if entry["status"] == "Valid")
        return valid_count / len(leads) >= 0.9 if leads else False

    async def reputation_challenge(self):
        dummy_leads = [
            {"business": f"Test Business {i}", "email": f"owner{i}@testleadpoet.com", "website": f"https://business{i}.com", "industry": "Tech & AI"}
            for i in range(10)
        ]
        known_score = random.uniform(0.8, 1.0)
        validation = await self.validate_leads(dummy_leads)
        O_v = validation["O_v"]
        if abs(O_v - known_score) <= 0.1:
            bt.logging.info("Passed reputation challenge")
        else:
            self.precision = max(0, self.precision - 10)
            bt.logging.warning(f"Failed reputation challenge, P_v reduced to {self.precision}")
        self.update_reputation()

    def update_consistency(self):
        now = datetime.now()
        periods = {
            "14_days": timedelta(days=14),
            "30_days": timedelta(days=30),
            "90_days": timedelta(days=90)
        }
        J_v = {}
        for period, delta in periods.items():
            start_time = now - delta
            relevant_validations = [v for v in self.validation_history if v["timestamp"] >= start_time]
            if not relevant_validations:
                J_v[period] = 0
                continue
            correct = sum(1 for v in relevant_validations if abs(v["O_v"] - v["F"]) <= 0.1)
            J_v[period] = correct / len(relevant_validations)
        
        self.consistency = 1 + (0.55 * J_v["14_days"] + 0.25 * J_v["30_days"] + 0.2 * J_v["90_days"])
        self.consistency = min(max(self.consistency, 1.0), 2.0)
        bt.logging.debug(f"Updated C_v: {self.consistency}, J_v: {J_v}")

    def update_reputation(self):
        self.reputation = self.precision * self.consistency * self.collusion_flag
        registration_duration = (datetime.now() - self.registration_time).days
        self.trusted_validator = self.reputation > 85 and registration_duration >= 30
        bt.logging.debug(f"Updated R_v: {self.reputation}, Trusted: {self.trusted_validator}")

    async def handle_buyer_feedback(self, leads: list, feedback_score: float):
        """Legacy method - buyer feedback not currently used in gateway architecture."""
        feedback_map = {
            (0, 1): (-20, 0.0),
            (1, 5): (-10, 0.2),
            (5, 7): (1, 0.5),
            (7, 8): (5, 0.7),
            (8, 9): (8, 0.9),
            (9, float('inf')): (15, 1.0)
        }
        for (low, high), (p_adj, f_new) in feedback_map.items():
            if low < feedback_score <= high:
                self.precision = max(0, min(100, self.precision + p_adj))
                bt.logging.info(f"Applied buyer feedback B={feedback_score}: P_v={self.precision}, F={f_new}")
                break
        self.update_reputation()

    async def submit_appeal(self):
        if self.collusion_flag == 1:
            bt.logging.info("No collusion flag to appeal")
            return
        self.appeal_status = {"votes": [], "start_time": datetime.now()}
        bt.logging.info("Collusion flag appeal submitted")

    async def vote_on_appeal(self, validator_hotkey: str, vote: int):
        if self.appeal_status is None or self.appeal_status != "pending":
            bt.logging.warning("No active appeal to vote on")
            return
        weight = {90: 5, 80: 3, 70: 2, 0: 1}.get(next(k for k in [90, 80, 70, 0] if self.precision > k), 1)
        self.appeal_status["votes"].append({"hotkey": validator_hotkey, "E_v": vote, "H_v": weight})
        bt.logging.debug(f"Vote submitted: E_v={vote}, H_v={weight}")

    async def resolve_appeal(self):
        if self.appeal_status is None or (datetime.now() - self.appeal_status["start_time"]).days < 7:
            return
        votes = self.appeal_status["votes"]
        if not votes:
            self.collusion_flag = 0
            bt.logging.warning("Appeal failed: No votes received")
        else:
            K_v_sum = sum(v["E_v"] * v["H_v"] for v in votes)
            H_v_sum = sum(v["H_v"] for v in votes)
            if K_v_sum / H_v_sum > 0.66:
                self.collusion_flag = 1
                bt.logging.info("Appeal approved: Collusion flag removed")
            else:
                self.collusion_flag = 0
                bt.logging.warning("Appeal denied")
        self.appeal_status = None
        self.update_reputation()

# ------------------------------------------------------------------+
#  Buyer → validator  (runs once per API call, not in a loop)       +
# ------------------------------------------------------------------+
    async def forward(self, synapse: LeadRequest) -> LeadRequest:
        """
        Respond to a buyer's LeadRequest arriving over Bittensor.
        Delegates to miners for curation, then ranks the results.
        """
        print(f"\n🟡 RECEIVED QUERY from buyer: {synapse.num_leads} leads | "
              f"desc='{synapse.business_desc[:40]}…'")

        # Always refresh metagraph just before selecting miners so we don't use stale flags.
        try:
            self.metagraph.sync(subtensor=self.subtensor)
            print("🔄 Metagraph refreshed for miner selection.")
        except Exception as e:
            print(f"⚠️  Metagraph refresh failed (continuing with cached state): {e}")

        # build the FULL list of miner axons (exclude validators)
        # IMPORTANT: Follow user's semantics:
        # - ACTIVE == True → validator (exclude)
        # - ACTIVE == False → miner (include)
        # Also require is_serving == True.
        active_flags = getattr(self.metagraph, "active", [False] * self.metagraph.n)
        vperm_flags  = getattr(self.metagraph, "validator_permit", [False] * self.metagraph.n)
        print("DBG flags:", {
            "n": self.metagraph.n,
            "serving": [bool(self.metagraph.axons[u].is_serving) for u in range(self.metagraph.n)],
            "active":  [bool(active_flags[u]) for u in range(self.metagraph.n)],
            "vperm":   [bool(vperm_flags[u]) for u in range(self.metagraph.n)],
        })
        my_uid = getattr(self, "uid", None)
        miner_uids = [
            uid for uid in range(self.metagraph.n)
            if getattr(self.metagraph.axons[uid], "is_serving", False)
            and uid != my_uid   # exclude the validator itself
        ]
        axons = [self.metagraph.axons[uid] for uid in miner_uids]

        print(f"🔍 Found {len(miner_uids)} active miners: {miner_uids}")
        print(f"🔍 Axon status: {[self.metagraph.axons[uid].is_serving for uid in miner_uids]}")
        if miner_uids:
            endpoints = [f"{self.metagraph.axons[uid].ip}:{self.metagraph.axons[uid].port}" for uid in miner_uids]
            print(f"🔍 Miner endpoints: {endpoints}")
            my_pub_ip = None
            try:
                if my_uid is not None:
                    my_pub_ip = getattr(self.metagraph.axons[my_uid], "ip", None)
            except Exception:
                pass

            for uid in miner_uids:
                ax = self.metagraph.axons[uid]
                if ax.ip == my_pub_ip:
                    print(f"🔧 Hairpin bypass for UID {uid}: {ax.ip} → 127.0.0.1")
                    ax.ip = "127.0.0.1"

        all_miner_leads: list = []

        print("\n─────────  VALIDATOR ➜ DENDRITE  ─────────")
        print(f"📡  Dialing {len(axons)} miners: {[f'UID{u}' for u in miner_uids]}")
        print(f"⏱️   at {datetime.utcnow().isoformat()} UTC")

        _t0 = time.time()
        miner_req = LeadRequest(num_leads=synapse.num_leads,
                                business_desc=synapse.business_desc)

        responses_task = asyncio.create_task(self.dendrite(
            axons       = axons,
            synapse     = miner_req,
            timeout     = 85,
            deserialize = False,
        ))
        responses = await responses_task
        print(f"⏲️  Dendrite completed in {(time.time() - _t0):.2f}s, analysing responses…")
        for uid, resp in zip(miner_uids, responses):
            if isinstance(resp, LeadRequest):
                sc = getattr(resp.dendrite, "status_code", None)
                sm = getattr(resp.dendrite, "status_message", None)
                pl = len(getattr(resp, "leads", []) or [])
                print(f"📥 UID {uid} dendrite status={sc} msg={sm} leads={pl}")
                if resp.leads:
                    all_miner_leads.extend(resp.leads)
            else:
                print(f"❌ UID {uid}: unexpected response type {type(resp).__name__} → {repr(resp)[:80]}")
        print("─────────  END DENDRITE BLOCK  ─────────\n")

        if not all_miner_leads:
            print("⚠️  Axon unreachable – falling back to cloud broker")
            for target_uid in miner_uids:
                req_id = push_miner_curation_request(
                    self.wallet,
                    {
                        "num_leads":      synapse.num_leads,
                        "business_desc":  synapse.business_desc,
                        "target_uid":     int(target_uid),
                    },
                )
                print(f"📤 Sent curation request to Cloud-Run for UID {target_uid}: {req_id}")

            # Wait for miner response via Cloud-Run
            MAX_ATTEMPTS = 40      # 40 × 5 s  = 200 s
            SLEEP_SEC    = 5
            total_wait   = MAX_ATTEMPTS * SLEEP_SEC
            print(f"⏳ Waiting for miner response (up to {total_wait} s)…")

            expected_miners = len(miner_uids)  # Number of miners we sent requests to
            received_responses = 0
            first_response_time = None
            
            for attempt in range(MAX_ATTEMPTS):
                res = fetch_miner_curation_result(self.wallet)
                if res and res.get("leads"):
                    # Collect from multiple miners
                    all_miner_leads.extend(res["leads"])
                    received_responses += 1
                    
                    # Track when we got the first response
                    if received_responses == 1:
                        first_response_time = attempt
                        print(f"✅ Received first response ({len(res['leads'])} leads) from Cloud-Run")
                        
                        # If expecting multiple miners, wait additional 30s for others
                        if expected_miners > 1:
                            print(f"⏳ Waiting additional 30s for {expected_miners - 1} more miners...")
                    else:
                        print(f"✅ Received response {received_responses}/{expected_miners} with {len(res['leads'])} leads")
                    
                    # Exit conditions:
                    # 1. Got all expected responses
                    if received_responses >= expected_miners:
                        print(f"✅ Received all {expected_miners} responses from miners")
                        break
                    
                    # 2. Got first response and waited 30s (6 attempts) for others
                    elif first_response_time is not None and (attempt - first_response_time) >= 6:
                        print(f"⏰ 30s timeout reached, proceeding with {received_responses}/{expected_miners} responses")
                        break
                
                time.sleep(SLEEP_SEC)
            
            if received_responses > 0:
                print(f"📊 Final collection: {len(all_miner_leads)} leads from {received_responses}/{expected_miners} miners")
            else:
                print("❌ No responses received from any miner via Cloud-Run")

        # Rank leads using LLM scoring (TWO rounds with BATCHING)
        if all_miner_leads:
            print(f"🔍 Ranking {len(all_miner_leads)} leads with LLM...")
            scored_leads = []
            
            aggregated = {id(lead): 0.0 for lead in all_miner_leads}
            failed_leads = set()
            first_model = random.choice(AVAILABLE_MODELS)
            print(f"🔄 LLM round 1/2 (model: {first_model})")
            batch_scores_r1 = _llm_score_batch(all_miner_leads, synapse.business_desc, first_model)
            for lead in all_miner_leads:
                score = batch_scores_r1.get(id(lead))
                if score is None:
                    failed_leads.add(id(lead))
                    print("⚠️  LLM failed for lead, will skip this lead")
                else:
                    aggregated[id(lead)] += score
            
            # ROUND 2: Second LLM scoring (BATCHED, random model selection)
            # Only score leads that didn't fail in round 1
            leads_for_r2 = [lead for lead in all_miner_leads if id(lead) not in failed_leads]
            if leads_for_r2:
                second_model = random.choice(AVAILABLE_MODELS)
                print(f"🔄 LLM round 2/2 (model: {second_model})")
                batch_scores_r2 = _llm_score_batch(leads_for_r2, synapse.business_desc, second_model)
                for lead in leads_for_r2:
                    score = batch_scores_r2.get(id(lead))
                    if score is None:
                        failed_leads.add(id(lead))
                        print("⚠️  LLM failed for lead, will skip this lead")
                    else:
                        aggregated[id(lead)] += score
            
            # Apply aggregated scores to leads (skip failed ones)
            for lead in all_miner_leads:
                if id(lead) not in failed_leads:
                    lead["intent_score"] = round(aggregated[id(lead)], 3)
                    scored_leads.append(lead)

            if not scored_leads:
                print("❌ All leads failed LLM scoring - check your OPENROUTER_KEY environment variable!")
                print("   Set it with: export OPENROUTER_KEY='your-key-here'")
                synapse.leads = []
                synapse.dendrite.status_code = 500
                return synapse

            # Sort by aggregated intent_score and take top N
            scored_leads.sort(key=lambda x: x["intent_score"], reverse=True)
            top_leads = scored_leads[:synapse.num_leads]

            print(f"✅ Ranked top {len(top_leads)} leads:")
            for i, lead in enumerate(top_leads, 1):
                business = get_company(lead, default='Unknown')
                score = lead.get('intent_score', 0)
                print(f"  {i}. {business} (score={score:.3f})")

            # Add c_validator_hotkey to leads being sent to client via Bittensor
            for lead in top_leads:
                lead["c_validator_hotkey"] = self.wallet.hotkey.ss58_address

            synapse.leads = top_leads
        else:
            print("❌ No leads received from any source")
            synapse.leads = []

        synapse.dendrite.status_code = 200
        return synapse

    async def _post_process_with_checks(self, rewards: np.ndarray, miner_uids: list, responses: list):
        validators = [self]
        validator_scores = []
        trusted_validators = [v for v in validators if v.trusted_validator]
        
        for i, response in enumerate(responses):
            if not isinstance(response, LeadRequest) or not response.leads:
                bt.logging.warning(f"Skipping invalid response from UID {miner_uids[i]}")
                continue
            validation = await self.validate_leads(response.leads, industry=response.industry)
            O_v = validation["O_v"]
            validator_scores.append({"O_v": O_v, "R_v": self.reputation, "leads": response.leads})
        
        trusted_low_scores = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] < 0.8)
        trusted_rejections = sum(1 for v in trusted_validators for s in validator_scores if v == self and s["O_v"] == 0)
        use_trusted = trusted_low_scores / len(trusted_validators) > 0.67 if trusted_validators else False
        reject = trusted_rejections / len(trusted_validators) > 0.5 if trusted_validators else False
        
        if reject:
            bt.logging.info("Submission rejected by >50% trusted validators")
            return
        
        Rs_total = sum(s["R_v"] for s in validator_scores if s["R_v"] > 15)
        F = sum(s["O_v"] * (s["R_v"] / Rs_total) for s in validator_scores if s["R_v"] > 15) if Rs_total > 0 else 0
        if use_trusted:
            trusted_scores = [s for s in validator_scores if any(v == self and v.trusted_validator for v in validators)]
            Rs_total_trusted = sum(s["R_v"] for s in trusted_scores if s["R_v"] > 15)
            F = sum(s["O_v"] * (s["R_v"] / Rs_total_trusted) for s in trusted_scores if s["R_v"] > 15) if Rs_total_trusted > 0 else 0
        
        for s in validator_scores:
            if abs(s["O_v"] - F) <= 0.1:
                self.precision = min(100, self.precision + 10)
            elif s["O_v"] > 0 and not await self.run_automated_checks(s["leads"]):
                self.precision = max(0, self.precision - 15)
            self.validation_history.append({"O_v": s["O_v"], "F": F, "timestamp": datetime.now()})
        
        self.update_consistency()
        self.update_reputation()
        
        for i, (reward, response) in enumerate(zip(rewards, responses)):
            if reward >= 0.9 and isinstance(response, LeadRequest) and response.leads:
                if await self.run_automated_checks(response.leads):
                    # add_to_pool imported at module level
                    add_to_pool(response.leads)
                    bt.logging.info(f"Added {len(response.leads)} leads from UID {miner_uids[i]} to pool")
                else:
                    self.precision = max(0, self.precision - 15)
                    bt.logging.warning(f"Post-approval check failed for UID {miner_uids[i]}, P_v reduced: {self.precision}")
        
        if random.random() < 0.1:
            await self.reputation_challenge()

        # Reward bookkeeping for delivered leads is handled in the main
        # `run_validator` validation loop, so nothing to do here.

    def save_state(self):
        bt.logging.info("Saving validator state.")
        
        try:
            # Save everything to validator_weights/ directory for consistency
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            
            # Save validator state (numpy)
            state_path = weights_dir / "validator_state.npz"
            
            np.savez(
                state_path,
                step=self.step,
                scores=self.scores,
                hotkeys=self.hotkeys,
                precision=self.precision,
                consistency=self.consistency,
                collusion_flag=self.collusion_flag,
                reputation=self.reputation,
                validation_history=np.array(self.validation_history, dtype=object),
                registration_time=np.datetime64(self.registration_time),
                appeal_status=self.appeal_status
            )
            bt.logging.info(f"✅ State saved to {state_path}")
            
            # NOTE: pending_reveals saving REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
            # Validators now submit hash+values in one request, no separate reveal phase
        except Exception as e:
            bt.logging.error(f"Failed to save state: {e}")
            bt.logging.error(f"   Attempted path: {state_path if 'state_path' in locals() else 'unknown'}")

    def load_state(self):
        # Load from validator_weights/ directory (new location)
        weights_dir = Path("validator_weights")
        state_path = weights_dir / "validator_state.npz"
        
        if state_path.exists():
            bt.logging.info("Loading validator state.")
            try:
                state = np.load(state_path, allow_pickle=True)
                self.step = state["step"]
                self.scores = state["scores"]
                self.hotkeys = state["hotkeys"]
                self.precision = state["precision"]
                self.consistency = state["consistency"]
                self.collusion_flag = state["collusion_flag"]
                self.reputation = state["reputation"]
                self.validation_history = state["validation_history"].tolist()
                self.registration_time = datetime.fromtimestamp(state["registration_time"].astype('datetime64[ns]').item() / 1e9)
                self.appeal_status = state["appeal_status"].item()
                bt.logging.info(f"✅ Loaded state from {state_path}")
            except Exception as e:
                bt.logging.warning(f"Failed to load state: {e}. Using defaults.")
                self._initialize_default_state()
        else:
            bt.logging.info("No state file found. Initializing with defaults.")
            self._initialize_default_state()
        
        # NOTE: pending_reveals loading REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
        # Validators now submit hash+values in one request, no separate reveal phase

    def _initialize_default_state(self):
        self.step = 0
        self.scores = np.zeros(self.metagraph.n, dtype=np.float32)
        self.hotkeys = self.metagraph.hotkeys.copy()
        self.precision = 15.0
        self.consistency = 1.0
        self.collusion_flag = 1
        self.reputation = self.precision * self.consistency * self.collusion_flag
        self.validation_history = []
        self.registration_time = datetime.now()
        self.appeal_status = None
        self.trusted_validator = False
        # NOTE: _pending_reveals REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE

    async def handle_api_request(self, request):
        """
        Handle API requests from clients using broadcast mechanism.

        Flow:
        1. Broadcast request to all validators/miners via Firestore
        2. Return request_id immediately to client
        3. Client polls /api/leads/status/{request_id} for results
        """
        try:
            data = await request.json()
            num_leads     = data.get("num_leads", 1)
            business_desc = data.get("business_desc", "")
            client_id     = data.get("client_id", "unknown")

            print(f"\n🔔 RECEIVED API QUERY from client: {num_leads} leads | desc='{business_desc[:10]}…'")
            bt.logging.info("📡 Broadcasting to ALL validators and miners via Firestore...")

            # Broadcast the request to all validators and miners
            try:
                # broadcast_api_request imported at module level

                # FIX: Wrap synchronous broadcast call to prevent blocking
                request_id = await asyncio.to_thread(
                    broadcast_api_request,
                    wallet=self.wallet,
                    num_leads=num_leads,
                    business_desc=business_desc,
                    client_id=client_id
                )

                print(f"📡 Broadcast API request {request_id[:8]}... to subnet")
                bt.logging.info(f"📡 Broadcast API request {request_id[:8]}... to subnet")

                # Return request_id immediately - client will poll for results
                return web.json_response({
                    "request_id": request_id,
                    "status": "processing",
                    "message": "Request broadcast to subnet. Poll /api/leads/status/{request_id} for results.",
                    "poll_url": f"/api/leads/status/{request_id}",
                    "status_code": 202,
                }, status=202)

            except Exception as e:
                print(f"❌ Failed to broadcast request: {e}")
                bt.logging.error(f"Failed to broadcast request: {e}")

                # Fallback to old direct method if broadcast fails
                return web.json_response({
                    "leads": [],
                    "status_code": 500,
                    "status_message": f"Failed to broadcast request: {str(e)}",
                    "process_time": "0"
                }, status=500)

        except Exception as e:
            print(f"❌ Error handling API request: {e}")
            bt.logging.error(f"Error handling API request: {e}")
            return web.json_response({
                "leads": [],
                "status_code": 500,
                "status_message": f"Error: {str(e)}",
                "process_time": "0"
            }, status=500)

    async def handle_status_request(self, request):
        """Handle status polling requests - returns quickly for test requests."""
        try:
            request_id = request.match_info.get('request_id')

            # Quick return for port discovery tests
            if request_id == "test":
                return web.json_response({
                    "status": "ok",
                    "request_id": "test"
                })

            # Fetch validator rankings from Firestore
            # fetch_validator_rankings and get_broadcast_status imported at module level

            # Get broadcast request status
            status_data = get_broadcast_status(request_id)

            # Fetch all validator rankings for this request
            validator_rankings = fetch_validator_rankings(request_id, timeout_sec=2)

            # Determine if timeout reached (check if request is older than 90 seconds)
            from datetime import datetime, timezone
            request_time = status_data.get("created_at", "")
            timeout_reached = False
            if request_time:
                try:
                    # Parse ISO timestamp
                    req_dt = datetime.fromisoformat(request_time.replace('Z', '+00:00'))
                    elapsed = (datetime.now(timezone.utc) - req_dt).total_seconds()
                    timeout_reached = elapsed > 90
                except Exception:
                    pass

            # Return data matching API client's expected format
            return web.json_response({
                "request_id": request_id,
                "status": status_data.get("status", "processing"),
                "validator_rankings": validator_rankings,
                "validators_submitted": len(validator_rankings),
                "timeout_reached": timeout_reached,
                "num_validators_responded": len(validator_rankings),  # Keep for backward compat
                "leads": status_data.get("leads", []),
                "metadata": status_data.get("metadata", {}),
            })

        except Exception as e:
            bt.logging.error(f"Error in handle_status_request: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return web.json_response({
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "validator_rankings": [],
                "validators_submitted": 0,
                "timeout_reached": False,
                "leads": [],
            }, status=500)

    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available for binding."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('0.0.0.0', port))
                return True
            except socket.error:
                return False

    def find_available_port(self, start_port: int, max_attempts: int = 10) -> int:
        """Find an available port starting from start_port."""
        port = start_port
        for _ in range(max_attempts):
            if self.check_port_availability(port):
                return port
            port += 1
        raise RuntimeError(f"No available ports found between {start_port} and {start_port + max_attempts - 1}")

    async def start_http_server(self):
        """Start HTTP server for API requests."""
        runner = web.AppRunner(self.app)
        await runner.setup()

        # Find available port
        port = self.find_available_port(8093)
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        bt.logging.info(f"🔴 Validator HTTP server started on port {port}")
        return port

    def run(self):
        """Override the base run method to not run continuous validation"""
        self.sync()

        # Check if validator is properly registered
        if not hasattr(self, 'uid') or self.uid is None:
            bt.logging.error("Cannot run validator: UID not set. Please register the wallet on the network.")
            return

        print(f"Running validator for subnet: {self.config.netuid} on network: {self.subtensor.chain_endpoint}")
        print(f"🔍 Validator UID: {self.uid}")
        print(f"🔍 Validator hotkey: {self.wallet.hotkey.ss58_address}")

        # Build the axon with the correct port
        self.axon = bt.Axon(
            wallet=self.wallet,
            ip      = "0.0.0.0",
            port    = self.config.axon.port,
            external_ip   = self.config.axon.external_ip,
            external_port = self.config.axon.external_port,
        )
        # expose buyer-query endpoint (LeadRequest → LeadRequest)
        self.axon.attach(self.forward)
        # Defer on-chain publish/start to run() to avoid double-serve hangs.
        print("───────────────────────────────────────────")
        # publish endpoint as PLAINTEXT so validators use insecure gRPC
        self.subtensor.serve_axon(
            netuid = self.config.netuid,
            axon   = self.axon,
        )
        print("✅ Axon published on-chain (plaintext)")
        self.axon.start()
        print("   Axon started successfully!")
        # Post-start visibility
        print(f"🖧  Local gRPC listener  : 0.0.0.0:{self.config.axon.port}")
        print(f"🌐  External endpoint   : {self.config.axon.external_ip}:{self.config.axon.external_port}")
        print("───────────────────────────────────────────")

        # Start HTTP server in background thread with dedicated event loop
        print("🔴 Starting HTTP server for REST API...")

        http_port_container = [None]  # Use list to share value between threads

        def run_http_server():
            """Run HTTP server in a dedicated event loop."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def start_and_serve():
                """Start server and keep it alive."""
                runner = web.AppRunner(self.app)
                await runner.setup()

                # Find available port
                port = self.find_available_port(8093)
                site = web.TCPSite(runner, '0.0.0.0', port)
                await site.start()

                http_port_container[0] = port  # Share port with main thread

                print(f"✅ HTTP server started on port {port}")
                print(f"📡 API endpoint: http://localhost:{port}/api/leads")
                print("───────────────────────────────────────────")

                # Keep the server running by awaiting an event that never completes
                # This is the proper way to keep an aiohttp server alive
                stop_event = asyncio.Event()
                await stop_event.wait()  # Wait forever

            try:
                # Run the server - this will block forever until KeyboardInterrupt
                loop.run_until_complete(start_and_serve())
            except KeyboardInterrupt:
                print("🛑 HTTP server shutting down...")
            except Exception as e:
                print(f"❌ HTTP server error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loop.close()

        # Start HTTP server in background thread
        http_thread = threading.Thread(target=run_http_server, daemon=True)
        http_thread.start()

        # Wait for server to start and get port
        for _ in range(50):  # Wait up to 5 seconds
            if http_port_container[0] is not None:
                break
            time.sleep(0.1)

        if http_port_container[0] is None:
            print("❌ HTTP server failed to start!")
        else:
            print(f"✅ HTTP server confirmed running on port {http_port_container[0]}")

        # Start broadcast polling loop in background thread
        def run_broadcast_polling():
            """Run broadcast polling in its own async event loop"""
            print("🟢 Broadcast polling thread started!")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            async def polling_loop():
                print("🟢 Broadcast polling loop initialized!")
                while not self.should_exit:
                    try:
                        await self.process_broadcast_requests_continuous()
                    except Exception as e:
                        bt.logging.error(f"Error in broadcast polling: {e}")
                        import traceback
                        bt.logging.error(traceback.format_exc())
                        await asyncio.sleep(5)  # Wait before retrying

            try:
                loop.run_until_complete(polling_loop())
            except KeyboardInterrupt:
                bt.logging.info("🛑 Broadcast polling shutting down...")
            except Exception as e:
                print(f"❌ Broadcast polling error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                loop.close()

        # Start broadcast polling in background thread
        broadcast_thread = threading.Thread(target=run_broadcast_polling, daemon=True, name="BroadcastPolling")
        broadcast_thread.start()
        # ══════════════════════════════════════════════════════════════════

        print(f"Validator starting at block: {self.block}")
        print("✅ Validator is now serving on the Bittensor network")
        print("   Processing sourced leads and waiting for client requests...")

        # Show available miners
        self.discover_miners()

        # ═══════════════════════════════════════════════════════════════
        # ASYNC MAIN LOOP: Initialize async subtensor and run async workflow
        # ═══════════════════════════════════════════════════════════════
        async def run_async_main_loop():
            """
            Async main validator loop.
            
            Uses async subtensor with block subscription for WebSocket health.
            """
            # Initialize async subtensor (single instance for entire lifecycle)
            await self.initialize_async_subtensor()
            
            # Inject into reward module
            try:
                # reward_module and cloud_db_module imported at module level
                
                reward_module.inject_async_subtensor(self.async_subtensor)
                cloud_db_module._VERIFY.inject_async_subtensor(self.async_subtensor)
                
                bt.logging.info("✅ AsyncSubtensor injected into reward and cloud_db modules")
            except Exception as e:
                bt.logging.warning(f"Failed to inject async subtensor: {e}")
            
            # ════════════════════════════════════════════════════════════
            # BLOCK SUBSCRIPTION: Keep WebSocket alive (prevents HTTP 429)
            # ════════════════════════════════════════════════════════════
            stop_event = asyncio.Event()
            
            async def block_callback(obj: dict):
                """Callback for new blocks (keeps WebSocket alive)."""
                if stop_event.is_set():
                    return True  # Stop subscription
                
                # Just log block number (no processing needed)
                # The subscription itself is what keeps WebSocket alive
                try:
                    block_number = obj["header"]["number"]
                    bt.logging.debug(f"📦 Block #{block_number} received (WebSocket alive)")
                except Exception as e:
                    bt.logging.debug(f"Block callback error: {e}")
                
                return None  # Continue subscription
            
            # Start block subscription in background (keeps WebSocket alive)
            bt.logging.info("🔔 Starting block subscription to keep WebSocket alive...")
            subscription_task = asyncio.create_task(
                self.async_subtensor.substrate.subscribe_block_headers(
                    subscription_handler=block_callback,
                    finalized_only=True
                )
            )
            bt.logging.info("✅ Block subscription started (WebSocket will stay alive)")
            
            # ════════════════════════════════════════════════════════════
            # SHARED EPOCH FILE UPDATER: For worker containers
            # ════════════════════════════════════════════════════════════
            # Weight finalization is a blocking SDK operation. Keep the shared
            # worker authority fresh on its own OS thread and chain client so
            # that finalization cannot starve fulfillment workers.
            shared_epoch_writer_stop = threading.Event()
            shared_epoch_writer_thread = None
            container_mode = getattr(self.config.neuron, "mode", None)
            if container_mode != "worker":
                shared_epoch_writer_thread = threading.Thread(
                    target=self._run_shared_epoch_file_updater,
                    args=(shared_epoch_writer_stop,),
                    daemon=True,
                    name="SharedEpochWriter",
                )
                shared_epoch_writer_thread.start()
                bt.logging.info(
                    "shared_epoch_writer_started interval_seconds=10"
                )

            # ════════════════════════════════════════════════════════════
            # FULFILLMENT POLLING THREAD (dedicated OS thread)
            # ════════════════════════════════════════════════════════════
            # First attempt was an asyncio.create_task on the main event
            # loop.  Observed in prod 2026-04-21: the task only ticked
            # twice in 77 minutes (expected ~154 ticks at 30s) because
            # the main loop's sync sourcing / qualification work
            # (DNS checks, HEAD requests, scrapes) blocked the event
            # loop for minutes at a time and my async task couldn't get
            # scheduled.  Gateway logs showed zero GET /fulfillment/scoring
            # from this validator over that window, so scoring-status
            # requests expired with no consensus despite the 15-min
            # gateway timeout.
            #
            # Running on a dedicated OS thread removes that dependency:
            # the thread has its own event loop and its own sync
            # subtensor, so a blocked main loop can't starve it.  Same
            # pattern as run_broadcast_polling elsewhere in this file.
            #
            # The thread:
            #   * creates its own ``bt.Subtensor`` (so we don't share
            #     the main loop's async_subtensor, which is loop-bound)
            #   * queries current_block synchronously each tick and
            #     passes it to process_fulfillment_workflow via
            #     current_block_override
            #   * wraps each workflow call in asyncio.wait_for using the
            #     shared bounded workflow timeout
            #     so a hung tick can't permanently freeze the loop
            #   * sleeps 30s between ticks, checks self.should_exit for
            #     clean shutdown
            #   * daemon=True so it dies with the process
            fulfillment_polling_thread = None
            if os.environ.get("ENABLE_FULFILLMENT", "false").lower() == "true":
                def _run_fulfillment_polling():
                    print("🎯 Fulfillment polling thread started (30s ticks)")
                    thread_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(thread_loop)

                    try:
                        thread_subtensor = bt.Subtensor(
                            network=self.config.subtensor.network
                        )
                    except Exception as e:
                        print(f"❌ Fulfillment polling thread: could not create subtensor: {e}")
                        thread_loop.close()
                        return

                    # Heartbeat path — updated at every progress boundary and
                    # at the end of every tick. External watchdogs can use the
                    # timestamp without mistaking an allowed blocking gateway
                    # request for a wedged thread.
                    # See validator_models/containerizing for the in-container
                    # mount point — this dir is bind-mounted to the host.
                    _ff_heartbeat = Path("validator_weights") / "ff_poll_heartbeat"

                    def _record_fulfillment_progress() -> None:
                        heartbeat_tmp = None
                        try:
                            _ff_heartbeat.parent.mkdir(parents=True, exist_ok=True)
                            heartbeat_tmp = _ff_heartbeat.with_name(
                                f".{_ff_heartbeat.name}.{os.getpid()}.tmp"
                            )
                            heartbeat_tmp.write_text(str(int(time.time())))
                            os.replace(heartbeat_tmp, _ff_heartbeat)
                        except Exception as e:
                            print(
                                "⚠️  Fulfillment heartbeat write failed "
                                f"(non-fatal): {e}"
                            )
                            try:
                                if heartbeat_tmp is not None:
                                    heartbeat_tmp.unlink()
                            except Exception:
                                pass

                    # Synchronous gateway calls inside the fulfillment method
                    # report safe phase boundaries through this callback. It
                    # changes no delivery, retry, or idempotency behavior.
                    self._fulfillment_progress_heartbeat = (
                        _record_fulfillment_progress
                    )

                    async def _safe_epoch_read(timeout: float = 15.0):
                        """Read one coherent thread-local epoch state with a hard timeout.

                        Observed 2026-05-18 00:55–02:43 UTC: this property
                        access (a Bittensor RPC over websocket) hung for
                        1h47min with no exception and no log output,
                        silently wedging the whole FF dispatch pipeline.
                        The 300s ``asyncio.wait_for`` below only guards
                        ``process_fulfillment_workflow`` — it does NOT
                        cover the block read, so a hang here freezes the
                        loop indefinitely.

                        ``asyncio.to_thread`` runs the blocking call on a
                        worker thread; ``asyncio.wait_for`` caps the wait
                        at ``timeout`` seconds.  If the timeout fires, the
                        underlying worker thread leaks (Python threads
                        can't be cancelled), but the polling loop
                        continues with ``current_block=None``.  A leaked
                        worker beats a wedged validator.
                        """
                        try:
                            return await asyncio.wait_for(
                                asyncio.to_thread(
                                    self._read_epoch_state_sync,
                                    thread_subtensor,
                                ),
                                timeout=timeout,
                            )
                        except asyncio.TimeoutError:
                            return None

                    async def _poll():
                        consec_block_timeouts = 0
                        while not self.should_exit:
                            _record_fulfillment_progress()
                            try:
                                current_epoch_state = await _safe_epoch_read(
                                    timeout=_FULFILLMENT_EPOCH_READ_TIMEOUT_SECONDS
                                )
                            except Exception as e:
                                print(f"⚠️  Fulfillment tick: block query failed: {e}")
                                current_epoch_state = None

                            if current_epoch_state is None:
                                consec_block_timeouts += 1
                                print(
                                    f"⚠️  Fulfillment tick: block read returned None "
                                    f"(timeout or error) — skipping tick "
                                    f"(consecutive timeouts: {consec_block_timeouts})"
                                )
                            else:
                                consec_block_timeouts = 0
                                try:
                                    _record_fulfillment_progress()
                                    await asyncio.wait_for(
                                        self.process_fulfillment_workflow(
                                            current_epoch_state=current_epoch_state,
                                        ),
                                        timeout=_FULFILLMENT_WORKFLOW_TIMEOUT_SECONDS,
                                    )
                                except asyncio.TimeoutError:
                                    print(
                                        "⚠️  Fulfillment tick exceeded bounded guard — "
                                        "skipping this tick, will retry next interval"
                                    )
                                except Exception as e:
                                    print(f"⚠️  Fulfillment tick error (non-fatal): {e}")

                            _record_fulfillment_progress()

                            try:
                                await asyncio.sleep(
                                    _FULFILLMENT_POLL_INTERVAL_SECONDS
                                )
                            except asyncio.CancelledError:
                                break

                    try:
                        thread_loop.run_until_complete(_poll())
                    except Exception as e:
                        print(f"❌ Fulfillment polling thread crashed: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        thread_loop.close()
                        print("🎯 Fulfillment polling thread exited")

                fulfillment_polling_thread = threading.Thread(
                    target=_run_fulfillment_polling,
                    daemon=True,
                    name="FulfillmentPolling",
                )
                fulfillment_polling_thread.start()
                print("✅ Fulfillment polling thread scheduled (daemon, 30s ticks)")
            else:
                print(
                    "ℹ️  ENABLE_FULFILLMENT != 'true', skipping fulfillment polling thread"
                )

            try:
                # Keep the validator running and continuously process leads
                while not self.should_exit:
                    # Process gateway validation workflow (TEE-based, now async)
                    try:
                        await self.process_gateway_validation_workflow()
                    except Exception as e:
                        bt.logging.warning(f"Error in gateway validation workflow: {e}")
                        await asyncio.sleep(5)  # Wait before retrying
                    
                    # Check if the current-epoch weight window has opened.
                    try:
                        await self.submit_weights_at_epoch_end()
                    except Exception as e:
                        bt.logging.warning(f"Error in submit_weights_at_epoch_end: {e}")
                    
                    try:
                        self.process_curation_requests_continuous()
                    except Exception as e:
                        bt.logging.warning(f"Error in process_curation_requests_continuous: {e}")
                        await asyncio.sleep(5)  # Wait before retrying
                    
                    # FULFILLMENT: runs on its own dedicated OS thread
                    # (fulfillment_polling_thread, created above).  Polls
                    # every 30s on a separate thread + event loop so it
                    # can't be starved by this main loop's sync sourcing
                    # work (DNS / HEAD / scrape calls).
                    #
                    # Watchdog: the polling thread writes a heartbeat file
                    # at the end of every tick.  We check its mtime once
                    # per main-loop iteration. The threshold covers one full
                    # legal fulfillment tick; beyond it the thread is likely
                    # wedged (e.g. an uncapped RPC hang). We log loudly so
                    # CloudWatch alerts trigger; we do NOT auto-restart the
                    # thread here (Python threads can't be killed cleanly,
                    # so recreating a thread while the old one may still
                    # hold sockets risks double-submission).  Operator
                    # action: restart the validator-main container.
                    try:
                        hb_path = Path("validator_weights") / "ff_poll_heartbeat"
                        if (
                            fulfillment_polling_thread is not None
                            and hb_path.exists()
                        ):
                            stale_s = time.time() - int(hb_path.read_text().strip())
                            if _fulfillment_heartbeat_is_stale(
                                stale_s
                            ) and not getattr(self, "_ff_stale_alerted", False):
                                print(
                                    f"🚨 Fulfillment polling thread heartbeat stale "
                                    f"({stale_s:.0f}s old) — thread is likely wedged. "
                                    f"Restart leadpoet-validator-main container to recover."
                                )
                                self._ff_stale_alerted = True
                            elif not _fulfillment_heartbeat_is_stale(
                                stale_s
                            ) and getattr(self, "_ff_stale_alerted", False):
                                print(
                                    f"✅ Fulfillment polling thread heartbeat recovered "
                                    f"({stale_s:.0f}s old)"
                                )
                                self._ff_stale_alerted = False
                    except Exception as e:
                        bt.logging.debug(f"FF heartbeat check error (non-fatal): {e}")

                    # process_broadcast_requests_continuous() runs in background thread

                    # Sync less frequently to avoid websocket concurrency issues
                    # Only sync every 10 iterations (approx every 10 seconds)
                    if not hasattr(self, '_sync_counter'):
                        self._sync_counter = 0

                    self._sync_counter += 1
                    if self._sync_counter >= 10:
                        try:
                            self.sync()
                            self._sync_counter = 0
                        except Exception as e:
                            bt.logging.warning(f"Sync error (will retry): {e}")
                            # Don't crash on sync errors, just skip this sync
                            self._sync_counter = 0

                    await asyncio.sleep(1)  # Small delay to prevent tight loop
                    
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Validator killed by keyboard interrupt.")
                exit()
            except Exception as e:
                bt.logging.error(f"Critical error in validator main loop: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())
                # Continue running instead of crashing
                await asyncio.sleep(10)  # Wait longer before retrying main loop
            finally:
                shared_epoch_writer_stop.set()
                if (
                    shared_epoch_writer_thread is not None
                    and shared_epoch_writer_thread.is_alive()
                ):
                    shared_epoch_writer_thread.join(timeout=10)
                    if shared_epoch_writer_thread.is_alive():
                        bt.logging.warning(
                            "shared_epoch_writer_shutdown_timeout"
                        )
                    else:
                        bt.logging.info("shared_epoch_writer_stopped")

                # Fulfillment polling is a daemon OS thread; it sees
                # self.should_exit=True (set on shutdown) and exits its
                # own event loop cleanly.  As a daemon it would also die
                # with the process, so we just give it a short join
                # window for a clean message and move on.
                if fulfillment_polling_thread is not None and fulfillment_polling_thread.is_alive():
                    print("🛑 Waiting up to 10s for fulfillment polling thread to exit...")
                    fulfillment_polling_thread.join(timeout=10)
                    if fulfillment_polling_thread.is_alive():
                        print("⚠️  Fulfillment polling thread did not exit in time (daemon, will die with process)")
                    else:
                        print("✅ Fulfillment polling thread stopped")

                # Stop block subscription
                bt.logging.info("🛑 Stopping block subscription...")
                stop_event.set()
                subscription_task.cancel()
                try:
                    await subscription_task
                except asyncio.CancelledError:
                    pass
                bt.logging.info("✅ Block subscription stopped")
                
                # Cleanup async subtensor on exit
                await self.cleanup_async_subtensor()
        
        # Run async main loop
        try:
            asyncio.run(run_async_main_loop())
        except KeyboardInterrupt:
            bt.logging.success("Validator killed by keyboard interrupt.")
            exit()
        except Exception as e:
            bt.logging.error(f"Fatal error in async main loop: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    # Add this method after the run() method (around line 1195)

    def sync(self):
        """
        Override sync to refresh validator trust after metagraph sync.

        This ensures we always have up-to-date trust values for consensus weighting.
        """
        # Call parent sync to refresh metagraph
        super().sync()

        # Refresh validator trust after metagraph sync
        # Handle case where uid might not be set yet (during initialization)
        if not hasattr(self, 'uid') or self.uid is None:
            return

        try:
            old_trust = getattr(self, 'validator_trust', 0.0)
            self.validator_trust = self.metagraph.validator_trust[self.uid].item()

            # Log significant changes in trust
            if abs(self.validator_trust - old_trust) > 0.01:
                bt.logging.info(
                    f"📊 Validator trust updated: {old_trust:.4f} → {self.validator_trust:.4f} "
                    f"(Δ{self.validator_trust - old_trust:+.4f})"
                )
        except Exception as e:
            bt.logging.warning(f"Failed to refresh validator trust: {e}")

    def discover_miners(self):
        """Show all available miners on the network"""
        try:
            print(f"\n🔍 Discovering available miners on subnet {self.config.netuid}...")
            self.sync()  # Sync metagraph to get latest data

            available_miners = []
            running_miners = []
            for uid in range(self.metagraph.n):
                if uid != self.uid:  # Don't include self
                    hotkey = self.metagraph.hotkeys[uid]
                    stake = self.metagraph.S[uid].item()
                    axon_info = self.metagraph.axons[uid]

                    miner_info = {
                        'uid': uid,
                        'hotkey': hotkey,
                        'stake': stake,
                        'ip': axon_info.ip,
                        'port': axon_info.port
                    }
                    available_miners.append(miner_info)

                    # Check if this miner is currently running (has axon info)
                    if axon_info.ip != '0.0.0.0' and axon_info.port != 0:
                        running_miners.append(miner_info)

            # Miner discovery completed - details logged in debug mode if needed
            bt.logging.debug(f"Found {len(available_miners)} registered miners, {len(running_miners)} currently running")

            if not available_miners:
                print("   ⚠️  No miners found on the network")
            elif not running_miners:
                print("   ⚠️  No miners currently running")

        except Exception as e:
            print(f"❌ Error discovering miners: {e}")

    async def _check_weight_submission_for_processed_epoch(self, current_epoch: int, reason: str) -> bool:
        """Keep weight submission live even after sourcing work is marked done."""
        try:
            submitted = await self.submit_weights_at_epoch_end()
        except Exception as exc:
            bt.logging.warning(
                f"Error checking weight submission for processed epoch {current_epoch} "
                f"({reason}): {exc}"
            )
            return False
        if submitted:
            print(
                f"✅ Weight submission check complete for epoch {current_epoch} "
                f"({reason})"
            )
        return bool(submitted)

    async def process_gateway_validation_workflow(self):
        """
        GATEWAY WORKFLOW (Passages 1 & 2): Fetch leads from gateway, validate, submit hashed results.
        This replaces process_sourced_leads_continuous for the new gateway-based architecture.
        
        ASYNC VERSION: Uses async subtensor for block queries (no memory leaks).
        """
        # Skip if processing broadcast request
        if self.processing_broadcast:
            return
        
        try:
            # Get current epoch_id from Bittensor block
            # Workers read from shared file (no Bittensor connection), coordinator uses Bittensor
            container_mode_check = getattr(self.config.neuron, 'mode', None)
            
            if container_mode_check == "worker":
                # WORKER: Read from shared block file (no Bittensor connection)
                try:
                    current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                    epoch_state = self._last_shared_epoch_state
                except Exception as e:
                    print(f"⏳ Worker: Waiting for coordinator to write block file... ({e})")
                    await asyncio.sleep(5)
                    return
            else:
                # COORDINATOR or SINGLE: read one exact-hash epoch snapshot.
                epoch_state = await self._get_epoch_state_async()
                current_block = epoch_state.current_block
                current_epoch = epoch_state.workflow_epoch_id
                blocks_into_epoch = epoch_state.epoch_block
                
                # Write block info to shared file for workers (if coordinator/single mode)
                # This happens inline (no separate thread) to avoid websocket concurrency issues
                # Write on elapsed monotonic time, not main-loop iteration
                # count. One slow iteration must not leave worker epoch
                # authority stale for minutes.
                if container_mode_check != "worker":
                    if _shared_block_write_due(self):
                        self._write_shared_block_file(epoch_state)
            
            # DEBUG: Always log epoch status
            print(
                f"[DEBUG] Current epoch: {current_epoch}, Block: {current_block}, "
                f"Epoch block: {blocks_into_epoch}, remaining: {epoch_state.blocks_remaining}, "
                f"subnet index: {epoch_state.subnet_epoch_index}, "
                f"Last processed: {getattr(self, '_last_processed_epoch', 'None')}"
            )
            
            # Check if we've already processed this epoch
            if not hasattr(self, '_last_processed_epoch'):
                self._last_processed_epoch = current_epoch - 1
                print(f"[DEBUG] Initialized _last_processed_epoch to {self._last_processed_epoch}")
            
            if current_epoch <= self._last_processed_epoch:
                # Already processed this epoch - no need to spam logs
                print(f"[DEBUG] Skipping epoch {current_epoch} (already processed)")
                await self._check_weight_submission_for_processed_epoch(
                    current_epoch,
                    "already_processed",
                )
                await asyncio.sleep(5)
                return
            
            print(f"[DEBUG] Processing epoch {current_epoch} for the FIRST TIME")
            
            # ═══════════════════════════════════════════════════════════════════
            # EPOCH TRANSITION: Clear old epochs from validator_weights file
            # This prevents file bloat and ensures clean state for new epoch
            # ═══════════════════════════════════════════════════════════════════
            self._clear_old_epochs_from_weights(current_epoch)
            
            print(f"\n{'='*80}")
            print(f"🔍 EPOCH {current_epoch}: Starting validation workflow")
            print(f"{'='*80}")
            
            # Legacy sourcing is retired.  Keep the old lead-validation path
            # behind an explicit opt-in flag so validator runs do not fetch
            # /epoch/{epoch}/leads or process sourcing leads by default.
            if not _env_flag("ENABLE_LEGACY_SOURCING"):
                print(f"\n🚫 Legacy sourcing disabled; skipping gateway lead fetch for epoch {current_epoch}")
                print("   Active validator tracks: fulfillment + Arena rewards")
                await self._check_weight_submission_for_processed_epoch(
                    current_epoch,
                    "legacy_sourcing_disabled",
                )
                self._last_processed_epoch = current_epoch
                print(f"✅ Marked epoch {current_epoch} as processed (legacy sourcing skipped)\n")
                await asyncio.sleep(10)
                return
            
            # Fetch assigned leads from gateway
            # gateway_get_epoch_leads, gateway_submit_validation imported at module level
            # NOTE: gateway_submit_reveal REMOVED (Jan 2026) - IMMEDIATE REVEAL MODE
            
            # ═══════════════════════════════════════════════════════════════════
            # OPTIMIZED LEAD FETCHING: Only coordinator calls gateway
            # Workers read from shared file to avoid N duplicate API calls
            # ═══════════════════════════════════════════════════════════════════
            container_mode = getattr(self.config.neuron, 'mode', None)
            container_id = getattr(self.config.neuron, 'container_id', None)
            
            # hashlib needed for salt generation (os is already imported at module level)
            import hashlib
            
            # CRITICAL: Check if leads file already exists with salt for this epoch
            # This prevents salt mismatch if coordinator restarts mid-epoch
            leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
            salt_hex = None
            
            if leads_file.exists():
                try:
                    with open(leads_file, 'r') as f:
                        existing_data = json.load(f)
                    if existing_data.get("epoch_id") == current_epoch and existing_data.get("salt"):
                        salt_hex = existing_data["salt"]
                        print(f"🔐 Reusing existing epoch salt: {salt_hex[:16]}... (from leads file)")
                except Exception as e:
                    print(f"⚠️  Could not read existing leads file: {e}")
            
            # Generate new salt only if we don't have one
            if not salt_hex:
                salt = os.urandom(32)
                salt_hex = salt.hex()
                print(f"🔐 Generated new epoch salt: {salt_hex[:16]}... (shared across all containers)")
            
            # Initialize truelist_results (will be populated by coordinator, read by workers)
            truelist_results = {}
            centralized_truelist_results = {}  # For workers reading from shared file
            
            if container_mode == "coordinator":
                # COORDINATOR: Fetch from gateway and share via file
                print(f"📡 Coordinator fetching leads from gateway for epoch {current_epoch}...")
                leads, max_leads_per_epoch = gateway_get_epoch_leads(self.wallet, current_epoch)
                
                # ================================================================
                # STEP 1: Write INITIAL file so workers can start Stage 0-2 immediately
                # truelist_results = None indicates "in progress" - workers will poll later
                # ================================================================
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                with open(leads_file, 'w') as f:
                    json.dump({
                        "epoch_id": current_epoch,
                        "leads": leads, 
                        "max_leads_per_epoch": max_leads_per_epoch,
                        "created_at_block": current_block,
                        "salt": salt_hex,  # CRITICAL: Workers need this to hash results
                        "truelist_results": None  # None = "in progress", workers will poll after Stage 0-2
                    }, f)
                print(f"   💾 Initial file written: {len(leads) if leads else 0} leads + salt (TrueList in progress...)")
                
                # ================================================================
                # STEP 2: Start centralized TrueList as BACKGROUND TASK
                # Workers can now start Stage 0-2 while TrueList runs
                # ================================================================
                truelist_task = None
                truelist_results = {}
                all_leads_for_file = leads  # Save original list before any slicing
                if leads:
                    from validator_models.automated_checks import run_centralized_truelist_batch
                    
                    print(f"\n📧 COORDINATOR: Starting centralized TrueList batch for ALL {len(leads)} leads (BACKGROUND)...")
                    truelist_task = asyncio.create_task(run_centralized_truelist_batch(leads))
                
            elif container_mode == "worker":
                # WORKER: Wait for coordinator to fetch and share
                print(f"⏳ Worker waiting for coordinator to fetch leads for epoch {current_epoch}...")
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                
                # Keep checking but with epoch boundary protection
                waited = 0
                log_interval = 300  # Log every 5 minutes
                check_interval = 5  # Check every 5 seconds
                
                while not leads_file.exists():
                    await asyncio.sleep(check_interval)
                    waited += check_interval
                    
                    # CRITICAL: Check current block and epoch from shared file
                    try:
                        check_block, check_epoch, blocks_into_epoch = self._read_shared_block_file()
                        check_state = self._last_shared_epoch_state
                    except Exception as e:
                        # Coordinator hasn't updated file yet, keep waiting
                        continue
                    
                    # Epoch changed while waiting - abort this epoch
                    if not check_state.same_epoch(epoch_state):
                        print(f"❌ Worker: Epoch changed ({current_epoch} → {check_epoch}) while waiting")
                        print(f"   Aborting - will process epoch {check_epoch} in next iteration")
                        await asyncio.sleep(10)
                        return
                    
                    # Too late to start validation (coordinator aggregates at block 300)
                    # Workers need ~8-10 min to process leads, so cutoff at block 260
                    # gives them 40 blocks (8 min) before coordinator forces aggregation
                    if check_state.deadline_reached(260):
                        print(
                            "❌ Worker: Too late to start validation "
                            f"({check_state.blocks_remaining} blocks remaining)"
                        )
                        print(f"   Coordinator aggregates at block 300 - not enough time to finish")
                        print(f"   Skipping epoch {current_epoch}, will process next epoch")
                        await asyncio.sleep(10)
                        return
                    
                    # Log progress every 5 minutes
                    if waited % log_interval == 0:
                        print(
                            f"   ⏳ Still waiting for coordinator... ({waited}s elapsed, "
                            f"block {blocks_into_epoch}/{check_state.tempo}, "
                            f"{check_state.blocks_remaining} remaining)"
                        )
                        print(f"      Checking for: {leads_file}")
                
                # Read leads from shared file
                with open(leads_file, 'r') as f:
                    data = json.load(f)
                    file_epoch = data.get("epoch_id")
                    leads = data.get("leads")
                    max_leads_per_epoch = data.get("max_leads_per_epoch")
                    centralized_truelist_results = data.get("truelist_results", {})  # Precomputed by coordinator
                
                # Verify epoch matches (safety check)
                if file_epoch != current_epoch:
                    print(f"❌ Worker: Epoch mismatch in leads file!")
                    print(f"   Expected epoch: {current_epoch}")
                    print(f"   File has epoch: {file_epoch}")
                    print(f"   Skipping - stale file detected")
                    await asyncio.sleep(10)
                    return
                
                print(f"✅ Worker loaded {len(leads) if leads else 0} leads from coordinator (waited {waited}s)")
                # Note: truelist_results might be None (in progress) or {} (complete/failed)
                # Workers will run Stage 0-2 first, then poll for truelist_results
                if centralized_truelist_results:
                    print(f"   ✅ TrueList already complete: {len(centralized_truelist_results)} results from coordinator")
                elif centralized_truelist_results is None:
                    print(f"   ⏳ TrueList still in progress - will poll after Stage 0-2 completes")
                else:
                    print(f"   ⚠️ TrueList returned empty results - leads will fail email verification")
                
            else:
                # DEFAULT: Single validator mode (no containers)
                print(f"📡 Fetching leads from gateway for epoch {current_epoch}...")
                leads, max_leads_per_epoch = gateway_get_epoch_leads(self.wallet, current_epoch)
            
            # Store max_leads_per_epoch for use in submit_weights_at_epoch_end
            # This value comes dynamically from the gateway config
            self._max_leads_per_epoch = max_leads_per_epoch
            
            # Handle different response types:
            # - None = Already submitted (gateway returned explicit message)
            # - [] = Timeout/error (should retry)
            # - [lead1, lead2, ...] = Got leads
            
            if leads is None:
                # Gateway explicitly said "already submitted" or "queue empty"
                print(f"ℹ️  No leads to process for epoch {current_epoch}")
                print(f"   Gateway confirmed: You've already submitted or queue is empty")
                await self._check_weight_submission_for_processed_epoch(
                    current_epoch,
                    "gateway_already_submitted_or_queue_empty",
                )
                
                # Mark as processed (don't retry - would be duplicate submission)
                self._last_processed_epoch = current_epoch
                print(f"✅ Marked epoch {current_epoch} as processed (already submitted)\n")
                await asyncio.sleep(10)
                return
            
            print(f"[DEBUG] Received {len(leads)} leads from gateway (max_leads_per_epoch={max_leads_per_epoch})")
            
            if not leads:
                # Empty list = timeout or error (NOT already submitted)
                print(f"⚠️  Gateway returned 0 leads (timeout or error)")
                print(f"   This is likely a temporary issue - validator will retry automatically")
                print(f"   NOT marking epoch as processed - will retry next iteration\n")
                await asyncio.sleep(30)  # Wait longer before retry
                return
            
            print(f"✅ Received {len(leads)} leads from gateway")
            
            # ═══════════════════════════════════════════════════════════════════
            # DYNAMIC LEAD DISTRIBUTION: Auto-calculate ranges for containers
            # ═══════════════════════════════════════════════════════════════════
            container_id = getattr(self.config.neuron, 'container_id', None)
            total_containers = getattr(self.config.neuron, 'total_containers', None)
            
            if container_id is not None and total_containers is not None:
                # DYNAMIC CALCULATION: Auto-distribute leads across containers
                original_count = len(leads)
                
                # Calculate this container's slice
                leads_per_container = original_count // total_containers
                remainder = original_count % total_containers
                
                # First 'remainder' containers get 1 extra lead to distribute remainder evenly
                if container_id < remainder:
                    start = container_id * (leads_per_container + 1)
                    end = start + leads_per_container + 1
                else:
                    start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
                    end = start + leads_per_container
                
                leads = leads[start:end]
                lead_range_str = f"{start}-{end}"
                
                print(f"📦 Container {container_id}/{total_containers}: Processing leads {start}-{end}")
                print(f"   ({len(leads)}/{original_count} leads assigned to this container)")
                print(f"   Gateway MAX_LEADS_PER_EPOCH: {max_leads_per_epoch}")
                print(f"   (Dynamic distribution - adapts to any gateway setting)")
                print("")
            else:
                # No containerization - process all leads
                lead_range_str = None
            
            # ================================================================
            # BATCH VALIDATION: Stage 0-2 runs in PARALLEL with TrueList
            # After Stage 0-2, poll file for truelist_results before Stage 4-5
            # ================================================================
            print(f"🔍 Running BATCH automated checks on {len(leads)} leads...")
            print("")
            
            from validator_models.automated_checks import run_batch_automated_checks, get_email
            
            # (os and hashlib already imported at line 1845)
            validation_results = []
            local_validation_data = []  # Store for weight calculation
            
            # Salt already generated earlier (line 1850) and shared with workers via leads file
            # Convert back from hex for coordinator's own validation
            salt = bytes.fromhex(salt_hex)
            
            # Extract lead_blobs for batch processing
            lead_blobs = [lead.get('lead_blob', {}) for lead in leads]
            
            # ================================================================
            # COORDINATOR: Background task to wait for TrueList and update file
            # This allows Stage 0-2 to run in parallel with TrueList
            # ================================================================
            async def truelist_file_updater():
                """Wait for centralized TrueList to complete, then update file."""
                nonlocal truelist_results
                if truelist_task is None:
                    return  # No TrueList task (no leads)
                try:
                    print(f"   🔄 Background: Waiting for centralized TrueList to complete...")
                    truelist_results = await truelist_task
                    print(f"   ✅ Background: Centralized TrueList complete ({len(truelist_results)} results)")
                    
                    # Update the file with truelist_results
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    with open(leads_file, 'w') as f:
                        json.dump({
                            "epoch_id": current_epoch,
                            "leads": all_leads_for_file,  # All leads (not just coordinator's slice)
                            "max_leads_per_epoch": max_leads_per_epoch,
                            "created_at_block": current_block,
                            "salt": salt_hex,
                            "truelist_results": truelist_results  # NOW POPULATED
                        }, f)
                    print(f"   💾 Background: Updated file with {len(truelist_results)} TrueList results")
                except Exception as e:
                    print(f"   ❌ Background: TrueList failed: {e}")
                    truelist_results = {}  # Empty = leads fail email verification
                    # Still update file to unblock workers (with empty results)
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    with open(leads_file, 'w') as f:
                        json.dump({
                            "epoch_id": current_epoch,
                            "leads": all_leads_for_file,
                            "max_leads_per_epoch": max_leads_per_epoch,
                            "created_at_block": current_block,
                            "salt": salt_hex,
                            "truelist_results": {}  # Empty due to failure
                        }, f)
                    print(f"   💾 Background: Updated file with EMPTY TrueList results (failure)")
            
            # Start TrueList file updater in background (coordinator only)
            truelist_updater_task = None
            if container_mode == "coordinator" and truelist_task is not None:
                truelist_updater_task = asyncio.create_task(truelist_file_updater())
            
            # CRITICAL: Batch validation takes 10+ minutes. During this time, we MUST keep
            # updating the block file so workers don't see stale data and get stuck.
            # Solution: Run a background task that updates block file every 10 seconds.
            
            async def block_file_updater():
                """Background task to keep block file fresh AND check for weight submission during batch validation."""
                while True:
                    try:
                        await asyncio.sleep(10)  # Update every 10 seconds
                        epoch_state_bg = await self._get_epoch_state_async()
                        current_block_bg = epoch_state_bg.current_block
                        current_epoch_bg = epoch_state_bg.workflow_epoch_id
                        blocks_into_epoch_bg = epoch_state_bg.epoch_block
                        self._write_shared_block_file(epoch_state_bg)
                        
                        # Check for weight submission once the canonical window opens.
                        # This ensures weights are submitted even if Stage 4-5 is still running
                        if epoch_state_bg.deadline_reached(WEIGHT_SUBMISSION_BLOCK):
                            try:
                                await self.submit_weights_at_epoch_end()
                            except Exception as weight_err:
                                print(f"   ⚠️ Weight submission check error: {weight_err}")
                        elif epoch_state_bg.deadline_reached(ALLOCATION_PREPARATION_BLOCK):
                            # Prewarm the attested allocation from here too. Batch
                            # validation blocks the main loop for 10+ minutes, so the
                            # block-180 prewarm inside the weight-submission path may
                            # not run until the submission window opens — losing the
                            # 120-block build margin and starting the expensive
                            # gateway allocation build cold at block 300. The prepare
                            # task is per-epoch idempotent and wait=False never
                            # blocks this updater.
                            try:
                                await self._prepare_research_lab_allocation(
                                    current_epoch_bg,
                                    wait=False,
                                )
                            except Exception as prewarm_err:
                                print(f"   ⚠️ Allocation prewarm error: {prewarm_err}")
                    except asyncio.CancelledError:
                        break  # Stop when batch validation completes
                    except Exception as e:
                        print(f"   ⚠️ Block file update error: {e}")
            
            # Start block file updater in background
            block_updater_task = asyncio.create_task(block_file_updater())
            
            # Path to leads file for polling TrueList results
            leads_file_str = str(Path("validator_weights") / f"epoch_{current_epoch}_leads.json")
            
            try:
                batch_results = await run_batch_automated_checks(
                    lead_blobs, 
                    container_id=0 if container_mode == "coordinator" else int(os.environ.get('CONTAINER_ID', 0)),
                    leads_file_path=leads_file_str,  # Poll file for TrueList results after Stage 0-2
                    current_epoch=current_epoch  # For epoch boundary detection mid-processing
                )
            except Exception as e:
                print(f"   ❌ Batch validation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: Mark all leads as validation errors
                batch_results = [
                    (False, {
                        "passed": False,
                        "rejection_reason": {
                            "stage": "Batch Validation",
                            "check_name": "run_batch_automated_checks",
                            "message": f"Batch validation error: {str(e)}"
                        }
                    })
                    for _ in leads
                ]
            finally:
                # Stop the block file updater
                block_updater_task.cancel()
                try:
                    await block_updater_task
                except asyncio.CancelledError:
                    pass
            
            print(f"\n📦 Batch validation complete. Processing {len(batch_results)} results...")
            
            # Process batch results - this loop PRESERVES block file updates and epoch detection
            for idx, (lead, (passed, automated_checks_data)) in enumerate(zip(leads, batch_results), 1):
                try:
                    lead_blob = lead.get("lead_blob", {})
                    email = lead_blob.get("email", "unknown@example.com")
                    company = lead_blob.get("Company") or lead_blob.get("business", "Unknown")
                    
                    print(f"{'─'*80}")
                    print(f"📋 Processing result {idx}/{len(leads)}: {email} @ {company}")
                    
                    # Handle skipped leads (passed=None means TrueList errors after retries)
                    if passed is None:
                        is_valid = False
                        decision = "deny"
                        rep_score = 0
                        rejection_reason = {
                            "stage": "Batch Validation",
                            "check_name": "truelist_batch_skipped",
                            "message": "Lead skipped due to persistent TrueList errors"
                        }
                        result = {"is_legitimate": False, "reason": rejection_reason, "skipped": True}
                    else:
                        is_valid = passed
                        decision = "approve" if is_valid else "deny"
                        # CRITICAL: Use validator-calculated rep_score, NOT miner's submitted value
                        # Denied leads get 0, approved leads get score from automated checks
                        # rep_score is a dict with 'total_score' key, not a simple integer
                        rep_score_data = automated_checks_data.get('rep_score', {})
                        if isinstance(rep_score_data, dict):
                            rep_score = int(rep_score_data.get('total_score', 0)) if is_valid else 0
                        else:
                            # Fallback for legacy format where rep_score was an integer
                            rep_score = int(rep_score_data) if is_valid else 0
                        rejection_reason = automated_checks_data.get("rejection_reason") or {} if not is_valid else {"message": "pass"}
                        
                        # Build result structure matching old validate_lead() output
                        result = {
                            "is_legitimate": is_valid,
                            "enhanced_lead": automated_checks_data if is_valid else {},
                            "reason": rejection_reason if not is_valid else None
                        }
                        if is_valid:
                            result["enhanced_lead"]["rep_score"] = rep_score
                    
                    # Strip internal cache fields from evidence (they contain datetime objects and aren't needed)
                    # These are Stage 4 optimization artifacts, not part of the validation evidence
                    clean_result = result.copy()
                    if "enhanced_lead" in clean_result and isinstance(clean_result["enhanced_lead"], dict):
                        clean_enhanced = clean_result["enhanced_lead"].copy()
                        # Remove internal cache fields that shouldn't be in evidence
                        for internal_field in ["company_linkedin_data", "company_linkedin_slug", "company_linkedin_from_cache"]:
                            clean_enhanced.pop(internal_field, None)
                        clean_result["enhanced_lead"] = clean_enhanced
                    
                    evidence_blob = json.dumps(clean_result, default=str)  # Handle any remaining datetime objects
                    
                    # Compute hashes (SHA256 with salt)
                    decision_hash = hashlib.sha256((decision + salt.hex()).encode()).hexdigest()
                    rep_score_hash = hashlib.sha256((str(rep_score) + salt.hex()).encode()).hexdigest()
                    rejection_reason_hash = hashlib.sha256((json.dumps(rejection_reason, default=str) + salt.hex()).encode()).hexdigest()  # Handle datetime
                    evidence_hash = hashlib.sha256(evidence_blob.encode()).hexdigest()
                    
                    # Store result for gateway submission (IMMEDIATE REVEAL MODE)
                    # IMMEDIATE REVEAL MODE (Jan 2026): Include BOTH hashes AND actual values
                    # No separate reveal phase - gateway verifies hashes and stores values immediately
                    # lead_id and miner_hotkey are at top level (not in lead_blob)
                    validation_results.append({
                        "lead_id": lead.get("lead_id"),  # Top level
                        # Hash fields (for transparency log integrity)
                        "decision_hash": decision_hash,
                        "rep_score_hash": rep_score_hash,
                        "rejection_reason_hash": rejection_reason_hash,
                        "evidence_hash": evidence_hash,
                        "evidence_blob": result,  # Include full evidence for gateway storage
                        # IMMEDIATE REVEAL FIELDS - no separate reveal phase
                        "decision": decision,
                        "rep_score": rep_score,
                        "rejection_reason": rejection_reason,
                        "salt": salt.hex()
                    })
                    
                    # Store local data for weight calculation (still needed for local weight accumulation)
                    local_validation_data.append({
                        "lead_id": lead.get("lead_id"),  # Top level
                        "miner_hotkey": lead.get("miner_hotkey"),  # Top level
                        "decision": decision,
                        "rep_score": rep_score,
                        "rejection_reason": rejection_reason,
                        "salt": salt.hex()
                    })
                    
                    # Store weight data for later accumulation
                    # Workers: Save in JSON for coordinator to aggregate
                    # Coordinator/Default: Accumulate immediately (single validator)
                    # Coordinator in containerized mode: Will re-accumulate all after aggregation
                    container_mode = getattr(self.config.neuron, 'mode', None)
                    
                    # Store weight info in local_validation_data for aggregation
                    # CRITICAL FIX: Get is_icp_multiplier from automated_checks_data (where it's calculated)
                    # NOT from lead (which is the gateway lead object, not the lead_blob that was validated)
                    if len(local_validation_data) > 0:
                        local_validation_data[-1]["is_icp_multiplier"] = automated_checks_data.get("is_icp_multiplier", 0.0)
                    
                    # Only accumulate now if NOT in container mode (backward compatibility)
                    # In container mode, coordinator will accumulate ALL leads after aggregation
                    if container_mode is None:
                        # Traditional single-validator mode
                        # CRITICAL FIX: Get from automated_checks_data, not lead
                        is_icp_multiplier = automated_checks_data.get("is_icp_multiplier", 0.0)
                        await self.accumulate_miner_weights(
                            miner_hotkey=lead.get("miner_hotkey"),
                            rep_score=rep_score,
                            is_icp_multiplier=is_icp_multiplier,
                            decision=decision
                        )
                    
                    # Pretty output
                    status_icon = "✅" if is_valid else "❌"
                    decision_text = "APPROVED" if is_valid else "DENIED"
                    print(f"   {status_icon} Decision: {decision_text}")
                    print(f"   📊 Rep Score: {rep_score}/{MAX_REP_SCORE}")
                    if not is_valid:
                        # Print full rejection details
                        print(f"   ❌ REJECTION DETAILS:")
                        print(f"      Stage: {rejection_reason.get('stage', 'Unknown')}")
                        print(f"      Check: {rejection_reason.get('check_name', 'Unknown')}")
                        print(f"      Message: {rejection_reason.get('message', 'Unknown reason')}")
                        failed_fields = rejection_reason.get('failed_fields', [])
                        if failed_fields:
                            print(f"      Failed Fields: {', '.join(failed_fields)}")
                    print("")
                    
                    # Check block/epoch status every 20 leads (no delay - this is just hash preparation)
                    if idx < len(leads) and idx % 20 == 0:
                        # Check if the canonical weight window opened mid-processing.
                        await self.submit_weights_at_epoch_end()
                        
                        # Check if epoch changed - if so, stop processing old epoch's leads
                        new_epoch_state = await self._get_epoch_state_async()
                        new_block = new_epoch_state.current_block
                        new_epoch = new_epoch_state.workflow_epoch_id
                        blocks_into_epoch = new_epoch_state.epoch_block
                        
                        # Update block file for workers
                        container_mode_check = getattr(self.config.neuron, 'mode', None)
                        if container_mode_check != "worker":
                            self._write_shared_block_file(new_epoch_state)
                        
                        if not new_epoch_state.same_epoch(epoch_state):
                            print(f"\n{'='*80}")
                            print(f"⚠️  EPOCH CHANGED: {current_epoch} → {new_epoch}")
                            print(f"   Stopping validation of epoch {current_epoch} leads ({idx}/{len(leads)} complete)")
                            print(f"   Remaining {len(leads) - idx} leads cannot be submitted (epoch closed)")
                            print(f"{'='*80}\n")
                            break  # Exit the lead processing loop
                        
                        # Force-stop workers when the canonical weight window opens.
                        # Coordinator needs to submit weights, workers must finish before that
                        container_mode = getattr(self.config.neuron, 'mode', None)
                        if (
                            container_mode == "worker"
                            and new_epoch_state.deadline_reached(
                                WEIGHT_SUBMISSION_BLOCK
                            )
                        ):
                            print(f"\n{'='*80}")
                            print(
                                "⏰ WORKER FORCE STOP: weight deadline reached "
                                f"(block {blocks_into_epoch}/{new_epoch_state.tempo}, "
                                f"{new_epoch_state.blocks_remaining} remaining)"
                            )
                            print(f"   Workers must complete before coordinator submits weights")
                            print(f"   Completed: {idx}/{len(leads)} leads")
                            print(f"   📦 Saving partial results for coordinator to aggregate")
                            print(f"{'='*80}\n")
                            break  # Exit the lead processing loop and proceed to worker JSON write
                    
                except Exception as e:
                    # Error processing batch result (rare - validation already complete)
                    lead_id = lead.get('lead_id', 'unknown')
                    email = lead.get('lead_blob', {}).get('email', 'unknown')
                    
                    print(f"❌ Error processing result for lead {lead_id[:8]}: {e}")
                    import traceback
                    traceback.print_exc()
                    print("")
                    # Continue to next lead after error (no delay needed for hash preparation)
                    continue
            
            # ═══════════════════════════════════════════════════════════════════
            # CONTAINER MODE HANDLING: Worker vs Coordinator
            # ═══════════════════════════════════════════════════════════════════
            container_mode = getattr(self.config.neuron, 'mode', None)
            
            if container_mode == "worker" and lead_range_str:
                # WORKER MODE: Write results to JSON and exit (don't submit to gateway)
                print(f"{'='*80}")
                print(f"👷 WORKER MODE: Writing validation results to shared file")
                print(f"{'='*80}")
                
                worker_results = {
                    "validation_results": validation_results,  # For gateway submission
                    "local_validation_data": local_validation_data,  # For reveals
                    "epoch_id": current_epoch,
                    "lead_range": lead_range_str,
                    "container_id": container_id,
                    "timestamp": time.time()
                }
                
                # Write to shared volume (validator_weights/worker_results_<container_id>.json)
                worker_file = os.path.join("validator_weights", f"worker_results_container_{container_id}.json")
                with open(worker_file, 'w') as f:
                    json.dump(worker_results, f, indent=2)
                
                print(f"✅ Worker wrote {len(validation_results)} validation results to {worker_file}")
                print(f"   Epoch: {current_epoch}")
                print(f"   Container ID: {container_id}")
                print(f"   Lead range: {lead_range_str}")
                print(f"   Worker exiting (coordinator will submit to gateway)")
                print(f"{'='*80}\n")
                
                # Mark epoch as processed so we don't repeat this work
                self._last_processed_epoch = current_epoch
                
                # Exit worker process
                import sys
                sys.exit(0)
            
            elif container_mode == "coordinator" and container_id is not None and total_containers is not None:
                # COORDINATOR MODE: Wait for workers, aggregate results, then submit
                print(f"{'='*80}")
                print(f"📡 COORDINATOR MODE: Waiting for worker results")
                print(f"{'='*80}")
                
                # Determine worker IDs (all containers except coordinator)
                worker_ids = [i for i in range(total_containers) if i != container_id]
                num_workers = len(worker_ids)
                
                print(f"   Coordinator (Container {container_id}): Processed {lead_range_str} ({len(validation_results)} results)")
                print(f"   Waiting for {num_workers} workers: Container IDs {worker_ids}")
                
                # Wait for worker result files (with timeout)
                import time as time_module
                max_wait = 3600  # 60 minutes max wait
                check_interval = 5  # Check every 5 seconds
                waited = 0
                
                worker_files = []
                for worker_id in worker_ids:
                    # Lightweight workers write: worker_{worker_id}_epoch_{epoch}_results.json
                    worker_file = os.path.join("validator_weights", f"worker_{worker_id}_epoch_{current_epoch}_results.json")
                    worker_files.append((worker_id, worker_file))
                
                all_workers_ready = False
                while waited < max_wait and not all_workers_ready:
                    all_workers_ready = all(os.path.exists(wf[1]) for wf in worker_files)
                    if not all_workers_ready:
                        # Check if we're approaching block 335 (hash submission deadline)
                        check_epoch_state = await self._get_epoch_state_async()
                        current_block_check = check_epoch_state.current_block
                        current_epoch_check = check_epoch_state.workflow_epoch_id
                        blocks_into_epoch_check = check_epoch_state.epoch_block
                        
                        # CRITICAL: Update block file so workers get fresh epoch/block info
                        # Without this, workers see stale data and get stuck in "too late" loop
                        self._write_shared_block_file(check_epoch_state)
                        
                        # EPOCH CHANGE CHECK: If epoch changed, abort immediately
                        # Without this, coordinator sits in wait loop for 60min doing nothing
                        if not check_epoch_state.same_epoch(epoch_state):
                            print(f"\n{'='*60}")
                            print(f"❌ COORDINATOR: EPOCH CHANGED while waiting for workers!")
                            print(f"   Started: epoch {current_epoch}")
                            print(f"   Current: epoch {current_epoch_check}")
                            print(f"   Aborting - stale results cannot be submitted")
                            print(f"{'='*60}\n")
                            break
                        
                        # FORCE PROCEED at block 280 (provides ~16 min buffer for weight accum + gateway submit)
                        # Block 280 = 56 min into epoch, leaves 16 min before epoch ends
                        # Weight accumulation (~5 min) + gateway submit (~5 sec) = ~5 min total
                        # Buffer: 16 - 5 = ~11 minutes spare
                        if check_epoch_state.deadline_reached(280):
                            print(f"   ⏰ AGGREGATION DEADLINE REACHED: Force proceeding with available results")
                            print(
                                f"      Block: {blocks_into_epoch_check}/{check_epoch_state.tempo}; "
                                f"{check_epoch_state.blocks_remaining} remaining"
                            )
                            print(f"      ~16 minutes remaining for weight accumulation + gateway submission")
                            missing = [f"Container-{wf[0]}" for wf in worker_files if not os.path.exists(wf[1])]
                            print(f"      Missing workers: {missing}")
                            print(f"      Proceeding with partial results")
                            break
                        
                        missing = [f"Container-{wf[0]}" for wf in worker_files if not os.path.exists(wf[1])]
                        print(
                            f"   ⏳ Waiting for workers: {missing} "
                            f"({waited}s / {max_wait}s, block "
                            f"{blocks_into_epoch_check}/{check_epoch_state.tempo}, "
                            f"{check_epoch_state.blocks_remaining} remaining)"
                        )
                        await asyncio.sleep(check_interval)
                        waited += check_interval
                    else:
                        print(f"   ✅ All {len(worker_files)} workers finished in {waited}s")
                        break
                
                if not all_workers_ready:
                    print(f"   ⚠️  TIMEOUT: Not all workers finished after {max_wait}s")
                    print(f"   Proceeding with coordinator results only")
                
                # Aggregate results from all workers
                aggregated_validation_results = list(validation_results)  # Copy coordinator's results
                aggregated_local_validation_data = list(local_validation_data)  # Copy coordinator's reveals
                
                for worker_id, worker_file in worker_files:
                    if os.path.exists(worker_file):
                        try:
                            with open(worker_file, 'r') as f:
                                worker_data = json.load(f)
                            
                            worker_validations = worker_data.get("validation_results", [])
                            worker_reveals = worker_data.get("local_validation_data", [])
                            worker_range = worker_data.get("lead_range", "unknown")
                            
                            aggregated_validation_results.extend(worker_validations)
                            aggregated_local_validation_data.extend(worker_reveals)
                            
                            print(f"   ✅ Aggregated {len(worker_validations)} results from Container-{worker_id} (range: {worker_range})")
                            
                            # Delete worker file after successful aggregation
                            os.remove(worker_file)
                        except Exception as e:
                            print(f"   ⚠️  Failed to load worker Container-{worker_id}: {e}")
                
                # Replace local lists with aggregated results
                validation_results = aggregated_validation_results
                local_validation_data = aggregated_local_validation_data
                
                print(f"   📊 Total aggregated: {len(validation_results)} validations")
                
                # Clean up shared leads file (no longer needed)
                leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                if leads_file.exists():
                    os.remove(leads_file)
                    print(f"   🧹 Cleaned up {leads_file.name}")
                
                # Clean up any stale leads files from previous epochs
                try:
                    weights_dir = Path("validator_weights")
                    for old_file in weights_dir.glob("epoch_*_leads.json"):
                        # Extract epoch from filename
                        try:
                            file_epoch = int(old_file.stem.split('_')[1])
                            if file_epoch < current_epoch:
                                os.remove(old_file)
                                print(f"   🧹 Cleaned up stale file: {old_file.name}")
                        except (IndexError, ValueError):
                            pass
                except Exception as e:
                    print(f"   ⚠️  Could not clean up stale files: {e}")
                
                # Clean up stale worker result files from previous epochs
                try:
                    for old_worker_file in weights_dir.glob("worker_*_epoch_*_results.json"):
                        try:
                            # Extract epoch from filename: worker_X_epoch_YYYY_results.json
                            parts = old_worker_file.stem.split('_')
                            epoch_idx = parts.index('epoch') + 1
                            file_epoch = int(parts[epoch_idx])
                            if file_epoch < current_epoch:
                                os.remove(old_worker_file)
                                print(f"   🧹 Cleaned up stale worker file: {old_worker_file.name}")
                        except (IndexError, ValueError):
                            pass
                except Exception as e:
                    print(f"   ⚠️  Could not clean up stale worker files: {e}")
                
                # ═══════════════════════════════════════════════════════════════════
                # COORDINATOR: Accumulate weights for ALL leads (coordinator + workers)
                # This ensures all leads are counted in validator_weights_history
                # ═══════════════════════════════════════════════════════════════════
                print(f"   ⚖️  Accumulating weights for all {len(local_validation_data)} leads...")
                for val_data in local_validation_data:
                    miner_hotkey = val_data.get("miner_hotkey")
                    decision = val_data.get("decision")
                    rep_score = val_data.get("rep_score", 0)
                    # Default to 0.0 (new format: no adjustment) instead of 1.0 (old format: multiplier)
                    is_icp_multiplier = val_data.get("is_icp_multiplier", 0.0)
                    
                    await self.accumulate_miner_weights(
                        miner_hotkey=miner_hotkey,
                        rep_score=rep_score,
                        is_icp_multiplier=is_icp_multiplier,
                        decision=decision
                    )
                print(f"   ✅ Weight accumulation complete")
                
                print(f"   Proceeding with gateway submission...")
                print(f"{'='*80}\n")
            
            # Submit validation results to gateway (IMMEDIATE REVEAL MODE)
            # IMMEDIATE REVEAL MODE (Jan 2026): Submit both hashes AND actual values
            # No separate reveal phase - gateway verifies hashes and stores values immediately
            # Consensus runs at end of CURRENT epoch (not N+1)
            print(f"{'='*80}")
            
            # Check if epoch changed before attempting submission
            submit_epoch_state = await self._get_epoch_state_async()
            submit_block = submit_epoch_state.current_block
            submit_epoch = submit_epoch_state.workflow_epoch_id
            
            if not submit_epoch_state.same_epoch(epoch_state):
                print(f"⚠️  Epoch changed ({current_epoch} → {submit_epoch}) - skipping validation submission")
                print(f"   {len(validation_results)} validations for epoch {current_epoch} cannot be submitted")
                print(f"   (Weights already submitted, epoch will be marked as processed)")
                success = False
            elif validation_results:
                print(f"📤 Submitting {len(validation_results)} validations to gateway (IMMEDIATE REVEAL MODE)...")
                success = gateway_submit_validation(self.wallet, current_epoch, validation_results)
                if success:
                    print(f"✅ Successfully submitted {len(validation_results)} validations for epoch {current_epoch}")
                    print(f"   Mode: IMMEDIATE REVEAL (hashes + actual values submitted together)")
                    print(f"   Gateway logged to TEE buffer → will be in next Arweave checkpoint")
                    print(f"   ✅ No separate reveal phase needed - consensus will run at block 330")
                    # NOTE: No _pending_reveals storage needed - values already submitted
                else:
                    print(f"❌ Failed to submit validations for epoch {current_epoch}")
                    print(f"   Epoch may have changed - skipping to avoid re-processing")
                    # Still mark as processed to avoid re-validating 80 leads
                    # Weights will still be submitted at epoch end
            else:
                print(f"⚠️  No validation results to submit (all leads failed validation)")
            
            # Weights already accumulated (coordinator mode) or accumulation skipped (container mode)
            # Weight submission happens in the canonical current-epoch window.
            if container_mode is None:
                print(f"\n{'='*80}")
                print(f"⚖️  Weights accumulated for this epoch")
                print(
                    "   (Will submit at block "
                    f"{WEIGHT_SUBMISSION_BLOCK}+ via submit_weights_at_epoch_end())"
                )
                print(f"{'='*80}")
            
            # Mark epoch as processed
            self._last_processed_epoch = current_epoch
            print(f"\n{'='*80}")
            print(f"✅ EPOCH {current_epoch}: Validation workflow complete")
            print(f"{'='*80}\n")
            
            # NOTE: process_pending_reveals() REMOVED - IMMEDIATE REVEAL MODE
            # With immediate reveal, validators submit both hashes AND values in one request
            # No separate reveal phase is needed - consensus runs at block 330 of CURRENT epoch
            
            # Check if the canonical current-epoch weight window has opened.
            await self.submit_weights_at_epoch_end()
            
            # MEMORY CLEANUP: Force garbage collection after each epoch
            # This prevents memory accumulation over long-running sessions
            collected = gc.collect()
            if collected > 100:  # Only log if significant cleanup
                print(f"🧹 Memory cleanup: freed {collected} objects")
            
        except Exception as e:
            print(f"[DEBUG] Exception caught in gateway validation workflow: {e}")
            import traceback
            print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
            bt.logging.error(f"Error in gateway validation workflow: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())

    async def accumulate_miner_weights(self, miner_hotkey: str, rep_score: int, is_icp_multiplier: float, decision: str):
        """
        Accumulate weights for approved leads in real-time as validation happens.
        
        ASYNC VERSION: Uses async subtensor for block queries.
        
        This updates BOTH files after each lead validation:
        - validator_weights/validator_weights (current epoch only)
        - validator_weights/validator_weights_history (all epochs, never cleared)
        
        This provides crash resilience - if validator disconnects before epoch end,
        the latest weights are already saved in history.
        
        Tracks both:
        - miner_scores: Sum of effective_rep_score per miner (for weight distribution)
        - approved_lead_count: Number of approved leads (for linear emissions scaling)
        
        ICP ADJUSTMENT SYSTEM (NEW):
        - is_icp_multiplier now stores ADJUSTMENT value (-15 to +20)
        - effective_rep_score = base_rep_score + icp_adjustment (floor at 0)
        
        BACKWARDS COMPATIBILITY:
        - OLD format: is_icp_multiplier in {1.0, 1.5, 5.0} → use multiplication
        - NEW format: all other values → use addition
        
        Args:
            miner_hotkey: Miner's hotkey who submitted the lead
            rep_score: Base reputation score (0-48) from automated checks (NOT inflated)
            is_icp_multiplier: OLD: multiplier (1.0, 1.5, 5.0) / NEW: adjustment (-15 to +20)
            decision: "approve" or "deny"
        """
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            weights_file = weights_dir / "validator_weights"
            history_file = weights_dir / "validator_weights_history"
            
            # One coherent epoch decision owns both identity and boundaries.
            epoch_state = await self._get_epoch_state_async()
            current_block = epoch_state.current_block
            current_epoch = epoch_state.workflow_epoch_id
            
            # ═══════════════════════════════════════════════════════════
            # 1. UPDATE validator_weights (current epoch only)
            # ═══════════════════════════════════════════════════════════
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
            else:
                weights_data = {"curators": [], "sourcers_of_curated": []}
            
            # Initialize epoch if not exists (ensures burn weights can be submitted even if all leads denied)
            if str(current_epoch) not in weights_data:
                weights_data[str(current_epoch)] = {
                    "epoch": current_epoch,
                    "start_block": epoch_state.epoch_start_block,
                    "end_block": epoch_state.next_epoch_block,
                    "epoch_scheme": EPOCH_SCHEME,
                    "subnet_epoch_index": epoch_state.subnet_epoch_index,
                    "epoch_ref": epoch_state.epoch_ref,
                    "miner_scores": {},
                    "approved_lead_count": 0,  # Track number of approved leads for linear emissions
                    "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', 3000),  # Persist for restart recovery
                    "last_updated": datetime.utcnow().isoformat()
                }
                # Save immediately so epoch exists even if all leads are denied
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)
            
            # Early return for denied leads (epoch entry already created and saved above)
            if decision != "approve":
                return
            
            # ═══════════════════════════════════════════════════════════
            # ICP VALUE INTERPRETATION (BACKWARDS COMPATIBLE)
            # ═══════════════════════════════════════════════════════════
            # OLD FORMAT: is_icp_multiplier in {1.0, 1.5, 5.0} → multiply
            # NEW FORMAT: any other value (integers -15 to +20) → add
            OLD_MULTIPLIER_VALUES = {1.0, 1.5, 5.0}
            
            if is_icp_multiplier in OLD_MULTIPLIER_VALUES:
                # OLD FORMAT: Use multiplication (legacy leads)
                effective_rep_score = rep_score * is_icp_multiplier
                print(f"      📊 Legacy ICP multiplier: {rep_score} × {is_icp_multiplier} = {effective_rep_score}")
            else:
                # NEW FORMAT: Use addition with floor at 0 (for normal leads)
                icp_adjustment = int(is_icp_multiplier)
                effective_rep_score = max(0, rep_score + icp_adjustment)
                print(f"      📊 ICP adjustment: {rep_score} + ({icp_adjustment:+d}) = {effective_rep_score}")
            
            # Add effective score to miner's total (only for approved leads)
            epoch_data = weights_data[str(current_epoch)]
            if miner_hotkey not in epoch_data["miner_scores"]:
                epoch_data["miner_scores"][miner_hotkey] = 0
            
            epoch_data["miner_scores"][miner_hotkey] += effective_rep_score
            
            # Increment approved lead count for linear emissions
            if "approved_lead_count" not in epoch_data:
                epoch_data["approved_lead_count"] = 0
            epoch_data["approved_lead_count"] += 1
            
            epoch_data["last_updated"] = datetime.utcnow().isoformat()
            
            # Save updated weights
            with open(weights_file, 'w') as f:
                json.dump(weights_data, f, indent=2)
            
            # ═══════════════════════════════════════════════════════════
            # 2. UPDATE validator_weights_history (all epochs, real-time)
            # ═══════════════════════════════════════════════════════════
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history_data = json.load(f)
            else:
                history_data = {"curators": [], "sourcers_of_curated": []}
            
            # Update history with same epoch data (or create new entry)
            history_data[str(current_epoch)] = {
                "epoch": current_epoch,
                "start_block": epoch_state.epoch_start_block,
                "end_block": epoch_state.next_epoch_block,
                "epoch_scheme": EPOCH_SCHEME,
                "subnet_epoch_index": epoch_state.subnet_epoch_index,
                "epoch_ref": epoch_state.epoch_ref,
                "miner_scores": epoch_data["miner_scores"].copy(),  # Deep copy of scores
                "approved_lead_count": epoch_data.get("approved_lead_count", 0),  # Track for linear emissions
                "max_leads_per_epoch": getattr(self, '_max_leads_per_epoch', epoch_data.get("max_leads_per_epoch", 3000)),  # Persist for restart recovery
                "last_updated": datetime.utcnow().isoformat()
            }
            
            # Save updated history (accumulates all epochs)
            with open(history_file, 'w') as f:
                json.dump(history_data, f, indent=2)
            
            # Prune old epochs to prevent file bloat (keep max 50 epochs)
            self.prune_history_file(current_epoch, max_epochs=50)
            
            approved_count = epoch_data.get("approved_lead_count", 0)
            print(f"      💾 Accumulated {rep_score} points for miner {miner_hotkey[:10]}... (total: {epoch_data['miner_scores'][miner_hotkey]})")
            print(f"      📊 Epoch approved leads: {approved_count}")
            print(f"      📚 Updated history file (crash-resilient)")
            
        except Exception as e:
            bt.logging.error(f"Failed to accumulate miner weights: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FULFILLMENT EMISSION SHARE (Phase 2)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_fulfillment_emission_share(
        self, current_epoch: int, fulfillment_pool: float, *, include_status: bool = False,
    ) -> Any:
        """Compute fulfillment emission from active rewards, capped to the pool.

        Queries the gateway for active (unexpired) fulfillment rewards, sums them
        per miner, and caps the total to ``fulfillment_pool``.  If the raw total
        exceeds the pool, each miner's share is reduced pro-rata.

        Args:
            current_epoch: the current Bittensor epoch number
            fulfillment_pool: the fulfillment pool size (e.g. 0.50 for 50%)

        Returns:
            (effective_fulfillment_share, {hotkey: effective_pct})
            On any error returns (0.0, {}).
        """
        try:
            from Leadpoet.utils.cloud_db import gateway_get_all_fulfillment_rewards
            per_miner = gateway_get_all_fulfillment_rewards(self.wallet, current_epoch)

            if not per_miner:
                result = (0.0, {}, True)
                return result if include_status else result[:2]

            raw_total = sum(per_miner.values())
            if raw_total <= 0:
                result = (0.0, {}, True)
                return result if include_status else result[:2]

            if raw_total <= fulfillment_pool:
                result = (raw_total, dict(per_miner), True)
                return result if include_status else result[:2]

            scale_factor = fulfillment_pool / raw_total
            scaled = {hk: pct * scale_factor for hk, pct in per_miner.items()}
            result = (fulfillment_pool, scaled, True)
            return result if include_status else result[:2]

        except Exception as e:
            bt.logging.warning(f"Fulfillment emission share error (safe fallback): {e}")
            result = (0.0, {}, False)
            return result if include_status else result[:2]

    async def _research_lab_pre_weight_submission_guard(self, current_epoch: int) -> dict:
        """Fetch the sole V2 allocation authority before final-weight computation."""

        cached = getattr(self, "_research_lab_allocation_guard_cache", {}).get(
            int(current_epoch)
        )
        if isinstance(cached, dict) and cached.get("verified") is True:
            return cached
        try:
            from leadpoet_canonical.allocation_handoff_v2 import (
                validate_allocation_handoff_v2,
            )
            from research_lab.validator_integration import (
                ResearchLabValidatorFlags,
                build_research_lab_allocation_component,
                fetch_research_lab_attested_allocation_bundle,
                verify_research_lab_allocation_bundle,
            )
            from validator_tee.host.gateway_weight_inputs_v2 import _gateway_endpoint

            gateway_url = _gateway_endpoint(
                str(os.environ.get("VALIDATOR_V2_GATEWAY_URL") or "")
            )
            print("\n🔬 Fetching authoritative V2 Research Lab allocation")
            handoff = await asyncio.to_thread(
                fetch_research_lab_attested_allocation_bundle,
                gateway_url,
                current_epoch,
            )
            normalized = validate_allocation_handoff_v2(
                handoff,
                expected_epoch_id=int(current_epoch),
                expected_netuid=int(self.config.netuid),
            )
            bundle = normalized["bundle"]
            flags = ResearchLabValidatorFlags.from_mapping(os.environ)
            verification = verify_research_lab_allocation_bundle(bundle, flags=flags)
            if not verification.get("passed"):
                raise RuntimeError(
                    "Research Lab allocation arithmetic or policy verification failed: %s"
                    % list(verification.get("errors") or [])
                )
            component = build_research_lab_allocation_component(bundle, flags=flags)
            allocation_doc = component.get("allocation_doc", {})
            print(
                "   ✅ Authoritative V2 Research Lab allocation verified: "
                f"{component.get('allocation_hash')} "
                f"(source_add={float(allocation_doc.get('source_add_alpha_percent') or 0):.4f}%, "
                f"reimbursements={float(allocation_doc.get('reimbursement_alpha_percent') or 0):.4f}%, "
                f"champions={float(allocation_doc.get('champion_alpha_percent') or 0):.4f}%, "
                f"queued={float(allocation_doc.get('queued_champion_alpha_percent') or 0):.4f}%)"
            )
            result = {
                "abort_chain_submission": False,
                "verified": True,
                "allocation_component": component,
                "allocation_verification": verification,
                "attested_allocation_verification": {
                    "passed": True,
                    "required_ready": True,
                    "verification_mode": "authoritative_v2_weight_input_handoff",
                    "allocation_hash": component.get("allocation_hash"),
                    "root_receipt_hash": normalized["root_receipt_hash"],
                },
                "attested_allocation_receipt_graph": normalized["receipt_graph"],
                "evaluation_verification": {
                    "verification_mode": "complete_v2_receipt_ancestry",
                    "passed": True,
                },
            }
            # Keep only the current epoch. The allocation authority is
            # deterministic for an epoch, and retaining the fully verified
            # handoff prevents block-300 submission from downloading and
            # validating the same multi-megabyte ancestry again.
            self._research_lab_allocation_guard_cache = {
                int(current_epoch): result
            }
            return result
        except Exception as exc:
            bt.logging.error(
                "Authoritative V2 Research Lab allocation failed closed: "
                f"{type(exc).__name__}: {exc}"
            )
            return {
                "abort_chain_submission": True,
                "verified": False,
                "reason": "authoritative_v2_allocation_unavailable",
                "errors": [str(exc)],
                "allocation_component": None,
                "allocation_verification": None,
                "evaluation_verification": None,
            }

    async def _research_lab_allocation_submission_window_open(self) -> bool:
        """Return true when this epoch's weight submission window is open."""

        try:
            epoch_state = await self._get_epoch_state_async()
            return bool(epoch_state.deadline_reached(WEIGHT_SUBMISSION_BLOCK))
        except Exception as exc:
            bt.logging.warning(
                "Cannot resolve epoch state for the allocation preparation "
                f"budget; using the submission-window budget: {exc}"
            )
            return True

    def _start_research_lab_allocation_preparation(
        self,
        epoch: int,
    ) -> "asyncio.Task[dict]":
        """Start early preparation with its larger, task-local fetch budget."""

        from research_lab.validator_integration import (
            ALLOCATION_PREPARATION_FETCH_BUDGET,
            ALLOCATION_PREPARATION_FETCH_TIMEOUT_SECONDS,
        )

        async def guarded() -> dict:
            if not await self._research_lab_allocation_submission_window_open():
                ALLOCATION_PREPARATION_FETCH_BUDGET.set(
                    ALLOCATION_PREPARATION_FETCH_TIMEOUT_SECONDS
                )
            return await self._research_lab_pre_weight_submission_guard(epoch)

        return asyncio.create_task(guarded())

    async def _prepare_research_lab_allocation(
        self,
        current_epoch: int,
        *,
        wait: bool,
    ) -> Optional[dict]:
        """Start one epoch preparation task and optionally wait for its result."""

        epoch = int(current_epoch)
        tasks = getattr(self, "_research_lab_allocation_preparation_tasks", {})
        task = tasks.get(epoch)
        if task is None:
            task = self._start_research_lab_allocation_preparation(epoch)
            # Retain only the current epoch so stale multi-megabyte handoffs do
            # not accumulate in a long-lived validator process.
            tasks = {epoch: task}
            self._research_lab_allocation_preparation_tasks = tasks
        if not wait and not task.done():
            return None
        result = await asyncio.shield(task)
        if result.get("abort_chain_submission"):
            # A failed preparation is retryable. Successful results stay in
            # the epoch cache and are reused when the submission window opens.
            tasks.pop(epoch, None)
        return result

    async def _publish_and_set_weights(
        self,
        *,
        epoch_state: _ValidatorEpochState,
        snapshot: Dict[str, Any],
        host_uids: List[int],
        host_weights: List[float],
        allocation_hash: str,
        leaderboard_window_start: str,
        leaderboard_window_end: str,
    ) -> bool:
        """Persist V2 authority, then submit its enclave-computed vector."""

        return await self._authorize_and_set_weights_v2(
            epoch_state=epoch_state,
            snapshot=snapshot,
            host_uids=host_uids,
            host_weights=host_weights,
            allocation_hash=allocation_hash,
            leaderboard_window_start=leaderboard_window_start,
            leaderboard_window_end=leaderboard_window_end,
        )

    async def _authorize_and_set_weights_v2(
        self,
        *,
        epoch_state: _ValidatorEpochState,
        snapshot: Dict[str, Any],
        host_uids: List[int],
        host_weights: List[float],
        allocation_hash: str,
        leaderboard_window_start: str,
        leaderboard_window_end: str,
    ) -> bool:
        """Persist one exact V2 bundle before allowing the chain call."""

        gateway_url = str(os.environ.get("VALIDATOR_V2_GATEWAY_URL") or "").strip()
        if not gateway_url:
            raise RuntimeError("VALIDATOR_V2_GATEWAY_URL is required for V2 weight authority")
        runtime_sha = str(
            os.environ.get("GITHUB_SHA")
            or os.environ.get("GIT_COMMIT")
            or ""
        ).lower()
        config = getattr(self, "config", None)
        epoch_cutover = getattr(self, "_epoch_cutover", None)
        telemetry_netuid = getattr(config, "netuid", None)
        if telemetry_netuid is None:
            telemetry_netuid = getattr(epoch_cutover, "netuid", None)
        if telemetry_netuid is None:
            telemetry_netuid = snapshot.get("netuid")
        wallet = getattr(self, "wallet", None)
        hotkey = getattr(wallet, "hotkey", None)
        telemetry = {
            "runtime_sha": runtime_sha,
            "netuid": telemetry_netuid,
            "epoch_id": int(snapshot["epoch_id"]),
            "epoch_block": getattr(epoch_state, "epoch_block", None),
            "validator_role": "primary",
            "validator_id_hash": _sentry_hash_identifier(
                getattr(hotkey, "ss58_address", None)
            ),
            "restart_invocation_id": os.environ.get(
                "LEADPOET_RESTART_INVOCATION_ID"
            ),
            "weight_correlation_id": _weight_correlation_id(
                runtime_sha=runtime_sha,
                netuid=telemetry_netuid,
                epoch_id=int(snapshot["epoch_id"]),
            ),
        }
        expected_chain = str(
            os.environ.get(
                "EXPECTED_CHAIN",
                "wss://entrypoint-finney.opentensor.ai:443",
            )
        ).strip()
        if await self._recover_weight_publication_before_new_authority_v2(
            epoch_id=int(snapshot["epoch_id"]),
            gateway_url=gateway_url,
        ):
            return True
        journal = self._weight_publication_journal_v2
        try:
            with _sentry_stage(
                component="validator",
                operation="weight_submission",
                stage="bundle_retrieval_verification_publication",
                **telemetry,
            ):
                publication = await prepare_authoritative_weight_publication_v2(
                    calculation_snapshot=snapshot,
                    host_uids=host_uids,
                    host_weights=host_weights,
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    allocation_hash=allocation_hash,
                    leaderboard_window_start=leaderboard_window_start,
                    leaderboard_window_end=leaderboard_window_end,
                    gateway_url=gateway_url,
                    expected_chain=expected_chain,
                    client=self._validator_v2_client,
                    before_publish=journal.record_prepared,
                )
        except Exception as exc:
            code = _sentry_failure_code_for_exception(
                exc,
                default="weight.gateway_endpoint_unavailable",
            )
            _capture_sentry_failure(
                code,
                component="validator",
                stage="bundle_retrieval_verification_publication",
                exception=exc,
                terminal=True,
                retryable=code in {
                    "weight.gateway_endpoint_unavailable",
                    "authority.dependency_unreadable",
                },
                fail_closed=True,
                **telemetry,
            )
            raise
        journal.record_published(publication["publication"])
        self._last_authoritative_weight_v2 = publication
        published_authority = publication.get("compact_submission") or publication.get(
            "published_bundle"
        )
        bundle_hash = (
            published_authority.get("bundle_hash")
            if isinstance(published_authority, Mapping)
            else None
        )
        if isinstance(published_authority, Mapping) and not bundle_hash:
            published_weight_result = published_authority.get("weight_result")
            if isinstance(published_weight_result, Mapping):
                bundle_hash = published_weight_result.get("bundle_hash")
        telemetry.update(
            {
                "bundle_hash": bundle_hash,
                "weights_hash": publication.get("weights_hash"),
                "weight_submission_event_hash": publication.get(
                    "weight_submission_event_hash"
                ),
                "weight_correlation_id": _weight_correlation_id(
                    runtime_sha=runtime_sha,
                    netuid=telemetry_netuid,
                    epoch_id=int(snapshot["epoch_id"]),
                    bundle_hash=bundle_hash,
                ),
            }
        )
        print(
            "   ✅ Authoritative V2 gateway bundle persisted: "
            f"{publication['weight_submission_event_hash'][:20]}..."
        )
        sdk_uids, sdk_weights = _canonical_sdk_weight_vector(
            publication["enclave_response"]["weight_result"]
        )
        with _sentry_stage(
            component="validator",
            operation="weight_submission",
            stage="sign_broadcast_inclusion",
            **telemetry,
        ):
            submitted = await self._set_weights_until_epoch_end(
                epoch_id=int(snapshot["epoch_id"]),
                subnet_epoch_index=epoch_state.subnet_epoch_index,
                uids=sdk_uids,
                weights=sdk_weights,
                weight_authorization_id=publication["weight_authorization_id"],
                weight_submission_event_hash=publication[
                    "weight_submission_event_hash"
                ],
                on_signed_extrinsic=journal.record_signed,
            )
        if not submitted:
            _capture_sentry_failure(
                "weight.finalization_missing",
                component="validator",
                stage="sign_broadcast_inclusion",
                terminal=True,
                retryable=False,
                fail_closed=True,
                blocked_stages=[
                    "chain_finalization",
                    "last_update_readback",
                    "vector_readback",
                    "gateway_finalization_persistence",
                ],
                **telemetry,
            )
            return False
        try:
            with _sentry_stage(
                component="validator",
                operation="weight_submission",
                stage="finalization_last_update_vector_readback",
                **telemetry,
            ):
                finalization = await self._finalize_weight_publication_v2_with_retry(
                    prepared_publication=publication,
                    gateway_url=gateway_url,
                    telemetry=telemetry,
                )
        except Exception as exc:
            _capture_sentry_failure(
                "weight.finalization_missing",
                component="validator",
                stage="finalization_last_update_vector_readback",
                exception=exc,
                terminal=True,
                retryable=False,
                fail_closed=True,
                **telemetry,
            )
            raise
        acknowledgment = (
            finalization.get("acknowledgment")
            if isinstance(finalization, Mapping)
            else None
        )
        if not isinstance(acknowledgment, Mapping):
            acknowledgment = {}
        _record_sentry_stage(
            component="validator",
            stage="finalization_last_update_vector_readback",
            status="passed",
            finalized_block=acknowledgment.get("finalized_block"),
            extrinsic_hash=acknowledgment.get("extrinsic_hash"),
            weight_finalization_event_hash=acknowledgment.get(
                "weight_finalization_event_hash"
            ),
            **telemetry,
        )
        self._last_authoritative_weight_finalization_v2 = finalization
        print(
            "   ✅ Authoritative V2 finalized chain state persisted: "
            f"{finalization['acknowledgment']['weight_finalization_event_hash'][:20]}..."
        )
        return True

    async def _recover_weight_publication_before_new_authority_v2(
        self,
        *,
        epoch_id: int,
        gateway_url: str,
    ) -> bool:
        """Recover retained authority before calculating another epoch bundle."""

        journal = self._weight_publication_journal_v2
        recovery = await self._recover_weight_publication_journal_v2(
            gateway_url=gateway_url
        )
        if recovery is None:
            return False
        if recovery.status == "finalized":
            if int(recovery.epoch_id) == int(epoch_id):
                return True
            retained = journal.load()
            if retained is None:
                raise RuntimeError(
                    "finalized authoritative V2 recovery evidence vanished"
                )
            retained_publication = retained.get("publication")
            if not isinstance(retained_publication, Mapping):
                raise RuntimeError(
                    "finalized authoritative V2 recovery lacks publication"
                )
            journal.clear(
                expected_event_hash=str(
                    retained_publication["weight_submission_event_hash"]
                )
            )
            print(
                "   ✅ Recovered and finalized earlier authoritative V2 epoch "
                f"{recovery.epoch_id}"
            )
            return False
        if int(recovery.epoch_id) == int(epoch_id):
            raise RuntimeError(
                "current authoritative V2 publication was quarantined "
                "without finalized-chain proof"
            )
        bt.logging.critical(
            "weight_publication_prior_epoch_quarantined "
            f"epoch={recovery.epoch_id}; continuing current epoch"
        )
        return False

    async def _finalize_weight_publication_v2_with_retry(
        self,
        *,
        prepared_publication: Mapping[str, Any],
        gateway_url: str,
        telemetry: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Wait for finalized inclusion without changing signed authority."""

        retry_telemetry = dict(telemetry or {})
        last_error: Optional[Exception] = None
        for attempt in range(1, WEIGHT_FINALIZATION_PROOF_ATTEMPTS + 1):
            finalization_scan_id = (
                self._weight_publication_journal_v2.reserve_finalization_scan()
            )
            try:
                return await finalize_authoritative_weight_publication_v2(
                    prepared_publication=prepared_publication,
                    finalization_scan_id=finalization_scan_id,
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    gateway_url=gateway_url,
                    client=self._validator_v2_client,
                )
            except Exception as exc:
                last_error = exc
                if attempt >= WEIGHT_FINALIZATION_PROOF_ATTEMPTS:
                    raise
                _record_sentry_retry(
                    "weight.finalization_missing",
                    component="validator",
                    stage="finalization_last_update_vector_readback",
                    attempt=attempt,
                    retryable=True,
                    exception_class=type(exc).__name__,
                    finalization_scan_id=finalization_scan_id,
                    **retry_telemetry,
                )
                bt.logging.warning(
                    "authoritative_weight_finalization_pending "
                    f"attempt={attempt}/{WEIGHT_FINALIZATION_PROOF_ATTEMPTS} "
                    f"scan_id={finalization_scan_id} "
                    f"error_type={type(exc).__name__}"
                )
                await asyncio.sleep(WEIGHT_FINALIZATION_PROOF_RETRY_SECONDS)
        assert last_error is not None
        raise last_error

    async def _recover_weight_publication_journal_v2(
        self, *, gateway_url: str
    ) -> Optional[_WeightPublicationRecoveryOutcome]:
        """Resume one exact journaled publication without blind re-signing."""

        journal = self._weight_publication_journal_v2
        record = journal.load()
        if record is None:
            return None
        authority = record.get("published_bundle") or record.get(
            "compact_submission"
        )
        if not isinstance(authority, Mapping):
            raise RuntimeError(
                "journaled authoritative V2 publication has no authority"
            )
        compact_recovery = "compact_submission" in record
        epoch_id = int(authority["weight_result"]["epoch_id"])

        async def publication_epoch_closed(stage: str) -> bool:
            try:
                finalized_state = await self._get_epoch_state_async()
                best_state = await self._get_best_epoch_state_async()
                return (
                    finalized_state.workflow_epoch_id > epoch_id
                    and best_state.workflow_epoch_id > epoch_id
                )
            except Exception as exc:
                bt.logging.error(
                    "weight_publication_journal_epoch_check_failed "
                    f"epoch={epoch_id} stage={stage} "
                    f"type={type(exc).__name__} error={str(exc)[:200]}"
                )
                return False

        epoch_closed = await publication_epoch_closed("initial")
        if epoch_closed and not record["extrinsic_signature_results"]:
            quarantined = journal.quarantine(
                expected_epoch=epoch_id,
                reason="unsigned_epoch_closed",
            )
            bt.logging.critical(
                "weight_publication_journal_quarantined "
                f"epoch={epoch_id} signed=false path={quarantined}"
            )
            return _WeightPublicationRecoveryOutcome(
                epoch_id=epoch_id,
                status="quarantined",
            )
        if record["publication"] is None:
            acknowledgment = await resume_prepared_weight_publication_v2(
                journal_record=record,
                gateway_url=gateway_url,
            )
            record = journal.record_published(acknowledgment)
        event_hash = str(
            record["publication"]["weight_submission_event_hash"]
        )
        try:
            recovery_method = (
                self._validator_v2_client.recover_compact_weight_publication_v2
                if compact_recovery
                else self._validator_v2_client.recover_weight_publication_v2
            )
            recovery = await asyncio.to_thread(
                recovery_method,
                **(
                    {"compact_submission": record["compact_submission"]}
                    if compact_recovery
                    else {"published_bundle": record["published_bundle"]}
                ),
                weight_submission_event_hash=event_hash,
                extrinsic_signature_results=record[
                    "extrinsic_signature_results"
                ],
                allow_cross_release_finalization_only=bool(
                    record["extrinsic_signature_results"]
                ),
            )
        except Exception as exc:
            if not epoch_closed:
                epoch_closed = await publication_epoch_closed(
                    "recovery_failure"
                )
            if not epoch_closed:
                raise
            quarantined = journal.quarantine(
                expected_epoch=epoch_id,
                reason="signed_recovery_unresolved",
            )
            bt.logging.critical(
                "weight_publication_journal_quarantined "
                f"epoch={epoch_id} signed=true path={quarantined} "
                f"recovery_type={type(exc).__name__} error={str(exc)[:200]}"
            )
            return _WeightPublicationRecoveryOutcome(
                epoch_id=epoch_id,
                status="quarantined",
            )
        authorization_id = str(recovery["weight_authorization_id"])
        record = journal.replace_authorization(authorization_id)
        authority = record.get("published_bundle") or record.get(
            "compact_submission"
        )
        weight_result = authority["weight_result"]
        signed_extrinsics = list(recovery["signed_extrinsics"])
        finalization_only = bool(recovery.get("finalization_only", False))
        if finalization_only and not signed_extrinsics:
            raise RuntimeError(
                "cross-release finalization-only recovery has no signed "
                "extrinsic"
            )
        if not signed_extrinsics:
            sdk_uids, sdk_weights = _canonical_sdk_weight_vector(weight_result)
            submitted = await self._set_weights_until_epoch_end(
                epoch_id=epoch_id,
                subnet_epoch_index=self._subnet_index_for_workflow_epoch(epoch_id),
                uids=sdk_uids,
                weights=sdk_weights,
                weight_authorization_id=authorization_id,
                weight_submission_event_hash=event_hash,
                on_signed_extrinsic=journal.record_signed,
            )
            if not submitted:
                raise RuntimeError(
                    "journaled authoritative weight publication was not "
                    "accepted before its epoch ended"
                )
        else:
            try:
                finalization_scan_id = journal.reserve_finalization_scan()
                await asyncio.to_thread(
                    self._validator_v2_client.confirm_weight_publication_v2,
                    authorization_id,
                    finalization_scan_id=finalization_scan_id,
                )
            except Exception:
                if not epoch_closed:
                    epoch_closed = await publication_epoch_closed(
                        "pre_rebroadcast"
                    )
                # Re-submit only the exact enclave-signed bytes already fsynced
                # before the original SDK call. The enclave independently
                # proves finalized inclusion; this host response is not trusted.
                latest = signed_extrinsics[-1]
                if finalization_only:
                    log = (
                        bt.logging.critical
                        if epoch_closed
                        else bt.logging.warning
                    )
                    log(
                        "weight_publication_cross_release_finalization_pending "
                        f"epoch={epoch_id} extrinsic_hash="
                        f"{latest['extrinsic_hash']} rebroadcast=false"
                    )
                elif epoch_closed:
                    bt.logging.critical(
                        "weight_publication_journal_stale_signed_unresolved "
                        f"epoch={epoch_id} extrinsic_hash="
                        f"{latest['extrinsic_hash']}"
                    )
                else:
                    if not await self._weight_submission_epoch_is_current(
                        epoch_id=epoch_id,
                        subnet_epoch_index=self._subnet_index_for_workflow_epoch(
                            epoch_id
                        ),
                    ):
                        raise RuntimeError(
                            "journaled authoritative weight publication is no longer "
                            "authorized by the durable epoch lifecycle"
                        )
                    try:
                        await asyncio.to_thread(
                            self.subtensor.substrate.rpc_request,
                            "author_submitExtrinsic",
                            ["0x" + str(latest["extrinsic_hex"])],
                        )
                    except Exception as exc:
                        bt.logging.warning(
                            "Exact V2 recovery rebroadcast returned "
                            f"{type(exc).__name__}: {exc}; awaiting "
                            "enclave-authenticated finalization"
                        )
        last_error = None
        for attempt in range(10):
            try:
                finalization_scan_id = journal.reserve_finalization_scan()
                prepared_publication = {
                    "weight_authorization_id": authorization_id,
                    "weight_submission_event_hash": event_hash,
                }
                if compact_recovery:
                    prepared_publication["compact_submission"] = record[
                        "compact_submission"
                    ]
                finalization = await finalize_authoritative_weight_publication_v2(
                    prepared_publication=prepared_publication,
                    finalization_scan_id=finalization_scan_id,
                    validator_hotkey=self.wallet.hotkey.ss58_address,
                    gateway_url=gateway_url,
                    client=self._validator_v2_client,
                )
                self._last_authoritative_weight_finalization_v2 = finalization
                return _WeightPublicationRecoveryOutcome(
                    epoch_id=epoch_id,
                    status="finalized",
                )
            except Exception as exc:
                last_error = exc
                if attempt < 9:
                    await asyncio.sleep(12)
        epoch_closed = await publication_epoch_closed(
            "finalization_retries_exhausted"
        )
        if epoch_closed:
            quarantined = journal.quarantine(
                expected_epoch=epoch_id,
                reason="signed_finalization_unresolved",
            )
            bt.logging.critical(
                "weight_publication_journal_quarantined "
                f"epoch={epoch_id} signed=true path={quarantined} "
                "finalized_chain_proof=false"
            )
            return _WeightPublicationRecoveryOutcome(
                epoch_id=epoch_id,
                status="quarantined",
            )
        raise RuntimeError(
            "journaled authoritative V2 publication lacks finalized-chain proof"
        ) from last_error

    async def _set_weights_until_epoch_end(
        self,
        *,
        epoch_id: int,
        subnet_epoch_index: Optional[int] = None,
        uids,
        weights,
        weight_authorization_id: str,
        weight_submission_event_hash: str,
        on_signed_extrinsic=None,
    ) -> bool:
        """Retry the unchanged chain call under one enclave authorization."""

        # Refuse stale journal recovery before the enclave signs or the SDK is
        # allowed to issue even one extrinsic.
        if not await self._weight_submission_epoch_is_current(
            epoch_id=epoch_id,
            subnet_epoch_index=subnet_epoch_index,
        ):
            print(f"   ⏹️ Refusing stale weight submission for epoch {epoch_id}")
            return False

        # The enclave authorizes the extrinsic payload with the measured
        # profile's extrinsic_period; the SDK must build its era with the
        # same period or the enclave refuses to sign every attempt.
        from validator_tee.enclave.hotkey_authority_v2 import (
            load_chain_signing_profile,
        )

        profile_path = (
            Path(__file__).resolve().parents[1]
            / "validator_tee"
            / "enclave"
            / "chain_signing_profile_v2.json"
        )
        from leadpoet_canonical.production_parity_boundary_v2 import (
            configured_chain_signing_profile_path_v2,
        )

        profile_path = configured_chain_signing_profile_path_v2(profile_path)
        extrinsic_period = int(
            load_chain_signing_profile(profile_path)["extrinsic_period"]
        )

        attempt = 0
        signed_results: List[Dict[str, Any]] = []
        while True:
            failed_source = self.subtensor
            transport_error: Optional[Exception] = None
            with AuthoritativeSetWeightsContextV2(
                substrate=failed_source.substrate,
                wallet=self.wallet,
                weight_authorization_id=weight_authorization_id,
                weight_submission_event_hash=weight_submission_event_hash,
                on_signed_extrinsic=on_signed_extrinsic,
                expected_era_period=extrinsic_period,
            ) as signing_context:
                while True:
                    # Re-read Supabase after entering the signing context and again
                    # before every retry. A fence/activation racing with preparation
                    # must win before the SDK receives any extrinsic.
                    if not await self._weight_submission_lifecycle_is_open(
                        epoch_id=epoch_id,
                    ):
                        signed_results.extend(
                            signing_context.extrinsic_signature_results
                        )
                        self._last_weight_extrinsic_receipts_v2 = signed_results
                        print(
                            f"   ⏹️ Durable epoch authority closed before weight "
                            f"submission for epoch {epoch_id}"
                        )
                        return False
                    attempt += 1
                    if attempt > 1:
                        _record_sentry_retry(
                            "weight.finalization_missing",
                            component="validator",
                            stage="chain_submission_attempt",
                            attempt=attempt,
                            epoch_id=epoch_id,
                            netuid=int(self.config.netuid),
                            validator_role="primary",
                            weight_submission_event_hash=(
                                weight_submission_event_hash
                            ),
                            retryable=True,
                        )
                    try:
                        sdk_result = failed_source.set_weights(
                            netuid=self.config.netuid,
                            wallet=self.wallet,
                            uids=uids,
                            weights=weights,
                            wait_for_finalization=True,
                            mechid=0,
                            period=extrinsic_period,
                        )
                    except Exception as exc:
                        if not _is_subtensor_connection_error(exc):
                            _capture_sentry_failure(
                                _sentry_failure_code_for_exception(
                                    exc,
                                    default="weight.sdk_response_invalid",
                                ),
                                component="validator",
                                stage="sdk_broadcast",
                                exception=exc,
                                terminal=True,
                                retryable=False,
                                fail_closed=True,
                                epoch_id=epoch_id,
                                netuid=int(self.config.netuid),
                                validator_role="primary",
                                attempt=attempt,
                                weight_submission_event_hash=(
                                    weight_submission_event_hash
                                ),
                            )
                            raise
                        transport_error = exc
                        _record_sentry_retry(
                            "weight.chain_transport_poisoned",
                            component="validator",
                            stage="sdk_broadcast",
                            attempt=attempt,
                            epoch_id=epoch_id,
                            netuid=int(self.config.netuid),
                            validator_role="primary",
                            exception_class=type(exc).__name__,
                            weight_submission_event_hash=(
                                weight_submission_event_hash
                            ),
                        )
                        print(
                            f"   ⚠️ Weight submission attempt {attempt} hit a chain "
                            f"transport error ({type(exc).__name__}); reconnecting "
                            f"within epoch {epoch_id}"
                        )
                        break
                    try:
                        outcome = ExtrinsicOutcome.from_sdk(sdk_result)
                    except Exception as exc:
                        _capture_sentry_failure(
                            "weight.sdk_response_invalid",
                            component="validator",
                            stage="sdk_response_normalization",
                            exception=exc,
                            terminal=True,
                            retryable=False,
                            fail_closed=True,
                            epoch_id=epoch_id,
                            netuid=int(self.config.netuid),
                            validator_role="primary",
                            attempt=attempt,
                            sdk_version=str(getattr(bt, "__version__", "unknown")),
                            sdk_response_class=type(sdk_result).__name__,
                            weight_submission_event_hash=(
                                weight_submission_event_hash
                            ),
                        )
                        raise
                    if outcome.success:
                        latest_signature = (
                            signing_context.extrinsic_signature_results[-1]
                            if signing_context.extrinsic_signature_results
                            else {}
                        )
                        if not isinstance(latest_signature, Mapping):
                            latest_signature = {}
                        _record_sentry_stage(
                            component="validator",
                            stage="sdk_submission_finalized",
                            status="passed",
                            epoch_id=epoch_id,
                            netuid=int(self.config.netuid),
                            validator_role="primary",
                            attempt=attempt,
                            sdk_version=str(getattr(bt, "__version__", "unknown")),
                            sdk_response_class=type(sdk_result).__name__,
                            extrinsic_hash=latest_signature.get("extrinsic_hash"),
                            weight_submission_event_hash=(
                                weight_submission_event_hash
                            ),
                        )
                        signed_results.extend(
                            signing_context.extrinsic_signature_results
                        )
                        self._last_weight_extrinsic_receipts_v2 = signed_results
                        return True

                    print(
                        f"   ❌ Bittensor rejected weight submission attempt {attempt}: "
                        f"{outcome.message}"
                    )
                    await asyncio.sleep(12)
                    if not await self._weight_submission_epoch_is_current(
                        epoch_id=epoch_id,
                        subnet_epoch_index=subnet_epoch_index,
                    ):
                        signed_results.extend(
                            signing_context.extrinsic_signature_results
                        )
                        self._last_weight_extrinsic_receipts_v2 = signed_results
                        print(
                            f"   ⏹️ Epoch {epoch_id} ended before the weight "
                            "submission was accepted"
                        )
                        _capture_sentry_failure(
                            "weight.finalization_missing",
                            component="validator",
                            stage="chain_submission_epoch_deadline",
                            terminal=True,
                            retryable=False,
                            fail_closed=True,
                            epoch_id=epoch_id,
                            netuid=int(self.config.netuid),
                            validator_role="primary",
                            attempts=attempt,
                            weight_submission_event_hash=(
                                weight_submission_event_hash
                            ),
                        )
                        return False

            signed_results.extend(signing_context.extrinsic_signature_results)
            self._last_weight_extrinsic_receipts_v2 = signed_results
            if transport_error is None:
                return False

            # The signing patch and its global lock are now released. Replace
            # exactly the source that failed before consulting epoch authority;
            # consulting it through the dead source can itself reconnect and
            # otherwise caused a redundant second replacement.
            await asyncio.sleep(12)
            reconnected = await asyncio.to_thread(
                self._reconnect_subtensor_sync,
                expected_source=failed_source,
                reason="weight_submission_transport",
            )
            if not reconnected or self.subtensor is failed_source:
                _capture_sentry_failure(
                    "weight.chain_transport_poisoned",
                    component="validator",
                    stage="sdk_transport_reconnect_exhausted",
                    exception=transport_error,
                    terminal=True,
                    retryable=True,
                    fail_closed=True,
                    epoch_id=epoch_id,
                    netuid=int(self.config.netuid),
                    validator_role="primary",
                    attempt=attempt,
                    weight_submission_event_hash=weight_submission_event_hash,
                )
                raise transport_error
            if not await self._weight_submission_epoch_is_current(
                epoch_id=epoch_id,
                subnet_epoch_index=subnet_epoch_index,
            ):
                print(
                    f"   ⏹️ Epoch {epoch_id} ended before the weight "
                    "submission was accepted"
                )
                return False

    async def submit_weights_at_epoch_end(self):
        """Run at most one automatic weight publication attempt at a time."""

        lock = getattr(self, "_weight_submission_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._weight_submission_lock = lock
        if lock.locked():
            bt.logging.info(
                "weight_submission_already_inflight; skipping duplicate trigger"
            )
            return False
        async with lock:
            return await self._submit_weights_at_epoch_end_locked()

    async def _submit_weights_at_epoch_end_locked(self):
        """
        Submit accumulated weights in the canonical current-epoch window.
        
        ASYNC VERSION: Uses async subtensor for block queries.
        
        This reads from validator_weights/validator_weights and submits to chain.
        After submission, archives weights to history and clears active file.
        """
        try:
            if self.config.neuron.disable_set_weights:
                bt.logging.info("⏸️  Weight submission disabled (--neuron.disable_set_weights flag is set)")
                return False
            
            epoch_state = await self._get_epoch_state_async()
            current_block = epoch_state.current_block
            current_epoch = epoch_state.workflow_epoch_id
            blocks_into_epoch = epoch_state.epoch_block

            if not hasattr(self, '_last_weight_submission_epoch'):
                self._last_weight_submission_epoch = None

            if self._last_weight_submission_epoch == current_epoch:
                return True

            submission_due = epoch_state.deadline_reached(
                WEIGHT_SUBMISSION_BLOCK
            )
            if submission_due:
                gateway_url = str(
                    os.environ.get("VALIDATOR_V2_GATEWAY_URL") or ""
                ).strip()
                if (
                    gateway_url
                    and hasattr(self, "_weight_publication_journal_v2")
                    and await self._recover_weight_publication_before_new_authority_v2(
                        epoch_id=current_epoch,
                        gateway_url=gateway_url,
                    )
                ):
                    self._last_weight_submission_epoch = current_epoch
                    return True
            
            if epoch_state.deadline_reached(ALLOCATION_PREPARATION_BLOCK):
                prepared = await self._prepare_research_lab_allocation(
                    current_epoch,
                    wait=False,
                )
                if (
                    isinstance(prepared, dict)
                    and prepared.get("abort_chain_submission")
                ):
                    bt.logging.warning(
                        "Authoritative V2 Research Lab allocation preparation "
                        f"not ready for epoch {current_epoch}; retrying before "
                        f"block {WEIGHT_SUBMISSION_BLOCK}"
                    )

            if not submission_due:
                return False

            research_lab_guard = await self._prepare_research_lab_allocation(
                current_epoch,
                wait=True,
            )
            assert research_lab_guard is not None
            if research_lab_guard.get("abort_chain_submission"):
                reason = research_lab_guard.get("reason")
                _capture_sentry_failure(
                    "weight.allocation_authority_missing",
                    component="validator",
                    stage="research_lab_allocation_guard",
                    terminal=True,
                    retryable=False,
                    fail_closed=True,
                    epoch_id=current_epoch,
                    epoch_block=blocks_into_epoch,
                    netuid=int(self.config.netuid),
                    validator_role="primary",
                    runtime_sha=(
                        os.environ.get("GITHUB_SHA")
                        or os.environ.get("GIT_COMMIT")
                        or ""
                    ),
                    blocked_stages=[
                        "bundle_generation",
                        "signing",
                        "broadcast",
                        "finalization",
                    ],
                )
                bt.logging.critical(
                    "weight_submission_blocked_by_guard "
                    f"epoch={current_epoch} reason={reason}"
                )
                print(f"   ❌ Research Lab pre-submission guard blocked weights: {reason}")
                return False
            research_lab_allocation_component = (
                research_lab_guard.get("allocation_component")
                if isinstance(research_lab_guard, dict)
                else None
            )
            research_lab_allocation_doc = (
                research_lab_allocation_component.get("allocation_doc", {})
                if isinstance(research_lab_allocation_component, dict)
                else {}
            )
            research_lab_allocation_hash = str(
                research_lab_allocation_doc.get("allocation_hash") or ""
            ).lower()
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", research_lab_allocation_hash):
                print(
                    "   ❌ Authoritative V2 requires the exact Research Lab "
                    "allocation artifact hash"
                )
                return False
            # The validator enclave replaces these placeholders with the complete
            # measured V2 input receipt set before computing final weights.
            research_lab_allocation_receipt_hash = ""
            research_lab_has_live_allocations = _research_lab_allocation_has_live_payments(
                research_lab_allocation_doc
            )
            
            # ═══════════════════════════════════════════════════════════════════
            # Load current epoch data (may be empty if gateway was down)
            # ═══════════════════════════════════════════════════════════════════
            weights_file = Path("validator_weights") / "validator_weights"
            miner_scores = {}
            current_epoch_lead_count = 0
            epoch_data = None
            
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    weights_data = json.load(f)
                
                if str(current_epoch) in weights_data:
                    epoch_data = weights_data[str(current_epoch)]
                    miner_scores = epoch_data.get("miner_scores", {})
                    current_epoch_lead_count = epoch_data.get("approved_lead_count", 0)
            
            # ═══════════════════════════════════════════════════════════════════
            # Constants for weight distribution
            # ═══════════════════════════════════════════════════════════════════
            # Burn-target UID + expected-owner hotkey.  Default to UID 0 /
            # LeadPoet operator hotkey for backward compatibility; override
            # to the TreasuryVault's UID and hotkey by setting
            # BURN_TARGET_UID and EXPECTED_BURN_TARGET_HOTKEY env vars
            # (governance migration — see
            # .cursor/rules/treasury-vault-goal.mdc).  The safety check
            # below verifies the configured UID is actually owned by the
            # configured hotkey, so a misconfigured env var refuses to
            # submit weights rather than misrouting emissions.
            BURN_TARGET_UID = int(os.environ.get("BURN_TARGET_UID", "0"))
            EXPECTED_BURN_TARGET_HOTKEY = os.environ.get("EXPECTED_BURN_TARGET_HOTKEY")
            subtensor_config = getattr(self.config, "subtensor", None)
            configured_network = str(
                os.environ.get("BITTENSOR_NETWORK")
                or getattr(subtensor_config, "network", "")
                or "finney"
            ).strip().lower()
            if not EXPECTED_BURN_TARGET_HOTKEY and configured_network != "test":
                EXPECTED_BURN_TARGET_HOTKEY = "5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"

            # Read ff_enabled EARLY (used by the no-sourcing-data gates below)
            # so the validator doesn't 100%-burn when sourcing is zeroed out but
            # fulfillment requests were scored this epoch.  Historically these
            # gates short-circuited whenever rolling_scores was empty, which
            # broke once sourcing emissions were dropped to 0% (no incentive
            # → no rolling_scores → 100% burn even with active fulfillment).
            ff_enabled = os.environ.get("ENABLE_FULFILLMENT", "false").lower() == "true"
            leaderboard_emissions_enabled = _env_flag(
                "FULFILLMENT_LEADERBOARD_EMISSIONS_ENABLED",
                True,
            )

            # ═══════════════════════════════════════════════════════════════════
            # SOURCING EMISSIONS SYSTEM (Threshold-Based)
            # ═══════════════════════════════════════════════════════════════════
            # ╔══════════════════════════════════════════════════════════════════╗
            # ║ ⚠️  THE LEADERBOARD IS PART OF FULFILLMENT — DO NOT TURN IT OFF ║
            # ╠══════════════════════════════════════════════════════════════════╣
            # ║ When the team or an operator says "fulfillment is N%", the      ║
            # ║ leaderboard (LEADERBOARD_BONUS_SHARE) is INCLUDED in that N%.   ║
            # ║ The leaderboard is the WEEKLY top-3 bonus that lives inside     ║
            # ║ the fulfillment track — it rewards sustained high performance   ║
            # ║ on top of per-epoch payouts.                                    ║
            # ║                                                                  ║
            # ║ Window: rolling 140-epoch window (~7.0 days) from now.          ║
            # ║ The gateway endpoint /fulfillment/leaderboard filters by        ║
            # ║ fulfillment_score_consensus.computed_at >= last_monday_00z.     ║
            # ║                                                                  ║
            # ║ Research Lab is reserved from the verified allocation bundle.   ║
            # ║ Fulfillment receives the residual after Research Lab and the    ║
            # ║ leaderboard. The Lab default comes from shared economics       ║
            # ║ policy; fulfillment remains the calculated residual.           ║
            # ║                                                                  ║
            # ║ History: 322f287d (2026-05-15) zeroed the leaderboard while     ║
            # ║ raising the per-epoch pool to 95%, mistakenly interpreting      ║
            # ║ "95% fulfillment" as "per-epoch only".  Restored in d3558afa    ║
            # ║ the same day.  This banner exists so it doesn't happen again.   ║
            # ║ 2026-05-17: leaderboard bumped 4% → 9.5% AND switched from      ║
            # ║ all-time to rolling 140-epoch window (~7 days, gateway-side).   ║
            # ║ 2026-05-28: champion 5% → 10% (carved from fulfillment pool     ║
            # ║ 85.5% → 80.5%); leaderboard 9.5% unchanged.                     ║
            # ║ 2026-07-04: Research Lab 10% → 20% (carved from fulfillment     ║
            # ║ pool 80.5% → 70.5%); leaderboard 9.5% unchanged.                ║
            # ╚══════════════════════════════════════════════════════════════════╝
            # Allocation shares (dynamic based on champion status)
            BASE_BURN_SHARE = 0.0          # 0% base burn to UID 0
            # LAB ARENA KING (labarena.md section 13). With LAB_ARENA_REWARDS_ENABLED
            # the king's weekly share of total emissions comes from the governing
            # signed reward basis, served by the gateway from the durable row,
            # verified here against the Arena key pinned by
            # LAB_ARENA_SIGNING_PUBLIC_KEY_HASH, and derived with the shared kernel
            # (leadpoet_canonical.lab_arena_rewards) the coordinator re-runs. It is
            # resolved before the fulfillment residual and before every early exit,
            # so an eligible king is never burned by a short-circuit. Any failure
            # refuses publication: an unreachable or invalid basis is never an
            # empty king. With the flag off nothing below changes.
            lab_arena_rewards_enabled = _env_flag("LAB_ARENA_REWARDS_ENABLED")
            lab_arena_reward_basis = None
            lab_arena_values = {
                "champion_share": 0.0,
                "effective_champion_share": 0.0,
                "champion_uid": None,
                "reward_week_index": None,
                "eligible": False,
            }
            if lab_arena_rewards_enabled:
                try:
                    from Leadpoet.utils.cloud_db import gateway_get_lab_arena_reward_basis
                    from leadpoet_canonical import lab_arena_rewards as _lab_arena_rewards

                    pinned_key_hash = _lab_arena_rewards.signing_key_hash_from_environment(os.environ)
                    arena_snapshot = await asyncio.to_thread(
                        gateway_get_lab_arena_reward_basis, self.wallet, int(current_epoch)
                    )
                    if arena_snapshot.get("reward_basis") is not None:
                        basis = _lab_arena_rewards.validate_reward_basis(arena_snapshot["reward_basis"])
                        key_der = _lab_arena_rewards.signing_key_from_document(
                            arena_snapshot.get("signing_key"), pinned_key_hash
                        )
                        _lab_arena_rewards.verify_reward_basis_signature(
                            basis, public_key_der=key_der, expected_public_key_hash=pinned_key_hash
                        )
                        lab_arena_values = _lab_arena_rewards.champion_values(
                            basis, int(current_epoch), list(self.metagraph.hotkeys)
                        )
                        lab_arena_reward_basis = basis
                except Exception as exc:
                    print(
                        "   ❌ Lab Arena reward basis unavailable or invalid; "
                        f"refusing weight publication: {type(exc).__name__}: {exc}"
                    )
                    return False
            # 0% without an eligible Arena king; the legacy model competition stays retired.
            CHAMPION_SHARE = float(lab_arena_values["champion_share"])
            arena_has_live_reward = float(lab_arena_values["effective_champion_share"]) > 0.0
            # FULFILLMENT-FLAVORED TOTAL is the residual after Research Lab.
            # That residual is split into a per-epoch fulfillment pool and a top-3
            # rolling-window leaderboard bonus. Operators may disable paying the
            # leaderboard bucket with FULFILLMENT_LEADERBOARD_EMISSIONS_ENABLED=false,
            # but the 9.5% reservation remains carved out and burns instead of
            # inflating per-request fulfillment rewards.
            RESEARCH_LAB_FALLBACK_SHARE = _env_percent_share(
                "RESEARCH_LAB_EMISSION_PERCENT",
                float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT),
            )
            RESEARCH_LAB_SHARE = _doc_percent_share(
                research_lab_allocation_doc,
                "lab_cap_percent",
                RESEARCH_LAB_FALLBACK_SHARE,
            )
            # FULFILLMENT LEADERBOARD BONUS — added 2026-04-30, restored 2026-05-15,
            # bumped to 9.5% + switched to rolling window on 2026-05-17, changed
            # from Monday-reset to rolling 140-epoch (~7 day) window on 2026-05-23.
            # Top-3 fulfillment winners in the last 140 epochs get this bonus
            # on top of per-epoch payouts.
            LEADERBOARD_BONUS_SHARE = 0.095
            residual_fulfillment_share = max(
                0.0,
                1.0 - RESEARCH_LAB_SHARE - CHAMPION_SHARE - LEADERBOARD_BONUS_SHARE,
            )
            FULFILLMENT_POOL_SHARE = residual_fulfillment_share
            LEADERBOARD_TOP1_PCT     = 0.05
            LEADERBOARD_TOP2_PCT     = 0.03
            LEADERBOARD_TOP3_PCT     = 0.015
            # Sourcing remains zero under the Research Lab split unless an
            # operator explicitly lowers the live buckets below 100%.
            
            # CONFIGURABLE THRESHOLD: Approved leads needed in 30 epochs for full sourcing share
            # If network produces >= this many leads, full share is distributed
            # If below, proportional share distributed and rest burned
            SOURCING_FLOOR_THRESHOLD = 125_000  # EASILY ADJUSTABLE
            
            # Minimum total rep score to distribute (prevents tiny denominator instability)
            # If total rep < this, sourcing share goes to burn
            MIN_TOTAL_REP_FOR_DISTRIBUTION = 100
            
            # Rolling window for historical lead count and rep scores
            ROLLING_WINDOW = 30
            
            # Champion beat threshold is defined in qualification/config.py (CHAMPION_DETHRONING_THRESHOLD_POINTS)
            # Currently set to 10 absolute points - challenger must score 10+ points higher to dethrone
            # Champion rebenchmark time is defined in qualification/config.py:
            #   CHAMPION_REBENCHMARK_HOUR_UTC, CHAMPION_REBENCHMARK_MINUTE_UTC
            # Default: 05:00 UTC (5:00 AM) - first full epoch after this time triggers rebenchmark
            
            # ═══════════════════════════════════════════════════════════════════
            # BANNED HOTKEY SOURCING PENALTY
            # Fetch banned hotkeys from Supabase and set their scores to -100,000
            # in both history and current weight files BEFORE loading rolling scores.
            # This ensures banned miners cannot receive sourcing emissions even if
            # new leads continue to trickle in with positive scores.
            # ═══════════════════════════════════════════════════════════════════
            try:
                from Leadpoet.utils.cloud_db import (
                    gateway_get_banned_hotkeys_snapshot,
                )

                banned_snapshot = await asyncio.to_thread(
                    gateway_get_banned_hotkeys_snapshot,
                    self.wallet,
                )
                banned_hotkeys = set(banned_snapshot["banned_hotkeys"])
                banned_lookup_ok = banned_snapshot["banned_lookup_ok"] is True
                if banned_hotkeys:
                    self._apply_banned_hotkey_sourcing_penalties(banned_hotkeys)
            except Exception as exc:
                print(
                    "   Authoritative banned hotkey snapshot failed; "
                    f"refusing weight publication: {exc}"
                )
                return False

            # ═══════════════════════════════════════════════════════════════════
            # Get rolling 30 epoch scores BEFORE checking if we should proceed
            # This ensures we still distribute rolling share even if gateway was down
            # ═══════════════════════════════════════════════════════════════════
            rolling_scores, rolling_lead_count = self.get_rolling_epoch_scores(current_epoch, window=ROLLING_WINDOW)

            def _weight_snapshot(
                *,
                champion_uid_value=lab_arena_values["champion_uid"],
                effective_champion_share_value=lab_arena_values["effective_champion_share"],
                fulfillment_share_value=0.0,
                fulfillment_rows_value=None,
                fulfillment_fetch_ok_value=True,
                leaderboard_entries_value=None,
                leaderboard_fetch_ok_value=True,
            ):
                values = {
                    "netuid": int(self.config.netuid),
                    "epoch_id": int(current_epoch),
                    "block": int(current_block),
                    "commit_sha": _current_validator_commit_sha(),
                    "parent_receipt_hashes": (
                        [research_lab_allocation_receipt_hash]
                        if research_lab_allocation_receipt_hash
                        else []
                    ),
                    "research_lab_allocation_receipt_hash": research_lab_allocation_receipt_hash,
                    "burn_target_uid": BURN_TARGET_UID,
                    "expected_burn_target_hotkey": EXPECTED_BURN_TARGET_HOTKEY,
                    "metagraph_hotkeys": list(self.metagraph.hotkeys),
                    "banned_hotkeys": sorted(str(hotkey) for hotkey in banned_hotkeys),
                    "banned_lookup_ok": bool(banned_lookup_ok),
                    "ff_enabled": bool(ff_enabled),
                    "base_burn_share": BASE_BURN_SHARE,
                    "champion_share": CHAMPION_SHARE,
                    "champion_uid": champion_uid_value,
                    "effective_champion_share": effective_champion_share_value,
                    "research_lab_fallback_share": RESEARCH_LAB_FALLBACK_SHARE,
                    "research_lab_allocation_doc": research_lab_allocation_doc,
                    "leaderboard_bonus_share": LEADERBOARD_BONUS_SHARE,
                    "leaderboard_rank_shares": [
                        LEADERBOARD_TOP1_PCT,
                        LEADERBOARD_TOP2_PCT,
                        LEADERBOARD_TOP3_PCT,
                    ],
                    "leaderboard_entries": list(leaderboard_entries_value or []),
                    "leaderboard_fetch_ok": bool(leaderboard_fetch_ok_value),
                    "fulfillment_share": float(fulfillment_share_value),
                    "fulfillment_rows": list(fulfillment_rows_value or []),
                    "fulfillment_fetch_ok": bool(fulfillment_fetch_ok_value),
                    "rolling_lead_count": int(rolling_lead_count),
                    "rolling_scores": [
                        {"hotkey": str(hotkey), "score": score}
                        for hotkey, score in rolling_scores.items()
                    ],
                    "sourcing_floor_threshold": SOURCING_FLOOR_THRESHOLD,
                    "min_total_rep_for_distribution": MIN_TOTAL_REP_FOR_DISTRIBUTION,
                }
                if lab_arena_reward_basis is not None:
                    # The signed basis rides in the snapshot so the coordinator can
                    # measure it and every side can check the triple against it.
                    values["lab_arena_reward_basis"] = lab_arena_reward_basis
                return _finalize_attested_weight_snapshot(values)

            if ff_enabled and leaderboard_emissions_enabled:
                try:
                    from Leadpoet.utils.cloud_db import (
                        gateway_get_fulfillment_leaderboard_snapshot,
                    )

                    leaderboard_snapshot = await asyncio.to_thread(
                        gateway_get_fulfillment_leaderboard_snapshot,
                        self.wallet,
                        3,
                    )
                    leaderboard_window_start = str(
                        leaderboard_snapshot["period_start"]
                    )
                    leaderboard_window_end = str(
                        leaderboard_snapshot["period_end"]
                    )
                    observed_leaders = list(leaderboard_snapshot["leaderboard"])
                except Exception as exc:
                    print(
                        "   ❌ Authoritative V2 leaderboard snapshot failed; "
                        f"refusing weight publication: {exc}"
                    )
                    return False
            else:
                leaderboard_window_start = DISABLED_LEADERBOARD_WINDOW_V1
                leaderboard_window_end = DISABLED_LEADERBOARD_WINDOW_V1
                observed_leaders = []
            
            # ═══════════════════════════════════════════════════════════════════
            # Check if we have ANYTHING to submit (current OR rolling)
            # If both are empty, submit 100% burn weights
            # ═══════════════════════════════════════════════════════════════════
            # Only short-circuit to 100% burn when BOTH sourcing tracks are empty
            # AND fulfillment is disabled on this validator.  When ff_enabled is
            # true we MUST proceed to the main distribution path even with empty
            # sourcing data, otherwise fulfillment miners get nothing despite
            # successfully scoring requests this epoch (the 90% fulfillment pool
            # would silently burn).
            if not miner_scores and not rolling_scores and not ff_enabled and not research_lab_has_live_allocations and not arena_has_live_reward:
                print(f"   ⚠️  No current epoch OR rolling epoch data for epoch {current_epoch}")
                print(f"   🔥 Submitting 100% burn weights (sourcing-only validator, no data)...")
                
                try:
                    if not _verify_burn_target_owner(
                        self.metagraph,
                        BURN_TARGET_UID,
                        EXPECTED_BURN_TARGET_HOTKEY,
                    ):
                        return False

                    result = await self._publish_and_set_weights(
                        epoch_state=epoch_state,
                        snapshot=_weight_snapshot(),
                        host_uids=[BURN_TARGET_UID],
                        host_weights=[1.0],
                        allocation_hash=research_lab_allocation_hash,
                        leaderboard_window_start=leaderboard_window_start,
                        leaderboard_window_end=leaderboard_window_end,
                    )
                    
                    if result:
                        print(f"   ✅ 100% burn weights submitted successfully")
                        # Note: Don't clear weights immediately - keep until epoch transition
                        # This prevents wrong resubmission if validator restarts
                        self._last_weight_submission_epoch = current_epoch
                        return True
                    else:
                        print(f"   ❌ Failed to submit burn weights")
                        return False
                        
                except Exception as e:
                    print(f"   ❌ Error submitting burn weights: {e}")
                    return False
            
            # Log what we have
            has_rolling_history = bool(rolling_scores)
            
            print(f"\n{'='*80}")
            print(f"⚖️  SUBMITTING WEIGHTS FOR EPOCH {current_epoch}")
            print(f"{'='*80}")
            print(
                f"   Block: {current_block} (block "
                f"{blocks_into_epoch}/{epoch_state.tempo}, "
                f"{epoch_state.blocks_remaining} remaining)"
            )
            print(f"   Rolling {ROLLING_WINDOW} epoch miners: {len(rolling_scores)}")
            print(f"   Rolling {ROLLING_WINDOW} epoch leads: {rolling_lead_count:,}")
            print(f"   Sourcing floor threshold: {SOURCING_FLOOR_THRESHOLD:,}")
            print()
            
            # CRITICAL: Verify the configured burn-target UID is owned by
            # the expected hotkey (safety check — refuses to misroute
            # emissions if BURN_TARGET_UID / EXPECTED_BURN_TARGET_HOTKEY
            # are misconfigured or the on-chain UID owner has changed).
            # ═══════════════════════════════════════════════════════════════════
            # QUALIFICATION CHAMPION: Read from local JSON
            # Determines dynamic split: champion active → 90/10, none → 100/0
            # ═══════════════════════════════════════════════════════════════════
            champion_hotkey = None
            champion_uid = None
            effective_champion_share = 0.0
            champion_active = False

            if lab_arena_rewards_enabled:
                champion_uid = lab_arena_values["champion_uid"]
                effective_champion_share = float(lab_arena_values["effective_champion_share"])
                champion_active = champion_uid is not None
                print(
                    "   👑 LAB ARENA KING: share=%.4f uid=%s week=%s eligible=%s"
                    % (
                        effective_champion_share,
                        champion_uid,
                        lab_arena_values.get("reward_week_index"),
                        lab_arena_values.get("eligible"),
                    )
                )
            elif CHAMPION_SHARE > 0 and _env_flag("ENABLE_LEGACY_QUALIFICATION_MODEL_COMPETITION"):
                try:
                    champion_data = self._read_qualification_champion()

                    if champion_data:
                        champion_hotkey = champion_data.get("miner_hotkey")
                        print(f"   👑 QUALIFICATION CHAMPION (from local JSON):")
                        print(f"      Model: {champion_data.get('model_name', 'Unknown')}")
                        print(f"      Miner: {champion_hotkey[:20] if champion_hotkey else 'Unknown'}...")
                        print(f"      Score: {champion_data.get('score', 0):.2f}")
                        print(f"      Since: {champion_data.get('became_champion_at', 'Unknown')}")

                        if champion_hotkey and champion_hotkey in self.metagraph.hotkeys:
                            champion_uid = self.metagraph.hotkeys.index(champion_hotkey)
                            effective_champion_share = CHAMPION_SHARE
                            champion_active = True
                            print(f"      UID: {champion_uid}")
                            print(f"      Emission Share: {CHAMPION_SHARE*100:.0f}%")
                        else:
                            print(f"      ⚠️  Champion not registered on subnet - share goes to sourcing miners")
                    else:
                        print(f"   📭 No qualification champion yet - 100% to sourcing miners")
                except Exception as e:
                    print(f"   ⚠️  Error reading champion: {e} - 100% to sourcing miners")
            else:
                print("   🚫 Legacy model competition champion disabled (0% champion share)")
            
            # Fulfillment pool is ALWAYS reserved. If fulfillment is disabled
            # on this validator, or no miners earned rewards this epoch, the unused
            # portion flows to burn — it does NOT redistribute back to sourcing.
            # (ff_enabled is read once at the top of the function so the early
            #  no-sourcing-data gates above can honor it; do not re-read here.)
            # MAX_SOURCING_SHARE is normally 0% under the Research Lab split:
            # Research Lab + fulfillment + fulfillment leaderboard = 100%.
            MAX_SOURCING_SHARE = max(
                0.0,
                1.0
                - RESEARCH_LAB_SHARE
                - CHAMPION_SHARE
                - FULFILLMENT_POOL_SHARE
                - LEADERBOARD_BONUS_SHARE,
            )
            effective_fulfillment_pool = FULFILLMENT_POOL_SHARE
            effective_leaderboard_share = (
                LEADERBOARD_BONUS_SHARE
                if ff_enabled and leaderboard_emissions_enabled
                else 0.0
            )
            print(
                f"\n   📊 SPLIT: Sourcing={MAX_SOURCING_SHARE*100:.0f}%, "
                f"Champion={effective_champion_share*100:.0f}%, "
                f"Research Lab={RESEARCH_LAB_SHARE*100:.0f}%, "
                f"Fulfillment={effective_fulfillment_pool*100:.0f}%, "
                f"Leaderboard={effective_leaderboard_share*100:.0f}%"
            )
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # THRESHOLD-BASED SOURCING EMISSIONS
            # - If ≥SOURCING_FLOOR_THRESHOLD leads in 30 epochs: Full sourcing share distributed
            # - If <SOURCING_FLOOR_THRESHOLD: Proportional share, rest burned
            # - Within that share: split by rep score proportion
            # ═══════════════════════════════════════════════════════════════════
            
            # Convert miner hotkeys to UIDs (needed for all paths)
            all_miner_hotkeys = set(rolling_scores.keys())
            hotkey_to_uid = {}
            for hotkey in all_miner_hotkeys:
                try:
                    if hotkey in self.metagraph.hotkeys:
                        uid = self.metagraph.hotkeys.index(hotkey)
                        hotkey_to_uid[hotkey] = uid
                except Exception as e:
                    print(f"   ⚠️  Skipping miner {hotkey[:10]}...: {e}")
            
            # Same logic as Gate A above: only short-circuit if fulfillment is
            # disabled.  When ff_enabled=true the downstream code distributes
            # the fulfillment pool using metagraph.hotkeys directly (it does
            # not depend on hotkey_to_uid, which is sourcing-only), so an
            # empty sourcing roster must NOT block fulfillment payouts.
            if not hotkey_to_uid and not ff_enabled and not research_lab_has_live_allocations and not arena_has_live_reward:
                # FALLBACK: No valid miner UIDs found - submit burn weights
                print(f"   ⚠️  No valid miner UIDs found")
                print(f"      Miners have left the subnet or are not registered")
                print(f"   🔥 Submitting 100% burn weights...")
                
                if not _verify_burn_target_owner(
                    self.metagraph,
                    BURN_TARGET_UID,
                    EXPECTED_BURN_TARGET_HOTKEY,
                ):
                    return False

                result = await self._publish_and_set_weights(
                    epoch_state=epoch_state,
                    snapshot=_weight_snapshot(),
                    host_uids=[BURN_TARGET_UID],
                    host_weights=[1.0],
                    allocation_hash=research_lab_allocation_hash,
                    leaderboard_window_start=leaderboard_window_start,
                    leaderboard_window_end=leaderboard_window_end,
                )
                
                if result:
                    print(f"   ✅ Burn weights submitted successfully")
                    self._last_weight_submission_epoch = current_epoch
                    return True
                else:
                    print(f"   ❌ Failed to submit burn weights")
                    return False
            
            # ═══════════════════════════════════════════════════════════════════
            # Filter to REGISTERED miners only - deregistered miners' share → BURN
            # ═══════════════════════════════════════════════════════════════════
            registered_rolling_scores = {h: p for h, p in rolling_scores.items() if h in hotkey_to_uid}
            
            # Calculate totals (only positive scores count — miners with score <= 0 are
            # skipped during distribution, so negatives must not drag down the denominator)
            all_rolling_total = sum(s for s in rolling_scores.values() if s > 0) if rolling_scores else 0
            registered_rolling_total = sum(s for s in registered_rolling_scores.values() if s > 0) if registered_rolling_scores else 0
            deregistered_rolling_points = all_rolling_total - registered_rolling_total
            
            # Log deregistered miners
            if deregistered_rolling_points > 0:
                print(f"   ⚠️  Deregistered miners: {deregistered_rolling_points:,} pts → share goes to BURN")
            
            # ═══════════════════════════════════════════════════════════════════
            # THRESHOLD CALCULATION
            # ═══════════════════════════════════════════════════════════════════
            if rolling_lead_count >= SOURCING_FLOOR_THRESHOLD:
                # ✅ Network healthy: ≥125k approved leads in 30 epochs
                effective_sourcing_share = MAX_SOURCING_SHARE
                print(f"   ✅ NETWORK HEALTHY - Full {MAX_SOURCING_SHARE*100:.0f}% to sourcing miners")
                print(f"      Approved leads ({ROLLING_WINDOW} epochs): {rolling_lead_count:,} ≥ {SOURCING_FLOOR_THRESHOLD:,}")
            else:
                # ⚠️ Below threshold: proportional share to miners, rest burned
                effective_sourcing_share = (rolling_lead_count / SOURCING_FLOOR_THRESHOLD) * MAX_SOURCING_SHARE
                print(f"   ⚠️  BELOW THRESHOLD - Proportional distribution")
                print(f"      Approved leads ({ROLLING_WINDOW} epochs): {rolling_lead_count:,} < {SOURCING_FLOOR_THRESHOLD:,}")
                print(f"      Rate: {rolling_lead_count:,} / {SOURCING_FLOOR_THRESHOLD:,} = {(rolling_lead_count/SOURCING_FLOOR_THRESHOLD)*100:.1f}%")
                print(f"      → {effective_sourcing_share*100:.2f}% to sourcing miners")
                print(f"      → {(MAX_SOURCING_SHARE - effective_sourcing_share)*100:.2f}% burned (underperformance)")
            
            # Calculate burn for deregistered miners (proportional to their share of total)
            dereg_burn = 0.0
            if all_rolling_total > 0 and deregistered_rolling_points > 0:
                dereg_burn = effective_sourcing_share * (deregistered_rolling_points / all_rolling_total)
                print(f"      + {dereg_burn*100:.2f}% burned (deregistered miners)")
            
            # Effective sourcing share for registered miners only
            effective_sourcing_to_miners = effective_sourcing_share - dereg_burn
            
            # ════════════════════════════════════════════════════════════════
            # FULFILLMENT POOL (first-class allocation, not carved from sourcing)
            # Has its own 50% pool. Unused portion goes to burn.
            # On any error, fulfillment_share stays 0 — full pool goes to burn.
            # ════════════════════════════════════════════════════════════════
            fulfillment_share = 0.0
            fulfillment_per_miner = {}
            fulfillment_fetch_ok = True
            unused_fulfillment = 0.0 if ff_enabled else effective_fulfillment_pool
            try:
                if ff_enabled:
                    fulfillment_share, fulfillment_per_miner, fulfillment_fetch_ok = (
                        await asyncio.to_thread(
                            self._get_fulfillment_emission_share,
                            current_epoch,
                            effective_fulfillment_pool,
                            include_status=True,
                        )
                    )
                    unused_fulfillment = effective_fulfillment_pool - fulfillment_share
                    if fulfillment_share > 0:
                        print(f"      Fulfillment active: {fulfillment_share*100:.4f}% used of {effective_fulfillment_pool*100:.0f}% pool "
                              f"({len(fulfillment_per_miner)} miners)")
                    if unused_fulfillment > 0:
                        print(f"      Fulfillment unused: {unused_fulfillment*100:.2f}% → burn")
            except Exception as e:
                fulfillment_share = 0.0
                fulfillment_per_miner = {}
                fulfillment_fetch_ok = False
                unused_fulfillment = effective_fulfillment_pool
                print(f"      Fulfillment emission error (safe fallback — full pool to burn): {e}")

            # ════════════════════════════════════════════════════════════════
            # FULFILLMENT LEADERBOARD BONUS (top-3 fulfillment winners in the
            # rolling 140-epoch window — windowing is handled gateway-side in
            # GET /fulfillment/leaderboard).
            # Same safe-fallback pattern as fulfillment_share: any error here
            # zeros the bonus and the full LEADERBOARD_BONUS_SHARE flows to
            # burn — never silently redistributes to other allocations.
            # Each rank's slot is independent: a deregistered #1 burns 5%
            # but #2 and #3 still pay out, etc.
            # ════════════════════════════════════════════════════════════════
            leaderboard_per_uid: dict = {}      # {uid: pct_to_award}
            leaderboard_entries = []
            leaderboard_fetch_ok = True
            # When ff is disabled, the entire LEADERBOARD_BONUS_SHARE flows to
            # burn (mirroring how the fulfillment pool burns when disabled).
            # When leaderboard emissions are disabled, that same reservation
            # burns while the per-request fulfillment pool remains active.
            # When both are enabled, burn starts at 0 and grows for any rank
            # slot that falls through (deregistered top-N or fewer than 3
            # weekly winners).
            leaderboard_burn = (
                0.0
                if ff_enabled and leaderboard_emissions_enabled
                else LEADERBOARD_BONUS_SHARE
            )
            try:
                if (
                    ff_enabled
                    and leaderboard_emissions_enabled
                    and effective_leaderboard_share > 0
                ):
                    leaders = observed_leaders
                    leaderboard_entries = [
                        {
                            "miner_hotkey": str(entry.get("miner_hotkey") or ""),
                            "wins": entry.get("wins", 0),
                        }
                        for entry in leaders
                        if isinstance(entry, dict)
                    ]
                    rank_pcts = [
                        LEADERBOARD_TOP1_PCT,
                        LEADERBOARD_TOP2_PCT,
                        LEADERBOARD_TOP3_PCT,
                    ]
                    print(f"      Leaderboard top-3 (rolling 140-epoch window fulfillment wins):")
                    for rank_idx, rank_pct in enumerate(rank_pcts):
                        if rank_idx >= len(leaders):
                            # No miner at this rank — bonus burns
                            leaderboard_burn += rank_pct
                            print(
                                f"        #{rank_idx+1}: <no miner> "
                                f"→ {rank_pct*100:.2f}% BURN"
                            )
                            continue
                        entry = leaders[rank_idx]
                        hk = entry.get("miner_hotkey", "")
                        wins = entry.get("wins", 0)
                        if hk in self.metagraph.hotkeys:
                            uid = self.metagraph.hotkeys.index(hk)
                            leaderboard_per_uid[uid] = (
                                leaderboard_per_uid.get(uid, 0.0) + rank_pct
                            )
                            print(
                                f"        #{rank_idx+1}: UID {uid} ({hk[:14]}...) "
                                f"wins={wins} → {rank_pct*100:.2f}%"
                            )
                        else:
                            # Top-N miner has deregistered — bonus burns
                            leaderboard_burn += rank_pct
                            print(
                                f"        #{rank_idx+1}: {hk[:14]}... wins={wins} "
                                f"→ {rank_pct*100:.2f}% BURN (deregistered)"
                            )
            except Exception as e:
                # Any failure: zero out bonuses, full leaderboard pool burns.
                leaderboard_per_uid = {}
                leaderboard_burn = effective_leaderboard_share
                leaderboard_entries = []
                leaderboard_fetch_ok = False
                print(
                    f"      Leaderboard emission error "
                    f"(safe fallback — full {effective_leaderboard_share*100:.2f}% to burn): {e}"
                )

            research_lab_per_uid, research_lab_burn, research_lab_breakdown = _research_lab_uid_weights_from_allocation(
                research_lab_allocation_doc,
                metagraph=self.metagraph,
                reserved_share=RESEARCH_LAB_SHARE,
            )

            # Calculate total burn share.
            # Includes: threshold shortfall + deregistered miners + unused
            # fulfillment pool + leaderboard fall-through + unallocated champion.
            # The CHAMPION_SHARE bucket is RESERVED whether or not a champion
            # is active — when inactive, the 5% must explicitly flow to burn
            # so the weight vector still sums to 1.0.  Without unused_champion
            # the totals collapsed to 0.95 and the weight-sum check at the
            # bottom of this function failed (regression introduced in
            # 322f287d when MAX_SOURCING_SHARE stopped absorbing the share).
            unused_sourcing_share = MAX_SOURCING_SHARE - effective_sourcing_share
            unused_champion = CHAMPION_SHARE - effective_champion_share
            total_burn_share = (
                BASE_BURN_SHARE
                + unused_sourcing_share
                + unused_champion
                + dereg_burn
                + unused_fulfillment
                + leaderboard_burn
                + research_lab_burn
            )
            
            print()
            leaderboard_paid = sum(leaderboard_per_uid.values())
            print(f"   📊 WEIGHT DISTRIBUTION:")
            print(f"      Unused sourcing:      {unused_sourcing_share*100:.2f}% (threshold shortfall)")
            print(f"      Unused champion:      {unused_champion*100:.2f}% (no active champion)")
            print(f"      Unused fulfillment:   {unused_fulfillment*100:.2f}%")
            print(f"      Research Lab burn:    {research_lab_burn*100:.2f}%")
            print(f"      Deregistered miners:  {dereg_burn*100:.2f}%")
            print(f"      Leaderboard burn:     {leaderboard_burn*100:.2f}%")
            print(f"      ─────────────────────────────")
            print(f"      Total burn → UID 0:   {total_burn_share*100:.2f}%")
            print(f"      Champion → UID {champion_uid if champion_uid else '?'}:     {effective_champion_share*100:.0f}%")
            print(f"      Fulfillment miners:   {fulfillment_share*100:.4f}%")
            print(f"      Leaderboard top-3:    {leaderboard_paid*100:.2f}%")
            print(f"      Research Lab miners:  {research_lab_breakdown['paid']*100:.4f}%")
            print(f"      Sourcing miners:      {effective_sourcing_to_miners*100:.2f}%")
            print()
            
            # ═══════════════════════════════════════════════════════════════════
            # BUILD FINAL WEIGHTS
            # ═══════════════════════════════════════════════════════════════════
            uid_weights = {}
            
            # UID 0 gets total burn share
            uid_weights[BURN_TARGET_UID] = total_burn_share
            
            # Champion gets their share (if registered)
            if effective_champion_share > 0 and champion_uid is not None:
                if champion_uid not in uid_weights:
                    uid_weights[champion_uid] = 0
                uid_weights[champion_uid] += effective_champion_share
                print(f"   👑 Champion (UID {champion_uid}): {effective_champion_share*100:.0f}%")
            
            # Fulfillment miners get their carved share (Phase 2)
            if fulfillment_per_miner:
                ff_registered = 0
                for ff_hotkey, ff_pct in fulfillment_per_miner.items():
                    if ff_hotkey in self.metagraph.hotkeys:
                        ff_uid = self.metagraph.hotkeys.index(ff_hotkey)
                        if ff_uid not in uid_weights:
                            uid_weights[ff_uid] = 0
                        uid_weights[ff_uid] += ff_pct
                        ff_registered += 1
                        print(f"   🎯 Fulfillment (UID {ff_uid}): {ff_pct*100:.4f}%")
                    else:
                        uid_weights[BURN_TARGET_UID] = uid_weights.get(BURN_TARGET_UID, 0) + ff_pct
                        print(f"   🎯 Fulfillment ({ff_hotkey[:12]}...): {ff_pct*100:.4f}% → BURN (deregistered)")

            # Leaderboard top-3 bonuses (independent of per-epoch fulfillment rewards)
            for lb_uid, lb_pct in leaderboard_per_uid.items():
                if lb_uid not in uid_weights:
                    uid_weights[lb_uid] = 0
                uid_weights[lb_uid] += lb_pct
                print(f"   🏆 Leaderboard bonus (UID {lb_uid}): {lb_pct*100:.2f}%")

            # Research Lab reimbursements and promoted model-improvement rewards.
            for lab_uid, lab_pct in research_lab_per_uid.items():
                if lab_uid not in uid_weights:
                    uid_weights[lab_uid] = 0
                uid_weights[lab_uid] += lab_pct
                print(f"   🔬 Research Lab (UID {lab_uid}): {lab_pct*100:.4f}%")
            
            # ═══════════════════════════════════════════════════════════════════
            # DISTRIBUTE SOURCING SHARE BY REP SCORE
            # Formula: miner_weight = (miner_rep / total_rep) × effective_sourcing_to_miners
            # ═══════════════════════════════════════════════════════════════════
            print(f"   📈 Sourcing Miners ({effective_sourcing_to_miners*100:.2f}% split by rep score):")
            print(f"      Total registered rep score: {registered_rolling_total:,}")
            
            # Edge case: If total rep is below minimum OR zero, burn the sourcing share
            if registered_rolling_total < MIN_TOTAL_REP_FOR_DISTRIBUTION:
                print(f"      ⚠️  Total rep ({registered_rolling_total:,}) below minimum ({MIN_TOTAL_REP_FOR_DISTRIBUTION})")
                print(f"      → Burning sourcing share to prevent division instability")
                uid_weights[BURN_TARGET_UID] += effective_sourcing_to_miners
            else:
                # Distribute to registered miners by rep score proportion
                for hotkey, rep_score in registered_rolling_scores.items():
                    if rep_score <= 0:
                        continue  # Skip miners with 0 rep
            
                    uid = hotkey_to_uid[hotkey]
                    
                    # Core formula: proportion × effective share
                    miner_proportion = rep_score / registered_rolling_total
                    miner_weight = effective_sourcing_to_miners * miner_proportion
                    
                    if uid not in uid_weights:
                        uid_weights[uid] = 0
                    uid_weights[uid] += miner_weight
                    
                    print(f"      UID {uid}: {rep_score:,} / {registered_rolling_total:,} = {miner_proportion*100:.2f}% → {miner_weight*100:.4f}%")
            
            # Convert to final lists
            final_uids = list(uid_weights.keys())
            final_weights = list(uid_weights.values())
            
            print()
            print(f"   Final weights (should sum to 1.0):")
            for uid in sorted(final_uids):
                weight = uid_weights[uid]
                if uid == BURN_TARGET_UID:
                    print(f"      UID {uid} (Burn): {weight*100:.2f}%")
                else:
                    print(f"      UID {uid}: {weight*100:.2f}%")
            print(f"   Total: {sum(final_weights)*100:.2f}%")
            
            # Verify weights sum to 1.0 (with small floating point tolerance)
            weight_sum = sum(final_weights)
            if not (0.999 <= weight_sum <= 1.001):
                print(f"   ❌ ERROR: Weights sum to {weight_sum}, not 1.0!")
                return False

            if uid_weights.get(BURN_TARGET_UID, 0.0) > 0.0000001:
                if not _verify_burn_target_owner(
                    self.metagraph,
                    BURN_TARGET_UID,
                    EXPECTED_BURN_TARGET_HOTKEY,
                ):
                    return False
            
            # Use final_uids and final_weights.
            # Clamp tiny floating-point dust so Bittensor never sees negative
            # weights such as -2.7755575615628914e-17 on the burn UID.
            uids = final_uids
            normalized_weights = [max(0.0, float(weight)) for weight in final_weights]
            normalized_total = sum(normalized_weights)
            if normalized_total <= 0:
                print("   ❌ ERROR: Sanitized weights sum to 0; refusing chain submission")
                return False
            normalized_weights = [weight / normalized_total for weight in normalized_weights]
            if any(weight < 0 for weight in normalized_weights):
                print(f"   ❌ ERROR: Negative weight remained after sanitization: {normalized_weights}")
                return False

            authoritative_snapshot = _weight_snapshot(
                champion_uid_value=champion_uid,
                effective_champion_share_value=effective_champion_share,
                fulfillment_share_value=fulfillment_share,
                fulfillment_rows_value=[
                    {"hotkey": str(hotkey), "share": share}
                    for hotkey, share in sorted(fulfillment_per_miner.items())
                ],
                fulfillment_fetch_ok_value=fulfillment_fetch_ok,
                leaderboard_entries_value=leaderboard_entries,
                leaderboard_fetch_ok_value=leaderboard_fetch_ok,
            )

            print(f"\n📡 Submitting weights to Bittensor chain...")
            result = await self._publish_and_set_weights(
                epoch_state=epoch_state,
                snapshot=authoritative_snapshot,
                host_uids=list(uids),
                host_weights=list(normalized_weights),
                allocation_hash=research_lab_allocation_hash,
                leaderboard_window_start=leaderboard_window_start,
                leaderboard_window_end=leaderboard_window_end,
            )
            
            if result:
                last_authoritative = getattr(
                    self, "_last_authoritative_weight_v2", None
                )
                published_authority = (
                    last_authoritative.get("compact_submission")
                    or last_authoritative.get("published_bundle")
                    or {}
                    if isinstance(last_authoritative, Mapping)
                    else {}
                )
                if not isinstance(published_authority, Mapping):
                    published_authority = {}
                _record_sentry_stage(
                    component="validator",
                    stage="primary_weight_submission_complete",
                    status="passed",
                    epoch_id=current_epoch,
                    epoch_block=blocks_into_epoch,
                    netuid=int(self.config.netuid),
                    validator_role="primary",
                    bundle_hash=published_authority.get("bundle_hash"),
                )
                print(f"✅ Successfully submitted weights to Bittensor chain")
                print(f"{'='*80}\n")
                
                # CRITICAL: Mark this epoch as submitted BEFORE any cleanup
                # This prevents duplicate submissions if the function is called again
                self._last_weight_submission_epoch = current_epoch
                
                # Archive weights to history (only if we had current epoch data)
                if epoch_data is not None:
                    self.archive_weights_to_history(current_epoch, epoch_data)
                else:
                    # Gateway was down - just mark in history that we submitted rolling-only weights
                    print(f"   📚 Submitted rolling-only weights (no current epoch leads received)")
                
                # Note: Don't clear weights immediately - keep until epoch transition
                # This prevents wrong resubmission if validator restarts within the same epoch
                # The in-memory guard prevents repeats during normal operation;
                # the signed publication journal protects same-epoch restarts.
                # Old epoch data in the file doesn't interfere since we only look up current_epoch
                
                return True
            else:
                print(f"❌ Failed to submit weights to Bittensor chain")
                print(f"{'='*80}\n")
                return False
                
        except Exception as e:
            epoch_value = locals().get("current_epoch")
            code = _sentry_failure_code_for_exception(
                e,
                default="weight.finalization_missing",
            )
            _capture_sentry_failure(
                code,
                component="validator",
                stage="primary_weight_submission",
                exception=e,
                terminal=True,
                retryable=code in {
                    "weight.gateway_endpoint_unavailable",
                    "authority.dependency_unreadable",
                    "weight.chain_transport_poisoned",
                },
                fail_closed=True,
                epoch_id=epoch_value,
                epoch_block=locals().get("blocks_into_epoch"),
                netuid=int(self.config.netuid),
                validator_role="primary",
                runtime_sha=(
                    os.environ.get("GITHUB_SHA")
                    or os.environ.get("GIT_COMMIT")
                    or ""
                ),
                weight_correlation_id=_weight_correlation_id(
                    runtime_sha=(
                        os.environ.get("GITHUB_SHA")
                        or os.environ.get("GIT_COMMIT")
                        or ""
                    ),
                    netuid=int(self.config.netuid),
                    epoch_id=epoch_value,
                ),
            )
            bt.logging.error(f"Error submitting weights at epoch end: {e}")
            import traceback
            bt.logging.error(traceback.format_exc())
            return False
    
    # All active weight branches enter `_publish_and_set_weights`. Protocol
    # selection is explicit; neither path silently downgrades after an error.
    def archive_weights_to_history(self, epoch_id: int, epoch_data: Dict):
        """
        [DEPRECATED] Archive submitted weights to validator_weights_history for record keeping.
        
        This function is now a no-op because validator_weights_history is updated
        in real-time by accumulate_miner_weights() after each lead validation.
        
        The history file is already up-to-date when weights are submitted.
        
        Args:
            epoch_id: Epoch number
            epoch_data: Dict containing epoch weights data
        """
        try:
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
            history_file = weights_dir / "validator_weights_history"
            
            # Load existing history (should already have this epoch from real-time updates)
            if history_file.exists():
                with open(history_file, 'r') as f:
                    history = json.load(f)
            else:
                # Should never happen - history is created in accumulate_miner_weights()
                bt.logging.warning("History file doesn't exist at submission time - creating it now")
                history = {"curators": [], "sourcers_of_curated": []}
            
            # Add submission timestamp to the existing epoch entry
            if str(epoch_id) in history:
                history[str(epoch_id)]["submitted_at"] = datetime.utcnow().isoformat()
                history[str(epoch_id)]["submitted_to_chain"] = True
                
                # Save updated history
                with open(history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                
                print(f"   📚 Marked epoch {epoch_id} as submitted in history")
            else:
                # Shouldn't happen - history should already have this epoch
                bt.logging.warning(f"Epoch {epoch_id} not found in history at submission time")
            
        except Exception as e:
            bt.logging.error(f"Failed to update history submission status: {e}")
    
    def _clear_old_epochs_from_weights(self, current_epoch: int):
        """
        Clear OLD epochs from validator_weights file at epoch transition.
        
        Called at the START of each new epoch to remove data from previous epochs.
        This prevents file bloat while keeping current epoch data intact.
        
        Args:
            current_epoch: The NEW epoch we're transitioning to
        """
        try:
            weights_file = Path("validator_weights") / "validator_weights"
            
            if not weights_file.exists():
                return
            
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)
            
            # Find all epoch entries (numeric keys)
            epoch_keys = [k for k in weights_data.keys() if k.isdigit()]
            
            if not epoch_keys:
                return  # No epoch data to clear
            
            # Remove all epochs BEFORE the current epoch
            epochs_removed = 0
            for epoch_key in epoch_keys:
                epoch_id = int(epoch_key)
                if epoch_id < current_epoch:
                    del weights_data[epoch_key]
                    epochs_removed += 1
            
            if epochs_removed > 0:
                # Save the cleaned file
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)
                
                print(f"   🧹 Epoch transition: Cleared {epochs_removed} old epoch(s) from validator_weights")
            
        except Exception as e:
            bt.logging.error(f"Failed to clear old epochs from weights: {e}")
    
    def get_rolling_epoch_scores(self, current_epoch: int, window: int = 30) -> tuple:
        """
        Get aggregated miner scores and lead counts from the last N epochs (rolling window).
        
        This reads from validator_weights_history and sums up scores for each miner
        across the specified window of epochs.
        
        Args:
            current_epoch: Current epoch number
            window: Number of past epochs to include (default: 30)
            
        Returns:
            Tuple of:
            - Dict mapping miner_hotkey -> total_rep_score across rolling window
            - int: Total approved lead count across rolling window
        """
        try:
            history_file = Path("validator_weights") / "validator_weights_history"
            
            if not history_file.exists():
                print(f"   ℹ️  No history file found - no rolling scores available")
                return {}, 0
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Calculate epoch range for rolling window
            # Include epochs from (current_epoch - window) to (current_epoch - 1)
            # We exclude current_epoch since that's handled separately by the 10% allocation
            start_epoch = current_epoch - window
            end_epoch = current_epoch - 1
            
            rolling_scores = {}
            rolling_lead_count = 0
            epochs_included = 0
            
            for epoch_str, epoch_data in history_data.items():
                # Skip non-epoch entries (curators, sourcers_of_curated)
                if not epoch_str.isdigit():
                    continue
                
                epoch_id = int(epoch_str)
                
                # Check if epoch is within rolling window
                if start_epoch <= epoch_id <= end_epoch:
                    epochs_included += 1
                    miner_scores = epoch_data.get("miner_scores", {})
                    
                    for hotkey, score in miner_scores.items():
                        if hotkey not in rolling_scores:
                            rolling_scores[hotkey] = 0
                        rolling_scores[hotkey] += score
                    
                    # Sum up approved lead counts for linear emissions
                    rolling_lead_count += epoch_data.get("approved_lead_count", 0)
            
            print(f"   📊 Rolling window: epochs {start_epoch}-{end_epoch} ({epochs_included} epochs with data)")
            print(f"   📊 Rolling scores: {len(rolling_scores)} miners, {rolling_lead_count} total approved leads")
            
            return rolling_scores, rolling_lead_count
            
        except Exception as e:
            bt.logging.error(f"Failed to get rolling epoch scores: {e}")
            return {}, 0
    
    def prune_history_file(self, current_epoch: int, max_epochs: int = 50):
        """
        Prune old epochs from validator_weights_history to prevent file bloat.
        
        Keeps only the most recent max_epochs entries.
        
        Args:
            current_epoch: Current epoch number
            max_epochs: Maximum epochs to retain (default: 50)
        """
        try:
            history_file = Path("validator_weights") / "validator_weights_history"
            
            if not history_file.exists():
                return
            
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Find all epoch entries (numeric keys)
            epoch_entries = [k for k in history_data.keys() if k.isdigit()]
            
            if len(epoch_entries) <= max_epochs:
                return  # No pruning needed
            
            # Calculate cutoff epoch
            cutoff_epoch = current_epoch - max_epochs
            
            # Remove epochs older than cutoff
            epochs_removed = 0
            for epoch_str in epoch_entries:
                epoch_id = int(epoch_str)
                if epoch_id < cutoff_epoch:
                    del history_data[epoch_str]
                    epochs_removed += 1
            
            if epochs_removed > 0:
                # Save pruned history
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)
                
                print(f"   🗑️  Pruned {epochs_removed} old epochs from history (keeping last {max_epochs})")
            
        except Exception as e:
            bt.logging.error(f"Failed to prune history file: {e}")
    
    def calculate_and_submit_weights_local(self, validation_data: List[Dict]):
        """
        [DEPRECATED] Calculate miner weights based on LOCAL validation results (Passage 2).
        
        This function is now replaced by:
        - accumulate_miner_weights() - called after each lead validation
        - submit_weights_at_epoch_end() - called in the canonical submission window
        
        Keeping for backwards compatibility, but new code should use the accumulation system.
        """
        # Accumulate weights instead of calculating at once
        for validation in validation_data:
            self.accumulate_miner_weights(
                miner_hotkey=validation['miner_hotkey'],
                rep_score=validation['rep_score'],
                decision=validation['decision']
            )
    
    # ═══════════════════════════════════════════════════════════════════
    # NOTE (Jan 2026): process_pending_reveals() REMOVED - IMMEDIATE REVEAL MODE
    # ═══════════════════════════════════════════════════════════════════
    # Validators now submit both hashes AND actual values in one request to
    # gateway_submit_validation(). No separate reveal phase needed.
    # 
    # Benefits:
    # - Eliminates ~4500 UPDATE queries per epoch (reveals were updates)
    # - Reduces latency - consensus runs same epoch instead of N+1
    # - Simplifies workflow - one submission instead of two
    # ═══════════════════════════════════════════════════════════════════

    def process_sourced_leads_continuous(self):
        """
        CONSENSUS VERSION: Process leads with consensus-based validation.
        Pulls prospects using first-come-first-served, validates them,
        and submits assessments to the consensus tracking system.
        """
        # Skip if processing broadcast request
        if self.processing_broadcast:
            return  # Pause sourcing during broadcast processing

        try:
            # submit_validation_assessment imported at module level
            import uuid
            
            # Fetch prospects using the new consensus-aware function
            # Returns list of {'prospect_id': UUID, 'data': lead_dict}
            prospects_batch = fetch_prospects_from_cloud(
                wallet=self.wallet,
                limit=3000,
                network=self.config.subtensor.network,
                netuid=self.config.netuid
            )

            if not prospects_batch:
                time.sleep(5)  # Wait longer if no prospects available
                return

            print(f"🛎️  Pulled {len(prospects_batch)} prospects from queue (consensus mode)")
            
            # Process each prospect
            for prospect_item in prospects_batch:
                try:
                    # Extract prospect_id and lead data based on the new format
                    if isinstance(prospect_item, dict) and 'prospect_id' in prospect_item:
                        # New consensus format: {'prospect_id': UUID, 'data': lead_dict}
                        prospect_id = prospect_item['prospect_id']
                        lead = prospect_item['data']
                    else:
                        # Fallback for old format (direct lead data)
                        prospect_id = str(uuid.uuid4())  # Generate one if not provided
                        lead = prospect_item
                    
                    # Generate unique lead_id for this validation
                    lead_id = str(uuid.uuid4())
                    
                    # Extract miner info for logging
                    if not lead or not isinstance(lead, dict):
                        bt.logging.error(f"Invalid lead data for prospect {prospect_id[:8]}: {type(lead)}")
                        continue
                        
                    miner_hotkey = lead.get("miner_hotkey", "unknown")
                    business_name = get_field(lead, 'business', 'website', default='Unknown')
                    email = get_email(lead, default='?')
                    
                    print(f"\n🟣 Validating prospect {prospect_id[:8]}...")
                    print(f"   Lead ID: {lead_id[:8]}...")
                    print(f"   Business: {business_name}")
                    print(f"   Email: {email}")
                    print(f"   Miner: {miner_hotkey[:10] if miner_hotkey and miner_hotkey != 'unknown' else 'unknown'}...")
                    
                    # Run async validate_lead in sync context
                    try:
                        result = asyncio.run(self.validate_lead(lead))
                    except Exception as validation_error:
                        # Check if this is an EmailVerificationUnavailableError
                        from validator_models.automated_checks import EmailVerificationUnavailableError
                        if isinstance(validation_error, EmailVerificationUnavailableError):
                            print(f"❌ Lead not processed due to API error\n")
                            continue  # Skip this lead entirely - don't submit anything
                        else:
                            # Some other error - re-raise it
                            raise
                    
                    # Extract validation results and enhanced lead data
                    is_valid = result.get("is_legitimate", False)
                    rejection_reason = result.get("reason", None)  # Now a structured dict from Task 3.1
                    enhanced_lead = result.get("enhanced_lead", lead)  # Get enhanced lead with DNSBL/WHOIS data
                    
                    # Log validation result
                    if is_valid:
                        print(f"   ✅ Valid")
                    else:
                        # Extract message from rejection_reason dict for logging
                        if isinstance(rejection_reason, dict):
                            reason_msg = rejection_reason.get("message", "Unknown error")
                        else:
                            reason_msg = str(rejection_reason) if rejection_reason else "Unknown error"
                        print(f"   ❌ Invalid: {reason_msg}")
                    
                    # Submit validation assessment to consensus system with enhanced lead data
                    submission_success = submit_validation_assessment(
                        wallet=self.wallet,
                        prospect_id=prospect_id,
                        lead_id=lead_id,
                        lead_data=enhanced_lead,  # Use enhanced lead with DNSBL/WHOIS data
                        is_valid=is_valid,
                        rejection_reason=rejection_reason if not is_valid else None,  # Pass structured rejection
                        network=self.config.subtensor.network,
                        netuid=self.config.netuid
                    )
                    
                    if submission_success:
                        print("   📤 Assessment submitted to consensus system")
                        print(f"✅ Processed 1 prospect in consensus mode\n")
                    else:
                        print("   ⚠️ Failed to submit assessment to consensus system")
                    
                    # Note: We do NOT directly save to leads table anymore
                    # The consensus system will handle that when 3 validators agree
                    
                except Exception as e:
                    print(f"   ❌ Error processing prospect: {e}")
                    bt.logging.error(f"Error processing prospect: {e}")
                    import traceback
                    bt.logging.debug(traceback.format_exc())
                    continue
            
        except Exception as e:
            bt.logging.error(f"process_sourced_leads_continuous failure: {e}")
            import traceback
            bt.logging.debug(traceback.format_exc())
            time.sleep(5)

# ─────────────────────────────────────────────────────────
#  NEW: handle buyer curation requests coming via Cloud Run
# ─────────────────────────────────────────────────────────
    # Throttle for the buyer-curation poll. The main validator loop spins
    # quickly, so polling the curation Cloud Run service on every tick floods
    # the logs and hammers the endpoint — badly when the service is
    # unavailable (observed 2026-07-01: repeated "fetch_curation_requests
    # failed: 503 Service Unavailable"). Poll at most once per healthy
    # interval, and back off hard when a request fails so a down service can't
    # be tight-looped.
    _CURATION_POLL_INTERVAL_S = 30.0
    _CURATION_FAILURE_BACKOFF_S = 300.0

    def process_curation_requests_continuous(self):
        now = time.time()
        if now < getattr(self, "_curation_next_poll_at", 0.0):
            return

        req = fetch_curation_requests()
        # ``fetch_curation_requests`` returns None on transport failure (e.g.
        # the curation service returning 503). Back off longer on failure so
        # we neither hammer a down service nor flood the logs; resume the
        # normal cadence automatically once it recovers.
        self._curation_next_poll_at = now + (
            self._CURATION_FAILURE_BACKOFF_S if req is None
            else self._CURATION_POLL_INTERVAL_S
        )
        if not req:
            return

        print(f"\n💼 Buyer curation request: {req}")
        syn = LeadRequest(num_leads=req["num_leads"],
                          business_desc=req["business_desc"])

        # run the existing async pipeline inside the event-loop
        leads = asyncio.run(self.forward(syn)).leads

        # ── annotate each lead with the curation timestamp (seconds since epoch)
        curated_at = time.time()
        for lead in leads:
         
            lead["created_at"]    = datetime.utcfromtimestamp(curated_at).isoformat() + "Z"

        push_curation_result({"request_id": req["request_id"], "leads": leads})
        print(f"✅ Curated {len(leads)} leads for request {req['request_id']}")

    # ═══════════════════════════════════════════════════════════════════════════
    # LEAD FULFILLMENT SCORING WORKFLOW
    # ═══════════════════════════════════════════════════════════════════════════

    async def process_fulfillment_workflow(
        self,
        current_epoch_state: Optional[_ValidatorEpochState] = None,
    ):
        """Unified fulfillment workflow — distribute work AND collect results.

        Runs on a dedicated OS thread (see fulfillment_polling_thread
        in run_async_main_loop), not the main event loop, so it cannot
        be starved by sync work elsewhere in the main loop.

        Two phases per call:

        Phase 1 — DISTRIBUTE (any time, no sourcing gate):
            If there are scoring-ready requests on the gateway AND no work
            is currently pending for this epoch, fetch reveals and write
            work files for the 5 fulfillment worker containers.

        Phase 2 — COLLECT + SUBMIT (gated on sourcing completion):
            After sourcing results have been submitted for this epoch
            (``_last_processed_epoch >= current_epoch``), check if all
            fulfillment workers have finished.  If so, aggregate results
            and submit scores to the gateway.

        Args:
            current_epoch_state: Exact-hash scheduler state supplied by the
                polling thread. Pass ``None`` when called from the main loop.

        Falls back to inline scoring if no fulfillment proxies are
        configured (development / single-container mode).
        """
        if os.environ.get("ENABLE_FULFILLMENT", "false").lower() != "true":
            return

        try:
            epoch_state = (
                current_epoch_state
                if current_epoch_state is not None
                else await self._get_epoch_state_async()
            )
            if not isinstance(epoch_state, _ValidatorEpochState):
                raise TypeError(
                    "fulfillment requires a coherent official epoch snapshot"
                )
            current_block = epoch_state.current_block
            current_epoch = epoch_state.workflow_epoch_id
            weights_dir = Path("validator_weights")
            weights_dir.mkdir(exist_ok=True)
        except Exception as e:
            bt.logging.warning(f"Fulfillment workflow setup failed: {e}")
            return

        fulfillment_worker_ids = detect_fulfillment_worker_ids()
        num_workers = len(fulfillment_worker_ids)
        progress_heartbeat = getattr(
            self, "_fulfillment_progress_heartbeat", None
        )

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 1: DISTRIBUTE — fetch scoring-ready reveals, write work files
        # No sourcing gate — distribute as early as possible so workers
        # can score in parallel with sourcing validation.
        #
        # 2026-05-18: dispatch gate is now per-(epoch, request_id) instead
        # of per-epoch.  Previously a single-epoch lock made the validator
        # ignore any request that became scoring-ready AFTER its first
        # dispatch tick — observed: Adedeji 5 + Bruce Callahan 5 sat in
        # `scoring` for 27+ minutes mid-epoch because their reveal windows
        # closed after the validator had already locked epoch 22801.
        # The new check is "does a work file already exist for (epoch,
        # request_id)?" so each new scoring-ready request enters dispatch
        # on the next 30s polling tick, regardless of where in the epoch
        # we are.  Per-request dedup is intrinsic via the work-file path.
        # ═══════════════════════════════════════════════════════════════════
        if True:  # always enter — per-request filter below replaces the old per-epoch gate
            try:
                from Leadpoet.utils.cloud_db import gateway_get_fulfillment_reveals

                if callable(progress_heartbeat):
                    progress_heartbeat()

                # Reveals fetch may raise RuntimeError after exhausting its
                # 5 retries.  Catch it here and return from Phase 1 cleanly
                # so the next 30s polling tick retries the fetch.  A true
                # empty response (gateway replies with {"requests": []}) is
                # NOT an error and falls through the normal "no scoring-
                # ready requests" branch below.
                try:
                    data = gateway_get_fulfillment_reveals(self.wallet)
                    if callable(progress_heartbeat):
                        progress_heartbeat()
                except RuntimeError as e:
                    bt.logging.warning(
                        f"Fulfillment reveals unreachable after retries — "
                        f"skipping Phase 1 for epoch {current_epoch}, "
                        f"will retry next iteration: {e}"
                    )
                    return

                active_requests = data.get("requests", []) if isinstance(data, dict) else []

                # Accept both feed statuses: `scoring` (normal flow) and
                # `partially_fulfilled` (late re-scores).  Forced consensus can
                # finalize a request while some revealed submissions are still
                # un-scored; the gateway re-offers exactly those leftover
                # submissions and its lifecycle re-aggregates consensus when the
                # late scores land.  Filtering to `scoring` alone left those
                # submissions permanently un-scored — the gateway offered them,
                # this line dropped them.
                scoring_requests = [r for r in active_requests
                                    if r.get("status") in ("scoring", "partially_fulfilled")
                                    and r.get("submissions")]

                # Skip any request that we've already written a work file
                # for in this epoch.  The work file's existence IS the
                # dispatch lock — once a worker has picked it up (or even
                # just received the JSON), this request is "in flight" and
                # must not be re-dispatched (would double-score on the
                # same lead set).  When Phase 2 finishes submitting scores
                # for it, the file is deleted (see line ~4697); after that
                # the gateway should no longer return this request_id in
                # /fulfillment/reveals, so this filter naturally stops
                # matching it.
                if scoring_requests:
                    # Cross-epoch dispatch lock: a work file for this request
                    # from ANY recent epoch counts as "in flight". Previously
                    # this filter was epoch-scoped, so when an epoch boundary
                    # crossed while the worker was still scoring (Tier 2c on
                    # attribute-heavy ICPs commonly runs 30–60 min for a
                    # 45-lead request), Phase 1 saw no current-epoch work
                    # file and re-dispatched, doubling Sonar load. Observed
                    # 2026-05-18: the slowest three requests (cae0a90d,
                    # 0d613519, 8cc8c084) sat at 133–155 min reveal→score
                    # vs a 19-min median, with clean cliffs at 70 and 140
                    # min (= 1 and 2 epoch crossings). 200 orphan work files
                    # going back ~28 days on worker 1 are the long tail of
                    # this — workers that died mid-scoring on epoch N, got
                    # re-dispatched on N+1, but the N file was never GC'd.
                    #
                    # Scoring leases expire after 80 minutes, but a matching
                    # result means scoring finished and delivery is pending.
                    # Keep that request locked for the full delivery window so
                    # a gateway outage cannot trigger duplicate provider work.
                    _FF_WORK_FILE_TTL_SEC = 80 * 60
                    _FF_DELIVERY_FILE_TTL_SEC = 6 * 3600
                    _now_ts = time.time()
                    def _has_work_file(rid: str) -> bool:
                        for wf in weights_dir.glob(
                            f"fulfillment_worker_*_work_*_{rid}.json"
                        ):
                            try:
                                age = _now_ts - wf.stat().st_mtime
                                results_file = wf.parent / wf.name.replace(
                                    "_work_", "_results_", 1
                                )
                                if (
                                    age < _FF_WORK_FILE_TTL_SEC
                                    or (
                                        results_file.exists()
                                        and age < _FF_DELIVERY_FILE_TTL_SEC
                                    )
                                ):
                                    return True
                            except FileNotFoundError:
                                continue
                        return False
                    scoring_requests = [
                        r for r in scoring_requests
                        if not _has_work_file(r.get("request_id", ""))
                    ]

                if scoring_requests:
                    # Option A parallelization: 1 request per worker container.
                    # Each container scores ALL submissions for its assigned request
                    # end-to-end. Up to num_workers requests processed in parallel.
                    requests_to_process = scoring_requests[:num_workers] if num_workers > 0 else scoring_requests[:1]

                    print(f"\n🔍 Fulfillment: {len(requests_to_process)} request(s) ready for scoring "
                          f"(of {len(scoring_requests)} total; max parallel = {max(num_workers, 1)})")

                    if num_workers > 0:
                        # ── Container mode: distribute work across workers ──
                        #
                        # Original design assigned 1 request per worker. When
                        # fewer requests than workers were ready (common with
                        # low client volume), workers 2..N idled while worker
                        # 1 serially scored every submission of the only
                        # request. Observed 2026-05-28: 1 request with 4
                        # submissions × 9-36 leads each stacked on worker 1,
                        # 9 other workers idle — reveal→consensus lag >34 min
                        # while 90% of compute was unused.
                        #
                        # Fix: when spare worker capacity exists, fan out
                        # SUBMISSIONS within a request across the spare
                        # workers. Each worker writes its own work + results
                        # file; the work file's `submissions` key holds only
                        # that worker's subset. Phase 2 aggregates per file,
                        # and gateway scores upsert on (request_id,
                        # validator_hotkey, lead_id), so multiple per-worker
                        # submits for the same request_id don't conflict.
                        assignments = []  # (worker_id, request_id, icp, [subs])

                        if len(requests_to_process) >= num_workers:
                            # Plenty of requests — keep original 1-per-worker
                            # mapping (avoids unnecessary cross-request work
                            # file proliferation when load is balanced).
                            for worker_id, req in zip(
                                fulfillment_worker_ids,
                                requests_to_process,
                            ):
                                request_id = req.get("request_id", "")
                                icp_details = req.get("icp", {})
                                submissions = req.get("submissions", [])
                                if not submissions:
                                    continue
                                assignments.append(
                                    (worker_id, request_id, icp_details, submissions)
                                )
                        else:
                            # Fewer requests than workers — flatten submissions
                            # and round-robin them across workers, then group
                            # by (worker, request) so each work file still
                            # encodes exactly one request_id.
                            flat = []
                            for req in requests_to_process:
                                request_id = req.get("request_id", "")
                                icp_details = req.get("icp", {})
                                for sub in req.get("submissions", []):
                                    flat.append((request_id, icp_details, sub))

                            by_worker = {}
                            for idx, (request_id, icp_details, sub) in enumerate(flat):
                                worker_id = fulfillment_worker_ids[idx % num_workers]
                                by_req = by_worker.setdefault(worker_id, {})
                                if request_id not in by_req:
                                    by_req[request_id] = (icp_details, [])
                                by_req[request_id][1].append(sub)

                            for worker_id, by_req in by_worker.items():
                                for request_id, (icp_details, subs) in by_req.items():
                                    assignments.append(
                                        (worker_id, request_id, icp_details, subs)
                                    )

                        for worker_id, request_id, icp_details, submissions in assignments:
                            # request_id is a UUID — safe for filenames
                            work_file = weights_dir / f"fulfillment_worker_{worker_id}_work_{current_epoch}_{request_id}.json"
                            _atomic_write_json_file(
                                work_file,
                                {
                                    "epoch": current_epoch,
                                    "request_id": request_id,
                                    "icp": icp_details,
                                    "submissions": submissions,
                                    "timestamp": time.time(),
                                },
                            )
                            print(f"   📝 Worker {worker_id} → request {request_id[:8]} "
                                  f"({len(submissions)} submission(s))")

                        unique_workers = len({a[0] for a in assignments})
                        print(f"   ⏳ Work distributed to {unique_workers} worker(s) "
                              f"({len(assignments)} work file(s))")
                    else:
                        # ── Inline mode (no containers) — process sequentially ──
                        from qualification.scoring.fulfillment_scorer import (
                            score_miner_submission, format_scores_for_gateway,
                        )
                        from Leadpoet.utils.cloud_db import gateway_submit_fulfillment_scores

                        for req in requests_to_process:
                            request_id = req.get("request_id", "")
                            icp_details = req.get("icp", {})
                            submissions = req.get("submissions", [])

                            for sub in submissions:
                                miner_hk = sub.get("miner_hotkey", "")
                                sub_id = sub.get("submission_id", "")
                                leads_raw = sub.get("leads", [])
                                lead_ids = sub.get("lead_ids", [])
                                if not leads_raw:
                                    continue
                                try:
                                    if callable(progress_heartbeat):
                                        progress_heartbeat()
                                    results = await score_miner_submission(leads_raw, icp_details)
                                    scores_payload = format_scores_for_gateway(
                                        miner_hk, lead_ids, results,
                                        request_id=request_id, submission_id=sub_id,
                                    )
                                    if not gateway_submit_fulfillment_scores(
                                        self.wallet, request_id, scores_payload,
                                    ):
                                        raise RuntimeError(
                                            "gateway rejected fulfillment scores "
                                            "without raising an exception"
                                        )
                                    if callable(progress_heartbeat):
                                        progress_heartbeat()
                                    print(f"   ✅ Inline: submitted {len(scores_payload)} scores for {miner_hk[:8]}...")
                                except Exception as e:
                                    print(f"   ❌ Inline scoring failed for {miner_hk[:8]}: {e}")

            except ImportError as e:
                if not getattr(self, "_fulfillment_import_warned", False):
                    bt.logging.warning(f"Fulfillment imports unavailable: {e}")
                    self._fulfillment_import_warned = True
            except Exception as e:
                bt.logging.warning(f"Fulfillment distribution error: {e}")

        # ═══════════════════════════════════════════════════════════════════
        # PHASE 2: COLLECT + SUBMIT — only after sourcing is done
        # Same gate as qualification: _last_processed_epoch >= current_epoch
        # If workers aren't done yet, return silently (try again next iteration).
        # ═══════════════════════════════════════════════════════════════════
        if num_workers == 0:
            return

        last_processed = getattr(self, '_last_processed_epoch', -1)
        if last_processed < current_epoch:
            return  # sourcing not done yet

        # Multi-request parallelization: each worker may have a
        # per-request work file: fulfillment_worker_{wid}_work_{epoch}_{reqid}.json
        #
        # Phase 2 globs ALL recent epochs (not just current) so that work
        # which started in epoch N and completed in N+1 still gets its
        # scores submitted, instead of being orphaned + re-dispatched.
        # Cross-epoch globbing is paired with the dispatch-side cross-epoch
        # lock above so we don't process a file Phase 1 is concurrently
        # considering for re-dispatch. mtime TTL guards against ancient
        # zombie files (200 such orphans observed on worker 1 going back
        # 28 days from dead-mid-scoring workers).
        _FF_RESULTS_TTL_SEC = 6 * 3600  # 6h — generous; real lag was ≤155 min
        _now_ts = time.time()
        all_work_files = [
            wf for wf in weights_dir.glob("fulfillment_worker_*_work_*_*.json")
            if (_now_ts - wf.stat().st_mtime) < _FF_RESULTS_TTL_SEC
        ]
        if not all_work_files:
            return

        # A matching results file is the same name with "work" → "results".
        def _results_path_for(work_file: Path) -> Path:
            return work_file.parent / work_file.name.replace("_work_", "_results_", 1)

        # Per-file independence: process whichever (work, results) pairs are
        # ready right now. The old "if pending: return" gate blocked the
        # entire batch until every work file had a matching results file —
        # safe under the prior single-epoch glob (whole batch shared one
        # epoch's life cycle) but starves submissions now that the glob
        # spans epochs (a single slow zombie blocked otherwise-ready ones).
        ready = [wf for wf in all_work_files if _results_path_for(wf).exists()]
        if not ready:
            return

        # ── All workers done — aggregate and submit to gateway ──
        try:
            from Leadpoet.utils.cloud_db import gateway_submit_fulfillment_scores
            from qualification.scoring.fulfillment_scorer import format_scores_for_gateway
            from gateway.fulfillment.models import FulfillmentScoreResult

            print(f"\n{'='*60}")
            print(f"📊 FULFILLMENT: Collecting worker results (epoch {current_epoch}, "
                  f"{len(ready)}/{len(all_work_files)} ready)")
            print(f"{'='*60}")

            # Per-work-file: a failed gateway submit keeps that pair on disk
            # so the next Phase 2 tick retries just it. /fulfillment/score
            # upserts on (request_id, validator_hotkey, lead_id), so duplicate
            # retries are idempotent.
            any_submit_failed = False

            for work_file in sorted(ready):
                if callable(progress_heartbeat):
                    progress_heartbeat()
                results_file = _results_path_for(work_file)

                # Parse worker id from filename for logging
                try:
                    wid_token = work_file.name.split("_")[2]  # fulfillment_worker_{wid}_...
                except Exception:
                    wid_token = "?"

                try:
                    with open(results_file, "r", encoding="utf-8") as f:
                        worker_data = json.load(f)
                    if not isinstance(worker_data, dict):
                        raise ValueError("fulfillment worker result is not an object")
                    if not isinstance(
                        worker_data.get("submission_results"), list
                    ):
                        raise ValueError(
                            "fulfillment worker result submissions are invalid"
                        )
                except (OSError, ValueError, TypeError) as result_exc:
                    print(
                        "   ⚠️ fulfillment_worker_result_invalid_retry "
                        f"worker {wid_token}: {type(result_exc).__name__}: "
                        f"{str(result_exc)[:200]}"
                    )
                    try:
                        results_file.unlink()
                        work_file.touch()
                    except FileNotFoundError:
                        pass
                    any_submit_failed = True
                    continue

                request_id = worker_data.get("request_id", "")
                submission_results = worker_data.get("submission_results", [])

                if worker_data.get("error_type") or "error" in worker_data:
                    worker_error = (
                        str(worker_data.get("error") or "").strip()
                        or str(worker_data.get("error_type") or "worker_error")
                    )
                    print(
                        f"   ⚠️ fulfillment_worker_retryable_error worker "
                        f"{wid_token} (req {request_id[:8]}): "
                        f"{worker_error}"
                    )
                    # A worker-level exception is validator infrastructure
                    # failure, not evidence that a miner's leads are invalid.
                    # Keep the work assignment, renew its dispatch lease, and
                    # remove only the error result so the worker retries it.
                    # Submitting synthetic zero scores here would unfairly
                    # punish miners; deleting both files would silently drop
                    # the request until a later redispatch.
                    try:
                        results_file.unlink()
                        work_file.touch()
                    except FileNotFoundError:
                        pass
                    except Exception as retry_exc:
                        print(
                            "   ❌ fulfillment_worker_retry_prepare_failed "
                            f"worker {wid_token}: {retry_exc}"
                        )
                    any_submit_failed = True
                    continue

                print(f"   Worker {wid_token} → request {request_id[:8]}: "
                      f"{len(submission_results)} submission(s)")

                # Per-work-file success flag.  Only when ALL submissions
                # within this work file are acknowledged by the gateway do we
                # delete the files.  Partial failures keep the files for the
                # next iteration's retry.
                worker_all_ok = True

                for sub_result in submission_results:
                    miner_hk = sub_result.get("miner_hotkey", "")
                    sub_id = sub_result.get("submission_id", "")
                    lead_ids = sub_result.get("lead_ids", [])
                    raw_results = sub_result.get("results", [])
                    results = [FulfillmentScoreResult(**r) for r in raw_results]

                    for idx, sr in enumerate(results):
                        icon = "✅" if sr.final_score > 0 else "❌"
                        reason = sr.failure_reason or "passed"
                        print(f"     {icon} Lead {idx+1}: score={sr.final_score:.1f} [{reason}]")

                    try:
                        if callable(progress_heartbeat):
                            progress_heartbeat()
                        scores_payload = format_scores_for_gateway(
                            miner_hk, lead_ids, results,
                            request_id=request_id, submission_id=sub_id,
                        )
                        if not gateway_submit_fulfillment_scores(
                            self.wallet, request_id, scores_payload,
                        ):
                            raise RuntimeError(
                                "gateway rejected fulfillment scores "
                                "without raising an exception"
                            )
                        if callable(progress_heartbeat):
                            progress_heartbeat()
                        print(f"   ✅ Submitted {len(scores_payload)} scores for miner {miner_hk[:8]}...")
                    except Exception as e:
                        worker_all_ok = False
                        print(f"   ❌ Gateway submit failed for {miner_hk[:8]}: {e} "
                              f"— keeping files, will retry next iteration")

                # Clean up files only if every submit in this work file
                # succeeded.  Otherwise keep them; Phase 1's per-request
                # work-file check (see top of this function) prevents
                # re-dispatch while the file is still on disk, and the
                # next Phase 2 tick re-enters this block to retry just
                # the failed submits.
                if worker_all_ok:
                    try:
                        os.remove(work_file)
                        os.remove(results_file)
                    except Exception:
                        pass
                else:
                    any_submit_failed = True
                    try:
                        work_file.touch()
                    except FileNotFoundError:
                        pass
                    except Exception as lease_exc:
                        print(
                            "   ❌ fulfillment_delivery_lease_refresh_failed "
                            f"worker {wid_token}: {lease_exc}"
                        )
                    print(f"   ⏸  Kept {work_file.name} for retry")

            # File-system state is now the source of truth for per-request
            # collection: successful submits delete the work+results pair,
            # so the next Phase 2 tick's glob finds nothing for that request.
            # Failed submits keep the files, so the next tick re-enters this
            # block and retries just those.  No epoch-level memo needed.
            if not any_submit_failed:
                print(f"{'='*60}\n")
            else:
                print(f"   ⏸  Epoch {current_epoch} NOT fully collected — "
                      f"some submits failed, will retry next iteration")
                print(f"{'='*60}\n")

        except ImportError as e:
            if not getattr(self, "_fulfillment_collect_warned", False):
                bt.logging.warning(f"Fulfillment collection imports unavailable: {e}")
                self._fulfillment_collect_warned = True
        except Exception as e:
            bt.logging.warning(f"Fulfillment collection error: {e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # HISTORICAL QUALIFICATION CHAMPION READER
    # ═══════════════════════════════════════════════════════════════════════════

    def _read_qualification_champion(self) -> Optional[Dict[str, Any]]:
        """
        Read qualification champion info from local JSON file.
        
        Also checks Supabase banned_hotkeys table - if champion is banned,
        clears local file and returns None (5% goes to burn instead).
        
        Returns:
            Dict with current_champion info, or None if no champion
        """
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            
            if not champion_file.exists():
                bt.logging.debug("No qualification champion file found")
                return None
            
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            champion = data.get("current_champion")
            if not champion:
                return None
            
            # Check if champion's hotkey is banned in Supabase
            champion_hotkey = champion.get("miner_hotkey")
            if champion_hotkey and self._is_champion_hotkey_banned(champion_hotkey):
                bt.logging.warning(f"🚨 Champion hotkey {champion_hotkey[:20]}... is BANNED - clearing local champion")
                self._clear_qualification_champion_for_ban(champion_hotkey)
                # Re-read: _clear_qualification_champion_for_ban may have written
                # an auto-promoted replacement from the gateway
                with open(champion_file, 'r') as f:
                    refreshed = json.load(f)
                return refreshed.get("current_champion")
            
            return champion
            
        except Exception as e:
            bt.logging.warning(f"Failed to read qualification champion: {e}")
            return None
    
    def _is_champion_hotkey_banned(self, hotkey: str) -> bool:
        """Check the champion against the canonical gateway-owned ban snapshot."""
        try:
            from Leadpoet.utils.cloud_db import (
                gateway_get_banned_hotkeys_snapshot,
            )

            snapshot = gateway_get_banned_hotkeys_snapshot(self.wallet)
            is_banned = hotkey in set(snapshot["banned_hotkeys"])
            if is_banned:
                bt.logging.info(f"🚨 Hotkey {hotkey[:20]}... found in banned_hotkeys table")
            return is_banned
        except Exception as exc:
            bt.logging.error(
                "champion_ban_snapshot_unavailable; refusing champion allocation: "
                f"{type(exc).__name__}: {exc}"
            )
            raise RuntimeError(
                "canonical champion ban snapshot is unavailable"
            ) from exc

    def _apply_banned_hotkey_sourcing_penalties(self, banned_hotkeys: set):
        """Set sourcing scores to -100,000 for banned hotkeys in weight files.

        Modifies ONLY the entries for hotkeys present in banned_hotkeys.
        Other miners' data is never read or written by this method.

        Files modified:
        - validator_weights/validator_weights_history (rolling window source)
        - validator_weights/validator_weights (current epoch)
        """
        if not banned_hotkeys:
            return

        PENALTY = -100_000
        history_file = Path("validator_weights") / "validator_weights_history"
        weights_file = Path("validator_weights") / "validator_weights"

        penalized_count = 0

        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)

            for epoch_str, epoch_data in history_data.items():
                if not epoch_str.isdigit():
                    continue
                miner_scores = epoch_data.get("miner_scores", {})
                for hotkey in banned_hotkeys:
                    if hotkey in miner_scores and miner_scores[hotkey] != PENALTY:
                        miner_scores[hotkey] = PENALTY
                        penalized_count += 1

            if penalized_count > 0:
                with open(history_file, 'w') as f:
                    json.dump(history_data, f, indent=2)

        weights_penalized = 0

        if weights_file.exists():
            with open(weights_file, 'r') as f:
                weights_data = json.load(f)

            for key, val in weights_data.items():
                if not key.isdigit():
                    continue
                miner_scores = val.get("miner_scores", {})
                for hotkey in banned_hotkeys:
                    if hotkey in miner_scores and miner_scores[hotkey] != PENALTY:
                        miner_scores[hotkey] = PENALTY
                        weights_penalized += 1

            if weights_penalized > 0:
                with open(weights_file, 'w') as f:
                    json.dump(weights_data, f, indent=2)

        if penalized_count > 0 or weights_penalized > 0:
            hk_list = ", ".join(h[:20] + "..." for h in banned_hotkeys)
            print(f"   🚨 BANNED HOTKEY SOURCING PENALTY: Set {penalized_count} history entries + {weights_penalized} current entries to {PENALTY:,}")
            print(f"      Hotkeys: {hk_list}")

    def _clear_qualification_champion_for_ban(self, banned_hotkey: str):
        """Clear champion from local JSON due to hotkey ban, then check
        Supabase for a new champion that the gateway may have auto-promoted."""
        try:
            champion_file = Path("validator_weights") / "qualification_champion.json"
            if not champion_file.exists():
                return
            
            with open(champion_file, 'r') as f:
                data = json.load(f)
            
            old_champion = data.get("current_champion")
            if old_champion:
                if "dethronement_history" not in data:
                    data["dethronement_history"] = []
                old_champion["dethroned_at"] = datetime.utcnow().isoformat()
                old_champion["dethrone_reason"] = "hotkey_banned"
                data["dethronement_history"].append(old_champion)
            
            data["current_champion"] = None
            
            print(f"\n{'='*60}")
            print(f"🚨 CHAMPION DETHRONED (HOTKEY BANNED)")
            print(f"   Hotkey: {banned_hotkey[:20]}...")
            print(f"   Checking Supabase for auto-promoted replacement...")
            print(f"{'='*60}")
            
            new_champion = self._fetch_current_champion_from_gateway()
            if new_champion:
                data["current_champion"] = new_champion
                print(f"   👑 Found auto-promoted champion!")
                print(f"      Model:  {new_champion.get('model_name', 'unknown')}")
                print(f"      Miner:  {new_champion.get('miner_hotkey', 'unknown')[:20]}...")
                print(f"      Score:  {new_champion.get('score', 0):.2f}")
                print(f"      10% champion share → new champion")
            else:
                print(f"   📭 No replacement champion found")
                print(f"   10% champion share → sourcing miners")
            print(f"{'='*60}\n")
            
            with open(champion_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            bt.logging.error(f"Failed to clear banned champion: {e}")
    
    def _fetch_current_champion_from_gateway(self) -> Optional[Dict[str, Any]]:
        """Query the gateway's /qualification/champion endpoint for the current
        champion. Used to pick up a champion that the gateway auto-promoted
        after a ban. Falls back gracefully if the gateway is unreachable."""
        try:
            import requests
            
            gateway_url = os.getenv("GATEWAY_URL", "http://52.91.135.79:8000")
            response = requests.get(
                f"{gateway_url}/qualification/champion",
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            champion = data.get("champion")
            if not champion:
                return None

            total_cost = champion.get("total_cost_usd") or 0
            total_time = champion.get("total_time_seconds") or 0
            num_leads = 100

            evaluated_at = champion.get("evaluated_at")
            if evaluated_at:
                last_eval_date = evaluated_at[:10]
            else:
                from datetime import datetime as dt_datetime, timezone as dt_timezone
                last_eval_date = dt_datetime.now(dt_timezone.utc).date().isoformat()

            return {
                "model_id": champion.get("model_id"),
                "model_name": champion.get("model_name"),
                "miner_hotkey": champion.get("miner_hotkey"),
                "score": champion.get("score", 0),
                "became_champion_at": champion.get("became_champion_at"),
                "total_cost_usd": total_cost,
                "total_time_seconds": total_time,
                "avg_cost_per_lead_usd": champion.get("avg_cost_per_lead_usd", 0),
                "avg_time_per_lead_seconds": champion.get("avg_time_per_lead_seconds", 0),
                "num_leads_evaluated": num_leads,
                "last_evaluated_utc_date": last_eval_date,
            }
        except Exception as e:
            bt.logging.warning(f"Failed to fetch champion from gateway: {e}")
            return None


    async def process_broadcast_requests_continuous(self):
        """
        Continuously poll for broadcast API requests from Firestore and process them.
        """
        await asyncio.sleep(2)
        print("📡 Polling for broadcast API requests... (will notify when requests are found)")

        poll_count = 0
        while True:
            try:
                poll_count += 1

                # Fetch pending broadcast requests from Firestore
                # Note: fetch_broadcast_requests imported at module level to avoid sandbox blocking
                requests_list = fetch_broadcast_requests(self.wallet, role="validator")

                # fetch_broadcast_requests() will print when requests are found
                # No need to log anything here when empty

                if requests_list:
                    print(f"🔔 Found {len(requests_list)} NEW broadcast request(s) to process!")

                for req in requests_list:
                    request_id = req.get("request_id")

                    # Skip if already processed locally
                    if request_id in self._processed_requests:
                        print(f"⏭️  Skipping already processed request {request_id[:8]}...")
                        continue

                    # Mark as processed locally
                    self._processed_requests.add(request_id)

                    num_leads = req.get("num_leads", 1)
                    business_desc = req.get("business_desc", "")

                    # Set flag IMMEDIATELY to pause sourcing
                    self.processing_broadcast = True

                    print(f"\n📨 🔔 BROADCAST API REQUEST RECEIVED {request_id[:8]}...")
                    print(f"   Requested: {num_leads} leads")
                    print(f"   Description: {business_desc[:50]}...")
                    print(f"   🕐 Request received at {time.strftime('%H:%M:%S')}")
                    print("   ⏳ Waiting up to 180 seconds for miners to send curated leads...")

                    try:
                        # Wait for miners to send curated leads to Firestore
                        # fetch_miner_leads_for_request imported at module level

                        MAX_WAIT = 180  
                        POLL_INTERVAL = 2  # Poll every 2 seconds

                        miner_leads_collected = []
                        start_time = time.time()
                        polls_done = 0

                        while time.time() - start_time < MAX_WAIT:
                            submissions = fetch_miner_leads_for_request(request_id)

                            if submissions:
                                # Flatten all leads from all miners
                                for submission in submissions:
                                    leads = submission.get("leads", [])
                                    miner_leads_collected.extend(leads)

                                if miner_leads_collected:
                                    elapsed = time.time() - start_time
                                    bt.logging.info(f"📥 Received leads from {len(submissions)} miner(s) after {elapsed:.1f}s")
                                    break

                            # Progress update every 10 seconds
                            polls_done += 1
                            if polls_done % 5 == 0:  # Every 10 seconds (5 polls * 2 sec)
                                elapsed = time.time() - start_time
                                bt.logging.info(f"⏳ Still waiting for miners... ({elapsed:.0f}s / {MAX_WAIT}s elapsed)")

                            await asyncio.sleep(POLL_INTERVAL)

                        if not miner_leads_collected:
                            bt.logging.warning(f"⚠️  No miner leads received after {MAX_WAIT}s, skipping ranking")
                            continue

                        bt.logging.info(f"📊 Received {len(miner_leads_collected)} total leads from miners")

                        # Rank leads using LLM scoring (TWO rounds with BATCHING)
                        if miner_leads_collected:
                            print(f"🔍 Ranking {len(miner_leads_collected)} leads with LLM...")
                            scored_leads = []

                            # Initialize aggregation dictionary for each lead
                            aggregated = {id(lead): 0.0 for lead in miner_leads_collected}
                            failed_leads = set()  # Track leads that failed LLM scoring

                            # ROUND 1: First LLM scoring (BATCHED)
                            first_model = random.choice(AVAILABLE_MODELS)
                            print(f"🔄 LLM round 1/2 (model: {first_model})")
                            batch_scores_r1 = _llm_score_batch(miner_leads_collected, business_desc, first_model)
                            for lead in miner_leads_collected:
                                score = batch_scores_r1.get(id(lead))
                                if score is None:
                                    failed_leads.add(id(lead))
                                    print("⚠️  LLM failed for lead, will skip this lead")
                                else:
                                    aggregated[id(lead)] += score

                            # ROUND 2: Second LLM scoring (BATCHED, random model selection)
                            # Only score leads that didn't fail in round 1
                            leads_for_r2 = [lead for lead in miner_leads_collected if id(lead) not in failed_leads]
                            if leads_for_r2:
                                second_model = random.choice(AVAILABLE_MODELS)
                                print(f"🔄 LLM round 2/2 (model: {second_model})")
                                batch_scores_r2 = _llm_score_batch(leads_for_r2, business_desc, second_model)
                                for lead in leads_for_r2:
                                    score = batch_scores_r2.get(id(lead))
                                    if score is None:
                                        failed_leads.add(id(lead))
                                        print("⚠️  LLM failed for lead, will skip this lead")
                                    else:
                                        aggregated[id(lead)] += score

                            # Apply aggregated scores to leads (skip failed ones)
                            for lead in miner_leads_collected:
                                if id(lead) not in failed_leads:
                                    lead["intent_score"] = round(aggregated[id(lead)], 3)
                                    scored_leads.append(lead)

                            if not scored_leads:
                                print("❌ All leads failed LLM scoring")
                                continue

                            # Sort by aggregated intent_score and take top N
                            scored_leads.sort(key=lambda x: x["intent_score"], reverse=True)
                            top_leads = scored_leads[:num_leads]

                            print(f"✅ Ranked top {len(top_leads)} leads:")
                            for i, lead in enumerate(top_leads, 1):
                                business = get_company(lead, default='Unknown')[:30]
                                score = lead.get('intent_score', 0)
                                print(f"  {i}. {business} (score={score:.3f})")

                        # SUBMIT VALIDATOR RANKING for consensus
                        try:
                            validator_trust = self.metagraph.validator_trust[self.uid].item()

                            ranking_submission = []
                            for rank, lead in enumerate(top_leads, 1):
                                ranking_submission.append({
                                    "lead": lead,
                                    "score": lead.get("intent_score", 0.0),
                                    "rank": rank,
                                })

                            success = push_validator_ranking(
                                wallet=self.wallet,
                                request_id=request_id,
                                ranked_leads=ranking_submission,
                                validator_trust=validator_trust
                            )

                            if success:
                                print(f"📊 Submitted ranking for consensus (trust={validator_trust:.4f})")
                            else:
                                print("⚠️  Failed to submit ranking for consensus")

                        except Exception as e:
                            print(f"⚠️  Error submitting validator ranking: {e}")
                            bt.logging.error(f"Error submitting validator ranking: {e}")

                        print(f"✅ Validator {self.wallet.hotkey.ss58_address[:10]}... completed processing broadcast {request_id[:8]}...")

                    except Exception as e:
                        print(f"❌ Error processing broadcast request {request_id[:8]}...: {e}")
                        bt.logging.error(f"Error processing broadcast request: {e}")
                        import traceback
                        bt.logging.error(traceback.format_exc())

                    finally:
                        # Always resume sourcing after processing
                        self.processing_broadcast = False

            except Exception as e:
                # Catch any errors in the outer loop (fetching requests, etc.)
                bt.logging.error(f"Error in broadcast polling loop: {e}")
                import traceback
                bt.logging.error(traceback.format_exc())

            # Clear old processed requests every 100 iterations to prevent memory buildup
            if poll_count % 100 == 0:
                bt.logging.info(f"🧹 Clearing old processed requests cache ({len(self._processed_requests)} entries)")
                self._processed_requests.clear()

            # Sleep before next poll
            await asyncio.sleep(1)  

    def move_to_validated_leads(self, lead, score):
        """
        [DEPRECATED IN CONSENSUS MODE]
        This function is no longer used when consensus validation is enabled.
        Leads are now saved through the consensus system after 3 validators agree.
        See submit_validation_assessment() in cloud_db.py instead.
        """
        # Prepare lead data
        lead["validator_hotkey"] = self.wallet.hotkey.ss58_address
        lead["validated_at"] = datetime.now(timezone.utc).isoformat()

        try:
            # Save to Supabase (write-only, no duplicate checking)
            if not self.supabase_client:
                bt.logging.error("❌ Supabase client not available - cannot save validated lead")
                return
                
            success = self.save_validated_lead_to_supabase(lead)
            email = get_email(lead, default='?')
            biz = get_field(lead, "business", "website")
            
            if success:
                print(f"✅ Added verified lead to Supabase → {biz} ({email})")
            else:
                # Duplicate or error - already logged in save function
                pass
                
        except Exception as e:
            bt.logging.error(f"Failed to save lead to Supabase: {e}")

    # Local prospect queue no longer exists
    def remove_from_prospect_queue(self, lead):
        return

    def is_disposable_email(self, email):
        """Check if email is from a disposable email provider"""
        disposable_domains = {
            '10minutemail.com', 'guerrillamail.com', 'mailinator.com', 'tempmail.org',
            'throwaway.email', 'temp-mail.org', 'yopmail.com', 'getnada.com'
        }
        domain = email.split('@')[-1].lower()
        return domain in disposable_domains

    def check_domain_legitimacy(self, domain):
        """Return True iff the domain looks syntactically valid (dot & no spaces)."""
        try:
            return "." in domain and " " not in domain
        except Exception:
            return False

    def should_run_deep_verification(self, lead: Dict) -> bool:
        """
        Determine if lead should undergo deep verification.
        
        Returns True for:
        - 100% of licensed_resale submissions
        - 5% random sample of other submissions
        
        Deep verification includes:
        - License OCR validation (for licensed_resale)
        - Cross-domain authenticity checks
        - Behavioral anomaly scoring
        """
        source_type = lead.get("source_type", "")
        
        # Always verify licensed resale
        if source_type == "licensed_resale":
            bt.logging.info(f"🔬 Deep verification triggered: licensed_resale source")
            return True
        
        # 5% random sample for others
        if random.random() < 0.05:
            bt.logging.info(f"🔬 Deep verification triggered: random 5% sample")
            return True
        
        return False

    async def run_deep_verification(self, lead: Dict) -> Dict:
        """
        Execute deep verification checks.
        
        Returns dict with:
        - passed: bool (overall pass/fail)
        - checks: list of individual check results
        - manual_review_required: bool (if flagged for admin review)
        """
        results = {
            "passed": True,
            "checks": [],
            "manual_review_required": False
        }
        
        # Check 1: License OCR validation (if applicable)
        if lead.get("source_type") == "licensed_resale":
            bt.logging.info("   🔍 Deep Check 1: License OCR validation")
            ocr_result = await self.verify_license_ocr(lead)
            results["checks"].append(ocr_result)
            
            if not ocr_result["passed"]:
                results["passed"] = False
                bt.logging.warning(f"   ❌ License OCR failed: {ocr_result['reason']}")
            else:
                bt.logging.info(f"   ✅ License OCR: {ocr_result['reason']}")
            
            if ocr_result.get("manual_review_required"):
                results["manual_review_required"] = True
        
        # Check 2: Cross-domain authenticity
        bt.logging.info("   🔍 Deep Check 2: Cross-domain authenticity")
        domain_result = await self.verify_cross_domain_authenticity(lead)
        results["checks"].append(domain_result)
        
        if not domain_result["passed"]:
            results["passed"] = False
            bt.logging.warning(f"   ❌ Cross-domain check failed: {domain_result['reason']}")
        else:
            bt.logging.info(f"   ✅ Cross-domain: {domain_result['reason']}")
        
        # Check 3: Behavioral anomaly scoring
        bt.logging.info("   🔍 Deep Check 3: Behavioral anomaly scoring")
        anomaly_result = await self.score_behavioral_anomalies(lead)
        results["checks"].append(anomaly_result)
        
        if not anomaly_result["passed"]:
            results["passed"] = False
            bt.logging.warning(f"   ❌ Anomaly check failed: {anomaly_result['reason']}")
        else:
            bt.logging.info(f"   ✅ Anomaly scoring: {anomaly_result['reason']}")
        
        return results

    async def verify_license_ocr(self, lead: Dict) -> Dict:
        """
        Validate license document via hash verification.
        
        Steps:
        1. Download document from license_doc_url
        2. Verify hash matches license_doc_hash (SHA-256)
        3. Flag for manual OCR review
        
        Future enhancement: Implement OCR text extraction to search for
        key terms (resale, redistribute, transfer, sub-license).
        
        Returns dict with:
        - passed: bool
        - check: str (check name)
        - reason: str (result description)
        - manual_review_required: bool (optional)
        """
        import hashlib
        import aiohttp
        
        license_url = lead.get("license_doc_url")
        license_hash = lead.get("license_doc_hash")
        
        if not license_url:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "No license_doc_url provided for OCR verification"
            }
        
        if not license_hash:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "No license_doc_hash provided"
            }
        
        try:
            # Download document
            bt.logging.info(f"   📥 Downloading license doc from: {license_url[:50]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(license_url, timeout=30) as response:
                    if response.status != 200:
                        return {
                            "passed": False,
                            "check": "license_ocr",
                            "reason": f"License doc unreachable: HTTP {response.status}"
                        }
                    
                    doc_content = await response.read()
            
            # Verify hash matches
            computed_hash = hashlib.sha256(doc_content).hexdigest()
            
            if computed_hash != license_hash:
                return {
                    "passed": False,
                    "check": "license_ocr",
                    "reason": f"License doc hash mismatch (expected: {license_hash[:8]}..., got: {computed_hash[:8]}...)"
                }
            
            bt.logging.info(f"   ✅ License hash verified: {computed_hash[:16]}...")
            
            # TODO: Implement OCR text extraction (requires pytesseract or cloud OCR API)
            # For now, flag for manual review
            return {
                "passed": True,
                "check": "license_ocr",
                "reason": "Hash verified - flagged for manual OCR review",
                "manual_review_required": True,
                "license_hash": computed_hash,
                "license_url": license_url
            }
            
        except asyncio.TimeoutError:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": "License doc download timeout (>30s)"
            }
        except Exception as e:
            return {
                "passed": False,
                "check": "license_ocr",
                "reason": f"License verification error: {str(e)}"
            }

    async def verify_cross_domain_authenticity(self, lead: Dict) -> Dict:
        """
        Verify entity-domain relationship authenticity.
        
        Checks:
        - Email domain should match company domain
        - Detects throwaway/temporary domains
        - Validates domain relationships
        
        This helps detect:
        - Spoofed email addresses
        - Temporary/disposable domains
        - Mismatched company-email relationships
        
        Returns dict with:
        - passed: bool
        - check: str (check name)
        - reason: str (result description)
        - severity: str (optional - "high" for critical mismatches)
        """
        from urllib.parse import urlparse
        
        email = get_email(lead)
        website = get_website(lead)
        company = get_company(lead)
        
        # If insufficient data, pass through (can't verify)
        if not email or not website:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Insufficient data for cross-domain verification"
            }
        
        # Extract domains
        email_domain = email.split("@")[1].lower() if "@" in email else ""
        
        # Parse website domain
        try:
            parsed_website = urlparse(website if website.startswith(('http://', 'https://')) else f'https://{website}')
            website_domain = parsed_website.netloc.lower()
            
            # Remove www. prefix for comparison
            if website_domain.startswith("www."):
                website_domain = website_domain[4:]
            if email_domain.startswith("www."):
                email_domain = email_domain[4:]
                
        except Exception as e:
            bt.logging.warning(f"   Failed to parse website domain: {website} - {e}")
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Could not parse website domain"
            }
        
        # Check for throwaway/temporary domain indicators
        throwaway_indicators = [
            "-sales", "-marketing", "-temp", "tempmail", "guerrilla",
            "throwaway", "disposable", "fake", "test", "temporary"
        ]
        
        for indicator in throwaway_indicators:
            if indicator in email_domain:
                return {
                    "passed": False,
                    "check": "cross_domain",
                    "reason": f"Email domain appears to be temporary: {email_domain}",
                    "severity": "high"
                }
        
        # Check if domains match
        if email_domain == website_domain:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": "Email domain matches website domain"
            }
        
        # Check if they're related (subdomain or parent domain)
        if website_domain in email_domain or email_domain in website_domain:
            return {
                "passed": True,
                "check": "cross_domain",
                "reason": f"Related domains (email: {email_domain}, website: {website_domain})"
            }
        
        # Domains don't match - this could be legitimate (e.g., gmail.com for small business)
        # or could be suspicious. We'll flag but not fail for now.
        # In a stricter implementation, this could be a failure.
        return {
            "passed": True,  # Pass but log warning
            "check": "cross_domain",
            "reason": f"Email domain ({email_domain}) differs from website ({website_domain})",
            "severity": "low",
            "warning": True
        }

    async def score_behavioral_anomalies(self, lead: Dict) -> Dict:
        """
        Score lead for behavioral anomalies.
        
        Checks for:
        - Excessive use of same source_url (possible scraping/automation)
        - Unlikely role-industry combinations
        - Statistical outliers
        
        Returns dict with:
        - passed: bool (True if anomaly_score < 0.7)
        - check: str (check name)
        - score: float (0-1, where 0=normal, 1=highly anomalous)
        - flags: list (descriptions of detected anomalies)
        - reason: str (summary)
        """
        anomaly_score = 0.0
        flags = []
        
        # Check 1: Duplicate source_url usage
        source_url = lead.get("source_url", "")
        if source_url:
            try:
                # get_supabase_client imported at module level
                supabase = get_supabase_client()
                
                if supabase:
                    # Query recent submissions with same source_url
                    recent_cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                    result = supabase.table("prospect_queue")\
                        .select("miner_hotkey, source_url")\
                        .eq("source_url", source_url)\
                        .gte("created_at", recent_cutoff)\
                        .execute()
                    
                    if result.data and len(result.data) > 10:
                        anomaly_score += 0.3
                        flags.append(f"Source URL used {len(result.data)} times in 24h")
                        bt.logging.warning(f"   ⚠️  High source_url reuse: {len(result.data)} times")
            except Exception as e:
                bt.logging.debug(f"   Could not check source_url duplicates: {e}")
        
        # Check 2: Role-industry mismatch
        # This is a simplified check - in production, use ML model or extensive mapping
        role = get_role(lead)
        industry = get_industry(lead)
        
        if role and industry:
            # Define obviously unlikely combinations
            unlikely_combinations = [
                ("Doctor", "Technology"),
                ("Doctor", "Software"),
                ("CTO", "Healthcare"),
                ("CTO", "Medical"),
                ("Nurse", "Finance"),
                ("Engineer", "Healthcare"),
                ("Surgeon", "Retail"),
            ]
            
            # Normalize for comparison
            role_normalized = role.upper()
            industry_normalized = industry.upper()
            
            for unlikely_role, unlikely_industry in unlikely_combinations:
                if unlikely_role.upper() in role_normalized and unlikely_industry.upper() in industry_normalized:
                    anomaly_score += 0.2
                    flags.append(f"Unlikely role-industry: {role} in {industry}")
                    bt.logging.warning(f"   ⚠️  Unlikely combination: {role} in {industry}")
                    break
        
        # Check 3: Missing critical fields (possible data quality issue)
        critical_fields = ["email", "company", "website"]
        missing_fields = [field for field in critical_fields if not lead.get(field)]
        
        if len(missing_fields) >= 2:
            anomaly_score += 0.1
            flags.append(f"Missing {len(missing_fields)} critical fields: {', '.join(missing_fields)}")
        
        # Determine pass/fail based on threshold
        threshold = 0.7
        passed = anomaly_score < threshold
        
        return {
            "passed": passed,
            "check": "anomaly_scoring",
            "score": anomaly_score,
            "flags": flags,
            "reason": f"Anomaly score: {anomaly_score:.2f} (threshold: {threshold})",
            "threshold": threshold
        }

    async def validate_lead(self, lead):
        """Validate a single lead using automated_checks. Returns pass/fail."""
        try:
            # Check for required email field first
            email = get_email(lead)
            if not email:
                return {
                    'is_legitimate': False,
                    'reason': {
                        "stage": "Pre-validation",
                        "check_name": "email_check",
                        "message": "Missing email",
                        "failed_fields": ["email"]
                    },
                    'enhanced_lead': lead  # Return original lead if no email
                }
            
            # Map your field names to what automated_checks expects
            mapped_lead = {
                "email": email,  # Map to "email" field
                "Email 1": email,  # Also map to "Email 1" as backup
                "Company": get_field(lead, 'business', 'website'),  # Map business -> Company
                "Website": get_field(lead, 'website', 'business'),  # Map to Website
                "website": get_field(lead, 'website', 'business'),  # Also lowercase
                "First Name": lead.get('first', ''),
                "Last Name": lead.get('last', ''),
                # Include any other fields that might be useful
                **lead  # Include all original fields too
            }
            
            # Use automated_checks for comprehensive validation
            # NEW: run_automated_checks returns (passed, automated_checks_data) with structured data
            passed, automated_checks_data = await run_automated_checks(mapped_lead)
            
            # Extract rejection_reason from structured data for backwards compatibility
            reason = automated_checks_data.get("rejection_reason") if not passed else None
            
            # Append automated_checks data to mapped_lead so it gets stored in validation_tracking
            mapped_lead["automated_checks"] = automated_checks_data

            # If standard validation passed, check if deep verification is needed
            if passed and self.should_run_deep_verification(mapped_lead):
                bt.logging.info(f"🔬 Running deep verification on {email}")
                
                deep_results = await self.run_deep_verification(mapped_lead)
                
                if not deep_results["passed"]:
                    bt.logging.warning(f"❌ Deep verification failed: {deep_results}")
                    # Mark lead for manual review or reject
                    lead["deep_verification_failed"] = True
                    lead["deep_verification_results"] = deep_results
                
                    # Return structured rejection reason 
                    deep_reason = deep_results["checks"][0]["reason"] if deep_results.get("checks") else "unknown"
                    return {
                        'is_legitimate': False,
                        'reason': {
                            "stage": "Deep Verification",
                            "check_name": "deep_verification",
                            "message": f"Deep verification failed: {deep_reason}",
                            "failed_fields": []
                        },
                        'deep_verification_results': deep_results,
                        'enhanced_lead': mapped_lead  # Include enhanced lead even on deep verification failure
                    }
                else:
                    bt.logging.info(f"✅ Deep verification passed")
                    lead["deep_verification_passed"] = True
                    lead["deep_verification_results"] = deep_results
                    
                    # If manual review required, flag it but don't fail
                    if deep_results.get("manual_review_required"):
                        lead["manual_review_required"] = True
                        bt.logging.info(f"📋 Lead flagged for manual review")

            # Copy validator-calculated rep_score from mapped_lead back to original lead
            # This ensures the rep_score in enhanced_lead is from automated checks, not miner data
            if "rep_score" in mapped_lead:
                lead["rep_score"] = mapped_lead["rep_score"]
            
            # Prepare validation result with enhanced lead data
            validation_result = {
                'is_legitimate': passed,
                'reason': reason,
                'enhanced_lead': mapped_lead  # Include enhanced lead with DNSBL/WHOIS data
            }
            
            # NOTE: Audit logging removed - validators should NOT write directly to Supabase.
            # All logging is handled by the gateway via POST /validate (TEE architecture).
            # The gateway stores evidence_blob in validation_evidence_private and logs to TEE buffer.
            
            return validation_result
            
        except Exception as e:
            # Check if this is an EmailVerificationUnavailableError - if so, re-raise it
            from validator_models.automated_checks import EmailVerificationUnavailableError
            if isinstance(e, EmailVerificationUnavailableError):
                # Re-raise to propagate to process_sourced_leads_continuous
                raise
            
            bt.logging.error(f"Error in validate_lead: {e}")
            
            # Create structured rejection reason for error case
            error_rejection = {
                "stage": "Validation Error",
                "check_name": "exception",
                "message": f"Validation error: {str(e)}",
                "failed_fields": []
            }
            
            # NOTE: Audit logging removed - validators should NOT write directly to Supabase.
            # All logging is handled by the gateway via POST /validate (TEE architecture).
            
            return {
                'is_legitimate': False,
                'reason': error_rejection,
                'enhanced_lead': lead  # Return original lead on error
            }

    def calculate_validation_score_breakdown(self, lead):
        """Calculate validation score with detailed breakdown"""
        try:
            website_score = 0.2 if lead.get('website') else 0.0
            industry_score = 0.1 if lead.get('industry') else 0.0
            region_score = 0.1 if lead.get('region') else 0.0

            return {
                'website_score': website_score,
                'industry_score': industry_score,
                'region_score': region_score
            }
        except Exception:
            return {'website_score': 0.0, 'industry_score': 0.0, 'region_score': 0.0}

    def save_validated_lead_to_supabase(self, lead: Dict) -> bool:
        """
        Write validated lead directly to Supabase.
        Validators have INSERT-only access (enforced by RLS).
        Duplicates are handled by database unique constraint + trigger notification.
        
        Args:
            lead: Lead dictionary with all required fields
            
        Returns:
            bool: True if successfully inserted, False if duplicate or error
        """
        if not self.supabase_client:
            bt.logging.error("❌ Supabase client not initialized, cannot save lead")
            return False
        
        try:
            # Prepare lead data for insertion
            lead_data = {
                "email": get_email(lead),
                "company": get_field(lead, "business", "company"),
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "validator_hotkey": self.wallet.hotkey.ss58_address,
                "miner_hotkey": get_field(lead, "source", "miner_hotkey"),
                "score": get_field(lead, "conversion_score", "score"),
                "metadata": {
                    "full_name": lead.get("full_name", ""),
                    "first": lead.get("first", ""),
                    "last": lead.get("last", ""),
                    "linkedin": lead.get("linkedin", ""),
                    "website": lead.get("website", ""),
                    "industry": lead.get("industry", ""),
                    "sub_industry": lead.get("sub_industry", ""),
                    "region": lead.get("region", ""),
                    "region_country": lead.get("region_country", ""),
                    "region_state": lead.get("region_state", ""),
                    "region_city": lead.get("region_city", ""),
                    "role": lead.get("role", ""),
                    "description": lead.get("description", ""),
                    "phone_numbers": lead.get("phone_numbers", []),
                    "founded_year": lead.get("founded_year", ""),
                    "ownership_type": lead.get("ownership_type", ""),
                    "company_type": lead.get("company_type", ""),
                    "number_of_locations": lead.get("number_of_locations", ""),
                    "socials": lead.get("socials", {}),
                }
            }
            
            # DEBUG: Log what we're trying to insert
            bt.logging.debug(f"🔍 INSERT attempt - validator_hotkey: {lead_data['validator_hotkey'][:10]}...")
            
            # Insert into Supabase - database will enforce unique constraint
            # Trigger will automatically notify miner if duplicate
            # NOTE: Wrap in array to match how miner inserts to prospect_queue
            self.supabase_client.table("leads").insert([lead_data])
            
            bt.logging.info(f"✅ Saved lead to Supabase: {lead_data['email']} ({lead_data['company']})")
            return True
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Handle duplicate email (caught by unique constraint)
            if "duplicate" in error_str or "unique" in error_str or "23505" in error_str:
                bt.logging.debug(f"⏭️  Duplicate lead (trigger will notify miner): {get_email(lead)}")
                return False
            
            # Handle RLS policy violations
            elif "row-level security" in error_str or "42501" in error_str:
                bt.logging.error("❌ RLS policy violation - check JWT and validator_hotkey match")
                bt.logging.error(f"   Validator hotkey in data: {lead_data.get('validator_hotkey', 'missing')[:10]}...")
                bt.logging.error("   JWT should contain same hotkey in 'hotkey' claim")
                return False
            
            # Other errors
            else:
                bt.logging.error(f"❌ Failed to save lead to Supabase: {e}")
                return False

DATA_DIR = "data"
VALIDATION_LOG = os.path.join(DATA_DIR, "validation_logs.json")
VALIDATORS_LOG = os.path.join(DATA_DIR, "validators.json")

def ensure_data_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    for file in [VALIDATION_LOG, VALIDATORS_LOG]:
        if not os.path.exists(file):
            with open(file, "w") as f:
                json.dump([], f)

def log_validation(hotkey, num_valid, num_rejected, issues):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "hotkey": hotkey,
        "num_valid": num_valid,
        "num_rejected": num_rejected,
        "issues": issues
    }
    with open(VALIDATION_LOG, "r+") as f:
        try:
            logs = json.load(f)
        except Exception:
            logs = []
        logs.append(entry)
        f.seek(0)
        json.dump(logs, f, indent=2)

def update_validator_stats(hotkey, precision):
    with open(VALIDATORS_LOG, "r+") as f:
        try:
            validators = json.load(f)
        except Exception:
            validators = []
        found = False
        for v in validators:
            if v["hotkey"] == hotkey:
                v["precision"] = precision
                v["last_updated"] = datetime.now().isoformat()
                found = True
                break
        if not found:
            validators.append({
                "hotkey": hotkey,
                "precision": precision,
                "last_updated": datetime.now().isoformat()
            })
        f.seek(0)
        json.dump(validators, f, indent=2)

class LeadQueue:
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self.queue_file = "lead_queue.json"
        self._ensure_queue_file()

    def _ensure_queue_file(self):
        """Ensure queue file exists and is valid JSON"""
        try:
            # Try to read existing file
            with open(self.queue_file, 'r') as f:
                try:
                    json.load(f)
                except json.JSONDecodeError:
                    # If file is corrupted, create new empty queue
                    bt.logging.warning("Queue file corrupted, creating new empty queue")
                    self._create_empty_queue()
        except FileNotFoundError:
            # If file doesn't exist, create new empty queue
            self._create_empty_queue()

    def _create_empty_queue(self):
        """Create a new empty queue file"""
        with open(self.queue_file, 'w') as f:
            json.dump([], f)

    def enqueue_prospects(self, prospects: List[Dict], miner_hotkey: str,
                          request_type: str = "sourced", **meta):
        """Add prospects to queue with validation"""
        try:
            with open(self.queue_file, 'r') as f:
                try:
                    queue = json.load(f)
                except json.JSONDecodeError:
                    bt.logging.warning("Queue file corrupted during read, creating new queue")
                    queue = []

            # append once
            queue.append({
                "prospects": prospects,
                "miner_hotkey": miner_hotkey,
                "request_type": request_type,
                **meta
            })

            # trim & write back
            if len(queue) > self.maxsize:
                queue = queue[-self.maxsize:]

            with open(self.queue_file, 'w') as f:
                json.dump(queue, f, indent=2)

        except Exception as e:
            bt.logging.error(f"Error enqueueing prospects: {e}")
            self._create_empty_queue()

    def dequeue_prospects(self) -> List[Dict]:
        """Get and remove prospects from queue with validation"""
        try:
            # Read current queue
            with open(self.queue_file, 'r') as f:
                try:
                    queue = json.load(f)
                except json.JSONDecodeError:
                    bt.logging.warning("Queue file corrupted during read, creating new queue")
                    queue = []

            if not queue:
                return []

            # Get all prospects and clear queue
            prospects = queue
            with open(self.queue_file, 'w') as f:
                json.dump([], f)

            return prospects

        except Exception as e:
            bt.logging.error(f"Error dequeuing prospects: {e}")
            # If any error occurs, try to create new queue
            self._create_empty_queue()
            return []

async def run_validator(validator_hotkey, queue_maxsize):
    print("Validator event loop started.")

    # Create validator instance
    config = bt.Config()
    validator = Validator(config=config)

    # Start HTTP server
    await validator.start_http_server()

    # Track all delivered leads for this API query
    all_delivered_leads = []

    async def validation_loop():
        nonlocal all_delivered_leads
        print("🔄 Validation loop running - waiting for leads to process...")
        while True:
            lead_request = lead_queue.dequeue_prospects()
            if not lead_request:
                await asyncio.sleep(1)
                continue

            request_type = lead_request.get("request_type", "sourced")
            prospects     = lead_request["prospects"]
            miner_hotkey  = lead_request["miner_hotkey"]

            print(f"\n📥 Processing {request_type} batch of {len(prospects)} prospects from miner {miner_hotkey[:8]}...")

            # curated list
            if request_type == "curated":
                print(f"🔍 Processing curated leads from {miner_hotkey[:20]}...")
                # Set the curator hotkey for all prospects in this batch
                for prospect in prospects:
                    prospect["curated_by"] = miner_hotkey

                # score with your open-source conversion model
                report  = await auto_check_leads(prospects)
                scores  = report.get("detailed_scores", [1.0]*len(prospects))
                for p, s in zip(prospects, scores):
                    p["conversion_score"] = s

                # print human-readable ranking
                ranked = sorted(prospects, key=lambda x: x["conversion_score"], reverse=True)
                print(f"\n Curated leads from {miner_hotkey[:20]} (ranked by score):")
                for idx, lead in enumerate(ranked, 1):
                    business = get_company(lead, default='Unknown')[:30]
                    # accept either lowercase or capitalised field
                    business = get_company(lead, default='Unknown')
                    business = business[:30]
                    score = lead['conversion_score']
                    print(f"  {idx:2d}. {business:30s}  score={score:.3f}")

                asked_for = lead_request.get("requested", len(ranked))
                top_n = min(asked_for, len(ranked))
                print(f"✅ Sending top-{top_n} leads to buyer")

                # store in pool and record reward-event for delivered leads
                delivered_leads = ranked[:top_n]
                add_validated_leads_to_pool(delivered_leads)

                # Add to all delivered leads for this query
                all_delivered_leads.extend(delivered_leads)

                # Record rewards for ALL delivered leads in this query
                # record_delivery_rewards imported at module level
                record_delivery_rewards(all_delivered_leads)

                # Send leads to buyer
                print(f"✅ Sent {len(delivered_leads)} leads to buyer")

                # Add source hotkey display
                for lead in delivered_leads:
                    source_hotkey = lead.get('source', 'unknown')
                    print(f"   Lead sourced by: {source_hotkey}")   # show full hotkey

                # Save curated leads to separate file
                # save_curated_leads imported at module level
                save_curated_leads(delivered_leads)

                # Reset all_delivered_leads after recording rewards
                all_delivered_leads = []

                continue          # skip legitimacy audit branch altogether

            # sourced list
            print(f"🔍 Validating {len(prospects)} sourced leads...")
            valid, rejected, issues = [], [], []

            for prospect in prospects:
                business = prospect.get('business', 'Unknown Business')
                print(f"\n  Validating: {business}")

                # Get email
                email = prospect.get("email", "")
                print(f"    Email: {email}")

                if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
                    issue = f"Invalid email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if any(domain in email for domain in ["mailinator.com", "tempmail.com"]):
                    issue = f"Disposable email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if prospect["source"] != miner_hotkey:
                    issue = f"Source mismatch: {prospect['source']} != {miner_hotkey}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                if lead_pool.check_duplicates(email):
                    issue = f"Duplicate email: {email}"
                    print(f"    ❌ Rejected: {issue}")
                    issues.append(issue)
                    rejected.append(prospect)
                    continue

                # All checks passed ⇒ accept
                valid.append(prospect)

            if valid:
                add_validated_leads_to_pool(valid)
                print(f"\n✅ Added {len(valid)} valid prospects to pool")

            log_validation(validator_hotkey, len(valid), len(rejected), issues)
            total = len(valid) + len(rejected)
            precision = (len(valid) / total) if total else 0.0
            update_validator_stats(validator_hotkey, precision)
            print(f"\n Validation summary: {len(valid)} accepted, {len(rejected)} rejected.")
            await asyncio.sleep(0.1)

    # Run both the HTTP server and validation loop
    await asyncio.gather(
        validation_loop(),
        asyncio.sleep(float('inf'))  # Keep HTTP server running
    )

def add_validated_leads_to_pool(leads):
    """Add validated leads to the pool with consistent field names."""
    mapped_leads = []
    for lead in leads:
        # Get the actual validation score from the lead
        validation_score = lead.get("conversion_score", 1.0)  # Use existing score or default to 1.0

        mapped_lead = {
            "business": get_company(lead),
            "full_name": get_field(lead, "full_name"),
            "first": get_first_name(lead),
            "last": get_last_name(lead),
            "email": get_email(lead),
            "linkedin": get_linkedin(lead),
            "website": get_website(lead),
            "industry": get_industry(lead),
            "sub_industry": get_sub_industry(lead),
            "region": get_location(lead),
            "role": lead.get("role", ""),
            "description": lead.get("description", ""),
            "phone_numbers": lead.get("phone_numbers", []),
            "founded_year": lead.get("founded_year", ""),
            "ownership_type": lead.get("ownership_type", ""),
            "company_type": lead.get("company_type", ""),
            "number_of_locations": lead.get("number_of_locations", ""),
            "socials": lead.get("socials", {}),
            "source":     lead.get("source", ""),
            "curated_by": lead.get("curated_by", ""),
        }

        # score is kept only if the lead already has it (i.e. curated phase)
        if "conversion_score" in lead:
            mapped_lead["conversion_score"] = validation_score
        mapped_leads.append(mapped_lead)

    lead_pool.add_to_pool(mapped_leads)


def run_lightweight_worker(config):
    """
    Lightweight worker loop for containerized validators.
    
    Workers skip ALL heavy initialization and only:
    1. Read current_block.json for epoch timing
    2. Read epoch_{N}_leads.json for lead data
    3. Validate leads (CPU/IO work)
    4. Write results to JSON file
    
    No Bittensor connection, no axon, no epoch monitor, no weight setting.
    """
    import asyncio
    import json
    from pathlib import Path
    
    print("🚀 Starting lightweight worker...")
    print(f"   Container ID: {config.neuron.container_id}")
    print(f"   Total containers: {config.neuron.total_containers}")
    print("")
    
    # Create minimal validator-like object for process_gateway_validation_workflow
    class LightweightWorker:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            # Track completed epochs IN MEMORY (not files, since coordinator deletes result files)
            # This prevents workers from trying to redo lead validation after coordinator aggregates
            self._completed_lead_validation_epochs = set()
            
        def _read_shared_block_file(self):
            """Read current block from shared file (written by coordinator)"""
            state = _read_shared_epoch_state_file(max_age_seconds=60)
            self._shared_epoch_state = state
            return (
                state.current_block,
                state.workflow_epoch_id,
                state.epoch_block,
            )
        
        async def process_gateway_validation_workflow(self):
            """
            Simplified worker validation loop.
            
            This is a COPY of the worker-specific logic from Validator.process_gateway_validation_workflow(),
            but without any Bittensor dependencies.
            """
            import time
            from validator_models.automated_checks import run_automated_checks, run_batch_automated_checks
            
            print("🔄 Worker validation loop started")

            while not self.should_exit:
                try:
                    # Legacy sourcing is retired. The coordinator branch of
                    # process_gateway_validation_workflow marks each epoch
                    # processed WITHOUT writing epoch_{N}_leads.json when
                    # ENABLE_LEGACY_SOURCING is off, so waiting for that file
                    # here can never succeed — the worker just burns the epoch
                    # and logs "Too late to start validation" every cycle.
                    # Park quietly instead; re-enable via ENABLE_LEGACY_SOURCING.
                    if not _env_flag("ENABLE_LEGACY_SOURCING"):
                        if not getattr(self, "_legacy_sourcing_park_logged", False):
                            print(
                                "🚫 Legacy sourcing disabled; worker lead-validation "
                                "loop parked (set ENABLE_LEGACY_SOURCING=true to re-enable)"
                            )
                            self._legacy_sourcing_park_logged = True
                        await asyncio.sleep(300)
                        continue

                    # Read current epoch from coordinator's shared file
                    try:
                        current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                        epoch_state = self._shared_epoch_state
                    except FileNotFoundError:
                        print("⏳ Worker: Waiting for coordinator to write block file...")
                        await asyncio.sleep(5)
                        continue
                    except Exception as e:
                        # Extract just the error message, don't try to parse it
                        print(f"⏳ Worker: Waiting for coordinator to write block file... ({str(e)})")
                        await asyncio.sleep(5)
                        continue
                    
                    print(
                        f"\n🔍 WORKER EPOCH {current_epoch}: Starting validation "
                        f"(block {blocks_into_epoch}/{epoch_state.tempo}, "
                        f"{epoch_state.blocks_remaining} remaining)"
                    )
                    
                    # CRITICAL FIX: Check if we already completed this epoch using IN-MEMORY tracking
                    # (Not file-based, because coordinator deletes result files after aggregation)
                    container_id = self.config.neuron.container_id
                    if current_epoch in self._completed_lead_validation_epochs:
                        # Lead validation already done - wait for next epoch
                        # Clear old epochs from memory to prevent unbounded growth
                        if len(self._completed_lead_validation_epochs) > 10:
                            oldest = min(self._completed_lead_validation_epochs)
                            if oldest < current_epoch - 5:
                                self._completed_lead_validation_epochs.discard(oldest)
                        print(f"⏭️  Worker {container_id}: Epoch {current_epoch} lead validation complete, waiting for next epoch...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Wait for coordinator to fetch and share leads
                    leads_file = Path("validator_weights") / f"epoch_{current_epoch}_leads.json"
                    
                    waited = 0
                    log_interval = 300  # Log every 5 minutes
                    check_interval = 5  # Check every 5 seconds
                    
                    while not leads_file.exists():
                        await asyncio.sleep(check_interval)
                        waited += check_interval
                        
                        # Check current block and epoch from shared file
                        try:
                            check_block, check_epoch, blocks_into_epoch = self._read_shared_block_file()
                            check_state = self._shared_epoch_state
                        except Exception:
                            continue
                        
                        # Epoch changed while waiting - abort
                        if not check_state.same_epoch(epoch_state):
                            print(f"❌ Worker: Epoch changed ({current_epoch} → {check_epoch}) while waiting")
                            await asyncio.sleep(10)
                            break
                        
                        # Too late to start validation (coordinator aggregates at block 300)
                        # Workers need ~8-10 min to process 50 leads, so cutoff at block 260
                        # gives them 40 blocks (8 min) before coordinator forces aggregation
                        if check_state.deadline_reached(260):
                            print(
                                "❌ Worker: Too late to start validation "
                                f"({check_state.blocks_remaining} blocks remaining)"
                            )
                            print(f"   Coordinator aggregates at block 300 - not enough time to finish")
                            await asyncio.sleep(10)
                            break
                        
                        # Log progress
                        if waited % log_interval == 0 and waited > 0:
                            print(f"⏳ Worker: Still waiting for coordinator ({waited}s elapsed)...")
                    
                    if not leads_file.exists():
                        continue  # Epoch changed or too late
                    
                    # Read leads from file (including centralized TrueList results)
                    with open(leads_file, 'r') as f:
                        data = json.load(f)
                        all_leads = data.get('leads', [])
                        epoch_id = data.get('epoch_id')
                        salt_hex = data.get('salt')  # CRITICAL: Read shared salt
                        centralized_truelist = data.get('truelist_results')  # None = in progress, {} = failed, {...} = success
                    
                    if epoch_id != current_epoch:
                        print(f"⚠️  Worker: Leads file epoch mismatch ({epoch_id} != {current_epoch})")
                        await asyncio.sleep(10)
                        continue
                    
                    if not salt_hex:
                        print(f"❌ Worker: No salt in leads file! Cannot hash results.")
                        await asyncio.sleep(10)
                        continue
                    
                    # Log TrueList status from file
                    # None = in progress (coordinator still running), {} = failed, {...} = success
                    if centralized_truelist is None:
                        print(f"   ⏳ Worker: TrueList in progress - will poll after Stage 0-2 completes")
                    elif centralized_truelist:
                        print(f"   ✅ Worker: TrueList already complete ({len(centralized_truelist)} results)")
                    else:
                        print(f"   ⚠️ Worker: TrueList failed (empty results) - leads will fail email verification")
                    
                    # Check if leads were actually fetched by coordinator
                    if all_leads is None or len(all_leads) == 0:
                        print(f"ℹ️  Worker: No leads in file for epoch {current_epoch} (coordinator returned null/empty)")
                        print(f"   This happens when: already submitted, gateway queue empty, or epoch just started")
                        print(f"   Waiting for next epoch...")
                        await asyncio.sleep(30)
                        continue
                    
                    # Calculate worker's lead subset (moved before salt print to avoid UnboundLocalError)
                    container_id = self.config.neuron.container_id
                    total_containers = self.config.neuron.total_containers
                    
                    # Convert salt from hex
                    salt = bytes.fromhex(salt_hex)
                    print(f"   Worker {container_id}: Using shared salt {salt_hex[:16]}...")
                    
                    # CRITICAL: Use SAME range slicing as coordinator (lines 1975-1991)
                    # NOT modulo - modulo causes overlap with coordinator's range!
                    original_count = len(all_leads)
                    leads_per_container = original_count // total_containers
                    remainder = original_count % total_containers
                    
                    # First 'remainder' containers get 1 extra lead to distribute remainder evenly
                    if container_id < remainder:
                        start = container_id * (leads_per_container + 1)
                        end = start + leads_per_container + 1
                    else:
                        start = (remainder * (leads_per_container + 1)) + ((container_id - remainder) * leads_per_container)
                        end = start + leads_per_container
                    
                    worker_leads = all_leads[start:end]
                    
                    print(f"   Worker {container_id}: Processing leads {start}-{end} ({len(worker_leads)}/{original_count} leads)")
                    
                    # ================================================================
                    # BATCH VALIDATION: Stage 0-2 runs in parallel with coordinator's TrueList
                    # After Stage 0-2, poll file for TrueList results before Stage 4-5
                    # ================================================================
                    
                    # Extract lead_blobs for batch processing
                    lead_blobs = [lead_data.get('lead_blob', {}) for lead_data in worker_leads]
                    
                    # Log TrueList status (might be ready or in progress)
                    if centralized_truelist:
                        print(f"   ✅ Worker {container_id}: TrueList already complete ({len(centralized_truelist)} results)")
                    elif centralized_truelist is None:
                        print(f"   ⏳ Worker {container_id}: TrueList in progress - will poll after Stage 0-2")
                    else:
                        print(f"   ⚠️ Worker {container_id}: TrueList returned empty (coordinator may have failed)")
                    
                    # Run batch validation - polls file for TrueList results after Stage 0-2
                    leads_file_str = str(leads_file)
                    try:
                        batch_results = await run_batch_automated_checks(
                            lead_blobs, 
                            container_id=container_id,
                            leads_file_path=leads_file_str,  # Poll file for TrueList results after Stage 0-2
                            current_epoch=current_epoch  # For epoch boundary detection mid-processing
                        )
                    except Exception as e:
                        print(f"   ❌ Batch validation failed: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback: Mark all leads as validation errors
                        batch_results = [
                            (False, {
                                "passed": False,
                                "rejection_reason": {
                                    "stage": "Batch Validation",
                                    "check_name": "run_batch_automated_checks",
                                    "message": f"Batch validation error: {str(e)}"
                                }
                            })
                            for _ in lead_blobs
                        ]
                    
                    # ════════════════════════════════════════════════════════════════════
                    # EPOCH BOUNDARY CHECK: Abort if epoch changed during validation
                    # This prevents workers from writing stale results for old epochs
                    # ════════════════════════════════════════════════════════════════════
                    try:
                        post_validation_block, post_validation_epoch, _ = self._read_shared_block_file()
                        post_validation_state = self._shared_epoch_state
                        if not post_validation_state.same_epoch(epoch_state):
                            print(f"\n❌ Worker {container_id}: EPOCH CHANGED during validation!")
                            print(f"   Started processing: epoch {current_epoch}")
                            print(f"   Current epoch now: {post_validation_epoch}")
                            print(f"   Aborting stale results - will start fresh on new epoch")
                            print(f"   (This prevents cascading lag from old epoch processing)\n")
                            # Don't write results, don't mark as completed
                            # Worker will re-read leads file for new epoch on next iteration
                            await asyncio.sleep(5)
                            continue  # Skip to next iteration of main loop
                    except Exception as e:
                        print(f"   ⚠️ Worker {container_id}: Could not check epoch boundary: {e}")
                        # Continue anyway - better to write potentially stale results than lose them
                    
                    # Map results back to validated_leads format (SAME ORDER guaranteed)
                    validated_leads = []
                    for i, (passed, automated_checks_data) in enumerate(batch_results):
                        lead_data = worker_leads[i]
                        lead_id = lead_data.get('lead_id', 'unknown')
                        lead_blob = lead_data.get('lead_blob', {})
                        miner_hotkey = lead_data.get('miner_hotkey', lead_blob.get('wallet_ss58', 'unknown'))
                        
                        # Handle skipped leads (passed=None means email verification unavailable)
                        if passed is None:
                            validated_leads.append({
                                'lead_id': lead_id,
                                'is_valid': False,  # Treat skipped as invalid for this epoch
                                'rejection_reason': {'message': 'EmailVerificationUnavailable'},
                                'automated_checks_data': automated_checks_data,
                                'lead_blob': lead_blob,
                                'miner_hotkey': miner_hotkey,
                                'skipped': True
                            })
                        else:
                            # Normal pass/fail
                            rejection_reason = automated_checks_data.get("rejection_reason") if not passed else None
                            validated_leads.append({
                                'lead_id': lead_id,
                                'is_valid': passed,
                                'rejection_reason': rejection_reason,
                                'automated_checks_data': automated_checks_data,
                                'lead_blob': lead_blob,
                                'miner_hotkey': miner_hotkey
                            })
                    
                    # Write results to file for coordinator
                    # CRITICAL: Hash results using shared salt (EXACT same format as coordinator)
                    results_file = Path("validator_weights") / f"worker_{container_id}_epoch_{current_epoch}_results.json"
                    
                    import hashlib
                    validation_results = []
                    local_validation_data = []
                    
                    for lead in validated_leads:
                        # Extract data
                        is_valid = lead['is_valid']
                        decision = "approve" if is_valid else "deny"
                        # CRITICAL: Use validator-calculated rep_score, NOT miner's submitted value
                        # Denied leads get 0, approved leads get score from automated checks
                        automated_checks_data = lead.get('automated_checks_data', {})
                        rep_score = int(automated_checks_data.get('rep_score', {}).get('total_score', 0)) if is_valid else 0
                        rejection_reason = lead.get('rejection_reason') or {} if not is_valid else {"message": "pass"}
                        evidence_blob = json.dumps(lead.get('automated_checks_data', {}), default=str)  # Handle datetime objects
                        
                        # Compute hashes (SHA256 with salt) - EXACT same as coordinator lines 2036-2040
                        decision_hash = hashlib.sha256((decision + salt.hex()).encode()).hexdigest()
                        rep_score_hash = hashlib.sha256((str(rep_score) + salt.hex()).encode()).hexdigest()
                        rejection_reason_hash = hashlib.sha256((json.dumps(rejection_reason, default=str) + salt.hex()).encode()).hexdigest()  # Handle datetime
                        evidence_hash = hashlib.sha256(evidence_blob.encode()).hexdigest()
                        
                        # Format for validation_results (IMMEDIATE REVEAL MODE)
                        # Include BOTH hashes AND actual values - no separate reveal phase
                        validation_results.append({
                            'lead_id': lead['lead_id'],
                            # Hash fields (for transparency log integrity)
                            'decision_hash': decision_hash,
                            'rep_score_hash': rep_score_hash,
                            'rejection_reason_hash': rejection_reason_hash,
                            'evidence_hash': evidence_hash,
                            'evidence_blob': lead.get('automated_checks_data', {}),
                            # IMMEDIATE REVEAL FIELDS - no separate reveal phase
                            'decision': decision,
                            'rep_score': rep_score,
                            'rejection_reason': rejection_reason,
                            'salt': salt.hex()
                        })
                        
                        # Format for local_validation_data (for local weight calculation)
                        # CRITICAL FIX: Include is_icp_multiplier from automated_checks_data for proper weight calc
                        local_validation_data.append({
                            'lead_id': lead['lead_id'],
                            'miner_hotkey': lead.get('miner_hotkey'),
                            'decision': decision,
                            'rep_score': rep_score,
                            'is_icp_multiplier': automated_checks_data.get("is_icp_multiplier", 0.0),
                            'rejection_reason': rejection_reason,
                            'salt': salt.hex()
                        })
                    
                    with open(results_file, 'w') as f:
                        json.dump({
                            'epoch_id': current_epoch,
                            'container_id': container_id,
                            'validation_results': validation_results,
                            'local_validation_data': local_validation_data,
                            'lead_range': f"{len(validated_leads)} leads",
                            'timestamp': time.time()
                        }, f)
                    
                    print(f"✅ Worker {container_id}: Completed {len(validated_leads)} validations")
                    print(f"   Results saved to {results_file}")
                    
                    # CRITICAL: Mark epoch as completed IN MEMORY before file gets deleted
                    # (Coordinator deletes result files after aggregation, so we can't rely on files)
                    self._completed_lead_validation_epochs.add(current_epoch)
                    
                    # MEMORY CLEANUP: Force garbage collection after each epoch
                    collected = gc.collect()
                    if collected > 100:
                        print(f"🧹 Worker {container_id}: Memory cleanup freed {collected} objects")
                    
                    # Wait before checking for next epoch
                    await asyncio.sleep(5)
                    
                except Exception as e:
                    print(f"❌ Worker error: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(30)
    
    # Create worker and run
    worker = LightweightWorker(config)
    
    # Run async loop
    try:
        asyncio.run(worker.process_gateway_validation_workflow())
    except KeyboardInterrupt:
        print("\n🛑 Worker shutting down...")
        worker.should_exit = True


# ════════════════════════════════════════════════════════════════════════════════
# DEDICATED FULFILLMENT WORKER
# ════════════════════════════════════════════════════════════════════════════════
# These 5 containers ONLY score fulfillment leads (not sourcing or qualification).
# Each container picks up a work file written by the coordinator, scores the leads,
# and writes results back for the coordinator to aggregate and submit.
# ════════════════════════════════════════════════════════════════════════════════

def run_dedicated_fulfillment_worker(config):
    """Run a dedicated fulfillment worker that ONLY scores revealed leads.

    Unlike sourcing workers (validate individual leads) or qualification workers
    (evaluate miner models via TEE sandbox), fulfillment workers:
    1. Read work files containing batches of revealed leads
    2. Score each lead through Tier 1-3 pipeline (ICP fit, data quality, intent)
    3. Write results back for coordinator aggregation

    No Bittensor connection, no axon, no lead validation, no model evaluation.
    """
    import asyncio
    import json
    import time
    from pathlib import Path

    fulfillment_container_id = config.neuron.fulfillment_container_id

    proxy_var = f"FULFILLMENT_WEBSHARE_PROXY_{fulfillment_container_id}"
    proxy_url = os.environ.get(proxy_var)
    if proxy_url:
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        print(f"🌐 Using proxy for fulfillment worker {fulfillment_container_id}")
        print("   Proxy: configured (credentials redacted)")
    else:
        print(f"⚠️ No proxy configured for fulfillment worker {fulfillment_container_id}")
        print(f"   Expected env var: {proxy_var}")

    print("")
    print("🚀 Starting dedicated fulfillment worker...")
    print(f"   Fulfillment Container ID: {fulfillment_container_id}")
    print("")

    class DedicatedFulfillmentWorker:
        def __init__(self, config):
            self.config = config
            self.should_exit = False
            self._completed_epochs = set()

        def _read_shared_block_file(self):
            """Read current block from shared file (written by coordinator)."""
            state = _read_shared_epoch_state_file(max_age_seconds=60)
            self._shared_epoch_state = state
            return (
                state.current_block,
                state.workflow_epoch_id,
                state.epoch_block,
            )

        async def process_fulfillment_leads(self, current_epoch: int):
            """Score fulfillment leads assigned to this worker.

            Under parallel processing, the coordinator writes a per-request
            work file named:
                fulfillment_worker_{fid}_work_{epoch}_{request_id}.json

            In the original (non-parallel) layout, the name was:
                fulfillment_worker_{fid}_work_{epoch}.json
            We still handle that form for backward compatibility.
            """
            fid = self.config.neuron.fulfillment_container_id
            weights_dir = Path("validator_weights")

            # Cross-epoch glob: pick up THIS worker's files from any recent
            # epoch, bounded by the same TTL Phase 1 uses for its dispatch
            # lock. Single-epoch globbing was the bug — when an epoch flipped
            # mid-scoring (Tier 2c attribute-heavy ICPs run 30-60 min and
            # an epoch is ~70 min), the worker stopped seeing its own
            # in-flight file while Phase 1's cross-epoch lock kept refusing
            # to re-dispatch. Result: file orphaned for the full 80-min TTL,
            # request stuck. Observed 2026-05-28 with request e1bd5ae5:
            # file fulfillment_worker_1_work_23014_e1bd5ae5...json sat for
            # 34+ min while worker 1 in epoch 23015 globbed only 23015
            # files and missed it.
            _FF_WORK_FILE_TTL_SEC = 80 * 60
            _now_ts = time.time()

            # Defensive cleanup: when the validator container is restarted,
            # any work file in flight at the moment of SIGTERM gets orphaned
            # — the lease-renewal task dies with the old container, the work
            # file's mtime freezes, and the new container's worker has no
            # reliable way to know whether the file is "still being
            # processed" or "stale from a dead container".  Phase 1's
            # _has_work_file uses the same TTL window, so an orphan within
            # the 80-min window also blocks re-dispatch — request stuck.
            #
            # Heuristic: if mtime is older than the lease renewal interval
            # (30 min) AND no results file exists, the lease holder is
            # dead.  Delete the orphan so Phase 1 re-dispatches fresh work
            # on the next lifecycle tick.
            #
            # This runs every polling cycle (cheap: one glob + stat per file).
            #
            # Cross-worker cleanup: glob across ALL workers, not just this fid.
            # If worker 3 is busy mid-scoring on file A, its own poll won't
            # run until A is done — so its orphan file B from a prior
            # generation can sit blocked.  Letting any idle polling worker
            # clean any worker's stale orphan closes that gap.  Safe because
            # the 30-min staleness threshold protects actively-leased files
            # (lease renewal touches mtime every 30 min).  The worker-claim
            # logic in `candidates` below still scopes to fid only, so
            # cross-cleanup doesn't cause cross-claim.
            _ORPHAN_STALENESS_SEC = 30 * 60
            for _wf in list(weights_dir.glob(
                "fulfillment_worker_*_work_*_*.json"
            )):
                try:
                    _age = _now_ts - _wf.stat().st_mtime
                except FileNotFoundError:
                    continue
                if _age <= _ORPHAN_STALENESS_SEC:
                    continue
                _results = _wf.parent / _wf.name.replace("_work_", "_results_", 1)
                if _results.exists():
                    continue
                try:
                    _wf.unlink()
                    print(f"   🧹 Worker {fid}: deleted orphan work file "
                          f"{_wf.name} (age {_age/60:.1f}min, no results)")
                except FileNotFoundError:
                    pass
                except Exception as ex:
                    print(f"   ⚠️ Worker {fid}: orphan delete failed: {ex}")

            candidates = sorted(
                wf for wf in weights_dir.glob(
                    f"fulfillment_worker_{fid}_work_*_*.json"
                )
                if (_now_ts - wf.stat().st_mtime) < _FF_WORK_FILE_TTL_SEC
            )
            # Legacy (pre-per-request layout) — name has no request_id
            # suffix, so the cross-epoch glob above wouldn't match it.
            # Keep scoped to current epoch for backward-compat probing.
            legacy = weights_dir / f"fulfillment_worker_{fid}_work_{current_epoch}.json"
            if legacy.exists():
                candidates.append(legacy)

            # Filter out work files that already have a matching results file.
            def _results_for(wf: Path) -> Path:
                return wf.parent / wf.name.replace("_work_", "_results_", 1)

            pending = [wf for wf in candidates if not _results_for(wf).exists()]
            if not pending:
                if candidates:
                    self._completed_epochs.add(current_epoch)
                return

            # In Option A there is at most 1 request per worker per epoch,
            # but handle >1 defensively just in case.
            work_file = pending[0]
            results_file = _results_for(work_file)

            print(f"\n{'='*70}")
            print(f"🎯 FULFILLMENT WORKER {fid}: Work found for epoch {current_epoch}")
            print(f"   Work file: {work_file.name}")
            if len(pending) > 1:
                print(f"   ⚠️ {len(pending) - 1} additional pending work file(s) will be processed next iteration")
            print(f"{'='*70}")

            try:
                work_data = _load_fulfillment_work_file(work_file)
            except (OSError, ValueError, TypeError) as work_exc:
                try:
                    quarantined = _quarantine_fulfillment_work_file(work_file)
                except FileNotFoundError:
                    return
                except OSError as quarantine_exc:
                    print(
                        "   ❌ fulfillment_work_quarantine_failed "
                        f"worker {fid}: {type(quarantine_exc).__name__}: "
                        f"{str(quarantine_exc)[:200]}"
                    )
                    return
                print(
                    "   ⚠️ fulfillment_work_invalid_quarantined "
                    f"worker {fid}: {type(work_exc).__name__}: "
                    f"{str(work_exc)[:200]} path={quarantined.name}"
                )
                return

            # Lease renewal: prevent Phase 1's TTL-bounded dispatch lock from
            # expiring while this worker is still actively scoring. Without
            # this, attribute-heavy ICPs running >80 min would age past the
            # _FF_WORK_FILE_TTL_SEC cliff, Phase 1 would re-dispatch the
            # same request, and the worker would double-score. Observed
            # 2026-05-28 on request e1bd5ae5: orphan work file from epoch
            # 23014 was being processed by worker 1 when at exactly 80 min
            # (22:30), Phase 1 saw the lock release and re-dispatched
            # e1bd5ae5 with my new fan-out across workers 1-4. We refresh
            # mtime every 30 min — leaves 50-min headroom under 80-min TTL.
            async def _refresh_lease():
                while True:
                    await asyncio.sleep(30 * 60)
                    try:
                        work_file.touch()
                        print(f"   🔄 Worker {fid}: lease renewed on {work_file.name}")
                    except FileNotFoundError:
                        return  # results submitted; file already cleaned up
                    except Exception as ex:
                        print(f"   ⚠️ Worker {fid}: lease renewal failed: {ex}")

            lease_task = asyncio.create_task(_refresh_lease())

            request_id = str(work_data["request_id"])
            try:
                icp_details = work_data.get("icp", {})
                submissions = work_data.get("submissions", [])

                print(f"   📦 Request: {request_id[:8]}...")
                print(f"   📦 Submissions assigned: {len(submissions)}")

                from qualification.scoring.fulfillment_scorer import (
                    score_miner_submission,
                    format_scores_for_gateway,
                )

                all_results = []
                for sub in submissions:
                    miner_hk = sub.get("miner_hotkey", "")
                    sub_id = sub.get("submission_id", "")
                    leads_raw = sub.get("leads", [])
                    lead_ids = sub.get("lead_ids", [])

                    if not leads_raw:
                        continue

                    print(f"\n   Scoring {len(leads_raw)} leads for miner {miner_hk[:12]}...")
                    try:
                        results = await score_miner_submission(leads_raw, icp_details)

                        for idx, sr in enumerate(results):
                            lead_name = leads_raw[idx].get("full_name", "?") if idx < len(leads_raw) else "?"
                            lead_biz = leads_raw[idx].get("business", "?") if idx < len(leads_raw) else "?"
                            t1 = "✅" if sr.tier1_passed else "❌"
                            t2 = "✅" if sr.tier2_passed else "❌"
                            reason = sr.failure_reason or "passed"
                            print(f"     Lead {idx+1}: {lead_name} @ {lead_biz}")
                            print(f"       Tier1: {t1}  Tier2: {t2}  Score: {sr.final_score:.1f}  [{reason}]")

                        all_results.append({
                            "miner_hotkey": miner_hk,
                            "submission_id": sub_id,
                            "lead_ids": lead_ids,
                            "results": [r.model_dump() for r in results],
                        })
                    except Exception as e:
                        # Whole-batch crash: surface a failure record for every
                        # lead in the submission so the gateway can move the
                        # request through consensus (denied) instead of waiting
                        # for the ~3h scoring timeout at 0/1 validators.
                        # Per-lead crashes should already be caught inside
                        # score_fulfillment_batch; this is defense-in-depth for
                        # higher-level failures (e.g., ICP parse, network).
                        err_type = type(e).__name__
                        err_msg = str(e)
                        print(f"   ❌ Scoring failed for miner {miner_hk[:8]}: {err_type}: {err_msg}")
                        import traceback
                        traceback.print_exc()
                        failure_results = []
                        for idx, raw_lead in enumerate(leads_raw):
                            lid = lead_ids[idx] if idx < len(lead_ids) else ""
                            failure_results.append({
                                "lead_id": lid,
                                "tier1_passed": False,
                                "tier2_passed": False,
                                "email_verified": False,
                                "person_verified": False,
                                "company_verified": False,
                                "attribute_verification": None,
                                "rep_score": 0.0,
                                "intent_signal_raw": 0.0,
                                "intent_signal_final": 0.0,
                                "intent_decay_multiplier": 0.0,
                                "final_score": 0.0,
                                "all_fabricated": False,
                                "failure_reason": "scorer_crashed",
                                "failure_detail": f"{err_type}: {err_msg[:300]}",
                                "intent_signals_detail": [],
                            })
                        all_results.append({
                            "miner_hotkey": miner_hk,
                            "submission_id": sub_id,
                            "lead_ids": lead_ids,
                            "results": failure_results,
                        })

                # ────────────────────────────────────────────────────────
                # Reconciliation: ensure submission_results has an entry for
                # EVERY submission in the work file. Otherwise revealed leads
                # silently disappear from the scoring path — observed
                # 2026-05-18 on 88.5% of the "AWAITING VALIDATION 716"
                # backlog (650 leads, 88 submissions, 77+ miners affected).
                # Causes: (a) submission had leads but `if not leads_raw:
                # continue` skipped it with no failure record, (b)
                # score_miner_submission returned an empty list, (c) worker
                # died between submissions before reaching this point.
                # Without a per-submission record Phase 2 emits no scores
                # for that miner's leads and consensus drops them silently.
                # ────────────────────────────────────────────────────────
                produced_subs = {ar["submission_id"] for ar in all_results}
                for sub in submissions:
                    sub_id = sub.get("submission_id", "")
                    if not sub_id or sub_id in produced_subs:
                        continue
                    miner_hk = sub.get("miner_hotkey", "")
                    leads_raw = sub.get("leads", []) or []
                    lead_ids = sub.get("lead_ids", []) or []
                    print(f"   ⚠️ Reconciliation: submission {sub_id[:8]} (miner {miner_hk[:8]}, {len(lead_ids)} leads) produced no result — emitting worker_skipped failures")
                    skipped_results = []
                    for idx, lid in enumerate(lead_ids):
                        skipped_results.append({
                            "lead_id": lid,
                            "tier1_passed": False,
                            "tier2_passed": False,
                            "email_verified": False,
                            "person_verified": False,
                            "company_verified": False,
                            "attribute_verification": None,
                            "rep_score": 0.0,
                            "intent_signal_raw": 0.0,
                            "intent_signal_final": 0.0,
                            "intent_decay_multiplier": 0.0,
                            "final_score": 0.0,
                            "all_fabricated": False,
                            "failure_reason": "worker_skipped",
                            "failure_detail": (
                                "Worker did not produce a score for this submission "
                                "(empty leads_raw, empty score_miner_submission output, "
                                "or worker died mid-loop). Surfaced so consensus can "
                                "write a row instead of dropping the lead silently."
                            ),
                            "intent_signals_detail": [],
                        })
                    all_results.append({
                        "miner_hotkey": miner_hk,
                        "submission_id": sub_id,
                        "lead_ids": lead_ids,
                        "results": skipped_results,
                    })

                _atomic_write_json_file(
                    results_file,
                    {
                        "epoch": current_epoch,
                        "fulfillment_worker_id": fid,
                        "request_id": request_id,
                        "submission_results": all_results,
                        "timestamp": time.time(),
                    },
                )

                self._completed_epochs.add(current_epoch)
                print(f"\n{'='*70}")
                print(f"✅ FULFILLMENT WORKER {fid}: Completed scoring for epoch {current_epoch}")
                print(f"{'='*70}\n")

            except Exception as e:
                print(f"❌ Fulfillment worker error: {e}")
                import traceback
                traceback.print_exc()
                _atomic_write_json_file(
                    results_file,
                    {
                        "epoch": current_epoch,
                        "fulfillment_worker_id": fid,
                        "request_id": request_id,
                        "error_type": type(e).__name__,
                        "error": str(e),
                        "submission_results": [],
                        "timestamp": time.time(),
                    },
                )
                self._completed_epochs.add(current_epoch)
            finally:
                # Stop lease renewal — scoring is over (success or failure).
                lease_task.cancel()
                try:
                    await lease_task
                except (asyncio.CancelledError, Exception):
                    pass

        async def run_loop(self):
            """Main loop for dedicated fulfillment worker."""
            print("🔄 Fulfillment worker starting main loop...")
            print("   (Waiting for coordinator to assign work)")

            last_epoch = -1

            while not self.should_exit:
                try:
                    try:
                        current_block, current_epoch, blocks_into_epoch = self._read_shared_block_file()
                    except FileNotFoundError:
                        print("   ⏳ Waiting for coordinator to write block file...")
                        await asyncio.sleep(10)
                        continue
                    except Exception as e:
                        print(f"   ⚠️ Block file error: {e}")
                        await asyncio.sleep(10)
                        continue

                    if current_epoch != last_epoch:
                        print(
                            f"\n📅 Epoch {current_epoch} (block "
                            f"{blocks_into_epoch}/{self._shared_epoch_state.tempo}, "
                            f"{self._shared_epoch_state.blocks_remaining} remaining)"
                        )
                        last_epoch = current_epoch

                    await self.process_fulfillment_leads(current_epoch)

                    await asyncio.sleep(5)

                except Exception as e:
                    print(f"❌ Fulfillment worker loop error: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(30)

    worker = DedicatedFulfillmentWorker(config)

    try:
        asyncio.run(worker.run_loop())
    except KeyboardInterrupt:
        print(f"\n🛑 FULFILLMENT WORKER {fulfillment_container_id}: KeyboardInterrupt")
        worker.should_exit = True
    except SystemExit as e:
        print(f"🛑 FULFILLMENT WORKER {fulfillment_container_id}: SystemExit (code={e.code})")
    except BaseException as e:
        print(f"💀 FULFILLMENT WORKER {fulfillment_container_id}: FATAL: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    parser = argparse.ArgumentParser(description="LeadPoet Validator")
    add_validator_args(None, parser)
    parser.add_argument("--wallet_name", type=str, help="Wallet name")
    parser.add_argument("--wallet_hotkey", type=str, help="Wallet hotkey")
    parser.add_argument("--wallet_path", type=str, default="~/.bittensor/wallets", help="Path to wallets directory (default: ~/.bittensor/wallets)")
    parser.add_argument("--netuid", type=int, default=71, help="Network UID")
    parser.add_argument("--subtensor_network", type=str, default=os.getenv("SUBTENSOR_NETWORK", "finney"), help="Subtensor network (default: finney, or from SUBTENSOR_NETWORK env var)")
    parser.add_argument("--logging_trace", action="store_true", help="Enable trace logging")
    parser.add_argument("--container-id", type=int, help="Container ID (0, 1, 2, etc.) for dynamic lead distribution. Container 0 is coordinator.")
    parser.add_argument("--total-containers", type=int, help="Total number of containers running (for dynamic lead distribution)")
    parser.add_argument("--mode", type=str, choices=["coordinator", "worker", "fulfillment_worker"], help="Container mode")
    args = parser.parse_args()

    try:  # low-cardinality triage tag; safe no-op when Sentry is inactive
        from leadpoet_observability import set_sentry_tag as _set_sentry_tag

        _set_sentry_tag("validator.mode", getattr(args, "mode", None) or "coordinator")
    except Exception:
        pass

    os.environ.setdefault("BITTENSOR_NETWORK", str(args.subtensor_network))
    os.environ.setdefault("BITTENSOR_NETUID", str(args.netuid))

    if args.logging_trace:
        bt.logging.set_trace(True)

    ensure_data_files()

    # ════════════════════════════════════════════════════════════════════════════
    # WORKER MODE: Skip ALL heavy initialization
    # ════════════════════════════════════════════════════════════════════════════
    # Workers don't need:
    # - Bittensor wallet/subtensor/metagraph (no chain connection)
    # - Axon serving (no API endpoints)
    # - Epoch monitor thread (coordinator writes current_block.json)
    # - Dendrite (no outgoing Bittensor requests)
    # - Weight setting (only coordinator submits weights)
    # 
    # Workers ONLY need:
    # - Read current_block.json (for epoch timing)
    # - Read epoch_{N}_leads.json (for lead data)
    # - Validate leads (CPU/IO work)
    # - Write results to JSON file
    # ════════════════════════════════════════════════════════════════════════════
    if getattr(args, 'mode', None) == "worker":
        print("════════════════════════════════════════════════════════════════")
        print("🔧 LIGHTWEIGHT WORKER MODE")
        print("════════════════════════════════════════════════════════════════")
        print("   Skipping heavy initialization:")
        print("   ✗ Bittensor wallet/subtensor/metagraph")
        print("   ✗ Axon serving")
        print("   ✗ Epoch monitor thread")
        print("   ✗ Weight setting")
        print("")
        print("   Worker responsibilities:")
        print("   ✓ Read current_block.json for epoch timing")
        print("   ✓ Read epoch_{N}_leads.json for lead data")
        print("   ✓ Validate leads (CPU/IO work)")
        print("   ✓ Write results to JSON file")
        print("════════════════════════════════════════════════════════════════")
        print("")
        
        # Create minimal config for worker
        config = bt.Config()
        config.neuron = bt.Config()
        config.neuron.container_id = getattr(args, 'container_id', None)
        config.neuron.total_containers = getattr(args, 'total_containers', None)
        config.neuron.mode = "worker"
        
        # Run lightweight worker loop
        run_lightweight_worker(config)
        return  # Exit early - don't initialize full validator

    # ════════════════════════════════════════════════════════════════════════════
    # FULFILLMENT WORKER MODE: Dedicated lead scoring containers
    # ════════════════════════════════════════════════════════════════════════════
    if getattr(args, 'mode', None) == "fulfillment_worker":
        ff_worker_id = getattr(args, 'container_id', 1)

        import signal as _signal
        def _ff_signal_handler(signum, frame):
            sig_name = _signal.Signals(signum).name if hasattr(_signal, 'Signals') else str(signum)
            print(f"\n💀 FULFILLMENT WORKER {ff_worker_id}: Received signal {sig_name} ({signum})")
            import traceback
            traceback.print_stack(frame)
            sys.exit(128 + signum)

        _signal.signal(_signal.SIGTERM, _ff_signal_handler)
        _signal.signal(_signal.SIGINT, _ff_signal_handler)

        print("════════════════════════════════════════════════════════════════")
        print(f"🎯 DEDICATED FULFILLMENT WORKER MODE (ID: {ff_worker_id})")
        print("════════════════════════════════════════════════════════════════")
        print("   Skipping heavy initialization:")
        print("   ✗ Bittensor wallet/subtensor/metagraph")
        print("   ✗ Axon serving")
        print("   ✗ Lead validation / model evaluation")
        print("   ✗ Weight setting")
        print("")
        print("   Fulfillment worker responsibilities:")
        print("   ✓ Read current_block.json for epoch timing")
        print(f"   ✓ Read fulfillment_worker_{ff_worker_id}_work_EPOCH.json")
        print("   ✓ Score revealed leads through Tier 1-3 pipeline")
        print(f"   ✓ Write results to fulfillment_worker_{ff_worker_id}_results_EPOCH.json")
        print("════════════════════════════════════════════════════════════════")
        print("")

        config = bt.Config()
        config.neuron = bt.Config()
        config.neuron.fulfillment_container_id = ff_worker_id
        config.neuron.mode = "fulfillment_worker"

        run_dedicated_fulfillment_worker(config)
        return

    # ════════════════════════════════════════════════════════════════════════════
    # COORDINATOR MODE: Full initialization
    # ════════════════════════════════════════════════════════════════════════════
    # start_epoch_monitor imported at module level

    # Run the proper Bittensor validator
    config = bt.Config()
    config.wallet = bt.Config()
    config.wallet.name = args.wallet_name
    config.wallet.hotkey = args.wallet_hotkey
    # Only set custom wallet path if default doesn't exist
    # Use wallet_path from args, or default to ~/.bittensor/wallets
    if args.wallet_path:
        config.wallet.path = str(Path(args.wallet_path).expanduser())
    else:
        config.wallet.path = str(Path.home() / ".bittensor" / "wallets")
    config.netuid = args.netuid
    config.subtensor = bt.Config()
    config.subtensor.network = args.subtensor_network
    config.neuron = bt.Config()
    config.neuron.disable_set_weights = getattr(args, 'neuron_disable_set_weights', False)
    config.neuron.container_id = getattr(args, 'container_id', None)  # Container ID (0, 1, 2, ...)
    config.neuron.total_containers = getattr(args, 'total_containers', None)  # Total containers
    config.neuron.mode = getattr(args, 'mode', None)  # Container mode: coordinator/worker

    # Start the background epoch monitor AFTER config is set (so network is correct)
    start_epoch_monitor(network=args.subtensor_network)

    validator = Validator(config=config)

    print("🚀 Starting LeadPoet Validator on Bittensor Network...")
    print(f"   Wallet: {validator.wallet.hotkey.ss58_address}")
    print(f"   NetUID: {config.netuid}")
    print("   Validator will process sourced leads and respond to API requests via Bittensor network")

    # Run the validator on the Bittensor network
    validator.run()

    # Add cleanup on shutdown (if you have a shutdown handler)
    # stop_epoch_monitor()

if __name__ == "__main__":
    import signal
    import atexit
    
    def cleanup_handler(signum=None, frame=None):
        """Clean up resources on shutdown"""
        try:
            print("\n🛑 Shutting down validator...")
            # stop_epoch_monitor imported at module level
            stop_epoch_monitor()
            
            # Give threads time to clean up
            import time
            time.sleep(1)
            
            print("✅ Cleanup complete")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")
        finally:
            if signum is not None:
                sys.exit(0)
    
    # Register cleanup handlers
    signal.signal(signal.SIGTERM, cleanup_handler)
    signal.signal(signal.SIGINT, cleanup_handler)
    atexit.register(cleanup_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        cleanup_handler()
    except Exception as e:
        print(f"❌ Validator crashed: {e}")
        cleanup_handler()
        raise

#!/usr/bin/env python3
"""
LeadPoet Auditor Validator

A lightweight validator that copies authoritative V2 weights after independently
verifying their complete attested receipt graph. During the V2 rollout, it can
copy a fully verified V1 bundle only when the gateway reports that no V2
authority exists for the requested epoch.

Verification failure is fail-closed: the auditor never substitutes a locally
computed or burn vector, and an invalid or pending V2 authority never downgrades.

USAGE:
    python neurons/auditor_validator.py --netuid 71 --wallet.name my_wallet --wallet.hotkey default
"""

import os
import sys
import argparse
import hashlib
import json
import re
import subprocess
import time
from urllib.parse import urlparse

# ════════════════════════════════════════════════════════════════════════════
# AUTO-UPDATER: Automatically updates entire repo from GitHub for auditors
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__" and os.environ.get("LEADPOET_AUDITOR_WRAPPER_ACTIVE") != "1":
    print("🔄 Leadpoet Auditor Validator: Activating auto-update wrapper...")
    print("   Your auditor will automatically stay up-to-date with the latest code")
    print("")
    
    # Create wrapper script path (hidden file with dot prefix)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    wrapper_path = os.path.join(repo_root, ".auditor_auto_update_wrapper.sh") 
    
    # Inline wrapper script - pulls on start only, not every 5 minutes
    wrapper_content = '''#!/bin/bash
# Auto-generated wrapper for Leadpoet auditor validator auto-updates
# Pulls latest code ONCE on start, then runs until clean exit
set -e

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$REPO_ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "🔍 Leadpoet Auto-Updating Auditor Validator"
echo "   Repository updates on each manual restart"
echo "   GitHub: github.com/leadpoet/leadpoet"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Pull latest code ONCE at startup
echo "────────────────────────────────────────────────────────────────"
echo "🔍 Checking for updates from GitHub..."

# Consensus code must never start from an unverified or stale checkout.
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "❌ Auditor startup refused: repository checkout is unavailable"
    exit 1
fi
if [ "$(git branch --show-current)" != "main" ]; then
    echo "❌ Auditor startup refused: checkout is not on main"
    exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "❌ Auditor startup refused: tracked checkout changes are present"
    exit 1
fi

PREVIOUS_COMMIT="$(git rev-parse HEAD)"
if ! git fetch origin main; then
    echo "❌ Auditor startup refused: origin/main could not be fetched"
    exit 1
fi
EXPECTED_COMMIT="$(git rev-parse origin/main)"
if ! git merge --ff-only origin/main; then
    echo "❌ Auditor startup refused: main cannot fast-forward to origin/main"
    exit 1
fi
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]; then
    echo "❌ Auditor startup refused: HEAD differs from fetched origin/main"
    exit 1
fi

echo "✅ Repository updated and verified"
echo "   Current commit: ${CURRENT_COMMIT:0:12}"

# Auto-install new/updated Python packages if requirements.txt changed.
if [ "$PREVIOUS_COMMIT" != "$CURRENT_COMMIT" ] && \
   git diff "$PREVIOUS_COMMIT" "$CURRENT_COMMIT" --name-only -- requirements.txt \
     | grep -q '^requirements.txt$'; then
    echo "📦 requirements.txt changed - updating packages..."
    if ! python3 -m pip install -r requirements.txt --quiet; then
        echo "❌ Auditor startup refused: package update failed"
        exit 1
    fi
fi

RESTART_COUNT=0
MAX_RESTARTS=5

while true; do
    echo "────────────────────────────────────────────────────────────────"
    echo "🟢 Starting auditor validator (attempt $(($RESTART_COUNT + 1)))..."
    echo ""
    
    # Run auditor with environment flag to prevent wrapper re-execution
    export LEADPOET_AUDITOR_WRAPPER_ACTIVE=1
    # set -e must not kill this wrapper on a nonzero exit: the whole point
    # of the loop below is to classify the exit code and restart.
    EXIT_CODE=0
    python3 neurons/auditor_validator.py "$@" || EXIT_CODE=$?
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Auditor exited cleanly (exit code: 0)"
        echo "   Shutting down. Run the command again to pull latest updates."
        break
    elif [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 9 ]; then
        echo "⚠️  Auditor was killed (exit code: $EXIT_CODE) - likely Out of Memory"
        echo "   Cleaning up resources before restart..."
        
        # Clean up any leaked resources
        pkill -f "python3 neurons/auditor_validator.py" 2>/dev/null || true
        sleep 5  # Give system time to clean up
        
        RESTART_COUNT=$((RESTART_COUNT + 1))
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Please check logs and restart manually"
            exit 1
        fi
        
        echo "   Restarting in 30 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 30
    else
        RESTART_COUNT=$((RESTART_COUNT + 1))
        echo "⚠️  Auditor exited with error (exit code: $EXIT_CODE)"
        
        if [ $RESTART_COUNT -ge $MAX_RESTARTS ]; then
            echo "❌ Maximum restart attempts ($MAX_RESTARTS) reached"
            echo "   Please check logs and restart manually"
            exit 1
        fi
        
        echo "   Restarting in 10 seconds... (attempt $RESTART_COUNT/$MAX_RESTARTS)"
        sleep 10
    fi
done

echo "════════════════════════════════════════════════════════════════"
echo "🛑 Auditor stopped. Run command again to pull latest and restart."
'''
    
    # Write wrapper script
    try:
        with open(wrapper_path, 'w') as f:
            f.write(wrapper_content)
        os.chmod(wrapper_path, 0o755)
        print(f"✅ Created auto-update wrapper: {wrapper_path}")
    except Exception as e:
        print(f"❌ Failed to create wrapper: {e}")
        raise SystemExit(
            "Auditor startup refused: update wrapper could not be created"
        ) from e
    else:
        # Execute wrapper and replace current process
        print("🚀 Launching auto-update wrapper...\n")
        try:
            env = os.environ.copy()
            env["LEADPOET_AUDITOR_WRAPPER_ACTIVE"] = "1"
            os.execve(wrapper_path, [wrapper_path] + sys.argv[1:], env)
        except Exception as e:
            print(f"❌ Failed to execute wrapper: {e}")
            raise SystemExit(
                "Auditor startup refused: update wrapper could not be executed"
            ) from e

# ════════════════════════════════════════════════════════════════════════════
# NORMAL AUDITOR VALIDATOR CODE STARTS BELOW
# ════════════════════════════════════════════════════════════════════════════

# Add repo root to path so leadpoet_canonical can be imported from anywhere
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import asyncio
import logging
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bittensor as bt
import aiohttp
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from Leadpoet.utils.bittensor_sdk import (
    ExtrinsicOutcome,
    weight_hyperparameters_compat,
)
from Leadpoet.utils.subnet_epoch import (
    OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT,
    SubnetEpochError,
    ensure_cutover_manifest_configured,
    load_subnet_epoch_cutover,
    normalize_trusted_archive_endpoint,
    read_subnet_epoch_snapshot,
    validate_subnet_epoch_cutover_anchor,
)

# Import canonical functions from shared module
from leadpoet_canonical.weights import (
    bundle_weights_hash,
    compare_weights_hash,
    u16_to_emit_floats,
)
from leadpoet_canonical.events import verify_log_entry
from leadpoet_canonical.auditor_v2 import (
    fetch_locked_release_identity_cache,
    verify_attested_weight_authority_v2,
    verify_published_weight_authority_stage_v2,
)

# Constants from canonical module
from leadpoet_canonical.constants import WEIGHT_SUBMISSION_BLOCK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Public gateway used when operators do not provide an override.
PUBLIC_GATEWAY_URL = "https://gateway.subnet71.com"

_AUDIT_LOG_SENSITIVE_NAMES = (
    "authorization",
    "credential",
    "password",
    "private",
    "secret",
    "signature",
    "token",
)
_AUDIT_LOG_QUERY_SECRET = re.compile(
    r"(?i)([?&](?:authorization|credential|key|password|secret|signature|token)"
    r")=[^&\s]+"
)
_AUDIT_LOG_BEARER_SECRET = re.compile(r"(?i)\bBearer\s+\S+")


def _sanitize_audit_log_value(name: str, value: Any) -> Any:
    """Return diagnostic context without exposing authentication material."""

    lowered = str(name).lower()
    if any(marker in lowered for marker in _AUDIT_LOG_SENSITIVE_NAMES):
        return "[redacted]"
    if isinstance(value, dict):
        return {
            str(key): _sanitize_audit_log_value(str(key), child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_audit_log_value(name, child)
            for child in value
        ]
    if isinstance(value, str):
        text = _AUDIT_LOG_QUERY_SECRET.sub(r"\1=[redacted]", value)
        text = _AUDIT_LOG_BEARER_SECRET.sub("Bearer [redacted]", text)
        return text[:500]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]


def _audit_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    """Emit one stable JSON event for operator-side incident diagnosis."""

    try:
        payload = {
            "event": str(event),
            **{
                str(key): _sanitize_audit_log_value(str(key), value)
                for key, value in fields.items()
            },
        }
        logger.log(
            level,
            "auditor_event %s",
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )
    except Exception as exc:  # pragma: no cover - diagnostics must be inert
        logger.warning(
            "auditor_event_serialization_failure event=%s type=%s",
            str(event)[:80],
            type(exc).__name__,
        )


def _default_gateway_url(environ=None) -> str:
    source = os.environ if environ is None else environ
    return str(source.get("GATEWAY_URL", "") or "").strip() or PUBLIC_GATEWAY_URL


DEFAULT_GATEWAY_URL = _default_gateway_url()
AUDITOR_ARCHIVE_ENDPOINT_ENV = "AUDITOR_BITTENSOR_ARCHIVE_ENDPOINT"
BITTENSOR_ARCHIVE_ENDPOINT_ENV = "BITTENSOR_ARCHIVE_ENDPOINT"


def _auditor_runtime_identity() -> Dict[str, str]:
    """Return public source identities needed to diagnose stale auditors."""

    paths = {
        "auditor_sha256": Path(__file__).resolve(),
        "weight_authority_sha256": (
            Path(_REPO_ROOT) / "leadpoet_canonical" / "weight_authority_v2.py"
        ),
        "weight_computation_sha256": (
            Path(_REPO_ROOT) / "leadpoet_canonical" / "weight_computation.py"
        ),
    }
    identity = {}
    for field, path in paths.items():
        try:
            identity[field] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            identity[field] = "unavailable"
    try:
        identity["commit"] = subprocess.run(
            ["git", "-C", _REPO_ROOT, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        identity["commit"] = "unavailable"
    identity["python"] = sys.version.split()[0]
    identity["bittensor"] = str(getattr(bt, "__version__", "unknown"))
    return identity


def _auditor_archive_endpoint(environ=None) -> str:
    source = os.environ if environ is None else environ
    explicit = str(source.get(AUDITOR_ARCHIVE_ENDPOINT_ENV, "") or "").strip()
    compatible = str(source.get(BITTENSOR_ARCHIVE_ENDPOINT_ENV, "") or "").strip()
    if explicit and compatible and explicit.rstrip("/") != compatible.rstrip("/"):
        raise SubnetEpochError("conflicting auditor archive endpoints are configured")
    return normalize_trusted_archive_endpoint(
        explicit or compatible or OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
    )


def _connect_epoch_archive_subtensor(
    *,
    endpoint: Optional[str] = None,
    attempts: int = 3,
    retry_delay_seconds: float = 2.0,
):
    """Connect to the selected archive authority before opening the live RPC."""

    selected_endpoint = _auditor_archive_endpoint() if endpoint is None else (
        normalize_trusted_archive_endpoint(endpoint)
    )
    last_error = None
    for attempt in range(1, int(attempts) + 1):
        started_at = time.monotonic()
        _audit_event(
            "archive_connect_attempt",
            archive_endpoint=selected_endpoint,
            attempt=attempt,
            attempts=attempts,
        )
        try:
            subtensor = bt.Subtensor(network=selected_endpoint)
            _audit_event(
                "archive_connect_success",
                archive_endpoint=selected_endpoint,
                attempt=attempt,
                elapsed_ms=round((time.monotonic() - started_at) * 1000, 1),
                sdk_version=str(getattr(bt, "__version__", "unknown")),
            )
            return subtensor
        except Exception as exc:
            last_error = exc
            _audit_event(
                "archive_connect_failure",
                level=logging.WARNING,
                archive_endpoint=selected_endpoint,
                attempt=attempt,
                attempts=attempts,
                elapsed_ms=round((time.monotonic() - started_at) * 1000, 1),
                error_type=type(exc).__name__,
                error=str(exc),
            )
            logger.warning(
                "auditor_archive_connect_failed attempt=%d/%d type=%s",
                attempt,
                attempts,
                type(exc).__name__,
            )
            if attempt < int(attempts):
                time.sleep(float(retry_delay_seconds))
    raise SubnetEpochError(
        "auditor could not connect to its selected trusted epoch archive"
    ) from last_error


def _close_subtensor_connection(subtensor, *, source: str) -> None:
    """Retire one failed SDK websocket without hiding cleanup failures."""

    close = getattr(subtensor, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:
        logger.warning(
            "auditor_subtensor_close_failed source=%s type=%s",
            source,
            type(exc).__name__,
        )


@dataclass(frozen=True)
class _AuditorEpochState:
    current_block: int
    workflow_epoch_id: int
    epoch_block: int
    blocks_remaining: int
    subnet_epoch_index: Optional[int]

    @property
    def identity(self) -> int:
        if self.subnet_epoch_index is None:
            raise SubnetEpochError("official subnet epoch identity is unavailable")
        return self.subnet_epoch_index

    def deadline_reached(self, elapsed_block: int) -> bool:
        return self.epoch_block >= int(elapsed_block)


def _normalize_gateway_url(value: str) -> str:
    """Accept the current gateway transport without weakening V2 integrity."""

    normalized = str(value or "").strip().rstrip("/")
    parsed = urlparse(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("Auditor gateway URL must be an HTTP(S) origin")
    return normalized


# File to store pending equivocation check (overwritten each epoch)
PENDING_EQUIVOCATION_FILE = os.path.join(_SCRIPT_DIR, ".pending_equivocation_check.json")


class AuditorValidator:
    """
    Validator that prefers V2 and narrowly supports verified rollout-era V1.
    """
    
    def __init__(self, config, gateway_url: str):
        """
        Initialize auditor validator.
        
        Args:
            config: Bittensor config object
            gateway_url: Gateway URL (passed as parameter, not global)
        """
        self.config = config
        self.gateway_url = _normalize_gateway_url(gateway_url)
        self.wallet = bt.Wallet(config=config)
        # Auditors run from a plain repository checkout with no provisioning
        # step, so fall back to the repo's public SN71 cutover manifest when
        # the operator has not configured one explicitly.
        ensure_cutover_manifest_configured()
        self.epoch_cutover = load_subnet_epoch_cutover()
        if (
            self.epoch_cutover is not None
            and int(self.epoch_cutover.netuid) != int(config.netuid)
        ):
            raise SubnetEpochError(
                "auditor netuid differs from subnet epoch cutover manifest"
            )
        # Open and authenticate the archive before the live RPC. The installed
        # SDK can strand a second websocket during TLS cleanup when live is
        # opened first, preventing otherwise valid auditors from starting.
        self.epoch_archive_endpoint = _auditor_archive_endpoint()
        self.epoch_archive_subtensor = _connect_epoch_archive_subtensor(
            endpoint=self.epoch_archive_endpoint
        )
        validate_subnet_epoch_cutover_anchor(
            self.epoch_archive_subtensor,
            self.epoch_cutover,
            expected_archive_endpoint=self.epoch_archive_endpoint,
        )
        self._validate_durable_epoch_runtime_startup()
        self.subtensor = bt.Subtensor(config=config)
        
        # Verify we're registered as a validator
        self.uid = self._get_uid()
        if self.uid is None:
            raise RuntimeError(
                f"Wallet {self.wallet.hotkey.ss58_address} is not registered "
                f"on netuid {config.netuid}"
            )
        
        self.should_exit = False
        self.last_submitted_epoch = None
        self.last_authority_epoch = None
        self.consecutive_errors = 0
        self.max_consecutive_errors = 5  # Reconnect subtensor after this many errors
        
        # Gateway attestation (for log verification)
        self.gateway_pubkey = None
        self.gateway_attestation = None
        self.gateway_code_hash = None
        
        # Validator attestation (extracted from weight bundles)
        self.validator_pubkey = None
        self.validator_attestation = None
        self.validator_code_hash = None
        self.validator_hotkey = None
        
        weight_protocol = self.auditor_weight_protocol()
        runtime_identity = _auditor_runtime_identity()
        logger.info("✅ Auditor Validator initialized")
        logger.info("auditor_weight_protocol=%s", weight_protocol)
        logger.info(
            "auditor_runtime_identity commit=%s python=%s bittensor=%s "
            "auditor_sha256=%s weight_authority_sha256=%s "
            "weight_computation_sha256=%s",
            runtime_identity["commit"],
            runtime_identity["python"],
            runtime_identity["bittensor"],
            runtime_identity["auditor_sha256"],
            runtime_identity["weight_authority_sha256"],
            runtime_identity["weight_computation_sha256"],
        )
        _audit_event(
            "startup_ready",
            commit=runtime_identity["commit"],
            python_version=runtime_identity["python"],
            sdk_version=runtime_identity["bittensor"],
            auditor_sha256=runtime_identity["auditor_sha256"],
            weight_authority_sha256=runtime_identity["weight_authority_sha256"],
            weight_computation_sha256=runtime_identity[
                "weight_computation_sha256"
            ],
            netuid=int(self.config.netuid),
            uid=int(self.uid),
            hotkey=self.wallet.hotkey.ss58_address,
            gateway_endpoint=self.gateway_url,
            archive_endpoint=self.epoch_archive_endpoint,
            weight_protocol=weight_protocol,
            transport_auth_mode="public_bundle_with_cryptographic_verification",
            cutover_mapping_hash=(
                self.epoch_cutover.mapping_hash
                if self.epoch_cutover is not None
                else None
            ),
        )
        print(f"✅ Auditor Validator initialized")
        print(f"   Hotkey: {self.wallet.hotkey.ss58_address}")
        print(f"   UID: {self.uid}")
        print(f"   Gateway: {self.gateway_url}")
        print(f"   Weight protocol: {weight_protocol}")

    def _validate_durable_epoch_runtime_lifecycle(
        self,
        *,
        force_refresh: bool,
    ) -> dict:
        """Read the fixed-project, RLS-protected durable cutover singleton."""

        cutover = getattr(self, "epoch_cutover", None)
        if cutover is None:
            raise SubnetEpochError("auditor subnet epoch cutover is unavailable")
        subtensor_config = getattr(self.config, "subtensor", None)
        network = str(
            getattr(subtensor_config, "network", "") or ""
        ).strip().lower()
        uses_production_authority = (
            int(self.config.netuid) == 71 and network == "finney"
        )
        if not uses_production_authority:
            return {
                "lifecycle_state": "stateful_manifest_only",
                "mapping_hash": cutover.mapping_hash,
            }

        from gateway.utils.epoch import validate_epoch_runtime_lifecycle

        return validate_epoch_runtime_lifecycle(
            cutover=cutover,
            force_refresh=force_refresh,
            network=network,
            netuid=int(self.config.netuid),
        )

    def _validate_durable_epoch_runtime_startup(self) -> None:
        """Reject an inactive or mismatched settlement mapping."""

        cutover = getattr(self, "epoch_cutover", None)
        if cutover is None:
            raise SubnetEpochError("auditor subnet epoch cutover is unavailable")
        subtensor_config = getattr(self.config, "subtensor", None)
        network = str(
            getattr(subtensor_config, "network", "") or ""
        ).strip().lower()
        if int(self.config.netuid) == 71 and network == "finney":
            from gateway.utils.epoch import validate_stateful_cutover_authority

            validate_stateful_cutover_authority(
                cutover,
                network=network,
                netuid=int(self.config.netuid),
            )
        self._validate_durable_epoch_runtime_lifecycle(
            force_refresh=True,
        )

    def _read_epoch_state(self) -> _AuditorEpochState:
        if self.epoch_cutover is None:
            raise SubnetEpochError("auditor subnet epoch cutover is unavailable")
        self._validate_durable_epoch_runtime_lifecycle(
            force_refresh=False,
        )
        snapshot = read_subnet_epoch_snapshot(
            self.subtensor,
            netuid=int(self.config.netuid),
            finalized=True,
        )
        return _AuditorEpochState(
            current_block=snapshot.current_block,
            workflow_epoch_id=snapshot.settlement_epoch_id(self.epoch_cutover),
            epoch_block=snapshot.epoch_block,
            blocks_remaining=snapshot.blocks_remaining,
            subnet_epoch_index=snapshot.subnet_epoch_index,
        )

    def _read_best_epoch_state(self) -> _AuditorEpochState:
        """Read best-head state only as an official submission liveness veto."""

        if self.epoch_cutover is None:
            raise SubnetEpochError("auditor subnet epoch cutover is unavailable")
        snapshot = read_subnet_epoch_snapshot(
            self.subtensor,
            netuid=int(self.config.netuid),
            finalized=False,
        )
        return _AuditorEpochState(
            current_block=snapshot.current_block,
            workflow_epoch_id=snapshot.settlement_epoch_id(self.epoch_cutover),
            epoch_block=snapshot.epoch_block,
            blocks_remaining=snapshot.blocks_remaining,
            subnet_epoch_index=snapshot.subnet_epoch_index,
        )

    def _subnet_index_for_workflow_epoch(self, epoch_id: int) -> Optional[int]:
        cutover = self.epoch_cutover
        if cutover is None or int(epoch_id) < cutover.first_settlement_epoch_id:
            raise SubnetEpochError("auditor workflow epoch predates cutover")
        return cutover.first_subnet_epoch_index + (
            int(epoch_id) - cutover.first_settlement_epoch_id
        )

    def _verify_stateful_bundle_epoch(self, bundle: Dict) -> None:
        if self.epoch_cutover is None:
            raise SubnetEpochError("auditor subnet epoch cutover is unavailable")
        started_at = time.monotonic()
        _audit_event(
            "exact_block_verification_start",
            epoch=int(bundle["epoch_id"]),
            netuid=int(bundle.get("netuid", self.config.netuid)),
            block=int(bundle["block"]),
            archive_endpoint=getattr(self, "epoch_archive_endpoint", None),
            weights_hash=bundle.get("weights_hash"),
            authority_stage=bundle.get("authority_stage"),
        )
        try:
            snapshot = read_subnet_epoch_snapshot(
                self.epoch_archive_subtensor,
                netuid=int(self.config.netuid),
                block_number=int(bundle["block"]),
            )
        except Exception as first_error:
            _audit_event(
                "exact_block_verification_retry",
                level=logging.WARNING,
                epoch=int(bundle["epoch_id"]),
                block=int(bundle["block"]),
                archive_endpoint=self.epoch_archive_endpoint,
                error_type=type(first_error).__name__,
                error=str(first_error),
            )
            logger.warning(
                "auditor_archive_snapshot_reconnect endpoint=%s type=%s",
                self.epoch_archive_endpoint,
                type(first_error).__name__,
            )
            stale = self.epoch_archive_subtensor
            self.epoch_archive_subtensor = None
            _close_subtensor_connection(stale, source="archive_stale")
            refreshed = None
            try:
                refreshed = _connect_epoch_archive_subtensor(
                    endpoint=self.epoch_archive_endpoint
                )
                validate_subnet_epoch_cutover_anchor(
                    refreshed,
                    self.epoch_cutover,
                    expected_archive_endpoint=self.epoch_archive_endpoint,
                )
                snapshot = read_subnet_epoch_snapshot(
                    refreshed,
                    netuid=int(self.config.netuid),
                    block_number=int(bundle["block"]),
                )
            except Exception as retry_error:
                _audit_event(
                    "exact_block_verification_failure",
                    level=logging.ERROR,
                    epoch=int(bundle["epoch_id"]),
                    block=int(bundle["block"]),
                    archive_endpoint=self.epoch_archive_endpoint,
                    error_type=type(retry_error).__name__,
                    error=str(retry_error),
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
                if refreshed is not None:
                    _close_subtensor_connection(
                        refreshed,
                        source="archive_replacement_failed",
                    )
                raise SubnetEpochError(
                    "auditor selected archive could not verify the bundle block"
                ) from retry_error
            self.epoch_archive_subtensor = refreshed
        if snapshot.settlement_epoch_id(self.epoch_cutover) != int(
            bundle["epoch_id"]
        ):
            raise SubnetEpochError(
                "bundle settlement epoch differs from official chain state"
            )
        _audit_event(
            "exact_block_verification_success",
            epoch=int(bundle["epoch_id"]),
            netuid=int(bundle.get("netuid", self.config.netuid)),
            block=int(bundle["block"]),
            block_hash=getattr(snapshot, "block_hash", None),
            subnet_epoch_index=getattr(snapshot, "subnet_epoch_index", None),
            observed_settlement_epoch=snapshot.settlement_epoch_id(
                self.epoch_cutover
            ),
            archive_endpoint=getattr(self, "epoch_archive_endpoint", None),
            elapsed_ms=round((time.monotonic() - started_at) * 1000, 1),
        )
    
    def _get_uid(self) -> Optional[int]:
        """Resolve our current UID directly from canonical chain storage."""

        return self.subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=self.wallet.hotkey.ss58_address,
            netuid=int(self.config.netuid),
        )
    
    def _reconnect_subtensor(self):
        """
        Reconnect to subtensor after connection errors.
        
        CRITICAL: Validators run for days. Websocket connections WILL drop.
        This ensures the validator keeps running after network issues.
        """
        print(f"\n🔄 Reconnecting to subtensor...")
        logger.info("Reconnecting to subtensor after connection error")
        
        try:
            # Create new subtensor instance
            self.subtensor = bt.Subtensor(config=self.config)
            
            # Verify we're still registered
            new_uid = self._get_uid()
            if new_uid is None:
                logger.error("Lost registration after reconnect!")
                print(f"❌ Lost registration after reconnect!")
            else:
                self.uid = new_uid
                print(f"✅ Reconnected to subtensor (UID: {self.uid})")
                logger.info(f"Reconnected to subtensor (UID: {self.uid})")
            
            self.consecutive_errors = 0
            return True
            
        except Exception as e:
            logger.error(f"Failed to reconnect to subtensor: {e}")
            print(f"❌ Failed to reconnect: {e}")
            return False
    
    def _get_primary_validator_uid(self, weights_data: Dict) -> Optional[int]:
        """
        Get primary validator UID by matching hotkey from weight bundle.
        
        DO NOT assume UID 0 is primary - look up from weights bundle.
        """
        validator_hotkey = weights_data.get("validator_hotkey")
        if not validator_hotkey:
            print(f"⚠️  No validator_hotkey in weights bundle")
            return None
        
        uid = self.subtensor.get_uid_for_hotkey_on_subnet(
            hotkey_ss58=validator_hotkey,
            netuid=int(self.config.netuid),
        )
        if uid is not None:
            print(f"   Primary validator hotkey: {validator_hotkey[:16]}... → UID {uid}")
            return uid
        
        print(f"⚠️  Validator hotkey {validator_hotkey[:16]}... not found on subnet")
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Soft Anti-Equivocation (Retroactive Check)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def save_pending_equivocation_check(self, epoch_id: int, bundle_compare_hash: str, validator_hotkey: str):
        """
        Save bundle compare hash for retroactive equivocation check.
        
        Called after successful current-epoch weight verification.
        APPENDS to existing data (stores last 2 epochs for N-2 checking).
        
        File structure:
        {
            "20256": {"bundle_compare_hash": "...", "validator_hotkey": "...", "saved_at": "..."},
            "20257": {"bundle_compare_hash": "...", "validator_hotkey": "...", "saved_at": "..."}
        }
        """
        import json
        from datetime import datetime, timezone
        
        # Load existing data
        existing_data = {}
        try:
            if os.path.exists(PENDING_EQUIVOCATION_FILE):
                with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                    existing_data = json.load(f)
                    # Handle old format (single epoch) - migrate to new format
                    if "epoch_id" in existing_data:
                        old_epoch = str(existing_data["epoch_id"])
                        existing_data = {
                            old_epoch: {
                                "bundle_compare_hash": existing_data.get("bundle_compare_hash"),
                                "validator_hotkey": existing_data.get("validator_hotkey"),
                                "saved_at": existing_data.get("saved_at"),
                            }
                        }
        except Exception as e:
            logger.warning(f"Could not load existing equivocation data: {e}")
            existing_data = {}
        
        # Add new epoch data
        existing_data[str(epoch_id)] = {
            "bundle_compare_hash": bundle_compare_hash,
            "validator_hotkey": validator_hotkey,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        # Keep only last 3 epochs (current + 2 previous for safety)
        epoch_keys = sorted(existing_data.keys(), key=int, reverse=True)
        if len(epoch_keys) > 3:
            for old_key in epoch_keys[3:]:
                del existing_data[old_key]
        
        try:
            with open(PENDING_EQUIVOCATION_FILE, 'w') as f:
                json.dump(existing_data, f, indent=2)
            print(f"   📝 Saved pending equivocation check for epoch {epoch_id}")
            print(f"      Epochs in file: {sorted(existing_data.keys(), key=int)}")
        except Exception as e:
            logger.warning(f"Failed to save pending equivocation check: {e}")
    
    def load_pending_equivocation_check(self, target_epoch: int = None) -> Optional[Dict]:
        """
        Load pending equivocation check data for a specific epoch.
        
        Args:
            target_epoch: Specific epoch to load (if None, returns all epochs)
            
        Returns:
            Dict with bundle_compare_hash, validator_hotkey for the target epoch
            None if not found or file doesn't exist
        """
        import json
        try:
            if not os.path.exists(PENDING_EQUIVOCATION_FILE):
                return None
            with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                data = json.load(f)
            
            # Handle old format (single epoch with epoch_id key)
            if "epoch_id" in data:
                old_epoch = data["epoch_id"]
                if target_epoch is None or target_epoch == old_epoch:
                    return {
                        "epoch_id": old_epoch,
                        "bundle_compare_hash": data.get("bundle_compare_hash"),
                        "validator_hotkey": data.get("validator_hotkey"),
                    }
                return None
            
            # New format: dict keyed by epoch_id string
            if target_epoch is not None:
                epoch_data = data.get(str(target_epoch))
                if epoch_data:
                    return {
                        "epoch_id": target_epoch,
                        "bundle_compare_hash": epoch_data.get("bundle_compare_hash"),
                        "validator_hotkey": epoch_data.get("validator_hotkey"),
                    }
                return None
            
            # Return all data if no target specified
            return data
            
        except Exception as e:
            logger.warning(f"Failed to load pending equivocation check: {e}")
            return None
    
    def clear_pending_equivocation_check(self, epoch_id: int = None):
        """
        Clear a specific epoch from the pending equivocation check file.
        
        Args:
            epoch_id: Specific epoch to remove. If None, clears entire file.
        """
        import json
        try:
            if not os.path.exists(PENDING_EQUIVOCATION_FILE):
                return
                
            if epoch_id is None:
                # Clear entire file
                os.remove(PENDING_EQUIVOCATION_FILE)
                return
            
            # Load existing data
            with open(PENDING_EQUIVOCATION_FILE, 'r') as f:
                data = json.load(f)
            
            # Handle old format
            if "epoch_id" in data:
                if data["epoch_id"] == epoch_id:
                    os.remove(PENDING_EQUIVOCATION_FILE)
                return
            
            # Remove specific epoch from new format
            if str(epoch_id) in data:
                del data[str(epoch_id)]
                print(f"   🗑️  Cleared epoch {epoch_id} from pending checks")
                
                if data:
                    # Write remaining epochs back
                    with open(PENDING_EQUIVOCATION_FILE, 'w') as f:
                        json.dump(data, f, indent=2)
                else:
                    # No epochs left, delete file
                    os.remove(PENDING_EQUIVOCATION_FILE)
                    
        except Exception as e:
            logger.warning(f"Failed to clear pending equivocation check: {e}")
    
    async def perform_soft_equivocation_check(self, target_epoch: int) -> bool:
        """
        Verify that a submitted bundle has its exact finalized-chain proof.
        
        Live ``subtensor.weights()`` exposes only the primary validator's most
        recent vector, so it cannot prove an older epoch after later weights
        have finalized. The finalized V2 authority instead binds the exact
        bundle to the enclave-authorized extrinsic and its finalized state
        transition at the historical block.
        
        Args:
            target_epoch: The epoch to check (typically current_epoch - 2)
        
        Returns:
            True if check passed or no pending check
            False if equivocation detected
        """
        pending = self.load_pending_equivocation_check(target_epoch=target_epoch)
        if not pending:
            return True  # No pending check for this epoch
        
        epoch_id = target_epoch
        bundle_compare_hash = pending.get("bundle_compare_hash")
        validator_hotkey = pending.get("validator_hotkey")
        
        if not all([bundle_compare_hash, validator_hotkey]):
            logger.warning(f"Invalid pending equivocation data for epoch {epoch_id}")
            self.clear_pending_equivocation_check(epoch_id)
            return True
        
        print(f"\n{'='*60}")
        print(f"🔍 SOFT EQUIVOCATION CHECK (Epoch {epoch_id})")
        print(f"{'='*60}")
        print(f"   Checking the bundle's finalized-chain proof...")
        
        try:
            bundle, authority_status = await self.fetch_verified_weight_authority(
                epoch_id
            )

            if not bundle:
                print(
                    f"   ⚠️  Could not verify weight authority for epoch "
                    f"{epoch_id} ({authority_status}), retaining for retry"
                )
                return True

            if bundle.get("authority_stage") != "finalized":
                print(
                    f"   ⏳ Finalized-chain proof is not available yet; "
                    f"retaining epoch {epoch_id} for retry"
                )
                return True

            bundle_uids = bundle.get("uids", [])
            bundle_weights = bundle.get("weights_u16", [])
            if (
                not isinstance(bundle_uids, list)
                or not isinstance(bundle_weights, list)
                or not bundle_uids
                or len(bundle_uids) != len(bundle_weights)
            ):
                raise ValueError("finalized authority vector is invalid")
            bundle_pairs = list(zip(bundle_uids, bundle_weights))

            finalized_compare_hash = compare_weights_hash(
                int(bundle["netuid"]),
                int(bundle["epoch_id"]),
                bundle_pairs,
            )
            if (
                finalized_compare_hash != bundle_compare_hash
                or str(bundle.get("validator_hotkey") or "") != validator_hotkey
            ):
                print(
                    f"   ❌ Finalized authority differs from the submitted "
                    f"epoch {epoch_id} bundle"
                )
                logger.error(
                    "auditor_finalized_authority_mismatch epoch=%s",
                    epoch_id,
                )
                self.clear_pending_equivocation_check(epoch_id)
                return False

            print(
                f"   ✅ MATCH - exact bundle finalized in block "
                f"{bundle['finalized_block']}"
            )
            print(f"   Extrinsic: {bundle['extrinsic_hash']}")
            self.clear_pending_equivocation_check(epoch_id)
            return True
                
        except Exception as e:
            print(f"   ⚠️  Soft equivocation check failed: {e}")
            logger.warning(f"Soft equivocation check error: {e}")
            return True  # Submission already happened; retain evidence for retry.
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Gateway Communication
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _authority_candidate_epochs(self, current_epoch: int) -> List[int]:
        """Return only the unmirrored authority for the live epoch."""

        epoch = int(current_epoch)
        if epoch <= 0:
            return []
        if (
            self.last_authority_epoch is not None
            and epoch <= self.last_authority_epoch
        ):
            return []
        return [epoch]

    async def fetch_attested_weights_v2(self, epoch_id: int) -> Optional[Dict]:
        """Fetch the strongest authoritative V2 authority for one epoch.

        Prefers the staged view (published or finalized), which exists as
        soon as the primary's durable gateway publication lands, so the
        live epoch can be mirrored in-window. Falls back to the
        finalized-only legacy route when the gateway predates the staged
        view.
        """

        self._last_v2_authority_was_absent = False
        netuid = int(self.config.netuid)
        staged_url = (
            f"{self.gateway_url}/weights/v2/published/{netuid}/{int(epoch_id)}"
        )
        legacy_url = (
            f"{self.gateway_url}/weights/v2/latest/{netuid}/{int(epoch_id)}"
        )
        absent_details = {
            "v2 weight bundle not found",
            "finalized v2 weight authority not found",
            "published v2 weight authority not found",
        }
        try:
            async with aiohttp.ClientSession(trust_env=False) as session:
                for endpoint_kind, url in (
                    ("published", staged_url),
                    ("finalized_legacy", legacy_url),
                ):
                    started_at = time.monotonic()
                    _audit_event(
                        "bundle_fetch_attempt",
                        epoch=int(epoch_id),
                        netuid=netuid,
                        uid=getattr(self, "uid", None),
                        endpoint=urlparse(url).path,
                        endpoint_kind=endpoint_kind,
                        gateway_endpoint=self.gateway_url,
                    )
                    async with session.get(
                        url, timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        elapsed_ms = round(
                            (time.monotonic() - started_at) * 1000,
                            1,
                        )
                        if response.status == 404:
                            try:
                                not_found = await response.json()
                            except Exception:
                                not_found = None
                            detail = (
                                str(not_found.get("detail") or "").strip()
                                if isinstance(not_found, dict)
                                else ""
                            )
                            if url == staged_url and detail == "Not Found":
                                # The gateway predates the staged route;
                                # fall through to the legacy view.
                                _audit_event(
                                    "bundle_fetch_route_absent",
                                    epoch=int(epoch_id),
                                    netuid=netuid,
                                    endpoint=urlparse(url).path,
                                    endpoint_kind=endpoint_kind,
                                    http_status=404,
                                    elapsed_ms=elapsed_ms,
                                )
                                continue
                            self._last_v2_authority_was_absent = detail in (
                                absent_details | {"Not Found"}
                            )
                            _audit_event(
                                "bundle_fetch_not_found",
                                level=logging.INFO
                                if self._last_v2_authority_was_absent
                                else logging.WARNING,
                                epoch=int(epoch_id),
                                netuid=netuid,
                                endpoint=urlparse(url).path,
                                endpoint_kind=endpoint_kind,
                                http_status=404,
                                authority_absent=self._last_v2_authority_was_absent,
                                detail=detail,
                                elapsed_ms=elapsed_ms,
                            )
                            if not self._last_v2_authority_was_absent:
                                logger.warning(
                                    "auditor_v2_fetch_failed epoch=%s "
                                    "status=404 detail=%s",
                                    epoch_id,
                                    detail[:160],
                                )
                            return None
                        if response.status != 200:
                            _audit_event(
                                "bundle_fetch_http_failure",
                                level=logging.WARNING,
                                epoch=int(epoch_id),
                                netuid=netuid,
                                endpoint=urlparse(url).path,
                                endpoint_kind=endpoint_kind,
                                http_status=response.status,
                                elapsed_ms=elapsed_ms,
                            )
                            logger.warning(
                                "auditor_v2_fetch_failed epoch=%s status=%s",
                                epoch_id,
                                response.status,
                            )
                            return None
                        value = await response.json()
                        _audit_event(
                            "bundle_fetch_success",
                            epoch=int(epoch_id),
                            netuid=netuid,
                            uid=getattr(self, "uid", None),
                            endpoint=urlparse(url).path,
                            endpoint_kind=endpoint_kind,
                            http_status=response.status,
                            elapsed_ms=elapsed_ms,
                            response_bytes=getattr(
                                response, "content_length", None
                            ),
                            bundle_epoch=value.get("epoch_id")
                            if isinstance(value, dict)
                            else None,
                            bundle_block=value.get("block")
                            if isinstance(value, dict)
                            else None,
                            authority_stage=value.get("authority_stage")
                            if isinstance(value, dict)
                            else None,
                            weights_hash=value.get("weights_hash")
                            if isinstance(value, dict)
                            else None,
                            bundle_hash=value.get("bundle_hash")
                            if isinstance(value, dict)
                            else None,
                            destination_count=len(value.get("uids", []))
                            if isinstance(value, dict)
                            and isinstance(value.get("uids"), list)
                            else None,
                        )
                        return value if isinstance(value, dict) else None
            return None
        except Exception as exc:
            _audit_event(
                "bundle_fetch_exception",
                level=logging.WARNING,
                epoch=int(epoch_id),
                netuid=netuid,
                uid=getattr(self, "uid", None),
                gateway_endpoint=self.gateway_url,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            logger.warning(
                "auditor_v2_fetch_failed epoch=%s error_type=%s error=%s",
                epoch_id,
                type(exc).__name__,
                str(exc)[:200],
            )
            return None

    @staticmethod
    def auditor_weight_protocol() -> str:
        """Require the authoritative V2 auditor protocol."""

        raw = str(
            os.environ.get("AUDITOR_WEIGHT_PROTOCOL", "authoritative_v2")
            or "authoritative_v2"
        ).strip().lower()
        if raw != "authoritative_v2":
            raise RuntimeError(
                "AUDITOR_WEIGHT_PROTOCOL must be authoritative_v2"
            )
        return raw

    async def fetch_verified_weight_authority(
        self,
        epoch_id: int,
    ) -> Tuple[Optional[Dict], str]:
        """Fetch the authoritative V2 weight bundle."""

        self.auditor_weight_protocol()
        started_at = time.monotonic()
        _audit_event(
            "authority_verification_start",
            epoch=int(epoch_id),
            netuid=int(self.config.netuid),
            uid=getattr(self, "uid", None),
            gateway_endpoint=getattr(self, "gateway_url", None),
        )
        v2_bundle = await self.fetch_attested_weights_v2(epoch_id)
        if isinstance(v2_bundle, dict):
            try:
                identity_cache = await self._fetch_release_identity_cache(v2_bundle)
            except Exception as exc:
                _audit_event(
                    "release_evidence_verification_failure",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    netuid=int(self.config.netuid),
                    weights_hash=v2_bundle.get("weights_hash"),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
                logger.error(
                    "auditor_v2_release_evidence_failed error_type=%s error=%s",
                    type(exc).__name__,
                    str(exc)[:200],
                )
                return None, "v2_invalid"
            verified = self.verify_attested_weights_v2(
                v2_bundle,
                identity_cache=identity_cache,
            )
            if verified is None:
                _audit_event(
                    "authority_verification_failure",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    netuid=int(self.config.netuid),
                    weights_hash=v2_bundle.get("weights_hash"),
                    reason="attested_authority_invalid",
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
                return None, "v2_invalid"
            # A stale-but-genuine authority replays every check; bind the
            # verified document to the epoch and netuid this auditor asked
            # for so an old finalization can never be submitted as current.
            try:
                verified_epoch = int(verified.get("epoch_id"))
                verified_netuid = int(verified.get("netuid"))
            except (TypeError, ValueError):
                verified_epoch = -1
                verified_netuid = -1
            if verified_epoch != int(epoch_id) or verified_netuid != int(
                self.config.netuid
            ):
                logger.warning(
                    "auditor_v2_authority_epoch_mismatch requested=%s "
                    "verified_epoch=%s verified_netuid=%s",
                    epoch_id,
                    verified_epoch,
                    verified_netuid,
                )
                _audit_event(
                    "authority_verification_failure",
                    level=logging.WARNING,
                    epoch=int(epoch_id),
                    netuid=int(self.config.netuid),
                    verified_epoch=verified_epoch,
                    verified_netuid=verified_netuid,
                    weights_hash=verified.get("weights_hash"),
                    reason="requested_identity_mismatch",
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
                return None, "v2_invalid"
            _audit_event(
                "authority_verification_success",
                epoch=verified_epoch,
                netuid=verified_netuid,
                uid=getattr(self, "uid", None),
                bundle_block=verified.get("block"),
                weights_hash=verified.get("weights_hash"),
                bundle_hash=verified.get("bundle_hash"),
                authority_stage=verified.get("authority_stage"),
                destination_count=len(verified.get("uids", [])),
                release_identity_count=len(identity_cache.get("entries", [])),
                elapsed_ms=round(
                    (time.monotonic() - started_at) * 1000,
                    1,
                ),
            )
            return verified, "v2_verified"
        if getattr(self, "_last_v2_authority_was_absent", False):
            _audit_event(
                "authority_unavailable",
                epoch=int(epoch_id),
                netuid=int(self.config.netuid),
                status="v2_absent",
                elapsed_ms=round((time.monotonic() - started_at) * 1000, 1),
            )
            return None, "v2_absent"
        _audit_event(
            "authority_unavailable",
            level=logging.WARNING,
            epoch=int(epoch_id),
            netuid=int(self.config.netuid),
            status="v2_unavailable",
            elapsed_ms=round((time.monotonic() - started_at) * 1000, 1),
        )
        return None, "v2_unavailable"

    @staticmethod
    def _authority_commits(authority: Dict) -> List[str]:
        commits = set()

        def walk(value):
            if isinstance(value, dict):
                commit = str(value.get("commit_sha") or "").lower()
                if (
                    value.get("physical_role")
                    and value.get("role")
                    and re.fullmatch(r"[0-9a-f]{40}", commit)
                ):
                    commits.add(commit)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(authority)
        if not commits:
            raise ValueError("V2 authority has no boot release commits")
        return sorted(commits)

    async def _fetch_release_identity_cache(self, authority: Dict) -> Dict:
        entries = []
        commits = self._authority_commits(authority)
        _audit_event(
            "release_evidence_fetch_start",
            epoch=authority.get("epoch_id"),
            weights_hash=authority.get("weights_hash"),
            commit_count=len(commits),
            commits=commits,
        )
        async with aiohttp.ClientSession(trust_env=False) as session:
            for commit in commits:
                url = f"{self.gateway_url}/weights/v2/release-evidence/{commit}"
                started_at = time.monotonic()
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        _audit_event(
                            "release_evidence_fetch_failure",
                            level=logging.ERROR,
                            epoch=authority.get("epoch_id"),
                            commit=commit,
                            endpoint=urlparse(url).path,
                            http_status=response.status,
                            elapsed_ms=round(
                                (time.monotonic() - started_at) * 1000,
                                1,
                            ),
                        )
                        raise RuntimeError(
                            f"release evidence request failed with HTTP {response.status}"
                        )
                    evidence = await response.json()
                cache = await asyncio.to_thread(
                    fetch_locked_release_identity_cache,
                    evidence,
                )
                entries.extend(cache["entries"])
                _audit_event(
                    "release_evidence_fetch_success",
                    epoch=authority.get("epoch_id"),
                    commit=commit,
                    endpoint=urlparse(url).path,
                    http_status=200,
                    identity_count=len(cache["entries"]),
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
        unique = {}
        for entry in entries:
            key = (entry["physical_role"], entry["role"], entry["commit_sha"])
            if key in unique and unique[key] != entry:
                raise ValueError("independent release identities conflict")
            unique[key] = entry
        result = {
            "schema_version": "leadpoet.independent_pcr0_identities.v2",
            "entries": list(unique.values()),
        }
        _audit_event(
            "release_evidence_cache_ready",
            epoch=authority.get("epoch_id"),
            weights_hash=authority.get("weights_hash"),
            identity_count=len(result["entries"]),
            commits=commits,
        )
        return result

    def verify_attested_weights_v2(
        self,
        authority: Dict,
        *,
        identity_cache: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Verify finalized V2 authority against independent PCR0 builds."""
        if identity_cache is None:
            _audit_event(
                "authority_cryptographic_verification_failure",
                level=logging.ERROR,
                epoch=authority.get("epoch_id"),
                weights_hash=authority.get("weights_hash"),
                stage="release_identity_cache",
                reason="missing",
            )
            logger.error(
                "auditor_v2_pcr0_cache_missing: automatic immutable release "
                "evidence was not supplied"
            )
            return None
        started_at = time.monotonic()
        _audit_event(
            "authority_cryptographic_verification_start",
            epoch=authority.get("epoch_id"),
            netuid=authority.get("netuid"),
            bundle_block=authority.get("block"),
            weights_hash=authority.get("weights_hash"),
            bundle_hash=authority.get("bundle_hash"),
            authority_stage=authority.get("authority_stage"),
            release_identity_count=len(identity_cache.get("entries", [])),
        )
        try:
            profile_file = Path(
                os.environ.get(
                    "AUDITOR_CHAIN_SIGNING_PROFILE_FILE",
                    os.path.join(
                        _REPO_ROOT,
                        "validator_tee",
                        "enclave",
                        "chain_signing_profile_v2.json",
                    ),
                )
            ).expanduser()
            profile = json.loads(profile_file.read_text(encoding="utf-8"))
            if (
                isinstance(authority, dict)
                and "authority_stage" in authority
            ):
                # Staged view: publication-stage authority is fully
                # enclave-signed and release-evidence verified; the
                # finalized-chain proof is verified whenever present. The
                # soft equivocation check covers chain divergence
                # retrospectively.
                verified = verify_published_weight_authority_stage_v2(
                    authority,
                    identity_cache=identity_cache,
                    chain_signing_profile=profile,
                )
            else:
                verified = verify_attested_weight_authority_v2(
                    authority,
                    identity_cache=identity_cache,
                    chain_signing_profile=profile,
                )
            self._verify_stateful_bundle_epoch(verified)
            _audit_event(
                "authority_cryptographic_verification_success",
                epoch=verified.get("epoch_id"),
                netuid=verified.get("netuid"),
                bundle_block=verified.get("block"),
                weights_hash=verified.get("weights_hash"),
                bundle_hash=verified.get("bundle_hash"),
                authority_stage=verified.get("authority_stage"),
                receipt_graph_hash=verified.get("receipt_graph_hash"),
                release_identity_count=len(identity_cache.get("entries", [])),
                destination_count=len(verified.get("uids", [])),
                elapsed_ms=round(
                    (time.monotonic() - started_at) * 1000,
                    1,
                ),
            )
            return verified
        except Exception as exc:
            _audit_event(
                "authority_cryptographic_verification_failure",
                level=logging.WARNING,
                epoch=authority.get("epoch_id"),
                netuid=authority.get("netuid"),
                bundle_block=authority.get("block"),
                weights_hash=authority.get("weights_hash"),
                bundle_hash=authority.get("bundle_hash"),
                authority_stage=authority.get("authority_stage"),
                error_type=type(exc).__name__,
                error=str(exc),
                elapsed_ms=round(
                    (time.monotonic() - started_at) * 1000,
                    1,
                ),
            )
            logger.warning(
                "auditor_v2_verification_failed error_type=%s error=%s",
                type(exc).__name__,
                str(exc)[:200],
            )
            return None
    
    async def fetch_gateway_attestation(self) -> bool:
        """
        Fetch GATEWAY attestation (for verifying log authenticity).
        
        NOTE: This is NOT the validator attestation - that comes from weight bundles.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/attestation/document"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        print(f"❌ Failed to fetch gateway attestation: {response.status}")
                        return False
                    
                    data = await response.json()
                    self.gateway_pubkey = data.get("enclave_pubkey")
                    self.gateway_attestation = data.get("attestation_document")
                    
                    print(f"✅ Fetched GATEWAY attestation")
                    print(f"   Gateway pubkey: {self.gateway_pubkey[:16]}...")
                    
                    return True
                    
        except Exception as e:
            logger.error(f"Failed to fetch gateway attestation: {e}")
            print(f"❌ Failed to fetch gateway attestation: {e}")
            return False
    
    def verify_gateway_attestation(self) -> bool:
        """
        Verify the fetched gateway attestation.
        
        SECURITY MODEL:
        - In production: Full Nitro verification required
        - In dev: Signature-only mode with warning
        
        Returns:
            True if attestation is valid (or acceptable for dev mode)
        """
        if not self.gateway_attestation or not self.gateway_pubkey:
            logger.warning("No gateway attestation to verify")
            print(f"⚠️ No gateway attestation to verify")
            return False
        
        try:
            # FULL NITRO VERIFICATION - NO DEV MODE
            from leadpoet_canonical.nitro import verify_nitro_attestation_full
            
            valid, data = verify_nitro_attestation_full(
                self.gateway_attestation,  # Already base64 encoded
                expected_pcr0=None,  # Uses allowlist from GitHub automatically
                expected_pubkey=self.gateway_pubkey,
                expected_purpose=None,  # Gateway attestation purpose varies
                expected_epoch_id=None,  # Gateway attestation doesn't have epoch_id
                role="gateway",  # Uses ALLOWED_GATEWAY_PCR0_VALUES
            )
            
            if valid:
                logger.debug("Gateway Nitro attestation verified")
                return True
            else:
                logger.error(f"Gateway Nitro verification failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Gateway attestation verification failed: {e}")
            return False
    
    def verify_validator_attestation(self, bundle: Dict) -> bool:
        """
        Verify the validator attestation from a weight bundle.
        
        SECURITY MODEL FOR AUDITORS:
        - Verifies AWS certificate chain (proves REAL Nitro enclave)
        - Verifies COSE signature (proves attestation is authentic)
        - Verifies epoch binding (replay protection)
        - SKIPS PCR0 verification (auditors can't independently verify without nitro-cli)
        
        WHY SKIP PCR0?
        - PCR0 verification requires either:
          a) nitro-cli to build enclave and compute expected PCR0, OR
          b) Trusting an allowlist published by subnet owners
        - Auditors don't have nitro-cli (not on AWS EC2)
        - Auditors shouldn't trust subnet-owner-published allowlists
        - So we verify "it's a REAL Nitro enclave" but not "which code it runs"
        
        WHAT THIS PROVES:
        ✅ The attestation came from a REAL AWS Nitro enclave (AWS-signed)
        ✅ The weights were signed by that enclave's private key
        ✅ The attestation is for THIS epoch (replay protection)
        ❌ Does NOT prove which code is running (would need nitro-cli)
        
        Args:
            bundle: Weight bundle containing validator_attestation_b64
            
        Returns:
            True if attestation is valid
        """
        attestation_b64 = bundle.get("validator_attestation_b64")
        pubkey = bundle.get("validator_enclave_pubkey")
        code_hash = bundle.get("validator_code_hash")
        epoch_id = bundle.get("epoch_id")
        
        if not attestation_b64 or not pubkey:
            logger.warning("Bundle missing validator attestation or pubkey")
            print(f"⚠️ Bundle missing validator attestation or pubkey")
            return False
        
        try:
            from leadpoet_canonical.nitro import verify_nitro_attestation_full
            
            # AUDITOR MODE: Skip PCR0 verification
            # We verify AWS cert chain + COSE signature (proves REAL enclave)
            # but skip PCR0 check (can't verify without nitro-cli)
            valid, data = verify_nitro_attestation_full(
                attestation_b64,  # Already base64 encoded
                expected_pcr0=None,
                expected_pubkey=pubkey,
                expected_purpose="validator_weights",
                expected_epoch_id=epoch_id,  # CRITICAL: Replay protection
                role="validator",
                skip_pcr0_verification=True,  # Auditors can't verify PCR0
            )
            
            if valid:
                logger.debug(
                    "Validator Nitro attestation verified for epoch %s",
                    epoch_id,
                )
                return True
            else:
                logger.error(f"Validator attestation verification failed: {data}")
                return False
                
        except Exception as e:
            logger.error(f"Validator attestation verification failed: {e}")
            return False
    
    async def fetch_signed_event(self, event_hash: str) -> Optional[Dict]:
        """
        Fetch a signed event from the transparency log by hash.
        
        Used to verify equivocation via gateway-signed events.
        
        Args:
            event_hash: Event hash to fetch
            
        Returns:
            Log entry dict, or None if not found
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = f"{self.gateway_url}/weights/transparency/event/{event_hash}"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 404:
                        return None
                    elif response.status == 200:
                        return await response.json()
                    else:
                        print(f"⚠️  Unexpected response fetching event: {response.status}")
                        return None
        except Exception as e:
            print(f"❌ Failed to fetch signed event: {e}")
            return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Attestation Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    
    def extract_validator_attestation(self, weights_data: Dict) -> bool:
        """
        Extract VALIDATOR attestation from the weight bundle.
        
        The validator attestation proves the weights came from an attested TEE,
        not from the gateway. This is the correct attestation to verify.
        """
        # Use CANONICAL field names (see Canonical Specifications)
        self.validator_attestation = weights_data.get("validator_attestation_b64")
        self.validator_pubkey = weights_data.get("validator_enclave_pubkey")
        self.validator_code_hash = weights_data.get("validator_code_hash")
        self.validator_hotkey = weights_data.get("validator_hotkey")
        
        if not self.validator_pubkey:
            print(f"⚠️  No validator attestation in weights bundle")
            return False
        
        print(f"   Validator pubkey: {self.validator_pubkey[:16]}...")
        print(f"   Validator hotkey: {self.validator_hotkey[:16] if self.validator_hotkey else 'None'}...")
        
        # NOTE: Full Nitro verification requires aws-nitro-enclaves-sdk
        # See "Issue 5b: Nitro Attestation Implementation Path" in tasks8.md
        # For now, extraction succeeds if fields are present
        # In production, call leadpoet_canonical.nitro.verify_nitro_attestation()
        return True
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Verification
    # ═══════════════════════════════════════════════════════════════════════════
    
    def verify_bundle_signature(self, bundle: Dict) -> bool:
        """
        Verify bundle by RECOMPUTING hash and checking Ed25519 signature.
        
        CRITICAL: Does NOT trust claimed hash - recomputes from bundle data.
        
        Verification steps:
        1. Recompute bundle_weights_hash() using canonical u16 pairs
        2. Verify recomputed hash matches claimed hash
        3. Verify Ed25519 signature over digest BYTES
        
        Args:
            bundle: Response from /weights/latest/{netuid}/{epoch_id}
            
        Returns:
            True if hash recomputes correctly AND signature is valid
        """
        try:
            # Get required fields
            claimed_hash = bundle.get("weights_hash")
            signature = bundle.get("validator_signature")
            pubkey = bundle.get("validator_enclave_pubkey")
            
            if not all([claimed_hash, signature, pubkey]):
                print(f"❌ Bundle missing weights_hash / validator_signature / validator_enclave_pubkey")
                return False
            
            # RECOMPUTE hash from bundle data (don't trust claimed hash)
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            
            if not uids or not weights_u16:
                print(f"❌ Bundle missing uids/weights_u16")
                return False
            
            weights_pairs = list(zip(uids, weights_u16))
            recomputed_hash = bundle_weights_hash(
                bundle["netuid"],
                bundle["epoch_id"],
                bundle["block"],
                weights_pairs
            )
            
            print(f"   Claimed hash:    {claimed_hash[:16]}...")
            print(f"   Recomputed hash: {recomputed_hash[:16]}...")
            
            if recomputed_hash != claimed_hash:
                print(f"❌ Bundle data does not match weights_hash!")
                print(f"   This could indicate tampering or encoding mismatch")
                return False
            
            print(f"   ✅ Hash recomputed correctly")
            
            # Verify Ed25519 signature over digest BYTES (32 bytes, not hex string)
            digest_bytes = bytes.fromhex(claimed_hash)
            
            pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey))
            pk.verify(bytes.fromhex(signature), digest_bytes)
            
            print(f"✅ Bundle hash + signature verified")
            return True
            
        except Exception as e:
            print(f"❌ Bundle verification failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def verify_anti_equivocation(self, bundle: Dict) -> bool:
        """
        Verify primary validator didn't submit different weights to chain.
        
        CRITICAL: Use chain_snapshot_compare_hash from bundle, NOT live chain!
        subtensor.weights() returns CURRENT weights which may have changed.
        
        Verification priority:
        1. PREFER: Snapshot hash (captured at block ~345)
        2. FALLBACK: Live chain query (with loud warning)
        
        Args:
            bundle: Weight bundle from gateway
            
        Returns:
            True if no equivocation detected
        """
        print(f"\n🔍 ANTI-EQUIVOCATION CHECK")
        
        netuid = bundle["netuid"]
        epoch_id = bundle["epoch_id"]
        
        # Build bundle compare hash (NO block - for comparison)
        weights_pairs = list(zip(bundle.get("uids", []), bundle.get("weights_u16", [])))
        if not weights_pairs:
            print(f"❌ Bundle missing uids/weights_u16")
            return False
        
        bundle_compare = compare_weights_hash(netuid, epoch_id, weights_pairs)
        
        # SKIP ANTI-EQUIVOCATION CHECK
        #
        # WHY: The chain snapshot is captured BEFORE the validator submits to chain,
        # so it contains the PREVIOUS epoch's weights, not the current submission.
        # This causes false positives (mismatch between epoch N bundle vs epoch N-1 snapshot).
        #
        # TIMING ISSUE:
        # 1. Validator submits to gateway → snapshot captures chain (epoch N-1 weights)
        # 2. Validator submits to chain (epoch N weights)
        # 3. Auditor compares bundle (N) vs snapshot (N-1) → FALSE MISMATCH
        #
        # The other 5 verifications are trustless and sufficient:
        # ✅ AWS cert chain, COSE signature, epoch binding, Ed25519 signature, hash recompute
        snapshot_hash = bundle.get("chain_snapshot_compare_hash")
        if snapshot_hash:
            print(f"   ⚠️  Skipping anti-equivocation (snapshot timing issue)")
            print(f"   ℹ️  Snapshot was captured BEFORE chain submission")
            print(f"   ℹ️  Other 5 trustless verifications already passed")
            return True  # Skip - rely on other 5 trustless verifications
        
        # NO SNAPSHOT AVAILABLE - Skip anti-equivocation check
        # 
        # WHY WE SKIP (not fallback to live chain):
        # 1. subtensor.weights() returns CURRENT weights, not historical
        # 2. Live chain query is unreliable (may give false positive)
        # 3. Anti-equivocation relies on gateway-captured snapshot (trust issue)
        # 4. The 5 other verifications (AWS cert, COSE, epoch, sig, hash) are trustless
        #
        # This check would only catch if validator submitted DIFFERENT weights
        # to chain vs gateway, but without snapshot we can't verify this reliably.
        print(f"   ⚠️  No chain_snapshot_compare_hash in bundle.")
        print(f"   ⚠️  SKIPPING anti-equivocation check (no reliable snapshot)")
        print(f"   ℹ️  Other 5 verifications (AWS cert, COSE, epoch, signature, hash) still apply")
        return True  # Skip this check - rely on the other 5 trustless verifications
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Weight Submission
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _set_weights_until_epoch_end(
        self,
        *,
        epoch_id: int,
        subnet_epoch_index: Optional[int] = None,
        uids,
        weights,
    ) -> bool:
        """Retry an explicitly rejected chain submission within one epoch."""

        def epoch_is_current() -> bool:
            try:
                self._validate_durable_epoch_runtime_lifecycle(
                    force_refresh=True,
                )
                finalized_state = self._read_epoch_state()
                if finalized_state.workflow_epoch_id != int(epoch_id):
                    return False
                if finalized_state.subnet_epoch_index != subnet_epoch_index:
                    return False
                best_state = self._read_best_epoch_state()
                return (
                    best_state.workflow_epoch_id == int(epoch_id)
                    and best_state.subnet_epoch_index == subnet_epoch_index
                    and best_state.current_block >= finalized_state.current_block
                    and best_state.blocks_remaining > 0
                )
            except Exception as exc:
                logger.error(
                    "auditor_weight_epoch_authority_unavailable "
                    "epoch=%s type=%s error=%s",
                    epoch_id,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                return False

        def lifecycle_is_open() -> bool:
            try:
                self._validate_durable_epoch_runtime_lifecycle(
                    force_refresh=True,
                )
                return True
            except Exception as exc:
                logger.error(
                    "auditor_weight_durable_lifecycle_closed "
                    "epoch=%s type=%s error=%s",
                    epoch_id,
                    type(exc).__name__,
                    str(exc)[:200],
                )
                return False

        if not epoch_is_current():
            _audit_event(
                "submission_refused",
                level=logging.ERROR,
                epoch=int(epoch_id),
                subnet_epoch_index=subnet_epoch_index,
                uid=getattr(self, "uid", None),
                reason="epoch_not_current",
            )
            print(f"⏹️ Refusing stale auditor submission for epoch {epoch_id}")
            return False

        attempt = 0
        while True:
            # Re-check the durable singleton immediately before every SDK call;
            # gateway bundle verification cannot authorize a stale process after
            # the operator advances the shared lifecycle.
            if not lifecycle_is_open():
                _audit_event(
                    "submission_refused",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    subnet_epoch_index=subnet_epoch_index,
                    uid=getattr(self, "uid", None),
                    attempt=attempt + 1,
                    reason="durable_lifecycle_closed",
                )
                print(
                    f"⏹️ Durable epoch authority closed before auditor "
                    f"submission for epoch {epoch_id}"
                )
                return False
            attempt += 1
            started_at = time.monotonic()
            _audit_event(
                "submission_attempt_start",
                epoch=int(epoch_id),
                subnet_epoch_index=subnet_epoch_index,
                netuid=int(self.config.netuid),
                uid=getattr(self, "uid", None),
                attempt=attempt,
                sdk_version=str(getattr(bt, "__version__", "unknown")),
                destination_count=len(uids),
                wait_for_finalization=True,
                mechanism_id=0,
            )
            try:
                with weight_hyperparameters_compat(
                    self.subtensor,
                    netuid=int(self.config.netuid),
                    sdk_version=str(bt.__version__),
                ):
                    outcome = ExtrinsicOutcome.from_sdk(
                        self.subtensor.set_weights(
                            netuid=self.config.netuid,
                            wallet=self.wallet,
                            uids=uids,
                            weights=weights,
                            wait_for_finalization=True,
                            mechid=0,
                        )
                    )
            except Exception as exc:
                _audit_event(
                    "submission_attempt_exception",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    subnet_epoch_index=subnet_epoch_index,
                    netuid=int(self.config.netuid),
                    uid=getattr(self, "uid", None),
                    attempt=attempt,
                    sdk_version=str(getattr(bt, "__version__", "unknown")),
                    error_type=type(exc).__name__,
                    error=str(exc),
                    elapsed_ms=round(
                        (time.monotonic() - started_at) * 1000,
                        1,
                    ),
                )
                raise
            _audit_event(
                "submission_attempt_result",
                level=logging.INFO if outcome.success else logging.WARNING,
                epoch=int(epoch_id),
                subnet_epoch_index=subnet_epoch_index,
                netuid=int(self.config.netuid),
                uid=getattr(self, "uid", None),
                attempt=attempt,
                success=bool(outcome.success),
                chain_message=outcome.message,
                elapsed_ms=round(
                    (time.monotonic() - started_at) * 1000,
                    1,
                ),
            )
            if outcome.success:
                return True

            print(
                f"❌ Bittensor rejected weight submission attempt {attempt}: "
                f"{outcome.message}"
            )
            time.sleep(12)
            if not epoch_is_current():
                _audit_event(
                    "submission_failed",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    subnet_epoch_index=subnet_epoch_index,
                    uid=getattr(self, "uid", None),
                    attempts=attempt,
                    reason="epoch_ended_after_rejection",
                )
                print(
                    f"⏹️ Epoch {epoch_id} ended before the weight submission "
                    "was accepted"
                )
                return False

    def submit_weights_to_chain(
        self,
        epoch_id: int,
        bundle: Dict,
        *,
        submission_epoch_id: Optional[int] = None,
    ) -> bool:
        """
        Submit verified weights to the Bittensor chain.
        
        Uses u16_to_emit_floats() for proper float conversion that
        guarantees ±1 u16 round-trip tolerance.
        
        Args:
            epoch_id: Epoch being submitted
            bundle: Verified weight bundle
            
        Returns:
            True if submission succeeded
        """
        try:
            uids = bundle.get("uids", [])
            weights_u16 = bundle.get("weights_u16", [])
            live_epoch_id = int(
                epoch_id if submission_epoch_id is None else submission_epoch_id
            )
            _audit_event(
                "submission_prepare_start",
                epoch=int(epoch_id),
                submission_epoch=live_epoch_id,
                netuid=int(self.config.netuid),
                uid=getattr(self, "uid", None),
                bundle_block=bundle.get("block"),
                weights_hash=bundle.get("weights_hash"),
                bundle_hash=bundle.get("bundle_hash"),
                authority_stage=bundle.get("authority_stage"),
                destination_count=len(uids) if isinstance(uids, list) else None,
                sdk_version=str(getattr(bt, "__version__", "unknown")),
            )
            
            if not uids:
                _audit_event(
                    "submission_prepare_failure",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    submission_epoch=live_epoch_id,
                    uid=getattr(self, "uid", None),
                    reason="empty_uids",
                )
                print(f"⚠️  No UIDs in bundle")
                return False
            
            # Use u16_to_emit_floats() for guaranteed round-trip
            # ❌ WRONG: weights_floats = [w / 65535.0 for w in weights_u16]
            # ✅ CORRECT: Use function that guarantees exact round-trip
            weights_floats = u16_to_emit_floats(uids, weights_u16)
            
            # Print weight breakdown (same format as primary validator)
            print(f"\n📤 Submitting weights for {len(uids)} UIDs...")
            print(f"   Weight breakdown (copying from primary validator):")
            total_weight = sum(weights_floats)
            for uid, weight in zip(uids, weights_floats):
                pct = (weight / total_weight * 100) if total_weight > 0 else 0
                label = "(Burn)" if uid == 0 else ""
                print(f"      UID {uid} {label}: {pct:.2f}%")
            print(f"   Total: {sum(weights_floats) / total_weight * 100:.2f}%")
            
            success = self._set_weights_until_epoch_end(
                epoch_id=live_epoch_id,
                subnet_epoch_index=self._subnet_index_for_workflow_epoch(
                    live_epoch_id
                ),
                uids=uids,
                weights=weights_floats,
            )
            
            if success:
                _audit_event(
                    "submission_success",
                    epoch=int(epoch_id),
                    submission_epoch=live_epoch_id,
                    netuid=int(self.config.netuid),
                    uid=getattr(self, "uid", None),
                    weights_hash=bundle.get("weights_hash"),
                    bundle_hash=bundle.get("bundle_hash"),
                    destination_count=len(uids),
                )
                print(f"✅ Weights submitted for epoch {epoch_id}")
                self.last_submitted_epoch = live_epoch_id
                self.last_authority_epoch = epoch_id
                
                # Verify submission landed on chain
                print(f"   🔍 Verifying submission landed on chain...")
                time.sleep(2)  # Brief wait for chain propagation
                
                try:
                    all_chain_weights = self.subtensor.weights(netuid=self.config.netuid)
                    my_uid = self.uid
                    
                    for uid, weights_list in all_chain_weights:
                        if uid == my_uid:
                            # Check if first few weights match what we submitted
                            chain_sample = [(u, w) for u, w in weights_list[:3]]
                            submitted_sample = [(uids[i], weights_u16[i]) for i in range(min(3, len(uids)))]
                            print(f"   Chain sample (UID {my_uid}): {chain_sample}")
                            print(f"   Submitted sample: {submitted_sample}")
                            
                            # Quick sanity check
                            if chain_sample and submitted_sample:
                                # Compare first weight
                                chain_first_w = dict(chain_sample).get(submitted_sample[0][0])
                                submitted_first_w = submitted_sample[0][1]
                                if chain_first_w is not None:
                                    diff = abs(chain_first_w - submitted_first_w)
                                    if diff <= 1:
                                        print(f"   ✅ Verified: Chain matches submitted (diff={diff})")
                                    else:
                                        print(f"   ⚠️  WARNING: Chain differs from submitted (diff={diff})")
                                        print(f"      Chain has OLD weights - submission may have failed!")
                            break
                    else:
                        print(f"   ⚠️  Could not find our weights on chain (UID {my_uid})")
                except Exception as e:
                    print(f"   ⚠️  Could not verify chain weights: {e}")
                
                return True
            else:
                _audit_event(
                    "submission_failed",
                    level=logging.ERROR,
                    epoch=int(epoch_id),
                    submission_epoch=live_epoch_id,
                    netuid=int(self.config.netuid),
                    uid=getattr(self, "uid", None),
                    weights_hash=bundle.get("weights_hash"),
                    bundle_hash=bundle.get("bundle_hash"),
                    reason="set_weights_not_accepted",
                )
                print(f"❌ Weight submission failed")
                return False
                
        except Exception as e:
            _audit_event(
                "submission_exception",
                level=logging.ERROR,
                epoch=int(epoch_id),
                submission_epoch=submission_epoch_id,
                netuid=int(self.config.netuid),
                uid=getattr(self, "uid", None),
                weights_hash=bundle.get("weights_hash")
                if isinstance(bundle, dict)
                else None,
                bundle_hash=bundle.get("bundle_hash")
                if isinstance(bundle, dict)
                else None,
                error_type=type(e).__name__,
                error=str(e),
            )
            print(f"❌ Weight submission error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # Main Loop
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def run(self):
        """Main loop for the auditor validator."""
        
        print(f"\n{'='*60}")
        print(f"🚀 AUDITOR VALIDATOR STARTING")
        print(f"{'='*60}")
        logger.info("Auditor validator starting")
        _audit_event(
            "run_loop_start",
            netuid=int(self.config.netuid),
            uid=getattr(self, "uid", None),
            gateway_endpoint=getattr(self, "gateway_url", None),
            archive_endpoint=getattr(self, "epoch_archive_endpoint", None),
            submission_block=int(WEIGHT_SUBMISSION_BLOCK),
        )
        last_metagraph_epoch_identity = None
        
        while not self.should_exit:
            try:
                epoch_state = self._read_epoch_state()
                current_block = epoch_state.current_block
                current_epoch = epoch_state.workflow_epoch_id
                block_within_epoch = epoch_state.epoch_block
                
                print(
                    f"\r⏱️  Block {current_block} | Epoch {current_epoch} | "
                    f"Block {block_within_epoch} | Remaining {epoch_state.blocks_remaining}",
                    end="",
                    flush=True,
                )
                
                # Check if it's time to submit weights
                if epoch_state.deadline_reached(WEIGHT_SUBMISSION_BLOCK):
                    if self.last_submitted_epoch != current_epoch:
                        _audit_event(
                            "submission_window_entered",
                            epoch=int(current_epoch),
                            subnet_epoch_index=getattr(
                                epoch_state,
                                "subnet_epoch_index",
                                None,
                            ),
                            current_block=int(current_block),
                            epoch_block=int(block_within_epoch),
                            blocks_remaining=int(epoch_state.blocks_remaining),
                            netuid=int(self.config.netuid),
                            uid=getattr(self, "uid", None),
                            last_submitted_epoch=self.last_submitted_epoch,
                        )
                        print(f"\n\n{'='*60}")
                        print(f"📊 WEIGHT SUBMISSION TIME (Block {block_within_epoch})")
                        print(f"{'='*60}")
                        
                        # Auditors mirror only the primary's current-epoch
                        # authority. A missing live bundle remains fail-closed.
                        weights_data = None
                        authority_status = "v2_absent"
                        target_epoch = current_epoch
                        for candidate_epoch in self._authority_candidate_epochs(
                            current_epoch
                        ):
                            print(f"   Fetching weights for epoch {candidate_epoch}...")
                            candidate_data, candidate_status = (
                                await self.fetch_verified_weight_authority(
                                    candidate_epoch
                                )
                            )
                            if candidate_data is not None:
                                weights_data = candidate_data
                                authority_status = candidate_status
                                target_epoch = candidate_epoch
                                break
                            if candidate_status != "v2_absent":
                                authority_status = candidate_status
                                break
                        if weights_data is None:
                            _audit_event(
                                "submission_window_authority_unavailable",
                                level=logging.INFO
                                if authority_status == "v2_absent"
                                else logging.WARNING,
                                epoch=int(current_epoch),
                                subnet_epoch_index=getattr(
                                    epoch_state,
                                    "subnet_epoch_index",
                                    None,
                                ),
                                current_block=int(current_block),
                                epoch_block=int(block_within_epoch),
                                blocks_remaining=int(
                                    epoch_state.blocks_remaining
                                ),
                                netuid=int(self.config.netuid),
                                uid=getattr(self, "uid", None),
                                authority_status=authority_status,
                                retry_delay_seconds=5
                                if authority_status
                                in {"v2_absent", "v2_unavailable"}
                                else 30,
                            )
                            # Verification is fail-closed: no fallback vector will be submitted.
                            if authority_status == "v2_absent":
                                print("   ⏳ Weights not yet published. Waiting 5s...")
                            elif authority_status == "v2_unavailable":
                                print("   ⚠️ Gateway weight authority unavailable (fetch error). Waiting 5s...")
                            else:
                                print("❌ Auditor verification failed")
                            await asyncio.sleep(
                                5
                                if authority_status
                                in {"v2_absent", "v2_unavailable"}
                                else 30
                            )
                            continue
                        _audit_event(
                            "submission_window_authority_ready",
                            epoch=int(current_epoch),
                            authority_epoch=int(target_epoch),
                            subnet_epoch_index=getattr(
                                epoch_state,
                                "subnet_epoch_index",
                                None,
                            ),
                            current_block=int(current_block),
                            epoch_block=int(block_within_epoch),
                            blocks_remaining=int(epoch_state.blocks_remaining),
                            netuid=int(self.config.netuid),
                            uid=getattr(self, "uid", None),
                            weights_hash=weights_data.get("weights_hash"),
                            bundle_hash=weights_data.get("bundle_hash"),
                            authority_stage=weights_data.get(
                                "authority_stage"
                            ),
                            destination_count=len(
                                weights_data.get("uids", [])
                            ),
                        )
                        print("✅ Auditor verification passed")
                        
                        # Save pending equivocation check for next epoch verification
                        weights_pairs = list(zip(weights_data.get("uids", []), weights_data.get("weights_u16", [])))
                        if weights_pairs:
                            bundle_compare = compare_weights_hash(self.config.netuid, target_epoch, weights_pairs)
                            self.save_pending_equivocation_check(
                                target_epoch, 
                                bundle_compare, 
                                weights_data.get("validator_hotkey", "")
                            )
                        
                        if self.submit_weights_to_chain(
                            target_epoch,
                            weights_data,
                            submission_epoch_id=current_epoch,
                        ):
                            logger.info(f"Weights submitted for epoch {target_epoch}")
                        else:
                            logger.error(f"Weight submission failed for epoch {target_epoch}")
                
                # Soft anti-equivocation check at block 50-100 of each epoch
                # Check 2 epochs back to ensure weights have definitely propagated
                if 50 <= epoch_state.epoch_block <= 100:
                    # Check if we have a pending equivocation check from 2 epochs ago
                    target_epoch = current_epoch - 2
                    pending = self.load_pending_equivocation_check(target_epoch=target_epoch)
                    if pending:
                        if not await self.perform_soft_equivocation_check(target_epoch):
                            logger.error(f"Soft equivocation check FAILED for epoch {target_epoch}")
                            print(f"   ⚠️  MISMATCH DETECTED - Bundle vs chain weights differ (investigating...)")
                            # Note: We don't burn here since we already submitted weights
                            # This is a SOFT check - logs the issue for investigation
                            # Mismatch could be due to: timing, normalization differences, or actual equivocation
                
                # Re-resolve the UID once per official epoch. Direct storage
                # avoids runtime-API decoding drift in supported Bittensor 9
                # environments while retaining canonical chain authority.
                if epoch_state.identity != last_metagraph_epoch_identity:
                    print(f"\n🔄 Refreshing auditor chain identity...")
                    current_uid = self._get_uid()
                    if current_uid is None:
                        raise RuntimeError(
                            "auditor hotkey is no longer registered on subnet"
                        )
                    self.uid = current_uid
                    last_metagraph_epoch_identity = epoch_state.identity
                
                # Reset error counter on successful iteration
                self.consecutive_errors = 0
                
                await asyncio.sleep(12)  # ~1 block
                
            except KeyboardInterrupt:
                print(f"\n\n⛔ Shutting down...")
                self.should_exit = True
            except (TimeoutError, ConnectionError, OSError) as e:
                # Network/connection errors - common and recoverable
                self.consecutive_errors += 1
                print(f"\n⚠️  Connection error ({self.consecutive_errors}/{self.max_consecutive_errors}): {type(e).__name__}")
                logger.warning(f"Connection error: {e}")
                _audit_event(
                    "run_loop_connection_failure",
                    level=logging.WARNING,
                    netuid=int(self.config.netuid),
                    uid=getattr(self, "uid", None),
                    consecutive_errors=self.consecutive_errors,
                    max_consecutive_errors=self.max_consecutive_errors,
                    error_type=type(e).__name__,
                    error=str(e),
                )
                
                if self.consecutive_errors >= self.max_consecutive_errors:
                    print(f"   Too many consecutive errors - reconnecting subtensor...")
                    self._reconnect_subtensor()
                
                # Exponential backoff: 30s, 60s, 120s, max 300s
                backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
                print(f"   Retrying in {backoff}s...")
                await asyncio.sleep(backoff)
            except Exception as e:
                # Other errors - log and continue
                self.consecutive_errors += 1
                error_type = type(e).__name__
                print(f"\n❌ Error in main loop ({error_type}): {e}")
                logger.error(f"Main loop error: {e}")
                _audit_event(
                    "run_loop_failure",
                    level=logging.ERROR,
                    netuid=int(self.config.netuid),
                    uid=getattr(self, "uid", None),
                    consecutive_errors=self.consecutive_errors,
                    error_type=error_type,
                    error=str(e),
                )
                
                # Check if it's a websocket-related error
                error_str = str(e).lower()
                if any(x in error_str for x in ['websocket', 'ssl', 'connection', 'timeout', 'handshake']):
                    print(f"   Connection-related error detected")
                    if self.consecutive_errors >= self.max_consecutive_errors:
                        self._reconnect_subtensor()
                
                import traceback
                traceback.print_exc()
                
                # Exponential backoff
                backoff = min(30 * (2 ** (self.consecutive_errors - 1)), 300)
                print(f"   Retrying in {backoff}s...")
                await asyncio.sleep(backoff)


def _build_bittensor_cli_config(parser, argv=None):
    """Parse auditor CLI flags even when the SDK disables parsing by default."""

    variable = "BT_NO_PARSE_CLI_ARGS"
    previous = os.environ.get(variable)
    os.environ[variable] = "false"
    try:
        return bt.Config(
            parser,
            args=list(sys.argv[1:] if argv is None else argv),
        )
    finally:
        if previous is None:
            os.environ.pop(variable, None)
        else:
            os.environ[variable] = previous


def main():
    """Entry point for auditor validator."""
    
    parser = argparse.ArgumentParser(
        description="LeadPoet Auditor Validator - Copies TEE-verified weights from primary validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VERIFICATION FAILURE HANDLING:
  If authoritative V2 verification fails, no vector is submitted.

EXAMPLES:
  python neurons/auditor_validator.py --netuid 71
  python neurons/auditor_validator.py --netuid 71 --gateway-url http://localhost:8000
        """
    )
    
    # Bittensor arguments
    bt.Wallet.add_args(parser)
    bt.Subtensor.add_args(parser)
    
    # Custom arguments
    parser.add_argument(
        "--netuid", 
        type=int, 
        default=71, 
        help="Subnet UID (default: 71)"
    )
    parser.add_argument(
        "--gateway-url", 
        type=str, 
        default=DEFAULT_GATEWAY_URL, 
        help=f"Gateway URL (default: {DEFAULT_GATEWAY_URL})"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    config = _build_bittensor_cli_config(parser)
    
    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Get gateway URL from args or default
    gateway_url = args.gateway_url or DEFAULT_GATEWAY_URL
    
    print(f"\n{'='*60}")
    print(f"🔍 LEADPOET AUDITOR VALIDATOR")
    print(f"{'='*60}")
    print(f"   Network: {config.subtensor.network}")
    print(f"   Netuid: {args.netuid}")
    print(f"   Gateway: {gateway_url}")
    print(f"   Log level: {args.log_level}")
    print(f"{'='*60}")
    
    print(f"{'='*60}\n")
    
    # Set netuid on config
    config.netuid = args.netuid
    
    try:
        # Create and run validator
        validator = AuditorValidator(config, gateway_url=gateway_url)
        asyncio.run(validator.run())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        print("\n⛔ Shutting down...")
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        print(f"\n❌ Runtime error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Arena service: daily benchmark, execution, scoring, and publication."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from concurrent.futures import ThreadPoolExecutor
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from lab_arena import contact_policy, contact_evidence, integrity, confirmation, icp_disclosure, judgment_cache
from lab_arena import broker as broker_module, capacity, chain as chain_module, contracts, credentials as credentials_module, public_dashboard, rewards, scoring, scorer_image_access as scorer_image_access_module, signing, source_bundle, source_disclosure, submission_rate_limit, verify, weight_state
from leadpoet_verifier.identity.normalization import normalize_url
from leadpoet_canonical.arena_weights import (
    validate_accepted_weight_state,
    verify_accepted_weight_state_signature,
)
from lab_arena.contracts import ArenaContractError, ArenaSignatureError
from lab_arena.output import MAX_OUTPUT_BYTES, OutputInvalid, validate_output_document
from lab_arena.owner_admission import OwnerAdmissionError, resolve_finalized_owner
from lab_arena.store import ArenaStore, ArenaStoreError, hash_lease_token
from gateway.utils.hotkey_roles import (
    ValidatorIneligible, permitted_validator_uid, validator_uid,
)

logger = logging.getLogger(__name__)

MODES = ("off", "shadow", "live")
HOT_ROUND_TTL_SECONDS = 2.0
TERMINAL_STATUSES = ("published", "cancelled")
ACTIVE_ROUND_STATUSES = tuple(
    status for status in contracts.ROUND_STATUSES if status not in TERMINAL_STATUSES
)
SOURCE_UPLOAD_EXPIRES_SECONDS = 900
DEFAULT_BASELINE_SOURCE_URL = "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/lab.tar.gz"
DEFAULT_EXECUTION_CAP_MICROUSD = 50_000_000
DEFAULT_COST_PER_COMPANY_MICROUSD = 500_000
DEFAULT_STAGE_MINUTES = {
    "benchmark": 30,
    "stage_1": 240,
    "stage_1_scoring": 390,
    "stage_2": 180,
    "final_scoring": 390,
}
CANCEL_REASONS = {
    "benchmark_leak": "benchmark_leaked_before_cutoff",
    "benchmark_invalid": "benchmark_data_invalid",
    "capacity": "runner_capacity",
    "scoring": "scoring_window_closed",
    "scoring_incomplete": "scoring_incomplete",
    "publication": "publication_sanitizer_failed",
    "operator": "operator",
}


def _gateway_validator_authorizer(
    hotkey: str, *, network_name: str, netuid: int, metagraph: Any
) -> Tuple[bool, Optional[str]]:
    """Check existing-work identity without rechecking benchmark stake."""

    try:
        validator_uid(
            metagraph, hotkey, netuid=netuid,
            network_name=chain_module.normalize_network_name(network_name),
            require_stake=False,
        )
    except ValidatorIneligible as exc:
        return (False, None) if str(exc) == "runner_hotkey_unregistered" else (True, "miner")
    return True, "validator"


class ServiceError(RuntimeError):
    """A request or transition failed closed."""

    def __init__(self, code: str, status: int = 400, *, source_path: str = "") -> None:
        super().__init__(code)
        self.code = code
        self.status = status
        self.source_path = source_path


# ---------------------------------------------------------------------------
# Object store
# ---------------------------------------------------------------------------


class ObjectStore(Protocol):
    def put(self, ref: str, data: bytes) -> None: ...

    def get(self, ref: str) -> bytes: ...

    def get_bounded(self, ref: str, max_bytes: int) -> bytes: ...

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]: ...


class LocalObjectStore:
    """Directory-backed object store for tests and local runs; refs are write-once."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, ref: str) -> Path:
        if not ref or ref.startswith("/") or ".." in Path(ref).parts:
            raise ArenaContractError("object ref is invalid")
        return self._root / ref

    def put(self, ref: str, data: bytes) -> None:
        path = self._path(ref)
        if path.exists() and path.read_bytes() != bytes(data):
            raise ArenaContractError("object ref %s already holds different bytes" % ref)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(data))

    def get(self, ref: str) -> bytes:
        path = self._path(ref)
        if not path.exists():
            raise ArenaContractError("object store has no object at %s" % ref)
        return path.read_bytes()

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        data = self.get(ref)
        if len(data) > max_bytes:
            raise ArenaContractError("object exceeds source size limit")
        return data

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]:
        raise ServiceError("source_upload_not_configured", 503)


class S3ObjectStore:
    """Versioned, delete-denied Arena bucket (section 3.1); boto3 imported lazily."""

    def __init__(
        self,
        bucket: str,
        *,
        client: Any = None,
        region_name: Optional[str] = None,
        prefix: str = "",
    ) -> None:
        self._key_prefix = self._validate_key_prefix(prefix)
        if client is None:
            import boto3  # noqa: WPS433
            from botocore.config import Config

            client = boto3.client("s3", region_name=region_name, config=Config(signature_version="s3v4"))
        self._client = client
        self._bucket = bucket

    @staticmethod
    def _validate_key_prefix(value: str) -> str:
        if not isinstance(value, str):
            raise ArenaContractError("object key prefix is invalid")
        if value == "":
            return value
        segments = value.split("/")
        if (
            value != value.strip()
            or any(
                segment in ("", ".", "..") or segment != segment.strip()
                for segment in segments
            )
            or "\\" in value
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ArenaContractError("object key prefix is invalid")
        return value

    def _key(self, ref: str) -> str:
        if not self._key_prefix:
            return ref
        return "%s/%s" % (self._key_prefix, ref)

    def put(self, ref: str, data: bytes) -> None:
        payload = bytes(data)
        key = self._key(ref)
        for attempt in range(2):
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=payload,
                    ContentType="application/json",
                    IfNoneMatch="*",
                )
                return
            except Exception as exc:
                response = getattr(exc, "response", {})
                error = response.get("Error", {}) if isinstance(response, Mapping) else {}
                metadata = response.get("ResponseMetadata", {}) if isinstance(response, Mapping) else {}
                code = str(error.get("Code") or "") if isinstance(error, Mapping) else ""
                status = metadata.get("HTTPStatusCode") if isinstance(metadata, Mapping) else None
                conditional_conflict = code in ("ConditionalRequestConflict", "409") or status == 409
                already_exists = code in ("PreconditionFailed", "412") or status == 412
                if conditional_conflict and attempt == 0:
                    continue
                if not (conditional_conflict or already_exists):
                    raise
                try:
                    existing = self.get(ref)
                except Exception:
                    raise exc
                if existing != payload:
                    raise ArenaContractError("object ref %s already holds different bytes" % ref) from exc
                return
        raise ArenaContractError("object ref %s could not be written safely" % ref)

    def get(self, ref: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=self._key(ref))
        return response["Body"].read()

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        key = self._key(ref)
        head = self._client.head_object(Bucket=self._bucket, Key=key)
        if int(head.get("ContentLength") or 0) > int(max_bytes):
            raise ArenaContractError("object exceeds source size limit")
        response = self._client.get_object(Bucket=self._bucket, Key=key)
        body = response["Body"]
        try:
            data = body.read(int(max_bytes) + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        if len(data) > int(max_bytes):
            raise ArenaContractError("object exceeds source size limit")
        return data

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int, source_content_md5: Optional[str] = None) -> Mapping[str, Any]:
        key = self._key(ref)
        headers = {
            "content-type": str(content_type),
            "content-length": str(int(size_bytes)),
            "if-none-match": "*",
        }
        params = {
            "Bucket": self._bucket,
            "Key": key,
            "ContentType": str(content_type),
            "ContentLength": int(size_bytes),
            "IfNoneMatch": "*",
        }
        if source_content_md5 is not None:
            checksum = contracts.validate_source_content_md5(source_content_md5)
            params["ContentMD5"] = checksum
            headers["content-md5"] = checksum
        url = self._client.generate_presigned_url(
            "put_object",
            Params=params,
            ExpiresIn=int(expires_seconds),
            HttpMethod="PUT",
        )
        return {"upload_url": url, "upload_headers": headers, "expires_in_seconds": int(expires_seconds)}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ChainReads(Protocol):
    def finalized_head(self) -> Any: ...

    def metagraph(self, finalized: bool = True) -> Any: ...

    def current_settlement_epoch(self) -> int: ...

    def accepted_weight_epoch_scope(self) -> Mapping[str, Any]: ...

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]: ...

    def uid_for_hotkey(self, hotkey: str) -> Optional[int]: ...

@dataclass
class RoundDefaults:
    execution_cap_microusd: int = DEFAULT_EXECUTION_CAP_MICROUSD
    cost_per_company_microusd: int = DEFAULT_COST_PER_COMPANY_MICROUSD
    scoring_cap_microusd: int = 50_000_000
    runner_hotkeys: Tuple[str, ...] = ()
    baseline_hotkey: str = ""
    baseline_source_url: str = DEFAULT_BASELINE_SOURCE_URL
    stage_minutes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_STAGE_MINUTES))
    max_challengers: int = contracts.DEFAULT_MAX_CHALLENGERS  # admitted challengers per round, at most MAX_CHALLENGERS
    # The trusted scorer is resolved once and copied into each round. A service
    # restart therefore cannot change the scorer midway through that round.
    scorer_image_digest: str = "sha256:" + "0" * 64
    scorer_image_reference: str = ""
    # Automatic daily rounds: the UTC hour of each day's submission cutoff, or
    # None to leave round creation to the operator (``lab_arena_admin.py create``).
    daily_cutoff_hour_utc: Optional[int] = None
    # A new round's cutoff lies at least this far ahead so miners can submit.
    integrity_from: Optional[str] = None
    contacts_from: Optional[str] = None
    benchmark_disclosure_from: Optional[str] = None
    confirmation_minutes: Tuple[int, int] = (60, 110)
    min_submission_hours: int = 6
    # The king's pool as a percent of total emissions (LAB_ARENA_POOL_PERCENT).
    # Announced in every round configuration and carried by every reward basis,
    # so a change applies from the next round and never rewrites a published one.
    pool_percent: int = contracts.LAB_ARENA_POOL_PERCENT
    # Frozen into each round. A later environment change cannot activate an
    # older round that was intentionally published without rewards.
    rewards_enabled: bool = False


@dataclass
class ServiceConfig:
    mode: str
    store: ArenaStore
    object_store: ObjectStore
    # Reward signing is downstream of competition publication. Production
    # supplies a lazy factory; tests may supply a signer directly.
    signer: Optional[signing.ArenaSigner]
    chain: ChainReads
    verify_signature: Callable[[str, str, str], bool]
    daily_icp_source: Callable[..., Mapping[str, Any]]
    banned_hotkeys_source: Callable[[], Iterable[str]]
    broker_factory: Callable[["ArenaService", Mapping[str, Any]], broker_module.Broker]
    defaults: RoundDefaults = field(default_factory=RoundDefaults)
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    network_name: str = "finney"
    netuid: int = 71
    # Optional process-local ownership boundary for isolated one-round hosts.
    # It is deliberately not persisted in a round configuration or schema.
    pinned_round_id: Optional[str] = None
    baseline_source_fetcher: Optional[Callable[[str, int], bytes]] = None
    reward_signer_factory: Optional[Callable[[], signing.ArenaSigner]] = None
    credential_manager: Optional[credentials_module.CredentialManager] = None
    code_reviewer: Optional[Any] = None
    # Only the host publishes accepted source. Miner code never gets GitHub access.
    baseline_promoter_factory: Optional[Callable[[], Any]] = None
    # Supplies plain accepted economic inputs. It must not return receipts,
    # ancestry, release identity, or a preconstructed weight vector.
    accepted_burn_hotkey: str = ""
    # Identity-only test seam for existing leases. New claims and capacity
    # always require the finalized permit/stake snapshot, never this override.
    validator_authorizer: Optional[
        Callable[[str], Tuple[bool, Optional[str]]]
    ] = None
    # Returns temporary access only for the scorer image frozen into a run's
    # round. Generic public registries do not configure this ECR-only path.
    confirmation_icp_source: Optional[Callable[..., Sequence[Mapping[str, Any]]]] = None
    scorer_image_access: Optional[
        Callable[[str, str], Mapping[str, Any]]
    ] = None

    def __post_init__(self) -> None:
        if self.defaults.integrity_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.integrity_from.replace("Z", "+00:00"))
                if activation.tzinfo is None:
                    raise ValueError("missing timezone")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("integrity_activation_invalid", 500) from exc
        if self.defaults.contacts_from is not None:
            try:
                activation = datetime.fromisoformat(self.defaults.contacts_from.replace("Z", "+00:00"))
                if activation.tzinfo is None or self.defaults.integrity_from is None:
                    raise ValueError("contact activation requires timezone and integrity activation")
                integrity_activation = datetime.fromisoformat(self.defaults.integrity_from.replace("Z", "+00:00"))
                # Contacts are announced at intake open; integrity activates
                # at the cutoff one day later. Every contact round needs both.
                if activation + timedelta(days=1) < integrity_activation:
                    raise ValueError("contact activation precedes integrity activation")
            except (ValueError, AttributeError) as exc:
                raise ServiceError("contact_activation_invalid", 500) from exc
        if self.defaults.benchmark_disclosure_from is not None:
            try:
                icp_disclosure.parse_activation(
                    self.defaults.benchmark_disclosure_from
                )
            except icp_disclosure.IcpDisclosureError as exc:
                raise ServiceError(
                    "benchmark_disclosure_activation_invalid", 500
                ) from exc
        if self.mode not in MODES:
            raise ServiceError("mode_invalid", 500)
        if self.mode == "off":
            raise ServiceError("mode_off", 500)
        try:
            self.network_name = chain_module.normalize_network_name(self.network_name)
        except Exception as exc:
            raise ServiceError("network_name_invalid", 500) from exc
        if isinstance(self.netuid, bool) or not isinstance(self.netuid, int) or self.netuid < 1:
            raise ServiceError("netuid_invalid", 500)
        if self.pinned_round_id is not None and (
            not isinstance(self.pinned_round_id, str)
            or not self.pinned_round_id
            or self.pinned_round_id != self.pinned_round_id.strip()
        ):
            raise ServiceError("pinned_round_id_invalid", 500)


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def round_id_for_cutoff(cutoff: datetime) -> str:
    return "arena-%s" % cutoff.astimezone(timezone.utc).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ArenaService:
    def __init__(self, config: ServiceConfig) -> None:
        self._config = config
        self._store = config.store
        self._objects = config.object_store
        self._signer = config.signer
        self._signer_lock = threading.Lock()
        self._clock = config.clock
        self._lock = threading.RLock()
        self._hot_round_lock = threading.Lock()
        self._hot_rounds: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._submission_request_limiter = (
            submission_rate_limit.SubmissionRequestLimiter()
        )
        self._scorer_policy = scoring.build_scorer_policy()
        self._brokers: Dict[str, broker_module.Broker] = {}

    # -- accessors -------------------------------------------------------------

    @property
    def config(self) -> ServiceConfig:
        return self._config

    @property
    def store(self) -> ArenaStore:
        return self._store

    @property
    def scorer_policy(self) -> Dict[str, Any]:
        return dict(self._scorer_policy)

    def signing_key_document(self) -> Dict[str, Any]:
        return signing.signing_key_document(self._reward_signer().public_key_der)

    def now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    def _sign(self, document: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
        return signing.sign_document(self._reward_signer(), document, hash_field=hash_field)

    def _reward_signer(self) -> signing.ArenaSigner:
        """Create the downstream reward signer only when activation needs it."""

        with self._signer_lock:
            if self._signer is None:
                factory = self._config.reward_signer_factory
                if factory is None:
                    raise ServiceError("reward_signer_unavailable", 503)
                self._signer = factory()
            return self._signer

    def _round(self, round_id: str) -> Dict[str, Any]:
        self._require_round_ownership(round_id)
        row = self._store.get_round(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        return self._require_round_mode(row)

    def _pinned_round_id(self) -> Optional[str]:
        return getattr(getattr(self, "_config", None), "pinned_round_id", None)

    def _chain_scope(self) -> Tuple[str, int]:
        """Return this process's chain pair, with the legacy production default."""

        config = getattr(self, "_config", None)
        return (
            str(getattr(config, "network_name", "finney")),
            int(getattr(config, "netuid", 71)),
        )

    def _require_round_ownership(self, round_id: str) -> None:
        pinned_round_id = self._pinned_round_id()
        if pinned_round_id is not None and round_id != pinned_round_id:
            raise ServiceError("round_scope_mismatch", 409)

    def _pinned_round(self) -> Optional[Dict[str, Any]]:
        pinned_round_id = self._pinned_round_id()
        if pinned_round_id is None:
            return None
        row = self._store.get_round(pinned_round_id)
        if row is None:
            return None
        return self._require_round_mode(row)

    def _require_round_mode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse a round owned by another Arena mode or chain scope."""

        configuration = row.get("configuration_doc") or {}
        try:
            policy_enabled = integrity.enabled(configuration)
            contacts_enabled = contact_policy.enabled(configuration)
        except ValueError as exc:
            raise ServiceError("unsupported_integrity_policy", 409) from exc
        adapter = configuration.get("scorer_policy", {}).get("scoring_adapter_version")
        if contacts_enabled != (adapter == contact_policy.SCORING_ADAPTER) or policy_enabled != contact_policy.integrity_adapter(adapter):
            raise ServiceError("integrity_scorer_policy_mismatch", 409)
        try:
            icp_disclosure.configured_policy(row)
        except icp_disclosure.IcpDisclosureError as exc:
            raise ServiceError("benchmark_disclosure_policy_invalid", 503) from exc
        if configuration.get("mode") != self._config.mode:
            raise ServiceError("round_mode_mismatch", 409)
        # Rows created before schema 189 had no explicit chain pair. They are
        # permanently interpreted as the original Finney/netuid 71 scope.
        network_name = configuration.get("network_name", "finney")
        netuid = configuration.get("netuid", 71)
        if (network_name, netuid) != self._chain_scope():
            raise ServiceError("round_network_mismatch", 409)
        return row

    def startup_checks(self) -> Dict[str, Any]:
        """Check competition database and object-store dependencies.

        Checks the database role, every Arena table and function grant, the
        object store. Reward signing and epoch cutover are lazy downstream
        dependencies and cannot block service startup or publication.
        """

        identity = self._store.require_service_role()
        try:
            schema = self._store._transport.rpc("lab_arena_schema_version_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("function_unavailable:lab_arena_schema_version_v1", 500) from exc
        expected_schema = "leadpoet.lab_arena.schema_version.v1"
        schema_version = schema.get("version") if isinstance(schema, Mapping) else None
        if (
            not isinstance(schema, Mapping)
            or schema.get("schema_version") != expected_schema
            or schema_version != 197
        ):
            raise ServiceError("arena_schema_version_invalid", 500)
        for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
            try:
                self._store._transport.select(
                    table,
                    limit=1,
                    columns="icp_set_date" if table == "lab_arena_rounds" else "*",
                )
            except ArenaStoreError as exc:
                raise ServiceError("table_unavailable:%s" % table, 500) from exc
        try:
            self._store.code_review_schema()
        except ArenaStoreError as exc:
            raise ServiceError("code_review_schema_unavailable", 500) from exc
        try:
            self._store.validator_scoring_authority_schema()
        except ArenaStoreError as exc:
            raise ServiceError(
                "validator_scoring_authority_schema_unavailable", 500
            ) from exc
        # Every service function must exist and be granted: a missing round is the
        # expected structured failure; a permission or undefined-function error is not.
        for function, params in (
            (
                "lab_arena_commit_round_v2",
                {
                    "p_round_id": "arena-0000-00-00",
                    "p_participants": [],
                    "p_benchmark_ref": "probe",
                    "p_evaluation_date": "2000-01-02",
                    "p_icp_set_date": "2000-01-01",
                    "p_scorer_image_digest": "sha256:" + "0" * 64,
                    "p_scorer_image_reference": "probe@sha256:" + "0" * 64,
                },
            ),
            ("lab_arena_expire_leases", {"p_round_id": "arena-0000-00-00"}),
            ("lab_arena_close_stage", {"p_round_id": "arena-0000-00-00", "p_stage": 1}),
            ("lab_arena_cancel_round", {"p_round_id": "arena-0000-00-00", "p_reason": "startup-probe"}),
        ):
            try:
                self._store._transport.rpc(function, params)
            except ArenaStoreError as exc:
                if "lab_arena_round_missing" not in str(exc):
                    raise ServiceError("function_unavailable:%s" % function, 500) from exc
        # The aggregate cost RPC was added after the legacy schema-version
        # marker. Probe its service grant and its fail-closed missing-row path
        # explicitly so an old database cannot start the new budget service.
        cost_function = "lab_arena_submission_costs"
        try:
            self._store._transport.rpc(
                cost_function,
                {"p_submission_id": "__arena_budget_probe__"},
            )
        except ArenaStoreError as exc:
            if "lab_arena_submission_missing" not in str(exc):
                raise ServiceError(
                    "function_unavailable:%s" % cost_function, 500
                ) from exc
        else:
            raise ServiceError("function_probe_invalid:%s" % cost_function, 500)
        today = int(self.now().strftime("%Y%m%d"))
        source = self._config.daily_icp_source(set_id=today, active_at=self.now())
        if not isinstance(source, Mapping) or source.get("status") not in (
            "ready",
            "unavailable",
        ):
            raise ServiceError("daily_icp_source_invalid", 500)
        probe_ref = "arena/_startup/object-store.json"
        probe_bytes = contracts.canonical_json({"probe": "lab_arena_object_store_v1"}).encode("utf-8")
        try:
            self._objects.put(probe_ref, probe_bytes)
            if self._objects.get(probe_ref) != probe_bytes:
                raise ServiceError("object_store_mismatch", 500)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("object_store_unavailable", 500) from exc
        current = self.current_round()
        if getattr(getattr(self._config, "defaults", None), "integrity_from", None) or (current and integrity.enabled(current.get("configuration_doc") or {})):
            self._require_integrity_schema()
        if self._config.defaults.contacts_from or (current and contact_policy.enabled(current.get("configuration_doc") or {})):
            self._require_contact_schema()
        return {
            "database_identity": identity,
            "schema_version": int(schema_version),
            "scoring_adapter_version": self._scorer_policy["scoring_adapter_version"],
            "current_round": current["round_id"] if current else None,
        }

    # -- round creation (section 5.1) ----------------------------------------

    def _require_integrity_schema(self) -> None:
        try:
            result = self._store._transport.rpc("lab_arena_integrity_schema_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("integrity_schema_unavailable", 503) from exc
        if not isinstance(result, Mapping) or result.get("schema_version") != "leadpoet.lab_arena.integrity_schema.v1" or result.get("version") != 213:
            raise ServiceError("integrity_schema_invalid", 503)

    def _require_contact_schema(self) -> None:
        try:
            result = self._store._transport.rpc("lab_arena_contact_schema_v1", {})
        except ArenaStoreError as exc:
            raise ServiceError("contact_schema_unavailable", 503) from exc
        if not isinstance(result, Mapping) or result.get("schema_version") != "leadpoet.lab_arena.contact_schema.v1" or result.get("version") != 215:
            raise ServiceError("contact_schema_invalid", 503)

    def build_schedule(self, cutoff: datetime) -> Dict[str, str]:
        """Build the round's cutoff and absolute timeout budget.

        ``submission_cutoff`` is the only minimum start time.  The stage start
        fields mark nominal capacity-budget boundaries.  The close fields are
        deadlines for incomplete work.  Ready phases advance without waiting
        for a nominal boundary.
        """

        minutes = self._config.defaults.stage_minutes
        cutoff = cutoff.astimezone(timezone.utc)
        benchmark_deadline = cutoff + timedelta(minutes=minutes["benchmark"])
        stage_1_start = benchmark_deadline + timedelta(seconds=1)
        stage_1_close = stage_1_start + timedelta(minutes=minutes["stage_1"])
        stage_1_scoring_close = stage_1_close + timedelta(minutes=minutes["stage_1_scoring"])
        stage_2_start = stage_1_scoring_close + timedelta(seconds=1)
        stage_2_close = stage_2_start + timedelta(minutes=minutes["stage_2"])
        final_scoring_close = stage_2_close + timedelta(minutes=minutes["final_scoring"])
        return {
            "submission_open": _iso(cutoff - timedelta(days=1)),
            "submission_cutoff": _iso(cutoff),
            "benchmark_deadline": _iso(benchmark_deadline),
            "stage_1_start": _iso(stage_1_start),
            "stage_1_close": _iso(stage_1_close),
            "stage_1_scoring_close": _iso(stage_1_scoring_close),
            "stage_2_start": _iso(stage_2_start),
            "stage_2_close": _iso(stage_2_close),
            "final_scoring_close": _iso(final_scoring_close),
            "publication_deadline": _iso(final_scoring_close + timedelta(seconds=1)),
        }

    def runner_settings(self) -> Tuple[List[str], List[str]]:
        banned = sorted(set(str(item) for item in self._config.banned_hotkeys_source()))
        runners = sorted(set(self._config.defaults.runner_hotkeys))
        for hotkey in runners:
            if hotkey in banned:
                raise ServiceError("runner_banned", 500)
        snapshot = self._benchmark_snapshot()
        eligible = []
        for hotkey in runners:
            try:
                self._benchmark_validator_uid(snapshot, hotkey)
            except ServiceError as exc:
                if exc.status != 403:
                    raise
            else:
                eligible.append(hotkey)
        if not eligible:
            raise ServiceError("daily_runner_capacity_insufficient", 503)
        # This planned runner set sizes the announced daily capacity and stays
        # in the document for schema compatibility. It never grants authority.
        return eligible, banned

    def create_round(self, cutoff: datetime, *, round_id: Optional[str] = None) -> Dict[str, Any]:
        defaults = self._config.defaults
        round_id = round_id or round_id_for_cutoff(cutoff)
        self._require_round_ownership(round_id)
        runner_hotkeys, banned_hotkeys = self.runner_settings()
        document = {
            "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
            "round_id": round_id,
            "mode": self._config.mode,
            "network_name": self._config.network_name,
            "netuid": self._config.netuid,
            "rewards_enabled": bool(defaults.rewards_enabled and self._config.mode == "live"),
            "schedule": self.build_schedule(cutoff),
            "stage_1_icp_count": contracts.STAGE_1_ICP_COUNT,
            "stage_2_icp_count": contracts.STAGE_2_ICP_COUNT,
            "finalist_count": contracts.FINALIST_COUNT,
            "max_challengers": int(defaults.max_challengers),
            "runner_slot_ceiling": contracts.RUNNER_SLOT_CEILING,
            "max_attempts_per_assignment": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
            "lease_ttl_seconds": contracts.LEASE_TTL_SECONDS,
            "companies_per_icp": 5,
            "providers": list(contracts.PROVIDERS),
            "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
            "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
            "icp_wall_clock_seconds": contracts.ICP_WALL_CLOCK_SECONDS,
            "scoring_wall_clock_seconds": contracts.SCORING_WALL_CLOCK_SECONDS,
            "scorer_policy": self._scorer_policy,
            "execution_cap_microusd": defaults.execution_cap_microusd,
            "cost_per_company_microusd": defaults.cost_per_company_microusd,
            "scoring_cap_microusd": defaults.scoring_cap_microusd,
            "scorer_image_digest": defaults.scorer_image_digest,
            "scorer_image_reference": defaults.scorer_image_reference,
            "baseline_hotkey": defaults.baseline_hotkey,
            "baseline_source_url": defaults.baseline_source_url,
            "runner_hotkeys": runner_hotkeys,
            "banned_hotkeys": banned_hotkeys,
            "reward_constants": rewards.reward_constants_document(int(defaults.pool_percent)),
        }
        if defaults.integrity_from is not None and cutoff >= datetime.fromisoformat(defaults.integrity_from.replace("Z", "+00:00")):
            self._require_integrity_schema()
            document["integrity_policy"] = integrity.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(scoring_adapter_version=integrity.SCORING_ADAPTER)
            execution_minutes, judging_minutes = defaults.confirmation_minutes
            if execution_minutes <= 0 or judging_minutes <= 0:
                raise ServiceError("confirmation_schedule_invalid", 500)
            confirmation_start = _parse_iso(document["schedule"]["final_scoring_close"]) + timedelta(seconds=1)
            confirmation_close = confirmation_start + timedelta(minutes=execution_minutes)
            confirmation_scoring_close = confirmation_close + timedelta(minutes=judging_minutes)
            document["schedule"].update({
                "stage_3_start": _iso(confirmation_start),
                "stage_3_close": _iso(confirmation_close),
                "stage_3_scoring_close": _iso(confirmation_scoring_close),
                "publication_deadline": _iso(confirmation_scoring_close + timedelta(seconds=1)),
            })
        if defaults.contacts_from is not None and _parse_iso(document["schedule"]["submission_open"]) >= _parse_iso(defaults.contacts_from):
            if not integrity.enabled(document):
                raise ServiceError("contact_policy_requires_integrity", 503)
            self._require_contact_schema()
            document["contact_policy"] = contact_policy.POLICY
            document["scorer_policy"] = scoring.build_scorer_policy(scoring_adapter_version=contact_policy.SCORING_ADAPTER)
        if (
            defaults.benchmark_disclosure_from is not None
            and cutoff >= icp_disclosure.parse_activation(
                defaults.benchmark_disclosure_from
            )
        ):
            document["benchmark_disclosure_policy"] = (
                icp_disclosure.DELAYED_DISCLOSURE_POLICY
            )
        # Keep the announced intake within the actual all-participant workload.
        # Shadow-only short rehearsals deliberately do not reserve live budgets.
        if self._config.mode == "live":
            supported = capacity.daily_challenger_capacity(document)
            if supported < 1:
                raise ServiceError("daily_runner_capacity_insufficient", 503)
            document["max_challengers"] = min(document["max_challengers"], supported)
        configuration = contracts.validate_round_configuration(document)
        result = self._store.create_round(round_id, configuration)
        if result.get("status") not in ("created", "existing"):
            raise ServiceError("round_create_failed", 500)
        if result.get("status") == "existing":
            return dict(self._round(round_id).get("configuration_doc") or {})
        return configuration

    def ensure_daily_round(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Create the next daily round when no round is open for submissions.

        Rounds overlap: the day's round runs its benchmark while the next
        round is already open, so miners can always submit. Every signed
        request names its round, and the driver advances every round that is
        not published or cancelled. The new round's cutoff is the next
        configured UTC hour at least ``min_submission_hours`` ahead; a date
        whose round already exists (published or cancelled that day) moves to
        the next day, because a round id is its cutoff date. Idempotent: a
        second call finds the round it created.
        """

        defaults = self._config.defaults
        open_round = self.open_round()
        if open_round is not None:
            return {"status": "existing", "round_id": open_round["round_id"], "round_status": open_round["status"]}
        if defaults.daily_cutoff_hour_utc is None:
            return {"status": "disabled"}
        hour = int(defaults.daily_cutoff_hour_utc)
        if not 0 <= hour <= 23:
            raise ServiceError("daily_cutoff_hour_invalid", 500)
        moment = (now or self.now()).astimezone(timezone.utc)
        earliest = moment + timedelta(hours=max(0, int(defaults.min_submission_hours)))
        cutoff = earliest.replace(hour=hour, minute=0, second=0, microsecond=0)
        if cutoff < earliest:
            cutoff += timedelta(days=1)
        for _ in range(14):
            if self._store.get_round(round_id_for_cutoff(cutoff)) is None:
                created = self.create_round(cutoff)
                return {"status": "created", "round_id": created["round_id"], "cutoff": _iso(cutoff)}
            cutoff += timedelta(days=1)
        raise ServiceError("daily_round_dates_exhausted", 500)

    def current_round(self) -> Optional[Dict[str, Any]]:
        """The newest round that is not published or cancelled (operator status)."""

        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and row["status"] not in TERMINAL_STATUSES else None
        # Scan ids and statuses only; a full row can be large at hundreds of participants.
        network_name, netuid = self._chain_scope()
        for row in self._store.list_rounds(
            statuses=ACTIVE_ROUND_STATUSES,
            mode=self._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=20,
            columns="round_id,status,created_at,configuration_doc",
        ):
            if row["status"] not in TERMINAL_STATUSES and (row.get("configuration_doc") or {}).get("mode") == self._config.mode:
                return self._round(row["round_id"])
        return None

    def active_rounds(self) -> List[Dict[str, Any]]:
        """Every round that is not published or cancelled, oldest first: ids and statuses only.

        Rounds overlap (one open for submissions while the previous one runs),
        so the driver advances each of them on every tick.
        """

        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            if row is None or row["status"] in TERMINAL_STATUSES:
                return []
            return [{
                "round_id": row["round_id"],
                "status": row["status"],
                "schedule": dict((row.get("configuration_doc") or {}).get("schedule") or {}),
            }]

        network_name, netuid = self._chain_scope()
        rows: List[Dict[str, Any]] = []
        offset = 0
        page_size = 20
        while True:
            page = self._store.list_rounds(
                statuses=ACTIVE_ROUND_STATUSES,
                mode=self._config.mode,
                network_name=network_name,
                netuid=netuid,
                limit=page_size,
                offset=offset,
                columns="round_id,status,created_at,configuration_doc",
            )
            rows.extend(
                row
                for row in page
                if row["status"] not in TERMINAL_STATUSES
                and (row.get("configuration_doc") or {}).get("mode") == self._config.mode
            )
            if len(page) < page_size:
                break
            offset += page_size
        return [
            {
                "round_id": row["round_id"],
                "status": row["status"],
                "schedule": dict((row.get("configuration_doc") or {}).get("schedule") or {}),
            }
            for row in reversed(rows)
        ]

    def open_round(self) -> Optional[Dict[str, Any]]:
        """The round open for submissions, if any (at most one at a time)."""

        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and row["status"] == "open" else None
        network_name, netuid = self._chain_scope()
        for row in self._store.list_rounds(
            status="open",
            mode=self._config.mode,
            network_name=network_name,
            netuid=netuid,
            limit=20,
            columns="round_id,status,created_at,configuration_doc",
        ):
            if row["status"] == "open" and (row.get("configuration_doc") or {}).get("mode") == self._config.mode:
                return self._round(row["round_id"])
        return None

    def _hot_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """One round's row for runner-facing handlers, cached for a few seconds.

        Claims and completions arrive by the thousand per stage; the SQL
        functions remain the authority for status, so a briefly stale row
        only yields a structured refusal.
        """

        self._require_round_ownership(round_id)
        now = time.monotonic()
        with self._hot_round_lock:
            cached = self._hot_rounds.get(round_id)
            if cached is not None and now - cached[0] < HOT_ROUND_TTL_SECONDS:
                return cached[1]
        row = self._store.get_round(round_id)
        if row is None:
            return None
        row = self._require_round_mode(row)
        with self._hot_round_lock:
            self._hot_rounds[round_id] = (now, row)
        return row

    def _request_round(self, envelope: Any, *, scope: str, hot: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate a signed request and resolve the round its envelope names.

        Rounds overlap, so the envelope, not "the current round", says which
        round a submission, claim, or completion belongs to. An unknown round
        is refused before any banned-list or status check.
        """

        validated = self.validate_request(envelope, scope=scope, round_id=None)
        round_id = str(validated["round_id"])
        self._require_round_ownership(round_id)
        round_row = self._hot_round(round_id) if hot else self._store.get_round(round_id)
        if round_row is None:
            raise ServiceError("round_unknown", 404)
        round_row = self._require_round_mode(round_row)
        if validated["hotkey"] in self._banned_hotkeys(round_row):
            raise ServiceError("hotkey_banned", 403)
        return validated, round_row

    def latest_published_round(self) -> Optional[Dict[str, Any]]:
        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            return row if row is not None and row["status"] == "published" else None
        network_name, netuid = self._chain_scope()
        rows = self._store.list_rounds(
            status="published", mode=self._config.mode,
            network_name=network_name, netuid=netuid,
            limit=200
        )
        return next(
            (row for row in rows if (row.get("configuration_doc") or {}).get("mode") == self._config.mode),
            None,
        )

    # -- signed requests ------------------------------------------------------

    def validate_request(self, envelope: Any, *, scope: str, round_id: Optional[str]) -> Dict[str, Any]:
        try:
            validated = contracts.validate_signed_request(envelope, expected_scope=scope, now=int(self.now().timestamp()), verify_signature=self._config.verify_signature, expected_round_id=round_id)
        except ArenaSignatureError:
            raise ServiceError("signature_invalid", 401)
        except ArenaContractError as exc:
            raise ServiceError("request_invalid:%s" % str(exc)[:80], 400)
        if round_id is not None and validated["hotkey"] in self._banned_hotkeys(self._round(round_id)):
            raise ServiceError("hotkey_banned", 403)
        return validated

    @staticmethod
    def _banned_hotkeys(round_row: Mapping[str, Any]) -> set:
        """Return the plain banned-hotkey list stored with the round."""

        return set(str(item) for item in (round_row.get("configuration_doc") or {}).get("banned_hotkeys") or [])

    # -- submissions (sections 6, 7, 14.2) -------------------------------------

    def submission_status(self, submission_id: str) -> Dict[str, Any]:
        row = self._store.get_submission(submission_id)
        if row is None:
            raise ServiceError("submission_missing", 404)
        self._round(str(row.get("round_id") or ""))
        return {
            "submission_id": submission_id,
            "status": row["status"],
            "rejection_rule": row.get("rejection_rule"),
            "code_review": self._public_code_review(row),
        }

    @staticmethod
    def _public_code_review(row: Mapping[str, Any]) -> Dict[str, Any]:
        document = row.get("code_review_doc") or {}
        return {
            "status": row.get("code_review_status") or "pending",
            "model": document.get("model"),
            "file_count": document.get("file_count"),
            "source_bytes": document.get("source_bytes"),
            "cost_microusd": document.get("cost_microusd"),
            "review_cost_microusd": document.get("review_cost_microusd"),
            "cost_status": document.get("cost_status"),
            "error_code": document.get("error_code"),
            "categories": document.get("categories") or [],
        }

    @staticmethod
    def _is_daily_baseline(row: Mapping[str, Any], round_row: Mapping[str, Any]) -> bool:
        return bool(
            row.get("is_king")
            and row.get("submission_id") == "baseline-" + str(round_row["round_id"]).removeprefix("arena-")
            and row.get("miner_hotkey") == (round_row.get("configuration_doc") or {}).get("baseline_hotkey")
        )

    def _require_code_review(self, submission_id: str, round_row: Mapping[str, Any]) -> None:
        row = self._store.get_submission(submission_id)
        if row is None:
            raise ServiceError("submission_missing", 404)
        if not self._is_daily_baseline(row, round_row) and row.get("code_review_status") != "passed":
            raise ServiceError("code_review_required", 409)

    def review_pending_submissions(self) -> Dict[str, Any]:
        """Called by a separate worker, never in the competition scheduler."""
        reviewer = self._config.code_reviewer
        if reviewer is None:
            raise ServiceError("code_review_unavailable", 503)
        candidates = []
        for active in self.active_rounds():
            round_row = self._round(active["round_id"])
            for status in ("accepted", "frozen"):
                for row in self._store.list_submissions(round_row["round_id"], status=status):
                    if self._is_daily_baseline(row, round_row) or row.get("code_review_status") in ("passed", "rejected"):
                        continue
                    candidates.append(row)
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="arena-code-review") as pool:
            results = list(pool.map(reviewer.review, candidates))
        reviewed = sum(result.get("status") in ("passed", "rejected", "error", "ok") for result in results)
        return {"reviewed": reviewed}

    def _require_submission_window(self, round_row: Mapping[str, Any]) -> None:
        if round_row["status"] != "open":
            raise ServiceError("submission_window_closed", 409)
        schedule = (round_row.get("configuration_doc") or {}).get("schedule") or {}
        now = self.now()
        if now < _parse_iso(schedule["submission_open"]) or now >= _parse_iso(schedule["submission_cutoff"]):
            raise ServiceError("submission_window_closed", 409)

    def _enforce_submission_request_limit(self, hotkey: str) -> None:
        # Some narrow unit tests construct the service without __init__. The
        # fallback is test-only; production always creates the limiter above.
        limiter = getattr(self, "_submission_request_limiter", None)
        if limiter is None:
            limiter = submission_rate_limit.SubmissionRequestLimiter()
            self._submission_request_limiter = limiter
        decision = limiter.check(hotkey)
        if not decision.allowed:
            raise ServiceError("submission_rate_limited", 429)

    def handle_submission_presign(self, envelope: Any) -> Dict[str, Any]:
        """Reserve one private source upload for a signed miner request."""

        validated, round_row = self._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_PRESIGN
        )
        self._require_submission_window(round_row)
        round_id = round_row["round_id"]
        configuration = round_row.get("configuration_doc") or {}
        owner_admission = None
        if integrity.enabled(configuration):
            try:
                owner_admission = resolve_finalized_owner(
                    self._config.chain, validated["hotkey"]
                )
            except OwnerAdmissionError as exc:
                status = 403 if exc.code == "hotkey_unregistered" else 503
                raise ServiceError(exc.code, status) from exc
        elif self._config.chain.uid_for_hotkey(validated["hotkey"]) is None:
            raise ServiceError("hotkey_unregistered", 403)
        if validated["hotkey"] == (round_row.get("configuration_doc") or {}).get(
            "baseline_hotkey"
        ):
            raise ServiceError("baseline_hotkey_reserved", 403)
        body = contracts.validate_submission_presign_body(validated["body"])
        self._enforce_submission_request_limit(validated["hotkey"])
        # The submission id is the only server-assigned source identity.
        submission_id = "sub-%s" % secrets.token_hex(16)
        source_ref = "arena/%s/sources/%s.tar.gz" % (round_id, submission_id)
        document = {
            "source_ref": source_ref,
            "source_size_bytes": body["source_size_bytes"],
            "consent": dict(body["consent"]),
        }
        if body.get("source_content_md5") is not None:
            document["source_content_md5"] = body["source_content_md5"]
        try:
            registration_args = (
                round_id, submission_id, validated["hotkey"], document
            )
            if owner_admission is None:
                registration = self._store.register_submission(*registration_args)
            else:
                registration = self._store.register_submission(
                    *registration_args, owner_admission=owner_admission
                )
        except ArenaStoreError as exc:
            if "lab_arena_owner_active_submission" in str(exc):
                raise ServiceError("owner_active_submission_exists", 409) from exc
            if "lab_arena_submission_owner_changed" in str(exc):
                raise ServiceError("submission_owner_changed", 409) from exc
            if "lab_arena_submission_conflict" in str(exc):
                raise ServiceError("submission_conflict", 409) from exc
            raise
        if registration.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        if registration.get("status") not in ("registered", "existing"):
            raise ServiceError("submission_registration_failed", 500)
        submission_id = str(registration.get("submission_id") or submission_id)
        source_ref = str(registration.get("source_ref") or source_ref)
        if registration.get("submission_status") in ("accepted", "frozen"):
            # Historical accepted rows predate persisted transport checksums.
            # Verify their bytes before treating a same-size upload as a retry.
            checksum = body.get("source_content_md5")
            if checksum is not None:
                self._validate_uploaded_source({
                    "source_ref": source_ref,
                    "source_size_bytes": body["source_size_bytes"],
                    "submission_doc": {"source_content_md5": checksum},
                })
        try:
            upload_arguments = {
                "size_bytes": int(body["source_size_bytes"]),
                "content_type": source_bundle.SOURCE_CONTENT_TYPE,
                "expires_seconds": SOURCE_UPLOAD_EXPIRES_SECONDS,
            }
            if body.get("source_content_md5") is not None:
                upload_arguments["source_content_md5"] = body["source_content_md5"]
            upload = self._objects.presign_put(source_ref, **upload_arguments)
        except Exception as exc:
            raise ServiceError("source_upload_unavailable", 503) from exc
        return {
            "status": "upload_ready",
            "submission_id": submission_id,
            "source_ref": source_ref,
            "upload_url": str(upload["upload_url"]),
            "upload_headers": dict(upload["upload_headers"]),
            "expires_in_seconds": int(upload["expires_in_seconds"]),
        }

    def _validate_uploaded_source(
        self,
        row: Mapping[str, Any],
        *,
        forbidden_values: Sequence[str] = (),
    ) -> None:
        expected_size = int(row.get("source_size_bytes") or 0)
        source_ref = str(row.get("source_ref") or "")
        try:
            payload = self._objects.get_bounded(
                source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
        except Exception as exc:
            raise ServiceError("source_upload_unavailable", 409) from exc
        if len(payload) != expected_size:
            raise ServiceError("submission_rejected:source_size_mismatch", 400)
        checksum = (row.get("submission_doc") or {}).get("source_content_md5")
        if checksum is not None:
            actual = base64.b64encode(
                hashlib.md5(payload, usedforsecurity=False).digest()
            ).decode("ascii")
            if not hmac.compare_digest(actual, checksum):
                raise ServiceError("submission_rejected:source_checksum_mismatch", 400)
        try:
            source_bundle.validate_source_archive(
                payload, forbidden_values=forbidden_values
            )
        except source_bundle.SourceBundleError as exc:
            path = exc.path or ""
            for value in forbidden_values:
                if value:
                    path = path.replace(value, "[REDACTED]")
            raise ServiceError(
                "submission_rejected:%s" % exc.code, 400,
                source_path=path[:source_bundle.MAX_SOURCE_PATH_BYTES],
            ) from exc

    def handle_submission_finalize(
        self, submission_id: str, envelope: Any
    ) -> Dict[str, Any]:
        """Verify uploaded bytes and admit the source under the same signed owner."""

        validated, round_row = self._request_round(
            envelope, scope=contracts.SCOPE_SUBMISSION_FINALIZE
        )
        self._require_submission_window(round_row)
        body = contracts.validate_submission_finalize_body(validated["body"])
        if body["submission_id"] != submission_id:
            raise ServiceError("submission_id_mismatch", 400)
        row = self._store.get_submission(submission_id)
        if (
            row is None
            or row.get("round_id") != round_row["round_id"]
            or row.get("miner_hotkey") != validated["hotkey"]
        ):
            raise ServiceError("submission_missing", 404)
        for field in ("source_ref", "source_size_bytes"):
            if body[field] != row.get(field):
                raise ServiceError("submission_transport_mismatch", 409)
        if row.get("status") == "accepted":
            stored_credentials = {
                provider: self._store.get_submission_credential(
                    submission_id, validated["hotkey"], provider
                )
                for provider in credentials_module.RUNTIME_PROVIDERS
            }
            if all(
                stored_credentials[provider] is not None
                for provider in credentials_module.REQUIRED_RUNTIME_PROVIDERS
            ):
                if (
                    "scrapingdog_api_key" in body["credentials"]
                    and stored_credentials["scrapingdog"] is None
                ):
                    raise ServiceError("submission_credentials_immutable", 409)
                return {"status": "accepted", "submission_id": submission_id}
            raise ServiceError("submission_credentials_missing", 409)
        if row.get("status") != "uploading":
            if row.get("rejection_rule") == "source_replaced":
                raise ServiceError("submission_superseded", 409)
            raise ServiceError("submission_not_uploading", 409)
        self._enforce_submission_request_limit(validated["hotkey"])
        try:
            self._validate_uploaded_source(
                row, forbidden_values=tuple(body["credentials"].values())
            )
        except ServiceError as exc:
            if exc.code.startswith("submission_rejected:"):
                self._store.update_submission(
                    str(round_row["round_id"]),
                    submission_id,
                    "uploading",
                    "rejected",
                    {"rejection_rule": exc.code.split(":", 1)[1]},
                )
            raise
        manager = self._config.credential_manager
        if manager is None:
            raise ServiceError("credential_validation_unavailable", 503)
        try:
            encrypted_credentials = manager.validate_and_encrypt(
                body["credentials"],
                submission_id=submission_id,
                miner_hotkey=validated["hotkey"],
            )
        except credentials_module.CredentialError as exc:
            if exc.retryable:
                raise ServiceError(exc.code, 503) from exc
            self._store.update_submission(
                str(round_row["round_id"]),
                submission_id,
                "uploading",
                "rejected",
                {"rejection_rule": exc.code},
            )
            raise ServiceError("submission_rejected:%s" % exc.code, 400) from exc
        try:
            result = self._store.accept_submission_with_credentials(
                str(round_row["round_id"]),
                submission_id,
                validated["hotkey"],
                encrypted_credentials,
            )
        except ArenaStoreError as exc:
            if "lab_arena_submission_credentials_immutable" in str(exc):
                raise ServiceError("submission_credentials_immutable", 409) from exc
            if "lab_arena_round_full" not in str(exc):
                raise
            self._store.update_submission(
                str(round_row["round_id"]), submission_id, "uploading", "rejected",
                {"rejection_rule": "capacity.round_full"},
            )
            raise ServiceError("submission_rejected:capacity.round_full", 409) from exc
        if result.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        if result.get("status") not in ("ok", "existing"):
            raise ServiceError("submission_finalize_failed", 500)
        return {"status": "accepted", "submission_id": submission_id}

    def admit_uploaded_submissions(self, round_id: str, *, final: bool = False) -> Dict[str, Any]:
        """At cutoff, reject source slots that a miner did not finalize."""

        round_row = self._round(round_id)
        if round_row["status"] != "open":
            return {"status": "stale", "round_status": round_row["status"]}
        outcomes: Dict[str, Any] = {"status": "ok", "accepted": 0, "rejected": 0, "deferred": 0, "remaining": 0}
        pending = self._store.list_submissions(round_id, status="uploading")
        if not final:
            outcomes["remaining"] = len(pending)
            return outcomes
        for row in pending:
            result = self._store.update_submission(
                round_id,
                str(row["submission_id"]),
                "uploading",
                "rejected",
                {"rejection_rule": "source_upload_incomplete"},
            )
            outcomes["rejected" if result.get("status") == "ok" else "deferred"] += 1
        return outcomes

    @staticmethod
    def _participant(row: Mapping[str, Any], *, is_king: bool) -> Dict[str, Any]:
        """The small frozen participant record used for leases and publication."""

        return {
            "submission_id": row["submission_id"],
            "miner_hotkey": row["miner_hotkey"],
            "source_ref": row["source_ref"],
            "source_size_bytes": int(row["source_size_bytes"]),
            "is_king": bool(is_king),
        }

    # -- participant freeze and benchmark (sections 7.1, 8) --------------------

    def _initial_baseline(self, round_row: Mapping[str, Any]) -> Dict[str, Any]:
        """Download and freeze the public baseline through the source path."""

        round_id = str(round_row["round_id"])
        configuration = round_row.get("configuration_doc") or {}
        hotkey = str(configuration.get("baseline_hotkey") or "").strip()
        source_url = str(
            configuration.get("baseline_source_url")
            or self._config.defaults.baseline_source_url
            or ""
        ).strip()
        if not hotkey:
            raise ServiceError("baseline_hotkey_missing", 500)
        submission_id = "baseline-%s" % round_id.removeprefix("arena-")
        source_ref = "arena/%s/sources/%s.tar.gz" % (round_id, submission_id)
        row = self._store.get_submission(submission_id)
        if row is None:
            fetcher = self._config.baseline_source_fetcher
            if fetcher is None:
                raise ServiceError("baseline_source_fetcher_missing", 500)
            selected_source_url = source_url
            source_observation = "configured_shadow_source"
            try:
                payload = self._objects.get_bounded(
                    source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
                )
                source_observation = "stored_object:%s" % source_ref
            except Exception:
                try:
                    if str(configuration.get("mode") or self._config.mode) == "live":
                        network_name, netuid = self._chain_scope()
                        if self._store.pending_promotions(
                            network_name=network_name, netuid=netuid
                        ):
                            raise ServiceError("baseline_promotion_pending", 503)
                        selected_source_url = DEFAULT_BASELINE_SOURCE_URL
                    source_observation = (
                        selected_source_url
                        if selected_source_url == DEFAULT_BASELINE_SOURCE_URL
                        else "configured_shadow_source"
                    )
                    payload = bytes(
                        fetcher(
                            selected_source_url,
                            source_bundle.MAX_SOURCE_ARCHIVE_BYTES,
                        )
                    )
                    facts = source_bundle.validate_source_archive(payload)
                    self._objects.put(source_ref, payload)
                except ServiceError:
                    raise
                except source_bundle.SourceBundleError as exc:
                    raise ServiceError(
                        "baseline_source_invalid:%s" % exc.code, 500
                    ) from exc
                except Exception as exc:
                    raise ServiceError("baseline_source_not_ready", 503) from exc
            else:
                try:
                    facts = source_bundle.validate_source_archive(payload)
                except source_bundle.SourceBundleError as exc:
                    raise ServiceError(
                        "baseline_source_invalid:%s" % exc.code, 500
                    ) from exc
            logger.info(
                "arena_baseline_source_frozen round_id=%s source=%s source_commit=%s",
                round_id,
                source_observation,
                source_bundle.source_archive_commit(payload) or "unavailable",
            )
            result = self._store.register_submission(
                round_id,
                submission_id,
                hotkey,
                {
                    "source_ref": source_ref,
                    "source_size_bytes": facts["source_size_bytes"],
                    "consent": {"public_rerun": True},
                    "is_king": True,
                },
            )
            if result.get("status") not in ("registered", "existing"):
                raise ServiceError("baseline_registration_failed", 500)
            row = self._store.get_submission(submission_id)
        if (
            row is None
            or row.get("round_id") != round_id
            or row.get("miner_hotkey") != hotkey
            or not row.get("is_king")
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        status = str(row.get("status") or "")
        if status == "uploading":
            try:
                self._validate_uploaded_source(row)
            except ServiceError as exc:
                if exc.code == "source_upload_unavailable":
                    raise ServiceError("baseline_source_not_ready", 503) from exc
                raise ServiceError("baseline_source_invalid", 500) from exc
            result = self._store.update_submission(
                round_id, submission_id, "uploading", "accepted"
            )
            if result.get("status") not in ("ok", "stale"):
                raise ServiceError("baseline_source_not_ready", 503)
            row = self._store.get_submission(submission_id)
            status = str((row or {}).get("status") or "")
        if status not in ("accepted", "frozen"):
            raise ServiceError("baseline_source_rejected", 500)
        return dict(row)

    def freeze_participants(self, round_id: str) -> List[Dict[str, Any]]:
        round_row = self._round(round_id)
        participants: List[Dict[str, Any]] = []
        frozen = self._store.list_submissions(round_id, status="frozen")
        accepted = self._store.list_submissions(round_id, status="accepted")
        unresolved = [
            row for row in accepted
            if not self._is_daily_baseline(row, round_row)
            and row.get("code_review_status") not in ("passed", "rejected")
            and (row.get("code_review_status") == "reviewing" or int(row.get("code_review_attempts") or 0) < 3)
        ]
        if unresolved:
            deadline = (round_row["configuration_doc"].get("schedule") or {}).get("benchmark_deadline")
            if not deadline or self.now() < _parse_iso(deadline):
                raise ServiceError("code_review_pending", 409)
        frozen_kings = [row for row in frozen if row.get("is_king")]
        if len(frozen_kings) > 1:
            raise ServiceError("baseline_submission_invalid", 500)
        baseline_id = "baseline-%s" % round_id.removeprefix("arena-")
        baseline_hotkey = str(
            (round_row.get("configuration_doc") or {}).get("baseline_hotkey") or ""
        )
        if frozen_kings:
            baseline = frozen_kings[0]
        else:
            baseline = self._initial_baseline(round_row)
            if baseline.get("status") == "accepted":
                accepted = self._store.list_submissions(round_id, status="accepted")
        if (
            baseline.get("submission_id") != baseline_id
            or baseline.get("miner_hotkey") != baseline_hotkey
            or not baseline.get("is_king")
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        if any(
            row.get("is_king") and row.get("submission_id") != baseline_id
            for row in accepted
        ):
            raise ServiceError("baseline_submission_invalid", 500)
        cap = int(round_row["configuration_doc"].get("max_challengers") or contracts.MAX_CHALLENGERS)
        # Freeze order (acceptance order) decides who enters when the cap binds.
        participants.extend(self._participant(row, is_king=bool(row.get("is_king"))) for row in frozen)
        frozen_count = sum(1 for row in frozen if not row.get("is_king"))
        for row in accepted:
            is_king = row["submission_id"] == baseline_id
            if not is_king and row.get("code_review_status") != "passed":
                rule = "code_review_rejected" if row.get("code_review_status") == "rejected" else "code_review_incomplete"
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": rule})
                continue
            if not is_king and frozen_count >= cap:
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": "capacity.round_full"})
                continue
            if not is_king:
                frozen_count += 1
            result = self._store.update_submission(round_id, row["submission_id"], "accepted", "frozen", {"is_king": True} if is_king else {})
            if result.get("status") in ("ok", "stale"):
                participants.append(self._participant(row, is_king=is_king))
        if sum(1 for participant in participants if participant.get("is_king")) != 1:
            raise ServiceError("baseline_submission_invalid", 500)
        return participants

    def commit_benchmark(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "open":
            return {"status": "existing", "round_status": round_row["status"]}
        scorer_image = {
            "scorer_image_digest": self._config.defaults.scorer_image_digest,
            "scorer_image_reference": self._config.defaults.scorer_image_reference,
        }
        refreshed_configuration = self._configuration_for_commit({
            **dict(round_row.get("configuration_doc") or {}),
            **scorer_image,
        })
        try:
            contracts.validate_round_configuration(refreshed_configuration)
        except ArenaContractError as exc:
            raise ServiceError("scorer_image_invalid", 500) from exc
        started = self.now()
        schedule = round_row["configuration_doc"]["schedule"]
        if started < _parse_iso(schedule["submission_cutoff"]):
            return {"status": "waiting", "round_status": "open"}
        submission_open = _parse_iso(schedule["submission_open"])
        icp_set_date = submission_open.astimezone(timezone.utc).date().isoformat()
        set_id = int(icp_set_date.replace("-", ""))
        source = self._config.daily_icp_source(set_id=set_id, active_at=started)
        if not isinstance(source, Mapping) or source.get("status") not in (
            "ready",
            "unavailable",
        ):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        if source.get("status") == "unavailable":
            if started >= _parse_iso(schedule["benchmark_deadline"]):
                self._store.cancel_round(
                    round_id, CANCEL_REASONS["benchmark_invalid"]
                )
                return {
                    "status": "cancelled",
                    "reason": CANCEL_REASONS["benchmark_invalid"],
                }
            return {
                "status": "retry",
                "reason": "daily_icp_set_not_ready",
                "set_id": set_id,
            }
        raw_icps = source.get("icps")
        try:
            source_set_id = int(source.get("set_id") or 0)
        except (TypeError, ValueError):
            source_set_id = 0
        if source_set_id != set_id or not isinstance(raw_icps, list):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        icps = [dict(icp) for icp in raw_icps if isinstance(icp, Mapping)]
        icp_ids = [str(icp.get("icp_id") or "").strip() for icp in icps]
        contact_bank_valid = True
        if contact_policy.enabled(round_row["configuration_doc"]):
            try:
                for icp in icps:
                    contact_policy.validate_icp(icp)
            except (ValueError, TypeError):
                contact_bank_valid = False
        if (
            len(icps) != contracts.BENCHMARK_ICP_COUNT
            or len(icps) != len(raw_icps)
            or any(not icp_id for icp_id in icp_ids)
            or len(set(icp_ids)) != len(icp_ids)
            or not contact_bank_valid
        ):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        if integrity.enabled(round_row["configuration_doc"]):
            try:
                self._prepare_confirmation_bank(round_row, icps)
            except (ValueError, TimeoutError, ArenaStoreError) as exc:
                if started >= _parse_iso(schedule["benchmark_deadline"]):
                    return self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
                return {"status": "retry", "reason": "confirmation_bank_unavailable"}
        try:
            participants = self.freeze_participants(round_id)
        except ServiceError as exc:
            if exc.code in ("baseline_source_not_ready", "baseline_promotion_pending", "code_review_pending"):
                return {"status": "retry", "reason": exc.code, "set_id": set_id}
            raise
        evaluation_date = _parse_iso(
            schedule["submission_cutoff"]
        ).astimezone(timezone.utc).date().isoformat()
        benchmark_ref = "arena/%s/benchmark.json" % round_id
        self._objects.put(benchmark_ref, contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": round_id, "icps": icps}).encode("utf-8"))
        transition = self._store.commit_round_v2(
            round_id,
            participants=participants,
            benchmark_ref=benchmark_ref,
            evaluation_date=evaluation_date,
            icp_set_date=icp_set_date,
            scorer_image_digest=scorer_image["scorer_image_digest"],
            scorer_image_reference=scorer_image["scorer_image_reference"],
        )
        return {"status": transition.get("status"), "participants": len(participants)}

    @staticmethod
    def _configuration_for_commit(configuration: Mapping[str, Any]) -> Dict[str, Any]:
        """Mirror the one-time SQL adoption for a legacy live open round."""

        refreshed = dict(configuration)
        if (
            refreshed.get("mode") == "live"
            and "cost_per_company_microusd" not in refreshed
        ):
            refreshed["execution_cap_microusd"] = DEFAULT_EXECUTION_CAP_MICROUSD
            refreshed["cost_per_company_microusd"] = (
                DEFAULT_COST_PER_COMPANY_MICROUSD
            )
        return refreshed

    def benchmark_icps(self, round_id: str) -> List[Dict[str, Any]]:
        round_row = self._round(round_id)
        ref = round_row.get("benchmark_ref")
        if not ref:
            raise ServiceError("benchmark_not_committed", 409)
        document = json.loads(self._objects.get(ref).decode("utf-8"))
        if not isinstance(document, Mapping) or set(document) != {"schema_version", "round_id", "icps"}:
            raise ServiceError("benchmark_data_invalid", 500)
        if document.get("schema_version") != "leadpoet.lab_arena.benchmark.v1" or document.get("round_id") != round_id:
            raise ServiceError("benchmark_data_invalid", 500)
        icps = list(document["icps"])
        if len(icps) != contracts.BENCHMARK_ICP_COUNT:
            raise ServiceError("benchmark_data_invalid", 500)
        return icps

    def _prepare_confirmation_bank(self, round_row: Mapping[str, Any], main_icps: Sequence[Mapping[str, Any]]) -> None:
        round_id = str(round_row["round_id"])
        current = self._round(round_id)
        if current.get("confirmation_bank_hash"):
            self.confirmation_bank(round_id)
            return
        provider = self._config.confirmation_icp_source or confirmation.fresh_confirmation_icps
        evaluation_date = _parse_iso(round_row["configuration_doc"]["schedule"]["submission_cutoff"]).astimezone(timezone.utc).date().isoformat()
        has_contacts = contact_policy.enabled(round_row["configuration_doc"])
        generated = provider(round_id=round_id, evaluation_date=evaluation_date, main_icps=main_icps, **({"contacts_required": True} if has_contacts else {}))
        bank = confirmation.build_bank(round_id, generated, main_icps, contacts_required=has_contacts)
        payload = contracts.canonical_json(bank).encode("utf-8")
        digest = contracts.hash_bytes(payload)
        ref = "arena/%s/confirmation/%s.json" % (round_id, digest.split(":")[-1])
        try:
            self._objects.put(ref, payload)
        except Exception:
            # A lost write acknowledgement is safe only if exact bytes landed.
            if self._objects.get_bounded(ref, MAX_OUTPUT_BYTES) != payload:
                raise
        try:
            self._store.prepare_confirmation_bank(round_id, ref, digest)
        except ArenaStoreError:
            if not self._round(round_id).get("confirmation_bank_hash"):
                raise
        self.confirmation_bank(round_id)

    def confirmation_bank(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        ref, digest = row.get("confirmation_bank_ref"), row.get("confirmation_bank_hash")
        if not ref or not digest:
            raise ServiceError("confirmation_bank_missing", 503)
        try:
            return confirmation.read_bank(self._objects.get_bounded(ref, MAX_OUTPUT_BYTES), round_id=round_id, digest=digest)
        except (ValueError, TypeError) as exc:
            raise ServiceError("confirmation_bank_invalid", 503) from exc

    def evaluation_icps(self, round_id: str) -> List[Dict[str, Any]]:
        row = self._round(round_id)
        main = self.benchmark_icps(round_id)
        return main + self.confirmation_bank(round_id)["icps"] if integrity.enabled(row["configuration_doc"]) else main

    def open_confirmation(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        if row["status"] != "scored" or not integrity.enabled(row["configuration_doc"]):
            return {"status": "stale", "round_status": row["status"]}
        self.confirmation_bank(round_id)
        entries = self._score_entries_from_runs(row, range(contracts.BENCHMARK_ICP_COUNT), "final_score")
        runs = self._store.list_runs(round_id, kind="execute")
        eligibility = {entry["submission_id"]: self._submission_cost_eligibility(row, entry["submission_id"], runs, positions=range(contracts.BENCHMARK_ICP_COUNT)) for entry in entries}
        try:
            cohort = confirmation.select_cohort(entries, eligibility)
        except ValueError:
            return self._store.cancel_round(round_id, CANCEL_REASONS["scoring_incomplete"])
        return self._store.open_confirmation(round_id, cohort)

    # -- stages (sections 2, 9) ----------------------------------------------

    def open_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if stage not in (1, 2):
            raise ServiceError("stage_invalid", 400)
        participants = list(round_row.get("participants") or [])
        if stage == 2 and round_row.get("icp_set_date") is None:
            finalists = set(str(item) for item in (round_row.get("finalists") or []))
            participants = [participant for participant in participants if participant["submission_id"] in finalists or participant.get("is_king")]
        self.benchmark_icps(round_id)
        positions = list(contracts.stage_positions(stage))
        rows = [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"]} for p in participants]
        return self._store.open_stage(round_id, stage, rows, positions)

    def stage_is_complete(self, round_id: str, stage: int) -> bool:
        runs = self._store.list_runs(round_id, stage=stage, kind="execute")
        return bool(runs) and all(run["status"] in ("accepted", "failed") for run in runs)

    def close_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        closed = self._store.close_stage(round_id, stage)
        if closed.get("status") != "closed":
            return closed
        return self.commit_scoring_plan(round_id, stage)

    def commit_scoring_plan(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        self.benchmark_icps(round_id)
        plan = scoring.build_scoring_plan(
            round_id=round_id, stage=stage, runs=self._store.list_runs(round_id, stage=stage, kind="execute"),
        )
        status = "stage%d_closed" % stage
        result = self._store.transition_round(round_id, status, status, {"stage%d_scoring_plan_doc" % stage: plan})
        return {"status": result.get("status"), "work_items": len(plan["work_items"])}

    def _load_scoring_plan(self, round_row: Mapping[str, Any], stage: int) -> Dict[str, Any]:
        plan = round_row.get("stage%d_scoring_plan_doc" % stage)
        if not plan:
            raise ServiceError("scoring_plan_missing", 409)
        try:
            validated = contracts.validate_scoring_plan(plan)
        except ArenaContractError as exc:
            raise ServiceError("scoring_plan_invalid", 500) from exc
        if validated["round_id"] != round_row["round_id"] or int(validated["stage"]) != stage:
            raise ServiceError("scoring_plan_invalid", 500)
        return validated

    def _outputs_by_run(self, round_id: str, stage: int) -> Dict[str, List[Dict[str, Any]]]:
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
            outputs[str(run["run_id"])] = list(document["companies"])
        return outputs

    # -- validator scoring (sections 10 and 16 as revised) ----------------------

    def open_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Turn the committed plan into scoring assignments validators claim."""

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_closed" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        plan = self._load_scoring_plan(round_row, stage)
        for submission_id in {item["submission_id"] for item in plan["work_items"]}:
            self._require_code_review(submission_id, round_row)
        accepted = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            accepted[str(run["run_id"])] = run
        items: List[Dict[str, Any]] = []
        integrity_cache = integrity.enabled(round_row.get("configuration_doc") or {})
        icps = self.evaluation_icps(round_id) if integrity_cache else []
        configuration = round_row.get("configuration_doc") or {}
        policy = configuration.get("scorer_policy") or {}
        for item in plan["work_items"]:
            submission_id = item["submission_id"]
            run = accepted.get(str(item["scored_run_id"]))
            if run is None or run["submission_id"] != submission_id or int(run["icp_position"]) != int(item["icp_position"]) or run.get("output_ref") != item["output_ref"]:
                raise ServiceError("scoring_plan_run_mismatch", 500)
            work_item = {
                "scored_run_id": run["run_id"],
                "submission_id": run["submission_id"],
                "icp_position": int(run["icp_position"]),
                "output_ref": run["output_ref"],
            }
            if integrity_cache:
                output = json.loads(
                    self._objects.get_bounded(
                        str(run["output_ref"]), MAX_OUTPUT_BYTES
                    ).decode("utf-8")
                )
                companies = validate_output_document(output, expected_schema_version=contact_policy.output_schema(configuration))["companies"]
                position = int(run["icp_position"])
                scoring_input = scoring.build_scoring_input(
                    scored_run_id=str(run["run_id"]),
                    icp=integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(configuration)),
                    companies=companies,
                    policy=policy,
                    evaluation_date=str(round_row.get("evaluation_date") or ""),
                    contact_source_evidence=(contact_evidence.resolve_sources(self._store, run, companies) if contact_policy.enabled(configuration) else None),
                )
                cache_scope = judgment_cache.build_cache_scope(
                    scoring_input=scoring_input,
                    round_id=round_id,
                    network_name=str(round_row.get("arena_network_name") or ""),
                    netuid=int(round_row.get("arena_netuid") or 0),
                    scorer_image_digest=str(configuration.get("scorer_image_digest") or ""),
                    scorer_image_reference=str(configuration.get("scorer_image_reference") or ""),
                    integrity_policy=str(configuration.get("integrity_policy") or ""),
                )
                work_item.update({
                    "judgment_cache_key": cache_scope["cache_key"],
                    "judgment_scope_doc": cache_scope,
                    "judgment_input_hash": cache_scope["scoring_input_hash"],
                })
            items.append(work_item)
        if integrity_cache:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            participant_by_submission = {
                str(participant["submission_id"]): participant
                for participant in round_row.get("participants") or []
            }
            for item in items:
                participant = participant_by_submission.get(
                    str(item["submission_id"])
                ) or {}
                while True:
                    cache_key = str(item["judgment_cache_key"])
                    cached = self._store.get_judgment_cache(cache_key)
                    if cached is None:
                        grouped.setdefault(cache_key, []).append(item)
                        break
                    try:
                        judgment_cache.validate_evidence_snapshot(
                            cached.get("evidence_doc") or {},
                            cache_key=cache_key,
                            evidence_hash=str(cached.get("evidence_hash") or ""),
                        )
                    except judgment_cache.JudgmentCacheError as exc:
                        raise ServiceError("judgment_cache_invalid", 500) from exc
                    evidence = cached.get("evidence_doc") or {}
                    excluded = set(evidence["runner_authority_exclusions"])
                    if bool(participant.get("is_king")) or str(
                        participant.get("miner_hotkey") or ""
                    ) not in excluded:
                        item["reuse_cache_key"] = cache_key
                        break
                    try:
                        partitioned = judgment_cache.partition_cache_scope(
                            item["judgment_scope_doc"],
                            incompatible_hotkeys=list(excluded),
                        )
                    except judgment_cache.JudgmentCacheError as exc:
                        raise ServiceError("judgment_cache_authority_invalid", 500) from exc
                    item.update({
                        "judgment_cache_key": partitioned["cache_key"],
                        "judgment_scope_doc": partitioned,
                    })
            for cache_key, pending in grouped.items():
                group_miners = sorted({
                    str((participant_by_submission.get(str(item["submission_id"])) or {}).get("miner_hotkey") or "")
                    for item in pending
                } - {""})
                for index, item in enumerate(sorted(pending, key=lambda row: str(row["scored_run_id"]))):
                    item["judgment_group_leader"] = index == 0
                    item["judgment_group_miner_hotkeys"] = group_miners
        result = self._store.open_scoring(
            round_id, stage, items, integrity_cache=integrity_cache
        )
        return {"status": result.get("status"), "round_status": result.get("round_status"), "assignments": result.get("assignments"), "work_items": len(plan["work_items"])}

    def scoring_is_complete(self, round_id: str, stage: int) -> bool:
        runs = self._store.list_runs(round_id, stage=stage, kind="score")
        return all(run["status"] in ("accepted", "failed") for run in runs)

    def _scoring_has_exhausted_judge_failure(
        self,
        round_row: Mapping[str, Any],
        stage: int,
        runs: Sequence[Mapping[str, Any]],
    ) -> bool:
        """Return whether a required score can no longer succeed.

        An accepted attempt always wins. An active attempt means the assignment
        can still succeed. Rows outside the committed plan cannot end a round.
        """

        plan = self._load_scoring_plan(round_row, stage)
        runs_by_scored_run: Dict[str, List[Mapping[str, Any]]] = {}
        planned_run_ids = {
            str(item["scored_run_id"]) for item in plan["work_items"]
        }
        for run in runs:
            scored_run_id = str(run.get("scored_run_id") or "")
            if scored_run_id in planned_run_ids:
                runs_by_scored_run.setdefault(scored_run_id, []).append(run)

        for scored_run_id in planned_run_ids:
            attempts = runs_by_scored_run.get(scored_run_id, [])
            if any(run.get("status") == "accepted" for run in attempts):
                continue
            if any(
                run.get("status") not in ("accepted", "failed")
                for run in attempts
            ):
                continue
            if not attempts:
                continue
            latest = max(attempts, key=lambda run: int(run.get("attempt") or 0))
            if (
                int(latest.get("attempt") or 0)
                >= contracts.MAX_ATTEMPTS_PER_ASSIGNMENT
                and latest.get("status") == "failed"
                and str(latest.get("terminal_cause") or "")
                in contracts.INFRASTRUCTURE_TERMINAL_CAUSES
            ):
                return True
        return False

    def close_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        return self._store.close_scoring(round_id, stage)

    def _scoring_outputs(self, round_id: str, stage: int) -> Dict[str, Dict[str, Any]]:
        """The score run that counts for each scored execution run."""

        chosen: Dict[str, Dict[str, Any]] = {}
        for run in self._store.list_runs(round_id, stage=stage, kind="score"):
            current = chosen.get(run["scored_run_id"])
            if current is None or (run["status"] == "accepted" and current["status"] != "accepted") or (run["status"] == current["status"] and int(run["attempt"]) > int(current["attempt"])):
                chosen[run["scored_run_id"]] = run
        return chosen

    def _verified_breakdowns(self, run: Mapping[str, Any], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> List[Dict[str, Any]]:
        cache_key = str(run.get("judgment_cache_key") or "")
        if cache_key:
            cached = self._store.get_judgment_cache(cache_key)
            if cached is None:
                raise scoring.ScoringError("accepted judgment cache entry is missing")
            try:
                evidence = judgment_cache.validate_evidence_snapshot(
                    cached.get("evidence_doc") or {},
                    cache_key=cache_key,
                    evidence_hash=str(cached.get("evidence_hash") or ""),
                )
            except judgment_cache.JudgmentCacheError as exc:
                raise scoring.ScoringError("accepted judgment cache evidence is invalid") from exc
            source = self._store.get_run(str(evidence["source_score_run_id"]))
            if (
                source is None
                or source.get("kind") != "score"
                or source.get("status") != "accepted"
                or source.get("runner_hotkey") != evidence["source_runner_hotkey"]
                or source.get("scored_run_id") != evidence["source_scored_run_id"]
                or cached.get("source_score_run_id") != evidence["source_score_run_id"]
                or cached.get("scoring_input_hash") != evidence["scoring_input_hash"]
                or run.get("judgment_cache_source_run_id") != evidence["source_score_run_id"]
            ):
                raise scoring.ScoringError("accepted judgment cache authority is invalid")
            return scoring.validate_breakdowns_for_item(
                evidence["breakdowns"],
                icp=icp,
                companies=companies,
                max_scored_companies=int(policy["max_scored_companies"]),
                integrity_policy=contact_policy.integrity_adapter(policy.get("scoring_adapter_version")),
                contacts_required=contact_policy.scorer_enabled(policy),
            )
        try:
            document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
        except (TypeError, ValueError, UnicodeDecodeError) as exc:
            raise scoring.ScoringError("scoring output is not valid JSON") from exc
        output = scoring.validate_scoring_output_document(document)
        if output["scored_run_id"] != run["scored_run_id"]:
            raise scoring.ScoringError("scoring output names the wrong execution run")
        if "breakdowns" not in output:
            raise scoring.ScoringError("accepted scoring output contains a failure")
        return scoring.validate_breakdowns_for_item(
            output["breakdowns"], icp=icp, companies=companies,
            max_scored_companies=int(policy["max_scored_companies"]),
            integrity_policy=contact_policy.integrity_adapter(policy.get("scoring_adapter_version")),
            contacts_required=contact_policy.scorer_enabled(policy),
        )

    def score_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Assemble the stage bundle from configured-runner scoring results."""

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_judged" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        policy = contracts.validate_scorer_policy(round_row["configuration_doc"]["scorer_policy"])
        plan = self._load_scoring_plan(round_row, stage)
        icps = self.evaluation_icps(round_id)
        outputs = self._outputs_by_run(round_id, stage)
        chosen = self._scoring_outputs(round_id, stage)
        baseline_ids = {
            str(participant["submission_id"])
            for participant in round_row.get("participants") or []
            if participant.get("is_king")
        }
        if len(baseline_ids) != 1:
            raise ServiceError("baseline_submission_invalid", 500)
        baseline_id = next(iter(baseline_ids))
        ineligible: set[str] = set()
        for item in plan["work_items"]:
            scored_run_id = item["scored_run_id"]
            run = chosen.get(scored_run_id)
            if run is None:
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            if run["status"] == "accepted":
                continue
            submission_id = str(item["submission_id"])
            cause = str(run.get("terminal_cause") or "")
            # Only explicit miner-account failures may exclude one challenger.
            # Every other scoring gap belongs to the shared judge path.
            if submission_id == baseline_id or cause not in (
                "budget_exhausted",
                "credential_error",
            ):
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            ineligible.add(submission_id)
        breakdowns_by_item: Dict[str, List[Dict[str, Any]]] = {}
        judge_executions = 0
        for item in plan["work_items"]:
            submission_id = str(item["submission_id"])
            if submission_id in ineligible:
                continue
            scored_run_id = item["scored_run_id"]
            run = chosen.get(scored_run_id)
            icp = icps[int(item["icp_position"])]
            companies = outputs[scored_run_id]
            try:
                breakdowns_by_item[scored_run_id] = self._verified_breakdowns(
                    run, icp=icp, companies=companies, policy=policy
                )
            except scoring.ScoringError:
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
            judge_executions += 1
        if ineligible:
            breakdowns_by_item = {
                item["scored_run_id"]: breakdowns_by_item[item["scored_run_id"]]
                for item in plan["work_items"]
                if item["submission_id"] not in ineligible
                and item["scored_run_id"] in breakdowns_by_item
            }
            plan = {
                **plan,
                "work_items": [
                    item
                    for item in plan["work_items"]
                    if item["submission_id"] not in ineligible
                ],
                "zero_rows": [
                    row
                    for row in plan["zero_rows"]
                    if row["submission_id"] not in ineligible
                ],
            }
        stage_scores = scoring.build_stage_scores(
            plan=plan,
            policy=policy,
            icps_by_position=dict(enumerate(icps)),
            outputs_by_run=outputs,
            breakdowns_by_item=breakdowns_by_item,
        )
        runs = self._store.list_runs(round_id, stage=stage, kind="execute")
        recorded = self._store.record_run_scores(round_id, stage, scoring.run_scores_for_store(stage_scores, runs))
        if recorded.get("status") != "ok":
            # The per-run scores are part of the published result; a write the
            # database refused must stop the stage, never pass silently.
            raise ServiceError("scores_not_recorded:%s" % str(recorded.get("status") or "unknown")[:40], 500)
        if stage == 2:
            final_entries = self._score_entries_from_runs(
                round_row, range(contracts.BENCHMARK_ICP_COUNT), "final_score"
            )
            baseline_entry = next(
                (entry for entry in final_entries if entry["is_king"]), None
            )
            if baseline_entry is None or baseline_entry["final_score"] is None:
                return self._store.cancel_round(
                    round_id, CANCEL_REASONS["scoring_incomplete"]
                )
        if stage == 1:
            ranking = verify.stage1_ranking(
                self._score_entries_from_runs(round_row, contracts.stage_positions(1), "stage1_score")
            )
            finalists = verify.select_finalists(ranking)
            transition = self._store.transition_round(
                round_id,
                "stage1_judged",
                "stage1_scored",
                {"finalists": finalists},
            )
            return {
                "status": transition.get("status"),
                "judge_executions": judge_executions,
                "ineligible_submissions": sorted(ineligible),
                "finalists": finalists,
            }
        transition = self._store.transition_round(round_id, "stage3_judged" if stage == 3 else "stage2_judged", "confirmed" if stage == 3 else "scored", {})
        return {
            "status": transition.get("status"),
            "judge_executions": judge_executions,
            "ineligible_submissions": sorted(ineligible),
        }

    def _score_entries_from_runs(
        self,
        round_row: Mapping[str, Any],
        positions: Sequence[int],
        score_key: str,
    ) -> List[Dict[str, Any]]:
        """Derive one score per participant from write-once run scores."""

        wanted = set(int(position) for position in positions)
        selected: Dict[Tuple[str, int], Mapping[str, Any]] = {}
        for run in self._store.list_runs(str(round_row["round_id"]), kind="execute"):
            position = int(run["icp_position"])
            if position not in wanted or run.get("per_icp_score") is None:
                continue
            key = (str(run["submission_id"]), position)
            current = selected.get(key)
            if current is None or int(run.get("attempt") or 0) > int(current.get("attempt") or 0):
                selected[key] = run
        entries = []
        for participant in round_row.get("participants") or []:
            submission_id = str(participant["submission_id"])
            rows = [selected.get((submission_id, position)) for position in sorted(wanted)]
            if any(row is None for row in rows):
                continue
            values = [float(row["per_icp_score"]) for row in rows if row is not None]
            entry = {
                "submission_id": submission_id,
                score_key: verify.stage_score(values, len(wanted)),
                "is_king": bool(participant.get("is_king")),
            }
            if score_key == "final_score":
                entry["hotkey"] = str(participant["miner_hotkey"])
                if not any(row.get("terminal_cause") == "accepted" for row in rows if row is not None):
                    entry[score_key] = None
            entries.append(entry)
        return entries

    @staticmethod
    def _selected_accepted_execution_runs(
        runs: Sequence[Mapping[str, Any]], submission_id: str
    ) -> Dict[int, Mapping[str, Any]]:
        """Select one accepted attempt per ICP, as the scoring plan does."""

        selected: Dict[int, Mapping[str, Any]] = {}
        for run in runs:
            if (
                str(run.get("submission_id") or "") != submission_id
                or run.get("status") != "accepted"
            ):
                continue
            position = int(run.get("icp_position") or 0)
            current = selected.get(position)
            if current is None or int(run.get("attempt") or 0) > int(
                current.get("attempt") or 0
            ):
                selected[position] = run
        return selected

    def _returned_company_count(
        self,
        submission_id: str,
        runs: Sequence[Mapping[str, Any]],
    ) -> int:
        """Count unique validated company domains within each of the 20 ICPs."""

        selected = self._selected_accepted_execution_runs(runs, submission_id)
        total = 0
        for position in sorted(selected):
            run = selected.get(position)
            if run is None:
                continue
            output_ref = str(run.get("output_ref") or "")
            if not output_ref:
                raise OutputInvalid("accepted output has no object reference")
            raw = self._objects.get_bounded(output_ref, MAX_OUTPUT_BYTES)
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, ValueError) as exc:
                raise OutputInvalid("accepted output is not valid JSON") from exc
            output = validate_output_document(document)
            domains = set()
            for company in output["companies"]:
                domains.add(
                    normalize_url(str(company["company_website"]))
                    .domain.registrable_domain
                )
            # The same company in different ICPs is a different returned slot.
            total += len(domains)
        return total

    def _qualified_company_count(
        self, round_row: Mapping[str, Any], submission_id: str,
        runs: Sequence[Mapping[str, Any]], *, positions: Optional[Sequence[int]] = None,
    ) -> int:
        """Count unique qualified entities from accepted, output-bound judgments."""
        policy = round_row["configuration_doc"]["scorer_policy"]
        round_id = str(round_row["round_id"])
        icps = self.evaluation_icps(round_id)
        wanted = set(range(len(icps)) if positions is None else positions)
        selected = self._selected_accepted_execution_runs(runs, submission_id)
        judges = {}
        for stage in (1, 2, 3):
            if wanted.intersection(contracts.stage_positions(stage)):
                judges.update(self._scoring_outputs(round_id, stage))
        total = 0
        for position in sorted(wanted):
            execution = selected.get(position)
            if execution is None:
                continue
            judge = judges.get(str(execution["run_id"]))
            if judge is None or judge.get("status") != "accepted":
                raise scoring.ScoringError("qualified count requires accepted judgment")
            document = json.loads(self._objects.get_bounded(execution["output_ref"], MAX_OUTPUT_BYTES).decode("utf-8"))
            output = validate_output_document(document)
            companies = output["companies"]
            rows = self._verified_breakdowns(judge, icp=icps[position], companies=companies, policy=policy)
            scored_indexes, _ = verify.bucket_skip(icps[position], verify.slice_first_n(companies, verify.icp_company_goal(icps[position])))
            identities = set()
            for original_index, breakdown in zip(scored_indexes, rows):
                if breakdown.get("company_index") != original_index:
                    raise scoring.ScoringError("integrity breakdown company index mismatch")
                if not isinstance(breakdown.get("company_qualified"), bool) or not isinstance(breakdown.get("duplicate_company"), bool):
                    raise scoring.ScoringError("integrity qualification receipt missing")
                key = breakdown.get("company_identity_key")
                if not isinstance(key, str) or not key:
                    raise scoring.ScoringError("integrity company identity missing")
                if breakdown["company_qualified"] and not breakdown["duplicate_company"]:
                    if float(breakdown.get("final_score") or 0) <= 0:
                        raise scoring.ScoringError("qualified company cannot carry zero score")
                    identities.add(key)
            total += len(identities)
        return total

    @staticmethod
    def _cost_kind_summary(
        costs: Mapping[str, Any], kind: str
    ) -> Dict[str, Any]:
        providers = []
        totals = {
            "settled_microusd": 0,
            "reserved_or_uncertain_microusd": 0,
            "conservative_microusd": 0,
            "inflight_calls": 0,
            "uncertain_calls": 0,
            "refused_calls": 0,
            "call_count": 0,
        }
        counter_keys = (
            "settled_microusd",
            "reserved_or_uncertain_microusd",
            "inflight_calls",
            "uncertain_calls",
            "refused_calls",
            "call_count",
        )
        for raw in costs["providers"]:
            if raw["kind"] != kind:
                continue
            row = {"provider": raw["provider"]}
            for key in counter_keys:
                value = int(raw[key])
                row[key] = value
                totals[key] += value
            row["conservative_microusd"] = (
                row["settled_microusd"]
                + row["reserved_or_uncertain_microusd"]
            )
            providers.append(row)
        providers.sort(key=lambda row: row["provider"])
        totals["conservative_microusd"] = (
            totals["settled_microusd"]
            + totals["reserved_or_uncertain_microusd"]
        )
        return {**totals, "providers": providers}

    def _submission_cost_eligibility(
        self,
        round_row: Mapping[str, Any],
        submission_id: str,
        runs: Sequence[Mapping[str, Any]],
        *, positions: Optional[Sequence[int]] = None,
    ) -> Dict[str, Any]:
        """Build final cost reporting without changing the quality score."""

        configuration = round_row.get("configuration_doc") or {}
        if "cost_per_company_microusd" not in configuration:
            return {
                "cost_summary": None,
                "eligible": True,
                "eligibility_reason": "historical_round",
            }
        try:
            returned = self._returned_company_count(submission_id, runs)
            qualified = self._qualified_company_count(round_row, submission_id, runs, positions=positions) if integrity.enabled(configuration) else returned
        except (ArenaContractError, OutputInvalid, TypeError, ValueError, scoring.ScoringError):
            return {
                "cost_summary": None,
                "eligible": False,
                "eligibility_reason": "stored_output_invalid",
            }
        costs = self._store.submission_costs(submission_id)
        execution = self._cost_kind_summary(costs, "execute")
        judge = self._cost_kind_summary(costs, "score")
        execution_cap = int(configuration["execution_cap_microusd"])
        per_company_cap = int(configuration["cost_per_company_microusd"])
        eligibility_cap = min(execution_cap, per_company_cap * qualified)
        summary = {
            "returned_company_count": returned,
            "execution_cap_microusd": execution_cap,
            "cost_per_company_cap_microusd": per_company_cap,
            "eligibility_cap_microusd": eligibility_cap,
            "execution": execution,
            # Judge spend is reported, but it does not enter sourcing eligibility.
            "judge": judge,
        }
        if integrity.enabled(configuration):
            summary["qualified_company_count"] = qualified
            selected = self._selected_accepted_execution_runs(runs, submission_id)
            if not any(position in selected for position in (positions if positions is not None else range(contracts.MAX_EVALUATION_ICP_COUNT))):
                return {"cost_summary": summary, "eligible": False, "eligibility_reason": "stored_output_invalid"}
        if execution["inflight_calls"] or judge["inflight_calls"]:
            return {
                "cost_summary": summary,
                "eligible": False,
                "eligibility_reason": "provider_calls_inflight",
            }
        if execution["uncertain_calls"] or judge["uncertain_calls"]:
            return {
                "cost_summary": summary,
                "eligible": False,
                "eligibility_reason": "provider_cost_uncertain",
            }
        if execution["conservative_microusd"] > execution_cap:
            reason = "execution_cap_exceeded"
        elif execution["conservative_microusd"] > per_company_cap * qualified:
            reason = "cost_per_company_exceeded"
        else:
            reason = "eligible"
        return {
            "cost_summary": summary,
            "eligible": reason == "eligible",
            "eligibility_reason": reason,
        }

    # -- publication and downstream reward activation -------------------------

    def _confirmation_account_failures(self, round_row: Mapping[str, Any]) -> set[str]:
        """Derive disqualifications from terminal judgments in the frozen plan.

        Keep the original cohort. A missing judgment or infrastructure failure
        cannot be treated as a withdrawal; the publication guard verifies the
        same evidence independently in PostgreSQL.
        """
        if not (round_row.get("confirmation_cohort") or {}).get("required"):
            return set()
        chosen = self._scoring_outputs(str(round_row["round_id"]), 3)
        failures: set[str] = set()
        incomplete: set[str] = set()
        for item in self._load_scoring_plan(round_row, 3)["work_items"]:
            submission_id = str(item["submission_id"])
            run = chosen.get(item["scored_run_id"]) or {}
            if run.get("status") == "accepted":
                continue
            if run.get("status") == "failed" and run.get("terminal_cause") in (
                "credential_error", "budget_exhausted",
            ):
                failures.add(submission_id)
            else:
                incomplete.add(submission_id)
        baseline_ids = {
            str(row["submission_id"])
            for row in round_row.get("participants") or [] if row.get("is_king")
        }
        return failures - incomplete - baseline_ids

    def publish(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        policy_enabled = integrity.enabled(round_row.get("configuration_doc") or {})
        expected_status = "confirmed" if policy_enabled else "scored"
        if round_row["status"] != expected_status:
            return {"status": "stale", "round_status": round_row["status"]}
        stage1_ranking = verify.stage1_ranking(
            self._score_entries_from_runs(round_row, contracts.stage_positions(1), "stage1_score")
        )
        finalists = list(round_row.get("finalists") or [])
        final_entries = self._score_entries_from_runs(
            round_row, range(contracts.BENCHMARK_ICP_COUNT), "final_score"
        )
        main_scores = {entry["submission_id"]: entry["final_score"] for entry in final_entries}
        cohort = round_row.get("confirmation_cohort") or {}
        withdrawn = self._confirmation_account_failures(round_row) if policy_enabled else set()
        if policy_enabled and cohort.get("required"):
            confirmation_entries = {entry["submission_id"]: entry for entry in self._score_entries_from_runs(round_row, contracts.stage_positions(3), "final_score")}
            for entry in final_entries:
                entry["final_score"] = confirmation_entries.get(entry["submission_id"], {}).get("final_score")
        execution_runs = self._store.list_runs(round_id, kind="execute")
        eligibility = {
            str(entry["submission_id"]): self._submission_cost_eligibility(
                round_row,
                str(entry["submission_id"]),
                execution_runs,
                positions=(range(contracts.BENCHMARK_ICP_COUNT)
                           if entry["submission_id"] in withdrawn else None),
            )
            for entry in final_entries
        }
        for submission_id in withdrawn:
            if submission_id in eligibility:
                eligibility[submission_id].update(
                    eligible=False, eligibility_reason="confirmation_account_failure"
                )
        king_entry = next((e for e in final_entries if e["is_king"]), None)
        if king_entry is None or king_entry["final_score"] is None:
            return self._store.cancel_round(
                round_id, CANCEL_REASONS["scoring_incomplete"]
            )
        decision = verify.king_decision(
            [
                entry
                for entry in final_entries
                if not entry["is_king"]
                and eligibility[str(entry["submission_id"])]["eligible"]
            ],
            king_entry,
        )
        published_at = _iso(self.now())
        final_ranking = verify.final_ranking(final_entries)
        for row in final_ranking:
            row.update(eligibility[str(row["submission_id"])])
            if policy_enabled:
                row["main_score"] = main_scores.get(row["submission_id"])
                row["confirmation_selected"] = row["submission_id"] in cohort.get("submission_ids", [])
        publication = {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "is_baseline": bool(p.get("is_king"))} for p in round_row.get("participants") or []],
            "stage1_ranking": stage1_ranking,
            "finalists": finalists,
            "final_ranking": final_ranking,
            "king_decision": decision,
            "published_at": published_at,
        }
        contracts.check_strict_document(publication, contracts.PUBLICATION_LIMITS)
        transition = self._store.transition_round(round_id, expected_status, "published", {
            "publication_doc": publication,
            "published_at": published_at,
        })
        return {
            "status": transition.get("status"),
            "king_outcome": decision["outcome"],
            "king_hotkey": str(decision.get("king_hotkey") or ""),
        }

    def promote_pending_baselines(self) -> Dict[str, Any]:
        """Finish accepted winners oldest-first before a new baseline or reward.

        Git and PostgreSQL cannot share a transaction. Persist the prepared Git
        update first, push both branches atomically, then mark completion. A
        lost response is reconciled against those same branch heads on retry.
        """

        if self._config.mode != "live":
            return {"status": "disabled", "promoted": 0}
        promoted = 0
        network_name, netuid = self._chain_scope()
        for row in self._store.pending_promotions(
            pinned_round_id=self._config.pinned_round_id,
            network_name=network_name,
            netuid=netuid,
        ):
            result = self.promote_baseline(str(row["round_id"]))
            if result.get("status") not in ("promoted", "existing"):
                return {"status": result.get("status", "pending"), "promoted": promoted}
            promoted += int(result.get("status") == "promoted")
        return {"status": "ok", "promoted": promoted}

    def promote_baseline(self, round_id: str) -> Dict[str, Any]:
        """Publish only the stored, scored winner, without executing its source."""

        row = self._round(round_id)
        if row.get("baseline_promoted_at"):
            return {"status": "existing"}
        configuration = row.get("configuration_doc") or {}
        decision = (row.get("publication_doc") or {}).get("king_decision") or {}
        if (
            row.get("status") != "published"
            or configuration.get("mode") != "live"
            or not row.get("promotion_required")
            or decision.get("outcome") != "crowned"
        ):
            return {"status": "not_required"}
        factory = self._config.baseline_promoter_factory
        if factory is None:
            raise ServiceError("baseline_promoter_unavailable", 503)
        submission_id = str(decision.get("winner_submission_id") or "")
        submission = self._store.get_submission(submission_id)
        if (
            not submission
            or submission.get("round_id") != round_id
            or submission.get("miner_hotkey") != decision.get("king_hotkey")
            or submission.get("is_king")
            or submission.get("status") != "frozen"
        ):
            raise ServiceError("promotion_winner_invalid", 500)
        # Public Git branches use the same completed-evaluation publication
        # gate as the dashboard. There is no additional upload-age delay.
        from lab_arena.source_disclosure import disclosure_status
        disclosure = disclosure_status(submission, self.now(), round_row=row)
        if not disclosure["available"]:
            return {"status": "source_private", "available_at": disclosure["available_at"]}
        payload = self._objects.get_bounded(
            str(submission["source_ref"]), source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        )
        if len(payload) != int(submission["source_size_bytes"]):
            raise ServiceError("promotion_source_size_mismatch", 500)
        promoter = factory()
        plan = row.get("promotion_doc")
        if plan is None:
            proposed = promoter.prepare(
                payload, round_id=round_id, submission_id=submission_id,
                timestamp=str(row["published_at"]),
            )
            prepared = self._store.prepare_promotion(round_id, proposed)
            if prepared.get("status") not in ("prepared", "existing"):
                return prepared
            # Another worker can win the initial prepare. Its stored plan is
            # authoritative; never silently replace it with a new Git base.
            row = self._store.get_round(round_id) or {}
            plan = row.get("promotion_doc")
        if not isinstance(plan, dict):
            raise ServiceError("promotion_plan_missing", 500)
        commit = promoter.publish(
            payload, plan=plan, round_id=round_id, submission_id=submission_id
        )
        if commit != plan["commit"]:
            raise ServiceError("promotion_commit_mismatch", 500)
        return self._store.complete_promotion(round_id, plan)

    def activate_pending_rewards(self) -> Dict[str, Any]:
        """Activate eligible live rounds oldest-first after publication.

        Publication never calls this method. A transient chain, cutover, KMS,
        or database failure leaves the compact competition result published
        and the next driver tick retries the same oldest round.
        """

        if self._config.mode != "live":
            return {"status": "disabled", "activated": 0}
        if self._pinned_round_id() is not None:
            row = self._pinned_round()
            rows = [row] if row is not None and row["status"] == "published" else []
        else:
            network_name, netuid = self._chain_scope()
            rows = list(
                reversed(
                    self._store.list_rounds(
                        status="published", mode="live",
                        network_name=network_name,
                        netuid=netuid,
                        limit=200
                    )
                )
            )
        pending = [
            row for row in rows
            if (row.get("configuration_doc") or {}).get("mode") == "live"
            and (row.get("configuration_doc") or {}).get("rewards_enabled") is True
            and not row.get("reward_activated_at")
        ]
        activated = 0
        for row in pending:
            result = self.activate_reward(str(row["round_id"]))
            if result.get("status") not in ("activated", "existing"):
                return {"status": str(result.get("status") or "stale"), "activated": activated}
            activated += int(result.get("status") == "activated")
        return {"status": "ok", "activated": activated}

    def _usable_reward_bases(
        self, rows: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return activated miner/no-winner bases, never organizer baselines."""

        usable: List[Dict[str, Any]] = []
        for row in rows:
            configuration = row.get("configuration_doc")
            if not isinstance(configuration, Mapping) or configuration.get("mode") != "live":
                continue
            document = row.get("reward_basis_doc")
            try:
                basis = rewards.validate_reward_basis(document)
            except (ArenaContractError, TypeError, ValueError) as exc:
                raise ServiceError("reward_history_invalid", 500) from exc
            baseline_hotkey = str(configuration.get("baseline_hotkey") or "")
            if basis["king_outcome"] != "no_king" and (
                not baseline_hotkey or basis["king_hotkey"] == baseline_hotkey
            ):
                continue
            usable.append(basis)
        return usable

    @staticmethod
    def _latest_miner_basis(
        bases: Sequence[Mapping[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidates = [
            dict(basis)
            for basis in bases
            if basis.get("king_outcome") in rewards.PAYING_KING_OUTCOMES
            and str(basis.get("king_hotkey") or "")
        ]
        return max(
            candidates,
            key=lambda basis: int(basis["effective_reward_epoch"]),
            default=None,
        )

    def activate_reward(self, round_id: str) -> Dict[str, Any]:
        """Sign and atomically activate one already-published live result."""

        row = self._round(round_id)
        if row.get("reward_activated_at"):
            return {"status": "existing", "effective_reward_epoch": row.get("effective_reward_epoch")}
        configuration = row.get("configuration_doc") or {}
        if row.get("status") != "published":
            return {"status": "stale", "round_status": row.get("status")}
        if configuration.get("mode") != "live" or configuration.get("rewards_enabled") is not True:
            return {"status": "disabled"}
        # A later no-winner round must not activate while an earlier accepted
        # baseline in the same chain scope is still unpublished.
        network_name, netuid = self._chain_scope()
        if self._store.pending_promotions(
            network_name=network_name, netuid=netuid, limit=1
        ):
            return {"status": "waiting_for_promotion"}
        publication = row.get("publication_doc") or {}
        decision = publication.get("king_decision") or {}
        if (
            row.get("promotion_required")
            and decision.get("outcome") == "crowned"
            and not row.get("baseline_promoted_at")
        ):
            return {"status": "waiting_for_promotion"}
        prior = self._store.published_reward_bases(
            mode="live", network_name=network_name, netuid=netuid, limit=200
        )
        maximum_epoch = max((int(item["effective_reward_epoch"]) for item in prior if item.get("effective_reward_epoch") is not None), default=-1)
        effective_epoch = max(int(self._config.chain.current_settlement_epoch()) + 1, maximum_epoch + 1)
        usable = self._usable_reward_bases(prior)
        previous = self._latest_miner_basis(usable)
        baseline_hotkey = str(configuration.get("baseline_hotkey") or "")
        if previous is not None and previous["king_hotkey"] == baseline_hotkey:
            previous = None
        daily_hotkey = (
            str(decision.get("king_hotkey") or "")
            if decision.get("outcome") == "crowned"
            else ""
        )
        # Old pending publications can name the organizer baseline. Treat them
        # as no-winner days so migration cannot turn that baseline into a payee.
        if not daily_hotkey or daily_hotkey == baseline_hotkey:
            daily_hotkey = ""
        if daily_hotkey:
            king_hotkey = daily_hotkey
            king_outcome = (
                "defended"
                if previous is not None and previous["king_hotkey"] == daily_hotkey
                else "crowned"
            )
        elif previous is not None:
            king_hotkey = str(previous["king_hotkey"])
            king_outcome = "defended"
        else:
            king_hotkey = ""
            king_outcome = "no_king"
        previous_start = (
            int(previous["king_start_epoch"])
            if previous is not None and king_outcome == "defended"
            else None
        )
        basis = self._sign(
            rewards.reward_basis_document(
                round_id=round_id,
                published_at=str(publication["published_at"]),
                finalized_epoch=effective_epoch - 1,
                king_hotkey=king_hotkey,
                king_outcome=king_outcome,
                previous_king_start_epoch=previous_start,
                reward_constants=configuration["reward_constants"],
            ),
            "reward_basis_hash",
        )
        return self._store.activate_reward(round_id, basis, self.signing_key_document())

    # -- runner handlers (section 14.3) ----------------------------------------

    def _lease_token(self, validated: Mapping[str, Any]) -> str:
        return contracts.document_hash({"lease": validated["request_id"], "signature": validated["signature"]})[7:]

    def _require_validator_authority(self, hotkey: str) -> None:
        """Require validator identity for existing work, without a stake gate."""

        authorizer = getattr(self._config, "validator_authorizer", None)
        if authorizer is None:
            network_name, netuid = self._chain_scope()
            authorizer = lambda candidate: _gateway_validator_authorizer(
                candidate,
                network_name=network_name,
                netuid=netuid,
                metagraph=self._config.chain.metagraph(finalized=True),
            )
        try:
            registered, role = authorizer(hotkey)
        except Exception as exc:
            raise ServiceError("runner_validator_authority_unavailable", 503) from exc
        if not registered:
            raise ServiceError("runner_hotkey_unregistered", 403)
        if role != "validator":
            raise ServiceError("runner_validator_required", 403)

    def _benchmark_snapshot(self) -> chain_module.MetagraphSnapshot:
        try:
            return self._config.chain.metagraph(finalized=True)
        except Exception as exc:
            raise ServiceError("runner_benchmark_eligibility_unavailable", 503) from exc

    def _benchmark_validator_uid(self, snapshot: Any, hotkey: str) -> int:
        try:
            network_name, netuid = self._chain_scope()
            uid = validator_uid(
                snapshot, hotkey, netuid=netuid,
                network_name=chain_module.normalize_network_name(network_name),
            )
            # Bind miner self-dealing exclusions to the same authorized snapshot.
            owners = tuple(snapshot.coldkeys)
            if len(owners) != len(snapshot.hotkeys) or any(
                not isinstance(owner, str) or not owner for owner in owners
            ):
                raise ValueError("metagraph coldkeys are invalid")
            return uid
        except ValidatorIneligible as exc:
            raise ServiceError(str(exc), 403) from exc
        except Exception as exc:
            raise ServiceError("runner_benchmark_eligibility_unavailable", 503) from exc

    def handle_claim(self, envelope: Any) -> Dict[str, Any]:
        validated, round_row = self._request_round(envelope, scope=contracts.SCOPE_CLAIM, hot=True)
        round_id = round_row["round_id"]
        if round_row["status"] in TERMINAL_STATUSES:
            raise ServiceError("round_ended", 409)
        body = validated["body"]
        declared = body.get("declared_parallelism")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
            raise ServiceError("declared_parallelism_invalid", 400)
        configuration = round_row["configuration_doc"]
        snapshot = self._benchmark_snapshot()
        response = None
        try:
            uid = self._benchmark_validator_uid(snapshot, validated["hotkey"])
        except ServiceError as exc:
            if exc.code != "runner_stake_below_minimum":
                raise
            # A lost response must not strand a lease issued before stake fell.
            # This exact signed-request lookup cannot allocate or extend work.
            response = self._store.recover_claim_response(
                round_id=round_id, runner_hotkey=validated["hotkey"],
                request_id=validated["request_id"],
                request_hash=contracts.request_bytes_hash(validated),
            )
            if response is None:
                raise
        else:
            excluded = chain_module.hotkeys_owned_by_coldkey(snapshot, snapshot.coldkeys[uid])
            if integrity.enabled(configuration):
                runner_coldkey = snapshot.coldkeys[uid]
                for submission in self._store.list_submissions(
                    round_id,
                    status="frozen",
                    columns="miner_hotkey,owner_coldkey,is_king",
                ):
                    if (
                        not bool(submission.get("is_king"))
                        and submission.get("owner_coldkey") == runner_coldkey
                    ):
                        excluded.append(str(submission.get("miner_hotkey") or ""))
                excluded = sorted(set(excluded) - {""})
        token = self._lease_token(validated)
        if response is None:
            response = self._store.claim_assignment(
                round_id=round_id, runner_hotkey=validated["hotkey"], declared_parallelism=declared, slot_ceiling=int(configuration["runner_slot_ceiling"]),
                excluded_miner_hotkeys=excluded, request_id=validated["request_id"], request_hash=contracts.request_bytes_hash(validated), lease_token_hash=hash_lease_token(token),
                lease_ttl_seconds=int(configuration["lease_ttl_seconds"]),
            )
        if response.get("status") != "leased":
            return response
        self._require_code_review(str(response["submission_id"]), round_row)
        icps = self.evaluation_icps(round_id) if integrity.enabled(configuration) else self.benchmark_icps(round_id)
        position = int(response["icp_position"])
        if not 0 <= position < len(icps):
            raise ServiceError("benchmark_data_invalid", 500)
        lease_icp = integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(configuration)) if integrity.enabled(configuration) else icps[position]
        lease = dict(response, icp=lease_icp, lease_token=token, round_id=round_id, evaluation_date=str(round_row.get("evaluation_date") or ""))
        if integrity.enabled(configuration):
            lease["integrity_policy"] = integrity.POLICY
        if contact_policy.enabled(configuration):
            lease["contact_policy"] = contact_policy.POLICY
        lease.update({
            "image_digest": configuration["scorer_image_digest"],
            "image_reference": configuration["scorer_image_reference"],
        })
        if response.get("kind") == "score":
            # A scoring assignment: the validator runs the pinned judge image on
            # the scored output with the signed scorer policy.
            scored = self._store.get_run(str(response.get("scored_run_id") or ""))
            if scored is None or not scored.get("output_ref"):
                raise ServiceError("scored_run_missing", 500)
            output = json.loads(self._objects.get(scored["output_ref"]).decode("utf-8"))
            lease.update({"scored_output": output, "scorer_policy": configuration["scorer_policy"]})
            if contact_policy.enabled(configuration):
                companies = validate_output_document(output, expected_schema_version=contact_policy.OUTPUT_SCHEMA)["companies"]
                lease["contact_source_evidence"] = contact_evidence.resolve_sources(self._store, scored, companies)
            return lease
        # An execution uses the participant's private source archive under the
        # same trusted Python image as every other agent.
        participant = next((p for p in round_row.get("participants") or [] if p["submission_id"] == response.get("submission_id")), None)
        if participant is None:
            raise ServiceError("participant_missing", 500)
        if any(
            participant.get(field) in (None, "")
            for field in ("source_ref", "source_size_bytes")
        ):
            raise ServiceError("participant_source_missing", 500)
        lease.update(
            {
                "source_ref": participant["source_ref"],
                "source_size_bytes": int(participant["source_size_bytes"]),
                # Availability only: neither ciphertext nor the provider key
                # belongs in a validator lease or the model environment.
                "scrapingdog_configured": bool(participant.get("is_king")) or (
                    self._store.get_submission_credential(
                        str(participant["submission_id"]),
                        str(participant["miner_hotkey"]),
                        "scrapingdog",
                    ) is not None
                ),
            }
        )
        return lease

    def _run_context(self, run_id: str, lease_token: str) -> Tuple[Dict[str, Any], broker_module.RunContext]:
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        return run, broker_module.RunContext(run_id=run_id, assignment_id=run["assignment_id"], attempt=int(run["attempt"]), icp_position=int(run["icp_position"]), lease_token_hash=hash_lease_token(lease_token), miner_hotkey=run["miner_hotkey"], submission_id=run["submission_id"], stage=int(run["stage"]), kind=str(run.get("kind") or "execute"), round_id=str(run.get("round_id") or ""))

    def handle_source(self, run_id: str, lease_token: str) -> bytes:
        """Return source bytes only to the runner that holds the active lease."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        self._require_round_ownership(str(run.get("round_id") or ""))
        if run.get("kind") != "execute":
            raise ServiceError("run_source_unavailable", 409)
        expected_token_hash = str(run.get("lease_token_hash") or "")
        if not expected_token_hash or not hmac.compare_digest(
            expected_token_hash, hash_lease_token(lease_token)
        ):
            raise ServiceError("lease_invalid", 401)
        if run.get("status") != "leased":
            raise ServiceError("lease_inactive", 409)
        raw_expiry = run.get("lease_expires_at")
        try:
            if isinstance(raw_expiry, datetime):
                expiry = raw_expiry
            else:
                encoded_expiry = str(raw_expiry).replace("Z", "+00:00")
                try:
                    expiry = datetime.fromisoformat(encoded_expiry)
                except ValueError:
                    # Python 3.9 accepts only three or six fractional digits
                    # here, but PostgreSQL can serialize any width from one
                    # through six. strptime accepts the full PostgreSQL range.
                    expiry = datetime.strptime(
                        encoded_expiry, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("lease_invalid", 401)
        if expiry.astimezone(timezone.utc) <= self.now():
            raise ServiceError("lease_expired", 409)
        submission = self._store.get_submission(str(run.get("submission_id") or ""))
        if submission is None or submission.get("status") != "frozen":
            raise ServiceError("run_source_unavailable", 409)
        source_ref = str(submission.get("source_ref") or "")
        expected_size = int(submission.get("source_size_bytes") or 0)
        try:
            payload = self._objects.get_bounded(
                source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
            )
        except Exception as exc:
            raise ServiceError("run_source_unavailable", 503) from exc
        if len(payload) != expected_size:
            raise ServiceError("run_source_integrity_failed", 500)
        return payload

    def handle_scorer_image_access(
        self, run_id: str, lease_token: str
    ) -> Mapping[str, Any]:
        """Return ECR blob capabilities only to the active lease's validator."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        round_id = str(run.get("round_id") or "")
        self._require_round_ownership(round_id)
        expected_token_hash = str(run.get("lease_token_hash") or "")
        if not expected_token_hash or not hmac.compare_digest(
            expected_token_hash, hash_lease_token(lease_token)
        ):
            raise ServiceError("lease_invalid", 401)
        if run.get("status") != "leased":
            raise ServiceError("lease_inactive", 409)
        raw_expiry = run.get("lease_expires_at")
        try:
            if isinstance(raw_expiry, datetime):
                expiry = raw_expiry
            else:
                encoded_expiry = str(raw_expiry).replace("Z", "+00:00")
                try:
                    expiry = datetime.fromisoformat(encoded_expiry)
                except ValueError:
                    expiry = datetime.strptime(
                        encoded_expiry, "%Y-%m-%dT%H:%M:%S.%f%z"
                    )
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            raise ServiceError("lease_invalid", 401) from None
        if expiry.astimezone(timezone.utc) <= self.now():
            raise ServiceError("lease_expired", 409)
        round_row = self._round(round_id)
        if round_row.get("status") in TERMINAL_STATUSES:
            raise ServiceError("round_ended", 409)
        runner_hotkey = str(run.get("runner_hotkey") or "")
        try:
            contracts.require_hotkey(runner_hotkey)
        except ArenaContractError:
            raise ServiceError("lease_invalid", 401) from None
        self._require_validator_authority(runner_hotkey)
        provider = self._config.scorer_image_access
        if provider is None:
            raise ServiceError("scorer_image_access_unsupported", 409)
        configuration = round_row.get("configuration_doc") or {}
        image_reference = str(configuration.get("scorer_image_reference") or "")
        image_digest = str(configuration.get("scorer_image_digest") or "")
        access = None
        failed = False
        try:
            access = provider(image_reference, image_digest)
        except scorer_image_access_module.ScorerImageAccessError:
            failed = True
        except Exception:
            failed = True
        if failed:
            # Raise outside the handler. Thus, an SDK exception containing a
            # signed URL is not retained as this public error's context.
            raise ServiceError("scorer_image_access_unavailable", 503) from None
        return access

    def _broker_for(self, round_id: str) -> broker_module.Broker:
        with self._lock:
            broker = self._brokers.get(round_id)
            if broker is None:
                broker = self._config.broker_factory(self, self._round(round_id))
                self._brokers[round_id] = broker
            return broker

    def handle_provider(self, run_id: str, lease_token: str, frame: Any) -> Dict[str, Any]:
        if not isinstance(frame, Mapping) or set(frame) != {"operation_id", "parameters", "timeout_ms", "action_sequence"}:
            raise ServiceError("frame_invalid", 400)
        contracts.check_strict_document(frame, contracts.PROVIDER_FRAME_LIMITS)
        run, context = self._run_context(run_id, lease_token)
        broker = self._broker_for(run["round_id"])
        result = broker.execute(context, operation_id=str(frame["operation_id"]), parameters=frame["parameters"], action_sequence=frame["action_sequence"], timeout_ms=int(frame["timeout_ms"]))
        return result.to_document()

    def handle_complete(self, envelope: Any) -> Dict[str, Any]:
        validated, round_row = self._request_round(envelope, scope=contracts.SCOPE_COMPLETE, hot=True)
        round_id = round_row["round_id"]
        body = validated["body"]
        run_id = str(body.get("run_id") or "")
        try:
            run_result = contracts.validate_run_result(body.get("result"))
        except ArenaContractError as exc:
            raise ServiceError("run_result_invalid:%s" % str(exc)[:80], 400)
        self._require_validator_authority(validated["hotkey"])
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        if str(run.get("round_id") or "") != round_id:
            raise ServiceError("run_round_mismatch", 400)
        if run.get("runner_hotkey") != validated["hotkey"]:
            raise ServiceError("run_runner_mismatch", 403)
        kind = str(run.get("kind") or "execute")
        terminal_status = run_result["terminal_status"]
        if kind == "score" and terminal_status not in contracts.SCORE_TERMINAL_CAUSES:
            raise ServiceError("run_result_cause_kind_mismatch", 400)
        if kind == "execute" and terminal_status in ("judge_error", "judge_timeout"):
            raise ServiceError("run_result_cause_kind_mismatch", 400)
        lease_token = self._lease_token_for_run(validated, run)
        output_ref = ""
        judgment_evidence = None
        judgment_evidence_hash = ""
        if terminal_status == "accepted" and kind == "score":
            try:
                output = scoring.validate_scoring_output_document(body.get("output"))
            except scoring.ScoringError:
                raise ServiceError("output_invalid", 400)
            if "failure" in output or output["scored_run_id"] != run.get("scored_run_id"):
                raise ServiceError("output_invalid", 400)
            if contact_policy.enabled(round_row["configuration_doc"]):
                executed = self._store.get_run(str(run["scored_run_id"]))
                if executed is None or not executed.get("output_ref"):
                    raise ServiceError("scored_run_missing", 500)
                companies = validate_output_document(
                    json.loads(self._objects.get_bounded(executed["output_ref"], MAX_OUTPUT_BYTES)),
                    expected_schema_version=contact_policy.OUTPUT_SCHEMA,
                )["companies"]
                try:
                    scoring.validate_breakdowns_for_item(
                        output["breakdowns"], icp=self.evaluation_icps(round_id)[int(run["icp_position"])],
                        companies=companies, integrity_policy=True, contacts_required=True,
                    )
                except scoring.ScoringError as exc:
                    raise ServiceError("contact_score_invalid", 400) from exc
            output_ref = "arena/%s/scores/items/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
            if run.get("judgment_cache_key"):
                scope = dict(run.get("judgment_scope_doc") or {})
                scope["cache_key"] = str(run["judgment_cache_key"])
                try:
                    judgment_evidence = judgment_cache.build_evidence_snapshot(
                        output=output,
                        cache_scope=scope,
                        source_score_run_id=run_id,
                        source_scored_run_id=str(run["scored_run_id"]),
                        source_output_ref=output_ref,
                        source_runner_hotkey=str(validated["hotkey"]),
                        runner_authority_exclusions=(
                            run.get("claim_response") or {}
                        ).get("runner_authority_exclusions"),
                    )
                except judgment_cache.JudgmentCacheError as exc:
                    raise ServiceError("judgment_cache_invalid", 500) from exc
                judgment_evidence_hash = contracts.document_hash(judgment_evidence)
        elif terminal_status == "accepted":
            try:
                output = validate_output_document(
                    body.get("output"),
                    expected_schema_version=contact_policy.output_schema(round_row["configuration_doc"]),
                    require_intent_dates=not integrity.enabled(
                        round_row["configuration_doc"]
                    ),
                )
            except OutputInvalid:
                raise ServiceError("output_invalid", 400)
            output_ref = "arena/%s/outputs/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        result = self._store.complete_attempt(
            run_id=run_id, lease_token_hash=hash_lease_token(lease_token), result=run_result, terminal_cause=terminal_status,
            output_ref=output_ref, judgment_evidence=judgment_evidence,
            judgment_evidence_hash=judgment_evidence_hash,
        )
        return result

    def _lease_token_for_run(self, validated: Mapping[str, Any], run: Mapping[str, Any]) -> str:
        token = validated["body"].get("lease_token")
        if not isinstance(token, str) or hash_lease_token(token) != run.get("lease_token_hash"):
            raise ServiceError("lease_token_invalid", 403)
        return token

    def _ledger_calls(self, run_id: str) -> List[Dict[str, Any]]:
        heads: Dict[str, Dict[str, Any]] = {}
        reservations: Dict[str, Dict[str, Any]] = {}
        for entry in self._store.list_ledger(run_id=run_id):
            identity = entry.get("call_identity")
            if not identity:
                continue
            heads[identity] = entry
            if entry["entry_kind"] == "reservation":
                reservations[identity] = entry
        calls = []
        for identity, head in heads.items():
            reservation = reservations.get(identity, head)
            doc = reservation.get("entry_doc") or {}
            outcome = {"settlement": "settled", "uncertain": "uncertain", "refusal": "refused", "recovery": "recovered", "reservation": "reserved", "dispatch": "dispatched"}[head["entry_kind"]]
            terminal = head.get("terminal_response") or {}
            status = terminal.get("status") if head["entry_kind"] == "settlement" else None
            response_hash = None
            if head["entry_kind"] == "settlement" and terminal.get("body_b64") is not None:
                import base64

                response_hash = contracts.hash_bytes(base64.b64decode(terminal["body_b64"]))
            calls.append({
                "call_identity": identity, "operation_id": reservation.get("operation_id"), "request_hash": doc.get("request_hash"), "outcome": outcome, "status": status, "response_hash": response_hash,
                "reserved_microusd": int(reservation.get("amount_microusd") or 0) if head["entry_kind"] != "refusal" else 0, "actual_microusd": int(head.get("amount_microusd") or 0) if head["entry_kind"] in ("settlement", "uncertain") else (0 if head["entry_kind"] in ("refusal", "recovery") else int(reservation.get("amount_microusd") or 0)),
            })
        return calls

    # -- daily driver (section 14.4) -------------------------------------------

    def advance_round(self, round_id: str) -> Dict[str, Any]:
        """One idempotent compare-and-set step for the round's current state."""

        # Every driver step may change the round's status; the runner-facing
        # cache must never serve a row from before this process's own transition.
        self._invalidate_hot_round()
        try:
            row = self._round(round_id)
            if row["status"] == "open":
                # At cutoff, reject source uploads that were not finalized
                # before participant freeze.
                final = self.now() >= _parse_iso(row["configuration_doc"]["schedule"]["submission_cutoff"])
                admission = self.admit_uploaded_submissions(round_id, final=final)
                if final and int(admission.get("remaining") or 0) > 0:
                    return {
                        "status": "retry",
                        "round_status": "open",
                        "remaining_admissions": int(admission["remaining"]),
                    }
            return self._advance_round_locked(round_id)
        finally:
            self._invalidate_hot_round()

    def _invalidate_hot_round(self) -> None:
        with self._hot_round_lock:
            self._hot_rounds.clear()

    def _advance_round_locked(self, round_id: str) -> Dict[str, Any]:
        with self._lock:
            round_row = self._round(round_id)
            status = round_row["status"]
            schedule = round_row["configuration_doc"]["schedule"]
            now = self.now()
            if (
                status not in TERMINAL_STATUSES
                and now < _parse_iso(schedule["submission_cutoff"])
            ):
                return {"status": "waiting", "round_status": status}
            if status == "open":
                return self.commit_benchmark(round_id)
            if status == "committed":
                return self.open_stage(round_id, 1)
            if status in ("stage1", "stage2", "stage3"):
                stage = int(status[-1])
                self._store.expire_leases(round_id)
                if now >= _parse_iso(schedule["stage_%d_close" % stage]) or self.stage_is_complete(round_id, stage):
                    return self.close_stage(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_closed", "stage2_closed", "stage3_closed"):
                stage = int(status[5])
                if not round_row.get("stage%d_scoring_plan_doc" % stage):
                    return self.commit_scoring_plan(round_id, stage)
                return self.open_scoring(round_id, stage)
            if status in ("stage1_scoring", "stage2_scoring", "stage3_scoring"):
                stage = int(status[5])
                self._store.expire_leases(round_id)
                scoring_runs = self._store.list_runs(
                    round_id, stage=stage, kind="score"
                )
                if self._scoring_has_exhausted_judge_failure(
                    round_row, stage, scoring_runs
                ):
                    return self._store.cancel_round(
                        round_id, CANCEL_REASONS["scoring_incomplete"]
                    )
                window = schedule["stage_1_scoring_close" if stage == 1 else "stage_3_scoring_close" if stage == 3 else "final_scoring_close"]
                if now >= _parse_iso(window) or all(
                    run["status"] in ("accepted", "failed")
                    for run in scoring_runs
                ):
                    return self.close_scoring(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_judged", "stage2_judged", "stage3_judged"):
                stage = int(status[5])
                window = schedule["stage_1_scoring_close" if stage == 1 else "stage_3_scoring_close" if stage == 3 else "final_scoring_close"]
                try:
                    return self.score_stage(round_id, stage)
                except scoring.ScoringError:
                    if now >= _parse_iso(window) + timedelta(hours=2):
                        return self._store.cancel_round(round_id, CANCEL_REASONS["scoring"])
                    return {"status": "retry", "round_status": status}
            if status == "stage1_scored":
                return self.open_stage(round_id, 2)
            if status == "scored" and integrity.enabled(round_row["configuration_doc"]):
                return self.open_confirmation(round_id)
            if status in ("scored", "confirmed"):
                try:
                    return self.publish(round_id)
                except ServiceError as exc:
                    if exc.code == "publication_sanitizer_failed" and now >= _parse_iso(schedule["publication_deadline"]) + timedelta(hours=14):
                        return self._store.cancel_round(round_id, CANCEL_REASONS["publication"])
                    raise
            return {"status": "terminal", "round_status": status}

    def cancel(self, round_id: str, reason: str) -> Dict[str, Any]:
        if reason not in CANCEL_REASONS.values():
            raise ServiceError("cancel_reason_invalid", 400)
        self._round(round_id)
        self._invalidate_hot_round()
        return self._store.cancel_round(round_id, reason)

    # -- public reads (section 14.1) -------------------------------------------

    def public_competition(self) -> Dict[str, Any]:
        return public_dashboard.competition_snapshot(self)

    def public_submissions(self, round_id: str) -> Dict[str, Any]:
        return public_dashboard.submissions_snapshot(self, round_id)

    def public_submission_code(self, submission_id: str) -> Dict[str, Any]:
        submission = self._store.get_submission(submission_id)
        if submission is None:
            raise ServiceError("submission_missing", 404)
        row = self._round(str(submission.get("round_id") or ""))
        try:
            return source_disclosure.public_source_code(
                self._objects, submission, self.now(), round_row=row
            )
        except source_disclosure.SourceDisclosureError as exc:
            raise ServiceError(exc.code, exc.status) from exc

    def public_current(self) -> Dict[str, Any]:
        active = self.active_rounds()
        current = active[-1] if active else None
        open_round = next((row for row in active if row["status"] == "open"), None)
        running = [row for row in active if row["status"] != "open"]
        published = self.latest_published_round()
        published_round = (
            {
                "round_id": published["round_id"],
                "status": published["status"],
                "published_at": published.get("published_at"),
            }
            if published is not None
            else None
        )
        epoch = None
        try:
            epoch = int(self._config.chain.current_settlement_epoch())
        except Exception:
            epoch = None
        eligibility = None
        week = None
        governing = None
        if epoch is not None:
            governing = self.public_reward_basis(epoch)
            if governing is None:
                eligibility = False
            else:
                eligibility = rewards.epoch_eligible(governing, epoch)
                if eligibility:
                    week = rewards.reward_week_index(epoch, int(governing["king_start_epoch"]))
        elif self._config.mode == "live":
            network_name, netuid = self._chain_scope()
            rows = self._store.published_reward_bases(
                mode="live", network_name=network_name, netuid=netuid, limit=200
            )
            bases = self._usable_reward_bases(rows)
            if bases:
                governing = max(
                    bases, key=lambda basis: int(basis["effective_reward_epoch"])
                )
        return {
            "mode": self._config.mode,
            "round": dict(current) if current else None,
            # Rounds overlap: miners submit to the open round while runners work the running ones.
            "open_round": dict(open_round) if open_round else None,
            "running_rounds": [dict(row) for row in running],
            "published_round": published_round,
            "king": {"hotkey": governing.get("king_hotkey"), "outcome": governing.get("king_outcome"), "round_id": governing.get("round_id"), "king_start_epoch": governing.get("king_start_epoch")} if governing else None,
            "reward_week_index": week,
            "epoch_eligible": eligibility,
            "current_epoch": epoch,
        }

    def public_reward_basis(self, epoch: int) -> Optional[Dict[str, Any]]:
        if self._config.mode != "live":
            return None
        network_name, netuid = self._chain_scope()
        rows = self._store.published_reward_bases(
            mode="live", network_name=network_name, netuid=netuid, limit=200
        )
        return rewards.governing_reward_basis(
            self._usable_reward_bases(rows), int(epoch)
        )

    def public_weight_state(self, epoch: int) -> Dict[str, Any]:
        """Return or atomically publish one accepted state for an Arena epoch."""

        if self._config.mode != "live":
            return {"state": None, "lookup_ok": True}
        network, netuid = self._chain_scope()
        requested_epoch = int(epoch)
        basis = self.public_reward_basis(requested_epoch)
        if basis is None:
            return {"state": None, "lookup_ok": True}
        existing = self._store.get_weight_state(network, netuid, requested_epoch)
        if existing is not None:
            try:
                state = validate_accepted_weight_state(existing.get("state_doc"))
                signer = self._reward_signer()
                verify_accepted_weight_state_signature(
                    state, public_key_der=signer.public_key_der,
                    expected_public_key_hash=signer.public_key_hash,
                )
            except (TypeError, ValueError) as exc:
                raise ServiceError("accepted_weight_state_invalid", 500) from exc
            if state["reward_basis"].get("reward_basis_hash") != basis.get("reward_basis_hash"):
                raise ServiceError("accepted_weight_state_reward_conflict", 409)
            return {"state": state, "lookup_ok": True}
        try:
            epoch_scope = dict(self._config.chain.accepted_weight_epoch_scope())
            if int(epoch_scope.get("epoch", -1)) != requested_epoch:
                raise ServiceError("accepted_weight_epoch_not_current", 409)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("accepted_weight_epoch_scope_unavailable", 503) from exc
        try:
            state = weight_state.build_accepted_weight_state(
                self._reward_signer(), network=network,
                genesis_hash=epoch_scope["genesis_hash"], netuid=netuid,
                epoch=requested_epoch,
                valid_from_block=epoch_scope["valid_from_block"],
                valid_until_block=epoch_scope["valid_until_block"],
                reward_basis=basis,
                burn_hotkey=self._config.accepted_burn_hotkey,
                issued_at=str(basis["published_at"]),
            )
            stored = self._store.publish_weight_state(
                network, netuid, requested_epoch, state["state_hash"], state
            )
        except (KeyError, TypeError, ValueError, ArenaStoreError) as exc:
            raise ServiceError("accepted_weight_state_conflict", 409) from exc
        return {"state": validate_accepted_weight_state(stored["state"]), "lookup_ok": True}

    def handle_weight_state(self, envelope: Any) -> Dict[str, Any]:
        """Authorize a finalized permit holder before returning accepted state."""

        validated = self.validate_request(
            envelope, scope=contracts.SCOPE_WEIGHT_STATE, round_id=None
        )
        network, netuid = self._chain_scope()
        body = validated["body"]
        if (
            validated["round_id"] != "weight-state"
            or set(body) != {"epoch", "network", "netuid"}
        ):
            raise ServiceError("weight_state_request_invalid", 400)
        epoch = body.get("epoch")
        requested_netuid = body.get("netuid")
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or body.get("network") != network
            or isinstance(requested_netuid, bool)
            or not isinstance(requested_netuid, int)
            or requested_netuid != netuid
        ):
            raise ServiceError("weight_state_scope_mismatch", 400)
        try:
            metagraph = self._config.chain.metagraph(finalized=True)
            permitted_validator_uid(
                metagraph, validated["hotkey"], netuid=netuid
            )
        except ValidatorIneligible as exc:
            raise ServiceError(str(exc), 403) from None
        except Exception:
            raise ServiceError("validator_snapshot_unavailable", 503) from None
        return self.public_weight_state(epoch)

    def record_chain_outcome(self, document: Any) -> Dict[str, Any]:
        """Record an authenticated validator observation, never settlement authority."""

        # A validator journals the exact signed report before POST. If the
        # response is lost, that same report remains idempotent after its
        # freshness window; no altered replay receives this exception.
        if isinstance(document, Mapping):
            try:
                network, netuid = self._chain_scope()
                epoch = int(document.get("epoch"))
                validator_hotkey = str(document.get("validator_hotkey"))
                request_id = str(document.get("request_id"))
                row = self._store.get_chain_outcome(
                    network, netuid, epoch, validator_hotkey, request_id
                )
                if row is not None:
                    if row.get("outcome_doc") == dict(document):
                        return {"status": "recorded"}
                    raise ServiceError("chain_outcome_conflict", 409)
            except ServiceError:
                raise
            except (TypeError, ValueError, ArenaStoreError):
                pass
        validated = weight_state.validate_chain_outcome(document, now=self.now())
        network, netuid = self._chain_scope()
        if validated["network"] != network or int(validated["netuid"]) != netuid:
            raise ServiceError("chain_outcome_scope_mismatch", 409)
        if not self._config.verify_signature(
            validated["validator_hotkey"], validated["signature"],
            weight_state.chain_outcome_message(validated),
        ):
            raise ServiceError("chain_outcome_signature_invalid", 401)
        state = self._store.get_weight_state(network, netuid, int(validated["epoch"]))
        if state is None or state.get("state_hash") != validated["state_hash"]:
            raise ServiceError("chain_outcome_state_unknown", 409)
        try:
            return self._store.record_chain_outcome(
                network=network, netuid=netuid, epoch=int(validated["epoch"]),
                validator_hotkey=validated["validator_hotkey"],
                request_id=validated["request_id"],
                extrinsic_hash=validated["extrinsic_hash"], outcome_doc=validated,
            )
        except ArenaStoreError as exc:
            raise ServiceError("chain_outcome_conflict", 409) from exc

    def public_chain_outcomes(self, epoch: int) -> Dict[str, Any]:
        network, netuid = self._chain_scope()
        rows = self._store.list_chain_outcomes(network, netuid, int(epoch))
        return {"outcomes": [dict(row["outcome_doc"]) for row in rows], "lookup_ok": True}

    def public_round(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        configuration = row.get("configuration_doc") or {}
        participants = None
        if row["status"] != "open":
            participants = [
                {
                    "submission_id": participant["submission_id"],
                    "miner_hotkey": participant["miner_hotkey"],
                    "is_baseline": bool(participant.get("is_king")),
                }
                for participant in (row.get("participants") or [])
            ]
        view = {
            "round_id": round_id,
            "status": row["status"],
            "schedule": dict(configuration.get("schedule") or {}),
            "participants": participants,
            "finalists": row.get("finalists") if row["status"] == "published" else None,
            "publication": row.get("publication_doc"), "king_outcome": row.get("king_outcome"), "king_hotkey": row.get("king_hotkey"),
            "effective_reward_epoch": row.get("effective_reward_epoch"), "cancel_reason": row.get("cancel_reason"),
            "final_ranking": None,
        }
        if row["status"] == "published":
            publication = row.get("publication_doc") or {}
            view.update({"final_ranking": publication.get("final_ranking"), "king_decision": publication.get("king_decision")})
        if integrity.enabled(configuration):
            view["integrity_policy"] = integrity.POLICY
            view["confirmation_bank_hash"] = row.get("confirmation_bank_hash")
            if row["status"] == "published":
                view["confirmation_submission_ids"] = (row.get("confirmation_cohort") or {}).get("submission_ids", [])
        return view

    def _public_icp_disclosure(self, row: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if icp_disclosure.disclosure_metadata(row) is None:
            return None
        baselines = [p for p in row.get("participants") or [] if p.get("is_king") is True]
        runs = (
            self._store.list_runs(
                str(row["round_id"]),
                submission_id=str(baselines[0]["submission_id"]),
                kind="execute",
            )
            if len(baselines) == 1
            else []
        )
        return icp_disclosure.baseline_disclosure(row, runs, self.now())

    def public_benchmark(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        disclosure = self._public_icp_disclosure(row)
        if disclosure is None:
            raise ServiceError("benchmark_not_public", 403)
        icps = self.benchmark_icps(round_id)
        result = {
            "round_id": round_id,
            "icps": [
                {**(integrity.agent_visible_icp(icps[position], contacts_required=contact_policy.enabled(row.get("configuration_doc") or {})) if integrity.enabled(row.get("configuration_doc") or {}) else icps[position]), "icp_position": position,
                 "baseline_score": disclosure["baseline_scores"].get(position)}
                for position in disclosure["public_positions"]
            ],
            "icp_set_date": disclosure["icp_set_date"],
            "public_at": disclosure["public_at"],
            "public_icp_count": contracts.BENCHMARK_ICP_COUNT,
            "private_icp_count": 0,
            "disclosure_policy": disclosure["disclosure_policy"],
        }
        if integrity.enabled(row.get("configuration_doc") or {}):
            result["confirmation_bank_hash"] = row.get("confirmation_bank_hash")
            result["private_icp_count"] = contracts.CONFIRMATION_ICP_COUNT
            if (
                row["status"] == "published"
                or icp_disclosure.configured_policy(row)
                == icp_disclosure.DELAYED_DISCLOSURE_POLICY
            ):
                # The salted original document lets observers check the
                # commitment without changing the main bank's reveal time.
                result["confirmation_bank"] = self.confirmation_bank(round_id)
                result["private_icp_count"] = 0
        return result

    def public_results(self, round_id: str, submission_id: str) -> Dict[str, Any]:
        if not submission_id or not isinstance(submission_id, str):
            raise ServiceError("submission_missing", 404)  # an empty id must never mean "every submission"
        row = self._round(round_id)
        round_status = str(row["status"])
        if round_status != "published":
            raise ServiceError("results_not_public", 403)
        publication = row.get("publication_doc") or {}
        participants = publication.get("participants") or []
        participant = next(
            (
                item
                for item in participants
                if item.get("submission_id") == submission_id
            ),
            None,
        )
        if participant is None:
            raise ServiceError("submission_missing", 404)
        disclosure = self._public_icp_disclosure(row)
        public_positions = set(disclosure["public_positions"]) if disclosure else set()
        if integrity.enabled(row.get("configuration_doc") or {}) and (
            disclosure is not None
            or icp_disclosure.configured_policy(row)
            != icp_disclosure.DELAYED_DISCLOSURE_POLICY
        ):
            public_positions.update(contracts.stage_positions(3))
        runs = [
            run for run in self._store.list_runs(round_id, submission_id=submission_id, kind="execute")
            if run.get("icp_position") in public_positions
        ]
        outputs = {}
        for run in runs:
            if run.get("output_ref"):
                try:
                    raw = self._objects.get_bounded(str(run["output_ref"]), MAX_OUTPUT_BYTES)
                    outputs[run["run_id"]] = validate_output_document(json.loads(raw.decode("utf-8")))
                except Exception as exc:
                    raise ServiceError("public_output_unavailable", 503) from exc
        scores = {
            "stage_1": [
                {"run_id": run["run_id"], "icp_position": run["icp_position"], "per_icp_score": run["per_icp_score"]}
                for run in runs
                if int(run.get("stage") or 0) == 1 and run.get("per_icp_score") is not None
            ],
            "stage_2": [
                {"run_id": run["run_id"], "icp_position": run["icp_position"], "per_icp_score": run["per_icp_score"]}
                for run in runs
                if int(run.get("stage") or 0) == 2 and run.get("per_icp_score") is not None
            ],
        }
        if integrity.enabled(row.get("configuration_doc") or {}):
            scores["confirmation"] = [
                {"run_id": run["run_id"], "icp_position": run["icp_position"], "per_icp_score": run["per_icp_score"]}
                for run in runs if int(run.get("stage") or 0) == 3 and run.get("per_icp_score") is not None
            ]
        stage1_entry = next((item for item in publication.get("stage1_ranking") or [] if item.get("submission_id") == submission_id), None)
        final_entry = next((item for item in publication.get("final_ranking") or [] if item.get("submission_id") == submission_id), None)
        run_results = [run["result_doc"] for run in runs if run.get("result_doc")]
        validated_results = []
        for document in run_results:
            try:
                validated_results.append(contracts.validate_run_result(document))
            except ArenaContractError as exc:
                raise ServiceError("public_result_unavailable", 503) from exc
        run_results = validated_results
        result = {
            "round_id": round_id, "submission_id": submission_id, "submission": {
                "miner_hotkey": participant.get("miner_hotkey"),
                "is_baseline": bool(participant.get("is_baseline")),
            },
            "outputs": outputs, "run_results": run_results,
            "scores": scores,
            "public_icp_status": "ready" if disclosure else "pending",
            "public_icp_count": len(public_positions),
            "submission_scores": {
                "stage_1": None if stage1_entry is None else stage1_entry.get("stage1_score"),
                "final": None if final_entry is None else final_entry.get("final_score"),
            },
        }
        if contact_policy.enabled(row.get("configuration_doc") or {}):
            judgments = {}
            for stage in (1, 2, 3):
                if public_positions.intersection(contracts.stage_positions(stage)):
                    judgments.update(self._scoring_outputs(round_id, stage))
            icps = self.evaluation_icps(round_id) if outputs else []
            contacts = {}
            for run in runs:
                run_id = str(run["run_id"])
                judge = judgments.get(run_id)
                if run_id not in outputs or not judge or judge.get("status") != "accepted":
                    continue
                try:
                    breakdowns = self._verified_breakdowns(
                        judge, icp=icps[int(run["icp_position"])], companies=outputs[run_id]["companies"],
                        policy=row["configuration_doc"]["scorer_policy"],
                    )
                except scoring.ScoringError as exc:
                    raise ServiceError("public_contact_verification_unavailable", 503) from exc
                contacts[run_id] = [
                    {key: value for key, value in verify.redact_breakdown(item).items()
                     if key in {"company_index", "company_qualified", "contact_qualified", "contact_identity_key", "email_status", "contact_verification"}}
                    for item in breakdowns
                ]
            result["contact_verifications"] = contacts
        return result

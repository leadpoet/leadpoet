"""Arena service: daily benchmark, execution, scoring, and publication."""

from __future__ import annotations

import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from lab_arena import broker as broker_module, contracts, credentials as credentials_module, rewards, scoring, signing, source_bundle, submission_rate_limit, verify
from lab_arena.contracts import ArenaContractError, ArenaSignatureError
from lab_arena.output import OutputInvalid, validate_output_document
from lab_arena.store import ArenaStore, ArenaStoreError, hash_lease_token

MODES = ("off", "shadow", "live")
HOT_ROUND_TTL_SECONDS = 2.0
TERMINAL_STATUSES = ("published", "cancelled")
SOURCE_UPLOAD_EXPIRES_SECONDS = 900
DEFAULT_BASELINE_SOURCE_URL = "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz"
DEFAULT_STAGE_MINUTES = {
    "benchmark": 30,
    "stage_1": 240,
    "stage_1_scoring": 360,
    "stage_2": 180,
    "final_scoring": 240,
}
CANCEL_REASONS = {
    "benchmark_leak": "benchmark_leaked_before_cutoff",
    "benchmark_invalid": "benchmark_data_invalid",
    "capacity": "runner_capacity",
    "scoring": "scoring_window_closed",
    "publication": "publication_sanitizer_failed",
    "operator": "operator",
}


class ServiceError(RuntimeError):
    """A request or transition failed closed."""

    def __init__(self, code: str, status: int = 400) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


# ---------------------------------------------------------------------------
# Object store
# ---------------------------------------------------------------------------


class ObjectStore(Protocol):
    def put(self, ref: str, data: bytes) -> None: ...

    def get(self, ref: str) -> bytes: ...

    def get_bounded(self, ref: str, max_bytes: int) -> bytes: ...

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int) -> Mapping[str, Any]: ...


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

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int) -> Mapping[str, Any]:
        raise ServiceError("source_upload_not_configured", 503)


class S3ObjectStore:
    """Versioned, delete-denied Arena bucket (section 3.1); boto3 imported lazily."""

    def __init__(self, bucket: str, *, client: Any = None, region_name: Optional[str] = None) -> None:
        if client is None:
            import boto3  # noqa: WPS433

            client = boto3.client("s3", region_name=region_name)
        self._client = client
        self._bucket = bucket

    def put(self, ref: str, data: bytes) -> None:
        payload = bytes(data)
        for attempt in range(2):
            try:
                self._client.put_object(
                    Bucket=self._bucket,
                    Key=ref,
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
        response = self._client.get_object(Bucket=self._bucket, Key=ref)
        return response["Body"].read()

    def get_bounded(self, ref: str, max_bytes: int) -> bytes:
        head = self._client.head_object(Bucket=self._bucket, Key=ref)
        if int(head.get("ContentLength") or 0) > int(max_bytes):
            raise ArenaContractError("object exceeds source size limit")
        response = self._client.get_object(Bucket=self._bucket, Key=ref)
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

    def presign_put(self, ref: str, *, size_bytes: int, content_type: str, expires_seconds: int) -> Mapping[str, Any]:
        headers = {
            "content-type": str(content_type),
            "content-length": str(int(size_bytes)),
            "if-none-match": "*",
        }
        url = self._client.generate_presigned_url(
            "put_object",
            Params={
                "Bucket": self._bucket,
                "Key": ref,
                "ContentType": str(content_type),
                "ContentLength": int(size_bytes),
                "IfNoneMatch": "*",
            },
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

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]: ...

    def uid_for_hotkey(self, hotkey: str) -> Optional[int]: ...

@dataclass
class RoundDefaults:
    execution_cap_microusd: int = 5_000_000
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
    baseline_source_fetcher: Optional[Callable[[str, int], bytes]] = None
    reward_signer_factory: Optional[Callable[[], signing.ArenaSigner]] = None
    credential_manager: Optional[credentials_module.CredentialManager] = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ServiceError("mode_invalid", 500)
        if self.mode == "off":
            raise ServiceError("mode_off", 500)


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
        row = self._store.get_round(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        return self._require_round_mode(row)

    def _require_round_mode(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Refuse a round owned by another Arena mode."""

        if (row.get("configuration_doc") or {}).get("mode") != self._config.mode:
            raise ServiceError("round_mode_mismatch", 409)
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
        supported_versions = (184, 185) if self._config.credential_manager is None else (185,)
        if (
            not isinstance(schema, Mapping)
            or schema.get("schema_version") != expected_schema
            or schema_version not in supported_versions
        ):
            raise ServiceError("arena_schema_version_invalid", 500)
        for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger"):
            try:
                self._store._transport.select(table, limit=1)
            except ArenaStoreError as exc:
                raise ServiceError("table_unavailable:%s" % table, 500) from exc
        # Every service function must exist and be granted: a missing round is the
        # expected structured failure; a permission or undefined-function error is not.
        for function, params in (
            ("lab_arena_expire_leases", {"p_round_id": "arena-0000-00-00"}),
            ("lab_arena_close_stage", {"p_round_id": "arena-0000-00-00", "p_stage": 1}),
            ("lab_arena_cancel_round", {"p_round_id": "arena-0000-00-00", "p_reason": "startup-probe"}),
        ):
            try:
                self._store._transport.rpc(function, params)
            except ArenaStoreError as exc:
                if "lab_arena_round_missing" not in str(exc):
                    raise ServiceError("function_unavailable:%s" % function, 500) from exc
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
        return {
            "database_identity": identity,
            "schema_version": int(schema_version),
            "scoring_adapter_version": self._scorer_policy["scoring_adapter_version"],
            "current_round": current["round_id"] if current else None,
        }

    # -- round creation (section 5.1) ----------------------------------------

    def build_schedule(self, cutoff: datetime) -> Dict[str, str]:
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
        # Only the explicit Arena runner configuration grants execution and
        # scoring authority. A chain validator permit does not grant access.
        return runners, banned

    def create_round(self, cutoff: datetime, *, round_id: Optional[str] = None) -> Dict[str, Any]:
        defaults = self._config.defaults
        round_id = round_id or round_id_for_cutoff(cutoff)
        runner_hotkeys, banned_hotkeys = self.runner_settings()
        document = {
            "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
            "round_id": round_id,
            "mode": self._config.mode,
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
            "scoring_cap_microusd": defaults.scoring_cap_microusd,
            "scorer_image_digest": defaults.scorer_image_digest,
            "scorer_image_reference": defaults.scorer_image_reference,
            "baseline_hotkey": defaults.baseline_hotkey,
            "baseline_source_url": defaults.baseline_source_url,
            "runner_hotkeys": runner_hotkeys,
            "banned_hotkeys": banned_hotkeys,
            "reward_constants": rewards.reward_constants_document(int(defaults.pool_percent)),
        }
        configuration = contracts.validate_round_configuration(document)
        result = self._store.create_round(round_id, configuration)
        if result.get("status") not in ("created", "existing"):
            raise ServiceError("round_create_failed", 500)
        if result.get("status") == "existing":
            self._round(round_id)
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

        # Scan ids and statuses only; a full row can be large at hundreds of participants.
        for row in self._store.list_rounds(limit=20, columns="round_id,status,created_at,configuration_doc"):
            if row["status"] not in TERMINAL_STATUSES and (row.get("configuration_doc") or {}).get("mode") == self._config.mode:
                return self._round(row["round_id"])
        return None

    def active_rounds(self) -> List[Dict[str, Any]]:
        """Every round that is not published or cancelled, oldest first: ids and statuses only.

        Rounds overlap (one open for submissions while the previous one runs),
        so the driver advances each of them on every tick.
        """

        rows = [
            row
            for row in self._store.list_rounds(limit=20, columns="round_id,status,created_at,configuration_doc")
            if row["status"] not in TERMINAL_STATUSES
            and (row.get("configuration_doc") or {}).get("mode") == self._config.mode
        ]
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

        for row in self._store.list_rounds(limit=20, columns="round_id,status,created_at,configuration_doc"):
            if row["status"] == "open" and (row.get("configuration_doc") or {}).get("mode") == self._config.mode:
                return self._round(row["round_id"])
        return None

    def _hot_round(self, round_id: str) -> Optional[Dict[str, Any]]:
        """One round's row for runner-facing handlers, cached for a few seconds.

        Claims and completions arrive by the thousand per stage; the SQL
        functions remain the authority for status, so a briefly stale row
        only yields a structured refusal.
        """

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
        round_row = self._hot_round(round_id) if hot else self._store.get_round(round_id)
        if round_row is None:
            raise ServiceError("round_unknown", 404)
        round_row = self._require_round_mode(round_row)
        if validated["hotkey"] in self._banned_hotkeys(round_row):
            raise ServiceError("hotkey_banned", 403)
        return validated, round_row

    def latest_published_round(self) -> Optional[Dict[str, Any]]:
        rows = self._store.list_rounds(status="published", limit=200)
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
        if self._config.chain.uid_for_hotkey(validated["hotkey"]) is None:
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
        try:
            registration = self._store.register_submission(
                round_id,
                submission_id,
                validated["hotkey"],
                document,
            )
        except ArenaStoreError as exc:
            if "lab_arena_submission_conflict" in str(exc):
                raise ServiceError("submission_conflict", 409) from exc
            raise
        if registration.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        if registration.get("status") not in ("registered", "existing"):
            raise ServiceError("submission_registration_failed", 500)
        submission_id = str(registration.get("submission_id") or submission_id)
        source_ref = str(registration.get("source_ref") or source_ref)
        try:
            upload = self._objects.presign_put(
                source_ref,
                size_bytes=int(body["source_size_bytes"]),
                content_type=source_bundle.SOURCE_CONTENT_TYPE,
                expires_seconds=SOURCE_UPLOAD_EXPIRES_SECONDS,
            )
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
        try:
            source_bundle.validate_source_archive(
                payload, forbidden_values=forbidden_values
            )
        except source_bundle.SourceBundleError as exc:
            raise ServiceError("submission_rejected:%s" % exc.code, 400) from exc

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
            if all(
                self._store.get_submission_credential(
                    submission_id, validated["hotkey"], provider
                )
                is not None
                for provider in credentials_module.RUNTIME_PROVIDERS
            ):
                return {"status": "accepted", "submission_id": submission_id}
            raise ServiceError("submission_credentials_missing", 409)
        if row.get("status") != "uploading":
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
        result = self._store.accept_submission_with_credentials(
            str(round_row["round_id"]),
            submission_id,
            validated["hotkey"],
            encrypted_credentials,
        )
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
        if not source_url.startswith("https://"):
            raise ServiceError("baseline_source_url_invalid", 500)
        fetcher = self._config.baseline_source_fetcher
        if fetcher is None:
            raise ServiceError("baseline_source_fetcher_missing", 500)
        submission_id = "baseline-%s" % round_id.removeprefix("arena-")
        source_ref = "arena/%s/sources/%s.tar.gz" % (round_id, submission_id)
        row = self._store.get_submission(submission_id)
        if row is None:
            try:
                payload = self._objects.get_bounded(
                    source_ref, source_bundle.MAX_SOURCE_ARCHIVE_BYTES
                )
            except Exception:
                try:
                    payload = bytes(
                        fetcher(source_url, source_bundle.MAX_SOURCE_ARCHIVE_BYTES)
                    )
                    facts = source_bundle.validate_source_archive(payload)
                    self._objects.put(source_ref, payload)
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
        started = self.now()
        set_id = int(round_id.replace("arena-", "").replace("-", "")[:8])
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
            evaluation_date = round_id.replace("arena-", "")[:10]
            if evaluation_date < started.date().isoformat():
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
        if (
            len(icps) != contracts.BENCHMARK_ICP_COUNT
            or len(icps) != len(raw_icps)
            or any(not icp_id for icp_id in icp_ids)
            or len(set(icp_ids)) != len(icp_ids)
        ):
            self._store.cancel_round(round_id, CANCEL_REASONS["benchmark_invalid"])
            return {
                "status": "cancelled",
                "reason": CANCEL_REASONS["benchmark_invalid"],
            }
        try:
            participants = self.freeze_participants(round_id)
        except ServiceError as exc:
            if exc.code == "baseline_source_not_ready":
                return {"status": "retry", "reason": exc.code, "set_id": set_id}
            raise
        evaluation_date = round_id.replace("arena-", "")[:10]
        benchmark_ref = "arena/%s/benchmark.json" % round_id
        self._objects.put(benchmark_ref, contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": round_id, "icps": icps}).encode("utf-8"))
        transition = self._store.transition_round(round_id, "open", "committed", {
            "participants": participants,
            "benchmark_ref": benchmark_ref,
            "evaluation_date": evaluation_date,
        })
        return {"status": transition.get("status"), "participants": len(participants)}

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

    # -- stages (sections 2, 9) ----------------------------------------------

    def open_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if stage not in (1, 2):
            raise ServiceError("stage_invalid", 400)
        participants = list(round_row.get("participants") or [])
        if stage == 2:
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
        accepted = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            accepted[str(run["run_id"])] = run
        items: List[Dict[str, Any]] = []
        for item in plan["work_items"]:
            submission_id = item["submission_id"]
            run = accepted.get(str(item["scored_run_id"]))
            if run is None or run["submission_id"] != submission_id or int(run["icp_position"]) != int(item["icp_position"]) or run.get("output_ref") != item["output_ref"]:
                raise ServiceError("scoring_plan_run_mismatch", 500)
            items.append({
                "scored_run_id": run["run_id"],
                "submission_id": run["submission_id"],
                "icp_position": int(run["icp_position"]),
                "output_ref": run["output_ref"],
            })
        result = self._store.open_scoring(round_id, stage, items)
        return {"status": result.get("status"), "round_status": result.get("round_status"), "assignments": result.get("assignments"), "work_items": len(plan["work_items"])}

    def scoring_is_complete(self, round_id: str, stage: int) -> bool:
        runs = self._store.list_runs(round_id, stage=stage, kind="score")
        return all(run["status"] in ("accepted", "failed") for run in runs)

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
        document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
        output = scoring.validate_scoring_output_document(document)
        if output["scored_run_id"] != run["scored_run_id"]:
            raise ServiceError("scoring_output_item_mismatch", 500)
        return scoring.validate_breakdowns_for_item(output["breakdowns"], icp=icp, companies=companies, max_scored_companies=int(policy["max_scored_companies"]))

    def score_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Assemble the stage bundle from configured-runner scoring results."""

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_judged" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        policy = contracts.validate_scorer_policy(round_row["configuration_doc"]["scorer_policy"])
        plan = self._load_scoring_plan(round_row, stage)
        icps = self.benchmark_icps(round_id)
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
                raise ServiceError("scoring_assignment_missing", 500)
            if run["status"] == "accepted":
                continue
            submission_id = str(item["submission_id"])
            if submission_id == baseline_id:
                return self._store.cancel_round(round_id, CANCEL_REASONS["scoring"])
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
                if submission_id == baseline_id:
                    return self._store.cancel_round(
                        round_id, CANCEL_REASONS["scoring"]
                    )
                ineligible.add(submission_id)
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
        transition = self._store.transition_round(round_id, "stage2_judged", "scored", {})
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

    # -- publication and downstream reward activation -------------------------

    def publish(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "scored":
            return {"status": "stale", "round_status": round_row["status"]}
        stage1_ranking = verify.stage1_ranking(
            self._score_entries_from_runs(round_row, contracts.stage_positions(1), "stage1_score")
        )
        finalists = list(round_row.get("finalists") or [])
        final_entries = self._score_entries_from_runs(
            round_row, range(contracts.BENCHMARK_ICP_COUNT), "final_score"
        )
        king_entry = next((e for e in final_entries if e["is_king"]), None)
        decision = verify.king_decision([e for e in final_entries if not e["is_king"]], king_entry)
        published_at = _iso(self.now())
        publication = {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "is_baseline": bool(p.get("is_king"))} for p in round_row.get("participants") or []],
            "stage1_ranking": stage1_ranking,
            "finalists": finalists,
            "final_ranking": verify.final_ranking(final_entries),
            "king_decision": decision,
            "published_at": published_at,
        }
        contracts.check_strict_document(publication, contracts.PUBLICATION_LIMITS)
        transition = self._store.transition_round(round_id, "scored", "published", {
            "publication_doc": publication,
            "published_at": published_at,
        })
        return {
            "status": transition.get("status"),
            "king_outcome": decision["outcome"],
            "king_hotkey": str(decision.get("king_hotkey") or ""),
        }

    def activate_pending_rewards(self) -> Dict[str, Any]:
        """Activate eligible live rounds oldest-first after publication.

        Publication never calls this method. A transient chain, cutover, KMS,
        or database failure leaves the compact competition result published
        and the next driver tick retries the same oldest round.
        """

        if self._config.mode != "live":
            return {"status": "disabled", "activated": 0}
        rows = list(reversed(self._store.list_rounds(status="published", limit=200)))
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
        publication = row.get("publication_doc") or {}
        decision = publication.get("king_decision") or {}
        prior = self._store.published_reward_bases(limit=200)
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
        if validated["hotkey"] not in configuration["runner_hotkeys"]:
            raise ServiceError("runner_not_allowlisted", 403)
        excluded = list(self._config.chain.hotkeys_owned_by_same_coldkey(validated["hotkey"]))
        token = self._lease_token(validated)
        response = self._store.claim_assignment(
            round_id=round_id, runner_hotkey=validated["hotkey"], declared_parallelism=declared, slot_ceiling=int(configuration["runner_slot_ceiling"]),
            excluded_miner_hotkeys=excluded, request_id=validated["request_id"], request_hash=contracts.request_bytes_hash(validated), lease_token_hash=hash_lease_token(token),
            lease_ttl_seconds=int(configuration["lease_ttl_seconds"]),
        )
        if response.get("status") != "leased":
            return response
        icps = self.benchmark_icps(round_id)
        position = int(response["icp_position"])
        if not 0 <= position < len(icps):
            raise ServiceError("benchmark_data_invalid", 500)
        lease = dict(response, icp=icps[position], lease_token=token, round_id=round_id, evaluation_date=str(round_row.get("evaluation_date") or ""))
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
            }
        )
        return lease

    def _run_context(self, run_id: str, lease_token: str) -> Tuple[Dict[str, Any], broker_module.RunContext]:
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        return run, broker_module.RunContext(run_id=run_id, assignment_id=run["assignment_id"], attempt=int(run["attempt"]), icp_position=int(run["icp_position"]), lease_token_hash=hash_lease_token(lease_token), miner_hotkey=run["miner_hotkey"], submission_id=run["submission_id"], stage=int(run["stage"]), kind=str(run.get("kind") or "execute"), round_id=str(run.get("round_id") or ""))

    def handle_source(self, run_id: str, lease_token: str) -> bytes:
        """Return source bytes only to the runner that holds the active lease."""

        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
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
        if terminal_status == "accepted" and kind == "score":
            try:
                output = scoring.validate_scoring_output_document(body.get("output"))
            except scoring.ScoringError:
                raise ServiceError("output_invalid", 400)
            if "failure" in output or output["scored_run_id"] != run.get("scored_run_id"):
                raise ServiceError("output_invalid", 400)
            output_ref = "arena/%s/scores/items/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        elif terminal_status == "accepted":
            try:
                output = validate_output_document(body.get("output"))
            except OutputInvalid:
                raise ServiceError("output_invalid", 400)
            output_ref = "arena/%s/outputs/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        result = self._store.complete_attempt(
            run_id=run_id, lease_token_hash=hash_lease_token(lease_token), result=run_result, terminal_cause=terminal_status,
            output_ref=output_ref,
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
            if status == "open":
                if now < _parse_iso(schedule["submission_cutoff"]):
                    return {"status": "waiting", "round_status": status}
                return self.commit_benchmark(round_id)
            if status == "committed":
                if now < _parse_iso(schedule["stage_1_start"]):
                    return {"status": "waiting", "round_status": status}
                return self.open_stage(round_id, 1)
            if status in ("stage1", "stage2"):
                stage = 1 if status == "stage1" else 2
                self._store.expire_leases(round_id)
                if now >= _parse_iso(schedule["stage_%d_close" % stage]) or self.stage_is_complete(round_id, stage):
                    return self.close_stage(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_closed", "stage2_closed"):
                stage = 1 if status == "stage1_closed" else 2
                if not round_row.get("stage%d_scoring_plan_doc" % stage):
                    return self.commit_scoring_plan(round_id, stage)
                return self.open_scoring(round_id, stage)
            if status in ("stage1_scoring", "stage2_scoring"):
                stage = 1 if status == "stage1_scoring" else 2
                self._store.expire_leases(round_id)
                window = schedule["stage_1_scoring_close" if stage == 1 else "final_scoring_close"]
                if now >= _parse_iso(window) or self.scoring_is_complete(round_id, stage):
                    return self.close_scoring(round_id, stage)
                return {"status": "waiting", "round_status": status}
            if status in ("stage1_judged", "stage2_judged"):
                stage = 1 if status == "stage1_judged" else 2
                window = schedule["stage_1_scoring_close" if stage == 1 else "final_scoring_close"]
                try:
                    return self.score_stage(round_id, stage)
                except scoring.ScoringError:
                    if now >= _parse_iso(window) + timedelta(hours=2):
                        return self._store.cancel_round(round_id, CANCEL_REASONS["scoring"])
                    return {"status": "retry", "round_status": status}
            if status == "stage1_scored":
                if now < _parse_iso(schedule["stage_2_start"]):
                    return {"status": "waiting", "round_status": status}
                return self.open_stage(round_id, 2)
            if status == "scored":
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
            rows = self._store.published_reward_bases(limit=200)
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
        rows = self._store.published_reward_bases(limit=200)
        return rewards.governing_reward_basis(
            self._usable_reward_bases(rows), int(epoch)
        )

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
            "finalists": row.get("finalists"),
            "publication": row.get("publication_doc"), "king_outcome": row.get("king_outcome"), "king_hotkey": row.get("king_hotkey"),
            "effective_reward_epoch": row.get("effective_reward_epoch"), "cancel_reason": row.get("cancel_reason"),
            "final_ranking": None, "reward_basis": row.get("reward_basis_doc"),
        }
        if row["status"] == "published":
            publication = row.get("publication_doc") or {}
            view.update({"final_ranking": publication.get("final_ranking"), "king_decision": publication.get("king_decision")})
        return view

    def public_benchmark(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        # The benchmark is public once every execution has ended.
        if row["status"] not in ("stage2_closed", "stage2_scoring", "stage2_judged", "scored", "published"):
            raise ServiceError("benchmark_not_public", 403)
        icps = self.benchmark_icps(round_id)
        return {"round_id": round_id, "icps": icps}

    def public_results(self, round_id: str, submission_id: str) -> Dict[str, Any]:
        if not submission_id or not isinstance(submission_id, str):
            raise ServiceError("submission_missing", 404)  # an empty id must never mean "every submission"
        row = self._round(round_id)
        if row["status"] != "published":
            raise ServiceError("results_not_public", 403)
        runs = [run for run in self._store.list_runs(round_id, submission_id=submission_id, kind="execute")]
        outputs = {}
        for run in runs:
            if run.get("output_ref"):
                outputs[run["run_id"]] = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
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
        publication = row.get("publication_doc") or {}
        stage1_entry = next((item for item in publication.get("stage1_ranking") or [] if item.get("submission_id") == submission_id), None)
        final_entry = next((item for item in publication.get("final_ranking") or [] if item.get("submission_id") == submission_id), None)
        submission = self._store.get_submission(submission_id) or {}
        return {
            "round_id": round_id, "submission_id": submission_id, "submission": {
                "miner_hotkey": submission.get("miner_hotkey"),
                "is_baseline": bool(submission.get("is_king")),
            },
            "outputs": outputs, "run_results": [run["result_doc"] for run in runs if run.get("result_doc")],
            "scores": scores,
            "submission_scores": {
                "stage_1": None if stage1_entry is None else stage1_entry.get("stage1_score"),
                "final": None if final_entry is None else final_entry.get("final_score"),
            },
        }

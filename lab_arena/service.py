"""Arena service: daily driver, benchmark commitment, stage transitions,
scoring-plan commitment, publication, and the request handlers behind the
``/arena/v1`` routes (labarena.md sections 2, 5, 8, 9, 12, 13, 14, 16).

Every durable write goes through ``ArenaStore``; every authority document is
signed with the Arena signing key; ``advance_round`` is idempotent so a
restart at any state is safe. Nothing here runs inside an enclave.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

from lab_arena import benchmark, broker as broker_module, build, contracts, credentials, model_release as model_release_module, operations, rewards, scoring, signing, verify
from lab_arena.contracts import ArenaContractError, ArenaSignatureError
from lab_arena.output import OutputInvalid, output_document_hash, validate_output_document
from lab_arena.runner import cost_record, provider_call_record, worker_release_identity
from lab_arena.store import ArenaStore, ArenaStoreError, hash_lease_token

MODES = ("off", "shadow", "live")
HOT_ROUND_TTL_SECONDS = 2.0
DEFAULT_STAGE_MINUTES = {
    "benchmark": 30,
    "stage_1": 210,
    "stage_1_scoring": 60,
    "stage_2": 210,
    "final_scoring": 90,
}
CANCEL_REASONS = {
    "benchmark_leak": "benchmark_leaked_before_cutoff",
    "generation": "generation_could_not_fill_slots",
    "root_changed": "benchmark_root_changed_after_commitment",
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
            raise benchmark.BenchmarkReplayError("object store has no object at %s" % ref)
        return path.read_bytes()


class S3ObjectStore:
    """Versioned, delete-denied Arena bucket (section 3.1); boto3 imported lazily."""

    def __init__(self, bucket: str, *, client: Any = None, region_name: Optional[str] = None) -> None:
        if client is None:
            import boto3  # noqa: WPS433

            client = boto3.client("s3", region_name=region_name)
        self._client = client
        self._bucket = bucket

    def put(self, ref: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=ref, Body=bytes(data), ContentType="application/json")

    def get(self, ref: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=ref)
        return response["Body"].read()


class StoreJournal:
    """``GenerationJournal`` over the round row's hash-chained journal."""

    def __init__(self, store: ArenaStore, round_id: str) -> None:
        self._store = store
        self._round_id = round_id

    def entries(self) -> Tuple[Dict[str, Any], ...]:
        row = self._store.get_round(self._round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        return tuple(dict(entry) for entry in (row.get("journal") or []))

    def append(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        finalized = contracts.finalize_journal_entry(entry)
        result = self._store.append_journal_entry(self._round_id, finalized)
        if result.get("status") not in ("appended", "existing"):
            raise ServiceError("journal_append_failed", 500)
        return finalized


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ChainReads(Protocol):
    def finalized_head(self) -> Any: ...

    def metagraph(self, finalized: bool = True) -> Any: ...

    def current_settlement_epoch(self) -> int: ...

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]: ...

    def uid_for_hotkey(self, hotkey: str) -> Optional[int]: ...

    def validator_permit_hotkeys(self) -> List[str]: ...


@dataclass
class RoundDefaults:
    scoring_cap_microusd: int = 50_000_000
    openrouter_allowed_models: Tuple[str, ...] = ("openai/gpt-4o-mini",)
    floor_runner_hotkeys: Tuple[str, ...] = ()
    max_package_bytes: int = 25 * 1024 * 1024
    max_files: int = 2000
    max_file_bytes: int = 5 * 1024 * 1024
    publication_terms_hash: str = contracts.document_hash("leadpoet.lab_arena.publication_terms.v1")
    all_participants_run_stage_2: bool = False
    stage_minutes: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_STAGE_MINUTES))
    base_image_digest: str = "sha256:" + "0" * 64
    repository_commit: str = "0" * 40
    max_challengers: int = contracts.MAX_CHALLENGERS  # admitted challengers per round, at most MAX_CHALLENGERS
    scorer_image_digest: str = "sha256:" + "0" * 64  # the Arena-built judge image validators run for scoring assignments
    # Automatic daily rounds: the UTC hour of each day's submission cutoff, or
    # None to leave round creation to the operator (``lab_arena_admin.py create``).
    daily_cutoff_hour_utc: Optional[int] = None
    # A new round's cutoff lies at least this far ahead so miners can submit.
    min_submission_hours: int = 6


@dataclass
class ServiceConfig:
    mode: str
    store: ArenaStore
    object_store: ObjectStore
    signer: signing.ArenaSigner
    chain: ChainReads
    verify_signature: Callable[[str, str, str], bool]
    generation_provider: benchmark.GenerationProvider
    price_table_source: Callable[[Sequence[str]], Mapping[str, Any]]
    banned_hotkeys_source: Callable[[], Iterable[str]]
    broker_factory: Callable[["ArenaService", Mapping[str, Any]], broker_module.Broker]
    scorer_factory: Callable[[Mapping[str, Any]], scoring.Scorer]
    defaults: RoundDefaults = field(default_factory=RoundDefaults)
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)
    runtime_lock_hash: str = contracts.document_hash("runtime-lock")
    scoring_workers: int = 1
    network_name: str = "finney"
    # The daily king's frozen source is committed to the public sales-agent
    # repository after publication (section 12.4 extension). None disables it.
    # Replay each accepted scoring assignment from its recorded judge responses
    # (section 16 as revised); the replayed breakdowns are authoritative.
    replay_verification: bool = True
    replay_entry_command: Optional[Sequence[str]] = None
    replay_work_dir: Optional[str] = None
    model_release_client: Optional[Any] = None
    model_release_branch: str = model_release_module.DEFAULT_BRANCH

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
        self._clock = config.clock
        self._lock = threading.RLock()
        self._hot_round_lock = threading.Lock()
        self._hot_round: Optional[Tuple[float, Optional[Dict[str, Any]]]] = None
        self._scorer_policy = scoring.build_scorer_policy()
        self._worker_release = worker_release_identity(repository_commit=config.defaults.repository_commit, runtime_lock_hash=config.runtime_lock_hash)
        self._brokers: Dict[str, broker_module.Broker] = {}
        self._scoring_results: Dict[Tuple[str, int], Dict[str, List[Dict[str, Any]]]] = {}
        self._banned_cache: Dict[str, set] = {}

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

    @property
    def worker_release_hash(self) -> str:
        return str(self._worker_release["worker_release_hash"])

    def signing_key_document(self) -> Dict[str, Any]:
        return signing.signing_key_document(self._signer.public_key_der)

    def now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    def _sign(self, document: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
        return signing.sign_document(self._signer, document, hash_field=hash_field)

    def _round(self, round_id: str) -> Dict[str, Any]:
        row = self._store.get_round(round_id)
        if row is None:
            raise ServiceError("round_missing", 404)
        return row

    def startup_checks(self) -> Dict[str, Any]:
        """Fail closed unless every required identity is present and consistent (section 16).

        Checks the database role, every Arena table and function grant, the
        object store, the signing key, and, when a round exists, that its pinned
        scorer policy, operation table, and worker release equal this build's.
        """

        identity = self._store.require_service_role()
        for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_events", "lab_arena_accounts", "lab_arena_ledger"):
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
        probe_ref = "arena/_startup/%s.json" % contracts.document_hash(self._signer.public_key_hash)[7:23]
        probe_bytes = contracts.canonical_json({"probe": self._signer.public_key_hash}).encode("utf-8")
        try:
            self._objects.put(probe_ref, probe_bytes)
            if self._objects.get(probe_ref) != probe_bytes:
                raise ServiceError("object_store_mismatch", 500)
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("object_store_unavailable", 500) from exc
        probe_document = self._sign(contracts.hashed_document({"startup": True}, "probe_hash"), "probe_hash")
        signing.verify_document_signature(probe_document, hash_field="probe_hash", public_key_der=self._signer.public_key_der, expected_public_key_hash=self._signer.public_key_hash)
        current = self.current_round()
        if current is not None:
            configuration = current["configuration_doc"]
            for name, pinned, ours in (
                ("scorer_policy_hash", configuration.get("scorer_policy_hash"), self._scorer_policy["policy_hash"]),
                ("operation_table_hash", configuration.get("operation_table_hash"), operations.OPERATION_TABLE_HASH),
                ("worker_release_hash", (configuration.get("release") or {}).get("worker_release_hash"), self.worker_release_hash),
                ("signing_public_key_hash", configuration.get("signing_public_key_hash"), self._signer.public_key_hash),
            ):
                if pinned != ours:
                    raise ServiceError("release_identity_mismatch:%s" % name, 500)
        return {"database_identity": identity, "signing_public_key_hash": self._signer.public_key_hash, "worker_release_hash": self.worker_release_hash, "operation_table_hash": operations.OPERATION_TABLE_HASH, "scorer_policy_hash": self._scorer_policy["policy_hash"], "current_round": current["round_id"] if current else None}

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

    def runner_allowlist(self) -> Tuple[List[str], Dict[str, Any]]:
        banned = sorted(set(str(item) for item in self._config.banned_hotkeys_source()))
        floor = list(self._config.defaults.floor_runner_hotkeys)
        for hotkey in floor:
            if hotkey in banned:
                raise ServiceError("floor_runner_banned", 500)
        permitted = [hotkey for hotkey in self._config.chain.validator_permit_hotkeys() if hotkey not in banned]
        allowlist = sorted(set(permitted) | set(floor))
        snapshot = {"schema_version": "leadpoet.lab_arena.banned_snapshot.v1", "hotkeys": banned}
        return allowlist, contracts.hashed_document(snapshot, "snapshot_hash")

    def create_round(self, cutoff: datetime, *, round_id: Optional[str] = None) -> Dict[str, Any]:
        defaults = self._config.defaults
        round_id = round_id or round_id_for_cutoff(cutoff)
        allowlist, banned_snapshot = self.runner_allowlist()
        # The table prices the miners' allowed models and the judge's models: a
        # scoring run reserves against the same table as an execution.
        priced_models = sorted(set(defaults.openrouter_allowed_models) | {str(model) for model in self._scorer_policy["judge_models"].values()})
        price_table = broker_module.validate_price_table(self._config.price_table_source(priced_models))
        document = {
            "schema_version": contracts.ROUND_CONFIGURATION_SCHEMA_VERSION,
            "round_id": round_id,
            "mode": self._config.mode,
            "schedule": self.build_schedule(cutoff),
            "generator": benchmark.generator_configuration(),
            "tie_break_rule": "finalized_block_after_cutoff.v1",
            "stage_1_icp_count": contracts.STAGE_1_ICP_COUNT,
            "stage_2_icp_count": contracts.STAGE_2_ICP_COUNT,
            "finalist_count": contracts.FINALIST_COUNT,
            "max_challengers": int(defaults.max_challengers),
            "runner_slot_ceiling": contracts.RUNNER_SLOT_CEILING,
            "max_attempts_per_assignment": contracts.MAX_ATTEMPTS_PER_ASSIGNMENT,
            "lease_ttl_seconds": contracts.LEASE_TTL_SECONDS,
            "companies_per_icp": benchmark.apply_arena_contract({"company_stage": "Seed", "employee_count": ["11-50"]})["max_companies"],
            "release": {
                "repository_commit": defaults.repository_commit,
                "runsc_lock_hash": self._config.runtime_lock_hash,
                "worker_release_hash": self.worker_release_hash,
                "shim_hash": self._worker_release["shim_hash"],
                "base_image_digest": defaults.base_image_digest,
                "scorer_image_digest": defaults.scorer_image_digest,
            },
            "operation_table_hash": operations.OPERATION_TABLE_HASH,
            "openrouter_price_table_hash": price_table["price_table_hash"],
            "openrouter_allowed_models": list(defaults.openrouter_allowed_models),
            "miner_key_providers": list(contracts.MINER_KEY_PROVIDERS),
            "call_quotas": dict(contracts.CALL_QUOTAS_PER_ICP),
            "scoring_call_quotas": dict(contracts.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
            "call_quota_hash": operations.CALL_QUOTA_HASH,
            "icp_wall_clock_seconds": contracts.ICP_WALL_CLOCK_SECONDS,
            "scorer_policy_hash": self._scorer_policy["policy_hash"],
            "scoring_cap_microusd": defaults.scoring_cap_microusd,
            "runner_allowlist": allowlist,
            "floor_runner_hotkeys": list(defaults.floor_runner_hotkeys),
            "banned_hotkeys_snapshot_hash": banned_snapshot["snapshot_hash"],
            "signing_public_key_hash": self._signer.public_key_hash,
            "artifact_rules": {
                "max_package_bytes": defaults.max_package_bytes,
                "max_files": defaults.max_files,
                "max_file_bytes": defaults.max_file_bytes,
                "approved_dependency_set_hash": build.approved_dependency_set_hash(),
            },
            "publication_terms_hash": defaults.publication_terms_hash,
            "reward_constants": rewards.reward_constants_document(),
            "all_participants_run_stage_2": bool(defaults.all_participants_run_stage_2 or self._config.mode == "shadow"),
        }
        configuration = self._sign(contracts.finalize_round_configuration(document), "configuration_hash")
        self._objects.put("arena/%s/price_table.json" % round_id, contracts.canonical_json(price_table).encode("utf-8"))
        self._objects.put("arena/%s/banned_snapshot.json" % round_id, contracts.canonical_json(banned_snapshot).encode("utf-8"))
        result = self._store.create_round(round_id, configuration)
        if result.get("status") not in ("created", "existing"):
            raise ServiceError("round_create_failed", 500)
        return configuration

    def ensure_daily_round(self, now: Optional[datetime] = None) -> Dict[str, Any]:
        """Create the next daily round when no round is open or running.

        V1 runs one round at a time: runner-facing handlers resolve the newest
        round that is not published or cancelled, so the next round is created
        only after the previous one ends. Its cutoff is the next configured
        UTC hour at least ``min_submission_hours`` ahead; a date whose round
        already exists (published or cancelled that day) moves to the next
        day, because a round id is its cutoff date. Idempotent: a second call
        finds the round it created.
        """

        defaults = self._config.defaults
        current = self.current_round()
        if current is not None:
            return {"status": "existing", "round_id": current["round_id"], "round_status": current["status"]}
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
        # Scan ids and statuses only; a full row can be large at hundreds of participants.
        for row in self._store.list_rounds(limit=20, columns="round_id,status,created_at"):
            if row["status"] not in ("published", "cancelled"):
                return self._round(row["round_id"])
        return None

    def _hot_current_round(self) -> Optional[Dict[str, Any]]:
        """The current round for runner-facing handlers, cached for a few seconds.

        Claims and completions arrive by the thousand per stage; the SQL
        functions remain the authority for status, so a briefly stale row
        only yields a structured refusal.
        """

        now = time.monotonic()
        with self._hot_round_lock:
            cached = self._hot_round
            if cached is not None and now - cached[0] < HOT_ROUND_TTL_SECONDS:
                return cached[1]
        row = self.current_round()
        with self._hot_round_lock:
            self._hot_round = (now, row)
        return row

    def latest_published_round(self) -> Optional[Dict[str, Any]]:
        rows = self._store.published_reward_bases(limit=1)
        return rows[0] if rows else None

    # -- signed requests ------------------------------------------------------

    def validate_request(self, envelope: Any, *, scope: str, round_id: Optional[str]) -> Dict[str, Any]:
        try:
            validated = contracts.validate_signed_request(envelope, expected_scope=scope, now=int(self.now().timestamp()), verify_signature=self._config.verify_signature, expected_round_id=round_id)
        except ArenaSignatureError:
            raise ServiceError("signature_invalid", 401)
        except ArenaContractError as exc:
            raise ServiceError("request_invalid:%s" % str(exc)[:80], 400)
        if round_id is not None and validated["hotkey"] in self._banned_snapshot(round_id):
            raise ServiceError("hotkey_banned", 403)
        return validated

    def _banned_snapshot(self, round_id: str) -> set:
        """The banned-hotkey set frozen into the round configuration, verified by hash."""

        with self._lock:
            cached = self._banned_cache.get(round_id)
            if cached is not None:
                return cached
        round_row = self._round(round_id)
        document = json.loads(self._objects.get("arena/%s/banned_snapshot.json" % round_id).decode("utf-8"))
        if contracts.verify_hashed_document(document, "snapshot_hash") != round_row["configuration_doc"]["banned_hotkeys_snapshot_hash"]:
            raise ServiceError("banned_snapshot_mismatch", 500)
        banned = set(str(item) for item in document.get("hotkeys") or [])
        with self._lock:
            self._banned_cache[round_id] = banned
        return banned

    # -- submissions (sections 6, 7, 14.2) -------------------------------------

    def handle_submission(self, envelope: Any, archive: bytes) -> Dict[str, Any]:
        round_row = self.current_round()
        if round_row is None or round_row["status"] != "open":
            raise ServiceError("submission_window_closed", 409)
        round_id = round_row["round_id"]
        validated = self.validate_request(envelope, scope=contracts.SCOPE_SUBMISSION, round_id=round_id)
        body = validated["body"]
        package_hash = contracts.hash_bytes(archive)
        if body.get("package_hash") != package_hash:
            raise ServiceError("package_hash_mismatch", 400)
        rules = round_row["configuration_doc"]["artifact_rules"]
        if len(archive) > int(rules["max_package_bytes"]):
            raise ServiceError("package.too_large", 413)
        # One submission id per (round, miner, package): the same bytes from a
        # second miner register separately and are rejected at acceptance under
        # the duplicate-artifact rule, never as a server error.
        submission_id = "sub-%s" % contracts.document_hash({"round_id": round_id, "miner_hotkey": validated["hotkey"], "package_hash": package_hash})[7:39]
        try:
            registration = self._store.register_submission(round_id, submission_id, validated["hotkey"], {"package_hash": package_hash, "package_ref": "arena/%s/packages/%s.tar.gz" % (round_id, submission_id), "consent": dict(body.get("consent") or {})})
        except ArenaStoreError as exc:
            if "lab_arena_submission_conflict" in str(exc):
                raise ServiceError("submission_conflict", 409) from exc
            raise
        if registration.get("status") == "window_closed":
            raise ServiceError("submission_window_closed", 409)
        self._objects.put("arena/%s/packages/%s.tar.gz" % (round_id, submission_id), archive)
        try:
            inspection = build.inspect_package(archive, build.PackageRules(max_package_bytes=int(rules["max_package_bytes"]), max_files=int(rules["max_files"]), max_file_bytes=int(rules["max_file_bytes"])))
            build.scan_source_archive_raise(inspection.files)
        except (build.PackageRejected, build.SecretMaterialFound) as exc:
            rule = getattr(exc, "rule_id", "package.rejected")
            self._store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": rule})
            return {"status": "rejected", "submission_id": submission_id, "rule": rule}
        return {"status": "uploaded", "submission_id": submission_id, "source_tree_hash": inspection.source_tree_hash}

    def accept_built_submission(self, round_id: str, submission_id: str, *, image_digest: str, source_tree_hash: str, scan_result: Mapping[str, Any], screening_result: Mapping[str, Any]) -> Dict[str, Any]:
        """Record the builder's immutable image and the screening pass (section 6.3)."""

        if not screening_result.get("accepted"):
            return self._store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": str(screening_result.get("rule") or "screening.rejected"), "screening_result": dict(screening_result)})
        result = self._store.update_submission(round_id, submission_id, "uploaded", "accepted", {"image_digest": image_digest, "source_tree_hash": source_tree_hash, "scan_result": dict(scan_result), "screening_result": dict(screening_result)})
        if result.get("status") == "duplicate_artifact":
            # One artifact competes once (section 6.2): the later submission is rejected under a published rule.
            self._store.update_submission(round_id, submission_id, "uploaded", "rejected", {"rejection_rule": "package.duplicate_artifact"})
            return {"status": "duplicate_artifact", "submission_status": "rejected", "rejection_rule": "package.duplicate_artifact"}
        return result

    def handle_credential(self, envelope: Any, *, register: Callable[[Mapping[str, Any]], Dict[str, Any]], provider: Optional[str] = None) -> Dict[str, Any]:
        """Register or replace one of a miner's encrypted provider keys (section 7.3).

        Miners bring their own Scrapingdog, Deepline, and OpenRouter keys.
        ``register`` decrypts once inside the broker identity, runs the
        provider's read-only preflight, and returns the non-secret record; the
        account stores the whole ciphertext envelope (never the plaintext) so
        the broker can decrypt it per call.
        """

        round_row = self.current_round()
        validated = self.validate_request(envelope, scope=contracts.SCOPE_CREDENTIAL, round_id=round_row["round_id"] if round_row else None)
        key_envelope = validated["body"].get("envelope")
        if not isinstance(key_envelope, Mapping):
            raise ServiceError("envelope_missing", 400)
        envelope_provider = key_envelope.get("provider")
        if envelope_provider not in contracts.MINER_KEY_PROVIDERS or (provider is not None and provider != envelope_provider):
            raise ServiceError("provider_invalid", 400)
        record = register(key_envelope)
        if record.get("provider") != envelope_provider:
            raise ServiceError("provider_invalid", 400)
        stored = contracts.canonical_json(dict(key_envelope))
        return self._store.upsert_account_credential(validated["hotkey"], envelope_provider, stored, str(record["key_hash"]), record)

    # -- participant freeze and benchmark (sections 7.1, 8) --------------------

    def freeze_participants(self, round_id: str) -> List[Dict[str, Any]]:
        round_row = self._round(round_id)
        participants: List[Dict[str, Any]] = []
        accepted = [row for row in self._store.list_submissions(round_id, status="accepted") if not row.get("is_king")]
        king = self._entering_king(round_id)
        cap = int(round_row["configuration_doc"].get("max_challengers") or contracts.MAX_CHALLENGERS)
        # Eligibility is checked before the cap so a miner without every
        # provider key preflighted never consumes an admission slot; freeze
        # order (acceptance order) decides who enters when the cap binds, and
        # every exclusion is recorded under a published rule.
        frozen_count = 0
        for row in accepted:
            account = self._store.get_account(row["miner_hotkey"]) or {}
            if account.get("preflight_status") != "ok":
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": "credential.preflight_not_ok"})
                continue
            if frozen_count >= cap:
                self._store.update_submission(round_id, row["submission_id"], "accepted", "rejected", {"rejection_rule": "capacity.round_full"})
                continue
            frozen_count += 1
            result = self._store.update_submission(round_id, row["submission_id"], "accepted", "frozen", {})
            if result.get("status") in ("ok", "stale"):
                participants.append({"submission_id": row["submission_id"], "miner_hotkey": row["miner_hotkey"], "image_digest": row["image_digest"], "source_tree_hash": row["source_tree_hash"], "is_king": False, "preflight_failed": False})
        if king is not None:
            participants.append(king)
        for row in self._store.list_submissions(round_id, status="frozen"):
            if not any(p["submission_id"] == row["submission_id"] for p in participants):
                participants.append({"submission_id": row["submission_id"], "miner_hotkey": row["miner_hotkey"], "image_digest": row["image_digest"], "source_tree_hash": row["source_tree_hash"], "is_king": bool(row.get("is_king")), "preflight_failed": False})
        return participants

    def _entering_king(self, round_id: str) -> Optional[Dict[str, Any]]:
        """The king published by the most recent published round enters automatically."""

        latest = self.latest_published_round()
        if latest is None or not latest.get("king_hotkey") or latest.get("king_outcome") == "no_king":
            return None
        king_hotkey = latest["king_hotkey"]
        previous_round = self._store.get_round(latest["round_id"])
        decision = (previous_round or {}).get("publication_doc", {}).get("king_decision", {})
        king_submission = self._store.get_submission(str(decision.get("king_submission_id") or ""))
        if king_submission is None:
            return None
        submission_id = "king-%s" % round_id
        existing = self._store.get_submission(submission_id)
        if existing is None:
            self._store.register_submission(round_id, submission_id, king_hotkey, {"package_hash": king_submission.get("package_hash") or contracts.document_hash(submission_id), "package_ref": king_submission.get("package_ref"), "consent": king_submission.get("consent") or {}, "is_king": True})
            self._store.update_submission(round_id, submission_id, "uploaded", "accepted", {"image_digest": king_submission["image_digest"], "source_tree_hash": king_submission["source_tree_hash"], "is_king": True})
            self._store.update_submission(round_id, submission_id, "accepted", "frozen", {"is_king": True})
        account = self._store.get_account(king_hotkey) or {}
        preflight_failed = account.get("preflight_status") != "ok"
        return {"submission_id": submission_id, "miner_hotkey": king_hotkey, "image_digest": king_submission["image_digest"], "source_tree_hash": king_submission["source_tree_hash"], "is_king": True, "preflight_failed": bool(preflight_failed)}

    def commit_benchmark(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "open":
            return {"status": "existing", "round_status": round_row["status"]}
        configuration = round_row["configuration_doc"]
        participants = self.freeze_participants(round_id)
        started = self.now()
        try:
            result = benchmark.generate_benchmark(
                round_id=round_id,
                set_id=int(round_id.replace("arena-", "").replace("-", "")[:8]),
                provider=self._config.generation_provider,
                journal=StoreJournal(self._store, round_id),
                object_store=self._objects,
                clock=self._clock,
                max_attempts=int(configuration["generator"]["max_generation_attempts"]),
            )
        except benchmark.BenchmarkGenerationFailed:
            self._store.cancel_round(round_id, CANCEL_REASONS["generation"])
            return {"status": "cancelled", "reason": CANCEL_REASONS["generation"]}
        head = self._config.chain.finalized_head()
        tie_break = {"number": int(head.number), "hash": str(head.hash)}
        commitment = self._sign(
            benchmark.commit_benchmark(result, round_configuration_hash=configuration["configuration_hash"], participants=participants, tie_break_block=tie_break, evaluation_date=round_id.replace("arena-", "")[:10], started_at=result.generation_started_at, finished_at=result.generation_finished_at),
            "commitment_hash",
        )
        benchmark_ref = "arena/%s/benchmark.json" % round_id
        self._objects.put(benchmark_ref, contracts.canonical_json({"schema_version": "leadpoet.lab_arena.benchmark.v1", "round_id": round_id, "icps": list(result.icps), "icp_hashes": list(result.icp_hashes)}).encode("utf-8"))
        transition = self._store.transition_round(round_id, "open", "committed", {
            "commitment_hash": commitment["commitment_hash"],
            "commitment_doc": commitment,
            "participant_set_hash": commitment["participant_set_hash"],
            "participants": participants,
            "benchmark_ref": benchmark_ref,
            "evaluation_date": commitment["evaluation_date"],
        })
        return {"status": transition.get("status"), "commitment_hash": commitment["commitment_hash"], "participants": len(participants)}

    def benchmark_icps(self, round_id: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        round_row = self._round(round_id)
        ref = round_row.get("benchmark_ref")
        if not ref:
            raise ServiceError("benchmark_not_committed", 409)
        document = json.loads(self._objects.get(ref).decode("utf-8"))
        icps = list(document["icps"])
        hashes = [contracts.document_hash(icp) for icp in icps]
        if hashes != list(document["icp_hashes"]):
            raise ServiceError("benchmark_root_changed", 500)
        commitment = round_row.get("commitment_doc") or {}
        if contracts.benchmark_roots(hashes)["benchmark_root"] != commitment.get("benchmark_root"):
            self._store.cancel_round(round_id, CANCEL_REASONS["root_changed"])
            raise ServiceError("benchmark_root_changed", 500)
        return icps, hashes

    # -- stages (sections 2, 9) ----------------------------------------------

    def open_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        round_row = self._round(round_id)
        configuration = round_row["configuration_doc"]
        participants = list(round_row.get("participants") or [])
        if stage == 2 and not configuration.get("all_participants_run_stage_2"):
            finalists = set(round_row.get("finalists") or [])
            participants = [p for p in participants if p["submission_id"] in finalists or p.get("is_king")]
        _icps, hashes = self.benchmark_icps(round_id)
        positions = list(range(0, contracts.STAGE_1_ICP_COUNT)) if stage == 1 else list(range(contracts.STAGE_1_ICP_COUNT, contracts.BENCHMARK_ICP_COUNT))
        rows = [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "preflight_failed": bool(p.get("preflight_failed"))} for p in participants]
        return self._store.open_stage(round_id, stage, rows, positions, [hashes[p] for p in positions])

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
        _icps, hashes = self.benchmark_icps(round_id)
        plan = scoring.build_scoring_plan(
            round_id=round_id, stage=stage, configuration_hash=round_row["configuration_hash"], commitment_hash=round_row["commitment_hash"],
            scorer_policy_hash=self._scorer_policy["policy_hash"], runs=self._store.list_runs(round_id, stage=stage, kind="execute"), icp_hashes_by_position=dict(enumerate(hashes)),
        )
        plan_ref = "arena/%s/scoring/stage%d_plan.json" % (round_id, stage)
        signed = self._put_signed(plan_ref, plan, "plan_hash")
        # The row holds a header only: at hundreds of participants the work
        # items run to megabytes, and the row is read on every driver tick.
        header = {key: value for key, value in signed.items() if key != "work_items"}
        header.update({"work_items_ref": plan_ref, "work_item_count": len(plan["work_items"])})
        status = "stage%d_closed" % stage
        result = self._store.transition_round(round_id, status, status, {"stage%d_scoring_plan_hash" % stage: signed["plan_hash"], "stage%d_scoring_plan_doc" % stage: header})
        return {"status": result.get("status"), "plan_hash": signed["plan_hash"], "work_items": len(plan["work_items"])}

    def _put_signed(self, ref: str, document: Mapping[str, Any], hash_field: str) -> Dict[str, Any]:
        """Sign and store a document once; a retry reuses the stored signature.

        Signatures are not deterministic, so a retry after a crash between
        the object write and the row transition would otherwise collide with
        the write-once object store. The stored copy is accepted only when
        its hash equals this document's hash and its signature verifies.
        """

        signed = self._sign(document, hash_field)
        try:
            self._objects.put(ref, contracts.canonical_json(signed).encode("utf-8"))
            return signed
        except contracts.ArenaContractError:
            stored = json.loads(self._objects.get(ref).decode("utf-8"))
            if stored.get(hash_field) != signed[hash_field] or contracts.verify_hashed_document(stored, hash_field) != signed[hash_field]:
                raise ServiceError("signed_object_conflict", 500)
            signing.verify_document_signature(stored, hash_field=hash_field, public_key_der=self._signer.public_key_der, expected_public_key_hash=self._signer.public_key_hash)
            return stored

    def _load_scoring_plan(self, round_row: Mapping[str, Any], stage: int) -> Dict[str, Any]:
        header = round_row.get("stage%d_scoring_plan_doc" % stage)
        if not header:
            raise ServiceError("scoring_plan_missing", 409)
        ref = header.get("work_items_ref") if isinstance(header, Mapping) else None
        if not isinstance(ref, str):
            raise ServiceError("scoring_plan_ref_missing", 500)
        plan = json.loads(self._objects.get(ref).decode("utf-8"))
        if contracts.verify_hashed_document(plan, "plan_hash") != round_row.get("stage%d_scoring_plan_hash" % stage):
            raise ServiceError("scoring_plan_hash_mismatch", 500)
        return plan

    def _outputs_by_hash(self, round_id: str, stage: int) -> Dict[str, List[Dict[str, Any]]]:
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        for run in self._store.list_runs(round_id, stage=stage, status="accepted", kind="execute"):
            if run["output_hash"] in outputs:
                continue
            document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
            if output_document_hash(document) != run["output_hash"]:
                raise ServiceError("output_hash_mismatch", 500)
            outputs[run["output_hash"]] = list(document["companies"])
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
            accepted[(run["submission_id"], int(run["icp_position"]))] = run
        items: List[Dict[str, Any]] = []
        for item in plan["work_items"]:
            submission_id = sorted(item["submission_ids"])[0]  # identical outputs are judged once, on the first miner's keys
            run = accepted.get((submission_id, int(item["icp_position"])))
            if run is None or run.get("output_hash") != item["output_hash"]:
                raise ServiceError("scoring_plan_run_mismatch", 500)
            items.append({"work_item_id": item["work_item_id"], "output_hash": item["output_hash"], "scored_run_id": run["run_id"]})
        result = self._store.open_scoring(round_id, stage, items)
        return {"status": result.get("status"), "round_status": result.get("round_status"), "assignments": result.get("assignments"), "work_items": len(plan["work_items"])}

    def scoring_is_complete(self, round_id: str, stage: int) -> bool:
        runs = self._store.list_runs(round_id, stage=stage, kind="score")
        return all(run["status"] in ("accepted", "failed") for run in runs)

    def close_scoring(self, round_id: str, stage: int) -> Dict[str, Any]:
        return self._store.close_scoring(round_id, stage)

    def _scoring_outputs(self, round_id: str, stage: int) -> Dict[str, Dict[str, Any]]:
        """The score run that counts for each work item: an accepted attempt, else the latest."""

        chosen: Dict[str, Dict[str, Any]] = {}
        for run in self._store.list_runs(round_id, stage=stage, kind="score"):
            current = chosen.get(run["work_item_id"])
            if current is None or (run["status"] == "accepted" and current["status"] != "accepted") or (run["status"] == current["status"] and int(run["attempt"]) > int(current["attempt"])):
                chosen[run["work_item_id"]] = run
        return chosen

    def _verified_breakdowns(self, run: Mapping[str, Any], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        document = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
        if contracts.document_hash(document) != run["output_hash"]:
            raise ServiceError("scoring_output_hash_mismatch", 500)
        output = scoring.validate_scoring_output_document(document)
        if output["work_item_id"] != run["work_item_id"]:
            raise ServiceError("scoring_output_item_mismatch", 500)
        return scoring.validate_breakdowns_for_item(output["breakdowns"], icp=icp, companies=companies, max_scored_companies=int(self._scorer_policy["max_scored_companies"]))

    def _replayed_breakdowns(self, run: Mapping[str, Any], *, icp: Mapping[str, Any], companies: Sequence[Mapping[str, Any]], reported: Sequence[Mapping[str, Any]], report: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Re-derive the breakdowns from the run's recorded judge responses.

        Reproduced with the same numbers: accepted. Reproduced with different
        numbers: the replayed numbers stand and the validator is flagged. Not
        reproducible at all: ``None``, and the scoring is rejected.
        """

        from lab_arena import replay as replay_module

        input_document = scoring.build_scoring_input(work_item_id=str(run["work_item_id"]), icp=icp, companies=companies, policy=self._scorer_policy, evaluation_date=str(self._round(run["round_id"]).get("evaluation_date") or ""))
        work_dir = Path(self._config.replay_work_dir or tempfile.gettempdir())
        entry = {"run_id": run["run_id"], "runner": run.get("runner_hotkey"), "work_item_id": run["work_item_id"]}
        try:
            output, summary = replay_module.replay_work_item(input_document=input_document, ledger_entries=self._store.list_ledger(run_id=run["run_id"]), work_dir=work_dir, entry_command=self._config.replay_entry_command)
        except (replay_module.ReplayError, scoring.ScoringError) as exc:
            report.append(dict(entry, outcome="rejected", reason="replay_failed", detail=str(exc)[:120]))
            return None
        if "failure" in output:
            report.append(dict(entry, outcome="rejected", reason="replay_" + str(output["failure"]), served=summary["served"], misses=len(summary["misses"])))
            return None
        try:
            replayed = scoring.validate_breakdowns_for_item(output["breakdowns"], icp=icp, companies=companies, max_scored_companies=int(self._scorer_policy["max_scored_companies"]))
        except scoring.ScoringError as exc:
            report.append(dict(entry, outcome="rejected", reason="replay_invalid", detail=str(exc)[:120]))
            return None
        # The score-bearing form decides a match: the redacted breakdown keeps
        # exactly what the score and FP derivation read, and drops payload
        # fields (quotes, traces, receipts) a judge may not reproduce verbatim.
        matched = contracts.document_hash([verify.redact_breakdown(b) for b in replayed]) == contracts.document_hash([verify.redact_breakdown(b) for b in reported])
        report.append(dict(entry, outcome="match" if matched else "mismatch", served=summary["served"], misses=len(summary["misses"])))
        return replayed

    def score_stage(self, round_id: str, stage: int) -> Dict[str, Any]:
        """Assemble the stage bundle from validator-scored work items.

        Every accepted scoring is replayed first. A scoring the replay cannot
        reproduce is rejected and re-scored: the round returns to the scoring
        window for the rejected items' second attempts.
        """

        round_row = self._round(round_id)
        if round_row["status"] != "stage%d_judged" % stage:
            return {"status": "stale", "round_status": round_row["status"]}
        plan = self._load_scoring_plan(round_row, stage)
        icps, _hashes = self.benchmark_icps(round_id)
        outputs = self._outputs_by_hash(round_id, stage)
        chosen = self._scoring_outputs(round_id, stage)
        scoring_started = _iso(self.now())
        breakdowns_by_item: Dict[str, List[Dict[str, Any]]] = {}
        refused: List[str] = []
        replay_report: List[Dict[str, Any]] = []
        rejected: List[str] = []
        judge_executions = 0
        for item in plan["work_items"]:
            work_item_id = item["work_item_id"]
            run = chosen.get(work_item_id)
            if run is None:
                raise ServiceError("scoring_assignment_missing", 500)
            icp = icps[int(item["icp_position"])]
            companies = outputs[item["output_hash"]]
            if run["status"] != "accepted":
                if run.get("terminal_cause") != "judge_key_refused":
                    raise scoring.ScoringError("work item %s was not judged" % work_item_id)
                refused.append(work_item_id)  # the scored miner's own key refused the judge: a declared zero row
                continue
            breakdowns = self._verified_breakdowns(run, icp=icp, companies=companies)
            judge_executions += 1
            if self._config.replay_verification:
                replayed = self._replayed_breakdowns(run, icp=icp, companies=companies, reported=breakdowns, report=replay_report)
                if replayed is None:
                    rejected.append(run["run_id"])
                    continue
                breakdowns = replayed
            breakdowns_by_item[work_item_id] = breakdowns
        timing_ref = "arena/%s/timing/stage%d_scoring.json" % (round_id, stage)
        if rejected:
            result = self._store.reject_scorings(round_id, stage, rejected)
            self._objects.put(
                "arena/%s/timing/stage%d_rejections_%d.json" % (round_id, stage, int(result.get("stage_generation") or 0)),
                contracts.canonical_json({"stage": stage, "at": _iso(self.now()), "rejected": rejected, "replays": replay_report, "result": result}).encode("utf-8"),
            )
            return {"status": "rescoring" if result.get("status") == "reopened" else result.get("status"), "round_status": result.get("round_status"), "rejected": len(rejected), "exhausted": result.get("exhausted")}
        stage_1_rows = None
        stage_1_bundle_hash = None
        if stage == 2:
            stage_1_bundle = json.loads(self._objects.get(round_row["stage1_scores_ref"]).decode("utf-8"))
            stage_1_rows = stage_1_bundle["rows"]
            stage_1_bundle_hash = stage_1_bundle["bundle_hash"]
        ref = "arena/%s/scores/stage%d.json" % (round_id, stage)
        bundle = self._put_signed(
            ref,
            scoring.build_score_bundle(plan=plan, policy=self._scorer_policy, icps_by_position=dict(enumerate(icps)), outputs_by_hash=outputs, breakdowns_by_item=breakdowns_by_item, refused_items={item: "judge_key_refused" for item in refused}, stage_1_rows=stage_1_rows, stage_1_bundle_hash=stage_1_bundle_hash),
            "bundle_hash",
        )
        mismatches = sum(1 for entry in replay_report if entry.get("outcome") == "mismatch")
        self._objects.put(
            timing_ref,
            contracts.canonical_json({"stage": stage, "started_at": scoring_started, "finished_at": _iso(self.now()), "judge_executions": judge_executions, "work_items": len(plan["work_items"]), "key_refused_items": refused, "replays": replay_report, "replay_mismatches": mismatches}).encode("utf-8"),
        )
        runs = self._store.list_runs(round_id, stage=stage, kind="execute")
        self._store.record_run_scores(round_id, stage, scoring.run_scores_for_store(bundle, runs, score_ref=ref))
        if stage == 1:
            finalists = self._select_finalists(round_row, bundle)
            transition = self._store.transition_round(round_id, "stage1_judged", "stage1_scored", {"finalists": finalists, "stage1_scores_ref": ref, "stage1_score_bundle_hash": bundle["bundle_hash"]})
        else:
            transition = self._store.transition_round(round_id, "stage2_judged", "scored", {"final_scores_ref": ref, "final_score_bundle_hash": bundle["bundle_hash"]})
        return {"status": transition.get("status"), "bundle_hash": bundle["bundle_hash"], "judge_executions": judge_executions, "replay_mismatches": mismatches}

    def _ranking_entries(self, round_row: Mapping[str, Any], bundle: Mapping[str, Any], score_key: str) -> List[Dict[str, Any]]:
        entries = []
        for participant in round_row.get("participants") or []:
            submission_id = participant["submission_id"]
            if submission_id not in bundle["submission_scores"]:
                continue
            entries.append({"submission_id": submission_id, "artifact_hash": participant["source_tree_hash"], score_key: bundle["submission_scores"][submission_id], "is_king": bool(participant.get("is_king"))})
        return entries

    def _salt(self, round_row: Mapping[str, Any]) -> str:
        return str((round_row.get("commitment_doc") or {}).get("tie_break_block_hash") or "")

    def _select_finalists(self, round_row: Mapping[str, Any], bundle: Mapping[str, Any]) -> List[str]:
        ranking = verify.stage1_ranking(self._ranking_entries(round_row, bundle, "stage1_score"), self._salt(round_row))
        return verify.select_finalists(ranking)

    # -- publication (sections 12.3, 12.4, 13) ---------------------------------

    def publish(self, round_id: str) -> Dict[str, Any]:
        round_row = self._round(round_id)
        if round_row["status"] != "scored":
            return {"status": "stale", "round_status": round_row["status"]}
        configuration = round_row["configuration_doc"]
        stage_1_bundle = json.loads(self._objects.get(round_row["stage1_scores_ref"]).decode("utf-8"))
        final_bundle = json.loads(self._objects.get(round_row["final_scores_ref"]).decode("utf-8"))
        salt = self._salt(round_row)
        stage1_ranking = verify.stage1_ranking(self._ranking_entries(round_row, stage_1_bundle, "stage1_score"), salt)
        finalists = list(round_row.get("finalists") or [])
        final_entries = [e for e in self._ranking_entries(round_row, final_bundle, "final_score") if e["submission_id"] in finalists or e["is_king"]]
        latest = self.latest_published_round()
        previous_king_hotkey = str(latest.get("king_hotkey") or "") if latest else ""
        participants_by_id = {p["submission_id"]: p for p in round_row.get("participants") or []}
        rows_by_submission: Dict[str, Dict[int, Dict[str, Any]]] = {}
        for row in list(stage_1_bundle["rows"]) + list(final_bundle["rows"]):
            rows_by_submission.setdefault(row["submission_id"], {})[int(row["icp_position"])] = row
        positions = list(range(contracts.BENCHMARK_ICP_COUNT))

        def _final_entry(entry: Mapping[str, Any]) -> Dict[str, Any]:
            valid = verify.result_is_valid(rows_by_submission.get(entry["submission_id"], {}), positions)
            return {
                "submission_id": entry["submission_id"],
                "hotkey": participants_by_id[entry["submission_id"]]["miner_hotkey"],
                "artifact_hash": entry["artifact_hash"],
                "final_score": entry["final_score"] if valid else None,
                "is_king": bool(entry["is_king"]),
            }

        final_entries = [_final_entry(e) for e in final_entries]
        king_entry = next((e for e in final_entries if e["is_king"]), None)
        decision = verify.king_decision([e for e in final_entries if not e["is_king"]], king_entry, salt)
        runs = self._store.list_runs(round_id, kind="execute")
        runner_fractions = self._runner_fractions(runs)
        ledger_totals = self._cost_totals(round_id)
        finalized_epoch = int(self._config.chain.current_settlement_epoch())
        effective_epoch = finalized_epoch + 1
        previous_start = (int(latest["king_start_epoch"]) if latest and latest.get("king_start_epoch") is not None and latest.get("king_outcome") in ("crowned", "defended") else None)
        published_at = _iso(self.now())
        king_hotkey = str(decision.get("king_hotkey") or "")
        public_bundle = {
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION,
            "round_id": round_id,
            "round_configuration": configuration,
            "benchmark_commitment": round_row["commitment_doc"],
            "benchmark_ref": round_row["benchmark_ref"],
            "participants": [{"submission_id": p["submission_id"], "miner_hotkey": p["miner_hotkey"], "image_digest": p["image_digest"], "source_tree_hash": p["source_tree_hash"], "is_king": bool(p.get("is_king")), "package_ref": (self._store.get_submission(p["submission_id"]) or {}).get("package_ref")} for p in round_row.get("participants") or []],
            "scorer_policy": self._scorer_policy,
            "stage_plans": {"stage_1": self._load_scoring_plan(round_row, 1), "stage_2": self._load_scoring_plan(round_row, 2)},
            "score_bundles": {"stage_1": stage_1_bundle, "final": final_bundle},
            "stage1_ranking": stage1_ranking,
            "finalists": finalists,
            "final_ranking": verify.final_ranking(final_entries, salt),
            "king_decision": decision,
            "receipts": [run["receipt_doc"] for run in runs if run.get("receipt_doc")],
            # Lists, not maps keyed by run: a round holds up to 650 runs and the
            # publication schema bounds object keys far below that.
            "outputs": [
                {"run_id": run["run_id"], "submission_id": run["submission_id"], "stage": int(run["stage"]), "icp_position": int(run["icp_position"]), "output_hash": run["output_hash"], "output_ref": run["output_ref"]}
                for run in sorted(runs, key=lambda r: (r["submission_id"], int(r["stage"]), int(r["icp_position"]), int(r["attempt"])))
                if run.get("output_ref")
            ],
            "runner_fractions": runner_fractions,
            "cost_totals": ledger_totals,
            "signing_key": self.signing_key_document(),
        }
        try:
            build.scan_document_raise(public_bundle)
            # The raise-mode secret scan runs again on every published source
            # archive (section 6.3 step 3) before anything becomes public.
            for participant in round_row.get("participants") or []:
                submission = self._store.get_submission(participant["submission_id"]) or {}
                package_ref = submission.get("package_ref")
                if not package_ref:
                    continue
                inspection = build.inspect_package(self._objects.get(package_ref))
                build.scan_source_archive_raise(inspection.files)
        except (build.SecretMaterialFound, build.PackageRejected):
            raise ServiceError("publication_sanitizer_failed", 500)
        contracts.check_strict_document(public_bundle, contracts.PUBLICATION_LIMITS)
        result_bundle_hash = contracts.document_hash(public_bundle)
        basis = self._sign(rewards.reward_basis_document(
            round_id=round_id, configuration_hash=round_row["configuration_hash"], commitment_hash=round_row["commitment_hash"], result_bundle_hash=result_bundle_hash,
            published_at=published_at, finalized_epoch=finalized_epoch, king_hotkey=king_hotkey, king_outcome=decision["outcome"], previous_king_start_epoch=previous_start,
        ), "reward_basis_hash")
        publication = self._sign(contracts.hashed_document({
            "schema_version": contracts.PUBLICATION_SCHEMA_VERSION, "round_id": round_id, "configuration_hash": round_row["configuration_hash"],
            "commitment_hash": round_row["commitment_hash"], "result_bundle_hash": result_bundle_hash, "result_bundle_ref": "arena/%s/public/bundle.json" % round_id,
            "king_decision": decision, "reward_basis_hash": basis["reward_basis_hash"], "published_at": published_at,
        }, "publication_hash"), "publication_hash")
        self._objects.put(publication["result_bundle_ref"], contracts.canonical_json(public_bundle).encode("utf-8"))
        self._objects.put("arena/%s/public/reward_basis.json" % round_id, contracts.canonical_json(basis).encode("utf-8"))
        transition = self._store.transition_round(round_id, "scored", "published", {
            "result_bundle_hash": result_bundle_hash,
            "publication_doc": publication,
            "king_outcome": decision["outcome"],
            "king_hotkey": basis["king_hotkey"],
            "king_start_epoch": basis["king_start_epoch"],
            "effective_reward_epoch": effective_epoch,
            "reward_basis_hash": basis["reward_basis_hash"],
            "reward_basis_doc": basis,
        })
        return {"status": transition.get("status"), "result_bundle_hash": result_bundle_hash, "king_outcome": decision["outcome"], "king_hotkey": basis["king_hotkey"], "effective_reward_epoch": effective_epoch}

    def _runner_fractions(self, runs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        executed: Dict[str, int] = {}
        abandoned: Dict[str, int] = {}
        for run in runs:
            hotkey = run.get("runner_hotkey")
            if not hotkey:
                continue
            if run["status"] == "accepted" or run.get("terminal_cause") in contracts.MODEL_CAUSED_TERMINAL_CAUSES:
                executed[hotkey] = executed.get(hotkey, 0) + 1
            elif run.get("terminal_cause") in ("lease_expired", "worker_lost", "stage_closed", "receipt_rejected"):
                abandoned[hotkey] = abandoned.get(hotkey, 0) + 1
        total = sum(executed.values()) or 1
        hotkeys = sorted(set(executed) | set(abandoned))
        return [
            {"runner_hotkey": hotkey, "executed_fraction": round(executed.get(hotkey, 0) / total, 6), "executed": executed.get(hotkey, 0), "abandoned": abandoned.get(hotkey, 0)}
            for hotkey in hotkeys
        ]

    def _cost_totals(self, round_id: str) -> Dict[str, Any]:
        totals: Dict[str, Dict[str, int]] = {}
        heads: Dict[str, Dict[str, Any]] = {}
        for entry in self._store.list_ledger():
            if entry.get("round_id") != round_id or not entry.get("call_identity"):
                continue
            heads[entry["call_identity"]] = entry
        calls: Dict[str, Dict[str, int]] = {}
        for entry in heads.values():
            if entry["entry_kind"] in ("settlement", "uncertain"):
                bucket = totals.setdefault(entry["submission_id"], {})
                bucket[entry["provider"]] = bucket.get(entry["provider"], 0) + int(entry["amount_microusd"])
                count = calls.setdefault(entry["submission_id"], {})
                count[entry["provider"]] = count.get(entry["provider"], 0) + 1
        # Every provider is billed to the miner's own key: the cost is the
        # provider-reported amount where one exists (OpenRouter) and the call
        # counts are what the quotas bound.
        return {submission: {"providers": sorted(costs), "total_microusd": sum(costs.values()), "calls": dict(sorted(calls.get(submission, {}).items()))} for submission, costs in sorted(totals.items())}

    # -- runner handlers (section 14.3) ----------------------------------------

    def _lease_token(self, validated: Mapping[str, Any]) -> str:
        return contracts.document_hash({"lease": validated["request_id"], "signature": validated["signature"]})[7:]

    def handle_claim(self, envelope: Any) -> Dict[str, Any]:
        round_row = self._hot_current_round()
        if round_row is None:
            raise ServiceError("no_open_round", 409)
        round_id = round_row["round_id"]
        validated = self.validate_request(envelope, scope=contracts.SCOPE_CLAIM, round_id=round_id)
        body = validated["body"]
        if body.get("worker_release_hash") != self.worker_release_hash:
            raise ServiceError("worker_release_mismatch", 403)
        declared = body.get("declared_parallelism")
        if isinstance(declared, bool) or not isinstance(declared, int) or declared < 1:
            raise ServiceError("declared_parallelism_invalid", 400)
        configuration = round_row["configuration_doc"]
        if validated["hotkey"] not in configuration["runner_allowlist"]:
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
        icps, hashes = self.benchmark_icps(round_id)
        position = int(response["icp_position"])
        if hashes[position] != response["icp_hash"]:
            raise ServiceError("benchmark_root_changed", 500)
        lease = dict(response, icp=icps[position], lease_token=token)
        if response.get("kind") == "score":
            # A scoring assignment: the validator runs the pinned judge image on
            # the scored output with the signed scorer policy.
            scored = self._store.get_run(str(response.get("scored_run_id") or ""))
            if scored is None or not scored.get("output_ref"):
                raise ServiceError("scored_run_missing", 500)
            output = json.loads(self._objects.get(scored["output_ref"]).decode("utf-8"))
            if output_document_hash(output) != scored["output_hash"]:
                raise ServiceError("output_hash_mismatch", 500)
            lease.update({"image_digest": configuration["release"]["scorer_image_digest"], "scored_output": output, "scorer_policy": self._scorer_policy})
        return lease

    def _run_context(self, run_id: str, lease_token: str) -> Tuple[Dict[str, Any], broker_module.RunContext]:
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        return run, broker_module.RunContext(run_id=run_id, assignment_id=run["assignment_id"], attempt=int(run["attempt"]), icp_position=int(run["icp_position"]), lease_token_hash=hash_lease_token(lease_token), miner_hotkey=run["miner_hotkey"], submission_id=run["submission_id"], stage=int(run["stage"]), kind=str(run.get("kind") or "execute"))

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
        contracts.check_strict_document(frame, contracts.REQUEST_LIMITS)
        run, context = self._run_context(run_id, lease_token)
        broker = self._broker_for(run["round_id"])
        result = broker.execute(context, operation_id=str(frame["operation_id"]), parameters=frame["parameters"], action_sequence=frame["action_sequence"], timeout_ms=int(frame["timeout_ms"]))
        return result.to_document()

    def handle_events(self, run_id: str, lease_token: str, events: Any) -> Dict[str, Any]:
        if not isinstance(events, list) or not events or len(events) > 256:
            raise ServiceError("events_invalid", 400)
        for event in events:
            body = {k: v for k, v in dict(event).items() if k not in ("prev_hash", "event_hash")}
            contracts.validate_private_event(body)
        return self._store.append_events(run_id=run_id, lease_token_hash=hash_lease_token(lease_token), events=events)

    def handle_complete(self, envelope: Any) -> Dict[str, Any]:
        round_row = self._hot_current_round()
        if round_row is None:
            raise ServiceError("no_open_round", 409)
        round_id = round_row["round_id"]
        validated = self.validate_request(envelope, scope=contracts.SCOPE_COMPLETE, round_id=round_id)
        body = validated["body"]
        raw_receipt = dict(body.get("receipt") or {})
        run_id = str(raw_receipt.pop("run_id", ""))
        try:
            receipt = contracts.validate_icp_receipt(raw_receipt, verify_signature=self._config.verify_signature)
        except ArenaSignatureError:
            raise ServiceError("receipt_signature_invalid", 401)
        except ArenaContractError as exc:
            raise ServiceError("receipt_invalid:%s" % str(exc)[:80], 400)
        run = self._store.get_run(run_id)
        if run is None:
            raise ServiceError("run_missing", 404)
        if receipt["runner_hotkey"] != validated["hotkey"] or run.get("runner_hotkey") != validated["hotkey"]:
            raise ServiceError("receipt_runner_mismatch", 403)
        submission = self._store.get_submission(run["submission_id"]) or {}
        kind = str(run.get("kind") or "execute")
        if str(receipt.get("kind") or "execute") != kind:
            raise ServiceError("receipt_identity_mismatch:kind", 400)
        expected_image = round_row["configuration_doc"]["release"]["scorer_image_digest"] if kind == "score" else submission.get("image_digest")
        for field_name, expected in (
            ("round_id", round_id), ("submission_id", run["submission_id"]), ("assignment_id", run["assignment_id"]), ("attempt", int(run["attempt"])),
            ("stage", int(run["stage"])), ("icp_position", int(run["icp_position"])), ("lease_generation", int(run["lease_generation"])),
            ("miner_hotkey", run["miner_hotkey"]), ("worker_release_hash", self.worker_release_hash), ("image_digest", expected_image), ("icp_hash", run["icp_hash"]),
        ):
            if receipt[field_name] != expected:
                raise ServiceError("receipt_identity_mismatch:%s" % field_name, 400)
        if kind == "score" and receipt["terminal_status"] not in contracts.SCORE_TERMINAL_CAUSES:
            raise ServiceError("receipt_cause_kind_mismatch", 400)
        if kind == "execute" and receipt["terminal_status"] in ("judge_error", "judge_timeout", "judge_key_refused"):
            raise ServiceError("receipt_cause_kind_mismatch", 400)
        lease_token = self._lease_token_for_run(validated, run)
        events = self._store.list_events(run_id)
        event_docs = [dict(row["event_doc"]) for row in events]
        try:
            if contracts.private_event_root(event_docs) != receipt["private_event_root"]:
                raise ServiceError("receipt_event_root_mismatch", 400)
        except ArenaContractError:
            raise ServiceError("event_chain_invalid", 400)
        calls = self._ledger_calls(run_id)
        if receipt["provider_call_root"] != contracts.ordered_root([contracts.document_hash(provider_call_record(c)) for c in calls]):
            raise ServiceError("receipt_call_root_mismatch", 400)
        if receipt["cost_root"] != contracts.ordered_root([contracts.document_hash(cost_record(c)) for c in calls]):
            raise ServiceError("receipt_cost_root_mismatch", 400)
        output_ref = ""
        output_hash = ""
        if receipt["terminal_status"] == "accepted" and kind == "score":
            try:
                output = scoring.validate_scoring_output_document(body.get("output"))
            except scoring.ScoringError:
                raise ServiceError("output_invalid", 400)
            if "failure" in output or output["work_item_id"] != run.get("work_item_id"):
                raise ServiceError("output_invalid", 400)
            output_hash = contracts.document_hash(output)
            if output_hash != receipt["output_hash"]:
                raise ServiceError("receipt_output_hash_mismatch", 400)
            output_ref = "arena/%s/scores/items/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        elif receipt["terminal_status"] == "accepted":
            try:
                output = validate_output_document(body.get("output"))
            except OutputInvalid:
                raise ServiceError("output_invalid", 400)
            output_hash = output_document_hash(output)
            if output_hash != receipt["output_hash"]:
                raise ServiceError("receipt_output_hash_mismatch", 400)
            output_ref = "arena/%s/outputs/%s.json" % (round_id, run_id)
            self._objects.put(output_ref, contracts.canonical_json(output).encode("utf-8"))
        result = self._store.complete_attempt(
            run_id=run_id, lease_token_hash=hash_lease_token(lease_token), receipt=receipt, receipt_hash=receipt["receipt_hash"], terminal_cause=receipt["terminal_status"],
            output_hash=output_hash, output_ref=output_ref, provider_call_root=receipt["provider_call_root"], private_event_root=receipt["private_event_root"], cost_root=receipt["cost_root"],
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
            return self._advance_round_locked(round_id)
        finally:
            self._invalidate_hot_round()

    def _invalidate_hot_round(self) -> None:
        with self._hot_round_lock:
            self._hot_round = None

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
                if not round_row.get("stage%d_scoring_plan_hash" % stage):
                    return self.commit_scoring_plan(round_id, stage)
                return self.open_scoring(round_id, stage)
            if status in ("stage1_scoring", "stage2_scoring"):
                # Validators score the committed plan; the window is the stage's scoring close.
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
            if status == "published" and self._config.model_release_client is not None and self._object_or_none(self._model_release_ref(round_id)) is None:
                return self.release_king_model(round_id)
            return {"status": "terminal", "round_status": status}

    # -- model release to the public sales-agent repository -------------------

    @staticmethod
    def _model_release_ref(round_id: str) -> str:
        return "arena/%s/public/model_release.json" % round_id

    def _object_or_none(self, ref: str) -> Optional[bytes]:
        try:
            return self._objects.get(ref)
        except Exception:
            return None

    def release_king_model(self, round_id: str) -> Dict[str, Any]:
        """Commit the published king's frozen source to the sales-agent repository.

        Idempotent per round through a signed receipt object: a round with a
        receipt is never released twice, a retry after a crash finds the
        repository already holding the model, and rounds without a crowned or
        defended king record a skipped receipt so the driver stops retrying.
        """

        row = self._round(round_id)
        if row["status"] != "published":
            raise ServiceError("round_not_published", 409)
        receipt_ref = self._model_release_ref(round_id)
        existing = self._object_or_none(receipt_ref)
        if existing is not None:
            return {"status": "ok", "model_release": json.loads(existing.decode("utf-8")), "round_status": "published"}
        client = self._config.model_release_client
        if client is None:
            raise ServiceError("model_release_unconfigured", 409)
        outcome = str(row.get("king_outcome") or "")
        publication = row["publication_doc"]
        if outcome not in ("crowned", "defended"):
            receipt = self._put_signed(receipt_ref, contracts.hashed_document({"schema_version": model_release_module.RELEASE_RECEIPT_SCHEMA_VERSION, "round_id": round_id, "changed": False, "skipped": outcome or "no_king", "commit_sha": None, "parent_sha": None, "tree_sha": None, "branch": self._config.model_release_branch, "repository": client.repository, "release_hash": None, "manifest": None}, "receipt_hash"), "receipt_hash")
            return {"status": "ok", "model_release": receipt, "round_status": "published"}
        decision = publication.get("king_decision") or {}
        king_submission = self._store.get_submission(str(decision.get("king_submission_id") or ""))
        if king_submission is None or king_submission.get("miner_hotkey") != row.get("king_hotkey"):
            raise ServiceError("model_release_king_missing", 500)
        try:
            inspection = build.inspect_package(self._objects.get(king_submission["package_ref"]))
        except (build.PackageRejected, build.SecretMaterialFound, contracts.ArenaContractError) as exc:
            raise ServiceError("model_release_package_invalid", 500) from exc
        if inspection.source_tree_hash != king_submission.get("source_tree_hash") or inspection.package_hash != king_submission.get("package_hash"):
            raise ServiceError("model_release_package_mismatch", 500)
        build.scan_source_archive_raise(inspection.files)  # never commit secret material, even if the scan rules changed
        configuration = row["configuration_doc"]
        base_image_digest = configuration.get("base_image_digest") or (configuration.get("release") or {}).get("base_image_digest") or (configuration.get("artifact_rules") or {}).get("base_image_digest")
        if not isinstance(base_image_digest, str):
            raise ServiceError("model_release_base_image_missing", 500)
        manifest = self._sign(model_release_module.release_manifest(
            repository=client.repository, branch=self._config.model_release_branch, round_id=round_id, king_hotkey=str(row["king_hotkey"]), king_outcome=outcome,
            submission_id=king_submission["submission_id"], package_hash=king_submission["package_hash"], source_tree_hash=king_submission["source_tree_hash"],
            image_digest=str(king_submission.get("image_digest")), entry_point=inspection.entry_point, dependency_lock=list(inspection.dependency_lock),
            base_image_digest=base_image_digest, file_count=len(inspection.files), configuration_hash=row["configuration_hash"], result_bundle_hash=row["result_bundle_hash"],
            publication_hash=publication["publication_hash"], reward_basis_hash=publication["reward_basis_hash"], signing_public_key_hash=self._signer.public_key_hash,
            released_at=_iso(self.now()),
        ), "release_hash")
        try:
            result = model_release_module.release_king_model(client, branch=self._config.model_release_branch, manifest=manifest, files=inspection.files)
        except model_release_module.ModelReleaseError as exc:
            raise ServiceError("model_release_failed", 502) from exc
        receipt_document = dict(result.to_document())
        receipt_document.update({"round_id": round_id, "manifest": manifest, "skipped": None})
        receipt = self._put_signed(receipt_ref, contracts.hashed_document(receipt_document, "receipt_hash"), "receipt_hash")
        return {"status": "ok", "model_release": receipt, "round_status": "published"}

    def cancel(self, round_id: str, reason: str) -> Dict[str, Any]:
        if reason not in CANCEL_REASONS.values():
            raise ServiceError("cancel_reason_invalid", 400)
        self._invalidate_hot_round()
        return self._store.cancel_round(round_id, reason)

    # -- public reads (section 14.1) -------------------------------------------

    def public_current(self) -> Dict[str, Any]:
        current = self.current_round()
        latest = self.latest_published_round()
        epoch = None
        try:
            epoch = int(self._config.chain.current_settlement_epoch())
        except Exception:
            epoch = None
        eligibility = None
        week = None
        if epoch is not None:
            governing = self.public_reward_basis(epoch)
            if governing is None:
                eligibility = False
            else:
                eligibility = rewards.epoch_eligible(governing, epoch)
                if eligibility:
                    week = rewards.reward_week_index(epoch, int(governing["king_start_epoch"]))
        return {
            "mode": self._config.mode,
            "round": {"round_id": current["round_id"], "status": current["status"]} if current else None,
            "king": {"hotkey": latest.get("king_hotkey"), "outcome": latest.get("king_outcome"), "round_id": latest.get("round_id"), "king_start_epoch": latest.get("king_start_epoch")} if latest else None,
            "reward_week_index": week,
            "epoch_eligible": eligibility,
            "current_epoch": epoch,
        }

    def public_reward_basis(self, epoch: int) -> Optional[Dict[str, Any]]:
        rows = [row["reward_basis_doc"] for row in self._store.published_reward_bases() if row.get("reward_basis_doc")]
        return rewards.governing_reward_basis(rows, int(epoch))

    def public_round(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        view = {
            "round_id": round_id, "status": row["status"], "configuration": row["configuration_doc"], "commitment": row.get("commitment_doc"),
            "participants": row.get("participants") if row["status"] not in ("open",) else None, "finalists": row.get("finalists"),
            "publication": row.get("publication_doc"), "king_outcome": row.get("king_outcome"), "king_hotkey": row.get("king_hotkey"),
            "effective_reward_epoch": row.get("effective_reward_epoch"), "cancel_reason": row.get("cancel_reason"),
            "stage1_ranking": None, "final_ranking": None, "runner_fractions": None, "reward_basis": row.get("reward_basis_doc"),
            "model_release": None,
        }
        release = self._object_or_none(self._model_release_ref(round_id))
        if release is not None:
            view["model_release"] = json.loads(release.decode("utf-8"))
        if row["status"] == "published":
            bundle = json.loads(self._objects.get(row["publication_doc"]["result_bundle_ref"]).decode("utf-8"))
            view.update({"stage1_ranking": bundle.get("stage1_ranking"), "final_ranking": bundle.get("final_ranking"), "runner_fractions": bundle.get("runner_fractions"), "king_decision": bundle.get("king_decision")})
        return view

    def public_benchmark(self, round_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        if row["status"] not in ("stage2_closed", "scored", "published"):
            raise ServiceError("benchmark_not_public", 403)
        icps, hashes = self.benchmark_icps(round_id)
        return {"round_id": round_id, "icps": icps, "icp_hashes": hashes, "commitment": row.get("commitment_doc")}

    def shadow_report(self, round_id: str) -> Dict[str, Any]:
        """The section 20 shadow gate report for a published shadow round."""

        from lab_arena import shadow

        row = self._round(round_id)
        if row["status"] != "published":
            raise ServiceError("round_not_published", 409)
        bundle = json.loads(self._objects.get(row["publication_doc"]["result_bundle_ref"]).decode("utf-8"))
        timings = []
        for stage in (1, 2):
            try:
                timings.append(json.loads(self._objects.get("arena/%s/timing/stage%d_scoring.json" % (round_id, stage)).decode("utf-8")))
            except benchmark.BenchmarkReplayError:
                continue
        return shadow.shadow_report(round_row=row, public_bundle=bundle, scoring_timings=timings)

    def public_results(self, round_id: str, submission_id: str) -> Dict[str, Any]:
        row = self._round(round_id)
        if row["status"] != "published":
            raise ServiceError("results_not_public", 403)
        runs = [run for run in self._store.list_runs(round_id, submission_id=submission_id, kind="execute")]
        outputs = {}
        for run in runs:
            if run.get("output_ref"):
                outputs[run["run_id"]] = json.loads(self._objects.get(run["output_ref"]).decode("utf-8"))
        bundles = {"stage_1": json.loads(self._objects.get(row["stage1_scores_ref"]).decode("utf-8")), "final": json.loads(self._objects.get(row["final_scores_ref"]).decode("utf-8"))}
        return {
            "round_id": round_id, "submission_id": submission_id, "submission": {k: v for k, v in (self._store.get_submission(submission_id) or {}).items() if k in ("miner_hotkey", "image_digest", "source_tree_hash", "package_ref", "is_king")},
            "outputs": outputs, "receipts": [run["receipt_doc"] for run in runs if run.get("receipt_doc")],
            "scores": {stage: [r for r in bundle["rows"] if r["submission_id"] == submission_id] for stage, bundle in bundles.items()},
            "submission_scores": {stage: bundle["submission_scores"].get(submission_id) for stage, bundle in bundles.items()},
        }

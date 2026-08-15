"""Canonical contracts for candidate-bound production-parity validation.

This module contains no infrastructure or product business logic.  It defines
the deterministic documents that bind a staging run to one Git candidate, one
read-only production snapshot, and one complete stage ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse


CONTRACT_SCHEMA_VERSION = "leadpoet.production_parity_contract.v1"
SNAPSHOT_SCHEMA_VERSION = "leadpoet.production_parity_snapshot.v2"
LEDGER_SCHEMA_VERSION = "leadpoet.production_parity_ledger.v1"
HISTORICAL_ORACLE_SCHEMA_VERSION = (
    "leadpoet.production_parity_historical_oracle.v1"
)
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
MIGRATION_RE = re.compile(
    r"^scripts/(?P<sequence>[0-9]+)-.+(?P<concurrent>\.concurrent)?\.sql$"
)
S3_URI_RE = re.compile(r"^s3://[A-Za-z0-9][A-Za-z0-9._-]*/[^\s]+$")


class ProductionParityError(RuntimeError):
    """A parity contract, snapshot, or ledger is incomplete or inconsistent."""


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_json(value: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(value))


def production_database_host_hash(host: str) -> str:
    normalized = str(host or "").strip().rstrip(".").lower()
    if (
        not normalized
        or any(character.isspace() for character in normalized)
        or any(character in normalized for character in ("/", "@", "?", "#"))
    ):
        raise ProductionParityError("production database host is invalid")
    return sha256_json({"host": normalized})


def _require_sha(value: Any, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA_RE.fullmatch(normalized):
        raise ProductionParityError(f"{field_name} must be a full lowercase Git SHA")
    return normalized


def _require_hash(value: Any, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not HASH_RE.fullmatch(normalized):
        raise ProductionParityError(f"{field_name} must be a sha256 commitment")
    return normalized


def _require_unique_texts(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ProductionParityError(f"{field_name} must be a list")
    normalized = [str(item or "").strip() for item in value]
    if any(not item for item in normalized) or len(normalized) != len(set(normalized)):
        raise ProductionParityError(f"{field_name} must contain unique nonempty strings")
    return normalized


def migration_sequence(path: str) -> tuple[int, str]:
    match = MIGRATION_RE.fullmatch(str(path))
    if match is None:
        raise ProductionParityError(f"invalid migration path: {path}")
    return int(match.group("sequence")), str(path)


def normalize_migrations(value: Any, *, field_name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ProductionParityError(f"{field_name} must be a list")
    normalized: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ProductionParityError(f"{field_name}[{index}] must be an object")
        path = str(raw.get("path") or "").strip()
        sequence, _ = migration_sequence(path)
        digest = _require_hash(
            raw.get("sha256"), field_name=f"{field_name}[{index}].sha256"
        )
        transaction_mode = str(raw.get("transaction_mode") or "").strip()
        expected_mode = (
            "autocommit" if path.endswith(".concurrent.sql") else "candidate-file"
        )
        if transaction_mode != expected_mode:
            raise ProductionParityError(
                f"{field_name}[{index}] transaction mode differs from its path"
            )
        if int(raw.get("sequence", -1)) != sequence or path in seen_paths:
            raise ProductionParityError(f"{field_name} has invalid migration identity")
        seen_paths.add(path)
        normalized.append(
            {
                "path": path,
                "sequence": sequence,
                "sha256": digest,
                "transaction_mode": transaction_mode,
            }
        )
    if normalized != sorted(normalized, key=lambda item: (item["sequence"], item["path"])):
        raise ProductionParityError(f"{field_name} is not in production order")
    return normalized


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    supplied_hash = _require_hash(
        document.pop("contract_hash", ""), field_name="contract_hash"
    )
    if document.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise ProductionParityError("production parity contract schema differs")
    document["base_sha"] = _require_sha(document.get("base_sha"), field_name="base_sha")
    document["candidate_sha"] = _require_sha(
        document.get("candidate_sha"), field_name="candidate_sha"
    )
    document["changed_paths"] = _require_unique_texts(
        document.get("changed_paths"), field_name="changed_paths"
    )
    if document["changed_paths"] != sorted(document["changed_paths"]):
        raise ProductionParityError("changed_paths must be sorted")
    risk = document.get("risk")
    if not isinstance(risk, Mapping):
        raise ProductionParityError("risk classification is missing")
    risk_class = str(risk.get("class") or "")
    if risk_class not in {"low", "high"}:
        raise ProductionParityError("risk class is unsupported")
    reasons = _require_unique_texts(risk.get("reasons"), field_name="risk.reasons")
    if reasons != sorted(reasons):
        raise ProductionParityError("risk reasons must be sorted")
    if bool(risk.get("full_physical_required")) != (risk_class == "high"):
        raise ProductionParityError("risk and full physical requirement differ")
    document["risk"] = {
        "class": risk_class,
        "full_physical_required": risk_class == "high",
        "reasons": reasons,
    }
    sources = document.get("source_commitments")
    if not isinstance(sources, list) or not sources:
        raise ProductionParityError("source commitments are missing")
    normalized_sources: list[dict[str, str]] = []
    source_paths: set[str] = set()
    for index, raw in enumerate(sources):
        if not isinstance(raw, Mapping):
            raise ProductionParityError(f"source_commitments[{index}] is invalid")
        path = str(raw.get("path") or "").strip()
        if not path or path.startswith("/") or ".." in Path(path).parts:
            raise ProductionParityError(f"source_commitments[{index}].path is invalid")
        digest = _require_hash(
            raw.get("sha256"), field_name=f"source_commitments[{index}].sha256"
        )
        if path in source_paths:
            raise ProductionParityError("source commitment paths are duplicated")
        source_paths.add(path)
        normalized_sources.append({"path": path, "sha256": digest})
    if normalized_sources != sorted(normalized_sources, key=lambda item: item["path"]):
        raise ProductionParityError("source commitments must be sorted")
    document["source_commitments"] = normalized_sources
    document["migrations"] = normalize_migrations(
        document.get("migrations"), field_name="migrations"
    )
    for field_name in (
        "behavior_contract_hash",
        "protected_manifest_hash",
        "historical_oracle_hash",
        "runtime_config_hash",
    ):
        document[field_name] = _require_hash(
            document.get(field_name), field_name=field_name
        )
    if not isinstance(document.get("policy_commitments"), Mapping):
        raise ProductionParityError("policy commitments are missing")
    runtime_config_keys = _require_unique_texts(
        document.get("runtime_config_keys"), field_name="runtime_config_keys"
    )
    if runtime_config_keys != sorted(runtime_config_keys):
        raise ProductionParityError("runtime config keys must be sorted")
    document["runtime_config_keys"] = runtime_config_keys
    expected_hash = sha256_json(document)
    if supplied_hash != expected_hash:
        raise ProductionParityError("production parity contract hash differs")
    return {**document, "contract_hash": expected_hash}


def validate_historical_oracle(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    if document.get("schema_version") != HISTORICAL_ORACLE_SCHEMA_VERSION:
        raise ProductionParityError("historical production oracle schema differs")
    reference = document.get("reference_window")
    if not isinstance(reference, Mapping):
        raise ProductionParityError("historical production oracle reference is missing")
    _require_sha(reference.get("end_commit"), field_name="reference_window.end_commit")
    behaviors = document.get("required_behaviors")
    if not isinstance(behaviors, list) or not behaviors:
        raise ProductionParityError("historical production behaviors are missing")
    normalized: list[dict[str, Any]] = []
    behavior_ids: set[str] = set()
    for index, raw in enumerate(behaviors):
        if not isinstance(raw, Mapping):
            raise ProductionParityError(f"required_behaviors[{index}] is invalid")
        behavior_id = str(raw.get("behavior_id") or "").strip()
        description = str(raw.get("description") or "").strip()
        if (
            not re.fullmatch(r"[a-z0-9-]+", behavior_id)
            or behavior_id in behavior_ids
            or not description
        ):
            raise ProductionParityError("historical behavior identity is invalid")
        behavior_ids.add(behavior_id)
        normalized.append(
            {
                "behavior_id": behavior_id,
                "description": description,
                "fast_stage_ids": _require_unique_texts(
                    raw.get("fast_stage_ids"),
                    field_name=f"required_behaviors[{index}].fast_stage_ids",
                ),
                "full_stage_ids": _require_unique_texts(
                    raw.get("full_stage_ids"),
                    field_name=f"required_behaviors[{index}].full_stage_ids",
                ),
            }
        )
    if document.get("use_policy") != (
        "behavioral oracle only; never restore historical source or configuration"
    ):
        raise ProductionParityError("historical production oracle policy differs")
    document["required_behaviors"] = normalized
    return document


def required_oracle_stage_ids(
    value: Mapping[str, Any], *, lane: str
) -> tuple[str, ...]:
    if lane not in {"fast", "full"}:
        raise ProductionParityError("historical oracle lane is invalid")
    oracle = validate_historical_oracle(value)
    key = f"{lane}_stage_ids"
    return tuple(
        sorted(
            {
                stage
                for behavior in oracle["required_behaviors"]
                for stage in behavior[key]
            }
        )
    )


def validate_snapshot_manifest(
    value: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    document = dict(value)
    supplied_hash = _require_hash(
        document.pop("manifest_hash", ""), field_name="manifest_hash"
    )
    if document.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ProductionParityError("production snapshot schema differs")
    if document.get("source_environment") != "production-read-only":
        raise ProductionParityError("snapshot was not captured from the read-only source")
    document["source_host_hash"] = _require_hash(
        document.get("source_host_hash"), field_name="source_host_hash"
    )
    document["capture_sha"] = _require_sha(
        document.get("capture_sha"), field_name="capture_sha"
    )
    document["capture_contract_hash"] = _require_hash(
        document.get("capture_contract_hash"),
        field_name="capture_contract_hash",
    )
    document["source_sha"] = _require_sha(
        document.get("source_sha"), field_name="source_sha"
    )
    try:
        captured_at = datetime.fromisoformat(str(document.get("captured_at") or ""))
        expires_at = datetime.fromisoformat(str(document.get("expires_at") or ""))
    except ValueError as exc:
        raise ProductionParityError("snapshot timestamps are invalid") from exc
    if captured_at.tzinfo is None or expires_at.tzinfo is None:
        raise ProductionParityError("snapshot timestamps must be timezone aware")
    if expires_at <= captured_at:
        raise ProductionParityError("snapshot expiry must follow capture")
    current = now or datetime.now(timezone.utc)
    if current.astimezone(timezone.utc) >= expires_at.astimezone(timezone.utc):
        raise ProductionParityError("production snapshot has expired")
    if document.get("capture_transaction_read_only") is not True:
        raise ProductionParityError("snapshot capture was not transaction-read-only")
    archive = document.get("archive")
    if not isinstance(archive, Mapping):
        raise ProductionParityError("snapshot archive metadata is missing")
    archive_uri = str(archive.get("s3_uri") or "")
    if not S3_URI_RE.fullmatch(archive_uri):
        raise ProductionParityError("snapshot archive URI is invalid")
    archive_hash = _require_hash(archive.get("sha256"), field_name="archive.sha256")
    archive_size = archive.get("size_bytes")
    if not isinstance(archive_size, int) or isinstance(archive_size, bool) or archive_size <= 0:
        raise ProductionParityError("snapshot archive size is invalid")
    kms_key = str(archive.get("kms_key_arn") or "")
    if not kms_key.startswith("arn:aws:kms:"):
        raise ProductionParityError("snapshot archive is not KMS-bound")
    if archive.get("format") != "postgres-custom":
        raise ProductionParityError("snapshot archive format is unsupported")
    document["archive"] = {
        "format": "postgres-custom",
        "kms_key_arn": kms_key,
        "s3_uri": archive_uri,
        "sha256": archive_hash,
        "size_bytes": archive_size,
    }
    database = document.get("database")
    if not isinstance(database, Mapping):
        raise ProductionParityError("snapshot database evidence is missing")
    if database.get("server_version_num") is None:
        raise ProductionParityError("snapshot server version is missing")
    relation_count = database.get("relation_count")
    total_relation_bytes = database.get("total_relation_bytes")
    largest_relation_bytes = database.get("largest_relation_bytes")
    capture_utc_date = str(database.get("capture_utc_date") or "")
    target_rebenchmark_date = str(database.get("target_rebenchmark_date") or "")
    try:
        capture_date_value = datetime.fromisoformat(capture_utc_date).date()
        target_date_value = datetime.fromisoformat(target_rebenchmark_date).date()
    except ValueError as exc:
        raise ProductionParityError("snapshot database date frontier is invalid") from exc
    latest_completed_benchmark_date = database.get(
        "latest_completed_benchmark_date"
    )
    current_day_rebenchmark_run_count = database.get(
        "current_day_rebenchmark_run_count"
    )
    current_day_benchmark_bundle_count = database.get(
        "current_day_benchmark_bundle_count"
    )
    if (
        not isinstance(relation_count, int)
        or isinstance(relation_count, bool)
        or relation_count <= 0
        or not isinstance(total_relation_bytes, int)
        or isinstance(total_relation_bytes, bool)
        or total_relation_bytes <= 0
        or not isinstance(largest_relation_bytes, int)
        or isinstance(largest_relation_bytes, bool)
        or largest_relation_bytes <= 0
        or largest_relation_bytes > total_relation_bytes
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", capture_utc_date)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", target_rebenchmark_date)
        or target_date_value < capture_date_value
        or target_date_value > capture_date_value + timedelta(days=1)
        or (
            latest_completed_benchmark_date is not None
            and not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}",
                str(latest_completed_benchmark_date),
            )
        )
        or (
            latest_completed_benchmark_date is not None
            and str(latest_completed_benchmark_date) > capture_utc_date
        )
        or not isinstance(current_day_rebenchmark_run_count, int)
        or isinstance(current_day_rebenchmark_run_count, bool)
        or current_day_rebenchmark_run_count < 0
        or not isinstance(current_day_benchmark_bundle_count, int)
        or isinstance(current_day_benchmark_bundle_count, bool)
        or current_day_benchmark_bundle_count < 0
    ):
        raise ProductionParityError("snapshot database shape is invalid")
    document["database"] = {
        "server_version_num": str(database["server_version_num"]),
        "relation_count": relation_count,
        "total_relation_bytes": total_relation_bytes,
        "largest_relation_bytes": largest_relation_bytes,
        "capture_utc_date": capture_utc_date,
        "target_rebenchmark_date": target_rebenchmark_date,
        "latest_completed_benchmark_date": (
            str(latest_completed_benchmark_date)
            if latest_completed_benchmark_date is not None
            else None
        ),
        "current_day_rebenchmark_run_count": current_day_rebenchmark_run_count,
        "current_day_benchmark_bundle_count": current_day_benchmark_bundle_count,
    }
    document["migrations"] = normalize_migrations(
        document.get("migrations"), field_name="migrations"
    )
    expected_hash = sha256_json(document)
    if supplied_hash != expected_hash:
        raise ProductionParityError("production snapshot manifest hash differs")
    return {**document, "manifest_hash": expected_hash}


def migration_delta(
    *,
    snapshot_migrations: Sequence[Mapping[str, Any]],
    candidate_migrations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return only unapplied immutable migrations, failing on rewritten history."""

    snapshot = normalize_migrations(list(snapshot_migrations), field_name="snapshot_migrations")
    candidate = normalize_migrations(list(candidate_migrations), field_name="candidate_migrations")
    candidate_by_path = {item["path"]: item for item in candidate}
    for applied in snapshot:
        current = candidate_by_path.get(applied["path"])
        if current is None:
            raise ProductionParityError(
                f"candidate removed applied migration {applied['path']}"
            )
        if current != applied:
            raise ProductionParityError(
                f"candidate rewrote applied migration {applied['path']}"
            )
    applied_paths = {item["path"] for item in snapshot}
    delta = [item for item in candidate if item["path"] not in applied_paths]
    if snapshot and delta:
        highest_applied = max(item["sequence"] for item in snapshot)
        if any(item["sequence"] < highest_applied for item in delta):
            raise ProductionParityError(
                "candidate inserts a migration before the production snapshot frontier"
            )
    return delta


@dataclass
class StageLedger:
    lane: str
    candidate_sha: str
    contract_hash: str
    snapshot_hash: str
    critical_stage_ids: tuple[str, ...]
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    stages: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.lane not in {"fast", "full"}:
            raise ProductionParityError("parity ledger lane is unsupported")
        self.candidate_sha = _require_sha(
            self.candidate_sha, field_name="candidate_sha"
        )
        self.contract_hash = _require_hash(
            self.contract_hash, field_name="contract_hash"
        )
        self.snapshot_hash = _require_hash(
            self.snapshot_hash, field_name="snapshot_hash"
        )
        if not self.critical_stage_ids or len(self.critical_stage_ids) != len(
            set(self.critical_stage_ids)
        ):
            raise ProductionParityError("critical parity stage identities are invalid")

    def record(
        self,
        stage_id: str,
        *,
        status: str,
        duration_seconds: float,
        evidence: Mapping[str, Any] | None = None,
        reason: str = "",
    ) -> None:
        normalized_id = str(stage_id or "").strip()
        if not normalized_id or any(item["stage_id"] == normalized_id for item in self.stages):
            raise ProductionParityError("parity stage identity is empty or duplicated")
        if status not in {"passed", "failed", "unexercised", "skipped"}:
            raise ProductionParityError("parity stage status is unsupported")
        if duration_seconds < 0:
            raise ProductionParityError("parity stage duration is invalid")
        evidence_doc = dict(evidence or {})
        self.stages.append(
            {
                "stage_id": normalized_id,
                "status": status,
                "duration_seconds": round(float(duration_seconds), 6),
                "evidence": evidence_doc,
                "evidence_hash": sha256_json(evidence_doc),
                "reason": str(reason or ""),
            }
        )

    def finalize(self, *, completed_at: datetime | None = None) -> dict[str, Any]:
        by_id = {item["stage_id"]: item for item in self.stages}
        missing = [stage for stage in self.critical_stage_ids if stage not in by_id]
        failed = [
            stage
            for stage in self.critical_stage_ids
            if stage in by_id and by_id[stage]["status"] != "passed"
        ]
        status = "passed" if not missing and not failed else "failed"
        body = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "lane": self.lane,
            "candidate_sha": self.candidate_sha,
            "contract_hash": self.contract_hash,
            "snapshot_hash": self.snapshot_hash,
            "status": status,
            "started_at": self.started_at.astimezone(timezone.utc).isoformat(),
            "completed_at": (completed_at or datetime.now(timezone.utc))
            .astimezone(timezone.utc)
            .isoformat(),
            "critical_stage_ids": list(self.critical_stage_ids),
            "missing_critical_stage_ids": missing,
            "failed_critical_stage_ids": failed,
            "stages": list(self.stages),
        }
        return {**body, "ledger_hash": sha256_json(body)}


def validate_ledger(value: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(value)
    supplied_hash = _require_hash(
        document.pop("ledger_hash", ""), field_name="ledger_hash"
    )
    if document.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ProductionParityError("production parity ledger schema differs")
    if document.get("lane") not in {"fast", "full"}:
        raise ProductionParityError("production parity ledger lane is invalid")
    _require_sha(document.get("candidate_sha"), field_name="candidate_sha")
    _require_hash(document.get("contract_hash"), field_name="contract_hash")
    _require_hash(document.get("snapshot_hash"), field_name="snapshot_hash")
    critical = _require_unique_texts(
        document.get("critical_stage_ids"), field_name="critical_stage_ids"
    )
    stages = document.get("stages")
    if not isinstance(stages, list):
        raise ProductionParityError("parity ledger stages are missing")
    try:
        started_at = datetime.fromisoformat(str(document.get("started_at") or ""))
        completed_at = datetime.fromisoformat(str(document.get("completed_at") or ""))
    except ValueError as exc:
        raise ProductionParityError("parity ledger timestamps are invalid") from exc
    if (
        started_at.tzinfo is None
        or completed_at.tzinfo is None
        or completed_at < started_at
    ):
        raise ProductionParityError("parity ledger timestamps are inconsistent")
    stage_ids: set[str] = set()
    for index, stage in enumerate(stages):
        if not isinstance(stage, Mapping):
            raise ProductionParityError(f"stages[{index}] is invalid")
        stage_id = str(stage.get("stage_id") or "")
        if not stage_id or stage_id in stage_ids:
            raise ProductionParityError("parity ledger stage identities are invalid")
        stage_ids.add(stage_id)
        if stage.get("status") not in {"passed", "failed", "unexercised", "skipped"}:
            raise ProductionParityError("parity ledger stage status is invalid")
        duration = stage.get("duration_seconds")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or duration < 0
            or not isinstance(stage.get("reason"), str)
        ):
            raise ProductionParityError("parity ledger stage metadata is invalid")
        _require_hash(stage.get("evidence_hash"), field_name=f"stages[{index}].evidence_hash")
        evidence = stage.get("evidence")
        if not isinstance(evidence, Mapping) or sha256_json(evidence) != stage["evidence_hash"]:
            raise ProductionParityError("parity ledger stage evidence differs")
    failed = [
        stage
        for stage in critical
        if stage not in stage_ids
        or next(item for item in stages if item["stage_id"] == stage).get("status") != "passed"
    ]
    expected_status = "passed" if not failed else "failed"
    if document.get("status") != expected_status:
        raise ProductionParityError("parity ledger status differs from critical stages")
    expected_missing = [stage for stage in critical if stage not in stage_ids]
    expected_failed = [stage for stage in critical if stage in stage_ids and stage in failed]
    if document.get("missing_critical_stage_ids") != expected_missing:
        raise ProductionParityError("parity ledger missing-stage summary differs")
    if document.get("failed_critical_stage_ids") != expected_failed:
        raise ProductionParityError("parity ledger failed-stage summary differs")
    expected_hash = sha256_json(document)
    if supplied_hash != expected_hash:
        raise ProductionParityError("production parity ledger hash differs")
    return {**document, "ledger_hash": expected_hash}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def validate_archive(path: Path, manifest: Mapping[str, Any]) -> None:
    normalized = validate_snapshot_manifest(manifest)
    archive = normalized["archive"]
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ProductionParityError("snapshot archive is unavailable") from exc
    if size != int(archive["size_bytes"]):
        raise ProductionParityError("snapshot archive size differs")
    if file_sha256(path) != archive["sha256"]:
        raise ProductionParityError("snapshot archive hash differs")


def safe_database_target(dsn: str, *, production_host: str) -> None:
    parsed = urlparse(str(dsn or ""))
    database = parsed.path.lstrip("/")
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.hostname
        or parsed.hostname.lower() == str(production_host or "").lower()
        or not database.startswith("leadpoet_parity_")
    ):
        raise ProductionParityError(
            "snapshot restore target is not an isolated parity database"
        )


def verify_contract_checkout(root: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify that every committed source is the candidate's exact Git blob."""

    contract = validate_contract(value)
    resolved_root = root.resolve()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=resolved_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProductionParityError("candidate checkout identity is unavailable") from exc
    if head.returncode != 0 or head.stdout.strip().lower() != contract["candidate_sha"]:
        raise ProductionParityError("candidate checkout HEAD differs from the contract")
    for commitment in contract["source_commitments"]:
        path = str(commitment["path"])
        blob = subprocess.run(
            ["git", "show", f"{contract['candidate_sha']}:{path}"],
            cwd=resolved_root,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if blob.returncode != 0 or sha256_bytes(blob.stdout) != commitment["sha256"]:
            raise ProductionParityError(f"candidate source commitment differs: {path}")
        worktree_path = resolved_root / path
        if not worktree_path.is_file() or file_sha256(worktree_path) != commitment["sha256"]:
            raise ProductionParityError(f"candidate worktree source differs: {path}")
    return contract


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "HASH_RE",
    "HISTORICAL_ORACLE_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "MIGRATION_RE",
    "ProductionParityError",
    "SHA_RE",
    "SNAPSHOT_SCHEMA_VERSION",
    "StageLedger",
    "canonical_bytes",
    "file_sha256",
    "migration_delta",
    "migration_sequence",
    "normalize_migrations",
    "production_database_host_hash",
    "required_oracle_stage_ids",
    "safe_database_target",
    "sha256_bytes",
    "sha256_json",
    "validate_archive",
    "validate_contract",
    "validate_historical_oracle",
    "verify_contract_checkout",
    "validate_ledger",
    "validate_snapshot_manifest",
]

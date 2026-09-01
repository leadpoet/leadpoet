"""Append-only SSE-KMS custody for exact official-baseline documents.

Supabase retains only hashes, accounting, and protected job references.  Full
model transitions and terminal CompanyOutput records live under deterministic
S3 keys, are written conditionally, and are byte-compared on every replay.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping

from gateway.research_lab.official_baseline_model_runner import (
    OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
    OfficialBaselineAttemptStore,
    OfficialBaselineModelError,
)
from gateway.research_lab.official_baseline_store import (
    official_baseline_action_replay_identity,
)
from research_lab.canonical import canonical_json, sha256_bytes, sha256_json
from research_lab.model_runner_protocol import ExactModelRunnerRegistration


OFFICIAL_BASELINE_TRANSITION_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_transition.v1"
)
OFFICIAL_BASELINE_FRONTIER_SNAPSHOT_SCHEMA_VERSION = (
    "leadpoet.research_lab.official_baseline_frontier_snapshot.v1"
)
OFFICIAL_BASELINE_CUSTODY_S3_PREFIX_ENV = (
    "RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX"
)
OFFICIAL_BASELINE_CUSTODY_KMS_KEY_ENV = (
    "RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID"
)
OFFICIAL_BASELINE_CUSTODY_SUBPREFIX = "official-baseline-v1"

_HASH_RE = re.compile(r"sha256:[0-9a-f]{64}")
_UNIT_RE = re.compile(r"baseline_icp:[0-9a-f]{64}")
_TERMINAL_RE = re.compile(r"baseline_terminal:[0-9a-f]{64}")
_S3_PREFIX_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._=@+-]{0,255}")
_KMS_KEY_ID_RE = re.compile(
    r"(?:arn:aws(?:-[a-z]+)?:kms:[a-z0-9-]+:[0-9]{12}:"
    r"(?:key/[A-Za-z0-9-]+|alias/[A-Za-z0-9/_-]+)"
    r"|alias/[A-Za-z0-9/_-]+|[A-Za-z0-9-]{1,128})"
)


class OfficialBaselineCustodyError(OfficialBaselineModelError):
    """Encrypted append-only custody is missing, mutable, or inconsistent."""


def official_baseline_custody_configuration(
    environment: Mapping[str, Any],
) -> dict[str, str]:
    """Resolve the existing production trace pair into one isolated prefix."""

    if not isinstance(environment, Mapping):
        raise OfficialBaselineCustodyError(
            "official baseline custody environment is invalid"
        )
    prefix_uri = (
        str(environment.get(OFFICIAL_BASELINE_CUSTODY_S3_PREFIX_ENV) or "")
        .strip()
        .rstrip("/")
    )
    kms_key_id = str(
        environment.get(OFFICIAL_BASELINE_CUSTODY_KMS_KEY_ENV) or ""
    ).strip()
    if not prefix_uri.startswith("s3://") or not kms_key_id:
        raise OfficialBaselineCustodyError(
            "official baseline encrypted custody is unavailable"
        )
    bucket, separator, prefix = prefix_uri[5:].partition("/")
    prefix_parts = prefix.strip("/").split("/")
    if (
        not separator
        or re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket) is None
        or not prefix_parts
        or any(_S3_PREFIX_PART_RE.fullmatch(part) is None for part in prefix_parts)
        or _KMS_KEY_ID_RE.fullmatch(kms_key_id) is None
    ):
        raise OfficialBaselineCustodyError(
            "official baseline encrypted custody configuration is invalid"
        )
    return {
        "bucket": bucket,
        "prefix": "/".join((*prefix_parts, OFFICIAL_BASELINE_CUSTODY_SUBPREFIX)),
        "kms_key_id": kms_key_id,
    }


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json(dict(value)).encode("utf-8")


def _missing_object(exc: BaseException) -> bool:
    response = getattr(exc, "response", None)
    error = response.get("Error") if isinstance(response, Mapping) else None
    code = str(error.get("Code") or "") if isinstance(error, Mapping) else ""
    return code in {"404", "NoSuchKey", "NotFound"} or type(exc).__name__ in {
        "NoSuchKey",
        "NotFound",
    }


class S3OfficialBaselineDocumentCustody:
    """One fixed encrypted prefix used by transitions and terminal records."""

    def __init__(
        self,
        *,
        client: Any,
        bucket: str,
        prefix: str,
        kms_key_id: str,
    ) -> None:
        if not all(
            callable(getattr(client, method, None))
            for method in ("get_object", "put_object")
        ):
            raise OfficialBaselineCustodyError(
                "official baseline S3 custody client is unavailable"
            )
        normalized_bucket = str(bucket or "").strip()
        normalized_prefix = str(prefix or "").strip("/")
        normalized_kms = str(kms_key_id or "").strip()
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{2,62}", normalized_bucket)
            or not normalized_prefix
            or ".." in normalized_prefix.split("/")
            or not normalized_kms
        ):
            raise OfficialBaselineCustodyError(
                "official baseline S3 custody configuration is invalid"
            )
        self._client = client
        self._bucket = normalized_bucket
        self._prefix = normalized_prefix
        self._kms_key_id = normalized_kms

    @property
    def custody_identity_sha256(self) -> str:
        return sha256_json(
            {
                "schema_version": (
                    "leadpoet.research_lab.official_baseline_s3_custody.v1"
                ),
                "bucket": self._bucket,
                "prefix": self._prefix,
                "kms_key_id_sha256": sha256_bytes(self._kms_key_id.encode("utf-8")),
            }
        )

    def _key(self, *parts: str) -> str:
        if any(not part or "/" in part or part in {".", ".."} for part in parts):
            raise OfficialBaselineCustodyError(
                "official baseline custody key is invalid"
            )
        return "/".join((self._prefix, *parts))

    def _read_bytes(self, key: str, *, allow_missing: bool = False) -> bytes | None:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - SDK errors are normalized
            if allow_missing and _missing_object(exc):
                return None
            raise OfficialBaselineCustodyError(
                "official baseline encrypted custody read failed"
            ) from exc
        metadata = response.get("Metadata")
        body_reader = response.get("Body")
        if (
            response.get("ServerSideEncryption") != "aws:kms"
            or not isinstance(metadata, Mapping)
            or not callable(getattr(body_reader, "read", None))
            or not callable(getattr(body_reader, "close", None))
        ):
            close = getattr(body_reader, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:  # noqa: BLE001 - cleanup is fail-closed
                    raise OfficialBaselineCustodyError(
                        "official baseline custody response cleanup failed"
                    ) from exc
            raise OfficialBaselineCustodyError(
                "official baseline custody object is not SSE-KMS protected"
            )
        read_error: Exception | None = None
        close_error: Exception | None = None
        body = b""
        try:
            body = bytes(body_reader.read())
        except Exception as exc:  # noqa: BLE001 - normalize SDK/body failures
            read_error = exc
        try:
            body_reader.close()
        except Exception as exc:  # noqa: BLE001 - cleanup is fail-closed
            close_error = exc
        if read_error is not None:
            raise OfficialBaselineCustodyError(
                "official baseline encrypted custody body read failed"
            ) from read_error
        if close_error is not None:
            raise OfficialBaselineCustodyError(
                "official baseline custody response cleanup failed"
            ) from close_error
        expected = str(metadata.get("content-sha256") or "")
        expected_kms = str(metadata.get("kms-key-id-sha256") or "")
        if (
            expected != sha256_bytes(body).removeprefix("sha256:")
            or expected_kms
            != sha256_bytes(self._kms_key_id.encode("utf-8")).removeprefix(
                "sha256:"
            )
        ):
            raise OfficialBaselineCustodyError(
                "official baseline custody content commitment differs"
            )
        return body

    def _read_document(
        self, key: str, *, allow_missing: bool = False
    ) -> dict[str, Any] | None:
        body = self._read_bytes(key, allow_missing=allow_missing)
        if body is None:
            return None
        try:
            value = json.loads(body.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - encrypted content is untrusted
            raise OfficialBaselineCustodyError(
                "official baseline custody document is invalid"
            ) from exc
        if not isinstance(value, Mapping) or _canonical_bytes(value) != body:
            raise OfficialBaselineCustodyError(
                "official baseline custody document is not canonical"
            )
        return dict(value)

    def _append_document(self, key: str, value: Mapping[str, Any]) -> bool:
        document = dict(value)
        body = _canonical_bytes(document)
        existing = self._read_bytes(key, allow_missing=True)
        if existing is not None:
            if existing != body:
                raise OfficialBaselineCustodyError(
                    "official baseline append-only custody conflict"
                )
            return False
        appended = True
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=self._kms_key_id,
                IfNoneMatch="*",
                Metadata={
                    "content-sha256": sha256_bytes(body).removeprefix("sha256:"),
                    "kms-key-id-sha256": sha256_bytes(
                        self._kms_key_id.encode("utf-8")
                    ).removeprefix("sha256:"),
                },
            )
        except Exception as exc:  # noqa: BLE001 - resolve conditional races
            raced = self._read_bytes(key, allow_missing=True)
            if raced != body:
                raise OfficialBaselineCustodyError(
                    "official baseline append-only custody write failed"
                ) from exc
            appended = False
        readback = self._read_bytes(key)
        if readback != body:
            raise OfficialBaselineCustodyError(
                "official baseline custody readback differs"
            )
        return appended

    def append_protected_action_claim(
        self,
        *,
        preparation_sha256: str,
        claim: Mapping[str, Any],
    ) -> bool:
        """Atomically claim one protected action before physical dispatch."""

        if _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline protected preparation hash is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        appended = self._append_document(
            self._key("protected-action", digest, "claim.json"), claim
        )
        if self.load_protected_action_claim(
            preparation_sha256=preparation_sha256
        ) != dict(claim):
            raise OfficialBaselineCustodyError(
                "official baseline protected claim readback differs"
            )
        return appended

    def load_protected_action_claim(
        self, *, preparation_sha256: str
    ) -> Mapping[str, Any] | None:
        if _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline protected preparation hash is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        return self._read_document(
            self._key("protected-action", digest, "claim.json"),
            allow_missing=True,
        )

    def append_protected_action_progress(
        self,
        *,
        preparation_sha256: str,
        progress: Mapping[str, Any],
    ) -> bool:
        """Append one compiler-owned provider progress checkpoint.

        Long-running providers use this after physical authorization returns a
        durable request/run reference and before polling.  Exact replay is
        idempotent; a different document for the same preparation is a hard
        conflict, so restart can never authorize the initial request twice.
        """

        if (
            _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None
            or not isinstance(progress, Mapping)
            or not progress
        ):
            raise OfficialBaselineCustodyError(
                "official baseline protected progress is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        appended = self._append_document(
            self._key("protected-action", digest, "progress.json"),
            progress,
        )
        if self.load_protected_action_progress(
            preparation_sha256=preparation_sha256
        ) != dict(progress):
            raise OfficialBaselineCustodyError(
                "official baseline protected progress readback differs"
            )
        return appended

    def load_protected_action_progress(
        self, *, preparation_sha256: str
    ) -> Mapping[str, Any] | None:
        if _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline protected preparation hash is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        return self._read_document(
            self._key("protected-action", digest, "progress.json"),
            allow_missing=True,
        )

    def persist_protected_action_terminal(
        self,
        *,
        preparation_sha256: str,
        terminal: Mapping[str, Any],
    ) -> None:
        if _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline protected preparation hash is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        self._append_document(
            self._key("protected-action", digest, "terminal.json"), terminal
        )
        if self.load_protected_action_terminal(
            preparation_sha256=preparation_sha256
        ) != dict(terminal):
            raise OfficialBaselineCustodyError(
                "official baseline protected terminal readback differs"
            )

    def load_protected_action_terminal(
        self, *, preparation_sha256: str
    ) -> Mapping[str, Any] | None:
        if _HASH_RE.fullmatch(str(preparation_sha256 or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline protected preparation hash is invalid"
            )
        digest = str(preparation_sha256).removeprefix("sha256:")
        return self._read_document(
            self._key("protected-action", digest, "terminal.json"),
            allow_missing=True,
        )

    def persist_terminal_record(
        self,
        *,
        record_identity_sha256: str,
        record: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if _HASH_RE.fullmatch(
            str(record_identity_sha256 or "")
        ) is None or not isinstance(record, Mapping):
            raise OfficialBaselineCustodyError(
                "official baseline terminal record identity is invalid"
            )
        digest = record_identity_sha256.removeprefix("sha256:")
        terminal_ref = "baseline_terminal:" + digest
        self._append_document(
            self._key("terminal", digest + ".json"),
            record,
        )
        return {
            "terminal_record_ref": terminal_ref,
            "terminal_record_sha256": sha256_json(dict(record)),
        }

    def load_terminal_record(self, *, terminal_record_ref: str) -> Mapping[str, Any]:
        if _TERMINAL_RE.fullmatch(str(terminal_record_ref or "")) is None:
            raise OfficialBaselineCustodyError(
                "official baseline terminal record reference is invalid"
            )
        digest = str(terminal_record_ref).split(":", 1)[1]
        value = self._read_document(self._key("terminal", digest + ".json"))
        if value is None:  # pragma: no cover - non-missing read cannot return None
            raise OfficialBaselineCustodyError(
                "official baseline terminal record is missing"
            )
        return value

    def transition_repository(
        self,
        *,
        run_sha256: str,
        unit_ref: str,
        registration: ExactModelRunnerRegistration,
        attempt_store: OfficialBaselineAttemptStore,
    ) -> "S3OfficialBaselineTransitionRepository":
        return S3OfficialBaselineTransitionRepository(
            custody=self,
            run_sha256=run_sha256,
            unit_ref=unit_ref,
            registration=registration,
            attempt_store=attempt_store,
        )


class S3OfficialBaselineTransitionRepository:
    """Generation-pinned full transition replay with append-only frontiers."""

    def __init__(
        self,
        *,
        custody: S3OfficialBaselineDocumentCustody,
        run_sha256: str,
        unit_ref: str,
        registration: ExactModelRunnerRegistration,
        attempt_store: OfficialBaselineAttemptStore,
    ) -> None:
        if (
            not isinstance(custody, S3OfficialBaselineDocumentCustody)
            or _HASH_RE.fullmatch(str(run_sha256 or "")) is None
            or _UNIT_RE.fullmatch(str(unit_ref or "")) is None
            or not isinstance(registration, ExactModelRunnerRegistration)
            or any(
                not callable(getattr(attempt_store, method, None))
                for method in ("load_replay",)
            )
        ):
            raise OfficialBaselineCustodyError(
                "official baseline transition repository is invalid"
            )
        self._custody = custody
        self._run_sha256 = str(run_sha256)
        self._unit_ref = str(unit_ref)
        self._registration = registration
        self._attempt_store = attempt_store
        self._unit_digest = sha256_json(
            {"run_sha256": self._run_sha256, "unit_ref": self._unit_ref}
        ).removeprefix("sha256:")

    def resolve_run_protocol_generation(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        artifact_key: str,
    ) -> str:
        if (
            experiment_hash != self._run_sha256
            or variant_id != "official_baseline"
            or artifact_key != self._registration.key
        ):
            raise OfficialBaselineCustodyError(
                "official baseline transition generation identity differs"
            )
        return self._registration.protocol_generation.protocol_generation_sha256

    def _transition_key(self, idempotency_key: str) -> str:
        value = str(idempotency_key or "").removeprefix("sha256:")
        if re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise OfficialBaselineCustodyError(
                "official baseline transition idempotency key is invalid"
            )
        return self._custody._key("transition", self._unit_digest, value + ".json")

    def _frontier_key(self, next_sequence: int) -> str:
        if type(next_sequence) is not int or not 0 <= next_sequence <= 10_000:
            raise OfficialBaselineCustodyError(
                "official baseline frontier sequence is invalid"
            )
        return self._custody._key(
            "frontier", self._unit_digest, f"{next_sequence:05d}.json"
        )

    @staticmethod
    def _empty_frontier() -> dict[str, Any]:
        return {
            "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
            "ordered_attempt_keys": [],
            "ordered_attempt_sha256s": [],
        }

    def frontier_before(self, action_sequence: int) -> Mapping[str, Any]:
        if action_sequence == 0:
            return self._empty_frontier()
        snapshot = self._custody._read_document(self._frontier_key(action_sequence))
        if not isinstance(snapshot, Mapping) or set(snapshot) != {
            "schema_version",
            "run_sha256",
            "unit_ref",
            "next_action_sequence",
            "frontier",
            "snapshot_sha256",
        }:
            raise OfficialBaselineCustodyError(
                "official baseline frontier snapshot is missing"
            )
        body = dict(snapshot)
        claimed = body.pop("snapshot_sha256")
        frontier = body.get("frontier")
        if (
            body.get("schema_version")
            != OFFICIAL_BASELINE_FRONTIER_SNAPSHOT_SCHEMA_VERSION
            or body.get("run_sha256") != self._run_sha256
            or body.get("unit_ref") != self._unit_ref
            or body.get("next_action_sequence") != action_sequence
            or not isinstance(frontier, Mapping)
            or set(frontier)
            != {
                "schema_version",
                "ordered_attempt_keys",
                "ordered_attempt_sha256s",
            }
            or frontier.get("schema_version")
            != OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION
            or sha256_json(body) != claimed
        ):
            raise OfficialBaselineCustodyError(
                "official baseline frontier snapshot differs"
            )
        keys = frontier.get("ordered_attempt_keys")
        hashes = frontier.get("ordered_attempt_sha256s")
        if (
            not isinstance(keys, list)
            or not isinstance(hashes, list)
            or len(keys) != action_sequence
            or len(hashes) != action_sequence
            or len(set(keys)) != len(keys)
            or any(
                _HASH_RE.fullmatch(str(value or "")) is None
                for value in [*keys, *hashes]
            )
        ):
            raise OfficialBaselineCustodyError(
                "official baseline frontier entries are invalid"
            )
        return deepcopy(dict(frontier))

    def expected_frontier_sha256(self, action_sequence: int) -> str:
        return sha256_json(dict(self.frontier_before(action_sequence)))

    def load_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        idempotency_key: str,
        artifact_key: str,
    ) -> Mapping[str, Any] | None:
        if (
            experiment_hash != self._run_sha256
            or variant_id != "official_baseline"
            or unit_ref != self._unit_ref
            or artifact_key != self._registration.key
        ):
            raise OfficialBaselineCustodyError(
                "official baseline transition lookup identity differs"
            )
        value = self._custody._read_document(
            self._transition_key(idempotency_key), allow_missing=True
        )
        if value is None:
            return None
        transition = self._validate_transition(value, idempotency_key=idempotency_key)
        self._ensure_frontier(transition)
        return {
            key: deepcopy(transition[key])
            for key in (
                "action",
                "continuation",
                "completion",
                "provider_receipt",
                "protocol_generation_sha256",
            )
        }

    def append_model_transition(
        self,
        *,
        experiment_hash: str,
        variant_id: str,
        unit_ref: str,
        artifact_key: str,
        action: Mapping[str, Any],
        continuation: Mapping[str, Any],
        completion: Mapping[str, Any],
        provider_receipt: Mapping[str, Any] | None,
        protocol_generation_sha256: str,
        replay_ref: Mapping[str, Any] | None = None,
    ) -> None:
        generation = self.resolve_run_protocol_generation(
            experiment_hash=experiment_hash,
            variant_id=variant_id,
            artifact_key=artifact_key,
        )
        if (
            unit_ref != self._unit_ref
            or generation != protocol_generation_sha256
            or not all(
                isinstance(value, Mapping)
                for value in (action, continuation, completion)
            )
            or (
                provider_receipt is not None
                and not isinstance(provider_receipt, Mapping)
            )
            or (replay_ref is not None and not isinstance(replay_ref, Mapping))
        ):
            raise OfficialBaselineCustodyError(
                "official baseline transition append identity differs"
            )
        sequence = action.get("sequence")
        if type(sequence) is not int or not 0 <= sequence <= 9_999:
            raise OfficialBaselineCustodyError(
                "official baseline transition sequence is invalid"
            )
        identity = {
            "run_sha256": self._run_sha256,
            "variant_id": "official_baseline",
            "unit_ref": self._unit_ref,
            "idempotency_key": str(action.get("idempotency_key") or ""),
        }
        body = {
            "schema_version": OFFICIAL_BASELINE_TRANSITION_SCHEMA_VERSION,
            "identity": identity,
            "action": dict(action),
            "continuation": dict(continuation),
            "completion": dict(completion),
            "provider_receipt": (
                None if provider_receipt is None else dict(provider_receipt)
            ),
            "replay_ref": None if replay_ref is None else dict(replay_ref),
            "protocol_generation_sha256": protocol_generation_sha256,
        }
        transition = {**body, "transition_sha256": sha256_json(body)}
        self._custody._append_document(
            self._transition_key(identity["idempotency_key"]), transition
        )
        self._ensure_frontier(transition)

    def _validate_transition(
        self, value: Mapping[str, Any], *, idempotency_key: str
    ) -> dict[str, Any]:
        if set(value) != {
            "schema_version",
            "identity",
            "action",
            "continuation",
            "completion",
            "provider_receipt",
            "replay_ref",
            "protocol_generation_sha256",
            "transition_sha256",
        }:
            raise OfficialBaselineCustodyError(
                "official baseline transition is not closed"
            )
        body = dict(value)
        claimed = body.pop("transition_sha256")
        identity = body.get("identity")
        action = body.get("action")
        if (
            body.get("schema_version") != OFFICIAL_BASELINE_TRANSITION_SCHEMA_VERSION
            or not isinstance(identity, Mapping)
            or dict(identity)
            != {
                "run_sha256": self._run_sha256,
                "variant_id": "official_baseline",
                "unit_ref": self._unit_ref,
                "idempotency_key": str(idempotency_key).removeprefix("sha256:"),
            }
            or not isinstance(action, Mapping)
            or str(action.get("idempotency_key") or "").removeprefix("sha256:")
            != str(idempotency_key).removeprefix("sha256:")
            or not isinstance(body.get("continuation"), Mapping)
            or not isinstance(body.get("completion"), Mapping)
            or body.get("protocol_generation_sha256")
            != self._registration.protocol_generation.protocol_generation_sha256
            or sha256_json(body) != claimed
        ):
            raise OfficialBaselineCustodyError(
                "official baseline transition identity differs"
            )
        return dict(value)

    def _ensure_frontier(self, transition: Mapping[str, Any]) -> None:
        action = transition["action"]
        sequence = action["sequence"]
        before = dict(self.frontier_before(sequence))
        identity = official_baseline_action_replay_identity(
            run_sha256=self._run_sha256,
            unit_ref=self._unit_ref,
            action=action,
        )
        replay = self._attempt_store.load_replay(identity=identity)
        if (
            replay.get("state") != "terminal_known"
            or replay.get("attempt_key") != identity["attempt_key"]
            or _HASH_RE.fullmatch(str(replay.get("attempt_sha256") or "")) is None
        ):
            raise OfficialBaselineCustodyError(
                "official baseline terminal attempt is not durable"
            )
        keys = [*before["ordered_attempt_keys"], identity["attempt_key"]]
        hashes = [*before["ordered_attempt_sha256s"], replay["attempt_sha256"]]
        frontier = {
            "schema_version": OFFICIAL_BASELINE_PROVIDER_FRONTIER_SCHEMA_VERSION,
            "ordered_attempt_keys": keys,
            "ordered_attempt_sha256s": hashes,
        }
        body = {
            "schema_version": OFFICIAL_BASELINE_FRONTIER_SNAPSHOT_SCHEMA_VERSION,
            "run_sha256": self._run_sha256,
            "unit_ref": self._unit_ref,
            "next_action_sequence": sequence + 1,
            "frontier": frontier,
        }
        self._custody._append_document(
            self._frontier_key(sequence + 1),
            {**body, "snapshot_sha256": sha256_json(body)},
        )


__all__ = [
    "OFFICIAL_BASELINE_CUSTODY_KMS_KEY_ENV",
    "OFFICIAL_BASELINE_CUSTODY_S3_PREFIX_ENV",
    "OFFICIAL_BASELINE_CUSTODY_SUBPREFIX",
    "OFFICIAL_BASELINE_FRONTIER_SNAPSHOT_SCHEMA_VERSION",
    "OFFICIAL_BASELINE_TRANSITION_SCHEMA_VERSION",
    "OfficialBaselineCustodyError",
    "S3OfficialBaselineDocumentCustody",
    "S3OfficialBaselineTransitionRepository",
    "official_baseline_custody_configuration",
]

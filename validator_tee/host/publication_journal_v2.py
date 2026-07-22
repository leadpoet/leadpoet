"""Crash-safe host journal for one authoritative V2 weight publication.

The journal contains public, signed material only.  It is written before the
gateway publication request and before any signed extrinsic is returned to the
Bittensor SDK, so a parent or process restart cannot create an ambiguous
publication window.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import sha256_json
from leadpoet_canonical.hotkey_authority_v2 import (
    validate_weight_extrinsic_authorization_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
)
from validator_tee.enclave.hotkey_authority_v2 import load_chain_signing_profile
from validator_tee.host.weight_authority_v2 import (
    HostWeightAuthorityV2Error,
    validate_stateful_epoch_evidence_v1,
)


LEGACY_JOURNAL_SCHEMA_VERSION = "leadpoet.validator_weight_publication_journal.v2"
JOURNAL_SCHEMA_VERSION = "leadpoet.validator_weight_publication_journal.v3"
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_EXTRINSIC_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
_SIGNATURE_RE = re.compile(r"^[0-9a-f]{128}$")


class WeightPublicationJournalV2Error(RuntimeError):
    """The durable publication journal is missing, corrupt, or conflicting."""


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_signature_result(
    value: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    event_hash: str,
    chain_profile: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_fields = {
        "schema_version",
        "authorization_hash",
        "validator_hotkey",
        "signature",
        "extrinsic_hash",
        "authorization",
        "receipt",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise WeightPublicationJournalV2Error(
            "weight extrinsic signature result fields are invalid"
        )
    authorization = validate_weight_extrinsic_authorization_v2(
        value["authorization"], profile=chain_profile
    )
    result = bundle["weight_result"]
    computed = [
        receipt
        for receipt in bundle["receipt_graph"]["receipts"]
        if receipt.get("purpose") == "validator.weights.computed.v2"
    ]
    if (
        len(computed) != 1
        or value.get("schema_version")
        != "leadpoet.weight_extrinsic_signature.v2"
        or value.get("authorization_hash") != authorization["authorization_hash"]
        or value.get("validator_hotkey") != bundle["validator_hotkey"]
        or authorization["validator_hotkey"] != bundle["validator_hotkey"]
        or authorization["weight_receipt_hash"] != computed[0]["receipt_hash"]
        or authorization["weight_submission_event_hash"] != event_hash
        or authorization["weights_hash"] != result["weights_hash"]
        or authorization["sparse_uids"] != result["sparse_uids"]
        or authorization["sparse_weights_u16"] != result["sparse_weights_u16"]
        or not _SIGNATURE_RE.fullmatch(str(value.get("signature") or ""))
        or not _EXTRINSIC_HASH_RE.fullmatch(
            str(value.get("extrinsic_hash") or "")
        )
        or not isinstance(value.get("receipt"), Mapping)
    ):
        raise WeightPublicationJournalV2Error(
            "weight extrinsic signature result differs from the publication"
        )
    return dict(value)


def validate_publication_journal_v2(
    value: Mapping[str, Any],
    *,
    chain_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    base_fields = {
        "schema_version",
        "state",
        "revision",
        "weight_authorization_id",
        "published_bundle",
        "publication",
        "extrinsic_signature_results",
        "updated_at",
        "journal_hash",
    }
    schema_version = value.get("schema_version") if isinstance(value, Mapping) else None
    fields = (
        base_fields
        if schema_version == LEGACY_JOURNAL_SCHEMA_VERSION
        else base_fields | {"epoch_evidence"}
    )
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WeightPublicationJournalV2Error("publication journal fields are invalid")
    if schema_version not in {
        LEGACY_JOURNAL_SCHEMA_VERSION,
        JOURNAL_SCHEMA_VERSION,
    }:
        raise WeightPublicationJournalV2Error("publication journal schema is invalid")
    if value.get("state") not in {"prepared", "published", "signed"}:
        raise WeightPublicationJournalV2Error("publication journal state is invalid")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise WeightPublicationJournalV2Error("publication journal revision is invalid")
    authorization_id = str(value.get("weight_authorization_id") or "").lower()
    if not _HASH_RE.fullmatch(authorization_id):
        raise WeightPublicationJournalV2Error(
            "publication journal authorization id is invalid"
        )
    bundle = value.get("published_bundle")
    if not isinstance(bundle, Mapping):
        raise WeightPublicationJournalV2Error("publication journal bundle is missing")
    verified = validate_published_weight_bundle_v2(bundle)
    try:
        epoch_evidence = validate_stateful_epoch_evidence_v1(
            (
                None
                if schema_version == LEGACY_JOURNAL_SCHEMA_VERSION
                else value.get("epoch_evidence")
            ),
            published_bundle=bundle,
        )
    except HostWeightAuthorityV2Error as exc:
        raise WeightPublicationJournalV2Error(
            "publication journal epoch evidence is invalid"
        ) from exc
    publication = value.get("publication")
    signatures = value.get("extrinsic_signature_results")
    if not isinstance(signatures, list):
        raise WeightPublicationJournalV2Error(
            "publication journal signature results are invalid"
        )
    if publication is None:
        if value["state"] != "prepared" or signatures:
            raise WeightPublicationJournalV2Error(
                "unpublished journal cannot contain signed chain state"
            )
        event_hash = None
    else:
        if not isinstance(publication, Mapping):
            raise WeightPublicationJournalV2Error(
                "publication journal gateway acknowledgment is invalid"
            )
        expected_publication_fields = {
            "success",
            "epoch_id",
            "weights_count",
            "weights_hash",
            "weight_receipt_hash",
            "weight_submission_event_hash",
            "message",
        }
        computed = [
            receipt
            for receipt in bundle["receipt_graph"]["receipts"]
            if receipt.get("purpose") == "validator.weights.computed.v2"
        ]
        event_hash = str(publication.get("weight_submission_event_hash") or "")
        if (
            set(publication) != expected_publication_fields
            or publication.get("success") is not True
            or int(publication.get("epoch_id", -1)) != verified["epoch_id"]
            or int(publication.get("weights_count", -1)) != len(verified["uids"])
            or publication.get("weights_hash") != verified["weights_hash"]
            or len(computed) != 1
            or publication.get("weight_receipt_hash")
            != computed[0]["receipt_hash"]
            or not _HASH_RE.fullmatch(event_hash)
        ):
            raise WeightPublicationJournalV2Error(
                "publication journal gateway acknowledgment is invalid"
            )
        expected_state = "signed" if signatures else "published"
        if value["state"] != expected_state:
            raise WeightPublicationJournalV2Error(
                "publication journal state does not match chain evidence"
            )
    profile = chain_profile or load_chain_signing_profile()
    normalized_signatures = []
    seen_authorizations = set()
    seen_extrinsics = set()
    for item in signatures:
        normalized = _validate_signature_result(
            item,
            bundle=bundle,
            event_hash=str(event_hash),
            chain_profile=profile,
        )
        authorization_hash = normalized["authorization_hash"]
        extrinsic_hash = normalized["extrinsic_hash"]
        if (
            authorization_hash in seen_authorizations
            or extrinsic_hash in seen_extrinsics
        ):
            raise WeightPublicationJournalV2Error(
                "publication journal contains duplicate signed extrinsics"
            )
        seen_authorizations.add(authorization_hash)
        seen_extrinsics.add(extrinsic_hash)
        normalized_signatures.append(normalized)
    body = {key: value[key] for key in fields if key != "journal_hash"}
    if value.get("journal_hash") != sha256_json(body):
        raise WeightPublicationJournalV2Error("publication journal hash is invalid")
    return {
        **body,
        "weight_authorization_id": authorization_id,
        "published_bundle": dict(bundle),
        "publication": dict(publication) if isinstance(publication, Mapping) else None,
        "extrinsic_signature_results": normalized_signatures,
        **(
            {"epoch_evidence": epoch_evidence}
            if schema_version == JOURNAL_SCHEMA_VERSION
            else {}
        ),
        "journal_hash": value["journal_hash"],
    }


class AuthoritativeWeightPublicationJournalV2:
    """Atomically retain exactly one unfinished V2 publication."""

    def __init__(
        self,
        path: Path,
        *,
        chain_profile: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self._chain_profile = chain_profile
        self._lock = threading.RLock()

    def load(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return None
            try:
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise WeightPublicationJournalV2Error(
                    "publication journal cannot be read"
                ) from exc
            return validate_publication_journal_v2(
                value, chain_profile=self._chain_profile
            )

    def record_prepared(self, prepared: Mapping[str, Any]) -> Dict[str, Any]:
        required = {"weight_authorization_id", "published_bundle"}
        if not isinstance(prepared, Mapping) or not required <= set(prepared):
            raise WeightPublicationJournalV2Error(
                "prepared publication journal input is incomplete"
            )
        with self._lock:
            existing = self.load()
            body = {
                "schema_version": JOURNAL_SCHEMA_VERSION,
                "state": "prepared",
                "revision": 0,
                "weight_authorization_id": str(
                    prepared["weight_authorization_id"]
                ),
                "published_bundle": dict(prepared["published_bundle"]),
                "epoch_evidence": (
                    dict(prepared["epoch_evidence"])
                    if isinstance(prepared.get("epoch_evidence"), Mapping)
                    else None
                ),
                "publication": None,
                "extrinsic_signature_results": [],
                "updated_at": _timestamp(),
            }
            candidate = {**body, "journal_hash": sha256_json(body)}
            validated = validate_publication_journal_v2(
                candidate, chain_profile=self._chain_profile
            )
            if existing is not None:
                if (
                    existing["weight_authorization_id"]
                    == validated["weight_authorization_id"]
                    and existing["published_bundle"]
                    == validated["published_bundle"]
                ):
                    return existing
                raise WeightPublicationJournalV2Error(
                    "another authoritative publication is unfinished"
                )
            self._write(validated)
            return validated

    def record_published(self, publication: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = self.load()
            if current is None:
                raise WeightPublicationJournalV2Error(
                    "cannot publish without a prepared journal"
                )
            if current["publication"] is not None:
                if current["publication"] == dict(publication):
                    return current
                raise WeightPublicationJournalV2Error(
                    "gateway publication acknowledgment conflicts"
                )
            return self._replace(
                current,
                state="published",
                publication=dict(publication),
            )

    def replace_authorization(self, authorization_id: str) -> Dict[str, Any]:
        with self._lock:
            current = self.load()
            if current is None:
                raise WeightPublicationJournalV2Error(
                    "cannot replace a missing weight authorization"
                )
            return self._replace(
                current,
                weight_authorization_id=str(authorization_id),
            )

    def record_signed(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            current = self.load()
            if current is None or current["publication"] is None:
                raise WeightPublicationJournalV2Error(
                    "signed extrinsic has no durable gateway publication"
                )
            normalized = dict(result)
            existing = list(current["extrinsic_signature_results"])
            if normalized in existing:
                return current
            return self._replace(
                current,
                state="signed",
                extrinsic_signature_results=existing + [normalized],
            )

    def clear(self, *, expected_event_hash: str) -> None:
        with self._lock:
            current = self.load()
            if current is None:
                return
            observed = str(
                (current.get("publication") or {}).get(
                    "weight_submission_event_hash"
                )
                or ""
            )
            if observed != str(expected_event_hash):
                raise WeightPublicationJournalV2Error(
                    "refusing to clear another weight publication"
                )
            try:
                self.path.unlink()
                self._fsync_directory()
            except OSError as exc:
                raise WeightPublicationJournalV2Error(
                    "publication journal could not be cleared"
                ) from exc

    def quarantine(self, *, expected_epoch: int, reason: str) -> Path:
        """Atomically remove a closed-epoch journal from the active slot.

        The exact validated journal remains on disk for reconciliation and
        audit. Quarantine never claims that a signed extrinsic was absent.
        """

        normalized_reason = str(reason or "").strip().lower()
        if not re.fullmatch(r"[a-z0-9_]{1,64}", normalized_reason):
            raise WeightPublicationJournalV2Error(
                "publication journal quarantine reason is invalid"
            )
        with self._lock:
            current = self.load()
            if current is None:
                raise WeightPublicationJournalV2Error(
                    "cannot quarantine a missing publication journal"
                )
            epoch_id = int(current["published_bundle"]["weight_result"]["epoch_id"])
            if epoch_id != int(expected_epoch):
                raise WeightPublicationJournalV2Error(
                    "refusing to quarantine another publication epoch"
                )
            suffix = str(current["journal_hash"]).removeprefix("sha256:")[:16]
            target = self.path.with_name(
                "%s.quarantined.%d.%s.%s"
                % (self.path.name, epoch_id, normalized_reason, suffix)
            )
            try:
                if target.exists():
                    if target.read_bytes() != self.path.read_bytes():
                        raise WeightPublicationJournalV2Error(
                            "publication journal quarantine target conflicts"
                        )
                    self.path.unlink()
                else:
                    os.replace(str(self.path), str(target))
                os.chmod(target, 0o600)
                self._fsync_directory()
            except WeightPublicationJournalV2Error:
                raise
            except OSError as exc:
                raise WeightPublicationJournalV2Error(
                    "publication journal could not be quarantined"
                ) from exc
            return target

    def _replace(self, current: Mapping[str, Any], **changes: Any) -> Dict[str, Any]:
        body = {
            key: current[key]
            for key in current
            if key != "journal_hash"
        }
        body.update(changes)
        body["revision"] = int(current["revision"]) + 1
        body["updated_at"] = _timestamp()
        candidate = {**body, "journal_hash": sha256_json(body)}
        validated = validate_publication_journal_v2(
            candidate, chain_profile=self._chain_profile
        )
        self._write(validated)
        return validated

    def _write(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".%s." % self.path.name,
            dir=str(self.path.parent),
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, 0o600)
            payload = json.dumps(
                dict(value), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                descriptor = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(self.path))
            os.chmod(self.path, 0o600)
            self._fsync_directory()
        except OSError as exc:
            raise WeightPublicationJournalV2Error(
                "publication journal atomic write failed"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _fsync_directory(self) -> None:
        descriptor = os.open(str(self.path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

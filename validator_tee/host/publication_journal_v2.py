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
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import (
    CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
    RECEIPT_GRAPH_SCHEMA_VERSION,
    sha256_json,
    validate_boot_identity,
)
from leadpoet_canonical.hotkey_authority_v2 import (
    validate_chain_signing_profile,
    validate_weight_extrinsic_authorization_v2,
)
from leadpoet_canonical.compact_weight_authority_v2 import (
    validate_compact_weight_submission_shape_v2,
)
from leadpoet_canonical.compact_auditor_authority_v2 import (
    verify_compact_weight_submission_v2,
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
EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION = (
    "leadpoet.validator_weight_publication_journal.v3"
)
JOURNAL_SCHEMA_VERSION = "leadpoet.validator_weight_publication_journal.v4"
COMPACT_JOURNAL_SCHEMA_VERSION = (
    "leadpoet.validator_weight_publication_journal.v5"
)
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
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
    weight_receipt_hash: str,
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
    if (
        value.get("schema_version")
        != "leadpoet.weight_extrinsic_signature.v2"
        or value.get("authorization_hash") != authorization["authorization_hash"]
        or value.get("validator_hotkey") != bundle["validator_hotkey"]
        or authorization["validator_hotkey"] != bundle["validator_hotkey"]
        or authorization["weight_receipt_hash"] != weight_receipt_hash
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
    if (
        isinstance(value, Mapping)
        and value.get("schema_version") == COMPACT_JOURNAL_SCHEMA_VERSION
    ):
        return _validate_compact_publication_journal_v2(
            value,
            chain_profile=chain_profile,
        )
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
    if schema_version == LEGACY_JOURNAL_SCHEMA_VERSION:
        fields = base_fields
    elif schema_version == EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION:
        fields = base_fields | {"epoch_evidence"}
    else:
        fields = base_fields | {
            "epoch_evidence",
            "finalization_scan_generation",
            "finalization_scan_id",
        }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise WeightPublicationJournalV2Error("publication journal fields are invalid")
    if schema_version not in {
        LEGACY_JOURNAL_SCHEMA_VERSION,
        EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION,
        JOURNAL_SCHEMA_VERSION,
    }:
        raise WeightPublicationJournalV2Error("publication journal schema is invalid")
    if value.get("state") not in {"prepared", "published", "signed"}:
        raise WeightPublicationJournalV2Error("publication journal state is invalid")
    revision = value.get("revision")
    if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
        raise WeightPublicationJournalV2Error("publication journal revision is invalid")
    if schema_version == JOURNAL_SCHEMA_VERSION:
        scan_generation = value.get("finalization_scan_generation")
        scan_id = value.get("finalization_scan_id")
        if (
            not isinstance(scan_generation, int)
            or isinstance(scan_generation, bool)
            or scan_generation < 0
            or (
                scan_id is not None
                and not _HASH_RE.fullmatch(str(scan_id))
            )
            or (scan_generation == 0 and scan_id is not None)
            or (scan_generation > 0 and scan_id is None)
        ):
            raise WeightPublicationJournalV2Error(
                "publication journal finalization scan state is invalid"
            )
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
        event_hash = str(publication.get("weight_submission_event_hash") or "")
        if (
            set(publication) != expected_publication_fields
            or publication.get("success") is not True
            or int(publication.get("epoch_id", -1)) != verified["epoch_id"]
            or int(publication.get("weights_count", -1)) != len(verified["uids"])
            or publication.get("weights_hash") != verified["weights_hash"]
            or publication.get("weight_receipt_hash")
            != verified["weight_receipt_hash"]
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
            weight_receipt_hash=verified["weight_receipt_hash"],
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
            if schema_version
            in {
                EPOCH_EVIDENCE_JOURNAL_SCHEMA_VERSION,
                JOURNAL_SCHEMA_VERSION,
            }
            else {}
        ),
        "journal_hash": value["journal_hash"],
    }


def _validate_compact_publication_journal_v2(
    value: Mapping[str, Any],
    *,
    chain_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "state",
        "revision",
        "weight_authorization_id",
        "compact_submission",
        "publication",
        "extrinsic_signature_results",
        "finalization_scan_generation",
        "finalization_scan_id",
        "updated_at",
        "journal_hash",
    }
    if set(value) != fields:
        raise WeightPublicationJournalV2Error(
            "compact publication journal fields are invalid"
        )
    if value.get("state") not in {"prepared", "published", "signed"}:
        raise WeightPublicationJournalV2Error(
            "compact publication journal state is invalid"
        )
    revision = value.get("revision")
    scan_generation = value.get("finalization_scan_generation")
    scan_id = value.get("finalization_scan_id")
    if (
        not isinstance(revision, int)
        or isinstance(revision, bool)
        or revision < 0
        or not isinstance(scan_generation, int)
        or isinstance(scan_generation, bool)
        or scan_generation < 0
        or (scan_id is not None and not _HASH_RE.fullmatch(str(scan_id)))
        or (scan_generation == 0 and scan_id is not None)
        or (scan_generation > 0 and scan_id is None)
    ):
        raise WeightPublicationJournalV2Error(
            "compact publication journal counters are invalid"
        )
    authorization_id = str(value.get("weight_authorization_id") or "").lower()
    if not _HASH_RE.fullmatch(authorization_id):
        raise WeightPublicationJournalV2Error(
            "compact publication authorization id is invalid"
        )
    try:
        compact = validate_compact_weight_submission_shape_v2(
            value.get("compact_submission")
        )
    except Exception as exc:
        raise WeightPublicationJournalV2Error(
            "compact publication authority is invalid"
        ) from exc
    result = compact["weight_result"]
    weight_receipt_hash = str(
        compact["validator_receipt_delta"]["root_receipt_hash"]
    )
    publication = value.get("publication")
    signatures = value.get("extrinsic_signature_results")
    if not isinstance(signatures, list):
        raise WeightPublicationJournalV2Error(
            "compact publication signatures are invalid"
        )
    if publication is None:
        if value["state"] != "prepared" or signatures:
            raise WeightPublicationJournalV2Error(
                "unpublished compact journal contains chain state"
            )
        event_hash = None
    else:
        if not isinstance(publication, Mapping):
            raise WeightPublicationJournalV2Error(
                "compact publication acknowledgment is invalid"
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
        event_hash = str(publication.get("weight_submission_event_hash") or "")
        if (
            set(publication) != expected_publication_fields
            or publication.get("success") is not True
            or int(publication.get("epoch_id", -1))
            != int(result["epoch_id"])
            or int(publication.get("weights_count", -1))
            != len(result["sparse_uids"])
            or publication.get("weights_hash") != result["weights_hash"]
            or publication.get("weight_receipt_hash") != weight_receipt_hash
            or not _HASH_RE.fullmatch(event_hash)
        ):
            raise WeightPublicationJournalV2Error(
                "compact publication acknowledgment is invalid"
            )
        expected_state = "signed" if signatures else "published"
        if value["state"] != expected_state:
            raise WeightPublicationJournalV2Error(
                "compact publication state differs from chain evidence"
            )
    profile = chain_profile or load_chain_signing_profile()
    normalized_signatures = []
    seen_authorizations = set()
    seen_extrinsics = set()
    for item in signatures:
        normalized = _validate_signature_result(
            item,
            bundle=compact,
            event_hash=str(event_hash),
            weight_receipt_hash=weight_receipt_hash,
            chain_profile=profile,
        )
        if (
            normalized["authorization_hash"] in seen_authorizations
            or normalized["extrinsic_hash"] in seen_extrinsics
        ):
            raise WeightPublicationJournalV2Error(
                "compact journal contains duplicate signed extrinsics"
            )
        seen_authorizations.add(normalized["authorization_hash"])
        seen_extrinsics.add(normalized["extrinsic_hash"])
        normalized_signatures.append(normalized)
    body = {key: value[key] for key in fields if key != "journal_hash"}
    if value.get("journal_hash") != sha256_json(body):
        raise WeightPublicationJournalV2Error(
            "compact publication journal hash is invalid"
        )
    return {
        **body,
        "weight_authorization_id": authorization_id,
        "compact_submission": compact,
        "publication": (
            dict(publication) if isinstance(publication, Mapping) else None
        ),
        "extrinsic_signature_results": normalized_signatures,
        "journal_hash": value["journal_hash"],
    }


def publication_journal_release_requirements_v2(
    journal: Optional[Mapping[str, Any]],
    *,
    expected_lineage_id: Optional[str] = None,
    expected_validator_hotkey: Optional[str] = None,
    boot_verifier: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    chain_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return every approved release needed to recover one durable journal.

    The complete journal and its embedded weight authority are validated before
    any release identity is extracted. When supplied, ``boot_verifier`` is
    applied once to every distinct validator, upstream, disclosed, and
    checkpoint-issuer boot identity.
    """

    if journal is None:
        return {"journal_hash": None, "required_commits": []}
    if not isinstance(journal, Mapping):
        raise WeightPublicationJournalV2Error(
            "publication journal release requirements are invalid"
        )
    lineage_id = (
        expected_lineage_id if isinstance(expected_lineage_id, str) else ""
    )
    validator_hotkey = str(expected_validator_hotkey or "")
    if not _HASH_RE.fullmatch(lineage_id):
        raise WeightPublicationJournalV2Error(
            "publication journal expected lineage is unavailable or invalid"
        )
    if (
        not 1 <= len(validator_hotkey) <= 128
        or any(character.isspace() for character in validator_hotkey)
    ):
        raise WeightPublicationJournalV2Error(
            "publication journal expected validator hotkey is unavailable or invalid"
        )
    if not isinstance(chain_profile, Mapping):
        raise WeightPublicationJournalV2Error(
            "publication journal expected chain signing profile is unavailable"
        )
    try:
        profile = validate_chain_signing_profile(chain_profile)
    except Exception as exc:
        raise WeightPublicationJournalV2Error(
            "publication journal expected chain signing profile is invalid"
        ) from exc
    expected_chain = str(profile.get("chain_endpoint") or "")
    if not expected_chain:
        raise WeightPublicationJournalV2Error(
            "publication journal expected chain is unavailable"
        )
    try:
        normalized = validate_publication_journal_v2(
            journal,
            chain_profile=profile,
        )
    except WeightPublicationJournalV2Error:
        raise
    except Exception as exc:
        raise WeightPublicationJournalV2Error(
            "publication journal release requirements are invalid"
        ) from exc

    verified_identities: Dict[str, Dict[str, Any]] = {}

    def verify_boot(identity: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            validate_boot_identity(identity)
            identity_hash = str(identity["boot_identity_hash"])
            commit = str(identity["commit_sha"])
            if not _COMMIT_RE.fullmatch(commit):
                raise ValueError("boot commit is invalid")
            previous = verified_identities.get(identity_hash)
            if previous is not None:
                if previous != dict(identity):
                    raise ValueError("boot identity conflicts")
                return dict(previous)
            if boot_verifier is not None:
                verified = boot_verifier(identity)
                if not isinstance(verified, Mapping):
                    raise ValueError("boot verifier returned no evidence")
            stored = dict(identity)
            verified_identities[identity_hash] = stored
            return dict(stored)
        except Exception as exc:
            raise WeightPublicationJournalV2Error(
                "publication journal release boot is invalid or unapproved"
            ) from exc

    def collect_boots(value: Any, label: str) -> None:
        if not isinstance(value, list):
            raise WeightPublicationJournalV2Error(
                "publication journal %s boots are invalid" % label
            )
        for identity in value:
            if not isinstance(identity, Mapping):
                raise WeightPublicationJournalV2Error(
                    "publication journal %s boot is invalid" % label
                )
            verify_boot(identity)

    def collect_proof(value: Any, label: str) -> None:
        certificate = value.get("certificate") if isinstance(value, Mapping) else None
        issuer = (
            certificate.get("issuer_boot_identity")
            if isinstance(certificate, Mapping)
            else None
        )
        disclosed = (
            value.get("disclosed_boot_identities")
            if isinstance(value, Mapping)
            else None
        )
        if not isinstance(issuer, Mapping):
            raise WeightPublicationJournalV2Error(
                "publication journal %s checkpoint issuer is invalid" % label
            )
        verify_boot(issuer)
        collect_boots(disclosed, "%s disclosed" % label)

    try:
        if normalized["schema_version"] == COMPACT_JOURNAL_SCHEMA_VERSION:
            compact = normalized["compact_submission"]
            verified_submission = verify_compact_weight_submission_v2(
                compact,
                expected_lineage_id=lineage_id,
                expected_chain=expected_chain,
                identity_cache=None,
                boot_verifier=verify_boot,
            )
            if verified_submission.get("validator_hotkey") != validator_hotkey:
                raise WeightPublicationJournalV2Error(
                    "compact publication journal uses another validator hotkey"
                )
            delta = compact.get("validator_receipt_delta")
            if not isinstance(delta, Mapping):
                raise WeightPublicationJournalV2Error(
                    "compact publication validator delta is invalid"
                )
            collect_boots(delta.get("boot_identities"), "validator delta")
            proofs = compact.get("upstream_ancestry_proofs")
            if not isinstance(proofs, Mapping):
                raise WeightPublicationJournalV2Error(
                    "compact publication upstream proofs are invalid"
                )
            for category in sorted(proofs):
                collect_proof(proofs[category], "upstream %s" % category)
            collect_proof(compact.get("validator_ancestry_proof"), "validator")
        else:
            bundle = normalized["published_bundle"]
            validate_published_weight_bundle_v2(
                bundle,
                boot_attestation_verifier=verify_boot,
                require_boot_attestation_verification=True,
            )
            graph = bundle.get("receipt_graph")
            if not isinstance(graph, Mapping):
                raise WeightPublicationJournalV2Error(
                    "publication journal receipt graph is invalid"
                )
            graph_schema = graph.get("schema_version")
            if graph_schema not in {
                RECEIPT_GRAPH_SCHEMA_VERSION,
                *CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS,
            }:
                raise WeightPublicationJournalV2Error(
                    "publication journal receipt graph schema is invalid"
                )
            if (
                graph_schema in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS
                and graph.get("ancestry_lineage_id") != lineage_id
            ):
                raise WeightPublicationJournalV2Error(
                    "publication journal receipt graph uses another lineage"
                )
            if bundle.get("validator_hotkey") != validator_hotkey:
                raise WeightPublicationJournalV2Error(
                    "publication journal uses another validator hotkey"
                )
            collect_boots(graph.get("boot_identities"), "receipt graph")
            if graph_schema in CHECKPOINTED_RECEIPT_GRAPH_SCHEMA_VERSIONS:
                collect_proof(graph.get("ancestry_proof"), "receipt graph")
    except WeightPublicationJournalV2Error:
        raise
    except Exception as exc:
        raise WeightPublicationJournalV2Error(
            "publication journal release requirements are invalid"
        ) from exc

    return {
        "journal_hash": normalized["journal_hash"],
        "required_commits": sorted(
            {identity["commit_sha"] for identity in verified_identities.values()}
        ),
    }


class AuthoritativeWeightPublicationJournalV2:
    """Atomically retain one active or most-recent finalized V2 publication."""

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
        has_bundle = isinstance(
            prepared.get("published_bundle")
            if isinstance(prepared, Mapping)
            else None,
            Mapping,
        )
        has_compact = isinstance(
            prepared.get("compact_submission")
            if isinstance(prepared, Mapping)
            else None,
            Mapping,
        )
        if (
            not isinstance(prepared, Mapping)
            or "weight_authorization_id" not in prepared
            or has_bundle == has_compact
        ):
            raise WeightPublicationJournalV2Error(
                "prepared publication journal input is incomplete"
            )
        with self._lock:
            existing = self.load()
            if has_compact:
                body = {
                    "schema_version": COMPACT_JOURNAL_SCHEMA_VERSION,
                    "state": "prepared",
                    "revision": 0,
                    "weight_authorization_id": str(
                        prepared["weight_authorization_id"]
                    ),
                    "compact_submission": dict(
                        prepared["compact_submission"]
                    ),
                    "finalization_scan_generation": 0,
                    "finalization_scan_id": None,
                    "publication": None,
                    "extrinsic_signature_results": [],
                    "updated_at": _timestamp(),
                }
            else:
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
                    "finalization_scan_generation": 0,
                    "finalization_scan_id": None,
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
                    and existing.get("published_bundle")
                    == validated.get("published_bundle")
                    and existing.get("compact_submission")
                    == validated.get("compact_submission")
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

    def reserve_finalization_scan(self) -> str:
        """Durably allocate one unique finalized-chain scan identity."""

        with self._lock:
            current = self.load()
            if (
                current is None
                or current.get("publication") is None
                or not current.get("extrinsic_signature_results")
            ):
                raise WeightPublicationJournalV2Error(
                    "finalization scan requires a signed publication"
                )
            generation = int(
                current.get("finalization_scan_generation") or 0
            ) + 1
            scan_id = sha256_json(
                {
                    "schema_version": (
                        "leadpoet.validator_weight_finalization_scan.v1"
                    ),
                    "weight_authorization_id": current[
                        "weight_authorization_id"
                    ],
                    "weight_submission_event_hash": current["publication"][
                        "weight_submission_event_hash"
                    ],
                    "generation": generation,
                    "prior_journal_hash": current["journal_hash"],
                }
            )
            changes = {
                "schema_version": current["schema_version"],
                "finalization_scan_generation": generation,
                "finalization_scan_id": scan_id,
            }
            if current["schema_version"] != COMPACT_JOURNAL_SCHEMA_VERSION:
                changes["schema_version"] = JOURNAL_SCHEMA_VERSION
                changes["epoch_evidence"] = current.get("epoch_evidence")
            updated = self._replace(current, **changes)
            if updated["finalization_scan_id"] != scan_id:
                raise WeightPublicationJournalV2Error(
                    "finalization scan reservation did not persist"
                )
            return scan_id

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
            authority = current.get("published_bundle") or current.get(
                "compact_submission"
            )
            if not isinstance(authority, Mapping):
                raise WeightPublicationJournalV2Error(
                    "publication journal authority is unavailable"
                )
            epoch_id = int(authority["weight_result"]["epoch_id"])
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

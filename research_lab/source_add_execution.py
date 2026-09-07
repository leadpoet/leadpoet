"""SOURCE_ADD intake validation and anti-spam admission."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .canonical import sha256_json
from .source_add import (
    SourceAddAdapterManifest,
    validate_source_add_adapter_manifest,
)
from .source_add_identity import source_identity_hash

# §8 launch defaults — env-tunable (T), inlined here as the code defaults.
DEFAULT_MAX_CONCURRENT_SUBMISSIONS_PER_HOTKEY = 3
DEFAULT_MAX_SUBMISSIONS_PER_DAY_PER_HOTKEY = 5
DEFAULT_MAX_SUBMISSIONS_PER_30D_PER_HOTKEY = 10


class SourceAddFunnelStage(str, Enum):
    SUBMITTED = "submitted"
    MANIFEST_VALIDATED = "manifest_validated"
    PROVENANCE_QUEUED = "provenance_queued"
    PROVENANCE_PRECHECK_PASSED = "provenance_precheck_passed"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"
    REJECTED_PRECHECK = "rejected_precheck"
    FUNCTIONAL_PROBE_QUEUED = "functional_probe_queued"
    AWAITING_OPERATOR_CREDENTIAL = "awaiting_operator_credential"
    FUNCTIONAL_PROBE_RETRYABLE = "functional_probe_retryable"
    FUNCTIONAL_PROBE_PASSED = "functional_probe_passed"
    FUNCTIONAL_PROBE_FAILED = "functional_probe_failed"
    LEG1_QUEUED = "leg1_queued"
    LEG1_CREATED = "leg1_created"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SourceAddRejectionReason(str, Enum):
    MANIFEST_INVALID = "manifest_invalid"
    DUPLICATE_SOURCE = "duplicate_source"
    HOTKEY_CONCURRENT_CAP = "hotkey_concurrent_cap"
    HOTKEY_DAY_CAP = "hotkey_day_cap"
    HOTKEY_30D_CAP = "hotkey_30d_cap"
    CREDENTIAL_INVALID = "credential_invalid"


@dataclass(frozen=True)
class SourceAddSubmissionRecord:
    submission_id: str
    adapter_id: str
    miner_hotkey: str
    manifest: SourceAddAdapterManifest
    stage: str = SourceAddFunnelStage.SUBMITTED.value
    stage_history: tuple[str, ...] = (SourceAddFunnelStage.SUBMITTED.value,)
    credential_envelope: dict[str, str] = field(default_factory=dict)
    source_brief: str = ""
    submitted_at: str = ""
    rejection_reasons: tuple[str, ...] = ()
    rejection_stage: str = ""
    trial_diagnostics: dict[str, Any] = field(default_factory=dict)
    measured_trial_yield: float = -1.0
    acceptance_human_gate_passed: bool = False
    precheck_status: str = ""
    precheck_doc: dict[str, Any] = field(default_factory=dict)
    source_identity_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = self.manifest.to_dict()
        data["stage_history"] = list(self.stage_history)
        data["rejection_reasons"] = list(self.rejection_reasons)
        return data


def normalize_base_domain(domain: str) -> str:
    lowered = str(domain or "").strip().lower()
    lowered = re.sub(r"^[a-z]+://", "", lowered).split("/", 1)[0].split(":", 1)[0]
    if lowered.startswith("www."):
        lowered = lowered[4:]
    return lowered


def _domains_overlap(candidate: Sequence[str], existing: Sequence[str]) -> bool:
    normalized_existing = {normalize_base_domain(item) for item in existing if normalize_base_domain(item)}
    return any(normalize_base_domain(item) in normalized_existing for item in candidate)


def intake_source_add_submission(
    manifest_doc: Mapping[str, Any],
    *,
    miner_hotkey: str,
    raw_credential: str = "",
    source_brief: str = "",
    submitted_at: str = "",
    existing_catalog_domains: Sequence[str] = (),
    existing_source_identity_hashes: Sequence[str] = (),
    source_identity_ref: str = "",
    open_submission_count_for_hotkey: int = 0,
    submissions_last_day_for_hotkey: int = 0,
    submissions_last_30d_for_hotkey: int = 0,
    max_concurrent_per_hotkey: int = DEFAULT_MAX_CONCURRENT_SUBMISSIONS_PER_HOTKEY,
    max_per_day_per_hotkey: int = DEFAULT_MAX_SUBMISSIONS_PER_DAY_PER_HOTKEY,
    max_per_30d_per_hotkey: int = DEFAULT_MAX_SUBMISSIONS_PER_30D_PER_HOTKEY,
    kms_encrypt: Callable[[str, str, str], Mapping[str, str]] | None = None,
) -> tuple[SourceAddSubmissionRecord | None, list[str]]:
    """Validate + admit one submission. Returns (record, errors).

    Public SOURCE_ADD intake never accepts credentials. Authenticated sources
    wait for an operator-scoped attested credential after provenance review.
    """

    errors: list[str] = []
    try:
        manifest = SourceAddAdapterManifest.from_mapping(manifest_doc)
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"{SourceAddRejectionReason.MANIFEST_INVALID.value}: {exc}"]
    manifest_errors = validate_source_add_adapter_manifest(manifest)
    if manifest_errors:
        return None, [f"{SourceAddRejectionReason.MANIFEST_INVALID.value}: {error}" for error in manifest_errors]

    identity_ref = str(source_identity_ref or "").strip() or source_identity_hash(
        declared_base_domains=manifest.declared_base_domains
    )

    # Anti-spam before anything costs money.
    existing_identity_refs = {str(item or "").strip() for item in existing_source_identity_hashes if str(item or "").strip()}
    if (identity_ref and identity_ref in existing_identity_refs) or _domains_overlap(
        manifest.declared_base_domains, existing_catalog_domains
    ):
        errors.append(SourceAddRejectionReason.DUPLICATE_SOURCE.value)
    if open_submission_count_for_hotkey >= max(1, int(max_concurrent_per_hotkey)):
        errors.append(SourceAddRejectionReason.HOTKEY_CONCURRENT_CAP.value)
    if submissions_last_day_for_hotkey >= max(1, int(max_per_day_per_hotkey)):
        errors.append(SourceAddRejectionReason.HOTKEY_DAY_CAP.value)
    if submissions_last_30d_for_hotkey >= max(1, int(max_per_30d_per_hotkey)):
        errors.append(SourceAddRejectionReason.HOTKEY_30D_CAP.value)
    if errors:
        return None, errors

    credential_envelope: dict[str, str] = {}
    if raw_credential or manifest.credential_policy != "no_credentials" or manifest.credential_ref:
        return None, [
            f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: miner credentials are not accepted"
        ]

    submission_id = "source_add_submission:" + sha256_json(
        {"adapter_id": manifest.adapter_id, "miner_hotkey": str(miner_hotkey), "bundle": manifest.code_bundle_hash}
    ).split(":", 1)[1][:16]
    record = SourceAddSubmissionRecord(
        submission_id=submission_id,
        adapter_id=manifest.adapter_id,
        miner_hotkey=str(miner_hotkey),
        manifest=manifest,
        stage=SourceAddFunnelStage.MANIFEST_VALIDATED.value,
        stage_history=(
            SourceAddFunnelStage.SUBMITTED.value,
            SourceAddFunnelStage.MANIFEST_VALIDATED.value,
        ),
        credential_envelope=credential_envelope,
        source_brief=str(source_brief)[:2000],
        submitted_at=str(submitted_at),
        source_identity_hash=identity_ref,
    )
    return record, []

"""SOURCE_ADD execution funnel (sourceexperiments.md W5).

P1.5 (``source_add.py``) defined the contracts without execution. This module
executes them behind ``RESEARCH_LAB_SOURCE_ADD_ENABLED``: submission intake
(anti-spam caps, catalog dedupe, KMS credential envelope), the cheapest-first
funnel (manifest validation → static scan → LLM review → sandboxed metered
trial), category-scoped trial yield, and the acceptance decision that feeds
the leg-1 reward (``source_add_rewards.py``).

Execution boundaries preserved from the plan:
- adapter code runs only in the sandbox, with no credentials inside and all
  hosts rewritten to the evidence proxy (upstream allowlist = that adapter's
  registry entry only, metered on the miner's key);
- trials run against lab-scheduled trial ICP sets, never the private
  benchmark window;
- loops never execute SOURCE_ADD — a loop emits a suggestion doc into the
  submission queue instead.

Side effects (KMS, sandbox container, evidence classification) are injected
callables so the funnel logic is testable and the gateway wires the real
implementations (key_vault KMS pattern, private_runtime container, existing
verification stack).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

from .canonical import sha256_json
from .source_add import (
    SourceAddAdapterManifest,
    SourceAddTrialOutputRecord,
    validate_source_add_adapter_manifest,
    validate_source_add_trial_output,
)

# §8 launch defaults — env-tunable (T), inlined here as the code defaults.
DEFAULT_ACCEPTANCE_FLOOR_YIELD = 0.10
DEFAULT_MAX_CONCURRENT_SUBMISSIONS_PER_HOTKEY = 3
DEFAULT_MAX_SUBMISSIONS_PER_30D_PER_HOTKEY = 10


class SourceAddFunnelStage(str, Enum):
    SUBMITTED = "submitted"
    MANIFEST_VALIDATED = "manifest_validated"
    PROVENANCE_PRECHECK_PASSED = "provenance_precheck_passed"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"
    REJECTED_PRECHECK = "rejected_precheck"
    STATIC_SCAN_PASSED = "static_scan_passed"
    LLM_REVIEW_PASSED = "llm_review_passed"
    TRIAL_COMPLETED = "trial_completed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SourceAddRejectionReason(str, Enum):
    MANIFEST_INVALID = "manifest_invalid"
    DUPLICATE_SOURCE = "duplicate_source"
    HOTKEY_CONCURRENT_CAP = "hotkey_concurrent_cap"
    HOTKEY_30D_CAP = "hotkey_30d_cap"
    CREDENTIAL_INVALID = "credential_invalid"
    STATIC_SCAN_FAILED = "static_scan_failed"
    LLM_REVIEW_FAILED = "llm_review_failed"
    SCHEMA_VIOLATION = "schema_violation"
    ZERO_YIELD = "zero_yield"
    CATEGORY_MISMATCH = "category_mismatch"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTH_FAILURE = "auth_failure"
    TIMEOUT = "timeout"
    BELOW_ACCEPTANCE_FLOOR = "below_acceptance_floor"
    HUMAN_GATE_NOT_PASSED = "human_gate_not_passed"


# Static scan: adapter bundles are declarative fetch/parse code; anything that
# escapes the sandbox contract fails before an LLM review or trial is paid for.
_STATIC_SCAN_FORBIDDEN_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("subprocess", re.compile(r"\bsubprocess\b")),
    ("os_system", re.compile(r"\bos\.system\s*\(")),
    ("eval", re.compile(r"(?<![\w.])eval\s*\(")),
    ("exec", re.compile(r"(?<![\w.])exec\s*\(")),
    ("socket", re.compile(r"\bimport\s+socket\b|\bsocket\.socket\b")),
    ("ctypes", re.compile(r"\bimport\s+ctypes\b")),
    ("raw_credential", re.compile(r"(?i)\b(api[_-]?key|secret|password|token)\s*=\s*[\"'][A-Za-z0-9_\-]{12,}[\"']")),
    ("env_read", re.compile(r"\bos\.environ\b|\bos\.getenv\s*\(")),
    ("dunder_import", re.compile(r"__import__\s*\(")),
)


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

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["manifest"] = self.manifest.to_dict()
        data["stage_history"] = list(self.stage_history)
        data["rejection_reasons"] = list(self.rejection_reasons)
        return data


@dataclass(frozen=True)
class SourceAddCatalogEntry:
    catalog_id: str
    adapter_id: str
    miner_ref: str
    source_name: str
    source_kind: str
    declared_base_domains: tuple[str, ...]
    registry_provider_id: str
    measured_trial_yield: float
    accepted_at: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["declared_base_domains"] = list(self.declared_base_domains)
        return data


@dataclass(frozen=True)
class SourceAddTrialResult:
    adapter_id: str
    trial_icp_refs: tuple[str, ...]
    output_refs: tuple[str, ...]
    total_evidence_count: int
    category_matching_evidence_count: int
    measured_trial_yield: float
    declared_category: str
    cost_cents_spent: int
    diagnostics: dict[str, Any] = field(default_factory=dict)
    failure_reason: str = ""  # empty = trial ran to completion

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trial_icp_refs"] = list(self.trial_icp_refs)
        data["output_refs"] = list(self.output_refs)
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
    open_submission_count_for_hotkey: int = 0,
    submissions_last_30d_for_hotkey: int = 0,
    max_concurrent_per_hotkey: int = DEFAULT_MAX_CONCURRENT_SUBMISSIONS_PER_HOTKEY,
    max_per_30d_per_hotkey: int = DEFAULT_MAX_SUBMISSIONS_PER_30D_PER_HOTKEY,
    kms_encrypt: Callable[[str, str, str], Mapping[str, str]] | None = None,
) -> tuple[SourceAddSubmissionRecord | None, list[str]]:
    """Validate + admit one submission. Returns (record, errors).

    ``raw_credential`` never persists: when the manifest declares
    ``credential_ref_only``, the key is passed straight to ``kms_encrypt``
    (key_vault KMS-envelope pattern; encryption context = miner hotkey +
    adapter ref) and only the ciphertext envelope enters the record.
    """

    errors: list[str] = []
    try:
        manifest = SourceAddAdapterManifest.from_mapping(manifest_doc)
    except (KeyError, TypeError, ValueError) as exc:
        return None, [f"{SourceAddRejectionReason.MANIFEST_INVALID.value}: {exc}"]
    manifest_errors = validate_source_add_adapter_manifest(manifest)
    if manifest_errors:
        return None, [f"{SourceAddRejectionReason.MANIFEST_INVALID.value}: {error}" for error in manifest_errors]

    # Anti-spam before anything costs money.
    if _domains_overlap(manifest.declared_base_domains, existing_catalog_domains):
        errors.append(SourceAddRejectionReason.DUPLICATE_SOURCE.value)
    if open_submission_count_for_hotkey >= max(1, int(max_concurrent_per_hotkey)):
        errors.append(SourceAddRejectionReason.HOTKEY_CONCURRENT_CAP.value)
    if submissions_last_30d_for_hotkey >= max(1, int(max_per_30d_per_hotkey)):
        errors.append(SourceAddRejectionReason.HOTKEY_30D_CAP.value)
    if errors:
        return None, errors

    credential_envelope: dict[str, str] = {}
    if manifest.credential_policy == "credential_ref_only":
        if raw_credential:
            if kms_encrypt is None:
                return None, [f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: KMS encryption unavailable"]
            adapter_ref = f"source_add:{manifest.adapter_id}"
            try:
                envelope = dict(kms_encrypt(raw_credential, str(miner_hotkey), adapter_ref))
            except Exception as exc:
                return None, [f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: {str(exc)[:120]}"]
            if not envelope.get("ciphertext_b64"):
                return None, [f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: empty ciphertext"]
            credential_envelope = {
                "ciphertext_b64": str(envelope["ciphertext_b64"]),
                "kms_key_id": str(envelope.get("kms_key_id") or ""),
                "encryption_context_hash": str(envelope.get("encryption_context_hash") or ""),
                "credential_ref": f"encrypted_ref:source_add:{_stable_ref(miner_hotkey, adapter_ref)}",
            }
        elif not manifest.credential_ref:
            return None, [
                f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: credential_ref_only requires a key at intake"
            ]
    elif raw_credential:
        return None, [
            f"{SourceAddRejectionReason.CREDENTIAL_INVALID.value}: manifest declares no_credentials but a key was supplied"
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
    )
    return record, []


def _stable_ref(miner_hotkey: str, adapter_ref: str) -> str:
    return hashlib.sha256(f"{miner_hotkey}:{adapter_ref}".encode("utf-8")).hexdigest()[:32]


def static_scan_adapter_bundle(bundle_files: Mapping[str, str]) -> list[str]:
    """Cheapest funnel stage after validation: pattern scan of bundle source."""

    errors: list[str] = []
    for path, content in bundle_files.items():
        text = str(content or "")
        for label, pattern in _STATIC_SCAN_FORBIDDEN_PATTERNS:
            if pattern.search(text):
                errors.append(f"{label}:{path}")
    return sorted(set(errors))


def _reject(
    record: SourceAddSubmissionRecord,
    *,
    stage: SourceAddFunnelStage,
    reasons: Sequence[str],
    diagnostics: Mapping[str, Any] | None = None,
) -> SourceAddSubmissionRecord:
    from dataclasses import replace

    return replace(
        record,
        stage=SourceAddFunnelStage.REJECTED.value,
        stage_history=record.stage_history + (SourceAddFunnelStage.REJECTED.value,),
        rejection_stage=stage.value,
        rejection_reasons=tuple(str(reason)[:200] for reason in reasons),
        trial_diagnostics=dict(diagnostics or record.trial_diagnostics),
    )


def _advance(record: SourceAddSubmissionRecord, stage: SourceAddFunnelStage, **updates: Any) -> SourceAddSubmissionRecord:
    from dataclasses import replace

    return replace(
        record,
        stage=stage.value,
        stage_history=record.stage_history + (stage.value,),
        **updates,
    )


def apply_provenance_precheck_result(
    record: SourceAddSubmissionRecord,
    *,
    precheck_status: str,
    precheck_doc: Mapping[str, Any],
) -> SourceAddSubmissionRecord:
    """Advance intake to the provenance precheck outcome.

    This gate never accepts a source into the catalog. It only classifies the
    submitted adapter for operator review or early fake/test rejection.
    """

    normalized = str(precheck_status or "").strip().lower()
    if normalized == SourceAddFunnelStage.PROVENANCE_PRECHECK_PASSED.value:
        stage = SourceAddFunnelStage.PROVENANCE_PRECHECK_PASSED
    elif normalized == SourceAddFunnelStage.REJECTED_PRECHECK.value:
        stage = SourceAddFunnelStage.REJECTED_PRECHECK
    else:
        normalized = SourceAddFunnelStage.NEEDS_MANUAL_REVIEW.value
        stage = SourceAddFunnelStage.NEEDS_MANUAL_REVIEW
    return _advance(record, stage, precheck_status=normalized, precheck_doc=dict(precheck_doc or {}))


def run_static_scan_stage(
    record: SourceAddSubmissionRecord,
    bundle_files: Mapping[str, str],
) -> SourceAddSubmissionRecord:
    errors = static_scan_adapter_bundle(bundle_files)
    if errors:
        return _reject(
            record,
            stage=SourceAddFunnelStage.STATIC_SCAN_PASSED,
            reasons=[SourceAddRejectionReason.STATIC_SCAN_FAILED.value, *errors[:8]],
        )
    return _advance(record, SourceAddFunnelStage.STATIC_SCAN_PASSED)


def run_llm_review_stage(
    record: SourceAddSubmissionRecord,
    *,
    llm_reviewer: Callable[[SourceAddSubmissionRecord], Mapping[str, Any]],
) -> SourceAddSubmissionRecord:
    """LLM review verdict: {"verdict": "pass"|"fail", "reasons": [...]}."""

    try:
        verdict = llm_reviewer(record)
    except Exception as exc:
        return _reject(
            record,
            stage=SourceAddFunnelStage.LLM_REVIEW_PASSED,
            reasons=[SourceAddRejectionReason.LLM_REVIEW_FAILED.value, str(exc)[:160]],
        )
    if str(verdict.get("verdict") or "").strip().lower() != "pass":
        reasons = [str(item)[:160] for item in (verdict.get("reasons") or [])][:6]
        return _reject(
            record,
            stage=SourceAddFunnelStage.LLM_REVIEW_PASSED,
            reasons=[SourceAddRejectionReason.LLM_REVIEW_FAILED.value, *reasons],
        )
    return _advance(record, SourceAddFunnelStage.LLM_REVIEW_PASSED)


def run_sandboxed_trial(
    record: SourceAddSubmissionRecord,
    *,
    trial_icp_refs: Sequence[str],
    sandbox_runner: Callable[[SourceAddSubmissionRecord, str], Mapping[str, Any]],
    evidence_classifier: Callable[[str], str],
) -> SourceAddTrialResult:
    """Metered sandbox trial with category-scoped yield.

    ``sandbox_runner(record, icp_ref)`` executes the adapter for one trial ICP
    inside the private_runtime container (no credentials inside; hosts
    rewritten to the proxy; upstream allowlist = the adapter's registry entry;
    wall-clock kill) and returns
    ``{"output": <SourceAddTrialOutputRecord mapping>, "cost_cents": int}`` or
    ``{"error": "quota_exceeded"|"auth_failure"|"timeout", ...}``.

    ``evidence_classifier(evidence_ref)`` is the existing verification stack's
    category call. **Trial yield is category-scoped**: only evidence classified
    into the manifest's declared category (``source_kind``) counts, so a source
    claiming intent data but returning firmographics scores ≈ 0 and the
    mismatch lands in the trial diagnostics.

    ``measured_trial_yield`` = category-matching evidence per trial ICP,
    capped at 1.0.
    """

    declared_category = record.manifest.source_kind
    max_cost_cents = int(record.manifest.max_trial_cost_cents)
    total_evidence = 0
    matching_evidence = 0
    output_refs: list[str] = []
    cost_cents = 0
    per_icp: list[dict[str, Any]] = []

    for icp_ref in trial_icp_refs:
        if cost_cents >= max_cost_cents:
            return SourceAddTrialResult(
                adapter_id=record.adapter_id,
                trial_icp_refs=tuple(trial_icp_refs),
                output_refs=tuple(output_refs),
                total_evidence_count=total_evidence,
                category_matching_evidence_count=matching_evidence,
                measured_trial_yield=0.0,
                declared_category=declared_category,
                cost_cents_spent=cost_cents,
                diagnostics={"per_icp": per_icp, "stopped_at_icp": icp_ref},
                failure_reason=SourceAddRejectionReason.QUOTA_EXCEEDED.value,
            )
        try:
            outcome = sandbox_runner(record, str(icp_ref))
        except Exception as exc:
            per_icp.append({"icp_ref": str(icp_ref), "error": str(exc)[:160]})
            continue
        error = str(outcome.get("error") or "")
        cost_cents += max(0, int(outcome.get("cost_cents") or 0))
        if error:
            per_icp.append({"icp_ref": str(icp_ref), "error": error[:160]})
            if error in {
                SourceAddRejectionReason.AUTH_FAILURE.value,
                SourceAddRejectionReason.TIMEOUT.value,
                SourceAddRejectionReason.QUOTA_EXCEEDED.value,
            }:
                return SourceAddTrialResult(
                    adapter_id=record.adapter_id,
                    trial_icp_refs=tuple(trial_icp_refs),
                    output_refs=tuple(output_refs),
                    total_evidence_count=total_evidence,
                    category_matching_evidence_count=matching_evidence,
                    measured_trial_yield=0.0,
                    declared_category=declared_category,
                    cost_cents_spent=cost_cents,
                    diagnostics={"per_icp": per_icp},
                    failure_reason=error,
                )
            continue
        output_doc = outcome.get("output")
        if not isinstance(output_doc, Mapping):
            per_icp.append({"icp_ref": str(icp_ref), "error": "missing_output"})
            continue
        schema_errors = validate_source_add_trial_output(output_doc)
        if schema_errors:
            return SourceAddTrialResult(
                adapter_id=record.adapter_id,
                trial_icp_refs=tuple(trial_icp_refs),
                output_refs=tuple(output_refs),
                total_evidence_count=total_evidence,
                category_matching_evidence_count=matching_evidence,
                measured_trial_yield=0.0,
                declared_category=declared_category,
                cost_cents_spent=cost_cents,
                diagnostics={"per_icp": per_icp, "schema_errors": schema_errors[:8]},
                failure_reason=SourceAddRejectionReason.SCHEMA_VIOLATION.value,
            )
        output = SourceAddTrialOutputRecord.from_mapping(output_doc)
        output_refs.append(output.output_ref)
        icp_matching = 0
        for evidence_ref in output.evidence_refs:
            total_evidence += 1
            try:
                category = str(evidence_classifier(evidence_ref) or "")
            except Exception:
                category = ""
            if category == declared_category:
                matching_evidence += 1
                icp_matching += 1
        per_icp.append(
            {
                "icp_ref": str(icp_ref),
                "evidence_count": len(output.evidence_refs),
                "category_matching": icp_matching,
            }
        )

    measured_yield = min(1.0, matching_evidence / max(1, len(trial_icp_refs)))
    diagnostics: dict[str, Any] = {
        "per_icp": per_icp,
        "declared_category": declared_category,
        "total_evidence_count": total_evidence,
        "category_matching_evidence_count": matching_evidence,
    }
    failure_reason = ""
    if total_evidence == 0:
        failure_reason = SourceAddRejectionReason.ZERO_YIELD.value
    elif matching_evidence == 0:
        failure_reason = SourceAddRejectionReason.CATEGORY_MISMATCH.value
        diagnostics["category_mismatch"] = (
            f"source declared {declared_category!r} but no trial evidence classified into it"
        )
    return SourceAddTrialResult(
        adapter_id=record.adapter_id,
        trial_icp_refs=tuple(trial_icp_refs),
        output_refs=tuple(output_refs),
        total_evidence_count=total_evidence,
        category_matching_evidence_count=matching_evidence,
        measured_trial_yield=measured_yield,
        declared_category=declared_category,
        cost_cents_spent=cost_cents,
        diagnostics=diagnostics,
        failure_reason=failure_reason,
    )


def apply_trial_result(
    record: SourceAddSubmissionRecord,
    trial: SourceAddTrialResult,
) -> SourceAddSubmissionRecord:
    if trial.failure_reason:
        return _reject(
            record,
            stage=SourceAddFunnelStage.TRIAL_COMPLETED,
            reasons=[trial.failure_reason],
            diagnostics=trial.diagnostics,
        )
    return _advance(
        record,
        SourceAddFunnelStage.TRIAL_COMPLETED,
        measured_trial_yield=float(trial.measured_trial_yield),
        trial_diagnostics=dict(trial.diagnostics),
    )


def evaluate_source_add_acceptance(
    record: SourceAddSubmissionRecord,
    *,
    human_gate_passed: bool,
    acceptance_floor_yield: float = DEFAULT_ACCEPTANCE_FLOOR_YIELD,
    accepted_at: str = "",
    registry_provider_id: str = "",
) -> tuple[SourceAddSubmissionRecord, SourceAddCatalogEntry | None]:
    """Acceptance = category-scoped yield ≥ floor AND the human gate.

    Returns the advanced (or rejected) record plus the catalog entry on
    acceptance. Leg-1 reward creation hangs off the catalog entry
    (source_add_rewards.create_leg1_reward), not off this function.
    """

    if record.stage != SourceAddFunnelStage.TRIAL_COMPLETED.value:
        return (
            _reject(record, stage=SourceAddFunnelStage.ACCEPTED, reasons=["acceptance_requires_completed_trial"]),
            None,
        )
    if record.measured_trial_yield < float(acceptance_floor_yield):
        return (
            _reject(
                record,
                stage=SourceAddFunnelStage.ACCEPTED,
                reasons=[
                    SourceAddRejectionReason.BELOW_ACCEPTANCE_FLOOR.value,
                    f"measured={record.measured_trial_yield:.4f} floor={float(acceptance_floor_yield):.4f}",
                ],
            ),
            None,
        )
    if not human_gate_passed:
        return (
            _reject(record, stage=SourceAddFunnelStage.ACCEPTED, reasons=[SourceAddRejectionReason.HUMAN_GATE_NOT_PASSED.value]),
            None,
        )
    accepted = _advance(record, SourceAddFunnelStage.ACCEPTED, acceptance_human_gate_passed=True)
    entry = SourceAddCatalogEntry(
        catalog_id="source_catalog:" + sha256_json({"adapter_id": record.adapter_id}).split(":", 1)[1][:16],
        adapter_id=record.adapter_id,
        miner_ref=record.manifest.miner_ref,
        source_name=record.manifest.source_name,
        source_kind=record.manifest.source_kind,
        declared_base_domains=record.manifest.declared_base_domains,
        registry_provider_id=str(registry_provider_id or ""),
        measured_trial_yield=float(record.measured_trial_yield),
        accepted_at=str(accepted_at),
    )
    return accepted, entry


@dataclass(frozen=True)
class SourceAddSuggestionDoc:
    """Loop-emitted source-add suggestion (loops never execute SOURCE_ADD).

    When probes suggest a missing source, the loop drops one of these into the
    submission queue: provider hint + endpoint class + the evidence gap it
    would close + probe receipts (event hashes only — no bodies, no queries).
    """

    suggestion_id: str
    created_by_run_id: str
    provider_hint: str
    endpoint_class: str
    evidence_gap: str
    probe_receipt_hashes: tuple[str, ...] = ()

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        provider_hint: str,
        endpoint_class: str,
        evidence_gap: str,
        probe_receipt_hashes: Sequence[str] = (),
    ) -> "SourceAddSuggestionDoc":
        payload = {
            "run_id": str(run_id),
            "provider_hint": str(provider_hint)[:120],
            "endpoint_class": str(endpoint_class)[:120],
            "evidence_gap": str(evidence_gap)[:700],
        }
        return cls(
            suggestion_id="source_add_suggestion:" + sha256_json(payload).split(":", 1)[1][:16],
            created_by_run_id=payload["run_id"],
            provider_hint=payload["provider_hint"],
            endpoint_class=payload["endpoint_class"],
            evidence_gap=payload["evidence_gap"],
            probe_receipt_hashes=tuple(str(item)[:80] for item in probe_receipt_hashes)[:8],
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["probe_receipt_hashes"] = list(self.probe_receipt_hashes)
        return data


def validate_source_add_suggestion(doc: SourceAddSuggestionDoc | Mapping[str, Any]) -> list[str]:
    if isinstance(doc, Mapping):
        try:
            doc = SourceAddSuggestionDoc(
                suggestion_id=str(doc["suggestion_id"]),
                created_by_run_id=str(doc["created_by_run_id"]),
                provider_hint=str(doc.get("provider_hint") or ""),
                endpoint_class=str(doc.get("endpoint_class") or ""),
                evidence_gap=str(doc.get("evidence_gap") or ""),
                probe_receipt_hashes=tuple(str(item) for item in doc.get("probe_receipt_hashes", [])),
            )
        except KeyError as exc:
            return [f"missing field: {exc}"]
    errors: list[str] = []
    if not doc.suggestion_id.startswith("source_add_suggestion:"):
        errors.append("suggestion_id must be source_add_suggestion:-prefixed")
    if not doc.created_by_run_id:
        errors.append("created_by_run_id is required")
    if not doc.evidence_gap:
        errors.append("evidence_gap is required")
    lowered = f"{doc.provider_hint} {doc.endpoint_class} {doc.evidence_gap}".lower()
    for marker in ("http://", "https://", "api_key", "bearer ", "sk-or-"):
        if marker in lowered:
            errors.append(f"suggestion doc must not carry URLs or credentials ({marker.strip()})")
    return errors

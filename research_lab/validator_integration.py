"""Validator-side Research Lab fetch/verify/weight integration.

This module is production-oriented but defaults to shadow/read-only behavior.
It lets validators fetch official Research Lab shadow bundles, verify them
locally, and compute the Research Lab weight component without submitting it
on-chain unless an explicit future live mutation flag is enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from urllib.parse import quote
from urllib.request import Request, urlopen

from leadpoet_canonical.attested_receipts import (
    SCORING_ROLE,
    SCORING_PURPOSES,
    sha256_json as attested_sha256_json,
    validate_signed_receipt,
    verify_receipt_lineage,
)
from leadpoet_verifier.golden_vectors import run_golden_vectors
from leadpoet_verifier.economics import DEFAULT_RESEARCH_LAB_EMISSION_PERCENT, allocate_research_lab_epoch
from leadpoet_verifier.research_evaluation import (
    build_research_evaluation_score_bundle,
    score_bundle_to_weight_input,
    verify_research_evaluation_score_bundle,
)

from .canonical import sha256_json
from .production_shadow import (
    assert_shadow_output_read_only,
    build_controlled_production_shadow,
    compute_verifier_divergence,
    load_production_shadow_fixture,
)


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "validator_integration_fixtures.json"
TRUTHY_VALUES = {"1", "true", "yes", "on"}
PERCENT_EPSILON = 0.000001
ATTESTED_ALLOCATION_SCHEMA_VERSION = "leadpoet.attested_allocation_bundle.v2"
ATTESTED_ALLOCATION_PURPOSE = "research_lab.allocation.v1"
IMMUTABLE_ECR_IMAGE_RE = re.compile(
    r"^[0-9]{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?/"
    r"[A-Za-z0-9._/-]+@sha256:[0-9a-f]{64}$"
)
PRIVATE_S3_MANIFEST_RE = re.compile(
    r"^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[A-Za-z0-9._/-]+\.json$"
)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_SERVING_IMAGE_DIGEST_PATHS = {
    ("serving_model_version", "parent_model", "image_digest"),
    ("serving_model_version", "candidate_model", "image_digest"),
}
_SERVING_MANIFEST_URI_PATHS = {
    ("serving_model_version", "parent_model", "manifest_uri"),
    ("serving_model_version", "candidate_model", "manifest_uri"),
}


def load_independent_gateway_identity(commit_sha: str) -> dict[str, Any] | None:
    """Load a repeated-build gateway PCR0 record for one exact commit."""

    from validator_tee.host.gateway_pcr0_builder import load_cached_gateway_identity

    configured = str(os.getenv("VALIDATOR_GATEWAY_PCR0_CACHE_FILE", "") or "").strip()
    cache_path = (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "leadpoet" / "gateway-pcr0-cache.json"
    )
    return load_cached_gateway_identity(cache_path, str(commit_sha or "").strip().lower())


def _request_headers(*, include_internal_key: bool = False) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if include_internal_key:
        internal_key = (
            os.getenv("RESEARCH_LAB_INTERNAL_API_KEY", "").strip()
            or os.getenv("LEADPOET_INTERNAL_SECRET", "").strip()
        )
        if internal_key:
            headers["x-leadpoet-internal-key"] = internal_key
    return headers


def _validator_lab_cap_ceiling_percent() -> float:
    raw = os.getenv("RESEARCH_LAB_EMISSION_PERCENT", "").strip()
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            return float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)
    return float(DEFAULT_RESEARCH_LAB_EMISSION_PERCENT)


def _argv_value(name: str) -> str:
    try:
        index = sys.argv.index(name)
    except ValueError:
        return ""
    if index + 1 >= len(sys.argv):
        return ""
    return str(sys.argv[index + 1] or "")


def _is_production_subnet(data: Mapping[str, Any] | None = None) -> bool:
    data = data or {}
    network = str(
        data.get("BITTENSOR_NETWORK")
        or data.get("SUBTENSOR_NETWORK")
        or os.getenv("BITTENSOR_NETWORK")
        or os.getenv("SUBTENSOR_NETWORK")
        or _argv_value("--subtensor_network")
        or ""
    ).strip().lower()
    netuid = str(
        data.get("BITTENSOR_NETUID")
        or data.get("NETUID")
        or os.getenv("BITTENSOR_NETUID")
        or os.getenv("NETUID")
        or _argv_value("--netuid")
        or ""
    ).strip()
    return network == "finney" and netuid == "71"


def _default_for_prod(data: Mapping[str, Any] | None = None) -> bool:
    return _is_production_subnet(data)


def _truthy_with_default(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str) and not value.strip():
        return bool(default)
    return _truthy(value)


@dataclass(frozen=True)
class ResearchLabValidatorFlags:
    fetch_enabled: bool = False
    shadow_verify_enabled: bool = False
    evaluation_verify_enabled: bool = False
    audit_verify_enabled: bool = False
    require_shadow_verification_before_submit: bool = False
    require_evaluation_verification_before_submit: bool = False
    reimbursements_enabled: bool = True
    weight_mutation_enabled: bool = False
    production_writes_enabled: bool = False
    submit_on_chain_enabled: bool = False
    fulfillment_mutation_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "ResearchLabValidatorFlags":
        data = data or {}
        prod_default = _default_for_prod(data)
        return cls(
            fetch_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_VALIDATOR_FETCH_ENABLED", data.get("fetch_enabled")),
                prod_default,
            ),
            shadow_verify_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_VALIDATOR_SHADOW_VERIFY_ENABLED", data.get("shadow_verify_enabled")),
                prod_default,
            ),
            evaluation_verify_enabled=_truthy_with_default(
                data.get(
                    "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED",
                    data.get("evaluation_verify_enabled"),
                ),
                prod_default,
            ),
            audit_verify_enabled=_truthy(
                data.get("RESEARCH_LAB_VALIDATOR_AUDIT_VERIFY_ENABLED", data.get("audit_verify_enabled", False))
            ),
            require_shadow_verification_before_submit=_truthy_with_default(
                data.get(
                    "RESEARCH_LAB_REQUIRE_SHADOW_VERIFICATION_BEFORE_SUBMIT",
                    data.get("require_shadow_verification_before_submit"),
                ),
                prod_default,
            ),
            require_evaluation_verification_before_submit=_truthy_with_default(
                data.get(
                    "RESEARCH_LAB_REQUIRE_EVALUATION_VERIFICATION_BEFORE_SUBMIT",
                    data.get("require_evaluation_verification_before_submit"),
                ),
                prod_default,
            ),
            reimbursements_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_REIMBURSEMENTS_ENABLED", data.get("reimbursements_enabled", True)),
                True,
            ),
            weight_mutation_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_WEIGHT_MUTATION_ENABLED", data.get("weight_mutation_enabled")),
                prod_default,
            ),
            production_writes_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_PRODUCTION_WRITES_ENABLED", data.get("production_writes_enabled")),
                prod_default,
            ),
            submit_on_chain_enabled=_truthy_with_default(
                data.get("RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED", data.get("submit_on_chain_enabled")),
                prod_default,
            ),
            fulfillment_mutation_enabled=_truthy(
                data.get("RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED", data.get("fulfillment_mutation_enabled", False))
            ),
        )

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)

    def enabled_mutation_flags(self) -> list[str]:
        return [
            name
            for name, enabled in {
                "RESEARCH_LAB_WEIGHT_MUTATION_ENABLED": self.weight_mutation_enabled,
                "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": self.production_writes_enabled,
                "RESEARCH_LAB_SUBMIT_ON_CHAIN_ENABLED": self.submit_on_chain_enabled,
                "RESEARCH_LAB_FULFILLMENT_MUTATION_ENABLED": self.fulfillment_mutation_enabled,
            }.items()
            if enabled
        ]

    def live_allocation_enabled(self) -> bool:
        return self.reimbursements_enabled or self.weight_mutation_enabled or self.submit_on_chain_enabled


def fetch_research_lab_shadow_bundle(gateway_url: str, epoch: int, *, timeout_seconds: int = 20) -> dict[str, Any]:
    """Fetch a Research Lab shadow report from the gateway.

    This performs a read-only HTTP GET. Callers still need to verify the bundle
    with ``verify_research_lab_shadow_bundle`` before trusting any weight field.
    """
    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/reports/shadow/{int(epoch)}",
        headers=_request_headers(),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_research_lab_evaluation_bundle_page(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Fetch official Research Lab evaluation score bundles for an epoch."""
    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/evaluations/latest/{int(epoch)}",
        headers=_request_headers(include_internal_key=True),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_research_lab_score_bundle(
    gateway_url: str,
    score_bundle_id: str,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Fetch one official Research Lab evaluation score bundle by id."""
    base = gateway_url.rstrip("/")
    safe_score_bundle_id = quote(str(score_bundle_id), safe="")
    request = Request(
        f"{base}/research-lab/evaluations/score-bundles/{safe_score_bundle_id}",
        headers=_request_headers(include_internal_key=True),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def allocation_referenced_score_bundle_ids(allocation_bundle: Mapping[str, Any] | None) -> list[str]:
    """Return score bundles that are evidence for live allocation obligations."""
    if not isinstance(allocation_bundle, Mapping):
        return []
    source_state = allocation_bundle.get("source_state")
    if not isinstance(source_state, Mapping):
        return []

    seen: set[str] = set()
    score_bundle_ids: list[str] = []
    for section in ("champion_obligations", "queued_champion_obligations"):
        rows = source_state.get(section) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key in ("score_bundle_id", "source_score_bundle_id", "derived_score_bundle_id"):
                value = str(row.get(key) or "").strip()
                if value.startswith("score_bundle:") and value not in seen:
                    seen.add(value)
                    score_bundle_ids.append(value)
    return score_bundle_ids


def allocation_can_skip_score_bundle_verification(allocation_doc: Mapping[str, Any] | None) -> bool:
    """True when all paid Lab rewards are independent of candidate scoring."""

    if not isinstance(allocation_doc, Mapping):
        return False
    for section in ("champion_allocations", "queued_champion_allocations"):
        rows = allocation_doc.get(section) or []
        if any(
            float(row.get("paid_alpha_percent") or 0.0) > 0
            for row in rows
            if isinstance(row, Mapping)
        ):
            return False
    for section in ("source_add_allocations", "reimbursement_allocations"):
        rows = allocation_doc.get(section) or []
        if any(
            float(row.get("paid_alpha_percent") or 0.0) > 0
            for row in rows
            if isinstance(row, Mapping)
        ):
            return True
    return False


def allocation_can_proceed_without_score_bundles(
    allocation_doc: Mapping[str, Any] | None,
    verification_errors: Iterable[str],
) -> bool:
    return allocation_can_skip_score_bundle_verification(allocation_doc) and set(
        str(error) for error in verification_errors
    ) == {"no_verified_evaluation_score_bundles"}


def merge_research_lab_evaluation_bundle_page(
    page: Mapping[str, Any],
    referenced_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Append explicitly referenced score-bundle rows to an evaluation page."""
    merged = dict(page)
    current_rows = merged.get("score_bundles", [])
    if not isinstance(current_rows, list):
        return merged

    rows = list(current_rows)
    row_index_by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        bundle = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
        score_bundle_id = str(row.get("score_bundle_id") or bundle.get("score_bundle_id") or "").strip()
        if score_bundle_id:
            row_index_by_id[score_bundle_id] = index

    added = 0
    for row in referenced_rows:
        if not isinstance(row, Mapping):
            continue
        bundle = row.get("score_bundle_doc") if isinstance(row.get("score_bundle_doc"), Mapping) else {}
        score_bundle_id = str(row.get("score_bundle_id") or bundle.get("score_bundle_id") or "").strip()
        if score_bundle_id and score_bundle_id in row_index_by_id:
            rows[row_index_by_id[score_bundle_id]] = dict(row)
            continue
        rows.append(dict(row))
        added += 1
        if score_bundle_id:
            row_index_by_id[score_bundle_id] = len(rows) - 1

    merged["score_bundles"] = rows
    merged["count"] = len(rows)
    if added:
        merged["referenced_score_bundle_count"] = added
    return merged


def fetch_research_lab_audit_bundle(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Fetch the latest Research Lab audit bundle for an epoch."""
    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/audit/latest/{int(epoch)}",
        headers=_request_headers(include_internal_key=True),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_research_lab_allocation_bundle(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = 20,
) -> dict[str, Any]:
    """Fetch the live Research Lab allocation bundle for an epoch."""
    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/allocations/live/{int(epoch)}",
        headers=_request_headers(),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_research_lab_attested_allocation_bundle(
    gateway_url: str,
    epoch: int,
    *,
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Fetch the additive enclave receipt for one live allocation bundle."""

    base = gateway_url.rstrip("/")
    request = Request(
        f"{base}/research-lab/allocations/attested/{int(epoch)}",
        headers=_request_headers(),
        method="GET",
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def verify_gateway_allocation_attestation(
    *,
    receipt: Mapping[str, Any],
    expected_epoch_id: int,
    expected_pcr0: str | None,
    expected_purpose: str = ATTESTED_ALLOCATION_PURPOSE,
) -> tuple[bool, dict[str, Any]]:
    """Verify AWS Nitro and optionally an independently rebuilt gateway PCR0."""

    from leadpoet_canonical.nitro import verify_nitro_attestation_full

    return verify_nitro_attestation_full(
        attestation_b64=str(receipt.get("attestation_document_b64") or ""),
        expected_pcr0=expected_pcr0,
        expected_pubkey=str(receipt.get("enclave_pubkey") or ""),
        expected_purpose=expected_purpose,
        expected_epoch_id=expected_epoch_id,
        role="gateway",
        # Never consult the static gateway allowlist for v2. Without an
        # independently rebuilt expected PCR0, this verifies AWS authenticity
        # only and reports required_ready=False below.
        skip_pcr0_verification=not bool(expected_pcr0),
    )


def verify_research_lab_shadow_bundle(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
    official_weight_hash: str | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.shadow_verify_enabled:
        errors.append("validator_shadow_verify_disabled")
    errors.extend(validator_flags.enabled_mutation_flags())

    read_only = assert_shadow_output_read_only(bundle)
    if not read_only["passed"]:
        errors.extend(read_only["errors"])

    if _contains_secret_material(bundle):
        errors.append("bundle_contains_raw_secret_material")

    source_state = bundle.get("source_state")
    source_state_hash = bundle.get("source_state_hash")
    if source_state is not None or source_state_hash is not None:
        if not source_state or not source_state_hash:
            errors.append("source_state_and_hash_must_be_present_together")
        elif sha256_json(source_state) != source_state_hash:
            errors.append("source_state_hash_diverged")

    golden_errors = run_golden_vectors()
    if golden_errors:
        errors.extend(f"open_verifier:{error}" for error in golden_errors)

    weight_vector = bundle.get("weight_vector", {})
    if not isinstance(weight_vector, Mapping):
        errors.append("weight_vector_must_be_object")
        weight_vector = {}
    divergence = compute_verifier_divergence(
        weight_vector,
        official_weight_hash=official_weight_hash or bundle.get("weight_vector_hash"),
    )
    if divergence["diverged"]:
        errors.append("weight_vector_hash_diverged")

    return {
        "passed": not errors,
        "errors": errors,
        "shadow_only": bool(bundle.get("shadow_only", False)),
        "read_only": bool(bundle.get("read_only", False)),
        "validator_flags": validator_flags.to_dict(),
        "bundle_id": bundle.get("bundle_id"),
        "epoch": bundle.get("epoch"),
        "weight_vector_hash": sha256_json(weight_vector) if weight_vector else None,
        "verifier_divergence": divergence,
        "on_chain_submission_allowed": False,
    }


def verify_research_lab_allocation_bundle(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.fetch_enabled:
        errors.append("validator_fetch_disabled")
    if not validator_flags.live_allocation_enabled():
        errors.append("validator_live_research_lab_weight_flags_disabled")
    if _contains_secret_material(bundle):
        errors.append("allocation_bundle_contains_raw_secret_material")
    if bundle.get("bundle_type") != "research_lab_live_allocation_bundle":
        errors.append("unexpected_allocation_bundle_type")
    if not bundle.get("submission_allowed") or not bundle.get("on_chain_submission_allowed"):
        errors.append("gateway_live_research_lab_weight_submission_disabled")

    source_state = bundle.get("source_state")
    source_state_hash = bundle.get("source_state_hash")
    if not isinstance(source_state, Mapping) or not source_state_hash:
        errors.append("allocation_source_state_required")
        source_state = {}
    elif sha256_json(source_state) != source_state_hash:
        errors.append("allocation_source_state_hash_diverged")
    else:
        try:
            if int(source_state.get("epoch")) != int(bundle.get("epoch")):
                errors.append("allocation_source_state_epoch_diverged")
        except (TypeError, ValueError):
            errors.append("allocation_source_state_epoch_invalid")
        if source_state.get("netuid") is not None and bundle.get("netuid") is not None:
            try:
                if int(source_state.get("netuid")) != int(bundle.get("netuid")):
                    errors.append("allocation_source_state_netuid_diverged")
            except (TypeError, ValueError):
                errors.append("allocation_source_state_netuid_invalid")

    allocation_doc = bundle.get("allocation_doc")
    allocation_hash = bundle.get("allocation_hash")
    if not isinstance(allocation_doc, Mapping) or not allocation_hash:
        errors.append("allocation_doc_and_hash_required")
        allocation_doc = {}
    else:
        expected_payload = {k: v for k, v in dict(allocation_doc).items() if k != "allocation_hash"}
        if sha256_json(expected_payload) != allocation_hash:
            errors.append("allocation_hash_diverged")
        if allocation_doc.get("allocation_hash") != allocation_hash:
            errors.append("allocation_doc_hash_field_diverged")

    lab_cap = float(allocation_doc.get("lab_cap_percent") or 0.0) if allocation_doc else 0.0
    validator_lab_cap_ceiling = _validator_lab_cap_ceiling_percent()
    paid = sum(
        float(allocation_doc.get(field) or 0.0)
        for field in (
            "source_add_alpha_percent",
            "reimbursement_alpha_percent",
            "champion_alpha_percent",
            "queued_champion_alpha_percent",
            "unallocated_percent",
        )
    )
    if lab_cap < 0 or lab_cap > 100:
        errors.append("invalid_lab_cap_percent")
    if lab_cap > validator_lab_cap_ceiling + PERCENT_EPSILON:
        errors.append("allocation_lab_cap_exceeds_validator_policy")
    if paid > lab_cap + PERCENT_EPSILON:
        errors.append("allocation_exceeds_lab_cap")

    recomputed_allocation_hash: str | None = None
    policy = source_state.get("policy") if isinstance(source_state, Mapping) else None
    source_add = source_state.get("source_add_obligations") if isinstance(source_state, Mapping) else None
    reimbursements = source_state.get("reimbursement_obligations") if isinstance(source_state, Mapping) else None
    champions = source_state.get("champion_obligations") if isinstance(source_state, Mapping) else None
    if not isinstance(policy, Mapping):
        errors.append("allocation_policy_required")
    if not isinstance(reimbursements, list):
        errors.append("allocation_reimbursement_obligations_must_be_array")
        reimbursements = []
    if source_add is None:
        source_add = []
    elif not isinstance(source_add, list):
        errors.append("allocation_source_add_obligations_must_be_array")
        source_add = []
    if not isinstance(champions, list):
        errors.append("allocation_champion_obligations_must_be_array")
        champions = []
    if isinstance(policy, Mapping):
        try:
            policy_lab_cap = float(policy.get("research_lab_emission_percent") or 0.0)
        except (TypeError, ValueError):
            errors.append("allocation_policy_lab_cap_invalid")
            policy_lab_cap = 0.0
        if policy_lab_cap > validator_lab_cap_ceiling + PERCENT_EPSILON:
            errors.append("allocation_policy_cap_exceeds_validator_policy")
        if abs(policy_lab_cap - lab_cap) > PERCENT_EPSILON:
            errors.append("allocation_policy_cap_diverged_from_doc")
        if isinstance(source_state, Mapping) and source_state.get("policy_id") is not None:
            if str(policy.get("policy_id") or "") != str(source_state.get("policy_id") or ""):
                errors.append("allocation_policy_id_diverged")
        try:
            allocation_epoch = int(source_state.get("epoch", bundle.get("epoch")))
            recomputed = allocate_research_lab_epoch(
                allocation_epoch,
                policy,
                reimbursements,
                champions,
                active_source_add_obligations=source_add,
            )
            recomputed_allocation_hash = str(recomputed.get("allocation_hash") or "")
            if allocation_hash and recomputed_allocation_hash != str(allocation_hash):
                errors.append("allocation_recompute_hash_diverged")
            if allocation_doc and dict(recomputed) != dict(allocation_doc):
                errors.append("allocation_recompute_doc_diverged")
        except Exception as exc:
            errors.append(f"allocation_recompute_failed:{str(exc)[:120]}")

    return {
        "passed": not errors,
        "errors": errors,
        "epoch": bundle.get("epoch"),
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": source_state_hash,
        "allocation_hash": allocation_hash,
        "recomputed_allocation_hash": recomputed_allocation_hash,
        "validator_lab_cap_ceiling_percent": validator_lab_cap_ceiling,
        "allocation_doc": dict(allocation_doc or {}),
        "on_chain_submission_allowed": not errors,
    }


def verify_research_lab_attested_allocation_bundle(
    document: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
    expected_gateway_pcr0: str | None = None,
    expected_gateway_commit: str | None = None,
    expected_gateway_identities: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify allocation arithmetic, receipt roots, Nitro, and rebuild identity."""

    errors: list[str] = []
    expected_fields = {
        "schema_version",
        "bundle",
        "receipt",
        "parent_receipts",
        "lineage_bindings",
        "lineage_complete",
        "gateway_pcr0",
        "persistence_status",
    }
    if not isinstance(document, Mapping) or set(document) != expected_fields:
        return {
            "passed": False,
            "required_ready": False,
            "errors": ["attested_allocation_fields_invalid"],
        }
    if document.get("schema_version") != ATTESTED_ALLOCATION_SCHEMA_VERSION:
        errors.append("attested_allocation_schema_invalid")

    bundle = document.get("bundle")
    receipt = document.get("receipt")
    parent_receipts_value = document.get("parent_receipts")
    lineage_bindings_value = document.get("lineage_bindings")
    lineage_complete = document.get("lineage_complete")
    if not isinstance(bundle, Mapping):
        errors.append("attested_allocation_bundle_missing")
        bundle = {}
    if not isinstance(receipt, Mapping):
        errors.append("attested_allocation_receipt_missing")
        receipt = {}
    if not isinstance(parent_receipts_value, list):
        errors.append("attested_allocation_parent_receipts_invalid")
        parent_receipts_value = []
    if not isinstance(lineage_bindings_value, list):
        errors.append("attested_allocation_lineage_bindings_invalid")
        lineage_bindings_value = []
    if not isinstance(lineage_complete, bool):
        errors.append("attested_allocation_lineage_complete_invalid")
        lineage_complete = False
    elif not lineage_complete:
        errors.append("attested_allocation_lineage_incomplete")

    allocation_verification = verify_research_lab_allocation_bundle(bundle, flags=flags)
    allocation_hash = str(bundle.get("allocation_hash") or "")
    errors.extend(
        "allocation:%s" % error
        for error in allocation_verification.get("errors", [])
    )

    parent_receipts: dict[str, Mapping[str, Any]] = {}
    normalized_bindings: list[dict[str, str]] = []
    try:
        for parent in parent_receipts_value:
            if not isinstance(parent, Mapping):
                raise ValueError("parent receipt is not an object")
            validate_signed_receipt(parent)
            parent_hash = str(parent.get("receipt_hash") or "")
            if parent_hash in parent_receipts:
                raise ValueError("parent receipt is duplicated")
            parent_receipts[parent_hash] = parent

        for binding in lineage_bindings_value:
            if not isinstance(binding, Mapping) or set(binding) != {
                "score_bundle_id",
                "score_bundle_hash",
                "receipt_hash",
                "receipt_purpose",
            }:
                raise ValueError("lineage binding fields are invalid")
            score_bundle_id = str(binding.get("score_bundle_id") or "")
            score_bundle_hash = str(binding.get("score_bundle_hash") or "")
            receipt_hash_value = str(binding.get("receipt_hash") or "")
            receipt_purpose = str(binding.get("receipt_purpose") or "")
            if not score_bundle_hash.startswith("sha256:"):
                raise ValueError("lineage score bundle hash is invalid")
            if score_bundle_id != "score_bundle:" + score_bundle_hash.split(":", 1)[1]:
                raise ValueError("lineage score bundle id does not match its hash")
            parent = parent_receipts.get(receipt_hash_value)
            if parent is None:
                raise ValueError("lineage binding receipt is missing")
            if parent.get("purpose") != receipt_purpose or receipt_purpose not in SCORING_PURPOSES:
                raise ValueError("lineage binding receipt purpose is invalid")
            if parent.get("evidence_roots", {}).get("score_bundle") != score_bundle_hash:
                raise ValueError("lineage binding receipt does not bind score bundle")
            if receipt_purpose != "research_lab.promotion_decision.v1":
                raise ValueError("champion allocation requires a promotion decision receipt")
            if parent.get("evidence_roots", {}).get(
                "promotion_decision_status"
            ) != attested_sha256_json({"status": "promotion_passed"}):
                raise ValueError("promotion decision receipt is not a passed gate")
            normalized_bindings.append(
                {
                    "score_bundle_id": score_bundle_id,
                    "score_bundle_hash": score_bundle_hash,
                    "receipt_hash": receipt_hash_value,
                    "receipt_purpose": receipt_purpose,
                }
            )

        source_state = bundle.get("source_state") if isinstance(bundle.get("source_state"), Mapping) else {}
        expected_score_bundle_ids = sorted(
            {
                str(item.get("score_bundle_id") or "")
                for item in (source_state.get("champion_obligations") or [])
                if isinstance(item, Mapping)
                and str(item.get("score_bundle_id") or "").startswith("score_bundle:")
            }
        )
        bound_score_bundle_ids = sorted(
            {item["score_bundle_id"] for item in normalized_bindings}
        )
        if bound_score_bundle_ids != expected_score_bundle_ids:
            errors.append("attested_allocation_lineage_obligations_diverged")

        validate_signed_receipt(receipt)
        if receipt.get("role") != SCORING_ROLE:
            errors.append("attested_allocation_receipt_role_invalid")
        if receipt.get("purpose") != ATTESTED_ALLOCATION_PURPOSE:
            errors.append("attested_allocation_receipt_purpose_invalid")
        if receipt.get("status") != "succeeded":
            errors.append("attested_allocation_receipt_not_successful")
        if int(receipt.get("epoch_id", -1)) != int(bundle.get("epoch", -2)):
            errors.append("attested_allocation_receipt_epoch_diverged")

        receipt_input = {
            "epoch": int(source_state.get("epoch")),
            "policy": source_state.get("policy"),
            "active_reimbursement_obligations": source_state.get("reimbursement_obligations"),
            "active_champion_obligations": source_state.get("champion_obligations"),
            "receipt_lineage_bindings": sorted(
                normalized_bindings,
                key=lambda item: (item["score_bundle_id"], item["receipt_hash"]),
            ),
        }
        if "source_add_obligations" in source_state:
            receipt_input["active_source_add_obligations"] = source_state.get("source_add_obligations")
        if receipt.get("input_root") != attested_sha256_json(receipt_input):
            errors.append("attested_allocation_receipt_input_diverged")
        receipt_output = {"allocation": bundle.get("allocation_doc")}
        if receipt.get("output_root") != attested_sha256_json(receipt_output):
            errors.append("attested_allocation_receipt_output_diverged")
        expected_direct_parents = sorted(
            {item["receipt_hash"] for item in normalized_bindings}
        )
        if receipt.get("parent_receipt_hashes") != expected_direct_parents:
            errors.append("attested_allocation_direct_parents_diverged")
        if receipt.get("evidence_roots") != {"allocation": allocation_hash}:
            errors.append("attested_allocation_evidence_diverged")
        ordered_lineage = verify_receipt_lineage(receipt, parent_receipts)
        if set(ordered_lineage) != set(parent_receipts) | {str(receipt["receipt_hash"])}:
            raise ValueError("allocation receipt contains disconnected lineage")
        for score_receipt in parent_receipts.values():
            if score_receipt.get("purpose") not in {
                "research_lab.candidate_score.v1",
                "research_lab.baseline_score.v1",
                "research_lab.benchmark.v1",
                "research_lab.rebenchmark.v1",
            }:
                continue
            baseline_hash = str(
                score_receipt.get("evidence_roots", {}).get("baseline_score_summary")
                or ""
            )
            if not baseline_hash:
                continue
            score_lineage = set(verify_receipt_lineage(score_receipt, parent_receipts))
            baseline_matches = [
                ancestor
                for ancestor in parent_receipts.values()
                if str(ancestor.get("receipt_hash") or "") in score_lineage
                and ancestor.get("purpose")
                in {
                    "research_lab.baseline_score.v1",
                    "research_lab.benchmark.v1",
                    "research_lab.rebenchmark.v1",
                }
                and ancestor.get("evidence_roots", {}).get("baseline_score_summary")
                == baseline_hash
            ]
            if not baseline_matches:
                raise ValueError("score receipt baseline ancestry is incomplete")
    except Exception as exc:
        errors.append("attested_allocation_receipt_invalid:%s" % str(exc)[:120])

    identities = {
        str(commit or "").lower(): dict(identity)
        for commit, identity in dict(expected_gateway_identities or {}).items()
        if isinstance(identity, Mapping)
    }
    if expected_gateway_commit and expected_gateway_pcr0:
        identities[str(expected_gateway_commit).lower()] = {
            "commit_sha": str(expected_gateway_commit).lower(),
            "pcr0": str(expected_gateway_pcr0).lower(),
        }
    all_gateway_identities_supplied = True
    attestation_data: dict[str, Any] = {}
    receipts_to_verify = [receipt, *parent_receipts.values()]
    for index, candidate_receipt in enumerate(receipts_to_verify):
        if not isinstance(candidate_receipt, Mapping):
            continue
        commit_sha = str(candidate_receipt.get("commit_sha") or "").lower()
        identity = identities.get(commit_sha)
        expected_pcr0_for_receipt = str((identity or {}).get("pcr0") or "") or None
        if expected_pcr0_for_receipt is None:
            all_gateway_identities_supplied = False
        try:
            valid, candidate_attestation = verify_gateway_allocation_attestation(
                receipt=candidate_receipt,
                expected_epoch_id=int(candidate_receipt.get("epoch_id", -1)),
                expected_pcr0=expected_pcr0_for_receipt,
                expected_purpose=str(candidate_receipt.get("purpose") or ""),
            )
            if index == 0:
                attestation_data = dict(candidate_attestation)
            if not valid:
                errors.append(
                    "attested_allocation_nitro_invalid:%s"
                    % str(candidate_attestation.get("error") or "unknown")[:120]
                )
            if candidate_attestation.get("purpose") != candidate_receipt.get("purpose"):
                errors.append("attested_allocation_attestation_purpose_invalid")
            if int(candidate_attestation.get("epoch_id", -1)) != int(candidate_receipt.get("epoch_id", -2)):
                errors.append("attested_allocation_attestation_epoch_diverged")
            if candidate_attestation.get("enclave_pubkey") != candidate_receipt.get("enclave_pubkey"):
                errors.append("attested_allocation_attestation_pubkey_diverged")
            if expected_pcr0_for_receipt and candidate_attestation.get("pcr0") != expected_pcr0_for_receipt:
                errors.append("attested_allocation_independent_pcr0_diverged")
        except Exception as exc:
            errors.append("attested_allocation_nitro_verification_failed:%s" % str(exc)[:120])

    gateway_pcr0 = str(document.get("gateway_pcr0") or "").lower()
    if attestation_data.get("pcr0") != gateway_pcr0:
        errors.append("attested_allocation_gateway_pcr0_diverged")

    if expected_gateway_commit and receipt.get("commit_sha") != expected_gateway_commit:
        errors.append("attested_allocation_gateway_commit_diverged")
    persistence_ready = document.get("persistence_status") == "persisted"
    required_ready = (
        not errors
        and all_gateway_identities_supplied
        and bool(lineage_complete)
        and persistence_ready
    )
    return {
        "passed": not errors,
        "required_ready": required_ready,
        "errors": errors,
        "allocation_verification": allocation_verification,
        "receipt_hash": receipt.get("receipt_hash"),
        "receipt": dict(receipt),
        "gateway_pcr0": attestation_data.get("pcr0"),
        "gateway_commit": receipt.get("commit_sha"),
        "independent_gateway_identity_supplied": all_gateway_identities_supplied,
        "lineage_complete": bool(lineage_complete),
        "lineage_receipt_count": len(parent_receipts),
        "persistence_ready": persistence_ready,
        "pcr0_verification_mode": (
            "independent_expected_pcr0"
            if all_gateway_identities_supplied
            else "aws_signature_only"
        ),
    }


def verify_research_lab_evaluation_bundle_page(
    page: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
    required_score_bundle_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.evaluation_verify_enabled:
        errors.append("validator_evaluation_verify_disabled")
    errors.extend(validator_flags.enabled_mutation_flags())
    if _evaluation_page_contains_secret_material(page):
        errors.append("evaluation_bundle_page_contains_raw_secret_material")
    if page.get("on_chain_submission_allowed"):
        errors.append("evaluation_bundle_page_must_not_allow_on_chain_submission")
    if page.get("bundle_type") != "research_lab_evaluation_score_bundle_page":
        errors.append("unexpected_evaluation_bundle_page_type")

    rows = page.get("score_bundles", [])
    if not isinstance(rows, list):
        errors.append("score_bundles_must_be_array")
        rows = []

    verified_inputs: list[dict[str, Any]] = []
    bundle_results: list[dict[str, Any]] = []
    verified_score_bundle_ids: set[str] = set()
    ignored_bundle_count = 0
    for row in rows:
        if not isinstance(row, Mapping):
            errors.append("score_bundle_row_must_be_object")
            continue
        if not _row_is_scored_bundle(row):
            ignored_bundle_count += 1
            continue
        bundle = row.get("score_bundle_doc", row)
        if not isinstance(bundle, Mapping):
            errors.append("score_bundle_doc_must_be_object")
            continue
        verification = verify_research_evaluation_score_bundle(bundle)
        bundle_results.append(verification)
        if not verification["passed"]:
            errors.extend(f"score_bundle:{verification.get('score_bundle_hash')}:{error}" for error in verification["errors"])
            continue
        score_bundle_hash = str(verification.get("score_bundle_hash") or bundle.get("score_bundle_hash") or "")
        canonical_score_bundle_id = (
            "score_bundle:" + score_bundle_hash.split(":", 1)[1]
            if score_bundle_hash.startswith("sha256:")
            else ""
        )
        row_score_bundle_id = str(row.get("score_bundle_id") or bundle.get("score_bundle_id") or "").strip()
        if row_score_bundle_id and canonical_score_bundle_id and row_score_bundle_id != canonical_score_bundle_id:
            errors.append(f"score_bundle_id_hash_mismatch:{row_score_bundle_id}")
            continue
        if canonical_score_bundle_id:
            verified_score_bundle_ids.add(canonical_score_bundle_id)
        try:
            verified_inputs.append(score_bundle_to_weight_input(bundle))
        except Exception as exc:
            errors.append(f"score_bundle_weight_input_failed:{str(exc)[:120]}")

    required_ids = []
    seen_required: set[str] = set()
    for value in required_score_bundle_ids or ():
        score_bundle_id = str(value or "").strip()
        if score_bundle_id.startswith("score_bundle:") and score_bundle_id not in seen_required:
            seen_required.add(score_bundle_id)
            required_ids.append(score_bundle_id)
    missing_required = [
        score_bundle_id
        for score_bundle_id in required_ids
        if score_bundle_id not in verified_score_bundle_ids
    ]
    errors.extend(
        f"missing_required_evaluation_score_bundle:{score_bundle_id}"
        for score_bundle_id in missing_required
    )

    if validator_flags.require_evaluation_verification_before_submit and not verified_inputs:
        errors.append("no_verified_evaluation_score_bundles")

    return {
        "passed": not errors,
        "errors": errors,
        "epoch": page.get("epoch"),
        "bundle_count": len(rows),
        "ignored_bundle_count": ignored_bundle_count,
        "verified_bundle_count": len(verified_inputs),
        "verified_score_bundle_ids": sorted(verified_score_bundle_ids),
        "required_score_bundle_ids": required_ids,
        "missing_required_score_bundle_ids": missing_required,
        "verified_weight_inputs": verified_inputs,
        "bundle_results": bundle_results,
        "on_chain_submission_allowed": False,
    }


def verify_research_lab_audit_bundle(
    bundle_or_row: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validator_flags = flags if isinstance(flags, ResearchLabValidatorFlags) else ResearchLabValidatorFlags.from_mapping(flags)
    errors: list[str] = []
    if not validator_flags.audit_verify_enabled:
        errors.append("validator_audit_verify_disabled")
    errors.extend(validator_flags.enabled_mutation_flags())

    bundle = bundle_or_row.get("bundle_doc") if isinstance(bundle_or_row.get("bundle_doc"), Mapping) else bundle_or_row
    if _contains_secret_material(bundle):
        errors.append("audit_bundle_contains_private_or_secret_material")
    if bundle.get("bundle_type") not in {"research_lab_signed_audit_bundle", "research_lab_audit_bundle_preview"}:
        errors.append("unexpected_audit_bundle_type")
    if bundle.get("on_chain_submission_allowed"):
        errors.append("audit_bundle_must_not_allow_on_chain_submission")
    if not bundle.get("read_only"):
        errors.append("audit_bundle_must_be_read_only")

    source_state = bundle.get("source_state")
    source_state_hash = bundle.get("source_state_hash")
    if not isinstance(source_state, Mapping) or not source_state_hash:
        errors.append("audit_bundle_source_state_required")
        source_state = {}
    elif sha256_json(source_state) != source_state_hash:
        errors.append("audit_bundle_source_state_hash_diverged")

    score_rows = source_state.get("score_bundle_rows", []) if isinstance(source_state, Mapping) else []
    verified_score_bundles = 0
    if isinstance(score_rows, list):
        for row in score_rows:
            if not isinstance(row, Mapping):
                errors.append("audit_score_bundle_row_must_be_object")
                continue
            bundle_doc = row.get("score_bundle_doc", row)
            if not isinstance(bundle_doc, Mapping):
                errors.append("audit_score_bundle_doc_must_be_object")
                continue
            verification = verify_research_evaluation_score_bundle(bundle_doc)
            if not verification["passed"]:
                errors.extend(f"audit_score_bundle:{error}" for error in verification["errors"])
            else:
                verified_score_bundles += 1
    else:
        errors.append("audit_score_bundle_rows_must_be_array")

    return {
        "passed": not errors,
        "errors": errors,
        "epoch": bundle.get("epoch"),
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": source_state_hash,
        "verified_score_bundle_count": verified_score_bundles,
        "on_chain_submission_allowed": False,
    }


def build_research_lab_weight_component(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verification = verify_research_lab_shadow_bundle(bundle, flags=flags)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    weight_vector = bundle["weight_vector"]
    return {
        "epoch": int(bundle["epoch"]),
        "shadow_only": True,
        "read_only": True,
        "submission_allowed": False,
        "on_chain_submission_allowed": False,
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": bundle.get("source_state_hash"),
        "u16_weights": dict(weight_vector.get("u16_weights", {})),
        "weight_sum": int(weight_vector.get("weight_sum", 0)),
        "weight_vector_hash": verification["weight_vector_hash"],
    }


def build_research_lab_allocation_component(
    bundle: Mapping[str, Any],
    *,
    flags: ResearchLabValidatorFlags | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    verification = verify_research_lab_allocation_bundle(bundle, flags=flags)
    if not verification["passed"]:
        raise ValueError("; ".join(verification["errors"]))
    allocation_doc = dict(verification["allocation_doc"])
    return {
        "epoch": int(bundle["epoch"]),
        "shadow_only": False,
        "read_only": False,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "bundle_id": bundle.get("bundle_id"),
        "source_state_hash": bundle.get("source_state_hash"),
        "allocation_hash": verification["allocation_hash"],
        "allocation_doc": allocation_doc,
        "observability": dict(bundle.get("observability") or {}),
    }


def build_on_chain_weight_payload(_component: Mapping[str, Any]) -> dict[str, Any]:
    raise RuntimeError("Research Lab shadow verification cannot build on-chain weight payloads")


def write_research_lab_validator_artifact(
    *,
    output_dir: Path | str,
    epoch: int,
    bundle: Mapping[str, Any],
    verification: Mapping[str, Any],
    component: Mapping[str, Any] | None = None,
    artifact_kind: str = "shadow",
) -> Path:
    """Write the validator's local Research Lab verification artifact.

    This is a local file only; it is not a production database write and never
    contains raw provider keys.
    """
    artifact_bundle = (
        _sanitize_evaluation_artifact(bundle)
        if artifact_kind == "evaluation"
        else dict(bundle)
    )
    if _contains_secret_material(artifact_bundle) or _contains_secret_material(verification) or _contains_secret_material(component or {}):
        raise ValueError("Research Lab validator artifact contains raw secret material")
    safe_kind = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in artifact_kind).strip("_") or "shadow"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifact_path = output_path / f"research_lab_{safe_kind}_epoch_{int(epoch)}.json"
    payload = {
        "schema_version": "1.0",
        "artifact_type": f"research_lab_validator_{safe_kind}_verification",
        "epoch": int(epoch),
        "bundle": artifact_bundle,
        "verification": dict(verification),
        "component": dict(component or {}),
    }
    artifact_path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")
    return artifact_path


def verify_research_lab_validator_integration(path: Path | str | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path else FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        fixture = json.load(handle)

    shadow_fixture = load_production_shadow_fixture()
    shadow_result = build_controlled_production_shadow(shadow_fixture["shadow_case"])
    bundle = shadow_result["bundle"]
    bundle = {
        **bundle,
        "weight_vector_hash": sha256_json(bundle["weight_vector"]),
    }

    verification = verify_research_lab_shadow_bundle(bundle, flags=fixture["validator_flags"])
    if not verification["passed"]:
        raise AssertionError("; ".join(verification["errors"]))

    component = build_research_lab_weight_component(bundle, flags=fixture["validator_flags"])
    try:
        build_on_chain_weight_payload(component)
        raise AssertionError("shadow component built an on-chain payload")
    except RuntimeError:
        pass

    unsafe = verify_research_lab_shadow_bundle(bundle, flags=fixture["unsafe_validator_flags"])
    if unsafe["passed"] or not unsafe["errors"]:
        raise AssertionError("unsafe mutation flags were not rejected")

    tampered_source = {
        **bundle,
        "source_state": {"tampered": True},
        "source_state_hash": sha256_json({"tampered": False}),
    }
    tampered = verify_research_lab_shadow_bundle(tampered_source, flags=fixture["validator_flags"])
    if tampered["passed"] or "source_state_hash_diverged" not in tampered["errors"]:
        raise AssertionError("source-state hash tamper was not rejected")

    eval_bundle = build_research_evaluation_score_bundle(
        run_id="11111111-1111-4111-8111-111111111111",
        ticket_id="22222222-2222-4222-8222-222222222222",
        miner_hotkey="5FevalMiner111111111111111111111111111111111",
        island="generalist",
        evaluation_epoch=int(bundle["epoch"]),
        parent_artifact_hash="sha256:" + "1" * 64,
        candidate_artifact_hash="sha256:" + "2" * 64,
        private_model_manifest_hash="sha256:" + "3" * 64,
        candidate_patch_hash="sha256:" + "4" * 64,
        icp_set_hash="sha256:" + "5" * 64,
        scoring_version="qualification-company-scorer:v1",
        evaluator_version="research-lab-private-evaluator:v1",
        per_icp_results=[
            {
                "icp_ref": "icp:a",
                "icp_hash": "sha256:" + "a" * 64,
                "base_company_scores": [80, 60],
                "candidate_company_scores": [90, 70],
            }
        ],
        evidence_bundle_refs=["evidence_bundle:sha256:" + "6" * 64],
        execution_trace_ref="execution_trace:11111111-1111-4111-8111-111111111111",
        cost_ledger_ref="cost_ledger:sha256:" + "7" * 64,
        benchmark_split_ref="sealed_benchmark:qualification:intent:v1",
        policy={
            "min_delta": 2.0,
            "min_successful_icps": 1,
            "min_candidate_score": 15.0,
            "observed_cost_usd": 1.25,
        },
        signature_ref="kms-signature:research-lab-eval:test",
    )
    eval_page = {
        "schema_version": "1.0",
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": int(bundle["epoch"]),
        "score_bundles": [
            {"bundle_status": "rejected", "current_event_status": "rejected", "score_bundle_doc": eval_bundle},
            {"bundle_status": "scored", "current_event_status": "scored", "score_bundle_doc": eval_bundle},
        ],
        "on_chain_submission_allowed": False,
    }
    eval_verification = verify_research_lab_evaluation_bundle_page(
        eval_page,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
    )
    if not eval_verification["passed"]:
        raise AssertionError("evaluation score-bundle page did not verify: " + "; ".join(eval_verification["errors"]))
    if eval_verification["verified_bundle_count"] != 1 or eval_verification["ignored_bundle_count"] != 1:
        raise AssertionError("evaluation verification did not ignore non-scored score bundles")

    tampered_eval = {
        **eval_page,
        "score_bundles": [
            {
                "score_bundle_doc": {
                    **eval_bundle,
                    "aggregates": {**eval_bundle["aggregates"], "candidate_score": 999.0},
                }
            }
        ],
    }
    tampered_eval_verification = verify_research_lab_evaluation_bundle_page(
        tampered_eval,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
    )
    if tampered_eval_verification["passed"] or not tampered_eval_verification["errors"]:
        raise AssertionError("tampered evaluation score bundle was not rejected")

    allocation_policy = {
        "policy_id": "research_lab_reimbursement_and_champion_rewards:v1",
        "research_lab_emission_percent": 20.0,
        "reward_epochs": 20,
        "usd_per_0_1_percent_epoch": 0.162,
        "champion_threshold_points": 2.0,
        "champion_min_alpha_percent": 4.0,
        "champion_extra_alpha_percent_per_point": 0.2,
        "champion_max_alpha_percent": 10.0,
    }
    reimbursement_obligations = [
        {
            "uid": 3,
            "miner_hotkey": "5FreimbMiner1111111111111111111111111111111",
            "source_id": "reimbursement_schedule:test",
            "schedule_id": "reimbursement_schedule:test",
            "award_id": "reimbursement_award:test",
            "run_id": "33333333-3333-4333-8333-333333333333",
            "island": "generalist",
            "status": "active",
            "start_epoch": int(bundle["epoch"]),
            "epoch_count": 20,
            "target_reimbursement_microusd": 3_240_000,
        }
    ]
    eval_score_bundle_id = "score_bundle:" + eval_bundle["score_bundle_hash"].split(":", 1)[1]
    champion_obligations = [
        {
            "uid": 4,
            "miner_hotkey": "5FchampMiner1111111111111111111111111111111",
            "source_id": "champion_reward:test",
            "champion_reward_id": "champion_reward:test",
            "candidate_id": "candidate:" + "9" * 64,
            "score_bundle_id": eval_score_bundle_id,
            "run_id": "44444444-4444-4444-8444-444444444444",
            "island": "generalist",
            "status": "active",
            "start_epoch": int(bundle["epoch"]),
            "epoch_count": 20,
            "improvement_points": 3.0,
            "desired_alpha_percent": 2.1,
        }
    ]
    allocation_doc = allocate_research_lab_epoch(
        int(bundle["epoch"]),
        allocation_policy,
        reimbursement_obligations,
        champion_obligations,
    )
    allocation_source_state = {
        "epoch": int(bundle["epoch"]),
        "netuid": 401,
        "policy_id": "research_lab_reimbursement_and_champion_rewards:v1",
        "policy": allocation_policy,
        "reimbursement_obligations": reimbursement_obligations,
        "champion_obligations": champion_obligations,
    }
    allocation_bundle = {
        "schema_version": "1.0",
        "bundle_type": "research_lab_live_allocation_bundle",
        "bundle_id": "research_lab_allocation_bundle:" + allocation_doc["allocation_hash"].split(":", 1)[1],
        "epoch": int(bundle["epoch"]),
        "netuid": 401,
        "submission_allowed": True,
        "on_chain_submission_allowed": True,
        "source_state": allocation_source_state,
        "source_state_hash": sha256_json(allocation_source_state),
        "allocation_doc": allocation_doc,
        "allocation_hash": allocation_doc["allocation_hash"],
    }
    allocation_flags = {
        **fixture["validator_flags"],
        "RESEARCH_LAB_VALIDATOR_FETCH_ENABLED": True,
        "RESEARCH_LAB_REIMBURSEMENTS_ENABLED": True,
    }
    allocation_verification = verify_research_lab_allocation_bundle(
        allocation_bundle,
        flags=allocation_flags,
    )
    if not allocation_verification["passed"]:
        raise AssertionError("live allocation bundle did not verify: " + "; ".join(allocation_verification["errors"]))
    allocation_component = build_research_lab_allocation_component(allocation_bundle, flags=allocation_flags)
    if not allocation_component["on_chain_submission_allowed"]:
        raise AssertionError("live allocation component did not allow on-chain submission")
    if float(allocation_component["allocation_doc"].get("reimbursement_alpha_percent") or 0.0) <= 0:
        raise AssertionError("live allocation did not include reimbursement alpha")
    if float(allocation_component["allocation_doc"].get("champion_alpha_percent") or 0.0) <= 0:
        raise AssertionError("live allocation did not include champion alpha")
    required_score_bundle_ids = allocation_referenced_score_bundle_ids(allocation_bundle)
    if required_score_bundle_ids != [eval_score_bundle_id]:
        raise AssertionError(f"allocation referenced score bundle ids diverged: {required_score_bundle_ids}")

    empty_current_epoch_page = {
        "schema_version": "1.0",
        "bundle_type": "research_lab_evaluation_score_bundle_page",
        "epoch": int(bundle["epoch"]),
        "score_bundles": [],
        "on_chain_submission_allowed": False,
    }
    missing_required_verification = verify_research_lab_evaluation_bundle_page(
        empty_current_epoch_page,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
        required_score_bundle_ids=required_score_bundle_ids,
    )
    if missing_required_verification["passed"] or eval_score_bundle_id not in missing_required_verification["missing_required_score_bundle_ids"]:
        raise AssertionError("missing allocation-referenced score bundle was not rejected")

    referenced_page = merge_research_lab_evaluation_bundle_page(
        empty_current_epoch_page,
        [
            {
                "score_bundle_id": eval_score_bundle_id,
                "bundle_status": "scored",
                "current_event_status": "scored",
                "score_bundle_doc": eval_bundle,
            }
        ],
    )
    referenced_verification = verify_research_lab_evaluation_bundle_page(
        referenced_page,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
        required_score_bundle_ids=required_score_bundle_ids,
    )
    if not referenced_verification["passed"]:
        raise AssertionError(
            "allocation-referenced score bundle page did not verify: "
            + "; ".join(referenced_verification["errors"])
        )
    if referenced_verification["verified_score_bundle_ids"] != [eval_score_bundle_id]:
        raise AssertionError("allocation-referenced score bundle id was not verified")
    replacement_page = merge_research_lab_evaluation_bundle_page(
        {
            **empty_current_epoch_page,
            "score_bundles": [
                {
                    "score_bundle_id": eval_score_bundle_id,
                    "bundle_status": "scored",
                    "current_event_status": "scored",
                    "score_bundle_doc": {
                        **eval_bundle,
                        "aggregates": {**eval_bundle["aggregates"], "candidate_score": 999.0},
                    },
                }
            ],
        },
        [
            {
                "score_bundle_id": eval_score_bundle_id,
                "bundle_status": "scored",
                "current_event_status": "scored",
                "score_bundle_doc": eval_bundle,
            }
        ],
    )
    replacement_verification = verify_research_lab_evaluation_bundle_page(
        replacement_page,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
        required_score_bundle_ids=required_score_bundle_ids,
    )
    if not replacement_verification["passed"]:
        raise AssertionError(
            "allocation-referenced score bundle replacement did not verify: "
            + "; ".join(replacement_verification["errors"])
        )
    mismatched_row_id_page = merge_research_lab_evaluation_bundle_page(
        empty_current_epoch_page,
        [
            {
                "score_bundle_id": "score_bundle:" + "0" * 64,
                "bundle_status": "scored",
                "current_event_status": "scored",
                "score_bundle_doc": eval_bundle,
            }
        ],
    )
    mismatched_row_id_verification = verify_research_lab_evaluation_bundle_page(
        mismatched_row_id_page,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_EVALUATION_VERIFY_ENABLED": True},
        required_score_bundle_ids=required_score_bundle_ids,
    )
    if mismatched_row_id_verification["passed"] or not any(
        str(error).startswith("score_bundle_id_hash_mismatch:")
        for error in mismatched_row_id_verification["errors"]
    ):
        raise AssertionError("mismatched score bundle row id was not rejected")

    tampered_allocation = {
        **allocation_bundle,
        "allocation_doc": {
            **allocation_doc,
            "reimbursement_alpha_percent": 9.99,
        },
    }
    tampered_allocation_verification = verify_research_lab_allocation_bundle(
        tampered_allocation,
        flags=allocation_flags,
    )
    if tampered_allocation_verification["passed"] or "allocation_hash_diverged" not in tampered_allocation_verification["errors"]:
        raise AssertionError("tampered live allocation bundle was not rejected")

    rehashed_bad_allocation_doc = {
        **allocation_doc,
        "reimbursement_alpha_percent": 9.99,
    }
    rehashed_bad_payload = {k: v for k, v in rehashed_bad_allocation_doc.items() if k != "allocation_hash"}
    rehashed_bad_allocation_hash = sha256_json(rehashed_bad_payload)
    rehashed_bad_allocation_doc["allocation_hash"] = rehashed_bad_allocation_hash
    rehashed_bad_allocation = {
        **allocation_bundle,
        "allocation_doc": rehashed_bad_allocation_doc,
        "allocation_hash": rehashed_bad_allocation_hash,
    }
    rehashed_bad_verification = verify_research_lab_allocation_bundle(
        rehashed_bad_allocation,
        flags=allocation_flags,
    )
    if rehashed_bad_verification["passed"] or "allocation_recompute_hash_diverged" not in rehashed_bad_verification["errors"]:
        raise AssertionError("rehashed live allocation tamper was not rejected")

    oversized_policy = {**allocation_policy, "research_lab_emission_percent": 100.0}
    oversized_allocation_doc = allocate_research_lab_epoch(
        int(bundle["epoch"]),
        oversized_policy,
        reimbursement_obligations,
        champion_obligations,
    )
    oversized_source_state = {**allocation_source_state, "policy": oversized_policy}
    oversized_allocation = {
        **allocation_bundle,
        "source_state": oversized_source_state,
        "source_state_hash": sha256_json(oversized_source_state),
        "allocation_doc": oversized_allocation_doc,
        "allocation_hash": oversized_allocation_doc["allocation_hash"],
    }
    oversized_verification = verify_research_lab_allocation_bundle(
        oversized_allocation,
        flags=allocation_flags,
    )
    if oversized_verification["passed"] or "allocation_lab_cap_exceeds_validator_policy" not in oversized_verification["errors"]:
        raise AssertionError("oversized Research Lab allocation cap was not rejected")

    audit_source_state = {
        "candidate_rows": [
            {
                "candidate_id": "candidate:" + "8" * 64,
                "candidate_artifact_hash": eval_bundle["candidate_artifact_hash"],
                "candidate_patch_hash": eval_bundle["candidate_patch_hash"],
                "private_model_manifest_hash": eval_bundle["private_model_manifest_hash"],
                "current_candidate_status": "scored",
                "current_score_bundle_id": "score_bundle:" + eval_bundle["score_bundle_hash"].split(":", 1)[1],
            }
        ],
        "scoring_dispatch_event_rows": [
            {
                "dispatch_type": "candidate_scoring",
                "dispatch_status": "scored",
                "rolling_window_hash": eval_bundle["icp_set_hash"],
                "score_bundle_id": "score_bundle:" + eval_bundle["score_bundle_hash"].split(":", 1)[1],
                "worker_ref": "gateway-qualification-worker:test",
                "proxy_ref_hash": "sha256:" + "9" * 64,
            }
        ],
        "score_bundle_rows": [
            {
                "bundle_status": "scored",
                "current_event_status": "scored",
                "score_bundle_doc": eval_bundle,
            }
        ],
    }
    audit_bundle = {
        "schema_version": "1.0",
        "bundle_type": "research_lab_signed_audit_bundle",
        "bundle_id": "research_lab_audit:" + "a" * 64,
        "epoch": int(bundle["epoch"]),
        "read_only": True,
        "on_chain_submission_allowed": False,
        "source_state": audit_source_state,
        "source_state_hash": sha256_json(audit_source_state),
        "signature_ref": "kms-signature:research-lab-audit:test",
    }
    audit_verification = verify_research_lab_audit_bundle(
        audit_bundle,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_AUDIT_VERIFY_ENABLED": True},
    )
    if not audit_verification["passed"]:
        raise AssertionError("audit bundle did not verify: " + "; ".join(audit_verification["errors"]))

    unsafe_audit = {
        **audit_bundle,
        "source_state": {**audit_source_state, "private_model_manifest_doc": {"image_digest": "123.dkr.ecr.us-east-1.amazonaws.com/x@sha256:" + "1" * 64}},
        "source_state_hash": sha256_json({**audit_source_state, "private_model_manifest_doc": {"image_digest": "123.dkr.ecr.us-east-1.amazonaws.com/x@sha256:" + "1" * 64}}),
    }
    unsafe_audit_verification = verify_research_lab_audit_bundle(
        unsafe_audit,
        flags={**fixture["validator_flags"], "RESEARCH_LAB_VALIDATOR_AUDIT_VERIFY_ENABLED": True},
    )
    if unsafe_audit_verification["passed"] or "audit_bundle_contains_private_or_secret_material" not in unsafe_audit_verification["errors"]:
        raise AssertionError("audit bundle private artifact leak was not rejected")

    return {
        "bundle_id": bundle["bundle_id"],
        "epoch": bundle["epoch"],
        "weight_vector_hash": component["weight_vector_hash"],
        "weight_sum": component["weight_sum"],
        "unsafe_errors": unsafe["errors"],
        "evaluation_bundle_count": eval_verification["verified_bundle_count"],
        "audit_score_bundle_count": audit_verification["verified_score_bundle_count"],
        "allocation_hash": allocation_component["allocation_hash"],
        "allocation_reimbursement_alpha_percent": allocation_component["allocation_doc"]["reimbursement_alpha_percent"],
        "allocation_champion_alpha_percent": allocation_component["allocation_doc"]["champion_alpha_percent"],
    }


def _contains_secret_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered_key = str(key).lower()
            if any(
                marker in lowered_key
                for marker in (
                    "api_key",
                    "raw_secret",
                    "raw_openrouter",
                    "credential",
                    "private_model_manifest_doc",
                    "candidate_patch_manifest",
                    "image_digest",
                    "proxy_url",
                )
            ):
                return True
            if _contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(_contains_secret_material(item) for item in value)
    elif isinstance(value, str):
        lowered = value.lower()
        return any(
            marker in lowered
            for marker in (
                "sk-or-",
                "raw_openrouter_key",
                "openrouter_api_key",
                "raw_secret",
                "hidden_icp",
                "icp_plaintext",
                ".dkr.ecr.",
                "private_repo",
                "judge_prompt",
            )
        )
    return False


def _evaluation_page_contains_secret_material(value: Any) -> bool:
    """Scan page metadata while score-bundle docs use their schema verifier."""

    if isinstance(value, Mapping):
        if value.get("bundle_type") == "research_lab_evaluation_score_bundle":
            return False
        for key, item in value.items():
            if key == "score_bundle_doc" and isinstance(item, Mapping):
                continue
            if _contains_secret_material({key: None}):
                return True
            if _evaluation_page_contains_secret_material(item):
                return True
    elif isinstance(value, list):
        return any(_evaluation_page_contains_secret_material(item) for item in value)
    return False


def _sanitize_evaluation_artifact(value: Any, *, _path: tuple[str, ...] = ()) -> Any:
    """Replace permitted private image locations with their existing public hash."""

    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            child_path = (*_path, str(key))
            if (
                child_path[-3:] in _SERVING_IMAGE_DIGEST_PATHS
                and isinstance(item, str)
                and IMMUTABLE_ECR_IMAGE_RE.fullmatch(item)
                and not _contains_secret_material(item.replace(".dkr.ecr.", ".ecr."))
            ):
                sanitized["image_ref_hash"] = sha256_json({"image_ref": item})
                continue
            if (
                child_path[-3:] in _SERVING_MANIFEST_URI_PATHS
                and isinstance(item, str)
                and PRIVATE_S3_MANIFEST_RE.fullmatch(item)
                and not _contains_secret_material(item)
                and SHA256_RE.fullmatch(str(value.get("manifest_hash") or ""))
                and SHA256_RE.fullmatch(str(value.get("model_artifact_hash") or ""))
                and isinstance(value.get("image_digest"), str)
                and IMMUTABLE_ECR_IMAGE_RE.fullmatch(str(value["image_digest"]))
            ):
                sanitized["manifest_ref_hash"] = sha256_json({"manifest_ref": item})
                continue
            sanitized[str(key)] = _sanitize_evaluation_artifact(item, _path=child_path)
        return sanitized
    if isinstance(value, list):
        return [
            _sanitize_evaluation_artifact(item, _path=(*_path, str(index)))
            for index, item in enumerate(value)
        ]
    return value


def _row_is_scored_bundle(row: Mapping[str, Any]) -> bool:
    status = row.get("bundle_status")
    current_status = row.get("current_event_status") or row.get("current_bundle_status")
    if status is not None and status != "scored":
        return False
    if current_status is not None and current_status != "scored":
        return False
    bundle = row.get("score_bundle_doc", row)
    if isinstance(bundle, Mapping) and bundle.get("bundle_status") not in (None, "scored"):
        return False
    return True


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in TRUTHY_VALUES

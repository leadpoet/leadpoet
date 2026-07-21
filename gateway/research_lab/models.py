"""Pydantic models for the Research Lab gateway API."""

from __future__ import annotations

import re
import hashlib
import json
import time
import base64
from typing import Any, Literal, Optional, Union
from urllib.parse import urlsplit
from uuid import UUID

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from gateway.research_lab.config import DEFAULT_LOOP_START_FEE_USD
from research_lab.canonical import sha256_json


SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "openrouter_management_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)

SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|token|credential)", re.I)
MODEL_TIER_RE = r"^[A-Za-z0-9_.:/-]+$"


class SignedResearchLabRequest(BaseModel):
    miner_hotkey: str = Field(min_length=16)
    signature: str = Field(min_length=16)
    timestamp: int
    idempotency_key: str = Field(min_length=8, max_length=160)

    @model_validator(mode="after")
    def timestamp_is_fresh(self) -> "SignedResearchLabRequest":
        now = int(time.time())
        if abs(now - self.timestamp) > 300:
            raise ValueError("timestamp must be within 5 minutes")
        return self

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"signature"}, exclude_unset=True, mode="json")


class AttestedCredentialCiphertextV2(BaseModel):
    request_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    ciphertext_b64: str = Field(min_length=64, max_length=1024)

    @field_validator("ciphertext_b64")
    @classmethod
    def valid_ciphertext(cls, value: str) -> str:
        try:
            decoded = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("credential ciphertext must be base64") from exc
        if not decoded or len(decoded) > 768:
            raise ValueError("credential ciphertext is outside limit")
        return value


class ResearchLabTicketCreateRequest(SignedResearchLabRequest):
    island: str = Field(min_length=1, max_length=80)
    brief_sanitized_ref: str = Field(min_length=8, max_length=256)
    brief_public_summary: Optional[str] = Field(default=None, max_length=2000)
    requested_loop_count: int = Field(default=1, gt=0, le=100)
    loop_start_fee_required_usd: float = Field(default=DEFAULT_LOOP_START_FEE_USD, ge=0)
    research_model_tier: str = Field(default="default", min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    requested_compute_budget_usd: float = Field(default=5.0, ge=0)
    max_compute_budget_usd: float = Field(default=25.0, ge=0)
    miner_openrouter_key_ref: Optional[str] = Field(default=None, max_length=256)
    miner_openrouter_key_handling: Optional[str] = Field(default=None)

    @field_validator("brief_sanitized_ref", "brief_public_summary", "miner_openrouter_key_ref")
    @classmethod
    def no_raw_secret_refs(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @model_validator(mode="after")
    def valid_budget_bounds(self) -> "ResearchLabTicketCreateRequest":
        if self.max_compute_budget_usd and self.requested_compute_budget_usd > self.max_compute_budget_usd:
            raise ValueError("requested_compute_budget_usd cannot exceed max_compute_budget_usd")
        return self


class ResearchLabProbeRequest(SignedResearchLabRequest):
    ticket_id: UUID
    probe_ref: str = Field(min_length=8, max_length=256)

    @field_validator("probe_ref")
    @classmethod
    def no_raw_probe_material(cls, value: str) -> str:
        reject_secret_material(value)
        return value


class ResearchLabLoopDiagnosticsRequest(SignedResearchLabRequest):
    """Signed request for a miner's OWN per-candidate diagnostics.

    Ownership is enforced server-side (ticket must belong to the signing
    hotkey), so a miner can only read diagnostics for loops they paid for.
    """

    ticket_id: UUID
    candidate_id: Optional[str] = Field(default=None, min_length=8, max_length=256)


_SOURCE_ADD_AUTH_TYPES = {"none", "api_key_header", "api_key_query", "bearer"}
_SOURCE_ADD_RUNTIME_AUTH_KINDS = {"none", "header", "query", "bearer"}
_SOURCE_ADD_SECRET_QUERY_NAMES = {
    "access_token",
    "api-key",
    "api_key",
    "apikey",
    "key",
    "token",
}
_SOURCE_ADD_FORBIDDEN_HEADERS = {
    "authorization",
    "connection",
    "content-length",
    "cookie",
    "host",
    "proxy-authorization",
    "transfer-encoding",
    "x-api-key",
}


def _source_add_https_url(value: str, *, field_name: str) -> str:
    raw = str(value or "").strip().rstrip("/")
    parsed = urlsplit(raw)
    try:
        port = parsed.port or 443
    except ValueError as exc:
        raise ValueError(f"{field_name} has an invalid port") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or port != 443
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(f"{field_name} must be an HTTPS URL on port 443 without credentials or query data")
    if field_name in {"api_base_url", "base_url"} and parsed.path:
        _source_add_fixed_path(parsed.path)
    return raw


def _source_add_fixed_path(value: str) -> str:
    path = str(value or "").strip()
    if (
        not path.startswith("/")
        or "?" in path
        or "#" in path
        or "%" in path
        or "\\" in path
        or any(part in {".", ".."} for part in path.split("/"))
        or any(character in path for character in "{}<>[]")
        or any(ord(character) < 32 or ord(character) == 127 for character in path)
        or any(character.isspace() for character in path)
    ):
        raise ValueError("SOURCE_ADD endpoint path must be fixed, relative, and safe")
    return path


def _bounded_source_add_json(value: Any) -> Any:
    node_count = 0

    def visit(item: Any, *, depth: int) -> None:
        nonlocal node_count
        node_count += 1
        if depth > 12 or node_count > 2_000:
            raise ValueError("SOURCE_ADD probe JSON exceeds structural limits")
        if isinstance(item, dict):
            if len(item) > 500:
                raise ValueError("SOURCE_ADD probe JSON has too many keys")
            for key, child in item.items():
                if not isinstance(key, str) or not key or len(key) > 120:
                    raise ValueError("SOURCE_ADD probe JSON key is invalid")
                visit(child, depth=depth + 1)
            return
        if isinstance(item, list):
            if len(item) > 500:
                raise ValueError("SOURCE_ADD probe JSON list is too large")
            for child in item:
                visit(child, depth=depth + 1)
            return
        if item is None or isinstance(item, (str, int, float, bool)):
            if isinstance(item, str) and len(item) > 4_096:
                raise ValueError("SOURCE_ADD probe JSON string is too large")
            return
        raise ValueError("SOURCE_ADD probe JSON contains an unsupported value")

    visit(value, depth=0)
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("SOURCE_ADD probe JSON is not canonicalizable") from exc
    if len(encoded) > 65_536:
        raise ValueError("SOURCE_ADD probe JSON exceeds 64 KiB")
    return value


class ResearchLabSourceEndpointExample(BaseModel):
    method: Literal["GET", "POST"]
    path: str = Field(min_length=1, max_length=300)
    purpose: str = Field(min_length=1, max_length=300)
    example_query: str = Field(min_length=1, max_length=500)

    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, value: Any) -> str:
        return str(value or "").strip().upper()

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _source_add_fixed_path(value)

    @field_validator("purpose", "example_query")
    @classmethod
    def no_secret_text(cls, value: str) -> str:
        reject_secret_material(value)
        return " ".join(value.strip().split())


class ResearchLabSourceMetadata(BaseModel):
    api_base_url: str = Field(min_length=8, max_length=500)
    documentation_url: str = Field(min_length=8, max_length=500)
    auth_type: str = Field(min_length=1, max_length=40)
    endpoint_examples: list[ResearchLabSourceEndpointExample] = Field(min_length=1, max_length=12)
    rate_limit_notes: str = Field(min_length=1, max_length=1000)
    data_provenance_notes: str = Field(default="", max_length=1000)
    third_party_refs: list[str] = Field(default_factory=list, max_length=8)

    @field_validator("api_base_url", "documentation_url")
    @classmethod
    def valid_https_url(cls, value: str, info: Any) -> str:
        return _source_add_https_url(value, field_name=str(info.field_name))

    @field_validator("third_party_refs")
    @classmethod
    def valid_third_party_refs(cls, value: list[str]) -> list[str]:
        return [
            _source_add_https_url(item, field_name="third_party_refs")
            for item in value
        ]

    @field_validator("auth_type")
    @classmethod
    def valid_auth_type(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in _SOURCE_ADD_AUTH_TYPES:
            raise ValueError("invalid SOURCE_ADD auth_type")
        return normalized

    @field_validator("rate_limit_notes", "data_provenance_notes")
    @classmethod
    def metadata_text_has_no_secret(cls, value: str) -> str:
        reject_secret_material(value)
        return " ".join(value.strip().split())


class ResearchLabSourceAdapterSubmissionRequest(SignedResearchLabRequest):
    """Credential-free miner SOURCE_ADD submission."""

    manifest: dict[str, Any] = Field()
    source_brief: Optional[str] = Field(default=None, max_length=2000)
    source_metadata: ResearchLabSourceMetadata
    # Retained only to reject old clients explicitly. SecretStr prevents a
    # validation error or model representation from echoing plaintext.
    adapter_credential: Optional[SecretStr] = Field(default=None, min_length=8, max_length=512)
    adapter_credential_v2: Optional[AttestedCredentialCiphertextV2] = None

    @model_validator(mode="after")
    def miner_credentials_are_forbidden(self) -> "ResearchLabSourceAdapterSubmissionRequest":
        if self.adapter_credential is not None or self.adapter_credential_v2 is not None:
            raise ValueError("miners must not submit SOURCE_ADD API credentials")
        return self

    @field_validator("source_brief")
    @classmethod
    def brief_has_no_secret_material(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={"signature", "adapter_credential", "adapter_credential_v2"},
            exclude_unset=True,
            mode="json",
        )


class ResearchLabSourceAddCredentialRecipientRequest(SignedResearchLabRequest):
    adapter_id: str = Field(min_length=1, max_length=200)


class ResearchLabCredentialRecipientResponse(BaseModel):
    schema_version: str
    purpose: str
    request_id: str
    boot_identity_hash: str
    miner_hotkey_hash: str
    adapter_ref_hash: str
    credential_ref: str
    key_ref_hash: str
    recipient_public_key_hash: str
    request_nonce: str
    recipient_public_key_der_b64: str
    attestation_document_b64: str
    key_encryption_algorithm: str


class ResearchLabSourceAdapterSubmissionResponse(BaseModel):
    submission_id: str
    adapter_id: str
    stage: str
    credential_ref: Optional[str] = None
    precheck_status: Optional[str] = None
    precheck_reasons: list[str] = Field(default_factory=list)


class ResearchLabSourceAdapterRecheckResponse(BaseModel):
    submission_id: str
    adapter_id: str
    stage: str
    queue_status: str
    work_id: str
    precheck_status: Optional[str] = None
    precheck_reasons: list[str] = Field(default_factory=list)
    leg1_reward_status: str = "not_evaluated"
    reward_ref: Optional[str] = None
    start_epoch: Optional[int] = None


class ResearchLabSourceAddProbeSpec(BaseModel):
    method: Literal["GET", "POST"]
    path: str = Field(min_length=1, max_length=300)
    query: dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict, max_length=20
    )
    body_json: Optional[Union[dict[str, Any], list[Any]]] = None

    @field_validator("method", mode="before")
    @classmethod
    def normalize_method(cls, value: Any) -> str:
        return str(value or "").strip().upper()

    @field_validator("path")
    @classmethod
    def valid_path(cls, value: str) -> str:
        return _source_add_fixed_path(value)

    @field_validator("query")
    @classmethod
    def valid_query(cls, value: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        for name, item in value.items():
            key = str(name).strip()
            if (
                not key
                or len(key) > 120
                or key.lower() in _SOURCE_ADD_SECRET_QUERY_NAMES
                or len(str(item)) > 500
            ):
                raise ValueError("SOURCE_ADD probe query is invalid or secret-bearing")
            normalized[key] = item
        return normalized

    @field_validator("body_json")
    @classmethod
    def body_has_no_secret_material(cls, value: Any) -> Any:
        if value is not None:
            reject_secret_material(value)
            return _bounded_source_add_json(value)
        return None


class ResearchLabSourceAdapterProbeConfigureRequest(BaseModel):
    base_url: str = Field(min_length=8, max_length=500)
    auth_kind: str = Field(default="none", max_length=20)
    auth_name: Optional[str] = Field(default=None, max_length=120)
    request_headers: dict[str, str] = Field(default_factory=dict, max_length=16)
    probes: list[ResearchLabSourceAddProbeSpec] = Field(min_length=1, max_length=3)
    api_credential: Optional[SecretStr] = Field(default=None, min_length=1, max_length=65536)
    api_credential_v2: Optional[AttestedCredentialCiphertextV2] = None
    operator_notes: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def valid_probe_config(self) -> "ResearchLabSourceAdapterProbeConfigureRequest":
        self.base_url = _source_add_https_url(self.base_url, field_name="base_url")
        self.auth_kind = self.auth_kind.strip().lower()
        if self.auth_kind not in _SOURCE_ADD_RUNTIME_AUTH_KINDS:
            raise ValueError("invalid SOURCE_ADD auth_kind")
        self.auth_name = str(self.auth_name or "").strip() or None
        if self.auth_kind in {"header", "query"} and not self.auth_name:
            raise ValueError("auth_name is required for header/query auth")
        if self.auth_kind == "bearer" and not self.auth_name:
            self.auth_name = "Authorization"
        if self.api_credential is not None:
            raise ValueError("plaintext SOURCE_ADD credentials are not accepted")
        if self.auth_kind != "none" and self.api_credential_v2 is None:
            raise ValueError("authenticated SOURCE_ADD test requires an attested credential")
        if self.auth_kind == "none" and self.api_credential_v2 is not None:
            raise ValueError("credential supplied for unauthenticated SOURCE_ADD test")
        normalized_headers: dict[str, str] = {}
        normalized_header_names: set[str] = set()
        for name, item in self.request_headers.items():
            header = str(name).strip()
            normalized_name = header.lower()
            header_value = str(item)
            if (
                not header
                or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,79}", header)
                or normalized_name in normalized_header_names
                or normalized_name in _SOURCE_ADD_FORBIDDEN_HEADERS
                or (
                    self.auth_kind in {"header", "bearer"}
                    and self.auth_name
                    and normalized_name == self.auth_name.lower()
                )
                or len(header_value) > 500
                or any(
                    ord(character) < 32 or ord(character) == 127
                    for character in header_value
                )
            ):
                raise ValueError("SOURCE_ADD request header is unsafe")
            reject_secret_material(header_value)
            normalized_header_names.add(normalized_name)
            normalized_headers[header] = header_value
        self.request_headers = normalized_headers
        if self.operator_notes:
            reject_secret_material(self.operator_notes)
        return self


class ResearchLabSourceAdapterProbeConfigureResponse(BaseModel):
    submission_id: str
    adapter_id: str
    config_ref: str
    work_id: str
    stage: str
    queue_status: str


class ResearchLabSourceAdapterProvisionRequest(BaseModel):
    registry_provider_id: str = Field(min_length=2, max_length=80)
    provision_status: str = Field(default="provisioned_autoresearch_eligible", max_length=80)
    base_url: Optional[str] = Field(default=None, max_length=500)
    auth_kind: str = Field(default="none", max_length=20)
    auth_name: Optional[str] = Field(default=None, max_length=120)
    credential_env_refs: list[str] = Field(default_factory=list, max_length=8)
    api_credential: Optional[SecretStr] = Field(default=None, min_length=1, max_length=65536)
    api_credential_v2: Optional[AttestedCredentialCiphertextV2] = None
    cost_model: dict[str, Any] = Field(default_factory=dict)
    probe_endpoints: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    request_headers: dict[str, str] = Field(default_factory=dict, max_length=16)
    test_probes: list[ResearchLabSourceAddProbeSpec] = Field(default_factory=list, max_length=3)
    operator_notes: Optional[str] = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def no_legacy_credential_transport(self) -> "ResearchLabSourceAdapterProvisionRequest":
        if self.api_credential is not None:
            raise ValueError("plaintext SOURCE_ADD credentials are not accepted")
        if self.credential_env_refs:
            raise ValueError("SOURCE_ADD process-environment credentials are retired")
        return self

    @field_validator("registry_provider_id")
    @classmethod
    def valid_registry_provider_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized.replace("_", "").replace("-", "").isalnum():
            raise ValueError("registry_provider_id must be a slug")
        return normalized

    @field_validator("operator_notes")
    @classmethod
    def notes_have_no_secret_material(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @field_validator("cost_model", "probe_endpoints", "request_headers")
    @classmethod
    def provision_docs_have_no_secret_material(cls, value: Any) -> Any:
        reject_secret_material(value)
        return _bounded_source_add_json(value)


class ResearchLabSourceAdapterProvisionResponse(BaseModel):
    submission_id: str
    adapter_id: str
    catalog_id: str
    registry_provider_id: str
    provision_status: str
    provision_ref: str
    credential_ref: Optional[str] = None
    requested_provision_status: Optional[str] = None
    queue_status: Optional[str] = None
    work_id: Optional[str] = None


class ResearchLabOpenRouterKeyRegisterRequest(SignedResearchLabRequest):
    openrouter_api_key: Optional[str] = Field(default=None, min_length=1, max_length=512)
    openrouter_management_key: Optional[str] = Field(default=None, min_length=1, max_length=512)
    openrouter_api_key_v2: Optional[AttestedCredentialCiphertextV2] = None
    openrouter_management_key_v2: Optional[AttestedCredentialCiphertextV2] = None
    key_label: Optional[str] = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def complete_credential_pair(self) -> "ResearchLabOpenRouterKeyRegisterRequest":
        raw = (self.openrouter_api_key, self.openrouter_management_key)
        sealed = (self.openrouter_api_key_v2, self.openrouter_management_key_v2)
        if any(raw) and not all(raw):
            raise ValueError("OpenRouter plaintext credential pair is incomplete")
        if any(sealed) and not all(sealed):
            raise ValueError("OpenRouter attested credential pair is incomplete")
        if all(raw) and all(sealed):
            raise ValueError("OpenRouter credential transports are mutually exclusive")
        if not all(raw) and not all(sealed):
            raise ValueError("OpenRouter attested credential pair is required")
        return self

    @field_validator("key_label")
    @classmethod
    def label_has_no_secret_material(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    def signed_payload(self) -> dict[str, Any]:
        return self.model_dump(
            exclude={
                "signature",
                "openrouter_api_key",
                "openrouter_management_key",
            },
            exclude_unset=True,
            mode="json",
        )


class ResearchLabOpenRouterCredentialRecipientRequest(SignedResearchLabRequest):
    pass


class ResearchLabOpenRouterCredentialRecipientV2(BaseModel):
    schema_version: str
    purpose: str
    request_id: str
    boot_identity_hash: str
    miner_hotkey_hash: str
    credential_kind: str
    credential_slot: str
    recipient_public_key_hash: str
    request_nonce: str
    recipient_public_key_der_b64: str
    attestation_document_b64: str
    key_encryption_algorithm: str


class ResearchLabOpenRouterCredentialReleaseEvidenceV2(BaseModel):
    schema_version: str
    coordinator_boot_identity: dict[str, Any]
    release_channel_version_id: str
    release_channel_get_url: str
    release_channel_head_url: str


class ResearchLabOpenRouterCredentialRecipientsResponse(BaseModel):
    runtime: ResearchLabOpenRouterCredentialRecipientV2
    management: ResearchLabOpenRouterCredentialRecipientV2
    release_evidence: ResearchLabOpenRouterCredentialReleaseEvidenceV2


class ResearchLabResumeCreditBlockedRequest(SignedResearchLabRequest):
    # Optional explicit run_ids; when omitted, all of this miner's blocked_for_credit
    # runs are considered.
    run_ids: Optional[list[str]] = Field(default=None)


class ResearchLabLoopStartRequest(SignedResearchLabRequest):
    ticket_id: UUID
    payment_block_hash: Optional[str] = Field(default=None, min_length=8, max_length=160)
    payment_extrinsic_index: Optional[int] = Field(default=None, ge=0)
    credit_id: Optional[str] = Field(default=None, min_length=8, max_length=256)
    miner_openrouter_key_ref: str = Field(min_length=8, max_length=256)
    miner_openrouter_key_handling: str = Field(pattern="^(encrypted_ref|ephemeral_ref)$")
    miner_openrouter_preflight_status: str = Field(pattern="^(passed|failed|not_run)$")
    requested_loop_count: int = Field(default=1, gt=0, le=100)
    research_model_tier: Optional[str] = Field(default=None, min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    requested_compute_budget_usd: Optional[float] = Field(default=None, ge=0)
    max_compute_budget_usd: Optional[float] = Field(default=None, ge=0)

    @field_validator("miner_openrouter_key_ref", "payment_block_hash", "credit_id")
    @classmethod
    def no_raw_loop_start_secret(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @model_validator(mode="after")
    def valid_budget_bounds(self) -> "ResearchLabLoopStartRequest":
        using_credit = bool(self.credit_id)
        has_block = bool(self.payment_block_hash)
        has_extrinsic = self.payment_extrinsic_index is not None
        if using_credit and (has_block or has_extrinsic):
            raise ValueError("credit_id cannot be combined with payment block/extrinsic fields")
        if not using_credit and not (has_block and has_extrinsic):
            raise ValueError("loop start requires either credit_id or both payment_block_hash and payment_extrinsic_index")
        if has_block != has_extrinsic:
            raise ValueError("payment_block_hash and payment_extrinsic_index must be provided together")
        if (
            self.requested_compute_budget_usd is not None
            and self.max_compute_budget_usd is not None
            and self.requested_compute_budget_usd > self.max_compute_budget_usd
        ):
            raise ValueError("requested_compute_budget_usd cannot exceed max_compute_budget_usd")
        return self


class ResearchLabLoopTopUpRequest(SignedResearchLabRequest):
    ticket_id: UUID
    continue_from_run_id: Optional[UUID] = None
    payment_block_hash: str = Field(min_length=8, max_length=160)
    payment_extrinsic_index: int = Field(ge=0)
    additional_compute_budget_usd: float = Field(gt=0)
    research_model_tier: Optional[str] = Field(default=None, min_length=1, max_length=80, pattern=MODEL_TIER_RE)
    topup_reason: str = Field(default="promising_needs_topup", pattern="^(promising_needs_topup|manual_budget_increase)$")
    miner_openrouter_key_ref: str = Field(min_length=8, max_length=256)
    miner_openrouter_key_handling: str = Field(pattern="^(encrypted_ref|ephemeral_ref)$")
    miner_openrouter_preflight_status: str = Field(pattern="^(passed|failed|not_run)$")

    @field_validator("miner_openrouter_key_ref", "payment_block_hash")
    @classmethod
    def no_raw_topup_secret(cls, value: str) -> str:
        reject_secret_material(value)
        return value


class ResearchLabReceiptCreateRequest(BaseModel):
    internal_run_ref: str = Field(min_length=8, max_length=256)
    ticket_id: UUID
    trajectory_id: Optional[UUID] = None
    run_id: Optional[UUID] = None
    loop_start_payment_id: Optional[UUID] = None
    loop_start_credit_id: Optional[str] = Field(default=None, max_length=256)
    miner_hotkey: str = Field(min_length=16)
    island: str = Field(min_length=1, max_length=80)
    receipt_status: str = Field(pattern="^(queued|completed|failed|cancelled|tombstoned)$")
    loop_count: int = Field(default=1, gt=0)
    miner_openrouter_key_ref: Optional[str] = Field(default=None, max_length=256)
    provider_usage: list[dict[str, Any]] = Field(default_factory=list)
    cost_ledger: dict[str, Any] = Field(default_factory=dict)
    receipt_doc: dict[str, Any] = Field(default_factory=dict)
    public_receipt_ref: Optional[str] = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def no_secret_material(self) -> "ResearchLabReceiptCreateRequest":
        reject_secret_material(self.model_dump())
        return self


class ResearchLabCandidateArtifactCreateRequest(BaseModel):
    run_id: UUID
    ticket_id: UUID
    receipt_id: Optional[UUID] = None
    miner_hotkey: str = Field(min_length=16)
    island: str = Field(min_length=1, max_length=80)
    candidate_kind: str = Field(default="image_build", pattern="^image_build$")
    private_model_manifest: dict[str, Any]
    candidate_patch_manifest: dict[str, Any]
    candidate_model_manifest: Optional[dict[str, Any]] = None
    candidate_source_diff_hash: Optional[str] = Field(default=None, pattern=r"^sha256:[0-9a-f]{64}$")
    candidate_build_doc: dict[str, Any] = Field(default_factory=dict)
    hypothesis_doc: dict[str, Any] = Field(default_factory=dict)
    redacted_public_summary: str = Field(default="", max_length=2000)
    git_tree_id: Optional[str] = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )
    git_tree_node_id: Optional[str] = Field(
        default=None, pattern=r"^tree-node:[0-9a-f]{64}$"
    )
    git_tree_root_commit: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    git_tree_node_commit: Optional[str] = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    git_tree_lineage_hash: Optional[str] = Field(
        default=None, pattern=r"^sha256:[0-9a-f]{64}$"
    )

    @model_validator(mode="after")
    def no_secret_material(self) -> "ResearchLabCandidateArtifactCreateRequest":
        reject_secret_material(self.model_dump())
        if not self.candidate_model_manifest:
            raise ValueError("image_build candidate requires candidate_model_manifest")
        if not self.candidate_source_diff_hash:
            raise ValueError("image_build candidate requires candidate_source_diff_hash")
        source_diff_uri = str(self.candidate_build_doc.get("source_diff_artifact_uri") or "").strip()
        if not source_diff_uri:
            raise ValueError("image_build candidate requires candidate_build_doc.source_diff_artifact_uri")
        if self.candidate_build_doc.get("source_diff_artifact_error"):
            raise ValueError("image_build candidate source diff artifact persistence failed")
        tree_values = (
            self.git_tree_id,
            self.git_tree_node_id,
            self.git_tree_root_commit,
            self.git_tree_node_commit,
            self.git_tree_lineage_hash,
        )
        if any(tree_values) and not all(tree_values):
            raise ValueError("Git-tree candidate lineage must be complete")
        if all(tree_values):
            lineage = self.candidate_build_doc.get("git_tree")
            if not isinstance(lineage, dict):
                raise ValueError("Git-tree candidate build lineage is missing")
            composition = lineage.get("composition")
            if not isinstance(composition, dict):
                raise ValueError("Git-tree candidate composition is missing")
            if (
                lineage.get("tree_id") != self.git_tree_id
                or lineage.get("node_id") != self.git_tree_node_id
                or lineage.get("git_commit") != self.git_tree_node_commit
                or composition.get("root_git_commit")
                != self.git_tree_root_commit
                or sha256_json(lineage) != self.git_tree_lineage_hash
            ):
                raise ValueError("Git-tree candidate lineage commitment differs")
        return self


class ResearchLabScoreBundleCreateRequest(BaseModel):
    bundle_status: str = Field(default="scored", pattern="^(scored|failed|rejected|tombstoned)$")
    receipt_id: Optional[UUID] = None
    score_bundle: dict[str, Any]

    @model_validator(mode="after")
    def valid_score_bundle_shape(self) -> "ResearchLabScoreBundleCreateRequest":
        reject_secret_material(self.model_dump())
        bundle = self.score_bundle
        if bundle.get("bundle_type") != "research_lab_evaluation_score_bundle":
            raise ValueError("score_bundle must be a Research Lab evaluation score bundle")
        if bundle.get("schema_version") not in {"1.0", "1.1"}:
            raise ValueError("unsupported score bundle schema version")
        if not bundle.get("signature_ref"):
            raise ValueError("score bundle signature_ref is required")
        if bundle.get("score_bundle_hash") != bundle.get("anchored_hash"):
            raise ValueError("score_bundle_hash must match anchored_hash")
        if bundle.get("score_bundle_hash") != _score_bundle_hash(bundle):
            raise ValueError("score bundle hash mismatch")
        reward_path = bundle.get("reward_path") or {}
        if reward_path.get("eligible_for_crown") or reward_path.get("eligible_for_improvement_grant"):
            raise ValueError("score bundle cannot directly create crown or improvement-grant eligibility")
        return self


class ResearchLabCandidateEvaluationResultRequest(BaseModel):
    candidate_id: str = Field(pattern=r"^candidate:[0-9a-f]{64}$")
    candidate_status: str = Field(pattern="^(evaluating|scored|failed|rejected)$")
    evaluator_ref: Optional[str] = Field(default=None, max_length=256)
    reason: Optional[str] = Field(default=None, max_length=500)
    score_bundle: Optional[dict[str, Any]] = None
    result_doc: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_result_shape(self) -> "ResearchLabCandidateEvaluationResultRequest":
        reject_secret_material(self.model_dump())
        if self.candidate_status == "scored" and not self.score_bundle:
            raise ValueError("scored Research Lab candidate requires score_bundle")
        if self.score_bundle:
            bundle = self.score_bundle
            if bundle.get("bundle_type") != "research_lab_evaluation_score_bundle":
                raise ValueError("score_bundle must be a Research Lab evaluation score bundle")
            if bundle.get("schema_version") not in {"1.0", "1.1"}:
                raise ValueError("unsupported score bundle schema version")
            if not bundle.get("signature_ref"):
                raise ValueError("score bundle signature_ref is required")
            if bundle.get("score_bundle_hash") != bundle.get("anchored_hash"):
                raise ValueError("score_bundle_hash must match anchored_hash")
            if bundle.get("score_bundle_hash") != _score_bundle_hash(bundle):
                raise ValueError("score bundle hash mismatch")
        return self


class ResearchLabTicketResponse(BaseModel):
    ticket_id: str
    status: str
    event_id: str
    event_seq: int
    ticket_hash: str
    unpaid_expires_at: Optional[str] = None


class ResearchLabLoopStartResponse(BaseModel):
    ticket_id: str
    run_id: str
    payment_id: str
    payment_ref: str
    queued: bool
    credit_preserved: bool = False
    credit_id: Optional[str] = None
    status: str


class ResearchLabLoopTopUpResponse(BaseModel):
    ticket_id: str
    run_id: str
    continued_from_run_id: Optional[str] = None
    topup_payment_id: str
    payment_ref: str
    queued: bool
    credit_preserved: bool = False
    credit_id: Optional[str] = None
    status: str


class ResearchLabOpenRouterKeyRegisterResponse(BaseModel):
    key_ref: str
    preflight_status: str
    key_hash: str
    limit_remaining: Optional[Any] = None
    limit_reset: Optional[str] = None


class ResearchLabResumeCreditBlockedResponse(BaseModel):
    requeued: int
    still_blocked: int
    results: list[dict[str, Any]]


class ResearchLabReceiptResponse(BaseModel):
    receipt_id: str
    receipt_hash: str
    status: str


class ResearchLabCandidateArtifactResponse(BaseModel):
    candidate_id: str
    candidate_artifact_hash: str
    candidate_patch_hash: str
    status: str
    event_id: str
    event_seq: int


class ResearchLabCandidateEvaluationResultResponse(BaseModel):
    candidate_id: str
    status: str
    event_id: str
    event_seq: int
    score_bundle_id: Optional[str] = None
    receipt_finalized: bool = False


class ResearchLabScoreBundleResponse(BaseModel):
    score_bundle_id: str
    score_bundle_hash: str
    status: str
    event_id: str
    event_seq: int


def reject_secret_material(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if SECRET_KEY_RE.search(str(key)):
                raise ValueError(f"raw secret field is not allowed: {key}")
            reject_secret_material(item)
    elif isinstance(value, list):
        for item in value:
            reject_secret_material(item)
    elif isinstance(value, str):
        lowered = value.lower()
        if any(marker in lowered for marker in SECRET_MARKERS):
            raise ValueError("raw provider secret material is not allowed")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _score_bundle_hash(bundle: dict[str, Any]) -> str:
    excluded = {"score_bundle_hash", "anchored_hash", "signature", "signature_ref"}
    payload = {key: value for key, value in dict(bundle).items() if key not in excluded}
    return "sha256:" + hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

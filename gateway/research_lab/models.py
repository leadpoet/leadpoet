"""Pydantic models for the Research Lab gateway API."""

from __future__ import annotations

import re
import json
import time
import base64
from datetime import datetime
from typing import Any, Literal, Optional, Union
from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from research_lab.source_add import source_add_contains_credential_material


SECRET_MARKERS = (
    "sk-or-",
    "openrouter_api_key",
    "openrouter_management_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
)

SECRET_KEY_RE = re.compile(r"(?:api[_-]?key|raw[_-]?secret|raw[_-]?openrouter|token|credential)", re.I)


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

    @model_validator(mode="before")
    @classmethod
    def raw_request_contains_no_credentials(cls, value: Any) -> Any:
        scanned_value = value
        if isinstance(value, dict):
            scanned_value = dict(value)
            for field_name in (
                "adapter_credential",
                "adapter_credential_v2",
            ):
                if scanned_value.get(field_name) is None:
                    scanned_value.pop(field_name, None)
        if source_add_contains_credential_material(scanned_value):
            raise ValueError("SOURCE_ADD request contains credential material")
        return value

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


class ResearchLabSourceAddStatusRequest(SignedResearchLabRequest):
    """Signed request for one miner's private SOURCE_ADD status page."""

    request_kind: Literal["source_add_status_v1"]
    limit: int = Field(default=20, ge=1, le=50)
    cursor: Optional[str] = Field(
        default=None,
        pattern=r"^source_add_submission:[0-9a-f]{16}$",
    )


class ResearchLabSourceAddStatusItem(BaseModel):
    submission_id: str = Field(pattern=r"^source_add_submission:[0-9a-f]{16}$")
    source_name: str = Field(min_length=1, max_length=160)
    submitted_at: datetime
    updated_at: datetime
    decision_status: Literal["pending", "approved", "rejected"]
    decision_reason_code: Literal[
        "automated_checks_in_progress",
        "additional_review_needed",
        "source_credibility_not_verified",
        "submission_details_not_verified",
        "documentation_not_verified",
        "provenance_not_verified",
        "technical_validation_not_passed",
        "automated_checks_not_passed",
        "leg1_reward_pending",
        "leg1_reward_active",
        "leg1_reward_stopped",
    ]
    decision_reason: Literal[
        "Automated Source Add checks are still in progress.",
        "Automated verification was inconclusive and needs additional review.",
        "The source did not pass the public credibility checks.",
        "The submitted API details were incomplete or could not be verified.",
        "The public API documentation could not be verified.",
        "Independent public evidence for the source could not be verified.",
        "The source did not pass technical validation.",
        "The submission did not pass automated Source Add checks.",
        "The source passed automated checks. Leg 1 reward setup is in progress.",
        "The source passed automated checks and the Leg 1 reward is active.",
        "The source passed automated checks. Future Leg 1 reward payments have stopped.",
    ]
    reward_status: Literal[
        "not_decided",
        "not_eligible",
        "pending",
        "active",
        "stopped",
    ]
    alpha_percent: Optional[float] = Field(default=None, gt=0, le=100)
    reward_epochs: Optional[int] = Field(default=None, gt=0)
    start_epoch: Optional[int] = Field(default=None, ge=0)
    end_epoch: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def status_contract_is_consistent(self) -> "ResearchLabSourceAddStatusItem":
        reason_contract = {
            "automated_checks_in_progress": (
                "pending",
                "Automated Source Add checks are still in progress.",
            ),
            "additional_review_needed": (
                "pending",
                "Automated verification was inconclusive and needs additional review.",
            ),
            "source_credibility_not_verified": (
                "rejected",
                "The source did not pass the public credibility checks.",
            ),
            "submission_details_not_verified": (
                "rejected",
                "The submitted API details were incomplete or could not be verified.",
            ),
            "documentation_not_verified": (
                "rejected",
                "The public API documentation could not be verified.",
            ),
            "provenance_not_verified": (
                "rejected",
                "Independent public evidence for the source could not be verified.",
            ),
            "technical_validation_not_passed": (
                "rejected",
                "The source did not pass technical validation.",
            ),
            "automated_checks_not_passed": (
                "rejected",
                "The submission did not pass automated Source Add checks.",
            ),
            "leg1_reward_pending": (
                "approved",
                "The source passed automated checks. Leg 1 reward setup is in progress.",
            ),
            "leg1_reward_active": (
                "approved",
                "The source passed automated checks and the Leg 1 reward is active.",
            ),
            "leg1_reward_stopped": (
                "approved",
                "The source passed automated checks. Future Leg 1 reward payments have stopped.",
            ),
        }
        expected_status, expected_reason = reason_contract[self.decision_reason_code]
        if self.decision_status != expected_status or self.decision_reason != expected_reason:
            raise ValueError("SOURCE_ADD public decision fields are inconsistent")
        expected_reward_state = {
            "automated_checks_in_progress": "not_decided",
            "additional_review_needed": "not_decided",
            "source_credibility_not_verified": "not_eligible",
            "submission_details_not_verified": "not_eligible",
            "documentation_not_verified": "not_eligible",
            "provenance_not_verified": "not_eligible",
            "technical_validation_not_passed": "not_eligible",
            "automated_checks_not_passed": "not_eligible",
            "leg1_reward_pending": "pending",
            "leg1_reward_active": "active",
            "leg1_reward_stopped": "stopped",
        }
        if self.reward_status != expected_reward_state[self.decision_reason_code]:
            raise ValueError("SOURCE_ADD public reward status is inconsistent")
        reward_schedule = (
            self.alpha_percent,
            self.reward_epochs,
            self.start_epoch,
            self.end_epoch,
        )
        schedule_present = tuple(item is not None for item in reward_schedule)
        if any(schedule_present) and not all(schedule_present):
            raise ValueError("SOURCE_ADD public reward schedule is incomplete")
        if self.decision_status != "approved" and any(schedule_present):
            raise ValueError("SOURCE_ADD non-approved decision has a reward schedule")
        if self.reward_status in {"active", "stopped"} and not all(schedule_present):
            raise ValueError("SOURCE_ADD active reward schedule is missing")
        if all(schedule_present) and self.end_epoch != self.start_epoch + self.reward_epochs - 1:
            raise ValueError("SOURCE_ADD public reward epoch range is inconsistent")
        return self


class ResearchLabSourceAddStatusResponse(BaseModel):
    schema_version: Literal["leadpoet.source_add_miner_status.v1"]
    submissions: list[ResearchLabSourceAddStatusItem] = Field(max_length=50)
    next_cursor: Optional[str] = Field(
        default=None,
        pattern=r"^source_add_submission:[0-9a-f]{16}$",
    )


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
    provider_alias: Optional[str] = Field(default=None, min_length=1, max_length=80)
    provision_status: str = Field(default="provisioned_autoresearch_eligible", max_length=80)
    base_url: Optional[str] = Field(default=None, max_length=500)
    auth_kind: str = Field(default="none", max_length=20)
    auth_name: Optional[str] = Field(default=None, max_length=120)
    credential_env_refs: list[str] = Field(default_factory=list, max_length=8)
    api_credential: Optional[SecretStr] = Field(default=None, min_length=1, max_length=65536)
    api_credential_v2: Optional[AttestedCredentialCiphertextV2] = None
    cost_model: dict[str, Any] = Field(default_factory=dict)
    routing_contract: dict[str, Any] = Field(default_factory=dict)
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
        if value != normalized or not re.fullmatch(
            r"[a-z][a-z0-9_-]{1,79}", normalized
        ):
            raise ValueError(
                "registry_provider_id must be a canonical lowercase slug"
            )
        return normalized

    @field_validator("operator_notes")
    @classmethod
    def notes_have_no_secret_material(cls, value: Optional[str]) -> Optional[str]:
        if value:
            reject_secret_material(value)
        return value

    @field_validator("provider_alias")
    @classmethod
    def normalize_provider_alias(cls, value: Optional[str]) -> Optional[str]:
        normalized = " ".join(str(value or "").split())
        if normalized:
            reject_secret_material(normalized)
        return normalized or None

    @field_validator(
        "cost_model",
        "routing_contract",
        "probe_endpoints",
        "request_headers",
    )
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

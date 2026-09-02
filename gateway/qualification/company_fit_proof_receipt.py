"""Strict consumer for the model-owned company-fit proof receipt.

This is a temporary, lossless consumer shim for the model-owned
``company-fit-proof-receipt:v1`` contract. The receipt is an audit artifact
and a prerequisite for the exact Research Lab v2 scorer; its citations and
free text never enter the independent scorer prompt or decide that scorer's
verdict. Remove the shim when Leadpoet consumes the model-owned parser directly
from the shared, immutable sourcing-model release.

Tracking reference: ``leadpoet/Sourcing_model`` PR #332 candidate
``3595a4e3b23b943cabc962d68b909d23664f9acc`` (consumer contract SHA-256
``9ff0567bd417c20ddf617c6599c6b954ba06e886b4ee8aa888d21788f15dfaaf``;
consumer parity SHA-256
``72d025e78959176896ca3f143baf63fde58cf87ba2f774df59cd8366dc0055b8``).
This source reference does not substitute for the signed immutable model
manifest required at activation. The explicit removal condition is a shared
champion release that exports the same validator for direct consumer import.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Final, Literal, Mapping, Optional
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, Field, StrictBool, StrictStr, model_validator


COMPANY_FIT_PROOF_RECEIPT_SCHEMA_VERSION: Final = (
    "company-fit-proof-receipt:v1"
)
COMPANY_FIT_PROOF_RECEIPT_OUTCOME_BINDING: Final = (
    "qualification_outcome.route_completion_receipt.extensions."
    "leadpoet.sourcing-model.companies_sha256"
)
COMPANY_FIT_PROOF_RECEIPT_EMPLOYEE_SOURCES: Final = (
    "scrapingdog_linkedin_company_profile",
    "cached_structured_acquisition_exact_count",
    "deepline_domain_firmographics",
    "exact_public_company_qualification_evidence",
)


def company_fit_proof_receipt_contract_identity() -> dict[str, Any]:
    """Return the exact model-owned v1 contract identity."""

    return {
        "contract_id": COMPANY_FIT_PROOF_RECEIPT_SCHEMA_VERSION,
        "receipt_schema_version": COMPANY_FIT_PROOF_RECEIPT_SCHEMA_VERSION,
        "outcome_binding": COMPANY_FIT_PROOF_RECEIPT_OUTCOME_BINDING,
        "hash_algorithm": "sha256",
        "canonical_json": "utf8-json-sort-keys-compact-no-nan",
        "closed_fields": {
            "receipt": [
                "schema_version",
                "contract_sha256",
                "outcome_binding",
                "decision",
                "company_binding",
                "icp_binding",
                "dimensions",
                "employee_size_proof",
                "stage_proof",
                "receipt_sha256",
            ],
            "company_binding": [
                "company_name",
                "company_website",
                "company_linkedin",
            ],
            "icp_binding": [
                "employee_count",
                "employee_count_required",
                "company_stage",
                "stage_required",
            ],
            "dimensions": [
                "identity",
                "employee_size",
                "industry",
                "geography",
                "stage",
            ],
            "employee_size_proof": [
                "decision",
                "observed_employee_count",
                "evidence_source",
                "evidence_url",
            ],
            "stage_proof": [
                "decision",
                "observed_company_stage",
                "evidence_url",
                "evidence_quote",
            ],
        },
        "required_decision": "match",
        "required_dimension_decision": "match",
        "not_required_decision": "not_required",
        "string_normalization": {
            "default": "strip_only",
            "company_binding.company_name": "strip_only",
            "company_binding.company_website": (
                "strip_then_lowercase_scheme_and_netloc_preserving_"
                "path_and_params"
            ),
            "company_binding.company_linkedin": "strip_only",
        },
        "binding_rules": {
            "company_binding": (
                "exact_emitted_company_fields_after_declared_normalization"
            ),
            "company_linkedin": (
                "exact_validated_url_or_empty_never_synthesized"
            ),
            "icp_binding": "effective_per_company_qualification_icp_fields",
            "icp_binding_authority": "audit_and_citation_context_only",
            "parent_scorer_authority": "independent_authoritative_invocation",
            "outer_invocation_replay_binding": (
                "qualification_outcome.route_completion_receipt."
                "invocation_sha256"
            ),
            "outer_company_replay_binding": (
                COMPANY_FIT_PROOF_RECEIPT_OUTCOME_BINDING
            ),
            "stage_required_observed_company_stage": (
                "token_equal_to_icp_binding_and_emitted_company_stage"
            ),
        },
        "url_policy": {
            "schemes": ["http", "https"],
            "absolute": True,
            "hostname_required": True,
            "userinfo_allowed": False,
            "ascii_control_or_whitespace_allowed": False,
            "query_allowed": False,
            "fragment_allowed": False,
            "port_minimum": 1,
            "port_maximum": 65535,
        },
        "employee_size": {
            "proof_sources": list(
                COMPANY_FIT_PROOF_RECEIPT_EMPLOYEE_SOURCES
            ),
            "required_fields_when_constrained": [
                "observed_employee_count",
                "evidence_source",
                "evidence_url",
            ],
            "evidence_quote_required": False,
            "observed_value_format": "model_clean_band",
        },
        "stage": {
            "required_fields_when_constrained": [
                "observed_company_stage",
                "evidence_url",
                "evidence_quote",
            ],
            "citation_fields": ["evidence_url", "evidence_quote"],
        },
        "not_required_proof_fields": "empty_strings",
        "max_lengths": {
            "company_name": 200,
            "url": 2048,
            "employee_count": 128,
            "company_stage": 200,
            "evidence_quote": 600,
        },
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


COMPANY_FIT_PROOF_RECEIPT_CONTRACT_SHA256: Final = _sha256_json(
    company_fit_proof_receipt_contract_identity()
)


def _credential_free_http_url(
    value: str,
    *,
    normalize_website_binding: bool = False,
) -> str:
    raw = value.strip()
    if (
        raw != value
        or any(ord(character) <= 0x20 or ord(character) == 0x7F for character in raw)
    ):
        return ""
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except (TypeError, ValueError):
        return ""
    if (
        parsed.scheme.casefold() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.query)
        or bool(parsed.fragment)
        or (port is not None and not 1 <= port <= 65535)
    ):
        return ""
    if normalize_website_binding:
        return urlunsplit(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return raw


def _canonical_text(value: str, *, required: bool = False) -> str:
    normalized = value.strip()
    if value != normalized or (required and not normalized):
        return ""
    return normalized


class CompanyFitProofCompanyBinding(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    company_name: StrictStr = Field(..., min_length=1, max_length=200)
    company_website: StrictStr = Field(..., min_length=1, max_length=2048)
    company_linkedin: StrictStr = Field(..., max_length=2048)

    @model_validator(mode="after")
    def _validate_canonical_binding(
        self,
    ) -> "CompanyFitProofCompanyBinding":
        if _canonical_text(self.company_name, required=True) != self.company_name:
            raise ValueError("company-fit company name is not canonical")
        if (
            _credential_free_http_url(
                self.company_website,
                normalize_website_binding=True,
            )
            != self.company_website
            or (
                self.company_linkedin
                and _credential_free_http_url(self.company_linkedin)
                != self.company_linkedin
            )
        ):
            raise ValueError("company-fit company URL binding is invalid")
        return self


class CompanyFitProofIcpBinding(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    employee_count: StrictStr = Field(..., max_length=128)
    employee_count_required: StrictBool
    company_stage: StrictStr = Field(..., max_length=200)
    stage_required: StrictBool

    @model_validator(mode="after")
    def _validate_canonical_binding(self) -> "CompanyFitProofIcpBinding":
        if any(
            (
                self.employee_count != self.employee_count.strip(),
                self.employee_count_required and not self.employee_count,
                self.company_stage != self.company_stage.strip(),
                self.stage_required and not self.company_stage,
            )
        ):
            raise ValueError("company-fit ICP binding is not canonical")
        return self


class CompanyFitProofDimensions(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    identity: Literal["match"]
    employee_size: Literal["match"]
    industry: Literal["match"]
    geography: Literal["match"]
    stage: Literal["match"]


class CompanyFitEmployeeSizeProof(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    decision: Literal["match", "not_required"]
    observed_employee_count: StrictStr = Field(..., max_length=128)
    evidence_source: StrictStr = Field(..., max_length=128)
    evidence_url: StrictStr = Field(..., max_length=2048)

    @model_validator(mode="after")
    def _validate_canonical_proof(self) -> "CompanyFitEmployeeSizeProof":
        if any(
            _canonical_text(value) != value
            for value in (
                self.observed_employee_count,
                self.evidence_source,
                self.evidence_url,
            )
        ):
            raise ValueError("company-fit employee proof is not canonical")
        return self


class CompanyFitStageProof(BaseModel):
    model_config = {"extra": "forbid", "frozen": True}

    decision: Literal["match", "not_required"]
    observed_company_stage: StrictStr = Field(..., max_length=200)
    evidence_url: StrictStr = Field(..., max_length=2048)
    evidence_quote: StrictStr = Field(..., max_length=600)

    @model_validator(mode="after")
    def _validate_canonical_proof(self) -> "CompanyFitStageProof":
        if any(
            _canonical_text(value) != value
            for value in (
                self.observed_company_stage,
                self.evidence_url,
                self.evidence_quote,
            )
        ):
            raise ValueError("company-fit stage proof is not canonical")
        return self


class CompanyFitProofReceipt(BaseModel):
    """Closed, hash-bound model receipt whose citations remain untrusted."""

    model_config = {"extra": "forbid", "frozen": True}

    schema_version: Literal["company-fit-proof-receipt:v1"]
    contract_sha256: StrictStr = Field(
        ..., pattern=r"^[0-9a-f]{64}$"
    )
    outcome_binding: Literal[
        "qualification_outcome.route_completion_receipt.extensions."
        "leadpoet.sourcing-model.companies_sha256"
    ]
    decision: Literal["match"]
    company_binding: CompanyFitProofCompanyBinding
    icp_binding: CompanyFitProofIcpBinding
    dimensions: CompanyFitProofDimensions
    employee_size_proof: CompanyFitEmployeeSizeProof
    stage_proof: CompanyFitStageProof
    receipt_sha256: StrictStr = Field(..., pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_contract_and_hash(self) -> "CompanyFitProofReceipt":
        if self.contract_sha256 != COMPANY_FIT_PROOF_RECEIPT_CONTRACT_SHA256:
            raise ValueError("company-fit proof contract hash does not match")

        employee = self.employee_size_proof
        if self.icp_binding.employee_count_required:
            if (
                employee.decision != "match"
                or not employee.observed_employee_count
                or employee.evidence_source
                not in COMPANY_FIT_PROOF_RECEIPT_EMPLOYEE_SOURCES
                or _credential_free_http_url(employee.evidence_url)
                != employee.evidence_url
            ):
                raise ValueError("company-fit employee proof is incomplete")
        elif any(
            (
                employee.decision != "not_required",
                bool(employee.observed_employee_count),
                bool(employee.evidence_source),
                bool(employee.evidence_url),
            )
        ):
            raise ValueError("unconstrained employee proof must be empty")

        stage = self.stage_proof
        if self.icp_binding.stage_required:
            observed_stage = re.sub(
                r"[^a-z0-9]+",
                " ",
                stage.observed_company_stage.casefold(),
            ).strip()
            required_stage = re.sub(
                r"[^a-z0-9]+",
                " ",
                self.icp_binding.company_stage.casefold(),
            ).strip()
            if (
                stage.decision != "match"
                or not stage.observed_company_stage
                or observed_stage != required_stage
                or _credential_free_http_url(stage.evidence_url)
                != stage.evidence_url
                or not stage.evidence_quote.strip()
                or stage.evidence_quote != stage.evidence_quote.strip()
            ):
                raise ValueError("company-fit stage proof is incomplete")
        elif any(
            (
                stage.decision != "not_required",
                bool(stage.observed_company_stage),
                bool(stage.evidence_url),
                bool(stage.evidence_quote),
            )
        ):
            raise ValueError("unconstrained stage proof must be empty")

        payload = self.model_dump(mode="json", exclude={"receipt_sha256"})
        if self.receipt_sha256 != _sha256_json(payload):
            raise ValueError("company-fit proof receipt hash does not match")
        return self


def _mapping_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def validate_company_fit_proof_receipt_binding(
    receipt: Optional[CompanyFitProofReceipt],
    *,
    company: Any,
) -> tuple[bool, str]:
    """Validate emitted-company bindings without trusting model ICP context.

    ``receipt.icp_binding`` is the model's effective per-company qualification
    context and remains audit-only. The protocol-v2 invocation and raw company
    hashes prevent cross-invocation replay; the independent scorer continues
    to use its authoritative Lab ICP.
    """

    if receipt is None:
        return False, "company_fit_proof_receipt_missing_or_invalid"
    expected_website = str(
        _mapping_value(company, "company_website", "") or ""
    ).strip()
    expected_company = {
        "company_name": str(
            _mapping_value(company, "company_name", "") or ""
        ).strip(),
        "company_website": _credential_free_http_url(
            expected_website,
            normalize_website_binding=True,
        ),
        "company_linkedin": str(
            _mapping_value(company, "company_linkedin", "") or ""
        ).strip(),
    }
    if receipt.company_binding.model_dump(mode="json") != expected_company:
        return False, "company_fit_proof_company_binding_mismatch"
    if (
        receipt.icp_binding.employee_count_required
        and receipt.employee_size_proof.observed_employee_count
        != str(_mapping_value(company, "employee_count", "") or "").strip()
    ):
        return False, "company_fit_proof_employee_binding_mismatch"
    if receipt.icp_binding.stage_required:
        observed_stage = re.sub(
            r"[^a-z0-9]+",
            " ",
            receipt.stage_proof.observed_company_stage.casefold(),
        ).strip()
        required_stage = re.sub(
            r"[^a-z0-9]+",
            " ",
            receipt.icp_binding.company_stage.casefold(),
        ).strip()
        emitted_stage = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(_mapping_value(company, "company_stage", "") or "")
            .strip()
            .casefold(),
        ).strip()
        if (
            not observed_stage
            or observed_stage != required_stage
            or emitted_stage != observed_stage
        ):
            return False, "company_fit_proof_stage_binding_mismatch"
    return True, ""


__all__ = [
    "COMPANY_FIT_PROOF_RECEIPT_CONTRACT_SHA256",
    "COMPANY_FIT_PROOF_RECEIPT_EMPLOYEE_SOURCES",
    "COMPANY_FIT_PROOF_RECEIPT_OUTCOME_BINDING",
    "COMPANY_FIT_PROOF_RECEIPT_SCHEMA_VERSION",
    "CompanyFitEmployeeSizeProof",
    "CompanyFitProofCompanyBinding",
    "CompanyFitProofDimensions",
    "CompanyFitProofIcpBinding",
    "CompanyFitProofReceipt",
    "CompanyFitStageProof",
    "company_fit_proof_receipt_contract_identity",
    "validate_company_fit_proof_receipt_binding",
]

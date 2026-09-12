"""Shared company scorer for the public baseline and miner bundles."""

from __future__ import annotations

from importlib import import_module
import hashlib
import json
import logging
import math
import os
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from leadpoet_verifier.aggregation import per_icp_normalized_score
from pydantic import ValidationError
from qualification.competition_models import CompetitionCompany
from qualification.employee_buckets import (
    normalize_employee_count_bucket,
    normalize_observed_employee_count_bucket,
)
from qualification.scoring.arena_integrity import (
    bounded_criterion_evidence,
    canonical_company_identity,
    company_fit_verified,
    company_identity_alias_keys,
    fit_evidence_url_hints,
    verified_identity_receipt,
)


SCORING_ADAPTER_VERSION = "qualification-company-scorer:v1"
DEFAULT_COMPANY_GOAL = 5
MAX_COMPANY_GOAL = 5
FP_PENALTY_POINTS = 10.0
logger = logging.getLogger(__name__)

_PENALIZABLE_FAILURE_MARKERS = (
    "exclusion list",
    "required_attribute",
    "missing employee_count",
    "missing company_stage",
    "country mismatch",
    "missing country",
    "duplicate company",
    "data quality issue",
    "missing industry",
    "company verification failed",
)
_NEVER_PENALIZE_MARKERS = ("error", "timeout", "provider", "429")
_MODEL_CONTRACT_INCOMPATIBLE_FAILURE_CLASS = "model_contract_incompatible"
_NON_RETRYABLE_UNAVAILABLE_FAILURE_CLASSES = frozenset({
    "insufficient_fit_evidence",
    _MODEL_CONTRACT_INCOMPATIBLE_FAILURE_CLASS,
})


class CompetitionScorerInputError(ValueError):
    """A company or ICP does not satisfy the public competition boundary."""


def _model_contract_incompatible_breakdown() -> dict[str, Any]:
    """Return a safe zero when one public company cannot enter the judge model."""

    return {
        "icp_fit": 0.0,
        "decision_maker": 0.0,
        "intent_signal_raw": 0.0,
        "time_decay_multiplier": 1.0,
        "intent_signal_final": 0.0,
        "cost_penalty": 0.0,
        "time_penalty": 0.0,
        "final_score": 0.0,
        "failure_reason": "company model contract incompatible",
        "intent_signals_detail": None,
        "verifier_gate_receipts": [
            {
                "gate": "company_fit",
                "decision": "unavailable",
                "reason": "company_model_contract_incompatible",
                "failure_class": _MODEL_CONTRACT_INCOMPATIBLE_FAILURE_CLASS,
            }
        ],
    }


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(
            value.get("intent_signal")
            or value.get("signal")
            or value.get("text")
            or ""
        ).strip()
    return str(value or "").strip()


def _category(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    text = str(
        value.get("intent_category")
        or value.get("category")
        or value.get("evidence_type")
        or ""
    ).strip().upper()
    return text or None


def _max_age_days(value: Any) -> int | None:
    if not isinstance(value, Mapping):
        return None
    raw = value.get("max_age_days")
    if raw is None:
        raw = value.get("intent_max_age_days")
    try:
        days = int(raw)
    except (TypeError, ValueError):
        return None
    return days if days > 0 else None


def employee_count_buckets_for_icp(icp: Mapping[str, Any]) -> list[str]:
    """Return the exact employee buckets declared by one ICP."""

    raw = icp.get("employee_count")
    values = (
        list(raw)
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray))
        else str(raw or "").replace(";", "|").split("|")
    )
    buckets: list[str] = []
    for value in values:
        bucket = normalize_employee_count_bucket(value, default=None)
        if bucket and bucket not in buckets:
            buckets.append(bucket)
    if not buckets:
        raise CompetitionScorerInputError("ICP employee_count has no valid bucket")
    return buckets


def _company_goal(icp: Mapping[str, Any]) -> int:
    try:
        value = int(icp.get("max_companies", DEFAULT_COMPANY_GOAL))
    except (TypeError, ValueError):
        value = DEFAULT_COMPANY_GOAL
    return max(1, min(MAX_COMPANY_GOAL, value))


def _normalized_icp(icp: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(icp, Mapping):
        raise CompetitionScorerInputError("ICP must be an object")
    industry = str(icp.get("industry") or "").strip()
    icp_id = str(icp.get("icp_id") or "").strip()
    if not industry or not icp_id:
        raise CompetitionScorerInputError("ICP is missing icp_id or industry")

    signals: list[str] = []
    evidence_types: list[str | None] = []
    signal_max_age_days: list[int] = []
    primary_category = str(icp.get("intent_category") or "").strip().upper() or None
    primary_max_age_days = max(1, int(icp.get("intent_max_age_days") or 365))
    bonus_metadata = {
        _text(item): (_category(item), _max_age_days(item))
        for item in (icp.get("bonus_intents") or [])
        if isinstance(item, Mapping) and _text(item)
    }
    raw_signals = icp.get("intent_signals") or [icp.get("intent_signal")]
    if isinstance(raw_signals, (str, Mapping)):
        raw_signals = [raw_signals]
    for index, item in enumerate(raw_signals or []):
        signal = _text(item)
        if not signal or signal in signals:
            continue
        signals.append(signal)
        bonus_category, bonus_max_age_days = bonus_metadata.get(
            signal, (None, None)
        )
        evidence_types.append(
            _category(item)
            or bonus_category
            or (primary_category if index == 0 else None)
        )
        signal_max_age_days.append(
            _max_age_days(item)
            or bonus_max_age_days
            or primary_max_age_days
        )
    for item in icp.get("bonus_intents") or []:
        signal = _text(item)
        if signal and signal not in signals:
            signals.append(signal)
            evidence_types.append(_category(item))
            signal_max_age_days.append(
                _max_age_days(item) or primary_max_age_days
            )
    if not signals:
        raise CompetitionScorerInputError("ICP has no intent signal")

    buckets = employee_count_buckets_for_icp(icp)
    stage = icp.get("company_stage") or "Any"
    if isinstance(stage, Sequence) and not isinstance(stage, (str, bytes, bytearray)):
        stage = next((str(value).strip() for value in stage if str(value).strip()), "Any")
    country = str(icp.get("country") or "").strip()
    geography = str(icp.get("geography") or country or "United States").strip()
    required_attribute = str(icp.get("required_attribute") or "").strip()
    product_service = str(
        icp.get("product_service") or required_attribute or industry
    ).strip()
    prompt = str(icp.get("prompt") or "").strip()
    if not prompt:
        prompt = f"Find {industry} companies in {geography} with {signals[0]}"
    excluded = icp.get("excluded_companies") or []
    if not isinstance(excluded, list):
        excluded = []
    return {
        "icp_id": icp_id,
        "prompt": prompt,
        "industry": industry,
        "sub_industry": str(icp.get("sub_industry") or industry).strip(),
        "target_roles": [],
        "target_seniority": "",
        "employee_count": "|".join(buckets),
        "company_stage": str(stage).strip() or "Any",
        "geography": geography,
        "country": country or geography,
        "product_service": product_service,
        "required_attribute": required_attribute,
        "excluded_companies": [str(value) for value in excluded],
        "intent_signals": signals,
        "intent_signal_evidence_types": evidence_types,
        "intent_signal_max_age_days": signal_max_age_days,
        "intent_max_age_days": primary_max_age_days,
    }


def _normalized_company(
    company: Mapping[str, Any], *, integrity_policy: bool = False,
    contacts_required: bool = False,
    company_quality: bool = False,
) -> dict[str, Any]:
    try:
        # V2 carries a contact claim which the independent contact gate owns.
        # The legacy company judge remains contact-blind and extra-forbidding.
        company_input = dict(company)
        if contacts_required:
            company_input.pop("contact", None)
        if company_quality:
            from qualification.competition_models import CompetitionCompanyV3
            from qualification.company_quality import normalize_company_claim
            row, _errors = normalize_company_claim(
                CompetitionCompanyV3.model_validate(company_input).model_dump(mode="json")
            )
        else:
            row = CompetitionCompany.model_validate(company_input).model_dump(mode="json")
    except Exception as exc:
        raise CompetitionScorerInputError(
            "company does not satisfy the competition output schema"
        ) from exc
    signals = [
        {
            "source": _evidence_source(
                signal["url"], company_website=row["company_website"]
            ),
            "description": signal["description"],
            "url": signal["url"],
            "date": signal["date"],
            "snippet": signal["snippet"],
            "matched_icp_signal": signal["matched_icp_signal"],
        }
        for signal in row["intent_signals"]
    ]
    if integrity_policy:
        signals = [signal for group in bounded_criterion_evidence(signals) for signal in group]
    return {
        "company_name": row["company_name"],
        "company_website": row["company_website"],
        "company_linkedin": row["company_linkedin"],
        "industry": row["industry"],
        "sub_industry": "",
        "employee_count": row["employee_count"],
        "company_stage": row["company_stage"],
        "country": row["country"],
        "state": row["state"],
        "description": row["fit_summary"][:500],
        "fit_evidence_urls": (fit_evidence_url_hints(row["fit_evidence_urls"])
                              if integrity_policy else row["fit_evidence_urls"]),
        "intent_signals": signals,
        "required_attribute": row.get("required_attribute"),
    }


def effective_competition_input(
    companies: Sequence[Mapping[str, Any]],
    icp: Mapping[str, Any],
    *,
    contacts_required: bool = False,
    contact_source_evidence: Mapping[str, Any] | None = None,
    company_quality: bool = False,
) -> dict[str, Any]:
    """Project the same normalized first-N inputs consumed by the adapter.

    Company positions and order remain significant. Bucket-skipped companies
    have no judge input; output padding, unused prose and repeated/capped
    evidence cannot buy a fresh judgment.
    """
    buckets = employee_count_buckets_for_icp(icp)
    from gateway.qualification.models import CompanyOutput

    rows = []
    for company in list(companies)[:_company_goal(icp)]:
        observed = company.get("employee_count")
        bucket = normalize_employee_count_bucket(observed, default=None) or normalize_observed_employee_count_bucket(observed, default=None)
        if bucket not in buckets and not company_quality:
            rows.append({"bucket_skipped": True})
            continue
        normalized = _normalized_company(
            company,
            integrity_policy=True,
            contacts_required=contacts_required,
            company_quality=company_quality,
        )
        if company_quality:
            from qualification.company_quality import normalize_company_claim
            _claim, errors = normalize_company_claim(company)
            if errors or bucket not in buckets:
                # No networked judgment, but retain every identity-affecting
                # input so a deterministic zero cannot bind to another firm.
                effective = dict(normalized)
                effective["company_quality_errors"] = list(errors)
                if bucket not in buckets:
                    effective["bucket_skipped"] = True
                if contacts_required:
                    effective.update(_effective_contact_input(company, contact_source_evidence))
                rows.append(effective)
                continue
        try:
            effective = CompanyOutput(**normalized).model_dump(mode="json")
        except ValidationError:
            # Preserve validation inputs for incompatible rows; none reach
            # the networked judge, and identity still affects their receipts.
            rows.append(normalized)
            continue
        # The binary Arena fit verifier independently resolves these facts.
        # These fields are validated above but never read during its judging.
        for ignored in (("description", "required_attribute") if company_quality else ("state", "description", "required_attribute")):
            effective.pop(ignored, None)
        if company_quality:
            from qualification.company_quality import is_united_states
            if not is_united_states(effective.get("country")):
                effective.pop("state", None)
        if contacts_required:
            effective.update(
                _effective_contact_input(company, contact_source_evidence)
            )
        rows.append(effective)
    effective_icp = _normalized_icp(icp)
    if contacts_required:
        geography = icp.get("contact_geography")
        effective_icp.update({
            "target_roles": _normalized_string_list(icp.get("target_roles")),
            "target_seniority": str(icp.get("target_seniority") or "").strip(),
            "contact_geography": _normalized_contact_geography(geography),
        })
    return {"icp": effective_icp, "companies": rows}


def _normalized_string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _normalized_contact_geography(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, Mapping):
        return {"countries": [], "regions": [], "cities": []}
    from qualification.contact_models import normalize_country_code

    countries: list[str] = []
    for item in _normalized_string_list(value.get("countries")):
        try:
            normalized = normalize_country_code(item)
        except ValueError:
            normalized = item.casefold()
        if normalized not in countries:
            countries.append(normalized)
    return {
        "countries": countries,
        "regions": [item.casefold() for item in _normalized_string_list(value.get("regions"))],
        "cities": [item.casefold() for item in _normalized_string_list(value.get("cities"))],
    }


def _semantic_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _without_cache_metadata(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _without_cache_metadata(item, depth + 1)
            for key, item in value.items()
            if str(key) not in {"broker_call_id", "call_identity", "observed_at"}
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_without_cache_metadata(item, depth + 1) for item in value]
    return value


def _effective_contact_input(
    company: Mapping[str, Any],
    source_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    from qualification.contact_models import validate_contact_claim

    raw_contact = company.get("contact")
    try:
        contact = validate_contact_claim(raw_contact)
    except (TypeError, ValueError):
        return {
            "contact_invalid_hash": _semantic_hash(
                _without_cache_metadata(raw_contact)
            )
        }
    source = (
        source_evidence.get(_contact_source_key(company))
        if isinstance(source_evidence, Mapping)
        else None
    )
    from qualification.scoring.contact_verification import (
        contact_source_semantics,
    )

    attribution = contact.get("email_source") or {}
    source_semantics = contact_source_semantics(
        source,
        broker_call_id=attribution.get("broker_call_id"),
        record_id=attribution.get("record_id"),
    )
    effective_source = {
        "provider": attribution.get("provider"),
        "tool": attribution.get("tool"),
    }
    if attribution.get("broker_call_id"):
        effective_source["broker_call_id"] = attribution["broker_call_id"]
    if attribution.get("record_id"):
        effective_source["record_id"] = attribution["record_id"]
    return {
        "contact": {
            "full_name": contact["full_name"],
            "role": contact["role"],
            "linkedin_url": contact["linkedin_url"],
            "location": contact["location"],
            "email": contact["email"],
            "email_source": effective_source,
        },
        "contact_source_evidence_hash": _semantic_hash(source_semantics),
    }


def _contact_source_key(company: Mapping[str, Any]) -> str:
    contact = company.get("contact")
    source = contact.get("email_source") if isinstance(contact, Mapping) else None
    if not isinstance(source, Mapping):
        return ""
    return str(source.get("broker_call_id") or source.get("record_id") or "")


def _evidence_source(url: str, *, company_website: str) -> str:
    """Infer the scorer's source class from the submitted public URL."""

    hostname = (urlsplit(str(url)).hostname or "").lower().removeprefix("www.")
    company_hostname = (
        (urlsplit(str(company_website)).hostname or "").lower().removeprefix("www.")
    )
    path = (urlsplit(str(url)).path or "").lower()
    if hostname == "linkedin.com" or hostname.endswith(".linkedin.com"):
        return "linkedin"
    if hostname == "github.com" or hostname.endswith(".github.com"):
        return "github"
    if any(marker in path for marker in ("/jobs", "/job/", "/careers")):
        return "job_board"
    if company_hostname and (
        hostname == company_hostname or hostname.endswith("." + company_hostname)
    ):
        return "company_website"
    return "news"


def _ensure_provider_environment() -> None:
    key = os.getenv("QUALIFICATION_OPENROUTER_API_KEY") or os.getenv(
        "OPENROUTER_API_KEY"
    )
    if not key:
        return
    os.environ.setdefault("QUALIFICATION_OPENROUTER_API_KEY", key)
    for module_name in ("qualification.scoring.verification_helpers",):
        module = import_module(module_name)
        if not getattr(module, "OPENROUTER_API_KEY", ""):
            setattr(module, "OPENROUTER_API_KEY", key)


def _not_evaluated_contact(company: Mapping[str, Any]) -> dict[str, Any]:
    from qualification.scoring.contact_verification import contact_identity_key

    reason = "company_not_qualified"
    return {
        "contact_qualified": False,
        "contact_identity_key": contact_identity_key(company.get("contact")),
        "email_status": "unknown",
        "contact_verification": {
            "decision": "not_evaluated",
            "reason": reason,
            "subchecks": {},
            "evidence_hashes": {},
            "evidence_timestamps": {},
        },
        "verifier_gate_receipts": [
            {"gate": "contact", "decision": "not_evaluated", "reason": reason}
        ],
    }


def _merge_contact_breakdown(
    breakdown: dict[str, Any], contact_result: Mapping[str, Any]
) -> None:
    existing_receipts = breakdown.get("verifier_gate_receipts")
    receipts = (
        list(existing_receipts)
        if isinstance(existing_receipts, Sequence)
        and not isinstance(existing_receipts, (str, bytes, bytearray))
        else []
    )
    contact_receipts = contact_result.get("verifier_gate_receipts")
    if isinstance(contact_receipts, Sequence) and not isinstance(
        contact_receipts, (str, bytes, bytearray)
    ):
        receipts.extend(contact_receipts)
    for key in (
        "contact_qualified",
        "contact_identity_key",
        "email_status",
        "contact_verification",
    ):
        breakdown[key] = contact_result[key]
    breakdown["verifier_gate_receipts"] = receipts


def _verified_company_for_contact(
    company: Mapping[str, Any], verified_receipt: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Bind contact employment to the identity proven by the company gate."""
    if not isinstance(verified_receipt, Mapping):
        return dict(company)
    observed_domain = str(verified_receipt.get("observed_domain") or "").strip()
    observed_slug = str(
        verified_receipt.get("observed_linkedin_slug") or ""
    ).strip()
    return {
        "company_name": str(
            company.get("company_name") or ""
        ),
        "company_website": (
            f"https://{observed_domain}" if observed_domain else ""
        ),
        "company_linkedin": (
            f"https://www.linkedin.com/company/{observed_slug}/"
            if observed_slug
            else ""
        ),
        "contact": company.get("contact"),
    }


async def _classify_contact_role(
    actual_role: str, target_roles: list[str], _target_seniority: str
) -> bool:
    """Use the existing pinned role judge only for deterministic gray zones."""
    role_module = import_module("qualification.scoring.role_batch_check")
    key = (
        os.environ.get("OPENROUTER_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("QUALIFICATION_OPENROUTER_API_KEY")
    )
    if not key:
        raise RuntimeError("contact role judge key unavailable")
    async with role_module.httpx.AsyncClient() as client:
        parsed = await role_module._judge_chunk(
            client,
            key,
            target_roles,
            [{"id": "contact", "role": actual_role}],
        )
    if (
        not isinstance(parsed, list)
        or len(parsed) != 1
        or not isinstance(parsed[0], Mapping)
        or parsed[0].get("id") != "contact"
        or type(parsed[0].get("match")) is not bool
    ):
        raise RuntimeError("contact role judge response unavailable")
    return parsed[0]["match"]


def raw_company_judgment(breakdown: Mapping[str, Any]) -> dict[str, Any]:
    """Remove list-dependent decisions before storing an intrinsic judgment."""
    return {
        key: value for key, value in breakdown.items()
        if key not in {"company_index", "company_qualified", "duplicate_company", "duplicate_of_index"}
    }


def bucket_skipped_judgment() -> dict[str, Any]:
    result = _model_contract_incompatible_breakdown()
    result.update(bucket_skipped=True, failure_reason="employee_bucket_skipped")
    return result


def company_quality_identity_verified(receipt: Any) -> bool:
    """A quality identity must contain independently observed usable handles."""
    from qualification.company_quality import canonical_company_linkedin
    from qualification.competition_models import public_http_url

    if not isinstance(receipt, Mapping):
        return False
    name = receipt.get("observed_name")
    domain = receipt.get("observed_domain")
    slug = receipt.get("observed_linkedin_slug")
    if any(not isinstance(value, str) or not value.strip() for value in (name, domain, slug)):
        return False
    try:
        observed = urlsplit("https://" + domain)
        if (observed.hostname != domain or observed.port is not None
            or observed.path or observed.query or observed.fragment
            or observed.username or observed.password):
            return False
        public_http_url("https://" + domain)
        linkedin = canonical_company_linkedin("https://linkedin.com/company/" + slug)
        if linkedin.rsplit("/", 1)[-1] != slug.casefold():
            return False
        identity = canonical_company_identity({}, verified_identity_receipt=receipt)
        return bool(identity.verified and identity.registrable_domain and identity.normalized_name and identity.verified_linkedin_slug)
    except (TypeError, ValueError):
        return False


def apply_company_judgment_context(
    companies: Sequence[Mapping[str, Any]],
    raw_breakdowns: Sequence[Mapping[str, Any]],
    *, contacts_required: bool = False,
) -> list[dict[str, Any]]:
    """Bind immutable single-company decisions to this list's verified identities.

    Inputs cover every first-N slot, including explicit bucket-skipped zeros.
    Only independently verified aliases can reserve another company's identity.
    """
    from qualification.company_quality import normalize_company_claim
    from qualification.scoring.company_fit_decision import company_quality_receipt_matches_claim

    if len(companies) != len(raw_breakdowns):
        raise CompetitionScorerInputError("company judgments must cover every company")
    result = []
    seen_linkedin: dict[str, tuple[str, int]] = {}
    for index, (company, raw) in enumerate(zip(companies, raw_breakdowns)):
        if raw.get("bucket_skipped") is True:
            continue
        row = json.loads(json.dumps(raw, allow_nan=False))
        claim, errors = normalize_company_claim(company)
        receipt = verified_identity_receipt(row.get("verifier_gate_receipts"))
        identity = canonical_company_identity(claim, verified_identity_receipt=receipt)
        aliases = list(company_identity_alias_keys(identity)) or [identity.key]
        fit = (company_fit_verified(row) and company_quality_identity_verified(receipt)
               and company_quality_receipt_matches_claim(receipt, claim) and not errors)
        company_ready = bool(
            fit
            and has_verified_primary_intent(row.get("intent_signals_detail") or [])
        )
        qualified = bool(company_ready)
        if contacts_required:
            qualified = qualified and row.get("contact_qualified") is True
        linkedin_key = (
            f"linkedin:{identity.verified_linkedin_slug}"
            if fit and identity.verified_linkedin_slug
            else ""
        )
        prior = (
            seen_linkedin.get(linkedin_key)
            if company_ready and linkedin_key
            else None
        )
        duplicate = prior is not None
        qualified = bool(qualified and not duplicate)
        row.update(
            company_index=index,
            company_identity_key=prior[0] if prior else identity.key,
            company_identity_alias_keys=aliases,
            company_qualified=bool(qualified),
            duplicate_company=duplicate,
        )
        if duplicate:
            row["duplicate_of_index"] = prior[1]
            row["failure_reason"] = "duplicate_company_identity"
            for field in ("icp_fit", "decision_maker", "intent_signal_raw", "intent_signal_final", "cost_penalty", "time_penalty"):
                row[field] = 0.0
        if not qualified:
            row["final_score"] = 0.0
        if contacts_required and (not company_ready or duplicate):
            row["verifier_gate_receipts"] = [
                item for item in row.get("verifier_gate_receipts") or []
                if not isinstance(item, Mapping) or item.get("gate") != "contact"
            ]
            _merge_contact_breakdown(row, _not_evaluated_contact(company))
        if qualified and linkedin_key:
            seen_linkedin[linkedin_key] = (identity.key, index)
        result.append(row)
    return result


class CompetitionCompanyScorer:
    """Use the production company judge for baseline and miner outputs."""

    def __init__(
        self,
        integrity_policy: bool = False,
        *,
        contacts_required: bool = False,
        contact_source_evidence: Mapping[str, Any] | None = None,
        company_quality: bool = False,
    ) -> None:
        self.company_quality = bool(company_quality)
        self.contacts_required = bool(contacts_required)
        self.integrity_policy = bool(integrity_policy or self.contacts_required or self.company_quality)
        self.contact_source_evidence = (
            dict(contact_source_evidence)
            if isinstance(contact_source_evidence, Mapping)
            else {}
        )

    async def __call__(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        rows = await self.score_with_breakdowns(companies, icp, is_reference_model)
        return [float(row.get("final_score") or 0.0) for row in rows]

    async def score_with_breakdowns(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        if not self.company_quality:
            return await self._score_with_breakdowns(companies, icp, is_reference_model)
        sliced = list(companies)[:_company_goal(icp)]
        raw = []
        for company in sliced:
            rows = await self._score_with_breakdowns([company], icp, is_reference_model)
            raw.append(raw_company_judgment(rows[0]) if rows else bucket_skipped_judgment())
        return apply_company_judgment_context(sliced, raw, contacts_required=self.contacts_required)

    async def _score_with_breakdowns(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[dict[str, Any]]:
        models = import_module("gateway.qualification.models")
        scorer_module = import_module("qualification.scoring.lead_scorer")
        _ensure_provider_environment()
        icp_data = _normalized_icp(icp)
        allowed_buckets = employee_count_buckets_for_icp(icp)
        icp_model = getattr(models, "ICPPrompt")(**icp_data)
        company_type = getattr(models, "CompanyOutput")
        score_company = scorer_module.score_company_competition_intent

        seen_companies: set[str] = set()
        verified_identity_key_by_alias: dict[str, str] = {}
        breakdowns: list[dict[str, Any]] = []
        for company_index, company in enumerate(
            list(companies)[: _company_goal(icp)]
        ):
            observed = (company or {}).get("employee_count")
            bucket = normalize_employee_count_bucket(
                observed, default=None
            ) or normalize_observed_employee_count_bucket(observed, default=None)
            if not bucket or bucket not in allowed_buckets:
                continue
            if self.company_quality:
                from qualification.company_quality import normalize_company_claim
                company, errors = normalize_company_claim(company)
                if errors:
                    invalid = _model_contract_incompatible_breakdown()
                    invalid["failure_reason"] = ";".join(errors)
                    if self.contacts_required:
                        _merge_contact_breakdown(invalid, _not_evaluated_contact(company))
                    breakdowns.append(invalid)
                    continue
            normalized_company = _normalized_company(
                company,
                integrity_policy=self.integrity_policy,
                contacts_required=self.contacts_required,
                company_quality=self.company_quality,
            )
            contact_company = dict(company)
            submitted_identity = canonical_company_identity(normalized_company)
            submitted_alias_keys = company_identity_alias_keys(submitted_identity)
            if (
                self.integrity_policy
                and any(
                    alias in verified_identity_key_by_alias
                    for alias in submitted_alias_keys
                )
            ):
                identity_key = next(
                    verified_identity_key_by_alias[alias]
                    for alias in submitted_alias_keys
                    if alias in verified_identity_key_by_alias
                )
                duplicate = _model_contract_incompatible_breakdown()
                duplicate.update({
                    "failure_reason": "Duplicate company identity",
                    "company_index": company_index,
                    "company_identity_key": identity_key,
                    "company_identity_alias_keys": list(submitted_alias_keys),
                    "company_qualified": False,
                    "duplicate_company": True,
                })
                if self.contacts_required:
                    _merge_contact_breakdown(
                        duplicate, _not_evaluated_contact(company)
                    )
                breakdowns.append(duplicate)
                continue
            try:
                company_model = company_type(**normalized_company)
            except ValidationError:
                incompatible = _model_contract_incompatible_breakdown()
                if self.integrity_policy:
                    incompatible.update({
                        "company_index": company_index,
                        "company_identity_key": submitted_identity.key,
                        "company_identity_alias_keys": list(submitted_alias_keys),
                        "company_qualified": False,
                        "duplicate_company": False,
                    })
                if self.contacts_required:
                    _merge_contact_breakdown(
                        incompatible, _not_evaluated_contact(company)
                    )
                breakdowns.append(incompatible)
                continue
            result = await score_company(
                company=company_model,
                icp=icp_model,
                run_cost_usd=0.0,
                run_time_seconds=0.0,
                seen_companies=(seen_companies if not self.integrity_policy else set()),
                is_reference_model=bool(is_reference_model),
                integrity_policy=self.integrity_policy,
                **({"company_quality": True} if self.company_quality else {}),
            )
            breakdown = (
                result.model_dump(mode="json")
                if hasattr(result, "model_dump")
                else dict(result)
            )
            if self.integrity_policy:
                receipts = breakdown.get("verifier_gate_receipts")
                fit_verified = company_fit_verified(receipts)
                observed_receipt = verified_identity_receipt(receipts)
                contact_company = _verified_company_for_contact(
                    company, observed_receipt
                )
                verified_identity = canonical_company_identity(
                    normalized_company,
                    verified_identity_receipt=observed_receipt,
                )
                current_identity_key = (
                    verified_identity.key if observed_receipt else submitted_identity.key
                )
                identity_alias_keys = tuple(dict.fromkeys(
                    (*submitted_alias_keys, *company_identity_alias_keys(verified_identity))
                ))
                prior_identity_key = next(
                    (
                        verified_identity_key_by_alias[alias]
                        for alias in identity_alias_keys
                        if alias in verified_identity_key_by_alias
                    ),
                    None,
                )
                duplicate_company = bool(
                    fit_verified and prior_identity_key is not None
                )
                identity_key = prior_identity_key or current_identity_key
                if duplicate_company:
                    for field, value in (
                        ("icp_fit", 0.0),
                        ("decision_maker", 0.0),
                        ("intent_signal_raw", 0.0),
                        ("time_decay_multiplier", 1.0),
                        ("intent_signal_final", 0.0),
                        ("cost_penalty", 0.0),
                        ("time_penalty", 0.0),
                        ("final_score", 0.0),
                    ):
                        breakdown[field] = value
                    breakdown["failure_reason"] = "Duplicate company identity"
                primary_verified = has_verified_primary_intent(
                    breakdown.get("intent_signals_detail") or []
                )
                company_qualified = bool(
                    fit_verified and primary_verified and not duplicate_company
                )
                breakdown.update({
                    "company_index": company_index,
                    "company_identity_key": identity_key,
                    "company_identity_alias_keys": list(identity_alias_keys),
                    "company_qualified": company_qualified,
                    "duplicate_company": duplicate_company,
                })
                if fit_verified and not duplicate_company:
                    for alias in identity_alias_keys:
                        verified_identity_key_by_alias[alias] = identity_key
            if self.contacts_required:
                if breakdown.get("company_qualified") is True:
                    from qualification.scoring.contact_verification import (
                        verify_contact,
                    )

                    contact_result = await verify_contact(
                        contact_company,
                        icp,
                        source_evidence=self.contact_source_evidence.get(
                            _contact_source_key(company)
                        ),
                        classify_role=_classify_contact_role,
                    )
                else:
                    contact_result = _not_evaluated_contact(company)
                _merge_contact_breakdown(breakdown, contact_result)
                breakdown["company_qualified"] = bool(
                    breakdown.get("company_qualified")
                    and contact_result.get("contact_qualified") is True
                )
                if not breakdown["company_qualified"]:
                    breakdown["final_score"] = 0.0
            breakdowns.append(breakdown)
        return breakdowns


def scorer_breakdown_has_retryable_infrastructure_failure(
    breakdown: Mapping[str, Any], *, integrity_policy: bool = False
) -> bool:
    if not isinstance(breakdown, Mapping):
        return False
    receipts = breakdown.get("verifier_gate_receipts")
    if isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)):
        for receipt in receipts:
            if (
                isinstance(receipt, Mapping)
                and str(receipt.get("decision") or "") == "unavailable"
                and str(receipt.get("failure_class") or "")
                not in _NON_RETRYABLE_UNAVAILABLE_FAILURE_CLASSES
            ):
                return True
    details = breakdown.get("intent_signals_detail")
    if isinstance(details, Sequence) and not isinstance(details, (str, bytes)):
        if intent_unavailability_requires_retry(
            details, integrity_policy=integrity_policy
        ):
            return True
    reason = str(breakdown.get("failure_reason") or "").strip().lower()
    return bool(reason) and any(
        marker in reason
        for marker in (
            "intent verification unavailable:",
            "llm scoring error:",
            "company verification error:",
            "company verification failed: website unreachable:",
            "company verification failed: website fetch error:",
            "company verification unavailable:",
            "company fit pre-check unavailable:",
            "company web re-verification unavailable:",
            "providerclientv2error",
            "runner external request must use https",
            "provider error",
            "provider timeout",
            "http 429",
            "no_openrouter_key",
        )
    )


def _intent_detail_is_unavailable(detail: Any) -> bool:
    if not isinstance(detail, Mapping):
        return False
    verdict = detail.get("judge_verdict")
    return isinstance(verdict, Mapping) and (
        str(verdict.get("decision") or "") == "rejected_verifier_error"
        or bool(verdict.get("error_class"))
        or str(verdict.get("pipeline_decision") or "") == "unavailable"
    )


def has_verified_primary_intent(details: Sequence[Any]) -> bool:
    """Return whether structured details contain a usable primary score."""

    rejected_statuses = {"contradicted", "unable_to_verify", "wrong_entity"}
    for detail in details:
        if not isinstance(detail, Mapping):
            continue
        index = detail.get("matched_icp_signal")
        score = detail.get("after_decay")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index != 0
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) <= 0.0
        ):
            continue
        verdict = detail.get("judge_verdict")
        if (
            not isinstance(verdict, Mapping)
            or str(verdict.get("decision") or "") != "verified"
            or bool(verdict.get("error_class"))
            or str(verdict.get("pipeline_decision") or "")
            in {"reject", "unavailable"}
        ):
            continue
        trace = verdict.get("verification_trace")
        intent_verdict = (
            trace.get("intent_verdict") if isinstance(trace, Mapping) else None
        )
        evaluations = (
            intent_verdict.get("signal_evaluations")
            if isinstance(intent_verdict, Mapping)
            else None
        )
        if isinstance(evaluations, Sequence) and not isinstance(
            evaluations, (str, bytes)
        ) and any(
            isinstance(evaluation, Mapping)
            and str(evaluation.get("signal_status") or "") in rejected_statuses
            for evaluation in evaluations
        ):
            continue
        return True
    return False


def intent_unavailability_requires_retry(
    details: Sequence[Any], *, integrity_policy: bool = False
) -> bool:
    """Retry any unavailable integrity signal; legacy needs no verified primary."""

    unavailable = any(_intent_detail_is_unavailable(detail) for detail in details)
    return unavailable and (
        integrity_policy or not has_verified_primary_intent(details)
    )


def _structured_fit_mismatch(breakdown: Mapping[str, Any]) -> bool:
    receipts = breakdown.get("verifier_gate_receipts")
    return isinstance(receipts, Sequence) and not isinstance(
        receipts, (str, bytes)
    ) and any(
        isinstance(receipt, Mapping)
        and str(receipt.get("gate") or "") == "company_fit"
        and str(receipt.get("decision") or "") == "mismatch"
        for receipt in receipts
    )


def count_penalizable_false_positives(
    breakdowns: Sequence[Mapping[str, Any]], *, icp_has_intent_signals: bool
) -> tuple[int, int]:
    gate_failures = 0
    unverified_primary = 0
    for row in breakdowns:
        if not isinstance(row, Mapping) or scorer_breakdown_has_retryable_infrastructure_failure(row):
            continue
        if _structured_fit_mismatch(row):
            gate_failures += 1
            continue
        reason = str(row.get("failure_reason") or "").strip().lower()
        if reason:
            if not any(marker in reason for marker in _NEVER_PENALIZE_MARKERS) and any(
                marker in reason for marker in _PENALIZABLE_FAILURE_MARKERS
            ):
                gate_failures += 1
                continue
        if not icp_has_intent_signals:
            continue
        details = row.get("intent_signals_detail")
        if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
            continue
        primary_verified = False
        verifier_failed = False
        for detail in details:
            if not isinstance(detail, Mapping):
                continue
            verdict = detail.get("judge_verdict")
            if isinstance(verdict, Mapping) and (
                str(verdict.get("decision") or "") == "rejected_verifier_error"
                or bool(verdict.get("error_class"))
            ):
                verifier_failed = True
            try:
                index = int(detail.get("matched_icp_signal", -1))
            except (TypeError, ValueError):
                continue
            if index == 0 and float(detail.get("after_decay") or 0.0) > 0.0:
                primary_verified = True
                break
        if details and not primary_verified and not verifier_failed:
            unverified_primary += 1
    return gate_failures, unverified_primary


def fp_penalty_total_from_breakdowns(
    breakdowns: Sequence[Mapping[str, Any]], icp: Mapping[str, Any]
) -> float:
    gate, primary = count_penalizable_false_positives(
        breakdowns,
        icp_has_intent_signals=bool(
            icp.get("intent_signals") or icp.get("intent_signal")
        ),
    )
    return float(gate + primary) * FP_PENALTY_POINTS


def competition_icp_score_from_company_scores(
    scores: Sequence[float], *, requested_count: int, fp_penalty_total: float = 0.0
) -> float:
    """Use the same capped score arithmetic as the Arena."""

    count = max(1, min(MAX_COMPANY_GOAL, int(requested_count)))
    normalized = float(
        per_icp_normalized_score(
            sorted((float(value or 0.0) for value in scores), reverse=True)[:count],
            max_leads=count,
        )
    )
    return max(0.0, normalized - max(0.0, float(fp_penalty_total)) / count)


def competition_score_from_breakdowns(
    icp: Mapping[str, Any],
    breakdowns: Sequence[Mapping[str, Any]],
    *,
    fp_penalty_points: float = FP_PENALTY_POINTS,
    fp_unverified_primary_penalty_points: float = FP_PENALTY_POINTS,
    score_floor: float = 0.0,
) -> dict[str, Any]:
    """Calculate one ICP score for both the baseline and miner bundles."""

    goal = _company_goal(icp)
    rows = [dict(row) for row in breakdowns]
    gate, primary = count_penalizable_false_positives(
        rows,
        icp_has_intent_signals=bool(
            icp.get("intent_signals") or icp.get("intent_signal")
        ),
    )
    company_scores = [float(row.get("final_score") or 0.0) for row in rows]
    normalized = float(
        per_icp_normalized_score(company_scores[:goal], max_leads=goal)
    )
    penalty = (
        gate * max(0.0, float(fp_penalty_points))
        + primary * max(0.0, float(fp_unverified_primary_penalty_points))
    ) / goal
    return {
        "per_icp_score": max(float(score_floor), normalized - penalty),
        "fp_gate_count": gate,
        "fp_unverified_primary_count": primary,
        "company_goal": goal,
        "company_scores": company_scores,
    }

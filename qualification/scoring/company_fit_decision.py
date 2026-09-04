"""Three-state company-fit decision contract.

The stable ``(passed, reason)`` API prevents non-empty decision strings from
becoming accidental truthy passes across qualification consumers.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Final, Iterable, Literal, Mapping, Optional
from urllib.parse import urlsplit


COMPANY_FIT_DECISION_CONTRACT_ID: Final = "company-fit-decision:v1"
COMPANY_FIT_DECISION_VERSION = COMPANY_FIT_DECISION_CONTRACT_ID
COMPANY_FIT_DECISIONS: Final = ("match", "mismatch", "unavailable")
COMPANY_FIT_DECISION_PRECEDENCE: Final = (
    "mismatch",
    "unavailable",
    "match",
)
COMPANY_FIT_PASSING_OUTCOME: Final = "match"
COMPANY_FIT_REQUIRED_DIMENSIONS: Final = (
    "identity",
    "employee_size",
    "industry",
    "geography",
)
COMPANY_FIT_CONDITIONAL_DIMENSIONS: Final = ("stage",)
COMPANY_FIT_DIMENSIONS: Final = (
    *COMPANY_FIT_REQUIRED_DIMENSIONS,
    *COMPANY_FIT_CONDITIONAL_DIMENSIONS,
)
COMPANY_FIT_MATCH = "match"
COMPANY_FIT_MISMATCH = "mismatch"
COMPANY_FIT_UNAVAILABLE = "unavailable"

CompanyFitDecision = Literal["match", "mismatch", "unavailable"]

_LEGAL_SUFFIXES: Final = frozenset({
    "incorporated", "corporation", "company", "limited", "holdings",
    "group", "inc", "corp", "co", "llc", "ltd", "plc", "gmbh",
    "ag", "sa", "nv", "bv", "oy", "ab", "as", "pty", "pte",
    "kk", "srl", "spa",
})
_LINKEDIN_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._%+-]{0,99}$")


def company_fit_decision_contract_identity() -> dict[str, object]:
    """Return the model-owned compatibility identity."""

    return {
        "contract_id": COMPANY_FIT_DECISION_CONTRACT_ID,
        "outcomes": list(COMPANY_FIT_DECISIONS),
        "precedence": list(COMPANY_FIT_DECISION_PRECEDENCE),
        "passing_outcome": COMPANY_FIT_PASSING_OUTCOME,
        "required_dimensions": list(COMPANY_FIT_REQUIRED_DIMENSIONS),
        "conditional_dimensions": list(COMPANY_FIT_CONDITIONAL_DIMENSIONS),
    }


def reconcile_company_fit_decisions(decisions: Iterable[Any]) -> str:
    """Apply model-owned mismatch-first decision precedence."""

    normalized = [
        value.strip().casefold() if isinstance(value, str) else ""
        for value in decisions
    ]
    if not normalized:
        return COMPANY_FIT_UNAVAILABLE
    if COMPANY_FIT_MISMATCH in normalized:
        return COMPANY_FIT_MISMATCH
    if any(value != COMPANY_FIT_MATCH for value in normalized):
        return COMPANY_FIT_UNAVAILABLE
    return COMPANY_FIT_MATCH


def aggregate_company_fit_decisions(
    decisions: Mapping[str, Any],
    *,
    stage_required: bool = False,
) -> str:
    """Aggregate a complete dimension map using the model-owned contract."""

    if not isinstance(decisions, Mapping) or not decisions:
        return COMPANY_FIT_UNAVAILABLE
    required = set(COMPANY_FIT_REQUIRED_DIMENSIONS)
    if stage_required:
        required.add("stage")
    values = [
        decisions.get(dimension, COMPANY_FIT_UNAVAILABLE)
        for dimension in required
    ]
    if any(key not in COMPANY_FIT_DIMENSIONS for key in decisions):
        values.append(COMPANY_FIT_UNAVAILABLE)
    return reconcile_company_fit_decisions(values)


def strict_company_fit_boolean(value: Any) -> Optional[bool]:
    """Accept only an actual JSON Boolean."""

    return value if isinstance(value, bool) else None


def _canonical_domain(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw or any(character.isspace() for character in raw):
        return ""
    if "://" not in raw and not raw.startswith("//"):
        raw = f"//{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if parsed.username or parsed.password:
        return ""
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    if not host or "." not in host or host.endswith(".") or ":" in host:
        return ""
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""


def _linkedin_slug(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    host = (parsed.hostname or "").casefold().removeprefix("www.")
    parts = [part for part in parsed.path.split("/") if part]
    if (
        not (host == "linkedin.com" or host.endswith(".linkedin.com"))
        or len(parts) < 2
        or parts[0].casefold() != "company"
    ):
        return ""
    slug = parts[1].casefold()
    return slug if _LINKEDIN_SLUG_RE.fullmatch(slug) else ""


def _company_name(value: Any) -> str:
    """Normalize a company name after removing terminal legal suffixes only.

    Some legal-looking tokens are also real leading name terms: ``Group Nine
    Media`` and ``AG Grid`` must not collapse into different companies merely
    because ``group`` and ``ag`` can occur as legal suffixes elsewhere.
    """

    words = re.findall(r"[a-z0-9]+", str(value or "").casefold())
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return "".join(words)


def evaluate_company_identity(
    *,
    submitted_name: Any,
    submitted_website: Any,
    submitted_linkedin: Any,
    observed_name: Any,
    observed_website: Any,
    observed_linkedin: Any = "",
    evidence_source: Any = "",
) -> dict[str, str]:
    """Bind the submitted name, website, and LinkedIn to one observed entity."""

    submitted_linkedin_raw = str(submitted_linkedin or "").strip()
    submitted = {
        "name": _company_name(submitted_name),
        "domain": _canonical_domain(submitted_website),
        "linkedin_slug": _linkedin_slug(submitted_linkedin),
    }
    observed = {
        "name": _company_name(observed_name),
        "domain": _canonical_domain(observed_website),
        "linkedin_slug": _linkedin_slug(observed_linkedin),
    }
    source = str(evidence_source or "").strip().casefold()
    receipt = {
        "decision": "unavailable",
        "reason_code": "identity_not_proven",
        "submitted_name": submitted["name"],
        "submitted_domain": submitted["domain"],
        "submitted_linkedin_slug": submitted["linkedin_slug"],
        "observed_name": observed["name"],
        "observed_domain": observed["domain"],
        "observed_linkedin_slug": observed["linkedin_slug"],
        "evidence_source": source,
    }
    if not submitted["name"] or not submitted["domain"]:
        receipt.update(decision="mismatch", reason_code="identity_unresolved")
        return receipt
    if submitted_linkedin_raw and not submitted["linkedin_slug"]:
        receipt.update(decision="mismatch", reason_code="identity_unresolved")
        return receipt
    if not observed["name"] or not observed["domain"]:
        return receipt
    if observed["domain"] != submitted["domain"]:
        receipt.update(decision="mismatch", reason_code="identity_mismatch")
        return receipt
    if submitted["linkedin_slug"] and not observed["linkedin_slug"]:
        return receipt
    if (
        submitted["linkedin_slug"]
        and observed["linkedin_slug"] != submitted["linkedin_slug"]
    ):
        # LinkedIn may expose an opaque numeric company ID on a first-party
        # homepage while the submitted record has the company's vanity slug.
        # Those two identifiers do not prove a conflict by themselves; a later
        # independently grounded web receipt can still bind the entity.
        if observed["linkedin_slug"].isdigit() != submitted["linkedin_slug"].isdigit():
            receipt.update(reason_code="identity_linkedin_alias_unresolved")
            return receipt
        receipt.update(decision="mismatch", reason_code="identity_mismatch")
        return receipt
    if observed["name"] != submitted["name"]:
        # A matching registrable domain with a different common/legal name is
        # not proof of a conflict by itself. Keep the result unavailable when
        # there is no second stable identifier to bind the alias.
        if submitted["linkedin_slug"]:
            shorter_name = min(
                (submitted["name"], observed["name"]),
                key=len,
            )
            longer_name = max(
                (submitted["name"], observed["name"]),
                key=len,
            )
            if len(shorter_name) < 4 or not longer_name.startswith(shorter_name):
                receipt.update(
                    decision="mismatch",
                    reason_code="identity_mismatch",
                )
                return receipt
        else:
            return receipt
    receipt.update(decision="match", reason_code="verifier_accepted")
    return receipt


class CompanyFitDecisionResult(tuple):
    """One decision with a real historical ``(passed, reason)`` tuple API.

    The decision and evidence stay available as named attributes.  This avoids
    a compatibility trap where an object only *looks* iterable but fails tuple
    checks in older validator code.
    """

    def __new__(
        cls,
        decision: CompanyFitDecision,
        reason: Optional[str] = None,
        *,
        details: Optional[Mapping[str, Any]] = None,
    ) -> "CompanyFitDecisionResult":
        if decision not in {
            COMPANY_FIT_MATCH,
            COMPANY_FIT_MISMATCH,
            COMPANY_FIT_UNAVAILABLE,
        }:
            raise ValueError(f"invalid company fit decision: {decision!r}")
        instance = super().__new__(cls, (decision == COMPANY_FIT_MATCH, reason))
        instance.decision = decision
        instance.details = copy.deepcopy(dict(details or {}))
        return instance

    @property
    def passed(self) -> bool:
        return bool(tuple.__getitem__(self, 0))

    @property
    def reason(self) -> Optional[str]:
        return tuple.__getitem__(self, 1)

    def __bool__(self) -> bool:
        return self.passed

    def __copy__(self) -> "CompanyFitDecisionResult":
        """Return an independent decision object for legacy copy callers."""

        return type(self)(
            self.decision,
            self.reason,
            details=copy.deepcopy(self.details),
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> "CompanyFitDecisionResult":
        """Preserve named decision state when copying this tuple subclass."""

        cloned = type(self)(
            self.decision,
            self.reason,
            details=copy.deepcopy(self.details, memo),
        )
        memo[id(self)] = cloned
        return cloned

    def __reduce_ex__(self, protocol: int):
        """Pickle the decision, rather than this class's legacy tuple payload."""

        return (
            _restore_company_fit_decision_result,
            (self.decision, self.reason, copy.deepcopy(self.details)),
        )

    def receipt(self, gate: str) -> dict:
        receipt = {
            "gate": str(gate),
            "contract_id": COMPANY_FIT_DECISION_CONTRACT_ID,
            "contract_version": COMPANY_FIT_DECISION_VERSION,
            "decision": self.decision,
            "reason": self.reason,
        }
        receipt.update(self.details)
        return receipt


def _restore_company_fit_decision_result(
    decision: CompanyFitDecision,
    reason: Optional[str],
    details: Mapping[str, Any],
) -> CompanyFitDecisionResult:
    """Unpickle a result without replaying the historical Boolean tuple."""

    return CompanyFitDecisionResult(decision, reason, details=details)


def company_fit_match(
    reason: Optional[str] = None,
    *,
    details: Optional[Mapping[str, Any]] = None,
) -> CompanyFitDecisionResult:
    return CompanyFitDecisionResult(COMPANY_FIT_MATCH, reason, details=details)


def company_fit_mismatch(
    reason: Optional[str],
    *,
    details: Optional[Mapping[str, Any]] = None,
) -> CompanyFitDecisionResult:
    return CompanyFitDecisionResult(COMPANY_FIT_MISMATCH, reason, details=details)


def company_fit_unavailable(
    reason: Optional[str],
    *,
    details: Optional[Mapping[str, Any]] = None,
) -> CompanyFitDecisionResult:
    return CompanyFitDecisionResult(COMPANY_FIT_UNAVAILABLE, reason, details=details)

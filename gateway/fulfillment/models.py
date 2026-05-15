"""
Pydantic models for the lead fulfillment system.

FulfillmentICP and FulfillmentLead constrain industry/sub_industry/role_type
to canonical taxonomy values so Tier 1 ICP Fit Gate checks are free
deterministic equality comparisons.
"""

import re
from typing import List, Optional, Union
from pydantic import BaseModel, Field, field_validator

from gateway.qualification.models import (
    IntentSignal,
    IntentSignalSource,  # noqa: F401 — re-exported for convenience
    Seniority,
    LeadOutput,
    ICPPrompt,
)


# ---------------------------------------------------------------------------
# IntentSignalSpec — structured form for buyer-side intent signals
# ---------------------------------------------------------------------------

class IntentSignalSpec(BaseModel):
    """One buyer-side intent signal on a fulfillment request.

    Was previously a bare ``str``. Promoted to a structured object so the
    operator can mark individual signals as ``required`` — meaning the
    lead must actually deliver verified evidence of this signal or it
    fails scoring with ``missing_required_intent_signal``.

    Every signal contributes to ``intent_signal_final`` once verified;
    there is no binary / unscored mode. (An earlier draft of this model
    carried an ``is_scored`` flag for binary yes/no signals; the flag
    was removed to keep scoring uniform. The Pydantic config silently
    accepts and discards a stray ``is_scored`` key on legacy inputs so
    in-flight rows from the brief window where the flag was live still
    parse cleanly.)

    Wire format on miner-visible /fulfillment/requests/active:
      ``{"text": str, "required": bool}``

    Back-compat: ``FulfillmentICP.intent_signals`` accepts the legacy
    ``List[str]`` form and coerces each entry to
    ``IntentSignalSpec(text=s, required=False)``. Every existing
    icp_details JSON row in the DB therefore re-parses cleanly with
    identical behaviour to before this change.
    """

    text: str = Field(..., min_length=1, max_length=350)
    required: bool = Field(
        default=False,
        description=(
            "If true, the lead MUST have at least one miner-supplied "
            "intent signal that the LLM matches to this spec AND that "
            "passes URL verification (after_decay_score > 0). Otherwise "
            "the lead fails scoring with 'missing_required_intent_signal'."
        ),
    )

    # Tolerate (and discard) legacy ``is_scored`` keys that may still be
    # sitting in icp_details rows from the short window where the flag
    # was wired through the pipeline. Pydantic v2 silently ignores
    # extras unless ``model_config["extra"]="forbid"``, but we set this
    # explicitly so the intent stays in the source rather than living
    # in defaults. New requests should never include the field.
    model_config = {"extra": "ignore"}

    @field_validator("text", mode="before")
    @classmethod
    def _strip_text(cls, v) -> str:
        if v is None:
            raise ValueError("intent signal text cannot be null")
        s = str(v).strip()
        if not s:
            raise ValueError("intent signal text cannot be empty")
        return s


def _coerce_intent_signal_spec(entry) -> IntentSignalSpec:
    """Coerce one entry of the inbound ``intent_signals`` list into an
    ``IntentSignalSpec``.

    Accepts:
      * ``str``                            → default flags
      * ``IntentSignalSpec`` instance      → as-is
      * ``dict`` with at least ``text``    → constructed via Pydantic
      * ``None`` / empty                   → caller filters these out
    """
    if entry is None:
        raise ValueError("intent signal entry cannot be null")
    if isinstance(entry, IntentSignalSpec):
        return entry
    if isinstance(entry, str):
        return IntentSignalSpec(text=entry)
    if isinstance(entry, dict):
        # Tolerate legacy dicts that may have been written with extra
        # keys (e.g. heuristic parser drafts, or rows from the short
        # window where ``is_scored`` was live). We pass only ``text`` and
        # ``required`` into ``IntentSignalSpec``; stray keys never reach
        # the constructor.
        return IntentSignalSpec(
            text=entry.get("text", entry.get("signal", entry.get("name", ""))),
            required=bool(entry.get("required", False)),
        )
    raise ValueError(
        f"intent_signals entry must be str, dict, or IntentSignalSpec, "
        f"got {type(entry).__name__}"
    )
try:
    from gateway.utils.industry_taxonomy import INDUSTRY_TAXONOMY
except ImportError:
    from validator_models.industry_taxonomy import INDUSTRY_TAXONOMY

# ---------------------------------------------------------------------------
# Taxonomy constraint sets (derived from the canonical taxonomy)
# ---------------------------------------------------------------------------
# Keys are sub-industries; values contain parent industry lists.
VALID_SUB_INDUSTRIES: set = set(INDUSTRY_TAXONOMY.keys())
VALID_INDUSTRIES: set = {
    ind for entry in INDUSTRY_TAXONOMY.values() for ind in entry["industries"]
}
SUB_INDUSTRY_TO_PARENTS: dict = {
    sub: entry["industries"] for sub, entry in INDUSTRY_TAXONOMY.items()
}

VALID_ROLE_TYPES: set = {
    "C-Level Executive", "VP", "Director", "Manager",
    "Sales", "Marketing", "Engineering", "Product",
    "Operations", "Finance", "HR", "Legal",
    "IT", "Customer Success", "Business Development",
    "Data & Analytics", "Design", "Research",
    "Supply Chain", "Consulting", "Other",
}


# ---------------------------------------------------------------------------
# Canonical employee-count buckets
# ---------------------------------------------------------------------------
# Must stay in sync with VALID_EMPLOYEE_COUNTS in gateway/api/submit.py — that's
# the vocabulary miners are allowed to submit, so clients must request in the
# same vocabulary for exact-match scoring to work.  Format mirrors LinkedIn's
# scraped strings (with thousands separators on the wider buckets).
CANONICAL_EMPLOYEE_BUCKETS: List[str] = [
    "0-1", "2-10", "11-50", "51-200", "201-500",
    "501-1,000", "1,001-5,000", "5,001-10,000", "10,001+",
]
_BUCKET_RANGES: dict = {
    "0-1":          (0, 1),
    "2-10":         (2, 10),
    "11-50":        (11, 50),
    "51-200":       (51, 200),
    "201-500":      (201, 500),
    "501-1,000":    (501, 1_000),
    "1,001-5,000":  (1_001, 5_000),
    "5,001-10,000": (5_001, 10_000),
    "10,001+":      (10_001, 10_000_000),
}


def _parse_legacy_range_to_bounds(s: str) -> Optional[tuple]:
    """Turn a legacy free-form range string like ``"200-5000"`` or ``"500+"``
    or ``"1000"`` into a ``(lo, hi)`` numeric tuple.  Returns ``None`` if the
    input can't be parsed."""
    s = (s or "").strip().replace(",", "")
    if not s:
        return None
    if re.match(r"^\d+$", s):
        n = int(s)
        return (n, n)
    m = re.match(r"^(\d+)-(\d+)$", s)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return (lo, hi) if lo <= hi else None
    m = re.match(r"^(\d+)\+$", s)
    if m:
        return (int(m.group(1)), 10_000_000)
    return None


def range_string_to_buckets(s: str) -> List[str]:
    """Coerce a legacy free-form employee-count range string to the list of
    canonical buckets whose numeric range is FULLY CONTAINED within it.

    Endpoint-touching partials are deliberately excluded — ``"200-5000"``
    matches ``"201-500"``/``"501-1,000"``/``"1,001-5,000"`` but NOT
    ``"51-200"`` (which would overlap only at a single employee).
    """
    bounds = _parse_legacy_range_to_bounds(s)
    if bounds is None:
        return []
    lo, hi = bounds
    return [
        b for b, (b_lo, b_hi) in _BUCKET_RANGES.items()
        if b_lo >= lo and b_hi <= hi
    ]


# ---------------------------------------------------------------------------
# Company-name scrubbing helper
# ---------------------------------------------------------------------------

COMPANY_PLACEHOLDER = "[company_name]"


def scrub_company_name(text: str, company: str) -> str:
    """Replace every whole-word occurrence of ``company`` in ``text`` with
    ``[company_name]``, case-insensitive.

    Used by the fulfillment request ingestion path so miners never see the
    identity of the client who submitted the request.  Possessive forms
    (``AcmeCorp's``) are preserved because the ``\\b`` word boundary ends
    before the apostrophe, so the ``'s`` stays attached to the placeholder
    (``[company_name]'s``).

    Short or generic company names (e.g. ``"Apple"``) may match unrelated
    occurrences of the same token elsewhere in the text.  That is an
    accepted trade-off; callers should supply a distinctive name.

    Returns the input unchanged when either ``text`` or ``company`` is
    empty.  Never raises.
    """
    if not text or not company:
        return text or ""

    pattern = re.compile(r"\b" + re.escape(company) + r"\b", re.IGNORECASE)
    scrubbed = pattern.sub(COMPANY_PLACEHOLDER, text)
    # Collapse any accidental whitespace runs that could have been created
    # if the original text had odd spacing around the name.
    scrubbed = re.sub(r"[ \t]{2,}", " ", scrubbed).strip()
    return scrubbed


# ---------------------------------------------------------------------------
# FulfillmentICP
# ---------------------------------------------------------------------------

class FulfillmentICP(BaseModel):
    """ICP published to miners for a fulfillment request."""

    icp_id: str = Field(default="")
    prompt: str = Field(..., min_length=1)
    # Industries and sub-industries the client will accept.  A single client
    # ICP frequently spans multiple verticals (e.g. local-business outreach
    # covering restaurants + gyms + medspas), so both fields are lists and
    # the Tier 1 gate accepts a lead matching ANY listed value (set
    # membership).  The validators (mode="before") accept either a list or
    # a single string and normalize to a list — required for back-compat
    # with historical icp_details JSON in the DB that was stored as a
    # plain string before this column became multi-valued.
    industry: List[str] = Field(default_factory=list)
    sub_industry: List[str] = Field(default_factory=list)
    target_role_types: List[str] = Field(default_factory=list)
    target_roles: List[str] = Field(default_factory=list)
    target_seniority: str = ""
    # List of canonical employee-count buckets (e.g.
    # ``["201-500", "501-1,000", "1,001-5,000"]``) that the client will
    # accept for this request.  Must use the same vocabulary miners are
    # allowed to submit (see gateway/api/submit.py::VALID_EMPLOYEE_COUNTS)
    # so Tier 1 ICP Fit is a pure set-membership check on exact strings.
    #
    # The field validator below also accepts a legacy free-form range
    # string like ``"200-5000"`` (for backward compat with older requests
    # that stored a single string) — it coerces into the list of canonical
    # buckets whose numeric range is FULLY CONTAINED within the requested
    # range (see range_string_to_buckets).
    employee_count: List[str] = Field(default_factory=list)
    company_stage: str = ""
    geography: str = ""
    # List of target countries the client will accept.  A single client ICP
    # frequently spans multiple countries (e.g. LATAM = 12 countries,
    # Nordics = 5, "Western Europe" = ~10), so this is a list and the Tier 1
    # gate accepts a lead matching ANY listed country (set membership, with
    # alias normalization through _normalize_country in scoring.py).
    #
    # Empty list = "any country accepted" (Tier 1 country check skipped
    # entirely, same as the legacy ``country=""`` default).
    #
    # The field validator below accepts BOTH a list and a single string —
    # the single-string form is required for back-compat with historical
    # icp_details JSON in the DB (every fulfillment_requests row created
    # before this column became multi-valued has ``"country": "United States"``
    # or similar).  Without that coercion, validator-side
    # ``FulfillmentICP(**icp_details)`` re-parses would crash on every
    # in-flight legacy request and wedge the entire scoring pipeline.
    country: List[str] = Field(default_factory=list)
    product_service: str = ""
    # Each spec is one buyer-side buying signal. Was previously
    # ``List[str]``; promoted to a list of structured objects with a
    # per-signal ``required`` flag. The validator below coerces a
    # legacy ``List[str]`` (or list of dicts) on the way in, so every
    # historical icp_details row still re-parses cleanly with the
    # default ``required=False``.
    intent_signals: List[IntentSignalSpec] = Field(default_factory=list)
    # Companies whose leads must be rejected at Tier 1 for this request.
    # Populated either (a) explicitly by the client in the create payload
    # (client-provided list takes precedence) or (b) automatically by the
    # gateway at create_request time, by pulling the set of company names
    # this client has already received as winners in prior FULFILLED
    # requests for the same `company` (matched via the non-nullable
    # client-company column on fulfillment_requests).  Intentionally NOT
    # Field(exclude=True) — miners need to see it in
    # /fulfillment/requests/active so they can pre-filter their search
    # and avoid wasting work on companies that would be Tier-1-rejected.
    excluded_companies: List[str] = Field(default_factory=list)
    num_leads: int = 10
    window_end: Optional[str] = None
    reveal_window_end: Optional[str] = None

    # Internal-only label for client identification in Supabase dashboards
    # (e.g. "Edward Burrowes 1").  Persisted to the dedicated `internal_label`
    # column on `fulfillment_requests`, NOT inside `icp_details`, so it is
    # never returned by /fulfillment/requests/active and miners never see it.
    # exclude=True makes model_dump() drop it, so it can't leak into the
    # hash/jsonb by accident.
    internal_label: str = Field(default="", exclude=True)

    # Client company name (e.g. "AcmeCorp").  Stored in the dedicated
    # `company` column on fulfillment_requests.  The gateway's create_request
    # endpoint additionally scrubs every occurrence of this string from the
    # free-text ICP fields (prompt, product_service, intent_signals,
    # target_roles) before persisting, replacing each match with
    # "[company_name]" so miners can never learn which client made the
    # request.  Like internal_label, Field(exclude=True) guarantees it never
    # reaches model_dump() -> icp_details -> miners.
    #
    # IMPORTANT: Defaults to "" at the MODEL layer because the gateway's
    # model_dump() strips this field before persisting to icp_details, so
    # when the validator later reconstructs FulfillmentICP(**icp_details)
    # the field will be absent.  Marking it required on the model would
    # crash every re-parse with "Field required [type=missing]" and wedge
    # the entire scoring pipeline (observed: 9 requests stuck for 13h+).
    # Enforcement of "company must be non-empty on client POST" is handled
    # explicitly in gateway/fulfillment/api.py::create_request rather than
    # via min_length on the field.
    company: str = Field(default="", exclude=True)

    # Gateway-only (create_request): when True (default), ``target_roles`` is
    # expanded via ``role_expander`` once before hashing. When False, the
    # submitted list is stored verbatim so operators can pin an exact title
    # set. Never serialized to ``icp_details`` / miners.
    expand_target_roles: bool = Field(default=True, exclude=True)

    @field_validator("expand_target_roles", mode="before")
    @classmethod
    def coerce_expand_target_roles(cls, v) -> bool:
        if v is None or v == "":
            return True
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(v)
        if isinstance(v, str):
            s = v.strip().lower()
            if not s:
                return True
            if s in ("false", "0", "no", "off"):
                return False
            if s in ("true", "1", "yes", "on"):
                return True
        raise ValueError(
            "expand_target_roles must be a boolean (or common string coercions "
            "true/false/1/0)"
        )

    @field_validator("intent_signals", mode="before")
    @classmethod
    def normalize_intent_signals(cls, v) -> List[IntentSignalSpec]:
        """Coerce ``intent_signals`` to ``List[IntentSignalSpec]``.

        Accepts:
          * ``None`` / ``""`` / ``[]``  → ``[]``
          * ``List[str]``               → each str → default-flag spec
                                          (handles every historical
                                          ``icp_details`` row in the DB)
          * ``List[dict]``              → each dict constructed via
                                          ``_coerce_intent_signal_spec``;
                                          missing flags default to
                                          ``required=False``. A stray
                                          ``is_scored`` key (legacy)
                                          is silently discarded.
          * ``List[IntentSignalSpec]``  → passed through
          * Mixed list                  → each entry coerced
            individually (operator can switch from legacy text-only
            entries to flag-bearing dicts incrementally inside one
            request).

        Empty / null entries are silently dropped so a stray empty
        textarea line in the admin UI doesn't crash the request.
        """
        if v is None or v == "" or v == []:
            return []
        if not isinstance(v, list):
            raise ValueError(
                f"intent_signals must be a list, got {type(v).__name__}"
            )
        out: List[IntentSignalSpec] = []
        for entry in v:
            # Skip stray blanks (admin textarea sometimes round-trips
            # trailing newlines as empty strings).
            if entry is None:
                continue
            if isinstance(entry, str) and not entry.strip():
                continue
            if isinstance(entry, dict) and not str(entry.get("text", "")).strip():
                continue
            out.append(_coerce_intent_signal_spec(entry))
        return out

    @field_validator("excluded_companies", mode="before")
    @classmethod
    def normalize_excluded_companies(cls, v) -> List[str]:
        """Strip entries, drop empties, dedupe case-insensitively while
        preserving the first casing seen.  Keeps the miner-facing payload
        tidy without changing the scoring contract (which lowercases both
        sides before comparing)."""
        if v is None or v == "":
            return []
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, list):
            raise ValueError(
                f"excluded_companies must be a list, got {type(v).__name__}"
            )
        seen = set()
        out: List[str] = []
        for entry in v:
            s = str(entry).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @field_validator("industry", mode="before")
    @classmethod
    def validate_industry(cls, v) -> List[str]:
        """Coerce to ``List[str]`` of taxonomy-valid industries.

        Accepts:
          * ``None`` / ``""`` / ``[]``   -> ``[]``
          * Single ``str``               -> ``[str]`` (back-compat with
            legacy icp_details rows where industry was stored as a plain
            string before the column became multi-valued)
          * Python-repr stringified list ``"['X', 'Y']"`` -> parsed list
            (recovers from legacy DB rows / CSV round-trips that lost the
            JSON list shape; see icp_checks._coerce_industry_list rationale)
          * ``List[str]``                -> validated, deduped (case-sensitive
            because taxonomy keys are canonical-case strings)

        Rejects any entry not present in ``VALID_INDUSTRIES``.
        """
        if v is None or v == "" or v == []:
            return []
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        v = list(parsed)
                    else:
                        v = [s]
                except (ValueError, SyntaxError):
                    v = [s]
            else:
                v = [s]
        if not isinstance(v, list):
            raise ValueError(
                f"industry must be a list or string, got {type(v).__name__}"
            )
        seen = set()
        out: List[str] = []
        for entry in v:
            s = str(entry).strip()
            if not s or s in seen:
                continue
            if s not in VALID_INDUSTRIES:
                raise ValueError(
                    f"Industry '{s}' not in taxonomy. Valid: {sorted(VALID_INDUSTRIES)}"
                )
            seen.add(s)
            out.append(s)
        return out

    @field_validator("sub_industry", mode="before")
    @classmethod
    def validate_sub_industry(cls, v, info) -> List[str]:
        """Coerce to ``List[str]`` of taxonomy-valid sub-industries.

        Each entry must belong to AT LEAST ONE of the request's
        ``industry`` entries (when ``industry`` is non-empty); otherwise
        the cross-check is skipped.  This lets a client target multiple
        sub-industries spanning multiple parent industries (e.g.
        ``industry=["Food and Beverage", "Health Care"]``,
        ``sub_industry=["Restaurants", "Fitness"]``).

        Also recovers Python-repr stringified lists (``"['X', 'Y']"``) the
        same way ``validate_industry`` does, for legacy DB rows.
        """
        if v is None or v == "" or v == []:
            return []
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                try:
                    import ast
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple)):
                        v = list(parsed)
                    else:
                        v = [s]
                except (ValueError, SyntaxError):
                    v = [s]
            else:
                v = [s]
        if not isinstance(v, list):
            raise ValueError(
                f"sub_industry must be a list or string, got {type(v).__name__}"
            )

        industries = info.data.get("industry") or []
        if isinstance(industries, str):
            industries = [industries] if industries else []

        seen = set()
        out: List[str] = []
        for entry in v:
            s = str(entry).strip()
            if not s or s in seen:
                continue
            if s not in VALID_SUB_INDUSTRIES:
                raise ValueError(f"Sub-industry '{s}' not in taxonomy")
            if industries:
                # Build the set of valid parent industries for this
                # sub-industry from BOTH directions of the taxonomy graph,
                # matching the legacy single-value validator's tolerance.
                parents = set(SUB_INDUSTRY_TO_PARENTS.get(s, []))
                parents.update(INDUSTRY_TAXONOMY.get(s, {}).get("industries", []))
                if not (parents & set(industries)):
                    raise ValueError(
                        f"Sub-industry '{s}' does not belong to any of "
                        f"the listed industries {industries}"
                    )
            seen.add(s)
            out.append(s)
        return out

    @field_validator("country", mode="before")
    @classmethod
    def validate_country(cls, v) -> List[str]:
        """Coerce ``country`` to ``List[str]``.

        Accepts:
          * ``None`` / ``""`` / ``[]``      -> ``[]`` (Tier 1 country check
            skipped — "any country accepted")
          * Single ``str``                   -> ``[str]`` (back-compat with
            legacy icp_details rows where ``country`` was stored as a plain
            string before the column became multi-valued — every
            ``fulfillment_requests`` row created before this change has
            this shape, and the validator must continue to re-parse them
            without crashing or scoring will wedge for in-flight requests)
          * ``List[str]``                    -> stripped, deduped (case-
            insensitive on the dedup key, but original casing preserved
            for display / failure-detail messages)

        Whitespace-only entries are dropped.  No taxonomy validation here
        because there's no canonical "valid country" set in the project
        (qualification has its own list of 199 valid countries used at
        lead-submit time, but fulfillment historically accepted any free-
        form country string and we shouldn't tighten that contract here).
        Normalization for the Tier 1 comparison happens in
        ``gateway/fulfillment/scoring.py::_normalize_country`` at check
        time so this validator stays cheap.
        """
        if v is None or v == "" or v == []:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if not isinstance(v, list):
            raise ValueError(
                f"country must be a list or string, got {type(v).__name__}"
            )
        seen = set()
        out: List[str] = []
        for entry in v:
            s = str(entry).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    @field_validator("target_role_types")
    @classmethod
    def validate_role_types(cls, v: List[str]) -> List[str]:
        invalid = [r for r in v if r not in VALID_ROLE_TYPES]
        if invalid:
            raise ValueError(f"Invalid role types: {invalid}. Valid: {sorted(VALID_ROLE_TYPES)}")
        return v

    @field_validator("employee_count", mode="before")
    @classmethod
    def validate_employee_count(cls, v) -> List[str]:
        """Normalize employee_count to a list of canonical buckets.

        Accepts:
          * ``[]`` / ``""`` / ``None``                   -> ``[]``
          * ``List[str]`` of canonical buckets           -> unchanged (deduped)
          * Legacy ``str`` range like ``"200-5000"``,
            ``"500+"``, or ``"1000"``                    -> coerced via
            range_string_to_buckets() to the list of buckets whose numeric
            range is fully contained within the input range.

        Rejects any list entry that isn't in CANONICAL_EMPLOYEE_BUCKETS.
        """
        if v is None or v == "" or v == []:
            return []
        if isinstance(v, str):
            buckets = range_string_to_buckets(v)
            if not buckets:
                raise ValueError(
                    f"Invalid employee_count range: '{v}'. Provide a list of "
                    f"canonical buckets or a range string fully containing at "
                    f"least one of {CANONICAL_EMPLOYEE_BUCKETS}."
                )
            return buckets
        if isinstance(v, list):
            seen = set()
            out: List[str] = []
            for entry in v:
                e = str(entry).strip()
                if e in seen:
                    continue
                if e not in CANONICAL_EMPLOYEE_BUCKETS:
                    raise ValueError(
                        f"employee_count entry '{e}' not in canonical buckets "
                        f"{CANONICAL_EMPLOYEE_BUCKETS}"
                    )
                seen.add(e)
                out.append(e)
            return out
        raise ValueError(
            f"Invalid employee_count type {type(v).__name__}; expected list or str"
        )

    def to_icp_prompt(self) -> ICPPrompt:
        """Convert to ICPPrompt for scoring functions.

        ``ICPPrompt.employee_count`` is still a single ``str`` across many
        downstream consumers (qualification/sourcing miners, validator
        ICPPrompt re-parses, etc.), so we collapse the list of allowed
        buckets to the smallest range string that contains all of them.
        Example: ``["201-500", "501-1,000", "1,001-5,000"]`` -> ``"201-5000"``.
        """
        roles = self.target_roles or self.target_role_types
        buckets = self.employee_count or []
        if buckets:
            los, his = zip(*(_BUCKET_RANGES[b] for b in buckets))
            ec_str = f"{min(los)}-{max(his)}"
        else:
            ec_str = ""
        # ``ICPPrompt`` is the legacy single-value schema shared with the
        # qualification/sourcing pipeline; collapse the multi-value
        # ``industry``/``sub_industry`` lists to comma-joined strings so
        # downstream consumers still receive a non-empty value.  The only
        # field of ``icp_prompt`` that fulfillment scoring actually reads
        # is ``intent_signals`` (see scoring.py L389), so the collapsed
        # form is purely cosmetic for the rest.
        industry_str = ", ".join(self.industry) if self.industry else ""
        sub_industry_str = ", ".join(self.sub_industry) if self.sub_industry else ""
        # ``ICPPrompt.country`` is still a single ``str`` (shared with
        # qualification/sourcing miners that don't yet handle multi-country
        # ICPs).  Collapse the list to a comma-joined string the same way
        # we do for industry / sub_industry above so downstream consumers
        # at least receive readable target context, even if their Tier 1
        # check still uses single-value equality.
        country_str = ", ".join(self.country) if self.country else ""
        # ``ICPPrompt.intent_signals`` is the legacy ``List[str]`` shape
        # that the shared lead_scorer's LLM prompt expects (the prompt
        # renders BUYER'S EXPECTED INTENT SIGNALS as a numbered list and
        # the LLM only needs the TEXT to score + match an index — it
        # does NOT need the required flag). We therefore project the
        # structured specs down to plain text here. The fulfillment
        # scorer in scoring.py separately consults
        # ``self.intent_signals`` (the structured specs) for the
        # required-pass check, so the flag IS enforced — just not
        # surfaced inside the per-signal LLM scoring prompt where it
        # would be irrelevant.
        intent_signal_texts = [s.text for s in self.intent_signals]
        return ICPPrompt(
            icp_id=self.icp_id,
            prompt=self.prompt,
            industry=industry_str,
            sub_industry=sub_industry_str,
            target_roles=roles,
            target_seniority=self.target_seniority,
            employee_count=ec_str,
            company_stage=self.company_stage,
            geography=self.geography,
            country=country_str,
            product_service=self.product_service,
            intent_signals=intent_signal_texts,
        )


# ---------------------------------------------------------------------------
# FulfillmentLead
# ---------------------------------------------------------------------------

class FulfillmentLead(BaseModel):
    """Lead schema with PII — used in fulfillment commit-reveal.

    All fields are required except ``phone``.  Miners that submit
    sparse leads will be rejected at parse time rather than silently
    scoring zero.
    """

    # PII fields (included in hash, stripped by to_lead_output)
    full_name: str
    email: str
    linkedin_url: str
    phone: str = ""

    # Company info
    business: str
    company_linkedin: str
    company_website: str
    employee_count: str

    # Company HQ location (used for ICP country/state matching)
    company_hq_country: str
    company_hq_state: str
    company_hq_city: str = ""

    # Industry
    industry: str
    sub_industry: str

    # Company description (free-form, written by the miner).
    # REQUIRED — this flows into the validator's Stage 5 classification
    # pipeline (validator_models/stage5_verification.py::classify_company_industry),
    # which performs a 3-stage check:
    #   1. Compare miner description against scraped website/LinkedIn content
    #      (INVALID → stage1_invalid_description → reject)
    #   2. Embed the refined description
    #   3. LLM ranks top-3 industry/sub_industry pairs
    # If the description is missing or doesn't match the website, Stage 5
    # rejects the lead BEFORE intent scoring runs, the same way sourcing
    # rejects leads with bad descriptions.
    description: str = Field(..., min_length=30)

    # Contact location
    country: str
    city: str
    state: str

    # Role
    role: str
    role_type: str
    seniority: str

    # Intent
    intent_signals: List[IntentSignal] = Field(..., min_length=1)

    @field_validator("industry")
    @classmethod
    def validate_industry(cls, v: str) -> str:
        if v not in VALID_INDUSTRIES:
            raise ValueError(f"Industry '{v}' not in taxonomy. Valid: {sorted(VALID_INDUSTRIES)}")
        return v

    @field_validator("sub_industry")
    @classmethod
    def validate_sub_industry(cls, v: str) -> str:
        if v not in VALID_SUB_INDUSTRIES:
            raise ValueError(f"Sub-industry '{v}' not in taxonomy. Valid: {sorted(VALID_SUB_INDUSTRIES)}")
        return v

    @field_validator("role_type")
    @classmethod
    def validate_role_type(cls, v: str) -> str:
        if v not in VALID_ROLE_TYPES:
            raise ValueError(f"Role type '{v}' not valid. Valid: {sorted(VALID_ROLE_TYPES)}")
        return v

    def to_lead_output(self) -> LeadOutput:
        """Strip PII and convert to LeadOutput for scoring functions.

        LeadOutput's country/state are company-level fields, so we map
        from the HQ fields here (contact-level country/state stay on
        the FulfillmentLead only).
        """
        seniority_value = self.seniority
        seniority_map = {"Senior": "Manager"}
        if seniority_value in seniority_map:
            seniority_value = seniority_map[seniority_value]

        return LeadOutput(
            lead_id=0,
            business=self.business,
            company_linkedin=self.company_linkedin,
            company_website=self.company_website,
            employee_count=self.employee_count,
            industry=self.industry,
            sub_industry=self.sub_industry,
            country=self.company_hq_country,
            city=self.city,
            state=self.company_hq_state,
            role=self.role,
            role_type=self.role_type,
            seniority=seniority_value,
            intent_signals=self.intent_signals,
        )

    def to_validator_dict(self) -> dict:
        """Convert to dict with keys expected by validator_models check functions.

        The validator extraction utilities (get_website, get_linkedin,
        get_first_name, get_last_name, etc.) expect specific key names
        that differ from FulfillmentLead fields.  The returned dict is
        intentionally mutable — Stage 0-2 checks add fields like
        ``domain_age_days``, ``has_mx``, etc. in-place, and Stage 4-5
        reads them back.
        """
        d = self.model_dump(exclude={"intent_signals"})
        d["website"] = self.company_website
        d["linkedin"] = self.linkedin_url
        d["hq_country"] = self.company_hq_country
        d["hq_state"] = self.company_hq_state
        d["hq_city"] = self.company_hq_city
        parts = self.full_name.strip().split(None, 1)
        d["first"] = parts[0] if parts else ""
        d["last"] = parts[1] if len(parts) > 1 else ""
        return d


# ---------------------------------------------------------------------------
# Commit / Reveal request models
# ---------------------------------------------------------------------------

class CommitHashEntry(BaseModel):
    """Single lead hash submitted during commit (no lead_id yet)."""
    hash: str


class LeadHashEntry(BaseModel):
    """Lead hash with gateway-assigned ID (stored after commit)."""
    lead_id: str
    hash: str


class FulfillmentCommitRequest(BaseModel):
    """Miner commit payload — hashes only, no lead data."""
    request_id: str
    miner_hotkey: str
    lead_hashes: List[CommitHashEntry]
    schema_version: int
    signature: str
    timestamp: int
    nonce: str


class FulfillmentRevealRequest(BaseModel):
    """Miner reveal payload — full lead data."""
    request_id: str
    submission_id: str
    miner_hotkey: str
    leads: List[FulfillmentLead]
    signature: str
    timestamp: int
    nonce: str


# ---------------------------------------------------------------------------
# Score result
# ---------------------------------------------------------------------------

class FulfillmentScoreResult(BaseModel):
    """Per-lead, per-validator score result."""
    lead_id: str = ""
    tier1_passed: bool = False
    tier2_passed: bool = False
    email_verified: bool = False
    person_verified: bool = False
    company_verified: bool = False
    rep_score: float = 0.0
    intent_signal_raw: float = 0.0
    intent_signal_final: float = 0.0
    intent_decay_multiplier: float = 0.0
    final_score: float = 0.0
    all_fabricated: bool = False
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None
    # Per-miner-signal breakdown, only populated when Tier 3 intent scoring runs.
    # Each entry maps a miner-submitted signal to the best-matching client
    # (ICP) intent signal, plus the raw/after-decay score for that signal.
    # Fields per entry:
    #   url, description, snippet, date, source
    #   raw_score, after_decay_score, decay_multiplier, confidence, date_status
    #   matched_icp_signal_idx (int, -1 if no match)
    #   matched_icp_signal (str or None)
    intent_signals_detail: List[dict] = Field(default_factory=list)

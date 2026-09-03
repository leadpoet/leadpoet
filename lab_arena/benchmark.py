"""Lab Arena V1 daily benchmark generation (labarena.md sections 5.2, 8, 18.6;
labarenaaudit.md blocker 1).

The Lab generator file ``gateway/tasks/icp_generator.py`` is an enclave build
input and its count parameter is not effective, so the Arena owns its own
generation flow here. This module imports only the Lab's pure constants and
normalization helpers, carries a count-parameterized copy of the Lab prompt
(byte-identical to the Lab's for twenty ICPs over the full industry list),
copies the Lab's exclusion prompt and parser, and applies the Lab's ICP
contract after generation. It never calls the Lab's store, activation, or
rotation functions and never reads a Lab table.

Every generation attempt is durable before the next one: the request document
is journaled before dispatch, the raw response object is written to the
object store and journaled before validation, and a request without a
terminal outcome is journaled as ``unknown`` before it is retried. A restart
replays the journal by re-executing the same deterministic flow against the
stored objects, verifying every recorded entry, and continues live from the
first missing entry. No accepted output is ever regenerated.

All I/O is injected: a provider adapter, a journal, an object store, and a
clock. Nothing here performs network, database, or filesystem access.
"""

from __future__ import annotations

import json
import time
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from gateway.tasks.icp_generator import (
    COMPANY_STAGES,
    GEOGRAPHIES,
    INDUSTRY_DISTRIBUTION,
    INTENT_SIGNALS,
    INTERNATIONAL_GEOGRAPHIES,
    OPENROUTER_MODEL,
    STAGE_EMPLOYEE_BUCKETS,
    SUB_INDUSTRIES,
    apply_generated_icp_contract,
    canonicalize_generated_icp,
    international_icp_target,
)
from leadpoet_canonical.attested_v2 import canonical_json


def normalize_employee_count_bucket(*args, **kwargs):
    from research_lab.employee_buckets import normalize_employee_count_bucket as normalize  # lazy: the research_lab package is heavy

    return normalize(*args, **kwargs)

from lab_arena import contracts
from lab_arena.contracts import ArenaContractError

# ---------------------------------------------------------------------------
# Generator identity (section 5.1: model and settings identities)
# ---------------------------------------------------------------------------

# The Lab's OpenRouter generation coroutine sends exactly these body values;
# the prompt-parity test asserts them against the captured Lab request.
GENERATOR_MODEL = OPENROUTER_MODEL
GENERATOR_SETTINGS: Mapping[str, Any] = {
    "model": OPENROUTER_MODEL,
    "temperature": 0.7,
    "max_tokens": 16000,
}
EXCLUSION_SETTINGS: Mapping[str, Any] = {
    "model": OPENROUTER_MODEL,
    "temperature": 0.0,
}
GENERATION_TIMEOUT_SECONDS = 180.0
EXCLUSION_TIMEOUT_SECONDS = 45.0

# Explicit canonicalization arguments: no environment value can change the
# Arena's employee-bucket expansion (section 8 step 3).
EMPLOYEE_BUCKET_RADIUS = 2
ALL_EMPLOYEE_BUCKETS = False

DEFAULT_EXCLUSION_COUNT = 1
MAX_EXCLUSION_COUNT = 3

# The Lab's ordered industry list is the slot vocabulary.
INDUSTRIES: Tuple[str, ...] = tuple(INDUSTRY_DISTRIBUTION)
SUPPORTED_GEOGRAPHIES: Tuple[str, ...] = tuple(dict.fromkeys([*GEOGRAPHIES, *INTERNATIONAL_GEOGRAPHIES]))
SUPPORTED_STAGES: Tuple[str, ...] = tuple(COMPANY_STAGES)

# Published rejection rule identifiers (section 8 step 3 and step 4).
REJECTION_RULES = (
    "response.unparsable",
    "schema.not_an_object",
    "schema.document_limits",
    "schema.prompt_missing",
    "schema.industry_mismatch",
    "slot.filled",
    "schema.geography_unsupported",
    "schema.employee_bucket_invalid",
    "schema.stage_invalid",
    "schema.intent_missing",
    "schema.example_company_missing",
    "duplicate.content_hash",
    "duplicate.intent_signature",
    "exclusion.failed",
)

# Fields removed before the content hash: slot identity is bound separately
# in the ordered leaf, and no timestamp or transport field is content.
CONTENT_HASH_EXCLUDED_FIELDS = (
    "icp_id",
    "round_id",
    "batch_id",
    "slot",
    "set_id",
    "generated_at",
    "created_at",
    "updated_at",
    "timestamp",
    "request_id",
    "response_id",
    "response_ref",
    "provider",
    "model",
    "usage",
)

GENERATION_OBJECT_KIND = "generation"
EXCLUSION_OBJECT_KIND = "exclusion"

_JOURNAL_ENVELOPE_FIELDS = ("schema_version", "sequence", "timestamp", "prev_hash", "entry_hash")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_BLOCK_HASH_RE = re.compile(r"^0x[0-9a-f]{64}$")
_OBJECT_REF_RE = re.compile(r"^arena/[^/]+/(generation|exclusion)/[0-9]+\.json$")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BenchmarkError(ArenaContractError):
    """Base class for benchmark generation failures. Always fail closed."""


class BenchmarkGenerationFailed(BenchmarkError):
    """The signed attempt limit was reached with slots still missing.

    The service cancels the round (section 17). The journal already holds
    every attempt, so the failure is auditable without this exception.
    """

    def __init__(self, *, missing_slots: Sequence[int], attempts_used: int, journal_length: int) -> None:
        self.missing_slots = tuple(int(slot) for slot in missing_slots)
        self.attempts_used = int(attempts_used)
        self.journal_length = int(journal_length)
        super().__init__(
            "benchmark generation exhausted %d attempts with %d slots missing"
            % (self.attempts_used, len(self.missing_slots))
        )


class BenchmarkReplayError(BenchmarkError):
    """The recorded journal or a stored object disagrees with re-execution."""


class GenerationParseError(BenchmarkError):
    """A generation response carried no parseable ICP list."""


class ExclusionParseError(BenchmarkError):
    """An exclusion response carried no parseable JSON array."""


class ProviderFailure(Exception):
    """Raised by a provider adapter for a transport failure, timeout, or
    non-200 status. Adapters must never include credential material in the
    message. The flow journals the request as ``unknown`` and retries."""


# ---------------------------------------------------------------------------
# Injected interfaces
# ---------------------------------------------------------------------------


class GenerationProvider(Protocol):
    """Chat-completion adapter bound to Arena credentials by the service.

    ``chat`` returns the raw OpenRouter response object (the parsed JSON body
    of an HTTP 200) and raises :class:`ProviderFailure` for anything else.
    ``max_tokens`` is ``None`` when the request carries no token limit.
    """

    def chat(
        self,
        *,
        messages: Sequence[Mapping[str, str]],
        temperature: float,
        max_tokens: Optional[int],
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


class GenerationJournal(Protocol):
    """Hash-chained journal stored on the round row (section 8).

    ``append`` receives an entry without ``entry_hash``, finalizes it with
    :func:`lab_arena.contracts.finalize_journal_entry`, persists it durably,
    and returns the finalized entry. ``entries`` returns every persisted entry
    in sequence order.
    """

    def entries(self) -> Sequence[Mapping[str, Any]]: ...

    def append(self, entry: Mapping[str, Any]) -> Mapping[str, Any]: ...


class ObjectStore(Protocol):
    """Private Arena bucket: versioned, delete-denied, write-once per ref."""

    def put(self, ref: str, data: bytes) -> None: ...

    def get(self, ref: str) -> bytes: ...


class MemoryJournal:
    """In-memory :class:`GenerationJournal` for tests and dry runs."""

    def __init__(self, entries: Sequence[Mapping[str, Any]] = ()) -> None:
        self._entries: List[Dict[str, Any]] = [dict(entry) for entry in entries]
        contracts.verify_journal_chain(self._entries)

    def entries(self) -> Tuple[Dict[str, Any], ...]:
        return tuple(dict(entry) for entry in self._entries)

    def append(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        finalized = contracts.finalize_journal_entry(entry)
        expected_sequence = len(self._entries)
        expected_prev = self._entries[-1]["entry_hash"] if self._entries else ""
        if finalized["sequence"] != expected_sequence:
            raise ArenaContractError("journal append sequence %d is not %d" % (finalized["sequence"], expected_sequence))
        if finalized["prev_hash"] != expected_prev:
            raise ArenaContractError("journal append does not chain from the current head")
        self._entries.append(finalized)
        return dict(finalized)


class MemoryObjectStore:
    """In-memory :class:`ObjectStore`; a ref never changes once written."""

    def __init__(self) -> None:
        self._objects: Dict[str, bytes] = {}

    def put(self, ref: str, data: bytes) -> None:
        if not isinstance(data, (bytes, bytearray)):
            raise ArenaContractError("object store data must be bytes")
        existing = self._objects.get(ref)
        if existing is not None and existing != bytes(data):
            raise ArenaContractError("object store ref %s already holds different bytes" % ref)
        self._objects[ref] = bytes(data)

    def get(self, ref: str) -> bytes:
        if ref not in self._objects:
            raise BenchmarkReplayError("object store has no object at %s" % ref)
        return self._objects[ref]

    def refs(self) -> Tuple[str, ...]:
        return tuple(self._objects)


def object_ref(round_id: str, kind: str, sequence: int) -> str:
    """``arena/<round_id>/<kind>/<sequence>.json`` for a journaled object."""

    if kind not in (GENERATION_OBJECT_KIND, EXCLUSION_OBJECT_KIND):
        raise ArenaContractError("unknown object kind %s" % kind)
    return "arena/%s/%s/%d.json" % (round_id, kind, int(sequence))


# ---------------------------------------------------------------------------
# Slot plan (section 8 step 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BatchPlan:
    """One fixed generation request: ordered slots and their industries."""

    batch_id: str
    slots: Tuple[int, ...]
    industries: Tuple[str, ...]


def _build_slot_plan() -> Tuple[BatchPlan, ...]:
    """Slots are 0..29 in order. Batch ``i`` covers the next
    ``GENERATION_BATCH_SIZES[i]`` slots and industry ``k`` of a batch is
    ``INDUSTRIES[k]``; so slot ``s`` maps to ``INDUSTRIES[s]`` for the
    twenty-slot batch and to ``INDUSTRIES[s - 20]`` for the ten-slot batch.
    The thirty slots therefore hold two ICPs for each of the first ten
    industries and one for each of the remaining ten."""

    plans: List[BatchPlan] = []
    offset = 0
    for index, size in enumerate(contracts.GENERATION_BATCH_SIZES):
        if size < 1 or size > len(INDUSTRIES):
            raise ArenaContractError("generation batch size %d exceeds the industry list" % size)
        slots = tuple(range(offset, offset + size))
        plans.append(BatchPlan("b%d" % (index + 1), slots, tuple(INDUSTRIES[:size])))
        offset += size
    if offset != contracts.BENCHMARK_ICP_COUNT:
        raise ArenaContractError("generation batch sizes do not cover the benchmark")
    return tuple(plans)


SLOT_PLAN: Tuple[BatchPlan, ...] = _build_slot_plan()
FIXED_BATCH_IDS: Tuple[str, ...] = tuple(plan.batch_id for plan in SLOT_PLAN)


def _require_slot(slot: int) -> int:
    if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < contracts.BENCHMARK_ICP_COUNT:
        raise ArenaContractError("slot must be an integer in 0..%d" % (contracts.BENCHMARK_ICP_COUNT - 1))
    return slot


def slot_batch(slot: int) -> str:
    """Batch id of the fixed request that first covers ``slot``."""

    _require_slot(slot)
    for plan in SLOT_PLAN:
        if slot in plan.slots:
            return plan.batch_id
    raise ArenaContractError("slot %d is not in the slot plan" % slot)


def slot_industry(slot: int) -> str:
    """Expected industry of ``slot`` under the fixed slot plan."""

    _require_slot(slot)
    for plan in SLOT_PLAN:
        if slot in plan.slots:
            return plan.industries[plan.slots.index(slot)]
    raise ArenaContractError("slot %d is not in the slot plan" % slot)


def stage_positions(stage: int) -> range:
    """The one stage covers every slot, 0..29 (section 8 step 7 as revised)."""

    if stage == 1:
        return range(0, contracts.BENCHMARK_ICP_COUNT)
    raise ArenaContractError("stage must be 1")


# ---------------------------------------------------------------------------
# Prompt copy (section 8; parity with the Lab for count=20 is tested)
# ---------------------------------------------------------------------------

# The Lab's system prompt, copied verbatim, with only the count and the
# industry list parameterized through the ``@@...@@`` markers. The literal
# ``{{``, ``}}`` and ``{set_id}`` artifacts and the ``{international_target}``,
# ``{total_icps}`` and ``{domestic_count}`` placeholders (substituted below
# through the Lab's own three ``.replace()`` calls) are preserved on purpose.
_SYSTEM_PROMPT_TEMPLATE = """You are generating B2B sales-targeting ICPs (Ideal Customer Profiles) for a benchmark. You have real-time web access — USE IT.

YOUR JOB
Generate exactly @@COUNT@@ ICPs, one per industry from the distribution list. Each ICP must describe a real, currently-existing target market that a salesperson could actually go prospect.

THE ONE RULE THAT MATTERS MOST — REALISM
Before outputting any ICP, mentally verify: "Can I name at least ONE real, currently-operating company that satisfies ALL the criteria of this ICP — with verifiable recent activity matching the intent signal?"

If you cannot name a specific real company that fits, the ICP is INVALID. Broaden one of the constraints (geography, stage, employee band, sub-industry) until you CAN name at least one real company. Do not output an unrealistic combination under any circumstances.

To enforce this, every ICP MUST include a `verified_example_company` field naming the real company you found while verifying. This is not optional. If you cannot fill this field with a real company, you must rewrite the ICP with broader constraints until you can.

DO NOT generate ICPs where:
- The industry × stage × geography intersection has zero real companies you can name
- The intent signal doesn't fit the industry shape (Consulting firms don't "launch products"; Industrial Manufacturers don't have SaaS-style product launches; Real Estate firms do deals, not product launches)
- The stage doesn't match the intent (Seed companies don't acquire other companies; Series A startups don't make big-name strategic partnerships)
- The geography is so narrow that no real candidates exist
- The product/service is so specific that the buyer's universe collapses (use broad categories, not single named tools)

PROMPT VOICE
Each ICP's `prompt` field should sound like a different real salesperson typed it. Mix tones across the @@COUNT@@ prompts:
- Direct first-person ("I need", "I'm looking for")
- Casual ("yo can you pull", "hey, gonna need")
- Shorthand / telegraphic
- Question format ("what AI companies in...", "anyone tracking...")
- Descriptive / detailed ("Searching for...")

Never use job titles, seniority levels, "decision-makers", "executives", or any contact-level descriptor. Company-only.

CONSTRAINT LISTS (use ONLY these values)

ALLOWED INDUSTRIES (exactly one ICP per industry, in this order):
@@INDUSTRIES@@

INTENT SIGNAL — WRITE IT LIKE A REAL SALES-INTELLIGENCE BRIEF (not a generic label):
Pick 1-2 intents per ICP. Do NOT output a bare category like "Launched a new product".
Write each intent as a specific, descriptive sentence in the style a real buyer would
brief a sourcing team — name the kind of behavior and the kind of evidence that proves
it. Match the intent to the industry.

The intent MUST be based on ONE of these underlying event types (this keeps it verifiable):
funding round, new product / major capability launch, expansion into a new market,
acquisition, leadership change, regulatory clearance or certification, strategic
partnership, hiring for specific roles, or a new facility / office / store opening.

Fulfillment-style examples (write yours the same way — specific wording, still broad enough
that MANY real companies match):
- "Launched a new product or major platform capability in the last 12 months, per a press
   release, product page, or changelog"
- "Announced a Series A or later funding round in the last 12 months, per a press release,
   Crunchbase, or news coverage"
- "Actively hiring for platform, integration, or revenue-operations roles, per current job
   postings or careers page"
- "Expanded into a new country or region in the last year, per a press release or company
   announcement"

CRITICAL — SPECIFIC WORDING, NOT A NARROW REQUIREMENT: the intent must be worded
specifically but stay broad enough that many real companies verifiably match. Do NOT
narrow it to a single tool, a single named event, or a niche so tight that fewer than a
handful of companies qualify. Specificity belongs in the phrasing, not in shrinking the
candidate pool.

ALLOWED COMPANY STAGES: Seed, Series A, Series B, Series C+, Private Equity, Public

STAGE DISTRIBUTION — SPREAD EVENLY ACROSS STAGES:
The benchmark needs to test miner performance at ALL stages, not just late-stage companies. Skewing toward Series C+/Public (because those have the most PR coverage) makes the benchmark too easy and fails to test the harder verification cases.

Target distribution across the @@COUNT@@ ICPs (approximate, ±2 per bucket is fine):
@@STAGE_DISTRIBUTION@@

Do NOT cluster on later stages just because they're easier to verify. The realism rule still applies (every ICP must have a real `verified_example_company`), but Seed and Series A startups exist with verifiable funding announcements — find them.

ALLOWED EMPLOYEE BANDS — USE THESE EXACT LINKEDIN BUCKETS ONLY:
11-50, 51-200, 201-500, 501-1,000, 1,001-5,000, 5,001-10,000, 10,001+
- `employee_count` must be a JSON array of the allowed LinkedIn buckets.
- Prefer 3-5 contiguous buckets around the most realistic target size.
- Do not use fake broad ranges like "51-5000".

ALLOWED GEOGRAPHIES — STRONGLY PREFER BROAD VALUES:
EXACTLY {international_target} of the {total_icps} ICPs MUST use an international geography from this list (pick industries where that market genuinely thrives, e.g. FinTech in London, Mining tech in Australia, Payments in Singapore):
- "United Kingdom" / "United Kingdom, London"
- "Ireland"
- "Canada" / "Canada, Toronto"
- "Australia" / "Australia, Sydney"
- "New Zealand"
- "Singapore"
- "United Arab Emirates, Dubai" / "United Arab Emirates, Abu Dhabi"
These are English-speaking business markets only; never use any other country and never use a bare region name like "Europe" or "APAC". For international ICPs, `country` must be the country portion of the geography (e.g. "United Kingdom", "Canada").

The remaining {domestic_count} ICPs are United States:
- "United States" (whole country — use this for ~50% of the US ICPs)
- "United States, West Coast" / "Northeast" / "Midwest" / "South" / "Southwest" (~40%)
- "United States, <State>" only when the industry has a known concentration there (~10%)

INDUSTRY × INTENT PAIRING (Sonar should naturally honor these):
- Service industries (Consulting, Legal, Accounting, Recruiting) → leadership change, hiring, expansion, acquisition, partnership — NOT product launch
- Real Estate / Commercial Real Estate / PropTech → acquisition, expansion, leadership change, funding, facility opening — NOT product launch
- Industrial Manufacturing → acquisition, expansion, partnership, facility opening, hiring — NOT SaaS-style product launches
- Tech / SaaS / AI / Hardware / Biotech / Payments / FinTech / Cyber / Health → all intents work
- Banking / large traditional finance → leadership change, acquisition, regulatory clearance, partnership

STAGE × INTENT PAIRING:
- "Acquired another company" → REQUIRES Series B or later
- "Announced a strategic partnership" → REQUIRES Series B or later
- "Recent factory / facility / store opening" → REQUIRES Series A or later
- All other intent × stage combinations are valid

OUTPUT — JSON ONLY, NO PROSE, NO MARKDOWN

{{
  "icps": [
    {{
      "icp_id": "icp_{set_id}_001",
      "prompt": "<one-sentence salesperson-voice description>",
      "industry": "<from industry list, in order>",
      "sub_industry": "<natural sub-industry>",
      "geography": "<from allowed geographies, prefer broad>",
      "country": "<the country portion of the geography, e.g. United States, United Kingdom, Canada>",
      "employee_count": ["<3-5 contiguous allowed LinkedIn buckets>"],
      "company_stage": "<from allowed stages>",
      "product_service": "<a specific, descriptive value proposition — what the company actually sells and the job it does for its buyer, e.g. 'A subscription B2B platform that revenue and operations teams use to manage pipeline, deals, and customer workflows'. NOT a bare category like 'B2B software'. Do NOT narrow to a single named tool.>",
      "required_attribute": "<a specific, descriptive attribute the company must have, written like the product_service — e.g. 'Sells a subscription software platform used by revenue or operations teams to manage pipeline, deals, or customer workflows'. Specific WORDING, but broad enough that many real companies match. NOT the bare template 'offers or provides X'.>",
      "intent_signal": "<a specific, descriptive intent sentence in fulfillment style — see the INTENT SIGNAL section above>",
      "intent_category": "<FUNDING | ACQUISITION | PARTNERSHIP | PRODUCT_LAUNCH | LEADERSHIP_CHANGE | MARKET_EXPANSION | REGULATORY_CLEARANCE | FACILITY_OPENING | HIRING>",
      "intent_max_age_days": 365,
      "intent_signals": ["<required intent first, specific fulfillment-style sentence>", "<optional bonus intent second, specific fulfillment-style sentence>"],
      "bonus_intents": [
        {{"intent_signal": "<optional bonus intent>", "intent_category": "<matching category>", "intent_max_age_days": 365}}
      ],
      "verified_example_company": "<MANDATORY: the real company name you found while verifying this ICP>"
    }}
  ]
}}

FINAL CHECK before output (for every ICP):
1. Is `verified_example_company` a real, currently-operating company? If not, REWRITE with broader constraints.
2. Does the named example company actually match ALL the ICP's stated criteria — including the specific `required_attribute` and `intent_signal`? If not, REWRITE.
3. SUPPLY CHECK: can you name at least THREE more real, currently-operating companies (four total) that verifiably match this ICP's specific `required_attribute` and `intent_signal`? If you cannot, the attribute or intent is too narrow — keep the specific WORDING but broaden the requirement (or broaden geography/stage/employee band) until many real companies match. Specific phrasing, broad candidate pool.
4. Are the `product_service` and `required_attribute` specific, descriptive value propositions (like a real sales brief) rather than bare categories or the "offers or provides X" template?
5. Is each `intent_signal` a specific fulfillment-style sentence naming the behavior and the kind of evidence, rather than a bare label?
6. Is the geography broad enough that real candidates exist?
7. Are there exactly @@COUNT@@ ICPs, one per industry in the listed order?
8. Are EXACTLY {international_target} ICPs international (non-US, from the allowed international list) with `country` matching their geography?
9. No job titles, no seniority, no contact-level descriptors in the prompts?"""

# Stage distribution the Lab prompt states for twenty ICPs: (stage, low, high).
_STAGE_DISTRIBUTION_20: Tuple[Tuple[str, int, int], ...] = (
    ("Seed", 2, 3),
    ("Series A", 4, 5),
    ("Series B", 4, 5),
    ("Series C+", 3, 4),
    ("Private Equity", 1, 2),
    ("Public", 2, 3),
)
_LAB_PROMPT_COUNT = 20

_USER_PROMPT_FIXED = (
    "Generate {count} ICPs for set_id={set_id}. Follow every instruction in the "
    "system message exactly. Output JSON only, no commentary."
)
_USER_PROMPT_REPLACEMENT = (
    "Generate {count} ICPs for set_id={set_id}, one for each of these industries in "
    "this order: {industries}. Follow every instruction in the system message "
    "exactly. Output JSON only, no commentary."
)


def stage_distribution_lines(count: int) -> Tuple[str, ...]:
    """The stage-distribution bullet lines for ``count`` ICPs.

    For twenty ICPs the lines are the Lab's exact text. Other counts scale
    each range proportionally: the low bound is ``floor(low * count / 20)``
    and the high bound is ``ceil(high * count / 20)``, so the ranges keep
    covering the whole count and never drop a stage to an impossible band.
    """

    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ArenaContractError("count must be a positive integer")
    lines = []
    for stage, low, high in _STAGE_DISTRIBUTION_20:
        scaled_low = (low * count) // _LAB_PROMPT_COUNT
        scaled_high = -((-high * count) // _LAB_PROMPT_COUNT)
        lines.append("- %s: %d-%d ICPs" % (stage, scaled_low, scaled_high))
    return tuple(lines)


def _validate_prompt_inputs(count: int, industries: Sequence[str], set_id: int) -> Tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count > contracts.BENCHMARK_ICP_COUNT:
        raise ArenaContractError("count must be an integer in 1..%d" % contracts.BENCHMARK_ICP_COUNT)
    ordered = tuple(industries)
    if len(ordered) != count:
        raise ArenaContractError("industries must list exactly one entry per requested ICP")
    for industry in ordered:
        if industry not in INDUSTRIES:
            raise ArenaContractError("industry %r is not in the Lab industry list" % (industry,))
    if isinstance(set_id, bool) or not isinstance(set_id, int) or set_id < 1:
        raise ArenaContractError("set_id must be a positive integer")
    return ordered


def build_generation_prompts(
    *,
    count: int,
    industries: Sequence[str],
    set_id: int,
    replacement: bool = False,
) -> Tuple[str, str]:
    """Return ``(system_prompt, user_prompt)`` for one generation request.

    For ``count=20`` and ``industries=list(INDUSTRY_DISTRIBUTION)`` with
    ``replacement=False`` both strings are byte-identical to the prompts the
    Lab sends for its daily set. A replacement request names the missing
    slots' industries in order in the user prompt (section 8 step 5).
    """

    ordered = _validate_prompt_inputs(count, industries, set_id)
    system_prompt = (
        _SYSTEM_PROMPT_TEMPLATE.replace("@@COUNT@@", str(count))
        .replace("@@INDUSTRIES@@", ", ".join(ordered))
        .replace("@@STAGE_DISTRIBUTION@@", "\n".join(stage_distribution_lines(count)))
    )
    international_target = international_icp_target(count)
    system_prompt = (
        system_prompt
        .replace("{international_target}", str(international_target))
        .replace("{total_icps}", str(count))
        .replace("{domestic_count}", str(count - international_target))
    )
    if "@@" in system_prompt:
        raise ArenaContractError("system prompt template has an unsubstituted marker")
    if replacement:
        user_prompt = _USER_PROMPT_REPLACEMENT.format(count=count, set_id=set_id, industries=", ".join(ordered))
    else:
        user_prompt = _USER_PROMPT_FIXED.format(count=count, set_id=set_id)
    return system_prompt, user_prompt


def build_generation_request(
    *,
    count: int,
    industries: Sequence[str],
    set_id: int,
    replacement: bool = False,
) -> Dict[str, Any]:
    """The exact OpenRouter request body for one generation request."""

    system_prompt, user_prompt = build_generation_prompts(
        count=count, industries=industries, set_id=set_id, replacement=replacement
    )
    return {
        "model": GENERATOR_SETTINGS["model"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": GENERATOR_SETTINGS["temperature"],
        "max_tokens": GENERATOR_SETTINGS["max_tokens"],
    }


# ---------------------------------------------------------------------------
# Exclusion prompt and parser (copied from the Lab generator's exclusion
# helper in gateway/tasks/icp_generator.py; fail-closed instead of fail-open)
# ---------------------------------------------------------------------------

_EXCLUSION_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}$")


def _require_exclusion_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_EXCLUSION_COUNT:
        raise ArenaContractError("exclusion count must be an integer in 1..%d" % MAX_EXCLUSION_COUNT)
    return count


def build_exclusion_prompt(icp: Mapping[str, Any], count: int) -> str:
    """The Lab's exclusion prompt for ``icp``, requesting ``count`` companies."""

    want = _require_exclusion_count(count)
    bands = icp.get("employee_count")
    bands_text = ", ".join(bands) if isinstance(bands, (list, tuple)) else str(bands or "")
    plural = "y" if want == 1 else "ies"
    stage = str(icp.get("company_stage") or "").strip()
    prompt = (
        f"Name {want} real, currently-operating compan{plural} that match "
        f"ALL of this profile:\n"
        f"- Industry: {icp.get('industry', '')} ({icp.get('sub_industry', '')})\n"
        f"- Headquarters: {icp.get('geography', icp.get('country', ''))}\n"
        f"- Employee count in one of these LinkedIn bands: {bands_text}\n"
        + (f"- Funding/ownership stage: {stage}\n" if stage else "")
        + "Only name companies you can verify exist on the web right now. "
          'Return STRICT JSON only: [{"name": "<company name>", "domain": "<bare homepage domain like example.com>"}]'
    )
    return prompt


def build_exclusion_request(icp: Mapping[str, Any], count: int) -> Dict[str, Any]:
    """The exact OpenRouter request body for one exclusion request."""

    return {
        "model": EXCLUSION_SETTINGS["model"],
        "messages": [{"role": "user", "content": build_exclusion_prompt(icp, count)}],
        "temperature": EXCLUSION_SETTINGS["temperature"],
    }


def parse_exclusion_response(content: Any, count: int) -> List[str]:
    """Flat exclusion entries (domain preferred, name fallback) from the
    model content. Raises :class:`ExclusionParseError` when the content holds
    no JSON array; an empty result is returned as an empty list and the
    caller treats it as a failure."""

    want = _require_exclusion_count(count)
    if not isinstance(content, str):
        raise ExclusionParseError("exclusion content is not a string")
    match = re.search(r"\[.*\]", content, re.S)
    if not match:
        raise ExclusionParseError("exclusion content holds no JSON array")
    try:
        rows = json.loads(match.group(0), parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise ExclusionParseError("exclusion content is not valid JSON") from exc
    if not isinstance(rows, list):
        raise ExclusionParseError("exclusion content is not a JSON array")
    entries: List[str] = []
    for row in rows[:want]:
        if not isinstance(row, dict):
            continue
        domain = str(row.get("domain") or "").strip().lower()
        domain = re.sub(r"^https?://", "", domain).split("/")[0]
        if domain.startswith("www."):
            domain = domain[4:]
        name = " ".join(str(row.get("name") or "").split())[:120]
        if domain and _EXCLUSION_DOMAIN_RE.match(domain):
            entries.append(domain)
        elif name:
            entries.append(name)
    return list(dict.fromkeys(entries))[:want]


# ---------------------------------------------------------------------------
# Response parsing (the Lab's content handling, fail-closed)
# ---------------------------------------------------------------------------


def _reject_json_constant(value: str) -> Any:
    raise ValueError("non-finite JSON constant %s" % value)


def response_content(response: Any) -> str:
    """``choices[0].message.content`` of a chat completion, else ``""``."""

    if not isinstance(response, Mapping):
        return ""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        return ""
    message = choices[0].get("message")
    if not isinstance(message, Mapping):
        return ""
    content = message.get("content")
    return content if isinstance(content, str) else ""


def parse_generation_content(content: str) -> List[Any]:
    """The Lab's fence stripping, brace extraction, and wrapper-key handling.

    Returns the raw ICP list in returned order. Raises
    :class:`GenerationParseError` when there is no list to process.
    """

    if not isinstance(content, str) or not content:
        raise GenerationParseError("generation response has empty content")
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped[3:]
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
        stripped = stripped.strip()
    if not stripped.startswith("{"):
        obrace = stripped.find("{")
        cbrace = stripped.rfind("}")
        if obrace >= 0 and cbrace > obrace:
            stripped = stripped[obrace : cbrace + 1]
    try:
        parsed = json.loads(stripped, parse_constant=_reject_json_constant)
    except ValueError as exc:
        raise GenerationParseError("generation response is not valid JSON") from exc
    if isinstance(parsed, dict):
        icps = parsed.get("icps") or parsed.get("icp_prompts") or parsed.get("prompts") or parsed.get("data")
        if icps is None and len(parsed) == 1:
            icps = list(parsed.values())[0]
        if icps is None:
            icps = list(parsed.values())[0] if parsed else []
    else:
        icps = parsed
    if not isinstance(icps, list):
        raise GenerationParseError("generation response does not hold a list of ICPs")
    return icps


# ---------------------------------------------------------------------------
# Published schema check on the raw generator output (section 8 step 3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RawIcpCheck:
    """Outcome of the schema check for one raw ICP against one request."""

    slot: Optional[int]
    rejection_rule: Optional[str]
    industry: Optional[str] = None
    geography: Optional[str] = None
    country: Optional[str] = None
    intent_signals: Tuple[str, ...] = ()
    verified_example_company: str = ""


def _fold(value: Any) -> str:
    return " ".join(str(value).strip().split())


def _match_industry(value: Any) -> Optional[str]:
    """Canonical industry name for a case-insensitive, whitespace-stripped
    match, mirroring the Lab's normalization; ``None`` when unknown."""

    if not isinstance(value, str):
        return None
    lowered = _fold(value).lower()
    for industry in INDUSTRIES:
        if lowered == industry.lower():
            return industry
    return None


def _match_geography(value: Any) -> Optional[str]:
    """Canonical supported geography for ``value``; ``None`` when unsupported."""

    if not isinstance(value, str):
        return None
    lowered = _fold(value).lower()
    for geography in SUPPORTED_GEOGRAPHIES:
        if lowered == geography.lower():
            return geography
    return None


def _country_for_geography(geography: str) -> str:
    """The Lab's country derivation: whole-US for any United States value,
    otherwise the international entry's first comma segment."""

    if "United States" in geography:
        return "United States"
    first = geography.split(",")[0].strip().lower()
    for candidate in INTERNATIONAL_GEOGRAPHIES:
        if first == candidate.split(",")[0].strip().lower():
            return candidate.split(",")[0].strip()
    raise ArenaContractError("geography %r has no supported country" % geography)


def _normalize_intent_signals(value: Any) -> List[str]:
    """The Lab's intent-signal normalization (stripped, folded, at most 5)."""

    if isinstance(value, str):
        signals = [value]
    elif isinstance(value, list):
        signals = [str(item) for item in value]
    else:
        signals = []
    cleaned = [" ".join(signal.strip().split()) for signal in signals if signal.strip()]
    return cleaned[:5]


def _employee_buckets_valid(value: Any) -> bool:
    if isinstance(value, str):
        return bool(normalize_employee_count_bucket(value, default=None))
    if isinstance(value, (list, tuple)):
        if not value:
            return False
        return all(isinstance(item, str) and normalize_employee_count_bucket(item, default=None) for item in value)
    return False


def check_raw_icp(
    raw: Any,
    *,
    requested: Sequence[Tuple[int, str]],
    filled: Collection[int],
) -> RawIcpCheck:
    """Run the published schema check on one raw generator output.

    ``requested`` lists the request's ``(slot, industry)`` pairs in order and
    ``filled`` the slots already accepted while processing this response. The
    ICP is bound to the first open requested slot whose industry it names.
    Checks run in the published order and the first failure names the rule;
    the check never canonicalizes, so no default or random back-fill can hide
    a defect.
    """

    if not isinstance(raw, Mapping):
        return RawIcpCheck(None, "schema.not_an_object")
    try:
        contracts.check_strict_document(raw)
    except ArenaContractError:
        return RawIcpCheck(None, "schema.document_limits")

    industry = _match_industry(raw.get("industry"))
    slot: Optional[int] = None
    industry_rule: Optional[str] = None
    if industry is None:
        industry_rule = "schema.industry_mismatch"
    elif industry not in {expected for _, expected in requested}:
        industry_rule = "schema.industry_mismatch"
    else:
        slot = next((s for s, expected in requested if expected == industry and s not in filled), None)
        if slot is None:
            industry_rule = "slot.filled"

    prompt = raw.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return RawIcpCheck(slot, "schema.prompt_missing", industry=industry)
    if industry_rule is not None:
        return RawIcpCheck(slot, industry_rule, industry=industry)
    geography = _match_geography(raw.get("geography"))
    if geography is None:
        return RawIcpCheck(slot, "schema.geography_unsupported", industry=industry)
    country = _country_for_geography(geography)
    if not _employee_buckets_valid(raw.get("employee_count")):
        return RawIcpCheck(slot, "schema.employee_bucket_invalid", industry=industry, geography=geography, country=country)
    stage = raw.get("company_stage")
    if not isinstance(stage, str) or _fold(stage) not in SUPPORTED_STAGES:
        return RawIcpCheck(slot, "schema.stage_invalid", industry=industry, geography=geography, country=country)
    intent_signals = _normalize_intent_signals(raw.get("intent_signals"))
    if not intent_signals and raw.get("intent_signal"):
        intent_signals = _normalize_intent_signals(raw.get("intent_signal"))
    if not intent_signals:
        return RawIcpCheck(slot, "schema.intent_missing", industry=industry, geography=geography, country=country)
    example = raw.get("verified_example_company")
    example = example.strip() if isinstance(example, str) else ""
    if not example:
        return RawIcpCheck(
            slot,
            "schema.example_company_missing",
            industry=industry,
            geography=geography,
            country=country,
            intent_signals=tuple(intent_signals),
        )
    return RawIcpCheck(
        slot,
        None,
        industry=industry,
        geography=geography,
        country=country,
        intent_signals=tuple(intent_signals),
        verified_example_company=example,
    )


# ---------------------------------------------------------------------------
# ICP construction, canonicalization, hashing
# ---------------------------------------------------------------------------


def build_validated_icp(raw: Mapping[str, Any], check: RawIcpCheck, *, icp_id: str) -> Dict[str, Any]:
    """The Lab's ``validated_icp`` dictionary (same keys and defaults) for one
    raw ICP that passed :func:`check_raw_icp`, re-keyed to ``icp_id``."""

    if check.rejection_rule is not None or check.industry is None or check.geography is None:
        raise ArenaContractError("cannot build an ICP from a rejected schema check")
    industry_normalized = check.industry
    prompt = raw["prompt"]
    intent_signals = list(check.intent_signals)
    sub_industry = raw.get("sub_industry", SUB_INDUSTRIES.get(industry_normalized, ["General"])[0])
    return {
        "icp_id": icp_id,
        "prompt": prompt,
        "industry": industry_normalized,
        "sub_industry": sub_industry,
        "target_roles": [],
        "target_seniority": "",
        "employee_count": raw.get("employee_count", "51-200"),
        "company_stage": _fold(raw["company_stage"]),
        "geography": check.geography,
        "country": check.country,
        "product_service": raw.get("product_service", "Software solution"),
        "intent_signals": intent_signals,
        "intent_signal": raw.get("intent_signal", intent_signals[0]),
        "intent_category": raw.get("intent_category", ""),
        "intent_max_age_days": raw.get("intent_max_age_days"),
        "bonus_intents": raw.get("bonus_intents", []),
        "required_attribute": raw.get("required_attribute", ""),
        "buyer_description": prompt,
        "verified_example_company": check.verified_example_company,
    }


def canonicalize_arena_icp(validated: Mapping[str, Any]) -> Dict[str, Any]:
    """The Lab canonicalizer with explicit employee-bucket arguments, so no
    environment value changes the output. The schema check has already
    guaranteed a nonempty intent list, so the canonicalizer's random
    back-fill branch is unreachable."""

    industry = str(validated.get("industry") or "")
    sub_industry = str(validated.get("sub_industry") or "")
    if not validated.get("intent_signals"):
        raise ArenaContractError("canonicalization requires at least one intent signal")
    return canonicalize_generated_icp(
        dict(validated),
        industry=industry,
        sub_industry=sub_industry,
        employee_bucket_radius=EMPLOYEE_BUCKET_RADIUS,
        all_employee_buckets=ALL_EMPLOYEE_BUCKETS,
    )


def content_hash(icp: Mapping[str, Any]) -> str:
    """Duplicate-detection hash over the ICP content, with slot identity,
    round, batch, timestamps and transport fields removed."""

    body = {key: value for key, value in icp.items() if key not in CONTENT_HASH_EXCLUDED_FIELDS}
    return contracts.document_hash(body)


def intent_signature(icp: Mapping[str, Any]) -> str:
    """Sorted, lowercased, whitespace-folded intent signals joined by ``|``."""

    signals = icp.get("intent_signals") or []
    if isinstance(signals, str):
        signals = [signals]
    normalized = sorted(
        {" ".join(str(signal).strip().lower().split()) for signal in signals if str(signal).strip()}
    )
    return "|".join(normalized)


def apply_arena_contract(canonical: Mapping[str, Any]) -> Dict[str, Any]:
    """The Lab contract helper (``max_companies`` and stage-widened
    ``employee_count``) applied to a copy of the canonical ICP."""

    icp = dict(canonical)
    apply_generated_icp_contract([icp])
    return icp


def icp_hash(icp: Mapping[str, Any]) -> str:
    """Hash of the final accepted ICP, exclusions included."""

    return contracts.document_hash(icp)


# ---------------------------------------------------------------------------
# Results and commitment (section 5.2, section 8 steps 6 and 7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AcceptedSlot:
    slot: int
    batch_id: str
    attempt: int
    content_hash: str
    icp_hash: str
    icp: Mapping[str, Any]


@dataclass(frozen=True)
class BenchmarkResult:
    round_id: str
    set_id: int
    icps: Tuple[Mapping[str, Any], ...]
    slots: Tuple[AcceptedSlot, ...]
    icp_hashes: Tuple[str, ...]
    content_hashes: Tuple[str, ...]
    icp_leaf_hashes: Tuple[str, ...]
    benchmark_root: str
    journal_entries: Tuple[Mapping[str, Any], ...]
    journal_head_hash: str
    journal_length: int
    attempts_used: int
    generation_started_at: str
    generation_finished_at: str


def stage_slice(result: BenchmarkResult, stage: int) -> Tuple[Mapping[str, Any], ...]:
    """The ICPs the one stage receives: every slot, 0..29, in slot order."""

    positions = stage_positions(stage)
    return tuple(result.icps[position] for position in positions)


def _iso_timestamp(value: Any, field_name: str) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ArenaContractError("%s must be timezone-aware" % field_name)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, str):
        return value
    raise ArenaContractError("%s must be a datetime or ISO-8601 string" % field_name)


def commit_benchmark(
    result: BenchmarkResult,
    *,
    round_configuration_hash: str,
    participants: Sequence[Mapping[str, Any]],
    tie_break_block: Mapping[str, Any],
    evaluation_date: str,
    started_at: Any,
    finished_at: Any,
) -> Dict[str, Any]:
    """Build the unsigned benchmark commitment for ``result``.

    ``tie_break_block`` carries ``number`` and ``hash`` of the finalized
    post-cutoff block. The commitment binds the journal head and length, the
    participant set, and the ordered roots; the service signs it.
    """

    if not isinstance(evaluation_date, str) or not _ISO_DATE_RE.match(evaluation_date):
        raise ArenaContractError("evaluation_date must be YYYY-MM-DD")
    if not isinstance(tie_break_block, Mapping):
        raise ArenaContractError("tie_break_block must carry number and hash")
    number = tie_break_block.get("number")
    block_hash = tie_break_block.get("hash")
    if isinstance(number, bool) or not isinstance(number, int) or number < 0:
        raise ArenaContractError("tie_break_block.number must be a non-negative integer")
    if not isinstance(block_hash, str) or not _BLOCK_HASH_RE.match(block_hash):
        raise ArenaContractError("tie_break_block.hash must be 0x-prefixed lowercase hex")
    roots = contracts.benchmark_roots(result.icp_hashes)
    document = {
        "schema_version": contracts.BENCHMARK_COMMITMENT_SCHEMA_VERSION,
        "round_id": result.round_id,
        "configuration_hash": contracts.require_sha256(round_configuration_hash, "configuration_hash"),
        "participant_set_hash": contracts.participant_set_hash(participants),
        "tie_break_block_number": number,
        "tie_break_block_hash": block_hash,
        "journal_head_hash": result.journal_head_hash,
        "journal_length": result.journal_length,
        "evaluation_date": evaluation_date,
        "benchmark_root": roots["benchmark_root"],
        "icp_leaf_hashes": roots["icp_leaf_hashes"],
        "generation_started_at": _iso_timestamp(started_at, "started_at"),
        "generation_finished_at": _iso_timestamp(finished_at, "finished_at"),
    }
    return contracts.finalize_benchmark_commitment(document)


def generator_configuration(
    *,
    exclusion_count: int = DEFAULT_EXCLUSION_COUNT,
    max_generation_attempts: int = contracts.MAX_GENERATION_ATTEMPTS,
) -> Dict[str, Any]:
    """The ``generator`` object of the round configuration (section 5.1).

    ``prompt_hash`` binds the count-parameterized templates, and
    ``settings`` binds the request settings, the explicit canonicalization
    arguments, and a hash of every Lab vocabulary the flow reads, so a Lab
    constant change is visible as a generator identity change.
    """

    want = _require_exclusion_count(exclusion_count)
    placeholder_icp = {
        "industry": "{industry}",
        "sub_industry": "{sub_industry}",
        "geography": "{geography}",
        "employee_count": ["{employee_count}"],
        "company_stage": "{company_stage}",
    }
    vocabulary_hash = contracts.document_hash(
        {
            "industries": list(INDUSTRIES),
            "sub_industries": {key: list(value) for key, value in SUB_INDUSTRIES.items()},
            "company_stages": list(SUPPORTED_STAGES),
            "stage_employee_buckets": {key: list(value) for key, value in STAGE_EMPLOYEE_BUCKETS.items()},
            "geographies": list(GEOGRAPHIES),
            "international_geographies": list(INTERNATIONAL_GEOGRAPHIES),
            "intent_signals": list(INTENT_SIGNALS),
        }
    )
    return {
        "prompt_hash": contracts.document_hash(
            {
                "system_template": _SYSTEM_PROMPT_TEMPLATE,
                "user_template": _USER_PROMPT_FIXED,
                "replacement_user_template": _USER_PROMPT_REPLACEMENT,
                "stage_distribution": [list(item) for item in _STAGE_DISTRIBUTION_20],
            }
        ),
        "exclusion_prompt_hash": contracts.document_hash(
            {"prompt": build_exclusion_prompt(placeholder_icp, want), "count": want}
        ),
        "model": GENERATOR_MODEL,
        "settings": {
            "temperature": GENERATOR_SETTINGS["temperature"],
            "max_tokens": GENERATOR_SETTINGS["max_tokens"],
            "generation_timeout_seconds": GENERATION_TIMEOUT_SECONDS,
            "exclusion_temperature": EXCLUSION_SETTINGS["temperature"],
            "exclusion_timeout_seconds": EXCLUSION_TIMEOUT_SECONDS,
            "exclusion_count": want,
            "employee_bucket_radius": EMPLOYEE_BUCKET_RADIUS,
            "all_employee_buckets": ALL_EMPLOYEE_BUCKETS,
            "rejection_rules": list(REJECTION_RULES),
            "content_hash_excluded_fields": list(CONTENT_HASH_EXCLUDED_FIELDS),
            "vocabulary_hash": vocabulary_hash,
        },
        "journal_schema_version": contracts.GENERATION_JOURNAL_SCHEMA_VERSION,
        "batch_sizes": list(contracts.GENERATION_BATCH_SIZES),
        "max_generation_attempts": int(max_generation_attempts),
    }


# ---------------------------------------------------------------------------
# Generation flow with journal replay (section 8)
# ---------------------------------------------------------------------------


class _GenerationRun:
    """One deterministic execution of the section 8 flow.

    The run re-executes from the beginning on every call. While recorded
    journal entries remain, each step verifies the entry it would have
    written and reads stored objects instead of calling the provider; once
    the recorded journal is exhausted, the run continues live.
    """

    def __init__(
        self,
        *,
        round_id: str,
        set_id: int,
        provider: GenerationProvider,
        journal: GenerationJournal,
        object_store: ObjectStore,
        clock: Callable[[], datetime],
        max_attempts: int,
        exclusion_count: int,
    ) -> None:
        if not isinstance(round_id, str) or not contracts.ROUND_ID_RE.match(round_id):
            raise ArenaContractError("round_id has an invalid shape")
        if isinstance(set_id, bool) or not isinstance(set_id, int) or set_id < 1:
            raise ArenaContractError("set_id must be a positive integer")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 3 <= max_attempts <= 64:
            raise ArenaContractError("max_attempts must be an integer in 3..64")
        self._round_id = round_id
        self._set_id = set_id
        self._provider = provider
        self._journal = journal
        self._store = object_store
        self._clock = clock
        self._max_attempts = max_attempts
        self._exclusion_count = _require_exclusion_count(exclusion_count)
        recorded = [dict(entry) for entry in journal.entries()]
        contracts.verify_journal_chain(recorded)
        self._recorded: Tuple[Dict[str, Any], ...] = tuple(recorded)
        self._entries: List[Dict[str, Any]] = list(recorded)
        self._cursor = 0
        self._accepted: Dict[int, AcceptedSlot] = {}
        self._content_hashes: Dict[str, int] = {}
        self._intent_signatures: Dict[str, int] = {}
        self._attempts = 0
        self._replacements = 0

    # -- journal primitives -------------------------------------------------

    @property
    def _replaying(self) -> bool:
        return self._cursor < len(self._recorded)

    def _head(self) -> str:
        return self._entries[-1]["entry_hash"] if self._entries else ""

    def _timestamp(self) -> str:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise ArenaContractError("clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _semantic(entry: Mapping[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in entry.items() if key not in _JOURNAL_ENVELOPE_FIELDS}

    def _peek(self) -> Optional[Dict[str, Any]]:
        return self._recorded[self._cursor] if self._replaying else None

    def _consume(self, expected: Mapping[str, Any], *, journal_fields: Sequence[str] = ()) -> Dict[str, Any]:
        """Verify the next recorded entry against ``expected`` and consume it.

        ``journal_fields`` are the fields whose values only the journal knows
        (response hashes and refs); every other semantic field must match.
        """

        recorded = self._recorded[self._cursor]
        semantic = self._semantic(recorded)
        for name in journal_fields:
            semantic.pop(name, None)
        if semantic != dict(expected):
            raise BenchmarkReplayError(
                "journal entry %d (%s) does not match re-execution (%s)"
                % (recorded["sequence"], recorded.get("kind"), expected.get("kind"))
            )
        self._cursor += 1
        return recorded

    def _append(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        full: Dict[str, Any] = {
            "schema_version": contracts.GENERATION_JOURNAL_SCHEMA_VERSION,
            "sequence": len(self._entries),
        }
        full.update(entry)
        full["timestamp"] = self._timestamp()
        full["prev_hash"] = self._head()
        expected = contracts.finalize_journal_entry(full)
        finalized = self._journal.append(full)
        if dict(finalized) != expected:
            raise BenchmarkReplayError("journal returned an entry that differs from the finalized entry")
        self._entries.append(expected)
        return expected

    def _emit(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
        """Verify-and-consume while replaying, append while live."""

        if self._replaying:
            return self._consume(entry)
        return self._append(entry)

    def _load_object(self, recorded: Mapping[str, Any], kind: str) -> Any:
        ref = recorded.get("response_ref")
        expected_ref = object_ref(self._round_id, kind, recorded["sequence"])
        if ref != expected_ref:
            raise BenchmarkReplayError("journal entry %d references %r, expected %r" % (recorded["sequence"], ref, expected_ref))
        data = self._store.get(ref)
        if contracts.hash_bytes(data) != recorded.get("response_hash"):
            raise BenchmarkReplayError("stored object %s does not match its journaled hash" % ref)
        try:
            return json.loads(data.decode("utf-8"), parse_constant=_reject_json_constant)
        except (UnicodeDecodeError, ValueError) as exc:
            raise BenchmarkReplayError("stored object %s is not JSON" % ref) from exc

    def _store_object(self, response: Any, kind: str) -> Tuple[str, str, bytes]:
        """Canonical bytes of ``response`` written at the ref of the entry
        about to be appended. Raises ``ValueError`` for unencodable data."""

        data = canonical_json(response).encode("utf-8")
        ref = object_ref(self._round_id, kind, len(self._entries))
        # A storage hiccup must not cost a generation attempt and a provider
        # call: the write is idempotent, so retry it briefly before giving up.
        for attempt in range(3):
            try:
                self._store.put(ref, data)
                break
            except ValueError:
                raise
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.2 * (attempt + 1))
        return ref, contracts.hash_bytes(data), data

    # -- flow ---------------------------------------------------------------

    def _missing_slots(self) -> List[int]:
        return [slot for slot in range(contracts.BENCHMARK_ICP_COUNT) if slot not in self._accepted]

    def _fail(self) -> BenchmarkGenerationFailed:
        return BenchmarkGenerationFailed(
            missing_slots=self._missing_slots(),
            attempts_used=self._attempts,
            journal_length=len(self._entries),
        )

    def _dispatch(
        self,
        batch_id: str,
        slots: Sequence[int],
        industries: Sequence[str],
        *,
        replacement: bool,
    ) -> Tuple[Dict[str, Any], Any]:
        """Send one generation request until it reaches a terminal outcome.

        Returns the ``response`` journal entry and the response object. Every
        dispatch consumes one attempt; an unknown outcome re-dispatches the
        same request. Raises :class:`BenchmarkGenerationFailed` once the
        attempt limit is reached.
        """

        body = build_generation_request(
            count=len(slots), industries=industries, set_id=self._set_id, replacement=replacement
        )
        while True:
            if self._attempts >= self._max_attempts:
                raise self._fail()
            self._attempts += 1
            attempt = self._attempts
            base = {
                "batch_id": batch_id,
                "attempt": attempt,
                "slots": [int(slot) for slot in slots],
                "industries": [str(industry) for industry in industries],
            }
            request_hash = contracts.document_hash({**base, "body": body})
            was_replaying = self._replaying
            self._emit({"kind": "request", **base, "request_hash": request_hash})
            if was_replaying:
                following = self._peek()
                if following is None:
                    # The journal ends with this request: the previous run
                    # crashed mid-request, so its outcome is unknown.
                    self._append({"kind": "unknown", **base})
                    continue
                if following.get("kind") == "unknown":
                    self._consume({"kind": "unknown", **base})
                    continue
                if following.get("kind") == "response":
                    recorded = self._consume(
                        {"kind": "response", **base, "request_hash": request_hash},
                        journal_fields=("response_hash", "response_ref"),
                    )
                    return recorded, self._load_object(recorded, GENERATION_OBJECT_KIND)
                raise BenchmarkReplayError(
                    "journal entry %d (%s) follows a request without a terminal outcome"
                    % (following["sequence"], following.get("kind"))
                )
            try:
                response = self._provider.chat(
                    messages=body["messages"],
                    temperature=body["temperature"],
                    max_tokens=body["max_tokens"],
                    timeout_seconds=GENERATION_TIMEOUT_SECONDS,
                )
            except ProviderFailure:
                self._append({"kind": "unknown", **base})
                continue
            try:
                ref, response_hash, _ = self._store_object(response, GENERATION_OBJECT_KIND)
            except ValueError:
                self._append({"kind": "unknown", **base})
                continue
            entry = self._append(
                {
                    "kind": "response",
                    **base,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "response_ref": ref,
                }
            )
            return entry, response

    def _exclusions(
        self,
        icp: Mapping[str, Any],
        base: Mapping[str, Any],
        slot: int,
        icp_content_hash: str,
    ) -> Optional[List[str]]:
        """Run the copied exclusion prompt for one contract-applied ICP.

        Returns the exclusion entries, or ``None`` when the provider failed,
        the response could not be stored or parsed, or the list is empty.
        """

        body = build_exclusion_request(icp, self._exclusion_count)
        request_hash = contracts.document_hash({"slot": slot, "content_hash": icp_content_hash, "body": body})
        expected = {
            "kind": "exclusion",
            **base,
            "slot": slot,
            "request_hash": request_hash,
            "content_hash": icp_content_hash,
        }
        if self._replaying:
            recorded = self._consume(expected, journal_fields=("response_hash", "response_ref"))
            if "response_hash" not in recorded:
                return None
            response = self._load_object(recorded, EXCLUSION_OBJECT_KIND)
        else:
            try:
                response = self._provider.chat(
                    messages=body["messages"],
                    temperature=body["temperature"],
                    max_tokens=None,
                    timeout_seconds=EXCLUSION_TIMEOUT_SECONDS,
                )
            except ProviderFailure:
                self._append(expected)
                return None
            try:
                ref, response_hash, _ = self._store_object(response, EXCLUSION_OBJECT_KIND)
            except ValueError:
                self._append(expected)
                return None
            self._append({**expected, "response_hash": response_hash, "response_ref": ref})
        try:
            entries = parse_exclusion_response(response_content(response), self._exclusion_count)
        except ExclusionParseError:
            return None
        return entries or None

    def _process_response(
        self,
        batch_id: str,
        attempt: int,
        slots: Sequence[int],
        industries: Sequence[str],
        response_entry: Mapping[str, Any],
        response: Any,
    ) -> None:
        base = {
            "batch_id": batch_id,
            "attempt": attempt,
            "slots": [int(slot) for slot in slots],
            "industries": [str(industry) for industry in industries],
        }
        try:
            raw_icps = parse_generation_content(response_content(response))
        except GenerationParseError:
            self._emit({"kind": "rejection", **base, "rejection_rule": "response.unparsable"})
            return
        requested = list(zip(slots, industries))
        filled: set = set()
        for raw in raw_icps:
            check = check_raw_icp(raw, requested=requested, filled=filled)
            if check.rejection_rule is not None:
                rejection: Dict[str, Any] = {"kind": "rejection", **base, "rejection_rule": check.rejection_rule}
                if check.slot is not None:
                    rejection["slot"] = check.slot
                if check.rejection_rule not in ("schema.not_an_object", "schema.document_limits"):
                    rejection["content_hash"] = contracts.document_hash(raw)
                self._emit(rejection)
                continue
            slot = check.slot
            assert slot is not None
            icp_id = "arena:%s:%s:%d" % (self._round_id, batch_id, slot)
            canonical = canonicalize_arena_icp(build_validated_icp(raw, check, icp_id=icp_id))
            icp_content_hash = content_hash(canonical)
            signature = intent_signature(canonical)
            if icp_content_hash in self._content_hashes:
                self._emit(
                    {"kind": "rejection", **base, "slot": slot, "rejection_rule": "duplicate.content_hash", "content_hash": icp_content_hash}
                )
                continue
            if signature in self._intent_signatures:
                self._emit(
                    {"kind": "rejection", **base, "slot": slot, "rejection_rule": "duplicate.intent_signature", "content_hash": icp_content_hash}
                )
                continue
            final = apply_arena_contract(canonical)
            exclusions = self._exclusions(final, base, slot, icp_content_hash)
            if not exclusions:
                self._emit(
                    {"kind": "rejection", **base, "slot": slot, "rejection_rule": "exclusion.failed", "content_hash": icp_content_hash}
                )
                continue
            final["excluded_companies"] = list(exclusions)
            try:
                contracts.check_strict_document(final)
            except ArenaContractError:
                self._emit(
                    {"kind": "rejection", **base, "slot": slot, "rejection_rule": "schema.document_limits", "content_hash": icp_content_hash}
                )
                continue
            final_hash = icp_hash(final)
            self._emit(
                {
                    "kind": "acceptance",
                    **base,
                    "slot": slot,
                    "icp_hash": final_hash,
                    "content_hash": icp_content_hash,
                    "response_hash": response_entry["response_hash"],
                    "response_ref": response_entry["response_ref"],
                }
            )
            self._accepted[slot] = AcceptedSlot(
                slot=slot,
                batch_id=batch_id,
                attempt=attempt,
                content_hash=icp_content_hash,
                icp_hash=final_hash,
                icp=final,
            )
            self._content_hashes[icp_content_hash] = slot
            self._intent_signatures[signature] = slot
            filled.add(slot)

    def run(self) -> BenchmarkResult:
        for plan in SLOT_PLAN:
            entry, response = self._dispatch(plan.batch_id, plan.slots, plan.industries, replacement=False)
            self._process_response(plan.batch_id, entry["attempt"], plan.slots, plan.industries, entry, response)
        while True:
            missing = self._missing_slots()
            if not missing:
                break
            if self._attempts >= self._max_attempts:
                raise self._fail()
            self._replacements += 1
            batch_id = "r%d" % self._replacements
            slots = tuple(missing)
            industries = tuple(slot_industry(slot) for slot in slots)
            entry, response = self._dispatch(batch_id, slots, industries, replacement=True)
            self._process_response(batch_id, entry["attempt"], slots, industries, entry, response)
        if self._replaying:
            raise BenchmarkReplayError(
                "journal holds %d entries beyond the completed generation"
                % (len(self._recorded) - self._cursor)
            )
        ordered = tuple(self._accepted[slot] for slot in range(contracts.BENCHMARK_ICP_COUNT))
        icp_hashes = tuple(item.icp_hash for item in ordered)
        roots = contracts.benchmark_roots(icp_hashes)
        entries = tuple(self._entries)
        return BenchmarkResult(
            round_id=self._round_id,
            set_id=self._set_id,
            icps=tuple(item.icp for item in ordered),
            slots=ordered,
            icp_hashes=icp_hashes,
            content_hashes=tuple(item.content_hash for item in ordered),
            icp_leaf_hashes=tuple(roots["icp_leaf_hashes"]),
            benchmark_root=roots["benchmark_root"],
            journal_entries=entries,
            journal_head_hash=entries[-1]["entry_hash"],
            journal_length=len(entries),
            attempts_used=self._attempts,
            generation_started_at=entries[0]["timestamp"],
            generation_finished_at=entries[-1]["timestamp"],
        )


def generate_benchmark(
    *,
    round_id: str,
    set_id: int,
    provider: GenerationProvider,
    journal: GenerationJournal,
    object_store: ObjectStore,
    clock: Callable[[], datetime],
    max_attempts: int = contracts.MAX_GENERATION_ATTEMPTS,
    exclusion_count: int = DEFAULT_EXCLUSION_COUNT,
) -> BenchmarkResult:
    """Generate, validate, and journal the fifty-ICP benchmark for a round.

    Section 8 flow: three fixed requests (``b1`` and ``b2`` over the ordered
    twenty industries, ``b3`` over the first ten), each response processed in
    returned order with the first valid unique ICP accepted per slot, then
    replacement requests (``r1``, ``r2``, ...) naming the missing slots and
    their industries until every slot is filled or ``max_attempts``
    generation requests have been sent. Accepted ICPs carry the Lab contract
    (``max_companies``, stage-widened ``employee_count``) and non-empty
    ``excluded_companies``; an exclusion failure rejects the slot.

    Passing a journal and object store from an interrupted run resumes it
    deterministically: recorded entries are verified against re-execution,
    stored response objects replace provider calls, a trailing request is
    journaled as ``unknown`` and retried, and no accepted slot is requested
    again. A completed journal replays to the same result without any
    provider call.
    """

    run = _GenerationRun(
        round_id=round_id,
        set_id=set_id,
        provider=provider,
        journal=journal,
        object_store=object_store,
        clock=clock,
        max_attempts=max_attempts,
        exclusion_count=exclusion_count,
    )
    return run.run()


__all__ = [
    "ALL_EMPLOYEE_BUCKETS",
    "AcceptedSlot",
    "BatchPlan",
    "BenchmarkError",
    "BenchmarkGenerationFailed",
    "BenchmarkReplayError",
    "BenchmarkResult",
    "CONTENT_HASH_EXCLUDED_FIELDS",
    "DEFAULT_EXCLUSION_COUNT",
    "EMPLOYEE_BUCKET_RADIUS",
    "EXCLUSION_SETTINGS",
    "EXCLUSION_TIMEOUT_SECONDS",
    "ExclusionParseError",
    "FIXED_BATCH_IDS",
    "GENERATION_TIMEOUT_SECONDS",
    "GENERATOR_MODEL",
    "GENERATOR_SETTINGS",
    "GenerationJournal",
    "GenerationParseError",
    "GenerationProvider",
    "INDUSTRIES",
    "MAX_EXCLUSION_COUNT",
    "MemoryJournal",
    "MemoryObjectStore",
    "ObjectStore",
    "ProviderFailure",
    "REJECTION_RULES",
    "RawIcpCheck",
    "SLOT_PLAN",
    "SUPPORTED_GEOGRAPHIES",
    "SUPPORTED_STAGES",
    "apply_arena_contract",
    "build_exclusion_prompt",
    "build_exclusion_request",
    "build_generation_prompts",
    "build_generation_request",
    "build_validated_icp",
    "canonicalize_arena_icp",
    "check_raw_icp",
    "commit_benchmark",
    "content_hash",
    "generate_benchmark",
    "generator_configuration",
    "icp_hash",
    "intent_signature",
    "object_ref",
    "parse_exclusion_response",
    "parse_generation_content",
    "response_content",
    "slot_batch",
    "slot_industry",
    "stage_distribution_lines",
    "stage_positions",
    "stage_slice",
]

"""L1 dev-eval harness: cheap deterministic per-iteration candidate scoring.

This is the §6.3-1 rung between "it builds" and the live L2 benchmark. Every
hypothesis today costs a full live-provider benchmark, so the inner loop gets
zero score feedback between edits. :func:`evaluate_dev` runs a candidate over
a frozen dev ICP set against a :class:`ProviderSnapshotStore` in replay mode
and scores the output with a deterministic mechanical scorer — no live
providers, no LLM calls, plan target <=$1/iteration, wall-clock
seconds-to-minutes.

Engine integration contract:

* ``code_loop_engine`` calls ``await evaluate_dev(candidate_runner=...,
  dev_items=..., snapshot_store=...)`` once per built candidate, per
  iteration. ``candidate_runner`` is any evaluator-style ``ModelRunner``
  callable ``(icp, context) -> [company, ...]`` — an in-process function, or
  a container runner launched with :func:`snapshot_store.dev_replay_bootstrap`
  prepended to its adapter bootstrap plus
  :func:`snapshot_store.container_replay_env` in its extra env.
* The engine ranks ``selected`` by ``DevEvalResult.aggregate_dev_score`` only
  when every built candidate is eligible for the exact same evaluator
  commitment. Otherwise it ignores all development scores and preserves
  ordinary build order.
* Iteration continuation / plateau-stop (§6.3-4) keys off dev-score deltas.

Dev-score discipline (leak-cluster guard, plan §8.2):

* The dev ICP set is built once with :func:`build_dev_icp_set`, which
  hard-excludes any ICP whose ref/hash/intent-signal signature appears in the
  private holdout window, and records the exclusion proof in its manifest.
* Dev scores are RANKING-ONLY — they order candidates within a run and are
  never promotion evidence. Expect and tolerate dev-vs-live divergence; the
  L2 live benchmark exists to catch it.
* ``dev_score_version`` pins the mechanical rubric; never compare dev scores
  across versions.

Score scale: each company is scored 0-100 from mechanical features shaped
like the live rubric (40 ICP-fit + 60 intent-evidence), the per-ICP score is
the verifier's capped sum(top-5)/5 (``per_icp_normalized_score`` — the exact
arithmetic ``_benchmark_style_score`` uses with
``RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE`` on), and the aggregate is the mean
over dev ICPs. Dev deltas are therefore comparable in spirit to live-gate
deltas without sharing their evidence weight.

The module is deterministic by construction (no wall clocks or unseeded
randomness) and is fail-closed at the terminal candidate selector.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import inspect
import re
from typing import Any, Awaitable, Callable, Mapping, Sequence, Union

from leadpoet_verifier.aggregation import per_icp_normalized_score
from research_lab.canonical import sha256_json
from research_lab.employee_buckets import normalize_employee_count_bucket

from .private_runtime import (
    PrivateModelRuntimeError,
    employee_count_buckets_for_icp,
    ensure_private_model_outputs,
)
from .snapshot_store import (
    MODE_REPLAY,
    DevSnapshotStoreError,
    ProviderSnapshotStore,
    SnapshotMiss,
)

DEV_SCORE_VERSION = "research-lab-dev-eval-mechanical-v1"
DEV_LEADS_PER_ICP = 5  # mirror evaluator._TOP5_LEADS_PER_ICP / the verifier lead budget

ModelRunner = Callable[
    [Mapping[str, Any], Mapping[str, Any]],
    Union[Awaitable[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]],
]
CompanyScorer = Callable[
    [Sequence[Mapping[str, Any]], Mapping[str, Any], bool],
    Union[Awaitable[list[float]], list[float]],
]

# Mechanical rubric weights. Shaped like the live rubric's 40-fit/60-intent
# split (leadpoet_verifier.aggregation.MAX_COMPANY_ICP_FIT_SCORE /
# MAX_COMPANY_INTENT_SIGNAL_SCORE) so a company score stays on the same
# 0-100 scale, but computed from mechanical features only — no LLM.
_FIT_COMPLETENESS_FIELDS = (
    "company_name",
    "company_website",
    "industry",
    "sub_industry",
    "employee_count",
    "country",
    "description",
)
_FIT_COMPLETENESS_POINTS_PER_FIELD = 2.0  # x7 fields = 14
_FIT_INDUSTRY_OVERLAP_POINTS = 12.0
_FIT_EMPLOYEE_BUCKET_POINTS = 8.0
_FIT_GEOGRAPHY_POINTS = 6.0  # fit total: 40
_INTENT_EVIDENCE_POINTS = 20.0
_INTENT_URL_POINTS = 10.0
_INTENT_DATE_POINTS = 10.0
_INTENT_OVERLAP_POINTS = 20.0  # intent total: 60

_TOKEN_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "that", "this", "from", "are", "its", "has", "have"}
)


class DevEvalError(RuntimeError):
    """Raised when a dev evaluation cannot run safely/deterministically."""


class DevSetLeakError(ValueError):
    """Raised when a built dev set overlaps the excluded holdout window."""


# ---------------------------------------------------------------------------
# Dev-set discipline (leak-cluster guard)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DevIcpSet:
    """A frozen dev ICP set with its exclusion-proof manifest."""

    items: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]
    dev_set_hash: str


def build_dev_icp_set(
    source_icps: Sequence[Mapping[str, Any]],
    *,
    exclude_window_hashes: Sequence[str],
    size: int,
    seed: str,
) -> DevIcpSet:
    """Deterministically sample a dev ICP set disjoint from the holdout window.

    ``source_icps`` accepts benchmark-item-shaped rows (``{"icp": {...},
    "icp_ref": ..., "icp_hash": ...}``) or raw ICP mappings.
    ``exclude_window_hashes`` carries the private holdout window's per-item
    ``icp_ref``/``icp_hash`` values (bare-hex and ``sha256:``-prefixed both
    match) and optionally intent-signal signatures; any source ICP matching on
    ref, hash, OR intent-signal signature is hard-excluded (leak-cluster
    guard: paraphrase near-duplicates share signal signatures). Selection is
    a seeded hash-rank sample — no RNG state, reproducible across Python
    versions — and the manifest records the exclusion proof.
    """
    if size <= 0:
        raise DevEvalError("dev ICP set size must be positive")
    seed_text = str(seed)
    exclusions = _normalized_exclusion_set(exclude_window_hashes)

    normalized: dict[str, dict[str, Any]] = {}
    for entry in source_icps:
        if not isinstance(entry, Mapping):
            continue
        item = _normalize_dev_item(entry)
        current = normalized.get(item["icp_hash"])
        if current is None or item["icp_ref"] < current["icp_ref"]:
            normalized[item["icp_hash"]] = item

    eligible: list[dict[str, Any]] = []
    excluded_proof: list[dict[str, Any]] = []
    for item in sorted(normalized.values(), key=lambda row: row["icp_hash"]):
        matched_on = _exclusion_matches(item, exclusions)
        if matched_on:
            excluded_proof.append(
                {
                    "icp_ref": item["icp_ref"],
                    "icp_hash": item["icp_hash"],
                    "matched_on": matched_on,
                }
            )
        else:
            eligible.append(item)

    if len(eligible) < size:
        raise DevEvalError(
            f"dev_icp_set_requires_{size}_eligible_icps_found_{len(eligible)}"
        )

    ranked = sorted(
        eligible,
        key=lambda row: sha256_json(
            {"dev_selection_seed": seed_text, "icp_hash": row["icp_hash"]}
        ),
    )
    selected = _select_diverse_dev_items(ranked, size=size)
    selected = sorted(selected, key=lambda row: row["icp_hash"])

    # Defense in depth: the guard above must make this unreachable.
    leaked = [item for item in selected if _exclusion_matches(item, exclusions)]
    if leaked:
        raise DevSetLeakError(
            "selected dev ICPs overlap the excluded holdout window: "
            + ", ".join(item["icp_ref"] for item in leaked)
        )

    dev_set_hash = compute_dev_set_hash(selected)
    payload = {
        "schema_version": "1.0",
        "manifest_type": "research_lab_dev_icp_set",
        "selection_seed": seed_text,
        "requested_size": int(size),
        "selected_count": len(selected),
        "source_icp_count": len(normalized),
        "dev_set_hash": dev_set_hash,
        "dev_score_version": DEV_SCORE_VERSION,
        "selected_items": [
            {"icp_ref": item["icp_ref"], "icp_hash": item["icp_hash"]}
            for item in selected
        ],
        "selection_policy": "seeded_greedy_diversity_v1",
        "diversity_proof": _dev_set_diversity_proof(
            eligible=eligible,
            selected=selected,
        ),
        "exclusion_proof": {
            "exclusion_entry_count": len(exclusions),
            "exclusion_set_hash": sha256_json(sorted(exclusions)),
            "excluded_item_count": len(excluded_proof),
            "excluded_items": excluded_proof,
            # Must be empty by construction; recorded so auditors can verify
            # disjointness from the manifest alone.
            "selected_overlap_with_exclusions": [],
        },
    }
    manifest = {**payload, "manifest_hash": sha256_json(payload)}
    return DevIcpSet(
        items=tuple(dict(item) for item in selected),
        manifest=manifest,
        dev_set_hash=dev_set_hash,
    )


_DEV_DIVERSITY_FIELDS = (
    "industry",
    "sub_industry",
    "country_or_geography",
    "employee_count",
)


def _dev_diversity_values(item: Mapping[str, Any]) -> tuple[str, ...]:
    icp = item.get("icp") if isinstance(item.get("icp"), Mapping) else item
    country = str(icp.get("country") or icp.get("geography") or "")
    values = (
        str(icp.get("industry") or ""),
        str(icp.get("sub_industry") or ""),
        country,
        str(icp.get("employee_count") or ""),
    )
    return tuple(" ".join(value.strip().lower().split()) for value in values)


def _select_diverse_dev_items(
    seeded_items: Sequence[Mapping[str, Any]], *, size: int
) -> list[dict[str, Any]]:
    """Greedily maximize field novelty, using seeded order for every tie."""
    remaining = [dict(item) for item in seeded_items]
    selected: list[dict[str, Any]] = []
    covered = [set() for _field in _DEV_DIVERSITY_FIELDS]
    while remaining and len(selected) < size:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (
                sum(
                    bool(value) and value not in covered[field_index]
                    for field_index, value in enumerate(
                        _dev_diversity_values(remaining[index])
                    )
                ),
                -index,
            ),
        )
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        for field_index, value in enumerate(_dev_diversity_values(chosen)):
            if value:
                covered[field_index].add(value)
    return selected


def _dev_set_diversity_proof(
    *,
    eligible: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def _counts(items: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        values = [_dev_diversity_values(item) for item in items]
        return {
            field: len({row[index] for row in values if row[index]})
            for index, field in enumerate(_DEV_DIVERSITY_FIELDS)
        }

    return {
        "fields": list(_DEV_DIVERSITY_FIELDS),
        "eligible_unique_counts": _counts(eligible),
        "selected_unique_counts": _counts(selected),
    }


def compute_dev_set_hash(items: Sequence[Mapping[str, Any]]) -> str:
    """Order-insensitive identity hash of a dev ICP set."""
    hashes = sorted(_item_icp_hash(item) for item in items)
    return sha256_json({"dev_icp_hashes": hashes})


def _normalize_dev_item(entry: Mapping[str, Any]) -> dict[str, Any]:
    icp = entry.get("icp") if isinstance(entry.get("icp"), Mapping) else entry
    icp_hash = str(entry.get("icp_hash") or "") or sha256_json({"icp": dict(icp)})
    icp_ref = str(entry.get("icp_ref") or "") or f"dev_icp:{icp_hash.split(':', 1)[-1][:18]}"
    return {
        "icp": dict(icp),
        "icp_ref": icp_ref,
        "icp_hash": icp_hash,
        "intent_signal_signature": str(
            entry.get("intent_signal_signature") or intent_signal_signature(icp)
        ),
    }


def _item_icp_hash(item: Mapping[str, Any]) -> str:
    given = str(item.get("icp_hash") or "")
    if given:
        return given
    icp = item.get("icp") if isinstance(item.get("icp"), Mapping) else item
    return sha256_json({"icp": dict(icp)})


def _normalized_exclusion_set(entries: Sequence[str]) -> frozenset[str]:
    normalized: set[str] = set()
    for entry in entries or ():
        text = str(entry or "").strip()
        if not text:
            continue
        normalized.add(text)
        lowered = text.lower()
        if lowered.startswith("sha256:"):
            normalized.add(lowered.split(":", 1)[1])
        elif len(lowered) == 64 and all(ch in "0123456789abcdef" for ch in lowered):
            normalized.add(f"sha256:{lowered}")
    return frozenset(normalized)


def _exclusion_matches(item: Mapping[str, Any], exclusions: frozenset[str]) -> list[str]:
    matched: list[str] = []
    icp_hash = str(item.get("icp_hash") or "")
    if str(item.get("icp_ref") or "") in exclusions:
        matched.append("icp_ref")
    if icp_hash in exclusions or icp_hash.split(":", 1)[-1] in exclusions:
        matched.append("icp_hash")
    if str(item.get("intent_signal_signature") or "") in exclusions:
        matched.append("intent_signal_signature")
    return matched


def intent_signal_signature(icp: Mapping[str, Any]) -> str:
    """Leak-cluster signature for an ICP's intent signals.

    Local mirror of ``gateway.research_lab.icp_window.intent_signal_signature``
    (not imported: research_lab.eval keeps no gateway dependency). Paraphrase
    near-duplicate ICPs share this signature, so excluding by it extends the
    holdout guard beyond exact ref/hash matches.
    """
    signals = icp.get("intent_signals") or []
    if isinstance(signals, str):
        signals = [signals]
    normalized = sorted(
        {
            " ".join(str(signal).strip().lower().split())
            for signal in signals
            if str(signal).strip()
        }
    )
    if normalized:
        return "|".join(normalized)
    fallback = [
        str(icp.get("industry") or "").strip().lower(),
        str(icp.get("sub_industry") or "").strip().lower(),
        str(icp.get("product_service") or "").strip().lower(),
    ]
    return "|".join(part for part in fallback if part) or "unknown"


# ---------------------------------------------------------------------------
# Deterministic mechanical scorer
# ---------------------------------------------------------------------------


class MechanicalDevScorer:
    """Deterministic, LLM-free company scorer for dev ranking.

    CompanyScorer-compatible (``(companies, icp, is_reference_model) ->
    [score, ...]``) so an injected LLM scorer can replace it later without
    touching :func:`evaluate_dev`. Scores are 0-100 per company:

    * ICP fit (max 40): field completeness (14), industry/keyword token
      overlap with the ICP (12), employee-bucket match (8), geography (6).
      A company whose employee bucket is missing or outside the ICP's allowed
      buckets scores 0 outright — mirroring the live scorer's pre-filter,
      which drops such companies before LLM scoring.
    * Intent evidence (max 60): evidence present (20), evidence URL (10),
      evidence date (10), evidence-text token overlap with the ICP's intent
      signals (20).
    * Duplicates (same normalized website domain or company name seen earlier
      in the list) score 0 — mirroring the live ``seen_companies`` dedupe.

    RANKING-ONLY: mechanical scores order candidates within a run; they are
    never promotion evidence.
    """

    def __call__(
        self,
        companies: Sequence[Mapping[str, Any]],
        icp: Mapping[str, Any],
        is_reference_model: bool,
    ) -> list[float]:
        seen: set[str] = set()
        scores: list[float] = []
        for company in companies:
            if not isinstance(company, Mapping):
                scores.append(0.0)
                continue
            duplicate_key = _company_duplicate_key(company)
            if duplicate_key and duplicate_key in seen:
                scores.append(0.0)
                continue
            if duplicate_key:
                seen.add(duplicate_key)
            scores.append(mechanical_company_score(company, icp))
        return scores


def mechanical_company_score(
    company: Mapping[str, Any],
    icp: Mapping[str, Any],
) -> float:
    """Score one company 0-100 from mechanical features (see scorer docstring)."""
    allowed_buckets = employee_count_buckets_for_icp(icp)
    company_bucket = normalize_employee_count_bucket(
        company.get("employee_count"), default=""
    )
    if not company_bucket or company_bucket not in allowed_buckets:
        return 0.0

    fit = 0.0
    for field_name in _FIT_COMPLETENESS_FIELDS:
        if str(company.get(field_name) or "").strip():
            fit += _FIT_COMPLETENESS_POINTS_PER_FIELD
    icp_industry_tokens = _tokens(
        " ".join(
            str(icp.get(name) or "")
            for name in ("industry", "sub_industry", "product_service", "required_attribute")
        )
    )
    company_industry_tokens = _tokens(
        " ".join(
            str(company.get(name) or "")
            for name in ("industry", "sub_industry", "description")
        )
    )
    fit += _FIT_INDUSTRY_OVERLAP_POINTS * _overlap_fraction(
        icp_industry_tokens, company_industry_tokens
    )
    fit += _FIT_EMPLOYEE_BUCKET_POINTS
    icp_geo_tokens = _tokens(
        " ".join(str(icp.get(name) or "") for name in ("geography", "country"))
    )
    company_geo_tokens = _tokens(
        " ".join(
            str(company.get(name) or "")
            for name in ("country", "hq_country", "state", "hq_state")
        )
    )
    if icp_geo_tokens and icp_geo_tokens & company_geo_tokens:
        fit += _FIT_GEOGRAPHY_POINTS

    evidence = _intent_evidence(company)
    intent = 0.0
    if evidence["texts"]:
        intent += _INTENT_EVIDENCE_POINTS
        if any(evidence["urls"]):
            intent += _INTENT_URL_POINTS
        if any(evidence["dates"]):
            intent += _INTENT_DATE_POINTS
        icp_intent_tokens = _tokens(" ".join(_icp_intent_texts(icp)))
        evidence_tokens = _tokens(" ".join(evidence["texts"]))
        intent += _INTENT_OVERLAP_POINTS * _overlap_fraction(
            icp_intent_tokens, evidence_tokens
        )
    return round(min(100.0, fit + intent), 4)


def _company_duplicate_key(company: Mapping[str, Any]) -> str:
    website = str(company.get("company_website") or "").strip().lower()
    if website:
        domain = re.sub(r"^https?://", "", website)
        domain = domain.removeprefix("www.").split("/", 1)[0].strip()
        if domain:
            return f"domain:{domain}"
    name = " ".join(str(company.get("company_name") or "").strip().lower().split())
    return f"name:{name}" if name else ""


def _intent_evidence(company: Mapping[str, Any]) -> dict[str, list[str]]:
    texts: list[str] = []
    urls: list[str] = []
    dates: list[str] = []

    def _collect(record: Mapping[str, Any]) -> None:
        text = str(
            record.get("intent_signal")
            or record.get("signal")
            or record.get("description")
            or record.get("snippet")
            or ""
        ).strip()
        if text:
            texts.append(text)
        urls.append(str(record.get("url") or "").strip())
        dates.append(str(record.get("date") or "").strip())

    signals = company.get("intent_signals")
    if isinstance(signals, Sequence) and not isinstance(signals, (str, bytes, bytearray)):
        for signal in signals:
            if isinstance(signal, Mapping):
                _collect(signal)
            elif str(signal or "").strip():
                texts.append(str(signal).strip())
                urls.append("")
                dates.append("")
    if isinstance(company.get("intent"), Mapping):
        _collect(company["intent"])
    flat_text = str(company.get("intent_signal") or "").strip()
    if flat_text:
        texts.append(flat_text)
        urls.append(str(company.get("intent_url") or "").strip())
        dates.append(str(company.get("intent_date") or "").strip())
    return {"texts": texts, "urls": urls, "dates": dates}


def _icp_intent_texts(icp: Mapping[str, Any]) -> list[str]:
    texts: list[str] = []
    signals = icp.get("intent_signals") or []
    if isinstance(signals, str):
        signals = [signals]
    if isinstance(signals, Sequence) and not isinstance(signals, (bytes, bytearray)):
        for signal in signals:
            if isinstance(signal, Mapping):
                text = str(signal.get("intent_signal") or signal.get("signal") or "").strip()
            else:
                text = str(signal or "").strip()
            if text:
                texts.append(text)
    flat = str(icp.get("intent_signal") or icp.get("intent_signal_text") or "").strip()
    if flat:
        texts.append(flat)
    return texts


def _tokens(text: str) -> frozenset[str]:
    words = re.split(r"[^a-z0-9]+", str(text or "").lower())
    return frozenset(
        word for word in words if len(word) >= 3 and word not in _TOKEN_STOPWORDS
    )


def _overlap_fraction(reference: frozenset[str], candidate: frozenset[str]) -> float:
    if not reference:
        return 0.0
    return round(len(reference & candidate) / len(reference), 4)


# ---------------------------------------------------------------------------
# The L1 evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DevEvalResult:
    """Per-ICP rows + aggregate for one candidate's dev evaluation.

    ``aggregate_dev_score`` is on the capped-top-5 live-gate scale (see module
    docstring) and is RANKING-ONLY — never promotion evidence.
    """

    dev_score_version: str
    aggregate_dev_score: float
    per_icp: tuple[dict[str, Any], ...]
    icp_count: int
    scored_icp_count: int
    snapshot_miss_count: int
    failure_count: int
    dev_set_hash: str
    snapshot_manifest_hash: str
    eligible: bool
    eligibility_reason: str
    execution_coverage: float
    true_miss_count: int
    zero_output_count: int
    miss_policy: str
    score_commitment: str
    receipt_root: str = ""
    run_label: str = ""
    ranking_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "dev_score_version": self.dev_score_version,
            "aggregate_dev_score": self.aggregate_dev_score,
            "per_icp": [dict(row) for row in self.per_icp],
            "icp_count": self.icp_count,
            "scored_icp_count": self.scored_icp_count,
            "snapshot_miss_count": self.snapshot_miss_count,
            "failure_count": self.failure_count,
            "dev_set_hash": self.dev_set_hash,
            "snapshot_manifest_hash": self.snapshot_manifest_hash,
            "eligible": self.eligible,
            "eligibility_reason": self.eligibility_reason,
            "execution_coverage": self.execution_coverage,
            "true_miss_count": self.true_miss_count,
            "zero_output_count": self.zero_output_count,
            "miss_policy": self.miss_policy,
            "score_commitment": self.score_commitment,
            "receipt_root": self.receipt_root,
            "run_label": self.run_label,
            "ranking_only": self.ranking_only,
        }


async def evaluate_dev(
    *,
    candidate_runner: ModelRunner,
    dev_items: Sequence[Mapping[str, Any]],
    snapshot_store: ProviderSnapshotStore,
    scorer: CompanyScorer | None = None,
    run_label: str = "",
    install_replay_seams: bool = True,
    require_manifest: bool = False,
) -> DevEvalResult:
    """Run one candidate over the frozen dev ICP set against replayed providers.

    ``candidate_runner`` follows the evaluator's ModelRunner contract
    (sync or async ``(icp, context) -> [company, ...]``). The snapshot store
    must be in replay mode; with ``install_replay_seams`` (default) the
    store's in-process urllib/requests/httpx replay seams are installed
    around the run — the same seams ``private_runtime``'s bug-35 hooks patch
    — so runners need no snapshot awareness. Container runners instead carry
    :func:`snapshot_store.dev_replay_bootstrap` +
    :func:`snapshot_store.container_replay_env` (the engine wiring wave).

    Per-ICP failures (snapshot misses under the strict policy, runner or
    scorer errors) never abort the paid run: the result is marked ineligible,
    causing the terminal selector to discard all development scores for that
    run. ICPs run serially in dev-set order, so the result is deterministic
    for a given candidate + snapshot set. A legitimate empty result is a
    successful zero score, not an infrastructure failure.
    """
    if candidate_runner is None:
        raise DevEvalError("candidate_runner is required")
    if not dev_items:
        raise DevEvalError("dev_items are required")
    if snapshot_store.mode != MODE_REPLAY:
        raise DevEvalError("evaluate_dev requires a replay-mode snapshot store")

    manifest = snapshot_store.load_manifest()
    if manifest is None:
        if require_manifest:
            raise DevEvalError("snapshot-set manifest is required and missing")
        snapshot_manifest_hash = ""
    else:
        verification = snapshot_store.verify_manifest(manifest)
        if not verification["passed"]:
            raise DevEvalError(
                "snapshot-set manifest failed verification: "
                + "; ".join(verification["errors"])
            )
        snapshot_manifest_hash = str(manifest.get("manifest_hash") or "")

    company_scorer = scorer or MechanicalDevScorer()
    dev_set_hash = compute_dev_set_hash(dev_items)
    run_context = {
        "dev_eval": True,
        "dev_score_version": DEV_SCORE_VERSION,
        "run_label": str(run_label or ""),
        "replay_miss_policy": snapshot_store.miss_policy,
        "snapshot_manifest_hash": snapshot_manifest_hash,
    }

    if install_replay_seams:
        with snapshot_store.replay_installed():
            rows = await _score_dev_items(
                candidate_runner=candidate_runner,
                dev_items=dev_items,
                scorer=company_scorer,
                run_context=run_context,
            )
    else:
        rows = await _score_dev_items(
            candidate_runner=candidate_runner,
            dev_items=dev_items,
            scorer=company_scorer,
            run_context=run_context,
        )

    dev_scores = [float(row["dev_score"]) for row in rows]
    aggregate = round(sum(dev_scores) / len(dev_scores), 6) if dev_scores else 0.0
    failure_count = sum(1 for row in rows if row["failure_reason"])
    miss_count = sum(1 for row in rows if row["snapshot_miss"])
    successful_count = len(rows) - failure_count
    eligible = bool(
        len(rows) == 8
        and failure_count == 0
        and miss_count == 0
        and snapshot_store.miss_policy == "strict"
    )
    if len(rows) != 8:
        eligibility_reason = "dev_set_size_must_equal_eight"
    elif snapshot_store.miss_policy != "strict":
        eligibility_reason = "snapshot_miss_policy_not_strict"
    elif miss_count:
        eligibility_reason = "snapshot_miss"
    elif failure_count:
        eligibility_reason = "evaluation_failure"
    else:
        eligibility_reason = "eligible"
    commitment_payload = {
        "schema_version": "research_lab.dev_score_commitment.v1",
        "dev_score_version": DEV_SCORE_VERSION,
        "dev_set_hash": dev_set_hash,
        "snapshot_manifest_hash": snapshot_manifest_hash,
        "miss_policy": snapshot_store.miss_policy,
    }
    return DevEvalResult(
        dev_score_version=DEV_SCORE_VERSION,
        aggregate_dev_score=aggregate,
        per_icp=tuple(rows),
        icp_count=len(rows),
        scored_icp_count=successful_count,
        snapshot_miss_count=miss_count,
        failure_count=failure_count,
        dev_set_hash=dev_set_hash,
        snapshot_manifest_hash=snapshot_manifest_hash,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
        execution_coverage=round(successful_count / len(rows), 6) if rows else 0.0,
        true_miss_count=miss_count,
        zero_output_count=sum(1 for row in rows if row["zero_output"]),
        miss_policy=snapshot_store.miss_policy,
        score_commitment=sha256_json(commitment_payload),
        run_label=str(run_label or ""),
    )


async def _score_dev_items(
    *,
    candidate_runner: ModelRunner,
    dev_items: Sequence[Mapping[str, Any]],
    scorer: CompanyScorer,
    run_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in dev_items:
        icp = item.get("icp")
        if not isinstance(icp, Mapping):
            raise DevEvalError("dev item is missing its ICP payload")
        icp_ref = str(item.get("icp_ref") or item.get("icp_hash") or "")
        failure_reason = ""
        snapshot_miss = False
        outputs: list[Mapping[str, Any]] = []
        try:
            outputs = ensure_private_model_outputs(
                await _call_runner(candidate_runner, icp, run_context),
                context_label=f"dev candidate for ICP {icp_ref}",
                require_non_empty=False,
            )
        except SnapshotMiss as exc:
            snapshot_miss = True
            failure_reason = f"dev_snapshot_miss:{exc.request_key}"
        except DevSnapshotStoreError as exc:
            failure_reason = f"dev_snapshot_error:{exc}"
        except PrivateModelRuntimeError as exc:
            failure_reason = f"dev_runner_error:{exc}"
        except Exception as exc:  # noqa: BLE001 - a crashing candidate ranks 0, never aborts
            failure_reason = f"dev_runner_error:{type(exc).__name__}: {exc}"

        scores: list[float] = []
        if outputs and not failure_reason:
            try:
                scores = [
                    float(score or 0.0)
                    for score in await _maybe_await(scorer(outputs, icp, False))
                ]
            except Exception as exc:  # noqa: BLE001 - scorer bugs must not abort the run
                failure_reason = f"dev_scorer_error:{type(exc).__name__}: {exc}"
                scores = []
        zero_output = not outputs and not failure_reason

        top_scores = sorted(scores, reverse=True)[:DEV_LEADS_PER_ICP]
        dev_score = float(per_icp_normalized_score(top_scores, max_leads=DEV_LEADS_PER_ICP))
        rows.append(
            {
                "icp_ref": icp_ref,
                "icp_hash": str(item.get("icp_hash") or ""),
                "dev_score": round(dev_score, 6),
                "company_count": len(outputs),
                "scored_company_count": len(scores),
                "company_scores": [round(score, 6) for score in scores],
                "failure_reason": failure_reason,
                "snapshot_miss": snapshot_miss,
                "zero_output": zero_output,
            }
        )
    return rows


async def _maybe_await(value: Any) -> Any:
    # Local mirror of evaluator._maybe_await; not imported so this module
    # stays dormant-standalone with no coupling to evaluator internals.
    if asyncio.iscoroutine(value):
        return await value
    return value


async def _call_runner(
    runner: ModelRunner,
    icp: Mapping[str, Any],
    context: Mapping[str, Any],
) -> Sequence[Mapping[str, Any]]:
    # Local mirror of evaluator._call_model_runner (same sync/async semantics)
    # so dev_eval keeps no dependency on evaluator private helpers.
    if inspect.iscoroutinefunction(runner) or inspect.iscoroutinefunction(
        getattr(runner, "__call__", None)
    ):
        result = runner(icp, context)
    else:
        result = await asyncio.to_thread(runner, icp, context)
    if inspect.isawaitable(result):
        return await result
    return result

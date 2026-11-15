"""Apply only legacy company progress to an exact reviewed repository pair.

The reference bytes come from fa88187ce, not an observed image extraction.
The 51bc image must match this complete pair before composition can succeed.
Unknown, partial and mixed source generations fail closed.
"""
from __future__ import annotations

import hashlib

CORRECTION_ID = "legacy_company_independent_progress_v1"
REFERENCE_SOURCE_COMMIT = "fa88187ceebc11ca6190c1158d035cc874373f95"
FROZEN_IMAGE_DIGEST = "sha256:51bc6de48dfb53de7903029b39f04e091b0d5af4669c0b4c333f947965a5116a"
SCORING_FILE = "lab_arena/scoring.py"
COMPETITION_FILE = "qualification/scoring/competition.py"
FILES = (SCORING_FILE, COMPETITION_FILE)
REFERENCE_SHA256 = {
    SCORING_FILE: "04e11a870b33ed821084449d43e56a5d1b0bbe3432f43d06ac36f8ce82b0bd7a",
    COMPETITION_FILE: "88b988931af3575c0ef431cac6ab27eb3c77f0051225fe1bafcdabf4e1b2a8da",
}
REPAIRED_SHA256 = {
    SCORING_FILE: "95357ccadd4e38120f6582f204366dcc948c33c51309752105e955540bb5ea0e",
    COMPETITION_FILE: "c8190c73cb55838c7f2924a4773b0203c44811daa26ba98d9f392fd1aefccbe8",
}
_CHANGES = {
    'lab_arena/scoring.py': (
        (
            (
                b'\n'
                b'class ScorerPolicyConflict(ScoringError):\n'
                b'    """The process environment conflicts with the signed scorer policy."""\n'
                b'\n'
                b'\n'
                b'\n'
            ),
            (
                b'\n'
                b'class ScorerPolicyConflict(ScoringError):\n'
                b'    """The process environment conflicts with the signed scorer policy."""\n'
                b'\n'
                b'\n'
                b'class _JudgeExhausted(ScoringError):\n'
                b'    """The native judge used its existing local attempt limit."""\n'
                b'\n'
                b'\n'
                b'\n'
            ),
        ),
        (
            (
                b'                unresolved_failures\n'
                b'            )\n'
                b'            continue\n'
                b'    raise ScoringError(\n'
                b'        ("judge_exhausted attempts=%d reason=%s %s"\n'
                b'         % (attempts, last_failure_reason or "unknown", last_private_diagnostic))[\n'
                b'             :MAX_FAILURE_DETAIL_CHARS\n'
            ),
            (
                b'                unresolved_failures\n'
                b'            )\n'
                b'            continue\n'
                b'    raise _JudgeExhausted(\n'
                b'        ("judge_exhausted attempts=%d reason=%s %s"\n'
                b'         % (attempts, last_failure_reason or "unknown", last_private_diagnostic))[\n'
                b'             :MAX_FAILURE_DETAIL_CHARS\n'
            ),
        ),
        (
            (
                b'    for miss in lease["misses"]:\n'
                b'        first_miss.setdefault(miss["cache_key"], miss)\n'
                b'    new_judgments = []\n'
                b'    def resolve(index):\n'
                b'        ref = refs[index]\n'
                b'        key = ref["cache_key"]\n'
                b'        if key not in raw_by_key:\n'
                b'            from contextlib import nullcontext\n'
                b'            from qualification.scoring.signal_judgment_cache import install_exchange\n'
            ),
            (
                b'    for miss in lease["misses"]:\n'
                b'        first_miss.setdefault(miss["cache_key"], miss)\n'
                b'    new_judgments = []\n'
                b'    failures_by_key = {}\n'
                b'    deferred_indexes = set()\n'
                b'    def resolve(index):\n'
                b'        ref = refs[index]\n'
                b'        key = ref["cache_key"]\n'
                b'        if key in failures_by_key:\n'
                b'            deferred_indexes.add(index)\n'
                b'            return None\n'
                b'        if key not in raw_by_key:\n'
                b'            from contextlib import nullcontext\n'
                b'            from qualification.scoring.signal_judgment_cache import install_exchange\n'
            ),
        ),
        (
            (
                b'            with (install_exchange(exchange_signal) if signal_exchange is not None else nullcontext()), \\\n'
                b'                 (install_company_fit_exchange(exchange_signal)\n'
                b'                  if signal_exchange is not None and enabled() else nullcontext()):\n'
                b'                rows = score_work_item(item, icp=icp, companies=[sliced[index]], scorer=scorer,\n'
                b'                                       _original_company_indexes=[index])\n'
                b'            if len(rows) != 1:\n'
                b'                raise ScoringError("intrinsic company judgment is missing")\n'
                b'            raw = {**raw_company_judgment(rows[0]), "company_judgment_policy": company_judgments.POLICY}\n'
            ),
            (
                b'            with (install_exchange(exchange_signal) if signal_exchange is not None else nullcontext()), \\\n'
                b'                 (install_company_fit_exchange(exchange_signal)\n'
                b'                  if signal_exchange is not None and enabled() else nullcontext()):\n'
                b'                try:\n'
                b'                    rows = score_work_item(item, icp=icp, companies=[sliced[index]], scorer=scorer,\n'
                b'                                           _original_company_indexes=[index])\n'
                b'                except _JudgeExhausted as exc:\n'
                b'                    # Keep the ordered pre-provider gates, while allowing later\n'
                b'                    # independent companies to checkpoint their real judgments.\n'
                b'                    failures_by_key[key] = exc\n'
                b'                    deferred_indexes.add(index)\n'
                b'                    return None\n'
                b'            if len(rows) != 1:\n'
                b'                raise ScoringError("intrinsic company judgment is missing")\n'
                b'            raw = {**raw_company_judgment(rows[0]), "company_judgment_policy": company_judgments.POLICY}\n'
            ),
        ),
        (
            (
                b'    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)\n'
                b'    full = apply_company_judgment_context(sliced, [None] * len(sliced), company_quality=False,\n'
                b'                                        scored_indexes=scored, resolve_raw=resolve,\n'
                b'                                        company_input_contract=getattr(scorer, "company_input_contract", ""))\n'
                b'    validate_breakdowns_for_item(full, icp=icp, companies=sliced,\n'
                b'        max_scored_companies=max_scored_companies, integrity_policy=True)\n'
                b'    return full, company_judgments.validate_new_company_judgments(\n'
            ),
            (
                b'    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)\n'
                b'    full = apply_company_judgment_context(sliced, [None] * len(sliced), company_quality=False,\n'
                b'                                        scored_indexes=scored, resolve_raw=resolve,\n'
                b'                                        deferred_indexes=deferred_indexes,\n'
                b'                                        company_input_contract=getattr(scorer, "company_input_contract", ""))\n'
                b'    if failures_by_key:\n'
                b'        raise next(iter(failures_by_key.values()))\n'
                b'    validate_breakdowns_for_item(full, icp=icp, companies=sliced,\n'
                b'        max_scored_companies=max_scored_companies, integrity_policy=True)\n'
                b'    return full, company_judgments.validate_new_company_judgments(\n'
            ),
        ),
    ),
    'qualification/scoring/competition.py': (
        (
            (
                b'def apply_legacy_company_judgment_context(\n'
                b'    companies: Sequence[Mapping[str, Any]], raw_breakdowns: Sequence[Mapping[str, Any] | None],\n'
                b'    *, scored_indexes: Sequence[int] | None = None,\n'
                b'    resolve_raw: Callable[[int], Mapping[str, Any]] | None = None,\n'
                b'    consumed_indexes: set[int] | None = None,\n'
                b'    company_input_contract: str = "",\n'
                b') -> list[dict[str, Any]]:\n'
                b'    """Replay the historical gates; omitted intrinsic rows must be proved skips."""\n'
                b'    if len(companies) != len(raw_breakdowns):\n'
                b'        raise CompetitionScorerInputError("company judgments must cover every company")\n'
                b'    from qualification.company_input_contract import validation_context\n'
            ),
            (
                b'def apply_legacy_company_judgment_context(\n'
                b'    companies: Sequence[Mapping[str, Any]], raw_breakdowns: Sequence[Mapping[str, Any] | None],\n'
                b'    *, scored_indexes: Sequence[int] | None = None,\n'
                b'    resolve_raw: Callable[[int], Mapping[str, Any] | None] | None = None,\n'
                b'    consumed_indexes: set[int] | None = None,\n'
                b'    deferred_indexes: set[int] | None = None,\n'
                b'    company_input_contract: str = "",\n'
                b') -> list[dict[str, Any]]:\n'
                b'    """Replay historical gates, with optional explicit unresolved-company deferral."""\n'
                b'    if len(companies) != len(raw_breakdowns):\n'
                b'        raise CompetitionScorerInputError("company judgments must cover every company")\n'
                b'    from qualification.company_input_contract import validation_context\n'
            ),
        ),
        (
            (
                b'            result.append(incompatible)\n'
                b'            continue\n'
                b'        raw = resolve_raw(index) if resolve_raw is not None else raw_breakdowns[index]\n'
                b'        if raw is None or raw.get("bucket_skipped"):\n'
                b'            raise CompetitionScorerInputError("company judgment is missing for an unskipped row")\n'
                b'        if consumed_indexes is not None:\n'
            ),
            (
                b'            result.append(incompatible)\n'
                b'            continue\n'
                b'        raw = resolve_raw(index) if resolve_raw is not None else raw_breakdowns[index]\n'
                b'        if raw is None and deferred_indexes is not None and index in deferred_indexes:\n'
                b'            continue\n'
                b'        if raw is None or raw.get("bucket_skipped"):\n'
                b'            raise CompetitionScorerInputError("company judgment is missing for an unskipped row")\n'
                b'        if consumed_indexes is not None:\n'
            ),
        ),
        (
            (
                b'    raw_breakdowns: Sequence[Mapping[str, Any]],\n'
                b'    *, contacts_required: bool = False, company_quality: bool = True,\n'
                b'    scored_indexes: Sequence[int] | None = None,\n'
                b'    resolve_raw: Callable[[int], Mapping[str, Any]] | None = None,\n'
                b'    consumed_indexes: set[int] | None = None,\n'
                b'    company_input_contract: str = "",\n'
                b') -> list[dict[str, Any]]:\n'
                b'    """Bind immutable single-company decisions to this list\'s verified identities.\n'
            ),
            (
                b'    raw_breakdowns: Sequence[Mapping[str, Any]],\n'
                b'    *, contacts_required: bool = False, company_quality: bool = True,\n'
                b'    scored_indexes: Sequence[int] | None = None,\n'
                b'    resolve_raw: Callable[[int], Mapping[str, Any] | None] | None = None,\n'
                b'    consumed_indexes: set[int] | None = None,\n'
                b'    deferred_indexes: set[int] | None = None,\n'
                b'    company_input_contract: str = "",\n'
                b') -> list[dict[str, Any]]:\n'
                b'    """Bind immutable single-company decisions to this list\'s verified identities.\n'
            ),
        ),
        (
            (
                b'    if not company_quality:\n'
                b'        return apply_legacy_company_judgment_context(\n'
                b'            companies, raw_breakdowns, scored_indexes=scored_indexes, resolve_raw=resolve_raw,\n'
                b'            consumed_indexes=consumed_indexes,\n'
                b'            company_input_contract=company_input_contract,\n'
                b'        )\n'
                b'    from qualification.company_quality import normalize_company_claim\n'
            ),
            (
                b'    if not company_quality:\n'
                b'        return apply_legacy_company_judgment_context(\n'
                b'            companies, raw_breakdowns, scored_indexes=scored_indexes, resolve_raw=resolve_raw,\n'
                b'            consumed_indexes=consumed_indexes, deferred_indexes=deferred_indexes,\n'
                b'            company_input_contract=company_input_contract,\n'
                b'        )\n'
                b'    from qualification.company_quality import normalize_company_claim\n'
            ),
        ),
    ),
}


class LegacyCompanyProgressSeamError(ValueError):
    """The pair differs from the reviewed complete reference generation."""


def convert_legacy_company_progress_files(sources: dict[str, bytes]) -> dict[str, bytes]:
    if set(sources) != set(FILES):
        raise LegacyCompanyProgressSeamError("legacy company progress requires exactly two modules")
    hashes = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
    if hashes == REPAIRED_SHA256:
        return dict(sources)
    if hashes != REFERENCE_SHA256:
        raise LegacyCompanyProgressSeamError("unrecognized or partial legacy company progress source")
    repaired = {}
    for name, source in sources.items():
        for before, after in _CHANGES[name]:
            if source.count(before) != 1:
                raise LegacyCompanyProgressSeamError("legacy company progress anchor differs")
            source = source.replace(before, after, 1)
        compile(source, name, "exec")
        repaired[name] = source
    if {name: hashlib.sha256(source).hexdigest() for name, source in repaired.items()} != REPAIRED_SHA256:
        raise LegacyCompanyProgressSeamError("legacy company progress differs from reviewed source")
    return repaired


# The frozen50de image already includes the repaired51bc pair above.
# Reuse its existing ordered projection and move only its local retry loop.
ORDERED_RETRY_CORRECTION_ID = "legacy_company_ordered_retry_passes_v1"
ORDERED_RETRY_SOURCE_COMMIT = "578180d01e9122b12359fa65ea682d5011100366"
ORDERED_RETRY_REFERENCE_SHA256 = dict(REPAIRED_SHA256)
ORDERED_RETRY_REPAIRED_SHA256 = {
    SCORING_FILE: "489dfcc7d87ddcbbf70781d2da7e52c9f37e560d33c79ecac18fbafb1a5e0cea",
    COMPETITION_FILE: REPAIRED_SHA256[COMPETITION_FILE],
}
_ORDERED_RETRY_DELTAS = (
    (
        (
            b'    _original_company_indexes: Optional[Sequence[int]] = None,\n'
            b'    prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,\n'
            b'    on_terminal_breakdowns: Optional[Callable[[List[Dict[str, Any]]], None]] = None,\n'
            b') -> List[Dict[str, Any]]:\n'
            b'    """Score one distinct output once: the first-N slice in model order.\n'
            b'\n'
        ),
        (
            b'    _original_company_indexes: Optional[Sequence[int]] = None,\n'
            b'    prior_terminal_breakdowns: Optional[Sequence[Mapping[str, Any]]] = None,\n'
            b'    on_terminal_breakdowns: Optional[Callable[[List[Dict[str, Any]]], None]] = None,\n'
            b'    _retry_evidence_scope: Optional[Dict[str, Any]] = None,\n'
            b'    _single_attempt_index: Optional[int] = None,\n'
            b') -> List[Dict[str, Any]]:\n'
            b'    """Score one distinct output once: the first-N slice in model order.\n'
            b'\n'
        ),
    ),
    (
        (
            b'        checkpoint()\n'
            b'    last_failure_reason = ""\n'
            b'    last_private_diagnostic = "origin=exception"\n'
            b'    # This private dictionary is the complete lifetime boundary for source text\n'
            b'    # retained between retries. A later score_work_item call always gets a new\n'
            b'    # scope, even when its caller reuses the same scorer object.\n'
            b'    retry_evidence_scope: Dict[str, Any] = {}\n'
            b'    attempts = max(1, int(max_retries))\n'
            b'    for attempt in range(attempts):\n'
            b'        if retain_terminal and attempt > 0 and not checkpoint_mode:\n'
            b'            invoked_positions = list(unresolved)\n'
            b'            invoked_companies = [\n'
        ),
        (
            b'        checkpoint()\n'
            b'    last_failure_reason = ""\n'
            b'    last_private_diagnostic = "origin=exception"\n'
            b'    # The intrinsic-company caller shares this same private scope across its\n'
            b'    # ordered passes. Every new scoring work item still starts with a new map.\n'
            b'    retry_evidence_scope = {} if _retry_evidence_scope is None else _retry_evidence_scope\n'
            b'    attempts = max(1, int(max_retries))\n'
            b'    if _single_attempt_index is not None and (\n'
            b'        type(_single_attempt_index) is not int or not 0 <= _single_attempt_index < attempts\n'
            b'    ):\n'
            b'        raise ScoringError("single company attempt index is invalid")\n'
            b'    attempt_indexes = (range(attempts) if _single_attempt_index is None\n'
            b'                       else (_single_attempt_index,))\n'
            b'    for attempt in attempt_indexes:\n'
            b'        if retain_terminal and attempt > 0 and not checkpoint_mode:\n'
            b'            invoked_positions = list(unresolved)\n'
            b'            invoked_companies = [\n'
        ),
    ),
    (
        (
            b'            continue\n'
            b'    raise _JudgeExhausted(\n'
            b'        ("judge_exhausted attempts=%d reason=%s %s"\n'
            b'         % (attempts, last_failure_reason or "unknown", last_private_diagnostic))[\n'
            b'             :MAX_FAILURE_DETAIL_CHARS\n'
            b'         ],\n'
            b'        failure_reason=last_failure_reason,\n'
        ),
        (
            b'            continue\n'
            b'    raise _JudgeExhausted(\n'
            b'        ("judge_exhausted attempts=%d reason=%s %s"\n'
            b'         % (attempt + 1, last_failure_reason or "unknown", last_private_diagnostic))[\n'
            b'             :MAX_FAILURE_DETAIL_CHARS\n'
            b'         ],\n'
            b'        failure_reason=last_failure_reason,\n'
        ),
    ),
    (
        (
            b'    new_judgments = []\n'
            b'    failures_by_key = {}\n'
            b'    deferred_indexes = set()\n'
            b'    def resolve(index):\n'
            b'        ref = refs[index]\n'
            b'        key = ref["cache_key"]\n'
        ),
        (
            b'    new_judgments = []\n'
            b'    failures_by_key = {}\n'
            b'    deferred_indexes = set()\n'
            b'    retry_evidence_scope: Dict[str, Any] = {}\n'
            b'    attempts = max(1, int(MAX_JUDGE_RETRIES))\n'
            b'    def resolve(index):\n'
            b'        ref = refs[index]\n'
            b'        key = ref["cache_key"]\n'
        ),
    ),
    (
        (
            b'                  if signal_exchange is not None and enabled() else nullcontext()):\n'
            b'                try:\n'
            b'                    rows = score_work_item(item, icp=icp, companies=[sliced[index]], scorer=scorer,\n'
            b'                                           _original_company_indexes=[index])\n'
            b'                except _JudgeExhausted as exc:\n'
            b'                    # Keep the ordered pre-provider gates, while allowing later\n'
            b'                    # independent companies to checkpoint their real judgments.\n'
            b'                    failures_by_key[key] = exc\n'
            b'                    deferred_indexes.add(index)\n'
            b'                    return None\n'
        ),
        (
            b'                  if signal_exchange is not None and enabled() else nullcontext()):\n'
            b'                try:\n'
            b'                    rows = score_work_item(item, icp=icp, companies=[sliced[index]], scorer=scorer,\n'
            b'                                           _original_company_indexes=[index],\n'
            b'                                           max_retries=attempts,\n'
            b'                                           _retry_evidence_scope=retry_evidence_scope,\n'
            b'                                           _single_attempt_index=attempt)\n'
            b'                except _JudgeExhausted as exc:\n'
            b'                    # Finish this ordered pass before the next existing attempt.\n'
            b'                    failures_by_key[key] = exc\n'
            b'                    deferred_indexes.add(index)\n'
            b'                    return None\n'
        ),
    ),
    (
        (
            b'                on_company_judgments(validated)\n'
            b'        return raw_by_key[key]\n'
            b'    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)\n'
            b'    full = apply_company_judgment_context(sliced, [None] * len(sliced), company_quality=False,\n'
            b'                                        scored_indexes=scored, resolve_raw=resolve,\n'
            b'                                        deferred_indexes=deferred_indexes,\n'
            b'                                        company_input_contract=getattr(scorer, "company_input_contract", ""))\n'
            b'    if failures_by_key:\n'
            b'        raise next(iter(failures_by_key.values()))\n'
            b'    validate_breakdowns_for_item(full, icp=icp, companies=sliced,\n'
        ),
        (
            b'                on_company_judgments(validated)\n'
            b'        return raw_by_key[key]\n'
            b'    scored, _ = verify.bucket_skip(icp, sliced, max_scored_companies=max_scored_companies)\n'
            b'    for attempt in range(attempts):\n'
            b'        failures_by_key.clear()\n'
            b'        deferred_indexes.clear()\n'
            b'        # Rebuild aliases in original order from intrinsic judgments. A newly\n'
            b'        # recovered earlier company can make a completed later row a duplicate;\n'
            b'        # its saved raw authority must keep its original substantive verdict.\n'
            b'        full = apply_company_judgment_context(sliced, [None] * len(sliced), company_quality=False,\n'
            b'                                            scored_indexes=scored, resolve_raw=resolve,\n'
            b'                                            deferred_indexes=deferred_indexes,\n'
            b'                                            company_input_contract=getattr(scorer, "company_input_contract", ""))\n'
            b'        if not failures_by_key:\n'
            b'            break\n'
            b'    if failures_by_key:\n'
            b'        raise next(iter(failures_by_key.values()))\n'
            b'    validate_breakdowns_for_item(full, icp=icp, companies=sliced,\n'
        ),
    ),
)


def convert_ordered_company_retry_passes(sources: dict[str, bytes]) -> dict[str, bytes]:
    """Validate the complete frozen pair and change only ordered retry passes."""
    if set(sources) != set(FILES):
        raise LegacyCompanyProgressSeamError("ordered retry passes require exactly two modules")
    hashes = {name: hashlib.sha256(source).hexdigest() for name, source in sources.items()}
    if hashes == ORDERED_RETRY_REPAIRED_SHA256:
        return dict(sources)
    if hashes != ORDERED_RETRY_REFERENCE_SHA256:
        raise LegacyCompanyProgressSeamError("unrecognized or partial ordered retry source")
    source = sources[SCORING_FILE]
    for before, after in _ORDERED_RETRY_DELTAS:
        if source.count(before) != 1:
            raise LegacyCompanyProgressSeamError("ordered retry anchor differs")
        source = source.replace(before, after, 1)
    compile(source, SCORING_FILE, "exec")
    repaired = {**sources, SCORING_FILE: source}
    if {name: hashlib.sha256(data).hexdigest() for name, data in repaired.items()} != ORDERED_RETRY_REPAIRED_SHA256:
        raise LegacyCompanyProgressSeamError("ordered retry postimage differs")
    return repaired

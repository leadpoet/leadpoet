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

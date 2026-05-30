"""Reference-model baseline persistence + champion-selection floor.

The qualification "champion" is the model the validator pays 10% emissions
to. For miners to dethrone the champion, the CTO's rule is: beat the
DAILY-RUN REFERENCE MODEL'S score by `CHAMPION_DETHRONING_THRESHOLD_POINTS`
(currently 10). The reference model lives at
``miner_models/qualification_model/`` and is exempt from the cost
variability penalty (see ``score_company(is_reference_model=True)`` in
``lead_scorer.py``).

This module owns:

  * The reserved ``REFERENCE_MODEL_ID`` constant used to mark the daily
    reference run in `qualification_baseline.json`. Champion selection
    must filter this ID out of the challenger pool — the reference model
    is the FLOOR, not a competitor.

  * ``BaselineRecord`` namedtuple — what we persist for the current set.

  * ``load_baseline(set_id)`` / ``save_baseline(record)`` — file-backed
    persistence under ``validator_weights/qualification_baseline.json``.
    Mirrors the existing ``qualification_champion.json`` pattern.

  * ``run_and_save_baseline(set_id, icp_set, qualify_fn, score_fn)`` — the
    actual daily-run entry point. Async. Dependencies are injected so the
    function can be unit-tested without importing the real reference
    model (Exa + OpenRouter + ScrapingDog calls). Production callers pass
    the real ``miner_models.qualification_model.qualify`` and
    ``qualification.scoring.lead_scorer.score_company``.

What this module does NOT do:

  * Schedule the daily run. That trigger lives in ``neurons/validator.py``
    next to the existing champion-rebenchmark dispatch at line ~6480.
    Once the CTO is ready to wire it, the validator's daily check should
    additionally call ``run_and_save_baseline(...)`` if today's record
    isn't already present.

  * Modify champion-selection arithmetic. That happens in
    ``qualification/scoring/champion.py``, which imports
    ``load_baseline`` to gate the dethroning threshold and the
    no-current-champion case.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


# Reserved model_id for the daily reference run. The validator must NEVER
# treat a record with this ID as a miner submission — champion selection
# filters it out of the challenger pool. The version suffix is included so
# we can bump it if the reference pipeline changes materially (e.g., a
# different qualify entry point) without polluting old baseline records.
REFERENCE_MODEL_ID: str = "reference:qualification_model:v1"

# File name (under ``validator_weights/``) we persist the daily record to.
# Same directory as ``qualification_champion.json`` for consistency.
BASELINE_FILE_NAME: str = "qualification_baseline.json"


class BaselineRecord(NamedTuple):
    """What we persist for a daily reference-model run.

    ``baseline_score`` is the aggregate the champion-selection logic
    compares against — the same scale as challenger ``total_score``
    (sum of per-ICP scores, where each per-ICP score is
    ``sum(score_company over up-to-5 leads) / 5`` per the v2 normalization
    in ``gateway/qualification/config.py``).

    ``per_icp_scores`` is kept for traceability — operators inspecting the
    daily run can see which ICPs the reference model bombed on.
    """

    set_id: int
    baseline_score: float
    per_icp_scores: List[float]
    scored_at: str
    model_id: str


# =============================================================================
# Persistence
# =============================================================================


def _baseline_path() -> Path:
    """Return the absolute path to the on-disk baseline record.

    Located alongside ``qualification_champion.json`` under
    ``<repo_root>/validator_weights/``. The directory is created lazily
    on first save.
    """
    # Anchor to repo root, not cwd — validator may be launched from a
    # different working directory (Docker entrypoint, systemd unit, etc.)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "validator_weights" / BASELINE_FILE_NAME


def load_baseline(set_id: int) -> Optional[BaselineRecord]:
    """Read the baseline record for ``set_id`` from disk.

    Returns ``None`` when:
      * the file doesn't exist (no daily run has happened yet)
      * the stored set_id doesn't match (today's ICP set hasn't been
        baselined yet — record is for an older set)
      * the file is malformed (corrupt JSON / missing fields)

    Callers (champion selection) treat ``None`` as "no baseline floor
    applies; fall through to legacy champion-only threshold logic." This
    keeps the system safe during the rollout window before the daily
    runner is wired up.
    """
    path = _baseline_path()
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"baseline file unreadable: {e}; treating as missing")
        return None
    if data.get("set_id") != set_id:
        # Common case during transition (e.g., set just rotated and the
        # daily run hasn't fired yet). Don't warn — too noisy.
        return None
    try:
        return BaselineRecord(
            set_id=int(data["set_id"]),
            baseline_score=float(data["baseline_score"]),
            per_icp_scores=list(data.get("per_icp_scores") or []),
            scored_at=str(data.get("scored_at") or ""),
            model_id=str(data.get("model_id") or REFERENCE_MODEL_ID),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"baseline file has malformed record: {e}; treating as missing")
        return None


def save_baseline(record: BaselineRecord) -> None:
    """Atomically persist ``record`` to ``qualification_baseline.json``.

    Atomicity is achieved via write-tmp-then-rename — readers (champion
    selection) see either the old record or the new one, never a
    half-written file.
    """
    path = _baseline_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({
            "set_id": record.set_id,
            "baseline_score": record.baseline_score,
            "per_icp_scores": record.per_icp_scores,
            "scored_at": record.scored_at,
            "model_id": record.model_id,
        }, f, indent=2)
    tmp.replace(path)


def is_reference_model_id(model_id: Optional[str]) -> bool:
    """Predicate used by champion selection to drop reference-model
    records from the challenger pool.

    Conservative match — only the exact reserved ID (with version) is
    treated as the reference, so a malicious miner cannot submit a model
    with a clever-looking name and have it silently excluded from
    competition.
    """
    return model_id == REFERENCE_MODEL_ID


# =============================================================================
# Supabase-backed persistence (production path)
#
# The gateway-side daily runner WRITES to `qualification_baselines`; the
# validator-side champion-selection READS from it. File-backed persistence
# above is kept for local development / unit tests, but production goes
# through these functions.
# =============================================================================

DB_TABLE_NAME: str = "qualification_baselines"


def save_baseline_to_db(
    record: BaselineRecord,
    supabase_client,
    *,
    icp_set_hash: Optional[str] = None,
    run_duration_seconds: Optional[float] = None,
    run_status: str = "completed",
) -> None:
    """Upsert ``record`` into ``qualification_baselines``.

    Upsert keyed on ``set_id`` (the table's primary key). Idempotent on
    re-run — re-running the daily baseline for the same set_id overwrites
    the prior row, which is the desired behavior if we ever need to
    re-evaluate the reference model mid-day.

    Raises whatever the supabase client raises on failure. The caller
    (gateway runner) wraps the call in try/except and logs — a failed
    baseline persistence leaves the DB with no row for today, and
    champion-selection falls back to legacy champion-only thresholding
    (safe degradation).
    """
    payload = {
        "set_id": record.set_id,
        "baseline_score": record.baseline_score,
        "per_icp_scores": record.per_icp_scores,
        "scored_at": record.scored_at,
        "model_id": record.model_id,
        "run_status": run_status,
    }
    if icp_set_hash is not None:
        payload["icp_set_hash"] = icp_set_hash
    if run_duration_seconds is not None:
        payload["run_duration_seconds"] = run_duration_seconds
    supabase_client.table(DB_TABLE_NAME).upsert(payload, on_conflict="set_id").execute()


def load_baseline_from_db(set_id: int, supabase_client) -> Optional[BaselineRecord]:
    """Read baseline row for ``set_id``.

    Returns ``None`` when:
      * no row exists for set_id (no daily run has fired yet today)
      * the row's ``run_status`` is anything other than ``completed``
        (treat partial/failed runs as no-baseline so we don't gate
        champion selection on incomplete data)
      * any DB error (transient network blip, etc.) — caller logs and
        falls back to legacy thresholding

    Mirrors ``load_baseline`` (file-based) on the return contract so
    champion-selection can swap which one it uses without touching the
    consumer side.
    """
    try:
        result = (
            supabase_client.table(DB_TABLE_NAME)
            .select("set_id, baseline_score, per_icp_scores, scored_at, model_id, run_status")
            .eq("set_id", set_id)
            .limit(1)
            .execute()
        )
    except Exception as e:
        logger.warning(f"baseline DB read failed for set_id={set_id}: {e}; treating as missing")
        return None

    rows = result.data or []
    if not rows:
        return None

    row = rows[0]
    if row.get("run_status") != "completed":
        logger.info(
            f"baseline row for set_id={set_id} has run_status="
            f"{row.get('run_status')!r}; treating as no-baseline"
        )
        return None

    try:
        return BaselineRecord(
            set_id=int(row["set_id"]),
            baseline_score=float(row["baseline_score"]),
            per_icp_scores=list(row.get("per_icp_scores") or []),
            scored_at=str(row.get("scored_at") or ""),
            model_id=str(row.get("model_id") or REFERENCE_MODEL_ID),
        )
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"baseline DB row for set_id={set_id} is malformed: {e}; treating as missing")
        return None


# =============================================================================
# Daily reference-model evaluation
# =============================================================================

# Type aliases for the injectable callables. Real callers pass:
#   qualify_fn = miner_models.qualification_model.qualify   (sync; uses asyncio.run internally)
#   score_fn   = qualification.scoring.lead_scorer.score_company   (async)
QualifyFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]
ScoreFn = Callable[..., Awaitable[Any]]


async def run_and_save_baseline(
    set_id: int,
    icp_set: List[Dict[str, Any]],
    qualify_fn: QualifyFn,
    score_fn: ScoreFn,
    leads_per_icp_normalizer: float = 5.0,
) -> BaselineRecord:
    """Run the reference model on every ICP in ``icp_set`` and persist a
    ``BaselineRecord`` matching the v2 scoring shape.

    Why this is the validator's responsibility (not the qualification
    queue's): the reference model needs direct outbound network to Exa,
    OpenRouter, and ScrapingDog. The qualification worker sandbox blocks
    Exa entirely (``qualification/validator/local_proxy.py:PAID_PROVIDERS``
    doesn't include exa.ai). Running it in the validator main process
    bypasses the sandbox and gives the model the same tools it would
    have in development.

    Aggregation matches the v2 model-competition scoring used by miner
    challengers: ``per_icp_score = sum(score_company over up-to-5 leads) /
    leads_per_icp_normalizer`` (5 by default), then we sum across the 20
    ICPs to get the comparable ``baseline_score``. This is the same
    arithmetic ``get_model_score`` returns for miners, so champion
    selection compares apples to apples.

    Failure isolation: any single ICP that crashes contributes 0.0 to
    that ICP's score rather than aborting the whole run. Same policy the
    qualification model itself uses (``qualify(icp)`` returns ``[]`` on
    failure rather than raising).
    """
    per_icp_scores: List[float] = []
    for icp in icp_set:
        icp_id = icp.get("icp_id") or icp.get("industry") or "?"
        try:
            # qualify is sync (asyncio.run internally) — run in a thread so
            # we don't block the validator's event loop.
            leads = await asyncio.to_thread(qualify_fn, icp)
        except Exception as e:
            logger.warning(f"reference qualify crashed on icp={icp_id}: {e}; scoring 0")
            per_icp_scores.append(0.0)
            continue

        if not isinstance(leads, list) or not leads:
            # Honest abstention is fine — contributes 0 to this ICP's score.
            per_icp_scores.append(0.0)
            continue

        icp_total = 0.0
        for lead in leads:
            try:
                result = await score_fn(
                    company=lead,
                    icp=icp,
                    is_reference_model=True,
                )
                # score_company returns either a (score, breakdown) tuple
                # or a Score-shaped object depending on version; accept both.
                score = (
                    result[0] if isinstance(result, tuple)
                    else getattr(result, "total_score", result)
                )
                icp_total += float(score or 0.0)
            except Exception as e:
                logger.warning(
                    f"reference score_company crashed on icp={icp_id} "
                    f"lead={lead.get('company') or '?'}: {e}; lead contributes 0"
                )
                continue

        per_icp_score = icp_total / leads_per_icp_normalizer if leads_per_icp_normalizer else icp_total
        per_icp_scores.append(per_icp_score)

    baseline_score = sum(per_icp_scores)
    record = BaselineRecord(
        set_id=set_id,
        baseline_score=baseline_score,
        per_icp_scores=per_icp_scores,
        scored_at=datetime.now(timezone.utc).isoformat(),
        model_id=REFERENCE_MODEL_ID,
    )
    save_baseline(record)
    logger.info(
        f"baseline run complete: set_id={set_id} score={baseline_score:.2f} "
        f"per_icp={['%.2f' % s for s in per_icp_scores]}"
    )
    return record

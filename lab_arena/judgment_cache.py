"""Content-addressed reuse of accepted Arena judge evidence.

Only the gateway constructs cache identities.  A validator supplies a normal
scoring output for its leased run; the database may freeze that accepted
output under the identity already attached to the run, but a validator cannot
name a cache entry or turn a failure into one.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence

from lab_arena import contracts


INTEGRITY_POLICY = "arena_integrity_v1"
CACHE_SCOPE_SCHEMA_VERSION = "leadpoet.lab_arena.judgment_cache_scope.v1"
EVIDENCE_SCHEMA_VERSION = "leadpoet.lab_arena.judgment_evidence.v1"
AUTHORITY_PARTITION_SCHEMA_VERSION = (
    "leadpoet.lab_arena.judgment_authority_partition.v1"
)

_SCORING_INPUT_FIELDS = (
    "schema_version",
    "scored_run_id",
    "icp",
    "companies",
    "scorer_policy",
    "evaluation_date",
)


class JudgmentCacheError(ValueError):
    """A cache identity or frozen evidence record is malformed."""


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(contracts.canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise JudgmentCacheError("judgment cache value is not canonical JSON") from exc


def effective_scoring_input(document: Mapping[str, Any]) -> Dict[str, Any]:
    """Hash the adapter's effective input, excluding unused submitted fields."""

    if not isinstance(document, Mapping) or tuple(document.keys()) != _SCORING_INPUT_FIELDS:
        raise JudgmentCacheError("scoring input fields or field order changed")
    if document.get("schema_version") != "leadpoet.lab_arena.scoring_input.v1":
        raise JudgmentCacheError("scoring input schema changed")
    copied = _json_copy(document)
    del copied["scored_run_id"]
    from qualification.scoring.competition import effective_competition_input

    copied.update(effective_competition_input(copied["companies"], copied["icp"]))
    return copied


def build_cache_scope(
    *,
    scoring_input: Mapping[str, Any],
    round_id: str,
    network_name: str,
    netuid: int,
    scorer_image_digest: str,
    scorer_image_reference: str,
    integrity_policy: str,
) -> Dict[str, Any]:
    """Build the gateway-owned cache identity for one effective judge input."""

    if integrity_policy != INTEGRITY_POLICY:
        raise JudgmentCacheError("unsupported judgment integrity policy")
    effective = effective_scoring_input(scoring_input)
    input_hash = contracts.document_hash(effective)
    scope = {
        "schema_version": CACHE_SCOPE_SCHEMA_VERSION,
        "integrity_policy": integrity_policy,
        "round_id": str(round_id),
        "network_name": str(network_name),
        "netuid": int(netuid),
        "scorer_image_digest": str(scorer_image_digest),
        "scorer_image_reference": str(scorer_image_reference),
        "evaluation_date": str(effective["evaluation_date"]),
        "scoring_input_hash": input_hash,
    }
    if not scope["round_id"] or not scope["network_name"] or scope["netuid"] < 0:
        raise JudgmentCacheError("judgment cache scope is incomplete")
    if not scope["scorer_image_digest"] or not scope["scorer_image_reference"]:
        raise JudgmentCacheError("scorer image identity is incomplete")
    return {**scope, "cache_key": contracts.document_hash(scope)}


def partition_cache_scope(
    cache_scope: Mapping[str, Any], *, incompatible_hotkeys: list[str]
) -> Dict[str, Any]:
    """Derive a distinct cache identity when an earlier source is ineligible.

    The partition label contains only a hash of the finalized same-coldkey
    family that made the earlier source unusable.  Sorting makes the label
    independent of chain response order.  Repeating a partition would create
    an authority loop, so it is refused instead of risking the earlier result.
    """

    scope = _json_copy(cache_scope)
    cache_key = scope.pop("cache_key", None)
    if cache_key != contracts.document_hash(scope):
        raise JudgmentCacheError("judgment cache scope hash mismatch")
    hotkeys = sorted(set(incompatible_hotkeys))
    if not hotkeys or any(not isinstance(hotkey, str) or not hotkey for hotkey in hotkeys):
        raise JudgmentCacheError("judgment authority partition is incomplete")
    partition = contracts.document_hash({
        "schema_version": AUTHORITY_PARTITION_SCHEMA_VERSION,
        "incompatible_hotkeys": hotkeys,
    })
    partitions = scope.get("authority_partitions", [])
    if (
        not isinstance(partitions, list)
        or any(not isinstance(item, str) for item in partitions)
        or partitions != sorted(set(partitions))
        or partition in partitions
    ):
        raise JudgmentCacheError("judgment authority partition loop")
    scope["authority_partitions"] = sorted(partitions + [partition])
    return {**scope, "cache_key": contracts.document_hash(scope)}


def build_evidence_snapshot(
    *,
    output: Mapping[str, Any],
    cache_scope: Mapping[str, Any],
    source_score_run_id: str,
    source_scored_run_id: str,
    source_output_ref: str,
    source_runner_hotkey: str,
    runner_authority_exclusions: Sequence[str] | None,
) -> Dict[str, Any]:
    """Freeze accepted judge evidence independently of its object-store path."""

    if not isinstance(output, Mapping) or set(output) != {
        "schema_version", "scored_run_id", "breakdowns"
    }:
        raise JudgmentCacheError("only accepted scoring output can be cached")
    if output.get("scored_run_id") != source_scored_run_id:
        raise JudgmentCacheError("scoring output is bound to another execution")
    scope = _json_copy(cache_scope)
    cache_key = scope.pop("cache_key", None)
    if cache_key != contracts.document_hash(scope):
        raise JudgmentCacheError("judgment cache scope hash mismatch")
    if (
        not isinstance(runner_authority_exclusions, Sequence)
        or isinstance(runner_authority_exclusions, (str, bytes))
        or any(
            not isinstance(hotkey, str) or not hotkey
            for hotkey in runner_authority_exclusions
        )
    ):
        raise JudgmentCacheError("judgment authority provenance is incomplete")
    exclusions = sorted(set(runner_authority_exclusions))
    if not exclusions or source_runner_hotkey not in exclusions:
        raise JudgmentCacheError("judgment authority provenance is incomplete")
    snapshot = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "cache_key": cache_key,
        "scoring_input_hash": scope.get("scoring_input_hash"),
        "source_score_run_id": str(source_score_run_id),
        "source_scored_run_id": str(source_scored_run_id),
        "source_output_ref": str(source_output_ref),
        "source_output_hash": contracts.document_hash(output),
        "source_runner_hotkey": str(source_runner_hotkey),
        "runner_authority_exclusions": exclusions,
        "scoring_output_schema_version": output.get("schema_version"),
        "breakdowns": _json_copy(output["breakdowns"]),
    }
    if any(not snapshot[key] for key in (
        "cache_key", "scoring_input_hash", "source_score_run_id",
        "source_scored_run_id", "source_output_ref", "source_runner_hotkey",
    )):
        raise JudgmentCacheError("judgment evidence provenance is incomplete")
    return snapshot


def validate_evidence_snapshot(
    document: Mapping[str, Any],
    *,
    cache_key: str,
    evidence_hash: str,
) -> Dict[str, Any]:
    """Validate a database cache row before its evidence is used for scoring."""

    if not isinstance(document, Mapping) or set(document) != {
        "schema_version", "cache_key", "scoring_input_hash",
        "source_score_run_id", "source_scored_run_id", "source_output_ref",
        "source_output_hash", "source_runner_hotkey",
        "runner_authority_exclusions",
        "scoring_output_schema_version", "breakdowns",
    }:
        raise JudgmentCacheError("judgment evidence fields changed")
    copied = _json_copy(document)
    if copied["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise JudgmentCacheError("judgment evidence schema changed")
    if copied["cache_key"] != cache_key:
        raise JudgmentCacheError("judgment evidence cache key mismatch")
    if contracts.document_hash(copied) != evidence_hash:
        raise JudgmentCacheError("judgment evidence hash mismatch")
    if copied["scoring_output_schema_version"] != "leadpoet.lab_arena.scoring_output.v1":
        raise JudgmentCacheError("judgment evidence output schema changed")
    exclusions = copied["runner_authority_exclusions"]
    if (
        not isinstance(exclusions, list)
        or any(not isinstance(hotkey, str) or not hotkey for hotkey in exclusions)
        or exclusions != sorted(set(exclusions))
        or copied["source_runner_hotkey"] not in exclusions
    ):
        raise JudgmentCacheError("judgment authority provenance is incomplete")
    return copied

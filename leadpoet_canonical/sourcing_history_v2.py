"""Canonical, dependency-free sourcing history records for V2.

The arithmetic is a direct extraction of ``Validator.accumulate_miner_weights``
and ``Validator.get_rolling_epoch_scores``.  It changes no score, multiplier,
adjustment, approval, or rolling-window rule; it only gives those existing
values a canonical form that can be signed and independently replayed.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

from leadpoet_canonical.attested_v2 import sha256_json


SOURCING_DECISION_SCHEMA_VERSION = "leadpoet.sourcing_decision.v2"
SOURCING_EPOCH_SCHEMA_VERSION = "leadpoet.sourcing_epoch.v2"
SOURCING_ROLLING_WINDOW = 30
LEGACY_ICP_MULTIPLIERS = frozenset({1.0, 1.5, 5.0})

_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourcingHistoryV2Error(ValueError):
    """A sourcing decision or aggregate is non-canonical."""


def effective_rep_score_v2(
    *, rep_score: Any, is_icp_multiplier: Any, decision: str
) -> Any:
    """Mirror the existing approved-lead score calculation exactly."""

    if str(decision) != "approve":
        return 0
    if isinstance(rep_score, bool) or not isinstance(rep_score, (int, float)):
        raise SourcingHistoryV2Error("rep_score must be numeric")
    if isinstance(is_icp_multiplier, bool) or not isinstance(
        is_icp_multiplier, (int, float)
    ):
        raise SourcingHistoryV2Error("is_icp_multiplier must be numeric")
    if not math.isfinite(float(rep_score)) or not math.isfinite(
        float(is_icp_multiplier)
    ):
        raise SourcingHistoryV2Error("sourcing score inputs must be finite")
    if is_icp_multiplier in LEGACY_ICP_MULTIPLIERS:
        return rep_score * is_icp_multiplier
    return max(0, rep_score + int(is_icp_multiplier))


def build_sourcing_decision_v2(
    *,
    epoch_id: int,
    sequence: int,
    lead_id_hash: str,
    miner_hotkey: str,
    decision: str,
    rep_score: Any,
    is_icp_multiplier: Any,
) -> Dict[str, Any]:
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
        raise SourcingHistoryV2Error("epoch_id is invalid")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0:
        raise SourcingHistoryV2Error("sequence is invalid")
    if not _HASH_RE.fullmatch(str(lead_id_hash or "")):
        raise SourcingHistoryV2Error("lead_id_hash is invalid")
    hotkey = str(miner_hotkey or "").strip()
    if not hotkey or len(hotkey) > 128 or any(char.isspace() for char in hotkey):
        raise SourcingHistoryV2Error("miner_hotkey is invalid")
    normalized_decision = str(decision or "")
    if normalized_decision not in {"approve", "deny"}:
        raise SourcingHistoryV2Error("decision is invalid")
    effective = effective_rep_score_v2(
        rep_score=rep_score,
        is_icp_multiplier=is_icp_multiplier,
        decision=normalized_decision,
    )
    body = {
        "schema_version": SOURCING_DECISION_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "sequence": sequence,
        "lead_id_hash": lead_id_hash,
        "miner_hotkey": hotkey,
        "decision": normalized_decision,
        "rep_score": rep_score,
        "is_icp_multiplier": is_icp_multiplier,
        "effective_rep_score": effective,
        "approved_lead_count": 1 if normalized_decision == "approve" else 0,
    }
    return {**body, "decision_hash": sha256_json(body)}


def validate_sourcing_decision_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "epoch_id",
        "sequence",
        "lead_id_hash",
        "miner_hotkey",
        "decision",
        "rep_score",
        "is_icp_multiplier",
        "effective_rep_score",
        "approved_lead_count",
        "decision_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourcingHistoryV2Error("sourcing decision fields are invalid")
    rebuilt = build_sourcing_decision_v2(
        epoch_id=value["epoch_id"],
        sequence=value["sequence"],
        lead_id_hash=value["lead_id_hash"],
        miner_hotkey=value["miner_hotkey"],
        decision=value["decision"],
        rep_score=value["rep_score"],
        is_icp_multiplier=value["is_icp_multiplier"],
    )
    if dict(value) != rebuilt:
        raise SourcingHistoryV2Error("sourcing decision is not canonical")
    return rebuilt


def build_sourcing_epoch_v2(
    *, epoch_id: int, decisions: Sequence[Mapping[str, Any]]
) -> Dict[str, Any]:
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
        raise SourcingHistoryV2Error("epoch_id is invalid")
    normalized = [validate_sourcing_decision_v2(item) for item in decisions]
    if any(item["epoch_id"] != epoch_id for item in normalized):
        raise SourcingHistoryV2Error("sourcing decision epoch differs")
    sequences = [item["sequence"] for item in normalized]
    if len(sequences) != len(set(sequences)):
        raise SourcingHistoryV2Error("sourcing decision sequence is duplicated")
    ordered = sorted(normalized, key=lambda item: item["sequence"])
    # Ordered decisions reproduce the insertion order of the legacy
    # ``miner_scores`` dict written by accumulate_miner_weights().
    miner_scores = {}  # type: Dict[str, float]
    approved_count = 0
    for item in ordered:
        if item["decision"] != "approve":
            continue
        hotkey = item["miner_hotkey"]
        miner_scores[hotkey] = miner_scores.get(hotkey, 0) + item[
            "effective_rep_score"
        ]
        approved_count += 1
    body = {
        "schema_version": SOURCING_EPOCH_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "miner_scores": [
            {"hotkey": hotkey, "score": score}
            for hotkey, score in miner_scores.items()
        ],
        "approved_lead_count": approved_count,
        "decision_count": len(ordered),
        "decision_root": sha256_json(
            [item["decision_hash"] for item in ordered]
        ),
    }
    return {**body, "epoch_hash": sha256_json(body)}


def validate_sourcing_epoch_v2(value: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {
        "schema_version",
        "epoch_id",
        "miner_scores",
        "approved_lead_count",
        "decision_count",
        "decision_root",
        "epoch_hash",
    }
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SourcingHistoryV2Error("sourcing epoch fields are invalid")
    if value.get("schema_version") != SOURCING_EPOCH_SCHEMA_VERSION:
        raise SourcingHistoryV2Error("sourcing epoch schema is invalid")
    epoch_id = value.get("epoch_id")
    if not isinstance(epoch_id, int) or isinstance(epoch_id, bool) or epoch_id < 0:
        raise SourcingHistoryV2Error("sourcing epoch id is invalid")
    rows = value.get("miner_scores")
    if not isinstance(rows, list):
        raise SourcingHistoryV2Error("sourcing epoch scores are invalid")
    normalized_rows = []
    seen = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"hotkey", "score"}:
            raise SourcingHistoryV2Error("sourcing epoch score row is invalid")
        hotkey = str(row["hotkey"] or "")
        score = row["score"]
        if (
            not hotkey
            or hotkey in seen
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) < 0
        ):
            raise SourcingHistoryV2Error("sourcing epoch score row is invalid")
        seen.add(hotkey)
        normalized_rows.append({"hotkey": hotkey, "score": score})
    for field in ("approved_lead_count", "decision_count"):
        if (
            not isinstance(value.get(field), int)
            or isinstance(value.get(field), bool)
            or value[field] < 0
        ):
            raise SourcingHistoryV2Error("%s is invalid" % field)
    if value["approved_lead_count"] > value["decision_count"]:
        raise SourcingHistoryV2Error("approved lead count exceeds decisions")
    if not _HASH_RE.fullmatch(str(value.get("decision_root") or "")):
        raise SourcingHistoryV2Error("decision_root is invalid")
    body = {key: value[key] for key in fields if key != "epoch_hash"}
    if value.get("epoch_hash") != sha256_json(body):
        raise SourcingHistoryV2Error("sourcing epoch hash differs")
    return dict(value)


def rolling_sourcing_history_v2(
    *, current_epoch: int, epochs: Sequence[Mapping[str, Any]], window: int = SOURCING_ROLLING_WINDOW
) -> Tuple[Dict[str, float], int]:
    if not isinstance(current_epoch, int) or current_epoch < 0:
        raise SourcingHistoryV2Error("current_epoch is invalid")
    if not isinstance(window, int) or isinstance(window, bool) or window <= 0:
        raise SourcingHistoryV2Error("window is invalid")
    start_epoch = current_epoch - window
    end_epoch = current_epoch - 1
    scores = {}  # type: Dict[str, float]
    lead_count = 0
    seen_epochs = set()
    for raw in epochs:
        epoch = validate_sourcing_epoch_v2(raw)
        epoch_id = epoch["epoch_id"]
        if epoch_id in seen_epochs:
            raise SourcingHistoryV2Error("sourcing epoch is duplicated")
        seen_epochs.add(epoch_id)
        if not start_epoch <= epoch_id <= end_epoch:
            continue
        for row in epoch["miner_scores"]:
            scores[row["hotkey"]] = scores.get(row["hotkey"], 0) + row["score"]
        lead_count += epoch["approved_lead_count"]
    return scores, lead_count

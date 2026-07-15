"""Fail-closed evaluation routing for Git-tree autoresearch candidates."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from leadpoet_canonical.attested_v2 import sha256_json


TREE_EVALUATION_PLAN_SCHEMA_VERSION = "research_lab.git_tree_eval_plan.v1"

_NETWORK_PATH_RE = re.compile(
    r"(?:^|[/_.-])(?:api|client|http|provider|query|route|search|source|transport)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_NETWORK_CHANGE_RE = re.compile(
    r"\b(?:auth|body|header|keyword|method|param|payload|query|request|route|"
    r"search|term|transport)_[a-z0-9_]+\b|"
    r"\b(?:aiohttp|auth|authorization|body|cursor|data|deepline|delete|endpoint|"
    r"exa|filters?|get|headers?|httpx|keywords?|limit|method|openrouter|pages?|"
    r"params?|patch|payload|post|provider|put|query|requests?|route|scrapingdog|"
    r"search|term|timeout|transport|urllib|urlopen)\b|"
    r"https?://|ClientSession|AsyncClient",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TreeEvaluationPlan:
    mode: str
    reason_codes: tuple[str, ...]
    changed_line_count: int
    target_file_count: int
    patch_hash: str

    def __post_init__(self) -> None:
        if self.mode not in {"replay", "hybrid"}:
            raise ValueError("tree evaluation plan mode is invalid")
        if not self.reason_codes:
            raise ValueError("tree evaluation plan requires a reason")
        if self.changed_line_count < 0 or self.target_file_count < 0:
            raise ValueError("tree evaluation plan counts are invalid")

    def to_dict(self) -> dict[str, Any]:
        document = {
            "schema_version": TREE_EVALUATION_PLAN_SCHEMA_VERSION,
            "mode": self.mode,
            "reason_codes": list(self.reason_codes),
            "changed_line_count": self.changed_line_count,
            "target_file_count": self.target_file_count,
            "patch_hash": self.patch_hash,
        }
        return {**document, "plan_hash": sha256_json(document)}


def classify_tree_evaluation(
    *,
    unified_diff: str,
    target_files: Sequence[str],
) -> TreeEvaluationPlan:
    """Require live discovery for outbound-contract changes or ambiguity.

    Only added/removed source lines are inspected, never private source or
    hidden examples. A malformed or incomplete patch is conservatively hybrid.
    """

    patch = str(unified_diff or "")
    files = tuple(str(item or "").strip() for item in target_files)
    patch_hash = sha256_json({"unified_diff": patch})
    reasons: set[str] = set()
    if not patch.strip() or not files or any(not item for item in files):
        reasons.add("ambiguous_patch_contract")
    if "--- " not in patch or "+++ " not in patch or "@@" not in patch:
        reasons.add("ambiguous_patch_contract")
    if any(_NETWORK_PATH_RE.search(path) for path in files):
        reasons.add("network_contract_file_changed")
    changed_lines = [
        line[1:]
        for line in patch.splitlines()
        if line.startswith(("+", "-"))
        and not line.startswith(("+++", "---"))
    ]
    if any(_NETWORK_CHANGE_RE.search(line) for line in changed_lines):
        reasons.add("outbound_request_contract_changed")
    if reasons:
        mode = "hybrid"
    else:
        mode = "replay"
        reasons.add("no_outbound_contract_change_detected")
    return TreeEvaluationPlan(
        mode=mode,
        reason_codes=tuple(sorted(reasons)),
        changed_line_count=len(changed_lines),
        target_file_count=len(files),
        patch_hash=patch_hash,
    )


def classify_candidate_tree_evaluation(candidate: Any) -> TreeEvaluationPlan:
    draft = getattr(candidate, "draft", None)
    if draft is None:
        return classify_tree_evaluation(unified_diff="", target_files=())
    return classify_tree_evaluation(
        unified_diff=str(getattr(draft, "unified_diff", "") or ""),
        target_files=tuple(getattr(draft, "target_files", ()) or ()),
    )


def evaluation_plan_from_mapping(value: Mapping[str, Any]) -> TreeEvaluationPlan:
    document = dict(value)
    plan_hash = str(document.pop("plan_hash", ""))
    if document.pop("schema_version", "") != TREE_EVALUATION_PLAN_SCHEMA_VERSION:
        raise ValueError("tree evaluation plan schema is invalid")
    plan = TreeEvaluationPlan(
        mode=str(document.get("mode") or ""),
        reason_codes=tuple(document.get("reason_codes") or ()),
        changed_line_count=int(document.get("changed_line_count") or 0),
        target_file_count=int(document.get("target_file_count") or 0),
        patch_hash=str(document.get("patch_hash") or ""),
    )
    if plan.to_dict()["plan_hash"] != plan_hash:
        raise ValueError("tree evaluation plan commitment differs")
    return plan

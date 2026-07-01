"""Code-edit auto-research contracts for candidate private model images."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import posixpath
import re
from typing import Any, Mapping, Sequence

from research_lab.canonical import sha256_json


FORBIDDEN_CODE_EDIT_TERMS = (
    "sk-or-",
    "openrouter_api_key",
    "raw_openrouter_key",
    "raw_secret",
    "service_role",
    "judge_prompt",
    "hidden_benchmark",
    "hidden_icp",
    "icp_plaintext",
    "private_repo",
)

DEFAULT_ALLOWED_PATH_PREFIXES = (
    "gateway/",
    "qualification/",
    "sourcing_model/",
    "validator_models/",
)
DEFAULT_ALLOWED_EXACT_PATHS = (
    "research_lab_adapter.py",
)
DEFAULT_ALLOWED_SUFFIXES = (".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".md")
DISALLOWED_PATH_PATTERNS = (
    r"(^|/)Dockerfile$",
    r"(^|/)docker-compose[^/]*\.ya?ml$",
    r"(^|/)\.github/",
    r"(^|/)\.git/",
    r"(^|/)\.env",
    r"(^|/)requirements[^/]*\.txt$",
    r"(^|/)pyproject\.toml$",
    r"(^|/)poetry\.lock$",
    r"(^|/)uv\.lock$",
    r"(^|/)Pipfile(\.lock)?$",
    r"(^|/)package(-lock)?\.json$",
)
DISALLOWED_DIFF_PATTERNS = (
    r"\bsubprocess\.",
    r"\bos\.system\s*\(",
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"\bsocket\.",
    r"\bparamiko\b",
)


@dataclass(frozen=True)
class CodeEditDraft:
    failure_mode: str
    mechanism: str
    expected_improvement: str
    risk: str
    lane: str
    target_files: tuple[str, ...]
    unified_diff: str
    redacted_summary: str
    test_plan: str
    rollback_plan: str
    predicted_delta: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["target_files"] = list(self.target_files)
        payload["unified_diff_hash"] = sha256_json({"unified_diff": self.unified_diff})
        return payload

    def with_unified_diff(self, unified_diff: str) -> "CodeEditDraft":
        return replace(self, unified_diff=normalize_unified_diff_text(unified_diff))


@dataclass(frozen=True)
class CodeEditSourceInspectionRequest:
    operation: str
    query: str = ""
    path: str = ""
    rationale: str = ""

    def to_event_doc(self) -> dict[str, Any]:
        payload = {
            "operation": self.operation,
            "query_hash": sha256_json({"query": self.query}) if self.query else "",
            "path": self.path,
            "rationale_hash": sha256_json({"rationale": self.rationale}) if self.rationale else "",
        }
        return {key: value for key, value in payload.items() if value not in {"", None}}


def build_code_edit_source_inspection_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    runtime_source_index: Mapping[str, Any],
    source_inspection_context: Mapping[str, Any] | None,
    budget_context: Mapping[str, Any] | None,
    max_requests: int = 4,
) -> list[dict[str, str]]:
    """Ask the model which extracted source files it needs before drafting."""

    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "runtime_source_index": _redacted_source_context(runtime_source_index),
        "source_inspection_context": _redacted_source_context(source_inspection_context or {}),
        "budget_context": _redacted_mapping(budget_context or {}),
        "max_requests": max(1, int(max_requests)),
        "allowed_operations": ["search", "read_file", "finish"],
    }
    system = (
        "You are Leadpoet Research Lab's source-inspection planner for code-edit "
        "autoresearch. You are inspecting the private sourcing model runtime extracted "
        "from the current ECR image. You cannot use external tools or GitHub. Request "
        "only local searches or exact file reads that are necessary to produce a small, "
        "generalizable improvement patch later. Never request secrets, hidden benchmark "
        "plaintext, judge prompts, provider keys, raw private data, or environment files."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Your job in this stage is not to write a patch. Request source context first.\n\n"
        "Allowed request shapes:\n"
        "{\"requests\":[{\"operation\":\"search\",\"query\":\"...\",\"rationale\":\"...\"}]}\n"
        "{\"requests\":[{\"operation\":\"read_file\",\"path\":\"sourcing_model/foo.py\",\"rationale\":\"...\"}]}\n"
        "{\"requests\":[{\"operation\":\"finish\",\"rationale\":\"enough exact source has been read\"}]}\n\n"
        "Rules:\n"
        "- Use search to locate relevant code when the exact path is unclear.\n"
        "- Use read_file before proposing edits to any file.\n"
        "- Only request paths listed in runtime_source_index.editable_files.\n"
        "- Do not request Dockerfile, dependency files, lockfiles, env files, CI, credentials, or new files.\n"
        "- Stop with finish once you have enough exact file content to draft a narrow patch.\n"
        "- Prefer source related to query construction, ICP normalization, provider fallback, intent evidence, ranking, and adapter output.\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_code_edit_auto_research_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    runtime_source_context: Mapping[str, Any] | None = None,
    source_inspection_context: Mapping[str, Any] | None = None,
    budget_context: Mapping[str, Any] | None,
    max_candidates: int,
) -> list[dict[str, str]]:
    """Build the code-edit candidate prompt.

    The prompt is intentionally explicit because generated code is treated as
    untrusted input even though miners do not supply code.
    """

    source_context = _redacted_source_context(runtime_source_context or {})
    inspection_context = _redacted_source_context(source_inspection_context or {})
    editable_files = source_context.get("editable_files") if isinstance(source_context, Mapping) else None
    read_files = inspection_context.get("read_files") if isinstance(inspection_context, Mapping) else None
    example_target = _example_target_file(read_files, None) or _example_target_file(editable_files, None)
    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "runtime_source_context": source_context,
        "source_inspection_context": inspection_context,
        "budget_context": _redacted_mapping(budget_context or {}),
        "max_candidates": max(1, int(max_candidates)),
        "source_mode": "parent_image_extract",
        "allowed_runtime_roots": [
            "gateway/",
            "qualification/",
            "sourcing_model/",
            "validator_models/",
            "research_lab_adapter.py",
        ],
        "allowed_lanes": [
            "icp_normalization",
            "query_construction",
            "provider_fallback",
            "intent_evidence_quality",
            "company_fit_filtering",
            "openrouter_model_selection",
            "output_ranking",
        ],
    }
    system = (
        "You are Leadpoet Research Lab's code-editing auto-research engine. "
        "Your task is to improve the private sourcing model so it finds more "
        "perfect-fit companies for a supplied ICP plus observable buying-intent "
        "signals. You may propose small source, prompt, or model logic edits only "
        "inside the runtime extracted from the current ECR image. Optimize for "
        "general improvements across future sealed ICPs, "
        "not one visible ICP. Never request, infer, reveal, or store secrets, hidden "
        "benchmark plaintext, judge prompts, provider keys, private repo URLs, or "
        "customer-private data."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Choose one improvement lane per candidate: ICP normalization, query construction, "
        "provider fallback, intent evidence quality, company fit filtering, OpenRouter model "
        "selection, or output ranking.\n\n"
        "Allowed edit scope:\n"
        "- gateway/\n"
        "- qualification/\n"
        "- sourcing_model/\n"
        "- validator_models/\n"
        "- research_lab_adapter.py\n\n"
        "Active extracted source rules:\n"
        "- The current ECR image has already been pulled and /app has already been extracted before this prompt.\n"
        "- Use only exact files listed in Context JSON runtime_source_context.editable_files.\n"
        "- Every target file must be listed in source_inspection_context.read_files.\n"
        "- Build hunks only from exact file content returned in source_inspection_context.results.\n"
        "- Do not target example, placeholder, guessed, deleted, or non-listed paths.\n\n"
        "Forbidden edits:\n"
        "- Dockerfile, CI, dependency files, lockfiles, deploy scripts, credentials, env files\n"
        "- new top-level folders or files outside the allowed runtime roots\n"
        "- new files, even under an allowed root, unless the path already appears in editable_files\n"
        "- new external endpoints or new network clients outside existing provider modules\n"
        "- subprocess/shell execution additions\n"
        "- hidden ICP access, raw judge prompts, raw model responses, secrets, or key handling changes\n\n"
        "Diff requirements:\n"
        "- Produce a small git unified diff that applies to the active runtime source extracted from the current ECR image.\n"
        "- The unified_diff string must begin with 'diff --git a/<path> b/<path>'.\n"
        "- Include the standard '--- a/<path>' and '+++ b/<path>' headers and valid '@@' hunks.\n"
        "- Do not use Codex/apply_patch syntax. Never include '*** Begin Patch', '*** Update File', or '*** End Patch'.\n"
        "- Build every hunk from exact source lines visible in source_inspection_context read_file results.\n"
        "- If a read_file result is truncated, edit only the visible excerpt, or inspect a narrower relevant file in the next iteration.\n"
        "- Do not guess function bodies, line numbers, imports, or context lines that are not visible in source_inspection_context.\n"
        "- Keep the change testable and reversible.\n"
        "- Prefer one narrow code path over broad rewrites.\n"
        "- Do not overfit to one public ICP; the improvement must generalize.\n\n"
        "Expected output shape:\n"
        "{\"candidates\":[{\"lane\":\"query_construction\",\"hypothesis\":{\"failure_mode\":\"...\","
        "\"mechanism\":\"...\",\"expected_improvement\":\"...\",\"risk\":\"...\","
        "\"predicted_delta\":1.0},\"code_edit\":{\"target_files\":[\"" + example_target + "\"],"
        "\"unified_diff\":\"diff --git a/" + example_target + " b/" + example_target + "\\n--- a/" + example_target + "\\n+++ b/" + example_target + "\\n@@ ...\","
        "\"redacted_summary\":\"...\","
        "\"test_plan\":\"...\",\"rollback_plan\":\"...\"}}]}\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_code_edit_repair_messages(
    *,
    draft: CodeEditDraft,
    apply_error: str,
    source_inspection_context: Mapping[str, Any],
    runtime_source_context: Mapping[str, Any] | None,
    budget_context: Mapping[str, Any] | None,
    repair_attempt: int,
    max_candidates: int = 1,
) -> list[dict[str, str]]:
    """Ask the model to repair a generated diff that failed git apply."""

    source_context = _redacted_source_context(runtime_source_context or {})
    inspection_context = _redacted_source_context(source_inspection_context or {})
    context = {
        "repair_attempt": max(1, int(repair_attempt)),
        "failed_patch": {
            "lane": draft.lane,
            "target_files": list(draft.target_files),
            "unified_diff": normalize_unified_diff_text(draft.unified_diff),
            "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
            "hypothesis": {
                "failure_mode": draft.failure_mode,
                "mechanism": draft.mechanism,
                "expected_improvement": draft.expected_improvement,
                "risk": draft.risk,
                "predicted_delta": draft.predicted_delta,
            },
            "redacted_summary": draft.redacted_summary,
            "test_plan": draft.test_plan,
            "rollback_plan": draft.rollback_plan,
        },
        "git_apply_error": str(apply_error or "")[:2000],
        "runtime_source_context": source_context,
        "source_inspection_context": inspection_context,
        "budget_context": _redacted_mapping(budget_context or {}),
        "max_candidates": max(1, int(max_candidates)),
    }
    system = (
        "You are Leadpoet Research Lab's patch repair engine. A previous "
        "code-edit diff failed git apply against the extracted current ECR image "
        "source. Repair only the unified diff formatting or hunk context needed "
        "to make it apply. Do not broaden scope, change intent, add files, edit "
        "unread files, or use external knowledge."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Repair the failed patch so it applies cleanly to the exact source shown "
        "in source_inspection_context. Keep the same improvement intent and only "
        "target files listed in source_inspection_context.read_files.\n\n"
        "Rules:\n"
        "- Output the same candidates JSON shape used by the original code-edit draft.\n"
        "- If you cannot safely restate the hypothesis, return a direct repair object instead: "
        "{\"code_edit\":{\"target_files\":[...],\"unified_diff\":\"diff --git ...\","
        "\"redacted_summary\":\"...\",\"test_plan\":\"...\",\"rollback_plan\":\"...\"}}.\n"
        "- Include exactly one repaired code edit.\n"
        "- The unified_diff must start with 'diff --git a/<path> b/<path>'; no prose.\n"
        "- Include valid '--- a/<path>' and '+++ b/<path>' headers and '@@' hunks.\n"
        "- Do not use Codex/apply_patch syntax. Never include '*** Begin Patch', '*** Update File', or '*** End Patch'.\n"
        "- Do not create new files.\n"
        "- Do not edit dependency, Docker, CI, env, credential, or lock files.\n"
        "- Do not include secrets, hidden ICPs, judge prompts, or provider keys.\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_code_edit_source_inspection_response(
    raw_text: str,
    *,
    max_requests: int = 4,
) -> list[CodeEditSourceInspectionRequest]:
    try:
        decoded = _decode_json_value(raw_text)
    except ValueError:
        parsed = _source_inspection_requests_from_text(raw_text, max_requests=max_requests)
        if parsed:
            return parsed
        raise
    if isinstance(decoded, str):
        try:
            return parse_code_edit_source_inspection_response(decoded, max_requests=max_requests)
        except ValueError:
            parsed = _source_inspection_requests_from_text(decoded, max_requests=max_requests)
            if parsed:
                return parsed
            raise
    if isinstance(decoded, list):
        decoded = {"requests": decoded}
    if not isinstance(decoded, Mapping):
        raise ValueError("source-inspection response must be a JSON object or request array")
    if _contains_forbidden_material(decoded):
        raise ValueError("source-inspection response contains forbidden private or secret material")
    requests = _source_inspection_request_items(decoded)
    if decoded.get("finish") is True and not requests:
        return [CodeEditSourceInspectionRequest(operation="finish", rationale=str(decoded.get("rationale") or "")[:500])]
    if not isinstance(requests, list) or not requests:
        raise ValueError("source-inspection response requires a non-empty requests array")
    parsed: list[CodeEditSourceInspectionRequest] = []
    for item in requests[: max(1, int(max_requests))]:
        if not isinstance(item, Mapping):
            raise ValueError("source-inspection request must be an object")
        operation = str(_get_first_present(item, ("operation", "op", "action", "type", "request_type", "requestType")) or "").strip().lower()
        operation = {
            "read": "read_file",
            "readfile": "read_file",
            "file": "read_file",
            "grep": "search",
            "find": "search",
            "done": "finish",
            "stop": "finish",
        }.get(operation, operation)
        if operation not in {"search", "read_file", "finish"}:
            raise ValueError(f"unsupported source-inspection operation:{operation}")
        query = str(_get_first_present(item, ("query", "search", "pattern", "term")) or "")[:500]
        path = ""
        raw_path = _get_first_present(item, ("path", "file", "filepath", "file_path", "target", "target_file", "targetFile"))
        if raw_path is not None:
            path = _normalize_repo_path(raw_path)
        rationale = str(_get_first_present(item, ("rationale", "reason", "why", "description")) or "")[:700]
        if operation == "search" and not query.strip():
            raise ValueError("source-inspection search requires query")
        if operation == "read_file" and not path:
            raise ValueError("source-inspection read_file requires path")
        if operation == "finish":
            query = ""
            path = ""
        parsed.append(
            CodeEditSourceInspectionRequest(
                operation=operation,
                query=query,
                path=path,
                rationale=rationale,
            )
        )
    return parsed


def parse_code_edit_response(raw_text: str, *, max_candidates: int = 1) -> list[CodeEditDraft]:
    """Parse first-pass code-edit output from different LLM families.

    The builder still treats parsed drafts as untrusted: every returned draft
    must pass ``validate_code_edit_draft`` and later source-context/git-apply
    checks. This parser is intentionally tolerant of equivalent wrapper shapes
    so Claude/GPT/GLM style formatting drift does not throw away a valid diff.
    """

    try:
        decoded = _decode_json_value(raw_text)
    except ValueError:
        draft = _draft_from_diff_only_response(raw_text)
        if draft is None:
            raise
        return [draft]

    if isinstance(decoded, str):
        try:
            return parse_code_edit_response(decoded, max_candidates=max_candidates)
        except ValueError:
            draft = _draft_from_diff_only_response(decoded)
            if draft is not None:
                return [draft]
            raise ValueError("code-edit response string did not contain a parseable candidate")

    if not isinstance(decoded, (Mapping, list)):
        raise ValueError("code-edit response must be a JSON object or candidate array")
    if _contains_forbidden_material(decoded):
        raise ValueError("code-edit response contains forbidden private or secret material")

    candidates = _code_edit_candidate_items(decoded)
    if not candidates:
        raise ValueError("code-edit response requires a non-empty candidates array or equivalent code_edit object")

    drafts: list[CodeEditDraft] = []
    parse_errors: list[str] = []
    for item in candidates:
        if len(drafts) >= max(1, int(max_candidates)):
            break
        try:
            draft = _draft_from_code_edit_candidate(item)
            validate_code_edit_draft(draft)
        except Exception as exc:
            parse_errors.append(str(exc)[:200])
            continue
        drafts.append(draft)
    if not drafts:
        joined = "; ".join(parse_errors[:3])
        raise ValueError("code-edit response contained no valid code-edit drafts" + (f": {joined}" if joined else ""))
    return drafts


def parse_code_edit_repair_response(
    raw_text: str,
    *,
    original_draft: CodeEditDraft,
) -> list[CodeEditDraft]:
    """Parse a repair response, accepting narrow repair-only JSON shapes.

    Live models sometimes follow the repair instruction literally and return
    only a corrected ``code_edit`` object instead of repeating the complete
    candidate/hypothesis wrapper. For repair calls the original hypothesis is
    the source of truth, so carrying it forward is safer than discarding an
    otherwise valid fixed diff.
    """

    decoded = _decode_json_value(raw_text)
    if not isinstance(decoded, Mapping):
        raise ValueError("code-edit repair response must be a JSON object")
    if _contains_forbidden_material(decoded):
        raise ValueError("code-edit repair response contains forbidden private or secret material")

    candidate_items: list[Mapping[str, Any]] = []
    candidates = decoded.get("candidates")
    if isinstance(candidates, list) and candidates:
        for item in candidates[:1]:
            if not isinstance(item, Mapping):
                raise ValueError("repair candidate must be an object")
            candidate_items.append(item)
    elif isinstance(decoded.get("code_edit"), Mapping):
        candidate_items.append(decoded)
    elif decoded.get("unified_diff") is not None:
        candidate_items.append({"code_edit": decoded})
    else:
        raise ValueError("code-edit repair response requires a candidate or code_edit object")

    drafts: list[CodeEditDraft] = []
    for item in candidate_items:
        hypothesis = item.get("hypothesis")
        if not isinstance(hypothesis, Mapping):
            hypothesis = {
                "failure_mode": original_draft.failure_mode,
                "mechanism": original_draft.mechanism,
                "expected_improvement": original_draft.expected_improvement,
                "risk": original_draft.risk,
                "predicted_delta": original_draft.predicted_delta,
            }
        code_edit = item.get("code_edit")
        if not isinstance(code_edit, Mapping):
            raise ValueError("repair candidate requires code_edit object")
        raw_target_files = code_edit.get("target_files") or item.get("target_files") or original_draft.target_files
        target_files = tuple(_normalize_repo_path(path) for path in raw_target_files)
        unified_diff = normalize_unified_diff_text(str(code_edit.get("unified_diff") or item.get("unified_diff") or ""))
        if not unified_diff.strip():
            raise ValueError("code_edit.unified_diff is required")
        draft = CodeEditDraft(
            failure_mode=str(hypothesis.get("failure_mode") or original_draft.failure_mode)[:700],
            mechanism=str(hypothesis.get("mechanism") or original_draft.mechanism)[:1000],
            expected_improvement=str(
                hypothesis.get("expected_improvement") or original_draft.expected_improvement
            )[:1000],
            risk=str(hypothesis.get("risk") or original_draft.risk)[:700],
            lane=str(item.get("lane") or original_draft.lane)[:80],
            target_files=target_files,
            unified_diff=unified_diff,
            redacted_summary=str(
                code_edit.get("redacted_summary") or item.get("redacted_summary") or original_draft.redacted_summary
            )[:1200],
            test_plan=str(code_edit.get("test_plan") or item.get("test_plan") or original_draft.test_plan)[:1200],
            rollback_plan=str(
                code_edit.get("rollback_plan") or item.get("rollback_plan") or original_draft.rollback_plan
            )[:1200],
            predicted_delta=float(hypothesis.get("predicted_delta") or original_draft.predicted_delta or 1.0),
        )
        validate_code_edit_draft(draft)
        drafts.append(draft)
    return drafts


def _code_edit_candidate_items(decoded: Any, *, _depth: int = 0) -> list[Mapping[str, Any]]:
    if _depth > 4:
        return []
    if isinstance(decoded, list):
        return [item for item in (_candidate_item_from_any(item) for item in decoded) if item is not None]
    if not isinstance(decoded, Mapping):
        return []

    candidate_keys = (
        "candidates",
        "candidate",
        "code_edits",
        "code_edit_candidates",
        "edits",
        "patches",
        "diffs",
        "changes",
        "change",
        "modifications",
        "file_changes",
        "fileChanges",
    )
    for key in candidate_keys:
        value = decoded.get(key)
        if isinstance(value, list):
            return [item for item in (_candidate_item_from_any(item) for item in value) if item is not None]
        if isinstance(value, Mapping):
            return [value]
        if isinstance(value, str) and normalize_unified_diff_text(value).lstrip().startswith("diff --git "):
            return [{"unified_diff": value}]

    if _looks_like_code_edit_candidate(decoded):
        return [decoded]

    wrapper_keys = ("result", "response", "output", "data", "message", "content", "json", "final", "answer", "text")
    for key in wrapper_keys:
        value = decoded.get(key)
        if isinstance(value, (Mapping, list)):
            nested = _code_edit_candidate_items(value, _depth=_depth + 1)
            if nested:
                return nested
        if isinstance(value, str):
            try:
                nested_decoded = _decode_json_value(value)
            except ValueError:
                if normalize_unified_diff_text(value).lstrip().startswith("diff --git "):
                    return [{"unified_diff": value}]
                continue
            nested = _code_edit_candidate_items(nested_decoded, _depth=_depth + 1)
            if nested:
                return nested
    return []


def _source_inspection_request_items(decoded: Mapping[str, Any], *, _depth: int = 0) -> list[Any] | None:
    if _depth > 4:
        return None
    for key in (
        "requests",
        "request",
        "source_requests",
        "sourceRequests",
        "inspection_requests",
        "inspectionRequests",
        "operations",
        "actions",
    ):
        value = decoded.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, Mapping):
            return [value]
    wrapper_keys = ("result", "response", "output", "data", "message", "content", "json", "final", "answer", "text")
    for key in wrapper_keys:
        value = decoded.get(key)
        if isinstance(value, Mapping):
            nested = _source_inspection_request_items(value, _depth=_depth + 1)
            if nested:
                return nested
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                nested_decoded = _decode_json_value(value)
            except ValueError:
                parsed = _source_inspection_requests_from_text(value, max_requests=4)
                if parsed:
                    return [
                        {
                            "operation": item.operation,
                            "query": item.query,
                            "path": item.path,
                            "rationale": item.rationale,
                        }
                        for item in parsed
                    ]
                continue
            if isinstance(nested_decoded, list):
                return nested_decoded
            if isinstance(nested_decoded, Mapping):
                nested = _source_inspection_request_items(nested_decoded, _depth=_depth + 1)
                if nested:
                    return nested
    return None


def _source_inspection_requests_from_text(raw_text: str, *, max_requests: int) -> list[CodeEditSourceInspectionRequest]:
    text = str(raw_text or "")
    parsed: list[CodeEditSourceInspectionRequest] = []
    seen: set[tuple[str, str, str]] = set()
    path_pattern = r"([A-Za-z0-9_./-]+(?:\.py|\.json|\.ya?ml|\.toml|\.txt|\.md))"
    for pattern in (
        r"(?:read_file|read file|read|file|path)\s*[:=\-]?\s*[`'\"]?" + path_pattern,
        r"[`'\"]path[`'\"]\s*:\s*[`'\"]" + path_pattern,
        r"[`'\"]file[`'\"]\s*:\s*[`'\"]" + path_pattern,
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            path = _normalize_repo_path(match.group(1))
            key = ("read_file", "", path)
            if key in seen:
                continue
            seen.add(key)
            parsed.append(CodeEditSourceInspectionRequest(operation="read_file", path=path, rationale="parsed from text fallback"))
            if len(parsed) >= max(1, int(max_requests)):
                return parsed
    for pattern in (
        r"(?:search|grep|find)\s*[:=\-]?\s*[`'\"]([^`'\"\n]{2,200})[`'\"]?",
        r"[`'\"]query[`'\"]\s*:\s*[`'\"]([^`'\"\n]{2,200})[`'\"]",
    ):
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            query = str(match.group(1) or "").strip()[:500]
            key = ("search", query, "")
            if not query or key in seen:
                continue
            seen.add(key)
            parsed.append(CodeEditSourceInspectionRequest(operation="search", query=query, rationale="parsed from text fallback"))
            if len(parsed) >= max(1, int(max_requests)):
                return parsed
    if re.search(r"\b(finish|done|stop)\b", text, flags=re.IGNORECASE):
        parsed.append(CodeEditSourceInspectionRequest(operation="finish", rationale="parsed from text fallback"))
    return parsed[: max(1, int(max_requests))]


def _candidate_item_from_any(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, str) and normalize_unified_diff_text(value).lstrip().startswith("diff --git "):
        return {"unified_diff": value}
    return None


def _looks_like_code_edit_candidate(value: Mapping[str, Any]) -> bool:
    if isinstance(value.get("code_edit"), Mapping):
        return True
    if isinstance(value.get("codeEdit"), Mapping):
        return True
    if isinstance(value.get("edit"), Mapping):
        return True
    if isinstance(value.get("patch"), Mapping):
        return True
    return _mapping_has_diff(value)


def _mapping_has_diff(value: Mapping[str, Any]) -> bool:
    if any(_get_first_present(value, keys) is not None for keys in (
        ("unified_diff", "unifiedDiff", "git_diff", "gitDiff"),
        ("diff", "patch_text", "patchText"),
    )):
        return True
    files = value.get("files") or value.get("file_changes") or value.get("fileChanges")
    if isinstance(files, list):
        return any(isinstance(item, Mapping) and _mapping_has_diff(item) for item in files)
    return False


def _draft_from_code_edit_candidate(item: Mapping[str, Any]) -> CodeEditDraft:
    hypothesis = item.get("hypothesis")
    if not isinstance(hypothesis, Mapping):
        hypothesis = item.get("rationale") if isinstance(item.get("rationale"), Mapping) else {}
    code_edit = _candidate_code_edit_mapping(item)
    unified_diff = normalize_unified_diff_text(_string_value(_candidate_unified_diff(item, code_edit)))
    if not unified_diff.strip():
        raise ValueError("code_edit.unified_diff is required")
    target_files = _coerce_target_files(
        _get_first_present(
            code_edit,
            ("target_files", "targetFiles", "targets", "files", "paths", "target_file", "targetFile", "file", "path"),
        )
        or _get_first_present(
            item,
            ("target_files", "targetFiles", "targets", "files", "paths", "target_file", "targetFile", "file", "path"),
        )
    )
    if not target_files:
        target_files = tuple(sorted(extract_unified_diff_paths(unified_diff)))
    lane = _string_value(_get_first_present(item, ("lane", "improvement_lane", "improvementLane", "category")))
    return CodeEditDraft(
        failure_mode=_string_value(
            _get_first_present(hypothesis, ("failure_mode", "failureMode", "problem", "issue", "why"))
            or _get_first_present(item, ("failure_mode", "failureMode", "problem", "issue"))
        )[:700],
        mechanism=_string_value(
            _get_first_present(hypothesis, ("mechanism", "approach", "change", "solution"))
            or _get_first_present(item, ("mechanism", "approach", "change", "solution"))
        )[:1000],
        expected_improvement=_string_value(
            _get_first_present(hypothesis, ("expected_improvement", "expectedImprovement", "impact", "benefit"))
            or _get_first_present(item, ("expected_improvement", "expectedImprovement", "impact", "benefit"))
        )[:1000],
        risk=_string_value(
            _get_first_present(hypothesis, ("risk", "risks", "tradeoff", "tradeoffs"))
            or _get_first_present(item, ("risk", "risks", "tradeoff", "tradeoffs"))
        )[:700],
        lane=lane[:80],
        target_files=target_files,
        unified_diff=unified_diff,
        redacted_summary=_string_value(
            _get_first_present(code_edit, ("redacted_summary", "redactedSummary", "summary", "description"))
            or _get_first_present(item, ("redacted_summary", "redactedSummary", "summary", "description"))
        )[:1200],
        test_plan=_string_value(
            _get_first_present(code_edit, ("test_plan", "testPlan", "tests", "verification"))
            or _get_first_present(item, ("test_plan", "testPlan", "tests", "verification"))
        )[:1200],
        rollback_plan=_string_value(
            _get_first_present(code_edit, ("rollback_plan", "rollbackPlan", "rollback", "revert_plan", "revertPlan"))
            or _get_first_present(item, ("rollback_plan", "rollbackPlan", "rollback", "revert_plan", "revertPlan"))
        )[:1200],
        predicted_delta=_float_value(
            _get_first_present(hypothesis, ("predicted_delta", "predictedDelta", "delta", "expected_delta", "expectedDelta")),
            default=1.0,
        ),
    )


def _candidate_code_edit_mapping(item: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("code_edit", "codeEdit", "edit"):
        value = item.get(key)
        if isinstance(value, Mapping):
            return value
    patch = item.get("patch")
    if isinstance(patch, Mapping):
        return patch
    return item


def _candidate_unified_diff(item: Mapping[str, Any], code_edit: Mapping[str, Any]) -> Any:
    for source in (code_edit, item):
        value = _get_first_present(
            source,
            ("unified_diff", "unifiedDiff", "git_diff", "gitDiff", "diff", "patch_text", "patchText"),
        )
        if value is not None:
            return value
        patch = source.get("patch")
        if isinstance(patch, str):
            return patch
        if isinstance(patch, Mapping):
            nested = _get_first_present(
                patch,
                ("unified_diff", "unifiedDiff", "git_diff", "gitDiff", "diff", "patch_text", "patchText"),
            )
            if nested is not None:
                return nested
        file_diffs = _diff_from_file_changes(source)
        if file_diffs:
            return file_diffs
    return ""


def _diff_from_file_changes(source: Mapping[str, Any]) -> str:
    files = source.get("files") or source.get("file_changes") or source.get("fileChanges")
    if not isinstance(files, list):
        return ""
    diffs: list[str] = []
    for item in files:
        if not isinstance(item, Mapping):
            continue
        diff = _get_first_present(
            item,
            ("unified_diff", "unifiedDiff", "git_diff", "gitDiff", "diff", "patch_text", "patchText"),
        )
        normalized = normalize_unified_diff_text(_string_value(diff))
        if normalized.lstrip().startswith("diff --git "):
            diffs.append(normalized.rstrip())
    return "\n".join(diffs) + ("\n" if diffs else "")


def _draft_from_diff_only_response(raw_text: str) -> CodeEditDraft | None:
    unified_diff = normalize_unified_diff_text(raw_text)
    if not unified_diff.lstrip().startswith("diff --git "):
        return None
    paths = tuple(sorted(extract_unified_diff_paths(unified_diff)))
    if not paths:
        return None
    draft = CodeEditDraft(
        failure_mode="",
        mechanism="",
        expected_improvement="",
        risk="",
        lane="",
        target_files=paths,
        unified_diff=unified_diff,
        redacted_summary="Direct git diff emitted without candidate wrapper.",
        test_plan="Run Research Lab candidate build and validation.",
        rollback_plan="Reject candidate or revert the emitted diff.",
        predicted_delta=1.0,
    )
    validate_code_edit_draft(draft)
    return draft


def _coerce_target_files(value: Any) -> tuple[str, ...]:
    raw_items: list[Any]
    if value is None:
        raw_items = []
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    paths: list[str] = []
    for item in raw_items:
        if isinstance(item, Mapping):
            item = _get_first_present(item, ("path", "file", "target_file", "targetFile", "name"))
        if item is None:
            continue
        path = _normalize_repo_path(item)
        if path and path not in paths:
            paths.append(path)
    return tuple(paths)


def _get_first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping and mapping.get(key) is not None:
            return mapping.get(key)
    return None


def _string_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "; ".join(_string_value(item) for item in value if item is not None)
    if isinstance(value, Mapping):
        try:
            return json.dumps(value, sort_keys=True, separators=(",", ":"))
        except TypeError:
            return str(value)
    return str(value)


def _float_value(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_unified_diff_text(value: str) -> str:
    """Normalize common LLM wrappers without changing patch semantics."""

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    diff_index = text.find("diff --git ")
    if diff_index > 0:
        text = text[diff_index:].strip()
    elif diff_index < 0:
        header_candidates = [
            index
            for marker in ("\n--- ", "--- ")
            if (index := text.find(marker)) >= 0
        ]
        if header_candidates:
            start = min(header_candidates)
            if text[start:].startswith("\n"):
                start += 1
            text = text[start:].strip()
    if text.startswith("```"):
        return normalize_unified_diff_text(text)
    # Keep apply_patch-style output parseable for diagnostics, but make sure it
    # cannot be mistaken for a valid git unified diff.
    if "*** Begin Patch" in text or "*** Update File:" in text or "*** End Patch" in text:
        return text.rstrip() + "\n"
    return text.rstrip() + "\n"


def validate_code_edit_draft(
    draft: CodeEditDraft,
    *,
    allowed_prefixes: Sequence[str] = DEFAULT_ALLOWED_PATH_PREFIXES,
    allowed_exact_paths: Sequence[str] = DEFAULT_ALLOWED_EXACT_PATHS,
    allowed_suffixes: Sequence[str] = DEFAULT_ALLOWED_SUFFIXES,
) -> list[str]:
    errors: list[str] = []
    payload = draft.to_dict()
    if _contains_forbidden_material(payload):
        errors.append("code_edit_contains_forbidden_material")
    stripped_diff = draft.unified_diff.lstrip()
    if "*** Begin Patch" in stripped_diff or "*** Update File:" in stripped_diff or "*** End Patch" in stripped_diff:
        errors.append("code_edit_uses_apply_patch_format")
    if stripped_diff and not stripped_diff.startswith("diff --git "):
        errors.append("code_edit_requires_git_unified_diff")
    diff_paths = extract_unified_diff_paths(draft.unified_diff)
    target_paths = set(draft.target_files)
    all_paths = sorted(diff_paths | target_paths)
    if not all_paths:
        errors.append("code_edit_has_no_target_paths")
    for path in all_paths:
        errors.extend(_validate_repo_path(
            path,
            allowed_prefixes=allowed_prefixes,
            allowed_exact_paths=allowed_exact_paths,
            allowed_suffixes=allowed_suffixes,
        ))
    for pattern in DISALLOWED_DIFF_PATTERNS:
        if re.search(pattern, draft.unified_diff):
            errors.append(f"code_edit_disallowed_diff_pattern:{pattern}")
    if errors:
        raise ValueError("; ".join(errors))
    return []


def code_edit_candidate_manifest(
    *,
    draft: CodeEditDraft,
    parent_artifact_hash: str,
    candidate_artifact_hash: str,
    candidate_model_manifest_hash: str,
    source_diff_hash: str,
    build_doc_hash: str,
) -> dict[str, Any]:
    payload = {
        "candidate_kind": "image_build",
        "patch_type": "IMAGE_BUILD",
        "target_component_id": "private_model_source_tree",
        "parent_artifact_hash": str(parent_artifact_hash),
        "candidate_artifact_hash": str(candidate_artifact_hash),
        "candidate_model_manifest_hash": str(candidate_model_manifest_hash),
        "patch_payload_hash": str(source_diff_hash),
        "candidate_source_diff_hash": str(source_diff_hash),
        "candidate_build_doc_hash": str(build_doc_hash),
        "redacted_summary": draft.redacted_summary,
        "validation_result": "passed",
        "patch_doc": {
            "edit_contract": "code_edit_image_build:v1",
            "lane": draft.lane,
            "target_files": list(draft.target_files),
            "unified_diff_hash": sha256_json({"unified_diff": draft.unified_diff}),
            "expected_improvement": draft.expected_improvement,
            "risk": draft.risk,
            "test_plan": draft.test_plan,
            "rollback_plan": draft.rollback_plan,
        },
    }
    return {**payload, "manifest_hash": sha256_json(payload)}


def extract_unified_diff_paths(diff_text: str) -> set[str]:
    paths: set[str] = set()
    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                for item in parts[2:4]:
                    paths.add(_normalize_diff_path(item))
        elif line.startswith("--- ") or line.startswith("+++ "):
            item = line[4:].split("\t", 1)[0].strip()
            normalized = _normalize_diff_path(item)
            if normalized:
                paths.add(normalized)
    return {path for path in paths if path}


def _validate_repo_path(
    path: str,
    *,
    allowed_prefixes: Sequence[str],
    allowed_exact_paths: Sequence[str],
    allowed_suffixes: Sequence[str],
) -> list[str]:
    errors: list[str] = []
    normalized = _normalize_repo_path(path)
    if normalized != path:
        errors.append(f"invalid_repo_path:{path}")
    for pattern in DISALLOWED_PATH_PATTERNS:
        if re.search(pattern, normalized):
            errors.append(f"disallowed_repo_path:{normalized}")
    if not (
        normalized in set(allowed_exact_paths)
        or any(normalized.startswith(prefix) for prefix in allowed_prefixes)
    ):
        errors.append(f"path_not_in_code_edit_allowlist:{normalized}")
    if not normalized.endswith(tuple(allowed_suffixes)):
        errors.append(f"path_suffix_not_allowed:{normalized}")
    return errors


def _normalize_diff_path(value: str) -> str:
    if value in {"/dev/null", "dev/null"}:
        return ""
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return _normalize_repo_path(value)


def _normalize_repo_path(value: Any) -> str:
    path = str(value or "").replace("\\", "/").strip()
    path = posixpath.normpath(path)
    if path in {".", ""} or path.startswith("../") or path.startswith("/") or "/../" in path:
        raise ValueError(f"unsafe repo path: {value}")
    return path


def _contains_forbidden_material(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_forbidden_material(key) or _contains_forbidden_material(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_forbidden_material(item) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(marker in lowered for marker in FORBIDDEN_CODE_EDIT_TERMS)
    return False


def _redacted_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    decoded = json.loads(json.dumps(value, default=str))
    if _contains_forbidden_material(decoded):
        return {"redacted": True, "hash": sha256_json({"value": decoded})}
    return decoded


def _redacted_source_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep source inventory usable while removing obvious raw secret values."""

    decoded = json.loads(json.dumps(value, default=str))
    secret_markers = ("sk-or-", "sb_secret", "aws_secret_access_key", "password=", "private_key")

    def scrub(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): scrub(val) for key, val in item.items()}
        if isinstance(item, list):
            return [scrub(val) for val in item]
        if isinstance(item, str):
            if any(marker in item.lower() for marker in secret_markers):
                return "[redacted secret-like value]"
            return item
        return item

    return scrub(decoded)


def _extract_json_object(raw_text: str) -> str:
    decoded = _decode_json_value(raw_text)
    if not isinstance(decoded, Mapping):
        raise ValueError("response did not contain a JSON object")
    return json.dumps(decoded)


def _decode_json_value(raw_text: str) -> Any:
    text = str(raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].lstrip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        raise ValueError("response did not contain a JSON object or array")
    start = min(starts)
    if start < 0:
        raise ValueError("response did not contain a JSON object or array")
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    return obj


def _example_target_file(editable_files: Any, file_previews: Any) -> str:
    if isinstance(file_previews, list):
        for item in file_previews:
            if isinstance(item, Mapping):
                path = str(item.get("path") or "")
                if path:
                    return path
    if isinstance(editable_files, list):
        for item in editable_files:
            path = str(item or "")
            if path:
                return path
    return "research_lab_adapter.py"

"""Code-edit auto-research contracts for candidate private model images."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def build_code_edit_auto_research_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    budget_context: Mapping[str, Any] | None,
    max_candidates: int,
) -> list[dict[str, str]]:
    """Build the code-edit candidate prompt.

    The prompt is intentionally explicit because generated code is treated as
    untrusted input even though miners do not supply code.
    """

    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
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
        "Forbidden edits:\n"
        "- Dockerfile, CI, dependency files, lockfiles, deploy scripts, credentials, env files\n"
        "- new top-level folders or files outside the allowed runtime roots\n"
        "- new external endpoints or new network clients outside existing provider modules\n"
        "- subprocess/shell execution additions\n"
        "- hidden ICP access, raw judge prompts, raw model responses, secrets, or key handling changes\n\n"
        "Diff requirements:\n"
        "- Produce a small unified diff that applies to the active runtime source extracted from the current ECR image.\n"
        "- Keep the change testable and reversible.\n"
        "- Prefer one narrow code path over broad rewrites.\n"
        "- Do not overfit to one public ICP; the improvement must generalize.\n\n"
        "Expected output shape:\n"
        "{\"candidates\":[{\"lane\":\"query_construction\",\"hypothesis\":{\"failure_mode\":\"...\","
        "\"mechanism\":\"...\",\"expected_improvement\":\"...\",\"risk\":\"...\","
        "\"predicted_delta\":1.0},\"code_edit\":{\"target_files\":[\"sourcing_model/example.py\"],"
        "\"unified_diff\":\"diff --git ...\",\"redacted_summary\":\"...\","
        "\"test_plan\":\"...\",\"rollback_plan\":\"...\"}}]}\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_code_edit_response(raw_text: str, *, max_candidates: int = 1) -> list[CodeEditDraft]:
    decoded = json.loads(_extract_json_object(raw_text))
    if not isinstance(decoded, Mapping):
        raise ValueError("code-edit response must be a JSON object")
    if _contains_forbidden_material(decoded):
        raise ValueError("code-edit response contains forbidden private or secret material")
    candidates = decoded.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("code-edit response requires a non-empty candidates array")

    drafts: list[CodeEditDraft] = []
    for item in candidates[: max(1, int(max_candidates))]:
        if not isinstance(item, Mapping):
            raise ValueError("candidate must be an object")
        hypothesis = item.get("hypothesis")
        code_edit = item.get("code_edit")
        if not isinstance(hypothesis, Mapping) or not isinstance(code_edit, Mapping):
            raise ValueError("candidate requires hypothesis and code_edit objects")
        target_files = tuple(_normalize_repo_path(path) for path in code_edit.get("target_files") or ())
        unified_diff = str(code_edit.get("unified_diff") or "")
        if not unified_diff.strip():
            raise ValueError("code_edit.unified_diff is required")
        draft = CodeEditDraft(
            failure_mode=str(hypothesis.get("failure_mode") or "")[:700],
            mechanism=str(hypothesis.get("mechanism") or "")[:1000],
            expected_improvement=str(hypothesis.get("expected_improvement") or "")[:1000],
            risk=str(hypothesis.get("risk") or "")[:700],
            lane=str(item.get("lane") or "")[:80],
            target_files=target_files,
            unified_diff=unified_diff,
            redacted_summary=str(code_edit.get("redacted_summary") or "")[:1200],
            test_plan=str(code_edit.get("test_plan") or "")[:1200],
            rollback_plan=str(code_edit.get("rollback_plan") or "")[:1200],
            predicted_delta=float(hypothesis.get("predicted_delta") or 1.0),
        )
        validate_code_edit_draft(draft)
        drafts.append(draft)
    return drafts


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


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response did not contain a JSON object")
    return text[start : end + 1]

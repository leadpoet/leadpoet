"""Code-edit auto-research contracts for candidate private model images."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field, replace
import io
import json
import os
import posixpath
import re
import tokenize
from typing import Any, Mapping, Sequence

from gateway.research_lab.config import DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG
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

_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "apikey",
        "authorization",
        "authorizationheader",
        "awsaccesskeyid",
        "awssecretaccesskey",
        "bearertoken",
        "credential",
        "credentials",
        "hiddenbenchmark",
        "hiddenicp",
        "icpplaintext",
        "judgeprompt",
        "openrouterapikey",
        "password",
        "passwd",
        "privatekey",
        "privaterepo",
        "privaterepourl",
        "rawopenrouterkey",
        "rawsecret",
        "refreshtoken",
        "secretkey",
        "secret",
        "servicerole",
        "servicerolekey",
        "supabaseservicerolekey",
        "token",
    }
)
_SENSITIVE_FIELD_SUFFIXES = (
    "apikey",
    "accesstoken",
    "authorization",
    "authorizationheader",
    "bearertoken",
    "credential",
    "credentials",
    "password",
    "passwd",
    "privatekey",
    "refreshtoken",
    "secret",
    "secretkey",
    "servicerolekey",
    "token",
)
_SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-or-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsb_secret_[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"(?i)\b[a-z][a-z0-9+.-]{1,20}://[^\s'\"@/]+:[^\s'\"@]+@[^\s'\"]+"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|password|passwd|"
        r"secret(?:[_-]?key)?|private[_-]?key|service[_-]?role(?:[_-]?key)?)\b"
        r"\s*[:=]\s*[\"']?[A-Za-z0-9_./+=-]{12,}"
    ),
)
_SENSITIVE_POLICY_PREFIX_RE = re.compile(
    r"(?i)\b(?:no|do not|don't|must not|mustn't|never|avoid|without|"
    r"forbid|prohibit|disallow|redact|exclude)(?:\s+[a-z0-9]+){0,8}\s*$"
)
_SENSITIVE_POLICY_SUFFIX_RE = re.compile(
    r"(?i)^\s*(?:must not|mustn't|should not|should never|cannot|can't|"
    r"is forbidden|is prohibited|is disallowed|is excluded|is redacted|"
    r"is not allowed|is not permitted|must be omitted|must be removed)\b"
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
LOOP_DIRECTION_ALLOWED_LANES = (
    "icp_normalization",
    "query_construction",
    "source_routing",
    "provider_fallback",
    "intent_evidence_quality",
    "company_fit_filtering",
    "openrouter_model_selection",
    "output_ranking",
)
LOOP_DIRECTION_VALIDATION_MODES = (
    "runtime_checks",
    "existing_test_files",
)
CODE_EDIT_NO_VIABLE_FAILURE_CLASSES = (
    "binding_plan_unimplementable",
    "insufficient_source_context",
    "unsafe_or_out_of_scope",
    "provider_probe_refuted_hypothesis",
    "no_safe_patch",
)
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

# plan_path_id values that are prompt-example placeholders, not real path ids.
_PLAN_PATH_ID_PLACEHOLDERS = frozenset(
    {
        "selected_path_id",
        "selectedpathid",
        "<selected_path_id>",
        "path_id_copied_from_loop_direction_plan",
    }
)

# Conservative synonyms live judge models use for a passing verdict. Anything
# not recognized here still defaults to "fail" (unchanged behavior).
_PASS_VERDICT_SYNONYMS = frozenset(
    {
        "pass",
        "passed",
        "passes",
        "accept",
        "accepted",
        "approve",
        "approved",
        "true",
        "yes",
        "ok",
        "okay",
        "aligned",
        "success",
        "successful",
        "valid",
    }
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
    plan_path_id: str = ""
    plan_alignment: dict[str, Any] = field(default_factory=dict)
    expected_metric_effect: dict[str, Any] = field(default_factory=dict)

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
    # Optional 1-based line range for read_file (0 = whole-file/head behavior).
    # Only honored when RESEARCH_LAB_SOURCE_ACCESS_V2 is enabled.
    start_line: int = 0
    max_lines: int = 0
    # probe_provider fields (W4): typed catalog entry + schema-checked params.
    provider: str = ""
    endpoint: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_event_doc(self) -> dict[str, Any]:
        payload = {
            "operation": self.operation,
            "query_hash": sha256_json({"query": self.query}) if self.query else "",
            "path": self.path,
            "rationale_hash": sha256_json({"rationale": self.rationale}) if self.rationale else "",
        }
        if self.start_line > 0:
            payload["start_line"] = self.start_line
        if self.max_lines > 0:
            payload["max_lines"] = self.max_lines
        if self.operation == "probe_provider":
            payload["provider"] = self.provider
            payload["endpoint"] = self.endpoint
            payload["params_hash"] = sha256_json({"params": self.params}) if self.params else ""
        return {key: value for key, value in payload.items() if value not in {"", None}}


@dataclass(frozen=True)
class CodeEditNoViablePatch:
    schema_version: str
    failure_class: str
    reason: str
    missing_references: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["no_viable_patch"] = True
        payload["missing_references"] = list(self.missing_references)
        return payload


@dataclass(frozen=True)
class LoopDirectionPlan:
    schema_version: str
    miner_focus_interpretation: str
    loop_goal: str
    required_lane: str
    required_mechanism: str
    target_behavior: tuple[str, ...]
    must_inspect: tuple[str, ...]
    allowed_lanes: tuple[str, ...]
    disallowed_lanes: tuple[str, ...]
    must_not_try: tuple[str, ...]
    success_criteria: tuple[str, ...]
    novelty_requirements: tuple[str, ...]
    anti_overfit_checks: tuple[str, ...]
    ranked_paths: tuple[dict[str, Any], ...]
    selected_path_id: str
    generalization_claim: str = ""
    novelty_contrast: str = ""
    no_new_safe_path: bool = False
    reason: str = ""
    unresolved_references: tuple[str, ...] = ()
    validation_mode: str = "runtime_checks"
    validation_paths: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "target_behavior",
            "must_inspect",
            "allowed_lanes",
            "disallowed_lanes",
            "must_not_try",
            "success_criteria",
            "novelty_requirements",
            "anti_overfit_checks",
            "ranked_paths",
            "unresolved_references",
            "validation_paths",
        ):
            payload[key] = list(payload[key])
        payload["plan_hash"] = sha256_json({key: value for key, value in payload.items() if key != "plan_hash"})
        return payload


@dataclass(frozen=True)
class PlanAlignmentVerdict:
    schema_version: str
    verdict: str
    reason: str
    detected_lane: str = ""
    detected_mechanism: str = ""
    novel: bool = True
    blocking_issue: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_loop_direction_planner_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    runtime_source_index: Mapping[str, Any],
    budget_context: Mapping[str, Any] | None,
    prior_attempts: Sequence[Mapping[str, Any]] | None = None,
    provider_outcome_digest: Mapping[str, Any] | None = None,
    provider_capability_summary: Mapping[str, Any] | None = None,
    candidate_edit_constraints: Mapping[str, Any] | None = None,
    branch_factor: int = DEFAULT_RESEARCH_LAB_GIT_TREE_CONFIG.branch_factor,
) -> list[dict[str, str]]:
    required_branch_count = max(1, min(8, int(branch_factor)))
    allowed_lanes = list(LOOP_DIRECTION_ALLOWED_LANES)
    indexed_paths = [
        str(item.get("path") or "")
        for item in runtime_source_index.get("files", [])
        if isinstance(item, Mapping) and item.get("path")
    ]
    if not indexed_paths:
        editable_files = runtime_source_index.get("editable_files")
        if isinstance(editable_files, Sequence) and not isinstance(
            editable_files, (str, bytes, bytearray)
        ):
            indexed_paths = [str(item) for item in editable_files if item]
    example_inspect_path = indexed_paths[0] if indexed_paths else "research_lab_adapter.py"
    example_inspect_reference = example_inspect_path
    for file_doc in runtime_source_index.get("files", []):
        if not isinstance(file_doc, Mapping) or str(file_doc.get("path") or "") != example_inspect_path:
            continue
        symbols = file_doc.get("symbols")
        if isinstance(symbols, Sequence) and not isinstance(symbols, (str, bytes, bytearray)):
            for symbol in symbols:
                if not isinstance(symbol, Mapping):
                    continue
                qualified_name = str(symbol.get("qualified_name") or "").strip()
                if qualified_name:
                    example_inspect_reference = f"{example_inspect_path}::{qualified_name}"
                    break
        break
    example_path = {
        "path_id": "query_decomposition_recall",
        "lane": "query_construction",
        "mechanism": "Add one bounded decomposed-search pass while preserving downstream gates.",
        "target_behavior": ["Recover sparse searches without weakening ICP gates."],
        "must_inspect": [example_inspect_reference],
        "allowed_lanes": ["query_construction"],
        "disallowed_lanes": ["provider_fallback"],
        "must_not_try": ["Do not weaken ICP constraints."],
        "success_criteria": ["Existing runtime checks pass and sparse-query behavior improves."],
        "novelty_requirements": ["Use a mechanism not present in prior attempts."],
        "anti_overfit_checks": ["Preserve multiple qualified outputs."],
        "validation_mode": "runtime_checks",
        "validation_paths": [],
    }
    example_plan = {
        "schema_version": "1.1",
        "miner_focus_interpretation": "Improve sparse-query recall.",
        "loop_goal": "Recover qualified companies without weakening ICP gates.",
        "required_lane": example_path["lane"],
        "required_mechanism": example_path["mechanism"],
        "generalization_claim": "The bounded behavior applies to future sealed ICPs.",
        "target_behavior": example_path["target_behavior"],
        "must_inspect": example_path["must_inspect"],
        "allowed_lanes": example_path["allowed_lanes"],
        "disallowed_lanes": example_path["disallowed_lanes"],
        "must_not_try": example_path["must_not_try"],
        "success_criteria": example_path["success_criteria"],
        "novelty_requirements": example_path["novelty_requirements"],
        "anti_overfit_checks": example_path["anti_overfit_checks"],
        "validation_mode": example_path["validation_mode"],
        "validation_paths": example_path["validation_paths"],
        "novelty_contrast": "This uses a new bounded decomposition mechanism.",
        "unresolved_references": [],
        "ranked_paths": [example_path],
        "selected_path_id": example_path["path_id"],
    }
    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "runtime_source_index": _redacted_source_context(runtime_source_index),
        "budget_context": _redacted_mapping(budget_context or {}),
        "prior_attempts": _redacted_mapping({"attempts": list(prior_attempts or [])}).get("attempts", []),
        "allowed_lanes": allowed_lanes,
        "candidate_edit_constraints": _redacted_mapping(candidate_edit_constraints or {}),
        "required_root_branch_count": required_branch_count,
    }
    if provider_outcome_digest:
        # W2: recorded provider truth (error/status histograms, zero-result
        # rates) sits beside the benchmark summary so plans can target real
        # provider failure modes instead of guessing them.
        context["provider_outcome_digest"] = _redacted_mapping(provider_outcome_digest)
    if provider_capability_summary:
        context["approved_provider_capabilities"] = _redacted_mapping(provider_capability_summary)
    system = (
        "You are Leadpoet Research Lab's loop-direction planner. Convert a miner's "
        "public research focus into a binding, auditable code-edit plan for a later "
        "executor model. You do not write code. Choose safe, generalizable paths "
        "that directly address the miner focus and avoid repeating prior attempts."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Rules:\n"
        "- Primary objective: improve true sourcing quality for future sealed ICPs. Treat public benchmark performance as a noisy validation signal, not as the objective.\n"
        "- Treat ticket.brief_public_summary as the miner research focus.\n"
        "- If ticket.brief_public_summary names a failure mode, selected_path_id must directly address that failure mode.\n"
        "- Choose required_lane from Context JSON allowed_lanes; do not force provider_fallback unless the failure is a provider/runtime failure.\n"
        "- Query decomposition, query relaxation, and term selection belong to query_construction.\n"
        "- Alternate discovery surface/provider routing after a completed-empty primary result belongs to source_routing.\n"
        "- Provider fallback is only for genuine provider/runtime failures such as HTTP errors, timeouts, malformed provider responses, or runtime exceptions.\n"
        "- Do not optimize for known public ICP quirks, visible examples, or benchmark-specific shortcuts.\n"
        "- If the miner focus is blank or broad, narrow it to one testable mechanism using benchmark and outcome context.\n"
        "- If prior_attempts show an already-tried path, choose a meaningfully different selected_path_id.\n"
        "- State how the selected path differs from prior_attempts by mechanism, target function/file, and expected behavior.\n"
        "- Do not improve apparent score by returning fewer companies, selecting only one safest company, weakening ICP constraints, removing hard filters without replacement, or hiding failures.\n"
        "- Do not remove ICP constraints unless replacing them with a more faithful constraint or adding a compensating downstream validation/filter.\n"
        "- Provider fallback paths must classify, retry, timeout, log, or route provider/runtime failures; pure query wording changes are not provider fallback.\n"
        "- Do not select paths that weaken ICP fit, remove constraints as the primary mechanism, or merely clean up code.\n"
        "- If the ticket names a concrete file/path/function that is not present in runtime_source_index, do not invent it or silently translate it. Return no_new_safe_path=true and list only the exact missing path/identifier in unresolved_references so the gateway can perform one bounded clarification pass.\n"
        "- A provider endpoint or text model absent from source inventory is viable only when approved_provider_capabilities explicitly permits its provider family/policy and runtime_source_index shows an existing editable provider module.\n"
        "- Capability expansion may edit that existing provider module only; never introduce a new host, credential/env reference, dependency, or network client.\n"
        "- candidate_edit_constraints is binding. Never require a new file. If it reports no editable test files, use validation_mode=runtime_checks and do not require adding tests.\n"
        "- success_criteria must describe observable behavior and existing validation, not repository work such as creating a test file.\n"
        "- Return exactly Context JSON required_root_branch_count independently safe, source-feasible ranked paths whenever that many genuinely distinct mechanisms exist. Return fewer only when another safe independent mechanism does not exist; never duplicate or paraphrase one path to fill the count.\n"
        "- Every schema_version 1.1 ranked path must carry its own target_behavior, must_inspect, allowed_lanes, disallowed_lanes, must_not_try, success_criteria, novelty_requirements, anti_overfit_checks, validation_mode, and validation_paths.\n"
        "- The ranked path named by selected_path_id is authoritative. Copy its lane, mechanism, target_behavior, must_inspect, lane constraints, safety constraints, success criteria, novelty requirements, anti-overfit checks, validation_mode, and validation_paths exactly into the duplicate top-level fields.\n"
        "- Prefer canonical source references as file.py::qualified.symbol. file.py:symbol, an exact bare file, and a uniquely resolvable bare symbol are accepted only when they bind to the extracted source index.\n"
        "- validation_mode must be runtime_checks or existing_test_files. existing_test_files requires exact paths from candidate_edit_constraints.editable_test_paths.\n"
        "- Do not request, reveal, or store secrets, hidden ICP plaintext, judge prompts, provider keys, private repo URLs, or raw private data.\n"
        "- If no safe new path exists, return no_new_safe_path=true with a clear reason.\n\n"
        "Required output shape (the selected path and duplicate top-level fields match exactly):\n"
        + json.dumps(example_plan, sort_keys=True, separators=(",", ":"))
        + "\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_loop_direction_reference_repair_messages(
    *,
    ticket: Mapping[str, Any],
    original_plan: Mapping[str, Any],
    reference_resolution: Mapping[str, Any],
    candidate_edit_constraints: Mapping[str, Any] | None = None,
    feasibility_errors: Sequence[str] = (),
) -> list[dict[str, str]]:
    """Ask the planner once to repair a plan bound to missing source names."""

    context: dict[str, Any] = {
        "ticket": _redacted_mapping(ticket),
        "original_plan": _redacted_mapping(original_plan),
        "reference_resolution": _redacted_source_context(reference_resolution),
        "allowed_lanes": list(LOOP_DIRECTION_ALLOWED_LANES),
        "candidate_edit_constraints": _redacted_mapping(candidate_edit_constraints or {}),
        "feasibility_errors": [str(item)[:300] for item in feasibility_errors[:12]],
    }
    system = (
        "You are Leadpoet Research Lab's loop-direction planner repairing one plan "
        "whose file or function references did not bind to the exact extracted model source. "
        "You do not write code. Preserve the miner's objective and return one complete replacement plan."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Use only paths and symbols present in reference_resolution matches or the original plan's valid paths. "
        "Do not invent an API, path, function, credential, dependency, host, or network client. "
        "A similar symbol is evidence to inspect, not permission to change the miner's requested mechanism. "
        "candidate_edit_constraints and feasibility_errors are binding. Never require a new file, and when no editable test file exists use validation_mode=runtime_checks. "
        "If the objective still cannot be implemented safely, return no_new_safe_path=true with exact "
        "identifier/path values in unresolved_references. Otherwise return no_new_safe_path=false and a "
        "complete schema_version 1.1 plan with required_lane, required_mechanism, selected_path_id, ranked_paths, target_behavior, "
        "must_inspect, allowed_lanes, disallowed_lanes, must_not_try, success_criteria, novelty_requirements, "
        "anti_overfit_checks, validation_mode, validation_paths, generalization_claim, and novelty_contrast. "
        "Every ranked path must include the same complete path-local fields. The selected ranked path is authoritative: "
        "copy all duplicate top-level lane, mechanism, behavior, source-reference, safety, novelty, and validation fields "
        "exactly from that path. Prefer canonical source references as file.py::qualified.symbol. "
        "unresolved_references must then be empty.\n\n"
        "Never expose secrets, private ICP text, raw provider bodies, hidden prompts, or private repository URLs.\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_code_edit_source_inspection_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    runtime_source_index: Mapping[str, Any],
    source_inspection_context: Mapping[str, Any] | None,
    budget_context: Mapping[str, Any] | None,
    loop_direction_plan: Mapping[str, Any] | None = None,
    max_requests: int = 4,
    source_access_v2: bool | None = None,
    provider_probe_catalog: Mapping[str, Any] | None = None,
    provider_outcome_digest: Mapping[str, Any] | None = None,
    provider_capability_summary: Mapping[str, Any] | None = None,
    candidate_edit_constraints: Mapping[str, Any] | None = None,
    inspection_round: int = 1,
    max_inspection_rounds: int = 1,
) -> list[dict[str, str]]:
    """Ask the model which extracted source files it needs before drafting.

    ``source_access_v2`` lets the gateway pass its builder-config decision so
    the prompt and the gateway resolver flip together; ``None`` falls back to
    this process's env flag. ``provider_probe_catalog`` (W4) advertises the
    metered ``probe_provider`` operation with its typed endpoint catalog and
    remaining budget; ``None`` keeps the operation entirely un-advertised.
    """

    probes_enabled = bool(provider_probe_catalog)
    allowed_operations = ["search", "read_file", "finish"]
    if probes_enabled:
        allowed_operations.insert(2, "probe_provider")
    resolved_max_rounds = max(1, int(max_inspection_rounds))
    resolved_round = min(resolved_max_rounds, max(1, int(inspection_round)))
    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "runtime_source_index": _redacted_source_context(runtime_source_index),
        "source_inspection_context": _redacted_source_context(source_inspection_context or {}),
        "budget_context": _redacted_mapping(budget_context or {}),
        "loop_direction_plan": _redacted_mapping(loop_direction_plan or {}),
        "max_requests": max(1, int(max_requests)),
        "allowed_operations": allowed_operations,
        "candidate_edit_constraints": _redacted_mapping(candidate_edit_constraints or {}),
        "inspection_round": resolved_round,
        "max_inspection_rounds": resolved_max_rounds,
        "remaining_inspection_rounds": max(0, resolved_max_rounds - resolved_round),
        "is_final_inspection_round": resolved_round >= resolved_max_rounds,
    }
    if probes_enabled:
        context["provider_probe_catalog"] = _redacted_mapping(provider_probe_catalog or {})
    if provider_outcome_digest:
        context["provider_outcome_digest"] = _redacted_mapping(provider_outcome_digest)
    if provider_capability_summary:
        context["approved_provider_capabilities"] = _redacted_mapping(provider_capability_summary)
    system = (
        "You are Leadpoet Research Lab's source-inspection planner for code-edit "
        "autoresearch. You are inspecting the private sourcing model runtime extracted "
        "from the current ECR image. You cannot use external tools or GitHub. Request "
        "only local searches or exact file reads that are necessary to produce a small, "
        "generalizable improvement patch later. Never request secrets, hidden benchmark "
        "plaintext, judge prompts, provider keys, raw private data, or environment files."
    )
    ranged_read_shape = ""
    ranged_read_rules = ""
    ranged_reads_enabled = _source_access_v2_enabled() if source_access_v2 is None else bool(source_access_v2)
    if ranged_reads_enabled:
        ranged_read_shape = (
            "{\"requests\":[{\"operation\":\"read_file\",\"path\":\"sourcing_model/foo.py\","
            "\"start_line\":400,\"max_lines\":200,\"rationale\":\"...\"}]}\n"
        )
        ranged_read_rules = (
            "- Large files may be read in ranges: pass optional start_line (1-based) and max_lines on read_file.\n"
            "- If an earlier read_file result was truncated, request the deeper range you still need instead of guessing.\n"
            "- After a search match, read a range around the matched line number (e.g. start_line about 40 lines above the hit) before drafting.\n"
        )
    probe_shape = ""
    probe_rules = ""
    if probes_enabled:
        probe_shape = (
            "{\"requests\":[{\"operation\":\"probe_provider\",\"provider\":\"exa\","
            "\"endpoint\":\"exa.search\",\"params\":{\"query\":\"...\"},\"rationale\":\"...\"}]}\n"
        )
        probe_rules = (
            "- probe_provider runs ONE metered live-or-replayed provider call through the evidence proxy "
            "to test a sourcing hypothesis (e.g. does this query form return results?).\n"
            "- Only endpoints listed in provider_probe_catalog are callable; params must match the endpoint schema. No URLs.\n"
            "- Probes are budgeted (see provider_probe_catalog.budget); prefer reading source first and probe only "
            "to confirm or refute a specific provider-behavior hypothesis.\n"
            "- Never probe with company names, ICP text, or anything from the private benchmark; blocked probes still consume a request.\n"
        )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Your job in this stage is not to write a patch. Request source context first.\n\n"
        "Allowed request shapes:\n"
        "{\"requests\":[{\"operation\":\"search\",\"query\":\"...\",\"rationale\":\"...\"}]}\n"
        "{\"requests\":[{\"operation\":\"read_file\",\"path\":\"sourcing_model/foo.py\",\"rationale\":\"...\"}]}\n"
        + ranged_read_shape
        + probe_shape
        + "{\"requests\":[{\"operation\":\"finish\",\"rationale\":\"enough exact source has been read\"}]}\n\n"
        "Rules:\n"
        "- Use search to locate relevant code when the exact path is unclear.\n"
        "- Use read_file before proposing edits to any file.\n"
        + ranged_read_rules
        + probe_rules
        + "- Only request paths listed in runtime_source_index.editable_files.\n"
        "- Do not request Dockerfile, dependency files, lockfiles, env files, CI, credentials, or new files.\n"
        "- Stop with finish once you have enough exact file content to draft a narrow patch.\n"
        "- If is_final_inspection_round is true, no file has been read, and prior search results contain an editable match, request read_file for the best exact match now; do not spend the final round on another search.\n"
        "- On the final round, finish without a read only when no safe editable source path exists.\n"
        "- Prefer source related to query construction, ICP normalization, provider fallback, intent evidence, ranking, and adapter output.\n\n"
        "LoopDirectionPlan binding:\n"
        "- If loop_direction_plan is present, request source context needed to implement selected_path_id only.\n"
        "- Do not inspect files solely for disallowed_lanes or must_not_try mechanisms.\n"
        "- If the plan cannot be inspected safely from editable files, return finish with rationale 'no safe source path'.\n\n"
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
    loop_direction_plan: Mapping[str, Any] | None = None,
    max_candidates: int,
    provider_outcome_digest: Mapping[str, Any] | None = None,
    provider_capability_summary: Mapping[str, Any] | None = None,
    candidate_edit_constraints: Mapping[str, Any] | None = None,
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
    example_path_id = ""
    if isinstance(loop_direction_plan, Mapping):
        example_path_id = _string_value(
            _get_first_present(loop_direction_plan, ("selected_path_id", "selectedPathId"))
        ).strip()[:120]
    if not example_path_id:
        # Never show the literal key name as the example value: models copy it
        # verbatim and the copied placeholder then fails plan_path_id matching.
        example_path_id = "path_id_copied_from_loop_direction_plan"
    context = {
        "ticket": _redacted_mapping(ticket),
        "artifact_manifest": _redacted_mapping(artifact_manifest),
        "component_registry": _redacted_mapping(component_registry),
        "benchmark_public_summary": _redacted_mapping(benchmark_public_summary),
        "runtime_source_context": source_context,
        "source_inspection_context": inspection_context,
        "budget_context": _redacted_mapping(budget_context or {}),
        "loop_direction_plan": _redacted_mapping(loop_direction_plan or {}),
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
            "source_routing",
            "provider_fallback",
            "intent_evidence_quality",
            "company_fit_filtering",
            "openrouter_model_selection",
            "output_ranking",
        ],
        "candidate_edit_constraints": _redacted_mapping(candidate_edit_constraints or {}),
    }
    if provider_outcome_digest:
        context["provider_outcome_digest"] = _redacted_mapping(provider_outcome_digest)
    if provider_capability_summary:
        context["approved_provider_capabilities"] = _redacted_mapping(provider_capability_summary)
    raw_within_run_memory = (
        budget_context.get("within_run_memory")
        if isinstance(budget_context, Mapping)
        else None
    )
    tree_branch_context = (
        raw_within_run_memory.get("git_tree_branch")
        if isinstance(raw_within_run_memory, Mapping)
        else None
    )
    tree_branch_rules = ""
    if isinstance(tree_branch_context, Mapping):
        tree_branch_rules = (
            "Git-tree branch ancestry:\n"
            "- The extracted runtime source is the exact committed parent for this node.\n"
            "- Produce exactly one incremental child edit directly on that parent source.\n"
            "- Use only Context JSON budget_context.within_run_memory.git_tree_branch.parent_feedback and this branch's ancestor commitments.\n"
            "- Never recreate the run-start source, merge a sibling, or infer sibling source or feedback.\n"
            "- Parent feedback examples are anonymous. Never infer hidden identities or hard-code example-specific values.\n"
            "- Improve the parent through a general mechanism supported by its detailed feedback; do not merely optimize the aggregate score.\n\n"
        )
    system = (
        "You are Leadpoet Research Lab's code-editing auto-research engine. "
        "Your task is to improve the private sourcing model so it finds more "
        "perfect-fit companies for a supplied ICP plus observable buying-intent "
        "signals. You may propose small source, prompt, or model logic edits only "
        "inside the runtime extracted from the current ECR image. Optimize for "
        "true sourcing quality across future sealed ICPs, "
        "not one visible ICP, public benchmark quirk, or score-looking shortcut. Never request, infer, reveal, or store secrets, hidden "
        "benchmark plaintext, judge prompts, provider keys, private repo URLs, or "
        "customer-private data."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Choose one improvement lane per candidate: ICP normalization, query construction, "
        "source routing, provider fallback, intent evidence quality, company fit filtering, "
        "OpenRouter model selection, or output ranking.\n\n"
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
        "- provider hosts, routes, or text models not permitted by Context JSON approved_provider_capabilities\n"
        "- new credential/env references even for an approved provider\n"
        "- copying private provider aliases, endpoint names, hosts, or model IDs into redacted_summary or other public-facing prose; describe the mechanism generically\n"
        "- subprocess/shell execution additions\n"
        "- hidden ICP access, raw judge prompts, raw model responses, secrets, or key handling changes\n\n"
        "Diff requirements:\n"
        "- Produce a small git unified diff that applies to the active runtime source extracted from the current ECR image.\n"
        "- The unified_diff string must begin with 'diff --git a/<path> b/<path>'.\n"
        "- Include the standard '--- a/<path>' and '+++ b/<path>' headers and valid '@@' hunks.\n"
        "- Every hunk must include at least one unchanged source line, prefixed by one space, before and after its changed lines.\n"
        "- Do not use Codex/apply_patch syntax. Never include '*** Begin Patch', '*** Update File', or '*** End Patch'.\n"
        "- Build every hunk from exact source lines visible in source_inspection_context read_file results.\n"
        "- If a read_file result is truncated, edit only the visible excerpt, or inspect a narrower relevant file in the next iteration.\n"
        "- Do not guess function bodies, line numbers, imports, or context lines that are not visible in source_inspection_context.\n"
        "- Keep the change testable and reversible.\n"
        "- Prefer one narrow code path over broad rewrites.\n"
        "- Do not overfit to one public ICP; the improvement must generalize to future sealed ICPs.\n"
        "- Treat public benchmark performance as a noisy validation signal, not the objective.\n"
        "- Do not improve apparent score by returning fewer companies, selecting only one safest company, weakening ICP constraints, removing hard filters without replacement, or hiding failures.\n"
        "- Do not remove ICP constraints unless replacing them with a more faithful constraint or adding a compensating downstream validation/filter.\n"
        "- If relaxing a brittle query constraint, preserve the semantic constraint elsewhere.\n"
        "- Provider fallback changes must classify, retry, timeout, log, or route provider/runtime failures; pure query wording changes are not provider fallback.\n\n"
        + tree_branch_rules
        +
        "LoopDirectionPlan binding:\n"
        "- If loop_direction_plan is present, it is binding.\n"
        "- Only emit candidates that directly implement loop_direction_plan.required_lane, required_mechanism, and selected_path_id.\n"
        "- Set plan_path_id to the exact value of loop_direction_plan.selected_path_id, not the literal text 'selected_path_id'.\n"
        "- If loop_direction_plan.validation_mode is runtime_checks, validate through the existing runtime checks; do not refuse solely because no editable test file exists.\n"
        "- If no safe patch can directly implement the plan from read source files, return {\"no_viable_patch\":true,\"failure_class\":\"binding_plan_unimplementable|insufficient_source_context|unsafe_or_out_of_scope|provider_probe_refuted_hypothesis|no_safe_patch\",\"reason\":\"...\",\"missing_references\":[]}.\n"
        "- Do not emit a plausible unrelated cleanup or switch lanes.\n"
        "- Do not claim alignment in prose while the diff implements another change.\n\n"
        "Expected output shape:\n"
        "{\"candidates\":[{\"lane\":\"query_construction\",\"plan_path_id\":\"" + example_path_id + "\","
        "\"plan_alignment\":{\"implements_required_mechanism\":true,\"alignment_summary\":\"...\","
        "\"success_criteria_addressed\":[\"...\"]},\"expected_metric_effect\":{"
        "\"sealed_icp_generalization\":\"...\",\"company_count\":\"...\","
        "\"provider_error_rate\":\"...\",\"precision_recall_tradeoff\":\"...\"},"
        "\"hypothesis\":{\"failure_mode\":\"...\","
        "\"mechanism\":\"...\",\"expected_improvement\":\"...\",\"risk\":\"...\","
        "\"predicted_delta\":1.0},\"code_edit\":{\"target_files\":[\"" + example_target + "\"],"
        "\"unified_diff\":\"diff --git a/" + example_target + " b/" + example_target + "\\n--- a/" + example_target + "\\n+++ b/" + example_target + "\\n@@ ...\","
        "\"redacted_summary\":\"...\","
        "\"test_plan\":\"...\",\"rollback_plan\":\"...\"}}]}\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_code_edit_fallback_messages(
    *,
    ticket: Mapping[str, Any],
    artifact_manifest: Mapping[str, Any],
    component_registry: Mapping[str, Any],
    benchmark_public_summary: Mapping[str, Any],
    runtime_source_context: Mapping[str, Any] | None = None,
    source_inspection_context: Mapping[str, Any] | None = None,
    budget_context: Mapping[str, Any] | None,
    loop_direction_plan: Mapping[str, Any] | None = None,
    fallback_reason: str,
    max_candidates: int = 1,
    max_target_files: int = 1,
    candidate_edit_constraints: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build one bounded fallback pass after the first candidate draft fails.

    The output shape intentionally matches ``build_code_edit_auto_research_messages``
    so the same parser, source-context gates, patch repair, build, scoring,
    reimbursement, and ledger paths remain in force. ``max_target_files`` > 1
    allows a bounded multi-file patch when the required contract change
    crosses modules — a strict one-file rule forced refusals on fixes that
    legitimately span discovery/adapter/client boundaries.
    """

    bounded_target_files = max(1, int(max_target_files))
    fallback_context = {
        **dict(budget_context or {}),
        "candidate_generation_fallback": {
            "schema_version": "1.0",
            "mode": "smaller_same_lane_inspected_files",
            "fallback_reason": str(fallback_reason or "")[:500],
            "max_candidates": 1,
            "max_target_files": bounded_target_files,
            "preserve_loop_direction_plan": bool(loop_direction_plan),
        },
    }
    messages = build_code_edit_auto_research_messages(
        ticket=ticket,
        artifact_manifest=artifact_manifest,
        component_registry=component_registry,
        benchmark_public_summary=benchmark_public_summary,
        runtime_source_context=runtime_source_context,
        source_inspection_context=source_inspection_context,
        budget_context=fallback_context,
        loop_direction_plan=loop_direction_plan,
        max_candidates=min(1, max(1, int(max_candidates))),
        candidate_edit_constraints=candidate_edit_constraints,
    )
    user = messages[-1]["content"]
    messages[-1] = {
        **messages[-1],
        "content": (
            user
            + "\n\nBounded fallback pass:\n"
            "- This is the only automatic fallback candidate-generation pass for this iteration.\n"
            + (
                "- Produce at most one candidate and target exactly one file from source_inspection_context.read_files.\n"
                if bounded_target_files <= 1
                else (
                    f"- Produce at most one candidate targeting at most {bounded_target_files} files "
                    "from source_inspection_context.read_files. Prefer one file; span multiple files "
                    "ONLY when the required contract change crosses module boundaries.\n"
                )
            )
            + "- Keep the same lane and same LoopDirectionPlan binding when a plan is present.\n"
            "- Do not add new files or broaden the target file set beyond the inspected files.\n"
            "- Prefer the smallest behavior-preserving patch that directly addresses the failure mode.\n"
            + (
                "- If no one-file inspected-source patch is safe, return the structured no_viable_patch shape with failure_class, reason, and missing_references.\n"
                if bounded_target_files <= 1
                else "- If no bounded inspected-source patch is safe, return the structured no_viable_patch shape with failure_class, reason, and missing_references.\n"
            )
        ),
    }
    return messages


def build_plan_alignment_judge_messages(
    *,
    loop_direction_plan: Mapping[str, Any],
    draft: CodeEditDraft,
    prior_attempts: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    context = {
        "loop_direction_plan": _redacted_mapping(loop_direction_plan),
        "candidate": {
            "lane": draft.lane,
            "plan_path_id": draft.plan_path_id,
            "target_files": list(draft.target_files),
            "unified_diff": draft.unified_diff,
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
            "expected_metric_effect": dict(draft.expected_metric_effect or {}),
        },
        "prior_attempts": _redacted_mapping({"attempts": list(prior_attempts or [])}).get("attempts", []),
    }
    system = (
        "You are a strict Research Lab plan-alignment judge. Decide whether the "
        "candidate diff directly implements the selected LoopDirectionPlan path. "
        "Reject unrelated cleanups, lane swaps, repeated mechanisms, and edits that "
        "weaken ICP constraints when the plan requires preserving them."
    )
    user = (
        "Return strict JSON only, no markdown.\n\n"
        "Fail if:\n"
        "- The diff implements a different lane than the plan.\n"
        "- The diff only changes query wording when the plan requires provider fallback, retry, or error handling.\n"
        "- The diff repeats a prior target/mechanism or exact diff.\n"
        "- The hypothesis claims alignment but the code does not.\n"
        "- The patch weakens ICP constraints when the plan requires preserving them.\n\n"
        "Required output shape (set \"verdict\" to exactly \"pass\" or exactly \"fail\"):\n"
        "{\"schema_version\":\"1.0\",\"verdict\":\"pass\",\"reason\":\"...\","
        "\"detected_lane\":\"...\",\"detected_mechanism\":\"...\",\"novel\":true,"
        "\"blocking_issue\":\"\",\"confidence\":0.0}\n\n"
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
            "expected_metric_effect": dict(draft.expected_metric_effect or {}),
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
        "- Reserve enough output budget for the final JSON; never return reasoning without a JSON repair payload.\n"
        "- If you cannot safely restate the hypothesis, return a direct repair object instead: "
        "{\"code_edit\":{\"target_files\":[...],\"unified_diff\":\"diff --git ...\","
        "\"redacted_summary\":\"...\",\"test_plan\":\"...\",\"rollback_plan\":\"...\"}}.\n"
        "- Include exactly one repaired code edit.\n"
        "- The unified_diff must start with 'diff --git a/<path> b/<path>'; no prose.\n"
        "- Include valid '--- a/<path>' and '+++ b/<path>' headers and '@@' hunks.\n"
        "- Every hunk must include at least one unchanged source line, prefixed by one space, before and after its changed lines.\n"
        "- Do not use Codex/apply_patch syntax. Never include '*** Begin Patch', '*** Update File', or '*** End Patch'.\n"
        "- Do not create new files.\n"
        "- Do not edit dependency, Docker, CI, env, credential, or lock files.\n"
        "- Do not include secrets, hidden ICPs, judge prompts, or provider keys.\n\n"
        "Context JSON:\n"
        + json.dumps(context, sort_keys=True, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_loop_direction_plan_response(raw_text: str) -> LoopDirectionPlan:
    decoded = _decode_json_value(raw_text)
    if not isinstance(decoded, Mapping):
        raise ValueError("loop direction plan response must be a JSON object")
    plan_mapping = _mapping_with_any_keys(
        decoded,
        (
            "loop_direction_plan",
            "loopDirectionPlan",
            "schema_version",
            "required_lane",
            "requiredLane",
            "selected_path_id",
            "selectedPathId",
            "no_new_safe_path",
            "noNewSafePath",
        ),
    )
    if not isinstance(plan_mapping, Mapping):
        raise ValueError("loop direction plan response did not contain a plan object")
    if isinstance(plan_mapping.get("loop_direction_plan"), Mapping):
        plan_mapping = plan_mapping["loop_direction_plan"]
    elif isinstance(plan_mapping.get("loopDirectionPlan"), Mapping):
        plan_mapping = plan_mapping["loopDirectionPlan"]
    return loop_direction_plan_from_mapping(plan_mapping)


def loop_direction_plan_from_mapping(value: Mapping[str, Any]) -> LoopDirectionPlan:
    if _contains_forbidden_material(value):
        raise ValueError("loop direction plan contains forbidden private or secret material")
    no_new_safe_path = bool(_get_first_present(value, ("no_new_safe_path", "noNewSafePath", "no_viable_path", "noViablePath")) or False)
    schema_version = _string_value(value.get("schema_version") or value.get("schemaVersion") or "1.0")[:20]
    required_lane = _string_value(_get_first_present(value, ("required_lane", "requiredLane", "lane"))).strip()[:80]
    required_mechanism = _string_value(
        _get_first_present(value, ("required_mechanism", "requiredMechanism", "mechanism"))
    )[:1200]
    selected_path_id = _string_value(
        _get_first_present(value, ("selected_path_id", "selectedPathId", "path_id", "pathId"))
    ).strip()[:120]
    ranked_paths = tuple(
        _compact_mapping(item, max_string=500)
        for item in _coerce_mapping_list(_get_first_present(value, ("ranked_paths", "rankedPaths", "paths")))
    )
    if not selected_path_id and len(ranked_paths) == 1:
        selected_path_id = _string_value(
            _get_first_present(ranked_paths[0], ("path_id", "pathId", "id"))
        ).strip()[:120]
    unresolved_references = _coerce_reference_tuple(
        _get_first_present(value, ("unresolved_references", "unresolvedReferences", "missing_references"))
    )
    selected_path = next(
        (
            item
            for item in ranked_paths
            if _string_value(_get_first_present(item, ("path_id", "pathId", "id"))).strip()
            == selected_path_id
        ),
        None,
    )
    authoritative = selected_path if schema_version == "1.1" and not no_new_safe_path else value
    if isinstance(authoritative, Mapping) and authoritative is selected_path:
        required_lane = _string_value(
            _get_first_present(authoritative, ("lane", "required_lane", "requiredLane"))
        ).strip()[:80]
        required_mechanism = _string_value(
            _get_first_present(authoritative, ("mechanism", "required_mechanism", "requiredMechanism"))
        )[:1200]
    validation_mode = _string_value(
        _get_first_present(authoritative, ("validation_mode", "validationMode")) or "runtime_checks"
    ).strip().lower()
    if validation_mode not in LOOP_DIRECTION_VALIDATION_MODES:
        raise ValueError("loop direction plan validation_mode is invalid")
    validation_paths = _coerce_target_files(
        _get_first_present(authoritative, ("validation_paths", "validationPaths", "test_paths", "testPaths"))
    )
    if validation_mode == "existing_test_files" and not validation_paths and not no_new_safe_path:
        raise ValueError("existing_test_files validation requires validation_paths")
    if not no_new_safe_path:
        if not required_lane:
            raise ValueError("loop direction plan requires required_lane")
        if not required_mechanism.strip():
            raise ValueError("loop direction plan requires required_mechanism")
        if not selected_path_id:
            raise ValueError("loop direction plan requires selected_path_id")
        if unresolved_references:
            raise ValueError("viable loop direction plan cannot retain unresolved references")
    allowed_lanes = _coerce_string_tuple(
        _get_first_present(authoritative, ("allowed_lanes", "allowedLanes")), max_items=10
    )
    if schema_version != "1.1" and required_lane and required_lane not in allowed_lanes:
        allowed_lanes = (required_lane, *tuple(item for item in allowed_lanes if item != required_lane))
    return LoopDirectionPlan(
        schema_version=schema_version,
        miner_focus_interpretation=_string_value(
            _get_first_present(value, ("miner_focus_interpretation", "minerFocusInterpretation", "focus_interpretation"))
        )[:1200],
        loop_goal=_string_value(_get_first_present(value, ("loop_goal", "loopGoal", "goal")))[:1200],
        required_lane=required_lane,
        required_mechanism=required_mechanism,
        generalization_claim=_string_value(
            _get_first_present(value, ("generalization_claim", "generalizationClaim", "sealed_icp_generalization", "sealedIcpGeneralization"))
        )[:1200],
        target_behavior=_coerce_string_tuple(
            _get_first_present(authoritative, ("target_behavior", "targetBehavior")), max_items=12
        ),
        must_inspect=_coerce_string_tuple(
            _get_first_present(authoritative, ("must_inspect", "mustInspect")), max_items=12
        ),
        allowed_lanes=allowed_lanes,
        disallowed_lanes=_coerce_string_tuple(
            _get_first_present(authoritative, ("disallowed_lanes", "disallowedLanes")), max_items=12
        ),
        must_not_try=_coerce_string_tuple(
            _get_first_present(authoritative, ("must_not_try", "mustNotTry")), max_items=16
        ),
        success_criteria=_coerce_string_tuple(
            _get_first_present(authoritative, ("success_criteria", "successCriteria")), max_items=16
        ),
        novelty_requirements=_coerce_string_tuple(
            _get_first_present(authoritative, ("novelty_requirements", "noveltyRequirements")),
            max_items=16,
        ),
        anti_overfit_checks=_coerce_string_tuple(
            _get_first_present(authoritative, ("anti_overfit_checks", "antiOverfitChecks", "overfit_checks", "overfitChecks")),
            max_items=12,
        ),
        novelty_contrast=_string_value(
            _get_first_present(value, ("novelty_contrast", "noveltyContrast", "prior_attempt_contrast", "priorAttemptContrast"))
        )[:1200],
        ranked_paths=ranked_paths,
        selected_path_id=selected_path_id,
        no_new_safe_path=no_new_safe_path,
        reason=_string_value(_get_first_present(value, ("reason", "rationale", "why")))[:1000],
        unresolved_references=unresolved_references,
        validation_mode=validation_mode,
        validation_paths=validation_paths,
    )


def loop_direction_plan_contract_errors(plan: LoopDirectionPlan) -> list[str]:
    """Validate the strict v1.1 path contract without rejecting v1.0 checkpoints."""

    if str(plan.schema_version or "1.0") != "1.1" or plan.no_new_safe_path:
        return []
    errors: list[str] = []
    if not plan.ranked_paths:
        errors.append("loop_direction_plan_v1_1_requires_ranked_paths")
        return errors
    if len(plan.ranked_paths) > 8:
        errors.append("loop_direction_plan_v1_1_allows_at_most_eight_ranked_paths")
    selected: Mapping[str, Any] | None = None
    seen_ids: set[str] = set()
    required_path_fields = {
        "target_behavior": ("target_behavior", "targetBehavior"),
        "must_inspect": ("must_inspect", "mustInspect"),
        "allowed_lanes": ("allowed_lanes", "allowedLanes"),
        "disallowed_lanes": ("disallowed_lanes", "disallowedLanes"),
        "must_not_try": ("must_not_try", "mustNotTry"),
        "success_criteria": ("success_criteria", "successCriteria"),
        "novelty_requirements": ("novelty_requirements", "noveltyRequirements"),
        "anti_overfit_checks": (
            "anti_overfit_checks",
            "antiOverfitChecks",
            "overfit_checks",
            "overfitChecks",
        ),
    }
    for raw_path in plan.ranked_paths:
        path_id = _string_value(_get_first_present(raw_path, ("path_id", "pathId", "id"))).strip()
        lane = _string_value(_get_first_present(raw_path, ("lane", "required_lane", "requiredLane"))).strip()
        mechanism = _string_value(
            _get_first_present(raw_path, ("mechanism", "required_mechanism", "requiredMechanism"))
        ).strip()
        if not path_id:
            errors.append("ranked_path_missing_path_id")
        elif path_id in seen_ids:
            errors.append(f"ranked_path_duplicate_path_id:{path_id[:120]}")
        else:
            seen_ids.add(path_id)
        if not lane:
            errors.append(f"ranked_path_missing_lane:{path_id[:120]}")
        if not mechanism:
            errors.append(f"ranked_path_missing_mechanism:{path_id[:120]}")
        for field_name, aliases in required_path_fields.items():
            field_present = any(alias in raw_path for alias in aliases)
            field_values = _coerce_string_tuple(
                _get_first_present(raw_path, aliases),
                max_items=16,
            )
            if not field_present or (field_name != "disallowed_lanes" and not field_values):
                errors.append(f"ranked_path_missing_{field_name}:{path_id[:120]}")
        path_validation_mode = _string_value(
            _get_first_present(raw_path, ("validation_mode", "validationMode"))
        ).strip().lower()
        if path_validation_mode not in LOOP_DIRECTION_VALIDATION_MODES:
            errors.append(f"ranked_path_invalid_validation_mode:{path_id[:120]}")
        path_validation_paths = _coerce_target_files(
            _get_first_present(raw_path, ("validation_paths", "validationPaths", "test_paths", "testPaths"))
        )
        if not any(
            key in raw_path
            for key in ("validation_paths", "validationPaths", "test_paths", "testPaths")
        ):
            errors.append(f"ranked_path_missing_validation_paths:{path_id[:120]}")
        if path_validation_mode == "existing_test_files" and not path_validation_paths:
            errors.append(f"ranked_path_existing_tests_missing_paths:{path_id[:120]}")
        path_allowed_lanes = _coerce_string_tuple(
            _get_first_present(raw_path, ("allowed_lanes", "allowedLanes")), max_items=10
        )
        if lane and lane not in path_allowed_lanes:
            errors.append(f"ranked_path_lane_not_allowed:{path_id[:120]}")
        if path_id == plan.selected_path_id:
            selected = raw_path
    if selected is None:
        errors.append("selected_path_id_not_in_ranked_paths")
        return errors
    selected_lane = _string_value(
        _get_first_present(selected, ("lane", "required_lane", "requiredLane"))
    ).strip()
    selected_mechanism = _string_value(
        _get_first_present(selected, ("mechanism", "required_mechanism", "requiredMechanism"))
    ).strip()
    selected_validation_mode = _string_value(
        _get_first_present(selected, ("validation_mode", "validationMode"))
    ).strip().lower()
    selected_validation_paths = _coerce_target_files(
        _get_first_present(selected, ("validation_paths", "validationPaths", "test_paths", "testPaths"))
    )
    if selected_lane != plan.required_lane:
        errors.append("selected_path_lane_mismatch")
    if selected_mechanism != plan.required_mechanism:
        errors.append("selected_path_mechanism_mismatch")
    if selected_validation_mode != plan.validation_mode:
        errors.append("selected_path_validation_mode_mismatch")
    if selected_validation_paths != plan.validation_paths:
        errors.append("selected_path_validation_paths_mismatch")
    mirrored_fields = {
        "target_behavior": (plan.target_behavior, ("target_behavior", "targetBehavior")),
        "must_inspect": (plan.must_inspect, ("must_inspect", "mustInspect")),
        "allowed_lanes": (plan.allowed_lanes, ("allowed_lanes", "allowedLanes")),
        "disallowed_lanes": (plan.disallowed_lanes, ("disallowed_lanes", "disallowedLanes")),
        "must_not_try": (plan.must_not_try, ("must_not_try", "mustNotTry")),
        "success_criteria": (plan.success_criteria, ("success_criteria", "successCriteria")),
        "novelty_requirements": (
            plan.novelty_requirements,
            ("novelty_requirements", "noveltyRequirements"),
        ),
        "anti_overfit_checks": (
            plan.anti_overfit_checks,
            ("anti_overfit_checks", "antiOverfitChecks", "overfit_checks", "overfitChecks"),
        ),
    }
    for field_name, (top_level_value, aliases) in mirrored_fields.items():
        selected_value = _coerce_string_tuple(
            _get_first_present(selected, aliases),
            max_items=16,
        )
        if selected_value != top_level_value:
            errors.append(f"selected_path_{field_name}_mismatch")
    return errors


def _legacy_no_viable_failure_class(reason: str) -> str:
    """Classify pre-v1 structured refusals conservatively for compatibility."""

    text = str(reason or "").strip().lower()
    binding_markers = (
        "outside the allowed edit scope",
        "not listed in editable_files",
        "not in editable_files",
        "not present in editable",
        "not present in the visible",
        "not present in visible",
        "not present in current",
        "required file is not present",
        "required path is not present",
        "no editable",
        "no visible",
        "no source path",
        "source path does not exist",
        "no existing test file",
        "no existing test is listed",
        "no test file appears in",
        "no test file is listed",
        "test file is not listed",
        "new files are forbidden",
        "cannot add a test file",
        "no sonar provider",
        "no sonar client",
        "no sonar parse",
        "no sonar response",
        "no discover_events_via_sonar",
        "no call_sonar",
        "no parse_sonar_json",
    )
    if any(marker in text for marker in binding_markers):
        return "binding_plan_unimplementable"
    if "probe" in text and any(marker in text for marker in ("refut", "disprov", "no result")):
        return "provider_probe_refuted_hypothesis"
    if any(
        marker in text
        for marker in (
            "insufficient source context",
            "not enough source context",
            "source was not read",
            "need to inspect",
            "must inspect",
        )
    ):
        return "insufficient_source_context"
    if any(marker in text for marker in ("unsafe", "out of scope", "forbidden", "would weaken")):
        return "unsafe_or_out_of_scope"
    return "no_safe_patch"


def _sanitize_no_viable_reason(value: Any) -> str:
    text = _string_value(value).replace("\x00", " ")
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:700]


def parse_plan_alignment_judge_response(raw_text: str) -> PlanAlignmentVerdict:
    decoded = _decode_json_value(raw_text)
    if not isinstance(decoded, Mapping):
        raise ValueError("plan alignment judge response must be a JSON object")
    if _contains_forbidden_material_diff_aware(decoded):
        raise ValueError("plan alignment judge response contains forbidden private or secret material")
    verdict_mapping = _mapping_with_any_keys(
        decoded,
        ("verdict", "passes", "pass", "plan_alignment", "planAlignment", "blocking_issue", "blockingIssue"),
    )
    if not isinstance(verdict_mapping, Mapping):
        raise ValueError("plan alignment judge response did not contain a verdict object")
    if isinstance(verdict_mapping.get("plan_alignment"), Mapping):
        verdict_mapping = verdict_mapping["plan_alignment"]
    elif isinstance(verdict_mapping.get("planAlignment"), Mapping):
        verdict_mapping = verdict_mapping["planAlignment"]
    verdict_raw = _string_value(_get_first_present(verdict_mapping, ("verdict", "decision", "status"))).strip().lower()
    if not verdict_raw:
        if verdict_mapping.get("passes") is True or verdict_mapping.get("pass") is True:
            verdict_raw = "pass"
        elif verdict_mapping.get("passes") is False or verdict_mapping.get("pass") is False:
            verdict_raw = "fail"
    verdict_raw = verdict_raw.strip().strip(".!\"'` ")
    verdict = "pass" if verdict_raw in _PASS_VERDICT_SYNONYMS else "fail"
    return PlanAlignmentVerdict(
        schema_version=_string_value(verdict_mapping.get("schema_version") or verdict_mapping.get("schemaVersion") or "1.0")[:20],
        verdict=verdict,
        reason=_string_value(_get_first_present(verdict_mapping, ("reason", "rationale", "summary")))[:1200],
        detected_lane=_string_value(_get_first_present(verdict_mapping, ("detected_lane", "detectedLane", "lane")))[:120],
        detected_mechanism=_string_value(
            _get_first_present(verdict_mapping, ("detected_mechanism", "detectedMechanism", "mechanism"))
        )[:1200],
        novel=bool(_get_first_present(verdict_mapping, ("novel", "is_novel", "isNovel")) is not False),
        blocking_issue=_string_value(_get_first_present(verdict_mapping, ("blocking_issue", "blockingIssue", "issue")))[:700],
        confidence=_float_value(_get_first_present(verdict_mapping, ("confidence", "score")), default=0.0),
    )


def parse_code_edit_no_viable_patch_response(raw_text: str) -> CodeEditNoViablePatch | None:
    try:
        decoded = _decode_json_value(raw_text)
    except ValueError:
        return None
    if not isinstance(decoded, Mapping):
        return None
    item = _mapping_with_any_keys(
        decoded,
        ("no_viable_patch", "noViablePatch", "no_safe_patch", "noSafePatch", "no_new_safe_path", "noNewSafePath"),
    )
    if not isinstance(item, Mapping):
        return None
    if not any(
        item.get(key) is True
        for key in (
            "no_viable_patch",
            "noViablePatch",
            "no_safe_patch",
            "noSafePatch",
            "no_new_safe_path",
            "noNewSafePath",
        )
    ):
        return None
    if _contains_forbidden_material(item):
        raise ValueError("no-viable-patch response contains forbidden private or secret material")
    reason = _sanitize_no_viable_reason(
        _get_first_present(item, ("reason", "rationale", "why", "message"))
    )
    reason = reason or "no viable patch"
    failure_class = _string_value(
        _get_first_present(item, ("failure_class", "failureClass", "reason_class", "reasonClass"))
    ).strip().lower()
    if not failure_class:
        failure_class = _legacy_no_viable_failure_class(reason)
    if failure_class not in CODE_EDIT_NO_VIABLE_FAILURE_CLASSES:
        raise ValueError("no-viable-patch failure_class is invalid")
    missing_references = _coerce_reference_tuple(
        _get_first_present(item, ("missing_references", "missingReferences", "unresolved_references"))
    )
    return CodeEditNoViablePatch(
        schema_version=_string_value(item.get("schema_version") or item.get("schemaVersion") or "1.0")[:20],
        failure_class=failure_class,
        reason=reason,
        missing_references=missing_references,
    )


def code_edit_no_viable_patch_reason(raw_text: str) -> str:
    refusal = parse_code_edit_no_viable_patch_response(raw_text)
    return refusal.reason if refusal is not None else ""


def code_edit_plan_alignment_errors(
    draft: CodeEditDraft,
    *,
    loop_direction_plan: Mapping[str, Any] | None,
    prior_attempts: Sequence[Mapping[str, Any]] | None = None,
    strict: bool = True,
) -> list[str]:
    if not loop_direction_plan:
        return []
    try:
        plan = loop_direction_plan_from_mapping(loop_direction_plan)
    except Exception as exc:
        return [f"loop_direction_plan_invalid:{str(exc)[:120]}"]
    if plan.no_new_safe_path:
        return ["loop_direction_plan_no_new_safe_path"]
    errors: list[str] = []
    required_lane = plan.required_lane.strip().lower()
    declared_lane = str(draft.lane or "").strip().lower()
    if required_lane:
        if declared_lane and declared_lane != required_lane:
            errors.append(f"declared_lane_mismatch:{declared_lane}!={required_lane}")
        elif strict and not declared_lane:
            errors.append("declared_lane_missing")
    selected_path_id = plan.selected_path_id.strip()
    if selected_path_id:
        draft_path_id = str(draft.plan_path_id or "").strip()
        if draft_path_id.lower() in _PLAN_PATH_ID_PLACEHOLDERS:
            # Prompt-taught placeholder (older prompt showed the literal key
            # name as the example value) — treat as missing, not a mismatch.
            draft_path_id = ""
        if draft_path_id and draft_path_id != selected_path_id:
            errors.append(f"plan_path_id_mismatch:{draft_path_id}!={selected_path_id}")
        elif strict and not draft_path_id:
            errors.append("plan_path_id_missing")

    diff_hash = sha256_json({"unified_diff": draft.unified_diff})
    target_files = set(draft.target_files)
    semantic_summary = _semantic_summary_key(draft)
    for attempt in prior_attempts or ():
        if not isinstance(attempt, Mapping):
            continue
        if diff_hash and diff_hash == str(attempt.get("unified_diff_hash") or ""):
            errors.append("duplicate_unified_diff_hash")
            break
        previous_files = set(_coerce_string_tuple(attempt.get("target_files"), max_items=20))
        previous_summary = _normalize_semantic_summary(
            _string_value(
                attempt.get("semantic_edit_summary")
                or attempt.get("redacted_summary")
                or attempt.get("expected_improvement")
                or attempt.get("test_plan")
                or attempt.get("mechanism")
                or attempt.get("summary")
            )[:500]
        )
        if strict and previous_files and target_files == previous_files and semantic_summary and semantic_summary == previous_summary:
            errors.append("duplicate_target_files_and_semantic_summary")
            break

    combined = " ".join(
        [
            draft.lane,
            draft.plan_path_id,
            " ".join(draft.target_files),
            draft.failure_mode,
            draft.mechanism,
            draft.expected_improvement,
            draft.redacted_summary,
            draft.unified_diff,
        ]
    ).lower()
    implementation_text = " ".join(
        [
            " ".join(draft.target_files),
            draft.redacted_summary,
            draft.unified_diff,
        ]
    ).lower()
    if required_lane == "provider_fallback":
        fallback_terms = ("fallback", "retry", "backoff", "provider", "http", "4xx", "5xx", "zero result", "zero-result", "timeout", "error handling")
        employee_query_terms = ("employee count", "employee-count", "linkedin employee", "linkedin", "headcount")
        if any(term in implementation_text for term in employee_query_terms) and not any(term in implementation_text for term in fallback_terms):
            errors.append("provider_fallback_plan_but_employee_count_query_edit")
    if required_lane == "intent_evidence_quality":
        intent_terms = ("intent", "evidence", "citation", "source", "freshness", "signal")
        if strict and not any(term in combined for term in intent_terms):
            errors.append("intent_evidence_plan_without_intent_evidence_terms")
    return errors


DEFAULT_SOURCE_INSPECTION_OPERATIONS = ("search", "read_file", "finish")


def parse_code_edit_source_inspection_response(
    raw_text: str,
    *,
    max_requests: int = 4,
    allowed_operations: Sequence[str] | None = None,
) -> list[CodeEditSourceInspectionRequest]:
    allowed_ops = tuple(allowed_operations or DEFAULT_SOURCE_INSPECTION_OPERATIONS)
    try:
        decoded = _decode_json_value(raw_text)
    except ValueError:
        parsed = _source_inspection_requests_from_text(raw_text, max_requests=max_requests)
        if parsed:
            return parsed
        raise
    if isinstance(decoded, str):
        try:
            return parse_code_edit_source_inspection_response(
                decoded, max_requests=max_requests, allowed_operations=allowed_ops
            )
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
            "probe": "probe_provider",
            "probeprovider": "probe_provider",
            "provider_probe": "probe_provider",
        }.get(operation, operation)
        if operation not in allowed_ops:
            raise ValueError(f"unsupported source-inspection operation:{operation}")
        query = str(_get_first_present(item, ("query", "search", "pattern", "term")) or "")[:500]
        path = ""
        raw_path = _get_first_present(item, ("path", "file", "filepath", "file_path", "target", "target_file", "targetFile"))
        if raw_path is not None:
            path = _normalize_repo_path(raw_path)
        rationale = str(_get_first_present(item, ("rationale", "reason", "why", "description")) or "")[:700]
        start_line = 0
        max_lines = 0
        if operation == "read_file":
            start_line = _int_value(
                _get_first_present(item, ("start_line", "startLine", "from_line", "fromLine", "offset", "line_offset")),
                default=0,
            )
            max_lines = _int_value(
                _get_first_present(item, ("max_lines", "maxLines", "line_limit", "lineLimit", "num_lines", "numLines")),
                default=0,
            )
        provider = ""
        endpoint = ""
        params: dict[str, Any] = {}
        if operation == "probe_provider":
            provider = str(_get_first_present(item, ("provider", "provider_id", "providerId")) or "").strip()[:80]
            endpoint = str(
                _get_first_present(item, ("endpoint", "endpoint_id", "endpointId", "catalog_id", "catalogId")) or ""
            ).strip()[:120]
            raw_params = _get_first_present(item, ("params", "parameters", "arguments", "args"))
            if isinstance(raw_params, Mapping):
                for key, value in list(raw_params.items())[:12]:
                    if isinstance(value, (str, int, float, bool)):
                        params[str(key)[:80]] = value[:500] if isinstance(value, str) else value
            if not endpoint:
                raise ValueError("source-inspection probe_provider requires endpoint")
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
                start_line=max(0, start_line),
                max_lines=max(0, max_lines),
                provider=provider,
                endpoint=endpoint,
                params=params,
            )
        )
    return parsed


def parse_code_edit_response(
    raw_text: str, *, max_candidates: int = 1, max_target_files: int = 0
) -> list[CodeEditDraft]:
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
            return parse_code_edit_response(
                decoded, max_candidates=max_candidates, max_target_files=max_target_files
            )
        except ValueError:
            draft = _draft_from_diff_only_response(decoded)
            if draft is not None:
                return [draft]
            raise ValueError("code-edit response string did not contain a parseable candidate")

    if not isinstance(decoded, (Mapping, list)):
        raise ValueError("code-edit response must be a JSON object or candidate array")
    if _contains_forbidden_material_diff_aware(decoded):
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
            validate_code_edit_draft(draft, max_target_files=max_target_files)
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

    try:
        decoded = _decode_json_value(raw_text)
    except ValueError:
        # The draft parser accepts a bare unified diff; accept it here too and
        # carry the original hypothesis forward.
        diff_only = _draft_from_diff_only_response(raw_text)
        if diff_only is None:
            raise
        decoded = {"code_edit": {"unified_diff": diff_only.unified_diff, "target_files": list(diff_only.target_files)}}
    if isinstance(decoded, str):
        return parse_code_edit_repair_response(decoded, original_draft=original_draft)
    if not isinstance(decoded, (Mapping, list)):
        raise ValueError("code-edit repair response must be a JSON object or candidate array")
    if _contains_forbidden_material_diff_aware(decoded):
        raise ValueError("code-edit repair response contains forbidden private or secret material")

    # Reuse the draft parser's candidate discovery so a repair never fails on a
    # wrapper shape the first-pass parser accepts (repair parser used to be
    # stricter than the draft parser).
    candidate_items = _code_edit_candidate_items(decoded)[:1]
    if not candidate_items:
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
        code_edit = _candidate_code_edit_mapping(item)
        raw_target_files = (
            _get_first_present(
                code_edit,
                ("target_files", "targetFiles", "targets", "files", "paths", "target_file", "targetFile", "file", "path"),
            )
            or _get_first_present(
                item,
                ("target_files", "targetFiles", "targets", "files", "paths", "target_file", "targetFile", "file", "path"),
            )
        )
        target_files = _coerce_target_files(raw_target_files) or tuple(original_draft.target_files)
        unified_diff = normalize_unified_diff_text(_string_value(_candidate_unified_diff(item, code_edit)))
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
                _get_first_present(code_edit, ("redacted_summary", "redactedSummary", "summary"))
                or _get_first_present(item, ("redacted_summary", "redactedSummary", "summary"))
                or original_draft.redacted_summary
            )[:1200],
            test_plan=str(
                _get_first_present(code_edit, ("test_plan", "testPlan"))
                or _get_first_present(item, ("test_plan", "testPlan"))
                or original_draft.test_plan
            )[:1200],
            rollback_plan=str(
                _get_first_present(code_edit, ("rollback_plan", "rollbackPlan"))
                or _get_first_present(item, ("rollback_plan", "rollbackPlan"))
                or original_draft.rollback_plan
            )[:1200],
            predicted_delta=_float_value(
                hypothesis.get("predicted_delta"),
                default=float(original_draft.predicted_delta or 1.0),
            ),
            plan_path_id=str(
                item.get("plan_path_id")
                or item.get("planPathId")
                or code_edit.get("plan_path_id")
                or code_edit.get("planPathId")
                or original_draft.plan_path_id
            )[:120],
            plan_alignment=dict(
                item.get("plan_alignment")
                if isinstance(item.get("plan_alignment"), Mapping)
                else (
                    item.get("planAlignment")
                    if isinstance(item.get("planAlignment"), Mapping)
                    else original_draft.plan_alignment
                )
            ),
            expected_metric_effect=_compact_mapping(
                _mapping_or_empty(
                    _get_first_present(
                        item,
                        ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                    )
                    or _get_first_present(
                        code_edit,
                        ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                    )
                    or _get_first_present(
                        hypothesis,
                        ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                    )
                    or original_draft.expected_metric_effect
                ),
                max_string=700,
            ),
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
        plan_path_id=_string_value(
            _get_first_present(item, ("plan_path_id", "planPathId", "selected_path_id", "selectedPathId"))
            or _get_first_present(code_edit, ("plan_path_id", "planPathId", "selected_path_id", "selectedPathId"))
        )[:120],
        plan_alignment=dict(
            item.get("plan_alignment")
            if isinstance(item.get("plan_alignment"), Mapping)
            else (item.get("planAlignment") if isinstance(item.get("planAlignment"), Mapping) else {})
        ),
        expected_metric_effect=_compact_mapping(
            _mapping_or_empty(
                _get_first_present(
                    item,
                    ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                )
                or _get_first_present(
                    code_edit,
                    ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                )
                or _get_first_present(
                    hypothesis,
                    ("expected_metric_effect", "expectedMetricEffect", "metric_effect", "metricEffect"),
                )
            ),
            max_string=700,
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


def _mapping_with_any_keys(decoded: Mapping[str, Any], keys: Sequence[str], *, _depth: int = 0) -> Mapping[str, Any] | None:
    if _depth > 4:
        return None
    if any(key in decoded for key in keys):
        return decoded
    for key in ("result", "response", "output", "data", "message", "content", "json", "final", "answer", "text"):
        value = decoded.get(key)
        if isinstance(value, Mapping):
            nested = _mapping_with_any_keys(value, keys, _depth=_depth + 1)
            if nested is not None:
                return nested
        elif isinstance(value, str):
            try:
                nested_decoded = _decode_json_value(value)
            except ValueError:
                continue
            if isinstance(nested_decoded, Mapping):
                nested = _mapping_with_any_keys(nested_decoded, keys, _depth=_depth + 1)
                if nested is not None:
                    return nested
    return None


def _coerce_string_tuple(value: Any, *, max_items: int) -> tuple[str, ...]:
    if value is None:
        raw_items: list[Any] = []
    elif isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]
    items: list[str] = []
    for item in raw_items:
        text = _string_value(item).strip()
        if text and text not in items:
            items.append(text[:1200])
        if len(items) >= max(1, int(max_items)):
            break
    return tuple(items)


def _coerce_reference_tuple(value: Any) -> tuple[str, ...]:
    raw_items = list(value) if isinstance(value, (list, tuple, set)) else ([] if value is None else [value])
    references: list[str] = []
    for item in raw_items[:8]:
        text = _string_value(item).strip()
        if not text:
            continue
        if len(text) > 120:
            raise ValueError("loop direction unresolved reference exceeds 120 characters")
        if any(ord(char) < 32 or ord(char) == 127 for char in text):
            raise ValueError("loop direction unresolved reference contains control characters")
        if ".." in text or "://" in text or not re.fullmatch(r"[A-Za-z0-9_./:\-]+", text):
            raise ValueError("loop direction unresolved reference must be a path or identifier")
        if _contains_forbidden_material(text):
            raise ValueError("loop direction unresolved reference contains forbidden material")
        if text not in references:
            references.append(text)
    return tuple(references)


def _coerce_mapping_list(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    raw_items = value if isinstance(value, list) else [value]
    return [item for item in raw_items if isinstance(item, Mapping)]


def _compact_mapping(value: Mapping[str, Any], *, max_string: int) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, item in value.items():
        clean_key = str(key)[:80]
        if isinstance(item, Mapping):
            payload[clean_key] = _compact_mapping(item, max_string=max_string)
        elif isinstance(item, list):
            payload[clean_key] = [
                _compact_mapping(child, max_string=max_string) if isinstance(child, Mapping) else _string_value(child)[:max_string]
                for child in item[:20]
            ]
        elif isinstance(item, (str, int, float, bool)) or item is None:
            payload[clean_key] = item[:max_string] if isinstance(item, str) else item
        else:
            payload[clean_key] = _string_value(item)[:max_string]
    return payload


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


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


def _int_value(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _source_access_v2_enabled() -> bool:
    """Local flag read; config.py is intentionally not touched here."""

    raw = str(os.getenv("RESEARCH_LAB_SOURCE_ACCESS_V2", "true") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_semantic_summary(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    return re.sub(r"\s+", " ", text)[:240]


def _semantic_summary_key(draft: CodeEditDraft) -> str:
    """Build the draft-side semantic key in the shape prior attempts store it.

    ``gateway/research_lab/worker.py::_candidate_attempt_memory`` persists
    ``semantic_edit_summary`` as the redacted summary (falling back to
    expected_improvement, then test_plan) truncated to 500 raw characters.
    The draft-side key must be built from the same material with the same
    truncation, or the semantic-duplicate comparison never matches and the
    novelty guard is dead code (bug #22).
    """

    raw = (
        draft.redacted_summary
        or draft.expected_improvement
        or draft.test_plan
        or draft.mechanism
        or ""
    )
    return _normalize_semantic_summary(str(raw)[:500])


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
    max_target_files: int = 0,
) -> list[str]:
    errors: list[str] = []
    payload = draft.to_dict()
    if _contains_forbidden_material_diff_aware(payload):
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
    if max_target_files and len(all_paths) > int(max_target_files):
        errors.append(
            f"code_edit_too_many_target_files:{len(all_paths)}>{int(max_target_files)}"
        )
    for path in all_paths:
        errors.extend(_validate_repo_path(
            path,
            allowed_prefixes=allowed_prefixes,
            allowed_exact_paths=allowed_exact_paths,
            allowed_suffixes=allowed_suffixes,
        ))
    # Scan only added lines: context/removed lines are verbatim parent source,
    # and rejecting on them blacklists files that already use these APIs.
    added_diff_material = _diff_added_line_material(draft.unified_diff)
    for pattern in DISALLOWED_DIFF_PATTERNS:
        if re.search(pattern, added_diff_material):
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
            "plan_path_id": draft.plan_path_id,
            "plan_alignment": dict(draft.plan_alignment or {}),
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
        for key, item in value.items():
            if _sensitive_field_name(key) or _contains_secret_shaped_value(str(key)):
                return True
            if _contains_forbidden_material(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_material(item) for item in value)
    if isinstance(value, str):
        return _contains_secret_shaped_value(value) or _contains_nonpolicy_sensitive_reference(value)
    return False


def _normalized_sensitive_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _sensitive_field_name(value: Any) -> bool:
    normalized = _normalized_sensitive_name(value)
    return normalized in _SENSITIVE_FIELD_NAMES or any(
        normalized.endswith(suffix) for suffix in _SENSITIVE_FIELD_SUFFIXES
    )


def _sensitive_code_identifier(value: Any) -> bool:
    normalized = _normalized_sensitive_name(value)
    if normalized in {"secret", "token"}:
        return False
    return _sensitive_field_name(value)


def _contains_secret_shaped_value(value: str) -> bool:
    return any(pattern.search(str(value or "")) for pattern in _SECRET_VALUE_PATTERNS)


def _contains_nonpolicy_sensitive_reference(value: str) -> bool:
    """Reject claimed sensitive material while allowing instructions to avoid it."""

    for clause in re.split(r"[\n.!?;]+", str(value or "")):
        policy_text = re.sub(r"[^a-z0-9']+", " ", clause.lower()).strip()
        if not policy_text:
            continue
        for term in FORBIDDEN_CODE_EDIT_TERMS:
            phrase = re.escape(term.replace("_", " "))
            match = re.search(rf"\b{phrase}\b", policy_text)
            if match is None:
                continue
            prefix = policy_text[: match.start()]
            suffix = policy_text[match.end() :]
            if _SENSITIVE_POLICY_PREFIX_RE.search(prefix) or _SENSITIVE_POLICY_SUFFIX_RE.search(suffix):
                continue
            return True
    return False


def _looks_like_unified_diff(value: str) -> bool:
    text = str(value or "")
    stripped = text.lstrip()
    return (
        stripped.startswith(("diff --git ", "--- a/", "@@ "))
        or "diff --git " in text
        or "\n@@ " in text
        or "\n--- a/" in text
    )


def _diff_added_line_material(diff_text: str) -> str:
    """Return only the parts of a unified diff a term filter should scan.

    Context (' ') and removed ('-') lines are verbatim parent-image source, so
    scanning them rejects any edit touching a file that merely *mentions* a
    forbidden term — blacklisting exactly the files the intent/provider lanes
    target (bug #18). Only added lines can introduce new material into the
    tree; hunk section headings ('@@ ... @@ def foo') are also verbatim source
    and are excluded. File headers are kept because paths are model-chosen.
    """

    kept: list[str] = []
    for line in str(diff_text or "").splitlines():
        if line.startswith("+++ "):
            kept.append(line)
        elif line.startswith("+"):
            kept.append(line)
        elif line.startswith(("diff --git ", "--- ")):
            kept.append(line)
    return "\n".join(kept)


def _contains_forbidden_material_diff_aware(value: Any) -> bool:
    """Forbidden-term scan that ignores unified-diff context/removed lines."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if _sensitive_field_name(key) or _contains_secret_shaped_value(str(key)):
                return True
            if _contains_forbidden_material_diff_aware(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_material_diff_aware(item) for item in value)
    if isinstance(value, str):
        if _looks_like_unified_diff(value):
            return _diff_additions_contain_sensitive_material(value)
        return _contains_forbidden_material(value)
    return False


def _diff_additions_contain_sensitive_material(diff_text: str) -> bool:
    material = _diff_added_line_material(diff_text)
    if _contains_secret_shaped_value(material):
        return True
    source_lines: list[str] = []
    for raw_line in material.splitlines():
        if raw_line.startswith(("+++ ", "diff --git ", "--- ")):
            normalized_header = _normalized_sensitive_name(raw_line)
            if any(
                sensitive_name in normalized_header
                for sensitive_name in _SENSITIVE_FIELD_NAMES
                if len(sensitive_name) >= 8
            ):
                return True
            continue
        if not raw_line.startswith("+"):
            continue
        source_line = raw_line[1:]
        if not source_line.strip() or source_line.lstrip().startswith("#"):
            continue
        source_lines.append(source_line)
    tokens: list[tokenize.TokenInfo] = []
    try:
        for item in tokenize.generate_tokens(io.StringIO("\n".join(source_lines)).readline):
            tokens.append(item)
    except (IndentationError, tokenize.TokenError):
        # Keep tokens emitted before an incomplete hunk ended. Git/apply and
        # py_compile still reject malformed code later; the security scan must
        # not discard already-visible environment access.
        pass
    significant = [
        item
        for item in tokens
        if item.type
        not in {
            tokenize.ENCODING,
            tokenize.ENDMARKER,
            tokenize.INDENT,
            tokenize.DEDENT,
            tokenize.NEWLINE,
            tokenize.NL,
            tokenize.COMMENT,
        }
    ]
    for index, item in enumerate(significant):
        if item.type == tokenize.NAME and _sensitive_code_identifier(item.string):
            return True
        if item.type != tokenize.STRING:
            continue
        try:
            literal = ast.literal_eval(item.string)
        except (SyntaxError, ValueError):
            literal = ""
        if not isinstance(literal, str) or not _sensitive_field_name(literal):
            continue
        previous = significant[index - 1].string if index else ""
        following = significant[index + 1].string if index + 1 < len(significant) else ""
        if following == ":":
            return True
        if previous == "[":
            before_open = significant[index - 2] if index >= 2 else None
            if before_open is not None and (
                before_open.type == tokenize.NAME or before_open.string in {"]", ")"}
            ):
                return True
        if previous == "(":
            call_names = {
                _normalized_sensitive_name(token.string)
                for token in significant[max(0, index - 6) : index]
                if token.type == tokenize.NAME
            }
            policy_call = any(
                name.startswith(("avoid", "disallow", "forbid", "redact"))
                for name in call_names
            )
            if not policy_call:
                return True
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

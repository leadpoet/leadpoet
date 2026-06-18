"""Phase 1 Engine v1 typed-patch loop contracts.

P1.3 defines local records and validators for Engine v1. It does not call
models, edit files, run sandboxes, publish lessons, or schedule production work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .loop_foundation import verify_loop_foundation


FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "engine_v1_fixtures.json"


class PatchType(str, Enum):
    PROMPT_EDIT = "PROMPT_EDIT"
    PARAM_EDIT = "PARAM_EDIT"
    STRATEGY_SWAP = "STRATEGY_SWAP"
    SOURCE_ADD = "SOURCE_ADD"
    CODE_EDIT = "CODE_EDIT"


class EvalStatus(str, Enum):
    SCORED = "scored"
    CRASH = "crash"
    TIMEOUT = "timeout"
    GUARDRAIL_BREACH = "guardrail_breach"


class PromotionDecision(str, Enum):
    KEEP = "keep"
    DISCARD = "discard"
    SIMPLIFICATION_KEEP = "simplification_keep"


ENGINE_V1_ENABLED_PATCH_TYPES: tuple[str, ...] = (
    PatchType.PROMPT_EDIT.value,
    PatchType.PARAM_EDIT.value,
    PatchType.STRATEGY_SWAP.value,
)

DEFERRED_PATCH_TYPES: tuple[str, ...] = (
    PatchType.SOURCE_ADD.value,
    PatchType.CODE_EDIT.value,
)

METRIC_NAMES: tuple[str, ...] = (
    "proxy_score",
    "evidence_defect_rate",
    "coverage",
    "cost_per_icp_cents",
    "latency_ms",
    "schema_validity",
    "complexity_total",
)

# target_delta is reported in improvement space: positive is better.
METRIC_DIRECTIONS: dict[str, str] = {
    "proxy_score": "higher",
    "evidence_defect_rate": "lower",
    "coverage": "higher",
    "cost_per_icp_cents": "lower",
    "latency_ms": "lower",
    "schema_validity": "higher",
    "complexity_total": "lower",
}


@dataclass(frozen=True)
class ComponentManifestEntry:
    name: str
    purpose: str
    input_contract: str
    output_contract: str
    ablation_leverage: float
    allowed_patch_types: tuple[str, ...]
    token_budget: int
    cost_budget_cents: int
    prompt_required_placeholders: tuple[str, ...] = ()
    param_bounds: dict[str, tuple[float, float]] | None = None
    strategy_options: tuple[str, ...] = ()
    engine_v0_evidence_refs: tuple[str, ...] = ()
    current_patch_seq: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentManifestEntry":
        return cls(
            name=str(data["name"]),
            purpose=str(data["purpose"]),
            input_contract=str(data["input_contract"]),
            output_contract=str(data["output_contract"]),
            ablation_leverage=float(data["ablation_leverage"]),
            allowed_patch_types=tuple(str(item) for item in data.get("allowed_patch_types", [])),
            token_budget=int(data["token_budget"]),
            cost_budget_cents=int(data["cost_budget_cents"]),
            prompt_required_placeholders=tuple(str(item) for item in data.get("prompt_required_placeholders", [])),
            param_bounds={
                str(name): (float(bounds["min"]), float(bounds["max"]))
                for name, bounds in dict(data.get("param_bounds") or {}).items()
            },
            strategy_options=tuple(str(item) for item in data.get("strategy_options", [])),
            engine_v0_evidence_refs=tuple(str(item) for item in data.get("engine_v0_evidence_refs", [])),
            current_patch_seq=int(data.get("current_patch_seq", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["allowed_patch_types"] = list(self.allowed_patch_types)
        data["prompt_required_placeholders"] = list(self.prompt_required_placeholders)
        data["param_bounds"] = {
            name: {"min": bounds[0], "max": bounds[1]}
            for name, bounds in (self.param_bounds or {}).items()
        }
        data["strategy_options"] = list(self.strategy_options)
        data["engine_v0_evidence_refs"] = list(self.engine_v0_evidence_refs)
        return data


@dataclass(frozen=True)
class ComponentRegistry:
    manifest_version: str
    champion_base: str
    eval_version: str
    entries: tuple[ComponentManifestEntry, ...]
    engine_v0_receipt_refs: tuple[str, ...]
    meta_allocator_priors_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentRegistry":
        return cls(
            manifest_version=str(data["manifest_version"]),
            champion_base=str(data["champion_base"]),
            eval_version=str(data["eval_version"]),
            entries=tuple(ComponentManifestEntry.from_mapping(item) for item in data.get("entries", [])),
            engine_v0_receipt_refs=tuple(str(item) for item in data.get("engine_v0_receipt_refs", [])),
            meta_allocator_priors_enabled=bool(data.get("meta_allocator_priors_enabled", False)),
        )

    def by_name(self) -> dict[str, ComponentManifestEntry]:
        return {entry.name: entry for entry in self.entries}

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": self.manifest_version,
            "champion_base": self.champion_base,
            "eval_version": self.eval_version,
            "entries": [entry.to_dict() for entry in self.entries],
            "engine_v0_receipt_refs": list(self.engine_v0_receipt_refs),
            "meta_allocator_priors_enabled": self.meta_allocator_priors_enabled,
        }


@dataclass(frozen=True)
class HypothesisRecord:
    hypothesis_id: str
    component: str
    failure_mode: str
    mechanism: str
    patch_type: str
    predicted_delta: float
    falsifier: str
    context_pack_refs: tuple[str, ...]
    miner_brief_ref: str
    generated_by: str = "engine"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "HypothesisRecord":
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            component=str(data["component"]),
            failure_mode=str(data["failure_mode"]),
            mechanism=str(data["mechanism"]),
            patch_type=str(data["patch_type"]),
            predicted_delta=float(data["predicted_delta"]),
            falsifier=str(data["falsifier"]),
            context_pack_refs=tuple(str(item) for item in data.get("context_pack_refs", [])),
            miner_brief_ref=str(data["miner_brief_ref"]),
            generated_by=str(data.get("generated_by", "engine")),
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["context_pack_refs"] = list(self.context_pack_refs)
        return data


@dataclass(frozen=True)
class PatchRecord:
    patch_id: str
    hypothesis_id: str
    component: str
    patch_type: str
    payload: dict[str, Any]

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PatchRecord":
        return cls(
            patch_id=str(data["patch_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            component=str(data["component"]),
            patch_type=str(data["patch_type"]),
            payload=dict(data.get("payload") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComplexityMetrics:
    diff_loc: int
    prompt_pack_tokens: int
    component_count: int
    strategy_count: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComplexityMetrics":
        return cls(
            diff_loc=int(data["diff_loc"]),
            prompt_pack_tokens=int(data["prompt_pack_tokens"]),
            component_count=int(data["component_count"]),
            strategy_count=int(data["strategy_count"]),
        )

    @property
    def total(self) -> int:
        return self.diff_loc + self.prompt_pack_tokens + self.component_count + self.strategy_count

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class MetricVector:
    proxy_score: float
    evidence_defect_rate: float
    coverage: float
    cost_per_icp_cents: int
    latency_ms: int
    schema_validity: bool
    complexity: ComplexityMetrics

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MetricVector":
        return cls(
            proxy_score=float(data["proxy_score"]),
            evidence_defect_rate=float(data["evidence_defect_rate"]),
            coverage=float(data["coverage"]),
            cost_per_icp_cents=int(data["cost_per_icp_cents"]),
            latency_ms=int(data["latency_ms"]),
            schema_validity=bool(data["schema_validity"]),
            complexity=ComplexityMetrics.from_mapping(data["complexity"]),
        )

    def metric_value(self, metric: str) -> float:
        if metric == "complexity_total":
            return float(self.complexity.total)
        value = getattr(self, metric)
        return float(value) if not isinstance(value, bool) else float(int(value))

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["complexity"] = self.complexity.to_dict()
        return data


@dataclass(frozen=True)
class DevEvalGuardrails:
    max_evidence_defect_rate_increase: float = 0.02
    min_coverage: float = 0.01
    max_cost_per_icp_cents: int = 250
    max_latency_ms: int = 15000
    max_complexity_increase: int = 50
    flat_metric_tolerance: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> "DevEvalGuardrails":
        data = data or {}
        return cls(
            max_evidence_defect_rate_increase=float(
                data.get("max_evidence_defect_rate_increase", cls.max_evidence_defect_rate_increase)
            ),
            min_coverage=float(data.get("min_coverage", cls.min_coverage)),
            max_cost_per_icp_cents=int(data.get("max_cost_per_icp_cents", cls.max_cost_per_icp_cents)),
            max_latency_ms=int(data.get("max_latency_ms", cls.max_latency_ms)),
            max_complexity_increase=int(data.get("max_complexity_increase", cls.max_complexity_increase)),
            flat_metric_tolerance=float(data.get("flat_metric_tolerance", cls.flat_metric_tolerance)),
        )


@dataclass(frozen=True)
class DevEvalResult:
    node_id: str
    patch_id: str
    targeted_metric: str
    status: str
    # Directional improvement for targeted_metric; positive means better.
    target_delta: float
    promotion_decision: str
    guardrail_errors: tuple[str, ...]
    parent_metrics: MetricVector
    candidate_metrics: MetricVector

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "patch_id": self.patch_id,
            "targeted_metric": self.targeted_metric,
            "status": self.status,
            "target_delta": self.target_delta,
            "promotion_decision": self.promotion_decision,
            "guardrail_errors": list(self.guardrail_errors),
            "parent_metrics": self.parent_metrics.to_dict(),
            "candidate_metrics": self.candidate_metrics.to_dict(),
        }


@dataclass(frozen=True)
class ReflectionRecord:
    lesson_id: str
    node_id: str
    worked: str
    failed: str
    why: str
    next_question: str
    champion_base: str
    component: str
    eval_version: str
    basis_patch_seq: int
    stale_basis: bool = False
    engine_authored: bool = True
    contradicted_by: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ReflectionRecord":
        return cls(
            lesson_id=str(data["lesson_id"]),
            node_id=str(data["node_id"]),
            worked=str(data["worked"]),
            failed=str(data["failed"]),
            why=str(data["why"]),
            next_question=str(data["next_question"]),
            champion_base=str(data["champion_base"]),
            component=str(data["component"]),
            eval_version=str(data["eval_version"]),
            basis_patch_seq=int(data["basis_patch_seq"]),
            stale_basis=bool(data.get("stale_basis", False)),
            engine_authored=bool(data.get("engine_authored", True)),
            contradicted_by=data.get("contradicted_by"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BanditCell:
    component: str
    patch_type: str
    prior_weight: float = 1.0
    pull_count: int = 0
    mean_reward: float = 0.0
    meta_allocator_prior_ref: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_component_registry(registry: ComponentRegistry | Mapping[str, Any]) -> list[str]:
    if not isinstance(registry, ComponentRegistry):
        registry = ComponentRegistry.from_mapping(registry)
    errors: list[str] = []
    names: set[str] = set()
    if not registry.entries:
        errors.append("component registry requires at least one entry")
    if not registry.engine_v0_receipt_refs:
        errors.append("component registry must reference P0.6 engine-v0 evidence")
    if registry.meta_allocator_priors_enabled:
        errors.append("meta-allocator priors are deferred to Phase 2")
    for entry in registry.entries:
        if entry.name in names:
            errors.append(f"duplicate component entry: {entry.name}")
        names.add(entry.name)
        if not entry.engine_v0_evidence_refs:
            errors.append(f"{entry.name}: engine_v0_evidence_refs must not be empty")
        if entry.ablation_leverage < 0:
            errors.append(f"{entry.name}: ablation_leverage must be non-negative")
        if entry.token_budget <= 0 or entry.cost_budget_cents <= 0:
            errors.append(f"{entry.name}: token and cost budgets must be positive")
        for patch_type in entry.allowed_patch_types:
            if patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
                errors.append(f"{entry.name}: patch type {patch_type} is not enabled in Engine v1")
    return errors


def validate_hypothesis(record: HypothesisRecord | Mapping[str, Any], registry: ComponentRegistry | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, HypothesisRecord):
        record = HypothesisRecord.from_mapping(record)
    if not isinstance(registry, ComponentRegistry):
        registry = ComponentRegistry.from_mapping(registry)
    errors: list[str] = []
    component = registry.by_name().get(record.component)
    if not component:
        errors.append(f"unknown component: {record.component}")
        return errors
    if record.generated_by != "engine":
        errors.append("hypotheses must be engine-authored")
    if record.patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
        errors.append(f"patch type {record.patch_type} is not enabled in Engine v1")
    if record.patch_type not in component.allowed_patch_types:
        errors.append(f"patch type {record.patch_type} is not allowed for component {record.component}")
    if record.falsifier not in METRIC_NAMES:
        errors.append(f"falsifier must name a dev metric, got {record.falsifier}")
    if not record.context_pack_refs:
        errors.append("context_pack_refs must not be empty")
    if not record.miner_brief_ref:
        errors.append("miner_brief_ref is required")
    return errors


def validate_patch(record: PatchRecord | Mapping[str, Any], registry: ComponentRegistry | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, PatchRecord):
        record = PatchRecord.from_mapping(record)
    if not isinstance(registry, ComponentRegistry):
        registry = ComponentRegistry.from_mapping(registry)
    errors: list[str] = []
    component = registry.by_name().get(record.component)
    if not component:
        errors.append(f"unknown component: {record.component}")
        return errors
    if record.patch_type in DEFERRED_PATCH_TYPES:
        errors.append(f"patch type {record.patch_type} is deferred and not enabled in Engine v1")
        return errors
    if record.patch_type not in ENGINE_V1_ENABLED_PATCH_TYPES:
        errors.append(f"unknown patch type: {record.patch_type}")
        return errors
    if record.patch_type not in component.allowed_patch_types:
        errors.append(f"patch type {record.patch_type} is not allowed for component {record.component}")
        return errors

    if record.patch_type == PatchType.PROMPT_EDIT.value:
        errors.extend(_validate_prompt_edit(record, component))
    elif record.patch_type == PatchType.PARAM_EDIT.value:
        errors.extend(_validate_param_edit(record, component))
    elif record.patch_type == PatchType.STRATEGY_SWAP.value:
        errors.extend(_validate_strategy_swap(record, component))
    return errors


def evaluate_dev_metrics(
    *,
    node_id: str,
    patch: PatchRecord | Mapping[str, Any],
    parent_metrics: MetricVector | Mapping[str, Any],
    candidate_metrics: MetricVector | Mapping[str, Any],
    targeted_metric: str,
    guardrails: DevEvalGuardrails | Mapping[str, Any] | None = None,
) -> DevEvalResult:
    if not isinstance(patch, PatchRecord):
        patch = PatchRecord.from_mapping(patch)
    if not isinstance(parent_metrics, MetricVector):
        parent_metrics = MetricVector.from_mapping(parent_metrics)
    if not isinstance(candidate_metrics, MetricVector):
        candidate_metrics = MetricVector.from_mapping(candidate_metrics)
    if not isinstance(guardrails, DevEvalGuardrails):
        guardrails = DevEvalGuardrails.from_mapping(guardrails)

    if targeted_metric not in METRIC_NAMES:
        raise ValueError(f"unknown targeted metric: {targeted_metric}")

    target_delta = round(_metric_improvement_delta(parent_metrics, candidate_metrics, targeted_metric), 6)
    errors = _guardrail_errors(parent_metrics, candidate_metrics, guardrails)

    if not candidate_metrics.schema_validity:
        status = EvalStatus.CRASH.value
    elif candidate_metrics.coverage <= 0:
        status = EvalStatus.CRASH.value
    elif candidate_metrics.latency_ms > guardrails.max_latency_ms:
        status = EvalStatus.TIMEOUT.value
    elif errors:
        status = EvalStatus.GUARDRAIL_BREACH.value
    else:
        status = EvalStatus.SCORED.value

    if status != EvalStatus.SCORED.value:
        decision = PromotionDecision.DISCARD.value
    elif target_delta > 0:
        decision = PromotionDecision.KEEP.value
    elif (
        abs(target_delta) <= guardrails.flat_metric_tolerance
        and candidate_metrics.complexity.total < parent_metrics.complexity.total
    ):
        decision = PromotionDecision.SIMPLIFICATION_KEEP.value
    else:
        decision = PromotionDecision.DISCARD.value

    return DevEvalResult(
        node_id=node_id,
        patch_id=patch.patch_id,
        targeted_metric=targeted_metric,
        status=status,
        target_delta=target_delta,
        promotion_decision=decision,
        guardrail_errors=tuple(errors),
        parent_metrics=parent_metrics,
        candidate_metrics=candidate_metrics,
    )


def build_reflection_record(
    *,
    lesson_id: str,
    node_id: str,
    worked: str,
    failed: str,
    why: str,
    next_question: str,
    registry: ComponentRegistry,
    component: str,
) -> ReflectionRecord:
    entry = registry.by_name()[component]
    return ReflectionRecord(
        lesson_id=lesson_id,
        node_id=node_id,
        worked=worked,
        failed=failed,
        why=why,
        next_question=next_question,
        champion_base=registry.champion_base,
        component=component,
        eval_version=registry.eval_version,
        basis_patch_seq=entry.current_patch_seq,
        stale_basis=False,
        engine_authored=True,
    )


def validate_reflection_record(record: ReflectionRecord | Mapping[str, Any], registry: ComponentRegistry | Mapping[str, Any]) -> list[str]:
    if not isinstance(record, ReflectionRecord):
        record = ReflectionRecord.from_mapping(record)
    if not isinstance(registry, ComponentRegistry):
        registry = ComponentRegistry.from_mapping(registry)
    errors: list[str] = []
    if not record.engine_authored:
        errors.append("reflection lessons must be engine-authored")
    if record.champion_base != registry.champion_base:
        errors.append("lesson champion_base does not match registry")
    if record.eval_version != registry.eval_version:
        errors.append("lesson eval_version does not match registry")
    component = registry.by_name().get(record.component)
    if not component:
        errors.append(f"unknown lesson component: {record.component}")
    elif record.basis_patch_seq > component.current_patch_seq:
        errors.append("lesson basis_patch_seq cannot be in the future")
    for field in ("worked", "failed", "why", "next_question"):
        if not getattr(record, field):
            errors.append(f"lesson field {field} is required")
    return errors


def mark_lesson_staleness(record: ReflectionRecord, registry: ComponentRegistry) -> ReflectionRecord:
    component = registry.by_name().get(record.component)
    stale = bool(component and component.current_patch_seq > record.basis_patch_seq)
    return ReflectionRecord(
        **{**record.to_dict(), "stale_basis": stale}
    )


def build_uniform_prior_bandit_cells(registry: ComponentRegistry | Mapping[str, Any]) -> list[BanditCell]:
    if not isinstance(registry, ComponentRegistry):
        registry = ComponentRegistry.from_mapping(registry)
    cells = [
        BanditCell(component=entry.name, patch_type=patch_type)
        for entry in registry.entries
        for patch_type in entry.allowed_patch_types
        if patch_type in ENGINE_V1_ENABLED_PATCH_TYPES
    ]
    return sorted(cells, key=lambda cell: (cell.component, cell.patch_type))


def select_uniform_prior_cell(cells: Sequence[BanditCell | Mapping[str, Any]]) -> BanditCell:
    normalized = [
        cell if isinstance(cell, BanditCell) else BanditCell(**dict(cell))
        for cell in cells
    ]
    if not normalized:
        raise ValueError("bandit selection requires at least one cell")
    return sorted(normalized, key=lambda cell: (cell.pull_count, -cell.mean_reward, cell.component, cell.patch_type))[0]


def verify_research_lab_engine_v1(fixture_path: Path | str = FIXTURE_PATH) -> dict[str, Any]:
    foundation_summary = verify_loop_foundation()
    fixture = _load_fixture(Path(fixture_path))
    registry = ComponentRegistry.from_mapping(fixture["registry"])

    _assert(not validate_component_registry(registry), "component registry validates")
    _assert(not registry.meta_allocator_priors_enabled, "meta-allocator priors are disabled")

    hypothesis = HypothesisRecord.from_mapping(fixture["hypothesis"])
    _assert(not validate_hypothesis(hypothesis, registry), "structured hypothesis validates")

    for patch in fixture["valid_patches"]:
        _assert(not validate_patch(patch, registry), f"valid patch passes: {patch['patch_id']}")
    for patch in fixture["invalid_patches"]:
        errors = validate_patch(patch, registry)
        _assert(errors, f"invalid patch fails: {patch['patch_id']}")
        expected = patch.get("expected_error_contains")
        if expected:
            _assert(
                any(str(expected) in error for error in errors),
                f"invalid patch has expected error {expected!r}: {patch['patch_id']}",
            )

    parent = MetricVector.from_mapping(fixture["dev_eval"]["parent_metrics"])
    good = evaluate_dev_metrics(
        node_id="node:good",
        patch=fixture["valid_patches"][0],
        parent_metrics=parent,
        candidate_metrics=fixture["dev_eval"]["candidate_good"],
        targeted_metric="coverage",
        guardrails=fixture["dev_eval"]["guardrails"],
    )
    _assert(good.status == EvalStatus.SCORED.value, "good candidate scores")
    _assert(good.promotion_decision == PromotionDecision.KEEP.value, "good candidate is kept")

    simplification = evaluate_dev_metrics(
        node_id="node:simplify",
        patch=fixture["valid_patches"][1],
        parent_metrics=parent,
        candidate_metrics=fixture["dev_eval"]["candidate_simplification"],
        targeted_metric="proxy_score",
        guardrails=fixture["dev_eval"]["guardrails"],
    )
    _assert(
        simplification.promotion_decision == PromotionDecision.SIMPLIFICATION_KEEP.value,
        "simplification candidate can be kept",
    )

    cost_cut = evaluate_dev_metrics(
        node_id="node:cost-cut",
        patch=fixture["valid_patches"][1],
        parent_metrics=parent,
        candidate_metrics=fixture["dev_eval"]["candidate_lower_is_better_keep"],
        targeted_metric="cost_per_icp_cents",
        guardrails=fixture["dev_eval"]["guardrails"],
    )
    _assert(cost_cut.status == EvalStatus.SCORED.value, "lower-is-better candidate scores")
    _assert(cost_cut.target_delta > 0, "lower-is-better improvement has positive target_delta")
    _assert(cost_cut.promotion_decision == PromotionDecision.KEEP.value, "lower-is-better improvement is kept")

    cost_regression = evaluate_dev_metrics(
        node_id="node:cost-regression",
        patch=fixture["valid_patches"][1],
        parent_metrics=parent,
        candidate_metrics=fixture["dev_eval"]["candidate_lower_is_better_discard"],
        targeted_metric="cost_per_icp_cents",
        guardrails=fixture["dev_eval"]["guardrails"],
    )
    _assert(cost_regression.status == EvalStatus.SCORED.value, "lower-is-better regression still scores")
    _assert(cost_regression.target_delta < 0, "lower-is-better regression has negative target_delta")
    _assert(
        cost_regression.promotion_decision == PromotionDecision.DISCARD.value,
        "lower-is-better regression is discarded",
    )

    for case in fixture["dev_eval"]["invalid_cases"]:
        result = evaluate_dev_metrics(
            node_id=case["node_id"],
            patch=fixture["valid_patches"][0],
            parent_metrics=parent,
            candidate_metrics=case["candidate_metrics"],
            targeted_metric=case["targeted_metric"],
            guardrails=fixture["dev_eval"]["guardrails"],
        )
        _assert(result.status == case["expected_status"], f"{case['node_id']} has expected status")
        _assert(result.promotion_decision == PromotionDecision.DISCARD.value, f"{case['node_id']} discards")

    reflection = build_reflection_record(
        lesson_id="lesson:p1.3:source-router",
        node_id="node:good",
        worked="source routing improved coverage",
        failed="cost still rose slightly",
        why="Sonar-first routing covered more current evidence while Exa filled gaps.",
        next_question="Can the same routing keep coverage while lowering cost?",
        registry=registry,
        component="source_router",
    )
    _assert(not validate_reflection_record(reflection, registry), "reflection validates")

    stale_registry = ComponentRegistry.from_mapping(
        {**registry.to_dict(), "entries": [_advance_component(entry, "source_router") for entry in registry.entries]}
    )
    stale = mark_lesson_staleness(reflection, stale_registry)
    _assert(stale.stale_basis, "lesson staleness is detected after component drift")

    cells = build_uniform_prior_bandit_cells(registry)
    _assert(cells, "bandit cells exist")
    _assert(all(cell.prior_weight == 1.0 for cell in cells), "bandit cells use uniform priors")
    _assert(all(cell.meta_allocator_prior_ref is None for cell in cells), "meta allocator priors are deferred")
    selected = select_uniform_prior_cell(cells)

    return {
        "foundation_invalid_release_records": foundation_summary["invalid_release_records"],
        "components": len(registry.entries),
        "valid_patches": len(fixture["valid_patches"]),
        "invalid_patches": len(fixture["invalid_patches"]),
        "bandit_cells": len(cells),
        "selected_cell": f"{selected.component}:{selected.patch_type}",
    }


def _validate_prompt_edit(record: PatchRecord, component: ComponentManifestEntry) -> list[str]:
    errors: list[str] = []
    template = str(record.payload.get("new_template") or "")
    template_name = str(record.payload.get("template_name") or "")
    if not template_name:
        errors.append("PROMPT_EDIT requires template_name")
    if not template:
        errors.append("PROMPT_EDIT requires new_template")
    for placeholder in component.prompt_required_placeholders:
        if "{" + placeholder + "}" not in template:
            errors.append(f"PROMPT_EDIT missing required placeholder {{{placeholder}}}")
    # Defense-in-depth tripwire; full content sensitivity is enforced downstream.
    if "raw_customer_data" in template or "judge_prompt" in template:
        errors.append("PROMPT_EDIT must not include raw customer data or judge prompts")
    return errors


def _validate_param_edit(record: PatchRecord, component: ComponentManifestEntry) -> list[str]:
    param_name = str(record.payload.get("param_name") or "")
    bounds = (component.param_bounds or {}).get(param_name)
    if not bounds:
        return [f"PARAM_EDIT unknown or disallowed param: {param_name}"]
    try:
        new_value = float(record.payload["new_value"])
    except Exception:
        return ["PARAM_EDIT requires numeric new_value"]
    if new_value < bounds[0] or new_value > bounds[1]:
        return [f"PARAM_EDIT {param_name}={new_value} outside bounds [{bounds[0]}, {bounds[1]}]"]
    return []


def _validate_strategy_swap(record: PatchRecord, component: ComponentManifestEntry) -> list[str]:
    strategy = str(record.payload.get("strategy_name") or "")
    if strategy not in component.strategy_options:
        return [f"STRATEGY_SWAP strategy {strategy!r} not registered for component {component.name}"]
    return []


def _guardrail_errors(parent: MetricVector, candidate: MetricVector, guardrails: DevEvalGuardrails) -> list[str]:
    errors: list[str] = []
    if candidate.coverage < guardrails.min_coverage:
        errors.append("coverage is below minimum")
    if candidate.cost_per_icp_cents > guardrails.max_cost_per_icp_cents:
        errors.append("cost_per_icp_cents exceeds guardrail")
    if candidate.evidence_defect_rate - parent.evidence_defect_rate > guardrails.max_evidence_defect_rate_increase:
        errors.append("evidence_defect_rate regressed beyond guardrail")
    if candidate.complexity.total - parent.complexity.total > guardrails.max_complexity_increase:
        errors.append("complexity regressed beyond guardrail")
    return errors


def _metric_improvement_delta(parent: MetricVector, candidate: MetricVector, metric: str) -> float:
    parent_value = parent.metric_value(metric)
    candidate_value = candidate.metric_value(metric)
    direction = METRIC_DIRECTIONS[metric]
    if direction == "lower":
        return parent_value - candidate_value
    return candidate_value - parent_value


def _advance_component(entry: ComponentManifestEntry, component_name: str) -> dict[str, Any]:
    data = entry.to_dict()
    if entry.name == component_name:
        data["current_patch_seq"] = entry.current_patch_seq + 1
    return data


def _load_fixture(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _assert(condition: bool, label: str) -> None:
    if not condition:
        raise AssertionError(label)

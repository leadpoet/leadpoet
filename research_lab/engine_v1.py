"""Phase 1 Engine v1 typed-patch loop contracts.

P1.3 defines local records and validators for Engine v1. It does not call
models, edit files, run sandboxes, publish lessons, or schedule production work.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class PatchType(str, Enum):
    PROMPT_EDIT = "PROMPT_EDIT"
    PARAM_EDIT = "PARAM_EDIT"
    STRATEGY_SWAP = "STRATEGY_SWAP"
    SOURCE_ADD = "SOURCE_ADD"
    CODE_EDIT = "CODE_EDIT"


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
    source_evidence_refs: tuple[str, ...] = ()
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
            source_evidence_refs=tuple(str(item) for item in data.get("source_evidence_refs", [])),
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
        data["source_evidence_refs"] = list(self.source_evidence_refs)
        return data


@dataclass(frozen=True)
class ComponentRegistry:
    manifest_version: str
    champion_base: str
    eval_version: str
    entries: tuple[ComponentManifestEntry, ...]
    source_receipt_refs: tuple[str, ...]
    meta_allocator_priors_enabled: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ComponentRegistry":
        return cls(
            manifest_version=str(data["manifest_version"]),
            champion_base=str(data["champion_base"]),
            eval_version=str(data["eval_version"]),
            entries=tuple(ComponentManifestEntry.from_mapping(item) for item in data.get("entries", [])),
            source_receipt_refs=tuple(str(item) for item in data.get("source_receipt_refs", [])),
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
            "source_receipt_refs": list(self.source_receipt_refs),
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
    if not registry.source_receipt_refs:
        errors.append("component registry must reference source evidence")
    if registry.meta_allocator_priors_enabled:
        errors.append("meta-allocator priors are deferred to Phase 2")
    for entry in registry.entries:
        if entry.name in names:
            errors.append(f"duplicate component entry: {entry.name}")
        names.add(entry.name)
        if not entry.source_evidence_refs:
            errors.append(f"{entry.name}: source_evidence_refs must not be empty")
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

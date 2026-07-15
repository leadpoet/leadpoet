"""Axis-A/B call provenance: the single auditable stage→emitter mapping.

trajectoryimprovements.md P11 (as amended by the 2026-07-02 cross-audit) and
v5 §8.3: ``call_emitter`` records whether the tool call *following* an LLM
call was emitted by the model (axis-A — teaches policy) or by pipeline code
(axis-B — the model classifies while code drives). The mapping is derived per
call from control-flow semantics, never hard-coded per stream:

* Fixed-order engine stages (planner / draft / repair / judge) — code always
  chooses the next call → ``"code"``.
* Source-inspection rounds — the model emits ``search`` / ``read_file`` /
  ``finish`` operations that choose the next tool call → ``"model"``.
* The in-container champion/candidate sourcing pipeline — axis-B *by
  construction* (v5 §8.3: "the champion pipeline is axis-B"); Stage-2 trains
  on these traces directly, so ``"code"`` is the plan-correct value unless a
  candidate runtime genuinely lets model output choose the next tool call.

``teacher_model_flag`` is likewise derived from the stage's role: judge/scorer
stages act as teachers over another model's output. P19 note: the flag says
"this call played a teacher role", not "this model is ToS-cleared for
distillation" — the provider-ToS inventory binds the latter when it exists.

Keep every stage this repo emits in ``STAGE_PROVENANCE``; a pinning test
(tests/test_axis_provenance.py) asserts each value so any change is a
deliberate, reviewed decision.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterable, Iterator, Mapping

CALL_EMITTER_MODEL = "model"
CALL_EMITTER_CODE = "code"

AXIS_A = "axis_a"
AXIS_B = "axis_b"

# stage → {call_emitter, purpose, component, teacher_model_flag,
#           drives_control_flow}. ``drives_control_flow`` marks the calls the
# v5 §8.3 rollup conjuncts over (calls that decide what the pipeline does
# next, as opposed to pure post-hoc annotation).
STAGE_PROVENANCE: dict[str, dict[str, Any]] = {
    # -- hosted code-improvement loop (gateway/research_lab/code_loop_engine.py)
    "loop_planner": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "plan_next_iteration",
        "component": "code_loop_engine",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    "plan_alignment_judge": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "judge_plan_alignment",
        "component": "code_loop_engine",
        "teacher_model_flag": True,
        "drives_control_flow": True,
    },
    "code_edit_draft": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "draft_code_patch",
        "component": "code_loop_engine",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    "code_edit_repair": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "repair_code_patch",
        "component": "code_loop_engine",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    # The one genuinely agentic stage: the model emits search/read_file/finish
    # operations across rounds — model output chooses the next tool call.
    "source_inspection": {
        "call_emitter": CALL_EMITTER_MODEL,
        "purpose": "agentic_source_inspection",
        "component": "code_loop_engine",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    # -- historical prompt-loop rows (the retired runtime no longer emits these)
    "loop_iteration": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "draft_prompt_iteration",
        "component": "loop_engine",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    # -- scoring boundary (gateway/research_lab/scoring_worker.py)
    "scorer_judgment": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "score_lead_quality",
        "component": "qualification_scorer",
        "teacher_model_flag": True,
        "drives_control_flow": False,
    },
    "operator_repair": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "operator_stale_parent_repair",
        "component": "scoring_worker",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
    # -- in-container private-model runtime (research_lab/eval/private_runtime.py)
    # Axis-B by construction: the champion pipeline is code-shaped; its LLM
    # calls classify/extract while pipeline code chooses the next tool call
    # (v5 §8.3, §8.6 — Stage-2 trains on these directly).
    "incontainer_model_runtime": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "champion_pipeline_inference",
        "component": "private_model_runtime",
        "teacher_model_flag": False,
        "drives_control_flow": True,
    },
}

# Live-event types → canonical stage. The projector falls back to the
# emitting event's type when a provider-usage item carries no ``call_stage``
# (older persisted rows), so historical data derives the same values.
STAGE_ALIASES: dict[str, str] = {
    "loop_direction_planned": "loop_planner",
    "plan_alignment_judged": "plan_alignment_judge",
    "code_edit_alignment_rejected": "plan_alignment_judge",
    "code_edit_drafted": "code_edit_draft",
    "code_edit_validation_failed": "code_edit_draft",
    "no_viable_patch": "code_edit_draft",
    "code_edit_repair_requested": "code_edit_repair",
    "code_edit_repair_drafted": "code_edit_repair",
    "code_edit_repair_failed": "code_edit_repair",
    "source_inspection_requested": "source_inspection",
    "source_inspection_resolved": "source_inspection",
    "source_inspection_failed": "source_inspection",
}

_DEFAULT_PROVENANCE: dict[str, Any] = {
    "call_emitter": CALL_EMITTER_CODE,
    "purpose": "",
    "component": "",
    "teacher_model_flag": False,
    "drives_control_flow": True,
}


def provenance_for_stage(stage: str) -> dict[str, Any]:
    """Resolve the derived provenance for one stage.

    Unknown stages default to ``call_emitter="code"`` (the conservative value:
    misclassifying agentic calls as axis-B hides a positive; the reverse would
    poison a Stage-3 ``call_emitter="model"`` curation filter with classifier
    calls — the exact ORO failure the field exists to prevent). ``purpose``
    falls back to the stage name so every captured call has a non-empty
    purpose.
    """
    key = str(stage or "").strip()
    key = STAGE_ALIASES.get(key, key)
    entry = STAGE_PROVENANCE.get(key)
    if entry is None:
        resolved = dict(_DEFAULT_PROVENANCE)
        resolved["purpose"] = key or "unknown_stage"
        return resolved
    return dict(entry)


def axis_rollup(calls: Iterable[Mapping[str, Any]]) -> str:
    """Trace-level axis rollup per v5 §8.3.

    The conjunction over the calls that drive control flow: a trace is axis-A
    only when every control-flow-driving call was model-emitted (and there is
    at least one). Mixed and empty traces roll up axis-B — today's champion
    traces are axis-B by construction.
    """
    saw_driving_call = False
    for call in calls:
        if not isinstance(call, Mapping):
            continue
        stage = str(call.get("stage") or call.get("call_stage") or "")
        emitter = str(call.get("call_emitter") or "")
        drives = call.get("drives_control_flow")
        if drives is None:
            drives = provenance_for_stage(stage).get("drives_control_flow", True)
        if not drives:
            continue
        saw_driving_call = True
        if emitter != CALL_EMITTER_MODEL:
            return AXIS_B
    return AXIS_A if saw_driving_call else AXIS_B


# ---------------------------------------------------------------------------
# Episode correlation (P11: source-inspection rounds form one multi-round
# agentic episode; the capture layer stamps the ambient episode onto every
# raw trace written inside the ``with call_episode(...)`` scope).
# ---------------------------------------------------------------------------

_CALL_EPISODE: ContextVar[dict[str, Any] | None] = ContextVar(
    "research_lab_call_episode", default=None
)


@contextmanager
def call_episode(**fields: Any) -> Iterator[None]:
    """Scope an episode correlation id over nested LLM calls.

    Example: the engine wraps each source-inspection round so the persisted
    raw trace carries ``{run_id, iteration, inspection_round}`` and the rounds
    reassemble into one episode.
    """
    payload = {
        key: value for key, value in fields.items() if value is not None and value != ""
    }
    token = _CALL_EPISODE.set(payload or None)
    try:
        yield
    finally:
        _CALL_EPISODE.reset(token)


def current_call_episode() -> dict[str, Any]:
    """The ambient episode fields, or {} outside any episode scope."""
    payload = _CALL_EPISODE.get()
    return dict(payload) if payload else {}


def episode_id(fields: Mapping[str, Any]) -> str:
    """Deterministic printable episode id (run + iteration + round)."""
    parts = [
        str(fields.get("run_id") or ""),
        f"i{fields.get('iteration')}" if fields.get("iteration") is not None else "",
        f"r{fields.get('inspection_round')}" if fields.get("inspection_round") is not None else "",
    ]
    return ":".join(part for part in parts if part)

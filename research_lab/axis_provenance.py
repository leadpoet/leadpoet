"""Minimal provenance labels for current OpenRouter telemetry."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

CALL_EMITTER_MODEL = "model"
CALL_EMITTER_CODE = "code"

AXIS_A = "axis_a"
AXIS_B = "axis_b"

# stage → {call_emitter, purpose, component, teacher_model_flag,
#           drives_control_flow}. ``drives_control_flow`` marks the calls the
# v5 §8.3 rollup conjuncts over (calls that decide what the pipeline does
# next, as opposed to pure post-hoc annotation).
STAGE_PROVENANCE: dict[str, dict[str, Any]] = {
    # Current qualification scoring annotates output; code controls the call.
    "scorer_judgment": {
        "call_emitter": CALL_EMITTER_CODE,
        "purpose": "score_lead_quality",
        "component": "qualification_scorer",
        "teacher_model_flag": True,
        "drives_control_flow": False,
    },
}

# Live-event types → canonical stage. The projector falls back to the
# emitting event's type when a provider-usage item carries no ``call_stage``
# (older persisted rows), so historical data derives the same values.
STAGE_ALIASES: dict[str, str] = {}

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

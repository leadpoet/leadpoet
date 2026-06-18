"""Research Lab baseline-arm configuration.

Champion gating must continue to read only ``REFERENCE_MODEL_ID``. This
module is for daily measurement jobs that intentionally run more than one
fixed arm on the same ICP set.
"""
from __future__ import annotations

import importlib
import logging
import os
from typing import Callable, Dict, List, NamedTuple, Optional

from qualification.scoring.baseline import REFERENCE_MODEL_ID

logger = logging.getLogger(__name__)

DUAL_ARM_BASELINES_ENV = "RESEARCH_LAB_DUAL_ARM_BASELINES"
DUAL_ARM_CONTRACT_READY_ENV = "RESEARCH_LAB_DUAL_ARM_CONTRACT_READY"
ARM_B_MODEL_ID_ENV = "RESEARCH_LAB_ARM_B_MODEL_ID"
ARM_B_MODULE_ENV = "RESEARCH_LAB_ARM_B_MODULE"
ARM_B_QUALIFY_ATTR_ENV = "RESEARCH_LAB_ARM_B_QUALIFY_ATTR"

RESEARCH_LAB_ARM_B_MODEL_ID = "research_lab:qualification_model:arm_b:v1"
RESEARCH_LAB_ARM_B_MODULE = "miner_models.qualification_research_arm_b"

_TRUTHY = {"1", "true", "t", "yes", "y", "on"}
_contract_not_ready_warning_emitted = False


class BaselineArmSpec(NamedTuple):
    """A fixed artifact to run against the daily ICP set."""

    model_id: str
    module_path: str
    qualify_attr: str
    label: str
    is_reference: bool = False


def _env_enabled(name: str, env: Optional[Dict[str, str]] = None) -> bool:
    values = os.environ if env is None else env
    return values.get(name, "").strip().lower() in _TRUTHY


def dual_arm_baselines_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True when lab-only dual-arm daily measurements are enabled."""
    return _env_enabled(DUAL_ARM_BASELINES_ENV, env)


def dual_arm_contract_ready(env: Optional[Dict[str, str]] = None) -> bool:
    """Return True after the composite-PK contract migration is verified."""
    return _env_enabled(DUAL_ARM_CONTRACT_READY_ENV, env)


def _validate_arm_specs(specs: List[BaselineArmSpec]) -> None:
    seen: set[str] = set()
    for arm in specs:
        if not arm.model_id:
            raise ValueError(f"baseline arm {arm.label} has empty model_id")
        if arm.model_id in seen:
            raise ValueError(f"duplicate baseline arm model_id: {arm.model_id}")
        seen.add(arm.model_id)
        if not arm.is_reference and arm.model_id == REFERENCE_MODEL_ID:
            raise ValueError(
                f"non-reference baseline arm {arm.label} cannot use "
                f"reserved reference model_id {REFERENCE_MODEL_ID}"
            )


def daily_baseline_arm_specs(
    *,
    include_lab_arm_b: Optional[bool] = None,
    env: Optional[Dict[str, str]] = None,
) -> List[BaselineArmSpec]:
    """Return the fixed arms for today's baseline job.

    The reference arm always runs. Arm B only appears when explicitly enabled
    and after the dual-arm contract migration has been smoke-tested.
    """
    values = os.environ if env is None else env
    specs = [
        BaselineArmSpec(
            model_id=REFERENCE_MODEL_ID,
            module_path="miner_models.qualification_model",
            qualify_attr="qualify",
            label="reference",
            is_reference=True,
        )
    ]

    enabled = (
        dual_arm_baselines_enabled(values)
        if include_lab_arm_b is None
        else include_lab_arm_b
    )
    if enabled and not dual_arm_contract_ready(values):
        global _contract_not_ready_warning_emitted
        log_fn = logger.debug if _contract_not_ready_warning_emitted else logger.warning
        _contract_not_ready_warning_emitted = True
        log_fn(
            "%s is enabled but %s is not set; running reference arm only "
            "until the dual-arm contract migration is applied and smoke-tested",
            DUAL_ARM_BASELINES_ENV,
            DUAL_ARM_CONTRACT_READY_ENV,
        )
        enabled = False

    if enabled:
        specs.append(
            BaselineArmSpec(
                model_id=values.get(ARM_B_MODEL_ID_ENV, RESEARCH_LAB_ARM_B_MODEL_ID),
                module_path=values.get(ARM_B_MODULE_ENV, RESEARCH_LAB_ARM_B_MODULE),
                qualify_attr=values.get(ARM_B_QUALIFY_ATTR_ENV, "qualify"),
                label="research_lab_arm_b",
                is_reference=False,
            )
        )
    _validate_arm_specs(specs)
    return specs


def resolve_qualify_fn(arm: BaselineArmSpec) -> Callable:
    """Import and return the configured ``qualify`` callable for ``arm``."""
    _validate_arm_specs([arm])
    module = importlib.import_module(arm.module_path)
    qualify_fn = getattr(module, arm.qualify_attr)
    if not callable(qualify_fn):
        raise TypeError(
            f"{arm.module_path}.{arm.qualify_attr} is not callable "
            f"for model_id={arm.model_id}"
        )
    return qualify_fn

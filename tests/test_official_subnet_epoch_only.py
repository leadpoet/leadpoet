from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

ACTIVE_EPOCH_PATHS = (
    "Leadpoet/utils/subnet_epoch.py",
    "Leadpoet/validator/reward.py",
    "gateway/api/epoch.py",
    "gateway/api/validate.py",
    "gateway/api/weights.py",
    "gateway/qualification/utils/chain.py",
    "gateway/research_lab/chain.py",
    "gateway/tasks/epoch_lifecycle.py",
    "gateway/tasks/epoch_monitor.py",
    "gateway/tasks/force_epoch_init.py",
    "gateway/tee/qualification_epoch_guard_v2.py",
    "gateway/tee/research_lab_runtime_config_v2.py",
    "gateway/utils/epoch.py",
    "neurons/auditor_validator.py",
    "neurons/validator.py",
    "validator_tee/enclave/chain_source_v2.py",
    "validator_tee/host/runtime_v2_bootstrap.py",
)

FORBIDDEN_TEXT = (
    "LEGACY_EPOCH_MODE",
    "EPOCH_MODE_ENV",
    "get_epoch_mode",
    "RESEARCH_LAB_GATEWAY_EPOCH_HINT",
    "RESEARCH_LAB_GATEWAY_BLOCK_HINT",
)

FORBIDDEN_PATTERNS = (
    re.compile(r"\bblock\s*//\s*360\b"),
    re.compile(r"\bblock\s*%\s*360\b"),
    re.compile(r"\bcurrent_block\s*//\s*360\b"),
    re.compile(r"\bcurrent_block\s*%\s*360\b"),
    re.compile(r"\bdivmod\s*\([^,]+,\s*360\s*\)"),
)


def test_active_epoch_paths_have_no_global_clock_or_mode_fallback():
    violations = []
    for relative_path in ACTIVE_EPOCH_PATHS:
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        for value in FORBIDDEN_TEXT:
            if value in source:
                violations.append(f"{relative_path}: contains {value}")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(source):
                violations.append(
                    f"{relative_path}: matches {pattern.pattern}"
                )
    assert violations == []

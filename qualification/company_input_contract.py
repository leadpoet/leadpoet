"""Explicit Arena company input grammar; absence retains the frozen grammar."""

from typing import Any, Mapping

BINDING = "ARENA_COMPANY_INPUT_CONTRACT"
POLICY = "linkedin_company_ampersand_v1"
CONTEXT_KEY = "arena_company_input_contract"


def validate(value: str) -> str:
    if not isinstance(value, str) or value not in ("", POLICY):
        raise ValueError("unsupported Arena company input contract")
    return value


def from_scorer_policy(policy: Mapping[str, Any]) -> str:
    return validate((policy.get("env_bindings") or {}).get(BINDING, ""))


def validation_context(value: str) -> dict[str, str]:
    return {CONTEXT_KEY: validate(value)}

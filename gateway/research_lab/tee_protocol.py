"""Research Lab TEE protocol selection shared by gateway runtime boundaries."""

from __future__ import annotations

import os


PROTOCOL_ENV = "RESEARCH_LAB_TEE_PROTOCOL"
LEGACY_V1_PROTOCOL = "legacy_v1"
V2_PROTOCOL = "v2"
_ALIASES = {
    V2_PROTOCOL: V2_PROTOCOL,
    "authoritative_v2": V2_PROTOCOL,
}


class ResearchLabTeeProtocolError(RuntimeError):
    """The configured Research Lab TEE protocol is invalid."""


def normalize_tee_protocol(value: str | None) -> str:
    """Normalize the sole production protocol and reject retired V1 modes."""

    normalized = str(value or V2_PROTOCOL).strip().lower()
    try:
        return _ALIASES[normalized]
    except KeyError as exc:
        raise ResearchLabTeeProtocolError(
            f"{PROTOCOL_ENV} must be {V2_PROTOCOL}; V1 authority is retired"
        ) from exc


def research_lab_tee_protocol() -> str:
    return normalize_tee_protocol(os.getenv(PROTOCOL_ENV))


def legacy_v1_enabled() -> bool:
    research_lab_tee_protocol()
    return False


def v2_enabled() -> bool:
    return research_lab_tee_protocol() == V2_PROTOCOL

"""Read environment variables by their Subnet 71 names, with the old names as a fallback.

Until this release every variable carried a ``LEADPOET_`` prefix. The new name
drops it (``TOOL_MARKET_MODE``) or, where the bare name belongs to another
tool or is too generic, uses ``SUBNET71_`` (``SUBNET71_SENTRY_DSN``,
``SUBNET71_REPO_ROOT``). For one release every reader also accepts the old
name; the new name wins when both are set. Code that starts another process
sets both names, so a rollback to an older release still sees its variables.

Remove this fallback, and the ``LEADPOET_`` mirrors in the shell scripts, in
the release after the one that ships it.
"""

from __future__ import annotations

import os
from typing import Mapping, MutableMapping, Optional, TypeVar

__all__ = ["LEGACY_PREFIX", "env_get", "env_pop", "env_set", "legacy_name"]

LEGACY_PREFIX = "LEADPOET_"
_SUBNET_PREFIX = "SUBNET71_"

_Default = TypeVar("_Default")


def legacy_name(name: str) -> str:
    """The pre-rename spelling of ``name``."""
    if name.startswith(_SUBNET_PREFIX):
        name = name[len(_SUBNET_PREFIX):]
    return LEGACY_PREFIX + name


def env_get(
    env: Optional[Mapping[str, str]], name: str, default: _Default = None
) -> "str | _Default":
    """``env[name]``, else the legacy name's value, else ``default``.

    ``env`` defaults to ``os.environ``.
    """
    source = os.environ if env is None else env
    if name in source:
        return source[name]
    return source.get(legacy_name(name), default)


def env_set(env: MutableMapping[str, str], name: str, value: str) -> None:
    """Set ``name`` and its legacy spelling, for a child that may be an older release."""
    env[name] = value
    env[legacy_name(name)] = value


def env_pop(env: MutableMapping[str, str], name: str) -> None:
    """Remove ``name`` and its legacy spelling."""
    env.pop(name, None)
    env.pop(legacy_name(name), None)

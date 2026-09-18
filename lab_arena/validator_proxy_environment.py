"""Read only indexed proxy entries from an existing private validator env file."""

from __future__ import annotations

import os
import re
import shlex
import stat
from pathlib import Path
from typing import Mapping

PROXY_NAME = re.compile(
    r"(?:LAB_ARENA_WEBSHARE_PROXY|QUALIFICATION_WEBSHARE_PROXY)_"
    r"[1-9][0-9]*\Z"
)


def validator_proxy_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Direct entries and the optional existing env file share one inventory.

    Other provider/wallet credentials in that file are never imported. Values
    are parsed as data and only the host's private environment can name it.
    """
    result = dict(environment)
    configured = str(environment.get("LAB_ARENA_PROXY_ENV_FILE") or "").strip()
    if not configured:
        return result
    path = Path(configured)
    if not path.is_absolute():
        raise ValueError("validator proxy env path must be absolute")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_uid not in (0, os.geteuid()):
            raise ValueError("validator proxy env must be a private owned regular file")
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("validator proxy env exceeds its size limit")
    seen = set()
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[7:].strip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not PROXY_NAME.fullmatch(name):
            continue
        try:
            parts = shlex.split("VALUE=" + value, comments=True, posix=True)
        except ValueError:
            raise ValueError("validator proxy env has an invalid value") from None
        if len(parts) != 1 or not parts[0].startswith("VALUE=") or name in seen:
            raise ValueError("validator proxy env has a duplicate or invalid value")
        seen.add(name)
        selected = parts[0][6:]
        if name in result and result[name].strip() and result[name] != selected:
            raise ValueError("validator proxy env conflicts with a process setting")
        result[name] = selected
    return result

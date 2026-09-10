"""Read only the pinned IAM user credentials from the protected gateway cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
from typing import Any

import boto3


DEFAULT_CACHE = Path("/home/ec2-user/.config/leadpoet/gateway.env")
FORBIDDEN = frozenset({
    "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "AWS_SECURITY_TOKEN", "AWS_PROFILE", "AWS_DEFAULT_PROFILE",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE",
    "AWS_WEB_IDENTITY_TOKEN_FILE", "AWS_ROLE_ARN",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI", "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
})


class GatewayIamSessionError(RuntimeError):
    pass


def gateway_iam_session(*, cache: Path = DEFAULT_CACHE, region: str = "us-east-1") -> Any:
    if region != "us-east-1" or any(os.environ.get(name) for name in FORBIDDEN):
        raise GatewayIamSessionError("gateway IAM session authority is invalid")
    before = cache.lstat()
    if (not stat.S_ISREG(before.st_mode) or cache.is_symlink() or
            before.st_uid != os.geteuid() or before.st_nlink != 1 or
            before.st_mode & 0o077 or not 1 <= before.st_size <= 1024 * 1024):
        raise GatewayIamSessionError("gateway IAM cache metadata is unsafe")
    descriptor = os.open(cache, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        if ((opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino) or
                opened.st_size != before.st_size):
            raise GatewayIamSessionError("gateway IAM cache changed during open")
        raw = bytearray()
        while len(raw) <= before.st_size:
            chunk = os.read(descriptor, min(65536, before.st_size + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    if len(raw) != before.st_size:
        raise GatewayIamSessionError("gateway IAM cache read differs")
    text = raw.decode("utf-8")
    raw[:] = b"\0" * len(raw)
    values: dict[str, str] = {}
    try:
        def unique(pairs):
            result = {}
            for name, value in pairs:
                if name in result:
                    raise ValueError("duplicate field")
                result[name] = value
            return result
        parsed = json.loads(text, object_pairs_hook=unique)
    except ValueError:
        parsed = None
    if isinstance(parsed, dict):
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
            if isinstance(parsed.get(name), str):
                values[name] = parsed[name]
    elif parsed is None:
        counts: dict[str, int] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            name, separator, value = line.partition("=")
            if not separator:
                raise GatewayIamSessionError("gateway IAM cache is invalid")
            name = name.removeprefix("export ").strip()
            if name in {"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"}:
                counts[name] = counts.get(name, 0) + 1
                values[name] = value.strip().strip("'\"")
        if any(counts.get(name) != 1 for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")):
            raise GatewayIamSessionError("gateway IAM cache fields are ambiguous")
    else:
        raise GatewayIamSessionError("gateway IAM cache is invalid")
    text = ""
    access_key = values.pop("AWS_ACCESS_KEY_ID", "")
    secret_key = values.pop("AWS_SECRET_ACCESS_KEY", "")
    if (re.fullmatch(r"AKIA[A-Z0-9]{16}", access_key) is None or
            not 32 <= len(secret_key) <= 128 or any(c.isspace() for c in secret_key)):
        raise GatewayIamSessionError("gateway IAM credentials are invalid")
    session = boto3.session.Session(aws_access_key_id=access_key,
                                    aws_secret_access_key=secret_key,
                                    region_name=region)
    access_key = secret_key = ""
    return session

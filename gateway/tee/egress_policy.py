"""Measured destination policy for gateway-enclave scoring egress.

The scoring enclave must reach public HTTPS providers and miner-supplied
company/evidence URLs.  A fixed hostname allowlist cannot cover those inputs,
so the policy allows DNS hostnames on HTTP(S) ports while rejecting every IP
literal and local/private naming convention.  The parent forwarder separately
rejects DNS answers that are not globally routable.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from typing import Any, Dict, Tuple


EGRESS_POLICY_VERSION = "leadpoet.gateway_enclave_egress.v1"
ALLOWED_PORTS = (80, 443)
BLOCKED_EXACT_HOSTS = (
    "0.0.0.0",
    "instance-data",
    "instance-data.ec2.internal",
    "localhost",
    "metadata.google.internal",
)
BLOCKED_SUFFIXES = (
    ".internal",
    ".invalid",
    ".local",
    ".localhost",
    ".onion",
    ".test",
)
_HOST_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)


class EgressPolicyError(ValueError):
    """A requested destination is outside the measured egress policy."""


def policy_document() -> Dict[str, Any]:
    return {
        "schema_version": EGRESS_POLICY_VERSION,
        "allowed_ports": list(ALLOWED_PORTS),
        "blocked_exact_hosts": list(BLOCKED_EXACT_HOSTS),
        "blocked_suffixes": list(BLOCKED_SUFFIXES),
        "ip_literals_allowed": False,
        "parent_requires_global_dns_answers": True,
    }


def destination_policy_hash() -> str:
    encoded = json.dumps(
        policy_document(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def normalize_destination(host: Any, port: Any) -> Tuple[str, int]:
    """Return one canonical public-DNS destination or fail closed."""

    raw_host = str(host or "").strip().rstrip(".")
    if not raw_host or any(character.isspace() for character in raw_host):
        raise EgressPolicyError("egress destination host is invalid")
    try:
        normalized_host = raw_host.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError) as exc:
        raise EgressPolicyError("egress destination host is invalid") from exc
    try:
        ipaddress.ip_address(normalized_host.strip("[]"))
    except ValueError:
        pass
    else:
        raise EgressPolicyError("egress destination cannot be an IP literal")
    if normalized_host in BLOCKED_EXACT_HOSTS:
        raise EgressPolicyError("egress destination host is blocked")
    if any(normalized_host.endswith(suffix) for suffix in BLOCKED_SUFFIXES):
        raise EgressPolicyError("egress destination suffix is blocked")
    if not _HOST_RE.fullmatch(normalized_host):
        raise EgressPolicyError("egress destination must be a DNS hostname")
    try:
        normalized_port = int(port)
    except (TypeError, ValueError) as exc:
        raise EgressPolicyError("egress destination port is invalid") from exc
    if normalized_port not in ALLOWED_PORTS:
        raise EgressPolicyError("egress destination port is blocked")
    return normalized_host, normalized_port

"""Container egress policy for the Arena host (labarena.md sections 4, 11, 12.1).

The scorer container may reach only public port 443; private, link-local,
loopback, carrier-grade NAT, multicast, and instance-metadata ranges are
denied for both address families. The builder has no network at all. This
module renders the nftables ruleset and the container arguments and models
the same policy in Python so tests can prove every denied destination
without a host. Live enforcement is verified on the host (section 18.4).
"""

from __future__ import annotations

import ipaddress
from typing import Any, Dict, List, Sequence, Tuple

DENIED_IPV4 = (
    "0.0.0.0/8",
    "10.0.0.0/8",
    "100.64.0.0/10",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "172.16.0.0/12",
    "192.0.0.0/24",
    "192.168.0.0/16",
    "198.18.0.0/15",
    "224.0.0.0/4",
    "240.0.0.0/4",
)
DENIED_IPV6 = (
    "::/128",
    "::1/128",
    "::ffff:0:0/96",
    "64:ff9b::/96",
    "fc00::/7",
    "fe80::/10",
    "ff00::/8",
)
METADATA_ADDRESSES = ("169.254.169.254", "fd00:ec2::254")
ALLOWED_PORT = 443
DEFAULT_RESOLVERS = ("1.1.1.1", "8.8.8.8", "2606:4700:4700::1111", "2001:4860:4860::8888")
SCORER_TABLE = "lab_arena_scorer"

_DENIED_NETWORKS = tuple(ipaddress.ip_network(item) for item in DENIED_IPV4 + DENIED_IPV6)


def destination_allowed(address: str, port: int, *, resolvers: Sequence[str] = DEFAULT_RESOLVERS) -> Tuple[bool, str]:
    """Policy oracle: ``(allowed, reason)`` for one destination."""

    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False, "not_an_ip_literal"
    if str(ip) in METADATA_ADDRESSES:
        return False, "instance_metadata"
    for network in _DENIED_NETWORKS:
        if ip in network:
            return False, "denied_range:%s" % network
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
        return False, "non_public"
    if port == 53 and str(ip) in {str(ipaddress.ip_address(r)) for r in resolvers}:
        return True, "resolver"
    if port != ALLOWED_PORT:
        return False, "port_not_443"
    return True, "public_443"


def scorer_nftables_ruleset(*, resolvers: Sequence[str] = DEFAULT_RESOLVERS, interface: str = "eth0") -> str:
    """nftables ruleset applied inside the scorer container's network namespace."""

    v4_resolvers = [r for r in resolvers if ipaddress.ip_address(r).version == 4]
    v6_resolvers = [r for r in resolvers if ipaddress.ip_address(r).version == 6]
    lines = [
        "table inet %s {" % SCORER_TABLE,
        "  chain output {",
        "    type filter hook output priority 0; policy drop;",
        "    ct state established,related accept",
        "    ip daddr { %s } drop" % ", ".join(DENIED_IPV4),
        "    ip6 daddr { %s } drop" % ", ".join(DENIED_IPV6),
        "    ip daddr 169.254.169.254 drop",
        "    ip6 daddr fd00:ec2::254 drop",
    ]
    if v4_resolvers:
        lines.append("    ip daddr { %s } udp dport 53 accept" % ", ".join(v4_resolvers))
        lines.append("    ip daddr { %s } tcp dport 53 accept" % ", ".join(v4_resolvers))
    if v6_resolvers:
        lines.append("    ip6 daddr { %s } udp dport 53 accept" % ", ".join(v6_resolvers))
        lines.append("    ip6 daddr { %s } tcp dport 53 accept" % ", ".join(v6_resolvers))
    lines.extend([
        "    oifname \"%s\" tcp dport %d accept" % (interface, ALLOWED_PORT),
        "    counter drop",
        "  }",
        "  chain input {",
        "    type filter hook input priority 0; policy drop;",
        "    ct state established,related accept",
        "  }",
        "}",
    ])
    return "\n".join(lines) + "\n"


def scorer_container_arguments(*, image: str, cache_dir: str, env_file: str) -> List[str]:
    """``docker run`` arguments for the scorer: no capabilities, read-only root, its own network."""

    return [
        "run", "--detach", "--name", "lab-arena-scorer", "--restart", "unless-stopped",
        "--network", "lab-arena-scorer-net", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=512m", "--pids-limit", "512",
        "--mount", "type=bind,src=%s,dst=/var/lib/lab-arena/scoring-cache" % cache_dir,
        "--env-file", env_file, "--env", "LAB_ARENA_CONTAINER=scorer", image,
    ]


def builder_container_arguments(*, image: str, context_dir: str) -> List[str]:
    """The builder has no network and no credentials, and never executes miner code."""

    return [
        "run", "--rm", "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=1g", "--pids-limit", "256",
        "--mount", "type=bind,src=%s,dst=/build,readonly" % context_dir, "--env", "LAB_ARENA_CONTAINER=builder", image,
    ]


def host_metadata_hop_limit_command(instance_id: str) -> List[str]:
    """Set the IMDS hop limit to 1 so no container reaches the instance metadata service."""

    return ["aws", "ec2", "modify-instance-metadata-options", "--instance-id", instance_id, "--http-put-response-hop-limit", "1", "--http-tokens", "required", "--http-endpoint", "enabled"]


def container_credential_split() -> Dict[str, Dict[str, Any]]:
    """Which container holds which credential (section 11, decision 11)."""

    return {
        "service": {"holds": ["LAB_ARENA_SERVICE_JWT", "LAB_ARENA_SIGNING_KEY_ID"], "network": "https egress to Supabase, KMS, S3, chain endpoint"},
        "broker": {"holds": ["LAB_ARENA_OPENROUTER_KMS_KEY_ID", "LAB_ARENA_EXA_API_KEY", "LAB_ARENA_SCRAPINGDOG_API_KEY"], "network": "https egress to providers, KMS", "only_decrypt_identity": True},
        "scorer": {"holds": ["LAB_ARENA_SCORING_OPENROUTER_API_KEY", "LAB_ARENA_SCORING_SCRAPINGDOG_API_KEY", "LAB_ARENA_SCORING_EXA_API_KEY"], "network": "public port 443 only (scorer_nftables_ruleset)"},
        "builder": {"holds": [], "network": "none"},
    }

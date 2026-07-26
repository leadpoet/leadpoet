"""Generate the authenticated TLS:443 relay for worker egress proxies.

The V2 provider transport requires worker egress proxies to be HTTPS on
port 443 (gateway/tee/provider_broker_v2.py:_validated_tls_proxy_url). The
configured upstream pool is plaintext HTTP on non-443 ports and the vendor
offers no TLS listener at all, so the fleet cannot pass sealing directly.
This generator emits everything needed to put a TLS front door on the
existing pool without touching provider code:

  * an stunnel configuration: one TLS listener on :443 with one SNI section
    per upstream — TLS terminates at the relay, then bytes flow unchanged to
    the upstream HTTP proxy, so the original Proxy-Authorization still
    authenticates end-to-end but now travels inside TLS on the wire;
  * the transformed proxy URL list (https://user:pass@proxyNN.<domain>:443)
    to place in Secrets Manager — each passes _validated_tls_proxy_url;
  * a systemd unit for the relay.

Inputs are read from a file of upstream proxy URLs (one per line,
http://user:pass@host:port) that must NEVER be committed; outputs containing
credentials are written 0600. Operator prerequisites (documented, one-time):
a wildcard DNS record *.<domain> -> relay host, and a wildcard certificate
(e.g. certbot DNS-01) at the configured cert/key paths.

Usage:
  python3 scripts/worker_proxy_relay/generate_relay.py \
      --upstreams /path/to/upstreams.txt --domain proxy-relay.example.com \
      --cert /etc/letsencrypt/live/DOMAIN/fullchain.pem \
      --key /etc/letsencrypt/live/DOMAIN/privkey.pem --out /tmp/relay
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlsplit


class RelayGenerationError(ValueError):
    pass


def parse_upstream(line: str) -> dict:
    """Validate one upstream proxy URL and return its parts."""
    raw = line.strip()
    if any(ch in raw for ch in "\x00\r\n\t"):
        # urlsplit silently strips ASCII control characters; reject them
        # explicitly so a malformed line cannot normalize into a valid URL.
        raise RelayGenerationError("upstream URL contains control characters")
    parsed = urlsplit(raw)
    if parsed.scheme != "http":
        raise RelayGenerationError("upstream must be an http:// proxy URL")
    if not parsed.hostname or not parsed.port:
        raise RelayGenerationError("upstream must include host and port")
    if not parsed.username or not parsed.password:
        raise RelayGenerationError("upstream must include credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RelayGenerationError("upstream URL must have no path or query")
    for item in (parsed.username, parsed.password):
        if any(ch in item for ch in "\x00\r\n@/:"):
            raise RelayGenerationError("upstream credentials are invalid")
    return {
        "host": parsed.hostname,
        "port": parsed.port,
        "username": parsed.username,
        "password": parsed.password,
    }


def relay_hostname(index: int, domain: str) -> str:
    normalized = str(domain or "").strip(".").lower()
    if not normalized or any(ch in normalized for ch in " /\\"):
        raise RelayGenerationError("relay domain is invalid")
    return "proxy%02d.%s" % (index, normalized)


def stunnel_config(upstreams: list, domain: str, cert: str, key: str) -> str:
    """One :443 TLS listener; SNI routes each hostname to its upstream."""
    if not upstreams:
        raise RelayGenerationError("at least one upstream is required")
    head = [
        "foreground = no",
        "pid = /run/stunnel-worker-proxy-relay.pid",
        "setuid = nobody",
        "setgid = nobody",
        "",
        "[relay]",
        "accept = 0.0.0.0:443",
        "cert = %s" % cert,
        "key = %s" % key,
        # Default section must go somewhere; route to the first upstream.
        "connect = %s:%d" % (upstreams[0]["host"], upstreams[0]["port"]),
    ]
    sections = []
    for index, upstream in enumerate(upstreams):
        name = "sni%02d" % index
        sections += [
            "",
            "[%s]" % name,
            "sni = relay:%s" % relay_hostname(index, domain),
            "cert = %s" % cert,
            "key = %s" % key,
            "connect = %s:%d" % (upstream["host"], upstream["port"]),
        ]
    return "\n".join(head + sections) + "\n"


def secrets_proxy_urls(upstreams: list, domain: str) -> list:
    """The HTTPS:443 URLs that replace the plaintext pool in Secrets Manager."""
    return [
        "https://%s:%s@%s:443"
        % (u["username"], u["password"], relay_hostname(i, domain))
        for i, u in enumerate(upstreams)
    ]


SYSTEMD_UNIT = """[Unit]
Description=Worker egress proxy TLS relay (stunnel)
After=network-online.target

[Service]
ExecStart=/usr/bin/stunnel /etc/stunnel/worker-proxy-relay.conf
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstreams", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    lines = [
        line
        for line in Path(args.upstreams).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    upstreams = [parse_upstream(line) for line in lines]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "worker-proxy-relay.conf").write_text(
        stunnel_config(upstreams, args.domain, args.cert, args.key)
    )
    secrets_path = out / "secrets_proxy_urls.txt"
    secrets_path.write_text("\n".join(secrets_proxy_urls(upstreams, args.domain)) + "\n")
    os.chmod(secrets_path, 0o600)
    (out / "worker-proxy-relay.service").write_text(SYSTEMD_UNIT)
    print(
        "generated relay for %d upstreams -> %s (secrets file is 0600; do not commit)"
        % (len(upstreams), out)
    )


if __name__ == "__main__":
    main()

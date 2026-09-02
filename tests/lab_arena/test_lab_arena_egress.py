"""Scorer and builder network policy model (labarena.md 12.1, 18.4)."""

from __future__ import annotations

import pytest

from lab_arena import egress


@pytest.mark.parametrize("address,port", [
    ("127.0.0.1", 443), ("::1", 443), ("10.1.2.3", 443), ("172.16.5.5", 443), ("192.168.1.1", 443),
    ("169.254.169.254", 80), ("169.254.169.254", 443), ("fd00:ec2::254", 443), ("169.254.1.1", 443),
    ("100.64.0.1", 443), ("224.0.0.1", 443), ("0.0.0.0", 443), ("fe80::1", 443), ("fc00::1", 443),
    ("93.184.216.34", 80), ("93.184.216.34", 8443), ("93.184.216.34", 22), ("2606:4700::6810:84e5", 8080),
    ("example.com", 443),
])
def test_denied_destinations(address, port):
    allowed, reason = egress.destination_allowed(address, port)
    assert allowed is False, (address, port, reason)


def test_public_443_and_configured_resolvers_are_the_only_allowed_destinations():
    assert egress.destination_allowed("93.184.216.34", 443) == (True, "public_443")
    assert egress.destination_allowed("2606:4700::6810:84e5", 443) == (True, "public_443")
    assert egress.destination_allowed("1.1.1.1", 53) == (True, "resolver")
    assert egress.destination_allowed("9.9.9.9", 53)[0] is False  # only configured resolvers
    assert egress.destination_allowed("1.1.1.1", 53, resolvers=("8.8.8.8",))[0] is False


def test_ruleset_drops_every_denied_range_and_allows_only_443():
    ruleset = egress.scorer_nftables_ruleset()
    assert "policy drop" in ruleset and ruleset.count("policy drop") == 2
    for network in egress.DENIED_IPV4 + egress.DENIED_IPV6:
        assert network in ruleset
    assert "169.254.169.254 drop" in ruleset and "fd00:ec2::254 drop" in ruleset
    accepts = [line.strip() for line in ruleset.splitlines() if line.strip().endswith("accept")]
    assert all(("dport 443" in line) or ("dport 53" in line) or ("established,related" in line) for line in accepts)
    assert not any("dport 80 " in line or "dport 8080" in line for line in accepts)
    # Drops precede the 443 accept so a private 443 destination never matches the accept.
    assert ruleset.index("drop") < ruleset.index("tcp dport 443 accept")


def test_container_arguments_split_credentials_and_network():
    scorer = egress.scorer_container_arguments(image="scorer@sha256:" + "a" * 64, cache_dir="/srv/cache", env_file="/etc/lab-arena/scorer.env")
    builder = egress.builder_container_arguments(image="builder@sha256:" + "b" * 64, context_dir="/srv/build")
    assert "--cap-drop" in scorer and "ALL" in scorer and "--read-only" in scorer and "no-new-privileges" in scorer
    assert builder[builder.index("--network") + 1] == "none" and not any(arg.startswith("--env-file") for arg in builder)
    assert "readonly" in builder[builder.index("--mount") + 1]
    split = egress.container_credential_split()
    assert split["builder"]["holds"] == [] and split["broker"]["only_decrypt_identity"] is True
    assert not set(split["scorer"]["holds"]) & set(split["broker"]["holds"])
    assert "LAB_ARENA_OPENROUTER_KMS_KEY_ID" not in split["service"]["holds"]
    hop = egress.host_metadata_hop_limit_command("i-0123456789abcdef0")
    assert "--http-put-response-hop-limit" in hop and hop[hop.index("--http-put-response-hop-limit") + 1] == "1"

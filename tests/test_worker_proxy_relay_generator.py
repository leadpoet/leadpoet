"""The relay generator must emit exactly what V2 sealing accepts."""
import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "generate_relay",
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "worker_proxy_relay" / "generate_relay.py",
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

from gateway.tee.provider_broker_v2 import _validated_tls_proxy_url


def test_generated_urls_pass_the_v2_transport_validator():
    upstreams = [
        gen.parse_upstream("http://user%02d:pw%02d@203.0.113.%d:8080" % (i, i, i + 1))
        for i in range(35)
    ]
    urls = gen.secrets_proxy_urls(upstreams, "proxy-relay.example.com")
    assert len(urls) == 35
    for url in urls:
        assert _validated_tls_proxy_url(url) == url  # exact V2 acceptance


def test_stunnel_config_routes_every_upstream_by_sni():
    upstreams = [
        gen.parse_upstream("http://u:p@203.0.113.%d:1080" % (i + 1)) for i in range(3)
    ]
    conf = gen.stunnel_config(upstreams, "r.example.com", "/c.pem", "/k.pem")
    assert conf.count("accept = 0.0.0.0:443") == 1
    for i in range(3):
        assert "sni = relay:proxy%02d.r.example.com" % i in conf
        assert "connect = 203.0.113.%d:1080" % (i + 1) in conf


def test_bad_upstreams_rejected():
    for bad in (
        "https://u:p@h:443",        # already TLS: not an upstream
        "http://h:8080",             # no credentials
        "http://u:p@h",              # no port
        "http://u:p\n@h:1",          # control chars
    ):
        with pytest.raises(gen.RelayGenerationError):
            gen.parse_upstream(bad)

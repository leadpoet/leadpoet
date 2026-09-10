"""Provider preflight probe classification, caching, and failure streaks."""

from __future__ import annotations

import io
import urllib.error
from unittest import mock

import pytest

from gateway.research_lab import provider_preflight as pp


def _http_error(code: int, body: bytes = b"", reason: str = "err") -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x.test", code, reason, {}, io.BytesIO(body))


class _Resp:
    def __init__(self, status: int = 200, body: bytes = b"{}"):
        self.status = status
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def _keys(monkeypatch):
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "sd-test-key")
    monkeypatch.setenv("EXA_API_KEY", "exa-test-key")
    for name in (
        "RESEARCH_LAB_PROVIDER_PREFLIGHT_ENABLED",
        "RESEARCH_LAB_PROVIDER_PREFLIGHT_TTL_SECONDS",
        "RESEARCH_LAB_PROVIDER_PREFLIGHT_FAILURE_STREAK",
        "RESEARCH_LAB_SCRAPINGDOG_API_KEY",
        "RESEARCH_LAB_BENCHMARK_EXA_API_KEY",
        "RESEARCH_LAB_EXA_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# probe classification
# ---------------------------------------------------------------------------


def test_scrapingdog_200_healthy():
    with mock.patch.object(pp.urllib.request, "urlopen", return_value=_Resp(200)):
        verdict = pp.probe_scrapingdog()
    assert verdict.healthy is True
    assert verdict.status == "healthy"


@pytest.mark.parametrize("code", [401, 402, 403, 429])
def test_scrapingdog_credit_auth_codes_unhealthy(code):
    with mock.patch.object(pp.urllib.request, "urlopen", side_effect=_http_error(code)):
        verdict = pp.probe_scrapingdog()
    assert verdict.healthy is False
    assert verdict.status == "credit_or_auth"
    assert verdict.http_status == code


@pytest.mark.parametrize("code", [400, 404, 410])
def test_scrapingdog_request_rejection_still_healthy(code):
    # The provider answered — reachable and credited enough to reject.
    with mock.patch.object(pp.urllib.request, "urlopen", side_effect=_http_error(code)):
        verdict = pp.probe_scrapingdog()
    assert verdict.healthy is True
    assert verdict.status == "healthy"


def test_scrapingdog_5xx_transport_failure():
    with mock.patch.object(pp.urllib.request, "urlopen", side_effect=_http_error(503)):
        verdict = pp.probe_scrapingdog()
    assert verdict.status == "transport_failure"


def test_scrapingdog_connection_error_transport_failure():
    with mock.patch.object(
        pp.urllib.request, "urlopen", side_effect=urllib.error.URLError("reset")
    ):
        verdict = pp.probe_scrapingdog()
    assert verdict.status == "transport_failure"


def test_scrapingdog_quota_text_in_4xx_body_unhealthy():
    err = _http_error(400, body=b'{"message": "request quota exceeded"}')
    with mock.patch.object(pp.urllib.request, "urlopen", side_effect=err):
        verdict = pp.probe_scrapingdog()
    assert verdict.status == "credit_or_auth"


def test_scrapingdog_missing_key_skips():
    with mock.patch.dict(pp.os.environ, {"SCRAPINGDOG_API_KEY": ""}):
        verdict = pp.probe_scrapingdog()
    assert verdict.healthy is True
    assert verdict.status == "no_credential"


def test_exa_402_out_of_credits_unhealthy():
    with mock.patch.object(pp.urllib.request, "urlopen", side_effect=_http_error(402)):
        verdict = pp.probe_exa()
    assert verdict.healthy is False
    assert verdict.status == "credit_or_auth"


def test_exa_200_healthy():
    with mock.patch.object(pp.urllib.request, "urlopen", return_value=_Resp(200)):
        verdict = pp.probe_exa()
    assert verdict.healthy is True


# ---------------------------------------------------------------------------
# cache + failure streak
# ---------------------------------------------------------------------------


def test_verdicts_cached_within_ttl(monkeypatch):
    calls = {"n": 0}

    def probe(_timeout=None):
        calls["n"] += 1
        return pp.ProviderVerdict(provider="scrapingdog", healthy=True, status="healthy")

    monkeypatch.setitem(pp._PROBES, "scrapingdog", probe)
    monkeypatch.setitem(
        pp._PROBES, "exa", lambda _timeout=None: pp.ProviderVerdict(provider="exa", healthy=True, status="healthy")
    )
    preflight = pp.ProviderPreflight()
    first = preflight.check()
    second = preflight.check()
    assert first["healthy"] and second["healthy"]
    assert calls["n"] == 1  # second check served from cache
    preflight.check(force=True)
    assert calls["n"] == 2


def test_transport_failures_pause_only_after_streak(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_PROVIDER_PREFLIGHT_FAILURE_STREAK", "3")
    monkeypatch.setitem(
        pp._PROBES,
        "scrapingdog",
        lambda _timeout=None: pp.ProviderVerdict(provider="scrapingdog", healthy=False, status="transport_failure"),
    )
    monkeypatch.setitem(
        pp._PROBES, "exa", lambda _timeout=None: pp.ProviderVerdict(provider="exa", healthy=True, status="healthy")
    )
    preflight = pp.ProviderPreflight()
    first = preflight.check(force=True)
    second = preflight.check(force=True)
    third = preflight.check(force=True)
    assert first["healthy"] is False and first["pause_worthy"] is False
    assert second["pause_worthy"] is False
    assert third["pause_worthy"] is True  # streak threshold reached


def test_failed_measured_transport_rolls_back_cached_verdict_and_streak(
    monkeypatch,
):
    state = {"healthy": True}

    def probe(provider):
        return pp.ProviderVerdict(
            provider=provider,
            healthy=state["healthy"],
            status="healthy" if state["healthy"] else "transport_failure",
        )

    monkeypatch.setitem(pp._PROBES, "exa", lambda _timeout=None: probe("exa"))
    monkeypatch.setitem(
        pp._PROBES,
        "scrapingdog",
        lambda _timeout=None: probe("scrapingdog"),
    )
    preflight = pp.ProviderPreflight()
    assert preflight.check(force=True)["healthy"] is True

    state["healthy"] = False
    with pytest.raises(RuntimeError, match="transport scope failed"):
        with preflight.measurement_transaction():
            assert preflight.check(force=True)["healthy"] is False
            raise RuntimeError("transport scope failed")

    state["healthy"] = True
    assert preflight.check(force=False)["healthy"] is True
    assert preflight._failure_streaks == {"exa": 0, "scrapingdog": 0}


def test_credit_failure_pause_worthy_immediately(monkeypatch):
    monkeypatch.setitem(
        pp._PROBES,
        "exa",
        lambda _timeout=None: pp.ProviderVerdict(provider="exa", healthy=False, status="credit_or_auth"),
    )
    monkeypatch.setitem(
        pp._PROBES,
        "scrapingdog",
        lambda _timeout=None: pp.ProviderVerdict(provider="scrapingdog", healthy=True, status="healthy"),
    )
    result = pp.ProviderPreflight().check(force=True)
    assert result["healthy"] is False
    assert result["pause_worthy"] is True


def test_disabled_preflight_short_circuits(monkeypatch):
    monkeypatch.setenv("RESEARCH_LAB_PROVIDER_PREFLIGHT_ENABLED", "false")
    result = pp.ProviderPreflight().check()
    assert result == {"healthy": True, "verdicts": [], "pause_worthy": False, "disabled": True}

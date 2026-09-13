"""Request correlation must not alter authority or expose private payloads."""

import httpx
import pytest

from gateway.api import arena_proxy
from tests.test_gateway_arena_proxy import _app


HOTKEY = "5Chnr6Y72gdfTFdoZnsCvkndKpMk8jAtt9JAYKaNG3LmU4BW"
CANARY = "PRIVATE-REQUEST-RESPONSE-CANARY"


@pytest.fixture(autouse=True)
def reset_log_budget(monkeypatch):
    arena_proxy._HANDOFF_LAST_LOGGED.clear()
    monkeypatch.setattr(arena_proxy, "_HANDOFF_LOG_COUNT", 0)
    monkeypatch.setattr(arena_proxy, "_HANDOFF_LOG_WINDOW", 0.0)


@pytest.mark.parametrize(
    "path,http_status,payload,expected,attributed",
    [
        ("runs/claim", 200, {"status": "leased", "icp": CANARY}, "leased", True),
        ("runs/claim", 200, {"status": "no_pending"}, "no_pending", True),
        ("runs/claim", 403, {"status": "rejected", "code": "runner_stake_below_minimum"}, "rejected", True),
        ("runs/claim", 401, {"status": "rejected", "code": "signature_invalid"}, "rejected", False),
        ("runs/claim", 500, {"status": [], "code": {"private": CANARY}}, "http_error", False),
        ("weight-state", 200, {"state": {"signature": CANARY}}, "ok", True),
    ],
)
def test_handoff_classifies_without_logging_payloads(
    monkeypatch, capsys, path, http_status, payload, expected, attributed
):
    async def forward(*_args, **_kwargs):
        return httpx.Response(http_status, json=payload, headers={"cache-control": "no-store"})

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    response = _app().post(
        "/arena/v1/" + path,
        json={"hotkey": HOTKEY, "signature": CANARY},
        headers={"authorization": CANARY, "user-agent": "python-httpx/" + CANARY},
    )
    assert response.status_code == http_status
    assert response.json() == payload
    assert response.headers["cache-control"] == "no-store"
    line = capsys.readouterr().out
    assert "arena_handoff observed_at=" in line
    assert "status=" + expected in line
    assert ("hotkey=" + HOTKEY in line) is attributed
    assert CANARY not in line
    assert "client=python-httpx" in line


def test_discovery_and_transport_failure_have_distinct_observations(monkeypatch, capsys):
    monkeypatch.setenv("LAB_ARENA_MODE", "live")

    async def forward(*_args, **_kwargs):
        return httpx.Response(200, json={"running_rounds": [], "private": CANARY})

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    assert _app().get("/arena/v1/current").status_code == 200
    line = capsys.readouterr().out
    assert "route=current http_status=200 status=ok" in line
    assert CANARY not in line

    async def failed(*_args, **_kwargs):
        raise httpx.ReadTimeout(CANARY)

    monkeypatch.setattr(arena_proxy, "_request_sidecar", failed)
    assert _app().get("/arena/v1/current").status_code == 503
    line = capsys.readouterr().out
    assert "route=current http_status=503 status=upstream_unavailable" in line
    assert CANARY not in line


def test_other_paths_and_testnet_do_not_log_payloads(monkeypatch, capsys):
    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setenv("LAB_ARENA_TESTNET_ENABLED", "true")

    async def forward(*_args, **_kwargs):
        return httpx.Response(200, json={"private": CANARY})

    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    client = _app()
    assert client.post("/arena/v1/runs/private-run/provider", json={"key": CANARY}).status_code == 200
    assert client.get("/testnet/arena/v1/current").status_code == 200
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("broken_sink", [False, True])
def test_malformed_upstream_and_broken_logging_never_change_forwarding(monkeypatch, broken_sink):
    payload = ("[" * 1100 + "0" + "]" * 1100).encode()

    async def forward(*_args, **_kwargs):
        return httpx.Response(502, content=payload, headers={"cache-control": "no-store"})

    def fail(*_args, **_kwargs):
        raise OSError(CANARY)

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.setattr(arena_proxy, "_request_sidecar", forward)
    if broken_sink:
        monkeypatch.setattr(arena_proxy, "print", fail, raising=False)
    response = _app().post("/arena/v1/runs/claim", json={"hotkey": HOTKEY})
    assert response.status_code == 502
    assert response.content == payload
    assert response.headers["cache-control"] == "no-store"


def test_large_response_is_not_parsed_for_logging(monkeypatch):
    response = httpx.Response(200, content=b"x" * 65_537)

    def forbidden():
        raise AssertionError("diagnostic must not parse large bodies")

    monkeypatch.setattr(response, "json", forbidden)
    from starlette.requests import Request
    request = Request({"type": "http", "method": "POST", "headers": []})
    arena_proxy._log_handoff_impl(request, "v1/runs/claim", b"{}", response)


def test_log_rate_and_memory_are_bounded(monkeypatch):
    monkeypatch.setattr(arena_proxy.time, "monotonic", lambda: 1000.0)
    assert arena_proxy._handoff_log_due(("same",))
    assert not arena_proxy._handoff_log_due(("same",))
    allowed = sum(arena_proxy._handoff_log_due((i,)) for i in range(1000))
    assert allowed == 119
    assert len(arena_proxy._HANDOFF_LAST_LOGGED) == 120
    for step in range(1, 4):
        monkeypatch.setattr(arena_proxy.time, "monotonic", lambda: 1000.0 + step * 60)
        for i in range(120):
            assert arena_proxy._handoff_log_due((step, i))
    assert len(arena_proxy._HANDOFF_LAST_LOGGED) == 256

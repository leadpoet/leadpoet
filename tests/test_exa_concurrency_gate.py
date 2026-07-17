"""Global Exa concurrency gate at the provider evidence proxy.

Exa allows a fixed number of ACTIVE agent runs (5) and ~25 concurrent
search requests account-wide. All containers route through the proxy, so
the gate there enforces the account caps: excess callers queue (then get a
transient 429) instead of erroring upstream and scoring candidates as bad
for provider reasons.
"""

import json
import os
from unittest import mock

from gateway.research_lab.provider_evidence_proxy import ExaConcurrencyGate


def _gate(agent=2, search=3, wait="0.05"):
    with mock.patch.dict(os.environ, {
        "EXA_AGENT_MAX_CONCURRENCY": str(agent),
        "EXA_SEARCH_MAX_CONCURRENCY": str(search),
        "EXA_GATE_WAIT_SECONDS": wait,
        "EXA_AGENT_RUN_TTL_SECONDS": "9999",
    }):
        return ExaConcurrencyGate()


def _run_body(rid, status="running"):
    return json.dumps({"object": "agent_run", "id": rid, "status": status}).encode()


def test_non_exa_never_gated():
    g = _gate()
    assert g.acquire("sd", "POST", "/scrape") == ("", True)
    assert g.acquire("or", "POST", "/chat/completions") == ("", True)


def test_agent_slots_hold_until_terminal_poll():
    g = _gate(agent=2)
    k1, ok1 = g.acquire("exa", "POST", "/agent/runs")
    k2, ok2 = g.acquire("exa", "POST", "/agent/runs")
    assert (k1, ok1) == ("agent_run", True) and ok2
    g.finish("agent_run", path="/agent/runs", status=200, body=_run_body("agent_run_a"))
    g.finish("agent_run", path="/agent/runs", status=200, body=_run_body("agent_run_b"))
    # both slots held -> third creation times out
    _, ok3 = g.acquire("exa", "POST", "/agent/runs")
    assert not ok3
    # polls are never gated
    assert g.acquire("exa", "GET", "/agent/runs/agent_run_a")[0] == ""
    # terminal poll releases the run's slot
    g.observe_agent_poll("/agent/runs/agent_run_a", _run_body("agent_run_a", "completed"))
    _, ok4 = g.acquire("exa", "POST", "/agent/runs")
    assert ok4


def test_failed_creation_releases_immediately():
    g = _gate(agent=1)
    k, ok = g.acquire("exa", "POST", "/agent/runs")
    assert ok
    g.finish("agent_run", path="/agent/runs", status=429, body=b"{}")  # creation failed
    _, ok2 = g.acquire("exa", "POST", "/agent/runs")
    assert ok2


def test_search_slots_release_per_request():
    g = _gate(search=1)
    k, ok = g.acquire("exa", "POST", "/search")
    assert (k, ok) == ("search", True)
    _, blocked = g.acquire("exa", "POST", "/search")
    assert not blocked
    g.finish("search", path="/search", status=200, body=b"{}")
    _, ok2 = g.acquire("exa", "POST", "/search")
    assert ok2


def test_ttl_reaper_recovers_lost_slots():
    g = _gate(agent=1)
    g._run_ttl = -1.0  # everything is instantly stale
    _, ok = g.acquire("exa", "POST", "/agent/runs")
    assert ok
    g.finish("agent_run", path="/agent/runs", status=200, body=_run_body("agent_run_x"))
    # reaper runs inside acquire and frees the leaked slot
    _, ok2 = g.acquire("exa", "POST", "/agent/runs")
    assert ok2


def test_ttl_reaper_cancels_run_upstream():
    import threading
    import gateway.research_lab.provider_evidence_proxy as proxy_module

    cancelled = []
    done = threading.Event()

    def _recorder(rid):
        cancelled.append(rid)
        done.set()

    g = _gate(agent=1)
    g._run_ttl = -1.0
    _, ok = g.acquire("exa", "POST", "/agent/runs")
    assert ok
    g.finish("agent_run", path="/agent/runs", status=200, body=_run_body("agent_run_y"))
    with mock.patch.object(proxy_module, "_cancel_abandoned_exa_run", _recorder):
        # reaper runs inside acquire; the upstream cancel is fired so the
        # abandoned run stops holding a provider-side agent-run slot
        g.acquire("exa", "POST", "/agent/runs")
        assert done.wait(2.0)
    assert cancelled == ["agent_run_y"]

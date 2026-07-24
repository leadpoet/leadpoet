"""The validator's weight-path allocation fetch must survive transient gateway
failures without ever running past the on-chain submission window.

A single transient gateway failure — connection refused, a 5xx, a brief restart
blip — used to cost the validator the whole epoch's weight submission, because
the fetch was a single ``urlopen`` with no retry and the pre-submission guard
fails closed on any exception. These tests pin the bounded in-window retry:
transient failures are retried, client rejections are not, the attempt count is
capped, and the whole sequence stays inside the caller's total time budget so a
tight window can never be exceeded.
"""

import json
import time
from urllib.error import HTTPError, URLError

import pytest

from research_lab import validator_integration as vi


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *_a):
        return False


def _urlopen_sequence(monkeypatch, items):
    """Patch urlopen to yield items in order; an Exception item is raised."""
    calls = {"n": 0}

    def _fake_urlopen(request, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        item = items[min(i, len(items) - 1)]
        if isinstance(item, BaseException):
            raise item
        return _FakeResponse(item)

    monkeypatch.setattr(vi, "urlopen", _fake_urlopen)
    return calls


def _http_error(code):
    return HTTPError("http://gw/x", code, "err", {}, None)


def test_success_on_first_attempt_does_not_retry(monkeypatch):
    calls = _urlopen_sequence(monkeypatch, [{"ok": 1}])
    out = vi._fetch_allocation_json("http://gw/x", deadline_seconds=60)
    assert out == {"ok": 1}
    assert calls["n"] == 1


def test_retries_transient_failures_then_succeeds(monkeypatch):
    calls = _urlopen_sequence(
        monkeypatch,
        [URLError("connection refused"), _http_error(503), {"ok": 2}],
    )
    out = vi._fetch_allocation_json(
        "http://gw/x", deadline_seconds=60, retry_delay_seconds=0.0
    )
    assert out == {"ok": 2}
    assert calls["n"] == 3  # two transient failures, third attempt succeeds


def test_client_rejection_4xx_is_not_retried(monkeypatch):
    calls = _urlopen_sequence(monkeypatch, [_http_error(404), {"ok": 3}])
    with pytest.raises(HTTPError):
        vi._fetch_allocation_json(
            "http://gw/x", deadline_seconds=60, retry_delay_seconds=0.0
        )
    assert calls["n"] == 1  # 404 will not resolve on retry; fail fast


def test_429_is_retried(monkeypatch):
    calls = _urlopen_sequence(monkeypatch, [_http_error(429), {"ok": 4}])
    out = vi._fetch_allocation_json(
        "http://gw/x", deadline_seconds=60, retry_delay_seconds=0.0
    )
    assert out == {"ok": 4}
    assert calls["n"] == 2


def test_exhausts_max_attempts_and_raises_last_error(monkeypatch):
    calls = _urlopen_sequence(monkeypatch, [URLError("down")] * 10)
    with pytest.raises(URLError):
        vi._fetch_allocation_json(
            "http://gw/x",
            deadline_seconds=60,
            max_attempts=3,
            retry_delay_seconds=0.0,
        )
    assert calls["n"] == 3  # capped at max_attempts


def test_tight_budget_makes_no_second_attempt(monkeypatch):
    # A budget below the minimum per-attempt reserve must not spend time on a
    # retry — the very first failure raises so the submission window is safe.
    calls = _urlopen_sequence(monkeypatch, [URLError("down")] * 5)
    with pytest.raises(URLError):
        vi._fetch_allocation_json(
            "http://gw/x",
            deadline_seconds=vi.ALLOCATION_FETCH_MIN_ATTEMPT_BUDGET_SECONDS - 1.0,
            retry_delay_seconds=0.0,
        )
    assert calls["n"] == 1


def test_retry_sequence_never_exceeds_budget(monkeypatch):
    # Simulated clock: each failed attempt burns real budget. The total wall
    # time the function is willing to spend must stay within deadline_seconds.
    _urlopen_sequence(monkeypatch, [URLError("down")] * 100)
    deadline = 20.0
    start = time.monotonic()
    now = {"t": start}
    monkeypatch.setattr(vi.time, "monotonic", lambda: now["t"])

    # Each attempt advances the simulated clock by 8s (a slow-ish failure).
    real_sleep = vi.time.sleep

    def _sleep(sec):
        now["t"] += float(sec)

    monkeypatch.setattr(vi.time, "sleep", _sleep)

    # Wrap urlopen to also advance the clock per attempt.
    orig = vi.urlopen

    def _advancing_urlopen(request, timeout=None):
        now["t"] += 8.0
        return orig(request, timeout=timeout)

    monkeypatch.setattr(vi, "urlopen", _advancing_urlopen)

    with pytest.raises(URLError):
        vi._fetch_allocation_json(
            "http://gw/x",
            deadline_seconds=deadline,
            max_attempts=100,
            retry_delay_seconds=1.0,
        )
    # The simulated clock must never advance past the deadline budget.
    assert now["t"] - start <= deadline + 8.0  # last in-flight attempt may finish


def test_attested_wrapper_targets_attested_path(monkeypatch):
    seen = {}

    def _fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        return _FakeResponse({"ok": 5})

    monkeypatch.setattr(vi, "urlopen", _fake_urlopen)
    vi.fetch_research_lab_attested_allocation_bundle("http://gw/", 24124)
    assert seen["url"] == "http://gw/research-lab/allocations/attested/24124"

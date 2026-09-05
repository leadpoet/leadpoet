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
        # Mirror the real urlopen response interface: the fetch inspects
        # Content-Encoding to decode a gzip allocation response.
        self.headers = {}

    def read(self, size=-1):
        return self._payload if size < 0 else self._payload[:size]

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


def test_preparation_fetch_budget_fits_between_build_cost_and_block_margin():
    """The block-180 preparation must not be cut off by its own deadline.

    Measured 2026-09-04 on the gateway's server spans for the attested
    allocation route (trailing 14 days, successful responses, sub-10s cache
    hits excluded, 244 cold builds): worst build 299.0s, p99 238.6s. The
    preparation budget must clear that worst observed build with real headroom
    for the bounded retry, while still finishing inside the block margin it has
    to spend.
    """
    from leadpoet_canonical.constants import (
        ALLOCATION_PREPARATION_BLOCK,
        WEIGHT_SUBMISSION_BLOCK,
    )
    from research_lab.validator_integration import (
        ALLOCATION_PREPARATION_FETCH_TIMEOUT_SECONDS as PREP_BUDGET,
        WEIGHT_INPUT_FETCH_TIMEOUT_SECONDS as WINDOW_BUDGET,
    )

    worst_observed_cold_build_seconds = 299
    # Not just "greater than" -- one second of headroom is not a budget. The
    # budget has to absorb a repeat of the worst build with room to spare.
    assert PREP_BUDGET >= 1.5 * worst_observed_cold_build_seconds
    assert PREP_BUDGET > WINDOW_BUDGET

    # ~12s Finney blocks; the preparation has to finish before the window opens.
    block_margin_seconds = (
        WEIGHT_SUBMISSION_BLOCK - ALLOCATION_PREPARATION_BLOCK
    ) * 12
    assert PREP_BUDGET < block_margin_seconds
    # And it must leave the window itself room to breathe: a preparation that
    # spends its whole budget still has to hand over before block 240.
    assert block_margin_seconds - PREP_BUDGET >= WINDOW_BUDGET


def test_ambient_preparation_budget_only_ever_raises_the_floor():
    from research_lab.validator_integration import (
        ALLOCATION_PREPARATION_FETCH_BUDGET,
        resolve_allocation_fetch_budget,
    )

    assert resolve_allocation_fetch_budget(90) == 90.0

    token = ALLOCATION_PREPARATION_FETCH_BUDGET.set(300)
    try:
        assert resolve_allocation_fetch_budget(90) == 300.0
        # A caller that already asked for more than the preparation budget keeps
        # its own number; this never shortens a deadline.
        assert resolve_allocation_fetch_budget(600) == 600.0
    finally:
        ALLOCATION_PREPARATION_FETCH_BUDGET.reset(token)

    assert resolve_allocation_fetch_budget(90) == 90.0


def test_attested_fetch_default_budget_is_unchanged():
    """The submission-window budget must stay at 90s with no preparation active."""
    import inspect

    from research_lab import validator_integration

    assert (
        inspect.signature(
            validator_integration.fetch_research_lab_attested_allocation_bundle
        )
        .parameters["timeout_seconds"]
        .default
        == 90
    )
    assert validator_integration.ALLOCATION_PREPARATION_FETCH_BUDGET.get() is None


def test_preparation_raises_the_budget_only_before_the_submission_window():
    """Before block 240 the guard task sees the long budget; in-window it does not."""
    import asyncio

    from neurons import validator as validator_module
    from research_lab.validator_integration import (
        ALLOCATION_PREPARATION_FETCH_BUDGET,
        ALLOCATION_PREPARATION_FETCH_TIMEOUT_SECONDS as PREP_BUDGET,
    )

    observed = []

    def run_preparation(window_open: bool):
        validator = validator_module.Validator.__new__(validator_module.Validator)

        async def guard(epoch):
            observed.append(ALLOCATION_PREPARATION_FETCH_BUDGET.get())
            return {"verified": True, "abort_chain_submission": False}

        async def window_open_probe():
            return window_open

        validator._research_lab_pre_weight_submission_guard = guard
        validator._research_lab_allocation_submission_window_open = (
            window_open_probe
        )
        asyncio.run(
            validator._prepare_research_lab_allocation(77, wait=True)
        )

    run_preparation(window_open=False)
    run_preparation(window_open=True)

    assert observed == [PREP_BUDGET, None]
    # The ambient value must not escape the task that set it.
    assert ALLOCATION_PREPARATION_FETCH_BUDGET.get() is None


def test_submission_window_probe_fails_safe_to_the_tight_budget():
    """An unresolvable epoch state must select the 90s budget, not the long one."""
    import asyncio

    from neurons import validator as validator_module

    validator = validator_module.Validator.__new__(validator_module.Validator)

    async def broken_epoch_state():
        raise RuntimeError("subtensor unavailable")

    validator._get_epoch_state_async = broken_epoch_state

    assert (
        asyncio.run(
            validator._research_lab_allocation_submission_window_open()
        )
        is True
    )

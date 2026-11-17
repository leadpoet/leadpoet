"""Canonical 429 cooldown is bounded separately from each HTTP attempt."""
import json
import shlex
import subprocess
import urllib.error
from pathlib import Path

import pytest

from lab_arena.chain_source import ValidatorChainSourceV2Error
from lab_arena.local_weight_signer import HttpsJsonRpcTransport
from tests.test_local_weight_signer import _Response
from tests.test_arena_validator_startup import startup_harness


class Clock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def transport(clock, statuses, *, headers=None, raised=True):
    calls = []

    def open_request(request, *, timeout):
        calls.append((clock.now, timeout, bytes(request.data), request.full_url))
        status = statuses[min(len(calls) - 1, len(statuses) - 1)]
        if status != 200 and raised:
            raise urllib.error.HTTPError(request.full_url, status, "transient", headers or {}, None)
        response = _Response(b'{"jsonrpc":"2.0","id":7,"result":"0xabc"}', status=status)
        response.headers.update(headers or {})
        return response

    rpc = HttpsJsonRpcTransport("https://archive.example/public", timeout_seconds=30,
        allow_https_path=True, opener=open_request, clock=lambda: clock.now,
        wall_clock=lambda: clock.now, sleep=clock.sleep)
    return rpc, calls


@pytest.mark.parametrize("raised", [False, True])
@pytest.mark.parametrize("header,delay", [
    (None, 60), ("", 60), ("invalid", 60), ("0", 60), ("5", 60),
    ("60", 60), ("90", 90), ("Thu, 01 Jan 1970 00:02:50 GMT", 70),
])
def test_429_honors_canonical_cooldown_and_finite_retry_after(raised, header, delay):
    clock = Clock()
    rpc, calls = transport(clock, [429, 200], headers={"Retry-After": header} if header is not None else {}, raised=raised)
    assert rpc.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert clock.sleeps == [delay]
    assert len(calls) == 2 and all(0 < call[1] <= 30 for call in calls)
    assert calls[0][2:] == calls[1][2:]


def test_two_rate_limits_use_fixed_canonical_budget_and_do_not_reset_it():
    clock = Clock()
    rpc, calls = transport(clock, [429, 429, 200])
    assert rpc.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert clock.sleeps == [60, 60] and clock.now == 220
    assert len(calls) == 3 and all(call[1] == 30 for call in calls)
    # A later successful request does not inherit a cooldown or wider timeout.
    rpc.call(method="chain_getBlockHash", params=[4], request_id=7)
    assert clock.now == 220 and calls[-1][1] == 30


def test_persistent_429_still_has_three_attempts():
    clock = Clock()
    rpc, calls = transport(clock, [429])
    with pytest.raises(ValidatorChainSourceV2Error, match="exhausted transient"):
        rpc.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 3 and clock.sleeps == [60, 60]


@pytest.mark.parametrize("header", ["150", "600", "9" * 10000, "Thu, 01 Jan 1970 00:10:00 GMT"])
def test_retry_after_cannot_extend_the_fixed_total_bound(header):
    clock = Clock()
    rpc, calls = transport(clock, [429, 200], headers={"Retry-After": header})
    with pytest.raises(ValidatorChainSourceV2Error, match="deadline"):
        rpc.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 1 and clock.sleeps == []


@pytest.mark.parametrize("statuses,sleeps", [([502, 502, 200], [1, 3]), ([502, 429, 200], [1, 60]), ([429, 502, 200], [60, 3])])
def test_other_transient_statuses_keep_short_backoff(statuses, sleeps):
    clock = Clock()
    rpc, calls = transport(clock, statuses)
    assert rpc.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert clock.sleeps == sleeps and len(calls) == 3
    assert all(0 < call[1] <= 30 for call in calls)


def test_rate_limit_wait_stops_at_fixed_bound_even_if_sleep_overruns():
    clock = Clock()
    rpc, calls = transport(clock, [429, 200])
    rpc._sleep = lambda delay: clock.sleep(delay + 100)
    with pytest.raises(ValidatorChainSourceV2Error, match="deadline"):
        rpc.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 1


def test_429_allowance_does_not_renew_the_original_io_budget():
    clock = Clock()
    rpc, calls = transport(clock, [429, 429, 200])
    original = rpc._opener

    def delayed_response(request, *, timeout):
        try:
            return original(request, timeout=timeout)
        finally:
            clock.now += 10

    rpc._opener = delayed_response
    assert rpc.call(method="chain_getBlockHash", params=[3], request_id=7) == "0xabc"
    assert clock.now == 250
    assert [call[1] for call in calls] == [30, 30, 10]


def test_502_retry_after_cannot_borrow_the_429_budget():
    clock = Clock()
    rpc, calls = transport(clock, [502, 200], headers={"Retry-After": "60"})
    with pytest.raises(ValidatorChainSourceV2Error, match="deadline"):
        rpc.call(method="chain_getBlockHash", params=[3], request_id=7)
    assert len(calls) == 1 and clock.sleeps == []


def test_preflight_timeout_fails_before_service_stop(tmp_path):
    root = Path(__file__).resolve().parents[1]
    text = (root / "validator_restart.sh").read_text()
    start = text.index('( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1')
    end = text.index('\n\nif [ -e "$CURRENT_LINK"', start)
    marker = tmp_path / "runtime-started"
    for name, script in {
        "sudo": '#!/bin/bash\nexec "$@"\n',
        "timeout": '#!/bin/bash\n[ "$1" = 300 ] || exit 99\necho PREFLIGHT_TIMEOUT >&2\nexit 124\n',
        "runtime": f'#!/bin/bash\ntouch {shlex.quote(str(marker))}\n',
    }.items():
        path = tmp_path / name
        path.write_text(script)
        path.chmod(0o755)
    program = f'''set -euo pipefail
export PATH={shlex.quote(str(tmp_path))}:"$PATH"
RELEASE={shlex.quote(str(root))}
CANDIDATE_SERVICE_ENV=/tmp/held-candidate.env
PYTHON={shlex.quote(str(tmp_path / "runtime"))}
READY_TIMEOUT=300
{text[start:end]}
echo STOP_REACHED
'''
    result = subprocess.run(["bash", "-c", program], text=True, capture_output=True, timeout=5)
    assert result.returncode == 124 and "PREFLIGHT_TIMEOUT" in result.stderr
    assert "STOP_REACHED" not in result.stdout and not marker.exists()


@pytest.mark.parametrize("start_delay,success", [(0, True), (60, True), (120, True), (280, True), (295, False), (600, False)])
def test_canonical_activation_window_includes_all_readiness_rpc_calls(start_delay, success):
    """The one startup budget must not multiply by the number of RPC calls."""
    root = Path(__file__).resolve().parents[1]
    text = (root / "validator_restart.sh").read_text()
    start = text.index('sudo systemctl enable "$SERVICE"')
    end = text.index('\necho "SUCCESS:', start)
    program = f'''set -euo pipefail
READY_TIMEOUT=300
SERVICE=validator.service
SECONDS=0
ACTIVATED=0
fail() {{ echo "$*" >&2; exit 1; }}
timeout() {{ shift; "$@"; }}
sudo() {{
  if [ "$2" = start ]; then SECONDS={start_delay}; return 0; fi
  if [ "$2" = show ]; then echo 314; fi
  return 0
}}
sleep() {{ SECONDS=$((SECONDS + 2)); }}
{text[start:end]}
printf '%s' "$ACTIVATED"
'''
    result = subprocess.run(["bash", "-c", program], text=True, capture_output=True, timeout=5)
    assert (result.returncode == 0) is success, result.stderr


def test_unit_and_preflight_share_the_existing_canonical_startup_bound():
    root = Path(__file__).resolve().parents[1]
    script = (root / "validator_restart.sh").read_text()
    unit = (root / "deploy/leadpoet-arena-validator.service").read_text()
    assert 'READY_TIMEOUT="${VALIDATOR_READY_TIMEOUT_SECONDS:-300}"' in script
    assert "TimeoutStartSec=300" in unit
    preflight = script[script.index('( cd "$RELEASE" && sudo env PYTHONDONTWRITEBYTECODE=1'):]
    preflight = preflight[:preflight.index('\n\nif [ -e "$CURRENT_LINK"')]
    assert 'timeout "$READY_TIMEOUT"' in preflight


@pytest.mark.parametrize("wrong_genesis", [False, True])
def test_check_only_runs_canonical_readiness_after_archive_cooldown(monkeypatch, startup_harness, wrong_genesis):
    """Hold RPC/runtime replies; run startup, real epoch proof and readiness."""
    from types import SimpleNamespace
    from lab_arena import validator
    from lab_arena.local_weight_signer import HostValidatorChainSource, LocalArenaWeightSigner, load_public_chain_signing_profile
    from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id
    from tests import test_validator_chain_source_v2 as fixture
    from tests.test_arena_validator_startup import _Chain, _clear_startup_environment

    _clear_startup_environment(monkeypatch)
    clock, requests = Clock(), []
    profile = load_public_chain_signing_profile("finney")
    genesis = "0x" + profile["genesis_hash"]
    calls, rpc = fixture._stateful_rpc(genesis_hash="0x" + "00" * 32 if wrong_genesis else genesis)
    archive_attempts = 0

    def opener(request, *, timeout):
        nonlocal archive_attempts
        body = json.loads(request.data)
        requests.append((request.full_url, body, timeout))
        if request.full_url == "https://archive.example/public":
            archive_attempts += 1
            if archive_attempts <= 2:
                raise urllib.error.HTTPError(request.full_url, 429, "rate limited", {"Retry-After": "60"}, None)
        result = rpc(method=body["method"], params=body["params"], request_id=body["id"])
        return _Response(json.dumps({"jsonrpc": "2.0", "id": body["id"], "result": result}).encode())

    def endpoint(url, archive=False):
        return HttpsJsonRpcTransport(url, timeout_seconds=30, opener=opener,
            allow_https_path=archive, clock=lambda: clock.now,
            wall_clock=lambda: clock.now, sleep=clock.sleep)

    source = HostValidatorChainSource(live_transport=endpoint("https://live.example"),
        archive_transport=endpoint("https://archive.example/public", True),
        cutover_manifest=fixture._stateful_cutover(network_genesis_hash=genesis),
        finalization_sleep=clock.sleep)
    monkeypatch.setattr(source, "read_chain_signing_runtime", lambda **_kwargs: {
        "spec_version": 438, "transaction_version": 1, "genesis_hash": profile["genesis_hash"]})
    hotkey = ss58_encode_account_id(fixture.OWNER)
    signer = LocalArenaWeightSigner(object(), chain_source=source, extrinsic_period=8,
        validator_hotkey=hotkey, network="finney", netuid=71, chain_profile=profile)
    monkeypatch.setattr(startup_harness.signer_module, "build_local_weight_signer", lambda **_kwargs: signer)
    monkeypatch.setattr(validator, "load_local_hotkey", lambda _args: SimpleNamespace(ss58_address=hotkey))
    monkeypatch.setattr(_Chain, "refresh_metagraph", lambda _self: SimpleNamespace(hotkeys=[hotkey]))
    monkeypatch.setattr(startup_harness.chain_module, "finalized_epoch_snapshot", lambda _chain: SimpleNamespace(
        settlement_epoch_id=lambda _cutover: fixture.STATEFUL_SETTLEMENT_EPOCH_ID))
    if wrong_genesis:
        with pytest.raises(ValidatorChainSourceV2Error, match="genesis differs"):
            validator.main(["--check-only"])
    else:
        assert validator.main(["--check-only"]) == 0
        assert any(call["method"] == "state_call" for call in calls)
    assert clock.sleeps == [60, 60] and clock.now == 220
    assert {url for url, _body, _timeout in requests} == {"https://live.example/", "https://archive.example/public"}
    assert all(0 < timeout <= 30 for _url, _body, timeout in requests)
    assert startup_harness.captured["loops"] == []

"""Runtime and secret-file boundaries for proxy-enabled source execution."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import ExitStack
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lab_arena import runtime
from lab_arena.runtime_host import (
    HOST_MEMORY_RESERVE_BYTES,
    PER_SLOT_MEMORY_RESERVE_BYTES,
    RuntimeHostError,
    parallel_memory_capacity,
    require_parallel_memory,
)
from lab_arena.validator_proxy_environment import validator_proxy_environment
from tests.lab_arena.test_lab_arena_runtime import make_spec, make_config


@pytest.mark.parametrize("model_failed,cleanup_failed", [(False, False), (True, False), (False, True), (True, True)])
def test_exit_is_quarantined_before_release_only_when_cleanup_fails(model_failed, cleanup_failed):
    from lab_arena.runner import _attempt_web_egress

    events = []
    worker = Mock()
    worker.quarantine.side_effect = lambda: events.append("quarantine")
    server = Mock()
    if cleanup_failed:
        server.stop.side_effect = RuntimeError("unproven cleanup")
    try:
        with ExitStack() as resources:
            resources.callback(lambda: events.append("release"))
            resources.enter_context(_attempt_web_egress(server, worker))
            if model_failed:
                raise ValueError("independent model failure")
    except (RuntimeError, ValueError):
        assert model_failed or cleanup_failed
    server.start.assert_called_once()
    server.stop.assert_called_once()
    assert events == (["quarantine", "release"] if cleanup_failed else ["release"])


def test_quarantine_reduces_signed_capacity_without_stopping_other_workers(tmp_path):
    from lab_arena import runner as rn
    from tests.lab_arena.test_lab_arena_runner import FakeApi, make_config as runner_config

    api = FakeApi([])
    pool = SimpleNamespace(capacity=10, quarantined=1)
    instance = rn.Runner(replace(runner_config(tmp_path, api, Mock(), parallel=10), proxy_worker_pool=pool))
    try:
        instance.claim_one()
        assert api.claims[-1]["body"]["declared_parallelism"] == 9
        pool.quarantined = 10
        assert instance.claim_one() == {"status": "no_pending"}
        assert len(api.claims) == 1
    finally:
        instance.close()


def test_quarantine_does_not_refill_above_reduced_usable_capacity(tmp_path):
    from lab_arena import runner as rn
    from tests.lab_arena.test_lab_arena_runner import (
        FakeApi,
        lease,
        make_config as runner_config,
    )

    api = FakeApi([lease("first", 0), lease("second", 1), lease("third", 2)])
    pool = SimpleNamespace(capacity=2, quarantined=0)
    instance = rn.Runner(
        replace(
            runner_config(tmp_path, api, Mock(), parallel=2),
            proxy_worker_pool=pool,
        )
    )
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    first_quarantined = threading.Event()
    third_started = threading.Event()

    def run_lease(item):
        try:
            if item["run_id"] == "first":
                first_started.set()
                assert release_first.wait(timeout=3)
                pool.quarantined = 1
                first_quarantined.set()
            elif item["run_id"] == "second":
                second_started.set()
                assert release_second.wait(timeout=3)
            else:
                third_started.set()
        finally:
            instance._slots.release()

    instance._run_lease = run_lease
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        result = caller.submit(instance.run_once, max_claims=3)
        assert first_started.wait(timeout=2)
        assert second_started.wait(timeout=2)
        release_first.set()
        assert first_quarantined.wait(timeout=2)
        assert not third_started.wait(timeout=0.2)
        assert len(api.claims) == 2
        release_second.set()
        assert third_started.wait(timeout=2)
        assert result.result(timeout=2) == 3
        assert [
            item["body"]["declared_parallelism"] for item in api.claims
        ] == [2, 2, 1]
    finally:
        release_first.set()
        release_second.set()
        caller.shutdown(wait=True)
        instance.close()


def test_memory_limited_runner_drains_queue_without_using_extra_proxy_slot(tmp_path):
    from lab_arena import runner as rn
    from tests.lab_arena.test_lab_arena_runner import (
        FakeApi,
        lease,
        make_config as runner_config,
    )

    api = FakeApi([lease("first", 0), lease("second", 1), lease("third", 2)])
    pool = SimpleNamespace(capacity=3, quarantined=0)
    instance = rn.Runner(
        replace(
            runner_config(tmp_path, api, Mock(), parallel=2),
            proxy_worker_pool=pool,
        )
    )
    release = threading.Event()
    first_wave_started = threading.Event()
    third_started = threading.Event()
    state = {"active": 0, "high_water": 0, "started": 0}
    lock = threading.Lock()

    def run_lease(item):
        with lock:
            state["active"] += 1
            state["started"] += 1
            state["high_water"] = max(state["high_water"], state["active"])
            if state["started"] == 2:
                first_wave_started.set()
            if item["run_id"] == "third":
                third_started.set()
        try:
            assert release.wait(timeout=3)
        finally:
            with lock:
                state["active"] -= 1
            instance._slots.release()

    instance._run_lease = run_lease
    caller = ThreadPoolExecutor(max_workers=1)
    try:
        result = caller.submit(instance.run_once, max_claims=3)
        assert first_wave_started.wait(timeout=2)
        assert not third_started.wait(timeout=0.2)
        assert len(api.claims) == 2
        release.set()
        assert result.result(timeout=2) == 3
        assert third_started.is_set()
        assert state["high_water"] == 2
    finally:
        release.set()
        caller.shutdown(wait=True)
        instance.close()


def test_proxy_file_imports_only_proxy_data_and_never_provider_secrets(tmp_path):
    path = tmp_path / "validator.env"
    path.write_text('QUALIFICATION_WEBSHARE_PROXY_3="http://user:pass@proxy.example.com:80"\nOPENROUTER_API_KEY=secret-value\n')
    path.chmod(0o600)
    env = validator_proxy_environment({"LAB_ARENA_PROXY_ENV_FILE": str(path)})
    assert env["QUALIFICATION_WEBSHARE_PROXY_3"].startswith("http://")
    assert "OPENROUTER_API_KEY" not in env
    with pytest.raises(ValueError, match="conflicts"):
        validator_proxy_environment(dict(env, QUALIFICATION_WEBSHARE_PROXY_3="different"))
    path.chmod(0o644)
    with pytest.raises(ValueError, match="private"):
        validator_proxy_environment({"LAB_ARENA_PROXY_ENV_FILE": str(path)})


def test_proxy_file_rejects_symlinks_and_duplicate_values(tmp_path):
    path = tmp_path / "validator.env"
    path.write_text("WEBSHARE_PROXY_1=one\nWEBSHARE_PROXY_1=two\n")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="duplicate"):
        validator_proxy_environment({"LAB_ARENA_PROXY_ENV_FILE": str(path)})
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        validator_proxy_environment({"LAB_ARENA_PROXY_ENV_FILE": str(link)})


def test_web_bridge_only_opens_private_gvisor_loopback(tmp_path):
    spec = make_spec(tmp_path, web_bridge_path=Path("/trusted/bridge.py"))
    document = runtime.oci_spec(spec)
    env = dict(item.split("=", 1) for item in document["process"]["env"])
    assert env["LAB_ARENA_WEB_EGRESS_SOCKET"] == "/run/lab_arena/web.sock"
    assert not any("pass" in value or "webshare" in value for value in env.values())
    mounted = next(m for m in document["mounts"] if m["destination"] == runtime.SANDBOX_WEB_BRIDGE_PATH)
    assert "ro" in mounted["options"]
    assert document["root"]["readonly"] and document["process"]["noNewPrivileges"]
    assert "--network=none" in runtime.runsc_run_command(make_config(tmp_path), tmp_path / "root", tmp_path / "bundle", spec.sandbox_id, pid_file=tmp_path / "pid")
    assert document["linux"]["seccomp"]["syscalls"][0]["args"] == [{"index": 0, "value": 2, "op": "SCMP_CMP_GT"}]


def test_historical_runtime_has_no_web_bridge(tmp_path):
    spec = make_spec(tmp_path)
    document = runtime.oci_spec(spec)
    assert "LAB_ARENA_WEB_EGRESS_SOCKET" not in runtime.sandbox_environment(spec)
    assert all(m["destination"] != runtime.SANDBOX_WEB_BRIDGE_PATH for m in document["mounts"])
    assert document["linux"]["seccomp"]["syscalls"][0]["args"] == [{"index": 0, "value": 1, "op": "SCMP_CMP_NE"}]


def test_parallel_memory_preserves_existing_sandbox_limits_and_host_reserve(tmp_path):
    info = tmp_path / "meminfo"
    info.write_text("MemTotal: 33554432 kB\nMemAvailable: 31457280 kB\n")
    membership = tmp_path / "membership"
    membership.write_text("0::/service\n")
    group = tmp_path / "cgroup"
    (group / "service").mkdir(parents=True)
    options = dict(meminfo_path=info, cgroup_root=group, membership_path=membership)
    require_parallel_memory(11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    with pytest.raises(RuntimeHostError) as failure:
        require_parallel_memory(20, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    assert failure.value.reason == "parallel_memory_insufficient"
    (group / "memory.max").write_text(str(8 * 1024**3))
    (group / "memory.current").write_text("0")
    with pytest.raises(RuntimeHostError) as failure:
        require_parallel_memory(10, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    assert failure.value.reason == "parallel_memory_insufficient"
    info.unlink()
    with pytest.raises(RuntimeHostError) as failure:
        require_parallel_memory(10, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    assert failure.value.reason == "parallel_memory_unavailable"


def _parallel_memory_paths(tmp_path, available_bytes):
    info = tmp_path / "meminfo"
    info.write_text(
        "MemTotal: 33554432 kB\nMemAvailable: %d kB\n"
        % (available_bytes // 1024)
    )
    membership = tmp_path / "membership"
    membership.write_text("0::/service\n")
    group = tmp_path / "cgroup"
    (group / "service").mkdir(parents=True)
    return dict(
        meminfo_path=info,
        cgroup_root=group,
        membership_path=membership,
    )


def _required_parallel_memory(slots):
    return (
        HOST_MEMORY_RESERVE_BYTES
        + slots * (runtime.DEFAULT_MEMORY_LIMIT_BYTES + PER_SLOT_MEMORY_RESERVE_BYTES)
    )


def test_parallel_memory_capacity_reduces_eleven_inventory_slots_to_ten(tmp_path):
    options = _parallel_memory_paths(tmp_path, 25_558_732 * 1024)
    assert parallel_memory_capacity(
        11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options
    ) == 10


def test_parallel_memory_capacity_accepts_the_exact_slot_boundary(tmp_path):
    options = _parallel_memory_paths(tmp_path, _required_parallel_memory(10))
    assert parallel_memory_capacity(
        11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options
    ) == 10


def test_parallel_memory_capacity_fails_when_one_slot_does_not_fit(tmp_path):
    options = _parallel_memory_paths(
        tmp_path, _required_parallel_memory(1) - 1024
    )
    with pytest.raises(RuntimeHostError) as failure:
        parallel_memory_capacity(11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    assert failure.value.reason == "parallel_memory_insufficient"


def test_parallel_memory_capacity_honors_a_tighter_ancestor_cgroup(tmp_path):
    options = _parallel_memory_paths(tmp_path, 32 * 1024**3)
    options["cgroup_root"].joinpath("memory.max").write_text(
        str(_required_parallel_memory(4))
    )
    options["cgroup_root"].joinpath("memory.current").write_text("0")
    assert parallel_memory_capacity(
        11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options
    ) == 4


@pytest.mark.parametrize("failure", ["missing_meminfo", "malformed_membership"])
def test_parallel_memory_capacity_fails_closed_when_memory_is_unavailable(
    tmp_path, failure
):
    options = _parallel_memory_paths(tmp_path, 32 * 1024**3)
    if failure == "missing_meminfo":
        options["meminfo_path"].unlink()
    else:
        options["membership_path"].write_text("malformed\n")
    with pytest.raises(RuntimeHostError) as raised:
        parallel_memory_capacity(11, runtime.DEFAULT_MEMORY_LIMIT_BYTES, **options)
    assert raised.value.reason == "parallel_memory_unavailable"

"""Shared validator scoring preflight stays complete and credential-safe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_arena import scoring_startup
from lab_arena import proxy_workers
from lab_arena import runtime_host
from lab_arena import validator_proxy_environment


@pytest.fixture
def args(tmp_path):
    return SimpleNamespace(
        runsc_path="/usr/local/bin/runsc",
        work_dir=str(tmp_path / "runner"),
    )


def _verified(count=1):
    return SimpleNamespace(
        total_process_capacity=count + 1,
        webshare_worker_count=count,
    )


def test_preflight_shares_host_proxy_and_memory_capacity(args, monkeypatch):
    calls = []
    monkeypatch.setattr(
        proxy_workers,
        "preflight_proxy_workers",
        lambda inventory: calls.append(inventory) or _verified(2),
    )
    result = scoring_startup.prepare_validator_scoring(
        args,
        environment={
            "LAB_ARENA_WEBSHARE_PROXY_1": "https://one:secret@proxy-one.example:443",
            "QUALIFICATION_WEBSHARE_PROXY_2": "http://two:secret@proxy-two.example:80",
        },
        prepare_host=lambda runsc, work: calls.append((runsc, work)),
        memory_capacity=lambda slots, sandbox: calls.append((slots, sandbox)) or 2,
    )

    assert result.verified_proxies.webshare_worker_count == 2
    assert result.parallelism == 2
    assert calls[0][0].name == "runsc"
    assert len(calls[1].workers) == 2
    assert calls[2] == (3, runtime_host.DEFAULT_SANDBOX_MEMORY_BYTES)


def test_thirty_verified_exits_use_memory_limited_capacity(args, monkeypatch):
    monkeypatch.setattr(
        proxy_workers,
        "preflight_proxy_workers",
        lambda inventory: _verified(len(inventory.workers)),
    )
    environment = {
        "LAB_ARENA_WEBSHARE_PROXY_%d" % index:
            "http://user:private@proxy-%d.example:80" % index
        for index in range(1, 30)
    }
    calls = []
    result = scoring_startup.prepare_validator_scoring(
        args,
        environment=environment,
        prepare_host=lambda *_args: None,
        memory_capacity=lambda slots, sandbox: calls.append(slots) or 13,
    )
    assert calls == [30]
    assert result.verified_proxies.total_process_capacity == 30
    assert result.parallelism == 13


@pytest.mark.parametrize(
    "reason,environment",
    [
        ("proxy_inventory_invalid", {}),
        (
            "retired_parallel_config",
            {
                "LAB_ARENA_WEBSHARE_PROXY_1": "https://user:secret@proxy.example:443",
                "LAB_ARENA_MAX_PARALLEL_RUNS": "2",
            },
        ),
    ],
)
def test_fixed_configuration_diagnostics_do_not_expose_values(
    args, reason, environment
):
    with pytest.raises(scoring_startup.ScoringStartupError) as failure:
        scoring_startup.prepare_validator_scoring(
            args,
            environment=environment,
            prepare_host=lambda *_args: None,
        )

    rendered = scoring_startup.scoring_startup_diagnostic(failure.value)
    assert "reason=" + reason in rendered
    assert "secret" not in rendered
    assert "proxy.example" not in rendered


def test_proxy_file_and_preflight_errors_are_redacted(args, monkeypatch):
    private_value = "private-value"
    monkeypatch.setattr(
        validator_proxy_environment,
        "validator_proxy_environment",
        lambda _environment: (_ for _ in ()).throw(ValueError(private_value)),
    )
    with pytest.raises(scoring_startup.ScoringStartupError) as file_failure:
        scoring_startup.prepare_validator_scoring(
            args, environment={}, prepare_host=lambda *_args: None
        )
    assert file_failure.value.reason == "proxy_environment_invalid"
    assert private_value not in scoring_startup.scoring_startup_diagnostic(
        file_failure.value
    )

    monkeypatch.setattr(
        validator_proxy_environment,
        "validator_proxy_environment",
        lambda environment: dict(environment),
    )
    monkeypatch.setattr(
        proxy_workers,
        "preflight_proxy_workers",
        lambda _inventory: (_ for _ in ()).throw(
            proxy_workers.ProxyWorkerPreflightError(private_value)
        ),
    )
    with pytest.raises(scoring_startup.ScoringStartupError) as proxy_failure:
        scoring_startup.prepare_validator_scoring(
            args,
            environment={
                "LAB_ARENA_WEBSHARE_PROXY_1": "https://user:secret@proxy.example:443"
            },
            prepare_host=lambda *_args: None,
        )
    assert proxy_failure.value.reason == "proxy_preflight_failed"
    assert private_value not in scoring_startup.scoring_startup_diagnostic(
        proxy_failure.value
    )


def test_preflight_details_reach_host_and_operational_events(args, monkeypatch):
    from lab_arena.validator_logging import ValidatorOperationalLogger
    original = proxy_workers.preflight_proxy_workers

    def denied(*_args, **_kwargs):
        raise proxy_workers.ProxyTransportError("private credential", http_status=407)

    monkeypatch.setattr(proxy_workers, "preflight_proxy_workers", lambda inventory: original(
        inventory, transport_probe=denied, native_ip_probe=lambda **kw: "1.1.1.1",
    ))
    with pytest.raises(scoring_startup.ScoringStartupError) as raised:
        scoring_startup.prepare_validator_scoring(
            args, environment={"LAB_ARENA_WEBSHARE_PROXY_1": "http://user:secret@proxy.example:80"},
            prepare_host=lambda *_args: None,
        )
    error = raised.value
    assert error.reason == "proxy_preflight_failed"
    assert error.operation == "proxy_worker_1_connect_http_error"
    assert error.http_status == 407
    message = scoring_startup.scoring_startup_diagnostic(error)
    assert "operation=proxy_worker_1_connect_http_error http_status=407" in message
    log = ValidatorOperationalLogger("https://gateway.invalid", keypair=None, network="finney", netuid=71)
    log.error("scoring_setup", error)
    assert len(log._queue) == 1
    content = log._queue[0]["content"]
    assert content["operation"] == error.operation
    assert content["http_status"] == 407
    assert all(secret not in message + str(content) for secret in ("private", "credential", "secret", "proxy.example"))


def test_untrusted_preflight_detail_is_never_rendered():
    error = scoring_startup.ScoringStartupError(
        reason="proxy_preflight_failed", operation="private credential", http_status="private credential",
    )
    assert error.operation is None and error.http_status is None
    assert "private" not in scoring_startup.scoring_startup_diagnostic(error)

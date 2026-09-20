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

"""Shared, credential-safe preflight for validator scoring startup."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


RETIRED_PARALLEL_ENV = "LAB_ARENA_MAX_PARALLEL_RUNS"
_REASONS = {
    "proxy_environment_invalid": (
        "check the private proxy environment file and its ownership, mode, "
        "path, and syntax"
    ),
    "proxy_inventory_invalid": (
        "configure at least one unique indexed Webshare HTTP CONNECT or "
        "HTTPS proxy"
    ),
    "proxy_preflight_failed": (
        "check each proxy connection, public exit IP, and distinct exit "
        "requirement"
    ),
    "retired_parallel_config": (
        "remove LAB_ARENA_MAX_PARALLEL_RUNS; capacity is derived from "
        "verified proxies and memory"
    ),
}


class ScoringStartupError(RuntimeError):
    """A scoring preflight failure with a fixed, credential-free reason."""

    def __init__(self, *, reason: str, operation: str | None = None,
                 http_status: int | None = None):
        self.reason = reason if reason in _REASONS else "proxy_environment_invalid"
        self.operation = operation if operation and re.fullmatch(
            r"(?:native_exit_ip|proxy_worker_[1-9][0-9]{0,2}_(?:connect|exit_ip|distinct_exit))"
            r"_(?:failed|timeout|dns|tls|http_error|invalid_response|duplicate_exit)", operation
        ) else None
        self.http_status = http_status if type(http_status) is int and 100 <= http_status <= 599 else None
        super().__init__(_REASONS[self.reason])


def scoring_startup_diagnostic(error: ScoringStartupError) -> str:
    """Render only allowlisted diagnostics, never underlying exception text."""

    reason = error.reason if error.reason in _REASONS else "proxy_environment_invalid"
    details = ""
    if error.operation:
        details += " operation=%s" % error.operation
    if error.http_status is not None:
        details += " http_status=%d" % error.http_status
    return "reason=%s%s hint=%r" % (reason, details, _REASONS[reason])


@dataclass(frozen=True)
class ScoringStartup:
    verified_proxies: Any
    parallelism: int


def prepare_validator_scoring(
    args,
    *,
    environment: Mapping[str, str] | None = None,
    prepare_host: Callable[[Path, Path], None] | None = None,
    memory_capacity: Callable[[int, int], int] | None = None,
) -> ScoringStartup:
    """Check the exact host, proxy, and memory prerequisites used by scoring."""

    from lab_arena import contracts
    from lab_arena.proxy_workers import (
        ProxyWorkerConfigurationError,
        ProxyWorkerPreflightError,
        preflight_proxy_workers,
        proxy_workers_from_environment,
    )
    from lab_arena.runtime_host import (
        DEFAULT_SANDBOX_MEMORY_BYTES,
        parallel_memory_capacity,
        prepare_scoring_host,
    )
    from lab_arena.validator_proxy_environment import validator_proxy_environment

    selected_environment = os.environ if environment is None else environment
    selected_prepare_host = (
        prepare_scoring_host if prepare_host is None else prepare_host
    )
    selected_memory_capacity = (
        parallel_memory_capacity if memory_capacity is None else memory_capacity
    )

    selected_prepare_host(Path(args.runsc_path), Path(args.work_dir))
    try:
        proxy_environment = validator_proxy_environment(selected_environment)
    except (OSError, UnicodeError, ValueError):
        raise ScoringStartupError(reason="proxy_environment_invalid") from None
    if str(proxy_environment.get(RETIRED_PARALLEL_ENV) or "").strip():
        raise ScoringStartupError(reason="retired_parallel_config")
    try:
        inventory = proxy_workers_from_environment(proxy_environment)
    except ProxyWorkerConfigurationError:
        raise ScoringStartupError(reason="proxy_inventory_invalid") from None
    try:
        verified_proxies = preflight_proxy_workers(inventory)
    except ProxyWorkerPreflightError as exc:
        raise ScoringStartupError(
            reason="proxy_preflight_failed",
            operation=(exc.operation + "_" + exc.failure_kind) if exc.operation else None,
            http_status=exc.http_status,
        ) from None
    proxy_parallelism = min(
        verified_proxies.total_process_capacity, contracts.RUNNER_SLOT_CEILING
    )
    parallelism = selected_memory_capacity(
        proxy_parallelism, DEFAULT_SANDBOX_MEMORY_BYTES
    )
    return ScoringStartup(
        verified_proxies=verified_proxies,
        parallelism=parallelism,
    )

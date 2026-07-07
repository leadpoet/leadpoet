"""Docker sandbox runner for SOURCE_ADD trials (W5 execution glue).

Bridges the funnel's injectable ``sandbox_runner`` seam to a real container
run. Per trial:

- a **dedicated evidence-proxy instance** spawns on loopback with a registry
  containing ONLY the adapter's provider entry; the miner's KMS-decrypted
  credential lives in that proxy's memory as a credential override — never in
  env, never inside the container;
- the adapter bundle mounts read-only into a plain sandbox image and runs the
  fixed entrypoint (``adapter.py``) with the proxy URL + trial ICP ref as its
  only inputs; the container carries no credentials and any provider it tries
  to reach other than its own registry entry 404s at the proxy;
- a wall-clock kill bounds each ICP run; spend is metered from the per-trial
  proxy usage ledger and charged against the miner-declared
  ``max_trial_cost_cents`` by the funnel;
- stdout's last JSON object is the ``SourceAddTrialOutputRecord`` mapping the
  funnel validates (hashes/refs only — raw content is rejected upstream).

``docker_exec`` is injectable so tests exercise the full flow without docker.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from gateway.research_lab.provider_evidence_proxy import (
    ProviderRegistryEntry,
    serve_evidence_proxy,
    validate_provider_registry_entries,
)
from research_lab.source_add_execution import SourceAddSubmissionRecord

logger = logging.getLogger(__name__)

SOURCE_ADD_ADAPTER_ENTRYPOINT = "adapter.py"
DEFAULT_SANDBOX_IMAGE = "python:3.11-slim"
DEFAULT_TRIAL_TIMEOUT_SECONDS = 300

# docker_exec contract: (argv, timeout_seconds) -> (exit_code, stdout, stderr)
DockerExec = Callable[[list[str], float], tuple[int, str, str]]


def _subprocess_docker_exec(argv: list[str], timeout_seconds: float) -> tuple[int, str, str]:
    completed = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.returncode, completed.stdout, completed.stderr


def build_trial_registry_entry(
    record: SourceAddSubmissionRecord,
    *,
    auth_kind: str = "header",
    auth_name: str = "x-api-key",
) -> ProviderRegistryEntry:
    """The adapter's own provider as a single-entry trial registry.

    The upstream base URL comes from the manifest's first declared base
    domain; the credential is supplied at proxy spawn as an in-memory
    override, so ``credential_ref`` is a placeholder that never resolves.
    """

    manifest = record.manifest
    domain = str(manifest.declared_base_domains[0]).strip().lower()
    domain = domain.split("://", 1)[-1].split("/", 1)[0]
    has_credential = bool(record.credential_envelope) or bool(manifest.credential_ref)
    entry = ProviderRegistryEntry(
        id="trial_" + "".join(ch if ch.isalnum() else "_" for ch in manifest.adapter_id)[-40:].strip("_"),
        base_url=f"https://{domain}",
        auth_kind=auth_kind if has_credential else "none",
        auth_name=auth_name if has_credential else "",
        credential_ref=("RESEARCH_LAB_SOURCE_ADD_TRIAL_OVERRIDE",) if has_credential else (),
        per_day_quota=0,
        cost_model={
            "currency": "usd",
            # Manifest cap is per REQUEST in cents; the ledger meters microusd.
            "est_cost_microusd_per_call": max(0, int(manifest.max_request_cost_cents)) * 10_000,
        },
    )
    errors = validate_provider_registry_entries([entry])
    if errors:
        raise ValueError("trial registry entry invalid: " + "; ".join(errors))
    return entry


def _ledger_cost_cents(ledger_path: Path) -> int:
    """Total metered live spend in the per-trial ledger, rounded up to cents."""

    total_microusd = 0
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and row.get("evidence") == "recorded":
                    total_microusd += max(0, int(row.get("est_cost_microusd") or 0))
    except OSError:
        return 0
    return (total_microusd + 9_999) // 10_000


def _ledger_saw_auth_failure(ledger_path: Path) -> bool:
    try:
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict) and int(row.get("status") or 0) in (401, 403):
                    return True
    except OSError:
        return False
    return False


def _last_json_object(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


def build_source_add_sandbox_runner(
    *,
    record: SourceAddSubmissionRecord,
    bundle_dir: Path,
    work_dir: Path,
    registry_entry: ProviderRegistryEntry | None = None,
    miner_credential: str = "",
    sandbox_image: str = DEFAULT_SANDBOX_IMAGE,
    timeout_seconds: float = DEFAULT_TRIAL_TIMEOUT_SECONDS,
    docker_exec: DockerExec | None = None,
    proxy_host: str = "127.0.0.1",
) -> tuple[Callable[[SourceAddSubmissionRecord, str], Mapping[str, Any]], Callable[[], None]]:
    """Build (sandbox_runner, shutdown) for ``run_sandboxed_trial``.

    Spawns the per-trial proxy immediately; the caller MUST invoke the
    returned ``shutdown`` after the trial (a ``try/finally`` around
    ``run_sandboxed_trial``). ``miner_credential`` is the already-decrypted
    key (see key_vault.decrypt_source_add_credential); it is held only by the
    proxy instance.
    """

    if not (bundle_dir / SOURCE_ADD_ADAPTER_ENTRYPOINT).is_file():
        raise ValueError(f"adapter bundle is missing {SOURCE_ADD_ADAPTER_ENTRYPOINT}")
    entry = registry_entry or build_trial_registry_entry(record)
    work_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = work_dir / "trial_usage_ledger.jsonl"
    server, _store, _thread = serve_evidence_proxy(
        host=proxy_host,
        port=0,
        registry=[entry],
        usage_ledger_path=str(ledger_path),
        caller_context={
            "caller_kind": "source_add_trial",
            "adapter_id": record.adapter_id,
            "submission_id": record.submission_id,
        },
        key_split=True,
        credential_overrides={entry.id: miner_credential} if miner_credential else None,
    )
    proxy_port = server.server_address[1]
    execute = docker_exec or _subprocess_docker_exec

    def _runner(rec: SourceAddSubmissionRecord, icp_ref: str) -> Mapping[str, Any]:
        cost_before = _ledger_cost_cents(ledger_path)
        argv = [
            "docker", "run", "--rm",
            "--network", "host",  # loopback proxy reach; no credentials inside
            "--memory", "1g",
            "--cpus", "1",
            "--read-only",
            "--tmpfs", "/tmp",
            "-v", f"{bundle_dir.resolve()}:/adapter:ro",
            "-e", f"RESEARCH_LAB_EVIDENCE_PROXY_URL=http://{proxy_host}:{proxy_port}/{entry.id}",
            "-e", f"SOURCE_ADD_ICP_REF={icp_ref}",
            "-e", f"SOURCE_ADD_ADAPTER_ID={rec.adapter_id}",
            sandbox_image,
            "python", f"/adapter/{SOURCE_ADD_ADAPTER_ENTRYPOINT}",
        ]
        started = time.monotonic()
        try:
            exit_code, stdout, stderr = execute(argv, float(timeout_seconds))
        except subprocess.TimeoutExpired:
            return {"error": "timeout", "cost_cents": max(0, _ledger_cost_cents(ledger_path) - cost_before)}
        except Exception as exc:
            logger.warning(
                "research_lab_source_add_trial_exec_failed adapter=%s error=%s",
                rec.adapter_id,
                str(exc)[:200],
            )
            return {"error": "sandbox_exec_failed", "cost_cents": 0}
        cost_cents = max(0, _ledger_cost_cents(ledger_path) - cost_before)
        if _ledger_saw_auth_failure(ledger_path):
            return {"error": "auth_failure", "cost_cents": cost_cents}
        if exit_code != 0:
            logger.info(
                "research_lab_source_add_trial_nonzero_exit adapter=%s icp=%s exit=%s elapsed=%.1fs",
                rec.adapter_id,
                icp_ref,
                exit_code,
                time.monotonic() - started,
            )
            return {"error": f"adapter_exit_{exit_code}", "cost_cents": cost_cents}
        output_doc = _last_json_object(stdout)
        if output_doc is None:
            return {"error": "no_output_json", "cost_cents": cost_cents}
        return {"output": output_doc, "cost_cents": cost_cents}

    def _shutdown() -> None:
        try:
            server.shutdown()
            server.server_close()
        except Exception:
            pass

    return _runner, _shutdown

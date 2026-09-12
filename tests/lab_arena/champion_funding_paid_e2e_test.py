"""Opt-in paid proof of champion-funded daily baseline recovery.

This test uses a disposable PostgreSQL database and local object store, plus a
fake chain clock, LocalSigner, and in-process service API.  It deliberately uses
the real submitted ``harness.py``, the host-owned
agent entrypoint, the trusted Research Lab scorer entrypoint, the Arena runner
socket, the broker, real provider HTTP responses, current OpenRouter prices,
and KMS-encrypted submission credentials.

The local process runtime runs the two production entrypoints as ordinary
subprocesses instead of in gVisor.  The named test controls above also remain
simulated boundaries.  Child environments contain no provider or AWS secrets;
agents use the native HTTP worker socket, and the trusted scorer uses its
closed operation-frame shim, matching the production entrypoint split.

Nothing runs unless ``LAB_ARENA_PAID_CHAMPION_E2E=1``.  The test creates one
dedicated OpenRouter child key and always attempts to delete it.  Its report is
safe to retain: it contains identifiers, hashes, status codes, counts, funding
labels, and integer micro-USD amounts, but no keys or provider response bodies.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import shutil
import site
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import parse_qs, urlsplit

import pytest

from lab_arena import broker as br
from lab_arena import code_review, contracts, credentials, runtime, scoring
from lab_arena import (
    service as svc,
    shim,
    source_bundle,
    submission_runtime,
    weight_state,
)
from lab_arena.code_review_runtime import SubmissionCodeReviewer
from leadpoet_canonical import arena_weights
from leadpoet_canonical.lab_arena_rewards import (
    champion_values,
    verify_reward_basis_signature,
)
from tests.lab_arena.champion_funding_round_test import (
    _activate,
    _baseline_id,
    _promote_first_winner,
)
from tests.lab_arena.lab_arena_pg_harness import (
    POSTGREST_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.test_lab_arena_service_round import (
    Harness,
    InProcessApi,
    SCORER_IMAGE_DIGEST,
    SCORER_IMAGE_REFERENCE,
    keypair,
    _run_stage_one_to_scoring,
    wallet_verify,
)


OPT_IN_ENV = "LAB_ARENA_PAID_CHAMPION_E2E"
REPORT_ENV = "LAB_ARENA_PAID_CHAMPION_REPORT"
MODEL_ENV = "LAB_ARENA_PAID_CHAMPION_MODEL"
HOST_OPENROUTER_ENV = "LAB_ARENA_PAID_HOST_OPENROUTER_API_KEY"
HOST_DEEPLINE_ENV = "LAB_ARENA_PAID_HOST_DEEPLINE_API_KEY"
HOST_SCRAPINGDOG_ENV = "LAB_ARENA_PAID_HOST_SCRAPINGDOG_API_KEY"
MINER_DEEPLINE_ENV = "LAB_ARENA_PAID_MINER_DEEPLINE_API_KEY"
MINER_SCRAPINGDOG_ENV = "LAB_ARENA_PAID_MINER_SCRAPINGDOG_API_KEY"
MANAGEMENT_ENV = "LAB_ARENA_PAID_OPENROUTER_MANAGEMENT_KEY"
KMS_KEY_ENV = "LAB_ARENA_PAID_KMS_KEY_ID"
DEFAULT_AGENT_MODEL = "perplexity/sonar-pro"
CHILD_LIMIT_USD = 100
CHILD_CONTROL_PROPAGATION_SECONDS = 75
SOURCE_ARCHIVE_ENV = "LAB_ARENA_PAID_SOURCE_ARCHIVE"
ICP_FILE_ENV = "LAB_ARENA_PAID_ICP_FILE"
MODEL_DEPS_ENV = "LAB_ARENA_PAID_MODEL_DEPS"
PAID_PARALLEL_RUNS = 4


def _paid_daily_icps() -> list[dict[str, Any]]:
    """Use the explicitly supplied public benchmark, without published scores."""
    document = json.loads(Path(_required_environment(ICP_FILE_ENV)).read_text())
    rows = document["icps"]
    assert len(rows) == contracts.BENCHMARK_ICP_COUNT
    assert len({row["icp_id"] for row in rows}) == contracts.BENCHMARK_ICP_COUNT
    return [
        {key: value for key, value in row.items() if key not in {"baseline_score", "icp_position"}}
        for row in rows
    ]


pytestmark = pytest.mark.skipif(
    os.environ.get(OPT_IN_ENV) != "1",
    reason="paid champion E2E is opt-in",
)


def _required_environment(name: str) -> str:
    value = str(os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError("%s is required for the paid champion E2E" % name)
    return value


def _safe_hash(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _safe_enum(value: Any, allowed: set[str]) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text in allowed else "other:" + _safe_hash(text.encode())


class Evidence:
    """Persist only bounded, non-secret checkpoints for a long paid run."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.document: dict[str, Any] = {
            "schema_version": "leadpoet.lab_arena.champion_paid_e2e.v1",
            "checkpoints": [],
        }

    def add(self, event: str, **fields: Any) -> None:
        allowed = (str, int, float, bool, type(None), list, dict)
        if any(not isinstance(value, allowed) for value in fields.values()):
            raise AssertionError("unsafe evidence value")
        with self._lock:
            row = {"event": str(event), **fields}
            self.document["checkpoints"].append(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(self.path.suffix + ".tmp")
            temporary.write_text(
                json.dumps(self.document, sort_keys=True, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path)
            print(
                "paid-champion-e2e %s %s" % (event, json.dumps(fields, sort_keys=True))
            )


class DiagnosticProviderTransport:
    """Real HTTP transport with a bounded, content-free diagnostic receipt."""

    _PROVIDER_BY_HOST = {
        "openrouter.ai": "openrouter",
        "code.deepline.com": "deepline",
        "api.scrapingdog.com": "scrapingdog",
    }

    def __init__(self, price_table: Mapping[str, Any], evidence: Evidence) -> None:
        self._inner = br.HttpxProviderTransport()
        self._prices = price_table
        self._evidence = evidence
        self._lock = threading.Lock()
        self._sequence = 0

    def _next_sequence(self) -> int:
        with self._lock:
            self._sequence += 1
            return self._sequence

    @staticmethod
    def _error_code_hash(body: bytes) -> str | None:
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        error = document.get("error") if isinstance(document, Mapping) else None
        code = error.get("code") if isinstance(error, Mapping) else None
        if not isinstance(code, (str, int)) or isinstance(code, bool):
            return None
        return _safe_hash(str(code).encode("utf-8"))

    @staticmethod
    def _credit_limit_error(body: bytes) -> bool:
        try:
            document = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return False
        error = document.get("error") if isinstance(document, Mapping) else None
        message = str(error.get("message") or "").lower() if isinstance(error, Mapping) else ""
        return any(term in message for term in ("credit", "balance", "limit exceeded", "spend"))

    def send(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int | None = None,
    ) -> br.ProviderResponse:
        sequence = self._next_sequence()
        host = str(urlsplit(url).hostname or "")
        provider = self._PROVIDER_BY_HOST.get(host, "unknown")
        authorization = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
        credential = (
            parse_qs(urlsplit(url).query).get("api_key", [""])[0]
            if provider == "scrapingdog" else authorization.removeprefix("Bearer ")
        )
        assert credential, "broker omitted the real outbound provider credential"
        credential_hash = _safe_hash(credential.encode())
        del credential, authorization
        try:
            response = self._inner.send(
                method=method,
                url=url,
                headers=headers,
                body=body,
                timeout_seconds=timeout_seconds,
                max_response_bytes=max_response_bytes,
            )
        except br.ProviderTransportError as exc:
            self._evidence.add(
                "provider_http_receipt",
                sequence=sequence,
                provider=provider,
                credential_hash=credential_hash,
                request_host=host,
                http_status=None,
                classification="infrastructure_transport_failure",
                response_hash=None,
                provider_error_code_hash=None,
                provider_reported_actual_microusd=None,
                terminal_marker="transport_exception",
                transport_error_hash=_safe_hash(str(exc).encode()),
            )
            raise
        account_failure = br._miner_credential_failure(
            provider, response, champion_credential_retry=True
        )
        if account_failure:
            classification = "account_credential_failure"
        elif response.status >= 500 or response.status == 429:
            classification = "infrastructure_provider_failure"
        elif response.status >= 400:
            classification = "provider_request_rejection"
        else:
            classification = "success"
        actual_microusd = None
        if provider == "openrouter":
            try:
                request_document = json.loads(body.decode("utf-8"))
                response_document = json.loads(response.body.decode("utf-8"))
                actual_microusd = br.actual_openrouter_cost_microusd(
                    self._prices,
                    str(request_document.get("model") or ""),
                    response_document,
                )
            except (UnicodeDecodeError, ValueError, AttributeError):
                actual_microusd = None
        elif provider == "deepline":
            actual_microusd = br.deepline_cost_microusd(response.body)
        self._evidence.add(
            "provider_http_receipt",
            sequence=sequence,
            provider=provider,
            credential_hash=credential_hash,
            request_host=host,
            http_status=int(response.status),
            classification=classification,
            response_hash=_safe_hash(response.body),
            provider_error_code_hash=self._error_code_hash(response.body),
            credit_limit_error=self._credit_limit_error(response.body),
            provider_reported_actual_microusd=actual_microusd,
            terminal_marker=_safe_enum(
                response.internal_provenance or "provider_response",
                {
                    "provider_response",
                    "credential_echo",
                    "redirect_rejected",
                    "response_too_large",
                },
            ),
            transport_error_hash=None,
        )
        return response

    def close(self) -> None:
        self._inner.close()


class OpenRouterChildKey:
    """Create, update, and delete one dedicated child key without logging it."""

    def __init__(self, management_key: str) -> None:
        self._management_key = management_key
        self._transport = br.HttpxProviderTransport()
        self.key_hash = ""
        self.runtime_key = ""

    def _request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        response = self._transport.send(
            method=method,
            url="https://openrouter.ai/api/v1/keys" + path,
            headers={
                "authorization": "Bearer " + self._management_key,
                "content-type": "application/json",
                "accept": "application/json",
            },
            body=(
                json.dumps(dict(body), separators=(",", ":")).encode()
                if body is not None
                else b""
            ),
            timeout_seconds=30,
            max_response_bytes=1_048_576,
        )
        if response.status < 200 or response.status >= 300:
            raise AssertionError(
                "OpenRouter child-key request failed with HTTP %d" % response.status
            )
        if not response.body:
            return {}
        try:
            document = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise AssertionError("OpenRouter child-key response was invalid") from None
        if not isinstance(document, dict):
            raise AssertionError("OpenRouter child-key response was invalid")
        return document

    def create(self) -> str:
        document = self._request(
            "POST",
            "",
            {
                "name": "arena-paid-champion-%d" % int(time.time()),
                "limit": CHILD_LIMIT_USD,
            },
        )
        data = (
            document.get("data") if isinstance(document.get("data"), dict) else document
        )
        runtime_key = document.get("key") or data.get("key")
        key_hash = data.get("hash")
        if not isinstance(runtime_key, str) or not runtime_key.startswith("sk-or-v1-"):
            raise AssertionError("OpenRouter did not return a child runtime key")
        if not isinstance(key_hash, str) or len(key_hash) != 64:
            key_hash = hashlib.sha256(runtime_key.encode()).hexdigest()
        self.runtime_key = runtime_key
        self.key_hash = key_hash
        return runtime_key

    def configure(self, *, disabled: bool, limit: float) -> dict[str, Any]:
        if not self.key_hash:
            raise AssertionError("child key was not created")
        self._request(
            "PATCH", "/" + self.key_hash, {"disabled": disabled, "limit": limit}
        )
        # Paid controls, not Arena retry policy: live requests can still use
        # the old key limit after 30 seconds. The 75-second control sequence
        # was verified against credit failure, disable, and restoration.
        for elapsed in range(0, CHILD_CONTROL_PROPAGATION_SECONDS, 25):
            time.sleep(min(25, CHILD_CONTROL_PROPAGATION_SECONDS - elapsed))
        usage = self.usage()
        assert usage["disabled"] is disabled and usage["limit"] == limit
        return usage

    def usage(self) -> dict[str, Any]:
        """Return only safe billing fields for this exact child key."""

        if not self.key_hash:
            raise AssertionError("child key was not created")
        document = self._request("GET", "/" + self.key_hash)
        data = (
            document.get("data") if isinstance(document.get("data"), dict) else document
        )
        safe: dict[str, Any] = {"key_hash": self.key_hash}
        for name in (
            "disabled",
            "limit",
            "limit_remaining",
            "usage",
            "usage_daily",
            "usage_weekly",
            "usage_monthly",
        ):
            value = data.get(name)
            if isinstance(value, bool) or (
                isinstance(value, (int, float)) and not isinstance(value, bool)
            ):
                safe[name] = value
        if "usage" not in safe:
            raise AssertionError("OpenRouter child metadata omitted usage")
        return safe

    def delete(self) -> None:
        if self.key_hash:
            try:
                self._request("DELETE", "/" + self.key_hash)
            finally:
                self.runtime_key = ""


def _source_archive(model: str, *, empty: bool = False) -> bytes:
    if not empty:
        payload = Path(_required_environment(SOURCE_ARCHIVE_ENV)).read_bytes()
        source_bundle.validate_source_archive(payload)
        return payload
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            data = b"def run_icp(icp):\n    return []\n"
            info = tarfile.TarInfo("harness.py")
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    payload = raw.getvalue()
    source_bundle.validate_source_archive(payload)
    return payload


def _preflight_paid_source(model: str) -> dict[str, Any]:
    """Validate and compile the exact public archive before creating a paid key."""
    payload = _source_archive(model)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                compile(archive.extractfile(member).read(), member.name, "exec")
                count += 1
    assert count > 0
    rows = _paid_daily_icps()
    # Verify the isolated dependency target before starting any provider spend.
    dependency_path = Path(_required_environment(MODEL_DEPS_ENV))
    assert (dependency_path / "pydantic_ai" / "__init__.py").is_file()
    return {
        "icp_count": len(rows),
        "icp_input_hash": contracts.document_hash(rows),
        "source_hash": _safe_hash(payload),
        "compiled_python_files": count,
        "real_provider_calls": 0,
    }


class LocalEntrypointRuntime:
    """Run production entrypoints locally while retaining the worker socket."""

    def __init__(
        self, root: Path, forbidden_secrets: list[str], evidence: Evidence
    ) -> None:
        self.root = root
        self.forbidden = [value.encode() for value in forbidden_secrets if value]
        self.evidence = evidence
        self.root.mkdir(parents=True, exist_ok=True)

    def _environment(self, spec: runtime.SandboxSpec, site_dir: Path) -> dict[str, str]:
        repo = str(Path(__file__).resolve().parents[2])
        environment = {
            "PATH": runtime.PROCESS_ENV["PATH"],
            "HOME": str(site_dir),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "TZ": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONPATH": os.pathsep.join(
                (str(site_dir), repo, _required_environment(MODEL_DEPS_ENV), site.getusersitepackages())
            ),
            shim.WORKER_SOCKET_ENV: str(spec.socket_path),
            "LAB_ARENA_INPUT_PATH": str(spec.input_dir / runtime.INPUT_FILE_NAME),
            "LAB_ARENA_OUTPUT_PATH": str(spec.output_path),
            "LAB_ARENA_EVALUATION_DATE": spec.evaluation_date,
            "LAB_ARENA_RANDOM_SEED": str(spec.random_seed),
            shim.TRACE_PATH_ENV: str(site_dir / "shim-trace.jsonl"),
        }
        environment.update(runtime.PROVIDER_BASE_URLS)
        environment.update({str(k): str(v) for k, v in spec.extra_environment.items()})
        blocked_names = {
            name
            for name in environment
            if "KEY" in name or "TOKEN" in name or name.startswith("AWS_")
        }
        assert blocked_names <= {shim.TRUSTED_SCORER_ENV, "SCRAPINGDOG_API_KEY"}
        if "SCRAPINGDOG_API_KEY" in environment:
            assert (
                environment["SCRAPINGDOG_API_KEY"] == runtime.SCRAPINGDOG_RUNTIME_HANDLE
            )
        return environment

    def run_icp(self, spec: runtime.SandboxSpec, **_: Any) -> runtime.SandboxResult:
        process_dir = Path(tempfile.mkdtemp(prefix="paid-entry-", dir=str(self.root)))
        try:
            (process_dir / "sitecustomize.py").write_text(
                shim.SITECUSTOMIZE_SOURCE, encoding="utf-8"
            )
            environment = self._environment(spec, process_dir)
            if spec.entry_command == runtime.SCORER_ENTRY_COMMAND:
                command = [sys.executable, "-m", "lab_arena.scorer_entrypoint"]
                cwd = process_dir
            else:
                if spec.source_dir is None:
                    raise AssertionError("agent source was not mounted")
                # Production starts submitted sources with -I; the scorer's
                # optional sitecustomize must not replace an agent's transport.
                # Explicit paths stand in for the production read-only mounts.
                command = [
                    sys.executable, "-I", "-u", "-B", "-c",
                    "import sys; from pathlib import Path; "
                    "sys.path.insert(0, sys.argv[4]); "
                    "from lab_arena import agent_entrypoint; "
                    "agent_entrypoint.DEPENDENCY_DIR=Path(sys.argv[5]); "
                    "agent_entrypoint.run(source_dir=Path(sys.argv[1]), "
                    "input_path=Path(sys.argv[2]), output_path=Path(sys.argv[3]))",
                    str(spec.source_dir),
                    str(spec.input_dir / runtime.INPUT_FILE_NAME),
                    str(spec.output_path),
                    str(Path(__file__).resolve().parents[2]),
                    _required_environment(MODEL_DEPS_ENV),
                ]
                cwd = spec.source_dir
            started = time.monotonic()
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=spec.wall_clock_seconds,
                check=False,
            )
            elapsed = time.monotonic() - started
            combined = completed.stdout + completed.stderr
            for secret in self.forbidden:
                if secret and secret in combined:
                    raise AssertionError("secret appeared in child process diagnostics")
            if completed.returncode and completed.stderr:
                diagnostics_dir = self.evidence.path.parent / "private-process-errors"
                diagnostics_dir.mkdir(mode=0o700, exist_ok=True)
                diagnostic_path = diagnostics_dir / (hashlib.sha256(completed.stderr).hexdigest() + ".txt")
                diagnostic_path.write_bytes(completed.stderr)
                diagnostic_path.chmod(0o600)
            output_bytes = (
                spec.output_path.read_bytes() if spec.output_path.is_file() else None
            )
            if (
                output_bytes is not None
                and len(output_bytes) > runtime.MAX_OUTPUT_BYTES
            ):
                output_bytes = None
            if output_bytes is not None:
                for secret in self.forbidden:
                    if secret and secret in output_bytes:
                        raise AssertionError("secret appeared in child process output")
                # Preserve the exact test output for private failure diagnosis.
                # The public receipt remains content-free. Secret checks above
                # cover this file before any bytes are retained.
                diagnostic_kind = "scorer" if spec.entry_command == runtime.SCORER_ENTRY_COMMAND else "agent"
                diagnostic_dir = self.evidence.path.parent / ("private-" + diagnostic_kind + "-results")
                diagnostic_dir.mkdir(mode=0o700, exist_ok=True)
                diagnostic_path = diagnostic_dir / (hashlib.sha256(output_bytes).hexdigest() + ".json")
                descriptor = os.open(diagnostic_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "wb") as diagnostic_file:
                    diagnostic_file.write(output_bytes)
                if spec.entry_command == runtime.SCORER_ENTRY_COMMAND:
                    document = json.loads(output_bytes)
                    self.evidence.add(
                        "scoring_receipt",
                        scored_run_id=document.get("scored_run_id"),
                        breakdowns=[
                            {
                                "final_score": item.get("final_score"),
                                "failure_reason_hash": _safe_hash(str(item.get("failure_reason") or "").encode()),
                                "fit_decisions": [
                                    {
                                        "gate": _safe_enum(receipt.get("gate"), {"company_fit", "contact"}),
                                        "decision": _safe_enum(receipt.get("decision"), {"match", "mismatch", "unavailable", "not_evaluated"}),
                                        "dimensions": {
                                            dimension: _safe_enum(decision, {"match", "mismatch", "unavailable"})
                                            for dimension, decision in (receipt.get("company_fit_dimensions") or {}).items()
                                            if dimension in {"identity", "employee_size", "industry", "geography", "stage"}
                                        },
                                    }
                                    for receipt in item.get("verifier_gate_receipts") or []
                                ],
                                "intent_indexes": [
                                    signal.get("matched_icp_signal")
                                    for signal in item.get("intent_signals_detail") or []
                                    if type(signal.get("matched_icp_signal")) is int
                                ],
                            }
                            for item in document.get("breakdowns") or []
                        ],
                    )
            diagnostic = (
                "stdout=%s stderr=%s"
                % (_safe_hash(completed.stdout), _safe_hash(completed.stderr))
            ).encode()
            self.evidence.add(
                "sandbox_process_receipt",
                entrypoint=(
                    "scorer"
                    if spec.entry_command == runtime.SCORER_ENTRY_COMMAND
                    else "agent"
                ),
                exit_code=completed.returncode,
                timed_out=False,
                stdout_hash=_safe_hash(completed.stdout),
                stderr_hash=_safe_hash(completed.stderr),
                output_hash=(
                    _safe_hash(output_bytes) if output_bytes is not None else None
                ),
            )
            return runtime.SandboxResult(
                exit_code=completed.returncode,
                timed_out=False,
                stdout=diagnostic,
                stderr=b"",
                stdout_truncated=bool(completed.stdout or completed.stderr),
                stderr_truncated=False,
                wall_seconds=elapsed,
                cpu_seconds=0.0,
                max_rss_bytes=0,
                output_bytes=output_bytes,
                output_path=str(spec.output_path),
                output_error=None if output_bytes is not None else "missing_output",
            )
        except subprocess.TimeoutExpired as exc:
            diagnostics = bytes(exc.stdout or b"") + bytes(exc.stderr or b"")
            for secret in self.forbidden:
                if secret and secret in diagnostics:
                    raise AssertionError(
                        "secret appeared in timed-out child diagnostics"
                    )
            self.evidence.add(
                "sandbox_process_receipt",
                entrypoint=(
                    "scorer"
                    if spec.entry_command == runtime.SCORER_ENTRY_COMMAND
                    else "agent"
                ),
                exit_code=None,
                timed_out=True,
                stdout_hash=_safe_hash(bytes(exc.stdout or b"")),
                stderr_hash=_safe_hash(bytes(exc.stderr or b"")),
                output_hash=None,
            )
            return runtime.fake_result(
                exit_code=None, timed_out=True, output_bytes=None
            )
        except Exception as exc:
            self.evidence.add(
                "sandbox_runtime_error",
                entrypoint=(
                    "scorer"
                    if spec.entry_command == runtime.SCORER_ENTRY_COMMAND
                    else "agent"
                ),
                error_class=type(exc).__name__,
                error_hash=_safe_hash(str(exc).encode()),
            )
            raise
        finally:
            shutil.rmtree(process_dir, ignore_errors=True)


class PaidHarness(Harness):
    def __init__(
        self,
        connect,
        tmp_path: Path,
        *,
        organizer_keys: Mapping[str, str],
        miner_credentials: Mapping[str, str],
        management_key: str,
        kms_key_id: str,
        price_table: Mapping[str, Any],
        model: str,
        evidence: Evidence,
    ) -> None:
        self._paid_organizer_keys = dict(organizer_keys)
        self._paid_miner_credentials = dict(miner_credentials)
        self._paid_management_key = management_key
        self._paid_price_table = dict(price_table)
        self._paid_model = model
        self._paid_transport = DiagnosticProviderTransport(price_table, evidence)
        self._paid_credentials = credentials.CredentialManager(kms_key_id=kms_key_id)
        self.evidence = evidence
        super().__init__(connect, tmp_path, challengers=[], runners=["alpha"])
        self.chain.netuid = 401
        self.baseline_source = _source_archive(model, empty=True)
        self.sandbox = LocalEntrypointRuntime(
            tmp_path / "paid-processes",
            list(organizer_keys.values())
            + list(miner_credentials.values())
            + [management_key],
            evidence,
        )
        self.service = self.build_service()

    def objects_key(self) -> str:
        return "champion-paid-e2e"

    def build_service(self) -> svc.ArenaService:
        store = self.make_store()
        payer = submission_runtime.SubmissionProviderKeys(
            store=store,
            credentials=self._paid_credentials,
            organizer_keys=self._paid_organizer_keys,
        )

        def broker_factory(_service, _round_row):
            return br.Broker(
                store=store,
                key_for=lambda provider: self._paid_organizer_keys[provider],
                credential_for=payer.credential_for,
                funding_source_for=payer.funding_source_for,
                provider_funding_source_for=payer.provider_funding_source_for,
                retry_miner_credential_for=payer.retry_miner_credential_for,
                mark_provider_fallback=payer.mark_provider_fallback,
                provider_restart_required_for=payer.provider_restart_required_for,
                price_table=self._paid_price_table,
                judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
                transport=self._paid_transport,
                clock=self.clock,
            )

        return svc.ArenaService(
            svc.ServiceConfig(
                mode="live",
                store=store,
                object_store=self.objects,
                signer=self.signer,
                chain=self.chain,
                verify_signature=wallet_verify,
                daily_icp_source=lambda **kwargs: {
                    "status": "ready",
                    "set_id": int(kwargs["set_id"]),
                    "icps": _paid_daily_icps(),
                },
                banned_hotkeys_source=lambda: [],
                broker_factory=broker_factory,
                defaults=svc.RoundDefaults(
                    runner_hotkeys=tuple(self.runner_keys),
                    baseline_hotkey=self.baseline_hotkey,
                    baseline_source_url=svc.DEFAULT_BASELINE_SOURCE_URL,
                    max_challengers=max(1, len(self.challengers)),
                    rewards_enabled=True,
                    scorer_image_digest=SCORER_IMAGE_DIGEST,
                    scorer_image_reference=SCORER_IMAGE_REFERENCE,
                ),
                clock=self.clock,
                network_name="test",
                netuid=401,
                baseline_source_fetcher=lambda _url, _limit: self.baseline_source,
                credential_manager=self._paid_credentials,
                code_reviewer=SubmissionCodeReviewer(
                    store=store,
                    objects=self.objects,
                    credential_for=payer.code_review_key,
                    price_table=self._paid_price_table,
                    transport=self._paid_transport,
                ),
            )
        )

    def submit(self, flavor: str, round_id: str, *, miner_label: str = "") -> str:
        schedule = self.service.store.get_round(round_id)["configuration_doc"][
            "schedule"
        ]
        opened = datetime.strptime(
            schedule["submission_open"], "%Y-%m-%dT%H:%M:%SZ"
        ).replace(tzinfo=timezone.utc)
        if self.clock() < opened:
            self.clock.advance_to(schedule["submission_open"])
        miner = keypair("svc-miner-" + (miner_label or flavor))
        payload = _source_archive(self._paid_model)
        facts = source_bundle.validate_source_archive(payload)
        presign = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_PRESIGN,
            round_id=round_id,
            hotkey=miner.ss58_address,
            body={
                "source_size_bytes": facts["source_size_bytes"],
                "consent": {"public_rerun": True},
            },
            timestamp=int(self.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex(),
        )
        target = self.service.handle_submission_presign(presign)
        self.flavors[target["submission_id"]] = flavor
        self.objects.put(target["source_ref"], payload)
        credential_document = {
            "openrouter_api_key": self._paid_miner_credentials["openrouter"],
            "openrouter_management_key": self._paid_management_key,
            "deepline_api_key": self._paid_miner_credentials["deepline"],
        }
        if self._paid_miner_credentials.get("scrapingdog"):
            credential_document["scrapingdog_api_key"] = self._paid_miner_credentials[
                "scrapingdog"
            ]
        finalize = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_FINALIZE,
            round_id=round_id,
            hotkey=miner.ss58_address,
            body={
                "submission_id": target["submission_id"],
                "source_ref": target["source_ref"],
                "source_size_bytes": facts["source_size_bytes"],
                "credentials": credential_document,
            },
            timestamp=int(self.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex(),
        )
        accepted = self.service.handle_submission_finalize(
            target["submission_id"], finalize
        )
        assert accepted["status"] == "accepted", accepted
        self.service.review_pending_submissions()
        row = self.service.store.get_submission(target["submission_id"])
        review = row.get("code_review_doc") or {}
        self.evidence.add(
            "code_review_result",
            submission_id=row["submission_id"],
            status=row["code_review_status"],
            categories=[_safe_enum(value, set(code_review.REVIEW_CATEGORIES)) for value in review.get("categories") or []],
            error_code_hash=_safe_hash(str(review.get("error_code") or "").encode()),
        )
        assert row["code_review_status"] == "passed", {
            "submission_id": row["submission_id"],
            "status": row["code_review_status"],
        }
        self.evidence.add(
            "submission_admitted",
            submission_id=row["submission_id"],
            source_hash=_safe_hash(payload),
            code_review_status=row["code_review_status"],
        )
        return target["submission_id"]

    def run_stage_with_runners(self, count: int = 1) -> None:
        scheduled_now = self.clock.now
        self.clock.now = datetime.now(timezone.utc)
        runners = [self.runner(index, parallel=PAID_PARALLEL_RUNS) for index in range(count)]
        completed = 0
        try:
            while True:
                progressed = False
                for runner in runners:
                    taken = runner.run_once(max_claims=PAID_PARALLEL_RUNS)
                    if taken:
                        completed += taken
                        progressed = True
                        rows = self.service.store.list_runs(self.round_id)
                        self.evidence.add(
                            "assignment_progress",
                            round_id=self.round_id,
                            completed_in_stage=completed,
                            total_runs=len(rows),
                            accepted_runs=sum(
                                row["status"] == "accepted" for row in rows
                            ),
                        )
                if not progressed:
                    break
        finally:
            for runner in runners:
                runner.close()
            self.clock.now = scheduled_now
        abandoned = [
            item for runner in runners for item in runner.completed if item.get("error")
        ]
        assert not abandoned, {
            "abandoned_count": len(abandoned),
            "api_error_count": len(InProcessApi.errors),
        }


@pytest.fixture
def paid_champion_database():
    yield from database_with_lab_arena_migration(POSTGREST_MIGRATIONS)


def _assert_round_complete(harness: PaidHarness, row: Mapping[str, Any]) -> None:
    baseline_id = _baseline_id(dict(row))
    executions = harness.service.store.list_runs(
        row["round_id"], submission_id=baseline_id, kind="execute"
    )
    scores = harness.service.store.list_runs(
        row["round_id"], submission_id=baseline_id, kind="score"
    )
    assert (
        len([run for run in executions if run["status"] == "accepted"])
        == contracts.BENCHMARK_ICP_COUNT
    )
    assert (
        len([run for run in scores if run["status"] == "accepted"])
        == contracts.BENCHMARK_ICP_COUNT
    )
    assert all(
        run.get("per_icp_score") is not None
        for run in executions
        if run["status"] == "accepted"
    )
    publication = row.get("publication_doc") or {}
    aggregate = next(
        item
        for item in publication.get("final_ranking") or []
        if item["submission_id"] == baseline_id
    )
    assert aggregate["final_score"] is not None


def _assert_real_paid_cycle(
    harness: PaidHarness,
    row: Mapping[str, Any],
    submission_id: str,
    *,
    require_champion_miner_funding: bool = False,
) -> None:
    executions = [
        run
        for run in harness.service.store.list_runs(
            row["round_id"], submission_id=submission_id, kind="execute"
        )
        if run["status"] == "accepted"
    ]
    scores = [
        run
        for run in harness.service.store.list_runs(
            row["round_id"], submission_id=submission_id, kind="score"
        )
        if run["status"] == "accepted"
    ]
    assert len(executions) == contracts.BENCHMARK_ICP_COUNT
    assert len(scores) == contracts.BENCHMARK_ICP_COUNT
    assert all(
        int(
            (run.get("result_doc") or {})
            .get("resource_summary", {})
            .get("provider_call_count")
            or 0
        )
        > 0
        for run in executions
    )
    # A size-bucket rejection is an accepted zero without a judge call.
    # All ICPs must still have an accepted score, with actual paid judging
    # elsewhere in the cycle.
    assert any(
        int((run.get("result_doc") or {}).get("resource_summary", {}).get("provider_call_count") or 0) > 0
        for run in scores
    )
    assert {run["scored_run_id"] for run in scores} == {
        run["run_id"] for run in executions
    }
    if require_champion_miner_funding:
        assert row["champion_fallback_providers"] == []
        assert all(
            run["champion_funding_sources"]
            == {provider: "miner_key" for provider in contracts.PROVIDERS}
            for run in executions
        )
    connection = harness.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT DISTINCT run_id, provider
                     FROM public.lab_arena_ledger
                    WHERE round_id = %s AND submission_id = %s
                      AND funding_source = 'miner_key'
                      AND entry_kind = 'settlement'
                      AND terminal_response ->> 'status' = '200'""",
                (row["round_id"], submission_id),
            )
            paid_run_providers = set(cursor.fetchall())
    finally:
        connection.close()
    assert paid_run_providers >= {
        (run["run_id"], provider) for run in executions for provider in contracts.PROVIDERS
    }
    for provider in contracts.PROVIDERS:
        expected = _safe_hash(harness._paid_miner_credentials[provider].encode())
        assert any(
            item.get("event") == "provider_http_receipt"
            and item.get("provider") == provider and item.get("http_status") == 200
            and item.get("credential_hash") == expected
            for item in harness.evidence.document["checkpoints"]
        )


def _ledger_summary(
    harness: PaidHarness, *, round_id: str | None = None
) -> dict[str, Any]:
    """Summarize final call heads; reservations are never reported as spend."""

    connection = harness.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """WITH heads AS (
                       SELECT DISTINCT ON (call_identity)
                              provider, funding_source, entry_kind,
                              amount_microusd, call_identity
                         FROM public.lab_arena_ledger
                        WHERE call_identity IS NOT NULL
                          AND (%s::text IS NULL OR round_id = %s::text)
                        ORDER BY call_identity, entry_id DESC
                   )
                   SELECT provider, funding_source, entry_kind,
                          count(*), coalesce(sum(amount_microusd), 0)
                     FROM heads
                    WHERE entry_kind IN ('settlement', 'uncertain')
                      AND EXISTS (
                          SELECT 1 FROM public.lab_arena_ledger AS dispatched
                           WHERE dispatched.call_identity = heads.call_identity
                             AND dispatched.entry_kind = 'dispatch'
                      )
                    GROUP BY provider, funding_source, entry_kind
                    ORDER BY provider, funding_source, entry_kind""",
                (round_id, round_id),
            )
            heads = cursor.fetchall()
            cursor.execute(
                """SELECT provider, funding_source, count(*)
                     FROM public.lab_arena_ledger
                    WHERE entry_kind = 'dispatch'
                      AND (%s::text IS NULL OR round_id = %s::text)
                    GROUP BY provider, funding_source
                    ORDER BY provider, funding_source""",
                (round_id, round_id),
            )
            dispatches = cursor.fetchall()
    finally:
        connection.close()
    settled = []
    uncertain = []
    for provider, funding, kind, count, amount in heads:
        row = {
            "provider": str(provider or "unknown"),
            "funding_source": str(funding or "unknown"),
            "state": str(kind),
            "call_count": int(count),
            "amount_microusd": int(amount),
        }
        (uncertain if kind == "uncertain" else settled).append(row)
    provider_dispatch_count = sum(int(item[2]) for item in dispatches)
    accounted_dispatch_count = sum(int(item[3]) for item in heads)
    return {
        "settled_charges": settled,
        "uncertain_liabilities": uncertain,
        "provider_dispatches": [
            {
                "provider": str(provider or "unknown"),
                "funding_source": str(funding or "unknown"),
                "count": int(count),
            }
            for provider, funding, count in dispatches
        ],
        "provider_dispatch_count": provider_dispatch_count,
        "accounted_dispatch_count": accounted_dispatch_count,
        "unaccounted_dispatch_count": (
            provider_dispatch_count - accounted_dispatch_count
        ),
    }


def _capture_database_snapshot(harness: PaidHarness, evidence: Evidence) -> None:
    """Persist bounded run and ledger heads before the disposable DB exits."""

    connection = harness.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT round_id, status, champion_fallback_providers,
                          champion_submission_id
                     FROM public.lab_arena_rounds
                    ORDER BY round_id"""
            )
            rounds = [
                {
                    "round_id": str(round_id),
                    "status": str(status),
                    "champion_fallback_providers": list(fallbacks or []),
                    "champion_submission_id": champion_submission_id,
                }
                for round_id, status, fallbacks, champion_submission_id in cursor.fetchall()
            ]
            cursor.execute(
                """SELECT round_id, run_id, submission_id, kind, stage,
                          icp_position, attempt, status, terminal_cause,
                          per_icp_score, scored_run_id,
                          champion_restart_required, champion_funding_sources, output_ref,
                          result_doc #>> '{failure_diagnostic,stage}',
                          result_doc #>> '{failure_diagnostic,error_class}'
                     FROM public.lab_arena_runs
                    ORDER BY round_id, stage, kind, icp_position, attempt"""
            )
            runs = []
            for (
                round_id,
                run_id,
                submission_id,
                kind,
                stage,
                position,
                attempt,
                status,
                terminal_cause,
                score,
                scored_run_id,
                restart_required,
                funding_sources,
                output_ref,
                failure_stage,
                failure_class,
            ) in cursor.fetchall():
                output_hash = None
                if output_ref:
                    try:
                        output_hash = _safe_hash(harness.objects.get(output_ref))
                    except Exception:
                        output_hash = "unavailable"
                runs.append(
                    {
                        "round_id": str(round_id),
                        "run_id": str(run_id),
                        "submission_id": str(submission_id),
                        "kind": str(kind),
                        "stage": int(stage),
                        "icp_position": int(position),
                        "attempt": int(attempt),
                        "status": str(status),
                        "terminal_cause": (
                            str(terminal_cause) if terminal_cause else None
                        ),
                        "per_icp_score": float(score) if score is not None else None,
                        "scored_run_id": str(scored_run_id) if scored_run_id else None,
                        "champion_restart_required": bool(restart_required),
                        "champion_funding_sources": funding_sources,
                        "output_hash": output_hash,
                        "failure_stage": _safe_enum(
                            failure_stage,
                            {
                                "provider_call",
                                "sandbox",
                                "sandbox_output",
                                "completion",
                            },
                        ),
                        "failure_class": _safe_enum(
                            failure_class,
                            {
                                "provider_unavailable",
                                "judge_error",
                                "judge_timeout",
                                "sandbox_output_error",
                                "missing_output",
                            },
                        ),
                    }
                )
            cursor.execute(
                """SELECT DISTINCT ON (call_identity)
                          call_identity, round_id, run_id, provider,
                          operation_id, funding_source, entry_kind,
                          amount_microusd, terminal_response ->> 'status',
                          coalesce(
                            terminal_response #>> '{account_failure_evidence,error_class}',
                            entry_doc #>> '{call,account_failure_evidence,error_class}'
                          ),
                          coalesce(
                            terminal_response #>> '{account_failure_evidence,provider_status}',
                            entry_doc #>> '{call,account_failure_evidence,provider_status}'
                          ),
                          coalesce(entry_doc #>> '{call,reason}',
                                   entry_doc ->> 'reason'),
                          entry_doc #>> '{call,failure_stage}',
                          entry_doc #>> '{call,error_class}',
                          entry_doc #>> '{call,provider_status}',
                          entry_doc #>> '{call,usage_present}',
                          entry_doc #>> '{call,billing_present}'
                     FROM public.lab_arena_ledger
                    WHERE call_identity IS NOT NULL
                    ORDER BY call_identity, entry_id DESC"""
            )
            ledger_heads = []
            for item in cursor.fetchall():
                reason = item[11]
                ledger_heads.append(
                    {
                        "call_identity": str(item[0]),
                        "round_id": str(item[1]) if item[1] else None,
                        "run_id": str(item[2]) if item[2] else None,
                        "provider": str(item[3] or "unknown"),
                        "operation_id": str(item[4] or "unknown"),
                        "funding_source": str(item[5] or "unknown"),
                        "entry_kind": str(item[6]),
                        "amount_microusd": int(item[7]),
                        "terminal_status": int(item[8]) if item[8] else None,
                        "account_failure_class": item[9],
                        "account_failure_status": (int(item[10]) if item[10] else None),
                        "reason_hash": (
                            _safe_hash(str(reason).encode()) if reason else None
                        ),
                        "reason": _safe_enum(
                            reason,
                            {
                                "missing_provider_cost",
                                "settle_failure",
                                "worker_reported",
                                "provider_cost_uncertain",
                            },
                        ),
                        "failure_stage": _safe_enum(
                            item[12],
                            {
                                "credential_resolution",
                                "transport",
                                "response_adaptation",
                                "response_sanitization",
                                "cost_accounting",
                                "terminal_response",
                                "settlement",
                            },
                        ),
                        "error_class": _safe_enum(
                            item[13],
                            {
                                "ProviderTransportError",
                                "OperationResponseError",
                                "BrokerError",
                                "ArenaStoreError",
                                "ValueError",
                                "TypeError",
                            },
                        ),
                        "provider_status": int(item[14]) if item[14] else None,
                        "usage_present": item[15] == "true",
                        "billing_present": item[16] == "true",
                    }
                )
    finally:
        connection.close()
    evidence.add(
        "final_database_snapshot",
        rounds=rounds,
        runs=runs,
        ledger_heads=ledger_heads,
        ledger=_ledger_summary(harness),
    )


def _record_round_summary(
    harness: PaidHarness,
    evidence: Evidence,
    row: Mapping[str, Any],
    submission_id: str,
) -> None:
    executions = [
        run
        for run in harness.service.store.list_runs(
            row["round_id"], submission_id=submission_id, kind="execute"
        )
        if run["status"] == "accepted"
    ]
    scores = {
        run["scored_run_id"]: run
        for run in harness.service.store.list_runs(
            row["round_id"], submission_id=submission_id, kind="score"
        )
        if run["status"] == "accepted"
    }
    per_icp = []
    for run in sorted(executions, key=lambda item: int(item["icp_position"])):
        score = scores.get(run["run_id"])
        per_icp.append(
            {
                "icp_position": int(run["icp_position"]),
                "run_id": str(run["run_id"]),
                "attempt": int(run["attempt"]),
                "output_hash": _safe_hash(harness.objects.get(run["output_ref"])),
                "per_icp_score": run.get("per_icp_score"),
                "score_run_id": str(score["run_id"]) if score else None,
            }
        )
    publication = row.get("publication_doc") or {}
    aggregate = next(
        (
            item
            for item in publication.get("final_ranking") or []
            if item["submission_id"] == submission_id
        ),
        None,
    )
    ledger = _ledger_summary(harness, round_id=str(row["round_id"]))
    assert ledger["unaccounted_dispatch_count"] == 0
    evidence.add(
        "round_summary",
        round_id=str(row["round_id"]),
        submission_id=submission_id,
        per_icp=per_icp,
        baseline_aggregate=(
            {
                "final_score": aggregate.get("final_score"),
                "eligible": aggregate.get("eligible"),
                "eligibility_reason": aggregate.get("eligibility_reason"),
            }
            if aggregate
            else None
        ),
        ledger=ledger,
    )


def _fallback_evidence(
    harness: PaidHarness, round_id: str, provider: str
) -> dict[str, Any]:
    rows = harness.service.store.list_runs(round_id, kind="execute")
    failed = [row for row in rows if row.get("champion_restart_required")]
    assert failed
    proofs = []
    connection = harness.connect()
    try:
        with connection.cursor() as cursor:
            for run in failed:
                cursor.execute(
                    """SELECT COALESCE(entry_doc ->> 'provider_attempt',
                                       entry_doc #>> '{call,provider_attempt}'), entry_kind,
                              amount_microusd, funding_source,
                              terminal_response ->> 'status'
                         FROM public.lab_arena_ledger
                        WHERE run_id = %s AND provider = %s
                        ORDER BY created_at, entry_id""",
                    (run["run_id"], provider),
                )
                ledger = cursor.fetchall()
                attempts = sorted({int(item[0]) for item in ledger if item[0]})
                assert set(attempts) <= {1, 2, 3, 4}
                if attempts != [1, 2, 3, 4]:
                    continue  # A concurrent run can observe the already durable latch.
                assert all(item[3] == "miner_key" for item in ledger)
                proofs.append({
                    "run_id": run["run_id"],
                    "attempts": attempts,
                    "provider_statuses": sorted({int(item[4]) for item in ledger if item[4]}),
                    "ledger_rows": len(ledger),
                    "miner_rows": sum(item[3] == "miner_key" for item in ledger),
                    "reserved_liability_microusd": sum(int(item[2] or 0) for item in ledger if item[1] == "reservation"),
                })
    finally:
        connection.close()
    assert proofs, "No account failure completed the bounded initial call and three retries"
    return proofs[0]


def _finish_interrupted_stage_one(harness: PaidHarness) -> dict[str, Any]:
    """Finish a stage whose ninth execution forced a gateway restart."""

    harness.run_stage_with_runners(1)
    closed = harness.service.advance_round(harness.round_id)
    assert closed["status"] == "ok" and harness.status() == "stage1_closed", closed
    scoring_opened = harness.service.advance_round(harness.round_id)
    assert scoring_opened["assignments"] == contracts.STAGE_1_ICP_COUNT, scoring_opened
    harness.run_stage_with_runners(1)
    harness.advance_until("published", runners=1)
    return harness.service.store.get_round(harness.round_id)


def _assert_frozen_owner(
    row: Mapping[str, Any], submission_id: str, hotkey: str
) -> None:
    assert row["champion_funding_frozen"] is True
    assert row["champion_submission_id"] == submission_id
    assert row["champion_hotkey"] == hotkey


def _activate_verified(harness: PaidHarness, factor_ppm: int) -> dict[str, Any]:
    basis = _activate(harness, factor_ppm)
    harness.service = harness.build_service()
    round_row = harness.service.store.get_round(harness.round_id)
    persisted = round_row["reward_basis_doc"]
    assert persisted == basis
    digest = verify_reward_basis_signature(
        persisted,
        public_key_der=harness.signer.public_key_der,
        expected_public_key_hash=harness.signer.public_key_hash,
    )
    assert digest == persisted["reward_basis_hash"]
    burn_hotkey = keypair("paid-champion-burn").ss58_address
    accepted = weight_state.build_accepted_weight_state(
        harness.signer, network="test", genesis_hash="1" * 64, netuid=401,
        epoch=int(persisted["effective_reward_epoch"]), valid_from_block=1,
        valid_until_block=360, reward_basis=persisted, burn_hotkey=burn_hotkey,
        issued_at=persisted["published_at"],
    )
    arena_weights.verify_accepted_weight_state_signature(
        accepted, public_key_der=harness.signer.public_key_der,
        expected_public_key_hash=harness.signer.public_key_hash,
    )
    # A no-new-winner publication has an empty daily king field. Rewards
    # retain the incumbent through the signed basis, which is authoritative.
    champion_hotkey = persisted["king_hotkey"]
    expected_owner = round_row.get("champion_hotkey") or (
        (round_row.get("publication_doc") or {}).get("king_decision") or {}
    ).get("king_hotkey")
    assert champion_hotkey and champion_hotkey == expected_owner
    vector = arena_weights.derive_arena_weights(accepted, [champion_hotkey, burn_hotkey])
    expected_share = champion_values(persisted, persisted["effective_reward_epoch"], [champion_hotkey])["champion_share"]
    assert vector["champion_share_ppb"] == round(expected_share * 1_000_000_000)
    assert vector["burned_residual_ppb"] == 1_000_000_000 - vector["champion_share_ppb"]
    harness.evidence.add(
        "signed_weight_vector_verified", round_id=harness.round_id,
        champion_hotkey=champion_hotkey, reward_outcome=persisted["king_outcome"],
        factor_ppm=factor_ppm, champion_share_ppb=vector["champion_share_ppb"],
        burned_residual_ppb=vector["burned_residual_ppb"],
        reward_basis_hash=persisted["reward_basis_hash"],
    )
    return persisted


def _publish_paid_round(harness: PaidHarness, participants: int) -> dict[str, Any]:
    """Publish with the one validator registered by this isolated chain."""

    _run_stage_one_to_scoring(harness, participants, runners=1)
    harness.advance_until("published", runners=1)
    return harness.service.store.get_round(harness.round_id)


def _start_paid_round(
    harness: PaidHarness, *, epoch: int, challengers=(), pool_percent: int = 5
) -> dict[str, Any]:
    """Use the service's canonical date identity with the accelerated clock."""
    harness.challengers = list(challengers)
    harness.chain.epoch = epoch
    harness.service._config.defaults = replace(
        harness.service._config.defaults,
        max_challengers=max(1, len(challengers)),
        pool_percent=pool_percent,
    )
    configuration = harness.service.create_round(harness.clock.now + timedelta(hours=12))
    harness.round_id = configuration["round_id"]
    for flavor in challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    row = harness.service.store.get_round(harness.round_id)
    for participant in row["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    return row


def test_paid_champion_funding_fallback_and_restoration(
    paid_champion_database, tmp_path: Path
) -> None:
    psycopg2, dsn = paid_champion_database

    def connect():
        return psycopg2.connect(**dsn)

    report = Path(os.environ.get(REPORT_ENV) or "/tmp/lab-arena-champion-paid-e2e.json")
    evidence = Evidence(report)
    management_key = _required_environment(MANAGEMENT_ENV)
    organizer = {
        "openrouter": _required_environment(HOST_OPENROUTER_ENV),
        "deepline": _required_environment(HOST_DEEPLINE_ENV),
        "scrapingdog": _required_environment(HOST_SCRAPINGDOG_ENV),
    }
    miner_deepline = _required_environment(MINER_DEEPLINE_ENV)
    miner_scrapingdog = _required_environment(MINER_SCRAPINGDOG_ENV)
    kms_key_id = _required_environment(KMS_KEY_ENV)
    model = str(os.environ.get(MODEL_ENV) or DEFAULT_AGENT_MODEL).strip()
    preflight = _preflight_paid_source(model)
    evidence.add("source_contract_preflight", **preflight)
    models = sorted(
        {
            model,
            code_review.DEFAULT_REVIEW_MODEL,
            *scoring.DEFAULT_JUDGE_MODELS.values(),
        }
    )
    prices = br.fetch_openrouter_price_table(models)
    evidence.add(
        "price_table_fetched",
        models=models,
        table_hash=contracts.document_hash(prices),
    )
    child = OpenRouterChildKey(management_key)
    runtime_key = ""
    harness: PaidHarness | None = None
    completed = False
    try:
        runtime_key = child.create()
        evidence.add(
            "child_key_created", key_hash=child.key_hash, limit_usd=CHILD_LIMIT_USD
        )
        miner = {
            "openrouter": runtime_key,
            "deepline": miner_deepline,
            "scrapingdog": miner_scrapingdog,
        }
        harness = PaidHarness(
            connect,
            tmp_path,
            organizer_keys=organizer,
            miner_credentials=miner,
            management_key=management_key,
            kms_key_id=kms_key_id,
            price_table=prices,
            model=model,
            evidence=evidence,
        )
        harness.clock.now = datetime.now(timezone.utc)

        precursor = _start_paid_round(
            harness, epoch=61000, challengers=("PaidChampion",)
        )
        precursor = _publish_paid_round(harness, len(precursor["participants"]))
        challenger = next(
            item for item in precursor["participants"] if not item["is_king"]
        )
        _record_round_summary(harness, evidence, precursor, challenger["submission_id"])
        assert precursor["king_outcome"] == "crowned", precursor["publication_doc"][
            "king_decision"
        ]
        winner_id = precursor["publication_doc"]["king_decision"][
            "winner_submission_id"
        ]
        assert winner_id == challenger["submission_id"]
        winner_hotkey = challenger["miner_hotkey"]
        _assert_real_paid_cycle(harness, precursor, winner_id)
        repository_root, remote, promoted_source = _promote_first_winner(
            harness, tmp_path, precursor
        )
        harness.baseline_source = promoted_source(
            "", source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        )
        promoted_source_hash = _safe_hash(harness.baseline_source)
        _activate_verified(harness, 1_000_000)
        _record_round_summary(harness, evidence, precursor, winner_id)
        evidence.add(
            "precursor_promoted",
            round_id=precursor["round_id"],
            submission_id=winner_id,
        )

        # A later repository edit can change the public baseline bytes.  It
        # cannot replace the immutable submission owner or stored ciphertext.
        owner_edit = repository_root / "paid-owner-edit"
        subprocess.run(
            ("git", "clone", "--branch", "lab", str(remote), str(owner_edit)),
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ("git", "config", "user.name", "Paid E2E"), cwd=owner_edit, check=True
        )
        subprocess.run(
            ("git", "config", "user.email", "paid-e2e@example.test"),
            cwd=owner_edit,
            check=True,
        )
        edited_harness = owner_edit / "harness.py"
        edited_harness.write_text(
            edited_harness.read_text() + "\n# Subnet-owner code edit after promotion.\n",
            encoding="utf-8",
        )
        subprocess.run(
            ("git", "add", "harness.py"),
            cwd=owner_edit,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ("git", "commit", "-m", "paid e2e owner edit"),
            cwd=owner_edit,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ("git", "push", "origin", "lab"),
            cwd=owner_edit,
            check=True,
            capture_output=True,
        )
        harness.baseline_source = promoted_source(
            "", source_bundle.MAX_SOURCE_ARCHIVE_BYTES
        )
        winner = harness.service.store.get_submission(winner_id)
        original_credential = harness.service.store.get_submission_credential(
            winner_id, winner_hotkey, "openrouter"
        )
        assert winner is not None and original_credential is not None
        original_ciphertext_hash = _safe_hash(
            original_credential["ciphertext_b64"].encode()
        )
        assert _safe_hash(harness.baseline_source) != promoted_source_hash
        evidence.add(
            "promoted_source_edited",
            prior_source_hash=promoted_source_hash,
            source_hash=_safe_hash(harness.baseline_source),
            champion_submission_id=winner_id,
            champion_hotkey=winner_hotkey,
        )

        clean = _start_paid_round(harness, epoch=61010)
        _assert_frozen_owner(clean, winner_id, winner_hotkey)
        clean = _publish_paid_round(harness, 1)
        _assert_round_complete(harness, clean)
        _assert_real_paid_cycle(
            harness,
            clean,
            _baseline_id(clean),
            require_champion_miner_funding=True,
        )
        clean_basis = _activate_verified(harness, 1_000_000)
        _record_round_summary(harness, evidence, clean, _baseline_id(clean))
        evidence.add(
            "clean_cycle_complete", round_id=clean["round_id"], factor_ppm=1_000_000
        )

        limited = _start_paid_round(harness, epoch=61020, pool_percent=25)
        _assert_frozen_owner(limited, winner_id, winner_hotkey)
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        assert (
            harness.service.advance_round(harness.round_id)["assignments"]
            == contracts.STAGE_1_ICP_COUNT
        )
        scheduled = harness.clock.now
        harness.clock.now = datetime.now(timezone.utc)
        runner = harness.runner(0, parallel=1)
        try:
            # A completed call can be a normal rejected model attempt.
            # Arm the real account failure only after eight accepted ICPs.
            for _ in range(32):
                before_restart = harness.service.store.list_runs(
                    harness.round_id, submission_id=_baseline_id(limited), kind="execute"
                )
                completed_positions = {
                    int(row["icp_position"])
                    for row in before_restart if row["status"] == "accepted"
                }
                assert completed_positions <= set(range(8))
                if completed_positions == set(range(8)):
                    break
                assert runner.run_once(max_claims=1) == 1
            else:
                raise AssertionError("first eight ICPs did not complete")
            before_restart = harness.service.store.list_runs(
                harness.round_id, submission_id=_baseline_id(limited), kind="execute"
            )
            stable = {
                row["run_id"]: _safe_hash(harness.objects.get(row["output_ref"]))
                for row in before_restart
                if row["icp_position"] < 8 and row["status"] == "accepted"
            }
            assert len(stable) == 8
            prior_attempts = {
                (row["run_id"], row["attempt"], row["status"])
                for row in before_restart if row["icp_position"] < 8
            }
            evidence.add("first_eight_accepted", round_id=harness.round_id, output_hashes=stable)
            # A positive cap below recorded usage gives a real exhausted
            # balance; a zero cap was still accepted by the live provider.
            prior_usage = child.usage()["usage"]
            assert prior_usage > 0
            exhausted_limit = prior_usage / 2
            limited_usage = child.configure(disabled=False, limit=exhausted_limit)
            assert limited_usage["usage"] > exhausted_limit
            assert limited_usage["limit_remaining"] == 0
            evidence.add(
                "child_key_limited", key_hash=child.key_hash,
                limit_usd=exhausted_limit, usage_usd=limited_usage["usage"],
                control_propagation_seconds=CHILD_CONTROL_PROPAGATION_SECONDS,
            )
            credit_failure_checkpoint = len(evidence.document["checkpoints"])
            assert runner.run_once(max_claims=1) == 1
        finally:
            runner.close()
            harness.clock.now = scheduled
        proof_credit = _fallback_evidence(harness, limited["round_id"], "openrouter")
        assert proof_credit["provider_statuses"] in ([402], [403])
        credit_errors = [
            item for item in evidence.document["checkpoints"][credit_failure_checkpoint:]
            if item.get("event") == "provider_http_receipt"
            and item.get("provider") == "openrouter"
            and item.get("classification") == "account_credential_failure"
        ]
        assert len(credit_errors) == 4 and all(item["credit_limit_error"] for item in credit_errors)
        evidence.add("fallback_credit_latched", credit_limit_error_responses=4, **proof_credit)
        harness.service = harness.build_service()
        limited = _finish_interrupted_stage_one(harness)
        _assert_round_complete(harness, limited)
        after_restart = harness.service.store.list_runs(
            limited["round_id"], submission_id=_baseline_id(limited), kind="execute"
        )
        for run_id, digest in stable.items():
            row = harness.service.store.get_run(run_id)
            assert row["status"] == "accepted"
            assert _safe_hash(harness.objects.get(row["output_ref"])) == digest
        assert {
            (row["run_id"], row["attempt"], row["status"])
            for row in after_restart if row["icp_position"] < 8
        } == prior_attempts
        failed_nine = [
            row
            for row in after_restart
            if row["icp_position"] == 8 and row.get("champion_restart_required")
        ]
        accepted_nine = [
            row
            for row in after_restart
            if row["icp_position"] == 8 and row["status"] == "accepted"
        ]
        assert (
            len(failed_nine) == 1
            and failed_nine[0]["champion_restart_required"] is True
        )
        assert failed_nine[0].get("output_ref") is None
        assert len(accepted_nine) == 1 and accepted_nine[0]["attempt"] >= 2
        replacement_nine = [row for row in after_restart if row["icp_position"] == 8 and row["attempt"] == 2]
        assert len(replacement_nine) == 1
        assert replacement_nine[0]["champion_funding_sources"]["openrouter"] == "host"
        assert accepted_nine[0]["champion_funding_sources"] == {
            "openrouter": "host",
            "deepline": "miner_key",
            "scrapingdog": "miner_key",
        }
        assert limited["champion_fallback_providers"] == ["openrouter"]
        for row in after_restart:
            if row["status"] == "accepted":
                assert row["champion_funding_sources"] == {
                    "openrouter": "miner_key" if row["icp_position"] < 8 else "host",
                    "deepline": "miner_key", "scrapingdog": "miner_key",
                }
        limited_basis = _activate_verified(harness, 500_000)
        _record_round_summary(harness, evidence, limited, _baseline_id(limited))
        evidence.add(
            "limited_cycle_complete", round_id=limited["round_id"], factor_ppm=500_000
        )

        child.configure(disabled=True, limit=CHILD_LIMIT_USD)
        disabled = _start_paid_round(harness, epoch=61030, pool_percent=20)
        _assert_frozen_owner(disabled, winner_id, winner_hotkey)
        disabled = _publish_paid_round(harness, 1)
        proof_auth = _fallback_evidence(harness, disabled["round_id"], "openrouter")
        assert proof_auth["provider_statuses"] in ([401], [403])
        _assert_round_complete(harness, disabled)
        assert disabled["champion_fallback_providers"] == ["openrouter"]
        for row in harness.service.store.list_runs(disabled["round_id"], kind="execute"):
            if row["status"] == "accepted":
                assert row["champion_funding_sources"] == {
                    "openrouter": "host", "deepline": "miner_key", "scrapingdog": "miner_key",
                }
        disabled_basis = _activate_verified(harness, 500_000)
        _record_round_summary(harness, evidence, disabled, _baseline_id(disabled))
        evidence.add(
            "disabled_cycle_complete",
            round_id=disabled["round_id"],
            factor_ppm=500_000,
            **proof_auth,
        )

        child.configure(disabled=False, limit=CHILD_LIMIT_USD)
        restored = _start_paid_round(harness, epoch=61040, pool_percent=20)
        _assert_frozen_owner(restored, winner_id, winner_hotkey)
        restored = _publish_paid_round(harness, 1)
        _assert_round_complete(harness, restored)
        _assert_real_paid_cycle(
            harness,
            restored,
            _baseline_id(restored),
            require_champion_miner_funding=True,
        )
        basis = _activate_verified(harness, 1_000_000)
        assert [
            clean_basis["champion_reward_factor_ppm"],
            limited_basis["champion_reward_factor_ppm"],
            disabled_basis["champion_reward_factor_ppm"],
            basis["champion_reward_factor_ppm"],
        ] == [1_000_000, 500_000, 500_000, 1_000_000]
        assert (
            champion_values(
                limited_basis, limited_basis["effective_reward_epoch"], [winner_hotkey]
            )["champion_share"]
            == 0.125
        )
        assert (
            champion_values(
                disabled_basis,
                disabled_basis["effective_reward_epoch"],
                [winner_hotkey],
            )["champion_share"]
            == 0.10
        )
        assert (
            champion_values(basis, basis["effective_reward_epoch"], [winner_hotkey])[
                "champion_share"
            ]
            == 0.20
        )
        burn_hotkey = keypair("paid-champion-burn").ss58_address
        accepted_weight_state = weight_state.build_accepted_weight_state(
            harness.signer,
            network="test",
            genesis_hash="1" * 64,
            netuid=401,
            epoch=int(basis["effective_reward_epoch"]),
            valid_from_block=1,
            valid_until_block=360,
            reward_basis=basis,
            burn_hotkey=burn_hotkey,
            issued_at=basis["published_at"],
        )
        arena_weights.verify_accepted_weight_state_signature(
            accepted_weight_state,
            public_key_der=harness.signer.public_key_der,
            expected_public_key_hash=harness.signer.public_key_hash,
        )
        vector = arena_weights.derive_arena_weights(
            accepted_weight_state, [winner_hotkey, burn_hotkey]
        )
        assert vector["champion_share_ppb"] == 200_000_000
        assert vector["burned_residual_ppb"] == 800_000_000
        winner = harness.service.store.get_submission(winner_id)
        assert winner is not None
        openrouter_row = harness.service.store.get_submission_credential(
            winner_id, winner["miner_hotkey"], "openrouter"
        )
        assert openrouter_row is not None
        assert (
            _safe_hash(openrouter_row["ciphertext_b64"].encode())
            == original_ciphertext_hash
        )
        _record_round_summary(harness, evidence, restored, _baseline_id(restored))
        evidence.add(
            "restored_cycle_complete",
            round_id=restored["round_id"],
            factor_ppm=1_000_000,
            effective_reward_epoch=basis["effective_reward_epoch"],
            credential_ciphertext_hash=_safe_hash(
                openrouter_row["ciphertext_b64"].encode()
            ),
        )
        completed = True
    finally:
        active_exception = sys.exc_info()[0] is not None
        deferred_diagnostic_error: Exception | None = None
        try:
            if harness is not None:
                _capture_database_snapshot(harness, evidence)
            if child.key_hash:
                evidence.add("child_key_final_usage", **child.usage())
            if harness is not None:
                final_costs = _ledger_summary(harness)
                if completed:
                    assert final_costs["unaccounted_dispatch_count"] == 0
                evidence.add("final_cost_summary", **final_costs)
        except Exception as exc:
            deferred_diagnostic_error = exc
            try:
                evidence.add(
                    "diagnostic_capture_failed", failure_class=type(exc).__name__
                )
            except Exception:
                pass
        try:
            child.delete()
            evidence.add("child_key_deleted", key_hash=child.key_hash)
        except Exception as exc:
            if deferred_diagnostic_error is None:
                deferred_diagnostic_error = exc
            try:
                evidence.add("child_key_cleanup_failed", key_hash=child.key_hash)
            except Exception:
                pass
        finally:
            runtime_key = ""
        if deferred_diagnostic_error is not None and not active_exception:
            raise deferred_diagnostic_error

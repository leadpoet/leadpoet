"""Disposable ten-ICP full-thinking Arena E2E; paid execution is import-only.

``--preflight`` validates inputs without network or provider spend. The owning
operator calls ``run_e2e`` with an explicit, shared-budget provider transport
and real provider keys. This module never creates keys or starts a paid run on
import. Local credential encryption is simulated with opaque random handles;
source review, agent/scorer entrypoints, broker, ledger, and publication are real.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lab_arena import (
    broker, code_review, contracts, deepline_catalog, integrity, reasoning_capture, runtime, scoring,
    shim, source_bundle, training_capture, training_reader, training_thinking, runtime_version,
)
from tests.lab_arena.champion_funding_paid_e2e_test import (
    Evidence, LocalEntrypointRuntime, PaidHarness, _capture_database_snapshot,
)
from tests.lab_arena import restore_recorded_results538_postgres_test as authority_fixture
from tests.lab_arena.test_lab_arena_service_round import FakeChain, Harness, InProcessApi, keypair


ACTIVATION = "2026-10-01T00:00:00Z"
BENCHMARK_COUNT = 10
MAX_PARALLEL_CLAIMS = 2
MANAGEMENT_PLACEHOLDER = "sk-or-v1-" + "m" * 40
CURRENT_COMPANY_ONLY_ACTIVATIONS = {
    "integrity_from": "2026-09-13T00:00:00Z",
    "company_quality_from": None,
    "company_judgment_from": "2026-10-19T00:00:00+00:00",
    "signal_judgment_from": "2026-10-20T00:00:00+00:00",
    "intent_details_from": "2026-09-15T00:00:00+00:00",
}
EXPECTED_COMPANY_ONLY_SCORER_POLICY = dict(scoring.build_scorer_policy(
    scoring_adapter_version=integrity.SCORING_ADAPTER,
    company_judgment_cache=True,
    intent_details=True,
    provider_observation_handoff=True,
    normalize_intent_scale=True,
    fit_fp_receipt_authority=True,
), signal_judgment_policy="signal_judgment_v1")
EXPECTED_COMPANY_ONLY_SCORER_POLICY_HASH = contracts.document_hash(
    EXPECTED_COMPANY_ONLY_SCORER_POLICY
)


def _private_file(path: Path, value: Any, *, markdown: bool = False) -> None:
    """Write exact private evidence once, before an assertion can discard it."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(value if markdown else json.dumps(
                value, sort_keys=True, indent=2, ensure_ascii=False,
            ))
            stream.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _preserve_object_store(harness: "FullThinkingHarness", report: Path) -> Path:
    """Keep the exact objects referenced by the disposable database snapshot."""
    identity = str(getattr(harness, "round_id", "pre-round"))
    destination = report.parent / (report.stem + "-objects-" + identity)
    destination.mkdir(mode=0o700)
    manifest = []
    for source in sorted(harness.objects_root.rglob("*")):
        target = destination / source.relative_to(harness.objects_root)
        if source.is_dir():
            target.mkdir(mode=0o700, parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            data = source.read_bytes()
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
            manifest.append({"ref": str(source.relative_to(harness.objects_root)),
                             "sha256": contracts.hash_bytes(data), "bytes": len(data)})
    _private_file(destination / "_manifest.json", manifest)
    return destination


def _capture_review_state(harness: "FullThinkingHarness", evidence: Evidence) -> None:
    """Save bounded committed review state without a claim token or source body."""
    if not getattr(harness, "round_id", None):
        return
    rows = []
    for submission in harness.service.store.list_submissions(harness.round_id):
        review = submission.get("code_review_doc") or {}
        research = review.get("research_control") or {}
        rows.append({
            "submission_id": submission["submission_id"],
            "submission_status": submission["status"],
            "review_status": submission.get("code_review_status"),
            "review_attempts": submission.get("code_review_attempts"),
            "review_claim_present": submission.get("code_review_claim") is not None,
            "review_error_code": review.get("error_code"),
            "research_classification": research.get("classification"),
            "research_policy_hash": research.get("policy_hash"),
        })
    evidence.add("committed_review_state", submissions=rows)


def _capture_private_ledger(harness: "FullThinkingHarness", report: Path,
                            evidence: Evidence) -> None:
    """Save exact append-only call records before the disposable DB exits."""
    if not getattr(harness, "round_id", None):
        return
    connection = harness.connect()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT to_jsonb(entry) FROM public.lab_arena_ledger entry "
                "WHERE entry.round_id=%s ORDER BY entry.entry_id",
                (harness.round_id,),
            )
            rows = [item[0] for item in cursor.fetchall()]
    finally:
        connection.close()
    encoded = json.dumps(rows, sort_keys=True, indent=2, ensure_ascii=False).encode("utf-8") + b"\n"
    for secret in harness.sandbox.forbidden:
        if secret and (secret in encoded or base64.b64encode(secret) in encoded):
            raise AssertionError("credential appeared in private ledger snapshot")
    path = report.parent / (report.stem + "-private-ledger.json")
    _private_file(path, rows)
    evidence.add("private_ledger_snapshot", path=str(path), rows=len(rows),
                 sha256=contracts.hash_bytes(encoded))


def full_thinking_database():
    """Disposable current authority through the full-thinking source freeze."""
    # The 538 fixture carries the 517 write-once trigger and scorer history.
    # It omits several independently tested mainline branches. Apply their
    # exact guarded migrations before the post-538 tail; the dated 542-546
    # October 22 data repairs do not belong in a fresh round database.
    database = authority_fixture.database.__wrapped__()
    try:
        psycopg2, dsn = next(database)
        connection = psycopg2.connect(**dsn)
        connection.autocommit = True  # Mainline concurrent index migrations require it.
        try:
            with connection.cursor() as cursor:
                for name in (
                    "462-lab-arena-deepline-native-settlement-binding.sql",
                    "505-lab-arena-execute-reconciliation-run-scope.sql",
                    "506-lab-arena-closed-reconciliation-eligible-run-scope.sql",
                    "507-lab-arena-closed-provider-round-membership.sql",
                    "508-lab-arena-openrouter-uncertain-run-index.sql",
                    "518-lab-arena-single-tool-response-cache.sql",
                    "524-lab-arena-proven-funding-recovery.sql",
                    "525-lab-arena-private-registration-helper.sql",
                    "526-lab-arena-funding-proof-scalars.sql",
                    "528-lab-arena-internal-table-public-access.sql",
                    "529-lab-arena-hide-public-icp-examples.sql",
                    "531-lab-arena-round-open-call-index.sql",
                    "533-lab-arena-cancel-latest-call-heads.sql",
                    "534-lab-arena-remove-open-call-index.sql",
                    "539-lab-arena-held-recovery-provider-admission.sql",
                    "540-lab-arena-incomplete-participant-publication.sql",
                    "541-lab-arena-public-execution-failure-reason.sql",
                    "547-lab-arena-unmeasured-terminal-publication.sql",
                    "548-lab-arena-company-judgment-validation-projection.sql",
                    "549-lab-arena-open-scoring-frozen-config-projection.sql",
                    "550-lab-arena-tool-reply-own-call-cache.sql",
                    "551-lab-arena-expire-leases-shared-round-lock.sql",
                    "552-lab-arena-expiry-recovery-candidate-locks.sql",
                    "553-lab-arena-paid-native-request-index.sql",
                    "554-lab-arena-funding-proof-throughput.sql",
                    "555-lab-arena-full-thinking-six-dollar-cap.sql",
                    "556-lab-arena-research-review-result.sql",
                    "557-lab-arena-closed-provider-scoped-candidates.sql",
                    "558-lab-arena-funding-proof-keyed-lookups.sql",
                    "559-lab-arena-deepline-uncertain-round-entry-index.sql",
                    "560-lab-arena-closed-provider-score-run-scope.sql",
                    "561-lab-arena-full-thinking-champion-baseline.sql",
                    "562-lab-arena-full-thinking-catalog-commit.sql",
                    "563-lab-arena-training-publication-eligibility.sql",
                ):
                    cursor.execute((REPO / "scripts" / name).read_text())
        finally:
            connection.close()
        yield psycopg2, dsn
    finally:
        database.close()


def preflight(*, source_archive: Path, baseline_archive: Path,
              icp_file: Path, dependency_dir: Path, baseline_url: str,
              model: str = "deepseek/deepseek-v4.1-flash-1017") -> dict[str, Any]:
    """Check submitted bytes and frozen inputs without making network calls."""
    submitted = source_archive.read_bytes()
    baseline = baseline_archive.read_bytes()
    submitted_facts = source_bundle.validate_source_archive(submitted)
    source_bundle.validate_source_archive(baseline)
    python_files = 0
    with tarfile.open(fileobj=io.BytesIO(submitted), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                compile(archive.extractfile(member).read(), member.name, "exec")
                python_files += 1
    if not python_files:
        raise ValueError("submitted archive has no Python source")
    document = json.loads(icp_file.read_text(encoding="utf-8"))
    icps = document.get("icps") if isinstance(document, dict) else None
    if (not isinstance(icps, list) or len(icps) != BENCHMARK_COUNT
            or any(not isinstance(row, dict) or not isinstance(row.get("icp_id"), str)
                   or not row["icp_id"] for row in icps)
            or len({row["icp_id"] for row in icps}) != BENCHMARK_COUNT):
        raise ValueError("ten distinct genuine ICP documents are required")
    if not dependency_dir.is_dir():
        raise ValueError("isolated agent dependency directory is missing")
    from lab_arena.service import FULL_THINKING_BASELINE_URL_RE
    if FULL_THINKING_BASELINE_URL_RE.fullmatch(baseline_url) is None:
        raise ValueError("full-thinking baseline URL must pin a source commit")
    registry = reasoning_capture.load_registry()
    if not any(row["model_id"] == model and row["api"] == "openrouter.chat"
               for row in registry["approved_routes"]):
        raise ValueError("chosen agent model has no reviewed Chat route")
    return {
        "icps": icps,
        "registry": registry,
        "summary": {
            "icp_count": len(icps),
            "icp_hash": contracts.document_hash(icps),
            "source_hash": contracts.hash_bytes(submitted),
            "baseline_source_hash": contracts.hash_bytes(baseline),
            "source_size_bytes": submitted_facts["source_size_bytes"],
            "compiled_python_files": python_files,
            "model": model,
            "provider_calls": 0,
        },
    }


class LocalCredentialHandles:
    """Local-only ciphertext boundary; plain runtime keys remain in memory."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str, str], tuple[str, str]] = {}

    def validate_and_encrypt(self, submitted: Mapping[str, Any], *,
                             submission_id: str, miner_hotkey: str) -> dict[str, str]:
        if not all(isinstance(submitted.get(name), str) and submitted[name]
                   for name in ("openrouter_api_key", "openrouter_management_key",
                                "deepline_api_key")):
            raise ValueError("local E2E credential fields are incomplete")
        keys = {
            "openrouter": submitted["openrouter_api_key"],
            "deepline": submitted["deepline_api_key"],
        }
        if submitted.get("scrapingdog_api_key"):
            keys["scrapingdog"] = submitted["scrapingdog_api_key"]
        encrypted = {}
        for provider, secret in keys.items():
            handle = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
            self._values[(submission_id, miner_hotkey, provider)] = (handle, secret)
            encrypted[provider] = handle
        return encrypted

    def runtime_key(self, row: Mapping[str, Any], provider: str) -> str:
        identity = (row["submission_id"], row["miner_hotkey"], provider)
        handle, secret = self._values[identity]
        if row["provider"] != provider or row["ciphertext_b64"] != handle:
            raise ValueError("local E2E credential identity changed")
        return secret


class FullThinkingLocalRuntime(LocalEntrypointRuntime):
    """Execute the production entrypoints with staged capture and web bridge."""

    def _environment(self, spec: runtime.SandboxSpec, site_dir: Path) -> dict[str, str]:
        environment = super()._environment(spec, site_dir)
        if spec.web_bridge_path is not None:
            environment["LAB_ARENA_WEB_EGRESS_SOCKET"] = str(
                spec.socket_path.parent / runtime.SANDBOX_WEB_SOCKET_NAME
            )
        return environment

    def run_icp(self, spec: runtime.SandboxSpec, **_: Any) -> runtime.SandboxResult:
        process_dir = Path(tempfile.mkdtemp(prefix="full-thinking-entry-", dir=str(self.root)))
        try:
            (process_dir / "sitecustomize.py").write_text(shim.SITECUSTOMIZE_SOURCE)
            environment = self._environment(spec, process_dir)
            if spec.entry_command == runtime.SCORER_ENTRY_COMMAND:
                command = [sys.executable, "-m", "lab_arena.scorer_entrypoint"]
                cwd = process_dir
            else:
                if not all((spec.source_dir, spec.web_bridge_path,
                            spec.automatic_capture_path)):
                    raise AssertionError("agent source, web bridge, or capture was not staged")
                # -I matches production agent isolation. Public web requests
                # enter _web_access; paid provider requests must use the worker
                # socket explicitly and cannot escape through that bridge.
                code = (
                    "import sys\nfrom pathlib import Path\n"
                    "sys.path.insert(0, sys.argv[4])\n"
                    "from lab_arena import agent_entrypoint\n"
                    "agent_entrypoint.DEPENDENCY_DIR = Path(sys.argv[5])\n"
                    "agent_entrypoint.WEB_BRIDGE_PATH = Path(sys.argv[6])\n"
                    "agent_entrypoint.AUTOMATIC_CAPTURE_PATH = Path(sys.argv[7])\n"
                    "with agent_entrypoint._web_access():\n"
                    "    agent_entrypoint.run(source_dir=Path(sys.argv[1]), "
                    "input_path=Path(sys.argv[2]), output_path=Path(sys.argv[3]))\n"
                )
                command = [
                    sys.executable, "-I", "-u", "-B", "-c", code,
                    str(spec.source_dir), str(spec.input_dir / runtime.INPUT_FILE_NAME),
                    str(spec.output_path), str(REPO),
                    str(Path(os.environ["LAB_ARENA_FULL_E2E_MODEL_DEPS"])),
                    str(spec.web_bridge_path), str(spec.automatic_capture_path),
                ]
                cwd = spec.source_dir
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    command, cwd=str(cwd), env=environment,
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=spec.wall_clock_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                return runtime.fake_result(
                    exit_code=None, timed_out=True, output_bytes=None,
                    stderr=b"subprocess timeout",
                )
            for secret in self.forbidden:
                if secret and (secret in completed.stdout or secret in completed.stderr):
                    raise AssertionError("secret appeared in child diagnostics")
            output = spec.output_path.read_bytes() if spec.output_path.is_file() else None
            if output is not None and len(output) > runtime.MAX_OUTPUT_BYTES:
                output = None
            if output is not None and any(secret and secret in output for secret in self.forbidden):
                raise AssertionError("secret appeared in child output")
            if output is not None:
                kind = "scorer" if spec.entry_command == runtime.SCORER_ENTRY_COMMAND else "agent"
                directory = self.evidence.path.parent / ("private-" + kind + "-results")
                directory.mkdir(mode=0o700, exist_ok=True)
                path = directory / (hashlib.sha256(output).hexdigest() + ".json")
                try:
                    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    if path.read_bytes() != output:
                        raise AssertionError("private process output hash collision")
                else:
                    with os.fdopen(descriptor, "wb") as stream:
                        stream.write(output)
            if completed.returncode and completed.stderr:
                private = self.evidence.path.parent / "private-process-errors"
                private.mkdir(mode=0o700, exist_ok=True)
                path = private / (hashlib.sha256(completed.stderr).hexdigest() + ".txt")
                path.write_bytes(completed.stderr)
                path.chmod(0o600)
            self.evidence.add(
                "sandbox_process_receipt",
                entrypoint="scorer" if spec.entry_command == runtime.SCORER_ENTRY_COMMAND else "agent",
                exit_code=completed.returncode,
                timed_out=False,
                stdout_hash=contracts.hash_bytes(completed.stdout),
                stderr_hash=contracts.hash_bytes(completed.stderr),
                output_hash=contracts.hash_bytes(output) if output is not None else None,
            )
            return runtime.SandboxResult(
                exit_code=completed.returncode, timed_out=False,
                stdout=b"", stderr=b"", stdout_truncated=bool(completed.stdout),
                stderr_truncated=bool(completed.stderr),
                wall_seconds=time.monotonic() - started, cpu_seconds=0.0,
                max_rss_bytes=0, output_bytes=output,
                output_path=str(spec.output_path),
                output_error=None if output is not None else "missing_output",
                automatic_capture_summary=(runtime.read_automatic_capture_summary(spec)
                                           if spec.entry_command == runtime.AGENT_ENTRY_COMMAND
                                           else None),
            )
        finally:
            shutil.rmtree(process_dir, ignore_errors=True)


class FullThinkingChain(FakeChain):
    """Register the challenger under a distinct finalized owner in the local chain."""

    def __init__(self, runners: list[str]) -> None:
        super().__init__(runners)
        self._miner_owners: dict[str, str] = {}

    def register_miner(self, hotkey: str) -> None:
        self._miner_owners[hotkey] = keypair("owner-" + hotkey).ss58_address

    def metagraph(self, finalized: bool = True):
        snapshot = super().metagraph(finalized=finalized)
        miners = tuple(sorted(self._miner_owners))
        if any(hotkey in snapshot.hotkeys for hotkey in miners):
            raise AssertionError("challenger registration overlaps validator")
        return replace(
            snapshot,
            hotkeys=snapshot.hotkeys + miners,
            coldkeys=snapshot.coldkeys + tuple(self._miner_owners[key] for key in miners),
            validator_permit=snapshot.validator_permit + (False,) * len(miners),
            stake=snapshot.stake + (1_000.0,) * len(miners),
            active=snapshot.active + (True,) * len(miners),
        )


class FullThinkingCaptureApi(InProcessApi):
    """Expose training capture and scorer controls to the local runner."""

    def signal_judgment(self, run_id, lease_token, request):
        return self._guard(lambda: self.service.handle_signal_judgment(
            run_id, lease_token, request))

    def quota_usage(self, run_id, lease_token, *, include_sourcing_cost=False):
        return self._guard(lambda: self.service.handle_quota_snapshot(
            run_id, lease_token, include_sourcing_cost=include_sourcing_cost))

    def trajectory(self, run_id, lease_token, events):
        return self._guard(lambda: self.service.handle_trajectory(
            run_id, lease_token, {"events": list(events)}))

    def training_input(self, run_id, lease_token, document):
        return self._guard(lambda: self.service.handle_training_input(run_id, lease_token, document))

    def training_observations(self, run_id, lease_token, document):
        return self._guard(lambda: self.service.handle_training_observations(run_id, lease_token, document))

    def training_observation_segment(self, run_id, lease_token, document):
        return self._guard(lambda: self.service.handle_training_observation_segment(run_id, lease_token, document))


class FullThinkingHarness(PaidHarness):
    """Existing signed paid harness with ten frozen ICPs and a real scorer."""

    def __init__(self, connect, tmp_path: Path, *, icps: list[dict[str, Any]],
                 baseline_archive: bytes, baseline_url: str, registry: dict[str, Any],
                 model: str, organizer_keys: Mapping[str, str],
                 miner_credentials: Mapping[str, str], price_table: Mapping[str, Any],
                 transport: Any,
                 proxy_worker_pool: Any, evidence: Evidence,
                 frozen_deepline_catalog: Mapping[str, Any],
                 policy_activations: Mapping[str, str | None]) -> None:
        self._icps = icps
        self._baseline_archive = baseline_archive
        self._baseline_url = baseline_url
        self._registry = registry
        self._frozen_deepline_catalog = deepline_catalog.validate_catalog(frozen_deepline_catalog)
        self._policy_activations = dict(policy_activations)
        self._proxy_worker_pool = proxy_worker_pool
        self._paid_organizer_keys = dict(organizer_keys)
        self._paid_miner_credentials = dict(miner_credentials)
        self._paid_management_key = MANAGEMENT_PLACEHOLDER
        self._paid_price_table = broker.validate_price_table(price_table)
        self._paid_model = model
        self._paid_transport = transport
        self._paid_credentials = LocalCredentialHandles()
        self.evidence = evidence
        self._full_thinking_objects_key = "full-thinking-" + secrets.token_hex(8)
        Harness.__init__(self, connect, tmp_path, challengers=[], runners=["alpha"])
        self.chain = FullThinkingChain(self.runner_keys)
        self.chain.netuid = 401
        self.baseline_source = baseline_archive
        self.sandbox = FullThinkingLocalRuntime(
            tmp_path / "full-thinking-processes",
            list(organizer_keys.values()) + list(miner_credentials.values()),
            evidence,
        )
        self.service = self.build_service()
        self.api_factory = lambda: FullThinkingCaptureApi(self.service)

    def objects_key(self) -> str:
        return self._full_thinking_objects_key

    def submit(self, flavor: str, round_id: str, *, miner_label: str = "") -> str:
        miner = keypair("svc-miner-" + (miner_label or flavor))
        self.chain.register_miner(miner.ss58_address)
        return super().submit(flavor, round_id, miner_label=miner_label)

    def build_service(self):
        service = super().build_service()
        service.config.training_object_store = self.objects
        original_broker_factory = service.config.broker_factory
        request_capture = training_capture.RequestCapture(self.objects).prepare
        def broker_factory(owner, round_row):
            instance = original_broker_factory(owner, round_row)
            instance._capture_request = request_capture
            return instance
        service.config.broker_factory = broker_factory
        defaults = service.config.defaults
        service.config.defaults = replace(
            defaults,
            benchmark_icp_count=BENCHMARK_COUNT,
            parallel_twenty_icp_execution=True,
            execution_sequence_from=None,
            per_icp_cost_policy=True,
            flat_per_icp_cost_policy=True,
            rewards_enabled=False,
            baseline_source_url=self._baseline_url,
            reasoning_capture_from=ACTIVATION,
            reasoning_baseline_source_sha256=contracts.hash_bytes(self._baseline_archive),
            reasoning_baseline_source_url=self._baseline_url,
            **self._policy_activations,
        )
        service.config.reasoning_registry_source = lambda: self._registry
        service.config.deepline_catalog_source = (
            lambda *, allow_people=False: self._frozen_deepline_catalog
        )
        service.config.daily_icp_source = lambda **kwargs: {
            "status": "ready", "set_id": int(kwargs["set_id"]), "icps": self._icps,
        }
        return service

    def runner(self, index: int, parallel: int = MAX_PARALLEL_CLAIMS):
        worker = super().runner(index, parallel=parallel)
        if worker._config.round_id != self.round_id:
            raise AssertionError("runner is not pinned to this E2E round")
        worker._config.proxy_worker_pool = self._proxy_worker_pool
        return worker

    def run_stage_with_runners(self, count: int = 1) -> None:
        if count != 1:
            raise ValueError("bounded paid E2E uses one runner")
        scheduled = self.clock.now
        self.clock.now = datetime.now(timezone.utc)
        worker = self.runner(0, parallel=MAX_PARALLEL_CLAIMS)
        completed_cursor = 0
        failed_sandboxes = 0
        try:
            while worker.run_once(max_claims=MAX_PARALLEL_CLAIMS):
                for receipt in worker.completed[completed_cursor:]:
                    run_id = receipt.get("run_id")
                    run = self.service.store.get_run(run_id) if run_id else None
                    if run is None or run.get("kind") not in ("execute", "score"):
                        continue
                    diagnostic = (run.get("result_doc") or {}).get("failure_diagnostic") or {}
                    stage = diagnostic.get("stage")
                    if run.get("status") == "failed" and stage in ("sandbox", "sandbox_output"):
                        failed_sandboxes += 1
                    elif run.get("status") == "accepted":
                        failed_sandboxes = 0
                    if failed_sandboxes >= 2:
                        self.evidence.add("sandbox_fail_fast", round_id=self.round_id,
                                          consecutive_failures=failed_sandboxes,
                                          last_run_id=run_id, failure_stage=stage)
                        raise AssertionError("two consecutive sandbox runs failed")
                completed_cursor = len(worker.completed)
        finally:
            worker.close()
            self.clock.now = scheduled
        if any(item.get("error") for item in worker.completed):
            raise AssertionError("runner abandoned an assignment")


def run_e2e(*, source_archive: Path, baseline_archive: Path, icp_file: Path,
            dependency_dir: Path, baseline_url: str, model: str,
            organizer_keys: Mapping[str, str], miner_credentials: Mapping[str, str],
            price_table: Mapping[str, Any], transport_factory: Any,
            proxy_worker_pool: Any,
            frozen_deepline_catalog: Mapping[str, Any],
            policy_activations: Mapping[str, str | None] = CURRENT_COMPANY_ONLY_ACTIVATIONS,
            expected_scorer_policy_hash: str = EXPECTED_COMPANY_ONLY_SCORER_POLICY_HASH,
            report: Path,
            database_factory: Any = full_thinking_database) -> dict[str, Any]:
    """Run real source review, worker/scorer, publication, and strict export.

    The caller owns paid authorization and supplies a fail-closed transport
    enforcing the shared OpenRouter and other-provider budgets before send.
    """
    # Strict export binds every runner result to exact committed source. Check
    # the import-time identity as well as the current checkout before any paid
    # review or model call; an in-process commit cannot repair a dirty import.
    source_identity = runtime_version.SOURCE_METADATA
    current_identity = runtime_version.source_metadata(runtime_version.__file__)
    if (source_identity != current_identity
            or source_identity.get("validator_source_origin") != "git_checkout"
            or source_identity.get("validator_source_dirty") != "clean"):
        raise ValueError("full-thinking E2E requires a clean committed runner checkout")
    prepared = preflight(
        source_archive=source_archive, baseline_archive=baseline_archive,
        icp_file=icp_file, dependency_dir=dependency_dir,
        baseline_url=baseline_url, model=model,
    )
    if transport_factory is None or proxy_worker_pool is None:
        raise ValueError("budget transport and verified public web pool are required")
    for label, keys in (("organizer", organizer_keys), ("miner", miner_credentials)):
        if any(not isinstance(keys.get(provider), str) or not keys[provider]
               for provider in contracts.PROVIDERS):
            raise ValueError("%s provider keys are incomplete" % label)
    required_models = {model, code_review.DEFAULT_REVIEW_MODEL,
                       *scoring.DEFAULT_JUDGE_MODELS.values()}
    checked_prices = broker.validate_price_table(price_table)
    checked_catalog = deepline_catalog.validate_catalog(frozen_deepline_catalog)
    if checked_catalog.get("allow_people") is not False:
        raise ValueError("company-only Deepline catalog is required")
    if not required_models <= set(checked_prices["models"]):
        raise ValueError("current model price table is incomplete")
    allowed_activations = {
        "integrity_from", "company_quality_from", "company_judgment_from",
        "signal_judgment_from", "intent_details_from",
    }
    if set(policy_activations) != allowed_activations:
        raise ValueError("explicit company-only policy activations are required")
    if not isinstance(expected_scorer_policy_hash, str) or not expected_scorer_policy_hash.startswith("sha256:"):
        raise ValueError("expected scorer policy hash is required")
    report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    evidence = Evidence(report)
    evidence.add("preflight", **prepared["summary"])
    previous = {
        "LAB_ARENA_PAID_SOURCE_ARCHIVE": os.environ.get("LAB_ARENA_PAID_SOURCE_ARCHIVE"),
        "LAB_ARENA_PAID_MODEL_DEPS": os.environ.get("LAB_ARENA_PAID_MODEL_DEPS"),
        "LAB_ARENA_FULL_E2E_MODEL_DEPS": os.environ.get("LAB_ARENA_FULL_E2E_MODEL_DEPS"),
    }
    os.environ.update({
        "LAB_ARENA_PAID_SOURCE_ARCHIVE": str(source_archive),
        "LAB_ARENA_PAID_MODEL_DEPS": str(dependency_dir),
        "LAB_ARENA_FULL_E2E_MODEL_DEPS": str(dependency_dir),
    })
    database = database_factory()
    harness = None
    temporary = None
    try:
        psycopg2, dsn = next(database)
        def connect():
            return psycopg2.connect(**dsn)
        temporary = tempfile.mkdtemp(prefix="arena-full-thinking-e2e-")
        transport = transport_factory(evidence)
        harness = FullThinkingHarness(
            connect, Path(temporary), icps=prepared["icps"],
            baseline_archive=baseline_archive.read_bytes(),
            baseline_url=baseline_url,
            registry={**prepared["registry"], "baseline_route": {
                "model_id": model, "api": "openrouter.chat",
            }},
            model=model, organizer_keys=organizer_keys,
            miner_credentials=miner_credentials, price_table=checked_prices,
            transport=transport,
            proxy_worker_pool=proxy_worker_pool, evidence=evidence,
            frozen_deepline_catalog=checked_catalog,
            policy_activations=policy_activations,
        )
        harness.clock.now = datetime.now(timezone.utc)
        harness.challengers = ["FullThinkingCandidate"]
        harness.chain.epoch = 61000
        # Review admission waits until one hour before the submission cutoff.
        # Use a real clock cutoff inside that window while signed submissions
        # are still open; the database enforces the same window independently.
        created = harness.service.create_round(harness.clock.now + timedelta(minutes=45))
        harness.round_id = created["round_id"]
        config = harness.service.store.get_round(harness.round_id)["configuration_doc"]
        if config.get("reasoning_capture_policy") != reasoning_capture.POLICY:
            raise AssertionError("full-thinking round did not activate")
        if contracts.document_hash(config["scorer_policy"]) != expected_scorer_policy_hash:
            raise AssertionError("frozen scorer policy differs from expected production policy")
        if (config["stage_1_icp_count"], config["stage_2_icp_count"]) != (5, 5):
            raise AssertionError("benchmark did not freeze five ICPs per stage")
        if config.get("deepline_catalog") != checked_catalog:
            raise AssertionError("round did not freeze the supplied Deepline catalog")
        evidence.add("round_frozen", round_id=harness.round_id,
                     scorer_policy_hash=expected_scorer_policy_hash,
                     reasoning_registry_hash=contracts.document_hash(config["reasoning_route_registry"]),
                     deepline_catalog_hash=checked_catalog["catalog_hash"],
                     stage_counts=[5, 5])
        harness.submit("FullThinkingCandidate", harness.round_id)
        harness.clock.advance_to(harness.schedule()["submission_cutoff"])
        committed = harness.service.advance_round(harness.round_id)
        if committed.get("status") != "ok":
            raise AssertionError("benchmark did not commit")
        harness.clock.advance_to(harness.schedule()["stage_1_start"])
        harness.advance_until("published", runners=1)
        row = harness.service.store.get_round(harness.round_id)
        candidate = next(item for item in row["participants"] if not item["is_king"])
        submission_id = candidate["submission_id"]
        executions = sorted(
            (run for run in harness.service.store.list_runs(
                row["round_id"], submission_id=submission_id, kind="execute")
             if run["status"] == "accepted"),
            key=lambda item: int(item["icp_position"]),
        )
        scores = [run for run in harness.service.store.list_runs(
            row["round_id"], submission_id=submission_id, kind="score")
            if run["status"] == "accepted"]
        baseline = next(item for item in row["participants"] if item["is_king"])
        baseline_executions = sorted(
            (run for run in harness.service.store.list_runs(
                row["round_id"], submission_id=baseline["submission_id"], kind="execute")
             if run["status"] == "accepted"),
            key=lambda item: int(item["icp_position"]),
        )
        baseline_scores = [run for run in harness.service.store.list_runs(
            row["round_id"], submission_id=baseline["submission_id"], kind="score")
            if run["status"] == "accepted"]
        baseline_private = report.parent / (report.stem + "-baseline-training")
        baseline_examples = []
        for run in baseline_executions:
            example = training_reader.export_run(harness.service.store, harness.objects, run["run_id"])
            baseline_examples.append(example)
            name = str(run["icp_position"])
            _private_file(baseline_private / (name + "-raw.json"), example)
            trace = training_thinking.readable_trace(example)
            _private_file(baseline_private / (name + "-readable-trace.json"), trace)
            _private_file(baseline_private / (name + "-readable-trace.md"),
                          training_thinking.readable_trace_markdown(
                              trace, raw_export_name=name + "-raw.json"), markdown=True)
        baseline_splits, baseline_manifest = training_thinking.build(baseline_examples)
        for split in ("train", "eval", "analysis"):
            _private_file(baseline_private / ("strict-" + split + ".json"),
                          baseline_splits[split])
        _private_file(baseline_private / "strict-manifest.json", baseline_manifest)
        evidence.add("baseline_training_export", round_id=row["round_id"],
                     accepted_executions=len(baseline_executions),
                     accepted_scores=len(baseline_scores),
                     complete_examples=sum(example.get("completeness", {}).get("complete") is True
                                           for example in baseline_examples),
                     strict_records=len(baseline_splits["train"]) + len(baseline_splits["eval"]))
        examples = []
        private = report.parent / (report.stem + "-training")
        for run in executions:
            example = training_reader.export_run(harness.service.store, harness.objects, run["run_id"])
            examples.append(example)
            _private_file(private / (str(run["icp_position"]) + "-raw.json"), example)
        for run, example in zip(executions, examples):
            name = str(run["icp_position"])
            trace = training_thinking.readable_trace(example)
            _private_file(private / (name + "-readable-trace.json"), trace)
            _private_file(private / (name + "-readable-trace.md"),
                          training_thinking.readable_trace_markdown(
                              trace, raw_export_name=name + "-raw.json"), markdown=True)
        combined, combined_manifest = training_thinking.build(examples)
        _private_file(private / "strict-train.json", combined["train"])
        _private_file(private / "strict-eval.json", combined["eval"])
        _private_file(private / "strict-analysis.json", combined["analysis"])
        _private_file(private / "strict-manifest.json", combined_manifest)
        per_icp = []
        score_by_execution = {run["scored_run_id"]: run for run in scores}
        for run, example in zip(executions, examples):
            splits, manifest = training_thinking.build([example])
            count = len(splits["train"]) + len(splits["eval"])
            judged = score_by_execution.get(run["run_id"])
            output = example.get("output") or {}
            companies = output.get("companies") if isinstance(output, Mapping) else None
            raw_score = run.get("per_icp_score")
            if raw_score is None:
                raw_score = judged.get("per_icp_score") if judged is not None else None
            per_icp.append({"icp_position": int(run["icp_position"]),
                            "icp_id": prepared["icps"][int(run["icp_position"])]["icp_id"],
                            "run_id": run["run_id"],
                            "score_run_id": judged["run_id"] if judged is not None else None,
                            "raw_score": float(raw_score) if raw_score is not None else None,
                            "returned_leads": len(companies) if isinstance(companies, list) else None,
                            "records": count,
                            "complete": (example.get("completeness") or {}).get("complete") is True,
                            "exclusion_reasons": manifest["counts"]["exclusion_reasons"],
                            "cost_eligibility": example.get("competition_cost_eligibility")})
        evidence.add("per_icp_observed", round_id=row["round_id"],
                     submission_id=submission_id, per_icp=per_icp)
        if (len(executions) != BENCHMARK_COUNT or len(scores) != BENCHMARK_COUNT
                or {run["scored_run_id"] for run in scores}
                != {run["run_id"] for run in executions}):
            raise AssertionError("ten accepted executions and linked real scores are required")
        if {item["icp_position"] for item in per_icp} != set(range(BENCHMARK_COUNT)):
            raise AssertionError("execution ICP positions are incomplete")
        if any(item["records"] < 1 or not item["complete"] for item in per_icp):
            raise AssertionError("every ICP requires a complete strict training record")
        combined_count = len(combined["train"]) + len(combined["eval"])
        if combined_count < BENCHMARK_COUNT:
            raise AssertionError("combined strict dataset has fewer than ten records")
        publication = row.get("publication_doc") or {}
        aggregate = next((item for item in publication.get("final_ranking") or []
                          if item["submission_id"] == submission_id), None)
        if aggregate is None or aggregate.get("final_score") is None:
            raise AssertionError("candidate aggregate was not published")
        public = harness.service.public_round(row["round_id"])
        public_aggregate = next((item for item in public.get("final_ranking") or []
                                 if item["submission_id"] == submission_id), None)
        if public_aggregate != aggregate:
            raise AssertionError("public aggregate differs from stored publication")
        evidence.add("full_thinking_e2e_complete", round_id=row["round_id"],
                     submission_id=submission_id, per_icp=per_icp,
                     aggregate_score=aggregate["final_score"],
                     aggregate_eligible=aggregate.get("eligible"),
                     aggregate_eligibility_reason=aggregate.get("eligibility_reason"),
                     score_run_count=len(scores),
                     strict_dataset_records=combined_count,
                     strict_dataset_hash=combined_manifest["dataset_version"])
        return {"round_id": row["round_id"], "submission_id": submission_id,
                "aggregate": aggregate, "per_icp": per_icp,
                "strict_dataset_records": combined_count,
                "strict_dataset_hash": combined_manifest["dataset_version"]}
    finally:
        if harness is not None:
            preservation_errors = []
            try:
                _capture_review_state(harness, evidence)
            except Exception as exc:
                preservation_errors.append("reviews:" + type(exc).__name__)
            try:
                _capture_database_snapshot(harness, evidence)
            except Exception as exc:
                preservation_errors.append("database:" + type(exc).__name__)
            try:
                _capture_private_ledger(harness, report, evidence)
            except Exception as exc:
                preservation_errors.append("ledger:" + type(exc).__name__)
            try:
                object_path = _preserve_object_store(harness, report)
                evidence.add("private_object_snapshot", path=str(object_path))
            except Exception as exc:
                preservation_errors.append("objects:" + type(exc).__name__)
            if preservation_errors:
                evidence.add("private_snapshot_failed", failures=preservation_errors)
            harness._paid_transport.close()
            shutil.rmtree(harness.objects_root, ignore_errors=True)
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        database.close()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if harness is not None and preservation_errors and sys.exc_info()[0] is None:
            raise RuntimeError("private E2E evidence snapshot failed: " + ", ".join(preservation_errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--icps", type=Path, required=True)
    parser.add_argument("--deps", type=Path, required=True)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash-1017")
    args = parser.parse_args()
    result = preflight(source_archive=args.source_archive,
                       baseline_archive=args.baseline_archive,
                       icp_file=args.icps, dependency_dir=args.deps,
                       baseline_url=args.baseline_url, model=args.model)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

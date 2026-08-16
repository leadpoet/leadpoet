#!/usr/bin/env python3
"""Run the full rebenchmark and non-forwarding weight path on one Nitro host."""

from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import boto3


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gateway.research_lab.maintenance import (  # noqa: E402
    set_autoresearch_maintenance_paused,
    set_scoring_maintenance_paused,
)
from leadpoet_canonical.allocation_handoff_v2 import (  # noqa: E402
    validate_allocation_handoff_v2,
)
from leadpoet_canonical.production_parity import (  # noqa: E402
    ProductionParityError,
    sha256_json,
    validate_snapshot_manifest,
)
from research_lab.validator_integration import (  # noqa: E402
    ResearchLabValidatorFlags,
    build_research_lab_allocation_component,
    fetch_research_lab_attested_allocation_bundle,
    verify_research_lab_allocation_bundle,
)
from scripts.build_production_parity_contract import build_contract  # noqa: E402
from scripts.capture_production_parity_runtime_config import capture  # noqa: E402
from scripts.check_production_parity_rebenchmark import check as check_rebenchmark  # noqa: E402
from scripts.materialize_production_parity_secrets import (  # noqa: E402
    _parse_environment_document,
    create as create_gateway_secret,
    delete as delete_gateway_secret,
)
from scripts.production_parity_snapshot import (  # noqa: E402
    capture_snapshot,
    restore_snapshot,
)
from scripts.run_production_parity_fast import _DockerDatabase  # noqa: E402


SHA_RE = re.compile(r"^[0-9a-f]{40}$")
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
RUN_RE = re.compile(r"^[a-z0-9-]{6,40}$")
PINNED_IMAGE_RE = re.compile(r"^[A-Za-z0-9._/:@-]+@sha256:[0-9a-f]{64}$")
SCHEMA_VERSION = "leadpoet.production_parity_full.v3"
OPENROUTER_RUNTIME_CREDENTIAL_REFS = (
    "RESEARCH_LAB_OPENROUTER_API_KEY",
    "RESEARCH_LAB_V2_OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY",
    "QUALIFICATION_OPENROUTER_API_KEY",
    "OPENROUTER_KEY",
)
OPENROUTER_MANAGEMENT_CREDENTIAL_REFS = (
    "RESEARCH_LAB_OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_MANAGEMENT_KEY",
    "OPENROUTER_API_MANAGEMENT_KEY",
    "OR_MANAGEMENT_KEY",
)
EARLY_BOOT_MARKER = Path(
    "/run/leadpoet-production-parity/early-boot-isolated"
)


class FullParityError(RuntimeError):
    """The full disposable workflow did not reach every required stage."""


def _run(
    command: Sequence[str],
    *,
    timeout: int,
    env: Mapping[str, str] | None = None,
    log_path: Path | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if log_path is None:
        return subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            text=True,
            capture_output=True,
            input=input_text,
            check=False,
            timeout=timeout,
        )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        return subprocess.run(
            list(command),
            cwd=ROOT,
            env=dict(env) if env is not None else None,
            text=True,
            input=input_text,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout,
        )


def _require(result: subprocess.CompletedProcess[str], *, stage: str) -> str:
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-1200:]
        raise FullParityError(f"{stage} failed: {detail}")
    return result.stdout or ""


def _checkout_identity(candidate_sha: str) -> None:
    head = _require(
        _run(["git", "rev-parse", "HEAD"], timeout=20),
        stage="candidate source identity",
    ).strip()
    dirty = _require(
        _run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            timeout=20,
        ),
        stage="candidate source cleanliness",
    ).strip()
    if head != candidate_sha or dirty:
        raise FullParityError("full parity checkout differs from the exact candidate")


def _secret_value(client: Any, secret_id: str, *, field: str) -> str:
    value = client.get_secret_value(SecretId=secret_id).get("SecretString")
    if not isinstance(value, str) or not value:
        raise FullParityError(f"{field} is unavailable")
    return value


def _dsn_from_secret(raw: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError:
        value = raw
    if isinstance(value, Mapping):
        candidates = [
            value.get("dsn"),
            value.get("url"),
            value.get("readonly_dsn"),
        ]
        value = next((item for item in candidates if item), "")
    dsn = str(value or "").strip()
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        raise FullParityError("read-only production DSN secret is invalid")
    return dsn


def _wait_https_origin(origin: str, *, timeout_seconds: int = 300) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "pending"
    while time.monotonic() < deadline:
        try:
            request = Request(origin.rstrip("/") + "/", method="GET")
            with urlopen(request, timeout=10) as response:
                if int(response.status) in {200, 401, 403, 404}:
                    return
        except Exception as exc:  # noqa: BLE001 - bounded readiness probe
            last_error = type(exc).__name__
        time.sleep(5)
    raise FullParityError(f"TLS clone origin did not become ready: {last_error}")


@contextmanager
def _gateway_environment_file(path: Path) -> Iterator[None]:
    previous = os.environ.get("GATEWAY_ENV_FILE")
    os.environ["GATEWAY_ENV_FILE"] = str(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("GATEWAY_ENV_FILE", None)
        else:
            os.environ["GATEWAY_ENV_FILE"] = previous


async def _set_clone_controls() -> dict[str, Any]:
    scoring = await set_scoring_maintenance_paused(
        paused=False,
        reason="production_parity_full_rebenchmark",
        actor_ref="system:production-parity",
        event_doc={"production_parity": True},
    )
    autoresearch = await set_autoresearch_maintenance_paused(
        paused=True,
        reason="production_parity_no_miner_or_candidate_activity",
        actor_ref="system:production-parity",
        event_doc={"production_parity": True},
    )
    return {
        "scoring_event_id": scoring.get("event_id"),
        "autoresearch_event_id": autoresearch.get("event_id"),
        "scoring_paused": False,
        "autoresearch_paused": True,
    }


def _gateway_json(path: str) -> dict[str, Any]:
    with urlopen("http://127.0.0.1:8000" + path, timeout=60) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise FullParityError(f"gateway response is invalid: {path}")
    return value


def _report_document(value: Mapping[str, Any]) -> Mapping[str, Any]:
    report = value.get("report_doc")
    return report if isinstance(report, Mapping) else value


def _rebenchmark_identity(
    value: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and identify one published result using candidate policy."""

    report = _report_document(value)
    expected_counts = {
        "public": int(policy.get("public_total_icps") or 0),
        "private": int(policy.get("private_total_icps") or 0),
        "conditional": int(policy.get("conditional_total_icps") or 0),
    }
    expected_total = sum(expected_counts.values())
    public_weak = int(policy.get("public_weak_total") or 0)
    private_weak = int(policy.get("private_weak_total") or 0)
    if (
        any(count <= 0 for count in expected_counts.values())
        or public_weak < 0
        or public_weak > expected_counts["public"]
        or private_weak < 0
        or private_weak > expected_counts["private"]
    ):
        raise FullParityError("candidate ICP policy is invalid")
    split = report.get("visibility_split")
    if not isinstance(split, Mapping):
        raise FullParityError("published rebenchmark assignment is missing")
    observed_counts = {
        "public": int(split.get("public_count") or 0),
        "private": int(split.get("private_count") or 0),
        "conditional": int(split.get("conditional_count") or 0),
    }
    try:
        aggregate_score = float(
            value.get("aggregate_score")
            if value.get("aggregate_score") is not None
            else report.get("aggregate_score")
        )
    except (TypeError, ValueError) as exc:
        raise FullParityError("published rebenchmark score is invalid") from exc
    public_strength = split.get("public_strength_counts")
    private_strength = split.get("private_strength_counts")
    expected_public_strength = {
        "strong": expected_counts["public"] - public_weak,
        "weak": public_weak,
    }
    expected_private_strength = {
        "strong": expected_counts["private"] - private_weak,
        "weak": private_weak,
    }
    expected_public_strength = {
        key: count
        for key, count in expected_public_strength.items()
        if count > 0
    }
    expected_private_strength = {
        key: count
        for key, count in expected_private_strength.items()
        if count > 0
    }
    report_hash = str(report.get("report_public_hash") or "")
    report_without_hash = {
        key: item for key, item in report.items() if key != "report_public_hash"
    }
    if (
        value.get("current_report_status") != "published"
        or value.get("benchmark_quality") != "passed"
        or report.get("report_type") != "research_lab_public_daily_benchmark"
        or int(report.get("item_count") or 0) != expected_total
        or observed_counts != expected_counts
        or dict(public_strength or {}) != expected_public_strength
        or dict(private_strength or {}) != expected_private_strength
        or str(split.get("split_policy") or "")
        != str(policy.get("selection_policy") or "")
        or str(split.get("rolling_window_hash") or "")
        != str(report.get("rolling_window_hash") or "")
        or not 0.0 <= aggregate_score <= 100.0
        or not HASH_RE.fullmatch(report_hash)
        or report_hash != sha256_json(report_without_hash)
    ):
        raise FullParityError(
            "published rebenchmark score or assignment differs from candidate policy"
        )
    identity = {
        "report_id": str(value.get("report_id") or ""),
        "benchmark_bundle_id": str(value.get("benchmark_bundle_id") or ""),
        "benchmark_date": str(value.get("benchmark_date") or ""),
        "rolling_window_hash": str(value.get("rolling_window_hash") or ""),
        "private_model_artifact_hash": str(
            value.get("private_model_artifact_hash") or ""
        ),
        "private_model_manifest_hash": str(
            value.get("private_model_manifest_hash") or ""
        ),
        "aggregate_score": aggregate_score,
        "item_count": expected_total,
        "report_public_hash": report_hash,
        "category_counts": observed_counts,
        "public_strength_counts": dict(public_strength or {}),
        "private_strength_counts": dict(private_strength or {}),
    }
    if (
        not identity["report_id"]
        or not identity["benchmark_bundle_id"]
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", identity["benchmark_date"])
        or not HASH_RE.fullmatch(identity["rolling_window_hash"])
        or not HASH_RE.fullmatch(identity["private_model_artifact_hash"])
        or not HASH_RE.fullmatch(identity["private_model_manifest_hash"])
    ):
        raise FullParityError("published rebenchmark identity is incomplete")
    return identity


def _contains_dashboard_identity(
    value: Any,
    identity: Mapping[str, Any],
) -> bool:
    """Match the public subnet-dashboard projection to a durable result."""

    if not isinstance(value, Mapping) or value.get("success") is not True:
        return False
    data = value.get("data")
    benchmark = data.get("benchmark") if isinstance(data, Mapping) else None
    if not isinstance(benchmark, Mapping):
        return False
    try:
        score_matches = float(benchmark.get("aggregateScore")) == float(
            identity["aggregate_score"]
        )
        count_matches = int(benchmark.get("itemCount")) == int(
            identity["item_count"]
        )
    except (TypeError, ValueError):
        return False
    return (
        str(benchmark.get("reportId") or "") == identity["report_id"]
        and str(benchmark.get("benchmarkDate") or "")
        == identity["benchmark_date"]
        and str(benchmark.get("rollingWindowHash") or "")
        == identity["rolling_window_hash"]
        and score_matches
        and count_matches
    )


def _parse_gateway_environment_file(path: Path) -> dict[str, str]:
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FullParityError("gateway environment file is unavailable") from exc
    values: dict[str, str] = {}
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise FullParityError("gateway environment file has an invalid row")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise FullParityError("gateway environment file has an invalid key")
        if raw_value == "":
            value = ""
        else:
            try:
                tokens = shlex.split(raw_value, posix=True)
            except ValueError as exc:
                raise FullParityError(
                    "gateway environment file has invalid quoting"
                ) from exc
            if len(tokens) != 1:
                raise FullParityError(
                    "gateway environment file has a multi-token value"
                )
            value = tokens[0]
        values[key] = value
    if not values:
        raise FullParityError("gateway environment file is empty")
    return values


def _required_secret_from_environment(
    values: Mapping[str, str],
    references: Sequence[str],
    *,
    field: str,
) -> str:
    for reference in references:
        value = str(values.get(reference) or "").strip()
        if value:
            return value
    raise FullParityError(f"{field} is unavailable in the authorized environment")


def _builtwith_key_from_secret(raw: str) -> str:
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise FullParityError("miner-intake secret is invalid") from exc
    key = (
        str(value.get("builtwith_api_key") or "").strip()
        if isinstance(value, Mapping)
        else ""
    )
    if (
        not 8 <= len(key) <= 512
        or any(character.isspace() for character in key)
        or "\x00" in key
    ):
        raise FullParityError("miner-intake secret is invalid")
    return key


def _last_json_document(output: str, *, field: str) -> dict[str, Any]:
    for raw_line in reversed(output.splitlines()):
        try:
            value = json.loads(raw_line)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    raise FullParityError(f"{field} did not return redacted JSON evidence")


def _run_miner_intake_path(
    *,
    candidate_sha: str,
    run_id: str,
    gateway_env_file: Path,
    production_gateway_environment: Mapping[str, str],
    miner_intake_secret: str,
) -> dict[str, Any]:
    runtime_credential = _required_secret_from_environment(
        production_gateway_environment,
        OPENROUTER_RUNTIME_CREDENTIAL_REFS,
        field="production OpenRouter runtime credential",
    )
    management_credential = _required_secret_from_environment(
        production_gateway_environment,
        OPENROUTER_MANAGEMENT_CREDENTIAL_REFS,
        field="production OpenRouter management credential",
    )
    builtwith_credential = _builtwith_key_from_secret(miner_intake_secret)
    parsed_env = _parse_gateway_environment_file(gateway_env_file)
    child_env = {
        **os.environ,
        **parsed_env,
        "GATEWAY_ENV_FILE": str(gateway_env_file),
        "RESEARCH_LAB_GATEWAY_API_ENABLED": "true",
        "RESEARCH_LAB_PRODUCTION_WRITES_ENABLED": "true",
        "RESEARCH_LAB_MINER_SUBMISSIONS_ENABLED": "true",
        "RESEARCH_LAB_SOURCE_ADD_ENABLED": "true",
        "RESEARCH_LAB_SOURCE_ADD_DISPATCHER_ENABLED": "false",
        "RESEARCH_LAB_PAID_LOOPS_ENABLED": "false",
    }
    request = {
        "schema_version": "leadpoet.production_parity_miner_intake_request.v1",
        "candidate_sha": candidate_sha,
        "run_id": run_id,
        "openrouter_runtime_credential": runtime_credential,
        "openrouter_management_credential": management_credential,
        "builtwith_credential": builtwith_credential,
    }
    result = _run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--miner-intake-child",
        ],
        timeout=360,
        env=child_env,
        input_text=json.dumps(request, separators=(",", ":")),
    )
    # Drop all parent references before interpreting child output. The child
    # emits only bounded booleans/counts; credentials never enter evidence.
    request.clear()
    runtime_credential = management_credential = builtwith_credential = ""
    if result.returncode != 0:
        raise FullParityError("exact miner-intake workflow failed closed")
    evidence = _last_json_document(
        result.stdout or "", field="miner-intake workflow"
    )
    if (
        evidence.get("schema_version")
        != "leadpoet.production_parity_miner_intake_evidence.v1"
        or evidence.get("candidate_sha") != candidate_sha
        or evidence.get("run_id") != run_id
        or evidence.get("status") != "passed"
        or evidence.get("production_database_mutated") is not False
        or evidence.get("production_chain_mutated") is not False
        or evidence.get("chain_registration_boundary")
        != "strict-ephemeral-hotkey"
        or evidence.get("openrouter", {}).get("admitted") is not True
        or evidence.get("source_add", {}).get("admitted") is not True
    ):
        raise FullParityError("miner-intake evidence is incomplete")
    return evidence


def _verify_builtwith_credential_live(credential: str) -> dict[str, Any]:
    url = (
        "https://api.builtwith.com/v23/api.json?LOOKUP=builtwith.com"
        "&HIDETEXT=yes&NOMETA=yes&NOPII=yes&NOATTR=yes"
    )
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"API {credential}",
            "User-Agent": "Leadpoet-Production-Parity/1.0",
        },
    )
    try:
        with urlopen(request, timeout=45) as response:
            payload = response.read(4 * 1024 * 1024 + 1)
            status = int(response.status)
    except Exception as exc:
        raise FullParityError(
            "BuiltWith live credential verification failed"
        ) from exc
    finally:
        url = ""
    if status != 200 or not payload or len(payload) > 4 * 1024 * 1024:
        raise FullParityError("BuiltWith live credential verification failed")
    try:
        document = json.loads(payload)
    except ValueError as exc:
        raise FullParityError(
            "BuiltWith live credential verification returned invalid JSON"
        ) from exc
    if not isinstance(document, (Mapping, list)):
        raise FullParityError(
            "BuiltWith live credential verification returned invalid JSON"
        )
    if isinstance(document, Mapping):
        errors = document.get("Errors") or document.get("errors")
        if errors:
            raise FullParityError("BuiltWith live credential verification failed")
    return {
        "http_status": status,
        "json_verified": True,
        "response_bytes": len(payload),
    }


async def _run_miner_intake_child(request: Mapping[str, Any]) -> dict[str, Any]:
    from types import SimpleNamespace

    import httpx
    from bittensor_wallet import Keypair

    from gateway.main import app
    from gateway.research_lab import api as research_lab_api
    from gateway.research_lab.maintenance import (
        get_autoresearch_maintenance_state,
        get_scoring_maintenance_state,
    )
    from gateway.research_lab.source_add_workflow import source_add_control_state
    from gateway.research_lab.store import call_rpc, select_many, select_one
    from leadpoet_canonical.credential_recipient_v2 import (
        verify_and_encrypt_openrouter_credential_v2,
        verify_openrouter_credential_release_v2,
    )
    from neurons.miner import (
        _research_lab_openrouter_key_signed_payload,
        _research_lab_signed_payload,
        _research_lab_source_add_signed_payload,
    )
    from research_lab.source_add_miner import build_source_add_submission_docs

    if (
        request.get("schema_version")
        != "leadpoet.production_parity_miner_intake_request.v1"
        or not SHA_RE.fullmatch(str(request.get("candidate_sha") or ""))
        or not RUN_RE.fullmatch(str(request.get("run_id") or ""))
        or os.getenv("LEADPOET_PARITY_CANDIDATE_SHA")
        != request.get("candidate_sha")
    ):
        raise FullParityError("miner-intake child identity differs")
    candidate_sha = str(request["candidate_sha"])
    run_id = str(request["run_id"])
    runtime_credential = str(request.get("openrouter_runtime_credential") or "").strip()
    management_credential = str(
        request.get("openrouter_management_credential") or ""
    ).strip()
    builtwith_credential = str(request.get("builtwith_credential") or "").strip()
    if not runtime_credential or not management_credential or not builtwith_credential:
        raise FullParityError("miner-intake credentials are incomplete")

    keypair = Keypair.create_from_mnemonic(Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    miner_hotkey = keypair.ss58_address
    observed_chain_checks: list[str] = []
    chain_check_milestones: list[int] = []
    original_chain_registration = research_lab_api.chain_is_hotkey_registered

    async def strict_chain_registration(hotkey: str):
        observed_chain_checks.append(str(hotkey))
        if str(hotkey) != miner_hotkey:
            return False, None
        return True, "miner"

    research_lab_api.chain_is_hotkey_registered = strict_chain_registration
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    source_controls: dict[str, Any] = {}
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://production-parity.invalid",
            timeout=httpx.Timeout(180.0),
        ) as client:
            timestamp = int(time.time())
            key_fingerprint = hashlib.sha256(
                f"{runtime_credential}:{management_credential}".encode("utf-8")
            ).hexdigest()[:24]
            recipient_payload = _research_lab_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": timestamp,
                    "idempotency_key": (
                        f"research-openrouter-recipient:{miner_hotkey}:"
                        f"{key_fingerprint}"
                    ),
                },
            )
            recipient_response = await client.post(
                "/research-lab/openrouter-keys/credential-recipient",
                json=recipient_payload,
            )
            if recipient_response.status_code != 200:
                raise FullParityError(
                    "OpenRouter credential recipient rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            recipients = recipient_response.json()
            verified_coordinator_boot = verify_openrouter_credential_release_v2(
                recipients["release_evidence"]
            )
            encrypted_runtime = verify_and_encrypt_openrouter_credential_v2(
                recipients["runtime"],
                runtime_credential,
                miner_hotkey=miner_hotkey,
                credential_kind="runtime",
                verified_coordinator_boot_identity=verified_coordinator_boot,
            )
            encrypted_management = verify_and_encrypt_openrouter_credential_v2(
                recipients["management"],
                management_credential,
                miner_hotkey=miner_hotkey,
                credential_kind="management",
                verified_coordinator_boot_identity=verified_coordinator_boot,
            )
            register_payload = _research_lab_openrouter_key_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": (
                        f"research-openrouter-key:{miner_hotkey}:"
                        f"{key_fingerprint}"
                    ),
                    "openrouter_api_key_v2": encrypted_runtime,
                    "openrouter_management_key_v2": encrypted_management,
                    "key_label": "production-parity-miner-intake",
                },
            )
            register_response = await client.post(
                "/research-lab/openrouter-keys", json=register_payload
            )
            if register_response.status_code != 200:
                raise FullParityError(
                    "OpenRouter registration rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            register_result = register_response.json()
            key_ref = str(register_result.get("key_ref") or "")
            key_row = await select_one(
                "research_lab_openrouter_key_refs",
                filters=(("key_ref", key_ref),),
            )
            envelope_rows = await select_many(
                "research_lab_provider_credential_envelopes_v2",
                filters=(("key_ref", key_ref),),
                limit=3,
            )
            openrouter_persistence = json.dumps(
                {
                    "response": register_result,
                    "key_ref": key_row,
                    "envelopes": envelope_rows,
                },
                sort_keys=True,
                default=str,
            )
            if (
                not key_ref
                or not isinstance(key_row, Mapping)
                or key_row.get("preflight_status") != "passed"
                or len(envelope_rows) != 2
                or {str(row.get("credential_kind") or "") for row in envelope_rows}
                != {"runtime", "management"}
                or runtime_credential in openrouter_persistence
                or management_credential in openrouter_persistence
            ):
                raise FullParityError(
                    "OpenRouter registration persistence is incomplete"
                )

            builtwith_probe = _verify_builtwith_credential_live(
                builtwith_credential
            )
            autoresearch_state = await get_autoresearch_maintenance_state()
            scoring_state = await get_scoring_maintenance_state()
            source_state = await source_add_control_state()
            source_controls = {
                "autoresearch_paused": bool(autoresearch_state.get("paused")),
                "scoring_paused": bool(scoring_state.get("paused")),
                "source_add_paused": bool(source_state.get("paused", True)),
            }
            if source_controls["autoresearch_paused"]:
                await set_autoresearch_maintenance_paused(
                    paused=False,
                    reason="production_parity_miner_intake",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
            if source_controls["scoring_paused"]:
                await set_scoring_maintenance_paused(
                    paused=False,
                    reason="production_parity_miner_intake",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
            if source_controls["source_add_paused"]:
                await call_rpc(
                    "research_lab_source_add_set_paused",
                    {
                        "p_paused": False,
                        "p_reason": "production_parity_miner_intake",
                        "p_actor_ref": "system:production-parity",
                    },
                )
            manifest, source_brief, idempotency_key, source_metadata = (
                build_source_add_submission_docs(
                    miner_hotkey=miner_hotkey,
                    source_name=f"BuiltWith API parity {run_id}",
                    source_kind="tech_stack",
                    api_base_url="https://api.builtwith.com/v23",
                    documentation_url=(
                        "https://github.com/builtwith/builtwith-ai-sdk"
                    ),
                    auth_type="api_key_header",
                    endpoint_examples=(
                        {
                            "method": "GET",
                            "path": "/api.json",
                            "purpose": "Retrieve company technology usage",
                            "example_query": (
                                "LOOKUP=builtwith.com&HIDETEXT=yes&NOMETA=yes"
                                "&NOPII=yes&NOATTR=yes"
                            ),
                        },
                    ),
                    rate_limit_notes=(
                        "Provider account limits apply; one bounded read-only "
                        "lookup is used for admission validation."
                    ),
                    data_provenance_notes=(
                        "Technology observations returned by the BuiltWith API."
                    ),
                    third_party_refs=(
                        "https://api.builtwith.com/domain-api",
                    ),
                    credential_supplied=False,
                )
            )
            source_payload = _research_lab_source_add_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": idempotency_key,
                    "manifest": manifest,
                    "source_brief": source_brief,
                    "source_metadata": source_metadata,
                },
            )
            source_response = await client.post(
                "/research-lab/source-adapters", json=source_payload
            )
            if source_response.status_code != 200:
                raise FullParityError(
                    "SOURCE_ADD admission rejected the exact miner request"
                )
            chain_check_milestones.append(len(observed_chain_checks))
            source_result = source_response.json()
            submission_id = str(source_result.get("submission_id") or "")
            source_row = await select_one(
                "research_lab_source_add_submission_current",
                filters=(("submission_id", submission_id),),
            )
            work_rows = await select_many(
                "research_lab_source_add_work_items",
                filters=(("submission_id", submission_id),),
                limit=5,
            )
            source_persistence = json.dumps(
                {
                    "response": source_result,
                    "submission": source_row,
                    "work": work_rows,
                },
                sort_keys=True,
                default=str,
            )
            current_doc = (
                source_row.get("submission_doc")
                if isinstance(source_row, Mapping)
                and isinstance(source_row.get("submission_doc"), Mapping)
                else {}
            )
            if (
                not submission_id
                or source_result.get("stage") != "provenance_queued"
                or not isinstance(source_row, Mapping)
                or source_row.get("stage") != "provenance_queued"
                or len(work_rows) != 1
                or work_rows[0].get("work_kind") != "provenance"
                or work_rows[0].get("work_status") != "queued"
                or int(work_rows[0].get("attempt_count") or 0) != 0
                or current_doc.get("credential_envelope") not in ({}, None)
                or builtwith_credential in source_persistence
                or runtime_credential in source_persistence
                or management_credential in source_persistence
            ):
                raise FullParityError("SOURCE_ADD admission persistence is incomplete")

            retired_payload = _research_lab_signed_payload(
                wallet,
                {
                    "miner_hotkey": miner_hotkey,
                    "timestamp": int(time.time()),
                    "idempotency_key": f"source-add-recipient:{run_id}",
                    "adapter_id": str(source_result.get("adapter_id") or ""),
                },
            )
            retired_response = await client.post(
                "/research-lab/source-adapters/credential-recipient",
                json=retired_payload,
            )
            chain_check_milestones.append(len(observed_chain_checks))
            forbidden_payload = {
                **source_payload,
                "adapter_credential": "production-parity-forbidden-value",
            }
            forbidden_response = await client.post(
                "/research-lab/source-adapters", json=forbidden_payload
            )
            if (
                retired_response.status_code != 410
                or forbidden_response.status_code != 422
            ):
                raise FullParityError(
                    "SOURCE_ADD public credential boundary did not fail closed"
                )

        if (
            any(hotkey != miner_hotkey for hotkey in observed_chain_checks)
            or len(chain_check_milestones) != 4
            or any(
                current <= previous
                for previous, current in zip(
                    (0, *chain_check_milestones[:-1]),
                    chain_check_milestones,
                )
            )
        ):
            raise FullParityError(
                "miner intake did not traverse the strict registration boundary"
            )
        return {
            "schema_version": (
                "leadpoet.production_parity_miner_intake_evidence.v1"
            ),
            "status": "passed",
            "candidate_sha": candidate_sha,
            "run_id": run_id,
            "chain_registration_boundary": "strict-ephemeral-hotkey",
            "production_database_mutated": False,
            "production_chain_mutated": False,
            "provider_security_write": "exact-idempotent-logging-disable",
            "openrouter": {
                "real_production_credentials": True,
                "recipient_attestation_verified": True,
                "miner_signature_verified": True,
                "measured_provider_preflight_passed": True,
                "admitted": True,
                "key_ref_persisted": True,
                "credential_envelope_count": 2,
                "plaintext_absent": True,
            },
            "source_add": {
                "provider": "builtwith",
                "live_provider_credential_verified": (
                    builtwith_probe["http_status"] == 200
                    and builtwith_probe["json_verified"] is True
                ),
                "miner_signature_verified": True,
                "admitted": True,
                "stage": "provenance_queued",
                "queued_work_count": 1,
                "downstream_executed": False,
                "credential_transport": "operator-managed-production-contract",
                "public_credentials_forbidden": True,
                "plaintext_absent": True,
            },
        }
    finally:
        research_lab_api.chain_is_hotkey_registered = original_chain_registration
        try:
            if source_controls.get("source_add_paused"):
                await call_rpc(
                    "research_lab_source_add_set_paused",
                    {
                        "p_paused": True,
                        "p_reason": "production_parity_miner_intake_complete",
                        "p_actor_ref": "system:production-parity",
                    },
                )
            if source_controls.get("autoresearch_paused"):
                await set_autoresearch_maintenance_paused(
                    paused=True,
                    reason="production_parity_miner_intake_complete",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
            if source_controls.get("scoring_paused"):
                await set_scoring_maintenance_paused(
                    paused=True,
                    reason="production_parity_miner_intake_complete",
                    actor_ref="system:production-parity",
                    event_doc={"production_parity": True},
                )
        finally:
            request = {}
            runtime_credential = management_credential = builtwith_credential = ""


def _wait_rebenchmark(
    *, candidate_sha: str, secret_id: str, timeout_seconds: int
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        last = asyncio.run(
            check_rebenchmark(
                root=ROOT,
                candidate_sha=candidate_sha,
                secret_id=secret_id,
            )
        )
        if last.get("available") is True:
            return last
        time.sleep(30)
    raise FullParityError(
        "full rebenchmark did not publish before timeout: "
        + str(last.get("reason") or "unknown")
    )


def _current_epoch_from_readiness(output: str) -> int:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, Mapping):
            for key in ("epoch_id", "effective_epoch", "epoch"):
                try:
                    epoch = int(value.get(key))
                except (TypeError, ValueError):
                    continue
                if epoch > 0:
                    return epoch
    raise FullParityError("weight readiness did not report its effective epoch")


def _validate_real_handoff(
    *, epoch: int, candidate_sha: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    handoff = fetch_research_lab_attested_allocation_bundle(
        "http://127.0.0.1:8000",
        epoch,
        timeout_seconds=360,
    )
    normalized = validate_allocation_handoff_v2(
        handoff,
        expected_epoch_id=epoch,
        expected_netuid=71,
    )
    flags = ResearchLabValidatorFlags.from_mapping(os.environ)
    verification = verify_research_lab_allocation_bundle(
        normalized["bundle"], flags=flags
    )
    if verification.get("passed") is not True:
        raise FullParityError("production validator rejected the real allocation handoff")
    component = build_research_lab_allocation_component(
        normalized["bundle"], flags=flags
    )
    public_evidence = {
        "epoch": epoch,
        "root_receipt_hash": normalized["root_receipt_hash"],
        "allocation_hash": component["allocation_hash"],
        "handoff_hash": sha256_json(normalized),
        "serialized_bytes": len(
            json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "validator_verification_passed": True,
    }
    rehearsal_input = {
        "schema_version": "leadpoet.rehearsal_production_allocation.v1",
        "candidate_sha": candidate_sha,
        "source_epoch": epoch,
        "root_receipt_hash": normalized["root_receipt_hash"],
        "handoff_hash": public_evidence["handoff_hash"],
        "allocation_hash": component["allocation_hash"],
        "allocation_doc": component["allocation_doc"],
    }
    return public_evidence, rehearsal_input


def _run_nonforwarding_weight_path(
    *,
    base_sha: str,
    candidate_sha: str,
    production_allocation: Path,
) -> dict[str, Any]:
    evidence = Path("/tmp") / f"leadpoet-restart-rehearsal-{candidate_sha}-prepush.json"
    evidence.unlink(missing_ok=True)
    result = _run(
        [
            sys.executable,
            "scripts/run_local_restart_rehearsal.py",
            "--from-sha",
            base_sha,
            "--candidate-sha",
            candidate_sha,
            "--transition",
            "forward",
            "--profile",
            "prepush",
            "--production-allocation",
            str(production_allocation),
        ],
        timeout=600,
    )
    _require(result, stage="primary/audit non-forwarding submission path")
    try:
        value = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FullParityError("weight-path evidence is unreadable") from exc
    if (
        value.get("status") != "passed"
        or value.get("release_sha") != candidate_sha
        or value.get("from_sha") != base_sha
    ):
        raise FullParityError("weight-path evidence identity differs")
    canonical = value.get("canonical_vector")
    auditor = value.get("auditor")
    if not isinstance(canonical, Mapping) or not isinstance(auditor, Mapping):
        raise FullParityError("primary/audit weight evidence is incomplete")
    allocation_input = json.loads(
        production_allocation.read_text(encoding="utf-8")
    )
    production_allocation_hash = sha256_json(
        allocation_input["allocation_doc"]
    )
    allocation_evidence = value.get("production_allocation")
    if (
        not isinstance(allocation_evidence, Mapping)
        or allocation_evidence.get("allocation_hash")
        != allocation_input["allocation_hash"]
        or allocation_evidence.get("handoff_hash")
        != allocation_input["handoff_hash"]
        or int(allocation_evidence.get("source_epoch") or -1)
        != int(allocation_input["source_epoch"])
    ):
        raise FullParityError(
            "primary/audit workflow did not consume the clone allocation"
        )
    return {
        "bundle_hash": value.get("bundle_hash"),
        "canonical_vector_hash": sha256_json(dict(canonical)),
        "primary_audit_equal": True,
        "sdk_signed": bool(value.get("signed_extrinsic")),
        "finalization_verified": bool(value.get("finalization")),
        "readback_verified": bool(value.get("reveal")),
        "chain_boundary": "strict-non-forwarding",
        "production_allocation_hash": allocation_input["allocation_hash"],
        "production_allocation_document_hash": production_allocation_hash,
        "production_allocation_bound": True,
    }


def run_full(
    *,
    region: str,
    run_id: str,
    base_sha: str,
    candidate_sha: str,
    production_gateway_secret_id: str,
    readonly_dsn_secret_id: str,
    miner_intake_secret_id: str,
    supabase_origin: str,
    artifact_bucket: str,
    postgres_image: str,
    postgrest_image: str,
    output: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    if (
        not RUN_RE.fullmatch(run_id)
        or not SHA_RE.fullmatch(base_sha)
        or not SHA_RE.fullmatch(candidate_sha)
        or base_sha == candidate_sha
        or not PINNED_IMAGE_RE.fullmatch(postgres_image)
        or not PINNED_IMAGE_RE.fullmatch(postgrest_image)
        or not re.fullmatch(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", artifact_bucket)
    ):
        raise FullParityError("full parity inputs are invalid")
    _checkout_identity(candidate_sha)
    try:
        boot_state = EARLY_BOOT_MARKER.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise FullParityError(
            "transient host did not prove early production-service isolation"
        ) from exc
    if boot_state != "isolated":
        raise FullParityError(
            "transient host early production-service isolation differs"
        )
    work = Path("/run/leadpoet-production-parity") / run_id
    work.mkdir(parents=True, mode=0o700, exist_ok=False)
    secrets_client = boto3.client("secretsmanager", region_name=region)
    runtime_config = work / "runtime-config.json"
    contract_path = work / "contract.json"
    archive_path = work / "production.dump"
    manifest_path = work / "snapshot-manifest.json"
    gateway_env_file = work / "gateway.env"
    gateway_log = work / "gateway-restart.log"
    allocation_override = work / "production-allocation.json"
    artifact_policy = work / "v2-config" / "encrypted-artifact-policy.json"
    secret_created = False
    database = _DockerDatabase(
        candidate_sha=candidate_sha,
        postgres_image=postgres_image,
        postgrest_image=postgrest_image,
        postgres_publish="127.0.0.1::5432",
        postgrest_publish="0.0.0.0:3000",
    )
    evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "candidate_sha": candidate_sha,
        "base_sha": base_sha,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    started = time.monotonic()
    try:
        capture(
            client=secrets_client,
            secret_id=production_gateway_secret_id,
            output=runtime_config,
        )
        contract = build_contract(
            root=ROOT,
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            runtime_config=runtime_config,
            require_runtime_config=True,
        )
        contract_path.write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        dsn = _dsn_from_secret(
            _secret_value(
                secrets_client,
                readonly_dsn_secret_id,
                field="read-only production DSN",
            )
        )
        production_host = str(urlparse(dsn).hostname or "").lower()
        manifest = capture_snapshot(
            contract_path=contract_path,
            archive_path=archive_path,
            manifest_path=manifest_path,
            dsn=dsn,
            expected_production_host=production_host,
            ttl_hours=24,
            source_sha=base_sha,
        )
        if manifest["capture_mode"] != "full":
            raise FullParityError("authoritative parity requires a full production clone")
        database.start()
        restore = restore_snapshot(
            root=ROOT,
            contract_path=contract_path,
            manifest_path=manifest_path,
            archive_path=archive_path,
            target_dsn=database.target_dsn,
            production_host=production_host,
        )
        _local_url, service_role_key = database.start_postgrest()
        _wait_https_origin(supabase_origin)
        secret_state = create_gateway_secret(
            client=secrets_client,
            source_secret_id=production_gateway_secret_id,
            run_id=run_id,
            candidate_sha=candidate_sha,
            supabase_origin=supabase_origin,
            artifact_bucket=artifact_bucket,
            benchmark_date=manifest["database"]["target_rebenchmark_date"],
            jwt_secret=database.jwt_secret,
        )
        secret_created = True
        artifact_policy.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        artifact_policy.write_text(
            json.dumps(
                {
                    "schema_version": "leadpoet.encrypted_artifact_policy.v2",
                    "bucket_host": (
                        f"{artifact_bucket}.s3.{region}.amazonaws.com"
                    ),
                    "key_prefix": "/encrypted-artifacts/",
                    "minimum_retention_days": 1,
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        artifact_policy.chmod(0o600)
        env = os.environ.copy()
        env.update(
            {
                "LEADPOET_REPO_ROOT": str(ROOT),
                "GATEWAY_ROOT": str(ROOT / "gateway"),
                "LEADPOET_GATEWAY_ENV_SECRET_ID": secret_state["secret_id"],
                "GATEWAY_ENV_FILE": str(gateway_env_file),
                "GATEWAY_LOG_ROOT": str(work / "gateway"),
                "GATEWAY_LOG_FILE": str(work / "gateway" / "gateway.log"),
                "GATEWAY_RESTART_CONTROLLER_ROOT": str(work / "restart-controller"),
                "GATEWAY_DEPLOYMENT_DIR": str(work / "deployments"),
                "GATEWAY_HOST_RESTART_SCRIPT": str(ROOT / "gw_restart.sh"),
                "GATEWAY_TEE_EIF_ROOT": str(work / "tee"),
                "GATEWAY_V2_CONFIG_DIR": str(work / "v2-config"),
                "GATEWAY_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts"),
                "VALIDATOR_V2_OFFLINE_ARTIFACT_ROOT": str(work / "offline-artifacts" / "validator-runtime"),
                "GATEWAY_RESTART_LOCK_FILE": str(work / "gateway-restart.lock"),
                "LEADPOET_DOCKER_OPERATION_LOCK_FILE": str(work / "docker-operation.lock"),
                "GATEWAY_DEPLOY_COMMIT": candidate_sha,
                "GATEWAY_V2_RELEASE_PREFIX": "attested-v2/releases",
            }
        )
        restart = _run(
            ["bash", str(ROOT / "gw_restart.sh"), "--commit", candidate_sha],
            timeout=min(timeout_seconds, 10800),
            env=env,
            log_path=gateway_log,
        )
        if restart.returncode != 0:
            detail = gateway_log.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise FullParityError(f"exact gateway restart failed: {detail}")
        health = _gateway_json("/health/v2-authority")
        build = _gateway_json("/build-info")
        if (
            health.get("status") != "ready"
            or str(health.get("commit_sha") or "").lower() != candidate_sha
            or str(build.get("git_commit") or "").lower() != candidate_sha
        ):
            raise FullParityError("gateway V2 health is not exact-candidate ready")

        with _gateway_environment_file(gateway_env_file):
            controls = asyncio.run(_set_clone_controls())
            rebenchmark = _wait_rebenchmark(
                candidate_sha=candidate_sha,
                secret_id=secret_state["secret_id"],
                timeout_seconds=max(600, timeout_seconds - int(time.monotonic() - started)),
            )

        parsed_env = _parse_gateway_environment_file(gateway_env_file)
        readiness = _run(
            [
                env.get("GATEWAY_PYTHON_BIN", "/home/ec2-user/venv311/bin/python3"),
                "-m",
                "gateway.tee.verify_weight_submission_ready_v2",
                "--gateway-url",
                "http://127.0.0.1:8000",
                "--http-timeout-seconds",
                "360",
            ],
            timeout=1800,
            env={**env, **parsed_env, "GATEWAY_ENV_FILE": str(gateway_env_file)},
        )
        readiness_output = _require(readiness, stage="real gateway weight readiness")
        epoch = _current_epoch_from_readiness(readiness_output)
        with _gateway_environment_file(gateway_env_file):
            # The integration layer reads its internal key from the same exact
            # gateway environment file used by production.
            previous = {key: os.environ.get(key) for key in parsed_env}
            os.environ.update(parsed_env)
            try:
                handoff, allocation_input = _validate_real_handoff(
                    epoch=epoch,
                    candidate_sha=candidate_sha,
                )
            finally:
                for key, old in previous.items():
                    if old is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = old
        allocation_override.write_text(
            json.dumps(allocation_input, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        allocation_override.chmod(0o600)
        weight_path = _run_nonforwarding_weight_path(
            base_sha=base_sha,
            candidate_sha=candidate_sha,
            production_allocation=allocation_override,
        )
        if (
            handoff["allocation_hash"]
            != weight_path["production_allocation_hash"]
        ):
            raise FullParityError(
                "gateway handoff and primary/audit allocation hashes differ"
            )
        production_gateway_environment = _parse_environment_document(
            _secret_value(
                secrets_client,
                production_gateway_secret_id,
                field="production gateway environment",
            ),
            field="production gateway environment",
        )
        miner_intake = _run_miner_intake_path(
            candidate_sha=candidate_sha,
            run_id=run_id,
            gateway_env_file=gateway_env_file,
            production_gateway_environment=production_gateway_environment,
            miner_intake_secret=_secret_value(
                secrets_client,
                miner_intake_secret_id,
                field="miner-intake credential",
            ),
        )
        shape = database.shape_evidence(
            service_role_key=service_role_key,
            expected_shape=validate_snapshot_manifest(manifest)["database"],
            capture_mode="full",
        )
        scale = database.weight_input_scale_evidence(
            service_role_key=service_role_key
        )
        evidence.update(
            {
                "status": "passed",
                "contract_hash": contract["contract_hash"],
                "snapshot_hash": manifest["manifest_hash"],
                "snapshot_restore": restore,
                "production_shape": shape,
                "production_weight_input_scale": scale,
                "gateway": {
                    "commit_sha": candidate_sha,
                    "pcr0": health.get("pcr0"),
                    "attestation_ready": True,
                },
                "controls": controls,
                "rebenchmark": rebenchmark,
                "allocation_handoff": handoff,
                "weight_path": weight_path,
                "miner_intake": miner_intake,
            }
        )
    finally:
        cleanup: dict[str, Any] = {}
        try:
            cleanup["database"] = database.cleanup()
        except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive
            cleanup["database_error"] = type(exc).__name__
        if secret_created:
            try:
                cleanup["secret"] = delete_gateway_secret(
                    client=secrets_client, run_id=run_id
                )
            except Exception as exc:  # noqa: BLE001
                cleanup["secret_error"] = type(exc).__name__
        for path in (
            archive_path,
            runtime_config,
            gateway_env_file,
            allocation_override,
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                cleanup.setdefault("file_cleanup_errors", []).append(path.name)
        evidence["cleanup"] = cleanup
        evidence["duration_seconds"] = round(time.monotonic() - started, 3)
        evidence["finished_at"] = datetime.now(timezone.utc).isoformat()
        if evidence.get("status") == "passed" and (
            "database_error" in cleanup
            or "secret_error" in cleanup
            or cleanup.get("file_cleanup_errors")
        ):
            evidence["status"] = "failed"
            evidence["failure"] = "run-scoped cleanup was incomplete"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(evidence, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    if evidence.get("status") != "passed":
        raise FullParityError(str(evidence.get("failure") or "full parity failed"))
    return evidence


def main(argv: Sequence[str] | None = None) -> int:
    if list(argv if argv is not None else sys.argv[1:]) == ["--miner-intake-child"]:
        try:
            request = json.load(sys.stdin)
            if not isinstance(request, Mapping):
                raise FullParityError("miner-intake request is invalid")
            result = asyncio.run(_run_miner_intake_child(request))
        except Exception as exc:  # noqa: BLE001 - never print secret-bearing detail
            print(
                f"ERROR: miner-intake child failed closed ({type(exc).__name__})",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--production-gateway-secret-id", required=True)
    parser.add_argument("--readonly-dsn-secret-id", required=True)
    parser.add_argument("--miner-intake-secret-id", required=True)
    parser.add_argument("--supabase-origin", required=True)
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--postgres-image", required=True)
    parser.add_argument("--postgrest-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=43200)
    args = parser.parse_args(argv)
    try:
        result = run_full(
            region=args.region,
            run_id=args.run_id,
            base_sha=args.base_sha.lower(),
            candidate_sha=args.candidate_sha.lower(),
            production_gateway_secret_id=args.production_gateway_secret_id,
            readonly_dsn_secret_id=args.readonly_dsn_secret_id,
            miner_intake_secret_id=args.miner_intake_secret_id,
            supabase_origin=args.supabase_origin,
            artifact_bucket=args.artifact_bucket,
            postgres_image=args.postgres_image,
            postgrest_image=args.postgrest_image,
            output=args.output,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, ProductionParityError, FullParityError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": result["status"],
                "candidate_sha": result["candidate_sha"],
                "duration_seconds": result["duration_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

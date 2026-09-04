"""Operator CLI for retained Research Lab controls."""

from __future__ import annotations

import argparse
import asyncio
import base64
import getpass
import hashlib
import json
import logging
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Mapping
from urllib.parse import urlsplit

from gateway.deploy_readiness import (
    build_deploy_readiness,
    write_deploy_readiness_manifest,
)
from .maintenance import (
    backfill_champion_reward_v2_authority,
    backfill_champion_settlement_v2_authority,
    champion_v2_cutover_readiness_report,
    default_actor_ref,
    dumps_status,
    reconcile_champion_reward_statuses,
)
from .store import call_rpc, select_all, select_one

logger = logging.getLogger(__name__)


def _add_deploy_readiness_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gateway-commit", help="current gateway source commit")
    parser.add_argument("--validator-commit", help="current validator source commit")
    parser.add_argument("--gateway-pcr0", help="current gateway enclave PCR0")
    parser.add_argument("--validator-pcr0", help="current validator enclave PCR0")
    parser.add_argument("--expected-gateway-commit", help="expected gateway source commit")
    parser.add_argument("--expected-validator-commit", help="expected validator source commit")
    parser.add_argument("--expected-gateway-pcr0", help="expected gateway enclave PCR0")
    parser.add_argument("--expected-validator-pcr0", help="expected validator enclave PCR0")
    parser.add_argument(
        "--require-same-commit",
        action="store_true",
        help="fail unless gateway and validator commits match",
    )
    parser.add_argument(
        "--require-pcr0",
        action="store_true",
        help="fail unless both supplied gateway and validator PCR0s are present",
    )
    parser.add_argument(
        "--require-pcr0-commit-match",
        action="store_true",
        help="fail unless matched static PCR0 allowlist metadata points at the running commit",
    )
    parser.add_argument(
        "--include-docker-health",
        action="store_true",
        help="include Docker daemon and disk headroom health as a warning check",
    )
    parser.add_argument(
        "--require-docker-build-health",
        action="store_true",
        help="run a tiny Docker smoke build and fail readiness if Docker/build storage is unhealthy",
    )

def _add_source_add_parser(sub) -> None:
    source_add = sub.add_parser(
        "source-add",
        help="Dry-run-first SOURCE_ADD queue, test, and provisioning controls",
    )
    commands = source_add.add_subparsers(dest="source_add_command", required=True)

    listing = commands.add_parser("list", help="List current SOURCE_ADD submissions")
    listing.add_argument("--stage")
    listing.add_argument("--limit", type=int, default=200)
    commands.add_parser(
        "status", help="Show SOURCE_ADD dispatcher, queue, probe, and reward state"
    )

    for action, paused in (("pause", True), ("resume", False)):
        control = commands.add_parser(
            action, help=("Pause" if paused else "Resume") + " SOURCE_ADD work claims"
        )
        control.add_argument("--reason", required=True)
        control.add_argument("--actor-ref", default=default_actor_ref())
        control.add_argument("--apply", action="store_true")

    recheck = commands.add_parser("recheck", help="Queue a provenance recheck")
    recheck.add_argument("--submission-id", required=True)
    recheck.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    recheck.add_argument("--apply", action="store_true")

    configure = commands.add_parser(
        "configure-test", help="Configure and queue an exact V2 functional API test"
    )
    configure.add_argument("--submission-id", required=True)
    configure.add_argument("--base-url", required=True)
    configure.add_argument(
        "--auth-kind", choices=("none", "header", "query", "bearer"), default="none"
    )
    configure.add_argument("--auth-name")
    configure.add_argument("--header", action="append", default=[])
    configure.add_argument("--probe-json", action="append", required=True)
    configure.add_argument("--operator-notes")
    configure.add_argument("--credential-stdin", action="store_true")
    configure.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    configure.add_argument("--apply", action="store_true")

    provision = commands.add_parser(
        "provision", help="Approve or make a tested source available"
    )
    provision.add_argument("--submission-id", required=True)
    provision.add_argument("--registry-provider-id", required=True)
    provision.add_argument("--provider-alias")
    provision.add_argument(
        "--status",
        dest="provision_status",
        choices=("approved_pending_provision", "provisioned_autoresearch_eligible"),
        default="approved_pending_provision",
        help="Persisted status; the legacy eligible value remains for DB compatibility",
    )
    provision.add_argument("--probe-endpoint-json", action="append", required=True)
    provision.add_argument("--cost-model-json", default="{}")
    provision.add_argument("--routing-contract-json", default="{}")
    provision.add_argument("--operator-notes")
    provision.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    provision.add_argument("--apply", action="store_true")

    disable = commands.add_parser(
        "disable", help="Disable a provisioned SOURCE_ADD provider"
    )
    disable.add_argument("--submission-id", required=True)
    disable.add_argument("--registry-provider-id", required=True)
    disable.add_argument("--operator-notes")
    disable.add_argument("--gateway-url", default="http://127.0.0.1:8000")
    disable.add_argument("--apply", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leadpoet Research Lab admin controls")
    sub = parser.add_subparsers(dest="command", required=True)

    for command, action in (
        ("pause-source-add", "pause"),
        ("resume-source-add", "resume"),
    ):
        control = sub.add_parser(command)
        control.add_argument(
            "--reason",
            required=action == "pause",
            default="maintenance complete" if action == "resume" else None,
        )
        control.add_argument("--actor-ref", default=default_actor_ref())
        control.set_defaults(source_add_command=action, apply=True)

    readiness = sub.add_parser(
        "check-deploy-readiness", help="Check gateway and validator release alignment"
    )
    _add_deploy_readiness_args(readiness)
    readiness.add_argument("--write-manifest", nargs="?", const="")
    readiness.add_argument("--no-enforce-resume-block", action="store_true")

    sub.add_parser("status", help="Print SOURCE_ADD maintenance state")

    champion = sub.add_parser("reconcile-champion-reward-statuses")
    champion.add_argument("--epoch", type=int)
    champion.add_argument("--netuid", type=int)
    champion.add_argument("--limit", type=int, default=50)
    champion.add_argument("--reason", default="champion_reward_status_reconciler")
    champion.add_argument("--actor-ref", default=default_actor_ref())
    champion.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    champion.add_argument("--write", dest="dry_run", action="store_false")

    source_reward = sub.add_parser("reconcile-source-add-reward-statuses")
    source_reward.add_argument("--epoch", type=int)
    source_reward.add_argument("--netuid", type=int)
    source_reward.add_argument("--limit", type=int, default=50)
    source_reward.add_argument("--reason", default="source_add_reward_fully_delivered")
    source_reward.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True
    )
    source_reward.add_argument("--write", dest="dry_run", action="store_false")

    champion_auth = sub.add_parser("backfill-champion-v2-authority")
    champion_auth.add_argument("--epoch", type=int)
    champion_auth.add_argument("--limit", type=int, default=1000)
    champion_auth.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True
    )
    champion_auth.add_argument("--write", dest="dry_run", action="store_false")

    settlement = sub.add_parser("backfill-champion-v2-settlements")
    settlement.add_argument("--epoch", type=int)
    settlement.add_argument("--netuid", type=int)
    settlement.add_argument("--limit", type=int, default=1000)
    settlement.add_argument(
        "--dry-run", dest="dry_run", action="store_true", default=True
    )
    settlement.add_argument("--write", dest="dry_run", action="store_false")

    cutover = sub.add_parser(
        "champion-v2-cutover-readiness",
        help="Require complete V2 receipt coverage for historical positive champion balances",
    )
    cutover.add_argument("--epoch", type=int)
    cutover.add_argument("--netuid", type=int)

    recover = sub.add_parser("recover-arweave-audit-epochs")
    recover.add_argument("--epoch", action="append", dest="epochs", type=int, required=True)
    recover.add_argument("--netuid", type=int)
    recover.add_argument("--dry-run", dest="dry_run", action="store_true", default=True)
    recover.add_argument("--write", dest="dry_run", action="store_false")

    checkpoint = sub.add_parser("checkpoint-arweave-now")
    checkpoint.add_argument("--write", action="store_true")

    _add_source_add_parser(sub)
    return parser


def _source_add_json_object(value: str, *, field: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field} must be a JSON object")
    return parsed


def _source_add_headers(values: list[str]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for raw in values:
        name, separator, value = str(raw).partition("=")
        name = name.strip()
        if not separator or not name or name in headers:
            raise ValueError("each --header must be a unique NAME=VALUE")
        headers[name] = value
    return headers


def _source_add_gateway_url(value: str) -> str:
    normalized = str(value or "").strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("SOURCE_ADD gateway URL is invalid")
    if parsed.scheme == "http" and parsed.hostname not in {
        "127.0.0.1",
        "::1",
        "localhost",
    }:
        raise ValueError("plaintext SOURCE_ADD admin auth is loopback-only")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("SOURCE_ADD gateway URL must not contain credentials or query data")
    return normalized


async def _source_add_admin_http(
    *, gateway_url: str, path: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    service_key = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not service_key:
        raise ValueError("SUPABASE_SERVICE_ROLE_KEY is required")
    url = _source_add_gateway_url(gateway_url) + str(path)
    body = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )

    def _call() -> dict[str, Any]:
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": "Bearer " + service_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                encoded = response.read(1024 * 1024)
        except urllib.error.HTTPError as exc:
            encoded = exc.read(64 * 1024)
            try:
                detail = json.loads(encoded.decode("utf-8")).get("detail")
            except Exception:
                detail = "request rejected"
            raise RuntimeError(
                f"SOURCE_ADD gateway returned HTTP {exc.code}: {detail}"
            ) from exc
        try:
            decoded = json.loads(encoded.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("SOURCE_ADD gateway returned invalid JSON") from exc
        if not isinstance(decoded, dict):
            raise RuntimeError("SOURCE_ADD gateway returned an invalid document")
        return decoded

    return await asyncio.to_thread(_call)


def _verify_and_encrypt_source_add_credential(
    recipient: Mapping[str, Any], credential: bytes
) -> dict[str, str]:
    """Verify the exact Nitro recipient claim before encrypting a secret locally."""

    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    from gateway.tee.kms_recipient_v2 import (
        KMS_KEY_ENCRYPTION_ALGORITHM,
        SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
        SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
    )
    from leadpoet_canonical.attested_v2 import canonical_json, sha256_json
    from leadpoet_canonical.nitro import verify_nitro_attestation_full

    required = {
        "schema_version",
        "purpose",
        "request_id",
        "boot_identity_hash",
        "miner_hotkey_hash",
        "adapter_ref_hash",
        "credential_ref",
        "key_ref_hash",
        "recipient_public_key_hash",
        "request_nonce",
        "recipient_public_key_der_b64",
        "attestation_document_b64",
        "key_encryption_algorithm",
    }
    if not isinstance(recipient, Mapping) or set(recipient) != required:
        raise ValueError("SOURCE_ADD recipient fields are invalid")
    if (
        recipient.get("schema_version")
        != SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION
        or recipient.get("purpose") != SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE
        or recipient.get("key_encryption_algorithm")
        != KMS_KEY_ENCRYPTION_ALGORITHM
    ):
        raise ValueError("SOURCE_ADD recipient policy is invalid")
    if not credential or len(credential) > 64 * 1024 or b"\x00" in credential:
        raise ValueError("SOURCE_ADD credential is empty or outside the allowed size")
    try:
        public_der = base64.b64decode(
            str(recipient["recipient_public_key_der_b64"]), validate=True
        )
    except Exception as exc:
        raise ValueError("SOURCE_ADD recipient public key is invalid") from exc
    public_hash = "sha256:" + hashlib.sha256(public_der).hexdigest()
    if public_hash != str(recipient.get("recipient_public_key_hash") or ""):
        raise ValueError("SOURCE_ADD recipient public key hash differs")
    claim = {
        name: recipient[name]
        for name in (
            "schema_version",
            "purpose",
            "boot_identity_hash",
            "miner_hotkey_hash",
            "adapter_ref_hash",
            "credential_ref",
            "key_ref_hash",
            "recipient_public_key_hash",
            "request_nonce",
        )
    }
    request_id = sha256_json(claim)
    if request_id != str(recipient.get("request_id") or ""):
        raise ValueError("SOURCE_ADD recipient claim hash differs")
    valid, attestation = verify_nitro_attestation_full(
        attestation_b64=str(recipient["attestation_document_b64"]),
        expected_pubkey=None,
        expected_purpose=SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
        role="gateway",
    )
    if not valid:
        raise ValueError("SOURCE_ADD Nitro attestation verification failed")
    expected_user_data = {
        "schema_version": SOURCE_ADD_INGRESS_RECIPIENT_SCHEMA_VERSION,
        "purpose": SOURCE_ADD_INGRESS_RECIPIENT_PURPOSE,
        "claim_hash": request_id,
    }
    if (
        attestation.get("attestation_public_key") != public_der.hex()
        or attestation.get("user_data") != expected_user_data
        or canonical_json(attestation["user_data"]) != canonical_json(expected_user_data)
    ):
        raise ValueError("SOURCE_ADD attestation is not bound to the recipient claim")
    try:
        public_key = serialization.load_der_public_key(public_der)
    except Exception as exc:
        raise ValueError("SOURCE_ADD recipient RSA key is invalid") from exc
    if not isinstance(public_key, rsa.RSAPublicKey) or public_key.key_size < 2048:
        raise ValueError("SOURCE_ADD recipient key policy is invalid")
    max_plaintext = (public_key.key_size // 8) - (2 * hashes.SHA256.digest_size) - 2
    if len(credential) > max_plaintext:
        raise ValueError("SOURCE_ADD credential exceeds the attested RSA-OAEP limit")
    ciphertext = public_key.encrypt(
        credential,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return {
        "request_id": request_id,
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


def _read_source_add_credential(*, from_stdin: bool) -> bytes:
    if from_stdin:
        value = sys.stdin.buffer.readline(64 * 1024 + 2).rstrip(b"\r\n")
    else:
        value = getpass.getpass("SOURCE_ADD API credential: ").encode("utf-8")
    if not value or len(value) > 64 * 1024:
        raise ValueError("SOURCE_ADD credential is empty or outside the allowed size")
    return value


async def _source_add_status() -> dict[str, Any]:
    control = await select_one(
        "research_lab_source_add_control", filters=(("singleton", True),)
    )
    work = await select_all(
        "research_lab_source_add_work_items",
        columns="work_kind,work_status",
        filters=(),
        max_rows=50000,
    )
    intents = await select_all(
        "research_lab_source_add_reward_intents",
        columns="intent_status",
        filters=(),
        max_rows=50000,
    )
    probes = await select_all(
        "research_lab_source_add_functional_probe_current",
        columns="result_status",
        filters=(),
        max_rows=50000,
    )

    def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
        output: dict[str, int] = {}
        for row in rows:
            key = str(row.get(field) or "unknown")
            output[key] = output.get(key, 0) + 1
        return dict(sorted(output.items()))

    work_counts: dict[str, dict[str, int]] = {}
    for row in work:
        kind = str(row.get("work_kind") or "unknown")
        status = str(row.get("work_status") or "unknown")
        work_counts.setdefault(kind, {})[status] = (
            work_counts.setdefault(kind, {}).get(status, 0) + 1
        )
    return {
        "ok": True,
        "action": "source-add status",
        "control": control or {},
        "work_counts": work_counts,
        "functional_probe_counts": _counts(probes, "result_status"),
        "reward_intent_counts": _counts(intents, "intent_status"),
    }


async def _run_source_add_admin(args: argparse.Namespace) -> dict[str, Any]:
    action = str(args.source_add_command)
    if action == "status":
        return await _source_add_status()
    if action == "list":
        if not 1 <= int(args.limit) <= 10000:
            raise ValueError("--limit must be between 1 and 10000")
        filters = (("stage", args.stage),) if args.stage else ()
        rows = await select_all(
            "research_lab_source_add_submission_current",
            columns=(
                "submission_id,adapter_id,miner_hotkey,stage,precheck_status,"
                "submission_doc,created_at"
            ),
            filters=filters,
            order_by=(("created_at", False),),
            max_rows=int(args.limit),
            allow_partial=True,
        )
        submissions = []
        for row in rows[: int(args.limit)]:
            document = row.get("submission_doc")
            manifest = document.get("manifest") if isinstance(document, Mapping) else {}
            submissions.append(
                {
                    "submission_id": row.get("submission_id"),
                    "adapter_id": row.get("adapter_id"),
                    "miner_hotkey": row.get("miner_hotkey"),
                    "source_name": manifest.get("source_name") if isinstance(manifest, Mapping) else "",
                    "stage": row.get("stage"),
                    "precheck_status": row.get("precheck_status"),
                    "created_at": row.get("created_at"),
                }
            )
        return {
            "ok": True,
            "action": "source-add list",
            "count": len(submissions),
            "submissions": submissions,
        }
    if action in {"pause", "resume"}:
        current = await select_one(
            "research_lab_source_add_control", filters=(("singleton", True),)
        )
        if not args.apply:
            return {
                "ok": True,
                "action": f"source-add {action}",
                "dry_run": True,
                "current": current or {},
                "requested_paused": action == "pause",
                "reason": args.reason,
                "actor_ref": args.actor_ref,
            }
        result = await call_rpc(
            "research_lab_source_add_set_paused",
            {
                "p_paused": action == "pause",
                "p_reason": args.reason,
                "p_actor_ref": args.actor_ref,
            },
        )
        if isinstance(result, list) and len(result) == 1:
            result = result[0]
        if not isinstance(result, Mapping):
            raise RuntimeError("SOURCE_ADD control RPC returned an invalid result")
        return {
            "ok": True,
            "action": f"source-add {action}",
            "dry_run": False,
            "control": dict(result),
        }
    if action == "recheck":
        if not args.apply:
            return {
                "ok": True,
                "action": "source-add recheck",
                "dry_run": True,
                "submission_id": args.submission_id,
            }
        result = await _source_add_admin_http(
            gateway_url=args.gateway_url,
            path=f"/research-lab/admin/source-adapters/{args.submission_id}/recheck-provenance",
            payload={},
        )
        return {"ok": True, "action": "source-add recheck", "dry_run": False, **result}
    if action == "configure-test":
        from .models import ResearchLabSourceAdapterProbeConfigureRequest

        probes = [
            _source_add_json_object(value, field="--probe-json")
            for value in args.probe_json
        ]
        if not 1 <= len(probes) <= 3:
            raise ValueError("configure-test requires one to three probes")
        payload: dict[str, Any] = {
            "base_url": args.base_url,
            "auth_kind": args.auth_kind,
            "auth_name": args.auth_name,
            "request_headers": _source_add_headers(args.header),
            "probes": probes,
            "operator_notes": args.operator_notes,
        }
        if args.auth_kind == "none" and args.credential_stdin:
            raise ValueError("--credential-stdin requires an authenticated --auth-kind")
        validation_payload = dict(payload)
        if args.auth_kind != "none":
            validation_payload["api_credential_v2"] = {
                "request_id": "sha256:" + "0" * 64,
                "ciphertext_b64": base64.b64encode(b"0" * 256).decode("ascii"),
            }
        validated = ResearchLabSourceAdapterProbeConfigureRequest.model_validate(
            validation_payload
        )
        payload = validated.model_dump(
            mode="json",
            exclude={"api_credential", "api_credential_v2"},
            exclude_none=True,
        )
        if not args.apply:
            return {
                "ok": True,
                "action": "source-add configure-test",
                "dry_run": True,
                "submission_id": args.submission_id,
                "credential_required": args.auth_kind != "none",
                "configuration": payload,
            }
        if args.auth_kind != "none":
            recipient = await _source_add_admin_http(
                gateway_url=args.gateway_url,
                path=f"/research-lab/admin/source-adapters/{args.submission_id}/credential-recipient",
                payload={},
            )
            credential = _read_source_add_credential(
                from_stdin=bool(args.credential_stdin)
            )
            payload["api_credential_v2"] = _verify_and_encrypt_source_add_credential(
                recipient, credential
            )
        result = await _source_add_admin_http(
            gateway_url=args.gateway_url,
            path=f"/research-lab/admin/source-adapters/{args.submission_id}/configure-test",
            payload=payload,
        )
        return {
            "ok": True,
            "action": "source-add configure-test",
            "dry_run": False,
            **result,
        }
    if action == "provision":
        from .models import ResearchLabSourceAdapterProvisionRequest

        endpoints = [
            _source_add_json_object(value, field="--probe-endpoint-json")
            for value in args.probe_endpoint_json
        ]
        payload = ResearchLabSourceAdapterProvisionRequest.model_validate({
            "registry_provider_id": args.registry_provider_id,
            "provider_alias": args.provider_alias,
            "provision_status": args.provision_status,
            "cost_model": _source_add_json_object(
                args.cost_model_json, field="--cost-model-json"
            ),
            "routing_contract": _source_add_json_object(
                args.routing_contract_json,
                field="--routing-contract-json",
            ),
            "probe_endpoints": endpoints,
            "operator_notes": args.operator_notes,
        }).model_dump(mode="json", exclude_none=True, exclude_unset=True)
        if not args.apply:
            return {
                "ok": True,
                "action": "source-add provision",
                "dry_run": True,
                "submission_id": args.submission_id,
                "provisioning": payload,
            }
        result = await _source_add_admin_http(
            gateway_url=args.gateway_url,
            path=f"/research-lab/admin/source-adapters/{args.submission_id}/provision",
            payload=payload,
        )
        return {
            "ok": True,
            "action": "source-add provision",
            "dry_run": False,
            **result,
        }
    if action == "disable":
        from .models import ResearchLabSourceAdapterProvisionRequest

        payload = ResearchLabSourceAdapterProvisionRequest.model_validate({
            "registry_provider_id": args.registry_provider_id,
            "provision_status": "disabled",
            "operator_notes": args.operator_notes,
        }).model_dump(mode="json", exclude_none=True)
        if not args.apply:
            return {
                "ok": True,
                "action": "source-add disable",
                "dry_run": True,
                "submission_id": args.submission_id,
                "provisioning": payload,
            }
        result = await _source_add_admin_http(
            gateway_url=args.gateway_url,
            path=f"/research-lab/admin/source-adapters/{args.submission_id}/provision",
            payload=payload,
        )
        return {
            "ok": True,
            "action": "source-add disable",
            "dry_run": False,
            **result,
        }
    raise ValueError(f"unknown SOURCE_ADD command: {action}")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command in {"source-add", "pause-source-add", "resume-source-add"}:
        return await _run_source_add_admin(args)
    if args.command == "check-deploy-readiness":
        result = build_deploy_readiness(
            gateway_commit=args.gateway_commit,
            validator_commit=args.validator_commit,
            gateway_pcr0=args.gateway_pcr0,
            validator_pcr0=args.validator_pcr0,
            expected_gateway_commit=args.expected_gateway_commit,
            expected_validator_commit=args.expected_validator_commit,
            expected_gateway_pcr0=args.expected_gateway_pcr0,
            expected_validator_pcr0=args.expected_validator_pcr0,
            require_same_commit=args.require_same_commit,
            require_pcr0=args.require_pcr0,
            require_pcr0_commit_match=args.require_pcr0_commit_match,
            include_docker_health=args.include_docker_health,
            require_docker_build_health=args.require_docker_build_health,
        )
        result["action"] = "check-deploy-readiness"
        if args.write_manifest is not None:
            result["manifest_path"] = str(
                write_deploy_readiness_manifest(
                    result,
                    args.write_manifest or None,
                    enforce_resume_block=not args.no_enforce_resume_block,
                )
            )
        return result
    if args.command == "status":
        return {
            "ok": True,
            "source_add": await _source_add_status(),
        }
    if args.command == "reconcile-champion-reward-statuses":
        return await reconcile_champion_reward_statuses(
            epoch=args.epoch,
            netuid=args.netuid,
            limit=args.limit,
            reason=args.reason,
            actor_ref=args.actor_ref,
            dry_run=args.dry_run,
        )
    if args.command == "reconcile-source-add-reward-statuses":
        from .maintenance import reconcile_source_add_reward_statuses
        return await reconcile_source_add_reward_statuses(
            epoch=args.epoch,
            netuid=args.netuid,
            limit=args.limit,
            reason=args.reason,
            dry_run=args.dry_run,
        )
    if args.command == "backfill-champion-v2-authority":
        return await backfill_champion_reward_v2_authority(
            epoch=args.epoch, limit=args.limit, dry_run=args.dry_run
        )
    if args.command == "backfill-champion-v2-settlements":
        return await backfill_champion_settlement_v2_authority(
            epoch=args.epoch,
            netuid=args.netuid,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    if args.command == "champion-v2-cutover-readiness":
        return await champion_v2_cutover_readiness_report(
            epoch=args.epoch, netuid=args.netuid
        )
    if args.command == "recover-arweave-audit-epochs":
        from gateway.config import BITTENSOR_NETUID
        from .arweave_audit import recover_research_lab_checkpointed_audit_epochs
        return await recover_research_lab_checkpointed_audit_epochs(
            epochs=args.epochs,
            netuid=int(args.netuid) if args.netuid is not None else int(BITTENSOR_NETUID),
            dry_run=args.dry_run,
        )
    if args.command == "checkpoint-arweave-now":
        if not args.write:
            return {
                "ok": True,
                "dry_run": True,
                "action": "checkpoint-arweave-now",
                "guidance": "pass --write to run one immediate checkpoint batch",
            }
        from gateway.tasks.hourly_batch import hourly_batch_task
        result = await hourly_batch_task(run_immediately=True, max_batches=1)
        if not isinstance(result, dict):
            raise RuntimeError("immediate Arweave checkpoint returned no result")
        return {**result, "dry_run": False, "action": "checkpoint-arweave-now"}
    raise ValueError(f"unknown command: {args.command}")


def main() -> int:
    args = build_parser().parse_args()
    result = asyncio.run(_run(args))
    print(dumps_status(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

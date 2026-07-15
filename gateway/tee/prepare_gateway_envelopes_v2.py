"""Prepare all KMS-sealed gateway V2 boot and worker-profile envelopes.

This is an operator-only, pre-cutover command. Plaintext is read from one
protected environment file, sent to AWS KMS Encrypt, and never written or
printed. The output directory contains ciphertext envelopes and a non-secret
transition report only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import tempfile
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.research_lab.worker_autostart import (
    HOSTED_PROXY_PREFIXES,
    SCORING_PROXY_PREFIXES,
    build_research_lab_worker_autostart_plan,
)
from gateway.tee.artifact_vault_v2 import artifact_master_key_reference_hash
from gateway.tee.provider_broker_v2 import credential_reference_hash
from gateway.utils.tee_kms_provision_v2 import build_provider_envelope_v2


_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_BOOT_SOURCES = {
    "openrouter": ("RESEARCH_LAB_V2_OPENROUTER_API_KEY",),
    "exa": ("RESEARCH_LAB_V2_EXA_API_KEY",),
    "scrapingdog": ("RESEARCH_LAB_V2_SCRAPINGDOG_API_KEY",),
    "deepline": ("RESEARCH_LAB_V2_DEEPLINE_API_KEY",),
    "supabase_service_role": ("SUPABASE_SERVICE_ROLE_KEY",),
    "truelist": ("TRUELIST_API_KEY",),
}
_SHARED_PARENT_SLOTS = frozenset(("supabase_service_role", "truelist"))
_SPECIAL_PROFILES = {
    "benchmark_exa.json": (
        "exa",
        ("RESEARCH_LAB_V2_BENCHMARK_EXA_API_KEY", *_BOOT_SOURCES["exa"]),
    ),
    "benchmark_openrouter.json": (
        "openrouter",
        (
            "RESEARCH_LAB_V2_BENCHMARK_OPENROUTER_API_KEY",
            *_BOOT_SOURCES["openrouter"],
        ),
    ),
    "benchmark_scrapingdog.json": (
        "scrapingdog",
        (
            "RESEARCH_LAB_V2_BENCHMARK_SCRAPINGDOG_API_KEY",
            *_BOOT_SOURCES["scrapingdog"],
        ),
    ),
    "stale_parent_openrouter.json": (
        "openrouter",
        _BOOT_SOURCES["openrouter"],
    ),
    "source_add_judge_openrouter.json": (
        "openrouter",
        (
            "RESEARCH_LAB_V2_SOURCE_ADD_JUDGE_OPENROUTER_API_KEY",
            *_BOOT_SOURCES["openrouter"],
        ),
    ),
}


class GatewayEnvelopePreparationV2Error(RuntimeError):
    """The operator input cannot produce a complete encrypted V2 profile."""


def load_environment_file(path: Path) -> Dict[str, str]:
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise GatewayEnvelopePreparationV2Error(
            "gateway source environment is unavailable"
        ) from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        if not isinstance(parsed, Mapping):
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment JSON must be an object"
            )
        return {str(name): str(value) for name, value in parsed.items()}
    result: Dict[str, str] = {}
    for raw_line in raw.replace("\x00", "\n").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        try:
            parts = shlex.split(line, posix=True)
        except ValueError as exc:
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment is malformed"
            ) from exc
        if len(parts) != 1 or "=" not in parts[0]:
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment is malformed"
            )
        name, value = parts[0].split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise GatewayEnvelopePreparationV2Error(
                "gateway source environment name is invalid"
            )
        result[name] = value
    return result


def _secret(env: Mapping[str, str], names: Sequence[str]) -> tuple[str, str]:
    for name in names:
        value = str(env.get(name) or "").strip()
        if value:
            return name, value
    raise GatewayEnvelopePreparationV2Error(
        "required gateway credential is unavailable: %s" % ",".join(names)
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2)
        handle.write("\n")
    path.chmod(0o600)


def _proxy_names(
    env: Mapping[str, str], prefixes: Sequence[str]
) -> Dict[str, str]:
    names: Dict[str, str] = {}
    for index in range(1, 501):
        for prefix in prefixes:
            name = "%s_%d" % (prefix, index)
            value = str(env.get(name) or "").strip()
            if value:
                names.setdefault(value, name)
                break
    for prefix in prefixes:
        value = str(env.get(prefix) or "").strip()
        if value:
            names.setdefault(value, prefix)
    return names


def prepare_gateway_envelopes_v2(
    *,
    environment: Mapping[str, str],
    kms_key_id: str,
    deploy_commit: str,
    output_dir: Path,
    kms_client: Any = None,
) -> Dict[str, Any]:
    commit = str(deploy_commit or "").lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise GatewayEnvelopePreparationV2Error("gateway deploy commit is invalid")
    destination = Path(output_dir)
    if destination.exists():
        raise GatewayEnvelopePreparationV2Error(
            "gateway V2 envelope output already exists"
        )
    if kms_client is None:
        import boto3

        kms_client = boto3.client("kms")
    plan = build_research_lab_worker_autostart_plan(environment)
    if not plan.hosted.enabled or not plan.scoring.enabled:
        raise GatewayEnvelopePreparationV2Error(
            "configured hosted and scoring worker fleets are required"
        )
    if not plan.hosted.proxy_values or not plan.scoring.proxy_values:
        raise GatewayEnvelopePreparationV2Error(
            "worker proxy values are required for initial V2 sealing"
        )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".gateway-v2-envelopes.", dir=parent))
    os.chmod(staging, 0o700)
    removal_names = set()
    try:
        artifact_key = secrets.token_bytes(32)
        artifact_context = {
            "leadpoet:commit": commit,
            "leadpoet:purpose": "gateway-artifact-master-key-v2",
            "leadpoet:slot": "artifact_master_key",
        }
        _write_json(
            staging / "artifact_master_key.json",
            build_provider_envelope_v2(
                credential_slot="artifact_master_key",
                plaintext=artifact_key,
                credential_ref_hash=artifact_master_key_reference_hash(artifact_key),
                kms_key_id=kms_key_id,
                encryption_context=artifact_context,
                kms_client=kms_client,
                allow_binary=True,
            ),
        )
        del artifact_key

        boot_values: Dict[str, str] = {}
        for slot, names in _BOOT_SOURCES.items():
            source_name, value = _secret(environment, names)
            if slot not in _SHARED_PARENT_SLOTS:
                removal_names.add(source_name)
            boot_values[slot] = value
            _write_json(
                staging / (slot + ".json"),
                build_provider_envelope_v2(
                    credential_slot=slot,
                    plaintext=value.encode("utf-8"),
                    credential_ref_hash=credential_reference_hash(value),
                    kms_key_id=kms_key_id,
                    encryption_context={
                        "leadpoet:commit": commit,
                        "leadpoet:purpose": "gateway-boot-credential-v2",
                        "leadpoet:slot": slot,
                    },
                    kms_client=kms_client,
                ),
            )

        for filename, (slot, names) in _SPECIAL_PROFILES.items():
            source_name, value = _secret(environment, names)
            if source_name.startswith("RESEARCH_LAB_V2_"):
                removal_names.add(source_name)
            _write_json(
                staging / filename,
                build_provider_envelope_v2(
                    credential_slot=slot,
                    plaintext=value.encode("utf-8"),
                    credential_ref_hash=credential_reference_hash(value),
                    kms_key_id=kms_key_id,
                    encryption_context={
                        "leadpoet:commit": commit,
                        "leadpoet:profile": filename.removesuffix(".json"),
                        "leadpoet:purpose": "gateway-provider-profile-v2",
                        "leadpoet:slot": slot,
                    },
                    kms_client=kms_client,
                ),
            )

        proxy_sources = {
            "gateway_autoresearch": _proxy_names(
                environment, HOSTED_PROXY_PREFIXES
            ),
            "gateway_scoring": _proxy_names(environment, SCORING_PROXY_PREFIXES),
        }
        fleets = {
            "gateway_autoresearch": (
                plan.hosted.proxy_values,
                "autoresearch_proxy_{:02d}.json",
            ),
            "gateway_scoring": (
                plan.scoring.proxy_values,
                "scoring_proxy_{:02d}.json",
            ),
        }
        for role, (values, filename_template) in fleets.items():
            for index, value in enumerate(values):
                source_name = proxy_sources[role].get(value)
                if not source_name:
                    raise GatewayEnvelopePreparationV2Error(
                        "worker proxy source identity is unavailable"
                    )
                removal_names.add(source_name)
                _write_json(
                    staging / filename_template.format(index),
                    build_provider_envelope_v2(
                        credential_slot="egress_proxy",
                        plaintext=value.encode("utf-8"),
                        credential_ref_hash=credential_reference_hash(value),
                        kms_key_id=kms_key_id,
                        encryption_context={
                            "leadpoet:commit": commit,
                            "leadpoet:purpose": "gateway-worker-egress-v2",
                            "leadpoet:role": role,
                            "leadpoet:worker_index": str(index),
                        },
                        kms_client=kms_client,
                    ),
                )
        report = {
            "schema_version": "leadpoet.gateway_envelope_transition.v2",
            "deploy_commit": commit,
            "hosted_worker_count": plan.hosted.worker_count,
            "scoring_worker_count": plan.scoring.worker_count,
            "required_count_environment": {
                "RESEARCH_LAB_HOSTED_WORKER_PROCESS_COUNT": str(
                    plan.hosted.worker_count
                ),
                "RESEARCH_LAB_SCORING_WORKER_PROCESS_COUNT": str(
                    plan.scoring.worker_count
                ),
            },
            "plaintext_environment_names_to_remove": sorted(removal_names),
            "envelope_file_count": len(list(staging.glob("*.json"))) + 1,
        }
        _write_json(staging / "gateway-v2-env-transition.json", report)
        staging.rename(destination)
        return report
    except Exception:
        for path in staging.glob("*"):
            path.unlink(missing_ok=True)
        staging.rmdir()
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--kms-key-id", required=True)
    parser.add_argument("--deploy-commit", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = prepare_gateway_envelopes_v2(
        environment=load_environment_file(args.env_file),
        kms_key_id=args.kms_key_id,
        deploy_commit=args.deploy_commit,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

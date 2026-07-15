"""Encrypted, job-scoped provider profiles for measured Research Lab work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence

from gateway.utils.tee_kms_provision_v2 import (
    JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
    provision_job_provider_envelope_v2,
    validate_job_provider_envelope,
    validate_provider_envelope,
)
from leadpoet_canonical.attested_v2 import sha256_json


DEFAULT_PROFILE = "default"
BENCHMARK_MODEL_PROFILE = "benchmark_model"
BENCHMARK_SCORER_PROFILE = "benchmark_scorer"
STALE_PARENT_REPAIR_PROFILE = "stale_parent_repair"
SOURCE_ADD_JUDGE_PROFILE = "source_add_judge"
DEFAULT_CONFIG_DIR = Path("/home/ec2-user/.config/leadpoet/v2")

_PROFILE_FILES = {
    DEFAULT_PROFILE: (),
    BENCHMARK_MODEL_PROFILE: (("exa", "benchmark_exa.json"),),
    BENCHMARK_SCORER_PROFILE: (
        ("openrouter", "benchmark_openrouter.json"),
        ("scrapingdog", "benchmark_scrapingdog.json"),
    ),
    STALE_PARENT_REPAIR_PROFILE: (
        ("openrouter", "stale_parent_openrouter.json"),
    ),
    SOURCE_ADD_JUDGE_PROFILE: (
        ("openrouter", "source_add_judge_openrouter.json"),
    ),
}
_EXECUTION_PROXY_FILES = {
    "gateway_scoring": "scoring_proxy_{worker_index:02d}.json",
    "gateway_autoresearch": "autoresearch_proxy_{worker_index:02d}.json",
}
_MAX_CONFIGURED_WORKERS = 500


class ProviderProfileV2Error(RuntimeError):
    """A measured provider profile is malformed or cannot be leased safely."""


def load_provider_profile_v2(
    profile: str,
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
    execution_role: str = "",
    worker_index: Optional[int] = None,
    require_egress_proxy: bool = False,
) -> Dict[str, Any]:
    normalized_profile = str(profile or DEFAULT_PROFILE)
    entries = _PROFILE_FILES.get(normalized_profile)
    if entries is None:
        raise ProviderProfileV2Error("provider profile is not measured")
    normalized_role = str(execution_role or "")
    if normalized_role:
        proxy_spec = _EXECUTION_PROXY_FILES.get(normalized_role)
        if proxy_spec is None or worker_index is None:
            raise ProviderProfileV2Error("provider profile execution scope is invalid")
        filename_template = proxy_spec
        normalized_worker_index = int(worker_index)
        if not 0 <= normalized_worker_index < _MAX_CONFIGURED_WORKERS:
            raise ProviderProfileV2Error("provider profile worker index is invalid")
        entries = tuple(entries) + (
            (
                "egress_proxy",
                filename_template.format(worker_index=normalized_worker_index),
            ),
        )
    elif worker_index is not None or require_egress_proxy:
        raise ProviderProfileV2Error("provider profile proxy scope is incomplete")
    else:
        normalized_worker_index = None
    envelopes = []
    credential_refs = {}
    for provider_id, filename in entries:
        path = Path(config_dir) / filename
        if not path.exists():
            if provider_id == "egress_proxy" and require_egress_proxy:
                raise ProviderProfileV2Error(
                    "required worker TLS proxy envelope is unavailable"
                )
            continue
        if not path.is_file() or path.is_symlink():
            raise ProviderProfileV2Error(
                "provider profile envelope path is not a regular file"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProviderProfileV2Error(
                "provider profile envelope is unreadable"
            ) from exc
        try:
            envelope = validate_provider_envelope(value)
        except Exception as exc:
            raise ProviderProfileV2Error(
                "provider profile envelope is invalid"
            ) from exc
        if envelope["credential_slot"] != provider_id:
            raise ProviderProfileV2Error(
                "provider profile envelope slot differs from provider"
            )
        credential_refs[provider_id] = str(envelope["credential_ref_hash"])
        envelopes.append(
            {
                "provider_id": provider_id,
                "path": str(path),
                "envelope": {
                    name: value
                    for name, value in envelope.items()
                    if name != "ciphertext_blob"
                },
            }
        )
    return {
        "profile": normalized_profile,
        "execution_role": normalized_role,
        "worker_index": normalized_worker_index,
        "egress_proxy_required": bool(require_egress_proxy),
        "credential_ref_hashes": dict(sorted(credential_refs.items())),
        "envelopes": envelopes,
        "profile_hash": sha256_json(
            {
                "schema_version": "leadpoet.provider_profile.v2",
                "profile": normalized_profile,
                "execution_role": normalized_role,
                "worker_index": normalized_worker_index,
                "egress_proxy_required": bool(require_egress_proxy),
                "credential_ref_hashes": dict(sorted(credential_refs.items())),
            }
        ),
    }


def bind_provider_profile_envelopes_v2(
    profile_document: Mapping[str, Any],
    *,
    job_id: str,
) -> Sequence[Dict[str, Any]]:
    profile = str(profile_document.get("profile") or "")
    expected_refs = dict(profile_document.get("credential_ref_hashes") or {})
    output = []
    for item in profile_document.get("envelopes") or ():
        if not isinstance(item, Mapping) or not isinstance(item.get("envelope"), Mapping):
            raise ProviderProfileV2Error("provider profile envelope entry is invalid")
        provider_id = str(item.get("provider_id") or "")
        envelope = validate_provider_envelope(item["envelope"])
        credential_hash = str(envelope["credential_ref_hash"])
        if expected_refs.get(provider_id) != credential_hash:
            raise ProviderProfileV2Error(
                "provider profile credential commitment differs"
            )
        job_envelope = {
            "schema_version": JOB_PROVIDER_ENVELOPE_SCHEMA_VERSION,
            "job_id": str(job_id),
            "credential_slot": provider_id,
            "credential_ref_hash": credential_hash,
            "credential_value_hash": credential_hash,
            "key_ref_hash": sha256_json(
                {
                    "schema_version": "leadpoet.provider_profile_key_ref.v2",
                    "profile": profile,
                    "provider_id": provider_id,
                    "credential_ref_hash": credential_hash,
                }
            ),
            "ciphertext_blob_b64": envelope["ciphertext_blob_b64"],
            "ciphertext_blob_hash": envelope["ciphertext_blob_hash"],
            "kms_key_id_hash": envelope["kms_key_id_hash"],
            "encryption_context": dict(envelope["encryption_context"]),
            "encryption_context_hash": envelope["encryption_context_hash"],
        }
        output.append(validate_job_provider_envelope(job_envelope))
    if {
        item["credential_slot"]: item["credential_value_hash"] for item in output
    } != expected_refs:
        raise ProviderProfileV2Error("provider profile envelope set is incomplete")
    return tuple(output)


async def provision_provider_profile_v2(
    profile_document: Mapping[str, Any],
    *,
    job_id: str,
    client: Any,
    provision: Any = provision_job_provider_envelope_v2,
) -> Dict[str, Any]:
    envelopes = bind_provider_profile_envelopes_v2(
        profile_document,
        job_id=job_id,
    )
    results = []
    try:
        for envelope in envelopes:
            results.append(await provision(envelope, client=client))
    except Exception:
        await client.v2_release_job_credentials(str(job_id))
        raise
    return {
        "profile": str(profile_document.get("profile") or ""),
        "job_id": str(job_id),
        "credential_ref_hashes": dict(
            profile_document.get("credential_ref_hashes") or {}
        ),
        "leased_credential_count": len(results),
        "results": results,
    }


def verify_required_worker_proxy_profiles_v2(
    *,
    config_dir: Path = DEFAULT_CONFIG_DIR,
) -> Dict[str, Any]:
    refs = {}
    worker_counts = {}
    for execution_role, filename_template in sorted(_EXECUTION_PROXY_FILES.items()):
        filename_prefix = filename_template.partition("{")[0]
        pattern = re.compile(r"^%s([0-9]+)\.json$" % re.escape(filename_prefix))
        worker_indexes = []
        for path in Path(config_dir).glob("%s*.json" % filename_prefix):
            match = pattern.fullmatch(path.name)
            if match is None:
                raise ProviderProfileV2Error(
                    "worker TLS proxy profile filename is invalid"
                )
            worker_index = int(match.group(1))
            if path.name != filename_template.format(worker_index=worker_index):
                raise ProviderProfileV2Error(
                    "worker TLS proxy profile filename is not canonical"
                )
            worker_indexes.append(worker_index)
        worker_indexes.sort()
        if not worker_indexes:
            raise ProviderProfileV2Error(
                "%s worker TLS proxy profiles are unavailable" % execution_role
            )
        if worker_indexes != list(range(len(worker_indexes))):
            raise ProviderProfileV2Error(
                "%s worker TLS proxy profiles must be contiguous" % execution_role
            )
        if len(worker_indexes) > _MAX_CONFIGURED_WORKERS:
            raise ProviderProfileV2Error(
                "%s worker TLS proxy profile count exceeds the limit"
                % execution_role
            )
        worker_counts[execution_role] = len(worker_indexes)
        for worker_index in worker_indexes:
            document = load_provider_profile_v2(
                DEFAULT_PROFILE,
                config_dir=config_dir,
                execution_role=execution_role,
                worker_index=worker_index,
                require_egress_proxy=True,
            )
            proxy_hash = dict(document["credential_ref_hashes"]).get(
                "egress_proxy"
            )
            if not proxy_hash:
                raise ProviderProfileV2Error(
                    "worker TLS proxy profile is incomplete"
                )
            refs["%s:%02d" % (execution_role, worker_index)] = proxy_hash
    required_profiles = {
        BENCHMARK_MODEL_PROFILE: frozenset({"exa"}),
        BENCHMARK_SCORER_PROFILE: frozenset({"openrouter", "scrapingdog"}),
        STALE_PARENT_REPAIR_PROFILE: frozenset({"openrouter"}),
        SOURCE_ADD_JUDGE_PROFILE: frozenset({"openrouter"}),
    }
    for profile, required_slots in sorted(required_profiles.items()):
        document = load_provider_profile_v2(profile, config_dir=config_dir)
        observed_slots = frozenset(document["credential_ref_hashes"])
        if observed_slots != required_slots:
            raise ProviderProfileV2Error(
                "%s provider profile is incomplete" % profile
            )
    return {
        "schema_version": "leadpoet.worker_proxy_profile_set.v2",
        "status": "ready",
        "profile_count": len(refs),
        "worker_counts": dict(sorted(worker_counts.items())),
        "profile_ref_hash": sha256_json(dict(sorted(refs.items()))),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--require-worker-proxies", action="store_true")
    args = parser.parse_args()
    if not args.require_worker_proxies:
        raise ProviderProfileV2Error(
            "provider profile verification mode is required"
        )
    print(
        json.dumps(
            verify_required_worker_proxy_profiles_v2(
                config_dir=args.config_dir,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

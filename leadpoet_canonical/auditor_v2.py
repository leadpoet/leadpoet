"""Independent PCR0 and Nitro verification for authoritative V2 weights."""

from __future__ import annotations

import functools
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional

from leadpoet_canonical.attested_v2 import (
    sha256_json,
    validate_receipt_graph,
    verify_boot_identity_nitro,
)
from leadpoet_canonical.weight_authority_v2 import (
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)


IDENTITY_CACHE_SCHEMA_VERSION = "leadpoet.independent_pcr0_identities.v2"
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class AuditorV2Error(ValueError):
    """A V2 bundle lacks independently rebuilt code identity evidence."""


def load_identity_cache(path: Path) -> Dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuditorV2Error(
            "independent PCR0 identity cache is unavailable"
        ) from exc
    if (
        not isinstance(document, Mapping)
        or document.get("schema_version") != IDENTITY_CACHE_SCHEMA_VERSION
        or not isinstance(document.get("entries"), list)
    ):
        raise AuditorV2Error("independent PCR0 identity cache schema is invalid")
    return {
        "schema_version": IDENTITY_CACHE_SCHEMA_VERSION,
        "entries": list(document["entries"]),
    }


def _identity_for_boot(
    cache: Mapping[str, Any],
    boot: Mapping[str, Any],
) -> Dict[str, Any]:
    physical_role = str(boot.get("physical_role") or "")
    service_role = str(boot.get("role") or "")
    commit_sha = str(boot.get("commit_sha") or "").lower()
    if not physical_role or not service_role or not _COMMIT_RE.fullmatch(commit_sha):
        raise AuditorV2Error("boot code identity is invalid")
    matches = [
        entry
        for entry in cache.get("entries", [])
        if isinstance(entry, Mapping)
        and entry.get("physical_role") == physical_role
        and entry.get("role") == service_role
        and str(entry.get("commit_sha") or "").lower() == commit_sha
    ]
    if len(matches) != 1:
        raise AuditorV2Error(
            "boot has no unique independently rebuilt PCR0 identity"
        )
    entry = dict(matches[0])
    pcr0 = str(entry.get("pcr0") or "").lower()
    manifest_hash = str(entry.get("build_manifest_hash") or "").lower()
    dependency_hash = str(entry.get("dependency_lock_hash") or "").lower()
    if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
        raise AuditorV2Error("independent PCR0 identity is invalid")
    if not _HASH_RE.fullmatch(manifest_hash) or not _HASH_RE.fullmatch(
        dependency_hash
    ):
        raise AuditorV2Error("independent build closure identity is invalid")
    if int(entry.get("verified_build_count") or 0) < 3:
        raise AuditorV2Error(
            "independent PCR0 identity needs three matching builds"
        )
    if pcr0 != str(boot.get("pcr0") or "").lower():
        raise AuditorV2Error("boot PCR0 differs from independent build")
    if manifest_hash != str(boot.get("build_manifest_hash") or "").lower():
        raise AuditorV2Error("boot manifest differs from independent build")
    if dependency_hash != str(boot.get("dependency_lock_hash") or "").lower():
        raise AuditorV2Error("boot dependency lock differs from independent build")
    return entry


def verify_attested_weight_bundle_v2(
    bundle: Mapping[str, Any],
    *,
    identity_cache: Mapping[str, Any],
    boot_verifier: Optional[Callable[..., Any]] = None,
    require_allocation_ancestry: bool = True,
) -> Dict[str, Any]:
    """Verify the canonical graph and every boot against independent builds."""

    # Kept in the signature for source compatibility. Authoritative V2 always
    # requires every semantic input category, including allocation ancestry.
    if not require_allocation_ancestry:
        raise AuditorV2Error("authoritative V2 allocation ancestry is mandatory")
    # Boot identities are issued once at enclave boot and verified for days
    # afterwards; the Nitro leaf certificate is only valid for hours, so the
    # real verifier must check certificate validity at attestation time, the
    # same setting every internal verifier uses. Injected test verifiers keep
    # their own signature.
    verifier = boot_verifier or functools.partial(
        verify_boot_identity_nitro,
        certificate_validity_at_attestation_time=True,
    )
    observed = []

    def verify_one(boot: Mapping[str, Any]) -> Any:
        identity = _identity_for_boot(identity_cache, boot)
        result = verifier(boot, expected_pcr0=identity["pcr0"])
        observed.append(
            {
                "boot_identity_hash": str(boot["boot_identity_hash"]),
                "physical_role": str(boot["physical_role"]),
                "role": str(boot["role"]),
                "commit_sha": str(boot["commit_sha"]),
                "pcr0": str(identity["pcr0"]),
                "build_manifest_hash": str(identity["build_manifest_hash"]),
                "dependency_lock_hash": str(identity["dependency_lock_hash"]),
                "verified_build_count": int(identity["verified_build_count"]),
            }
        )
        return result

    verified = validate_published_weight_bundle_v2(
        bundle,
        boot_attestation_verifier=verify_one,
        require_boot_attestation_verification=True,
    )
    return {
        **verified,
        "independent_receipt_identities": sorted(
            observed,
            key=lambda item: (item["physical_role"], item["boot_identity_hash"]),
        ),
    }


def verify_attested_weight_authority_v2(
    authority: Mapping[str, Any],
    *,
    identity_cache: Mapping[str, Any],
    chain_signing_profile: Mapping[str, Any],
    boot_verifier: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Verify bundle, durable gateway publication, and finalized chain proof."""

    if not isinstance(authority, Mapping) or set(authority) != {
        "schema_version",
        "bundle",
        "publication",
        "finalization",
    }:
        raise AuditorV2Error("published V2 weight authority fields are invalid")
    if authority.get("schema_version") != "leadpoet.published_weight_authority.v2":
        raise AuditorV2Error("published V2 weight authority schema is invalid")
    bundle = authority.get("bundle")
    publication = authority.get("publication")
    finalization = authority.get("finalization")
    if not all(isinstance(item, Mapping) for item in (bundle, publication, finalization)):
        raise AuditorV2Error("published V2 weight authority components are invalid")
    verified_bundle = verify_attested_weight_bundle_v2(
        bundle,
        identity_cache=identity_cache,
        boot_verifier=boot_verifier,
    )

    # Boot identities are issued once at enclave boot and verified for days
    # afterwards; the Nitro leaf certificate is only valid for hours, so the
    # real verifier must check certificate validity at attestation time, the
    # same setting every internal verifier uses. Injected test verifiers keep
    # their own signature.
    verifier = boot_verifier or functools.partial(
        verify_boot_identity_nitro,
        certificate_validity_at_attestation_time=True,
    )
    observed = []

    def verify_one(boot: Mapping[str, Any]) -> Any:
        identity = _identity_for_boot(identity_cache, boot)
        result = verifier(boot, expected_pcr0=identity["pcr0"])
        observed.append(str(boot["boot_identity_hash"]))
        return result

    if set(publication) != {
        "weight_submission_event_hash",
        "publication_receipt_hash",
        "publication_doc",
        "receipt_graph",
    }:
        raise AuditorV2Error("V2 publication fields are invalid")
    publication_graph = publication.get("receipt_graph")
    publication_doc = publication.get("publication_doc")
    if not isinstance(publication_graph, Mapping) or not isinstance(
        publication_doc, Mapping
    ):
        raise AuditorV2Error("V2 publication evidence is invalid")
    validate_receipt_graph(
        publication_graph,
        required_purposes={"gateway.weights.publication.v2"},
        boot_attestation_verifier=verify_one,
        require_boot_attestation_verification=True,
    )
    publication_root = str(publication_graph["root_receipt_hash"])
    receipt_by_hash = {
        str(receipt["receipt_hash"]): receipt
        for receipt in publication_graph["receipts"]
    }
    root_receipt = receipt_by_hash.get(publication_root)
    expected_publication_doc = {
        "schema_version": "leadpoet.weight_publication.v2",
        "bundle_hash": verified_bundle["bundle_hash"],
        "root_receipt_hash": verified_bundle["root_receipt_hash"],
        "durable_readback_hash": publication_doc.get("durable_readback_hash"),
        "transparency_event_hash": publication_doc.get("transparency_event_hash"),
    }
    if (
        dict(publication_doc) != expected_publication_doc
        or publication.get("publication_receipt_hash") != publication_root
        or not isinstance(root_receipt, Mapping)
        or root_receipt.get("role") != "gateway_coordinator"
        or root_receipt.get("purpose") != "gateway.weights.publication.v2"
        or root_receipt.get("parent_receipt_hashes")
        != [verified_bundle["root_receipt_hash"]]
        or root_receipt.get("output_root") != sha256_json(expected_publication_doc)
    ):
        raise AuditorV2Error("V2 publication receipt differs from bundle")
    expected_submission_event = sha256_json(
        {
            "bundle_hash": verified_bundle["bundle_hash"],
            "publication_receipt_hash": publication_root,
            "transparency_event_hash": expected_publication_doc[
                "transparency_event_hash"
            ],
            "durable_readback_hash": expected_publication_doc[
                "durable_readback_hash"
            ],
        }
    )
    if publication.get("weight_submission_event_hash") != expected_submission_event:
        raise AuditorV2Error("V2 publication event hash differs")

    if set(finalization) != {
        "weight_finalization_event_hash",
        "submission",
    } or not isinstance(finalization.get("submission"), Mapping):
        raise AuditorV2Error("V2 finalization fields are invalid")
    finalization_submission = finalization["submission"]
    verified_finalization = validate_weight_finalization_submission_v2(
        finalization_submission,
        chain_signing_profile=chain_signing_profile,
    )
    validate_receipt_graph(
        finalization_submission["receipt_graph"],
        required_purposes={
            "validator.weights.computed.v2",
            "validator.set_weights_extrinsic.v2",
            "validator.weights.finalized.v2",
        },
        boot_attestation_verifier=verify_one,
        require_boot_attestation_verification=True,
    )
    for field in (
        "validator_hotkey",
        "netuid",
        "epoch_id",
        "weights_hash",
        "weight_receipt_hash",
    ):
        if verified_finalization[field] != verified_bundle[field]:
            raise AuditorV2Error("V2 finalization differs from bundle at %s" % field)
    if (
        verified_finalization["weight_submission_event_hash"]
        != expected_submission_event
    ):
        raise AuditorV2Error("V2 finalization differs from publication event")
    expected_finalization_event = sha256_json(
        {
            "weight_submission_event_hash": expected_submission_event,
            "bundle_hash": verified_bundle["bundle_hash"],
            "finalization_receipt_hash": verified_finalization[
                "finalization_receipt_hash"
            ],
            "extrinsic_authorization_hash": verified_finalization[
                "extrinsic_authorization_hash"
            ],
            "extrinsic_hash": verified_finalization["extrinsic_hash"],
            "finalized_block": verified_finalization["finalized_block"],
            "finalized_block_hash": verified_finalization[
                "finalized_block_hash"
            ],
            "state_transition_hash": verified_finalization[
                "state_transition_hash"
            ],
        }
    )
    if finalization.get("weight_finalization_event_hash") != expected_finalization_event:
        raise AuditorV2Error("V2 finalization event hash differs")
    return {
        **verified_bundle,
        "weight_submission_event_hash": expected_submission_event,
        "weight_finalization_event_hash": expected_finalization_event,
        "extrinsic_hash": verified_finalization["extrinsic_hash"],
        "finalized_block": verified_finalization["finalized_block"],
        "finalized_block_hash": verified_finalization["finalized_block_hash"],
        "state_transition_hash": verified_finalization["state_transition_hash"],
        "finalization_receipt_hash": verified_finalization[
            "finalization_receipt_hash"
        ],
        "additional_verified_boots": sorted(set(observed)),
    }

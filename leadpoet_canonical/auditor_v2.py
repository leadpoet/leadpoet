"""Independent PCR0 and Nitro verification for authoritative V2 weights."""

from __future__ import annotations

import functools
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

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
_RELEASE_BUCKET = "leadpoet-attested-v2-artifacts-493765492819"
_RELEASE_PREFIX = "attested-v2/releases"
_MAX_RELEASE_CHANNEL_BYTES = 2 * 1024 * 1024


class AuditorV2Error(ValueError):
    """A V2 bundle lacks independently rebuilt code identity evidence."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _release_url(value: str, *, commit: str, version_id: str) -> str:
    parsed = urlsplit(str(value or ""))
    if (
        parsed.scheme != "https"
        or parsed.hostname
        not in {
            f"{_RELEASE_BUCKET}.s3.amazonaws.com",
            f"{_RELEASE_BUCKET}.s3.us-east-1.amazonaws.com",
        }
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or unquote(parsed.path)
        != f"/{_RELEASE_PREFIX}/{commit}/release-channel-v2.json"
    ):
        raise AuditorV2Error("auditor release verification URL violates policy")
    query = {key.lower(): values for key, values in parse_qs(parsed.query).items()}
    required = {
        "x-amz-algorithm",
        "x-amz-credential",
        "x-amz-date",
        "x-amz-expires",
        "x-amz-signedheaders",
        "x-amz-signature",
        "versionid",
    }
    if set(query) - (required | {"x-amz-security-token"}) or not required.issubset(query):
        raise AuditorV2Error("auditor release URL is not an exact S3 signature")
    try:
        expires = int(query["x-amz-expires"][0])
    except (TypeError, ValueError, IndexError) as exc:
        raise AuditorV2Error("auditor release URL expiry is invalid") from exc
    if (
        query["x-amz-algorithm"] != ["AWS4-HMAC-SHA256"]
        or query["versionid"] != [version_id]
        or "host" not in str(query["x-amz-signedheaders"][0]).split(";")
        or not 1 <= expires <= 300
    ):
        raise AuditorV2Error("auditor release URL signature policy differs")
    return parsed.geturl()


def _open_exact_url(url: str, *, method: str):
    return build_opener(_NoRedirect()).open(Request(url, method=method), timeout=30)


def identity_cache_from_release_channel(channel: Mapping[str, Any]) -> Dict[str, Any]:
    """Derive strict auditor identities from one validated six-build channel."""

    from gateway.tee.release_channel_v2 import validate_release_channel_v2

    normalized = validate_release_channel_v2(channel)
    entries = []
    gateway = normalized["gateway_release_manifest"]
    for physical_role, role in sorted(gateway["roles"].items()):
        entries.append(
            {
                "physical_role": physical_role,
                "role": role["service_role"],
                "commit_sha": role["commit_sha"],
                "pcr0": role["pcr0"],
                "build_manifest_hash": role["execution_manifest_hash"],
                "dependency_lock_hash": role["dependency_lock_hash"],
                "verified_build_count": role["verified_build_count"],
            }
        )
    validator_manifest = normalized["validator_release_manifest"]
    validator = validator_manifest["release"]
    entries.append(
        {
            "physical_role": "validator_weights",
            "role": "validator_weights",
            "commit_sha": validator["commit_sha"],
            "pcr0": validator["pcr0"],
            "build_manifest_hash": validator["app_manifest_hash"],
            "dependency_lock_hash": validator["dependency_lock_hash"],
            "verified_build_count": validator_manifest["verified_build_count"],
        }
    )
    cache = {"schema_version": IDENTITY_CACHE_SCHEMA_VERSION, "entries": entries}
    for entry in entries:
        _identity_for_boot(cache, entry)
    return cache


def fetch_locked_release_identity_cache(
    evidence: Mapping[str, Any], *, http_open: Any = None
) -> Dict[str, Any]:
    """Fetch one exact Object-Locked release and derive its identity cache."""

    fields = {
        "schema_version",
        "commit_sha",
        "release_channel_version_id",
        "release_channel_get_url",
        "release_channel_head_url",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        raise AuditorV2Error("auditor release evidence fields are invalid")
    if evidence.get("schema_version") != "leadpoet.auditor_release_evidence.v2":
        raise AuditorV2Error("auditor release evidence schema is invalid")
    commit = str(evidence.get("commit_sha") or "").lower()
    version_id = str(evidence.get("release_channel_version_id") or "")
    if not _COMMIT_RE.fullmatch(commit) or len(commit) != 40 or not version_id:
        raise AuditorV2Error("auditor release identity is invalid")
    get_url = _release_url(
        str(evidence["release_channel_get_url"]),
        commit=commit,
        version_id=version_id,
    )
    head_url = _release_url(
        str(evidence["release_channel_head_url"]),
        commit=commit,
        version_id=version_id,
    )
    open_url = http_open or _open_exact_url
    try:
        with open_url(head_url, method="HEAD") as response:
            headers = {key.lower(): value for key, value in response.headers.items()}
            if response.geturl() != head_url:
                raise AuditorV2Error("auditor release verification redirected")
        with open_url(get_url, method="GET") as response:
            get_headers = {key.lower(): value for key, value in response.headers.items()}
            if response.geturl() != get_url:
                raise AuditorV2Error("auditor release verification redirected")
            payload = response.read(_MAX_RELEASE_CHANNEL_BYTES + 1)
    except AuditorV2Error:
        raise
    except Exception as exc:
        raise AuditorV2Error("auditor immutable release is unavailable") from exc
    if (
        headers.get("x-amz-object-lock-mode", "").upper() != "COMPLIANCE"
        or headers.get("x-amz-version-id") != version_id
        or get_headers.get("x-amz-version-id") != version_id
    ):
        raise AuditorV2Error("auditor release is not the Object-Locked version")
    try:
        retain_until = datetime.fromisoformat(
            headers["x-amz-object-lock-retain-until-date"].replace("Z", "+00:00")
        )
    except (KeyError, ValueError) as exc:
        raise AuditorV2Error("auditor release retention is invalid") from exc
    if retain_until <= datetime.now(timezone.utc):
        raise AuditorV2Error("auditor release retention has expired")
    if not payload or len(payload) > _MAX_RELEASE_CHANNEL_BYTES:
        raise AuditorV2Error("auditor release channel size is invalid")
    try:
        channel = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise AuditorV2Error("auditor release channel is invalid JSON") from exc
    cache = identity_cache_from_release_channel(channel)
    if any(entry["commit_sha"] != commit for entry in cache["entries"]):
        raise AuditorV2Error("auditor release channel commit differs")
    return cache


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

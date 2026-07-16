"""
Weight Submission Endpoint for TEE-Verified Validators
======================================================

This endpoint accepts weight submissions from primary validators running in TEE.
The gateway acts as a VERIFIER, NOT an ORACLE - it only verifies signatures and
records authenticated data.

Verification Order (MANDATORY):
0. Basic invariants (lengths, sorted UIDs, no zeros, valid u16 range)
1. Authorization: validator_hotkey in PRIMARY_VALIDATOR_HOTKEYS
2. Single-source-of-truth: Reject duplicate (netuid, epoch_id, validator_hotkey)
3. Epoch freshness: Validate block against gateway-observed chain
4. Attestation: Nitro verification + epoch_id in user_data (fail-closed in production)
5. Hotkey binding: sr25519 signature over binding_message
6. Ed25519 signature: Over digest BYTES of weights_hash
7. Recompute hash: bundle_weights_hash must match submission.weights_hash

Security:
- FAIL-CLOSED: Production MUST verify all attestations
- Gateway does NOT interpret or decide validation results
- All events are signed and hash-chained for auditor verification
"""

import asyncio
import os
import base64
import hashlib
import json
import logging
import re
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Canonical imports (MUST use shared module)
from leadpoet_canonical.weights import bundle_weights_hash, compare_weights_hash
from leadpoet_canonical.chain import normalize_chain_weights
from leadpoet_canonical.binding import parse_binding_message, verify_binding_message
from leadpoet_canonical.timestamps import canonical_timestamp
from leadpoet_canonical.constants import EPOCH_LENGTH, WEIGHT_SUBMISSION_BLOCK
from leadpoet_canonical.attested_receipts import SCORING_PURPOSES, WEIGHT_PURPOSE
from leadpoet_canonical.weight_bundle_v2 import (
    WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    validate_weight_bundle_v2,
)
from leadpoet_canonical.attested_v2 import sha256_json, verify_boot_identity_nitro
from leadpoet_canonical.hotkey_authority_v2 import (
    validate_weight_inputs_request_v2,
    weight_inputs_request_message_v2,
)
from leadpoet_canonical.weight_authority_v2 import (
    PUBLISHED_WEIGHT_BUNDLE_V2_SCHEMA_VERSION,
    validate_published_weight_bundle_v2,
    validate_weight_finalization_submission_v2,
)
from gateway.research_lab.arweave_audit import publish_research_lab_epoch_audit
from gateway.research_lab.config import ResearchLabGatewayConfig
from gateway.utils.signature import verify_wallet_signature

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/weights", tags=["weights"])

# ============================================================================
# Configuration
# ============================================================================

# Import network config for testnet guard
from gateway.config import BITTENSOR_NETWORK

MAX_BLOCK_DRIFT = 30  # Max allowed drift from gateway-observed block

# Build identifier for transparency log
BUILD_ID = os.environ.get("BUILD_ID", "production-gateway-tee")
PCR0_ALLOWLIST_URL = os.environ.get(
    "PCR0_ALLOWLIST_URL",
    "https://raw.githubusercontent.com/leadpoet/leadpoet/main/pcr0_allowlist.json",
)
_COMMIT_HASH_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_NOTES_COMMIT_RE = re.compile(r"\bcommit\s+([0-9a-fA-F]{7,40})\b")
_PCR0_RE = re.compile(r"^[0-9a-f]{96}$")

# PCR0 is the ROOT OF TRUST - code_hash in user_data is INFORMATIONAL only
# The verify_nitro_attestation_full function checks PCR0 against the allowlist
# which is fetched from GitHub automatically

# Filter empty strings - .split(",") on "" produces [""]
_primary_hotkeys_str = os.environ.get("PRIMARY_VALIDATOR_HOTKEYS", "")
PRIMARY_VALIDATOR_HOTKEYS: Set[str] = {x.strip() for x in _primary_hotkeys_str.split(",") if x.strip()}

# Allowed netuids (empty = allow all in dev mode)
_allowed_netuids_str = os.environ.get("ALLOWED_NETUIDS", "")
ALLOWED_NETUIDS: Set[int] = {int(x) for x in _allowed_netuids_str.split(",") if x.strip().isdigit()}

# Chain endpoint for binding message verification
EXPECTED_CHAIN = os.environ.get("EXPECTED_CHAIN", "wss://entrypoint-finney.opentensor.ai:443")

# Subtensor for block validation (lazily initialized)
_subtensor = None


def get_subtensor():
    """Get or create subtensor instance for chain queries."""
    global _subtensor
    if _subtensor is None:
        import bittensor as bt
        _subtensor = bt.Subtensor()
    return _subtensor


def _extract_commit_hash_from_allowlist_entry(entry: dict) -> Optional[str]:
    """Return commit metadata from a PCR0 allowlist entry, if present."""
    return (
        _extract_explicit_commit_hash_from_allowlist_entry(entry)
        or _extract_notes_commit_hash_from_allowlist_entry(entry)
    )


def _extract_explicit_commit_hash_from_allowlist_entry(entry: dict) -> Optional[str]:
    """Return structured commit metadata from a PCR0 allowlist entry."""
    for key in ("commit_hash", "git_commit_sha", "git_commit", "commit"):
        value = entry.get(key)
        if isinstance(value, str) and _COMMIT_HASH_RE.fullmatch(value.strip()):
            return value.strip().lower()

    return None


def _extract_notes_commit_hash_from_allowlist_entry(entry: dict) -> Optional[str]:
    """Return legacy free-text commit metadata from a PCR0 allowlist entry."""
    notes = entry.get("notes")
    if isinstance(notes, str):
        match = _NOTES_COMMIT_RE.search(notes)
        if match:
            return match.group(1).lower()

    return None


def _lookup_pcr0_commit_hash_from_allowlist(pcr0_hex: str) -> Optional[str]:
    """
    Resolve static-allowlist PCR0 approvals back to a commit for audit storage.

    Dynamic GitHub verification already returns pcr0_commit. Static allowlist
    fallback must also store a non-null commit hash because auditors rely on
    published_weight_bundles.pcr0_commit_hash.
    """
    if not pcr0_hex or pcr0_hex == "N/A":
        return None

    allowlist_docs = []
    try:
        request = urllib.request.Request(
            PCR0_ALLOWLIST_URL,
            headers={"User-Agent": "LeadPoet-Gateway/1.0"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            allowlist_docs.append(json.loads(response.read().decode("utf-8")))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as e:
        logger.warning(f"[PCR0] Could not fetch allowlist metadata from GitHub: {e}")

    try:
        with open("pcr0_allowlist.json", "r", encoding="utf-8") as handle:
            allowlist_docs.append(json.load(handle))
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"[PCR0] Could not read local allowlist metadata: {e}")

    explicit_matches = []
    legacy_note_matches = []
    for doc in allowlist_docs:
        for entry in doc.get("validator_pcr0", []):
            if entry.get("pcr0") != pcr0_hex:
                continue

            explicit = _extract_explicit_commit_hash_from_allowlist_entry(entry)
            if explicit:
                explicit_matches.append(explicit)
                continue

            notes_commit = _extract_notes_commit_hash_from_allowlist_entry(entry)
            if notes_commit:
                legacy_note_matches.append(notes_commit)

    if explicit_matches:
        return explicit_matches[0]
    if legacy_note_matches:
        return legacy_note_matches[0]

    return None


# ============================================================================
# Models
# ============================================================================

class WeightSubmission(BaseModel):
    """
    Weight submission from validator TEE.
    
    MUST match Canonical Specifications exactly.
    NO floats, NO hotkeys in weights - only UIDs and u16.
    """
    netuid: int
    epoch_id: int
    block: int  # Block when weights were computed
    
    # Weights as parallel arrays (compact, matches bundle table)
    uids: List[int]  # Sorted ascending
    weights_u16: List[int]  # Corresponding u16 weights [0-65535]
    
    # Verification (from bundle_weights_hash - includes block)
    weights_hash: str  # SHA256 digest hex
    
    # Validator identity
    validator_hotkey: str  # ss58 address
    validator_enclave_pubkey: str  # Ed25519 hex
    validator_signature: str  # Ed25519 over digest BYTES (not hex string)
    
    # Attestation (with epoch_id in user_data for freshness)
    validator_attestation_b64: str
    validator_code_hash: str
    
    # Hotkey binding (proves enclave authorized by this hotkey)
    binding_message: str  # LEADPOET_VALIDATOR_BINDING|netuid=...|...
    validator_hotkey_signature: str  # sr25519 over binding_message


class WeightSubmissionResponse(BaseModel):
    """Response after weight submission."""
    success: bool
    epoch_id: int
    weights_count: int
    message: str
    weight_submission_event_hash: Optional[str] = None


class WeightSubmissionV2(BaseModel):
    """Authoritative enclave-computed weight bundle and complete graph."""

    schema_version: str
    validator_hotkey: str
    binding_message: str
    validator_hotkey_signature: str
    weight_snapshot: Dict[str, Any]
    weight_result: Dict[str, Any]
    weights_signature: str
    receipt_graph: Dict[str, Any]


class WeightSubmissionV2Response(BaseModel):
    success: bool
    epoch_id: int
    weights_count: int
    weights_hash: str
    weight_receipt_hash: str
    weight_submission_event_hash: str
    message: str


class WeightFinalizationV2(BaseModel):
    """Validator-enclave proof of finalized extrinsic state transition."""

    schema_version: str
    validator_hotkey: str
    weight_submission_event_hash: str
    finalization: Dict[str, Any]
    receipt_graph: Dict[str, Any]


class WeightFinalizationV2Response(BaseModel):
    success: bool
    epoch_id: int
    weights_hash: str
    extrinsic_hash: str
    finalized_block: int
    weight_submission_event_hash: str
    weight_finalization_event_hash: str
    message: str


class WeightInputsV2Authorization(BaseModel):
    """Signed request for measured gateway-owned final-weight inputs."""

    request: Dict[str, Any]
    calculation_snapshot: Dict[str, Any]
    validator_hotkey_signature: str


class WeightInputsV2Response(BaseModel):
    request_hash: str
    calculation_snapshot_hash: str
    input_receipt_hashes: Dict[str, str]
    gateway_authority_event_hash: str
    upstream_receipt_set: Dict[str, Any]


def _gateway_v2_release_manifest() -> Dict[str, Any]:
    from gateway.tee.release_manifest_v2 import validate_release_manifest

    path = Path(
        os.environ.get(
            "GATEWAY_V2_RELEASE_MANIFEST",
            "/home/ec2-user/tee/gateway-v2-release-manifest.json",
        )
    ).expanduser()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("approved gateway V2 release manifest is unavailable") from exc
    return validate_release_manifest(value)


def _verify_authoritative_v2_boot(identity: Dict[str, Any]) -> Dict[str, Any]:
    """Verify one boot via a six-build gateway release or dynamic validator build."""

    physical_role = str(identity.get("physical_role") or "")
    if physical_role == "validator_weights":
        from gateway.utils.pcr0_builder import verify_pcr0

        rebuilt = verify_pcr0(str(identity.get("pcr0") or ""))
        if not rebuilt.get("valid"):
            raise ValueError("validator PCR0 is absent from the dynamic Git build cache")
        if str(rebuilt.get("commit_hash") or "").lower() != str(
            identity.get("commit_sha") or ""
        ).lower():
            raise ValueError("validator PCR0 commit differs from boot identity")
    else:
        release = _gateway_v2_release_manifest()
        expected = release["roles"].get(physical_role)
        if not isinstance(expected, dict):
            raise ValueError("gateway boot role is absent from the approved release")
        comparisons = {
            "commit_sha": identity.get("commit_sha"),
            "pcr0": identity.get("pcr0"),
            "build_manifest_hash": identity.get("build_manifest_hash"),
            "dependency_lock_hash": identity.get("dependency_lock_hash"),
        }
        for field, observed in comparisons.items():
            if str(observed or "").lower() != str(expected[field]).lower():
                raise ValueError("gateway boot differs from approved release at %s" % field)
    return verify_boot_identity_nitro(
        identity,
        expected_pcr0=str(identity.get("pcr0") or ""),
    )


def _validate_authoritative_v2_submission(
    submission: WeightSubmissionV2,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = submission.model_dump(mode="python")
    verified = validate_published_weight_bundle_v2(
        bundle,
        boot_attestation_verifier=_verify_authoritative_v2_boot,
        require_boot_attestation_verification=True,
    )
    validator_boots = [
        boot
        for boot in bundle["receipt_graph"].get("boot_identities", [])
        if isinstance(boot, dict)
        and boot.get("physical_role") == "validator_weights"
    ]
    if len(validator_boots) != 1:
        raise ValueError("V2 bundle needs exactly one validator boot identity")
    return verified, dict(validator_boots[0])


# ============================================================================
# Verification Helpers
# ============================================================================

def verify_validator_attestation(
    attestation_b64: str,
    expected_pubkey: str,
    expected_epoch_id: int,
) -> tuple:
    """
    Verify the validator's AWS Nitro attestation document.
    
    FAIL-CLOSED: NO DEV MODE BYPASS. Attestation verification is ALWAYS required.
    
    PCR0 (enclave image hash) is the ROOT OF TRUST:
    - PCR0 is checked against the allowlist (fetched from GitHub)
    - code_hash in user_data is INFORMATIONAL only (do NOT trust it alone)
    - A malicious enclave could claim any code_hash, but cannot fake PCR0
    
    Returns:
        (valid: bool, extracted_data: dict)
    """
    try:
        # FAIL-CLOSED: Full Nitro verification ALWAYS required
        # NO dev mode bypass - attestation is critical security
        from leadpoet_canonical.nitro import verify_nitro_attestation_full
        
        # PCR0 is verified against the allowlist which is:
        # 1. Fetched from GitHub automatically (cached 5 minutes)
        # 2. Contains only PCR0 values for approved enclave builds
        # 3. The validator CANNOT fake PCR0 - it's inside the AWS-signed attestation
        valid, data = verify_nitro_attestation_full(
            attestation_b64=attestation_b64,
            expected_pubkey=expected_pubkey,
            expected_purpose="validator_weights",
            expected_epoch_id=expected_epoch_id,
            role="validator",  # Uses ALLOWED_VALIDATOR_PCR0_VALUES
        )
        
        if valid:
            logger.info(f"[ATTESTATION] ✅ Full Nitro verification passed")
            logger.info(f"[ATTESTATION]    PCR0: {data.get('pcr0', 'N/A')[:32]}...")
            logger.info(f"[ATTESTATION]    Steps: {data.get('verification_steps', [])[-1]}")
        else:
            logger.error(f"[ATTESTATION] ❌ Verification failed: {data.get('error', 'Unknown')}")
        
        return valid, data
        
    except ImportError as e:
        # If nitro module not available, FAIL (don't bypass)
        logger.error(f"[ATTESTATION] ❌ CRITICAL: Nitro verification module not available: {e}")
        return False, {"error": f"Nitro verification unavailable: {e}"}
    except Exception as e:
        logger.error(f"[ATTESTATION] ❌ Failed: {e}")
        return False, {"error": str(e)}


def verify_validator_attestation_v2(
    *,
    receipt: Dict[str, Any],
    expected_epoch_id: int,
) -> tuple:
    """Require genuine Nitro plus the dynamic Git-derived validator PCR0 path."""

    try:
        from leadpoet_canonical.nitro import verify_nitro_attestation_full

        valid, data = verify_nitro_attestation_full(
            attestation_b64=str(receipt.get("attestation_document_b64") or ""),
            expected_pubkey=str(receipt.get("enclave_pubkey") or ""),
            expected_purpose=WEIGHT_PURPOSE,
            expected_epoch_id=expected_epoch_id,
            role="validator",
        )
        if not valid:
            return False, data
        if data.get("purpose") != WEIGHT_PURPOSE:
            return False, {**data, "error": "v2 attestation purpose is missing or incorrect"}
        if int(data.get("epoch_id")) != int(expected_epoch_id):
            return False, {**data, "error": "v2 attestation epoch binding is incorrect"}
        if data.get("pcr0_verification_mode") != "dynamic_github":
            return False, {**data, "error": "v2 requires dynamic Git-derived PCR0 verification"}
        if data.get("pcr0_commit") != receipt.get("commit_sha"):
            return False, {**data, "error": "v2 PCR0 commit does not match receipt commit"}
        return True, data
    except Exception as exc:
        logger.error("[ATTESTATION_V2] Verification failed: %s", exc)
        return False, {"error": str(exc)}


def verify_scoring_receipt_attestation_v2(receipt: Dict[str, Any]) -> tuple:
    """Verify AWS Nitro authenticity and exact bindings for one scoring receipt.

    The primary validator separately checks this PCR0 against an independently
    rebuilt gateway EIF before it creates the v2 weight receipt. This gateway
    check proves every supplied ancestor is a genuine Nitro receipt with exact
    role/purpose/epoch/key binding; it deliberately does not use the v1 static
    gateway allowlist as v2 proof.
    """

    purpose = str(receipt.get("purpose") or "")
    epoch_id = int(receipt.get("epoch_id", -1))
    if purpose not in SCORING_PURPOSES:
        return False, {"error": "v2 scoring receipt purpose is invalid"}
    try:
        from leadpoet_canonical.nitro import verify_nitro_attestation_full

        valid, data = verify_nitro_attestation_full(
            attestation_b64=str(receipt.get("attestation_document_b64") or ""),
            expected_pubkey=str(receipt.get("enclave_pubkey") or ""),
            expected_purpose=purpose,
            expected_epoch_id=epoch_id,
            role="gateway",
            skip_pcr0_verification=True,
        )
        if not valid:
            return False, data
        if data.get("purpose") != purpose:
            return False, {**data, "error": "v2 scoring attestation purpose is missing or incorrect"}
        if int(data.get("epoch_id", -1)) != epoch_id:
            return False, {**data, "error": "v2 scoring attestation epoch binding is incorrect"}
        if data.get("enclave_pubkey") != receipt.get("enclave_pubkey"):
            return False, {**data, "error": "v2 scoring attestation key binding is incorrect"}
        pcr0 = str(data.get("pcr0") or "").lower()
        if not _PCR0_RE.fullmatch(pcr0) or pcr0 == "0" * 96:
            return False, {**data, "error": "v2 scoring attestation PCR0 is invalid"}
        return True, data
    except Exception as exc:
        logger.error("[SCORING_ATTESTATION_V2] Verification failed: %s", exc)
        return False, {"error": str(exc)}


def verify_ed25519_signature(digest_bytes: bytes, signature_hex: str, pubkey_hex: str) -> bool:
    """
    Verify an Ed25519 signature over raw digest bytes.
    
    CANONICAL RULE: Ed25519 signatures are ALWAYS over SHA256 digest BYTES (32 bytes),
    never over hex strings. Transport signatures as hex for JSON compatibility.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        
        pk = Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
        pk.verify(bytes.fromhex(signature_hex), digest_bytes)
        return True
    except Exception as e:
        logger.error(f"[VERIFY] ❌ Ed25519 verification failed: {e}")
        return False


# ============================================================================
# Endpoints
# ============================================================================


@router.post("/inputs/v2")
async def get_weight_inputs_v2(
    authorization: WeightInputsV2Authorization,
) -> WeightInputsV2Response:
    """Return the complete measured gateway-owned input ancestry for one epoch."""

    try:
        request = validate_weight_inputs_request_v2(authorization.request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid V2 weight input request: {exc}") from exc
    validator_hotkey = request["validator_hotkey"]
    if not PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=503,
            detail="PRIMARY_VALIDATOR_HOTKEYS is required for authoritative v2",
        )
    if validator_hotkey not in PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(status_code=403, detail="Unauthorized validator hotkey")
    if ALLOWED_NETUIDS and request["netuid"] not in ALLOWED_NETUIDS:
        raise HTTPException(status_code=400, detail=f"Invalid netuid: {request['netuid']}")

    calculation = authorization.calculation_snapshot
    if sha256_json(calculation) != request["calculation_snapshot_hash"]:
        raise HTTPException(
            status_code=400,
            detail="V2 weight input request does not bind the calculation snapshot",
        )
    for field in ("netuid", "epoch_id", "block"):
        if calculation.get(field) != request[field]:
            raise HTTPException(
                status_code=400,
                detail=f"V2 weight input request differs from snapshot at {field}",
            )
    allocation_doc = calculation.get("research_lab_allocation_doc")
    if (
        not isinstance(allocation_doc, dict)
        or allocation_doc.get("allocation_hash") != request["allocation_hash"]
    ):
        raise HTTPException(
            status_code=400,
            detail="V2 weight input request differs from the Research Lab allocation",
        )
    message = weight_inputs_request_message_v2(request)
    if not await asyncio.to_thread(
        verify_wallet_signature,
        message,
        authorization.validator_hotkey_signature,
        validator_hotkey,
    ):
        raise HTTPException(status_code=403, detail="Invalid V2 weight input signature")

    subtensor = get_subtensor()
    gateway_block = await asyncio.to_thread(subtensor.get_current_block)
    if abs(gateway_block - request["block"]) > MAX_BLOCK_DRIFT:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Block drift too large: requested {request['block']}, "
                f"gateway sees {gateway_block}"
            ),
        )
    gateway_epoch = gateway_block // EPOCH_LENGTH
    valid_epochs = (
        {gateway_epoch, gateway_epoch - 1}
        if gateway_block % EPOCH_LENGTH < 30
        else {gateway_epoch}
    )
    if request["epoch_id"] not in valid_epochs:
        raise HTTPException(
            status_code=400,
            detail=f"epoch_id {request['epoch_id']} not in valid range {valid_epochs}",
        )
    if request["epoch_id"] != request["block"] // EPOCH_LENGTH:
        raise HTTPException(status_code=400, detail="epoch_id does not match requested block")

    try:
        from gateway.research_lab.attested_v2_store import (
            load_business_artifact_graph_v2,
        )
        from gateway.research_lab.attested_weight_inputs_v2 import (
            build_gateway_weight_inputs_v2,
        )

        allocation_graph = await load_business_artifact_graph_v2(
            artifact_kind="allocation",
            artifact_ref=f"epoch:{request['epoch_id']}",
            artifact_hash=request["allocation_hash"],
        )
        result = await build_gateway_weight_inputs_v2(
            calculation_snapshot=calculation,
            allocation_graph=allocation_graph,
            leaderboard_window_start=request["leaderboard_window_start"],
            leaderboard_window_end=request["leaderboard_window_end"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "authoritative_weight_inputs_v2_failed epoch=%s type=%s error=%s",
            request["epoch_id"],
            type(exc).__name__,
            str(exc)[:300],
        )
        raise HTTPException(
            status_code=503,
            detail="Authoritative V2 weight input reconstruction failed closed",
        ) from exc

    return WeightInputsV2Response(
        request_hash=request["request_hash"],
        calculation_snapshot_hash=request["calculation_snapshot_hash"],
        input_receipt_hashes=result["input_receipt_hashes"],
        gateway_authority_event_hash=result["gateway_authority_event_hash"],
        upstream_receipt_set=result["upstream_receipt_set"],
    )

@router.post("/submit/v2")
async def submit_weights_v2(submission: WeightSubmissionV2) -> WeightSubmissionV2Response:
    """Persist and publish the only authoritative V2 weight bundle."""

    try:
        verified, validator_boot = await asyncio.to_thread(
            _validate_authoritative_v2_submission,
            submission,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid authoritative v2 weight bundle: {exc}",
        ) from exc

    if not PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=503,
            detail="PRIMARY_VALIDATOR_HOTKEYS is required for authoritative v2",
        )
    if submission.validator_hotkey not in PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(status_code=403, detail="Unauthorized validator hotkey")
    if ALLOWED_NETUIDS and verified["netuid"] not in ALLOWED_NETUIDS:
        raise HTTPException(status_code=400, detail=f"Invalid netuid: {verified['netuid']}")

    parsed, binding_fields, _binding_error = parse_binding_message(
        submission.binding_message
    )
    if not parsed or not isinstance(binding_fields, dict):
        raise HTTPException(status_code=403, detail="Invalid v2 hotkey binding format")
    if binding_fields.get("version") != validator_boot["commit_sha"]:
        raise HTTPException(status_code=403, detail="V2 hotkey binding commit mismatch")
    if not verify_binding_message(
        submission.binding_message,
        submission.validator_hotkey_signature,
        submission.validator_hotkey,
        expected_netuid=verified["netuid"],
        expected_chain=EXPECTED_CHAIN,
        expected_enclave_pubkey=verified["validator_enclave_pubkey"],
        expected_code_hash=validator_boot["build_manifest_hash"],
    ):
        raise HTTPException(status_code=403, detail="Invalid v2 hotkey binding")

    from gateway.research_lab.attested_v2_store import (
        load_weight_bundle_v2,
        load_weight_publication_v2,
        persist_receipt_graph_v2,
        persist_weight_bundle_v2,
        persist_weight_publication_v2,
    )

    existing = await load_weight_bundle_v2(
        netuid=verified["netuid"],
        epoch_id=verified["epoch_id"],
        validator_hotkey=verified["validator_hotkey"],
    )
    if existing is not None:
        submitted_bundle = submission.model_dump(mode="python")
        if existing != submitted_bundle:
            raise HTTPException(
                status_code=409,
                detail="Different authoritative V2 weights already exist for this epoch",
            )
        try:
            existing_publication = await load_weight_publication_v2(
                bundle_hash=verified["bundle_hash"]
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Existing authoritative V2 publication failed verification",
            ) from exc
        if existing_publication is not None:
            return WeightSubmissionV2Response(
                success=True,
                epoch_id=verified["epoch_id"],
                weights_count=len(verified["uids"]),
                weights_hash=verified["weights_hash"],
                weight_receipt_hash=verified["weight_receipt_hash"],
                weight_submission_event_hash=existing_publication[
                    "weight_submission_event_hash"
                ],
                message="Authoritative V2 bundle was already durably published",
            )

    if existing is None:
        subtensor = get_subtensor()
        gateway_block = await asyncio.to_thread(subtensor.get_current_block)
        if abs(gateway_block - verified["block"]) > MAX_BLOCK_DRIFT:
            raise HTTPException(
                status_code=400,
                detail=f"Block drift too large: submitted {verified['block']}, gateway sees {gateway_block}",
            )
        gateway_epoch = gateway_block // EPOCH_LENGTH
        valid_epochs = (
            {gateway_epoch, gateway_epoch - 1}
            if gateway_block % EPOCH_LENGTH < 30
            else {gateway_epoch}
        )
        if verified["epoch_id"] not in valid_epochs:
            raise HTTPException(
                status_code=400,
                detail=f"epoch_id {verified['epoch_id']} not in valid range {valid_epochs}",
            )
        if verified["epoch_id"] != verified["block"] // EPOCH_LENGTH:
            raise HTTPException(status_code=400, detail="epoch_id does not match submitted block")
        if verified["block"] % EPOCH_LENGTH < WEIGHT_SUBMISSION_BLOCK - 15:
            raise HTTPException(status_code=400, detail="Block is too early for weight submission")
    try:
        bundle_result = await persist_weight_bundle_v2(
            submission.model_dump(mode="python")
        )
        from gateway.utils.logger import log_event

        transparency = await log_event(
            "WEIGHT_SUBMISSION_V2",
            {
                "actor_hotkey": verified["validator_hotkey"],
                "netuid": verified["netuid"],
                "epoch_id": verified["epoch_id"],
                "block": verified["block"],
                "weights_hash": verified["weights_hash"],
                "bundle_hash": verified["bundle_hash"],
                "root_receipt_hash": verified["root_receipt_hash"],
            },
        )
        raw_transparency_hash = str(transparency.get("event_hash") or "").lower()
        transparency_hash = (
            raw_transparency_hash
            if raw_transparency_hash.startswith("sha256:")
            else "sha256:" + raw_transparency_hash
        )
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", transparency_hash):
            raise RuntimeError("signed transparency event hash is invalid")
        from gateway.research_lab.attested_coordinator_v2 import (
            execute_coordinator_v2,
        )
        from gateway.tee.coordinator_executor_v2 import (
            OP_ATTEST_WEIGHT_PUBLICATION,
        )

        publication = await execute_coordinator_v2(
            operation=OP_ATTEST_WEIGHT_PUBLICATION,
            purpose="gateway.weights.publication.v2",
            epoch_id=verified["epoch_id"],
            sequence=0,
            payload={
                "bundle_hash": bundle_result["bundle_hash"],
                "root_receipt_hash": bundle_result["root_receipt_hash"],
                "durable_readback_hash": bundle_result[
                    "durable_readback_hash"
                ],
                "transparency_event_hash": transparency_hash,
            },
            parent_graphs=(submission.receipt_graph,),
            input_artifact_hashes=(
                bundle_result["bundle_hash"],
                bundle_result["durable_readback_hash"],
                transparency_hash,
            ),
            persist_graph=persist_receipt_graph_v2,
        )
        publication_result = await persist_weight_publication_v2(
            bundle_result=bundle_result,
            publication_graph=publication["receipt_graph"],
            publication_doc=publication["result"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "authoritative_weight_submission_v2_failed epoch=%s type=%s error=%s",
            verified["epoch_id"],
            type(exc).__name__,
            str(exc)[:300],
        )
        raise HTTPException(
            status_code=503,
            detail="Authoritative V2 persistence/publication failed closed",
        ) from exc

    logger.info(
        "weight_submission_v2_authoritative epoch=%s hash=%s receipt=%s event=%s",
        verified["epoch_id"],
        verified["weights_hash"],
        verified["weight_receipt_hash"],
        publication_result["weight_submission_event_hash"],
    )
    return WeightSubmissionV2Response(
        success=True,
        epoch_id=verified["epoch_id"],
        weights_count=len(verified["uids"]),
        weights_hash=verified["weights_hash"],
        weight_receipt_hash=verified["weight_receipt_hash"],
        weight_submission_event_hash=publication_result[
            "weight_submission_event_hash"
        ],
        message="Authoritative V2 bundle durably published",
    )


@router.post("/finalize/v2")
async def finalize_weights_v2(
    submission: WeightFinalizationV2,
) -> WeightFinalizationV2Response:
    """Persist only enclave-proven finalized inclusion and chain state change."""

    payload = submission.model_dump(mode="python")
    try:
        verified = await asyncio.to_thread(
            validate_weight_finalization_submission_v2,
            payload,
        )
        from leadpoet_canonical.attested_v2 import validate_receipt_graph

        await asyncio.to_thread(
            validate_receipt_graph,
            payload["receipt_graph"],
            required_purposes={
                "validator.weights.computed.v2",
                "validator.set_weights_extrinsic.v2",
                "validator.weights.finalized.v2",
            },
            boot_attestation_verifier=_verify_authoritative_v2_boot,
            require_boot_attestation_verification=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid authoritative v2 weight finalization: {exc}",
        ) from exc
    if not PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(
            status_code=503,
            detail="PRIMARY_VALIDATOR_HOTKEYS is required for authoritative v2",
        )
    if verified["validator_hotkey"] not in PRIMARY_VALIDATOR_HOTKEYS:
        raise HTTPException(status_code=403, detail="Unauthorized validator hotkey")
    if ALLOWED_NETUIDS and verified["netuid"] not in ALLOWED_NETUIDS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid netuid: {verified['netuid']}",
        )
    try:
        from gateway.research_lab.attested_v2_store import (
            persist_weight_finalization_v2,
        )

        result = await persist_weight_finalization_v2(submission=payload)
    except Exception as exc:
        logger.error(
            "authoritative_weight_finalization_v2_failed epoch=%s type=%s error=%s",
            verified["epoch_id"],
            type(exc).__name__,
            str(exc)[:300],
        )
        raise HTTPException(
            status_code=503,
            detail="Authoritative V2 finalization persistence failed closed",
        ) from exc
    return WeightFinalizationV2Response(
        success=True,
        epoch_id=verified["epoch_id"],
        weights_hash=verified["weights_hash"],
        extrinsic_hash=verified["extrinsic_hash"],
        finalized_block=verified["finalized_block"],
        weight_submission_event_hash=verified["weight_submission_event_hash"],
        weight_finalization_event_hash=result[
            "weight_finalization_event_hash"
        ],
        message="Authoritative V2 finalized chain state durably published",
    )


@router.get("/v2/latest/{netuid}/{epoch_id}")
async def get_attested_weights_v2(netuid: int, epoch_id: int) -> Dict[str, Any]:
    """Return only complete V2 authority with finalized-chain evidence."""

    try:
        from gateway.research_lab.attested_v2_store import load_weight_authority_v2

        if len(PRIMARY_VALIDATOR_HOTKEYS) != 1:
            raise RuntimeError("authoritative primary validator hotkey is ambiguous")
        authority = await load_weight_authority_v2(
            netuid=int(netuid),
            epoch_id=int(epoch_id),
            validator_hotkey=next(iter(PRIMARY_VALIDATOR_HOTKEYS)),
        )
    except Exception as exc:
        logger.error(
            "weight_submission_v2_sidecar_load_failed netuid=%s epoch=%s "
            "error_type=%s error=%s",
            netuid,
            epoch_id,
            type(exc).__name__,
            str(exc)[:240],
        )
        raise HTTPException(status_code=500, detail="v2 weight sidecar verification failed") from exc
    if authority is None:
        raise HTTPException(
            status_code=404,
            detail="finalized v2 weight authority not found",
        )
    return authority

@router.post("/submit")
async def submit_weights(submission: WeightSubmission) -> WeightSubmissionResponse:
    """Reject retired V1 writes while preserving historical read endpoints."""

    raise HTTPException(
        status_code=410,
        detail="V1 weight submission is retired; use /weights/submit/v2",
    )

    # Retained temporarily as unreachable historical decoding code so existing
    # V1 rows remain explainable during the V2 migration audit. No runtime
    # configuration can enter this block.
    from gateway.db.client import get_read_client, get_write_client
    from gateway.utils.logger import log_event

    if len(submission.uids) != len(submission.weights_u16):
        raise HTTPException(status_code=400, detail="uids/weights_u16 length mismatch")
    if not submission.uids:
        raise HTTPException(status_code=400, detail="weight submission is empty")
    if submission.uids != sorted(submission.uids):
        raise HTTPException(status_code=400, detail="uids must be sorted ascending")
    if len(submission.uids) != len(set(submission.uids)):
        raise HTTPException(status_code=400, detail="duplicate uids detected")
    if any(not 1 <= int(weight) <= 65535 for weight in submission.weights_u16):
        raise HTTPException(
            status_code=400,
            detail="sparse weights must all be in [1, 65535]",
        )
    if ALLOWED_NETUIDS and submission.netuid not in ALLOWED_NETUIDS:
        raise HTTPException(status_code=400, detail="invalid netuid")
    if (
        PRIMARY_VALIDATOR_HOTKEYS
        and submission.validator_hotkey not in PRIMARY_VALIDATOR_HOTKEYS
    ):
        raise HTTPException(status_code=403, detail="unauthorized validator hotkey")

    def _existing_bundle() -> bool:
        result = (
            get_read_client()
            .table("published_weight_bundles")
            .select("id")
            .eq("netuid", submission.netuid)
            .eq("epoch_id", submission.epoch_id)
            .eq("validator_hotkey", submission.validator_hotkey)
            .limit(1)
            .execute()
        )
        return bool(result.data)

    if await asyncio.to_thread(_existing_bundle):
        raise HTTPException(status_code=409, detail="duplicate submission for this epoch")

    subtensor = await asyncio.to_thread(get_subtensor)
    gateway_block = int(await asyncio.to_thread(subtensor.get_current_block))
    gateway_epoch = gateway_block // EPOCH_LENGTH
    if abs(gateway_block - submission.block) > MAX_BLOCK_DRIFT:
        raise HTTPException(status_code=400, detail="block drift is too large")
    valid_epochs = (
        {gateway_epoch, gateway_epoch - 1}
        if gateway_block % EPOCH_LENGTH < 30
        else {gateway_epoch}
    )
    if submission.epoch_id not in valid_epochs:
        raise HTTPException(status_code=400, detail="epoch is outside the valid range")
    if submission.epoch_id != submission.block // EPOCH_LENGTH:
        raise HTTPException(status_code=400, detail="epoch does not match block")
    if submission.block % EPOCH_LENGTH < WEIGHT_SUBMISSION_BLOCK - 15:
        raise HTTPException(status_code=400, detail="weight submission is too early")

    attestation_valid, attestation_data = await asyncio.to_thread(
        verify_validator_attestation,
        submission.validator_attestation_b64,
        submission.validator_enclave_pubkey,
        submission.epoch_id,
    )
    if not attestation_valid:
        raise HTTPException(
            status_code=403,
            detail=(
                "invalid validator attestation: "
                + str(attestation_data.get("error") or "unknown")[:240]
            ),
        )
    verified_pcr0 = str(attestation_data.get("pcr0") or "")
    pcr0_commit_hash = str(attestation_data.get("pcr0_commit") or "").strip()
    if not pcr0_commit_hash:
        pcr0_commit_hash = str(
            await asyncio.to_thread(
                _lookup_pcr0_commit_hash_from_allowlist,
                verified_pcr0,
            )
            or ""
        )
    if not pcr0_commit_hash:
        raise HTTPException(
            status_code=500,
            detail="verified validator PCR0 has no auditable commit",
        )

    binding_valid = await asyncio.to_thread(
        verify_binding_message,
        submission.binding_message,
        submission.validator_hotkey_signature,
        submission.validator_hotkey,
        expected_netuid=submission.netuid,
        expected_chain=EXPECTED_CHAIN,
        expected_enclave_pubkey=submission.validator_enclave_pubkey,
        expected_code_hash=submission.validator_code_hash,
    )
    if not binding_valid:
        raise HTTPException(status_code=403, detail="invalid hotkey binding")
    try:
        digest_bytes = bytes.fromhex(submission.weights_hash)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="weights_hash is not hex") from exc
    signature_valid = await asyncio.to_thread(
        verify_ed25519_signature,
        digest_bytes,
        submission.validator_signature,
        submission.validator_enclave_pubkey,
    )
    if not signature_valid:
        raise HTTPException(status_code=401, detail="invalid validator signature")
    recomputed_hash = bundle_weights_hash(
        submission.netuid,
        submission.epoch_id,
        submission.block,
        list(zip(submission.uids, submission.weights_u16)),
    )
    if recomputed_hash != submission.weights_hash:
        raise HTTPException(status_code=400, detail="weights_hash does not match payload")

    # Annotation must stay 3.9-compatible: nested-def annotations are
    # evaluated on every call, and `int | None` needs Python 3.10.
    def _chain_snapshot() -> tuple[Optional[int], Optional[str]]:
        try:
            metagraph = subtensor.metagraph(netuid=submission.netuid)
            primary_uid = next(
                (
                    uid
                    for uid, hotkey in enumerate(metagraph.hotkeys)
                    if hotkey == submission.validator_hotkey
                ),
                None,
            )
            if primary_uid is None:
                return None, None
            snapshot_block = int(subtensor.get_current_block())
            primary_weights = next(
                (
                    weights
                    for uid, weights in subtensor.weights(netuid=submission.netuid)
                    if uid == primary_uid
                ),
                None,
            )
            if not primary_weights:
                return snapshot_block, None
            compare_hash = compare_weights_hash(
                submission.netuid,
                submission.epoch_id,
                normalize_chain_weights(primary_weights),
            )
            return snapshot_block, compare_hash
        except Exception as exc:  # snapshot is audit enrichment, not authority
            logger.warning("legacy_v1_chain_snapshot_failed error=%s", str(exc)[:240])
            return None, None

    chain_snapshot_block, chain_snapshot_compare_hash = await asyncio.to_thread(
        _chain_snapshot
    )
    submission_payload = {
        "actor_hotkey": submission.validator_hotkey,
        "validator_signature": submission.validator_signature,
        "epoch_id": submission.epoch_id,
        "netuid": submission.netuid,
        "block": submission.block,
        "weights_hash": submission.weights_hash,
        "weights_count": len(submission.uids),
        "chain_snapshot_block": chain_snapshot_block,
        "chain_snapshot_compare_hash": chain_snapshot_compare_hash,
    }
    if BITTENSOR_NETWORK == "test":
        weight_submission_event_hash = (
            f"TESTNET_MOCK_{submission.epoch_id}_{submission.netuid}"
        )
    else:
        log_entry = await log_event("WEIGHT_SUBMISSION", submission_payload)
        weight_submission_event_hash = str(log_entry.get("event_hash") or "")
        if not weight_submission_event_hash:
            raise HTTPException(
                status_code=500,
                detail="weight submission transparency event was not persisted",
            )

    bundle_data = {
        "netuid": submission.netuid,
        "epoch_id": submission.epoch_id,
        "block": submission.block,
        "uids": submission.uids,
        "weights_u16": submission.weights_u16,
        "weights_hash": submission.weights_hash,
        "validator_hotkey": submission.validator_hotkey,
        "validator_enclave_pubkey": submission.validator_enclave_pubkey,
        "validator_signature": submission.validator_signature,
        "validator_attestation_b64": submission.validator_attestation_b64,
        "validator_code_hash": submission.validator_code_hash,
        "validator_pcr0": verified_pcr0,
        "pcr0_commit_hash": pcr0_commit_hash,
        "chain_snapshot_block": chain_snapshot_block,
        "chain_snapshot_compare_hash": chain_snapshot_compare_hash,
        "weight_submission_event_hash": weight_submission_event_hash,
    }

    def _insert_bundle() -> None:
        get_write_client().table("published_weight_bundles").insert(
            bundle_data
        ).execute()

    if BITTENSOR_NETWORK != "test":
        try:
            await asyncio.to_thread(_insert_bundle)
        except Exception as exc:
            if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
                raise HTTPException(
                    status_code=409,
                    detail="duplicate submission (concurrent race)",
                ) from exc
            raise

    research_lab_config = ResearchLabGatewayConfig.from_env()
    audit_enabled = (
        research_lab_config.arweave_audit_shadow_enabled
        if BITTENSOR_NETWORK == "test"
        else research_lab_config.arweave_audit_enabled
    )
    if audit_enabled:
        try:
            await publish_research_lab_epoch_audit(
                epoch=submission.epoch_id,
                netuid=submission.netuid,
                audit_kind=("shadow" if BITTENSOR_NETWORK == "test" else "active"),
                actor_hotkey=submission.validator_hotkey,
                weight_bundle=bundle_data,
                config=research_lab_config,
            )
        except Exception as exc:
            logger.warning(
                "legacy_v1_research_lab_arweave_publish_failed error=%s",
                str(exc)[:240],
            )
    return WeightSubmissionResponse(
        success=True,
        epoch_id=submission.epoch_id,
        weights_count=len(submission.uids),
        message="Weights accepted and logged",
        weight_submission_event_hash=weight_submission_event_hash,
    )


@router.get("/latest/{netuid}/{epoch_id}")
async def get_latest_weights(netuid: int, epoch_id: int) -> dict:
    """
    Get the latest published weights bundle for a specific netuid and epoch.
    
    USES ANON KEY FOR READS (not service role).
    Returns the complete bundle needed for auditor verification.
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    result = read_client.table("published_weight_bundles") \
        .select("*") \
        .eq("netuid", netuid) \
        .eq("epoch_id", epoch_id) \
        .order("created_at", desc=True) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No weights for epoch {epoch_id}")
    
    bundle = result.data[0]
    
    return {
        "netuid": bundle["netuid"],
        "epoch_id": bundle["epoch_id"],
        "block": bundle["block"],
        "uids": bundle["uids"],
        "weights_u16": bundle["weights_u16"],
        "weights_hash": bundle["weights_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "validator_enclave_pubkey": bundle["validator_enclave_pubkey"],
        "validator_signature": bundle["validator_signature"],
        "validator_attestation_b64": bundle["validator_attestation_b64"],
        "validator_code_hash": bundle["validator_code_hash"],
        "validator_pcr0": bundle.get("validator_pcr0"),  # For auditor PCR0 verification
        "pcr0_commit_hash": bundle.get("pcr0_commit_hash"),  # Git commit - auditors verify exists
        "chain_snapshot_block": bundle.get("chain_snapshot_block"),
        "chain_snapshot_compare_hash": bundle.get("chain_snapshot_compare_hash"),
        "weight_submission_event_hash": bundle.get("weight_submission_event_hash"),
    }


@router.get("/current/{netuid}")
async def get_current_weights(netuid: int) -> dict:
    """
    Get the most recently published weights bundle for a netuid.
    
    Use this to find the latest epoch without knowing the epoch_id.
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    result = read_client.table("published_weight_bundles") \
        .select("*") \
        .eq("netuid", netuid) \
        .order("epoch_id", desc=True) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail=f"No weights for netuid {netuid}")
    
    bundle = result.data[0]
    
    return {
        "netuid": bundle["netuid"],
        "epoch_id": bundle["epoch_id"],
        "block": bundle["block"],
        "uids": bundle["uids"],
        "weights_u16": bundle["weights_u16"],
        "weights_hash": bundle["weights_hash"],
        "validator_hotkey": bundle["validator_hotkey"],
        "validator_enclave_pubkey": bundle["validator_enclave_pubkey"],
        "validator_signature": bundle["validator_signature"],
        "validator_attestation_b64": bundle["validator_attestation_b64"],
        "validator_code_hash": bundle["validator_code_hash"],
        "validator_pcr0": bundle.get("validator_pcr0"),  # For auditor PCR0 verification
        "pcr0_commit_hash": bundle.get("pcr0_commit_hash"),  # Git commit - auditors verify exists
        "chain_snapshot_block": bundle.get("chain_snapshot_block"),
        "chain_snapshot_compare_hash": bundle.get("chain_snapshot_compare_hash"),
        "weight_submission_event_hash": bundle.get("weight_submission_event_hash"),
    }


# ============================================================================
# Transparency Log Endpoints (for auditor verification)
# ============================================================================

@router.get("/transparency/event/{event_hash}")
async def get_transparency_event(event_hash: str) -> dict:
    """
    Fetch a signed event from the transparency log by its event_hash.
    
    CRITICAL FOR AUDITORS: This endpoint allows independent verification of:
    - Event authenticity (verify enclave signature)
    - Hash-chain integrity (check prev_event_hash links)
    - Event contents (access signed payload)
    
    The returned log_entry contains:
    - signed_event: {event_type, timestamp, boot_id, monotonic_seq, prev_event_hash, payload}
    - event_hash: SHA256 of canonical signed_event
    - enclave_pubkey: Gateway pubkey that signed this event
    - enclave_signature: Ed25519 signature over event_hash
    
    Auditors MUST:
    1. Verify enclave_pubkey matches attested gateway pubkey
    2. Recompute event_hash from signed_event
    3. Verify enclave_signature over event_hash
    4. Check prev_event_hash for chain continuity
    
    Returns:
        Full log_entry object for verification
        
    Raises:
        404: Event not found
    """
    from gateway.db.client import get_read_client
    
    read_client = get_read_client()
    
    # Query by event_hash (unique identifier)
    result = read_client.table("transparency_log") \
        .select("payload") \
        .eq("event_hash", event_hash) \
        .limit(1) \
        .execute()
    
    if not result.data:
        raise HTTPException(
            status_code=404, 
            detail=f"Event not found: {event_hash[:16]}..."
        )
    
    # The payload column contains the full log_entry (signed_event + signature)
    log_entry = result.data[0]["payload"]
    
    # Validate structure before returning
    if not log_entry or not isinstance(log_entry, dict):
        raise HTTPException(
            status_code=500,
            detail="Invalid log entry format in database"
        )
    
    # For old (legacy) events, payload might not have signed_event structure
    # Return as-is and let auditor handle format differences
    return log_entry


@router.get("/transparency/events/range")
async def get_transparency_events_range(
    start_seq: int = 0,
    limit: int = 100,
    boot_id: Optional[str] = None,
) -> dict:
    """
    Fetch a range of events from the transparency log for chain verification.
    
    This endpoint supports auditors who need to verify hash-chain continuity
    across multiple events. Events are returned in monotonic_seq order.
    
    Args:
        start_seq: Starting monotonic sequence number (inclusive)
        limit: Maximum events to return (max 1000)
        boot_id: Optional filter by boot session
        
    Returns:
        {
            "events": [log_entry, ...],
            "count": int,
            "has_more": bool
        }
    """
    from gateway.db.client import get_read_client
    
    # Cap limit to prevent abuse
    limit = min(limit, 1000)
    
    read_client = get_read_client()
    
    query = read_client.table("transparency_log") \
        .select("payload, event_hash, monotonic_seq, boot_id") \
        .gte("monotonic_seq", start_seq) \
        .order("monotonic_seq", desc=False) \
        .limit(limit + 1)  # +1 to check if there's more
    
    if boot_id:
        query = query.eq("boot_id", boot_id)
    
    result = query.execute()
    
    events = result.data[:limit] if result.data else []
    has_more = len(result.data) > limit if result.data else False
    
    return {
        "events": [
            {
                "event_hash": e.get("event_hash"),
                "monotonic_seq": e.get("monotonic_seq"),
                "boot_id": e.get("boot_id"),
                "log_entry": e.get("payload"),
            }
            for e in events
        ],
        "count": len(events),
        "has_more": has_more,
    }

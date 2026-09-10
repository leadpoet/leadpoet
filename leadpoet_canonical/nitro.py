"""
LeadPoet Canonical AWS Nitro Attestation Verification

This module provides AWS Nitro Enclave attestation verification for the TEE system.

TRUST MODEL:
- PCR0 (enclave image hash) is the ROOT OF TRUST
- user_data.code_hash is INFORMATIONAL ONLY (do NOT use as root of trust)
- A malicious enclave can report ANY code_hash it wants in user_data
- ONLY PCR0 from the attestation document proves code integrity

VERIFICATION ORDER (MANDATORY):
1. Parse COSE_Sign1 structure (CBOR-encoded)
2. Verify certificate chain to Amazon Nitro root (PINNED)
3. Verify COSE signature using public key from certificate
4. Extract PCR0 from attestation body (NOT user_data)
5. COMPARE PCR0 against PINNED expected value ← ROOT OF TRUST
6. ONLY THEN trust user_data contents (pubkey, purpose, epoch_id)

FAIL-CLOSED: This module returns False (or raises) on ANY verification failure.
"""

import base64
import json
import logging
import os
import time
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, List

from leadpoet_canonical.constants import TRUST_LEVEL_FULL_NITRO, TRUST_LEVEL_SIGNATURE_ONLY

logger = logging.getLogger(__name__)


class AttestationError(Exception):
    """Raised when attestation verification fails."""
    pass


def _get_cert_not_valid_before(cert) -> datetime:
    """Get certificate not_valid_before, compatible with old and new cryptography versions."""
    # cryptography >= 42.0.0 uses not_valid_before_utc
    # cryptography < 42.0.0 uses not_valid_before
    if hasattr(cert, 'not_valid_before_utc'):
        return cert.not_valid_before_utc
    else:
        # Old version returns naive datetime, make it UTC-aware
        nvb = cert.not_valid_before
        if nvb.tzinfo is None:
            return nvb.replace(tzinfo=timezone.utc)
        return nvb


def _get_cert_not_valid_after(cert) -> datetime:
    """Get certificate not_valid_after, compatible with old and new cryptography versions."""
    if hasattr(cert, 'not_valid_after_utc'):
        return cert.not_valid_after_utc
    else:
        nva = cert.not_valid_after
        if nva.tzinfo is None:
            return nva.replace(tzinfo=timezone.utc)
        return nva


def _attestation_certificate_validation_time(att_doc: Dict[str, Any]) -> datetime:
    """Return the signed Nitro document timestamp as an aware UTC datetime."""
    timestamp_ms = att_doc.get("timestamp")
    if isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, int):
        raise AttestationError("Attestation timestamp is missing or invalid")
    if timestamp_ms <= 0:
        raise AttestationError("Attestation timestamp is missing or invalid")
    try:
        return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc)
    except (OverflowError, OSError, ValueError) as exc:
        raise AttestationError("Attestation timestamp is outside the supported range") from exc


def _verify_leaf_certificate_validity(cert, validation_time: datetime) -> None:
    """Require a leaf certificate to be valid at one explicit UTC instant."""
    if validation_time.tzinfo is None:
        raise AttestationError("Certificate validation time must be timezone-aware")
    cert_not_before = _get_cert_not_valid_before(cert)
    cert_not_after = _get_cert_not_valid_after(cert)
    if validation_time < cert_not_before:
        raise AttestationError(f"Certificate not yet valid (starts {cert_not_before})")
    if validation_time > cert_not_after:
        raise AttestationError(f"Certificate expired ({cert_not_after})")


# =============================================================================
# PINNED VALUES - PRODUCTION CONFIGURATION
# =============================================================================

# Amazon Nitro root certificate (DER format)
# Source: https://aws-nitro-enclaves.amazonaws.com/AWS_NitroEnclaves_Root-G1.zip
# This is PINNED (not fetched at runtime) to prevent MITM attacks
# Certificate valid: 2019-10-28 to 2049-10-28
# Size: 533 bytes
NITRO_ROOT_CERT_DER: bytes = bytes.fromhex(
    "3082021130820196a003020102021100f93175681b90afe11d46ccb4e4e7f856"
    "300a06082a8648ce3d0403033049310b3009060355040613025553310f300d06"
    "0355040a0c06416d617a6f6e310c300a060355040b0c03415753311b30190603"
    "5504030c126177732e6e6974726f2d656e636c61766573301e170d3139313032"
    "383133323830355a170d3439313032383134323830355a3049310b3009060355"
    "040613025553310f300d060355040a0c06416d617a6f6e310c300a060355040b"
    "0c03415753311b301906035504030c126177732e6e6974726f2d656e636c6176"
    "65733076301006072a8648ce3d020106052b8104002203620004fc0254eba608"
    "c1f36870e29ada90be46383292736e894bfff672d989444b5051e534a4b1f6db"
    "e3c0bc581a32b7b176070ede12d69a3fea211b66e752cf7dd1dd095f6f1370f4"
    "170843d9dc100121e4cf63012809664487c9796284304dc53ff4a3423040300f"
    "0603551d130101ff040530030101ff301d0603551d0e041604149025b50dd905"
    "47e796c396fa729dcf99a9df4b96300e0603551d0f0101ff040403020186300a"
    "06082a8648ce3d0403030369003066023100a37f2f91a1c9bd5ee7b8627c1698"
    "d255038e1f0343f95b63a9628c3d39809545a11ebcbf2e3b55d8aeee71b4c3d6"
    "adf3023100a2f39b1605b27028a5dd4ba069b5016e65b4fbde8fe0061d6a5319"
    "7f9cdaf5d943bc61fc2beb03cb6fee8d2302f3dff6"
)

# =============================================================================
# PCR0 ALLOWLIST - FETCHED FROM GITHUB (AUTO-UPDATED)
# =============================================================================

# GitHub raw URL for the allowlist file
# This is the source of truth for allowed PCR0 values
PCR0_ALLOWLIST_URL = os.environ.get(
    "PCR0_ALLOWLIST_URL",
    # Repo was renamed LeadPoet/Bittensor-subnet -> leadpoet/leadpoet; the old URL
    # 404s, which silently dropped the gateway to the (stale) static fallback and
    # rejected every validator weight-bundle publish. Point at the live repo.
    "https://raw.githubusercontent.com/leadpoet/leadpoet/main/pcr0_allowlist.json"
)

# Cache TTL in seconds (default: 5 minutes)
PCR0_CACHE_TTL_SECONDS = int(os.environ.get("PCR0_CACHE_TTL_SECONDS", "300"))

# Thread-safe cache for PCR0 allowlist
_pcr0_cache: Dict[str, Any] = {
    "gateway_pcr0": [],
    "last_fetch": 0,
    "fetch_error": None,
}
_pcr0_cache_lock = threading.Lock()
_pcr0_cache_condition = threading.Condition(_pcr0_cache_lock)
_pcr0_refresh_in_flight = False

# Fallback values (used if GitHub fetch fails on first attempt)
# These should match the initial values in pcr0_allowlist.json
FALLBACK_GATEWAY_PCR0_VALUES: List[str] = [
    "02797d0a3b02fdda186db756b7cae6ef283592bae6ea879c0c19e4ab0a787766bbbd2008eb49eb9de58f7346d6c834d5",
]

def _fetch_pcr0_allowlist_from_github() -> Dict[str, List[str]]:
    """
    Fetch PCR0 allowlist from GitHub.
    
    Returns:
        Dictionary with the gateway PCR0 list
        
    Raises:
        Exception on fetch failure
    """
    import urllib.request
    import urllib.error
    
    try:
        # Fetch with timeout
        request = urllib.request.Request(
            PCR0_ALLOWLIST_URL,
            headers={"User-Agent": "LeadPoet-Gateway/1.0"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
        
        # Extract PCR0 values from the structured format
        gateway_pcr0 = [item["pcr0"] for item in data.get("gateway_pcr0", [])]
        logger.info(f"[PCR0] Fetched allowlist from GitHub: {len(gateway_pcr0)} gateway PCR0 values")
        return {"gateway_pcr0": gateway_pcr0}
        
    except urllib.error.URLError as e:
        raise Exception(f"Failed to fetch PCR0 allowlist from GitHub: {e}")
    except json.JSONDecodeError as e:
        raise Exception(f"Invalid JSON in PCR0 allowlist: {e}")
    except KeyError as e:
        raise Exception(f"Invalid PCR0 allowlist format: missing key {e}")


def _refresh_pcr0_cache_if_needed() -> None:
    """
    Refresh the PCR0 cache if TTL has expired.
    Thread-safe implementation.
    """
    global _pcr0_refresh_in_flight
    
    current_time = time.time()
    
    with _pcr0_cache_condition:
        if _pcr0_refresh_in_flight:
            # A previous allowlist is safe to use during a refresh. On the
            # first process fetch there is no previous authority, so wait for
            # the fetch (or pinned fallback) instead of returning an empty
            # allowlist and spuriously rejecting valid attestations.
            if (
                _pcr0_cache["gateway_pcr0"]
            ):
                return
            while _pcr0_refresh_in_flight:
                _pcr0_cache_condition.wait()
            return
        # Check freshness after in-flight state. The refresh owner reserves
        # last_fetch before network I/O, so checking TTL first would let a
        # concurrent first-process verifier return an empty allowlist.
        if current_time - _pcr0_cache["last_fetch"] < PCR0_CACHE_TTL_SECONDS:
            return  # Cache is still valid

        # Reserve the refresh window BEFORE fetching. The fetch used to run
        # while holding this lock and, on failure with a non-empty cache,
        # last_fetch was never advanced — so with GitHub slow or down every
        # PCR0 verification re-attempted the fetch and serialized behind a
        # 10-second urlopen. Reserving first makes the refresh single-flight
        # (concurrent verifiers keep using the previous allowlist instead of
        # stalling) and turns a failed fetch into a TTL-length backoff. The
        # allowlist values themselves only ever change under the lock, and a
        # PCR0 mismatch still fails closed downstream exactly as before.
        _pcr0_cache["last_fetch"] = current_time
        _pcr0_refresh_in_flight = True

    try:
        # Fetch new allowlist (network I/O outside the lock)
        allowlist = _fetch_pcr0_allowlist_from_github()
    except Exception as e:
        logger.warning(f"[PCR0] Failed to refresh allowlist from GitHub: {e}")
        with _pcr0_cache_condition:
            try:
                _pcr0_cache["fetch_error"] = str(e)

                # If this is the first fetch (cache is empty), use fallback values
                if not _pcr0_cache["gateway_pcr0"]:
                    logger.warning("[PCR0] Using fallback PCR0 values")
                    _pcr0_cache["gateway_pcr0"] = FALLBACK_GATEWAY_PCR0_VALUES.copy()
            finally:
                _pcr0_refresh_in_flight = False
                _pcr0_cache_condition.notify_all()
        return
    except BaseException:
        with _pcr0_cache_condition:
            _pcr0_refresh_in_flight = False
            _pcr0_cache_condition.notify_all()
        raise

    with _pcr0_cache_condition:
        try:
            _pcr0_cache["gateway_pcr0"] = allowlist["gateway_pcr0"]
            _pcr0_cache["last_fetch"] = time.time()
            _pcr0_cache["fetch_error"] = None
        finally:
            _pcr0_refresh_in_flight = False
            _pcr0_cache_condition.notify_all()


def get_allowed_gateway_pcr0() -> List[str]:
    """
    Get the list of allowed gateway PCR0 values.
    Automatically refreshes from GitHub if cache is stale.
    
    Returns:
        List of allowed PCR0 hex strings
    """
    _refresh_pcr0_cache_if_needed()
    with _pcr0_cache_lock:
        return _pcr0_cache["gateway_pcr0"].copy()





# =============================================================================
# FULL NITRO ATTESTATION VERIFICATION
# =============================================================================

def verify_nitro_attestation_full(
    attestation_b64: str,
    expected_pcr0: str = None,
    expected_pubkey: str = None,
    expected_purpose: str = None,
    skip_pcr0_verification: bool = False,  # For callers that verify PCR policy separately
    certificate_validity_at_attestation_time: bool = False,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Full AWS Nitro attestation verification.
    
    This function performs cryptographic verification of AWS Nitro Enclave
    attestation documents, proving that code is running in a genuine AWS
    Nitro Enclave with the expected PCR0 measurement.
    
    Args:
        attestation_b64: Base64-encoded Nitro attestation document
        expected_pcr0: Expected PCR0 value (hex string). If None, checks against
                      ALLOWED_GATEWAY_PCR0_VALUES or ALLOWED_VALIDATOR_PCR0_VALUES
        expected_pubkey: Expected enclave public key (hex string). If None, skips check.
        expected_purpose: Expected purpose in user_data. If None, purpose check is skipped.
        skip_pcr0_verification: If True, skip PCR0 verification against allowlist/GitHub.
                               STILL verifies AWS cert chain + COSE signature (proves REAL
                               Nitro enclave), but doesn't verify the specific CODE.
                               Use when the caller will separately
                               verify PCR0. Trust level will be "aws_verified".
        certificate_validity_at_attestation_time: Validate the leaf certificate at
                               the signed Nitro document timestamp. This is only for
                               replaying immutable historical receipt graphs; live
                               boot verification must retain the default current-time
                               check.
                          
    Returns:
        Tuple of (success, extracted_data)
        - success: True if all verification steps pass
        - extracted_data: Dictionary with extracted values and verification status
        
    Security Notes:
        - PCR0 is the ROOT OF TRUST, NOT user_data.code_hash
        - Certificate chain MUST be verified to pinned Amazon Nitro root
        - Epoch binding prevents replay of old validator attestations
        - With skip_pcr0_verification=True, you prove it's a REAL Nitro enclave
          but NOT that it's running specific code. Auditors should note this.
    """
    try:
        import cbor2
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.exceptions import InvalidSignature
    except ImportError as e:
        return False, {"error": f"Missing required library: {e}. Install with: pip install cbor2 cryptography"}
    
    result = {
        "trust_level": TRUST_LEVEL_FULL_NITRO,
        "verification_steps": [],
    }
    
    try:
        # =================================================================
        # Step 1: Decode attestation document
        # =================================================================
        try:
            att_bytes = base64.b64decode(attestation_b64)
            result["verification_steps"].append("✓ Base64 decode successful")
        except Exception as e:
            raise AttestationError(f"Invalid base64 encoding: {e}")
        
        # =================================================================
        # Step 2: Parse COSE_Sign1 structure
        # COSE_Sign1 = [protected, unprotected, payload, signature]
        # =================================================================
        try:
            cose_sign1 = cbor2.loads(att_bytes)
            
            # Handle CBOR tagged value (tag 18 = COSE_Sign1)
            if hasattr(cose_sign1, 'value'):
                cose_array = cose_sign1.value
            elif isinstance(cose_sign1, list):
                cose_array = cose_sign1
            else:
                raise AttestationError(f"Unexpected COSE structure type: {type(cose_sign1)}")
            
            if len(cose_array) != 4:
                raise AttestationError(f"Invalid COSE_Sign1: expected 4 elements, got {len(cose_array)}")
            
            protected, unprotected, payload, signature = cose_array
            result["verification_steps"].append("✓ COSE_Sign1 structure parsed")
        except AttestationError:
            raise
        except Exception as e:
            raise AttestationError(f"Failed to parse COSE_Sign1: {e}")
        
        # =================================================================
        # Step 3: Parse attestation document from payload
        # =================================================================
        try:
            att_doc = cbor2.loads(payload)
            result["verification_steps"].append("✓ Attestation document parsed")
            result["module_id"] = att_doc.get("module_id")
            result["timestamp"] = att_doc.get("timestamp")
            result["digest"] = att_doc.get("digest")
        except Exception as e:
            raise AttestationError(f"Failed to parse attestation payload: {e}")
        
        # =================================================================
        # Step 4: Extract and verify certificate chain
        # =================================================================
        cert_der = att_doc.get("certificate")
        cabundle = att_doc.get("cabundle", [])
        
        if not cert_der:
            raise AttestationError("No certificate found in attestation")
        
        try:
            # Parse leaf certificate
            leaf_cert = x509.load_der_x509_certificate(cert_der, default_backend())
            result["leaf_cert_subject"] = str(leaf_cert.subject)
            result["leaf_cert_issuer"] = str(leaf_cert.issuer)
            
            validation_time = (
                _attestation_certificate_validation_time(att_doc)
                if certificate_validity_at_attestation_time
                else datetime.now(timezone.utc)
            )
            _verify_leaf_certificate_validity(leaf_cert, validation_time)
            result["certificate_validity_mode"] = (
                "attestation_time"
                if certificate_validity_at_attestation_time
                else "current_time"
            )
            result["certificate_validation_time"] = validation_time.isoformat()
            
            result["verification_steps"].append(
                "✓ Leaf certificate valid at verification time"
            )
            
            # Build and verify certificate chain
            chain_verified = _verify_certificate_chain(leaf_cert, cabundle, NITRO_ROOT_CERT_DER)
            if not chain_verified:
                raise AttestationError("Certificate chain verification failed")
            
            result["verification_steps"].append("✓ Certificate chain verified to Amazon Nitro root")
            
        except AttestationError:
            raise
        except Exception as e:
            raise AttestationError(f"Certificate verification failed: {e}")
        
        # =================================================================
        # Step 5: Verify COSE signature
        # =================================================================
        try:
            from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
            
            # Get public key from leaf certificate
            public_key = leaf_cert.public_key()
            
            # Verify COSE_Sign1 signature
            # Sig_structure = ["Signature1", protected, external_aad, payload]
            sig_structure = cbor2.dumps(["Signature1", protected, b"", payload])
            
            # COSE signatures are in raw (r || s) format, but cryptography library
            # requires DER-encoded signatures for ECDSA. Convert:
            sig_len = len(signature)
            r = int.from_bytes(signature[:sig_len//2], 'big')
            s = int.from_bytes(signature[sig_len//2:], 'big')
            der_signature = encode_dss_signature(r, s)
            
            # AWS Nitro uses ECDSA with SHA-384 (algorithm -35 in COSE)
            public_key.verify(der_signature, sig_structure, ec.ECDSA(hashes.SHA384()))
            
            result["verification_steps"].append("✓ COSE signature verified")
            
        except InvalidSignature:
            raise AttestationError("COSE signature verification failed - attestation may be forged")
        except Exception as e:
            raise AttestationError(f"Signature verification error: {e}")
        
        # =================================================================
        # Step 6: Extract and verify PCR0 (ROOT OF TRUST)
        # =================================================================
        pcrs = att_doc.get("pcrs", {})
        pcr0_raw = pcrs.get(0)
        
        if pcr0_raw is None:
            raise AttestationError("PCR0 not found in attestation")
        
        pcr0_hex = pcr0_raw.hex() if isinstance(pcr0_raw, bytes) else str(pcr0_raw)
        result["pcr0"] = pcr0_hex
        result["pcr1"] = pcrs.get(1, b"").hex() if pcrs.get(1) else None
        result["pcr2"] = pcrs.get(2, b"").hex() if pcrs.get(2) else None
        
        # PCR0 verification mode:
        # - If skip_pcr0_verification: extract PCR0 but don't verify (caller verifies measurement policy separately)
        # - If expected_pcr0 is provided: strict match required
        # - Otherwise: check against static allowlist
        
        if skip_pcr0_verification:
            # AUDITOR MODE: Skip PCR0 verification
            # We've already verified:
            # - AWS certificate chain (proves REAL Nitro enclave)
            # - COSE signature (proves attestation is authentic)
            # - Certificate validity (not expired)
            # We extract PCR0 for logging but DON'T verify it against any allowlist
            result["verification_steps"].append("⚠️ PCR0 verification SKIPPED (external PCR policy check)")
            result["pcr0_verification_mode"] = "skipped"
            result["trust_level"] = "aws_verified"  # Lower trust level - proves real enclave, not specific code
            logger.info(f"[NITRO] ⚠️ PCR0 verification skipped (external PCR policy check), PCR0: {pcr0_hex[:32]}...")
        elif expected_pcr0:
            # Strict mode: must match specific PCR0
            if pcr0_hex != expected_pcr0:
                raise AttestationError(
                    f"PCR0 mismatch!\n"
                    f"  Got:      {pcr0_hex}\n"
                    f"  Expected: {expected_pcr0[:32]}..."
                )
            result["verification_steps"].append("✓ PCR0 matches expected value")
        else:
            # Gateway mode: check against static allowlist
            allowed_pcr0_list = get_allowed_gateway_pcr0()
            
            if not allowed_pcr0_list:
                raise AttestationError("No allowed gateway PCR0 values configured")
            
            if pcr0_hex not in allowed_pcr0_list:
                raise AttestationError(
                    f"PCR0 mismatch (ROOT OF TRUST FAILURE)!\n"
                    f"  Got:      {pcr0_hex}\n"
                    f"  Expected: {allowed_pcr0_list[0][:32]}...\n"
                    f"  This enclave is running UNKNOWN code!"
                )
            result["verification_steps"].append("✓ PCR0 matches allowlist value")
        
        # =================================================================
        # Step 7: Extract user_data (NOW we can trust it after PCR0 verified)
        # =================================================================
        user_data_raw = att_doc.get("user_data")
        public_key_raw = att_doc.get("public_key")
        
        if user_data_raw:
            user_data = None
            
            # Try to decode user_data - could be CBOR, JSON, or raw bytes
            if isinstance(user_data_raw, bytes):
                # Try JSON first (gateway uses JSON in user_data)
                try:
                    user_data = json.loads(user_data_raw.decode('utf-8'))
                    result["verification_steps"].append("✓ user_data extracted (JSON format)")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Try CBOR
                    try:
                        user_data = cbor2.loads(user_data_raw)
                        result["verification_steps"].append("✓ user_data extracted (CBOR format)")
                    except Exception:
                        result["user_data_raw"] = user_data_raw.hex()
                        result["verification_steps"].append("⚠ user_data not JSON/CBOR")
            elif isinstance(user_data_raw, dict):
                user_data = user_data_raw
                result["verification_steps"].append("✓ user_data extracted (dict)")
            
            if user_data:
                result["user_data"] = user_data
                # Handle both naming conventions: enclave_pubkey and enclave_public_key
                result["enclave_pubkey"] = user_data.get("enclave_pubkey") or user_data.get("enclave_public_key")
                result["code_hash"] = user_data.get("code_hash")
                result["purpose"] = user_data.get("purpose")
                result["epoch_id"] = user_data.get("epoch_id")
        
        if public_key_raw:
            result["attestation_public_key"] = public_key_raw.hex() if isinstance(public_key_raw, bytes) else str(public_key_raw)
        
        # =================================================================
        # Step 8: Verify enclave pubkey binding (optional)
        # =================================================================
        if expected_pubkey is not None:
            actual_pubkey = result.get("enclave_pubkey")
            if actual_pubkey != expected_pubkey:
                raise AttestationError(
                    f"Enclave pubkey mismatch!\n"
                    f"  Got:      {actual_pubkey}\n"
                    f"  Expected: {expected_pubkey}"
                )
            result["verification_steps"].append("✓ Enclave pubkey matches expected")
        
        # =================================================================
        # Step 9: Verify purpose (optional - only fail if purpose provided but doesn't match)
        # =================================================================
        if expected_purpose is not None:
            actual_purpose = result.get("purpose")
            if actual_purpose is not None and actual_purpose != expected_purpose:
                raise AttestationError(
                    f"Purpose mismatch!\n"
                    f"  Got:      {actual_purpose}\n"
                    f"  Expected: {expected_purpose}"
                )
            elif actual_purpose == expected_purpose:
                result["verification_steps"].append(f"✓ Purpose verified: {expected_purpose}")
            else:
                # Purpose not in attestation - that's OK, not all enclaves set it
                result["verification_steps"].append(f"⚠ Purpose not in attestation (expected: {expected_purpose})")
        
        # =================================================================
        # ALL CHECKS PASSED
        # =================================================================
        result["verified"] = True
        result["verification_steps"].append("✅ ALL VERIFICATION STEPS PASSED")
        
        return True, result
        
    except AttestationError as e:
        return False, {
            "error": str(e),
            "trust_level": TRUST_LEVEL_SIGNATURE_ONLY,
            "verified": False,
            **{k: v for k, v in result.items() if k != "verified"}
        }
    except Exception as e:
        return False, {
            "error": f"Unexpected error: {e}",
            "trust_level": TRUST_LEVEL_SIGNATURE_ONLY,
            "verified": False,
        }


def _verify_certificate_chain(
    leaf_cert,
    cabundle: List[bytes],
    root_cert_der: bytes
) -> bool:
    """
    Verify certificate chain from leaf to pinned Amazon Nitro root.
    
    AWS Nitro attestation certificate chain structure:
    - cabundle[0] = Root CA (aws.nitro-enclaves, self-signed)
    - cabundle[1] = Regional CA (signed by root)
    - cabundle[2] = Zonal CA (signed by regional)
    - cabundle[3] = Instance CA (signed by zonal) - signs the leaf
    - leaf = Enclave cert (signed by instance CA)
    
    The cabundle is ordered from ROOT to LEAF (not leaf to root).
    
    Args:
        leaf_cert: Parsed X.509 leaf certificate
        cabundle: List of CA certificates (DER-encoded), 
                  ordered from root to closest-to-leaf
        root_cert_der: Pinned Amazon Nitro root certificate (DER-encoded)
        
    Returns:
        True if chain is valid
        
    Raises:
        AttestationError: If chain verification fails
    """
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.serialization import Encoding
    
    try:
        # Parse all certificates from cabundle
        ca_certs = []
        for ca_der in cabundle:
            ca_cert = x509.load_der_x509_certificate(ca_der, default_backend())
            ca_certs.append(ca_cert)
        
        if not ca_certs:
            raise AttestationError("Empty CA bundle - cannot verify chain")
        
        # cabundle[0] should be the root - verify it matches our pinned root
        bundle_root = ca_certs[0]
        bundle_root_der = bundle_root.public_bytes(Encoding.DER)
        if bundle_root_der != root_cert_der:
            raise AttestationError(
                "Bundle root does not match pinned Amazon Nitro root certificate. "
                f"Bundle root subject: {bundle_root.subject}"
            )
        
        # Verify root is self-signed
        _verify_cert_signature(bundle_root, bundle_root, 0, "root self-signature")
        
        # Verify chain: each cert is signed by the previous one
        # cabundle[1] is signed by cabundle[0] (root)
        # cabundle[2] is signed by cabundle[1]
        # ... and so on
        for i in range(1, len(ca_certs)):
            cert = ca_certs[i]
            issuer = ca_certs[i - 1]
            _verify_cert_signature(cert, issuer, i, f"CA[{i}] signed by CA[{i-1}]")
        
        # Finally, verify leaf is signed by the last CA in the bundle
        last_ca = ca_certs[-1]
        _verify_cert_signature(leaf_cert, last_ca, len(ca_certs), f"leaf signed by CA[{len(ca_certs)-1}]")
        
        return True
        
    except AttestationError:
        raise
    except Exception as e:
        raise AttestationError(f"Certificate chain verification error: {e}")


def _verify_cert_signature(cert, issuer_cert, position: int, description: str = "") -> None:
    """
    Verify that cert is signed by issuer_cert.
    
    Note: AWS Nitro certificates have dynamic CN values that include instance IDs,
    so we only verify the signature, not the issuer/subject name matching.
    The cryptographic signature is what matters for security.
    """
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.exceptions import InvalidSignature
    
    try:
        issuer_public_key = issuer_cert.public_key()
        
        if isinstance(issuer_public_key, ec.EllipticCurvePublicKey):
            issuer_public_key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                ec.ECDSA(cert.signature_hash_algorithm)
            )
        else:
            raise AttestationError(f"Unexpected key type at position {position}: {type(issuer_public_key)}")
            
    except InvalidSignature:
        desc = f" ({description})" if description else ""
        raise AttestationError(
            f"Certificate signature verification failed at position {position}{desc}. "
            f"Cert subject: {cert.subject}"
        )


# =============================================================================
# SIGNATURE-ONLY MODE (for development/fallback)
# =============================================================================

def verify_nitro_attestation_signature_only(
    attestation_b64: str,
    expected_pubkey: str = None,
    expected_purpose: str = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Parse attestation and verify user_data WITHOUT full Nitro verification.
    
    ⚠️ WARNING: This is SIGNATURE-ONLY mode. It does NOT verify:
    - Certificate chain to Amazon Nitro root
    - COSE signature
    - PCR0 measurement against pinned values
    
    This mode is ONLY acceptable for development/testing. Production MUST use
    verify_nitro_attestation_full().
    
    Args:
        attestation_b64: Base64-encoded Nitro attestation document
        expected_pubkey: Expected enclave public key (hex string)
        expected_purpose: Expected purpose in user_data (optional)
        
    Returns:
        Tuple of (success, extracted_data)
        - success: True if user_data parsing succeeded and values match
        - extracted_data: Dictionary with extracted values (or error info)
        
    Security Note:
        This function does NOT provide trustless verification. An attacker
        could fabricate attestation documents. Only use for:
        - Development/testing
        - Fallback when full verification is not available
        - Must always report TRUST_LEVEL_SIGNATURE_ONLY in outputs
    """
    try:
        import cbor2
        
        # Decode attestation
        att_bytes = base64.b64decode(attestation_b64)
        
        # Parse COSE_Sign1 structure
        cose_sign1 = cbor2.loads(att_bytes)
        if hasattr(cose_sign1, 'value'):
            protected, unprotected, payload, signature = cose_sign1.value
        else:
            # Handle non-tagged CBOR
            if isinstance(cose_sign1, list) and len(cose_sign1) == 4:
                protected, unprotected, payload, signature = cose_sign1
            else:
                return False, {"error": "Invalid COSE_Sign1 structure"}
        
        # Parse attestation document from payload
        att_doc = cbor2.loads(payload)
        
        # Extract PCR0
        pcrs = att_doc.get("pcrs", {})
        pcr0 = pcrs.get(0, b"").hex() if pcrs.get(0) else None
        
        # Extract user_data (CBOR-encoded)
        user_data_raw = att_doc.get("user_data")
        if user_data_raw:
            try:
                user_data = cbor2.loads(user_data_raw) if isinstance(user_data_raw, bytes) else user_data_raw
            except:
                user_data = {"raw": user_data_raw.hex() if isinstance(user_data_raw, bytes) else str(user_data_raw)}
        else:
            user_data = {}
        
        # Extract values
        extracted = {
            "enclave_pubkey": user_data.get("enclave_pubkey"),
            "code_hash": user_data.get("code_hash"),
            "purpose": user_data.get("purpose"),
            "epoch_id": user_data.get("epoch_id"),
            "trust_level": TRUST_LEVEL_SIGNATURE_ONLY,  # Always signature-only
            "pcr0": pcr0,
            "module_id": att_doc.get("module_id"),
            "timestamp": att_doc.get("timestamp"),
            "warning": "⚠️ SIGNATURE-ONLY MODE: Certificate chain and COSE signature NOT verified",
        }
        
        # Verify expected values (if provided)
        if expected_pubkey is not None:
            if extracted["enclave_pubkey"] != expected_pubkey:
                return False, {
                    "error": f"Pubkey mismatch: got {extracted['enclave_pubkey']}, expected {expected_pubkey}",
                    **extracted
                }
        
        if expected_purpose is not None:
            if extracted["purpose"] != expected_purpose:
                return False, {
                    "error": f"Purpose mismatch: got {extracted['purpose']}, expected {expected_purpose}",
                    **extracted
                }
        
        return True, extracted
        
    except Exception as e:
        return False, {"error": f"Parse error: {str(e)}", "trust_level": TRUST_LEVEL_SIGNATURE_ONLY}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_nitro_verification_available() -> bool:
    """Return whether full gateway Nitro verification dependencies are ready."""
    if not NITRO_ROOT_CERT_DER or not get_allowed_gateway_pcr0():
        return False
    try:
        __import__("cbor2")
        __import__("cryptography.x509")
    except ImportError:
        return False
    return True

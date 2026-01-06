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
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Tuple, List

from leadpoet_canonical.constants import TRUST_LEVEL_FULL_NITRO, TRUST_LEVEL_SIGNATURE_ONLY


class AttestationError(Exception):
    """Raised when attestation verification fails."""
    pass


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

# Allowed PCR0 values for gateway enclave
# Update this list when deploying new gateway builds
# PCR0 = SHA384 hash of the enclave image (EIF file)
ALLOWED_GATEWAY_PCR0_VALUES: List[str] = [
    # Gateway enclave deployed 2026-01-05
    "02797d0a3b02fdda186db756b7cae6ef283592bae6ea879c0c19e4ab0a787766bbbd2008eb49eb9de58f7346d6c834d5",
]

# Allowed PCR0 values for validator enclave
# Update this list when deploying new validator builds
ALLOWED_VALIDATOR_PCR0_VALUES: List[str] = [
    # Validator enclave deployed 2026-01-06
    "1697ef7e8c095ff5fc3d7e0e79bb7d00d29d0bdfa487d2c7353812ebafb35667ebd428c42db59ad1efe1c2999d1e5d85",
]


# =============================================================================
# FULL NITRO ATTESTATION VERIFICATION
# =============================================================================

def verify_nitro_attestation_full(
    attestation_b64: str,
    expected_pcr0: str = None,
    expected_pubkey: str = None,
    expected_purpose: str = None,
    expected_epoch_id: Optional[int] = None,
    role: str = "gateway",  # "gateway" or "validator"
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
        expected_purpose: Expected purpose in user_data ("gateway_event_signing" or 
                         "validator_weights"). If None, purpose check is skipped.
        expected_epoch_id: Expected epoch_id for validator attestations (required if
                          purpose is "validator_weights")
        role: "gateway" or "validator" - determines which PCR0 allowlist to use
                          
    Returns:
        Tuple of (success, extracted_data)
        - success: True if all verification steps pass
        - extracted_data: Dictionary with extracted values and verification status
        
    Security Notes:
        - PCR0 is the ROOT OF TRUST, NOT user_data.code_hash
        - Certificate chain MUST be verified to pinned Amazon Nitro root
        - Epoch binding prevents replay of old validator attestations
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
            
            # Verify certificate is currently valid
            now = datetime.now(timezone.utc)
            if now < leaf_cert.not_valid_before_utc:
                raise AttestationError(f"Certificate not yet valid (starts {leaf_cert.not_valid_before_utc})")
            if now > leaf_cert.not_valid_after_utc:
                raise AttestationError(f"Certificate expired ({leaf_cert.not_valid_after_utc})")
            
            result["verification_steps"].append("✓ Leaf certificate valid and not expired")
            
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
        
        # Determine which PCR0 allowlist to use
        if expected_pcr0:
            allowed_pcr0_list = [expected_pcr0]
        elif role == "gateway":
            allowed_pcr0_list = ALLOWED_GATEWAY_PCR0_VALUES
        elif role == "validator":
            allowed_pcr0_list = ALLOWED_VALIDATOR_PCR0_VALUES
        else:
            raise AttestationError(f"Unknown role: {role}")
        
        if not allowed_pcr0_list:
            raise AttestationError(f"No allowed PCR0 values configured for role '{role}'")
        
        if pcr0_hex not in allowed_pcr0_list:
            raise AttestationError(
                f"PCR0 mismatch (ROOT OF TRUST FAILURE)!\n"
                f"  Got:      {pcr0_hex}\n"
                f"  Expected: {allowed_pcr0_list[0][:32]}...\n"
                f"  This enclave is running UNKNOWN code!"
            )
        
        result["verification_steps"].append("✓ PCR0 matches pinned value (ROOT OF TRUST verified)")
        
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
        # Step 10: Verify epoch binding for validators
        # =================================================================
        if expected_purpose == "validator_weights":
            if expected_epoch_id is None:
                raise AttestationError("expected_epoch_id required for validator attestations (replay protection)")
            
            actual_epoch = result.get("epoch_id")
            if actual_epoch != expected_epoch_id:
                raise AttestationError(
                    f"Epoch mismatch (potential replay attack)!\n"
                    f"  Got:      {actual_epoch}\n"
                    f"  Expected: {expected_epoch_id}"
                )
            result["verification_steps"].append(f"✓ Epoch binding verified: {expected_epoch_id}")
        
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
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.hazmat.primitives import hashes
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.serialization import Encoding
    
    try:
        # Parse pinned root certificate
        pinned_root = x509.load_der_x509_certificate(root_cert_der, default_backend())
        
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
    expected_epoch_id: Optional[int] = None,
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
        expected_epoch_id: Expected epoch_id for validator attestations (optional)
        
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
        
        if expected_purpose == "validator_weights" and expected_epoch_id is not None:
            if extracted["epoch_id"] != expected_epoch_id:
                return False, {
                    "error": f"Epoch mismatch: got {extracted['epoch_id']}, expected {expected_epoch_id}",
                    **extracted
                }
        
        return True, extracted
        
    except Exception as e:
        return False, {"error": f"Parse error: {str(e)}", "trust_level": TRUST_LEVEL_SIGNATURE_ONLY}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_nitro_verification_available() -> bool:
    """
    Check if full Nitro attestation verification is available.
    
    Returns:
        True if all requirements are met for full verification:
        - NITRO_ROOT_CERT_DER is populated
        - Required libraries are available (cbor2, cryptography)
        - PCR0 allowlists are populated
        
    Use this to determine trust level in verification outputs.
    """
    if NITRO_ROOT_CERT_DER is None or len(NITRO_ROOT_CERT_DER) == 0:
        return False
    
    if not ALLOWED_GATEWAY_PCR0_VALUES and not ALLOWED_VALIDATOR_PCR0_VALUES:
        return False
    
    try:
        import cbor2
        from cryptography.x509 import load_der_x509_certificate
        from cryptography.hazmat.primitives.asymmetric import ec
        return True
    except ImportError:
        return False


def get_current_trust_level() -> str:
    """
    Get the current trust level based on available verification capabilities.
    
    Returns:
        TRUST_LEVEL_FULL_NITRO if full verification is available
        TRUST_LEVEL_SIGNATURE_ONLY otherwise
    """
    if is_nitro_verification_available():
        return TRUST_LEVEL_FULL_NITRO
    return TRUST_LEVEL_SIGNATURE_ONLY


def get_allowed_pcr0_values(role: str = "gateway") -> List[str]:
    """
    Get the list of allowed PCR0 values for a given role.
    
    Args:
        role: "gateway" or "validator"
        
    Returns:
        List of allowed PCR0 hex strings
    """
    if role == "gateway":
        return ALLOWED_GATEWAY_PCR0_VALUES.copy()
    elif role == "validator":
        return ALLOWED_VALIDATOR_PCR0_VALUES.copy()
    else:
        return []


def add_allowed_pcr0(pcr0_hex: str, role: str = "gateway") -> None:
    """
    Add a PCR0 value to the allowlist (for deployment updates).
    
    ⚠️ WARNING: In production, allowlists should be hardcoded, not modified at runtime.
    This function is for testing/development only.
    
    Args:
        pcr0_hex: PCR0 value to add (hex string, 96 characters for SHA-384)
        role: "gateway" or "validator"
    """
    if len(pcr0_hex) != 96:
        raise ValueError(f"PCR0 must be 96 hex characters (SHA-384), got {len(pcr0_hex)}")
    
    if role == "gateway":
        if pcr0_hex not in ALLOWED_GATEWAY_PCR0_VALUES:
            ALLOWED_GATEWAY_PCR0_VALUES.append(pcr0_hex)
    elif role == "validator":
        if pcr0_hex not in ALLOWED_VALIDATOR_PCR0_VALUES:
            ALLOWED_VALIDATOR_PCR0_VALUES.append(pcr0_hex)
    else:
        raise ValueError(f"Unknown role: {role}")


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_is_nitro_verification_available():
    """Test availability check."""
    available = is_nitro_verification_available()
    print(f"Nitro verification available: {available}")
    if available:
        print("✅ Full Nitro verification is AVAILABLE")
    else:
        print("⚠️ Nitro verification not available (missing libraries or config)")


def test_get_current_trust_level():
    """Test trust level reporting."""
    level = get_current_trust_level()
    print(f"Current trust level: {level}")
    if level == TRUST_LEVEL_FULL_NITRO:
        print("✅ Trust level: FULL_NITRO")
    else:
        print(f"⚠️ Trust level: {level}")


def test_pinned_values():
    """Test that pinned values are populated."""
    print("\n--- Pinned Values Check ---")
    
    if NITRO_ROOT_CERT_DER:
        print(f"✅ NITRO_ROOT_CERT_DER: {len(NITRO_ROOT_CERT_DER)} bytes")
    else:
        print("❌ NITRO_ROOT_CERT_DER: Not populated")
    
    if ALLOWED_GATEWAY_PCR0_VALUES:
        print(f"✅ ALLOWED_GATEWAY_PCR0_VALUES: {len(ALLOWED_GATEWAY_PCR0_VALUES)} value(s)")
        for pcr0 in ALLOWED_GATEWAY_PCR0_VALUES:
            print(f"   - {pcr0[:32]}...{pcr0[-16:]}")
    else:
        print("❌ ALLOWED_GATEWAY_PCR0_VALUES: Empty")
    
    if ALLOWED_VALIDATOR_PCR0_VALUES:
        print(f"✅ ALLOWED_VALIDATOR_PCR0_VALUES: {len(ALLOWED_VALIDATOR_PCR0_VALUES)} value(s)")
    else:
        print("⚠️ ALLOWED_VALIDATOR_PCR0_VALUES: Empty (validator TEE not deployed)")


def test_root_cert_parsing():
    """Test that the pinned root certificate can be parsed."""
    print("\n--- Root Certificate Test ---")
    
    if not NITRO_ROOT_CERT_DER:
        print("❌ Cannot test: NITRO_ROOT_CERT_DER not populated")
        return
    
    try:
        from cryptography import x509
        from cryptography.hazmat.backends import default_backend
        
        root_cert = x509.load_der_x509_certificate(NITRO_ROOT_CERT_DER, default_backend())
        print(f"✅ Root certificate parsed successfully")
        print(f"   Subject: {root_cert.subject}")
        print(f"   Issuer: {root_cert.issuer}")
        print(f"   Valid from: {root_cert.not_valid_before_utc}")
        print(f"   Valid until: {root_cert.not_valid_after_utc}")
    except Exception as e:
        print(f"❌ Failed to parse root certificate: {e}")


def test_verify_signature_only_invalid_input():
    """Test signature-only mode with invalid input."""
    print("\n--- Signature-Only Invalid Input Test ---")
    
    success, result = verify_nitro_attestation_signature_only(
        attestation_b64="invalid_base64!!!",
        expected_pubkey="abc123",
    )
    assert success == False, "Should fail on invalid input"
    assert "error" in result, "Should have error message"
    print(f"✅ Invalid input correctly rejected: {result['error'][:50]}...")


if __name__ == "__main__":
    print("=" * 60)
    print("LeadPoet Nitro Attestation Verification - Unit Tests")
    print("=" * 60)
    
    test_is_nitro_verification_available()
    test_get_current_trust_level()
    test_pinned_values()
    test_root_cert_parsing()
    test_verify_signature_only_invalid_input()
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

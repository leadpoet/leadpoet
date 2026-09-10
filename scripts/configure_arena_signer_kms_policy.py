#!/usr/bin/env python3
"""Inspect or install the fixed Nitro recipient guards on the Arena signer key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCOUNT = "493765492819"
REGION = "us-east-1"
KEY_ARN = "arn:aws:kms:us-east-1:493765492819:key/6822c852-4bf2-4b2a-be0b-91a61057f92d"
CALLER_ARN = "arn:aws:iam::493765492819:user/pranav-main"
POLICY_NAME = "default"
RECIPIENT_DENY_SID = "DenyArenaSignerDecryptWithoutApprovedNitroImage"
REENCRYPT_DENY_SID = "DenyArenaSignerReEncryptFrom"
OLD_PCR0 = "45c8766b1a4dc51460dbda7c9fad1f4aecb96b33f71faf372486de582abc19c0a99dc5542942bc08f7aefa9b437d0529"
HASH_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PCR_RE = re.compile(r"^[0-9a-f]{96}$")


class PolicyError(RuntimeError):
    pass


def canonical_hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def normalize_pcrs(values):
    result = sorted(set(str(value).strip().lower() for value in values))
    if not result or any(not PCR_RE.fullmatch(value) or value == "0" * 96 for value in result):
        raise PolicyError("approved PCR0 values must be nonzero 96-character lowercase hex")
    return result


def guarded_policy(current, approved_pcr0s):
    if not isinstance(current, dict) or not isinstance(current.get("Statement"), list):
        raise PolicyError("KMS default policy document is invalid")
    pcrs = normalize_pcrs(approved_pcr0s)
    managed = {RECIPIENT_DENY_SID, REENCRYPT_DENY_SID}
    statements = [statement for statement in current["Statement"] if not (
        isinstance(statement, dict) and statement.get("Sid") in managed
    )]
    statements.extend([
        {
            "Sid": RECIPIENT_DENY_SID,
            "Effect": "Deny",
            "Principal": {"AWS": "*"},
            "Action": "kms:Decrypt",
            "Resource": "*",
            "Condition": {
                "StringNotEquals": {
                    "kms:RecipientAttestation:ImageSha384": pcrs,
                }
            },
        },
        {
            "Sid": REENCRYPT_DENY_SID,
            "Effect": "Deny",
            "Principal": {"AWS": "*"},
            "Action": "kms:ReEncryptFrom",
            "Resource": "*",
        },
    ])
    return {**current, "Statement": statements}


def _private_write(path, value):
    path = Path(path)
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as target:
        json.dump(value, target, sort_keys=True, separators=(",", ":"))
        target.write("\n")
        target.flush(); os.fsync(target.fileno())


def _read_policy(kms):
    response = kms.get_key_policy(KeyId=KEY_ARN, PolicyName=POLICY_NAME)
    raw = response.get("Policy")
    if not isinstance(raw, str):
        raise PolicyError("KMS did not return the default policy")
    try:
        return json.loads(raw)
    except ValueError:
        raise PolicyError("KMS returned invalid policy JSON") from None


def operate(*, session, approved_pcr0s, expected_policy_hash=None, apply=False,
            before_path=None, after_path=None):
    identity = session.client("sts", region_name=REGION).get_caller_identity()
    if identity.get("Account") != ACCOUNT or identity.get("Arn") != CALLER_ARN:
        raise PolicyError("AWS caller is not the fixed Arena policy operator")
    kms = session.client("kms", region_name=REGION)
    current = _read_policy(kms)
    before_hash = canonical_hash(current)
    target = guarded_policy(current, approved_pcr0s)
    after_hash = canonical_hash(target)
    if before_path:
        _private_write(before_path, current)
    if not apply:
        return {"applied": False, "before_hash": before_hash, "after_hash": after_hash,
                "approved_pcr0s": normalize_pcrs(approved_pcr0s)}
    if not HASH_RE.fullmatch(str(expected_policy_hash or "")) or expected_policy_hash != before_hash:
        raise PolicyError("current KMS policy hash does not equal --expected-policy-sha256")
    # AWS KMS has no conditional policy write. Minimize the race with an exact
    # second read immediately before the bounded write.
    if canonical_hash(_read_policy(kms)) != before_hash:
        raise PolicyError("KMS policy changed before apply")
    kms.put_key_policy(
        KeyId=KEY_ARN, PolicyName=POLICY_NAME,
        Policy=json.dumps(target, sort_keys=True, separators=(",", ":")),
        BypassPolicyLockoutSafetyCheck=False,
    )
    readback = _read_policy(kms)
    if canonical_hash(readback) != after_hash:
        raise PolicyError("KMS policy readback differs after apply")
    if after_path:
        _private_write(after_path, readback)
    return {"applied": True, "before_hash": before_hash, "after_hash": after_hash,
            "approved_pcr0s": normalize_pcrs(approved_pcr0s)}


def _session():
    from scripts.gateway_iam_session import gateway_iam_session
    return gateway_iam_session(region=REGION)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-pcr0", action="append", required=True)
    parser.add_argument("--expected-policy-sha256")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--before-evidence", type=Path)
    parser.add_argument("--after-evidence", type=Path)
    args = parser.parse_args(argv)
    if args.apply and not args.expected_policy_sha256:
        parser.error("--apply requires --expected-policy-sha256")
    receipt = operate(
        session=_session(), approved_pcr0s=args.approved_pcr0,
        expected_policy_hash=args.expected_policy_sha256, apply=args.apply,
        before_path=args.before_evidence, after_path=args.after_evidence,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

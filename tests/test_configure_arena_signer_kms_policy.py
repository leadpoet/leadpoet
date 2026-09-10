import json

import pytest

from scripts import configure_arena_signer_kms_policy as module


def _policy():
    return {
        "Version": "2012-10-17",
        "Id": "keep-id",
        "Statement": [{
            "Sid": "EnableRootControl", "Effect": "Allow",
            "Principal": {"AWS": "arn:aws:iam::493765492819:root"},
            "Action": "kms:*", "Resource": "*",
        }],
    }


class FakeKms:
    def __init__(self, policy): self.policy = policy; self.puts = []
    def get_key_policy(self, **kwargs):
        assert kwargs == {"KeyId": module.KEY_ARN, "PolicyName": "default"}
        return {"Policy": json.dumps(self.policy)}
    def put_key_policy(self, **kwargs):
        self.puts.append(kwargs); self.policy = json.loads(kwargs["Policy"])


class FakeSts:
    def __init__(self, arn=module.CALLER_ARN): self.arn = arn
    def get_caller_identity(self): return {"Account": module.ACCOUNT, "Arn": self.arn}


class FakeSession:
    def __init__(self, kms, sts=None): self.kms = kms; self.sts = sts or FakeSts()
    def client(self, service, region_name=None):
        assert region_name == module.REGION
        return self.kms if service == "kms" else self.sts


def test_guard_preserves_unrelated_policy_and_adds_only_fixed_denies():
    current = _policy(); new = "1" * 96
    guarded = module.guarded_policy(current, [module.OLD_PCR0, new, new])
    assert current == _policy()
    assert guarded["Id"] == "keep-id" and guarded["Statement"][0] == current["Statement"][0]
    recipient, reencrypt = guarded["Statement"][1:]
    assert recipient == {
        "Sid": module.RECIPIENT_DENY_SID, "Effect": "Deny",
        "Principal": {"AWS": "*"}, "Action": "kms:Decrypt", "Resource": "*",
        "Condition": {"StringNotEquals": {
            "kms:RecipientAttestation:ImageSha384": sorted([module.OLD_PCR0, new])}},
    }
    assert reencrypt["Sid"] == module.REENCRYPT_DENY_SID
    assert reencrypt["Action"] == "kms:ReEncryptFrom" and reencrypt["Effect"] == "Deny"


@pytest.mark.parametrize("pcrs", [[], ["0" * 96], ["a" * 95]])
def test_invalid_or_zero_pcr_is_rejected(pcrs):
    with pytest.raises(module.PolicyError, match="PCR0"):
        module.guarded_policy(_policy(), pcrs)


def test_dry_run_has_no_write_and_apply_requires_exact_current_hash():
    kms = FakeKms(_policy()); session = FakeSession(kms); new = "1" * 96
    dry = module.operate(session=session, approved_pcr0s=[module.OLD_PCR0, new])
    assert dry["applied"] is False and not kms.puts
    with pytest.raises(module.PolicyError, match="does not equal"):
        module.operate(session=session, approved_pcr0s=[new], apply=True,
                       expected_policy_hash="sha256:" + "0" * 64)
    receipt = module.operate(session=session, approved_pcr0s=[module.OLD_PCR0, new],
                             apply=True, expected_policy_hash=dry["before_hash"])
    assert receipt["applied"] is True and len(kms.puts) == 1
    assert kms.puts[0]["BypassPolicyLockoutSafetyCheck"] is False
    assert kms.policy["Statement"][0]["Sid"] == "EnableRootControl"


def test_wrong_caller_fails_before_kms_read():
    kms = FakeKms(_policy())
    with pytest.raises(module.PolicyError, match="fixed Arena policy operator"):
        module.operate(session=FakeSession(kms, FakeSts("arn:aws:iam::493765492819:user/other")),
                       approved_pcr0s=["1" * 96])
    assert not kms.puts


def test_managed_statements_are_replaced_without_duplicate_or_other_change():
    current = module.guarded_policy(_policy(), [module.OLD_PCR0])
    current["Statement"].append({"Sid": "UnrelatedDeny", "Effect": "Deny"})
    changed = module.guarded_policy(current, ["2" * 96])
    assert [s["Sid"] for s in changed["Statement"]].count(module.RECIPIENT_DENY_SID) == 1
    assert any(s.get("Sid") == "UnrelatedDeny" for s in changed["Statement"])

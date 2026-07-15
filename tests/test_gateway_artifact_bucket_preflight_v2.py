import pytest

from gateway.tee.restart_preflight_v2 import (
    GatewayRestartPreflightV2Error,
    verify_artifact_bucket_lock_v2,
)


POLICY = {
    "schema_version": "leadpoet.encrypted_artifact_policy.v2",
    "bucket_host": "leadpoet-v2.s3.us-east-1.amazonaws.com",
    "key_prefix": "/research-lab/v2/",
    "minimum_retention_days": 30,
}


class S3:
    def __init__(self, mode="COMPLIANCE", days=30):
        self.mode = mode
        self.days = days

    def get_bucket_versioning(self, **request):
        assert request == {"Bucket": "leadpoet-v2"}
        return {"Status": "Enabled"}

    def get_object_lock_configuration(self, **request):
        assert request == {"Bucket": "leadpoet-v2"}
        return {
            "ObjectLockConfiguration": {
                "ObjectLockEnabled": "Enabled",
                "Rule": {
                    "DefaultRetention": {
                        "Mode": self.mode,
                        "Days": self.days,
                    }
                },
            }
        }


def test_requires_live_compliance_retention():
    assert verify_artifact_bucket_lock_v2(POLICY, s3_client=S3()) == {
        "bucket": "leadpoet-v2",
        "object_lock_mode": "COMPLIANCE",
        "retention_days": 30,
        "versioning": "Enabled",
    }


@pytest.mark.parametrize("mode,days", [("GOVERNANCE", 30), ("COMPLIANCE", 29)])
def test_rejects_weaker_bucket_protection(mode, days):
    with pytest.raises(GatewayRestartPreflightV2Error):
        verify_artifact_bucket_lock_v2(POLICY, s3_client=S3(mode=mode, days=days))

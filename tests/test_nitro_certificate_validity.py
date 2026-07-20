from datetime import datetime, timezone

import pytest

from leadpoet_canonical.nitro import (
    AttestationError,
    _attestation_certificate_validation_time,
    _verify_leaf_certificate_validity,
)


class _Certificate:
    not_valid_before_utc = datetime(2026, 7, 19, 5, 0, tzinfo=timezone.utc)
    not_valid_after_utc = datetime(2026, 7, 19, 8, 24, 4, tzinfo=timezone.utc)


def test_historical_certificate_is_checked_at_signed_attestation_time():
    signed_at = datetime(2026, 7, 19, 6, 0, tzinfo=timezone.utc)
    validation_time = _attestation_certificate_validation_time(
        {"timestamp": int(signed_at.timestamp() * 1000)}
    )

    _verify_leaf_certificate_validity(_Certificate(), validation_time)

    with pytest.raises(AttestationError, match="Certificate expired"):
        _verify_leaf_certificate_validity(
            _Certificate(),
            datetime(2026, 7, 20, 3, 0, tzinfo=timezone.utc),
        )


@pytest.mark.parametrize("timestamp", [None, True, 0, "1752904800000"])
def test_historical_certificate_requires_signed_nitro_timestamp(timestamp):
    with pytest.raises(AttestationError, match="timestamp"):
        _attestation_certificate_validation_time({"timestamp": timestamp})

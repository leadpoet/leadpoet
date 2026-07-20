"""Authority-boundary tests for the V2-only weight publication runtime.

The cryptographic V2 publication, receipt-graph, enclave signing, persistence,
and auditor verification paths are exercised in the dedicated V2 suites. These
tests guard the public boundary that previously allowed rollout-era V1 writes.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException


def test_retired_v1_weight_write_is_unconditionally_gone():
    from gateway.api import weights as weights_api

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(weights_api.submit_weights(None))

    assert exc_info.value.status_code == 410
    assert "/weights/submit/v2" in str(exc_info.value.detail)


def test_primary_and_auditor_have_no_v1_weight_fallback():
    root = Path(__file__).resolve().parents[1]
    primary = (root / "neurons" / "validator.py").read_text(encoding="utf-8")
    auditor = (root / "neurons" / "auditor_validator.py").read_text(
        encoding="utf-8"
    )

    assert "build_legacy_v1_submission" not in primary
    assert "_publish_legacy_v1_bundle" not in primary
    assert "_set_legacy_weights_until_epoch_end" not in primary
    assert "AUDITOR_WEIGHT_PROTOCOL" in auditor
    assert "verify_attested_weights_v1" not in auditor
    assert "fetch_attested_weights_v1" not in auditor
    assert "no fallback vector will be submitted" in auditor

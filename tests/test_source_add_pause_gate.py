"""Maintenance pause must also stop SOURCE_ADD intake.

SOURCE_ADD submissions mint leg-1 emission rewards, so a scoring or
autoresearch maintenance pause has to close the intake door too;
otherwise rewards keep draining the burn share while everything else
is frozen.
"""

import time

import pytest

from gateway.research_lab import api
from gateway.research_lab.models import ResearchLabSourceAdapterSubmissionRequest
from fastapi import HTTPException

from tests.test_source_add_catalog_provisioning import (
    _manifest_doc,
    _source_metadata_doc,
)


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


@pytest.mark.asyncio
async def test_source_adapter_intake_rejected_while_scoring_paused(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    from gateway.research_lab import maintenance

    monkeypatch.setattr(
        maintenance, "is_scoring_maintenance_paused", _async_value(True)
    )
    monkeypatch.setattr(
        maintenance, "is_autoresearch_maintenance_paused", _async_value(False)
    )
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-paused-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )
    with pytest.raises(HTTPException) as exc:
        await api.submit_research_lab_source_adapter(payload)
    assert exc.value.status_code == 503
    assert "paused" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_source_adapter_intake_rejected_while_autoresearch_paused(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(
            lambda: SimpleNamespace(
                api_enabled=True,
                production_writes_enabled=True,
                miner_submissions_enabled=True,
                source_add_enabled=True,
            )
        ),
    )
    from gateway.research_lab import maintenance

    monkeypatch.setattr(
        maintenance, "is_scoring_maintenance_paused", _async_value(False)
    )
    monkeypatch.setattr(
        maintenance, "is_autoresearch_maintenance_paused", _async_value(True)
    )
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-paused-2",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )
    with pytest.raises(HTTPException) as exc:
        await api.submit_research_lab_source_adapter(payload)
    assert exc.value.status_code == 503

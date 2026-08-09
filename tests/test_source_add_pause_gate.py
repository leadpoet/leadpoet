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
async def test_public_status_exposes_effective_source_add_pause(monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(
        api_enabled=False,
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
        public_status=lambda: {
            "source_add": {
                "enabled": True,
                "dispatcher_enabled": True,
            }
        },
    )
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(lambda: config),
    )
    monkeypatch.setattr(
        api,
        "get_autoresearch_maintenance_state",
        _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        _async_value(
            {
                "paused": True,
                "status": "paused",
                "updated_at": "2026-08-08T00:00:00+00:00",
                "unavailable": False,
            }
        ),
    )
    monkeypatch.setattr(
        api,
        "private_repo_head_alignment_status",
        _async_value({"status": "aligned"}),
    )

    status = await api.research_lab_status()

    assert status["source_add"]["control"] == {
        "paused": True,
        "status": "paused",
        "updated_at": "2026-08-08T00:00:00+00:00",
        "unavailable": False,
    }
    assert status["source_add"]["effective_dispatcher_enabled"] is False


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


@pytest.mark.asyncio
async def test_source_adapter_intake_rejected_while_source_add_queue_paused(
    monkeypatch,
):
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
        maintenance, "is_autoresearch_maintenance_paused", _async_value(False)
    )
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        _async_value({"paused": True, "status": "paused"}),
    )
    monkeypatch.setattr(
        api,
        "_verify_signed_miner",
        lambda *_args, **_kwargs: pytest.fail(
            "paused SOURCE_ADD intake must fail before signature work"
        ),
    )
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-paused-3",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    with pytest.raises(HTTPException) as exc:
        await api.submit_research_lab_source_adapter(payload)

    assert exc.value.status_code == 503
    assert exc.value.detail == "SOURCE_ADD workflow is paused"

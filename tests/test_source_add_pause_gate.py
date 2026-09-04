"""SOURCE_ADD intake has an independent, fail-closed maintenance control."""

import time

import pytest

from gateway.research_lab import api
from gateway.research_lab.models import (
    ResearchLabSourceAdapterSubmissionRequest,
)
from fastapi import HTTPException

from tests.test_source_add_catalog_provisioning import (
    _manifest_doc,
    _source_metadata_doc,
)


def _async_value(value):
    async def _inner(*_args, **_kwargs):
        return value

    return _inner


def _status_request(
    *,
    dispatcher_ready: bool = True,
    worker_authority_ready: bool = True,
):
    from types import SimpleNamespace

    dispatcher_task = SimpleNamespace(done=lambda: not dispatcher_ready)
    worker_task = SimpleNamespace(
        done=lambda: worker_authority_ready,
        cancelled=lambda: False,
        exception=lambda: None,
    )
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                source_add_dispatcher_task=dispatcher_task,
                research_lab_worker_startup_task=worker_task,
            )
        )
    )


@pytest.mark.asyncio
async def test_public_status_exposes_effective_source_add_pause(monkeypatch):
    from types import SimpleNamespace

    config = SimpleNamespace(
        api_enabled=False,
        production_writes_enabled=True,
        miner_submissions_enabled=True,
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
    status = await api.research_lab_status(_status_request())

    assert status["source_add"]["control"] == {
        "paused": True,
        "status": "paused",
        "updated_at": "2026-08-08T00:00:00+00:00",
        "unavailable": False,
    }
    assert status["source_add"]["effective_dispatcher_enabled"] is False
    assert status["source_add"]["intake_enabled"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("closed_gate", "closed_value", "expected"),
    (
        (None, None, True),
        ("api_enabled", False, False),
        ("production_writes_enabled", False, False),
        ("miner_submissions_enabled", False, True),
        ("source_add_enabled", False, False),
        ("source_add_dispatcher_enabled", False, False),
        ("source_add_dispatcher_ready", False, False),
        ("worker_authority_ready", False, True),
        ("autoresearch_paused", True, True),
        ("source_add_paused", True, False),
    ),
)
async def test_public_status_source_add_intake_gate_matches_admission_state(
    monkeypatch,
    closed_gate,
    closed_value,
    expected,
):
    from types import SimpleNamespace

    values = {
        "api_enabled": True,
        "production_writes_enabled": True,
        "miner_submissions_enabled": True,
        "source_add_enabled": True,
        "source_add_dispatcher_enabled": True,
        "source_add_dispatcher_ready": True,
        "worker_authority_ready": True,
        "autoresearch_paused": False,
        "source_add_paused": False,
    }
    if closed_gate is not None:
        values[closed_gate] = closed_value
    config = SimpleNamespace(
        api_enabled=values["api_enabled"],
        production_writes_enabled=values["production_writes_enabled"],
        miner_submissions_enabled=values["miner_submissions_enabled"],
        source_add_enabled=values["source_add_enabled"],
        source_add_dispatcher_enabled=values[
            "source_add_dispatcher_enabled"
        ],
        reports_enabled=False,
        public_status=lambda: {
            "source_add": {
                "enabled": values["source_add_enabled"],
                "dispatcher_enabled": values[
                    "source_add_dispatcher_enabled"
                ],
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
        "source_add_control_state",
        _async_value({"paused": values["source_add_paused"]}),
    )
    status = await api.research_lab_status(
        _status_request(
            dispatcher_ready=values["source_add_dispatcher_ready"],
            worker_authority_ready=values["worker_authority_ready"],
        )
    )

    assert status["source_add"]["intake_enabled"] is expected


@pytest.mark.asyncio
async def test_public_status_source_add_intake_fails_closed_without_any_authority(
    monkeypatch,
):
    from types import SimpleNamespace

    config = SimpleNamespace(
        api_enabled=True,
        production_writes_enabled=True,
        miner_submissions_enabled=False,
        source_add_enabled=True,
        source_add_dispatcher_enabled=True,
        reports_enabled=False,
        public_status=lambda: {
            "source_add": {"enabled": True, "dispatcher_enabled": True}
        },
    )
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        staticmethod(lambda: config),
    )
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        _async_value({"paused": False}),
    )
    status = await api.research_lab_status(
        _status_request(
            dispatcher_ready=False,
            worker_authority_ready=False,
        )
    )

    assert status["source_add"]["effective_dispatcher_enabled"] is False
    assert status["source_add"]["intake_enabled"] is False


@pytest.mark.asyncio
async def test_source_adapter_intake_remains_open_when_miner_submissions_are_disabled(
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
                miner_submissions_enabled=False,
                source_add_enabled=True,
                source_add_max_concurrent_per_hotkey=3,
                source_add_max_per_day_per_hotkey=5,
                source_add_max_per_30d_per_hotkey=10,
            )
        ),
    )
    monkeypatch.setattr(
        api,
        "source_add_control_state",
        _async_value({"paused": False, "status": "active"}),
    )
    monkeypatch.setattr(api, "_verify_signed_miner", _async_value(None))
    monkeypatch.setattr(
        api.source_add_catalog_contract,
        "source_add_api_is_current_builtin_sync",
        lambda *_args, **_kwargs: False,
    )

    async def admitted(_name, _params):
        return {"status": "admitted", "stage": "provenance_queued"}

    monkeypatch.setattr(api, "_source_add_rpc", admitted)
    payload = ResearchLabSourceAdapterSubmissionRequest(
        miner_hotkey="miner-hotkey-value",
        signature="signature-value-123",
        timestamp=int(time.time()),
        idempotency_key="source-submit-independent-1",
        manifest=_manifest_doc(),
        source_metadata=_source_metadata_doc(),
    )

    response = await api.submit_research_lab_source_adapter(payload)

    assert response.stage == "provenance_queued"


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

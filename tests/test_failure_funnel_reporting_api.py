"""Route-level coverage for optional failure-funnel reporting."""

from __future__ import annotations

import time
from types import SimpleNamespace
from uuid import UUID

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from gateway.research_lab import api, failure_funnel


TICKET_ID = UUID("11111111-1111-4111-8111-111111111111")


def _enable_reports(monkeypatch) -> None:
    config = SimpleNamespace(api_enabled=True, reports_enabled=True)
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        classmethod(lambda _cls: config),
    )


async def _none(*_args, **_kwargs):
    return None


def _full_telemetry(status="partial", **overrides):
    telemetry = {
        field: 0 for field in failure_funnel._TELEMETRY_COUNT_FIELDS
    }
    telemetry.update(overrides)
    return {"status": status, **telemetry}


async def _post_loop_diagnostics(**overrides):
    app = FastAPI()
    app.include_router(api.router)
    payload = {
        "ticket_id": str(TICKET_ID),
        "candidate_id": "candidate-1",
        "miner_hotkey": "miner-hotkey-0001",
        "signature": "signature-0000001",
        "timestamp": int(time.time()),
        "idempotency_key": "diagnostics-request-1",
    }
    payload.update(overrides)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        return await client.post("/research-lab/loop-diagnostics", json=payload)


async def test_miner_loop_diagnostics_returns_sanitized_failure_funnel(monkeypatch):
    _enable_reports(monkeypatch)
    ownership_checks: list[tuple[str, str]] = []

    async def get_owned_ticket(ticket_id: str, miner_hotkey: str):
        ownership_checks.append((ticket_id, miner_hotkey))
        return {"ticket_id": ticket_id}

    async def failure_report(name, params):
        assert name == "get_research_lab_failure_funnel"
        assert params == {
            "p_ticket_id": str(TICKET_ID),
            "p_candidate_id": "candidate-1",
        }
        return {
            "schema_version": "research_lab_failure_funnel.v1",
            "ticket_id": str(TICKET_ID),
            "candidate_id": "candidate-1",
            "stages": [
                {
                    "stage": "sourcing",
                    "unit": "icp_attempts",
                    "reviewed": 1,
                    "passed": 0,
                    "rejected": 1,
                }
            ],
            "rejections": [
                {
                    "stage": "provider_transport",
                    "reason_code": "provider_timeout",
                    "unit": "icp_attempts",
                    "count": 1,
                }
            ],
            "model_revisions": ["sha256:" + "a" * 64],
            "telemetry": _full_telemetry(infrastructure_failure_count=1),
        }

    monkeypatch.setattr(api, "_verify_signed_miner", _none)
    monkeypatch.setattr(api, "_get_ticket_for_miner", get_owned_ticket)
    monkeypatch.setattr(
        api, "_build_ticket_candidate_diagnostics", lambda *_args: _async_value([{"ok": True}])
    )
    monkeypatch.setattr(api, "_build_ticket_run_summaries", lambda *_args: _async_value([]))
    monkeypatch.setattr(failure_funnel, "call_rpc", failure_report)

    response = await _post_loop_diagnostics()

    assert response.status_code == 200
    response_doc = response.json()
    assert ownership_checks == [(str(TICKET_ID), "miner-hotkey-0001")]
    assert "model_revisions" not in response_doc["failure_funnel"]
    assert response_doc["failure_funnel"]["rejections"] == [
        {
            "stage": "infrastructure",
            "reason_code": "external_service_failure",
            "unit": "icp_attempts",
            "count": 1,
        }
    ]


async def test_miner_loop_diagnostics_survives_optional_report_unavailability(monkeypatch):
    _enable_reports(monkeypatch)
    monkeypatch.setattr(api, "_verify_signed_miner", _none)
    monkeypatch.setattr(api, "_get_ticket_for_miner", lambda *_args: _async_value({}))
    monkeypatch.setattr(
        api, "_build_ticket_candidate_diagnostics", lambda *_args: _async_value([{"ok": True}])
    )
    monkeypatch.setattr(api, "_build_ticket_run_summaries", lambda *_args: _async_value([]))
    monkeypatch.setattr(
        failure_funnel,
        "call_rpc",
        lambda *_args, **_kwargs: _raise_async(RuntimeError("rpc unavailable")),
    )

    response = await _post_loop_diagnostics(candidate_id=None)

    assert response.status_code == 200
    response_doc = response.json()
    assert response_doc["diagnostics"] == [{"ok": True}]
    assert response_doc["failure_funnel"]["telemetry"] == {
        "status": "missing",
        "report_available": False,
    }


async def test_miner_loop_diagnostics_checks_ownership_before_report(monkeypatch):
    _enable_reports(monkeypatch)
    report_called = False

    async def reject_unowned_ticket(*_args, **_kwargs):
        raise HTTPException(status_code=404, detail="ticket not found")

    async def failure_report(*_args, **_kwargs):
        nonlocal report_called
        report_called = True
        return {}

    monkeypatch.setattr(api, "_verify_signed_miner", _none)
    monkeypatch.setattr(api, "_get_ticket_for_miner", reject_unowned_ticket)
    monkeypatch.setattr(api, "build_ticket_failure_funnel", failure_report)

    response = await _post_loop_diagnostics()

    assert response.status_code == 404
    assert report_called is False


async def test_miner_loop_diagnostics_preserves_no_terminal_data_404(monkeypatch):
    _enable_reports(monkeypatch)
    report_called = False

    async def failure_report(*_args, **_kwargs):
        nonlocal report_called
        report_called = True
        return {}

    monkeypatch.setattr(api, "_verify_signed_miner", _none)
    monkeypatch.setattr(api, "_get_ticket_for_miner", lambda *_args: _async_value({}))
    monkeypatch.setattr(
        api, "_build_ticket_candidate_diagnostics", lambda *_args: _async_value([])
    )
    monkeypatch.setattr(api, "_build_ticket_run_summaries", lambda *_args: _async_value([]))
    monkeypatch.setattr(api, "build_ticket_failure_funnel", failure_report)

    response = await _post_loop_diagnostics()

    assert response.status_code == 404
    assert report_called is False


async def test_admin_loop_diagnostics_survives_optional_report_unavailability(monkeypatch):
    _enable_reports(monkeypatch)
    monkeypatch.setattr(api, "_require_internal_key", lambda *_args: None)
    monkeypatch.setattr(
        api,
        "fetch_public_loop_detail",
        lambda *_args: _async_value({"card": {"ticket_id": str(TICKET_ID)}, "events": []}),
    )
    monkeypatch.setattr(
        api, "_build_ticket_candidate_diagnostics", lambda *_args: _async_value([{"ok": True}])
    )
    monkeypatch.setattr(
        api,
        "build_ticket_failure_funnel",
        lambda *_args: _async_value(
            {
                "schema_version": "research_lab_failure_funnel.v1",
                "ticket_id": str(TICKET_ID),
                "candidate_id": None,
                "stages": [],
                "rejections": [],
                "model_revisions": [],
                "telemetry": {"status": "missing", "report_available": False},
            }
        ),
    )

    response = await api.get_research_lab_admin_loop_diagnostics(
        str(TICKET_ID), candidate_id=None, x_leadpoet_internal_key="internal"
    )

    assert response["candidate_diagnostics"] == [{"ok": True}]
    assert response["failure_funnel"]["telemetry"]["status"] == "missing"


async def test_admin_optional_report_preserves_existing_storage_read_order(monkeypatch):
    _enable_reports(monkeypatch)
    calls = []

    async def detail(*_args):
        calls.append("detail")
        return {"card": {"ticket_id": str(TICKET_ID)}, "events": []}

    async def diagnostics(*_args):
        calls.append("diagnostics")
        return [{"ok": True}]

    async def report(*_args):
        calls.append("failure_funnel")
        return {
            "telemetry": {"status": "missing", "report_available": False}
        }

    monkeypatch.setattr(api, "_require_internal_key", lambda *_args: None)
    monkeypatch.setattr(api, "fetch_public_loop_detail", detail)
    monkeypatch.setattr(api, "_build_ticket_candidate_diagnostics", diagnostics)
    monkeypatch.setattr(api, "build_ticket_failure_funnel", report)

    await api.get_research_lab_admin_loop_diagnostics(
        str(TICKET_ID), candidate_id=None, x_leadpoet_internal_key="internal"
    )

    assert calls == ["detail", "diagnostics", "failure_funnel"]


async def _async_value(value):
    return value


async def _raise_async(exc):
    raise exc

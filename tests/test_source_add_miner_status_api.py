"""Private miner-facing SOURCE_ADD status coverage."""

from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path
import time
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
import pytest
from pydantic import ValidationError

from gateway.research_lab import api
from gateway.research_lab.models import (
    ResearchLabSourceAddStatusItem,
    ResearchLabSourceAddStatusResponse,
)


ROOT = Path(__file__).resolve().parents[1]
MINER_HOTKEY = "5" + "A" * 47
OTHER_HOTKEY = "5" + "B" * 47
SUBMISSION_ONE = "source_add_submission:" + "1" * 16
SUBMISSION_TWO = "source_add_submission:" + "2" * 16
SUBMISSION_THREE = "source_add_submission:" + "3" * 16


def _status_row(
    *,
    submission_id: str = SUBMISSION_ONE,
    miner_hotkey: str = MINER_HOTKEY,
    source_name: str = "BuiltWith Trends API",
    decision_status: str = "pending",
    decision_reason_code: str = "automated_checks_in_progress",
    decision_reason: str = "Automated Source Add checks are still in progress.",
    reward_status: str = "not_decided",
    alpha_percent=None,
    reward_epochs=None,
    start_epoch=None,
    end_epoch=None,
    **extra,
):
    return {
        "schema_version": "leadpoet.source_add_miner_status.v1",
        "submission_id": submission_id,
        "miner_hotkey": miner_hotkey,
        "source_name": source_name,
        "submitted_at": "2026-09-03T14:00:00Z",
        "updated_at": "2026-09-03T14:01:00Z",
        "decision_status": decision_status,
        "decision_reason_code": decision_reason_code,
        "decision_reason": decision_reason,
        "reward_status": reward_status,
        "alpha_percent": alpha_percent,
        "reward_epochs": reward_epochs,
        "start_epoch": start_epoch,
        "end_epoch": end_epoch,
        **extra,
    }


def _request_body(**overrides):
    body = {
        "miner_hotkey": MINER_HOTKEY,
        "signature": "signed-status-request",
        "timestamp": int(time.time()),
        "idempotency_key": "source-add-status-request-1",
        "request_kind": "source_add_status_v1",
        "limit": 20,
    }
    body.update(overrides)
    return body


def _enable_api(monkeypatch):
    # The read path intentionally has no SOURCE_ADD pause/dispatcher flag.
    config = SimpleNamespace(api_enabled=True, source_add_enabled=False)
    monkeypatch.setattr(
        api.ResearchLabGatewayConfig,
        "from_env",
        classmethod(lambda _cls: config),
    )


async def _post_status(body):
    app = FastAPI()
    app.include_router(api.router)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await client.post(
            "/research-lab/source-adapters/status",
            json=body,
        )


@pytest.mark.asyncio
async def test_status_uses_verified_signer_as_only_owner_filter_and_allowlists_response(
    monkeypatch,
):
    _enable_api(monkeypatch)
    calls = []

    async def verify(payload):
        calls.append(("verify", payload.miner_hotkey))

    async def call_rpc(name, params):
        calls.append(("rpc", name, params))
        return [
            _status_row(
                private_reason_codes=["duplicate_source"],
                adapter_id="adapter:private",
                source_identity_hash="sha256:" + "f" * 64,
                api_base_url="https://must-not-be-returned.example",
            )
        ]

    monkeypatch.setattr(api, "_verify_signed_miner", verify)
    monkeypatch.setattr(api, "call_rpc", call_rpc)

    response = await _post_status(_request_body())

    assert response.status_code == 200
    assert calls == [
        ("verify", MINER_HOTKEY),
        (
            "rpc",
            "research_lab_source_add_miner_status_page_v1",
            {
                "p_miner_hotkey": MINER_HOTKEY,
                "p_cursor_submission_id": None,
                "p_limit": 20,
            },
        ),
    ]
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["pragma"] == "no-cache"
    document = response.json()
    assert set(document) == {"schema_version", "submissions", "next_cursor"}
    assert set(document["submissions"][0]) == {
        "submission_id",
        "source_name",
        "submitted_at",
        "updated_at",
        "decision_status",
        "decision_reason_code",
        "decision_reason",
        "reward_status",
        "alpha_percent",
        "reward_epochs",
        "start_epoch",
        "end_epoch",
    }
    assert "duplicate_source" not in response.text
    assert "adapter:private" not in response.text
    assert "must-not-be-returned" not in response.text
    assert MINER_HOTKEY not in response.text


@pytest.mark.asyncio
async def test_status_fails_closed_if_storage_returns_another_miner(monkeypatch):
    _enable_api(monkeypatch)

    async def verify(_payload):
        return None

    async def call_rpc(_name, _params):
        return [
            _status_row(
                miner_hotkey=OTHER_HOTKEY,
                source_name="Other miner private source",
            )
        ]

    monkeypatch.setattr(api, "_verify_signed_miner", verify)
    monkeypatch.setattr(api, "call_rpc", call_rpc)

    response = await _post_status(_request_body())

    assert response.status_code == 503
    assert response.json() == {
        "detail": "SOURCE_ADD status is temporarily unavailable"
    }
    assert "Other miner private source" not in response.text
    assert OTHER_HOTKEY not in response.text


@pytest.mark.asyncio
async def test_status_rejects_signature_before_any_status_read(monkeypatch):
    _enable_api(monkeypatch)
    storage_called = False

    async def reject(_payload):
        raise HTTPException(status_code=401, detail="invalid miner hotkey signature")

    async def call_rpc(_name, _params):
        nonlocal storage_called
        storage_called = True
        return []

    monkeypatch.setattr(api, "_verify_signed_miner", reject)
    monkeypatch.setattr(api, "call_rpc", call_rpc)

    response = await _post_status(_request_body())

    assert response.status_code == 401
    assert storage_called is False


@pytest.mark.asyncio
async def test_status_keyset_pagination_is_bounded_and_cursor_is_signed_input(
    monkeypatch,
):
    _enable_api(monkeypatch)
    rpc_params = []

    async def verify(_payload):
        return None

    async def call_rpc(_name, params):
        rpc_params.append(params)
        if params["p_cursor_submission_id"] is None:
            return [
                _status_row(submission_id=SUBMISSION_ONE),
                _status_row(submission_id=SUBMISSION_TWO),
                _status_row(submission_id=SUBMISSION_THREE),
            ]
        return [_status_row(submission_id=SUBMISSION_THREE)]

    monkeypatch.setattr(api, "_verify_signed_miner", verify)
    monkeypatch.setattr(api, "call_rpc", call_rpc)

    first = await _post_status(_request_body(limit=2))
    second = await _post_status(
        _request_body(
            limit=2,
            cursor=first.json()["next_cursor"],
            idempotency_key="source-add-status-request-2",
        )
    )

    assert first.status_code == 200
    assert [item["submission_id"] for item in first.json()["submissions"]] == [
        SUBMISSION_ONE,
        SUBMISSION_TWO,
    ]
    assert first.json()["next_cursor"] == SUBMISSION_TWO
    assert second.status_code == 200
    assert [item["submission_id"] for item in second.json()["submissions"]] == [
        SUBMISSION_THREE
    ]
    assert second.json()["next_cursor"] is None
    assert rpc_params == [
        {
            "p_miner_hotkey": MINER_HOTKEY,
            "p_cursor_submission_id": None,
            "p_limit": 2,
        },
        {
            "p_miner_hotkey": MINER_HOTKEY,
            "p_cursor_submission_id": SUBMISSION_TWO,
            "p_limit": 2,
        },
    ]


def test_status_models_round_trip_and_reject_inconsistent_public_state():
    item = ResearchLabSourceAddStatusItem.model_validate(
        _status_row(
            miner_hotkey=MINER_HOTKEY,
            decision_status="approved",
            decision_reason_code="leg1_reward_active",
            decision_reason=(
                "The source passed automated checks and the Leg 1 reward is active."
            ),
            reward_status="active",
            alpha_percent=0.2,
            reward_epochs=20,
            start_epoch=24942,
            end_epoch=24961,
        )
    )
    response = ResearchLabSourceAddStatusResponse(
        schema_version="leadpoet.source_add_miner_status.v1",
        submissions=[item],
        next_cursor=None,
    )
    dumped = response.model_dump(mode="json")
    assert ResearchLabSourceAddStatusResponse.model_validate(dumped) == response

    with pytest.raises(ValidationError, match="decision fields are inconsistent"):
        ResearchLabSourceAddStatusItem.model_validate(
            {
                **item.model_dump(mode="json"),
                "decision_status": "rejected",
            }
        )
    with pytest.raises(ValidationError, match="reward status is inconsistent"):
        ResearchLabSourceAddStatusItem.model_validate(
            {
                **item.model_dump(mode="json"),
                "reward_status": "pending",
                "alpha_percent": None,
                "reward_epochs": None,
                "start_epoch": None,
                "end_epoch": None,
            }
        )


def _load_miner_function(name: str, namespace: dict):
    source_path = ROOT / "neurons" / "miner.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    node = next(
        item
        for item in tree.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and item.name == name
    )
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace[name]


def test_miner_status_flow_signs_pages_prints_safe_results_and_advances_cursor(
    monkeypatch,
    capsys,
):
    signed_messages = []
    requests = []

    class Hotkey:
        ss58_address = MINER_HOTKEY

        @staticmethod
        def sign(message):
            signed_messages.append(message.decode("utf-8"))
            return b"signed"

    wallet = SimpleNamespace(hotkey=Hotkey())
    pages = [
        {
            "schema_version": "leadpoet.source_add_miner_status.v1",
            "submissions": [
                {
                    "submission_id": SUBMISSION_ONE,
                    "source_name": "BuiltWith Trends API",
                    "submitted_at": "2026-09-03T14:00:00Z",
                    "decision_status": "approved",
                    "decision_reason": (
                        "The source passed automated checks and the Leg 1 reward is active."
                    ),
                    "reward_status": "active",
                    "alpha_percent": 0.2,
                    "reward_epochs": 20,
                    "start_epoch": 24942,
                    "end_epoch": 24961,
                    "api_base_url": "https://private.example",
                    "private_reason_codes": ["must-not-print"],
                }
            ],
            "next_cursor": SUBMISSION_ONE,
        },
        {
            "schema_version": "leadpoet.source_add_miner_status.v1",
            "submissions": [
                {
                    "submission_id": SUBMISSION_TWO,
                    "source_name": "Second API",
                    "submitted_at": "2026-09-02T14:00:00Z",
                    "decision_status": "rejected",
                    "decision_reason": (
                        "The submission did not pass automated Source Add checks."
                    ),
                    "reward_status": "not_eligible",
                }
            ],
            "next_cursor": None,
        },
    ]

    def post(path, payload, *, timeout):
        requests.append((path, payload, timeout))
        return pages.pop(0)

    namespace = {
        "time": time,
        "_post_research_lab_json": post,
        "json": json,
    }
    _load_miner_function("_research_lab_signed_payload", namespace)
    flow = _load_miner_function(
        "run_research_lab_source_add_status_flow",
        namespace,
    )
    monkeypatch.setattr(builtins, "input", lambda _prompt: "y")

    flow(wallet, SimpleNamespace(), 71)

    output = capsys.readouterr().out
    assert "APPROVED: BuiltWith Trends API" in output
    assert "0.2% per epoch for 20 epochs (24942–24961; active)" in output
    assert "REJECTED: Second API" in output
    assert "private.example" not in output
    assert "must-not-print" not in output
    assert [request[0] for request in requests] == [
        "/research-lab/source-adapters/status",
        "/research-lab/source-adapters/status",
    ]
    assert requests[0][1]["request_kind"] == "source_add_status_v1"
    assert "cursor" not in requests[0][1]
    assert requests[1][1]["cursor"] == SUBMISSION_ONE
    assert requests[0][1]["signature"] == b"signed".hex()
    assert json.loads(signed_messages[0])["request_kind"] == "source_add_status_v1"
    assert json.loads(signed_messages[1])["cursor"] == SUBMISSION_ONE


def test_miner_status_flow_does_not_echo_gateway_error(monkeypatch, capsys):
    class Hotkey:
        ss58_address = MINER_HOTKEY

        @staticmethod
        def sign(_message):
            return b"signed"

    namespace = {
        "time": time,
        "json": json,
        "_post_research_lab_json": lambda *_args, **_kwargs: {
            "status_code": 503,
            "error": "other-miner-private-source should never print",
        },
    }
    _load_miner_function("_research_lab_signed_payload", namespace)
    flow = _load_miner_function(
        "run_research_lab_source_add_status_flow",
        namespace,
    )

    flow(SimpleNamespace(hotkey=Hotkey()), SimpleNamespace(), 71)

    output = capsys.readouterr().out
    assert "SOURCE_ADD status failed: HTTP 503" in output
    assert "Submission status is temporarily unavailable." in output
    assert "other-miner-private-source" not in output

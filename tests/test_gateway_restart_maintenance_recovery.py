from __future__ import annotations

import argparse

import pytest

from gateway.research_lab import admin
from gateway.research_lab import maintenance
from gateway.research_lab import store


def _pause_state(
    *,
    control_key: str,
    action: str,
    seq: int,
    actor_ref: str = maintenance.GATEWAY_RESTART_ACTOR_REF,
    reason: str = maintenance.GATEWAY_RESTART_PAUSE_REASON,
) -> dict:
    return {
        "control_key": control_key,
        "paused": True,
        "status": "active",
        "event_type": "pause_requested",
        "reason": reason,
        "actor_ref": actor_ref,
        "event_seq": seq,
        "event_hash": f"sha256:{seq:064x}",
        "event_doc": {"operator_action": action},
    }


@pytest.mark.asyncio
async def test_restart_recovery_preserves_owned_autoresearch_and_scoring_pauses(
    monkeypatch,
) -> None:
    autoresearch = _pause_state(
        control_key=maintenance.AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
        action="pause-autoresearch",
        seq=8,
    )
    scoring = _pause_state(
        control_key=maintenance.SCORING_MAINTENANCE_CONTROL_KEY,
        action="pause-scoring",
        seq=12,
    )

    async def get_autoresearch():
        return autoresearch

    async def get_scoring():
        return scoring

    async def unexpected_write(**_kwargs):
        raise AssertionError("gateway restart must not change maintenance state")

    monkeypatch.setattr(
        maintenance, "get_autoresearch_maintenance_state", get_autoresearch
    )
    monkeypatch.setattr(maintenance, "get_scoring_maintenance_state", get_scoring)
    monkeypatch.setattr(
        maintenance,
        "set_autoresearch_maintenance_paused",
        unexpected_write,
    )
    monkeypatch.setattr(
        maintenance, "set_scoring_maintenance_paused", unexpected_write
    )
    monkeypatch.setattr(
        maintenance,
        "requeue_paused_autoresearch_runs",
        lambda **_kwargs: pytest.fail(
            "gateway restart must not requeue autoresearch runs"
        ),
    )

    result = await maintenance.resume_gateway_restart_owned_maintenance()

    assert result["ok"] is True
    assert result["autoresearch"]["status"] == "preserved_restart_pause"
    assert result["autoresearch"]["state"] == autoresearch
    assert result["scoring"]["status"] == "preserved_restart_pause"
    assert result["scoring"]["state"] == scoring


@pytest.mark.asyncio
async def test_restart_recovery_preserves_inactive_controls_without_writes(
    monkeypatch,
) -> None:
    autoresearch = {"paused": False, "status": "inactive", "event_seq": 9}
    scoring = {"paused": False, "status": "inactive", "event_seq": 13}

    async def unexpected_write(**_kwargs):
        raise AssertionError("gateway restart must not change maintenance state")

    monkeypatch.setattr(
        maintenance,
        "get_autoresearch_maintenance_state",
        lambda: _async_value(autoresearch),
    )
    monkeypatch.setattr(
        maintenance,
        "get_scoring_maintenance_state",
        lambda: _async_value(scoring),
    )
    monkeypatch.setattr(
        maintenance, "set_autoresearch_maintenance_paused", unexpected_write
    )
    monkeypatch.setattr(
        maintenance, "set_scoring_maintenance_paused", unexpected_write
    )

    result = await maintenance.resume_gateway_restart_owned_maintenance()

    assert result == {
        "ok": True,
        "action": "resume-restart-maintenance",
        "autoresearch": {"status": "already_inactive"},
        "scoring": {"status": "already_inactive"},
    }


@pytest.mark.asyncio
async def test_restart_recovery_preserves_operator_and_provider_pauses(
    monkeypatch,
) -> None:
    autoresearch = _pause_state(
        control_key=maintenance.AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
        action="pause-autoresearch",
        seq=8,
        actor_ref="operator:human",
        reason="investigation",
    )
    scoring = _pause_state(
        control_key=maintenance.SCORING_MAINTENANCE_CONTROL_KEY,
        action="pause-scoring",
        seq=12,
        actor_ref="provider-preflight",
        reason="provider_preflight:openrouter",
    )

    async def unexpected_write(**_kwargs):
        raise AssertionError("non-restart pause must not be changed")

    monkeypatch.setattr(
        maintenance,
        "get_autoresearch_maintenance_state",
        lambda: _async_value(autoresearch),
    )
    monkeypatch.setattr(
        maintenance,
        "get_scoring_maintenance_state",
        lambda: _async_value(scoring),
    )
    monkeypatch.setattr(
        maintenance, "set_autoresearch_maintenance_paused", unexpected_write
    )
    monkeypatch.setattr(
        maintenance, "set_scoring_maintenance_paused", unexpected_write
    )

    result = await maintenance.resume_gateway_restart_owned_maintenance()

    assert result["ok"] is True
    assert result["autoresearch"]["status"] == "preserved_non_restart_pause"
    assert result["scoring"]["status"] == "preserved_non_restart_pause"


async def _async_value(value):
    return value


def test_restart_recovery_cli_requires_exact_commit() -> None:
    with pytest.raises(SystemExit):
        admin.build_parser().parse_args(["resume-restart-maintenance"])


@pytest.mark.parametrize(
    ("command", "action", "extra_arguments"),
    (
        (
            "pause-source-add",
            "pause",
            ["--reason", "gateway_restart"],
        ),
        ("resume-source-add", "resume", []),
    ),
)
def test_source_add_control_aliases_apply_immediately(
    command: str,
    action: str,
    extra_arguments: list[str],
) -> None:
    parsed = admin.build_parser().parse_args(
        [command, *extra_arguments, "--actor-ref", "operator:gateway-restart"]
    )

    assert parsed.command == command
    assert parsed.source_add_command == action
    assert parsed.apply is True
    assert parsed.actor_ref == "operator:gateway-restart"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "action", "paused"),
    (
        ("pause-source-add", "pause", True),
        ("resume-source-add", "resume", False),
    ),
)
async def test_source_add_control_aliases_only_change_source_add(
    monkeypatch,
    command: str,
    action: str,
    paused: bool,
) -> None:
    calls: list[tuple[str, dict]] = []

    async def call_rpc(name: str, parameters: dict):
        calls.append((name, parameters))
        return [{"paused": paused, "reason": "gateway_restart"}]

    async def select_one(*_args, **_kwargs):
        return {"paused": not paused, "reason": "prior_state"}

    monkeypatch.setattr(admin, "call_rpc", call_rpc)
    monkeypatch.setattr(admin, "select_one", select_one)
    result = await admin._run(
        argparse.Namespace(
            command=command,
            source_add_command=action,
            reason="gateway_restart",
            actor_ref="operator:gateway-restart",
            apply=True,
        )
    )

    assert calls == [
        (
            "research_lab_source_add_set_paused",
            {
                "p_paused": paused,
                "p_reason": "gateway_restart",
                "p_actor_ref": "operator:gateway-restart",
            },
        )
    ]
    assert result["ok"] is True
    assert result["action"] == f"source-add {action}"
    assert result["dry_run"] is False


@pytest.mark.asyncio
async def test_restart_recovery_cli_rejects_wrong_runtime_commit(
    monkeypatch,
) -> None:
    expected = "a" * 40
    monkeypatch.setattr(
        admin,
        "get_build_info",
        lambda: {"git_commit": "b" * 40},
    )

    result = await admin._run(
        argparse.Namespace(
            command="resume-restart-maintenance",
            expected_commit=expected,
        )
    )

    assert result == {
        "ok": False,
        "action": "resume-restart-maintenance",
        "blocked_reason": "exact_commit_mismatch",
        "expected_commit": expected,
        "actual_commit": "b" * 40,
    }


@pytest.mark.asyncio
async def test_restart_recovery_cli_preserves_controls_for_exact_runtime_commit(
    monkeypatch,
) -> None:
    expected = "a" * 40
    preserved = {
        "ok": True,
        "action": "resume-restart-maintenance",
        "autoresearch": {"status": "preserved_non_restart_pause"},
        "scoring": {"status": "preserved_non_restart_pause"},
    }
    calls = 0

    async def preserve_controls():
        nonlocal calls
        calls += 1
        return preserved

    monkeypatch.setattr(admin, "get_build_info", lambda: {"git_commit": expected})
    monkeypatch.setattr(
        admin, "resume_gateway_restart_owned_maintenance", preserve_controls
    )

    result = await admin._run(
        argparse.Namespace(
            command="resume-restart-maintenance",
            expected_commit=expected,
        )
    )

    assert calls == 1
    assert result == {**preserved, "verified_commit": expected}


@pytest.mark.asyncio
async def test_restart_recovery_control_write_rejects_changed_sequence(
    monkeypatch,
) -> None:
    async def append_event(
        _table,
        _key_field,
        _key_value,
        build_payload,
        *,
        attempts,
    ):
        assert attempts == 1
        return build_payload(11)

    monkeypatch.setattr(store, "append_event_with_seq", append_event)

    with pytest.raises(
        RuntimeError,
        match="gateway control state changed before the requested transition",
    ):
        await store.create_gateway_control_event(
            control_key=maintenance.AUTORESEARCH_MAINTENANCE_CONTROL_KEY,
            event_type="resume_requested",
            control_status="inactive",
            reason=maintenance.GATEWAY_RESTART_RESUME_REASON,
            expected_prior_seq=8,
        )

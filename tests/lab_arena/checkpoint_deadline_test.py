"""Signed lease limits and checkpoint completion stay bound to the runner path."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from lab_arena import contracts, runner, runtime, scoring
from lab_arena.service import ArenaService, RoundDefaults


class _Cache:
    def __init__(self, root: Path, *, source: bool = False) -> None:
        self.root = root
        self.source = source

    @contextmanager
    def acquire(self, *_args, **_kwargs):
        if self.source:
            source = self.root / "source"
            deps = self.root / "deps"
            source.mkdir(parents=True, exist_ok=True)
            deps.mkdir(parents=True, exist_ok=True)
            yield source, deps
        else:
            self.root.mkdir(exist_ok=True)
            yield self.root


class _Api:
    def provider(self, *_args):
        raise AssertionError("no provider call expected")


def _execute(tmp_path, lease, result, *, local_wall=30):
    runtime_fake = runtime.FakeRuntime([result])
    work = tmp_path / "work"
    work.mkdir()
    with tempfile.TemporaryDirectory(prefix="lacp-", dir="/tmp") as short_socket_root:
        config = runner.RunnerConfig(
            round_id="arena-2026-09-15",
            identity=runner.RunnerIdentity(hotkey="5" * 48, sign=lambda _msg: "sig"),
            api=_Api(), sandbox_runtime=runtime_fake,
            image_cache=_Cache(tmp_path / "rootfs"),
            source_cache=_Cache(tmp_path / "cache", source=True),
            work_dir=work, socket_root=Path(short_socket_root),
            wall_clock_seconds=local_wall,
        )
        envelope = runner.AssignmentExecutor(config).execute(
            lease, "lease-token", {"max_companies": 5},
        )
    return envelope["body"]["result"], runtime_fake.specs[0]


def _lease(*, checkpoint=True, duration=2700):
    digest = "sha256:" + "a" * 64
    lease = {
        "round_id": "arena-2026-09-15", "run_id": "run-1",
        "assignment_id": "assignment-1", "kind": "execute",
        "image_digest": digest,
        "image_reference": "registry.example/scorer@" + digest,
        "source_ref": "source-1", "submission_id": "submission-1",
        "source_size_bytes": 1, "evaluation_date": "2026-09-15",
        "icp_wall_clock_seconds": duration,
    }
    if checkpoint:
        lease["checkpoint_deadline_policy"] = contracts.CHECKPOINT_DEADLINE_POLICY
        lease["lease_ttl_seconds"] = contracts.CHECKPOINT_LEASE_TTL_SECONDS
    return lease


def test_signed_checkpoint_duration_overrides_local_runner_and_accepts_timeout_snapshot(tmp_path):
    result, spec = _execute(
        tmp_path, _lease(), runtime.fake_result(
            timed_out=True, exit_code=None, output_bytes=b'{"companies":[]}',
        ), local_wall=30,
    )
    assert spec.wall_clock_seconds == 2700
    assert spec.checkpoint_deadline_policy == contracts.CHECKPOINT_DEADLINE_POLICY
    assert spec.checkpoint_validator(b'{"companies":[]}')
    assert not spec.checkpoint_validator(b'{"companies":[{"bad":true}]}')
    assert result["terminal_status"] == "accepted"


def test_new_checkpoint_lease_missing_signed_duration_fails_closed(tmp_path):
    with pytest.raises(runner.RunnerError, match="signed round"):
        _execute(
            tmp_path, _lease(duration=300),
            runtime.fake_result(output_bytes=b'{"companies":[]}'),
        )


def test_historical_signed_duration_preserved_and_timeout_stays_failure(tmp_path):
    result, spec = _execute(
        tmp_path, _lease(checkpoint=False, duration=123),
        runtime.fake_result(
            timed_out=True, exit_code=None, output_bytes=b'{"companies":[]}',
        ), local_wall=30,
    )
    assert spec.wall_clock_seconds == 123
    assert spec.checkpoint_deadline_policy is None
    assert result["terminal_status"] == "model_timeout"


def test_checkpoint_timeout_without_valid_output_stays_failure(tmp_path):
    result, _spec = _execute(
        tmp_path, _lease(),
        runtime.fake_result(timed_out=True, exit_code=None, output_bytes=None),
    )
    assert result["terminal_status"] == "model_timeout"


def test_nonzero_exit_with_complete_checkpoint_keeps_existing_output_rule(tmp_path):
    result, _spec = _execute(
        tmp_path, _lease(),
        runtime.fake_result(exit_code=1, output_bytes=b'{"companies":[]}'),
    )
    assert result["terminal_status"] == "accepted"


def test_new_round_freezes_deadline_and_lease_while_legacy_opt_out_stays_300_seconds():
    service = object.__new__(ArenaService)
    digest = "sha256:" + "a" * 64
    defaults = RoundDefaults(
        runner_hotkeys=("5" * 48,), baseline_hotkey="5" * 48,
        scorer_image_digest=digest,
        scorer_image_reference="registry.example/scorer@" + digest,
    )
    service._config = SimpleNamespace(
        defaults=defaults, mode="live", network_name="finney", netuid=71,
    )
    service._scorer_policy = scoring.build_scorer_policy()
    service.runner_settings = lambda: (["5" * 48], [])
    service._require_round_ownership = lambda _round_id: None
    service._store = SimpleNamespace(
        create_round=lambda *_args: {"status": "created"},
    )
    current = service.create_round(
        datetime(2026, 9, 16, tzinfo=timezone.utc),
        round_id="arena-2026-09-16",
    )
    assert current["checkpoint_deadline_policy"] == contracts.CHECKPOINT_DEADLINE_POLICY
    assert current["icp_wall_clock_seconds"] == 2700
    assert current["lease_ttl_seconds"] == 3600
    assert current["call_quotas"] == {
        "scrapingdog": 200,
        "deepline": 200,
        "openrouter": 500,
    }
    assert current["scoring_call_quotas"]["openrouter"] == 120
    with pytest.raises(contracts.ArenaContractError):
        contracts.validate_round_configuration(
            dict(current, icp_wall_clock_seconds=300)
        )
    service._config.defaults = replace(defaults, checkpoint_deadline_enabled=False)
    historical = service.create_round(
        datetime(2026, 9, 17, tzinfo=timezone.utc),
        round_id="arena-2026-09-17",
    )
    assert "checkpoint_deadline_policy" not in historical
    assert historical["icp_wall_clock_seconds"] == 300
    assert historical["lease_ttl_seconds"] == 1200

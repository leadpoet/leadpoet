"""Champion surplus accounting, allocation GET guards, and Supabase retry.

Production incidents covered:
- A sole champion received the remaining Lab pool each epoch (21-25% vs the
  scheduled 9.4545%) and the surplus retired its 20-epoch obligation in 8
  epochs; surplus must be a bonus, not schedule retirement.
- Anonymous GET /research-lab/allocations/live/{epoch} persisted active
  snapshots for future epochs (23891-23894 were found pre-created).
- Supabase's edge terminates pooled HTTP/2 connections; the shared client
  surfaced httpx.ReadError as API 500s instead of retrying on a fresh
  connection.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from gateway.research_lab.allocations import (
    _champion_paid_alpha_to_date_from_snapshots,
    _champion_replay_obligation,
)


def _snapshot_row(entries, section="champion_allocations", epoch=23890):
    return {"epoch": epoch, "allocation_doc": {section: entries}}


def _entry(source_id, paid, scheduled=None):
    entry = {"source_id": source_id, "paid_alpha_percent": paid}
    if scheduled is not None:
        entry["base_desired_alpha_percent"] = scheduled
    return entry


class TestChampionPaidToDate:
    def test_surplus_epoch_credits_only_scheduled_rate(self):
        rows = [_snapshot_row([_entry("champion_reward:r1", 24.802759, scheduled=9.4545)])]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(9.4545)

    def test_eight_surplus_epochs_do_not_retire_twenty_epoch_schedule(self):
        rows = [
            _snapshot_row([_entry("champion_reward:r1", 23.5, scheduled=9.4545)])
            for _ in range(8)
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(8 * 9.4545)
        obligation = _champion_replay_obligation(
            {
                "champion_reward_id": "champion_reward:r1",
                "start_epoch": 23876,
                "epoch_count": 20,
                "desired_alpha_percent": 9.4545,
            },
            paid_by_reward=paid,
            epoch=23891,
        )
        assert obligation is not None
        assert obligation["remaining_alpha_percent"] == pytest.approx(189.09 - 8 * 9.4545)
        assert obligation["replay_status"] == "nominal_window"

    def test_underpaid_epoch_credits_actual_payment(self):
        rows = [_snapshot_row([_entry("champion_reward:r1", 4.2, scheduled=9.4545)])]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(4.2)

    def test_legacy_entry_without_schedule_credits_full_payment(self):
        rows = [_snapshot_row([_entry("champion_reward:r1", 24.8)])]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(24.8)

    def test_intended_alpha_percent_used_when_base_desired_absent(self):
        rows = [
            _snapshot_row(
                [{"source_id": "champion_reward:r1", "paid_alpha_percent": 20.0, "intended_alpha_percent": 9.4545}]
            )
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(9.4545)

    def test_pre_cutoff_epochs_credit_full_payment(self):
        rows = [
            _snapshot_row([_entry("champion_reward:old", 21.9, scheduled=3.1)], epoch=23700)
            for _ in range(8)
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:old"] == pytest.approx(8 * 21.9)

    def test_cutoff_boundary_epoch_is_capped(self):
        rows = [
            _snapshot_row([_entry("champion_reward:r1", 21.9, scheduled=9.4545)], epoch=23877),
            _snapshot_row([_entry("champion_reward:r1", 21.9, scheduled=9.4545)], epoch=23878),
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(21.9 + 9.4545)

    def test_cutoff_env_override(self, monkeypatch):
        monkeypatch.setenv("RESEARCH_LAB_CHAMPION_SCHEDULE_CAP_START_EPOCH", "23700")
        rows = [_snapshot_row([_entry("champion_reward:old", 21.9, scheduled=3.1)], epoch=23700)]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:old"] == pytest.approx(3.1)

    def test_queued_section_counted_with_same_cap(self):
        rows = [
            _snapshot_row(
                [_entry("champion_reward:r1", 12.0, scheduled=9.4545)],
                section="queued_champion_allocations",
            )
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        assert paid["champion_reward:r1"] == pytest.approx(9.4545)

    def test_fully_paid_schedule_yields_no_obligation(self):
        rows = [
            _snapshot_row([_entry("champion_reward:r1", 9.4545, scheduled=9.4545)])
            for _ in range(20)
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(rows)
        obligation = _champion_replay_obligation(
            {
                "champion_reward_id": "champion_reward:r1",
                "start_epoch": 23876,
                "epoch_count": 20,
                "desired_alpha_percent": 9.4545,
            },
            paid_by_reward=paid,
            epoch=23899,
        )
        assert obligation is None

    def test_final_credit_is_capped_at_remaining_obligation_balance(self):
        rows = [
            _snapshot_row(
                [_entry("champion_reward:r1", 6.0)],
                epoch=100 + index,
            )
            for index in range(3)
        ]
        paid = _champion_paid_alpha_to_date_from_snapshots(
            rows,
            obligation_caps={"champion_reward:r1": 10.0},
        )
        assert paid["champion_reward:r1"] == pytest.approx(10.0)
        assert _champion_replay_obligation(
            {
                "champion_reward_id": "champion_reward:r1",
                "start_epoch": 100,
                "epoch_count": 2,
                "desired_alpha_percent": 5.0,
            },
            paid_by_reward={"champion_reward:r1": 99.0},
            epoch=103,
        ) is None


class TestAllocationSnapshotPersistenceDecision:
    @staticmethod
    def _decide(requested_epoch, key, *, configured="sekret", live=True, current=23891):
        from gateway.research_lab.allocations import allocation_snapshot_persistence_decision

        return allocation_snapshot_persistence_decision(
            current_epoch=current,
            requested_epoch=requested_epoch,
            provided_key=key,
            configured_key=configured,
            live_allocation_enabled=live,
        )

    def test_future_epoch_rejected_even_with_valid_key(self):
        assert self._decide(23892, None) == "future_epoch"
        assert self._decide(23895, "sekret") == "future_epoch"

    def test_anonymous_request_is_read_only(self):
        assert self._decide(23891, None) == "read_only"
        assert self._decide(23880, None) == "read_only"

    def test_valid_key_persists_only_current_epoch(self):
        assert self._decide(23891, "sekret") == "persist"
        assert self._decide(23880, "sekret") == "authenticated_read_only"

    def test_valid_key_without_live_flags_is_read_only(self):
        assert self._decide(23891, "sekret", live=False) == "authenticated_read_only"

    def test_invalid_or_unconfigured_key_rejected(self):
        assert self._decide(23891, "wrong") == "invalid_key"
        assert self._decide(23891, "anything", configured="") == "key_not_configured"


class _FakeSession:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0

    def send(self, request, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "response"


class _AsyncFakeSession(_FakeSession):
    async def send(self, request, **kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return "response"


class _FakeClient:
    def __init__(self, session):
        class _PG:
            pass

        self.postgrest = _PG()
        self.postgrest.session = session


def _request():
    return httpx.Request("GET", "https://db.example/rest/v1/rows")


class TestSupabaseSendRetry:
    def test_sync_retries_once_on_protocol_termination(self):
        from gateway.db.client import _install_sync_send_retry

        session = _FakeSession([httpx.RemoteProtocolError("GOAWAY")])
        _install_sync_send_retry(_FakeClient(session))
        assert session.send is not _FakeSession.send
        result = _FakeClient(session).postgrest.session.send(_request())
        assert result == "response"
        assert session.calls == 2

    def test_sync_does_not_retry_timeouts(self):
        from gateway.db.client import _install_sync_send_retry

        session = _FakeSession([httpx.ReadTimeout("slow")])
        _install_sync_send_retry(_FakeClient(session))
        with pytest.raises(httpx.ReadTimeout):
            session.send(_request())
        assert session.calls == 1

    def test_sync_second_failure_gets_final_attempt(self):
        from gateway.db.client import _install_sync_send_retry

        session = _FakeSession([httpx.ReadError("dead"), httpx.ReadError("dead again")])
        _install_sync_send_retry(_FakeClient(session))
        result = session.send(_request())
        assert result == "response"
        assert session.calls == 3

    def test_sync_third_failure_propagates(self):
        from gateway.db.client import _install_sync_send_retry

        session = _FakeSession(
            [httpx.ReadError("dead"), httpx.ReadError("dead again"), httpx.ReadError("dead thrice")]
        )
        _install_sync_send_retry(_FakeClient(session))
        with pytest.raises(httpx.ReadError):
            session.send(_request())
        assert session.calls == 3

    def test_async_retries_once_on_protocol_termination(self):
        from gateway.db.client import _install_async_send_retry

        session = _AsyncFakeSession([httpx.ReadError("errno 11")])
        _install_async_send_retry(_FakeClient(session))
        result = asyncio.run(session.send(_request()))
        assert result == "response"
        assert session.calls == 2

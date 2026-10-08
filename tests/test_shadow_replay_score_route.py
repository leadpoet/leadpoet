"""Host-paid saved-output probes can use the miner score provider route."""

from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import pytest

from lab_arena import broker, scoring_provider_compat, wiring
from qualification.scoring import company_evidence_investigator
from scripts.verify_arena_parallel_round import _replay_score_route_ledger
from tests.lab_arena.test_lab_arena_broker import (
    CONTEXT, FakeTransport, HOST_KEYS, make_broker,
)


ROUND_ID = "arena-2026-10-08-judgeprobea"
ARCHIVE_HASH = "sha256:" + "a" * 64
SOURCE_URL = (
    scoring_provider_compat.SHADOW_REPLAY_ROUTE_PREFIX
    + "arena-2026-10-07/" + ARCHIVE_HASH.removeprefix("sha256:") + ".tar.gz"
)


@pytest.mark.parametrize(("operation", "parameters"), [
    ("scrapingdog.scrape", {"url": "https://example.com/about"}),
    ("scrapingdog.x_post", {"tweetId": "1234567890"}),
    ("scrapingdog.linkedinjobs", {"job_id": "1234567890"}),
    ("scrapingdog.profile_post", {"id": "1234567890"}),
])
def test_host_opt_in_uses_the_same_full_score_route_as_miner(operation, parameters):
    common = {
        "kind": "score", "round_id": ROUND_ID,
        "operation_id": operation, "parameters": parameters,
        "timeout_ms": 60_000,
    }
    miner = scoring_provider_compat.route_for(
        **common, funding_source="miner_key",
    )
    host = scoring_provider_compat.route_for(
        **common, funding_source="host", allow_host_shadow_score=True,
    )
    assert miner is not None and host == miner
    assert scoring_provider_compat.route_for(
        **common, funding_source="host",
    ) is None
    assert scoring_provider_compat.route_for(
        **{**common, "kind": "execute"}, funding_source="host",
        allow_host_shadow_score=True,
    ) is None


def test_historical_linkedin_job_route_keeps_source_evaluation_date():
    parameters = {"job_id": "1234567890"}
    source = scoring_provider_compat.route_for(
        kind="score", funding_source="miner_key",
        round_id="arena-2026-10-07", operation_id="scrapingdog.linkedinjobs",
        parameters=parameters,
    )
    target = scoring_provider_compat.route_for(
        kind="score", funding_source="host", allow_host_shadow_score=True,
        round_id="arena-2026-10-08-judgeprobea",
        operation_id="scrapingdog.linkedinjobs", parameters=parameters,
    )
    replay = scoring_provider_compat.route_for(
        kind="score", funding_source="host", allow_host_shadow_score=True,
        round_id="arena-2026-10-07", operation_id="scrapingdog.linkedinjobs",
        parameters=parameters,
    )
    assert source == replay
    assert replay.evaluation_date == date(2026, 10, 7)
    assert target.evaluation_date == date(2026, 10, 8)


def test_frozen_shadow_source_is_required_to_enable_host_route():
    configuration = {
        "mode": "shadow", "rewards_enabled": False,
        "baseline_source_url": SOURCE_URL,
    }
    row = {"round_id": ROUND_ID, "configuration_doc": configuration}
    replay = {
        "source_round": "arena-2026-10-07",
        "archive_hash": ARCHIVE_HASH,
        "score_provider_route": scoring_provider_compat.COMPATIBILITY_VERSION,
    }
    service = SimpleNamespace(
        config=SimpleNamespace(
            mode="shadow", pinned_round_id=ROUND_ID,
            defaults=SimpleNamespace(
                rewards_enabled=False, baseline_source_url=SOURCE_URL,
            ),
        ),
        _saved_output_replay=replay,
    )
    assert wiring._shadow_replay_score_compat_round_id(service, row) == ROUND_ID
    for changed in (
        {"round_id": ROUND_ID + "x", "configuration_doc": configuration},
        {"round_id": ROUND_ID, "configuration_doc": {
            **configuration, "mode": "live",
        }},
        {"round_id": ROUND_ID, "configuration_doc": {
            **configuration, "rewards_enabled": True,
        }},
        {"round_id": ROUND_ID, "configuration_doc": {
            **configuration, "baseline_source_url": "https://arena.invalid/other",
        }},
    ):
        assert wiring._shadow_replay_score_compat_round_id(service, changed) is None
    service._saved_output_replay = {**replay, "score_provider_route": "other"}
    assert wiring._shadow_replay_score_compat_round_id(service, row) is None
    service._saved_output_replay = {**replay, "source_round": ROUND_ID}
    assert wiring._shadow_replay_score_compat_round_id(service, row) is None
    service._saved_output_replay = replay
    service.config.mode = "live"
    assert wiring._shadow_replay_score_compat_round_id(service, row) is None


def test_host_score_route_settles_once_and_adapts_firecrawl_target_404(monkeypatch):
    target = "https://example.com/missing"
    envelope = {
        "status": "completed",
        "job_id": "iad1::replay-firecrawl-404",
        "billing": {"credits_charged": 0.02, "cost_usd": 0.002},
        "result": {"data": {
            "rawHtml": "<html>Not found</html>",
            "metadata": {
                "sourceURL": target, "url": target, "statusCode": 404,
            },
        }},
    }
    transport = FakeTransport([(200, envelope)])
    route_round_ids = []
    original_route_for = scoring_provider_compat.route_for

    def capture_route(**kwargs):
        route_round_ids.append(kwargs["round_id"])
        return original_route_for(**kwargs)

    monkeypatch.setattr(scoring_provider_compat, "route_for", capture_route)
    scored = replace(CONTEXT, kind="score", round_id=ROUND_ID)
    paid, store, _ = make_broker(
        transport=transport,
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "host",
        host_shadow_score_compat_round_id=ROUND_ID,
        host_shadow_score_compat_source_round_id="arena-2026-10-07",
    )
    parameters = {"url": target}
    first = paid.execute(
        scored, operation_id="scrapingdog.scrape", parameters=parameters,
        action_sequence=7, timeout_ms=60_000,
    )
    repeated = paid.execute(
        scored, operation_id="scrapingdog.scrape", parameters=parameters,
        action_sequence=7, timeout_ms=60_000,
    )
    assert first.status == repeated.status == 404
    assert first.call["call_identity"] == repeated.call["call_identity"]
    assert first.call["actual_microusd"] == repeated.call["actual_microusd"] == 2_000
    assert first.call["provider"] == "deepline"
    assert first.call["funding_source"] == "host"
    assert first.call["operation_id"] == "scrapingdog.scrape"
    assert first.call["compatibility_version"] == scoring_provider_compat.COMPATIBILITY_VERSION
    assert len(transport.sent) == 1
    assert route_round_ids == ["arena-2026-10-07", "arena-2026-10-07"]
    assert len(store.calls) == 1
    settled = next(iter(store.calls.values()))
    assert settled["kind"] == "settlement"
    assert settled["provider"] == "deepline"
    assert settled["funding_source"] == "host"
    assert settled["operation_id"] == "scrapingdog.scrape"
    assert company_evidence_investigator._fetch_outcome(
        target, {"ok": False, "error": f"http_{first.status}"},
    )["error_class"] == "source_not_found"

    wrong_round, wrong_store, wrong_transport = make_broker(
        transport=FakeTransport([(200, "<html>direct</html>")]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "host",
        host_shadow_score_compat_round_id="arena-2026-10-08-other",
    )
    refused = wrong_round.execute(
        scored, operation_id="scrapingdog.scrape", parameters=parameters,
        action_sequence=8, timeout_ms=60_000,
    )
    assert refused.status != 200
    assert not wrong_store.calls and not wrong_transport.sent

    ordinary, _store, ordinary_transport = make_broker(
        transport=FakeTransport([(200, "<html>direct</html>")]),
        credential_for=lambda _context, provider: HOST_KEYS[provider],
        provider_funding_source_for=lambda _context, _provider: "host",
    )
    not_opted = ordinary.execute(
        scored, operation_id="scrapingdog.scrape", parameters=parameters,
        action_sequence=9, timeout_ms=60_000,
    )
    assert not_opted.status == 200
    assert not_opted.call["provider"] == "scrapingdog"
    assert "api.scrapingdog.com" in ordinary_transport.sent[0]["url"]


def test_replay_ledger_proof_checks_effective_route_without_request_data():
    rows = [{
        "entry_kind": "reservation", "operation_id": "scrapingdog.scrape",
        "provider": "deepline", "funding_source": "host",
        "entry_doc": {
            "compatibility_version": scoring_provider_compat.COMPATIBILITY_VERSION,
            "effective_operation_id": scoring_provider_compat.EFFECTIVE_OPERATION_ID,
            "adapter": "firecrawl_raw_html",
            "request_hash": "private-request-identity",
        },
    }]

    class Store:
        def list_ledger(self, *, run_id):
            assert run_id == "score-run"
            return rows

    service = SimpleNamespace(store=Store())
    runs = [{"kind": "score", "run_id": "score-run"}]
    expected = {
        "eligible_reservations": 1,
        "host_compat_reservations": 1,
        "mismatched_reservations": 0,
    }
    assert _replay_score_route_ledger(service, runs) == expected
    rows[0] = {**rows[0], "provider": "scrapingdog"}
    assert _replay_score_route_ledger(service, runs) == {
        "eligible_reservations": 1,
        "host_compat_reservations": 0,
        "mismatched_reservations": 1,
    }

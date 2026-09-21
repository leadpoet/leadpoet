"""Daily ICP banks use their requested size and validate actual diversity."""

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json

import pytest

from gateway.tasks import icp_generator as generator


def _provider_rows(slots):
    intents = [
        ("Recently raised funding", "FUNDING"),
        ("Launched or announced a new product", "PRODUCT_LAUNCH"),
        ("Expanded to new markets", "MARKET_EXPANSION"),
        ("Recent leadership change", "LEADERSHIP_CHANGE"),
        ("Hiring for senior engineering or sales roles", "HIRING"),
        ("Achieved regulatory clearance or certification", "REGULATORY_CLEARANCE"),
    ]
    rows = []
    seen_industries = Counter()
    for index, industry in enumerate(slots):
        occurrence = seen_industries[industry]
        seen_industries[industry] += 1
        intent, category = intents[index % len(intents)]
        geography = (
            generator.INTERNATIONAL_GEOGRAPHIES[index % len(generator.INTERNATIONAL_GEOGRAPHIES)]
            if index < generator.international_icp_target(len(slots))
            else "United States"
        )
        rows.append({
            "icp_id": f"icp_20260920_{index + 1:03d}",
            "prompt": f"Find {industry} companies with {intent.lower()} in {geography}.",
            "industry": industry,
            "sub_industry": generator.SUB_INDUSTRIES[industry][occurrence % len(generator.SUB_INDUSTRIES[industry])],
            "company_stage": generator.COMPANY_STAGES[index % len(generator.COMPANY_STAGES)],
            "employee_count": ["51-200"],
            "geography": geography,
            "country": geography.split(",")[0],
            "product_service": f"{industry} workflow platform",
            "intent_signal": intent,
            "intent_category": category,
            "verified_example_company": f"Example {index}",
        })
    return rows


def _stub_provider(monkeypatch, mutate=None):
    captured = {}

    class Response:
        status_code = 200

        def __init__(self, rows):
            self.rows = rows

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"icps": self.rows})}}]}

    class Client:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def post(self, _url, *, headers, json):
            del headers
            captured.update(json)
            prompt = json["messages"][0]["content"]
            slots = prompt.split("ORDERED INDUSTRY SLOTS (one ICP per slot, in this order; repeated names are intentional):\n", 1)[1].split("\n", 1)[0].split(", ")
            rows = _provider_rows(slots)
            if mutate:
                mutate(rows)
            return Response(rows)

    monkeypatch.setattr(generator.httpx, "AsyncClient", Client)
    return captured


@pytest.mark.parametrize("count", [10, 15, 20, 30])
def test_provider_bank_count_and_diversity(monkeypatch, count):
    captured = _stub_provider(monkeypatch)
    result = asyncio.run(generator.generate_icps_with_openrouter(
        20260920, total_icps=count, api_key="test-only",
    ))
    assert result is not None
    icps, distribution, digest = result
    assert len(icps) == count
    assert Counter(icp["industry"] for icp in icps) == Counter(distribution)
    assert max(distribution.values()) - min(distribution.values()) <= 1
    assert len(distribution) == min(count, 20)
    assert generator.generated_bank_error(icps, count, distribution) is None
    assert digest == generator.compute_icp_set_hash(icps)
    assert captured["max_tokens"] >= count * 400


@pytest.mark.parametrize("count", [10, 15, 20, 30])
def test_template_bank_count_and_diversity(monkeypatch, count):
    monkeypatch.setenv("LAB_ARENA_ICP_EXCLUSIONS_ENABLED", "0")
    icps, distribution, digest = generator.generate_icp_set(
        20260920, total_icps=count, base_seed=42,
    )
    assert len(icps) == count
    assert len(distribution) == min(count, 20)
    assert max(distribution.values()) - min(distribution.values()) <= 1
    assert generator.generated_bank_error(icps, count, distribution) is None
    if count != 20:
        assert len({icp["industry"] for icp in icps}) == min(count, 20)
    assert digest == generator.compute_icp_set_hash(icps)


@pytest.mark.parametrize("change", [
    lambda rows: rows.pop(),
    lambda rows: rows[1].update(industry=rows[0]["industry"]),
    lambda rows: rows[1].update(icp_id=rows[0]["icp_id"]),
])
def test_provider_rejects_bad_bank(monkeypatch, change):
    _stub_provider(monkeypatch, change)
    assert asyncio.run(generator.generate_icps_with_openrouter(
        20260920, total_icps=10, api_key="test-only",
    )) is None


def test_duplicate_market_is_rejected():
    slots = generator.industry_slots(30, seed=42)
    rows = _provider_rows(slots)
    first = next(i for i, industry in enumerate(slots) if slots.count(industry) > 1)
    duplicate = next(i for i, industry in enumerate(slots) if i > first and industry == slots[first])
    rows[duplicate].update({
        key: rows[first][key]
        for key in ("sub_industry", "company_stage", "geography", "intent_category")
    })
    expected = dict(Counter(slots))
    assert generator.generated_bank_error(rows, 30, expected) == "duplicate_market"


def test_provider_rejects_duplicate_market_in_repeated_industry(monkeypatch):
    def duplicate_market(rows):
        first = next(i for i, row in enumerate(rows) if sum(other["industry"] == row["industry"] for other in rows) > 1)
        duplicate = next(i for i, row in enumerate(rows) if i > first and row["industry"] == rows[first]["industry"])
        rows[duplicate].update({
            key: rows[first][key]
            for key in ("sub_industry", "company_stage", "geography", "intent_category")
        })

    _stub_provider(monkeypatch, duplicate_market)
    assert asyncio.run(generator.generate_icps_with_openrouter(
        20260920, total_icps=30, api_key="test-only",
    )) is None


def test_activation_rejects_bad_bank_before_storage(monkeypatch):
    rows = _provider_rows(generator.industry_slots(10, seed=42))[:-1]

    async def fake_generate(*_args, **_kwargs):
        return rows, dict(Counter(row["industry"] for row in rows)), "unused"

    async def store(*_args, **_kwargs):
        pytest.fail("bad bank reached storage")

    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-only")
    async def no_active(*, strict=False):
        return None
    monkeypatch.setattr(generator, "get_active_icp_set", no_active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", lambda _set_id: None)
    monkeypatch.setattr(generator, "generate_icps_with_openrouter", fake_generate)
    monkeypatch.setattr(generator, "store_icp_set", store)
    assert asyncio.run(generator.generate_and_activate_icp_set(
        datetime(2026, 9, 20, tzinfo=timezone.utc), total_icps=10,
    )) is None


def test_activation_passes_count_to_generation_and_stores_exact_bank(monkeypatch):
    rows = _provider_rows(generator.industry_slots(10, seed=42))
    calls = {}

    async def fake_generate(_set_id, *, total_icps, **_kwargs):
        calls["requested"] = total_icps
        return rows, dict(Counter(row["industry"] for row in rows)), "before-contract"

    async def store(*, icps, icp_set_hash, **_kwargs):
        calls["stored"] = len(icps)
        assert icp_set_hash == generator.compute_icp_set_hash(icps)
        return True

    async def activate(_set_id):
        return True

    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-only")
    async def no_active(*, strict=False):
        return None
    monkeypatch.setattr(generator, "get_active_icp_set", no_active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", lambda _set_id: None)
    monkeypatch.setattr(generator, "generate_icps_with_openrouter", fake_generate)
    monkeypatch.setattr(generator, "store_icp_set", store)
    monkeypatch.setattr(generator, "activate_icp_set", activate)
    monkeypatch.setattr(generator, "attach_generated_exclusions", lambda _icps: None)
    assert asyncio.run(generator.generate_and_activate_icp_set(
        datetime(2026, 9, 20, tzinfo=timezone.utc), total_icps=10,
    )) == 20260920
    assert calls == {"requested": 10, "stored": 10}


def test_next_bank_count_uses_shared_arena_default_and_bounds(monkeypatch):
    monkeypatch.delenv("LAB_ARENA_BENCHMARK_ICP_COUNT", raising=False)
    from lab_arena import contracts
    assert generator.configured_benchmark_icp_count() == contracts.DEFAULT_BENCHMARK_ICP_COUNT
    for value in ("", "   "):
        monkeypatch.setenv("LAB_ARENA_BENCHMARK_ICP_COUNT", value)
        assert generator.configured_benchmark_icp_count() == contracts.DEFAULT_BENCHMARK_ICP_COUNT
    for count in (2, 10, 15, 30, 100):
        monkeypatch.setenv("LAB_ARENA_BENCHMARK_ICP_COUNT", str(count))
        assert generator.configured_benchmark_icp_count() == count
    for value in ("1", "101", "abc", "10.5"):
        monkeypatch.setenv("LAB_ARENA_BENCHMARK_ICP_COUNT", value)
        with pytest.raises(ValueError):
            generator.configured_benchmark_icp_count()


def _round_row(day, count, *, mode="live"):
    return {"configuration_doc": {
        "mode": mode,
        "schedule": {"submission_open": f"{day}T12:00:00Z"},
        "stage_1_icp_count": (count + 1) // 2,
        "stage_2_icp_count": count // 2,
    }}


def _stub_round_query(monkeypatch, rows=None, error=None):
    from lab_arena import store as arena_store

    calls = []

    class Store:
        def __init__(self, transport):
            assert transport == "scoped transport"

        def list_rounds(self, **kwargs):
            calls.append(kwargs)
            if error:
                raise error
            return rows

        def close(self):
            calls.append("closed")

    def transport(origin, *, service_key):
        assert origin == "https://db.example.test"
        assert service_key == "sb_secret_test"
        return "scoped transport"

    monkeypatch.setenv("LAB_ARENA_SUPABASE_URL", "https://db.example.test")
    monkeypatch.setenv("LAB_ARENA_SERVICE_KEY", "sb_secret_test")
    monkeypatch.setattr(arena_store, "PostgrestTransport", transport)
    monkeypatch.setattr(arena_store, "ArenaStore", Store)
    return calls


def test_frozen_count_uses_open_live_round_for_bank_date(monkeypatch):
    monkeypatch.setenv("LAB_ARENA_NETWORK", "finney")
    monkeypatch.setenv("LAB_ARENA_NETUID", "71")
    calls = _stub_round_query(monkeypatch, [
        _round_row("2026-09-19", 30),
        _round_row("2026-09-20", 15, mode="shadow"),
        _round_row("2026-09-20", 20),
    ])
    assert generator.frozen_benchmark_icp_count(20260920) == 20
    assert calls[0] == {
        "status": "open", "mode": "live", "network_name": "finney",
        "netuid": 71, "limit": 100, "columns": "round_id,configuration_doc",
    }
    assert "closed" in calls
    assert generator.frozen_benchmark_icp_count(20260921) is None


def test_frozen_count_query_failure_or_conflict_fails_closed(monkeypatch):
    _stub_round_query(monkeypatch, error=RuntimeError("database unavailable"))
    with pytest.raises(RuntimeError):
        generator.frozen_benchmark_icp_count(20260920)
    _stub_round_query(monkeypatch, rows=[
        _round_row("2026-09-20", 20), _round_row("2026-09-20", 10),
    ])
    with pytest.raises(ValueError, match="disagree"):
        generator.frozen_benchmark_icp_count(20260920)


def test_activation_prefers_frozen_count_and_existing_bank(monkeypatch):
    rows = _provider_rows(generator.industry_slots(20))
    calls = []

    async def no_active(*, strict=False):
        calls.append(("active", strict))
        return None

    async def already_active(*, strict=False):
        return {"set_id": 20260920}

    async def fake_generate(_set_id, *, total_icps, **_kwargs):
        calls.append(("generate", total_icps))
        return rows, dict(Counter(row["industry"] for row in rows)), "before-contract"

    async def store(*, icps, **_kwargs):
        calls.append(("store", len(icps)))
        return True

    async def activate(_set_id):
        return True

    monkeypatch.setattr(generator, "OPENROUTER_API_KEY", "test-only")
    monkeypatch.setattr(generator, "get_active_icp_set", no_active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", lambda _set_id: 20)
    monkeypatch.setattr(generator, "generate_icps_with_openrouter", fake_generate)
    monkeypatch.setattr(generator, "store_icp_set", store)
    monkeypatch.setattr(generator, "activate_icp_set", activate)
    monkeypatch.setattr(generator, "attach_generated_exclusions", lambda _icps: None)
    date = datetime(2026, 9, 20, tzinfo=timezone.utc)
    assert asyncio.run(generator.generate_and_activate_icp_set(date, total_icps=10)) == 20260920
    assert calls == [("active", True), ("generate", 20), ("store", 20)]

    monkeypatch.setattr(generator, "get_active_icp_set", already_active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", lambda _set_id: pytest.fail("existing bank queried frozen rounds"))
    assert asyncio.run(generator.generate_and_activate_icp_set(date, total_icps=10)) == 20260920


def test_activation_stops_before_generation_when_frozen_read_fails(monkeypatch):
    async def no_active(*, strict=False):
        return None

    def failed_read(_set_id):
        raise RuntimeError("database unavailable")

    async def no_generate(*_args, **_kwargs):
        pytest.fail("frozen count failure reached provider")

    monkeypatch.setattr(generator, "get_active_icp_set", no_active)
    monkeypatch.setattr(generator, "frozen_benchmark_icp_count", failed_read)
    monkeypatch.setattr(generator, "generate_icps_with_openrouter", no_generate)
    assert asyncio.run(generator.generate_and_activate_icp_set(
        datetime(2026, 9, 20, tzinfo=timezone.utc), total_icps=10,
    )) is None

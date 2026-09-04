"""Benchmark generation (labarena.md sections 8 and 18.6) against recorded tapes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from gateway.tasks import icp_generator as lab_generator

from lab_arena import benchmark as b
from lab_arena import contracts
from tests.lab_arena.lab_arena_benchmark_tape import (
    FIXTURES,
    INDUSTRIES,
    TapeProvider,
    completion,
    load_tape,
    raw_icp,
    write_default_tapes,
)

ROOT = Path(__file__).resolve().parents[2]
ROUND_ID = "arena-2026-09-02"
SET_ID = 20260902


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 9, 2, 0, 5, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        self.now = self.now + timedelta(seconds=7)
        return self.now


def run(provider, *, max_attempts=12):
    result = b.generate_benchmark(
        round_id=ROUND_ID, set_id=SET_ID, provider=provider,
        clock=Clock(), max_attempts=max_attempts,
    )
    return result


def test_fixture_tapes_are_the_committed_bytes():
    names = ("clean_run.json", "replacement_run.json", "exhausted_run.json")
    committed = {name: (FIXTURES / name).read_bytes() for name in names}
    write_default_tapes()
    for name in names:
        assert (FIXTURES / name).read_bytes() == committed[name]


def test_lab_prompt_parity_for_twenty_icps(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps({"icps": []})}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResponse()

    monkeypatch.setattr(lab_generator, "OPENROUTER_API_KEY", "sk-or-v1-" + "a" * 40)
    monkeypatch.setattr(lab_generator, "httpx", types.SimpleNamespace(AsyncClient=FakeClient, TimeoutException=Exception))
    asyncio.run(lab_generator.generate_icps_with_openrouter(SET_ID, 20))
    system_prompt, user_prompt = b.build_generation_prompts(count=20, industries=INDUSTRIES, set_id=SET_ID)
    messages = captured["json"]["messages"]
    assert messages[0]["content"] == system_prompt
    assert messages[1]["content"] == user_prompt
    request = b.build_generation_request(count=20, industries=INDUSTRIES, set_id=SET_ID)
    assert request["model"] == captured["json"]["model"] == "perplexity/sonar-pro"
    assert request["temperature"] == captured["json"]["temperature"]
    assert request["max_tokens"] == captured["json"]["max_tokens"]
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    ten_system, ten_user = b.build_generation_prompts(count=10, industries=INDUSTRIES[:10], set_id=SET_ID)
    assert "Generate exactly 10 ICPs" in ten_system and "Transportation" not in ten_system
    assert ten_user.startswith("Generate 10 ICPs for set_id=%d." % SET_ID)


def test_lab_generator_file_is_untouched_and_never_called_from_arena_code():
    path = ROOT / "gateway/tasks/icp_generator.py"
    committed = subprocess.run(["git", "show", "HEAD:gateway/tasks/icp_generator.py"], cwd=ROOT, capture_output=True, check=False)
    if committed.returncode == 0:
        assert hashlib.sha256(committed.stdout).hexdigest() == hashlib.sha256(path.read_bytes()).hexdigest()
    source = (ROOT / "lab_arena/benchmark.py").read_text(encoding="utf-8")
    for forbidden in (
        "store_icp_set", "activate_icp_set", "get_active_icp_set", "generate_and_activate_icp_set",
        "icp_rotation_task", "ensure_icp_set_exists", "attach_generated_exclusions",
        "generate_exclusions_for_icp", "generate_icps_with_openrouter", "compute_icp_set_hash",
    ):
        assert forbidden not in source, forbidden


def test_clean_tape_fills_thirty_slots_in_order_with_contract_fields():
    provider = TapeProvider(load_tape("clean_run.json"))
    result = run(provider)
    assert provider.generation_calls == 2 and provider.exclusion_calls == 30
    assert len(result.icps) == 30 and result.attempts_used == 2
    for slot, icp in enumerate(result.icps):
        assert icp["icp_id"] == "arena:%s:%s:%d" % (ROUND_ID, b.slot_batch(slot), slot)
        assert icp["industry"] == b.slot_industry(slot)
        assert icp["max_companies"] == 5
        assert isinstance(icp["employee_count"], list) and icp["employee_count"]
        assert icp["excluded_companies"] and all(".example.com" in item for item in icp["excluded_companies"])
        assert icp["intent_signals"] and icp["intent_signal"] == icp["intent_signals"][0]
    assert [i["industry"] for i in b.stage_slice(result, 1)] == INDUSTRIES[:10]
    assert [i["industry"] for i in b.stage_slice(result, 2)] == INDUSTRIES[10:] + INDUSTRIES[:10]
    with pytest.raises(contracts.ArenaContractError):
        b.stage_slice(result, 3)
    kinds = [entry["kind"] for entry in result.diagnostics]
    assert kinds.count("request") == 2 and kinds.count("response") == 2
    assert kinds.count("acceptance") == 30 and kinds.count("exclusion") == 30 and "unknown" not in kinds


def test_replacement_tape_replaces_only_rejected_slots_and_records_each_rule():
    provider = TapeProvider(load_tape("replacement_run.json"))
    result = run(provider)
    assert provider.generation_calls == 3 and result.attempts_used == 3
    rejections = [entry for entry in result.diagnostics if entry["kind"] == "rejection"]
    rules = sorted((entry.get("slot"), entry["rejection_rule"]) for entry in rejections)
    assert (3, "schema.example_company_missing") in rules
    assert (5, "schema.geography_unsupported") in rules
    assert (27, "duplicate.content_hash") in rules
    replacement_request = [entry for entry in result.diagnostics if entry["kind"] == "request" and entry["batch_id"] == "r1"][0]
    assert replacement_request["slots"] == [3, 5, 27]
    assert replacement_request["industries"] == ["Hardware", "Privacy and Security", "Biotechnology"]
    user_prompts = [r["messages"][-1]["content"] for r in provider.requests if r["messages"][0]["role"] == "system"]
    assert user_prompts[-1] == "Generate 3 ICPs for set_id=%d, one for each of these industries in this order: Hardware, Privacy and Security, Biotechnology. Follow every instruction in the system message exactly. Output JSON only, no commentary." % SET_ID
    assert result.icps[3]["verified_example_company"] == "Hardware Example Co 9"
    assert result.icps[5]["industry"] == "Privacy and Security"
    assert result.icps[27]["verified_example_company"] == "Biotechnology Example Co 9"
    accepted = [entry for entry in result.diagnostics if entry["kind"] == "acceptance"]
    assert len(accepted) == 30 and len({entry["slot"] for entry in accepted}) == 30


def test_exhausted_attempts_stop_at_the_bound():
    provider = TapeProvider(load_tape("exhausted_run.json") + [completion([dict(raw_icp("Software", 50 + i), company_stage="Unicorn")], response_id="never-valid-%d" % i) for i in range(20)])
    with pytest.raises(b.BenchmarkGenerationFailed) as failure:
        run(provider, max_attempts=5)
    assert failure.value.attempts_used == 5
    assert sorted(failure.value.missing_slots) == [0, 20]


def test_provider_failure_is_recorded_and_retried():
    provider = TapeProvider(load_tape("clean_run.json"), fail_at={2})
    result = run(provider)
    kinds = [entry["kind"] for entry in result.diagnostics]
    assert kinds.count("unknown") == 1 and kinds.count("request") == 3
    assert result.attempts_used == 3 and len(result.icps) == 30


def test_exclusion_failure_rejects_the_slot_and_replacement_refills_it():
    tape = load_tape("clean_run.json") + [completion([raw_icp("Energy", 11)], response_id="gen-r1-energy")]
    provider = TapeProvider(tape)
    calls = {"n": 0}
    original = provider.chat

    def flaky(**kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if prompt.startswith("Name ") and "- Industry: Energy (" in prompt and calls["n"] == 0:
            calls["n"] += 1
            raise b.ProviderFailure("exclusion provider unavailable")
        return original(**kwargs)

    provider.chat = flaky
    result = run(provider)
    rejected = [entry for entry in result.diagnostics if entry["kind"] == "rejection" and entry["rejection_rule"] == "exclusion.failed"]
    assert [entry["slot"] for entry in rejected] == [17]
    exclusion_entries = [entry for entry in result.diagnostics if entry["kind"] == "exclusion" and entry["slot"] == 17]
    assert exclusion_entries[0]["outcome"] == "provider_failure" and exclusion_entries[-1]["outcome"] == "accepted"
    assert result.icps[17]["excluded_companies"]
    assert all(icp["excluded_companies"] for icp in result.icps)


def test_environment_bucket_values_do_not_change_output(monkeypatch):
    reference = run(TapeProvider(load_tape("clean_run.json")))
    monkeypatch.setenv("RESEARCH_LAB_ICP_EMPLOYEE_BUCKET_RADIUS", "0")
    monkeypatch.setenv("RESEARCH_LAB_ICP_EMPLOYEE_ALL_BUCKETS", "1")
    again = run(TapeProvider(load_tape("clean_run.json")))
    assert again.icps == reference.icps


def test_schema_check_runs_before_canonicalization_on_raw_output():
    requested = [(0, "Software")]
    good = raw_icp("Software", 1)
    assert b.check_raw_icp(good, requested=requested, filled=set()).rejection_rule is None
    cases = {
        "schema.prompt_missing": dict(good, prompt="  "),
        "schema.industry_mismatch": dict(good, industry="Quantum"),
        "schema.geography_unsupported": dict(good, geography="Europe"),
        "schema.employee_bucket_invalid": dict(good, employee_count=["51-5000"]),
        "schema.stage_invalid": dict(good, company_stage="Unicorn"),
        "schema.intent_missing": dict(good, intent_signals=[], intent_signal=""),
        "schema.example_company_missing": dict(good, verified_example_company=""),
        "schema.not_an_object": ["not", "an", "object"],
        "slot.filled": good,
    }
    for rule, raw in cases.items():
        filled = {0} if rule == "slot.filled" else set()
        assert b.check_raw_icp(raw, requested=requested, filled=filled).rejection_rule == rule, rule
    assert b.check_raw_icp(dict(good, industry=" software "), requested=requested, filled=set()).slot == 0
    with pytest.raises(contracts.ArenaContractError):
        b.canonicalize_arena_icp(dict(good, intent_signals=[]))


def test_duplicate_intent_signature_and_generator_configuration():
    icps = [raw_icp(industry, 1) for industry in INDUSTRIES]
    icps[1]["intent_signals"] = list(icps[0]["intent_signals"])
    icps[1]["intent_signal"] = icps[0]["intent_signal"]
    responses = [
        completion(icps, response_id="dup"),
        completion([raw_icp(i, 2) for i in INDUSTRIES[:10]], response_id="b2"),
        completion([raw_icp("Information Technology", 12)], response_id="r1"),
    ]
    result = run(TapeProvider(responses))
    dup = [entry for entry in result.diagnostics if entry["kind"] == "rejection" and entry["rejection_rule"] == "duplicate.intent_signature"]
    assert [entry["slot"] for entry in dup] == [1]
    assert result.icps[1]["verified_example_company"] == "Information Technology Example Co 12"
    configuration = b.generator_configuration()
    assert configuration["model"] == "perplexity/sonar-pro" and configuration["batch_sizes"] == [20, 10]
    assert configuration == b.generator_configuration()

"""Unit coverage for lab_arena.contracts: strict limits, signed requests,
hashed documents, and the fixed public constants (labarena.md sections 1, 5, 9.1, 14)."""

from __future__ import annotations

import time

import pytest
from bittensor_wallet import Keypair

from lab_arena import contracts as c


def wallet_verifier(hotkey: str, signature: str, message: str) -> bool:
    try:
        raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
        return bool(Keypair(ss58_address=hotkey).verify(message.encode("utf-8"), raw))
    except Exception:
        return False


def signed(keypair: Keypair, **overrides):
    now = int(time.time())
    return c.build_signed_request(
        scope=overrides.pop("scope", c.SCOPE_CLAIM),
        round_id=overrides.pop("round_id", "arena-2026-09-02"),
        hotkey=keypair.ss58_address,
        body=overrides.pop("body", {"declared_parallelism": 2}),
        timestamp=overrides.pop("timestamp", now),
        sign_message=lambda message: keypair.sign(message.encode("utf-8")).hex(),
        **overrides,
    )


def test_public_constants_are_the_plan_values():
    assert (c.STAGE_1_ICP_COUNT, c.STAGE_2_ICP_COUNT, c.BENCHMARK_ICP_COUNT, c.FINALIST_COUNT) == (10, 10, 20, 10)
    assert (c.MAX_CHALLENGERS, c.RUNNER_SLOT_CEILING, c.MAX_ATTEMPTS_PER_ASSIGNMENT) == (256, 8, 2)
    assert c.LAB_ARENA_POOL_PERCENT == 25
    assert c.KING_POOL_SHARE_PERCENT_BY_WEEK == (100, 80, 60, 40, 20)
    assert (c.EPOCHS_PER_REWARD_WEEK, c.ELIGIBILITY_MAX_EPOCHS) == (140, 45)
    assert c.PROVIDERS == ("scrapingdog", "deepline", "openrouter") and c.CALL_QUOTAS_PER_ICP == {"scrapingdog": 30, "deepline": 30, "openrouter": 60}
    assert (c.ICP_WALL_CLOCK_SECONDS, c.SCORING_WALL_CLOCK_SECONDS, c.LEASE_TTL_SECONDS) == (300, 900, 1200)
    from leadpoet_canonical.constants import EPOCH_LENGTH

    assert EPOCH_LENGTH * 12 * c.EPOCHS_PER_REWARD_WEEK == 7 * 24 * 3600


def test_strict_document_limits_reject_every_hostile_shape():
    limits = c.StrictLimits(max_depth=2, max_list_items=2, max_object_keys=2, max_string_bytes=4, max_total_bytes=64)
    c.check_strict_document({"a": [1, 2], "b": "abcd"}, limits)
    for bad in (
        {"a": {"b": {"c": 1}}},
        {"a": [1, 2, 3]},
        {"a": 1, "b": 2, "c": 3},
        {"a": "abcde"},
        {"a": "ab\x00"},
        {"a": "ab\x7f"},
        {"a": float("nan")},
        {"a": float("inf")},
        {"a": 2 ** 60},
        {1: "x"},
        {"a": object()},
        {"a": "\ud800"},
    ):
        with pytest.raises(c.ArenaContractError):
            c.check_strict_document(bad, limits)
    with pytest.raises(c.ArenaContractError):
        c.check_strict_document({"k": "abcd", "j": "abcd"}, c.StrictLimits(max_total_bytes=10))


def test_signed_request_roundtrip_and_rejections():
    keypair = Keypair.create_from_uri("//Alice")
    envelope = signed(keypair)
    now = int(time.time())
    validated = c.validate_signed_request(envelope, expected_scope=c.SCOPE_CLAIM, now=now, verify_signature=wallet_verifier)
    assert validated["hotkey"] == keypair.ss58_address
    assert c.request_bytes_hash(envelope) == c.request_bytes_hash(dict(envelope))
    rejects = [
        (dict(envelope, scope=c.SCOPE_COMPLETE), {}),
        (envelope, {"expected_round_id": "arena-2026-09-03"}),
        (dict(envelope, timestamp=now - c.REQUEST_TIMESTAMP_WINDOW_SECONDS - 1), {}),
        (dict(envelope, timestamp=now + c.REQUEST_TIMESTAMP_WINDOW_SECONDS + 1), {}),
        (dict(envelope, body={"declared_parallelism": 3}), {}),
        (dict(envelope, request_id="short"), {}),
        (dict(envelope, extra=1), {}),
        (dict(envelope, signature="0x" + "0" * 128), {}),
        (dict(envelope, schema_version="other"), {}),
        ({k: v for k, v in envelope.items() if k != "signature"}, {}),
    ]
    for document, kwargs in rejects:
        with pytest.raises(c.ArenaContractError):
            c.validate_signed_request(document, expected_scope=c.SCOPE_CLAIM, now=now, verify_signature=wallet_verifier, **kwargs)
    # A signature from another hotkey over the same bytes is rejected.
    other = Keypair.create_from_uri("//Bob")
    forged = dict(envelope, signature="0x" + other.sign(c.signed_request_message(envelope).encode()).hex())
    with pytest.raises(c.ArenaSignatureError):
        c.validate_signed_request(forged, expected_scope=c.SCOPE_CLAIM, now=now, verify_signature=wallet_verifier)
    with pytest.raises(c.ArenaContractError):
        signed(keypair, scope="lab_arena.other.v1")


def test_completion_signed_request_uses_its_larger_shared_limit():
    keypair = Keypair.create_from_uri("//Alice")
    chunks = ["x" * 65_000 for _ in range(32)]
    envelope = signed(
        keypair,
        scope=c.SCOPE_COMPLETE,
        body={"run_id": "run-1", "output": {"chunks": chunks}},
    )
    assert len(c.canonical_json(envelope).encode("utf-8")) > c.REQUEST_LIMITS.max_total_bytes
    assert c.COMPLETION_REQUEST_LIMITS.max_total_bytes > 2 * 1024 * 1024
    assert c.validate_signed_request(
        envelope,
        expected_scope=c.SCOPE_COMPLETE,
        now=int(time.time()),
        verify_signature=wallet_verifier,
    ) == envelope

    oversized = signed(
        keypair,
        scope=c.SCOPE_COMPLETE,
        body={"run_id": "run-1", "output": {"chunks": chunks + chunks[:2]}},
    )
    with pytest.raises(c.ArenaContractError):
        c.validate_signed_request(
            oversized,
            expected_scope=c.SCOPE_COMPLETE,
            now=int(time.time()),
            verify_signature=wallet_verifier,
        )

    with pytest.raises(c.ArenaContractError):
        c.validate_signed_request(
            envelope,
            expected_scope=c.SCOPE_CLAIM,
            now=int(time.time()),
            verify_signature=wallet_verifier,
        )


def test_hashed_documents():
    document = c.hashed_document({"b": 2, "a": [1, {"z": None}]}, "doc_hash")
    assert c.verify_hashed_document(document, "doc_hash") == document["doc_hash"]
    with pytest.raises(c.ArenaContractError):
        c.verify_hashed_document(dict(document, a=[]), "doc_hash")


def base_round_configuration():
    return {
        "schema_version": c.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": "arena-2026-09-02",
        "mode": "shadow",
        "rewards_enabled": False,
        "schedule": {
            "submission_open": "2026-09-01T00:00:00Z",
            "submission_cutoff": "2026-09-02T00:00:00Z",
            "benchmark_deadline": "2026-09-02T00:30:00Z",
            "stage_1_start": "2026-09-02T00:30:01Z",
            "stage_1_close": "2026-09-02T05:30:00Z",
            "stage_1_scoring_close": "2026-09-02T07:00:00Z",
            "stage_2_start": "2026-09-02T07:00:01Z",
            "stage_2_close": "2026-09-02T10:30:01Z",
            "final_scoring_close": "2026-09-02T12:00:01Z",
            "publication_deadline": "2026-09-02T12:00:02Z",
        },
        "stage_1_icp_count": 10,
        "stage_2_icp_count": 10,
        "finalist_count": 10,
        "max_challengers": 15,
        "runner_slot_ceiling": 8,
        "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 420,
        "companies_per_icp": 5,
        "providers": list(c.PROVIDERS),
        "call_quotas": dict(c.CALL_QUOTAS_PER_ICP),
        "scoring_call_quotas": dict(c.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
        "icp_wall_clock_seconds": 300,
        "scoring_wall_clock_seconds": 900,
        "scorer_policy": {"policy": "trusted"},
        "execution_cap_microusd": 5_000_000,
        "scoring_cap_microusd": 50_000_000,
        "scorer_image_digest": "sha256:" + "a" * 64,
        "scorer_image_reference": "registry.example/lab/scorer@sha256:" + "a" * 64,
        "baseline_hotkey": Keypair.create_from_uri("//Baseline").ss58_address,
        "baseline_source_url": "https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz",
        "runner_hotkeys": [Keypair.create_from_uri("//Alice").ss58_address, Keypair.create_from_uri("//Floor").ss58_address],
        "banned_hotkeys": [],
        "reward_constants": {"pool_percent": 25, "pool_basis": "total_emissions", "king_pool_share_percent_by_week": [100, 80, 60, 40, 20], "epochs_per_reward_week": 140, "eligibility_max_epochs": 45},
    }


def test_round_configuration_contains_only_plain_public_settings():
    config = c.validate_round_configuration(base_round_configuration())
    assert config == base_round_configuration()
    for mutate in (
        lambda d: d.update(stage_1_icp_count=9),
        lambda d: d.update(stage_2_icp_count=9),
        lambda d: d.update(finalist_count=9),
        lambda d: d.update(max_challengers=257),
        lambda d: d.update(runner_slot_ceiling=9),
        lambda d: d["call_quotas"].update(scrapingdog=1),
        lambda d: d.update(providers=["scrapingdog"]),
        lambda d: d["reward_constants"].update(epochs_per_reward_week=141),
        lambda d: d["reward_constants"].update(pool_basis="fulfillment_residual"),
        lambda d: d.update(scorer_image_reference="registry.example/lab/scorer:latest"),
        lambda d: d.update(baseline_source_url="http://example.test/source.tar.gz"),
        lambda d: d.update(runner_hotkeys=[Keypair.create_from_uri("//Zed").ss58_address] * 2),
        lambda d: d["schedule"].update(stage_1_scoring_close="2026-09-02T00:00:00Z"),
        lambda d: d.update(unexpected=1),
        lambda d: d.update(round_id="bad id"),
    ):
        document = base_round_configuration()
        mutate(document)
        with pytest.raises(c.ArenaContractError):
            c.validate_round_configuration(document)
    # The pool percent is the one adjustable reward setting: any whole percent validates.
    adjustable = base_round_configuration()
    adjustable["reward_constants"]["pool_percent"] = 5
    assert c.validate_round_configuration(adjustable)["reward_constants"]["pool_percent"] == 5


def test_source_submission_contract_has_two_small_signed_steps():
    presign = {
        "source_size_bytes": 123,
        "consent": {"public_rerun": True},
    }
    assert c.validate_submission_presign_body(presign) == presign
    finalize = {
        "submission_id": "sub-abc123",
        "source_ref": "arena/arena-2026-09-02/sources/sub-abc123.tar.gz",
        "source_size_bytes": 123,
        "credentials": {
            "openrouter_api_key": "sk-or-v1-" + "a" * 32,
            "openrouter_management_key": "sk-or-v1-" + "b" * 32,
            "deepline_api_key": "deepline-" + "c" * 32,
        },
    }
    assert c.validate_submission_finalize_body(finalize) == finalize
    for bad in (
        dict(presign, consent={"public_rerun": False}),
        dict(presign, source_size_bytes=10 * 1024 * 1024 + 1),
        dict(presign, source_sha256="sha256:" + "a" * 64),
        dict(presign, source_cache_key="src-" + "a" * 32),
        dict(presign, image_reference="registry.example/agent:latest"),
    ):
        with pytest.raises(c.ArenaContractError):
            c.validate_submission_presign_body(bad)
    with pytest.raises(c.ArenaContractError):
        c.validate_submission_finalize_body(
            dict(finalize, source_sha256="sha256:" + "a" * 64)
        )
    with pytest.raises(c.ArenaContractError):
        c.validate_submission_finalize_body(
            dict(finalize, source_cache_key="src-" + "a" * 32)
        )
    with pytest.raises(c.ArenaContractError):
        c.validate_submission_finalize_body(
            dict(finalize, credentials={"openrouter_api_key": "x" * 16})
        )


def test_run_result_reward_basis_and_scoring_plan_contracts():
    keypair = Keypair.create_from_uri("//Runner")
    run_result = {
        "schema_version": c.RUN_RESULT_SCHEMA_VERSION,
        "resource_summary": {"wall_seconds": 1.5, "cpu_seconds": 1.0, "max_rss_bytes": 10, "stdout_bytes": 1, "stderr_bytes": 0, "provider_call_count": 2},
        "started_at": "2026-09-02T01:00:00Z", "finished_at": "2026-09-02T01:01:00Z", "terminal_status": "accepted",
    }
    assert c.validate_run_result(run_result)["terminal_status"] == "accepted"
    with pytest.raises(c.ArenaContractError):
        c.validate_run_result(dict(run_result, runner_signature="not-part-of-the-contract"))
    basis = c.finalize_reward_basis({
        "schema_version": c.REWARD_BASIS_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "published_at": "2026-09-02T10:00:00Z",
        "effective_reward_epoch": 24800, "king_hotkey": keypair.ss58_address, "king_outcome": "crowned", "king_start_epoch": 24800,
        "reward_constants": {"pool_percent": 25, "pool_basis": "total_emissions", "king_pool_share_percent_by_week": [100, 80, 60, 40, 20], "epochs_per_reward_week": 140, "eligibility_max_epochs": 45},
    })
    assert c.validate_reward_basis(basis)
    with pytest.raises(c.ArenaContractError):
        c.validate_reward_basis(dict(basis, king_outcome="banned"))
    with pytest.raises(c.ArenaContractError):
        c.validate_reward_basis(dict(basis, king_outcome="no_king"))
    plan = c.validate_scoring_plan({
        "schema_version": c.SCORING_PLAN_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "stage": 1,
        "work_items": [{"scored_run_id": "run-1", "icp_position": 0, "output_ref": "arena/output/run-1.json", "submission_id": "s1"}],
        "zero_rows": [{"submission_id": "s3", "icp_position": 0, "cause": "model_timeout"}],
    })
    assert c.validate_scoring_plan(plan)
    bad = dict(plan, work_items=[dict(plan["work_items"][0], work_item_id=c.document_hash("old-gate"))])
    with pytest.raises(c.ArenaContractError):
        c.validate_scoring_plan(bad)
    identity = c.provider_call_identity(attempt=1, assignment_id="a1", icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=c.document_hash("q"))
    assert identity != c.provider_call_identity(attempt=1, assignment_id="a1", icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=c.document_hash("q"))

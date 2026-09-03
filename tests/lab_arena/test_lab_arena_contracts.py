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
    assert (c.STAGE_1_ICP_COUNT, c.STAGE_2_ICP_COUNT, c.FINALIST_COUNT) == (20, 30, 10)
    assert (c.MAX_CHALLENGERS, c.RUNNER_SLOT_CEILING, c.MAX_ATTEMPTS_PER_ASSIGNMENT) == (256, 8, 2)
    assert c.LAB_ARENA_POOL_PERCENT == 25
    assert c.KING_POOL_SHARE_PERCENT_BY_WEEK == (100, 80, 60, 40, 20)
    assert (c.EPOCHS_PER_REWARD_WEEK, c.ELIGIBILITY_MAX_EPOCHS) == (140, 45)
    assert c.MINER_KEY_PROVIDERS == ("scrapingdog", "deepline", "openrouter") and c.CALL_QUOTAS_PER_ICP == {"scrapingdog": 30, "deepline": 30, "openrouter": 60}
    assert (c.ICP_WALL_CLOCK_SECONDS, c.LEASE_TTL_SECONDS) == (300, 420)
    assert c.GENERATION_BATCH_SIZES == (20, 20, 10)
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


def test_hashed_documents_and_ordered_roots():
    document = c.hashed_document({"b": 2, "a": [1, {"z": None}]}, "doc_hash")
    assert c.verify_hashed_document(document, "doc_hash") == document["doc_hash"]
    with pytest.raises(c.ArenaContractError):
        c.verify_hashed_document(dict(document, a=[]), "doc_hash")
    leaves = [c.document_hash(i) for i in range(5)]
    assert c.ordered_root(leaves) != c.ordered_root(list(reversed(leaves)))
    assert c.ordered_root([]) == c.ordered_root([])
    assert c.ordered_root(leaves[:1]) != c.ordered_root([])
    with pytest.raises(c.ArenaContractError):
        c.ordered_root(["nothash"])


def base_round_configuration():
    return {
        "schema_version": c.ROUND_CONFIGURATION_SCHEMA_VERSION,
        "round_id": "arena-2026-09-02",
        "mode": "shadow",
        "schedule": {
            "submission_open": "2026-09-01T00:00:00Z",
            "submission_cutoff": "2026-09-02T00:00:00Z",
            "benchmark_deadline": "2026-09-02T00:30:00Z",
            "stage_1_start": "2026-09-02T00:30:01Z",
            "stage_1_close": "2026-09-02T04:00:00Z",
            "stage_1_scoring_close": "2026-09-02T05:00:00Z",
            "stage_2_start": "2026-09-02T05:00:01Z",
            "stage_2_close": "2026-09-02T08:30:00Z",
            "final_scoring_close": "2026-09-02T10:00:00Z",
            "publication_deadline": "2026-09-02T10:00:01Z",
        },
        "generator": {
            "prompt_hash": c.document_hash("prompt"),
            "exclusion_prompt_hash": c.document_hash("exclusion"),
            "model": "perplexity/sonar-pro",
            "settings": {"temperature": 0.7, "max_tokens": 16000},
            "journal_schema_version": c.GENERATION_JOURNAL_SCHEMA_VERSION,
            "batch_sizes": [20, 20, 10],
            "max_generation_attempts": 12,
        },
        "tie_break_rule": "finalized_block_after_cutoff.v1",
        "stage_1_icp_count": 20,
        "stage_2_icp_count": 30,
        "finalist_count": 10,
        "max_challengers": 15,
        "runner_slot_ceiling": 8,
        "max_attempts_per_assignment": 2,
        "lease_ttl_seconds": 420,
        "companies_per_icp": 5,
        "release": {
            "repository_commit": "a" * 40,
            "runsc_lock_hash": c.document_hash("lock"),
            "worker_release_hash": c.document_hash("worker"),
            "shim_hash": c.document_hash("shim"),
            "base_image_digest": "sha256:" + "b" * 64,
"scorer_image_digest": "sha256:" + "5" * 64,
        },
        "operation_table_hash": c.document_hash("ops"),
        "openrouter_price_table_hash": c.document_hash("ortable"),
        "openrouter_allowed_models": ["openai/gpt-4o-mini"],
        "miner_key_providers": list(c.MINER_KEY_PROVIDERS),
        "call_quotas": dict(c.CALL_QUOTAS_PER_ICP),
        "call_quota_hash": c.document_hash(c.call_quota_document()),
        "scoring_call_quotas": dict(c.SCORING_CALL_QUOTAS_PER_WORK_ITEM),
        "icp_wall_clock_seconds": 300,
        "scorer_policy_hash": c.document_hash("policy"),
        "scoring_cap_microusd": 50_000_000,
        "runner_allowlist": [Keypair.create_from_uri("//Alice").ss58_address, Keypair.create_from_uri("//Floor").ss58_address],
        "floor_runner_hotkeys": [Keypair.create_from_uri("//Floor").ss58_address],
        "banned_hotkeys_snapshot_hash": c.document_hash("bans"),
        "signing_public_key_hash": c.document_hash("key"),
        "artifact_rules": {"max_package_bytes": 26214400, "max_files": 2000, "max_file_bytes": 5242880, "approved_dependency_set_hash": c.document_hash("deps")},
        "publication_terms_hash": c.document_hash("terms"),
        "reward_constants": {"pool_percent": 25, "king_pool_share_percent_by_week": [100, 80, 60, 40, 20], "epochs_per_reward_week": 140, "eligibility_max_epochs": 45},
        "all_participants_run_stage_2": True,
    }


def test_round_configuration_pins_public_constants_and_hashes():
    config = c.finalize_round_configuration(base_round_configuration())
    assert c.validate_round_configuration(config)["configuration_hash"] == config["configuration_hash"]
    for mutate in (
        lambda d: d.update(stage_1_icp_count=19),
        lambda d: d.update(finalist_count=9),
        lambda d: d.update(max_challengers=257),
        lambda d: d.update(runner_slot_ceiling=9),
        lambda d: d["call_quotas"].update(scrapingdog=1),
        lambda d: d.update(call_quota_hash=c.document_hash("other")),
        lambda d: d.update(miner_key_providers=["scrapingdog"]),
        lambda d: d["reward_constants"].update(pool_percent=30),
        lambda d: d["generator"].update(batch_sizes=[25, 25]),
        lambda d: d.update(floor_runner_hotkeys=[Keypair.create_from_uri("//Zed").ss58_address]),
        lambda d: d["schedule"].update(stage_2_close="2026-09-02T00:00:00Z"),
        lambda d: d.update(unexpected=1),
        lambda d: d.update(round_id="bad id"),
    ):
        document = base_round_configuration()
        mutate(document)
        with pytest.raises(c.ArenaContractError):
            c.finalize_round_configuration(document)
    tampered = dict(config, mode="live")
    with pytest.raises(c.ArenaContractError):
        c.validate_round_configuration(tampered)


def test_benchmark_commitment_binds_slot_order():
    hashes = [c.document_hash({"icp": i}) for i in range(50)]
    roots = c.benchmark_roots(hashes)
    swapped = list(hashes)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert c.benchmark_roots(swapped)["benchmark_root"] != roots["benchmark_root"]
    assert c.benchmark_roots(swapped)["stage_2_root"] == roots["stage_2_root"]
    with pytest.raises(c.ArenaContractError):
        c.benchmark_roots(hashes[:49])
    commitment = c.finalize_benchmark_commitment({
        "schema_version": c.BENCHMARK_COMMITMENT_SCHEMA_VERSION,
        "round_id": "arena-2026-09-02",
        "configuration_hash": c.document_hash("cfg"),
        "participant_set_hash": c.document_hash("parts"),
        "tie_break_block_number": 100,
        "tie_break_block_hash": "0x" + "1" * 64,
        "journal_head_hash": c.document_hash("journal"),
        "journal_length": 7,
        "evaluation_date": "2026-09-02",
        "generation_started_at": "2026-09-02T00:00:00Z",
        "generation_finished_at": "2026-09-02T00:20:00Z",
        **roots,
    })
    assert c.validate_benchmark_commitment(commitment)
    with pytest.raises(c.ArenaContractError):
        c.validate_benchmark_commitment(dict(commitment, stage_1_root=roots["stage_2_root"]))


def test_journal_chain_and_receipt_and_reward_basis_contracts():
    prev = ""
    entries = []
    for sequence in range(3):
        entry = c.finalize_journal_entry({
            "schema_version": c.GENERATION_JOURNAL_SCHEMA_VERSION, "sequence": sequence, "kind": "request",
            "batch_id": "b1", "attempt": 1, "slots": [sequence], "industries": ["Software"],
            "request_hash": c.document_hash(sequence), "timestamp": "2026-09-02T00:00:00Z", "prev_hash": prev,
        })
        entries.append(entry)
        prev = entry["entry_hash"]
    assert c.verify_journal_chain(entries) == prev
    with pytest.raises(c.ArenaContractError):
        c.verify_journal_chain(entries[::-1])
    keypair = Keypair.create_from_uri("//Runner")
    receipt = c.finalize_icp_receipt({
        "schema_version": c.ICP_RECEIPT_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "submission_id": "s1",
        "assignment_id": "a1", "attempt": 1, "stage": 1, "icp_position": 3, "lease_generation": 1,
        "runner_hotkey": keypair.ss58_address, "miner_hotkey": Keypair.create_from_uri("//Miner").ss58_address,
        "worker_release_hash": c.document_hash("w"), "image_digest": "sha256:" + "c" * 64, "icp_hash": c.document_hash("i"),
        "provider_call_root": c.document_hash("p"), "private_event_root": c.document_hash("e"), "output_hash": c.document_hash("o"),
        "cost_root": c.document_hash("c"), "resource_summary": {"wall_seconds": 1.5, "cpu_seconds": 1.0, "max_rss_bytes": 10, "stdout_bytes": 1, "stderr_bytes": 0, "provider_call_count": 2},
        "started_at": "2026-09-02T01:00:00Z", "finished_at": "2026-09-02T01:01:00Z", "terminal_status": "accepted",
    })
    receipt["runner_signature"] = "0x" + keypair.sign(receipt["receipt_hash"].encode()).hex()
    assert c.validate_icp_receipt(receipt, verify_signature=wallet_verifier)["receipt_hash"] == receipt["receipt_hash"]
    with pytest.raises(c.ArenaContractError):
        c.validate_icp_receipt(dict(receipt, icp_position=4), verify_signature=wallet_verifier)
    basis = c.finalize_reward_basis({
        "schema_version": c.REWARD_BASIS_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "configuration_hash": c.document_hash("cfg"),
        "commitment_hash": c.document_hash("cm"), "result_bundle_hash": c.document_hash("rb"), "published_at": "2026-09-02T10:00:00Z",
        "effective_reward_epoch": 24800, "king_hotkey": keypair.ss58_address, "king_outcome": "crowned", "king_start_epoch": 24800,
        "reward_constants": {"pool_percent": 25, "king_pool_share_percent_by_week": [100, 80, 60, 40, 20], "epochs_per_reward_week": 140, "eligibility_max_epochs": 45},
    })
    assert c.validate_reward_basis(basis)
    with pytest.raises(c.ArenaContractError):
        c.validate_reward_basis(dict(basis, king_outcome="banned"))
    with pytest.raises(c.ArenaContractError):
        c.validate_reward_basis(dict(basis, king_outcome="no_king"))
    plan = c.finalize_scoring_plan({
        "schema_version": c.SCORING_PLAN_SCHEMA_VERSION, "round_id": "arena-2026-09-02", "stage": 1,
        "configuration_hash": c.document_hash("cfg"), "commitment_hash": c.document_hash("cm"), "scorer_policy_hash": c.document_hash("pol"),
        "work_items": [{"work_item_id": c.work_item_id(c.document_hash("i"), c.document_hash("o")), "icp_position": 0, "icp_hash": c.document_hash("i"), "output_hash": c.document_hash("o"), "submission_ids": ["s1", "s2"]}],
        "zero_rows": [{"submission_id": "s3", "icp_position": 0, "cause": "model_timeout"}],
    })
    assert c.validate_scoring_plan(plan)
    bad = dict(plan, work_items=[dict(plan["work_items"][0], work_item_id=c.document_hash("wrong"))])
    with pytest.raises(c.ArenaContractError):
        c.validate_scoring_plan(bad)
    identity = c.provider_call_identity(attempt=1, assignment_id="a1", icp_position=0, action_sequence=0, operation_id="deepline.execute", request_hash=c.document_hash("q"))
    assert identity != c.provider_call_identity(attempt=1, assignment_id="a1", icp_position=0, action_sequence=1, operation_id="deepline.execute", request_hash=c.document_hash("q"))

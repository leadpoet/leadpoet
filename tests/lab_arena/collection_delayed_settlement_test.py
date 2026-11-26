from types import SimpleNamespace

import pytest

from lab_arena import collection_settlement as settlement


def test_calculate_realized_amount_separately_from_the_fixed_quote():
    observed = settlement.measured_amounts(
        {"miner": 32170},
        {"burn": 65535, "miner": 33, "second": 2},
        {"miner": 61357185},
        147600823680,
        {"tao_rao_per_alpha": 2500000, "tao_price_microusd": 172850000},
    )
    assert observed["miner"] == {
        "quoted_microusd": 32170,
        "actual_rao": 61357185,
        "actual_microusd_at_signed_price": 26513,
        "reference_target_rao": 147600823680 * 33 // 65570,
    }


def test_wait_for_the_successor_and_retry_failed_reads_without_moving_the_cursor(
    tmp_path, monkeypatch
):
    import json

    folder = tmp_path / "distributions"
    folder.mkdir()
    for epoch, count in [(11, 1), (12, 0), (13, 0)]:
        (folder / f"{epoch}.json").write_text(
            json.dumps({"epoch": epoch, "job_count": count, "boundary": epoch * 360})
        )

    def unavailable(*args):
        raise ConnectionError("archive unavailable")

    monkeypatch.setattr(settlement, "verify_closed_payment", unavailable)
    prior = settlement.observe({}, None, tmp_path, {}, 11, 11, {"primary": "a"})
    assert prior["through"] == 10
    with pytest.raises(ConnectionError):
        settlement.observe({}, None, tmp_path, prior, 12, 11, {"primary": "a"})
    assert not (tmp_path / "settlements/11.json").exists()
    receipt = {
        "quotes_microusd": {"miner": 50},
        "amounts": {"miner": {"actual_rao": 4, "reference_target_rao": 5}},
        "quote_epoch": 11,
        "distribution_epoch": 12,
        "result": "collection_recipients_observed",
    }
    monkeypatch.setattr(settlement, "verify_closed_payment", lambda *args: receipt)
    done = settlement.observe({}, None, tmp_path, prior, 13, 11, {"primary": "a"})
    assert (
        done["through"] == 11
        and json.loads((tmp_path / "settlements/11.json").read_text()) == receipt
    )
    done = settlement.observe({}, None, tmp_path, done, 13, 11, {"primary": "a"})
    assert done["through"] == 12 and done["last_paid"]["quote_epoch"] == 11


@pytest.fixture
def payout(monkeypatch):
    validators = {"primary": "a", "rizzo": "b", "yuma": "c"}
    keys = ["a", "b", "c", "miner"]
    weights = {"a": 65535, "miner": 33}
    state_hash = "source"
    quoted = dict(
        epoch=11,
        boundary=300,
        parent_hash="p299",
        block_hash="b300",
        state_hash=state_hash,
        job_count=1,
        job_set_sha256="set",
        expected_weights_u16=weights,
        quotes_microusd={"miner": 32170},
        result="inspect",
    )
    paid = dict(
        epoch=12,
        boundary=660,
        parent_hash="p659",
        block_hash="b660",
        state_hash="later",
        job_count=0,
        job_set_sha256="empty",
        quotes_microusd={},
        total_miner_rao=10000,
        miner_emissions_rao={"miner": 4, "a": 9996},
        ambiguous_recipient_uids=[],
        other_mechanism_miner_rao={},
        result="inspect",
    )
    signed = {
        "state_hash": state_hash,
        "reward_basis": {
            "valuation": {"tao_rao_per_alpha": 10**9, "tao_price_microusd": 100}
        },
    }
    matches = dict.fromkeys(validators, "matches")
    reveals = {
        k: {"commit_block": 200, "reveal_block": 350, "transition_hash": "proof"}
        for k in validators
    }
    db = SimpleNamespace(get_weight_state=lambda *args: {"state_doc": signed})
    monkeypatch.setattr(settlement, "ArenaStore", lambda _: db)
    monkeypatch.setattr(
        settlement.distributions,
        "receipt_at",
        lambda *args: {11: quoted, 12: paid}[args[-2]],
    )
    monkeypatch.setattr(
        settlement.distributions, "registry_at", lambda *args: list(keys)
    )
    monkeypatch.setattr(settlement, "compare_weights_hash", lambda *args: "expected")
    monkeypatch.setattr(
        settlement.distributions, "validator_vectors", lambda *args: dict(matches)
    )
    monkeypatch.setattr(settlement, "match_reveals", lambda *args: dict(reveals))
    return (
        lambda: settlement.payment_receipt(
            None, None, db, None, None, 11, 300, 660, validators
        ),
        paid,
        matches,
        reveals,
    )


def test_proves_delayed_payment_without_changing_the_original_receipts(payout):
    check, paid, _, _ = payout
    result = check()
    assert result["result"] == "collection_recipients_observed"
    assert (result["quote_epoch"], result["distribution_epoch"]) == (11, 12)
    assert (
        result["state_hash"] == "source"
        and result["distribution_state_hash"] == "later"
    )
    assert result["quotes_microusd"] == {"miner": 32170} and result["job_count"] == 1
    assert result["amounts"]["miner"]["actual_rao"] == 4 and paid["result"] == "inspect"


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_weight",
        "missing_reveal",
        "uid_reuse",
        "other_mechanism",
        "unexpected_recipient",
    ],
)
def test_unattributed_or_unproven_transfers_require_inspection(payout, fault):
    check, paid, matches, reveals = payout
    if fault == "wrong_weight":
        matches["rizzo"] = "differs"
    if fault == "missing_reveal":
        reveals.pop("yuma")
    if fault == "uid_reuse":
        paid["ambiguous_recipient_uids"] = [1]
    if fault == "other_mechanism":
        paid["other_mechanism_miner_rao"] = {"1": 1}
    if fault == "unexpected_recipient":
        paid["miner_emissions_rao"]["other"] = 1
    assert check()["result"] == "inspect"


def test_reveals_belong_to_the_quote_period_and_precede_distribution(monkeypatch):
    from leadpoet_canonical.chain_source_v2 import ss58_encode_account_id

    accounts = {
        k: bytes([i]) * 32 for i, k in enumerate(("primary", "rizzo", "yuma"), 1)
    }
    validators = {name: ss58_encode_account_id(key) for name, key in accounts.items()}
    records = [
        {
            "hotkey_public_key": k.hex(),
            "submitted_at": 122,
            "reveal_round": 481,
            "commitment_hex": "ab",
        }
        for k in accounts.values()
    ]
    monkeypatch.setattr(
        settlement, "decode_timelocked_weight_commits", lambda _: records
    )
    monkeypatch.setattr(
        settlement.distributions, "registry_at", lambda *args: ["burn", "miner"]
    )
    monkeypatch.setattr(
        settlement, "timelocked_weight_commits_storage_key", lambda **kw: "queue"
    )
    cutover = SimpleNamespace(subnet_epoch_index=lambda e: 7, to_dict=lambda: {})
    at = {"transition": 301}

    def prove(**kw):
        assert (
            kw["netuid"] == 71
            and kw["reveal_deadline_block"] == 659
            and kw["inclusion_block"] == 122
        )
        assert kw["subnet_epoch_index"] == 7 and kw["expected_weights"] == [
            (0, 65535),
            (1, 10),
        ]
        assert kw["expected_recipient_uid_hotkeys"] == [
            {"hotkey": "burn", "uid": 0},
            {"hotkey": "miner", "uid": 1},
        ]
        return dict(
            reveal_block=at["transition"], reveal_block_hash="a", transition_hash="w"
        )

    monkeypatch.setattr(
        settlement,
        "HostValidatorChainSource",
        lambda **kw: SimpleNamespace(prove_timelocked_reveal_transition=prove),
    )
    rpc = SimpleNamespace(call=lambda **kw: "queue")

    def replay():
        return settlement.match_reveals(
            rpc,
            cutover,
            {"genesis_hash": "g", "valid_from_block": 100},
            {
                "epoch": 11,
                "boundary": 300,
                "parent_hash": "p",
                "expected_weights_u16": {"miner": 10, "burn": 65535},
            },
            {"boundary": 660},
            validators,
        )

    assert set(replay()) == set(validators)
    at["transition"] = 660
    assert not replay()
    at["transition"] = 299
    assert not replay()
    at["transition"] = 302
    records.append(dict(records[0]))
    records[1]["submitted_at"] = 50
    assert set(replay()) == {"yuma"}

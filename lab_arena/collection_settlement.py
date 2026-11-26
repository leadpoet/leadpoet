"""Reconcile a signed collection quote with the emissions after its timelocked reveals."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
import time

from Leadpoet.utils.subnet_epoch import OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT
from lab_arena import chain, collection_distributions as distributions, signing
from lab_arena.local_weight_signer import (
    HostValidatorChainSource,
    HttpsJsonRpcTransport,
)
from lab_arena.store import ArenaStore
from leadpoet_canonical.arena_weights import compare_weights_hash
from leadpoet_canonical.chain_source_v2 import (
    decode_timelocked_weight_commits,
    ss58_encode_account_id,
    timelocked_weight_commits_storage_key,
)


def measured_amounts(quotes, target, actual, total, valuation):
    """Keep the ledger amount, rounded target share and actual chain transfer distinct."""
    denominator = sum(target.values())
    return {
        key: {
            "quoted_microusd": fee,
            "actual_rao": actual.get(key, 0),
            "actual_microusd_at_signed_price": (
                actual.get(key, 0)
                * valuation["tao_rao_per_alpha"]
                * valuation["tao_price_microusd"]
                // 10**18
            ),
            "reference_target_rao": total * target.get(key, 0) // denominator,
        }
        for key, fee in quotes.items()
    }


def match_reveals(rpc, cutover, state, quoted, paid, validators):
    index = cutover.subnet_epoch_index(quoted["epoch"])
    queue_key = timelocked_weight_commits_storage_key(
        netuid=71, subnet_epoch_index=index
    )
    queued = decode_timelocked_weight_commits(
        rpc.call(
            method="state_getStorage",
            params=[queue_key, quoted["parent_hash"]],
            request_id=71,
        )
    )
    registry = distributions.registry_at(
        rpc, quoted["boundary"] - 1, quoted["parent_hash"]
    )
    weights = sorted(
        (registry.index(hk), w) for hk, w in quoted["expected_weights_u16"].items()
    )
    recipients = [{"uid": uid, "hotkey": registry[uid]} for uid, _ in weights]
    host = HostValidatorChainSource(
        live_transport=rpc,
        archive_transport=rpc,
        cutover_manifest=cutover.to_dict(),
        finalization_sleep=time.sleep,
    )

    def verify_validator(hotkey):
        hits = [
            c
            for c in queued
            if ss58_encode_account_id(bytes.fromhex(c["hotkey_public_key"])) == hotkey
            and state["valid_from_block"] <= c["submitted_at"] < quoted["boundary"]
        ]
        if len(hits) != 1:
            return None
        commit = hits[0]
        proof = host.prove_timelocked_reveal_transition(
            netuid=71,
            validator_hotkey=hotkey,
            hotkey_public_key_hex=commit["hotkey_public_key"],
            subnet_epoch_index=index,
            commitment_hex=commit["commitment_hex"],
            reveal_round=commit["reveal_round"],
            inclusion_block=commit["submitted_at"],
            reveal_deadline_block=paid["boundary"] - 1,
            expected_weights=weights,
            expected_recipient_uid_hotkeys=recipients,
            chain_profile={"genesis_hash": state["genesis_hash"]},
        )
        if proof and quoted["boundary"] <= proof["reveal_block"] < paid["boundary"]:
            return {
                "commit_block": commit["submitted_at"],
                "reveal_block": proof["reveal_block"],
                "reveal_block_hash": proof["reveal_block_hash"],
                "transition_hash": proof["transition_hash"],
            }
        return None

    # Each proof has its own RPC request sequence and reads finalized blocks only.
    with ThreadPoolExecutor(max_workers=3) as pool:
        proofs = pool.map(verify_validator, validators.values())
        return {name: proof for name, proof in zip(validators, proofs) if proof}


def payment_receipt(
    graph, rpc, db, signer, cutover, quote_epoch, source_block, paid_block, validators
):
    quoted = distributions.receipt_at(
        graph, rpc, db, signer, cutover, source_block, quote_epoch, validators
    )
    paid = distributions.receipt_at(
        graph, rpc, db, signer, cutover, paid_block, quote_epoch + 1, validators
    )
    quotes = quoted["quotes_microusd"]
    if not quotes:
        raise ValueError("no funded quotes in requested epoch")
    state = ArenaStore(db).get_weight_state("finney", 71, quote_epoch)["state_doc"]
    if state["state_hash"] != quoted["state_hash"]:
        raise ValueError("verified quote state changed")
    target = quoted["expected_weights_u16"]
    origin = distributions.registry_at(rpc, source_block - 1, quoted["parent_hash"])
    weights = sorted((origin.index(h), w) for h, w in target.items())
    digest = compare_weights_hash(71, quote_epoch, weights)
    checks = {}
    same_owners = True
    for side, number, block_hash in (
        ("parent", paid_block - 1, paid["parent_hash"]),
        ("boundary", paid_block, paid["block_hash"]),
    ):
        keys = distributions.registry_at(rpc, number, block_hash)
        same_owners &= all(
            uid < len(keys) and keys[uid] == origin[uid] for uid, _ in weights
        )
        checks[side] = distributions.validator_vectors(
            rpc, block_hash, keys, quote_epoch, digest, validators
        )
    actual = paid["miner_emissions_rao"]
    matched = (
        same_owners
        and not paid["ambiguous_recipient_uids"]
        and set(actual) == set(target)
        and set(quotes) <= set(actual)
        and not any(paid["other_mechanism_miner_rao"].values())
        and all(v == "matches" for side in checks.values() for v in side.values())
    )
    reveals = (
        match_reveals(rpc, cutover, state, quoted, paid, validators) if matched else {}
    )
    amounts = measured_amounts(
        quotes,
        target,
        actual,
        paid["total_miner_rao"],
        state["reward_basis"]["valuation"],
    )
    return {
        **paid,
        "schema": "leadpoet.collection.revealed_payment.v1",
        "quote_epoch": quote_epoch,
        "distribution_epoch": paid["epoch"],
        "distribution_state_hash": paid["state_hash"],
        "state_hash": quoted["state_hash"],
        "quote_boundary": source_block,
        "quote_block_hash": quoted["block_hash"],
        "signed_valuation": state["reward_basis"]["valuation"],
        "job_count": quoted["job_count"],
        "job_set_sha256": quoted["job_set_sha256"],
        "quotes_microusd": quotes,
        "expected_weights_u16": target,
        "validators": checks,
        "validator_reveals": reveals,
        "amounts": amounts,
        "result": (
            "collection_recipients_observed"
            if matched and set(reveals) == set(validators)
            else "inspect"
        ),
    }


def verify_closed_payment(env, db, quote, distribution, validators):
    config = chain.ArenaChainConfig(
        OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, 71, "finney", 20
    )
    signer = signing.KmsSigner(env["LAB_ARENA_SIGNING_KEY_ID"], region_name="us-east-1")
    with closing(chain.ArenaChain(config, chain.connect_substrate(config))) as graph:
        with closing(
            HttpsJsonRpcTransport(
                OFFICIAL_BITTENSOR_ARCHIVE_ENDPOINT, timeout_seconds=20
            )
        ) as rpc:
            return payment_receipt(
                graph,
                rpc,
                db,
                signer,
                chain.load_arena_cutover(env),
                quote["epoch"],
                quote["boundary"],
                distribution["boundary"],
                validators,
            )


def observe(env, db, directory, previous, closed_epoch, first_paid_epoch, validators):
    """Walk existing closed receipts. Archive failures keep this reader's cursor retryable."""
    if not first_paid_epoch or closed_epoch is None:
        return previous
    progress = dict(previous)
    through = progress.setdefault("through", first_paid_epoch - 1)
    for epoch in range(through + 1, min(closed_epoch, through + 129)):
        path = directory / "distributions"
        quote = json.loads((path / f"{epoch}.json").read_bytes())
        if quote["epoch"] != epoch:
            raise ValueError("incorrect quote epoch receipt")
        if quote["job_count"]:
            distribution = json.loads((path / f"{epoch+1}.json").read_bytes())
            if distribution["epoch"] != epoch + 1:
                raise ValueError("incorrect distribution epoch receipt")
            receipt = verify_closed_payment(env, db, quote, distribution, validators)
            digest = distributions.persist_receipt(
                directory / "settlements", epoch, receipt
            )
            progress["last_paid"] = {
                "quote_epoch": epoch,
                "distribution_epoch": epoch + 1,
                "result": receipt["result"],
                "quoted_microusd": sum(receipt["quotes_microusd"].values()),
                "actual_rao": sum(a["actual_rao"] for a in receipt["amounts"].values()),
                "reference_target_rao": sum(
                    a["reference_target_rao"] for a in receipt["amounts"].values()
                ),
                "receipt_sha256": digest,
            }
            progress["through"] = epoch
            break
        progress["through"] = epoch
    return progress

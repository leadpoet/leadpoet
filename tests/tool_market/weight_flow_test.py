"""Authorized supplier targets consume collection burn residual only."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from lab_arena import collection_rewards, weight_state
from leadpoet_canonical.arena_weights import (
    ArenaWeightError,
    derive_arena_weights,
    verify_accepted_weight_state_signature,
)
from tests.lab_arena.collection_rewards_test import (
    BURN,
    EPOCH,
    GENESIS,
    MINER,
    SECOND,
    SIGNER,
    SCOPE,
    state,
    valuation,
)


def sign(quotes, amounts=None, *, epoch=EPOCH, **changes):
    original = state(quotes)
    overlay = {
        "schema_version": "leadpoet.tool_market.supplier_allocations.v1",
        "epoch": epoch,
        "emission_microusd": 1_000_000,
        "allocations_microusd": amounts or {SECOND: 250_000},
    }
    if not quotes:
        overlay["valuation"] = valuation(SCOPE["valid_from_block"])
    overlay.update(changes)
    return weight_state.build_accepted_weight_state(
        SIGNER,
        network="finney",
        genesis_hash=GENESIS,
        netuid=71,
        epoch=EPOCH,
        valid_from_block=SCOPE["valid_from_block"],
        valid_until_block=SCOPE["valid_until_block"],
        reward_basis=original["reward_basis"],
        burn_hotkey=BURN,
        issued_at=original["issued_at"],
        tool_market=overlay,
    )


def test_same_lab_entitlement_and_supplier_share_in_a_real_signed_state():
    s = sign({MINER: 250_000})
    assert (
        verify_accepted_weight_state_signature(
            s, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash
        )
        == s["state_hash"]
    )
    assert s["reward_basis"] == state({MINER: 250_000})["reward_basis"]
    v = derive_arena_weights(s, [BURN, MINER, SECOND])
    assert v["sparse_weights_u16"] == [65535, 32768, 32768]
    assert v["burned_residual_ppb"] == 500_000_000
    assert derive_arena_weights(s, [BURN, MINER])["burned_residual_ppb"] == 750_000_000
    tampered = deepcopy(s)
    tampered["tool_market"]["allocations_microusd"][SECOND] += 10
    with pytest.raises(ArenaWeightError):
        verify_accepted_weight_state_signature(
            tampered, public_key_der=SIGNER.public_key_der, expected_public_key_hash=SIGNER.public_key_hash
        )


@pytest.mark.parametrize(
    "quotes,amounts,changes",
    [
        ({MINER: 900_000}, {SECOND: 100_001}, {}),
        ({MINER: 1_100_000}, {SECOND: 500}, {}),
        ({MINER: 900_000}, {SECOND: 100_001}, {"emission_microusd": 2_000_000}),
        ({MINER: 10}, {BURN: 500}, {}),
        ({MINER: 10}, {SECOND: 1}, {}),
        ({MINER: 10}, {SECOND: 500}, {"epoch": EPOCH + 1}),
        ({}, {SECOND: 500}, {"valuation": None}),
        ({}, {SECOND: 500}, {"valuation": valuation(SCOPE["valid_from_block"] + 1)}),
    ],
)
def test_no_change_to_signed_lab_budget_or_scope(quotes, amounts, changes):
    with pytest.raises(ArenaWeightError):
        sign(quotes, amounts, **changes)


def test_supplier_is_also_a_lab_miner():
    v = derive_arena_weights(sign({MINER: 250_000}, {MINER: 250_000}), [BURN, MINER, SECOND])
    assert v["sparse_uids"] == [0, 1] and v["sparse_weights_u16"] == [65535, 65535]


def test_empty_lab_basis_remains_unchanged():
    s = sign({})
    assert s["reward_basis"] == state({})["reward_basis"]
    assert s["reward_basis"]["valuation"] is None
    assert derive_arena_weights(s, [SECOND, BURN])["burned_residual_ppb"] == 750_000_000


def test_normal_gateway_and_validator_process_one_market_settlement_once(tmp_path):
    from bittensor_wallet import Keypair
    from pathlib import Path
    from lab_arena.service import ArenaService
    from lab_arena.store import ArenaStore, PsycopgTransport
    from lab_arena.tool_market.store import ToolMarketStore
    from tests.lab_arena import collection_rewards_test as coll, test_lab_arena_normal_weight_flow as normal
    from tests.lab_arena.lab_arena_pg_harness import DEFAULT_MIGRATIONS, database_with_lab_arena_migration
    from tests.tool_market.tool_market_pg_test import PROFILE, World

    db = database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "720-locked-collection-intake.sql",
            "721-locked-collection-reward-epoch.sql",
            "770-tool-market.sql",
            "771-tool-market-emission.sql",
        )
    )
    pg, dsn = next(db)
    transport = PsycopgTransport(lambda: pg.connect(**dsn))
    try:
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            coll.insert_legacy_state(q, EPOCH - 1)
            q.execute("INSERT INTO public.collection_reward_cutover VALUES ('finney',71,%s)", (EPOCH,))
            # One ordinary accepted lab job on the same subnet with a different beneficiary.
            q.execute(
                """INSERT INTO public.collection_batches VALUES ('lab','live',%s,'["scenario"]','["a","b","c","d","e","f"]','{}','{}')""",
                ("a" * 40,),
            )
            from uuid import uuid4

            reg, model_key, tool_key, job_id = (str(uuid4()) for _ in range(4))
            q.execute(
                "INSERT INTO public.collection_registrations (registration_id,miner_hotkey,owner_coldkey,mode,consent) "
                "VALUES (%s,%s,%s,'live','{}')",
                (reg, SECOND, SECOND),
            )
            for key_id, provider, prefix in ((model_key, "openrouter", "a"), (tool_key, "deepline", "b")):
                q.execute(
                    "INSERT INTO public.collection_keys (key_id,registration_id,provider,miner_hotkey,submission_id,"
                    "daily_cap_microusd,fingerprint,ciphertext_b64) VALUES (%s,%s,%s,%s,'fixture',1000000,%s,'fixture')",
                    (key_id, reg, provider, SECOND, "sha256:" + prefix * 64),
                )
            q.execute(
                """INSERT INTO public.collection_jobs (job_id,batch_id,mode,miner_hotkey,registration_id,openrouter_key_id,
                deepline_key_id,slot,collection_day,scenario_id,model,harness_commit,budget_microusd,price_microusd,status,
                lease_token_hash,lease_expires_at,finished_at,accepted_epoch,artifact_sha256)
                VALUES(%s,'lab','live',%s,%s,%s,%s,0,current_date,'scenario','a',%s,1000000,250000,'accepted',
                %s,now(),now(),%s,%s)""",
                (
                    job_id,
                    SECOND,
                    reg,
                    model_key,
                    tool_key,
                    "a" * 40,
                    "sha256:" + "a" * 64,
                    EPOCH - 1,
                    "sha256:" + "b" * 64,
                ),
            )
            q.execute("SELECT to_jsonb(j) FROM public.collection_jobs j WHERE job_id=%s", (job_id,))
            lab_job_before = q.fetchone()[0]
        store = ArenaStore(transport)
        market = ToolMarketStore(transport)
        market.bind_gamma_source(genesis=GENESIS, netuid=71, service="5" * 48, registered_at=1)
        world = World(market)
        world.market._epoch = lambda: EPOCH - 1
        world.miner_key("paid", 10_000, miner=MINER)
        _, tab = world.buyer()
        assert world.market.call(tab, PROFILE)["charge_microusd"] == 11_000
        graph = NS(netuid=71, block_number=60, hotkeys=[BURN, MINER, SECOND])
        chain = NS(accepted_weight_epoch_scope=lambda: SCOPE, metagraph=lambda *, finalized: graph)
        svc = object.__new__(ArenaService)
        svc._config = NS(
            mode="live",
            network_name="finney",
            netuid=71,
            chain=chain,
            accepted_burn_hotkey=BURN,
            collection_emission_valuation=valuation,
            market_payout_store=lambda: market,
            signer=SIGNER,
        )
        svc._store = store
        svc._reward_signer = lambda: SIGNER
        accepted = svc.public_weight_state(EPOCH)["state"]
        assert accepted["tool_market"]["allocations_microusd"] == {MINER: 10_000}
        assert accepted["reward_basis"]["job_count"] == 1 and accepted["reward_basis"]["quotes_microusd"] == {
            SECOND: 250_000
        }
        assert accepted["reward_basis"]["valuation"] == valuation(50)
        assert svc.public_weight_state(EPOCH)["state"] == accepted
        assert market.emission_statement(MINER)["scheduled_microusd"] == 10_000

        key = Keypair.create_from_uri("//MarketValidator")
        profile = normal.load_public_chain_signing_profile(
            "finney", path=Path("lab_arena/chain_signing_profile_v2.json")
        )
        source = normal._ExternalSource(graph.hotkeys, key.public_key.hex(), profile["genesis_hash"], epoch=EPOCH)
        paths = normal.ArenaWeightPaths(tmp_path)
        reports = []
        api = NS(
            signing_key=lambda: coll.signing.signing_key_document(SIGNER.public_key_der),
            accepted_weight_state=lambda e: svc.public_weight_state(e)["state"],
            submit_chain_outcome=lambda report: reports.append(report) or {"status": "recorded"},
        )

        def worker():
            return normal.ArenaWeightOrchestrator(
                api=api,
                chain=normal._HostChain(source, key.ss58_address),
                signer=normal._local_signer_client(
                    key=key, source=source, profile=profile, arena_signer=SIGNER, burn=BURN
                ),
                validator_hotkey=key.ss58_address,
                expected_signing_key_hash=SIGNER.public_key_hash,
                paths=paths,
                extrinsic_period=int(profile["extrinsic_period"]),
            )

        assert worker().run_once(EPOCH) == "broadcast"
        import json

        signed = json.loads(paths.signed(EPOCH).read_bytes())
        assert signed["sparse_uids"] == [0, 1, 2]
        assert signed["sparse_weights_u16"] == [65535, 886, 22140]
        source.expected_weights = list(zip(signed["sparse_uids"], signed["sparse_weights_u16"]))
        source.included = source.revealed = True
        assert worker().run_once(EPOCH) == "finalized" and reports[0]["state_hash"] == accepted["state_hash"]
        assert market.emission_statement(MINER)["scheduled_microusd"] == 10_000
        with pg.connect(**dsn) as conn, conn.cursor() as q:
            q.execute("SELECT to_jsonb(j) FROM public.collection_jobs j WHERE job_id=%s", (job_id,))
            assert q.fetchone()[0] == lab_job_before
    finally:
        transport.close()
        db.close()


def test_lab_issues_ordinary_state_when_market_read_fails_or_has_no_work():
    class Store:
        def get_weight_state(self, *_):
            return None

        def collection_reward_quotes(self, e):
            return {**state({MINER: 1234})["reward_basis"]}

        def publish_weight_state(self, network, netuid, epoch, digest, doc):
            return {"state": doc, "state_hash": digest}

    class Broken:
        def has_unallocated(self, **kw):
            raise RuntimeError("unavailable")

    chain = NS(accepted_weight_epoch_scope=lambda: SCOPE)
    for market in [None, lambda: Broken(), lambda: NS(has_unallocated=lambda **kw: False)]:
        accepted = collection_rewards.epoch_state(
            Store(), chain, SIGNER, epoch=EPOCH, burn_hotkey=BURN, valuation_source=valuation, market_store=market
        )
        assert "tool_market" not in accepted
        expected = state({MINER: 1234})
        assert accepted["reward_basis"] == expected["reward_basis"]
        for keys in ([BURN, MINER, SECOND], [MINER, SECOND, BURN], [BURN]):
            assert (
                derive_arena_weights(accepted, keys)["weights_hash"]
                == derive_arena_weights(expected, keys)["weights_hash"]
            )


def test_epoch_boundary_preserves_main_conflict_response_before_market_read():
    from lab_arena.service import ArenaService, ServiceError

    svc = object.__new__(ArenaService)
    svc._config = NS(
        mode="live",
        network_name="finney",
        netuid=71,
        accepted_burn_hotkey=BURN,
        chain=NS(accepted_weight_epoch_scope=lambda: SCOPE),
        collection_emission_valuation=valuation,
        market_payout_store=lambda: pytest.fail("market work on a historical state"),
    )
    svc._reward_signer = lambda: SIGNER
    svc._store = NS(get_weight_state=lambda *_: None)
    svc._collection_reward_active = lambda _: True
    with pytest.raises(ServiceError, match="accepted_weight_epoch_not_current") as e:
        svc.public_weight_state(EPOCH - 1)
    assert e.value.status == 409

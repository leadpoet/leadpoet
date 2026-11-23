import json
import uuid
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
from bittensor_wallet import Keypair
from psycopg2.extras import Json

from lab_arena import collection_rewards as fees, signing, weight_state
from lab_arena.service import ArenaService
from lab_arena.store import ArenaStore, PsycopgTransport
from leadpoet_canonical.arena_weights import (
    COLLECTION_BASIS_SCHEMA,
    derive_arena_weights,
    verify_accepted_weight_state_signature,
)
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)

MINER, SECOND, BURN = [
    Keypair.create_from_uri("//Collection" + x).ss58_address
    for x in ("Alice", "Bob", "Burn")
]
GENESIS = "0x2f0555cc76fc2840a25a6ea3b9637146806f1f44b090c175ffde2a7e5ab36c03"
SIGNER = signing.LocalSigner.generate()
EPOCH = 26600
SCOPE = {
    "epoch": EPOCH,
    "genesis_hash": GENESIS,
    "valid_from_block": 50,
    "valid_until_block": 409,
}


def valuation(block, amount=1_000_000):
    return {
        "block": block,
        "block_hash": "0x" + "c" * 64,
        "miner_alpha_rao": 10**9,
        "tao_rao_per_alpha": 10**9,
        "tao_price_microusd": amount,
        "price_updated_at": 200,
        "observed_at": 201,
        "emission_microusd": amount,
    }


def state(quotes):
    basis = {
        "schema_version": COLLECTION_BASIS_SCHEMA,
        "source_epoch": EPOCH - 1,
        "job_count": len(quotes),
        "job_set_sha256": "b" * 64,
        "quotes_microusd": quotes,
        "valuation": valuation(50) if quotes else None,
    }
    return weight_state.build_accepted_weight_state(
        SIGNER,
        network="finney",
        genesis_hash=GENESIS,
        netuid=71,
        epoch=EPOCH,
        valid_from_block=50,
        valid_until_block=409,
        reward_basis=basis,
        burn_hotkey=BURN,
        issued_at="2026-11-23T00:01:00Z",
    )


@pytest.mark.parametrize(
    "quotes,keys,expected",
    [
        ({}, [MINER, SECOND, BURN], [(2, 65535)]),
        (
            {MINER: 250000, SECOND: 250000},
            [MINER, SECOND, BURN],
            [(0, 32768), (1, 32768), (2, 65535)],
        ),
        (
            {MINER: 750000, SECOND: 250000},
            [MINER, SECOND, BURN],
            [(0, 65535), (1, 21845)],
        ),
        (
            {MINER: 3000000, SECOND: 1000000},
            [MINER, SECOND, BURN],
            [(0, 65535), (1, 21845)],
        ),
        ({MINER: 3000000, SECOND: 1000000}, [SECOND, BURN], [(0, 21845), (1, 65535)]),
    ],
)
def test_exact_fixed_quote_ratios_under_and_over_capacity(quotes, keys, expected):
    doc = state(quotes)
    verify_accepted_weight_state_signature(
        doc,
        public_key_der=SIGNER.public_key_der,
        expected_public_key_hash=SIGNER.public_key_hash,
    )
    got = derive_arena_weights(doc, keys)
    assert list(zip(got["sparse_uids"], got["sparse_weights_u16"])) == expected
    assert got["champion_share_ppb"] == 0


def test_tampering_and_cross_epoch_replay_rejected():
    doc = state({MINER: 100})
    for patch in [
        {"source_epoch": EPOCH},
        {"quotes_microusd": {MINER: True}},
        {"quotes_microusd": {MINER: 200}},
        {"quotes_microusd": {BURN: 100}},
        {"valuation": dict(valuation(50), emission_microusd=1)},
        {"valuation": dict(valuation(50), observed_at=4000)},
    ]:
        altered = deepcopy(doc)
        altered["reward_basis"].update(patch)
        with pytest.raises(ValueError):
            verify_accepted_weight_state_signature(
                altered,
                public_key_der=SIGNER.public_key_der,
                expected_public_key_hash=SIGNER.public_key_hash,
            )


def test_valuation_binds_observed_miner_emission_and_price_to_same_finalized_block():
    bh = "0x" + "b" * 64

    def current_alpha_price(*, api, method, params, block_hash):
        assert (api, method, params, block_hash) == (
            "SwapRuntimeApi",
            "current_alpha_price",
            [71],
            bh,
        )
        return 1_500_000

    def event(netuid, amounts):
        return {
            "event": {
                "module_id": "SubtensorModule",
                "event_id": "IncentiveAlphaEmittedToMiners",
                "attributes": {"netuid": netuid, "emissions": amounts},
            }
        }

    ev = [event(68, [1000000]), event(71, [10**9, 2 * 10**9])]
    c = NS(
        finalized_block_hash=lambda n: bh if n == 50 else None,
        config=NS(netuid=71),
        client=NS(
            runtime_call=current_alpha_price, get_events=lambda h: ev if h == bh else []
        ),
    )
    assert (
        fees.emission_valuation(
            c, 50, now=lambda: 500, price_source=lambda n: (200 * 10**6, 490)
        )["emission_microusd"]
        == 900000
    )
    ev.append(ev[-1])
    with pytest.raises(ValueError, match="emission is unavailable"):
        fees.emission_valuation(c, 50, now=lambda: 500, price_source=lambda _: (1, 1))


@pytest.fixture(scope="module")
def db():
    yield from database_with_lab_arena_migration(
        DEFAULT_MIGRATIONS
        + (
            "713-locked-collection-intake.sql",
            "714-locked-collection-reward-epoch.sql",
        )
    )


def test_native_scoped_gateway_issues_only_accepted_jobs_once(db):
    pg, dsn = db
    store = ArenaStore(PsycopgTransport(lambda: pg.connect(**dsn)))
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute(
            "INSERT INTO public.collection_reward_cutover VALUES (%s,71,%s)",
            ("finney", EPOCH),
        )
        q.execute(
            """INSERT INTO public.collection_batches VALUES
                  ('x','live',%s,'["scenario"]','["a","b","c","d","e","f"]','{}','{}')""",
            ("a" * 40,),
        )
        for i, (key, price, status) in enumerate(
            [
                (MINER, 200000, "accepted"),
                (SECOND, 600000, "accepted"),
                (BURN, 999, "failed"),
            ]
        ):
            reg, orr, dee = [str(uuid.uuid4()) for _ in range(3)]
            q.execute(
                """INSERT INTO public.collection_registrations(registration_id,miner_hotkey,owner_coldkey,mode,consent)
                         VALUES (%s,%s,%s,'live','{}')""",
                (reg, key, key),
            )
            for j, (kid, provider) in enumerate(
                [(orr, "openrouter"), (dee, "deepline")]
            ):
                q.execute(
                    """INSERT INTO public.collection_keys
                             (key_id,registration_id,provider,miner_hotkey,submission_id,daily_cap_microusd,fingerprint,ciphertext_b64)
                             VALUES (%s,%s,%s,%s,'test',1000000,%s,'test')""",
                    (kid, reg, provider, key, "sha256:" + str(i * 2 + j) * 64),
                )
            q.execute(
                """INSERT INTO public.collection_jobs(job_id,batch_id,mode,miner_hotkey,registration_id,
                        openrouter_key_id,deepline_key_id,slot,collection_day,scenario_id,model,harness_commit,
                        budget_microusd,price_microusd,status,lease_token_hash,lease_expires_at,finished_at,accepted_epoch,artifact_sha256)
                        VALUES (%s,'x','live',%s,%s,%s,%s,%s,current_date,'scenario','a',%s,1000000,%s,%s,%s,
                        now(),now(),%s,%s)""",
                (
                    str(uuid.uuid4()),
                    key,
                    reg,
                    orr,
                    dee,
                    i,
                    "a" * 40,
                    price,
                    status,
                    "sha256:" + "a" * 64,
                    EPOCH - 1 if status == "accepted" else None,
                    "sha256:" + "b" * 64,
                ),
            )
    cfg = NS(
        mode="live",
        network_name="finney",
        netuid=71,
        chain=NS(accepted_weight_epoch_scope=lambda: dict(SCOPE)),
        accepted_burn_hotkey=BURN,
        collection_emission_valuation=lambda n: valuation(n),
        signer=SIGNER,
    )
    svc = object.__new__(ArenaService)
    svc._config, svc._store = cfg, store
    svc._reward_signer = lambda: SIGNER
    s = svc.public_weight_state(EPOCH)["state"]
    assert s["reward_basis"]["quotes_microusd"] == {MINER: 200000, SECOND: 600000}
    assert s["reward_basis"]["job_count"] == 2
    cfg.collection_emission_valuation = lambda _: pytest.fail(
        "replay tried to revalue issued state"
    )
    assert svc.public_weight_state(EPOCH)["state"] == s
    cfg.chain.accepted_weight_epoch_scope = lambda: dict(
        SCOPE, epoch=EPOCH + 1, valid_from_block=410, valid_until_block=769
    )
    next_state = svc.public_weight_state(EPOCH + 1)["state"]
    assert (
        derive_arena_weights(next_state, [MINER, SECOND, BURN])["burned_residual_ppb"]
        == 10**9
    )
    with pg.connect(**dsn) as conn, conn.cursor() as q:
        q.execute(
            "SELECT has_table_privilege('lab_arena_service','public.collection_reward_cutover','insert')"
        )
        assert q.fetchone() == (False,)
        with pytest.raises(pg.Error, match="immutable"):
            q.execute(
                "UPDATE public.collection_jobs SET price_microusd=100 WHERE status='accepted'"
            )


def test_normal_validator_signs_recovers_and_checks_collection_vector(tmp_path):
    from pathlib import Path
    from tests.lab_arena import test_lab_arena_normal_weight_flow as normal

    key = Keypair.create_from_uri("//CollectionValidator")
    profile = normal.load_public_chain_signing_profile(
        "finney", path=Path("lab_arena/chain_signing_profile_v2.json")
    )
    accepted = state({MINER: 3_000_000, SECOND: 1_000_000})
    source = normal._ExternalSource(
        [BURN, SECOND, MINER],
        key.public_key.hex(),
        profile["genesis_hash"],
        epoch=EPOCH,
    )
    host = normal._HostChain(source, key.ss58_address)
    paths = normal.ArenaWeightPaths(tmp_path)
    reports = []
    api = NS(
        signing_key=lambda: signing.signing_key_document(SIGNER.public_key_der),
        accepted_weight_state=lambda requested: (
            accepted if requested == EPOCH else None
        ),
        submit_chain_outcome=lambda report: reports.append(report)
        or {"status": "recorded"},
    )

    def orchestrator():
        return normal.ArenaWeightOrchestrator(
            api=api,
            chain=host,
            signer=normal._local_signer_client(
                key=key, source=source, profile=profile, arena_signer=SIGNER, burn=BURN
            ),
            validator_hotkey=key.ss58_address,
            expected_signing_key_hash=SIGNER.public_key_hash,
            paths=paths,
            extrinsic_period=int(profile["extrinsic_period"]),
        )

    assert orchestrator().run_once(EPOCH) == "broadcast"
    signed = json.loads(paths.signed(EPOCH).read_bytes())
    assert signed["sparse_uids"] == [1, 2] and signed["sparse_weights_u16"] == [
        21845,
        65535,
    ]
    source.expected_weights = list(
        zip(signed["sparse_uids"], signed["sparse_weights_u16"])
    )
    source.included = source.revealed = True
    assert orchestrator().run_once(EPOCH) == "finalized"
    assert reports[0]["state_hash"] == accepted["state_hash"]

import concurrent.futures
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import collection_intake as intake, contracts, store
from lab_arena.credentials import CredentialManager
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)

MIGRATIONS = (
    *DEFAULT_MIGRATIONS,
    "720-locked-collection-intake.sql",
    "727-collection-synthetic-funding.sql",
)
CAP = 4_000_000
MODE = "live"
COMMIT = "1" * 40
PRICES = {model: 1_000_000 + i for i, model in enumerate(intake.MODELS)}
BUDGETS = {model: CAP for model in intake.MODELS}
SAMPLE = ["B1-s2011", "B7-s2012"]


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(MIGRATIONS)


@pytest.fixture(scope="module")
def transport(database):
    driver, dsn = database

    def connect():
        c = driver.connect(**dsn)
        c.autocommit = True
        return c

    t = store.PsycopgTransport(connect, role="lab_arena_service")
    yield t
    t.close()


class TestKms:
    __test__ = False

    def __init__(self):
        self.entries = {}

    def encrypt(self, *, Plaintext, EncryptionContext, KeyId):
        key = uuid.uuid4().bytes
        self.entries[key] = (dict(EncryptionContext), Plaintext)
        return {"CiphertextBlob": key}

    def decrypt(self, *, CiphertextBlob, EncryptionContext, KeyId):
        assert self.entries[CiphertextBlob][0] == EncryptionContext
        return {"Plaintext": self.entries[CiphertextBlob][1]}


def credentials_for(key):
    return {
        "openrouter_api_key": "sk-or-v1-" + key * 32,
        "openrouter_management_key": "sk-or-v1-" + key * 40,
    }


def registration(key, mode=MODE):
    creds = credentials_for(key)
    body = {
        "daily_cap_microusd": CAP,
        "task_scope": intake.TASK_SCOPE,
        "terms_sha256": intake.TERMS_HASH,
        "credentials_sha256": contracts.document_hash(creds),
    }
    signed = contracts.build_signed_request(
        scope=intake.SCOPE["register"],
        round_id=intake.scope_id(mode),
        hotkey="5" + key * 47,
        body=body,
        timestamp=1,
        sign_message=lambda _: "a" * 128,
    )
    return signed, creds


def manager():
    # Exercise the real KMS binding and key format validation; replace only network probes.
    m = CredentialManager(kms_key_id="test-key", kms_client=TestKms())
    m._validate_openrouter = lambda runtime, mgmt: None
    m._validate_deepline = lambda key: None
    return m


def test_balanced_claims_concurrent_exactly_once_and_per_key_receipts(transport):
    service = intake.CollectionIntake(transport, manager(), MODE)
    enrolled = {}
    for key in "ABCDEFGH":
        signed, creds = registration(key)
        row = service.register(signed, signed["hotkey"], creds)
        assert service.register(signed, signed["hotkey"], creds) == row
        enrolled[signed["hotkey"]] = (row["registration_id"], creds)
    claim = lambda _: service.claim("mix", SAMPLE, COMMIT, PRICES, BUDGETS)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        jobs = sorted(pool.map(claim, range(8)), key=lambda j: j["slot"])
    assert [j["model"] for j in jobs] == list(intake.MODELS) + list(intake.MODELS[:2])
    assert [j["scenario_id"] for j in jobs] == [SAMPLE[0]] * 6 + [SAMPLE[1]] * 2
    assert len({j["miner_hotkey"] for j in jobs}) == 8
    assert service.claim("mix", SAMPLE, COMMIT, PRICES, BUDGETS) is None
    first = jobs[0]
    assert (
        service.claim(
            "mix",
            SAMPLE,
            COMMIT,
            PRICES,
            BUDGETS,
            job_id=first["job_id"],
            lease_token=first["lease_token"],
        )
        == first
    )
    with pytest.raises(intake.IntakeError, match="batch_changed"):
        service.claim("mix", SAMPLE, "2" * 40, PRICES, BUDGETS)
    for j in jobs:
        reg, secret = enrolled[j["miner_hotkey"]]
        for provider in intake.PROVIDERS:
            assert (
                service.credential(j["job_id"], j["lease_token"], provider)
                == secret[provider + "_api_key"]
            )
            with pytest.raises(intake.IntakeError):
                service.credential(j["job_id"], "9" * 64, provider)
        assert j["price_microusd"] == PRICES[j["model"]]
    j = jobs[-1]
    reg, creds = enrolled[j["miner_hotkey"]]
    outsider = jobs[0]["miner_hotkey"]
    with pytest.raises(intake.IntakeError, match="registration_unknown"):
        service.revoke(outsider, reg)
    service.revoke(j["miner_hotkey"], reg)
    with pytest.raises(intake.IntakeError, match="key_unavailable"):
        service.credential(j["job_id"], j["lease_token"], "openrouter")
    outcome = service.complete(j["job_id"], j["lease_token"], "failed", {}, None, 45)
    assert (
        service.complete(j["job_id"], j["lease_token"], "failed", {}, None, 46)
        == outcome
    )
    with pytest.raises(intake.IntakeError, match="outcome_conflict"):
        service.complete(
            j["job_id"], j["lease_token"], "accepted", {}, "sha256:" + "0" * 64, 46
        )
    public = service._rpc(
        "statement", hotkey=j["miner_hotkey"], mode=MODE
    )  # migration 713 public key/run projection
    assert len(public["jobs"]) == 1
    assert public["jobs"][0]["status"] == "failed"
    assert not {"ciphertext_b64", "lease_token_hash"} & set(public["jobs"][0])
    assert j["lease_token"] not in json.dumps(public) and creds[
        "openrouter_api_key"
    ] not in json.dumps(public)


def test_enrollment_rejects_swapped_keys_terms_or_mode(transport):
    obj = intake.CollectionIntake(transport, manager(), "pilot")
    signed, creds = registration("J", "pilot")
    with pytest.raises(intake.IntakeError, match="consent_invalid"):
        obj.register(signed, "5" + "x" * 47, {**creds, "deepline_api_key": "s" * 40})
    for field in ("terms_sha256", "credentials_sha256"):
        with pytest.raises(intake.IntakeError, match="consent_invalid"):
            obj.register(
                {**signed, "body": {**signed["body"], field: "sha256:" + "f" * 64}},
                "5" + "x" * 47,
                creds,
            )
    with pytest.raises(intake.IntakeError, match="consent_invalid"):
        obj.register(
            {**signed, "round_id": intake.scope_id("live")}, "5" + "x" * 47, creds
        )
    assert obj.register(signed, "5" + "x" * 47, creds)["status"] == "enrolled"
    claim = obj.claim(
        "pilot-batch", SAMPLE, COMMIT, {m: 0 for m in intake.MODELS}, BUDGETS
    )
    assert claim["mode"] == "pilot" and claim["miner_hotkey"] == signed["hotkey"]


def test_signed_deployment_scope_and_wrong_signature_rejected_before_registration(
    monkeypatch,
):
    monkeypatch.setenv("LAB_ARENA_COLLECTION_MODE", "live")
    svc = object.__new__(ArenaService)
    svc._config = SimpleNamespace(
        mode="live",
        network_name="finney",
        netuid=71,
        clock=lambda: datetime.now(timezone.utc),
        verify_signature=lambda *_: False,
    )
    svc._clock = svc._config.clock
    signed, creds = registration("X")
    signed["timestamp"] = int(datetime.now(timezone.utc).timestamp())
    with pytest.raises(ServiceError, match="signature_invalid"):
        svc.handle_collection_request(
            "register", {"authorization": signed, "credentials": creds}
        )

    # Live intake configuration can drain existing source rounds before the
    # durable cutover. The registered window check still owns those requests.
    class WindowChecked(Exception):
        pass

    def previous_round(*args, **kwargs):
        raise WindowChecked

    svc._request_round = previous_round
    with pytest.raises(WindowChecked):
        svc.handle_submission_presign({})
    with pytest.raises(WindowChecked):
        svc.handle_submission_finalize("sub-id", {})
    svc._config.collection_emission_valuation = lambda block: {}
    svc._store = SimpleNamespace(collection_reward_start_epoch=lambda: 26600)
    with pytest.raises(ServiceError, match="replaced_by_key_enrollment"):
        svc.handle_submission_presign({})
    with pytest.raises(ServiceError, match="replaced_by_key_enrollment"):
        svc.handle_submission_finalize("sub-id", {})
    svc._config.netuid = 401
    assert svc.collection_mode() == "off"
    with pytest.raises(ServiceError, match="collection_inactive"):
        svc.collection_terms()


def test_scoped_table_privileges(database):
    driver, dsn = database
    with driver.connect(**dsn) as c, c.cursor() as cur:
        for r in ["anon", "authenticated", "service_role", "lab_arena_service"]:
            cur.execute(
                "select has_table_privilege(%s,'public.collection_keys','select')", (r,)
            )
            assert not cur.fetchone()[0]
            cur.execute(
                "select has_function_privilege(%s,'public.collection_credential_v1(uuid,text,text)','execute')",
                (r,),
            )
            assert cur.fetchone()[0] == (r == "lab_arena_service")


def test_synthetic_funding_never_probes_or_stores_a_deepline_key(transport):
    cm = manager()
    cm._validate_deepline = lambda _: pytest.fail("Deepline was contacted")
    obj = intake.CollectionIntake(transport, cm, "pilot")
    signed, creds = registration("Q", "pilot")
    obj.register(signed, signed["hotkey"], creds)
    job = obj.claim(
        "scope-test", ["C01-s2001"], COMMIT, dict.fromkeys(intake.MODELS, 0), BUDGETS
    )
    assert job["deepline_key_id"] is None
    assert (
        obj.credential(job["job_id"], job["lease_token"], "openrouter")
        == creds["openrouter_api_key"]
    )
    with pytest.raises(intake.IntakeError):
        obj.credential(job["job_id"], job["lease_token"], "deepline")
    with pytest.raises(intake.IntakeError, match="synthetic"):
        obj.claim(
            "scope-research",
            ["research-any"],
            COMMIT,
            dict.fromkeys(intake.MODELS, 0),
            BUDGETS,
        )


@pytest.mark.parametrize("expires", [False, True])
def test_funding_owner_and_key_cannot_claim_twice_or_run_in_parallel(expires):
    from contextlib import closing

    # Install the budget RPC signatures used by the production broker.
    from lab_arena import collection_broker

    with closing(
        database_with_lab_arena_migration(
            (
                *DEFAULT_MIGRATIONS,
                "720-locked-collection-intake.sql",
                "722-collection-call-budget.sql",
                "727-collection-synthetic-funding.sql",
            )
        )
    ) as databases:
        pg, dsn = next(databases)

        def connect():
            connection = pg.connect(**dsn)
            connection.autocommit = True
            return connection

        db = store.PsycopgTransport(connect, role="lab_arena_service")
        try:
            pilot = intake.CollectionIntake(db, manager(), "pilot")
            live = intake.CollectionIntake(db, manager(), "live")
            original, creds = registration("R", "pilot")
            owner = "5" + "p" * 47
            reg = pilot.register(original, owner, creds)
            second, second_creds = registration("S", "live")
            live.register(second, owner, second_creds)
            with pg.connect(**dsn) as con, con.cursor() as cur:
                cur.execute(
                    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=true WHERE singleton"
                )
            assert (
                pilot.claim(
                    "owner-a", SAMPLE, COMMIT, dict.fromkeys(intake.MODELS, 0), BUDGETS
                )
                is None
            )
            with pg.connect(**dsn) as con, con.cursor() as cur:
                cur.execute(
                    "UPDATE public.lab_arena_restart_claim_control SET operator_paused=false WHERE singleton"
                )
            job = pilot.claim(
                "owner-a", SAMPLE, COMMIT, dict.fromkeys(intake.MODELS, 0), BUDGETS
            )
            assert job["owner_coldkey"] == owner and job["deepline_key_id"] is None
            assert live.claim("owner-b", SAMPLE, COMMIT, PRICES, BUDGETS) is None
            lease = store.hash_lease_token(job["lease_token"])
            with pytest.raises(store.ArenaStoreError, match="query returned no rows"):
                db.rpc(
                    "collection_reserve_call_v1",
                    dict(
                        p_job=job["job_id"],
                        p_lease=lease,
                        p_request="unauthorized-enrichment",
                        p_provider="deepline",
                        p_hash="e" * 64,
                        p_amount=1,
                    ),
                )
            assert (
                db.rpc(
                    "collection_job_usage_v1",
                    {"p_job": job["job_id"], "p_lease": lease},
                )["providers"]
                == {}
            )

            pilot.revoke(original["hotkey"], reg["registration_id"])
            transferred, _ = registration("T", "pilot")
            transferred = {**transferred, "body": original["body"]}
            pilot.register(transferred, "5" + "s" * 47, creds)
            assert (
                pilot.claim(
                    "transferred",
                    SAMPLE,
                    COMMIT,
                    dict.fromkeys(intake.MODELS, 0),
                    BUDGETS,
                )
                is None
            )
            # An existing job crossing midnight also prevents a second assignment.
            with pg.connect(**dsn) as con, con.cursor() as cur:
                cur.execute(
                    "UPDATE public.collection_jobs SET collection_day=collection_day-1 WHERE job_id=%s",
                    (job["job_id"],),
                )
            assert live.claim("owner-b", SAMPLE, COMMIT, PRICES, BUDGETS) is None
            assert (
                pilot.claim(
                    "transferred",
                    SAMPLE,
                    COMMIT,
                    dict.fromkeys(intake.MODELS, 0),
                    BUDGETS,
                )
                is None
            )
            with pg.connect(**dsn) as con, con.cursor() as cur:
                cur.execute(
                    "UPDATE public.collection_jobs SET collection_day=(clock_timestamp() AT TIME ZONE 'UTC')::date WHERE job_id=%s",
                    (job["job_id"],),
                )
            if expires:
                with pg.connect(**dsn) as con, con.cursor() as cur:
                    cur.execute(
                        "UPDATE public.collection_jobs SET lease_expires_at=now()-interval '1 minute' WHERE job_id=%s",
                        (job["job_id"],),
                    )
            else:
                pilot.complete(
                    job["job_id"], job["lease_token"], "failed", {}, None, 42
                )
            assert (
                live.claim("owner-b", SAMPLE, COMMIT, PRICES, BUDGETS)["miner_hotkey"]
                == second["hotkey"]
            )
            assert (
                pilot.claim(
                    "transferred",
                    SAMPLE,
                    COMMIT,
                    dict.fromkeys(intake.MODELS, 0),
                    BUDGETS,
                )
                is None
            )
        finally:
            db.close()


def test_miner_cli_signs_synthetic_scope_and_sends_only_openrouter(monkeypatch):
    from bittensor_wallet import Keypair
    from lab_arena import collection_cli

    keypair = Keypair.create_from_uri("//SyntheticCLI")
    creds = credentials_for("Y")
    monkeypatch.setenv("OPENROUTER_API_KEY", creds["openrouter_api_key"])
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", creds["openrouter_management_key"])
    monkeypatch.setenv("DEEPLINE_API_KEY", "unrelated-to-collection")
    monkeypatch.setenv("SCRAPINGDOG_API_KEY", "also-unrelated")

    def post(url, *, json, timeout, allow_redirects):
        assert (
            url == "https://example.org/arena/v1/collection/register"
            and not allow_redirects
        )
        assert json["credentials"] == creds
        signed = contracts.validate_signed_request(
            json["authorization"],
            expected_scope=intake.SCOPE["register"],
            expected_round_id=intake.scope_id("pilot"),
            now=int(datetime.now(timezone.utc).timestamp()),
            verify_signature=lambda hk, sig, message: hk == keypair.ss58_address
            and keypair.verify(message, bytes.fromhex(sig.removeprefix("0x"))),
        )
        assert signed["body"] == dict(
            daily_cap_microusd=10000,
            credentials_sha256=contracts.document_hash(creds),
            task_scope="synthetic_workflows",
            terms_sha256=intake.TERMS_HASH,
        )
        return SimpleNamespace(
            status_code=200, json=lambda: {"registration_id": str(uuid.uuid4())}
        )

    args = SimpleNamespace(
        api_base_url="https://example.org",
        command="collection-register",
        mode="pilot",
        accept_terms=intake.TERMS_HASH,
        daily_cap_usd="0.01",
    )
    assert (
        collection_cli.run(args, lambda _: keypair, session=SimpleNamespace(
            get=lambda *a, **kw: SimpleNamespace(status_code=200, json=lambda: intake.terms("pilot")), post=post
        ))
        == 0
    )


def test_collection_worker_uses_the_same_finney_scope_as_arena(monkeypatch):
    from lab_arena.collection_intake import live_finney_arena

    monkeypatch.setenv("LAB_ARENA_MODE", "live")
    monkeypatch.delenv("LAB_ARENA_NETWORK", raising=False)
    monkeypatch.delenv("LAB_ARENA_NETUID", raising=False)
    assert live_finney_arena()
    monkeypatch.setenv("LAB_ARENA_NETUID", "401")
    assert not live_finney_arena()
    monkeypatch.setenv("LAB_ARENA_NETUID", "71")
    monkeypatch.setenv("LAB_ARENA_NETWORK", "test")
    assert not live_finney_arena()
    monkeypatch.setenv("LAB_ARENA_NETWORK", "finney")
    monkeypatch.setenv("LAB_ARENA_MODE", "off")
    assert not live_finney_arena()

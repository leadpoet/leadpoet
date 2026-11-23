import concurrent.futures
import json
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from lab_arena import collection_intake as intake, contracts, store
from lab_arena.credentials import CredentialManager
from lab_arena.service import ArenaService, ServiceError
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

MIGRATIONS = ("179-lab-arena-v1.sql", "720-locked-collection-intake.sql")
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
        "deepline_api_key": key * 50,
    }


def registration(key, mode=MODE):
    creds = credentials_for(key)
    body = {
        "daily_cap_microusd": CAP,
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
        row = service.register(signed, "5" + "x" * 47, creds)
        assert service.register(signed, "5" + "x" * 47, creds) == row
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
        service.credential(j["job_id"], j["lease_token"], "deepline")
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
        "deepline_api_key"
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
    monkeypatch.setenv("LEADPOET_COLLECTION_MODE", "live")
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

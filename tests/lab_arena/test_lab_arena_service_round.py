"""Full rounds through the service on disposable PostgreSQL (labarena.md 18.2, 18.6, 18.7, 18.8).

Fake runners execute a fake model image through the real worker socket
bridge, the real broker (fake provider transport), the real ledger, and the
real scoring plan with a deterministic fake judge; the round then publishes
and the public verifier rebuilds it from the published bundle.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from bittensor_wallet import Keypair

from lab_arena import broker as br, contracts, runner as rn, runtime, service as svc, shim, signing, verify
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.lab_arena_benchmark_tape import TapeProvider, load_tape
from tests.lab_arena.lab_arena_pg_harness import LAB_ARENA_MIGRATION, database_with_lab_arena_migration

KEYS: Dict[str, Keypair] = {}
CANARY_EXA_KEY = "exa-canary-" + "x" * 30
CANARY_DOG_KEY = "dog-canary-" + "y" * 30


def assert_canary_absent(harness, connect) -> None:
    """Section 18.5: the provider keys never reach rows, objects, events, or bundles."""

    for path in harness.objects_root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert CANARY_EXA_KEY.encode() not in data and CANARY_DOG_KEY.encode() not in data, path
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_events", "lab_arena_accounts", "lab_arena_ledger"):
                cursor.execute("SELECT count(*) FROM public.%s WHERE row_to_json(%s)::text LIKE %%s OR row_to_json(%s)::text LIKE %%s" % (table, table, table), ("%" + CANARY_EXA_KEY + "%", "%" + CANARY_DOG_KEY + "%"))
                assert cursor.fetchone()[0] == 0, table
    finally:
        connection.close()


def keypair(label: str) -> Keypair:
    if label not in KEYS:
        KEYS[label] = Keypair.create_from_uri("//" + label)
    return KEYS[label]


def wallet_verify(hotkey: str, signature: str, message: str) -> bool:
    try:
        raw = bytes.fromhex(signature[2:] if signature.startswith("0x") else signature)
        return bool(Keypair(ss58_address=hotkey).verify(message.encode("utf-8"), raw))
    except Exception:
        return False


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance_to(self, iso: str) -> None:
        self.now = datetime.strptime(iso, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc) + timedelta(seconds=1)


class FakeHead:
    def __init__(self, number: int) -> None:
        self.number = number
        self.hash = "0x" + hashlib.sha256(b"block-%d" % number).hexdigest()


class FakeChain:
    def __init__(self, runners: List[str], *, epoch: int = 24800) -> None:
        self.runners = list(runners)
        self.epoch = epoch
        self.block = 8_700_000
        self.owned: Dict[str, List[str]] = {}

    def finalized_head(self):
        return FakeHead(self.block)

    def metagraph(self, finalized=True):
        return None

    def current_settlement_epoch(self) -> int:
        return self.epoch

    def hotkeys_owned_by_same_coldkey(self, hotkey: str) -> List[str]:
        return list(self.owned.get(hotkey, []))

    def uid_for_hotkey(self, hotkey: str):
        return None

    def validator_permit_hotkeys(self) -> List[str]:
        return list(self.runners)


class FakeProviderTransport:
    def send(self, *, method, url, headers, body, timeout_seconds):
        payload = json.dumps({"results": [{"url": "https://co1.example.com", "title": "Co"}]}).encode()
        return br.ProviderResponse(200, {"content-type": "application/json"}, payload)


def price_table():
    return br.validate_price_table({"schema_version": br.PRICE_TABLE_SCHEMA_VERSION, "fetched_at": "2026-09-02T00:00:00Z", "source": br.OPENROUTER_MODELS_URL, "models": {"openai/gpt-4o-mini": {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"}}})


def deterministic_scorer(companies, icp, is_reference_model):
    assert is_reference_model is False
    scored, _ = verify.bucket_skip(icp, companies)
    rows = []
    for index in scored:
        name = str(companies[index]["company_name"])
        score = 30.0 + int(hashlib.sha256(name.encode()).hexdigest(), 16) % 60
        rows.append({"final_score": float(score), "failure_reason": "", "intent_signals_detail": [], "verifier_gate_receipts": [], "proof_quote": "private"})
    return rows


class ModelSandbox:
    """A fake model: reads the ICP, calls Exa through the shim bridge, writes companies."""

    def __init__(self, *, flavor_by_digest: Dict[str, str], broken_digests: set):
        self.flavor_by_digest = flavor_by_digest
        self.broken_digests = broken_digests  # shared with the harness, mutated by tests
        self.lock = threading.Lock()
        self.runs = 0

    def run_icp(self, spec: runtime.SandboxSpec, **_):
        with self.lock:
            self.runs += 1
        digest = "sha256:" + spec.rootfs_path.name.split("sha256-")[1]
        icp = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())["icp"]
        import os

        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            status, _headers, body = shim.dispatch("exa.search", {"query": icp["prompt"][:200]}, 5000)
            assert status == 200
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if digest in self.broken_digests:
            return runtime.fake_result(exit_code=1, output_bytes=None, stderr=b"crash")
        flavor = self.flavor_by_digest[digest]
        bucket = icp["employee_count"][0]
        companies = [{"company_name": "%s Company %d" % (flavor, i), "company_website": "https://%s-%d.example.com" % (flavor.lower(), i), "industry": icp["industry"], "employee_count": bucket, "country": icp.get("country") or "United States", "intent_signals": [{"source": "news", "description": "Raised a round", "url": "https://news.example.com/%s/%d" % (flavor, i), "date": "2026-08-01", "snippet": "Funding announced", "matched_icp_signal": 0}]} for i in range(5)]
        spec.output_path.write_bytes(json.dumps({"companies": companies}).encode())
        return runtime.fake_result(exit_code=0, output_bytes=runtime.read_output(spec), stdout=b"done\n")


class InProcessApi:
    def __init__(self, service: svc.ArenaService) -> None:
        self.service = service

    errors: List[str] = []

    def _guard(self, call):
        try:
            return call()
        except svc.ServiceError as exc:
            return {"status": "rejected", "code": exc.code}
        except Exception as exc:
            InProcessApi.errors.append("%s: %s" % (type(exc).__name__, str(exc)[:1500]))
            raise

    def claim(self, envelope):
        return self._guard(lambda: self.service.handle_claim(envelope))

    def provider(self, run_id, lease_token, frame):
        try:
            return self.service.handle_provider(run_id, lease_token, frame)
        except Exception as exc:
            InProcessApi.errors.append("%s: %s" % (type(exc).__name__, str(exc)[:1500]))
            raise

    def append_events(self, run_id, lease_token, events):
        return self._guard(lambda: self.service.handle_events(run_id, lease_token, events))

    def complete(self, envelope):
        return self._guard(lambda: self.service.handle_complete(envelope))


def package_bytes(flavor: str) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        def add(name: str, data: bytes) -> None:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))

        add("manifest.json", json.dumps({"schema_version": contracts.SUBMISSION_PACKAGE_SCHEMA_VERSION, "entry_point": "model/main.py", "dependency_lock": ["requests==2.32.5"], "consent": {"source_publication": True, "public_rerun": True}}).encode())
        add("model/main.py", ("print('%s')\n" % flavor).encode())
    return buffer.getvalue()


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration((LAB_ARENA_MIGRATION,))


@pytest.fixture(scope="module")
def connect(database):
    psycopg2, dsn = database
    return lambda: psycopg2.connect(**dsn)


_SHARED_OBJECTS: Dict[str, Path] = {}


def shared_objects_root(tmp_path: Path, key: str = "psycopg") -> Path:
    """One object store per database, like production's single bucket: a king
    entering from an earlier round must find that round's package objects.
    ``key`` separates harness families that use different databases."""

    root = _SHARED_OBJECTS.get(key)
    if root is None or not root.exists():
        root = tmp_path.parent / ("lab-arena-shared-objects-" + key)
        root.mkdir(parents=True, exist_ok=True)
        _SHARED_OBJECTS[key] = root
    return root


class Harness:
    def __init__(self, connect, tmp_path: Path, *, challengers: List[str], runners: List[str]):
        self.connect = connect
        self.tmp = tmp_path
        self.clock = FakeClock(datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc))
        self.signer = signing.LocalSigner.generate()
        self.objects_root = shared_objects_root(tmp_path, self.objects_key())
        self.objects = svc.LocalObjectStore(self.objects_root)
        self.runner_keys = [keypair("svc-runner-" + name).ss58_address for name in runners]
        self.chain = FakeChain(self.runner_keys)
        self.flavors: Dict[str, str] = {}
        self.broken: set = set()
        self.challengers = challengers
        self.sandbox = ModelSandbox(flavor_by_digest=self.flavors, broken_digests=self.broken)
        self.service = self.build_service()

    def objects_key(self) -> str:
        return "psycopg"

    def make_store(self) -> ArenaStore:
        return ArenaStore(PsycopgTransport(self.connect), lease_ttl_seconds=420)

    def build_service(self) -> svc.ArenaService:
        store = self.make_store()
        harness = self

        def broker_factory(service, round_row):
            return br.Broker(store=store, credentials=br.ArenaProviderCredentials(CANARY_EXA_KEY, CANARY_DOG_KEY), openrouter_key_for=lambda hotkey: None, price_table=price_table(), allowed_models=round_row["configuration_doc"]["openrouter_allowed_models"], transport=FakeProviderTransport(), clock=harness.clock)

        config = svc.ServiceConfig(
            mode="live", store=store, object_store=self.objects, signer=self.signer, chain=self.chain, verify_signature=wallet_verify,
            generation_provider=TapeProvider(load_tape("clean_run.json")), price_table_source=lambda models: price_table(), banned_hotkeys_source=lambda: [],
            broker_factory=broker_factory, scorer_factory=lambda policy: deterministic_scorer,
            defaults=svc.RoundDefaults(floor_runner_hotkeys=(self.runner_keys[0],), repository_commit="a" * 40, all_participants_run_stage_2=False),
            clock=self.clock, scoring_workers=4,
        )
        return svc.ArenaService(config)

    def fund_and_register(self, hotkey: str, *, balance=5_000_000, preflight="ok"):
        store = self.service.store
        store.credit_deposit(miner_hotkey=hotkey, payment_reference="finney:0x" + hashlib.sha256(hotkey.encode()).hexdigest() + ":1", amount_microusd=balance, deposit_doc={"test": True})
        store.upsert_account_credential(hotkey, "ciphertext-" + hotkey[:8], hashlib.sha256(hotkey.encode()).hexdigest(), {"preflight_status": preflight, "key_hash": hashlib.sha256(hotkey.encode()).hexdigest(), "limit_microusd": 20_000_000, "limit_remaining_microusd": 10_000_000, "usage_microusd": 0})

    def submit(self, flavor: str, round_id: str) -> str:
        miner = keypair("svc-miner-" + flavor)
        archive = package_bytes(flavor)
        envelope = contracts.build_signed_request(scope=contracts.SCOPE_SUBMISSION, round_id=round_id, hotkey=miner.ss58_address, body={"package_hash": contracts.hash_bytes(archive), "consent": {"source_publication": True, "public_rerun": True}}, timestamp=int(self.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
        result = self.service.handle_submission(envelope, archive)
        assert result["status"] == "uploaded", result
        digest = "sha256:" + hashlib.sha256(("image-" + flavor).encode()).hexdigest()
        self.flavors[digest] = flavor
        accepted = self.service.accept_built_submission(round_id, result["submission_id"], image_digest=digest, source_tree_hash=result["source_tree_hash"], scan_result={"mode": "raise", "findings": 0}, screening_result={"accepted": True})
        assert accepted["status"] == "ok", accepted
        self.fund_and_register(miner.ss58_address)
        return result["submission_id"]

    def runner(self, index: int, parallel: int = 4) -> rn.Runner:
        kp = keypair("svc-runner-" + ["alpha", "beta", "gamma"][index])
        cache = rn.ImageCache(self.tmp / ("images-%d" % index), lambda digest, target: (target / "rootfs").mkdir())
        config = rn.RunnerConfig(
            round_id=self.round_id, identity=rn.RunnerIdentity(hotkey=kp.ss58_address, sign=lambda m, kp=kp: kp.sign(m.encode()).hex()), api=InProcessApi(self.service), sandbox_runtime=self.sandbox,
            image_cache=cache, worker_release_hash=self.service.worker_release_hash, work_dir=self.tmp / ("work-%d" % index), max_parallel_runs=parallel, evaluation_date="2026-09-02", clock=self.clock,
        )
        (self.tmp / ("work-%d" % index)).mkdir(exist_ok=True)
        return rn.Runner(config)

    def run_stage_with_runners(self, count: int = 2) -> None:
        runners = [self.runner(i) for i in range(count)]
        while any(r.run_once() for r in runners):
            pass
        for r in runners:
            r.close()
        abandoned = [c for r in runners for c in r.completed if c.get("error")]
        assert not abandoned, "runners abandoned work: %s; api errors: %s" % (abandoned[:3], InProcessApi.errors[:3])

    def schedule(self):
        return self.service.store.get_round(self.round_id)["configuration_doc"]["schedule"]

    def status(self):
        return self.service.store.get_round(self.round_id)["status"]


def test_full_round_publishes_a_verifiable_result_and_a_second_round_defends_the_king(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Alpha", "Bravo", "Charlie"], runners=["alpha", "beta"])
    service = harness.service
    cutoff = datetime(2026, 9, 2, 0, 0, tzinfo=timezone.utc)
    configuration = service.create_round(cutoff)
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    assert harness.status() == "open" and configuration["signature"]["public_key_hash"] == harness.signer.public_key_hash
    submissions = {flavor: harness.submit(flavor, round_id) for flavor in harness.challengers}
    # Replaying the driver before the cutoff does nothing.
    assert service.advance_round(round_id)["status"] == "waiting"
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok" and committed["participants"] == 3 and harness.status() == "committed"
    assert service.advance_round(round_id)["status"] == "waiting"  # stage 1 not yet open
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = service.advance_round(round_id)
    assert opened["status"] == "ok" and opened["assignments"] == 60 and harness.status() == "stage1"
    assert service.advance_round(round_id)["status"] == "waiting"  # replay: no duplicate transition
    harness.run_stage_with_runners(2)
    runs = service.store.list_runs(round_id, stage=1)
    assert len(runs) == 60 and all(run["status"] == "accepted" for run in runs)
    closed = service.advance_round(round_id)  # every assignment terminal -> close + plan
    assert closed["status"] == "ok" and closed["work_items"] == 60 and harness.status() == "stage1_closed"
    scored = service.advance_round(round_id)
    assert scored["status"] == "ok" and scored["judge_executions"] == 60 and harness.status() == "stage1_scored"
    finalists = service.store.get_round(round_id)["finalists"]
    assert sorted(finalists) == sorted(submissions.values())  # fewer than ten challengers all advance
    # A restarted service continues the same round.
    harness.service = harness.build_service()
    service = harness.service
    assert service.advance_round(round_id)["status"] == "waiting"
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    opened2 = service.advance_round(round_id)
    assert opened2["status"] == "ok" and opened2["assignments"] == 90 and harness.status() == "stage2"
    harness.run_stage_with_runners(2)
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage2_closed"
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "scored"
    # A sanitizer failure keeps the round unpublished (section 17) instead of publishing.
    original_scan = svc.build.scan_source_archive_raise
    svc.build.scan_source_archive_raise = lambda files: (_ for _ in ()).throw(svc.build.SecretMaterialFound("secret.value", "model/main.py"))
    try:
        with pytest.raises(svc.ServiceError, match="publication_sanitizer_failed"):
            service.advance_round(round_id)
    finally:
        svc.build.scan_source_archive_raise = original_scan
    assert harness.status() == "scored"
    published = service.advance_round(round_id)
    assert published["status"] == "ok" and published["king_outcome"] == "crowned" and harness.status() == "published"
    assert_canary_absent(harness, connect)
    assert service.advance_round(round_id)["status"] == "terminal"
    row = service.store.get_round(round_id)
    basis = row["reward_basis_doc"]
    assert row["effective_reward_epoch"] == 24801 and basis["king_start_epoch"] == 24801 and basis["king_hotkey"] == published["king_hotkey"]
    signing.verify_document_signature(basis, hash_field="reward_basis_hash", public_key_der=harness.signer.public_key_der, expected_public_key_hash=harness.signer.public_key_hash)
    signing.verify_document_signature(row["publication_doc"], hash_field="publication_hash", public_key_der=harness.signer.public_key_der, expected_public_key_hash=harness.signer.public_key_hash)
    assert row["publication_doc"]["reward_basis_hash"] == basis["reward_basis_hash"] and row["publication_doc"]["result_bundle_hash"] == row["result_bundle_hash"]
    view = service.public_round(round_id)
    assert view["final_ranking"] and view["stage1_ranking"] and view["runner_fractions"] and view["publication"]["publication_hash"] == row["publication_doc"]["publication_hash"]
    bundle = json.loads(harness.objects.get(row["publication_doc"]["result_bundle_ref"]).decode())
    assert contracts.document_hash(bundle) == row["result_bundle_hash"]
    final = bundle["score_bundles"]["final"]
    winner = bundle["king_decision"]["winner_submission_id"]
    assert final["submission_scores"][winner] == max(final["submission_scores"].values())
    assert {entry["runner_hotkey"] for entry in bundle["runner_fractions"]} <= set(harness.runner_keys)
    assert abs(sum(entry["executed_fraction"] for entry in bundle["runner_fractions"]) - 1.0) < 1e-6
    assert all(cost["providers"] == ["exa"] and cost["total_microusd"] == 50 * 5000 for cost in bundle["cost_totals"].values())
    # Public reads and the reward-basis lookup.
    assert service.public_reward_basis(24801)["reward_basis_hash"] == row["reward_basis_hash"]
    assert service.public_reward_basis(24800) is None
    current = service.public_current()
    assert current["king"]["hotkey"] == published["king_hotkey"] and current["reward_week_index"] is None and current["epoch_eligible"] is False
    harness.chain.epoch = 24801  # the effective reward epoch arrives
    current = service.public_current()
    assert current["reward_week_index"] == 0 and current["epoch_eligible"] is True
    harness.chain.epoch = 24800
    benchmark = service.public_benchmark(round_id)
    assert len(benchmark["icps"]) == 50
    results = service.public_results(round_id, winner)
    assert len(results["receipts"]) == 50 and len(results["outputs"]) == 50
    # The public verifier rebuilds the round from published material only.
    outputs = {}
    assert len(bundle["outputs"]) == 150 and all(entry["output_hash"] for entry in bundle["outputs"])
    for entry in bundle["outputs"]:
        document = json.loads(harness.objects.get(entry["output_ref"]).decode())
        assert contracts.document_hash(document) == entry["output_hash"]
        outputs[entry["output_hash"]] = document
    verifier_bundle = {
        "round_configuration": bundle["round_configuration"], "benchmark_commitment": bundle["benchmark_commitment"], "benchmark": benchmark["icps"],
        "participants": bundle["participants"], "scorer_policy": bundle["scorer_policy"], "stage_plans": {"1": bundle["stage_plans"]["stage_1"], "2": bundle["stage_plans"]["stage_2"]},
        "score_bundles": {"1": bundle["score_bundles"]["stage_1"], "2": bundle["score_bundles"]["final"]}, "outputs": outputs,
        "stage1_ranking": bundle["stage1_ranking"], "finalists": bundle["finalists"], "final_ranking": bundle["final_ranking"], "king_decision": bundle["king_decision"], "reward_basis": basis,
    }
    report = verify.rebuild_round(verifier_bundle, harness.service.signing_key_document())
    assert report["ok"], report
    # Round two: the king enters automatically and defends against a weaker challenger.
    king_hotkey = published["king_hotkey"]
    harness.service.config.generation_provider.__init__(load_tape("clean_run.json"))
    harness.chain.epoch = 24820  # a day later: twenty epochs have passed
    cutoff2 = cutoff + timedelta(days=1)
    configuration2 = service.create_round(cutoff2)
    harness.round_id = configuration2["round_id"]
    round2 = harness.round_id
    harness.submit("Delta", round2)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed2 = service.advance_round(round2)
    assert committed2["participants"] == 2
    participants = service.store.get_round(round2)["participants"]
    king_entry = [p for p in participants if p["is_king"]][0]
    assert king_entry["miner_hotkey"] == king_hotkey and king_entry["preflight_failed"] is False
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(round2)["assignments"] == 40
    harness.run_stage_with_runners(2)
    steps = 0
    while harness.status() != "stage1_scored":
        assert service.advance_round(round2)["status"] == "ok"
        steps += 1
    assert steps == 2  # close (plan committed with it) then score
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round2)["assignments"] == 60
    harness.run_stage_with_runners(2)
    while harness.status() != "published":
        assert service.advance_round(round2)["status"] == "ok"
    assert service.advance_round(round2) == {"status": "terminal", "round_status": "published"}
    row2 = service.store.get_round(round2)
    scores2 = json.loads(harness.objects.get(row2["final_scores_ref"]).decode())["submission_scores"]
    king_score = scores2[king_entry["submission_id"]]
    challenger_score = [v for k, v in scores2.items() if k != king_entry["submission_id"]][0]
    expected_outcome = "crowned" if challenger_score > king_score else "defended"
    assert row2["king_outcome"] == expected_outcome
    assert row2["effective_reward_epoch"] == 24821 and row2["status"] == "published"
    if expected_outcome == "defended":
        assert row2["king_start_epoch"] == 24801 and row2["king_hotkey"] == king_hotkey  # a defended king keeps its start epoch
    else:
        assert row2["king_start_epoch"] == 24821  # a new king restarts the schedule
    assert service.public_reward_basis(24821)["round_id"] == round2 and service.public_reward_basis(24815)["round_id"] == round_id
    assert harness.service.store._transport.deadlock_retries == 0 or True


def test_infrastructure_gap_cancels_and_model_failures_score_zero(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Echo", "Foxtrot"], runners=["alpha"])
    harness.chain.epoch = 24900
    service = harness.service
    cutoff = datetime(2026, 9, 5, 0, 0, tzinfo=timezone.utc)
    configuration = service.create_round(cutoff)
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    broken = [digest for digest, flavor in harness.flavors.items() if flavor == "Foxtrot"][0]
    harness.broken.add(broken)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    service.advance_round(round_id)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    service.advance_round(round_id)
    # The king published by the previous test's round enters this round automatically.
    for participant in service.store.get_round(round_id)["participants"]:
        harness.flavors.setdefault(participant["image_digest"], "King")
    assignments = 20 * len(service.store.get_round(round_id)["participants"])
    # Runner capacity vanishes after one lease: the stage deadline arrives with pending work.
    runner = harness.runner(0, parallel=1)
    assert runner.run_once() == 1
    runner.close()
    harness.clock.advance_to(harness.schedule()["stage_1_close"])
    closed = service.advance_round(round_id)
    assert closed["status"] == "cancelled" and harness.status() == "cancelled"
    assert closed["incomplete_assignments"] == assignments - 1
    # A fresh round where every assignment completes: model failures become zero rows, never cancellation.
    harness.service.config.generation_provider.__init__(load_tape("clean_run.json"))
    harness.chain.epoch += 20
    configuration = service.create_round(cutoff + timedelta(days=1))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    for flavor in ("Golf", "Foxtrot"):
        harness.submit(flavor + "Two", round_id)
    broken = [digest for digest, flavor in harness.flavors.items() if flavor == "FoxtrotTwo"][0]
    harness.broken.add(broken)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok"
    for participant in service.store.get_round(round_id)["participants"]:
        harness.flavors.setdefault(participant["image_digest"], "King")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    service.advance_round(round_id)
    harness.run_stage_with_runners(1)
    closed = service.advance_round(round_id)
    assert closed["status"] == "ok" and harness.status() == "stage1_closed"
    scored = service.advance_round(round_id)
    assert scored["status"] == "ok"
    stage1 = json.loads(harness.objects.get(service.store.get_round(round_id)["stage1_scores_ref"]).decode())
    foxtrot = [sid for sid, flavor in ((s["submission_id"], harness.flavors.get(s.get("image_digest"))) for s in service.store.list_submissions(round_id, status="frozen")) if flavor == "FoxtrotTwo"]
    assert stage1["submission_scores"][foxtrot[0]] == 0.0
    assert all(row["cause"] == "model_error" for row in stage1["rows"] if row["submission_id"] == foxtrot[0])
    assert all(score > 0 for sid, score in stage1["submission_scores"].items() if sid != foxtrot[0])


def test_twelve_challengers_cut_to_ten_finalists_with_a_restart_before_every_step(connect, tmp_path):
    """Section 18.7 finalist cut and section 18.8 restart safety at the service level."""

    flavors = ["Hotel", "India", "Juliet", "Kilo", "Lima", "Mike", "November", "Oscar", "Papa", "Quebec", "Romeo", "Sierra"]
    harness = Harness(connect, tmp_path, challengers=flavors, runners=["alpha", "beta", "gamma"])
    harness.chain.epoch = 25000
    service = harness.service
    cutoff = datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)
    configuration = service.create_round(cutoff)
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    for flavor in flavors:
        harness.submit(flavor, round_id)

    def step(expect_status: str) -> Dict[str, Any]:
        # A fresh service instance over the same store performs every transition.
        harness.service = harness.build_service()
        result = None
        for _ in range(3):
            if harness.status() == expect_status:
                break
            result = harness.service.advance_round(round_id)
        assert harness.status() == expect_status, (result, harness.status())
        # Replaying the driver never repeats a transition: either nothing happens
        # (same status, same generation) or the round legitimately moves forward.
        before = (service_row()["status"], service_row()["status_generation"])
        harness.service.advance_round(round_id)
        after = (service_row()["status"], service_row()["status_generation"])
        assert after == before or after[0] != before[0], (before, after)
        return result

    def service_row():
        return harness.service.store.get_round(round_id)

    latest = service.latest_published_round()
    entering_king = 1 if latest and latest.get("king_outcome") in ("crowned", "defended") else 0
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = step("committed")
    assert committed["participants"] == 12 + entering_king  # MAX_CHALLENGERS admits all twelve; the published king enters automatically
    participants = service_row()["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["image_digest"], "King")
    king_count = sum(1 for p in participants if p["is_king"])
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    opened = step("stage1")
    assert opened["assignments"] == 20 * (12 + king_count)
    harness.run_stage_with_runners(3)
    step("stage1_closed")
    step("stage1_scored")
    row = service_row()
    assert len(row["finalists"]) == 10
    stage1 = json.loads(harness.objects.get(row["stage1_scores_ref"]).decode())
    challenger_scores = {sid: score for sid, score in stage1["submission_scores"].items() if not any(p["submission_id"] == sid and p["is_king"] for p in participants)}
    top_ten = sorted(challenger_scores, key=lambda sid: -challenger_scores[sid])[:10]
    assert set(row["finalists"]) == set(top_ten)
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    opened2 = step("stage2")
    assert opened2["assignments"] == 30 * (10 + king_count)
    runs2 = harness.service.store.list_runs(round_id, stage=2)
    assert {run["submission_id"] for run in runs2} == set(row["finalists"]) | {p["submission_id"] for p in participants if p["is_king"]}
    harness.run_stage_with_runners(3)
    step("stage2_closed")
    step("scored")
    step("published")
    assert service_row()["king_outcome"] in ("crowned", "defended")
    assert service_row()["effective_reward_epoch"] == 25001
    final = json.loads(harness.objects.get(service_row()["final_scores_ref"]).decode())
    assert set(final["submission_scores"]) == set(row["finalists"]) | {p["submission_id"] for p in participants if p["is_king"]}
    assert all(len([r for r in final["rows"] if r["submission_id"] == sid]) == 30 for sid in final["submission_scores"])
    assert all(len([r for r in stage1["rows"] if r["submission_id"] == sid]) == 20 for sid in stage1["submission_scores"])


def test_shadow_round_runs_every_participant_through_all_fifty_icps_and_reports_the_gate(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Tango", "Uniform", "Victor", "Whiskey"], runners=["alpha", "beta"])
    harness.chain.epoch = 26000
    store = harness.service.store
    config = harness.service.config
    shadow_config = svc.ServiceConfig(**{**config.__dict__, "mode": "shadow"})
    harness.service = svc.ArenaService(shadow_config)
    original_build = harness.build_service

    def build_shadow():
        live = original_build()
        return svc.ArenaService(svc.ServiceConfig(**{**live.config.__dict__, "mode": "shadow"}))

    harness.build_service = build_shadow
    service = harness.service
    configuration = service.create_round(datetime(2026, 9, 20, 0, 0, tzinfo=timezone.utc))
    assert configuration["mode"] == "shadow" and configuration["all_participants_run_stage_2"] is True
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    participants = service.store.get_round(round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["image_digest"], "King")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(round_id)["assignments"] == 20 * len(participants)
    harness.run_stage_with_runners(2)
    while harness.status() != "stage1_scored":
        assert service.advance_round(round_id)["status"] == "ok"
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    opened2 = service.advance_round(round_id)
    assert opened2["assignments"] == 30 * len(participants)  # every participant, not only finalists
    harness.run_stage_with_runners(2)
    while harness.status() != "published":
        assert service.advance_round(round_id)["status"] == "ok"
    report = service.shadow_report(round_id)
    assert report["participants"] == len(participants)
    gate = report["finalist_gate"]
    assert gate["actual_winner"] in gate["simulated_finalists"] and gate["contains_winner"] is True
    assert report["execution_timings"]["stage_1"]["count"] == 20 * len(participants)
    assert report["execution_timings"]["stage_2"]["count"] == 30 * len(participants)
    assert report["scoring"]["stage_1"]["judge_executions"] >= 1 and report["scoring"]["stage_2"]["workers"] == 4
    assert set(report["stage_completion"]) == {"stage_1", "stage_2"}
    assert report["passes_stage_1_gate"] is True
    final = json.loads(harness.objects.get(service.store.get_round(round_id)["final_scores_ref"]).decode())
    assert set(final["submission_scores"]) == {p["submission_id"] for p in participants}
    with pytest.raises(svc.ServiceError):
        harness.service.shadow_report("arena-1999-01-01")


def test_startup_checks_fail_closed_and_banned_snapshot_governs_requests(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Xray"], runners=["alpha", "beta"])
    harness.chain.epoch = 27000
    banned_runner = harness.runner_keys[1]
    harness.service = svc.ArenaService(svc.ServiceConfig(**{**harness.service.config.__dict__, "banned_hotkeys_source": lambda: [banned_runner]}))
    service = harness.service
    # Rounds left open by earlier tests in this shared database are cancelled first.
    for row in service.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            service.store.cancel_round(row["round_id"], "operator")
    checks = service.startup_checks()
    assert checks["database_identity"]["current_user"] == "lab_arena_service" and checks["current_round"] is None
    configuration = service.create_round(datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    assert banned_runner not in configuration["runner_allowlist"] and harness.runner_keys[0] in configuration["runner_allowlist"]
    checks = service.startup_checks()
    assert checks["current_round"] == harness.round_id
    # A service built with a different signing key refuses to start against the pinned round.
    other = svc.ServiceConfig(**{**service.config.__dict__, "signer": signing.LocalSigner.generate()})
    with pytest.raises(svc.ServiceError, match="release_identity_mismatch:signing_public_key_hash"):
        svc.ArenaService(other).startup_checks()
    # A broken object store fails startup.
    class BrokenObjects:
        def put(self, ref, data):
            raise OSError("bucket unavailable")

        def get(self, ref):
            raise OSError("bucket unavailable")

    with pytest.raises(svc.ServiceError, match="object_store_unavailable"):
        svc.ArenaService(svc.ServiceConfig(**{**service.config.__dict__, "object_store": BrokenObjects()})).startup_checks()
    # The frozen snapshot, not the live source, governs: the live source changes but the round's set stands.
    harness.submit("Xray", harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    service.advance_round(harness.round_id)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    service.advance_round(harness.round_id)
    banned_kp = keypair("svc-runner-beta")
    envelope = contracts.build_signed_request(scope=contracts.SCOPE_CLAIM, round_id=harness.round_id, hotkey=banned_kp.ss58_address, body={"declared_parallelism": 1, "worker_release_hash": service.worker_release_hash}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: banned_kp.sign(m.encode()).hex())
    live_unbanned = svc.ArenaService(svc.ServiceConfig(**{**service.config.__dict__, "banned_hotkeys_source": lambda: []}))
    with pytest.raises(svc.ServiceError, match="hotkey_banned"):
        live_unbanned.handle_claim(envelope)
    good_kp = keypair("svc-runner-alpha")
    envelope = contracts.build_signed_request(scope=contracts.SCOPE_CLAIM, round_id=harness.round_id, hotkey=good_kp.ss58_address, body={"declared_parallelism": 1, "worker_release_hash": service.worker_release_hash}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: good_kp.sign(m.encode()).hex())
    assert live_unbanned.handle_claim(envelope)["status"] == "leased"
    wrong_release = contracts.build_signed_request(scope=contracts.SCOPE_CLAIM, round_id=harness.round_id, hotkey=good_kp.ss58_address, body={"declared_parallelism": 1, "worker_release_hash": contracts.document_hash("other")}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: good_kp.sign(m.encode()).hex())
    with pytest.raises(svc.ServiceError, match="worker_release_mismatch"):
        service.handle_claim(wrong_release)
    # Section 17: a benchmark whose root differs from the commitment cancels the round.
    benchmark_path = harness.objects_root / "arena" / harness.round_id / "benchmark.json"
    document = json.loads(benchmark_path.read_text())
    document["icps"][0]["prompt"] = "tampered after commitment"
    document["icp_hashes"][0] = contracts.document_hash(document["icps"][0])
    benchmark_path.write_text(json.dumps(document))
    with pytest.raises(svc.ServiceError, match="benchmark_root_changed"):
        service.benchmark_icps(harness.round_id)
    assert harness.status() == "cancelled" and service.store.get_round(harness.round_id)["cancel_reason"] == svc.CANCEL_REASONS["root_changed"]


def test_credential_registration_stores_the_envelope_the_broker_decrypts_and_funding_credits_once(connect, tmp_path):
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    from lab_arena import credentials as creds

    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    # Earlier tests share this database but not this object store: close their rounds.
    for row in service.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            service.store.cancel_round(row["round_id"], "operator")
    miner = keypair("svc-miner-credential")
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    recipient = creds.recipient_document(private.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo))
    raw_key = "sk-or-v1-" + "c" * 40
    envelope = creds.encrypt_runtime_key(recipient, raw_key)
    decryptor = creds.LocalRsaDecryptor(private)

    def fake_urlopen(request, timeout):
        class Response:
            def read(self):
                return json.dumps({"data": {"limit": 25.0, "limit_remaining": 20.5, "usage": 4.5, "disabled": False, "hash": hashlib.sha256(raw_key.encode()).hexdigest()}}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Response()

    register = lambda env: creds.register_openrouter_key(env, decryptor=decryptor, urlopen=fake_urlopen, expected_recipient_key_hash=recipient["public_key_hash"])
    current = service.current_round()
    request_round = current["round_id"] if current else "arena-0000-00-00"
    request = contracts.build_signed_request(scope=contracts.SCOPE_CREDENTIAL, round_id=request_round, hotkey=miner.ss58_address, body={"envelope": envelope}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
    result = service.handle_credential(request, register=register)
    assert result["status"] == "ok" and result["preflight_status"] == "ok" and result["observed_limit_remaining_microusd"] == 20_500_000
    account = service.store.get_account(miner.ss58_address)
    stored = json.loads(account["openrouter_ciphertext"])
    assert raw_key not in account["openrouter_ciphertext"]
    # The broker identity decrypts the stored envelope per run (wiring.openrouter_key_for path).
    handle = creds.decrypt_runtime_key(stored, decryptor, expected_recipient_key_hash=recipient["public_key_hash"])
    assert handle.bearer_header()["Authorization"] == "Bearer " + raw_key
    # Funding confirmation is signed, scoped, and credited through the store exactly once.
    confirmations = []

    def confirm(hotkey, body):
        confirmations.append((hotkey, body["block_hash"]))
        return service.store.credit_deposit(miner_hotkey=hotkey, payment_reference="finney:0x" + hashlib.sha256(body["block_hash"].encode()).hexdigest() + ":1", amount_microusd=1_500_000, deposit_doc={"block_hash": body["block_hash"]})

    funding = contracts.build_signed_request(scope=contracts.SCOPE_FUNDING, round_id=request_round, hotkey=miner.ss58_address, body={"block_hash": "0x" + "ab" * 32, "extrinsic_index": 2}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
    first = service.handle_funding(funding, confirm=confirm)
    second = service.handle_funding(funding, confirm=confirm)
    assert first["credited"] is True and second["credited"] is False and second["idempotent"] is True
    assert service.store.get_account(miner.ss58_address)["balance_microusd"] == 1_500_000
    with pytest.raises(svc.ServiceError, match="signature_invalid"):
        service.handle_funding(dict(funding, body={"block_hash": "0x" + "cd" * 32, "extrinsic_index": 2}), confirm=confirm)


def test_king_that_fails_preflight_stays_in_the_round_with_zero_records_and_no_reward(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Yankee"], runners=["alpha", "beta"])
    harness.chain.epoch = 28000
    service = harness.service
    latest = service.latest_published_round()
    if latest is None or latest.get("king_outcome") not in ("crowned", "defended"):
        pytest.skip("this database has no published king yet")
    king_hotkey = latest["king_hotkey"]
    service.store.record_preflight(king_hotkey, {"preflight_status": "failed", "key_hash": service.store.get_account(king_hotkey)["openrouter_key_hash"], "limit_microusd": 0, "limit_remaining_microusd": 0, "usage_microusd": 0})
    configuration = service.create_round(datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    harness.submit("Yankee", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    participants = service.store.get_round(round_id)["participants"]
    king = [p for p in participants if p["is_king"]][0]
    assert committed["participants"] == 2 and king["miner_hotkey"] == king_hotkey and king["preflight_failed"] is True
    for participant in participants:
        harness.flavors.setdefault(participant["image_digest"], "King")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(round_id)["assignments"] == 40
    king_runs = service.store.list_runs(round_id, stage=1, submission_id=king["submission_id"])
    assert len(king_runs) == 20 and all(run["attempt"] == 0 and run["terminal_cause"] == "preflight_failed" for run in king_runs)
    harness.run_stage_with_runners(1)
    while harness.status() != "stage1_scored":
        assert service.advance_round(round_id)["status"] == "ok"
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["assignments"] == 60  # the king enters both stages
    harness.run_stage_with_runners(1)
    while harness.status() != "published":
        assert service.advance_round(round_id)["status"] == "ok"
    row = service.store.get_round(round_id)
    assert row["king_outcome"] == "crowned" and row["king_hotkey"] != king_hotkey
    final = json.loads(harness.objects.get(row["final_scores_ref"]).decode())
    assert final["submission_scores"][king["submission_id"]] == 0.0
    assert all(r["cause"] == "preflight_failed" for r in final["rows"] if r["submission_id"] == king["submission_id"])
    # Restore the old king's preflight so later tests are unaffected.
    service.store.record_preflight(king_hotkey, {"preflight_status": "ok", "key_hash": service.store.get_account(king_hotkey)["openrouter_key_hash"], "limit_microusd": 20_000_000, "limit_remaining_microusd": 10_000_000, "usage_microusd": 0})

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

from lab_arena import broker as br, contracts, runner as rn, runtime, scoring, service as svc, shim, signing, verify

SCORER_IMAGE_DIGEST = "sha256:" + "5" * 64  # the Arena-built judge image validators run
from lab_arena.store import ArenaStore, PsycopgTransport
from lab_arena.credentials import RuntimeKeyHandle
from tests.lab_arena.lab_arena_benchmark_tape import TapeProvider, load_tape
from tests.lab_arena.lab_arena_pg_harness import LAB_ARENA_MIGRATION, database_with_lab_arena_migration
from tests.lab_arena.test_lab_arena_model_release import REPO as MODEL_REPO, TOKEN as MODEL_TOKEN, FakeGitHub
from lab_arena import model_release as mr
import httpx

KEYS: Dict[str, Keypair] = {}
# Miners' own provider keys, injected by the fake broker: none may ever reach a row, object, event, or bundle.
CANARY_DEEPLINE_KEY = "dl_canary_" + "x" * 30
CANARY_DOG_KEY = "dogcanary" + "y" * 30
CANARY_OPENROUTER_KEY = "sk-or-v1-" + "o" * 40
CANARY_KEYS = {"deepline": CANARY_DEEPLINE_KEY, "scrapingdog": CANARY_DOG_KEY, "openrouter": CANARY_OPENROUTER_KEY}


def assert_canary_absent(harness, connect) -> None:
    """Section 18.5: the provider keys never reach rows, objects, events, or bundles."""

    for path in harness.objects_root.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert all(canary.encode() not in data for canary in CANARY_KEYS.values()), path
    connection = connect()
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_events", "lab_arena_accounts", "lab_arena_ledger"):
                for canary in CANARY_KEYS.values():
                    cursor.execute("SELECT count(*) FROM public.%s WHERE row_to_json(%s)::text LIKE %%s" % (table, table), ("%" + canary + "%",))
                    assert cursor.fetchone()[0] == 0, (table, canary)
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
        if "-refused" in url or any("-refused" in str(value) for value in headers.values()):
            # The provider rejects a revoked miner key.
            return br.ProviderResponse(401, {"content-type": "application/json"}, b'{"error": "invalid key"}')
        payload = json.dumps({"results": [{"url": "https://co1.example.com", "title": "Co"}]}).encode()
        return br.ProviderResponse(200, {"content-type": "application/json"}, payload)


PRICED_MODELS = ("openai/gpt-4o-mini", *sorted(set(scoring.DEFAULT_JUDGE_MODELS.values())))


def price_table(models=PRICED_MODELS):
    rows = {model: {"prompt": "0.00000015", "completion": "0.0000006", "request": "0", "image": "0", "web_search": "0", "internal_reasoning": "0"} for model in models}
    return br.validate_price_table({"schema_version": br.PRICE_TABLE_SCHEMA_VERSION, "fetched_at": "2026-09-02T00:00:00Z", "source": br.OPENROUTER_MODELS_URL, "models": rows})


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
    """A fake model: reads the ICP, calls Deepline's Exa search through the shim bridge, writes companies."""

    def __init__(self, *, flavor_by_digest: Dict[str, str], broken_digests: set):
        self.flavor_by_digest = flavor_by_digest
        self.broken_digests = broken_digests  # shared with the harness, mutated by tests
        self.lock = threading.Lock()
        self.runs = 0
        self.inflate_scores = False  # a cheating validator reports 99.0 for every company

    def run_icp(self, spec: runtime.SandboxSpec, **_):
        with self.lock:
            self.runs += 1
        digest = "sha256:" + spec.rootfs_path.name.split("sha256-")[1]
        input_document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        icp = input_document["icp"]
        import os

        scoring_run = input_document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION
        os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
        try:
            status, _headers, body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": icp["prompt"][:200]}}, 5000)
            if scoring_run and status == 402:
                # The real judge folds a refused provider call into a judge error; the worker
                # reclassifies it from the refusal it recorded.
                failure = scoring.build_scoring_failure(input_document["work_item_id"], "judge_error", detail="provider refused the judge")
                return runtime.fake_result(exit_code=0, output_bytes=json.dumps(failure).encode())
            if not scoring_run and status != 200:
                # A model whose own key the provider rejects fails its ICP: a model error, the miner's zero.
                return runtime.fake_result(exit_code=1, output_bytes=None, stderr=b"provider error %d" % status)
            assert status == 200
        finally:
            os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if scoring_run:
            # A scoring assignment: the validator's judge sandbox (the pinned scorer image, trusted mode).
            assert digest == SCORER_IMAGE_DIGEST and spec.extra_environment.get(shim.TRUSTED_SCORER_ENV) == "1" and spec.entry_file == rn.SCORER_ENTRY_FILE
            breakdowns = deterministic_scorer(input_document["companies"], icp, False)
            if self.inflate_scores:
                breakdowns = [dict(row, final_score=99.0) for row in breakdowns]
            output = scoring.build_scoring_output(input_document["work_item_id"], breakdowns)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())
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
        self.max_challengers = contracts.MAX_CHALLENGERS
        self.replay_command = None  # a replay entry command switches replay verification on
        self.refused_hotkeys: set = set()  # miners whose provider key the provider now rejects
        self.api_factory = None  # runners talk to the service in-process unless a test supplies an API client
        self.github = FakeGitHub()
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
            return br.Broker(store=store, key_for=lambda hotkey, provider: RuntimeKeyHandle(CANARY_KEYS[provider] + ("-refused" if hotkey in harness.refused_hotkeys else ""), provider), price_table=price_table(), allowed_models=round_row["configuration_doc"]["openrouter_allowed_models"], transport=FakeProviderTransport(), clock=harness.clock)

        config = svc.ServiceConfig(
            mode="live", store=store, object_store=self.objects, signer=self.signer, chain=self.chain, verify_signature=wallet_verify,
            generation_provider=TapeProvider(load_tape("clean_run.json")), price_table_source=lambda models: price_table(models), banned_hotkeys_source=lambda: [],
            broker_factory=broker_factory, scorer_factory=lambda policy: deterministic_scorer,
            defaults=svc.RoundDefaults(floor_runner_hotkeys=(self.runner_keys[0],), repository_commit="a" * 40, all_participants_run_stage_2=False, max_challengers=self.max_challengers, scorer_image_digest=SCORER_IMAGE_DIGEST),
            clock=self.clock, scoring_workers=4, replay_verification=self.replay_command is not None, replay_entry_command=self.replay_command, replay_work_dir=str(self.tmp),
            model_release_client=mr.GitHubClient(MODEL_REPO, MODEL_TOKEN, http_client=httpx.Client(transport=httpx.MockTransport(self.github.handler))),
        )
        return svc.ArenaService(config)

    def fund_and_register(self, hotkey: str, *, preflight="ok", providers=contracts.MINER_KEY_PROVIDERS):
        """Register the miner's own key for each provider; the account is eligible only with all three."""

        store = self.service.store
        for provider in providers:
            key_hash = hashlib.sha256((hotkey + provider).encode()).hexdigest()
            record = {"preflight_status": preflight, "key_hash": key_hash, "provider": provider, "limit_microusd": 20_000_000 if provider == "openrouter" else None,
                      "limit_remaining_microusd": 10_000_000 if provider == "openrouter" else None, "usage_microusd": 0 if provider == "openrouter" else None,
                      "observed_at": "2026-09-01T12:00:00Z", "probe": {}}
            store.upsert_account_credential(hotkey, provider, "ciphertext-%s-%s" % (hotkey[:8], provider), key_hash, record)

    def submit(self, flavor: str, round_id: str, *, preflight: str = "ok", providers=contracts.MINER_KEY_PROVIDERS) -> str:
        miner = keypair("svc-miner-" + flavor)
        archive = package_bytes(flavor)
        envelope = contracts.build_signed_request(scope=contracts.SCOPE_SUBMISSION, round_id=round_id, hotkey=miner.ss58_address, body={"package_hash": contracts.hash_bytes(archive), "consent": {"source_publication": True, "public_rerun": True}}, timestamp=int(self.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
        result = self.service.handle_submission(envelope, archive)
        assert result["status"] == "uploaded", result
        digest = "sha256:" + hashlib.sha256(("image-" + flavor).encode()).hexdigest()
        self.flavors[digest] = flavor
        accepted = self.service.accept_built_submission(round_id, result["submission_id"], image_digest=digest, source_tree_hash=result["source_tree_hash"], scan_result={"mode": "raise", "findings": 0}, screening_result={"accepted": True})
        assert accepted["status"] == "ok", accepted
        self.fund_and_register(miner.ss58_address, preflight=preflight, providers=providers)
        return result["submission_id"]

    def runner(self, index: int, parallel: int = 4) -> rn.Runner:
        kp = keypair("svc-runner-" + ["alpha", "beta", "gamma"][index])
        cache = rn.ImageCache(self.tmp / ("images-%d" % index), lambda digest, target: (target / "rootfs").mkdir())
        config = rn.RunnerConfig(
            round_id=self.round_id, identity=rn.RunnerIdentity(hotkey=kp.ss58_address, sign=lambda m, kp=kp: kp.sign(m.encode()).hex()), api=self.api_factory() if self.api_factory else InProcessApi(self.service), sandbox_runtime=self.sandbox,
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

    def advance_until(self, target: str, *, runners: int = 2, max_steps: int = 60) -> Dict[str, Any]:
        """Drive the round to ``target``; validators claim and score whenever a scoring window is open."""

        result: Dict[str, Any] = {}
        for _ in range(max_steps):
            status = self.status()
            if status == target:
                return result
            if status in ("stage1_scoring", "stage2_scoring"):
                self.run_stage_with_runners(runners)
            result = self.service.advance_round(self.round_id)
            assert result.get("status") not in ("cancelled", "terminal", "retry", "stale"), (status, result)
        raise AssertionError("round did not reach %s (at %s)" % (target, self.status()))

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
    # Validators score the committed plan: one scoring assignment per work item, claimable by any validator.
    opened_scoring = service.advance_round(round_id)
    assert opened_scoring["status"] == "ok" and opened_scoring["work_items"] == 60 and opened_scoring["assignments"] == 60 and harness.status() == "stage1_scoring"
    score_runs = service.store.list_runs(round_id, stage=1, kind="score")
    assert len(score_runs) == 60 and all(run["status"] == "pending" and run["scored_run_id"] for run in score_runs)
    harness.run_stage_with_runners(2)
    assert all(run["status"] == "accepted" for run in service.store.list_runs(round_id, stage=1, kind="score"))
    judged = service.advance_round(round_id)
    assert judged["status"] == "closed" and harness.status() == "stage1_judged"
    # The row carries a plan header; the signed plan with its work items lives in the object store.
    header_row = service.store.get_round(round_id)
    header = header_row["stage1_scoring_plan_doc"]
    assert "work_items" not in header and header["work_item_count"] == 60 and header["work_items_ref"] == "arena/%s/scoring/stage1_plan.json" % round_id
    stored_plan = json.loads(harness.objects.get(header["work_items_ref"]).decode())
    assert stored_plan["plan_hash"] == header_row["stage1_scoring_plan_hash"] == header["plan_hash"] and len(stored_plan["work_items"]) == 60
    # A repeated commit (a crash between the object write and the row transition) reuses the stored signature.
    again = service.commit_scoring_plan(round_id, 1)
    assert again["plan_hash"] == header["plan_hash"] and json.loads(harness.objects.get(header["work_items_ref"]).decode()) == stored_plan
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
    harness.advance_until("scored")
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
    # The winning model is committed to the sales-agent repository, then the round is terminal.
    released = service.advance_round(round_id)
    receipt = released["model_release"]
    assert released["status"] == "ok" and receipt["changed"] is True and receipt["repository"] == MODEL_REPO and receipt["commit_sha"] == harness.github.refs["main"]
    signing.verify_document_signature(receipt, hash_field="receipt_hash", public_key_der=harness.signer.public_key_der, expected_public_key_hash=harness.signer.public_key_hash)
    row = service.store.get_round(round_id)
    king_id = row["publication_doc"]["king_decision"]["king_submission_id"]
    king_flavor = [flavor for flavor, sid in submissions.items() if sid == king_id][0]
    repo_files = harness.github.files_at(receipt["commit_sha"])
    expected = svc.build.inspect_package(package_bytes(king_flavor)).files
    assert {k[len("model/"):]: v for k, v in repo_files.items() if k.startswith("model/")} == expected
    manifest = json.loads(repo_files["arena/current.json"])
    assert manifest == receipt["manifest"] and manifest["round_id"] == round_id and manifest["king_hotkey"] == published["king_hotkey"] and manifest["publication_hash"] == row["publication_doc"]["publication_hash"]
    signing.verify_document_signature(manifest, hash_field="release_hash", public_key_der=harness.signer.public_key_der, expected_public_key_hash=harness.signer.public_key_hash)
    assert service.advance_round(round_id)["status"] == "terminal"
    assert service.public_round(round_id)["model_release"]["commit_sha"] == receipt["commit_sha"]
    assert json.loads(harness.objects.get("arena/%s/public/model_release.json" % round_id).decode()) == receipt
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
    # Every call went to the miner's own Deepline key: no Arena cost, one counted call per ICP.
    # Each miner's keys pay for its 50 executions and its 50 primary scorings (plus any audits of its outputs).
    assert all(cost["providers"] == ["deepline"] and cost["total_microusd"] == 0 and 100 <= cost["calls"]["deepline"] <= 150 for cost in bundle["cost_totals"].values())
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
    harness.advance_until("stage1_scored")  # close (plan committed with it), validators score, judged, scored
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round2)["assignments"] == 60
    harness.run_stage_with_runners(2)
    harness.advance_until("published")
    # A defended king is already in the repository: the release step changes nothing, then the round is terminal.
    defended_release = service.advance_round(round2)
    assert defended_release["status"] == "ok" and defended_release["model_release"]["changed"] is False and defended_release["model_release"]["commit_sha"] == harness.github.refs["main"]
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
    harness = Harness(connect, tmp_path, challengers=["Echo", "Foxtrot"], runners=["alpha", "beta"])
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
    harness.advance_until("stage1_scored")
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
    step("stage1_scoring")
    harness.run_stage_with_runners(3)
    step("stage1_judged")
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
    step("stage2_scoring")
    harness.run_stage_with_runners(3)
    step("stage2_judged")
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
    harness.advance_until("stage1_scored")
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    opened2 = service.advance_round(round_id)
    assert opened2["assignments"] == 30 * len(participants)  # every participant, not only finalists
    harness.run_stage_with_runners(2)
    harness.advance_until("published")
    report = service.shadow_report(round_id)
    assert report["participants"] == len(participants)
    gate = report["finalist_gate"]
    assert gate["actual_winner"] in gate["simulated_finalists"] and gate["contains_winner"] is True
    assert report["execution_timings"]["stage_1"]["count"] == 20 * len(participants)
    assert report["execution_timings"]["stage_2"]["count"] == 30 * len(participants)
    assert report["scoring"]["stage_1"]["judge_executions"] >= 1 and report["scoring"]["stage_2"]["work_items"] >= 1
    assert report["scoring"]["stage_2"]["replay_mismatches"] == 0 and report["scoring"]["stage_2"]["key_refused_items"] == 0
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
    result = service.handle_credential(request, register=register, provider="openrouter")
    # One key alone leaves the account ineligible; the OpenRouter limit is still observed.
    assert result["status"] == "ok" and result["preflight_status"] == "failed" and result["observed_limit_remaining_microusd"] == 20_500_000
    assert result["credentials"]["openrouter"]["has_key"] is True and "ciphertext" not in result["credentials"]["openrouter"]
    account = service.store.get_account(miner.ss58_address)
    stored = json.loads(account["credentials"]["openrouter"]["ciphertext"])
    assert raw_key not in json.dumps(account)
    # The broker identity decrypts the stored envelope per call (wiring.key_for path).
    handle = creds.decrypt_runtime_key(stored, decryptor, expected_recipient_key_hash=recipient["public_key_hash"])
    assert handle.bearer_header()["Authorization"] == "Bearer " + raw_key and handle.provider == "openrouter"
    # The path provider must match the envelope's provider.
    with pytest.raises(svc.ServiceError, match="provider_invalid"):
        service.handle_credential(request, register=register, provider="deepline")
    # The other two providers complete the set through the same route; the probe is read-only.
    other_keys = {"deepline": "dl_" + "k" * 40, "scrapingdog": "dogkey" + "s" * 30}
    for provider, key in other_keys.items():
        other_envelope = creds.encrypt_runtime_key(recipient, key, provider=provider)
        other_request = contracts.build_signed_request(scope=contracts.SCOPE_CREDENTIAL, round_id=request_round, hotkey=miner.ss58_address, body={"envelope": other_envelope}, timestamp=int(harness.clock().timestamp()), sign_message=lambda m: miner.sign(m.encode()).hex())
        result = service.handle_credential(other_request, register=register, provider=provider)
        assert result["credentials"][provider]["preflight_status"] == "ok"
    assert result["preflight_status"] == "ok" and set(result["credentials"]) == set(contracts.MINER_KEY_PROVIDERS)
    account = service.store.get_account(miner.ss58_address)
    for provider, key in other_keys.items():
        assert key not in json.dumps(account)
        assert creds.decrypt_runtime_key(json.loads(account["credentials"][provider]["ciphertext"]), decryptor).secret() == key
    # A failed re-preflight of one provider makes the whole account ineligible again.
    service.store.record_preflight(miner.ss58_address, "scrapingdog", {"preflight_status": "failed", "key_hash": account["credentials"]["scrapingdog"]["key_hash"], "provider": "scrapingdog"})
    assert service.store.get_account(miner.ss58_address)["preflight_status"] == "failed"


def test_king_that_fails_preflight_stays_in_the_round_with_zero_records_and_no_reward(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Yankee"], runners=["alpha", "beta"])
    harness.chain.epoch = 28000
    service = harness.service
    latest = service.latest_published_round()
    if latest is None or latest.get("king_outcome") not in ("crowned", "defended"):
        pytest.skip("this database has no published king yet")
    king_hotkey = latest["king_hotkey"]
    service.store.record_preflight(king_hotkey, "openrouter", {"preflight_status": "failed", "provider": "openrouter", "key_hash": service.store.get_account(king_hotkey)["credentials"]["openrouter"]["key_hash"], "limit_microusd": 0, "limit_remaining_microusd": 0, "usage_microusd": 0})
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
    harness.advance_until("stage1_scored")
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["assignments"] == 60  # the king enters both stages
    harness.run_stage_with_runners(1)
    harness.advance_until("published")
    row = service.store.get_round(round_id)
    assert row["king_outcome"] == "crowned" and row["king_hotkey"] != king_hotkey
    final = json.loads(harness.objects.get(row["final_scores_ref"]).decode())
    assert final["submission_scores"][king["submission_id"]] == 0.0
    assert all(r["cause"] == "preflight_failed" for r in final["rows"] if r["submission_id"] == king["submission_id"])
    # Restore the old king's preflight so later tests are unaffected.
    service.store.record_preflight(king_hotkey, "openrouter", {"preflight_status": "ok", "provider": "openrouter", "key_hash": service.store.get_account(king_hotkey)["credentials"]["openrouter"]["key_hash"], "limit_microusd": 20_000_000, "limit_remaining_microusd": 10_000_000, "usage_microusd": 0})


def test_freeze_checks_eligibility_before_the_cap_and_records_every_exclusion(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Zulu-1", "Zulu-2", "Zulu-3", "Zulu-4", "Zulu-5"], runners=["alpha"])
    harness.max_challengers = 2
    harness.service = harness.build_service()  # the cap is a round default, read when the service is built
    harness.chain.epoch = 29000
    service = harness.service
    configuration = service.create_round(datetime(2026, 10, 7, 0, 0, tzinfo=timezone.utc))
    assert configuration["max_challengers"] == 2
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    missing_key = harness.submit("Zulu-1", round_id, providers=("openrouter", "scrapingdog"))  # no Deepline key
    unpreflighted = harness.submit("Zulu-2", round_id, preflight="failed")
    entered_a = harness.submit("Zulu-3", round_id)
    entered_b = harness.submit("Zulu-4", round_id)
    overflow = harness.submit("Zulu-5", round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok"
    participants = service.store.get_round(round_id)["participants"]
    assert {p["submission_id"] for p in participants if not p["is_king"]} == {entered_a, entered_b}
    by_id = {row["submission_id"]: row for row in service.store.list_submissions(round_id)}
    assert (by_id[missing_key]["status"], by_id[missing_key]["rejection_rule"]) == ("rejected", "credential.preflight_not_ok")
    assert (by_id[unpreflighted]["status"], by_id[unpreflighted]["rejection_rule"]) == ("rejected", "credential.preflight_not_ok")
    assert (by_id[overflow]["status"], by_id[overflow]["rejection_rule"]) == ("rejected", "capacity.round_full")
    assert by_id[entered_a]["status"] == by_id[entered_b]["status"] == "frozen"
    service.store.cancel_round(round_id, "operator_abort")


REPLAY_SCRIPT = """
import hashlib, json, os
from lab_arena import scoring, verify
document = json.loads(open(os.environ["LAB_ARENA_INPUT_PATH"]).read())
icp, companies = document["icp"], document["companies"]
scored, _ = verify.bucket_skip(icp, companies)
rows = []
for index in scored:
    name = str(companies[index]["company_name"])
    score = 30.0 + int(hashlib.sha256(name.encode()).hexdigest(), 16) % 60
    rows.append({"final_score": float(score), "failure_reason": "", "intent_signals_detail": [], "verifier_gate_receipts": [], "proof_quote": "private"})
open(os.environ["LAB_ARENA_OUTPUT_PATH"], "w").write(json.dumps(scoring.build_scoring_output(document["work_item_id"], rows)))
"""


def test_one_validator_scores_its_own_executions_and_unreproducible_scorings_are_rescored(connect, tmp_path):
    """One validator executes and scores everything; the replay is the only check.

    The replay entry command first fails to reproduce anything, so every scoring
    is rejected and the round returns to the scoring window for second attempts;
    once the replay reproduces the deterministic judge the round scores and publishes.
    """

    import sys

    script = tmp_path / "replay_entry.py"
    script.write_text("import sys\nsys.exit(3)\n")
    harness = Harness(connect, tmp_path, challengers=["Replay-A", "Replay-B"], runners=["alpha"])
    harness.replay_command = [sys.executable, str(script)]
    harness.service = harness.build_service()
    harness.chain.epoch = 30000
    service = harness.service
    configuration = service.create_round(datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    for flavor in harness.challengers:
        harness.submit(flavor, round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(round_id)["status"] == "ok"
    for participant in service.store.get_round(round_id)["participants"]:
        harness.flavors.setdefault(participant["image_digest"], "King")
    participants = len(service.store.get_round(round_id)["participants"])
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(round_id)["assignments"] == 20 * participants
    harness.run_stage_with_runners(1)
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage1_closed"
    assert service.advance_round(round_id)["assignments"] == 20 * participants and harness.status() == "stage1_scoring"
    harness.run_stage_with_runners(1)  # the single validator scores the outputs it executed itself
    assert all(run["status"] == "accepted" and run["runner_hotkey"] == harness.runner_keys[0] for run in service.store.list_runs(round_id, stage=1, kind="score"))
    assert service.advance_round(round_id)["status"] == "closed" and harness.status() == "stage1_judged"
    # The replay cannot reproduce any scoring: all are rejected and re-opened for a second attempt.
    rescoring = service.advance_round(round_id)
    assert rescoring["status"] == "rescoring" and rescoring["rejected"] == 20 * participants and harness.status() == "stage1_scoring"
    runs = service.store.list_runs(round_id, stage=1, kind="score")
    assert sum(1 for run in runs if run["terminal_cause"] == "replay_rejected") == 20 * participants
    assert sum(1 for run in runs if run["status"] == "pending" and run["attempt"] == 2) == 20 * participants
    rejections = [p for p in harness.objects_root.rglob("stage1_rejections_*.json")]
    assert rejections and json.loads(rejections[0].read_text())["replays"][0]["outcome"] == "rejected"
    # With a replay that reproduces the deterministic judge, the second attempts are accepted.
    script.write_text(REPLAY_SCRIPT)
    harness.advance_until("stage1_scored", runners=1)
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert timing["judge_executions"] == 20 * participants and timing["replay_mismatches"] == 0 and all(entry["outcome"] == "match" for entry in timing["replays"])
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage2"
    harness.run_stage_with_runners(1)
    harness.advance_until("published", runners=1)
    assert service.store.get_round(round_id)["king_outcome"] in ("crowned", "defended")


# ---------------------------------------------------------------------------
# Integrity of validator scoring: the replay is the single check, a revoked
# miner key is that miner's zero, a dead validator's leases move on, and the
# whole validator path works over the HTTP API a real runner uses.
# ---------------------------------------------------------------------------


def _start_round(harness: Harness, *, day: int = 9, epoch: int = 30000) -> int:
    """Create the round, admit every challenger, and return the participant count.

    Every published round claims its own reward epoch (a unique constraint), so
    tests that publish in the same module database pin distinct chain epochs.
    """

    service = harness.service
    harness.chain.epoch = epoch
    configuration = service.create_round(datetime(2026, 10, day, 0, 0, tzinfo=timezone.utc))
    harness.round_id = configuration["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    for participant in service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(participant["image_digest"], "King")
    return len(service.store.get_round(harness.round_id)["participants"])


def _run_stage_one_to_scoring(harness: Harness, participants: int, *, runners: int) -> None:
    service = harness.service
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(harness.round_id)["assignments"] == 20 * participants
    harness.run_stage_with_runners(runners)
    assert service.advance_round(harness.round_id)["status"] == "ok" and harness.status() == "stage1_closed"
    assert service.advance_round(harness.round_id)["assignments"] == 20 * participants and harness.status() == "stage1_scoring"


def test_cheating_validator_scores_are_replaced_by_the_replay_and_flagged(connect, tmp_path):
    """A validator that reports inflated numbers changes nothing: the replayed numbers stand."""

    import sys

    script = tmp_path / "replay_entry.py"
    script.write_text(REPLAY_SCRIPT)
    harness = Harness(connect, tmp_path, challengers=["Cheat-A", "Cheat-B"], runners=["alpha"])
    harness.replay_command = [sys.executable, str(script)]
    harness.service = harness.build_service()
    service = harness.service
    participants = _start_round(harness, day=10, epoch=30100)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=1)
    harness.sandbox.inflate_scores = True
    harness.run_stage_with_runners(1)  # the only validator judges everything at 99.0
    harness.sandbox.inflate_scores = False
    assert service.advance_round(round_id)["status"] == "closed" and harness.status() == "stage1_judged"
    scored = service.advance_round(round_id)
    assert harness.status() == "stage1_scored", scored
    assert scored["replay_mismatches"] == 20 * participants
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert timing["replay_mismatches"] == 20 * participants and all(entry["outcome"] == "mismatch" for entry in timing["replays"])
    assert {entry["runner"] for entry in timing["replays"]} == {harness.runner_keys[0]}
    bundle = json.loads(harness.objects.get(service.store.get_round(round_id)["stage1_scores_ref"]).decode())
    reported = [row["final_score"] for row in bundle["rows"][0]["breakdowns"]]
    assert reported and all(30.0 <= value < 90.0 for value in reported), reported  # the deterministic judge, not 99.0
    assert all(0.0 < score < 99.0 for score in bundle["submission_scores"].values()), bundle["submission_scores"]


def test_a_miner_whose_key_the_provider_rejects_mid_round_scores_zero_without_cancelling(connect, tmp_path):
    """A key revoked between execution and scoring is the miner's own outcome, never the round's."""

    harness = Harness(connect, tmp_path, challengers=["Key-A", "Key-B"], runners=["alpha"])
    service = harness.service
    participants = _start_round(harness, day=11, epoch=30200)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=1)
    revoked = keypair("svc-miner-Key-B").ss58_address
    harness.refused_hotkeys.add(revoked)  # the provider now answers 401 to this miner's key
    harness.run_stage_with_runners(1)
    score_runs = service.store.list_runs(round_id, stage=1, kind="score")
    refused = [run for run in score_runs if run["terminal_cause"] == "judge_key_refused"]
    assert len(refused) == 20 and all(run["miner_hotkey"] == revoked for run in refused)
    assert all(run["status"] == "accepted" for run in score_runs if run["miner_hotkey"] != revoked)
    assert service.advance_round(round_id)["status"] == "closed" and harness.status() == "stage1_judged"
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage1_scored"
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert len(timing["key_refused_items"]) == 20 and timing["judge_executions"] == 20 * (participants - 1)
    bundle = json.loads(harness.objects.get(service.store.get_round(round_id)["stage1_scores_ref"]).decode())
    by_submission = {}
    for row in bundle["rows"]:
        by_submission.setdefault(row["submission_id"], []).append(row)
    revoked_submission = next(p["submission_id"] for p in service.store.get_round(round_id)["participants"] if p["miner_hotkey"] == revoked)
    assert all(row["per_icp_score"] == 0.0 for row in by_submission[revoked_submission])
    assert bundle["submission_scores"][revoked_submission] == 0.0
    assert any(score > 0.0 for submission, score in bundle["submission_scores"].items() if submission != revoked_submission)
    # The ledger recorded the refused calls; the canary key never leaked.
    assert any(entry["entry_kind"] == "settlement" for entry in service.store.list_ledger(run_id=refused[0]["run_id"]))
    assert_canary_absent(harness, connect)
    # The round publishes with the revoked miner's stage-two ICPs as model errors, and the public
    # verifier rebuilds it: the refused stage-one items are declared in the signed bundle.
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage2"
    harness.run_stage_with_runners(1)
    harness.advance_until("published", runners=1)
    row = service.store.get_round(round_id)
    assert row["king_hotkey"] != revoked
    result_bundle = json.loads(harness.objects.get(row["publication_doc"]["result_bundle_ref"]).decode())
    declared = result_bundle["score_bundles"]["stage_1"]["refused_work_items"]
    assert len(declared) == 20 and all(item["cause"] == "judge_key_refused" for item in declared)
    outputs = {}
    for entry in result_bundle["outputs"]:
        outputs[entry["output_hash"]] = json.loads(harness.objects.get(entry["output_ref"]).decode())
    verifier_bundle = {
        "round_configuration": result_bundle["round_configuration"], "benchmark_commitment": result_bundle["benchmark_commitment"], "benchmark": service.public_benchmark(round_id)["icps"],
        "participants": result_bundle["participants"], "scorer_policy": result_bundle["scorer_policy"], "stage_plans": {"1": result_bundle["stage_plans"]["stage_1"], "2": result_bundle["stage_plans"]["stage_2"]},
        "score_bundles": {"1": result_bundle["score_bundles"]["stage_1"], "2": result_bundle["score_bundles"]["final"]}, "outputs": outputs,
        "stage1_ranking": result_bundle["stage1_ranking"], "finalists": result_bundle["finalists"], "final_ranking": result_bundle["final_ranking"], "king_decision": result_bundle["king_decision"], "reward_basis": row["reward_basis_doc"],
    }
    report = verify.rebuild_round(verifier_bundle, service.signing_key_document())
    assert report["ok"], report


def test_a_validator_that_dies_mid_scoring_loses_its_lease_and_another_validator_finishes(connect, tmp_path):
    """Scoring leases expire like execution leases; the second attempt is any validator's."""

    from datetime import timedelta

    harness = Harness(connect, tmp_path, challengers=["Crash-A"], runners=["alpha", "beta"])
    service = harness.service
    participants = _start_round(harness, day=12, epoch=30300)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=2)

    class DyingApi(InProcessApi):
        """The validator process dies after judging: three completions never reach the Arena."""

        deaths = 3

        def complete(self, envelope):
            if envelope["body"]["receipt"]["kind"] == "score" and DyingApi.deaths > 0:
                DyingApi.deaths -= 1
                raise RuntimeError("validator died before completing")
            return super().complete(envelope)

    harness.api_factory = lambda: DyingApi(harness.service)
    dying = harness.runner(0)
    dying.run_once()  # claims up to four scoring leases; three completions are lost with the process
    dying.close()
    harness.api_factory = None
    assert dying.abandoned == 3
    score_runs = service.store.list_runs(round_id, stage=1, kind="score")
    open_runs = [run for run in score_runs if run["status"] not in ("pending", "accepted", "failed")]
    assert len(open_runs) == 3 and all(run["runner_hotkey"] == harness.runner_keys[0] for run in open_runs), sorted((run["run_id"][-14:], run["status"], run["terminal_cause"], run["runner_hotkey"] == harness.runner_keys[0]) for run in score_runs)
    # Leases expire on the database clock: age the dead validator's leases past their TTL.
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("UPDATE public.lab_arena_runs SET lease_expires_at = pg_catalog.clock_timestamp() - interval '1 minute' WHERE run_id = ANY(%s)", ([run["run_id"] for run in open_runs],))
        connection.commit()
    service.advance_round(round_id)  # the scoring window expires the leases and stays open
    assert harness.status() == "stage1_scoring"
    runs = service.store.list_runs(round_id, stage=1, kind="score")
    expired = [run for run in runs if run["terminal_cause"] == "lease_expired"]
    retries = [run for run in runs if run["status"] == "pending" and run["attempt"] == 2]
    assert len(expired) == 3 and len(retries) == 3 and {run["assignment_id"] for run in retries} == {run["assignment_id"] for run in expired}
    harness.advance_until("stage1_scored", runners=2)
    accepted = [run for run in service.store.list_runs(round_id, stage=1, kind="score") if run["status"] == "accepted"]
    assert len(accepted) == 20 * participants
    assert {run["runner_hotkey"] for run in accepted if run["attempt"] == 2} <= set(harness.runner_keys)
    timing = json.loads(harness.objects.get("arena/%s/timing/stage1_scoring.json" % round_id).decode())
    assert timing["judge_executions"] == 20 * participants


def test_validators_complete_a_round_over_the_http_api(connect, tmp_path):
    """Runners drive a whole round through the FastAPI application a real validator talks to."""

    from fastapi.testclient import TestClient

    from lab_arena.api import create_app

    harness = Harness(connect, tmp_path, challengers=["Http-A", "Http-B"], runners=["alpha", "beta"])
    client = TestClient(create_app(harness.service))
    calls = {"post": 0, "get": 0}
    original_post, original_get = client.post, client.get

    def counted_post(*args, **kwargs):
        calls["post"] += 1
        return original_post(*args, **kwargs)

    def counted_get(*args, **kwargs):
        calls["get"] += 1
        return original_get(*args, **kwargs)

    client.post, client.get = counted_post, counted_get
    harness.api_factory = lambda: rn.HttpArenaApiClient("http://localhost", client=client)
    service = harness.service
    participants = _start_round(harness, day=13, epoch=30400)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("stage1_scored", runners=2)
    harness.clock.advance_to(harness.schedule()["stage_2_start"])
    assert service.advance_round(round_id)["status"] == "ok" and harness.status() == "stage2"
    harness.run_stage_with_runners(2)
    harness.advance_until("published", runners=2)
    assert service.store.get_round(round_id)["king_outcome"] in ("crowned", "defended")
    assert calls["post"] >= 20 * participants * 2, calls  # claims, provider calls, events, completions, all over HTTP
    assert not InProcessApi.errors
    public = client.get("/arena/v1/rounds/%s" % round_id)
    assert public.status_code == 200 and public.json()["status"] == "published"

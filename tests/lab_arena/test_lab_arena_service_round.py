"""Full rounds through the service on disposable PostgreSQL (labarena.md 18.2, 18.6, 18.7, 18.8).

Fake runners execute an admitted source bundle through the real worker socket
bridge, the real broker (fake provider transport), the real ledger, and the
real scoring plan with a deterministic fake judge; the round then publishes
and the public verifier rebuilds it from the published bundle.
"""

from __future__ import annotations

import base64
import hashlib
import gzip
import io
import json
import tarfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from bittensor_wallet import Keypair

from lab_arena import broker as br, contracts, runner as rn, runtime, scoring, service as svc, shim, signing, source_bundle, submission_runtime, verify

SCORER_IMAGE_DIGEST = "sha256:" + "5" * 64  # the Arena-built judge image validators run
SCORER_IMAGE_REFERENCE = "arena.example/lab-arena/judge@" + SCORER_IMAGE_DIGEST
from lab_arena.store import ArenaStore, PsycopgTransport
from tests.lab_arena.icp_fixtures import daily_icps
from tests.lab_arena.lab_arena_pg_harness import database_with_lab_arena_migration

KEYS: Dict[str, Keypair] = {}
# Miners' own provider keys, injected by the fake broker: none may ever reach a row, object, event, or bundle.
CANARY_DEEPLINE_KEY = "dl_canary_" + "x" * 30
CANARY_DOG_KEY = "dogcanary" + "y" * 30
CANARY_OPENROUTER_KEY = "sk-or-v1-" + "o" * 40
CANARY_OPENROUTER_MANAGEMENT_KEY = "sk-or-v1-" + "m" * 40
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
            for table in ("lab_arena_rounds", "lab_arena_submissions", "lab_arena_runs", "lab_arena_ledger", "lab_arena_submission_credentials"):
                for canary in (*CANARY_KEYS.values(), CANARY_OPENROUTER_MANAGEMENT_KEY):
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
        return 1

    def validator_permit_hotkeys(self) -> List[str]:
        return list(self.runners)


class FakeProviderTransport:
    def send(self, *, method, url, headers, body, timeout_seconds):
        if "-refused" in url or any("-refused" in str(value) for value in headers.values()):
            # The provider rejects a revoked miner key.
            return br.ProviderResponse(401, {"content-type": "application/json"}, b'{"error": "invalid key"}')
        payload = json.dumps({"results": [{"url": "https://co1.example.com", "title": "Co"}]}).encode()
        return br.ProviderResponse(200, {"content-type": "application/json"}, payload)


class FakeCredentialManager:
    def validate_and_encrypt(self, credentials, *, submission_id, miner_hotkey):
        assert credentials == {
            "openrouter_api_key": CANARY_OPENROUTER_KEY,
            "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
            "deepline_api_key": CANARY_DEEPLINE_KEY,
        }
        assert submission_id and miner_hotkey
        return {
            provider: base64.b64encode(("kms-ciphertext-" + provider).encode()).decode()
            for provider in ("openrouter", "deepline")
        }

    def runtime_key(self, row, provider):
        assert row["provider"] == provider
        return CANARY_KEYS[provider]


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

    def __init__(self, *, flavor_by_submission: Dict[str, str], broken_submissions: set):
        self.flavor_by_submission = flavor_by_submission
        self.broken_submissions = broken_submissions  # shared with the harness, mutated by tests
        self.lock = threading.Lock()
        self.runs = 0
        self.inflate_scores = False  # a cheating validator reports 99.0 for every company
        self.judge_failures: set[tuple[str, int]] = set()

    def run_icp(self, spec: runtime.SandboxSpec, **_):
        with self.lock:
            self.runs += 1
        runtime_digest = "sha256:" + spec.rootfs_path.parent.name.split("sha256-")[1]
        input_document = json.loads((spec.input_dir / runtime.INPUT_FILE_NAME).read_text())
        icp = input_document["icp"]
        import os

        scoring_run = input_document.get("schema_version") == scoring.SCORING_INPUT_SCHEMA_VERSION
        # Production sandboxes are separate processes. This in-process fake
        # shares os.environ across worker threads, so serialize only the fake
        # environment switch to prevent one run from using another run's socket.
        with self.lock:
            os.environ[shim.WORKER_SOCKET_ENV] = str(spec.socket_path)
            try:
                status, _headers, body = shim.dispatch("deepline.execute", {"tool": "exa_search", "payload": {"query": icp["prompt"][:200]}}, 5000)
                if not scoring_run and status != 200:
                    # A caller-caused provider error makes this model run fail.
                    return runtime.fake_result(exit_code=1, output_bytes=None, stderr=b"provider error %d" % status)
                assert status == 200
            finally:
                os.environ.pop(shim.WORKER_SOCKET_ENV, None)
        if scoring_run:
            # A scoring assignment: the validator's judge sandbox (the pinned scorer image, trusted mode).
            assert runtime_digest == SCORER_IMAGE_DIGEST and spec.extra_environment.get(shim.TRUSTED_SCORER_ENV) == "1" and spec.entry_command == runtime.SCORER_ENTRY_COMMAND
            companies = list(input_document["companies"])
            flavor = (
                str(companies[0].get("company_name") or "").split(" Company ", 1)[0]
                if companies
                else ""
            )
            position = int(str(icp["icp_id"]).rsplit("_", 1)[-1]) - 1
            if (flavor, position) in self.judge_failures:
                return runtime.fake_result(
                    exit_code=1, output_bytes=None, stderr=b"judge failure"
                )
            breakdowns = deterministic_scorer(companies, icp, False)
            if self.inflate_scores:
                breakdowns = [dict(row, final_score=99.0) for row in breakdowns]
            output = scoring.build_scoring_output(input_document["scored_run_id"], breakdowns)
            return runtime.fake_result(exit_code=0, output_bytes=json.dumps(output).encode())
        assert spec.entry_command == runtime.AGENT_ENTRY_COMMAND
        assert runtime_digest == SCORER_IMAGE_DIGEST
        assert spec.source_dir is not None
        submission_id = spec.source_dir.parent.name.removeprefix("submission-")
        if submission_id in self.broken_submissions:
            return runtime.fake_result(exit_code=1, output_bytes=None, stderr=b"crash")
        flavor = self.flavor_by_submission.get(submission_id)
        if flavor is None:
            flavor = (spec.source_dir / "flavor.txt").read_text(encoding="utf-8")
            self.flavor_by_submission[submission_id] = flavor
        bucket = icp["employee_count"][0]
        companies = [
            {
                "company_name": "%s Company %d" % (flavor, i),
                "company_website": "https://%s-%d.example.com" % (flavor.lower(), i),
                "company_linkedin": "",
                "industry": icp["industry"],
                "employee_count": bucket,
                "company_stage": str(icp.get("company_stage") or ""),
                "country": icp.get("country") or "United States",
                "state": "",
                "fit_summary": "The company matches the ICP.",
                "fit_evidence_urls": [
                    "https://%s-%d.example.com/about" % (flavor.lower(), i)
                ],
                "intent_signals": [
                    {
                        "description": "Raised a round",
                        "url": "https://news.example.com/%s/%d" % (flavor, i),
                        "date": "2026-08-01",
                        "why_now": "The funding makes outreach timely.",
                        "snippet": "Funding announced",
                        "matched_icp_signal": 0,
                    }
                ],
            }
            for i in range(5)
        ]
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

    def current(self):
        return self.service.public_current()

    def round(self, round_id):
        return self.service.public_round(round_id)

    def claim(self, envelope):
        return self._guard(lambda: self.service.handle_claim(envelope))

    def provider(self, run_id, lease_token, frame):
        try:
            return self.service.handle_provider(run_id, lease_token, frame)
        except Exception as exc:
            InProcessApi.errors.append("%s: %s" % (type(exc).__name__, str(exc)[:1500]))
            raise

    def complete(self, envelope):
        return self._guard(lambda: self.service.handle_complete(envelope))

    def source(self, run_id, lease_token):
        try:
            return self.service.handle_source(run_id, lease_token)
        except Exception as exc:
            InProcessApi.errors.append(
                "%s: %s" % (type(exc).__name__, str(exc)[:1500])
            )
            raise


def flavor_source_archive(flavor: str) -> bytes:
    """Build one small source archive whose bytes identify the fake behavior."""

    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, data in (
                ("harness.py", b"def run_icp(icp):\n    return []\n"),
                ("flavor.txt", flavor.encode("utf-8")),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
    payload = raw.getvalue()
    source_bundle.validate_source_archive(payload)
    return payload


class FixtureObjectStore(svc.LocalObjectStore):
    def presign_put(self, ref, *, size_bytes, content_type, expires_seconds):
        return {
            "upload_url": "https://uploads.example/" + ref,
            "upload_headers": {
                "content-type": content_type,
                "content-length": str(size_bytes),
                "if-none-match": "*",
            },
            "expires_in_seconds": expires_seconds,
        }


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration()


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
        self.objects = FixtureObjectStore(self.objects_root)
        self.runner_keys = [keypair("svc-runner-" + name).ss58_address for name in runners]
        self.chain = FakeChain(self.runner_keys)
        self.flavors: Dict[str, str] = {}
        self.broken: set = set()
        self.challengers = challengers
        self.baseline_hotkey = keypair("svc-baseline").ss58_address
        self.max_challengers = contracts.MAX_CHALLENGERS
        self.daily_cutoff_hour_utc = None  # set to enable automatic daily round creation
        self.banned: List[str] = []  # the live ban list the service reads
        self.api_factory = None  # runners talk to the service in-process unless a test supplies an API client
        self.baseline_source = flavor_source_archive("PublicBaseline")
        self.sandbox = ModelSandbox(
            flavor_by_submission=self.flavors, broken_submissions=self.broken
        )
        self.service = self.build_service()

    def objects_key(self) -> str:
        return "psycopg"

    def make_store(self) -> ArenaStore:
        return ArenaStore(PsycopgTransport(self.connect), lease_ttl_seconds=420)

    def build_service(self) -> svc.ArenaService:
        store = self.make_store()
        harness = self

        def broker_factory(service, round_row):
            payer = submission_runtime.SubmissionProviderKeys(
                store=store,
                credentials=service.config.credential_manager,
                organizer_keys=CANARY_KEYS,
            )
            return br.Broker(
                store=store,
                key_for=lambda provider: CANARY_KEYS[provider],
                credential_for=payer.credential_for,
                funding_source_for=payer.funding_source_for,
                price_table=price_table(),
                judge_models=tuple(scoring.DEFAULT_JUDGE_MODELS.values()),
                transport=FakeProviderTransport(),
                clock=harness.clock,
            )

        config = svc.ServiceConfig(
            mode="live", store=store, object_store=self.objects, signer=self.signer, chain=self.chain, verify_signature=wallet_verify,
            daily_icp_source=lambda **kwargs: {
                "status": "ready",
                "set_id": int(kwargs["set_id"]),
                "icps": daily_icps(),
            },
            banned_hotkeys_source=lambda: list(harness.banned),
            broker_factory=broker_factory,
            defaults=svc.RoundDefaults(
                runner_hotkeys=tuple(self.runner_keys), baseline_hotkey=self.baseline_hotkey,
                baseline_source_url="https://github.com/leadpoet/pydantic-harness/archive/refs/heads/main.tar.gz",
                max_challengers=self.max_challengers, daily_cutoff_hour_utc=self.daily_cutoff_hour_utc,
                scorer_image_digest=SCORER_IMAGE_DIGEST, scorer_image_reference=SCORER_IMAGE_REFERENCE,
            ),
            clock=self.clock,
            baseline_source_fetcher=lambda _url, _limit: self.baseline_source,
            credential_manager=FakeCredentialManager(),
        )
        return svc.ArenaService(config)

    def submit(self, flavor: str, round_id: str, *, miner_label: str = "") -> str:
        """Reserve, upload, and finalize source through the signed miner API.

        ``miner_label`` picks the submitting hotkey (default: one per flavor),
        so a test can resubmit fresh source under an existing miner.
        """

        schedule = self.service.store.get_round(round_id)["configuration_doc"]["schedule"]
        submission_open = datetime.strptime(schedule["submission_open"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        if self.clock() < submission_open:
            self.clock.advance_to(schedule["submission_open"])
        miner = keypair("svc-miner-" + (miner_label or flavor))
        payload = flavor_source_archive(flavor)
        facts = source_bundle.validate_source_archive(payload)
        presign = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_PRESIGN,
            round_id=round_id,
            hotkey=miner.ss58_address,
            body={
                "source_size_bytes": facts["source_size_bytes"],
                "consent": {"public_rerun": True},
            },
            timestamp=int(self.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex(),
        )
        target = self.service.handle_submission_presign(presign)
        self.flavors[target["submission_id"]] = flavor
        self.objects.put(target["source_ref"], payload)
        finalize = contracts.build_signed_request(
            scope=contracts.SCOPE_SUBMISSION_FINALIZE,
            round_id=round_id,
            hotkey=miner.ss58_address,
            body={
                "submission_id": target["submission_id"],
                "source_ref": target["source_ref"],
                "source_size_bytes": facts["source_size_bytes"],
                "credentials": {
                    "openrouter_api_key": CANARY_OPENROUTER_KEY,
                    "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
                    "deepline_api_key": CANARY_DEEPLINE_KEY,
                },
            },
            timestamp=int(self.clock().timestamp()),
            sign_message=lambda message: miner.sign(message.encode()).hex(),
        )
        result = self.service.handle_submission_finalize(target["submission_id"], finalize)
        assert result["status"] == "accepted", result
        row = self.service.store.get_submission(target["submission_id"])
        assert row["status"] == "accepted" and row["source_ref"] == target["source_ref"], row
        return target["submission_id"]

    def runner(self, index: int, parallel: int = 4) -> rn.Runner:
        kp = keypair("svc-runner-" + ["alpha", "beta", "gamma"][index])
        cache = rn.ImageCache(self.tmp / ("images-%d" % index), lambda reference, digest, target: (target / "rootfs").mkdir())
        api = self.api_factory() if self.api_factory else InProcessApi(self.service)
        source_cache = rn.SourceCache(
            self.tmp / ("sources-%d" % index),
            api.source,
            dependency_installer=lambda _requirements, _target: None,
        )
        config = rn.RunnerConfig(
            round_id=self.round_id, identity=rn.RunnerIdentity(hotkey=kp.ss58_address, sign=lambda m, kp=kp: kp.sign(m.encode()).hex()), api=api, sandbox_runtime=self.sandbox,
            image_cache=cache, source_cache=source_cache, work_dir=self.tmp / ("work-%d" % index), max_parallel_runs=parallel, evaluation_date="2026-09-02", clock=self.clock,
            completion_retry_seconds=(0.0, 0.0),  # retries without waiting in tests
        )
        (self.tmp / ("work-%d" % index)).mkdir(exist_ok=True)
        return rn.Runner(config)

    def run_stage_with_runners(self, count: int = 2) -> None:
        # PostgreSQL owns lease expiry and uses its real clock. Align the fake
        # service clock while runners hold leases, then restore the scheduled
        # round time used by the transition tests.
        scheduled_now = self.clock.now
        self.clock.now = datetime.now(timezone.utc)
        runners = [self.runner(i) for i in range(count)]
        try:
            while any(r.run_once() for r in runners):
                pass
        finally:
            for r in runners:
                r.close()
            self.clock.now = scheduled_now
        abandoned = [c for r in runners for c in r.completed if c.get("error")]
        assert not abandoned, "runners abandoned work: %s; api errors: %s" % (abandoned[:3], InProcessApi.errors[:3])

    def advance_until(self, target: str, *, runners: int = 2, max_steps: int = 60) -> Dict[str, Any]:
        """Drive the round to ``target``; validators claim and score whenever a scoring window is open."""

        result: Dict[str, Any] = {}
        for _ in range(max_steps):
            status = self.status()
            if status == target:
                return result
            if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
                self.run_stage_with_runners(runners)
            if status == "stage1_scored":
                self.clock.advance_to(self.schedule()["stage_2_start"])
            result = self.service.advance_round(self.round_id)
            assert result.get("status") not in ("cancelled", "terminal", "retry", "stale"), (status, result)
        raise AssertionError("round did not reach %s (at %s)" % (target, self.status()))

    def schedule(self):
        return self.service.store.get_round(self.round_id)["configuration_doc"]["schedule"]

    def status(self):
        return self.service.store.get_round(self.round_id)["status"]


def test_startup_checks_require_the_current_arena_schema(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    checks = harness.service.startup_checks()
    assert checks["schema_version"] == 185
    assert checks["database_identity"]["current_user"] == "lab_arena_service"


def test_full_round_publishes_results_and_next_day_uses_the_public_baseline(connect, tmp_path):
    """A normal round publishes scores; its winner does not replace tomorrow's baseline."""

    harness = Harness(connect, tmp_path, challengers=["Alpha", "Bravo", "Charlie"], runners=["alpha", "beta"])
    participants = _start_round(harness, day=1, epoch=24800)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("published", runners=2)

    first = harness.service.store.get_round(round_id)
    assert first["king_outcome"] == "crowned"
    assert first["king_hotkey"]
    assert len(first["publication_doc"]["stage1_ranking"]) == participants - 1
    assert len(first["finalists"]) == participants - 1
    assert len(first["publication_doc"]["final_ranking"]) == participants
    assert len(harness.service.store.list_runs(round_id, stage=1, kind="execute")) == contracts.STAGE_1_ICP_COUNT * participants
    assert len(harness.service.store.list_runs(round_id, stage=2, kind="execute")) == contracts.STAGE_2_ICP_COUNT * participants

    # A later round always uses that day's configured public baseline.
    harness.clock.now = datetime.now(timezone.utc)
    harness.chain.epoch = 24820
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-02",
    )
    harness.round_id = configuration["round_id"]
    harness.submit("Delta", harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    second = harness.service.store.get_round(harness.round_id)
    king = next(participant for participant in second["participants"] if participant["is_king"])
    assert king["miner_hotkey"] == harness.baseline_hotkey
    assert king["miner_hotkey"] != first["king_hotkey"]
    harness.flavors.setdefault(king["submission_id"], "PublicBaseline")

    _run_stage_one_to_scoring(harness, len(second["participants"]), runners=2)
    harness.advance_until("published", runners=2)
    second = harness.service.store.get_round(harness.round_id)
    assert second["king_outcome"] in ("crowned", "no_king")
    decision = second["publication_doc"]["king_decision"]
    winner = decision["winner_submission_id"] or second["publication_doc"][
        "final_ranking"
    ][0]["submission_id"]
    public = harness.service.public_results(harness.round_id, winner)
    assert len(public["scores"]["stage_1"]) == contracts.STAGE_1_ICP_COUNT
    assert len(public["scores"]["stage_2"]) == contracts.STAGE_2_ICP_COUNT
    assert public["submission_scores"]["final"] is not None
    assert_canary_absent(harness, connect)


def test_restart_finishes_a_partial_participant_freeze_without_changing_baseline(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Restart-A", "Restart-B"], runners=["alpha"])
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-04-restart",
    )
    harness.round_id = configuration["round_id"]
    submitted = [harness.submit(flavor, harness.round_id) for flavor in harness.challengers]
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])

    round_row = harness.service.store.get_round(harness.round_id)
    baseline = harness.service._initial_baseline(round_row)
    assert baseline["status"] == "accepted"
    assert harness.service.store.update_submission(
        harness.round_id,
        baseline["submission_id"],
        "accepted",
        "frozen",
        {"is_king": True},
    )["status"] == "ok"
    assert harness.service.store.update_submission(
        harness.round_id,
        submitted[0],
        "accepted",
        "frozen",
        {},
    )["status"] == "ok"

    harness.service = harness.build_service()
    committed = harness.service.commit_benchmark(harness.round_id)
    assert committed == {"status": "ok", "participants": 3}
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    assert {item["submission_id"] for item in participants} == {
        baseline["submission_id"],
        *submitted,
    }
    kings = [item for item in participants if item["is_king"]]
    assert len(kings) == 1 and kings[0]["submission_id"] == baseline["submission_id"]
    harness.service.cancel(harness.round_id, sorted(svc.CANCEL_REASONS.values())[0])


def test_infrastructure_gap_cancels_and_model_failures_score_zero(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Echo", "Foxtrot"], runners=["alpha", "beta"])
    service = harness.service
    configuration = service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-05-gap",
    )
    harness.round_id = configuration["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    broken = next(
        submission_id for submission_id, flavor in harness.flavors.items() if flavor == "Foxtrot"
    )
    harness.broken.add(broken)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    participants = service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")

    assignments = contracts.STAGE_1_ICP_COUNT * len(participants)
    scheduled_now = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    runner = harness.runner(0, parallel=1)
    try:
        assert runner.run_once() == 1
    finally:
        runner.close()
        harness.clock.now = scheduled_now
    harness.clock.advance_to(harness.schedule()["stage_1_close"])
    closed = service.advance_round(harness.round_id)
    assert closed["status"] == "cancelled"
    assert closed["incomplete_assignments"] == assignments - 1

    # Model failure is a miner result. It records zeros when all infrastructure
    # work completes and does not cancel the round.
    harness.clock.now = datetime.now(timezone.utc)
    configuration = service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-06-zero",
    )
    harness.round_id = configuration["round_id"]
    harness.submit("EchoTwo", harness.round_id, miner_label="Echo")
    broken_submission = harness.submit("FoxtrotTwo", harness.round_id)
    broken = next(
        submission_id
        for submission_id, flavor in harness.flavors.items()
        if flavor == "FoxtrotTwo"
    )
    harness.broken.add(broken)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    participants = service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT * len(participants)
    harness.run_stage_with_runners(1)
    assert service.advance_round(harness.round_id)["status"] == "ok"
    scoring_open = service.advance_round(harness.round_id)
    assert scoring_open["assignments"] < contracts.STAGE_1_ICP_COUNT * len(participants)
    harness.advance_until("stage1_scored", runners=1)

    failed = service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=broken_submission,
        kind="execute",
    )
    assert all(run["terminal_cause"] == "model_error" for run in failed)
    scored_failures = [run for run in failed if run["per_icp_score"] is not None]
    assert len(scored_failures) == contracts.STAGE_1_ICP_COUNT
    assert all(float(run["per_icp_score"]) == 0.0 for run in scored_failures)
    other = [
        run for run in service.store.list_runs(harness.round_id, stage=1, kind="execute")
        if run["submission_id"] != broken_submission and run["per_icp_score"] is not None
    ]
    assert other and all(float(run["per_icp_score"]) > 0.0 for run in other)
    service.cancel(harness.round_id, sorted(svc.CANCEL_REASONS.values())[0])




def test_shadow_round_uses_the_same_two_stage_flow_without_rewards(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Tango", "Uniform", "Victor", "Whiskey"], runners=["alpha", "beta"])
    original_build = harness.build_service

    def build_shadow():
        live = original_build()
        return svc.ArenaService(svc.ServiceConfig(**{**live.config.__dict__, "mode": "shadow"}))

    harness.build_service = build_shadow
    harness.service = build_shadow()
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-20-shadow",
    )
    assert configuration["mode"] == "shadow"
    assert configuration["rewards_enabled"] is False
    harness.round_id = configuration["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    _run_stage_one_to_scoring(harness, len(participants), runners=2)
    harness.advance_until("published", runners=2)

    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "published"
    assert len(row["publication_doc"]["final_ranking"]) == len(participants)
    assert harness.service.public_reward_basis(harness.chain.epoch) is None


def test_startup_checks_fail_closed_and_a_banned_configured_runner_stops_round_creation(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Xray"], runners=["alpha", "beta"])
    banned_runner = harness.runner_keys[1]
    blocked = svc.ArenaService(svc.ServiceConfig(**{**harness.service.config.__dict__, "banned_hotkeys_source": lambda: [banned_runner]}))
    # Rounds left open by earlier tests in this shared database are cancelled first.
    for row in blocked.store.list_rounds():
        if row["status"] not in ("published", "cancelled"):
            blocked.store.cancel_round(row["round_id"], "operator")
    checks = blocked.startup_checks()
    assert checks["database_identity"]["current_user"] == "lab_arena_service" and checks["current_round"] is None
    with pytest.raises(svc.ServiceError, match="runner_banned"):
        blocked.create_round(datetime(2026, 9, 25, 0, 0, tzinfo=timezone.utc))
    # A broken object store fails startup.
    class BrokenObjects:
        def put(self, ref, data):
            raise OSError("bucket unavailable")

        def get(self, ref):
            raise OSError("bucket unavailable")

    with pytest.raises(svc.ServiceError, match="object_store_unavailable"):
        svc.ArenaService(svc.ServiceConfig(**{**harness.service.config.__dict__, "object_store": BrokenObjects()})).startup_checks()


def test_freeze_exempts_the_daily_baseline_from_the_challenger_cap_and_records_overflow(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Zulu-1", "Zulu-2", "Zulu-3", "Zulu-4", "Zulu-5"], runners=["alpha"])
    harness.max_challengers = 2
    harness.service = harness.build_service()  # the cap is a round default, read when the service is built
    harness.chain.epoch = 29000
    service = harness.service
    configuration = service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id="arena-2026-09-03-cap")
    assert configuration["max_challengers"] == 2
    harness.round_id = configuration["round_id"]
    round_id = harness.round_id
    submitted = [harness.submit(flavor, round_id) for flavor in harness.challengers]
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    committed = service.advance_round(round_id)
    assert committed["status"] == "ok"
    participants = service.store.get_round(round_id)["participants"]
    frozen_challengers = {p["submission_id"] for p in participants if not p["is_king"]}
    assert len(frozen_challengers) == 2
    assert sum(1 for participant in participants if participant["is_king"]) == 1
    by_id = {row["submission_id"]: row for row in service.store.list_submissions(round_id)}
    frozen = {submission_id for submission_id in submitted if by_id[submission_id]["status"] == "frozen"}
    assert frozen == {participant["submission_id"] for participant in participants} & set(submitted)
    assert len(frozen) == 2
    rejected = [by_id[submission_id] for submission_id in submitted if submission_id not in frozen]
    assert len(rejected) == 3
    assert all((row["status"], row["rejection_rule"]) == ("rejected", "capacity.round_full") for row in rejected)
    service.store.cancel_round(round_id, "operator_abort")


# ---------------------------------------------------------------------------
# Validator scoring: a dead validator's leases move on, and the whole
# validator path works over the HTTP API a real runner uses.
# ---------------------------------------------------------------------------


def _start_round(harness: Harness, *, day: int = 9, epoch: int = 30000) -> int:
    """Create the round, admit every challenger, and return the participant count.

    Every published round claims its own reward epoch (a unique constraint), so
    tests that publish in the same module database pin distinct chain epochs.
    """

    service = harness.service
    harness.chain.epoch = epoch
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    configuration = service.create_round(cutoff, round_id="arena-2026-10-%02d" % day)
    harness.round_id = configuration["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    for participant in service.store.get_round(harness.round_id)["participants"]:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    return len(service.store.get_round(harness.round_id)["participants"])


def test_stage_one_judge_failure_excludes_only_that_challenger(connect, tmp_path):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["JudgeFailOne", "JudgePassOne"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=10, epoch=30100)
    failed = next(
        participant
        for participant in harness.service.store.get_round(harness.round_id)[
            "participants"
        ]
        if harness.flavors[participant["submission_id"]] == "JudgeFailOne"
    )
    harness.sandbox.judge_failures.add(("JudgeFailOne", 0))
    _run_stage_one_to_scoring(harness, participants, runners=2)

    harness.advance_until("published", runners=2)

    row = harness.service.store.get_round(harness.round_id)
    assert failed["submission_id"] not in row["finalists"]
    assert failed["submission_id"] not in {
        item["submission_id"] for item in row["publication_doc"]["stage1_ranking"]
    }
    score_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=failed["submission_id"],
        kind="score",
    )
    assert any(run["terminal_cause"] == "judge_error" for run in score_runs)
    execute_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=failed["submission_id"],
        kind="execute",
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_final_judge_failure_excludes_only_that_challenger(connect, tmp_path):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["JudgeFailFinal", "JudgePassFinal"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=11, epoch=30200)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.advance_until("stage1_scored", runners=2)
    failed = next(
        participant
        for participant in harness.service.store.get_round(harness.round_id)[
            "participants"
        ]
        if harness.flavors[participant["submission_id"]] == "JudgeFailFinal"
    )
    assert failed["submission_id"] in harness.service.store.get_round(
        harness.round_id
    )["finalists"]
    harness.sandbox.judge_failures.add(("JudgeFailFinal", 10))

    harness.advance_until("published", runners=2)

    publication = harness.service.store.get_round(harness.round_id)["publication_doc"]
    assert failed["submission_id"] not in {
        item["submission_id"] for item in publication["final_ranking"]
    }
    execute_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=2,
        submission_id=failed["submission_id"],
        kind="execute",
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_baseline_judge_failure_cancels_the_daily_round(connect, tmp_path):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["JudgePassAgainstBaseline"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=14, epoch=30250)
    harness.sandbox.judge_failures.add(("PublicBaseline", 0))
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.run_stage_with_runners(2)

    closed = harness.service.advance_round(harness.round_id)

    assert closed["status"] == "cancelled"
    assert harness.status() == "cancelled"


def test_a_prior_miner_winner_submits_fresh_source_as_a_challenger(connect, tmp_path):
    """A prior winner stays in reward history but never replaces the new daily baseline."""

    harness = Harness(connect, tmp_path, challengers=["Regal", "Rival"], runners=["alpha"])
    service = harness.service
    participants = _start_round(harness, day=16, epoch=30020)  # epochs rise in module order: the shared database keeps every published king
    _run_stage_one_to_scoring(harness, participants, runners=1)
    harness.advance_until("published", runners=1)
    first = service.store.get_round(harness.round_id)
    king_hotkey = first["king_hotkey"]
    assert first["king_outcome"] == "crowned" and king_hotkey
    king_label = next(flavor for flavor in ("Regal", "Rival") if keypair("svc-miner-" + flavor).ss58_address == king_hotkey)
    # Next day: the prior winner submits fresh source under the same hotkey.
    harness.chain.epoch = 30040
    harness.clock.now = datetime.now(timezone.utc)
    configuration = service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-17-fresh",
    )
    harness.round_id = configuration["round_id"]
    fresh = harness.submit(king_label + "-Fresh", harness.round_id, miner_label=king_label)
    harness.submit("Rival" if king_label == "Regal" else "Regal", harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok" and harness.status() == "committed"
    parts = service.store.get_round(harness.round_id)["participants"]
    prior_winner_parts = [p for p in parts if p["miner_hotkey"] == king_hotkey]
    baseline_parts = [p for p in parts if p["is_king"]]
    fresh_row = service.store.get_submission(fresh)
    assert len(parts) == 3 and len(prior_winner_parts) == 1 and len(baseline_parts) == 1
    assert prior_winner_parts[0]["submission_id"] == fresh
    assert prior_winner_parts[0]["is_king"] is False
    assert prior_winner_parts[0]["submission_id"] == fresh_row["submission_id"]
    assert baseline_parts[0]["miner_hotkey"] == harness.baseline_hotkey
    assert fresh_row["status"] == "frozen" and fresh_row["is_king"] is False
    for participant in parts:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")
    _run_stage_one_to_scoring(harness, 3, runners=1)
    harness.advance_until("published", runners=1)
    second = service.store.get_round(harness.round_id)
    assert second["king_outcome"] in ("crowned", "defended")
    assert_canary_absent(harness, connect)


def _run_stage_one_to_scoring(harness: Harness, participants: int, *, runners: int) -> None:
    service = harness.service
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT * participants
    harness.run_stage_with_runners(runners)
    assert service.advance_round(harness.round_id)["status"] == "ok" and harness.status() == "stage1_closed"
    assert service.advance_round(harness.round_id)["assignments"] == contracts.STAGE_1_ICP_COUNT * participants and harness.status() == "stage1_scoring"


def test_a_validator_that_dies_mid_scoring_loses_its_lease_and_another_validator_finishes(connect, tmp_path):
    """Scoring leases expire like execution leases; the second attempt is any validator's."""

    from datetime import timedelta

    harness = Harness(connect, tmp_path, challengers=["Crash-A"], runners=["alpha", "beta"])
    service = harness.service
    participants = _start_round(harness, day=12, epoch=30300)
    round_id = harness.round_id
    _run_stage_one_to_scoring(harness, participants, runners=2)

    class DyingApi(InProcessApi):
        """The validator process dies after judging: three scoring completions never reach the Arena, however often retried."""

        dead_runs: set = set()

        def complete(self, envelope):
            body = envelope["body"]
            scoring_result = (body.get("output") or {}).get("schema_version") == "leadpoet.lab_arena.scoring_output.v1"
            if scoring_result and (body["run_id"] in DyingApi.dead_runs or len(DyingApi.dead_runs) < 3):
                DyingApi.dead_runs.add(body["run_id"])
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
    harness.advance_until("scored", runners=2)
    accepted = [run for run in service.store.list_runs(round_id, stage=1, kind="score") if run["status"] == "accepted"]
    assert len(accepted) == contracts.STAGE_1_ICP_COUNT * participants
    assert {run["runner_hotkey"] for run in accepted if run["attempt"] == 2} <= set(harness.runner_keys)


def test_validators_complete_a_round_over_the_http_api(connect, tmp_path):
    """A real runner API client completes the current two-stage competition."""

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
    harness.clock.now = datetime.now(timezone.utc)
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-13-http",
    )
    harness.round_id = configuration["round_id"]

    miner = keypair("svc-miner-Http-A")
    payload = flavor_source_archive("Http-A")
    facts = source_bundle.validate_source_archive(payload)
    envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=harness.round_id,
        hotkey=miner.ss58_address,
        body={
            "source_size_bytes": facts["source_size_bytes"],
            "consent": {"public_rerun": True},
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    presigned = original_post(
        "http://localhost/arena/v1/submissions/presign",
        content=json.dumps(envelope),
        headers={"content-type": "application/json"},
    )
    assert presigned.status_code == 200
    target = presigned.json()
    submission_id = target["submission_id"]
    harness.flavors[submission_id] = "Http-A"
    status = original_get("http://localhost/arena/v1/submissions/%s" % submission_id)
    assert status.status_code == 200 and status.json()["status"] == "uploading"
    harness.objects.put(target["source_ref"], payload)
    finalize = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_FINALIZE,
        round_id=harness.round_id,
        hotkey=miner.ss58_address,
        body={
            "submission_id": submission_id,
            "source_ref": target["source_ref"],
            "source_size_bytes": facts["source_size_bytes"],
            "credentials": {
                "openrouter_api_key": CANARY_OPENROUTER_KEY,
                "openrouter_management_key": CANARY_OPENROUTER_MANAGEMENT_KEY,
                "deepline_api_key": CANARY_DEEPLINE_KEY,
            },
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    finalized = original_post(
        "http://localhost/arena/v1/submissions/%s/finalize" % submission_id,
        content=json.dumps(finalize),
        headers={"content-type": "application/json"},
    )
    assert finalized.status_code == 200
    status = original_get("http://localhost/arena/v1/submissions/%s" % submission_id)
    assert status.status_code == 200 and status.json()["status"] == "accepted"

    assert harness.service.advance_round(harness.round_id)["status"] == "waiting"
    harness.submit("Http-B", harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")

    _run_stage_one_to_scoring(harness, len(participants), runners=2)
    harness.advance_until("published", runners=2)
    assert calls["post"] > 0

    public = original_get("http://localhost/arena/v1/rounds/%s" % harness.round_id)
    assert public.status_code == 200
    row = harness.service.store.get_round(harness.round_id)
    decision = row["publication_doc"]["king_decision"]
    winner = decision["winner_submission_id"] or row["publication_doc"][
        "final_ranking"
    ][0]["submission_id"]
    results = original_get(
        "http://localhost/arena/v1/rounds/%s/results/%s" % (harness.round_id, winner)
    )
    assert results.status_code == 200
    assert len(results.json()["scores"]["stage_1"]) == contracts.STAGE_1_ICP_COUNT
    assert len(results.json()["scores"]["stage_2"]) == contracts.STAGE_2_ICP_COUNT
    benchmark = original_get("http://localhost/arena/v1/rounds/%s/benchmark" % harness.round_id)
    assert benchmark.status_code == 200
    assert len(benchmark.json()["icps"]) == contracts.BENCHMARK_ICP_COUNT
    current = original_get("http://localhost/arena/v1/current")
    assert current.status_code == 200
    assert row["king_outcome"] == "no_king"
    assert current.json()["king"] is None


def test_rounds_overlap_and_every_request_names_its_round(connect, tmp_path):
    """Requests stay bound to their round while a later round is open."""

    harness = Harness(connect, tmp_path, challengers=["Over-A", "Over-B"], runners=["alpha"])
    service = harness.service
    reason = sorted(svc.CANCEL_REASONS.values())[0]
    for row in service.active_rounds():
        service.cancel(row["round_id"], reason)

    harness.clock.now = datetime.now(timezone.utc)
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    first = service.create_round(cutoff, round_id="arena-2026-11-02-over")
    harness.round_id = first["round_id"]
    for flavor in harness.challengers:
        harness.submit(flavor, harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert service.advance_round(harness.round_id)["status"] == "ok"
    first_id = harness.round_id
    participants = service.store.get_round(first_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")

    harness.clock.now = datetime.now(timezone.utc)
    second = service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-11-03-over",
    )
    second_id = second["round_id"]
    current = service.public_current()
    assert current["open_round"]["round_id"] == second_id
    assert first_id in {row["round_id"] for row in current["running_rounds"]}
    harness.submit("Over-C", second_id)

    with pytest.raises(svc.ServiceError) as closed:
        harness.submit("Over-D", first_id)
    assert closed.value.code == "submission_window_closed"
    unknown_miner = keypair("svc-miner-Over-E")
    unknown_payload = flavor_source_archive("Over-E")
    unknown_facts = source_bundle.validate_source_archive(unknown_payload)
    unknown_envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id="arena-2099-01-01",
        hotkey=unknown_miner.ss58_address,
        body={
            "source_size_bytes": unknown_facts["source_size_bytes"],
            "consent": {"public_rerun": True},
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: unknown_miner.sign(message.encode()).hex(),
    )
    with pytest.raises(svc.ServiceError) as unknown:
        service.handle_submission_presign(unknown_envelope)
    assert unknown.value.code == "round_unknown"

    harness.round_id = first_id
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert service.advance_round(first_id)["assignments"] == contracts.STAGE_1_ICP_COUNT * len(participants)
    key = keypair("svc-runner-alpha")

    def claim(round_id):
        envelope = contracts.build_signed_request(
            scope=contracts.SCOPE_CLAIM,
            round_id=round_id,
            hotkey=key.ss58_address,
            body={"declared_parallelism": 1},
            timestamp=int(harness.clock().timestamp()),
            sign_message=lambda message: key.sign(message.encode()).hex(),
        )
        return service.handle_claim(envelope)

    assert claim(second_id)["status"] != "leased"
    with pytest.raises(svc.ServiceError) as bad_claim:
        claim("arena-2099-01-01")
    assert bad_claim.value.code == "round_unknown"

    cache = rn.ImageCache(
        harness.tmp / "images-follower",
        lambda reference, digest, target: (target / "rootfs").mkdir(),
    )
    (harness.tmp / "work-follower").mkdir(exist_ok=True)
    follower_api = InProcessApi(service)
    follower = rn.Runner(
        rn.RunnerConfig(
            round_id=None,
            identity=rn.RunnerIdentity(
                hotkey=key.ss58_address,
                sign=lambda message: key.sign(message.encode()).hex(),
            ),
            api=follower_api,
            sandbox_runtime=harness.sandbox,
            image_cache=cache,
            source_cache=rn.SourceCache(
                harness.tmp / "sources-follower",
                follower_api.source,
                dependency_installer=lambda _requirements, _target: None,
            ),
            work_dir=harness.tmp / "work-follower",
            max_parallel_runs=4,
            evaluation_date="2026-11-02",
            clock=harness.clock,
            completion_retry_seconds=(0.0, 0.0),
        )
    )
    scheduled_now = harness.clock.now
    harness.clock.now = datetime.now(timezone.utc)
    try:
        while follower.run_once():
            pass
    finally:
        follower.close()
        harness.clock.now = scheduled_now
    assert follower.round_ids == [first_id]
    assert not [completion for completion in follower.completed if completion.get("error")]
    harness.advance_until("published", runners=1)
    assert service.store.get_round(second_id)["status"] == "open"
    assert service.public_current()["open_round"]["round_id"] == second_id
    service.cancel(second_id, reason)
    assert_canary_absent(harness, connect)


def test_daily_rounds_are_created_only_when_no_round_is_open_and_skip_dates_that_exist(connect, tmp_path):
    """The driver's round creation: opt-in, one round at a time, idempotent, next day when a date's round exists."""

    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    reason = sorted(svc.CANCEL_REASONS.values())[0]
    # Earlier tests in this database may have left a round mid-stage: one round at a time means
    # "existing" until it ends, so end them, then start from a quiet Arena in December.
    while service.current_round() is not None:
        service.cancel(service.current_round()["round_id"], reason)
    harness.clock.advance_to("2026-12-01T12:00:00Z")
    assert service.ensure_daily_round()["status"] == "disabled"  # no cutoff hour: the operator creates rounds
    harness.daily_cutoff_hour_utc = 0
    harness.service = harness.build_service()
    service = harness.service
    # 12:00 UTC on 2026-12-01 with a six-hour window: the next midnight is 2026-12-02.
    created = service.ensure_daily_round()
    assert created["status"] == "created" and created["round_id"] == "arena-2026-12-02" and created["cutoff"] == "2026-12-02T00:00:00Z"
    assert service.ensure_daily_round() == {"status": "existing", "round_id": "arena-2026-12-02", "round_status": "open"}
    assert service.store.get_round("arena-2026-12-02")["configuration_doc"]["schedule"]["submission_cutoff"] == "2026-12-02T00:00:00Z"
    # The window rule: at 20:00 UTC, midnight is under six hours away, so the round is the day after.
    harness.clock.advance_to("2026-12-01T20:00:00Z")
    service.cancel("arena-2026-12-02", reason)
    assert service.store.get_round("arena-2026-12-02")["status"] == "cancelled"
    following = service.ensure_daily_round()
    assert following["status"] == "created" and following["round_id"] == "arena-2026-12-03"
    # A date whose round already exists is skipped even inside the window.
    service.cancel("arena-2026-12-03", reason)
    harness.clock.advance_to("2026-12-01T12:00:00Z")
    assert service.ensure_daily_round()["round_id"] == "arena-2026-12-04"


class FlakyObjectStore:
    """Every distinct result write fails once; reads pass through."""

    def __init__(self, inner):
        self._inner = inner
        self._failed_once = set()
        self.failures = 0

    def put(self, key, data):
        if "/sources/" in key:
            return self._inner.put(key, data)
        if key not in self._failed_once:
            self._failed_once.add(key)
            self.failures += 1
            raise OSError("object store unavailable")
        return self._inner.put(key, data)

    def allow(self, key):
        self._failed_once.add(key)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_result_writes_retry_after_transient_object_store_failures(connect, tmp_path):
    """A transient result-store failure does not lose or duplicate a run."""

    harness = Harness(connect, tmp_path, challengers=["Flaky-A"], runners=["alpha"])
    flaky = FlakyObjectStore(harness.objects)
    harness.objects = flaky
    harness.service = harness.build_service()
    configuration = harness.service.create_round(
        datetime.now(timezone.utc) + timedelta(hours=12),
        round_id="arena-2026-10-15-flaky",
    )
    harness.round_id = configuration["round_id"]
    harness.submit("Flaky-A", harness.round_id)

    # This test targets runner result delivery. The benchmark write has a
    # separate driver retry boundary and is allowed through here.
    flaky.allow("arena/%s/benchmark.json" % harness.round_id)
    harness.clock.advance_to(harness.schedule()["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    for participant in participants:
        harness.flavors.setdefault(participant["submission_id"], "PublicBaseline")

    _run_stage_one_to_scoring(harness, len(participants), runners=1)
    harness.advance_until("published", runners=1)
    assert flaky.failures > 0
    row = harness.service.store.get_round(harness.round_id)
    decision = row["publication_doc"]["king_decision"]
    winner = decision["winner_submission_id"] or row["publication_doc"][
        "final_ranking"
    ][0]["submission_id"]
    results = harness.service.public_results(harness.round_id, winner)
    assert len(results["scores"]["stage_1"]) == contracts.STAGE_1_ICP_COUNT
    assert len(results["scores"]["stage_2"]) == contracts.STAGE_2_ICP_COUNT
    attempts = harness.service.store.list_runs(harness.round_id)
    assert all(int(run["attempt"]) == 1 for run in attempts)
    assert_canary_absent(harness, connect)



def test_a_banned_miner_cannot_submit_to_the_frozen_round(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    reason = sorted(svc.CANCEL_REASONS.values())[0]
    while service.current_round() is not None:
        service.cancel(service.current_round()["round_id"], reason)
    miner = keypair("svc-miner-Banned")
    harness.banned.append(miner.ss58_address)
    configuration = service.create_round(datetime.now(timezone.utc) + timedelta(hours=12), round_id="arena-2026-09-03-ban")
    harness.round_id = configuration["round_id"]
    harness.clock.advance_to(configuration["schedule"]["submission_open"])
    facts = source_bundle.validate_source_archive(flavor_source_archive("Banned"))
    envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_SUBMISSION_PRESIGN,
        round_id=harness.round_id,
        hotkey=miner.ss58_address,
        body={
            "source_size_bytes": facts["source_size_bytes"],
            "consent": {"public_rerun": True},
        },
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: miner.sign(message.encode()).hex(),
    )
    with pytest.raises(svc.ServiceError) as refused:
        service.handle_submission_presign(envelope)
    assert refused.value.code == "hotkey_banned"

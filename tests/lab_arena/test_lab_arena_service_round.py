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
import subprocess
import tarfile
import threading
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest
from bittensor_wallet import Keypair

from lab_arena import broker as br, contracts, driver as arena_driver, runner as rn, runtime, scoring, service as svc, shim, signing, source_bundle, submission_runtime, verify

SCORER_IMAGE_DIGEST = "sha256:" + "5" * 64  # the Arena-built judge image validators run
SCORER_IMAGE_REFERENCE = "arena.example/lab-arena/judge@" + SCORER_IMAGE_DIGEST
from lab_arena.store import (
    ArenaStore,
    PsycopgTransport,
    hash_lease_token,
    new_lease_token,
)
from lab_arena.promotion import GitPromoter
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
                if scoring_run and status != 200:
                    failure = scoring.build_scoring_failure(
                        input_document["scored_run_id"],
                        "judge_error",
                        detail="provider error %d" % status,
                    )
                    return runtime.fake_result(
                        exit_code=0, output_bytes=json.dumps(failure).encode()
                    )
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


def promotion_repository(root: Path) -> Path:
    """Create local main/lab heads for the real baseline promoter."""

    remote = root / "remote.git"
    seed = root / "seed"

    def git(cwd: Path, *arguments: str) -> str:
        return subprocess.run(
            ("git", *arguments), cwd=cwd, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout.decode().strip()

    git(root, "init", "--bare", str(remote))
    git(root, "init", str(seed))
    git(seed, "config", "user.name", "Test")
    git(seed, "config", "user.email", "test@example.test")
    (seed / "old.txt").write_text("main", encoding="utf-8")
    git(seed, "add", ".")
    git(seed, "commit", "-m", "main")
    main = git(seed, "rev-parse", "HEAD")
    git(seed, "switch", "-c", "lab")
    (seed / "old.txt").write_text("lab", encoding="utf-8")
    git(seed, "commit", "-am", "lab")
    lab = git(seed, "rev-parse", "HEAD")
    git(seed, "remote", "add", "origin", str(remote))
    git(seed, "push", "origin", "%s:refs/heads/main" % main, "%s:refs/heads/lab" % lab)
    return remote


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
                baseline_source_url=svc.DEFAULT_BASELINE_SOURCE_URL,
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
    assert checks["schema_version"] == 197
    assert checks["database_identity"]["current_user"] == "lab_arena_service"


def test_benchmark_commit_refreshes_a_delayed_open_round_scorer_before_jobs(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=["Refresh"], runners=["alpha"])
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    configuration = harness.service.create_round(
        cutoff, round_id="arena-2026-09-25-refresh"
    )
    submission_id = harness.submit("Refresh", "arena-2026-09-25-refresh")
    accepted = harness.service.store.get_submission(submission_id)
    accepted_source = harness.objects.get(accepted["source_ref"])
    assert harness.service.store.list_runs("arena-2026-09-25-refresh") == []
    new_digest = "sha256:" + "6" * 64
    new_reference = "arena.example/lab-arena/judge@" + new_digest
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        scorer_image_digest=new_digest,
        scorer_image_reference=new_reference,
    )
    harness.round_id = "arena-2026-09-25-refresh"
    harness.clock.advance_to(configuration["schedule"]["submission_cutoff"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    committed = harness.service.store.get_round(harness.round_id)
    assert committed["status"] == "committed"
    assert committed["configuration_doc"] == {
        **configuration,
        "scorer_image_digest": new_digest,
        "scorer_image_reference": new_reference,
    }
    frozen = harness.service.store.get_submission(submission_id)
    assert frozen["status"] == "frozen"
    assert frozen["source_ref"] == accepted["source_ref"]
    assert frozen["submission_doc"] == accepted["submission_doc"]
    assert harness.objects.get(frozen["source_ref"]) == accepted_source
    assert harness.service.store.list_runs(harness.round_id) == []


def test_benchmark_commit_rejects_an_invalid_current_scorer_before_freeze(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    configuration = harness.service.create_round(
        cutoff, round_id="arena-2026-09-26-badscorer"
    )
    harness.service.config.defaults = replace(
        harness.service.config.defaults,
        scorer_image_digest="sha256:" + "6" * 64,
        scorer_image_reference="arena.example/lab-arena/judge@sha256:" + "7" * 64,
    )
    harness.round_id = "arena-2026-09-26-badscorer"
    harness.clock.advance_to(configuration["schedule"]["submission_cutoff"])
    with pytest.raises(svc.ServiceError, match="scorer_image_invalid"):
        harness.service.commit_benchmark(harness.round_id)
    row = harness.service.store.get_round(harness.round_id)
    assert row["status"] == "open"
    assert row["configuration_doc"] == configuration
    assert harness.service.store.list_submissions(harness.round_id) == []
    assert harness.service.store.list_runs(harness.round_id) == []


def test_full_round_publishes_results_and_next_day_uses_the_public_baseline(connect, tmp_path):
    """A crowned source is promoted before it becomes tomorrow's baseline."""

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

    decision = first["publication_doc"]["king_decision"]
    winner_submission_id = decision["winner_submission_id"]
    winner_flavor = harness.flavors[winner_submission_id]
    repository_root = tmp_path / "promotion-repository"
    repository_root.mkdir()
    remote = promotion_repository(repository_root)
    harness.service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "promotion-objects"
    )

    def promoted_baseline(_url, _limit):
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    harness.service._config.baseline_source_fetcher = promoted_baseline
    published_at = datetime.fromisoformat(
        str(first["published_at"]).replace("Z", "+00:00")
    )
    harness.clock.now = published_at - timedelta(microseconds=1)
    assert harness.service.promote_pending_baselines() == {"status": "source_private", "promoted": 0}
    assert harness.service.store.get_round(round_id)["baseline_promoted_at"] is None
    harness.clock.now = published_at
    assert harness.service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
    promoted_source = promoted_baseline("", source_bundle.MAX_SOURCE_ARCHIVE_BYTES)
    with tarfile.open(fileobj=io.BytesIO(promoted_source), mode="r:gz") as archive:
        assert archive.extractfile("flavor.txt").read().decode("utf-8") == winner_flavor

    # A later round freezes the newly promoted source under the organizer's
    # baseline identity. The winning miner remains the reward payee identity.
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
    baseline_submission = harness.service.store.get_submission(king["submission_id"])
    frozen_source = harness.objects.get(baseline_submission["source_ref"])
    with tarfile.open(fileobj=io.BytesIO(frozen_source), mode="r:gz") as archive:
        assert archive.extractfile("flavor.txt").read().decode("utf-8") == winner_flavor

    _run_stage_one_to_scoring(harness, len(second["participants"]), runners=2)
    harness.advance_until("published", runners=2)
    second = harness.service.store.get_round(harness.round_id)
    assert second["king_outcome"] in ("crowned", "no_king")
    decision = second["publication_doc"]["king_decision"]
    winner = decision["winner_submission_id"] or second["publication_doc"][
        "final_ranking"
    ][0]["submission_id"]
    public = harness.service.public_results(harness.round_id, winner)
    assert len(public["scores"]["stage_1"]) + len(public["scores"]["stage_2"]) == 20
    disclosed = harness.service.public_benchmark(harness.round_id)
    assert {item["icp_position"] for item in disclosed["icps"]} == {
        item["icp_position"] for item in public["scores"]["stage_1"] + public["scores"]["stage_2"]
    }
    assert public["submission_scores"]["final"] is not None
    assert_canary_absent(harness, connect)


def test_driver_discovers_an_older_live_round_through_unrelated_history_and_publishes(
    connect, tmp_path
):
    """The production history shape cannot hide a full baseline-only round."""

    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    service = harness.service
    for active in service.active_rounds():
        service.cancel(active["round_id"], svc.CANCEL_REASONS["operator"])

    harness.chain.epoch = 24950
    cutoff = datetime.now(timezone.utc) + timedelta(hours=12)
    configuration = service.create_round(
        cutoff, round_id="arena-2026-12-29-discovery"
    )
    harness.round_id = configuration["round_id"]
    schedule = dict(configuration["schedule"])

    history_ids = []
    for index in range(25):
        round_id = "arena-2026-12-30-h%02d" % index
        history_configuration = dict(configuration)
        history_configuration["round_id"] = round_id
        history_configuration["mode"] = "live" if index < 21 else "shadow"
        history_configuration["rewards_enabled"] = False
        assert service.store.create_round(round_id, history_configuration)["status"] == "created"
        assert service.store.cancel_round(
            round_id, svc.CANCEL_REASONS["operator"]
        )["status"] == "cancelled"
        history_ids.append(round_id)

    hidden = service.store.get_round(harness.round_id)
    assert hidden["status"] == "open"
    assert hidden["configuration_doc"]["schedule"] == schedule
    assert service.current_round()["round_id"] == harness.round_id
    assert service.open_round()["round_id"] == harness.round_id
    assert [row["round_id"] for row in service.active_rounds()] == [harness.round_id]

    harness.clock.advance_to(schedule["submission_cutoff"])
    outcome = arena_driver.drive_once(service)
    assert outcome == "advanced %s" % harness.round_id
    committed = service.store.get_round(harness.round_id)
    assert committed["status"] == "committed"
    assert committed["configuration_doc"]["schedule"] == schedule
    participants = committed["participants"]
    assert len(participants) == 1 and participants[0]["is_king"] is True
    harness.flavors[participants[0]["submission_id"]] = "PublicBaseline"

    _run_stage_one_to_scoring(harness, participants=1, runners=1)
    harness.advance_until("published", runners=1)

    published = service.store.get_round(harness.round_id)
    execute_runs = service.store.list_runs(harness.round_id, kind="execute")
    score_runs = service.store.list_runs(harness.round_id, kind="score")
    assert published["status"] == "published"
    assert published["configuration_doc"]["schedule"] == schedule
    assert len(execute_runs) == contracts.BENCHMARK_ICP_COUNT
    assert len(score_runs) == contracts.BENCHMARK_ICP_COUNT
    assert all(run["status"] == "accepted" for run in execute_runs + score_runs)
    public = service.public_results(harness.round_id, participants[0]["submission_id"])
    assert len(public["scores"]["stage_1"]) + len(public["scores"]["stage_2"]) == 20
    assert {
        item["icp_position"]
        for item in public["scores"]["stage_1"] + public["scores"]["stage_2"]
    } == set(service._public_icp_disclosure(published)["public_positions"])
    assert all(service.store.get_round(round_id)["status"] == "cancelled" for round_id in history_ids)
    assert_canary_absent(harness, connect)


def test_all_failed_baseline_keeps_zero_rows_and_cancels_before_publication(
    connect, tmp_path
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    prior_published = harness.service.latest_published_round()
    _start_round(harness, day=7, epoch=24840)
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    baseline = next(
        participant for participant in participants if participant["is_king"]
    )
    harness.broken.add(baseline["submission_id"])

    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    result = {}
    for _ in range(30):
        status = harness.status()
        if status == "cancelled":
            break
        if status in ("stage1", "stage1_scoring", "stage2", "stage2_scoring"):
            harness.run_stage_with_runners(1)
        if status == "stage1_scored":
            harness.clock.advance_to(harness.schedule()["stage_2_start"])
        result = harness.service.advance_round(harness.round_id)
    assert harness.status() == "cancelled", result

    runs = harness.service.store.list_runs(
        harness.round_id,
        submission_id=baseline["submission_id"],
        kind="execute",
    )
    scored = [run for run in runs if run["per_icp_score"] is not None]
    assert len(scored) == contracts.BENCHMARK_ICP_COUNT
    assert all(float(run["per_icp_score"]) == 0.0 for run in scored)
    row = harness.service.store.get_round(harness.round_id)
    assert row["publication_doc"] is None
    assert harness.service.latest_published_round() == prior_published


def test_partially_successful_baseline_publishes_a_numeric_mean(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    _start_round(harness, day=8, epoch=24860)
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    baseline = next(
        participant for participant in participants if participant["is_king"]
    )
    original_run_icp = harness.sandbox.run_icp

    def run_icp(spec, **kwargs):
        if spec.source_dir is None:
            return original_run_icp(spec, **kwargs)
        document = json.loads(
            (spec.input_dir / runtime.INPUT_FILE_NAME).read_text(encoding="utf-8")
        )
        submission_id = spec.source_dir.parent.name.removeprefix("submission-")
        position = int(str(document["icp"]["icp_id"]).rsplit("_", 1)[-1]) - 1
        if submission_id == baseline["submission_id"] and position != 0:
            return runtime.fake_result(
                exit_code=1, output_bytes=None, stderr=b"model failure"
            )
        return original_run_icp(spec, **kwargs)

    harness.sandbox.run_icp = run_icp
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    harness.advance_until("published", runners=1)

    public = harness.service.public_results(
        harness.round_id, baseline["submission_id"]
    )
    assert isinstance(public["submission_scores"]["final"], float)
    scores = public["scores"]["stage_1"] + public["scores"]["stage_2"]
    assert len(scores) == 20
    assert sum(float(item["per_icp_score"]) == 0.0 for item in scores) == 19
    # Final ranking and the post-evaluation public view both use all 20 scores.
    all_runs = [item for item in harness.service.store.list_runs(harness.round_id, submission_id=baseline["submission_id"], kind="execute") if item.get("per_icp_score") is not None]
    expected = verify.stage_score(
        [float(item["per_icp_score"]) for item in all_runs], len(all_runs)
    )
    assert public["submission_scores"]["final"] == expected


def test_publish_cancels_an_existing_scored_state_with_no_valid_baseline(
    connect, tmp_path, monkeypatch
):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    _start_round(harness, day=9, epoch=24880)
    participants = harness.service.store.get_round(harness.round_id)["participants"]
    baseline = next(
        participant for participant in participants if participant["is_king"]
    )
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["status"] == "ok"
    harness.advance_until("scored", runners=1)

    original_entries = harness.service._score_entries_from_runs

    def entries_with_invalid_baseline(round_row, positions, score_key):
        entries = original_entries(round_row, positions, score_key)
        if score_key == "final_score":
            entries = [
                {
                    **entry,
                    "final_score": (
                        None
                        if entry["submission_id"] == baseline["submission_id"]
                        else entry["final_score"]
                    ),
                }
                for entry in entries
            ]
        return entries

    monkeypatch.setattr(
        harness.service, "_score_entries_from_runs", entries_with_invalid_baseline
    )
    assert harness.service.publish(harness.round_id)["status"] == "cancelled"
    row = harness.service.store.get_round(harness.round_id)
    assert row["publication_doc"] is None


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


def test_fresh_completion_signature_keeps_the_same_result_idempotent(connect, tmp_path):
    harness = Harness(connect, tmp_path, challengers=[], runners=["alpha"])
    participants = _start_round(harness, day=24, epoch=31400)
    harness.clock.advance_to(harness.schedule()["stage_1_start"])
    assert harness.service.advance_round(harness.round_id)["assignments"] == (
        contracts.STAGE_1_ICP_COUNT * participants
    )
    runner_key = keypair("svc-runner-alpha")
    claim_envelope = contracts.build_signed_request(
        scope=contracts.SCOPE_CLAIM,
        round_id=harness.round_id,
        hotkey=runner_key.ss58_address,
        body={"declared_parallelism": 1},
        timestamp=int(harness.clock().timestamp()),
        sign_message=lambda message: runner_key.sign(message.encode()).hex(),
    )
    lease = harness.service.handle_claim(claim_envelope)
    assert lease["status"] == "leased"
    icp = lease["icp"]
    output = rn.output_document_from_bytes(
        json.dumps(
            {
                "companies": [
                    {
                        "company_name": "Late Settlement Company",
                        "company_website": "https://late-settlement.example.com",
                        "company_linkedin": "",
                        "industry": icp["industry"],
                        "employee_count": icp["employee_count"][0],
                        "company_stage": str(icp.get("company_stage") or ""),
                        "country": icp.get("country") or "United States",
                        "state": "",
                        "fit_summary": "The company matches the ICP.",
                        "fit_evidence_urls": [
                            "https://late-settlement.example.com/about"
                        ],
                        "intent_signals": [
                            {
                                "description": "Raised a round",
                                "url": "https://late-settlement.example.com/news",
                                "date": "2026-08-01",
                                "why_now": "The funding makes outreach timely.",
                                "snippet": "Funding announced",
                                "matched_icp_signal": 0,
                            }
                        ],
                    }
                ]
            }
        ).encode()
    )
    body = {
        "run_id": lease["run_id"],
        "lease_token": lease["lease_token"],
        "result": {
            "schema_version": contracts.RUN_RESULT_SCHEMA_VERSION,
            "resource_summary": {
                "wall_seconds": 1.0,
                "cpu_seconds": 1.0,
                "max_rss_bytes": 1024,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "provider_call_count": 0,
            },
            "started_at": "2026-10-24T01:00:00Z",
            "finished_at": "2026-10-24T01:00:01Z",
            "terminal_status": "accepted",
        },
        "output": output,
    }
    request_id = "b" * 32

    def complete():
        return harness.service.handle_complete(
            contracts.build_signed_request(
                scope=contracts.SCOPE_COMPLETE,
                round_id=harness.round_id,
                hotkey=runner_key.ss58_address,
                body=body,
                timestamp=int(harness.clock().timestamp()),
                request_id=request_id,
                sign_message=lambda message: runner_key.sign(message.encode()).hex(),
            )
        )

    first = complete()
    assert first["status"] == "accepted"
    stored_before = harness.service.store.get_run(lease["run_id"])
    object_before = harness.objects.get(stored_before["output_ref"])

    harness.clock.now += timedelta(seconds=280)
    replay = complete()

    assert replay["status"] == "accepted" and replay["idempotent"] is True
    stored_after = harness.service.store.get_run(lease["run_id"])
    assert stored_after["output_ref"] == stored_before["output_ref"]
    assert stored_after["result_doc"] == stored_before["result_doc"]
    assert harness.objects.get(stored_after["output_ref"]) == object_before
    assert len(
        list((harness.objects_root / "arena" / harness.round_id / "outputs").glob("*.json"))
    ) == 1


def test_persistent_stage_one_judge_failure_cancels_without_partial_scores(connect, tmp_path):
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
    harness.run_stage_with_runners(2)

    cancelled = harness.service.advance_round(harness.round_id)
    terminal = harness.service.advance_round(harness.round_id)

    row = harness.service.store.get_round(harness.round_id)
    assert cancelled["status"] == "cancelled" and row["status"] == "cancelled"
    assert terminal == {"status": "terminal", "round_status": "cancelled"}
    assert row["cancel_reason"] == svc.CANCEL_REASONS["scoring_incomplete"]
    assert row["publication_doc"] is None and not row["finalists"]
    score_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=failed["submission_id"],
        kind="score",
    )
    failed_score_runs = [run for run in score_runs if run["status"] == "failed"]
    assert len(failed_score_runs) == 2
    assert {run["terminal_cause"] for run in failed_score_runs} == {"judge_error"}
    assert {int(run["attempt"]) for run in failed_score_runs} == {1, 2}
    execute_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        kind="execute",
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_exhausted_judge_failure_stops_pending_scoring_and_retains_completed_evidence(
    connect, tmp_path
):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["EarlyCancelFail", "EarlyCancelVisible"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=23, epoch=30423)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    assert len(harness.service.public_benchmark(harness.round_id)["icps"]) == 20
    expected_icps = harness.service.benchmark_icps(harness.round_id)
    store = harness.service.store
    round_participants = store.get_round(harness.round_id)["participants"]
    failing = next(
        participant
        for participant in round_participants
        if harness.flavors[participant["submission_id"]] == "EarlyCancelFail"
    )
    visible = next(
        participant
        for participant in round_participants
        if harness.flavors[participant["submission_id"]] == "EarlyCancelVisible"
    )

    def claim_for(participant, runner_index):
        token = new_lease_token()
        request_id = contracts.new_request_id()
        result = store.claim_assignment(
            round_id=harness.round_id,
            runner_hotkey=harness.runner_keys[runner_index],
            declared_parallelism=1,
            slot_ceiling=1,
            excluded_miner_hotkeys=[
                row["miner_hotkey"]
                for row in round_participants
                if row["submission_id"] != participant["submission_id"]
            ],
            request_id=request_id,
            request_hash=contracts.document_hash({"request_id": request_id}),
            lease_token_hash=hash_lease_token(token),
        )
        assert result["status"] == "leased", result
        return result, token

    accepted, accepted_token = claim_for(visible, 0)
    accepted_ref = "arena/%s/outputs/%s.json" % (
        harness.round_id,
        accepted["run_id"],
    )
    scored_execution = store.get_run(accepted["scored_run_id"])
    companies = json.loads(
        harness.objects.get(scored_execution["output_ref"]).decode("utf-8")
    )["companies"]
    assert companies
    icp = harness.service.benchmark_icps(harness.round_id)[
        int(accepted["icp_position"])
    ]
    harness.objects.put(
        accepted_ref,
        json.dumps(
            scoring.build_scoring_output(
                accepted["scored_run_id"],
                deterministic_scorer(companies, icp, False),
            )
        ).encode("utf-8"),
    )
    assert store.complete_attempt(
        run_id=accepted["run_id"],
        lease_token_hash=hash_lease_token(accepted_token),
        result={"terminal_status": "accepted"},
        terminal_cause="accepted",
        output_ref=accepted_ref,
    )["status"] == "accepted"

    first, first_token = claim_for(failing, 0)
    first_failure = store.complete_attempt(
        run_id=first["run_id"],
        lease_token_hash=hash_lease_token(first_token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
    )
    assert first_failure["confirmation_attempt"] == 2
    assert harness.service.advance_round(harness.round_id) == {
        "status": "waiting",
        "round_status": "stage1_scoring",
    }

    second, second_token = claim_for(failing, 1)
    assert second["assignment_id"] == first["assignment_id"]
    assert int(second["attempt"]) == contracts.MAX_ATTEMPTS_PER_ASSIGNMENT
    assert store.complete_attempt(
        run_id=second["run_id"],
        lease_token_hash=hash_lease_token(second_token),
        result={"terminal_status": "judge_error"},
        terminal_cause="judge_error",
        output_ref="",
    )["status"] == "failed"
    assert store.list_runs(
        harness.round_id, stage=1, status="pending", kind="score"
    )

    cancelled = harness.service.advance_round(harness.round_id)

    assert cancelled["status"] == "cancelled"
    row = store.get_round(harness.round_id)
    assert row["status"] == "cancelled"
    assert row["cancel_reason"] == svc.CANCEL_REASONS["scoring_incomplete"]
    assert not store.list_runs(harness.round_id, stage=1, status="pending")
    assert not store.list_runs(harness.round_id, stage=1, status="leased")
    post_cancel_token = new_lease_token()
    post_cancel_request_id = contracts.new_request_id()
    post_cancel = store.claim_assignment(
        round_id=harness.round_id,
        runner_hotkey=harness.runner_keys[0],
        declared_parallelism=1,
        slot_ceiling=1,
        excluded_miner_hotkeys=[],
        request_id=post_cancel_request_id,
        request_hash=contracts.document_hash(
            {"request_id": post_cancel_request_id}
        ),
        lease_token_hash=hash_lease_token(post_cancel_token),
    )
    assert post_cancel == {"status": "stage_closed", "round_status": "cancelled"}

    accepted_after = store.get_run(accepted["run_id"])
    assert accepted_after["status"] == "accepted"
    assert accepted_after["output_ref"] == accepted_ref
    with pytest.raises(svc.ServiceError, match="results_not_public"):
        harness.service.public_results(harness.round_id, accepted["submission_id"])
    # Cancellation preserves evidence internally. The Day 0 bank is public
    # after cutoff, but no model output or score is public without evaluation.
    assert len(harness.service.public_benchmark(harness.round_id)["icps"]) == 20
    assert harness.service.benchmark_icps(harness.round_id) == expected_icps
    execution_runs = store.list_runs(
        harness.round_id, stage=1, kind="execute"
    )
    assert len(execution_runs) == contracts.STAGE_1_ICP_COUNT * participants
    assert all(run["status"] == "accepted" for run in execution_runs)


@pytest.mark.parametrize(
    ("malformation", "day"),
    (
        ("invalid_json", 18),
        ("wrong_run_id", 19),
        ("invalid_breakdowns", 20),
        ("failure_document", 21),
    ),
)
def test_malformed_accepted_scoring_artifact_cancels_without_partial_scores(
    connect, tmp_path, monkeypatch, malformation, day
):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["MalformedJudgeArtifact"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=day, epoch=30400 + day)
    challenger_id = next(
        participant["submission_id"]
        for participant in harness.service.store.get_round(harness.round_id)[
            "participants"
        ]
        if not participant["is_king"]
    )
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.run_stage_with_runners(2)
    closed = harness.service.advance_round(harness.round_id)
    score_run = next(
        run
        for run in harness.service.store.list_runs(
            harness.round_id,
            stage=1,
            submission_id=challenger_id,
            kind="score",
        )
        if run["status"] == "accepted"
    )
    original_get = harness.objects.get
    document = json.loads(original_get(score_run["output_ref"]).decode("utf-8"))
    if malformation == "invalid_json":
        malformed = b"{"
    elif malformation == "wrong_run_id":
        document["scored_run_id"] = "another-run"
        malformed = json.dumps(document).encode("utf-8")
    elif malformation == "invalid_breakdowns":
        document["breakdowns"] = []
        malformed = json.dumps(document).encode("utf-8")
    else:
        malformed = json.dumps(
            scoring.build_scoring_failure(
                score_run["scored_run_id"], "judge_error", "late failure"
            )
        ).encode("utf-8")

    def get(ref):
        return malformed if ref == score_run["output_ref"] else original_get(ref)

    monkeypatch.setattr(harness.objects, "get", get)
    cancelled = harness.service.advance_round(harness.round_id)

    row = harness.service.store.get_round(harness.round_id)
    assert closed["status"] == "closed" and closed["round_status"] == "stage1_judged"
    assert cancelled["status"] == "cancelled" and row["status"] == "cancelled"
    assert row["cancel_reason"] == svc.CANCEL_REASONS["scoring_incomplete"]
    assert row["publication_doc"] is None and not row["finalists"]
    execute_runs = harness.service.store.list_runs(
        harness.round_id, stage=1, kind="execute"
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_missing_scoring_result_reports_incomplete_without_partial_scores(
    connect, tmp_path, monkeypatch
):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["MissingJudgeResult"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=22, epoch=30422)
    _run_stage_one_to_scoring(harness, participants, runners=2)
    harness.run_stage_with_runners(2)
    closed = harness.service.advance_round(harness.round_id)
    participants_before = list(
        harness.service.store.get_round(harness.round_id)["participants"]
    )
    scoring_outputs = harness.service._scoring_outputs(harness.round_id, 1)
    missing_run_id = next(iter(scoring_outputs))
    monkeypatch.setattr(
        harness.service,
        "_scoring_outputs",
        lambda round_id, stage: {
            run_id: run
            for run_id, run in scoring_outputs.items()
            if run_id != missing_run_id
        },
    )

    cancelled = harness.service.advance_round(harness.round_id)

    row = harness.service.store.get_round(harness.round_id)
    assert closed["status"] == "closed" and closed["round_status"] == "stage1_judged"
    assert cancelled["status"] == "cancelled" and row["status"] == "cancelled"
    assert row["cancel_reason"] == svc.CANCEL_REASONS["scoring_incomplete"]
    assert row["participants"] == participants_before and not row["finalists"]
    execute_runs = harness.service.store.list_runs(
        harness.round_id, stage=1, kind="execute"
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_persistent_final_judge_failure_cancels_without_partial_final_scores(connect, tmp_path):
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
    finalists_before = list(
        harness.service.store.get_round(harness.round_id)["finalists"]
    )
    harness.sandbox.judge_failures.add(("JudgeFailFinal", 10))
    harness.advance_until("stage2_scoring", runners=2)
    harness.run_stage_with_runners(2)
    final_scoring_close = datetime.strptime(
        harness.schedule()["final_scoring_close"], "%Y-%m-%dT%H:%M:%SZ"
    ).replace(tzinfo=timezone.utc)
    assert harness.clock.now < final_scoring_close

    cancelled = harness.service.advance_round(harness.round_id)
    terminal = harness.service.advance_round(harness.round_id)

    row = harness.service.store.get_round(harness.round_id)
    assert cancelled["status"] == "cancelled" and row["status"] == "cancelled"
    assert terminal == {"status": "terminal", "round_status": "cancelled"}
    assert row["cancel_reason"] == svc.CANCEL_REASONS["scoring_incomplete"]
    assert row["finalists"] == finalists_before
    assert row["publication_doc"] is None
    execute_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=2,
        submission_id=failed["submission_id"],
        kind="execute",
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)


def test_miner_credential_failure_remains_challenger_ineligibility(
    connect, tmp_path, monkeypatch
):
    harness = Harness(
        connect,
        tmp_path,
        challengers=["CredentialFail", "CredentialPass"],
        runners=["alpha", "beta"],
    )
    participants = _start_round(harness, day=15, epoch=30275)
    failed = next(
        participant
        for participant in harness.service.store.get_round(harness.round_id)[
            "participants"
        ]
        if harness.flavors[participant["submission_id"]] == "CredentialFail"
    )
    _run_stage_one_to_scoring(harness, participants, runners=2)
    credentials = harness.service.config.credential_manager
    original_runtime_key = credentials.runtime_key

    def runtime_key(row, provider):
        if row["submission_id"] == failed["submission_id"]:
            return "miner-refused"
        return original_runtime_key(row, provider)

    monkeypatch.setattr(credentials, "runtime_key", runtime_key)
    harness.advance_until("stage1_scored", runners=2)

    row = harness.service.store.get_round(harness.round_id)
    assert failed["submission_id"] not in row["finalists"]
    score_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=failed["submission_id"],
        kind="score",
    )
    assert score_runs and {run["terminal_cause"] for run in score_runs} == {
        "credential_error"
    }
    assert {int(run["attempt"]) for run in score_runs} == {1}
    execute_runs = harness.service.store.list_runs(
        harness.round_id,
        stage=1,
        submission_id=failed["submission_id"],
        kind="execute",
    )
    assert all(run["per_icp_score"] is None for run in execute_runs)
    harness.service.cancel(harness.round_id, sorted(svc.CANCEL_REASONS.values())[0])


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
    """A prior winner can submit fresh source beside its promoted baseline source."""

    harness = Harness(connect, tmp_path, challengers=["Regal", "Rival"], runners=["alpha"])
    service = harness.service
    participants = _start_round(harness, day=16, epoch=30020)  # epochs rise in module order: the shared database keeps every published king
    _run_stage_one_to_scoring(harness, participants, runners=1)
    harness.advance_until("published", runners=1)
    first = service.store.get_round(harness.round_id)
    king_hotkey = first["king_hotkey"]
    assert first["king_outcome"] == "crowned" and king_hotkey
    king_label = next(flavor for flavor in ("Regal", "Rival") if keypair("svc-miner-" + flavor).ss58_address == king_hotkey)
    repository_root = tmp_path / "fresh-promotion-repository"
    repository_root.mkdir()
    remote = promotion_repository(repository_root)
    service._config.baseline_promoter_factory = lambda: GitPromoter(
        str(remote), tmp_path / "fresh-promotion-objects"
    )

    def promoted_baseline(_url, _limit):
        return subprocess.run(
            ("git", "--git-dir", str(remote), "archive", "--format=tar.gz", "lab"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout

    service._config.baseline_source_fetcher = promoted_baseline
    harness.clock.now = datetime.fromisoformat(
        str(first["published_at"]).replace("Z", "+00:00")
    )
    assert service.promote_pending_baselines() == {"status": "ok", "promoted": 1}
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
    baseline_row = service.store.get_submission(baseline_parts[0]["submission_id"])
    with tarfile.open(
        fileobj=io.BytesIO(harness.objects.get(baseline_row["source_ref"])), mode="r:gz"
    ) as archive:
        assert archive.extractfile("flavor.txt").read().decode("utf-8") == king_label
    _run_stage_one_to_scoring(harness, 3, runners=1)
    harness.advance_until("published", runners=1)
    second = service.store.get_round(harness.round_id)
    assert second["king_outcome"] in ("crowned", "no_king")
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
    retried_execution_ids = {
        run["scored_run_id"] for run in accepted if int(run["attempt"]) == 2
    }
    retried_execution_runs = [
        run
        for run in service.store.list_runs(round_id, stage=1, kind="execute")
        if run["run_id"] in retried_execution_ids
    ]
    assert len(retried_execution_runs) == 3
    assert all(float(run["per_icp_score"]) > 0.0 for run in retried_execution_runs)


def test_validators_complete_a_round_over_the_http_api(connect, tmp_path, monkeypatch):
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
    public_scores = results.json()["scores"]["stage_1"] + results.json()["scores"]["stage_2"]
    assert len(public_scores) == 20
    benchmark = original_get("http://localhost/arena/v1/rounds/%s/benchmark" % harness.round_id)
    assert benchmark.status_code == 200
    public_positions = {icp["icp_position"] for icp in benchmark.json()["icps"]}
    assert len(public_positions) == 20
    assert {score["icp_position"] for score in public_scores} == public_positions
    assert benchmark.json()["disclosure_policy"] == "all_20_next_day"
    assert benchmark.json()["private_icp_count"] == 0
    # Full parity must validate all persisted baseline outputs while respecting
    # the same public HTTP partition used by the dashboard.
    from scripts.run_production_parity_full_host import (
        FullParityError,
        _verify_arena_daily_public_results,
    )

    baseline_id = next(p["submission_id"] for p in row["participants"] if p["is_king"])
    baseline_public = original_get(
        "http://localhost/arena/v1/rounds/%s/results/%s" % (harness.round_id, baseline_id)
    ).json()
    verify_args = dict(
        service=harness.service, round_id=harness.round_id,
        baseline_submission_id=baseline_id,
        icps=harness.service.benchmark_icps(harness.round_id),
        round_view=public.json(), benchmark_view=benchmark.json(),
        results_view=baseline_public,
    )
    baseline_runs, persisted_outputs = _verify_arena_daily_public_results(**verify_args)
    assert len(baseline_runs) == len(persisted_outputs) == 20
    assert len(baseline_public["outputs"]) == 20
    missing = dict(baseline_public, scores={**baseline_public["scores"], "stage_1": []})
    with pytest.raises(FullParityError, match="public daily scores differ"):
        _verify_arena_daily_public_results(**dict(verify_args, results_view=missing))
    original_read = harness.service._objects.get_bounded

    missing_run = baseline_runs[0]

    def missing_persisted_output(ref, maximum):
        if ref == missing_run["output_ref"]:
            raise FileNotFoundError("private output unavailable")
        return original_read(ref, maximum)

    with monkeypatch.context() as patch:
        patch.setattr(harness.service._objects, "get_bounded", missing_persisted_output)
        with pytest.raises(FullParityError, match="persisted company output is invalid"):
            _verify_arena_daily_public_results(**verify_args)
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
    assert len(results["scores"]["stage_1"]) + len(results["scores"]["stage_2"]) == 20
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

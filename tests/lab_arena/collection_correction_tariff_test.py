"""Bounded fixed correction quotes use original receipts and ordinary settlement."""

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest
from psycopg2.extras import Json
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from lab_arena import collection_intake as funding, contracts
from lab_arena import collection_correction_tariff as tariff
from lab_arena.collection_experiment import load_experiment
from tests.lab_arena.collection_experiment_test import settings, signed, pub
from tests.lab_arena.collection_calibration_test import exercise_ordinary_cutover
from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    database_with_lab_arena_migration,
)


@pytest.fixture
def database():
    files = []
    for n in (720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 732, 733, 738, 740):
        (path,) = Path("scripts").glob(str(n) + "-*.sql")
        files.append(path.name)
    yield from database_with_lab_arena_migration((*DEFAULT_MIGRATIONS, *files))


def test_new_terms_and_class_require_separate_authority(monkeypatch):
    pilot = contracts.hash_bytes(funding.CORRECTION_TERMS.encode())
    live = contracts.hash_bytes(funding.LIVE_CORRECTION_TERMS.encode())
    assert pilot != live
    monkeypatch.setenv("LAB_ARENA_CORRECTION_TERMS_SHA256", pilot)
    monkeypatch.setenv("LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256", pilot)
    with pytest.raises(funding.IntakeError):
        funding.terms("live", funding.CORRECTION_SCOPE)
    monkeypatch.setenv("LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256", live)
    assert funding.terms("live", funding.CORRECTION_SCOPE)["terms_sha256"] == live
    raw = settings()
    priced = dict(raw, min_pricing_runs=2)
    basem = load_experiment(priced, {})
    assert basem.batch_id != load_experiment(raw, {}).batch_id
    prospective = dict(
        raw, mode="live", pricing_pilot=basem.batch_id, terms_sha256=live
    )
    assert load_experiment(prospective, {}).policy.pilot is False
    with pytest.raises(ValueError):
        load_experiment(dict(priced, min_pricing_runs=3), {})
    with pytest.raises(ValueError):
        load_experiment(
            dict(prospective, arms=[dict(raw["arms"][0], role="expert")]), {}
        )


@pytest.mark.parametrize("template", ["C06", "P7"])
def test_native_priced_job_reuses_pro_rata_quotes_once(database, template):
    pg, dsn = database
    pid = "xp-" + "3" * 64
    live = "xp-" + "4" * 64
    sig = "v" * 86 + "=="
    panel = "2" * 64
    model = funding.MODELS[1]
    budget = 5000
    ph = contracts.hash_bytes(funding.CORRECTION_TERMS.encode())
    lh = contracts.hash_bytes(funding.LIVE_CORRECTION_TERMS.encode())
    hk = "5" + "T" * 47

    def reg(q, mode, digest):
        rid, key = map(str, (uuid4(), uuid4()))
        q.execute(
            "SELECT public.collection_register_v1(%s,%s,%s,%s,%s,10000,%s,NULL)",
            (
                rid,
                hk,
                hk,
                mode,
                Json(
                    {
                        "body": {
                            "task_scope": "synthetic_corrections",
                            "terms_sha256": digest,
                        }
                    }
                ),
                Json(
                    {
                        "key_id": key,
                        "fingerprint": "sha256:"
                        + ("c" if mode == "pilot" else "a") * 64,
                        "ciphertext_b64": "fictional",
                    }
                ),
            ),
        )
        return q.fetchone()[0]

    def claim(q, batch, mode, price, digest):
        token = "sha256:" + "b" * 64
        q.execute(
            "SELECT public.collection_claim_v1(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (
                str(uuid4()),
                token,
                batch,
                mode,
                "b" * 40,
                Json([f"{template}-s3013"]),
                Json([model]),
                Json({model: price}),
                Json({model: budget}),
                digest,
            ),
        )
        return q.fetchone()[0]

    def bill(q, j, actual):
        lease = "sha256:" + "b" * 64
        job = j["job_id"]
        q.execute(
            "SELECT public.collection_reserve_call_v1(%s,%s,'one','openrouter',%s,250)",
            (job, lease, "2" * 64),
        )
        assert q.fetchone()[0]["status"] == "reserved"
        q.execute(
            "SELECT public.collection_record_call_v1(%s,%s,'one',%s,%s,%s,200)",
            (job, lease, "3" * 64, "gen-" + job, actual),
        )
        assert q.fetchone()[0]["status"] == "recorded"
        q.execute(
            "SELECT public.collection_finish_v1(%s,%s,'accepted',%s,%s,103)",
            (
                job,
                lease,
                Json({"training_status": "code_rejected"}),
                "sha256:" + "f" * 64,
            ),
        )
        assert q.fetchone()[0]["status"] == "accepted"

    def sample(q):
        assert reg(q, "pilot", ph)["status"] == "enrolled"
        pj = claim(q, pid, "pilot", 0, ph)
        bill(q, pj, 104)
        q.execute("SELECT public.collection_correction_measurement_v1(%s)", (pid,))
        m = q.fetchone()[0]
        assert (m["completed"], m["quote_microusd"], m["cost_microusd"]) == (
            1,
            130,
            104,
        )
        return {
            "schema": tariff.SCHEMA,
            "live_batch": live,
            "pilot_batch": pid,
            "measurement": m,
            "minimum_runs": 1,
            "class_sha256": "4" * 64,
            "cases": [f"{template}-s3013"],
            "panel_sha256": panel,
            "terms_sha256": lh,
            "evidence_sha256": "sha256:" + "1" * 64,
            "signature": sig,
        }

    with pg.connect(**dsn) as c, c.cursor() as q:
        d = sample(q)
        assert reg(q, "live", lh)["error"] == "collection_consent_invalid"
        assert (
            claim(q, live, "live", 130, lh)["error"] == "unapproved_correction_tariff"
        )
        q.execute("SAVEPOINT prematurely")
        with pytest.raises(pg.Error):
            q.execute(
                "INSERT INTO public.collection_correction_tariffs(batch_id,pilot_id,approval) VALUES(%s,%s,%s)",
                (live, pid, Json(d)),
            )
        q.execute("ROLLBACK TO prematurely")

    def after(q):
        def insert(document):
            q.execute(
                "INSERT INTO public.collection_correction_tariffs(batch_id,pilot_id,approval) VALUES(%s,%s,%s)",
                (live, pid, Json(document)),
            )

        q.execute("SAVEPOINT tampered")
        with pytest.raises(pg.Error, match="unapproved"):
            insert(dict(d, measurement=dict(d["measurement"], quote_microusd=999)))
        q.execute("ROLLBACK TO tampered")
        q.execute("SET LOCAL ROLE lab_arena_service")
        q.execute("SAVEPOINT no_pay_grant")
        with pytest.raises(pg.Error, match="permission denied"):
            insert(d)
        q.execute("ROLLBACK TO no_pay_grant")
        q.execute("RESET ROLE")
        insert(d)
        assert reg(q, "live", ph)["error"] == "collection_consent_invalid"
        assert reg(q, "live", lh)["status"] == "enrolled"
        assert (
            claim(q, live, "live", 131, lh)["error"] == "unapproved_correction_tariff"
        )
        accepted = claim(q, live, "live", 130, lh)
        assert accepted["price_microusd"] == 130
        bill(q, accepted, 220)
        q.execute(
            "SELECT public.collection_finish_v1(%s,%s,'accepted',%s,%s,103)",
            (
                accepted["job_id"],
                "sha256:" + "b" * 64,
                Json({"training_status": "code_rejected"}),
                "sha256:" + "f" * 64,
            ),
        )
        assert q.fetchone()[0]["status"] == "accepted"
        q.execute("SELECT public.collection_reward_quotes_v1(104)")
        assert q.fetchone()[0]["quotes_microusd"][hk] == 130
        assert claim(q, live, "live", 130, lh) is None
        q.execute("SELECT public.collection_reward_quotes_v1(105)")
        assert hk not in q.fetchone()[0]["quotes_microusd"]

    exercise_ordinary_cutover(database, paid_panel=panel, after_cutover=after)


def test_worker_requires_exact_operator_signature_class_review_and_current_measurement(
    tmp_path, monkeypatch
):
    from lab_arena import collection_judge as judge
    from lab_arena.collection_workflow import store_artifact
    from lab_arena.service import LocalObjectStore
    from cryptography.exceptions import InvalidSignature

    obj = LocalObjectStore(tmp_path / "objects")
    raw = settings()
    raw["min_pricing_runs"] = 1
    pilot = load_experiment(raw, {})
    live_raw = dict(
        raw,
        mode="live",
        pricing_pilot=pilot.batch_id,
        terms_sha256=contracts.hash_bytes(funding.LIVE_CORRECTION_TERMS.encode()),
    )
    del live_raw["min_pricing_runs"]
    live = load_experiment(live_raw, {})
    test_key = Ed25519PrivateKey.generate()
    k = tmp_path / "test-tariff.pub"
    k.write_bytes(pub(test_key))
    monkeypatch.setenv("LAB_ARENA_COLLECTION_TARIFF_PUBLIC_KEY_FILE", str(k))
    monkeypatch.setenv(
        "LAB_ARENA_LIVE_CORRECTION_TERMS_SHA256", live_raw["terms_sha256"]
    )
    model = raw["arms"][0]["model"]
    measured = {
        "completed": 2,
        "model": model,
        "template": "C06",
        "harness_commit": "1" * 40,
        "cap_microusd": 120000,
        "cost_microusd": 210,
        "quote_microusd": 132,
        "receipts_sha256": "b" * 64,
    }
    sources = (
        {
            "source": (
                {"student_model": "student/4b"},
                {
                    "template_id": "C06",
                    "rule_pool": "rules",
                    "tool_variant": "fn",
                    "manifest_sha256": "e" * 64,
                },
            )
        },
        {},
        {"sources": {"source": {"model_revision": "9" * 40}}},
    )
    cert = {
        "schema": "fixture",
        "allow_live": True,
        "signature": "fixture-panel-signature",
    }
    panel = judge.digest(cert)
    release = {
        "batch_id": pilot.batch_id,
        "settings_sha256": pilot.settings_hash,
        "rows": 1,
        "qualification_sha256": "sha256:"
        + judge.digest({k: v for k, v in cert.items() if k != "signature"}),
        "price_observations": [
            {
                "model": model,
                "role": "continuation",
                "task_class": "C06",
                "measured_runs": 2,
                "miner_cost_microusd": 210,
                "proposed_fixed_microusd": 132,
            }
        ],
    }

    class Database:
        approval = None

        def rpc(self, name, p):
            if name == "collection_reward_schedule_v1":
                return {
                    "first_epoch": 10,
                    "pilot_evidence": {"panel_qualification_sha256": panel},
                }
            if name == "collection_correction_measurement_v1":
                return dict(measured)
            assert name == "collection_correction_tariff_v1"
            return self.approval

    db = Database()

    def released(spec, db, objects, dest, *, qualification):
        (dest.parent / "guarded-release-check").write_text("test only")
        store_artifact(objects, pilot.batch_id, "export", "release", release)
        return release

    monkeypatch.setattr(tariff, "write_dataset", released)
    monkeypatch.setattr(tariff, "read_inputs", lambda *args: sources)
    candidate = tariff.prepare(live, pilot, None, None, db, obj, {"approval": cert})
    db.approval = signed(test_key, candidate)
    verified = tariff.load_approved(live, sources, {}, db, obj, lambda: 10)
    assert verified == candidate
    with pytest.raises(ValueError, match="epoch"):
        tariff.load_approved(live, sources, {}, db, obj, lambda: 9)
    other = load_experiment(dict(live_raw, id="new-authorization-needed"), {})
    with pytest.raises(ValueError, match="differs"):
        tariff.load_approved(other, sources, {}, db, obj, lambda: 10)
    altered = deepcopy(sources)
    altered[2]["sources"]["source"]["model_revision"] = "d" * 40
    with pytest.raises(ValueError, match="differs"):
        tariff.load_approved(live, altered, {}, db, obj, lambda: 10)
    db.approval = dict(db.approval, terms_sha256="sha256:" + "f" * 64)
    with pytest.raises(InvalidSignature):
        tariff.load_approved(live, sources, {}, db, obj, lambda: 10)
    db.approval = signed(test_key, candidate)
    measured["cost_microusd"] = 205
    with pytest.raises(ValueError, match="receipts changed"):
        tariff.load_approved(live, sources, {}, db, obj, lambda: 10)

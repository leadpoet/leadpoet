"""Budget-only Deepline admission and immutable execution-key recovery."""

from dataclasses import replace
from pathlib import Path

import pytest
from contextlib import contextmanager

from lab_arena import contracts
from lab_arena.store import ArenaStoreError, hash_lease_token
from tests.lab_arena.lab_arena_pg_harness import (
    CURRENT_SERVICE_MIGRATIONS,
    database_with_lab_arena_migration,
)
from tests.lab_arena.parallel_twenty_icp_execution_postgres_test import (
    _start_parallel_round,
)
from tests.lab_arena.per_icp_cost_admission_postgres_test import _settle
from tests.lab_arena.test_lab_arena_migration_postgres import claim, sha
from tests.lab_arena.test_lab_arena_service_round import Harness

MIGRATION = "415-lab-arena-deepline-budget-only-exact-recovery.sql"


@pytest.fixture(scope="module")
def database():
    yield from database_with_lab_arena_migration(
        CURRENT_SERVICE_MIGRATIONS
        + (
            "264-lab-arena-codex-cost-reconciliation.sql",
            "311-lab-arena-per-icp-closed-billing-reconciliation.sql",
            "312-lab-arena-temporary-hold-admission.sql",
            "314-lab-arena-openrouter-web-search-reservation.sql",
            "319-lab-arena-quota-sourcing-cost.sql",
            "321-lab-arena-confirmed-cost-admission.sql",
            "407-lab-arena-cost-run-lookup.sql",
            "408-lab-arena-cost-run-index.sql",
            "411-lab-arena-settlement-success-json-once.sql",
            "413-lab-arena-closed-openrouter-judge-billing.sql",
            MIGRATION,
            MIGRATION,
            "417-lab-arena-deepline-response-recovery.sql",
            "417-lab-arena-deepline-response-recovery.sql",
        )
    )


@contextmanager
def fixture_configuration(connect):
    # This disposable fixture emulates a newly frozen policy before service
    # wiring lands. Production migration never bypasses write-once guards.
    with connect() as c, c.cursor() as cur:
        cur.execute(
            "ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once"
        )
        yield cur
        cur.execute(
            "ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once"
        )


def run(database, tmp_path, label, catalog=True):
    connect = lambda: database[0].connect(**database[1])
    h = Harness(connect, tmp_path, challengers=[], runners=[label])
    h.service.config.defaults = replace(
        h.service.config.defaults,
        per_icp_cost_policy=True,
        integrity_from="2000-01-01T00:00:00Z",
    )
    _start_parallel_round(h, "arena-2099-08-" + label, slot_ceiling=2)
    if catalog:
        with fixture_configuration(connect) as cur:
            cur.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc = "
                "jsonb_set(jsonb_set(configuration_doc,'{call_quotas,deepline}','0'),"
                "'{deepline_catalog}',%s::jsonb) WHERE round_id=%s",
                (
                    '{"schema_version":"leadpoet.lab_arena.deepline_catalog.v1"}',
                    h.round_id,
                ),
            )
    lease, token = claim(
        h.service.store, h.round_id, h.runner_keys[0], parallelism=2, ceiling=2
    )[:2]
    return h, lease, token, connect


def reserve(h, lease, token, label, *, provider="deepline", amount=0, doc=None):
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha(label),
    )
    result = h.service.store.reserve_call(
        run_id=lease["run_id"],
        lease_token_hash=hash_lease_token(token),
        call_identity=identity,
        operation_id="deepline.execute",
        provider=provider,
        funding_source=h.service.store.provider_funding(lease["run_id"], provider)[
            "funding_source"
        ],
        amount_microusd=amount,
        call_doc={"request_hash": sha(label), **(doc or {})},
    )
    return identity, result


def test_more_than_200_free_calls_and_confirmed_budget(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "01")
    for n in range(205):
        ident, result = reserve(h, lease, token, "free-" + str(n))
        assert result["status"] == "reserved"
        _settle(h.service.store, lease, token, ident, 0)
    snapshot = h.service.store.run_quota_snapshot(
        lease["run_id"], hash_lease_token(token)
    )
    assert snapshot["providers"]["deepline"] == {
        "limit": 0,
        "used": 205,
        "remaining": None,
        "inflight": 0,
    }
    ident, result = reserve(h, lease, token, "confirmed-cap", amount=1)
    assert result["status"] == "reserved"
    _settle(h.service.store, lease, token, ident, 4_000_000)
    _, result = reserve(h, lease, token, "next-paid", amount=1)
    assert result["status"] == "refused" and result["reason"] == "money_cap"
    _, result = reserve(h, lease, token, "still-free")
    assert result["status"] == "reserved"
    with fixture_configuration(connect) as cur:
        cur.execute(
            "SELECT max(amount_microusd) FROM public.lab_arena_ledger WHERE run_id=%s AND entry_kind='reservation'",
            (lease["run_id"],),
        )
        assert cur.fetchone()[0] == 0


def test_legacy_and_other_provider_caps_stay_enforced(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "02", catalog=False)
    with fixture_configuration(connect) as cur:
        cur.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas,deepline}','2') WHERE round_id=%s",
            (h.round_id,),
        )
    for n in range(2):
        ident, result = reserve(h, lease, token, "legacy-" + str(n))
        assert result["status"] == "reserved"
        _settle(h.service.store, lease, token, ident, 0)
    _, result = reserve(h, lease, token, "legacy-3")
    assert result["reason"] == "per_icp_quota"
    assert (
        h.service.store.run_quota_snapshot(lease["run_id"], hash_lease_token(token))[
            "providers"
        ]["deepline"]["remaining"]
        == 0
    )
    h, lease, token, connect = run(database, tmp_path, "03")
    with fixture_configuration(connect) as cur:
        cur.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas,scrapingdog}','1') WHERE round_id=%s",
            (h.round_id,),
        )
    ident, result = reserve(h, lease, token, "other-1", provider="scrapingdog")
    assert result["status"] == "reserved"
    _settle(h.service.store, lease, token, ident, 0)
    _, result = reserve(h, lease, token, "other-2", provider="scrapingdog")
    assert result["reason"] == "per_icp_quota"


@pytest.mark.parametrize(
    "change",
    [
        "configuration_doc - 'deepline_catalog'",
        "jsonb_set(configuration_doc,'{deepline_catalog,schema_version}','\"wrong\"')",
    ],
)
def test_zero_requires_catalog_policy(database, tmp_path, change):
    h, lease, token, connect = run(database, tmp_path, "04-" + sha(change)[-8:])
    with fixture_configuration(connect) as cur:
        cur.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc="
            + change
            + " WHERE round_id=%s",
            (h.round_id,),
        )
    with pytest.raises(ArenaStoreError, match="quota_missing"):
        reserve(h, lease, token, "invalid-zero")


def test_keyed_recovery_keeps_candidate_identity_and_rejects_mismatches(
    database, tmp_path
):
    h, lease, token, connect = run(database, tmp_path, "05")
    label = "keyed"
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha(label),
    )
    local = "ctx-tool-" + identity[7:39]
    key = "arena:" + identity[7:]
    fingerprint = "sha256:" + "a" * 64
    ident, result = reserve(
        h,
        lease,
        token,
        label,
        amount=1,
        doc={
            "deepline_request_id": local,
            "deepline_execution_key": key,
            "tool": "exa_search",
            "credential_fingerprint": fingerprint,
            "deepline_billing_provider": "exa",
            "deepline_operation_aliases": ["search", "exa_search"],
        },
    )
    assert ident == identity and result["status"] == "reserved"
    store = h.service.store
    assert (
        store.mark_dispatched(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
        )["status"]
        == "dispatched"
    )
    assert (
        store.mark_uncertain(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=identity,
            call_doc={
                "reason": "transport_failure",
                "call_succeeded": False,
                "deepline_request_id": local,
                "deepline_operation": "exa_search",
                "credential_fingerprint": fingerprint,
                "deepline_execution_key": key,
            },
        )["status"]
        == "uncertain"
    )
    candidate = store.list_deepline_cost_reconciliations(
        h.round_id, run_id=lease["run_id"]
    )[0]
    assert candidate["request_id"] == local and candidate["execution_key"] == key
    assert candidate["billing_provider"] == "exa" and candidate[
        "operation_aliases"
    ] == ["search", "exa_search"]
    args = dict(
        round_id=h.round_id,
        run_id=lease["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=local,
        operation="exa_search",
        credential_fingerprint=fingerprint,
        actual_microusd=2000,
        cost_units="0.02",
        execution_key=key,
        recovered_request_id="exa::search-1791325000000-" + "b" * 32,
    )
    with pytest.raises(ArenaStoreError, match="execution_recovery_input_invalid"):
        store.reconcile_deepline_cost(**{**args, "execution_key": "arena:" + "0" * 64})
    with pytest.raises(ArenaStoreError, match="reconciliation_input_invalid"):
        store.reconcile_deepline_cost(**{**args, "request_id": "ctx-tool-" + "0" * 32})
    result = store.reconcile_deepline_cost(**args)
    assert result["status"] == "settled" and result["actual_microusd"] == 2000
    assert store.reconcile_deepline_cost(**args)["idempotent"] is True
    assert (
        store.reconcile_deepline_cost(
            **{**args, "recovered_request_id": "exa::search-1791325000000-" + "c" * 32}
        )["status"]
        == "conflict"
    )
    with fixture_configuration(connect) as cur:
        cur.execute(
            "SELECT entry_doc,terminal_response FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'",
            (identity,),
        )
        doc, terminal = cur.fetchone()
        assert doc["deepline_request_id"] == local
        assert terminal["provider_cost"]["request_id"] == args["recovered_request_id"]
        cur.execute(
            "SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'",
            (identity,),
        )
        assert cur.fetchone()[0] == 1


def test_pending_zero_liabilities_do_not_block_paid_work(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "06")
    ident, result = reserve(h, lease, token, "pending", amount=900_000)
    assert result["status"] == "reserved"
    assert (
        h.service.store.mark_dispatched(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=ident,
        )["status"]
        == "dispatched"
    )
    _, result = reserve(h, lease, token, "after-pending", amount=1)
    assert result["status"] == "reserved"
    with connect() as c, c.cursor() as cur:
        cur.execute(
            "SELECT amount_microusd FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='reservation'",
            (ident,),
        )
        assert cur.fetchone()[0] == 0


def test_zero_is_never_unlimited_for_other_provider(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "07")
    with fixture_configuration(connect) as cur:
        cur.execute(
            "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{call_quotas,openrouter}','0') WHERE round_id=%s",
            (h.round_id,),
        )
    with pytest.raises(ArenaStoreError, match="quota_missing"):
        reserve(h, lease, token, "invalid-other-zero", provider="openrouter")
    with pytest.raises(ArenaStoreError, match="quota_missing"):
        h.service.store.run_quota_snapshot(lease["run_id"], hash_lease_token(token))


def test_ledger_cursor_scans_more_than_400_rows_without_leakage(database, tmp_path):
    h, lease, token, connect = run(database, tmp_path, "08")
    for n in range(205):
        ident, result = reserve(h, lease, token, "paged-" + str(n))
        assert result["status"] == "reserved"
        _settle(h.service.store, lease, token, ident, 0)
    rows = []
    cursor = 0
    while True:
        page = h.service.store.list_ledger(
            run_id=lease["run_id"],
            provider="deepline",
            operation_id="deepline.execute",
            after_entry_id=cursor,
            limit=100,
        )
        if not page:
            break
        rows.extend(page)
        cursor = page[-1]["entry_id"]
    assert len(rows) == 615
    assert len({row["entry_id"] for row in rows}) == 615
    assert all(row["run_id"] == lease["run_id"] for row in rows)
    assert (
        h.service.store.list_ledger(
            run_id=lease["run_id"],
            operation_id="openrouter.responses",
            after_entry_id=0,
        )
        == []
    )


def test_catalog_freezes_atomically_and_cannot_change_later(database, tmp_path):
    from datetime import datetime, timedelta, timezone
    from tests.lab_arena.test_lab_arena_migration_postgres import (
        round_config,
        frozen_participants,
        hotkey,
    )
    from lab_arena.store import ArenaStore, PsycopgTransport

    connect = lambda: database[0].connect(**database[1])
    store = ArenaStore(PsycopgTransport(connect))
    rid = "arena-2099-08-09"
    day = datetime.now(timezone.utc).date()
    bank = day - timedelta(days=1)
    cfg = round_config(rid, [hotkey("freeze-runner")], cost_per_company_microusd=800000)
    cfg.update(
        sourcing_cost_eligibility_policy="successful_calls_per_icp_v1",
        execution_icp_cap_microusd=4000000,
    )
    cfg["schedule"] = {
        "submission_open": bank.isoformat() + "T00:00:00Z",
        "submission_cutoff": (datetime.now(timezone.utc) + timedelta(minutes=30))
        .isoformat()
        .replace("+00:00", "Z"),
    }
    assert store.create_round(rid, cfg)["status"] == "created"
    participants = frozen_participants(
        store, rid, 1, prefix="catalog-freeze", king_index=0
    )
    catalog = {
        "schema_version": "leadpoet.lab_arena.deepline_catalog.v1",
        "policy_version": "leadpoet.lab_arena.deepline_company_research.v1",
        "allow_people": False,
        "tools": [{"tool_id": "exa_search"}],
        "catalog_hash": "a" * 64,
    }
    args = dict(
        participants=participants,
        benchmark_ref="arena/" + rid + "/benchmark.json",
        evaluation_date=day.isoformat(),
        icp_set_date=bank.isoformat(),
        scorer_image_digest=cfg["scorer_image_digest"],
        scorer_image_reference=cfg["scorer_image_reference"],
        deepline_catalog=catalog,
    )
    before = store.get_round(rid)["configuration_doc"]
    with pytest.raises(ArenaStoreError, match="catalog_commit_invalid"):
        store.commit_round_v2(
            rid, **{**args, "deepline_catalog": {**catalog, "allow_people": True}}
        )
    assert store.get_round(rid)["configuration_doc"] == before
    assert store.get_round(rid)["status"] == "open"
    assert store.commit_round_v2(rid, **args)["status"] == "ok"
    after = store.get_round(rid)["configuration_doc"]
    assert after["deepline_catalog"] == catalog
    assert after["call_quotas"] == {**before["call_quotas"], "deepline": 0}
    assert after["scoring_call_quotas"] == {
        **before["scoring_call_quotas"],
        "deepline": 0,
    }
    assert (
        store.commit_round_v2(
            rid, **{**args, "deepline_catalog": {**catalog, "catalog_hash": "b" * 64}}
        )["status"]
        == "stale"
    )
    with connect() as c, c.cursor() as cur:
        with pytest.raises(database[0].Error, match="write-once"):
            cur.execute(
                "UPDATE public.lab_arena_rounds SET configuration_doc=jsonb_set(configuration_doc,'{deepline_catalog,catalog_hash}','\"different\"') WHERE round_id=%s",
                (rid,),
            )
    assert store.get_round(rid)["configuration_doc"] == after
    store.close()


def test_catalog_readiness_grants_only_service_and_checks_sql_guards(database):
    from lab_arena.store import ArenaStore, PsycopgTransport

    connect = lambda: database[0].connect(**database[1])
    store = ArenaStore(PsycopgTransport(connect))
    assert store.deepline_catalog_schema() == {
        "schema_version": "leadpoet.lab_arena.deepline_catalog_schema.v1",
        "version": 415,
    }
    with connect() as c, c.cursor() as cur:
        for role in ("anon", "authenticated", "service_role", "lab_arena_service"):
            cur.execute(
                "SELECT has_function_privilege(%s,'public.lab_arena_deepline_catalog_schema_v1()','EXECUTE')",
                (role,),
            )
            assert cur.fetchone()[0] == (role == "lab_arena_service")
        cur.execute(
            "ALTER FUNCTION public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb) RENAME TO temporarily_absent_commit_v3"
        )
        with pytest.raises(database[0].Error, match="schema_incomplete"):
            cur.execute("SELECT public.lab_arena_deepline_catalog_schema_v1()")
        c.rollback()
    assert store.deepline_catalog_schema()["version"] == 415
    store.close()


@pytest.mark.parametrize("reservation_case", ["native", "no_provider", "local_only", "keyed"])
def test_unkeyed_async_receipt_settles_only_its_immutable_native_id(
    database, tmp_path, reservation_case
):
    h, lease, token, connect = run(database, tmp_path, "10-" + sha(reservation_case)[-8:])
    label = "unkeyed"
    identity = contracts.provider_call_identity(
        attempt=lease["attempt"],
        assignment_id=lease["assignment_id"],
        icp_position=lease["icp_position"],
        action_sequence=0,
        operation_id="deepline.execute",
        request_hash=sha(label),
    )
    local = "ctx-tool-" + identity[7:39]
    fingerprint = "sha256:" + "a" * 64
    native = local if reservation_case == "local_only" else "firecrawl.batch:request-001"
    reservation_doc = {
        "deepline_request_id": local,
        "tool": "firecrawl_batch_scrape",
        "credential_fingerprint": fingerprint,
        "deepline_operation_aliases": ["batch_scrape"],
    }
    if reservation_case != "no_provider":
        reservation_doc["deepline_billing_provider"] = "firecrawl"
    if reservation_case == "keyed":
        reservation_doc["deepline_execution_key"] = "arena:" + identity[7:]
    ident, result = reserve(
        h,
        lease,
        token,
        label,
        amount=1,
        doc=reservation_doc,
    )
    store = h.service.store
    assert (
        store.mark_dispatched(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=ident,
        )["status"]
        == "dispatched"
    )
    assert (
        store.mark_uncertain(
            run_id=lease["run_id"],
            lease_token_hash=hash_lease_token(token),
            call_identity=ident,
            call_doc={
                "reason": "missing_provider_cost",
                "call_succeeded": True,
                "deepline_request_id": local,
                "deepline_job_id": native,
                "deepline_operation": "firecrawl_batch_scrape",
                "credential_fingerprint": fingerprint,
            },
        )["status"]
        == "uncertain"
    )
    candidate = store.list_deepline_cost_reconciliations(
        h.round_id, run_id=lease["run_id"]
    )[0]
    assert candidate["request_id"] == native
    assert candidate["execution_key"] == reservation_doc.get("deepline_execution_key")
    args = dict(
        round_id=h.round_id,
        run_id=lease["run_id"],
        call_identity=identity,
        uncertain_entry_id=candidate["uncertain_entry_id"],
        request_id=native,
        operation="firecrawl_batch_scrape",
        credential_fingerprint=fingerprint,
        actual_microusd=2000,
        cost_units="0.02",
        recovered_request_id=native,
    )
    assert (
        store.reconcile_deepline_cost(
            **{**args, "recovered_request_id": "other-request"}
        )["status"]
        == "stale"
    )
    if reservation_case != "native":
        assert store.reconcile_deepline_cost(**args)["status"] == "stale"
        with connect() as c, c.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM public.lab_arena_ledger WHERE call_identity=%s AND entry_kind='settlement'",
                (identity,),
            )
            assert cur.fetchone()[0] == 0
        return
    assert store.reconcile_deepline_cost(**args)["status"] == "settled"
    assert store.reconcile_deepline_cost(**args)["idempotent"] is True

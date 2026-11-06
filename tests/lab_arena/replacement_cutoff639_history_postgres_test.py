"""Cutoff metadata composes with retained633 authority through publication."""
import pytest

from tests.lab_arena import nov05_infrastructure_recovery633_postgres_test as recovery
from tests.postgres_migration_harness import SCRIPTS


@pytest.fixture(scope='module')
def database():
    fixture=recovery.database.__wrapped__()
    try:
        pg,dsn=next(fixture)
        with pg.connect(**dsn) as c,c.cursor() as q:
            for filename in ('594-lab-arena-private-review-diagnostics.sql','599-lab-arena-refusal-review-fallback.sql'):
                q.execute((SCRIPTS/filename).read_text())
            text=(SCRIPTS/'605-arena-2026-11-03-admission-review-recovery.sql').read_text()
            #621 already installed the frozen trigger lineage; only review
            #helpers and exact review begin/finish are needed for this composition.
            q.execute(text.split('DO $patch605_2$')[0]+'DO $restore605$'+text.split('DO $restore605$',1)[1])
            for filename in ('624-lab-arena-invalid-review-reservation-budget.sql','634-lab-arena-invalid-review-alternate.sql'):
                q.execute((SCRIPTS/filename).read_text())
            text=(SCRIPTS/'635-arena-2026-11-06-late-review-recovery.sql').read_text()
            q.execute(text.split('DO $recover635$')[0]+'DO $restore635$'+text.split('DO $restore635$',1)[1])
        yield pg,dsn
    finally:fixture.close()


def test_633_native_recovery_claim_completion_publication_and_history_after639(database, monkeypatch):
    p=recovery.seed(database)
    try:
        original_apply,original_snapshot=recovery.apply,recovery.snapshot
        applied=False
        def apply_with_cutoff(prepared,sql=None):
            nonlocal applied
            original_apply(prepared,sql)
            if not applied:
                # Preserve an already retained633 receipt, then exercise its
                # replay and every subsequent native stage after additive DDL.
                history=p.store.scoring_policy_history(recovery.ROUND)
                assert history['revisions']
                frozen={key:row for key,row in original_snapshot(p)['rounds'].items() if row['status']!='open'}
                with p.conn.cursor() as q:
                    q.execute((SCRIPTS/'642-lab-arena-prospective-replacement-cutoff.sql').read_text())
                p.conn.commit();applied=True
                assert p.store.scoring_policy_history(recovery.ROUND)==history
                for key,row in frozen.items():
                    assert {k:v for k,v in p.store.get_round(key).items() if k!='replacement_freeze_seconds'}==row
        def original_fields(prepared):
            snapshot=original_snapshot(prepared)
            # The reused pre639 proof compares all old round fields byte for
            # byte; the new nullable metadata is asserted separately below.
            for row in snapshot['rounds'].values():
                row.pop('replacement_freeze_seconds',None)
                # Safe open rows receive the new metadata and its update time.
                if row['status']=='open':row.pop('updated_at',None)
            return snapshot
        monkeypatch.setattr(recovery,'apply',apply_with_cutoff)
        monkeypatch.setattr(recovery,'snapshot',original_fields)
        # Reuse the native571-row proof rather than copying its scoring flow.
        # It applies633 twice, preserves all original333 accepted rows/charges,
        # exercises100 synthetic claims/completions and publishes natively.
        recovery.test_exact_571_row_recovery_replay_and_native_merged_scoring(p)
        assert [r['generation_floor'] for r in p.store.scoring_policy_history(recovery.ROUND)['revisions']]==[0]
        assert p.store.get_round(recovery.ROUND)['replacement_freeze_seconds'] is None
    finally:p.store.close();p.conn.close()

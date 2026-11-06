"""Completed late-review receipts retain their public history and terminal cap."""
import pytest
from lab_arena.store import ArenaStoreError
from tests.lab_arena.nov06_late_review635_postgres_test import (
 database, context, apply, settle, M636, ROUND,
)
from tests.lab_arena.refusal_review_fallback599_postgres_test import begin, ALTERNATE
from tests.postgres_migration_harness import SCRIPTS


def test_completed635_receipts_and_terminal_rows_survive_cutoff_metadata(context):
    store,c,grant,targets,_,_=context
    apply(c,grant)
    for target in targets:settle(store,target['submission_id'])
    apply(c,M636.read_text())
    history=store.scoring_policy_history(ROUND)
    rows={target['submission_id']:store.get_submission(target['submission_id']) for target in targets}
    ledgers={sid:store.list_ledger(submission_id=sid) for sid in rows}
    with c.cursor() as q:
        q.execute('SELECT jsonb_agg(to_jsonb(h) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions h WHERE round_id=%s',(ROUND,));receipts=q.fetchone()[0]
    apply(c,(SCRIPTS/'639-lab-arena-prospective-replacement-cutoff.sql').read_text())
    assert store.scoring_policy_history(ROUND)==history
    apply(c,M636.read_text())
    with c.cursor() as q:
        q.execute('SELECT jsonb_agg(to_jsonb(h) ORDER BY generation_floor) FROM public.lab_arena_scoring_supersessions h WHERE round_id=%s',(ROUND,));assert q.fetchone()[0]==receipts
    for sid,row in rows.items():
        assert store.get_submission(sid)==row
        with pytest.raises(ArenaStoreError,match='submission_missing'):
            begin(store,sid,row['miner_hotkey'],ALTERNATE)
        assert store.list_ledger(submission_id=sid)==ledgers[sid]

-- Reuse validated559; keep both run fences and every outer billing predicate.
-- The round/provider/kind predicates already apply outside OFFSET0. Non-null
-- call identity follows from the required strict reservation/dispatch joins.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $active_deepline_filter_pushdown616$
DECLARE
  patch RECORD;
  target OID;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  index_definition TEXT;
  index_owner TEXT;
BEGIN
  SELECT pg_get_indexdef(i.indexrelid),pg_get_userbyid(c.relowner)
    INTO index_definition,index_owner
  FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
  JOIN pg_am am ON am.oid=c.relam
  WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_deepline_uncertain_round_entry_idx')
    AND i.indrelid='public.lab_arena_ledger'::regclass
    AND am.amname='btree' AND NOT i.indisunique
    AND i.indisvalid AND i.indisready;
  IF index_owner IS DISTINCT FROM 'lab_arena_owner' OR index_definition IS DISTINCT FROM
    'CREATE INDEX lab_arena_ledger_deepline_uncertain_round_entry_idx ON public.lab_arena_ledger USING btree (round_id, entry_id) INCLUDE (run_id, call_identity) WHERE ((provider = ''deepline''::text) AND (entry_kind = ''uncertain''::text) AND (call_identity IS NOT NULL))' THEN
    RAISE EXCEPTION 'Active Deepline round index absent, invalid, or wrong shape' USING ERRCODE='55000';
  END IF;

  FOR patch IN SELECT * FROM (VALUES
    ('public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
     'ed9585bada5629700c89f12a204a26df9e84460e6ec27f38da0817f335a2231e',
     'e5cf1d34296e2151ac6e298a680721b9758337c66ad01faee69359dc03fb8d8d',
     $old1616$WHERE entry.run_id = scope.run_id
              OFFSET 0$old1616$,
     $new1616$WHERE entry.run_id = scope.run_id
                AND entry.round_id = p_round_id
                AND entry.entry_kind = 'uncertain'
                AND entry.provider = 'deepline'
                AND entry.call_identity IS NOT NULL
              OFFSET 0$new1616$),
    ('public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)',
     '5b6056afa13230ba797a552b99644583a9345544cb8304422f5e671746e78ba6',
     'a29df6fda72d78d823de54abd621b8e694e2501d88dc782d41322db0a8e0dc95',
     $old2616$WHERE scoped.run_id = scope.run_id
              OFFSET 0$old2616$,
     $new2616$WHERE scoped.run_id = scope.run_id
                AND scoped.round_id = p_round_id
                AND scoped.entry_kind = 'uncertain'
                AND scoped.provider = 'deepline'
                AND scoped.call_identity IS NOT NULL
              OFFSET 0$new2616$)
  ) AS patches(signature,preimage,postimage,old_scope,new_scope) LOOP
    target:=to_regprocedure(patch.signature);
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
        p.provolatile,p.proconfig,l.lanname)
    INTO definition,metadata,identity
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF definition IS NULL OR identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
      RAISE EXCEPTION 'Active Deepline candidate security shape differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=patch.postimage THEN
      CONTINUE;
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>patch.preimage THEN
      RAISE EXCEPTION 'Active Deepline candidate preimage differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    IF length(definition)-length(replace(definition,patch.old_scope,''))<>length(patch.old_scope) THEN
      RAISE EXCEPTION 'Active Deepline candidate anchor differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    updated:=replace(definition,patch.old_scope,patch.new_scope);
    IF encode(extensions.digest(updated,'sha256'),'hex')<>patch.postimage THEN
      RAISE EXCEPTION 'Active Deepline candidate postimage differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>patch.postimage
       OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'Active Deepline candidate readback differs: %',patch.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$active_deepline_filter_pushdown616$;

NOTIFY pgrst,'reload schema';
COMMIT;

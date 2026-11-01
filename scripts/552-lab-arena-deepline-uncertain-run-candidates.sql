-- Use the validated partial run index for active Deepline uncertainty heads.
-- Preserve the existing run fence, full billing checks, cursor order and output.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $deepline_uncertain_run_candidates552$
DECLARE
  patch RECORD;
  target OID;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  index_definition TEXT;
BEGIN
  SELECT pg_get_indexdef(i.indexrelid)
    INTO index_definition
  FROM pg_index i
  JOIN pg_class c ON c.oid=i.indexrelid
  JOIN pg_am am ON am.oid=c.relam
  WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_deepline_uncertain_run_entry_idx')
    AND i.indrelid='public.lab_arena_ledger'::regclass
    AND am.amname='btree' AND NOT i.indisunique
    AND i.indisvalid AND i.indisready;
  IF index_definition IS DISTINCT FROM
    'CREATE INDEX lab_arena_ledger_deepline_uncertain_run_entry_idx ON public.lab_arena_ledger USING btree (run_id, entry_id) WHERE ((provider = ''deepline''::text) AND (entry_kind = ''uncertain''::text))' THEN
    RAISE EXCEPTION 'Deepline uncertainty index absent, invalid, or wrong shape' USING ERRCODE='55000';
  END IF;

  FOR patch IN SELECT * FROM (VALUES
    ('public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)',
     'ed9585bada5629700c89f12a204a26df9e84460e6ec27f38da0817f335a2231e',
     'bedaa8c553065504baec0f365c79963981e10130fc23a829e5ad42a6c45e793f',
     $v1_old$WHERE entry.run_id = scope.run_id
              OFFSET 0$v1_old$,
     $v1_new$WHERE entry.run_id = scope.run_id
                AND entry.entry_kind = 'uncertain'
                AND entry.provider = 'deepline'
              OFFSET 0$v1_new$,
     $v1_filter_old$              AND uncertainty.entry_kind = 'uncertain'
              AND uncertainty.provider = 'deepline'
$v1_filter_old$),
    ('public.lab_arena_list_deepline_cost_reconciliations_v2(text,text,bigint,integer,boolean)',
     '5b6056afa13230ba797a552b99644583a9345544cb8304422f5e671746e78ba6',
     '0f0a7e3635094e22140b627d1c352f81867f20a51ba5d24b74576e34ab9bc3a0',
     $v2_old$WHERE scoped.run_id = scope.run_id
              OFFSET 0$v2_old$,
     $v2_new$WHERE scoped.run_id = scope.run_id
                AND scoped.entry_kind = 'uncertain'
                AND scoped.provider = 'deepline'
              OFFSET 0$v2_new$,
     $v2_filter_old$              AND early.entry_kind = 'uncertain'
              AND early.provider = 'deepline'
$v2_filter_old$)
  ) AS patches(signature,preimage,postimage,old_scope,new_scope,old_filter) LOOP
    target:=to_regprocedure(patch.signature);
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
        p.provolatile,p.proconfig,l.lanname)
    INTO definition,metadata,identity
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
    IF definition IS NULL OR identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
      RAISE EXCEPTION 'Deepline candidate security shape differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')=patch.postimage THEN
      CONTINUE;
    END IF;
    IF encode(extensions.digest(definition,'sha256'),'hex')<>patch.preimage THEN
      RAISE EXCEPTION 'Deepline candidate preimage differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    IF length(definition)-length(replace(definition,patch.old_scope,''))<>length(patch.old_scope)
       OR length(definition)-length(replace(definition,patch.old_filter,''))<>length(patch.old_filter) THEN
      RAISE EXCEPTION 'Deepline candidate anchor differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    updated:=replace(replace(definition,patch.old_scope,patch.new_scope),patch.old_filter,'');
    IF encode(extensions.digest(updated,'sha256'),'hex')<>patch.postimage THEN
      RAISE EXCEPTION 'Deepline candidate postimage differs: %',patch.signature USING ERRCODE='55000';
    END IF;
    EXECUTE updated;
    IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>patch.postimage
       OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
      RAISE EXCEPTION 'Deepline candidate readback differs: %',patch.signature USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$deepline_uncertain_run_candidates552$;

NOTIFY pgrst,'reload schema';
COMMIT;

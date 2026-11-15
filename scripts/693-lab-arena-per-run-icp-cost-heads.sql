-- Read latest per-ICP cost heads through the existing per-run ledger index.
-- Preserve the exact discovered run set, latest-head ordering, cost expressions,
-- nulls, errors, infrastructure/supersession exclusions and all function security.
-- OFFSET 0 keeps this read parameterized by the already selected run.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $cost_heads693$
DECLARE
  target OID:=to_regprocedure('public.lab_arena__successful_icp_cost_state(text,text,integer)');
  definition TEXT;
  metadata JSONB;
  identity JSONB;
  source_hash TEXT;
  old_heads TEXT:=$old693$    FROM public.lab_arena_ledger AS ledger
    JOIN matching_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.round_id = p_round_id
      AND ledger.submission_id = p_submission_id
      AND ledger.call_identity IS NOT NULL$old693$;
  new_heads TEXT:=$new693$    FROM matching_runs AS runs
    CROSS JOIN LATERAL (
      SELECT found.* FROM public.lab_arena_ledger AS found
      WHERE found.submission_id = p_submission_id AND found.run_id = runs.run_id
        AND found.round_id = p_round_id AND found.call_identity IS NOT NULL
      OFFSET 0
    ) AS ledger$new693$;
BEGIN
  IF NOT EXISTS(SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
    WHERE i.indexrelid=to_regclass('public.lab_arena_ledger_cost_run_idx')
      AND i.indrelid=to_regclass('public.lab_arena_ledger')
      AND i.indisvalid AND i.indisready
      AND pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND pg_get_indexdef(i.indexrelid)=
        'CREATE INDEX lab_arena_ledger_cost_run_idx ON public.lab_arena_ledger USING btree (submission_id, run_id) WHERE (call_identity IS NOT NULL)') THEN
    RAISE EXCEPTION 'Cost heads693 index differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname),
    encode(pg_catalog.sha256(pg_catalog.convert_to(p.prosrc,'UTF8')),'hex')
  INTO definition,metadata,identity,source_hash
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
    ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Cost heads693 security differs' USING ERRCODE='55000';
  END IF;
  IF source_hash='afa7867e295ef3ef9ded3b52633513c56c6ed69c9d5270bafff985a157567ec2' THEN RETURN; END IF;
  IF source_hash IS DISTINCT FROM '03163f8ece8c6755df514f9ef1e6dcbbcd5ce54d01dfe2fc76cf5c12aef462be' THEN
    RAISE EXCEPTION 'Cost heads693 preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(definition,old_heads,new_heads);
  IF (SELECT encode(pg_catalog.sha256(pg_catalog.convert_to(p.prosrc,'UTF8')),'hex')
      FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM 'afa7867e295ef3ef9ded3b52633513c56c6ed69c9d5270bafff985a157567ec2'
    OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
    RAISE EXCEPTION 'Cost heads693 readback differs' USING ERRCODE='55000';
  END IF;
END;
$cost_heads693$;
COMMIT;

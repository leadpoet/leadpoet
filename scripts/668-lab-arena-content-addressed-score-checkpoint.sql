-- Bind legacy partial-score receipts to either their original run key or the
-- new digest-suffixed key used after a failed terminal write.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- The hosted migration role cannot replace a function owned by the restricted
-- helper owner until that owner has temporary CREATE authority on public.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $score_checkpoint_ref$
DECLARE
  signature CONSTANT TEXT :=
    'public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)';
  old_gate CONSTANT TEXT := $old$AND p_output_ref = 'arena/' || p_run.round_id || '/scores/items/' || p_run.run_id || '.json'$old$;
  new_gate CONSTANT TEXT := $new$AND (
      p_output_ref = 'arena/' || p_run.round_id || '/scores/items/' || p_run.run_id || '.json'
      OR p_output_ref = 'arena/' || p_run.round_id || '/scores/items/' || p_run.run_id
        || '-' || pg_catalog.substr(p_receipt ->> 'source_output_hash', 8) || '.json'
    )$new$;
  definition TEXT;
  updated TEXT;
  prior_metadata JSONB;
  metadata JSONB;
  identity JSONB;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
      p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO definition,prior_metadata,identity
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::REGPROCEDURE;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
    RAISE EXCEPTION 'legacy partial checkpoint security shape differs';
  END IF;
  IF pg_catalog.strpos(definition,new_gate) > 0 THEN
    RETURN;
  END IF;
  IF (pg_catalog.length(definition)-pg_catalog.length(
        pg_catalog.replace(definition,old_gate,''))) <> pg_catalog.length(old_gate) THEN
    RAISE EXCEPTION 'legacy partial checkpoint output reference preimage differs';
  END IF;
  updated := pg_catalog.replace(definition,old_gate,new_gate);
  EXECUTE updated;
  SELECT pg_catalog.pg_get_functiondef(p.oid), pg_catalog.to_jsonb(p)-'prosrc',
    pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
      p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_language l ON l.oid=p.prolang
  WHERE p.oid=signature::REGPROCEDURE;
  IF metadata IS DISTINCT FROM prior_metadata
     OR identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB
     OR pg_catalog.strpos(definition,new_gate)=0
     OR pg_catalog.strpos(definition,old_gate)>0 THEN
    RAISE EXCEPTION 'legacy partial checkpoint output reference readback differs';
  END IF;
END;
$score_checkpoint_ref$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

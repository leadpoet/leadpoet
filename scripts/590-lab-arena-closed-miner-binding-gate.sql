-- Skip the unchanged pure binding proof when its required execution key fails.
-- Retain the original key predicate outside CASE and every full billing proof.
-- No ledger, caller, settlement, score, data, owner or privilege change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $closed_miner_binding_gate590$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)';
  definition TEXT; updated TEXT; metadata_before JSONB; metadata_after JSONB;
  dependency RECORD; actual_security JSONB;
  old_gate CONSTANT TEXT := $old_gate$      AND CASE WHEN uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB$old_gate$;
  new_gate CONSTANT TEXT := $new_gate$      AND CASE WHEN uncertainty.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
          AND reservation.entry_doc ->> 'deepline_execution_key' =
              'arena:' || pg_catalog.substr(uncertainty.call_identity, 8)$new_gate$;
BEGIN
  -- Check dependencies on replay too. The skipped proof is pure, and the
  -- exact current selector retains all source, latest-head and cursor guards.
  FOR dependency IN SELECT * FROM (VALUES
    ('public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',
     '8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49',
     '{lab_arena_owner=X/lab_arena_owner}',FALSE,'i'),
    ('public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)',
     'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89',
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'s')
  ) AS required(signature,sha256,acl,security_definer,volatility) LOOP
    SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
      INTO actual_security
    FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
      JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(dependency.signature);
    IF actual_security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',dependency.acl,
        dependency.security_definer,dependency.volatility,ARRAY['search_path=pg_catalog, public'],'sql')
       OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(dependency.signature)),
         'sha256'),'hex') IS DISTINCT FROM dependency.sha256 THEN
      RAISE EXCEPTION 'Closed miner binding gate dependency differs: %',dependency.signature;
    END IF;
  END LOOP;
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
    INTO definition,metadata_before,actual_security
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
    JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(signature);
  IF definition IS NULL OR actual_security IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
      ARRAY['search_path=pg_catalog, public'],'sql') THEN
    RAISE EXCEPTION 'Closed miner binding gate security preimage differs';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')=
      'ff3aaddd632b24f70651c42530eae854a522927c291540bf34faa46f8d23597a' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM
      '67fcd9dc83d29491aa4608e069b6680ac652e85b93b1feac64cc01a72b1f93f8'
     OR (length(definition)-length(replace(definition,old_gate,'')))<>length(old_gate) THEN
    RAISE EXCEPTION 'Closed miner binding gate exact preimage differs';
  END IF;
  updated:=replace(definition,old_gate,new_gate);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM
      'ff3aaddd632b24f70651c42530eae854a522927c291540bf34faa46f8d23597a' THEN
    RAISE EXCEPTION 'Closed miner binding gate target differs';
  END IF;
  EXECUTE updated;
  SELECT to_jsonb(p)-'prosrc' INTO metadata_after FROM pg_proc p WHERE p.oid=to_regprocedure(signature);
  IF metadata_after IS DISTINCT FROM metadata_before
     OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(signature)),'sha256'),'hex')
       IS DISTINCT FROM 'ff3aaddd632b24f70651c42530eae854a522927c291540bf34faa46f8d23597a' THEN
    RAISE EXCEPTION 'Closed miner binding gate readback differs';
  END IF;
END;
$closed_miner_binding_gate590$;
COMMIT;

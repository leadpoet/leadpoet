-- Skip only impossible full-proof lanes after native ordered candidates.
-- Keep 573 materialized CTEs, 578 wrapper, all four proof helpers and all data.
-- Scalar metadata is fetched by primary key only after the ordered OFFSET 0.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $closed_deepline_proof_guards617$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_next_closed_deepline_reconciliation_v1(text,text,integer,text,bigint)';
  target OID;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  prerequisite RECORD;
  index_definition TEXT;
  index_owner TEXT;
BEGIN
  -- Check unchanged source and security prerequisites before accepting replay.
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena_next_closed_provider_reconciliation_v1(text,text,integer,text,bigint)',
     '91a5e8ebb464aaf2252a34f4b80e00411fdb0decac2f267d25c099c9bf2d9d17',
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',TRUE,'s'),
    ('public.lab_arena__closed_host_score_success_uncertainty_v1(bigint)',
     'e347886710c2461366607941044af1123d092622cfb65ecae1c5b70a1a22f914',
     '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s'),
    ('public.lab_arena__closed_miner_score_success_uncertainty_v1(bigint)',
     'ff3aaddd632b24f70651c42530eae854a522927c291540bf34faa46f8d23597a',
     '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s'),
    ('public.lab_arena__closed_score_adaptation_uncertainty_v1(bigint)',
     '0b876ab01b0c9e88ca9dc4a1cfcd3eb9e8cc13dc453a2bd418593e5dd3a96f82',
     '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s'),
    ('public.lab_arena__closed_score_dynamic_uncertainty_v1(bigint)',
     'd5885be424024b95f7aac98b175872857baa6b0856e407ee9108b2382581ff0d',
     '{lab_arena_owner=X/lab_arena_owner}',TRUE,'s'),
    ('public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)',
     '8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49',
     '{lab_arena_owner=X/lab_arena_owner}',FALSE,'i')
  ) AS required(signature,sha256,acl,security_definer,volatility) LOOP
    SELECT pg_get_functiondef(p.oid),
      jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
        p.provolatile,p.proconfig,l.lanname)
    INTO definition,identity
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
    WHERE p.oid=to_regprocedure(prerequisite.signature);
    IF identity IS DISTINCT FROM jsonb_build_array('lab_arena_owner',prerequisite.acl,
        prerequisite.security_definer,prerequisite.volatility,
        ARRAY['search_path=pg_catalog, public'],'sql')
       OR encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex')
          IS DISTINCT FROM prerequisite.sha256 THEN
      RAISE EXCEPTION 'Closed Deepline proof guards dependency differs: %',prerequisite.signature USING ERRCODE='55000';
    END IF;
  END LOOP;

  -- These existing indexes preserve the reviewed narrow candidate access paths.
  FOR prerequisite IN SELECT * FROM (VALUES
    ('public.lab_arena_ledger_deepline_uncertain_round_entry_idx','public.lab_arena_ledger',
     'CREATE INDEX lab_arena_ledger_deepline_uncertain_round_entry_idx ON public.lab_arena_ledger USING btree (round_id, entry_id) INCLUDE (run_id, call_identity) WHERE ((provider = ''deepline''::text) AND (entry_kind = ''uncertain''::text) AND (call_identity IS NOT NULL))'),
    ('public.lab_arena_runs_closed_score_covering_idx','public.lab_arena_runs',
     'CREATE INDEX lab_arena_runs_closed_score_covering_idx ON public.lab_arena_runs USING btree (round_id, run_id) INCLUDE (status, terminal_cause, submission_id, miner_hotkey, stage) WHERE ((kind = ''score''::text) AND (status = ANY (ARRAY[''accepted''::text, ''failed''::text])))')
  ) AS required(name,relation,definition) LOOP
    SELECT pg_get_indexdef(i.indexrelid),pg_get_userbyid(c.relowner)
      INTO index_definition,index_owner
    FROM pg_index i JOIN pg_class c ON c.oid=i.indexrelid
    JOIN pg_am am ON am.oid=c.relam
    WHERE i.indexrelid=to_regclass(prerequisite.name)
      AND i.indrelid=to_regclass(prerequisite.relation)
      AND am.amname='btree' AND NOT i.indisunique
      AND i.indisvalid AND i.indisready;
    IF index_owner IS DISTINCT FROM 'lab_arena_owner'
       OR index_definition IS DISTINCT FROM prerequisite.definition THEN
      RAISE EXCEPTION 'Closed Deepline proof guards index absent, invalid, or wrong shape: %',prerequisite.name USING ERRCODE='55000';
    END IF;
  END LOOP;

  target:=to_regprocedure(signature);
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"sql"]'::JSONB THEN
    RAISE EXCEPTION 'Closed Deepline proof guards selector security differs' USING ERRCODE='55000';
  END IF;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex')=
      '96edb88529b6a4e42077d638a70069adf59d64de52ffe1c6decaee36887ec557' THEN RETURN; END IF;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(definition,'UTF8')),'hex') IS DISTINCT FROM
      'ce8691e166d24a87ff6bbf475dca8cc0eb526b6688f3af22fad6ad8b0a8b6c89' THEN
    RAISE EXCEPTION 'Closed Deepline proof guards selector preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,$old_lane617$    ) AS uncertainty
    WHERE (
      public.lab_arena__closed_score_dynamic_uncertainty_v1(uncertainty.entry_id)
      OR public.lab_arena__closed_host_score_success_uncertainty_v1(uncertainty.entry_id)
      OR public.lab_arena__closed_miner_score_success_uncertainty_v1(uncertainty.entry_id)
      OR public.lab_arena__closed_score_adaptation_uncertainty_v1(uncertainty.entry_id)
    )
$old_lane617$,$new_lane617$    ) AS uncertainty
    CROSS JOIN LATERAL (
      SELECT metadata.funding_source,
      metadata.entry_doc #> '{call,call_succeeded}' = 'false'::jsonb AS failed_call,
      (metadata.entry_doc ->> 'reason' = 'worker_reported'
       AND metadata.entry_doc #>> '{call,reason}' = 'missing_provider_cost'
       AND metadata.entry_doc #> '{call,call_succeeded}' = 'true'::jsonb
       AND metadata.entry_doc #> '{call,provider_status}' = '200'::jsonb) AS success_call,
      (metadata.entry_doc ->> 'reason' = 'worker_reported'
       AND metadata.entry_doc #>> '{call,reason}' = 'settle_failure'
       AND metadata.entry_doc #>> '{call,failure_stage}' = 'response_adaptation'
       AND metadata.entry_doc #>> '{call,error_class}' = 'CompatibilityResponseError') AS adaptation_details
      FROM public.lab_arena_ledger AS metadata
      WHERE metadata.entry_id = uncertainty.entry_id
      OFFSET 0
    ) AS flags
    WHERE (
      CASE WHEN (flags.funding_source = 'miner_key' AND flags.failed_call) IS TRUE THEN public.lab_arena__closed_score_dynamic_uncertainty_v1(uncertainty.entry_id) ELSE FALSE END
      OR CASE WHEN (flags.funding_source = 'host' AND flags.success_call) IS TRUE THEN public.lab_arena__closed_host_score_success_uncertainty_v1(uncertainty.entry_id) ELSE FALSE END
      OR CASE WHEN (flags.funding_source = 'miner_key' AND flags.success_call) IS TRUE THEN public.lab_arena__closed_miner_score_success_uncertainty_v1(uncertainty.entry_id) ELSE FALSE END
      OR CASE WHEN (flags.funding_source IN ('host','miner_key') AND flags.failed_call AND flags.adaptation_details) IS TRUE THEN public.lab_arena__closed_score_adaptation_uncertainty_v1(uncertainty.entry_id) ELSE FALSE END
    )
$new_lane617$);
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(updated,'UTF8')),'hex') IS DISTINCT FROM
      '96edb88529b6a4e42077d638a70069adf59d64de52ffe1c6decaee36887ec557' THEN
    RAISE EXCEPTION 'Closed Deepline proof guards selector postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF encode(pg_catalog.sha256(pg_catalog.convert_to(pg_get_functiondef(target),'UTF8')),'hex')
       IS DISTINCT FROM '96edb88529b6a4e42077d638a70069adf59d64de52ffe1c6decaee36887ec557'
     OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
    RAISE EXCEPTION 'Closed Deepline proof guards selector readback differs' USING ERRCODE='55000';
  END IF;
END;
$closed_deepline_proof_guards617$;

NOTIFY pgrst,'reload schema';
COMMIT;

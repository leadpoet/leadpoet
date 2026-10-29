-- A frozen legacy champion is a measured comparison, not a training pass.
-- Keep publication cost/score authority and every existing reward predicate.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $legacy_baseline_reward591$
DECLARE
  v_signature REGPROCEDURE := 'public.lab_arena_reward_slot_snapshot(text,jsonb)'::REGPROCEDURE;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_schema_acl ACLITEM[];
  v_hash TEXT;
  v_old_candidate TEXT := $old$      r.publication_doc
    FROM public.lab_arena_rounds AS r$old$;
  v_new_candidate TEXT := $new$      r.publication_doc,
      (r.configuration_doc -> 'baseline_reasoning_configuration' = '{}'::JSONB
        AND r.configuration_doc ->> 'reasoning_capture_policy' = 'full_generated_thinking_v1'
        AND r.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'
        AND r.configuration_doc ->> 'sourcing_cost_eligibility_policy'
          IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
      ) AS frozen_legacy_baseline_training_exempt
    FROM public.lab_arena_rounds AS r$new$;
  v_old_baseline TEXT := $old$      AND baseline.entry -> 'eligible' = 'true'::JSONB$old$;
  v_new_baseline TEXT := $new$      AND (baseline.entry -> 'eligible' = 'true'::JSONB
        OR (c.frozen_legacy_baseline_training_exempt IS TRUE
          AND baseline.entry -> 'eligible' = 'false'::JSONB
          AND baseline.entry ->> 'eligibility_reason' = 'training_data_incomplete'
          AND baseline.entry ->> 'submission_id' =
            'baseline-' || pg_catalog.regexp_replace(c.round_id, '^arena-', '')
          AND baseline.entry -> 'training_data_eligibility' =
            '{"eligible":false,"reasons":["baseline_reasoning_not_required"],"exemption":"champion_baseline"}'::JSONB)
      )$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
    INTO v_definition, v_identity
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid = p.prolang
    WHERE p.oid = v_signature;
  SELECT nspacl INTO v_schema_acl FROM pg_catalog.pg_namespace WHERE nspname = 'public';
  IF v_identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION '591 reward slot security differs' USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '670235bc84b77ebc11ca69be3729a1bdb430bb5119a32021db862e90f453f499' THEN
    RETURN;
  END IF;
  IF v_hash <> '78bf61278a796e0c7b06c025148bee5e0ad9bc8d249b27b2528ae337684bba33' THEN
    RAISE EXCEPTION '591 reward slot definition differs' USING ERRCODE = '55000';
  END IF;
  v_updated := pg_catalog.replace(v_definition, v_old_candidate, v_new_candidate);
  v_updated := pg_catalog.replace(v_updated, v_old_baseline, v_new_baseline);
  IF pg_catalog.encode(extensions.digest(v_updated, 'sha256'), 'hex') <> '670235bc84b77ebc11ca69be3729a1bdb430bb5119a32021db862e90f453f499' THEN
    RAISE EXCEPTION '591 reward slot target differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
    INTO v_after_identity
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid = p.prolang
    WHERE p.oid = v_signature;
  IF v_after_identity IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated
     OR (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
        IS DISTINCT FROM v_schema_acl THEN
    RAISE EXCEPTION '591 reward slot readback differs' USING ERRCODE = '55000';
  END IF;
END;
$legacy_baseline_reward591$;

NOTIFY pgrst, 'reload schema';
COMMIT;

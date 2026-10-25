-- Publish measured scores when a full-thinking contender lacks training evidence.
-- The frozen policy, cost authority, and scoring authority remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $training_publication_563$
DECLARE
  v_signature REGPROCEDURE := pg_catalog.to_regprocedure(
    'public.lab_arena__per_icp_publication_valid(text,jsonb)');
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_old TEXT := $old$  END LOOP;
  -- lab_arena_per_icp_null_final_score_v1:$old$;
  v_new TEXT := $new$  END LOOP;
  -- Only frozen full-thinking rounds carry training eligibility. The service
  -- derives this report from source review and selected execution exports.
  SELECT configuration_doc ->> 'reasoning_capture_policy'
    INTO v_reasoning_policy
    FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF v_reasoning_policy = 'full_generated_thinking_v1' THEN
    IF pg_catalog.jsonb_typeof(v_training -> 'eligible') IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(v_training -> 'reasons') IS DISTINCT FROM 'array'
       OR (v_training ->> 'eligible')::BOOLEAN IS DISTINCT FROM
          (pg_catalog.jsonb_array_length(v_training -> 'reasons') = 0)
       OR ((v_training ->> 'eligible')::BOOLEAN AND (
         pg_catalog.jsonb_typeof(v_training -> 'source_review') IS DISTINCT FROM 'object'
         OR pg_catalog.jsonb_typeof(v_training -> 'capture') IS DISTINCT FROM 'object'
         OR v_training -> 'source_review' ->> 'eligible' IS DISTINCT FROM 'true'
         OR v_training -> 'capture' ->> 'eligible' IS DISTINCT FROM 'true'
       )) THEN
      RAISE EXCEPTION 'lab_arena_publication_training_report_mismatch'
        USING ERRCODE = '22023';
    END IF;
    IF v_hard_reason IS NULL AND NOT (v_training ->> 'eligible')::BOOLEAN THEN
      v_hard_reason := 'training_data_incomplete';
    END IF;
  END IF;
  -- lab_arena_per_icp_null_final_score_v1:$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
    INTO v_definition, v_identity
    FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid = p.prolang
    WHERE p.oid = v_signature;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Arena publication guard security shape differs'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(v_definition, '-- lab_arena_training_publication_563') > 0 THEN
    IF pg_catalog.strpos(v_definition, v_new) = 0
       OR pg_catalog.strpos(v_definition, v_old) > 0 THEN
      RAISE EXCEPTION 'Arena training publication patch differs'
        USING ERRCODE = '55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       IS DISTINCT FROM '1ed2b288d7bc614088c05b245cb8ee43a3c0e8eea3281de595a8694871e266ad'
     OR (pg_catalog.length(v_definition) - pg_catalog.length(
         pg_catalog.replace(v_definition, v_old, ''))) <> pg_catalog.length(v_old)
     OR (pg_catalog.length(v_definition) - pg_catalog.length(
         pg_catalog.replace(v_definition, '  v_hard_reason TEXT;', '')))
         <> pg_catalog.length('  v_hard_reason TEXT;') THEN
    RAISE EXCEPTION 'Arena publication guard exact preimage differs'
      USING ERRCODE = '55000';
  END IF;
  v_updated := pg_catalog.replace(v_definition,
    '  v_hard_reason TEXT;',
    E'  v_hard_reason TEXT;\n  v_training JSONB := p_ranking -> ''training_data_eligibility'';\n  v_reasoning_policy TEXT;\n  -- lab_arena_training_publication_563');
  v_updated := pg_catalog.replace(v_updated, v_old, v_new);
  EXECUTE v_updated;
  SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
           p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
    INTO v_after_identity FROM pg_catalog.pg_proc p
    JOIN pg_catalog.pg_language l ON l.oid = p.prolang WHERE p.oid = v_signature;
  IF v_after_identity IS DISTINCT FROM v_identity
     OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated THEN
    RAISE EXCEPTION 'Arena publication guard readback differs'
      USING ERRCODE = '55000';
  END IF;
END;
$training_publication_563$;

NOTIFY pgrst, 'reload schema';
COMMIT;

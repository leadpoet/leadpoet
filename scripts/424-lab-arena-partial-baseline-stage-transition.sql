-- A proof-backed incomplete baseline can finish stage-one scoring and let
-- challengers execute. The final publication still yields no new king.
-- Keep the old complete-score guard for every other configuration or cause.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $partial_baseline_transition_424$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_transition_round(text,text,text,jsonb)';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old TEXT := $old$    IF v_missing_scores <> 0 THEN
      RAISE EXCEPTION 'lab_arena_stage1_baseline_scores_incomplete' USING ERRCODE = '22023';
    END IF;$old$;
  v_new TEXT := $new$    -- lab_arena_partial_baseline_transition_424: a missing baseline
    -- score is allowed only when the publication proof verifies every missing
    -- ICP against a failed, deadline-closed execution or judge assignment.
    IF v_missing_scores <> 0 AND NOT (
      v_missing_scores = 1
      AND v_baseline_count = 1
      AND v_round.configuration_doc ->> 'execution_sequence_policy'
            = 'baseline_scored_first_v1'
      AND v_round.configuration_doc ->> 'integrity_policy'
            = 'arena_integrity_v1'
      AND v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
            = 'successful_calls_per_icp_v1'
      AND EXISTS (
        SELECT 1 FROM pg_catalog.jsonb_array_elements(v_round.participants)
          AS baseline
        WHERE COALESCE((baseline ->> 'is_king')::BOOLEAN, FALSE)
          AND public.lab_arena__publication_execution_incomplete_v1(
            p_round_id, baseline ->> 'submission_id'
          )
      )
    ) THEN
      RAISE EXCEPTION 'lab_arena_stage1_baseline_scores_incomplete' USING ERRCODE = '22023';
    END IF;$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner
    ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public',
                     'statement_timeout=60s']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena partial baseline transition security shape differs';
  END IF;
  IF pg_catalog.to_regprocedure(
      'public.lab_arena__publication_execution_incomplete_v1(text,text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'Arena partial baseline transition requires migration 423';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash='6091b7159caaada4d14758f11304150d9e602a121aaf7a39e2820a07140b9521' THEN RETURN; END IF;
  IF v_hash<>'79145c6f31a0ae6502b0d5165e15b19dbd9c137d63e97b841bc917b46a2bf144' THEN
    RAISE EXCEPTION 'Arena partial baseline transition preimage differs';
  END IF;
  IF pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old,'')) <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena partial baseline transition fragment differs';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>'6091b7159caaada4d14758f11304150d9e602a121aaf7a39e2820a07140b9521' THEN
    RAISE EXCEPTION 'Arena partial baseline transition postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    v_signature::REGPROCEDURE),'sha256'),'hex')<>'6091b7159caaada4d14758f11304150d9e602a121aaf7a39e2820a07140b9521' THEN
    RAISE EXCEPTION 'Arena partial baseline transition readback differs';
  END IF;
END;
$partial_baseline_transition_424$;
COMMIT;

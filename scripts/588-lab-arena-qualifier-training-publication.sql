-- Full training evidence is required for every contender that can meet the
-- frozen promotion margin. Other rows retain an explicit unknown report.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $qualifier_publication_588$
DECLARE
  v_helper REGPROCEDURE := 'public.lab_arena__per_icp_publication_valid(text,jsonb)'::REGPROCEDURE;
  v_guard REGPROCEDURE := 'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE;
  v_signature REGPROCEDURE;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_after_identity JSONB;
  v_expected_identity JSONB;
  v_preimage TEXT;
  v_postimage TEXT;
  v_old TEXT;
  v_new TEXT;
  v_old_tail TEXT;
  v_new_tail TEXT;
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[v_helper, v_guard] LOOP
    SELECT pg_catalog.pg_get_functiondef(p.oid),
           pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
             p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      INTO v_definition, v_identity
      FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = v_signature;
    IF v_signature = v_helper THEN
      v_expected_identity :=
        '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB;
      v_preimage := 'f57a259ad59c9bb40b3597c485125f82f16939fcb52ee799318e5f8ddc1c28de';
      v_postimage := '0de3baf2d97622fd821c5959570a1b680f4c7ca0a565d3fe0b41f975c8c048e4';
      v_old := $old$  IF v_reasoning_policy = 'full_generated_thinking_v1' THEN
    IF pg_catalog.jsonb_typeof(v_training -> 'eligible')$old$;
      v_new := $new$  IF v_reasoning_policy = 'full_generated_thinking_v1' THEN
    -- lab_arena_qualifier_training_publication_588
    IF v_training IS DISTINCT FROM
       '{"status":"not_assessed","eligible":null,"reasons":[]}'::JSONB THEN
      IF pg_catalog.jsonb_typeof(v_training -> 'eligible')$new$;
      v_old_tail := $old$    IF v_hard_reason IS NULL AND NOT (v_training ->> 'eligible')::BOOLEAN THEN
      v_hard_reason := 'training_data_incomplete';
    END IF;
  END IF;
  -- lab_arena_per_icp_null_final_score_v1:$old$;
      v_new_tail := $new$    IF v_hard_reason IS NULL AND NOT (v_training ->> 'eligible')::BOOLEAN THEN
      v_hard_reason := 'training_data_incomplete';
    END IF;
    END IF;
  END IF;
  -- lab_arena_per_icp_null_final_score_v1:$new$;
    END IF;
    IF v_signature = v_guard THEN
      v_expected_identity :=
        '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}",false,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB;
      v_preimage := '0897b15b52a28f5ccaf08523c38c92c19fa24dbb056796f867a79e71524c9bb3';
      v_postimage := 'da7b69efcc26abf82a9c4e187e0445554134b7dc744bf8fe33788763d8bfcbff';
      v_old := $old$  v_decision := NEW.publication_doc -> 'king_decision';$old$;
      v_new := $new$  -- lab_arena_qualifier_training_publication_588
  -- The row validator has already checked every measured score and cost
  -- report. An unassessed row may not cross the same frozen promotion margin
  -- used by the winner query below.
  IF EXISTS (
    SELECT 1
    FROM pg_catalog.jsonb_array_elements(NEW.publication_doc -> 'final_ranking') AS ranking
    WHERE ranking -> 'training_data_eligibility' =
      '{"status":"not_assessed","eligible":null,"reasons":[]}'::JSONB
      AND (
        NEW.configuration_doc ->> 'reasoning_capture_policy'
          IS DISTINCT FROM 'full_generated_thinking_v1'
        OR COALESCE(NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy', '')
          NOT IN ('successful_calls_per_icp_v1', 'successful_calls_flat_per_icp_v1')
        OR (COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
            AND NEW.publication_doc #>> '{king_decision,outcome}' = 'crowned')
        OR (
          NOT COALESCE((ranking ->> 'is_baseline')::BOOLEAN, FALSE)
          AND (ranking ->> 'eligible')::BOOLEAN IS TRUE
          AND pg_catalog.jsonb_typeof(ranking -> 'final_score') = 'number'
          AND v_baseline_score IS NOT NULL
          AND (ranking ->> 'final_score')::NUMERIC >=
                v_baseline_score + v_promotion_margin
        )
      )
  ) THEN
    RAISE EXCEPTION 'lab_arena_publication_training_report_mismatch'
      USING ERRCODE = '22023';
  END IF;

  v_decision := NEW.publication_doc -> 'king_decision';$new$;
    END IF;
    IF v_definition IS NULL OR v_identity IS DISTINCT FROM v_expected_identity THEN
      RAISE EXCEPTION 'Arena qualifier publication security shape differs'
        USING ERRCODE = '55000';
    END IF;
    IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex') = v_postimage THEN
      CONTINUE;
    END IF;
    IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
         IS DISTINCT FROM v_preimage
       OR (pg_catalog.length(v_definition) - pg_catalog.length(
           pg_catalog.replace(v_definition, v_old, ''))) <> pg_catalog.length(v_old) THEN
      RAISE EXCEPTION 'Arena qualifier publication exact preimage differs'
        USING ERRCODE = '55000';
    END IF;
    v_updated := pg_catalog.replace(v_definition, v_old, v_new);
    IF v_signature = v_helper THEN
      IF (pg_catalog.length(v_updated) - pg_catalog.length(
          pg_catalog.replace(v_updated, v_old_tail, ''))) <> pg_catalog.length(v_old_tail) THEN
        RAISE EXCEPTION 'Arena qualifier publication helper anchor differs'
          USING ERRCODE = '55000';
      END IF;
      v_updated := pg_catalog.replace(v_updated, v_old_tail, v_new_tail);
    END IF;
    EXECUTE v_updated;
    SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
             p.proacl::TEXT, p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      INTO v_after_identity FROM pg_catalog.pg_proc p
      JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = v_signature;
    IF v_after_identity IS DISTINCT FROM v_identity
       OR pg_catalog.pg_get_functiondef(v_signature) IS DISTINCT FROM v_updated
       OR pg_catalog.encode(extensions.digest(
            pg_catalog.pg_get_functiondef(v_signature), 'sha256'), 'hex')
            IS DISTINCT FROM v_postimage THEN
      RAISE EXCEPTION 'Arena qualifier publication readback differs'
        USING ERRCODE = '55000';
    END IF;
  END LOOP;
END;
$qualifier_publication_588$;

NOTIFY pgrst, 'reload schema';
COMMIT;

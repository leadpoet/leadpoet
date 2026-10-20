-- Skip expensive claim proofs only when the same predicate already admits.
-- Preserve every lock, cutoff, funding/cache/admission rule and existing row.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $claim_eligibility_fast_paths509$
DECLARE
  target OID:=to_regprocedure('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  patch RECORD;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Claim eligibility fast paths security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='6d9bdb660a476838c42bee9dd9de5588be9f19f41ee32ea167b1f47bcafbf2ef' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'aa00d7f981f0c255d85e84d650b6986ed0d400feccebacb1fc9aa702192c3b52' THEN
    RAISE EXCEPTION 'Claim eligibility fast paths preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=definition;
  FOR patch IN SELECT * FROM (VALUES
    ($credit_old$    AND (NOT public.lab_arena_miner_credit_failure_v1(runs.run_id)
      OR (runs.attempt <= (v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
        AND clock_timestamp() < (v_round.configuration_doc#>>ARRAY['schedule',
          CASE runs.stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ))
$credit_old$,$credit_new$    -- Avoid credit-history proof when its existing allowance already admits.
    AND CASE WHEN (runs.attempt <= (v_round.configuration_doc->>'max_attempts_per_assignment')::INTEGER
        AND clock_timestamp() < (v_round.configuration_doc#>>ARRAY['schedule',
          CASE runs.stage WHEN 1 THEN 'stage_1_scoring_close' ELSE 'final_scoring_close' END])::TIMESTAMPTZ) THEN TRUE
      ELSE NOT public.lab_arena_miner_credit_failure_v1(runs.run_id) END
$credit_new$),
    ($window_old$    AND (
      runs.kind <> 'score'
      OR public.lab_arena__judge_continuation_v1(v_round,runs)
      OR pg_catalog.clock_timestamp() <
           (v_round.configuration_doc #>> ARRAY[
             'schedule', CASE v_stage WHEN 1 THEN 'stage_1_scoring_close'
               ELSE 'final_scoring_close' END
           ])::TIMESTAMPTZ
    )
$window_old$,$window_new$    -- Ordinary admission before cutoff needs no unfinished-judge proof.
    AND CASE WHEN (runs.kind <> 'score' OR pg_catalog.clock_timestamp() <
           (v_round.configuration_doc #>> ARRAY[
             'schedule', CASE v_stage WHEN 1 THEN 'stage_1_scoring_close'
               ELSE 'final_scoring_close' END
           ])::TIMESTAMPTZ
    ) THEN TRUE
      ELSE public.lab_arena__judge_continuation_v1(v_round,runs) END
$window_new$)
  ) AS patches(old_text,new_text) LOOP
    IF length(updated)-length(replace(updated,patch.old_text,''))<>length(patch.old_text) THEN
      RAISE EXCEPTION 'Claim eligibility fast paths anchor differs' USING ERRCODE='55000';
    END IF;
    updated:=replace(updated,patch.old_text,patch.new_text);
  END LOOP;
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'6d9bdb660a476838c42bee9dd9de5588be9f19f41ee32ea167b1f47bcafbf2ef' THEN
    RAISE EXCEPTION 'Claim eligibility fast paths postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'6d9bdb660a476838c42bee9dd9de5588be9f19f41ee32ea167b1f47bcafbf2ef'
    OR (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata THEN
    RAISE EXCEPTION 'Claim eligibility fast paths readback differs' USING ERRCODE='55000';
  END IF;
END;
$claim_eligibility_fast_paths509$;
NOTIFY pgrst,'reload schema';
COMMIT;

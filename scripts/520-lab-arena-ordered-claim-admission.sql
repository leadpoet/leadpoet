-- Admit ordered candidates lazily, retaining every guard before row locking.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $ordered_claim_admission$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
  old_header CONSTANT TEXT:=$old_header$  SELECT * INTO v_run FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id$old_header$;
  new_header CONSTANT TEXT:=$new_header$  SELECT admitted.* INTO v_run FROM (
    SELECT runs.run_id FROM public.lab_arena_runs AS runs
  WHERE runs.round_id = p_round_id
    AND (runs.kind<>'score' OR runs.scoring_generation_floor=v_round.scoring_generation_floor)
    AND runs.status = 'pending'
  ORDER BY
    -- lab_arena_baseline_recovery_priority_v1: finish the existing baseline
    -- recovery before ordinary miner work; every admission guard stays above.
    CASE WHEN runs.kind = 'execute' AND runs.stage = 1
              AND public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      THEN 0 ELSE 1 END,
    CASE WHEN runs.kind = 'execute' AND (
        (v_round.status = 'stage1'
         AND v_round.configuration_doc ->> 'parallel_twenty_icp_execution' = 'true')
        OR (v_round.status = 'stage2'
            AND v_round.configuration_doc ->> 'execution_sequence_policy'
                = 'baseline_scored_first_v1')
      )
      THEN (
        SELECT participant.ordinality
        FROM pg_catalog.jsonb_array_elements(v_round.participants)
             WITH ORDINALITY AS participant(value, ordinality)
        WHERE participant.value ->> 'submission_id' = runs.submission_id
      )
    END,
    CASE WHEN v_round.status = 'stage1'
                  AND v_round.configuration_doc ->>
                        'parallel_twenty_icp_execution' = 'true'
                  AND runs.kind = 'execute'
      THEN runs.icp_position
    END,
    runs.icp_position, runs.created_at, runs.assignment_id
  OFFSET 0
  ) AS candidate
  CROSS JOIN LATERAL (
    SELECT runs.* FROM public.lab_arena_runs AS runs
  WHERE runs.run_id=candidate.run_id
    AND runs.round_id = p_round_id$new_header$;
  old_tail CONSTANT TEXT:=$old_tail$  ORDER BY
    -- lab_arena_baseline_recovery_priority_v1: finish the existing baseline
    -- recovery before ordinary miner work; every admission guard stays above.
    CASE WHEN runs.kind = 'execute' AND runs.stage = 1
              AND public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      THEN 0 ELSE 1 END,
    CASE WHEN runs.kind = 'execute' AND (
        (v_round.status = 'stage1'
         AND v_round.configuration_doc ->> 'parallel_twenty_icp_execution' = 'true')
        OR (v_round.status = 'stage2'
            AND v_round.configuration_doc ->> 'execution_sequence_policy'
                = 'baseline_scored_first_v1')
      )
      THEN (
        SELECT participant.ordinality
        FROM pg_catalog.jsonb_array_elements(v_round.participants)
             WITH ORDINALITY AS participant(value, ordinality)
        WHERE participant.value ->> 'submission_id' = runs.submission_id
      )
    END,
    CASE WHEN v_round.status = 'stage1'
                  AND v_round.configuration_doc ->>
                        'parallel_twenty_icp_execution' = 'true'
                  AND runs.kind = 'execute'
      THEN runs.icp_position
    END,
    runs.icp_position, runs.created_at, runs.assignment_id
  FOR UPDATE SKIP LOCKED
  LIMIT 1;$old_tail$;
  new_tail CONSTANT TEXT:=$new_tail$    FOR UPDATE SKIP LOCKED
    LIMIT 1
  ) AS admitted
  LIMIT 1;$new_tail$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='b645001e6ff490dacc0bf5255b1fa9d2b1c2def53a4980056b29b9ce35edd427' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'4c1c8aa24e6c0f63bdb0ed40fb4cd47266438668445134b017beb5c062845133'
    OR (length(d)-length(replace(d,old_header,'')))<>length(old_header)
    OR (length(d)-length(replace(d,old_tail,'')))<>length(old_tail) THEN
    RAISE EXCEPTION 'ordered claim admission source differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(replace(d,old_header,new_header),old_tail,new_tail);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'b645001e6ff490dacc0bf5255b1fa9d2b1c2def53a4980056b29b9ce35edd427' THEN
    RAISE EXCEPTION 'ordered claim admission readback differs' USING ERRCODE='55000';
  END IF;
END;
$ordered_claim_admission$;
COMMIT;

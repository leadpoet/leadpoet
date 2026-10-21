-- A strictly retained partial judge is not abandoned before completion.
-- Keep its authenticated host fault visible to the existing claim safeguards.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $partial_judge_host_fault$
DECLARE d TEXT; metadata JSONB;
  target OID:='public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::regprocedure;
  old_shared CONSTANT TEXT:=$old_shared$         AND host_error.content->>'status'='abandoned'
         AND host_error.content->>'failure_stage'='runtime'$old_shared$;
  new_shared CONSTANT TEXT:=$new_shared$         AND (host_error.content->>'status'='abandoned'
           OR (host_fault.kind='score'
             AND host_error.content->>'status'='judge_error'))
         AND host_error.content->>'failure_stage'='runtime'$new_shared$;
  old_handoff CONSTANT TEXT:=$old_handoff$                AND host_error.content->>'status'='abandoned'
                AND host_error.content->>'failure_stage'='runtime'$old_handoff$;
  new_handoff CONSTANT TEXT:=$new_handoff$                AND (host_error.content->>'status'='abandoned'
                  OR (prior_score.kind='score'
                    AND host_error.content->>'status'='judge_error'))
                AND host_error.content->>'failure_stage'='runtime'$new_handoff$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
    FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
      AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='v'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
      AND l.lanname='plpgsql';
  IF encode(extensions.digest(d,'sha256'),'hex')='17c0b8d87e6662d01c56d9c3c31d01f833b229e2e5f825d848efa2aca00d4106' THEN RETURN; END IF;
  IF encode(extensions.digest(d,'sha256'),'hex')<>'b645001e6ff490dacc0bf5255b1fa9d2b1c2def53a4980056b29b9ce35edd427'
    OR (length(d)-length(replace(d,old_shared,'')))<>length(old_shared)
    OR (length(d)-length(replace(d,old_handoff,'')))<>length(old_handoff) THEN
    RAISE EXCEPTION 'partial judge host fault source differs' USING ERRCODE='55000';
  END IF;
  EXECUTE replace(replace(d,old_shared,new_shared),old_handoff,new_handoff);
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'17c0b8d87e6662d01c56d9c3c31d01f833b229e2e5f825d848efa2aca00d4106' THEN
    RAISE EXCEPTION 'partial judge host fault readback differs' USING ERRCODE='55000';
  END IF;
END;
$partial_judge_host_fault$;
COMMIT;

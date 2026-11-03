-- A bound terminal provider rejection closes a stopped worker call, not its bill.
-- Apply the existing execute proof to score work. Preserve unknown charges,
-- exact provider bindings, host cleanup evidence and ordinary retry authority.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $closed_failed_score_recovery571$
DECLARE
  signatures TEXT[] := ARRAY['public.lab_arena_expire_leases(text)',
    'public.lab_arena__restart_drain_state_v1()'];
  preimages TEXT[] := ARRAY['5025e77ac2a602941e7166d0a201ba49efb03d0da8af1a0fda1e1889056de40d','a38782b4b6555c455fae64a2434d0bbd394f2f970bc42326475f187a4a1756b8'];
  postimages TEXT[] := ARRAY['cbe667688a31c3e0fbc39e968721b92ff206780c6818df8afde4eeaedd98c187','6de3a3673357bbe823992075fd00443247e1b19c50e7cc3d01307ea381eab984'];
  definitions TEXT[] := ARRAY[]::TEXT[];
  identities JSONB[] := ARRAY[]::JSONB[];
  d TEXT; metadata JSONB; i INTEGER; installed BOOLEAN := TRUE;
BEGIN
  FOR i IN 1..2 LOOP
    SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
      FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
      WHERE p.oid=signatures[i]::regprocedure
        AND pg_get_userbyid(p.proowner)='lab_arena_owner'
        AND p.proacl::TEXT=CASE i WHEN 1 THEN
          '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
          ELSE '{lab_arena_owner=X/lab_arena_owner}' END
        AND p.prosecdef AND p.provolatile::TEXT=CASE i WHEN 1 THEN 'v' ELSE 's' END
        AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
        AND l.lanname='plpgsql';
    definitions:=array_append(definitions,d);
    identities:=array_append(identities,metadata);
    installed:=installed AND encode(extensions.digest(d,'sha256'),'hex')=postimages[i];
  END LOOP;
  IF installed THEN RETURN; END IF;
  FOR i IN 1..2 LOOP
    IF encode(extensions.digest(definitions[i],'sha256'),'hex')<>preimages[i] THEN
      RAISE EXCEPTION 'closed failed score recovery preimage differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
  d:=definitions[1];
  IF (length(d)-length(replace(d,$anchor1_0$OR (v_run.kind = 'execute'
               AND v_round.evaluation_date$anchor1_0$,'')))<>length($anchor1_0$OR (v_run.kind = 'execute'
               AND v_round.evaluation_date$anchor1_0$) THEN
    RAISE EXCEPTION 'closed failed score recovery anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$anchor1_0$OR (v_run.kind = 'execute'
               AND v_round.evaluation_date$anchor1_0$,$updated1_0$OR (v_run.kind IN ('score','execute')
               AND v_round.evaluation_date$updated1_0$);
  IF (length(d)-length(replace(d,$anchor1_1$WHEN 'score' THEN CASE WHEN EXISTS ($anchor1_1$,'')))<>length($anchor1_1$WHEN 'score' THEN CASE WHEN EXISTS ($anchor1_1$) THEN
    RAISE EXCEPTION 'closed failed score recovery anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$anchor1_1$WHEN 'score' THEN CASE WHEN EXISTS ($anchor1_1$,$updated1_1$WHEN 'score' THEN CASE WHEN EXISTS (
                  SELECT 1 FROM (SELECT DISTINCT ON (cost.call_identity) cost.entry_kind,cost.entry_doc
                    FROM public.lab_arena_ledger AS cost WHERE cost.run_id = v_run.run_id
                    ORDER BY cost.call_identity,cost.entry_id DESC) AS heads
                  WHERE heads.entry_kind = 'uncertain'
                    AND heads.entry_doc #> '{call,call_succeeded}' = 'false'::JSONB)
                  THEN 'authenticated_closed_failed_score_runtime_host_error'
                WHEN EXISTS ($updated1_1$);
  definitions[1]:=d;
  d:=definitions[2];
  IF (length(d)-length(replace(d,$anchor2_0$OR (runs.kind = 'execute'
               AND recovery_round.evaluation_date$anchor2_0$,'')))<>length($anchor2_0$OR (runs.kind = 'execute'
               AND recovery_round.evaluation_date$anchor2_0$) THEN
    RAISE EXCEPTION 'closed failed score recovery anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$anchor2_0$OR (runs.kind = 'execute'
               AND recovery_round.evaluation_date$anchor2_0$,$updated2_0$OR (runs.kind IN ('score','execute')
               AND recovery_round.evaluation_date$updated2_0$);
  IF (length(d)-length(replace(d,$anchor2_1$'authenticated_completed_unpriced_score_runtime_host_error'))$anchor2_1$,'')))<>length($anchor2_1$'authenticated_completed_unpriced_score_runtime_host_error'))$anchor2_1$) THEN
    RAISE EXCEPTION 'closed failed score recovery anchor differs' USING ERRCODE='55000';
  END IF;
  d:=replace(d,$anchor2_1$'authenticated_completed_unpriced_score_runtime_host_error'))$anchor2_1$,$updated2_1$'authenticated_completed_unpriced_score_runtime_host_error',
            'authenticated_closed_failed_score_runtime_host_error'))$updated2_1$);
  definitions[2]:=d;
  FOR i IN 1..2 LOOP
    IF encode(extensions.digest(definitions[i],'sha256'),'hex')<>postimages[i] THEN
      RAISE EXCEPTION 'closed failed score recovery postimage differs' USING ERRCODE='55000';
    END IF;
    EXECUTE definitions[i];
    IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=signatures[i]::regprocedure)
        IS DISTINCT FROM identities[i]
      OR encode(extensions.digest(pg_get_functiondef(signatures[i]::regprocedure),'sha256'),'hex')<>postimages[i] THEN
      RAISE EXCEPTION 'closed failed score recovery readback differs' USING ERRCODE='55000';
    END IF;
  END LOOP;
END;
$closed_failed_score_recovery571$;
COMMIT;

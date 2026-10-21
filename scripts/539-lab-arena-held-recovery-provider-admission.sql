-- Quiesce new execution liabilities under an existing recovered-round hold.
-- Existing reservation replay, dispatch, settlement and all leases stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $recovery_admission539$
DECLARE d TEXT; metadata JSONB;
 target OID:='public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
  AND pg_get_userbyid(p.proowner)='lab_arena_owner'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  AND p.prosecdef AND p.provolatile='v'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='8cd8236c6f185956035c327357f5c7d7ea67bee9f83bbab8f7976c97e1b5b4f8' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'ebf02eae4ca75ea99d420fb585bb3daa22ccdb2a97b7a28d39a5368c13134128'
  OR (length(d)-length(replace(d,$old539$  IF v_run.champion_restart_required OR
$old539$,'')))<>length($old539$  IF v_run.champion_restart_required OR
$old539$) THEN
  RAISE EXCEPTION 'held recovery admission source differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,$old539$  IF v_run.champion_restart_required OR
$old539$,$new539$  -- A held recovered evaluation admits no fresh execution liability.
  -- Existing heads returned above: their replay, dispatch and billing drain.
  IF v_run.kind='execute' AND EXISTS(
      SELECT 1 FROM public.lab_arena_restart_claim_control control
      WHERE control.singleton AND control.operator_paused) AND EXISTS(
      SELECT 1 FROM public.lab_arena_scoring_supersessions recovery
      WHERE recovery.round_id=v_round.round_id
        AND recovery.generation_floor=v_round.scoring_generation_floor
        AND recovery.recovery_doc->>'operation'='cancelled_evaluation_recovery_v1'
        AND recovery.new_round_doc->'configuration_doc'=v_round.configuration_doc) THEN
    RETURN pg_catalog.jsonb_build_object('status','stale');
  END IF;
  IF v_run.champion_restart_required OR
$new539$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'8cd8236c6f185956035c327357f5c7d7ea67bee9f83bbab8f7976c97e1b5b4f8' THEN
  RAISE EXCEPTION 'held recovery admission readback differs' USING ERRCODE='55000'; END IF;
END;
$recovery_admission539$;
NOTIFY pgrst,'reload schema';
COMMIT;

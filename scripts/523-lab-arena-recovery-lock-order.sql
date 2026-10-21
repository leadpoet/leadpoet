-- Keep completion and expiry in round-before-run order.
-- Healthy leases need no abandoned-worker recovery scan. All proofs remain unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $v3_lock_order$
DECLARE d TEXT; metadata JSONB; target OID:='public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
  AND pg_get_userbyid(p.proowner)='lab_arena_owner'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  AND p.prosecdef AND p.provolatile='v'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='7046ae819685ab5ca76df9f79cc1f6b8a778129b026b8552f4f2530bfbce24a7' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'efeab3b7ae954b9c52e0dc16d6af02af548a66fc665ed06ef930865a31ff4d1f' THEN
  RAISE EXCEPTION 'v3_lock_order preimage differs' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$old_v3_lock_order_0$  v_run public.lab_arena_runs;$old_v3_lock_order_0$,'')))<>length($old_v3_lock_order_0$  v_run public.lab_arena_runs;$old_v3_lock_order_0$) THEN RAISE EXCEPTION 'v3_lock_order anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_v3_lock_order_0$  v_run public.lab_arena_runs;$old_v3_lock_order_0$,$new_v3_lock_order_0$  v_run public.lab_arena_runs;
  v_source_round_id TEXT;$new_v3_lock_order_0$);
 IF (length(d)-length(replace(d,$old_v3_lock_order_1$BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs$old_v3_lock_order_1$,'')))<>length($old_v3_lock_order_1$BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs$old_v3_lock_order_1$) THEN RAISE EXCEPTION 'v3_lock_order anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_v3_lock_order_1$BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs$old_v3_lock_order_1$,$new_v3_lock_order_1$BEGIN
  -- Match ordinary lease validation: round before run. Reread the run
  -- under lock and reject a source-round move while acquiring that lock.
  SELECT source.round_id INTO v_source_round_id FROM public.lab_arena_runs source
    WHERE source.run_id=p_run_id;
  PERFORM 1 FROM public.lab_arena_rounds round_row
    WHERE round_row.round_id=v_source_round_id FOR SHARE;
  SELECT * INTO v_run FROM public.lab_arena_runs$new_v3_lock_order_1$);
 IF (length(d)-length(replace(d,$old_v3_lock_order_2$  IF v_run.status IN ('accepted', 'failed') THEN$old_v3_lock_order_2$,'')))<>length($old_v3_lock_order_2$  IF v_run.status IN ('accepted', 'failed') THEN$old_v3_lock_order_2$) THEN RAISE EXCEPTION 'v3_lock_order anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_v3_lock_order_2$  IF v_run.status IN ('accepted', 'failed') THEN$old_v3_lock_order_2$,$new_v3_lock_order_2$  IF v_run.round_id IS DISTINCT FROM v_source_round_id THEN
    RAISE EXCEPTION 'lab_arena_run_round_changed' USING ERRCODE = '40001';
  END IF;
  IF v_run.status IN ('accepted', 'failed') THEN$new_v3_lock_order_2$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'7046ae819685ab5ca76df9f79cc1f6b8a778129b026b8552f4f2530bfbce24a7' THEN
  RAISE EXCEPTION 'v3_lock_order readback differs' USING ERRCODE='55000'; END IF;
END;
$v3_lock_order$;
DO $healthy_expiry$
DECLARE d TEXT; metadata JSONB; target OID:='public.lab_arena_expire_leases(text)'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,metadata
 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target
  AND pg_get_userbyid(p.proowner)='lab_arena_owner'
  AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  AND p.prosecdef AND p.provolatile='v'
  AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[] AND l.lanname='plpgsql';
 IF encode(extensions.digest(d,'sha256'),'hex')='61e1dd5c5c472ce8060d356794bd9f621a8fc0f579ce091941826e19bf261468' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'86d4aa32b7549d87ac7a1c21de6a70751276bfcbfd10b6700ef4ce27cad58217' THEN
  RAISE EXCEPTION 'healthy_expiry preimage differs' USING ERRCODE='55000'; END IF;
 IF (length(d)-length(replace(d,$old_healthy_expiry_0$  v_captured_recovery_allowed BOOLEAN := FALSE;$old_healthy_expiry_0$,'')))<>length($old_healthy_expiry_0$  v_captured_recovery_allowed BOOLEAN := FALSE;$old_healthy_expiry_0$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_0$  v_captured_recovery_allowed BOOLEAN := FALSE;$old_healthy_expiry_0$,$new_healthy_expiry_0$  v_captured_recovery_allowed BOOLEAN := FALSE;
  v_has_runtime_error BOOLEAN;$new_healthy_expiry_0$);
 IF (length(d)-length(replace(d,$old_healthy_expiry_1$  -- lab_arena_abandoned_host_recovery_v1:$old_healthy_expiry_1$,'')))<>length($old_healthy_expiry_1$  -- lab_arena_abandoned_host_recovery_v1:$old_healthy_expiry_1$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_1$  -- lab_arena_abandoned_host_recovery_v1:$old_healthy_expiry_1$,$new_healthy_expiry_1$  -- A cheap recovery signal, never authorization. The existing locked
  -- predicates below still prove each abandoned worker and accounting state.
  SELECT EXISTS(SELECT 1 FROM public.lab_arena_runs active
    WHERE active.round_id=p_round_id AND active.status='leased'
      AND active.kind IN ('score','execute') AND EXISTS(
        SELECT 1 FROM public.lab_arena_trajectory_events event
        WHERE event.run_id=active.run_id AND event.event_kind='runtime.error'))
    INTO v_has_runtime_error;
  -- lab_arena_abandoned_host_recovery_v1:$new_healthy_expiry_1$);
 IF (length(d)-length(replace(d,$old_healthy_expiry_2$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      -- A captured lease must retain$old_healthy_expiry_2$,'')))<>length($old_healthy_expiry_2$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      -- A captured lease must retain$old_healthy_expiry_2$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_2$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      -- A captured lease must retain$old_healthy_expiry_2$,$new_healthy_expiry_2$  IF v_has_runtime_error THEN
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      -- A captured lease must retain$new_healthy_expiry_2$);
 IF (length(d)-length(replace(d,$old_healthy_expiry_3$  END LOOP;
  -- The existing whole-round expiry$old_healthy_expiry_3$,'')))<>length($old_healthy_expiry_3$  END LOOP;
  -- The existing whole-round expiry$old_healthy_expiry_3$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_3$  END LOOP;
  -- The existing whole-round expiry$old_healthy_expiry_3$,$new_healthy_expiry_3$  END LOOP;
  END IF;
  -- The existing whole-round expiry$new_healthy_expiry_3$);
 IF (length(d)-length(replace(d,$old_healthy_expiry_4$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND (v_early_recovery_allowed OR$old_healthy_expiry_4$,'')))<>length($old_healthy_expiry_4$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND (v_early_recovery_allowed OR$old_healthy_expiry_4$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_4$  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND (v_early_recovery_allowed OR$old_healthy_expiry_4$,$new_healthy_expiry_4$  IF v_has_runtime_error THEN
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs AS candidate
    WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      AND (v_early_recovery_allowed OR$new_healthy_expiry_4$);
 IF (length(d)-length(replace(d,$old_healthy_expiry_5$  END LOOP;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$old_healthy_expiry_5$,'')))<>length($old_healthy_expiry_5$  END LOOP;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$old_healthy_expiry_5$) THEN RAISE EXCEPTION 'healthy_expiry anchor differs' USING ERRCODE='55000'; END IF;
 d:=replace(d,$old_healthy_expiry_5$  END LOOP;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$old_healthy_expiry_5$,$new_healthy_expiry_5$  END LOOP;
  END IF;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND status = 'leased'$new_healthy_expiry_5$);
 EXECUTE d;
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'61e1dd5c5c472ce8060d356794bd9f621a8fc0f579ce091941826e19bf261468' THEN
  RAISE EXCEPTION 'healthy_expiry readback differs' USING ERRCODE='55000'; END IF;
END;
$healthy_expiry$;
COMMIT;

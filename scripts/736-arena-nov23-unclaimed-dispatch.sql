-- Installation only. Activate separately after forward rewards are ready.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='45s';
CREATE TEMP TABLE recovery736_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
SET LOCAL ROLE lab_arena_owner;

DO $patch736_0$
DECLARE target oid:='public.lab_arena_open_stage(text,smallint,jsonb,integer[])'::regprocedure;
 d text; original jsonb;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,original FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((original->>'proowner')::oid)<>'lab_arena_owner' THEN
   RAISE EXCEPTION 'Unexpected Nov23 function owner'; END IF;
 IF encode(sha256(convert_to(d,'UTF8')),'hex')='77ce187f2db164e68d8395300153984793217140e664caf0bd85c4689aec6309' THEN RETURN; END IF;
 IF encode(sha256(convert_to(d,'UTF8')),'hex')<>'fd1d9ea1870b749014e0685c10a6c3c1a3fdf68c8042dc02bf8d32162851b05f' THEN
   RAISE EXCEPTION 'Nov23 lab_arena_open_stage preimage differs'; END IF;
 EXECUTE replace(d,$anchor$      INSERT INTO public.lab_arena_runs ($anchor$,$replacement$      IF p_round_id='arena-2026-11-23' AND EXISTS(
        SELECT FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round_id
          AND h.request_id='nov23-restore-unclaimed-736-v1') THEN
        UPDATE public.lab_arena_runs SET status='pending',terminal_cause=NULL,terminal_doc=NULL,
          stage_generation=v_generation
        WHERE run_id=v_assignment||':1' AND assignment_id=v_assignment AND round_id=p_round_id
          AND submission_id=v_submission.submission_id AND miner_hotkey=v_submission.miner_hotkey
          AND stage=p_stage AND icp_position=v_position AND attempt=1 AND kind='execute'
          AND status='failed' AND terminal_cause='stage_closed' AND lease_generation=0;
        IF NOT FOUND THEN RAISE EXCEPTION 'Nov23 archived assignment differs'; END IF;
        v_created := v_created + 1;
        CONTINUE;
      END IF;
      INSERT INTO public.lab_arena_runs ($replacement$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM original
   OR encode(sha256(convert_to(pg_get_functiondef(target),'UTF8')),'hex')<>'77ce187f2db164e68d8395300153984793217140e664caf0bd85c4689aec6309'
 THEN RAISE EXCEPTION 'Nov23 lab_arena_open_stage readback differs'; END IF;
END;
$patch736_0$;

DO $patch736_1$
DECLARE target oid:='public.lab_arena_runs_terminal_v1()'::regprocedure;
 d text; original jsonb;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO STRICT d,original FROM pg_proc p WHERE p.oid=target;
 IF pg_get_userbyid((original->>'proowner')::oid)<>'lab_arena_owner' THEN
   RAISE EXCEPTION 'Unexpected Nov23 function owner'; END IF;
 IF encode(sha256(convert_to(d,'UTF8')),'hex')='6abdf453c9ec33b4b189722bf3da00ef7730097cec2c749a9828561b90c2383f' THEN RETURN; END IF;
 IF encode(sha256(convert_to(d,'UTF8')),'hex')<>'b20c68699730fbc0acca6d70c8deebbbdd098389e76e515f203963f284b048ab' THEN
   RAISE EXCEPTION 'Nov23 lab_arena_runs_terminal_v1 preimage differs'; END IF;
 EXECUTE replace(d,$anchor$  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'$anchor$,$replacement$  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-23' AND OLD.kind='execute'
    AND OLD.attempt=1 AND OLD.status='failed' AND OLD.terminal_cause='stage_closed'
    AND OLD.lease_generation=0 AND OLD.runner_hotkey IS NULL AND OLD.claim_request_id IS NULL
    AND OLD.claim_request_hash IS NULL AND OLD.lease_token_hash IS NULL
    AND OLD.terminal_doc->>'previous_status'='pending' AND OLD.result_doc IS NULL
    AND OLD.output_ref IS NULL AND OLD.per_icp_score IS NULL AND OLD.qualification_doc IS NULL
    AND NEW.status='pending' AND NEW.terminal_cause IS NULL AND NEW.terminal_doc IS NULL
    AND (to_jsonb(NEW)-ARRAY['status','stage_generation','terminal_cause','terminal_doc','updated_at'])
        =(to_jsonb(OLD)-ARRAY['status','stage_generation','terminal_cause','terminal_doc','updated_at'])
    AND EXISTS (
      SELECT FROM public.lab_arena_scoring_supersessions h JOIN public.lab_arena_rounds r USING(round_id)
       WHERE h.round_id=OLD.round_id AND h.request_id='nov23-restore-unclaimed-736-v1'
         AND OLD IS NOT DISTINCT FROM jsonb_populate_record(NULL::public.lab_arena_runs,
             h.recovery_doc->'unclaimed_runs'->OLD.run_id)
         AND r.stage_generation+1=NEW.stage_generation
         AND ((OLD.stage=1 AND r.status='committed') OR
              (OLD.stage=2 AND r.status IN('stage1_scoring','stage1_judged','stage1_scored')))
    ) THEN NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
  IF TG_OP='UPDATE' AND TG_TABLE_NAME='lab_arena_runs' AND OLD.round_id='arena-2026-11-04'$replacement$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM original
   OR encode(sha256(convert_to(pg_get_functiondef(target),'UTF8')),'hex')<>'6abdf453c9ec33b4b189722bf3da00ef7730097cec2c749a9828561b90c2383f'
 THEN RAISE EXCEPTION 'Nov23 lab_arena_runs_terminal_v1 readback differs'; END IF;
END;
$patch736_1$;
CREATE OR REPLACE FUNCTION public.lab_arena__reopen_unclaimed_nov23_v1()
RETURNS jsonb LANGUAGE plpgsql VOLATILE SECURITY INVOKER SET search_path=pg_catalog,public
SET timezone='UTC' AS $activate$
DECLARE
 rid constant text := 'arena-2026-11-23';
 later_id constant text := 'arena-2026-11-24';
 later_basis constant text := 'sha256:94f99614eaed7e18b7e8114dc9e69b1a5c03fa2dddbe436d390216b94939abe7';
 request constant text := 'nov23-restore-unclaimed-736-v1';
 r public.lab_arena_rounds; n public.lab_arena_rounds; h public.lab_arena_scoring_supersessions;
 windows jsonb; evidence jsonb; snapshot jsonb; originals jsonb; ok boolean;
 delta interval; opened timestamptz;
BEGIN
 PERFORM pg_advisory_xact_lock_shared(hashtextextended('collection_reward_cutover',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO STRICT r FROM public.lab_arena_rounds WHERE round_id=rid FOR UPDATE;
 SELECT * INTO h FROM public.lab_arena_scoring_supersessions WHERE round_id=rid AND request_id=request;
 IF FOUND THEN
   IF h.recovery_doc->>'reason' IS DISTINCT FROM 'nov23_dispatch_after_predecessor'
     OR h.recovery_doc->>'later_reward_basis_hash' IS DISTINCT FROM later_basis
     OR h.request_hash IS DISTINCT FROM 'sha256:'||encode(sha256(convert_to(h.recovery_doc::text,'UTF8')),'hex')
   THEN RAISE EXCEPTION 'Nov23 recovery receipt differs'; END IF;
   RETURN jsonb_build_object('status','existing','round_status',r.status);
 END IF;
 IF current_setting('session_replication_role')<>'origin'
    OR EXISTS(SELECT FROM public.collection_reward_cutover) OR EXISTS(SELECT FROM public.collection_reward_launch)
    OR to_regprocedure('public.lab_arena_prospective_win_737(text)') IS NULL
    OR (public.lab_arena_restart_guard_state_v1()->>'paused')::boolean IS DISTINCT FROM false
    OR NOT EXISTS (SELECT FROM public.lab_arena_rounds WHERE round_id=later_id
        AND status='published' AND king_outcome='no_king' AND reward_activated_at IS NOT NULL
        AND reward_basis_hash=later_basis)
    OR r.status<>'cancelled' OR r.cancel_reason IS DISTINCT FROM 'scoring_incomplete'
    OR r.published_at IS NOT NULL OR r.reward_activated_at IS NOT NULL
    OR encode(sha256(convert_to((to_jsonb(r)-'updated_at')::text,'UTF8')),'hex')
       <> 'c74554c9cbdf44a784c410ac6c3e397960d4b2cf7359e233f186d2f7d426a7a4'
    OR (SELECT encode(sha256(convert_to(jsonb_agg(to_jsonb(s) ORDER BY submission_id)::text,'UTF8')),'hex')
          FROM public.lab_arena_submissions s WHERE round_id=rid)
       IS DISTINCT FROM 'ad5c789d618ff3701a27433f80fed9a8ead697a310625f77043c7c87531fa220'
    OR (SELECT encode(sha256(convert_to(icps::text,'UTF8')),'hex')
          FROM public.qualification_private_icp_sets WHERE set_id=20261122)
       IS DISTINCT FROM 'f61c02a13c24a818dc7e8c4b2097efda9ecccfad474d611a252ed9fe4079c784'
 THEN RAISE EXCEPTION 'Nov23 round, submissions, bank or award authority differs'; END IF;

 SELECT jsonb_agg(to_jsonb(a) ORDER BY run_id),jsonb_object_agg(run_id,to_jsonb(a))
 INTO originals,snapshot FROM public.lab_arena_runs a WHERE round_id=rid;
 IF encode(sha256(convert_to(originals::text,'UTF8')),'hex') IS DISTINCT FROM
   '1f6c08839c87f687e979a94e2107754f0c34effa0406f430ecf750714500e307'
   OR jsonb_array_length(originals)<>70
   OR EXISTS(SELECT FROM public.lab_arena_runs a WHERE round_id=rid
     AND (a.status<>'failed' OR a.kind<>'execute' OR a.attempt<>1
          OR a.terminal_cause IS DISTINCT FROM 'stage_closed' OR a.lease_generation<>0
          OR a.terminal_doc->>'previous_status' IS DISTINCT FROM 'pending'
          OR a.terminal_doc->'infrastructure_incomplete' IS DISTINCT FROM 'true'::jsonb
          OR a.runner_hotkey IS NOT NULL OR a.claim_request_id IS NOT NULL
          OR a.claim_request_hash IS NOT NULL OR a.lease_token_hash IS NOT NULL
          OR a.claim_response IS NOT NULL OR a.output_ref IS NOT NULL OR a.result_doc IS NOT NULL
          OR a.per_icp_score IS NOT NULL OR a.qualification_doc IS NOT NULL
          OR EXISTS(SELECT FROM public.lab_arena_ledger l WHERE l.run_id=a.run_id)
          OR EXISTS(SELECT FROM public.lab_arena_trajectory_events t WHERE t.run_id=a.run_id)))
 THEN RAISE EXCEPTION 'Nov23 requires exactly 70 untouched, unclaimed assignments'; END IF;

 opened:=date_trunc('second',clock_timestamp());
 delta:=opened-(r.configuration_doc#>>'{schedule,stage_1_start}')::timestamptz;
 SELECT jsonb_object_agg(k,to_char((r.configuration_doc#>>ARRAY['schedule',k])::timestamptz+delta,
        'YYYY-MM-DD"T"HH24:MI:SS"Z"'))
 INTO windows FROM unnest(ARRAY['stage_1_start','stage_1_close','stage_1_scoring_close','stage_2_start',
      'stage_2_close','final_scoring_close','publication_deadline']) AS k;
 n:=r; n.status:='committed'; n.status_generation:=r.status_generation+1;
 n.cancel_reason:=NULL; n.stage1_scoring_plan_doc:=NULL; n.stage2_scoring_plan_doc:=NULL; n.finalists:=NULL;
 n.configuration_doc:=jsonb_set(r.configuration_doc,'{schedule}',r.configuration_doc->'schedule'||windows);
 evidence:=jsonb_build_object('operation','round_operational_update_v1','reason','nov23_dispatch_after_predecessor',
   'later_round_id',later_id,'later_reward_basis_hash',later_basis,'windows',windows,'unclaimed_runs',snapshot);
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
   transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 SELECT rid,coalesce(max(generation_floor),0)+1,request,'sha256:'||encode(sha256(convert_to(evidence::text,'UTF8')),'hex'),
   'authorized-nov23-dispatch-recovery',txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at',
   '{}',evidence FROM public.lab_arena_scoring_supersessions WHERE round_id=rid;
 UPDATE public.lab_arena_rounds SET status=n.status,status_generation=n.status_generation,cancel_reason=NULL,
   stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL,finalists=NULL,configuration_doc=n.configuration_doc WHERE round_id=rid;
 EXECUTE 'SELECT public.lab_arena_prospective_win_737($1)' INTO ok USING rid;
 IF ok IS DISTINCT FROM true OR (SELECT to_jsonb(after)-'updated_at' FROM public.lab_arena_rounds after WHERE round_id=rid)
     IS DISTINCT FROM to_jsonb(n)-'updated_at' THEN RAISE EXCEPTION 'Nov23 activation verification failed'; END IF;
 RETURN jsonb_build_object('status','reopened','round_status',n.status,'assignments',70,'schedule',n.configuration_doc->'schedule');
END;
$activate$;
REVOKE ALL ON FUNCTION public.lab_arena__reopen_unclaimed_nov23_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
RESET ROLE;
DO $acl$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.recovery736_acl) THEN REVOKE CREATE ON SCHEMA public FROM lab_arena_owner; END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM (SELECT nspacl FROM pg_temp.recovery736_acl)
 THEN RAISE EXCEPTION 'Nov23 schema ACL differs'; END IF;
END;
$acl$;
COMMIT;

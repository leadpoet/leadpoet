-- Restore an operator-selected archived authority without rewriting judgments.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';
DO $schema_create538$
DECLARE before_doc JSONB;
BEGIN
 SELECT jsonb_build_object('oid',oid,'owner',nspowner,'acl',to_jsonb(nspacl),
   'owner_had_create',has_schema_privilege('lab_arena_owner',oid,'CREATE'))
 INTO before_doc FROM pg_namespace WHERE nspname='public';
 IF before_doc IS NULL THEN
  RAISE EXCEPTION 'Supersession public schema missing' USING ERRCODE='55000';
 END IF;
 PERFORM set_config('leadpoet.arena538_public_schema_before',before_doc::TEXT,TRUE);
 IF NOT (before_doc->>'owner_had_create')::BOOLEAN THEN
  GRANT CREATE ON SCHEMA public TO lab_arena_owner;
 END IF;
END;
$schema_create538$;

DO $recovery_relations538$
BEGIN
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_published_results_v1'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],false,false,["security_invoker=true"]]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_published_results_v1' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_restart_claim_control'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_restart_claim_control' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_rounds'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_rounds' USING ERRCODE='42501';
 END IF;
 IF (SELECT jsonb_build_array(pg_get_userbyid(c.relowner),(SELECT jsonb_agg(jsonb_build_array(pg_get_userbyid(a.grantee),a.privilege_type,a.is_grantable,pg_get_userbyid(a.grantor)) ORDER BY pg_get_userbyid(a.grantee),a.privilege_type) FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner)))a WHERE a.grantee<>c.relowner),c.relrowsecurity,c.relforcerowsecurity,c.reloptions)
   FROM pg_class c WHERE c.oid=to_regclass('public.lab_arena_runs'))
   IS DISTINCT FROM $relation517$["lab_arena_owner",[["lab_arena_service","SELECT",false,"lab_arena_owner"],["leadpoet_parity_reader","SELECT",false,"lab_arena_owner"]],true,false,null]$relation517$::JSONB THEN
  RAISE EXCEPTION 'Supersession relation security differs: lab_arena_runs' USING ERRCODE='42501';
 END IF;
END;
$recovery_relations538$;
DO $recovery_history538$
DECLARE t OID:='public.lab_arena_scoring_supersessions'::regclass;
BEGIN
 IF t IS NOT NULL AND EXISTS(SELECT 1 FROM pg_class c WHERE c.oid=t AND (
   pg_get_userbyid(c.relowner)<>'lab_arena_owner' OR NOT c.relrowsecurity
   OR c.relforcerowsecurity OR c.reloptions IS NOT NULL
   OR EXISTS(SELECT 1 FROM pg_policy policy WHERE policy.polrelid=c.oid)
   OR NOT EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname='lab_arena_scoring_supersessions_append_only'
     AND trigger.tgfoid=to_regprocedure('public.lab_arena_append_only_v1()')
     AND trigger.tgenabled='O' AND trigger.tgtype=27)
   OR EXISTS(SELECT 1 FROM pg_trigger trigger WHERE trigger.tgrelid=c.oid AND NOT trigger.tgisinternal
     AND trigger.tgname<>'lab_arena_scoring_supersessions_append_only')
   OR EXISTS(SELECT 1 FROM aclexplode(coalesce(c.relacl,acldefault('r',c.relowner))) a
     WHERE a.grantee<>c.relowner AND NOT (a.grantee=to_regrole('lab_arena_service')
       AND a.privilege_type='SELECT' AND NOT a.is_grantable)))) THEN
  RAISE EXCEPTION 'Supersession history security differs' USING ERRCODE='42501';
 END IF;
END;
$recovery_history538$;
DO $security538$
DECLARE actual RECORD; wanted RECORD;
BEGIN
 FOR wanted IN SELECT * FROM(VALUES
  ('public.lab_arena_restore_recorded_results_v1(text,jsonb,bigint,bigint,bigint,text,text,text)','c961e216ad7ea6e715dca742d2f816734aa12b138e23906d938c67be1d91035b','v','plpgsql','{lab_arena_owner=X/lab_arena_owner}'),
  ('public.lab_arena_scoring_policy_history_v1(text)','e0f898dbdd6f78fa26991b007ee68b781062201d27de4dfde1457e811545c370','s','sql','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
 )v(signature,source_hash,volatility,language,acl) LOOP
 SELECT pg_get_userbyid(p.proowner)owner,p.proacl::TEXT acl,p.prosecdef,p.provolatile,p.proconfig,l.lanname,
  encode(extensions.digest(p.prosrc,'sha256'),'hex')source_hash,encode(extensions.digest(pg_get_functiondef(p.oid),'sha256'),'hex')definition_hash
 INTO actual FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=to_regprocedure(wanted.signature);
 IF NOT FOUND THEN CONTINUE; END IF;
 IF actual.owner<>'lab_arena_owner' OR actual.acl IS DISTINCT FROM wanted.acl
  OR actual.prosecdef IS DISTINCT FROM TRUE OR actual.provolatile::TEXT<>wanted.volatility
  OR actual.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR actual.lanname<>wanted.language
  OR(actual.source_hash<>wanted.source_hash AND NOT(wanted.signature='public.lab_arena_scoring_policy_history_v1(text)'
    AND actual.definition_hash='8b74489179ad4a3fcadd806e6b5a36390a8165ec150d07d6b67919695772b2be')) THEN
  RAISE EXCEPTION 'recorded restoration function security or source differs' USING ERRCODE='55000'; END IF;
 END LOOP;
END;
$security538$;
DO $patch538_terminal$
DECLARE d TEXT; m JSONB; target OID:='public.lab_arena_runs_terminal_v1()'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
   FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM
  jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',false,'v',ARRAY['search_path=pg_catalog']::TEXT[],'plpgsql') THEN
  RAISE EXCEPTION 'recorded restoration trigger security differs' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='98edd53fc1f3756ff1829abce68f0f3ffae0911151778d4208d084160284c868' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'e4e49916138628b720271f6f467656574f8ae62b08024f4891b3f74bfa7f5920' THEN
  RAISE EXCEPTION 'recorded restoration trigger source differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,$old538$BEGIN
$old538$,$new538$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new538$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'98edd53fc1f3756ff1829abce68f0f3ffae0911151778d4208d084160284c868' THEN
  RAISE EXCEPTION 'recorded restoration trigger readback differs' USING ERRCODE='55000'; END IF;
END;
$patch538_terminal$;

DO $patch538_integrity$
DECLARE d TEXT; m JSONB; target OID:='public.lab_arena_integrity_run_guard_v1()'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
   FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM
  jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,anon=X/lab_arena_owner,authenticated=X/lab_arena_owner,service_role=X/lab_arena_owner}',false,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
  RAISE EXCEPTION 'recorded restoration trigger security differs' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='9c001214f9b067d2b0cb285915a52ff326909f250b55f0ac478fea209d2e70b3' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'dd7fcd3811ea5f85a5437df0a9c77b80bb688ff3a972527949d29816fcf4376d' THEN
  RAISE EXCEPTION 'recorded restoration trigger source differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,$old538$BEGIN
$old538$,$new538$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new538$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'9c001214f9b067d2b0cb285915a52ff326909f250b55f0ac478fea209d2e70b3' THEN
  RAISE EXCEPTION 'recorded restoration trigger readback differs' USING ERRCODE='55000'; END IF;
END;
$patch538_integrity$;

DO $patch538_holdguard$
DECLARE d TEXT; m JSONB; target OID:='public.lab_arena_operator_hold_transition_guard_v1()'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
   FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM
  jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',true,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
  RAISE EXCEPTION 'recorded restoration trigger security differs' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='a1180771aa74f37a92fedd26d35a6e3fadc6fa474a50eda5da08df26b5e00268' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'cb2c8cc81d983508de7edacb7d910116cef081dba3bf04c82ab9081e5085331d' THEN
  RAISE EXCEPTION 'recorded restoration trigger source differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,$old538$BEGIN
$old538$,$new538$BEGIN
  IF EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
    WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
     AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
     AND h.old_round_doc=to_jsonb(OLD)-'updated_at'
     AND (h.new_round_doc-ARRAY['arena_network_name','arena_netuid'])=
       (to_jsonb(NEW)-ARRAY['updated_at','arena_network_name','arena_netuid'])) THEN RETURN NEW; END IF;
$new538$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'a1180771aa74f37a92fedd26d35a6e3fadc6fa474a50eda5da08df26b5e00268' THEN
  RAISE EXCEPTION 'recorded restoration trigger readback differs' USING ERRCODE='55000'; END IF;
END;
$patch538_holdguard$;
DO $patch538_holdscore$
DECLARE d TEXT; m JSONB; target OID:='public.lab_arena_operator_hold_score_guard_v1()'::regprocedure;
BEGIN
 SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc' INTO d,m FROM pg_proc p WHERE p.oid=target;
 IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
   FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target) IS DISTINCT FROM
  jsonb_build_array('lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',true,'v',ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql') THEN
  RAISE EXCEPTION 'recorded restoration trigger security differs' USING ERRCODE='42501'; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')='c12bef0857ba2c00c417f21baa589bac63024f51514dcb2a698ec7953f4793d9' THEN RETURN; END IF;
 IF encode(extensions.digest(d,'sha256'),'hex')<>'eca26214ccf546d09745ab1c3bb7b1e460a21fda5b4594b430564e2efcde26f1' THEN
  RAISE EXCEPTION 'recorded restoration trigger source differs' USING ERRCODE='55000'; END IF;
 EXECUTE replace(d,$old538$BEGIN
$old538$,$new538$BEGIN
  IF TG_OP='UPDATE' AND OLD.kind='execute'
    AND (to_jsonb(NEW)-ARRAY['per_icp_score','qualification_doc','updated_at'])=
        (to_jsonb(OLD)-ARRAY['per_icp_score','qualification_doc','updated_at'])
    AND EXISTS(SELECT 1 FROM public.lab_arena_scoring_supersessions h
      JOIN public.lab_arena_scoring_supersessions original_history ON original_history.round_id=h.round_id
       AND original_history.generation_floor=(h.recovery_doc->>'restored_from_generation_floor')::BIGINT
      WHERE h.round_id=OLD.round_id AND h.transaction_id=txid_current() AND h.backend_pid=pg_backend_pid()
       AND h.recovery_doc->>'operation'='restore_recorded_result_v1'
       AND h.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',OLD.per_icp_score,'qualification_doc',OLD.qualification_doc)
       AND original_history.old_score_projections->OLD.run_id=jsonb_build_object('per_icp_score',NEW.per_icp_score,'qualification_doc',NEW.qualification_doc)) THEN
    NEW.updated_at:=clock_timestamp(); RETURN NEW;
  END IF;
$new538$);
 IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM m
  OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'c12bef0857ba2c00c417f21baa589bac63024f51514dcb2a698ec7953f4793d9' THEN
  RAISE EXCEPTION 'recorded restoration trigger readback differs' USING ERRCODE='55000'; END IF;
END;
$patch538_holdscore$;
CREATE OR REPLACE FUNCTION public.lab_arena_restore_recorded_results_v1(
 p_round_id TEXT,p_expected_configuration JSONB,p_expected_status_generation BIGINT,
 p_expected_stage_generation BIGINT,p_source_history_generation_floor BIGINT,
 p_request_id TEXT,p_actor_ref TEXT,p_reason TEXT)
RETURNS JSONB LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $restore538$
DECLARE r public.lab_arena_rounds; n public.lab_arena_rounds; c public.lab_arena_restart_claim_control;
 source_history public.lab_arena_scoring_supersessions; receipt public.lab_arena_scoring_supersessions;
 participant JSONB; keep JSONB:='[]'; excluded JSONB:='[]'; blocked JSONB; plan JSONB;
 projections JSONB; request_hash TEXT; count_positions INTEGER; expected_count INTEGER;
 selected_floor BIGINT; revision BIGINT; stopped INTEGER; k TEXT; stopped_runs JSONB;
BEGIN
 IF jsonb_typeof(p_expected_configuration) IS DISTINCT FROM 'object'
  OR p_expected_status_generation IS NULL OR p_expected_stage_generation IS NULL
  OR p_source_history_generation_floor IS NULL
  OR char_length(coalesce(p_request_id,'')) NOT BETWEEN 16 AND 128
  OR char_length(coalesce(p_actor_ref,'')) NOT BETWEEN 1 AND 128
  OR char_length(coalesce(p_reason,'')) NOT BETWEEN 1 AND 256 THEN
  RAISE EXCEPTION 'recorded restoration input invalid' USING ERRCODE='22023'; END IF;
 request_hash:='sha256:'||encode(extensions.digest(jsonb_build_object('configuration',p_expected_configuration,
  'status_generation',p_expected_status_generation,'stage_generation',p_expected_stage_generation,
  'source_history_generation_floor',p_source_history_generation_floor,'actor_ref',p_actor_ref,'reason',p_reason)::TEXT,'sha256'),'hex');
 PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
 SELECT * INTO r FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
 IF NOT FOUND THEN RAISE EXCEPTION 'recorded restoration round missing' USING ERRCODE='P0002'; END IF;
 SELECT * INTO receipt FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id AND request_id=p_request_id;
 IF FOUND THEN
  IF receipt.request_hash IS DISTINCT FROM request_hash OR receipt.recovery_doc->>'operation' IS DISTINCT FROM 'restore_recorded_result_v1' THEN
   RAISE EXCEPTION 'recorded restoration request conflict' USING ERRCODE='22023'; END IF;
  RETURN jsonb_build_object('status','existing','revision_generation',receipt.generation_floor,
   'selected_generation_floor',receipt.new_round_doc->'scoring_generation_floor');
 END IF;
 SELECT * INTO c FROM public.lab_arena_restart_claim_control WHERE singleton FOR SHARE;
 SELECT * INTO source_history FROM public.lab_arena_scoring_supersessions
  WHERE round_id=p_round_id AND generation_floor=p_source_history_generation_floor;
 IF NOT FOUND OR source_history.recovery_doc->>'operation' IS DISTINCT FROM 'cancelled_evaluation_recovery_v1'
  OR c.singleton IS NULL OR c.operator_paused IS DISTINCT FROM TRUE
  OR c.actor_ref IS DISTINCT FROM p_actor_ref OR c.pause_reason IS DISTINCT FROM p_reason
  OR coalesce(c.guard_commitment,'')<>'' OR coalesce(c.owner_commitment,'')<>''
  OR c.guard_expires_at IS NOT NULL OR coalesce(c.candidate_commit,'')<>''
  OR coalesce(c.restart_scope,'')<>'' OR coalesce(c.restart_phase,'')<>''
  OR coalesce(c.captured_leases,'[]'::JSONB)<>'[]'::JSONB
  OR r.configuration_doc IS DISTINCT FROM p_expected_configuration
  OR r.status_generation IS DISTINCT FROM p_expected_status_generation
  OR r.stage_generation IS DISTINCT FROM p_expected_stage_generation
  OR r.status NOT IN ('stage2','stage2_closed','stage2_scoring','stage2_judged','scored','cancelled')
  OR r.published_at IS NOT NULL OR r.publication_doc IS NOT NULL
  OR r.reward_activated_at IS NOT NULL OR r.reward_basis_doc IS NOT NULL OR r.reward_basis_hash IS NOT NULL
  OR r.effective_reward_epoch IS NOT NULL OR r.promotion_doc IS NOT NULL OR r.baseline_promoted_at IS NOT NULL
  OR r.scoring_generation_floor IS DISTINCT FROM source_history.generation_floor
  OR r.configuration_doc IS DISTINCT FROM source_history.new_round_doc->'configuration_doc'
  OR source_history.old_round_doc#>>'{configuration_doc,execution_sequence_policy}' IS DISTINCT FROM 'baseline_scored_first_v1' THEN
  RAISE EXCEPTION 'recorded restoration precondition failed' USING ERRCODE='55000'; END IF;
 selected_floor:=(source_history.old_round_doc->>'scoring_generation_floor')::BIGINT;
 IF selected_floor IS NULL OR selected_floor<0 OR selected_floor>=r.scoring_generation_floor THEN
  RAISE EXCEPTION 'recorded restoration authority invalid' USING ERRCODE='55000'; END IF;
 PERFORM 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status IN('pending','leased','submitted') FOR UPDATE;
 IF EXISTS(SELECT 1 FROM(SELECT DISTINCT assignment_id FROM public.lab_arena_runs WHERE round_id=p_round_id)a
    WHERE NOT public.lab_arena__judge_calls_clear_v1(a.assignment_id))
  OR EXISTS(WITH round_call_rows AS MATERIALIZED(
    SELECT l.call_identity,l.entry_id,l.entry_kind FROM public.lab_arena_ledger l
    WHERE l.round_id=p_round_id AND l.call_identity IS NOT NULL)
    SELECT 1 FROM(SELECT DISTINCT ON(l.call_identity)l.call_identity,l.entry_id,l.entry_kind
      FROM round_call_rows l ORDER BY l.call_identity,l.entry_id DESC)head
    WHERE head.entry_kind IN('reservation','dispatch') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=head.call_identity AND later.entry_id>head.entry_id)) THEN
  RAISE EXCEPTION 'recorded restoration not drained' USING ERRCODE='55000'; END IF;
 expected_count:=(source_history.old_round_doc#>>'{configuration_doc,stage_1_icp_count}')::INTEGER
  +(source_history.old_round_doc#>>'{configuration_doc,stage_2_icp_count}')::INTEGER;
 IF expected_count IS NULL OR expected_count NOT BETWEEN 2 AND 100 THEN
  RAISE EXCEPTION 'recorded restoration benchmark invalid' USING ERRCODE='55000'; END IF;
 FOR participant IN SELECT value FROM jsonb_array_elements(source_history.old_round_doc->'participants') LOOP
  SELECT value INTO blocked FROM jsonb_array_elements(coalesce(source_history.recovery_doc->'excluded_participants','[]'))
   WHERE value->>'submission_id'=participant->>'submission_id';
  SELECT count(DISTINCT position) INTO count_positions FROM generate_series(0,expected_count-1)position
   WHERE EXISTS(SELECT 1 FROM public.lab_arena_runs original
    WHERE original.round_id=p_round_id AND original.submission_id=participant->>'submission_id'
      AND original.miner_hotkey=participant->>'miner_hotkey' AND original.kind='execute'
      AND original.stage=CASE WHEN participant->'is_king'='true'::JSONB THEN 1 ELSE 2 END
      AND original.icp_position=position AND source_history.old_score_projections ? original.run_id
      AND ((original.status='accepted' AND original.terminal_cause='accepted' AND original.output_ref IS NOT NULL
        AND EXISTS(SELECT 1 FROM public.lab_arena_runs judge
          WHERE judge.round_id=p_round_id AND judge.kind='score' AND judge.status='accepted'
           AND judge.terminal_cause='accepted' AND judge.scoring_generation_floor=selected_floor
           AND judge.scored_run_id=original.run_id AND judge.output_ref IS NOT NULL AND judge.result_doc IS NOT NULL
           AND (judge.submission_id,judge.miner_hotkey,judge.stage,judge.icp_position)
             IS NOT DISTINCT FROM(original.submission_id,original.miner_hotkey,original.stage,original.icp_position)))
       OR (original.status='failed' AND NOT coalesce((original.terminal_doc->>'infrastructure_incomplete')::BOOLEAN,FALSE)
        AND EXISTS(SELECT 1 FROM jsonb_array_elements(coalesce(source_history.old_round_doc->'stage2_scoring_plan_doc'->'zero_rows','[]'))z
          WHERE z->>'submission_id'=original.submission_id AND (z->>'icp_position')::INTEGER=position))));
  IF blocked IS NULL AND count_positions=expected_count AND EXISTS(
    SELECT 1 FROM public.lab_arena_runs original JOIN public.lab_arena_runs judge ON judge.scored_run_id=original.run_id
    WHERE original.round_id=p_round_id AND original.submission_id=participant->>'submission_id'
      AND original.miner_hotkey=participant->>'miner_hotkey' AND original.kind='execute' AND original.status='accepted'
      AND source_history.old_score_projections ? original.run_id AND judge.kind='score' AND judge.status='accepted'
      AND judge.terminal_cause='accepted' AND judge.scoring_generation_floor=selected_floor)
  THEN keep:=keep||jsonb_build_array(participant);
  ELSE
   IF participant->'is_king'='true'::JSONB THEN RAISE EXCEPTION 'recorded restoration baseline incomplete' USING ERRCODE='55000'; END IF;
   excluded:=excluded||jsonb_build_array(jsonb_build_object('submission_id',participant->>'submission_id',
    'miner_hotkey',participant->>'miner_hotkey','status',CASE WHEN blocked IS NOT NULL THEN 'provider_blocked' ELSE 'evaluation_incomplete' END,
    'evaluation','not_evaluated','reason',CASE WHEN blocked IS NOT NULL THEN 'openrouter_zdr_route_unavailable' ELSE 'operator_cutoff' END));
  END IF;
 END LOOP;
 IF (SELECT count(*) FROM jsonb_array_elements(keep)p WHERE p->'is_king'='true'::JSONB)<>1
  OR jsonb_array_length(keep)<2 THEN RAISE EXCEPTION 'recorded restoration cohort invalid' USING ERRCODE='55000'; END IF;
 IF EXISTS(SELECT 1 FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status IN('leased','submitted')
    AND (a.status<>'leased' OR a.kind<>'execute' OR a.stage<>2 OR a.stage_generation<>r.stage_generation
      OR a.attempt<>2 OR EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE p->>'submission_id'=a.submission_id)
      OR NOT EXISTS(SELECT 1 FROM public.lab_arena_runs parent WHERE parent.assignment_id=a.assignment_id
        AND parent.attempt=1 AND parent.kind='execute' AND parent.status='failed' AND parent.terminal_cause='stage_closed'
        AND parent.terminal_doc->>'previous_status'='pending' AND parent.terminal_doc->'infrastructure_incomplete'='true'::JSONB
        AND parent.lease_generation=0 AND parent.claim_request_id IS NULL AND parent.runner_hotkey IS NULL
        AND source_history.old_score_projections ? parent.run_id))) THEN
  RAISE EXCEPTION 'recorded restoration active selected or unrelated work' USING ERRCODE='55000'; END IF;
 IF EXISTS(WITH active_call_rows AS MATERIALIZED(
    SELECT l.call_identity,l.entry_id,l.entry_kind FROM public.lab_arena_ledger l JOIN public.lab_arena_runs a ON a.run_id=l.run_id
    WHERE a.round_id=p_round_id AND a.status='leased' AND l.call_identity IS NOT NULL)
    SELECT 1 FROM(SELECT DISTINCT ON(l.call_identity)l.call_identity,l.entry_id,l.entry_kind
      FROM active_call_rows l ORDER BY l.call_identity,l.entry_id DESC)head
    WHERE head.entry_kind IN('reservation','dispatch','uncertain') AND NOT EXISTS(SELECT 1 FROM public.lab_arena_ledger later
      WHERE later.call_identity=head.call_identity AND later.entry_id>head.entry_id)) THEN
  RAISE EXCEPTION 'recorded restoration active provider not settled' USING ERRCODE='55000'; END IF;
 SELECT coalesce(jsonb_agg(to_jsonb(a) ORDER BY a.run_id),'[]') INTO stopped_runs
  FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.status IN('pending','leased');
 n:=r; n.configuration_doc:=source_history.old_round_doc->'configuration_doc'; n.participants:=keep;
 n.scoring_generation_floor:=selected_floor; n.status:='stage2_judged'; n.cancel_reason:=NULL;
 n.status_generation:=r.status_generation+1; n.stage_generation:=r.stage_generation+1;
 n.stage1_scoring_plan_doc:=source_history.old_round_doc->'stage1_scoring_plan_doc';
 plan:=source_history.old_round_doc->'stage2_scoring_plan_doc';
 IF jsonb_typeof(plan) IS DISTINCT FROM 'object' THEN RAISE EXCEPTION 'recorded restoration plan missing' USING ERRCODE='55000'; END IF;
 FOREACH k IN ARRAY ARRAY['work_items','zero_rows','incomplete_rows'] LOOP
  IF plan ? k THEN plan:=jsonb_set(plan,ARRAY[k],coalesce((SELECT jsonb_agg(item) FROM jsonb_array_elements(plan->k)item
    WHERE EXISTS(SELECT 1 FROM jsonb_array_elements(keep)p WHERE p->>'submission_id'=item->>'submission_id')),'[]')); END IF;
 END LOOP;
 n.stage2_scoring_plan_doc:=plan;
 SELECT jsonb_agg(p->>'submission_id' ORDER BY p->>'submission_id') INTO n.finalists FROM jsonb_array_elements(keep)p WHERE p->'is_king' IS DISTINCT FROM 'true'::JSONB;
 SELECT greatest(n.stage_generation,coalesce(max(generation_floor),0)+1) INTO revision FROM public.lab_arena_scoring_supersessions WHERE round_id=p_round_id;
 IF revision<>n.stage_generation THEN RAISE EXCEPTION 'recorded restoration revision differs' USING ERRCODE='55000'; END IF;
 SELECT coalesce(jsonb_object_agg(a.run_id,jsonb_build_object('per_icp_score',a.per_icp_score,'qualification_doc',a.qualification_doc)),'{}')
  INTO projections FROM public.lab_arena_runs a WHERE a.round_id=p_round_id AND a.kind='execute';
 INSERT INTO public.lab_arena_scoring_supersessions(round_id,generation_floor,request_id,request_hash,actor_ref,
  transaction_id,backend_pid,old_round_doc,new_round_doc,old_score_projections,recovery_doc)
 VALUES(p_round_id,revision,p_request_id,request_hash,p_actor_ref,txid_current(),pg_backend_pid(),to_jsonb(r)-'updated_at',to_jsonb(n)-'updated_at',projections,
  jsonb_build_object('operation','restore_recorded_result_v1','restored_from_generation_floor',source_history.generation_floor,
   'selected_generation_floor',selected_floor,'excluded_participants',excluded,'operator_reason',p_reason,'stopped_runs',stopped_runs));
 UPDATE public.lab_arena_runs SET status='failed',terminal_cause='stage_closed',terminal_doc=coalesce(terminal_doc,'{}')||
  jsonb_build_object('previous_status',status,'closed_at',clock_timestamp(),'infrastructure_incomplete',TRUE,
   'operator_stopped',TRUE,'operator_actor',p_actor_ref,'original_lease_expires_at',lease_expires_at)
  WHERE round_id=p_round_id AND status IN('pending','leased');
 GET DIAGNOSTICS stopped=ROW_COUNT;
 UPDATE public.lab_arena_rounds SET configuration_doc=n.configuration_doc,participants=n.participants,status=n.status,
  cancel_reason=n.cancel_reason,status_generation=n.status_generation,stage_generation=n.stage_generation,
  scoring_generation_floor=n.scoring_generation_floor,stage1_scoring_plan_doc=n.stage1_scoring_plan_doc,
  stage2_scoring_plan_doc=n.stage2_scoring_plan_doc,finalists=n.finalists WHERE round_id=p_round_id;
 UPDATE public.lab_arena_runs a SET per_icp_score=(saved.value->>'per_icp_score')::NUMERIC,
  qualification_doc=CASE WHEN saved.value->'qualification_doc'='null'::JSONB THEN NULL ELSE saved.value->'qualification_doc' END
  FROM jsonb_each(source_history.old_score_projections)saved WHERE a.run_id=saved.key AND a.round_id=p_round_id
   AND a.kind='execute' AND jsonb_build_object('per_icp_score',a.per_icp_score,'qualification_doc',a.qualification_doc) IS DISTINCT FROM saved.value;
 RETURN jsonb_build_object('status','restored','revision_generation',revision,'selected_generation_floor',selected_floor,
  'retained_participants',keep,'excluded_participants',excluded,'stopped_pending',stopped,'round_status',n.status);
END;
$restore538$;
ALTER FUNCTION public.lab_arena_restore_recorded_results_v1(TEXT,JSONB,BIGINT,BIGINT,BIGINT,TEXT,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_restore_recorded_results_v1(TEXT,JSONB,BIGINT,BIGINT,BIGINT,TEXT,TEXT,TEXT)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
CREATE OR REPLACE FUNCTION public.lab_arena_scoring_policy_history_v1(p_round_id TEXT)
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $history538$
 SELECT jsonb_build_object('round_id',p_round_id,'revisions',coalesce(jsonb_agg(jsonb_build_object(
  'previous_generation_floor',coalesce((h.old_round_doc->>'scoring_generation_floor')::BIGINT,0),
  'generation_floor',h.generation_floor,'old_scorer_policy',h.old_round_doc#>'{configuration_doc,scorer_policy}',
  'new_scorer_policy',h.new_round_doc#>'{configuration_doc,scorer_policy}',
  'recovery',CASE WHEN h.recovery_doc IS NOT NULL AND h.recovery_doc->>'operation'<>'restore_recorded_result_v1' THEN
   jsonb_build_object('operation',h.recovery_doc->>'operation','excluded_participants',coalesce((
    SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id','miner_hotkey',p->>'miner_hotkey',
      'status','provider_blocked','evaluation','not_evaluated','reason','openrouter_zdr_route_unavailable'))
    FROM jsonb_array_elements(h.recovery_doc->'excluded_participants')p),'[]')) END,
  'restoration',CASE WHEN h.recovery_doc->>'operation'='restore_recorded_result_v1' THEN
   jsonb_build_object('selected_generation_floor',h.recovery_doc->'selected_generation_floor',
    'excluded_participants',coalesce((SELECT jsonb_agg(jsonb_build_object('submission_id',p->>'submission_id',
     'miner_hotkey',p->>'miner_hotkey','status',p->>'status','evaluation','not_evaluated','reason',p->>'reason'))
     FROM jsonb_array_elements(h.recovery_doc->'excluded_participants')p),'[]')) END
 ) ORDER BY h.generation_floor),'[]')) FROM public.lab_arena_scoring_supersessions h WHERE h.round_id=p_round_id;
$history538$;
ALTER FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_scoring_policy_history_v1(TEXT) TO lab_arena_service;
DO $schema_restore538$
DECLARE before_doc JSONB:=current_setting('leadpoet.arena538_public_schema_before',TRUE)::JSONB;
 actual_doc JSONB;
BEGIN
 IF before_doc IS NULL THEN
  RAISE EXCEPTION 'Supersession schema permission snapshot missing' USING ERRCODE='55000';
 END IF;
 IF NOT (before_doc->>'owner_had_create')::BOOLEAN THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 SELECT jsonb_build_object('oid',oid,'owner',nspowner,'acl',to_jsonb(nspacl))
 INTO actual_doc FROM pg_namespace WHERE nspname='public';
 IF actual_doc IS DISTINCT FROM before_doc-'owner_had_create' THEN
  RAISE EXCEPTION 'Supersession schema permissions changed' USING ERRCODE='42501';
 END IF;
END;
$schema_restore538$;
NOTIFY pgrst,'reload schema';
COMMIT;

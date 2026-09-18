-- One-time Sep18 terminal303 baseline-source and new-judge rerun304.
BEGIN;
SET LOCAL TIME ZONE 'UTC';
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='120s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $pin_previous_archive$
DECLARE definition TEXT; current_hash TEXT;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_sep18_rerun302_score_archive_valid303_v1()'::pg_catalog.regprocedure);
 current_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF current_hash='6a3eed441026446276762881be25560f1c8920232a272fab043399eb1b661387' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(definition,$old_submission$ac.round_id='arena-2026-09-18' AND ar.submission_id=ac.submission_id||':r302archive'$old_submission$,'')))<>pg_catalog.length($old_submission$ac.round_id='arena-2026-09-18' AND ar.submission_id=ac.submission_id||':r302archive'$old_submission$)
   OR (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(definition,$old_execute$active.round_id='arena-2026-09-18' AND active.kind='execute' AND active.run_id=archived.value->>'run_id'$old_execute$,'')))<>pg_catalog.length($old_execute$active.round_id='arena-2026-09-18' AND active.kind='execute' AND active.run_id=archived.value->>'run_id'$old_execute$) THEN
   RAISE EXCEPTION 'rerun302 score archive validator seam differs';
  END IF;
  definition:=pg_catalog.replace(definition,$old_submission$ac.round_id='arena-2026-09-18' AND ar.submission_id=ac.submission_id||':r302archive'$old_submission$,$new_submission$((ac.round_id='arena-2026-09-18' AND ac.submission_id<>'baseline-2026-09-18' AND ar.submission_id=ac.submission_id||':r302archive')
     OR (NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive') AND ac.round_id='arena-2026-09-18' AND ac.submission_id='baseline-2026-09-18' AND ar.submission_id='baseline-2026-09-18:r302archive')
     OR (ac.round_id='arena-2026-09-18-rerun303archive' AND ac.submission_id='baseline-2026-09-18:r303archive' AND ar.submission_id='baseline-2026-09-18:r302archive'))$new_submission$);
  definition:=pg_catalog.replace(definition,$old_execute$active.round_id='arena-2026-09-18' AND active.kind='execute' AND active.run_id=archived.value->>'run_id'$old_execute$,$new_execute$((active.round_id='arena-2026-09-18' AND active.submission_id<>'baseline-2026-09-18')
     OR (NOT EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive') AND active.round_id='arena-2026-09-18' AND active.submission_id='baseline-2026-09-18')
     OR (active.round_id='arena-2026-09-18-rerun303archive' AND active.submission_id='baseline-2026-09-18:r303archive')) AND active.kind='execute' AND active.run_id=archived.value->>'run_id'$new_execute$);
  EXECUTE definition;
 ELSIF current_hash<>'ef002bdc9a42a27fc211d2af3c41152de7c460d342bfc12e7a7137e6b127232b' THEN
  RAISE EXCEPTION 'rerun302 score archive validator differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
  'public.lab_arena_sep18_rerun302_score_archive_valid303_v1()'::pg_catalog.regprocedure),
  'sha256'),'hex')<>'ef002bdc9a42a27fc211d2af3c41152de7c460d342bfc12e7a7137e6b127232b' THEN RAISE EXCEPTION 'rerun302 score archive patch differs'; END IF;
END $pin_previous_archive$;
DO $pin_rerun304_scorer$
DECLARE definition TEXT; current_hash TEXT;
BEGIN
 definition:=pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure);
 current_hash:=pg_catalog.encode(extensions.digest(definition,'sha256'),'hex');
 IF current_hash='a22585895de7ea68950b53b65e7ab13bdfd94f2aac95f40a53c84cc49b99cc5b' THEN
  IF (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(definition,$old_guard$  IF p_round_id = 'arena-2026-09-18'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun302archive') THEN$old_guard$,'')))<>pg_catalog.length($old_guard$  IF p_round_id = 'arena-2026-09-18'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun302archive') THEN$old_guard$)
   OR (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(definition,$old_loop$  FOR v_item IN$old_loop$,'')))<>pg_catalog.length($old_loop$  FOR v_item IN$old_loop$)
   OR (pg_catalog.length(definition)-pg_catalog.length(pg_catalog.replace(definition,$old_namespace$      CASE WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun302archive')
           THEN ':rerun303'$old_namespace$,'')))<>pg_catalog.length($old_namespace$      CASE WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun302archive')
           THEN ':rerun303'$old_namespace$) THEN
   RAISE EXCEPTION 'terminal303 scorer seam differs';
  END IF;
  definition:=pg_catalog.replace(definition,$old_guard$  IF p_round_id = 'arena-2026-09-18'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun302archive') THEN$old_guard$,$new_guard$  IF p_round_id = 'arena-2026-09-18'
     AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                 WHERE round_id = 'arena-2026-09-18-rerun302archive')
     AND NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-09-18-rerun303archive') THEN$new_guard$);
  definition:=pg_catalog.replace(definition,$old_loop$  FOR v_item IN$old_loop$,$new_loop$  IF p_round_id = 'arena-2026-09-18' AND EXISTS (
      SELECT 1 FROM public.lab_arena_rounds WHERE round_id = 'arena-2026-09-18-rerun303archive') THEN
    IF v_round.configuration_doc IS DISTINCT FROM
         $rerun304_configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$rerun304_configuration$::JSONB
       OR public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
       OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE
       OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions baseline
          WHERE baseline.round_id=p_round_id AND baseline.submission_id='baseline-2026-09-18'
            AND baseline.source_ref='arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz'
            AND baseline.source_size_bytes=673162
            AND baseline.submission_doc->>'source_sha256'='780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0'
            AND baseline.submission_doc->>'source_commit'='e8572c97f5b69b0bd39ebad8edc47f91df2bfe59') THEN
      RAISE EXCEPTION 'lab_arena_rerun304_frozen_state_invalid'
        USING ERRCODE='22023';
    END IF;
  END IF;
  FOR v_item IN$new_loop$);
  definition:=pg_catalog.replace(definition,$old_namespace$      CASE WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun302archive')
           THEN ':rerun303'$old_namespace$,$new_namespace$      CASE WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun303archive')
           THEN ':rerun304'
           WHEN p_round_id = 'arena-2026-09-18'
                  AND EXISTS (SELECT 1 FROM public.lab_arena_rounds
                    WHERE round_id = 'arena-2026-09-18-rerun302archive')
           THEN ':rerun303'$new_namespace$);
  EXECUTE definition;
 ELSIF current_hash<>'b81eda5ae6bb29673db0a0136d0dd4bcc26c644357ec3a895f7b918a82e3c5ca' THEN
  RAISE EXCEPTION 'terminal303 scorer differs';
 END IF;
 IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
  'public.lab_arena_open_scoring_v2(text,smallint,jsonb)'::pg_catalog.regprocedure),
  'sha256'),'hex')<>'b81eda5ae6bb29673db0a0136d0dd4bcc26c644357ec3a895f7b918a82e3c5ca' THEN RAISE EXCEPTION 'rerun304 scorer patch differs'; END IF;
END $pin_rerun304_scorer$;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun303_score_namespace_guard ON public.lab_arena_runs;
DROP TRIGGER IF EXISTS lab_arena_sep18_cancelled_rerun303_publication_guard ON public.lab_arena_rounds;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_rerun303_archive_valid304_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $archive304$
DECLARE
 archived public.lab_arena_rounds;
 archived_baseline public.lab_arena_submissions;
 derived JSONB; terminal JSONB; terminal_config JSONB; expected_config JSONB;
 baseline_run_hash TEXT; miner_run_hash TEXT;
 baseline_ledger_hash TEXT; miner_ledger_hash TEXT; miner_submission_hash TEXT;
BEGIN
 SELECT * INTO archived FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive';
 IF NOT FOUND THEN RETURN FALSE; END IF;
 derived:=archived.configuration_doc->'archived_execution_judgments';
 terminal_config:=(archived.configuration_doc-'archived_execution_judgments'
  -'archived_reward_basis_hash'-'archived_effective_reward_epoch'
  -'archived_reward_activated_at')||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-18','mode','live','rewards_enabled',TRUE);
 terminal:=(pg_catalog.to_jsonb(archived)-'configuration_doc')||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-18','status','published','configuration_doc',terminal_config,
  'rewards_enabled',TRUE,
  'reward_basis_hash',archived.configuration_doc->'archived_reward_basis_hash',
  'effective_reward_epoch',archived.configuration_doc->'archived_effective_reward_epoch',
  'reward_activated_at',archived.configuration_doc->'archived_reward_activated_at',
  'cancel_reason',NULL);
 SELECT * INTO archived_baseline FROM public.lab_arena_submissions
  WHERE round_id='arena-2026-09-18-rerun303archive' AND submission_id='baseline-2026-09-18:r303archive';
 IF NOT FOUND THEN RETURN FALSE; END IF;
 expected_config:=terminal_config||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-18-rerun303archive','mode','shadow','rewards_enabled',FALSE,
  'archived_reward_basis_hash',terminal->'reward_basis_hash',
  'archived_effective_reward_epoch',terminal->'effective_reward_epoch',
  'archived_reward_activated_at',terminal->'reward_activated_at');
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(r)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') FROM public.lab_arena_runs r WHERE r.round_id='arena-2026-09-18-rerun303archive' AND r.submission_id='baseline-2026-09-18:r303archive' INTO baseline_run_hash;
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((doc)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex') FROM (SELECT ((pg_catalog.to_jsonb(e)-'per_icp_score'-'qualification_doc'-'updated_at')||pg_catalog.jsonb_build_object('per_icp_score',d.value->'per_icp_score','qualification_doc',d.value->'qualification_doc','updated_at',d.value->'updated_at')) doc,e.run_id FROM public.lab_arena_runs e JOIN LATERAL (SELECT value FROM pg_catalog.jsonb_array_elements(derived) x(value) WHERE value->>'run_id'=e.run_id) d ON TRUE WHERE e.round_id='arena-2026-09-18' AND e.submission_id<>'baseline-2026-09-18' AND e.kind='execute' UNION ALL SELECT (pg_catalog.to_jsonb(s)-'round_id'-'submission_id')||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18','submission_id',pg_catalog.left(s.submission_id,pg_catalog.length(s.submission_id)-12)) doc,s.run_id FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-18-rerun303archive' AND s.submission_id<>'baseline-2026-09-18:r303archive' AND s.kind='score') rows INTO miner_run_hash;
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(l)-'round_id'-'submission_id')::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') FROM public.lab_arena_ledger l WHERE l.round_id='arena-2026-09-18-rerun303archive' AND l.submission_id='baseline-2026-09-18:r303archive' INTO baseline_ledger_hash;
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((doc)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex') FROM (SELECT pg_catalog.to_jsonb(l) doc,l.entry_id FROM public.lab_arena_ledger l WHERE l.round_id='arena-2026-09-18' AND l.submission_id<>'baseline-2026-09-18' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id='arena-2026-09-18' AND s.kind='score' AND s.run_id=l.run_id) UNION ALL SELECT (pg_catalog.to_jsonb(l)-'round_id'-'submission_id')||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18','submission_id',pg_catalog.left(l.submission_id,pg_catalog.length(l.submission_id)-12)) doc,l.entry_id FROM public.lab_arena_ledger l WHERE l.round_id='arena-2026-09-18-rerun303archive' AND l.submission_id<>'baseline-2026-09-18:r303archive') rows INTO miner_ledger_hash;
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(((pg_catalog.to_jsonb(s)-'round_id'-'submission_id')||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18','submission_id',pg_catalog.left(s.submission_id,pg_catalog.length(s.submission_id)-12)))::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-18-rerun303archive' AND s.submission_id<>'baseline-2026-09-18:r303archive' INTO miner_submission_hash;
 RETURN pg_catalog.encode(extensions.digest(terminal::TEXT,'sha256'),'hex')='c51c2f4a259ab68ea8615e644deae95e9fc4fd775bee951a2f30004401f8ac5b'
  AND pg_catalog.encode(extensions.digest(((pg_catalog.to_jsonb(archived_baseline)-'round_id'-'submission_id')||
    pg_catalog.jsonb_build_object('round_id','arena-2026-09-18','submission_id','baseline-2026-09-18'))::TEXT,'sha256'),'hex')='8c33829f0a3899324099570930178425d608809da32eb466301fbdc562a54901'
  AND (pg_catalog.to_jsonb(archived)-'configuration_doc') IS NOT DISTINCT FROM
   ((terminal||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun303archive','status','cancelled',
    'rewards_enabled',FALSE,'reward_basis_hash',NULL,'effective_reward_epoch',NULL,
    'reward_activated_at',NULL,'cancel_reason','authorized_sep18_terminal303_newjudge_archive'))-'configuration_doc')
  AND (archived.configuration_doc-'archived_execution_judgments') IS NOT DISTINCT FROM expected_config
  AND pg_catalog.jsonb_typeof(derived)='array' AND pg_catalog.jsonb_array_length(derived)=100
  AND (SELECT pg_catalog.count(DISTINCT value->>'run_id') FROM pg_catalog.jsonb_array_elements(derived))=100
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18-rerun303archive')=5
  AND miner_submission_hash='sha256:40318c429446ccfc8e2ad48a1bec6ce640e526f453062bd9e4302c0ca5d8a221'
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18-rerun303archive')=120
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18-rerun303archive' AND kind='execute' AND submission_id<>'baseline-2026-09-18:r303archive')
  AND baseline_run_hash='sha256:bdf48d90c1c6510af31f1d6cc50c40a906fcaae8dd4b8bb4d29b80a9c52d7bad' AND miner_run_hash='sha256:ca5f4dc4c3fb7cf3cd6ed080e8b45c06cda8bdd4eafa1b1d9aef9d37d07acbe5'
  AND baseline_ledger_hash='sha256:e854fd6c45206e7330907abd07ff34ffc67f1284645d3db4c3cca7c3d0465253' AND miner_ledger_hash='sha256:6c9e13d8e65f209d44de95a78b8c5d9ee4ef826a511ed2d4d309bcc1614dd8bd'
  AND public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS TRUE;
END $archive304$;
ALTER FUNCTION public.lab_arena_sep18_rerun303_archive_valid304_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_rerun303_archive_valid304_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_newjudge_rerun304_active_valid_v1()
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $active304$
DECLARE miner_submission_hash TEXT;
BEGIN
 SELECT 'sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest((pg_catalog.to_jsonb(s))::TEXT,'sha256'),'hex'),'' ORDER BY submission_id),''),'sha256'),'hex') FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-18' AND s.submission_id<>'baseline-2026-09-18' INTO miner_submission_hash;
 RETURN public.lab_arena_sep18_rerun303_archive_valid304_v1() IS TRUE
  AND (SELECT configuration_doc FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18') IS NOT DISTINCT FROM
   $configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18')=5
  AND miner_submission_hash='sha256:40318c429446ccfc8e2ad48a1bec6ce640e526f453062bd9e4302c0ca5d8a221'
  AND EXISTS(SELECT 1 FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18'
    AND source_ref='arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz' AND source_size_bytes=673162
    AND submission_doc->>'source_ref'='arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz'
    AND (submission_doc->>'source_size_bytes')::BIGINT=673162
    AND submission_doc->>'source_sha256'='780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0'
    AND submission_doc->>'source_commit'='e8572c97f5b69b0bd39ebad8edc47f91df2bfe59')
  AND (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id<>'baseline-2026-09-18' AND kind='execute' AND status='accepted' AND terminal_cause='accepted')=80
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id<>'baseline-2026-09-18' AND kind='execute' AND output_ref IS NULL)
  AND (SELECT pg_catalog.count(DISTINCT assignment_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' AND kind='execute' AND assignment_id LIKE '%:rerun304')=20
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' AND kind='execute' AND assignment_id NOT LIKE '%:rerun304')
  AND ((SELECT status FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18')<>'stage1'
    OR ((SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' AND kind='execute' AND status='pending')=20
      AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND submission_id<>'baseline-2026-09-18' AND kind='execute' AND (per_icp_score IS NOT NULL OR qualification_doc IS NOT NULL))))
  AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND assignment_id NOT LIKE '%:score:rerun304');
END $active304$;
ALTER FUNCTION public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_prepare_sep18_newjudge_rerun304_v1(
 p_source_size_bytes BIGINT,p_source_sha256 TEXT,p_source_commit TEXT,
 p_bank_sha256 TEXT,p_schedule JSONB,p_scorer_digest TEXT,p_scorer_reference TEXT)
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,public,extensions AS $prepare304$
DECLARE
 active_round public.lab_arena_rounds; active_baseline public.lab_arena_submissions;
 terminal JSONB; baseline JSONB; derived JSONB; archive_config JSONB; new_participants JSONB;
 before_other JSONB; after_other JSONB; active_count BIGINT; unsettled BIGINT;
 moved_scores BIGINT; moved_score_ledger BIGINT; moved_baseline_runs BIGINT;
 moved_baseline_ledger BIGINT; cleared_miners BIGINT; archived_submissions BIGINT;
 position INTEGER; stage SMALLINT;
 assignment TEXT;
BEGIN
 PERFORM pg_catalog.set_config('lock_timeout','5s',TRUE);
 PERFORM pg_catalog.set_config('statement_timeout','120s',TRUE);
 IF p_source_size_bytes<>673162
  OR p_source_sha256<>'780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0' OR p_source_commit<>'e8572c97f5b69b0bd39ebad8edc47f91df2bfe59'
  OR p_bank_sha256<>'6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91' OR p_schedule IS DISTINCT FROM $schedule${"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$schedule$::JSONB
  OR p_scorer_digest<>'sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c'
  OR p_scorer_reference<>'493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c' THEN
  RAISE EXCEPTION 'rerun304 source, judge, bank, or schedule differs' USING ERRCODE='22023';
 END IF;
 PERFORM pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('arena-2026-09-18-newjudge-rerun304',0));
 LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_submissions IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_runs IN ACCESS EXCLUSIVE MODE;
 LOCK TABLE public.lab_arena_ledger IN ACCESS EXCLUSIVE MODE;
 SELECT * INTO STRICT active_round FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18' FOR UPDATE;
 SELECT * INTO STRICT active_baseline FROM public.lab_arena_submissions WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' FOR UPDATE;
 IF EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive') THEN
  IF public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
     OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE THEN
   RAISE EXCEPTION 'existing rerun304 differs' USING ERRCODE='55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status','existing','round_id','arena-2026-09-18',
   'fresh_baseline_execute_count',20,'retained_miner_execute_count',80,
   'score_namespace','score:rerun304');
 END IF;
 IF (p_schedule->>'benchmark_deadline')::TIMESTAMPTZ<=pg_catalog.clock_timestamp() THEN
  RAISE EXCEPTION 'rerun304 admission window closed' USING ERRCODE='22023';
 END IF;
 SELECT pg_catalog.count(*) INTO active_count FROM public.lab_arena_runs
  WHERE round_id='arena-2026-09-18' AND status IN('pending','leased','submitted');
 SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s
  CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
  CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(s.submission_id,k.kind,NULL) state)c
  WHERE s.round_id='arena-2026-09-18' AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
   OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
 terminal:=pg_catalog.to_jsonb(active_round); baseline:=pg_catalog.to_jsonb(active_baseline);
 IF pg_catalog.encode(extensions.digest(terminal::TEXT,'sha256'),'hex')<>'c51c2f4a259ab68ea8615e644deae95e9fc4fd775bee951a2f30004401f8ac5b'
  OR pg_catalog.encode(extensions.digest(baseline::TEXT,'sha256'),'hex')<>'8c33829f0a3899324099570930178425d608809da32eb466301fbdc562a54901'
  OR active_count<>0 OR unsettled<>0
  OR public.lab_arena_sep18_rerun302_score_archive_valid303_v1() IS NOT TRUE
  OR public.lab_arena_sep18_cancelled_rerun303_active_frozen_valid_v1() IS NOT TRUE
  OR active_round.status<>'published' OR active_round.reward_activated_at IS NULL
  OR active_round.publication_doc#>>'{king_decision,outcome}'<>'no_king'
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute' AND status='accepted' AND terminal_cause='accepted')<>100
  OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun303')<>100
  OR (SELECT pg_catalog.count(DISTINCT scored_run_id) FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun303')<>100
  OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='score' AND status NOT IN('accepted','failed')) THEN
  RAISE EXCEPTION 'terminal303 preimage differs' USING ERRCODE='55000';
 END IF;
 SELECT pg_catalog.jsonb_agg(pg_catalog.jsonb_build_object(
  'run_id',run_id,'per_icp_score',per_icp_score,'qualification_doc',qualification_doc,
  'updated_at',updated_at)
  ORDER BY run_id) INTO derived FROM public.lab_arena_runs WHERE round_id='arena-2026-09-18' AND kind='execute';
 IF pg_catalog.jsonb_array_length(derived)<>100 THEN RAISE EXCEPTION 'terminal303 derived judgment count differs'; END IF;
 SELECT pg_catalog.jsonb_agg(CASE WHEN item->>'submission_id'='baseline-2026-09-18' THEN
  item||pg_catalog.jsonb_build_object('source_ref','arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz',
   'source_size_bytes',673162) ELSE item END ORDER BY ordinal)
  INTO new_participants FROM pg_catalog.jsonb_array_elements(terminal->'participants') WITH ORDINALITY entries(item,ordinal);
 archive_config:=(terminal->'configuration_doc')||pg_catalog.jsonb_build_object(
  'round_id','arena-2026-09-18-rerun303archive','mode','shadow','rewards_enabled',FALSE,
  'archived_reward_basis_hash',terminal->'reward_basis_hash',
  'archived_effective_reward_epoch',terminal->'effective_reward_epoch',
  'archived_reward_activated_at',terminal->'reward_activated_at',
  'archived_execution_judgments',derived);
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds r WHERE round_id<>'arena-2026-09-18'),
  'submissions',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id,submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions r WHERE round_id<>'arena-2026-09-18'),
  'runs',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r WHERE round_id<>'arena-2026-09-18'),
  'ledger',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger r WHERE round_id<>'arena-2026-09-18'),
  'weights',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY network,netuid,epoch),''),'sha256'),'hex')) FROM public.lab_arena_accepted_weight_states r)) INTO before_other;
 ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs DISABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_ledger DISABLE TRIGGER USER;
 INSERT INTO public.lab_arena_rounds(round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers) SELECT round_id,status,status_generation,stage_generation,configuration_doc,rewards_enabled,participants,benchmark_ref,evaluation_date,stage1_scoring_plan_doc,stage2_scoring_plan_doc,finalists,publication_doc,king_outcome,king_hotkey,king_start_epoch,effective_reward_epoch,reward_basis_hash,reward_basis_doc,signing_key_doc,reward_activated_at,cancel_reason,published_at,created_at,updated_at,promotion_required,promotion_doc,baseline_promoted_at,icp_set_date,confirmation_bank_ref,confirmation_bank_hash,confirmation_cohort,stage3_scoring_plan_doc,champion_funding_frozen,champion_submission_id,champion_hotkey,champion_fallback_providers
  FROM pg_catalog.jsonb_populate_record(NULL::public.lab_arena_rounds,
   terminal||pg_catalog.jsonb_build_object('round_id','arena-2026-09-18-rerun303archive','status','cancelled',
    'configuration_doc',archive_config,'rewards_enabled',FALSE,'reward_basis_hash',NULL,
    'effective_reward_epoch',NULL,'reward_activated_at',NULL,
    'cancel_reason','authorized_sep18_terminal303_newjudge_archive'));
 INSERT INTO public.lab_arena_submissions SELECT (pg_catalog.jsonb_populate_record(
  NULL::public.lab_arena_submissions,pg_catalog.to_jsonb(s)||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-18-rerun303archive','submission_id',s.submission_id||':r303archive'))).*
  FROM public.lab_arena_submissions s WHERE s.round_id='arena-2026-09-18';
 GET DIAGNOSTICS archived_submissions=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-18-rerun303archive',submission_id=submission_id||':r303archive'
  WHERE round_id='arena-2026-09-18' AND kind='score'; GET DIAGNOSTICS moved_scores=ROW_COUNT;
 UPDATE public.lab_arena_ledger l SET round_id='arena-2026-09-18-rerun303archive',submission_id=submission_id||':r303archive'
  WHERE round_id='arena-2026-09-18' AND EXISTS(SELECT 1 FROM public.lab_arena_runs s
   WHERE s.round_id='arena-2026-09-18-rerun303archive' AND s.kind='score' AND s.run_id=l.run_id);
 GET DIAGNOSTICS moved_score_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET round_id='arena-2026-09-18-rerun303archive',submission_id=submission_id||':r303archive'
  WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18' AND kind='execute';
 GET DIAGNOSTICS moved_baseline_runs=ROW_COUNT;
 UPDATE public.lab_arena_ledger SET round_id='arena-2026-09-18-rerun303archive',submission_id=submission_id||':r303archive'
  WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18';
 GET DIAGNOSTICS moved_baseline_ledger=ROW_COUNT;
 UPDATE public.lab_arena_runs SET per_icp_score=NULL,qualification_doc=NULL
  WHERE round_id='arena-2026-09-18' AND submission_id<>'baseline-2026-09-18' AND kind='execute';
 GET DIAGNOSTICS cleared_miners=ROW_COUNT;
 UPDATE public.lab_arena_submissions SET source_ref='arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz',
  source_size_bytes=673162,submission_doc=COALESCE(submission_doc,'{}'::JSONB)||
  pg_catalog.jsonb_build_object('source_ref','arena/arena-2026-09-18/sources/baseline-2026-09-18-rerun304-latest.tar.gz',
   'source_size_bytes',673162,'source_sha256','780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0',
   'source_commit','e8572c97f5b69b0bd39ebad8edc47f91df2bfe59')
  WHERE round_id='arena-2026-09-18' AND submission_id='baseline-2026-09-18';
 UPDATE public.lab_arena_rounds SET status='stage1',status_generation=status_generation+1,
  stage_generation=stage_generation+1,configuration_doc=$configuration${"banned_hotkeys":[],"baseline_hotkey":"5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9","baseline_source_url":"https://github.com/leadpoet/leadpoet-sales-agent/archive/refs/heads/lab.tar.gz","benchmark_disclosure_policy":"after_scoring_day2_v1","call_quotas":{"deepline":30,"openrouter":200,"scrapingdog":30},"checkpoint_deadline_policy":"atomic_checkpoint_45m_v1","companies_per_icp":5,"contact_policy":"contacts_v1","cost_per_company_microusd":800000,"execution_cap_microusd":80000000,"execution_icp_cap_microusd":4000000,"finalist_count":10,"icp_wall_clock_seconds":2700,"integrity_policy":"arena_integrity_v1","intent_details_policy":"intent_details_v1","lease_ttl_seconds":3600,"max_attempts_per_assignment":2,"max_challengers":20,"mode":"live","netuid":71,"network_name":"finney","parallel_twenty_icp_execution":true,"providers":["scrapingdog","deepline","openrouter"],"reward_constants":{"eligibility_max_epochs":45,"epochs_per_reward_week":140,"king_pool_share_percent_by_week":[100,80,60,40,20],"pool_basis":"total_emissions","pool_percent":25},"rewards_enabled":true,"round_id":"arena-2026-09-18","runner_hotkeys":["5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9"],"runner_slot_ceiling":20,"schedule":{"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"},"schema_version":"leadpoet.lab_arena.round_configuration.v1","scorer_image_digest":"sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_image_reference":"493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c","scorer_policy":{"company_cap_rule":"icp_max_companies","employee_bucket_rule":"lab_relaxed_buckets","env_bindings":{"RESEARCH_LAB_EVAL_CANDIDATE_CONCURRENCY":"1","RESEARCH_LAB_EVAL_CAPPED_TOP5_SCORE":"0","RESEARCH_LAB_EVAL_FP_PENALTY_POINTS":"10","RESEARCH_LAB_EVAL_FP_UNVERIFIED_PRIMARY_PENALTY":"10","RESEARCH_LAB_EVAL_MAX_SCORED_COMPANIES":"0","RESEARCH_LAB_EVAL_PROVIDER_FLAKE_RETRY":"1","RESEARCH_LAB_EVAL_TIMEOUT_LATCH_LEGACY":"0","RESEARCH_LAB_EVAL_WORK_CONSERVING":"0","RESEARCH_LAB_GLOBAL_SCORING_QUEUE":"0","RESEARCH_LAB_INCONTAINER_TRACE_KMS_KEY_ID":"","RESEARCH_LAB_INCONTAINER_TRACE_S3_PREFIX":"","RESEARCH_LAB_OPENROUTER_TRACE_CAPTURE":"0"},"fp_penalty_icp_floor":0,"fp_penalty_points":10,"fp_unverified_primary_penalty_points":10,"intent_details_policy":"intent_details_v1","judge_models":{"company_fit_reverification":"perplexity/sonar","intent_precheck":"google/gemini-2.5-flash-lite","intent_signal_judge":"anthropic/claude-sonnet-4.5","intent_three_stage_stage3":"perplexity/sonar-pro","intent_verification":"openai/gpt-4o-mini","role_batch_check":"google/gemini-2.5-flash"},"max_scored_companies":0,"pre_slice_rule":"first_n_model_order","provider_profile":"lab_arena","schema_version":"leadpoet.lab_arena.scorer_policy.v1","scoring_adapter_version":"qualification_contacts_v3"},"scoring_call_quotas":{"deepline":40,"openrouter":120,"scrapingdog":150},"scoring_cap_microusd":50000000,"scoring_wall_clock_seconds":900,"sourcing_cost_eligibility_policy":"successful_calls_per_icp_v1","stage_1_icp_count":10,"stage_2_icp_count":10}$configuration$::JSONB,
  participants=new_participants,stage1_scoring_plan_doc=NULL,stage2_scoring_plan_doc=NULL,
  stage3_scoring_plan_doc=NULL,finalists=NULL,publication_doc=NULL,published_at=NULL,
  cancel_reason=NULL,updated_at=pg_catalog.clock_timestamp() WHERE round_id='arena-2026-09-18';
 FOR position IN 0..19 LOOP stage:=CASE WHEN position<10 THEN 1 ELSE 2 END;
  assignment:='arena-2026-09-18:baseline-2026-09-18:'||stage::TEXT||':'||position::TEXT||':rerun304';
  INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,
   stage,icp_position,attempt,kind,status,stage_generation)
  VALUES(assignment||':1',assignment,'arena-2026-09-18','baseline-2026-09-18',active_baseline.miner_hotkey,
   stage,position,1,'execute','pending',active_round.stage_generation+1);
 END LOOP;
 ALTER TABLE public.lab_arena_ledger ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_runs ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_submissions ENABLE TRIGGER USER;
 ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER USER;
 SELECT pg_catalog.jsonb_build_object(
  'rounds',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id),''),'sha256'),'hex')) FROM public.lab_arena_rounds r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun303archive')),
  'submissions',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY round_id,submission_id),''),'sha256'),'hex')) FROM public.lab_arena_submissions r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun303archive')),
  'runs',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY run_id),''),'sha256'),'hex')) FROM public.lab_arena_runs r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun303archive')),
  'ledger',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY entry_id),''),'sha256'),'hex')) FROM public.lab_arena_ledger r WHERE round_id NOT IN('arena-2026-09-18','arena-2026-09-18-rerun303archive')),
  'weights',(SELECT pg_catalog.jsonb_build_object('count',pg_catalog.count(*),'sha256','sha256:'||pg_catalog.encode(extensions.digest(COALESCE(pg_catalog.string_agg(pg_catalog.encode(extensions.digest(pg_catalog.to_jsonb(r)::TEXT,'sha256'),'hex'),'' ORDER BY network,netuid,epoch),''),'sha256'),'hex')) FROM public.lab_arena_accepted_weight_states r)) INTO after_other;
 IF before_other IS DISTINCT FROM after_other OR archived_submissions<>5
  OR moved_baseline_runs<>20 OR cleared_miners<>80
  OR moved_scores<>100
  OR public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
  OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE THEN
  RAISE EXCEPTION 'rerun304 atomic preservation differs' USING ERRCODE='55000';
 END IF;
 RETURN pg_catalog.jsonb_build_object('status','prepared','round_id','arena-2026-09-18',
  'fresh_baseline_execute_count',20,'retained_miner_execute_count',80,
  'archived_score_attempt_count',moved_scores,'archived_score_ledger_count',moved_score_ledger,
  'archived_baseline_ledger_count',moved_baseline_ledger,'score_namespace','score:rerun304');
END $prepare304$;
ALTER FUNCTION public.lab_arena_prepare_sep18_newjudge_rerun304_v1(BIGINT,TEXT,TEXT,TEXT,JSONB,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_prepare_sep18_newjudge_rerun304_v1(BIGINT,TEXT,TEXT,TEXT,JSONB,TEXT,TEXT) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_prepare_sep18_newjudge_rerun304_v1(BIGINT,TEXT,TEXT,TEXT,JSONB,TEXT,TEXT) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_newjudge_rerun304_score_namespace_guard_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $namespace304$
BEGIN
 IF NEW.round_id='arena-2026-09-18' AND NEW.kind='score'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive')
  AND NEW.assignment_id IS DISTINCT FROM NEW.round_id||':'||NEW.submission_id||':'||
   NEW.stage::TEXT||':'||NEW.icp_position::TEXT||':score:rerun304' THEN
  RAISE EXCEPTION 'Sep18 rerun304 scoring requires exact namespace' USING ERRCODE='55000';
 END IF; RETURN NEW;
END $namespace304$;
ALTER FUNCTION public.lab_arena_sep18_newjudge_rerun304_score_namespace_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_newjudge_rerun304_score_namespace_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_newjudge_rerun304_score_namespace_guard ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_sep18_newjudge_rerun304_score_namespace_guard BEFORE INSERT ON public.lab_arena_runs
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_sep18_newjudge_rerun304_score_namespace_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1()
RETURNS TRIGGER LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog,public AS $publication304$
DECLARE archived public.lab_arena_rounds; terminal JSONB; terminal_config JSONB; unsettled BIGINT;
BEGIN
 IF NEW.round_id='arena-2026-09-18' AND OLD.status='scored' AND NEW.status='published'
  AND EXISTS(SELECT 1 FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive') THEN
  SELECT * INTO STRICT archived FROM public.lab_arena_rounds WHERE round_id='arena-2026-09-18-rerun303archive';
  terminal_config:=(archived.configuration_doc-'archived_execution_judgments'
   -'archived_reward_basis_hash'-'archived_effective_reward_epoch'
   -'archived_reward_activated_at')||pg_catalog.jsonb_build_object(
    'round_id','arena-2026-09-18','mode','live','rewards_enabled',TRUE);
  terminal:=(pg_catalog.to_jsonb(archived)-'configuration_doc')||pg_catalog.jsonb_build_object(
   'round_id','arena-2026-09-18','status','published','configuration_doc',terminal_config,
   'rewards_enabled',TRUE,
   'reward_basis_hash',archived.configuration_doc->'archived_reward_basis_hash',
   'effective_reward_epoch',archived.configuration_doc->'archived_effective_reward_epoch',
   'reward_activated_at',archived.configuration_doc->'archived_reward_activated_at',
   'cancel_reason',NULL);
  SELECT pg_catalog.count(*) INTO unsettled FROM public.lab_arena_submissions s
   CROSS JOIN(VALUES('execute'::TEXT),('score'::TEXT)) k(kind)
   CROSS JOIN LATERAL(SELECT public.lab_arena__successful_call_cost_state(s.submission_id,k.kind,NULL) state)c
   WHERE s.round_id=NEW.round_id AND (COALESCE((c.state->>'inflight_calls')::BIGINT,-1)<>0
    OR COALESCE((c.state->>'success_unresolved_calls')::BIGINT,-1)<>0);
  IF pg_catalog.encode(extensions.digest(terminal::TEXT,'sha256'),'hex')<>'c51c2f4a259ab68ea8615e644deae95e9fc4fd775bee951a2f30004401f8ac5b'
   OR public.lab_arena_sep18_rerun303_archive_valid304_v1() IS NOT TRUE
   OR public.lab_arena_sep18_newjudge_rerun304_active_valid_v1() IS NOT TRUE
   OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=NEW.round_id AND kind='execute' AND status='accepted' AND terminal_cause='accepted')<>100
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs WHERE round_id=NEW.round_id AND kind='score' AND status NOT IN('accepted','failed'))
   OR (SELECT pg_catalog.count(*) FROM public.lab_arena_runs WHERE round_id=NEW.round_id AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun304')<>100
   OR (SELECT pg_catalog.count(DISTINCT scored_run_id) FROM public.lab_arena_runs WHERE round_id=NEW.round_id AND kind='score' AND status='accepted' AND terminal_cause='accepted' AND assignment_id LIKE '%:score:rerun304')<>100
   OR EXISTS(SELECT 1 FROM public.lab_arena_runs e WHERE e.round_id=NEW.round_id AND e.kind='execute' AND NOT EXISTS(SELECT 1 FROM public.lab_arena_runs s WHERE s.round_id=e.round_id AND s.kind='score' AND s.status='accepted' AND s.terminal_cause='accepted' AND s.scored_run_id=e.run_id AND s.assignment_id LIKE '%:score:rerun304'))
   OR unsettled<>0
   OR NEW.reward_basis_hash IS DISTINCT FROM terminal->>'reward_basis_hash'
   OR NEW.reward_basis_doc IS DISTINCT FROM terminal->'reward_basis_doc'
   OR NEW.signing_key_doc IS DISTINCT FROM terminal->'signing_key_doc'
   OR NEW.effective_reward_epoch IS DISTINCT FROM (terminal->>'effective_reward_epoch')::BIGINT
   OR NEW.reward_activated_at IS DISTINCT FROM (terminal->>'reward_activated_at')::TIMESTAMPTZ
   OR NEW.king_outcome IS DISTINCT FROM terminal->>'king_outcome'
   OR NEW.king_hotkey IS DISTINCT FROM terminal->>'king_hotkey'
   OR NEW.king_start_epoch IS DISTINCT FROM (terminal->>'king_start_epoch')::BIGINT
   OR NEW.promotion_required IS DISTINCT FROM (terminal->>'promotion_required')::BOOLEAN
   OR NEW.promotion_doc IS DISTINCT FROM NULLIF(terminal->'promotion_doc','null'::JSONB)
   OR NEW.baseline_promoted_at IS DISTINCT FROM (terminal->>'baseline_promoted_at')::TIMESTAMPTZ
   OR NEW.champion_funding_frozen IS DISTINCT FROM (terminal->>'champion_funding_frozen')::BOOLEAN
   OR NEW.champion_submission_id IS DISTINCT FROM terminal->>'champion_submission_id'
   OR NEW.champion_hotkey IS DISTINCT FROM terminal->>'champion_hotkey'
   OR NEW.champion_fallback_providers IS DISTINCT FROM ARRAY(SELECT pg_catalog.jsonb_array_elements_text(COALESCE(terminal->'champion_fallback_providers','[]'::JSONB)))
   OR NEW.publication_doc#>>'{king_decision,outcome}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,outcome}'
   OR NEW.publication_doc#>>'{king_decision,king_submission_id}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,king_submission_id}'
   OR NEW.publication_doc#>>'{king_decision,king_hotkey}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,king_hotkey}'
   OR NEW.publication_doc#>>'{king_decision,winner_submission_id}' IS DISTINCT FROM terminal#>>'{publication_doc,king_decision,winner_submission_id}' THEN
   RAISE EXCEPTION 'Sep18 rerun304 publication conflicts with sealed completion or reward authority' USING ERRCODE='55000';
  END IF;
 END IF; RETURN NEW;
END $publication304$;
ALTER FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1() FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
DROP TRIGGER IF EXISTS lab_arena_sep18_newjudge_rerun304_publication_guard ON public.lab_arena_rounds;
CREATE TRIGGER lab_arena_sep18_newjudge_rerun304_publication_guard BEFORE UPDATE ON public.lab_arena_rounds
 FOR EACH ROW EXECUTE FUNCTION public.lab_arena_sep18_newjudge_rerun304_publication_guard_v1();
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
SELECT public.lab_arena_prepare_sep18_newjudge_rerun304_v1(673162,'780d959564bd075fd4ead854c2878f9f4581cebdf73d1a83f9d4d7a3b4252ef0','e8572c97f5b69b0bd39ebad8edc47f91df2bfe59','6999fdd7bcc09f95943f29127471659d966a86bdb370863f4d895376863acf91',$schedule${"benchmark_deadline":"2026-09-19T01:00:00Z","final_scoring_close":"2026-09-19T11:30:04Z","publication_deadline":"2026-09-19T11:30:05Z","stage_1_close":"2026-09-19T01:00:02Z","stage_1_scoring_close":"2026-09-19T06:15:02Z","stage_1_start":"2026-09-19T01:00:01Z","stage_2_close":"2026-09-19T06:15:04Z","stage_2_start":"2026-09-19T06:15:03Z","submission_cutoff":"2026-09-18T00:00:00Z","submission_open":"2026-09-17T00:00:00Z"}$schedule$::JSONB,'sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c','493765492819.dkr.ecr.us-east-1.amazonaws.com/leadpoet/sourcing-model@sha256:085e6b586ccac28ca7ec2c8996dd871f1a53050e843398b514f67d8c798aae5c');
NOTIFY pgrst,'reload schema';
COMMIT;

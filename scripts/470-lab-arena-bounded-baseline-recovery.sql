-- One infrastructure recovery attempt in the existing remaining-day windows.
-- Original assignments, frozen artifacts, prior results and liabilities stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

-- Derived lineage needs no grant table: canonical attempt 3 consumes the one
-- exception permanently. A judge of that recovered execution uses normal 1/2.
CREATE OR REPLACE FUNCTION public.lab_arena__baseline_recovery_run_v1(p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $recovery_run$
DECLARE v_round public.lab_arena_rounds; v_source public.lab_arena_runs;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=p_run.round_id;
  IF NOT FOUND OR p_run.stage<>1
    OR p_run.submission_id IS DISTINCT FROM 'baseline-'||regexp_replace(p_run.round_id,'^arena-','')
    OR p_run.miner_hotkey IS DISTINCT FROM v_round.configuration_doc->>'baseline_hotkey'
    OR (SELECT count(*) FROM jsonb_array_elements(v_round.participants) p WHERE p->'is_king'='true'::JSONB)<>1
    OR v_round.configuration_doc->>'execution_sequence_policy' IS DISTINCT FROM 'baseline_scored_first_v1'
    OR v_round.configuration_doc->>'sourcing_cost_eligibility_policy' IS DISTINCT FROM 'successful_calls_per_icp_v1'
    OR v_round.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
    OR v_round.configuration_doc ? 'company_quality_policy'
    OR (v_round.configuration_doc->'scorer_policy') ? 'company_quality_policy'
    OR COALESCE(v_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}','') NOT IN ('qualification_integrity_v2','qualification_contacts_v3')
    OR v_round.configuration_doc->>'max_attempts_per_assignment' IS DISTINCT FROM '2'
    OR p_run.icp_position NOT BETWEEN 0 AND
      (v_round.configuration_doc->>'stage_1_icp_count')::INTEGER+
      (v_round.configuration_doc->>'stage_2_icp_count')::INTEGER-1
    OR p_run.assignment_id IS DISTINCT FROM p_run.round_id||':'||p_run.submission_id||':1:'||p_run.icp_position::TEXT||
      (CASE p_run.kind WHEN 'score' THEN ':score' ELSE '' END)
    OR p_run.run_id IS DISTINCT FROM p_run.assignment_id||':'||p_run.attempt::TEXT
    OR p_run.champion_restart_required
    OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
      WHERE s.submission_id=p_run.submission_id AND s.round_id=p_run.round_id
        AND s.miner_hotkey=p_run.miner_hotkey AND s.is_king AND s.status='frozen'
        AND EXISTS (SELECT 1 FROM jsonb_array_elements(v_round.participants) p
          WHERE p->>'submission_id'=s.submission_id AND p->>'miner_hotkey'=s.miner_hotkey
            AND p->'is_king'='true'::JSONB)) THEN RETURN FALSE; END IF;
  IF p_run.kind='score' THEN
    SELECT * INTO v_source FROM public.lab_arena_runs WHERE run_id=p_run.scored_run_id;
    IF NOT FOUND OR v_source.kind<>'execute' OR v_source.status<>'accepted'
      OR v_source.terminal_cause IS DISTINCT FROM 'accepted' OR v_source.output_ref IS NULL
      OR (v_source.round_id,v_source.submission_id,v_source.miner_hotkey,v_source.stage,v_source.icp_position)
        IS DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position)
      OR v_source.assignment_id IS DISTINCT FROM p_run.round_id||':'||p_run.submission_id||':1:'||p_run.icp_position::TEXT
      OR v_source.run_id IS DISTINCT FROM v_source.assignment_id||':'||v_source.attempt::TEXT THEN RETURN FALSE; END IF;
    IF p_run.attempt IN (1,2) AND v_source.attempt=3 THEN
      RETURN public.lab_arena__baseline_recovery_run_v1(v_source);
    END IF;
    IF v_source.attempt NOT IN (1,2) THEN RETURN FALSE; END IF;
  END IF;
  RETURN p_run.attempt=3 AND
    (SELECT count(*)=2 AND bool_and(COALESCE(parent.status='failed'
      AND parent.terminal_cause IN (
        CASE p_run.kind WHEN 'execute' THEN 'provider_error' ELSE 'judge_error' END,
        CASE p_run.kind WHEN 'execute' THEN 'lease_expired' ELSE 'judge_timeout' END,
        'lease_expired') AND NOT parent.champion_restart_required
      AND (parent.round_id,parent.submission_id,parent.miner_hotkey,parent.stage,parent.icp_position,parent.kind)
        IS NOT DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage,p_run.icp_position,p_run.kind)
      AND parent.run_id=parent.assignment_id||':'||parent.attempt::TEXT
      AND (parent.terminal_cause='lease_expired' OR (jsonb_typeof(parent.result_doc)='object' AND parent.result_doc->>'terminal_status'=parent.terminal_cause))
      AND (p_run.kind<>'execute' OR parent.output_ref IS NULL)
      AND (p_run.kind<>'score' OR parent.scored_run_id=p_run.scored_run_id),FALSE))
     FROM public.lab_arena_runs parent WHERE parent.assignment_id=p_run.assignment_id AND parent.attempt IN (1,2))
    AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs other
      WHERE other.assignment_id=p_run.assignment_id AND other.attempt>3);
END;
$recovery_run$;
ALTER FUNCTION public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_recover_baseline_v1(p_round_id TEXT,p_work_items JSONB DEFAULT '[]')
RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path=pg_catalog,public AS $recover_baseline$
DECLARE v_round public.lab_arena_rounds; v_parent public.lab_arena_runs;
  v_new public.lab_arena_runs; v_source public.lab_arena_runs;
  v_item JSONB; v_created INTEGER:=0; v_state JSONB; v_cache public.lab_arena_judgment_cache;
BEGIN
  IF jsonb_typeof(p_work_items) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_recovery_input_invalid' USING ERRCODE='22023'; END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('lab-arena-claim-control',0));
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id=p_round_id FOR UPDATE;
  IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE='P0002'; END IF;
  IF v_round.status NOT IN ('stage2','stage2_closed','stage2_scoring')
    OR clock_timestamp()>=(v_round.configuration_doc#>>'{schedule,final_scoring_close}')::TIMESTAMPTZ THEN
    RETURN jsonb_build_object('status','stale','queued',0); END IF;
  IF jsonb_array_length(p_work_items) >
      (v_round.configuration_doc->>'stage_1_icp_count')::INTEGER+
      (v_round.configuration_doc->>'stage_2_icp_count')::INTEGER
    OR (SELECT count(*)<>count(DISTINCT value->>'scored_run_id') FROM jsonb_array_elements(p_work_items)) THEN
    RAISE EXCEPTION 'lab_arena_recovery_input_invalid' USING ERRCODE='22023'; END IF;
  -- A restart hold remains authoritative. Queue creation takes the same lock
  -- as claims and transitions; no grant can bypass a captured drain.
  IF EXISTS (SELECT 1 FROM public.lab_arena_restart_claim_control WHERE operator_paused) THEN
    RETURN jsonb_build_object('status','paused','queued',0); END IF;
  FOR v_parent IN SELECT * FROM public.lab_arena_runs
    WHERE round_id=p_round_id AND stage=1 AND attempt=2 AND status='failed'
      AND kind IN ('execute','score') ORDER BY assignment_id FOR UPDATE
  LOOP
    v_new:=jsonb_populate_record(v_parent,jsonb_build_object('run_id',v_parent.assignment_id||':3','attempt',3));
    IF public.lab_arena__baseline_recovery_run_v1(v_new)
      AND (v_parent.kind<>'execute' OR (v_round.status='stage2'
        AND clock_timestamp()<(v_round.configuration_doc#>>'{schedule,stage_2_close}')::TIMESTAMPTZ))
      AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs other
        WHERE other.assignment_id=v_parent.assignment_id AND (other.attempt>=3 OR other.status IN ('pending','leased','submitted','accepted')))
    THEN
      -- Frozen successful-call policy: retain unknown FAILED fees, but never
      -- overlap an in-flight call or proceed with unresolved successful cost.
      v_state:=public.lab_arena__successful_icp_cost_state(p_round_id,v_parent.submission_id,v_parent.icp_position);
      IF (v_state->>'inflight_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
        OR (v_state->>'success_unresolved_calls')::BIGINT IS DISTINCT FROM 0::BIGINT
        OR EXISTS (SELECT 1 FROM (SELECT DISTINCT ON (l.run_id,l.call_identity)
          l.entry_kind,l.entry_doc,l.terminal_response FROM public.lab_arena_ledger l
          JOIN public.lab_arena_runs r ON r.run_id=l.run_id
          WHERE r.assignment_id=v_parent.assignment_id AND l.call_identity IS NOT NULL
          ORDER BY l.run_id,l.call_identity,l.entry_id DESC) heads
          WHERE entry_kind IN ('reservation','dispatch') OR
            (entry_kind='uncertain' AND entry_doc#>'{call,call_succeeded}' IS DISTINCT FROM 'false'::JSONB) OR
            (entry_kind='settlement' AND jsonb_typeof(terminal_response->'call_succeeded') IS DISTINCT FROM 'boolean'))
      THEN CONTINUE; END IF;
      INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
        icp_position,attempt,status,lease_generation,stage_generation,kind,scored_run_id,previous_runner_hotkey,
        judgment_cache_key,judgment_input_hash,judgment_scope_doc,judgment_group_leader,
        judgment_group_miner_hotkeys,company_judgment_refs)
      VALUES(v_new.run_id,v_new.assignment_id,p_round_id,v_new.submission_id,v_new.miner_hotkey,1,
        v_new.icp_position,3,'pending',v_parent.lease_generation,v_round.stage_generation,v_new.kind,
        v_new.scored_run_id,v_parent.runner_hotkey,v_new.judgment_cache_key,v_new.judgment_input_hash,
        v_new.judgment_scope_doc,v_new.judgment_group_leader,v_new.judgment_group_miner_hotkeys,v_new.company_judgment_refs);
      v_created:=v_created+1;
    END IF;
  END LOOP;
  FOR v_item IN SELECT value FROM jsonb_array_elements(p_work_items) LOOP
    SELECT * INTO v_source FROM public.lab_arena_runs WHERE run_id=v_item->>'scored_run_id' FOR SHARE;
    IF NOT FOUND OR v_source.status<>'accepted' OR v_source.kind<>'execute'
      OR v_source.round_id IS DISTINCT FROM p_round_id
      OR NOT public.lab_arena__baseline_recovery_run_v1(v_source)
      OR v_item->>'submission_id' IS DISTINCT FROM v_source.submission_id
      OR v_item->'icp_position' IS DISTINCT FROM to_jsonb(v_source.icp_position)
      OR v_item->>'output_ref' IS DISTINCT FROM v_source.output_ref
      OR COALESCE(v_item->>'judgment_cache_key','') !~ '^sha256:[0-9a-f]{64}$'
      OR COALESCE(v_item->>'judgment_input_hash','') !~ '^sha256:[0-9a-f]{64}$'
      OR jsonb_typeof(v_item->'judgment_scope_doc') IS DISTINCT FROM 'object'
      OR v_item#>>'{judgment_scope_doc,cache_key}' IS DISTINCT FROM v_item->>'judgment_cache_key'
      OR v_item#>>'{judgment_scope_doc,scoring_input_hash}' IS DISTINCT FROM v_item->>'judgment_input_hash'
      OR v_item#>>'{judgment_scope_doc,round_id}' IS DISTINCT FROM p_round_id
      OR v_item#>>'{judgment_scope_doc,network_name}' IS DISTINCT FROM v_round.arena_network_name
      OR v_item#>'{judgment_scope_doc,netuid}' IS DISTINCT FROM to_jsonb(v_round.arena_netuid)
      OR v_item#>>'{judgment_scope_doc,evaluation_date}' IS DISTINCT FROM v_round.evaluation_date
      OR v_item#>>'{judgment_scope_doc,integrity_policy}' IS DISTINCT FROM v_round.configuration_doc->>'integrity_policy'
      OR v_item#>>'{judgment_scope_doc,scorer_image_digest}' IS DISTINCT FROM v_round.configuration_doc->>'scorer_image_digest'
      OR v_item#>>'{judgment_scope_doc,scorer_image_reference}' IS DISTINCT FROM v_round.configuration_doc->>'scorer_image_reference'
    THEN RAISE EXCEPTION 'lab_arena_recovery_scoring_item_invalid' USING ERRCODE='22023'; END IF;
    v_cache:=NULL;
    IF v_item ? 'reuse_cache_key' THEN
      SELECT * INTO v_cache FROM public.lab_arena_judgment_cache WHERE cache_key=v_item->>'reuse_cache_key';
      IF NOT FOUND OR v_item->>'reuse_cache_key' IS DISTINCT FROM v_item->>'judgment_cache_key'
        OR v_cache.scoring_input_hash IS DISTINCT FROM v_item->>'judgment_input_hash'
        OR v_cache.scope_doc IS DISTINCT FROM v_item->'judgment_scope_doc'
        OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source
          WHERE source.run_id=v_cache.source_score_run_id AND source.kind='score' AND source.status='accepted'
            AND source.runner_hotkey=v_cache.source_runner_hotkey AND source.scored_run_id=v_cache.source_scored_run_id)
      THEN RAISE EXCEPTION 'lab_arena_judgment_cache_invalid' USING ERRCODE='22023'; END IF;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE scored_run_id=v_source.run_id AND kind='score') THEN
      INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
        icp_position,attempt,status,stage_generation,kind,scored_run_id,judgment_cache_key,
        judgment_input_hash,judgment_scope_doc,judgment_group_leader,judgment_group_miner_hotkeys,
        result_doc,terminal_cause,output_ref,judgment_cache_source_run_id)
      VALUES(v_source.assignment_id||':score:1',v_source.assignment_id||':score',p_round_id,v_source.submission_id,
        v_source.miner_hotkey,1,v_source.icp_position,1,CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END,v_round.stage_generation,'score',v_source.run_id,
        v_item->>'judgment_cache_key',v_item->>'judgment_input_hash',v_item->'judgment_scope_doc',
        v_cache.cache_key IS NULL,ARRAY[v_source.miner_hotkey],
        CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE jsonb_build_object(
          'schema_version','leadpoet.lab_arena.cached_run_result.v1','terminal_status','accepted',
          'cache_key',v_cache.cache_key,'source_score_run_id',v_cache.source_score_run_id) END,
        CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE 'accepted' END,
        CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE v_cache.evidence_doc->>'source_output_ref' END,
        v_cache.source_score_run_id);
      v_created:=v_created+1;
    END IF;
  END LOOP;
  -- Pending recovery judges cross existing generation transitions, never
  -- leased work. Claims remain blocked until the ordinary stage2 score phase.
  UPDATE public.lab_arena_runs r SET stage_generation=v_round.stage_generation
  WHERE r.round_id=p_round_id AND r.status='pending' AND r.kind='score'
    AND r.stage_generation<>v_round.stage_generation AND public.lab_arena__baseline_recovery_run_v1(r);
  RETURN jsonb_build_object('status','ok','queued',v_created);
END;
$recover_baseline$;
ALTER FUNCTION public.lab_arena_recover_baseline_v1(TEXT,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_recover_baseline_v1(TEXT,JSONB) FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_recover_baseline_v1(TEXT,JSONB) TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena__baseline_recovery_phase_v1(p_round public.lab_arena_rounds,p_run public.lab_arena_runs)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER SET search_path=pg_catalog,public AS $phase$
 SELECT COALESCE(public.lab_arena__baseline_recovery_run_v1(p_run) AND
   ((p_run.kind='execute' AND p_round.status='stage2') OR
    (p_run.kind='score' AND p_round.status='stage2_scoring')),FALSE);
$phase$;
ALTER FUNCTION public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__baseline_recovery_phase_v1(public.lab_arena_rounds,public.lab_arena_runs) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Edit only the existing shared guards and stage scopes. Owner, ACLs and
-- hardened search paths must match before any replacement is accepted.
DO $patch_recovery470$
DECLARE v_patch RECORD; v_definition TEXT; v_identity JSONB; v_count INTEGER;
BEGIN
 FOR v_patch IN SELECT * FROM (VALUES
 ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)',
  $old$      runs.stage = v_stage
      OR ($old$,
  $new$      runs.stage = v_stage
      OR public.lab_arena__baseline_recovery_phase_v1(v_round,runs)
      OR ($new$,1),
 ('public.lab_arena__lock_current_lease(text,text)',
  $old$       OR (
         -- lab_arena_parallel_twenty_lease_guard$old$,
  $new$       OR public.lab_arena__baseline_recovery_phase_v1(v_round,v_run)
       OR (
         -- lab_arena_parallel_twenty_lease_guard$new$,1),
 ('public.lab_arena_complete_attempt(text,text,jsonb,text,text)',
  $old$  IF v_status = 'failed'
     AND p_terminal_cause$old$,
  $new$  IF v_status = 'failed'
     AND NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
     AND p_terminal_cause$new$,1),
 ('public.lab_arena_expire_leases(text)',
  $old$    ) AND v_run.stage_generation = v_round.stage_generation$old$,
  $new$    ) AND NOT (v_run.attempt>=3 AND public.lab_arena__baseline_recovery_run_v1(v_run))
      AND v_run.stage_generation = v_round.stage_generation$new$,1),
 ('public.lab_arena_close_stage(text,smallint)',
  $old$active.round_id=p_round_id AND active.stage=p_stage$old$,
  $new$active.round_id=p_round_id AND (active.stage=p_stage OR
           (p_stage=2 AND public.lab_arena__baseline_recovery_run_v1(active)))$new$,1),
 ('public.lab_arena_close_stage(text,smallint)',
  $old$WHERE round_id = p_round_id AND stage = p_stage AND kind = 'execute'$old$,
  $new$WHERE round_id = p_round_id AND (stage = p_stage OR
      (p_stage=2 AND public.lab_arena__baseline_recovery_run_v1(lab_arena_runs))) AND kind = 'execute'$new$,1),
 ('public.lab_arena_close_scoring(text,smallint)',
  $old$WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score'$old$,
  $new$WHERE round_id = p_round_id AND (stage = p_stage OR
      (p_stage=2 AND public.lab_arena__baseline_recovery_run_v1(lab_arena_runs))) AND kind = 'score'$new$,1),
 ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
  $old$  v_integrity BOOLEAN;$old$,
  $new$  v_integrity BOOLEAN;
  v_late_baseline BOOLEAN;$new$,1),
 ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
  $old$  IF v_round.status NOT IN ($old$,
  $new$  v_late_baseline := p_stage=1 AND v_round.status='stage2_judged'
    AND v_round.configuration_doc->>'execution_sequence_policy'='baseline_scored_first_v1'
    AND v_round.configuration_doc->>'sourcing_cost_eligibility_policy'='successful_calls_per_icp_v1';
  IF NOT COALESCE(v_late_baseline,FALSE) AND v_round.status NOT IN ($new$,1),
 ('public.lab_arena_record_run_scores(text,smallint,jsonb)',
  $old$    IF v_run.per_icp_score IS NOT NULL THEN$old$,
  $new$    IF v_late_baseline AND (v_run.status<>'accepted'
      OR v_run.assignment_id IS DISTINCT FROM v_run.round_id||':'||v_run.submission_id||':1:'||v_run.icp_position::TEXT
      OR v_run.run_id IS DISTINCT FROM v_run.assignment_id||':'||v_run.attempt::TEXT
      OR NOT EXISTS (SELECT 1 FROM public.lab_arena_submissions s
        WHERE s.submission_id=v_run.submission_id AND s.round_id=v_run.round_id
          AND s.miner_hotkey=v_run.miner_hotkey AND s.is_king)
      OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judged
        WHERE judged.scored_run_id=v_run.run_id AND judged.round_id=v_run.round_id
          AND judged.submission_id=v_run.submission_id AND judged.miner_hotkey=v_run.miner_hotkey
          AND judged.stage=1 AND judged.icp_position=v_run.icp_position
          AND judged.assignment_id=v_run.assignment_id||':score'
          AND judged.run_id=judged.assignment_id||':'||judged.attempt::TEXT
          AND judged.kind='score' AND judged.status='accepted' AND judged.terminal_cause='accepted'
          AND judged.output_ref IS NOT NULL
          AND (v_run.per_icp_score IS NOT NULL OR public.lab_arena__baseline_recovery_run_v1(judged)))) THEN
      RAISE EXCEPTION 'lab_arena_late_baseline_score_invalid' USING ERRCODE='42501';
    END IF;
    IF v_run.per_icp_score IS NOT NULL THEN$new$,1)
 ) AS patches(signature,old_text,new_text,expected_count)
 LOOP
   SELECT pg_get_functiondef(p.oid),jsonb_build_array(owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
   INTO v_definition,v_identity FROM pg_proc p JOIN pg_roles owner ON owner.oid=p.proowner
   WHERE p.oid=to_regprocedure(v_patch.signature);
   -- Internal lease helper has no service EXECUTE grant. All public RPCs do.
   IF v_definition IS NULL OR v_identity->>0 IS DISTINCT FROM 'lab_arena_owner'
     OR v_identity->2 IS DISTINCT FROM 'true'::JSONB
     OR v_identity->>3 IS DISTINCT FROM 'v'
     OR v_identity->4 IS DISTINCT FROM to_jsonb(ARRAY['search_path=pg_catalog, public']::TEXT[])
     OR v_identity->>1 IS DISTINCT FROM (CASE WHEN v_patch.signature LIKE '%__lock_current_lease%'
       THEN '{lab_arena_owner=X/lab_arena_owner}'
       ELSE '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' END) THEN
     RAISE EXCEPTION 'Arena baseline recovery security shape differs: %',v_patch.signature USING ERRCODE='55000'; END IF;
   IF strpos(v_definition,v_patch.new_text)>0 THEN CONTINUE; END IF;
   v_count:=(length(v_definition)-length(replace(v_definition,v_patch.old_text,'')))/length(v_patch.old_text);
   IF v_count<>v_patch.expected_count THEN
     RAISE EXCEPTION 'Arena baseline recovery anchor differs: %',v_patch.signature USING ERRCODE='55000'; END IF;
   EXECUTE replace(v_definition,v_patch.old_text,v_patch.new_text);
 END LOOP;
END;
$patch_recovery470$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;

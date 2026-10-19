-- Future legacy integrity rounds share raw per-company judgments through the
-- existing lease, reservation, immutable evidence and authority-slot protocol.
-- No historical frozen round, score, provider cost or retry allowance changes.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena__company_judgment_policy_v1(p_round public.lab_arena_rounds)
RETURNS BOOLEAN LANGUAGE sql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $policy$
 SELECT COALESCE(p_round.configuration_doc->>'integrity_policy'='arena_integrity_v1'
   AND p_round.configuration_doc->>'company_judgment_policy'='company_judgment_v1'
   AND p_round.configuration_doc#>>'{scorer_policy,company_judgment_policy}'='company_judgment_v1'
   AND p_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}'='qualification_integrity_v2'
   AND NOT p_round.configuration_doc ? 'contact_policy'
   AND NOT (p_round.configuration_doc->'scorer_policy') ? 'contact_policy'
   AND NOT p_round.configuration_doc ? 'company_quality_policy'
   AND NOT (p_round.configuration_doc->'scorer_policy') ? 'company_quality_policy',FALSE);
$policy$;
ALTER FUNCTION public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

-- Every ordered ref remains bound to the frozen round and scorer. Destination
-- membership remains in the score assignment, outside reusable raw evidence.
CREATE OR REPLACE FUNCTION public.lab_arena__company_judgment_refs_v1(p_round public.lab_arena_rounds,p_refs JSONB)
RETURNS BOOLEAN LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $refs$
BEGIN
 IF NOT public.lab_arena__company_judgment_policy_v1(p_round)
   OR jsonb_typeof(p_refs) IS DISTINCT FROM 'array' THEN RETURN FALSE; END IF;
 RETURN jsonb_array_length(p_refs) BETWEEN 0 AND 5 AND NOT EXISTS (
   SELECT 1 FROM jsonb_array_elements(p_refs) WITH ORDINALITY refs(ref,ordinal)
   WHERE jsonb_typeof(ref) IS DISTINCT FROM 'object'
     OR ref->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.company_judgment_ref.v1'
     OR ref->'company_index' IS DISTINCT FROM to_jsonb((ordinal-1)::INTEGER)
     OR COALESCE(ref->>'cache_key','') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(ref->>'company_input_hash','') !~ '^sha256:[0-9a-f]{64}$'
     OR jsonb_typeof(ref->'scope_doc') IS DISTINCT FROM 'object'
     OR ref#>>'{scope_doc,cache_key}' IS DISTINCT FROM ref->>'cache_key'
     OR ref#>>'{scope_doc,company_input_hash}' IS DISTINCT FROM ref->>'company_input_hash'
     OR ref#>>'{scope_doc,round_id}' IS DISTINCT FROM p_round.round_id
     OR ref#>>'{scope_doc,network_name}' IS DISTINCT FROM p_round.arena_network_name
     OR ref#>'{scope_doc,netuid}' IS DISTINCT FROM to_jsonb(p_round.arena_netuid)
     OR ref#>>'{scope_doc,evaluation_date}' IS DISTINCT FROM p_round.evaluation_date
     OR ref#>>'{scope_doc,integrity_policy}' IS DISTINCT FROM 'arena_integrity_v1'
     OR ref#>>'{scope_doc,company_judgment_policy}' IS DISTINCT FROM 'company_judgment_v1'
     OR (ref->'scope_doc') ? 'company_quality_policy'
     OR ref#>>'{scope_doc,scorer_image_digest}' IS DISTINCT FROM p_round.configuration_doc->>'scorer_image_digest'
     OR ref#>>'{scope_doc,scorer_image_reference}' IS DISTINCT FROM p_round.configuration_doc->>'scorer_image_reference');
END;
$refs$;
ALTER FUNCTION public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,JSONB)
 FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;

DO $company_judgment_patches$
DECLARE p RECORD; definition TEXT; before_identity JSONB; after_identity JSONB;
BEGIN
 FOR p IN SELECT * FROM (VALUES
  ('public.lab_arena_open_scoring_v3(text,smallint,jsonb)',$old501$  IF v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'company_quality_policy'
       IS DISTINCT FROM 'company_quality_v1'
     OR v_round.configuration_doc #>> '{scorer_policy,company_quality_policy}'
       IS DISTINCT FROM 'company_quality_v1' THEN
$old501$,$new501$  IF (v_round.configuration_doc ? 'company_judgment_policy'
      OR (v_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy')
    AND NOT public.lab_arena__company_judgment_policy_v1(v_round) THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_policy_invalid' USING ERRCODE='22023';
  END IF;
  IF NOT public.lab_arena__company_judgment_policy_v1(v_round) AND (v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'company_quality_policy'
       IS DISTINCT FROM 'company_quality_v1'
     OR v_round.configuration_doc #>> '{scorer_policy,company_quality_policy}'
       IS DISTINCT FROM 'company_quality_v1') THEN
$new501$),
  ('public.lab_arena_open_scoring_v3(text,smallint,jsonb)',$old501$OR refs.ref #>> '{scope_doc,company_quality_policy}' IS DISTINCT FROM
              'company_quality_v1'$old501$,$new501$OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN
             NOT public.lab_arena__company_judgment_refs_v1(v_round,v_item->'company_judgment_refs')
             ELSE refs.ref #>> '{scope_doc,company_quality_policy}' IS DISTINCT FROM
              'company_quality_v1' END)$new501$),
  ('public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)',$old501$IF (p_terminal_cause = 'accepted' AND v_expected <> v_supplied)$old501$,$new501$-- Only the frozen company-judgment policy allows deterministic skips.
  -- The gateway replays the full destination projection before this RPC.
  IF (p_terminal_cause = 'accepted' AND v_expected <> v_supplied
       AND NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds frozen
         WHERE frozen.round_id=v_run.round_id
           AND public.lab_arena__company_judgment_policy_v1(frozen)))$new501$),
  ('public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)',$old501$OR pg_catalog.jsonb_typeof(v_doc -> 'raw_judgment')$old501$,$new501$OR (CASE WHEN EXISTS (SELECT 1 FROM public.lab_arena_rounds frozen
         WHERE frozen.round_id=v_run.round_id AND public.lab_arena__company_judgment_policy_v1(frozen))
         THEN v_doc#>>'{raw_judgment,company_judgment_policy}' IS DISTINCT FROM 'company_judgment_v1'
         ELSE (v_doc->'raw_judgment') ? 'company_judgment_policy' END)
       OR pg_catalog.jsonb_typeof(v_doc -> 'raw_judgment')$new501$),
  ('public.lab_arena_complete_attempt_v3(text,text,jsonb,text,text,text,jsonb,text)',$old501$      ) ELSE p_result END,
    p_terminal_cause, p_output_ref$old501$,$new501$      ) WHEN EXISTS (SELECT 1 FROM public.lab_arena_rounds frozen
          WHERE frozen.round_id=v_run.round_id AND public.lab_arena__company_judgment_policy_v1(frozen)) THEN
        p_result || jsonb_build_object('company_judgments_stored',v_supplied)
      ELSE p_result END,
    p_terminal_cause, p_output_ref$new501$),
  ('public.lab_arena_runs_participation_v1()',$old501$pg_catalog.jsonb_array_length(v_company_cache -> 'misses') > 0;$old501$,$new501$pg_catalog.jsonb_array_length(v_company_cache -> 'misses') > 0
        AND (NOT EXISTS (SELECT 1 FROM public.lab_arena_rounds frozen
          WHERE frozen.round_id=OLD.round_id AND frozen.configuration_doc->>'company_judgment_policy'='company_judgment_v1')
          OR COALESCE(NEW.result_doc->>'company_judgments_stored','0') ~ '^[1-5]$');$new501$),
  ('public.lab_arena_retry_credit_failures_v1(text,text,text,text)',$old501$AND v_round.configuration_doc ->> 'company_quality_policy' = 'company_quality_v1'$old501$,$new501$AND (v_round.configuration_doc ->> 'company_quality_policy' = 'company_quality_v1'
            OR public.lab_arena__company_judgment_policy_v1(v_round))$new501$),
  ('public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)',$old501$AND NOT frozen.configuration_doc ? 'company_quality_policy'$old501$,$new501$AND NOT frozen.configuration_doc ? 'company_quality_policy'
        AND NOT frozen.configuration_doc ? 'company_judgment_policy'
        AND NOT (frozen.configuration_doc->'scorer_policy') ? 'company_judgment_policy'$new501$),
  ('public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)',$old501$OR v_round.configuration_doc ? 'company_quality_policy'$old501$,$new501$OR ((v_round.configuration_doc ? 'company_judgment_policy'
          OR (v_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy')
        AND NOT public.lab_arena__company_judgment_policy_v1(v_round))
    OR v_round.configuration_doc ? 'company_quality_policy'$new501$),
  ('public.lab_arena_recover_baseline_v1(text,jsonb)',$old501$      OR COALESCE(v_item->>'judgment_cache_key','') !~ '^sha256:[0-9a-f]{64}$'
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
$old501$,$new501$      OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN
        NOT public.lab_arena__company_judgment_refs_v1(v_round,v_item->'company_judgment_refs')
        OR v_item ?| ARRAY['reuse_cache_key','judgment_cache_key','judgment_input_hash','judgment_scope_doc']
        ELSE (FALSE
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
      ) END)
$new501$),
  ('public.lab_arena_recover_baseline_v1(text,jsonb)',$old501$      INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
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
$old501$,$new501$      IF public.lab_arena__company_judgment_policy_v1(v_round) THEN
        INSERT INTO public.lab_arena_runs(run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
          icp_position,attempt,status,stage_generation,kind,scored_run_id,company_judgment_refs)
        VALUES(v_source.assignment_id||':score:1',v_source.assignment_id||':score',p_round_id,
          v_source.submission_id,v_source.miner_hotkey,1,v_source.icp_position,1,'pending',
          v_round.stage_generation,'score',v_source.run_id,v_item->'company_judgment_refs');
      ELSE
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
      END IF;
$new501$),
  ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)',$old501$    OR p_run.judgment_scope_doc->>'scorer_image_digest' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_digest'
    OR p_run.judgment_scope_doc->>'scorer_image_reference' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_reference'
    OR p_run.judgment_scope_doc->>'round_id' IS DISTINCT FROM p_round.round_id
    OR p_run.judgment_scope_doc->>'scoring_input_hash' IS DISTINCT FROM p_run.judgment_input_hash
    OR p_run.judgment_scope_doc->>'cache_key' IS DISTINCT FROM p_run.judgment_cache_key
$old501$,$new501$    OR (CASE WHEN public.lab_arena__company_judgment_policy_v1(p_round) THEN
      NOT public.lab_arena__company_judgment_refs_v1(p_round,p_run.company_judgment_refs)
      OR p_run.judgment_cache_key IS NOT NULL OR p_run.judgment_input_hash IS NOT NULL
      OR p_run.judgment_scope_doc IS NOT NULL
      ELSE (FALSE
    OR p_run.judgment_scope_doc->>'scorer_image_digest' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_digest'
    OR p_run.judgment_scope_doc->>'scorer_image_reference' IS DISTINCT FROM
      p_round.configuration_doc->>'scorer_image_reference'
    OR p_run.judgment_scope_doc->>'round_id' IS DISTINCT FROM p_round.round_id
    OR p_run.judgment_scope_doc->>'scoring_input_hash' IS DISTINCT FROM p_run.judgment_input_hash
    OR p_run.judgment_scope_doc->>'cache_key' IS DISTINCT FROM p_run.judgment_cache_key
    ) END)
$new501$),
  ('public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)',$old501$AND v_parent.judgment_scope_doc IS NOT DISTINCT FROM p_run.judgment_scope_doc;$old501$,$new501$AND v_parent.judgment_scope_doc IS NOT DISTINCT FROM p_run.judgment_scope_doc
    AND v_parent.company_judgment_refs IS NOT DISTINCT FROM p_run.company_judgment_refs;$new501$),
  ('public.lab_arena_record_run_scores(text,smallint,jsonb)',$old501$              AND judged.judgment_scope_doc->>'round_id'=v_round.round_id
              AND judged.judgment_scope_doc->>'scorer_image_digest'=
                v_round.configuration_doc->>'scorer_image_digest'
              AND judged.judgment_scope_doc->>'scorer_image_reference'=
                v_round.configuration_doc->>'scorer_image_reference'
              AND judged.judgment_scope_doc->>'scoring_input_hash'=judged.judgment_input_hash
              AND judged.judgment_scope_doc->>'cache_key'=judged.judgment_cache_key$old501$,$new501$              AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(v_round) THEN
                public.lab_arena__company_judgment_refs_v1(v_round,judged.company_judgment_refs)
                AND judged.judgment_cache_key IS NULL AND judged.judgment_input_hash IS NULL
                AND judged.judgment_scope_doc IS NULL
                ELSE (TRUE
              AND judged.judgment_scope_doc->>'round_id'=v_round.round_id
              AND judged.judgment_scope_doc->>'scorer_image_digest'=
                v_round.configuration_doc->>'scorer_image_digest'
              AND judged.judgment_scope_doc->>'scorer_image_reference'=
                v_round.configuration_doc->>'scorer_image_reference'
              AND judged.judgment_scope_doc->>'scoring_input_hash'=judged.judgment_input_hash
              AND judged.judgment_scope_doc->>'cache_key'=judged.judgment_cache_key              ) END)$new501$),
  ('public.lab_arena_public_configuration_v1(public.lab_arena_rounds)',$old501$'company_quality_policy', 'intent_details_policy', 'baseline_hotkey'$old501$,$new501$'company_quality_policy', 'company_judgment_policy', 'intent_details_policy', 'baseline_hotkey'$new501$)
 ) patches(signature,old_text,new_text) LOOP
   SELECT pg_get_functiondef(proc.oid),jsonb_build_array(pg_get_userbyid(proc.proowner),
     proc.proacl::TEXT,proc.prosecdef,proc.provolatile,proc.proconfig)
   INTO definition,before_identity FROM pg_proc proc WHERE proc.oid=to_regprocedure(p.signature);
   IF definition IS NULL OR before_identity->>0 IS DISTINCT FROM 'lab_arena_owner'
     OR before_identity->>1 IS DISTINCT FROM (CASE
       WHEN p.signature='public.lab_arena_runs_participation_v1()' OR strpos(p.signature,'public.lab_arena__')=1
         THEN '{lab_arena_owner=X/lab_arena_owner}'
       ELSE '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}' END)
     OR before_identity->2 IS DISTINCT FROM to_jsonb(p.signature NOT IN ('public.lab_arena_runs_participation_v1()','public.lab_arena_public_configuration_v1(public.lab_arena_rounds)'))
     OR before_identity->>3 IS DISTINCT FROM (CASE WHEN p.signature IN (
       'public.lab_arena__legacy_partial_bound(public.lab_arena_runs,jsonb,text)',
       'public.lab_arena__baseline_recovery_run_v1(public.lab_arena_runs)',
       'public.lab_arena__judge_continuation_v1(public.lab_arena_rounds,public.lab_arena_runs)',
       'public.lab_arena_public_configuration_v1(public.lab_arena_rounds)') THEN CASE WHEN p.signature='public.lab_arena_public_configuration_v1(public.lab_arena_rounds)' THEN 'i' ELSE 's' END ELSE 'v' END)
     OR before_identity->4->>0 IS DISTINCT FROM (CASE
       WHEN p.signature='public.lab_arena_runs_participation_v1()' THEN 'search_path=pg_catalog'
       ELSE 'search_path=pg_catalog, public' END) THEN
     RAISE EXCEPTION 'company judgment security shape differs: %',p.signature USING ERRCODE='55000';
   END IF;
   IF strpos(definition,p.new_text)>0 THEN CONTINUE; END IF;
   IF (length(definition)-length(replace(definition,p.old_text,'')))/length(p.old_text)<>1 THEN
     RAISE EXCEPTION 'company judgment patch shape differs: %',p.signature USING ERRCODE='55000'; END IF;
   EXECUTE replace(definition,p.old_text,p.new_text);
   SELECT jsonb_build_array(pg_get_userbyid(proc.proowner),proc.proacl::TEXT,
     proc.prosecdef,proc.provolatile,proc.proconfig) INTO after_identity
     FROM pg_proc proc WHERE proc.oid=to_regprocedure(p.signature);
   IF after_identity IS DISTINCT FROM before_identity THEN
     RAISE EXCEPTION 'company judgment security changed: %',p.signature USING ERRCODE='55000'; END IF;
 END LOOP;
END;
$company_judgment_patches$;

-- The canonical commit for new-mode rounds is derived from the exact current
-- v3 body, as v3 was derived from v2 in415. Future commit fixes must include v4.
-- Its sole policy addition is in the same OPEN -> COMMITTED write as the pin.
DO $company_commit$
DECLARE definition TEXT; original TEXT; security JSONB; old_text TEXT; new_text TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(pg_get_userbyid(p.proowner),
   p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO original,security FROM pg_proc p WHERE p.oid=to_regprocedure(
   'public.lab_arena_commit_round_v3(text,jsonb,text,text,date,text,text,jsonb)');
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
   TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
   RAISE EXCEPTION 'company judgment commit authority differs'; END IF;
 definition:=replace(original,'FUNCTION public.lab_arena_commit_round_v3(',
   'FUNCTION public.lab_arena_commit_round_v4(');
 definition:=replace(definition,'p_deepline_catalog jsonb)',
   'p_deepline_catalog jsonb, p_company_judgment_policy text)');
 old_text:=$old$  IF pg_catalog.jsonb_typeof(p_deepline_catalog) IS DISTINCT FROM 'object'$old$;
 new_text:=$new$  IF p_company_judgment_policy IS DISTINCT FROM 'company_judgment_v1'
    OR v_round.configuration_doc->>'integrity_policy' IS DISTINCT FROM 'arena_integrity_v1'
    OR v_round.configuration_doc#>>'{scorer_policy,scoring_adapter_version}' IS DISTINCT FROM 'qualification_integrity_v2'
    OR v_round.configuration_doc ?| ARRAY['company_quality_policy','contact_policy']
    OR (v_round.configuration_doc->'scorer_policy') ?| ARRAY['company_quality_policy','contact_policy']
    OR (v_round.configuration_doc ? 'company_judgment_policy' AND
      v_round.configuration_doc->>'company_judgment_policy' IS DISTINCT FROM p_company_judgment_policy)
    OR ((v_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy' AND
      v_round.configuration_doc#>>'{scorer_policy,company_judgment_policy}' IS DISTINCT FROM p_company_judgment_policy)
    OR (v_round.configuration_doc ? 'company_judgment_policy') IS DISTINCT FROM
      ((v_round.configuration_doc->'scorer_policy') ? 'company_judgment_policy')
    OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=p_round_id)
    OR v_round.benchmark_ref IS NOT NULL OR v_round.icp_set_date IS NOT NULL THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_commit_invalid' USING ERRCODE='22023'; END IF;
  IF p_deepline_catalog IS NOT NULL THEN
  IF pg_catalog.jsonb_typeof(p_deepline_catalog) IS DISTINCT FROM 'object'$new$;
 IF strpos(definition,old_text)=0 THEN RAISE EXCEPTION 'company commit catalog anchor differs'; END IF;
 definition:=replace(definition,old_text,new_text);
 old_text:=$old$  UPDATE public.lab_arena_rounds
  SET status = 'committed',$old$;
 new_text:=$new$  END IF;
  v_round.configuration_doc:=jsonb_set(v_round.configuration_doc ||
    jsonb_build_object('company_judgment_policy',p_company_judgment_policy),
    '{scorer_policy,company_judgment_policy}',to_jsonb(p_company_judgment_policy));
  UPDATE public.lab_arena_rounds
  SET status = 'committed',$new$;
 IF strpos(definition,old_text)=0 THEN RAISE EXCEPTION 'company commit transition anchor differs'; END IF;
 definition:=replace(definition,old_text,new_text);
 old_text:=$old$pg_catalog.jsonb_set(pg_catalog.jsonb_set(
        v_round.configuration_doc || pg_catalog.jsonb_build_object('deepline_catalog', p_deepline_catalog),
        '{call_quotas,deepline}', '0'), '{scoring_call_quotas,deepline}', '0')$old$;
 new_text:=$new$(CASE WHEN p_deepline_catalog IS NULL THEN v_round.configuration_doc
      ELSE pg_catalog.jsonb_set(pg_catalog.jsonb_set(
        v_round.configuration_doc || pg_catalog.jsonb_build_object('deepline_catalog', p_deepline_catalog),
        '{call_quotas,deepline}', '0'), '{scoring_call_quotas,deepline}', '0') END)$new$;
 IF strpos(definition,old_text)=0 THEN RAISE EXCEPTION 'company commit config anchor differs'; END IF;
 definition:=replace(definition,old_text,new_text);
 IF definition=original OR strpos(definition,'p_company_judgment_policy text)')=0 THEN
   RAISE EXCEPTION 'company commit signature differs'; END IF;
 EXECUTE definition;
END;
$company_commit$;
ALTER FUNCTION public.lab_arena_commit_round_v4(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_commit_round_v4(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB,TEXT)
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_commit_round_v4(TEXT,JSONB,TEXT,TEXT,DATE,TEXT,TEXT,JSONB,TEXT)
 TO lab_arena_service;

-- Strip only the newly admitted markers when checking the original config
-- equality rules. The original OPEN -> COMMITTED, image, catalog and quota
-- guards remain authoritative. OPEN -> OPEN remains forbidden.
DO $company_commit_guard$
DECLARE definition TEXT; security JSONB; old_text TEXT; new_text TEXT; gate TEXT;
BEGIN
 SELECT pg_get_functiondef(p.oid),jsonb_build_array(pg_get_userbyid(p.proowner),
   p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
 INTO definition,security FROM pg_proc p WHERE p.oid=to_regprocedure(
   'public.lab_arena_rounds_write_once_v1()');
 IF security IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
   '{lab_arena_owner=X/lab_arena_owner}',FALSE,'v',ARRAY['search_path=pg_catalog']::TEXT[]) THEN
   RAISE EXCEPTION 'company judgment round guard authority differs'; END IF;
 old_text:=$old$(NEW.configuration_doc - 'infrastructure_retry_cost_policy'$old$;
 new_text:=$new$(CASE WHEN
       -- lab_arena_company_judgment_commit_guard
       public.lab_arena__company_judgment_policy_v1(NEW)
       AND NOT OLD.configuration_doc ? 'company_judgment_policy'
       AND NOT (OLD.configuration_doc->'scorer_policy') ? 'company_judgment_policy'
       AND NOT EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id=OLD.round_id)
       THEN jsonb_set(NEW.configuration_doc-'company_judgment_policy','{scorer_policy}',
         (NEW.configuration_doc->'scorer_policy')-'company_judgment_policy')
       ELSE NEW.configuration_doc END - 'infrastructure_retry_cost_policy'$new$;
 IF strpos(definition,'lab_arena_company_judgment_commit_guard')=0 THEN
   gate:=substring(definition FROM strpos(definition,'  IF NEW.configuration_doc IS DISTINCT'));
   IF (length(gate)-length(replace(gate,old_text,'')))/length(old_text)<>2 THEN
     RAISE EXCEPTION 'company judgment round guard anchor differs'; END IF;
   EXECUTE replace(definition,gate,replace(gate,old_text,new_text));
 END IF;
END;
$company_commit_guard$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;

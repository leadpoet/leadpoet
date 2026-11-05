-- Prospective evidence-bound reuse. No historical verdict is rewritten.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
ALTER TABLE public.lab_arena_signal_judgment_uses ADD COLUMN IF NOT EXISTS acceptance_doc JSONB;
CREATE INDEX IF NOT EXISTS lab_arena_signal_judgments_evidence_family_v2
 ON public.lab_arena_signal_judgments ((scope_doc->>'family_hash'),created_at DESC)
 WHERE scope_doc->>'evidence_judgment_policy'='evidence_judgment_v2';

-- Patch only the reviewed625 functions. Retain their lease, authority,
-- reservation and retired621 source exclusion paths without duplication.
DO $patch631$
DECLARE signature TEXT; definition TEXT; updated TEXT; metadata JSONB; preimage TEXT; postimage TEXT; identity JSONB;
BEGIN
 FOREACH signature IN ARRAY ARRAY[
  'public.lab_arena_claim_signal_judgment_v1(text,text,integer,integer,jsonb)',
  'public.lab_arena_complete_signal_judgment_v1(text,text,integer,integer,jsonb,jsonb,jsonb,text)'] LOOP
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
   jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
   INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
   WHERE p.oid=to_regprocedure(signature);
  IF identity IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}",true,"v",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
   RAISE EXCEPTION 'evidence631 security shape differs' USING ERRCODE='55000'; END IF;
  postimage:=CASE WHEN signature LIKE '%claim_signal%' THEN
   '9483007e0bd7982a00920f063185fe0f1138bd56d935f428a5a20235d0b833b8'
   ELSE '2863daa50a0aa9f0c27d94bc740af9bd6c42a330f11df8927cfa45267adb8a12' END;
  IF encode(extensions.digest(definition,'sha256'),'hex')=postimage THEN CONTINUE; END IF;
  preimage:=CASE WHEN signature LIKE '%claim_signal%' THEN
   '5397c48a740fddef5ea7c9e5aa0a319450567407308267c5ebc6ed56bf87b6ef'
   ELSE 'eb40ae7ccb5b5791bcfc3ff0764922f401158dbe314275a0a421f1f22fc12c30' END;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM preimage THEN
   RAISE EXCEPTION 'evidence631 preimage differs' USING ERRCODE='55000'; END IF;
  updated:=replace(definition,
   $old$p_scope->>'checkpoint_kind'='verified_company_fit_v1'$old$,
   $new$(p_scope->>'checkpoint_kind'='verified_company_fit_v1' OR
       (p_scope->>'checkpoint_kind'='company_fit_evidence_v2'
        AND p_scope->>'evidence_judgment_policy'='evidence_judgment_v2'))$new$);
  updated:=replace(updated,$old$IF p_signal_index=-1 AND ($old$,
   $new$IF p_signal_index=-1 AND p_scope->>'checkpoint_kind'='verified_company_fit_v1' AND ($new$);
  -- Only v2 bypasses execution identity; its reusable identity is bound to
  -- complete evidence plus native inputs and the selected runtime instead.
  updated:=replace(updated,'BEGIN', $new$BEGIN
 -- evidence631 gated native evidence
 IF p_scope->>'evidence_judgment_policy'='evidence_judgment_v2' AND (
  COALESCE(p_scope->>'family_hash','') !~ '^sha256:[0-9a-f]{64}$'
  OR COALESCE(p_scope->>'evidence_hash','') !~ '^sha256:[0-9a-f]{64}$'
  OR jsonb_typeof(p_scope->'evidence_input') IS DISTINCT FROM 'object'
  OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs s JOIN public.lab_arena_rounds r ON r.round_id=s.round_id
    WHERE s.run_id=p_run_id AND r.configuration_doc#>>'{scorer_policy,evidence_judgment_policy}'='evidence_judgment_v2'
    AND p_scope->>'runtime_image_digest'=COALESCE(s.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
    AND p_scope->>'runtime_image_reference'=COALESCE(s.reviewer_runtime_doc->>'runtime_image_reference',r.configuration_doc->>'scorer_image_reference'))
 ) THEN RAISE EXCEPTION 'lab_arena_evidence_scope_invalid' USING ERRCODE='22023'; END IF;
$new$);
  IF signature LIKE '%complete_signal%' THEN
   updated:=replace(updated,$old$ -- This immutable verdict carries no cost authority.$old$,
    $new$ IF p_scope->>'evidence_judgment_policy'='evidence_judgment_v2' AND (
     octet_length(p_judgment::TEXT)>1048576
     OR (p_signal_index=-1 AND (
       p_judgment->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.company_fit_evidence.v2'
       OR p_judgment->'evidence_manifest' IS DISTINCT FROM p_scope->'evidence_input'
       OR p_judgment->>'source_snapshot_hash' IS DISTINCT FROM p_scope->>'evidence_hash'
       OR COALESCE(p_judgment#>>'{company_fit_receipt,decision}','') NOT IN ('match','mismatch')
       OR p_judgment#>>'{company_fit_receipt,company_fit_decision}' IS DISTINCT FROM p_judgment#>>'{company_fit_receipt,decision}'
       OR p_judgment#>>'{company_fit_receipt,gate}' IS DISTINCT FROM 'company_fit'
       OR p_judgment#>>'{company_fit_receipt,contract_id}' IS DISTINCT FROM 'company-fit-decision:v1'
       OR p_judgment#>>'{company_fit_receipt,contract_version}' IS DISTINCT FROM 'company-fit-decision:v1'
       OR jsonb_typeof(p_judgment->'linkedin_profile_source') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_judgment->'matched_company_sources') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_judgment->'required_attribute_sources') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_judgment#>'{company_fit_receipt,company_fit_stage_required}') IS DISTINCT FROM 'boolean'
       OR jsonb_typeof(p_judgment#>'{company_fit_receipt,company_fit_dimensions}') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_judgment#>'{company_fit_receipt,dimension_evidence}') IS DISTINCT FROM 'object'
       OR NOT (p_judgment#>'{company_fit_receipt,company_fit_dimensions}') ?& ARRAY['identity','employee_size','industry','geography','stage']
       OR EXISTS (SELECT 1 FROM unnest(ARRAY['identity','employee_size','industry','geography','stage']) dimension
         WHERE COALESCE(p_judgment#>>ARRAY['company_fit_receipt','company_fit_dimensions',dimension],'') NOT IN ('match','mismatch','unavailable'))
       OR (p_judgment#>>'{company_fit_receipt,decision}'='match' AND (
         p_judgment#>>'{company_fit_receipt,required_attribute_decision}' IS DISTINCT FROM 'match'
         OR EXISTS (SELECT 1 FROM unnest(ARRAY['identity','employee_size','industry','geography','stage']) dimension
           WHERE (dimension<>'stage' OR p_judgment#>'{company_fit_receipt,company_fit_stage_required}'='true'::JSONB)
            AND (p_judgment#>>ARRAY['company_fit_receipt','company_fit_dimensions',dimension] IS DISTINCT FROM 'match'
             OR p_judgment#>>ARRAY['company_fit_receipt','dimension_evidence',dimension,'decision'] IS DISTINCT FROM 'match'))
         OR p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,decision}' IS DISTINCT FROM 'match'
         OR COALESCE(p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,observed_name}','')=''
         OR COALESCE(p_judgment#>>'{company_fit_receipt,dimension_evidence,identity,web_identity_receipt,observed_domain}','')=''))
       OR (p_judgment#>>'{company_fit_receipt,decision}'='mismatch' AND NOT EXISTS (
         SELECT 1 FROM unnest(ARRAY['identity','employee_size','industry','geography','stage','required_attribute']) dimension
         WHERE (dimension<>'stage' OR p_judgment#>'{company_fit_receipt,company_fit_stage_required}'='true'::JSONB)
          AND (CASE WHEN dimension='required_attribute' THEN p_judgment#>>'{company_fit_receipt,required_attribute_decision}'
               ELSE p_judgment#>>ARRAY['company_fit_receipt','company_fit_dimensions',dimension] END)='mismatch'
          AND p_judgment#>>ARRAY['company_fit_receipt','dimension_evidence',dimension,'decision']='mismatch'))
       OR COALESCE(p_judgment#>>'{company_fit_receipt,failure_class}','')<>''
       OR COALESCE(p_judgment#>>'{company_fit_receipt,failure_reason_code}','')<>''))
     OR (p_signal_index>=0 AND (
       p_judgment->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.signal_evidence.v2'
       OR p_judgment->'evidence_input' IS DISTINCT FROM p_scope->'evidence_input'
       OR p_judgment->'claim' IS DISTINCT FROM p_scope->'claim'
       OR jsonb_typeof(p_judgment->'raw_judgment') IS DISTINCT FROM 'object'
       OR jsonb_typeof(p_judgment->'verifier_result') IS DISTINCT FROM 'object'
       OR p_judgment->'verifier_result'='{}'::JSONB))
    ) THEN RAISE EXCEPTION 'lab_arena_evidence_judgment_invalid' USING ERRCODE='22023'; END IF;
 -- This immutable verdict carries no cost authority.$new$);
  END IF;
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM postimage THEN
   RAISE EXCEPTION 'evidence631 patch failed' USING ERRCODE='55000'; END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=to_regprocedure(signature)) IS DISTINCT FROM metadata
   OR encode(extensions.digest(pg_get_functiondef(to_regprocedure(signature)),'sha256'),'hex') IS DISTINCT FROM postimage THEN
   RAISE EXCEPTION 'evidence631 security changed' USING ERRCODE='55000'; END IF;
 END LOOP;
END;
$patch631$;

CREATE OR REPLACE FUNCTION public.lab_arena_evidence_judgment_v2(
 p_operation TEXT,p_run_id TEXT,p_lease_token_hash TEXT,p_company_index INTEGER,p_signal_index INTEGER,
 p_scope JSONB,p_candidate_reference JSONB DEFAULT NULL,p_equivalence_receipt JSONB DEFAULT NULL
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER SET search_path=pg_catalog,public AS $$
DECLARE v_run public.lab_arena_runs; v_source public.lab_arena_runs;
 v_row public.lab_arena_signal_judgments; v_use public.lab_arena_signal_judgment_uses;
 v_candidates JSONB; v_ref JSONB; v_policy JSONB; v_model TEXT; v_receipt JSONB;
BEGIN
 SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id=p_run_id FOR UPDATE;
 IF NOT FOUND OR NOT public.lab_arena__signal_lease_valid_v1(v_run,p_lease_token_hash)
  OR p_company_index NOT BETWEEN 0 AND 4 OR p_signal_index NOT BETWEEN -1 AND 99
  OR p_operation NOT IN ('candidates','claim')
  OR jsonb_typeof(p_scope) IS DISTINCT FROM 'object'
  OR p_scope->>'round_id' IS DISTINCT FROM v_run.round_id
  OR p_scope->>'evidence_judgment_policy' IS DISTINCT FROM 'evidence_judgment_v2'
  OR (p_signal_index=-1) IS DISTINCT FROM COALESCE(p_scope->>'checkpoint_kind'='company_fit_evidence_v2',FALSE)
  OR (p_signal_index>=0 AND (p_scope ? 'checkpoint_kind' OR p_scope->>'judgment_kind' IS DISTINCT FROM 'signal_evidence_v2'))
  OR COALESCE(p_scope->>'family_hash','') !~ '^sha256:[0-9a-f]{64}$' THEN
  RAISE EXCEPTION 'lab_arena_evidence_request_invalid' USING ERRCODE='22023'; END IF;
 SELECT r.configuration_doc->'scorer_policy' INTO v_policy FROM public.lab_arena_rounds r
  WHERE r.round_id=v_run.round_id
   AND r.configuration_doc#>>'{scorer_policy,evidence_judgment_policy}'='evidence_judgment_v2'
   AND p_scope->>'evaluation_date'=r.evaluation_date
   AND p_scope->>'scorer_image_digest'=r.configuration_doc->>'scorer_image_digest'
   AND p_scope->>'scorer_image_reference'=r.configuration_doc->>'scorer_image_reference'
   AND p_scope->>'runtime_image_digest'=COALESCE(v_run.reviewer_runtime_doc->>'runtime_image_digest',r.configuration_doc->>'scorer_image_digest')
   AND p_scope->>'runtime_image_reference'=COALESCE(v_run.reviewer_runtime_doc->>'runtime_image_reference',r.configuration_doc->>'scorer_image_reference');
 IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_evidence_policy_invalid' USING ERRCODE='22023'; END IF;
 IF (p_operation='claim' OR p_signal_index>=0) AND (
   COALESCE(p_scope->>'evidence_hash','') !~ '^sha256:[0-9a-f]{64}$'
   OR jsonb_typeof(p_scope->'evidence_input') IS DISTINCT FROM 'object'
   OR p_scope->'evidence_input'='{}'::JSONB) THEN
  RAISE EXCEPTION 'lab_arena_current_evidence_required' USING ERRCODE='22023'; END IF;
 IF p_operation='candidates' THEN
  SELECT COALESCE(jsonb_agg(candidate),'[]'::JSONB) INTO v_candidates FROM (
   SELECT jsonb_build_object('reference',jsonb_build_object(
    'schema_version','leadpoet.lab_arena.signal_judgment_reference.v1','cache_key',j.cache_key,
    'signal_input_hash',j.signal_input_hash,'authority_slot',j.authority_slot),
    'scope',j.scope_doc,'judgment',j.judgment_doc) candidate
   FROM public.lab_arena_signal_judgments j JOIN public.lab_arena_runs s ON s.run_id=j.source_score_run_id
   WHERE j.scope_doc->>'evidence_judgment_policy'='evidence_judgment_v2'
    AND j.scope_doc->>'family_hash'=p_scope->>'family_hash'
    AND (p_signal_index=-1 OR j.scope_doc->>'evidence_hash'=p_scope->>'evidence_hash')
    AND (NOT j.runner_authority_exclusions @> jsonb_build_array(v_run.miner_hotkey) OR j.source_score_run_id=p_run_id)
    AND s.kind='score' AND s.status IN ('leased','accepted','failed') AND s.round_id=v_run.round_id
    AND s.scored_run_id=j.source_scored_run_id AND s.runner_hotkey=j.source_runner_hotkey
    AND s.claim_request_hash=j.source_claim_request_hash AND s.lease_generation=j.source_lease_generation
    AND NOT EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions retired621
     WHERE retired621.round_id='arena-2026-11-04'
      AND retired621.request_id='nov04-baseline-source621-v1'
      AND retired621.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND retired621.recovery_doc->'superseded_runs' ? j.source_scored_run_id
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'kind'='execute'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'round_id'='arena-2026-11-04'
      AND retired621.recovery_doc->'superseded_runs'->j.source_scored_run_id->>'submission_id'='baseline-2026-11-04')
   ORDER BY j.created_at DESC,j.cache_key,j.authority_slot LIMIT 8) bounded;
  RETURN jsonb_build_object('status','candidates','candidates',v_candidates);
 END IF;
 IF p_candidate_reference IS NULL THEN
  RETURN public.lab_arena_claim_signal_judgment_v1(p_run_id,p_lease_token_hash,p_company_index,p_signal_index,p_scope);
 END IF;
 SELECT * INTO v_row FROM public.lab_arena_signal_judgments j
  WHERE j.cache_key=p_candidate_reference->>'cache_key'
   AND j.authority_slot=(p_candidate_reference->>'authority_slot')::INTEGER;
 IF NOT FOUND THEN RAISE EXCEPTION 'lab_arena_evidence_candidate_missing' USING ERRCODE='22023'; END IF;
 v_ref:=jsonb_build_object('schema_version','leadpoet.lab_arena.signal_judgment_reference.v1',
  'cache_key',v_row.cache_key,'signal_input_hash',v_row.signal_input_hash,'authority_slot',v_row.authority_slot);
 IF p_candidate_reference IS DISTINCT FROM v_ref
  OR v_row.scope_doc->>'evidence_judgment_policy' IS DISTINCT FROM 'evidence_judgment_v2'
  OR v_row.scope_doc->>'family_hash' IS DISTINCT FROM p_scope->>'family_hash'
  OR v_row.scope_doc->'family_doc' IS DISTINCT FROM p_scope->'family_doc'
  OR v_row.scope_doc->>'evidence_hash' IS DISTINCT FROM p_scope->>'evidence_hash'
  OR v_row.scope_doc->'evidence_input' IS DISTINCT FROM p_scope->'evidence_input'
  OR (v_row.runner_authority_exclusions @> jsonb_build_array(v_run.miner_hotkey)
      AND v_row.source_score_run_id<>p_run_id)
  OR EXISTS (SELECT 1 FROM public.lab_arena_scoring_supersessions retired621
     WHERE retired621.round_id='arena-2026-11-04'
      AND retired621.request_id='nov04-baseline-source621-v1'
      AND retired621.recovery_doc->>'operation'='baseline_source_supersession_v1'
      AND retired621.recovery_doc->'superseded_runs' ? v_row.source_scored_run_id
      AND retired621.recovery_doc->'superseded_runs'->v_row.source_scored_run_id->>'kind'='execute'
      AND retired621.recovery_doc->'superseded_runs'->v_row.source_scored_run_id->>'round_id'='arena-2026-11-04'
      AND retired621.recovery_doc->'superseded_runs'->v_row.source_scored_run_id->>'submission_id'='baseline-2026-11-04') THEN
  RAISE EXCEPTION 'lab_arena_evidence_candidate_invalid' USING ERRCODE='22023'; END IF;
 SELECT * INTO v_source FROM public.lab_arena_runs WHERE run_id=v_row.source_score_run_id;
 IF NOT FOUND OR v_source.kind<>'score' OR v_source.status NOT IN ('leased','accepted','failed')
  OR v_source.round_id<>v_run.round_id OR v_source.scored_run_id<>v_row.source_scored_run_id
  OR v_source.runner_hotkey<>v_row.source_runner_hotkey OR v_source.claim_request_hash<>v_row.source_claim_request_hash
  OR v_source.lease_generation<>v_row.source_lease_generation THEN
  RAISE EXCEPTION 'lab_arena_evidence_source_invalid' USING ERRCODE='22023'; END IF;
 IF v_row.scope_doc->'claim' IS DISTINCT FROM p_scope->'claim' THEN
  v_model:=COALESCE(v_policy#>>'{judge_models,intent_fetched_evidence}',v_policy#>>'{judge_models,intent_three_stage_stage3}');
  IF p_equivalence_receipt->>'schema_version' IS DISTINCT FROM 'leadpoet.lab_arena.claim_equivalence.v1'
   OR p_equivalence_receipt->>'model' IS DISTINCT FROM v_model OR COALESCE(v_model,'')=''
   OR p_equivalence_receipt->>'decision' IS DISTINCT FROM 'equivalent'
   OR p_equivalence_receipt->'material_differences' IS DISTINCT FROM '[]'::JSONB
   OR p_equivalence_receipt->>'source_claim_hash' IS DISTINCT FROM v_row.scope_doc->>'claim_hash'
   OR p_equivalence_receipt->>'current_claim_hash' IS DISTINCT FROM p_scope->>'claim_hash' THEN
   RAISE EXCEPTION 'lab_arena_evidence_equivalence_invalid' USING ERRCODE='22023'; END IF;
  v_receipt:=p_equivalence_receipt;
 ELSE v_receipt:=NULL; END IF;
 SELECT * INTO v_use FROM public.lab_arena_signal_judgment_uses
  WHERE run_id=p_run_id AND company_index=p_company_index AND signal_index=p_signal_index;
 IF FOUND THEN
  IF v_use.cache_key<>v_row.cache_key OR v_use.authority_slot<>v_row.authority_slot
   OR (v_use.acceptance_doc->'current_scope' IS DISTINCT FROM p_scope
       AND NOT (v_use.acceptance_doc IS NULL AND v_row.scope_doc=p_scope
                AND v_use.signal_input_hash=v_row.signal_input_hash
                AND v_use.judgment_hash=v_row.judgment_hash
                AND v_use.source_score_run_id=v_row.source_score_run_id)) THEN
   RAISE EXCEPTION 'lab_arena_signal_use_conflict' USING ERRCODE='23505'; END IF;
 ELSE
  INSERT INTO public.lab_arena_signal_judgment_uses
   (run_id,company_index,signal_index,cache_key,authority_slot,signal_input_hash,source_score_run_id,judgment_hash,acceptance_doc)
  VALUES (p_run_id,p_company_index,p_signal_index,v_row.cache_key,v_row.authority_slot,v_row.signal_input_hash,
   v_row.source_score_run_id,v_row.judgment_hash,jsonb_build_object('schema_version','leadpoet.lab_arena.evidence_use.v2',
    'current_scope',p_scope,'candidate_reference',v_ref,'equivalence_receipt',v_receipt));
 END IF;
 DELETE FROM public.lab_arena_signal_judgment_reservations WHERE cache_key=p_scope->>'cache_key'
  AND run_id=p_run_id AND lease_generation=v_run.lease_generation;
 RETURN jsonb_build_object('status','hit','reference',v_ref,'judgment',v_row.judgment_doc,'current_scope',p_scope);
END;
$$;
ALTER FUNCTION public.lab_arena_evidence_judgment_v2(TEXT,TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_evidence_judgment_v2(TEXT,TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB) FROM PUBLIC,anon,authenticated,service_role,lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_evidence_judgment_v2(TEXT,TEXT,TEXT,INTEGER,INTEGER,JSONB,JSONB,JSONB) TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;

-- A bound signed company-judgment partial is a durable failed-score receipt.
-- Preserve capture identity, terminal policy, frozen refs and ledger finality.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $company_partial_drain513$
DECLARE
  target OID:=to_regprocedure('public.lab_arena__restart_drain_state_v1()');
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  identity JSONB;
  old_gate CONSTANT TEXT := $old$              AND public.lab_arena__legacy_partial_bound(
                runs, runs.result_doc -> 'legacy_partial_checkpoint', runs.output_ref)))$old$;
  new_gate CONSTANT TEXT := $new$              AND public.lab_arena__legacy_partial_bound(
                runs, runs.result_doc -> 'legacy_partial_checkpoint', runs.output_ref))
            -- A signed v3 failure can retain immutable company judgments.
            OR (runs.kind = 'score'
              AND pg_catalog.char_length(runs.output_ref) BETWEEN 1 AND 1024
              AND runs.result_doc ->> 'company_judgment_completion_request_hash'
                ~ '^sha256:[0-9a-f]{64}$'
              AND EXISTS (
                SELECT 1 FROM public.lab_arena_company_judgments AS partial_company
                JOIN public.lab_arena_rounds AS frozen
                  ON frozen.round_id = runs.round_id
                WHERE (frozen.configuration_doc ->> 'company_quality_policy' = 'company_quality_v1'
                    OR public.lab_arena__company_judgment_policy_v1(frozen))
                  AND partial_company.source_score_run_id = runs.run_id
                  AND partial_company.source_scored_run_id = runs.scored_run_id
                  AND partial_company.source_runner_hotkey = runs.runner_hotkey
                  AND partial_company.evidence_hash ~ '^sha256:[0-9a-f]{64}$'
                  AND partial_company.evidence_doc ->> 'schema_version' =
                    'leadpoet.lab_arena.company_judgment_evidence.v1'
                  AND partial_company.evidence_doc ->> 'cache_key' = partial_company.cache_key
                  AND partial_company.evidence_doc ->> 'company_input_hash' = partial_company.company_input_hash
                  AND partial_company.evidence_doc -> 'authority_slot' = to_jsonb(partial_company.authority_slot)
                  AND partial_company.evidence_doc ->> 'source_score_run_id' = runs.run_id
                  AND partial_company.evidence_doc ->> 'source_scored_run_id' = runs.scored_run_id
                  AND partial_company.evidence_doc ->> 'source_runner_hotkey' = runs.runner_hotkey
                  AND partial_company.evidence_doc ->> 'source_output_ref' = runs.output_ref
                  AND partial_company.evidence_doc ->> 'source_output_hash' ~ '^sha256:[0-9a-f]{64}$'
                  AND partial_company.evidence_doc ->> 'source_completion_request_hash' =
                    runs.result_doc ->> 'company_judgment_completion_request_hash'
                  AND partial_company.evidence_doc ->> 'source_claim_request_id' = runs.claim_request_id
                  AND partial_company.evidence_doc ->> 'source_claim_request_hash' = runs.claim_request_hash
                  AND partial_company.evidence_doc -> 'source_lease_generation' = to_jsonb(runs.lease_generation)
                  AND partial_company.evidence_doc -> 'runner_authority_exclusions' =
                    runs.claim_response -> 'runner_authority_exclusions'
                  AND EXISTS (
                    SELECT 1 FROM jsonb_array_elements(CASE
                      WHEN jsonb_typeof(runs.company_judgment_refs) = 'array'
                      THEN runs.company_judgment_refs ELSE '[]'::jsonb END) AS refs(ref)
                    JOIN jsonb_array_elements(CASE
                      WHEN jsonb_typeof(runs.claim_response #> '{company_judgment_cache,misses}') = 'array'
                      THEN runs.claim_response #> '{company_judgment_cache,misses}' ELSE '[]'::jsonb END) AS misses(miss)
                      ON miss ->> 'cache_key' = ref ->> 'cache_key'
                    WHERE ref ->> 'cache_key' = partial_company.cache_key
                      AND ref ->> 'company_input_hash' = partial_company.company_input_hash
                      AND ref -> 'scope_doc' = partial_company.scope_doc
                      AND miss ->> 'company_input_hash' = partial_company.company_input_hash
                      AND miss -> 'authority_slot' = to_jsonb(partial_company.authority_slot)
                  )
              )))$new$;
BEGIN
  SELECT pg_get_functiondef(p.oid),to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname)
  INTO definition,metadata,identity FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target;
  IF definition IS NULL OR identity IS DISTINCT FROM
    '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'Company partial drain security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='49f2d7713419d20c182bf1548f9e731cb369611b5121ab6b116239ba2a8f77f2' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')<>'3ed11092880c10bad608420b33c076773557a308288d39ca0a45df9a95990e5a'
    OR (length(definition)-length(replace(definition,old_gate,'')))<>length(old_gate) THEN
    RAISE EXCEPTION 'Company partial drain preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,old_gate,new_gate);
  IF encode(extensions.digest(updated,'sha256'),'hex')<>'49f2d7713419d20c182bf1548f9e731cb369611b5121ab6b116239ba2a8f77f2' THEN
    RAISE EXCEPTION 'Company partial drain postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex')<>'49f2d7713419d20c182bf1548f9e731cb369611b5121ab6b116239ba2a8f77f2' THEN
    RAISE EXCEPTION 'Company partial drain readback differs' USING ERRCODE='55000';
  END IF;
END;
$company_partial_drain513$;
COMMIT;

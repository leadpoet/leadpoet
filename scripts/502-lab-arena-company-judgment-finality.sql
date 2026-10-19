-- Company-cache scores freeze ordered per-company scopes instead of the
-- legacy whole-item scope. Finality and miner-credit authority must recognize
-- that same frozen protocol. Accepted work, costs and retry rules stay intact.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $company_judgment_finality$
DECLARE patch RECORD; definition TEXT; before_identity JSONB; after_identity JSONB;
BEGIN
  IF to_regprocedure('public.lab_arena__company_judgment_refs_v1(public.lab_arena_rounds,jsonb)') IS NULL
    OR to_regprocedure('public.lab_arena__company_judgment_policy_v1(public.lab_arena_rounds)') IS NULL THEN
    RAISE EXCEPTION 'company judgment finality requires migration501'; END IF;
  FOR patch IN SELECT * FROM (VALUES
    ('public.lab_arena__all_judgments_ready_v1(text)',
      $old$AND judge.judgment_scope_doc->>'scorer_image_digest'=
                    round_row.configuration_doc->>'scorer_image_digest'
                AND judge.judgment_scope_doc->>'scorer_image_reference'=
                    round_row.configuration_doc->>'scorer_image_reference'$old$,
      $new$AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(round_row) THEN
                  public.lab_arena__company_judgment_refs_v1(round_row,judge.company_judgment_refs)
                  AND judge.judgment_cache_key IS NULL
                  AND judge.judgment_input_hash IS NULL
                  AND judge.judgment_scope_doc IS NULL
                ELSE (
                  NOT round_row.configuration_doc ? 'company_judgment_policy'
                  AND NOT (round_row.configuration_doc->'scorer_policy') ? 'company_judgment_policy'
                  AND judge.judgment_scope_doc->>'scorer_image_digest'=
                    round_row.configuration_doc->>'scorer_image_digest'
                AND judge.judgment_scope_doc->>'scorer_image_reference'=
                    round_row.configuration_doc->>'scorer_image_reference'
                ) END)$new$),
    ('public.lab_arena_miner_credit_failure_v1(text)',
      $old$AND r.judgment_scope_doc->>'round_id'=r.round_id
        AND r.judgment_scope_doc->>'scoring_input_hash'=r.judgment_input_hash
        AND r.judgment_scope_doc->>'cache_key'=r.judgment_cache_key
        AND r.judgment_scope_doc->>'scorer_image_digest'=round_row.configuration_doc->>'scorer_image_digest'
        AND r.judgment_scope_doc->>'scorer_image_reference'=round_row.configuration_doc->>'scorer_image_reference'$old$,
      $new$AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(round_row) THEN
          public.lab_arena__company_judgment_refs_v1(round_row,r.company_judgment_refs)
          AND r.judgment_cache_key IS NULL
          AND r.judgment_input_hash IS NULL
          AND r.judgment_scope_doc IS NULL
        ELSE (
          NOT round_row.configuration_doc ? 'company_judgment_policy'
          AND NOT (round_row.configuration_doc->'scorer_policy') ? 'company_judgment_policy'
          AND r.judgment_scope_doc->>'round_id'=r.round_id
        AND r.judgment_scope_doc->>'scoring_input_hash'=r.judgment_input_hash
        AND r.judgment_scope_doc->>'cache_key'=r.judgment_cache_key
        AND r.judgment_scope_doc->>'scorer_image_digest'=round_row.configuration_doc->>'scorer_image_digest'
        AND r.judgment_scope_doc->>'scorer_image_reference'=round_row.configuration_doc->>'scorer_image_reference'
        ) END)$new$),
    ('public.lab_arena_miner_credit_failure_v1(text)',
      $old$r.judgment_input_hash,r.judgment_scope_doc)
      OR EXISTS (SELECT 1 FROM public.lab_arena_runs intervening$old$,
      $new$r.judgment_input_hash,r.judgment_scope_doc)
      OR previous.company_judgment_refs IS DISTINCT FROM r.company_judgment_refs
      OR EXISTS (SELECT 1 FROM public.lab_arena_runs intervening$new$)
  ) patches(signature,old_text,new_text) LOOP
    SELECT pg_get_functiondef(p.oid),jsonb_build_array(pg_get_userbyid(p.proowner),
      p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO definition,before_identity FROM pg_proc p WHERE p.oid=to_regprocedure(patch.signature);
    IF definition IS NULL OR before_identity IS DISTINCT FROM jsonb_build_array(
      'lab_arena_owner',CASE WHEN patch.signature='public.lab_arena_miner_credit_failure_v1(text)'
        THEN '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
        ELSE '{lab_arena_owner=X/lab_arena_owner}' END,TRUE,'s',
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
      RAISE EXCEPTION 'company judgment finality security shape differs: %',patch.signature
        USING ERRCODE='55000'; END IF;
    IF strpos(definition,patch.new_text)>0 THEN CONTINUE; END IF;
    IF (length(definition)-length(replace(definition,patch.old_text,'')))/length(patch.old_text)<>1 THEN
      RAISE EXCEPTION 'company judgment finality anchor differs: %',patch.signature
        USING ERRCODE='55000'; END IF;
    EXECUTE replace(definition,patch.old_text,patch.new_text);
    SELECT jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig) INTO after_identity
      FROM pg_proc p WHERE p.oid=to_regprocedure(patch.signature);
    IF after_identity IS DISTINCT FROM before_identity THEN
      RAISE EXCEPTION 'company judgment finality security changed: %',patch.signature
        USING ERRCODE='55000'; END IF;
  END LOOP;
END;
$company_judgment_finality$;
NOTIFY pgrst,'reload schema';
COMMIT;

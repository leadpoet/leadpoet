-- Materialize the unchanged frozen configuration once before repeatedly
-- checking accepted judgments. No round, run, or ledger row is changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $guard548$
DECLARE target OID:='public.lab_arena__all_judgments_ready_v1(text)'::regprocedure;
BEGIN
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex') NOT IN (
   '5188eedfb5b60116979b9a81a203b972384df208a21cf301749240e5a9f4e071',
   '39e4ca21bb87af27fd0114b5f2d562a445ca93c2ef6ae3a515e1445dc0d6d1dc') THEN
   RAISE EXCEPTION 'Arena judgment validation preimage differs' USING ERRCODE='55000';
 END IF;
 IF NOT EXISTS (SELECT 1 FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
    WHERE p.oid=target AND pg_get_userbyid(p.proowner)='lab_arena_owner'
      AND p.proacl::text='{lab_arena_owner=X/lab_arena_owner}'
      AND p.prosecdef AND p.provolatile='s'
      AND p.proconfig=ARRAY['search_path=pg_catalog, public']::text[]
      AND l.lanname='sql') THEN
   RAISE EXCEPTION 'Arena judgment validation security differs' USING ERRCODE='42501';
 END IF;
END $guard548$;

CREATE OR REPLACE FUNCTION public.lab_arena__all_judgments_ready_v1(p_round_id text)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'pg_catalog', 'public'
AS $function$
  WITH validation_round548 AS MATERIALIZED (
    SELECT jsonb_populate_record(r,jsonb_build_object('configuration_doc',
      CASE WHEN jsonb_typeof(r.configuration_doc)='object'
        THEN r.configuration_doc || '{}'::jsonb
        ELSE r.configuration_doc END)) AS round_row
    FROM public.lab_arena_rounds r WHERE r.round_id=p_round_id),
  incomplete540 AS MATERIALIZED (
    SELECT participant->>'submission_id' AS submission_id,
      public.lab_arena__participant_incomplete_reason_v1(p_round_id,participant->>'submission_id') AS reason
    FROM public.lab_arena_rounds r,jsonb_array_elements(r.participants)participant WHERE r.round_id=p_round_id)
  SELECT NOT EXISTS (SELECT 1 FROM public.lab_arena_runs source
    WHERE source.round_id=p_round_id AND source.kind='execute'
      AND NOT EXISTS(SELECT 1 FROM incomplete540 i WHERE i.submission_id=source.submission_id AND i.reason IS NOT NULL)
      AND EXISTS(SELECT 1 FROM public.lab_arena_rounds admitted_round,
        jsonb_array_elements(admitted_round.participants) admitted
        WHERE admitted_round.round_id=p_round_id AND admitted->>'submission_id'=source.submission_id)
      AND source.stage IN (1,2) AND source.status='accepted'
      AND NOT public.lab_arena__miner_credit_judgment_terminal_v1(source.run_id)
      AND (source.terminal_cause<>'accepted' OR source.output_ref IS NULL
        OR source.per_icp_score IS NULL
        OR NOT EXISTS (SELECT 1 FROM public.lab_arena_runs judge
          WHERE judge.assignment_id=source.assignment_id||':score'
            AND judge.scored_run_id=source.run_id AND judge.kind='score'
            AND judge.status='accepted' AND judge.terminal_cause='accepted'
            AND judge.output_ref IS NOT NULL
            AND (judge.round_id,judge.submission_id,judge.miner_hotkey,
                 judge.stage,judge.icp_position) IS NOT DISTINCT FROM
                (source.round_id,source.submission_id,source.miner_hotkey,
                 source.stage,source.icp_position)
            AND EXISTS (SELECT 1 FROM validation_round548 validation
              WHERE (validation.round_row).round_id=p_round_id
                AND judge.scoring_generation_floor=(validation.round_row).scoring_generation_floor
                AND (CASE WHEN public.lab_arena__company_judgment_policy_v1(validation.round_row) THEN
                  public.lab_arena__company_judgment_refs_v1(validation.round_row,judge.company_judgment_refs)
                  AND judge.judgment_cache_key IS NULL
                  AND judge.judgment_input_hash IS NULL
                  AND judge.judgment_scope_doc IS NULL
                ELSE (
                  NOT (validation.round_row).configuration_doc ? 'company_judgment_policy'
                  AND NOT ((validation.round_row).configuration_doc->'scorer_policy') ? 'company_judgment_policy'
                  AND judge.judgment_scope_doc->>'scorer_image_digest'=
                    (validation.round_row).configuration_doc->>'scorer_image_digest'
                AND judge.judgment_scope_doc->>'scorer_image_reference'=
                    (validation.round_row).configuration_doc->>'scorer_image_reference'
                ) END)))));
$function$
;
DO $verify548$
DECLARE target OID:='public.lab_arena__all_judgments_ready_v1(text)'::regprocedure;
BEGIN
 IF encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex') <>
   '39e4ca21bb87af27fd0114b5f2d562a445ca93c2ef6ae3a515e1445dc0d6d1dc' THEN
   RAISE EXCEPTION 'Arena judgment validation postimage differs: %',
     encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex') USING ERRCODE='55000';
 END IF;
END $verify548$;
COMMIT;

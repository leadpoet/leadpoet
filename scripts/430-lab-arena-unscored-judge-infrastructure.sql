-- An exhausted judge infrastructure failure has no measured ICP score.
-- Extend only the existing partial-publication proof. Keep accepted scores,
-- payer failures, frozen deadlines, and every other publication guard intact.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $unscored_judge_infrastructure_430$
DECLARE
  v_signature CONSTANT TEXT :=
    'public.lab_arena__publication_execution_incomplete_v1(text,text)';
  v_preimage CONSTANT TEXT :=
    'f89c2e9892a44a6a0f89ee4ada448d5395b3bee7ad48f1b246ed7821d0dd37d4';
  v_postimage CONSTANT TEXT :=
    '1d0b203f8e0b8d2ece0db3c288806e85dc310a69b9786fce27e0ca9c3d308ecf';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
  v_old TEXT := $old$      IF pg_catalog.now() <
           (v_round.configuration_doc #>> ARRAY['schedule', v_close_key])::TIMESTAMPTZ
         OR NOT EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS judge
           WHERE judge.round_id = p_round_id
             AND judge.submission_id = p_submission_id
             AND judge.kind = 'score'
             AND judge.scored_run_id = v_execution.run_id
             AND judge.status = 'failed'
             AND judge.terminal_cause = 'stage_closed'
         )
         OR EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS judge
           WHERE judge.round_id = p_round_id
             AND judge.submission_id = p_submission_id
             AND judge.kind = 'score'
             AND judge.scored_run_id = v_execution.run_id
             AND judge.status = 'accepted'
         ) THEN
        RETURN FALSE;
      END IF;$old$;
  v_new TEXT := $new$      -- lab_arena_unscored_judge_infrastructure_430: use the
      -- effective attempt, not an older failure or another cache alias.
      IF NOT EXISTS (
        SELECT 1
        FROM (
          SELECT judge.* FROM public.lab_arena_runs AS judge
          WHERE judge.round_id = p_round_id
            AND judge.submission_id = p_submission_id
            AND judge.kind = 'score'
            AND judge.scored_run_id = v_execution.run_id
          ORDER BY (judge.status = 'accepted') DESC,
                   judge.stage_generation DESC, judge.attempt DESC,
                   judge.run_id DESC
          LIMIT 1
        ) AS effective
        WHERE effective.status = 'failed'
          AND (
            (effective.terminal_cause = 'stage_closed'
             AND pg_catalog.now() >=
               (v_round.configuration_doc #>> ARRAY[
                 'schedule', v_close_key])::TIMESTAMPTZ)
            OR (effective.terminal_cause IN ('judge_error','judge_timeout')
              AND (
                (v_round.configuration_doc ->> 'max_attempts_per_assignment'
                   = '2' AND effective.attempt >= 2)
                OR pg_catalog.now() >=
                  (v_round.configuration_doc #>> ARRAY[
                    'schedule', v_close_key])::TIMESTAMPTZ
              ))
          )
          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_runs AS active
            WHERE active.round_id = p_round_id
              AND active.kind = 'score'
              AND active.status IN ('pending','leased','submitted')
              AND (
                active.scored_run_id = v_execution.run_id
                OR (effective.judgment_cache_key IS NOT NULL
                    AND active.stage = effective.stage
                    AND active.stage_generation = effective.stage_generation
                    AND active.judgment_cache_key =
                        effective.judgment_cache_key)
              )
          )
          -- A cache hit can still complete an alias before the deadline.
          AND (pg_catalog.now() >=
                (v_round.configuration_doc #>> ARRAY[
                  'schedule', v_close_key])::TIMESTAMPTZ
               OR effective.judgment_cache_key IS NULL
               OR NOT EXISTS (
                 SELECT 1 FROM public.lab_arena_judgment_cache AS cache
                 WHERE cache.cache_key = effective.judgment_cache_key
               ))
      ) THEN
        RETURN FALSE;
      END IF;$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
         pg_catalog.jsonb_build_array(owner.rolname, p.proacl::TEXT,
           p.prosecdef, p.provolatile, p.proconfig)
  INTO v_definition, v_identity
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner}', TRUE, 's',
      ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena unscored judge proof security shape differs';
  END IF;
  v_hash := pg_catalog.encode(
    extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = v_postimage THEN RETURN; END IF;
  IF v_hash <> v_preimage THEN
    RAISE EXCEPTION 'Arena unscored judge proof preimage differs';
  END IF;
  IF pg_catalog.length(v_definition) - pg_catalog.length(
       pg_catalog.replace(v_definition, v_old, ''))
       <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena unscored judge proof fragment differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       <> v_postimage THEN
    RAISE EXCEPTION 'Arena unscored judge proof postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(
       pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),
       'sha256'), 'hex') <> v_postimage THEN
    RAISE EXCEPTION 'Arena unscored judge proof readback differs';
  END IF;
END;
$unscored_judge_infrastructure_430$;
COMMIT;

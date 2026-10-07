-- Publish complete participants when another assignment was closed unfinished.
-- The incomplete participant keeps a NULL aggregate and no cost claim. Every
-- complete ranking still passes the existing score and cost guards unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE lab_arena_423_schema_acl ON COMMIT DROP AS
SELECT namespace.nspacl AS acl,
       pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE')
         AS had_create
FROM pg_catalog.pg_namespace AS namespace
WHERE namespace.nspname = 'public';
DO $temporary_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_423_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_create$;

CREATE OR REPLACE FUNCTION public.lab_arena__publication_execution_incomplete_v1(
  p_round_id TEXT, p_submission_id TEXT
)
RETURNS BOOLEAN
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $partial_publication$
DECLARE
  v_round public.lab_arena_rounds;
  v_count INTEGER;
  v_position INTEGER;
  v_scored BOOLEAN;
  v_missing INTEGER := 0;
  v_execution public.lab_arena_runs;
  v_close_key TEXT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id;
  IF NOT FOUND OR v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'sourcing_cost_eligibility_policy'
       IS DISTINCT FROM 'successful_calls_per_icp_v1'
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
       WHERE participant ->> 'submission_id' = p_submission_id
     ) THEN
    RETURN FALSE;
  END IF;
  v_count := (v_round.configuration_doc ->> 'stage_1_icp_count')::INTEGER
           + (v_round.configuration_doc ->> 'stage_2_icp_count')::INTEGER;
  IF v_count NOT BETWEEN 2 AND 100 THEN
    RETURN FALSE;
  END IF;
  FOR v_position IN 0..v_count - 1 LOOP
    SELECT EXISTS (
      SELECT 1 FROM public.lab_arena_runs AS run
      WHERE run.round_id = p_round_id
        AND run.submission_id = p_submission_id
        AND run.kind = 'execute'
        AND run.icp_position = v_position
        AND run.per_icp_score IS NOT NULL
    ) INTO v_scored;
    IF v_scored THEN
      IF NOT COALESCE((public.lab_arena__integrity_submission_summary(
          p_round_id, p_submission_id, ARRAY[v_position]
        ) ->> 'valid')::BOOLEAN, FALSE) THEN
        RETURN FALSE;
      END IF;
      CONTINUE;
    END IF;
    v_missing := v_missing + 1;
    IF EXISTS (
      SELECT 1 FROM public.lab_arena_runs AS judge
      JOIN public.lab_arena_runs AS execution
        ON execution.run_id = judge.scored_run_id
      WHERE execution.round_id = p_round_id
        AND execution.submission_id = p_submission_id
        AND execution.kind = 'execute'
        AND execution.icp_position = v_position
        AND judge.kind = 'score'
        AND judge.status = 'accepted'
    ) THEN
      RETURN FALSE;
    END IF;
    -- An accepted execution must have its own unfinished judge assignment.
    -- A missing judge row cannot turn accepted work into an incomplete result.
    SELECT * INTO v_execution FROM public.lab_arena_runs AS run
    WHERE run.round_id = p_round_id
      AND run.submission_id = p_submission_id
      AND run.kind = 'execute'
      AND run.icp_position = v_position
      AND run.status = 'accepted'
    ORDER BY run.attempt DESC, run.run_id DESC LIMIT 1;
    IF FOUND THEN
      v_close_key := CASE v_execution.stage
        WHEN 1 THEN 'stage_1_scoring_close'
        WHEN 2 THEN 'final_scoring_close'
        ELSE NULL END;
      IF v_close_key IS NULL OR pg_catalog.jsonb_typeof(
           v_round.configuration_doc #> ARRAY['schedule', v_close_key]
         ) IS DISTINCT FROM 'string' THEN
        RETURN FALSE;
      END IF;
      IF pg_catalog.now() <
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
      END IF;
    ELSE
      SELECT * INTO v_execution FROM public.lab_arena_runs AS run
      WHERE run.round_id = p_round_id
        AND run.submission_id = p_submission_id
        AND run.kind = 'execute'
        AND run.icp_position = v_position
      ORDER BY run.attempt DESC, run.run_id DESC LIMIT 1;
      IF NOT FOUND THEN
        RETURN FALSE;
      END IF;
      v_close_key := CASE v_execution.stage
        WHEN 1 THEN 'stage_1_close'
        WHEN 2 THEN 'stage_2_close'
        ELSE NULL END;
      IF v_close_key IS NULL OR pg_catalog.jsonb_typeof(
           v_round.configuration_doc #> ARRAY['schedule', v_close_key]
         ) IS DISTINCT FROM 'string' THEN
        RETURN FALSE;
      END IF;
      IF pg_catalog.now() <
           (v_round.configuration_doc #>> ARRAY['schedule', v_close_key])::TIMESTAMPTZ
         OR v_execution.status IS DISTINCT FROM 'failed'
         OR v_execution.terminal_doc -> 'infrastructure_incomplete'
              IS DISTINCT FROM 'true'::JSONB
         OR EXISTS (
           SELECT 1 FROM public.lab_arena_runs AS judge
           WHERE judge.round_id = p_round_id
             AND judge.kind = 'score'
             AND judge.scored_run_id = v_execution.run_id
             AND judge.status = 'accepted'
         ) THEN
        RETURN FALSE;
      END IF;
    END IF;
  END LOOP;
  RETURN v_missing > 0;
END;
$partial_publication$;
ALTER FUNCTION public.lab_arena__publication_execution_incomplete_v1(TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__publication_execution_incomplete_v1(TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;

DO $patch_publication$
DECLARE
  v_guard TEXT;
  v_baseline TEXT;
  v_guard_owner NAME;
  v_baseline_owner NAME;
  v_guard_acl ACLITEM[];
  v_baseline_acl ACLITEM[];
  v_guard_security BOOLEAN;
  v_baseline_security BOOLEAN;
  v_guard_volatility "char";
  v_baseline_volatility "char";
  v_guard_config TEXT[];
  v_baseline_config TEXT[];
  v_old TEXT;
  v_new TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(proc.oid), owner.rolname, proc.proacl,
         proc.prosecdef, proc.provolatile, proc.proconfig
  INTO v_guard, v_guard_owner, v_guard_acl, v_guard_security,
       v_guard_volatility, v_guard_config
  FROM pg_catalog.pg_proc AS proc
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = proc.proowner
  WHERE proc.oid = 'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE;
  SELECT pg_catalog.pg_get_functiondef(proc.oid), owner.rolname, proc.proacl,
         proc.prosecdef, proc.provolatile, proc.proconfig
  INTO v_baseline, v_baseline_owner, v_baseline_acl, v_baseline_security,
       v_baseline_volatility, v_baseline_config
  FROM pg_catalog.pg_proc AS proc
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = proc.proowner
  WHERE proc.oid = 'public.lab_arena_publication_baseline_guard_v1()'::REGPROCEDURE;
  IF v_guard_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_baseline_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_guard_security OR v_baseline_security
     OR v_guard_volatility IS DISTINCT FROM 'v'
     OR v_baseline_volatility IS DISTINCT FROM 'v' THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_guard_shape_changed';
  END IF;
  IF pg_catalog.strpos(v_guard, 'lab_arena_partial_publication_423') > 0
     AND pg_catalog.strpos(v_baseline, 'lab_arena_partial_publication_423') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_guard, 'lab_arena_partial_publication_423') > 0
     OR pg_catalog.strpos(v_baseline, 'lab_arena_partial_publication_423') > 0 THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_guard_partial_patch';
  END IF;

  v_old := $old$    IF v_ranked_count <> 0 THEN$old$;
  v_new := $new$    -- lab_arena_partial_publication_423
    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_per_icp_v1'
       AND public.lab_arena__publication_execution_incomplete_v1(
         NEW.round_id, v_submission_id
       ) THEN
      IF v_ranked_count <> 1 THEN
        RAISE EXCEPTION 'lab_arena_publication_ranking_incomplete'
          USING ERRCODE = '22023';
      END IF;
      CONTINUE;
    END IF;
    IF v_ranked_count <> 0 THEN$new$;
  IF pg_catalog.strpos(v_guard, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_ranking_anchor_changed';
  END IF;
  v_guard := pg_catalog.replace(v_guard, v_old, v_new);

  v_old := $old$    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_per_icp_v1' THEN
      PERFORM public.lab_arena__per_icp_publication_valid(NEW.round_id, v_ranking);$old$;
  v_new := $new$    -- A proven unfinished assignment has no comparable aggregate or
    -- published cost result. All fully scored rows take the original guard.
    IF v_ranking -> 'final_score' = 'null'::JSONB
       AND v_ranking -> 'cost_summary' = 'null'::JSONB
       AND (v_ranking ->> 'eligible')::BOOLEAN IS FALSE
       AND v_ranking ->> 'eligibility_reason' = 'execution_incomplete'
       AND public.lab_arena__publication_execution_incomplete_v1(
         NEW.round_id, v_submission_id
       ) THEN
      CONTINUE;
    END IF;
    IF NEW.configuration_doc ->> 'sourcing_cost_eligibility_policy'
         = 'successful_calls_per_icp_v1' THEN
      PERFORM public.lab_arena__per_icp_publication_valid(NEW.round_id, v_ranking);$new$;
  IF pg_catalog.strpos(v_guard, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_cost_anchor_changed';
  END IF;
  v_guard := pg_catalog.replace(v_guard, v_old, v_new);

  v_old := $old$  IF v_baseline_score IS NULL THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;$old$;
  v_new := $new$  IF v_baseline_score IS NULL
     AND NOT (
       NEW.publication_doc #>> '{king_decision,outcome}' = 'no_king'
       AND public.lab_arena__publication_execution_incomplete_v1(
         NEW.round_id, v_baseline_id
       )
     ) THEN
    RAISE EXCEPTION 'lab_arena_publication_baseline_invalid'
      USING ERRCODE = '22023';
  END IF;$new$;
  IF pg_catalog.strpos(v_guard, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_baseline_anchor_changed';
  END IF;
  v_guard := pg_catalog.replace(v_guard, v_old, v_new);

  v_old := $old$      IF pg_catalog.jsonb_typeof(v_summary) IS DISTINCT FROM 'object'$old$;
  v_new := $new$      -- lab_arena_partial_publication_423
      IF v_summary = 'null'::JSONB
         AND (v_ranking ->> 'eligible')::BOOLEAN IS FALSE
         AND v_ranking ->> 'eligibility_reason' = 'execution_incomplete'
         AND public.lab_arena__publication_execution_incomplete_v1(
           NEW.round_id, v_submission_id
         ) THEN
        CONTINUE;
      END IF;
      IF pg_catalog.jsonb_typeof(v_summary) IS DISTINCT FROM 'object'$new$;
  IF pg_catalog.strpos(v_baseline, v_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_baseline_cost_anchor_changed';
  END IF;
  v_baseline := pg_catalog.replace(v_baseline, v_old, v_new);

  EXECUTE v_guard;
  EXECUTE v_baseline;
  IF (SELECT pg_catalog.jsonb_build_array(owner.rolname, proc.proacl::TEXT,
            proc.prosecdef, proc.provolatile, proc.proconfig)
      FROM pg_catalog.pg_proc AS proc
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = proc.proowner
      WHERE proc.oid = 'public.lab_arena_integrity_publication_guard_v1()'::REGPROCEDURE)
      IS DISTINCT FROM pg_catalog.jsonb_build_array(v_guard_owner, v_guard_acl::TEXT,
        v_guard_security, v_guard_volatility, v_guard_config)
     OR (SELECT pg_catalog.jsonb_build_array(owner.rolname, proc.proacl::TEXT,
            proc.prosecdef, proc.provolatile, proc.proconfig)
      FROM pg_catalog.pg_proc AS proc
      JOIN pg_catalog.pg_roles AS owner ON owner.oid = proc.proowner
      WHERE proc.oid = 'public.lab_arena_publication_baseline_guard_v1()'::REGPROCEDURE)
      IS DISTINCT FROM pg_catalog.jsonb_build_array(v_baseline_owner, v_baseline_acl::TEXT,
        v_baseline_security, v_baseline_volatility, v_baseline_config) THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_guard_permissions_changed';
  END IF;
END;
$patch_publication$;

DO $restore_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_423_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
       IS DISTINCT FROM (SELECT acl FROM pg_temp.lab_arena_423_schema_acl) THEN
    RAISE EXCEPTION 'lab_arena_partial_publication_schema_acl_changed';
  END IF;
END;
$restore_create$;
NOTIFY pgrst, 'reload schema';
COMMIT;

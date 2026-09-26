-- Keep an identical-judgment group claimable after its current leader uses
-- its final bounded judge attempt.  The claim RPC still admits only leaders;
-- completion and expiry promote exactly one follower after no retry is made.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $judgment_group_terminal_handoff_prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_complete_attempt(text,text,jsonb,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_expire_leases(text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
     ) IS NULL
     OR NOT EXISTS (
       SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'lab_arena_owner'
     ) THEN
    RAISE EXCEPTION 'apply current Arena migrations before migration 363';
  END IF;
END;
$judgment_group_terminal_handoff_prerequisites$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $judgment_group_completion_handoff_patch$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
  -- A miner-account failure belongs only to its submitted output. Let the
  -- next identical output obtain an independent judgment instead of leaving
  -- every non-leader permanently unclaimable. Judge infrastructure failures
  -- retain the same leader and normal retry path.
  IF v_run.kind = 'score'
     AND v_run.judgment_cache_key IS NOT NULL
     AND COALESCE(v_run.judgment_group_leader, FALSE)
     AND p_terminal_cause = 'credential_error' THEN
    UPDATE public.lab_arena_runs
    SET judgment_group_leader = TRUE
    WHERE run_id = (
      SELECT follower.run_id
      FROM public.lab_arena_runs AS follower
      WHERE follower.round_id = v_run.round_id
        AND follower.stage = v_run.stage
        AND follower.kind = 'score'
        AND follower.judgment_cache_key = v_run.judgment_cache_key
        AND follower.status = 'pending'
        AND NOT COALESCE(follower.judgment_group_leader, FALSE)
      ORDER BY follower.run_id
      FOR UPDATE
      LIMIT 1
    );
  END IF;
$old$;
  v_new TEXT := $new$
  -- lab_arena_judgment_group_terminal_handoff_v1: the normal retry branch
  -- returns above. A terminal leader failure now hands the unchanged cache
  -- identity to one follower instead of leaving every follower unclaimable.
  IF v_run.kind = 'score'
     AND v_run.judgment_cache_key IS NOT NULL
     AND COALESCE(v_run.judgment_group_leader, FALSE)
     AND p_terminal_cause IN (
       'credential_error', 'judge_error', 'judge_timeout'
     ) THEN
    -- The completion path already owns the run row. Use a group advisory lock
    -- instead of taking the round row in reverse order from expiry.
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'lab_arena.judgment-group:' || v_run.round_id || ':' ||
        v_run.stage::TEXT || ':' || v_run.stage_generation::TEXT || ':' ||
        v_run.judgment_cache_key,
        0
      )
    );
  END IF;
  IF v_run.kind = 'score'
     AND v_run.judgment_cache_key IS NOT NULL
     AND COALESCE(v_run.judgment_group_leader, FALSE)
     AND p_terminal_cause IN (
       'credential_error', 'judge_error', 'judge_timeout'
     )
     AND EXISTS (
       SELECT 1
       FROM public.lab_arena_rounds AS active_round
       WHERE active_round.round_id = v_run.round_id
         AND active_round.stage_generation = v_run.stage_generation
         AND active_round.status =
             'stage' || v_run.stage::TEXT || '_scoring'
     )
     AND NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_runs AS active_leader
       WHERE active_leader.round_id = v_run.round_id
         AND active_leader.stage = v_run.stage
         AND active_leader.stage_generation = v_run.stage_generation
         AND active_leader.kind = 'score'
         AND active_leader.judgment_cache_key = v_run.judgment_cache_key
         AND COALESCE(active_leader.judgment_group_leader, FALSE)
         AND active_leader.status IN ('pending', 'leased', 'submitted')
     )
     AND NOT EXISTS (
       SELECT 1
       FROM public.lab_arena_judgment_cache AS accepted_cache
       WHERE accepted_cache.cache_key = v_run.judgment_cache_key
     ) THEN
    UPDATE public.lab_arena_runs
    SET judgment_group_leader = TRUE
    WHERE run_id = (
      SELECT follower.run_id
      FROM public.lab_arena_runs AS follower
      WHERE follower.round_id = v_run.round_id
        AND follower.stage = v_run.stage
        AND follower.stage_generation = v_run.stage_generation
        AND follower.kind = 'score'
        AND follower.judgment_cache_key = v_run.judgment_cache_key
        AND follower.status = 'pending'
        AND NOT COALESCE(follower.judgment_group_leader, FALSE)
      ORDER BY follower.run_id
      FOR UPDATE
      LIMIT 1
    );
  END IF;
$new$;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_after_owner NAME;
  v_after_acl ACLITEM[];
  v_after_security_definer BOOLEAN;
  v_after_volatility "char";
  v_after_config TEXT[];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid), owner.rolname,
         procedure.proacl, procedure.prosecdef, procedure.provolatile,
         procedure.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;

  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR NOT COALESCE(v_security_definer, FALSE)
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config)) THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_security_shape_unexpected';
  END IF;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_judgment_group_terminal_handoff_v1'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(
         pg_catalog.substr(
           v_definition,
           pg_catalog.strpos(v_definition, v_old) + pg_catalog.length(v_old)
         ),
         v_old
       ) > 0 THEN
      RAISE EXCEPTION 'lab_arena_complete_attempt_handoff_shape_unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;

  SELECT owner.rolname, procedure.proacl, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
  INTO v_after_owner, v_after_acl, v_after_security_definer,
       v_after_volatility, v_after_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;

  IF v_after_owner IS DISTINCT FROM v_owner
     OR v_after_acl IS DISTINCT FROM v_acl
     OR v_after_security_definer IS DISTINCT FROM v_security_definer
     OR v_after_volatility IS DISTINCT FROM v_volatility
     OR v_after_config IS DISTINCT FROM v_config THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_permissions_changed';
  END IF;
END;
$judgment_group_completion_handoff_patch$;

DO $judgment_group_expiry_handoff_patch$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
      v_retried := v_retried + 1;
    END IF;
  END LOOP;
$old$;
  v_new TEXT := $new$
      v_retried := v_retried + 1;
    ELSIF v_run.kind = 'score'
       AND v_run.judgment_cache_key IS NOT NULL
       AND COALESCE(v_run.judgment_group_leader, FALSE)
       AND v_run.stage_generation = v_round.stage_generation
       AND v_round.status =
           'stage' || v_run.stage::TEXT || '_scoring'
       AND NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_runs AS active_leader
         WHERE active_leader.round_id = v_run.round_id
           AND active_leader.stage = v_run.stage
           AND active_leader.stage_generation = v_run.stage_generation
           AND active_leader.kind = 'score'
           AND active_leader.judgment_cache_key = v_run.judgment_cache_key
           AND COALESCE(active_leader.judgment_group_leader, FALSE)
           AND active_leader.status IN ('pending', 'leased', 'submitted')
       )
       AND NOT EXISTS (
         SELECT 1
         FROM public.lab_arena_judgment_cache AS accepted_cache
         WHERE accepted_cache.cache_key = v_run.judgment_cache_key
       ) THEN
      -- lab_arena_judgment_group_expiry_handoff_v1
      PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
          'lab_arena.judgment-group:' || v_run.round_id || ':' ||
          v_run.stage::TEXT || ':' || v_run.stage_generation::TEXT || ':' ||
          v_run.judgment_cache_key,
          0
        )
      );
      UPDATE public.lab_arena_runs
      SET judgment_group_leader = TRUE
      WHERE run_id = (
        SELECT follower.run_id
        FROM public.lab_arena_runs AS follower
        WHERE follower.round_id = v_run.round_id
          AND follower.stage = v_run.stage
          AND follower.stage_generation = v_run.stage_generation
          AND follower.kind = 'score'
          AND follower.judgment_cache_key = v_run.judgment_cache_key
          AND follower.status = 'pending'
          AND NOT COALESCE(follower.judgment_group_leader, FALSE)
        ORDER BY follower.run_id
        FOR UPDATE
        LIMIT 1
      );
    END IF;
  END LOOP;
$new$;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_after_owner NAME;
  v_after_acl ACLITEM[];
  v_after_security_definer BOOLEAN;
  v_after_volatility "char";
  v_after_config TEXT[];
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid), owner.rolname,
         procedure.proacl, procedure.prosecdef, procedure.provolatile,
         procedure.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_expire_leases'
    AND procedure.pronargs = 1;

  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR NOT COALESCE(v_security_definer, FALSE)
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public' = ANY(v_config)) THEN
    RAISE EXCEPTION 'lab_arena_expire_leases_security_shape_unexpected';
  END IF;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_judgment_group_expiry_handoff_v1'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0
       OR pg_catalog.strpos(
         pg_catalog.substr(
           v_definition,
           pg_catalog.strpos(v_definition, v_old) + pg_catalog.length(v_old)
         ),
         v_old
       ) > 0 THEN
      RAISE EXCEPTION 'lab_arena_expire_leases_handoff_shape_unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;

  SELECT owner.rolname, procedure.proacl, procedure.prosecdef,
         procedure.provolatile, procedure.proconfig
  INTO v_after_owner, v_after_acl, v_after_security_definer,
       v_after_volatility, v_after_config
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = procedure.proowner
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_expire_leases'
    AND procedure.pronargs = 1;

  IF v_after_owner IS DISTINCT FROM v_owner
     OR v_after_acl IS DISTINCT FROM v_acl
     OR v_after_security_definer IS DISTINCT FROM v_security_definer
     OR v_after_volatility IS DISTINCT FROM v_volatility
     OR v_after_config IS DISTINCT FROM v_config THEN
    RAISE EXCEPTION 'lab_arena_expire_leases_permissions_changed';
  END IF;
END;
$judgment_group_expiry_handoff_patch$;

-- Expiry already takes the round row before terminal handoff. Completion keeps
-- its existing shared round lock. Lock only active scoring rounds, in stable
-- order, before repairing their groups.
DO $judgment_group_repair_round_locks$
DECLARE
  v_round_id TEXT;
BEGIN
  FOR v_round_id IN
    SELECT active_round.round_id
    FROM public.lab_arena_rounds AS active_round
    WHERE active_round.status IN ('stage1_scoring', 'stage2_scoring')
    ORDER BY active_round.round_id
    FOR UPDATE
  LOOP
    NULL;
  END LOOP;
END;
$judgment_group_repair_round_locks$;

-- Completion does not take the round row because it already owns the run row.
-- Take the same per-group advisory locks after the stable round locks so a
-- concurrent terminal completion cannot select another follower.
DO $judgment_group_repair_advisory_locks$
DECLARE
  v_group RECORD;
BEGIN
  FOR v_group IN
    SELECT DISTINCT
      follower.round_id, follower.stage, follower.stage_generation,
      follower.judgment_cache_key
    FROM public.lab_arena_runs AS follower
    JOIN public.lab_arena_rounds AS active_round
      ON active_round.round_id = follower.round_id
     AND active_round.stage_generation = follower.stage_generation
     AND active_round.status =
         'stage' || follower.stage::TEXT || '_scoring'
    WHERE follower.kind = 'score'
      AND follower.status = 'pending'
      AND follower.judgment_cache_key IS NOT NULL
      AND NOT COALESCE(follower.judgment_group_leader, FALSE)
    ORDER BY follower.round_id, follower.stage,
             follower.stage_generation, follower.judgment_cache_key
  LOOP
    PERFORM pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'lab_arena.judgment-group:' || v_group.round_id || ':' ||
        v_group.stage::TEXT || ':' || v_group.stage_generation::TEXT || ':' ||
        v_group.judgment_cache_key,
        0
      )
    );
  END LOOP;
END;
$judgment_group_repair_advisory_locks$;

-- Repair an already orphaned current-generation group. Choose only one
-- follower per cache identity. Any cache row or accepted group result keeps
-- the group fail-closed for operator review instead of starting new work.
WITH orphaned_followers AS (
  SELECT
    follower.run_id,
    pg_catalog.row_number() OVER (
      PARTITION BY follower.round_id, follower.stage,
                   follower.stage_generation, follower.judgment_cache_key
      ORDER BY follower.run_id
    ) AS follower_order
  FROM public.lab_arena_runs AS follower
  JOIN public.lab_arena_rounds AS active_round
    ON active_round.round_id = follower.round_id
   AND active_round.stage_generation = follower.stage_generation
   AND active_round.status =
       'stage' || follower.stage::TEXT || '_scoring'
  WHERE follower.kind = 'score'
    AND follower.status = 'pending'
    AND follower.judgment_cache_key IS NOT NULL
    AND NOT COALESCE(follower.judgment_group_leader, FALSE)
    AND NOT EXISTS (
      SELECT 1
      FROM public.lab_arena_judgment_cache AS accepted_cache
      WHERE accepted_cache.cache_key = follower.judgment_cache_key
    )
    AND NOT EXISTS (
      SELECT 1
      FROM public.lab_arena_runs AS accepted_group_run
      WHERE accepted_group_run.round_id = follower.round_id
        AND accepted_group_run.stage = follower.stage
        AND accepted_group_run.stage_generation = follower.stage_generation
        AND accepted_group_run.kind = 'score'
        AND accepted_group_run.judgment_cache_key =
            follower.judgment_cache_key
        AND accepted_group_run.status = 'accepted'
    )
    AND NOT EXISTS (
      SELECT 1
      FROM public.lab_arena_runs AS active_leader
      WHERE active_leader.round_id = follower.round_id
        AND active_leader.stage = follower.stage
        AND active_leader.stage_generation = follower.stage_generation
        AND active_leader.kind = 'score'
        AND active_leader.judgment_cache_key = follower.judgment_cache_key
        AND COALESCE(active_leader.judgment_group_leader, FALSE)
        AND active_leader.status IN ('pending', 'leased', 'submitted')
    )
)
UPDATE public.lab_arena_runs AS follower
SET judgment_group_leader = TRUE
FROM orphaned_followers AS orphaned
WHERE follower.run_id = orphaned.run_id
  AND orphaned.follower_order = 1;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

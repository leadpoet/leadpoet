-- Retire the one-time live-round cost-policy adoption after every active live
-- round has a complete policy. Historical terminal configurations stay intact.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

LOCK TABLE public.lab_arena_rounds IN SHARE MODE;

DO $lab_arena_294_state_guard$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_rounds_write_once_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'lab_arena cost-policy functions are missing'
      USING ERRCODE = '55000';
  END IF;
  IF EXISTS (
    SELECT 1
    FROM public.lab_arena_rounds
    WHERE status NOT IN ('published', 'cancelled')
      AND configuration_doc ->> 'mode' = 'live'
      AND (
        NOT configuration_doc ? 'execution_cap_microusd'
        OR NOT configuration_doc ? 'cost_per_company_microusd'
      )
  ) THEN
    RAISE EXCEPTION 'active live round is missing its cost policy'
      USING ERRCODE = '55000';
  END IF;
END;
$lab_arena_294_state_guard$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $lab_arena_294_commit$
DECLARE
  v_definition TEXT;
  v_guard_anchor TEXT := '  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions';
  v_guarded_anchor TEXT := '  IF v_round.configuration_doc ->> ''mode'' = ''live''
     AND (
       NOT v_round.configuration_doc ? ''execution_cap_microusd''
       OR NOT v_round.configuration_doc ? ''cost_per_company_microusd''
     ) THEN
    RAISE EXCEPTION ''lab_arena_round_cost_policy_missing''
      USING ERRCODE = ''22023'';
  END IF;
  SELECT COUNT(*) INTO v_pending_admissions
  FROM public.lab_arena_submissions';
  v_legacy_assignment TEXT := 'configuration_doc = (
        CASE
          WHEN v_round.configuration_doc ->> ''mode'' = ''live''
               AND NOT v_round.configuration_doc ? ''cost_per_company_microusd''
          THEN v_round.configuration_doc || pg_catalog.jsonb_build_object(
            ''execution_cap_microusd'', 50000000,
            ''cost_per_company_microusd'', 500000
          )
          ELSE v_round.configuration_doc
        END
      ) || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
  v_current_assignment TEXT := 'configuration_doc = v_round.configuration_doc || pg_catalog.jsonb_build_object(
        ''scorer_image_digest'', p_scorer_image_digest,
        ''scorer_image_reference'', p_scorer_image_reference
      )';
  v_legacy_marker TEXT :=
    'AND NOT v_round.configuration_doc ? ''cost_per_company_microusd''';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'
    )
  ) INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_commit_round_v2_missing'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(v_definition, v_legacy_assignment) > 0 THEN
    v_definition := pg_catalog.replace(
      v_definition, v_legacy_assignment, v_current_assignment
    );
  ELSIF pg_catalog.strpos(v_definition, v_legacy_marker) > 0 THEN
    RAISE EXCEPTION 'lab_arena_commit_cost_policy_unknown'
      USING ERRCODE = '55000';
  ELSIF pg_catalog.strpos(v_definition, v_current_assignment) = 0 THEN
    RAISE EXCEPTION 'lab_arena_commit_cost_policy_unknown'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_round_cost_policy_missing'
     ) = 0 THEN
    IF pg_catalog.strpos(v_definition, v_guard_anchor) = 0 THEN
      RAISE EXCEPTION 'lab_arena_commit_cost_policy_guard_unknown'
        USING ERRCODE = '55000';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_guard_anchor, v_guarded_anchor
    );
  END IF;
  EXECUTE v_definition;

  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure(
      'public.lab_arena_commit_round_v2(text,jsonb,text,text,date,text,text)'
    )
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_legacy_marker) > 0
     OR pg_catalog.strpos(v_definition, v_current_assignment) = 0
     OR pg_catalog.strpos(
       v_definition, 'lab_arena_round_cost_policy_missing'
     ) = 0 THEN
    RAISE EXCEPTION 'lab_arena commit cost-policy retirement failed'
      USING ERRCODE = '55000';
  END IF;
END;
$lab_arena_294_commit$;

DO $lab_arena_294_write_guard$
DECLARE
  v_definition TEXT;
  v_legacy_guard TEXT := '(
        (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')
        OR (
          OLD.configuration_doc ->> ''mode'' = ''live''
          AND NOT OLD.configuration_doc ? ''cost_per_company_microusd''
          AND OLD.icp_set_date IS NULL
          AND NEW.icp_set_date IS NOT NULL
          AND (NEW.configuration_doc ->> ''execution_cap_microusd'')::BIGINT = 50000000
          AND (NEW.configuration_doc ->> ''cost_per_company_microusd'')::BIGINT = 500000
          AND (NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'' - ''cost_per_company_microusd'') =
              (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference''
               - ''execution_cap_microusd'')
        )
      )';
  v_current_guard TEXT := '(NEW.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'') =
          (OLD.configuration_doc - ''scorer_image_digest'' - ''scorer_image_reference'')';
  v_legacy_marker TEXT :=
    'AND NOT OLD.configuration_doc ? ''cost_per_company_microusd''';
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_rounds_write_once_v1()')
  ) INTO v_definition;
  IF v_definition IS NULL THEN
    RAISE EXCEPTION 'lab_arena_cost_policy_write_guard_missing'
      USING ERRCODE = '55000';
  END IF;
  IF pg_catalog.strpos(v_definition, v_legacy_guard) > 0 THEN
    v_definition := pg_catalog.replace(
      v_definition, v_legacy_guard, v_current_guard
    );
    EXECUTE v_definition;
  ELSIF pg_catalog.strpos(v_definition, v_legacy_marker) > 0 THEN
    RAISE EXCEPTION 'lab_arena_cost_policy_write_guard_unknown'
      USING ERRCODE = '55000';
  ELSIF pg_catalog.strpos(v_definition, v_current_guard) = 0 THEN
    RAISE EXCEPTION 'lab_arena_cost_policy_write_guard_unknown'
      USING ERRCODE = '55000';
  END IF;

  SELECT pg_catalog.pg_get_functiondef(
    pg_catalog.to_regprocedure('public.lab_arena_rounds_write_once_v1()')
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, v_legacy_marker) > 0
     OR pg_catalog.strpos(v_definition, v_current_guard) = 0 THEN
    RAISE EXCEPTION 'lab_arena cost-policy write guard retirement failed'
      USING ERRCODE = '55000';
  END IF;
END;
$lab_arena_294_write_guard$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

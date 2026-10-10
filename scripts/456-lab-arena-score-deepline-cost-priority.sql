-- Give score-call liabilities an indexed, rotating read-only candidate lane.
-- Clone the current V1 definition so every eligibility and billing guard stays.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $score_deepline_candidates$
DECLARE
  v_definition TEXT;
  v_existing TEXT;
  v_anchor TEXT := $anchor$            WHERE scope.round_id = p_round_id$anchor$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, 'lab_arena_deepline_candidate_query') = 0
     OR pg_catalog.strpos(v_definition, 'lab_arena__deepline_cost_binding_v1') = 0
     OR (pg_catalog.length(v_definition) - pg_catalog.length(
       pg_catalog.replace(v_definition, v_anchor, ''))) / pg_catalog.length(v_anchor) <> 1 THEN
    RAISE EXCEPTION 'score Deepline candidate preimage differs';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v1(',
    'FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(');
  -- Scope the indexed run scan before the uncertainty ordering. All later
  -- V1 reservation, dispatch, latest-head, identity, and round guards remain.
  v_definition := pg_catalog.replace(v_definition, v_anchor,
    v_anchor || E'\n              AND scope.kind = ''score''');
  IF pg_catalog.to_regprocedure(
      'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)'
    ) IS NOT NULL THEN
    SELECT pg_catalog.pg_get_functiondef(
      'public.lab_arena_list_deepline_cost_reconciliations_v3(text,text,bigint,integer)'::pg_catalog.regprocedure
    ) INTO v_existing;
    IF v_existing <> v_definition THEN
      RAISE EXCEPTION 'score Deepline candidate applied shape differs';
    END IF;
  ELSE
    EXECUTE v_definition;
  END IF;
END;
$score_deepline_candidates$;

ALTER FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(
  TEXT, TEXT, BIGINT, INTEGER
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(
  TEXT, TEXT, BIGINT, INTEGER
) FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_list_deepline_cost_reconciliations_v3(
  TEXT, TEXT, BIGINT, INTEGER
) TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

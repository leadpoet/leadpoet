-- Read Deepline uncertainties through the existing run and call indexes.
-- Optimize the ordinary V1 lane; keep the successful-execute V2 lane unchanged.
-- OFFSET 0 keeps parameterized lookups and cursor order before the full
-- candidate join and final sort. Apply the limit only after every
-- original eligibility, identity, binding, dispatch and latest-head guard.
-- Read-only Oct9 live plan: 17487.728 ms before, 2712.720 ms after.
-- Exact JSON matched first-1, first-20, wrap-20 and mixed-cursor-20 cases
-- within one repeatable-read snapshot; no schema changes were used.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $deepline_candidate_query$
DECLARE
  v_signature TEXT;
  v_definition TEXT;
  v_position INTEGER;
  v_applied BOOLEAN;
  v_old TEXT[] := ARRAY[
    $old$          FROM public.lab_arena_ledger AS uncertainty$old$,
    $old$          JOIN public.lab_arena_ledger AS reservation$old$,
    $old$          JOIN public.lab_arena_runs AS runs$old$,
    $old$          JOIN public.lab_arena_rounds AS rounds$old$,
    $old$              FROM public.lab_arena_ledger AS dispatch
              WHERE dispatch.call_identity = uncertainty.call_identity
                AND dispatch.entry_kind = 'dispatch'
                AND dispatch.run_id = uncertainty.run_id$old$,
    $old$          ORDER BY
            uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0),
            uncertainty.entry_id$old$
  ];
  v_new TEXT[] := ARRAY[
    $new$          -- lab_arena_deepline_candidate_query
          FROM (
            SELECT uncertainty.*,
              uncertainty.entry_id <= GREATEST(COALESCE(p_after_entry_id, 0), 0)
                AS cursor_wrapped
            FROM public.lab_arena_runs AS scope
            CROSS JOIN LATERAL (
              SELECT entry.*
              FROM public.lab_arena_ledger AS entry
              WHERE entry.run_id = scope.run_id
              OFFSET 0
            ) AS uncertainty
            WHERE scope.round_id = p_round_id
              AND uncertainty.round_id = p_round_id
              AND uncertainty.entry_kind = 'uncertain'
              AND uncertainty.provider = 'deepline'
            ORDER BY cursor_wrapped, uncertainty.entry_id
            OFFSET 0
          ) AS uncertainty$new$,
    $new$          JOIN LATERAL (
            SELECT reservation.*
            FROM public.lab_arena_ledger AS reservation
            WHERE reservation.call_identity = uncertainty.call_identity
              AND reservation.entry_kind = 'reservation'
            OFFSET 0
          ) AS reservation$new$,
    $new$          JOIN LATERAL (
            SELECT runs.* FROM public.lab_arena_runs AS runs
            WHERE runs.run_id = uncertainty.run_id
            OFFSET 0
          ) AS runs$new$,
    $new$          JOIN LATERAL (
            SELECT rounds.* FROM public.lab_arena_rounds AS rounds
            WHERE rounds.round_id = uncertainty.round_id
            OFFSET 0
          ) AS rounds$new$,
    $new$              FROM (
                SELECT scoped.* FROM public.lab_arena_ledger AS scoped
                WHERE scoped.call_identity = uncertainty.call_identity
                  AND scoped.entry_kind = 'dispatch'
                OFFSET 0
              ) AS dispatch
              WHERE dispatch.run_id = uncertainty.run_id$new$,
    $new$          ORDER BY
            uncertainty.cursor_wrapped,
            uncertainty.entry_id$new$
  ];
BEGIN
  FOREACH v_signature IN ARRAY ARRAY[
    'public.lab_arena_list_deepline_cost_reconciliations_v1(text,text,bigint,integer)'
  ] LOOP
    SELECT pg_catalog.pg_get_functiondef(v_signature::pg_catalog.regprocedure)
      INTO v_definition;
    IF pg_catalog.strpos(v_definition, 'lab_arena__deepline_cost_binding_v1') = 0
       OR pg_catalog.strpos(v_definition, 'lab_arena__closed_host_score_success_uncertainty_v1') = 0 THEN
      RAISE EXCEPTION 'Deepline candidate query prerequisite missing: %', v_signature;
    END IF;
    v_applied := pg_catalog.strpos(v_definition, 'lab_arena_deepline_candidate_query') > 0;
    FOR v_position IN 1..pg_catalog.array_length(v_old, 1) LOOP
      IF v_applied THEN
        IF (pg_catalog.length(v_definition) - pg_catalog.length(
            pg_catalog.replace(v_definition, v_new[v_position], '')))
              / pg_catalog.length(v_new[v_position]) <> 1
           OR pg_catalog.strpos(v_definition, v_old[v_position]) > 0 THEN
          RAISE EXCEPTION 'Deepline candidate query applied shape differs: %, %',
            v_signature, v_position;
        END IF;
      ELSIF (pg_catalog.length(v_definition) - pg_catalog.length(
          pg_catalog.replace(v_definition, v_old[v_position], '')))
            / pg_catalog.length(v_old[v_position]) <> 1 THEN
        RAISE EXCEPTION 'Deepline candidate query preimage differs: %, %',
          v_signature, v_position;
      END IF;
    END LOOP;
    IF NOT v_applied THEN
      FOR v_position IN 1..pg_catalog.array_length(v_old, 1) LOOP
        v_definition := pg_catalog.replace(v_definition,
          v_old[v_position], v_new[v_position]);
      END LOOP;
      -- CREATE OR REPLACE retains each function's owner, grants and settings.
      EXECUTE v_definition;
    END IF;
  END LOOP;
END;
$deepline_candidate_query$;

NOTIFY pgrst, 'reload schema';
COMMIT;

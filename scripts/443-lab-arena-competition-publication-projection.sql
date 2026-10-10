-- Bound competition history reads without changing durable publications.
-- Results and miner history still select the full publication_doc column.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE lab_arena_443_schema_acl ON COMMIT DROP AS
SELECT namespace.nspacl AS acl,
       pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE')
         AS had_create
FROM pg_catalog.pg_namespace AS namespace
WHERE namespace.nspname = 'public';
DO $temporary_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_443_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_create$;

CREATE OR REPLACE FUNCTION public.lab_arena_competition_publication_v1(
  p_round public.lab_arena_rounds
)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $competition_publication$
DECLARE
  v_doc JSONB := p_round.publication_doc;
  v_participant JSONB;
  v_ranking JSONB;
  v_cost JSONB;
  v_item JSONB;
  v_bucket JSONB;
  v_decision JSONB;
  v_champion TEXT;
  v_ids TEXT[] := ARRAY[]::TEXT[];
  v_final JSONB := '[]'::JSONB;
BEGIN
  -- Legacy coercions and NULL/missing values belong to the Python projection.
  -- Only compact canonical published shapes; otherwise preserve the whole doc.
  IF p_round.status IS DISTINCT FROM 'published'
     OR pg_catalog.jsonb_typeof(v_doc) IS DISTINCT FROM 'object'
     OR pg_catalog.jsonb_typeof(v_doc -> 'participants') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(v_doc -> 'final_ranking') IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(v_doc -> 'king_decision') IS DISTINCT FROM 'object' THEN
    RETURN v_doc;
  END IF;
  v_decision := v_doc -> 'king_decision';
  IF pg_catalog.jsonb_typeof(v_decision -> 'outcome') IS DISTINCT FROM 'string'
     OR v_decision ->> 'outcome' NOT IN ('crowned', 'defended', 'retained_ineligible', 'no_king') THEN
    RETURN v_doc;
  END IF;
  -- Validate even unused decision IDs rather than emulate Python truthiness.
  FOREACH v_champion IN ARRAY ARRAY['winner_submission_id', 'king_submission_id'] LOOP
    IF v_decision ? v_champion AND v_decision -> v_champion <> 'null'::JSONB
       AND (pg_catalog.jsonb_typeof(v_decision -> v_champion) IS DISTINCT FROM 'string'
            OR v_decision ->> v_champion = '') THEN
      RETURN v_doc;
    END IF;
  END LOOP;
  v_champion := CASE v_decision ->> 'outcome'
    WHEN 'crowned' THEN v_decision ->> 'winner_submission_id'
    WHEN 'defended' THEN v_decision ->> 'king_submission_id'
    WHEN 'retained_ineligible' THEN v_decision ->> 'king_submission_id'
    ELSE NULL END;
  IF v_decision ->> 'outcome' <> 'no_king' AND v_champion IS NULL THEN
    RETURN v_doc;
  END IF;
  IF v_champion IS NOT NULL THEN
    v_ids := pg_catalog.array_append(v_ids, v_champion);
  END IF;
  FOR v_participant IN SELECT value FROM pg_catalog.jsonb_array_elements(v_doc -> 'participants') LOOP
    IF pg_catalog.jsonb_typeof(v_participant) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(v_participant -> 'submission_id') IS DISTINCT FROM 'string'
       OR v_participant ->> 'submission_id' = ''
       OR (v_participant ? 'is_baseline' AND pg_catalog.jsonb_typeof(v_participant -> 'is_baseline') IS DISTINCT FROM 'boolean')
       OR (v_participant ? 'is_king' AND pg_catalog.jsonb_typeof(v_participant -> 'is_king') IS DISTINCT FROM 'boolean') THEN
      RETURN v_doc;
    END IF;
    -- Presence of is_baseline overrides is_king, including explicit false.
    IF COALESCE(v_participant -> 'is_baseline', v_participant -> 'is_king', 'false'::JSONB) = 'true'::JSONB THEN
      v_ids := pg_catalog.array_append(v_ids, v_participant ->> 'submission_id');
    END IF;
  END LOOP;
  FOR v_ranking IN SELECT value FROM pg_catalog.jsonb_array_elements(v_doc -> 'final_ranking') LOOP
    IF pg_catalog.jsonb_typeof(v_ranking) IS DISTINCT FROM 'object'
       OR pg_catalog.jsonb_typeof(v_ranking -> 'submission_id') IS DISTINCT FROM 'string'
       OR v_ranking ->> 'submission_id' = ''
       OR pg_catalog.jsonb_typeof(v_ranking -> 'eligible') IS DISTINCT FROM 'boolean'
       OR pg_catalog.jsonb_typeof(v_ranking -> 'eligibility_reason') IS DISTINCT FROM 'string' THEN
      RETURN v_doc;
    END IF;
    v_cost := v_ranking -> 'cost_summary';
    IF v_cost IS NOT NULL AND v_cost <> 'null'::JSONB THEN
      IF pg_catalog.jsonb_typeof(v_cost) IS DISTINCT FROM 'object' THEN
        RETURN v_doc;
      END IF;
      -- Python checks costs for every participant, even an unused ranking.
      -- Keep malformed containers/reasons intact, including their failure path.
      IF v_cost ? 'per_icp' THEN
        IF pg_catalog.jsonb_typeof(v_cost -> 'per_icp') IS DISTINCT FROM 'array' THEN
          RETURN v_doc;
        END IF;
        FOR v_item IN SELECT value FROM pg_catalog.jsonb_array_elements(v_cost -> 'per_icp') LOOP
          IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
             OR pg_catalog.jsonb_typeof(v_item -> 'eligibility_reason') IS DISTINCT FROM 'string' THEN
            RETURN v_doc;
          END IF;
        END LOOP;
      END IF;
      FOREACH v_champion IN ARRAY ARRAY['execution', 'judge'] LOOP
        IF v_cost ? v_champion THEN
          v_bucket := v_cost -> v_champion;
          IF pg_catalog.jsonb_typeof(v_bucket) IS DISTINCT FROM 'object'
             OR pg_catalog.jsonb_typeof(v_bucket -> 'providers') IS DISTINCT FROM 'array' THEN
            RETURN v_doc;
          END IF;
          FOR v_item IN SELECT value FROM pg_catalog.jsonb_array_elements(v_bucket -> 'providers') LOOP
            IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
               OR pg_catalog.jsonb_typeof(v_item -> 'provider') IS DISTINCT FROM 'string' THEN
              RETURN v_doc;
            END IF;
          END LOOP;
        END IF;
      END LOOP;
    END IF;
    IF v_ranking ->> 'submission_id' = ANY(v_ids) THEN
      -- Append all matching entries, in order: Python's last duplicate wins.
      v_final := v_final || pg_catalog.jsonb_build_array(v_ranking);
    END IF;
  END LOOP;
  RETURN pg_catalog.jsonb_build_object(
    'participants', v_doc -> 'participants',
    'king_decision', v_decision,
    'final_ranking', v_final
  );
END;
$competition_publication$;

ALTER FUNCTION public.lab_arena_competition_publication_v1(public.lab_arena_rounds)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_competition_publication_v1(public.lab_arena_rounds)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_competition_publication_v1(public.lab_arena_rounds)
  TO lab_arena_service;

DO $restore_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_443_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
       IS DISTINCT FROM (SELECT acl FROM pg_temp.lab_arena_443_schema_acl) THEN
    RAISE EXCEPTION 'lab_arena_competition_publication_schema_acl_changed';
  END IF;
END;
$restore_create$;
NOTIFY pgrst, 'reload schema';
COMMIT;

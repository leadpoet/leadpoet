-- 184-lab-arena-scoring-failure-isolation.sql
-- Keep one challenger's judge failure from cancelling the daily competition.

BEGIN;

DO $lab_arena_184_requires_183$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_trigger
    WHERE tgname = 'lab_arena_publication_baseline_guard'
      AND NOT tgisinternal
  ) THEN
    RAISE EXCEPTION 'apply 183-lab-arena-miner-reward-basis.sql first';
  END IF;
END;
$lab_arena_184_requires_183$;

-- Required only for ownership transfers by the hosted migration role.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_transition_round(
  p_round_id TEXT,
  p_expected_status TEXT,
  p_next_status TEXT,
  p_patch JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_transition_round$
DECLARE
  v_round public.lab_arena_rounds;
  v_patch JSONB := COALESCE(p_patch, '{}'::JSONB);
  v_keys TEXT[];
  v_allowed TEXT[];
  v_baseline_count INTEGER;
  v_challenger_count INTEGER;
  v_missing_scores INTEGER;
  v_invalid_finalists INTEGER;
  v_publication JSONB;
  v_decision JSONB;
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(key ORDER BY key), ARRAY[]::TEXT[]) INTO v_keys
  FROM pg_catalog.jsonb_object_keys(v_patch) AS key;

  IF p_expected_status = 'open' AND p_next_status = 'committed' THEN
    v_allowed := ARRAY['participants', 'benchmark_ref', 'evaluation_date'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys) THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF pg_catalog.jsonb_typeof(v_patch -> 'participants') <> 'array'
       OR pg_catalog.char_length(COALESCE(v_patch ->> 'benchmark_ref', '')) NOT BETWEEN 1 AND 1024
       OR COALESCE(v_patch ->> 'evaluation_date', '') !~ '^[0-9]{4}-[0-9]{2}-[0-9]{2}$' THEN
      RAISE EXCEPTION 'lab_arena_round_commit_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'committed', status_generation = status_generation + 1,
        participants = v_patch -> 'participants',
        benchmark_ref = v_patch ->> 'benchmark_ref',
        evaluation_date = v_patch ->> 'evaluation_date'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_closed' AND p_next_status = 'stage1_closed' THEN
    v_allowed := ARRAY['stage1_scoring_plan_doc'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'stage1_scoring_plan_doc') IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage1_scoring_plan_doc IS NOT NULL THEN
      IF v_round.stage1_scoring_plan_doc = (v_patch -> 'stage1_scoring_plan_doc') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage1_scoring_plan_doc = v_patch -> 'stage1_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_closed' AND p_next_status = 'stage2_closed' THEN
    v_allowed := ARRAY['stage2_scoring_plan_doc'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'stage2_scoring_plan_doc') IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    IF v_round.stage2_scoring_plan_doc IS NOT NULL THEN
      IF v_round.stage2_scoring_plan_doc = (v_patch -> 'stage2_scoring_plan_doc') THEN
        RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
      END IF;
      RAISE EXCEPTION 'lab_arena_scoring_plan_write_once' USING ERRCODE = '42501';
    END IF;
    UPDATE public.lab_arena_rounds
    SET stage2_scoring_plan_doc = v_patch -> 'stage2_scoring_plan_doc'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage1_judged' AND p_next_status = 'stage1_scored' THEN
    v_allowed := ARRAY['finalists'];
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'finalists') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') > 10
       OR v_round.stage1_scoring_plan_doc IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT COUNT(*) INTO v_baseline_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
    SELECT COUNT(*) INTO v_missing_scores
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE (
      SELECT COUNT(DISTINCT runs.icp_position)
      FROM public.lab_arena_runs AS runs
      WHERE runs.round_id = p_round_id
        AND runs.stage = 1
        AND runs.kind = 'execute'
        AND runs.submission_id = participant ->> 'submission_id'
        AND runs.per_icp_score IS NOT NULL
    ) <> 10
      AND (
        v_baseline_count <> 1
        OR COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
      );
    IF v_missing_scores <> 0 THEN
      RAISE EXCEPTION 'lab_arena_stage1_baseline_scores_incomplete' USING ERRCODE = '22023';
    END IF;
    SELECT COUNT(*) INTO v_challenger_count
    FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
    WHERE NOT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
      AND (
        v_baseline_count <> 1
        OR (
          SELECT COUNT(DISTINCT runs.icp_position)
          FROM public.lab_arena_runs AS runs
          WHERE runs.round_id = p_round_id
            AND runs.stage = 1
            AND runs.kind = 'execute'
            AND runs.submission_id = participant ->> 'submission_id'
            AND runs.per_icp_score IS NOT NULL
        ) = 10
      );
    SELECT COUNT(*) INTO v_invalid_finalists
    FROM pg_catalog.jsonb_array_elements(v_patch -> 'finalists') AS finalist
    WHERE pg_catalog.jsonb_typeof(finalist) <> 'string'
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(v_round.participants) AS participant
         WHERE participant ->> 'submission_id' = finalist #>> '{}'
           AND NOT COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
           AND (
             v_baseline_count <> 1
             OR (
               SELECT COUNT(DISTINCT runs.icp_position)
               FROM public.lab_arena_runs AS runs
               WHERE runs.round_id = p_round_id
                 AND runs.stage = 1
                 AND runs.kind = 'execute'
                 AND runs.submission_id = participant ->> 'submission_id'
                 AND runs.per_icp_score IS NOT NULL
             ) = 10
           )
       );
    IF v_invalid_finalists <> 0
       OR pg_catalog.jsonb_array_length(v_patch -> 'finalists') <> LEAST(10, v_challenger_count)
       OR (
         SELECT COUNT(DISTINCT finalist #>> '{}')
         FROM pg_catalog.jsonb_array_elements(v_patch -> 'finalists') AS finalist
       ) <> pg_catalog.jsonb_array_length(v_patch -> 'finalists') THEN
      RAISE EXCEPTION 'lab_arena_finalists_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'stage1_scored', status_generation = status_generation + 1,
        finalists = v_patch -> 'finalists'
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'stage2_judged' AND p_next_status = 'scored' THEN
    v_allowed := ARRAY[]::TEXT[];
    IF v_keys <> ARRAY[]::TEXT[]
       OR v_round.stage2_scoring_plan_doc IS NULL THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'scored', status_generation = status_generation + 1
    WHERE round_id = p_round_id;
  ELSIF p_expected_status = 'scored' AND p_next_status = 'published' THEN
    v_allowed := ARRAY['publication_doc', 'published_at'];
    v_publication := v_patch -> 'publication_doc';
    v_decision := v_publication -> 'king_decision';
    IF NOT (v_keys @> v_allowed AND v_allowed @> v_keys)
       OR pg_catalog.jsonb_typeof(v_patch -> 'published_at') <> 'string'
       OR COALESCE(v_patch ->> 'published_at', '') = ''
       OR pg_catalog.jsonb_typeof(v_publication) IS DISTINCT FROM 'object'
       OR (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(v_publication)) <> 8
       OR v_publication ->> 'schema_version' <> 'leadpoet.lab_arena.publication.v1'
       OR v_publication ->> 'round_id' <> p_round_id
       OR v_publication ->> 'published_at' <> v_patch ->> 'published_at'
       OR pg_catalog.jsonb_typeof(v_publication -> 'participants') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_publication -> 'stage1_ranking') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_publication -> 'finalists') IS DISTINCT FROM 'array'
       OR v_publication -> 'finalists' IS DISTINCT FROM v_round.finalists
       OR pg_catalog.jsonb_typeof(v_publication -> 'final_ranking') IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_typeof(v_decision) IS DISTINCT FROM 'object'
       OR v_decision ->> 'outcome' NOT IN ('crowned', 'defended', 'retained_ineligible', 'no_king')
       OR (v_decision ->> 'outcome' = 'no_king' AND COALESCE(v_decision ->> 'king_hotkey', '') <> '')
       OR (v_decision ->> 'outcome' <> 'no_king' AND COALESCE(v_decision ->> 'king_hotkey', '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$') THEN
      RAISE EXCEPTION 'lab_arena_publication_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_rounds
    SET status = 'published', status_generation = status_generation + 1,
        publication_doc = v_patch -> 'publication_doc',
        king_outcome = v_decision ->> 'outcome',
        king_hotkey = NULLIF(v_decision ->> 'king_hotkey', ''),
        published_at = (v_patch ->> 'published_at')::TIMESTAMPTZ
    WHERE round_id = p_round_id;
  ELSE
    RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_round.status, 'status_generation', v_round.status_generation);
END;
$lab_arena_transition_round$;
ALTER FUNCTION public.lab_arena_transition_round(TEXT, TEXT, TEXT, JSONB) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_close_scoring(p_round_id TEXT, p_stage SMALLINT)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_close_scoring$
DECLARE
  v_round public.lab_arena_rounds;
  v_run public.lab_arena_runs;
  v_generation BIGINT;
  v_baseline_count INTEGER;
  v_baseline_incomplete INTEGER;
  v_incomplete INTEGER;
  v_next TEXT;
BEGIN
  IF p_stage NOT IN (1, 2) THEN
    RAISE EXCEPTION 'lab_arena_stage_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> ('stage' || p_stage::TEXT || '_scoring') THEN
    IF (p_stage = 1 AND v_round.status IN ('stage1_judged', 'stage1_scored',
                                           'stage2', 'stage2_closed', 'stage2_scoring', 'stage2_judged', 'scored', 'published'))
       OR (p_stage = 2 AND v_round.status IN ('stage2_judged', 'scored', 'published')) THEN
      RETURN pg_catalog.jsonb_build_object('status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation);
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation);
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_run IN
    SELECT * FROM public.lab_arena_runs
    WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score' AND status IN ('leased', 'pending', 'submitted')
    ORDER BY assignment_id, attempt
    FOR UPDATE
  LOOP
    IF v_run.status = 'leased' THEN
      PERFORM public.lab_arena__terminate_open_calls(v_run.run_id, 'stage_closed');
    END IF;
    UPDATE public.lab_arena_runs
    SET status = 'failed', terminal_cause = 'stage_closed',
        terminal_doc = pg_catalog.jsonb_build_object('closed_at', pg_catalog.clock_timestamp(),
          'previous_status', v_run.status)
    WHERE run_id = v_run.run_id;
  END LOOP;
  SELECT COUNT(*) INTO v_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id) runs.assignment_id, runs.status, runs.terminal_cause
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
  WHERE latest.status <> 'accepted';
  SELECT COUNT(*) INTO v_baseline_count
  FROM pg_catalog.jsonb_array_elements(COALESCE(v_round.participants, '[]'::JSONB)) AS participant
  WHERE COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE);
  SELECT COUNT(*) INTO v_baseline_incomplete FROM (
    SELECT DISTINCT ON (runs.assignment_id)
      runs.assignment_id, runs.submission_id, runs.status
    FROM public.lab_arena_runs AS runs
    WHERE runs.round_id = p_round_id AND runs.stage = p_stage AND runs.kind = 'score'
    ORDER BY runs.assignment_id, (runs.status = 'accepted') DESC, runs.attempt DESC
  ) AS latest
  WHERE latest.status <> 'accepted'
    AND EXISTS (
      SELECT 1
      FROM pg_catalog.jsonb_array_elements(COALESCE(v_round.participants, '[]'::JSONB)) AS participant
      WHERE participant ->> 'submission_id' = latest.submission_id
        AND COALESCE((participant ->> 'is_king')::BOOLEAN, FALSE)
    );
  IF v_incomplete > 0 AND (v_baseline_count <> 1 OR v_baseline_incomplete > 0) THEN
    v_next := 'cancelled';
    UPDATE public.lab_arena_rounds
    SET status = 'cancelled', status_generation = status_generation + 1, stage_generation = v_generation,
        cancel_reason = 'capacity:scoring' || p_stage::TEXT || ':' || v_incomplete::TEXT
    WHERE round_id = p_round_id;
  ELSE
    v_next := 'stage' || p_stage::TEXT || '_judged';
    UPDATE public.lab_arena_rounds
    SET status = v_next, status_generation = status_generation + 1, stage_generation = v_generation
    WHERE round_id = p_round_id;
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', CASE WHEN v_next = 'cancelled' THEN 'cancelled' ELSE 'closed' END,
    'round_status', v_next, 'incomplete_assignments', v_incomplete, 'stage_generation', v_generation);
END;
$lab_arena_close_scoring$;
ALTER FUNCTION public.lab_arena_close_scoring(TEXT, SMALLINT) OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_schema_version_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_schema_version$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.schema_version.v1',
    'version', 184
  );
$lab_arena_schema_version$;
ALTER FUNCTION public.lab_arena_schema_version_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_schema_version_v1() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_schema_version_v1() TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

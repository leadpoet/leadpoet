-- Reuse one accepted Arena judgment for byte-equivalent effective scoring
-- inputs.  Cache identities are gateway-computed and attached before any
-- validator sees a lease; failures can never populate the accepted cache.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS judgment_cache_key TEXT,
  ADD COLUMN IF NOT EXISTS judgment_input_hash TEXT,
  ADD COLUMN IF NOT EXISTS judgment_scope_doc JSONB,
  ADD COLUMN IF NOT EXISTS judgment_group_leader BOOLEAN,
  ADD COLUMN IF NOT EXISTS judgment_group_miner_hotkeys TEXT[],
  ADD COLUMN IF NOT EXISTS judgment_cache_source_run_id TEXT;

ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_judgment_cache_key_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_judgment_cache_key_check CHECK (
    judgment_cache_key IS NULL
    OR judgment_cache_key ~ '^sha256:[0-9a-f]{64}$'
  );
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_judgment_input_hash_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_judgment_input_hash_check CHECK (
    judgment_input_hash IS NULL
    OR judgment_input_hash ~ '^sha256:[0-9a-f]{64}$'
  );

CREATE TABLE IF NOT EXISTS public.lab_arena_judgment_cache (
  cache_key TEXT PRIMARY KEY CHECK (cache_key ~ '^sha256:[0-9a-f]{64}$'),
  scope_doc JSONB NOT NULL,
  scoring_input_hash TEXT NOT NULL CHECK (
    scoring_input_hash ~ '^sha256:[0-9a-f]{64}$'
  ),
  evidence_hash TEXT NOT NULL CHECK (evidence_hash ~ '^sha256:[0-9a-f]{64}$'),
  evidence_doc JSONB NOT NULL,
  source_score_run_id TEXT NOT NULL UNIQUE
    REFERENCES public.lab_arena_runs(run_id),
  source_scored_run_id TEXT NOT NULL
    REFERENCES public.lab_arena_runs(run_id),
  source_runner_hotkey TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  CHECK (pg_catalog.jsonb_typeof(scope_doc) = 'object'),
  CHECK (pg_catalog.jsonb_typeof(evidence_doc) = 'object'),
  CHECK (scope_doc ->> 'cache_key' = cache_key),
  CHECK (scope_doc ->> 'scoring_input_hash' = scoring_input_hash),
  CHECK (evidence_doc ->> 'cache_key' = cache_key),
  CHECK (evidence_doc ->> 'scoring_input_hash' = scoring_input_hash),
  CHECK (evidence_doc ->> 'source_score_run_id' = source_score_run_id),
  CHECK (evidence_doc ->> 'source_scored_run_id' = source_scored_run_id),
  CHECK (evidence_doc ->> 'source_runner_hotkey' = source_runner_hotkey)
);
ALTER TABLE public.lab_arena_judgment_cache OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_judgment_cache
  DROP CONSTRAINT IF EXISTS lab_arena_judgment_cache_authority_snapshot_check;
ALTER TABLE public.lab_arena_judgment_cache
  ADD CONSTRAINT lab_arena_judgment_cache_authority_snapshot_check CHECK (
    pg_catalog.jsonb_typeof(
      evidence_doc -> 'runner_authority_exclusions'
    ) IS NOT DISTINCT FROM 'array'
    AND evidence_doc -> 'runner_authority_exclusions'
      @> pg_catalog.jsonb_build_array(source_runner_hotkey)
  );

DROP TRIGGER IF EXISTS lab_arena_judgment_cache_append_only
  ON public.lab_arena_judgment_cache;
CREATE TRIGGER lab_arena_judgment_cache_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_judgment_cache
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

ALTER TABLE public.lab_arena_judgment_cache ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS lab_arena_judgment_cache_service_read
  ON public.lab_arena_judgment_cache;
CREATE POLICY lab_arena_judgment_cache_service_read
  ON public.lab_arena_judgment_cache FOR SELECT TO lab_arena_service
  USING (TRUE);
REVOKE ALL ON TABLE public.lab_arena_judgment_cache FROM PUBLIC;
DO $lab_arena_212_cache_table_acl$
DECLARE role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON TABLE public.lab_arena_judgment_cache FROM %I', role_name
      );
    END IF;
  END LOOP;
END;
$lab_arena_212_cache_table_acl$;
GRANT SELECT ON TABLE public.lab_arena_judgment_cache TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v2(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_work_items JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_scoring_v2$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_item JSONB;
  v_scored public.lab_arena_runs;
  v_cache public.lab_arena_judgment_cache;
  v_assignment TEXT;
  v_created INTEGER := 0;
  v_reused INTEGER := 0;
  v_status TEXT;
BEGIN
  IF p_stage IS NULL OR p_stage < 1
     OR pg_catalog.jsonb_typeof(p_work_items) IS DISTINCT FROM 'array' THEN
    RAISE EXCEPTION 'lab_arena_scoring_input_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.configuration_doc ->> 'integrity_policy'
       IS DISTINCT FROM 'arena_integrity_v1' THEN
    RAISE EXCEPTION 'lab_arena_integrity_policy_required' USING ERRCODE = '22023';
  END IF;
  v_expected := 'stage' || p_stage::TEXT || '_closed';
  v_next := 'stage' || p_stage::TEXT || '_scoring';
  IF v_round.status <> v_expected THEN
    IF EXISTS (
      SELECT 1 FROM public.lab_arena_runs
      WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score'
        AND judgment_cache_key IS NOT NULL
    ) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'round_status', v_round.status,
        'stage_generation', v_round.stage_generation
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'round_status', v_round.status,
      'stage_generation', v_round.stage_generation
    );
  END IF;
  v_generation := v_round.stage_generation + 1;
  FOR v_item IN
    SELECT value FROM pg_catalog.jsonb_array_elements(p_work_items)
  LOOP
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'scored_run_id', '') = ''
       OR COALESCE(v_item ->> 'submission_id', '') !~ '^[A-Za-z0-9._:-]{1,64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'icp_position') IS DISTINCT FROM 'number'
       OR (v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29
       OR pg_catalog.char_length(COALESCE(v_item ->> 'output_ref', '')) NOT BETWEEN 1 AND 1024
       OR COALESCE(v_item ->> 'judgment_cache_key', '') !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'judgment_input_hash', '') !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'judgment_scope_doc') IS DISTINCT FROM 'object'
       OR v_item #>> '{judgment_scope_doc,cache_key}' IS DISTINCT FROM
          v_item ->> 'judgment_cache_key'
       OR v_item #>> '{judgment_scope_doc,scoring_input_hash}' IS DISTINCT FROM
          v_item ->> 'judgment_input_hash'
       OR v_item #>> '{judgment_scope_doc,round_id}' IS DISTINCT FROM p_round_id
       OR v_item #>> '{judgment_scope_doc,network_name}' IS DISTINCT FROM
          v_round.arena_network_name
       OR (v_item #>> '{judgment_scope_doc,netuid}')::INTEGER IS DISTINCT FROM
          v_round.arena_netuid
       OR v_item #>> '{judgment_scope_doc,integrity_policy}' IS DISTINCT FROM
          'arena_integrity_v1'
       OR v_item #>> '{judgment_scope_doc,evaluation_date}' IS DISTINCT FROM
          v_round.evaluation_date
       OR v_item #>> '{judgment_scope_doc,scorer_image_digest}' IS DISTINCT FROM
          v_round.configuration_doc ->> 'scorer_image_digest'
       OR v_item #>> '{judgment_scope_doc,scorer_image_reference}' IS DISTINCT FROM
          v_round.configuration_doc ->> 'scorer_image_reference'
       OR (
         v_item ? 'reuse_cache_key'
         AND v_item ->> 'reuse_cache_key' IS DISTINCT FROM
             v_item ->> 'judgment_cache_key'
       )
       OR (
         NOT (v_item ? 'reuse_cache_key')
         AND (
           pg_catalog.jsonb_typeof(v_item -> 'judgment_group_leader')
             IS DISTINCT FROM 'boolean'
           OR pg_catalog.jsonb_typeof(v_item -> 'judgment_group_miner_hotkeys')
             IS DISTINCT FROM 'array'
         )
       ) THEN
      RAISE EXCEPTION 'lab_arena_scoring_item_invalid' USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_scored FROM public.lab_arena_runs
    WHERE run_id = v_item ->> 'scored_run_id'
      AND round_id = p_round_id
      AND stage = p_stage
      AND submission_id = v_item ->> 'submission_id'
      AND icp_position = (v_item ->> 'icp_position')::INTEGER
      AND output_ref = v_item ->> 'output_ref'
      AND kind = 'execute' AND status = 'accepted';
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_scored_run_invalid' USING ERRCODE = '22023';
    END IF;
    v_cache := NULL;
    IF v_item ? 'reuse_cache_key' THEN
      SELECT * INTO v_cache FROM public.lab_arena_judgment_cache
      WHERE cache_key = v_item ->> 'reuse_cache_key';
      IF NOT FOUND OR v_cache.scoring_input_hash IS DISTINCT FROM
          v_item ->> 'judgment_input_hash' THEN
        RAISE EXCEPTION 'lab_arena_judgment_cache_invalid' USING ERRCODE = '22023';
      END IF;
      IF NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs source
        WHERE source.run_id = v_cache.source_score_run_id
          AND source.kind = 'score' AND source.status = 'accepted'
          AND source.runner_hotkey = v_cache.source_runner_hotkey
          AND source.scored_run_id = v_cache.source_scored_run_id
      ) THEN
        RAISE EXCEPTION 'lab_arena_judgment_cache_source_invalid'
          USING ERRCODE = '22023';
      END IF;
    END IF;
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';
    v_status := CASE WHEN v_cache.cache_key IS NULL THEN 'pending' ELSE 'accepted' END;
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      result_doc, terminal_cause, output_ref, judgment_cache_key,
      judgment_input_hash, judgment_scope_doc, judgment_group_leader,
      judgment_group_miner_hotkeys, judgment_cache_source_run_id
    ) VALUES (
      v_assignment || ':1', v_assignment, p_round_id, v_scored.submission_id,
      v_scored.miner_hotkey, p_stage, v_scored.icp_position, 1, v_status,
      v_generation, 'score', v_scored.run_id,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE
        pg_catalog.jsonb_build_object(
          'schema_version', 'leadpoet.lab_arena.cached_run_result.v1',
          'terminal_status', 'accepted',
          'cache_key', v_cache.cache_key,
          'source_score_run_id', v_cache.source_score_run_id
        ) END,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE 'accepted' END,
      CASE WHEN v_cache.cache_key IS NULL THEN NULL ELSE
        v_cache.evidence_doc ->> 'source_output_ref' END,
      v_item ->> 'judgment_cache_key', v_item ->> 'judgment_input_hash',
      v_item -> 'judgment_scope_doc',
      CASE WHEN v_cache.cache_key IS NULL THEN
        (v_item ->> 'judgment_group_leader')::BOOLEAN ELSE FALSE END,
      CASE WHEN v_cache.cache_key IS NULL THEN ARRAY(
        SELECT pg_catalog.jsonb_array_elements_text(
          v_item -> 'judgment_group_miner_hotkeys'
        )
      ) ELSE ARRAY[]::TEXT[] END,
      v_cache.source_score_run_id
    );
    v_created := v_created + 1;
    IF v_cache.cache_key IS NOT NULL THEN
      v_reused := v_reused + 1;
    END IF;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_next,
    'stage_generation', v_generation, 'assignments', v_created,
    'reused', v_reused
  );
END;
$lab_arena_open_scoring_v2$;
ALTER FUNCTION public.lab_arena_open_scoring_v2(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt_v2(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_result JSONB,
  p_terminal_cause TEXT,
  p_output_ref TEXT,
  p_judgment_evidence JSONB,
  p_judgment_evidence_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_complete_attempt_v2$
DECLARE
  v_run public.lab_arena_runs;
  v_result JSONB;
  v_cache public.lab_arena_judgment_cache;
BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs WHERE run_id = p_run_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_run.kind <> 'score'
     OR v_run.judgment_cache_key IS NULL
     OR NOT COALESCE(v_run.judgment_group_leader, FALSE)
     OR p_terminal_cause <> 'accepted'
     OR pg_catalog.jsonb_typeof(p_judgment_evidence) IS DISTINCT FROM 'object'
     OR COALESCE(p_judgment_evidence_hash, '') !~ '^sha256:[0-9a-f]{64}$'
     OR p_judgment_evidence ->> 'cache_key' IS DISTINCT FROM
        v_run.judgment_cache_key
     OR p_judgment_evidence ->> 'scoring_input_hash' IS DISTINCT FROM
        v_run.judgment_input_hash
     OR p_judgment_evidence ->> 'source_score_run_id' IS DISTINCT FROM p_run_id
     OR p_judgment_evidence ->> 'source_scored_run_id' IS DISTINCT FROM
        v_run.scored_run_id
     OR p_judgment_evidence ->> 'source_output_ref' IS DISTINCT FROM p_output_ref
     OR p_judgment_evidence ->> 'source_runner_hotkey' IS DISTINCT FROM
        v_run.runner_hotkey
     OR pg_catalog.jsonb_typeof(
          v_run.claim_response -> 'runner_authority_exclusions'
        ) IS DISTINCT FROM 'array'
     OR NOT (
       v_run.claim_response -> 'runner_authority_exclusions'
       @> pg_catalog.jsonb_build_array(v_run.runner_hotkey)
     )
     OR p_judgment_evidence -> 'runner_authority_exclusions'
        IS DISTINCT FROM
        v_run.claim_response -> 'runner_authority_exclusions' THEN
    RAISE EXCEPTION 'lab_arena_judgment_completion_invalid'
      USING ERRCODE = '22023';
  END IF;
  v_result := public.lab_arena_complete_attempt(
    p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
  );
  IF v_result ->> 'status' <> 'accepted' THEN
    RETURN v_result;
  END IF;
  INSERT INTO public.lab_arena_judgment_cache (
    cache_key, scope_doc, scoring_input_hash, evidence_hash, evidence_doc,
    source_score_run_id, source_scored_run_id, source_runner_hotkey
  ) VALUES (
    v_run.judgment_cache_key, v_run.judgment_scope_doc,
    v_run.judgment_input_hash, p_judgment_evidence_hash,
    p_judgment_evidence, p_run_id, v_run.scored_run_id, v_run.runner_hotkey
  ) ON CONFLICT (cache_key) DO NOTHING;
  SELECT * INTO STRICT v_cache FROM public.lab_arena_judgment_cache
  WHERE cache_key = v_run.judgment_cache_key;
  UPDATE public.lab_arena_runs
  SET judgment_cache_source_run_id = v_cache.source_score_run_id
  WHERE run_id = p_run_id;
  UPDATE public.lab_arena_runs
  SET status = 'accepted', terminal_cause = 'accepted',
      result_doc = pg_catalog.jsonb_build_object(
        'schema_version', 'leadpoet.lab_arena.cached_run_result.v1',
        'terminal_status', 'accepted',
        'cache_key', v_cache.cache_key,
        'source_score_run_id', v_cache.source_score_run_id
      ),
      output_ref = v_cache.evidence_doc ->> 'source_output_ref',
      judgment_cache_source_run_id = v_cache.source_score_run_id
  WHERE round_id = v_run.round_id
    AND stage = v_run.stage
    AND kind = 'score'
    AND judgment_cache_key = v_run.judgment_cache_key
    AND run_id <> p_run_id
    AND status = 'pending'
    AND COALESCE(judgment_group_leader, FALSE) = FALSE;
  RETURN v_result || pg_catalog.jsonb_build_object(
    'judgment_cache_key', v_cache.cache_key,
    'judgment_source_run_id', v_cache.source_score_run_id
  );
END;
$lab_arena_complete_attempt_v2$;
ALTER FUNCTION public.lab_arena_complete_attempt_v2(
  TEXT, TEXT, JSONB, TEXT, TEXT, JSONB, TEXT
) OWNER TO lab_arena_owner;

-- Every retry remains the leader for the same gateway-computed identity.  A
-- failure or lease expiry cannot silently fall back to an uncached lucky roll.
DO $lab_arena_212_completion_retry_cache$
DECLARE
  v_definition TEXT;
  v_old_columns TEXT := $old$
        kind, scored_run_id, previous_runner_hotkey
$old$;
  v_new_columns TEXT := $new$
        kind, scored_run_id, previous_runner_hotkey, judgment_cache_key,
        judgment_input_hash, judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys
$new$;
  v_old_values TEXT := $old$
        v_run.runner_hotkey
$old$;
  v_new_values TEXT := $new$
        v_run.runner_hotkey, v_run.judgment_cache_key,
        v_run.judgment_input_hash, v_run.judgment_scope_doc,
        v_run.judgment_group_leader, v_run.judgment_group_miner_hotkeys
$new$;
  v_return_old TEXT := $old$
  RETURN pg_catalog.jsonb_build_object(
    'status', v_status, 'idempotent', FALSE,
    'run_id', p_run_id, 'attempt', v_run.attempt
  );
$old$;
  v_return_new TEXT := $new$
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
  RETURN pg_catalog.jsonb_build_object(
    'status', v_status, 'idempotent', FALSE,
    'run_id', p_run_id, 'attempt', v_run.attempt
  );
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;
  IF pg_catalog.strpos(v_definition, 'v_run.judgment_cache_key') > 0 THEN
    IF pg_catalog.strpos(
         v_definition, 'next identical output obtain an independent judgment'
       ) = 0 THEN
      IF pg_catalog.strpos(v_definition, v_return_old) = 0 THEN
        RAISE EXCEPTION 'lab_arena_complete_attempt_return_shape_unexpected';
      END IF;
      EXECUTE pg_catalog.replace(v_definition, v_return_old, v_return_new);
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old_columns) = 0
     OR pg_catalog.strpos(v_definition, v_old_values) = 0 THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_retry_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_old_columns, v_new_columns
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_values, v_new_values
  );
  IF pg_catalog.strpos(v_definition, v_return_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_complete_attempt_return_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_return_old, v_return_new
  );
  EXECUTE v_definition;
END;
$lab_arena_212_completion_retry_cache$;

DO $lab_arena_212_expiry_retry_cache$
DECLARE
  v_definition TEXT;
  v_old_columns TEXT := $old$
        attempt, status, lease_generation, stage_generation, kind, scored_run_id
$old$;
  v_new_columns TEXT := $new$
        attempt, status, lease_generation, stage_generation, kind, scored_run_id,
        previous_runner_hotkey, judgment_cache_key, judgment_input_hash,
        judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys
$new$;
  v_old_values TEXT := $old$
        v_run.kind, v_run.scored_run_id
$old$;
  v_new_values TEXT := $new$
        v_run.kind, v_run.scored_run_id, v_run.runner_hotkey,
        v_run.judgment_cache_key, v_run.judgment_input_hash,
        v_run.judgment_scope_doc, v_run.judgment_group_leader,
        v_run.judgment_group_miner_hotkeys
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_expire_leases'
    AND procedure.pronargs = 1;
  IF pg_catalog.strpos(v_definition, 'v_run.judgment_cache_key') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_old_columns) = 0
     OR pg_catalog.strpos(v_definition, v_old_values) = 0 THEN
    RAISE EXCEPTION 'lab_arena_expire_leases_retry_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(
    v_definition, v_old_columns, v_new_columns
  );
  v_definition := pg_catalog.replace(
    v_definition, v_old_values, v_new_values
  );
  EXECUTE v_definition;
END;
$lab_arena_212_expiry_retry_cache$;

-- Extend the latest validator-authority claim function without copying its
-- scheduling and review rules.  A duplicate group's one leader is claimable
-- only by a validator whose coldkey owns none of the group's miner hotkeys;
-- the configured organizer baseline keeps its existing exemption.
DO $lab_arena_212_group_claim_guard$
DECLARE
  v_definition TEXT;
  v_response_old TEXT := $old$
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
$old$;
  v_response_new TEXT := $new$
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
    'runner_authority_exclusions', pg_catalog.to_jsonb(ARRAY(
      SELECT DISTINCT excluded.hotkey
      FROM pg_catalog.unnest(
        COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[])
      ) AS excluded(hotkey)
      WHERE COALESCE(excluded.hotkey, '') <> ''
      ORDER BY excluded.hotkey
    )),
$new$;
  v_old TEXT := $old$
    AND (
      runs.miner_hotkey <> ALL (COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[]))
      OR EXISTS (
        SELECT 1
        FROM public.lab_arena_submissions AS baseline_submission
        WHERE baseline_submission.submission_id = runs.submission_id
          AND baseline_submission.round_id = runs.round_id
          AND baseline_submission.status = 'frozen'
          AND baseline_submission.is_king
          AND baseline_submission.miner_hotkey =
              (v_round.configuration_doc ->> 'baseline_hotkey')
      )
    )
$old$;
  v_new TEXT := $new$
    AND (
      (runs.kind = 'score'
       AND COALESCE(runs.judgment_group_leader, TRUE)
       AND (
         NOT COALESCE(runs.judgment_group_miner_hotkeys, ARRAY[runs.miner_hotkey])
           && COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[])
         OR NOT EXISTS (
           SELECT 1
           FROM pg_catalog.unnest(
             COALESCE(
               runs.judgment_group_miner_hotkeys,
               ARRAY[runs.miner_hotkey]
             )
           ) AS grouped_miner(hotkey)
           WHERE grouped_miner.hotkey = ANY (
             COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[])
           )
             AND grouped_miner.hotkey IS DISTINCT FROM
                 (v_round.configuration_doc ->> 'baseline_hotkey')
         )
       ))
      OR (runs.kind <> 'score' AND (
        runs.miner_hotkey <> ALL (COALESCE(p_excluded_miner_hotkeys, ARRAY[]::TEXT[]))
        OR EXISTS (
          SELECT 1
          FROM public.lab_arena_submissions AS baseline_submission
          WHERE baseline_submission.submission_id = runs.submission_id
            AND baseline_submission.round_id = runs.round_id
            AND baseline_submission.status = 'frozen'
            AND baseline_submission.is_king
            AND baseline_submission.miner_hotkey =
                (v_round.configuration_doc ->> 'baseline_hotkey')
        )
      ))
    )
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_claim_assignment'
    AND procedure.pronargs = 9;
  IF pg_catalog.strpos(v_definition, 'judgment_group_miner_hotkeys') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_claim_assignment_shape_unexpected';
    END IF;
    v_definition := pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
  IF pg_catalog.strpos(v_definition, 'runner_authority_exclusions') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_response_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_claim_response_shape_unexpected';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_response_old, v_response_new
    );
  END IF;
  EXECUTE v_definition;
END;
$lab_arena_212_group_claim_guard$;

DO $lab_arena_212_function_acl$
DECLARE signature TEXT; role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_open_scoring_v2(TEXT, SMALLINT, JSONB)',
    'public.lab_arena_complete_attempt_v2(TEXT, TEXT, JSONB, TEXT, TEXT, JSONB, TEXT)'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
END;
$lab_arena_212_function_acl$;

COMMENT ON TABLE public.lab_arena_judgment_cache IS
  'Immutable first accepted judge evidence for one gateway-computed Arena integrity cache identity.';
COMMENT ON FUNCTION public.lab_arena_open_scoring_v2(TEXT, SMALLINT, JSONB) IS
  'Opens stage-neutral integrity scoring; validated cache hits remain one accepted score run per execution output.';
COMMENT ON FUNCTION public.lab_arena_complete_attempt_v2(
  TEXT, TEXT, JSONB, TEXT, TEXT, JSONB, TEXT
) IS
  'Atomically freezes the first accepted judgment and binds duplicate score runs to its evidence; failures use the legacy completion path.';

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

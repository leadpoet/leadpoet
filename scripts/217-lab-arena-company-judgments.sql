-- Share context-independent company judgments across Arena scoring items.
-- Reusable evidence is immutable. A trusted score lease reserves every cache
-- miss atomically before any provider call, while destination duplicate and
-- qualification context remains outside the cache.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_company_judgments_requires_integrity$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_integrity_schema_v1()'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_complete_attempt_v2(text,text,jsonb,text,text,jsonb,text)'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply Arena judgment integrity migrations first';
  END IF;
END;
$lab_arena_company_judgments_requires_integrity$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS company_judgment_refs JSONB;
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_runs_company_judgment_refs_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_runs_company_judgment_refs_check CHECK (
    company_judgment_refs IS NULL
    OR (
      kind = 'score'
      AND pg_catalog.jsonb_typeof(company_judgment_refs) = 'array'
      AND pg_catalog.jsonb_array_length(company_judgment_refs) BETWEEN 0 AND 5
    )
  );

CREATE TABLE IF NOT EXISTS public.lab_arena_company_judgments (
  cache_key TEXT NOT NULL CHECK (cache_key ~ '^sha256:[0-9a-f]{64}$'),
  authority_slot INTEGER NOT NULL CHECK (authority_slot >= 0),
  scope_doc JSONB NOT NULL,
  company_input_hash TEXT NOT NULL CHECK (
    company_input_hash ~ '^sha256:[0-9a-f]{64}$'
  ),
  evidence_hash TEXT NOT NULL CHECK (evidence_hash ~ '^sha256:[0-9a-f]{64}$'),
  evidence_doc JSONB NOT NULL,
  source_score_run_id TEXT NOT NULL
    REFERENCES public.lab_arena_runs(run_id),
  source_scored_run_id TEXT NOT NULL
    REFERENCES public.lab_arena_runs(run_id),
  source_runner_hotkey TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  PRIMARY KEY (cache_key, authority_slot),
  UNIQUE (source_score_run_id, cache_key),
  CHECK (pg_catalog.jsonb_typeof(scope_doc) = 'object'),
  CHECK (pg_catalog.jsonb_typeof(evidence_doc) = 'object'),
  CHECK (scope_doc ->> 'cache_key' = cache_key),
  CHECK (scope_doc ->> 'company_input_hash' = company_input_hash),
  CHECK (evidence_doc ->> 'cache_key' = cache_key),
  CHECK (evidence_doc ->> 'company_input_hash' = company_input_hash),
  CHECK ((evidence_doc ->> 'authority_slot')::INTEGER = authority_slot),
  CHECK (evidence_doc ->> 'source_score_run_id' = source_score_run_id),
  CHECK (evidence_doc ->> 'source_scored_run_id' = source_scored_run_id),
  CHECK (evidence_doc ->> 'source_runner_hotkey' = source_runner_hotkey),
  CHECK (
    pg_catalog.jsonb_typeof(
      evidence_doc -> 'runner_authority_exclusions'
    ) = 'array'
    AND evidence_doc -> 'runner_authority_exclusions'
      @> pg_catalog.jsonb_build_array(source_runner_hotkey)
  ),
  CHECK (
    pg_catalog.jsonb_typeof(evidence_doc -> 'raw_judgment') = 'object'
    AND NOT (evidence_doc -> 'raw_judgment')
      ?| ARRAY[
        'company_index', 'company_qualified', 'duplicate_company',
        'duplicate_of_index'
      ]
  )
);
ALTER TABLE public.lab_arena_company_judgments OWNER TO lab_arena_owner;

DROP TRIGGER IF EXISTS lab_arena_company_judgments_append_only
  ON public.lab_arena_company_judgments;
CREATE TRIGGER lab_arena_company_judgments_append_only
  BEFORE UPDATE OR DELETE ON public.lab_arena_company_judgments
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_append_only_v1();

ALTER TABLE public.lab_arena_company_judgments ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS lab_arena_company_judgments_service_read
  ON public.lab_arena_company_judgments;
CREATE POLICY lab_arena_company_judgments_service_read
  ON public.lab_arena_company_judgments FOR SELECT TO lab_arena_service
  USING (TRUE);
REVOKE ALL ON TABLE public.lab_arena_company_judgments FROM PUBLIC;
GRANT SELECT ON TABLE public.lab_arena_company_judgments TO lab_arena_service;

CREATE TABLE IF NOT EXISTS public.lab_arena_company_judgment_reservations (
  cache_key TEXT PRIMARY KEY CHECK (cache_key ~ '^sha256:[0-9a-f]{64}$'),
  authority_slot INTEGER NOT NULL CHECK (authority_slot >= 0),
  company_input_hash TEXT NOT NULL CHECK (
    company_input_hash ~ '^sha256:[0-9a-f]{64}$'
  ),
  run_id TEXT NOT NULL REFERENCES public.lab_arena_runs(run_id),
  lease_generation BIGINT NOT NULL CHECK (lease_generation >= 1),
  runner_hotkey TEXT NOT NULL,
  lease_expires_at TIMESTAMPTZ NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp(),
  UNIQUE (cache_key, authority_slot),
  UNIQUE (run_id, cache_key)
);
ALTER TABLE public.lab_arena_company_judgment_reservations
  OWNER TO lab_arena_owner;
ALTER TABLE public.lab_arena_company_judgment_reservations
  ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.lab_arena_company_judgment_reservations FROM PUBLIC;

DO $lab_arena_company_judgment_table_acl$
DECLARE role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (
      SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
    ) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON TABLE public.lab_arena_company_judgments FROM %I',
        role_name
      );
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON TABLE public.lab_arena_company_judgment_reservations FROM %I',
        role_name
      );
    END IF;
  END LOOP;
END;
$lab_arena_company_judgment_table_acl$;

CREATE OR REPLACE FUNCTION public.lab_arena_company_quality_schema_v1()
RETURNS JSONB
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $lab_arena_company_quality_schema$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.company_quality_schema.v1',
    'version', 1
  );
$lab_arena_company_quality_schema$;
ALTER FUNCTION public.lab_arena_company_quality_schema_v1()
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_open_scoring_v3(
  p_round_id TEXT,
  p_stage SMALLINT,
  p_work_items JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_open_scoring_v3$
DECLARE
  v_round public.lab_arena_rounds;
  v_expected TEXT;
  v_next TEXT;
  v_generation BIGINT;
  v_item JSONB;
  v_scored public.lab_arena_runs;
  v_assignment TEXT;
  v_created INTEGER := 0;
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
       IS DISTINCT FROM 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'company_quality_policy'
       IS DISTINCT FROM 'company_quality_v1'
     OR v_round.configuration_doc #>> '{scorer_policy,company_quality_policy}'
       IS DISTINCT FROM 'company_quality_v1' THEN
    RAISE EXCEPTION 'lab_arena_company_quality_policy_required'
      USING ERRCODE = '22023';
  END IF;
  v_expected := 'stage' || p_stage::TEXT || '_closed';
  v_next := 'stage' || p_stage::TEXT || '_scoring';
  IF v_round.status <> v_expected THEN
    IF EXISTS (
      SELECT 1 FROM public.lab_arena_runs
      WHERE round_id = p_round_id AND stage = p_stage AND kind = 'score'
        AND company_judgment_refs IS NOT NULL
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
       OR COALESCE(v_item ->> 'submission_id', '')
          !~ '^[A-Za-z0-9._:-]{1,64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'icp_position')
          IS DISTINCT FROM 'number'
       OR (v_item ->> 'icp_position')::INTEGER NOT BETWEEN 0 AND 29
       OR pg_catalog.char_length(COALESCE(v_item ->> 'output_ref', ''))
          NOT BETWEEN 1 AND 1024
       OR pg_catalog.jsonb_typeof(v_item -> 'company_judgment_refs')
          IS DISTINCT FROM 'array'
       OR pg_catalog.jsonb_array_length(v_item -> 'company_judgment_refs')
          NOT BETWEEN 0 AND 5
       OR EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(
           v_item -> 'company_judgment_refs'
         ) WITH ORDINALITY AS refs(ref, ordinal)
         WHERE pg_catalog.jsonb_typeof(refs.ref) IS DISTINCT FROM 'object'
           OR refs.ref ->> 'schema_version' IS DISTINCT FROM
              'leadpoet.lab_arena.company_judgment_ref.v1'
           OR (refs.ref ->> 'company_index')::INTEGER
              IS DISTINCT FROM (refs.ordinal - 1)::INTEGER
           OR COALESCE(refs.ref ->> 'cache_key', '')
              !~ '^sha256:[0-9a-f]{64}$'
           OR COALESCE(refs.ref ->> 'company_input_hash', '')
              !~ '^sha256:[0-9a-f]{64}$'
           OR pg_catalog.jsonb_typeof(refs.ref -> 'scope_doc')
              IS DISTINCT FROM 'object'
           OR refs.ref #>> '{scope_doc,cache_key}' IS DISTINCT FROM
              refs.ref ->> 'cache_key'
           OR refs.ref #>> '{scope_doc,company_input_hash}' IS DISTINCT FROM
              refs.ref ->> 'company_input_hash'
           OR refs.ref #>> '{scope_doc,round_id}' IS DISTINCT FROM p_round_id
           OR refs.ref #>> '{scope_doc,network_name}' IS DISTINCT FROM
              v_round.arena_network_name
           OR (refs.ref #>> '{scope_doc,netuid}')::INTEGER IS DISTINCT FROM
              v_round.arena_netuid
           OR refs.ref #>> '{scope_doc,evaluation_date}' IS DISTINCT FROM
              v_round.evaluation_date
           OR refs.ref #>> '{scope_doc,integrity_policy}' IS DISTINCT FROM
              'arena_integrity_v1'
           OR refs.ref #>> '{scope_doc,company_quality_policy}' IS DISTINCT FROM
              'company_quality_v1'
           OR refs.ref #>> '{scope_doc,scorer_image_digest}' IS DISTINCT FROM
              v_round.configuration_doc ->> 'scorer_image_digest'
           OR refs.ref #>> '{scope_doc,scorer_image_reference}' IS DISTINCT FROM
              v_round.configuration_doc ->> 'scorer_image_reference'
       ) THEN
      RAISE EXCEPTION 'lab_arena_company_scoring_item_invalid'
        USING ERRCODE = '22023';
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
    v_assignment := p_round_id || ':' || v_scored.submission_id || ':' ||
      p_stage::TEXT || ':' || v_scored.icp_position::TEXT || ':score';
    INSERT INTO public.lab_arena_runs (
      run_id, assignment_id, round_id, submission_id, miner_hotkey, stage,
      icp_position, attempt, status, stage_generation, kind, scored_run_id,
      company_judgment_refs
    ) VALUES (
      v_assignment || ':1', v_assignment, p_round_id, v_scored.submission_id,
      v_scored.miner_hotkey, p_stage, v_scored.icp_position, 1, 'pending',
      v_generation, 'score', v_scored.run_id,
      v_item -> 'company_judgment_refs'
    );
    v_created := v_created + 1;
  END LOOP;
  UPDATE public.lab_arena_rounds
  SET status = v_next, status_generation = status_generation + 1,
      stage_generation = v_generation
  WHERE round_id = p_round_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'round_status', v_next,
    'stage_generation', v_generation, 'assignments', v_created,
    'reused', 0
  );
END;
$lab_arena_open_scoring_v3$;
ALTER FUNCTION public.lab_arena_open_scoring_v3(TEXT, SMALLINT, JSONB)
  OWNER TO lab_arena_owner;

-- Called only by the claim RPC while its chosen score row is locked. It
-- chooses compatible immutable evidence, then reserves all unique misses in
-- sorted key order. Any contention rolls back this claim's reservations.
CREATE OR REPLACE FUNCTION public.lab_arena__prepare_company_judgment_claim_v1(
  p_run_id TEXT,
  p_runner_hotkey TEXT,
  p_lease_generation BIGINT,
  p_lease_expires_at TIMESTAMPTZ
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_prepare_company_judgment_claim$
DECLARE
  v_run public.lab_arena_runs;
  v_submission public.lab_arena_submissions;
  v_ref JSONB;
  v_hit public.lab_arena_company_judgments;
  v_reservation public.lab_arena_company_judgment_reservations;
  v_slot INTEGER;
  v_inserted INTEGER;
  v_reserved JSONB := '{}'::JSONB;
  v_hits JSONB := '[]'::JSONB;
  v_misses JSONB := '[]'::JSONB;
BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs
  WHERE run_id = p_run_id FOR UPDATE;
  IF NOT FOUND OR v_run.kind <> 'score'
     OR pg_catalog.jsonb_typeof(v_run.company_judgment_refs)
        IS DISTINCT FROM 'array'
     OR p_lease_generation <> v_run.lease_generation + 1
     OR p_lease_expires_at <= pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_claim_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT * INTO STRICT v_submission FROM public.lab_arena_submissions
  WHERE submission_id = v_run.submission_id AND round_id = v_run.round_id;

  FOR v_ref IN
    SELECT value
    FROM pg_catalog.jsonb_array_elements(v_run.company_judgment_refs)
    ORDER BY value ->> 'cache_key', (value ->> 'company_index')::INTEGER
  LOOP
    SELECT * INTO v_hit
    FROM public.lab_arena_company_judgments AS accepted
    WHERE accepted.cache_key = v_ref ->> 'cache_key'
      AND accepted.company_input_hash = v_ref ->> 'company_input_hash'
      AND (
        v_submission.is_king
        OR NOT accepted.evidence_doc -> 'runner_authority_exclusions'
          @> pg_catalog.jsonb_build_array(v_run.miner_hotkey)
      )
    ORDER BY accepted.authority_slot
    LIMIT 1;
    IF FOUND THEN
      v_hits := v_hits || pg_catalog.jsonb_build_array(
        pg_catalog.jsonb_build_object(
          'company_index', (v_ref ->> 'company_index')::INTEGER,
          'cache_key', v_hit.cache_key,
          'company_input_hash', v_hit.company_input_hash,
          'authority_slot', v_hit.authority_slot,
          'evidence_hash', v_hit.evidence_hash,
          'evidence_doc', v_hit.evidence_doc
        )
      );
      CONTINUE;
    END IF;

    IF v_reserved ? (v_ref ->> 'cache_key') THEN
      v_slot := (v_reserved ->> (v_ref ->> 'cache_key'))::INTEGER;
    ELSE
      DELETE FROM public.lab_arena_company_judgment_reservations
      WHERE cache_key = v_ref ->> 'cache_key'
        AND lease_expires_at <= pg_catalog.clock_timestamp();
      SELECT * INTO v_reservation
      FROM public.lab_arena_company_judgment_reservations
      WHERE cache_key = v_ref ->> 'cache_key';
      IF FOUND THEN
        IF v_reservation.run_id <> v_run.run_id
           OR v_reservation.lease_generation <> p_lease_generation
           OR v_reservation.runner_hotkey <> p_runner_hotkey THEN
          DELETE FROM public.lab_arena_company_judgment_reservations
          WHERE run_id = v_run.run_id
            AND lease_generation = p_lease_generation;
          RETURN pg_catalog.jsonb_build_object(
            'status', 'cache_busy', 'cache_key', v_ref ->> 'cache_key'
          );
        END IF;
        v_slot := v_reservation.authority_slot;
        v_reserved := v_reserved || pg_catalog.jsonb_build_object(
          v_ref ->> 'cache_key', v_slot
        );
      ELSE
      SELECT GREATEST(
        COALESCE((
          SELECT pg_catalog.max(authority_slot)
          FROM public.lab_arena_company_judgments
          WHERE cache_key = v_ref ->> 'cache_key'
        ), -1),
        COALESCE((
          SELECT pg_catalog.max(authority_slot)
          FROM public.lab_arena_company_judgment_reservations
          WHERE cache_key = v_ref ->> 'cache_key'
        ), -1)
      ) + 1 INTO v_slot;
      INSERT INTO public.lab_arena_company_judgment_reservations (
        cache_key, authority_slot, company_input_hash, run_id,
        lease_generation, runner_hotkey, lease_expires_at
      ) VALUES (
        v_ref ->> 'cache_key', v_slot,
        v_ref ->> 'company_input_hash', v_run.run_id,
        p_lease_generation, p_runner_hotkey, p_lease_expires_at
      ) ON CONFLICT (cache_key) DO NOTHING;
      GET DIAGNOSTICS v_inserted = ROW_COUNT;
      IF v_inserted <> 1 THEN
        DELETE FROM public.lab_arena_company_judgment_reservations
        WHERE run_id = v_run.run_id
          AND lease_generation = p_lease_generation;
        RETURN pg_catalog.jsonb_build_object(
          'status', 'cache_busy', 'cache_key', v_ref ->> 'cache_key'
        );
      END IF;
      v_reserved := v_reserved || pg_catalog.jsonb_build_object(
        v_ref ->> 'cache_key', v_slot
      );
      END IF;
    END IF;
    v_misses := v_misses || pg_catalog.jsonb_build_array(
      pg_catalog.jsonb_build_object(
        'company_index', (v_ref ->> 'company_index')::INTEGER,
        'cache_key', v_ref ->> 'cache_key',
        'company_input_hash', v_ref ->> 'company_input_hash',
        'authority_slot', v_slot
      )
    );
  END LOOP;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.company_judgment_lease.v1',
    'hits', v_hits,
    'misses', v_misses
  );
END;
$lab_arena_prepare_company_judgment_claim$;
ALTER FUNCTION public.lab_arena__prepare_company_judgment_claim_v1(
  TEXT, TEXT, BIGINT, TIMESTAMPTZ
) OWNER TO lab_arena_owner;

-- Keep reservations on the same expiry as the score lease and release them
-- on every terminal transition, including legacy failure and stage closure.
CREATE OR REPLACE FUNCTION public.lab_arena_company_judgment_lease_sync_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_company_judgment_lease_sync$
BEGIN
  IF NEW.status <> 'leased' THEN
    DELETE FROM public.lab_arena_company_judgment_reservations
    WHERE run_id = NEW.run_id;
  ELSIF NEW.lease_expires_at IS DISTINCT FROM OLD.lease_expires_at THEN
    UPDATE public.lab_arena_company_judgment_reservations
    SET lease_expires_at = NEW.lease_expires_at
    WHERE run_id = NEW.run_id
      AND lease_generation = NEW.lease_generation;
  END IF;
  RETURN NULL;
END;
$lab_arena_company_judgment_lease_sync$;
ALTER FUNCTION public.lab_arena_company_judgment_lease_sync_v1()
  OWNER TO lab_arena_owner;
DROP TRIGGER IF EXISTS lab_arena_company_judgment_lease_sync
  ON public.lab_arena_runs;
CREATE TRIGGER lab_arena_company_judgment_lease_sync
  AFTER UPDATE OF status, lease_expires_at ON public.lab_arena_runs
  FOR EACH ROW
  WHEN (OLD.company_judgment_refs IS NOT NULL)
  EXECUTE FUNCTION public.lab_arena_company_judgment_lease_sync_v1();

-- Add reservation eligibility and lease payloads to the latest claim
-- function without copying its validator authority, review, or budget rules.
DO $lab_arena_company_judgment_claim_patch$
DECLARE
  v_definition TEXT;
  v_guard_old TEXT := $old$
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
$old$;
  v_guard_new TEXT := $new$
    AND (
      runs.company_judgment_refs IS NULL
      OR NOT EXISTS (
        SELECT 1
        FROM pg_catalog.jsonb_array_elements(
          runs.company_judgment_refs
        ) AS company_ref(ref)
        JOIN public.lab_arena_company_judgment_reservations AS reservation
          ON reservation.cache_key = company_ref.ref ->> 'cache_key'
        WHERE reservation.lease_expires_at > pg_catalog.clock_timestamp()
      )
    )
    AND (runs.previous_runner_hotkey IS NULL OR runs.previous_runner_hotkey <> p_runner_hotkey
$new$;
  v_response_old TEXT := $old$
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
$old$;
  v_response_new TEXT := $new$
  IF v_run.company_judgment_refs IS NOT NULL THEN
    v_response := public.lab_arena__prepare_company_judgment_claim_v1(
      v_run.run_id, p_runner_hotkey, v_run.lease_generation + 1, v_expires
    );
    IF v_response ->> 'status' = 'cache_busy' THEN
      RETURN v_response;
    END IF;
  END IF;
  v_response := pg_catalog.jsonb_build_object(
    'status', 'leased',
$new$;
  v_update_old TEXT := $old$
  UPDATE public.lab_arena_runs
  SET status = 'leased', runner_hotkey = p_runner_hotkey, lease_token_hash = p_lease_token_hash,
$old$;
  v_update_new TEXT := $new$
  IF v_run.company_judgment_refs IS NOT NULL THEN
    v_response := v_response || pg_catalog.jsonb_build_object(
      'company_judgment_cache', public.lab_arena__prepare_company_judgment_claim_v1(
        v_run.run_id, p_runner_hotkey, v_run.lease_generation + 1, v_expires
      )
    );
  END IF;
  UPDATE public.lab_arena_runs
  SET status = 'leased', runner_hotkey = p_runner_hotkey, lease_token_hash = p_lease_token_hash,
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_claim_assignment'
    AND procedure.pronargs = 9;
  IF pg_catalog.strpos(v_definition, 'company_judgment_cache') > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(v_definition, v_guard_old) = 0
     OR pg_catalog.strpos(v_definition, v_response_old) = 0
     OR pg_catalog.strpos(v_definition, v_update_old) = 0 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment_company_shape_unexpected';
  END IF;
  v_definition := pg_catalog.replace(v_definition, v_guard_old, v_guard_new);
  v_definition := pg_catalog.replace(
    v_definition, v_response_old, v_response_new
  );
  v_definition := pg_catalog.replace(v_definition, v_update_old, v_update_new);
  EXECUTE v_definition;
END;
$lab_arena_company_judgment_claim_patch$;

CREATE OR REPLACE FUNCTION public.lab_arena_complete_attempt_v3(
  p_run_id TEXT,
  p_lease_token_hash TEXT,
  p_result JSONB,
  p_terminal_cause TEXT,
  p_output_ref TEXT,
  p_output_hash TEXT,
  p_company_judgment_evidence JSONB,
  p_completion_request_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_complete_attempt_v3$
DECLARE
  v_run public.lab_arena_runs;
  v_result JSONB;
  v_context JSONB;
  v_item JSONB;
  v_doc JSONB;
  v_reservation public.lab_arena_company_judgment_reservations;
  v_expected INTEGER;
  v_supplied INTEGER;
BEGIN
  SELECT * INTO v_run FROM public.lab_arena_runs
  WHERE run_id = p_run_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_run_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_run.status IN ('accepted', 'failed') THEN
    RETURN public.lab_arena_complete_attempt(
      p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
    );
  END IF;
  v_context := v_run.claim_response -> 'company_judgment_cache';
  IF v_run.kind <> 'score'
     OR pg_catalog.jsonb_typeof(v_run.company_judgment_refs)
        IS DISTINCT FROM 'array'
     OR pg_catalog.jsonb_typeof(v_context) IS DISTINCT FROM 'object'
     OR v_context ->> 'schema_version' IS DISTINCT FROM
        'leadpoet.lab_arena.company_judgment_lease.v1'
     OR pg_catalog.jsonb_typeof(p_company_judgment_evidence)
        IS DISTINCT FROM 'array'
     OR COALESCE(p_completion_request_hash, '')
        !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_completion_invalid'
      USING ERRCODE = '22023';
  END IF;
  IF p_terminal_cause <> 'accepted' THEN
    IF pg_catalog.jsonb_array_length(p_company_judgment_evidence) <> 0
       OR COALESCE(p_output_hash, '') <> '' THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_failure_invalid'
        USING ERRCODE = '22023';
    END IF;
    RETURN public.lab_arena_complete_attempt(
      p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
    );
  END IF;
  IF COALESCE(p_output_hash, '') !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_output_hash_invalid'
      USING ERRCODE = '22023';
  END IF;
  SELECT pg_catalog.count(DISTINCT miss ->> 'cache_key')::INTEGER
  INTO v_expected
  FROM pg_catalog.jsonb_array_elements(v_context -> 'misses') AS misses(miss);
  SELECT pg_catalog.count(DISTINCT evidence ->> 'cache_key')::INTEGER
  INTO v_supplied
  FROM pg_catalog.jsonb_array_elements(
    p_company_judgment_evidence
  ) AS evidence_rows(evidence);
  IF v_expected <> v_supplied
     OR v_supplied <> pg_catalog.jsonb_array_length(
       p_company_judgment_evidence
     ) THEN
    RAISE EXCEPTION 'lab_arena_company_judgment_evidence_incomplete'
      USING ERRCODE = '22023';
  END IF;
  FOR v_item IN
    SELECT value
    FROM pg_catalog.jsonb_array_elements(p_company_judgment_evidence)
    ORDER BY value ->> 'cache_key'
  LOOP
    v_doc := v_item -> 'evidence_doc';
    IF pg_catalog.jsonb_typeof(v_item) IS DISTINCT FROM 'object'
       OR COALESCE(v_item ->> 'cache_key', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR COALESCE(v_item ->> 'company_input_hash', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_item -> 'authority_slot')
          IS DISTINCT FROM 'number'
       OR COALESCE(v_item ->> 'evidence_hash', '')
          !~ '^sha256:[0-9a-f]{64}$'
       OR pg_catalog.jsonb_typeof(v_doc) IS DISTINCT FROM 'object'
       OR NOT EXISTS (
         SELECT 1
         FROM pg_catalog.jsonb_array_elements(v_context -> 'misses') AS miss(ref)
         WHERE miss.ref ->> 'cache_key' = v_item ->> 'cache_key'
           AND miss.ref ->> 'company_input_hash' =
               v_item ->> 'company_input_hash'
           AND (miss.ref ->> 'authority_slot')::INTEGER =
               (v_item ->> 'authority_slot')::INTEGER
       )
       OR v_doc ->> 'schema_version' IS DISTINCT FROM
          'leadpoet.lab_arena.company_judgment_evidence.v1'
       OR v_doc ->> 'cache_key' IS DISTINCT FROM v_item ->> 'cache_key'
       OR v_doc ->> 'company_input_hash' IS DISTINCT FROM
          v_item ->> 'company_input_hash'
       OR (v_doc ->> 'authority_slot')::INTEGER IS DISTINCT FROM
          (v_item ->> 'authority_slot')::INTEGER
       OR v_doc ->> 'source_score_run_id' IS DISTINCT FROM p_run_id
       OR v_doc ->> 'source_scored_run_id' IS DISTINCT FROM
          v_run.scored_run_id
       OR v_doc ->> 'source_output_ref' IS DISTINCT FROM p_output_ref
       OR v_doc ->> 'source_output_hash' IS DISTINCT FROM p_output_hash
       OR v_doc ->> 'source_runner_hotkey' IS DISTINCT FROM
          v_run.runner_hotkey
       OR v_doc ->> 'source_claim_request_id' IS DISTINCT FROM
          v_run.claim_request_id
       OR v_doc ->> 'source_claim_request_hash' IS DISTINCT FROM
          v_run.claim_request_hash
       OR (v_doc ->> 'source_lease_generation')::BIGINT IS DISTINCT FROM
          v_run.lease_generation
       OR v_doc ->> 'source_completion_request_hash' IS DISTINCT FROM
          p_completion_request_hash
       OR v_doc -> 'runner_authority_exclusions' IS DISTINCT FROM
          v_run.claim_response -> 'runner_authority_exclusions'
       OR pg_catalog.jsonb_typeof(v_doc -> 'raw_judgment')
          IS DISTINCT FROM 'object'
       OR (v_doc -> 'raw_judgment') ?| ARRAY[
         'company_index', 'company_qualified', 'duplicate_company',
         'duplicate_of_index'
       ] THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_evidence_invalid'
        USING ERRCODE = '22023';
    END IF;
    SELECT * INTO v_reservation
    FROM public.lab_arena_company_judgment_reservations
    WHERE cache_key = v_item ->> 'cache_key'
      AND authority_slot = (v_item ->> 'authority_slot')::INTEGER
      AND company_input_hash = v_item ->> 'company_input_hash'
      AND run_id = p_run_id
      AND lease_generation = v_run.lease_generation
      AND runner_hotkey = v_run.runner_hotkey
      AND lease_expires_at > pg_catalog.clock_timestamp()
    FOR UPDATE;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_reservation_stale'
        USING ERRCODE = 'P0003';
    END IF;
  END LOOP;

  v_result := public.lab_arena_complete_attempt(
    p_run_id, p_lease_token_hash, p_result, p_terminal_cause, p_output_ref
  );
  IF v_result ->> 'status' <> 'accepted' THEN
    RETURN v_result;
  END IF;
  FOR v_item IN
    SELECT value
    FROM pg_catalog.jsonb_array_elements(p_company_judgment_evidence)
    ORDER BY value ->> 'cache_key'
  LOOP
    v_doc := v_item -> 'evidence_doc';
    INSERT INTO public.lab_arena_company_judgments (
      cache_key, authority_slot, scope_doc, company_input_hash,
      evidence_hash, evidence_doc, source_score_run_id,
      source_scored_run_id, source_runner_hotkey
    )
    SELECT
      v_item ->> 'cache_key',
      (v_item ->> 'authority_slot')::INTEGER,
      ref -> 'scope_doc',
      v_item ->> 'company_input_hash',
      v_item ->> 'evidence_hash',
      v_doc,
      p_run_id,
      v_run.scored_run_id,
      v_run.runner_hotkey
    FROM pg_catalog.jsonb_array_elements(
      v_run.company_judgment_refs
    ) AS company_refs(ref)
    WHERE ref ->> 'cache_key' = v_item ->> 'cache_key'
      AND ref ->> 'company_input_hash' = v_item ->> 'company_input_hash'
    ORDER BY (ref ->> 'company_index')::INTEGER
    LIMIT 1;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_company_judgment_ref_missing'
        USING ERRCODE = '22023';
    END IF;
  END LOOP;
  RETURN v_result || pg_catalog.jsonb_build_object(
    'company_judgments_stored', v_supplied,
    'company_judgments_reused',
      pg_catalog.jsonb_array_length(v_context -> 'hits')
  );
END;
$lab_arena_complete_attempt_v3$;
ALTER FUNCTION public.lab_arena_complete_attempt_v3(
  TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, JSONB, TEXT
) OWNER TO lab_arena_owner;

-- Preserve refs across ordinary judge failures and expired leases. The
-- historical whole-item cache columns remain unchanged for old rounds.
DO $lab_arena_company_judgment_retry_patch$
DECLARE
  v_definition TEXT;
  v_complete_columns_old TEXT := $old$
        kind, scored_run_id, previous_runner_hotkey, judgment_cache_key,
        judgment_input_hash, judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys
$old$;
  v_complete_columns_new TEXT := $new$
        kind, scored_run_id, previous_runner_hotkey, judgment_cache_key,
        judgment_input_hash, judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys, company_judgment_refs
$new$;
  v_complete_values_old TEXT := $old$
        v_run.runner_hotkey, v_run.judgment_cache_key,
        v_run.judgment_input_hash, v_run.judgment_scope_doc,
        v_run.judgment_group_leader, v_run.judgment_group_miner_hotkeys
$old$;
  v_complete_values_new TEXT := $new$
        v_run.runner_hotkey, v_run.judgment_cache_key,
        v_run.judgment_input_hash, v_run.judgment_scope_doc,
        v_run.judgment_group_leader, v_run.judgment_group_miner_hotkeys,
        v_run.company_judgment_refs
$new$;
  v_expiry_columns_old TEXT := $old$
        attempt, status, lease_generation, stage_generation, kind, scored_run_id,
        previous_runner_hotkey, judgment_cache_key, judgment_input_hash,
        judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys
$old$;
  v_expiry_columns_new TEXT := $new$
        attempt, status, lease_generation, stage_generation, kind, scored_run_id,
        previous_runner_hotkey, judgment_cache_key, judgment_input_hash,
        judgment_scope_doc, judgment_group_leader,
        judgment_group_miner_hotkeys, company_judgment_refs
$new$;
  v_expiry_values_old TEXT := $old$
        v_run.kind, v_run.scored_run_id, v_run.runner_hotkey,
        v_run.judgment_cache_key, v_run.judgment_input_hash,
        v_run.judgment_scope_doc, v_run.judgment_group_leader,
        v_run.judgment_group_miner_hotkeys
$old$;
  v_expiry_values_new TEXT := $new$
        v_run.kind, v_run.scored_run_id, v_run.runner_hotkey,
        v_run.judgment_cache_key, v_run.judgment_input_hash,
        v_run.judgment_scope_doc, v_run.judgment_group_leader,
        v_run.judgment_group_miner_hotkeys, v_run.company_judgment_refs
$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_complete_attempt'
    AND procedure.pronargs = 5;
  IF pg_catalog.strpos(v_definition, 'v_run.company_judgment_refs') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_complete_columns_old) = 0
       OR pg_catalog.strpos(v_definition, v_complete_values_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_complete_attempt_company_retry_unexpected';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_complete_columns_old, v_complete_columns_new
    );
    v_definition := pg_catalog.replace(
      v_definition, v_complete_values_old, v_complete_values_new
    );
    EXECUTE v_definition;
  END IF;

  SELECT pg_catalog.pg_get_functiondef(procedure.oid) INTO v_definition
  FROM pg_catalog.pg_proc AS procedure
  JOIN pg_catalog.pg_namespace AS namespace
    ON namespace.oid = procedure.pronamespace
  WHERE namespace.nspname = 'public'
    AND procedure.proname = 'lab_arena_expire_leases'
    AND procedure.pronargs = 1;
  IF pg_catalog.strpos(v_definition, 'v_run.company_judgment_refs') = 0 THEN
    IF pg_catalog.strpos(v_definition, v_expiry_columns_old) = 0
       OR pg_catalog.strpos(v_definition, v_expiry_values_old) = 0 THEN
      RAISE EXCEPTION 'lab_arena_expire_leases_company_retry_unexpected';
    END IF;
    v_definition := pg_catalog.replace(
      v_definition, v_expiry_columns_old, v_expiry_columns_new
    );
    v_definition := pg_catalog.replace(
      v_definition, v_expiry_values_old, v_expiry_values_new
    );
    EXECUTE v_definition;
  END IF;
END;
$lab_arena_company_judgment_retry_patch$;

DO $lab_arena_company_judgment_function_acl$
DECLARE signature TEXT; role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_company_quality_schema_v1()',
    'public.lab_arena_open_scoring_v3(TEXT, SMALLINT, JSONB)',
    'public.lab_arena_complete_attempt_v3(TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, JSONB, TEXT)'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
      ) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
  REVOKE ALL ON FUNCTION
    public.lab_arena__prepare_company_judgment_claim_v1(
      TEXT, TEXT, BIGINT, TIMESTAMPTZ
    ) FROM PUBLIC, lab_arena_service;
  REVOKE ALL ON FUNCTION
    public.lab_arena_company_judgment_lease_sync_v1()
    FROM PUBLIC, lab_arena_service;
END;
$lab_arena_company_judgment_function_acl$;

COMMENT ON TABLE public.lab_arena_company_judgments IS
  'Immutable raw company-quality judgments keyed by normalized input and authority slot.';
COMMENT ON TABLE public.lab_arena_company_judgment_reservations IS
  'One active trusted score lease for every unresolved company cache key.';
COMMENT ON FUNCTION public.lab_arena_open_scoring_v3(
  TEXT, SMALLINT, JSONB
) IS
  'Opens company-quality scoring with one assignment per ICP and ordered company refs.';
COMMENT ON FUNCTION public.lab_arena_complete_attempt_v3(
  TEXT, TEXT, JSONB, TEXT, TEXT, TEXT, JSONB, TEXT
) IS
  'Atomically completes one score and freezes only cacheable raw judgments reserved by its trusted lease.';

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

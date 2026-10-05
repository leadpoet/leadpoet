-- Miner-authorized recovery for a proved, zero-charge provider credit failure.
-- Existing attempts and accounting remain immutable. The ordinary claim path
-- leases the new pending attempt and keeps the frozen round policy in force.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

ALTER TABLE public.lab_arena_runs
  ADD COLUMN IF NOT EXISTS credit_retry_parent_run_id TEXT
    REFERENCES public.lab_arena_runs(run_id),
  ADD COLUMN IF NOT EXISTS credit_retry_request_hash TEXT;
ALTER TABLE public.lab_arena_runs
  DROP CONSTRAINT IF EXISTS lab_arena_credit_retry_request_hash_check;
ALTER TABLE public.lab_arena_runs
  ADD CONSTRAINT lab_arena_credit_retry_request_hash_check CHECK (
    credit_retry_request_hash IS NULL OR
    credit_retry_request_hash ~ '^sha256:[0-9a-f]{64}$'
  );
CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_credit_retry_parent_uq
  ON public.lab_arena_runs(credit_retry_parent_run_id)
  WHERE credit_retry_parent_run_id IS NOT NULL;

CREATE OR REPLACE FUNCTION public.lab_arena_credit_retry_lineage_immutable_v1()
RETURNS TRIGGER LANGUAGE plpgsql
SET search_path = pg_catalog
AS $credit_retry_lineage_immutable$
BEGIN
  IF NEW.credit_retry_parent_run_id IS DISTINCT FROM
       OLD.credit_retry_parent_run_id OR
     NEW.credit_retry_request_hash IS DISTINCT FROM
       OLD.credit_retry_request_hash THEN
    RAISE EXCEPTION 'lab_arena_credit_retry_lineage_immutable'
      USING ERRCODE = '42501';
  END IF;
  RETURN NEW;
END;
$credit_retry_lineage_immutable$;
ALTER FUNCTION public.lab_arena_credit_retry_lineage_immutable_v1()
  OWNER TO lab_arena_owner;
DO $credit_retry_lineage_trigger$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_trigger
    WHERE tgrelid='public.lab_arena_runs'::regclass
      AND tgname='lab_arena_credit_retry_lineage_immutable'
      AND NOT tgisinternal
  ) THEN
    CREATE TRIGGER lab_arena_credit_retry_lineage_immutable
      BEFORE UPDATE OF credit_retry_parent_run_id,credit_retry_request_hash
      ON public.lab_arena_runs FOR EACH ROW
      EXECUTE FUNCTION public.lab_arena_credit_retry_lineage_immutable_v1();
  END IF;
END;
$credit_retry_lineage_trigger$;

CREATE OR REPLACE FUNCTION public.lab_arena_retry_credit_failures_v1(
  p_round_id TEXT, p_submission_id TEXT, p_miner_hotkey TEXT,
  p_request_hash TEXT
) RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_retry_credit_failures$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_stage SMALLINT;
  v_kind TEXT;
  v_deadline_text TEXT;
  v_deadline TIMESTAMPTZ;
  v_stage_count INTEGER;
  v_stage1_count INTEGER;
  v_total_count INTEGER;
  v_parallel BOOLEAN;
  v_baseline_first BOOLEAN;
  v_replayed INTEGER;
  v_created INTEGER := 0;
  v_candidate public.lab_arena_runs;
  v_group_leader BOOLEAN;
BEGIN
  IF p_request_hash IS NULL OR
     p_request_hash !~ '^sha256:[0-9a-f]{64}$' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','invalid_request_hash');
  END IF;

  -- This RPC is exposed only to the gateway service. Its signed-owner check
  -- occurs before the call; the database independently binds that owner to
  -- the exact round and submission before revealing a replay or run count.
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  SELECT * INTO v_submission FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id
    AND miner_hotkey = p_miner_hotkey FOR UPDATE;
  IF NOT FOUND OR v_round.round_id IS NULL OR
     v_round.arena_network_name IS NULL OR v_round.arena_netuid IS NULL OR
     v_submission.is_king OR v_submission.status <> 'frozen' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','submission_unavailable');
  END IF;

  SELECT count(*) INTO v_replayed FROM public.lab_arena_runs
  WHERE round_id = p_round_id AND submission_id = p_submission_id
    AND miner_hotkey = p_miner_hotkey
    AND credit_retry_request_hash = p_request_hash;
  IF v_replayed > 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','replayed','requeued_count',v_replayed);
  END IF;

  IF v_round.cancel_reason IS NOT NULL OR
     v_round.status NOT IN ('stage1','stage2',
                            'stage1_scoring','stage2_scoring') OR
     v_round.configuration_doc ->> 'max_attempts_per_assignment'
       IS DISTINCT FROM '2' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','stage_unavailable');
  END IF;
  v_stage := CASE WHEN v_round.status IN ('stage1','stage1_scoring')
                  THEN 1 ELSE 2 END;
  v_kind := CASE WHEN v_round.status IN ('stage1','stage2')
                 THEN 'execute' ELSE 'score' END;
  v_parallel := COALESCE(
    v_round.configuration_doc -> 'parallel_twenty_icp_execution' =
      'true'::JSONB,FALSE);
  v_baseline_first := v_round.configuration_doc ->>
    'execution_sequence_policy' = 'baseline_scored_first_v1';
  v_deadline_text := v_round.configuration_doc #>> ARRAY[
    'schedule', CASE v_round.status
      WHEN 'stage1' THEN 'stage_1_close'
      WHEN 'stage2' THEN 'stage_2_close'
      WHEN 'stage1_scoring' THEN 'stage_1_scoring_close'
      ELSE 'final_scoring_close' END];
  IF v_deadline_text IS NULL OR
     v_deadline_text !~
       '^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','deadline_unavailable');
  END IF;
  v_deadline := v_deadline_text::TIMESTAMPTZ;
  IF pg_catalog.clock_timestamp() >= v_deadline
     AND NOT (v_round.status='stage1' AND v_parallel) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','deadline_passed');
  END IF;
  v_stage1_count := (v_round.configuration_doc ->>
    'stage_1_icp_count')::INTEGER;
  v_stage_count := (v_round.configuration_doc ->>
    'stage_2_icp_count')::INTEGER;
  v_total_count := v_stage1_count + v_stage_count;
  IF v_stage1_count IS NULL OR v_stage_count IS NULL OR
     v_stage1_count < 1 OR v_stage_count < 1 OR
     v_total_count NOT BETWEEN 2 AND 20 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','benchmark_unavailable');
  END IF;

  FOR v_candidate IN
    SELECT previous.* FROM public.lab_arena_runs AS previous
    WHERE previous.round_id = p_round_id
      AND previous.submission_id = p_submission_id
      AND previous.miner_hotkey = p_miner_hotkey
      AND (previous.stage = v_stage OR
           (v_round.status='stage1' AND v_parallel AND
            previous.stage=2))
      AND previous.kind = v_kind
      AND previous.stage_generation = v_round.stage_generation
      AND previous.icp_position >= CASE
        WHEN previous.stage=1 OR v_baseline_first THEN 0
        ELSE v_stage1_count END
      AND previous.icp_position < CASE
        WHEN previous.stage=1 THEN v_stage1_count
        ELSE v_total_count END
      AND previous.attempt = 1 AND previous.status = 'failed'
      AND previous.terminal_cause = 'credential_error'
      AND previous.result_doc ->> 'terminal_status' = 'credential_error'
      AND previous.output_ref IS NULL
      AND previous.assignment_id || ':1' = previous.run_id
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS later
        WHERE later.assignment_id = previous.assignment_id
          AND later.attempt = 2)
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS accepted
        WHERE accepted.assignment_id = previous.assignment_id
          AND accepted.status = 'accepted')
    ORDER BY previous.icp_position, previous.run_id
    LIMIT 20
  LOOP
    -- Parallel execution may queue stage-2 assignments while the round is
    -- still stage1. Each assignment keeps its own frozen stage deadline.
    IF v_candidate.stage=2 AND v_round.status='stage1' THEN
      v_deadline_text := v_round.configuration_doc #>>
        '{schedule,stage_2_close}';
      IF v_deadline_text IS NULL OR v_deadline_text !~
           '^20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$'
         OR pg_catalog.clock_timestamp() >=
           v_deadline_text::TIMESTAMPTZ THEN
        CONTINUE;
      END IF;
    ELSIF pg_catalog.clock_timestamp() >= v_deadline THEN
      CONTINUE;
    END IF;
    -- Read only the newest ledger entry for each call identity. Every actual
    -- failure must have an exact structured proof and no account ambiguity;
    -- successful charged calls are permitted and retain their existing spend.
    IF NOT EXISTS (
      SELECT 1 FROM (
        SELECT DISTINCT ON (ledger.call_identity) ledger.*
        FROM public.lab_arena_ledger AS ledger
        WHERE ledger.run_id = v_candidate.run_id
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head
      WHERE head.entry_kind = 'settlement'
        AND head.funding_source = 'miner_key'
        AND head.miner_hotkey = p_miner_hotkey
        AND head.provider IN ('openrouter','scrapingdog','deepline')
        AND head.amount_microusd = 0
        AND head.terminal_response -> 'call_succeeded' = 'false'::JSONB
        AND head.terminal_response -> 'credit_failure_proof' =
          pg_catalog.jsonb_build_object(
            'schema_version','leadpoet.lab_arena.credit_failure_proof.v1',
            'provider',head.provider,'reason','out_of_credit',
            'provider_status',402,'actual_microusd',0)
    ) OR EXISTS (
      SELECT 1 FROM public.lab_arena_ledger AS unbound
      WHERE unbound.run_id = v_candidate.run_id
        AND unbound.call_identity IS NULL
    ) OR EXISTS (
      SELECT 1 FROM (
        SELECT DISTINCT ON (ledger.call_identity) ledger.*
        FROM public.lab_arena_ledger AS ledger
        WHERE ledger.run_id = v_candidate.run_id
          AND ledger.call_identity IS NOT NULL
        ORDER BY ledger.call_identity, ledger.entry_id DESC
      ) AS head
      WHERE head.round_id IS DISTINCT FROM v_candidate.round_id
        OR head.submission_id IS DISTINCT FROM v_candidate.submission_id
        OR head.stage IS DISTINCT FROM v_candidate.stage
        OR head.entry_kind IS DISTINCT FROM 'settlement'
        OR head.entry_kind = 'settlement' AND (
          head.miner_hotkey IS DISTINCT FROM p_miner_hotkey
          OR head.funding_source IS DISTINCT FROM 'miner_key'
          OR head.provider NOT IN ('openrouter','scrapingdog','deepline')
          OR (head.terminal_response #>>
               '{account_failure_evidence,provider_status}')
               IN ('401','403','429')
          OR head.terminal_response -> 'account_failure_evidence' IS NOT NULL
            AND head.terminal_response -> 'credit_failure_proof' IS NULL
          OR (
            head.terminal_response -> 'call_succeeded' = 'true'::JSONB
            AND head.terminal_response ->> 'status' ~ '^2[0-9]{2}$'
            AND head.terminal_response -> 'credit_failure_proof' IS NULL
            OR head.amount_microusd = 0
            AND head.terminal_response -> 'call_succeeded' = 'false'::JSONB
            AND head.terminal_response -> 'credit_failure_proof' =
              pg_catalog.jsonb_build_object(
                'schema_version','leadpoet.lab_arena.credit_failure_proof.v1',
                'provider',head.provider,'reason','out_of_credit',
                'provider_status',402,'actual_microusd',0)
          ) IS NOT TRUE
        )
    ) THEN
      CONTINUE;
    END IF;

    v_group_leader := COALESCE(v_candidate.judgment_group_leader,FALSE);
    IF v_kind = 'score' AND v_candidate.judgment_cache_key IS NOT NULL THEN
      PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(
          'lab_arena.judgment-group:' || v_candidate.round_id || ':' ||
          v_candidate.stage::TEXT || ':' ||
          v_candidate.stage_generation::TEXT || ':' ||
          v_candidate.judgment_cache_key,0));
      IF EXISTS (SELECT 1 FROM public.lab_arena_judgment_cache AS cache
                 WHERE cache.cache_key=v_candidate.judgment_cache_key) THEN
        CONTINUE;
      END IF;
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS leader
        WHERE leader.round_id=v_candidate.round_id
          AND leader.stage=v_candidate.stage
          AND leader.stage_generation=v_candidate.stage_generation
          AND leader.kind='score'
          AND leader.judgment_cache_key=v_candidate.judgment_cache_key
          AND COALESCE(leader.judgment_group_leader,FALSE)
          AND leader.status IN ('pending','leased','submitted')
      ) THEN
        -- Pending nonleaders consume a later accepted cache entry, or the
        -- normal terminal handoff promotes one after the current leader ends.
        v_group_leader := FALSE;
      ELSE
        v_group_leader := TRUE;
      END IF;
    END IF;

    INSERT INTO public.lab_arena_runs (
      run_id,assignment_id,round_id,submission_id,miner_hotkey,stage,
      icp_position,attempt,kind,scored_run_id,previous_runner_hotkey,
      status,lease_generation,stage_generation,
      judgment_cache_key,judgment_input_hash,judgment_scope_doc,
      judgment_group_leader,judgment_group_miner_hotkeys,
      company_judgment_refs,
      credit_retry_parent_run_id,credit_retry_request_hash
    ) VALUES (
      v_candidate.assignment_id || ':2',v_candidate.assignment_id,
      v_candidate.round_id,v_candidate.submission_id,v_candidate.miner_hotkey,
      v_candidate.stage,v_candidate.icp_position,2,v_candidate.kind,
      v_candidate.scored_run_id,v_candidate.runner_hotkey,'pending',
      v_candidate.lease_generation,v_candidate.stage_generation,
      v_candidate.judgment_cache_key,v_candidate.judgment_input_hash,
      v_candidate.judgment_scope_doc,v_group_leader,
      v_candidate.judgment_group_miner_hotkeys,
      v_candidate.company_judgment_refs,
      v_candidate.run_id,p_request_hash
    );
    v_created := v_created + 1;
  END LOOP;
  IF v_created = 0 THEN
    RETURN pg_catalog.jsonb_build_object(
      'status','no_eligible','requeued_count',0,'reason','no_proved_failures');
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status','queued','requeued_count',v_created);
END;
$lab_arena_retry_credit_failures$;

ALTER FUNCTION public.lab_arena_retry_credit_failures_v1(TEXT,TEXT,TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_retry_credit_failures_v1(TEXT,TEXT,TEXT,TEXT)
  FROM PUBLIC;
DO $credit_retry_acl$
DECLARE role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon','authenticated','service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname=role_name) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_retry_credit_failures_v1(TEXT,TEXT,TEXT,TEXT) FROM %I',
        role_name);
    END IF;
  END LOOP;
END;
$credit_retry_acl$;
GRANT EXECUTE ON FUNCTION public.lab_arena_retry_credit_failures_v1(TEXT,TEXT,TEXT,TEXT)
  TO lab_arena_service;
COMMIT;

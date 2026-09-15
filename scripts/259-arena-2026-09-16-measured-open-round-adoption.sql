-- Replace the installed Sep16 adoption RPC with measured physical-capacity
-- and completed-admission-ledger guards. This migration does not mutate a round.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_257$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_adopt_sep16_open_config_v1(jsonb)'
     ) IS NULL
     OR pg_catalog.to_regclass(
       'public.lab_arena_sep16_open_config_audit'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply exact 257 before 259';
  END IF;
END;
$requires_257$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_adopt_sep16_open_config_v1(
  p_capacity_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $adopt_sep16_open_config$
DECLARE
  v_round public.lab_arena_rounds%ROWTYPE;
  v_audit public.lab_arena_sep16_open_config_audit%ROWTYPE;
  v_new_config JSONB;
  v_original_schedule JSONB;
  v_forward_schedule JSONB;
  v_concurrent_submissions BIGINT;
  v_accepted_submissions BIGINT;
  v_ledger_rows BIGINT;
  v_ledger_submissions BIGINT;
  v_slots INTEGER;
  v_verified_slots INTEGER;
  v_attempts INTEGER;
  v_scoring_wave_seconds INTEGER;
  v_supported INTEGER;
BEGIN
  IF pg_catalog.jsonb_typeof(p_capacity_doc) IS DISTINCT FROM 'object'
     OR p_capacity_doc ->> 'round_id' IS DISTINCT FROM 'arena-2026-09-16'
     OR p_capacity_doc -> 'parallel_twenty_icp_execution'
          IS DISTINCT FROM 'true'::JSONB
     OR p_capacity_doc ->> 'checkpoint_deadline_policy'
          IS DISTINCT FROM 'atomic_checkpoint_45m_v1'
     OR (p_capacity_doc ->> 'icp_wall_clock_seconds')::INTEGER
          IS DISTINCT FROM 2700
     OR (p_capacity_doc ->> 'runner_slot_ceiling')::INTEGER
          IS DISTINCT FROM 20
     OR COALESCE(
          (p_capacity_doc ->> 'verified_parallel_runner_slots')::INTEGER, 0
        ) NOT BETWEEN 1 AND 20
     OR COALESCE(
          (p_capacity_doc ->> 'configured_challenger_capacity')::INTEGER, 0
        ) < 1
     OR COALESCE(
          (p_capacity_doc ->> 'already_accepted_challengers')::INTEGER, -1
        ) < 0
     OR pg_catalog.jsonb_typeof(p_capacity_doc -> 'schedule')
          IS DISTINCT FROM 'object'
     OR COALESCE(p_capacity_doc ->> 'validated_by', '') <> 'arena_service_capacity_v1' THEN
    RAISE EXCEPTION 'sep16 capacity proof invalid' USING ERRCODE = '22023';
  END IF;
  LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
  LOCK TABLE public.lab_arena_submissions IN SHARE MODE;
  LOCK TABLE public.lab_arena_runs IN SHARE MODE;
  LOCK TABLE public.lab_arena_ledger IN SHARE MODE;
  SELECT * INTO v_round FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-16' FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'sep16 round missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_audit FROM public.lab_arena_sep16_open_config_audit
  WHERE round_id = v_round.round_id FOR UPDATE;
  IF FOUND THEN
    IF v_round.status <> 'open'
       OR v_round.status_generation <> 0
       OR v_round.stage_generation <> 0
       OR v_round.participants IS NOT NULL
       OR v_round.benchmark_ref IS NOT NULL
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                  WHERE round_id = v_round.round_id)
       OR v_round.configuration_doc IS DISTINCT FROM v_audit.new_configuration_doc
       OR p_capacity_doc IS DISTINCT FROM v_audit.capacity_doc THEN
      RAISE EXCEPTION 'sep16 adoption replay differs' USING ERRCODE = '55000';
    END IF;
    RETURN pg_catalog.jsonb_build_object('status', 'existing',
      'round_id', v_round.round_id);
  END IF;
  IF v_round.status <> 'open'
     OR v_round.status_generation <> 0
     OR v_round.stage_generation <> 0
     OR v_round.configuration_doc ->> 'mode' <> 'live'
     OR NOT v_round.rewards_enabled
     OR v_round.configuration_doc ->> 'round_id' <> v_round.round_id
     OR v_round.configuration_doc ->> 'integrity_policy' <> 'arena_integrity_v1'
     OR v_round.configuration_doc ->> 'contact_policy' <> 'contacts_v1'
     OR v_round.configuration_doc ->> 'intent_details_policy' <> 'intent_details_v1'
     OR v_round.configuration_doc ? 'parallel_twenty_icp_execution'
     OR v_round.configuration_doc ? 'checkpoint_deadline_policy'
     OR (v_round.configuration_doc ->> 'icp_wall_clock_seconds')::INTEGER <> 300
     OR (v_round.configuration_doc ->> 'lease_ttl_seconds')::INTEGER <> 1200
     OR (v_round.configuration_doc ->> 'runner_slot_ceiling')::INTEGER <> 8
     OR (v_round.configuration_doc ->> 'max_challengers')::INTEGER <> 20
     OR v_round.participants IS NOT NULL
     OR v_round.benchmark_ref IS NOT NULL
     OR v_round.evaluation_date IS NOT NULL
     OR v_round.icp_set_date IS NOT NULL
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.reward_basis_doc IS NOT NULL
     OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                WHERE round_id = v_round.round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE round_id = v_round.round_id AND is_king)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE round_id = v_round.round_id AND status = 'frozen') THEN
    RAISE EXCEPTION 'sep16 open-only state differs' USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(*),
         pg_catalog.count(*) FILTER (
           WHERE status = 'accepted' AND NOT is_king
         )
  INTO v_concurrent_submissions, v_accepted_submissions
  FROM public.lab_arena_submissions WHERE round_id = v_round.round_id;
  IF (p_capacity_doc ->> 'already_accepted_challengers')::BIGINT
       IS DISTINCT FROM v_accepted_submissions THEN
    RAISE EXCEPTION 'sep16 accepted submissions changed since capacity proof'
      USING ERRCODE = '55000';
  END IF;
  IF v_accepted_submissions >
       (p_capacity_doc ->> 'configured_challenger_capacity')::INTEGER THEN
    RAISE EXCEPTION 'sep16 accepted workload exceeds measured capacity proof'
      USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(*), pg_catalog.count(DISTINCT submission_id)
  INTO v_ledger_rows, v_ledger_submissions
  FROM public.lab_arena_ledger WHERE round_id = v_round.round_id;
  -- These are completed, one-call code-review triples attached to accepted
  -- admissions. They are immutable history, not execution or scoring ledger.
  IF v_ledger_rows <> 3 * v_accepted_submissions
     OR v_ledger_submissions <> v_accepted_submissions
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS ledger
       LEFT JOIN public.lab_arena_submissions AS submission
         ON submission.submission_id = ledger.submission_id
        AND submission.round_id = ledger.round_id
       WHERE ledger.round_id = v_round.round_id
         AND (ledger.call_identity IS NULL
              OR ledger.run_id IS NOT NULL OR ledger.stage IS NOT NULL
              OR ledger.provider IS DISTINCT FROM 'openrouter'
              OR ledger.operation_id IS DISTINCT FROM 'openrouter.code_review'
              OR submission.status IS DISTINCT FROM 'accepted'
              OR submission.is_king IS DISTINCT FROM FALSE)
     )
     OR EXISTS (
       SELECT 1 FROM public.lab_arena_ledger AS ledger
       WHERE ledger.round_id = v_round.round_id
       GROUP BY ledger.submission_id, ledger.call_identity
       HAVING pg_catalog.count(*) <> 3
          OR pg_catalog.count(*) FILTER (
               WHERE entry_kind = 'reservation'
             ) <> 1
          OR pg_catalog.count(*) FILTER (
               WHERE entry_kind = 'dispatch'
             ) <> 1
          OR pg_catalog.count(*) FILTER (
               WHERE entry_kind = 'settlement'
                 AND entry_doc ->> 'review_status' = 'passed'
             ) <> 1
     ) THEN
    RAISE EXCEPTION 'sep16 admission ledger differs from completed triples'
      USING ERRCODE = '55000';
  END IF;
  v_original_schedule := v_round.configuration_doc -> 'schedule';
  v_forward_schedule := p_capacity_doc -> 'schedule';
  IF pg_catalog.jsonb_typeof(v_original_schedule) IS DISTINCT FROM 'object'
     OR (SELECT pg_catalog.array_agg(key ORDER BY key)
         FROM pg_catalog.jsonb_object_keys(v_original_schedule) AS keys(key))
        IS DISTINCT FROM
        (SELECT pg_catalog.array_agg(key ORDER BY key)
         FROM pg_catalog.jsonb_object_keys(v_forward_schedule) AS keys(key))
     OR v_forward_schedule ->> 'submission_open'
          IS DISTINCT FROM v_original_schedule ->> 'submission_open'
     OR v_forward_schedule ->> 'submission_cutoff'
          IS DISTINCT FROM v_original_schedule ->> 'submission_cutoff'
     OR v_forward_schedule ->> 'benchmark_deadline'
          IS DISTINCT FROM v_original_schedule ->> 'benchmark_deadline'
     OR v_forward_schedule ->> 'stage_1_start'
          IS DISTINCT FROM v_original_schedule ->> 'stage_1_start' THEN
    RAISE EXCEPTION 'sep16 forward schedule changes immutable intake or start'
      USING ERRCODE = '22023';
  END IF;
  v_new_config := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'schedule', v_forward_schedule,
    'icp_wall_clock_seconds', 2700,
    'lease_ttl_seconds', 3600,
    'runner_slot_ceiling', 20,
    'parallel_twenty_icp_execution', TRUE,
    'checkpoint_deadline_policy', 'atomic_checkpoint_45m_v1'
  );
  v_verified_slots := (p_capacity_doc ->> 'verified_parallel_runner_slots')::INTEGER;
  SELECT pg_catalog.count(DISTINCT hotkey)::INTEGER INTO v_slots
  FROM pg_catalog.jsonb_array_elements_text(
    v_new_config -> 'runner_hotkeys'
  ) AS hotkey;
  -- Sep16 has exactly one frozen runner. Model its verified physical slots,
  -- while retaining the public 20-slot ceiling for dynamic proxy capacity.
  IF v_slots <> 1 OR pg_catalog.jsonb_array_length(
       v_new_config -> 'runner_hotkeys'
     ) <> 1 THEN
    RAISE EXCEPTION 'sep16 frozen runner set differs' USING ERRCODE = '55000';
  END IF;
  v_slots := v_verified_slots;
  v_attempts := (v_new_config ->> 'max_attempts_per_assignment')::INTEGER;
  v_scoring_wave_seconds :=
    (v_new_config ->> 'scoring_wall_clock_seconds')::INTEGER + 60;
  IF v_slots NOT BETWEEN 1 AND 20 OR v_attempts <> 2
     OR v_scoring_wave_seconds <= 60
     OR NOT COALESCE(
       (v_new_config #>> '{schedule,submission_open}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,benchmark_deadline}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,benchmark_deadline}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,stage_1_start}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,stage_1_start}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,stage_2_start}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,stage_2_start}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,final_scoring_close}')::TIMESTAMPTZ
       AND (v_new_config #>> '{schedule,final_scoring_close}')::TIMESTAMPTZ
         < (v_new_config #>> '{schedule,publication_deadline}')::TIMESTAMPTZ,
       FALSE
     )
     OR EXTRACT(EPOCH FROM (
       (v_new_config #>> '{schedule,publication_deadline}')::TIMESTAMPTZ
       - (v_new_config #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
     )) >= 86400 THEN
    RAISE EXCEPTION 'sep16 capacity schedule invalid' USING ERRCODE = '55000';
  END IF;
  -- Parallel execution completes all twenty positions before stage-one
  -- scoring. Count the already-accepted miners plus the future baseline,
  -- including their full retry reserve; rejected rows consume no slots.
  IF EXTRACT(EPOCH FROM (
       (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
       - GREATEST(
           (v_new_config #>> '{schedule,stage_1_start}')::TIMESTAMPTZ,
           pg_catalog.clock_timestamp()
         )
     )) < v_attempts * pg_catalog.ceil(
       20.0 * (v_accepted_submissions + 1) / v_slots
     ) * 2760 THEN
    RAISE EXCEPTION 'sep16 accepted parallel workload exceeds stage1 window'
      USING ERRCODE = '55000';
  END IF;
  IF EXTRACT(EPOCH FROM (
       (v_new_config #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
       - (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
     )) < v_attempts * pg_catalog.ceil(
       10.0 * (v_accepted_submissions + 1) / v_slots
     ) * v_scoring_wave_seconds
     OR EXTRACT(EPOCH FROM (
       (v_new_config #>> '{schedule,final_scoring_close}')::TIMESTAMPTZ
       - (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
     )) < v_attempts * pg_catalog.ceil(
       10.0 * (v_accepted_submissions + 1) / v_slots
     ) * v_scoring_wave_seconds THEN
    RAISE EXCEPTION 'sep16 accepted scoring workload exceeds phase window'
      USING ERRCODE = '55000';
  END IF;
  v_supported := GREATEST(0, LEAST(
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_1_start}')::TIMESTAMPTZ
    )) / 2760)::INTEGER * v_slots) / (v_attempts * 20) - 1,
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
    )) / v_scoring_wave_seconds)::INTEGER * v_slots) / (v_attempts * 10) - 1,
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,final_scoring_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
    )) / v_scoring_wave_seconds)::INTEGER * v_slots) / (v_attempts * 10) - 1
  ));
  IF v_supported < 1
     OR v_accepted_submissions > v_supported
     OR (p_capacity_doc ->> 'configured_challenger_capacity')::INTEGER
          IS DISTINCT FROM v_supported THEN
    RAISE EXCEPTION 'sep16 capacity proof differs from frozen schedule'
      USING ERRCODE = '55000';
  END IF;
  -- The operator's capacity result must bind the unchanged runner set and
  -- schedule as well as the exact proposed configuration.
  IF p_capacity_doc -> 'runner_hotkeys' IS DISTINCT FROM
       v_new_config -> 'runner_hotkeys'
     OR p_capacity_doc -> 'schedule' IS DISTINCT FROM
       v_new_config -> 'schedule'
     OR p_capacity_doc -> 'configuration_doc' IS DISTINCT FROM
       v_new_config THEN
    RAISE EXCEPTION 'sep16 capacity proof config differs' USING ERRCODE = '22023';
  END IF;
  INSERT INTO public.lab_arena_sep16_open_config_audit (
    round_id, old_round_doc, new_configuration_doc, capacity_doc
  ) VALUES (
    v_round.round_id, pg_catalog.to_jsonb(v_round), v_new_config, p_capacity_doc
  );
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = v_new_config,
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round.round_id;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
  IF (SELECT pg_catalog.count(*) FROM public.lab_arena_submissions
      WHERE round_id = v_round.round_id) <> v_concurrent_submissions
     OR (SELECT pg_catalog.count(*) FROM public.lab_arena_ledger
         WHERE round_id = v_round.round_id) <> v_ledger_rows
     OR (SELECT configuration_doc FROM public.lab_arena_rounds
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_new_config THEN
    RAISE EXCEPTION 'sep16 adoption verification differs' USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status', 'adopted',
    'round_id', v_round.round_id, 'submissions_preserved', v_concurrent_submissions,
    'admission_ledger_rows_preserved', v_ledger_rows,
    'already_accepted_challengers', v_accepted_submissions,
    'verified_parallel_runner_slots', v_verified_slots,
    'future_admissions_may_exceed_measured_capacity',
      (v_new_config ->> 'max_challengers')::INTEGER > v_supported,
    'configured_challenger_capacity',
    p_capacity_doc -> 'configured_challenger_capacity');
END;
$adopt_sep16_open_config$;
ALTER FUNCTION public.lab_arena_adopt_sep16_open_config_v1(JSONB)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_adopt_sep16_open_config_v1(JSONB)
  FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.lab_arena_adopt_sep16_open_config_v1(JSONB)
  TO lab_arena_service;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

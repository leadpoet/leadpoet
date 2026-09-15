-- Adopt the already-open Sep16 round to the tested 45-minute execution policy.
-- Applying this migration installs the exact guarded RPC; it does not mutate
-- the round. The operator validates worker capacity and invokes the RPC.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

DO $requires_256$
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_prepare_sep15_baseline_rerun_v1(bigint,text,text,text,jsonb)'
     ) IS NULL
     OR pg_catalog.to_regprocedure(
       'public.lab_arena_parallel_execution_schema_v1()'
     ) IS NULL THEN
    RAISE EXCEPTION 'apply current Arena and 256 before 257';
  END IF;
END;
$requires_256$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE TABLE IF NOT EXISTS public.lab_arena_sep16_open_config_audit (
  round_id TEXT PRIMARY KEY CHECK (round_id = 'arena-2026-09-16'),
  old_round_doc JSONB NOT NULL,
  new_configuration_doc JSONB NOT NULL,
  capacity_doc JSONB NOT NULL,
  adopted_at TIMESTAMPTZ NOT NULL DEFAULT pg_catalog.clock_timestamp()
);
ALTER TABLE public.lab_arena_sep16_open_config_audit OWNER TO lab_arena_owner;
REVOKE ALL ON public.lab_arena_sep16_open_config_audit FROM PUBLIC;
REVOKE ALL ON public.lab_arena_sep16_open_config_audit FROM lab_arena_service;

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
  v_concurrent_submissions BIGINT;
  v_slots INTEGER;
  v_attempts INTEGER;
  v_scoring_wave_seconds INTEGER;
  v_supported INTEGER;
BEGIN
  IF pg_catalog.jsonb_typeof(p_capacity_doc) IS DISTINCT FROM 'object'
     OR p_capacity_doc ->> 'round_id' <> 'arena-2026-09-16'
     OR p_capacity_doc ->> 'parallel_twenty_icp_execution' <> 'true'
     OR p_capacity_doc ->> 'checkpoint_deadline_policy' <> 'atomic_checkpoint_45m_v1'
     OR (p_capacity_doc ->> 'icp_wall_clock_seconds')::INTEGER <> 2700
     OR (p_capacity_doc ->> 'runner_slot_ceiling')::INTEGER <> 20
     OR (p_capacity_doc ->> 'configured_challenger_capacity')::INTEGER < 1
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
     OR EXISTS (SELECT 1 FROM public.lab_arena_ledger
                WHERE round_id = v_round.round_id)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE round_id = v_round.round_id AND is_king)
     OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                WHERE round_id = v_round.round_id AND status = 'frozen') THEN
    RAISE EXCEPTION 'sep16 open-only state differs' USING ERRCODE = '55000';
  END IF;
  SELECT pg_catalog.count(*) INTO v_concurrent_submissions
  FROM public.lab_arena_submissions WHERE round_id = v_round.round_id;
  v_new_config := v_round.configuration_doc || pg_catalog.jsonb_build_object(
    'icp_wall_clock_seconds', 2700,
    'lease_ttl_seconds', 3600,
    'runner_slot_ceiling', 20,
    'parallel_twenty_icp_execution', TRUE,
    'checkpoint_deadline_policy', 'atomic_checkpoint_45m_v1'
  );
  SELECT 20 * pg_catalog.count(DISTINCT hotkey)::INTEGER INTO v_slots
  FROM pg_catalog.jsonb_array_elements_text(
    v_new_config -> 'runner_hotkeys'
  ) AS hotkey;
  v_attempts := (v_new_config ->> 'max_attempts_per_assignment')::INTEGER;
  v_scoring_wave_seconds :=
    (v_new_config ->> 'scoring_wall_clock_seconds')::INTEGER + 60;
  IF v_slots < 20 OR v_attempts NOT BETWEEN 1 AND 2
     OR v_scoring_wave_seconds <= 60
     OR EXTRACT(EPOCH FROM (
       (v_new_config #>> '{schedule,publication_deadline}')::TIMESTAMPTZ
       - (v_new_config #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
     )) >= 86400 THEN
    RAISE EXCEPTION 'sep16 capacity schedule invalid' USING ERRCODE = '55000';
  END IF;
  v_supported := GREATEST(0, LEAST(
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_1_start}')::TIMESTAMPTZ
    )) / 2760)::INTEGER * v_slots) / (v_attempts * 10) - 1,
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,stage_1_scoring_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_1_close}')::TIMESTAMPTZ
    )) / v_scoring_wave_seconds)::INTEGER * v_slots) / (v_attempts * 10) - 1,
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_2_start}')::TIMESTAMPTZ
    )) / 2760)::INTEGER * v_slots) / (v_attempts * 10) - 1,
    (pg_catalog.floor(EXTRACT(EPOCH FROM (
      (v_new_config #>> '{schedule,final_scoring_close}')::TIMESTAMPTZ
      - (v_new_config #>> '{schedule,stage_2_close}')::TIMESTAMPTZ
    )) / v_scoring_wave_seconds)::INTEGER * v_slots) / (v_attempts * 10) - 1
  ));
  IF v_supported < 1
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
     OR (SELECT configuration_doc FROM public.lab_arena_rounds
         WHERE round_id = v_round.round_id) IS DISTINCT FROM v_new_config THEN
    RAISE EXCEPTION 'sep16 adoption verification differs' USING ERRCODE = '55000';
  END IF;
  RETURN pg_catalog.jsonb_build_object('status', 'adopted',
    'round_id', v_round.round_id, 'submissions_preserved', v_concurrent_submissions,
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

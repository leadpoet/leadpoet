-- One-time transition for the open 2026-09-11 Arena round.
--
-- The round configuration remains immutable in normal operation.  This data
-- migration appends the primary normal validator while the round is still
-- open and before any scoring plan or result exists.  Fresh installations do
-- not contain this dated round and therefore only install the capability
-- marker.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $requires_203$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_incentive_retirement_schema_v1()') IS NULL
     OR (public.lab_arena_incentive_retirement_schema_v1() ->> 'version')::INTEGER <> 203 THEN
    RAISE EXCEPTION 'apply 203-retire-legacy-incentive-weight-bridge.sql first';
  END IF;
END;
$requires_203$;

LOCK TABLE public.lab_arena_rounds IN SHARE ROW EXCLUSIVE MODE;

DO $append_primary_runner$
DECLARE
  v_round public.lab_arena_rounds%ROWTYPE;
  v_pre_hash CONSTANT TEXT := '0001ab8b055e176aff4d9e98ac6a9221f91328fe89208369afad38018614b8af';
  v_old_runner CONSTANT TEXT := '5GsGcRyR4kWCcsa1qEAwxtbDq34ZwkQt3rHAGniPFjv1JoXW';
  v_primary_runner CONSTANT TEXT := '5FNVgRnrxMibhcBGEAaajGrYjsaCn441a5HuGUBUNnxEBLo9';
  v_current_hash TEXT;
  v_reconstructed_hash TEXT;
BEGIN
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = 'arena-2026-09-11'
  FOR UPDATE;

  -- A fresh database has no dated production row.  Applying the migration is
  -- still valid and records that this transition was considered.
  IF NOT FOUND THEN
    RETURN;
  END IF;

  v_current_hash := pg_catalog.encode(
    extensions.digest(
      pg_catalog.convert_to(v_round.configuration_doc::TEXT, 'UTF8'),
      'sha256'
    ),
    'hex'
  );

  -- Exact idempotence: reconstruct the original document from the only
  -- accepted post-state and require its database-native digest to match the
  -- observed production document.
  IF v_round.configuration_doc -> 'runner_hotkeys' =
       pg_catalog.jsonb_build_array(v_old_runner, v_primary_runner) THEN
    v_reconstructed_hash := pg_catalog.encode(
      extensions.digest(
        pg_catalog.convert_to(
          pg_catalog.jsonb_set(
            v_round.configuration_doc,
            '{runner_hotkeys}',
            pg_catalog.jsonb_build_array(v_old_runner),
            FALSE
          )::TEXT,
          'UTF8'
        ),
        'sha256'
      ),
      'hex'
    );
    IF v_reconstructed_hash <> v_pre_hash THEN
      RAISE EXCEPTION 'arena primary runner transition post-state differs';
    END IF;
    RETURN;
  END IF;

  IF v_current_hash <> v_pre_hash THEN
    RAISE EXCEPTION 'arena primary runner transition configuration differs';
  END IF;
  IF v_round.status <> 'open'
     OR v_round.stage1_scoring_plan_doc IS NOT NULL
     OR v_round.stage2_scoring_plan_doc IS NOT NULL
     OR v_round.finalists IS NOT NULL
     OR v_round.publication_doc IS NOT NULL
     OR v_round.king_outcome IS NOT NULL
     OR v_round.published_at IS NOT NULL THEN
    RAISE EXCEPTION 'arena primary runner transition round has started';
  END IF;
  IF v_round.configuration_doc -> 'runner_hotkeys' <>
       pg_catalog.jsonb_build_array(v_old_runner) THEN
    RAISE EXCEPTION 'arena primary runner transition runner set differs';
  END IF;
  IF v_round.configuration_doc -> 'banned_hotkeys' ? v_primary_runner THEN
    RAISE EXCEPTION 'arena primary runner transition runner is banned';
  END IF;
  IF COALESCE(v_round.configuration_doc #>> '{schedule,submission_cutoff}', '') = ''
     OR (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
          <= pg_catalog.clock_timestamp() THEN
    RAISE EXCEPTION 'arena primary runner transition submission cutoff passed';
  END IF;

  -- Disable only the named immutable-row trigger, for this locked transaction
  -- and this single-field data update.  Any failure rolls the ALTER and UPDATE
  -- back together.
  ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;
  UPDATE public.lab_arena_rounds
  SET configuration_doc = pg_catalog.jsonb_set(
        configuration_doc,
        '{runner_hotkeys}',
        pg_catalog.jsonb_build_array(v_old_runner, v_primary_runner),
        FALSE
      ),
      updated_at = pg_catalog.clock_timestamp()
  WHERE round_id = v_round.round_id;
  ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;
END;
$append_primary_runner$;

COMMIT;

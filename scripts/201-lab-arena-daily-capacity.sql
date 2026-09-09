-- Move only unstarted daily rounds off the old top-ten capacity schedule.
-- No scores, submissions, disclosure dates, or reward state are changed.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- The table lock serializes this one-time configuration repair with cutoff
-- and source registration. The write-once trigger is restored in this same
-- transaction; any error rolls back both the data and trigger state.
LOCK TABLE public.lab_arena_rounds IN ACCESS EXCLUSIVE MODE;
ALTER TABLE public.lab_arena_rounds DISABLE TRIGGER lab_arena_rounds_write_once;

DO $daily_capacity$
DECLARE
  r public.lab_arena_rounds;
  c JSONB;
  s JSONB;
  cutoff TIMESTAMPTZ;
  retained INTEGER;
BEGIN
  FOR r IN SELECT * FROM public.lab_arena_rounds
    WHERE status = 'open' AND benchmark_ref IS NULL
      AND configuration_doc ->> 'mode' = 'live'
      AND COALESCE(configuration_doc ->> 'network_name', 'finney') = 'finney'
      AND COALESCE((configuration_doc ->> 'netuid')::INTEGER, 71) = 71
      AND configuration_doc ->> 'runner_slot_ceiling' = '8'
      AND configuration_doc ->> 'max_challengers' = '16'
      AND jsonb_array_length(configuration_doc -> 'runner_hotkeys') = 1
      AND (configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ > clock_timestamp()
  LOOP
    c := r.configuration_doc;
    s := c -> 'schedule';
    cutoff := (s ->> 'submission_cutoff')::TIMESTAMPTZ;
    -- Exact old native schedule only. Reapplication and operator schedules
    -- remain untouched; committed/running/historical rounds are excluded.
    IF (s ->> 'submission_open')::TIMESTAMPTZ IS DISTINCT FROM cutoff - INTERVAL '1 day'
       OR (s ->> 'benchmark_deadline')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '30 minutes'
       OR (s ->> 'stage_1_start')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '30 minutes 1 second'
       OR (s ->> 'stage_1_close')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '4 hours 30 minutes 1 second'
       OR (s ->> 'stage_1_scoring_close')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '10 hours 30 minutes 1 second'
       OR (s ->> 'stage_2_start')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '10 hours 30 minutes 2 seconds'
       OR (s ->> 'stage_2_close')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '13 hours 30 minutes 2 seconds'
       OR (s ->> 'final_scoring_close')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '17 hours 30 minutes 2 seconds'
       OR (s ->> 'publication_deadline')::TIMESTAMPTZ IS DISTINCT FROM cutoff + INTERVAL '17 hours 30 minutes 3 seconds' THEN
      CONTINUE;
    END IF;
    SELECT COUNT(*) INTO retained FROM public.lab_arena_submissions
      WHERE round_id = r.round_id AND NOT is_king
        AND status IN ('uploading', 'accepted', 'frozen');
    IF retained > 8 OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id = r.round_id) THEN
      RAISE EXCEPTION 'daily_capacity_repair_requires_preserving_all_submissions';
    END IF;
    s := s || jsonb_build_object(
      'stage_1_scoring_close', to_char(cutoff AT TIME ZONE 'UTC' + INTERVAL '11 hours 1 second', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
      'stage_2_start', to_char(cutoff AT TIME ZONE 'UTC' + INTERVAL '11 hours 2 seconds', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
      'stage_2_close', to_char(cutoff AT TIME ZONE 'UTC' + INTERVAL '14 hours 2 seconds', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
      'final_scoring_close', to_char(cutoff AT TIME ZONE 'UTC' + INTERVAL '20 hours 30 minutes 2 seconds', 'YYYY-MM-DD"T"HH24:MI:SS"Z"'),
      'publication_deadline', to_char(cutoff AT TIME ZONE 'UTC' + INTERVAL '20 hours 30 minutes 3 seconds', 'YYYY-MM-DD"T"HH24:MI:SS"Z"')
    );
    UPDATE public.lab_arena_rounds
      SET configuration_doc = c || jsonb_build_object(
        'schedule', s,
        'max_challengers', 8
      )
      WHERE round_id = r.round_id;
  END LOOP;
END;
$daily_capacity$;

ALTER TABLE public.lab_arena_rounds ENABLE TRIGGER lab_arena_rounds_write_once;

-- Final admission, not an end-of-day eviction. Serializing this small check
-- prevents concurrent finalized uploads from exceeding the announced cap.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
CREATE OR REPLACE FUNCTION public.lab_arena_submission_capacity_v1()
RETURNS TRIGGER LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $submission_capacity$
DECLARE
  c JSONB;
  accepted INTEGER;
BEGIN
  IF NEW.status <> 'accepted' OR (TG_OP = 'UPDATE' AND OLD.status IN ('accepted', 'frozen')) THEN
    RETURN NEW;
  END IF;
  SELECT configuration_doc INTO c FROM public.lab_arena_rounds WHERE round_id = NEW.round_id;
  IF NEW.is_king AND NEW.miner_hotkey = c ->> 'baseline_hotkey'
     AND NEW.submission_id = 'baseline-' || regexp_replace(NEW.round_id, '^arena-', '') THEN
    RETURN NEW;
  END IF;
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.capacity.' || NEW.round_id, 0));
  SELECT COUNT(*) INTO accepted FROM public.lab_arena_submissions
    WHERE round_id = NEW.round_id AND NOT is_king
      AND submission_id <> NEW.submission_id AND status IN ('accepted', 'frozen');
  IF accepted >= COALESCE((c ->> 'max_challengers')::INTEGER, 256) THEN
    RAISE EXCEPTION 'lab_arena_round_full' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$submission_capacity$;
ALTER FUNCTION public.lab_arena_submission_capacity_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_capacity_v1() FROM PUBLIC;
DROP TRIGGER IF EXISTS lab_arena_submission_capacity ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_submission_capacity
  BEFORE INSERT OR UPDATE OF status ON public.lab_arena_submissions
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_submission_capacity_v1();
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

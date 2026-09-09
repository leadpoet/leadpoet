-- A database-owned submission time for the 24-hour public source delay.
-- No source identity, release manifest, or new verification protocol.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS accepted_at TIMESTAMPTZ;

CREATE OR REPLACE FUNCTION public.lab_arena_stamp_submission_acceptance()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $stamp$
BEGIN
  IF TG_OP = 'UPDATE' AND OLD.accepted_at IS NOT NULL THEN
    NEW.accepted_at := OLD.accepted_at;
  ELSIF TG_OP = 'UPDATE' AND OLD.status = 'accepted' THEN
    NEW.accepted_at := OLD.updated_at;
  ELSIF TG_OP = 'UPDATE' AND OLD.status = 'frozen' THEN
    NEW.accepted_at := OLD.accepted_at;
  ELSIF NEW.status IN ('accepted', 'frozen') THEN
    NEW.accepted_at := pg_catalog.clock_timestamp();
  ELSE
    NEW.accepted_at := NULL;
  END IF;
  RETURN NEW;
END;
$stamp$;
ALTER FUNCTION public.lab_arena_stamp_submission_acceptance() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_stamp_submission_acceptance() FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_submission_acceptance_time ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_submission_acceptance_time
BEFORE INSERT OR UPDATE ON public.lab_arena_submissions
FOR EACH ROW EXECUTE FUNCTION public.lab_arena_stamp_submission_acceptance();

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

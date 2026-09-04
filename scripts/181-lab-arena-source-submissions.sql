-- 181-lab-arena-source-submissions.sql
-- Replace miner image intake with one bounded private source archive.

BEGIN;

-- Required only for ownership transfers by the hosted migration role.
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS source_ref TEXT,
  ADD COLUMN IF NOT EXISTS source_size_bytes BIGINT;

-- Image uploads from the retired intake cannot enter the source competition.
UPDATE public.lab_arena_submissions
SET status = 'rejected',
    rejection_rule = 'source_reupload_required',
    updated_at = pg_catalog.clock_timestamp()
WHERE status = 'uploaded';

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_status_check;
ALTER TABLE public.lab_arena_submissions
  ALTER COLUMN status SET DEFAULT 'uploading';
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_status_check
  CHECK (status IN ('uploading', 'accepted', 'rejected', 'frozen'));

ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_source_ref_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_source_ref_check
  CHECK (
    source_ref IS NULL
    OR source_ref ~ '^arena/arena-[0-9]{4}-[0-9]{2}-[0-9]{2}(-[a-z0-9]{1,16})?/sources/[A-Za-z0-9._:-]{1,64}\.tar\.gz$'
  );
ALTER TABLE public.lab_arena_submissions
  DROP CONSTRAINT IF EXISTS lab_arena_submissions_source_size_bytes_check;
ALTER TABLE public.lab_arena_submissions
  ADD CONSTRAINT lab_arena_submissions_source_size_bytes_check
  CHECK (
    source_size_bytes IS NULL
    OR source_size_bytes BETWEEN 1 AND 10485760
  );

DROP INDEX IF EXISTS public.lab_arena_submissions_one_accepted_per_miner_uq;
DROP INDEX IF EXISTS public.lab_arena_submissions_one_active_per_miner_uq;
CREATE UNIQUE INDEX lab_arena_submissions_one_active_per_miner_uq
  ON public.lab_arena_submissions (round_id, miner_hotkey)
  WHERE status IN ('uploading', 'accepted', 'frozen');

CREATE OR REPLACE FUNCTION public.lab_arena_register_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission$
DECLARE
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_submissions;
  v_collision public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_expected_ref TEXT;
BEGIN
  v_expected_ref := 'arena/' || p_round_id || '/sources/' || p_submission_id || '.tar.gz';
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR p_doc ->> 'source_ref' IS DISTINCT FROM v_expected_ref
     OR COALESCE((p_doc ->> 'source_size_bytes')::BIGINT, 0) NOT BETWEEN 1 AND 10485760
     OR COALESCE((p_doc #>> '{consent,public_rerun}')::BOOLEAN, FALSE) IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_submission_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_is_baseline := COALESCE((p_doc ->> 'is_king')::BOOLEAN, FALSE);

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open'
     OR (
       NOT v_is_baseline
       AND (
         COALESCE(pg_catalog.clock_timestamp() < (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ, TRUE)
         OR COALESCE(pg_catalog.clock_timestamp() >= (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ, TRUE)
       )
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed',
      'round_status', v_round.status
    );
  END IF;

  -- One active slot per miner and round. A retry with the same bounded upload
  -- size reuses that server-assigned submission.
  SELECT * INTO v_existing
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id
    AND miner_hotkey = p_miner_hotkey
    AND status IN ('uploading', 'accepted', 'frozen')
  ORDER BY created_at
  LIMIT 1;
  IF FOUND THEN
    IF v_existing.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing',
        'submission_status', v_existing.status,
        'submission_id', v_existing.submission_id,
        'source_ref', v_existing.source_ref
      );
    END IF;
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;

  SELECT * INTO v_collision
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id;
  IF FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;

  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king,
    source_ref, source_size_bytes, consent, submission_doc
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploading', v_is_baseline,
    p_doc ->> 'source_ref', (p_doc ->> 'source_size_bytes')::BIGINT,
    p_doc -> 'consent', p_doc
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'registered',
    'submission_status', 'uploading',
    'submission_id', p_submission_id,
    'source_ref', p_doc ->> 'source_ref'
  );
END;
$lab_arena_register_submission$;
ALTER FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_update_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_expected_status TEXT,
  p_next_status TEXT,
  p_patch JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_update_submission$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_patch JSONB := COALESCE(p_patch, '{}'::JSONB);
BEGIN
  IF pg_catalog.jsonb_typeof(v_patch) <> 'object' THEN
    RAISE EXCEPTION 'lab_arena_patch_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR SHARE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed', 'round_status', v_round.status
    );
  END IF;
  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.status <> p_expected_status THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'stale', 'submission_status', v_submission.status
    );
  END IF;

  IF p_expected_status = 'uploading' AND p_next_status = 'accepted' THEN
    IF v_submission.source_ref IS NULL
       OR v_submission.source_size_bytes IS NULL
       OR (v_patch - 'is_king') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'accepted',
        is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king),
        updated_at = pg_catalog.clock_timestamp()
    WHERE submission_id = p_submission_id;
  ELSIF p_expected_status IN ('uploading', 'accepted') AND p_next_status = 'rejected' THEN
    IF COALESCE(v_patch ->> 'rejection_rule', '') = ''
       OR (v_patch - 'rejection_rule') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'rejected',
        rejection_rule = v_patch ->> 'rejection_rule',
        updated_at = pg_catalog.clock_timestamp()
    WHERE submission_id = p_submission_id;
  ELSIF p_expected_status = 'accepted' AND p_next_status = 'frozen' THEN
    IF (v_patch - 'is_king') <> '{}'::JSONB THEN
      RAISE EXCEPTION 'lab_arena_patch_keys_invalid' USING ERRCODE = '22023';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'frozen',
        frozen_at = pg_catalog.clock_timestamp(),
        is_king = COALESCE((v_patch ->> 'is_king')::BOOLEAN, is_king),
        updated_at = pg_catalog.clock_timestamp()
    WHERE submission_id = p_submission_id;
  ELSE
    RAISE EXCEPTION 'lab_arena_transition_invalid' USING ERRCODE = '22023';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'submission_status', p_next_status
  );
END;
$lab_arena_update_submission$;
ALTER FUNCTION public.lab_arena_update_submission(TEXT, TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

NOTIFY pgrst, 'reload schema';

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

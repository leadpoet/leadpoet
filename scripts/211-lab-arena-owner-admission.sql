-- Bind each new integrity-policy challenger entry to one finalized coldkey.
-- Historical rounds and the server-owned baseline retain their existing shape.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS owner_coldkey TEXT;
ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS owner_block_number BIGINT;
ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS owner_block_hash TEXT;

DO $lab_arena_owner_admission_constraints$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_constraint
    WHERE conrelid = 'public.lab_arena_submissions'::pg_catalog.regclass
      AND conname = 'lab_arena_submissions_owner_admission_shape'
  ) THEN
    ALTER TABLE public.lab_arena_submissions
      ADD CONSTRAINT lab_arena_submissions_owner_admission_shape CHECK (
        (owner_coldkey IS NULL AND owner_block_number IS NULL AND owner_block_hash IS NULL)
        OR (
          owner_coldkey IS NOT NULL
          AND owner_block_number IS NOT NULL
          AND owner_block_hash IS NOT NULL
          AND owner_coldkey ~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
          AND owner_block_number >= 0
          AND owner_block_hash ~ '^0x[0-9a-f]{64}$'
        )
      );
  END IF;
END;
$lab_arena_owner_admission_constraints$;

CREATE UNIQUE INDEX IF NOT EXISTS lab_arena_submissions_one_active_owner_uq
  ON public.lab_arena_submissions (round_id, owner_coldkey)
  WHERE owner_coldkey IS NOT NULL
    AND is_king IS FALSE
    AND status IN ('uploading', 'accepted', 'frozen');

CREATE OR REPLACE FUNCTION public.lab_arena_owner_admission_guard_v1()
RETURNS TRIGGER
LANGUAGE plpgsql
SET search_path = pg_catalog, public
AS $lab_arena_owner_admission_guard$
DECLARE
  v_integrity BOOLEAN;
BEGIN
  IF TG_OP = 'UPDATE' AND (
    OLD.owner_coldkey IS DISTINCT FROM NEW.owner_coldkey
    OR OLD.owner_block_number IS DISTINCT FROM NEW.owner_block_number
    OR OLD.owner_block_hash IS DISTINCT FROM NEW.owner_block_hash
  ) THEN
    RAISE EXCEPTION 'lab_arena_submission_owner_immutable' USING ERRCODE = '22023';
  END IF;
  SELECT configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1'
  INTO v_integrity
  FROM public.lab_arena_rounds
  WHERE round_id = NEW.round_id;
  IF COALESCE(v_integrity, FALSE) AND NOT NEW.is_king AND (
    NEW.owner_coldkey IS NULL
    OR NEW.owner_block_number IS NULL
    OR NEW.owner_block_hash IS NULL
  ) THEN
    RAISE EXCEPTION 'lab_arena_owner_admission_required' USING ERRCODE = '23502';
  END IF;
  IF NOT COALESCE(v_integrity, FALSE) AND (
    NEW.owner_coldkey IS NOT NULL
    OR NEW.owner_block_number IS NOT NULL
    OR NEW.owner_block_hash IS NOT NULL
  ) THEN
    RAISE EXCEPTION 'lab_arena_owner_admission_policy_mismatch' USING ERRCODE = '22023';
  END IF;
  RETURN NEW;
END;
$lab_arena_owner_admission_guard$;
ALTER FUNCTION public.lab_arena_owner_admission_guard_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_owner_admission_guard_v1() FROM PUBLIC;

DROP TRIGGER IF EXISTS lab_arena_submissions_owner_admission
  ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_submissions_owner_admission
BEFORE INSERT OR UPDATE ON public.lab_arena_submissions
FOR EACH ROW EXECUTE FUNCTION public.lab_arena_owner_admission_guard_v1();

CREATE OR REPLACE FUNCTION public.lab_arena__register_submission_v2(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB,
  p_owner_coldkey TEXT,
  p_owner_block_number BIGINT,
  p_owner_block_hash TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission_core$
DECLARE
  v_round public.lab_arena_rounds;
  v_existing public.lab_arena_submissions;
  v_owner_existing public.lab_arena_submissions;
  v_collision public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_requested_baseline BOOLEAN;
  v_integrity BOOLEAN;
  v_expected_ref TEXT;
  v_checksum TEXT;
  v_existing_checksum TEXT;
  v_fixed_owner_coldkey TEXT := p_owner_coldkey;
  v_fixed_owner_block_number BIGINT := p_owner_block_number;
  v_fixed_owner_block_hash TEXT := p_owner_block_hash;
BEGIN
  v_expected_ref := 'arena/' || p_round_id || '/sources/' || p_submission_id || '.tar.gz';
  IF COALESCE(p_submission_id, '') !~ '^[A-Za-z0-9._:-]{1,64}$'
     OR pg_catalog.jsonb_typeof(p_doc) IS DISTINCT FROM 'object'
     OR p_doc ->> 'source_ref' IS DISTINCT FROM v_expected_ref
     OR COALESCE((p_doc ->> 'source_size_bytes')::BIGINT, 0) NOT BETWEEN 1 AND 10485760
     OR COALESCE((p_doc #>> '{consent,public_rerun}')::BOOLEAN, FALSE) IS NOT TRUE THEN
    RAISE EXCEPTION 'lab_arena_submission_input_invalid' USING ERRCODE = '22023';
  END IF;
  v_checksum := p_doc ->> 'source_content_md5';
  IF v_checksum IS NOT NULL AND (
    pg_catalog.length(pg_catalog.decode(v_checksum, 'base64')) <> 16
    OR pg_catalog.encode(pg_catalog.decode(v_checksum, 'base64'), 'base64') <> v_checksum
  ) THEN
    RAISE EXCEPTION 'lab_arena_source_checksum_invalid' USING ERRCODE = '22023';
  END IF;
  v_requested_baseline := COALESCE((p_doc ->> 'is_king')::BOOLEAN, FALSE);
  v_is_baseline := v_requested_baseline;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  v_integrity := COALESCE(
    v_round.configuration_doc ->> 'integrity_policy' = 'arena_integrity_v1',
    FALSE
  );
  IF v_integrity AND v_requested_baseline THEN
    v_is_baseline := (
      p_submission_id = 'baseline-' || pg_catalog.substr(p_round_id, 7)
      AND p_miner_hotkey = v_round.configuration_doc ->> 'baseline_hotkey'
    );
    IF NOT v_is_baseline THEN
      RAISE EXCEPTION 'lab_arena_baseline_identity_invalid' USING ERRCODE = '22023';
    END IF;
  END IF;
  IF v_integrity AND NOT v_is_baseline AND (
    COALESCE(p_owner_coldkey, '') !~ '^[1-9A-HJ-NP-Za-km-z]{46,48}$'
    OR p_owner_block_number IS NULL OR p_owner_block_number < 0
    OR COALESCE(p_owner_block_hash, '') !~ '^0x[0-9a-f]{64}$'
  ) THEN
    RAISE EXCEPTION 'lab_arena_owner_admission_required' USING ERRCODE = '23502';
  END IF;
  IF (NOT v_integrity OR v_is_baseline) AND (
    p_owner_coldkey IS NOT NULL
    OR p_owner_block_number IS NOT NULL
    OR p_owner_block_hash IS NOT NULL
  ) THEN
    RAISE EXCEPTION 'lab_arena_owner_admission_policy_mismatch' USING ERRCODE = '22023';
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

  SELECT * INTO v_existing
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
    AND status IN ('uploading', 'accepted', 'frozen')
  ORDER BY created_at LIMIT 1 FOR UPDATE;
  IF FOUND THEN
    IF v_integrity AND NOT v_is_baseline THEN
      IF v_existing.owner_coldkey IS NULL
         OR v_existing.owner_coldkey IS DISTINCT FROM p_owner_coldkey THEN
        RAISE EXCEPTION 'lab_arena_submission_owner_changed' USING ERRCODE = '23505';
      END IF;
      v_fixed_owner_coldkey := v_existing.owner_coldkey;
      v_fixed_owner_block_number := v_existing.owner_block_number;
      v_fixed_owner_block_hash := v_existing.owner_block_hash;
    END IF;
    v_existing_checksum := v_existing.submission_doc ->> 'source_content_md5';
    IF v_existing.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT
       AND (v_existing_checksum IS NOT DISTINCT FROM v_checksum
         OR (v_existing.status IN ('accepted', 'frozen') AND v_existing_checksum IS NULL)) THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'submission_status', v_existing.status,
        'submission_id', v_existing.submission_id, 'source_ref', v_existing.source_ref
      );
    END IF;
    IF v_existing.status <> 'uploading' OR v_is_baseline OR v_existing.is_king THEN
      RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
    END IF;
    UPDATE public.lab_arena_submissions
    SET status = 'rejected', rejection_rule = 'source_replaced'
    WHERE submission_id = v_existing.submission_id AND status = 'uploading';
  END IF;

  IF v_integrity AND NOT v_is_baseline THEN
    SELECT * INTO v_owner_existing
    FROM public.lab_arena_submissions
    WHERE round_id = p_round_id
      AND owner_coldkey = v_fixed_owner_coldkey
      AND status IN ('uploading', 'accepted', 'frozen')
    ORDER BY created_at LIMIT 1 FOR UPDATE;
    IF FOUND THEN
      RAISE EXCEPTION 'lab_arena_owner_active_submission' USING ERRCODE = '23505';
    END IF;
  END IF;

  SELECT * INTO v_collision
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id;
  IF FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;

  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king,
    source_ref, source_size_bytes, consent, submission_doc,
    owner_coldkey, owner_block_number, owner_block_hash
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploading', v_is_baseline,
    p_doc ->> 'source_ref', (p_doc ->> 'source_size_bytes')::BIGINT,
    p_doc -> 'consent', p_doc,
    v_fixed_owner_coldkey, v_fixed_owner_block_number, v_fixed_owner_block_hash
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'registered',
    'submission_status', 'uploading',
    'submission_id', p_submission_id,
    'source_ref', p_doc ->> 'source_ref'
  );
END;
$lab_arena_register_submission_core$;
ALTER FUNCTION public.lab_arena__register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena__register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) FROM PUBLIC;

CREATE OR REPLACE FUNCTION public.lab_arena_register_submission(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB
)
RETURNS JSONB
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission$
  SELECT public.lab_arena__register_submission_v2(
    p_round_id, p_submission_id, p_miner_hotkey, p_doc,
    NULL::TEXT, NULL::BIGINT, NULL::TEXT
  );
$lab_arena_register_submission$;
ALTER FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_register_submission_v2(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_doc JSONB,
  p_owner_coldkey TEXT,
  p_owner_block_number BIGINT,
  p_owner_block_hash TEXT
)
RETURNS JSONB
LANGUAGE sql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_register_submission_v2$
  SELECT public.lab_arena__register_submission_v2(
    p_round_id, p_submission_id, p_miner_hotkey, p_doc,
    p_owner_coldkey, p_owner_block_number, p_owner_block_hash
  );
$lab_arena_register_submission_v2$;
ALTER FUNCTION public.lab_arena_register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) OWNER TO lab_arena_owner;

REVOKE ALL ON FUNCTION public.lab_arena_register_submission(TEXT, TEXT, TEXT, JSONB)
  FROM PUBLIC;
REVOKE ALL ON FUNCTION public.lab_arena_register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) FROM PUBLIC;
DO $lab_arena_owner_admission_acl$
DECLARE
  role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
    IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
      EXECUTE pg_catalog.format(
        'REVOKE ALL ON FUNCTION public.lab_arena_register_submission_v2(TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT) FROM %I',
        role_name
      );
    END IF;
  END LOOP;
END;
$lab_arena_owner_admission_acl$;
GRANT EXECUTE ON FUNCTION public.lab_arena_register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) TO lab_arena_service;

NOTIFY pgrst, 'reload schema';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
COMMIT;

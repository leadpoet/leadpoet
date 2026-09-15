-- Enforce one durable replacement reservation per hotkey and daily Arena round.
-- Every linked submission, including a rejected or abandoned one, consumes the
-- allowance. Existing submission history and audit links remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $replacement_limit_prerequisites$
DECLARE
  v_registration TEXT;
  v_finalize TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_registration;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)'::pg_catalog.regprocedure
  ) INTO v_finalize;
  IF pg_catalog.to_regclass('public.lab_arena_submissions_replaces_idx') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions_one_current_per_miner_uq') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions_one_uploading_per_miner_uq') IS NULL
     OR pg_catalog.strpos(v_registration, 'v_replaces_submission_id') = 0
     OR pg_catalog.strpos(v_finalize, 'v_prior.status') = 0 THEN
    RAISE EXCEPTION 'lab_arena_replacement_limit_prerequisite_shape_unexpected';
  END IF;
END;
$replacement_limit_prerequisites$;

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
  v_current public.lab_arena_submissions;
  v_collision public.lab_arena_submissions;
  v_attempt public.lab_arena_submissions;
  v_prior_original public.lab_arena_submissions;
  v_is_baseline BOOLEAN;
  v_requested_baseline BOOLEAN;
  v_integrity BOOLEAN;
  v_expected_ref TEXT;
  v_checksum TEXT;
  v_existing_checksum TEXT;
  v_replaces_submission_id TEXT;
  v_now TIMESTAMPTZ;
  v_freeze_at TIMESTAMPTZ;
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

  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      - INTERVAL '1 hour';
  IF v_freeze_at IS NULL THEN
    RAISE EXCEPTION 'lab_arena_replacement_schedule_invalid' USING ERRCODE = '22023';
  END IF;

  -- The unfinished reservation is the retry target. It can coexist with the
  -- previously accepted fallback, but only one upload is pending at a time.
  SELECT * INTO v_existing
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
    AND status = 'uploading'
  ORDER BY created_at LIMIT 1 FOR UPDATE;
  IF FOUND THEN
    v_now := pg_catalog.clock_timestamp();
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
    IF v_existing.replaces_submission_id IS NOT NULL
       AND v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    IF v_existing.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT
       AND v_existing_checksum IS NOT DISTINCT FROM v_checksum THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'submission_status', v_existing.status,
        'submission_id', v_existing.submission_id, 'source_ref', v_existing.source_ref
      );
    END IF;
    IF v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    IF v_is_baseline OR v_existing.is_king THEN
      RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
    END IF;
    v_replaces_submission_id := v_existing.replaces_submission_id;
    IF v_replaces_submission_id IS NULL THEN
      v_replaces_submission_id := v_existing.submission_id;
    END IF;
  END IF;

  -- Any earlier linked reservation consumed the one replacement allowance,
  -- regardless of whether upload, validation, review, or finalization succeeded.
  -- The round row lock above serializes this read with every registration.
  SELECT * INTO v_attempt
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
    AND replaces_submission_id IS NOT NULL
  ORDER BY created_at, submission_id LIMIT 1 FOR UPDATE;
  IF FOUND THEN
    IF v_integrity AND NOT v_is_baseline AND
       v_attempt.owner_coldkey IS DISTINCT FROM p_owner_coldkey THEN
      RAISE EXCEPTION 'lab_arena_submission_owner_changed' USING ERRCODE = '23505';
    END IF;
    -- An accepted current source is still an idempotent transport target.
    -- A rejected historical attempt is terminal, even for identical bytes.
    SELECT * INTO v_current
    FROM public.lab_arena_submissions
    WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
      AND status IN ('accepted', 'frozen')
    ORDER BY created_at LIMIT 1 FOR UPDATE;
    IF FOUND AND v_current.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT
       AND v_current.submission_doc ->> 'source_content_md5'
           IS NOT DISTINCT FROM v_checksum THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'existing', 'submission_status', v_current.status,
        'submission_id', v_current.submission_id,
        'source_ref', v_current.source_ref
      );
    END IF;
    v_now := pg_catalog.clock_timestamp();
    IF v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'replacement_limit_reached',
      'max_replacement_attempts', 1
    );
  END IF;

  SELECT * INTO v_current
  FROM public.lab_arena_submissions
  WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
    AND status IN ('accepted', 'frozen')
  ORDER BY created_at LIMIT 1 FOR UPDATE;
  IF FOUND THEN
    v_now := pg_catalog.clock_timestamp();
    IF v_integrity AND NOT v_is_baseline AND
       v_current.owner_coldkey IS DISTINCT FROM p_owner_coldkey THEN
      RAISE EXCEPTION 'lab_arena_submission_owner_changed' USING ERRCODE = '23505';
    END IF;
    IF v_replaces_submission_id IS NULL THEN
      v_existing_checksum := v_current.submission_doc ->> 'source_content_md5';
      IF v_current.source_size_bytes = (p_doc ->> 'source_size_bytes')::BIGINT
         AND v_existing_checksum IS NOT DISTINCT FROM v_checksum THEN
        RETURN pg_catalog.jsonb_build_object(
          'status', 'existing', 'submission_status', v_current.status,
          'submission_id', v_current.submission_id,
          'source_ref', v_current.source_ref
        );
      END IF;
      v_replaces_submission_id := v_current.submission_id;
    END IF;
    IF v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    IF v_is_baseline OR v_current.is_king OR v_current.status <> 'accepted'
       OR v_current.code_review_status = 'reviewing'
       OR v_round.benchmark_ref IS NOT NULL
       OR NOT (v_round.participants IS NULL
         OR v_round.participants = '[]'::JSONB)
       OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id = p_round_id)
    THEN
      RAISE EXCEPTION 'lab_arena_submission_replacement_ineligible'
        USING ERRCODE = '23514';
    END IF;
    v_fixed_owner_coldkey := v_current.owner_coldkey;
    v_fixed_owner_block_number := v_current.owner_block_number;
    v_fixed_owner_block_hash := v_current.owner_block_hash;
  END IF;

  -- A rejected original is still the first submission for this hotkey.
  -- Link the next distinct source to it instead of admitting another original.
  IF v_replaces_submission_id IS NULL THEN
    SELECT * INTO v_prior_original
    FROM public.lab_arena_submissions
    WHERE round_id = p_round_id AND miner_hotkey = p_miner_hotkey
      AND replaces_submission_id IS NULL AND status = 'rejected'
    ORDER BY created_at, submission_id LIMIT 1 FOR UPDATE;
    IF FOUND THEN
      v_now := pg_catalog.clock_timestamp();
      IF v_now >= v_freeze_at THEN
        RETURN pg_catalog.jsonb_build_object(
          'status', 'replacement_closed', 'submission_cutoff',
          v_round.configuration_doc #>> '{schedule,submission_cutoff}'
        );
      END IF;
      IF v_is_baseline OR v_prior_original.is_king
         OR v_round.benchmark_ref IS NOT NULL
         OR NOT (v_round.participants IS NULL
           OR v_round.participants = '[]'::JSONB)
         OR EXISTS (SELECT 1 FROM public.lab_arena_runs WHERE round_id = p_round_id)
      THEN
        RAISE EXCEPTION 'lab_arena_submission_replacement_ineligible'
          USING ERRCODE = '23514';
      END IF;
      IF v_integrity AND
         v_prior_original.owner_coldkey IS DISTINCT FROM p_owner_coldkey THEN
        RAISE EXCEPTION 'lab_arena_submission_owner_changed' USING ERRCODE = '23505';
      END IF;
      v_fixed_owner_coldkey := v_prior_original.owner_coldkey;
      v_fixed_owner_block_number := v_prior_original.owner_block_number;
      v_fixed_owner_block_hash := v_prior_original.owner_block_hash;
      v_replaces_submission_id := v_prior_original.submission_id;
    END IF;
  END IF;

  SELECT * INTO v_collision
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id;
  IF FOUND THEN
    RAISE EXCEPTION 'lab_arena_submission_conflict' USING ERRCODE = '23505';
  END IF;

  IF v_existing.submission_id IS NOT NULL THEN
    UPDATE public.lab_arena_submissions
    SET status = 'rejected', rejection_rule = 'source_replaced',
        replaced_by_submission_id = p_submission_id
    WHERE submission_id = v_existing.submission_id AND status = 'uploading';
  END IF;

  INSERT INTO public.lab_arena_submissions (
    submission_id, round_id, miner_hotkey, status, is_king,
    source_ref, source_size_bytes, consent, submission_doc,
    owner_coldkey, owner_block_number, owner_block_hash,
    replaces_submission_id
  ) VALUES (
    p_submission_id, p_round_id, p_miner_hotkey, 'uploading', v_is_baseline,
    p_doc ->> 'source_ref', (p_doc ->> 'source_size_bytes')::BIGINT,
    p_doc -> 'consent', p_doc,
    v_fixed_owner_coldkey, v_fixed_owner_block_number, v_fixed_owner_block_hash,
    v_replaces_submission_id
  );
  RETURN pg_catalog.jsonb_build_object(
    'status', 'registered',
    'submission_status', 'uploading',
    'submission_id', p_submission_id,
    'source_ref', p_doc ->> 'source_ref'
  );
END;
$lab_arena_register_submission_core$;
CREATE OR REPLACE FUNCTION public.lab_arena_accept_submission_with_credentials(
  p_round_id TEXT,
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_credentials JSONB
)
RETURNS JSONB
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_accept_submission_with_credentials$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_prior public.lab_arena_submissions;
  v_now TIMESTAMPTZ;
  v_freeze_at TIMESTAMPTZ;
  v_keys TEXT[];
  v_existing_required INTEGER;
  v_existing_scrapingdog BOOLEAN;
  v_openrouter TEXT;
  v_deepline TEXT;
  v_scrapingdog TEXT;
BEGIN
  IF pg_catalog.jsonb_typeof(p_credentials) IS DISTINCT FROM 'object' THEN
    RAISE EXCEPTION 'lab_arena_credentials_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT COALESCE(pg_catalog.array_agg(key ORDER BY key), ARRAY[]::TEXT[])
  INTO v_keys
  FROM pg_catalog.jsonb_object_keys(p_credentials) AS key;
  v_openrouter := p_credentials ->> 'openrouter';
  v_deepline := p_credentials ->> 'deepline';
  v_scrapingdog := p_credentials ->> 'scrapingdog';
  IF (v_keys <> ARRAY['deepline', 'openrouter']::TEXT[]
      AND v_keys <> ARRAY['deepline', 'openrouter', 'scrapingdog']::TEXT[])
     OR pg_catalog.char_length(COALESCE(v_openrouter, '')) NOT BETWEEN 4 AND 10924
     OR pg_catalog.char_length(COALESCE(v_deepline, '')) NOT BETWEEN 4 AND 10924
     OR v_openrouter !~ '^[A-Za-z0-9+/]+={0,2}$'
     OR v_deepline !~ '^[A-Za-z0-9+/]+={0,2}$'
     OR pg_catalog.char_length(v_openrouter) % 4 <> 0
     OR pg_catalog.char_length(v_deepline) % 4 <> 0
     OR (
       v_keys = ARRAY['deepline', 'openrouter', 'scrapingdog']::TEXT[]
       AND (
         pg_catalog.char_length(COALESCE(v_scrapingdog, '')) NOT BETWEEN 4 AND 10924
         OR v_scrapingdog !~ '^[A-Za-z0-9+/]+={0,2}$'
         OR pg_catalog.char_length(v_scrapingdog) % 4 <> 0
       )
     ) THEN
    RAISE EXCEPTION 'lab_arena_credentials_invalid' USING ERRCODE = '22023';
  END IF;

  SELECT * INTO v_round
  FROM public.lab_arena_rounds
  WHERE round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.status <> 'open'
     OR COALESCE(
       pg_catalog.clock_timestamp() <
         (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ,
       TRUE
     )
     OR COALESCE(
       pg_catalog.clock_timestamp() >=
         (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ,
       TRUE
     ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'window_closed', 'round_status', v_round.status
    );
  END IF;

  SELECT * INTO v_submission
  FROM public.lab_arena_submissions
  WHERE submission_id = p_submission_id AND round_id = p_round_id
  FOR UPDATE;
  IF NOT FOUND OR v_submission.miner_hotkey IS DISTINCT FROM p_miner_hotkey THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.is_king
     AND v_submission.submission_id =
       'baseline-' || pg_catalog.regexp_replace(v_submission.round_id, '^arena-', '')
     AND v_submission.miner_hotkey =
       v_round.configuration_doc ->> 'baseline_hotkey' THEN
    RAISE EXCEPTION 'lab_arena_baseline_credentials_forbidden' USING ERRCODE = '42501';
  END IF;
  IF v_submission.status IN ('accepted', 'frozen') THEN
    SELECT
      COUNT(*) FILTER (WHERE provider IN ('openrouter', 'deepline')),
      COALESCE(bool_or(provider = 'scrapingdog'), FALSE)
    INTO v_existing_required, v_existing_scrapingdog
    FROM public.lab_arena_submission_credentials
    WHERE submission_id = p_submission_id
      AND miner_hotkey = p_miner_hotkey;
    IF v_existing_required <> 2 THEN
      RAISE EXCEPTION 'lab_arena_submission_credentials_missing' USING ERRCODE = '23514';
    END IF;
    IF v_scrapingdog IS NOT NULL AND NOT v_existing_scrapingdog THEN
      RAISE EXCEPTION 'lab_arena_submission_credentials_immutable' USING ERRCODE = '23514';
    END IF;
    RETURN pg_catalog.jsonb_build_object(
      'status', 'existing', 'submission_status', v_submission.status
    );
  END IF;
  IF v_submission.status <> 'uploading'
     OR v_submission.source_ref IS NULL
     OR v_submission.source_size_bytes IS NULL THEN
    RAISE EXCEPTION 'lab_arena_submission_not_uploading' USING ERRCODE = '22023';
  END IF;

  -- The row and round locks close every race with another reservation,
  -- finalize, review claim, participant freeze, or cutoff transition.
  v_now := pg_catalog.clock_timestamp();
  v_freeze_at :=
    (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
      - INTERVAL '1 hour';
  IF v_submission.replaces_submission_id IS NOT NULL THEN
    IF v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    SELECT * INTO v_prior
    FROM public.lab_arena_submissions
    WHERE submission_id = v_submission.replaces_submission_id
      AND round_id = p_round_id
      AND miner_hotkey = p_miner_hotkey
    FOR UPDATE;
    IF NOT FOUND THEN
      RAISE EXCEPTION 'lab_arena_submission_replacement_link_invalid'
        USING ERRCODE = '23514';
    END IF;
    v_now := pg_catalog.clock_timestamp();
    IF v_now >= v_freeze_at THEN
      RETURN pg_catalog.jsonb_build_object(
        'status', 'replacement_closed', 'submission_cutoff',
        v_round.configuration_doc #>> '{schedule,submission_cutoff}'
      );
    END IF;
    IF v_prior.status = 'accepted' THEN
      IF v_prior.is_king OR v_prior.code_review_status = 'reviewing'
         OR v_round.benchmark_ref IS NOT NULL
         OR NOT (v_round.participants IS NULL
           OR v_round.participants = '[]'::JSONB)
         OR EXISTS (SELECT 1 FROM public.lab_arena_runs
           WHERE round_id = p_round_id)
      THEN
        RAISE EXCEPTION 'lab_arena_submission_replacement_ineligible'
          USING ERRCODE = '23514';
      END IF;
    ELSIF v_prior.status <> 'rejected'
       OR (v_prior.replaces_submission_id IS NOT NULL
           AND v_prior.rejection_rule <> 'source_replaced') THEN
      RAISE EXCEPTION 'lab_arena_submission_replacement_stale'
        USING ERRCODE = '23514';
    END IF;
  END IF;

  IF v_prior.status = 'accepted' THEN
    UPDATE public.lab_arena_submissions
    SET status = 'rejected', rejection_rule = 'source_replaced',
        replaced_by_submission_id = p_submission_id
    WHERE submission_id = v_prior.submission_id AND status = 'accepted';
  END IF;

  INSERT INTO public.lab_arena_submission_credentials (
    submission_id, miner_hotkey, provider, ciphertext
  ) VALUES
    (p_submission_id, p_miner_hotkey, 'openrouter', pg_catalog.decode(v_openrouter, 'base64')),
    (p_submission_id, p_miner_hotkey, 'deepline', pg_catalog.decode(v_deepline, 'base64'));
  IF v_scrapingdog IS NOT NULL THEN
    INSERT INTO public.lab_arena_submission_credentials (
      submission_id, miner_hotkey, provider, ciphertext
    ) VALUES (
      p_submission_id, p_miner_hotkey, 'scrapingdog',
      pg_catalog.decode(v_scrapingdog, 'base64')
    );
  END IF;
  -- Credential INSERT can wait past either deadline. An exception rolls back
  -- the prior-source transition and all new ciphertext in this RPC.
  v_now := pg_catalog.clock_timestamp();
  IF v_now >=
       (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ THEN
    RAISE EXCEPTION 'lab_arena_submission_window_closed'
      USING ERRCODE = '23514';
  END IF;
  IF v_submission.replaces_submission_id IS NOT NULL
     AND v_now >= v_freeze_at THEN
    RAISE EXCEPTION 'lab_arena_submission_replacement_closed'
      USING ERRCODE = '23514';
  END IF;
  UPDATE public.lab_arena_submissions
  SET status = 'accepted', updated_at = pg_catalog.clock_timestamp()
  WHERE submission_id = p_submission_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'submission_status', 'accepted'
  );
END;
$lab_arena_accept_submission_with_credentials$;
CREATE OR REPLACE FUNCTION public.lab_arena_submission_replacement_schema_v1()
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $replacement_schema$
DECLARE
  v_registration TEXT;
  v_finalize TEXT;
  v_review TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__register_submission_v2(text,text,text,jsonb,text,bigint,text)'::pg_catalog.regprocedure
  ) INTO v_registration;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)'::pg_catalog.regprocedure
  ) INTO v_finalize;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_begin_submission_review(text,text,text,bigint,text,integer,bigint)'::pg_catalog.regprocedure
  ) INTO v_review;
  IF pg_catalog.to_regclass('public.lab_arena_submissions_one_current_per_miner_uq') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions_one_uploading_per_miner_uq') IS NULL
     OR pg_catalog.strpos(v_registration, 'replacement_limit_reached') = 0
     OR pg_catalog.strpos(v_registration, 'v_prior_original') = 0
     OR pg_catalog.strpos(v_finalize, 'v_prior.replaces_submission_id IS NOT NULL') = 0
     OR pg_catalog.strpos(v_review, 'lab_arena_queued_replacement_review_deferral') = 0 THEN
    RAISE EXCEPTION 'lab_arena_queued_replacement_schema_incomplete';
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.submission_replacement_schema.v1',
    'version', 258,
    'replacement_freeze_seconds', 3600,
    'max_replacement_attempts', 1
  );
END;
$replacement_schema$;
ALTER FUNCTION public.lab_arena__register_submission_v2(
  TEXT, TEXT, TEXT, JSONB, TEXT, BIGINT, TEXT
) OWNER TO lab_arena_owner;
ALTER FUNCTION public.lab_arena_accept_submission_with_credentials(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;
ALTER FUNCTION public.lab_arena_submission_replacement_schema_v1()
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_replacement_schema_v1()
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_replacement_schema_v1()
  TO lab_arena_service;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

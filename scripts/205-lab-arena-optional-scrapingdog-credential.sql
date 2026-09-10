-- 205-lab-arena-optional-scrapingdog-credential.sql
-- Add one optional miner-owned Scrapingdog ciphertext without changing the
-- two runtime credentials required for every accepted miner submission.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_205_requires_current_arena$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_schema_version_v1()') IS NULL
     OR (public.lab_arena_schema_version_v1() ->> 'version')::INTEGER <> 197
     OR pg_catalog.to_regclass(
          'public.lab_arena_submission_credentials'
        ) IS NULL
     OR pg_catalog.to_regprocedure(
          'public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)'
        ) IS NULL
     OR pg_catalog.to_regprocedure(
          'public.lab_arena_get_submission_credential(text,text,text)'
        ) IS NULL THEN
    RAISE EXCEPTION 'apply the Arena schema 197 credential migrations first';
  END IF;
END;
$lab_arena_205_requires_current_arena$;

GRANT CREATE ON SCHEMA public TO lab_arena_owner;

ALTER TABLE public.lab_arena_submission_credentials
  DROP CONSTRAINT IF EXISTS lab_arena_submission_credentials_provider_check;
ALTER TABLE public.lab_arena_submission_credentials
  ADD CONSTRAINT lab_arena_submission_credentials_provider_check
  CHECK (provider IN ('openrouter', 'deepline', 'scrapingdog'));

-- Save the two required ciphertexts and, when present, the optional
-- Scrapingdog ciphertext in the same transaction as admission. Existing
-- accepted credentials remain append-only. A replay cannot add the optional
-- key to a submission that was already accepted without it.
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
  FOR SHARE;
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
  UPDATE public.lab_arena_submissions
  SET status = 'accepted', updated_at = pg_catalog.clock_timestamp()
  WHERE submission_id = p_submission_id;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'ok', 'submission_status', 'accepted'
  );
END;
$lab_arena_accept_submission_with_credentials$;
ALTER FUNCTION public.lab_arena_accept_submission_with_credentials(TEXT, TEXT, TEXT, JSONB)
  OWNER TO lab_arena_owner;

-- Return only one exact provider row for its accepted/frozen submission owner.
-- The OpenRouter management key still has no stored row or provider value.
CREATE OR REPLACE FUNCTION public.lab_arena_get_submission_credential(
  p_submission_id TEXT,
  p_miner_hotkey TEXT,
  p_provider TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $lab_arena_get_submission_credential$
DECLARE
  v_credential public.lab_arena_submission_credentials;
BEGIN
  IF p_provider NOT IN ('openrouter', 'deepline', 'scrapingdog') THEN
    RAISE EXCEPTION 'lab_arena_credential_provider_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT credentials.* INTO v_credential
  FROM public.lab_arena_submission_credentials AS credentials
  JOIN public.lab_arena_submissions AS submissions
    ON submissions.submission_id = credentials.submission_id
  WHERE credentials.submission_id = p_submission_id
    AND credentials.miner_hotkey = p_miner_hotkey
    AND credentials.provider = p_provider
    AND submissions.miner_hotkey = p_miner_hotkey
    AND submissions.status IN ('accepted', 'frozen');
  IF NOT FOUND THEN
    RETURN pg_catalog.jsonb_build_object('status', 'missing');
  END IF;
  RETURN pg_catalog.jsonb_build_object(
    'status', 'available',
    'submission_id', v_credential.submission_id,
    'miner_hotkey', v_credential.miner_hotkey,
    'provider', v_credential.provider,
    'ciphertext_b64', pg_catalog.replace(
      pg_catalog.encode(v_credential.ciphertext, 'base64'), E'\n', ''
    )
  );
END;
$lab_arena_get_submission_credential$;
ALTER FUNCTION public.lab_arena_get_submission_credential(TEXT, TEXT, TEXT)
  OWNER TO lab_arena_owner;

DO $lab_arena_credential_function_acl$
DECLARE
  signature TEXT;
  role_name TEXT;
BEGIN
  FOREACH signature IN ARRAY ARRAY[
    'public.lab_arena_accept_submission_with_credentials(TEXT, TEXT, TEXT, JSONB)',
    'public.lab_arena_get_submission_credential(TEXT, TEXT, TEXT)'
  ] LOOP
    EXECUTE pg_catalog.format('REVOKE ALL ON FUNCTION %s FROM PUBLIC', signature);
    FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
      IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name) THEN
        EXECUTE pg_catalog.format(
          'REVOKE ALL ON FUNCTION %s FROM %I', signature, role_name
        );
      END IF;
    END LOOP;
    EXECUTE pg_catalog.format(
      'GRANT EXECUTE ON FUNCTION %s TO lab_arena_service', signature
    );
  END LOOP;
END;
$lab_arena_credential_function_acl$;

REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

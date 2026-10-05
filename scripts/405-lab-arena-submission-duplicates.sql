-- Admit verified source identities atomically with credential acceptance.
-- Historical submissions retain NULL identities and their existing behavior.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

DO $prerequisites$
BEGIN
  IF pg_catalog.to_regprocedure('public.lab_arena_accept_submission_with_credentials(text,text,text,jsonb)') IS NULL
     OR pg_catalog.to_regprocedure('public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)') IS NULL
     OR pg_catalog.to_regclass('public.lab_arena_submissions') IS NULL THEN
    RAISE EXCEPTION 'lab_arena_duplicate_prerequisite_missing';
  END IF;
END;
$prerequisites$;

ALTER TABLE public.lab_arena_submissions
  ADD COLUMN IF NOT EXISTS source_archive_sha256 TEXT,
  ADD COLUMN IF NOT EXISTS source_normalized_sha256 TEXT;

DO $constraints$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint WHERE conname = 'lab_arena_source_digests_shape'
    AND conrelid = 'public.lab_arena_submissions'::pg_catalog.regclass) THEN
    ALTER TABLE public.lab_arena_submissions ADD CONSTRAINT lab_arena_source_digests_shape CHECK (
      (source_archive_sha256 IS NULL AND source_normalized_sha256 IS NULL)
      OR (source_archive_sha256 IS NOT NULL
          AND source_normalized_sha256 IS NOT NULL
          AND source_archive_sha256 ~ '^sha256:[0-9a-f]{64}$'
          AND source_normalized_sha256 ~ '^sha256:[0-9a-f]{64}$')
    );
  END IF;
END;
$constraints$;

CREATE INDEX IF NOT EXISTS lab_arena_submissions_normalized_digest_idx
  ON public.lab_arena_submissions (round_id, source_normalized_sha256)
  WHERE source_normalized_sha256 IS NOT NULL AND status IN ('accepted', 'frozen');

CREATE OR REPLACE FUNCTION public.lab_arena_source_digests_immutable_v1()
RETURNS trigger LANGUAGE plpgsql SET search_path = pg_catalog, public AS $fn$
BEGIN
  IF (OLD.source_archive_sha256 IS NOT NULL OR OLD.source_normalized_sha256 IS NOT NULL)
     AND (NEW.source_archive_sha256 IS DISTINCT FROM OLD.source_archive_sha256
       OR NEW.source_normalized_sha256 IS DISTINCT FROM OLD.source_normalized_sha256) THEN
    RAISE EXCEPTION 'lab_arena_source_digest_immutable' USING ERRCODE = '23514';
  END IF;
  RETURN NEW;
END;
$fn$;
ALTER FUNCTION public.lab_arena_source_digests_immutable_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_source_digests_immutable_v1() FROM PUBLIC;
DROP TRIGGER IF EXISTS lab_arena_source_digests_immutable ON public.lab_arena_submissions;
CREATE TRIGGER lab_arena_source_digests_immutable BEFORE UPDATE ON public.lab_arena_submissions
  FOR EACH ROW EXECUTE FUNCTION public.lab_arena_source_digests_immutable_v1();

CREATE OR REPLACE FUNCTION public.lab_arena_accept_submission_source_with_credentials(
  p_round_id TEXT, p_submission_id TEXT, p_miner_hotkey TEXT, p_credentials JSONB,
  p_archive_sha256 TEXT, p_normalized_sha256 TEXT
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission public.lab_arena_submissions;
  v_result JSONB;
BEGIN
  IF COALESCE(p_archive_sha256, '') !~ '^sha256:[0-9a-f]{64}$'
     OR COALESCE(p_normalized_sha256, '') !~ '^sha256:[0-9a-f]{64}$' THEN
    RAISE EXCEPTION 'lab_arena_source_digest_invalid' USING ERRCODE = '22023';
  END IF;
  SELECT * INTO v_round FROM public.lab_arena_rounds
    WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  SELECT * INTO v_submission FROM public.lab_arena_submissions
    WHERE submission_id = p_submission_id AND round_id = p_round_id FOR UPDATE;
  IF NOT FOUND OR v_submission.miner_hotkey IS DISTINCT FROM p_miner_hotkey THEN
    RAISE EXCEPTION 'lab_arena_submission_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_submission.status IN ('accepted', 'frozen') THEN
    IF v_submission.source_archive_sha256 IS DISTINCT FROM p_archive_sha256
       OR v_submission.source_normalized_sha256 IS DISTINCT FROM p_normalized_sha256 THEN
      RAISE EXCEPTION 'lab_arena_source_digest_conflict' USING ERRCODE = '23514';
    END IF;
    RETURN public.lab_arena_accept_submission_with_credentials(
      p_round_id, p_submission_id, p_miner_hotkey, p_credentials);
  END IF;
  IF v_submission.status <> 'uploading' THEN
    RAISE EXCEPTION 'lab_arena_submission_not_uploading' USING ERRCODE = '22023';
  END IF;
  -- Match the legacy admission window before making a terminal rejection.
  IF v_round.status <> 'open'
     OR COALESCE(pg_catalog.clock_timestamp() <
       (v_round.configuration_doc #>> '{schedule,submission_open}')::TIMESTAMPTZ, TRUE)
     OR COALESCE(pg_catalog.clock_timestamp() >=
       (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ, TRUE) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'window_closed', 'round_status', v_round.status);
  END IF;
  IF v_submission.replaces_submission_id IS NOT NULL AND
     pg_catalog.clock_timestamp() >=
       (v_round.configuration_doc #>> '{schedule,submission_cutoff}')::TIMESTAMPTZ
       - INTERVAL '1 hour' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'replacement_closed');
  END IF;
  IF NOT v_submission.is_king AND EXISTS (
    SELECT 1 FROM public.lab_arena_submissions AS previous
    WHERE previous.round_id = p_round_id
      AND previous.miner_hotkey <> p_miner_hotkey
      AND NOT previous.is_king
      AND previous.status IN ('accepted', 'frozen')
      AND previous.source_normalized_sha256 = p_normalized_sha256
  ) THEN
    UPDATE public.lab_arena_submissions SET status = 'rejected',
      rejection_rule = 'duplicate_submission'
    WHERE submission_id = p_submission_id AND status = 'uploading';
    RETURN pg_catalog.jsonb_build_object('status', 'rejected_duplicate');
  END IF;
  v_result := public.lab_arena_accept_submission_with_credentials(
    p_round_id, p_submission_id, p_miner_hotkey, p_credentials);
  IF v_result ->> 'status' = 'ok' THEN
    UPDATE public.lab_arena_submissions
    SET source_archive_sha256 = p_archive_sha256,
        source_normalized_sha256 = p_normalized_sha256
    WHERE submission_id = p_submission_id AND status = 'accepted';
  END IF;
  RETURN v_result;
END;
$fn$;
ALTER FUNCTION public.lab_arena_accept_submission_source_with_credentials(TEXT,TEXT,TEXT,JSONB,TEXT,TEXT)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_accept_submission_source_with_credentials(TEXT,TEXT,TEXT,JSONB,TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_accept_submission_source_with_credentials(TEXT,TEXT,TEXT,JSONB,TEXT,TEXT)
  TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_submission_duplicate_schema_v1()
RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER SET search_path = pg_catalog, public AS $fn$
  SELECT pg_catalog.jsonb_build_object(
    'schema_version', 'leadpoet.lab_arena.submission_duplicate.v1',
    'version', 405,
    'digest_format', 'sha256:hex',
    'cross_hotkey_scope', 'round'
  );
$fn$;
ALTER FUNCTION public.lab_arena_submission_duplicate_schema_v1() OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_duplicate_schema_v1() FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_duplicate_schema_v1() TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_submission_similarity_champion(p_round_id TEXT)
RETURNS JSONB LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $fn$
DECLARE
  v_round public.lab_arena_rounds;
  v_submission_id TEXT;
BEGIN
  SELECT * INTO v_round FROM public.lab_arena_rounds WHERE round_id = p_round_id;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF v_round.champion_funding_frozen THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'ready', 'submission_id', v_round.champion_submission_id);
  END IF;
  IF EXISTS (
    SELECT 1 FROM public.lab_arena_rounds AS prior
    WHERE prior.arena_network_name = v_round.arena_network_name
      AND prior.arena_netuid = v_round.arena_netuid
      AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
      AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
      AND prior.status = 'published' AND prior.promotion_required
      AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
      AND prior.baseline_promoted_at IS NULL
      AND (v_round.configuration_doc ->> 'mode' <> 'live' OR NOT EXISTS (
        SELECT 1 FROM public.lab_arena_rounds AS newer
        WHERE newer.status = 'published'
          AND newer.configuration_doc ->> 'mode' = 'live'
          AND newer.arena_network_name = prior.arena_network_name
          AND newer.arena_netuid = prior.arena_netuid
          AND newer.evaluation_date > prior.evaluation_date
      ))
  ) THEN
    RETURN pg_catalog.jsonb_build_object(
      'status', 'promotion_pending', 'submission_id', NULL);
  END IF;
  SELECT submission.submission_id INTO v_submission_id
  FROM public.lab_arena_rounds AS prior
  JOIN public.lab_arena_submissions AS submission
    ON submission.submission_id = prior.publication_doc #>> '{king_decision,winner_submission_id}'
    AND submission.round_id = prior.round_id
    AND submission.miner_hotkey = prior.publication_doc #>> '{king_decision,king_hotkey}'
    AND submission.status = 'frozen' AND NOT submission.is_king
  WHERE prior.arena_network_name = v_round.arena_network_name
    AND prior.arena_netuid = v_round.arena_netuid
    AND prior.configuration_doc ->> 'mode' = v_round.configuration_doc ->> 'mode'
    AND (v_round.configuration_doc ->> 'mode' <> 'live' OR prior.rewards_enabled)
    AND prior.status = 'published' AND prior.baseline_promoted_at IS NOT NULL
    AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
    AND prior.round_id <> p_round_id
  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;
  RETURN pg_catalog.jsonb_build_object('status', 'ready', 'submission_id', v_submission_id);
END;
$fn$;
ALTER FUNCTION public.lab_arena_submission_similarity_champion(TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_submission_similarity_champion(TEXT)
  FROM PUBLIC, anon, authenticated, service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_submission_similarity_champion(TEXT)
  TO lab_arena_service;

-- The existing bounded review can report a duplicate classification without
-- changing its ledger, claim, or completion protocol.
DO $review_category$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$'malicious_behavior', 'reviewer_manipulation'$old$;
  v_new TEXT := $new$'malicious_behavior', 'reviewer_manipulation',
           'duplicate_submission'$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_finish_submission_review(text,text,text,text,jsonb,bigint)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(v_definition, '''duplicate_submission''') = 0 THEN
    IF pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))
       <> pg_catalog.length(v_old) THEN
      RAISE EXCEPTION 'lab_arena_review_category_shape_unexpected';
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  END IF;
END;
$review_category$;

COMMENT ON COLUMN public.lab_arena_submissions.source_archive_sha256 IS
  'SHA-256 of gateway-validated archive bytes; never source text.';
COMMENT ON COLUMN public.lab_arena_submissions.source_normalized_sha256 IS
  'SHA-256 of normalized gateway-validated source contents; never source text.';
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst, 'reload schema';
COMMIT;

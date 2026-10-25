-- Freeze a full-thinking round's source snapshot with the same completed
-- champion that funds its baseline. Open intake may precede promotion.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Transferring the new functions requires CREATE for the target owner even
-- when the release role is already its member. Restore the exact schema ACL.
CREATE TEMP TABLE arena567_acl ON COMMIT DROP AS
SELECT nspacl AS acl,
  has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
FROM pg_namespace WHERE nspname='public';
DO $grant567$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena567_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$grant567$;

CREATE INDEX IF NOT EXISTS lab_arena_rounds_promoted_champion_idx
  ON public.lab_arena_rounds (
    arena_network_name, arena_netuid, baseline_promoted_at DESC, round_id DESC
  )
  WHERE status = 'published' AND rewards_enabled
    AND baseline_promoted_at IS NOT NULL;

-- The usual write-once guard remains in force. Only this RPC's transaction
-- may replace the two source fields, on an open round before baseline work.
DO $full_thinking_source_guard561$
DECLARE
  definition TEXT;
  amended TEXT;
  before_hash TEXT;
  target OID := to_regprocedure('public.lab_arena_rounds_write_once_v1()');
  marker CONSTANT TEXT := '  IF NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc THEN';
BEGIN
  SELECT pg_get_functiondef(target) INTO definition;
  before_hash := encode(extensions.digest(definition, 'sha256'), 'hex');
  IF definition IS NULL OR (SELECT pg_catalog.jsonb_build_array(
      pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
      p.provolatile, p.proconfig, language.lanname)
      FROM pg_catalog.pg_proc AS p
      JOIN pg_catalog.pg_language AS language ON language.oid = p.prolang
      WHERE p.oid = target) IS DISTINCT FROM
      '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner}",false,"v",["search_path=pg_catalog"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'full-thinking source guard security differs' USING ERRCODE = '42501';
  END IF;
  IF before_hash = '835694d66c3d32fd7c64c130159ec7029364f7b64350dc1fbd9fcf70e322d136' THEN
    RETURN;
  END IF;
  IF before_hash <>
      'f6e03c8ad4080e3e713e282719fae154ba5d3d98b9151fab282ac098e44a7c77' THEN
    RAISE EXCEPTION 'full-thinking source guard preimage differs' USING ERRCODE = '55000';
  END IF;
  IF (length(definition) - length(replace(definition, marker, ''))) <> length(marker) THEN
    RAISE EXCEPTION 'full-thinking source guard anchor differs' USING ERRCODE = '55000';
  END IF;
  amended := replace(definition, marker, $new_guard561$
  IF TG_OP = 'UPDATE'
     AND current_setting('lab_arena.full_thinking_baseline_freeze_round', true) = OLD.round_id
     AND OLD.status = 'open' AND NEW.status = 'open'
     AND OLD.champion_funding_frozen AND NEW.champion_funding_frozen
     AND OLD.participants IS NULL AND NEW.participants IS NULL
     AND OLD.benchmark_ref IS NULL AND NEW.benchmark_ref IS NULL
     AND OLD.configuration_doc ->> 'reasoning_capture_policy' = 'full_generated_thinking_v1'
     AND NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc
     AND (NEW.configuration_doc - 'baseline_source_url' - 'reasoning_baseline_source_sha256') =
         (OLD.configuration_doc - 'baseline_source_url' - 'reasoning_baseline_source_sha256')
     AND NEW.configuration_doc ->> 'baseline_source_url' ~
         '^https://github[.]com/leadpoet/leadpoet-sales-agent/archive/[0-9a-f]{40}[.]tar[.]gz$'
     AND NEW.configuration_doc ->> 'reasoning_baseline_source_sha256' ~
         '^sha256:[0-9a-f]{64}$'
     AND (to_jsonb(NEW) - ARRAY['configuration_doc','updated_at','arena_network_name','arena_netuid']) =
         (to_jsonb(OLD) - ARRAY['configuration_doc','updated_at','arena_network_name','arena_netuid']) THEN
    NEW.updated_at := clock_timestamp();
    RETURN NEW;
  END IF;
  IF NEW.configuration_doc IS DISTINCT FROM OLD.configuration_doc THEN$new_guard561$);
  IF encode(extensions.digest(amended, 'sha256'), 'hex') <>
      '835694d66c3d32fd7c64c130159ec7029364f7b64350dc1fbd9fcf70e322d136' THEN
    RAISE EXCEPTION 'full-thinking source guard target differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE amended;
  IF encode(extensions.digest(pg_get_functiondef(target), 'sha256'), 'hex') <>
      '835694d66c3d32fd7c64c130159ec7029364f7b64350dc1fbd9fcf70e322d136' THEN
    RAISE EXCEPTION 'full-thinking source guard readback differs' USING ERRCODE = '55000';
  END IF;
END;
$full_thinking_source_guard561$;

CREATE OR REPLACE FUNCTION public.lab_arena_latest_promoted_champion_v1(
  p_network_name TEXT, p_netuid INTEGER
) RETURNS JSONB LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public AS $latest_full_thinking_champion561$
  SELECT pg_catalog.jsonb_build_object(
    'round_id', prior.round_id,
    'winner_submission_id', submission.submission_id,
    'commit', prior.promotion_doc ->> 'commit',
    'reasoning_capture_policy', prior.configuration_doc ->> 'reasoning_capture_policy'
  )
  FROM public.lab_arena_rounds AS prior
  JOIN public.lab_arena_submissions AS submission
    ON submission.submission_id = prior.publication_doc #>> '{king_decision,winner_submission_id}'
    AND submission.round_id = prior.round_id
    AND submission.miner_hotkey = prior.publication_doc #>> '{king_decision,king_hotkey}'
    AND submission.status = 'frozen' AND NOT submission.is_king
  WHERE prior.arena_network_name = p_network_name
    AND prior.arena_netuid = p_netuid
    AND prior.configuration_doc ->> 'mode' = 'live'
    AND prior.rewards_enabled
    AND prior.status = 'published' AND prior.baseline_promoted_at IS NOT NULL
    AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;
$latest_full_thinking_champion561$;
ALTER FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER)
  OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_latest_promoted_champion_v1(TEXT,INTEGER)
  TO lab_arena_service;

CREATE OR REPLACE FUNCTION public.lab_arena_freeze_full_thinking_baseline_v1(
  p_round_id TEXT,
  p_expected_promotion_round_id TEXT,
  p_expected_winner_submission_id TEXT,
  p_expected_commit TEXT,
  p_expected_source_url TEXT,
  p_expected_source_sha256 TEXT,
  p_source_url TEXT,
  p_source_sha256 TEXT
) RETURNS JSONB LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public AS $full_thinking_freeze561$
DECLARE
  current_round public.lab_arena_rounds;
  promoted_round public.lab_arena_rounds;
  funding JSONB;
  winner_id TEXT;
  promoted_commit TEXT;
  next_configuration JSONB;
BEGIN
  -- The promotion RPCs take this same lock before changing completion state.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab_arena.baseline_promotion', 0)
  );
  SELECT * INTO current_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_round_missing' USING ERRCODE = 'P0002';
  END IF;
  IF current_round.champion_funding_frozen OR current_round.status <> 'open' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'existing');
  END IF;
  IF current_round.configuration_doc ->> 'mode' <> 'live'
     OR current_round.configuration_doc ->> 'reasoning_capture_policy' <>
        'full_generated_thinking_v1'
     OR current_round.configuration_doc ->> 'baseline_source_url'
        IS DISTINCT FROM p_expected_source_url
     OR current_round.configuration_doc ->> 'reasoning_baseline_source_sha256'
        IS DISTINCT FROM p_expected_source_sha256 THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END IF;
  IF p_source_sha256 IS NULL OR p_source_url IS NULL
     OR p_source_sha256 !~ '^sha256:[0-9a-f]{64}$'
     OR p_source_url !~
        '^https://github[.]com/leadpoet/leadpoet-sales-agent/archive/[0-9a-f]{40}[.]tar[.]gz$' THEN
    RAISE EXCEPTION 'lab_arena_full_thinking_source_invalid' USING ERRCODE = '22023';
  END IF;
  -- Match the champion funding RPC's eligible winner and ordering exactly.
  SELECT prior.* INTO promoted_round
  FROM public.lab_arena_rounds AS prior
  JOIN public.lab_arena_submissions AS submission
    ON submission.submission_id = prior.publication_doc #>> '{king_decision,winner_submission_id}'
    AND submission.round_id = prior.round_id
    AND submission.miner_hotkey = prior.publication_doc #>> '{king_decision,king_hotkey}'
    AND submission.status = 'frozen' AND NOT submission.is_king
  WHERE prior.arena_network_name = current_round.arena_network_name
    AND prior.arena_netuid = current_round.arena_netuid
    AND prior.configuration_doc ->> 'mode' = 'live'
    AND prior.rewards_enabled
    AND prior.status = 'published' AND prior.baseline_promoted_at IS NOT NULL
    AND prior.publication_doc #>> '{king_decision,outcome}' = 'crowned'
    AND prior.round_id <> p_round_id
  ORDER BY prior.baseline_promoted_at DESC, prior.round_id DESC LIMIT 1;
  IF FOUND THEN
    winner_id := promoted_round.publication_doc #>> '{king_decision,winner_submission_id}';
    promoted_commit := promoted_round.promotion_doc ->> 'commit';
    IF promoted_round.round_id IS DISTINCT FROM p_expected_promotion_round_id
       OR winner_id IS DISTINCT FROM p_expected_winner_submission_id
       OR promoted_commit IS DISTINCT FROM p_expected_commit THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
    IF promoted_round.configuration_doc ->> 'reasoning_capture_policy' =
       'full_generated_thinking_v1' THEN
      IF promoted_commit IS NULL OR promoted_commit !~ '^[0-9a-f]{40}$'
         OR p_source_url IS DISTINCT FROM
            'https://github.com/leadpoet/leadpoet-sales-agent/archive/' ||
            promoted_commit || '.tar.gz' THEN
        RAISE EXCEPTION 'lab_arena_full_thinking_promotion_source_invalid' USING ERRCODE = '22023';
      END IF;
    ELSIF p_source_url IS DISTINCT FROM p_expected_source_url
       OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256 THEN
      -- A legacy champion still pays. Its unreviewed source cannot replace
      -- the tested full-thinking bootstrap archive.
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
  ELSE
    IF p_expected_promotion_round_id IS NOT NULL
       OR p_expected_winner_submission_id IS NOT NULL
       OR p_expected_commit IS NOT NULL
       OR p_source_url IS DISTINCT FROM p_expected_source_url
       OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256 THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
  END IF;
  IF (p_source_url IS DISTINCT FROM p_expected_source_url
      OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256)
     AND (current_round.participants IS NOT NULL
          OR current_round.benchmark_ref IS NOT NULL
          OR EXISTS (SELECT 1 FROM public.lab_arena_submissions
                     WHERE round_id = p_round_id AND is_king)
          OR EXISTS (SELECT 1 FROM public.lab_arena_runs
                     WHERE round_id = p_round_id)) THEN
    RAISE EXCEPTION 'lab_arena_full_thinking_baseline_already_started' USING ERRCODE = '55000';
  END IF;
  funding := public.lab_arena_freeze_champion_funding(p_round_id);
  IF funding ->> 'status' = 'promotion_pending' THEN
    RETURN funding;
  END IF;
  IF funding ->> 'status' <> 'frozen' THEN
    RAISE EXCEPTION 'lab_arena_full_thinking_funding_invalid' USING ERRCODE = '55000';
  END IF;
  SELECT * INTO current_round FROM public.lab_arena_rounds
  WHERE round_id = p_round_id FOR UPDATE;
  IF current_round.champion_submission_id IS DISTINCT FROM winner_id THEN
    -- A failed transaction rolls back the funding freeze as well.
    RAISE EXCEPTION 'lab_arena_full_thinking_payer_mismatch' USING ERRCODE = '55000';
  END IF;
  IF p_source_url IS DISTINCT FROM p_expected_source_url
     OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256 THEN
    next_configuration := pg_catalog.jsonb_set(
      pg_catalog.jsonb_set(current_round.configuration_doc, '{baseline_source_url}',
        pg_catalog.to_jsonb(p_source_url)),
      '{reasoning_baseline_source_sha256}', pg_catalog.to_jsonb(p_source_sha256)
    );
    PERFORM pg_catalog.set_config(
      'lab_arena.full_thinking_baseline_freeze_round', p_round_id, true
    );
    UPDATE public.lab_arena_rounds SET configuration_doc = next_configuration
    WHERE round_id = p_round_id;
    PERFORM pg_catalog.set_config(
      'lab_arena.full_thinking_baseline_freeze_round', '', true
    );
  END IF;
  RETURN pg_catalog.jsonb_build_object('status', 'frozen');
END;
$full_thinking_freeze561$;
ALTER FUNCTION public.lab_arena_freeze_full_thinking_baseline_v1(
  TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_freeze_full_thinking_baseline_v1(
  TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT)
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT EXECUTE ON FUNCTION public.lab_arena_freeze_full_thinking_baseline_v1(
  TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT,TEXT) TO lab_arena_service;

DO $restore567$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.arena567_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public') IS DISTINCT FROM
      (SELECT acl FROM pg_temp.arena567_acl) THEN
    RAISE EXCEPTION 'full-thinking baseline schema ACL changed' USING ERRCODE = '42501';
  END IF;
END;
$restore567$;

COMMIT;

-- Preserve the exact promoted champion source and its capture contract.
-- This changes only future baseline freezes. Existing frozen rounds are untouched.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $security585$
DECLARE
  signature TEXT;
  expected_acl TEXT;
BEGIN
  FOR signature, expected_acl IN VALUES
    ('public.lab_arena_rounds_write_once_v1()', '{lab_arena_owner=X/lab_arena_owner}'),
    ('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)',
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
  LOOP
    IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
          p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
          FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang
          WHERE p.oid=signature::REGPROCEDURE) IS DISTINCT FROM
          pg_catalog.jsonb_build_array('lab_arena_owner',expected_acl,
            signature LIKE '%freeze_full%', 'v',
            pg_catalog.jsonb_build_array(CASE WHEN signature LIKE '%freeze_full%'
              THEN 'search_path=pg_catalog, public' ELSE 'search_path=pg_catalog' END), 'plpgsql') THEN
        RAISE EXCEPTION 'champion baseline function security differs: %',signature USING ERRCODE='42501';
    END IF;
  END LOOP;
END;
$security585$;

DO $guard585$
DECLARE
  target REGPROCEDURE := 'public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE;
  definition TEXT;
  updated TEXT;
  before_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(target) INTO definition;
  before_hash := encode(extensions.digest(definition,'sha256'),'hex');
  IF before_hash = 'fc99c04a17205b97c55f522b445cd6533847eaaf48d2807ebd05e55ea9de5d25' THEN RETURN; END IF;
  IF before_hash <> '835694d66c3d32fd7c64c130159ec7029364f7b64350dc1fbd9fcf70e322d136' THEN
    RAISE EXCEPTION 'champion baseline guard preimage differs' USING ERRCODE='55000';
  END IF;
  updated := replace(definition,
    $old$- 'baseline_source_url' - 'reasoning_baseline_source_sha256')$old$,
    $new$- 'baseline_source_url' - 'reasoning_baseline_source_sha256' - 'baseline_reasoning_configuration')$new$);
  IF updated = definition THEN RAISE EXCEPTION 'champion baseline guard anchor differs'; END IF;
  EXECUTE updated;
END;
$guard585$;

DO $preimage585$
BEGIN
  IF encode(extensions.digest(pg_catalog.pg_get_functiondef(
      'public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)'::REGPROCEDURE),
      'sha256'),'hex') NOT IN ('ae2294509d9386d190be44a46efe88b60a6c393b4975d7e071e5724e08bde58f','6cb941811025501152d69891d190b61341acabce0b710b02322f4325a62dbf59') THEN
    RAISE EXCEPTION 'champion baseline freeze preimage differs' USING ERRCODE='55000';
  END IF;
END;
$preimage585$;

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
  baseline_reasoning JSONB;
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
    IF promoted_commit IS NULL OR promoted_commit !~ '^[0-9a-f]{40}$'
       OR p_source_url IS DISTINCT FROM
          'https://github.com/leadpoet/leadpoet-sales-agent/archive/' ||
          promoted_commit || '.tar.gz' THEN
      RAISE EXCEPTION 'lab_arena_champion_baseline_source_invalid' USING ERRCODE = '22023';
    END IF;
    baseline_reasoning := promoted_round.configuration_doc
      - ARRAY(SELECT key FROM pg_catalog.jsonb_object_keys(promoted_round.configuration_doc) AS key
              WHERE key NOT IN ('reasoning_capture_policy','reasoning_route_registry'));
  ELSE
    baseline_reasoning := current_round.configuration_doc
      - ARRAY(SELECT key FROM pg_catalog.jsonb_object_keys(current_round.configuration_doc) AS key
              WHERE key NOT IN ('reasoning_capture_policy','reasoning_route_registry'));
    IF p_expected_promotion_round_id IS NOT NULL
       OR p_expected_winner_submission_id IS NOT NULL
       OR p_expected_commit IS NOT NULL
       OR p_source_url IS DISTINCT FROM p_expected_source_url
       OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256 THEN
      RETURN pg_catalog.jsonb_build_object('status', 'stale');
    END IF;
  END IF;
  IF (p_source_url IS DISTINCT FROM p_expected_source_url
      OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256
      OR NOT current_round.configuration_doc ? 'baseline_reasoning_configuration')
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
     OR p_source_sha256 IS DISTINCT FROM p_expected_source_sha256
     OR NOT current_round.configuration_doc ? 'baseline_reasoning_configuration' THEN
    next_configuration := pg_catalog.jsonb_set(
      pg_catalog.jsonb_set(pg_catalog.jsonb_set(current_round.configuration_doc,
        '{baseline_reasoning_configuration}', baseline_reasoning), '{baseline_source_url}',
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

-- CREATE OR REPLACE retains the existing owner and service-only ACL.
DO $readback585$
BEGIN
  IF encode(extensions.digest(pg_catalog.pg_get_functiondef(
      'public.lab_arena_rounds_write_once_v1()'::REGPROCEDURE),'sha256'),'hex') <> 'fc99c04a17205b97c55f522b445cd6533847eaaf48d2807ebd05e55ea9de5d25'
     OR encode(extensions.digest(pg_catalog.pg_get_functiondef(
      'public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)'::REGPROCEDURE),
      'sha256'),'hex') <> '6cb941811025501152d69891d190b61341acabce0b710b02322f4325a62dbf59' THEN
    RAISE EXCEPTION 'champion baseline readback differs' USING ERRCODE='55000';
  END IF;
END;
$readback585$;
DO $security585$
DECLARE
  signature TEXT;
  expected_acl TEXT;
BEGIN
  FOR signature, expected_acl IN VALUES
    ('public.lab_arena_rounds_write_once_v1()', '{lab_arena_owner=X/lab_arena_owner}'),
    ('public.lab_arena_freeze_full_thinking_baseline_v1(text,text,text,text,text,text,text,text)',
     '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}')
  LOOP
    IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner),
          p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig,l.lanname)
          FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang
          WHERE p.oid=signature::REGPROCEDURE) IS DISTINCT FROM
          pg_catalog.jsonb_build_array('lab_arena_owner',expected_acl,
            signature LIKE '%freeze_full%', 'v',
            pg_catalog.jsonb_build_array(CASE WHEN signature LIKE '%freeze_full%'
              THEN 'search_path=pg_catalog, public' ELSE 'search_path=pg_catalog' END), 'plpgsql') THEN
        RAISE EXCEPTION 'champion baseline function security differs: %',signature USING ERRCODE='42501';
    END IF;
  END LOOP;
END;
$security585$;

NOTIFY pgrst,'reload schema';
COMMIT;

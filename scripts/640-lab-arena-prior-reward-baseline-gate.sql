-- Wait for the authoritative prior reward day before new baseline liability.
-- Reuse every existing 627 gate; preserve all data, policies, attempts and ACLs.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $prior_reward_baseline640$
DECLARE
  target OID := 'public.lab_arena__baseline_source_current_v1(public.lab_arena_rounds)'::regprocedure;
  definition TEXT;
  updated TEXT;
  body TEXT;
  metadata JSONB;
  identity JSONB;
  old_fragment TEXT := $old640$  IF evaluation_day IS NULL THEN RETURN FALSE; END IF;$old640$;
  new_fragment TEXT := $new640$  IF evaluation_day IS NULL THEN RETURN FALSE; END IF;
  -- An earlier unresolved reward day can still change this baseline.
  -- A newer published prior day supersedes older pending authority (377/627).
  IF p_round.rewards_enabled AND EXISTS (
    SELECT 1 FROM public.lab_arena_rounds pending
    CROSS JOIN LATERAL (
      SELECT coalesce(pending.evaluation_date,
        to_char((pending.configuration_doc#>>'{schedule,submission_cutoff}')::TIMESTAMPTZ
          AT TIME ZONE 'UTC','YYYY-MM-DD')) AS day
    ) pending_day
    WHERE pending.arena_network_name=p_round.arena_network_name
      AND pending.arena_netuid=p_round.arena_netuid
      AND pending.configuration_doc->>'mode'='live' AND pending.rewards_enabled
      AND pending.status NOT IN ('published','cancelled')
      AND pending_day.day<evaluation_day
      AND NOT EXISTS (
        SELECT 1 FROM public.lab_arena_rounds newer
        WHERE newer.arena_network_name=p_round.arena_network_name
          AND newer.arena_netuid=p_round.arena_netuid
          AND newer.configuration_doc->>'mode'='live' AND newer.status='published'
          AND newer.evaluation_date>pending_day.day
          AND newer.evaluation_date<evaluation_day
      )
  ) THEN RETURN FALSE; END IF;$new640$;
BEGIN
  PERFORM pg_advisory_xact_lock(hashtextextended('lab_arena.baseline_promotion',0));
  SELECT pg_get_functiondef(p.oid),p.prosrc,to_jsonb(p)-'prosrc',
    jsonb_build_array(pg_get_userbyid(p.proowner),p.proacl::TEXT,p.prosecdef,
      p.provolatile,p.proconfig,l.lanname,p.proisstrict,p.proleakproof,p.proparallel)
  INTO definition,body,metadata,identity
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang WHERE p.oid=target;
  IF identity IS DISTINCT FROM jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner}',TRUE,'s',
    ARRAY['search_path=pg_catalog, public']::TEXT[],'plpgsql',FALSE,FALSE,'u') THEN
    RAISE EXCEPTION 'Prior reward baseline640 security differs' USING ERRCODE='42501';
  END IF;
  IF encode(extensions.digest(body,'sha256'),'hex')='9f6e76c7f195b2699bf97c176e3a78572b903662c225a120ee3674711fcb68b4' THEN RETURN; END IF;
  IF encode(extensions.digest(body,'sha256'),'hex')<>
      '9f6d44f3be0463154a4fa2e1db64c930a7035e8ac4623b235679c5fe58af8153'
    OR length(definition)-length(replace(definition,old_fragment,''))<>length(old_fragment) THEN
    RAISE EXCEPTION 'Prior reward baseline640 preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,old_fragment,new_fragment);
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
    OR (SELECT encode(extensions.digest(p.prosrc,'sha256'),'hex') FROM pg_proc p WHERE p.oid=target)
      IS DISTINCT FROM '9f6e76c7f195b2699bf97c176e3a78572b903662c225a120ee3674711fcb68b4' THEN
    RAISE EXCEPTION 'Prior reward baseline640 readback differs' USING ERRCODE='55000';
  END IF;
END;
$prior_reward_baseline640$;
NOTIFY pgrst,'reload schema';
COMMIT;

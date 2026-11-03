-- Read each eligible run once before counting the original latest call heads.
-- The measured generic plan repeated 5,721 run-row lookups for eight runs.
-- Preserve every ledger predicate, run-kind filter, head order and counted state.
-- No locks, limits, accounting rules, indexes or stored rows change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $stage_quota_run_lookup584$
DECLARE
  target OID := 'public.lab_arena__submission_stage_consumed(text,smallint,text,text)'::regprocedure;
  definition TEXT;
  updated TEXT;
  metadata JSONB;
  preimage CONSTANT TEXT := 'b00d1e9f60bbeaa3dad776811b33fd4a81df9f9278d1091d47a8b9b02768039e';
  postimage CONSTANT TEXT := '488e418fcd9c479b6898ac661bb64392a0fcb7cd271039bc28c062c1923e78e9';
BEGIN
  SELECT pg_get_functiondef(p.oid), to_jsonb(p)-'prosrc' INTO STRICT definition, metadata
  FROM pg_proc p JOIN pg_language l ON l.oid=p.prolang
  WHERE p.oid=target
    AND pg_get_userbyid(p.proowner)='lab_arena_owner'
    AND p.proacl::TEXT='{lab_arena_owner=X/lab_arena_owner}'
    AND p.prosecdef AND p.provolatile='s'
    AND p.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
    AND l.lanname='sql';
  IF encode(extensions.digest(definition,'sha256'),'hex')=postimage THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM preimage THEN
    RAISE EXCEPTION 'Stage quota584 exact preimage differs' USING ERRCODE='55000';
  END IF;
  updated := replace(definition, $old584$  SELECT COUNT(*)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM public.lab_arena_ledger AS ledger
    JOIN public.lab_arena_runs AS runs ON runs.run_id = ledger.run_id
    WHERE ledger.submission_id = p_submission_id
      AND ledger.stage = p_stage
      AND ledger.provider = p_provider
      AND runs.kind = p_kind
      AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain');
$old584$, $new584$  WITH scoped AS MATERIALIZED (
    SELECT ledger.call_identity, ledger.entry_id, ledger.entry_kind, ledger.run_id
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.submission_id = p_submission_id
      AND ledger.stage = p_stage
      AND ledger.provider = p_provider
      AND ledger.call_identity IS NOT NULL
  ), eligible_runs AS MATERIALIZED (
    SELECT runs.run_id
    FROM (SELECT DISTINCT run_id FROM scoped) AS requested
    JOIN public.lab_arena_runs AS runs ON runs.run_id = requested.run_id
    WHERE runs.kind = p_kind
  )
  SELECT COUNT(*)::BIGINT
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
    FROM scoped AS ledger
    JOIN eligible_runs AS runs ON runs.run_id = ledger.run_id
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind IN ('reservation', 'dispatch', 'settlement', 'uncertain');
$new584$);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM postimage THEN
    RAISE EXCEPTION 'Stage quota584 exact target differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF (SELECT to_jsonb(p)-'prosrc' FROM pg_proc p WHERE p.oid=target) IS DISTINCT FROM metadata
     OR encode(extensions.digest(pg_get_functiondef(target),'sha256'),'hex') IS DISTINCT FROM postimage THEN
    RAISE EXCEPTION 'Stage quota584 readback differs' USING ERRCODE='55000';
  END IF;
END;
$stage_quota_run_lookup584$;
COMMIT;

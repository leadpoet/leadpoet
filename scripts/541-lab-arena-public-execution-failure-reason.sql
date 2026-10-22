-- Read-only public diagnostic from the existing native privacy-route proof.
-- No historical runs, scores, receipts, costs or publication data are changed.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
CREATE TEMP TABLE arena541_schema_acl ON COMMIT DROP AS
 SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
 FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $preflight541$
DECLARE p RECORD;
BEGIN
 SELECT a.*,l.lanname INTO p FROM pg_proc a JOIN pg_language l ON l.oid=a.prolang
 WHERE a.oid=to_regprocedure('public.lab_arena__execution_privacy_route_unavailable_v1(text)');
 IF NOT FOUND OR encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'4ac0ef13106cfe30274bc06247d830d384bc8693314ccbe0a619c49b24886bd5'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql' THEN
  RAISE EXCEPTION 'Arena native privacy-route proof source/security differs'; END IF;
 SELECT a.*,l.lanname INTO p FROM pg_proc a JOIN pg_language l ON l.oid=a.prolang
 WHERE a.oid=to_regprocedure('public.lab_arena_public_execution_failure_reason_v1(public.lab_arena_runs)');
 IF FOUND AND (encode(extensions.digest(p.prosrc,'sha256'),'hex')<>'4db50e4f93d4d7fb9a8a80f2976eeb0ae1a03bac8479672c2cf7615610ac42c3'
  OR pg_get_userbyid(p.proowner)<>'lab_arena_owner' OR NOT p.prosecdef OR p.provolatile<>'s'
  OR p.proacl::TEXT IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'
  OR p.proconfig IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] OR p.lanname<>'plpgsql') THEN
  RAISE EXCEPTION 'Arena public execution diagnostic source/security differs'; END IF;
END;
$preflight541$;
CREATE OR REPLACE FUNCTION public.lab_arena_public_execution_failure_reason_v1(p_run public.lab_arena_runs)
RETURNS TEXT LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path=pg_catalog,public AS $body541$
DECLARE issuance public.lab_arena_ledger;
BEGIN
 -- Only a saved execution failure can acquire this public diagnostic. Keep
 -- scores, terminal causes, retries, receipts and provider accounting intact.
 IF p_run.kind IS DISTINCT FROM 'execute' OR p_run.status IS DISTINCT FROM 'failed'
  OR p_run.terminal_cause IS NULL OR p_run.terminal_cause NOT IN('provider_error','stage_closed') THEN
  RETURN NULL;
 END IF;
 SELECT * INTO issuance FROM public.lab_arena_ledger l WHERE l.run_id=p_run.run_id
  AND l.provider='openrouter' AND l.entry_kind='reservation'
  ORDER BY (l.entry_doc->'action_sequence') DESC NULLS FIRST,
   (l.entry_doc->'provider_attempt') DESC NULLS LAST,l.entry_id DESC LIMIT 1;
 IF NOT FOUND OR (issuance.round_id,issuance.submission_id,issuance.miner_hotkey,issuance.stage)
  IS DISTINCT FROM (p_run.round_id,p_run.submission_id,p_run.miner_hotkey,p_run.stage) THEN
  RETURN NULL;
 END IF;
 -- The private canonical proof validates the latest native call lifecycle and
 -- exact data-policy 404. Provider prose and bodies never leave this function.
 IF public.lab_arena__execution_privacy_route_unavailable_v1(p_run.run_id) THEN
  RETURN 'openrouter_zdr_route_unavailable';
 END IF;
 RETURN NULL;
END;
$body541$;
ALTER FUNCTION public.lab_arena_public_execution_failure_reason_v1(public.lab_arena_runs) OWNER TO lab_arena_owner;
REVOKE ALL ON FUNCTION public.lab_arena_public_execution_failure_reason_v1(public.lab_arena_runs)
 FROM PUBLIC,anon,authenticated,service_role;
GRANT EXECUTE ON FUNCTION public.lab_arena_public_execution_failure_reason_v1(public.lab_arena_runs) TO lab_arena_service;
DO $restore541$
BEGIN
 IF NOT (SELECT had_create FROM pg_temp.arena541_schema_acl) THEN
  REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
 END IF;
 IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')
  IS DISTINCT FROM (SELECT nspacl FROM pg_temp.arena541_schema_acl) THEN
  RAISE EXCEPTION 'Arena public diagnostic schema ACL changed';
 END IF;
END;
$restore541$;
NOTIFY pgrst,'reload schema';
COMMIT;

-- Close only never-dispatched reservations when a valid current lease completes.
-- Preserve dispatched/uncertain calls, charges, accepted work and scoring rules.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $completion_reservation_closeout493$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_complete_attempt(text,text,jsonb,text,text)';
  v_preimage CONSTANT TEXT := 'dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b';
  v_postimage CONSTANT TEXT := '084e1afa3a4c277427d05c5ca509f235ff5814c7ecdae1b9dd29c39ec6f56073';
  v_old CONSTANT TEXT := $old493$  SELECT COUNT(*) INTO v_open FROM ($old493$;
  v_new CONSTANT TEXT := $new493$  -- The current lease/run lock serializes this closeout with dispatch.
  -- A committed dispatch remains open; a later marker sees recovery or stale.
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source, amount_microusd, entry_doc
  )
  SELECT 'recovery', head.miner_hotkey, head.round_id, head.submission_id,
    head.run_id, head.stage, head.call_identity, head.provider, head.operation_id,
    head.funding_source, 0, pg_catalog.jsonb_build_object(
      'reason', 'completion_before_dispatch',
      'reserved_microusd', head.amount_microusd)
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.*
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS head
  WHERE head.entry_kind = 'reservation';
  SELECT COUNT(*) INTO v_open FROM ($new493$;
  v_oid OID;
  v_definition TEXT;
  v_updated TEXT;
  v_identity JSONB;
  v_metadata JSONB;
BEGIN
  v_oid:=pg_catalog.to_regprocedure(v_signature);
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig),
    pg_catalog.to_jsonb(p)-'prosrc'
  INTO v_definition,v_identity,v_metadata
  FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=v_oid;
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner',
    '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena completion493 security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
       pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena completion493 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_updated,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'Arena completion493 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_updated;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(v_oid),'sha256'),'hex')
      <>v_postimage OR (SELECT pg_catalog.to_jsonb(p)-'prosrc'
        FROM pg_catalog.pg_proc p WHERE p.oid=v_oid) IS DISTINCT FROM v_metadata THEN
    RAISE EXCEPTION 'Arena completion493 readback differs' USING ERRCODE='55000';
  END IF;
END;
$completion_reservation_closeout493$;
COMMIT;

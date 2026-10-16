-- Complete finished attempts without waiting for unused reservations to expire.
-- Dispatched calls and all terminal accounting receipts remain unchanged.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $unused_reservation_completion493$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_complete_attempt(text,text,jsonb,text,text)';
  v_preimage CONSTANT TEXT := 'dbee6214ac7404bafee705fafa7380dad4d6fb1f6095f3904e91376228bc297b';
  v_postimage CONSTANT TEXT := '20da61fe353b6b7ceba81eb6deabf6597e1b6db42efd5ceef0219539d004171b';
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
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage THEN
    RAISE EXCEPTION 'Arena completion493 preimage differs' USING ERRCODE='55000';
  END IF;
  v_updated:=pg_catalog.replace(v_definition,$old493input$  IF pg_catalog.jsonb_typeof(p_result) IS DISTINCT FROM 'object'$old493input$,
    $new493input$  IF p_terminal_cause IS NULL
     OR pg_catalog.jsonb_typeof(p_result) IS DISTINCT FROM 'object'$new493input$);
  v_updated:=pg_catalog.replace(v_updated,$old493$  SELECT COUNT(*) INTO v_open FROM ($old493$,
    $new493$  -- The current lease holds the run lock shared with mark_dispatched.
  -- Only a reservation head proves that no provider request was dispatched.
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage, call_identity,
    provider, operation_id, funding_source, amount_microusd, entry_doc
  )
  SELECT 'recovery', heads.miner_hotkey, heads.round_id, heads.submission_id,
    heads.run_id, heads.stage, heads.call_identity, heads.provider,
    heads.operation_id, heads.funding_source, 0,
    pg_catalog.jsonb_build_object(
      'reason', 'attempt_completed_before_dispatch',
      'reserved_microusd', heads.amount_microusd)
  FROM (
    SELECT DISTINCT ON (ledger.call_identity) ledger.*
    FROM public.lab_arena_ledger AS ledger
    WHERE ledger.run_id = p_run_id AND ledger.call_identity IS NOT NULL
    ORDER BY ledger.call_identity, ledger.entry_id DESC
  ) AS heads WHERE heads.entry_kind = 'reservation';
  SELECT COUNT(*) INTO v_open FROM ($new493$);
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
$unused_reservation_completion493$;

COMMIT;

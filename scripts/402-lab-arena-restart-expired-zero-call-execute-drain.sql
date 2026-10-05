-- Continue guarded natural expiry for captured zero-ledger execute leases in
-- their current execution stage. Preserve every receipt and dormant retry.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $restart_expired_zero_call_execute_drain$
DECLARE
  v_drain TEXT;
  v_quiescence TEXT;
  v_drain_owner NAME;
  v_quiescence_owner NAME;
  v_drain_security BOOLEAN;
  v_quiescence_security BOOLEAN;
  v_drain_volatility "char";
  v_quiescence_volatility "char";
  v_drain_config TEXT[];
  v_quiescence_config TEXT[];
  v_drain_preimage CONSTANT TEXT := '183fbb7525dd7561451bb386960ef9e602efd9c205dce754683c9e60ead2ee47';
  v_quiescence_preimage CONSTANT TEXT := 'c3a87110def2cb47833ec4bc353c5c6a965c655f0f03c490019fd34d5de4dcfb';
  v_drain_postimage CONSTANT TEXT := '9515a96a25174d1bb5daba23f9920f7c665edf11aa6f9d73570ecd87f3c47707';
  v_quiescence_postimage CONSTANT TEXT := '3a4f36e387c478abbd09805639a56e1e71b0c1d927605439399b5e133ebe5d45';
  v_old_receipt CONSTANT TEXT := '          AND runs.kind = ''score''';
  v_new_receipt CONSTANT TEXT := '          AND runs.kind IN (''score'', ''execute'')';
  v_old_write_set CONSTANT TEXT := $old_write_set$          AND (overdue.kind<>'score'
            OR round.status NOT IN ('stage1_scoring','stage2_scoring')
            OR overdue.stage_generation<>round.stage_generation$old_write_set$;
  v_new_write_set CONSTANT TEXT := $new_write_set$          AND (NOT (
            (overdue.kind='score'
              AND round.status IN ('stage1_scoring','stage2_scoring'))
            OR (overdue.kind='execute'
              AND ((round.status='stage1' AND overdue.stage=1)
                OR (round.status='stage2' AND overdue.stage=2)))
          )
            OR overdue.stage_generation<>round.stage_generation$new_write_set$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_drain, v_drain_owner, v_drain_security,
       v_drain_volatility, v_drain_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena__restart_drain_state_v1'
    AND p.pronargs=0;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_quiescence, v_quiescence_owner, v_quiescence_security,
       v_quiescence_volatility, v_quiescence_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena_restart_quiescence_v1'
    AND p.pronargs=3;
  IF v_drain IS NULL OR v_quiescence IS NULL
     OR v_drain_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_quiescence_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_drain_security IS DISTINCT FROM TRUE
     OR v_quiescence_security IS DISTINCT FROM TRUE
     OR v_drain_volatility IS DISTINCT FROM 's'
     OR v_quiescence_volatility IS DISTINCT FROM 'v'
     OR v_drain_config IS NULL OR v_quiescence_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_drain_config))
     OR NOT ('search_path=pg_catalog, public'=ANY(v_quiescence_config)) THEN
    RAISE EXCEPTION 'Arena restart execute expiry function security shape differs'
      USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')=v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')=v_quiescence_postimage THEN
    IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
       OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
      RAISE EXCEPTION 'Arena restart execute expiry replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_preimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_preimage THEN
    RAISE EXCEPTION 'Arena restart execute expiry preimage differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_drain,v_old_receipt)=0
     OR pg_catalog.strpos(pg_catalog.substr(v_drain,
       pg_catalog.strpos(v_drain,v_old_receipt)+pg_catalog.length(v_old_receipt)),v_old_receipt)>0
     OR pg_catalog.strpos(v_quiescence,v_old_write_set)=0
     OR pg_catalog.strpos(pg_catalog.substr(v_quiescence,
       pg_catalog.strpos(v_quiescence,v_old_write_set)+pg_catalog.length(v_old_write_set)),v_old_write_set)>0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_guard_owned_natural_expiry_v1')=0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_restart_guard_owner_or_generation_differs')=0 THEN
    RAISE EXCEPTION 'Arena restart execute expiry anchor differs' USING ERRCODE='55000';
  END IF;
  v_drain := pg_catalog.replace(v_drain,v_old_receipt,v_new_receipt);
  v_quiescence := pg_catalog.replace(v_quiescence,v_old_write_set,v_new_write_set);
  EXECUTE v_drain;
  EXECUTE v_quiescence;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE),
    pg_catalog.pg_get_functiondef(
    'public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE)
  INTO v_drain,v_quiescence;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
    RAISE EXCEPTION 'Arena restart execute expiry postimage differs' USING ERRCODE='55000';
  END IF;
END;
$restart_expired_zero_call_execute_drain$;

NOTIFY pgrst, 'reload schema';
COMMIT;

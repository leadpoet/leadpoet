-- Permit captured expired leases with terminal provider-call heads to follow
-- ordinary expiry. Open reservations and dispatches still block the drain.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $restart_expired_closed_ledger_drain$
DECLARE
  v_drain TEXT;
  v_quiescence TEXT;
  v_expire TEXT;
  v_terminate TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_drain_preimage CONSTANT TEXT := '9515a96a25174d1bb5daba23f9920f7c665edf11aa6f9d73570ecd87f3c47707';
  v_quiescence_preimage CONSTANT TEXT := '3a4f36e387c478abbd09805639a56e1e71b0c1d927605439399b5e133ebe5d45';
  v_drain_postimage CONSTANT TEXT := 'd36a3c7ca0ca92b03f50ee7ca97fba3580ff7a50ab20b8c2e5e44f443a1bdbdb';
  v_quiescence_postimage CONSTANT TEXT := 'dbb9f5270a190b6604ca8ed77abf5577bff2a58b11ec39820cadba3e662e214c';
  v_expire_preimage CONSTANT TEXT := '21634a503b8526bd51dbe102bb89b0bbcbdef582057070ae1d66324b0ebc0532';
  v_terminate_preimage CONSTANT TEXT := '6d59ebe2f7dc254bc190ddbb6b5b1f3cc0df0f46807b0a029c44f123d9dcbca1';
  v_old_receipt CONSTANT TEXT := $old_receipt$          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS expired_cost
            WHERE expired_cost.run_id = runs.run_id
          ) THEN 'expired'$old_receipt$;
  v_new_receipt CONSTANT TEXT := $new_receipt$          AND NOT EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id = runs.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity, ledger.entry_id DESC
            ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch')
          ) THEN 'expired'$new_receipt$;
  v_old_block CONSTANT TEXT := $old_block$      -- Cost-bearing work may still complete and settle. Never apply the
      -- whole-round expiry RPC while any of its leased rows has ledger data.
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS leased
        WHERE leased.round_id=v_round_id AND leased.status='leased'
          AND EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS cost
            WHERE cost.run_id=leased.run_id
          )
      ) THEN
        CONTINUE;
      END IF;$old_block$;
  v_new_block CONSTANT TEXT := $new_block$      -- Closed call heads cannot be changed by ordinary expiry. A live
      -- reservation or dispatch must finish before this whole-round RPC.
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS leased
        WHERE leased.round_id=v_round_id AND leased.status='leased'
          AND EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id=leased.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity, ledger.entry_id DESC
            ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch')
          )
      ) THEN
        CONTINUE;
      END IF;$new_block$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_drain, v_owner, v_acl, v_security, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner}'::ACLITEM[]
     OR v_security IS DISTINCT FROM TRUE OR v_volatility IS DISTINCT FROM 's'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena closed-ledger drain security shape differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_quiescence, v_owner, v_acl, v_security, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner,service_role=X/lab_arena_owner}'::ACLITEM[]
     OR v_security IS DISTINCT FROM TRUE OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena closed-ledger quiescence security shape differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_expire,v_owner,v_acl,v_security,v_volatility,v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security IS DISTINCT FROM TRUE OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena closed-ledger expiry security shape differs' USING ERRCODE='55000';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_terminate,v_owner,v_acl,v_security,v_volatility,v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE p.oid='public.lab_arena__terminate_open_calls(text,text)'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner}'::ACLITEM[]
     OR v_security IS DISTINCT FROM TRUE OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config)) THEN
    RAISE EXCEPTION 'Arena closed-ledger call termination security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_expire,'sha256'),'hex')<>v_expire_preimage
     OR pg_catalog.encode(extensions.digest(v_terminate,'sha256'),'hex')<>v_terminate_preimage THEN
    RAISE EXCEPTION 'Arena closed-ledger accounting preimage differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')=v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')=v_quiescence_postimage THEN
    IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
       OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
      RAISE EXCEPTION 'Arena closed-ledger drain replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_preimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_preimage
     OR pg_catalog.strpos(v_drain,v_old_receipt)=0
     OR pg_catalog.strpos(v_quiescence,v_old_block)=0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_guard_owned_natural_expiry_v1')=0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_restart_guard_owner_or_generation_differs')=0 THEN
    RAISE EXCEPTION 'Arena closed-ledger drain preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_drain,v_old_receipt,v_new_receipt);
  EXECUTE pg_catalog.replace(v_quiescence,v_old_block,v_new_block);
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE),
         pg_catalog.pg_get_functiondef('public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE)
  INTO v_drain,v_quiescence;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
    RAISE EXCEPTION 'Arena closed-ledger drain postimage differs' USING ERRCODE='55000';
  END IF;
END;
$restart_expired_closed_ledger_drain$;

NOTIFY pgrst, 'reload schema';
COMMIT;

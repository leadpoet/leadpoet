-- Let the guarded restart use ordinary expiry for captured leases whose
-- provider accounting is already terminal. Existing ledger receipts remain
-- append-only; open reservations and dispatches still hold the drain.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $restart_closed_ledger_expiry$
DECLARE
  v_drain TEXT;
  v_quiescence TEXT;
  v_drain_owner NAME;
  v_quiescence_owner NAME;
  v_drain_acl ACLITEM[];
  v_quiescence_acl ACLITEM[];
  v_drain_security BOOLEAN;
  v_quiescence_security BOOLEAN;
  v_drain_volatility "char";
  v_quiescence_volatility "char";
  v_drain_config TEXT[];
  v_quiescence_config TEXT[];
  v_drain_preimage CONSTANT TEXT := '9515a96a25174d1bb5daba23f9920f7c665edf11aa6f9d73570ecd87f3c47707';
  v_quiescence_preimage CONSTANT TEXT := '3a4f36e387c478abbd09805639a56e1e71b0c1d927605439399b5e133ebe5d45';
  v_drain_postimage CONSTANT TEXT := 'b9cfb9b59ef428d8e43b052258cac72f1fff3d141a736080c8012d644211497b';
  v_quiescence_postimage CONSTANT TEXT := '94c8c23deeccd6b89bb575b0d2782709b0c030810e00bef5691b254b5bb84d35';
  v_old_receipt CONSTANT TEXT := $old_receipt$          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS expired_cost
            WHERE expired_cost.run_id = runs.run_id
          ) THEN 'expired'$old_receipt$;
  v_new_receipt CONSTANT TEXT := $new_receipt$          -- lab_arena_closed_ledger_expired_receipt_v1
          AND NOT EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id = runs.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity, ledger.entry_id DESC
            ) AS heads WHERE heads.entry_kind IN ('reservation', 'dispatch')
          )
          AND NOT EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS unidentified
            WHERE unidentified.run_id = runs.run_id
              AND unidentified.call_identity IS NULL
          ) THEN 'expired'$new_receipt$;
  v_old_hold CONSTANT TEXT := $old_hold$      -- Cost-bearing work may still complete and settle. Never apply the
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
      END IF;$old_hold$;
  v_new_hold CONSTANT TEXT := $new_hold$      -- lab_arena_guard_closed_ledger_expiry_v1: ordinary expiry touches
      -- every overdue row in this round. Keep open provider accounting held;
      -- the write-set check above already proves capture and stage identity.
      IF EXISTS (
        SELECT 1 FROM public.lab_arena_runs AS leased
        WHERE leased.round_id=v_round_id AND leased.status='leased'
          AND (EXISTS (
            SELECT 1 FROM (
              SELECT DISTINCT ON (ledger.call_identity) ledger.entry_kind
              FROM public.lab_arena_ledger AS ledger
              WHERE ledger.run_id=leased.run_id
                AND ledger.call_identity IS NOT NULL
              ORDER BY ledger.call_identity,ledger.entry_id DESC
            ) AS heads WHERE heads.entry_kind IN ('reservation','dispatch')
          ) OR EXISTS (
            SELECT 1 FROM public.lab_arena_ledger AS unidentified
            WHERE unidentified.run_id=leased.run_id
              AND unidentified.call_identity IS NULL
          ))
      ) THEN
        CONTINUE;
      END IF;$new_hold$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_drain, v_drain_owner, v_drain_acl, v_drain_security,
       v_drain_volatility, v_drain_config
  FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena__restart_drain_state_v1'
    AND p.pronargs=0;
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname,
         p.proacl, p.prosecdef, p.provolatile, p.proconfig
  INTO v_quiescence, v_quiescence_owner, v_quiescence_acl,
       v_quiescence_security, v_quiescence_volatility, v_quiescence_config
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
    RAISE EXCEPTION 'Arena restart closed-ledger function security shape differs'
      USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')=v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')=v_quiescence_postimage THEN
    IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
       OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
      RAISE EXCEPTION 'Arena restart closed-ledger replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_preimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_preimage
     OR pg_catalog.strpos(v_drain,v_old_receipt)=0
     OR pg_catalog.strpos(pg_catalog.substr(v_drain,
       pg_catalog.strpos(v_drain,v_old_receipt)+pg_catalog.length(v_old_receipt)),v_old_receipt)>0
     OR pg_catalog.strpos(v_quiescence,v_old_hold)=0
     OR pg_catalog.strpos(pg_catalog.substr(v_quiescence,
       pg_catalog.strpos(v_quiescence,v_old_hold)+pg_catalog.length(v_old_hold)),v_old_hold)>0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_guard_owned_natural_expiry_v1')=0
     OR pg_catalog.strpos(v_quiescence,'lab_arena_restart_guard_owner_or_generation_differs')=0 THEN
    RAISE EXCEPTION 'Arena restart closed-ledger preimage differs' USING ERRCODE='55000';
  END IF;
  v_drain := pg_catalog.replace(v_drain,v_old_receipt,v_new_receipt);
  v_quiescence := pg_catalog.replace(v_quiescence,v_old_hold,v_new_hold);
  EXECUTE v_drain;
  EXECUTE v_quiescence;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE),
    pg_catalog.pg_get_functiondef(
    'public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE)
  INTO v_drain,v_quiescence;
  IF pg_catalog.encode(extensions.digest(v_drain,'sha256'),'hex')<>v_drain_postimage
     OR pg_catalog.encode(extensions.digest(v_quiescence,'sha256'),'hex')<>v_quiescence_postimage THEN
    RAISE EXCEPTION 'Arena restart closed-ledger postimage differs'
      USING ERRCODE='55000';
  END IF;
  IF (SELECT p.proacl FROM pg_catalog.pg_proc AS p
      WHERE p.oid='public.lab_arena__restart_drain_state_v1()'::REGPROCEDURE)
       IS DISTINCT FROM v_drain_acl
     OR (SELECT p.proacl FROM pg_catalog.pg_proc AS p
      WHERE p.oid='public.lab_arena_restart_quiescence_v1(text,text,bigint)'::REGPROCEDURE)
       IS DISTINCT FROM v_quiescence_acl THEN
    RAISE EXCEPTION 'Arena restart closed-ledger permissions changed'
      USING ERRCODE='55000';
  END IF;
END;
$restart_closed_ledger_expiry$;

NOTIFY pgrst, 'reload schema';
COMMIT;

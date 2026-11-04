-- Release only a current lease's exact reservation before provider dispatch.
-- Reuse recovery accounting; never invent a dispatch or a zero-cost settlement.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

-- Only a fresh dispatch reply grants its caller ownership of this marker.
DO $dispatch_owner619$
DECLARE
  signature CONSTANT TEXT := 'public.lab_arena_mark_dispatched(text,text,text)';
  definition TEXT; updated TEXT; security_before JSONB;
BEGIN
  SELECT pg_get_functiondef(p.oid), jsonb_build_array(o.rolname,p.proacl::TEXT,
    p.prosecdef,p.provolatile,p.proconfig) INTO definition,security_before
  FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
  WHERE p.oid=to_regprocedure(signature);
  IF security_before IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'Arena dispatch619 security shape differs' USING ERRCODE='55000';
  END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex')='0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab' THEN RETURN; END IF;
  IF encode(extensions.digest(definition,'sha256'),'hex') IS DISTINCT FROM '3149e4a6f481ca8dee40831b811674f7c540936d8d73ae346f55b0a1eea4f12f' THEN
    RAISE EXCEPTION 'Arena dispatch619 preimage differs' USING ERRCODE='55000';
  END IF;
  updated:=replace(definition,$old619decl$  v_head public.lab_arena_ledger;$old619decl$,
    $new619decl$  v_head public.lab_arena_ledger;
  v_dispatch_entry_id BIGINT;$new619decl$);
  updated:=replace(updated,$old619insert$    pg_catalog.jsonb_build_object('dispatched_at', pg_catalog.clock_timestamp())
  );$old619insert$,$new619insert$    pg_catalog.jsonb_build_object('dispatched_at', pg_catalog.clock_timestamp())
  ) RETURNING entry_id INTO v_dispatch_entry_id;$new619insert$);
  updated:=replace(updated,$old619reply$    'call_identity', p_call_identity, 'amount_microusd', v_head.amount_microusd,$old619reply$,
    $new619reply$    'call_identity', p_call_identity, 'amount_microusd', v_head.amount_microusd,
    'dispatch_entry_id', v_dispatch_entry_id,$new619reply$);
  IF encode(extensions.digest(updated,'sha256'),'hex') IS DISTINCT FROM '0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab' THEN
    RAISE EXCEPTION 'Arena dispatch619 postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE updated;
  IF encode(extensions.digest(pg_get_functiondef(to_regprocedure(signature)),'sha256'),'hex')
      IS DISTINCT FROM '0b7b0f601204f13e2a65fd218a1c6286066aa797dd2ccfeaaa24f2954eb436ab' OR (SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,
        p.prosecdef,p.provolatile,p.proconfig) FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner
        WHERE p.oid=to_regprocedure(signature)) IS DISTINCT FROM security_before THEN
    RAISE EXCEPTION 'Arena dispatch619 readback differs' USING ERRCODE='55000';
  END IF;
END;
$dispatch_owner619$;
CREATE TEMP TABLE release619_schema ON COMMIT DROP AS
  SELECT nspacl,has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
  FROM pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_release_undispatched_call(
  p_run_id TEXT, p_lease_token_hash TEXT, p_call_identity TEXT,
  p_call_doc JSONB, p_reason TEXT, p_dispatch_entry_id BIGINT DEFAULT NULL
)
RETURNS JSONB
LANGUAGE plpgsql VOLATILE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $release_undispatched$
DECLARE
  v_run public.lab_arena_runs;
  v_head public.lab_arena_ledger;
BEGIN
  IF pg_catalog.jsonb_typeof(p_call_doc) IS DISTINCT FROM 'object'
     OR p_reason IS NULL OR p_reason NOT IN (
       'provider_admission_cancelled', 'provider_admission_deadline')
     OR (p_dispatch_entry_id IS NOT NULL AND p_dispatch_entry_id < 1) THEN
    RAISE EXCEPTION 'lab_arena_release_undispatched_input_invalid' USING ERRCODE = '22023';
  END IF;
  -- Same run lock and lock order as mark_dispatched. Dispatch and release
  -- cannot both win, including a repeated/lost RPC reply.
  BEGIN
    v_run := public.lab_arena__lock_current_lease(p_run_id, p_lease_token_hash);
  EXCEPTION WHEN SQLSTATE 'P0003' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'stale');
  END;
  v_head := public.lab_arena__ledger_head(p_call_identity);
  IF v_head.entry_id IS NULL OR v_head.run_id <> p_run_id
     OR v_head.provider <> 'openrouter' THEN
    RETURN pg_catalog.jsonb_build_object('status', 'not_reserved');
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.lab_arena_ledger
    WHERE call_identity = p_call_identity AND run_id = p_run_id
      AND entry_kind = 'reservation' AND entry_doc = p_call_doc) THEN
    RETURN pg_catalog.jsonb_build_object('status', 'reservation_mismatch');
  END IF;
  -- The broker can supply this ID only from its fresh dispatch reply, and
  -- only before entering transport.send. A lost/idempotent reply is not proof.
  IF NOT (v_head.entry_kind = 'reservation' AND p_dispatch_entry_id IS NULL)
     AND NOT (v_head.entry_kind = 'dispatch' AND p_dispatch_entry_id IS NOT NULL
              AND p_dispatch_entry_id = v_head.entry_id) THEN
    RETURN public.lab_arena__call_state_view(v_head, v_run);
  END IF;
  INSERT INTO public.lab_arena_ledger (
    entry_kind, miner_hotkey, round_id, submission_id, run_id, stage,
    call_identity, provider, operation_id, funding_source, amount_microusd, entry_doc
  ) VALUES (
    'recovery', v_head.miner_hotkey, v_head.round_id, v_head.submission_id,
    p_run_id, v_head.stage, p_call_identity, v_head.provider, v_head.operation_id,
    v_head.funding_source, 0, pg_catalog.jsonb_build_object(
      'reason', p_reason, 'reserved_microusd', v_head.amount_microusd)
      || CASE WHEN p_dispatch_entry_id IS NOT NULL THEN pg_catalog.jsonb_build_object(
        'transport_not_started', TRUE, 'undispatched_dispatch_entry_id', p_dispatch_entry_id)
        ELSE '{}'::JSONB END
  );
  RETURN pg_catalog.jsonb_build_object('status', 'recovered', 'idempotent', FALSE,
    'call_identity', p_call_identity, 'amount_microusd', 0);
END;
$release_undispatched$;
ALTER FUNCTION public.lab_arena_release_undispatched_call(TEXT,TEXT,TEXT,JSONB,TEXT,BIGINT)
  OWNER TO lab_arena_owner;
DO $release_acl619$
DECLARE target REGPROCEDURE := 'public.lab_arena_release_undispatched_call(text,text,text,jsonb,text,bigint)'::REGPROCEDURE;
  grantee TEXT;
BEGIN
  FOR grantee IN SELECT DISTINCT CASE WHEN a.grantee=0 THEN 'PUBLIC' ELSE quote_ident(r.rolname) END
    FROM pg_proc p, aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a
    LEFT JOIN pg_roles r ON r.oid=a.grantee
    WHERE p.oid=target AND a.grantee<>p.proowner
  LOOP
    EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %s',target,grantee);
  END LOOP;
  GRANT EXECUTE ON FUNCTION public.lab_arena_release_undispatched_call(TEXT,TEXT,TEXT,JSONB,TEXT,BIGINT)
    TO lab_arena_service;
  IF (SELECT jsonb_build_array(o.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
      FROM pg_proc p JOIN pg_roles o ON o.oid=p.proowner WHERE p.oid=target)
      IS DISTINCT FROM jsonb_build_array('lab_arena_owner',
        '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
        TRUE,'v',ARRAY['search_path=pg_catalog, public']) THEN
    RAISE EXCEPTION 'Arena release619 security shape differs' USING ERRCODE='55000';
  END IF;
  IF NOT (SELECT had_create FROM pg_temp.release619_schema) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_namespace WHERE nspname='public')
      IS DISTINCT FROM (SELECT nspacl FROM pg_temp.release619_schema) THEN
    RAISE EXCEPTION 'Arena release619 schema ACL changed' USING ERRCODE='55000';
  END IF;
END;
$release_acl619$;
NOTIFY pgrst, 'reload schema';
COMMIT;

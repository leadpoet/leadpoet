-- Hold early host recovery during canonical restart capture or operator pause.
-- Ordinary natural expiry and its captured receipts remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $abandoned_host_restart_guard$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_hash TEXT;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname, p.proacl,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_roles AS owner ON owner.oid = p.proowner
  WHERE p.oid = 'public.lab_arena_expire_leases(text)'::REGPROCEDURE;
  IF v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::TEXT[] THEN
    RAISE EXCEPTION 'Arena abandoned host restart guard security shape differs'
      USING ERRCODE = '55000';
  END IF;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '823ce89127543ab7d59dd642903b4b5d9a5dd0bfb79fde412fb3a0963b8e81da' THEN
    RETURN;
  END IF;
  IF v_hash <> 'ac1e99fd5bc1b4a458602ce05012d562847bb37d3cf4e69857e45db19077fd00' THEN
    RAISE EXCEPTION 'Arena abandoned host restart guard preimage differs'
      USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $old0$  v_retried INTEGER := 0;$old0$,
    $new0$  v_retried INTEGER := 0;
  v_early_recovery_allowed BOOLEAN;$new0$);
  v_definition := pg_catalog.replace(v_definition,
    $old1$BEGIN
  SELECT * INTO v_round$old1$,
    $new1$BEGIN
  -- lab_arena_abandoned_host_restart_guard_v1: serialize with restart capture
  -- and operator holds before taking the ordinary expiry round/run locks.
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('lab-arena-claim-control', 0)
  );
  SELECT NOT (operator_paused OR guard_commitment <> '')
  INTO v_early_recovery_allowed
  FROM public.lab_arena_restart_claim_control WHERE singleton;
  IF NOT FOUND THEN
    RAISE EXCEPTION 'lab_arena_claim_control_missing' USING ERRCODE = '55000';
  END IF;
  SELECT * INTO v_round$new1$);
  v_definition := pg_catalog.replace(v_definition,
    $old2$WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'$old2$,
    $new2$WHERE candidate.round_id = p_round_id AND candidate.status = 'leased'
      -- A captured lease must retain its natural-expiry receipt shape.
      -- A stale but unreleased guard also holds work, as in the claim gate.
      AND v_early_recovery_allowed$new2$);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex')
       <> '823ce89127543ab7d59dd642903b4b5d9a5dd0bfb79fde412fb3a0963b8e81da' THEN
    RAISE EXCEPTION 'Arena abandoned host restart guard postimage differs'
      USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
END;
$abandoned_host_restart_guard$;
COMMIT;

-- Keep an authenticated host fault in the claim guard after lease expiry,
-- even before the normal expiry tick changes the run from leased to failed.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $untransitioned_host_fault_guard$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_old CONSTANT TEXT := $old$           (host_fault.status='leased'
             AND host_fault.lease_expires_at>pg_catalog.clock_timestamp())$old$;
  v_new CONSTANT TEXT := $new$           (host_fault.status='leased'
             -- lab_arena_untransitioned_host_fault_guard_v1: retain the
             -- lease-bound fault through its frozen-TTL expiry window.
             AND host_fault.lease_expires_at>
               pg_catalog.clock_timestamp()-pg_catalog.make_interval(
                 secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid), owner.rolname, p.proacl,
         p.prosecdef, p.provolatile, p.proconfig
  INTO v_definition, v_owner, v_acl, v_security_definer, v_volatility, v_config
  FROM pg_catalog.pg_proc AS p
  JOIN pg_catalog.pg_namespace AS n ON n.oid=p.pronamespace
  JOIN pg_catalog.pg_roles AS owner ON owner.oid=p.proowner
  WHERE n.nspname='public' AND p.proname='lab_arena_claim_assignment'
    AND p.pronargs=9;
  IF v_definition IS NULL
     OR v_owner IS DISTINCT FROM 'lab_arena_owner'
     OR v_acl IS DISTINCT FROM '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config))
     OR pg_catalog.strpos(v_definition,'lab_arena_active_host_fault_claim_guard_v1')=0 THEN
    RAISE EXCEPTION 'Arena untransitioned host fault claim security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,'lab_arena_untransitioned_host_fault_guard_v1')>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_new,'')))
         <> pg_catalog.length(v_new)
       OR pg_catalog.strpos(v_definition,v_old)>0 THEN
      RAISE EXCEPTION 'Arena untransitioned host fault claim replay differs' USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
       IS DISTINCT FROM '6c977cc24ce500a76ba97e94ab0be50572d6395172109267c156d2de6f461cbc'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(pg_catalog.replace(v_definition,v_old,'')))
          <> pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'Arena untransitioned host fault claim function preimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
END;
$untransitioned_host_fault_guard$;

COMMIT;

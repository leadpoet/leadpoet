-- Keep a runner with three authenticated, zero-call host faults out of the
-- current round. The next round resets the guard without changing leases.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $round_host_fault_quarantine$
DECLARE
  v_definition TEXT;
  v_owner NAME;
  v_acl ACLITEM[];
  v_security_definer BOOLEAN;
  v_volatility "char";
  v_config TEXT[];
  v_old_scope CONSTANT TEXT := $old$    -- lab_arena_active_host_fault_claim_guard_v1: three distinct
    -- current or recently expired lease-bound host faults stop new claims.
    AND (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage2_scoring')
      OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS host_fault
       WHERE host_fault.round_id=p_round_id
         AND host_fault.stage=v_stage
         AND host_fault.stage_generation=v_round.stage_generation
         AND host_fault.kind=CASE
           WHEN v_round.status IN ('stage1', 'stage2') THEN 'execute'
           ELSE 'score' END
         AND host_fault.runner_hotkey=p_runner_hotkey
$old$;
  v_new_scope CONSTANT TEXT := $new$    -- lab_arena_active_host_fault_claim_guard_v1: three distinct
    -- authenticated, zero-call host faults stop new claims.
    -- lab_arena_round_host_fault_quarantine_v1: count both kinds and stages
    -- in this round; a new round_id resets the runner's eligibility.
    AND (v_round.status NOT IN ('stage1', 'stage2', 'stage1_scoring', 'stage2_scoring')
      OR NOT EXISTS (
       SELECT 1 FROM public.lab_arena_runs AS host_fault
       WHERE host_fault.round_id=p_round_id
         AND host_fault.runner_hotkey=p_runner_hotkey
$new$;
  v_old_status CONSTANT TEXT := $old$         AND (
           (host_fault.status='leased'
             -- lab_arena_untransitioned_host_fault_guard_v1: retain the
             -- lease-bound fault through its frozen-TTL expiry window.
             AND host_fault.lease_expires_at>
               pg_catalog.clock_timestamp()-pg_catalog.make_interval(
                 secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
           OR (host_fault.status='failed'
             AND host_fault.terminal_cause='lease_expired'
             AND host_fault.lease_expires_at<=pg_catalog.clock_timestamp()
             AND host_fault.lease_expires_at>
               pg_catalog.clock_timestamp()-pg_catalog.make_interval(
                 secs => (v_round.configuration_doc->>'lease_ttl_seconds')::INTEGER))
         )
$old$;
  v_new_status CONSTANT TEXT := $new$         AND (
           (host_fault.status='leased'
             -- lab_arena_untransitioned_host_fault_guard_v1: a signed
             -- runtime fault remains evidence before the expiry tick.
             AND host_fault.lease_expires_at IS NOT NULL)
           OR (host_fault.status='failed'
             AND host_fault.terminal_cause='lease_expired'
             AND host_fault.lease_expires_at<=pg_catalog.clock_timestamp())
         )
$new$;
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
     OR v_acl IS DISTINCT FROM
       '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}'::ACLITEM[]
     OR v_security_definer IS DISTINCT FROM TRUE
     OR v_volatility IS DISTINCT FROM 'v'
     OR v_config IS NULL
     OR NOT ('search_path=pg_catalog, public'=ANY(v_config))
     OR pg_catalog.strpos(v_definition,'lab_arena_score_submission_serialization')=0
     OR pg_catalog.strpos(v_definition,'lab_arena_execute_host_cooldown_v1')=0 THEN
    RAISE EXCEPTION 'Arena round host fault claim security shape differs'
      USING ERRCODE='55000';
  END IF;
  IF pg_catalog.strpos(v_definition,'lab_arena_round_host_fault_quarantine_v1')>0 THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_new_scope,'')))
         <> pg_catalog.length(v_new_scope)
       OR (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_new_status,'')))
         <> pg_catalog.length(v_new_status)
       OR pg_catalog.strpos(v_definition,v_old_scope)>0
       OR pg_catalog.strpos(v_definition,v_old_status)>0 THEN
      RAISE EXCEPTION 'Arena round host fault claim replay differs'
        USING ERRCODE='55000';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')
       IS DISTINCT FROM 'f1f82a1dc510e9339ad76e0b08ad29d8dad3d8194d99700603d359b5d85ac7f7'
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_old_scope,'')))
          <> pg_catalog.length(v_old_scope)
     OR (pg_catalog.length(v_definition)-pg_catalog.length(
          pg_catalog.replace(v_definition,v_old_status,'')))
          <> pg_catalog.length(v_old_status) THEN
    RAISE EXCEPTION 'Arena round host fault claim preimage differs'
      USING ERRCODE='55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,v_old_scope,v_new_scope);
  EXECUTE pg_catalog.replace(v_definition,v_old_status,v_new_status);
END;
$round_host_fault_quarantine$;

COMMIT;

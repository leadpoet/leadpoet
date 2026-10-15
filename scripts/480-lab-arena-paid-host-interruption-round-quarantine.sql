-- A paid execute host interruption still counts toward the existing
-- three-distinct-assignment, same-runner, same-round claim quarantine.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';

DO $paid_host_quarantine$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)';
  v_preimage CONSTANT TEXT := '23c29c40dd0f91e16035675c13a2e1c64046c81a447c372f50f1e94391983199';
  v_postimage CONSTANT TEXT := '8ea400f297a8a1da73536098a9e216d1017255d1a963c24cc9129ccb1445736c';
  v_definition TEXT;
  v_identity JSONB;
  v_old CONSTANT TEXT := $old$        AND NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS prior_cost
          WHERE prior_cost.run_id=host_fault.run_id
        )$old$;
  v_new CONSTANT TEXT := $new$        AND (NOT EXISTS (
          SELECT 1 FROM public.lab_arena_ledger AS prior_cost
          WHERE prior_cost.run_id=host_fault.run_id
        ) OR public.lab_arena__execute_host_interruption_v1(host_fault))$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),
    pg_catalog.jsonb_build_array(owner.rolname,p.proacl::TEXT,
      p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM
    pg_catalog.jsonb_build_array('lab_arena_owner',
      '{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[])
    OR NOT EXISTS (SELECT 1 FROM pg_catalog.pg_proc helper
      WHERE helper.oid=pg_catalog.to_regprocedure(
        'public.lab_arena__execute_host_interruption_v1(public.lab_arena_runs)')
        AND pg_catalog.pg_get_userbyid(helper.proowner)='lab_arena_owner'
        AND helper.prosecdef IS TRUE
        AND helper.provolatile='s'
        AND helper.proconfig=ARRAY['search_path=pg_catalog, public']::TEXT[]
        AND helper.proacl='{lab_arena_owner=X/lab_arena_owner}'::ACLITEM[]) THEN
    RAISE EXCEPTION 'Arena paid host quarantine security shape differs';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN
    IF (pg_catalog.length(v_definition)-pg_catalog.length(
        pg_catalog.replace(v_definition,v_new,'')))<>pg_catalog.length(v_new)
      OR pg_catalog.strpos(v_definition,v_old)>0 THEN
      RAISE EXCEPTION 'Arena paid host quarantine replay differs';
    END IF;
    RETURN;
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old)
    OR pg_catalog.strpos(v_definition,'lab_arena_round_host_fault_quarantine_v1')=0 THEN
    RAISE EXCEPTION 'Arena paid host quarantine preimage differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(
      pg_catalog.pg_get_functiondef(v_signature::REGPROCEDURE),'sha256'),'hex')
      <>v_postimage THEN
    RAISE EXCEPTION 'Arena paid host quarantine readback differs';
  END IF;
END;
$paid_host_quarantine$;

COMMIT;

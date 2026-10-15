-- Acknowledge authenticated re-observations only for the same immutable proof.
-- Preserve exact-request conflicts and the first canonical document/time.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;
DO $chain_outcome_retry488$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_record_chain_outcome_v1(text,integer,bigint,text,text,text,jsonb)';
  v_preimage CONSTANT TEXT := '4b59587948bfe87087c8cc5de1f8dd68b1ef53932b849387a4082b8e70611c65';
  v_postimage CONSTANT TEXT := '107dfd90dd15606d3ca169414c8c0376b734d6a813e1e57552b7c7e60a7d65ba';
  v_definition TEXT; v_identity JSONB;
  v_old CONSTANT TEXT := $old488$  ) ON CONFLICT (network, netuid, epoch, validator_hotkey, request_id) DO NOTHING;
  SELECT * INTO existing FROM public.lab_arena_chain_outcomes
    WHERE network = p_network AND netuid = p_netuid AND epoch = p_epoch
      AND validator_hotkey = p_validator_hotkey AND request_id = p_request_id;
  IF existing.outcome_doc IS DISTINCT FROM p_outcome_doc THEN$old488$;
  v_new CONSTANT TEXT := $new488$  ) ON CONFLICT DO NOTHING;
  -- Either unique key may win a racing insert. Read its actual immutable row;
  -- no UPDATE or replacement document is needed for a semantic retry.
  BEGIN
    SELECT * INTO STRICT existing FROM public.lab_arena_chain_outcomes
      WHERE network = p_network AND netuid = p_netuid AND epoch = p_epoch
        AND validator_hotkey = p_validator_hotkey
        AND (request_id = p_request_id OR extrinsic_hash = p_extrinsic_hash);
  EXCEPTION WHEN NO_DATA_FOUND OR TOO_MANY_ROWS THEN
    RAISE EXCEPTION 'lab_arena_chain_outcome_conflict' USING ERRCODE = '23505';
  END;
  -- The service validates the incoming contract/hash, validator signature and
  -- accepted state before this RPC. A new observation of the same extrinsic
  -- may change only its timestamp, content-addressed request and signature.
  -- An existing request ID retains its original exact-document contract.
  IF existing.outcome_doc IS DISTINCT FROM p_outcome_doc AND
    (existing.request_id = p_request_id OR
      (existing.outcome_doc - ARRAY['observed_at','request_id','signature'])
        IS DISTINCT FROM
      (p_outcome_doc - ARRAY['observed_at','request_id','signature'])) THEN$new488$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(p.oid),pg_catalog.jsonb_build_array(
    owner.rolname,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
  INTO v_definition,v_identity FROM pg_catalog.pg_proc p
  JOIN pg_catalog.pg_roles owner ON owner.oid=p.proowner
  WHERE p.oid=pg_catalog.to_regprocedure(v_signature);
  IF v_definition IS NULL OR v_identity IS DISTINCT FROM pg_catalog.jsonb_build_array(
    'lab_arena_owner','{lab_arena_owner=X/lab_arena_owner,lab_arena_service=X/lab_arena_owner}',
    TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'chain outcome retry security shape differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')=v_postimage THEN RETURN; END IF;
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_preimage
    OR (pg_catalog.length(v_definition)-pg_catalog.length(
      pg_catalog.replace(v_definition,v_old,'')))<>pg_catalog.length(v_old) THEN
    RAISE EXCEPTION 'chain outcome retry preimage differs' USING ERRCODE='55000';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,v_old,v_new);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'chain outcome retry postimage differs' USING ERRCODE='55000';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    v_signature::REGPROCEDURE),'sha256'),'hex')<>v_postimage THEN
    RAISE EXCEPTION 'chain outcome retry readback differs' USING ERRCODE='55000';
  END IF;
END;
$chain_outcome_retry488$;
REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
NOTIFY pgrst,'reload schema';
COMMIT;

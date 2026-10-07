-- Preserve budget refusal semantics; attribute a prior credential failure only
-- when the broker retained its existing structured account-failure evidence.
-- No ledger history, costs, limits, frozen inputs, scores or rewards change.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='30s';
DO $bound_account_refusal_421$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)';
  v_definition TEXT;
  v_identity JSONB;
  v_hash TEXT;
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
      TRUE,'v',ARRAY['search_path=pg_catalog, public']::TEXT[]) THEN
    RAISE EXCEPTION 'Arena account-refusal security shape differs';
  END IF;
  v_hash:=pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash='f41d91bec7f3182accf9aad86541d8abe6380ed6ff000bac1aeca83c78dc222d' THEN RETURN; END IF;
  IF v_hash<>'46fac530ef785c670bb400adaf89f8371ebb858be7043694754e0ec18458bf54' THEN
    RAISE EXCEPTION 'Arena account-refusal preimage differs';
  END IF;
  v_definition:=pg_catalog.replace(v_definition,$before$          AND head.entry_doc #>> '{call,provider_status}'
            IN ('401', '402', '403')$before$,$after$          AND head.entry_doc #>> '{call,provider_status}'
            IN ('401', '402', '403')
          -- Native HTTP status alone can describe Deepline's managed provider.
          -- Only the broker's existing account proof attributes it to a miner.
          AND head.entry_doc #>> '{call,account_failure_evidence,error_class}'
            = 'account_credential_failure'
          AND head.entry_doc #> '{call,account_failure_evidence,provider_status}'
            = head.entry_doc #> '{call,provider_status}'$after$);
  IF pg_catalog.encode(extensions.digest(v_definition,'sha256'),'hex')<>'f41d91bec7f3182accf9aad86541d8abe6380ed6ff000bac1aeca83c78dc222d' THEN
    RAISE EXCEPTION 'Arena account-refusal postimage differs';
  END IF;
  EXECUTE v_definition;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
    v_signature::REGPROCEDURE),'sha256'),'hex')<>'f41d91bec7f3182accf9aad86541d8abe6380ed6ff000bac1aeca83c78dc222d' THEN
    RAISE EXCEPTION 'Arena account-refusal readback differs';
  END IF;
END;
$bound_account_refusal_421$;
COMMIT;

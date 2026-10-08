-- Scale newly signed cumulative slot allocations within the existing miner pot.
-- Only the optional signed policy field is admitted here. Slot selection,
-- published scores, reward activation, accepted states and old payments stay intact.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $slot_pool_policy$
DECLARE
  v_oid OID := 'public.lab_arena_reward_slot_snapshot(text,jsonb)'::REGPROCEDURE;
  v_definition TEXT := pg_catalog.pg_get_functiondef(v_oid);
  v_hash TEXT;
  v_owner OID;
  v_acl ACLITEM[];
  v_schema_acl ACLITEM[];
BEGIN
  SELECT proowner, proacl INTO v_owner, v_acl FROM pg_catalog.pg_proc WHERE oid=v_oid;
  SELECT nspacl INTO v_schema_acl FROM pg_catalog.pg_namespace WHERE nspname='public';
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '6ed3bb56996027966c22e301476cbf1d10d73e5be8bfa12cecec11c64fb38858' THEN
    RETURN;
  END IF;
  IF v_hash <> '5f5da36d491ea7f61b41246f73b7208e6ecf54b1052f84ae2536f8d96955309a' THEN
    RAISE EXCEPTION '429 reward slot definition differs' USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $old_shape$  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy)) <> 2$old_shape$,
    $new_shape$  IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy - 'pool_percent')) <> 2$new_shape$);
  v_definition := pg_catalog.replace(v_definition,
    $old_tiers$  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN$old_tiers$,
    $new_tiers$  IF p_slot_policy ? 'pool_percent' AND (
       pg_catalog.jsonb_typeof(p_slot_policy -> 'pool_percent') IS DISTINCT FROM 'number'
       OR COALESCE(p_slot_policy ->> 'pool_percent', '') !~ '^(0|[1-9][0-9]?)$|^100$'
     ) THEN
    RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
  END IF;
  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN$new_tiers$);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex') <>
       '6ed3bb56996027966c22e301476cbf1d10d73e5be8bfa12cecec11c64fb38858' THEN
    RAISE EXCEPTION '429 reward slot policy seam differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
  IF (SELECT proowner FROM pg_catalog.pg_proc WHERE oid=v_oid) IS DISTINCT FROM v_owner
     OR (SELECT proacl FROM pg_catalog.pg_proc WHERE oid=v_oid) IS DISTINCT FROM v_acl
     OR (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public') IS DISTINCT FROM v_schema_acl
  THEN
    RAISE EXCEPTION '429 reward slot permissions changed' USING ERRCODE = '55000';
  END IF;
END;
$slot_pool_policy$;

NOTIFY pgrst, 'reload schema';
COMMIT;

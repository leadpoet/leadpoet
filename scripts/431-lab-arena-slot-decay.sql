-- Freeze slot achievement epochs for optional signed weekly decay.
-- Legacy snapshots, slot selection, promotion and accepted reward history stay intact.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $slot_decay_policy$
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
  IF v_hash = '78bf61278a796e0c7b06c025148bee5e0ad9bc8d249b27b2528ae337684bba33' THEN
    RETURN;
  END IF;
  IF v_hash <> '6ed3bb56996027966c22e301476cbf1d10d73e5be8bfa12cecec11c64fb38858' THEN
    RAISE EXCEPTION '431 reward slot definition differs' USING ERRCODE = '55000';
  END IF;
  v_definition := pg_catalog.replace(v_definition,
    $old_shape$p_slot_policy - 'pool_percent'$old_shape$,
    $new_shape$p_slot_policy - 'pool_percent' - 'decay'$new_shape$);
  v_definition := pg_catalog.replace(v_definition,
    $old_tiers$  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN$old_tiers$,
    $new_tiers$  IF p_slot_policy ? 'decay' THEN
    IF pg_catalog.jsonb_typeof(p_slot_policy -> 'decay') IS DISTINCT FROM 'object' THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
    IF (SELECT pg_catalog.count(*) FROM pg_catalog.jsonb_object_keys(p_slot_policy -> 'decay')) <> 2
       OR pg_catalog.jsonb_typeof(p_slot_policy #> '{decay,epochs_per_halving}') IS DISTINCT FROM 'number'
       OR COALESCE(p_slot_policy #>> '{decay,epochs_per_halving}', '') !~ '^[1-9][0-9]{0,6}$'
       OR pg_catalog.jsonb_typeof(p_slot_policy #> '{decay,max_halvings}') IS DISTINCT FROM 'number'
       OR COALESCE(p_slot_policy #>> '{decay,max_halvings}', '') !~ '^(0|[1-9][0-9]?)$' THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
    IF (p_slot_policy #>> '{decay,epochs_per_halving}')::NUMERIC > 1000000
       OR (p_slot_policy #>> '{decay,max_halvings}')::NUMERIC > 12 THEN
      RAISE EXCEPTION 'lab_arena_slot_policy_invalid' USING ERRCODE = '22023';
    END IF;
  END IF;
  IF pg_catalog.jsonb_array_length(p_slot_policy -> 'tiers') <> 3 THEN$new_tiers$);
  v_definition := pg_catalog.replace(v_definition,
    $old_candidate$    SELECT r.round_id, r.evaluation_date, r.baseline_promoted_at,$old_candidate$,
    $new_candidate$    SELECT r.round_id, r.evaluation_date, r.baseline_promoted_at, r.effective_reward_epoch,$new_candidate$);
  v_definition := pg_catalog.replace(v_definition,
    $old_entry$    ) AS entry
    FROM verified AS v$old_entry$,
    $new_entry$    ) || CASE WHEN p_slot_policy ? 'decay' THEN pg_catalog.jsonb_build_object(
      'start_epoch', CASE WHEN v.round_id = p_round_id THEN NULL ELSE v.effective_reward_epoch END
    ) ELSE '{}'::JSONB END AS entry
    FROM verified AS v$new_entry$);
  v_definition := pg_catalog.replace(v_definition,
    $old_return$  RETURN pg_catalog.jsonb_build_object('reward_slots', v_slots);$old_return$,
    $new_return$  IF p_slot_policy ? 'decay' AND EXISTS (
    SELECT 1 FROM pg_catalog.jsonb_array_elements(v_slots) AS slot(entry)
    WHERE entry ->> 'round_id' <> p_round_id
      AND (entry ->> 'start_epoch' IS NULL OR (entry ->> 'start_epoch')::BIGINT < 0)
  ) THEN
    RAISE EXCEPTION 'lab_arena_reward_slot_start_epoch_invalid' USING ERRCODE = '22023';
  END IF;
  RETURN pg_catalog.jsonb_build_object('reward_slots', v_slots);$new_return$);
  IF pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex') <>
       '78bf61278a796e0c7b06c025148bee5e0ad9bc8d249b27b2528ae337684bba33' THEN
    RAISE EXCEPTION '431 reward slot decay seam differs' USING ERRCODE = '55000';
  END IF;
  EXECUTE v_definition;
  IF (SELECT proowner FROM pg_catalog.pg_proc WHERE oid=v_oid) IS DISTINCT FROM v_owner
     OR (SELECT proacl FROM pg_catalog.pg_proc WHERE oid=v_oid) IS DISTINCT FROM v_acl
     OR (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname='public') IS DISTINCT FROM v_schema_acl
  THEN
    RAISE EXCEPTION '431 reward slot permissions changed' USING ERRCODE = '55000';
  END IF;
END;
$slot_decay_policy$;

NOTIFY pgrst, 'reload schema';
COMMIT;

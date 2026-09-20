-- Permit the explicit 90-minute recovery lease through the four existing
-- lease RPCs. Historical 60..3600-second leases remain unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lease_6300_claim_assignment$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_old TEXT := $old_claim_assignment$COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600$old_claim_assignment$;
  v_new TEXT := $new_claim_assignment$(COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600
          AND COALESCE(p_lease_ttl_seconds, 0) <> 6300)$new_claim_assignment$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure)
  INTO v_definition;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '4c5eb83c3daddfee5bfa2c33eaf33002be07bcd2f7a6fc23b5ff700d5980100a' THEN
    RETURN;
  ELSIF v_hash <> 'f0c877d0083a1b1bb7798b90cda7bf40dc6fd64d575d87f4213fe7bca84f610b' THEN
    RAISE EXCEPTION 'lease 6300 claim_assignment preimage differs';
  END IF;
  IF (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lease 6300 claim_assignment guard differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure), 'sha256'), 'hex')
  INTO v_hash;
  IF v_hash <> '4c5eb83c3daddfee5bfa2c33eaf33002be07bcd2f7a6fc23b5ff700d5980100a' THEN
    RAISE EXCEPTION 'lease 6300 claim_assignment postimage differs';
  END IF;
END $lease_6300_claim_assignment$;

DO $lease_6300_mark_uncertain$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_old TEXT := $old_mark_uncertain$COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600$old_mark_uncertain$;
  v_new TEXT := $new_mark_uncertain$(COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600
          AND COALESCE(p_lease_ttl_seconds, 0) <> 6300)$new_mark_uncertain$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)'::pg_catalog.regprocedure)
  INTO v_definition;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = 'da6b73e012983bb59ed83be2976e435af8ea5c0c9fea1b6ceca177cd892b8d99' THEN
    RETURN;
  ELSIF v_hash <> 'abb32a9b90e59f79192e34b799bb4bd331d2459f49766689232016ed72f59ce1' THEN
    RAISE EXCEPTION 'lease 6300 mark_uncertain preimage differs';
  END IF;
  IF (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lease 6300 mark_uncertain guard differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)'::pg_catalog.regprocedure), 'sha256'), 'hex')
  INTO v_hash;
  IF v_hash <> 'da6b73e012983bb59ed83be2976e435af8ea5c0c9fea1b6ceca177cd892b8d99' THEN
    RAISE EXCEPTION 'lease 6300 mark_uncertain postimage differs';
  END IF;
END $lease_6300_mark_uncertain$;

DO $lease_6300_reserve_call$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_old TEXT := $old_reserve_call$COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600$old_reserve_call$;
  v_new TEXT := $new_reserve_call$(COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600
          AND COALESCE(p_lease_ttl_seconds, 0) <> 6300)$new_reserve_call$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure)
  INTO v_definition;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = 'eb08656f73d4421be399a7bb24599a2b3d63912511372d6738e7d989dd134263' THEN
    RETURN;
  ELSIF v_hash <> 'fe57bec18824648750d5e8988a875be043a739b2867dd243f2109ed6347a4429' THEN
    RAISE EXCEPTION 'lease 6300 reserve_call preimage differs';
  END IF;
  IF (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lease 6300 reserve_call guard differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure), 'sha256'), 'hex')
  INTO v_hash;
  IF v_hash <> 'eb08656f73d4421be399a7bb24599a2b3d63912511372d6738e7d989dd134263' THEN
    RAISE EXCEPTION 'lease 6300 reserve_call postimage differs';
  END IF;
END $lease_6300_reserve_call$;

DO $lease_6300_settle_call$
DECLARE
  v_definition TEXT;
  v_hash TEXT;
  v_old TEXT := $old_settle_call$COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600$old_settle_call$;
  v_new TEXT := $new_settle_call$(COALESCE(p_lease_ttl_seconds, 0) NOT BETWEEN 60 AND 3600
          AND COALESCE(p_lease_ttl_seconds, 0) <> 6300)$new_settle_call$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef('public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure)
  INTO v_definition;
  v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
  IF v_hash = '7f8537fa88aadd0cfe796c1ce326a6fc0b16a30f76d1f51fc800bd5e18052356' THEN
    RETURN;
  ELSIF v_hash <> 'b479098c6a5b3d4bfd832e1960f91c6d46a6d13bb9358ee0568579ce2d4c8cc5' THEN
    RAISE EXCEPTION 'lease 6300 settle_call preimage differs';
  END IF;
  IF (pg_catalog.length(v_definition) - pg_catalog.length(pg_catalog.replace(v_definition, v_old, ''))) / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lease 6300 settle_call guard differs';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
  SELECT pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef('public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)'::pg_catalog.regprocedure), 'sha256'), 'hex')
  INTO v_hash;
  IF v_hash <> '7f8537fa88aadd0cfe796c1ce326a6fc0b16a30f76d1f51fc800bd5e18052356' THEN
    RAISE EXCEPTION 'lease 6300 settle_call postimage differs';
  END IF;
END $lease_6300_settle_call$;

COMMIT;

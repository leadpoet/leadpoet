-- Permit the exact 4500-second lease for new 60-minute execution rounds.
-- The existing 45- and 90-minute lease profiles keep their installed guards.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lease_4500$
DECLARE
  v_rpc RECORD;
  v_definition TEXT;
  v_hash TEXT;
  v_old CONSTANT TEXT := $old$AND COALESCE(p_lease_ttl_seconds, 0) <> 6300)$old$;
  v_new CONSTANT TEXT := $new$AND COALESCE(p_lease_ttl_seconds, 0) <> 6300
          AND COALESCE(p_lease_ttl_seconds, 0) <> 4500)$new$;
BEGIN
  FOR v_rpc IN
    SELECT * FROM (VALUES
      ('public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::REGPROCEDURE,
       '4c5eb83c3daddfee5bfa2c33eaf33002be07bcd2f7a6fc23b5ff700d5980100a',
       '09f708260fdaeca445e3b7b98fcd8d367146de8833e7cb718802b7b64c8d42ae'),
      ('public.lab_arena_mark_uncertain(text,text,text,jsonb,integer)'::REGPROCEDURE,
       'da6b73e012983bb59ed83be2976e435af8ea5c0c9fea1b6ceca177cd892b8d99',
       'ed7046ece252cd97252b2828c1087318d9da7b995d41e66b635190130bdb5b4e'),
      ('public.lab_arena_reserve_call(text,text,text,text,text,text,bigint,jsonb,integer)'::REGPROCEDURE,
       'eb08656f73d4421be399a7bb24599a2b3d63912511372d6738e7d989dd134263',
       '4b96b984aa6c471588e84f08ab3a832300de4cb9e8cda386cd46a884fb431422'),
      ('public.lab_arena_settle_call(text,text,text,bigint,jsonb,integer)'::REGPROCEDURE,
       '7f8537fa88aadd0cfe796c1ce326a6fc0b16a30f76d1f51fc800bd5e18052356',
       'b8cb76f7838ae6f5daffaa801127808879f0d18281aec903f1a906137a6b3303')
    ) AS rpc(signature, preimage_sha256, postimage_sha256)
  LOOP
    SELECT pg_catalog.pg_get_functiondef(v_rpc.signature)
    INTO v_definition;
    v_hash := pg_catalog.encode(extensions.digest(v_definition, 'sha256'), 'hex');
    IF v_hash = v_rpc.postimage_sha256 THEN
      CONTINUE;
    ELSIF v_hash IS DISTINCT FROM v_rpc.preimage_sha256 THEN
      RAISE EXCEPTION 'lease 4500 RPC preimage differs: %', v_rpc.signature;
    END IF;
    IF (pg_catalog.length(v_definition)
        - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
        / pg_catalog.length(v_old) <> 1 THEN
      RAISE EXCEPTION 'lease 4500 RPC guard differs: %', v_rpc.signature;
    END IF;
    EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
    SELECT pg_catalog.encode(
      extensions.digest(pg_catalog.pg_get_functiondef(v_rpc.signature), 'sha256'),
      'hex'
    ) INTO v_hash;
    IF v_hash IS DISTINCT FROM v_rpc.postimage_sha256 THEN
      RAISE EXCEPTION 'lease 4500 RPC postimage differs: %', v_rpc.signature;
    END IF;
  END LOOP;
END;
$lease_4500$;

COMMIT;

-- Admit only the broker's failed compatibility-adaptation witness to exact
-- Deepline billing recovery. No ledger, lease, score, response or price changes.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $adaptation_cost_witness453$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)';
  v_definition TEXT;
  v_before_security JSONB;
  v_after_security JSONB;
  v_hash TEXT;
  v_old CONSTANT TEXT := $old$            AND p_uncertainty_doc #>> '{call,failure_stage}' = 'settlement'
            AND p_uncertainty_doc #>> '{call,error_class}' = 'ArenaStoreError'$old$;
  v_new CONSTANT TEXT := $new$            AND (
              (p_uncertainty_doc #>> '{call,failure_stage}' = 'settlement'
               AND p_uncertainty_doc #>> '{call,error_class}' = 'ArenaStoreError')
              OR
              -- lab_arena_compatibility_adaptation_cost_reconciliation_v1:
              -- A rejected reply is not a successful execution. The immutable
              -- deterministic reservation still permits exact GET-only billing.
              (p_uncertainty_doc #>> '{call,failure_stage}' = 'response_adaptation'
               AND p_uncertainty_doc #>> '{call,error_class}' = 'CompatibilityResponseError'
               AND p_uncertainty_doc #> '{call,call_succeeded}' = 'false'::JSONB
               AND p_reservation_doc ->> 'deepline_execution_key' =
                   'arena:' || pg_catalog.substr(p_call_identity, 8))
            )$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(v_signature::regprocedure),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_before_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  v_hash := encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash='d8ec50ba997a2aab723feac8341864e2f6cdcedc34cd03ba74fb275a5cdfa550' THEN RETURN; END IF;
  IF v_hash IS DISTINCT FROM 'e90a382a37756fa64768c66b399124746e3bd621ac03f6e6e27956581bfab137'
     OR (length(v_definition)-length(replace(v_definition,v_old,'')))<>length(v_old) THEN
    RAISE EXCEPTION 'Deepline adaptation453 binding preimage differs';
  END IF;
  EXECUTE replace(v_definition,v_old,v_new);
  SELECT encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::regprocedure),'sha256'),'hex'),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_hash,v_after_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  IF v_hash IS DISTINCT FROM 'd8ec50ba997a2aab723feac8341864e2f6cdcedc34cd03ba74fb275a5cdfa550'
     OR v_before_security IS DISTINCT FROM v_after_security THEN
    RAISE EXCEPTION 'Deepline adaptation453 binding postimage differs';
  END IF;
END;
$adaptation_cost_witness453$;

-- V2 already validates this key against the immutable reservation before the
-- binding helper call. Carry that validated key into its synthetic document.
-- V1 remains unchanged and cannot authorize this new adaptation witness.
DO $adaptation_cost_binding_v2453$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena_reconcile_deepline_cost_v2(text,text,text,bigint,text,text,text,bigint,text,text,text)';
  v_definition TEXT;
  v_before_security JSONB;
  v_after_security JSONB;
  v_hash TEXT;
  v_old CONSTANT TEXT := $old$            'tool', p_operation,
            'credential_fingerprint', p_credential_fingerprint
          ),
          p_call_identity
        ) THEN$old$;
  v_new CONSTANT TEXT := $new$            'tool', p_operation,
            'credential_fingerprint', p_credential_fingerprint,
            'deepline_execution_key', p_execution_key
          ),
          p_call_identity
        ) THEN$new$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(v_signature::regprocedure),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_before_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  v_hash := encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash='f54d869add37b31940b83ea4d71f5fade4d9763e25f223b5ee29d89d19173d1a' THEN RETURN; END IF;
  IF v_hash IS DISTINCT FROM 'f3d2e7a1810d4a77cec1ae32e883ab76cd7fec506d2420b3039893eed535e115'
     OR (length(v_definition)-length(replace(v_definition,v_old,'')))<>length(v_old) THEN
    RAISE EXCEPTION 'Deepline adaptation453 V2 preimage differs';
  END IF;
  EXECUTE replace(v_definition,v_old,v_new);
  SELECT encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::regprocedure),'sha256'),'hex'),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_hash,v_after_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  IF v_hash IS DISTINCT FROM 'f54d869add37b31940b83ea4d71f5fade4d9763e25f223b5ee29d89d19173d1a'
     OR v_before_security IS DISTINCT FROM v_after_security THEN
    RAISE EXCEPTION 'Deepline adaptation453 V2 postimage differs';
  END IF;
END;
$adaptation_cost_binding_v2453$;
COMMIT;

-- Admit bound native public-execute receipts to existing GET-only accounting.
-- No ledger, lease, provider dispatch, score, result, price or grants are changed.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $native_settlement_binding462$
DECLARE
  v_signature CONSTANT TEXT := 'public.lab_arena__deepline_cost_binding_v1(jsonb,jsonb,text)';
  v_definition TEXT;
  v_before_security JSONB;
  v_after_security JSONB;
  v_hash TEXT;
  v_old_class CONSTANT TEXT := $old_class$              (p_uncertainty_doc #>> '{call,failure_stage}' = 'settlement'
               AND p_uncertainty_doc #>> '{call,error_class}' = 'ArenaStoreError')$old_class$;
  v_new_class CONSTANT TEXT := $new_class$              (p_uncertainty_doc #>> '{call,failure_stage}' = 'settlement'
               AND (
                 p_uncertainty_doc #>> '{call,error_class}' = 'ArenaStoreError'
                 OR
                 -- Old workers sanitized ArenaStoreUnavailable to Exception.
                 -- This receipt admits exact original-payer GET only; the
                 -- provider lookup still supplies the authoritative charge.
                 (p_uncertainty_doc #>> '{call,error_class}' = 'Exception'
                  AND p_uncertainty_doc #> '{call,call_succeeded}' = 'true'::JSONB
                  AND p_uncertainty_doc #> '{call,provider_status}' = '200'::JSONB
                  AND p_uncertainty_doc #>> '{call,top_level_job_status}' = 'completed'
                  AND p_uncertainty_doc #> '{call,billing_present}' = 'true'::JSONB
                  AND p_uncertainty_doc #>> '{call,deepline_request_id}' =
                      p_reservation_doc ->> 'deepline_request_id'
                  AND p_uncertainty_doc #>> '{call,deepline_operation}' =
                      p_reservation_doc ->> 'tool'
                  AND p_uncertainty_doc #>> '{call,credential_fingerprint}' =
                      p_reservation_doc ->> 'credential_fingerprint'
                  AND p_uncertainty_doc #>> '{call,deepline_execution_key}' =
                      'arena:' || pg_catalog.substr(p_call_identity, 8)
                  AND p_reservation_doc ->> 'deepline_execution_key' =
                      'arena:' || pg_catalog.substr(p_call_identity, 8)
                  AND p_uncertainty_doc #>> '{call,deepline_job_id}' ~
                      '^public-execute:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                  AND p_uncertainty_doc #>> '{call,provider_cost,request_id}' =
                      p_uncertainty_doc #>> '{call,deepline_job_id}'
                  AND p_uncertainty_doc #>> '{call,provider_cost,operation}' =
                      p_reservation_doc ->> 'tool'
                  AND p_uncertainty_doc #>> '{call,provider_cost,basis}' =
                      'deepline_exact_request_credits_x_0.10_usd'
                  AND p_uncertainty_doc #>> '{call,provider_cost,unit_name}' = 'credits'
                  AND pg_catalog.jsonb_typeof(p_uncertainty_doc #> '{call,known_actual_microusd}') = 'number'
                  AND pg_catalog.jsonb_typeof(p_uncertainty_doc #> '{call,provider_cost,units}') = 'string'
                  AND CASE WHEN
                      p_uncertainty_doc #>> '{call,known_actual_microusd}' ~ '^[0-9]{1,18}$'
                      AND p_uncertainty_doc #>> '{call,provider_cost,units}' ~ '^[0-9]{1,12}(\.[0-9]{1,12})?$'
                    THEN (p_uncertainty_doc #>> '{call,known_actual_microusd}')::NUMERIC =
                         pg_catalog.ceil((p_uncertainty_doc #>> '{call,provider_cost,units}')::NUMERIC * 100000)
                    ELSE FALSE END
                 )
               ))$new_class$;
  v_old_id CONSTANT TEXT := $old_id$                    OR p_uncertainty_doc #>> '{call,deepline_job_id}' ~
                      '^[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64}$'))$old_id$;
  v_new_id CONSTANT TEXT := $new_id$                    OR p_uncertainty_doc #>> '{call,deepline_job_id}' ~
                      '^[a-z0-9]{3,8}::[a-z0-9]{1,16}-[0-9]{13}-[a-f0-9]{12,64}$'
                    OR (p_uncertainty_doc #>> '{call,deepline_job_id}' ~
                          '^public-execute:[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
                        AND p_reservation_doc ->> 'deepline_execution_key' =
                            'arena:' || pg_catalog.substr(p_call_identity, 8)
                        AND p_uncertainty_doc #>> '{call,deepline_execution_key}' =
                            p_reservation_doc ->> 'deepline_execution_key')))$new_id$;
BEGIN
  SELECT pg_catalog.pg_get_functiondef(v_signature::regprocedure),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_definition,v_before_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  v_hash := encode(extensions.digest(v_definition,'sha256'),'hex');
  IF v_hash='8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49' THEN RETURN; END IF;
  IF v_hash IS DISTINCT FROM 'd8ec50ba997a2aab723feac8341864e2f6cdcedc34cd03ba74fb275a5cdfa550'
     OR (length(v_definition)-length(replace(v_definition,v_old_class,'')))<>length(v_old_class)
     OR (length(v_definition)-length(replace(v_definition,v_old_id,'')))<>length(v_old_id) THEN
    RAISE EXCEPTION 'Deepline native462 binding preimage differs';
  END IF;
  EXECUTE replace(replace(v_definition,v_old_class,v_new_class),v_old_id,v_new_id);
  SELECT encode(extensions.digest(pg_catalog.pg_get_functiondef(v_signature::regprocedure),'sha256'),'hex'),
    jsonb_build_array(p.proowner,p.proacl::TEXT,p.prosecdef,p.provolatile,p.proconfig)
    INTO v_hash,v_after_security
    FROM pg_catalog.pg_proc p WHERE p.oid=v_signature::regprocedure;
  IF v_hash IS DISTINCT FROM '8a973077fb121ef9f01866b301aba0ec5dd05da8143f87644ad8e28afa530f49'
     OR v_before_security IS DISTINCT FROM v_after_security THEN
    RAISE EXCEPTION 'Deepline native462 binding postimage differs: %', v_hash;
  END IF;
END;
$native_settlement_binding462$;
COMMIT;

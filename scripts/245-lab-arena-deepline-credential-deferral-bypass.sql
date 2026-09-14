-- Let scoring continue after a provider-authenticated Deepline credential
-- refusal. The uncertain cost remains reserved and must still be reconciled;
-- this changes only the submission-local claim deferral added by migration 243.

BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $lab_arena_245_deepline_credential_deferral_bypass$
DECLARE
  v_definition TEXT;
  v_old TEXT := $old$
          AND uncertainty.entry_doc #>> '{call,credential_fingerprint}' =
              reservation.entry_doc ->> 'credential_fingerprint'
          AND NOT EXISTS (
$old$;
  v_new TEXT := $new$
          AND uncertainty.entry_doc #>> '{call,credential_fingerprint}' =
              reservation.entry_doc ->> 'credential_fingerprint'
          -- lab_arena_deepline_credential_deferral_bypass: a strictly bound
          -- 401/402/403 permits the existing no-dispatch credential-refusal
          -- path to finish the submission. The charge stays uncertain.
          AND NOT COALESCE((
            reservation.funding_source = 'miner_key'
            AND uncertainty.funding_source = 'miner_key'
            AND pg_catalog.jsonb_typeof(
              uncertainty.entry_doc #>
                '{call,account_failure_evidence}'
            ) = 'object'
            AND uncertainty.entry_doc #>>
                  '{call,account_failure_evidence,error_class}' =
                'account_credential_failure'
            AND pg_catalog.jsonb_typeof(
              uncertainty.entry_doc #>
                '{call,account_failure_evidence,provider_status}'
            ) = 'number'
            AND uncertainty.entry_doc #>>
                  '{call,account_failure_evidence,provider_status}'
                IN ('401', '402', '403')
            AND pg_catalog.jsonb_typeof(
              uncertainty.entry_doc #>
                '{call,account_failure_evidence,base_call_identity}'
            ) = 'string'
            AND uncertainty.entry_doc #>>
                  '{call,account_failure_evidence,base_call_identity}' ~
                '^sha256:[0-9a-f]{64}$'
            AND pg_catalog.jsonb_typeof(
              uncertainty.entry_doc #>
                '{call,account_failure_evidence,provider_attempt}'
            ) = 'number'
            AND uncertainty.entry_doc #>>
                  '{call,account_failure_evidence,provider_attempt}'
                IN ('1', '2', '3', '4')
            AND pg_catalog.jsonb_typeof(
              uncertainty.entry_doc #>
                '{call,account_failure_evidence,action_sequence}'
            ) = 'number'
            AND uncertainty.entry_doc #>>
                  '{call,account_failure_evidence,action_sequence}' ~
                '^[0-9]+$'
            AND uncertainty.entry_doc #>
                  '{call,account_failure_evidence,base_call_identity}' =
                reservation.entry_doc -> 'base_call_identity'
            AND uncertainty.entry_doc #>
                  '{call,account_failure_evidence,provider_attempt}' =
                reservation.entry_doc -> 'provider_attempt'
            AND uncertainty.entry_doc #>
                  '{call,account_failure_evidence,action_sequence}' =
                reservation.entry_doc -> 'action_sequence'
          ), FALSE)
          AND NOT EXISTS (
$new$;
BEGIN
  IF pg_catalog.to_regprocedure(
       'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'
     ) IS NULL THEN
    RAISE EXCEPTION
      'apply 236-lab-arena-score-submission-serialization.sql first';
  END IF;
  SELECT pg_catalog.pg_get_functiondef(
    'public.lab_arena_claim_assignment(text,text,integer,integer,text[],text,text,text,integer)'::pg_catalog.regprocedure
  ) INTO v_definition;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_credential_deferral_bypass'
     ) > 0 THEN
    RETURN;
  END IF;
  IF pg_catalog.strpos(
       v_definition, 'lab_arena_deepline_reconciliation_retry_deferral'
     ) = 0
     OR (pg_catalog.length(v_definition)
       - pg_catalog.length(pg_catalog.replace(v_definition, v_old, '')))
       / pg_catalog.length(v_old) <> 1 THEN
    RAISE EXCEPTION 'lab_arena_claim_assignment credential bypass shape unexpected';
  END IF;
  EXECUTE pg_catalog.replace(v_definition, v_old, v_new);
END;
$lab_arena_245_deepline_credential_deferral_bypass$;

NOTIFY pgrst, 'reload schema';
COMMIT;

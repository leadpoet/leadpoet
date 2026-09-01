-- Scope new official-baseline provider request custody to one protected
-- attempt while permitting exact, same-run recovery of legacy dispatch-only
-- request references already durably recorded by the gateway.
BEGIN;

CREATE OR REPLACE FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v2()
RETURNS TRIGGER
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $validate_provider_request_replay$
DECLARE
    prior_terminal
        public.research_lab_official_baseline_action_terminals_v1%ROWTYPE;
    prior_attempt
        public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
    current_attempt
        public.research_lab_official_baseline_action_attempts_v1%ROWTYPE;
BEGIN
    IF NEW.provider_request_ref IS NULL THEN
        RETURN NEW;
    END IF;

    -- Serialize the transition away from dispatch-only request references.
    -- PostgreSQL has no conditional unique index whose predicate can compare
    -- another row's immutable custody, so the trigger owns that exact check.
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended(NEW.provider_request_ref, 0)
    );

    SELECT * INTO prior_terminal
      FROM public.research_lab_official_baseline_action_terminals_v1 terminal
     WHERE terminal.provider_request_ref = NEW.provider_request_ref
     ORDER BY terminal.attempt_key
     LIMIT 1;
    IF NOT FOUND THEN
        RETURN NEW;
    END IF;

    SELECT * INTO prior_attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 attempt
     WHERE attempt.attempt_key = prior_terminal.attempt_key;
    SELECT * INTO current_attempt
      FROM public.research_lab_official_baseline_action_attempts_v1 attempt
     WHERE attempt.attempt_key = NEW.attempt_key;
    IF prior_attempt.attempt_key IS NULL OR current_attempt.attempt_key IS NULL THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_attempt_missing'
            USING ERRCODE = '23503';
    END IF;

    -- A duplicate is recoverable only for the old dispatch-hash identity and
    -- only when it is an exact provider-cache replay in another protected unit
    -- of the same official run. New v2 identities remain globally unique.
    IF NEW.provider_request_ref !~ '^provider_request:[0-9a-f]{64}$'
       OR NEW.terminal_state IS DISTINCT FROM 'terminal_known'
       OR prior_terminal.terminal_state IS DISTINCT FROM 'terminal_known'
       OR prior_attempt.run_sha256 IS DISTINCT FROM current_attempt.run_sha256
       OR prior_attempt.unit_ref IS NOT DISTINCT FROM current_attempt.unit_ref
       OR prior_attempt.action_idempotency_sha256 IS DISTINCT FROM
            current_attempt.action_idempotency_sha256
       OR prior_attempt.action_sha256 IS DISTINCT FROM
            current_attempt.action_sha256
       OR prior_attempt.action_sequence IS DISTINCT FROM
            current_attempt.action_sequence
       OR prior_attempt.action_type IS DISTINCT FROM current_attempt.action_type
       OR prior_attempt.tool_id IS DISTINCT FROM current_attempt.tool_id
       OR prior_attempt.binding_contract_sha256 IS DISTINCT FROM
            current_attempt.binding_contract_sha256
       OR prior_attempt.request_fingerprint_sha256 IS DISTINCT FROM
            current_attempt.request_fingerprint_sha256
       OR prior_attempt.request_body_sha256 IS DISTINCT FROM
            current_attempt.request_body_sha256
       OR prior_attempt.call_cap IS DISTINCT FROM current_attempt.call_cap
       OR prior_attempt.credit_cap_microunits IS DISTINCT FROM
            current_attempt.credit_cap_microunits
       OR prior_attempt.timeout_ms IS DISTINCT FROM current_attempt.timeout_ms
       OR prior_terminal.provider_identity_sha256 IS DISTINCT FROM
            NEW.provider_identity_sha256
       OR prior_terminal.model_provider_response_sha256 IS DISTINCT FROM
            NEW.model_provider_response_sha256
       OR prior_terminal.outcome IS DISTINCT FROM NEW.outcome
       OR prior_terminal.call_count IS DISTINCT FROM NEW.call_count
       OR prior_terminal.cost_microunits IS DISTINCT FROM NEW.cost_microunits
       OR prior_terminal.provider_receipt_ref IS NOT DISTINCT FROM
            NEW.provider_receipt_ref
    THEN
        RAISE EXCEPTION
            'research_lab_official_baseline_provider_request_replay_conflict'
            USING ERRCODE = '23505';
    END IF;

    RETURN NEW;
END;
$validate_provider_request_replay$;

DROP TRIGGER IF EXISTS
    trg_research_lab_official_baseline_provider_request_replay_v2
    ON public.research_lab_official_baseline_action_terminals_v1;
CREATE TRIGGER trg_research_lab_official_baseline_provider_request_replay_v2
BEFORE INSERT ON public.research_lab_official_baseline_action_terminals_v1
FOR EACH ROW
EXECUTE FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v2();

DROP INDEX IF EXISTS public.idx_rl_official_baseline_provider_request_v1;
CREATE INDEX IF NOT EXISTS idx_rl_official_baseline_provider_request_lookup_v2
    ON public.research_lab_official_baseline_action_terminals_v1(
        provider_request_ref
    ) WHERE provider_request_ref IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_rl_official_baseline_provider_request_v2
    ON public.research_lab_official_baseline_action_terminals_v1(
        provider_request_ref
    ) WHERE provider_request_ref ~ '^provider_request_v2:[0-9a-f]{64}$';

CREATE OR REPLACE FUNCTION
    public.research_lab_official_baseline_request_scope_v2()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $official_baseline_request_scope$
    SELECT pg_catalog.jsonb_build_object(
        'schema_version',
            'leadpoet.research_lab.official_baseline_provider_request.v2',
        'identity_scope', 'protected_preparation_and_dispatch',
        'legacy_replay', 'same_run_exact_provider_response_only',
        'new_identity_unique', TRUE
    );
$official_baseline_request_scope$;

REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_request_replay_guard_v2()
FROM PUBLIC, anon, authenticated, service_role;
REVOKE ALL ON FUNCTION
    public.research_lab_official_baseline_request_scope_v2()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION
    public.research_lab_official_baseline_request_scope_v2()
TO service_role;

NOTIFY pgrst, 'reload schema';

COMMIT;

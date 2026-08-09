-- Linearize SOURCE_ADD miner admission with the durable pause control.
--
-- Migration 96 made queue claiming honor this control. This trigger extends
-- the same authority to the initial miner work item so a concurrent pause and
-- admission have one deterministic order. Operator provenance rechecks remain
-- available while paused because they do not carry admission_kind.

BEGIN;

SET LOCAL lock_timeout = '5s';

CREATE OR REPLACE FUNCTION public.enforce_research_lab_source_add_admission_control()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF NEW.work_kind = 'provenance'
       AND NEW.job_doc->>'admission_kind' = 'miner_submission' THEN
        PERFORM pg_catalog.pg_advisory_xact_lock(
            pg_catalog.hashtextextended('source-add-control', 0)
        );
        IF COALESCE((
            SELECT control.paused
            FROM public.research_lab_source_add_control control
            WHERE control.singleton
        ), TRUE) THEN
            RAISE EXCEPTION 'SOURCE_ADD admission is paused'
                USING ERRCODE = '55000';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_source_add_work_admission_control
    ON public.research_lab_source_add_work_items;
CREATE TRIGGER trg_source_add_work_admission_control
    BEFORE INSERT ON public.research_lab_source_add_work_items
    FOR EACH ROW EXECUTE FUNCTION public.enforce_research_lab_source_add_admission_control();

CREATE OR REPLACE FUNCTION public.research_lab_source_add_set_paused(
    p_paused BOOLEAN,
    p_reason TEXT,
    p_actor_ref TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
BEGIN
    IF btrim(p_reason) = '' OR btrim(p_actor_ref) = '' THEN
        RAISE EXCEPTION 'SOURCE_ADD pause reason and actor are required';
    END IF;
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('source-add-control', 0)
    );
    UPDATE public.research_lab_source_add_control
    SET paused = p_paused, reason = left(p_reason, 500),
        actor_ref = left(p_actor_ref, 200), updated_at = NOW()
    WHERE singleton;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'SOURCE_ADD control row is unavailable';
    END IF;
    RETURN (
        SELECT to_jsonb(control)
        FROM public.research_lab_source_add_control control
        WHERE control.singleton
    );
END;
$$;

CREATE OR REPLACE FUNCTION public.research_lab_source_add_admission_control_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
    SELECT jsonb_build_object(
        'schema_version', 'leadpoet.source_add_admission_control_contract.v1',
        'control_row_present', EXISTS (
            SELECT 1
            FROM public.research_lab_source_add_control control
            WHERE control.singleton
        ),
        'trigger_enabled', COALESCE((
            SELECT trigger.tgenabled IN ('O', 'A')
            FROM pg_catalog.pg_trigger trigger
            JOIN pg_catalog.pg_class relation
              ON relation.oid = trigger.tgrelid
            JOIN pg_catalog.pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            WHERE namespace.nspname = 'public'
              AND relation.relname = 'research_lab_source_add_work_items'
              AND trigger.tgname = 'trg_source_add_work_admission_control'
              AND NOT trigger.tgisinternal
        ), FALSE),
        'pause_rpc', 'research_lab_source_add_set_paused',
        'admission_trigger', 'trg_source_add_work_admission_control'
    );
$$;

REVOKE ALL ON FUNCTION public.enforce_research_lab_source_add_admission_control()
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_source_add_admission_control_contract_v1()
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_set_paused(BOOLEAN, TEXT, TEXT)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_source_add_admission_control_contract_v1()
    TO service_role;

COMMENT ON FUNCTION public.enforce_research_lab_source_add_admission_control() IS
    'Fails closed and serializes initial miner SOURCE_ADD work against pause/resume.';
COMMENT ON FUNCTION public.research_lab_source_add_admission_control_contract_v1() IS
    'Read-only release preflight contract for SOURCE_ADD admission pause authority.';

NOTIFY pgrst, 'reload schema';

COMMIT;

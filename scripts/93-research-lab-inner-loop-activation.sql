-- Append-only automatic activation evidence for Research Lab inner-loop ranking.
-- This migration does not change scoring, promotion, rewards, payments,
-- allocations, emissions, validator weights, or chain submission.
-- Apply after scripts/86-research-lab-attested-v2-authority.sql; safe to rerun.

BEGIN;

CREATE TABLE IF NOT EXISTS public.research_lab_inner_loop_activation_events (
    seq                 BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    schema_version      TEXT NOT NULL
                        CHECK (schema_version = 'research_lab.inner_loop_activation_event.v1'),
    event_type          TEXT NOT NULL CHECK (event_type IN ('phase_transition', 'run_observed')),
    phase               TEXT NOT NULL CHECK (phase IN ('off', 'observe', 'shadow', 'rank')),
    run_id              UUID NULL,
    evidence_doc        JSONB NOT NULL CHECK (jsonb_typeof(evidence_doc) = 'object'),
    event_hash          TEXT NOT NULL UNIQUE CHECK (event_hash ~ '^sha256:[0-9a-f]{64}$'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_research_lab_inner_loop_activation_created
    ON public.research_lab_inner_loop_activation_events(created_at DESC, seq DESC);
CREATE INDEX IF NOT EXISTS idx_research_lab_inner_loop_activation_run
    ON public.research_lab_inner_loop_activation_events(run_id, seq DESC)
    WHERE run_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_research_lab_inner_loop_run_observed
    ON public.research_lab_inner_loop_activation_events(run_id)
    WHERE event_type = 'run_observed' AND run_id IS NOT NULL;

CREATE OR REPLACE FUNCTION public.append_research_lab_inner_loop_activation_event(
    requested_event_type TEXT,
    requested_phase TEXT,
    requested_run_id UUID,
    requested_evidence_doc JSONB,
    requested_event_hash TEXT,
    expected_current_phase TEXT DEFAULT NULL
)
RETURNS public.research_lab_inner_loop_activation_events
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = ''
AS $$
DECLARE
    current_phase TEXT;
    inserted public.research_lab_inner_loop_activation_events;
BEGIN
    PERFORM pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtext('research_lab_inner_loop_activation')
    );

    SELECT e.phase
      INTO current_phase
      FROM public.research_lab_inner_loop_activation_events e
     WHERE e.event_type = 'phase_transition'
     ORDER BY e.seq DESC
     LIMIT 1;

    IF expected_current_phase IS NOT NULL
       AND COALESCE(current_phase, 'observe') IS DISTINCT FROM expected_current_phase
    THEN
        RAISE EXCEPTION
            'research_lab_inner_loop_phase_conflict: expected %, current %',
            expected_current_phase,
            COALESCE(current_phase, 'observe')
            USING ERRCODE = '40001';
    END IF;

    INSERT INTO public.research_lab_inner_loop_activation_events (
        schema_version,
        event_type,
        phase,
        run_id,
        evidence_doc,
        event_hash
    ) VALUES (
        'research_lab.inner_loop_activation_event.v1',
        requested_event_type,
        requested_phase,
        requested_run_id,
        requested_evidence_doc,
        requested_event_hash
    )
    ON CONFLICT (event_hash) DO NOTHING
    RETURNING * INTO inserted;

    IF inserted.seq IS NULL THEN
        SELECT e.*
          INTO inserted
          FROM public.research_lab_inner_loop_activation_events e
         WHERE e.event_hash = requested_event_hash;
    END IF;

    RETURN inserted;
END;
$$;

REVOKE ALL ON FUNCTION public.append_research_lab_inner_loop_activation_event(
    TEXT, TEXT, UUID, JSONB, TEXT, TEXT
) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.append_research_lab_inner_loop_activation_event(
    TEXT, TEXT, UUID, JSONB, TEXT, TEXT
) TO service_role;

DROP TRIGGER IF EXISTS prevent_research_lab_inner_loop_activation_mutation
    ON public.research_lab_inner_loop_activation_events;
CREATE TRIGGER prevent_research_lab_inner_loop_activation_mutation
    BEFORE UPDATE OR DELETE ON public.research_lab_inner_loop_activation_events
    FOR EACH ROW EXECUTE FUNCTION public.prevent_research_lab_attested_v2_mutation();

CREATE OR REPLACE VIEW public.research_lab_inner_loop_activation_current
WITH (security_invoker = true) AS
SELECT DISTINCT ON (event_type)
    seq,
    schema_version,
    event_type,
    phase,
    run_id,
    evidence_doc,
    event_hash,
    created_at
FROM public.research_lab_inner_loop_activation_events
ORDER BY event_type, seq DESC;

REVOKE ALL ON TABLE public.research_lab_inner_loop_activation_events
    FROM anon, authenticated;
REVOKE ALL ON TABLE public.research_lab_inner_loop_activation_current
    FROM anon, authenticated;
GRANT SELECT, INSERT ON TABLE public.research_lab_inner_loop_activation_events
    TO service_role;
GRANT SELECT ON TABLE public.research_lab_inner_loop_activation_current
    TO service_role;

ALTER TABLE public.research_lab_inner_loop_activation_events ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_read
    ON public.research_lab_inner_loop_activation_events;
CREATE POLICY service_role_read
    ON public.research_lab_inner_loop_activation_events
    FOR SELECT TO service_role USING (true);

DROP POLICY IF EXISTS service_role_insert
    ON public.research_lab_inner_loop_activation_events;
CREATE POLICY service_role_insert
    ON public.research_lab_inner_loop_activation_events
    FOR INSERT TO service_role WITH CHECK (true);

COMMENT ON TABLE public.research_lab_inner_loop_activation_events IS
    'Append-only evidence and automatic phase transitions for ranking-only Research Lab inner-loop evaluation.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Research Lab: allow Deepline provider labels in provider cost events.
--
-- Deployment policy:
--   * Apply after script 69.
--   * Safe to apply repeatedly.
--   * Metadata-only constraint update; does not change stored costs.

BEGIN;

ALTER TABLE public.research_lab_provider_cost_events
    DROP CONSTRAINT IF EXISTS research_lab_provider_cost_events_provider_check;

ALTER TABLE public.research_lab_provider_cost_events
    ADD CONSTRAINT research_lab_provider_cost_events_provider_check
    CHECK (provider IN ('exa', 'or', 'sd', 'deepline', 'unknown'));

COMMIT;


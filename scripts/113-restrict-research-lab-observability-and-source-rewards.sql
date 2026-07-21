-- Restrict internal Research Lab reward authority and observability relations.
-- Existing rows and reward behavior are unchanged.

BEGIN;

REVOKE ALL ON TABLE
    public.research_lab_source_add_reward_obligations,
    public.research_lab_source_add_reward_events
FROM PUBLIC, anon, authenticated;

GRANT SELECT, INSERT ON TABLE
    public.research_lab_source_add_reward_obligations,
    public.research_lab_source_add_reward_events
TO service_role;

ALTER TABLE public.research_lab_source_add_reward_obligations
    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.research_lab_source_add_reward_events
    ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_role_all
    ON public.research_lab_source_add_reward_obligations;
CREATE POLICY service_role_all
    ON public.research_lab_source_add_reward_obligations
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

DROP POLICY IF EXISTS service_role_all
    ON public.research_lab_source_add_reward_events;
CREATE POLICY service_role_all
    ON public.research_lab_source_add_reward_events
    FOR ALL TO service_role
    USING (true)
    WITH CHECK (true);

ALTER VIEW public.research_lab_source_add_reward_current
    SET (security_invoker = true);
ALTER VIEW public.research_lab_icp_churn_reversal_report
    SET (security_invoker = true);

REVOKE ALL ON TABLE
    public.research_lab_source_add_reward_current,
    public.research_lab_icp_churn_reversal_report
FROM PUBLIC, anon, authenticated;

GRANT SELECT ON TABLE
    public.research_lab_source_add_reward_current,
    public.research_lab_icp_churn_reversal_report
TO service_role;

COMMENT ON VIEW public.research_lab_icp_churn_reversal_report IS
    'Service-role-only Research Lab observability for per-ICP score reversals; contains no private ICP text or provider payloads.';

NOTIFY pgrst, 'reload schema';

COMMIT;

-- Keep generator examples in private storage, never in the public ICP view.
-- Preserve the existing expiry filter, public requirements and read-only ACLs.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $hide_public_icp_examples$
BEGIN
  IF to_regclass('public.qualification_expired_icp_sets') IS NULL THEN
    RETURN;
  END IF;
  EXECUTE $view$
    CREATE OR REPLACE VIEW public.qualification_expired_icp_sets
    WITH (security_invoker=false) AS
    SELECT set_id, created_at, active_from, active_until,
      CASE WHEN icps IS NULL THEN NULL ELSE COALESCE((
        SELECT jsonb_agg(icp - 'verified_example_company' ORDER BY ordinal)
        FROM jsonb_array_elements(icps) WITH ORDINALITY AS entries(icp, ordinal)
      ), '[]'::jsonb) END AS icps
    FROM public.qualification_private_icp_sets
    WHERE is_active = false
    ORDER BY set_id DESC
  $view$;
END;
$hide_public_icp_examples$;

NOTIFY pgrst, 'reload schema';
COMMIT;

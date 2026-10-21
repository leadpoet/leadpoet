-- Legacy internal audit tables are not public API resources. Keep their data,
-- owner, service-role access and trusted read-only access unchanged.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

DO $internal_object_permissions$
DECLARE
  object_name TEXT;
BEGIN
  FOREACH object_name IN ARRAY ARRAY[
    'lab_arena_sep15_baseline_rerun_audit',
    'lab_arena_sep15_rerun_release_authority',
    'lab_arena_sep16_open_config_audit'
  ] LOOP
    -- These historical audit tables need not exist on a fresh installation.
    IF to_regclass(format('public.%I', object_name)) IS NOT NULL THEN
      EXECUTE format('REVOKE ALL ON TABLE public.%I FROM PUBLIC, anon, authenticated', object_name);
      EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', object_name);
    END IF;
  END LOOP;

  IF to_regclass('public.qualification_expired_icp_sets') IS NOT NULL THEN
    -- This updatable view exposes only expired ICPs. Its definer privileges
    -- are required for public reads without access to the private base table.
    -- API roles may read it, but all writes must use the protected base table.
    REVOKE ALL ON TABLE public.qualification_expired_icp_sets
      FROM PUBLIC, anon, authenticated, service_role;
    GRANT SELECT ON TABLE public.qualification_expired_icp_sets
      TO anon, authenticated, service_role;
  END IF;
END;
$internal_object_permissions$;

NOTIFY pgrst, 'reload schema';
COMMIT;

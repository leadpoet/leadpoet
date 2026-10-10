-- Keep stored publication authority intact while excluding result-unused costs.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';

CREATE TEMP TABLE lab_arena_449_schema_acl ON COMMIT DROP AS
SELECT namespace.nspacl AS acl,
       pg_catalog.has_schema_privilege('lab_arena_owner', 'public', 'CREATE')
         AS had_create
FROM pg_catalog.pg_namespace AS namespace
WHERE namespace.nspname = 'public';
DO $temporary_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_449_schema_acl) THEN
    GRANT CREATE ON SCHEMA public TO lab_arena_owner;
  END IF;
END;
$temporary_create$;

CREATE OR REPLACE VIEW public.lab_arena_published_results_v1
WITH (security_invoker = true)
AS
SELECT r.round_id, r.status, r.participants, r.benchmark_ref,
       r.evaluation_date, r.icp_set_date,
       CASE
         WHEN pg_catalog.jsonb_typeof(r.publication_doc) = 'object'
          AND pg_catalog.jsonb_typeof(r.publication_doc -> 'final_ranking') = 'array'
         THEN pg_catalog.jsonb_set(
           r.publication_doc,
           '{final_ranking}',
           (
             SELECT COALESCE(
               pg_catalog.jsonb_agg(
                 CASE WHEN pg_catalog.jsonb_typeof(entry.value) = 'object'
                      THEN entry.value - 'cost_summary'
                      ELSE entry.value END
                 ORDER BY entry.ordinality
               ),
               '[]'::JSONB
             )
             FROM pg_catalog.jsonb_array_elements(
               r.publication_doc -> 'final_ranking'
             ) WITH ORDINALITY AS entry(value, ordinality)
           ),
           false
         )
         ELSE r.publication_doc
       END AS publication_doc,
       r.configuration_doc
FROM public.lab_arena_rounds AS r
WHERE r.status = 'published';

ALTER VIEW public.lab_arena_published_results_v1 OWNER TO lab_arena_owner;
REVOKE ALL ON TABLE public.lab_arena_published_results_v1
  FROM PUBLIC, anon, authenticated, service_role, lab_arena_service;
GRANT SELECT ON TABLE public.lab_arena_published_results_v1
  TO lab_arena_service;

DO $restore_create$
BEGIN
  IF NOT (SELECT had_create FROM pg_temp.lab_arena_449_schema_acl) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
       IS DISTINCT FROM (SELECT acl FROM pg_temp.lab_arena_449_schema_acl) THEN
    RAISE EXCEPTION 'lab_arena_published_results_schema_acl_changed';
  END IF;
END;
$restore_create$;
NOTIFY pgrst, 'reload schema';
COMMIT;

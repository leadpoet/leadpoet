-- Research Lab egress reduction: true database-side trajectory discovery delta.
--
-- project_completed_runs() and backfill_corpus_trace_rows() each downloaded up
-- to 5000 terminal run rows and then decided per-run whether a trajectory /
-- trace existed. research_trajectories is keyed only on the deterministic
-- double-hashed uuid5 trajectory_id (no run_id column), which is why the
-- existence test lived client-side. These functions reproduce that id (and the
-- engine execution_trace id) exactly in SQL, so the whole "next N runs that
-- still need work" question is answered by one server-side query returning only
-- the run ids -- no bulk download, no per-run round-trips.
--
-- The id reproduction is verified byte-for-byte against the Python
-- trajectory_id_for_run / execution_trace_id_for_run on 200 live production
-- runs (see tests/test_migrations_postgres_integration.py and the standalone
-- parity check). uuid5 = sha1(namespace_bytes || name) with the version/variant
-- bits set; the name is 'sha256:' || sha256hex(canonical_json(parts)), matching
-- store.deterministic_uuid()/canonical_hash(). pgcrypto lives in the extensions
-- schema on Supabase (confirmed live).

BEGIN;

-- uuid5(RESEARCH_LAB_UUID_NAMESPACE, 'sha256:'||sha256hex(p_canonical_json)).
-- RESEARCH_LAB_UUID_NAMESPACE = 51d344ea-fd61-519d-b133-54dac81449dd.
CREATE OR REPLACE FUNCTION public.research_lab_deterministic_uuid(p_canonical_json text)
RETURNS uuid
LANGUAGE sql
IMMUTABLE
SET search_path = ''
AS $$
  WITH n AS (
    SELECT 'sha256:' || pg_catalog.encode(
      extensions.digest(pg_catalog.convert_to(p_canonical_json, 'UTF8'), 'sha256'),
      'hex') AS name
  ),
  h AS (
    SELECT pg_catalog.substr(
      extensions.digest(
        '\x51d344eafd61519db13354dac81449dd'::bytea
          || pg_catalog.convert_to((SELECT name FROM n), 'UTF8'),
        'sha1'),
      1, 16) AS b
  ),
  v AS (
    SELECT pg_catalog.set_byte(
             pg_catalog.set_byte(b, 6, (pg_catalog.get_byte(b, 6) & 15) | 80),
             8, (pg_catalog.get_byte(b, 8) & 63) | 128) AS b
    FROM h
  )
  SELECT pg_catalog.encode((SELECT b FROM v), 'hex')::uuid
$$;

-- trajectory_id_for_run(run_id) = deterministic_uuid("research_trajectory", run_id).
CREATE OR REPLACE FUNCTION public.research_lab_trajectory_id(p_run_id uuid)
RETURNS uuid
LANGUAGE sql
IMMUTABLE
SET search_path = ''
AS $$
  SELECT public.research_lab_deterministic_uuid(
    '["research_trajectory","' || pg_catalog.lower(p_run_id::text) || '"]')
$$;

-- execution_trace_id_for_run(run_id) = deterministic_uuid("execution_trace", run_id, "engine").
CREATE OR REPLACE FUNCTION public.research_lab_execution_trace_id(p_run_id uuid)
RETURNS uuid
LANGUAGE sql
IMMUTABLE
SET search_path = ''
AS $$
  SELECT public.research_lab_deterministic_uuid(
    '["execution_trace","' || pg_catalog.lower(p_run_id::text) || '","engine"]')
$$;

-- Item 3: the next p_limit terminal runs with NO projected trajectory yet.
CREATE OR REPLACE FUNCTION public.research_lab_next_unprojected_terminal_runs(
    p_limit        integer,
    p_newest_first boolean DEFAULT false
)
RETURNS TABLE(run_id uuid)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT q.run_id
  FROM public.research_loop_run_queue_current q
  WHERE q.current_queue_status IN ('completed', 'failed')
    AND NOT EXISTS (
      SELECT 1 FROM public.research_trajectories t
      WHERE t.trajectory_id = public.research_lab_trajectory_id(q.run_id)
    )
  ORDER BY
    (CASE WHEN p_newest_first THEN q.current_status_at END) DESC NULLS LAST,
    (CASE WHEN NOT p_newest_first THEN q.current_status_at END) ASC NULLS LAST
  LIMIT GREATEST(1, p_limit);
$$;

-- Item 3b: the next p_limit PROJECTED terminal runs still missing their engine
-- execution_trace row (the dominant "needs corpus backfill" signal). The
-- provider-usage ledger has no run_id column, so its completeness stays the
-- per-run projector check; this delta only stops re-inspecting fully-traced
-- runs, which was the repeated-scan cost.
CREATE OR REPLACE FUNCTION public.research_lab_terminal_runs_missing_traces(
    p_limit        integer,
    p_newest_first boolean DEFAULT true
)
RETURNS TABLE(run_id uuid)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = ''
AS $$
  SELECT q.run_id
  FROM public.research_loop_run_queue_current q
  WHERE q.current_queue_status IN ('completed', 'failed')
    AND EXISTS (
      SELECT 1 FROM public.research_trajectories t
      WHERE t.trajectory_id = public.research_lab_trajectory_id(q.run_id)
    )
    AND NOT EXISTS (
      SELECT 1 FROM public.execution_traces e
      WHERE e.run_id = public.research_lab_execution_trace_id(q.run_id)
    )
  ORDER BY
    (CASE WHEN p_newest_first THEN q.current_status_at END) DESC NULLS LAST,
    (CASE WHEN NOT p_newest_first THEN q.current_status_at END) ASC NULLS LAST
  LIMIT GREATEST(1, p_limit);
$$;

REVOKE ALL ON FUNCTION public.research_lab_deterministic_uuid(text) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_trajectory_id(uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_execution_trace_id(uuid) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_next_unprojected_terminal_runs(integer, boolean) FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.research_lab_terminal_runs_missing_traces(integer, boolean) FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.research_lab_deterministic_uuid(text) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_trajectory_id(uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_execution_trace_id(uuid) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_next_unprojected_terminal_runs(integer, boolean) TO service_role;
GRANT EXECUTE ON FUNCTION public.research_lab_terminal_runs_missing_traces(integer, boolean) TO service_role;

COMMIT;

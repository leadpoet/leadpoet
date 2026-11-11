-- Execute each of these three top-level statements separately in one
-- session-affine autocommit session (for example psql -v ON_ERROR_STOP=1 -f).
-- Do not wrap this file in BEGIN or send the complete file as one SQL query.
-- Bound unchanged run reads to their round before ordered run-ID pagination.
-- Keep stage, kind, status and submission predicates in their current readers.
-- No payload columns, partial predicates, row changes, functions or grants.
-- Failed/invalid concurrent builds require explicit approved repair; replay
-- never silently accepts an unusable or differently defined existing object.
DO $guard688$
DECLARE
  v_index OID := pg_catalog.to_regclass('public.lab_arena_runs_round_run_idx');
BEGIN
  IF (SELECT pg_catalog.pg_get_userbyid(relowner) FROM pg_catalog.pg_class
      WHERE oid='public.lab_arena_runs'::REGCLASS) IS DISTINCT FROM 'lab_arena_owner' THEN
    RAISE EXCEPTION '688 runs owner differs' USING ERRCODE='55000';
  END IF;
  IF v_index IS NOT NULL AND NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_index i
    JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
    WHERE c.oid=v_index AND c.relkind='i'
      AND i.indrelid='public.lab_arena_runs'::REGCLASS
      AND i.indisvalid AND i.indisready AND i.indislive
      AND NOT i.indisunique AND c.relacl IS NULL
      AND pg_catalog.pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND pg_catalog.pg_get_indexdef(c.oid)=
        'CREATE INDEX lab_arena_runs_round_run_idx ON public.lab_arena_runs USING btree (round_id, run_id)'
  ) THEN
    RAISE EXCEPTION '688 existing index differs or is unusable' USING ERRCODE='55000';
  END IF;
END;
$guard688$;

CREATE INDEX CONCURRENTLY IF NOT EXISTS lab_arena_runs_round_run_idx
ON public.lab_arena_runs (round_id, run_id);

DO $verify688$
DECLARE
  v_index OID := pg_catalog.to_regclass('public.lab_arena_runs_round_run_idx');
BEGIN
  IF (SELECT pg_catalog.pg_get_userbyid(relowner) FROM pg_catalog.pg_class
      WHERE oid='public.lab_arena_runs'::REGCLASS) IS DISTINCT FROM 'lab_arena_owner' THEN
    RAISE EXCEPTION '688 runs owner differs' USING ERRCODE='55000';
  END IF;
  IF v_index IS NULL OR NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_index i
    JOIN pg_catalog.pg_class c ON c.oid=i.indexrelid
    WHERE c.oid=v_index AND c.relkind='i'
      AND i.indrelid='public.lab_arena_runs'::REGCLASS
      AND i.indisvalid AND i.indisready AND i.indislive
      AND NOT i.indisunique AND c.relacl IS NULL
      AND pg_catalog.pg_get_userbyid(c.relowner)='lab_arena_owner'
      AND pg_catalog.pg_get_indexdef(c.oid)=
        'CREATE INDEX lab_arena_runs_round_run_idx ON public.lab_arena_runs USING btree (round_id, run_id)'
  ) THEN
    RAISE EXCEPTION '688 existing index differs or is unusable' USING ERRCODE='55000';
  END IF;
END;
$verify688$;

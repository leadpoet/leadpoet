-- Leadpoet's admin site reads captured model-call usage for one submission
-- as aggregates only. A NOLOGIN reader role can execute this one SECURITY
-- DEFINER function and holds no table privileges. It returns no hotkeys,
-- credentials, prompts, bodies or model output. No stored row, table ACL,
-- existing function or existing role changes.
--
-- Count saved provider usage with the terminal/token rules used by
-- lab_arena/training_export.py and training_thinking.py:
-- a call is one openrouter.chat/openrouter.responses call_identity with a
-- reservation in the submission's execute runs; its latest terminal (highest
-- entry_id) must hold a saved 2xx success (broker
-- _has_successful_terminal_response). Its body is strict base64 of at most
-- 16 MiB, UTF-8 (optional BOM) JSON. Reasoning tokens are the largest
-- non-negative integer with 1 to 19 digits at usage.reasoning_tokens,
-- usage.completion_tokens_details.reasoning_tokens or
-- usage.output_tokens_details.reasoning_tokens. Any other body counts as a
-- completed call with no reported usage and never raises.
-- This is not an export-completeness or training-eligibility check: it does not
-- run the exporter's privacy and structure sanitizer. An aggregate above the
-- BIGINT range returns NULL (unknown) while retaining the usage call count.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $capture_reader_role593$
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('leadpoet.lab-arena-roles.v1', 0));
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader') THEN
    CREATE ROLE leadpoet_capture_reader NOLOGIN NOINHERIT;
  END IF;
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader'
      AND (rolcanlogin OR rolsuper OR rolbypassrls OR rolreplication OR rolcreatedb
           OR rolcreaterole OR rolinherit))
    OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members m
      WHERE m.member='leadpoet_capture_reader'::regrole) THEN
    RAISE EXCEPTION 'capture reader593 role attributes differ' USING ERRCODE='55000';
  END IF;
  ALTER ROLE leadpoet_capture_reader SET statement_timeout='60s';
  -- PostgREST switches into request roles. The hosted-compatible name is
  -- granted here; an RDS authenticator with another reviewed name is granted
  -- by its owner (lab_arena/RUNBOOK.md, "Capture summary reader").
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='authenticator')
     AND NOT pg_catalog.pg_has_role('authenticator','leadpoet_capture_reader','MEMBER') THEN
    EXECUTE 'GRANT leadpoet_capture_reader TO authenticator WITH INHERIT FALSE, SET TRUE';
  END IF;
  GRANT USAGE ON SCHEMA public TO leadpoet_capture_reader;
END;
$capture_reader_role593$;
CREATE TEMP TABLE capture_reader593_schema ON COMMIT DROP AS
  SELECT nspacl, pg_catalog.has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
  FROM pg_catalog.pg_namespace WHERE nspname='public';
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_capture_summary_v1(p_round_id TEXT, p_submission_id TEXT)
RETURNS TABLE (
  round_id TEXT, submission_id TEXT, execute_runs INTEGER, icp_positions INTEGER[],
  model_calls INTEGER, usage_calls INTEGER, reasoning_tokens BIGINT,
  first_call_at TIMESTAMPTZ, last_call_at TIMESTAMPTZ)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $summary593$
DECLARE
  head RECORD;
  run_ids TEXT[];
  encoded TEXT;
  raw BYTEA;
  body_text TEXT;
  usage_doc JSON;
  tokens NUMERIC;
  token_total NUMERIC := 0;
BEGIN
  round_id := p_round_id;
  submission_id := p_submission_id;
  model_calls := 0;
  usage_calls := 0;
  SELECT count(*)::INTEGER,
    coalesce(array_agg(DISTINCT r.icp_position::INTEGER ORDER BY r.icp_position::INTEGER), '{}'),
    coalesce(array_agg(r.run_id), '{}')
  INTO execute_runs, icp_positions, run_ids
  FROM public.lab_arena_runs AS r
  WHERE r.round_id = p_round_id AND r.submission_id = p_submission_id AND r.kind = 'execute';

  FOR head IN
    -- An id array keeps the ledger read on lab_arena_ledger_run_idx.
    WITH scoped AS MATERIALIZED (
      SELECT l.entry_id, l.entry_kind, l.call_identity, l.created_at
      FROM public.lab_arena_ledger AS l
      WHERE l.run_id = ANY (run_ids) AND l.call_identity IS NOT NULL AND l.provider = 'openrouter'
        AND l.operation_id IN ('openrouter.chat', 'openrouter.responses')
    ), reserved AS (
      SELECT s.call_identity, min(s.created_at) AS started_at
      FROM scoped AS s WHERE s.entry_kind = 'reservation' GROUP BY s.call_identity
    ), heads AS (
      SELECT DISTINCT ON (s.call_identity) s.call_identity, s.entry_id, s.created_at
      FROM scoped AS s
      WHERE s.entry_kind IN ('settlement', 'uncertain', 'recovery', 'refusal')
      ORDER BY s.call_identity, s.entry_id DESC
    )
    SELECT reserved.started_at, heads.created_at AS finished_at,
      l.terminal_response -> 'body_b64' AS body
    FROM heads
    JOIN reserved ON reserved.call_identity = heads.call_identity
    JOIN public.lab_arena_ledger AS l ON l.entry_id = heads.entry_id
    WHERE l.terminal_response -> 'call_succeeded' = 'true'::JSONB
      AND jsonb_typeof(l.terminal_response -> 'status') = 'number'
      AND (l.terminal_response ->> 'status') ~ '^2[0-9][0-9]$'
  LOOP
    model_calls := model_calls + 1;
    first_call_at := least(first_call_at, head.started_at);
    last_call_at := greatest(last_call_at, head.finished_at);
    tokens := NULL;
    IF jsonb_typeof(head.body) = 'string' THEN
      encoded := head.body #>> '{}';
      -- Python b64decode(validate=True) and MAX_RECEIPT_BYTES.
      IF length(encoded) % 4 = 0 AND encoded ~ '^[A-Za-z0-9+/]*={0,2}$'
         AND length(encoded) / 4 * 3 - (length(encoded) - length(rtrim(encoded, '='))) <= 16777216 THEN
        BEGIN
          raw := decode(encoded, 'base64');
          IF substring(raw FROM 1 FOR 3) = decode('efbbbf', 'hex') THEN
            raw := substring(raw FROM 4);
          END IF;
          body_text := convert_from(raw, 'UTF8');
          -- JSON keeps number spelling, so 5.0 or 1e2 stays non-integer as
          -- in Python.
          BEGIN
            usage_doc := body_text::JSON -> 'usage';
          EXCEPTION WHEN untranslatable_character OR invalid_text_representation THEN
            -- Python also accepts NUL and lone surrogate escapes, which
            -- PostgreSQL rejects. They sit only inside strings, so replacing
            -- them never changes a usage number.
            usage_doc := regexp_replace(body_text,
              '\\u(0000|[dD][89a-fA-F][0-9a-fA-F]{2})', '\\ufffd', 'g')::JSON -> 'usage';
          END;
          IF json_typeof(usage_doc) = 'object' THEN
            SELECT max(v::TEXT::NUMERIC) INTO tokens
            FROM unnest(ARRAY[usage_doc -> 'reasoning_tokens',
                              usage_doc -> 'completion_tokens_details' -> 'reasoning_tokens',
                              usage_doc -> 'output_tokens_details' -> 'reasoning_tokens']) AS v
            WHERE json_typeof(v) = 'number' AND v::TEXT ~ '^-?[0-9]{1,19}$';
          END IF;
        EXCEPTION
          WHEN character_not_in_repertoire OR invalid_text_representation
            OR untranslatable_character OR statement_too_complex THEN
            tokens := NULL;
        END;
      END IF;
    END IF;
    IF tokens >= 0 THEN
      usage_calls := usage_calls + 1;
      token_total := token_total + tokens;
    END IF;
  END LOOP;
  IF usage_calls > 0 AND token_total <= 9223372036854775807 THEN
    reasoning_tokens := token_total::BIGINT;
  END IF;
  RETURN NEXT;
END;
$summary593$;
ALTER FUNCTION public.lab_arena_capture_summary_v1(TEXT, TEXT) OWNER TO lab_arena_owner;

DO $capture_reader_acl593$
DECLARE
  target CONSTANT REGPROCEDURE := 'public.lab_arena_capture_summary_v1(text,text)'::regprocedure;
  grantee TEXT;
BEGIN
  -- Default privileges may name other roles; revoke every non-owner grant.
  FOR grantee IN
    SELECT DISTINCT CASE WHEN a.grantee = 0 THEN 'PUBLIC' ELSE quote_ident(pg_get_userbyid(a.grantee)) END
    FROM pg_catalog.pg_proc p, aclexplode(coalesce(p.proacl, acldefault('f', p.proowner))) AS a
    WHERE p.oid = target AND a.grantee <> p.proowner
  LOOP
    EXECUTE format('REVOKE ALL ON FUNCTION %s FROM %s', target, grantee);
  END LOOP;
  REVOKE ALL ON FUNCTION public.lab_arena_capture_summary_v1(TEXT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
  GRANT EXECUTE ON FUNCTION public.lab_arena_capture_summary_v1(TEXT, TEXT) TO leadpoet_capture_reader;
  IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
        p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target) IS DISTINCT FROM
     '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader593 function security differs' USING ERRCODE='55000';
  END IF;
  -- The reader reaches Arena data only through this function.
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c, aclexplode(c.relacl) AS a
             WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_attribute att, aclexplode(att.attacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p, aclexplode(p.proacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole AND p.oid <> target) THEN
    RAISE EXCEPTION 'capture reader593 holds other privileges' USING ERRCODE='55000';
  END IF;
  IF NOT (SELECT had_create FROM pg_temp.capture_reader593_schema) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
     IS DISTINCT FROM (SELECT nspacl FROM pg_temp.capture_reader593_schema) THEN
    RAISE EXCEPTION 'capture reader593 schema ACL changed' USING ERRCODE='55000';
  END IF;
END;
$capture_reader_acl593$;
NOTIFY pgrst, 'reload schema';
COMMIT;

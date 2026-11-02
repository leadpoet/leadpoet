-- Additive aggregate-only capture API. V1 remains byte-for-byte unchanged.
-- Uses the exact 593/595 call, ICP, timestamp and reasoning-token semantics.
-- New token fields accept only 1-19 digit nonnegative whole JSON numbers;
-- missing/invalid totals and sums above BIGINT remain NULL, including per model.
-- Completion/output tokens include provider-reported reasoning; do not subtract it.
-- Models use only bounded identifier-shaped response.model, never request/body text.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $capture_reader_guard598$
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('leadpoet.lab-arena-roles.v1', 0));
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader'
       AND (rolcanlogin OR rolsuper OR rolbypassrls OR rolreplication OR rolcreatedb
            OR rolcreaterole OR rolinherit))
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members m
       WHERE m.member='leadpoet_capture_reader'::regrole) THEN
    RAISE EXCEPTION 'capture reader598 role attributes differ' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_capture_summary_v1(text,text)'::regprocedure),'sha256'),'hex')
       IS DISTINCT FROM '10c9ee3a9543bf65fb95c198efdbf14bf6ece7b340f9fceccc4e1637de5cb2c4' THEN
    RAISE EXCEPTION 'capture reader598 v1 preimage differs' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
       p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang
      WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure)
      IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader598 v1 security differs' USING ERRCODE='55000';
  END IF;
END;
$capture_reader_guard598$;
CREATE TEMP TABLE capture_reader598_schema ON COMMIT DROP AS
  SELECT nspacl, pg_catalog.has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
  FROM pg_catalog.pg_namespace WHERE nspname='public';
CREATE TEMP TABLE capture_reader598_v1 ON COMMIT DROP AS
  SELECT pg_catalog.to_jsonb(p) AS catalog FROM pg_catalog.pg_proc p
  WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_capture_summary_v2(p_round_id TEXT, p_submission_id TEXT)
RETURNS TABLE (
  round_id TEXT, submission_id TEXT, execute_runs INTEGER, icp_positions INTEGER[],
  model_calls INTEGER, usage_calls INTEGER, reasoning_tokens BIGINT,
  first_call_at TIMESTAMPTZ, last_call_at TIMESTAMPTZ,
  input_tokens BIGINT, output_tokens BIGINT, cached_input_tokens BIGINT,
  token_calls INTEGER, models JSONB)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $summary598$
DECLARE
  head RECORD;
  run_ids TEXT[];
  encoded TEXT;
  raw BYTEA;
  body_text TEXT;
  response_doc JSON;
  usage_doc JSON;
  call_input NUMERIC;
  call_output NUMERIC;
  call_cached NUMERIC;
  input_total NUMERIC := 0;
  output_total NUMERIC := 0;
  cached_total NUMERIC := 0;
  input_calls INTEGER := 0;
  output_calls INTEGER := 0;
  cached_calls INTEGER := 0;
  model_name TEXT;
  model_totals JSONB := '{}'::JSONB;
  model_previous JSONB;
  tokens NUMERIC;
  token_total NUMERIC := 0;
BEGIN
  round_id := p_round_id;
  submission_id := p_submission_id;
  model_calls := 0;
  usage_calls := 0;
  token_calls := 0;
  SELECT count(*)::INTEGER, coalesce(array_agg(r.run_id), '{}')
  INTO execute_runs, run_ids
  FROM public.lab_arena_runs AS r
  WHERE r.round_id = p_round_id AND r.submission_id = p_submission_id AND r.kind = 'execute';

  -- Match the public cost summary's call_count > 0 trace definition across
  -- providers and terminal states. Merely creating an execute run is not a call.
  SELECT coalesce(array_agg(DISTINCT r.icp_position::INTEGER ORDER BY r.icp_position::INTEGER), '{}')
  INTO icp_positions
  FROM public.lab_arena_runs AS r
  WHERE r.run_id = ANY (run_ids)
    AND EXISTS (
      SELECT 1 FROM public.lab_arena_ledger AS l
      WHERE l.run_id = r.run_id AND l.round_id = p_round_id
        AND l.submission_id = p_submission_id AND l.call_identity IS NOT NULL
    );

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
    call_input := NULL;
    call_output := NULL;
    call_cached := NULL;
    model_name := 'unknown';
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
            response_doc := body_text::JSON;
            usage_doc := response_doc -> 'usage';
          EXCEPTION WHEN untranslatable_character OR invalid_text_representation THEN
            -- Python also accepts NUL and lone surrogate escapes, which
            -- PostgreSQL rejects. They sit only inside strings, so replacing
            -- them never changes a usage number.
            response_doc := regexp_replace(body_text,
              '\\u(0000|[dD][89a-fA-F][0-9a-fA-F]{2})', '\\ufffd', 'g')::JSON;
            usage_doc := response_doc -> 'usage';
          END;
          IF json_typeof(usage_doc) = 'object' THEN
            SELECT max(v::TEXT::NUMERIC) INTO tokens
            FROM unnest(ARRAY[usage_doc -> 'reasoning_tokens',
                              usage_doc -> 'completion_tokens_details' -> 'reasoning_tokens',
                              usage_doc -> 'output_tokens_details' -> 'reasoning_tokens']) AS v
            WHERE json_typeof(v) = 'number' AND v::TEXT ~ '^-?[0-9]{1,19}$';
            -- Aliases are alternatives, never additive: take the first valid
            -- whole nonnegative JSON number in canonical order. Preserve
            -- spelling so strings, fractions, exponents and -0 are rejected.
            SELECT v::TEXT::NUMERIC INTO call_input
            FROM unnest(ARRAY[usage_doc -> 'prompt_tokens', usage_doc -> 'input_tokens'])
              WITH ORDINALITY AS candidate(v, priority)
            WHERE json_typeof(v) = 'number' AND v::TEXT ~ '^[0-9]{1,19}$'
            ORDER BY priority LIMIT 1;
            SELECT v::TEXT::NUMERIC INTO call_output
            FROM unnest(ARRAY[usage_doc -> 'completion_tokens', usage_doc -> 'output_tokens'])
              WITH ORDINALITY AS candidate(v, priority)
            WHERE json_typeof(v) = 'number' AND v::TEXT ~ '^[0-9]{1,19}$'
            ORDER BY priority LIMIT 1;
            SELECT v::TEXT::NUMERIC INTO call_cached
            FROM unnest(ARRAY[usage_doc -> 'prompt_tokens_details' -> 'cached_tokens',
                              usage_doc -> 'input_tokens_details' -> 'cached_tokens'])
              WITH ORDINALITY AS candidate(v, priority)
            WHERE json_typeof(v) = 'number' AND v::TEXT ~ '^[0-9]{1,19}$'
            ORDER BY priority LIMIT 1;
          END IF;
          -- A malformed or non-identifier model must never expose response text
          -- or invalidate otherwise valid usage.
          BEGIN
            IF json_typeof(response_doc -> 'model') = 'string'
               AND (response_doc ->> 'model') COLLATE "C" ~ '^[A-Za-z0-9._:/@+-]{1,200}$' THEN
              model_name := response_doc ->> 'model';
            END IF;
          EXCEPTION WHEN character_not_in_repertoire OR invalid_text_representation
            OR untranslatable_character THEN
            model_name := 'unknown';
          END;
        EXCEPTION
          WHEN character_not_in_repertoire OR invalid_text_representation
            OR untranslatable_character OR statement_too_complex THEN
            tokens := NULL;
            call_input := NULL;
            call_output := NULL;
            call_cached := NULL;
            model_name := 'unknown';
        END;
      END IF;
    END IF;
    IF tokens >= 0 THEN
      usage_calls := usage_calls + 1;
      token_total := token_total + tokens;
    END IF;
    IF call_input IS NOT NULL THEN
      input_calls := input_calls + 1;
      input_total := input_total + call_input;
    END IF;
    IF call_output IS NOT NULL THEN
      output_calls := output_calls + 1;
      output_total := output_total + call_output;
    END IF;
    IF call_cached IS NOT NULL THEN
      cached_calls := cached_calls + 1;
      cached_total := cached_total + call_cached;
    END IF;
    IF call_input IS NOT NULL AND call_output IS NOT NULL THEN
      token_calls := token_calls + 1;
    END IF;
    model_previous := model_totals -> model_name;
    model_totals := jsonb_set(model_totals, ARRAY[model_name], jsonb_build_object(
      'calls', coalesce((model_previous ->> 'calls')::INTEGER, 0) + 1,
      'input_tokens', CASE WHEN call_input IS NOT NULL OR model_previous ->> 'input_tokens' IS NOT NULL
        THEN coalesce((model_previous ->> 'input_tokens')::NUMERIC, 0) + coalesce(call_input, 0) END,
      'output_tokens', CASE WHEN call_output IS NOT NULL OR model_previous ->> 'output_tokens' IS NOT NULL
        THEN coalesce((model_previous ->> 'output_tokens')::NUMERIC, 0) + coalesce(call_output, 0) END,
      'reasoning_tokens', CASE WHEN tokens >= 0 OR model_previous ->> 'reasoning_tokens' IS NOT NULL
        THEN coalesce((model_previous ->> 'reasoning_tokens')::NUMERIC, 0)
          + CASE WHEN tokens >= 0 THEN tokens ELSE 0 END END));
  END LOOP;
  IF input_calls > 0 AND input_total <= 9223372036854775807 THEN
    input_tokens := input_total::BIGINT;
  END IF;
  IF output_calls > 0 AND output_total <= 9223372036854775807 THEN
    output_tokens := output_total::BIGINT;
  END IF;
  IF cached_calls > 0 AND cached_total <= 9223372036854775807 THEN
    cached_input_tokens := cached_total::BIGINT;
  END IF;
  -- Keep at most 50 unique names. When folding is needed, reserve 'other'
  -- for the remainder and include any real model named 'other' there. This
  -- prevents duplicate labels without hiding it in ordinary <=50-model output.
  WITH ranked AS (
    SELECT key AS model, value,
      row_number() OVER (ORDER BY (key = 'other'), (value ->> 'calls')::INTEGER DESC, key COLLATE "C") AS rank,
      count(*) OVER () AS total
    FROM jsonb_each(model_totals)
  ), grouped AS (
    SELECT CASE WHEN total > 50 AND (rank > 49 OR model = 'other') THEN 'other' ELSE model END AS model,
      sum((value ->> 'calls')::INTEGER)::INTEGER AS calls,
      sum((value ->> 'input_tokens')::NUMERIC) AS input_sum,
      sum((value ->> 'output_tokens')::NUMERIC) AS output_sum,
      sum((value ->> 'reasoning_tokens')::NUMERIC) AS reasoning_sum
    FROM ranked GROUP BY 1
  )
  SELECT coalesce(jsonb_agg(jsonb_build_object(
      'model', model, 'calls', calls,
      'input_tokens', CASE WHEN input_sum <= 9223372036854775807 THEN input_sum::BIGINT END,
      'output_tokens', CASE WHEN output_sum <= 9223372036854775807 THEN output_sum::BIGINT END,
      'reasoning_tokens', CASE WHEN reasoning_sum <= 9223372036854775807 THEN reasoning_sum::BIGINT END)
      ORDER BY calls DESC, model COLLATE "C"), '[]'::JSONB)
  INTO models FROM grouped;
  IF usage_calls > 0 AND token_total <= 9223372036854775807 THEN
    reasoning_tokens := token_total::BIGINT;
  END IF;
  RETURN NEXT;
END;
$summary598$;
ALTER FUNCTION public.lab_arena_capture_summary_v2(TEXT, TEXT) OWNER TO lab_arena_owner;

DO $capture_reader_acl598$
DECLARE
  target CONSTANT REGPROCEDURE := 'public.lab_arena_capture_summary_v2(text,text)'::regprocedure;
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
  REVOKE ALL ON FUNCTION public.lab_arena_capture_summary_v2(TEXT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
  GRANT EXECUTE ON FUNCTION public.lab_arena_capture_summary_v2(TEXT, TEXT) TO leadpoet_capture_reader;
  IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
        p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target) IS DISTINCT FROM
     '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader598 function security differs' USING ERRCODE='55000';
  END IF;
  -- The reader reaches Arena data only through the two aggregate functions.
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c, aclexplode(c.relacl) AS a
             WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_attribute att, aclexplode(att.attacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p, aclexplode(p.proacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole AND p.oid NOT IN (target, 'public.lab_arena_capture_summary_v1(text,text)'::regprocedure)) THEN
    RAISE EXCEPTION 'capture reader598 holds other privileges' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p
      WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure)
      IS DISTINCT FROM (SELECT catalog FROM pg_temp.capture_reader598_v1) THEN
    RAISE EXCEPTION 'capture reader598 v1 catalog changed' USING ERRCODE='55000';
  END IF;
  IF NOT (SELECT had_create FROM pg_temp.capture_reader598_schema) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
     IS DISTINCT FROM (SELECT nspacl FROM pg_temp.capture_reader598_schema) THEN
    RAISE EXCEPTION 'capture reader598 schema ACL changed' USING ERRCODE='55000';
  END IF;
END;
$capture_reader_acl598$;
NOTIFY pgrst, 'reload schema';
COMMIT;

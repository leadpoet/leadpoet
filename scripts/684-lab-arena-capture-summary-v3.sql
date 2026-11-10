-- Additive aggregate-only capture API. V1 and V2 remain byte-for-byte unchanged.
-- V3 is V2 plus reasoning_calls: completed calls whose provider-reported reasoning
-- tokens are greater than zero. A call with missing, invalid or zero reasoning
-- usage is not counted. reasoning_capture partitions saved successful replies by
-- readable response text and provider completion, without training-readiness claims.
-- Every V2 column keeps the exact deployed post-620 semantics.
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
DO $capture_reader_guard684$
BEGIN
  PERFORM pg_catalog.pg_advisory_xact_lock(
    pg_catalog.hashtextextended('leadpoet.lab-arena-roles.v1', 0));
  IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader')
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_roles WHERE rolname='leadpoet_capture_reader'
       AND (rolcanlogin OR rolsuper OR rolbypassrls OR rolreplication OR rolcreatedb
            OR rolcreaterole OR rolinherit))
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_auth_members m
       WHERE m.member='leadpoet_capture_reader'::regrole) THEN
    RAISE EXCEPTION 'capture reader684 role attributes differ' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_capture_summary_v1(text,text)'::regprocedure),'sha256'),'hex')
       IS DISTINCT FROM 'f45011f2996d630902104d3fe5a8ed89ca1923a93316d9f09a6aec37de7f42b1' THEN
    RAISE EXCEPTION 'capture reader684 v1 preimage differs' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
       p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang
      WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure)
      IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader684 v1 security differs' USING ERRCODE='55000';
  END IF;
  IF pg_catalog.encode(extensions.digest(pg_catalog.pg_get_functiondef(
       'public.lab_arena_capture_summary_v2(text,text)'::regprocedure),'sha256'),'hex')
       IS DISTINCT FROM '5b6218fe07c41e6d9ab938ae6ed1656538bc1f1c55546039637d7b75dd629950' THEN
    RAISE EXCEPTION 'capture reader684 v2 preimage differs' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.jsonb_build_array(pg_catalog.pg_get_userbyid(p.proowner), p.proacl::TEXT,
       p.prosecdef, p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid=p.prolang
      WHERE p.oid='public.lab_arena_capture_summary_v2(text,text)'::regprocedure)
      IS DISTINCT FROM '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader684 v2 security differs' USING ERRCODE='55000';
  END IF;
END;
$capture_reader_guard684$;
CREATE TEMP TABLE capture_reader684_schema ON COMMIT DROP AS
  SELECT nspacl, pg_catalog.has_schema_privilege('lab_arena_owner','public','CREATE') AS had_create
  FROM pg_catalog.pg_namespace WHERE nspname='public';
CREATE TEMP TABLE capture_reader684_v1 ON COMMIT DROP AS
  SELECT pg_catalog.to_jsonb(p) AS catalog FROM pg_catalog.pg_proc p
  WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure;
CREATE TEMP TABLE capture_reader684_v2 ON COMMIT DROP AS
  SELECT pg_catalog.to_jsonb(p) AS catalog FROM pg_catalog.pg_proc p
  WHERE p.oid='public.lab_arena_capture_summary_v2(text,text)'::regprocedure;
GRANT CREATE ON SCHEMA public TO lab_arena_owner;

CREATE OR REPLACE FUNCTION public.lab_arena_capture_summary_v3(p_round_id TEXT, p_submission_id TEXT)
RETURNS TABLE (
  round_id TEXT, submission_id TEXT, execute_runs INTEGER, icp_positions INTEGER[],
  model_calls INTEGER, usage_calls INTEGER, reasoning_tokens BIGINT,
  first_call_at TIMESTAMPTZ, last_call_at TIMESTAMPTZ,
  input_tokens BIGINT, output_tokens BIGINT, cached_input_tokens BIGINT,
  token_calls INTEGER, models JSONB, reasoning_calls INTEGER, reasoning_capture JSONB)
LANGUAGE plpgsql STABLE SECURITY DEFINER
SET search_path = pg_catalog, public
AS $summary684$
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
  parsed_body BOOLEAN;
  readable BOOLEAN;
  complete_response BOOLEAN;
  capture_class TEXT;
  choice JSON;
  message JSON;
  item JSON;
  part JSON;
  payload_value JSON;
  field TEXT;
  -- Match Python str.strip(), including Unicode whitespace.
  blank_characters CONSTANT TEXT := U&'\0009\000a\000b\000c\000d\001c\001d\001e\001f\0020\0085\00a0\1680\2000\2001\2002\2003\2004\2005\2006\2007\2008\2009\200a\2028\2029\202f\205f\3000';
BEGIN
  round_id := p_round_id;
  submission_id := p_submission_id;
  model_calls := 0;
  usage_calls := 0;
  token_calls := 0;
  reasoning_calls := 0;
  reasoning_capture := '{"complete_readable":0,"partial_readable":0,"tokens_only":0,"no_readable":0,"unparseable":0}'::JSONB;
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
    SELECT reserved.started_at, heads.created_at AS finished_at, l.operation_id,
      selected.terminal -> 'body_b64' AS body
    FROM heads
    JOIN reserved ON reserved.call_identity = heads.call_identity
    JOIN public.lab_arena_ledger AS l ON l.entry_id = heads.entry_id
    JOIN public.lab_arena_ledger AS reservation
      ON reservation.call_identity = heads.call_identity AND reservation.entry_kind = 'reservation'
    LEFT JOIN LATERAL (
      -- Only the explicit sanitized reply for this exact reservation can
      -- survive a billing-only head. Private upstream receipts are not replies.
      SELECT u.entry_doc #> '{call,openrouter_terminal_response}' AS terminal
      FROM public.lab_arena_ledger AS u
      WHERE u.run_id = l.run_id AND u.call_identity = l.call_identity
        AND u.entry_kind = 'uncertain' AND u.entry_id > reservation.entry_id AND u.entry_id <= l.entry_id
        AND u.round_id = l.round_id AND u.submission_id = l.submission_id
        AND u.provider = l.provider AND u.operation_id = l.operation_id
        AND u.miner_hotkey = l.miner_hotkey AND u.stage = l.stage AND u.funding_source = l.funding_source
        AND reservation.run_id = l.run_id AND reservation.round_id = l.round_id
        AND reservation.submission_id = l.submission_id AND reservation.provider = l.provider
        AND reservation.operation_id = l.operation_id
        AND reservation.miner_hotkey = l.miner_hotkey AND reservation.stage = l.stage
        AND reservation.funding_source = l.funding_source
        AND l.round_id = p_round_id AND l.submission_id = p_submission_id
        AND (u.entry_id = l.entry_id OR (
          l.entry_kind = 'settlement'
          AND l.entry_doc -> 'openrouter_delayed_reconciliation' = 'true'::JSONB
          AND NOT l.entry_doc ? 'deepline_delayed_reconciliation'
          AND l.entry_doc -> 'reconciled_uncertainty_entry_id' = to_jsonb(u.entry_id)))
        AND u.entry_doc #>> '{call,reason}' IN ('missing_provider_cost', 'settle_failure')
        AND u.entry_doc #> '{call,call_succeeded}' = 'true'::JSONB
        AND jsonb_typeof(u.entry_doc #> '{call,provider_status}') = 'number'
        AND (u.entry_doc #>> '{call,provider_status}') ~ '^2[0-9][0-9]$'
        AND (u.entry_doc #>> '{call,credential_fingerprint}') ~ '^sha256:[0-9a-f]{64}$'
        AND NOT EXISTS (SELECT 1 FROM unnest(ARRAY[
          'credential_fingerprint', 'openrouter_generation_id', 'openrouter_model', 'request_hash']) AS field(name)
          WHERE reservation.entry_doc ? field.name AND (u.entry_doc -> 'call') ? field.name
            AND reservation.entry_doc -> field.name IS DISTINCT FROM u.entry_doc -> 'call' -> field.name)
        AND (NOT (u.entry_doc -> 'call') ? 'openrouter_generation_id'
          OR (u.entry_doc #>> '{call,openrouter_generation_id}') ~ '^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$')
        AND u.entry_doc #> '{call,openrouter_terminal_response,call_succeeded}' = 'true'::JSONB
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,status}') = 'number'
        AND (u.entry_doc #>> '{call,openrouter_terminal_response,status}') ~ '^2[0-9][0-9]$'
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,headers}') = 'object'
        AND jsonb_typeof(u.entry_doc #> '{call,openrouter_terminal_response,body_b64}') = 'string'
        AND length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') % 4 = 0
        AND (u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') ~ '^[A-Za-z0-9+/]*={0,2}$'
        AND length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}') / 4 * 3
          - (length(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}')
            - length(rtrim(u.entry_doc #>> '{call,openrouter_terminal_response,body_b64}', '='))) <= 16777216
        AND (u.entry_doc #> '{call,openrouter_terminal_response}') - ARRAY[
          'status', 'headers', 'body_b64', 'call_succeeded'] = '{}'::JSONB
        AND NOT (u.entry_doc #> '{call,openrouter_terminal_response}') ?| ARRAY[
          'judgment_cache_source_call_identity', 'judgment_cache_source_run_id']
    ) AS retained ON TRUE
    CROSS JOIN LATERAL (
      SELECT CASE WHEN l.terminal_response -> 'call_succeeded' = 'true'::JSONB
        AND jsonb_typeof(l.terminal_response -> 'status') = 'number'
        AND (l.terminal_response ->> 'status') ~ '^2[0-9][0-9]$'
        THEN l.terminal_response ELSE retained.terminal END AS terminal
    ) AS selected
    WHERE selected.terminal -> 'call_succeeded' = 'true'::JSONB
      AND jsonb_typeof(selected.terminal -> 'status') = 'number'
      AND (selected.terminal ->> 'status') ~ '^2[0-9][0-9]$'
  LOOP
    model_calls := model_calls + 1;
    first_call_at := least(first_call_at, head.started_at);
    last_call_at := greatest(last_call_at, head.finished_at);
    tokens := NULL;
    call_input := NULL;
    call_output := NULL;
    call_cached := NULL;
    model_name := 'unknown';
    parsed_body := FALSE;
    readable := FALSE;
    complete_response := FALSE;
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
          parsed_body := TRUE;
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
    -- Inventory saved response text only. This does not assert reviewed route,
    -- exact request/delivery proof, complete execution, or training eligibility.
    -- Shape checks make malformed provider containers safe to inspect.
    IF parsed_body THEN
      BEGIN
        IF head.operation_id = 'openrouter.chat' THEN
          complete_response := CASE WHEN json_typeof(response_doc -> 'choices') = 'array'
            THEN json_array_length(response_doc -> 'choices') = 1 ELSE FALSE END
            AND response_doc #>> '{choices,0,finish_reason}' IN ('stop', 'tool_calls')
            AND (response_doc ->> 'incomplete_details') IS NULL
            AND (response_doc ->> 'error') IS NULL
            AND ((response_doc ->> 'status') IS NULL OR response_doc ->> 'status' = 'completed');
          FOR choice IN SELECT element FROM json_array_elements(
            CASE WHEN json_typeof(response_doc -> 'choices') = 'array'
                 THEN response_doc -> 'choices' ELSE '[]'::JSON END) AS choices(element)
          LOOP
            IF choice ->> 'error' IS NOT NULL
               OR (choice ->> 'status' IS NOT NULL AND choice ->> 'status' <> 'completed') THEN
              complete_response := FALSE;
            END IF;
            message := choice -> 'message';
            FOREACH field IN ARRAY ARRAY['reasoning', 'reasoning_content'] LOOP
              payload_value := message -> field;
              readable := readable OR (json_typeof(payload_value) = 'string'
                AND btrim(payload_value #>> '{}', blank_characters) <> '');
            END LOOP;
            FOR item IN SELECT element FROM json_array_elements(
              CASE WHEN json_typeof(message -> 'reasoning_details') = 'array'
                   THEN message -> 'reasoning_details' ELSE '[]'::JSON END) AS details(element)
            LOOP
              -- Summary and opaque typed objects never establish readable thinking.
              IF coalesce(item ->> 'type', '') IN
                   ('reasoning.summary', 'summary_text', 'reasoning.encrypted', 'redacted_thinking') THEN
                CONTINUE;
              END IF;
              FOREACH field IN ARRAY ARRAY['text', 'thinking', 'content'] LOOP
                payload_value := item -> field;
                readable := readable OR (json_typeof(payload_value) = 'string'
                  AND btrim(payload_value #>> '{}', blank_characters) <> '');
                FOR part IN SELECT element FROM json_array_elements(
                  CASE WHEN json_typeof(payload_value) = 'array' THEN payload_value ELSE '[]'::JSON END) AS parts(element)
                LOOP
                  readable := readable OR (coalesce(part ->> 'type', '') NOT IN
                    ('reasoning.summary', 'summary_text', 'reasoning.encrypted', 'redacted_thinking')
                    AND json_typeof(part -> 'text') = 'string'
                    AND btrim(part ->> 'text', blank_characters) <> '');
                END LOOP;
              END LOOP;
            END LOOP;
          END LOOP;
        ELSE
          complete_response := response_doc ->> 'status' = 'completed'
            AND (response_doc ->> 'incomplete_details') IS NULL
            AND (response_doc ->> 'error') IS NULL;
          FOR item IN SELECT element FROM json_array_elements(
            CASE WHEN json_typeof(response_doc -> 'output') = 'array'
                 THEN response_doc -> 'output' ELSE '[]'::JSON END) AS outputs(element)
          LOOP
            -- Top-level completion cannot override an incomplete output item.
            IF item ->> 'error' IS NOT NULL
               OR (item ->> 'status' IS NOT NULL AND item ->> 'status' <> 'completed') THEN
              complete_response := FALSE;
            END IF;
            IF item ->> 'type' IS DISTINCT FROM 'reasoning' THEN CONTINUE; END IF;
            FOREACH field IN ARRAY ARRAY['text', 'thinking', 'content'] LOOP
              payload_value := item -> field;
              readable := readable OR (json_typeof(payload_value) = 'string'
                AND btrim(payload_value #>> '{}', blank_characters) <> '');
              FOR part IN SELECT element FROM json_array_elements(
                CASE WHEN json_typeof(payload_value) = 'array' THEN payload_value ELSE '[]'::JSON END) AS parts(element)
              LOOP
                readable := readable OR (coalesce(part ->> 'type', '') NOT IN
                  ('reasoning.summary', 'summary_text', 'reasoning.encrypted', 'redacted_thinking')
                  AND json_typeof(part -> 'text') = 'string'
                  AND btrim(part ->> 'text', blank_characters) <> '');
              END LOOP;
            END LOOP;
          END LOOP;
        END IF;
      EXCEPTION WHEN character_not_in_repertoire OR invalid_text_representation
        OR untranslatable_character OR statement_too_complex THEN
        parsed_body := FALSE;
        readable := FALSE;
      END;
    END IF;
    capture_class := CASE WHEN NOT parsed_body THEN 'unparseable'
      WHEN readable AND complete_response THEN 'complete_readable'
      WHEN readable THEN 'partial_readable'
      WHEN tokens > 0 THEN 'tokens_only' ELSE 'no_readable' END;
    reasoning_capture := jsonb_set(reasoning_capture, ARRAY[capture_class],
      to_jsonb((reasoning_capture ->> capture_class)::INTEGER + 1));
    IF tokens >= 0 THEN
      usage_calls := usage_calls + 1;
      token_total := token_total + tokens;
    END IF;
    IF tokens > 0 THEN
      reasoning_calls := reasoning_calls + 1;
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
$summary684$;
ALTER FUNCTION public.lab_arena_capture_summary_v3(TEXT, TEXT) OWNER TO lab_arena_owner;

DO $capture_reader_acl684$
DECLARE
  target CONSTANT REGPROCEDURE := 'public.lab_arena_capture_summary_v3(text,text)'::regprocedure;
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
  REVOKE ALL ON FUNCTION public.lab_arena_capture_summary_v3(TEXT, TEXT)
    FROM PUBLIC, anon, authenticated, service_role;
  GRANT EXECUTE ON FUNCTION public.lab_arena_capture_summary_v3(TEXT, TEXT) TO leadpoet_capture_reader;
  IF (SELECT jsonb_build_array(pg_get_userbyid(p.proowner), p.proacl::TEXT, p.prosecdef,
        p.provolatile, p.proconfig, l.lanname)
      FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_language l ON l.oid = p.prolang
      WHERE p.oid = target) IS DISTINCT FROM
     '["lab_arena_owner","{lab_arena_owner=X/lab_arena_owner,leadpoet_capture_reader=X/lab_arena_owner}",true,"s",["search_path=pg_catalog, public"],"plpgsql"]'::JSONB THEN
    RAISE EXCEPTION 'capture reader684 function security differs' USING ERRCODE='55000';
  END IF;
  -- The reader reaches Arena data only through the three aggregate functions.
  IF EXISTS (SELECT 1 FROM pg_catalog.pg_class c, aclexplode(c.relacl) AS a
             WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_attribute att, aclexplode(att.attacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole)
     OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc p, aclexplode(p.proacl) AS a
                WHERE a.grantee = 'leadpoet_capture_reader'::regrole AND p.oid NOT IN (target, 'public.lab_arena_capture_summary_v1(text,text)'::regprocedure,
                  'public.lab_arena_capture_summary_v2(text,text)'::regprocedure)) THEN
    RAISE EXCEPTION 'capture reader684 holds other privileges' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p
      WHERE p.oid='public.lab_arena_capture_summary_v1(text,text)'::regprocedure)
      IS DISTINCT FROM (SELECT catalog FROM pg_temp.capture_reader684_v1) THEN
    RAISE EXCEPTION 'capture reader684 v1 catalog changed' USING ERRCODE='55000';
  END IF;
  IF (SELECT pg_catalog.to_jsonb(p) FROM pg_catalog.pg_proc p
      WHERE p.oid='public.lab_arena_capture_summary_v2(text,text)'::regprocedure)
      IS DISTINCT FROM (SELECT catalog FROM pg_temp.capture_reader684_v2) THEN
    RAISE EXCEPTION 'capture reader684 v2 catalog changed' USING ERRCODE='55000';
  END IF;
  IF NOT (SELECT had_create FROM pg_temp.capture_reader684_schema) THEN
    REVOKE CREATE ON SCHEMA public FROM lab_arena_owner;
  END IF;
  IF (SELECT nspacl FROM pg_catalog.pg_namespace WHERE nspname = 'public')
     IS DISTINCT FROM (SELECT nspacl FROM pg_temp.capture_reader684_schema) THEN
    RAISE EXCEPTION 'capture reader684 schema ACL changed' USING ERRCODE='55000';
  END IF;
END;
$capture_reader_acl684$;
NOTIFY pgrst, 'reload schema';
COMMIT;

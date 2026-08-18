-- Dedicated production-parity snapshot reader and one fixed password setter.
--
-- The exact migration contains no credential.  The authorized Keychain-backed
-- overnight-rebenchmark skill helper applies these exact commit/hash-bound
-- bytes.  The repository bootstrap then verifies this live
-- contract and calls only the fixed postgres-only function through the
-- Supabase Management API with a hidden 64-character lowercase hexadecimal
-- password binding.

BEGIN;

DO $leadpoet_parity_reader$
DECLARE
  membership RECORD;
BEGIN
  PERFORM pg_advisory_xact_lock(
    hashtextextended('leadpoet.production-parity-reader.v1', 0)
  );

  IF NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
  ) THEN
    EXECUTE 'CREATE ROLE leadpoet_parity_reader NOLOGIN';
  END IF;

  IF EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
      AND rolsuper
  ) THEN
    RAISE EXCEPTION 'production parity reader role is unexpectedly superuser';
  END IF;

  FOR membership IN
    SELECT granted.rolname
    FROM pg_catalog.pg_auth_members member
    JOIN pg_catalog.pg_roles granted ON granted.oid = member.roleid
    JOIN pg_catalog.pg_roles recipient ON recipient.oid = member.member
    WHERE recipient.rolname = 'leadpoet_parity_reader'
  LOOP
    EXECUTE format(
      'REVOKE %I FROM leadpoet_parity_reader',
      membership.rolname
    );
  END LOOP;

  -- Hosted Supabase delegates BYPASSRLS through supautils but rejects any
  -- superuser-option syntax in ALTER ROLE, including a redundant false value.
  -- The exact precheck above and the default on CREATE preserve fail-closed
  -- non-superuser identity without crossing that reserved-option boundary.
  ALTER ROLE leadpoet_parity_reader WITH
    NOCREATEDB
    NOCREATEROLE
    NOINHERIT
    NOREPLICATION
    BYPASSRLS
    CONNECTION LIMIT 2;
  IF EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
      AND rolsuper
  ) THEN
    RAISE EXCEPTION 'production parity reader role is unexpectedly superuser';
  END IF;
  ALTER ROLE leadpoet_parity_reader
    SET default_transaction_read_only = on;
  ALTER ROLE leadpoet_parity_reader
    SET idle_in_transaction_session_timeout = '5min';
END
$leadpoet_parity_reader$;

GRANT CONNECT ON DATABASE postgres TO leadpoet_parity_reader;
GRANT USAGE ON SCHEMA public TO leadpoet_parity_reader;

REVOKE CREATE ON SCHEMA public FROM leadpoet_parity_reader;
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
  ON ALL TABLES IN SCHEMA public
  FROM leadpoet_parity_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO leadpoet_parity_reader;

REVOKE USAGE, UPDATE
  ON ALL SEQUENCES IN SCHEMA public
  FROM leadpoet_parity_reader;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO leadpoet_parity_reader;

ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER
  ON TABLES FROM leadpoet_parity_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT SELECT ON TABLES TO leadpoet_parity_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  REVOKE USAGE, UPDATE ON SEQUENCES FROM leadpoet_parity_reader;
ALTER DEFAULT PRIVILEGES FOR ROLE postgres IN SCHEMA public
  GRANT SELECT ON SEQUENCES TO leadpoet_parity_reader;

CREATE OR REPLACE FUNCTION public.leadpoet_set_production_parity_reader_password_v1(
  p_password TEXT
)
RETURNS JSONB
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog
AS $leadpoet_parity_reader_password$
DECLARE
  role_row pg_catalog.pg_roles%ROWTYPE;
  role_contract_valid BOOLEAN;
BEGIN
  IF session_user <> 'postgres' OR current_user <> 'postgres' THEN
    RAISE EXCEPTION 'production parity password setter requires postgres';
  END IF;
  PERFORM pg_advisory_xact_lock(
    hashtextextended('leadpoet.production-parity-reader.v1', 0)
  );
  IF NOT EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader'
  ) THEN
    RAISE EXCEPTION 'production parity reader role is unavailable';
  END IF;

  -- A failed retry must never leave an unsafe pre-existing LOGIN enabled.
  ALTER ROLE leadpoet_parity_reader NOLOGIN;
  BEGIN
    IF p_password IS NULL OR p_password !~ '^[0-9a-f]{64}$' THEN
      RAISE EXCEPTION 'production parity reader password binding is invalid';
    END IF;
    SELECT *
    INTO STRICT role_row
    FROM pg_catalog.pg_roles
    WHERE rolname = 'leadpoet_parity_reader';
    SELECT (
      NOT role_row.rolcanlogin
      AND NOT role_row.rolsuper
      AND role_row.rolbypassrls
      AND NOT role_row.rolcreatedb
      AND NOT role_row.rolcreaterole
      AND NOT role_row.rolinherit
      AND NOT role_row.rolreplication
      AND role_row.rolconnlimit = 2
      AND COALESCE(
        'default_transaction_read_only=on' = ANY(role_row.rolconfig),
        false
      )
      AND NOT pg_catalog.has_schema_privilege(
        'leadpoet_parity_reader', 'public', 'CREATE'
      )
      AND NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_auth_members member
        JOIN pg_catalog.pg_roles recipient ON recipient.oid = member.member
        WHERE recipient.rolname = 'leadpoet_parity_reader'
      )
      AND NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = 'public'
          AND relation.relkind IN ('r', 'p')
          AND pg_catalog.has_table_privilege(
            'leadpoet_parity_reader',
            relation.oid,
            'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
          )
      )
      AND NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class sequence
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = sequence.relnamespace
        WHERE namespace.nspname = 'public'
          AND CASE WHEN sequence.relkind = 'S' THEN
            pg_catalog.has_sequence_privilege(
              'leadpoet_parity_reader', sequence.oid, 'USAGE,UPDATE'
            )
          ELSE false END
      )
    ) INTO role_contract_valid;
    IF NOT role_contract_valid THEN
      RAISE EXCEPTION 'production parity reader role contract differs';
    END IF;

    EXECUTE format(
      'ALTER ROLE leadpoet_parity_reader WITH LOGIN PASSWORD %L',
      p_password
    );
    RETURN jsonb_build_object(
      'status', 'bound',
      'reader_role', 'leadpoet_parity_reader',
      'login_enabled', true,
      'password_format', 'hex64'
    );
  EXCEPTION WHEN OTHERS THEN
    RETURN jsonb_build_object(
      'status', 'disabled',
      'reader_role', 'leadpoet_parity_reader',
      'login_enabled', false,
      'password_format', 'hex64'
    );
  END;
END
$leadpoet_parity_reader_password$;

ALTER FUNCTION public.leadpoet_set_production_parity_reader_password_v1(TEXT)
  OWNER TO postgres;
REVOKE ALL
  ON FUNCTION public.leadpoet_set_production_parity_reader_password_v1(TEXT)
  FROM PUBLIC;

DO $leadpoet_parity_reader_password_acl$
DECLARE
  role_name TEXT;
BEGIN
  FOREACH role_name IN ARRAY ARRAY['anon', 'authenticated', 'service_role']
  LOOP
    IF EXISTS (
      SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = role_name
    ) THEN
      EXECUTE format(
        'REVOKE ALL ON FUNCTION '
        'public.leadpoet_set_production_parity_reader_password_v1(TEXT) FROM %I',
        role_name
      );
    END IF;
  END LOOP;
END
$leadpoet_parity_reader_password_acl$;

COMMENT ON FUNCTION public.leadpoet_set_production_parity_reader_password_v1(TEXT)
  IS 'Postgres-only crash-safe 64-hex password binder for the production-parity reader.';

CREATE OR REPLACE FUNCTION public.leadpoet_production_parity_reader_contract_v1()
RETURNS JSONB
LANGUAGE SQL
STABLE
SECURITY DEFINER
SET search_path = pg_catalog
AS $leadpoet_parity_reader_contract$
  SELECT jsonb_build_object(
    'schema_version', 'leadpoet.production-parity-reader-contract.v1',
    'database_name', current_database(),
    'reader_role', role.rolname,
    'login_enabled', role.rolcanlogin,
    'superuser', role.rolsuper,
    'bypass_rls', role.rolbypassrls,
    'createdb', role.rolcreatedb,
    'createrole', role.rolcreaterole,
    'inherit', role.rolinherit,
    'replication', role.rolreplication,
    'connection_limit', role.rolconnlimit,
    'default_read_only', COALESCE(
      'default_transaction_read_only=on' = ANY(role.rolconfig), false
    ),
    'membership_count', (
      SELECT count(*)
      FROM pg_catalog.pg_auth_members member
      WHERE member.member = role.oid
    ),
    'schema_create_capable', pg_catalog.has_schema_privilege(
      role.rolname, 'public', 'CREATE'
    ),
    'table_write_capable', EXISTS (
      SELECT 1
      FROM pg_catalog.pg_class relation
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = relation.relnamespace
      WHERE namespace.nspname = 'public'
        AND relation.relkind IN ('r', 'p')
        AND pg_catalog.has_table_privilege(
          role.rolname,
          relation.oid,
          'INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'
        )
    ),
    'sequence_write_capable', EXISTS (
      SELECT 1
      FROM pg_catalog.pg_class sequence
      JOIN pg_catalog.pg_namespace namespace
        ON namespace.oid = sequence.relnamespace
      WHERE namespace.nspname = 'public'
        AND CASE WHEN sequence.relkind = 'S' THEN
          pg_catalog.has_sequence_privilege(
            role.rolname, sequence.oid, 'USAGE,UPDATE'
          )
        ELSE false END
    )
  )
  FROM pg_catalog.pg_roles role
  WHERE role.rolname = 'leadpoet_parity_reader'
$leadpoet_parity_reader_contract$;

ALTER FUNCTION public.leadpoet_production_parity_reader_contract_v1()
  OWNER TO postgres;
REVOKE ALL
  ON FUNCTION public.leadpoet_production_parity_reader_contract_v1()
  FROM PUBLIC;

DO $leadpoet_parity_reader_contract_acl$
BEGIN
  IF EXISTS (
    SELECT 1 FROM pg_catalog.pg_roles WHERE rolname = 'service_role'
  ) THEN
    GRANT EXECUTE
      ON FUNCTION public.leadpoet_production_parity_reader_contract_v1()
      TO service_role;
  END IF;
END
$leadpoet_parity_reader_contract_acl$;

COMMENT ON FUNCTION public.leadpoet_production_parity_reader_contract_v1()
  IS 'Non-secret production-parity reader contract for exact migration preflight.';

NOTIFY pgrst, 'reload schema';
COMMIT;

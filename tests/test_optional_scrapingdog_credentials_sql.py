import base64
import json

import pytest

from tests.lab_arena.lab_arena_pg_harness import (
    DEFAULT_MIGRATIONS,
    LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION,
    LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION,
    database_with_lab_arena_migration,
)
from tests.postgres_migration_harness import SCRIPTS


HOTKEY = "5" * 48
ROUND_ID = "arena-2099-01-01"


def _ciphertext(value):
    return base64.b64encode(value.encode()).decode()


def _insert_submission(cursor, submission_id, hotkey=HOTKEY):
    cursor.execute(
        """
        INSERT INTO public.lab_arena_submissions (
          submission_id, round_id, miner_hotkey, status, is_king,
          source_ref, source_size_bytes, consent, submission_doc
        ) VALUES (%s, %s, %s, 'uploading', FALSE, %s, 100,
                  '{"public_rerun":true}'::jsonb, '{}'::jsonb)
        """,
        (
            submission_id,
            ROUND_ID,
            hotkey,
            "arena/%s/sources/%s.tar.gz" % (ROUND_ID, submission_id),
        ),
    )


def _accept(cursor, submission_id, providers, hotkey=HOTKEY):
    cursor.execute(
        "SELECT public.lab_arena_accept_submission_with_credentials(%s, %s, %s, %s::jsonb)",
        (ROUND_ID, submission_id, hotkey, json.dumps(providers)),
    )
    return cursor.fetchone()[0]


def test_optional_scrapingdog_migration_upgrades_reapplies_and_admits_two_or_three_atomically():
    pre_weight_migrations = DEFAULT_MIGRATIONS[
        : DEFAULT_MIGRATIONS.index(LAB_ARENA_ACCEPTED_WEIGHT_STATE_MIGRATION)
    ]
    database = database_with_lab_arena_migration(pre_weight_migrations)
    try:
        psycopg2, dsn = next(database)
        with psycopg2.connect(**dsn) as connection:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO public.lab_arena_rounds (
                      round_id, status, configuration_doc, rewards_enabled
                    ) VALUES (
                      %s, 'open',
                      '{"mode":"shadow","rewards_enabled":false,"max_challengers":16,
                        "schedule":{"submission_open":"2000-01-01T00:00:00Z",
                        "submission_cutoff":"2100-01-01T00:00:00Z"}}'::jsonb,
                      FALSE
                    )
                    """,
                    (ROUND_ID,),
                )
                _insert_submission(cursor, "two-keys")
                required = {
                    "openrouter": _ciphertext("openrouter-cipher"),
                    "deepline": _ciphertext("deepline-cipher"),
                }
                assert _accept(cursor, "two-keys", required)["status"] == "ok"
                cursor.execute(
                    "SELECT provider FROM public.lab_arena_submission_credentials "
                    "WHERE submission_id = 'two-keys' ORDER BY provider"
                )
                assert [row[0] for row in cursor.fetchall()] == ["deepline", "openrouter"]
                connection.commit()

                cursor.execute(
                    "SELECT provider, encode(ciphertext, 'base64') "
                    "FROM public.lab_arena_submission_credentials "
                    "WHERE submission_id = 'two-keys' ORDER BY provider"
                )
                before_upgrade = cursor.fetchall()
                cursor.execute(
                    "SELECT to_regprocedure("
                    "'public.lab_arena_weight_state_schema_v1()'), "
                    "to_regprocedure("
                    "'public.lab_arena_incentive_retirement_schema_v1()')"
                )
                assert cursor.fetchone() == (None, None)
                cursor.execute(
                    (SCRIPTS / LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION).read_text()
                )
                cursor.execute(
                    "SELECT provider, encode(ciphertext, 'base64') "
                    "FROM public.lab_arena_submission_credentials "
                    "WHERE submission_id = 'two-keys' ORDER BY provider"
                )
                assert cursor.fetchall() == before_upgrade
                connection.commit()

                with pytest.raises(psycopg2.Error, match="lab_arena_submission_credentials_immutable"):
                    _accept(
                        cursor,
                        "two-keys",
                        {**required, "scrapingdog": _ciphertext("scrapingdog-cipher")},
                    )
                connection.rollback()
                cursor.execute(
                    "SELECT COUNT(*) FROM public.lab_arena_submission_credentials "
                    "WHERE submission_id = 'two-keys'"
                )
                assert cursor.fetchone()[0] == 2

                three_key_hotkey = "6" * 48
                _insert_submission(cursor, "three-keys", three_key_hotkey)
                all_providers = {
                    **required,
                    "scrapingdog": _ciphertext("scrapingdog-cipher"),
                }
                assert _accept(cursor, "three-keys", all_providers, three_key_hotkey)["status"] == "ok"
                cursor.execute(
                    "SELECT public.lab_arena_get_submission_credential(%s, %s, 'scrapingdog')",
                    ("three-keys", three_key_hotkey),
                )
                returned = cursor.fetchone()[0]
                assert returned["status"] == "available"
                assert returned["provider"] == "scrapingdog"
                assert base64.b64decode(returned["ciphertext_b64"]) == b"scrapingdog-cipher"
                cursor.execute(
                    "SELECT public.lab_arena_get_submission_credential(%s, %s, 'scrapingdog')",
                    ("three-keys", "8" * 48),
                )
                assert cursor.fetchone()[0] == {"status": "missing"}
                cursor.execute(
                    "SELECT rolname, "
                    "has_table_privilege(rolname, 'public.lab_arena_submission_credentials', 'SELECT'), "
                    "has_function_privilege(rolname, "
                    "'public.lab_arena_get_submission_credential(text,text,text)', 'EXECUTE') "
                    "FROM pg_roles WHERE rolname IN ('anon', 'authenticated', 'service_role') "
                    "ORDER BY rolname"
                )
                assert cursor.fetchall() == [
                    ("anon", False, False),
                    ("authenticated", False, False),
                    ("service_role", False, False),
                ]
                connection.commit()

                malformed_hotkey = "7" * 48
                _insert_submission(cursor, "malformed", malformed_hotkey)
                connection.commit()
                with pytest.raises(psycopg2.Error, match="lab_arena_credentials_invalid"):
                    _accept(
                        cursor,
                        "malformed",
                        {**required, "unknown": "YWJjZA=="},
                        malformed_hotkey,
                    )
                connection.rollback()
                cursor.execute(
                    "SELECT status FROM public.lab_arena_submissions WHERE submission_id = 'malformed'"
                )
                assert cursor.fetchone()[0] == "uploading"
                cursor.execute(
                    "SELECT COUNT(*) FROM public.lab_arena_submission_credentials "
                    "WHERE submission_id = 'malformed'"
                )
                assert cursor.fetchone()[0] == 0

                cursor.execute(
                    (SCRIPTS / LAB_ARENA_OPTIONAL_SCRAPINGDOG_CREDENTIAL_MIGRATION).read_text()
                )
                cursor.execute(
                    "SELECT COUNT(*) FROM public.lab_arena_submission_credentials"
                )
                assert cursor.fetchone()[0] == 5
    finally:
        database.close()
